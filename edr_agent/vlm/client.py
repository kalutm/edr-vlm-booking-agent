"""
vlm/client.py — Gemini API adapter with dynamic 429 rate limit retry & model fallback logic.

Isolates all Gemini-specific code behind a clean interface.
Swap this file to change VLM provider without touching the rest of the system.
"""

from __future__ import annotations

import json
import logging
import re
import time
import warnings
from typing import Any, Type, TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from pydantic import BaseModel

from edr_agent.config import settings

# Suppress the Gemini SDK's informational AFC warning (not an error)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]


def extract_retry_delay_seconds(error: Exception) -> float:
    """
    Extract retryDelay value from a Gemini API 429 / RESOURCE_EXHAUSTED error.
    Returns extracted delay in seconds, or default fallback of 60.0s.
    """
    err_str = str(error)

    # Pattern 1: "Please retry in 52.403317006s."
    match_msg = re.search(r'retry in (\d+(?:\.\d+)?)s', err_str, re.IGNORECASE)
    if match_msg:
        return float(match_msg.group(1))

    # Pattern 2: 'retryDelay': '52s' or "retryDelay": "52s"
    match_delay = re.search(r'[\'\"]retryDelay[\'\"]:\s*[\'\"](\d+(?:\.\d+)?)(s?)[\'\"]', err_str)
    if match_delay:
        return float(match_delay.group(1))

    # Pattern 3: retryDelay: 52
    match_sec = re.search(r'retryDelay["\s:]+(\d+(?:\.\d+)?)', err_str, re.IGNORECASE)
    if match_sec:
        return float(match_sec.group(1))

    # Pattern 4: Check response_json if available on ClientError
    if isinstance(error, (ClientError, APIError)):
        try:
            resp = getattr(error, "response_json", None)
            if isinstance(resp, dict):
                error_obj = resp.get("error", {})
                details = error_obj.get("details", [])
                for detail in details:
                    if isinstance(detail, dict) and "retryDelay" in detail:
                        raw_delay = str(detail["retryDelay"]).rstrip("s")
                        return float(raw_delay)
        except Exception:
            pass

    return 60.0  # Default fallback pause duration if unspecified


class VLMClient:
    """
    Thin wrapper around the Gemini API with dynamic 429 rate-limit handling & model fallbacks.
    All calls use structured output (response_schema) for predictable parsing.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        logger.info(f"VLMClient initialized — primary model: {self._model}")

    def _generate_content_with_retry(
        self,
        contents: Any,
        config: types.GenerateContentConfig,
        max_retries: int = 5,
        override_model: str | None = None,
    ) -> types.GenerateContentResponse:
        """
        Execute API call with dynamic 429 rate-limit handling and model fallbacks.
        Parses retryDelay from RESOURCE_EXHAUSTED errors and sleeps exact duration + 2s buffer.
        Handles 5xx errors with exponential backoff.
        """
        primary_model = override_model or self._model
        candidate_models = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

        for model_name in candidate_models:
            for attempt in range(1, max_retries + 1):
                try:
                    return self._client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                except Exception as e:
                    is_rate_limit = False
                    is_5xx_error = False
                    status_code = None
                    err_str = str(e).lower()

                    if isinstance(e, (ClientError, APIError)):
                        code = getattr(e, "code", None) or getattr(e, "status_code", None)
                        if code == 429 or "resource_exhausted" in err_str or "429" in err_str:
                            is_rate_limit = True
                        elif code in (500, 502, 503, 504):
                            is_5xx_error = True
                            status_code = code
                    
                    if not is_rate_limit and not is_5xx_error:
                        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                            is_rate_limit = True
                        elif "500" in err_str:
                            is_5xx_error = True
                            status_code = 500
                        elif "502" in err_str:
                            is_5xx_error = True
                            status_code = 502
                        elif "503" in err_str:
                            is_5xx_error = True
                            status_code = 503
                        elif "504" in err_str:
                            is_5xx_error = True
                            status_code = 504

                    if is_rate_limit:
                        retry_delay = extract_retry_delay_seconds(e)
                        wait_seconds = retry_delay + 2.0  # Exact duration + 2-second safety buffer
                        logger.warning(
                            f"[GEMINI API 429] Rate limit hit for model {model_name} (attempt {attempt}/{max_retries}). "
                            f"Extracted retryDelay: {retry_delay:.1f}s. Pausing for {wait_seconds:.1f}s (incl. 2s buffer)..."
                        )
                        time.sleep(wait_seconds)
                    elif is_5xx_error:
                        wait_seconds = 5.0 * (2 ** (attempt - 1))
                        status_msg_map = {
                            500: "Internal Server Error",
                            502: "Bad Gateway",
                            503: "Service Unavailable",
                            504: "Gateway Timeout"
                        }
                        status_msg = status_msg_map.get(status_code, "Server Error")
                        logger.warning(
                            f"[VLM API] {status_code} {status_msg} hit for model {model_name} (attempt {attempt}/{max_retries}). "
                            f"Backing off for {wait_seconds} seconds..."
                        )
                        time.sleep(wait_seconds)
                    else:
                        logger.error(f"[GEMINI API ERROR] Call failed for {model_name}: {e}")
                        raise e

            logger.warning(f"[GEMINI API] Max retries reached for model {model_name}, trying fallback model...")

        raise RuntimeError("Gemini API call failed on all available models due to rate limits.")

    def perceive(
        self,
        screenshot_bytes: bytes,
        prompt: str,
        schema: Type[T],
    ) -> T:
        """
        Send a screenshot + prompt to the VLM and get a structured response.
        """
        image_part = types.Part.from_bytes(
            data=screenshot_bytes,
            mime_type="image/png",
        )

        response = self._generate_content_with_retry(
            contents=[
                types.Content(
                    role="user",
                    parts=[image_part, types.Part.from_text(text=prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,  # Low temp for consistent structured output
                max_output_tokens=2048,
            ),
            override_model="gemini-3.6-flash",
        )

        raw_text = response.text
        logger.debug(f"VLM raw response: {raw_text[:300]}...")
        return schema.model_validate_json(raw_text)

    def text_only(self, prompt: str) -> str:
        """Simple text-only call for non-visual tasks."""
        response = self._generate_content_with_retry(
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return response.text

"""
vlm/perception.py — Visual Perception Module

Input:  browser screenshot (bytes) + task context
Output: PerceptionResult (structured)

This module is responsible for understanding what is currently visible in the
browser. It does NOT make decisions — it only reports what it sees.
"""

from __future__ import annotations

import logging
from datetime import date

from edr_agent.vlm.client import VLMClient
from edr_agent.vlm.schemas import PerceptionResult

logger = logging.getLogger(__name__)


PERCEPTION_PROMPT_TEMPLATE = """You are the Visual Perception Module of an AI booking agent for the Ethiopian-Djibouti Railway (EDR) website.

## Your Task
Analyze the provided browser screenshot of the SEARCH_RESULTS page and extract structured availability data.

## Booking Context
- Route: {origin} → {destination}
- Target Date: {target_date}
- Preferred Seat Type: {preferred_seat}

## Output Fields

1. **current_page**: Identify which page you are on:
   - HOME: The main booking search form (has "Departure station" / "Destination station" inputs)
   - SEARCH_RESULTS: A list of available train schedules shown after searching
   - ERROR_PAGE: An error or "no results" page
   - UNKNOWN: Cannot determine

2. **schedule_state** (on SEARCH_RESULTS only):
   - SCHEDULE_FOUND: At least one schedule card has a "Select" button available
   - SCHEDULE_FULL: All schedule cards show "Fully booked" — no "Select" button available
   - NO_RESULTS: "No trains found" or no schedule cards present
   - UNKNOWN: Cannot determine

3. **available_seats**: Read the availability text on each schedule card verbatim.
   EDR seat class labels (use EXACTLY as shown):
   - "Regular Seat" — e.g. "Regular Seat 744 left" or "Regular Seat Full"
   - "Economy Bed Lower"
   - "Economy Bed Middle"
   - "Economy Bed Upper"
   - "VIP Bed Lower"
   - "VIP Bed Upper"
   Set `available=true` if count > 0, `available=false` if "Full" or count is 0.
   Include the integer count if visible.

4. **preferred_seat_available**: Is "{preferred_seat}" specifically available? Set null if unknown.

5. **error_message**: Quote any error or warning text visible on screen.

6. **confidence**: Your confidence in this analysis (0.0–1.0).

7. **raw_description**: 1–2 sentences describing what you see.

Be precise. Do not guess. Use UNKNOWN if anything is unclear.
"""


class PerceptionModule:
    """
    Module 1: Visual Perception

    Takes a screenshot and returns a structured understanding of the current
    page state. Only invoked on HOME and SEARCH_RESULTS pages — all other
    navigation is handled deterministically by Playwright.
    """

    def __init__(self, vlm_client: VLMClient) -> None:
        self._client = vlm_client

    def perceive(
        self,
        screenshot_bytes: bytes,
        origin: str,
        destination: str,
        target_date: date,
        preferred_seat: str,
    ) -> PerceptionResult:
        """
        Analyze a SEARCH_RESULTS screenshot and return structured availability data.

        Args:
            screenshot_bytes: PNG screenshot from the browser
            origin: Departure station name
            destination: Arrival station name
            target_date: The date we're trying to book
            preferred_seat: User's preferred seat type name

        Returns:
            PerceptionResult with schedule state and seat availability
        """
        prompt = PERCEPTION_PROMPT_TEMPLATE.format(
            origin=origin,
            destination=destination,
            target_date=target_date.isoformat(),
            preferred_seat=preferred_seat,
        )

        logger.info("[PERCEPTION] Analyzing SEARCH_RESULTS screenshot")
        result = self._client.perceive(screenshot_bytes, prompt, PerceptionResult)
        logger.info(
            f"[PERCEPTION] Result: page={result.current_page.value}, "
            f"schedule={result.schedule_state.value}, "
            f"seats={len(result.available_seats)}, "
            f"confidence={result.confidence:.2f}"
        )
        return result

"""
browser/actions.py — Validated browser action executor.

Receives a PredictedAction and executes it safely via Playwright.
All execution logic is here — the VLM never directly controls the browser.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from edr_agent.vlm.schemas import ActionType, PredictedAction

logger = logging.getLogger(__name__)

MAX_ACTION_TIMEOUT = 10000  # 10 seconds per action


class ActionExecutor:
    """
    Executes validated PredictedActions against the Playwright page.

    Safety guarantees:
    - HUMAN_HANDOFF and STOP never touch the browser
    - All selectors tried with fallback
    - Errors are caught and returned, not raised
    """

    def __init__(self, page: Page) -> None:
        self._page = page

    async def execute(self, action: PredictedAction) -> tuple[bool, str]:
        """
        Execute a predicted action.

        Returns:
            (success: bool, message: str)
        """
        atype = action.action_type
        logger.info(f"[EXECUTOR] Executing: {atype.value} | selector={action.target_selector} | value={action.value}")

        try:
            if atype == ActionType.CLICK:
                return await self._click(action)
            elif atype == ActionType.TYPE:
                return await self._type(action)
            elif atype == ActionType.SELECT:
                return await self._select(action)
            elif atype == ActionType.SCROLL:
                return await self._scroll(action)
            elif atype == ActionType.WAIT:
                return await self._wait(action)
            elif atype == ActionType.NAVIGATE:
                return await self._navigate(action)
            elif atype == ActionType.HUMAN_HANDOFF:
                logger.info("[EXECUTOR] HUMAN_HANDOFF — no browser action taken")
                return True, "Human handoff requested — control passed to user"
            elif atype == ActionType.STOP:
                logger.info("[EXECUTOR] STOP — no browser action taken")
                return True, "Agent stopping"
            else:
                return False, f"Unknown action type: {atype}"

        except PlaywrightTimeoutError as e:
            msg = f"Timeout executing {atype.value}: {e}"
            logger.warning(f"[EXECUTOR] {msg}")
            return False, msg
        except Exception as e:
            msg = f"Error executing {atype.value}: {type(e).__name__}: {e}"
            logger.error(f"[EXECUTOR] {msg}")
            return False, msg

    async def _click(self, action: PredictedAction) -> tuple[bool, str]:
        """Click an element by selector or coordinates."""
        # Try primary selector
        if action.target_selector:
            try:
                element = self._page.locator(action.target_selector).first
                await element.click(timeout=MAX_ACTION_TIMEOUT)
                await asyncio.sleep(0.8)
                return True, f"Clicked: {action.target_selector}"
            except Exception as e:
                logger.warning(f"[EXECUTOR] Primary selector failed: {e}")

        # Try fallback selector
        if action.fallback_selector:
            try:
                element = self._page.locator(action.fallback_selector).first
                await element.click(timeout=MAX_ACTION_TIMEOUT)
                await asyncio.sleep(0.8)
                return True, f"Clicked (fallback): {action.fallback_selector}"
            except Exception as e:
                logger.warning(f"[EXECUTOR] Fallback selector failed: {e}")

        # Try coordinates
        if action.target_coordinates and len(action.target_coordinates) == 2:
            x, y = action.target_coordinates
            await self._page.mouse.click(x, y)
            await asyncio.sleep(0.8)
            return True, f"Clicked at coordinates ({x}, {y})"

        return False, "Click failed: no valid selector or coordinates"

    async def _type(self, action: PredictedAction) -> tuple[bool, str]:
        """Type text into an input field."""
        selector = action.target_selector or action.fallback_selector
        if not selector:
            return False, "TYPE action has no selector"

        # Guard: manual text entry for dates is prohibited on EDR site
        sel_lower = selector.lower()
        val_lower = action.value.lower()
        if "date" in sel_lower or "calendar" in sel_lower:
            return False, f"TYPE rejected on date field '{selector}': EDR site requires clicking DATE button to invoke calendar card"

        try:
            element = self._page.locator(selector).first
            await element.click(timeout=MAX_ACTION_TIMEOUT)
            await asyncio.sleep(0.3)
            await element.fill(action.value)
            await asyncio.sleep(0.5)
            return True, f"Typed '{action.value}' into {selector}"
        except Exception as e:
            return False, f"Type failed: {e}"

    async def _select(self, action: PredictedAction) -> tuple[bool, str]:
        """
        Select from a dropdown or autocomplete.
        Tries <select> first, then looks for list items matching the value.
        """
        if not action.value:
            return False, "SELECT action has no value"

        selector = action.target_selector or "select"

        # Try standard select element
        try:
            await self._page.select_option(selector, label=action.value, timeout=5000)
            await asyncio.sleep(0.5)
            return True, f"Selected '{action.value}' from {selector}"
        except Exception:
            pass

        # Try clicking a list item that matches the text (autocomplete dropdowns)
        try:
            option_selectors = [
                f"[role='option']:has-text('{action.value}')",
                f"li:has-text('{action.value}')",
                f"[data-value='{action.value}']",
                f"button:has-text('{action.value}')",
            ]
            for opt_sel in option_selectors:
                try:
                    await self._page.click(opt_sel, timeout=3000)
                    await asyncio.sleep(0.5)
                    return True, f"Selected '{action.value}' via option click"
                except Exception:
                    continue
        except Exception:
            pass

        return False, f"Select failed for value '{action.value}'"

    async def _scroll(self, action: PredictedAction) -> tuple[bool, str]:
        """Scroll the page."""
        direction = (action.value or "down").lower()
        delta_y = 500 if direction == "down" else -500
        await self._page.mouse.wheel(0, delta_y)
        await asyncio.sleep(0.5)
        return True, f"Scrolled {direction}"

    async def _wait(self, action: PredictedAction) -> tuple[bool, str]:
        """Wait for a number of seconds."""
        try:
            seconds = float(action.value or "2")
            seconds = min(seconds, 15.0)  # Cap at 15 seconds
        except ValueError:
            seconds = 2.0
        await asyncio.sleep(seconds)
        return True, f"Waited {seconds}s"

    async def _navigate(self, action: PredictedAction) -> tuple[bool, str]:
        """Navigate to a URL."""
        url = action.value
        if not url or not url.startswith("http"):
            return False, f"Invalid URL: {url}"
        await self._page.goto(url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(1.5)
        return True, f"Navigated to {url}"

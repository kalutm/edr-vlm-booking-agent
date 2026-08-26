"""
vlm/perception.py — Visual Perception Module (Module 1 of 4)

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
Analyze the provided browser screenshot and return a structured description of the current page state.

## Booking Context
- Route: {origin} → {destination}
- Target Date: {target_date}
- Preferred Seat Type: {preferred_seat}
- Current Workflow Step: {workflow_step}

## Instructions

1. **current_page**: Identify which page you are on:
   - HOME: The main booking search form (has "Departure station" / "Destination station" inputs)
   - SEARCH_RESULTS: A list of available train schedules shown after searching. Look for schedule cards with departure/arrival times and seat availability.
   - AUTH_CHECK: The /booking/auth-check page. Shows a choice between "Continue as guest" and "Sign in / Register". Green guest button visible.
   - PASSENGERS: The /booking/passengers page. Shows passenger details form and a "Verify with Fayda" button for biometric ID verification.
   - SEAT_MAP: The /booking/seats page. Shows an interactive seat grid/map with an "Auto Assign Seats" button and a Selection Summary panel.
   - REVIEW: The /booking/review page. Shows a full booking summary (route, seats, price) and a green "Confirm and pay" button.
   - SEAT_SELECTION: A seat class selection screen (older flow)
   - PASSENGER_DETAILS: A passenger details form (older flow)
   - PAYMENT: A payment form or payment provider page
   - VERIFICATION: An identity verification screen (non-Fayda)
   - BOOKING_CONFIRMATION: A booking success / confirmation page with booking reference number
   - ERROR_PAGE: An error or "no results" page
   - UNKNOWN: Cannot determine

2. **date_state** (only relevant on HOME or calendar pages):
   - AVAILABLE: The target date ({target_date}) appears selectable/highlighted in the calendar
   - NOT_YET_OPEN: The date exists on the calendar but appears grayed out / disabled
   - INVALID: The date is not shown or marked as non-operating
   - UNKNOWN: Cannot determine date state from this screenshot

3. **schedule_state** (only relevant on SEARCH_RESULTS pages):
   - SCHEDULE_FOUND: At least one train schedule card is visible with a "Select" button available
   - SCHEDULE_FULL: Schedule cards are visible BUT all show a "Fully booked" icon/badge or text, with no available "Select" button
   - NO_RESULTS: "No trains found" or similar message visible, no schedule cards present
   - UNKNOWN: Cannot determine

4. **available_seats**: On SEARCH_RESULTS pages, read the availability text on each schedule card.
   EDR seat classes use these EXACT labels — report them verbatim:
   - "Regular Seat" — upright seating. Look for text like "Regular Seat 744 left" or "Regular Seat Full"
   - "Economy Bed Lower" — lower berth in Economy coach
   - "Economy Bed Middle" — middle berth in Economy coach
   - "Economy Bed Upper" — upper berth in Economy coach
   - "VIP Bed Lower" — lower berth in VIP private compartment
   - "VIP Bed Upper" — upper berth in VIP private compartment
   Set `available=true` if a count > 0 is shown, `available=false` if "Full" or count is 0.
   Include the count (integer) if visible.

5. **preferred_seat_available**: Is "{preferred_seat}" specifically available? null if unknown.

6. **visible_controls**: List the main interactive elements visible. Be specific about:
   - Selector hints (e.g., "button:has-text('Select')", "button:has-text('Auto Assign Seats')")
   - Whether controls are enabled or disabled
   - Do NOT attempt to guess how to interact with complex dropdowns, date pickers, or nationality selectors. The deterministic application logic will handle them.

7. **requires_human**: Set true if:
   - You see a payment form requiring credit card entry
   - You see a national ID / passport upload or entry form (but NOT Fayda — that is handled by the agent's deterministic step)
   - You see a CAPTCHA
   - You see OTP / SMS verification

8. **error_message**: Quote any error text visible on screen.

9. **confidence**: How confident are you in this analysis? (0.0 = no idea, 1.0 = completely certain)

10. **raw_description**: Write 1-2 sentences describing what you see.

Be precise. Do not guess. If something is unclear, use UNKNOWN values.
"""


class PerceptionModule:
    """
    Module 1: Visual Perception
    
    Takes a screenshot and returns a structured understanding of the current
    page state. No decisions are made here — only observation.
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
        workflow_step: str,
    ) -> PerceptionResult:
        """
        Analyze a screenshot and return structured perception.

        Args:
            screenshot_bytes: PNG screenshot from the browser
            origin: Departure station name
            destination: Arrival station name
            target_date: The date we're trying to book
            preferred_seat: User's preferred seat type name
            workflow_step: Current workflow step for context

        Returns:
            PerceptionResult with structured understanding of the page
        """
        prompt = PERCEPTION_PROMPT_TEMPLATE.format(
            origin=origin,
            destination=destination,
            target_date=target_date.isoformat(),
            preferred_seat=preferred_seat,
            workflow_step=workflow_step,
        )

        logger.info(f"[PERCEPTION] Analyzing screenshot — step: {workflow_step}")
        result = self._client.perceive(screenshot_bytes, prompt, PerceptionResult)
        logger.info(
            f"[PERCEPTION] Result: page={result.current_page.value}, "
            f"date={result.date_state.value}, schedule={result.schedule_state.value}, "
            f"seats={len(result.available_seats)}, human={result.requires_human}, "
            f"confidence={result.confidence:.2f}"
        )
        return result

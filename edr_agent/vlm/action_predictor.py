"""
vlm/action_predictor.py — Action Prediction Module (Module 2 of 4)

Input:  PerceptionResult + AgentState + goal context
Output: PredictedAction (structured, validated)

This module decides WHAT to do next based on what was perceived.
Actions are constrained to an enum — no arbitrary commands are generated.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from edr_agent.vlm.client import VLMClient
from edr_agent.vlm.schemas import PerceptionResult, PredictedAction, ActionType

logger = logging.getLogger(__name__)


ACTION_PROMPT_TEMPLATE = """You are the Action Prediction Module of an AI booking agent for the Ethiopian-Djibouti Railway (EDR) website.

## Your Task
Given the current browser state (described below) and the agent's goal, predict the SINGLE BEST next browser action.

## Current Page Analysis
{perception_summary}

## Agent Goal
{goal_description}

## Booking Context
- Route: {origin} → {destination}
- Target Date: {target_date}
- Preferred Seat: {preferred_seat}
- Current Step: {workflow_step}
- Retry Count: {retry_count}

## Available Action Types
You MUST choose one of these action types ONLY:
- CLICK: Click a button, link, or interactive element
- TYPE: Type text into an input field (clears field first)
- SELECT: Choose an option from a dropdown
- SCROLL: Scroll the page (specify direction in value: "down", "up")
- WAIT: Wait for page to load (specify seconds in value: "2")
- NAVIGATE: Go to a URL (specify URL in value)
- FILL_SEARCH_FORM: Hand off the entire initial search form (origin, destination, date, nationality) to the deterministic helper
- STOP: Agent should stop — goal achieved or impossible
- HUMAN_HANDOFF: Hand control to human (payment, verification, etc.)

## Rules
1. Choose HUMAN_HANDOFF if the current page requires personal identity info, payment, or CAPTCHA.
2. Choose STOP if booking is confirmed or the situation is unresolvable.
3. Prefer specific CSS selectors over coordinates when possible.
4. For dropdown/station selection, click the option button from the suggestion list.
5. DO NOT attempt to type a date string into the date field. Manual text entry for dates is prohibited. Click the "DATE" input field / calendar icon (button:has(svg.lucide-calendar)) to open the pop-up calendar card, then click the target day number inside the calendar modal.
6. Do not attempt to guess how to interact with complex dropdowns, date pickers, or nationality selectors. If the current task requires filling out the initial search form, predict a high-level FILL_SEARCH_FORM action, and the deterministic application logic will handle the specific clicks.
7. If you see a "Search" button and the form looks filled, click it.
8. One action at a time — never chain multiple actions.

## Selector Guidance for EDR Website
- Departure station input: input[placeholder="Departure station"]
- Destination station input: input[placeholder="Destination station"]  
- Date field button (opens calendar pop-up): button:has(svg.lucide-calendar)
- Day number inside calendar modal: button:has-text("26") (where 26 is the target day)
- Search/Submit button: button[type="submit"], button:has-text("Search")
- Dropdown options: Look for list items after clicking a station input

Provide a specific, actionable prediction. Explain your reasoning in the "reason" field.
"""


# Guard: actions that are always safe vs. require review
SAFE_ACTIONS = {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT,
                ActionType.SCROLL, ActionType.WAIT, ActionType.NAVIGATE, ActionType.FILL_SEARCH_FORM}
SENSITIVE_ACTIONS = {ActionType.HUMAN_HANDOFF, ActionType.STOP}


class ActionPredictorModule:
    """
    Module 2: Action Prediction
    
    Given the current perceived state and agent goal, predicts the next
    browser action. Actions are constrained to a safe, validated enum.
    """

    def __init__(self, vlm_client: VLMClient) -> None:
        self._client = vlm_client

    def predict(
        self,
        screenshot_bytes: bytes,
        perception: PerceptionResult,
        origin: str,
        destination: str,
        target_date: date,
        preferred_seat: str,
        workflow_step: str,
        goal_description: str,
        retry_count: int = 0,
    ) -> PredictedAction:
        """
        Predict the next browser action based on current state.

        Args:
            screenshot_bytes: Current page screenshot
            perception: Result from the perception module
            origin/destination/target_date/preferred_seat: Booking parameters
            workflow_step: Current step for context
            goal_description: What we're trying to achieve right now
            retry_count: How many retries have occurred

        Returns:
            PredictedAction with validated action type and parameters
        """
        perception_summary = self._format_perception(perception)

        prompt = ACTION_PROMPT_TEMPLATE.format(
            perception_summary=perception_summary,
            goal_description=goal_description,
            origin=origin,
            destination=destination,
            target_date=target_date.isoformat(),
            preferred_seat=preferred_seat,
            workflow_step=workflow_step,
            retry_count=retry_count,
        )

        logger.info(f"[ACTION] Predicting action — step: {workflow_step}, goal: {goal_description[:60]}...")
        action = self._client.predict_action(screenshot_bytes, prompt, PredictedAction)

        # Validate the action before returning
        action = self._validate(action)

        logger.info(
            f"[ACTION] Predicted: {action.action_type.value} | "
            f"selector={action.target_selector} | "
            f"value={action.value} | "
            f"reason={action.reason[:80]}..."
        )
        return action

    def _format_perception(self, p: PerceptionResult) -> str:
        """Format perception result as readable context for the action prompt."""
        seats_str = ", ".join(
            f"{s.seat_type}({'available' if s.available else 'unavailable'})"
            for s in p.available_seats
        ) or "none visible"

        controls_str = "\n".join(
            f"  - [{c.control_type}] '{c.label}' → {c.selector_hint}"
            for c in p.visible_controls[:8]
        ) or "  - none identified"

        return f"""
Page: {p.current_page.value}
Date State: {p.date_state.value}
Schedule State: {p.schedule_state.value}
Available Seats: {seats_str}
Preferred Seat Available: {p.preferred_seat_available}
Requires Human: {p.requires_human}
Error: {p.error_message or 'none'}
Confidence: {p.confidence:.2f}
Description: {p.raw_description}
Visible Controls:
{controls_str}
""".strip()

    def _validate(self, action: PredictedAction) -> PredictedAction:
        """
        Apply safety validation rules to a predicted action.
        Ensures actions are safe and properly formed.
        """
        # NAVIGATE requires a value (URL)
        if action.action_type == ActionType.NAVIGATE:
            if not action.value or not action.value.startswith("http"):
                logger.warning("[ACTION] NAVIGATE without valid URL — converting to WAIT")
                action.action_type = ActionType.WAIT
                action.value = "2"
                action.reason += " [VALIDATED: missing URL, converted to WAIT]"

        # TYPE validation: prohibit typing into date fields
        if action.action_type == ActionType.TYPE:
            val = (action.value or "").lower()
            sel = (action.target_selector or "").lower()
            if "date" in sel or "calendar" in sel or re.search(r"\d{4}-\d{2}-\d{2}", val):
                logger.warning("[ACTION] Prohibited TYPE action on date field detected — converting to CLICK")
                action.action_type = ActionType.CLICK
                action.target_selector = "button:has(svg.lucide-calendar)"
                action.value = None
                action.reason += " [VALIDATED: text entry for dates is prohibited; converted TYPE to CLICK on date field]"

        return action

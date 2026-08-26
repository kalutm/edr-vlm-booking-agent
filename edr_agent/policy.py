"""
policy.py — Booking Policy Engine

Implements the three booking policies (NORMAL, SEAT_FIRST, DATE_FIRST).
All policy logic is deterministic — no VLM involvement here.

Each policy receives the current AgentState and returns a PolicyDecision
telling the agent what to do next.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from edr_agent.config import BookingMode, SeatType
from edr_agent.date_logic import next_operating_date
from edr_agent.state import AgentState, WorkflowStep
from edr_agent.vlm.schemas import DateAvailability, ScheduleAvailability

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    PROCEED_TO_BOOKING = "PROCEED_TO_BOOKING"   # Everything looks good, book it
    ADVANCE_DATE = "ADVANCE_DATE"               # Move to next operating date (SEAT-FIRST)
    TRY_NEXT_SEAT = "TRY_NEXT_SEAT"            # Try next ranked seat (DATE-FIRST)
    NOTIFY_AND_STOP = "NOTIFY_AND_STOP"         # Inform user and terminate
    WAIT_AND_RETRY = "WAIT_AND_RETRY"           # Date not open yet, wait for next cycle
    HUMAN_HANDOFF = "HUMAN_HANDOFF"             # Hand to human for verification/payment


@dataclass
class PolicyDecision:
    action: PolicyAction
    reason: str
    new_date: Optional[date] = None           # Set when ADVANCE_DATE
    new_seat: Optional[SeatType] = None       # Set when TRY_NEXT_SEAT
    notification_message: str = ""            # Message to show user


class PolicyEngine:
    """
    Evaluates the current agent state against the configured booking policy
    and returns a deterministic decision about what to do next.
    """

    def evaluate(self, state: AgentState) -> PolicyDecision:
        """
        Evaluate current state and return policy decision.

        Called after perception has updated the state with:
        - date_availability
        - schedule_availability
        - seat_availability / preferred_seat_available
        """
        mode = state.booking_mode

        # --- Handle date states first (common to all modes) ---
        if state.date_availability == DateAvailability.INVALID:
            return PolicyDecision(
                action=PolicyAction.NOTIFY_AND_STOP,
                reason=f"Date {state.current_date} is not a valid EDR operating day",
                notification_message=(
                    f"❌ The date {state.current_date} is not a valid operating day "
                    f"for the {state.origin} → {state.destination} route. "
                    f"Please choose a different date."
                ),
            )

        if state.date_availability == DateAvailability.NOT_YET_OPEN:
            return PolicyDecision(
                action=PolicyAction.WAIT_AND_RETRY,
                reason=f"Date {state.current_date} is valid but booking not yet open",
                notification_message=(
                    f"⏳ Booking for {state.current_date} is not yet open. "
                    f"Will retry in {state.monitoring_interval_minutes} minutes."
                ),
            )

        # --- Date is AVAILABLE — evaluate schedule ---
        if state.schedule_availability == ScheduleAvailability.NO_RESULTS:
            return self._handle_no_schedule(state, mode)

        if state.schedule_availability == ScheduleAvailability.SCHEDULE_FULL:
            return self._handle_full_schedule(state, mode)

        # --- Schedule found — evaluate seats ---
        if state.schedule_availability == ScheduleAvailability.SCHEDULE_FOUND:
            return self._handle_seats(state, mode)

        # Default: not enough info yet, proceed with workflow
        return PolicyDecision(
            action=PolicyAction.PROCEED_TO_BOOKING,
            reason="Insufficient information to evaluate — continuing workflow",
        )

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _handle_no_schedule(self, state: AgentState, mode: BookingMode) -> PolicyDecision:
        """Handle the case where no schedule was found."""
        if mode == BookingMode.SEAT_FIRST:
            next_date = next_operating_date(state.current_date)
            if next_date:
                logger.info(f"[POLICY] SEAT-FIRST: no schedule on {state.current_date}, advancing to {next_date}")
                return PolicyDecision(
                    action=PolicyAction.ADVANCE_DATE,
                    reason=f"SEAT-FIRST: no schedule on {state.current_date}, trying next operating date",
                    new_date=next_date,
                    notification_message=f"🔄 No schedule on {state.current_date}, trying {next_date}",
                )
            else:
                return PolicyDecision(
                    action=PolicyAction.NOTIFY_AND_STOP,
                    reason="SEAT-FIRST: no operating dates found within look-ahead window",
                    notification_message="❌ No available schedules found in the coming weeks.",
                )

        # NORMAL and DATE-FIRST: stop
        return PolicyDecision(
            action=PolicyAction.NOTIFY_AND_STOP,
            reason=f"No schedule found for {state.current_date}",
            notification_message=(
                f"❌ No train schedule found for {state.current_date} "
                f"on the {state.origin} → {state.destination} route."
            ),
        )

    def _handle_full_schedule(self, state: AgentState, mode: BookingMode) -> PolicyDecision:
        """Handle the case where schedules exist but all are full."""
        if mode == BookingMode.SEAT_FIRST:
            next_date = next_operating_date(state.current_date)
            if next_date:
                logger.info(f"[POLICY] SEAT-FIRST: train is fully booked on {state.current_date}, advancing to {next_date}")
                return PolicyDecision(
                    action=PolicyAction.ADVANCE_DATE,
                    reason=f"SEAT-FIRST: schedule fully booked on {state.current_date}, trying next operating date",
                    new_date=next_date,
                    notification_message=f"🔄 Train fully booked on {state.current_date}, trying {next_date}",
                )
            else:
                return PolicyDecision(
                    action=PolicyAction.NOTIFY_AND_STOP,
                    reason="SEAT-FIRST: no operating dates found within look-ahead window",
                    notification_message="❌ Train is fully booked and no future available schedules were found.",
                )

        # NORMAL and DATE-FIRST: stop
        return PolicyDecision(
            action=PolicyAction.NOTIFY_AND_STOP,
            reason=f"Train is fully booked for {state.current_date}",
            notification_message=(
                f"❌ The train is fully booked for {state.current_date} "
                f"on the {state.origin} → {state.destination} route."
            ),
        )

    def _handle_seats(self, state: AgentState, mode: BookingMode) -> PolicyDecision:
        """Handle seat availability evaluation."""
        preferred = state.preferred_seat
        preferred_key = preferred.value

        # Check if preferred seat is available
        preferred_available = state.seat_availability.get(preferred_key, False)

        if preferred_available:
            logger.info(f"[POLICY] Preferred seat '{preferred_key}' is available — proceed to booking")
            return PolicyDecision(
                action=PolicyAction.PROCEED_TO_BOOKING,
                reason=f"Preferred seat '{preferred_key}' is available",
                notification_message=f"✅ {preferred_key} seat available! Proceeding to booking.",
            )

        # Preferred seat unavailable — apply mode-specific logic
        if mode == BookingMode.NORMAL:
            available_seats = [k for k, v in state.seat_availability.items() if v]
            return PolicyDecision(
                action=PolicyAction.NOTIFY_AND_STOP,
                reason=f"NORMAL mode: preferred seat '{preferred_key}' unavailable",
                notification_message=(
                    f"⚠️ {preferred_key} class is unavailable on {state.current_date}.\n"
                    f"Available seats: {', '.join(available_seats) or 'none'}.\n"
                    f"Monitoring stopped (NORMAL mode)."
                ),
            )

        elif mode == BookingMode.SEAT_FIRST:
            next_date = next_operating_date(state.current_date)
            if next_date:
                logger.info(f"[POLICY] SEAT-FIRST: '{preferred_key}' unavailable on {state.current_date}, advancing")
                return PolicyDecision(
                    action=PolicyAction.ADVANCE_DATE,
                    reason=f"SEAT-FIRST: preferred seat unavailable, trying next date",
                    new_date=next_date,
                    notification_message=f"🔄 {preferred_key} unavailable on {state.current_date}, trying {next_date}",
                )
            else:
                return PolicyDecision(
                    action=PolicyAction.NOTIFY_AND_STOP,
                    reason="SEAT-FIRST: exhausted look-ahead window without finding preferred seat",
                    notification_message=f"❌ {preferred_key} not available in coming weeks.",
                )

        elif mode == BookingMode.DATE_FIRST:
            return self._date_first_seat_fallback(state)

        return PolicyDecision(
            action=PolicyAction.NOTIFY_AND_STOP,
            reason="Unknown mode or unresolvable state",
        )

    def _date_first_seat_fallback(self, state: AgentState) -> PolicyDecision:
        """
        DATE-FIRST mode: try the next ranked seat type.
        Falls back to NORMAL behavior when the ranked list is exhausted.
        """
        ranking = state.seat_ranking
        current_idx = state.current_seat_rank_index

        # Try the next ranked seats
        for idx in range(current_idx, len(ranking)):
            seat = ranking[idx]
            if state.seat_availability.get(seat.value, False):
                logger.info(f"[POLICY] DATE-FIRST: using ranked seat '{seat.value}' (rank {idx})")
                return PolicyDecision(
                    action=PolicyAction.PROCEED_TO_BOOKING,
                    reason=f"DATE-FIRST: using seat rank {idx}: '{seat.value}'",
                    new_seat=seat,
                    notification_message=f"✅ Using {seat.value} seat (rank #{idx+1} preference) on {state.current_date}",
                )

        # All ranked seats exhausted — switch to NORMAL behavior
        available_seats = [k for k, v in state.seat_availability.items() if v]
        return PolicyDecision(
            action=PolicyAction.NOTIFY_AND_STOP,
            reason="DATE-FIRST: all ranked seats unavailable, falling back to NORMAL stop",
            notification_message=(
                f"⚠️ None of your preferred seat types are available on {state.current_date}.\n"
                f"Available: {', '.join(available_seats) or 'none'}."
            ),
        )

"""
state.py — Agent State Machine

Explicit state tracking for the EDR booking agent.
This is the ground truth of what the agent knows/believes at any moment.
The VLM's conversation memory is never relied upon for state persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

from edr_agent.config import BookingMode, SeatType, UserConfig
from edr_agent.vlm.schemas import (
    DateAvailability,
    PerceptionResult,
    ScheduleAvailability,
)


# ---------------------------------------------------------------------------
# Workflow step enum — all possible states in the booking state machine
# ---------------------------------------------------------------------------

class WorkflowStep(str, Enum):
    # Phase 1 — VLM-assisted search (Playwright fills form; VLM reads schedule/seats)
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    FILLING_FORM = "FILLING_FORM"   # Playwright deterministically fills the search form
    SEARCHING = "SEARCHING"         # VLM reads SEARCH_RESULTS page for schedules
    INSPECTING_SEATS = "INSPECTING_SEATS"  # VLM reads seat availability
    APPLYING_POLICY = "APPLYING_POLICY"
    BOOKING = "BOOKING"
    # Phase 2 — deterministic post-search booking steps (Playwright only, no VLM)
    SELECTING_SCHEDULE = "SELECTING_SCHEDULE"   # Clicking 'Select' + coach modal on results page
    AUTH_CHECK = "AUTH_CHECK"                   # /booking/auth-check — clicking 'Continue as guest'
    FAYDA_VERIFICATION = "FAYDA_VERIFICATION"   # /booking/passengers — Fayda biometric handoff
    SEAT_MAP = "SEAT_MAP"                       # /booking/seats — auto-assign + continue
    REVIEWING = "REVIEWING"                     # /booking/review — final review page
    PAYMENT_HANDOFF = "PAYMENT_HANDOFF"         # After 'Confirm and pay' — human takes over
    # Control states
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    MONITORING = "MONITORING"
    WAITING_INTERVAL = "WAITING_INTERVAL"
    RETRYING = "RETRYING"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"
    SUCCESS = "SUCCESS"


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """
    The complete runtime state of the booking agent.
    Updated deterministically by application logic after each perception cycle.
    """

    # --- Route ---
    origin: str = ""
    destination: str = ""

    # --- Dates ---
    target_date: Optional[date] = None       # User's originally requested date
    current_date: Optional[date] = None      # Current date being attempted (may shift in SEAT-FIRST)

    # --- Preferences ---
    preferred_seat: SeatType = SeatType.REGULAR_SEAT
    seat_ranking: list[SeatType] = field(default_factory=list)
    current_seat_rank_index: int = 0         # Index into seat_ranking for DATE-FIRST fallback
    booking_mode: BookingMode = BookingMode.NORMAL
    nationality: str = "Ethiopian"
    num_passengers: int = 1

    # --- Workflow ---
    workflow_step: WorkflowStep = WorkflowStep.IDLE
    cycle_count: int = 0
    retry_count: int = 0
    consecutive_failures: int = 0
    monitoring_cycle: int = 0

    # --- Observed availability ---
    date_availability: DateAvailability = DateAvailability.UNKNOWN
    schedule_availability: ScheduleAvailability = ScheduleAvailability.UNKNOWN
    seat_availability: dict[str, bool] = field(default_factory=dict)

    # --- Last VLM outputs ---
    last_perception: Optional[PerceptionResult] = None
    last_screenshot_path: Optional[str] = None

    # --- Monitoring ---
    monitoring_active: bool = False
    monitoring_interval_minutes: int = 10
    max_monitoring_cycles: int = 48
    next_check_time: Optional[datetime] = None

    # --- Human handoff ---
    waiting_for_human: bool = False
    human_handoff_reason: Optional[str] = None
    # Stage of handoff: None | "FAYDA" | "PAYMENT"
    # Used by the UI to show the correct resume/action button.
    handoff_stage: Optional[str] = None

    # --- Error / termination ---
    error: Optional[str] = None
    terminated: bool = False
    success: bool = False

    # --- Timestamps ---
    started_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    @classmethod
    def from_config(cls, config: UserConfig) -> "AgentState":
        """Create initial state from user configuration."""
        now = datetime.now()
        return cls(
            origin=config.origin,
            destination=config.destination,
            target_date=config.travel_date,
            current_date=config.travel_date,
            preferred_seat=config.preferred_seat,
            seat_ranking=config.seat_ranking,
            booking_mode=config.booking_mode,
            nationality=config.nationality,
            num_passengers=config.num_passengers,
            monitoring_interval_minutes=config.monitoring_interval_minutes,
            max_monitoring_cycles=config.max_monitoring_cycles,
            workflow_step=WorkflowStep.IDLE,
            started_at=now,
            last_updated=now,
        )

    def transition(self, new_step: WorkflowStep, log: bool = True) -> None:
        """Transition to a new workflow step."""
        old_step = self.workflow_step
        self.workflow_step = new_step
        self.last_updated = datetime.now()
        if log:
            print(f"[STATE] {old_step.value} → {new_step.value}")

    def record_failure(self, error_msg: str) -> None:
        self.retry_count += 1
        self.consecutive_failures += 1
        self.error = error_msg
        self.last_updated = datetime.now()

    def record_success_cycle(self) -> None:
        self.consecutive_failures = 0
        self.error = None
        self.last_updated = datetime.now()

    def reset_for_next_monitoring_cycle(self) -> None:
        """Reset transient state for the next monitoring attempt."""
        self.retry_count = 0
        self.date_availability = DateAvailability.UNKNOWN
        self.schedule_availability = ScheduleAvailability.UNKNOWN
        self.seat_availability = {}
        self.last_perception = None
        self.error = None
        self.monitoring_cycle += 1
        self.last_updated = datetime.now()

    def to_summary_dict(self) -> dict:
        """Compact dict for UI display and logging."""
        return {
            "workflow_step": self.workflow_step.value,
            "origin": self.origin,
            "destination": self.destination,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "current_date": self.current_date.isoformat() if self.current_date else None,
            "preferred_seat": self.preferred_seat.value,
            "booking_mode": self.booking_mode.value,
            "date_availability": self.date_availability.value,
            "schedule_availability": self.schedule_availability.value,
            "seat_availability": self.seat_availability,
            "retry_count": self.retry_count,
            "monitoring_cycle": self.monitoring_cycle,
            "cycle_count": self.cycle_count,
            "consecutive_failures": self.consecutive_failures,
            "waiting_for_human": self.waiting_for_human,
            "human_handoff_reason": self.human_handoff_reason,
            "handoff_stage": self.handoff_stage,
            "next_check_time": self.next_check_time.isoformat() if self.next_check_time else None,
            "error": self.error,
            "terminated": self.terminated,
            "success": self.success,
            "monitoring_active": self.monitoring_active,
        }

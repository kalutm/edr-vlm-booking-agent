"""
vlm/schemas.py — Pydantic schemas for all VLM inputs and outputs.

These are the structured data contracts between the application and the VLM.
The VLM must return data that matches these schemas — enforced via Gemini's
response_schema parameter.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums used in VLM outputs
# ---------------------------------------------------------------------------

class PageType(str, Enum):
    # Pages the VLM is actually invoked on
    HOME = "HOME"                   # Main booking search form
    SEARCH_RESULTS = "SEARCH_RESULTS"  # Train schedule list after searching
    ERROR_PAGE = "ERROR_PAGE"       # Any error / no-results page
    UNKNOWN = "UNKNOWN"             # Cannot determine


class DateAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"        # Date is selectable / open for booking
    NOT_YET_OPEN = "NOT_YET_OPEN"  # Valid route day but booking not open yet
    INVALID = "INVALID"            # Not an operating day for this route
    UNKNOWN = "UNKNOWN"            # Cannot determine from the screenshot


class ScheduleAvailability(str, Enum):
    SCHEDULE_FOUND = "SCHEDULE_FOUND"   # At least one train found
    SCHEDULE_FULL = "SCHEDULE_FULL"     # Trains exist but all full
    NO_RESULTS = "NO_RESULTS"           # No schedules returned at all
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# VLM Output: Perception Result
# ---------------------------------------------------------------------------

class SeatInfo(BaseModel):
    seat_type: str = Field(description="Name of the seat class (e.g. Regular, Economy, VIP)")
    available: bool = Field(description="Whether this seat type has availability")
    count: Optional[int] = Field(default=None, description="Number of available seats if visible")
    price: Optional[str] = Field(default=None, description="Price shown for this seat type")


class PerceptionResult(BaseModel):
    """
    Structured output from the Visual Perception Module.
    Represents what the VLM understands about the current browser state.
    """
    current_page: PageType = Field(description="Which page/screen is currently visible")
    schedule_state: ScheduleAvailability = Field(
        default=ScheduleAvailability.UNKNOWN,
        description="State of train schedule availability"
    )
    available_seats: list[SeatInfo] = Field(
        default_factory=list,
        description="List of seat types visible on screen with their availability"
    )
    preferred_seat_available: Optional[bool] = Field(
        default=None,
        description="Whether the user's preferred seat type is available"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Any error or warning message visible on screen"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="VLM confidence in this perception (0.0-1.0)"
    )
    raw_description: str = Field(
        default="",
        description="Brief free-text description of what is visible, for logging/debugging"
    )


# ---------------------------------------------------------------------------
# Internal event types (for WebSocket broadcast to UI)
# ---------------------------------------------------------------------------

class CycleEvent(BaseModel):
    """Emitted after each feedback loop cycle for UI updates."""
    cycle_number: int
    timestamp: str
    workflow_step: str
    perception: Optional[PerceptionResult] = None
    state_summary: dict = Field(default_factory=dict)
    screenshot_path: Optional[str] = None
    log_message: str = ""
    is_error: bool = False

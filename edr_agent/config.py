"""
config.py — UserConfig, station definitions, constants.

All deterministic application settings live here.
The VLM never reads this directly — it only gets structured context passed to it.
"""

from __future__ import annotations

import os
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enums & Domain Data
# ---------------------------------------------------------------------------

class BookingMode(str, Enum):
    NORMAL = "NORMAL"
    SEAT_FIRST = "SEAT_FIRST"
    DATE_FIRST = "DATE_FIRST"


class SeatType(str, Enum):
    # These values exactly match the text labels in the EDR "Choose Your Coach" modal.
    # Regular Seat: upright seating in open carriage (no berth)
    REGULAR_SEAT = "Regular Seat"
    # Economy Bed: open-corridor compartment with 6 bunks per bay (3 berth levels)
    ECONOMY_BED_LOWER = "Economy Bed Lower"
    ECONOMY_BED_MIDDLE = "Economy Bed Middle"
    ECONOMY_BED_UPPER = "Economy Bed Upper"
    # VIP Bed: private enclosed compartment with 4 bunks (2 berth levels)
    VIP_BED_LOWER = "VIP Bed Lower"
    VIP_BED_UPPER = "VIP Bed Upper"
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Known EDR stations
# West-to-East route order: Sebeta -> Lebu -> Bishoftu -> Mojo -> Adama -> Mieso -> Bike -> Dire Dawa ...
# ---------------------------------------------------------------------------

EDR_STATIONS: list[dict] = [
    {"code": "SBT",  "name": "Sebeta",      "city": "Sheger",     "country": "ET", "sequence": 1},
    {"code": "LEB",  "name": "Lebu",        "city": "Sheger",     "country": "ET", "sequence": 2},
    {"code": "BISH", "name": "Bishoftu",    "city": "Bishoftu",   "country": "ET", "sequence": 3},
    {"code": "MOJ",  "name": "Modjo",       "city": "Modjo",      "country": "ET", "sequence": 4},
    {"code": "ADM",  "name": "Adama",       "city": "Adama",      "country": "ET", "sequence": 5},
    {"code": "MTH",  "name": "Metehara",    "city": "Metehara",   "country": "ET", "sequence": 6},
    {"code": "MSO",  "name": "Mieso",       "city": "Mieso",      "country": "ET", "sequence": 7},
    {"code": "BIK",  "name": "Bike",        "city": "Bike",       "country": "ET", "sequence": 8},
    {"code": "DIR",  "name": "Dire Dawa",   "city": "Dire Dawa",  "country": "ET", "sequence": 9},
    {"code": "ADG",  "name": "Adigala",     "city": "Adigala",    "country": "ET", "sequence": 10},
    {"code": "AYS",  "name": "Aysha",       "city": "Aysha",      "country": "ET", "sequence": 11},
    {"code": "DAW",  "name": "Dawanle",     "city": "Dawanle",    "country": "ET", "sequence": 12},
    {"code": "ALI",  "name": "Alisabieh",   "city": "Alisabieh",  "country": "DJ", "sequence": 13},
    {"code": "HOL",  "name": "Holhol",      "city": "Holhol",     "country": "DJ", "sequence": 14},
    {"code": "NAG",  "name": "Nagad",       "city": "Nagad",      "country": "DJ", "sequence": 15},
]

STATION_NAMES: list[str] = [s["name"] for s in EDR_STATIONS]
STATION_BY_CODE: dict[str, dict] = {s["code"]: s for s in EDR_STATIONS}
STATION_SEQUENCE: dict[str, int] = {s["name"].lower(): s["sequence"] for s in EDR_STATIONS}
# Add alias for Mojo
STATION_SEQUENCE["mojo"] = STATION_SEQUENCE["modjo"]


def get_travel_direction(origin: str, destination: str) -> str:
    """
    Determine if route is EASTBOUND (West to East, e.g. Sebeta/Lebu -> Dire Dawa)
    or WESTBOUND (East to West, e.g. Dire Dawa -> Lebu/Sebeta).
    Default is EASTBOUND.
    """
    orig_seq = STATION_SEQUENCE.get(origin.strip().lower(), 2)  # Default Lebu (2)
    dest_seq = STATION_SEQUENCE.get(destination.strip().lower(), 9)  # Default Dire Dawa (9)

    if orig_seq <= dest_seq:
        return "EASTBOUND"
    else:
        return "WESTBOUND"


# ---------------------------------------------------------------------------
# Application settings (loaded from environment / .env file)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")
    edr_base_url: str = Field(default="https://bookingedr.et", alias="EDR_BASE_URL")
    browser_headless: bool = Field(default=False, alias="BROWSER_HEADLESS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    server_host: str = Field(default="127.0.0.1", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "populate_by_name": True}


settings = Settings()


# ---------------------------------------------------------------------------
# UserConfig — what the user supplies when starting the agent
# ---------------------------------------------------------------------------

class UserConfig:
    """Represents the user's booking request and operating parameters."""

    def __init__(
        self,
        origin: str,
        destination: str,
        travel_date: date,
        preferred_seat: SeatType = SeatType.REGULAR_SEAT,
        seat_ranking: Optional[list[SeatType]] = None,
        booking_mode: BookingMode = BookingMode.NORMAL,
        monitoring_interval_minutes: int = 10,
        max_monitoring_cycles: int = 48,   # stop after 48 cycles (~8 hrs at 10 min)
        trip_type: str = "one_way",
        nationality: str = "Ethiopian",
        num_passengers: int = 1,
    ):
        self.origin = origin
        self.destination = destination
        self.travel_date = travel_date
        self.preferred_seat = preferred_seat
        # seat_ranking used for DATE-FIRST mode (up to 6 ranked seats)
        self.seat_ranking = seat_ranking or [preferred_seat]
        self.booking_mode = booking_mode
        self.monitoring_interval_minutes = monitoring_interval_minutes
        self.max_monitoring_cycles = max_monitoring_cycles
        self.trip_type = trip_type
        self.nationality = nationality
        self.num_passengers = num_passengers

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "travel_date": self.travel_date.isoformat(),
            "preferred_seat": self.preferred_seat.value,
            "seat_ranking": [s.value for s in self.seat_ranking],
            "booking_mode": self.booking_mode.value,
            "monitoring_interval_minutes": self.monitoring_interval_minutes,
            "nationality": self.nationality,
            "num_passengers": self.num_passengers,
        }

"""
date_logic.py — Deterministic EDR operating-date rules.

All date arithmetic is handled here in ordinary Python logic.
The VLM never performs date calculations — it only visually confirms
whether a date appears selectable on screen.

EDR Operating Pattern (Confirmed Domain Rules):
1. Odd-Numbered Days (Train 101): Departs from Sebeta/Lebu (Addis Ababa area) and travels EAST toward Dire Dawa (1st, 3rd, 5th, etc.)
2. Even-Numbered Days (Train 102): Departs from Dire Dawa and travels WEST back to Sebeta (2nd, 4th, 6th, etc.)
3. CRITICAL RULE: NO train operates on the 31st of any month. If logic lands on the 31st, it skips to the 1st of the next month.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from edr_agent.config import get_travel_direction


def is_edr_operating_day(d: date, origin: str = "Lebu", destination: str = "Dire Dawa") -> bool:
    """
    Return True if the given date is an operating day for the specified route.

    Rules:
    - 31st of any month: NO TRAIN OPERATES (always False).
    - EASTBOUND (Sebeta/Lebu -> Dire Dawa, Train 101): Runs on ODD calendar days (1, 3, 5... 29).
    - WESTBOUND (Dire Dawa -> Lebu/Sebeta, Train 102): Runs on EVEN calendar days (2, 4, 6... 30).
    """
    # CRITICAL RULE: No train operates on the 31st of any month
    if d.day == 31:
        return False

    direction = get_travel_direction(origin, destination)
    if direction == "EASTBOUND":
        return d.day % 2 == 1
    else:  # WESTBOUND
        return d.day % 2 == 0


def next_operating_date(
    from_date: date,
    origin: str = "Lebu",
    destination: str = "Dire Dawa",
    max_look_ahead: int = 30,
) -> Optional[date]:
    """
    Find the next EDR operating date strictly after from_date.
    If date arithmetic lands on the 31st of any month, immediately skips to the 1st of the next month.

    Args:
        from_date: Starting date (exclusive)
        origin: Departure station name
        destination: Destination station name
        max_look_ahead: Maximum days to look ahead

    Returns:
        Next operating date, or None if not found within window
    """
    candidate = from_date + timedelta(days=1)
    for _ in range(max_look_ahead):
        # CRITICAL RULE: Skip 31st of any month immediately to 1st of next month
        if candidate.day == 31:
            if candidate.month == 12:
                candidate = date(candidate.year + 1, 1, 1)
            else:
                candidate = date(candidate.year, candidate.month + 1, 1)

        if is_edr_operating_day(candidate, origin=origin, destination=destination):
            return candidate

        candidate += timedelta(days=1)

    return None


def classify_date(
    d: date,
    origin: str = "Lebu",
    destination: str = "Dire Dawa",
    today: Optional[date] = None,
) -> str:
    """
    Classify a date as VALID_OPERATING, NOT_OPERATING, or PAST.

    Returns:
        "VALID_OPERATING" | "NOT_OPERATING" | "PAST"
    """
    if today is None:
        today = date.today()

    if d < today:
        return "PAST"

    if is_edr_operating_day(d, origin=origin, destination=destination):
        return "VALID_OPERATING"
    else:
        return "NOT_OPERATING"


def get_next_n_operating_dates(
    from_date: date,
    n: int = 7,
    origin: str = "Lebu",
    destination: str = "Dire Dawa",
) -> list[date]:
    """
    Get the next N operating dates starting from (and including) from_date.
    Respects directional train rules (Odd vs Even) and skips 31st calendar days.
    """
    results: list[date] = []
    candidate = from_date
    attempts = 0

    while len(results) < n and attempts < 60:
        if candidate.day == 31:
            if candidate.month == 12:
                candidate = date(candidate.year + 1, 1, 1)
            else:
                candidate = date(candidate.year, candidate.month + 1, 1)

        if is_edr_operating_day(candidate, origin=origin, destination=destination):
            results.append(candidate)

        candidate += timedelta(days=1)
        attempts += 1

    return results


def format_date_for_edr(d: date) -> str:
    """Format a date as the EDR website expects it (YYYY-MM-DD)."""
    return d.isoformat()

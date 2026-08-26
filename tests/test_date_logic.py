"""
tests/test_date_logic.py — Unit tests for deterministic date & route logic.
Tests directional train schedules (Train 101 odd / Train 102 even) and 31st skipping rule.
"""

import pytest
from datetime import date
from edr_agent.date_logic import (
    is_edr_operating_day,
    next_operating_date,
    classify_date,
    get_next_n_operating_dates,
)
from edr_agent.config import get_travel_direction


class TestTravelDirection:
    def test_eastbound_direction(self):
        assert get_travel_direction("Lebu", "Dire Dawa") == "EASTBOUND"
        assert get_travel_direction("Sebeta", "Adama") == "EASTBOUND"

    def test_westbound_direction(self):
        assert get_travel_direction("Dire Dawa", "Lebu") == "WESTBOUND"
        assert get_travel_direction("Adama", "Sebeta") == "WESTBOUND"


class TestOperatingDay:
    def test_eastbound_train_101_odd_days(self):
        # Eastbound (Lebu -> Dire Dawa) runs on ODD days
        assert is_edr_operating_day(date(2025, 9, 1), origin="Lebu", destination="Dire Dawa") is True
        assert is_edr_operating_day(date(2025, 9, 3), origin="Lebu", destination="Dire Dawa") is True
        assert is_edr_operating_day(date(2025, 9, 2), origin="Lebu", destination="Dire Dawa") is False
        assert is_edr_operating_day(date(2025, 9, 4), origin="Lebu", destination="Dire Dawa") is False

    def test_westbound_train_102_even_days(self):
        # Westbound (Dire Dawa -> Lebu) runs on EVEN days
        assert is_edr_operating_day(date(2025, 9, 2), origin="Dire Dawa", destination="Lebu") is True
        assert is_edr_operating_day(date(2025, 9, 4), origin="Dire Dawa", destination="Lebu") is True
        assert is_edr_operating_day(date(2025, 9, 1), origin="Dire Dawa", destination="Lebu") is False
        assert is_edr_operating_day(date(2025, 9, 3), origin="Dire Dawa", destination="Lebu") is False

    def test_no_train_on_31st(self):
        # CRITICAL RULE: No train operates on the 31st of any month regardless of direction
        assert is_edr_operating_day(date(2025, 8, 31), origin="Lebu", destination="Dire Dawa") is False
        assert is_edr_operating_day(date(2025, 8, 31), origin="Dire Dawa", destination="Lebu") is False
        assert is_edr_operating_day(date(2025, 10, 31), origin="Lebu", destination="Dire Dawa") is False


class TestNextOperatingDate:
    def test_next_from_odd_day_eastbound(self):
        # Eastbound: From Sept 1 (odd), next should be Sept 3
        result = next_operating_date(date(2025, 9, 1), origin="Lebu", destination="Dire Dawa")
        assert result == date(2025, 9, 3)

    def test_next_from_even_day_westbound(self):
        # Westbound: From Sept 2 (even), next should be Sept 4
        result = next_operating_date(date(2025, 9, 2), origin="Dire Dawa", destination="Lebu")
        assert result == date(2025, 9, 4)

    def test_skips_31st_to_first_of_next_month(self):
        # Eastbound: Aug 30 -> Aug 31 is skipped -> Sept 1 (odd)
        result = next_operating_date(date(2025, 8, 30), origin="Lebu", destination="Dire Dawa")
        assert result == date(2025, 9, 1)

    def test_crosses_month_boundary_30_day_month(self):
        # Eastbound: From Sept 30, next operating should be Oct 1
        result = next_operating_date(date(2025, 9, 30), origin="Lebu", destination="Dire Dawa")
        assert result == date(2025, 10, 1)


class TestClassifyDate:
    def test_past_date(self):
        result = classify_date(date(2020, 1, 1), today=date(2025, 9, 1))
        assert result == "PAST"

    def test_valid_operating(self):
        result = classify_date(date(2025, 9, 3), origin="Lebu", destination="Dire Dawa", today=date(2025, 9, 1))
        assert result == "VALID_OPERATING"

    def test_not_operating_even_day_eastbound(self):
        result = classify_date(date(2025, 9, 4), origin="Lebu", destination="Dire Dawa", today=date(2025, 9, 1))
        assert result == "NOT_OPERATING"

    def test_31st_is_not_operating(self):
        result = classify_date(date(2025, 8, 31), origin="Lebu", destination="Dire Dawa", today=date(2025, 8, 1))
        assert result == "NOT_OPERATING"


class TestGetNextNDates:
    def test_returns_n_dates_eastbound(self):
        dates = get_next_n_operating_dates(date(2025, 9, 1), n=5, origin="Lebu", destination="Dire Dawa")
        assert len(dates) == 5
        assert all(is_edr_operating_day(d, origin="Lebu", destination="Dire Dawa") for d in dates)
        assert all(d.day % 2 == 1 for d in dates)

    def test_returns_n_dates_westbound(self):
        dates = get_next_n_operating_dates(date(2025, 9, 2), n=5, origin="Dire Dawa", destination="Lebu")
        assert len(dates) == 5
        assert all(is_edr_operating_day(d, origin="Dire Dawa", destination="Lebu") for d in dates)
        assert all(d.day % 2 == 0 for d in dates)

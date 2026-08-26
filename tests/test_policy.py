"""
tests/test_policy.py — Unit tests for booking policy engine.
Tests all three policies deterministically.
"""

import pytest
from datetime import date
from edr_agent.config import BookingMode, SeatType
from edr_agent.policy import PolicyEngine, PolicyAction
from edr_agent.state import AgentState, WorkflowStep
from edr_agent.vlm.schemas import DateAvailability, ScheduleAvailability


def make_state(
    mode: BookingMode = BookingMode.NORMAL,
    seat: SeatType = SeatType.REGULAR_SEAT,
    date_avail: DateAvailability = DateAvailability.AVAILABLE,
    schedule_avail: ScheduleAvailability = ScheduleAvailability.SCHEDULE_FOUND,
    seat_avail: dict | None = None,
    seat_ranking: list | None = None,
) -> AgentState:
    s = AgentState()
    s.origin = "Lebu"
    s.destination = "Dire Dawa"
    s.target_date = date(2025, 9, 1)
    s.current_date = date(2025, 9, 1)
    s.booking_mode = mode
    s.preferred_seat = seat
    s.seat_ranking = seat_ranking or [seat]
    s.date_availability = date_avail
    s.schedule_availability = schedule_avail
    s.seat_availability = seat_avail or {}
    return s


engine = PolicyEngine()


class TestNormalPolicy:
    def test_seat_available_proceeds(self):
        state = make_state(
            mode=BookingMode.NORMAL,
            seat=SeatType.REGULAR_SEAT,
            seat_avail={"Regular Seat": True}
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.PROCEED_TO_BOOKING

    def test_seat_unavailable_stops(self):
        state = make_state(
            mode=BookingMode.NORMAL,
            seat=SeatType.REGULAR_SEAT,
            seat_avail={"Regular Seat": False, "Economy Bed Lower": True}
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.NOTIFY_AND_STOP

    def test_date_not_open_waits(self):
        state = make_state(
            mode=BookingMode.NORMAL,
            date_avail=DateAvailability.NOT_YET_OPEN
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.WAIT_AND_RETRY

    def test_invalid_date_stops(self):
        state = make_state(
            mode=BookingMode.NORMAL,
            date_avail=DateAvailability.INVALID
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.NOTIFY_AND_STOP


class TestSeatFirstPolicy:
    def test_seat_unavailable_advances_date(self):
        state = make_state(
            mode=BookingMode.SEAT_FIRST,
            seat=SeatType.ECONOMY_BED_LOWER,
            seat_avail={"Economy Bed Lower": False}
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.ADVANCE_DATE
        assert decision.new_date is not None
        assert decision.new_date > state.current_date

    def test_no_schedule_advances_date(self):
        state = make_state(
            mode=BookingMode.SEAT_FIRST,
            schedule_avail=ScheduleAvailability.NO_RESULTS,
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.ADVANCE_DATE

    def test_seat_available_proceeds(self):
        state = make_state(
            mode=BookingMode.SEAT_FIRST,
            seat=SeatType.VIP_BED_LOWER,
            seat_avail={"VIP Bed Lower": True}
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.PROCEED_TO_BOOKING


class TestDateFirstPolicy:
    def test_falls_back_to_ranked_seat(self):
        ranking = [SeatType.REGULAR_SEAT, SeatType.ECONOMY_BED_LOWER, SeatType.VIP_BED_LOWER]
        state = make_state(
            mode=BookingMode.DATE_FIRST,
            seat=SeatType.REGULAR_SEAT,
            seat_ranking=ranking,
            seat_avail={"Regular Seat": False, "Economy Bed Lower": True, "VIP Bed Lower": True}
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.PROCEED_TO_BOOKING
        assert decision.new_seat == SeatType.ECONOMY_BED_LOWER

    def test_all_ranked_unavailable_stops(self):
        ranking = [SeatType.REGULAR_SEAT, SeatType.ECONOMY_BED_LOWER]
        state = make_state(
            mode=BookingMode.DATE_FIRST,
            seat=SeatType.REGULAR_SEAT,
            seat_ranking=ranking,
            seat_avail={"Regular Seat": False, "Economy Bed Lower": False}
        )
        decision = engine.evaluate(state)
        assert decision.action == PolicyAction.NOTIFY_AND_STOP


"""
agent.py — State Machine + Booking Orchestrator

Hybrid design:
  - Playwright owns all navigation, form-filling, and post-search booking steps.
  - VLM is invoked ONCE per booking attempt, only on the SEARCH_RESULTS page,
    to read schedule availability and seat counts (unstructured visual data).

This is Module 3 (State Tracking) combined with the booking workflow.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

from edr_agent.browser.controller import BrowserController
from edr_agent.config import UserConfig, settings
from edr_agent.policy import PolicyAction, PolicyEngine
from edr_agent.state import AgentState, WorkflowStep
from edr_agent.vlm.client import VLMClient
from edr_agent.vlm.perception import PerceptionModule
from edr_agent.vlm.schemas import (
    CycleEvent,
    DateAvailability,
    PerceptionResult,
    PredictedAction,
    ScheduleAvailability,
)

logger = logging.getLogger(__name__)

MAX_RETRIES_PER_CYCLE = 3


class BookingAgent:
    """
    The main booking agent.

    Architecture:
    - BrowserController: manages Playwright browser
    - PerceptionModule: screenshot → PerceptionResult (VLM, SEARCH_RESULTS only)
    - PolicyEngine: deterministic policy decisions
    - AgentState: the ground truth state
    """

    def __init__(self) -> None:
        self.vlm = VLMClient()
        self.browser = BrowserController()
        self.perception = PerceptionModule(self.vlm)
        self.policy = PolicyEngine()
        self.state: Optional[AgentState] = None
        self._event_callback: Optional[Callable[[CycleEvent], None]] = None
        self._stop_requested = False
        # asyncio.Event used to pause the agent at the Fayda verification handoff.
        # The /api/resume endpoint calls resume_from_handoff() to set this event.
        self._resume_event: Optional[asyncio.Event] = None

    def set_event_callback(self, callback: Callable[[CycleEvent], None]) -> None:
        """Register a callback to receive CycleEvents (used by UI WebSocket)."""
        self._event_callback = callback

    def request_stop(self) -> None:
        """Request the agent to stop after the current cycle."""
        self._stop_requested = True
        logger.info("[AGENT] Stop requested")

    def resume_from_handoff(self) -> None:
        """
        Called by the /api/resume endpoint when the human has completed the
        Fayda verification step. Unblocks the waiting coroutine.
        """
        if self._resume_event and not self._resume_event.is_set():
            self._resume_event.set()
            logger.info("[AGENT] Resume signal received — continuing from Fayda handoff")
        else:
            logger.warning("[AGENT] resume_from_handoff() called but no active pause event")

    # ---------------------------------------------------------------------------
    # Main entry points
    # ---------------------------------------------------------------------------

    async def run_once(self, config: UserConfig) -> AgentState:
        """
        Run a single booking attempt (no monitoring loop).
        Used for POC demo and testing.
        """
        self.state = AgentState.from_config(config)
        self._stop_requested = False

        await self.browser.launch()
        try:
            await self._booking_workflow()
        finally:
            if self.state and self.state.current_step == WorkflowStep.PAYMENT_HANDOFF:
                logger.info("[AGENT] Booking successful. Keeping browser open indefinitely for human to complete payment.")
                await asyncio.Event().wait()
            else:
                await self.browser.close()

        return self.state

    async def run_with_monitoring(self, config: UserConfig) -> AgentState:
        """
        Run the full monitoring loop.
        Repeats booking attempts at configured intervals until:
        - Booking succeeds
        - Human handoff is triggered
        - Agent is stopped manually
        - Max cycles reached
        """
        self.state = AgentState.from_config(config)
        self.state.monitoring_active = True
        self._stop_requested = False

        await self.browser.launch()
        try:
            while not self._should_stop():
                logger.info(f"[AGENT] === Monitoring cycle {self.state.monitoring_cycle + 1} ===")
                self.state.reset_for_next_monitoring_cycle()
                self.state.transition(WorkflowStep.MONITORING)

                self._emit_event(log_message=f"Starting monitoring cycle {self.state.monitoring_cycle}")

                try:
                    await self._booking_workflow()
                except Exception as e:
                    logger.error(f"[AGENT] Booking workflow error: {e}")
                    self.state.record_failure(str(e))

                if self.state.terminated or self.state.success or self.state.waiting_for_human:
                    break

                if not self._should_stop():
                    await self._wait_for_next_cycle()

        finally:
            if self.state and self.state.current_step == WorkflowStep.PAYMENT_HANDOFF:
                logger.info("[AGENT] Booking successful. Keeping browser open indefinitely for human to complete payment.")
                await asyncio.Event().wait()
            else:
                await self.browser.close()

        return self.state

    # ---------------------------------------------------------------------------
    # Core booking workflow
    # ---------------------------------------------------------------------------

    async def _booking_workflow(self) -> None:
        """
        Execute the booking workflow state machine.

        Hybrid design — Playwright handles all navigation/form-filling;
        VLM is only used to read unstructured data on SEARCH_RESULTS.

        States:
        IDLE → NAVIGATING → FILLING_FORM → SEARCHING → APPLYING_POLICY
             → BOOKING → WAITING_FOR_HUMAN | SUCCESS | FAILED

        Date validation is handled deterministically inside _fill_booking_form:
        if the calendar day is unclickable, it sets date_availability and
        terminates the cycle — no VLM perception call is needed.
        """
        state = self.state

        # STEP 1: Navigate to booking site
        state.transition(WorkflowStep.NAVIGATING)
        await self._navigate_with_retry(settings.edr_base_url)
        if state.terminated:
            return

        # STEP 2: Fill the booking form (Playwright — includes date validation)
        state.transition(WorkflowStep.FILLING_FORM)
        await self._fill_booking_form()
        if state.terminated:
            return

        # STEP 3: Single VLM pass — read schedule + seats from SEARCH_RESULTS
        state.transition(WorkflowStep.SEARCHING)
        await self._perceive_search_results()
        if state.terminated:
            return

        # STEP 4: Apply seat policy
        state.transition(WorkflowStep.APPLYING_POLICY)
        await self._apply_seat_policy()

    # ---------------------------------------------------------------------------
    # Workflow steps
    # ---------------------------------------------------------------------------

    async def _navigate_with_retry(self, url: str) -> None:
        """Navigate to a URL with retry logic."""
        for attempt in range(MAX_RETRIES_PER_CYCLE):
            try:
                await self.browser.navigate(url)
                screenshot, path = await self.browser.screenshot("navigate")
                self.state.last_screenshot_path = path
                self._emit_event(
                    log_message=f"Navigated to {url}",
                    screenshot_path=path,
                )
                return
            except Exception as e:
                logger.warning(f"[AGENT] Navigation attempt {attempt+1} failed: {e}")
                self.state.record_failure(str(e))
                if attempt < MAX_RETRIES_PER_CYCLE - 1:
                    await asyncio.sleep(3)

        self.state.transition(WorkflowStep.FAILED)
        self.state.terminated = True
        self._emit_event(log_message="Navigation failed after all retries", is_error=True)

    async def _fill_booking_form(self) -> None:
        """
        Fill the EDR booking form deterministically using EDRFormHelper.

        Playwright handles all station autocomplete, date selection, and form
        submission. No VLM is involved here.
        If the calendar day is unclickable, EDRFormHelper returns date_blocked=True
        and this method terminates the cycle cleanly.
        """
        from edr_agent.browser.edr_forms import EDRFormHelper
        state = self.state

        form = EDRFormHelper(self.browser.page)

        # Ensure we're on the home page before filling
        if not await form.is_on_home_page():
            logger.info("[AGENT] Not on home page, navigating...")
            await self._navigate_with_retry(settings.edr_base_url)
            if state.terminated:
                return

        # Deterministically fill origin, destination, date, and submit
        results = await form.fill_booking_form(
            origin=state.origin,
            destination=state.destination,
            travel_date=state.current_date,
        )
        logger.info(f"[AGENT] Form fill results: {results}")

        # --- Date-blocked guard ---
        # If the calendar day was unclickable (not-yet-open), EDRFormHelper signals
        # this via date_blocked. We record the state and end the cycle — no VLM needed.
        if results.get("date_blocked"):
            target = state.current_date.isoformat() if state.current_date else "requested date"
            msg = (
                f"⏳ Bookings for {target} are not yet open in the booking calendar. "
                f"The agent will stop this cycle and check again at the next monitoring interval."
            )
            logger.info(f"[AGENT] {msg}")
            state.date_availability = DateAvailability.NOT_YET_OPEN
            self._emit_event(log_message=msg, is_error=False)
            state.terminated = True
            return

    async def _perceive_search_results(self) -> None:
        """
        Single VLM perception pass on the SEARCH_RESULTS page.

        Reads both schedule availability and seat counts in one API call.
        Previously this was split across _search_schedules() + _inspect_seats(),
        which took two screenshots and made two identical API calls on the same page.
        """
        # Wait for the search results to actually render.
        # The EDR site is an SPA — wait_for_load_state("networkidle") returns
        # immediately after click_search()'s 2s sleep, before results are painted.
        # Instead, wait for either: a 'Select' button (results loaded) or a
        # no-results/error indicator — whichever appears first.
        try:
            await self.browser.page.wait_for_selector(
                "button:has-text('Select'), "
                ":has-text('No trains found'), "
                ":has-text('No results'), "
                "[data-testid='select-schedule']",
                timeout=15000,
            )
            logger.info("[AGENT] Search results page ready")
        except Exception as e:
            logger.warning(f"[AGENT] Results page ready-check timed out — proceeding: {e}")

        screenshot, path = await self.browser.screenshot("search_results")
        await asyncio.sleep(3.0)  # Rate-limit pacing before VLM call

        result = self.perception.perceive(
            screenshot,
            origin=self.state.origin,
            destination=self.state.destination,
            target_date=self.state.current_date,
            preferred_seat=self.state.preferred_seat.value,
        )
        self.state.last_perception = result
        self.state.last_screenshot_path = path

        # Update schedule availability
        if result.schedule_state != ScheduleAvailability.UNKNOWN:
            self.state.schedule_availability = result.schedule_state

        # Update seat availability
        for seat_info in result.available_seats:
            self.state.seat_availability[seat_info.seat_type] = seat_info.available

        if result.preferred_seat_available is not None:
            self.state.seat_availability[self.state.preferred_seat.value] = (
                result.preferred_seat_available
            )

        self._emit_event(
            perception=result,
            screenshot_path=path,
            log_message=(
                f"Schedule: {self.state.schedule_availability.value} | "
                f"Seats: {self.state.seat_availability}"
            ),
        )

        # If no schedule found, apply policy immediately
        if self.state.schedule_availability in (
            ScheduleAvailability.NO_RESULTS,
            ScheduleAvailability.SCHEDULE_FULL,
        ):
            decision = self.policy.evaluate(self.state)
            await self._apply_policy_decision(decision)

    async def _apply_seat_policy(self) -> None:
        """Apply the booking policy based on seat availability."""
        decision = self.policy.evaluate(self.state)
        logger.info(f"[AGENT] Seat policy: {decision.action.value} — {decision.reason}")
        await self._apply_policy_decision(decision)

    async def _apply_policy_decision(self, decision) -> None:
        """Execute a policy decision."""
        from edr_agent.policy import PolicyAction

        if decision.action == PolicyAction.PROCEED_TO_BOOKING:
            if decision.new_seat:
                self.state.preferred_seat = decision.new_seat
            self.state.transition(WorkflowStep.BOOKING)
            self._emit_event(log_message=decision.notification_message)
            await self._proceed_to_booking()

        elif decision.action == PolicyAction.ADVANCE_DATE and decision.new_date:
            logger.info(f"[AGENT] Advancing date: {self.state.current_date} → {decision.new_date}")
            self.state.current_date = decision.new_date
            self.state.date_availability = DateAvailability.UNKNOWN
            self.state.schedule_availability = ScheduleAvailability.UNKNOWN
            self.state.seat_availability = {}
            self._emit_event(log_message=decision.notification_message)
            # Re-navigate with new date
            self.state.transition(WorkflowStep.FILLING_FORM)
            await self._fill_booking_form()
            # _fill_booking_form handles the date_blocked case internally;
            # if it set terminated=True, we propagate that by simply returning.
            if self.state.terminated:
                return

        elif decision.action == PolicyAction.TRY_NEXT_SEAT:
            self.state.current_seat_rank_index += 1
            self._emit_event(log_message=decision.notification_message)
            await self._inspect_seats()

        elif decision.action == PolicyAction.NOTIFY_AND_STOP:
            self._emit_event(log_message=decision.notification_message, is_error=True)
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            self.state.monitoring_active = False

        elif decision.action == PolicyAction.WAIT_AND_RETRY:
            self._emit_event(log_message=decision.notification_message)
            self.state.terminated = True

    async def _proceed_to_booking(self) -> None:
        """
        Phase 2 deterministic booking pipeline.

        Executes the post-search booking steps using hardcoded Playwright actions:
          A. Select schedule + coach modal
          B. Continue as guest (auth-check bypass)
          C. Click Verify with Fayda → HUMAN HANDOFF 1 (pause)
          D. (post-resume) Continue to seat selection
          E. Auto-assign seats + Continue
          F. Confirm and pay → HUMAN HANDOFF 2 (final, stop loop)

        The VLM is NOT used for steps B–F. All navigation is deterministic.
        """
        from edr_agent.browser.edr_forms import EDRFormHelper
        form = EDRFormHelper(self.browser.page)

        await self._step_select_schedule_and_coach(form)
        if self.state.terminated or self._stop_requested:
            return

        await self._step_auth_check(form)
        if self.state.terminated or self._stop_requested:
            return

        await self._step_fayda_verification(form)
        if self.state.terminated or self._stop_requested:
            return

        # --- PAUSE: Wait for human to complete Fayda ---
        await self._wait_for_resume()
        if self.state.terminated or self._stop_requested:
            return

        await self._step_continue_after_fayda(form)
        if self.state.terminated or self._stop_requested:
            return

        await self._step_seat_map(form)
        if self.state.terminated or self._stop_requested:
            return

        await self._step_review_and_handoff(form)

    # ---------------------------------------------------------------------------
    # Phase 2 — individual deterministic booking step methods
    # ---------------------------------------------------------------------------

    async def _step_select_schedule_and_coach(self, form) -> None:
        """
        Step A: On the search results page, the VLM has already confirmed a schedule
        exists and the preferred seat is available. Playwright now clicks 'Select'
        and then picks the correct coach class in the modal.
        """
        self.state.transition(WorkflowStep.SELECTING_SCHEDULE)
        screenshot, path = await self.browser.screenshot("selecting_schedule")
        self.state.last_screenshot_path = path

        seat_type = self.state.preferred_seat.value
        self._emit_event(
            screenshot_path=path,
            log_message=f"[Step A] Selecting schedule and coach: '{seat_type}'",
        )

        success = await form.select_schedule_and_coach(seat_type)
        if not success:
            self.state.record_failure("Could not select schedule or coach")
            self._emit_event(
                log_message="[Step A] 🛑 FATAL: Failed to select schedule/coach — aborting workflow",
                is_error=True,
            )
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return
        else:
            self._emit_event(log_message=f"[Step A] ✅ Schedule selected, coach '{seat_type}' chosen")

        await asyncio.sleep(1.5)

    async def _step_auth_check(self, form) -> None:
        """
        Step B: Wait for /booking/auth-check, then click 'Continue as guest'.
        """
        self.state.transition(WorkflowStep.AUTH_CHECK)
        self._emit_event(log_message="[Step B] Waiting for auth-check page...")

        nav_success = await form.wait_for_page_url_contains("/booking/auth-check", timeout_ms=15000)
        if not nav_success:
            self.state.record_failure("Timeout waiting for /booking/auth-check")
            self._emit_event(log_message="[Step B] 🛑 FATAL: Timeout waiting for auth-check page", is_error=True)
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return

        screenshot, path = await self.browser.screenshot("auth_check")
        self.state.last_screenshot_path = path
        self._emit_event(screenshot_path=path, log_message="[Step B] Clicking 'Continue as guest'")

        success = await form.click_continue_as_guest()
        if not success:
            self.state.record_failure("Could not click 'Continue as guest'")
            self._emit_event(
                log_message="[Step B] 🛑 FATAL: 'Continue as guest' not found — aborting workflow",
                is_error=True,
            )
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return
        else:
            self._emit_event(log_message="[Step B] ✅ Bypassed auth-check as guest")

    async def _step_fayda_verification(self, form) -> None:
        """
        Step C: On /booking/passengers, click 'Verify with Fayda' then immediately
        transition to WAITING_FOR_HUMAN.
        """
        self.state.transition(WorkflowStep.FAYDA_VERIFICATION)
        self._emit_event(log_message="[Step C] Waiting for passengers page...")

        nav_success = await form.wait_for_page_url_contains("/booking/passengers", timeout_ms=15000)
        if not nav_success:
            self.state.record_failure("Timeout waiting for /booking/passengers")
            self._emit_event(log_message="[Step C] 🛑 FATAL: Timeout waiting for passengers page", is_error=True)
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return

        screenshot, path = await self.browser.screenshot("passengers")
        self.state.last_screenshot_path = path
        self._emit_event(screenshot_path=path, log_message="[Step C] Clicking 'Verify with Fayda'")

        success = await form.click_verify_with_fayda()
        if not success:
            self.state.record_failure("Could not click 'Verify with Fayda'")
            self._emit_event(
                log_message="[Step C] 🛑 FATAL: 'Verify with Fayda' not found — aborting workflow",
                is_error=True,
            )
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return

        # Transition to human handoff — FAYDA stage
        self.state.transition(WorkflowStep.WAITING_FOR_HUMAN)
        self.state.waiting_for_human = True
        self.state.handoff_stage = "FAYDA"
        self.state.human_handoff_reason = "Fayda biometric verification required"

        handoff_msg = (
            "🤝 HUMAN HANDOFF 1: Please complete Fayda verification manually. "
            "Resume the agent once details are locked."
        )
        self._emit_event(log_message=handoff_msg)
        logger.info(f"[AGENT] {handoff_msg}")

    async def _wait_for_resume(self) -> None:
        """
        Pause execution until resume_from_handoff() is called (by /api/resume).
        Initialises a fresh asyncio.Event so it resets each booking run.
        """
        self._resume_event = asyncio.Event()
        logger.info("[AGENT] Pausing — waiting for human to complete Fayda and call /api/resume")

        # Wait indefinitely (or until stop is requested via polling)
        while not self._resume_event.is_set():
            if self._stop_requested:
                logger.info("[AGENT] Stop requested during Fayda wait — aborting")
                self.state.terminated = True
                return
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._resume_event.wait()), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue  # Loop and check stop flag

        # Clear handoff state
        self.state.waiting_for_human = False
        self.state.handoff_stage = None
        self._resume_event = None
        logger.info("[AGENT] Resumed — continuing booking workflow")

    async def _step_continue_after_fayda(self, form) -> None:
        """
        Step D: After human completes Fayda, click 'Continue to seat selection'.
        """
        self.state.transition(WorkflowStep.FAYDA_VERIFICATION)
        self._emit_event(log_message="[Step D] Clicking 'Continue to seat selection'")

        screenshot, path = await self.browser.screenshot("post_fayda")
        self.state.last_screenshot_path = path
        self._emit_event(screenshot_path=path)

        success = await form.click_continue_to_seat_selection()
        if not success:
            self.state.record_failure("Could not click 'Continue to seat selection'")
            self._emit_event(
                log_message="[Step D] 🛑 FATAL: 'Continue to seat selection' not found — aborting workflow",
                is_error=True,
            )
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return
        else:
            self._emit_event(log_message="[Step D] ✅ Navigating to seat map")

    async def _step_seat_map(self, form) -> None:
        """
        Step E: On /booking/seats, auto-assign seats then continue.
        """
        self.state.transition(WorkflowStep.SEAT_MAP)
        self._emit_event(log_message="[Step E] Waiting for seat map page...")

        nav_success = await form.wait_for_page_url_contains("/booking/seats", timeout_ms=15000)
        if not nav_success:
            self.state.record_failure("Timeout waiting for /booking/seats")
            self._emit_event(log_message="[Step E] 🛑 FATAL: Timeout waiting for seat map page", is_error=True)
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return

        screenshot, path = await self.browser.screenshot("seat_map")
        self.state.last_screenshot_path = path
        self._emit_event(screenshot_path=path, log_message="[Step E] Auto-assigning seats...")

        success = await form.auto_assign_and_continue_seats()
        if not success:
            # Handle inventory mismatch gracefully without full agent crash
            self.state.record_failure("Seat assignment failed (likely 0 seats available for coach type)")
            self._emit_event(
                log_message="Seat assignment failed: 0 seats available for this specific coach type. Aborting current booking attempt.",
                is_error=True,
            )
            # Reset agent state gracefully to allow future retries
            self.state.transition(WorkflowStep.IDLE)
            self.state.terminated = True
            return
        else:
            self._emit_event(log_message="[Step E] ✅ Seats assigned, continuing to review")

    async def _step_review_and_handoff(self, form) -> None:
        """
        Step F: On /booking/review, click 'Confirm and pay' then trigger
        the final PAYMENT HANDOFF — the agent's work is complete.
        """
        self.state.transition(WorkflowStep.REVIEWING)
        self._emit_event(log_message="[Step F] Waiting for review page...")

        nav_success = await form.wait_for_page_url_contains("/booking/review", timeout_ms=15000)
        if not nav_success:
            self.state.record_failure("Timeout waiting for /booking/review")
            self._emit_event(log_message="[Step F] 🛑 FATAL: Timeout waiting for review page", is_error=True)
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return

        screenshot, path = await self.browser.screenshot("review")
        self.state.last_screenshot_path = path
        self._emit_event(screenshot_path=path, log_message="[Step F] Clicking 'Confirm and pay'")

        success = await form.click_confirm_and_pay()
        if not success:
            self.state.record_failure("Could not click 'Confirm and pay'")
            self._emit_event(
                log_message="[Step F] 🛑 FATAL: 'Confirm and pay' not found — aborting workflow",
                is_error=True,
            )
            self.state.transition(WorkflowStep.FAILED)
            self.state.terminated = True
            return

        # Transition to final handoff — payment stage
        self.state.transition(WorkflowStep.PAYMENT_HANDOFF)
        self.state.waiting_for_human = True
        self.state.handoff_stage = "PAYMENT"
        self.state.success = True
        self.state.terminated = True

        final_msg = (
            "🎉 HUMAN HANDOFF 2: Agent has successfully completed the booking workflow. "
            "Handing over to human for final payment."
        )
        self._emit_event(log_message=final_msg)
        logger.info(f"[AGENT] {final_msg}")


    # ---------------------------------------------------------------------------
    # Monitoring helpers
    # ---------------------------------------------------------------------------

    async def _wait_for_next_cycle(self) -> None:
        """Wait for the configured monitoring interval."""
        interval_seconds = self.state.monitoring_interval_minutes * 60
        next_check = datetime.now()

        from datetime import timedelta
        self.state.next_check_time = datetime.now() + timedelta(seconds=interval_seconds)

        self.state.transition(WorkflowStep.WAITING_INTERVAL)
        self._emit_event(
            log_message=(
                f"Waiting {self.state.monitoring_interval_minutes} minutes "
                f"until next check at {self.state.next_check_time.strftime('%H:%M:%S')}"
            )
        )

        # Wait in small increments so stop requests are responsive
        elapsed = 0
        while elapsed < interval_seconds:
            if self._stop_requested:
                break
            await asyncio.sleep(5)
            elapsed += 5

    def _should_stop(self) -> bool:
        """Check if the monitoring loop should stop."""
        state = self.state
        if self._stop_requested:
            return True
        if state.terminated or state.success or state.waiting_for_human:
            return True
        if state.monitoring_cycle >= state.max_monitoring_cycles:
            logger.info("[AGENT] Max monitoring cycles reached")
            return True
        return False

    # ---------------------------------------------------------------------------
    # Event emission
    # ---------------------------------------------------------------------------

    def _emit_event(
        self,
        perception: Optional[PerceptionResult] = None,
        action: Optional[PredictedAction] = None,
        screenshot_path: Optional[str] = None,
        log_message: str = "",
        is_error: bool = False,
    ) -> None:
        """Emit a CycleEvent to the registered callback (UI WebSocket)."""
        if not self._event_callback:
            return

        event = CycleEvent(
            cycle_number=self.state.cycle_count,
            timestamp=datetime.now().isoformat(),
            workflow_step=self.state.workflow_step.value,
            perception=perception,
            action=action,
            state_summary=self.state.to_summary_dict(),
            screenshot_path=screenshot_path,
            log_message=log_message,
            is_error=is_error,
        )

        try:
            self._event_callback(event)
        except Exception as e:
            logger.warning(f"[AGENT] Event callback error: {e}")

"""
browser/edr_forms.py — EDR-specific form filling helpers.

The EDR website uses custom React autocomplete dropdowns that require
specific interaction patterns:
1. Click the input field to focus it
2. Clear and type the station name
3. Wait for the dropdown to appear
4. Click the correct dropdown option

This module encapsulates these patterns so the VLM doesn't need to worry
about EDR-specific UI quirks — it just declares "select station X".
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Optional

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)


class EDRFormHelper:
    """
    Handles EDR-website-specific form interactions.
    These are deterministic implementations that the VLM delegates to.
    """

    def __init__(self, page: Page) -> None:
        self._page = page

    async def fill_origin(self, station_name: str) -> bool:
        """Fill the departure station field using EDR's autocomplete."""
        return await self._fill_station_field(
            input_selector='input[placeholder="Departure station"]',
            station_name=station_name,
            label="origin",
        )

    async def fill_destination(self, station_name: str) -> bool:
        """Fill the destination station field using EDR's autocomplete."""
        return await self._fill_station_field(
            input_selector='input[placeholder="Destination station"]',
            station_name=station_name,
            label="destination",
        )

    async def open_calendar_dialog(self) -> bool:
        """Explicitly click the DATE input field to invoke the calendar pop-up card."""
        date_selectors = [
            'button:has(svg.lucide-calendar)',
            'div:has-text("DATE") + button',
            'button:has-text("Departure")',
            'div:has-text("DATE") button',
        ]
        for sel in date_selectors:
            try:
                el = self._page.locator(sel).first
                if await el.is_visible():
                    await el.click(timeout=3000)
                    await asyncio.sleep(0.5)
                    logger.info(f"[EDR-FORM] Opened calendar dialog using selector: {sel}")
                    return True
            except Exception:
                continue
        logger.warning("[EDR-FORM] Could not click DATE field to open calendar dialog")
        return False

    async def fill_date(self, travel_date: date) -> bool:
        """
        Click the DATE field to open the calendar dialog pop-up, navigate to the target month,
        then click the target day. Manual text typing into date fields is strictly avoided.
        """
        try:
            # Step 1: Open calendar pop-up dialog
            opened = await self.open_calendar_dialog()
            if not opened:
                return False

            # Wait for calendar modal title "Select Date" (or just wait a bit)
            try:
                await self._page.wait_for_selector('text="Select Date"', timeout=3000)
            except Exception:
                await asyncio.sleep(0.5)

            # Step 2: Navigate to the correct month
            target_month = travel_date.month
            target_year = travel_date.year

            for _ in range(12):  # Max 12 clicks to prevent infinite loops
                try:
                    # Look for the calendar modal or body
                    header_loc = self._page.locator('div[role="dialog"], .rdp, .react-datepicker, body').first
                    
                    header_text = await header_loc.inner_text()
                    header_text = header_text.strip().lower()
                    
                    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
                    current_month = None
                    for i, m in enumerate(months):
                        if m in header_text:
                            current_month = i + 1
                            break
                            
                    current_year = None
                    for y in range(2020, 2040):
                        if str(y) in header_text:
                            current_year = y
                            break
                            
                    if current_year is None:
                        # Fallback to current year if not found in text
                        current_year = date.today().year
                            
                    if current_month is None:
                        logger.warning(f"[EDR-FORM] Could not parse month from calendar text")
                        break
                        
                    if current_year == target_year and current_month == target_month:
                        logger.info(f"[EDR-FORM] Reached target month/year: {current_month}/{current_year}")
                        break
                        
                    # Navigate if needed
                    if target_year > current_year or (target_year == current_year and target_month > current_month):
                        next_btn = self._page.locator('button:has(svg.lucide-chevron-right), button[aria-label*="next" i], button[aria-label*="Next" i], .rdp-nav_button_next').first
                        await next_btn.click(timeout=3000)
                        await asyncio.sleep(0.8)
                    else:
                        prev_btn = self._page.locator('button:has(svg.lucide-chevron-left), button[aria-label*="prev" i], button[aria-label*="Previous" i], .rdp-nav_button_previous').first
                        await prev_btn.click(timeout=3000)
                        await asyncio.sleep(0.8)
                        
                except Exception as e:
                    logger.warning(f"[EDR-FORM] Month navigation error: {e}")
                    break

            # Step 3: Click the target day number inside calendar pop-up
            day = travel_date.day
            day_selectors = [
                # Precise selector ignoring days from outside the month
                f'button.rdp-day:not(.rdp-day_outside):text-is("{day}"):not([disabled])',
                f'button:not(.rdp-day_outside):text-is("{day}"):not([disabled])',
                f'div[role="dialog"] button:text-is("{day}")',
                f'button:text-is("{day}")',
            ]

            for sel in day_selectors:
                try:
                    el = self._page.locator(sel).first
                    if await el.is_visible():
                        await el.click(timeout=3000)
                        await asyncio.sleep(0.5)
                        logger.info(f"[EDR-FORM] Date {travel_date} clicked in pop-up calendar card")
                        return True
                except Exception:
                    continue

            # All day selectors failed — the date is likely disabled/not-yet-open.
            # CRITICAL: close the calendar modal so its backdrop doesn't block the page.
            logger.warning(f"[EDR-FORM] Could not click day {day} in calendar — date may be disabled. Closing modal.")
            await self._close_calendar_modal()
            return False

        except Exception as e:
            logger.error(f"[EDR-FORM] Date selection error: {e}")
            await self._close_calendar_modal()
            return False

    async def _close_calendar_modal(self) -> None:
        """
        Attempt to close an open calendar/date-picker modal so its backdrop
        (z-index: 99) does not intercept subsequent clicks.

        Strategy (ordered by reliability):
          1. Press Escape — works for most modal/dialog implementations.
          2. Click a visible 'X' or 'Close' button inside the dialog.
          3. Click outside the dialog (top-left corner) as a last resort.
        """
        # 1. Escape key
        try:
            await self._page.keyboard.press("Escape")
            await asyncio.sleep(0.4)
            logger.info("[EDR-FORM] Calendar modal closed via Escape")
            return
        except Exception:
            pass

        # 2. Explicit close button inside dialog
        close_selectors = [
            'div[role="dialog"] button[aria-label*="close" i]',
            'div[role="dialog"] button[aria-label*="Close" i]',
            'div[role="dialog"] button:has(svg.lucide-x)',
            'div[role="dialog"] button.close',
        ]
        for sel in close_selectors:
            try:
                el = self._page.locator(sel).first
                if await el.is_visible(timeout=1500):
                    await el.click(timeout=2000)
                    await asyncio.sleep(0.4)
                    logger.info(f"[EDR-FORM] Calendar modal closed via button: {sel}")
                    return
            except Exception:
                continue

        # 3. Click outside the dialog (top-left safe zone)
        try:
            await self._page.mouse.click(10, 10)
            await asyncio.sleep(0.4)
            logger.info("[EDR-FORM] Calendar modal dismissed by clicking outside")
        except Exception as e:
            logger.warning(f"[EDR-FORM] Could not close calendar modal: {e}")

    async def click_search(self) -> bool:
        """Click the Search button to submit the booking form."""
        selectors = [
            'button[type="submit"]',
            'button:has-text("Search")',
            'button:has(svg.lucide-search)',
        ]
        for sel in selectors:
            try:
                await self._page.click(sel, timeout=5000)
                await asyncio.sleep(2.0)  # Wait for navigation/results
                logger.info("[EDR-FORM] Search button clicked")
                return True
            except Exception:
                continue
        logger.warning("[EDR-FORM] Could not find Search button")
        return False

    async def is_on_home_page(self) -> bool:
        """Check if we're on the EDR home page with the booking form."""
        try:
            await self._page.wait_for_selector(
                'input[placeholder="Departure station"]', timeout=5000
            )
            return True
        except Exception:
            return False

    async def fill_booking_form(self, origin: str, destination: str, travel_date: date) -> dict:
        """
        Fill the complete booking form deterministically.

        Returns a result dict with success status for each field.
        If the date step fails (day unclickable / disabled), the dict will contain
        ``date_blocked=True`` and the method returns immediately without attempting
        Nationality or Search — this avoids submitting a broken form and signals
        the caller to treat this as a NOT_YET_OPEN condition.
        """
        results = {}

        logger.info(f"[EDR-FORM] Filling form: {origin} → {destination} on {travel_date}")

        # Wait for form to be ready
        try:
            await self._page.wait_for_selector(
                'input[placeholder="Departure station"]', timeout=10000
            )
        except Exception:
            return {"error": "Booking form not found on page"}

        # Fill origin
        results["origin"] = await self.fill_origin(origin)
        await asyncio.sleep(0.5)

        # Fill destination
        results["destination"] = await self.fill_destination(destination)
        await asyncio.sleep(0.5)

        # Select travel date via pop-up calendar UI
        results["date"] = await self.fill_date(travel_date)
        await asyncio.sleep(0.5)

        # --- ABORT EARLY if the date could not be selected ---
        # This happens when the target day is disabled in the calendar (date not yet open).
        # The modal has already been closed inside fill_date(); we must NOT click
        # Nationality or Search with a missing date — that would submit a broken form
        # and likely show an error page that blocks all subsequent steps.
        if not results["date"]:
            logger.warning(
                f"[EDR-FORM] Date {travel_date} could not be selected — "
                "aborting form fill. Date is likely not yet open for booking."
            )
            results["date_blocked"] = True
            return results

        # Select nationality
        results["nationality"] = await self.fill_nationality("Ethiopian")
        await asyncio.sleep(0.5)

        # Click search
        results["search"] = await self.click_search()

        return results

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    async def fill_nationality(self, nationality: str = "Ethiopian") -> bool:
        """
        Click the Passengers / Nationality dropdown, select the specified nationality,
        and confirm the selection in the pop-up modal.
        """
        try:
            # 1. Open the Modal: Locate and click the "PASSENGERS" input field
            trigger = self._page.locator('button:has-text("Passenger")').first
            if not await trigger.is_visible():
                trigger = self._page.locator('div:has-text("Passenger")').last
            
            await trigger.click(timeout=3000)
            
            # 2. Wait for Modal to become visible
            await self._page.wait_for_selector('text="Passengers & Nationality"', timeout=3000)
            await asyncio.sleep(0.5) # Wait for animation to settle
            
            # 3. Select Nationality: Locate and click the specific pill button
            try:
                option = self._page.locator(f'button:has-text("{nationality}")').first
                if await option.is_visible():
                    await option.click(timeout=3000)
                else:
                    await self._page.locator(f'text="{nationality}"').first.click(timeout=3000)
            except Exception as e:
                logger.warning(f"[EDR-FORM] Could not click nationality '{nationality}': {e}")
                
            await asyncio.sleep(0.5)
            
            # 4. Click "Continue": Locate and click the confirmation button at the bottom
            continue_btn = self._page.locator('button:has-text("Continue")').first
            if await continue_btn.is_visible():
                await continue_btn.click(timeout=3000)
            else:
                logger.warning("[EDR-FORM] 'Continue' button not found in nationality modal")
                
            # 5. Wait for State Update: Ensure the modal closes
            await asyncio.sleep(1.0)
            
            logger.info(f"[EDR-FORM] Nationality modal completed for {nationality}")
            return True
            
        except Exception as e:
            logger.error(f"[EDR-FORM] Nationality modal interaction error: {e}")
            return False

    async def _fill_station_field(
        self,
        input_selector: str,
        station_name: str,
        label: str,
    ) -> bool:
        """
        Fill a station autocomplete field.
        
        EDR uses a custom React autocomplete:
        1. Click the input
        2. Type the station name (this triggers the dropdown)
        3. Wait for dropdown options to appear
        4. Click the matching option
        """
        try:
            # Click and clear the input
            input_el = self._page.locator(input_selector).first
            await input_el.click(timeout=5000)
            await asyncio.sleep(0.3)
            await input_el.fill("")
            await asyncio.sleep(0.2)

            # Type the station name (triggers autocomplete)
            await input_el.type(station_name, delay=80)
            await asyncio.sleep(1.0)  # Wait for dropdown to appear

            # Try various selectors for the dropdown option
            # CONFIRMED from DOM inspection:
            # EDR dropdown = BUTTON elements with class "w-full flex items-center gap-2"
            # containing text "Lebu" + "LEB" (concatenated as "LebuLEB")
            option_selectors = [
                # Primary: button with station name (EDR confirmed pattern)
                f'button.w-full:has-text("{station_name}")',
                f'button:has(span:has-text("{station_name}"))',
                # Fallback: any button below the input with the name
                f'button:has-text("{station_name}"):below(input)',
                # Generic patterns
                f'[role="option"]:has-text("{station_name}")',
                f'[role="listbox"] button:has-text("{station_name}")',
                f'li:has-text("{station_name}")',
                # Last resort: Playwright text locator
                f'text={station_name}',
            ]

            for opt_sel in option_selectors:
                try:
                    opt_el = self._page.locator(opt_sel).first
                    await opt_el.click(timeout=3000)
                    await asyncio.sleep(0.5)
                    logger.info(f"[EDR-FORM] {label} station set to: {station_name}")
                    return True
                except Exception:
                    continue

            # If no dropdown appeared, the value might already be set (input accepted it)
            current_value = await input_el.input_value()
            if station_name.lower() in current_value.lower():
                logger.info(f"[EDR-FORM] {label} station typed (no dropdown): {station_name}")
                return True

            logger.warning(f"[EDR-FORM] Could not select {label} station: {station_name}")
            return False

        except Exception as e:
            logger.error(f"[EDR-FORM] {label} station fill error: {e}")
            return False

    # ---------------------------------------------------------------------------
    # Phase 2 — Deterministic post-search booking steps
    # ---------------------------------------------------------------------------

    async def select_schedule_and_coach(self, seat_type: str) -> bool:
        """
        Step A: On the search-results page, click 'Select' on the first available
        schedule, then pick the requested coach/class in the 'Choose Your Coach' modal.

        Args:
            seat_type: Exact EDR seat-type string, e.g. 'Economy Bed Lower (Local)'.
                       The method tries an exact match first, then a contains match.

        Returns:
            True if both clicks succeeded, False otherwise.
        """
        # --- 1. Click the 'Select' button on the schedule card ---
        select_selectors = [
            "button:has-text(\"Select\")",
            "button.select-btn",
            "[data-testid='select-schedule']",
        ]
        selected = False
        for sel in select_selectors:
            try:
                el = self._page.locator(sel).first
                if await el.is_visible(timeout=5000):
                    await el.click(timeout=8000)
                    await asyncio.sleep(1.5)
                    logger.info(f"[EDR-FORM] Schedule 'Select' clicked via: {sel}")
                    selected = True
                    break
            except Exception:
                continue

        if not selected:
            logger.warning("[EDR-FORM] Could not click 'Select' on any schedule card")
            return False

        # --- 2. Wait for the 'Choose Your Coach' modal ---
        try:
            await self._page.wait_for_selector(
                "text='Choose Your Coach'", timeout=8000
            )
            await asyncio.sleep(0.8)
        except Exception:
            # Modal might use different heading text; continue anyway
            logger.warning("[EDR-FORM] 'Choose Your Coach' modal heading not found — attempting coach click anyway")

        # --- 3. Click the requested coach/class inside the modal ---
        # Specifically target the exact string used in the sub-tier rows
        target_text = f"{seat_type} (Local)"

        clicked = False
        try:
            # Use .last to ensure we hit the row inside the card body, not the card header
            await self._page.locator(f'text="{target_text}"').last.click(timeout=5000)
            clicked = True
            logger.info(f"[EDR-FORM] Coach '{seat_type}' selected via exact text.")
        except Exception as e:
            logger.error(f"[EDR-FORM] Coach selection exact text failed: {str(e)}")
            # Fallback: target the parent div of the text
            try:
                await self._page.locator(f'div:has-text("{target_text}")').last.click(timeout=5000)
                clicked = True
                logger.info(f"[EDR-FORM] Coach '{seat_type}' selected via parent div.")
            except Exception as fallback_e:
                logger.error(f"[EDR-FORM] Coach selection fallback failed: {str(fallback_e)}")

        if not clicked:
            logger.warning(f"[EDR-FORM] Could not select coach '{seat_type}' in modal")
            return False

        await asyncio.sleep(1.0)
        # Click modal submit button
        submit_sel = 'button:has-text("Continue to Passenger Details")'
        try:
            submit_btn = self._page.locator(submit_sel).first
            if await submit_btn.is_visible(timeout=3000):
                await submit_btn.click(timeout=5000)
                # Wait for modal to close (or page navigation)
                try:
                    await self._page.wait_for_selector(submit_sel, state="hidden", timeout=5000)
                except Exception:
                    pass
                await asyncio.sleep(1.0)
                logger.info(f"[EDR-FORM] Clicked modal submit: Continue to Passenger Details")
                return True
            else:
                logger.warning(f"[EDR-FORM] Modal submit button not found after selecting coach")
                return False
        except Exception as e:
            logger.error(f"[EDR-FORM] Failed to click modal submit button: {e}")
            return False

    async def click_continue_as_guest(self) -> bool:
        """
        Step B: On /booking/auth-check, click the green 'Continue as guest' button.
        """
        selectors = [
            "button:has-text(\"Continue as guest\")",
            "a:has-text(\"Continue as guest\")",
            "button:has-text(\"Guest\")",
        ]
        for sel in selectors:
            try:
                el = self._page.locator(sel).first
                if await el.is_visible(timeout=8000):
                    await el.click(timeout=8000)
                    await asyncio.sleep(2.0)
                    logger.info("[EDR-FORM] 'Continue as guest' clicked")
                    return True
            except Exception:
                continue
        logger.warning("[EDR-FORM] Could not find 'Continue as guest' button on auth-check page")
        return False

    async def click_verify_with_fayda(self) -> bool:
        """
        Step C (pre-handoff): On /booking/passengers, click 'Verify with Fayda'.
        After this the agent immediately hands off to the human and pauses.
        """
        selectors = [
            "button:has-text(\"Verify with Fayda\")",
            "button:has-text(\"Fayda\")",
            "[data-testid='fayda-verify']",
        ]
        for sel in selectors:
            try:
                el = self._page.locator(sel).first
                if await el.is_visible(timeout=8000):
                    await el.click(timeout=8000)
                    await asyncio.sleep(1.0)
                    logger.info("[EDR-FORM] 'Verify with Fayda' clicked — human handoff imminent")
                    return True
            except Exception:
                continue
        logger.warning("[EDR-FORM] Could not find 'Verify with Fayda' button")
        return False

    async def click_continue_to_seat_selection(self) -> bool:
        """
        Step D (post-resume): After human completes Fayda verification, click the
        green 'Continue to seat selection' button to advance to the seat map.
        """
        selectors = [
            "button:has-text(\"Continue to seat selection\")",
            "button:has-text(\"Continue to Seat Selection\")",
            "button:has-text(\"Continue\")",
        ]
        for sel in selectors:
            try:
                el = self._page.locator(sel).first
                if await el.is_visible(timeout=10000):
                    await el.click(timeout=10000)
                    await asyncio.sleep(2.0)
                    logger.info("[EDR-FORM] 'Continue to seat selection' clicked")
                    return True
            except Exception:
                continue
        logger.warning("[EDR-FORM] Could not find 'Continue to seat selection' button")
        return False

    async def auto_assign_and_continue_seats(self) -> bool:
        """
        Step E: On /booking/seats, wait for hydration, click auto-assign.
        The site auto-navigates to /booking/review after assignment.
        Returns True if the URL transitions to /booking/review.
        """
        # --- 1. Wait for Hydration and Page Stability ---
        try:
            await self._page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        await asyncio.sleep(1.5)  # Allow React event handlers to attach

        # --- 2. Click Auto-Assign and Wait for Auto-Navigation ---
        try:
            auto_assign_btn = self._page.locator('button:has-text("Auto Assign Seats")').first
            await auto_assign_btn.wait_for(state="visible", timeout=5000)
            await auto_assign_btn.click(timeout=5000)
            logger.info("[EDR-FORM] 'Auto Assign Seats' clicked")
            
            # Immediately wait for the automatic redirect to the review page
            await self._page.wait_for_url("**/booking/review", timeout=15000)
            logger.info("[EDR-FORM] Reached /booking/review via auto-navigation")
            return True
            
        except Exception as e:
            logger.warning(f"[EDR-FORM] Auto-assign or navigation failed: {e}")
            return False

    async def click_confirm_and_pay(self) -> bool:
        """
        Step F: On /booking/review, click 'Confirm and pay' to submit the booking.
        After this the agent hands off to the human for payment completion.
        """
        try:
            # Target the button that has the text, is NOT disabled, and IS currently visible on screen
            pay_btn = self._page.locator('button:has-text("Confirm and pay"):not([disabled]):visible')
            
            # Wait for it to attach and settle
            await pay_btn.wait_for(timeout=15000)
            
            # Brief pause for DOM stability, then click
            await asyncio.sleep(1.0)
            await pay_btn.click()
            
            # Wait for navigation to the payment page so the backend has time to process the SMS
            await self._page.wait_for_url("**/booking/payment", timeout=20000)
            
            logger.info("[EDR-FORM] 'Confirm and pay' clicked — handing off to human for payment")
            return True
        except Exception as e:
            logger.warning(f"[EDR-FORM] Could not click 'Confirm and pay' button on review page: {str(e)}")
            return False

    async def wait_for_page_url_contains(self, path_fragment: str, timeout_ms: int = 15000) -> bool:
        """
        Wait until the current URL contains path_fragment (e.g. '/booking/auth-check').
        Used between deterministic steps to confirm navigation completed.
        """
        try:
            await self._page.wait_for_url(f"**{path_fragment}**", timeout=timeout_ms)
            logger.info(f"[EDR-FORM] Reached URL containing: {path_fragment}")
            return True
        except Exception:
            current = self._page.url
            logger.warning(
                f"[EDR-FORM] Timeout waiting for URL '{path_fragment}'. "
                f"Current URL: {current}"
            )
            return False


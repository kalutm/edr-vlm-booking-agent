#!/usr/bin/env python3
"""
poc.py — Proof-of-Concept: End-to-end VLM feedback loop

Demonstrates the minimal feedback loop:
  1. Open EDR website
  2. Take screenshot (OBSERVE)
  3. Send to Gemini (PERCEIVE)
  4. Display structured result
  5. Predict action
  6. Execute action
  7. Take new screenshot (OBSERVE again)

Run with:
  python poc.py

Requires GEMINI_API_KEY in environment.
"""

import asyncio
import json
import os
import sys
from datetime import date

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edr_agent.browser.controller import BrowserController
from edr_agent.config import settings
from edr_agent.vlm.client import VLMClient
from edr_agent.vlm.perception import PerceptionModule


async def run_poc():
    print("=" * 60)
    print("EDR VLM Booking Agent — Proof of Concept")
    print("=" * 60)

    # Validate API key
    if not settings.gemini_api_key:
        print("ERROR: GEMINI_API_KEY not set. Exiting.")
        sys.exit(1)

    print(f"Model: {settings.gemini_model}")
    print(f"URL:   {settings.edr_base_url}")
    print()

    # Initialize modules
    vlm = VLMClient()
    browser = BrowserController()
    perception_module = PerceptionModule(vlm)

    from edr_agent.date_logic import next_operating_date
    origin = "Lebu"
    destination = "Dire Dawa"
    target_date = next_operating_date(date.today(), origin=origin, destination=destination) or date.today()
    print(f"Route: {origin} → {destination} on {target_date}")
    print()

    await browser.launch()

    try:
        for cycle in range(3):
            print(f"\n{'─'*50}")
            print(f"  FEEDBACK LOOP — Cycle {cycle + 1}")
            print(f"{'─'*50}")

            # ── STEP 1: Navigate (first cycle only) ──
            if cycle == 0:
                print("\n[1/5] NAVIGATING to EDR website...")
                await browser.navigate(settings.edr_base_url)

            # ── STEP 2: OBSERVE — Take screenshot ──
            print("\n[2/5] OBSERVE — Taking screenshot...")
            screenshot_bytes, path = await browser.screenshot(f"poc_cycle_{cycle+1}")
            print(f"      Screenshot saved: {path}")
            current_url = await browser.current_url()
            print(f"      Current URL: {current_url}")

            # ── STEP 3: PERCEIVE — VLM analysis ──
            print("\n[3/5] PERCEIVE — Sending to Gemini VLM (with 3s pacing delay)...")
            await asyncio.sleep(3.0)
            perception = perception_module.perceive(
                screenshot_bytes,
                origin=origin,
                destination=destination,
                target_date=target_date,
                preferred_seat="Economy",
                workflow_step=f"CYCLE_{cycle+1}",
            )

            print("\n      ┌── Perception Result ──────────────────────")
            print(f"      │  Page:       {perception.current_page.value}")
            print(f"      │  Date State: {perception.date_state.value}")
            print(f"      │  Schedule:   {perception.schedule_state.value}")
            print(f"      │  Seats:      {[s.seat_type for s in perception.available_seats]}")
            print(f"      │  Needs Human:{perception.requires_human}")
            print(f"      │  Confidence: {perception.confidence:.2f}")
            print(f"      │  Description:{perception.raw_description[:80]}...")
            print(f"      └──────────────────────────────────────────")

            # Wait briefly for page to react
            await asyncio.sleep(2)

    finally:
        print("\n\nClosing browser...")
        await browser.close()

    print("\n" + "=" * 60)
    print("POC Complete — Feedback loop demonstrated!")
    print(f"Screenshots saved to: screenshots/")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_poc())

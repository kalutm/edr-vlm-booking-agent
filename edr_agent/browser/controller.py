"""
browser/controller.py — Playwright browser controller.

Manages browser lifecycle and exposes clean methods for:
- Taking screenshots
- Navigating to URLs
- Providing screenshots to the VLM
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from edr_agent.config import settings

logger = logging.getLogger(__name__)

SCREENSHOTS_DIR = Path("screenshots")


class BrowserController:
    """
    Manages a headed Playwright browser instance.
    The browser is always visible during demos so evaluators can see the agent working.
    """

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        SCREENSHOTS_DIR.mkdir(exist_ok=True)

    async def launch(self) -> None:
        """Launch the browser. Call once before using any other methods."""
        logger.info(f"[BROWSER] Launching Chromium (headless={settings.browser_headless})")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=settings.browser_headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        logger.info("[BROWSER] Browser launched successfully")

    async def close(self) -> None:
        """Clean up browser resources."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            logger.info("[BROWSER] Browser closed")
        except Exception as e:
            logger.warning(f"[BROWSER] Error during close: {e}")

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._page

    async def navigate(self, url: str, wait_for: str = "networkidle", timeout: int = 30000) -> None:
        """Navigate to a URL and wait for the page to settle."""
        logger.info(f"[BROWSER] Navigating to: {url}")
        await self.page.goto(url, wait_until=wait_for, timeout=timeout)
        await asyncio.sleep(1.5)  # Extra settle time for JS-heavy pages

    async def screenshot(self, label: str = "step") -> tuple[bytes, str]:
        """
        Take a full-page screenshot.

        Returns:
            Tuple of (png_bytes, file_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{label}.png"
        filepath = str(SCREENSHOTS_DIR / filename)

        screenshot_bytes = await self.page.screenshot(
            path=filepath,
            full_page=False,   # Viewport only — what the user would see
            type="png",
        )
        logger.info(f"[BROWSER] Screenshot saved: {filepath}")
        return screenshot_bytes, filepath

    async def current_url(self) -> str:
        return self.page.url

    async def get_page_title(self) -> str:
        return await self.page.title()

    async def wait_for_selector(self, selector: str, timeout: int = 10000) -> bool:
        """Wait for an element to appear. Returns True if found, False if timeout."""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    async def is_alive(self) -> bool:
        """Check if the browser and page are still functional."""
        try:
            _ = self.page.url
            return True
        except Exception:
            return False

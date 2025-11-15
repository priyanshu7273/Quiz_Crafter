"""Utilities for rendering JavaScript heavy quiz pages."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from app.config import get_settings
from app.utils.logger import configure_logger, log_extra

logger = configure_logger(__name__)

try:  # pragma: no cover - imported lazily in runtime environments
    from playwright.async_api import async_playwright, Browser, Page
except ModuleNotFoundError:  # pragma: no cover
    async_playwright = None  # type: ignore[assignment]
    Browser = Page = None  # type: ignore[misc]

try:  # pragma: no cover
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
except ModuleNotFoundError:  # pragma: no cover
    webdriver = ChromeOptions = None  # type: ignore[misc]


class BrowserManager:
    """Launches either Playwright or Selenium depending on availability."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._use_playwright = async_playwright is not None

    async def __aenter__(self) -> "BrowserManager":
        await self._start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _start(self) -> None:
        settings = get_settings()
        if self._use_playwright:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=settings.use_headless_browser,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            logger.info("Playwright browser ready", extra=log_extra(engine="playwright"))
        elif webdriver is not None:
            logger.info("Falling back to Selenium webdriver", extra=log_extra(engine="selenium"))
        else:
            raise RuntimeError("Neither Playwright nor Selenium is installed")

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def render(self, url: str, *, wait_for: Optional[str] = None) -> Dict[str, Any]:
        """Render the supplied URL and return a dictionary of extracted artefacts."""

        if self._use_playwright and self._browser:
            page = await self._browser.new_page()
            await page.goto(url, wait_until="networkidle")
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=10000)
            else:
                await asyncio.sleep(2)
            html = await page.content()
            text = await page.inner_text("body")
            screenshot_bytes = await page.screenshot(full_page=True)
            tables = await page.evaluate(
                """
                () => Array.from(document.querySelectorAll('table')).map(table => ({
                    headers: Array.from(table.querySelectorAll('th')).map(h => h.innerText.trim()),
                    rows: Array.from(table.querySelectorAll('tr')).map(row =>
                        Array.from(row.querySelectorAll('td')).map(cell => cell.innerText.trim())
                    ).filter(r => r.length > 0)
                }))
                """
            )
            links = await page.evaluate(
                """
                () => Array.from(document.querySelectorAll('a')).map(anchor => ({
                    text: anchor.innerText.trim(),
                    href: anchor.href
                }))
                """
            )
            await page.close()
            return {
                "html": html,
                "text": text,
                "tables": tables,
                "links": links,
                "screenshot": base64.b64encode(screenshot_bytes).decode("utf-8"),
            }
        if webdriver is None:
            raise RuntimeError("Selenium webdriver unavailable")
        options = ChromeOptions()
        settings = get_settings()
        if settings.use_headless_browser:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            if wait_for:
                # blocking wait using webdriver built-ins
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC

                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_for)))
            else:
                await asyncio.sleep(2)
            html = driver.page_source
            text = driver.find_element("tag name", "body").text
            screenshot_bytes = driver.get_screenshot_as_png()
            return {
                "html": html,
                "text": text,
                "tables": [],
                "links": [],
                "screenshot": base64.b64encode(screenshot_bytes).decode("utf-8"),
            }
        finally:
            driver.quit()


@asynccontextmanager
def browser_session() -> Any:
    """Convenience async context manager returning an initialised browser manager."""

    manager = BrowserManager()
    await manager._start()
    try:
        yield manager
    finally:
        await manager.close()

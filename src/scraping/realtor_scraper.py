"""
Realtor.com listing scraper.

Design principles:
- Rate-limit aggressively (2-5s between requests)
- Cache every raw HTML response before parsing — if the parser breaks,
  we don't re-scrape
- Log every URL: success, failure, reason
- Retry with exponential backoff on transient failures
- Rotate user agents

Note: Always check robots.txt and terms of service before scraping.
This is for educational/portfolio use. For production, prefer official APIs.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import (
    HTML_CACHE_DIR,
    SCRAPE_DELAY_MAX,
    SCRAPE_DELAY_MIN,
    SCRAPE_MAX_RETRIES,
    SCRAPE_TIMEOUT,
)

logger = logging.getLogger(__name__)
ua = UserAgent()


class ScrapingError(Exception):
    """Raised when a request fails after all retries."""


def _cache_path(url: str) -> Path:
    """Deterministic cache path for a URL."""
    HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    return HTML_CACHE_DIR / f"{digest}.html"


def _read_cache(url: str) -> Optional[str]:
    path = _cache_path(url)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _write_cache(url: str, html: str) -> None:
    _cache_path(url).write_text(html, encoding="utf-8")


@retry(
    stop=stop_after_attempt(SCRAPE_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True,
)
def _fetch(url: str, headers: dict) -> str:
    response = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
    response.raise_for_status()
    return response.text


def fetch_url(url: str, use_cache: bool = True) -> str:
    """
    Fetch a URL with caching, rate limiting, and retries.

    Args:
        url: The URL to fetch.
        use_cache: If True, returns cached HTML when available.

    Returns:
        The raw HTML content.

    Raises:
        ScrapingError: If the URL cannot be fetched after retries.
    """
    if use_cache:
        cached = _read_cache(url)
        if cached is not None:
            logger.debug("Cache hit: %s", url)
            return cached

    headers = {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Polite rate limit
    time.sleep(random.uniform(SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX))

    try:
        html = _fetch(url, headers)
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        raise ScrapingError(f"Could not fetch {url}") from exc

    _write_cache(url, html)
    logger.info("Fetched and cached: %s", url)
    return html


def parse_listing(html: str) -> dict:
    """
    Parse a single Realtor.com listing page.

    TODO: Implement actual selectors after inspecting page structure.
    Use SelectorGadget or browser DevTools to find stable selectors.

    Returns a dict with keys:
        url, address, city, state, zip, price, beds, baths, sqft,
        lot_sqft, year_built, property_type, days_on_market, description,
        latitude, longitude, listing_id
    """
    soup = BeautifulSoup(html, "lxml")

    # Placeholder structure — fill in real selectors on Day 2
    return {
        "address": None,
        "price": None,
        "beds": None,
        "baths": None,
        "sqft": None,
        "year_built": None,
        "description": None,
        # ...
    }

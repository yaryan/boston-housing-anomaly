"""
Day 2-3 entry point: scrape active Realtor.com listings for target cities.

Usage:
    python scripts/run_scraper.py --cities boston cambridge somerville
    python scripts/run_scraper.py --max-listings 100  # for testing
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import TARGET_CITIES  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__, Path("logs/scraper.log"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Realtor.com listings.")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=TARGET_CITIES,
        help="Cities to scrape (default: all configured).",
    )
    parser.add_argument(
        "--max-listings",
        type=int,
        default=None,
        help="Cap number of listings per city (for testing).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip HTML cache and re-fetch.",
    )
    args = parser.parse_args()

    logger.info("Starting scrape for cities: %s", args.cities)
    logger.info("Max listings per city: %s", args.max_listings or "no limit")

    # TODO: Day 2 — implement actual scrape loop
    # for city in args.cities:
    #     urls = discover_listing_urls(city, max_listings=args.max_listings)
    #     for url in urls:
    #         html = fetch_url(url, use_cache=not args.no_cache)
    #         listing = parse_listing(html)
    #         save_to_db(listing)

    logger.info("Scrape complete.")


if __name__ == "__main__":
    main()

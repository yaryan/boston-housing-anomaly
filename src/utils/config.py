"""Project-wide configuration. Edit paths and target cities here."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

DUCKDB_PATH = PROCESSED_DIR / "housing.duckdb"
HTML_CACHE_DIR = RAW_DIR / "html_cache"

# Target cities for v1
TARGET_CITIES = [
    "Boston",
    "Cambridge",
    "Somerville",
    "Brookline",
    "Newton",
]

# Scraping
SCRAPE_DELAY_MIN = 2.0  # seconds
SCRAPE_DELAY_MAX = 5.0
SCRAPE_TIMEOUT = 30
SCRAPE_MAX_RETRIES = 3

# Modeling
RANDOM_SEED = 42
TIME_SPLIT_DATE = "2024-01-01"  # train: <, test: >=
MIN_SQFT = 300
MAX_SQFT = 10000
MIN_PRICE = 100_000
MAX_PRICE = 10_000_000

# Anomaly threshold (in standard deviations of residual)
ANOMALY_THRESHOLD = 1.5

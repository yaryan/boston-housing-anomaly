"""
Day 6, refinement: post-hoc anomaly classification.

After running score_and_explain.py, we discovered the residual distribution has
two failure modes that aren't real pricing anomalies:

1. EXTREME UNDERPRICED (actual < 30% of predicted): these are mostly
   nominal-consideration deeds, family transfers, pre-construction sales,
   and other sub-market transactions, not real "underpricing".
2. EXTREME OVERPRICED (actual > 3x predicted): these are mostly bulk
   building purchases that the assessor recorded against every individual
   unit, not a real "overpricing" by 1000%.

This script reclassifies the ANOMALY_FLAG into 5 categories:
   normal                       — within ±1.5 z-score
   overpriced                   — z > 1.5 AND ratio (actual/pred) <= 3.0
   underpriced                  — z < -1.5 AND ratio >= 0.30
   bulk_undersale_or_transfer   — actual < 30% of predicted (likely data artifact)
   bulk_oversale_or_building    — actual > 3x predicted (likely data artifact)

Run from project root:
    python scripts/refine_anomaly_flags.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_PATH = Path("data/processed/predictions.parquet")

UNDER_RATIO_THRESHOLD = 0.30   # actual must be at least 30% of predicted
OVER_RATIO_THRESHOLD = 3.0     # actual must be at most 3x of predicted
Z_THRESHOLD = 1.5


def reclassify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ratio = df["LS_PRICE"] / df["PREDICTED_PRICE"]
    z = df["LOG_RESIDUAL_Z"]

    new_flag = np.where(
        ratio < UNDER_RATIO_THRESHOLD, "bulk_undersale_or_transfer",
        np.where(
            ratio > OVER_RATIO_THRESHOLD, "bulk_oversale_or_building",
            np.where(z > Z_THRESHOLD, "overpriced",
                     np.where(z < -Z_THRESHOLD, "underpriced", "normal"))
        )
    )
    df["ANOMALY_FLAG"] = new_flag
    df["PRICE_RATIO"] = ratio
    return df


def main() -> None:
    print(f"Loading {PREDICTIONS_PATH}...")
    df = pd.read_parquet(PREDICTIONS_PATH)
    print(f"  Rows: {len(df):,}")

    df = reclassify(df)

    print("\nNew anomaly flag distribution:")
    counts = df["ANOMALY_FLAG"].value_counts()
    for flag, n in counts.items():
        pct = n / len(df) * 100
        print(f"  {flag:<32s} {n:>6,}  ({pct:>5.2f}%)")

    print(f"\nSaving back to {PREDICTIONS_PATH}...")
    df.to_parquet(PREDICTIONS_PATH, index=False)
    print("Done.")

    # Print samples from each category for sanity check
    print("\n" + "=" * 80)
    print("SAMPLES FROM EACH CATEGORY (3 each)")
    print("=" * 80)
    cols = ["SITE_ADDR", "CITY_NAME", "LS_YEAR", "LS_PRICE", "PREDICTED_PRICE",
            "PRICE_RATIO", "ANOMALY_FLAG"]
    cols = [c for c in cols if c in df.columns]
    for flag in ["overpriced", "underpriced", "bulk_undersale_or_transfer",
                 "bulk_oversale_or_building"]:
        subset = df[df["ANOMALY_FLAG"] == flag]
        if len(subset) == 0:
            continue
        print(f"\n--- {flag.upper()} ---")
        sample = subset.sample(min(3, len(subset)), random_state=42)
        print(sample[cols].to_string(index=False))


if __name__ == "__main__":
    main()
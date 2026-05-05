"""
Day 6: Score every property and compute SHAP explanations.

Outputs three things:
1. predictions.parquet     — for every property: predicted price, actual price,
                              residual (actual - predicted), z-score of residual,
                              and an anomaly_flag (overpriced / underpriced / normal)
2. shap_values.parquet     — for every property: the SHAP attribution for each
                              feature (how much it pushed the prediction up or down)
3. anomaly_summary.txt     — a human-readable summary of the most over- and
                              under-priced properties

Run from project root:
    python scripts/score_and_explain.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

INPUT_FEATURES = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/lgb_model.pkl")
OUTPUT_PREDICTIONS = Path("data/processed/predictions.parquet")
OUTPUT_SHAP = Path("data/processed/shap_values.parquet")
OUTPUT_SUMMARY = Path("data/processed/anomaly_summary.txt")

# Threshold for flagging anomalies (z-score of log-residual)
Z_THRESHOLD = 1.5

# Feature columns must match training order
NUMERIC_FEATURES = [
    "RES_AREA", "YEAR_BUILT", "UNITS", "NUM_ROOMS", "STORIES",
    "LS_YEAR", "DIST_TO_T_KM", "NBR_MEDIAN_PSF", "NBR_SAMPLE_SIZE",
    "LATITUDE", "LONGITUDE",
]
CATEGORICAL_FEATURES = ["CITY_NAME", "USE_CODE_NUM", "STYLE", "ZIP"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def score_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """Predict for every row, compute residuals and anomaly flags."""
    print("\nScoring predictions for all properties...")
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols]

    log_pred = model.predict(X)
    pred_price = np.exp(log_pred)
    actual = df["LS_PRICE"].to_numpy()

    # Residual in log space (% over/under), then z-score
    log_actual = np.log(actual)
    log_residual = log_actual - log_pred
    z = (log_residual - log_residual.mean()) / log_residual.std()

    flag = np.where(
        z > Z_THRESHOLD, "overpriced",
        np.where(z < -Z_THRESHOLD, "underpriced", "normal")
    )

    out = df.copy()
    out["PREDICTED_PRICE"] = pred_price
    out["RESIDUAL"] = actual - pred_price
    out["RESIDUAL_PCT"] = (actual - pred_price) / pred_price * 100
    out["LOG_RESIDUAL_Z"] = z
    out["ANOMALY_FLAG"] = flag

    print(f"  Predicted prices: ${pred_price.min():,.0f} to ${pred_price.max():,.0f}")
    print(f"  Anomaly distribution:")
    for f, n in pd.Series(flag).value_counts().items():
        print(f"    {f}: {n:,} ({n/len(flag):.1%})")

    return out


def compute_shap(df: pd.DataFrame, model) -> pd.DataFrame:
    """Compute SHAP values for every property. This is the slow step."""
    print("\nComputing SHAP values for all properties...")
    print("  (This will take ~10-15 minutes. Go make tea.)")
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP values are in log-price space. Convert to dollar contribution
    # by exponentiating at the base + cumulative shap, but simpler is to
    # just keep them in log space and let the UI describe directionally.
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    shap_df["base_value"] = explainer.expected_value
    shap_df["LOC_ID"] = df["LOC_ID"].values

    print(f"  SHAP computed: {shap_values.shape[0]:,} rows × {shap_values.shape[1]} features")
    return shap_df


def write_summary(scored: pd.DataFrame, n_top: int = 20) -> None:
    """Write a human-readable summary of the top anomalies."""
    print(f"\nWriting anomaly summary to {OUTPUT_SUMMARY}...")

    cols_to_show = ["SITE_ADDR", "CITY_NAME", "ZIP", "LS_YEAR",
                    "LS_PRICE", "PREDICTED_PRICE", "RESIDUAL_PCT",
                    "RES_AREA", "YEAR_BUILT", "USE_DESC"]
    cols_to_show = [c for c in cols_to_show if c in scored.columns]

    over = scored.nlargest(n_top, "LOG_RESIDUAL_Z")[cols_to_show]
    under = scored.nsmallest(n_top, "LOG_RESIDUAL_Z")[cols_to_show]

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TOP {n_top} OVERPRICED PROPERTIES (sold for much more than predicted)\n")
        f.write("=" * 80 + "\n")
        f.write(over.to_string(index=False))
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"TOP {n_top} UNDERPRICED PROPERTIES (sold for much less than predicted)\n")
        f.write("=" * 80 + "\n")
        f.write(under.to_string(index=False))
        f.write("\n")

    print(f"  Top overpriced ({n_top}): MAE={over['RESIDUAL_PCT'].mean():.0f}% above prediction")
    print(f"  Top underpriced ({n_top}): MAE={under['RESIDUAL_PCT'].mean():.0f}% below prediction")


def main() -> None:
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    print(f"Loading features from {INPUT_FEATURES}...")
    df = pd.read_parquet(INPUT_FEATURES)
    df = prepare_features(df)
    print(f"  Rows: {len(df):,}")

    scored = score_predictions(df, model)

    print(f"\nSaving predictions to {OUTPUT_PREDICTIONS}...")
    scored.to_parquet(OUTPUT_PREDICTIONS, index=False)

    shap_df = compute_shap(df, model)
    print(f"\nSaving SHAP values to {OUTPUT_SHAP}...")
    shap_df.to_parquet(OUTPUT_SHAP, index=False)

    write_summary(scored)

    print("\nDay 6 complete.")
    print(f"  Predictions: {OUTPUT_PREDICTIONS} ({OUTPUT_PREDICTIONS.stat().st_size / 1e6:.1f} MB)")
    print(f"  SHAP values: {OUTPUT_SHAP} ({OUTPUT_SHAP.stat().st_size / 1e6:.1f} MB)")
    print(f"  Summary:     {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
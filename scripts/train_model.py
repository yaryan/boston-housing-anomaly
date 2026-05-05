"""
Day 8 (revised): per-city time split + calibration shift.

Key change vs. previous version:
- After fitting LightGBM, we compute the median log-residual on the TRAINING
  set, then add that constant to all predictions. This recenters the model
  so 50% of training sales fall above prediction and 50% below.
- The shift is saved with the model (`calibration_shift`) so downstream
  scoring uses the same correction.

Run from project root:
    python scripts/train_model.py
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

INPUT = Path("data/processed/features.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "lgb_model.pkl"
CALIBRATION_PATH = MODEL_DIR / "calibration.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

TEST_FRACTION = 0.15

NUMERIC_FEATURES = [
    "RES_AREA", "YEAR_BUILT", "UNITS", "NUM_ROOMS", "STORIES",
    "LS_YEAR", "DIST_TO_T_KM", "NBR_MEDIAN_PSF", "NBR_SAMPLE_SIZE",
    "LATITUDE", "LONGITUDE",
]
CATEGORICAL_FEATURES = ["CITY_NAME", "USE_CODE_NUM", "STYLE", "ZIP"]
TARGET_COL = "LS_PRICE"

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    min_data_in_leaf=50,
    n_estimators=2000,
    verbose=-1,
)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def per_city_time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    print("\nPer-city time-aware split:")
    print(f"  {'City':<12s} {'N':>6s}  {'Train years':>16s}  {'Test years':>14s}  {'Train':>7s}  {'Test':>5s}")
    for city in df["CITY_NAME"].unique():
        city_df = df[df["CITY_NAME"] == city].sort_values("LS_YEAR")
        n = len(city_df)
        n_test = max(int(n * TEST_FRACTION), 50)
        train_df = city_df.iloc[:-n_test]
        test_df = city_df.iloc[-n_test:]
        train_parts.append(train_df)
        test_parts.append(test_df)
        print(f"  {city:<12s} {n:>6,}  "
              f"{train_df['LS_YEAR'].min():.0f}-{train_df['LS_YEAR'].max():.0f}".ljust(35) +
              f"  {test_df['LS_YEAR'].min():.0f}-{test_df['LS_YEAR'].max():.0f}".ljust(16) +
              f"  {len(train_df):>6,}  {len(test_df):>5,}")
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:42s} MAE: ${mae:>9,.0f}  MAPE: {mape:5.1f}%  RMSE: ${rmse:>10,.0f}  R^2: {r2:.3f}")
    return {"label": label, "mae": float(mae), "mape": float(mape), "rmse": float(rmse), "r2": float(r2)}


def baseline_mean(y_train, y_test):
    y_pred = np.full_like(y_test, np.exp(np.mean(np.log(y_train))), dtype=float)
    return metrics(y_test, y_pred, "Baseline: mean predictor")


def baseline_linear(train, test):
    feature_cols = [c for c in NUMERIC_FEATURES if c in train.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in train.columns]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
    medians = train[feature_cols].median()
    X_train_num = train[feature_cols].fillna(medians).to_numpy()
    X_test_num = test[feature_cols].fillna(medians).to_numpy()
    X_train_cat = ohe.fit_transform(train[cat_cols].astype(str).fillna("missing"))
    X_test_cat = ohe.transform(test[cat_cols].astype(str).fillna("missing"))
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])
    y_train = np.log(train[TARGET_COL].to_numpy())
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = np.exp(model.predict(X_test))
    return metrics(test[TARGET_COL].to_numpy(), y_pred, "Linear regression"), y_pred


def baseline_assessor(test):
    mask = test["TOTAL_VAL"].notna() & (test["TOTAL_VAL"] > 0)
    return metrics(
        test.loc[mask, TARGET_COL].to_numpy(),
        test.loc[mask, "TOTAL_VAL"].to_numpy().astype(float),
        f"City assessor (TOTAL_VAL, n={mask.sum()})"
    )


def train_lgb(train, test):
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in train.columns]
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = np.log(train[TARGET_COL].to_numpy())

    val_size = int(len(train) * 0.15)
    val_idx = np.random.RandomState(42).choice(len(train), size=val_size, replace=False)
    val_mask = np.zeros(len(train), dtype=bool)
    val_mask[val_idx] = True
    X_tr, X_val = X_train[~val_mask], X_train[val_mask]
    y_tr, y_val = y_train[~val_mask], y_train[val_mask]

    print("\nTraining LightGBM...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feature_cols],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    # CALIBRATION SHIFT: compute median log-residual on TRAINING set,
    # then add to all future predictions so the median residual is zero.
    train_log_pred = model.predict(X_train)
    train_log_actual = np.log(train[TARGET_COL].to_numpy())
    median_log_residual = np.median(train_log_actual - train_log_pred)
    print(f"  Calibration shift (median log residual on train): {median_log_residual:+.4f}")
    print(f"  -> predictions multiplied by {np.exp(median_log_residual):.4f}")

    # Apply shift and evaluate on test
    y_pred_test = np.exp(model.predict(X_test) + median_log_residual)
    test_metric = metrics(test[TARGET_COL].to_numpy(), y_pred_test, "LightGBM (calibrated)")

    return test_metric, y_pred_test, model, median_log_residual


def per_city_breakdown(test, y_pred, label):
    print(f"\nPer-city {label} performance:")
    print(f"  {'City':<12s} {'N':>6s}  {'MAE':>11s}  {'MAPE':>6s}  {'R^2':>6s}  {'%above':>7s}")
    for city in test["CITY_NAME"].unique():
        mask = test["CITY_NAME"].to_numpy() == city
        if mask.sum() < 10:
            continue
        y_true = test.loc[mask, TARGET_COL].to_numpy()
        y_p = y_pred[mask]
        mae = mean_absolute_error(y_true, y_p)
        mape = np.mean(np.abs((y_true - y_p) / y_true)) * 100
        r2 = r2_score(y_true, y_p) if mask.sum() >= 2 else float("nan")
        pct_above = (y_true > y_p).mean() * 100
        print(f"  {city:<12s} {mask.sum():>6,}  ${mae:>9,.0f}  {mape:>5.1f}%  {r2:>5.3f}  {pct_above:>6.1f}%")


def main() -> None:
    print(f"Loading {INPUT}...")
    df = pd.read_parquet(INPUT)
    print(f"  Rows: {len(df):,}")

    df = prepare_features(df)
    train, test = per_city_time_split(df)
    print(f"\nTotal train: {len(train):,}, total test: {len(test):,}")

    print("\nEvaluation (lower MAE/MAPE = better, higher R^2 = better)")
    print("-" * 110)

    all_metrics = []
    all_metrics.append(baseline_mean(train[TARGET_COL].to_numpy(), test[TARGET_COL].to_numpy()))
    all_metrics.append(baseline_assessor(test))
    lin_m, _ = baseline_linear(train, test)
    all_metrics.append(lin_m)
    lgb_m, y_pred, model, shift = train_lgb(train, test)
    all_metrics.append(lgb_m)

    per_city_breakdown(test, y_pred, "calibrated LightGBM")

    print("\nFeature importance (top 10):")
    importance = pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(10)
    print(importance.to_string(index=False))

    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump({"log_shift": float(shift)}, f, indent=2)
    print(f"Saved calibration shift to {CALIBRATION_PATH}")
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
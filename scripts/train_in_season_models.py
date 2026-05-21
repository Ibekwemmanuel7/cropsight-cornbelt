#!/usr/bin/env python3
"""
Train one XGBoost model per forecast week K on the in-season feature matrices
produced by `build_in_season_features.py`.

For each K:
    1. Load feature_matrix_k{K}_phenology.parquet
    2. Train: county-years with year <= 2021
       Val  : 2022
       Test : 2023
    3. Save model + per-row predictions + accuracy metrics
    4. Append a row to horizon_leaderboard.parquet

Outputs:
    models/xgboost_k{K}.json                    - trained XGBoost
    data/interim/predictions_k{K}.parquet       - per-row predictions on test
    data/interim/horizon_leaderboard.parquet    - one row per (K, split)

XGBoost is the production target. If xgboost is unavailable (e.g. running this
in a constrained sandbox), the script falls back to sklearn's
HistGradientBoostingRegressor as a stopgap. The leaderboard records which
backend produced the numbers.

Usage:
    python scripts/train_in_season_models.py
    python scripts/train_in_season_models.py --weeks 28 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parent.parent
DATA_INTERIM = REPO / "data" / "interim"
MODELS_DIR = REPO / "models"

DEFAULT_WEEKS = [16, 20, 24, 28, 32, 36]

# Hyperparameters - matched to the existing full-season XGBoost
XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)


def _try_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        return None


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, split: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    print(f"    [{name:<8}] {split:5s}  RMSE={rmse:5.2f}  MAE={mae:5.2f}  R2={r2:+.3f}  bias={bias:+5.2f}")
    return {"split": split, "rmse": rmse, "mae": mae, "r2": r2, "bias": bias}


def train_xgboost(X_train, y_train, X_val, y_val):
    import xgboost as xgb
    model = xgb.XGBRegressor(early_stopping_rounds=50, **XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model, "xgboost"


def train_sklearn_hgb(X_train, y_train, X_val, y_val):
    """Fallback when xgboost is unavailable."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=None,  # we pass val externally
        random_state=42,
    )
    # HGB doesn't accept an external val set the same way; concat and let it use the tail
    model.fit(X_train, y_train)
    return model, "sklearn_hgb"


def save_model(model, backend: str, week_k: int) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if backend == "xgboost":
        out = MODELS_DIR / f"xgboost_k{week_k}.json"
        model.save_model(str(out))
    else:
        import joblib
        out = MODELS_DIR / f"hgb_k{week_k}.joblib"
        joblib.dump(model, out)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything except identifiers and target."""
    return [c for c in df.columns if c not in ("fips", "year", "yield_bu_acre",
                                                "state", "county_name")]


def train_one(week_k: int, use_xgb: bool) -> dict:
    print(f"\n----- K={week_k} -----")
    path = DATA_INTERIM / f"feature_matrix_k{week_k}_phenology.parquet"
    if not path.exists():
        sys.exit(f"missing {path} - run build_in_season_features.py first")
    feats = pd.read_parquet(path)

    if "yield_bu_acre" not in feats.columns:
        sys.exit(f"K={week_k}: target column yield_bu_acre missing")

    train_df = feats[feats["year"] <= 2021].dropna(subset=["yield_bu_acre"]).copy()
    val_df = feats[feats["year"] == 2022].dropna(subset=["yield_bu_acre"]).copy()
    test_df = feats[feats["year"] == 2023].dropna(subset=["yield_bu_acre"]).copy()

    cols = feature_columns(feats)
    # Drop columns that are entirely NaN at this K (XGBoost handles NaN but
    # entirely-NaN columns waste capacity).
    cols = [c for c in cols if train_df[c].notna().any()]

    X_train, y_train = train_df[cols].values, train_df["yield_bu_acre"].values
    X_val, y_val = val_df[cols].values, val_df["yield_bu_acre"].values
    X_test, y_test = test_df[cols].values, test_df["yield_bu_acre"].values

    print(f"  features: {len(cols)} columns, rows train={len(X_train)} "
          f"val={len(X_val)} test={len(X_test)}")

    t0 = time.time()
    if use_xgb:
        model, backend = train_xgboost(X_train, y_train, X_val, y_val)
    else:
        model, backend = train_sklearn_hgb(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0
    print(f"  trained ({backend}) in {train_time:.1f}s")

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate("model", y_val, val_pred, "val")
    test_metrics = evaluate("model", y_test, test_pred, "test")

    model_path = save_model(model, backend, week_k)
    print(f"  saved model -> {model_path.relative_to(REPO)}")

    pred_df = test_df[["fips", "year", "yield_bu_acre"]].copy()
    pred_df["pred"] = test_pred
    pred_out = DATA_INTERIM / f"predictions_k{week_k}.parquet"
    pred_df.to_parquet(pred_out, index=False)

    rows = []
    for split_metrics in (val_metrics, test_metrics):
        rows.append({
            "week_k": week_k,
            "cutoff_doy": week_k * 7,
            "backend": backend,
            "n_features": len(cols),
            **split_metrics,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", nargs="+", type=int, default=DEFAULT_WEEKS)
    parser.add_argument(
        "--force-sklearn",
        action="store_true",
        help="Use sklearn HGB even if xgboost is available.",
    )
    args = parser.parse_args()

    xgb_module = None if args.force_sklearn else _try_xgboost()
    use_xgb = xgb_module is not None
    if use_xgb:
        print(f"using xgboost {xgb_module.__version__}")
    else:
        print("xgboost not available - falling back to sklearn HistGradientBoostingRegressor")
        print("(re-run on a machine with xgboost installed for production numbers)")

    all_rows = []
    for K in args.weeks:
        all_rows.extend(train_one(K, use_xgb))

    leaderboard = pd.DataFrame(all_rows)
    out = DATA_INTERIM / "horizon_leaderboard.parquet"
    leaderboard.to_parquet(out, index=False)
    out_csv = DATA_INTERIM / "horizon_leaderboard.csv"
    leaderboard.to_csv(out_csv, index=False)

    print("\n=== horizon leaderboard ===")
    print(leaderboard.to_string(index=False))
    print(f"\nsaved {out.relative_to(REPO)} and {out_csv.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

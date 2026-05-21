#!/usr/bin/env python3
"""
Train one gradient-boosted regressor per forecast week K, with split conformal
prediction intervals.

Year-based split:
    train     : 2000 - 2019  (model fitting)
    calibrate : 2020 - 2021  (residuals -> q_alpha for conformal interval)
    val       : 2022          (early stopping reference; not used for q_alpha)
    test      : 2023          (final evaluation, intervals reported)

XGBoost is the production target. If unavailable, falls back to sklearn's
HistGradientBoostingRegressor. The leaderboard records the backend used.

Outputs:
    models/xgboost_k{K}.json       (or hgb_k{K}.joblib)
    data/interim/predictions_k{K}.parquet   - per-county: pred, lower, upper, within
    data/interim/horizon_leaderboard.parquet - one row per (K, split)

Usage:
    python scripts/train_in_season_models.py
    python scripts/train_in_season_models.py --weeks 28 32 --alpha 0.1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cropsight.uncertainty import conformal  # noqa: E402

DATA_INTERIM = REPO / "data" / "interim"
MODELS_DIR = REPO / "models"

DEFAULT_WEEKS = [16, 20, 24, 28, 32, 36]

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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"split": split, "rmse": rmse, "mae": mae, "r2": r2, "bias": bias}


def train_xgboost(X_train, y_train, X_val, y_val):
    import xgboost as xgb
    model = xgb.XGBRegressor(early_stopping_rounds=50, **XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, "xgboost"


def train_sklearn_hgb(X_train, y_train, X_val, y_val):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=42,
    )
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
    return [c for c in df.columns
            if c not in ("fips", "year", "yield_bu_acre", "state", "county_name")]


def train_one(week_k: int, alpha: float, use_xgb: bool, log) -> list[dict]:
    log(f"\n----- K={week_k} -----")
    path = DATA_INTERIM / f"feature_matrix_k{week_k}_phenology.parquet"
    if not path.exists():
        sys.exit(f"missing {path}; run build_in_season_features.py first")
    feats = pd.read_parquet(path)
    if "yield_bu_acre" not in feats.columns:
        sys.exit(f"K={week_k}: target column yield_bu_acre missing")

    train_df = feats[feats["year"] <= 2019].dropna(subset=["yield_bu_acre"]).copy()
    calib_df = feats[(feats["year"] >= 2020) & (feats["year"] <= 2021)] \
                 .dropna(subset=["yield_bu_acre"]).copy()
    val_df = feats[feats["year"] == 2022].dropna(subset=["yield_bu_acre"]).copy()
    test_df = feats[feats["year"] == 2023].dropna(subset=["yield_bu_acre"]).copy()

    cols = feature_columns(feats)
    cols = [c for c in cols if train_df[c].notna().any()]

    X_train, y_train = train_df[cols].values, train_df["yield_bu_acre"].values
    X_calib, y_calib = calib_df[cols].values, calib_df["yield_bu_acre"].values
    X_val, y_val = val_df[cols].values, val_df["yield_bu_acre"].values
    X_test, y_test = test_df[cols].values, test_df["yield_bu_acre"].values

    log(f"  features {len(cols)} cols | rows train={len(X_train)} "
        f"calib={len(X_calib)} val={len(X_val)} test={len(X_test)}")

    t0 = time.time()
    if use_xgb:
        model, backend = train_xgboost(X_train, y_train, X_val, y_val)
    else:
        model, backend = train_sklearn_hgb(X_train, y_train, X_val, y_val)
    log(f"  trained ({backend}) in {time.time()-t0:.1f}s")

    cal = conformal.calibrate_and_predict(model, X_calib, y_calib, X_test, alpha=alpha)
    q_alpha = cal["q_alpha"]
    test_pred = cal["pred"]
    test_lower = cal["lower"]
    test_upper = cal["upper"]
    log(f"  q_alpha={q_alpha:.2f} bu/ac  (1-alpha={1-alpha:.0%} interval, n_calib={cal['calib_n']})")

    val_pred = model.predict(X_val)
    val_metrics = evaluate(y_val, val_pred, "val")
    test_metrics = evaluate(y_test, test_pred, "test")
    log(f"    [val ] RMSE={val_metrics['rmse']:.2f}  MAE={val_metrics['mae']:.2f}  R2={val_metrics['r2']:+.3f}")
    log(f"    [test] RMSE={test_metrics['rmse']:.2f}  MAE={test_metrics['mae']:.2f}  R2={test_metrics['r2']:+.3f}")

    coverage = conformal.evaluate_coverage(y_test, test_lower, test_upper)
    mean_width = conformal.evaluate_mean_width(test_lower, test_upper)
    log(f"    [test] conformal coverage={coverage:.3f}  mean width={mean_width:.2f} bu/ac")

    model_path = save_model(model, backend, week_k)
    log(f"  saved model -> {model_path.relative_to(REPO)}")

    pred_df = test_df[["fips", "year", "yield_bu_acre"]].copy()
    pred_df["pred"] = test_pred
    pred_df["lower"] = test_lower
    pred_df["upper"] = test_upper
    pred_df["within_interval"] = (
        (y_test >= test_lower) & (y_test <= test_upper)
    ).astype(int)
    pred_df["q_alpha"] = q_alpha
    pred_out = DATA_INTERIM / f"predictions_k{week_k}.parquet"
    pred_df.to_parquet(pred_out, index=False)

    common = dict(week_k=week_k, cutoff_doy=week_k * 7, backend=backend,
                  n_features=len(cols), alpha=alpha, q_alpha=q_alpha)
    rows = [
        {**common, **val_metrics, "coverage": float("nan"), "mean_width": float("nan")},
        {**common, **test_metrics, "coverage": coverage, "mean_width": mean_width},
    ]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", nargs="+", type=int, default=DEFAULT_WEEKS)
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Miscoverage rate (default 0.1 -> 90%% intervals).",
    )
    parser.add_argument(
        "--force-sklearn", action="store_true",
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
        all_rows.extend(train_one(K, args.alpha, use_xgb, log=print))

    leaderboard = pd.DataFrame(all_rows)
    out = DATA_INTERIM / "horizon_leaderboard.parquet"
    leaderboard.to_parquet(out, index=False)
    leaderboard.to_csv(DATA_INTERIM / "horizon_leaderboard.csv", index=False)

    print("\n=== horizon leaderboard ===")
    cols_to_show = ["week_k", "cutoff_doy", "split", "rmse", "mae", "r2",
                    "coverage", "mean_width", "n_features"]
    print(leaderboard[cols_to_show].to_string(index=False))
    print(f"\nsaved {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

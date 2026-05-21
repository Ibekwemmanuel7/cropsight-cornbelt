#!/usr/bin/env python3
"""
Build the in-season phenology + VCI feature matrix at a given forecast week K.

Phase 1 deliverable: phenology-only matrix (NDVI-derived). Weather and water
balance features wait for Phase 3 (ERA5 download).

Usage:
    python scripts/build_in_season_features.py --week 28
    python scripts/build_in_season_features.py --week 28 --forecast-year 2023

Output:
    data/interim/feature_matrix_k{K}_phenology.parquet  - all (fips, year) rows
                                                          for the chosen K.

The script performs three checks before writing:
    1. NDVI input is properly truncated.
    2. Phenology gates (eos_doy etc.) are NaN at this K.
    3. VCI baseline excludes the forecast year.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make the cropsight package importable when run from repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cropsight.features import phenology, leakage  # noqa: E402

DATA_RAW = REPO / "data" / "raw"
DATA_INTERIM = REPO / "data" / "interim"


def load_ndvi() -> pd.DataFrame:
    path = DATA_RAW / "modis" / "ndvi_county_2000_2023.parquet"
    if not path.exists():
        sys.exit(f"NDVI file not found: {path}. Run module 1 first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear.astype("int64")
    df["fips"] = df["fips"].astype(str)
    df["year"] = df["year"].astype("int64")
    return df


def truncate_ndvi(ndvi_df: pd.DataFrame, week_k: int) -> pd.DataFrame:
    cutoff_doy = week_k * 7
    return ndvi_df[ndvi_df["doy"] <= cutoff_doy].copy()


def build_phenology(
    ndvi_truncated: pd.DataFrame,
    week_k: int,
) -> pd.DataFrame:
    records = []
    by_county_year = ndvi_truncated.groupby(["fips", "year"], sort=False)
    for (fips, year), sub in tqdm(by_county_year, desc=f"Phenology K={week_k}"):
        rec = {"fips": fips, "year": int(year)}
        if len(sub) < 4:
            rec.update({k: np.nan for k in phenology.PHENOLOGY_FEATURE_KEYS})
        else:
            smooth = phenology.smooth_truncated(
                sub["doy"].to_numpy(float),
                sub["ndvi_mean"].to_numpy(float),
                week_k,
            )
            rec.update(phenology.extract_phenology_to_week(smooth, week_k))
        records.append(rec)
    return pd.DataFrame(records)


def build_vci(
    ndvi_df: pd.DataFrame,
    week_k: int,
    baseline_years: list[int],
) -> pd.DataFrame:
    vci_df = phenology.compute_vci_to_week(ndvi_df, week_k, baseline_years)
    return phenology.vci_window_means_to_week(vci_df, week_k)


def attach_static(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Year trend and soil features. Both are static within (fips, year)."""
    soil_path = DATA_RAW / "soil" / "ssurgo_topsoil_corn_belt.csv"
    if not soil_path.exists():
        print(f"warn: soil file missing ({soil_path}); skipping soil features")
        feature_df["year_trend"] = feature_df["year"] - 2000
        return feature_df

    soil = pd.read_csv(soil_path, dtype={"fips": str})
    keep = ["fips", "sand_pct", "clay_pct", "om_pct", "awc", "ph", "cec"]
    soil = soil[[c for c in keep if c in soil.columns]]
    feature_df = feature_df.merge(soil, on="fips", how="left")
    feature_df["year_trend"] = feature_df["year"] - 2000
    return feature_df


def attach_target(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Merge NASS county yield as the prediction target."""
    yld_path = DATA_RAW / "nass" / "corn_yield_county_2000_2023.csv"
    if not yld_path.exists():
        print(f"warn: NASS yield file missing; skipping target column")
        return feature_df
    yld = pd.read_csv(yld_path, dtype={"fips": str})
    yld = yld[["fips", "year", "yield_bu_acre"]]
    return feature_df.merge(yld, on=["fips", "year"], how="left")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build in-season phenology features.")
    parser.add_argument("--week", type=int, required=True, help="Forecast week K (e.g., 28).")
    parser.add_argument(
        "--baseline-years",
        nargs="+",
        type=int,
        default=list(range(2000, 2022)),
        help="VCI baseline years (default 2000-2021).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_INTERIM,
        help="Output directory (default data/interim/).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    K = args.week

    print(f"loading NDVI ...")
    ndvi_all = load_ndvi()
    print(f"  loaded {len(ndvi_all):,} rows, {ndvi_all['fips'].nunique()} fips, "
          f"{ndvi_all['year'].min()}-{ndvi_all['year'].max()}")

    ndvi_k = truncate_ndvi(ndvi_all, K)
    print(f"truncated to doy <= {K*7}: {len(ndvi_k):,} rows")

    leakage_warnings = leakage.audit_ndvi_input_doy(ndvi_k, K)
    if leakage_warnings:
        sys.exit("Input audit failed:\n  - " + "\n  - ".join(leakage_warnings))

    t0 = time.time()
    pheno = build_phenology(ndvi_k, K)
    print(f"phenology  : {pheno.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    vci = build_vci(ndvi_k, K, baseline_years=args.baseline_years)
    print(f"vci windows: {vci.shape} ({time.time()-t0:.1f}s)")

    feats = pheno.merge(vci, on=["fips", "year"], how="left")
    feats = attach_static(feats)
    feats = attach_target(feats)

    # Audits before writing
    static_cols = ["sand_pct", "clay_pct", "om_pct", "awc", "ph", "cec", "year_trend"]
    audit_warnings = []
    audit_warnings += leakage.audit_no_future_data(feats, K, strict=False)
    audit_warnings += leakage.audit_static_features(feats, static_cols)
    if audit_warnings:
        print("AUDIT WARNINGS:")
        for w in audit_warnings:
            print(f"  - {w}")
        sys.exit(1)
    else:
        print("audits PASSED")

    out_path = args.out_dir / f"feature_matrix_k{K}_phenology.parquet"
    feats.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print()

    print("== feature summary ==")
    feat_cols = [c for c in feats.columns if c not in ("fips", "year", "yield_bu_acre")]
    described = feats[feat_cols].describe().T[["count", "mean", "std", "min", "max"]]
    described = described.round(2)
    print(described.to_string())

    if "yield_bu_acre" in feats.columns:
        n_train = (feats["year"] <= 2021).sum()
        n_val = (feats["year"] == 2022).sum()
        n_test = (feats["year"] == 2023).sum()
        print(f"\nrow counts:  train={n_train}  val={n_val}  test={n_test}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

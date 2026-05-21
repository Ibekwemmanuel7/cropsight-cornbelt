#!/usr/bin/env python3
"""
Build in-season phenology + VCI feature matrices, one parquet per week K.

Phase 1/2 deliverable: phenology-only matrix (NDVI-derived). Weather and
water-balance features wait for Phase 3 (ERA5 download).

Usage:
    python scripts/build_in_season_features.py                       # all six Ks
    python scripts/build_in_season_features.py --weeks 28            # just K=28
    python scripts/build_in_season_features.py --weeks 20 28 36      # subset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cropsight.features import leakage, phenology  # noqa: E402

DATA_RAW = REPO / "data" / "raw"
DATA_INTERIM = REPO / "data" / "interim"

DEFAULT_WEEKS = [16, 20, 24, 28, 32, 36]


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


def build_phenology(ndvi_truncated: pd.DataFrame, week_k: int) -> pd.DataFrame:
    records = []
    by_county_year = ndvi_truncated.groupby(["fips", "year"], sort=False)
    for (fips, year), sub in tqdm(by_county_year, desc=f"phenology K={week_k}"):
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


def build_vci(ndvi_df: pd.DataFrame, week_k: int, baseline_years: list[int]) -> pd.DataFrame:
    vci_df = phenology.compute_vci_to_week(ndvi_df, week_k, baseline_years)
    return phenology.vci_window_means_to_week(vci_df, week_k)


def attach_static(feature_df: pd.DataFrame) -> pd.DataFrame:
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
    yld_path = DATA_RAW / "nass" / "corn_yield_county_2000_2023.csv"
    if not yld_path.exists():
        print("warn: NASS yield file missing; skipping target column")
        return feature_df
    yld = pd.read_csv(yld_path, dtype={"fips": str})
    yld = yld[["fips", "year", "yield_bu_acre"]]
    return feature_df.merge(yld, on=["fips", "year"], how="left")


def build_one_week(
    ndvi_all: pd.DataFrame,
    week_k: int,
    baseline_years: list[int],
    out_dir: Path,
) -> Path:
    print(f"\n----- K={week_k} (cutoff DOY {week_k * 7}) -----")

    ndvi_k = truncate_ndvi(ndvi_all, week_k)
    input_warnings = leakage.audit_ndvi_input_doy(ndvi_k, week_k)
    if input_warnings:
        sys.exit("Input audit failed:\n  - " + "\n  - ".join(input_warnings))

    t0 = time.time()
    pheno = build_phenology(ndvi_k, week_k)
    print(f"  phenology   {pheno.shape}  {time.time() - t0:.1f}s")

    t0 = time.time()
    vci = build_vci(ndvi_k, week_k, baseline_years=baseline_years)
    print(f"  vci windows {vci.shape}  {time.time() - t0:.1f}s")

    feats = pheno.merge(vci, on=["fips", "year"], how="left")
    feats = attach_static(feats)
    feats = attach_target(feats)

    static_cols = ["sand_pct", "clay_pct", "om_pct", "awc", "ph", "cec", "year_trend"]
    audit_warnings = []
    audit_warnings += leakage.audit_no_future_data(feats, week_k, strict=False)
    audit_warnings += leakage.audit_static_features(feats, static_cols)
    if audit_warnings:
        for w in audit_warnings:
            print(f"  AUDIT: {w}")
        sys.exit(1)

    out_path = out_dir / f"feature_matrix_k{week_k}_phenology.parquet"
    feats.to_parquet(out_path, index=False)
    non_static = [
        c for c in feats.columns if c not in ("fips", "year", "yield_bu_acre", *static_cols)
    ]
    n_obs_features = sum(int(feats[c].notna().any()) for c in non_static)
    print(
        f"  wrote {out_path.name}  ({out_path.stat().st_size / 1e6:.2f} MB)  "
        f"{n_obs_features}/{len(non_static)} dynamic features have data"
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build in-season phenology features.")
    parser.add_argument(
        "--weeks",
        "--week",
        nargs="+",
        type=int,
        default=DEFAULT_WEEKS,
        help="Forecast weeks to build (default: 16 20 24 28 32 36).",
    )
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

    print("loading NDVI ...")
    ndvi_all = load_ndvi()
    fips_count = ndvi_all["fips"].nunique()
    year_min = int(ndvi_all["year"].min())
    year_max = int(ndvi_all["year"].max())
    print(f"  loaded {len(ndvi_all):,} rows, {fips_count} fips, {year_min}-{year_max}")

    total_t0 = time.time()
    for K in args.weeks:
        build_one_week(ndvi_all, K, args.baseline_years, args.out_dir)
    print(f"\nall {len(args.weeks)} weeks built in {time.time() - total_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

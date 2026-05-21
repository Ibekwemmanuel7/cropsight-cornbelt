"""
cropsight.features.leakage
==========================

Leakage audit checks for in-season feature matrices.

The rule: at forecast week K, no feature value may incorporate any input data
point with DOY > K*7 or YEAR > the forecast year.

Usage
-----
    from cropsight.features import leakage
    warnings = leakage.audit_no_future_data(feature_matrix, week_k=28, strict=True)

Failure modes
-------------
- strict=True (default): any violation raises LeakageError. Use in CI and
  before model.fit().
- strict=False: returns the list of warnings without raising. Use for
  diagnostic reports.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


class LeakageError(AssertionError):
    """Raised when a leakage audit check fails."""


# Features that MUST be all-NaN until certain DOYs have passed.
# Key: feature column name. Value: earliest DOY at which the feature can have
# any non-NaN values.
PHENOLOGY_GATE_DOY: dict[str, int] = {
    "eos_doy":         287,   # End-of-season requires late-October data
    "season_length":   287,   # Requires both SOS and EOS
    "senescence_rate": 287,   # Requires post-peak observations
    "ndvi_vegetative": 130,   # Vegetative window starts DOY 130
    "ndvi_silking":    180,   # Silking window starts DOY 180
    "ndvi_grainfill":  220,   # Grainfill window starts DOY 220
    "vci_vegetative":  130,
    "vci_silking":     180,
    "vci_grainfill":   220,
}


def audit_no_future_data(
    feature_matrix: pd.DataFrame,
    week_k: int,
    *,
    strict: bool = True,
) -> list[str]:
    """
    Verify the feature matrix produced for forecast week K does not contain
    non-NaN values in features that should be unobservable at this K.

    Returns a list of human-readable warnings (empty = clean).
    Raises LeakageError if strict=True and any check fails.
    """
    cutoff_doy = week_k * 7
    warnings: list[str] = []

    for feat, min_doy in PHENOLOGY_GATE_DOY.items():
        if feat not in feature_matrix.columns:
            continue
        if cutoff_doy < min_doy:
            non_nan = int(feature_matrix[feat].notna().sum())
            if non_nan > 0:
                warnings.append(
                    f"feature `{feat}` has {non_nan} non-NaN rows at week K={week_k} "
                    f"(cutoff DOY={cutoff_doy}); expected all-NaN before DOY {min_doy}"
                )

    if strict and warnings:
        raise LeakageError("Leakage audit failed:\n  - " + "\n  - ".join(warnings))
    return warnings


def audit_static_features(
    feature_matrix: pd.DataFrame,
    static_columns: Iterable[str],
) -> list[str]:
    """
    Static features (soil, year_trend) must be constant within each (fips, year).
    Variance across K cuts for the same (fips, year) is a bug.
    """
    warnings: list[str] = []
    for col in static_columns:
        if col not in feature_matrix.columns:
            continue
        per_group_std = feature_matrix.groupby(["fips", "year"])[col].std(ddof=0)
        violations = int((per_group_std > 1e-9).sum())
        if violations > 0:
            warnings.append(
                f"static feature `{col}` varies within {violations} (fips, year) groups"
            )
    return warnings


def audit_train_val_test_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[str]:
    """No year may appear in more than one of train/val/test."""
    warnings: list[str] = []
    train_years = set(train_df["year"].unique())
    val_years = set(val_df["year"].unique())
    test_years = set(test_df["year"].unique())
    if train_years & val_years:
        warnings.append(
            f"train and val share years: {sorted(train_years & val_years)}"
        )
    if train_years & test_years:
        warnings.append(
            f"train and test share years: {sorted(train_years & test_years)}"
        )
    if val_years & test_years:
        warnings.append(
            f"val and test share years: {sorted(val_years & test_years)}"
        )
    return warnings


def audit_ndvi_input_doy(
    ndvi_input: pd.DataFrame,
    week_k: int,
) -> list[str]:
    """
    Verify the NDVI input dataframe passed into feature builders has already
    been filtered to doy <= week_k*7. Any leak here propagates everywhere.
    """
    cutoff_doy = week_k * 7
    n_leak = int((ndvi_input["doy"] > cutoff_doy).sum())
    if n_leak > 0:
        return [
            f"NDVI input contains {n_leak} rows with doy > {cutoff_doy} "
            f"(week K={week_k}); caller must filter input before feature derivation"
        ]
    return []


def audit_vci_baseline(
    baseline_years: Iterable[int],
    forecast_year: int,
) -> list[str]:
    """
    VCI baseline years must not include the forecast year or any future years.
    """
    warnings: list[str] = []
    baseline = set(int(y) for y in baseline_years)
    bad = {y for y in baseline if y >= forecast_year}
    if bad:
        warnings.append(
            f"VCI baseline_years contains forecast/future years: {sorted(bad)} "
            f"(forecast year = {forecast_year})"
        )
    return warnings

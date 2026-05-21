"""
Tests for cropsight.features.leakage.

These cover the audit functions that gate model fitting in the in-season
pipeline. The audits must catch obvious violations and must not flag a
correct feature matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropsight.features import leakage


def _clean_k28_matrix(n: int = 100) -> pd.DataFrame:
    """
    Synthetic matrix consistent with K=28 expectations: phenology gates that
    should be all-NaN are all-NaN; observed features are populated.
    """
    rng = np.random.default_rng(0)
    fips = [f"{17001 + i:05d}" for i in range(n)]
    years = rng.integers(2000, 2024, size=n)
    return pd.DataFrame({
        "fips": fips,
        "year": years,
        "peak_ndvi": rng.uniform(0.5, 0.95, n),
        "peak_doy":  rng.uniform(140, 196, n),
        "ndvi_silking": rng.uniform(0.5, 0.95, n),
        "ndvi_grainfill": np.nan,   # gated at K=28
        "eos_doy": np.nan,           # gated at K=28
        "season_length": np.nan,     # gated at K=28
        "senescence_rate": np.nan,   # gated at K=28
        "year_trend": years - 2000,
        "awc": rng.uniform(0.14, 0.21, n),
    })


def test_clean_matrix_passes_audit():
    df = _clean_k28_matrix()
    warnings = leakage.audit_no_future_data(df, week_k=28, strict=False)
    assert warnings == []


def test_leaked_grainfill_raises():
    df = _clean_k28_matrix()
    df.loc[df.index[0], "ndvi_grainfill"] = 0.8  # leak
    with pytest.raises(leakage.LeakageError):
        leakage.audit_no_future_data(df, week_k=28, strict=True)


def test_leaked_eos_raises():
    df = _clean_k28_matrix()
    df.loc[df.index[:3], "eos_doy"] = [270, 280, 290]  # leaks
    with pytest.raises(leakage.LeakageError):
        leakage.audit_no_future_data(df, week_k=28, strict=True)


def test_audit_static_features_detects_variance():
    """If soil varies within (fips, year), that's a bug."""
    df = pd.DataFrame({
        "fips": ["17001", "17001", "17001"],
        "year": [2020, 2020, 2020],
        "awc": [0.18, 0.20, 0.18],  # varies within group - violation
        "year_trend": [20, 20, 20],
    })
    warnings = leakage.audit_static_features(df, ["awc", "year_trend"])
    assert any("awc" in w for w in warnings)
    assert all("year_trend" not in w for w in warnings)


def test_audit_train_val_test_split_detects_overlap():
    train = pd.DataFrame({"year": [2020, 2021]})
    val = pd.DataFrame({"year": [2021]})  # overlaps with train
    test = pd.DataFrame({"year": [2023]})
    warnings = leakage.audit_train_val_test_split(train, val, test)
    assert any("train and val" in w for w in warnings)


def test_audit_ndvi_input_doy_detects_leak():
    ndvi = pd.DataFrame({"doy": [60, 100, 200, 250]})  # 250 > 196 = K*7 at K=28
    warnings = leakage.audit_ndvi_input_doy(ndvi, week_k=28)
    assert len(warnings) == 1
    assert "doy >" in warnings[0]


def test_audit_vci_baseline_detects_forecast_year_in_baseline():
    warnings = leakage.audit_vci_baseline(
        baseline_years=[2018, 2019, 2020, 2021, 2022],  # 2022 is the forecast year
        forecast_year=2022,
    )
    assert any("2022" in w for w in warnings)


def test_audit_vci_baseline_clean():
    warnings = leakage.audit_vci_baseline(
        baseline_years=list(range(2000, 2022)),
        forecast_year=2022,
    )
    assert warnings == []


def test_phenology_gate_doy_completeness():
    """Every gated feature must have an entry in PHENOLOGY_GATE_DOY."""
    gates = leakage.PHENOLOGY_GATE_DOY
    # Smoke check: these are the features that must be gated
    for must_have in ("eos_doy", "ndvi_grainfill", "vci_grainfill", "senescence_rate"):
        assert must_have in gates

"""
cropsight.features.weather
==========================

Weather-derived features at week K. Status: STUB (Phase 3).

This module is intentionally not implemented yet. Real implementation is
blocked on the ERA5 download (scripts/download_era5.py). The existing
state-level proxy weather generator in module 2 cell 10 has no temporal
structure - it produces full-season aggregates per (fips, year, state) - and
naive linear scaling to week K would introduce more error than dropping the
features entirely.

Decision (documented in docs/specs/in_season_pipeline.md, design point 4):
Phase 1 ships in-season feature matrices WITHOUT weather features. Phase 3
swaps in real ERA5 and implements the functions below.

Planned API
-----------
    gdd_to_week(era5_county_daily, week_k, base_c=10.0, ceiling_c=30.0) -> pd.DataFrame
    precip_to_week(era5_county_daily, week_k) -> pd.DataFrame
    heat_stress_days_to_week(era5_county_daily, week_k, threshold_c=35.0) -> pd.DataFrame
    vpd_to_week(era5_county_daily, week_k) -> pd.DataFrame
    spi_to_week(era5_county_daily, week_k, baseline_years) -> pd.DataFrame

All return one row per (fips, year) with cumulative-to-DOY=K*7 values.
"""

from __future__ import annotations


class WeatherPhase3Stub(NotImplementedError):
    """Raised when weather features are requested before Phase 3 lands."""


def _not_implemented(name: str):
    raise WeatherPhase3Stub(
        f"{name} is Phase 3 work, blocked on ERA5 download. "
        f"See docs/specs/in_season_pipeline.md and scripts/download_era5.py."
    )


def gdd_to_week(*args, **kwargs):
    _not_implemented("gdd_to_week")


def precip_to_week(*args, **kwargs):
    _not_implemented("precip_to_week")


def heat_stress_days_to_week(*args, **kwargs):
    _not_implemented("heat_stress_days_to_week")


def vpd_to_week(*args, **kwargs):
    _not_implemented("vpd_to_week")


def spi_to_week(*args, **kwargs):
    _not_implemented("spi_to_week")

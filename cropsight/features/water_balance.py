"""
cropsight.features.water_balance
================================

DSSAT-proxy soil water balance features at week K. Status: STUB (Phase 3).

The water balance integrates daily over the elapsed growing season, taking
precip and PET as inputs. PET requires daily T2m and (Hargreaves-Samani)
extraterrestrial radiation. Both are blocked on the ERA5 download.

Phase 3 implementation will port `run_water_balance` from module 2 cell 12,
parameterizing the integration window by week_k:
    n_days = max(0, week_k * 7 - planting_doy)
    precip = sum(daily_precip[planting_doy : planting_doy + n_days])
    pet    = sum(daily_pet   [planting_doy : planting_doy + n_days])
    ... soil bucket update ...

Outputs at week K:
    aet_to_week_mm, pet_to_week_mm
    water_stress_frac (== AET / PET) - this is also the PINN physics constraint
    soil_water_deficit_mm, water_stress_days, drought_index
"""

from __future__ import annotations


class WaterBalancePhase3Stub(NotImplementedError):
    """Raised when DSSAT-proxy features are requested before Phase 3 lands."""


def run_water_balance_to_week(*args, **kwargs):
    raise WaterBalancePhase3Stub(
        "run_water_balance_to_week is Phase 3 work, blocked on ERA5 download. "
        "See docs/specs/in_season_pipeline.md."
    )

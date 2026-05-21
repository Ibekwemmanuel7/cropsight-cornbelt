"""
cropsight.features
==================

Feature engineering for in-season yield forecasting. The package mirrors the
four feature groups from module 2:

    phenology       - NDVI-derived (smooth, SOS/EOS, peak, integrated, VCI, ...)
    weather         - GDD, SPI, VPD, heat-stress (Phase 3 - blocked on ERA5)
    water_balance   - DSSAT-proxy AET/PET (Phase 3 - blocked on real weather)
    leakage         - audit checks: no feature may use data with DOY > K*7

All feature builders take a week_k parameter (forecast week, integer 16..40)
and return values restricted to the observable window doy <= week_k * 7.
"""

from . import leakage, phenology, water_balance, weather

__all__ = ["leakage", "phenology", "water_balance", "weather"]

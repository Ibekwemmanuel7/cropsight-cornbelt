#!/usr/bin/env python3
"""
Download ERA5 daily reanalysis for the US Corn Belt (IA / IL / IN) over the
2000-2023 growing seasons (Mar-Nov). Replaces the proxy weather features used
in earlier versions of CropSight with real gridded reanalysis data.

Variables downloaded (single-level, hourly, aggregated to daily downstream):

    2m_temperature                       -> T2m (degC)
    2m_dewpoint_temperature              -> Td2m, used with T2m for VPD
    total_precipitation                  -> daily precip (m -> mm)
    surface_solar_radiation_downwards    -> SSRD, used for PET (Hargreaves)

Approximate footprint:
    bbox     : 47N / -98W / 36S / -84E   (Corn Belt)
    months   : Mar-Nov (growing season only)
    storage  : ~50 MB per (var, year) -> ~5 GB total for 4 vars * 24 years
    runtime  : CDS queues each request 5-30 min; expect ~12-30 hours total
               wall-clock. The script is resumable - already-downloaded files
               are skipped.

Usage:
    # one-time CDS setup (see ~/.cdsapirc instructions in README Section 0)
    python scripts/download_era5.py

    # subset for testing:
    python scripts/download_era5.py --years 2012 --vars 2m_temperature

    # download in parallel (CDS allows ~4 concurrent requests per user):
    python scripts/download_era5.py --workers 4

Output files land in data/raw/era5/era5_<varshort>_<year>.nc and can be loaded
with xarray.open_dataset().
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---- config ---------------------------------------------------------------

# Corn Belt bbox: [North, West, South, East] degrees
CORN_BELT_AREA = [47.0, -98.0, 36.0, -84.0]

# Variable map: CDS variable name -> short label used in output filenames
VARIABLES = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "total_precipitation": "tp",
    "surface_solar_radiation_downwards": "ssrd",
}

YEARS_DEFAULT = list(range(2000, 2024))
MONTHS = [f"{m:02d}" for m in range(3, 12)]  # Mar-Nov
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(0, 24)]

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "era5"


# ---- helpers --------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("era5")


def _check_cdsapirc() -> None:
    """Fail fast if ~/.cdsapirc is missing."""
    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        sys.exit(
            "ERROR: ~/.cdsapirc not found. Register at https://cds.climate.copernicus.eu, "
            "then create the file as described in README Section 0.3."
        )


def _import_cdsapi():
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        sys.exit("ERROR: cdsapi is not installed. Run: pip install cdsapi")
    return __import__("cdsapi")


def _retrieve_one(
    client,
    variable: str,
    short: str,
    year: int,
    out_dir: Path,
    log: logging.Logger,
) -> tuple[str, int, Path, bool]:
    """
    Download one (variable, year) combination if not already present.
    Returns (variable, year, path, was_downloaded).
    """
    out = out_dir / f"era5_{short}_{year}.nc"
    if out.exists() and out.stat().st_size > 0:
        log.info("skip   %s %s (already %.1f MB)", year, short, out.stat().st_size / 1e6)
        return variable, year, out, False

    request = {
        "product_type": "reanalysis",
        "variable": variable,
        "year": str(year),
        "month": MONTHS,
        "day": DAYS,
        "time": HOURS,
        "area": CORN_BELT_AREA,
        "format": "netcdf",
    }

    log.info("queue  %s %s -> CDS", year, short)
    t0 = time.time()
    client.retrieve("reanalysis-era5-single-levels", request, str(out))
    elapsed = time.time() - t0
    size_mb = out.stat().st_size / 1e6
    log.info("done   %s %s  %.1f MB in %.0fs", year, short, size_mb, elapsed)
    return variable, year, out, True


# ---- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=YEARS_DEFAULT,
        help="Years to download (default: 2000-2023).",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        choices=list(VARIABLES.keys()),
        default=list(VARIABLES.keys()),
        help="CDS variables to download (default: all four).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: data/raw/era5/).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent CDS requests (default: 1). CDS quota allows ~4.",
    )
    args = parser.parse_args()

    log = _setup_logging()
    _check_cdsapirc()
    cdsapi = _import_cdsapi()

    args.out.mkdir(parents=True, exist_ok=True)
    log.info("output dir: %s", args.out)
    log.info("variables : %s", ", ".join(args.vars))
    log.info("years     : %d - %d (%d years)", min(args.years), max(args.years), len(args.years))
    log.info("workers   : %d", args.workers)

    jobs = [(v, VARIABLES[v], y) for v in args.vars for y in args.years]
    total = len(jobs)
    log.info("planning %d (variable, year) downloads", total)

    if args.workers == 1:
        # Serial: one shared client, deterministic order
        client = cdsapi.Client()
        downloaded = 0
        for i, (v, s, y) in enumerate(jobs, 1):
            log.info("[ %3d / %3d ]  %s %s", i, total, y, s)
            _, _, _, did = _retrieve_one(client, v, s, y, args.out, log)
            downloaded += int(did)
        log.info("finished: %d downloaded, %d skipped", downloaded, total - downloaded)
        return 0

    # Parallel: one client per thread, careful with CDS rate limits
    def task(job):
        v, s, y = job
        client = cdsapi.Client()
        return _retrieve_one(client, v, s, y, args.out, log)

    downloaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, j): j for j in jobs}
        for fut in as_completed(futures):
            _, _, _, did = fut.result()
            downloaded += int(did)
    log.info("finished: %d downloaded, %d skipped", downloaded, total - downloaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

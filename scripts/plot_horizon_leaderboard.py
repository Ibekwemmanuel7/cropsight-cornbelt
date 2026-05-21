#!/usr/bin/env python3
"""
Plot the in-season horizon leaderboard: RMSE, conformal coverage, mean interval
width, and feature count, each as a function of forecast week K. Saves a PNG
under docs/images/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_INTERIM = REPO / "data" / "interim"
DOCS_IMAGES = REPO / "docs" / "images"


def main() -> int:
    src = DATA_INTERIM / "horizon_leaderboard.parquet"
    if not src.exists():
        sys.exit(f"leaderboard not found: {src} (run train_in_season_models.py first)")

    df = pd.read_parquet(src)
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

    val = df[df["split"] == "val"].sort_values("week_k")
    test = df[df["split"] == "test"].sort_values("week_k")
    backend = df["backend"].iloc[0]
    alpha = df["alpha"].iloc[0] if "alpha" in df.columns else 0.1

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Panel A: RMSE
    ax = axes[0, 0]
    ax.plot(val["week_k"], val["rmse"], marker="o", linewidth=2.0,
            label="Val (2022)", color="#1f3864")
    ax.plot(test["week_k"], test["rmse"], marker="s", linewidth=2.0,
            label="Test (2023)", color="#c00000")
    ax.set_xlabel("Forecast week K (DOY = K * 7)")
    ax.set_ylabel("RMSE (bu/acre)")
    ax.set_title("(a) Forecast RMSE by week K")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xticks([16, 20, 24, 28, 32, 36])

    # Panel B: conformal coverage
    ax = axes[0, 1]
    ax.plot(test["week_k"], test["coverage"], marker="s", linewidth=2.0,
            color="#2e7d32", label="Empirical coverage (2023)")
    target = 1.0 - alpha
    ax.axhline(target, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"Target {target:.0%}")
    ax.set_xlabel("Forecast week K")
    ax.set_ylabel(f"{target:.0%} conformal coverage")
    ax.set_title("(b) Conformal interval coverage")
    ax.set_ylim(0.75, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xticks([16, 20, 24, 28, 32, 36])

    # Panel C: mean interval width
    ax = axes[1, 0]
    ax.plot(test["week_k"], test["mean_width"], marker="s", linewidth=2.0,
            color="#7b1fa2")
    ax.set_xlabel("Forecast week K")
    ax.set_ylabel("Mean interval width (bu/acre)")
    ax.set_title("(c) Conformal interval width")
    ax.grid(alpha=0.3)
    ax.set_xticks([16, 20, 24, 28, 32, 36])

    # Panel D: feature count
    ax = axes[1, 1]
    counts = df.drop_duplicates("week_k")[["week_k", "n_features"]].sort_values("week_k")
    ax.bar(counts["week_k"], counts["n_features"], width=2.5,
           color="#4472c4", alpha=0.8)
    ax.set_xlabel("Forecast week K")
    ax.set_ylabel("non-NaN feature count")
    ax.set_title("(d) Features available at week K")
    ax.grid(alpha=0.3, axis="y")
    ax.set_xticks([16, 20, 24, 28, 32, 36])

    fig.suptitle(
        f"In-season corn yield forecast — accuracy and conformal uncertainty by horizon  "
        f"({backend}, alpha={alpha}, 2023 test set)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    out = DOCS_IMAGES / "horizon_leaderboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

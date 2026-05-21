#!/usr/bin/env python3
"""
Plot the in-season horizon leaderboard: RMSE / MAE / R2 as a function of
forecast week K. Saves a PNG to docs/images/.
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: RMSE vs K
    ax = axes[0]
    ax.plot(val["week_k"], val["rmse"], marker="o", linewidth=2.0,
            label="Val (2022)", color="#1f3864")
    ax.plot(test["week_k"], test["rmse"], marker="s", linewidth=2.0,
            label="Test (2023)", color="#c00000")
    ax.set_xlabel("Forecast week K (DOY = K * 7)")
    ax.set_ylabel("RMSE (bu/acre)")
    ax.set_title("Forecast RMSE by week K")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xticks([16, 20, 24, 28, 32, 36])
    for k, doy_label in [
        (16, "pre-plant"), (20, "planting"), (24, "vegetative"),
        (28, "pre-silk"), (32, "silking/grainfill"), (36, "grainfill"),
    ]:
        ax.axvline(k, color="gray", alpha=0.15, linestyle=":")

    # Panel 2: MAE
    ax = axes[1]
    ax.plot(val["week_k"], val["mae"], marker="o", linewidth=2.0,
            label="Val (2022)", color="#1f3864")
    ax.plot(test["week_k"], test["mae"], marker="s", linewidth=2.0,
            label="Test (2023)", color="#c00000")
    ax.set_xlabel("Forecast week K")
    ax.set_ylabel("MAE (bu/acre)")
    ax.set_title("Forecast MAE by week K")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xticks([16, 20, 24, 28, 32, 36])

    # Panel 3: feature count
    ax = axes[2]
    counts = df.drop_duplicates("week_k")[["week_k", "n_features"]].sort_values("week_k")
    ax.bar(counts["week_k"], counts["n_features"], width=2.5,
           color="#4472c4", alpha=0.8)
    ax.set_xlabel("Forecast week K")
    ax.set_ylabel("non-NaN feature count")
    ax.set_title("Features available at week K")
    ax.grid(alpha=0.3, axis="y")
    ax.set_xticks([16, 20, 24, 28, 32, 36])

    fig.suptitle(
        f"In-season corn yield forecast accuracy by horizon  ({backend} preview)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = DOCS_IMAGES / "horizon_leaderboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

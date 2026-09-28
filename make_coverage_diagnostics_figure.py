"""Render publication-ready diagnostics from the pipeline's attrition reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import config


def _read_parquet(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else [root]
    if not files:
        raise FileNotFoundError(f"No Parquet files found at {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=config.PIPELINE_REPORTS_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = args.reports_dir
    speed = _read_parquet(reports / "speed_derivation_exclusions.parquet")
    unmatched = _read_parquet(reports / "unmatched_distance.parquet")

    speed_totals = speed.groupby("speed_derivation_reason", as_index=False)["point_count"].sum()
    unmatched_totals = unmatched.groupby("distance_bucket", as_index=False)[
        "unmatched_point_count"
    ].sum()
    monthly = unmatched.groupby(["year", "month"], as_index=False)[
        "unmatched_point_count"
    ].sum()
    monthly["period"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    axes[0].barh(speed_totals["speed_derivation_reason"], speed_totals["point_count"], color="#C44E52")
    axes[0].set_title("Speed derivation outcomes")
    axes[0].set_xlabel("Telemetry points")
    axes[1].bar(unmatched_totals["distance_bucket"], unmatched_totals["unmatched_point_count"], color="#4C72B0")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Unmatched points: nearest OSM road")
    axes[1].set_ylabel("Telemetry points")
    axes[2].plot(monthly["period"], monthly["unmatched_point_count"], marker="o", color="#55A868")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].set_title("Monthly unmatched points")
    axes[2].set_ylabel("Telemetry points")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

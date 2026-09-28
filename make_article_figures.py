"""Create release-ready figures for the Kraków ambulance speed data descriptor.

The script consumes Gold tables and aggregate technical reports only. It never
writes or plots raw GPS coordinates, timestamps, or source identifiers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from config import config

COLORS = {
    "blue": "#2d9dca",
    "navy": "#1b3e50",
    "pale": "#dceef5",
    "orange": "#e67e22",
    "green": "#3d8c6b",
    "grey": "#aab7bd",
    "dark_grey": "#64747c",
    "red": "#c44e52",
}


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file or a Spark-style directory deterministically."""
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"No Parquet files found at {path}")
    return pd.concat((pd.read_parquet(item) for item in paths), ignore_index=True)


def _attrition_value(frame: pd.DataFrame, category: str) -> int:
    rows = frame.loc[
        (frame["report_type"] == "ATTRITION") & (frame["category"] == category),
        "count",
    ]
    if rows.empty:
        raise ValueError(f"Missing attrition category {category!r}")
    return int(rows.iloc[0])


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {output}")


def figure_1_extent_and_coverage(gold_dir: Path, run_manifest: Path, output: Path) -> None:
    """Map aggregate road-feature support without exposing telemetry points."""
    base = gpd.read_parquet(gold_dir / "krakow_roads_base.parquet").to_crs(4326)
    profiles = pd.read_parquet(gold_dir / "speeds_by_time_of_day.parquet")
    support = profiles.groupby("road_feature_id", as_index=False).agg(
        n_samples=("n_samples", "sum"),
        supported=("meets_min_sample_count", "any"),
    )
    roads = base.merge(support, on="road_feature_id", how="left")
    roads["coverage"] = np.select(
        [roads["n_samples"].isna(), roads["supported"].fillna(False)],
        ["No observed profile", "At least one supported profile"],
        default="Observed, no supported profile",
    )
    if run_manifest.is_file():
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
        bbox = manifest["scientific_parameters"]["official_study_aoi_bbox_wgs84"]
    else:
        bbox = config.KRAKOW_BBOX

    figure, axis = plt.subplots(figsize=(8.5, 8))
    order = [
        ("No observed profile", COLORS["grey"]),
        ("Observed, no supported profile", COLORS["orange"]),
        ("At least one supported profile", COLORS["blue"]),
    ]
    for label, color in order:
        roads.loc[roads["coverage"] == label].plot(
            ax=axis, color=color, linewidth=0.24, label=label
        )
    axis.add_patch(
        Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            fill=False, edgecolor=COLORS["navy"], linewidth=1.1,
            linestyle="--", label="Official Kraków study bbox",
        )
    )
    axis.set_title("Road-feature coverage in the released processing envelope", loc="left", weight="bold")
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.legend(loc="lower left", frameon=True, fontsize=8)
    axis.set_aspect("equal")
    _save(figure, output)


def figure_2_workflow(output: Path) -> None:
    """Render a data-flow figure from the documented release architecture."""
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.set_axis_off()
    boxes = [
        (0.03, 0.62, 0.20, 0.22, "Reference inputs", "OpenStreetMap road network\nPinned ERA5 / Open-Meteo weather", COLORS["pale"]),
        (0.29, 0.62, 0.18, 0.22, "Bronze", "Schema, timestamps, coordinate\nchecks and source-field audit", "#fff1dc"),
        (0.53, 0.62, 0.18, 0.22, "Silver", "GPS-primary speed, road matching\n15 m primary / sensitivity diagnostics", "#e5f4f8"),
        (0.77, 0.70, 0.19, 0.16, "Gold: relational", "Base roads + season / month /\nweekday / time-of-day profiles", "#e7f4eb"),
        (0.77, 0.43, 0.19, 0.16, "Gold: flat", "Observed joint combinations\nfor interoperability", "#f1f4f5"),
    ]
    for x, y, w, h, title, body, fill in boxes:
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fill, ec=COLORS["navy"], lw=1.1))
        axis.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", weight="bold", color=COLORS["navy"], fontsize=10)
        axis.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=8.4)
    for start, end in [((0.23, 0.73), (0.29, 0.73)), ((0.47, 0.73), (0.53, 0.73)), ((0.71, 0.73), (0.77, 0.78)), ((0.71, 0.70), (0.77, 0.51))]:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=COLORS["dark_grey"]))
    axis.text(0.03, 0.20, "Release controls: provenance manifests • configuration fingerprint chain • schema and numeric reconciliation • checksums • independent verification", fontsize=9, color=COLORS["dark_grey"])
    axis.set_title("Processing workflow for the aggregate release", loc="left", weight="bold", color=COLORS["navy"])
    _save(figure, output)


def figure_3_temporal_coverage_and_attrition(
    gold_dir: Path, reports_dir: Path, output: Path
) -> None:
    """Show coverage gaps and processing retention from aggregate reports."""
    bronze = _read_parquet(reports_dir / "bronze_attrition.parquet")
    silver = _read_parquet(reports_dir / "silver_attrition.parquet")
    raw_monthly = bronze.loc[bronze["report_type"] == "TEMPORAL_COVERAGE"].copy()
    raw_monthly["period"] = pd.to_datetime(dict(year=raw_monthly.year, month=raw_monthly.month, day=1))
    matched_monthly = silver.loc[silver["report_type"] == "SILVER_DISTRIBUTION"].copy()
    matched_monthly["period"] = pd.to_datetime(dict(year=matched_monthly.year, month=matched_monthly.month, day=1))
    matched_monthly = matched_monthly.groupby("period", as_index=False)["count"].sum()
    raw_monthly = raw_monthly.groupby("period", as_index=False)["count"].sum()

    flow = pd.DataFrame({
        "stage": ["Raw rows", "Deduplicated\nvalid rows", "Sensitivity-radius\nmatches", "Gold included"],
        "count": [
            _attrition_value(bronze, "raw_rows"),
            _attrition_value(bronze, "deduplicated_valid_rows"),
            _attrition_value(silver, "matched_rows_at_sensitivity_max_radius"),
            int(
                pd.read_parquet(
                    gold_dir / "speeds_by_season.parquet", columns=["n_samples"]
                )["n_samples"].sum()
            ),
        ],
    })
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})
    axes[0].plot(raw_monthly.period, raw_monthly["count"], marker="o", ms=3, label="Raw GPS records", color=COLORS["dark_grey"])
    axes[0].plot(matched_monthly.period, matched_monthly["count"], marker="o", ms=3, label="Silver matched records", color=COLORS["blue"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Records (log scale)")
    axes[0].set_title("Monthly telemetry coverage", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.25)
    bars = axes[1].barh(flow.stage, flow["count"], color=[COLORS["grey"], COLORS["blue"], COLORS["orange"], COLORS["green"]])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Records (log scale)")
    axes[1].set_title("Processing retention", loc="left", weight="bold")
    for bar, value in zip(bars, flow["count"]):
        axes[1].text(value, bar.get_y() + bar.get_height() / 2, f" {value:,.0f}", va="center", fontsize=8)
    axes[1].grid(axis="x", alpha=0.25)
    _save(figure, output)


def figure_4_sampling_support(gold_dir: Path, output: Path) -> None:
    """Show road-level sample heterogeneity and support coverage."""
    tables = {
        "Season": "speeds_by_season.parquet",
        "Month": "speeds_by_month.parquet",
        "Weekday": "speeds_by_day_of_week.parquet",
        "Time of day": "speeds_by_time_of_day.parquet",
    }
    profiles = {name: pd.read_parquet(gold_dir / filename) for name, filename in tables.items()}
    representative = profiles["Time of day"]
    per_road = representative.groupby("road_feature_id", as_index=False)["n_samples"].sum()
    support_rates = pd.DataFrame([
        {
            "table": name,
            "row_support_rate": frame["meets_min_sample_count"].mean() * 100,
            "roads_with_supported": frame.loc[frame["meets_min_sample_count"]].road_feature_id.nunique(),
        }
        for name, frame in profiles.items()
    ])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].hist(np.log10(per_road.n_samples), bins=40, color=COLORS["blue"], edgecolor="white")
    axes[0].set_title("Road-level observation support", loc="left", weight="bold")
    axes[0].set_xlabel("log10(total n_samples per road; time-of-day table)")
    axes[0].set_ylabel("Road features")
    axes[0].grid(axis="y", alpha=0.25)
    bars = axes[1].bar(support_rates.table, support_rates.row_support_rate, color=COLORS["green"])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Profile rows meeting n_samples ≥ 10 (%)")
    axes[1].set_title("Support by temporal representation", loc="left", weight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, support_rates.row_support_rate):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
    _save(figure, output)


def figure_5_matching_diagnostics(reports_dir: Path, output: Path) -> None:
    """Render aggregate diagnostics from the controlled validation sample."""
    sample_path = reports_dir / "manual_15m_review" / "map_matching_validation_sample.parquet"
    sample = pd.read_parquet(sample_path)
    distance = pd.to_numeric(sample["dist_to_axis"], errors="coerce").dropna()
    candidates = pd.to_numeric(sample["candidate_count"], errors="coerce").fillna(0)
    candidate_bins = pd.Series(np.select([candidates == 1, candidates == 2], ["1", "2"], default="3+"))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].hist(distance, bins=np.arange(0, 42, 2), color=COLORS["blue"], edgecolor="white")
    for threshold in (5, 10, 15):
        axes[0].axvline(threshold, color=COLORS["navy"], linestyle="--", linewidth=1)
    axes[0].set_xlabel("Distance to selected road centreline (m)")
    axes[0].set_ylabel("Validation-sample records")
    axes[0].set_title("Selected-road distance", loc="left", weight="bold")
    counts = candidate_bins.value_counts().reindex(["1", "2", "3+"], fill_value=0)
    axes[1].bar(counts.index, counts.values / counts.sum() * 100, color=[COLORS["green"], COLORS["orange"], COLORS["red"]])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Validation-sample records (%)")
    axes[1].set_xlabel("Candidate roads within matching radius")
    axes[1].set_title("Candidate ambiguity", loc="left", weight="bold")
    axes[1].text(0.02, 0.96, "Internal diagnostic only; not manual ground truth", transform=axes[1].transAxes, va="top", fontsize=8, color=COLORS["dark_grey"])
    _save(figure, output)


def figure_6_illustrative_profiles(gold_dir: Path, output: Path) -> None:
    """Create descriptive weighted speed profiles with aligned support bars."""
    specs = [
        ("Season", "speeds_by_season.parquet", "season", ["WINTER", "SPRING", "SUMMER", "AUTUMN"]),
        ("Month", "speeds_by_month.parquet", "month", list(range(1, 13))),
        ("Weekday", "speeds_by_day_of_week.parquet", "day_of_week", list(range(1, 8))),
        ("Time of day", "speeds_by_time_of_day.parquet", "time_of_day", ["PEAK_MORNING", "OFF_PEAK_DAY", "PEAK_EVENING", "NIGHT"]),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(15, 6), gridspec_kw={"height_ratios": [2.2, 1]}, constrained_layout=True)
    for index, (title, filename, key, order) in enumerate(specs):
        frame = pd.read_parquet(gold_dir / filename)
        grouped = frame.groupby(key, as_index=False).apply(
            lambda x: pd.Series({
                "weighted_speed": np.average(x.avg_travel_speed_kmh, weights=x.n_samples),
                "n_samples": x.n_samples.sum(),
            }), include_groups=False,
        ).set_index(key).reindex(order).reset_index()
        x = np.arange(len(grouped))
        axes[0, index].plot(x, grouped.weighted_speed, marker="o", color=COLORS["blue"])
        axes[0, index].set_title(title, weight="bold", fontsize=10)
        axes[0, index].set_ylabel("km/h" if index == 0 else "")
        axes[0, index].set_xticks(x, [])
        axes[0, index].grid(axis="y", alpha=0.25)
        axes[1, index].bar(x, grouped.n_samples, color=COLORS["green"])
        axes[1, index].set_yscale("log")
        axes[1, index].set_ylabel("n" if index == 0 else "")
        axes[1, index].set_xticks(x, [str(v).replace("_", "\n") for v in grouped[key]], fontsize=7)
        axes[1, index].grid(axis="y", alpha=0.25)
    figure.suptitle("Descriptive observation-weighted operating-speed profiles", x=0.01, ha="left", weight="bold", color=COLORS["navy"])
    _save(figure, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=config.GOLD_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.PIPELINE_REPORTS_DIR)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    gold_dir = args.gold_dir.resolve()
    reports_dir = args.reports_dir.resolve()
    manifest = args.run_manifest or reports_dir / "pipeline_run_manifest.json"
    outputs = args.output_dir.resolve()
    figure_1_extent_and_coverage(gold_dir, manifest, outputs / "figure_1_extent_and_coverage.png")
    figure_2_workflow(outputs / "figure_2_processing_workflow.png")
    figure_3_temporal_coverage_and_attrition(
        gold_dir,
        reports_dir,
        outputs / "figure_3_temporal_coverage_and_attrition.png",
    )
    figure_4_sampling_support(gold_dir, outputs / "figure_4_sampling_support.png")
    figure_5_matching_diagnostics(reports_dir, outputs / "figure_5_matching_diagnostics.png")
    figure_6_illustrative_profiles(gold_dir, outputs / "figure_6_illustrative_profiles.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

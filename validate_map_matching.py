"""Reproducible, stratified validation of the corrected Silver map matches."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from config import config
from pipeline_utils import utc_now_iso, write_json_atomic

logger = logging.getLogger(__name__)

SAMPLE_COLUMNS = [
    "point_id",
    "year",
    "month",
    "timestamp_utc",
    "GPS_LAT",
    "GPS_LON",
    "source_entity_id",
    "road_feature_id",
    "osm_id",
    "highway",
    "signal_mode",
    "weather_category",
    "weather_join_status",
    "speed_source",
    "travel_direction",
    "calc_speed_kmh",
    "dist_to_axis",
    "azimuth_diff",
    "heading_alignment_diff",
    "match_score",
    "candidate_count",
    "candidate_count_primary_radius",
    "candidate_count_radius_1",
    "candidate_count_radius_2",
    "candidate_count_radius_3",
    "within_primary_match_radius",
    "selected_match_radius_m",
    "match_radius_tier",
    "score_margin",
    "dist_margin_m",
    "is_high_speed_iqr_outlier",
]


def _lowest_point_ids(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Return a stable lexicographic bottom-k of SHA-256 point identifiers.

    ``point_id`` is intentionally a string digest.  ``DataFrame.nsmallest``
    accepts only numeric dtypes, so use a stable lexical sort instead.
    """
    return frame.sort_values("point_id", kind="stable").head(count)


def deterministic_stratified_sample(
    silver_dir: Path,
    per_stratum: int = 10_000,
    strata: tuple[str, ...] = ("year", "highway"),
    batch_size: int = 131_072,
) -> pd.DataFrame:
    """Select the lowest deterministic point hashes within every stratum.

    ``point_id`` is a SHA-256 digest of source fields. Selecting the lexicographically
    smallest hashes is equivalent to a reproducible uniform bottom-k sample without
    depending on file order, which fixes the previous first-500k bias.
    """
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    dataset = ds.dataset(str(silver_dir), format="parquet", partitioning="hive")
    missing = set(SAMPLE_COLUMNS).difference(dataset.schema.names)
    if missing:
        raise ValueError(f"Silver validation schema is missing: {sorted(missing)}")

    reservoirs: dict[tuple, pd.DataFrame] = {}
    for batch in dataset.to_batches(columns=SAMPLE_COLUMNS, batch_size=batch_size):
        frame = batch.to_pandas()
        frame = frame.dropna(subset=["point_id", *strata])
        for key, group in frame.groupby(list(strata), sort=False, dropna=False):
            normalized_key = key if isinstance(key, tuple) else (key,)
            candidate = _lowest_point_ids(group, min(per_stratum, len(group)))
            previous = reservoirs.get(normalized_key)
            if previous is not None:
                candidate = _lowest_point_ids(
                    pd.concat([previous, candidate], ignore_index=True), per_stratum
                )
            reservoirs[normalized_key] = candidate
    if not reservoirs:
        raise ValueError(f"No Silver records found in {silver_dir}")
    return pd.concat(reservoirs.values(), ignore_index=True).sort_values(
        [*strata, "point_id"], kind="stable"
    )


def _percentile(series: pd.Series, percentile: float) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.percentile(clean, percentile)) if len(clean) else None


def recorded_matching_configuration(
    silver_dir: Path,
) -> tuple[float, float, str]:
    """Resolve buffer/radius values from the build's own run manifest.

    Evaluating invariants against *current* module defaults silently flips
    results whenever radii changed between the Silver run and validation.
    The manifest written by the producing run is authoritative; module
    defaults are a labelled fallback for ad-hoc directories without one.
    """
    manifest_path = silver_dir.parent / "reports" / "pipeline_run_manifest.json"
    try:
        parameters = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            .get("scientific_parameters", {})
        )
        buffer = float(parameters["road_buffer_m"])
        max_radius = float(max(parameters["matching_sensitivity_radii_m"]))
        return buffer, max_radius, "pipeline_run_manifest"
    except (OSError, KeyError, TypeError, ValueError):
        return (
            float(config.ROAD_BUFFER_METERS),
            float(max(config.MATCHING_SENSITIVITY_RADII_METERS)),
            "module_defaults",
        )


def metric_rows(
    sample: pd.DataFrame,
    road_buffer_meters: Optional[float] = None,
    max_sensitivity_radius_m: Optional[float] = None,
) -> list[dict]:
    """Calculate observed validation metrics from one authoritative sample."""
    buffer = (
        float(config.ROAD_BUFFER_METERS)
        if road_buffer_meters is None
        else float(road_buffer_meters)
    )
    search_radius = (
        float(max(config.MATCHING_SENSITIVITY_RADII_METERS))
        if max_sensitivity_radius_m is None
        else float(max_sensitivity_radius_m)
    )
    metrics: list[dict] = []

    def add(
        metric: str, value: object, unit: str = "count", scope: str = "sample"
    ) -> None:
        metrics.append({"metric": metric, "value": value, "unit": unit, "scope": scope})

    add("sample_rows", len(sample))
    add(
        "primary_radius_sample_rows",
        int(sample["within_primary_match_radius"].fillna(False).astype(bool).sum()),
    )
    add(
        "sensitivity_only_sample_rows",
        int((~sample["within_primary_match_radius"].fillna(False).astype(bool)).sum()),
    )
    add("sample_strata", sample[["year", "highway"]].drop_duplicates().shape[0])
    distances = pd.to_numeric(sample["dist_to_axis"], errors="coerce")
    add("distance_mean", float(distances.mean()), "m")
    add("distance_median", float(distances.median()), "m")
    add("distance_p90", _percentile(distances, 90), "m")
    add("distance_p95", _percentile(distances, 95), "m")
    for threshold in (5.0, 10.0, buffer):
        add(
            f"distance_within_{threshold:g}m",
            float((distances <= threshold).mean() * 100.0),
            "percent",
        )

    heading = pd.to_numeric(sample["heading_alignment_diff"], errors="coerce")
    heading_available = heading.notna()
    add("heading_available", int(heading_available.sum()))
    add(
        "heading_aligned_within_45deg",
        float((heading[heading_available] <= math.pi / 4.0).mean() * 100.0)
        if heading_available.any()
        else None,
        "percent",
    )
    candidates = pd.to_numeric(sample["candidate_count"], errors="coerce")
    add("multi_candidate", float((candidates > 1).mean() * 100.0), "percent")
    add("candidate_count_median", float(candidates.median()))
    add("score_margin_median", _percentile(sample["score_margin"], 50), "score")
    add("distance_margin_median", _percentile(sample["dist_margin_m"], 50), "m")
    add(
        "missing_weather_join",
        int((sample["weather_join_status"] != "MATCHED").sum()),
    )
    add(
        "high_speed_iqr_outliers",
        int(sample["is_high_speed_iqr_outlier"].fillna(False).astype(bool).sum()),
    )
    low_order_classes = sample["highway"].isin(
        ["residential", "living_street", "service"]
    )
    high_speed = pd.to_numeric(sample["calc_speed_kmh"], errors="coerce") > 80.0
    add(
        "high_speed_low_order_road_observations",
        int((low_order_classes & high_speed).sum()),
    )

    invariant_violations = {
        "negative_distance": int((distances < -1e-9).sum()),
        "distance_beyond_search_radius": int(
            (
                distances
                > search_radius + 1e-6
            ).sum()
        ),
        "primary_radius_flag_mismatch": int(
            (
                sample["within_primary_match_radius"]
                .fillna(False)
                .astype(bool)
                != (distances <= buffer + 1e-6)
            ).sum()
        ),
        "azimuth_out_of_range": int(
            (
                (pd.to_numeric(sample["azimuth_diff"], errors="coerce") < 0)
                | (
                    pd.to_numeric(sample["azimuth_diff"], errors="coerce")
                    > math.pi + 1e-9
                )
            ).sum()
        ),
        "alignment_out_of_range": int(
            ((heading < 0) | (heading > math.pi / 2.0 + 1e-9)).sum()
        ),
        "negative_score_margin": int(
            (pd.to_numeric(sample["score_margin"], errors="coerce") < -1e-9).sum()
        ),
        "candidate_count_below_one": int((candidates < 1).sum()),
    }
    for name, count in invariant_violations.items():
        add(f"invariant_{name}", count)
    add("invariant_violation_total", sum(invariant_violations.values()))
    return metrics


def add_manual_ground_truth_metrics(
    metrics: list[dict],
    sample: pd.DataFrame,
    labels_path: Optional[Path],
) -> tuple[list[dict], Optional[pd.DataFrame]]:
    """Join human review labels when available; never fabricate accuracy otherwise."""
    if labels_path is None:
        metrics.append(
            {
                "metric": "manual_ground_truth_status",
                "value": "NOT_PROVIDED",
                "unit": "status",
                "scope": "sample",
            }
        )
        return metrics, None
    labels = pd.read_csv(labels_path)
    required = {"point_id", "correct_road_match"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Manual-label file is missing: {sorted(missing)}")

    # The exported review template deliberately leaves labels blank.  Pandas
    # reads those blank cells as NaN; they are unanswered review rows, not
    # malformed boolean labels.  Keep only completed decisions for validation.
    raw_labels = labels["correct_road_match"]
    completed = raw_labels.notna() & raw_labels.astype(str).str.strip().ne("")
    labels = labels.loc[completed].copy()
    if labels.empty:
        metrics.append(
            {
                "metric": "manual_ground_truth_status",
                "value": "NO_COMPLETED_LABELS",
                "unit": "status",
                "scope": "manual_ground_truth",
            }
        )
        return metrics, None

    if labels["point_id"].duplicated().any():
        raise ValueError("Manual-label file contains duplicate point_id values")
    reviewed = sample.merge(labels, on="point_id", how="inner", validate="one_to_one")
    if reviewed.empty:
        raise ValueError("Manual labels do not overlap the validation sample")
    normalized = reviewed["correct_road_match"].map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"true", "1", "yes", "y"}
    )
    invalid_labels = ~reviewed["correct_road_match"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        or str(value).strip().lower()
        in {"true", "false", "1", "0", "yes", "no", "y", "n"}
    )
    if invalid_labels.any():
        invalid_values = sorted(
            reviewed.loc[invalid_labels, "correct_road_match"].astype(str).unique()
        )
        raise ValueError(f"Unrecognized correct_road_match labels: {invalid_values}")
    correct = normalized.astype(bool)
    metrics.extend(
        [
            {
                "metric": "manual_reviewed_rows",
                "value": len(reviewed),
                "unit": "count",
                "scope": "manual_ground_truth",
            },
            {
                "metric": "manual_match_accuracy",
                "value": float(correct.mean() * 100.0),
                "unit": "percent",
                "scope": "manual_ground_truth",
            },
        ]
    )
    return metrics, reviewed


def run_validation(
    silver_dir: Path,
    reports_dir: Path,
    per_stratum: int = 10_000,
    manual_labels: Optional[Path] = None,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    buffer_meters, search_radius, configuration_source = (
        recorded_matching_configuration(silver_dir)
    )
    if configuration_source == "module_defaults":
        logger.warning(
            "No pipeline_run_manifest.json next to %s; evaluating invariants "
            "against current module defaults instead of the producing "
            "run's recorded radii",
            silver_dir,
        )
    sample = deterministic_stratified_sample(silver_dir, per_stratum=per_stratum)
    metrics = metric_rows(sample, buffer_meters, search_radius)
    metrics, reviewed = add_manual_ground_truth_metrics(metrics, sample, manual_labels)
    metric_frame = pd.DataFrame(metrics)
    invariant_total = int(
        metric_frame.loc[
            metric_frame["metric"] == "invariant_violation_total", "value"
        ].iloc[0]
    )
    summary = {
        "generated_at_utc": utc_now_iso(),
        "silver_directory": str(silver_dir.resolve()),
        "sampling_method": "deterministic SHA-256 bottom-k within year/highway strata",
        "per_stratum": per_stratum,
        "sample_rows": int(len(sample)),
        "strata": ["year", "highway"],
        "manual_ground_truth": (
            str(manual_labels.resolve()) if reviewed is not None else None
        ),
        "used_road_buffer_m": buffer_meters,
        "used_max_sensitivity_radius_m": search_radius,
        "configuration_source": configuration_source,
        "status": "FAIL_INVARIANTS"
        if invariant_total
        else (
            "PASS_INVARIANTS_WITH_MANUAL_REVIEW"
            if reviewed is not None
            else "PASS_INVARIANTS_NO_MANUAL_ACCURACY_CLAIM"
        ),
        "invariant_violation_total": invariant_total,
        "accuracy_claim_permitted": reviewed is not None,
    }
    sample.to_parquet(
        reports_dir / "map_matching_validation_sample.parquet", index=False
    )
    metric_frame.to_csv(reports_dir / "report_map_matching_validation.csv", index=False)
    if reviewed is not None:
        reviewed.to_csv(
            reports_dir / "map_matching_manual_review_results.csv", index=False
        )
    write_json_atomic(reports_dir / "map_matching_validation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-dir", type=Path, default=config.SILVER_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.PIPELINE_REPORTS_DIR)
    parser.add_argument("--per-stratum", type=int, default=10_000)
    parser.add_argument("--manual-labels", type=Path)
    args = parser.parse_args()
    summary = run_validation(
        args.silver_dir,
        args.reports_dir,
        args.per_stratum,
        args.manual_labels,
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["invariant_violation_total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

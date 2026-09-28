"""Summarize point-to-selected-road distances for every Silver record.

The program scans only ``dist_to_axis`` from the Parquet dataset, so it can
process a full Silver layer without loading it into memory.  Counts, mean and
standard deviation are exact. Percentiles use a fixed-width histogram (1 cm by
default), making their maximum rounding uncertainty one histogram bin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


DEFAULT_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
DEFAULT_THRESHOLDS_M = (5.0, 10.0, 15.0, 25.0, 40.0)


def parse_numbers(value: str, label: str) -> tuple[float, ...]:
    """Parse a non-empty comma-separated list of finite numeric values."""
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {label}: {value}") from exc
    if not parsed or any(not math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError(f"Invalid {label}: {value}")
    return parsed


def percentile_from_histogram(histogram: np.ndarray, percentile: float, bin_width: float) -> float | None:
    """Return a linearly interpolated percentile estimate from a histogram."""
    total = int(histogram.sum())
    if total == 0:
        return None
    rank = (percentile / 100.0) * (total - 1)
    cumulative = np.cumsum(histogram)
    lower = int(np.searchsorted(cumulative, math.floor(rank), side="right"))
    upper = int(np.searchsorted(cumulative, math.ceil(rank), side="right"))
    lower_value = (lower + 0.5) * bin_width
    upper_value = (upper + 0.5) * bin_width
    return lower_value + (rank - math.floor(rank)) * (upper_value - lower_value)


def summarize(
    silver_dir: Path,
    max_distance_m: float,
    bin_width_m: float,
    percentiles: tuple[float, ...],
    thresholds_m: tuple[float, ...],
    batch_size: int,
) -> dict:
    """Scan a Silver dataset and return exact aggregates plus binned quantiles."""
    if max_distance_m <= 0 or bin_width_m <= 0 or batch_size <= 0:
        raise ValueError("max distance, bin width, and batch size must be positive")
    if any(not 0 <= percentile <= 100 for percentile in percentiles):
        raise ValueError("percentiles must be between 0 and 100")

    dataset = ds.dataset(str(silver_dir), format="parquet", partitioning="hive")
    if "dist_to_axis" not in dataset.schema.names:
        raise ValueError(f"Silver dataset has no dist_to_axis column: {silver_dir}")

    bin_count = math.ceil(max_distance_m / bin_width_m)
    histogram = np.zeros(bin_count, dtype=np.int64)
    row_count = finite_count = null_or_nonfinite_count = overflow_count = 0
    distance_sum = distance_sum_squares = 0.0
    threshold_counts = {threshold: 0 for threshold in thresholds_m}
    observed_min = math.inf
    observed_max = -math.inf

    for batch in dataset.to_batches(columns=["dist_to_axis"], batch_size=batch_size):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        row_count += len(values)
        numeric = np.asarray(values, dtype=np.float64)
        finite = numeric[np.isfinite(numeric)]
        null_or_nonfinite_count += len(numeric) - len(finite)
        if not len(finite):
            continue
        finite_count += len(finite)
        distance_sum += float(finite.sum())
        distance_sum_squares += float(np.dot(finite, finite))
        observed_min = min(observed_min, float(finite.min()))
        observed_max = max(observed_max, float(finite.max()))
        for threshold in thresholds_m:
            threshold_counts[threshold] += int(np.count_nonzero(finite <= threshold))

        in_range = finite[(finite >= 0.0) & (finite <= max_distance_m)]
        overflow_count += len(finite) - len(in_range)
        if len(in_range):
            indices = np.minimum((in_range / bin_width_m).astype(np.int64), bin_count - 1)
            histogram += np.bincount(indices, minlength=bin_count)

    if overflow_count:
        raise ValueError(
            f"{overflow_count} finite distance values fall outside 0–{max_distance_m:g} m; "
            "increase --max-distance-m before accepting percentiles"
        )

    mean = distance_sum / finite_count if finite_count else None
    sample_std = None
    if finite_count > 1:
        variance = max((distance_sum_squares - distance_sum**2 / finite_count) / (finite_count - 1), 0.0)
        sample_std = math.sqrt(variance)
    return {
        "silver_directory": str(silver_dir.resolve()),
        "row_count": row_count,
        "finite_distance_count": finite_count,
        "null_or_nonfinite_distance_count": null_or_nonfinite_count,
        "distance_min_m": None if finite_count == 0 else observed_min,
        "distance_max_m": None if finite_count == 0 else observed_max,
        "distance_mean_m": mean,
        "distance_sample_std_m": sample_std,
        "percentile_method": "fixed-width histogram with linear interpolation between bin midpoints",
        "percentile_bin_width_m": bin_width_m,
        "percentiles_m": {
            f"p{percentile:g}": percentile_from_histogram(histogram, percentile, bin_width_m)
            for percentile in percentiles
        },
        "shares_within_threshold_m": {
            f"le_{threshold:g}m": None if finite_count == 0 else threshold_counts[threshold] / finite_count
            for threshold in thresholds_m
        },
    }


def write_reports(summary: dict, output_dir: Path) -> None:
    """Write portable JSON and one-metric-per-row CSV reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "silver_road_distance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        ("row_count", summary["row_count"], "count"),
        ("finite_distance_count", summary["finite_distance_count"], "count"),
        ("null_or_nonfinite_distance_count", summary["null_or_nonfinite_distance_count"], "count"),
        ("distance_min", summary["distance_min_m"], "m"),
        ("distance_max", summary["distance_max_m"], "m"),
        ("distance_mean", summary["distance_mean_m"], "m"),
        ("distance_sample_std", summary["distance_sample_std_m"], "m"),
    ]
    rows.extend((name, value, "m") for name, value in summary["percentiles_m"].items())
    rows.extend((name, value * 100.0 if value is not None else None, "percent") for name, value in summary["shares_within_threshold_m"].items())
    with (output_dir / "silver_road_distance_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance-m", type=float, default=40.0)
    parser.add_argument("--bin-width-m", type=float, default=0.01)
    parser.add_argument("--percentiles", type=lambda value: parse_numbers(value, "percentiles"), default=DEFAULT_PERCENTILES)
    parser.add_argument("--thresholds-m", type=lambda value: parse_numbers(value, "thresholds"), default=DEFAULT_THRESHOLDS_M)
    parser.add_argument("--batch-size", type=int, default=131_072)
    args = parser.parse_args()
    summary = summarize(
        args.silver_dir,
        args.max_distance_m,
        args.bin_width_m,
        args.percentiles,
        args.thresholds_m,
        args.batch_size,
    )
    write_reports(summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

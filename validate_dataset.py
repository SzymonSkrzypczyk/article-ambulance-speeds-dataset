"""Validate the corrected relational Gold release and produce reconciled reports."""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from config import config
from pipeline_utils import (
    is_execution_sidecar,
    read_layer_completion_marker,
    sha256_file,
    utc_now_iso,
    write_json_atomic,
)

PROFILE_FILES = {
    "season": "speeds_by_season",
    "month": "speeds_by_month",
    "weekday": "speeds_by_day_of_week",
    "time_of_day": "speeds_by_time_of_day",
}
TEMPORAL_KEYS = {
    "season": ["season"],
    "month": ["season", "month", "month_name"],
    "weekday": ["day_of_week", "day_name", "is_weekend"],
    "time_of_day": ["time_of_day"],
}
COMMON_KEYS = [
    "road_feature_id",
    "operating_speed_reference_kmh",
    "operating_speed_reference_source",
    "travel_direction",
    "signal_mode",
    "weather_category",
    "spatial_scope",
]
REQUIRED_PROFILE_COLUMNS = set(COMMON_KEYS).union(
    {
        "avg_travel_speed_kmh",
        "avg_running_speed_kmh",
        "median_speed_kmh",
        "p85_speed_kmh",
        "stddev_speed_kmh",
        "n_samples",
        "n_unique_source_ids",
        "n_unique_observation_days",
        "n_speed_derivation_methods",
        "stopped_observation_ratio",
        "delta_operating_speed_reference_kmh",
        "meets_min_sample_count",
        "min_sample_count",
    }
)

# Controlled vocabularies realized by the pipeline (data_dictionary.md).
# Column presence alone is not enough; a corrupted value such as
# weather_category="SUNNY" must fail validation, not ship.
ALLOWED_ENUM_VALUES = {
    "travel_direction": {
        "FORWARD",
        "BACKWARD",
        "AGAINST_ONEWAY_DIGITIZED_DIRECTION",
        "UNKNOWN_STATIONARY_OR_SHORT_MOVE",
    },
    "signal_mode": {
        "LIGHT_ON_SOUND_ON",
        "LIGHT_ON_SOUND_OFF",
        "LIGHT_OFF_SOUND_ON",
        "LIGHT_OFF_SOUND_OFF",
        "SIGNAL_MISSING",
        "SIGNAL_INVALID",
    },
    "weather_category": {
        "NO_RECORDED_PRECIPITATION",
        "RAIN_OR_LIQUID_PRECIPITATION",
        "SNOWFALL_OR_FREEZING_PRECIPITATION",
        "FOG_OR_LOW_VISIBILITY",
        "OTHER_OR_UNCLASSIFIED_WEATHER",
        "UNKNOWN_WEATHER",
    },
    "operating_speed_reference_source": {
        "OSM_NUMERIC_KMH",
        "OSM_NUMERIC_MPH_CONVERTED",
        "OSM_SYMBOLIC_TAG",
        "IMPUTED_HIGHWAY_DEFAULT",
        "IMPUTED_UNPARSEABLE_TAG",
    },
    "spatial_scope": {
        "OFFICIAL_KRAKOW_BOUNDING_BOX",
        "EXTENDED_PROCESSING_AREA",
    },
    "time_of_day": {"PEAK_MORNING", "OFF_PEAK_DAY", "PEAK_EVENING", "NIGHT"},
    "season": {"WINTER", "SPRING", "SUMMER", "AUTUMN"},
}


def _unexpected_enum_values(table: pd.DataFrame) -> dict[str, list[str]]:
    """Return out-of-vocabulary sample values per categorical column."""
    unexpected: dict[str, list[str]] = {}
    for column, allowed in ALLOWED_ENUM_VALUES.items():
        if column not in table.columns:
            continue
        observed = set(table[column].dropna().astype(str).unique())
        violations = sorted(observed - allowed)
        if violations:
            unexpected[column] = violations[:5]
    return unexpected


def _read_table(gold_dir: Path, stem: str) -> pd.DataFrame:
    parquet = gold_dir / f"{stem}.parquet"
    csv = gold_dir / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {parquet.name} nor {csv.name} exists")


def _check_manifest(gold_dir: Path) -> tuple[list[dict], list[str]]:
    manifest = gold_dir / "checksums.sha256"
    if not manifest.exists():
        return [], ["checksums.sha256 is missing"]
    rows, errors = [], []
    listed = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(None, 1)
            relative = relative.strip().replace("\\", "/")
        except ValueError:
            errors.append(f"Malformed checksum line {line_number}")
            continue
        target = gold_dir / relative
        listed.add(relative)
        actual = sha256_file(target) if target.is_file() else None
        matched = actual == expected
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matched": matched,
            }
        )
        if not matched:
            errors.append(f"Checksum mismatch or missing file: {relative}")
    actual_files = {
        path.relative_to(gold_dir).as_posix()
        for path in gold_dir.rglob("*")
        if path.is_file()
        and path.name != "checksums.sha256"
        and not is_execution_sidecar(path.name)
    }
    for unlisted in sorted(actual_files - listed):
        errors.append(f"File is not listed in checksum manifest: {unlisted}")
    return rows, errors


def _verify_csv_twin(
    gold_dir: Path,
    stem: str,
    expected_rows: int,
    expected_columns: list[str],
    failures: list[str],
) -> None:
    """Fail when a published CSV twin diverges from its Parquet source.

    Both representations ship to users; until now nothing verified that they
    describe the same table (row count and identical ordered schema), so a
    stale or truncated CSV could be distributed alongside correct Parquet.
    ``geometry`` is exempt by design: spatial columns ship only in
    Parquet/GeoPackage, never in the attribute-only CSV twins.
    """
    parquet = gold_dir / f"{stem}.parquet"
    csv = gold_dir / f"{stem}.csv"
    if not (parquet.exists() and csv.exists()):
        return
    header = pd.read_csv(csv, nrows=0)
    expected_non_geometry = [
        column for column in expected_columns if column != "geometry"
    ]
    if list(header.columns) != expected_non_geometry:
        failures.append(
            f"{stem}.csv columns differ from {stem}.parquet: "
            f"{list(header.columns)} vs {expected_non_geometry}"
        )
        return
    first_column = header.columns[0]
    csv_rows = sum(
        len(chunk)
        for chunk in pd.read_csv(csv, usecols=[first_column], chunksize=1_000_000)
    )
    if csv_rows != expected_rows:
        failures.append(
            f"{stem}.csv has {csv_rows} data rows but {stem}.parquet has "
            f"{expected_rows}"
        )


def validate_gold(
    gold_dir: Path,
    reports_dir: Optional[Path] = None,
    verify_manifest: bool = True,
) -> dict:
    """Run structural, relational, numeric, coverage, and package checks."""
    gold_dir = gold_dir.resolve()
    reports_dir = (
        reports_dir or gold_dir.parent / f"{gold_dir.name}_validation"
    ).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    base_path = gold_dir / "krakow_roads_base.parquet"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    base = gpd.read_parquet(base_path)
    required_base = {
        "road_feature_id",
        "osm_id",
        "operating_speed_reference_kmh",
        "operating_speed_reference_source",
        "road_length_m",
        "intersects_official_krakow_bbox",
        "geometry",
    }
    missing_base = required_base.difference(base.columns)
    if missing_base:
        failures.append(f"Base schema missing: {sorted(missing_base)}")
    base_enum_violations = _unexpected_enum_values(base)
    if base_enum_violations:
        failures.append(
            f"Base contains out-of-vocabulary values: {base_enum_violations}"
        )
    if base["road_feature_id"].duplicated().any():
        failures.append("Base road_feature_id is not unique")
    if (
        base.geometry.isna().any()
        or base.geometry.is_empty.any()
        or (~base.geometry.is_valid).any()
    ):
        failures.append("Base contains null, empty, or invalid geometry")

    tables = {name: _read_table(gold_dir, stem) for name, stem in PROFILE_FILES.items()}
    flat = _read_table(gold_dir, "krakow_ambulance_speeds_2021_2023_flat")
    base_ids = set(base["road_feature_id"])
    integrity_rows, coverage_rows, schema_rows, consistency_rows = [], [], [], []
    sample_sums = {}

    for name, table in {**tables, "flat": flat}.items():
        missing_columns = REQUIRED_PROFILE_COLUMNS.difference(table.columns)
        schema_rows.append(
            {
                "table": name,
                "rows": len(table),
                "columns": len(table.columns),
                "missing_required_columns": ";".join(sorted(missing_columns)),
            }
        )
        if missing_columns:
            failures.append(f"{name} schema missing: {sorted(missing_columns)}")
            continue

        orphan_count = int((~table["road_feature_id"].isin(base_ids)).sum())
        temporal = TEMPORAL_KEYS.get(
            name,
            [
                "season",
                "month",
                "month_name",
                "day_of_week",
                "day_name",
                "is_weekend",
                "time_of_day",
            ],
        )
        key = [column for column in COMMON_KEYS + temporal if column in table.columns]
        duplicate_count = int(table.duplicated(key).sum())
        integrity_rows.append(
            {
                "table": name,
                "orphan_rows": orphan_count,
                "duplicate_key_rows": duplicate_count,
            }
        )
        if orphan_count:
            failures.append(f"{name} has {orphan_count} orphan rows")
        if duplicate_count:
            failures.append(f"{name} has {duplicate_count} duplicate profile keys")

        sample_sums[name] = int(table["n_samples"].sum())
        invalid_counts = int(
            (
                (table["n_samples"] <= 0)
                | (table["n_unique_source_ids"] > table["n_samples"])
                | (table["n_unique_observation_days"] > table["n_samples"])
                | (table["n_speed_derivation_methods"] > table["n_samples"])
            ).sum()
        )
        invalid_ratios = int(
            (
                ~table["stopped_observation_ratio"].between(0.0, 1.0, inclusive="both")
            ).sum()
        )
        support_mismatch = int(
            (
                table["meets_min_sample_count"].astype(bool)
                != (table["n_samples"] >= table["min_sample_count"])
            ).sum()
        )
        expected_delta = (
            table["avg_travel_speed_kmh"] - table["operating_speed_reference_kmh"]
        )
        delta_mismatch = int(
            (
                ~np.isclose(
                    table["delta_operating_speed_reference_kmh"],
                    expected_delta,
                    atol=0.011,
                )
            ).sum()
        )
        enum_violations = _unexpected_enum_values(table)
        if enum_violations:
            failures.append(f"{name} out-of-vocabulary values: {enum_violations}")
        consistency_rows.append(
            {
                "table": name,
                "invalid_count_relationships": invalid_counts,
                "invalid_stopped_ratios": invalid_ratios,
                "sample_support_flag_mismatches": support_mismatch,
                "delta_mismatches": delta_mismatch,
                "invalid_enum_values": sum(
                    1 for _ in enum_violations
                ),
            }
        )
        if invalid_counts or invalid_ratios or support_mismatch or delta_mismatch:
            failures.append(f"{name} contains numeric/logical inconsistencies")

        supported = table[table["meets_min_sample_count"].astype(bool)]
        coverage_rows.append(
            {
                "table": name,
                "rows": len(table),
                "sample_sum": sample_sums[name],
                "roads_with_any_observation": int(table["road_feature_id"].nunique()),
                "rows_meeting_min_sample_count": int(len(supported)),
                "row_support_rate_pct": float(len(supported) / len(table) * 100.0)
                if len(table)
                else 0.0,
                "roads_with_supported_profile": int(
                    supported["road_feature_id"].nunique()
                ),
                "base_roads": int(len(base)),
            }
        )

    relational_sums = {sample_sums.get(name) for name in PROFILE_FILES}
    if len(relational_sums) != 1:
        failures.append(f"Relational profile sample sums disagree: {sample_sums}")
    if sample_sums.get("flat") not in relational_sums:
        failures.append(f"Flat and relational sample sums disagree: {sample_sums}")

    _verify_csv_twin(
        gold_dir,
        "krakow_roads_base",
        int(len(base)),
        list(base.columns),
        failures,
    )
    for name, stem in PROFILE_FILES.items():
        _verify_csv_twin(
            gold_dir,
            stem,
            int(len(tables[name])),
            list(tables[name].columns),
            failures,
        )
    _verify_csv_twin(
        gold_dir,
        "krakow_ambulance_speeds_2021_2023_flat",
        int(len(flat)),
        list(flat.columns),
        failures,
    )

    gpkg_results = []
    for filename in (
        "krakow_ambulance_speeds_2021_2023.gpkg",
        "krakow_ambulance_speeds_2021_2023_flat.gpkg",
    ):
        path = gold_dir / filename
        if not path.exists():
            failures.append(f"Missing GeoPackage: {filename}")
            continue
        with contextlib.closing(sqlite3.connect(path)) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        gpkg_results.append({"file": filename, "integrity_check": result})
        if result != "ok":
            failures.append(f"GeoPackage integrity failed: {filename}: {result}")

    if verify_manifest:
        checksum_rows, checksum_errors = _check_manifest(gold_dir)
        failures.extend(checksum_errors)
    else:
        checksum_rows = []

    pd.DataFrame(schema_rows).to_csv(reports_dir / "report_schema.csv", index=False)
    pd.DataFrame(integrity_rows).to_csv(
        reports_dir / "report_relational_integrity.csv", index=False
    )
    pd.DataFrame(consistency_rows).to_csv(
        reports_dir / "report_numeric_consistency.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(
        reports_dir / "report_sampling_coverage.csv", index=False
    )
    pd.DataFrame(gpkg_results).to_csv(
        reports_dir / "report_geopackage_integrity.csv", index=False
    )
    pd.DataFrame(checksum_rows).to_csv(
        reports_dir / "report_checksums.csv", index=False
    )
    bounds = [float(value) for value in base.to_crs(config.CRS_WGS84).total_bounds]
    completion = read_layer_completion_marker(gold_dir)
    summary = {
        "generated_at_utc": utc_now_iso(),
        "gold_directory": str(gold_dir),
        "status": "PASS" if not failures else "FAIL",
        "failure_count": len(failures),
        "failures": failures,
        "completion_marker_present": completion is not None,
        "layer_configuration_fingerprint": (
            completion.get("configuration_fingerprint") if completion else None
        ),
        "base_road_features": int(len(base)),
        "actual_geometry_bounds_wgs84": bounds,
        "sample_sums": sample_sums,
        "profile_tables": list(PROFILE_FILES),
        "flat_representation": "observed joint combinations",
    }
    write_json_atomic(reports_dir / "dataset_validation_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=config.GOLD_DIR)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()
    summary = validate_gold(
        args.gold_dir, args.reports_dir, verify_manifest=not args.skip_manifest
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

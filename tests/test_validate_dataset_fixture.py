"""Synthetic Gold fixtures proving validate_gold passes good data and fails
each specific mutation: orphan keys, out-of-vocabulary values, tampered
formulas, CSV-twin drift, and unlisted files in the checksum manifest."""

from __future__ import annotations

import hashlib
import sqlite3

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from validate_dataset import validate_gold

TEMPORAL = {
    "season": {"season": "WINTER"},
    "month": {"season": "WINTER", "month": 1, "month_name": "January"},
    "weekday": {"day_of_week": 1, "day_name": "Monday", "is_weekend": False},
    "time_of_day": {"time_of_day": "PEAK_MORNING"},
}
FLAT_TEMPORAL = {
    **TEMPORAL["month"],
    "day_of_week": 1,
    "day_name": "Monday",
    "is_weekend": False,
    "time_of_day": "PEAK_MORNING",
}


def _write_checksums(gold_dir):
    """Create the checksum fixture without importing an excluded packager."""
    entries = []
    for path in sorted(gold_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(gold_dir).as_posix()}")
    (gold_dir / "checksums.sha256").write_text(
        "\n".join(entries) + "\n", encoding="utf-8", newline="\n"
    )


def _base_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "road_feature_id": ["r1", "r2"],
            "osm_id": [1, 2],
            "osm_ids_raw": ["1", "2"],
            "name": ["Alpha", "Beta"],
            "highway": ["residential", "service"],
            "maxspeed_raw": ["50", None],
            "operating_speed_reference_kmh": [50, 30],
            "operating_speed_reference_source": [
                "OSM_NUMERIC_KMH",
                "IMPUTED_HIGHWAY_DEFAULT",
            ],
            "oneway": [False, True],
            "oneway_raw": ["false", "yes"],
            "oneway_reversed": [False, False],
            "intersects_official_krakow_bbox": [True, True],
            "road_length_m": [10.0, 5.0],
        },
        geometry=[
            LineString([(19.80, 50.00), (19.81, 50.01)]),
            LineString([(19.90, 50.00), (19.91, 50.00)]),
        ],
        crs="EPSG:4326",
    )


def _profile_frame(
    road_ids=("r1", "r2"),
    temporal=None,
    weather="NO_RECORDED_PRECIPITATION",
    reference=(50, 50),
):
    rows = []
    for rid, ref in zip(road_ids, reference):
        rows.append(
            {
                "road_feature_id": rid,
                "operating_speed_reference_kmh": ref,
                "operating_speed_reference_source": "OSM_NUMERIC_KMH",
                "travel_direction": "FORWARD",
                "signal_mode": "LIGHT_OFF_SOUND_OFF",
                "weather_category": weather,
                "spatial_scope": "OFFICIAL_KRAKOW_BOUNDING_BOX",
                **(temporal or {}),
                "avg_travel_speed_kmh": 30.0,
                "avg_running_speed_kmh": 31.5,
                "median_speed_kmh": 30.0,
                "p85_speed_kmh": 35.0,
                "stddev_speed_kmh": 2.0,
                "n_samples": 12,
                "n_unique_source_ids": 3,
                "n_unique_observation_days": 4,
                "n_speed_derivation_methods": 1,
                "stopped_observation_ratio": 0.25,
                "delta_operating_speed_reference_kmh": round(30.0 - ref, 2),
                "meets_min_sample_count": True,
                "min_sample_count": 10,
            }
        )
    return pd.DataFrame(rows)


def _write_table(gold_dir, stem, frame: pd.DataFrame, geometry=False):
    frame.to_parquet(gold_dir / f"{stem}.parquet", index=False)
    if not geometry:
        frame.to_csv(gold_dir / f"{stem}.csv", index=False)
    else:
        frame.drop(columns=["geometry"]).to_csv(gold_dir / f"{stem}.csv", index=False)


def _build_gold(tmp_path: __import__("pathlib").Path) -> __import__("pathlib").Path:
    gold = tmp_path / "gold"
    gold.mkdir()
    _write_table(gold, "krakow_roads_base", _base_frame(), geometry=True)

    tables = {
        name: _profile_frame(temporal=temporal) for name, temporal in TEMPORAL.items()
    }
    for name, frame in tables.items():
        stem = {
            "season": "speeds_by_season",
            "month": "speeds_by_month",
            "weekday": "speeds_by_day_of_week",
            "time_of_day": "speeds_by_time_of_day",
        }[name]
        _write_table(gold, stem, frame)

    flat = _profile_frame(temporal=FLAT_TEMPORAL)
    _write_table(gold, "krakow_ambulance_speeds_2021_2023_flat", flat)

    # The validator only runs SQLite integrity checking on these databases;
    # an initialized SQLite container satisfies it without OGR tooling.
    for filename in (
        "krakow_ambulance_speeds_2021_2023.gpkg",
        "krakow_ambulance_speeds_2021_2023_flat.gpkg",
    ):
        sqlite3.connect(gold / filename).close()
    return gold


def _run(gold, manifest=False):
    return validate_gold(
        gold, gold.parent / "reports", verify_manifest=manifest
    )


def test_synthetic_gold_passes(tmp_path):
    summary = _run(_build_gold(tmp_path))
    assert summary["status"] == "PASS", summary["failures"]


def test_orphan_profile_key_fails(tmp_path):
    gold = _build_gold(tmp_path)
    frame = pd.read_parquet(gold / "speeds_by_month.parquet")
    frame.loc[0, "road_feature_id"] = "ghost"
    frame.to_parquet(gold / "speeds_by_month.parquet", index=False)
    summary = _run(gold)
    assert summary["status"] == "FAIL"
    assert any("orphan" in failure for failure in summary["failures"])


def test_out_of_vocabulary_value_fails(tmp_path):
    gold = _build_gold(tmp_path)
    frame = pd.read_parquet(gold / "speeds_by_month.parquet")
    frame.loc[0, "weather_category"] = "SUNNY"
    frame.to_parquet(gold / "speeds_by_month.parquet", index=False)
    summary = _run(gold)
    assert summary["status"] == "FAIL"
    assert any("vocabulary" in failure for failure in summary["failures"])


def test_tampered_delta_formula_fails(tmp_path):
    gold = _build_gold(tmp_path)
    frame = pd.read_parquet(gold / "speeds_by_season.parquet")
    frame.loc[0, "delta_operating_speed_reference_kmh"] = 99.0
    frame.to_parquet(gold / "speeds_by_season.parquet", index=False)
    summary = _run(gold)
    assert summary["status"] == "FAIL"
    assert any("inconsistencies" in failure for failure in summary["failures"])


def test_csv_twin_column_drift_fails(tmp_path):
    gold = _build_gold(tmp_path)
    frame = pd.read_parquet(gold / "speeds_by_month.parquet")
    frame.drop(columns=["stopped_observation_ratio"]).to_csv(
        gold / "speeds_by_month.csv", index=False
    )
    summary = _run(gold)
    assert summary["status"] == "FAIL"
    assert any("speeds_by_month.csv columns differ" in f for f in summary["failures"])


def test_geometry_only_in_parquet_does_not_fail_twins(tmp_path):
    """Real flat Parquet carries geometry; its CSV omits it by design."""
    gold = _build_gold(tmp_path)
    frame = pd.read_parquet(gold / "krakow_ambulance_speeds_2021_2023_flat.parquet")
    frame["geometry"] = b"\x00" * 21  # arbitrary WKB placeholder column
    frame.to_parquet(
        gold / "krakow_ambulance_speeds_2021_2023_flat.parquet", index=False
    )
    # CSV twin intentionally keeps the attribute-only schema.
    summary = _run(gold)
    assert summary["status"] == "PASS", summary["failures"]
    frame_without_one_attribute = frame.drop(columns=["stopped_observation_ratio"])
    frame_without_one_attribute.to_csv(
        gold / "krakow_ambulance_speeds_2021_2023_flat.csv", index=False
    )
    summary = _run(gold)
    assert summary["status"] == "FAIL"
    assert any(
        "flat.csv columns differ" in failure for failure in summary["failures"]
    )


def test_unlisted_file_breaks_checksum_verification(tmp_path):
    gold = _build_gold(tmp_path)
    assert _run(gold, manifest=False)["status"] == "PASS"
    _write_checksums(gold)
    (gold / "stowaway.txt").write_text("unlisted", encoding="utf-8")
    summary = _run(gold, manifest=True)
    assert summary["status"] == "FAIL"
    assert any("not listed" in failure for failure in summary["failures"])

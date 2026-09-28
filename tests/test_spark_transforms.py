"""Opt-in integration tests for the pinned Spark/Sedona runtime.

Run with ``KARETKI_RUN_SPARK_TESTS=1 python -m pytest tests/test_spark_transforms.py``.
The first run can download the pinned JVM artifacts declared in main_pipeline.py.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("KARETKI_RUN_SPARK_TESTS") != "1",
    reason="set KARETKI_RUN_SPARK_TESTS=1 to run Spark/Sedona integration tests",
)
pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from config import config  # noqa: E402
from etl import bronze_gps  # noqa: E402
from etl.bronze_gps import process_bronze_gps_logs  # noqa: E402
from etl.gold_aggregation import _write_gold_qa, aggregate_speed_profiles  # noqa: E402
from etl.silver_matching import (  # noqa: E402
    _add_direction_weather_and_outlier_flags,
    _derive_trajectory_kinematics,
    _load_weather,
    _select_best_road,
    _speed_source_comparison_summary,
    process_silver_map_matching,
)
from etl.weather import _write_weather_parquet  # noqa: E402
from main_pipeline import init_sedona_spark_session  # noqa: E402
from pipeline_utils import replace_directory_from_staging  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = init_sedona_spark_session(default_parallelism=4, shuffle_partitions=4)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_gps_primary_speed_and_matching_execute(spark, monkeypatch) -> None:
    monkeypatch.setattr(config, "SPEED_SOURCE", "gps_speed")
    bronze = spark.createDataFrame(
        [
            ("p1", "source-1", datetime(2021, 1, 1, 12, 0, 0), 50.0647, 19.9450, 0.0),
            ("p2", "source-1", datetime(2021, 1, 1, 12, 0, 10), 50.0647, 19.9451, 3333.0),
        ],
        "point_id string, source_entity_id string, timestamp_utc timestamp, "
        "GPS_LAT double, GPS_LON double, GPS_SPEED double",
    )
    kinematics = _derive_trajectory_kinematics(bronze)
    speed_row = kinematics.filter("point_id = 'p2'").first()
    assert speed_row.calc_speed_kmh == pytest.approx(119.988)
    assert speed_row.speed_source == "GPS_SPEED_CM_PER_S_PRIMARY"

    roads = spark.sql(
        """
        SELECT
          'road-1' AS road_feature_id,
          'road-1:0' AS match_segment_id,
          CAST(1 AS BIGINT) AS osm_id,
          'Synthetic road' AS name,
          'residential' AS highway,
          '50' AS maxspeed_raw,
          CAST(50 AS INT) AS maxspeed_kmh,
          'OSM_NUMERIC_KMH' AS maxspeed_source,
          false AS oneway,
          'no' AS oneway_raw,
          ST_Transform(
            ST_GeomFromWKT('LINESTRING (19.944 50.0647, 19.946 50.0647)'),
            'EPSG:4326',
            'EPSG:2180'
          ) AS road_segment_geom
        """
    )
    roads = roads.selectExpr(
        "*",
        "ST_Buffer(road_segment_geom, 15D, 'quad_segs=2') AS road_buffer",
        "ST_StartPoint(road_segment_geom) AS road_start_point",
        "ST_EndPoint(road_segment_geom) AS road_end_point",
        "ST_Azimuth(ST_StartPoint(road_segment_geom), "
        "ST_EndPoint(road_segment_geom)) AS road_azimuth",
    )
    matched = _select_best_road(kinematics, roads)
    match = matched.filter("point_id = 'p2'").first()
    assert match.point_id == "p2"
    assert match.within_primary_match_radius is True
    assert match.selected_match_radius_m == pytest.approx(15.0)
    assert match.candidate_count_primary_radius == 1

    weather = spark.createDataFrame(
        [(datetime(2021, 1, 1, 12, 0, 0), "DRY", "era5")],
        "weather_hour_utc timestamp, weather_category string, weather_model string",
    )
    silver = _add_direction_weather_and_outlier_flags(
        matched.filter("point_id = 'p2'"), weather
    )
    result = silver.first()
    assert result.weather_join_status == "MATCHED"
    assert result.road_feature_id == "road-1"
    assert result.is_high_speed_iqr_outlier is False


def test_spark_session_disables_nio_transfer_to(spark) -> None:
    assert spark.conf.get("spark.file.transferTo") == "false"


def test_gps_primary_rejects_invalid_values_and_compares_displacement(spark, monkeypatch) -> None:
    monkeypatch.setattr(config, "SPEED_SOURCE", "gps_speed")
    bronze = spark.createDataFrame(
        [
            ("zero", "a", datetime(2021, 1, 1, 0, 0, 0), 50.0, 19.9, 0.0, 2021, 1, True),
            ("missing", "b", datetime(2021, 1, 1, 0, 0, 0), 50.0, 19.9, None, 2021, 1, True),
            ("nonfinite", "n", datetime(2021, 1, 1, 0, 0, 0), 50.0, 19.9, float("nan"), 2021, 1, True),
            ("negative", "c", datetime(2021, 1, 1, 0, 0, 0), 50.0, 19.9, -1.0, 2021, 1, True),
            ("first", "d", datetime(2021, 1, 1, 0, 0, 0), 50.0, 19.9, 1000.0, 2021, 1, False),
            ("second", "d", datetime(2021, 1, 1, 0, 0, 10), 50.0, 19.9001, 1000.0, 2021, 1, False),
        ],
        "point_id string, source_entity_id string, timestamp_utc timestamp, "
        "GPS_LAT double, GPS_LON double, GPS_SPEED double, year int, month int, "
        "inside_study_aoi boolean",
    )
    kinematics = _derive_trajectory_kinematics(bronze)
    rows = {row.point_id: row for row in kinematics.collect()}
    assert rows["zero"].calc_speed_kmh == 0.0
    assert rows["missing"].speed_derivation_reason == "GPS_SPEED_MISSING"
    assert rows["missing"].calc_speed_kmh is None
    assert rows["nonfinite"].speed_derivation_reason == "GPS_SPEED_NON_FINITE"
    assert rows["nonfinite"].calc_speed_kmh is None
    assert rows["negative"].speed_derivation_reason == "GPS_SPEED_NEGATIVE"
    assert rows["negative"].calc_speed_kmh is None
    comparison = _speed_source_comparison_summary(kinematics).first()
    assert comparison.overlap_count == 1
    assert comparison.mean_signed_difference_kmh is not None


def test_gps_primary_allows_speed_above_150_before_matching(spark, monkeypatch) -> None:
    monkeypatch.setattr(config, "SPEED_SOURCE", "gps_speed")
    bronze = spark.createDataFrame(
        [
            ("p1", "source-1", datetime(2021, 1, 1, 12, 0, 0), 50.0647, 19.9450, 5000.0),
        ],
        "point_id string, source_entity_id string, timestamp_utc timestamp, "
        "GPS_LAT double, GPS_LON double, GPS_SPEED double",
    )
    kinematics = _derive_trajectory_kinematics(bronze)
    roads = spark.sql(
        """
        SELECT 'road-1' AS road_feature_id, 'road-1:0' AS match_segment_id,
          CAST(1 AS BIGINT) AS osm_id, 'Synthetic' AS name,
          'residential' AS highway, '50' AS maxspeed_raw, CAST(50 AS INT) AS maxspeed_kmh,
          'OSM_NUMERIC_KMH' AS maxspeed_source, false AS oneway, 'no' AS oneway_raw,
          ST_Transform(ST_GeomFromWKT('LINESTRING (19.944 50.0647, 19.946 50.0647)'),
          'EPSG:4326', 'EPSG:2180') AS road_segment_geom
        """
    ).selectExpr(
        "*", "ST_Buffer(road_segment_geom, 15D, 'quad_segs=2') AS road_buffer",
        "ST_StartPoint(road_segment_geom) AS road_start_point",
        "ST_EndPoint(road_segment_geom) AS road_end_point",
        "ST_Azimuth(ST_StartPoint(road_segment_geom), ST_EndPoint(road_segment_geom)) AS road_azimuth",
    )
    matched = _select_best_road(kinematics, roads).first()
    assert matched.calc_speed_kmh == pytest.approx(180.0)


def test_displacement_mode_preserves_interval_eligibility(spark, monkeypatch) -> None:
    monkeypatch.setattr(config, "SPEED_SOURCE", "displacement")
    bronze = spark.createDataFrame(
        [
            ("p1", "source-1", datetime(2021, 1, 1, 0, 0, 0), 50.0, 19.9, 1000.0),
            ("p2", "source-1", datetime(2021, 1, 1, 0, 1, 0), 50.0, 19.91, 1000.0),
        ],
        "point_id string, source_entity_id string, timestamp_utc timestamp, "
        "GPS_LAT double, GPS_LON double, GPS_SPEED double",
    )
    row = _derive_trajectory_kinematics(bronze).filter("point_id = 'p2'").first()
    assert row.calc_speed_kmh is None
    assert row.speed_source == "UNUSABLE_INTERVAL"


def test_gold_profile_aggregation_executes(spark) -> None:
    observations = spark.createDataFrame(
        [
            (
                "road-1",
                50,
                "OSM_NUMERIC_KMH",
                "p1",
                "source-1",
                datetime(2021, 1, 1),
                0.0,
                "DISPLACEMENT",
            ),
            (
                "road-1",
                50,
                "OSM_NUMERIC_KMH",
                "p2",
                "source-1",
                datetime(2021, 1, 2),
                20.0,
                "DISPLACEMENT",
            ),
        ],
        "road_feature_id string, operating_speed_reference_kmh int, "
        "operating_speed_reference_source string, point_id string, "
        "source_entity_id string, observation_date timestamp, "
        "calc_speed_kmh double, speed_source string",
    )
    profile = aggregate_speed_profiles(
        observations,
        [
            "road_feature_id",
            "operating_speed_reference_kmh",
            "operating_speed_reference_source",
        ],
    ).first()
    assert profile.n_samples == 2
    assert profile.n_unique_source_ids == 1
    assert profile.n_unique_observation_days == 2
    assert profile.avg_travel_speed_kmh == pytest.approx(10.0)
    assert profile.avg_running_speed_kmh == pytest.approx(20.0)
    assert profile.stopped_observation_ratio == pytest.approx(0.5)


def test_spark_reads_weather_parquet_timestamp(spark, tmp_path, monkeypatch) -> None:
    path = tmp_path / "weather.parquet"
    frame = pd.DataFrame(
        {
            "weather_hour_utc": [pd.Timestamp("2021-01-01T00:00:00")],
            "weather_category": ["NO_RECORDED_PRECIPITATION"],
            "weather_model": ["era5"],
        }
    )
    _write_weather_parquet(frame, path)
    monkeypatch.setattr(config, "WEATHER_PARQUET_PATH", path)
    assert _load_weather(spark).count() == 1


def test_chunked_silver_preserves_cross_quarter_predecessor(
    spark, tmp_path, monkeypatch
) -> None:
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    roads_path = tmp_path / "segments.parquet"
    weather_path = tmp_path / "weather.parquet"
    qa_path = tmp_path / "reports" / "silver.parquet"
    qa_path.parent.mkdir(parents=True)

    rows = [
        {
            "point_id": "p1",
            "source_entity_id": "source-1",
            "source_entity_id_semantic": "UNVERIFIED_SOURCE_IDENTIFIER",
            "timestamp_utc": datetime(2021, 3, 31, 23, 59, 50),
            "timestamp_warsaw": datetime(2021, 4, 1, 1, 59, 50),
            "timestamp_source": "GPS_TIME_UTC",
            "timestamp_disagreement_sec": 0,
            "timestamp_sources_disagree": False,
            "year": 2021,
            "month": 3,
            "quarter": "Q1",
            "season": "SPRING",
            "day": 31,
            "hour": 1,
            "day_of_week_iso": 3,
            "is_weekend": False,
            "inside_study_aoi": True,
            "inside_road_extraction_bbox": True,
            "GPS_LAT": 50.0647,
            "GPS_LON": 19.9450,
            "GPS_SPEED": 0.0,
            "signal_mode": "LIGHT_OFF_SOUND_OFF",
        },
        {
            "point_id": "p2",
            "source_entity_id": "source-1",
            "source_entity_id_semantic": "UNVERIFIED_SOURCE_IDENTIFIER",
            "timestamp_utc": datetime(2021, 4, 1, 0, 0, 0),
            "timestamp_warsaw": datetime(2021, 4, 1, 2, 0, 0),
            "timestamp_source": "GPS_TIME_UTC",
            "timestamp_disagreement_sec": 0,
            "timestamp_sources_disagree": False,
            "year": 2021,
            "month": 4,
            "quarter": "Q2",
            "season": "SPRING",
            "day": 1,
            "hour": 2,
            "day_of_week_iso": 4,
            "is_weekend": False,
            "inside_study_aoi": True,
            "inside_road_extraction_bbox": True,
            "GPS_LAT": 50.0647,
            "GPS_LON": 19.9451,
            "GPS_SPEED": 0.0,
            "signal_mode": "LIGHT_OFF_SOUND_OFF",
        },
    ]
    spark.createDataFrame(rows).write.mode("overwrite").partitionBy(
        "year", "quarter"
    ).parquet(str(bronze_dir))

    roads = spark.sql(
        """
        SELECT
          'road-1' AS road_feature_id,
          'road-1:0' AS match_segment_id,
          CAST(1 AS BIGINT) AS osm_id,
          'Synthetic road' AS name,
          'residential' AS highway,
          '50' AS maxspeed_raw,
          CAST(50 AS INT) AS maxspeed_kmh,
          'OSM_NUMERIC_KMH' AS maxspeed_source,
          false AS oneway,
          'no' AS oneway_raw,
          ST_AsBinary(ST_Transform(
            ST_GeomFromWKT('LINESTRING (19.944 50.0647, 19.946 50.0647)'),
            'EPSG:4326', 'EPSG:2180'
          )) AS geometry
        """
    )
    roads.write.mode("overwrite").parquet(str(roads_path))
    _write_weather_parquet(
        pd.DataFrame(
            {
                "weather_hour_utc": [pd.Timestamp("2021-04-01T00:00:00")],
                "weather_category": ["NO_RECORDED_PRECIPITATION"],
                "weather_model": ["era5"],
            }
        ),
        weather_path,
    )

    monkeypatch.setattr(config, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(config, "SILVER_DIR", silver_dir)
    monkeypatch.setattr(config, "OSM_MATCH_SEGMENTS_PARQUET_PATH", roads_path)
    monkeypatch.setattr(config, "WEATHER_PARQUET_PATH", weather_path)
    monkeypatch.setattr(config, "SILVER_QA_PATH", qa_path)
    monkeypatch.setattr(config, "SPEED_SOURCE", "displacement")

    process_silver_map_matching(spark, source_shards=2)

    result = spark.read.parquet(str(silver_dir)).filter("point_id = 'p2'").first()
    assert result.time_delta_sec == 10
    assert result.calc_speed_kmh > 0
    qa = {
        row["category"]: row["count"]
        for row in spark.read.parquet(str(qa_path))
        .filter("report_type = 'ATTRITION'")
        .collect()
    }
    assert qa["bronze_rows"] == 2
    assert qa["usable_speed_rows"] == 1


def test_bounded_bronze_ignores_json_checkpoint_metadata_during_finalization(
    spark, tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "csv"
    bronze_dir = tmp_path / "bronze"
    qa_path = tmp_path / "reports" / "bronze.parquet"
    input_dir.mkdir()
    qa_path.parent.mkdir(parents=True)
    header = (
        "GPS,GPS_HEIGHT,GPS_LAT,GPS_LON,GPS_SPEED,GPS_TIME,DIN,ZAPLON,"
        "SYGNALIZACJA_SWIETLNA,SYGNALIZACJA_DZWIEKOWA,ETL_CZAS\n"
    )
    first = "1,200,50.0647,19.9450,0,2021-01-01 00:00:00,0,1,0,0,2021-01-01 01:00:00\n"
    second = "1,200,50.0647,19.9451,0,2021-01-01 00:00:10,0,1,0,0,2021-01-01 01:00:10\n"
    (input_dir / "a.csv").write_text(header + first)
    (input_dir / "b.csv").write_text(header + first + second)

    monkeypatch.setattr(config, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(config, "BRONZE_QA_PATH", qa_path)
    monkeypatch.setattr(config, "RAW_CSV_DIR", tmp_path / "unused-raw")
    monkeypatch.setattr(config, "INPUT_ZIP_DIR", tmp_path / "missing-zips")
    monkeypatch.setattr(config, "INPUT_DATA_DIR", tmp_path / "missing-input")

    process_bronze_gps_logs(
        spark,
        input_path=input_dir,
        force_rebuild=True,
        batch_target_bytes=1,
    )

    # The pipeline writes JSON batch-completion manifests beside the Parquet
    # checkpoints before it reaches this point.  Successful finalization
    # proves it reads only batch checkpoint directories as Parquet.
    assert spark.read.parquet(str(bronze_dir)).count() == 2
    qa = {
        row["category"]: row["count"]
        for row in spark.read.parquet(str(qa_path))
        .filter("report_type = 'ATTRITION'")
        .collect()
    }
    assert qa["raw_rows"] == 3
    assert qa["deduplicated_valid_rows"] == 2


def test_resumed_bronze_staging_reuses_marked_partitions_without_raw_csvs(
    spark, tmp_path, monkeypatch
) -> None:
    """A stable staging retry retains completed quarters after candidates free."""
    input_dir = tmp_path / "csv"
    published_bronze = tmp_path / "output" / "bronze"
    staged_bronze = tmp_path / "scratch" / "staging" / "bronze"
    qa_path = tmp_path / "output" / "reports" / "bronze.parquet"
    input_dir.mkdir()
    qa_path.parent.mkdir(parents=True)
    header = (
        "GPS,GPS_HEIGHT,GPS_LAT,GPS_LON,GPS_SPEED,GPS_TIME,DIN,ZAPLON,"
        "SYGNALIZACJA_SWIETLNA,SYGNALIZACJA_DZWIEKOWA,ETL_CZAS\n"
    )
    (input_dir / "q1.csv").write_text(
        header + "1,200,50.0647,19.9450,0,2021-01-01 00:00:00,0,1,0,0,2021-01-01 01:00:00\n"
    )
    (input_dir / "q2.csv").write_text(
        header + "2,200,50.0647,19.9451,0,2021-04-01 00:00:00,0,1,0,0,2021-04-01 01:00:00\n"
    )

    monkeypatch.setattr(config, "BRONZE_DIR", staged_bronze)
    monkeypatch.setattr(config, "BRONZE_QA_PATH", qa_path)
    monkeypatch.setattr(config, "RAW_CSV_DIR", tmp_path / "unused-raw")
    monkeypatch.setattr(config, "INPUT_ZIP_DIR", tmp_path / "missing-zips")
    monkeypatch.setattr(config, "INPUT_DATA_DIR", tmp_path / "missing-input")

    original_write_json_atomic = bronze_gps.write_json_atomic

    def interrupt_after_q1(path: Path, payload: dict) -> None:
        if path.name == "2021_Q2.json" and path.parent.name == "publish-markers":
            raise RuntimeError("simulated interruption")
        original_write_json_atomic(path, payload)

    monkeypatch.setattr(bronze_gps, "write_json_atomic", interrupt_after_q1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        process_bronze_gps_logs(
            spark,
            input_path=input_dir,
            force_rebuild=True,
            batch_target_bytes=1,
            work_root_parent=tmp_path / "output",
        )

    candidate_root = tmp_path / "output" / ".bronze-work" / "candidates"
    assert not list(candidate_root.glob("batch_*/year=2021/quarter=Q1"))

    monkeypatch.setattr(bronze_gps, "write_json_atomic", original_write_json_atomic)
    shutil.rmtree(input_dir)
    process_bronze_gps_logs(
        spark,
        batch_target_bytes=1,
        work_root_parent=tmp_path / "output",
    )
    replace_directory_from_staging(
        staged_bronze,
        published_bronze,
        staging_root=tmp_path / "scratch",
    )

    assert {
        path.relative_to(published_bronze) for path in published_bronze.glob("year=*/quarter=*")
    } == {
        Path("year=2021/quarter=Q1"),
        Path("year=2021/quarter=Q2"),
    }


def test_bronze_qa_separates_invalid_coordinates_from_outside_aoi(
    spark, tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "csv"
    bronze_dir = tmp_path / "bronze"
    qa_path = tmp_path / "reports" / "bronze.parquet"
    input_dir.mkdir()
    qa_path.parent.mkdir(parents=True)
    header = (
        "GPS,GPS_HEIGHT,GPS_LAT,GPS_LON,GPS_SPEED,GPS_TIME,DIN,ZAPLON,"
        "SYGNALIZACJA_SWIETLNA,SYGNALIZACJA_DZWIEKOWA,ETL_CZAS\n"
    )
    inside = "1,200,50.0647,19.9450,0,2021-01-01 00:00:00,0,1,0,0,2021-01-01 01:00:00\n"
    outside = "2,200,50.5000,19.9450,0,2021-01-01 00:00:00,0,1,0,0,2021-01-01 01:00:00\n"
    invalid = "3,200,0.0,0.0,0,2021-01-01 00:00:00,0,1,0,0,2021-01-01 01:00:00\n"
    (input_dir / "coordinates.csv").write_text(header + inside + outside + invalid)

    monkeypatch.setattr(config, "BRONZE_DIR", bronze_dir)
    monkeypatch.setattr(config, "BRONZE_QA_PATH", qa_path)
    monkeypatch.setattr(config, "RAW_CSV_DIR", tmp_path / "unused-raw")
    monkeypatch.setattr(config, "INPUT_ZIP_DIR", tmp_path / "missing-zips")

    process_bronze_gps_logs(
        spark,
        input_path=input_dir,
        force_rebuild=True,
    )

    qa = {
        row["category"]: row["count"]
        for row in spark.read.parquet(str(qa_path))
        .filter("report_type = 'ATTRITION'")
        .collect()
    }
    assert qa["raw_rows"] == 3
    assert qa["invalid_coordinate_rows"] == 1
    assert qa["outside_study_aoi_rows"] == 1
    assert qa["inside_study_aoi_rows"] == 1
    assert qa["deduplicated_valid_rows"] == 2
    bronze = spark.read.parquet(str(bronze_dir))
    assert bronze.count() == 2
    assert bronze.filter(~F.col("inside_study_aoi")).count() == 1


def test_gold_qa_reuses_materialized_profile_tables(
    spark, tmp_path, monkeypatch
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(config, "PIPELINE_REPORTS_DIR", reports_dir)
    silver = spark.createDataFrame(
        [("p1", False, True), ("p2", True, True), ("p3", False, False)],
        "point_id string, is_high_speed_iqr_outlier boolean, "
        "within_primary_match_radius boolean",
    )
    included = silver.filter(
        F.col("within_primary_match_radius")
        & ~F.col("is_high_speed_iqr_outlier")
    )
    tables = {
        name: pd.DataFrame({"n_samples": [1, 2]})
        for name in ("season", "month", "weekday", "time_of_day")
    }

    _write_gold_qa(spark, silver, included, tables)

    report = {
        row["metric"]: row["value"]
        for row in spark.read.parquet(
            str(reports_dir / "gold_aggregation_reconciliation.parquet")
        ).collect()
    }
    assert report["silver_input_rows"] == 3
    assert report["primary_radius_silver_rows"] == 2
    assert report["sensitivity_only_silver_rows"] == 1
    assert report["gold_included_rows"] == 1
    assert report["excluded_sensitivity_only_rows"] == 1
    assert report["excluded_primary_high_speed_iqr_outlier_rows"] == 1
    assert report["season_rows"] == 2
    assert report["season_sample_sum"] == 3

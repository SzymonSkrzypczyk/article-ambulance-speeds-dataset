"""Whole-history trajectory speed derivation and auditable road map matching."""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import config
from pipeline_utils import (
    configuration_fingerprint,
    layer_is_complete,
    step_is_complete,
    write_layer_completion_marker,
    write_step_completion_manifest,
)

logger = logging.getLogger(__name__)


def circular_angle_difference(column_a: F.Column, column_b: F.Column) -> F.Column:
    """Spark expression for the smallest directional angle in ``[0, pi]``."""
    raw = F.abs(column_a - column_b)
    return F.least(raw, F.lit(2.0 * math.pi) - raw)


def _load_matching_segments(spark: SparkSession) -> DataFrame:
    path = config.OSM_MATCH_SEGMENTS_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Atomic OSM matching segments not found at {path}; rerun the OSM stage"
        )
    roads = spark.read.parquet(str(path))
    required = {
        "road_feature_id",
        "match_segment_id",
        "osm_id",
        "maxspeed_kmh",
        "maxspeed_source",
        "geometry",
    }
    missing = required.difference(roads.columns)
    if missing:
        raise ValueError(f"OSM matching-segment schema is missing: {sorted(missing)}")
    maximum_radius = max(
        config.ROAD_BUFFER_METERS, *config.MATCHING_SENSITIVITY_RADII_METERS
    )
    return (
        roads.withColumn("road_segment_geom", F.expr("ST_GeomFromWKB(geometry)"))
        .drop("geometry")
        .withColumn(
            "road_buffer",
            F.expr(
                f"ST_Buffer(road_segment_geom, {maximum_radius}, 'quad_segs=2')"
            ),
        )
        .withColumn("road_start_point", F.expr("ST_StartPoint(road_segment_geom)"))
        .withColumn("road_end_point", F.expr("ST_EndPoint(road_segment_geom)"))
        .withColumn(
            "road_azimuth", F.expr("ST_Azimuth(road_start_point, road_end_point)")
        )
    )


def _load_weather(spark: SparkSession) -> DataFrame:
    path = config.WEATHER_PARQUET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Weather dataset not found at {path}; missing weather is never assumed dry"
        )
    weather = spark.read.parquet(str(path))
    required = {"weather_hour_utc", "weather_category", "weather_model"}
    missing = required.difference(weather.columns)
    if missing:
        raise ValueError(f"Weather schema is missing: {sorted(missing)}")
    return (
        weather.withColumn(
            "weather_hour_utc", F.col("weather_hour_utc").cast("timestamp")
        )
        .select("weather_hour_utc", "weather_category", "weather_model")
        .dropDuplicates(["weather_hour_utc"])
    )


def _derive_trajectory_kinematics(bronze: DataFrame) -> DataFrame:
    """Compute intervals over complete source-identifier histories.

    This intentionally happens before output partitioning so quarter/year boundary
    records can access their true predecessor.
    """
    trajectory = Window.partitionBy("source_entity_id").orderBy(
        "timestamp_utc", "point_id"
    )
    points = (
        bronze.withColumn(
            "point_geom_wgs84",
            F.expr(
                "ST_Point(CAST(GPS_LON AS Decimal(24,20)), CAST(GPS_LAT AS Decimal(24,20)))"
            ),
        )
        .withColumn(
            "geom_2180",
            F.expr(
                f"ST_Transform(point_geom_wgs84, '{config.CRS_WGS84}', '{config.CRS_METRIC_PL}')"
            ),
        )
        .withColumn("prev_timestamp_utc", F.lag("timestamp_utc").over(trajectory))
        .withColumn("prev_geom_2180", F.lag("geom_2180").over(trajectory))
        .withColumn(
            "time_delta_sec",
            F.col("timestamp_utc").cast("long")
            - F.col("prev_timestamp_utc").cast("long"),
        )
        .withColumn(
            "dist_delta_meters",
            F.when(
                F.col("prev_geom_2180").isNotNull(),
                F.expr("ST_Distance(geom_2180, prev_geom_2180)"),
            ).otherwise(F.lit(None).cast("double")),
        )
    )
    interval_valid = (
        (F.col("time_delta_sec") > 0)
        & (F.col("time_delta_sec") <= config.MAX_GAP_SECONDS)
        & F.col("dist_delta_meters").isNotNull()
    )
    raw_gps_speed = F.col("GPS_SPEED").cast("double")
    gps_speed_finite = (
        raw_gps_speed.isNotNull()
        & ~F.isnan(raw_gps_speed)
        & (F.abs(raw_gps_speed) < F.lit(float("inf")))
    )
    gps_speed_valid = gps_speed_finite & (raw_gps_speed >= F.lit(0.0))
    derived = (
        points.withColumn(
            "displacement_speed_kmh",
            F.when(
                interval_valid,
                F.col("dist_delta_meters") / F.col("time_delta_sec") * 3.6,
            ).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "gps_speed_kmh",
            F.when(
                gps_speed_valid,
                raw_gps_speed * F.lit(config.GPS_SPEED_TO_KMH_FACTOR),
            ).otherwise(F.lit(None).cast("double")),
        )
    )
    if config.SPEED_SOURCE == "gps_speed":
        derived = (
            derived.withColumn(
                "calc_speed_kmh",
                F.when(gps_speed_valid, F.col("gps_speed_kmh")).otherwise(
                    F.lit(None).cast("double")
                ),
            )
            .withColumn(
                "speed_source",
                F.when(gps_speed_valid, "GPS_SPEED_CM_PER_S_PRIMARY").otherwise(
                    "UNUSABLE_GPS_SPEED"
                ),
            )
            .withColumn(
                "speed_derivation_reason",
                F.when(gps_speed_valid, "USABLE_GPS_SPEED_CM_PER_S")
                .when(raw_gps_speed.isNull(), "GPS_SPEED_MISSING")
                .when(~gps_speed_finite, "GPS_SPEED_NON_FINITE")
                .when(raw_gps_speed < F.lit(0.0), "GPS_SPEED_NEGATIVE")
                .otherwise("GPS_SPEED_UNUSABLE"),
            )
        )
    else:
        derived = (
            derived.withColumn("calc_speed_kmh", F.col("displacement_speed_kmh"))
            .withColumn(
                "speed_source",
                F.when(interval_valid, "DISPLACEMENT_BETWEEN_PINGS").otherwise(
                    "UNUSABLE_INTERVAL"
                ),
            )
            .withColumn(
                "speed_derivation_reason",
                F.when(interval_valid, "USABLE_DISPLACEMENT_INTERVAL")
                .when(F.col("prev_timestamp_utc").isNull(), "FIRST_POINT_FOR_SOURCE")
                .when(F.col("time_delta_sec") <= 0, "NON_INCREASING_TIMESTAMP")
                .when(F.col("time_delta_sec") > config.MAX_GAP_SECONDS, "GAP_EXCEEDS_MAX")
                .when(F.col("prev_geom_2180").isNull(), "PREVIOUS_GEOMETRY_MISSING")
                .otherwise("INTERVAL_NOT_USABLE"),
            )
        )
    return (
        derived
        .withColumn(
            "traj_azimuth",
            F.when(
                interval_valid
                & (
                    F.col("dist_delta_meters") >= config.MIN_MOVEMENT_FOR_HEADING_METERS
                ),
                F.expr("ST_Azimuth(prev_geom_2180, geom_2180)"),
            ).otherwise(F.lit(None).cast("double")),
        )
    )


def _selected_speed_is_valid() -> F.Column:
    """Eligibility before matching for the configured primary speed source."""
    if config.SPEED_SOURCE == "gps_speed":
        return F.col("calc_speed_kmh").isNotNull() & (F.col("calc_speed_kmh") >= 0.0)
    return F.col("calc_speed_kmh").between(
        config.MIN_VALID_SPEED_KMH, config.MAX_VALID_SPEED_KMH
    )


def _select_best_road(kinematics: DataFrame, road_segments: DataFrame) -> DataFrame:
    valid_speed = kinematics.filter(_selected_speed_is_valid())
    joined = valid_speed.join(
        road_segments.hint("broadcast"),
        F.expr("ST_Intersects(geom_2180, road_buffer)"),
        "inner",
    )
    directional_diff = circular_angle_difference(
        F.col("traj_azimuth"), F.col("road_azimuth")
    )
    alignment_diff = F.least(directional_diff, F.lit(math.pi) - directional_diff)
    distance_score = F.col("dist_to_axis") / F.lit(config.ROAD_BUFFER_METERS)
    heading_score = alignment_diff / F.lit(math.pi / 2.0)
    scored = (
        joined.withColumn(
            "dist_to_axis", F.expr("ST_Distance(geom_2180, road_segment_geom)")
        )
        .withColumn("azimuth_diff", directional_diff)
        .withColumn("heading_alignment_diff", alignment_diff)
        .withColumn(
            "match_score",
            F.when(F.col("traj_azimuth").isNull(), distance_score).otherwise(
                (1.0 - config.HEADING_SCORE_WEIGHT) * distance_score
                + config.HEADING_SCORE_WEIGHT * heading_score
            ),
        )
    )

    # Multiple atomic segments can belong to one road feature. First retain the
    # best local segment for each feature, then compare distinct road features.
    segment_window = Window.partitionBy("point_id", "road_feature_id").orderBy(
        F.col("match_score"),
        F.col("dist_to_axis"),
        F.col("heading_alignment_diff").asc_nulls_last(),
        F.col("match_segment_id"),
    )
    per_feature = scored.withColumn(
        "segment_rank_within_feature", F.row_number().over(segment_window)
    ).filter(F.col("segment_rank_within_feature") == 1)

    radius_1, radius_2, radius_3 = config.MATCHING_SENSITIVITY_RADII_METERS
    point_window = Window.partitionBy("point_id")
    per_feature = (
        per_feature.withColumn(
            "candidate_count_primary_radius",
            F.sum(
                (F.col("dist_to_axis") <= config.ROAD_BUFFER_METERS).cast("long")
            ).over(point_window),
        )
        .withColumn(
            "candidate_count_radius_1",
            F.sum((F.col("dist_to_axis") <= radius_1).cast("long")).over(
                point_window
            ),
        )
        .withColumn(
            "candidate_count_radius_2",
            F.sum((F.col("dist_to_axis") <= radius_2).cast("long")).over(
                point_window
            ),
        )
        .withColumn(
            "candidate_count_radius_3",
            F.sum((F.col("dist_to_axis") <= radius_3).cast("long")).over(
                point_window
            ),
        )
    )
    # A primary-radius candidate always outranks a sensitivity-only candidate.
    # The wider search therefore measures recoverability without changing any
    # assignment that the configured primary method could already make.
    road_window = Window.partitionBy("point_id").orderBy(
        (F.col("dist_to_axis") <= config.ROAD_BUFFER_METERS).desc(),
        F.col("match_score"),
        F.col("dist_to_axis"),
        F.col("heading_alignment_diff").asc_nulls_last(),
        F.col("road_feature_id"),
    )
    ranked = (
        per_feature.withColumn("road_rank", F.row_number().over(road_window))
        .withColumn(
            "candidate_count_sensitivity_max_radius",
            F.count("road_feature_id").over(point_window),
        )
        .withColumn("next_match_score", F.lead("match_score").over(road_window))
        .withColumn("next_dist_to_axis", F.lead("dist_to_axis").over(road_window))
    )
    return (
        ranked.filter(F.col("road_rank") == 1)
        .withColumn(
            "within_primary_match_radius",
            F.col("dist_to_axis") <= config.ROAD_BUFFER_METERS,
        )
        .withColumn(
            "candidate_count",
            F.when(
                F.col("within_primary_match_radius"),
                F.col("candidate_count_primary_radius"),
            ).otherwise(F.col("candidate_count_sensitivity_max_radius")),
        )
        .withColumn(
            "selected_match_radius_m",
            F.when(F.col("dist_to_axis") <= radius_1, F.lit(radius_1))
            .when(F.col("dist_to_axis") <= radius_2, F.lit(radius_2))
            .otherwise(F.lit(radius_3)),
        )
        .withColumn(
            "match_radius_tier",
            F.when(
                F.col("dist_to_axis") <= radius_1,
                F.lit(f"LE_{radius_1:g}_M"),
            )
            .when(
                F.col("dist_to_axis") <= radius_2,
                F.lit(f"GT_{radius_1:g}_LE_{radius_2:g}_M"),
            )
            .otherwise(F.lit(f"GT_{radius_2:g}_LE_{radius_3:g}_M")),
        )
        .withColumn(
            "score_margin",
            F.when(
                F.col("candidate_count") > 1,
                F.col("next_match_score") - F.col("match_score"),
            ),
        )
        .withColumn(
            "dist_margin_m",
            F.when(
                F.col("candidate_count") > 1,
                F.col("next_dist_to_axis") - F.col("dist_to_axis"),
            ),
        )
    )


def _add_direction_weather_and_outlier_flags(
    matched: DataFrame, weather: DataFrame
) -> DataFrame:
    direction = _add_direction_and_weather(matched, weather)
    return _add_outlier_flags(direction, _road_speed_stats(direction))


def _unmatched_distance_summary(
    kinematics: DataFrame, road_segments: DataFrame, diagnostic_radius_m: float = 150.0
) -> DataFrame:
    """Summarise valid-speed points that have no primary/sensitivity road match.

    The production matcher only needs candidates up to the largest configured
    sensitivity radius. This aggregate-only diagnostic deliberately looks farther
    afield so recovery work can distinguish modest GPS/network offsets from
    genuinely off-network telemetry. It never changes a match or writes rejected
    point locations.
    """
    max_radius = max(
        config.ROAD_BUFFER_METERS, *config.MATCHING_SENSITIVITY_RADII_METERS
    )
    valid_speed = kinematics.filter(_selected_speed_is_valid()).select(
        "point_id", "year", "month", "inside_study_aoi", "geom_2180"
    )
    diagnostic_segments = road_segments.select(
        "match_segment_id", "road_segment_geom"
    ).withColumn(
        "diagnostic_buffer",
        F.expr(
            f"ST_Buffer(road_segment_geom, {float(diagnostic_radius_m)}, 'quad_segs=2')"
        ),
    )
    nearest = (
        valid_speed.join(
            diagnostic_segments.hint("broadcast"),
            F.expr("ST_Intersects(geom_2180, diagnostic_buffer)"),
            "left",
        )
        .withColumn(
            "candidate_distance_m", F.expr("ST_Distance(geom_2180, road_segment_geom)")
        )
        .groupBy("point_id", "year", "month", "inside_study_aoi")
        .agg(F.min("candidate_distance_m").alias("nearest_road_distance_m"))
        .filter(
            F.col("nearest_road_distance_m").isNull()
            | (F.col("nearest_road_distance_m") > F.lit(max_radius))
        )
        .withColumn(
            "distance_bucket",
            F.when(F.col("nearest_road_distance_m") <= 15.0, "LE_15_M")
            .when(F.col("nearest_road_distance_m") <= 25.0, "GT_15_LE_25_M")
            .when(F.col("nearest_road_distance_m") <= 40.0, "GT_25_LE_40_M")
            .when(F.col("nearest_road_distance_m") <= 75.0, "GT_40_LE_75_M")
            .when(
                F.col("nearest_road_distance_m") <= diagnostic_radius_m,
                "GT_75_LE_150_M",
            )
            .otherwise("GT_150_M_OR_NO_CANDIDATE"),
        )
    )
    return nearest.groupBy(
        "year", "month", "inside_study_aoi", "distance_bucket"
    ).agg(F.count("*").alias("unmatched_point_count"))


def _speed_derivation_summary(kinematics: DataFrame) -> DataFrame:
    """Report every speed-derivation outcome without retaining excluded points."""
    return kinematics.groupBy(
        "year", "month", "inside_study_aoi", "speed_derivation_reason"
    ).agg(F.count("*").alias("point_count"))


def _speed_source_comparison_values(kinematics: DataFrame) -> DataFrame:
    """Return valid paired speeds for a globally aggregated technical QA report."""
    difference = F.col("gps_speed_kmh") - F.col("displacement_speed_kmh")
    return (
        kinematics.filter(
            F.col("gps_speed_kmh").isNotNull()
            & F.col("displacement_speed_kmh").isNotNull()
        )
        .withColumn("_signed_difference_kmh", difference)
        .withColumn("_absolute_difference_kmh", F.abs(difference))
        .select(
            "year",
            "month",
            "inside_study_aoi",
            "_signed_difference_kmh",
            "_absolute_difference_kmh",
        )
    )


def _speed_source_comparison_summary(kinematics: DataFrame) -> DataFrame:
    """Compare provider and displacement speeds without adding fields to Gold."""
    return (
        _speed_source_comparison_values(kinematics)
        .groupBy("year", "month", "inside_study_aoi")
        .agg(
            F.count("*").alias("overlap_count"),
            F.avg("_signed_difference_kmh").alias("mean_signed_difference_kmh"),
            F.expr("percentile_approx(_absolute_difference_kmh, 0.5, 10000)").alias(
                "median_absolute_difference_kmh"
            ),
            F.expr("percentile_approx(_absolute_difference_kmh, 0.95, 10000)").alias(
                "p95_absolute_difference_kmh"
            ),
        )
    )


def _add_direction_and_weather(matched: DataFrame, weather: DataFrame) -> DataFrame:
    """Add direction and pinned weather without whole-dataset statistics."""
    return (
        matched.withColumn(
            "travel_direction",
            F.when(F.col("traj_azimuth").isNull(), "UNKNOWN_STATIONARY_OR_SHORT_MOVE")
            .when(
                (F.col("azimuth_diff") > math.pi / 2.0) & F.col("oneway"),
                "AGAINST_ONEWAY_DIGITIZED_DIRECTION",
            )
            .when(F.col("azimuth_diff") > math.pi / 2.0, "BACKWARD")
            .otherwise("FORWARD"),
        )
        .withColumn("weather_hour_utc", F.date_trunc("hour", F.col("timestamp_utc")))
        .join(weather, "weather_hour_utc", "left")
        .withColumn(
            "weather_join_status",
            F.when(
                F.col("weather_category").isNull(), "MISSING_WEATHER_HOUR"
            ).otherwise("MATCHED"),
        )
        .withColumn(
            "weather_category",
            F.coalesce(F.col("weather_category"), F.lit("UNKNOWN_WEATHER")),
        )
        .withColumn(
            "weather_model",
            F.coalesce(F.col("weather_model"), F.lit(config.WEATHER_MODEL)),
        )
    )


def _road_speed_stats(direction: DataFrame) -> DataFrame:
    """Calculate dataset-wide per-road speed support and IQR statistics."""
    eligible = direction
    if "within_primary_match_radius" in direction.columns:
        eligible = direction.filter(F.col("within_primary_match_radius"))
    return eligible.groupBy("road_feature_id").agg(
        F.count("calc_speed_kmh").alias("road_speed_sample_count"),
        F.expr("percentile_approx(calc_speed_kmh, 0.25, 10000)").alias(
            "road_speed_q1_kmh"
        ),
        F.expr("percentile_approx(calc_speed_kmh, 0.75, 10000)").alias(
            "road_speed_q3_kmh"
        ),
    )


def _add_outlier_flags(direction: DataFrame, road_stats: DataFrame) -> DataFrame:
    """Join global road statistics and apply the publication outlier rule."""
    with_stats = direction.join(
        road_stats.hint("broadcast"), "road_feature_id", "left"
    ).withColumn(
        "road_speed_iqr_kmh", F.col("road_speed_q3_kmh") - F.col("road_speed_q1_kmh")
    )
    return with_stats.withColumn(
        "high_speed_outlier_threshold_kmh",
        F.when(
            F.col("road_speed_sample_count") >= config.OUTLIER_MIN_GROUP_SIZE,
            F.col("road_speed_q3_kmh")
            + config.OUTLIER_IQR_MULTIPLIER * F.col("road_speed_iqr_kmh"),
        ),
    ).withColumn(
        "is_high_speed_iqr_outlier",
        F.coalesce(
            F.col("calc_speed_kmh") > F.col("high_speed_outlier_threshold_kmh"),
            F.lit(False),
        ),
    )


def _write_silver_qa(
    silver: DataFrame,
    bronze_rows: int,
    usable_speed_rows: int,
) -> int:
    summary = (
        silver.agg(
            F.count("*").alias("matched_rows_at_sensitivity_max_radius"),
            F.sum((F.col("weather_join_status") != "MATCHED").cast("long")).alias(
                "missing_weather_rows"
            ),
            F.sum(F.col("is_high_speed_iqr_outlier").cast("long")).alias(
                "high_speed_iqr_outlier_rows"
            ),
            F.sum((F.col("candidate_count") > 1).cast("long")).alias(
                "multi_candidate_rows"
            ),
            F.sum(F.col("within_primary_match_radius").cast("long")).alias(
                "primary_radius_matched_rows"
            ),
            F.sum((~F.col("within_primary_match_radius")).cast("long")).alias(
                "sensitivity_only_matched_rows"
            ),
        )
        .first()
        .asDict()
    )
    summary["bronze_rows"] = bronze_rows
    summary["usable_speed_rows"] = usable_speed_rows
    rows = [
        ("ATTRITION", None, None, metric, int(value or 0))
        for metric, value in summary.items()
    ]
    rows.extend(
        (
            "SILVER_DISTRIBUTION",
            int(row["year"]),
            int(row["month"]),
            f"speed_source={row['speed_source']};weather_join={row['weather_join_status']}",
            int(row["count"]),
        )
        for row in silver.groupBy(
            "year", "month", "speed_source", "weather_join_status"
        )
        .count()
        .collect()
    )
    report = silver.sparkSession.createDataFrame(
        rows, "report_type string, year int, month int, category string, count long"
    )
    report.coalesce(1).write.mode("overwrite").parquet(str(config.SILVER_QA_PATH))
    return int(summary["matched_rows_at_sensitivity_max_radius"] or 0)


def process_silver_map_matching(
    spark: SparkSession,
    force_rebuild: bool = False,
    work_root_parent: Path | None = None,
    source_shards: int = 1,
) -> Path:
    """Build Silver from full histories, then partition only the completed result."""
    if source_shards < 1:
        raise ValueError("source_shards must be at least 1")
    config.ensure_directories()
    if (
        config.SILVER_DIR.exists()
        and any(config.SILVER_DIR.rglob("*.parquet"))
        and layer_is_complete(
            config.SILVER_DIR,
            layer="silver",
            configuration_fingerprint=config.configuration_fingerprint(),
        )
        and not force_rebuild
    ):
        logger.info("Silver dataset already exists at %s; skipping", config.SILVER_DIR)
        return config.SILVER_DIR
    if config.SILVER_DIR.exists() and not layer_is_complete(
        config.SILVER_DIR,
        layer="silver",
        configuration_fingerprint=config.configuration_fingerprint(),
    ):
        logger.warning(
            "Silver directory %s lacks a completion marker; treating it as an "
            "incomplete previous run and rebuilding it",
            config.SILVER_DIR,
        )
    if not config.BRONZE_DIR.exists() or not any(config.BRONZE_DIR.rglob("*.parquet")):
        raise FileNotFoundError(f"Bronze dataset not found at {config.BRONZE_DIR}")

    chunk_paths = sorted(config.BRONZE_DIR.glob("year=*/quarter=*"))
    if not chunk_paths:
        raise FileNotFoundError(
            f"Partitioned Bronze year/quarter data not found at {config.BRONZE_DIR}"
        )

    road_segments = _load_matching_segments(spark).persist(StorageLevel.DISK_ONLY)
    weather = _load_weather(spark)
    # This intentionally is not a temporary directory.  Direction chunks,
    # carries and their manifests are the recovery boundary for Silver.
    work_root = Path(work_root_parent or config.SILVER_DIR.parent) / ".silver-work"
    run_fingerprint = configuration_fingerprint(config.scientific_parameters())
    if force_rebuild:
        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    direction_root = work_root / "direction"
    unmatched_root = work_root / "unmatched-diagnostics"
    speed_exclusion_root = work_root / "speed-exclusion-diagnostics"
    speed_comparison_root = work_root / "speed-source-comparison"
    exclusion_ledger_root = work_root / "speed-exclusion-ledger"
    carry_root = work_root / "carry"
    stats_path = work_root / "road_stats"
    bronze_rows = 0
    usable_speed_rows = 0
    direction_chunks: list[Path] = []
    unmatched_chunks: list[Path] = []
    speed_exclusion_chunks: list[Path] = []
    speed_comparison_chunks: list[Path] = []
    exclusion_ledger_chunks: list[Path] = []

    output_columns = [
        "point_id",
        "source_entity_id",
        "source_entity_id_semantic",
        "timestamp_utc",
        "timestamp_warsaw",
        "timestamp_source",
        "timestamp_disagreement_sec",
        "timestamp_sources_disagree",
        "year",
        "month",
        "quarter",
        "season",
        "day",
        "hour",
        "day_of_week_iso",
        "is_weekend",
        "inside_study_aoi",
        "inside_road_extraction_bbox",
        "GPS_LAT",
        "GPS_LON",
        "calc_speed_kmh",
        "speed_source",
        "time_delta_sec",
        "dist_delta_meters",
        "traj_azimuth",
        "signal_mode",
        "weather_category",
        "weather_model",
        "weather_join_status",
        "road_feature_id",
        "match_segment_id",
        "osm_id",
        "name",
        "highway",
        "maxspeed_raw",
        "maxspeed_kmh",
        "maxspeed_source",
        "oneway",
        "oneway_raw",
        "travel_direction",
        "dist_to_axis",
        "azimuth_diff",
        "heading_alignment_diff",
        "match_score",
        "candidate_count",
        "candidate_count_sensitivity_max_radius",
        "candidate_count_primary_radius",
        "candidate_count_radius_1",
        "candidate_count_radius_2",
        "candidate_count_radius_3",
        "within_primary_match_radius",
        "selected_match_radius_m",
        "match_radius_tier",
        "score_margin",
        "dist_margin_m",
        "road_speed_sample_count",
        "road_speed_q1_kmh",
        "road_speed_q3_kmh",
        "road_speed_iqr_kmh",
        "high_speed_outlier_threshold_kmh",
        "is_high_speed_iqr_outlier",
    ]
    try:
        work_items = [
            (quarter_index, shard_index, chunk_path)
            for quarter_index, chunk_path in enumerate(chunk_paths, start=1)
            for shard_index in range(source_shards)
        ]
        for work_index, (quarter_index, shard_index, chunk_path) in enumerate(
            work_items, start=1
        ):
            if source_shards == 1:
                chunk_name = f"chunk-{quarter_index:02d}"
                step_name = f"direction-{quarter_index:04d}"
            else:
                chunk_name = f"quarter-{quarter_index:02d}/shard-{shard_index:03d}"
                step_name = f"direction-{quarter_index:04d}-shard-{shard_index:03d}"
            logger.info(
                "Processing Silver history work item %d/%d: %s (source shard %d/%d)",
                work_index,
                len(work_items),
                chunk_path.relative_to(config.BRONZE_DIR),
                shard_index + 1,
                source_shards,
            )
            direction_path = direction_root / chunk_name
            unmatched_path = unmatched_root / chunk_name
            speed_exclusion_path = speed_exclusion_root / chunk_name
            speed_comparison_path = speed_comparison_root / chunk_name
            exclusion_ledger_path = exclusion_ledger_root / chunk_name
            carry_path = carry_root / chunk_name
            if step_is_complete(
                work_root,
                layer="silver",
                step=step_name,
                configuration_fingerprint=run_fingerprint,
            ) and unmatched_path.exists() and speed_exclusion_path.exists() and speed_comparison_path.exists() and exclusion_ledger_path.exists():
                # A completed chunk includes the exact carry state needed by
                # the next chunk, so this is a genuine resume point.
                payload = json.loads(
                    (work_root / ".karetki-step-manifests" / f"{step_name}.json").read_text()
                )
                details = payload.get("details", {})
                bronze_rows += int(details.get("bronze_rows", 0))
                usable_speed_rows += int(details.get("usable_speed_rows", 0))
                direction_chunks.append(direction_path)
                unmatched_chunks.append(unmatched_path)
                speed_exclusion_chunks.append(speed_exclusion_path)
                speed_comparison_chunks.append(speed_comparison_path)
                exclusion_ledger_chunks.append(exclusion_ledger_path)
                logger.info(
                    "Resuming Silver: work item %d/%d already complete",
                    work_index,
                    len(work_items),
                )
                continue
            current = (
                spark.read.option("basePath", str(config.BRONZE_DIR))
                .parquet(str(chunk_path))
                .withColumn("_is_carry", F.lit(False))
            )
            if source_shards > 1:
                current = current.filter(
                    F.pmod(F.hash("source_entity_id"), F.lit(source_shards))
                    == F.lit(shard_index)
                )
            bronze_columns = [name for name in current.columns if name != "_is_carry"]
            if source_shards == 1:
                previous_chunk_name = f"chunk-{quarter_index - 1:02d}"
            else:
                previous_chunk_name = (
                    f"quarter-{quarter_index - 1:02d}/shard-{shard_index:03d}"
                )
            previous_carry = carry_root / previous_chunk_name
            if quarter_index > 1 and previous_carry.exists():
                carry = spark.read.parquet(str(previous_carry)).withColumn(
                    "_is_carry", F.lit(True)
                )
                combined = carry.unionByName(current)
            else:
                combined = current

            all_kinematics = _derive_trajectory_kinematics(
                combined.repartition("source_entity_id")
            ).persist(StorageLevel.DISK_ONLY)
            current_kinematics = all_kinematics.filter(~F.col("_is_carry"))
            metrics = (
                current_kinematics.agg(
                    F.count("*").alias("bronze_rows"),
                    F.sum(F.col("calc_speed_kmh").isNotNull().cast("long")).alias(
                        "usable_speed_rows"
                    ),
                )
                .first()
                .asDict()
            )
            bronze_rows += int(metrics["bronze_rows"] or 0)
            usable_speed_rows += int(metrics["usable_speed_rows"] or 0)

            direction = _add_direction_and_weather(
                _select_best_road(current_kinematics, road_segments), weather
            )
            direction.write.mode("overwrite").parquet(str(direction_path))
            direction_chunks.append(direction_path)
            _unmatched_distance_summary(current_kinematics, road_segments).write.mode(
                "overwrite"
            ).parquet(str(unmatched_path))
            unmatched_chunks.append(unmatched_path)
            _speed_derivation_summary(current_kinematics).write.mode("overwrite").parquet(
                str(speed_exclusion_path)
            )
            speed_exclusion_chunks.append(speed_exclusion_path)
            _speed_source_comparison_values(current_kinematics).write.mode(
                "overwrite"
            ).parquet(str(speed_comparison_path))
            speed_comparison_chunks.append(speed_comparison_path)
            (
                current_kinematics.filter(F.col("calc_speed_kmh").isNull())
                .select(
                    "point_id", "source_entity_id", "timestamp_utc", "timestamp_warsaw",
                    "year", "month", "GPS_LAT", "GPS_LON", "time_delta_sec",
                    "dist_delta_meters", "speed_derivation_reason",
                )
                .write.mode("overwrite")
                .parquet(str(exclusion_ledger_path))
            )
            exclusion_ledger_chunks.append(exclusion_ledger_path)

            latest = Window.partitionBy("source_entity_id").orderBy(
                F.col("timestamp_utc").desc(), F.col("point_id").desc()
            )
            (
                all_kinematics.select(*bronze_columns)
                .withColumn("_carry_rank", F.row_number().over(latest))
                .filter(F.col("_carry_rank") == 1)
                .drop("_carry_rank")
                .write.mode("overwrite")
                .parquet(str(carry_path))
            )
            all_kinematics.unpersist()
            write_step_completion_manifest(
                work_root,
                layer="silver",
                step=step_name,
                artifacts=[
                    direction_path.relative_to(work_root),
                    unmatched_path.relative_to(work_root),
                    speed_exclusion_path.relative_to(work_root),
                    speed_comparison_path.relative_to(work_root),
                    exclusion_ledger_path.relative_to(work_root),
                    carry_path.relative_to(work_root),
                ],
                configuration_fingerprint=run_fingerprint,
                details={"bronze_rows": int(metrics["bronze_rows"] or 0), "usable_speed_rows": int(metrics["usable_speed_rows"] or 0)},
            )

        logger.info("Writing unmatched-distance diagnostics...")
        unmatched_summary = spark.read.parquet(*[str(path) for path in unmatched_chunks])
        (
            unmatched_summary.groupBy(
                "year", "month", "inside_study_aoi", "distance_bucket"
            )
            .agg(F.sum("unmatched_point_count").alias("unmatched_point_count"))
            .orderBy("year", "month", "inside_study_aoi", "distance_bucket")
            .write.mode("overwrite")
            .parquet(str(config.UNMATCHED_QA_PATH))
        )
        speed_exclusions = spark.read.parquet(
            *[str(path) for path in speed_exclusion_chunks]
        )
        (
            speed_exclusions.groupBy(
                "year", "month", "inside_study_aoi", "speed_derivation_reason"
            )
            .agg(F.sum("point_count").alias("point_count"))
            .orderBy("year", "month", "inside_study_aoi", "speed_derivation_reason")
            .write.mode("overwrite")
            .parquet(str(config.SPEED_EXCLUSION_QA_PATH))
        )
        logger.info("Writing GPS/displacement speed-comparison diagnostics...")
        speed_comparisons = spark.read.parquet(
            *[str(path) for path in speed_comparison_chunks]
        )
        (
            speed_comparisons
            .groupBy("year", "month", "inside_study_aoi")
            .agg(
                F.count("*").alias("overlap_count"),
                F.avg("_signed_difference_kmh").alias(
                    "mean_signed_difference_kmh"
                ),
                F.expr(
                    "percentile_approx(_absolute_difference_kmh, 0.5, 10000)"
                ).alias("median_absolute_difference_kmh"),
                F.expr(
                    "percentile_approx(_absolute_difference_kmh, 0.95, 10000)"
                ).alias("p95_absolute_difference_kmh"),
            )
            .orderBy("year", "month", "inside_study_aoi")
            .write.mode("overwrite")
            .parquet(str(config.SPEED_SOURCE_COMPARISON_QA_PATH))
        )
        (
            spark.read.parquet(*[str(path) for path in exclusion_ledger_chunks])
            .write.mode("overwrite")
            .partitionBy("year", "month")
            .parquet(str(config.SILVER_EXCLUSIONS_DIR))
        )

        logger.info("Calculating global per-road Silver statistics...")
        if not step_is_complete(work_root, layer="silver", step="road-stats", configuration_fingerprint=run_fingerprint):
            all_direction = spark.read.parquet(*[str(path) for path in direction_chunks])
            _road_speed_stats(all_direction).write.mode("overwrite").parquet(str(stats_path))
            write_step_completion_manifest(
                work_root, layer="silver", step="road-stats", artifacts=["road_stats"],
                configuration_fingerprint=run_fingerprint,
            )
        road_stats = spark.read.parquet(str(stats_path))

        has_recoverable_output = config.SILVER_DIR.exists() and any(
            config.SILVER_DIR.rglob("*.parquet")
        )
        has_published_steps = any(
            step_is_complete(work_root, layer="silver", step=f"publish-{index:04d}", configuration_fingerprint=run_fingerprint)
            for index in range(1, len(direction_chunks) + 1)
        )
        # With external-drive staging the work manifests live at the durable
        # destination while the staged output is deliberately ephemeral.  In
        # that case reuse transformations but republish all output chunks.
        can_resume_publish = has_recoverable_output and has_published_steps
        if config.SILVER_DIR.exists() and not can_resume_publish:
            shutil.rmtree(config.SILVER_DIR)
        logger.info("Writing corrected Silver dataset to %s", config.SILVER_DIR)
        for index, direction_path in enumerate(direction_chunks, start=1):
            logger.info(
                "Finalizing Silver output chunk %d/%d...",
                index,
                len(direction_chunks),
            )
            finalized = _add_outlier_flags(
                spark.read.parquet(str(direction_path)), road_stats
            )
            publish_step = f"publish-{index:04d}"
            if can_resume_publish and step_is_complete(work_root, layer="silver", step=publish_step, configuration_fingerprint=run_fingerprint):
                logger.info("Resuming Silver: output chunk %d/%d already published", index, len(direction_chunks))
                continue
            (
                finalized.select(*output_columns)
                .write.mode("append")
                .partitionBy("year", "month")
                .parquet(str(config.SILVER_DIR))
            )
            write_step_completion_manifest(
                work_root, layer="silver", step=publish_step,
                artifacts=[str(path.relative_to(work_root)) for path in direction_chunks],
                configuration_fingerprint=run_fingerprint,
            )

        written_silver = spark.read.parquet(str(config.SILVER_DIR))
        written_silver_rows = _write_silver_qa(
            written_silver, bronze_rows, usable_speed_rows
        )
        write_layer_completion_marker(
            config.SILVER_DIR,
            {
                "layer": "silver",
                "rows": written_silver_rows,
                "dataset_version": config.DATASET_VERSION,
                "artifacts": [
                    str(path.relative_to(config.SILVER_DIR))
                    for path in sorted(config.SILVER_DIR.glob("year=*/month=*"))
                ],
                "configuration_fingerprint": configuration_fingerprint(
                    config.scientific_parameters()
                ),
            },
        )
    finally:
        road_segments.unpersist()
    logger.info("Silver map matching completed")
    return config.SILVER_DIR

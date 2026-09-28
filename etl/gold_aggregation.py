"""Gold aggregation with neutral semantics and explicit observation support."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from config import PipelineConfig, config
from pipeline_utils import (
    configuration_fingerprint,
    step_is_complete,
    write_layer_completion_marker,
    write_step_completion_manifest,
)

logger = logging.getLogger(__name__)


def _named_roads(roads: DataFrame) -> DataFrame:
    return roads.withColumn(
        "name",
        F.when(
            F.col("name").isNull() | (F.trim(F.col("name")) == ""),
            F.when(
                F.col("highway").endswith("_link"),
                F.concat(F.lit("Unnamed link ("), F.col("highway"), F.lit(")")),
            ).otherwise(
                F.concat(F.lit("Unnamed road ("), F.col("highway"), F.lit(")"))
            ),
        ).otherwise(F.col("name")),
    )


def _prepare_base_roads(spark: SparkSession) -> DataFrame:
    roads = spark.read.parquet(str(config.OSM_ROADS_PARQUET_PATH))
    required = {
        "road_feature_id",
        "osm_id",
        "maxspeed_kmh",
        "maxspeed_source",
        "intersects_official_krakow_bbox",
        "geometry",
    }
    missing = required.difference(roads.columns)
    if missing:
        raise ValueError(f"Base-road schema is missing: {sorted(missing)}")
    prepared = _named_roads(roads).withColumn(
        "road_length_m", F.expr("ST_Length(ST_GeomFromWKB(geometry))")
    )
    return prepared.select(
        "road_feature_id",
        "osm_id",
        "osm_ids_raw",
        "name",
        "highway",
        "maxspeed_raw",
        F.col("maxspeed_kmh").alias("operating_speed_reference_kmh"),
        F.col("maxspeed_source").alias("operating_speed_reference_source"),
        "oneway",
        "oneway_raw",
        "oneway_reversed",
        "intersects_official_krakow_bbox",
        F.round("road_length_m", 3).alias("road_length_m"),
        F.expr(
            f"ST_AsBinary(ST_Transform(ST_GeomFromWKB(geometry), '{config.CRS_METRIC_PL}', '{config.CRS_WGS84}'))"
        ).alias("geometry"),
    )


def _prepare_observations(silver: DataFrame) -> DataFrame:
    prepared = silver
    if not config.INCLUDE_SENSITIVITY_ONLY_MATCHES_IN_GOLD:
        prepared = prepared.filter(F.col("within_primary_match_radius"))
    if config.EXCLUDE_FLAGGED_SPEED_OUTLIERS_FROM_GOLD:
        prepared = prepared.filter(~F.col("is_high_speed_iqr_outlier"))

    day_names = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday",
    }
    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    day_map = F.create_map(
        *[
            value
            for pair in day_names.items()
            for value in (F.lit(pair[0]), F.lit(pair[1]))
        ]
    )
    month_map = F.create_map(
        *[
            value
            for pair in month_names.items()
            for value in (F.lit(pair[0]), F.lit(pair[1]))
        ]
    )
    return (
        prepared.withColumn("day_name", day_map[F.col("day_of_week_iso")])
        .withColumn("month_name", month_map[F.col("month")])
        .withColumn(
            "time_of_day",
            F.when(F.col("hour").between(7, 9), "PEAK_MORNING")
            .when(F.col("hour").between(10, 14), "OFF_PEAK_DAY")
            .when(F.col("hour").between(15, 18), "PEAK_EVENING")
            .otherwise("NIGHT"),
        )
        .withColumn(
            "spatial_scope",
            F.when(
                F.col("inside_study_aoi"), "OFFICIAL_KRAKOW_BOUNDING_BOX"
            ).otherwise("EXTENDED_PROCESSING_AREA"),
        )
        .withColumn("observation_date", F.to_date("timestamp_warsaw"))
    )


def aggregate_speed_profiles(source: DataFrame, group_columns: list[str]) -> DataFrame:
    """Aggregate point-sampled operating speeds with factual support measures."""
    aggregated = source.groupBy(*group_columns).agg(
        F.round(F.avg("calc_speed_kmh"), 2).alias("avg_travel_speed_kmh"),
        F.round(
            F.avg(
                F.when(
                    F.col("calc_speed_kmh") > config.RUNNING_SPEED_THRESHOLD_KMH,
                    F.col("calc_speed_kmh"),
                )
            ),
            2,
        ).alias("avg_running_speed_kmh"),
        F.round(F.expr("percentile_approx(calc_speed_kmh, 0.5, 10000)"), 2).alias(
            "median_speed_kmh"
        ),
        F.round(F.expr("percentile_approx(calc_speed_kmh, 0.85, 10000)"), 2).alias(
            "p85_speed_kmh"
        ),
        F.round(F.stddev("calc_speed_kmh"), 2).alias("stddev_speed_kmh"),
        F.count("point_id").alias("n_samples"),
        F.countDistinct("source_entity_id").alias("n_unique_source_ids"),
        F.countDistinct("observation_date").alias("n_unique_observation_days"),
        F.countDistinct("speed_source").alias("n_speed_derivation_methods"),
        F.round(
            F.avg(
                F.when(
                    F.col("calc_speed_kmh") <= config.RUNNING_SPEED_THRESHOLD_KMH,
                    1.0,
                ).otherwise(0.0)
            ),
            4,
        ).alias("stopped_observation_ratio"),
    )
    return (
        aggregated.withColumn(
            "delta_operating_speed_reference_kmh",
            F.round(
                F.col("avg_travel_speed_kmh") - F.col("operating_speed_reference_kmh"),
                2,
            ),
        )
        .withColumn(
            "meets_min_sample_count",
            F.col("n_samples") >= config.MIN_PROFILE_SAMPLE_COUNT,
        )
        .withColumn("min_sample_count", F.lit(config.MIN_PROFILE_SAMPLE_COUNT))
    )


def _write_spark_then_single_parquet(
    frame: DataFrame, path: Path, sort_columns: list[str]
) -> pd.DataFrame:
    """Generate via Spark, then consolidate into one canonically ordered file.

    Spark group-by output order depends on shuffle scheduling; sorting on the
    full grouping key before writing makes the published bytes reproducible
    across runs and machines (cell values were already deterministic).
    """
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    frame.write.mode("overwrite").parquet(str(path))
    pandas_frame = pd.read_parquet(path)
    if path.is_dir():
        shutil.rmtree(path)
    existing_sort = [column for column in sort_columns if column in pandas_frame.columns]
    pandas_frame = pandas_frame.sort_values(existing_sort, kind="stable").reset_index(
        drop=True
    )
    pandas_frame.to_parquet(path, index=False)
    return pandas_frame


def _write_geoparquet(
    frame: DataFrame, path: Path, sort_columns: list[str]
) -> gpd.GeoDataFrame:
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    frame.write.mode("overwrite").parquet(str(path))
    pandas_frame = pd.read_parquet(path)
    geodata = gpd.GeoDataFrame(
        pandas_frame,
        geometry=gpd.GeoSeries.from_wkb(pandas_frame["geometry"]),
        crs=config.CRS_WGS84,
    )
    if path.is_dir():
        shutil.rmtree(path)
    existing_sort = [column for column in sort_columns if column in geodata.columns]
    if existing_sort:
        geodata = geodata.sort_values(existing_sort, kind="stable").reset_index(drop=True)
    geodata.to_parquet(path, index=False)
    return geodata


def _write_gold_qa(
    spark: SparkSession,
    silver: DataFrame,
    filtered: DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    silver_summary = (
        silver.agg(
            F.count("*").alias("silver_input_rows"),
            F.sum(F.col("within_primary_match_radius").cast("long")).alias(
                "primary_radius_silver_rows"
            ),
            F.sum((~F.col("within_primary_match_radius")).cast("long")).alias(
                "sensitivity_only_silver_rows"
            ),
            F.sum(
                (
                    ~F.col("within_primary_match_radius")
                    & F.lit(not config.INCLUDE_SENSITIVITY_ONLY_MATCHES_IN_GOLD)
                ).cast("long")
            ).alias(
                "excluded_sensitivity_only_rows"
            ),
            F.sum(
                (
                    (
                        F.col("within_primary_match_radius")
                        | F.lit(config.INCLUDE_SENSITIVITY_ONLY_MATCHES_IN_GOLD)
                    )
                    & F.col("is_high_speed_iqr_outlier")
                    & F.lit(config.EXCLUDE_FLAGGED_SPEED_OUTLIERS_FROM_GOLD)
                ).cast("long")
            ).alias(
                "excluded_primary_high_speed_iqr_outlier_rows"
            ),
        )
        .first()
        .asDict()
    )
    rows = [
        ("silver_input_rows", silver_summary["silver_input_rows"]),
        (
            "primary_radius_silver_rows",
            silver_summary["primary_radius_silver_rows"] or 0,
        ),
        ("gold_included_rows", filtered.count()),
        (
            "sensitivity_only_silver_rows",
            silver_summary["sensitivity_only_silver_rows"] or 0,
        ),
        (
            "excluded_sensitivity_only_rows",
            silver_summary["excluded_sensitivity_only_rows"] or 0,
        ),
        (
            "excluded_primary_high_speed_iqr_outlier_rows",
            silver_summary["excluded_primary_high_speed_iqr_outlier_rows"] or 0,
        ),
    ]
    rows.extend((f"{name}_rows", len(table)) for name, table in tables.items())
    rows.extend(
        (f"{name}_sample_sum", int(table["n_samples"].sum()))
        for name, table in tables.items()
    )
    report = spark.createDataFrame(
        [(metric, int(value)) for metric, value in rows],
        "metric string, value long",
    )
    report.coalesce(1).write.mode("overwrite").parquet(
        str(config.PIPELINE_REPORTS_DIR / "gold_aggregation_reconciliation.parquet")
    )


def process_gold_aggregation(
    spark: SparkSession,
    force_rebuild: bool = False,
    work_root_parent: Path | None = None,
) -> Path:
    """Create relational profiles and an observed-combinations flat alternative."""
    config.ensure_directories()
    if not config.SILVER_DIR.exists() or not any(config.SILVER_DIR.rglob("*.parquet")):
        raise FileNotFoundError(f"Silver dataset not found at {config.SILVER_DIR}")

    run_fingerprint = configuration_fingerprint(config.scientific_parameters())
    # Gold's relational exports are expensive but deterministic.  Materialize
    # their two common inputs in a stable work area so a failed export resumes
    # at the export that failed, not at the Silver scan.
    work_root = Path(work_root_parent or config.GOLD_DIR.parent) / ".gold-work"
    if force_rebuild:
        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    base_checkpoint = work_root / "base_roads"
    observations_checkpoint = work_root / "observations"
    if not step_is_complete(work_root, layer="gold", step="prepare-base-roads", configuration_fingerprint=run_fingerprint):
        _prepare_base_roads(spark).write.mode("overwrite").parquet(str(base_checkpoint))
        write_step_completion_manifest(
            work_root, layer="gold", step="prepare-base-roads", artifacts=["base_roads"],
            configuration_fingerprint=run_fingerprint,
        )
    base_roads = spark.read.parquet(str(base_checkpoint)).persist(StorageLevel.DISK_ONLY)

    road_reference = base_roads.select(
        "road_feature_id",
        "operating_speed_reference_kmh",
        "operating_speed_reference_source",
    )
    if not step_is_complete(work_root, layer="gold", step="prepare-observations", configuration_fingerprint=run_fingerprint):
        silver = spark.read.parquet(str(config.SILVER_DIR))
        (
            _prepare_observations(silver).drop("maxspeed_kmh", "maxspeed_source")
            .join(road_reference, "road_feature_id", "inner")
            .write.mode("overwrite").parquet(str(observations_checkpoint))
        )
        write_step_completion_manifest(
            work_root, layer="gold", step="prepare-observations", artifacts=["observations"],
            configuration_fingerprint=run_fingerprint,
        )
    observations = spark.read.parquet(str(observations_checkpoint)).persist(StorageLevel.DISK_ONLY)
    common = [
        "road_feature_id",
        "operating_speed_reference_kmh",
        "operating_speed_reference_source",
        "travel_direction",
        "signal_mode",
        "weather_category",
        "spatial_scope",
    ]
    tables = {
        "season": aggregate_speed_profiles(observations, common + ["season"]),
        "month": aggregate_speed_profiles(
            observations, common + ["season", "month", "month_name"]
        ),
        "weekday": aggregate_speed_profiles(
            observations,
            common + ["day_of_week_iso", "day_name", "is_weekend"],
        ).withColumnRenamed("day_of_week_iso", "day_of_week"),
        "time_of_day": aggregate_speed_profiles(observations, common + ["time_of_day"]),
    }
    flat_dimensions = common + [
        "season",
        "month",
        "month_name",
        "day_of_week_iso",
        "day_name",
        "is_weekend",
        "time_of_day",
    ]
    flat = aggregate_speed_profiles(observations, flat_dimensions).withColumnRenamed(
        "day_of_week_iso", "day_of_week"
    )

    base_gdf = _write_geoparquet(
        base_roads, config.GOLD_BASE_GEOPARQUET_PATH, ["road_feature_id"]
    )
    base_gdf.drop(columns=["geometry"]).to_csv(config.GOLD_BASE_CSV_PATH, index=False)
    write_step_completion_manifest(
        config.GOLD_DIR, layer="gold", step="export-base-roads",
        artifacts=["krakow_roads_base.parquet", "krakow_roads_base.csv"],
        configuration_fingerprint=run_fingerprint,
    )

    profile_paths = {
        "season": (config.GOLD_SEASON_PARQUET_PATH, config.GOLD_SEASON_CSV_PATH),
        "month": (config.GOLD_MONTH_PARQUET_PATH, config.GOLD_MONTH_CSV_PATH),
        "weekday": (config.GOLD_WEEK_PARQUET_PATH, config.GOLD_WEEK_CSV_PATH),
        "time_of_day": (config.GOLD_TOD_PARQUET_PATH, config.GOLD_TOD_CSV_PATH),
    }
    profile_sort_keys = {
        "season": common + ["season"],
        "month": common + ["season", "month", "month_name"],
        "weekday": common + ["day_of_week", "day_name", "is_weekend"],
        "time_of_day": common + ["time_of_day"],
    }
    pandas_tables: dict[str, pd.DataFrame] = {}
    for name, frame in tables.items():
        parquet_path, csv_path = profile_paths[name]
        pandas_tables[name] = _write_spark_then_single_parquet(
            frame, parquet_path, profile_sort_keys[name]
        )
        pandas_tables[name].to_csv(csv_path, index=False)
        write_step_completion_manifest(
            config.GOLD_DIR, layer="gold", step=f"export-profile-{name}",
            artifacts=[str(parquet_path.relative_to(config.GOLD_DIR)), str(csv_path.relative_to(config.GOLD_DIR))],
            configuration_fingerprint=run_fingerprint,
        )

    # Profile QA uses the already-materialized portable tables. This preserves
    # the reconciliation values without executing every expensive aggregation
    # two additional times before export.
    silver = spark.read.parquet(str(config.SILVER_DIR))
    _write_gold_qa(spark, silver, observations, pandas_tables)

    flat_spatial = flat.join(
        base_roads.select(
            "road_feature_id", "osm_id", "name", "highway", "oneway", "geometry"
        ),
        "road_feature_id",
        "left",
    )
    flat_sort_keys = flat_dimensions.copy()
    flat_sort_keys[flat_sort_keys.index("day_of_week_iso")] = "day_of_week"
    flat_gdf = _write_geoparquet(
        flat_spatial, config.GOLD_FLAT_GEOPARQUET_PATH, flat_sort_keys
    )
    flat_gdf.drop(columns=["geometry"]).to_csv(config.GOLD_FLAT_CSV_PATH, index=False)
    write_step_completion_manifest(
        config.GOLD_DIR, layer="gold", step="export-flat-profile",
        artifacts=["krakow_ambulance_speeds_2021_2023_flat.parquet", "krakow_ambulance_speeds_2021_2023_flat.csv"],
        configuration_fingerprint=run_fingerprint,
    )

    # One relational GeoPackage: base geometry plus four indexed attribute tables.
    if config.GOLD_GEOPACKAGE_PATH.exists():
        config.GOLD_GEOPACKAGE_PATH.unlink()
    base_gdf.to_file(
        config.GOLD_GEOPACKAGE_PATH,
        layer="krakow_roads_base",
        driver="GPKG",
        engine="pyogrio",
    )
    connection = sqlite3.connect(config.GOLD_GEOPACKAGE_PATH)
    try:
        sqlite_names = {
            "season": "speeds_by_season",
            "month": "speeds_by_month",
            "weekday": "speeds_by_day_of_week",
            "time_of_day": "speeds_by_time_of_day",
        }
        for name, table in pandas_tables.items():
            sql_name = sqlite_names[name]
            table.to_sql(sql_name, connection, if_exists="replace", index=False)
            connection.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{sql_name}_road_feature_id" '
                f'ON "{sql_name}"("road_feature_id")'
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"GeoPackage integrity check failed: {integrity}")
    finally:
        connection.close()
    write_step_completion_manifest(
        config.GOLD_DIR, layer="gold", step="export-relational-geopackage",
        artifacts=["krakow_ambulance_speeds_2021_2023.gpkg"],
        configuration_fingerprint=run_fingerprint,
    )

    if config.GOLD_FLAT_GEOPACKAGE_PATH.exists():
        config.GOLD_FLAT_GEOPACKAGE_PATH.unlink()
    flat_gdf.to_file(
        config.GOLD_FLAT_GEOPACKAGE_PATH,
        layer="krakow_ambulance_speed_observed_combinations",
        driver="GPKG",
        engine="pyogrio",
    )
    write_step_completion_manifest(
        config.GOLD_DIR, layer="gold", step="export-flat-geopackage",
        artifacts=["krakow_ambulance_speeds_2021_2023_flat.gpkg"],
        configuration_fingerprint=run_fingerprint,
    )

    observations.unpersist()
    base_roads.unpersist()
    write_layer_completion_marker(
        config.GOLD_DIR,
        {
            "layer": "gold",
            "profile_rows": {name: len(table) for name, table in pandas_tables.items()},
            "dataset_version": config.DATASET_VERSION,
            "artifacts": list(PipelineConfig.gold_relative_files().values()),
            "configuration_fingerprint": configuration_fingerprint(
                config.scientific_parameters()
            ),
        },
    )
    logger.info("Gold aggregation completed with neutral support and provenance fields")
    return config.GOLD_DIR

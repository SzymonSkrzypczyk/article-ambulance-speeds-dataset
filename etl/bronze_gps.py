"""Bronze ingestion with explicit timestamp provenance and attrition reporting."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pyspark.sql import DataFrame, Observation, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from config import config
from pipeline_utils import (
    configuration_fingerprint,
    layer_is_complete,
    utc_now_iso,
    write_json_atomic,
    write_layer_completion_marker,
    write_step_completion_manifest,
)

logger = logging.getLogger(__name__)

BATCH_COMPLETION_MARKER = "_KARETKI_BATCH_COMPLETE.json"


def _valid_bronze_checkpoints(candidate_root: Path) -> dict[str, tuple[Path, dict]]:
    """Return valid batch checkpoints, discarding only incomplete artifacts."""
    valid_checkpoints: dict[str, tuple[Path, dict]] = {}
    if not candidate_root.is_dir():
        return valid_checkpoints
    # ``candidate_root`` also stores restart metadata in hidden directories
    # (for example, ``.karetki-step-manifests``). Only batch directories can
    # be resumed or read as Parquet.
    for entry in sorted(candidate_root.glob("batch_*")):
        if not entry.is_dir():
            continue
        marker_file = entry / BATCH_COMPLETION_MARKER
        payload = None
        if marker_file.is_file():
            try:
                payload = json.loads(marker_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        batch_fingerprint = (
            payload.get("batch_fingerprint") if isinstance(payload, dict) else None
        )
        if not batch_fingerprint or not isinstance(metrics, dict):
            logger.warning("Discarding incomplete Bronze batch checkpoint %s", entry.name)
            shutil.rmtree(entry)
            continue
        if batch_fingerprint in valid_checkpoints:
            logger.warning("Discarding duplicate Bronze batch checkpoint %s", entry.name)
            shutil.rmtree(entry)
            continue
        valid_checkpoints[batch_fingerprint] = (entry, payload)
    return valid_checkpoints


def _candidate_only_resume_is_safe(
    candidate_root: Path, publish_root: Path, valid_checkpoints: dict[str, tuple[Path, dict]]
) -> bool:
    """Whether completed candidates can be finalized without raw CSV inputs.

    A publish marker proves that an earlier run completed every input batch and
    reached finalization. Candidate manifests must also agree with the current
    scientific configuration. This prevents a partial ingestion or a changed
    configuration from being mistaken for a complete, reusable candidate set.
    """
    if not valid_checkpoints or not any(publish_root.glob("*.json")):
        return False
    manifests = candidate_root / ".karetki-step-manifests"
    fingerprints = set()
    for path in manifests.glob("candidate-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        fingerprint = payload.get("configuration_fingerprint")
        if not isinstance(fingerprint, str):
            return False
        fingerprints.add(fingerprint)
    return fingerprints == {configuration_fingerprint(config.scientific_parameters())}


def _remove_finalized_candidate_chunks(chunks: list[Path]) -> None:
    """Release only candidates covered by a durable Bronze quarter marker."""
    for chunk in chunks:
        shutil.rmtree(chunk, ignore_errors=True)


def _safe_archive_target(root: Path, member_name: str) -> Path:
    """Resolve a ZIP member and reject path traversal outside ``root``."""
    root = root.resolve()
    target = (root / member_name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe ZIP member path: {member_name!r}") from exc
    return target


def extract_zips_if_needed(zip_dir: Path, target_csv_dir: Path) -> None:
    """Extract non-empty CSV members safely and fail on corrupt archives."""
    if target_csv_dir.exists() and any(target_csv_dir.rglob("*.csv")):
        logger.info(
            "Raw CSV files already exist in %s; skipping extraction", target_csv_dir
        )
        return
    if not zip_dir.exists():
        logger.info(
            "ZIP directory %s does not exist; direct CSV input will be used", zip_dir
        )
        return

    archives = sorted(
        path
        for path in zip_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".zip"
        and not path.name.startswith("._")
    )
    target_csv_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        logger.info("Extracting CSV members from %s", archive)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.is_dir() or Path(member.filename).suffix.lower() != ".csv":
                    continue
                if member.file_size <= 50:
                    logger.warning("Skipping empty/tiny CSV member %s", member.filename)
                    continue
                destination = _safe_archive_target(target_csv_dir, member.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (
                    zip_ref.open(member, "r") as source,
                    destination.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target)


def get_raw_csv_schema() -> StructType:
    """Return the source schema, including a corrupt-record capture column."""
    return StructType(
        [
            StructField("GPS", LongType(), True),
            StructField("GPS_HEIGHT", IntegerType(), True),
            StructField("GPS_LAT", DoubleType(), True),
            StructField("GPS_LON", DoubleType(), True),
            StructField("GPS_SPEED", DoubleType(), True),
            StructField("GPS_TIME", StringType(), True),
            StructField("DIN", IntegerType(), True),
            StructField("ZAPLON", IntegerType(), True),
            StructField("SYGNALIZACJA_SWIETLNA", IntegerType(), True),
            StructField("SYGNALIZACJA_DZWIEKOWA", IntegerType(), True),
            StructField("ETL_CZAS", StringType(), True),
            StructField("_corrupt_record", StringType(), True),
        ]
    )


def _timestamp_columns(df: DataFrame) -> DataFrame:
    """Parse UTC and local source timestamps while retaining their provenance."""
    if config.SPARK_SESSION_TIMEZONE != "UTC":
        raise ValueError("Bronze timestamp parsing requires a UTC Spark session")

    gps_utc = F.to_timestamp("GPS_TIME")
    etl_local_naive = F.to_timestamp("ETL_CZAS")
    etl_utc = F.to_utc_timestamp(etl_local_naive, config.TIMEZONE_WARSAW)

    if config.PRIMARY_TIMESTAMP_SOURCE == "GPS_TIME_UTC":
        chosen = F.coalesce(gps_utc, etl_utc)
        source = (
            F.when(gps_utc.isNotNull(), F.lit("GPS_TIME_UTC"))
            .when(etl_utc.isNotNull(), F.lit("ETL_CZAS_LOCAL_FALLBACK"))
            .otherwise(F.lit("MISSING_OR_INVALID"))
        )
    elif config.PRIMARY_TIMESTAMP_SOURCE == "ETL_CZAS_LOCAL":
        chosen = F.coalesce(etl_utc, gps_utc)
        source = (
            F.when(etl_utc.isNotNull(), F.lit("ETL_CZAS_LOCAL"))
            .when(gps_utc.isNotNull(), F.lit("GPS_TIME_UTC_FALLBACK"))
            .otherwise(F.lit("MISSING_OR_INVALID"))
        )
    else:
        raise ValueError(
            f"Unsupported PRIMARY_TIMESTAMP_SOURCE={config.PRIMARY_TIMESTAMP_SOURCE!r}"
        )

    return (
        df.withColumn("timestamp_gps_utc", gps_utc)
        .withColumn("timestamp_etl_utc", etl_utc)
        .withColumn("timestamp_utc", chosen)
        .withColumn("timestamp_source", source)
        .withColumn(
            "timestamp_disagreement_sec",
            F.when(
                gps_utc.isNotNull() & etl_utc.isNotNull(),
                F.abs(gps_utc.cast("long") - etl_utc.cast("long")),
            ).otherwise(F.lit(None).cast("long")),
        )
        .withColumn(
            "timestamp_sources_disagree",
            F.col("timestamp_disagreement_sec")
            > config.MAX_TIMESTAMP_DISAGREEMENT_SECONDS,
        )
        .withColumn(
            "timestamp_warsaw",
            F.from_utc_timestamp(F.col("timestamp_utc"), config.TIMEZONE_WARSAW),
        )
    )


def _signal_mode() -> F.Column:
    lights = F.col("SYGNALIZACJA_SWIETLNA")
    sound = F.col("SYGNALIZACJA_DZWIEKOWA")
    return (
        F.when(lights.isNull() | sound.isNull(), F.lit("SIGNAL_MISSING"))
        .when(~lights.isin(0, 1) | ~sound.isin(0, 1), F.lit("SIGNAL_INVALID"))
        .when((lights == 1) & (sound == 1), F.lit("LIGHT_ON_SOUND_ON"))
        .when((lights == 1) & (sound == 0), F.lit("LIGHT_ON_SOUND_OFF"))
        .when((lights == 0) & (sound == 1), F.lit("LIGHT_OFF_SOUND_ON"))
        .otherwise(F.lit("LIGHT_OFF_SOUND_OFF"))
    )


def _write_bronze_qa(totals: dict[str, int], df_bronze: DataFrame) -> dict[str, int]:
    """Write reconciled attrition/temporal coverage; return the merged totals."""
    totals = dict(totals)
    bronze_summary = (
        df_bronze.agg(
            F.count("*").alias("deduplicated_valid_rows"),
            F.min(F.col("timestamp_utc").cast("long")).alias(
                "min_timestamp_utc_epoch_seconds"
            ),
            F.max(F.col("timestamp_utc").cast("long")).alias(
                "max_timestamp_utc_epoch_seconds"
            ),
        )
        .first()
        .asDict()
    )
    totals.update(bronze_summary)

    total_rows = [
        ("ATTRITION", None, None, metric, int(value or 0))
        for metric, value in totals.items()
    ]
    coverage_rows = [
        (
            "TEMPORAL_COVERAGE",
            int(row["year"]),
            int(row["month"]),
            str(row["timestamp_source"]),
            int(row["count"]),
        )
        for row in df_bronze.groupBy("year", "month", "timestamp_source")
        .count()
        .collect()
    ]
    qa = df_bronze.sparkSession.createDataFrame(
        total_rows + coverage_rows,
        "report_type string, year int, month int, category string, count long",
    )
    qa.coalesce(1).write.mode("overwrite").parquet(str(config.BRONZE_QA_PATH))
    return totals


def _csv_input_files(path: Path) -> list[Path]:
    """Return deterministic non-metadata CSV inputs below a file or directory."""
    path = path.resolve()
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() == ".csv"
        and not candidate.name.startswith("._")
    )


def _batch_files_by_size(
    files: list[Path], target_bytes: int = 8 * 1024**3
) -> list[list[Path]]:
    """Group deterministic inputs into bounded sequential batches."""
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in files:
        size = path.stat().st_size
        if current and current_bytes + size > target_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def _flag_raw_rows(df_raw: DataFrame) -> DataFrame:
    """Add timestamp and validation flags used by Bronze and attrition QA.

    ``physically_valid_coordinate`` keeps its historical meaning so retained
    rows are unchanged; ``coordinate_reject_reason`` additionally splits the
    old single ``invalid_coordinate_rows`` bucket into actionable reasons
    (VAL-003): NULL_FIELD, ZERO_COORDINATE, OUT_OF_WORLD_RANGE (which also
    catches projected metre-scale values), and SWAPPED_LAT_LON_SUSPECT for
    coordinates that fall inside the transposed study bounding box.
    """
    west, south, east, north = config.KRAKOW_BBOX
    road_west, road_south, road_east, road_north = config.ROAD_EXTRACTION_BBOX
    return (
        _timestamp_columns(df_raw)
        .withColumn(
            "valid_identifier", F.col(config.SOURCE_ENTITY_ID_COLUMN).isNotNull()
        )
        .withColumn(
            "physically_valid_coordinate",
            F.col("GPS_LAT").isNotNull()
            & F.col("GPS_LON").isNotNull()
            & F.col("GPS_LAT").between(-90.0, 90.0)
            & F.col("GPS_LON").between(-180.0, 180.0)
            & (F.col("GPS_LAT") != 0.0)
            & (F.col("GPS_LON") != 0.0),
        )
        .withColumn(
            "coordinate_reject_reason",
            F.when(F.col("physically_valid_coordinate"), F.lit(None).cast("string"))
            .when(
                F.col("GPS_LAT").isNull() | F.col("GPS_LON").isNull(),
                F.lit("NULL_FIELD"),
            )
            .when(
                (F.col("GPS_LAT") == 0.0) | (F.col("GPS_LON") == 0.0),
                F.lit("ZERO_COORDINATE"),
            )
            .when(
                ~F.col("GPS_LAT").between(-90.0, 90.0)
                | ~F.col("GPS_LON").between(-180.0, 180.0),
                F.lit("OUT_OF_WORLD_RANGE"),
            )
            .when(
                F.col("GPS_LAT").between(west, east)
                & F.col("GPS_LON").between(south, north),
                F.lit("SWAPPED_LAT_LON_SUSPECT"),
            )
            .otherwise(F.lit("UNCLASSIFIED")),
        )
        .withColumn(
            "inside_study_aoi",
            F.col("physically_valid_coordinate")
            & F.col("GPS_LAT").between(south, north)
            & F.col("GPS_LON").between(west, east),
        )
        .withColumn(
            "inside_road_extraction_bbox",
            F.col("physically_valid_coordinate")
            & F.col("GPS_LAT").between(road_south, road_north)
            & F.col("GPS_LON").between(road_west, road_east),
        )
        .withColumn("date_warsaw", F.to_date("timestamp_warsaw"))
        .withColumn(
            "within_expected_period",
            F.col("date_warsaw").between(
                F.to_date(F.lit(config.EXPECTED_START_DATE)),
                F.to_date(F.lit(config.EXPECTED_END_DATE)),
            ),
        )
    )


def _bronze_candidates(df_flagged: DataFrame) -> DataFrame:
    """Create valid provenance-rich Bronze candidates before global deduplication."""
    hash_columns = [
        "GPS",
        "GPS_LAT",
        "GPS_LON",
        "GPS_SPEED",
        "GPS_TIME",
        "ETL_CZAS",
        "SYGNALIZACJA_SWIETLNA",
        "SYGNALIZACJA_DZWIEKOWA",
    ]
    hash_expression = F.sha2(
        F.concat_ws(
            "\u001f",
            *[
                F.coalesce(F.col(name).cast("string"), F.lit("<NULL>"))
                for name in hash_columns
            ],
        ),
        256,
    )
    return (
        df_flagged.filter(
            F.col("_corrupt_record").isNull()
            & F.col("valid_identifier")
            & F.col("physically_valid_coordinate")
            & F.col("timestamp_utc").isNotNull()
            & F.col("within_expected_period")
        )
        .withColumn(
            "source_entity_id", F.col(config.SOURCE_ENTITY_ID_COLUMN).cast("string")
        )
        .withColumn(
            "source_entity_id_semantic", F.lit(config.SOURCE_ENTITY_ID_SEMANTIC)
        )
        .withColumn("signal_mode", _signal_mode())
        .withColumn("point_id", hash_expression)
        .withColumn("year", F.year("timestamp_warsaw"))
        .withColumn("month", F.month("timestamp_warsaw"))
        .withColumn("quarter", F.concat(F.lit("Q"), F.quarter("timestamp_warsaw")))
        .withColumn("day", F.dayofmonth("timestamp_warsaw"))
        .withColumn("hour", F.hour("timestamp_warsaw"))
        .withColumn("spark_day_of_week", F.dayofweek("timestamp_warsaw"))
        .withColumn(
            "day_of_week_iso",
            F.when(F.col("spark_day_of_week") == 1, 7).otherwise(
                F.col("spark_day_of_week") - 1
            ),
        )
        .drop("spark_day_of_week")
        .withColumn("is_weekend", F.col("day_of_week_iso").isin(6, 7))
        .withColumn(
            "season",
            F.when(F.col("month").isin(12, 1, 2), "WINTER")
            .when(F.col("month").isin(3, 4, 5), "SPRING")
            .when(F.col("month").isin(6, 7, 8), "SUMMER")
            .otherwise("AUTUMN"),
        )
        .withColumn(
            "geometry",
            F.expr(
                "ST_Point(CAST(GPS_LON AS Decimal(24,20)), CAST(GPS_LAT AS Decimal(24,20)))"
            ),
        )
    )


def _batch_fingerprint(batch: list[Path]) -> str:
    """Digest the exact file paths and sizes that define one input batch."""
    digest = hashlib.sha256()
    for path in batch:
        digest.update(f"{path.resolve()}\0{path.stat().st_size}\0".encode("utf-8"))
    return digest.hexdigest()


def _deduplicate_candidates(candidates: DataFrame) -> DataFrame:
    """Select one deterministic source row for each content-derived point ID.

    Hash aggregation replaces the former window ranking: map-side partial
    aggregation shrinks the deduplication shuffle and avoids a global sort,
    which previously drove multi-hour swap thrash on memory-constrained
    machines. The winner is the lexicographically smallest full row ordered by
    ``source_file`` first, preserving the historical earliest-file preference
    while making same-file ties reproducible instead of arbitrary.
    ``geometry`` is a pure function of GPS_LAT/GPS_LON but a non-orderable UDT
    column, so it is excluded from comparison and recomputed for the winner.
    """
    ordered = ["source_file"] + [
        name for name in candidates.columns if name not in {"source_file", "geometry"}
    ]
    winner = F.min(F.struct(*[F.col(name) for name in ordered])).alias("_winner")
    selected = candidates.groupBy("point_id").agg(winner)
    projected = []
    for name in candidates.columns:
        if name == "point_id":
            projected.append(F.col("point_id"))
        elif name != "geometry":
            projected.append(F.col("_winner")[name].alias(name))
    # geometry is re-added last, matching its position in _bronze_candidates.
    return selected.select(*projected).withColumn(
        "geometry",
        F.expr(
            "ST_Point(CAST(GPS_LON AS Decimal(24,20)), CAST(GPS_LAT AS Decimal(24,20)))"
        ),
    )


def process_bronze_gps_logs(
    spark: SparkSession,
    input_path: Optional[Path] = None,
    force_rebuild: bool = False,
    batch_target_bytes: int = 4 * 1024**3,
    work_root_parent: Optional[Path] = None,
) -> Path:
    """Ingest source CSVs into a deduplicated, provenance-rich Bronze layer.

    Ingestion is checkpointed under a stable work root (default: a sibling of
    the Bronze output directory; pass ``work_root_parent`` to anchor it
    elsewhere, e.g. the real output root when Bronze itself is staged). Each
    completed batch persists its candidate Parquet files plus a completion
    marker carrying the batch's observed attrition metrics, and each published
    year/quarter partition records its own marker. A rerun after a crash
    verifies a run fingerprint over the scientific parameters and the full
    input manifest, skips already completed batches, and republishes partitions
    idempotently via dynamic partition overwrite. After a quarter is written,
    read back, and marked complete, its candidates are reclaimed; a later retry
    can finalize remaining quarters without extracted raw CSV files. Do not run
    two ingestions concurrently against one work root.
    """
    config.ensure_directories()
    output_dir = config.BRONZE_DIR
    if (
        output_dir.exists()
        and any(output_dir.rglob("*.parquet"))
        and layer_is_complete(
            output_dir,
            layer="bronze",
            configuration_fingerprint=config.configuration_fingerprint(),
        )
        and not force_rebuild
    ):
        logger.info(
            "Bronze dataset already exists at %s; skipping ingestion", output_dir
        )
        return output_dir
    if output_dir.exists() and not layer_is_complete(
        output_dir,
        layer="bronze",
        configuration_fingerprint=config.configuration_fingerprint(),
    ):
        logger.warning(
            "Bronze directory %s lacks a completion marker; treating it as an "
            "incomplete previous run and rebuilding it",
            output_dir,
        )

    # A deterministic work root lets a later invocation resume this run's
    # checkpoints; mkdtemp-style unique roots would orphan them instead.
    work_root = Path(work_root_parent or output_dir.parent) / ".bronze-work"
    work_root.mkdir(parents=True, exist_ok=True)
    candidate_root = work_root / "candidates"
    publish_root = work_root / "publish-markers"
    fingerprint_path = work_root / "run-fingerprint"
    valid_checkpoints = _valid_bronze_checkpoints(candidate_root)

    if input_path is not None:
        input_root = Path(input_path).resolve()
    elif any(config.RAW_CSV_DIR.rglob("*.csv")):
        input_root = config.RAW_CSV_DIR
    elif config.INPUT_DATA_DIR.exists() and any(config.INPUT_DATA_DIR.rglob("*.csv")):
        input_root = config.INPUT_DATA_DIR
    else:
        input_root = None

    candidate_only_resume = input_root is None and _candidate_only_resume_is_safe(
        candidate_root, publish_root, valid_checkpoints
    )
    batches = []
    if candidate_only_resume:
        logger.info(
            "Raw CSV files are absent; resuming Bronze finalization from %d "
            "validated checkpointed batch(es)",
            len(valid_checkpoints),
        )
    else:
        extract_zips_if_needed(config.INPUT_ZIP_DIR, config.RAW_CSV_DIR)
        if input_path is not None:
            input_root = Path(input_path).resolve()
        elif any(config.RAW_CSV_DIR.rglob("*.csv")):
            input_root = config.RAW_CSV_DIR
        elif config.INPUT_DATA_DIR.exists() and any(config.INPUT_DATA_DIR.rglob("*.csv")):
            input_root = config.INPUT_DATA_DIR
        else:
            raise FileNotFoundError(
                f"No input CSV files found in {config.RAW_CSV_DIR} or {config.INPUT_DATA_DIR}; "
                "no complete Bronze checkpoint set is available for a safe resume"
            )
        input_files = _csv_input_files(input_root)
        if not input_files:
            raise FileNotFoundError(f"No CSV input files found below {input_root}")
        batches = _batch_files_by_size(input_files, target_bytes=batch_target_bytes)
        logger.info(
            "Reading %d raw GPS CSV files from %s in %d bounded batches",
            len(input_files),
            input_root,
            len(batches),
        )
        run_inputs = "".join(_batch_fingerprint(batch) for batch in batches)
        run_fingerprint = hashlib.sha256(
            (
                configuration_fingerprint(config.scientific_parameters()) + run_inputs
            ).encode("utf-8")
        ).hexdigest()

    if not candidate_only_resume:
        stored_fingerprint = None
        if fingerprint_path.is_file():
            try:
                stored_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip()
            except OSError:
                stored_fingerprint = None
        has_stale_candidates = candidate_root.is_dir() and any(candidate_root.iterdir())
        if stored_fingerprint != run_fingerprint and has_stale_candidates:
            logger.warning(
                "Inputs or scientific parameters changed since the interrupted "
                "Bronze run; discarding stale checkpoints in %s",
                candidate_root,
            )
            shutil.rmtree(candidate_root, ignore_errors=True)
            shutil.rmtree(publish_root, ignore_errors=True)
            valid_checkpoints = {}
        fingerprint_path.write_text(run_fingerprint + "\n", encoding="utf-8")
    resumed_run = bool(valid_checkpoints)
    if resumed_run:
        logger.info(
            "Resuming Bronze ingestion: %d checkpointed batch(es) found",
            len(valid_checkpoints),
        )

    metric_names = [
        "raw_rows",
        "corrupt_rows",
        "missing_identifier_rows",
        "invalid_coordinate_rows",
        "invalid_coord_null_field_rows",
        "invalid_coord_zero_rows",
        "invalid_coord_out_of_world_rows",
        "invalid_coord_swapped_suspect_rows",
        "outside_study_aoi_rows",
        "inside_study_aoi_rows",
        "outside_road_extraction_bbox_rows",
        "inside_road_extraction_bbox_rows",
        "invalid_timestamp_rows",
        "outside_period_rows",
        "timestamp_disagreement_rows",
    ]
    totals = {name: 0 for name in metric_names}
    if candidate_only_resume:
        for _, payload in valid_checkpoints.values():
            for name in metric_names:
                totals[name] += int(payload["metrics"].get(name) or 0)
    success = False
    try:
        for index, batch in enumerate(batches, start=1):
            batch_bytes = sum(path.stat().st_size for path in batch)
            batch_fingerprint = _batch_fingerprint(batch)
            checkpoint = valid_checkpoints.get(batch_fingerprint)
            if checkpoint is not None:
                _, payload = checkpoint
                for name in metric_names:
                    totals[name] += int(payload["metrics"].get(name) or 0)
                logger.info(
                    "Skipping Bronze input batch %d/%d (%d files, %.2f GiB); "
                    "checkpoint %s already completed",
                    index,
                    len(batches),
                    len(batch),
                    batch_bytes / 1024**3,
                    batch_fingerprint[:12],
                )
                continue
            logger.info(
                "Processing Bronze input batch %d/%d: %d files, %.2f GiB",
                index,
                len(batches),
                len(batch),
                batch_bytes / 1024**3,
            )
            df_raw = (
                spark.read.option("header", "true")
                .option("mode", "PERMISSIVE")
                .option("columnNameOfCorruptRecord", "_corrupt_record")
                .schema(get_raw_csv_schema())
                .csv([str(path) for path in batch])
                .withColumn("source_file", F.input_file_name())
            )
            observation = Observation(f"bronze_batch_{index}")
            observed = _flag_raw_rows(df_raw).observe(
                observation,
                F.count("*").alias("raw_rows"),
                F.sum(F.col("_corrupt_record").isNotNull().cast("long")).alias(
                    "corrupt_rows"
                ),
                F.sum((~F.col("valid_identifier")).cast("long")).alias(
                    "missing_identifier_rows"
                ),
                F.sum((~F.col("physically_valid_coordinate")).cast("long")).alias(
                    "invalid_coordinate_rows"
                ),
                F.sum(
                    (F.col("coordinate_reject_reason") == "NULL_FIELD").cast("long")
                ).alias("invalid_coord_null_field_rows"),
                F.sum(
                    (F.col("coordinate_reject_reason") == "ZERO_COORDINATE").cast(
                        "long"
                    )
                ).alias("invalid_coord_zero_rows"),
                F.sum(
                    (F.col("coordinate_reject_reason") == "OUT_OF_WORLD_RANGE").cast(
                        "long"
                    )
                ).alias("invalid_coord_out_of_world_rows"),
                F.sum(
                    (F.col("coordinate_reject_reason") == "SWAPPED_LAT_LON_SUSPECT")
                    .cast("long")
                ).alias("invalid_coord_swapped_suspect_rows"),
                F.sum(
                    (
                        F.col("physically_valid_coordinate")
                        & ~F.col("inside_study_aoi")
                    ).cast("long")
                ).alias("outside_study_aoi_rows"),
                F.sum(F.col("inside_study_aoi").cast("long")).alias(
                    "inside_study_aoi_rows"
                ),
                F.sum(
                    (
                        F.col("physically_valid_coordinate")
                        & ~F.col("inside_road_extraction_bbox")
                    ).cast("long")
                ).alias("outside_road_extraction_bbox_rows"),
                F.sum(F.col("inside_road_extraction_bbox").cast("long")).alias(
                    "inside_road_extraction_bbox_rows"
                ),
                F.sum(F.col("timestamp_utc").isNull().cast("long")).alias(
                    "invalid_timestamp_rows"
                ),
                F.sum((~F.col("within_expected_period")).cast("long")).alias(
                    "outside_period_rows"
                ),
                F.sum(F.col("timestamp_sources_disagree").cast("long")).alias(
                    "timestamp_disagreement_rows"
                ),
            )
            batch_dir = candidate_root / f"batch_{index:06d}_{uuid4().hex[:8]}"
            (
                _deduplicate_candidates(_bronze_candidates(observed))
                .write.mode("append")
                .partitionBy("year", "quarter")
                .parquet(str(batch_dir))
            )
            observed_metrics = observation.get
            batch_totals = {
                name: int(observed_metrics.get(name) or 0) for name in metric_names
            }
            for name in metric_names:
                totals[name] += batch_totals[name]
            write_json_atomic(
                batch_dir / BATCH_COMPLETION_MARKER,
                {
                    "batch_fingerprint": batch_fingerprint,
                    "batch_index": index,
                    "completed_at_utc": utc_now_iso(),
                    "metrics": batch_totals,
                },
            )
            write_step_completion_manifest(
                candidate_root,
                layer="bronze",
                step=f"candidate-{index:06d}-{batch_fingerprint[:12]}",
                artifacts=[batch_dir.relative_to(candidate_root)],
                configuration_fingerprint=configuration_fingerprint(
                    config.scientific_parameters()
                ),
                details={"batch_fingerprint": batch_fingerprint, "metrics": batch_totals},
            )

        quarter_groups: dict[tuple[int, str], list[Path]] = {}
        for chunk in sorted(candidate_root.glob("batch_*/year=*/quarter=*")):
            key = (
                int(chunk.parent.name.split("=", 1)[1]),
                chunk.name.split("=", 1)[1],
            )
            quarter_groups.setdefault(key, []).append(chunk)
        if not quarter_groups:
            raise ValueError("No valid Bronze candidates remained after validation")
        if spark.conf.get("spark.sql.sources.partitionOverwriteMode", "") != "dynamic":
            raise ValueError(
                "Bronze finalization requires "
                "spark.sql.sources.partitionOverwriteMode=dynamic so that "
                "quarter republishes stay idempotent across resumed runs"
            )
        if not resumed_run and output_dir.exists():
            shutil.rmtree(output_dir)
        logger.info(
            "Finalizing %d deduplicated Bronze quarter partitions", len(quarter_groups)
        )
        publish_root.mkdir(parents=True, exist_ok=True)
        ordered_quarters = sorted(quarter_groups.items())
        for index, ((year_value, quarter_value), chunks) in enumerate(ordered_quarters, start=1):
            publish_marker = publish_root / f"{year_value}_{quarter_value}.json"
            partition_dir = (
                output_dir / f"year={year_value}" / f"quarter={quarter_value}"
            )
            if publish_marker.is_file() and any(partition_dir.rglob("*.parquet")):
                logger.info(
                    "Finalizing Bronze partition %d/%d: %d/%s already published",
                    index,
                    len(quarter_groups),
                    year_value,
                    quarter_value,
                )
                if index < len(ordered_quarters):
                    _remove_finalized_candidate_chunks(chunks)
                continue
            logger.info(
                "Finalizing Bronze partition %d/%d: %d/%s from %d candidate chunk(s)",
                index,
                len(quarter_groups),
                year_value,
                quarter_value,
                len(chunks),
            )
            candidates = None
            for chunk in chunks:
                # Batch checkpoints are separate Parquet roots, so Spark cannot
                # infer one common partition base from a multi-path read. Add
                # the known partition values after each isolated scan.
                chunk_candidates = (
                    spark.read.parquet(str(chunk))
                    .withColumn("year", F.lit(year_value))
                    .withColumn("quarter", F.lit(quarter_value))
                )
                candidates = (
                    chunk_candidates
                    if candidates is None
                    else candidates.unionByName(chunk_candidates)
                )
            if candidates is None:
                raise ValueError(
                    f"No readable candidate chunks for {year_value}/{quarter_value}"
                )
            finalized = _deduplicate_candidates(candidates)
            (
                finalized
                .write.mode("overwrite")
                .partitionBy("year", "quarter")
                .parquet(str(output_dir))
            )
            written_rows = spark.read.parquet(str(partition_dir)).count()
            if written_rows == 0:
                raise ValueError(
                    f"Bronze partition {year_value}/{quarter_value} was empty after write"
                )
            write_json_atomic(
                publish_marker,
                {"published_at_utc": utc_now_iso(), "rows": written_rows},
            )
            if index < len(ordered_quarters):
                _remove_finalized_candidate_chunks(chunks)

        written_bronze = spark.read.parquet(str(output_dir))
        totals = _write_bronze_qa(totals, written_bronze)
        write_layer_completion_marker(
            output_dir,
            {
                "layer": "bronze",
                "rows": int(totals["deduplicated_valid_rows"]),
                "dataset_version": config.DATASET_VERSION,
                "artifacts": [
                    str(path.relative_to(output_dir))
                    for path in sorted(output_dir.glob("year=*/quarter=*"))
                ],
                "configuration_fingerprint": configuration_fingerprint(
                    config.scientific_parameters()
                ),
            },
        )
        shutil.rmtree(candidate_root, ignore_errors=True)
        success = True
    finally:
        if success:
            shutil.rmtree(work_root, ignore_errors=True)
        else:
            logger.warning(
                "Bronze work root retained at %s; rerunning the pipeline will "
                "resume from completed checkpoints instead of restarting.",
                work_root,
            )
    logger.info("Bronze ingestion completed")
    return output_dir

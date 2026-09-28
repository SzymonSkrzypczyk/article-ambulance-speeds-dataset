"""CLI Main Runner for Kraków Ambulance Speed Dataset Apache Sedona Pipeline."""

import argparse
import logging
import os
import platform
import signal
import sys
import tempfile
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from config import PipelineConfig, config
from etl.bronze_gps import process_bronze_gps_logs
from etl.gold_aggregation import process_gold_aggregation
from etl.osm_roads import process_osm_road_network
from etl.silver_matching import process_silver_map_matching
from etl.weather import process_weather_data
from pipeline_utils import (
    git_revision,
    layer_is_complete,
    read_layer_completion_marker,
    remove_staging_directory,
    replace_directory_from_staging,
    utc_now_iso,
    write_json_atomic,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def is_external_drive(path: Path) -> bool:
    """Determine if a path is on an external drive or non-system partition on macOS, Linux, and Windows."""
    resolved_path = str(path.resolve())
    current_os = platform.system()

    if current_os == "Windows":
        system_drive = os.getenv("SystemDrive", "C:")
        if resolved_path.startswith(("\\\\", "//")):
            return True
        path_drive = path.resolve().anchor
        return path_drive.strip("\\/:").upper() != system_drive.strip("\\/:").upper()

    elif current_os == "Darwin":  # macOS
        return resolved_path.startswith("/Volumes/")

    elif current_os == "Linux":
        return resolved_path.startswith(("/media/", "/mnt/"))

    return False


def bronze_resume_staging_directory(staging_parent: Path) -> Path:
    """Return the stable Bronze staging path used across interrupted runs.

    Bronze publishes year/quarter partitions incrementally. A fresh staging
    directory would discard completed partitions after a crash and force their
    expensive finalization to run again. This path is removed after a
    successful publish, but remains intact when a run fails.
    """
    return staging_parent / ".karetki-bronze-resume-staging"


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter to add ANSI colors for terminal logging."""

    GREY = "\x1b[38;20m"
    BLUE = "\x1b[36;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    GREEN = "\x1b[32;20m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
        logging.INFO: BLUE + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
        logging.WARNING: YELLOW + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED
        + "%(asctime)s - %(levelname)s - %(message)s"
        + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        msg = record.getMessage()
        if record.levelno == logging.INFO and any(
            word in msg.lower()
            for word in ["success", "finished", "completed", "done", "saved"]
        ):
            log_fmt = (
                self.GREEN + "%(asctime)s - %(levelname)s - %(message)s" + self.RESET
            )

        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def _configure_logging() -> None:
    """Install the console handler at run time, not at import time.

    Configuring the root logger during import hijacked logging for every
    process that merely imported this module (including skipped Spark test
    collection) and made logger behavior depend on import order.
    """
    handler = logging.StreamHandler(sys.stdout)
    if sys.stdout.isatty():
        handler.setFormatter(ColoredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger = logging.getLogger()
    for existing in root_logger.handlers[:]:
        root_logger.removeHandler(existing)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


logger = logging.getLogger("main_pipeline")


def init_sedona_spark_session(
    default_parallelism: int = 32,
    shuffle_partitions: int = 32,
    scratch_dir: Path | None = None,
    cores: int = 4,
    driver_memory: str = "6g",
    executor_memory: str = "6g",
) -> "SparkSession":
    """Create and configure Apache Sedona SparkSession."""
    import pyspark

    # Use the Spark distribution shipped with the pinned PySpark package. A
    # machine-level SPARK_HOME can otherwise substitute an incompatible Spark.
    bundled_spark_home = Path(pyspark.__file__).resolve().parent
    os.environ["SPARK_HOME"] = str(bundled_spark_home)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    try:
        from sedona.spark import SedonaContext
    except ImportError as exc:
        raise RuntimeError(
            "The pinned Apache Sedona environment is incomplete; recreate environment.yml"
        ) from exc

    logger.info("Initializing Apache Sedona Spark Session...")

    spark_local_dir = (scratch_dir or Path(tempfile.gettempdir())) / "spark"
    spark_local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using Spark scratch directory: %s", spark_local_dir)
    # Container images are read-only under Apptainer. The Slurm wrapper copies
    # the pre-resolved JAR cache into a writable bind mount before startup.
    spark_jars_ivy = os.environ.get("SPARK_JARS_IVY")

    builder = (
        SedonaContext.builder()
        .master(f"local[{cores}]")
        .appName("Krakow_Ambulance_Speed_Dataset_Sedona")
        .config(
            "spark.jars.packages",
            "org.apache.sedona:sedona-spark-3.4_2.12:1.5.1,org.datasyslab:geotools-wrapper:1.5.1-28.2",
        )
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.session.timeZone", config.SPARK_SESSION_TIMEZONE)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        # Sized for the 16 GiB reference laptop: a 6 GiB heap plus Sedona
        # native allocations, Python, and the filesystem cache fit in physical
        # RAM, preventing the swap-driven GC death spiral that previously froze
        # the driver for 9+ minutes until macOS killed it. ExitOnOutOfMemoryError
        # turns unrecoverable heap exhaustion into an immediate failure that
        # resumable Bronze checkpoints can retry, instead of hanging for hours.
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config(
            "spark.driver.extraJavaOptions",
            "-XX:+UseG1GC -XX:+ExitOnOutOfMemoryError",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-XX:+UseG1GC -XX:+ExitOnOutOfMemoryError",
        )
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.offHeap.enabled", "false")
        # Smaller input splits bound per-task CSV parse buffers on laptops.
        .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(default_parallelism))
        # macOS can report a short NIO transfer during a shuffle merge, causing
        # Spark's `copyFileStreamNIO` assertion to abort the whole stage. Use
        # the buffered stream path instead of FileChannel.transferTo.
        .config("spark.file.transferTo", "false")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config(
            "spark.local.dir",
            str(spark_local_dir),
        )
    )
    if spark_jars_ivy:
        builder = builder.config("spark.jars.ivy", spark_jars_ivy)
        logger.info("Using writable Spark Ivy cache: %s", spark_jars_ivy)

    spark = SedonaContext.create(builder.getOrCreate())
    logger.info("Sedona Spark Session initialized successfully.")
    return spark


def warn_on_configuration_drift(layer_dir: Path, label: str) -> None:
    """Loudly flag an existing layer built under a different configuration."""
    payload = read_layer_completion_marker(layer_dir)
    if not payload:
        return
    recorded = payload.get("configuration_fingerprint")
    current = config.configuration_fingerprint()
    if recorded and recorded != current:
        logger.warning(
            "%s at %s was built under a different scientific configuration "
            "(recorded fingerprint %s, current %s). It will be REUSED as-is; "
            "pass --force-rebuild to regenerate it under the current "
            "configuration.",
            label,
            layer_dir,
            recorded,
            current,
        )


def clean_output_directory(directory: Path) -> None:
    """Remove OS metadata while preserving Spark completion markers."""
    logger.info(
        "Cleaning up transient metadata files (._*, .DS_Store, _SUCCESS) in %s...",
        directory,
    )
    count = 0
    patterns = ["._*", ".DS_Store"]
    try:
        for pattern in patterns:
            for path in directory.rglob(pattern):
                if path.is_file():
                    try:
                        path.unlink()
                        count += 1
                    except Exception as e:
                        logger.debug("Failed to remove transient file %s: %s", path, e)
    except Exception as e:
        logger.debug("Error traversing directory for metadata cleanup: %s", e)
    if count > 0:
        logger.info("Removed %d transient metadata files from output directory.", count)


def write_run_manifest(
    stage: str, force_rebuild: bool, *, runtime_settings: dict[str, int] | None = None
) -> None:
    """Record exact code, configuration, and dependency state for the run."""
    packages = {}
    for package in (
        "pyspark",
        "apache-sedona",
        "pandas",
        "geopandas",
        "pyarrow",
        "shapely",
        "httpx",
        "osmnx",
        "pyogrio",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    write_json_atomic(
        config.RUN_MANIFEST_PATH,
        {
            "generated_at_utc": utc_now_iso(),
            "stage": stage,
            "force_rebuild": force_rebuild,
            "scientific_parameters": config.scientific_parameters(),
            "git": git_revision(config.BASE_DIR),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "execution_settings": runtime_settings or {},
            },
            "paths": {
                "data_root": str(config.DATA_DIR),
                "bronze": str(config.BRONZE_DIR),
                "silver": str(config.SILVER_DIR),
                "gold": str(config.GOLD_DIR),
                "osm_provenance": str(config.OSM_PROVENANCE_PATH),
                "weather_provenance": str(config.WEATHER_PROVENANCE_PATH),
            },
        },
    )


def main():
    """Main CLI entry point."""
    _configure_logging()
    # SIGTERM must trigger the same finally-block cleanup as Ctrl-C;
    # the default disposition kills the process with no Spark stop or
    # staging removal, leaking scratch directories.
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    parser = argparse.ArgumentParser(
        description="Idempotent Apache Sedona Pipeline for Kraków Ambulance Speed Dataset (2021-2023)."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["weather", "osm", "bronze", "silver", "gold", "all"],
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force re-fetching and overwriting existing intermediate datasets.",
    )

    parser.add_argument(
        "--zips-dir",
        type=str,
        default=None,
        help="Custom path to the directory containing ZIP archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom root directory for storing output layers (bronze, silver, gold, weather, osm).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help=(
            "Directory for Spark shuffle and spill files. Defaults to the system "
            "temporary directory."
        ),
    )
    parser.add_argument(
        "--bronze-staging-dir",
        type=str,
        default=None,
        help=(
            "Parent directory for temporary Bronze publication output. Use a "
            "directory on the output volume to keep the large Bronze staging copy "
            "off the Spark scratch disk. Spark shuffle and spill files still use "
            "--scratch-dir."
        ),
    )
    parser.add_argument(
        "--silver-staging-dir",
        type=str,
        default=None,
        help=(
            "Parent directory for temporary Silver publication output. Use a "
            "directory on the output volume to keep the Silver staging copy off "
            "the Spark scratch disk."
        ),
    )
    parser.add_argument(
        "--gold-staging-dir",
        type=str,
        default=None,
        help=(
            "Parent directory for temporary Gold publication output. Use a "
            "directory on the output volume to keep the Gold staging copy off "
            "the Spark scratch disk."
        ),
    )
    parser.add_argument(
        "--bronze-batch-gb",
        type=float,
        default=4.0,
        help=(
            "Target size of each Bronze input batch in GiB. Smaller batches lower "
            "peak memory on constrained machines; checkpoints make any batch "
            "count resumable. Note: changing it mid-run invalidates existing "
            "checkpoints because batches are re-grouped."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "Spark local worker count. Fewer workers lower concurrent task "
            "memory (use 2 on 16 GiB machines that still hit heap limits); "
            "results are identical regardless of thread count."
        ),
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=32,
        help=(
            "Spark shuffle partition count. Higher values reduce per-task spill "
            "size; use 400 for the low-local-disk trial."
        ),
    )
    parser.add_argument(
        "--spark-driver-memory",
        default="6g",
        help="Spark driver JVM heap (default: 6g; use 48g for the Athena layer job).",
    )
    parser.add_argument(
        "--spark-executor-memory",
        default="6g",
        help="Spark executor JVM heap (default: 6g; use 48g for the Athena layer job).",
    )
    parser.add_argument(
        "--silver-source-shards",
        type=int,
        default=1,
        help=(
            "Deterministic source-ID shards per Bronze quarter during Silver. "
            "Use 8 for the low-local-disk trial; 1 preserves current behavior."
        ),
    )
    parser.add_argument("--dataset-version", default=config.DATASET_VERSION)
    parser.add_argument(
        "--aoi-bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help=(
            "Override the study-area bounding box in WGS84. The same AOI is used "
            "for telemetry inclusion and OSM extraction. Choose it from a documented "
            "study-region definition and raw-coordinate coverage, not from the desired "
            "number of retained rows."
        ),
    )
    parser.add_argument(
        "--road-buffer-meters",
        type=float,
        default=None,
        help=(
            "Override the road-candidate search radius in metres. Increasing it can "
            "raise the match rate but also candidate ambiguity and must be validated."
        ),
    )
    parser.add_argument(
        "--road-extraction-bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help=(
            "Override the broader WGS84 road-processing envelope. This envelope "
            "must contain the official study AOI and is not an administrative boundary."
        ),
    )
    parser.add_argument(
        "--source-id-semantic",
        default=config.SOURCE_ENTITY_ID_SEMANTIC,
        help="Provider-verified meaning of the source ID; retain the default when unknown.",
    )
    parser.add_argument(
        "--speed-source",
        choices=["gps_speed", "displacement"],
        default=config.SPEED_SOURCE,
        help=(
            "Primary published speed: provider GPS_SPEED interpreted as cm/s "
            "(default), or consecutive-ping displacement."
        ),
    )
    parser.add_argument("--weather-model", default=config.WEATHER_MODEL)
    parser.add_argument(
        "--osm-source",
        choices=["osmnx_overpass", "local_pbf"],
        default=config.OSM_SOURCE,
    )
    parser.add_argument("--osm-pbf", default=None)

    args = parser.parse_args()

    if args.bronze_batch_gb <= 0:
        parser.error("--bronze-batch-gb must be positive")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.shuffle_partitions < 1:
        parser.error("--shuffle-partitions must be at least 1")
    if args.silver_source_shards < 1:
        parser.error("--silver-source-shards must be at least 1")

    bronze_staging_parent = (
        Path(args.bronze_staging_dir).resolve()
        if args.bronze_staging_dir
        else None
    )
    silver_staging_parent = (
        Path(args.silver_staging_dir).resolve()
        if args.silver_staging_dir
        else None
    )
    gold_staging_parent = (
        Path(args.gold_staging_dir).resolve() if args.gold_staging_dir else None
    )

    config.DATASET_VERSION = args.dataset_version
    config.SOURCE_ENTITY_ID_SEMANTIC = args.source_id_semantic
    config.SPEED_SOURCE = args.speed_source
    config.WEATHER_MODEL = args.weather_model
    config.OSM_SOURCE = args.osm_source
    config.OSM_PBF_PATH = Path(args.osm_pbf).resolve() if args.osm_pbf else None
    if args.aoi_bbox is not None:
        config.KRAKOW_BBOX = tuple(args.aoi_bbox)
        config.AOI_BOUNDARY_NAME = "User-supplied rectangular study AOI"
        config.AOI_BOUNDARY_SOURCE = "CLI override"
        config.AOI_BOUNDARY_SOURCE_URL = ""
        config.AOI_BOUNDARY_IDENTIFIER = None
        config.AOI_BOUNDARY_RETRIEVED_DATE = None
    if args.road_buffer_meters is not None:
        config.ROAD_BUFFER_METERS = args.road_buffer_meters
    if args.road_extraction_bbox is not None:
        config.ROAD_EXTRACTION_BBOX = tuple(args.road_extraction_bbox)

    # Override configurations dynamically if custom paths are specified
    if args.zips_dir:
        logger.info("Overriding INPUT_ZIP_DIR to: %s", args.zips_dir)
        object.__setattr__(config, "INPUT_ZIP_DIR", Path(args.zips_dir).resolve())

    if args.output_dir:
        out_path = Path(args.output_dir).resolve()
        logger.info("Overriding output data directory (DATA_DIR) to: %s", out_path)
        config.update_output_directory(out_path)

    for staging_parent, layer_name, layer_dir in (
        (bronze_staging_parent, "Bronze", config.BRONZE_DIR),
        (silver_staging_parent, "Silver", config.SILVER_DIR),
        (gold_staging_parent, "Gold", config.GOLD_DIR),
    ):
        if staging_parent is None:
            continue
        try:
            staging_parent.relative_to(layer_dir.resolve())
        except ValueError:
            pass
        else:
            parser.error(
                f"--{layer_name.lower()}-staging-dir must not be inside the final "
                f"{layer_name} directory ({layer_dir})"
            )

    config.validate()

    logger.info("Starting pipeline execution (stage=%s)...", args.stage)

    # 1. Stage 0: Weather
    if args.stage in {"weather", "all"}:
        logger.info("=== STAGE 0: Weather Ingestion (Open-Meteo) ===")
        process_weather_data(force_rebuild=args.force_rebuild)

    # 2. Stage 1: OSM Roads
    if args.stage in {"osm", "all"}:
        logger.info("=== STAGE 1: OSM Kraków Road Extraction ===")
        process_osm_road_network(force_rebuild=args.force_rebuild)

    # Stages requiring Spark / Sedona
    if args.stage in {"bronze", "silver", "gold", "all"}:
        scratch_root = (
            Path(args.scratch_dir).resolve()
            if args.scratch_dir
            else Path(tempfile.gettempdir()) / "karetki_pipeline_scratch"
        )
        scratch_root.mkdir(parents=True, exist_ok=True)
        run_scratch_root = Path(
            tempfile.mkdtemp(prefix="run-", dir=scratch_root)
        )
        spark = init_sedona_spark_session(
            scratch_dir=run_scratch_root,
            cores=args.threads,
            default_parallelism=args.shuffle_partitions,
            shuffle_partitions=args.shuffle_partitions,
            driver_memory=args.spark_driver_memory,
            executor_memory=args.spark_executor_memory,
        )
        staging_parent = run_scratch_root / "staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        run_staging_root = Path(tempfile.mkdtemp(prefix="run-", dir=staging_parent))
        configured_staging_parents = {
            "silver": silver_staging_parent,
            "gold": gold_staging_parent,
        }
        configured_staging_roots: dict[str, Path] = {}
        for layer_name, configured_parent in configured_staging_parents.items():
            if configured_parent is None:
                continue
            configured_parent.mkdir(parents=True, exist_ok=True)
            run_root = Path(
                tempfile.mkdtemp(
                    prefix=f"karetki-{layer_name}-staging-", dir=configured_parent
                )
            )
            configured_staging_roots[layer_name] = run_root
            logger.info(
                "Using configured %s staging directory: %s", layer_name, run_root
            )

        try:
            if args.stage in {"bronze", "all"}:
                logger.info("=== STAGE 2: Bronze GPS Log Ingestion ===")
                real_bronze_dir = config.BRONZE_DIR

                if (
                    not args.force_rebuild
                    and any(real_bronze_dir.rglob("*.parquet"))
                    and layer_is_complete(
                        real_bronze_dir,
                        layer="bronze",
                        configuration_fingerprint=config.configuration_fingerprint(),
                    )
                ):
                    warn_on_configuration_drift(real_bronze_dir, "Bronze")
                    logger.info(
                        "Bronze Parquet dataset already exists at %s. Skipping raw CSV ingestion.",
                        real_bronze_dir,
                    )
                else:
                    # Bronze finalization is resumable by year/quarter only when
                    # its staging directory survives a restarted process. Use a
                    # stable child of the requested staging parent (or scratch
                    # root for external outputs), rather than a per-run directory.
                    use_local_staging = (
                        bronze_staging_parent is not None
                        or is_external_drive(real_bronze_dir)
                    )
                    if use_local_staging:
                        bronze_staging_root = (
                            bronze_staging_parent
                            if bronze_staging_parent is not None
                            else scratch_root
                        )
                        bronze_staging_root.mkdir(parents=True, exist_ok=True)
                        local_bronze_dir = bronze_resume_staging_directory(
                            bronze_staging_root
                        )
                        object.__setattr__(config, "BRONZE_DIR", local_bronze_dir)
                        logger.info(
                            "Resumable Bronze staging enabled. Writing Bronze to %s",
                            local_bronze_dir,
                        )

                    process_bronze_gps_logs(
                        spark,
                        force_rebuild=args.force_rebuild,
                        batch_target_bytes=int(args.bronze_batch_gb * 1024**3),
                        work_root_parent=config.DATA_DIR,
                    )

                    if use_local_staging:
                        logger.info(
                            "Publishing staged Bronze dataset to %s", real_bronze_dir
                        )
                        object.__setattr__(config, "BRONZE_DIR", real_bronze_dir)
                        replace_directory_from_staging(
                            local_bronze_dir,
                            real_bronze_dir,
                            staging_root=bronze_staging_root,
                        )

            if args.stage in {"silver", "all"}:
                logger.info("=== STAGE 3: Silver Map Matching & Speed Engine ===")
                real_silver_dir = config.SILVER_DIR

                if (
                    not args.force_rebuild
                    and any(real_silver_dir.rglob("*.parquet"))
                    and layer_is_complete(
                        real_silver_dir,
                        layer="silver",
                        configuration_fingerprint=config.configuration_fingerprint(),
                    )
                ):
                    warn_on_configuration_drift(real_silver_dir, "Silver")
                    logger.info(
                        "Silver Parquet dataset already exists at %s. Skipping map matching.",
                        real_silver_dir,
                    )
                else:
                    use_local_staging = (
                        silver_staging_parent is not None
                        or is_external_drive(real_silver_dir)
                    )
                    if use_local_staging:
                        silver_staging_root = (
                            configured_staging_roots["silver"]
                            if silver_staging_parent is not None
                            else run_staging_root
                        )
                        local_silver_dir = silver_staging_root / "silver"
                        object.__setattr__(config, "SILVER_DIR", local_silver_dir)
                        logger.info(
                            "Temporary staging enabled. Writing Silver to %s",
                            local_silver_dir,
                        )

                    process_silver_map_matching(
                        spark,
                        force_rebuild=args.force_rebuild,
                        work_root_parent=real_silver_dir.parent,
                        source_shards=args.silver_source_shards,
                    )

                    if use_local_staging:
                        object.__setattr__(config, "SILVER_DIR", real_silver_dir)
                        replace_directory_from_staging(
                            local_silver_dir,
                            real_silver_dir,
                            staging_root=silver_staging_root,
                        )

            if args.stage in {"gold", "all"}:
                logger.info("=== STAGE 4: Gold Dataset Aggregation & Exporter ===")
                real_gold_dir = config.GOLD_DIR

                # Derive the completeness checklist from the single source of
                # truth in PipelineConfig instead of a hand-maintained list.
                all_exist = layer_is_complete(
                    real_gold_dir,
                    layer="gold",
                    configuration_fingerprint=config.configuration_fingerprint(),
                    required_artifacts=PipelineConfig.gold_relative_files().values(),
                )

                if not args.force_rebuild and all_exist:
                    warn_on_configuration_drift(real_gold_dir, "Gold")
                    logger.info(
                        "Gold datasets already exist at %s. Skipping aggregation.",
                        real_gold_dir,
                    )
                else:
                    use_local_staging = (
                        gold_staging_parent is not None
                        or is_external_drive(real_gold_dir)
                    )
                    if use_local_staging:
                        gold_staging_root = (
                            configured_staging_roots["gold"]
                            if gold_staging_parent is not None
                            else run_staging_root
                        )
                        local_gold_dir = gold_staging_root / "gold"

                        config.update_gold_directory(local_gold_dir)
                        logger.info(
                            "Temporary staging enabled. Writing Gold to %s",
                            local_gold_dir,
                        )

                    process_gold_aggregation(
                        spark,
                        force_rebuild=args.force_rebuild,
                        work_root_parent=real_gold_dir.parent,
                    )

                    if use_local_staging:
                        logger.info(
                            "Publishing staged Gold dataset to %s", real_gold_dir
                        )
                        config.update_gold_directory(real_gold_dir)

                        replace_directory_from_staging(
                            local_gold_dir,
                            real_gold_dir,
                            staging_root=gold_staging_root,
                        )

        finally:
            try:
                spark.stop()
                logger.info("Spark Session stopped.")
            finally:
                for layer_name, configured_root in configured_staging_roots.items():
                    remove_staging_directory(
                        configured_root,
                        staging_root=configured_staging_parents[layer_name],
                    )
                remove_staging_directory(
                    run_scratch_root,
                    staging_root=scratch_root,
                )

    logger.info("Pipeline execution completed successfully.")
    clean_output_directory(config.DATA_DIR)
    write_run_manifest(
        args.stage,
        args.force_rebuild,
        runtime_settings={
            "threads": args.threads,
            "shuffle_partitions": args.shuffle_partitions,
            "silver_source_shards": args.silver_source_shards,
            "bronze_staging_dir": (
                str(bronze_staging_parent) if bronze_staging_parent is not None else None
            ),
            "silver_staging_dir": (
                str(silver_staging_parent) if silver_staging_parent is not None else None
            ),
            "gold_staging_dir": (
                str(gold_staging_parent) if gold_staging_parent is not None else None
            ),
        },
    )


def run_cli() -> int:
    """Entry point mapping expected failures to concise messages and exit codes."""
    try:
        return main()
    except FileNotFoundError as exc:
        print(f"error: missing input or output artifact: {exc}", file=sys.stderr)
        _print_debug_hint()
        return 3
    except ValueError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        _print_debug_hint()
        return 2


def _print_debug_hint() -> None:
    if os.environ.get("KARETKI_DEBUG"):
        traceback.print_exc()
    else:
        print("Set KARETKI_DEBUG=1 for a full traceback.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(run_cli())

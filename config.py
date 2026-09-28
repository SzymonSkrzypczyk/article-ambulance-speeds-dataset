"""Central, publication-oriented configuration for the ambulance-speed ETL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class PipelineConfig:
    """Configuration shared by every pipeline stage.

    Scientific choices live here so that a publication run can serialize one
    complete configuration rather than relying on values duplicated in scripts.
    Values with unknown source semantics use neutral names and conservative
    defaults; provider-specific meaning must be supplied outside the code.
    """

    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    DATA_DIR: Path = field(init=False)

    INPUT_DATA_DIR: Path = field(init=False)
    INPUT_ZIP_DIR: Path = field(init=False)
    OSM_PBF_PATH: Optional[Path] = None

    RAW_CSV_DIR: Path = field(init=False)
    WEATHER_DIR: Path = field(init=False)
    WEATHER_PARQUET_PATH: Path = field(init=False)
    WEATHER_RAW_RESPONSE_PATH: Path = field(init=False)
    WEATHER_PROVENANCE_PATH: Path = field(init=False)
    OSM_DIR: Path = field(init=False)
    OSM_ROADS_PARQUET_PATH: Path = field(init=False)
    OSM_MATCH_SEGMENTS_PARQUET_PATH: Path = field(init=False)
    OSM_PROVENANCE_PATH: Path = field(init=False)
    BRONZE_DIR: Path = field(init=False)
    SILVER_DIR: Path = field(init=False)
    SILVER_EXCLUSIONS_DIR: Path = field(init=False)
    GOLD_DIR: Path = field(init=False)
    PIPELINE_REPORTS_DIR: Path = field(init=False)
    REPORTS_DIR: Path = field(init=False)
    BRONZE_QA_PATH: Path = field(init=False)
    SILVER_QA_PATH: Path = field(init=False)
    UNMATCHED_QA_PATH: Path = field(init=False)
    SPEED_EXCLUSION_QA_PATH: Path = field(init=False)
    SPEED_SOURCE_COMPARISON_QA_PATH: Path = field(init=False)
    RUN_MANIFEST_PATH: Path = field(init=False)

    GOLD_GEOPARQUET_PATH: Path = field(init=False)
    GOLD_GEOPACKAGE_PATH: Path = field(init=False)
    GOLD_CSV_PATH: Path = field(init=False)
    GOLD_FLAT_GEOPARQUET_PATH: Path = field(init=False)
    GOLD_FLAT_GEOPACKAGE_PATH: Path = field(init=False)
    GOLD_FLAT_CSV_PATH: Path = field(init=False)
    GOLD_BASE_GEOPARQUET_PATH: Path = field(init=False)
    GOLD_BASE_CSV_PATH: Path = field(init=False)
    GOLD_SEASON_PARQUET_PATH: Path = field(init=False)
    GOLD_SEASON_CSV_PATH: Path = field(init=False)
    GOLD_MONTH_PARQUET_PATH: Path = field(init=False)
    GOLD_MONTH_CSV_PATH: Path = field(init=False)
    GOLD_WEEK_PARQUET_PATH: Path = field(init=False)
    GOLD_WEEK_CSV_PATH: Path = field(init=False)
    GOLD_TOD_PARQUET_PATH: Path = field(init=False)
    GOLD_TOD_CSV_PATH: Path = field(init=False)
    # dev.2: first version carrying the hardening changes (canonical export
    # ordering, weather boundary day, coordinate reason codes, dissolved
    # compatibility lengths). dev.1 artifacts must not be reproducible under
    # this identifier because published bytes and QA reports differ.
    DATASET_VERSION: str = "2.0.0-dev.2"
    EXPECTED_START_DATE: str = "2021-01-01"
    EXPECTED_END_DATE: str = "2023-12-31"

    # Bounding box of the official Kraków municipality polygon (west, south,
    # east, north). Source: GUGiK Państwowy Rejestr Granic, municipality layer
    # A03_Granice_gmin, TERYT 1261011, queried in EPSG:4326 on 2026-08-24.
    # A bounding box includes some land outside the irregular city polygon.
    KRAKOW_BBOX: Tuple[float, float, float, float] = (
        19.792238,
        49.967665,
        20.217346,
        50.126135,
    )
    AOI_BOUNDARY_NAME: str = "Kraków municipality bounding box"
    AOI_BOUNDARY_SOURCE: str = "GUGiK Państwowy Rejestr Granic"
    AOI_BOUNDARY_SOURCE_URL: str = (
        "https://mapy.geoportal.gov.pl/wss/service/PZGIK/PRG/WFS/"
        "AdministrativeBoundaries"
    )
    AOI_BOUNDARY_IDENTIFIER: Optional[str] = "TERYT:1261011"
    AOI_BOUNDARY_RETRIEVED_DATE: Optional[str] = "2026-08-24"
    # A separate, deliberately broader processing envelope prevents telemetry
    # near Kraków from being deleted merely because it lies outside the city.
    # The city AOI remains available as an observation-level classification.
    ROAD_EXTRACTION_BBOX: Tuple[float, float, float, float] = (
        19.65,
        49.90,
        20.30,
        50.22,
    )
    KRAKOW_LAT: float = 50.0647
    KRAKOW_LON: float = 19.9450

    # Deterministic OSM acquisition. There is deliberately no automatic fallback.
    OSM_SOURCE: str = "osmnx_overpass"  # or "local_pbf"
    OSM_SNAPSHOT_UTC: str = "2023-12-31T23:59:59Z"
    OSM_NETWORK_TYPE: str = "drive"
    OSM_OVERPASS_TIMEOUT_SECONDS: int = 300

    # Pinning ERA5 avoids Open-Meteo's time-varying "Best Match" composition.
    WEATHER_API_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    WEATHER_MODEL: str = "era5"
    WEATHER_TIMEZONE: str = "UTC"
    WEATHER_LOCATION_DESCRIPTION: str = "single Kraków-centre grid-cell"

    CRS_WGS84: str = "EPSG:4326"
    CRS_METRIC_PL: str = "EPSG:2180"
    TIMEZONE_WARSAW: str = "Europe/Warsaw"
    SPARK_SESSION_TIMEZONE: str = "UTC"
    PRIMARY_TIMESTAMP_SOURCE: str = "GPS_TIME_UTC"
    MAX_TIMESTAMP_DISAGREEMENT_SECONDS: int = 300

    # Neutral until the data provider supplies an authoritative definition.
    SOURCE_ENTITY_ID_COLUMN: str = "GPS"
    SOURCE_ENTITY_ID_SEMANTIC: str = "UNVERIFIED_SOURCE_IDENTIFIER"

    ROAD_BUFFER_METERS: float = 15.0
    MATCHING_SENSITIVITY_RADII_METERS: Tuple[float, float, float] = (15.0, 25.0, 40.0)
    INCLUDE_SENSITIVITY_ONLY_MATCHES_IN_GOLD: bool = False
    MAX_VALID_SPEED_KMH: float = 150.0
    MIN_VALID_SPEED_KMH: float = 0.0
    RUNNING_SPEED_THRESHOLD_KMH: float = 2.0
    MAX_GAP_SECONDS: int = 30
    MIN_MOVEMENT_FOR_HEADING_METERS: float = 1.0
    # Provider telemetry is interpreted as centimetres per second.  This is the
    # default published speed source pending provider confirmation; displacement
    # remains available as an explicit, reproducible alternative.
    SPEED_SOURCE: str = "gps_speed"
    GPS_SPEED_UNIT: str = "cm/s"
    GPS_SPEED_TO_KMH_FACTOR: float = 0.036
    HEADING_SCORE_WEIGHT: float = 0.35

    # Silver retains this flag for audit; Gold can exclude flagged observations.
    OUTLIER_IQR_MULTIPLIER: float = 3.0
    OUTLIER_MIN_GROUP_SIZE: int = 20
    EXCLUDE_FLAGGED_SPEED_OUTLIERS_FROM_GOLD: bool = True

    # A support threshold, not a claim of statistical reliability.
    MIN_PROFILE_SAMPLE_COUNT: int = 10

    # Imputed operating-speed benchmarks, not asserted statutory/legal limits.
    DEFAULT_SPEED_LIMITS: Dict[str, int] = field(
        default_factory=lambda: {
            "motorway": 140,
            "motorway_link": 90,
            "trunk": 120,
            "trunk_link": 80,
            "primary": 50,
            "primary_link": 50,
            "secondary": 50,
            "secondary_link": 50,
            "tertiary": 50,
            "tertiary_link": 50,
            "unclassified": 50,
            "residential": 50,
            "living_street": 20,
            "service": 30,
        }
    )

    def __post_init__(self) -> None:
        self.INPUT_DATA_DIR = self.BASE_DIR.parent / "dataset_karetki" / "data"
        self.INPUT_ZIP_DIR = self.BASE_DIR / "zips"
        self.update_output_directory(self.BASE_DIR / "data")
        self.validate()

    @property
    def WEATHER_YEARS(self) -> Tuple[int, ...]:
        start_year = int(self.EXPECTED_START_DATE[:4])
        end_year = int(self.EXPECTED_END_DATE[:4])
        return tuple(range(start_year, end_year + 1))

    def validate(self) -> None:
        """Fail fast on scientifically unsafe or inconsistent settings."""
        west, south, east, north = self.KRAKOW_BBOX
        if not (west < east and south < north):
            raise ValueError("KRAKOW_BBOX must be ordered west, south, east, north")
        if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
            raise ValueError("KRAKOW_BBOX longitudes must be within [-180, 180]")
        if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
            raise ValueError("KRAKOW_BBOX latitudes must be within [-90, 90]")
        extraction_west, extraction_south, extraction_east, extraction_north = (
            self.ROAD_EXTRACTION_BBOX
        )
        if not (
            extraction_west < extraction_east
            and extraction_south < extraction_north
        ):
            raise ValueError(
                "ROAD_EXTRACTION_BBOX must be ordered west, south, east, north"
            )
        if not (
            extraction_west <= west
            and extraction_south <= south
            and extraction_east >= east
            and extraction_north >= north
        ):
            raise ValueError("ROAD_EXTRACTION_BBOX must contain KRAKOW_BBOX")
        if self.ROAD_BUFFER_METERS <= 0:
            raise ValueError("ROAD_BUFFER_METERS must be greater than zero")
        radii = self.MATCHING_SENSITIVITY_RADII_METERS
        if len(radii) != 3 or any(radius <= 0 for radius in radii):
            raise ValueError("MATCHING_SENSITIVITY_RADII_METERS must contain three positive radii")
        if tuple(sorted(radii)) != radii:
            raise ValueError("MATCHING_SENSITIVITY_RADII_METERS must be increasing")
        if self.ROAD_BUFFER_METERS > radii[-1]:
            raise ValueError(
                "ROAD_BUFFER_METERS cannot exceed the largest matching sensitivity radius"
            )
        if self.OSM_SOURCE not in {"osmnx_overpass", "local_pbf"}:
            raise ValueError("OSM_SOURCE must be 'osmnx_overpass' or 'local_pbf'")
        if self.OSM_SOURCE == "local_pbf" and self.OSM_PBF_PATH is None:
            raise ValueError("OSM_PBF_PATH is required when OSM_SOURCE='local_pbf'")
        if not self.WEATHER_MODEL:
            raise ValueError("WEATHER_MODEL must be explicitly pinned")
        if self.SPEED_SOURCE not in {"gps_speed", "displacement"}:
            raise ValueError("SPEED_SOURCE must be 'gps_speed' or 'displacement'")
        if self.GPS_SPEED_UNIT != "cm/s" or self.GPS_SPEED_TO_KMH_FACTOR != 0.036:
            raise ValueError("GPS_SPEED must use the fixed cm/s to km/h factor 0.036")
        if not 0.0 <= self.HEADING_SCORE_WEIGHT <= 1.0:
            raise ValueError("HEADING_SCORE_WEIGHT must be between 0 and 1")
        if self.OUTLIER_IQR_MULTIPLIER <= 0 or self.OUTLIER_MIN_GROUP_SIZE < 4:
            raise ValueError("Invalid IQR outlier configuration")

    def update_output_directory(self, new_dir: Path) -> None:
        """Update every output path from one root directory."""
        new_dir = Path(new_dir).resolve()
        self.DATA_DIR = new_dir
        self.RAW_CSV_DIR = new_dir / "raw_csvs"
        self.WEATHER_DIR = new_dir / "weather"
        self.WEATHER_PARQUET_PATH = (
            self.WEATHER_DIR / "krakow_weather_2021_2023.parquet"
        )
        self.WEATHER_RAW_RESPONSE_PATH = self.WEATHER_DIR / "open_meteo_response.json"
        self.WEATHER_PROVENANCE_PATH = self.WEATHER_DIR / "weather_provenance.json"
        self.OSM_DIR = new_dir / "osm"
        self.OSM_ROADS_PARQUET_PATH = self.OSM_DIR / "krakow_roads_epsg2180.parquet"
        self.OSM_MATCH_SEGMENTS_PARQUET_PATH = (
            self.OSM_DIR / "krakow_road_match_segments_epsg2180.parquet"
        )
        self.OSM_PROVENANCE_PATH = self.OSM_DIR / "osm_provenance.json"
        self.BRONZE_DIR = new_dir / "bronze"
        self.SILVER_DIR = new_dir / "silver"
        self.SILVER_EXCLUSIONS_DIR = new_dir / "silver_exclusions"
        self.GOLD_DIR = new_dir / "gold"
        self.PIPELINE_REPORTS_DIR = new_dir / "reports"
        self.REPORTS_DIR = self.GOLD_DIR / "reports"
        self.BRONZE_QA_PATH = self.PIPELINE_REPORTS_DIR / "bronze_attrition.parquet"
        self.SILVER_QA_PATH = self.PIPELINE_REPORTS_DIR / "silver_attrition.parquet"
        self.UNMATCHED_QA_PATH = self.PIPELINE_REPORTS_DIR / "unmatched_distance.parquet"
        self.SPEED_EXCLUSION_QA_PATH = (
            self.PIPELINE_REPORTS_DIR / "speed_derivation_exclusions.parquet"
        )
        self.SPEED_SOURCE_COMPARISON_QA_PATH = (
            self.PIPELINE_REPORTS_DIR / "speed_source_comparison.parquet"
        )
        self.RUN_MANIFEST_PATH = (
            self.PIPELINE_REPORTS_DIR / "pipeline_run_manifest.json"
        )
        self.update_gold_directory(self.GOLD_DIR)

    def update_gold_directory(self, new_gold_dir: Path) -> None:
        """Update paths when Gold is written through a staging directory."""
        new_gold_dir = Path(new_gold_dir).resolve()
        self.GOLD_DIR = new_gold_dir
        self.REPORTS_DIR = new_gold_dir / "reports"
        for attribute, relative in self.gold_relative_files().items():
            setattr(self, attribute, new_gold_dir / relative)

    @staticmethod
    def gold_relative_files() -> Dict[str, str]:
        """Single source of truth for every published Gold artifact path.

        Keyed by the PipelineConfig attribute each relative path is assigned
        to. Consumers (skip checks, validators, packaging) must derive the
        expected file set from here instead of duplicating names.
        """
        return {
            "GOLD_GEOPARQUET_PATH": "krakow_ambulance_speeds_2021_2023.parquet",
            "GOLD_GEOPACKAGE_PATH": "krakow_ambulance_speeds_2021_2023.gpkg",
            "GOLD_CSV_PATH": "krakow_ambulance_speeds_2021_2023.csv",
            "GOLD_FLAT_GEOPARQUET_PATH": (
                "krakow_ambulance_speeds_2021_2023_flat.parquet"
            ),
            "GOLD_FLAT_GEOPACKAGE_PATH": (
                "krakow_ambulance_speeds_2021_2023_flat.gpkg"
            ),
            "GOLD_FLAT_CSV_PATH": "krakow_ambulance_speeds_2021_2023_flat.csv",
            "GOLD_BASE_GEOPARQUET_PATH": "krakow_roads_base.parquet",
            "GOLD_BASE_CSV_PATH": "krakow_roads_base.csv",
            "GOLD_SEASON_PARQUET_PATH": "speeds_by_season.parquet",
            "GOLD_SEASON_CSV_PATH": "speeds_by_season.csv",
            "GOLD_MONTH_PARQUET_PATH": "speeds_by_month.parquet",
            "GOLD_MONTH_CSV_PATH": "speeds_by_month.csv",
            "GOLD_WEEK_PARQUET_PATH": "speeds_by_day_of_week.parquet",
            "GOLD_WEEK_CSV_PATH": "speeds_by_day_of_week.csv",
            "GOLD_TOD_PARQUET_PATH": "speeds_by_time_of_day.parquet",
            "GOLD_TOD_CSV_PATH": "speeds_by_time_of_day.csv",
        }

    def ensure_directories(self) -> None:
        for path in (
            self.DATA_DIR,
            self.RAW_CSV_DIR,
            self.WEATHER_DIR,
            self.OSM_DIR,
            self.BRONZE_DIR,
            self.SILVER_DIR,
            self.SILVER_EXCLUSIONS_DIR,
            self.GOLD_DIR,
            self.PIPELINE_REPORTS_DIR,
            self.REPORTS_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def configuration_fingerprint(self) -> str:
        from pipeline_utils import configuration_fingerprint as _fingerprint

        return _fingerprint(self.scientific_parameters())

    def scientific_parameters(self) -> dict:
        """Return parameters that determine scientific output values."""
        return {
            "dataset_version": self.DATASET_VERSION,
            "expected_start_date": self.EXPECTED_START_DATE,
            "expected_end_date": self.EXPECTED_END_DATE,
            "official_study_aoi_bbox_wgs84": list(self.KRAKOW_BBOX),
            "road_extraction_bbox_wgs84": list(self.ROAD_EXTRACTION_BBOX),
            "aoi_boundary_name": self.AOI_BOUNDARY_NAME,
            "aoi_boundary_source": self.AOI_BOUNDARY_SOURCE,
            "aoi_boundary_source_url": self.AOI_BOUNDARY_SOURCE_URL,
            "aoi_boundary_identifier": self.AOI_BOUNDARY_IDENTIFIER,
            "aoi_boundary_retrieved_date": self.AOI_BOUNDARY_RETRIEVED_DATE,
            "osm_source": self.OSM_SOURCE,
            "osm_snapshot_utc": self.OSM_SNAPSHOT_UTC,
            "weather_model": self.WEATHER_MODEL,
            "weather_coordinate": [self.KRAKOW_LAT, self.KRAKOW_LON],
            "source_entity_id_column": self.SOURCE_ENTITY_ID_COLUMN,
            "source_entity_id_semantic": self.SOURCE_ENTITY_ID_SEMANTIC,
            "primary_timestamp_source": self.PRIMARY_TIMESTAMP_SOURCE,
            "road_buffer_m": self.ROAD_BUFFER_METERS,
            "matching_sensitivity_radii_m": list(
                self.MATCHING_SENSITIVITY_RADII_METERS
            ),
            "include_sensitivity_only_matches_in_gold": (
                self.INCLUDE_SENSITIVITY_ONLY_MATCHES_IN_GOLD
            ),
            "valid_speed_range_kmh": [
                self.MIN_VALID_SPEED_KMH,
                self.MAX_VALID_SPEED_KMH,
            ],
            "max_gap_seconds": self.MAX_GAP_SECONDS,
            "speed_source": self.SPEED_SOURCE,
            "gps_speed_unit": self.GPS_SPEED_UNIT,
            "gps_speed_to_kmh_factor": self.GPS_SPEED_TO_KMH_FACTOR,
            "heading_score_weight": self.HEADING_SCORE_WEIGHT,
            "outlier_iqr_multiplier": self.OUTLIER_IQR_MULTIPLIER,
            "outlier_min_group_size": self.OUTLIER_MIN_GROUP_SIZE,
            "exclude_flagged_outliers_from_gold": self.EXCLUDE_FLAGGED_SPEED_OUTLIERS_FROM_GOLD,
            "minimum_profile_sample_count": self.MIN_PROFILE_SAMPLE_COUNT,
        }


config = PipelineConfig()

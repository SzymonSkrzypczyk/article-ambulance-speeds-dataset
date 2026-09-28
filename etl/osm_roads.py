"""Deterministic OSM road acquisition and matching-segment preparation."""

from __future__ import annotations

import hashlib
import logging
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional, Tuple

import geopandas as gpd
from shapely import reverse
from shapely.geometry import LineString, box

from config import config
from pipeline_utils import (
    layer_is_complete,
    parse_osm_maxspeed,
    sha256_file,
    utc_now_iso,
    write_json_atomic,
    write_layer_completion_marker,
)

logger = logging.getLogger(__name__)

DRIVABLE_HIGHWAY_CLASSES = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
]


def _package_version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def parse_maxspeed_value(value: Any, highway_type: str) -> int:
    """Backward-compatible numeric parser; new code also preserves provenance."""
    return parse_osm_maxspeed(
        value, highway_type, config.DEFAULT_SPEED_LIMITS
    ).value_kmh


def _fetch_osmnx() -> Tuple[gpd.GeoDataFrame, dict]:
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError(
            "OSM_SOURCE='osmnx_overpass' requires the pinned osmnx dependency"
        ) from exc

    snapshot_clause = f'[date:"{config.OSM_SNAPSHOT_UTC}"]'
    ox.settings.overpass_settings = (
        f"[out:json][timeout:{config.OSM_OVERPASS_TIMEOUT_SECONDS}]{snapshot_clause}"
    )
    logger.info(
        "Requesting OSMnx drive network for bbox=%s at snapshot=%s",
        config.ROAD_EXTRACTION_BBOX,
        config.OSM_SNAPSHOT_UTC,
    )
    graph = ox.graph_from_bbox(
        config.ROAD_EXTRACTION_BBOX,
        network_type=config.OSM_NETWORK_TYPE,
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    return edges, {
        # Keep the route machine-readable for provenance validation and retain
        # a separate display label for people reading the JSON directly.
        "source_type": "osmnx_overpass",
        "source_label": "OpenStreetMap via OSMnx/Overpass",
        "osmnx_version": _package_version("osmnx"),
        "overpass_settings": ox.settings.overpass_settings,
        "snapshot_utc": config.OSM_SNAPSHOT_UTC,
    }


def _fetch_local_pbf() -> Tuple[gpd.GeoDataFrame, dict]:
    import pyogrio

    pbf_path = Path(config.OSM_PBF_PATH).resolve()
    if not pbf_path.is_file():
        raise FileNotFoundError(f"Configured OSM PBF not found: {pbf_path}")
    west, south, east, north = config.ROAD_EXTRACTION_BBOX
    network = pyogrio.read_dataframe(
        pbf_path,
        layer="lines",
        bbox=(west, south, east, north),
        use_arrow=True,
    )
    if network.empty:
        raise ValueError(
            f"Local OSM PBF contains no line features in the configured bbox: {pbf_path}"
        )

    # GDAL's default OSM driver promotes `highway` but normally leaves these
    # two tags in its escaped hstore-like `other_tags` field.
    if "other_tags" in network.columns:
        for column in ("maxspeed", "oneway"):
            parsed = network["other_tags"].map(
                lambda value, tag_name=column: _extract_osm_other_tag(value, tag_name)
            )
            if column in network.columns:
                network[column] = network[column].where(network[column].notna(), parsed)
            else:
                network[column] = parsed
    if "highway" not in network.columns:
        raise ValueError(
            "GDAL OSM lines layer does not expose the required highway field"
        )
    network = network[network["highway"].isin(DRIVABLE_HIGHWAY_CLASSES)].copy()
    if network.empty:
        raise ValueError(
            "Local OSM PBF contains no supported drivable highway features"
        )
    return network.reset_index(drop=False), {
        "source_type": "local_pbf",
        "source_label": "OpenStreetMap local PBF via GDAL OSM driver/pyogrio",
        "pyogrio_version": _package_version("pyogrio"),
        "gdal_version": ".".join(str(value) for value in pyogrio.__gdal_version__),
        "pbf_path": str(pbf_path),
        "pbf_sha256": sha256_file(pbf_path),
        "snapshot_utc": None,
    }


_OSM_OTHER_TAG_PATTERN = re.compile(
    r'"(?P<key>(?:[^"\\]|\\.)*)"=>"(?P<value>(?:[^"\\]|\\.)*)"'
)


def _extract_osm_other_tag(other_tags: Any, requested_key: str) -> Optional[str]:
    """Extract one GDAL OSM `other_tags` value without splitting on commas."""
    if not isinstance(other_tags, str):
        return None
    for match in _OSM_OTHER_TAG_PATTERN.finditer(other_tags):
        key = match.group("key").replace(r"\"", '"').replace(r"\\", "\\")
        if key == requested_key:
            return match.group("value").replace(r"\"", '"').replace(r"\\", "\\")
    return None


def _normalize_edges(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize source fields while preserving tag and direction provenance."""
    if raw.crs is None:
        raw = raw.set_crs(config.CRS_WGS84)
    else:
        raw = raw.to_crs(config.CRS_WGS84)

    if "osm_id" not in raw.columns:
        if "osmid" in raw.columns:
            raw["osm_id"] = raw["osmid"]
        elif "id" in raw.columns:
            raw["osm_id"] = raw["id"]
        else:
            raise ValueError("OSM source contains no osm_id/osmid/id field")

    raw = raw.copy()
    raw["osm_ids_raw"] = raw["osm_id"].astype(str)
    raw["osm_id"] = raw["osm_id"].map(_first).astype("int64")
    for column, default in (
        ("name", None),
        ("highway", "unclassified"),
        ("maxspeed", None),
        ("oneway", False),
    ):
        if column not in raw.columns:
            raw[column] = default
        raw[column] = raw[column].map(_first)

    raw["name"] = raw["name"].where(raw["name"].notna(), None)
    raw["highway"] = raw["highway"].fillna("unclassified").astype(str)
    raw["maxspeed_raw"] = raw["maxspeed"].where(raw["maxspeed"].notna(), None)
    parsed = [
        parse_osm_maxspeed(value, highway, config.DEFAULT_SPEED_LIMITS)
        for value, highway in zip(raw["maxspeed_raw"], raw["highway"])
    ]
    raw["maxspeed_kmh"] = [item.value_kmh for item in parsed]
    raw["maxspeed_source"] = [item.source for item in parsed]

    raw["oneway_raw"] = raw["oneway"].astype(str)
    raw["oneway"] = raw["oneway_raw"].str.lower().isin({"true", "1", "yes", "-1"})
    raw["oneway_reversed"] = raw["oneway_raw"].str.lower().eq("-1")
    raw.loc[raw["oneway_reversed"], "geometry"] = raw.loc[
        raw["oneway_reversed"], "geometry"
    ].map(reverse)

    envelope = gpd.GeoDataFrame(
        geometry=[box(*config.ROAD_EXTRACTION_BBOX)], crs=config.CRS_WGS84
    )
    clipped = gpd.clip(raw, envelope, keep_geom_type=True)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    clipped = clipped.explode(index_parts=False, ignore_index=True)
    clipped = clipped[clipped.geometry.geom_type == "LineString"].copy()
    clipped = clipped[clipped.geometry.length > 0].copy()
    official_envelope = box(*config.KRAKOW_BBOX)
    clipped["intersects_official_krakow_bbox"] = clipped.geometry.intersects(
        official_envelope
    )

    def feature_id(row: Any) -> str:
        identifiers = ":".join(
            str(row.get(key, "")) for key in ("osm_id", "u", "v", "key")
        )
        digest = hashlib.sha256(row.geometry.wkb).hexdigest()[:16]
        return f"{identifiers}:{digest}"

    clipped["road_feature_id"] = clipped.apply(feature_id, axis=1)
    clipped = clipped.drop_duplicates("road_feature_id").reset_index(drop=True)
    return clipped[
        [
            "road_feature_id",
            "osm_id",
            "osm_ids_raw",
            "name",
            "highway",
            "maxspeed_raw",
            "maxspeed_kmh",
            "maxspeed_source",
            "oneway",
            "oneway_raw",
            "oneway_reversed",
            "intersects_official_krakow_bbox",
            "geometry",
        ]
    ]


def _atomic_match_segments(roads_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Split curved features into atomic coordinate-pair segments for local heading."""
    rows = []
    attribute_columns = [
        column for column in roads_metric.columns if column != "geometry"
    ]
    for road in roads_metric.itertuples(index=False):
        attributes = {column: getattr(road, column) for column in attribute_columns}
        coordinates = list(road.geometry.coords)
        for index, (start, end) in enumerate(zip(coordinates, coordinates[1:])):
            segment = LineString([start, end])
            if segment.length <= 0:
                continue
            rows.append(
                {
                    **attributes,
                    "match_segment_id": f"{attributes['road_feature_id']}:{index}",
                    "geometry": segment,
                }
            )
    if not rows:
        raise ValueError("OSM preprocessing produced no matchable line segments")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=roads_metric.crs)


def load_krakow_osm_roads() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """Acquire exactly one configured OSM source and prepare base/match layers."""
    if config.OSM_SOURCE == "osmnx_overpass":
        raw, provenance = _fetch_osmnx()
    elif config.OSM_SOURCE == "local_pbf":
        raw, provenance = _fetch_local_pbf()
    else:  # guarded by config.validate, retained for defensive callers
        raise ValueError(f"Unsupported OSM_SOURCE={config.OSM_SOURCE!r}")

    roads = _normalize_edges(raw).to_crs(config.CRS_METRIC_PL)
    segments = _atomic_match_segments(roads)
    # Canonical ordering makes the layer bytes independent of graph
    # iteration order, so identical inputs hash identically across runs.
    roads = roads.sort_values("road_feature_id", kind="stable").reset_index(drop=True)
    segments = segments.sort_values("match_segment_id", kind="stable").reset_index(
        drop=True
    )
    provenance.update(
        {
            "retrieved_at_utc": utc_now_iso(),
            "official_study_aoi_bbox_wgs84": list(config.KRAKOW_BBOX),
            "configured_extraction_bbox_wgs84": list(config.ROAD_EXTRACTION_BBOX),
            "actual_bounds_epsg2180": [float(value) for value in roads.total_bounds],
            "base_feature_count": int(len(roads)),
            "match_segment_count": int(len(segments)),
            "base_geometry_types": {
                str(key): int(value)
                for key, value in roads.geometry.geom_type.value_counts().items()
            },
            "crs": config.CRS_METRIC_PL,
            "automatic_fallback_used": False,
        }
    )
    return roads, segments, provenance


def process_osm_road_network(force_rebuild: bool = False) -> Path:
    """Create deterministic base-road and atomic matching-segment GeoParquet files."""
    outputs_exist = (
        config.OSM_ROADS_PARQUET_PATH.exists()
        and config.OSM_MATCH_SEGMENTS_PARQUET_PATH.exists()
        and config.OSM_PROVENANCE_PATH.exists()
    )
    if outputs_exist and layer_is_complete(
        config.OSM_DIR,
        layer="osm",
        configuration_fingerprint=config.configuration_fingerprint(),
        required_artifacts=(
            config.OSM_ROADS_PARQUET_PATH.relative_to(config.OSM_DIR),
            config.OSM_MATCH_SEGMENTS_PARQUET_PATH.relative_to(config.OSM_DIR),
            config.OSM_PROVENANCE_PATH.relative_to(config.OSM_DIR),
        ),
    ) and not force_rebuild:
        logger.info("OSM road outputs already exist; skipping acquisition")
        return config.OSM_ROADS_PARQUET_PATH

    config.ensure_directories()
    roads, segments, provenance = load_krakow_osm_roads()
    roads.to_parquet(config.OSM_ROADS_PARQUET_PATH, index=False)
    segments.to_parquet(config.OSM_MATCH_SEGMENTS_PARQUET_PATH, index=False)
    provenance["base_parquet_sha256"] = sha256_file(config.OSM_ROADS_PARQUET_PATH)
    provenance["match_segments_parquet_sha256"] = sha256_file(
        config.OSM_MATCH_SEGMENTS_PARQUET_PATH
    )
    write_json_atomic(config.OSM_PROVENANCE_PATH, provenance)
    write_layer_completion_marker(
        config.OSM_DIR,
        {
            "layer": "osm",
            "dataset_version": config.DATASET_VERSION,
            "artifacts": [
                str(config.OSM_ROADS_PARQUET_PATH.relative_to(config.OSM_DIR)),
                str(config.OSM_MATCH_SEGMENTS_PARQUET_PATH.relative_to(config.OSM_DIR)),
                str(config.OSM_PROVENANCE_PATH.relative_to(config.OSM_DIR)),
            ],
            "configuration_fingerprint": config.configuration_fingerprint(),
        },
    )
    logger.info(
        "Saved %d road features and %d atomic match segments",
        len(roads),
        len(segments),
    )
    return config.OSM_ROADS_PARQUET_PATH


if __name__ == "__main__":
    process_osm_road_network(force_rebuild=True)

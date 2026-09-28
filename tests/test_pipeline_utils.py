import math
import os

import pytest

from config import PipelineConfig, config
from etl.osm_roads import _extract_osm_other_tag
from pipeline_utils import (
    bidirectional_heading_difference_radians,
    circular_angle_difference_radians,
    layer_is_complete,
    parse_osm_maxspeed,
    step_is_complete,
    write_layer_completion_marker,
    write_step_completion_manifest,
)

DEFAULTS = {"residential": 50, "living_street": 20}


def test_default_aoi_uses_official_krakow_municipality_envelope():
    cfg = PipelineConfig()
    assert cfg.KRAKOW_BBOX == (19.792238, 49.967665, 20.217346, 50.126135)
    assert cfg.AOI_BOUNDARY_IDENTIFIER == "TERYT:1261011"
    assert cfg.ROAD_EXTRACTION_BBOX == (19.65, 49.90, 20.30, 50.22)


def test_layer_manifest_requires_matching_configuration_and_artifacts(tmp_path):
    layer = tmp_path / "gold"
    layer.mkdir()
    (layer / "primary.parquet").write_bytes(b"primary")
    (layer / "compatibility" / "variant.parquet").parent.mkdir()
    (layer / "compatibility" / "variant.parquet").write_bytes(b"variant")
    write_layer_completion_marker(
        layer,
        {
            "layer": "gold",
            "configuration_fingerprint": "current-config",
            "artifacts": ["primary.parquet", "compatibility/variant.parquet"],
        },
    )

    assert layer_is_complete(
        layer,
        layer="gold",
        configuration_fingerprint="current-config",
        required_artifacts=["primary.parquet", "compatibility/variant.parquet"],
    )
    assert not layer_is_complete(
        layer,
        layer="gold",
        configuration_fingerprint="changed-config",
        required_artifacts=["primary.parquet", "compatibility/variant.parquet"],
    )

    primary = layer / "primary.parquet"
    primary.write_bytes(b"changed")
    modified_at = primary.stat().st_mtime_ns + 1_000_000
    os.utime(primary, ns=(modified_at, modified_at))
    assert not layer_is_complete(
        layer,
        layer="gold",
        configuration_fingerprint="current-config",
        required_artifacts=["primary.parquet", "compatibility/variant.parquet"],
    )


def test_step_manifest_requires_unchanged_artifacts_and_configuration(tmp_path):
    root = tmp_path / "silver-work"
    artifact = root / "direction" / "chunk-01"
    artifact.mkdir(parents=True)
    (artifact / "part-0000.parquet").write_bytes(b"checkpoint")
    write_step_completion_manifest(
        root,
        layer="silver",
        step="direction-0001",
        artifacts=["direction/chunk-01"],
        configuration_fingerprint="config-a",
        details={"bronze_rows": 12},
    )
    assert step_is_complete(
        root, layer="silver", step="direction-0001", configuration_fingerprint="config-a"
    )
    assert not step_is_complete(
        root, layer="silver", step="direction-0001", configuration_fingerprint="config-b"
    )
    (artifact / "part-0000.parquet").write_bytes(b"modified")
    assert not step_is_complete(
        root, layer="silver", step="direction-0001", configuration_fingerprint="config-a"
    )

@pytest.mark.parametrize(
    ("raw", "expected", "source"),
    [
        ("50", 50, "OSM_NUMERIC_KMH"),
        ("30 mph", 48, "OSM_NUMERIC_MPH_CONVERTED"),
        ("PL:living_street", 20, "OSM_SYMBOLIC_TAG"),
        (None, 50, "IMPUTED_HIGHWAY_DEFAULT"),
        ("none", 50, "IMPUTED_UNPARSEABLE_TAG"),
    ],
)
def test_maxspeed_provenance(raw, expected, source):
    parsed = parse_osm_maxspeed(raw, "residential", DEFAULTS)
    assert parsed.value_kmh == expected
    assert parsed.source == source


def test_circular_angle_wraparound():
    difference = circular_angle_difference_radians(math.radians(359), math.radians(1))
    assert math.degrees(difference) == pytest.approx(2.0)


def test_bidirectional_heading_treats_reverse_as_aligned():
    assert bidirectional_heading_difference_radians(0.0, math.pi) == pytest.approx(0.0)


def test_invalid_speed_source_is_rejected():
    cfg = PipelineConfig()
    cfg.SPEED_SOURCE = "fallback"
    with pytest.raises(ValueError, match="SPEED_SOURCE"):
        cfg.validate()


def test_speed_source_changes_scientific_fingerprint():
    gps_primary = PipelineConfig()
    displacement = PipelineConfig()
    displacement.SPEED_SOURCE = "displacement"
    assert gps_primary.configuration_fingerprint() != displacement.configuration_fingerprint()


def test_local_pbf_requires_explicit_path():
    cfg = PipelineConfig()
    cfg.OSM_SOURCE = "local_pbf"
    with pytest.raises(ValueError, match="OSM_PBF_PATH"):
        cfg.validate()


def test_aoi_must_use_valid_ordered_wgs84_bounds():
    cfg = PipelineConfig()
    cfg.KRAKOW_BBOX = (20.0, 50.0, 19.0, 51.0)
    with pytest.raises(ValueError, match="ordered"):
        cfg.validate()

    cfg.KRAKOW_BBOX = (-181.0, 50.0, 19.0, 51.0)
    with pytest.raises(ValueError, match="longitudes"):
        cfg.validate()


def test_snap_radius_must_be_positive():
    cfg = PipelineConfig()
    cfg.ROAD_BUFFER_METERS = 0.0
    with pytest.raises(ValueError, match="greater than zero"):
        cfg.validate()


def test_road_extraction_bbox_must_contain_official_aoi():
    cfg = PipelineConfig()
    cfg.ROAD_EXTRACTION_BBOX = (19.9, 50.0, 20.0, 50.1)
    with pytest.raises(ValueError, match="must contain"):
        cfg.validate()


def test_extract_gdal_osm_other_tags_without_naive_comma_split():
    tags = r'"maxspeed"=>"50","description"=>"A, B","oneway"=>"-1"'
    assert _extract_osm_other_tag(tags, "maxspeed") == "50"
    assert _extract_osm_other_tag(tags, "description") == "A, B"
    assert _extract_osm_other_tag(tags, "oneway") == "-1"
    assert _extract_osm_other_tag(tags, "missing") is None

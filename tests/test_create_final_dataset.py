"""Tests for the exact-file minimal Gold export."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from create_final_dataset import ALL_FILES, PARQUET_FILES, build, verify_output


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "_KARETKI_COMPLETE.json").write_text(
        json.dumps({"layer": "gold", "dataset_version": "2.0.0",
                    "configuration_fingerprint": "fixture-fingerprint"}), encoding="utf-8"
    )
    for name in PARQUET_FILES:
        if name == "krakow_roads_base.parquet":
            table = pa.table({"road_feature_id": ["road-1"]})
        else:
            table = pa.table({
                "road_feature_id": ["road-1", "road-1"],
                "n_samples": [2, 3],
                "spatial_scope": ["OFFICIAL_KRAKOW_BOUNDING_BOX", "EXTENDED_PROCESSING_AREA"],
            })
        pq.write_table(table, gold / name)
    (gold / "unrelated.csv").write_text("not for publication", encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "title": "Test road-speed dataset",
        "version": "2.0.0",
        "authors": [{"given-names": "Example", "family-names": "Author"}],
        "licence_text": "Approved example component terms.",
    }), encoding="utf-8")
    return gold, metadata


def test_build_exact_files_and_checksums(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    gold, metadata = inputs
    output = tmp_path / "release"
    build(gold, output, metadata)
    assert {path.name for path in output.iterdir()} == ALL_FILES
    assert not any(path.is_dir() for path in output.iterdir())
    assert "5 retained speed observations" in (output / "README.md").read_text()
    assert "2 in the official" in (output / "README.md").read_text()
    assert "3 in the extended" in (output / "README.md").read_text()
    assert "release version **2.0.0**" in (output / "README.md").read_text()
    assert "Example" in (output / "citations.cff").read_text()
    verify_output(output)
    assert (gold / "unrelated.csv").is_file()


def test_refuses_to_replace_existing_destination(inputs: tuple[Path, Path],
                                                 tmp_path: Path) -> None:
    gold, metadata = inputs
    output = tmp_path / "release"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build(gold, output, metadata)
    assert (output / "keep.txt").read_text() == "keep"


def test_refuses_unapproved_licence(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    gold, metadata = inputs
    data = json.loads(metadata.read_text())
    data["licence_text"] = "REPLACE WITH TERMS"
    metadata.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="placeholders"):
        build(gold, tmp_path / "release", metadata)
    assert not (tmp_path / "release").exists()


def test_refuses_inconsistent_counts(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    gold, metadata = inputs
    pq.write_table(pa.table({
        "road_feature_id": ["road-1"],
        "n_samples": [999],
        "spatial_scope": ["OFFICIAL_KRAKOW_BOUNDING_BOX"],
    }), gold / "speeds_by_month.parquet")
    with pytest.raises(ValueError, match="do not reconcile"):
        build(gold, tmp_path / "release", metadata)
    assert not (tmp_path / "release").exists()


def test_distinct_package_version_requires_explicit_source_version(
        inputs: tuple[Path, Path], tmp_path: Path) -> None:
    gold, metadata = inputs
    data = json.loads(metadata.read_text())
    data["version"] = "1.0.0"
    metadata.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="source_gold_version"):
        build(gold, tmp_path / "release", metadata)
    data["source_gold_version"] = "2.0.0"
    metadata.write_text(json.dumps(data), encoding="utf-8")
    build(gold, tmp_path / "release", metadata)
    assert "release version **1.0.0**" in (tmp_path / "release/README.md").read_text()
    assert "version: 1.0.0" in (tmp_path / "release/citations.cff").read_text()
    assert "2.0.0" not in (tmp_path / "release/README.md").read_text()

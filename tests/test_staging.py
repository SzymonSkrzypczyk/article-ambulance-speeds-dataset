from pathlib import Path

import pytest

from etl.bronze_gps import _batch_files_by_size
from main_pipeline import bronze_resume_staging_directory
from pipeline_utils import (
    _refresh_layer_completion_inventory,
    layer_is_complete,
    remove_staging_directory,
    replace_directory_from_staging,
    write_layer_completion_marker,
)


def test_staging_swap_replaces_target_and_removes_source(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    source = staging_root / "silver"
    source.mkdir(parents=True)
    (source / "part-00000.parquet").write_bytes(b"new")

    target = tmp_path / "output" / "silver"
    target.mkdir(parents=True)
    (target / "part-old.parquet").write_bytes(b"old")

    replace_directory_from_staging(source, target, staging_root=staging_root)

    assert (target / "part-00000.parquet").read_bytes() == b"new"
    assert not (target / "part-old.parquet").exists()
    assert not source.exists()


def test_completion_inventory_can_be_refreshed_after_copy_timestamp_change(
    tmp_path: Path,
) -> None:
    layer = tmp_path / "bronze"
    partition = layer / "year=2021" / "quarter=Q1"
    partition.mkdir(parents=True)
    part_file = partition / "part-00000.parquet"
    part_file.write_bytes(b"data")
    write_layer_completion_marker(
        layer,
        {
            "layer": "bronze",
            "artifacts": ["year=2021/quarter=Q1"],
            "configuration_fingerprint": "a" * 64,
        },
    )
    assert layer_is_complete(
        layer, layer="bronze", configuration_fingerprint="a" * 64
    )

    # Simulate a destination filesystem changing only the copied file mtime.
    stat = part_file.stat()
    part_file.touch()
    assert part_file.stat().st_mtime_ns >= stat.st_mtime_ns
    _refresh_layer_completion_inventory(layer)
    assert layer_is_complete(
        layer, layer="bronze", configuration_fingerprint="a" * 64
    )


def test_staging_removal_rejects_root_and_outside_path(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="staging root"):
        remove_staging_directory(staging_root, staging_root=staging_root)

    with pytest.raises(ValueError, match="non-staging"):
        remove_staging_directory(outside, staging_root=staging_root)

    assert staging_root.exists()
    assert outside.exists()


def test_staging_swap_requires_parquet_output(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    source = staging_root / "empty"
    source.mkdir(parents=True)

    with pytest.raises(ValueError, match="missing Parquet"):
        replace_directory_from_staging(
            source,
            tmp_path / "target",
            staging_root=staging_root,
        )


def test_same_volume_staging_swap_renames_without_copy(
    tmp_path: Path, monkeypatch
) -> None:
    staging_root = tmp_path / "staging"
    source = staging_root / "run" / "silver"
    target = tmp_path / "output" / "silver"
    source.mkdir(parents=True)
    (source / "part.parquet").write_bytes(b"data")

    def fail_copy(*args, **kwargs):
        raise AssertionError("same-volume publication must not copy the dataset")

    monkeypatch.setattr("pipeline_utils.shutil.copytree", fail_copy)
    replace_directory_from_staging(source, target, staging_root=staging_root)

    assert not source.exists()
    assert (target / "part.parquet").read_bytes() == b"data"


def test_file_batches_are_bounded_and_deterministic(tmp_path: Path) -> None:
    files = [tmp_path / f"{name}.csv" for name in ("a", "b", "c")]
    for path, size in zip(files, (4, 4, 2)):
        path.write_bytes(b"x" * size)

    batches = _batch_files_by_size(files, target_bytes=6)

    assert batches == [[files[0]], [files[1], files[2]]]


def test_bronze_resume_staging_directory_is_stable(tmp_path: Path) -> None:
    staging_parent = tmp_path / "scratch"

    assert bronze_resume_staging_directory(staging_parent) == (
        staging_parent / ".karetki-bronze-resume-staging"
    )

"""Pure utility functions shared by ETL, validation, packaging, and tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from uuid import uuid4


@dataclass(frozen=True)
class MaxspeedResult:
    """Parsed OSM maxspeed plus explicit provenance."""

    value_kmh: int
    source: str
    raw: Optional[str]


def parse_osm_maxspeed(
    value: Any,
    highway_type: str,
    defaults: Mapping[str, int],
) -> MaxspeedResult:
    """Parse an OSM maxspeed tag without disguising imputation as observation."""
    fallback = int(defaults.get(str(highway_type), 50))
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return MaxspeedResult(fallback, "IMPUTED_HIGHWAY_DEFAULT", None)

    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return MaxspeedResult(fallback, "IMPUTED_HIGHWAY_DEFAULT", raw or None)

    first = raw.split(";")[0].strip().lower()
    symbolic = {
        "pl:urban": 50,
        "pl:rural": 90,
        "pl:trunk": 120,
        "pl:motorway": 140,
        "pl:living_street": 20,
        "walk": 5,
    }
    if first in symbolic:
        return MaxspeedResult(symbolic[first], "OSM_SYMBOLIC_TAG", raw)

    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(mph|km/?h|kph)?", first)
    if match:
        numeric = float(match.group(1))
        if match.group(2) == "mph":
            numeric *= 1.609344
            source = "OSM_NUMERIC_MPH_CONVERTED"
        else:
            source = "OSM_NUMERIC_KMH"
        rounded = int(round(numeric))
        if 5 <= rounded <= 140:
            return MaxspeedResult(rounded, source, raw)

    return MaxspeedResult(fallback, "IMPUTED_UNPARSEABLE_TAG", raw)


def circular_angle_difference_radians(angle_a: float, angle_b: float) -> float:
    """Return the smallest angular difference in [0, pi]."""
    return abs((angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi)


def bidirectional_heading_difference_radians(angle_a: float, angle_b: float) -> float:
    """Return alignment error to an undirected line in [0, pi/2]."""
    directional = circular_angle_difference_radians(angle_a, angle_b)
    return min(directional, math.pi - directional)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write strict JSON through a unique temp file and atomically replace target.

    Non-finite floats are normalized to null and ``allow_nan=False`` forbids
    the rest, so reports can never emit ``NaN``/``Infinity`` tokens that strict
    JSON parsers reject. The temporary name carries a UUID suffix so two
    concurrent writers cannot corrupt each other's sibling file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    text = (
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    """Recursively convert non-finite floats to None for strict-JSON output."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def is_execution_sidecar(name: str) -> bool:
    """True for Spark/macOS execution artifacts that never belong in a release."""
    base = name.rsplit("/", 1)[-1]
    return (
        base.startswith("._")
        or base == ".DS_Store"
        or base.endswith(".crc")
        or base == "_SUCCESS"
        or base == LAYER_COMPLETE_MARKER
    )


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


LAYER_COMPLETE_MARKER = "_KARETKI_COMPLETE.json"
STEP_MANIFEST_DIRECTORY = ".karetki-step-manifests"


def step_manifest_path(root: Path, step: str) -> Path:
    """Return the durable, per-step completion manifest path.

    Step names are deliberately restricted to simple path components: a
    manifest must never be able to escape the layer/work directory.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", step):
        raise ValueError(f"Invalid pipeline step name: {step!r}")
    return Path(root) / STEP_MANIFEST_DIRECTORY / f"{step}.json"


def write_step_completion_manifest(
    root: Path,
    *,
    layer: str,
    step: str,
    artifacts: Iterable[Path | str],
    configuration_fingerprint: str,
    details: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically mark one restartable pipeline step complete.

    The manifest is written *after* its artifacts.  On restart callers use
    :func:`step_is_complete` rather than trusting a file's mere existence;
    this catches interrupted writes and accidental modification.
    """
    root = Path(root)
    relative_artifacts = [str(Path(artifact)) for artifact in artifacts]
    if not relative_artifacts:
        raise ValueError("Step manifests require at least one artifact")
    manifest = step_manifest_path(root, step)
    payload = {
        "layer": layer,
        "step": step,
        "completed_at_utc": utc_now_iso(),
        "configuration_fingerprint": configuration_fingerprint,
        "artifacts": relative_artifacts,
        "artifact_inventory": _artifact_inventory(root, relative_artifacts),
    }
    if details:
        payload["details"] = dict(details)
    write_json_atomic(manifest, payload)
    return manifest


def step_is_complete(
    root: Path,
    *,
    layer: str,
    step: str,
    configuration_fingerprint: str,
) -> bool:
    """Validate that a previously completed step is safe to reuse."""
    root = Path(root)
    manifest = step_manifest_path(root, step)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    artifacts = payload.get("artifacts")
    return (
        payload.get("layer") == layer
        and payload.get("step") == step
        and payload.get("configuration_fingerprint") == configuration_fingerprint
        and isinstance(artifacts, list)
        and bool(artifacts)
        and all(isinstance(artifact, str) and (root / artifact).exists() for artifact in artifacts)
        and payload.get("artifact_inventory") == _artifact_inventory(root, artifacts)
    )


def _artifact_inventory(
    root: Path, artifacts: Iterable[str]
) -> dict[str, dict[str, int | str]]:
    """Return a cheap metadata fingerprint without reading artifact contents."""
    inventory: dict[str, dict[str, int | str]] = {}
    for artifact in artifacts:
        path = root / artifact
        files = (
            [path]
            if path.is_file()
            else sorted(file for file in path.rglob("*") if file.is_file())
        )
        metadata = []
        for file in files:
            stat = file.stat()
            metadata.append(
                f"{file.relative_to(root)}\\0{stat.st_size}\\0{stat.st_mtime_ns}"
            )
        inventory[artifact] = {
            "file_count": len(files),
            "bytes": sum(file.stat().st_size for file in files),
            "metadata_sha256": hashlib.sha256("\\n".join(metadata).encode()).hexdigest(),
        }
    return inventory


def write_layer_completion_marker(layer_dir: Path, payload: dict) -> Path:
    """Write the completion sentinel as the LAST artifact of a pipeline layer.

    A crash mid-publish therefore leaves a layer without the marker, and no
    downstream stage will mistake the partial output for a finished one.
    """
    root = Path(layer_dir)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
        raise ValueError("Layer completion manifests require a string artifact inventory")
    path = root / LAYER_COMPLETE_MARKER
    write_json_atomic(
        path,
        {
            "completed_at_utc": utc_now_iso(),
            **payload,
            "artifact_inventory": _artifact_inventory(root, artifacts),
        },
    )
    return path


def _refresh_layer_completion_inventory(layer_dir: Path) -> None:
    """Refresh a copied layer marker after destination filesystem timestamp changes.

    Cross-volume publication uses ``copytree``. Filesystems such as exFAT can
    round or otherwise alter copied mtimes, while retaining identical file
    paths and bytes. Completion inventories intentionally include mtimes to
    catch later local edits, so recompute the inventory immediately after this
    controlled copy rather than treating the published layer as incomplete.
    """
    payload = read_layer_completion_marker(layer_dir)
    if payload is not None:
        write_layer_completion_marker(layer_dir, payload)


def layer_is_complete(
    layer_dir: Path,
    *,
    layer: Optional[str] = None,
    configuration_fingerprint: Optional[str] = None,
    required_artifacts: Iterable[Path | str] = (),
) -> bool:
    """Return whether a layer manifest proves it is reusable for this run."""
    root = Path(layer_dir)
    payload = read_layer_completion_marker(root)
    if payload is None:
        return False
    if layer is not None and payload.get("layer") != layer:
        return False
    if (
        configuration_fingerprint is not None
        and payload.get("configuration_fingerprint") != configuration_fingerprint
    ):
        return False
    manifest_artifacts = payload.get("artifacts")
    if not isinstance(manifest_artifacts, list) or not all(
        isinstance(artifact, str) for artifact in manifest_artifacts
    ):
        return False
    # Completion manifests use POSIX-relative paths so their contents are
    # portable between Windows and POSIX hosts.
    expected_artifacts = [Path(artifact).as_posix() for artifact in required_artifacts]
    if expected_artifacts and set(manifest_artifacts) != set(expected_artifacts):
        return False
    if not all((root / Path(artifact)).exists() for artifact in manifest_artifacts):
        return False
    return payload.get("artifact_inventory") == _artifact_inventory(root, manifest_artifacts)


def configuration_fingerprint(parameters: Mapping) -> str:
    """Deterministic digest of a scientific-parameter mapping.

    Stored inside each layer's completion sentinel so reuse decisions and
    publication packaging can prove that every layer was produced under one
    identical scientific configuration rather than assuming it.
    """
    canonical = json.dumps(
        parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_layer_completion_marker(layer_dir: Path) -> Optional[dict]:
    """Parse a completion sentinel; None when absent or unreadable."""
    path = Path(layer_dir) / LAYER_COMPLETE_MARKER
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def git_revision(repo_dir: Path) -> dict:
    """Return reproducibility-relevant Git state without failing outside Git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", str(repo_dir)],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def remove_staging_directory(path: Path, staging_root: Optional[Path] = None) -> None:
    """Remove one explicit staging directory and reject broader targets."""
    resolved = path.resolve()
    root = (
        staging_root.resolve()
        if staging_root is not None
        else (Path(tempfile.gettempdir()) / "karetki_pipeline_staging").resolve()
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove non-staging path: {resolved}") from exc
    if resolved == root:
        raise ValueError(f"Refusing to remove the staging root itself: {root}")
    if resolved.exists():
        shutil.rmtree(resolved)


def replace_directory_from_staging(
    source: Path,
    target: Path,
    staging_root: Optional[Path] = None,
) -> None:
    """Copy to the target volume, then swap directories with rollback support."""
    source = source.resolve()
    target = target.resolve()
    root = (
        staging_root.resolve()
        if staging_root is not None
        else (Path(tempfile.gettempdir()) / "karetki_pipeline_staging").resolve()
    )
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing non-staging source: {source}") from exc
    if not source.is_dir() or not any(source.rglob("*.parquet")):
        raise ValueError(f"Staging output is missing Parquet data: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    incoming = target.parent / f".{target.name}.incoming-{token}"
    backup = target.parent / f".{target.name}.backup-{token}"
    if incoming.exists() or backup.exists():
        raise FileExistsError("Unexpected staging swap path collision")

    same_volume = source.stat().st_dev == target.parent.stat().st_dev
    source_moved = False
    if same_volume:
        source.rename(incoming)
        source_moved = True
    else:
        shutil.copytree(source, incoming)
    target_was_present = target.exists()
    try:
        if target_was_present:
            target.rename(backup)
        incoming.rename(target)
    except Exception:
        if target_was_present and backup.exists() and not target.exists():
            backup.rename(target)
        if incoming.exists():
            if source_moved and not source.exists():
                incoming.rename(source)
            else:
                shutil.rmtree(incoming)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    _refresh_layer_completion_inventory(target)
    if not source_moved:
        remove_staging_directory(source, root)

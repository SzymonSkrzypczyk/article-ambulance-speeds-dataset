"""Build a minimal, checksummed public dataset from a completed Gold layer.

The output contains exactly six Parquet tables, README.md, LICENCE,
citations.cff, and checksums.sha256. No input directory is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

PARQUET_FILES = (
    "krakow_roads_base.parquet",
    "speeds_by_season.parquet",
    "speeds_by_month.parquet",
    "speeds_by_day_of_week.parquet",
    "speeds_by_time_of_day.parquet",
    "krakow_ambulance_speeds_2021_2023_flat.parquet",
)
DOCUMENT_FILES = ("README.md", "LICENCE", "citations.cff")
CHECKSUM_FILE = "checksums.sha256"
ALL_FILES = frozenset((*PARQUET_FILES, *DOCUMENT_FILES, CHECKSUM_FILE))
SCOPE_OFFICIAL = "OFFICIAL_KRAKOW_BOUNDING_BOX"
SCOPE_EXTENDED = "EXTENDED_PROCESSING_AREA"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("title", "authors", "licence_text"):
        if not data.get(key):
            raise ValueError(f"Metadata must contain a nonempty {key!r}")
    if not isinstance(data["authors"], list) or not data["authors"]:
        raise ValueError("Metadata authors must be a nonempty list")
    for author in data["authors"]:
        if not all(author.get(key) for key in ("given-names", "family-names")):
            raise ValueError("Every author needs given-names and family-names")
    serialized = json.dumps(data).upper()
    if any(marker in serialized for marker in
           ("[[", "TO COMPLETE", "TO CONFIRM", "REPLACE WITH")):
        raise ValueError("Resolve metadata placeholders before packaging")
    if data.get("doi") and not re.fullmatch(r"10\.\d{4,9}/\S+", data["doi"]):
        raise ValueError("doi must be a bare DOI, for example 10.5281/zenodo.1234567")
    return data


def gold_build(gold: Path) -> tuple[str, str]:
    sentinel = gold / "_KARETKI_COMPLETE.json"
    if not sentinel.is_file():
        raise ValueError(f"Gold completion sentinel is missing: {sentinel}")
    data = json.loads(sentinel.read_text(encoding="utf-8"))
    if (data.get("layer") != "gold" or not data.get("dataset_version")
            or not data.get("configuration_fingerprint")):
        raise ValueError("Invalid Gold completion sentinel")
    return str(data["dataset_version"]), str(data["configuration_fingerprint"])


def table_details(gold: Path) -> tuple[dict[str, int], dict[str, int], int]:
    rows: dict[str, int] = {}
    totals: dict[str, int] = {}
    scopes: dict[str, int] = {SCOPE_OFFICIAL: 0, SCOPE_EXTENDED: 0}
    for name in PARQUET_FILES:
        path = gold / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required regular Gold file is missing: {path}")
        table = pq.ParquetFile(path)
        rows[name] = table.metadata.num_rows
        if name == PARQUET_FILES[0]:
            continue
        if not {"n_samples", "spatial_scope"}.issubset(table.schema_arrow.names):
            raise ValueError(f"Missing count/scope columns: {path}")
        total = 0
        for batch in table.iter_batches(columns=["n_samples", "spatial_scope"]):
            counts, areas = batch.column(0), batch.column(1)
            valid_scope = pc.or_(pc.equal(areas, SCOPE_OFFICIAL),
                                 pc.equal(areas, SCOPE_EXTENDED))
            if (counts.null_count or areas.null_count
                    or pc.any(pc.less(counts, 1)).as_py()
                    or not pc.all(valid_scope).as_py()):
                raise ValueError(f"Invalid n_samples or spatial_scope in {name}")
            total += int(pc.sum(counts).as_py())
            if name == "speeds_by_time_of_day.parquet":
                for scope in scopes:
                    subset = pc.filter(counts, pc.equal(areas, scope))
                    scopes[scope] += int(pc.sum(subset).as_py() or 0)
        totals[name] = total
    if len(set(totals.values())) != 1:
        raise ValueError(f"Gold observation totals do not reconcile: {totals}")
    total = next(iter(totals.values()))
    if sum(scopes.values()) != total:
        raise ValueError("Scope totals do not reconcile to the Gold total")
    return rows, scopes, total


def write_readme(path: Path, rows: dict[str, int], scopes: dict[str, int],
                 total: int, version: str, fingerprint: str, title: str) -> None:
    template = Path(__file__).with_name("final_dataset_readme.md").read_text(encoding="utf-8")
    table = "\n".join(f"| `{name}` | {rows[name]:,} |" for name in PARQUET_FILES)
    values = {
        "TITLE": title,
        "VERSION": version,
        "FINGERPRINT": fingerprint,
        "TOTAL": f"{total:,}",
        "OFFICIAL": f"{scopes[SCOPE_OFFICIAL]:,}",
        "EXTENDED": f"{scopes[SCOPE_EXTENDED]:,}",
        "TABLE_ROWS": table,
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", template):
        raise ValueError("README has an unexpanded template field")
    path.write_text(template, encoding="utf-8", newline="\n")


def write_citation(path: Path, metadata: dict, version: str) -> None:
    cff = {
        "cff-version": "1.2.0",
        "message": "If you use this dataset, cite this version of the data.",
        "type": "dataset",
        "title": metadata["title"],
        "version": version,
        "authors": metadata["authors"],
    }
    for key in ("doi", "date-released", "url"):
        if metadata.get(key):
            cff[key] = metadata[key]
    path.write_text(yaml.safe_dump(cff, allow_unicode=True, sort_keys=False),
                    encoding="utf-8", newline="\n")


def verify_output(directory: Path) -> None:
    actual = {entry.name for entry in directory.iterdir()}
    if actual != ALL_FILES or any(not entry.is_file() or entry.is_symlink()
                                  for entry in directory.iterdir()):
        raise ValueError(f"Unexpected output contents: {actual ^ ALL_FILES}")
    lines = (directory / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines()
    expected_names = sorted(ALL_FILES - {CHECKSUM_FILE})
    if len(lines) != len(expected_names):
        raise ValueError("Checksum manifest has the wrong number of entries")
    for line, name in zip(lines, expected_names, strict=True):
        if line != f"{sha256(directory / name)}  {name}":
            raise ValueError(f"Checksum mismatch: {name}")


def build(gold: Path, output: Path, metadata_path: Path) -> None:
    gold = gold.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing output: {output}")
    if output.parent.resolve() == gold or gold in output.parents:
        raise ValueError("Output must not be inside the Gold input directory")
    metadata = load_metadata(metadata_path)
    source_version, fingerprint = gold_build(gold)
    version = str(metadata.get("version") or source_version)
    if (version != source_version
            and metadata.get("source_gold_version") != source_version):
        raise ValueError("A distinct package version requires the actual source_gold_version")
    rows, scopes, total = table_details(gold)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temp:
        stage = Path(temp)
        for name in PARQUET_FILES:
            shutil.copyfile(gold / name, stage / name)
            if sha256(gold / name) != sha256(stage / name):
                raise ValueError(f"Copied Parquet differs from Gold input: {name}")
        write_readme(stage / "README.md", rows, scopes, total, version,
                     fingerprint, metadata["title"])
        (stage / "LICENCE").write_text(metadata["licence_text"].rstrip() + "\n",
                                       encoding="utf-8", newline="\n")
        write_citation(stage / "citations.cff", metadata, version)
        (stage / CHECKSUM_FILE).write_text("".join(
            f"{sha256(stage / name)}  {name}\n"
            for name in sorted(ALL_FILES - {CHECKSUM_FILE})
        ), encoding="utf-8", newline="\n")
        verify_output(stage)
        stage.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=Path("/Volumes/dysk/gold"))
    parser.add_argument("--output", type=Path, required=True,
                        help="New, nonexistent directory for the ten-file dataset")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Approved JSON title, authors, and component licence text")
    args = parser.parse_args()
    build(args.gold_dir, args.output, args.metadata)
    print(f"Created {args.output} with exactly {len(ALL_FILES)} files")


if __name__ == "__main__":
    main()

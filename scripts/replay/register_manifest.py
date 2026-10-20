"""Register partition manifest artifacts with tracker-friendly metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Sequence,
)

from tnfr._exact_time import finite_represented_real
from tnfr.config.parsing import parse_bool
from tnfr.metrics.common import finite_mean
from tnfr.utils.io import json_dumps, json_loads, safe_write

STRUCTURAL_KEYS = ("phi_s", "phase_gradient", "phase_curvature", "coherence_length")


@dataclass(frozen=True)
class PartitionFileDigest:
    count: int
    sha256: str
    sample: List[str]
    source: str


def _load_json(path: Path) -> MutableMapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    payload = json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest JSON must be an object: {path}")
    return payload


def _resolve_path(base: Path, candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = (base / candidate).resolve()
    return path


def _iter_archive_paths(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            candidate = line.strip()
            if candidate:
                yield candidate


def _partition_paths(
    manifest: Mapping[str, Any], *, inline: bool, archive_path: Path | None
) -> Iterator[str]:
    if inline:
        entries = manifest.get("partition_files", [])
        if not isinstance(entries, list):
            raise ValueError("partition_files must be a list")
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("partition_files must contain nonempty strings")
            yield entry
        return
    if archive_path is None:
        raise FileNotFoundError("Archive path required for non-inline partition lists")
    yield from _iter_archive_paths(archive_path)


def _digest_partition_paths(
    paths: Iterable[str], *, sample_size: int
) -> PartitionFileDigest:
    digest = hashlib.sha256()
    sample: List[str] = []
    count = 0
    for path in paths:
        encoded = path.encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
        count += 1
        if len(sample) < sample_size:
            sample.append(path)
    return PartitionFileDigest(
        count=count, sha256=digest.hexdigest(), sample=sample, source="inline"
    )


def _structural_ranges(
    entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, float]]:
    ranges: Dict[str, Mapping[str, float]] = {}
    for key in STRUCTURAL_KEYS:
        values: List[float] = []
        for entry in entries:
            telemetry = entry.get("telemetry", {})
            if telemetry is None:
                continue
            if not isinstance(telemetry, Mapping):
                raise ValueError("entry telemetry must be an object or null")
            value = telemetry.get(key)
            if value is not None:
                values.append(finite_represented_real(value, key)[0])
        if values:
            ranges[key] = {
                "min": min(values),
                "max": max(values),
                "avg": finite_mean(values, name=key),
                "count": float(len(values)),
            }
    return ranges


def register_manifest(
    *,
    experiment_id: str,
    summary_path: Path,
    manifest_path: Path | None = None,
    archive_path: Path | None = None,
    sample_size: int = 5,
) -> Dict[str, Any]:
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    summary_path = summary_path.resolve()
    summary = _load_json(summary_path)

    base_dir = summary_path.parent
    file_index = summary.get("file_index", {})
    if not isinstance(file_index, dict):
        raise ValueError("file_index must be an object")
    default_manifest = file_index.get("manifest")
    manifest_path = (
        manifest_path
        or _resolve_path(base_dir, default_manifest)
        or summary_path.with_name("_manifest.json")
    ).resolve()
    manifest = _load_json(manifest_path)

    raw_inline = file_index.get("inline", bool(manifest.get("partition_files")))
    if not isinstance(raw_inline, (bool, str)):
        raise ValueError(
            "file_index.inline must be Boolean or an explicit Boolean string"
        )
    inline = parse_bool(raw_inline)
    archive_candidate = archive_path or _resolve_path(
        base_dir, file_index.get("archive")
    )
    partition_iter = _partition_paths(
        manifest, inline=inline, archive_path=archive_candidate
    )
    digest = _digest_partition_paths(partition_iter, sample_size=sample_size)
    digest_source = "inline" if inline else "archive"
    digest = PartitionFileDigest(
        count=digest.count,
        sha256=digest.sha256,
        sample=digest.sample,
        source=digest_source,
    )

    entries = manifest.get("entries", [])
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) for entry in entries
    ):
        raise ValueError("manifest entries must be a list of objects")
    structural_ranges = _structural_ranges(entries)

    record = {
        "experiment_id": experiment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "archive_path": str(archive_candidate) if archive_candidate else None,
        "partition_count": summary.get("partition_count"),
        "candidate_stats": summary.get("candidate_stats"),
        "node_stats": summary.get("node_stats"),
        "boundary_stats": summary.get("boundary_stats"),
        "telemetry_keys": summary.get("telemetry_keys", []),
        "file_index": {
            "inline": inline,
            "threshold": file_index.get("threshold"),
            "manifest": file_index.get("manifest"),
            "manifest_absolute": str(manifest_path),
            "archive": file_index.get("archive"),
            "archive_absolute": str(archive_candidate) if archive_candidate else None,
        },
        "partition_files": {
            "count": digest.count,
            "sha256": digest.sha256,
            "sample": digest.sample,
            "source": digest.source,
        },
        "structural_fields": structural_ranges,
        "manifest_entries": len(entries),
    }
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register manifest summary for replay automation"
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment identifier used by the tracker",
    )
    parser.add_argument(
        "--summary", required=True, type=Path, help="Path to _manifest_summary.json"
    )
    parser.add_argument("--manifest", type=Path, help="Optional path to _manifest.json")
    parser.add_argument(
        "--archive", type=Path, help="Optional override for _partition_files.txt.gz"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to write JSON payload"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Sample size for partition file listing",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    args = parser.parse_args(argv)

    record = register_manifest(
        experiment_id=args.experiment_id,
        summary_path=args.summary,
        manifest_path=args.manifest,
        archive_path=args.archive,
        sample_size=args.sample_size,
    )

    payload = json_dumps(record, indent=2 if args.pretty else None, allow_nan=False)
    if args.output:
        sources = (
            record["summary_path"],
            record["manifest_path"],
            record["archive_path"],
        )
        if any(
            source and Path(source).resolve() == args.output.resolve()
            for source in sources
        ):
            raise ValueError("registration output cannot replace a source artifact")
        safe_write(
            args.output,
            lambda stream: stream.write(payload + ("\n" if args.pretty else "")),
        )
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

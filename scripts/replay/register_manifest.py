"""Register partition manifest artifacts with tracker-friendly metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _partition_paths(manifest: Mapping[str, Any], *, inline: bool, archive_path: Path | None) -> Iterator[str]:
    if inline:
        for entry in manifest.get("partition_files", []) or []:
            if isinstance(entry, str) and entry:
                yield entry
        return
    if archive_path is None:
        raise FileNotFoundError("Archive path required for non-inline partition lists")
    yield from _iter_archive_paths(archive_path)


def _digest_partition_paths(paths: Iterable[str], *, sample_size: int) -> PartitionFileDigest:
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
    return PartitionFileDigest(count=count, sha256=digest.hexdigest(), sample=sample, source="inline")


def _structural_ranges(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, float]]:
    ranges: Dict[str, Mapping[str, float]] = {}
    for key in STRUCTURAL_KEYS:
        values: List[float] = []
        for entry in entries:
            telemetry = entry.get("telemetry", {})
            if not isinstance(telemetry, Mapping):
                continue
            value = telemetry.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            ranges[key] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
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
    summary_path = summary_path.resolve()
    summary = _load_json(summary_path)

    base_dir = summary_path.parent
    file_index = summary.get("file_index", {})
    default_manifest = file_index.get("manifest")
    manifest_path = (manifest_path or _resolve_path(base_dir, default_manifest) or summary_path.with_name("_manifest.json")).resolve()
    manifest = _load_json(manifest_path)

    inline = bool(file_index.get("inline", bool(manifest.get("partition_files"))))
    archive_candidate = archive_path or _resolve_path(base_dir, file_index.get("archive"))
    partition_iter = _partition_paths(manifest, inline=inline, archive_path=archive_candidate)
    digest = _digest_partition_paths(partition_iter, sample_size=sample_size)
    digest_source = "inline" if inline else "archive"
    digest = PartitionFileDigest(count=digest.count, sha256=digest.sha256, sample=digest.sample, source=digest_source)

    structural_ranges = _structural_ranges(manifest.get("entries", []))

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
        "manifest_entries": len(manifest.get("entries", [])),
    }
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register manifest summary for replay automation")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier used by the tracker")
    parser.add_argument("--summary", required=True, type=Path, help="Path to _manifest_summary.json")
    parser.add_argument("--manifest", type=Path, help="Optional path to _manifest.json")
    parser.add_argument("--archive", type=Path, help="Optional override for _partition_files.txt.gz")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON payload")
    parser.add_argument("--sample-size", type=int, default=5, help="Sample size for partition file listing")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)

    record = register_manifest(
        experiment_id=args.experiment_id,
        summary_path=args.summary,
        manifest_path=args.manifest,
        archive_path=args.archive,
        sample_size=max(1, args.sample_size),
    )

    payload = json.dumps(record, indent=2 if args.pretty else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + ("\n" if args.pretty else ""), encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

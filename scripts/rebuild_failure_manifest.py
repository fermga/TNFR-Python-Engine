"""Rebuild the failure manifest from telemetry artifacts.

This utility scans the telemetry artifact directory, consolidates entries per
`n`, and writes a refreshed manifest JSON with one record per node.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _coerce_bottlenecks(raw: Any) -> list[str]:
    """Return a list of bottleneck codes from the artifact payload."""

    if not raw:
        return []
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        result: list[str] = []
        for entry in raw:
            if isinstance(entry, dict):
                code = entry.get("code")
                if code:
                    result.append(str(code))
            elif isinstance(entry, str):
                result.append(entry)
        return result
    return [str(raw)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        default="results/failure_telemetry/full_run_gpu/2025-12-01",
        help="Directory containing failure_*.json telemetry artifacts",
    )
    parser.add_argument(
        "--manifest",
        default="results/failure_telemetry/full_run_gpu/failure_manifest.json",
        help="Path to write the rebuilt manifest",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=4000,
        help="Required number of unique n entries before writing the manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    manifest_path = Path(args.manifest)

    if not artifacts_dir.exists():
        print(f"Artifacts directory does not exist: {artifacts_dir}", file=sys.stderr)
        return 1

    repo_root = Path.cwd()
    records_by_n: dict[int, dict[str, Any]] = {}

    observed_ns: set[int] = set()
    for artifact_path in sorted(artifacts_dir.glob("failure_*.json")):
        try:
            payload = json.loads(artifact_path.read_text())
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping {artifact_path}: {exc}", file=sys.stderr)
            continue

        n_value = payload.get("n")
        if not isinstance(n_value, int):
            continue

        observed_ns.add(n_value)
        timestamp = payload.get("timestamp", 0)
        try:
            rel_path = artifact_path.relative_to(repo_root)
        except ValueError:
            rel_path = artifact_path

        record = {
            "run_id": payload.get("run_id"),
            "timestamp": timestamp,
            "n": n_value,
            "modulus": payload.get("modulus"),
            "failure_reason": payload.get("failure_reason"),
            "failure_stage": payload.get("failure_stage"),
            "bottlenecks": _coerce_bottlenecks(payload.get("bottlenecks")),
            "artifact_path": str(rel_path).replace("/", "\\"),
        }

        existing = records_by_n.get(n_value)
        if existing is None or timestamp >= existing.get("timestamp", 0):
            records_by_n[n_value] = record

    unique_count = len(records_by_n)
    if unique_count != args.expected_count:
        expected_ns = set(range(2, 2 + args.expected_count))
        missing = sorted(expected_ns - observed_ns)
        extra = sorted(observed_ns - expected_ns)
        print(
            "Unique entry count mismatch:",
            unique_count,
            "(expected)",
            args.expected_count,
            file=sys.stderr,
        )
        print(f"Missing n values (first 20): {missing[:20]}", file=sys.stderr)
        print(f"Unexpected n values (first 20): {extra[:20]}", file=sys.stderr)
        return 1

    ordered_records = [records_by_n[n] for n in sorted(records_by_n)]
    manifest_data = {"version": "1.0", "records": ordered_records}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
    print(
        f"Wrote manifest with {unique_count} records to {manifest_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rebuild a compact failure manifest from an explicitly selected artifact folder.

The version-1.0 manifest retains the greatest timestamp for each integer ``n``;
ties select the lexically later artifact filename. This is latest-per-n
compaction, not reconstruction of the producer's complete attempt history.
Only top-level ``failure_*.json`` artifacts are scanned. Every matching artifact
must be valid, and at least one record must survive, before output is replaced.
Artifact paths use forward slashes, relative to the working directory when
possible. Source artifacts cannot also be the destination manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


def _coerce_bottlenecks(raw: Any) -> list[str]:
    """Read producer signal objects or historical string codes without omission."""
    if not isinstance(raw, list):
        raise ValueError("bottlenecks must be a list of codes or signal objects")
    result = []
    for entry in raw:
        code = entry.get("code") if isinstance(entry, dict) else entry
        if not isinstance(code, str) or not code.strip():
            raise ValueError("each bottleneck must have a nonempty string code")
        result.append(code)
    return result


def _positive_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected count must be a positive integer"
        ) from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("expected count must be a positive integer")
    return count


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=Path,
        help="Folder containing the selected top-level failure_*.json artifacts",
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Destination compact manifest"
    )
    parser.add_argument(
        "--expected-count",
        type=_positive_count,
        help="Optional required number of unique n values (no consecutive-range assumption)",
    )
    return parser.parse_args(argv)


def _read_record(artifact_path: Path, working_directory: Path) -> dict[str, Any]:
    """Validate only fields copied into the producer-compatible manifest."""
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact must contain a JSON object")
    n_value, modulus = payload.get("n"), payload.get("modulus")
    if type(n_value) is not int or n_value < 2:
        raise ValueError("n must be an integer >= 2")
    if type(modulus) is not int or modulus <= 0:
        raise ValueError("modulus must be a positive integer")
    timestamp = payload.get("timestamp")
    if type(timestamp) not in (int, float) or (
        isinstance(timestamp, float) and not math.isfinite(timestamp)
    ):
        raise ValueError("timestamp must be a finite number")
    for field in ("run_id", "failure_reason", "failure_stage"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    try:
        retained_path = artifact_path.relative_to(working_directory)
    except ValueError:
        retained_path = artifact_path
    return {
        "run_id": payload["run_id"],
        "timestamp": timestamp,
        "n": n_value,
        "modulus": modulus,
        "failure_reason": payload["failure_reason"],
        "failure_stage": payload["failure_stage"],
        "bottlenecks": _coerce_bottlenecks(payload.get("bottlenecks")),
        "artifact_path": retained_path.as_posix(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    records_by_n: dict[int, dict[str, Any]] = {}
    try:
        if not artifacts_dir.is_dir():
            raise ValueError(f"artifacts directory does not exist: {artifacts_dir}")
        working_directory = Path.cwd().resolve()
        for artifact_path in sorted(artifacts_dir.glob("failure_*.json")):
            if artifact_path.resolve() == manifest_path:
                raise ValueError(
                    "destination manifest cannot also be a source artifact"
                )
            try:
                record = _read_record(artifact_path, working_directory)
            except (OSError, UnicodeError, ValueError) as exc:
                raise ValueError(f"invalid artifact {artifact_path}: {exc}") from exc
            n_value = record["n"]
            existing = records_by_n.get(n_value)
            if existing is None or record["timestamp"] >= existing["timestamp"]:
                records_by_n[n_value] = record
        unique_count = len(records_by_n)
        if not unique_count:
            raise ValueError("no failure artifacts found; refusing an empty manifest")
        if args.expected_count is not None and unique_count != args.expected_count:
            raise ValueError(
                f"unique entry count mismatch: {unique_count}; expected {args.expected_count}"
            )
        manifest_data = {
            "version": "1.0",
            "records": [records_by_n[n] for n in sorted(records_by_n)],
        }
        serialized = (
            json.dumps(manifest_data, ensure_ascii=False, allow_nan=False, indent=2)
            + "\n"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=manifest_path.parent,
                prefix=".failure-manifest-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(serialized)
            temporary_path.replace(manifest_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Cannot rebuild failure manifest: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote compact manifest with {unique_count} records to {manifest_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

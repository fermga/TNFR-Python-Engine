"""Validate TNFR self-optimization recommendations against targeted pytest suites."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD_ROOT = REPO_ROOT / "results" / "self_optimization"
DEFAULT_REPORT_PATH = REPO_ROOT / "results" / "self_optimization_validation.json"
OPERATION_TESTS: Dict[str, List[str]] = {
    "paley_partition": ["factorization-lab/tests/test_spectral_paley.py"],
}


@dataclass
class RecommendationRecord:
    path: Path
    metadata: Dict[str, Any]
    operation_type: str
    tests: List[str]
    status: str = "pending"
    exit_code: Optional[int] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload-root",
        type=Path,
        default=DEFAULT_PAYLOAD_ROOT,
        help="Directory containing self-optimization payloads (seed folders with *.json files)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to write validation summary JSON (default: results/self_optimization_validation.json)",
    )
    parser.add_argument(
        "--pytest-cmd",
        type=str,
        default=None,
        help="Override pytest command (default: python -m pytest)",
    )
    parser.add_argument(
        "--pytest-args",
        type=str,
        default="",
        help="Additional arguments appended to the pytest command",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 1 if any recommendation regresses",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging (summary JSON still printed)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    payload_root = args.payload_root
    payloads = _discover_payloads(payload_root)
    grouped = _group_by_tests(payloads)
    test_runs: List[Dict[str, Any]] = []

    if not args.quiet:
        print(f"[validation] discovered {len(payloads)} recommendation payload(s)")

    for tests_key, records in grouped.items():
        if not tests_key:
            for record in records:
                record.status = "pending"
            continue
        exit_code = _run_pytest(
            list(tests_key),
            pytest_cmd=args.pytest_cmd,
            pytest_args=args.pytest_args,
        )
        status = "validated" if exit_code == 0 else "regressed"
        for record in records:
            record.status = status
            record.exit_code = exit_code
        test_runs.append(
            {
                "tests": list(tests_key),
                "exit_code": exit_code,
                "status": status,
                "count": len(records),
            }
        )
        if not args.quiet:
            print(
                f"[validation] tests={' '.join(tests_key)} status={status} affected={len(records)}"
            )

    summary = _build_summary(payload_root, payloads, test_runs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"[validation] summary written to {args.report}")

    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if args.fail_on_regression and summary["status_counts"].get("regressed", 0) > 0:
        raise SystemExit(1)

    return summary


def _discover_payloads(root: Path) -> List[RecommendationRecord]:
    if not root.exists():
        return []
    payloads: List[RecommendationRecord] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".sha256"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            continue
        operation_type = str(metadata.get("operation_type") or data.get("operation_type") or "unknown")
        tests = list(OPERATION_TESTS.get(operation_type, []))
        payloads.append(RecommendationRecord(path=path, metadata=metadata, operation_type=operation_type, tests=tests))
    return payloads


def _group_by_tests(records: Sequence[RecommendationRecord]) -> Dict[Tuple[str, ...], List[RecommendationRecord]]:
    grouped: Dict[Tuple[str, ...], List[RecommendationRecord]] = {}
    for record in records:
        key = tuple(record.tests)
        grouped.setdefault(key, []).append(record)
    return grouped


def _run_pytest(
    tests: Sequence[str],
    *,
    pytest_cmd: Optional[str],
    pytest_args: str,
) -> int:
    base_cmd = shlex.split(pytest_cmd) if pytest_cmd else [sys.executable, "-m", "pytest"]
    extra_args = shlex.split(pytest_args) if pytest_args else []
    cmd = [*base_cmd, *extra_args, *tests]
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


def _build_summary(
    payload_root: Path,
    records: Sequence[RecommendationRecord],
    test_runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    status_counts = {"validated": 0, "regressed": 0, "pending": 0}
    results = []
    for record in records:
        status_counts.setdefault(record.status, 0)
        status_counts[record.status] += 1
        results.append(
            {
                "path": str(record.path),
                "operation_type": record.operation_type,
                "tests": record.tests,
                "status": record.status,
                "exit_code": record.exit_code,
                "metadata": record.metadata,
            }
        )
    summary = {
        "payload_root": str(payload_root),
        "total_recommendations": len(records),
        "status_counts": status_counts,
        "results": results,
        "test_runs": list(test_runs),
    }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

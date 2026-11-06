"""Run mapped regression suites for TNFR self-optimization operation types.

Payloads select suites; this runner does not apply recommendations or compare
their predicted effects. Legacy ``validated``/``regressed`` labels describe
the current code's suite exit status, not recommendation-specific evidence.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tnfr.sdk.utils import export_to_json
from tnfr.utils.io import json_dumps, json_loads

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD_ROOT = REPO_ROOT / "results" / "self_optimization"
DEFAULT_REPORT_PATH = REPO_ROOT / "results" / "self_optimization_validation.json"
# The historical arithmetic suite did not validate recommendation effects and
# is not an active regression owner. Unmapped operations remain pending.
OPERATION_TESTS: Dict[str, List[str]] = {}


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
        help="Exit with status 1 if a mapped regression suite fails",
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
    if any(record.path.resolve() == args.report.resolve() for record in payloads):
        raise ValueError("validation report cannot replace a recommendation payload")
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
    export_to_json(summary, args.report)

    if not args.quiet:
        print(f"[validation] summary written to {args.report}")

    sys.stdout.write(json_dumps(summary, indent=2, allow_nan=False) + "\n")

    if args.fail_on_regression and summary["status_counts"].get("regressed", 0) > 0:
        raise SystemExit(1)

    return summary


def _discover_payloads(root: Path) -> List[RecommendationRecord]:
    if not root.exists():
        return []
    payloads: List[RecommendationRecord] = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json_loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f"invalid payload JSON {path}: {exc}") from exc
        if not isinstance(data, dict):
            continue
        if "metadata" not in data:
            continue
        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"recommendation metadata must be an object: {path}")
        operation_type = metadata.get(
            "operation_type", data.get("operation_type", "unknown")
        )
        if not isinstance(operation_type, str) or not operation_type.strip():
            raise ValueError(f"operation_type must be a nonempty string: {path}")
        tests = list(OPERATION_TESTS.get(operation_type, []))
        payloads.append(
            RecommendationRecord(
                path=path, metadata=metadata, operation_type=operation_type, tests=tests
            )
        )
    return payloads


def _group_by_tests(
    records: Sequence[RecommendationRecord],
) -> Dict[Tuple[str, ...], List[RecommendationRecord]]:
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
    base_cmd = (
        shlex.split(pytest_cmd) if pytest_cmd else [sys.executable, "-m", "pytest"]
    )
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
        "validation_scope": "current_code_regression_suites",
        "recommendations_applied": False,
        "recommendation_effects_evaluated": False,
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

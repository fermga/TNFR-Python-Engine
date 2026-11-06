"""Evaluate the frozen static handoff once and preserve its response and checks.

Run from the repository root with the declared environment. This corrective retained
producer is immutable after evaluation; byte checks are not authentication.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tnfr.research.artifact_io import (  # noqa: E402
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
    write_json_once,
)
from tnfr.utils.io import json_loads  # noqa: E402


def _read(name):
    return json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=2**20))


def _verify():
    freeze = _read("freeze.json")
    if freeze["evaluation_status_at_freeze"] != "not_evaluated":
        raise ValueError("the original freeze must precede evaluation")
    for expected in freeze["prior_artifacts"]:
        if (
            file_receipt(EVIDENCE.parent / expected["path"], label=expected["path"])
            != expected
        ):
            raise ValueError(f"prior frozen evidence changed: {expected['path']}")
    for expected in freeze["artifacts"]:
        if (
            file_receipt(EVIDENCE / expected["path"], label=expected["path"])
            != expected
        ):
            raise ValueError(f"frozen artifact changed: {expected['path']}")
    members = _read("source-manifest.json")["files"]
    verify_archive_members(
        EVIDENCE / "source.zip",
        {item["path"]: item["sha256"] for item in members},
        max_bytes=32 * 1024**2,
    )
    for expected in members:
        if file_receipt(ROOT / expected["path"], label=expected["path"]) != expected:
            raise ValueError(f"working source differs from freeze: {expected['path']}")
    if any(
        (EVIDENCE / name).exists()
        for name in ("attempt.json", "response.json", "report.json", "failure.json")
    ):
        raise FileExistsError(
            "retained attempt or response prevents another evaluation"
        )


def main():
    _verify()
    write_json_once(
        EVIDENCE / "attempt.json",
        {
            "schema": "tnfr.collective-pressure-handoff-attempt.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": file_receipt(EVIDENCE / "freeze.json")["sha256"],
            "maximum_evaluations": 1,
            "numeric_threads": 1,
            "python": sys.version,
            "executable": sys.executable,
            "versions": {
                name: version(name)
                for name in ("numpy", "networkx", "mpmath", "pytest")
            },
        },
    )
    try:
        sys.path.insert(0, str(ROOT / "tests" / "operators"))
        study = importlib.import_module("test_collective_pressure_handoff")
        result = study.evaluate_handoff()
        write_json_once(EVIDENCE / "response.json", study.evidence_export(result))
        checks = []
        for name in sorted(vars(study)):
            function = getattr(study, name)
            if not name.startswith("test_") or not callable(function):
                continue
            try:
                # Local checks consume the same captured response. They never
                # invoke a second source or word to obtain a preferred result.
                if function.__code__.co_argcount:
                    function(result)
                else:
                    function()
            except Exception:
                checks.append(
                    {
                        "name": name,
                        "passed": False,
                        "diagnostic": traceback.format_exc(),
                    }
                )
            else:
                checks.append({"name": name, "passed": True})
        report = {
            "schema": "tnfr.collective-pressure-handoff-report.v1",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": file_receipt(EVIDENCE / "freeze.json")["sha256"],
            "response": file_receipt(EVIDENCE / "response.json", label="response.json"),
            "execution_status": result["status"],
            "checks": checks,
            "all_checks_passed": bool(checks) and all(row["passed"] for row in checks),
            "scope": "One represented static word; conditional law and supplied events",
        }
        write_json_once(EVIDENCE / "report.json", report)
        print(
            f"Execution: {result['status']}; checks passed: "
            f"{sum(row['passed'] for row in checks)}/{len(checks)}"
        )
        return 0 if report["all_checks_passed"] else 1
    except Exception:
        write_json_once(
            EVIDENCE / "failure.json",
            {
                "schema": "tnfr.collective-pressure-handoff-failure.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "diagnostic": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

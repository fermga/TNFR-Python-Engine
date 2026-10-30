"""Evaluate the frozen local-capacity history protocol once, preserving failures."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tnfr.research.artifact_io import (  # noqa: E402
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
    write_json_once,
)
from tnfr.research.local_capacity_history import evaluate_local_history  # noqa: E402
from tnfr.research.retained_graph_state import (  # noqa: E402
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_loads  # noqa: E402


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=2**24))
    )


def verify_frozen_inputs():
    freeze = _read("freeze.json")
    if freeze["evaluation_status_at_freeze"] != "not_evaluated":
        raise ValueError("the protocol must precede the response")
    for row in freeze["artifacts"]:
        if file_receipt(EVIDENCE / row["path"], label=row["path"]) != row:
            raise ValueError(f"frozen input changed: {row['path']}")
    for row in freeze["prior_artifacts"]:
        if file_receipt(ROOT / row["path"], label=row["path"]) != row:
            raise ValueError(f"retained evidence changed: {row['path']}")
    members = _read("source-manifest.json")["files"]
    verify_archive_members(
        EVIDENCE / "source.zip",
        {row["path"]: row["sha256"] for row in members},
        max_bytes=32 * 1024**2,
    )
    for row in members:
        if file_receipt(ROOT / row["path"], label=row["path"]) != row:
            raise ValueError(f"working source differs from freeze: {row['path']}")
    if any(
        (EVIDENCE / name).exists()
        for name in ("attempt.json", "response.json", "report.json", "failure.json")
    ):
        raise FileExistsError("a retained attempt prevents another response evaluation")
    return freeze


def main():
    verify_frozen_inputs()
    write_json_once(
        EVIDENCE / "attempt.json",
        {
            "schema": "tnfr.local-capacity-history-attempt.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": file_receipt(EVIDENCE / "freeze.json")["sha256"],
            "python": sys.version,
            "executable": sys.executable,
            "versions": {
                name: version(name) for name in ("numpy", "networkx", "mpmath")
            },
            "maximum_evaluations": 1,
        },
    )
    try:
        graph = restore_recorded_graph_state(_read("source.json")["graph"]).graph
        result = evaluate_local_history(graph)
        write_json_once(EVIDENCE / "response.json", result)
        certificate = result["certificate"]
        report = {
            "schema": "tnfr.local-capacity-history-verdict.v1",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "response": file_receipt(EVIDENCE / "response.json", label="response.json"),
            "checks": result["checks"],
            "all_checks_passed": result["all_checks_passed"],
            "observed_acceleration": result["observation"]["value"],
            "certified_acceleration_lower": certificate["trigger_lower_bound"],
            "parent_endpoint_error_bounds": certificate["parent_error_bounds"],
            "scope": result["scope"],
        }
        write_json_once(EVIDENCE / "report.json", report)
        print(f"Checks: {sum(result['checks'].values())}/{len(result['checks'])}")
        print(f"Observed acceleration: {report['observed_acceleration']}")
        print(f"Certified lower: {float(report['certified_acceleration_lower'])}")
        print(
            f"Maximum parent error: {float(max(report['parent_endpoint_error_bounds']))}"
        )
        return 0 if result["all_checks_passed"] else 1
    except Exception:
        write_json_once(
            EVIDENCE / "failure.json",
            {
                "schema": "tnfr.local-capacity-history-failure.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "diagnostic": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

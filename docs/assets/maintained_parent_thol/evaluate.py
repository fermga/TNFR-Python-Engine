"""Evaluate one frozen maintained-parent THOL reset and retain its outcome."""

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
    validate_artifact_path,
    verify_archive_members,
    write_json_once,
)
from tnfr.research.local_capacity_history import encode_result  # noqa: E402
from tnfr.research.maintained_parent_thol import load_retained_parent  # noqa: E402
from tnfr.research.maintained_parent_thol_execution import (
    evaluate_thol_event,
)  # noqa: E402
from tnfr.research.maintained_parent_thol_prediction import (
    predict_thol_reset,
)  # noqa: E402
from tnfr.utils.io import json_loads  # noqa: E402


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=2**25))
    )


def _verify_rows(base, rows):
    for row in rows:
        name = validate_artifact_path(row["path"])
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError("artifact resolves outside its declared root")
        if file_receipt(path, label=name) != row:
            raise ValueError(f"frozen content changed: {name}")


def verify_frozen_inputs():
    if any(
        (EVIDENCE / name).exists()
        for name in ("attempt.json", "response.json", "report.json", "failure.json")
    ):
        raise FileExistsError("a retained attempt prevents another event evaluation")
    freeze = _read("freeze.json")
    if freeze["evaluation_status_at_freeze"] != "not_evaluated":
        raise ValueError("the protocol and prediction must precede the event")
    _verify_rows(EVIDENCE, freeze["artifacts"])
    _verify_rows(ROOT, freeze["prior_artifacts"])
    members = _read("source-manifest.json")["files"]
    verify_archive_members(
        EVIDENCE / "source.zip",
        {row["path"]: row["sha256"] for row in members},
        max_bytes=32 * 1024**2,
    )
    _verify_rows(ROOT, members)


def main():
    verify_frozen_inputs()
    write_json_once(
        EVIDENCE / "attempt.json",
        {
            "schema": "tnfr.maintained-parent-thol-attempt.v1",
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
        admitted = load_retained_parent(
            ROOT / "docs/assets/local_capacity_history", repository_root=ROOT
        )
        prediction = predict_thol_reset(admitted.graph)
        frozen_prediction = _read("prediction.json")["prediction"]
        if encode_result(prediction) != frozen_prediction:
            raise ValueError(
                "recomputed prospective prediction differs from its freeze"
            )
        result = evaluate_thol_event(admitted, prediction)
        write_json_once(EVIDENCE / "response.json", result)
        report = {
            "schema": "tnfr.maintained-parent-thol-verdict.v1",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "response": file_receipt(EVIDENCE / "response.json", label="response.json"),
            "checks": result["checks"],
            "all_checks_passed": result["all_checks_passed"],
            "edge_storage_jump": result["edge_storage_jump"],
            "parent_raw_pressure": result["parent_raw_pressure"],
            "scope": result["scope"],
        }
        write_json_once(EVIDENCE / "report.json", report)
        print(f"Checks: {sum(result['checks'].values())}/{len(result['checks'])}")
        print(f"Raw parent pressure: {float(result['parent_raw_pressure'])}")
        print(f"Exact edge storage jump: {result['edge_storage_jump']}")
        return 0 if result["all_checks_passed"] else 1
    except Exception:
        write_json_once(
            EVIDENCE / "failure.json",
            {
                "schema": "tnfr.maintained-parent-thol-failure.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "diagnostic": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

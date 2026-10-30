"""Post-evaluation theorem admission of immutable upper-corner endpoints.

No time evolution or amended prediction occurs here. This separate audit
checks the exact represented final states with a subsequently derived theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import networkx as nx
import numpy as np

from tnfr.dynamics import relational as owner
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_capture import certify_relational_sector_capture
from tnfr.sdk.relational_reports import relational_report_to_dict

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "docs/assets/relational_capture_response"
EXPECTED_SHA256 = {
    "result.prediction.json": "4045ab698013f2bb071c5083a99e8aea760c7ce42e568e1be5a230af9629db7c",
    "result.json": "7085c782f0b85c465c530d2e84ff2990e71dfcc89d950f1f993025a952310ed9",
}


def audit_endpoints():
    """Bind the retained records and assess their endpoints without stepping."""
    if Path(owner.__file__).resolve() != ROOT / "src/tnfr/dynamics/relational.py":
        raise ValueError("this audit requires workspace source; set PYTHONPATH=src")
    raw = {}
    for name, digest in EXPECTED_SHA256.items():
        data = (RECORDS / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"immutable response changed: {name}")
        raw[name] = json.loads(data)
    prediction, response = raw["result.prediction.json"], raw["result.json"]
    if prediction != response["prediction"]:
        raise ValueError("the retained response is not bound to its prediction")
    model = RelationalExchangeModel(**prediction["model"])
    endpoints = {}
    admitted = []
    for count, trace in response["traces"].items():
        frame = trace["checkpoints"][trace["last_admitted_checkpoint"]]
        field = frame["reflected"]["report"]["field"]
        graph = nx.Graph()
        graph.add_nodes_from(field["nodes"])
        graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in field["edges"])
        graph.graph.update(
            GAMMA={"type": "none"}, vectorized_dnfr=True, _t=frame["time"]
        )
        for index, node in enumerate(field["nodes"]):
            graph.nodes[node].update(
                EPI=field["epi"][index],
                theta=field["phase"][index],
                nu_f=field["capacity"][index],
            )
        certificate = certify_relational_sector_capture(
            graph, model=model, cycles=prediction["cycles"], target_sector=1
        )
        endpoints[count] = {
            "time": frame["time"],
            "original_status": trace["status"],
            "original_capture_admitted": frame["capture_admitted"],
            "sector_capture_admitted": certificate.admitted,
            "certificate": relational_report_to_dict(certificate),
        }
        admitted.append(certificate.admitted)
    paths = sorted((ROOT / "src/tnfr").rglob("*.py")) + [Path(__file__).resolve()]
    return {
        "analysis": "post_evaluation_full_state_acute_sector_barrier_v1",
        "original_records_sha256": dict(EXPECTED_SHA256),
        "original_finite_prediction_passed": response["finite_prediction_passed"],
        "endpoints": endpoints,
        "all_endpoints_admitted": all(admitted),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "temporal_execution": "none",
        },
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "scope": "ideal_continuation_from_actual_retained_endpoints; no_retroactive_prediction_success_or_original_IVP_error_tube; no_state_projection",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to replace a retained endpoint audit")
    report = audit_endpoints()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "all_endpoints_admitted": report["all_endpoints_admitted"],
                "original_finite_prediction_passed": report[
                    "original_finite_prediction_passed"
                ],
            }
        )
    )
    return 0 if report["all_endpoints_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Review one retained phase-reset certificate without replaying its producer.

Protocol and implementation are frozen before this one corrective evaluation.
The original response, all previous attempts and the original 8/9 verdict are
read-only inputs. The existing retained-state reader supplies strict primitive
admission and independent later work/identity and OZ projection checks.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as Q
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import mpmath as mp  # noqa: E402

from tnfr.mathematics._rational_interval import (  # noqa: E402
    INTERVAL_BITS,
    INTERVAL_METHOD,
    I,
)
from tnfr.research.artifact_io import (  # noqa: E402
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
    write_json_once,
)
from tnfr.research.phase_reset_certificate import bound_phase_reset  # noqa: E402
from tnfr.utils.io import json_loads  # noqa: E402


def read_record(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=32 * 1024**2))
    )


def _encoded(value):
    if isinstance(value, I):
        return {"lo": value.lo, "hi": value.hi}
    if isinstance(value, dict):
        return {key: _encoded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_encoded(item) for item in value)
    return value


def retained_reader():
    """Load only the existing detached reader, never an operator producer."""
    path = ROOT / "tests/research/test_collective_pressure_handoff_evidence.py"
    spec = importlib.util.spec_from_file_location("retained_phase_review_input", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate():
    """Re-admit saved primitives, certify phases, and check later claims separately."""
    protocol = read_record("protocol.json")
    if INTERVAL_BITS != protocol["interval_bits"]:
        raise ValueError("interval precision differs from the frozen protocol")
    reader = retained_reader()
    response = reader._read("response.json")
    captures = response["captures"]
    old = reader._rebuild(captures["initial"])
    new = reader._rebuild(captures["IL_raw"])
    if old.nodes != tuple(range(20)) or old.edges != new.edges:
        raise ValueError("retained node order or support differs")
    if old.epi != new.epi or old.capacity != new.capacity:
        raise ValueError("IL changed an excluded primitive coordinate")
    if any(old.phase[i] != new.phase[i] for i in range(2, 20)):
        raise ValueError("an untargeted phase changed")
    neighbors = tuple(
        tuple(
            right if left == i else left
            for left, right in old.edges
            if i in (left, right)
        )
        for i in old.nodes
    )
    coefficient = exact_record(response["settings"]["phase_coefficient"])
    if coefficient != Q(float(0.3)):
        raise ValueError(
            "retained coefficient differs from the frozen represented value"
        )
    certificates = bound_phase_reset(
        old.phase,
        new.phase,
        neighbors,
        tuple(protocol["targets"]),
        coefficient=coefficient,
        budget=exact_record(protocol["absolute_residual_allowance"]),
    )
    oracles = []
    context = mp.mp.clone()
    context.dps = protocol["cross_check_decimal_digits"]

    def number(value):
        return context.mpf(value.numerator) / value.denominator

    for certificate in certificates:
        i = certificate.target
        resultant = sum(
            context.exp(context.j * number(old.phase[j] - old.phase[i]))
            for j in neighbors[i]
        )
        ideal = number(coefficient) * context.arg(resultant)
        actual = context.arg(
            context.exp(context.j * number(new.phase[i] - old.phase[i]))
        )
        point = context.nstr(actual - ideal, context.dps)
        oracles.append(
            {
                "target": i,
                "decimal_digits": context.dps,
                "defect_decimal": point,
                "inside_rigorous_box": certificate.defect.contains(Q(point)),
                "role": "Independent point cross-check, not a certificate",
            }
        )
    checks = []
    records = {"response.json": response}
    # Reuse the detached controls; these never invoke a word or create state.
    fields = {name: reader._rebuild(capture) for name, capture in captures.items()}
    for name in (
        "test_il_work_and_identity_follow_from_all_retained_edges",
        "test_oz_projection_and_all_candidate_products_retain_their_separate_roles",
    ):
        try:
            getattr(reader, name)(records, fields)
        except Exception:
            checks.append(
                {"name": name, "passed": False, "diagnostic": traceback.format_exc()}
            )
        else:
            checks.append({"name": name, "passed": True})
    measured = response["IL"]
    budget = certificates[0].budget
    # The original compound assertion stopped at its first phase check. Retain
    # the old, already computed later boxes as a separate consistency check;
    # the independently rebuilt controls above carry their primitive evidence.
    remaining = {
        "shared_edge_work_identity": reader._interval(
            measured["work_identity_residual"]
        ).abs_max
        <= budget,
        "represented_ideal_work_bridge": reader._interval(
            measured["represented_work_defect"]
        ).abs_max
        <= 14 * budget,
        "actual_ideal_phase_mean_bridge": (
            reader._interval(measured["actual_mean_phase_jump"])
            - reader._interval(measured["mean_phase_jump"])
        ).abs_max
        <= budget,
        "strict_negative_work": reader._interval(measured["work"]).hi < 0,
        "retained_global_rho": exact_record(measured["global_rho"]) == 104,
        "strict_joint_identity": reader._interval(
            measured["geometry"]["joint_norm_squared"]
        ).hi
        < Q(1, 256),
        "strict_storage_margin": reader._interval(
            measured["geometry"]["storage_excess"]
        ).hi
        < Q(1, 6400),
    }
    return {
        "schema": "tnfr.collective-pressure-phase-review-response.v1",
        "interval_bits": INTERVAL_BITS,
        "interval_method": INTERVAL_METHOD,
        "coefficient": coefficient,
        "source_captures": ("initial", "IL_raw"),
        "node_order": old.nodes,
        "edges": old.edges,
        "neighbors": neighbors,
        "before_phase": old.phase,
        "after_phase": new.phase,
        "certificates": tuple(_encoded(asdict(item)) for item in certificates),
        "cross_checks": oracles,
        "independent_retained_state_checks": checks,
        "later_original_compound_boxes": remaining,
        "original_report_reclassified": False,
        "operator_words_executed": 0,
        "numerical_trajectory_steps": 0,
    }


def verify_freeze():
    freeze = read_record("freeze.json")
    if freeze["evaluation_status_at_freeze"] != "not_evaluated":
        raise ValueError("the corrective review freeze must precede its evaluation")
    for group, directory in (
        ("prior_artifacts", EVIDENCE.parent),
        ("artifacts", EVIDENCE),
    ):
        for expected in freeze[group]:
            if (
                file_receipt(directory / expected["path"], label=expected["path"])
                != expected
            ):
                raise ValueError(f"frozen artifact changed: {expected['path']}")
    members = read_record("source-manifest.json")["files"]
    verify_archive_members(
        EVIDENCE / "source.zip", {item["path"]: item["sha256"] for item in members}
    )
    for expected in members:
        if file_receipt(ROOT / expected["path"], label=expected["path"]) != expected:
            raise ValueError(f"working source differs from freeze: {expected['path']}")
    if any(
        (EVIDENCE / name).exists()
        for name in ("attempt.json", "response.json", "report.json", "failure.json")
    ):
        raise FileExistsError("a retained corrective evaluation prevents another run")


def main():
    verify_freeze()
    write_json_once(
        EVIDENCE / "attempt.json",
        {
            "schema": "tnfr.collective-pressure-phase-review-attempt.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": file_receipt(EVIDENCE / "freeze.json")["sha256"],
            "python": sys.version,
            "executable": sys.executable,
            "versions": {
                name: version(name) for name in ("mpmath", "networkx", "pytest")
            },
            "maximum_evaluations": 1,
        },
    )
    try:
        result = calculate()
        write_json_once(EVIDENCE / "response.json", result)
        phase_pass = all(
            item["classification"] == "certified_within_budget"
            for item in result["certificates"]
        )
        auxiliary_pass = all(
            item["passed"] for item in result["independent_retained_state_checks"]
        )
        boxes_pass = all(result["later_original_compound_boxes"].values())
        oracles_pass = all(
            item["inside_rigorous_box"] for item in result["cross_checks"]
        )
        report = {
            "schema": "tnfr.collective-pressure-phase-review-report.v1",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": file_receipt(EVIDENCE / "freeze.json")["sha256"],
            "response": file_receipt(EVIDENCE / "response.json", label="response.json"),
            "phase_accuracy_certified": phase_pass,
            "independent_later_checks_passed": auxiliary_pass,
            "later_original_boxes_consistent": boxes_pass,
            "point_cross_checks_consistent": oracles_pass,
            "original_report_pass_count": "8/9 (unchanged)",
            "scope": "Separate retrospective numerical correction from retained primitives; no new physical evidence, event, trajectory or original-verdict rewrite",
        }
        write_json_once(EVIDENCE / "report.json", report)
        print(
            f"Phase accuracy certified: {phase_pass}; independent later checks: {auxiliary_pass}; later boxes: {boxes_pass}; point checks: {oracles_pass}"
        )
        return 0 if phase_pass and auxiliary_pass and boxes_pass and oracles_pass else 1
    except Exception:
        write_json_once(
            EVIDENCE / "failure.json", {"diagnostic": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

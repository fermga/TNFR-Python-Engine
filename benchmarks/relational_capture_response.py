"""One frozen upper-corner preparation: winding formation and basin entry.

The production Euler owner is the only evolution path. Endpoint certificates
concern the ideal law restarted at the retained state, not an integration-error
enclosure for the original continuous initial-value problem.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform

import networkx as nx
import numpy as np

from tnfr.dynamics import relational as owner
from tnfr.dynamics.relational import RelationalExchangeModel, step_relational_exchange
from tnfr.physics.relational_capture import (
    certify_relational_capture,
    certify_relational_local_capture,
)
from tnfr.sdk.relational_reports import relational_report_to_dict

ROOT = Path(__file__).resolve().parents[1]
NODES = tuple(range(10))
CYCLES = (tuple(range(5)), tuple(range(5, 10)))
EDGES = tuple(
    sorted(
        [
            (i + shift, j + shift)
            for shift in (0, 5)
            for i, j in ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))
        ]
        + [(0, 5), (1, 6)]
    )
)
ANGLE = math.pi / 2 - 1 / 64
INITIAL_FORM = (0.4, -0.4, -0.2, 0.0, 0.2) * 2
INITIAL_PHASE = (ANGLE, -ANGLE, -ANGLE, 0.0, ANGLE) * 2
HORIZON = Q(64)
STEP_COUNTS = (4096, 8192, 16384)
CHECKPOINTS = (Q(0), Q(1, 4), Q(1), Q(4), Q(8), Q(16), Q(32), HORIZON)


def _model():
    return RelationalExchangeModel(1.0, phase_domain="positive_resultant")


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in EDGES)
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True, _t=0.0)
    for i in NODES:
        graph.nodes[i].update(EPI=INITIAL_FORM[i], theta=INITIAL_PHASE[i], nu_f=1.0)
    return graph


def prepare_prediction():
    """Retain the single analytic choice before any temporal evaluation."""
    if Path(owner.__file__).resolve() != ROOT / "src/tnfr/dynamics/relational.py":
        raise ValueError("this producer requires workspace source; set PYTHONPATH=src")
    paths = sorted((ROOT / "src/tnfr").rglob("*.py")) + [Path(__file__).resolve()]
    return {
        "protocol": "relational-upper-corner-capture-v1",
        "nodes": NODES,
        "edges": EDGES,
        "cycles": CYCLES,
        "initial_form": INITIAL_FORM,
        "initial_phase": INITIAL_PHASE,
        "capacity": (1.0,) * 10,
        "analytic_preparation": "a=b=pi/2-1/64; A=2/5; B=1/5; c=m=0",
        "representation": "listed_binary64_values_are_authoritative_not_exact_pi_or_fifths",
        "model": {
            "epi_weight": 0.5,
            "phase_weight": 0.5,
            "storage_scale": 1.0,
            "phase_domain": "positive_resultant",
        },
        "horizon": HORIZON,
        "step_counts": STEP_COUNTS,
        "timesteps": tuple(HORIZON / n for n in STEP_COUNTS),
        "checkpoints": CHECKPOINTS,
        "prediction": {
            "initial_windings": (0, 0),
            "windings_from_time": Q(1, 4),
            "expected_windings": (1, 1),
            "target_sector": 1,
            "capture_by_horizon": "an_admitted_reflected_or_full_state_local_certificate_on_each_grid",
            "status": "prospective_finite_hypothesis_not_a_proved_transit_bound",
        },
        "stopping": "first_owner_rejection_or_first_checkpoint_with_admitted_target_certificate_or_horizon",
        "gates": {"maximum_balance_residual": 1e-10},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "randomness": "none",
            "precision": "binary64_with_exact_rational_defects_and_trigonometric_enclosures",
            "integrator": "production_simultaneous_Euler_with_positive_resultant_segment_guard",
        },
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "scope": "prepared_support_and_form_budget; no_forcing_control_projection_or_retuning; endpoint_ideal_continuation_not_original_ODE_error_certificate",
    }


def _owned_state(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _snapshot(graph):
    reflected = certify_relational_capture(graph, model=_model(), cycles=CYCLES)
    local = certify_relational_local_capture(
        graph, model=_model(), cycles=CYCLES, target_sector=1
    )
    return {
        "time": graph.graph["_t"],
        "reflected": relational_report_to_dict(reflected),
        "local": relational_report_to_dict(local),
        "windings": tuple(item.winding for item in reflected.winding),
        "winding_defined": tuple(item.is_defined for item in reflected.winding),
        "capture_admitted": (
            (reflected.admitted and reflected.target_sector == 1) or local.admitted
        ),
    }


def _trace(count, prediction):
    graph = _graph()
    dt = HORIZON / count
    checkpoints = {"0": _snapshot(graph)}
    initial = checkpoints["0"]["reflected"]["report"]["field"]
    maximum = {
        key: Q(0)
        for key in (
            "balance_residual",
            "energy_step_defect",
            "clock_defect",
            "epi_update_defect",
            "phase_update_defect",
            "pressure_split_residual",
            "nodal_rate_rounding_defect",
        )
    }
    maximum["balance_residual"] = abs(_fraction(initial["balance_residual"]))
    for key in ("pressure_split_residual", "nodal_rate_rounding_defect"):
        maximum[key] = max(abs(_fraction(value)) for value in initial[key])
    minimum_resultant = min(
        _fraction(v) for v in initial["resultant_real_lower_bounds"]
    )
    minimum_segment = None
    completed, stop, status = 0, None, "horizon_reached_without_certificate"
    indices = {int(time / dt) for time in CHECKPOINTS}
    for index in range(1, count + 1):
        before = _owned_state(graph)
        try:
            step = step_relational_exchange(graph, model=_model(), dt=float(dt))
        except ValueError as error:
            stop = {
                "attempted_step": index,
                "time_before": graph.graph["_t"],
                "reason": str(error),
                "owned_state_unchanged": before == _owned_state(graph),
                "rejected_proposal": None,
            }
            status = "stopped_by_shared_owner"
            break
        completed = index
        field = step.after
        if (
            field.nodes != NODES
            or field.edges != EDGES
            or field.capacity != (1.0,) * 10
        ):
            raise RuntimeError("the shared owner changed held support or capacity")
        for key in ("energy_step_defect", "clock_defect"):
            maximum[key] = max(maximum[key], abs(getattr(step, key)))
        for key in ("epi_update_defect", "phase_update_defect"):
            maximum[key] = max(maximum[key], *map(abs, getattr(step, key)))
        for key in ("pressure_split_residual", "nodal_rate_rounding_defect"):
            maximum[key] = max(maximum[key], *map(abs, getattr(field, key)))
        maximum["balance_residual"] = max(
            maximum["balance_residual"], abs(field.balance_residual)
        )
        minimum_resultant = min(minimum_resultant, *field.resultant_real_lower_bounds)
        segment = min(step.segment_resultant_real_lower_bounds)
        minimum_segment = (
            segment if minimum_segment is None else min(minimum_segment, segment)
        )
        if index in indices:
            checkpoints[str(index)] = _snapshot(graph)
            if checkpoints[str(index)]["capture_admitted"]:
                status = "target_basin_certified_at_checkpoint"
                break
    if str(completed) not in checkpoints:
        checkpoints[str(completed)] = _snapshot(graph)
    last = checkpoints[str(completed)]
    observed_after_gate = [
        frame
        for frame in checkpoints.values()
        if frame["time"] >= float(prediction["prediction"]["windings_from_time"])
    ]
    winding_passed = bool(observed_after_gate) and all(
        frame["windings"] == (1, 1) and all(frame["winding_defined"])
        for frame in observed_after_gate
    )
    return {
        "requested_steps": count,
        "completed_steps": completed,
        "dt": dt,
        "status": status,
        "stop": stop,
        "checkpoints": checkpoints,
        "last_admitted_checkpoint": str(completed),
        "initial_winding_zero": checkpoints["0"]["windings"] == (0, 0),
        "observed_winding_prediction_passed": winding_passed,
        "endpoint_capture_admitted": last["capture_admitted"],
        "maximum_defects": maximum,
        "minimum_resultant_real_lower_bound": minimum_resultant,
        "minimum_segment_resultant_real_lower_bound": minimum_segment,
    }


def _fraction(value):
    return (
        Q(value["numerator"], value["denominator"])
        if isinstance(value, dict)
        else Q(value)
    )


def _comparison(traces):
    results = {}
    for time in CHECKPOINTS[1:]:
        states = []
        for count in STEP_COUNTS:
            frame = traces[str(count)]["checkpoints"].get(
                str(int(time / (HORIZON / count)))
            )
            if frame is None:
                break
            field = frame["reflected"]["report"]["field"]
            states.append(tuple(map(Q, field["epi"] + field["phase"])))
        results[str(time)] = (
            {
                "status": "common_observed_checkpoint",
                "coarse_to_middle_inf": max(
                    abs(a - b) for a, b in zip(states[0], states[1], strict=True)
                ),
                "middle_to_fine_inf": max(
                    abs(a - b) for a, b in zip(states[1], states[2], strict=True)
                ),
            }
            if len(states) == 3
            else {"status": "unavailable_after_protocol_stop"}
        )
    return results


def evaluate_prediction(prediction):
    if _encoded(prediction) != _encoded(prepare_prediction()):
        raise ValueError("prediction must match the frozen protocol and current source")
    traces = {}
    for count in STEP_COUNTS:
        traces[str(count)] = _trace(count, prediction)
        row = traces[str(count)]
        print(
            json.dumps(
                {
                    "grid": count,
                    "steps": row["completed_steps"],
                    "status": row["status"],
                }
            ),
            flush=True,
        )
    passed = all(
        row["initial_winding_zero"]
        and row["observed_winding_prediction_passed"]
        and row["endpoint_capture_admitted"]
        for row in traces.values()
    )
    return {
        "prediction": prediction,
        "traces": traces,
        "finite_prediction_passed": passed,
        "numerical_work_policy_admitted": all(
            row["maximum_defects"]["balance_residual"]
            <= prediction["gates"]["maximum_balance_residual"]
            for row in traces.values()
        ),
        "grid_comparisons": _comparison(traces),
        "scope": "finite_executed_sector_change_plus_ideal_endpoint_basin; no_validated_error_tube_from_original_W0_state",
    }


def _encoded(value):
    def exact(item):
        if isinstance(item, Q):
            return {"numerator": item.numerator, "denominator": item.denominator}
        raise TypeError(f"unsupported report value {type(item).__name__}")

    return (
        json.dumps(value, default=exact, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    prediction = prepare_prediction()
    frozen = args.output.with_suffix(".prediction.json")
    if args.prepare:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        with frozen.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_encoded(prediction))
        print(f"Prediction frozen: {frozen}")
        return 0
    if frozen.read_text(encoding="utf-8") != _encoded(prediction):
        raise ValueError("freeze the current protocol before evaluation")
    if args.output.exists():
        raise FileExistsError("refusing to replace a retained capture response")
    try:
        report = evaluate_prediction(prediction)
    except Exception as error:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_encoded({"prediction": prediction, "error": repr(error)}))
        raise
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_encoded(report))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "finite_prediction_passed",
                    "numerical_work_policy_admitted",
                )
            }
        )
    )
    return (
        0
        if report["finite_prediction_passed"]
        and report["numerical_work_policy_admitted"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

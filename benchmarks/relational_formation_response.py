"""Frozen regular-chamber crossing and bounded follow-through of two cycles.

The short winding prediction is distinct from later acute entrance or local
basin indicators. Failed admission stops a grid and retains its last admitted
state; this producer never retries with a weaker domain or a smaller step.
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

from tnfr.dynamics.relational import RelationalExchangeModel, step_relational_exchange
from tnfr.dynamics import relational as relational_owner
from tnfr.physics.relational_observations import observe_relational_pattern
from tnfr.physics.winding_certificates import certify_phase_winding

ROOT = Path(__file__).resolve().parents[1]
NODES = tuple(range(10))
CYCLES = (tuple(range(5)), tuple(range(5, 10)))
EDGES = tuple(
    sorted(
        [
            (a + shift, b + shift)
            for shift in (0, 5)
            for a, b in ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))
        ]
        + [(0, 5), (1, 6)]
    )
)
DELTA = math.pi / 4 - 1 / 256
INITIAL_FORM = (
    tuple(
        (2 * math.sqrt(2) / 5) * value
        for value in (math.pi + 8, -math.pi - 8, -3 * math.pi - 4, 0, 3 * math.pi + 4)
    )
    * 2
)
INITIAL_PHASE = tuple(DELTA * coefficient for coefficient in (0, -4, -3, -2, -1)) * 2
REFERENCE_PHASE = (
    tuple((math.tau / 5) * coefficient for coefficient in (0, -4, -3, -2, -1)) * 2
)
HORIZON = Q(16)
STEP_COUNTS = (1024, 2048, 4096)
CHECKPOINT_TIMES = (Q(0), Q(1, 64), Q(1, 32), Q(1, 4), Q(1), Q(4), Q(8), HORIZON)


def _model():
    return RelationalExchangeModel(1.0, phase_domain="positive_resultant")


def _source_fingerprints():
    # The complete package records helper changes as well as dispatch changes.
    paths = sorted((ROOT / "src/tnfr").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def prepare_prediction():
    """Freeze supplied state, finite questions and provenance before execution."""
    if (
        Path(relational_owner.__file__).resolve()
        != ROOT / "src/tnfr/dynamics/relational.py"
    ):
        raise ValueError("this producer requires workspace source; set PYTHONPATH=src")
    model = _model()
    return {
        "protocol": "relational-formation-response-v1",
        "nodes": NODES,
        "edges": EDGES,
        "cycles": CYCLES,
        "initial_delta": DELTA,
        "initial_delta_formula": "pi/4-1/256",
        "initial_form": INITIAL_FORM,
        "initial_form_formula": "(2*sqrt(2)/5)*(pi+8,-pi-8,-3*pi-4,0,3*pi+4), repeated",
        "initial_phase": INITIAL_PHASE,
        "capacity": (1.0,) * 10,
        "reference_phase": REFERENCE_PHASE,
        "model": {
            "epi_weight": model.epi_weight,
            "phase_weight": model.phase_weight,
            "storage_scale": model.storage_scale,
            "phase_domain": model.phase_domain,
        },
        "horizon": HORIZON,
        "step_counts": STEP_COUNTS,
        "timesteps": tuple(HORIZON / count for count in STEP_COUNTS),
        "checkpoint_times": CHECKPOINT_TIMES,
        "short_prediction": {
            "initial_windings": (0, 0),
            "check_times": (Q(1, 64), Q(1, 32)),
            "expected_windings": (1, 1),
        },
        "continuation": "stop_on_first_shared_owner_rejection_without_retry_or_changed_domain",
        "numerical_basin_indicator": {
            "reference": "both aligned acute winding-one cycles on the full joined support",
            "quotient_metric": "sum_of_separate_form_and_phase_squared_norms_in_declared_chart_beta1",
            "radius": math.pi / (20 * math.sqrt(2)),
            "quotient_squared_threshold": Q(9, 800),
            "energy_threshold": 1 / 100000,
            "target_phase_storage_estimate": 10 * (1 - math.cos(math.tau / 5)),
            "premises": "positive capacities;e=w=1/2;beta1;lambda2_lower1/45;cosine_margin>1/10",
            "scope": "numerical_state_indicator_not_rigorous_energy_enclosure_or_exact_ODE_capture",
        },
        "gates": {
            "maximum_actual_balance_residual": 1e-10,
            "scope": "numerical_work_residual_policy_not_a_physical_threshold",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "randomness": "none",
            "precision": "binary64_states_exact_Fraction_defects_and_certified_resultant_bounds",
            "integrator": "production_simultaneous_Euler_positive_resultant_segment_admission",
        },
        "source_sha256": _source_fingerprints(),
        "scope": "conditional_prepared_same_law_crossing_and_bounded_continuation_not_autonomous_support_birth_or_physical_identification",
    }


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in EDGES)
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True, _t=0.0)
    for i in NODES:
        graph.nodes[i].update(EPI=INITIAL_FORM[i], theta=INITIAL_PHASE[i], nu_f=1.0)
    return graph


def _owned_state(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _geometry(graph, field):
    winding = tuple(certify_phase_winding(graph, nodes) for nodes in CYCLES)
    phase = dict(zip(field.nodes, field.phase, strict=True))
    gaps = tuple(
        math.remainder(float(Q(phase[j]) - Q(phase[i])), math.tau)
        for i, j in field.edges
    )
    return {
        "windings": tuple(item.winding for item in winding),
        "winding_defined": tuple(item.is_defined for item in winding),
        "minimum_acute_margin": min(math.pi / 2 - abs(gap) for gap in gaps),
        "strict_acute_edges": all(abs(gap) < math.pi / 2 for gap in gaps),
    }


def _snapshot(graph, model, prediction):
    report = observe_relational_pattern(
        graph,
        model=model,
        reference_phase=dict(zip(NODES, REFERENCE_PHASE, strict=True)),
        regions=CYCLES + (NODES,),
        cycles=CYCLES,
    )
    field = report.field
    geometry = _geometry(graph, field)
    whole = report.regions[-1]
    quotient_squared = whole.form_norm_squared + whole.phase_norm_squared
    policy = prediction["numerical_basin_indicator"]
    excess = float(field.storage) - policy["target_phase_storage_estimate"]
    eligible = (
        geometry["windings"] == (1, 1)
        and geometry["strict_acute_edges"]
        and quotient_squared < policy["quotient_squared_threshold"]
        and 0 <= excess < policy["energy_threshold"]
    )
    return {
        "time": graph.graph["_t"],
        "nodes": field.nodes,
        "edges": field.edges,
        "epi": field.epi,
        "phase": field.phase,
        "capacity": field.capacity,
        "pressure": field.pressure,
        "phase_source": field.phase_source,
        "form_rate": field.form_rate,
        "phase_rate": field.phase_rate,
        "phase_metric": field.phase_metric,
        "pressure_split_residual": field.pressure_split_residual,
        "nodal_rate_rounding_defect": field.nodal_rate_rounding_defect,
        "field_scope": field.scope,
        "storage": field.storage,
        "balance_residual": field.balance_residual,
        "resultant_real_lower_bounds": field.resultant_real_lower_bounds,
        "geometry": geometry,
        "regions": tuple(
            {
                "nodes": row.nodes,
                "form_mean": row.form_mean,
                "phase_error_mean": row.phase_error_mean,
                "centered_form": row.centered_form,
                "centered_phase_error": row.centered_phase_error,
                "form_norm_squared": row.form_norm_squared,
                "phase_norm_squared": row.phase_norm_squared,
            }
            for row in report.regions
        ),
        "numerical_basin": {
            "eligible": eligible,
            "quotient_squared_norm": quotient_squared,
            "excess_storage_estimate": excess,
            "scope": policy["scope"],
        },
    }


def _trace(steps, prediction):
    graph, model = _graph(), _model()
    dt = float(HORIZON / steps)
    checkpoints = {"0": _snapshot(graph, model, prediction)}
    initial = checkpoints["0"]
    maximum_balance = abs(initial["balance_residual"])
    minimum_resultant = min(initial["resultant_real_lower_bounds"])
    minimum_segment = None
    maximum_step_defect = maximum_clock_defect = Q(0)
    maximum_epi_update_defect = maximum_phase_update_defect = Q(0)
    maximum_pressure_split = max(map(abs, initial["pressure_split_residual"]))
    maximum_nodal_rounding = max(map(abs, initial["nodal_rate_rounding_defect"]))
    first_winding_one = first_acute = None
    completed, stopped = 0, None
    checkpoint_indices = {int(time / (HORIZON / steps)) for time in CHECKPOINT_TIMES}
    for index in range(1, steps + 1):
        before = _owned_state(graph)
        try:
            step = step_relational_exchange(graph, model=model, dt=dt)
        except ValueError as error:
            stopped = {
                "attempted_step": index,
                "time_before": graph.graph["_t"],
                "attempted_dt": dt,
                "attempted_clock_target": Q(graph.graph["_t"]) + Q(dt),
                "exception_type": type(error).__name__,
                "reason": str(error),
                "owned_state_unchanged": _owned_state(graph) == before,
                "rejected_proposal": None,
                "rejected_proposal_status": "shared_owner_returned_no_proposal_evidence",
            }
            break
        completed = index
        field = step.after
        if (
            field.nodes != NODES
            or field.edges != EDGES
            or field.capacity != (1.0,) * 10
        ):
            raise RuntimeError(
                "the shared step changed held scientific support or capacity"
            )
        maximum_balance = max(maximum_balance, abs(field.balance_residual))
        minimum_resultant = min(minimum_resultant, *field.resultant_real_lower_bounds)
        segment = min(step.segment_resultant_real_lower_bounds)
        minimum_segment = (
            segment if minimum_segment is None else min(minimum_segment, segment)
        )
        maximum_step_defect = max(maximum_step_defect, abs(step.energy_step_defect))
        maximum_clock_defect = max(maximum_clock_defect, abs(step.clock_defect))
        maximum_epi_update_defect = max(
            maximum_epi_update_defect, *map(abs, step.epi_update_defect)
        )
        maximum_phase_update_defect = max(
            maximum_phase_update_defect, *map(abs, step.phase_update_defect)
        )
        maximum_pressure_split = max(
            maximum_pressure_split, *map(abs, field.pressure_split_residual)
        )
        maximum_nodal_rounding = max(
            maximum_nodal_rounding, *map(abs, field.nodal_rate_rounding_defect)
        )
        geometry = _geometry(graph, field)
        if first_winding_one is None and geometry["windings"] == (1, 1):
            first_winding_one = step.t_after
        if first_acute is None and geometry["strict_acute_edges"]:
            first_acute = step.t_after
        if index in checkpoint_indices:
            checkpoints[str(index)] = _snapshot(graph, model, prediction)
    if str(completed) not in checkpoints:
        checkpoints[str(completed)] = _snapshot(graph, model, prediction)
    checks = {}
    for time in prediction["short_prediction"]["check_times"]:
        index = int(time / (HORIZON / steps))
        saved = checkpoints.get(str(index))
        checks[str(time)] = saved is not None and saved["geometry"]["windings"] == (
            1,
            1,
        )
    return {
        "requested_steps": steps,
        "completed_steps": completed,
        "dt": dt,
        "status": "completed" if stopped is None else "stopped_by_shared_owner",
        "stop": stopped,
        "checkpoints": checkpoints,
        "initial_winding_zero": initial["geometry"]["windings"] == (0, 0),
        "short_prediction_checks": checks,
        "first_observed_winding_one_time": first_winding_one,
        "first_observed_acute_time": first_acute,
        "minimum_resultant_real_lower_bound": minimum_resultant,
        "minimum_segment_resultant_real_lower_bound": minimum_segment,
        "maximum_actual_balance_residual": maximum_balance,
        "maximum_energy_step_defect": maximum_step_defect,
        "maximum_clock_defect": maximum_clock_defect,
        "maximum_epi_update_defect": maximum_epi_update_defect,
        "maximum_phase_update_defect": maximum_phase_update_defect,
        "maximum_pressure_split_residual": maximum_pressure_split,
        "maximum_nodal_rate_rounding_defect": maximum_nodal_rounding,
        "last_admitted_checkpoint": str(completed),
        "final_horizon_basin_indicator": (
            checkpoints[str(completed)]["numerical_basin"]["eligible"]
            if completed == steps
            else None
        ),
    }


def _comparison(traces):
    comparisons = {}
    for time in CHECKPOINT_TIMES[1:]:
        states = []
        for count in STEP_COUNTS:
            frame = traces[str(count)]["checkpoints"].get(
                str(int(time / (HORIZON / count)))
            )
            if frame is None:
                states = []
                break
            states.append(tuple(map(Q, frame["epi"] + frame["phase"])))
        comparisons[str(time)] = (
            {
                "status": "observed_on_all_grids",
                "coarse_to_middle_inf": max(
                    abs(b - a) for a, b in zip(states[0], states[1], strict=True)
                ),
                "middle_to_fine_inf": max(
                    abs(b - a) for a, b in zip(states[1], states[2], strict=True)
                ),
            }
            if states
            else {"status": "unavailable_not_all_grids_reached_checkpoint"}
        )
    return comparisons


def evaluate_prediction(prediction):
    if _encoded(prediction) != _encoded(prepare_prediction()):
        raise ValueError("prediction must match the current frozen protocol and source")
    traces = {str(count): _trace(count, prediction) for count in STEP_COUNTS}
    short_passed = all(
        trace["initial_winding_zero"] and all(trace["short_prediction_checks"].values())
        for trace in traces.values()
    )
    work_admitted = all(
        trace["maximum_actual_balance_residual"]
        <= prediction["gates"]["maximum_actual_balance_residual"]
        for trace in traces.values()
    )
    return {
        "prediction": prediction,
        "traces": traces,
        "short_prediction_passed": short_passed,
        "numerical_work_policy_admitted": work_admitted,
        "continuation_status": (
            "all_grids_completed"
            if all(trace["status"] == "completed" for trace in traces.values())
            else "stopped_attempts_retained"
        ),
        "grid_comparisons": _comparison(traces),
        "scope": "finite_same_law_crossing_and_bounded_continuation_not_exact_ODE_capture_or_autonomous_formation",
    }


def _encoded(value):
    def exact(item):
        if isinstance(item, Q):
            return {"numerator": item.numerator, "denominator": item.denominator}
        raise TypeError(f"unsupported report value {type(item).__name__}")

    return (
        json.dumps(value, default=exact, sort_keys=True, indent=2, allow_nan=False)
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
        raise FileExistsError("refusing to replace a retained formation response")
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
                    "short_prediction_passed",
                    "numerical_work_policy_admitted",
                    "continuation_status",
                )
            },
            sort_keys=True,
        )
    )
    return (
        0
        if report["short_prediction_passed"]
        and report["numerical_work_policy_admitted"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

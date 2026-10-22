"""Prospective finite recovery of a prepared C5 winding geometry.

Two Euler grids execute the shared relational owner, with one frozen
perturbation and a zero-capacity boundary. This is a finite conditional
recovery observation, not autonomous formation or physical identification.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import platform

import networkx as nx
import numpy as np

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)
from tnfr.physics.winding_certificates import certify_phase_winding

ROOT = Path(__file__).resolve().parents[1]
NODES = tuple(range(5))
EDGES = ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))
CAPACITY = (1.0, 1.25, 0.75, 1.5, 1.0)
EPSILON = Q(1, 256)
FORM_DIRECTION = (1, -1, 0, 0, 0)
PHASE_DIRECTION = (0, 1, -1, 0, 0)
REFERENCE_PHASE = tuple(math.tau * i / 5 for i in NODES)
HORIZON = Q(32)
STEP_COUNTS = (2048, 4096)
MODEL = RelationalExchangeModel(1.0)
FINGERPRINT_PATHS = (
    "benchmarks/relational_local_recovery.py",
    "src/tnfr/dynamics/relational.py",
    "src/tnfr/dynamics/_euler_kernel.py",
    "src/tnfr/dynamics/canonical.py",
    "src/tnfr/dynamics/dnfr.py",
    "src/tnfr/dynamics/fused_dnfr.py",
    "src/tnfr/config/defaults_core.py",
    "src/tnfr/_exact_time.py",
    "src/tnfr/alias.py",
    "src/tnfr/types.py",
    "src/tnfr/gamma.py",
    "src/tnfr/metrics/common.py",
    "src/tnfr/mathematics/epi.py",
    "src/tnfr/mathematics/_neighbor_differences.py",
    "src/tnfr/mathematics/_phase_midpoint.py",
    "src/tnfr/mathematics/unified_numerical.py",
    "src/tnfr/utils/graph.py",
    "src/tnfr/utils/numeric.py",
    "src/tnfr/physics/winding_certificates.py",
)


def prepare_prediction():
    """Freeze scientific settings and listed provenance without a trajectory."""
    return {
        "protocol": "relational-local-recovery-v1",
        "nodes": NODES,
        "edges": EDGES,
        "capacity": CAPACITY,
        "epsilon": EPSILON,
        "form_direction": FORM_DIRECTION,
        "phase_direction": PHASE_DIRECTION,
        "reference_phase": REFERENCE_PHASE,
        "model": {"epi_weight": 0.5, "phase_weight": 0.5, "storage_scale": 1.0},
        "horizon": HORIZON,
        "step_counts": STEP_COUNTS,
        "timesteps": tuple(HORIZON / count for count in STEP_COUNTS),
        "identity": "centered_form_and_centered_real_lift_error_from_prepared_W1_twist",
        "gates": {
            "maximum_final_to_initial_distance": 0.5,
            "maximum_grid_difference_to_initial_distance": 1 / 64,
            "minimum_acute_margin": math.pi / 20,
            "maximum_actual_balance_residual": 1e-12,
            "winding": 1,
        },
        "prior_linear_estimate": {
            "slowest_decay_lower_bound": 0.75 * (2 - 2 * math.cos(math.tau / 5)) / 8,
            "scope": "linear_spectral_estimate_not_nonlinear_finite_error_bound",
        },
        "negative_control": "same_perturbation_all_capacities_zero_one_step_dt_1_128",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "precision": "binary64_states_exact_Fraction_defects",
            "integrator": "production_step_relational_exchange_explicit_Euler",
            "pressure_path": "fused_canonical",
            "randomness": "none",
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in FINGERPRINT_PATHS
        },
        "scope": "prepared_conditional_geometry_finite_recovery_not_formation_or_physical_validation",
    }


def _graph(*, frozen=False):
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in EDGES)
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True, _t=0.0)
    for i in NODES:
        graph.nodes[i].update(
            EPI=float(EPSILON * FORM_DIRECTION[i]),
            theta=REFERENCE_PHASE[i] + float(EPSILON * PHASE_DIRECTION[i]),
            nu_f=0.0 if frozen else CAPACITY[i],
        )
    return graph


def _center(values):
    mean = math.fsum(values) / len(values)
    return tuple(value - mean for value in values), mean


def _observation(graph, field, *, time):
    form, form_mean = _center(field.epi)
    phase, phase_mean = _center(
        tuple(a - b for a, b in zip(field.phase, REFERENCE_PHASE, strict=True))
    )
    winding = certify_phase_winding(graph, NODES)
    return {
        "time": time,
        "epi": field.epi,
        "phase": field.phase,
        "capacity": field.capacity,
        "pressure": field.pressure,
        "form_rate": field.form_rate,
        "phase_rate": field.phase_rate,
        "form_mean": form_mean,
        "phase_error_mean": phase_mean,
        "quotient_state": form + phase,
        "quotient_distance": math.hypot(*form, *phase),
        "storage": field.storage,
        "balance_residual": field.balance_residual,
        "winding": winding.winding,
        "winding_defined": winding.is_defined,
        "acute_margin": winding.minimum_u3_margin,
        "winding_residual": winding.quantization_residual,
        "pressure_path": field.pressure_path,
    }


def _trace(steps):
    graph = _graph()
    field = evaluate_relational_exchange(graph, model=MODEL)
    initial = _observation(graph, field, time=0.0)
    checkpoints = {"0": initial}
    minimum_margin = initial["acute_margin"]
    maximum_balance = abs(field.balance_residual)
    maximum_step_defect = Q(0)
    maximum_energy_increase = Q(0)
    maximum_clock_defect = Q(0)
    held = winding_preserved = pressure_path_retained = True
    dt = float(HORIZON / steps)
    for index in range(1, steps + 1):
        step = step_relational_exchange(graph, model=MODEL, dt=dt)
        field = step.after
        observed = _observation(graph, field, time=step.t_after)
        minimum_margin = min(minimum_margin, observed["acute_margin"])
        maximum_balance = max(maximum_balance, abs(field.balance_residual))
        maximum_step_defect = max(maximum_step_defect, abs(step.energy_step_defect))
        maximum_energy_increase = max(maximum_energy_increase, step.energy_change)
        maximum_clock_defect = max(maximum_clock_defect, abs(step.clock_defect))
        held = held and field.capacity == CAPACITY and field.edges == EDGES
        winding_preserved = winding_preserved and (
            observed["winding_defined"] and observed["winding"] == 1
        )
        pressure_path_retained = pressure_path_retained and (
            field.pressure_path == "fused_canonical"
        )
        if index in (steps // 4, steps // 2, steps):
            checkpoints[str(index)] = observed
    return {
        "steps": steps,
        "dt": dt,
        "checkpoints": checkpoints,
        "final_to_initial_distance": observed["quotient_distance"]
        / initial["quotient_distance"],
        "minimum_acute_margin": minimum_margin,
        "maximum_actual_balance_residual": maximum_balance,
        "maximum_energy_step_defect": maximum_step_defect,
        "maximum_energy_increase": maximum_energy_increase,
        "maximum_clock_defect": maximum_clock_defect,
        "capacity_and_support_held": held,
        "winding_preserved": winding_preserved,
        "pressure_path_retained": pressure_path_retained,
        "observed_state_count": steps + 1,
    }


def _zero_capacity_control():
    graph = _graph(frozen=True)
    step = step_relational_exchange(graph, model=MODEL, dt=1 / 128)
    before, after = step.before, step.after
    return {
        "epi_before": before.epi,
        "epi_after": after.epi,
        "phase_before": before.phase,
        "phase_after": after.phase,
        "pressure": after.pressure,
        "form_rate": after.form_rate,
        "phase_rate": after.phase_rate,
        "clock_advanced": step.t_after > step.t_before,
        "state_frozen": before.epi == after.epi and before.phase == after.phase,
        "nonzero_pressure_retained": any(after.pressure),
    }


def evaluate_prediction(prediction):
    """Execute only the exact current native protocol; never fit a response."""
    if _encoded(prediction) != _encoded(prepare_prediction()):
        raise ValueError("prediction must match the current frozen protocol and source")
    traces = {str(count): _trace(count) for count in STEP_COUNTS}
    initial = traces[str(STEP_COUNTS[0])]["checkpoints"]["0"]["quotient_distance"]
    final = tuple(
        traces[str(count)]["checkpoints"][str(count)]["quotient_state"]
        for count in STEP_COUNTS
    )
    difference = math.dist(*final)
    boundary = _zero_capacity_control()
    gates = prediction["gates"]
    checks = {
        "recovery_on_both_grids": all(
            trace["final_to_initial_distance"]
            <= gates["maximum_final_to_initial_distance"]
            for trace in traces.values()
        ),
        "small_endpoint_grid_difference": difference
        <= initial * gates["maximum_grid_difference_to_initial_distance"],
        "acute_margin_retained": all(
            trace["minimum_acute_margin"] >= gates["minimum_acute_margin"]
            for trace in traces.values()
        ),
        "balance_residual_admitted": all(
            trace["maximum_actual_balance_residual"]
            <= gates["maximum_actual_balance_residual"]
            for trace in traces.values()
        ),
        "held_state_and_path": all(
            trace["capacity_and_support_held"]
            and trace["pressure_path_retained"]
            and trace["winding_preserved"]
            for trace in traces.values()
        ),
        "zero_capacity_boundary": all(
            boundary[key]
            for key in ("state_frozen", "clock_advanced", "nonzero_pressure_retained")
        ),
    }
    return {
        "prediction": prediction,
        "traces": traces,
        "endpoint_grid_difference": difference,
        "endpoint_grid_difference_to_initial_distance": difference / initial,
        "zero_capacity": boundary,
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "finite_prepared_recovery_two_grid_difference_not_an_ODE_error_certificate",
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
        raise FileExistsError("refusing to replace a retained recovery response")
    try:
        report = evaluate_prediction(prediction)
    except Exception as error:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(
                _encoded(
                    {"prediction": prediction, "passed": False, "error": repr(error)}
                )
            )
        raise
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_encoded(report))
    print(
        json.dumps(
            {"passed": report["passed"], "checks": report["checks"]}, sort_keys=True
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

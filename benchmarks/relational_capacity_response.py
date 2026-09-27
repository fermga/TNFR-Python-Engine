"""Frozen finite nonlinear capacity response of two conditional joint laws.

The relational traces execute the production opt-in solver. The competing
capacity-pair phase term belongs only to this benchmark. Twelve traces compare
the same K3 preparation, two held capacities and three Euler grids. Refinement
and Taylor-ratio gates are prospective numerical hypotheses, not rigorous ODE
remainder bounds or evidence selecting a physical law. Freeze before execution.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import networkx as nx
import numpy as np

from tnfr._exact_time import finite_represented_real
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    _admit_phase_segment,
    _advance,
    evaluate_relational_exchange,
    step_relational_exchange,
)

ROOT = Path(__file__).resolve().parents[1]
NODES = (0, 1, 2)
EDGES = ((0, 1), (0, 2), (1, 2))
HORIZON = Q(1, 32)
STEP_COUNTS = (128, 256, 512)
INITIAL_FORM = (Q(1, 3), Q(0), Q(-1, 3))
INITIAL_PHASE = (0.0, math.pi / 6, -math.pi / 6)
CAPACITIES = {"baseline": (1.0, 1.0, 1.0), "intervened": (1.0, 0.5, 1.0)}
LAWS = ("relational", "counterpart")
MODEL = RelationalExchangeModel(1.0, epi_weight=0.5, phase_weight=0.5)
PRESSURE_PATH = "fused_canonical"
SLACK = Q(1, 10**12)
CONTRACTION = Q(3, 4)
SIGNAL_FACTOR = 32
FINGERPRINT_PATHS = (
    "benchmarks/relational_capacity_response.py",
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
)


def _finite(value, label):
    return finite_represented_real(value, label)[0]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_prediction():
    """Declare inputs, numerical hypotheses and provenance without a trajectory."""
    slope = -(2 + math.sqrt(3)) / 192
    acceleration = (2 + math.sqrt(3)) / (384 * math.pi)
    return {
        "protocol": "relational-capacity-finite-response-v1",
        "scope": "finite_internal_conditional_model_comparison_not_physical_selection",
        "nodes": NODES,
        "edges": EDGES,
        "conductance": 1,
        "initial_epi_exact": INITIAL_FORM,
        "initial_epi_materialized": tuple(Q(float(value)) for value in INITIAL_FORM),
        "initial_phase": tuple(map(Q, INITIAL_PHASE)),
        "capacities": dict(CAPACITIES),
        "horizon": HORIZON,
        "step_counts": STEP_COUNTS,
        "timesteps": tuple(HORIZON / steps for steps in STEP_COUNTS),
        "epi_weight": MODEL.epi_weight,
        "phase_weight": MODEL.phase_weight,
        "storage_scale": MODEL.storage_scale,
        "laws": LAWS,
        "reference_execution": "production_step_relational_exchange",
        "counterpart_execution": "benchmark_only_S_gradV_with_shared_field_and_Euler_arithmetic",
        "counterpart_formula": "Sij=(w/beta)*Aij*nu_i*nu_j/(nu_i+nu_j)/(d_i*d_j)*(Lx_i+Lx_j)*sin(theta_j-theta_i); Z=S*gradV",
        "observable": "chi=theta0-(theta1+theta2)/2; form=EPI0",
        "contrast": "counterpart_intervened-counterpart_baseline-(relational_intervened-relational_baseline)",
        "taylor": {
            "chi_initial_slope_exact": "-(2+sqrt(3))/192",
            "form_initial_acceleration_exact": "(2+sqrt(3))/(384*pi)",
            "chi_initial_slope_estimate": slope,
            "form_initial_acceleration_estimate": acceleration,
            "chi_leading_response_estimate": slope * float(HORIZON),
            "form_leading_response_estimate": acceleration * float(HORIZON) ** 2 / 2,
            "scope": "estimates_for_prospective_finite_hypotheses_not_remainder_bounds",
        },
        "gates": {
            "normalized_response_interval": (Q(3, 4), Q(5, 4)),
            "refinement_contraction": CONTRACTION,
            "numerical_slack": SLACK,
            "signal_to_last_grid_difference": SIGNAL_FACTOR,
            "minimum_chart_margin": math.pi / 8,
            "maximum_actual_balance_residual": SLACK,
            "refinement_scope": "three_grid_observation_not_certified_error_or_convergence",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "precision": "binary64_states_and_exact_Fraction_arithmetic_on_retained_scalars",
            "integrator": "simultaneous_explicit_Euler_with_initial_acute_lift_admission",
            "pressure_path": PRESSURE_PATH,
            "randomness": "none",
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in FINGERPRINT_PATHS
        },
    }


def _graph(capacity_name):
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in EDGES)
    graph.graph.update(
        GAMMA={"type": "none"},
        DNFR_WEIGHTS={"epi": 0.5, "phase": 0.5, "vf": 0.0, "topo": 0.0},
        vectorized_dnfr=True,
        _t=0.0,
    )
    for node, epi, phase, capacity in zip(
        NODES, INITIAL_FORM, INITIAL_PHASE, CAPACITIES[capacity_name], strict=True
    ):
        graph.nodes[node].update(EPI=float(epi), theta=phase, nu_f=capacity)
    return graph


def counterpart_phase_rate(field):
    """Benchmark-only capacity-pair counterpart, with no runtime installation.

    Form differences, phase gradient and native rates come from the production
    field owner. Each undirected edge constructs its skew coefficient once.
    Materialized rates need not give exactly zero work; that residual is read.
    """
    _require(
        field.nodes == NODES and field.edges == EDGES, "counterpart requires ordered K3"
    )
    degrees = tuple(sum(node in edge for edge in field.edges) for node in field.nodes)
    normalized = tuple(
        Q(q) / degree for q, degree in zip(field.form_gradient, degrees, strict=True)
    )
    extra = [Q(0)] * len(field.nodes)
    for i, j in field.edges:
        ni, nj = Q(field.capacity[i]), Q(field.capacity[j])
        mobility = Q(0) if ni + nj == 0 else ni * nj / (ni + nj)
        gap = math.remainder(
            _finite(Q(field.phase[j]) - Q(field.phase[i]), "counterpart phase gap"),
            math.tau,
        )
        coefficient = (
            Q(field.model.phase_weight)
            * mobility
            * (normalized[i] + normalized[j])
            * Q(math.sin(gap))
            / (Q(field.model.storage_scale) * degrees[i] * degrees[j])
        )
        extra[i] += coefficient * Q(field.phase_gradient[j])
        extra[j] -= coefficient * Q(field.phase_gradient[i])
    return tuple(
        _finite(Q(base) + addition, "counterpart phase rate")
        for base, addition in zip(field.phase_rate, extra, strict=True)
    )


def _actual_rates(field, law):
    _require(law in LAWS, "unknown comparison law")
    phase_rate = (
        field.phase_rate if law == "relational" else counterpart_phase_rate(field)
    )
    extra_work = Q(field.model.storage_scale) * sum(
        (
            Q(gradient) * (Q(actual) - Q(base))
            for gradient, actual, base in zip(
                field.phase_gradient, phase_rate, field.phase_rate, strict=True
            )
        ),
        Q(0),
    )
    actual_work = field.storage_rate + extra_work
    return phase_rate, extra_work, actual_work


def _counterpart_step(graph, *, dt, t):
    """One comparison-only step, admitted by the same scalar/segment owners."""
    before = evaluate_relational_exchange(graph, model=MODEL)
    phase_rate, extra_work, actual_work = _actual_rates(before, "counterpart")
    duration = _finite(dt, "dt")
    clock = _finite(t, "t")
    _require(duration > 0, "dt must be positive")
    next_clock, clock_defect = _advance(clock, 1.0, duration, "clock")
    _require(next_clock > clock, "dt must advance the clock")
    staged = graph.copy()
    epi_defects, phase_defects = [], []
    for i, node in enumerate(before.nodes):
        form, epi_defect = _advance(before.epi[i], before.form_rate[i], duration, "EPI")
        phase, phase_defect = _advance(
            before.phase[i], phase_rate[i], duration, "phase"
        )
        staged.nodes[node].update(EPI=form, theta=phase)
        epi_defects.append(epi_defect)
        phase_defects.append(phase_defect)
    _admit_phase_segment(before, staged)
    after = evaluate_relational_exchange(staged, model=MODEL)
    change = after.storage - before.storage
    for i, node in enumerate(before.nodes):
        graph.nodes[node].update(
            EPI=after.epi[i], theta=after.phase[i], delta_nfr=after.pressure[i]
        )
    graph.graph["_t"] = next_clock
    return {
        "before": before,
        "after": after,
        "epi_update_defect": tuple(epi_defects),
        "phase_update_defect": tuple(phase_defects),
        "clock_defect": clock_defect,
        "energy_change": change,
        "energy_step_defect": change - Q(duration) * actual_work,
        "extra_work": extra_work,
    }


def _chart_margin(field):
    return min(
        math.pi / 2
        - abs(
            math.remainder(
                _finite(Q(field.phase[j]) - Q(field.phase[i]), "phase gap"), math.tau
            )
        )
        for i, j in field.edges
    )


def _checkpoint(field, law, clock):
    phase_rate, extra_work, actual_work = _actual_rates(field, law)
    return {
        "nodes": field.nodes,
        "edges": field.edges,
        "epi": field.epi,
        "phase": field.phase,
        "capacity": field.capacity,
        "pressure": field.pressure,
        "form_rate": field.form_rate,
        "phase_rate": phase_rate,
        "phase_metric": field.phase_metric,
        "phase_source": field.phase_source,
        "storage": field.storage,
        "continuous_loss": field.continuous_loss,
        "actual_storage_rate": actual_work,
        "actual_balance_residual": actual_work + field.continuous_loss,
        "extra_work": extra_work,
        "clock": clock,
    }


def _trace(law, capacity_name, steps):
    _require(steps in STEP_COUNTS, "unfrozen step count")
    _require(law in LAWS, "unfrozen comparison law")
    graph = _graph(capacity_name)
    dt = HORIZON / steps
    started = perf_counter()
    field = evaluate_relational_exchange(graph, model=MODEL)
    checkpoints = {"initial": _checkpoint(field, law, Q(0))}
    minimum_margin = minimum_metric = math.inf
    max_work = max_extra_work = max_storage_defect = max_update_defect = Q(0)
    max_clock_defect = max_pressure_split_defect = Q(0)
    pressure_paths = set()

    def retain_evidence(current):
        nonlocal minimum_margin, minimum_metric, max_work, max_extra_work
        nonlocal max_pressure_split_defect
        _require(
            current.nodes == NODES and current.edges == EDGES, "held support changed"
        )
        _require(current.capacity == CAPACITIES[capacity_name], "held capacity changed")
        pressure_paths.add(current.pressure_path)
        _require(
            current.pressure_path == PRESSURE_PATH, "pressure execution path changed"
        )
        margin = _chart_margin(current)
        minimum_margin = min(minimum_margin, margin)
        minimum_metric = min(minimum_metric, *current.phase_metric)
        _require(margin > math.pi / 8, "chart margin crossed its frozen guard")
        _require(minimum_metric > 0, "phase metric is not positive")
        _, extra_work, actual_work = _actual_rates(current, law)
        residual = actual_work + current.continuous_loss
        max_work = max(max_work, abs(residual))
        max_extra_work = max(max_extra_work, abs(extra_work))
        max_pressure_split_defect = max(
            max_pressure_split_defect, *map(abs, current.pressure_split_residual)
        )
        _require(max_work <= SLACK, "actual work residual exceeded the frozen guard")

    retain_evidence(field)
    for index in range(steps):
        clock = float(index * dt)
        if law == "relational":
            step = step_relational_exchange(graph, model=MODEL, dt=float(dt), t=clock)
            field = step.after
            epi_defects, phase_defects = (
                step.epi_update_defect,
                step.phase_update_defect,
            )
            storage_defect, clock_defect = step.energy_step_defect, step.clock_defect
        else:
            step = _counterpart_step(graph, dt=float(dt), t=clock)
            field = step["after"]
            epi_defects, phase_defects = (
                step["epi_update_defect"],
                step["phase_update_defect"],
            )
            storage_defect, clock_defect = (
                step["energy_step_defect"],
                step["clock_defect"],
            )
        retain_evidence(field)
        max_storage_defect = max(max_storage_defect, abs(storage_defect))
        max_update_defect = max(
            max_update_defect, *map(abs, epi_defects + phase_defects)
        )
        max_clock_defect = max(max_clock_defect, abs(clock_defect))
        _require(
            Q(graph.graph["_t"]) == (index + 1) * dt,
            "structural clock differs from the frozen grid",
        )
        if index + 1 in (steps // 2, steps):
            label = "midpoint" if index + 1 == steps // 2 else "final"
            checkpoints[label] = _checkpoint(field, law, Q(graph.graph["_t"]))
    return {
        "law": law,
        "capacity_case": capacity_name,
        "steps": steps,
        "dt": dt,
        "checkpoints": checkpoints,
        "minimum_chart_margin": minimum_margin,
        "minimum_phase_metric": minimum_metric,
        "maximum_actual_balance_residual": max_work,
        "maximum_extra_work_residual": max_extra_work,
        "maximum_pressure_split_residual": max_pressure_split_defect,
        "maximum_euler_storage_defect": max_storage_defect,
        "maximum_update_defect": max_update_defect,
        "maximum_clock_defect": max_clock_defect,
        "pressure_paths": tuple(sorted(pressure_paths)),
        "elapsed_structural_clock": Q(graph.graph["_t"]),
        "elapsed_wall_seconds": perf_counter() - started,
    }


def _observable(checkpoint, name):
    if name == "chi":
        phase = tuple(map(Q, checkpoint["phase"]))
        return phase[0] - (phase[1] + phase[2]) / 2
    return Q(checkpoint["epi"][0])


def _refinement(coarse, middle, fine, *, signal=None):
    first, second = abs(middle - coarse), abs(fine - middle)
    result = {
        "coarse_to_middle": first,
        "middle_to_fine": second,
        "contracting": second <= CONTRACTION * first + SLACK,
    }
    if signal is not None:
        result["signal_resolved"] = abs(signal) > SIGNAL_FACTOR * (second + SLACK)
    return result


def evaluate_prediction(prediction):
    """Execute only the unchanged protocol and retain all finite decisions."""
    _require(
        _encoded(prediction) == _encoded(prepare_prediction()),
        "prediction differs from the frozen protocol/source",
    )
    trajectories = {
        law: {
            capacity: {
                str(steps): _trace(law, capacity, steps) for steps in STEP_COUNTS
            }
            for capacity in CAPACITIES
        }
        for law in LAWS
    }
    observables, state_refinement, observable_refinement = {}, {}, {}
    checks = []
    lower, upper = map(float, prediction["gates"]["normalized_response_interval"])
    for steps in STEP_COUNTS:
        rows = {}
        for name in ("chi", "form"):
            responses = {
                law: _observable(
                    trajectories[law]["intervened"][str(steps)]["checkpoints"]["final"],
                    name,
                )
                - _observable(
                    trajectories[law]["baseline"][str(steps)]["checkpoints"]["final"],
                    name,
                )
                for law in LAWS
            }
            difference = responses["counterpart"] - responses["relational"]
            leading = prediction["taylor"][f"{name}_leading_response_estimate"]
            ratio = float(difference) / leading
            admitted = lower <= ratio <= upper
            checks.append(admitted)
            rows[name] = {
                "capacity_responses": responses,
                "difference_of_differences": difference,
                "normalized_response": ratio,
                "inside_frozen_taylor_band": admitted,
            }
        observables[str(steps)] = rows
    for name in ("chi", "form"):
        values = tuple(
            observables[str(steps)][name]["difference_of_differences"]
            for steps in STEP_COUNTS
        )
        refinement = _refinement(*values, signal=values[-1])
        observable_refinement[name] = refinement
        checks.extend((refinement["contracting"], refinement["signal_resolved"]))
    for law in LAWS:
        state_refinement[law] = {}
        for capacity in CAPACITIES:
            rows = {}
            for checkpoint in ("midpoint", "final"):
                states = []
                for steps in STEP_COUNTS:
                    retained = trajectories[law][capacity][str(steps)]["checkpoints"][
                        checkpoint
                    ]
                    states.append(tuple(map(Q, retained["epi"] + retained["phase"])))
                first = max(
                    abs(b - a) for a, b in zip(states[0], states[1], strict=True)
                )
                second = max(
                    abs(b - a) for a, b in zip(states[1], states[2], strict=True)
                )
                contracting = second <= CONTRACTION * first + SLACK
                checks.append(contracting)
                rows[checkpoint] = {
                    "coarse_to_middle_inf": first,
                    "middle_to_fine_inf": second,
                    "contracting": contracting,
                }
            state_refinement[law][capacity] = rows
    return {
        "prediction": prediction,
        "trajectories": trajectories,
        "observables": observables,
        "observable_refinement": observable_refinement,
        "state_refinement": state_refinement,
        "passed": all(checks),
        "scope": "finite_nonlinear_internal_comparison; empirical_grid_and_Taylor_checks_not_certified_ODE_bounds_or_physical_selection",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prediction = prepare_prediction()
    frozen_path = args.output.with_suffix(".prediction.json")
    if args.prepare:
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        with frozen_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_encoded(prediction))
        print(f"Prediction frozen: {frozen_path}")
        return 0
    _require(
        frozen_path.read_text(encoding="utf-8") == _encoded(prediction),
        "freeze the current protocol before evaluation",
    )
    if args.output.exists():
        raise FileExistsError("refusing to replace a retained response")
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
            {
                "passed": report["passed"],
                "finest": {
                    name: float(row["difference_of_differences"])
                    for name, row in report["observables"][str(STEP_COUNTS[-1])].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

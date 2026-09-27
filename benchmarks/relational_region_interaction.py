"""Frozen transmission between two prepared relational cycle geometries.

One internal form impulse is compared on joined support and on two separately
executed rings. Three short Euler grids test a prospective second-order
response. No new force, regional closure or physical identification is added.
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
from tnfr.physics.support_transport import (
    observe_regional_support_balance,
    observe_support_transport,
)
from tnfr.physics.winding_certificates import certify_phase_winding

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {"left": tuple(range(5)), "right": tuple(range(5, 10))}
NODES = REGIONS["left"] + REGIONS["right"]
EDGES = tuple(
    sorted(
        [
            (a + offset, b + offset)
            for offset in (0, 5)
            for a, b in ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))
        ]
        + [(0, 5)]
    )
)
REFERENCE_PHASE = tuple(math.tau * (i % 5) / 5 for i in NODES)
EPSILON = Q(1, 256)
HORIZON = Q(1, 16)
STEP_COUNTS = (64, 128, 256)
MODEL = RelationalExchangeModel(1.0)
FINGERPRINT_PATHS = (
    "benchmarks/relational_region_interaction.py",
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
    "src/tnfr/mathematics/_weight_normalization.py",
    "src/tnfr/utils/graph.py",
    "src/tnfr/utils/numeric.py",
    "src/tnfr/physics/winding_certificates.py",
    "src/tnfr/physics/support_transport.py",
    "src/tnfr/physics/structural_diffusion.py",
    "src/tnfr/physics/_conductance.py",
)


def prepare_prediction():
    """Declare the native protocol without evaluating a temporal response."""
    h = 1 + 2 * math.cos(math.tau / 5)
    acceleration = {
        "form": float(EPSILON) * (1 / 36 - 1 / (4 * math.pi**2 * h**2)),
        "phase": -float(EPSILON) / (12 * math.pi * h),
    }
    return {
        "protocol": "relational-region-interaction-v1",
        "nodes": NODES,
        "edges": EDGES,
        "regions": dict(REGIONS),
        "initial_form": tuple(EPSILON if i == 1 else Q(0) for i in NODES),
        "initial_phase": REFERENCE_PHASE,
        "capacity": (1.0,) * 10,
        "model": {"epi_weight": 0.5, "phase_weight": 0.5, "storage_scale": 1.0},
        "horizon": HORIZON,
        "step_counts": STEP_COUNTS,
        "timesteps": tuple(HORIZON / steps for steps in STEP_COUNTS),
        "scenarios": ("joined", "left", "right"),
        "contrast": "joined_right_minus_separately_executed_right",
        "observables": "right_port_minus_other_four_mean_form_and_reference_relative_phase",
        "initial_acceleration_estimates": acceleration,
        "leading_response_estimates": {
            key: value * float(HORIZON) ** 2 / 2 for key, value in acceleration.items()
        },
        "gates": {
            "normalized_response_interval": (0.75, 1.25),
            "refinement_contraction": 0.75,
            "numerical_slack": 1e-12,
            "signal_to_last_grid_difference": 32,
            "minimum_acute_margin": math.pi / 20,
            "maximum_actual_balance_residual": 1e-12,
            "maximum_disconnected_right_drift": 1e-12,
            "winding": 1,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "precision": "binary64_states_exact_Fraction_observations_and_defects",
            "integrator": "production_step_relational_exchange_explicit_Euler",
            "pressure_path": "fused_canonical",
            "randomness": "none",
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in FINGERPRINT_PATHS
        },
        "scope": "prepared_conditional_interaction_not_formation_closed_regional_law_or_physical_validation",
    }


def _graph(scenario, *, impulse_node=1):
    nodes = NODES if scenario == "joined" else REGIONS[scenario]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        (i, j, {"weight": 1.0}) for i, j in EDGES if i in nodes and j in nodes
    )
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True, _t=0.0)
    for i in nodes:
        graph.nodes[i].update(
            EPI=float(EPSILON) if i == impulse_node else 0.0,
            theta=REFERENCE_PHASE[i],
            nu_f=1.0,
        )
    return graph


def _region_state(field, region):
    lookup = {node: i for i, node in enumerate(field.nodes)}
    form = tuple(Q(field.epi[lookup[node]]) for node in region)
    phase = tuple(
        Q(field.phase[lookup[node]]) - Q(REFERENCE_PHASE[node]) for node in region
    )
    mean_form, mean_phase = sum(form) / 5, sum(phase) / 5
    centered = tuple(value - mean_form for value in form) + tuple(
        value - mean_phase for value in phase
    )
    return {
        "form_mean": mean_form,
        "phase_error_mean": mean_phase,
        "chi_form": form[0] - sum(form[1:]) / 4,
        "chi_phase": phase[0] - sum(phase[1:]) / 4,
        "centered_state": centered,
        "centered_distance": math.hypot(*(float(value) for value in centered)),
    }


def _snapshot(graph, field, *, time):
    regions = {
        name: nodes for name, nodes in REGIONS.items() if set(nodes) <= set(field.nodes)
    }
    result = {
        "time": time,
        "nodes": field.nodes,
        "epi": field.epi,
        "phase": field.phase,
        "capacity": field.capacity,
        "pressure": field.pressure,
        "phase_source": field.phase_source,
        "form_rate": field.form_rate,
        "phase_rate": field.phase_rate,
        "storage": field.storage,
        "balance_residual": field.balance_residual,
        "regions": {
            name: _region_state(field, nodes) for name, nodes in regions.items()
        },
    }
    if len(regions) == 2:
        # The read-only transport owner consumes stored pressure; use the fresh
        # field on a detached graph, retaining its native split discrepancy.
        detached = graph.copy()
        for node, pressure in zip(field.nodes, field.pressure, strict=True):
            detached.nodes[node]["delta_nfr"] = pressure
        source = observe_support_transport(detached)
        forcing = tuple(
            Q(MODEL.phase_weight) * Q(value) for value in field.phase_source
        )
        names = (
            "mean",
            "variance",
            "outward_cut_current",
            "mass_boundary_rate",
            "mass_forcing_rate",
            "mass_defect_rate",
            "internal_dissipation",
            "variance_boundary_rate",
            "variance_forcing_rate",
            "variance_defect_rate",
            "mass_identity_residual",
            "variance_identity_residual",
        )
        result["regional_transport"] = {
            name: {key: getattr(balance, key) for key in names}
            for name, nodes in regions.items()
            for balance in (
                observe_regional_support_balance(
                    source, nodes, epi_weight=Q(1, 2), forcing=forcing
                ),
            )
        }
    return result


def _trace(scenario, steps):
    graph = _graph(scenario)
    field = evaluate_relational_exchange(graph, model=MODEL)
    checkpoints = {"0": _snapshot(graph, field, time=0.0)}
    initial = field
    cycles = tuple(
        nodes for nodes in REGIONS.values() if set(nodes) <= set(field.nodes)
    )
    minimum_margin = math.inf
    maximum_balance = Q(0)
    maximum_step_defect = Q(0)
    maximum_clock_defect = Q(0)
    maximum_control_drift = 0.0
    winding_preserved = held = True
    dt = float(HORIZON / steps)
    for index in range(steps + 1):
        if index:
            step = step_relational_exchange(graph, model=MODEL, dt=dt)
            field = step.after
            maximum_step_defect = max(maximum_step_defect, abs(step.energy_step_defect))
            maximum_clock_defect = max(maximum_clock_defect, abs(step.clock_defect))
        for cycle in cycles:
            certificate = certify_phase_winding(graph, cycle)
            winding_preserved = (
                winding_preserved
                and certificate.is_defined
                and certificate.winding == 1
            )
        margins = [
            math.pi / 2
            - abs(
                math.remainder(
                    graph.nodes[j]["theta"] - graph.nodes[i]["theta"], math.tau
                )
            )
            for i, j in field.edges
        ]
        minimum_margin = min(minimum_margin, *margins)
        maximum_balance = max(maximum_balance, abs(field.balance_residual))
        held = (
            held
            and field.capacity == initial.capacity
            and field.edges == initial.edges
            and field.pressure_path == "fused_canonical"
        )
        if scenario == "right":
            differences = tuple(
                Q(a) - Q(b)
                for a, b in zip(
                    field.epi + field.phase, initial.epi + initial.phase, strict=True
                )
            )
            maximum_control_drift = max(
                maximum_control_drift,
                math.hypot(*(float(value) for value in differences)),
            )
        if index in (steps // 2, steps):
            checkpoints[str(index)] = _snapshot(graph, field, time=graph.graph["_t"])
    return {
        "steps": steps,
        "dt": dt,
        "checkpoints": checkpoints,
        "minimum_acute_margin": minimum_margin,
        "maximum_actual_balance_residual": maximum_balance,
        "maximum_energy_step_defect": maximum_step_defect,
        "maximum_clock_defect": maximum_clock_defect,
        "maximum_disconnected_right_drift": (
            maximum_control_drift if scenario == "right" else None
        ),
        "winding_preserved": winding_preserved,
        "capacity_support_and_path_held": held,
        "observed_state_count": steps + 1,
    }


def evaluate_prediction(prediction):
    """Execute the frozen preparation once, retaining every declared outcome."""
    if _encoded(prediction) != _encoded(prepare_prediction()):
        raise ValueError("prediction must match the current frozen protocol and source")
    traces = {
        scenario: {str(count): _trace(scenario, count) for count in STEP_COUNTS}
        for scenario in ("joined", "left", "right")
    }
    gates = prediction["gates"]
    lower, upper = gates["normalized_response_interval"]
    responses, refinements = {}, {}
    for count in STEP_COUNTS:
        joined = traces["joined"][str(count)]["checkpoints"][str(count)]["regions"][
            "right"
        ]
        control = traces["right"][str(count)]["checkpoints"][str(count)]["regions"][
            "right"
        ]
        responses[str(count)] = {}
        for name in ("form", "phase"):
            response = joined[f"chi_{name}"] - control[f"chi_{name}"]
            ratio = float(response) / prediction["leading_response_estimates"][name]
            responses[str(count)][name] = {
                "difference": response,
                "normalized_response": ratio,
            }
    for name in ("form", "phase"):
        values = tuple(
            responses[str(count)][name]["difference"] for count in STEP_COUNTS
        )
        first, last = abs(values[1] - values[0]), abs(values[2] - values[1])
        refinements[name] = {
            "coarse_to_middle": first,
            "middle_to_fine": last,
            "contracting": last
            <= gates["refinement_contraction"] * first + gates["numerical_slack"],
            "signal_resolved": abs(values[-1])
            > gates["signal_to_last_grid_difference"]
            * (last + gates["numerical_slack"]),
        }
    all_traces = tuple(
        trace for scenario in traces.values() for trace in scenario.values()
    )
    checks = {
        "predicted_response_bands": all(
            lower <= row["normalized_response"] <= upper
            for value in responses.values()
            for row in value.values()
        ),
        "refinement_and_signal": all(
            row["contracting"] and row["signal_resolved"]
            for row in refinements.values()
        ),
        "acute_winding_and_held_support": all(
            trace["minimum_acute_margin"] >= gates["minimum_acute_margin"]
            and trace["winding_preserved"]
            and trace["capacity_support_and_path_held"]
            for trace in all_traces
        ),
        "actual_balance_residual": all(
            trace["maximum_actual_balance_residual"]
            <= gates["maximum_actual_balance_residual"]
            for trace in all_traces
        ),
        "disconnected_receiver_control": all(
            trace["maximum_disconnected_right_drift"]
            <= gates["maximum_disconnected_right_drift"]
            for trace in traces["right"].values()
        ),
    }
    return {
        "prediction": prediction,
        "traces": traces,
        "responses": responses,
        "refinements": refinements,
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "finite_prepared_transmission_Taylor_and_grid_checks_not_ODE_error_bounds_or_closed_region_dynamics",
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
        raise FileExistsError("refusing to replace a retained interaction response")
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

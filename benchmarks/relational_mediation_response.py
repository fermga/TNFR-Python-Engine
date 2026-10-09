"""Freeze and evaluate one finite, matched-Euler mediated response.

The analytic equilibrium Jacobian predicts before the reserved native response
is accessed. Eliminated mediator coordinates are retained through their derived
linear memory realization. No kernel, gate, amplitude or clock is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import zipfile
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import numpy as np

from tnfr.dynamics import _euler_kernel as euler_owner
from tnfr.dynamics import relational as engine_owner
from tnfr.dynamics._euler_kernel import euler_update
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)
from tnfr.mathematics import linear_observation as memory_owner
from tnfr.mathematics.linear_observation import derive_coordinate_memory
from tnfr.utils import io as json_owner
from tnfr.utils.io import json_dumps

ROOT = Path(__file__).resolve().parents[1]
MODEL = RelationalExchangeModel(1.0)
NODES = tuple(range(11))
EDGES = tuple(
    sorted(
        {
            tuple(sorted((offset + i, offset + (i + 1) % 5)))
            for offset in (0, 5)
            for i in range(5)
        }
        | {(0, 10), (5, 10)}
    )
)
VISIBLE = tuple(range(10)) + tuple(range(11, 21))
EPSILON = Q(1, 64)
HORIZON = Q(1, 4)
STEP_COUNTS = (64, 128)
MEDIATOR_CAPACITIES = (0, 1, 2)
REFERENCE_PHASE = tuple(math.tau * (i % 5) / 5 if i < 10 else 0.0 for i in NODES)
INITIAL_FORM = (float(EPSILON),) + (0.0,) * 10
GATES = {
    "maximum_relative_receiver_error": 0.01,
    "numerical_slack": 1e-12,
    "minimum_predicted_signal_in_slack_units": 100,
    "maximum_zero_capacity_receiver": 1e-12,
    "minimum_acute_margin": math.pi / 20,
    "maximum_actual_balance_residual": 1e-12,
}


def _encoded(value):
    def exact(item):
        if isinstance(item, Q):
            return {"numerator": item.numerator, "denominator": item.denominator}
        raise TypeError(f"unsupported report value {type(item).__name__}")

    return (
        json_dumps(
            value,
            default=exact,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
            allow_nan=False,
        )
        + "\n"
    )


def _source_paths():
    return ("benchmarks/relational_mediation_response.py",) + tuple(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "src/tnfr").rglob("*.py"))
    )


def _admit_workspace():
    for module, relative in (
        (engine_owner, "src/tnfr/dynamics/relational.py"),
        (euler_owner, "src/tnfr/dynamics/_euler_kernel.py"),
        (memory_owner, "src/tnfr/mathematics/linear_observation.py"),
        (json_owner, "src/tnfr/utils/io.py"),
    ):
        if Path(module.__file__).resolve() != (ROOT / relative).resolve():
            raise ValueError("execution must import the fingerprinted workspace owners")


def _capacity(mu):
    if type(mu) is not int or mu not in MEDIATOR_CAPACITIES:
        raise ValueError("mediator capacity must belong to the frozen protocol")
    return (1.0,) * 10 + (float(mu),)


def _generator(mu):
    """Derive the joint tangent using represented cosine and inverse-pi inputs."""
    capacity = tuple(map(Q, _capacity(mu)))
    cosine, rho = Q(math.cos(math.tau / 5)), Q(1 / math.pi)
    laplacian = [[Q(0)] * 11 for _ in NODES]
    hessian = [[Q(0)] * 11 for _ in NODES]
    degrees, strengths = [0] * 11, [Q(0)] * 11
    for left, right in EDGES:
        stiffness = Q(1) if 10 in (left, right) else cosine
        for matrix, weight in ((laplacian, Q(1)), (hessian, stiffness)):
            matrix[left][left] += weight
            matrix[right][right] += weight
            matrix[left][right] -= weight
            matrix[right][left] -= weight
        for node in (left, right):
            degrees[node] += 1
            strengths[node] += stiffness
    e, w, beta = map(Q, (MODEL.epi_weight, MODEL.phase_weight, MODEL.storage_scale))
    return tuple(
        tuple(-e * capacity[i] * value / degrees[i] for value in laplacian[i])
        + tuple(-w * capacity[i] * rho * value / strengths[i] for value in hessian[i])
        for i in NODES
    ) + tuple(
        tuple(
            w * capacity[i] * rho * value / (beta * strengths[i])
            for value in laplacian[i]
        )
        + (Q(0),) * 11
        for i in NODES
    )


def _mv(matrix, vector):
    return tuple(
        math.fsum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix
    )


def _difference(left, right):
    return tuple(float(Q(a) - Q(b)) for a, b in zip(left, right, strict=True))


def _linf(values):
    return max(map(abs, values))


def _pair(state, node):
    return state[node], state[11 + node]


def _quasistatic_hidden(visible):
    # For the admitted positive-mu two-coordinate block, -D^-1 C y equals
    # the arithmetic average of the two port form/phase coordinates.
    return ((visible[0] + visible[5]) / 2, (visible[10] + visible[15]) / 2)


def predict(mu, steps):
    """Advance the fixed tangent and its omitted-memory control, never a graph."""
    if type(steps) is not int or steps not in STEP_COUNTS:
        raise ValueError("step count must belong to the frozen protocol")
    memory = derive_coordinate_memory(_generator(mu), VISIBLE)
    a, b, c, d = (
        tuple(tuple(map(float, row)) for row in matrix)
        for matrix in (
            memory.visible_generator,
            memory.hidden_to_visible,
            memory.visible_to_hidden,
            memory.hidden_generator,
        )
    )
    initial = INITIAL_FORM + (0.0,) * 11
    y = tuple(initial[i] for i in memory.visible_indices)
    hidden = tuple(initial[i] for i in memory.hidden_indices)
    zero_memory = y
    quasistatic = y if mu else None
    dt = float(HORIZON / steps)
    for _ in range(steps):
        visible_rate = tuple(
            u + v for u, v in zip(_mv(a, y), _mv(b, hidden), strict=True)
        )
        hidden_rate = tuple(
            u + v for u, v in zip(_mv(c, y), _mv(d, hidden), strict=True)
        )
        zero_rate = _mv(a, zero_memory)
        if quasistatic is not None:
            quasi_rate = tuple(
                u + v
                for u, v in zip(
                    _mv(a, quasistatic),
                    _mv(b, _quasistatic_hidden(quasistatic)),
                    strict=True,
                )
            )
            quasistatic = tuple(
                euler_update(value, dt, rate)
                for value, rate in zip(quasistatic, quasi_rate, strict=True)
            )
        # All three rates use the old states. In particular newly generated
        # mediator form/phase cannot affect the visible state in the same step.
        y, hidden, zero_memory = (
            tuple(
                euler_update(value, dt, rate)
                for value, rate in zip(values, rates, strict=True)
            )
            for values, rates in (
                (y, visible_rate),
                (hidden, hidden_rate),
                (zero_memory, zero_rate),
            )
        )
    state = [0.0] * 22
    for indices, values in (
        (memory.visible_indices, y),
        (memory.hidden_indices, hidden),
    ):
        for i, value in zip(indices, values, strict=True):
            state[i] = value
    zero_receiver = (zero_memory[VISIBLE.index(5)], zero_memory[VISIBLE.index(16)])
    return {
        "steps": steps,
        "dt": dt,
        "state_deviation": tuple(state),
        "epi": tuple(state[:11]),
        "phase": tuple(
            reference + value
            for reference, value in zip(REFERENCE_PHASE, state[11:], strict=True)
        ),
        "receiver_port": _pair(state, 5),
        "mediator": _pair(state, 10),
        "zero_memory_visible": zero_memory,
        "zero_memory_receiver_port": zero_receiver,
        "hidden_initial": tuple(initial[i] for i in memory.hidden_indices),
        "quasistatic_visible": quasistatic,
        "quasistatic_receiver_port": (
            None if quasistatic is None else (quasistatic[5], quasistatic[15])
        ),
        "quasistatic_hidden_initial": (
            None if mu == 0 else _quasistatic_hidden(tuple(initial[i] for i in VISIBLE))
        ),
    }


def prepare_prediction():
    """Declare forecasts, gates and source/runtime identity without native access."""
    _admit_workspace()
    return {
        "protocol": "relational-mediation-response-v1",
        "nodes": NODES,
        "edges": EDGES,
        "reference_phase": REFERENCE_PHASE,
        "initial_phase": REFERENCE_PHASE,
        "initial_form": INITIAL_FORM,
        "epsilon": EPSILON,
        "horizon": HORIZON,
        "mediator_capacities": MEDIATOR_CAPACITIES,
        "step_counts": STEP_COUNTS,
        "model": {
            "epi_weight": MODEL.epi_weight,
            "phase_weight": MODEL.phase_weight,
            "storage_scale": MODEL.storage_scale,
            "phase_domain": MODEL.phase_domain,
        },
        "coefficient_constants": {
            "cosine_hex": math.cos(math.tau / 5).hex(),
            "inverse_pi_hex": (1 / math.pi).hex(),
        },
        "linear_models": {
            str(mu): {
                "capacity": _capacity(mu),
                "generator": _generator(mu),
                "visible_indices": VISIBLE,
                "hidden_indices": (10, 21),
            }
            for mu in MEDIATOR_CAPACITIES
        },
        "forecasts": {
            str(mu): {str(count): predict(mu, count) for count in STEP_COUNTS}
            for mu in MEDIATOR_CAPACITIES
        },
        "gates": dict(GATES),
        "comparison": "receiver_port_pair_linf_against_same_grid_native_Euler; mu2_minus_mu1; zero_capacity_control",
        "controls": [
            "omit_both_hidden_mediator_coordinates_and_memory",
            "zero_mediator_capacity",
            "positive_capacity_quasistatic_mediator_with_equilibrated_initial_hidden_state",
        ],
        "quasistatic_scope": "h=-D^-1Cy=(port0+port5)/2; distinct_initial_hidden_state; visible_generator_independent_of_positive_mu; not_exact_all_state_dynamics",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "randomness": "none",
            "precision": "binary64_recurrence; exact_rational_block_decomposition",
            "integrator": "production_step_relational_exchange_explicit_Euler",
            "pressure_path": "fused_canonical",
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in _source_paths()
        },
        "source_archive_scope": "all_tnfr_Python_sources_and_this_producer; external_dependencies_identified_by_versions",
        "scope": (
            "fixed_prepared_geometry_and_law; receiver_and_mediator_tangent_remainder_is_conditionally_O(epsilon^3) "
            "by_odd_symmetry_on_fixed_finite_admitted_horizons; full_state_can_have_O(epsilon^2)_terms; "
            "no_certified_remainder_constant_or_neighborhood; represented_coefficients_and_reference_drift_are_not_ideal_trigonometric_certificates; "
            "no_ODE_accuracy_or_persistence_or_primitive_edge_birth_or_physical_identification_claim"
        ),
    }


def _graph(mu, *, reference=False):
    graph = nx.Graph()
    graph.add_nodes_from(NODES)
    graph.add_edges_from(EDGES, weight=1.0)
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True, _t=0.0)
    for i, phase, capacity in zip(NODES, REFERENCE_PHASE, _capacity(mu), strict=True):
        graph.nodes[i].update(
            EPI=0.0 if reference else INITIAL_FORM[i], theta=phase, nu_f=capacity
        )
    return graph


def _snapshot(field, time):
    state = field.epi + _difference(field.phase, REFERENCE_PHASE)
    return {
        "time": time,
        "epi": field.epi,
        "phase": field.phase,
        "state_deviation": state,
        "receiver_port": _pair(state, 5),
        "mediator": _pair(state, 10),
        "storage": field.storage,
        "continuous_loss": field.continuous_loss,
    }


def _reference_probe(mu):
    field = evaluate_relational_exchange(_graph(mu, reference=True), model=MODEL)
    return {
        "form_rate": field.form_rate,
        "phase_rate": field.phase_rate,
        "rate_linf": _linf(field.form_rate + field.phase_rate),
        "scope": "static_materialized_reference_drift_not_a_propagated_error_bound",
    }


def _trace(mu, count):
    graph = _graph(mu)
    dt = float(HORIZON / count)
    field = evaluate_relational_exchange(graph, model=MODEL)
    initial = _snapshot(field, graph.graph["_t"])
    maximum = {
        name: Q(0)
        for name in (
            "balance_residual",
            "pressure_split_residual",
            "nodal_rate_rounding_defect",
            "phase_rate_rounding_defect",
            "form_work_residual",
            "phase_work_residual",
            "energy_step_defect",
            "clock_defect",
            "epi_update_defect",
            "phase_update_defect",
        )
    }

    def inspect(current):
        for name in (
            "balance_residual",
            "pressure_split_residual",
            "nodal_rate_rounding_defect",
            "phase_rate_rounding_defect",
        ):
            value = getattr(current, name)
            maximum[name] = max(
                maximum[name],
                *(map(abs, value) if isinstance(value, tuple) else (abs(value),)),
            )
        for name in ("form", "phase"):
            maximum[name + "_work_residual"] = max(
                maximum[name + "_work_residual"],
                *map(abs, getattr(current.work, name + "_residual")),
            )
        margin = min(
            math.pi / 2
            - abs(math.remainder(current.phase[j] - current.phase[i], math.tau))
            for i, j in EDGES
        )
        held = (
            current.nodes == NODES
            and current.edges == EDGES
            and current.capacity == _capacity(mu)
            and current.pressure_path == "fused_canonical"
            and current.clipping == "none"
            and current.model == MODEL
            and tuple((i, j, graph.edges[i, j].get("weight")) for i, j in EDGES)
            == tuple((i, j, 1.0) for i, j in EDGES)
        )
        return margin, held

    minimum_margin, held = inspect(field)
    completed, stop = 0, None
    for index in range(1, count + 1):
        try:
            step = step_relational_exchange(graph, model=MODEL, dt=dt)
        except Exception as error:
            stop = {"attempted_step": index, "error": repr(error)}
            break
        completed, field = index, step.after
        margin, admitted = inspect(field)
        minimum_margin, held = min(minimum_margin, margin), held and admitted
        for name in ("energy_step_defect", "clock_defect"):
            maximum[name] = max(maximum[name], abs(getattr(step, name)))
        for name in ("epi_update_defect", "phase_update_defect"):
            maximum[name] = max(maximum[name], *map(abs, getattr(step, name)))
    return {
        "requested_steps": count,
        "completed_steps": completed,
        "dt": dt,
        "stop": stop,
        "initial": initial,
        "endpoint": _snapshot(field, graph.graph["_t"]),
        "minimum_acute_margin": minimum_margin,
        "capacity_support_and_path_held": held,
        "maximum_defects": maximum,
    }


def _comparison(actual, forecast, gates):
    signal = _linf(forecast)
    error = _linf(_difference(actual, forecast))
    resolved = (
        signal
        > gates["minimum_predicted_signal_in_slack_units"] * gates["numerical_slack"]
    )
    return {
        "actual": actual,
        "predicted": forecast,
        "predicted_linf": signal,
        "error_linf": error,
        "signal_resolved": resolved,
        "passed": resolved
        and error
        <= gates["maximum_relative_receiver_error"] * signal + gates["numerical_slack"],
    }


def evaluate_prediction(prediction):
    """Access the reserved fields/steps only after authenticating the frozen recipe."""
    if _encoded(prediction) != _encoded(prepare_prediction()):
        raise ValueError(
            "prediction must match current protocol, coefficients and source"
        )
    traces, comparisons, differences = {}, {}, {}
    reference = {str(mu): _reference_probe(mu) for mu in MEDIATOR_CAPACITIES}
    gates = prediction["gates"]
    for mu in MEDIATOR_CAPACITIES:
        key = str(mu)
        traces[key], comparisons[key] = {}, {}
        for count in STEP_COUNTS:
            step_key = str(count)
            trace = _trace(mu, count)
            traces[key][step_key] = trace
            if trace["completed_steps"] != count:
                comparisons[key][step_key] = {"available": False, "passed": False}
                continue
            endpoint = trace["endpoint"]
            forecast = prediction["forecasts"][key][step_key]
            row = _comparison(
                endpoint["receiver_port"], forecast["receiver_port"], gates
            )
            row.update(
                available=True,
                zero_memory_prediction=forecast["zero_memory_receiver_port"],
                zero_memory_error_linf=_linf(
                    _difference(
                        endpoint["receiver_port"], forecast["zero_memory_receiver_port"]
                    )
                ),
                mediator_error_linf=_linf(
                    _difference(endpoint["mediator"], forecast["mediator"])
                ),
                full_state_error_linf=_linf(
                    _difference(
                        endpoint["state_deviation"], forecast["state_deviation"]
                    )
                ),
            )
            row["quasistatic_prediction"] = forecast["quasistatic_receiver_port"]
            row["quasistatic_error_linf"] = (
                None
                if mu == 0
                else _linf(
                    _difference(
                        endpoint["receiver_port"], forecast["quasistatic_receiver_port"]
                    )
                )
            )
            if mu == 0:
                row["passed"] = (
                    _linf(endpoint["receiver_port"])
                    <= gates["maximum_zero_capacity_receiver"]
                )
                row["scope"] = "zero_capacity_control_not_positive_capacity_recovery"
            comparisons[key][step_key] = row
    complete = all(
        row["completed_steps"] == row["requested_steps"]
        for rows in traces.values()
        for row in rows.values()
    )
    if complete:
        for count in STEP_COUNTS:
            key = str(count)
            actual = _difference(
                traces["2"][key]["endpoint"]["receiver_port"],
                traces["1"][key]["endpoint"]["receiver_port"],
            )
            forecast = _difference(
                prediction["forecasts"]["2"][key]["receiver_port"],
                prediction["forecasts"]["1"][key]["receiver_port"],
            )
            differences[key] = _comparison(actual, forecast, gates)
            differences[key]["quasistatic_prediction"] = _difference(
                prediction["forecasts"]["2"][key]["quasistatic_receiver_port"],
                prediction["forecasts"]["1"][key]["quasistatic_receiver_port"],
            )
            differences[key]["quasistatic_error_linf"] = _linf(
                _difference(actual, differences[key]["quasistatic_prediction"])
            )
    grid = {}
    if complete:
        low, high = map(str, STEP_COUNTS)
        for mu in map(str, MEDIATOR_CAPACITIES):
            grid[mu] = {
                "native_state_linf": _linf(
                    _difference(
                        traces[mu][low]["endpoint"]["state_deviation"],
                        traces[mu][high]["endpoint"]["state_deviation"],
                    )
                ),
                "forecast_state_linf": _linf(
                    _difference(
                        prediction["forecasts"][mu][low]["state_deviation"],
                        prediction["forecasts"][mu][high]["state_deviation"],
                    )
                ),
            }
    checks = {
        "complete_response": complete,
        "receiver_prediction_and_zero_capacity_control": all(
            row["passed"] for rows in comparisons.values() for row in rows.values()
        ),
        "capacity_difference_prediction": complete
        and all(row["passed"] for row in differences.values()),
        "held_state_and_acute_margin": all(
            row["capacity_support_and_path_held"]
            and row["minimum_acute_margin"] >= gates["minimum_acute_margin"]
            for rows in traces.values()
            for row in rows.values()
        ),
        "actual_balance_residual": all(
            row["maximum_defects"]["balance_residual"]
            <= gates["maximum_actual_balance_residual"]
            for rows in traces.values()
            for row in rows.values()
        ),
    }
    return {
        "prediction": prediction,
        "reference_probe": reference,
        "traces": traces,
        "comparisons": comparisons,
        "capacity_differences": differences,
        "grid_differences": grid,
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "finite_same_grid_receiver_prediction; mediator_and_full_state_errors_reported_separately; grid_differences_are_not_validated_ODE_error_bounds",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    frozen, archive = args.output.with_suffix(
        ".prediction.json"
    ), args.output.with_suffix(".sources.zip")
    if args.output.exists():
        raise FileExistsError("refusing to replace a retained response")
    if args.prepare and (frozen.exists() or archive.exists()):
        raise FileExistsError(
            "refusing to replace retained prediction or source archive"
        )
    prediction = prepare_prediction()
    if args.prepare:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        captured = {
            path: (ROOT / path).read_bytes() for path in prediction["source_sha256"]
        }
        if any(
            hashlib.sha256(data).hexdigest() != prediction["source_sha256"][path]
            for path, data in captured.items()
        ):
            raise ValueError("source changed while preparing the frozen archive")
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
            for path, data in captured.items():
                bundle.writestr(path, data)
        with frozen.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_encoded(prediction))
        print(f"Prediction and package sources frozen: {frozen}")
        return 0
    if frozen.read_text(encoding="utf-8") != _encoded(prediction):
        raise ValueError("freeze the current protocol before evaluating it")
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(prediction["source_sha256"]) or set(
            bundle.namelist()
        ) != set(prediction["source_sha256"]):
            raise ValueError("frozen source archive membership differs from prediction")
        for path, digest in prediction["source_sha256"].items():
            if hashlib.sha256(bundle.read(path)).hexdigest() != digest:
                raise ValueError("frozen source archive does not match prediction")
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
        json_dumps(
            {"passed": report["passed"], "checks": report["checks"]}, allow_nan=False
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

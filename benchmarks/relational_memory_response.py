"""Prospective, matched-Euler memory prediction of one prepared native response.

The Taylor coefficients are derived before execution. This finite comparison
does not certify continuous-time error, exact compression or physical identity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import platform
import zipfile

import networkx as nx
import numpy as np

from benchmarks import relational_memory_prediction as prediction_owner
from benchmarks import relational_local_composition as composition_owner
from tnfr.dynamics import relational as engine_owner
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = RelationalExchangeModel(1.0)
SOURCE_PATHS = (
    "benchmarks/relational_memory_response.py",
    "benchmarks/relational_memory_prediction.py",
    "benchmarks/relational_local_composition.py",
) + tuple(
    path.relative_to(ROOT).as_posix()
    for path in sorted((ROOT / "src/tnfr").rglob("*.py"))
)


def _encoded(value):
    def exact(item):
        if isinstance(item, Q):
            return {"numerator": item.numerator, "denominator": item.denominator}
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(f"unsupported report value {type(item).__name__}")

    return (
        json.dumps(value, default=exact, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _mv(matrix, vector):
    return tuple(
        sum((Q(a) * Q(b) for a, b in zip(row, vector, strict=True)), Q(0))
        for row in matrix
    )


def _difference(left, right):
    return tuple(Q(a) - Q(b) for a, b in zip(left, right, strict=True))


def _norm(matrix, vector):
    square = sum((a * b for a, b in zip(vector, _mv(matrix, vector))), Q(0))
    if square < 0:
        raise ValueError("the declared energy norm must be nonnegative")
    return math.sqrt(float(square))


def prepare_prediction():
    """Freeze coefficient predictions and acceptance rules without engine evolution."""
    for module, relative in (
        (prediction_owner, "benchmarks/relational_memory_prediction.py"),
        (composition_owner, "benchmarks/relational_local_composition.py"),
        (engine_owner, "src/tnfr/dynamics/relational.py"),
    ):
        if Path(module.__file__).resolve() != (ROOT / relative).resolve():
            raise ValueError("execution must import the fingerprinted workspace owners")
    forecasts = {}
    for count in prediction_owner.STEP_COUNTS:
        native = prediction_owner.predict(count)
        high = prediction_owner.predict(count, decimal_precision=50)
        forecasts[str(count)] = {
            "binary64": asdict(native),
            "decimal50": asdict(high),
            "maximum_coefficient_evaluation_difference": max(
                abs(float(Q(a) - Q(b)))
                for name in (
                    "linear_visible",
                    "direct_visible",
                    "memory_visible",
                    "hidden_second_order",
                )
                for a, b in zip(getattr(native, name), getattr(high, name), strict=True)
            ),
        }
    return {
        "protocol": "prepared-relational-memory-v1",
        "nodes": prediction_owner.NODES,
        "edges": prediction_owner.EDGES,
        "initial_form": prediction_owner.INITIAL_FORM,
        "initial_phase": prediction_owner.INITIAL_PHASE,
        "reference_phase": prediction_owner.REFERENCE_PHASE,
        "capacity": (1.0,) * 10,
        "model": {"epi_weight": 0.5, "phase_weight": 0.5, "storage_scale": 1.0},
        "epsilon": prediction_owner.EPSILON,
        "horizon": prediction_owner.HORIZON,
        "step_counts": prediction_owner.STEP_COUNTS,
        "forecasts": forecasts,
        "comparison": "endpoint_C10_energy_norm_against_same_grid_native_Euler",
        "controls": ["linear", "cubic_without_hidden_feedback"],
        "gates": {
            "maximum_memory_to_each_control_error": 0.1,
            "numerical_slack": 1e-12,
            "minimum_control_error_in_slack_units": 100,
            "maximum_hidden_relative_error": 0.05,
            "minimum_acute_margin": math.pi / 20,
            "maximum_actual_balance_residual": 1e-12,
            "maximum_coefficient_evaluation_difference": 1e-12,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "randomness": "none",
            "precision": "binary64_engine; exact_represented_projections_and_norm_squares",
            "integrator": "production_step_relational_exchange_explicit_Euler",
            "pressure_path": "fused_canonical",
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in SOURCE_PATHS
        },
        "source_archive_scope": "all_tnfr_Python_sources_and_three_producers; external_dependencies_identified_by_versions",
        "scope": (
            "one_prepared_conditional_model; matched_Euler_amplitude_approximation; "
            "Decimal_replay_checks_evaluation_not_irrational_constant_or_ODE_error; "
            "no_fitted_kernel_no_runtime_compression_no_physical_identification"
        ),
    }


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from(prediction_owner.NODES)
    graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in prediction_owner.EDGES)
    graph.graph.update(GAMMA={"type": "none"}, vectorized_dnfr=True, _t=0.0)
    for i, x, phase in zip(
        prediction_owner.NODES,
        prediction_owner.INITIAL_FORM,
        prediction_owner.INITIAL_PHASE,
        strict=True,
    ):
        graph.nodes[i].update(EPI=float(x), theta=float(phase), nu_f=1.0)
    return graph


def _snapshot(field, time, geometry):
    state = tuple(map(Q, field.epi)) + _difference(
        field.phase, prediction_owner.REFERENCE_PHASE
    )
    return {
        "time": time,
        "epi": field.epi,
        "phase": field.phase,
        "visible": _mv(geometry.visible_projection, state),
        "hidden": _mv(geometry.hidden_projection, state),
        "storage": field.storage,
        "continuous_loss": field.continuous_loss,
    }


def _reference_probe(geometry):
    """Retain represented lock drift rather than declaring exact native symmetry."""
    graph = _graph()
    for i in prediction_owner.NODES:
        graph.nodes[i].update(EPI=0.0, theta=prediction_owner.REFERENCE_PHASE[i])
    field = evaluate_relational_exchange(graph, model=MODEL)
    rates = tuple(map(Q, field.form_rate + field.phase_rate))
    initial = tuple(map(Q, prediction_owner.INITIAL_FORM)) + _difference(
        prediction_owner.INITIAL_PHASE, prediction_owner.REFERENCE_PHASE
    )
    stipulated = tuple(
        Q(float(prediction_owner.EPSILON)) * Q(v) for v in geometry.initial_direction
    )
    return {
        "form_rate": field.form_rate,
        "phase_rate": field.phase_rate,
        "visible_rate_norm": _norm(
            geometry.visible_energy, _mv(geometry.visible_projection, rates)
        ),
        "hidden_rate_norm": _norm(
            geometry.hidden_energy, _mv(geometry.hidden_projection, rates)
        ),
        "maximum_preparation_materialization_difference": max(
            map(abs, _difference(initial, stipulated))
        ),
        "scope": "static_roundoff_observation_not_a_propagated_error_certificate",
    }


def _trace(count, geometry):
    graph = _graph()
    dt = float(prediction_owner.HORIZON / count)
    field = evaluate_relational_exchange(graph, model=MODEL)
    frames = {"0": _snapshot(field, 0.0, geometry)}
    maximum = {
        name: 0.0
        for name in (
            "balance_residual",
            "energy_step_defect",
            "clock_defect",
            "epi_update_defect",
            "phase_update_defect",
        )
    }
    minimum_margin, held, stop, completed = math.inf, True, None, 0

    def inspect(current):
        margin = min(
            math.pi / 2
            - abs(math.remainder(current.phase[j] - current.phase[i], math.tau))
            for i, j in current.edges
        )
        admitted = (
            current.nodes == prediction_owner.NODES
            and current.edges == prediction_owner.EDGES
            and current.capacity == (1.0,) * 10
            and current.pressure_path == "fused_canonical"
        )
        maximum["balance_residual"] = max(
            maximum["balance_residual"], abs(current.balance_residual)
        )
        return margin, admitted

    minimum_margin, held = inspect(field)
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
        if index in (count // 2, count):
            frames[str(index)] = _snapshot(field, graph.graph["_t"], geometry)
    return {
        "requested_steps": count,
        "completed_steps": completed,
        "dt": dt,
        "stop": stop,
        "checkpoints": frames,
        "minimum_acute_margin": minimum_margin,
        "capacity_support_and_path_held": held,
        "maximum_defects": maximum,
    }


def evaluate_prediction(prediction):
    """Execute the reserved response only after checking its complete frozen recipe."""
    if _encoded(prediction) != _encoded(prepare_prediction()):
        raise ValueError(
            "prediction must match current protocol, coefficients and source"
        )
    geometry = prediction_owner.prepare_geometry()
    reference_probe = _reference_probe(geometry)
    traces, errors = {}, {}
    gates = prediction["gates"]
    slack = gates["numerical_slack"]
    for count in prediction_owner.STEP_COUNTS:
        key = str(count)
        trace = _trace(count, geometry)
        traces[key] = trace
        if trace["completed_steps"] != count:
            errors[key] = {"available": False, "passed": False}
            continue
        frame = trace["checkpoints"][key]
        forecast = prediction["forecasts"][key]["binary64"]
        values = {
            name: _norm(
                geometry.visible_energy,
                _difference(frame["visible"], forecast[name + "_visible"]),
            )
            for name in ("linear", "direct", "memory")
        }
        hidden_error = _norm(
            geometry.hidden_energy,
            _difference(frame["hidden"], forecast["hidden_second_order"]),
        )
        hidden_signal = _norm(geometry.hidden_energy, frame["hidden"])
        values.update(
            available=True,
            hidden_error=hidden_error,
            hidden_signal=hidden_signal,
            passed=all(
                values["memory"]
                <= gates["maximum_memory_to_each_control_error"] * values[name] + slack
                and values[name] > gates["minimum_control_error_in_slack_units"] * slack
                for name in ("linear", "direct")
            )
            and hidden_signal > gates["minimum_control_error_in_slack_units"] * slack
            and hidden_error
            <= gates["maximum_hidden_relative_error"] * hidden_signal + slack,
        )
        errors[key] = values
    complete = all(
        row["completed_steps"] == row["requested_steps"] for row in traces.values()
    )
    grid_differences = []
    if complete:
        for a, b in zip(prediction_owner.STEP_COUNTS, prediction_owner.STEP_COUNTS[1:]):
            native = _difference(
                traces[str(a)]["checkpoints"][str(a)]["visible"],
                traces[str(b)]["checkpoints"][str(b)]["visible"],
            )
            forecast = _difference(
                prediction["forecasts"][str(a)]["binary64"]["memory_visible"],
                prediction["forecasts"][str(b)]["binary64"]["memory_visible"],
            )
            grid_differences.append(
                {
                    "steps": (a, b),
                    "native_visible": _norm(geometry.visible_energy, native),
                    "memory_forecast": _norm(geometry.visible_energy, forecast),
                }
            )
    checks = {
        "complete_response": complete,
        "memory_advantage_and_hidden_prediction": all(
            row["passed"] for row in errors.values()
        ),
        "held_state_and_acute_margin": all(
            row["capacity_support_and_path_held"]
            and row["minimum_acute_margin"] >= gates["minimum_acute_margin"]
            for row in traces.values()
        ),
        "actual_balance_residual": all(
            row["maximum_defects"]["balance_residual"]
            <= gates["maximum_actual_balance_residual"]
            for row in traces.values()
        ),
        "coefficient_evaluation_control": all(
            row["maximum_coefficient_evaluation_difference"]
            <= gates["maximum_coefficient_evaluation_difference"]
            for row in prediction["forecasts"].values()
        ),
    }
    return {
        "prediction": prediction,
        "reference_probe": reference_probe,
        "traces": traces,
        "errors": errors,
        "grid_differences": grid_differences,
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "finite_same_grid_endpoint_prediction; grid_differences_are_not_validated_ODE_error_bounds",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    prediction = prepare_prediction()
    frozen = args.output.with_suffix(".prediction.json")
    archive = args.output.with_suffix(".sources.zip")
    if args.prepare:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists() or archive.exists() or args.output.exists():
            raise FileExistsError("refusing to replace retained prediction or response")
        captured = {path: (ROOT / path).read_bytes() for path in SOURCE_PATHS}
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
        for path, digest in prediction["source_sha256"].items():
            if hashlib.sha256(bundle.read(path)).hexdigest() != digest:
                raise ValueError("frozen source archive does not match prediction")
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
                "checks": report["checks"],
                "errors": report["errors"],
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen exploratory TCLab forecasts; no physical measurement admission.

The driven models use normalized EPI transport on supplied support, a fixed
bath, and declared heater input. Positive rates and exchange symmetry are
constitutive premises, not a derivation of thermodynamics. The detached affine
forecast reuses the shared matrix exponential; it is not a live engine trace.
Only calibration acquisitions identify parameters. Evaluation supplies commands,
timestamps and the first sensor pair before later responses are scored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import networkx as nx
import numpy as np
import scipy
from scipy.optimize import least_squares

from tnfr.physics.observability import linear_observability_certificate
from tnfr.physics.spectral_projectors import matrix_exponential
from tnfr.physics.structural_diffusion import structural_diffusion_operator
from tnfr.research import current_git_source_provenance
from tnfr.validation.nodal_prediction import NodalMeasurementRun, _digest


PARAMETER_NAMES = {
    "memory": ("leak", "coupling", "backreaction", "sensor_rate", "gain"),
    "markov": ("leak", "coupling", "gain"),
}
START = {
    "memory": [1 / 150, 1 / 500, 1 / 30, 1 / 20, 0.004],
    "markov": [1 / 150, 1 / 500, 0.004],
}
FIT_POLICY = {
    "lower": 1e-6,
    "upper": 1.0,
    "rate_start_factors": [0.25, 1.0, 4.0],
    "max_nfev_per_start": 200,
    "tolerance": 1e-8,
    "objective": "unweighted trajectory residuals, both sensors, all calibration rows after initialization",
    "input_hold": "left-endpoint zero-order hold on recorded timestamps",
    "bath": "mean of first two sensor readings in each acquisition, held constant",
    "latent_initialization": "each heater equals its first sensor reading",
    "physical_status": "not_admitted",
}


def _write(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_split(calibration_entries, evaluation_entries):
    entries = list(calibration_entries) + list(evaluation_entries)
    if not calibration_entries or not evaluation_entries:
        raise ValueError("both acquisition groups must be nonempty")
    if len({x["name"] for x in entries}) != len(entries):
        raise ValueError("duplicate acquisition filename")
    if len({x["sha"] for x in entries}) != len(entries):
        raise ValueError("duplicate acquisition content or alias")


def load_csv(path, expected_blob_sha, *, responses=False):
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(100_001)
    if not 0 < len(raw) <= 100_000:
        raise ValueError("CSV outside bounded intake size")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != expected_blob_sha:
        raise ValueError("source Git blob identity mismatch")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if len(reader.fieldnames or ()) != 5 or set(reader.fieldnames or ()) != {
        "Time",
        "Q1",
        "Q2",
        "T1",
        "T2",
    }:
        raise ValueError("unexpected TCLab schema")
    rows = list(reader)
    if len(rows) < 2 or len(rows) > 2000 or any(None in row for row in rows):
        raise ValueError("unsupported row count")
    times = [float(row["Time"]) for row in rows]
    commands = [[float(row["Q1"]), float(row["Q2"])] for row in rows]
    initial = [float(rows[0]["T1"]), float(rows[0]["T2"])]
    # Shared measurement admission checks the clock and consumed initial state.
    NodalMeasurementRun(
        run_id=path.name,
        acquisition_id=path.name,
        channel_ids=("Q1", "Q2"),
        timestamps=tuple(times),
        samples=tuple(tuple(row[j] for row in commands) for j in range(2)),
        value_unit="command_percent",
        time_unit="s",
    )
    if (
        not np.all(np.isfinite(initial))
        or np.any(np.asarray(commands) < 0)
        or np.any(np.asarray(commands) > 100)
    ):
        raise ValueError("invalid initialization or command range")
    result = {
        "run_id": path.name,
        "blob_sha": blob,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "times": times,
        "commands": commands,
        "initial": initial,
    }
    if responses:
        observations = [[float(row["T1"]), float(row["T2"])] for row in rows]
        NodalMeasurementRun(
            run_id=path.name,
            acquisition_id=path.name,
            channel_ids=("T1", "T2"),
            timestamps=tuple(times),
            samples=tuple(tuple(row[j] for row in observations) for j in range(2)),
            value_unit="degree_C",
            time_unit="s",
        )
        result["observations"] = observations
    return result


def model_matrices(kind, parameters):
    if kind not in PARAMETER_NAMES:
        raise ValueError("unsupported model")
    p = np.asarray(parameters, dtype=float)
    if (
        p.shape != (len(PARAMETER_NAMES[kind]),)
        or not np.all(np.isfinite(p))
        or np.any(p <= 0)
    ):
        raise ValueError("model requires finite positive parameters")
    graph = nx.Graph()
    graph.graph.update(
        DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
        use_extended_dynamics=False,
    )
    if kind == "memory":
        leak, coupling, backreaction, sensor_rate, gain = p
        graph.add_nodes_from(range(5))
        for heater, sensor in ((0, 2), (1, 3)):
            graph.add_edge(heater, sensor, weight=1.0)
            graph.add_edge(heater, 4, weight=leak / backreaction)
        graph.add_edge(0, 1, weight=coupling / backreaction)
        capacity = np.array(
            [leak + coupling + backreaction] * 2 + [sensor_rate] * 2 + [0.0]
        )
        observation = np.eye(4)[[2, 3]]
    else:
        leak, coupling, gain = p
        graph.add_nodes_from(range(3))
        graph.add_edge(0, 1, weight=1.0)
        graph.add_edge(0, 2, weight=leak / coupling)
        graph.add_edge(1, 2, weight=leak / coupling)
        capacity = np.array([leak + coupling] * 2 + [0.0])
        observation = np.eye(2)
    for i, frequency in enumerate(capacity):
        graph.nodes[i].update(epi=0.0, nu_f=float(frequency), theta=0.0)
    nodes, laplacian = structural_diffusion_operator(graph)
    if list(nodes) != list(range(len(capacity))):
        raise ValueError("unexpected nodal order")
    full = -capacity[:, None] * laplacian
    generator = full[:-1, :-1]
    source = np.zeros((len(generator), 2))
    source[0, 0] = source[1, 1] = gain
    return graph, generator, source, observation


def predict(kind, parameters, times, commands, initial):
    _, generator, source, observation = model_matrices(kind, parameters)
    times = np.asarray(times, dtype=float)
    commands = np.asarray(commands, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or commands.shape != (len(times), 2)
        or initial.shape != (2,)
    ):
        raise ValueError("incompatible forecast coordinates")
    if not all(np.all(np.isfinite(x)) for x in (times, commands, initial)) or np.any(
        np.diff(times) <= 0
    ):
        raise ValueError("nonfinite data or invalid clock")
    ambient = float(np.mean(initial))
    state = np.tile(initial - ambient, 2) if kind == "memory" else initial - ambient
    dimension = len(state)
    augmented = np.zeros((dimension + 2, dimension + 2))
    augmented[:dimension, :dimension] = generator
    augmented[:dimension, dimension:] = source
    output = [initial.copy()]
    transitions = {}
    for dt, command in zip(np.diff(times), commands[:-1]):
        # Exact represented time differences are retained; no time quantization.
        key = float(dt)
        if key not in transitions:
            propagator = matrix_exponential(augmented * dt)
            transitions[key] = (
                propagator[:dimension, :dimension],
                propagator[:dimension, dimension:],
            )
        flow, drive = transitions[key]
        state = flow @ state + drive @ command
        output.append(observation @ state + ambient)
    result = np.asarray(output)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("unrepresentable forecast")
    return result


def fit_model(kind, runs):
    def residual(log_parameters):
        parameters = np.exp(log_parameters)
        return np.concatenate(
            [
                (
                    predict(
                        kind, parameters, run["times"], run["commands"], run["initial"]
                    )[1:]
                    - np.asarray(run["observations"])[1:]
                ).ravel()
                for run in runs
            ]
        )

    attempts = []
    for factor in FIT_POLICY["rate_start_factors"]:
        start = np.array(START[kind])
        start[:-1] *= factor
        fit = least_squares(
            residual,
            np.log(start),
            bounds=(np.log(FIT_POLICY["lower"]), np.log(FIT_POLICY["upper"])),
            max_nfev=FIT_POLICY["max_nfev_per_start"],
            ftol=FIT_POLICY["tolerance"],
            xtol=FIT_POLICY["tolerance"],
            gtol=FIT_POLICY["tolerance"],
            x_scale="jac",
        )
        attempts.append(
            (
                fit,
                {
                    "start_factor": factor,
                    "success": bool(fit.success),
                    "nfev": fit.nfev,
                    "cost": float(fit.cost),
                    "message": fit.message,
                    "parameters": np.exp(fit.x).tolist(),
                    "active_bounds": fit.active_mask.tolist(),
                },
            )
        )
    successful = [item for item in attempts if item[0].success]
    if not successful:
        raise RuntimeError(
            "all fixed-budget calibration starts failed: "
            + json.dumps([x[1] for x in attempts])
        )
    fit, best = min(successful, key=lambda item: item[0].cost)
    parameters = np.exp(fit.x)
    _, generator, _, observer = model_matrices(kind, parameters)
    observability = linear_observability_certificate(generator, observer)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    return {
        "parameters": parameters.tolist(),
        "parameter_names": PARAMETER_NAMES[kind],
        "calibration_rmse_C": float(np.sqrt(np.mean(fit.fun**2))),
        "attempts": [item[1] for item in attempts],
        "selected_start_factor": best["start_factor"],
        "jacobian_singular_values_log_parameters": singular.tolist(),
        "generator_eigenvalues": np.linalg.eigvals(generator).real.tolist(),
        "state_observability_rank": observability.rank,
        "state_dimension": len(generator),
        "physical_status": "not_admitted",
    }


def score_forecast(observations, prediction):
    measured = np.asarray(observations, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if (
        measured.shape != predicted.shape
        or measured.ndim != 2
        or measured.shape[1] != 2
    ):
        raise ValueError("score coordinates disagree")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(predicted)):
        raise ValueError("nonfinite scored observations")
    error = predicted[1:] - measured[1:]
    return {
        "rmse_C": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "mae_C": np.mean(np.abs(error), axis=0).tolist(),
        "bias_C": np.mean(error, axis=0).tolist(),
        "max_absolute_error_C": np.max(np.abs(error), axis=0).tolist(),
        "joint_rmse_C": float(np.sqrt(np.mean(error**2))),
    }


def execute(spec_path, data_directory, output):
    spec_path, data_directory, output = map(Path, (spec_path, data_directory, output))
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    if spec["fit_policy"] != FIT_POLICY or spec["initial_starts"] != START:
        raise ValueError("frozen numerical/model policy differs from executable")
    validate_split(spec["calibration"], spec["evaluation"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "spec.json").write_bytes(spec_bytes)
    calibration = [
        load_csv(data_directory / entry["name"], entry["sha"], responses=True)
        for entry in spec["calibration"]
    ]
    fits = {}
    for kind in PARAMETER_NAMES:
        fits[kind] = fit_model(kind, calibration)
        print(kind + " calibration complete", flush=True)
    _write(
        output / "calibration.json",
        {
            "fits": fits,
            "inputs": [
                {key: run[key] for key in ("run_id", "sha256", "blob_sha")}
                for run in calibration
            ],
        },
    )
    # Only timestamps, commands and the initial sensor pair reach prediction.
    issued = []
    for entry in spec["evaluation"]:
        context = load_csv(
            data_directory / entry["name"], entry["sha"], responses=False
        )
        forecasts = {
            kind: predict(
                kind,
                fit["parameters"],
                context["times"],
                context["commands"],
                context["initial"],
            ).tolist()
            for kind, fit in fits.items()
        }
        forecasts["persistence"] = [context["initial"]] * len(context["times"])
        issued.append({"context": context, "forecasts": forecasts})
    forecast_hash = _digest(issued)
    _write(output / "forecast.json", {"content_hash": forecast_hash, "issued": issued})
    # Retain every prediction before converting later responses for scoring.
    if (
        _digest(json.loads((output / "forecast.json").read_text())["issued"])
        != forecast_hash
    ):
        raise ValueError("issued forecast changed")
    scores = []
    for entry, issue in zip(spec["evaluation"], issued):
        run = load_csv(data_directory / entry["name"], entry["sha"], responses=True)
        if run["sha256"] != issue["context"]["sha256"]:
            raise ValueError("evaluation bytes changed after forecast")
        scores.append(
            {
                "run_id": run["run_id"],
                "sha256": run["sha256"],
                "scores": {
                    kind: score_forecast(run["observations"], values)
                    for kind, values in issue["forecasts"].items()
                },
            }
        )
    revision, dirty, dirty_hash = current_git_source_provenance(
        ROOT, ("src/tnfr", "benchmarks")
    )
    result = {
        "schema": "tnfr-tclab-exploration-v1",
        "scores": scores,
        "forecast_hash": forecast_hash,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "source": {
            "git_sha": revision,
            "dirty": dirty,
            "dirty_source_hash": dirty_hash,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "physical_status": "not_admitted",
        "measurement_verdict": "not_assessed",
        "scope": "file-separated exploratory conditional response; acquisition independence and physical map unverified",
        "solver": "shared approximate matrix exponential, held input per recorded interval; not a live graph trace",
        "numerical_error_bound": None,
        "sensor_error_bound": None,
        "clock_error_bound": None,
        "future_response_used_for_fit_or_state_correction": False,
        "whole_csv_bytes_available_before_issue": True,
        "post_initial_response_values_passed_to_predictor": False,
        "predictor_input": "entire recorded command/time schedule and first sensor pair; commanded input is conditioned on",
        "external_trusted_chronology": False,
    }
    _write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = execute(arguments.spec, arguments.data, arguments.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

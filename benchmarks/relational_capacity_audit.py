"""Read-only audit of the retained relational capacity response.

No temporal solver runs. Decisions and the K3 pressure identity are reconstructed
from three retained checkpoints per trace. Compatible source/runtime evidence
also permits detached field replay. Omitted steps, chronology, absence of other
runs and complete dependency provenance cannot be authenticated by this record.
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

from tnfr._exact_time import finite_represented_real
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)

if __package__ in (None, ""):
    import relational_capacity_response as campaign
else:
    from benchmarks import relational_capacity_response as campaign

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "docs/assets/relational_capacity_response"
AUDIT_TOLERANCE = Q(1, 10**12)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, expected, label):
    _require(set(value) == set(expected), f"{label} inventory differs")


def _scientific_spec(prediction):
    result = {
        key: value
        for key, value in prediction.items()
        if key not in ("source_sha256", "runtime")
    }
    result["runtime"] = {
        key: value
        for key, value in prediction["runtime"].items()
        if key not in ("python", "numpy", "networkx")
    }
    return campaign._encoded(result)


def _number(value, label):
    if type(value) in (Q, int):
        return Q(value)
    return finite_represented_real(value, label)[1]


def _vector(value, label):
    _require(
        isinstance(value, (tuple, list)) and len(value) == 3,
        f"invalid {label} dimensions",
    )
    return tuple(finite_represented_real(item, label)[0] for item in value)


def load_exact_json(path):
    """Load finite JSON with strictly typed exact-rational diagnostic payloads."""

    def decode(value):
        if set(value) == {"numerator", "denominator"}:
            n, d = value["numerator"], value["denominator"]
            _require(
                type(n) is int and type(d) is int and d > 0, "invalid exact fraction"
            )
            return Q(n, d)
        return value

    def invalid(value):
        raise ValueError(f"nonfinite JSON scalar: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_hook=decode,
        parse_constant=invalid,
    )


def _compatibility(prediction, root):
    declared = prediction["source_sha256"]
    source = {}
    for name in set(campaign.FINGERPRINT_PATHS) | set(declared):
        if name not in campaign.FINGERPRINT_PATHS or name not in declared:
            source[name] = "manifest coverage differs"
            continue
        path = (root / name).resolve()
        _require(path.is_relative_to(root), "source path leaves the audit root")
        digest = declared[name]
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            "invalid source digest",
        )
        actual = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if actual != digest:
            source[name] = {"recorded": digest, "current": actual}
    current = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "networkx": nx.__version__,
    }
    runtime = {
        key: {"recorded": prediction["runtime"].get(key), "current": value}
        for key, value in current.items()
        if prediction["runtime"].get(key) != value
    }
    return source, runtime


def _chi(phase):
    return phase[0] - (phase[1] + phase[2]) / 2


def _independent_fields(state, vectors, q, model, law):
    """K3 analytic oracle on the common lift, independent of current engine replay."""
    theta, nu, omega = (vectors[key] for key in ("phase", "capacity", "phase_rate"))
    beta, weight = Q(model.storage_scale), Q(model.phase_weight)
    metric, gradient = [], []
    for i in range(3):
        j, k = (node for node in range(3) if node != i)
        displacement = float((Q(theta[j]) + Q(theta[k])) / 2 - Q(theta[i]))
        spread = float((Q(theta[j]) - Q(theta[k])) / 2)
        sinc = math.sin(displacement) / displacement if displacement else 1.0
        metric.append(Q(2 * math.pi * math.cos(spread) * sinc))
        gradient.append(
            sum(Q(math.sin(float(Q(theta[i]) - Q(theta[n])))) for n in (j, k))
        )
    base = tuple(weight * Q(nu[i]) * q[i] / (beta * metric[i]) for i in range(3))
    expected_phase = list(base)
    if law == "counterpart":
        for i, j in campaign.EDGES:
            ni, nj = Q(nu[i]), Q(nu[j])
            mobility = ni * nj / (ni + nj) if ni + nj else Q(0)
            coefficient = (
                weight
                * mobility
                * (q[i] + q[j])
                * Q(math.sin(float(Q(theta[j]) - Q(theta[i]))))
                / (8 * beta)
            )
            expected_phase[i] += coefficient * gradient[j]
            expected_phase[j] -= coefficient * gradient[i]
    x = tuple(map(Q, vectors["epi"]))
    storage = sum(
        (x[i] - x[j]) ** 2 / 2
        + 2 * beta * Q(math.sin(float((Q(theta[j]) - Q(theta[i])) / 2))) ** 2
        for i, j in campaign.EDGES
    )
    expected = {
        "storage": storage,
        "actual_storage_rate": sum(
            q[i] * Q(vectors["form_rate"][i]) + beta * gradient[i] * Q(omega[i])
            for i in range(3)
        ),
        "extra_work": beta
        * sum(gradient[i] * (Q(omega[i]) - base[i]) for i in range(3)),
    }
    defects = [abs(_number(state[key], key) - value) for key, value in expected.items()]
    defects.extend(
        abs(Q(a) - b) for a, b in zip(vectors["phase_metric"], metric, strict=True)
    )
    defects.extend(abs(Q(a) - b) for a, b in zip(omega, expected_phase, strict=True))
    _require(
        max(defects) <= AUDIT_TOLERANCE,
        "checkpoint violates independent K3 metric/rate/work/storage",
    )


def _checkpoint(state, *, prediction, law, capacity, label, replay, model):
    _require(
        campaign._encoded({key: state[key] for key in ("nodes", "edges")})
        == campaign._encoded({key: prediction[key] for key in ("nodes", "edges")}),
        "checkpoint support differs",
    )
    vectors = {
        key: _vector(state[key], key)
        for key in (
            "epi",
            "phase",
            "capacity",
            "pressure",
            "form_rate",
            "phase_rate",
            "phase_metric",
            "phase_source",
        )
    }
    _require(
        vectors["capacity"] == tuple(prediction["capacities"][capacity]),
        "checkpoint capacity differs",
    )
    x, theta, nu, pressure, rate = (
        tuple(map(Q, vectors[key]))
        for key in ("epi", "phase", "capacity", "pressure", "form_rate")
    )
    _require(
        all(value >= 0 for value in nu) and nu[0] == 1, "invalid receiver capacity"
    )
    if label == "initial":
        _require(
            x == tuple(prediction["initial_epi_materialized"])
            and theta == tuple(prediction["initial_phase"]),
            "initial checkpoint differs from frozen preparation",
        )
    _require(
        all(abs(theta[j] - theta[i]) < Q(math.pi) / 2 for i, j in campaign.EDGES),
        "checkpoint leaves the prepared common acute lift",
    )
    _require(
        all(value > 0 for value in vectors["phase_metric"]),
        "nonpositive checkpoint metric",
    )
    epi_weight, phase_weight = Q(model.epi_weight), Q(model.phase_weight)
    q = tuple(3 * value - sum(x) for value in x)
    source = tuple((sum(theta) - 3 * value) / (2 * Q(math.pi)) for value in theta)
    pressure_defects = tuple(
        pressure[i] + epi_weight * q[i] / 2 - phase_weight * source[i] for i in range(3)
    )
    source_defects = tuple(
        Q(value) - expected
        for value, expected in zip(vectors["phase_source"], source, strict=True)
    )
    _require(
        max(map(abs, pressure_defects + source_defects)) <= AUDIT_TOLERANCE,
        "checkpoint violates independent K3 pressure/source identity",
    )
    for i in range(3):
        _require(
            vectors["form_rate"][i]
            == finite_represented_real(nu[i] * pressure[i], "nodal product")[0],
            "checkpoint violates the represented nodal row",
        )
    loss = epi_weight * sum(nu[i] * q[i] ** 2 for i in range(3)) / 2
    _require(
        _number(state["continuous_loss"], "continuous loss") == loss,
        "checkpoint loss differs from state",
    )
    work = _number(state["actual_storage_rate"], "actual work")
    balance = _number(state["actual_balance_residual"], "balance residual")
    _require(balance == work + loss, "checkpoint work balance bookkeeping differs")
    storage = _number(state["storage"], "storage")
    _require(
        storage >= sum((x[i] - x[j]) ** 2 for i, j in campaign.EDGES) / 2,
        "checkpoint storage below its form component",
    )
    extra = _number(state["extra_work"], "extra work")
    _independent_fields(state, vectors, q, model, law)
    replay_mismatches = []
    if replay:
        graph = nx.Graph()
        graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in campaign.EDGES)
        for i in campaign.NODES:
            graph.nodes[i].update(
                EPI=vectors["epi"][i],
                theta=vectors["phase"][i],
                nu_f=vectors["capacity"][i],
            )
        field = evaluate_relational_exchange(graph, model=model)
        actual_phase, expected_extra, expected_work = campaign._actual_rates(field, law)
        expected = {
            key: getattr(field, key)
            for key in (
                "pressure",
                "form_rate",
                "phase_metric",
                "phase_source",
                "storage",
                "continuous_loss",
            )
        }
        expected.update(
            phase_rate=actual_phase,
            extra_work=expected_extra,
            actual_storage_rate=expected_work,
            actual_balance_residual=expected_work + field.continuous_loss,
        )
        for key, value in expected.items():
            observed = tuple(state[key]) if isinstance(value, tuple) else state[key]
            if observed != value:
                replay_mismatches.append(key)
        if field.pressure_path != prediction["runtime"]["pressure_path"]:
            replay_mismatches.append("pressure_path")
    rho, chi = _chi(x), _chi(theta)
    residual = rate[0] + epi_weight * rho + phase_weight * chi / Q(math.pi)
    pieces = (
        rate[0] - pressure[0],
        pressure[0] + epi_weight * rho - phase_weight * Q(vectors["phase_source"][0]),
        phase_weight * (Q(vectors["phase_source"][0]) + chi / Q(math.pi)),
    )
    _require(residual == sum(pieces), "receiver residual decomposition differs")
    return {
        "x": x,
        "chi": chi,
        "rate0": rate[0],
        "residual": residual,
        "pieces": pieces,
        "pressure_defect": max(map(abs, pressure_defects)),
        "split_defect": max(
            abs(
                pressure[i]
                + epi_weight * q[i] / 2
                - phase_weight * Q(vectors["phase_source"][i])
            )
            for i in range(3)
        ),
        "metric_min": min(vectors["phase_metric"]),
        "margin": math.pi / 2
        - max(abs(float(theta[j] - theta[i])) for i, j in campaign.EDGES),
        "work": abs(balance),
        "extra": abs(extra),
        "replay_mismatches": replay_mismatches,
    }


def audit_retained(prediction, report, *, root=ROOT):
    """Audit record arithmetic and compatible snapshots; never evolve a graph.

    Source/runtime mismatches disable current field replay, without discarding
    the historical decision. Consistency is limited to retained evidence; it
    does not authenticate the unretained Euler steps or their reported extrema.
    """
    _require(
        campaign._encoded(report["prediction"]) == campaign._encoded(prediction),
        "retained prediction differs",
    )
    _require(
        _scientific_spec(prediction) == _scientific_spec(campaign.prepare_prediction()),
        "scientific specification differs from the named v1 protocol",
    )
    model = RelationalExchangeModel(
        prediction["storage_scale"],
        epi_weight=prediction["epi_weight"],
        phase_weight=prediction["phase_weight"],
    )
    horizon = _number(prediction["horizon"], "horizon")
    source_mismatches, runtime_mismatches = _compatibility(
        prediction, Path(root).resolve()
    )
    replay = not source_mismatches and not runtime_mismatches
    gates = prediction["gates"]
    lower, upper = (
        _number(value, "Taylor band") for value in gates["normalized_response_interval"]
    )
    contraction = _number(gates["refinement_contraction"], "refinement contraction")
    slack = _number(gates["numerical_slack"], "numerical slack")
    signal_factor = _number(gates["signal_to_last_grid_difference"], "signal factor")
    clocks = {"initial": Q(0), "midpoint": horizon / 2, "final": horizon}
    grids = tuple(map(str, prediction["step_counts"]))
    laws, capacities = prediction["laws"], prediction["capacities"]
    _keys(report["trajectories"], laws, "trajectory laws")
    _keys(report["observables"], grids, "observable grids")
    _keys(report["observable_refinement"], ("chi", "form"), "observable refinement")
    _keys(report["state_refinement"], laws, "state refinement laws")
    for law in laws:
        _keys(report["trajectories"][law], capacities, "trajectory capacities")
        _keys(report["state_refinement"][law], capacities, "refinement capacities")
        for capacity in capacities:
            _keys(report["trajectories"][law][capacity], grids, "trajectory grids")
            _keys(
                report["state_refinement"][law][capacity],
                ("midpoint", "final"),
                "refinement checkpoints",
            )
    snapshots, receiver, values, decisions = {}, {}, {"chi": [], "form": []}, []
    max_pressure = max_receiver = max_piece = Q(0)
    replay_mismatches = {}
    count = 0
    for steps in prediction["step_counts"]:
        key = str(steps)
        _keys(report["observables"][key], values, "observable fields")
        snapshots[key] = {}
        for law in prediction["laws"]:
            snapshots[key][law] = {}
            for capacity in prediction["capacities"]:
                trace = report["trajectories"][law][capacity][key]
                _require(
                    trace["law"] == law
                    and trace["capacity_case"] == capacity
                    and type(trace["steps"]) is int
                    and trace["steps"] == steps,
                    "trace tags differ",
                )
                _require(
                    _number(trace["dt"], "dt") * steps == horizon
                    and _number(trace["elapsed_structural_clock"], "elapsed clock")
                    == horizon,
                    "trace clock differs",
                )
                _require(
                    trace["pressure_paths"] == [prediction["runtime"]["pressure_path"]],
                    "trace pressure path differs",
                )
                states = {}
                _keys(trace["checkpoints"], clocks, "trace checkpoints")
                for label, clock in clocks.items():
                    state = trace["checkpoints"][label]
                    _require(
                        _number(state["clock"], "checkpoint clock") == clock,
                        "checkpoint clock differs",
                    )
                    states[label] = row = _checkpoint(
                        state,
                        prediction=prediction,
                        law=law,
                        capacity=capacity,
                        label=label,
                        replay=replay,
                        model=model,
                    )
                    count += 1
                    max_pressure = max(max_pressure, row["pressure_defect"])
                    max_receiver = max(max_receiver, abs(row["residual"]))
                    max_piece = max(max_piece, *map(abs, row["pieces"]))
                    if row["replay_mismatches"]:
                        replay_mismatches[f"{law}/{capacity}/{key}/{label}"] = row[
                            "replay_mismatches"
                        ]
                snapshots[key][law][capacity] = states
                min_metric = finite_represented_real(
                    trace["minimum_phase_metric"], "minimum metric"
                )[0]
                min_margin = finite_represented_real(
                    trace["minimum_chart_margin"], "minimum margin"
                )[0]
                max_work = _number(
                    trace["maximum_actual_balance_residual"], "maximum balance residual"
                )
                _require(
                    0 < min_metric <= min(row["metric_min"] for row in states.values()),
                    "reported metric minimum omits a checkpoint",
                )
                _require(
                    _number(gates["minimum_chart_margin"], "chart guard")
                    < min_margin
                    <= min(row["margin"] for row in states.values()),
                    "reported chart minimum omits a checkpoint",
                )
                _require(
                    max(row["work"] for row in states.values())
                    <= max_work
                    <= _number(gates["maximum_actual_balance_residual"], "work guard"),
                    "reported work maximum omits a checkpoint or fails guard",
                )
                _require(
                    _number(trace["maximum_extra_work_residual"], "extra work maximum")
                    >= max(row["extra"] for row in states.values()),
                    "reported extra-work maximum omits a checkpoint",
                )
                _require(
                    _number(
                        trace["maximum_pressure_split_residual"], "pressure maximum"
                    )
                    >= max(row["split_defect"] for row in states.values()),
                    "reported pressure maximum omits a checkpoint",
                )
                for label in (
                    "maximum_euler_storage_defect",
                    "maximum_update_defect",
                    "maximum_clock_defect",
                    "elapsed_wall_seconds",
                ):
                    _require(_number(trace[label], label) >= 0, f"invalid {label}")
        receiver[key] = {}
        for label in clocks:
            case = snapshots[key]
            difference = {
                name: case["counterpart"]["intervened"][label][name]
                - case["counterpart"]["baseline"][label][name]
                - case["relational"]["intervened"][label][name]
                + case["relational"]["baseline"][label][name]
                for name in ("chi", "rate0", "residual")
            }
            form = tuple(
                case["counterpart"]["intervened"][label]["x"][i]
                - case["counterpart"]["baseline"][label]["x"][i]
                - case["relational"]["intervened"][label]["x"][i]
                + case["relational"]["baseline"][label]["x"][i]
                for i in range(3)
            )
            defect = (
                difference["rate0"]
                + Q(model.epi_weight) * _chi(form)
                + Q(model.phase_weight) * difference["chi"] / Q(math.pi)
            )
            _require(
                defect == difference["residual"] and abs(defect) <= AUDIT_TOLERANCE,
                "receiver identity fails",
            )
            receiver[key][label] = defect
        for name in values:
            response = {}
            for law in prediction["laws"]:
                states = snapshots[key][law]
                response[law] = (
                    states["intervened"]["final"]["chi"]
                    - states["baseline"]["final"]["chi"]
                    if name == "chi"
                    else states["intervened"]["final"]["x"][0]
                    - states["baseline"]["final"]["x"][0]
                )
            difference = response["counterpart"] - response["relational"]
            row = report["observables"][key][name]
            leading = finite_represented_real(
                prediction["taylor"][f"{name}_leading_response_estimate"],
                "leading response",
            )[0]
            _require(leading != 0, "zero leading response")
            ratio = float(difference) / leading
            admitted = float(lower) <= ratio <= float(upper)
            _require(
                row["capacity_responses"] == response
                and _number(row["difference_of_differences"], "response") == difference
                and finite_represented_real(
                    row["normalized_response"], "normalized response"
                )[0]
                == ratio
                and row["inside_frozen_taylor_band"] is admitted,
                "observable decision differs",
            )
            values[name].append(difference)
            decisions.append(admitted)
    for name, (coarse, middle, fine) in values.items():
        first, second = abs(middle - coarse), abs(fine - middle)
        row = report["observable_refinement"][name]
        admitted, resolved = second <= contraction * first + slack, abs(
            fine
        ) > signal_factor * (second + slack)
        _require(
            _number(row["coarse_to_middle"], "coarse difference") == first
            and _number(row["middle_to_fine"], "fine difference") == second
            and row["contracting"] is admitted
            and row["signal_resolved"] is resolved,
            "observable refinement differs",
        )
        decisions.extend((admitted, resolved))
    for law in prediction["laws"]:
        for capacity in prediction["capacities"]:
            for label in ("midpoint", "final"):
                vectors = []
                for steps in prediction["step_counts"]:
                    state = report["trajectories"][law][capacity][str(steps)][
                        "checkpoints"
                    ][label]
                    vectors.append(tuple(map(Q, state["epi"] + state["phase"])))
                first, second = (
                    max(abs(b - a) for a, b in zip(left, right, strict=True))
                    for left, right in zip(vectors, vectors[1:], strict=False)
                )
                row = report["state_refinement"][law][capacity][label]
                admitted = second <= contraction * first + slack
                _require(
                    _number(row["coarse_to_middle_inf"], "coarse state difference")
                    == first
                    and _number(row["middle_to_fine_inf"], "fine state difference")
                    == second
                    and row["contracting"] is admitted,
                    "state refinement differs",
                )
                decisions.append(admitted)
    _require(report["passed"] is all(decisions), "recorded overall decision differs")
    return {
        "record_consistent": True,
        "recorded_passed": report["passed"],
        "checkpoint_count": count,
        "source_compatible": not source_mismatches,
        "runtime_compatible": not runtime_mismatches,
        "source_mismatches": source_mismatches,
        "runtime_mismatches": runtime_mismatches,
        "replay_available": replay,
        "replayed_checkpoints": count if replay else 0,
        "replay_exact": not replay_mismatches if replay else None,
        "replay_mismatches": replay_mismatches,
        "maximum_independent_pressure_defect": max_pressure,
        "maximum_individual_receiver_defect": max_receiver,
        "maximum_receiver_decomposition_term": max_piece,
        "receiver_identity_defects": receiver,
        "audit_tolerance": AUDIT_TOLERANCE,
        "scope": "retained arithmetic and compatible read-only fields; omitted steps/extrema, causal freeze order and complete dependencies are not authenticated",
    }


def audit_files(
    prediction_path=DEFAULT_DIRECTORY / "result.prediction.json",
    result_path=DEFAULT_DIRECTORY / "result.json",
    *,
    root=ROOT,
):
    """Read immutable records and report their current byte fingerprints."""
    result = audit_retained(
        load_exact_json(prediction_path), load_exact_json(result_path), root=root
    )
    result["artifact_sha256"] = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (prediction_path, result_path)
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction", type=Path, default=DEFAULT_DIRECTORY / "result.prediction.json"
    )
    parser.add_argument(
        "--result", type=Path, default=DEFAULT_DIRECTORY / "result.json"
    )
    args = parser.parse_args(argv)
    report = audit_files(args.prediction, args.result)
    print(campaign._encoded(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

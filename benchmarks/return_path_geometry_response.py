"""Freeze geometry, predict a nodal tangent response, then evaluate it.

This is a declared known-source mathematical control. The geometry-only
predictor never receives the source coefficient or the reserved response.
Write-once records and source archives reuse the established evidence owner;
hashes bind bytes, not independent provenance or authenticated chronology.
"""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp
import networkx as nx

from benchmarks import relational_seeded_response as evidence
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.phase_cycle_geometry import (
    _return_path_storage_residual,
    assess_return_path_geometry_response,
)
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.sdk import relational_report_to_dict
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/assets/return_path_geometry_response/declaration.json"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/research/return_path_geometry_response/response-v1.json"
)
_MAX_BYTES = 32 * 1024**2


def _read(path):
    evidence._require(path.stat().st_size <= _MAX_BYTES, "record exceeds byte budget")
    data = path.read_bytes()
    evidence._require(len(data) <= _MAX_BYTES, "record exceeds byte budget")
    return data, json_loads(data)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _source_files():
    files = evidence._source_files()
    for path in (Path(__file__).resolve(), DECLARATION):
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return files


def _runtime():
    return {**evidence._runtime(), "mpmath": mp.__version__, "networkx": nx.__version__}


def _graph(declaration):
    graph = nx.Graph()
    graph.add_nodes_from(declaration["nodes"])
    for key in ("left_cycle", "right_cycle"):
        cycle = declaration[key]
        graph.add_edges_from(zip(cycle, cycle[1:] + cycle[:1]), weight=1)
    graph.add_edges_from(declaration["additional_edges"], weight=1)
    return graph


def _source_root(declaration):
    coefficient = declaration["source_coefficient"]

    def current(angle):
        sine = mp.sin(angle)
        return sine + coefficient * sine**3

    return mp.findroot(
        lambda angle: current(mp.pi / 2 - angle / 4)
        - current(angle)
        - current(2 * angle / 3),
        (mp.pi / 6, 2 * mp.pi / 5),
    )


def prepare_observation(declaration):
    """Quantize a known-source geometry, certifying its enclosing root signs."""
    settings = declaration["geometry_observation"]
    with mp.workdps(settings["decimal_places"]):
        scaled = (
            _source_root(declaration) / (2 * mp.pi) * settings["turn_grid_denominator"]
        )
        lower = Q(int(mp.floor(scaled)), settings["turn_grid_denominator"])
        upper = Q(int(mp.ceil(scaled)), settings["turn_grid_denominator"])
    coefficient = Q(declaration["source_coefficient"])
    signs = (
        _return_path_storage_residual(lower, coefficient),
        _return_path_storage_residual(upper, coefficient),
    )
    evidence._require(
        lower < upper and signs[0].lo > 0 > signs[1].hi,
        "quantized source geometry has unresolved strict root signs",
    )
    return {
        "special_turn_bounds": [str(lower), str(upper)],
        "origin": settings["origin"],
    }


def predict_response(declaration, observation):
    """Use geometry and independent scales; do not consume the source coefficient."""
    graph = _graph(declaration)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    return assess_return_path_geometry_response(
        source,
        left_cycle=declaration["left_cycle"],
        right_cycle=declaration["right_cycle"],
        mediator=declaration["mediator"],
        special_turn_bounds=tuple(
            Q(value) for value in observation["special_turn_bounds"]
        ),
        form_direction=declaration["form_direction"],
        observation_origin=observation["origin"],
    )


def _prediction_input(declaration):
    """Expose only graph preparation and the declared perturbation to inference."""
    return {
        key: declaration[key]
        for key in (
            "nodes",
            "left_cycle",
            "right_cycle",
            "mediator",
            "additional_edges",
            "form_direction",
        )
    }


def evaluate_response(declaration):
    """Differentiate full neighbor sums independently of the predictor's Hessian."""
    graph = _graph(declaration)
    settings = declaration["geometry_observation"]
    with mp.workdps(settings["decimal_places"]):
        v = _source_root(declaration)
        bulk, join = mp.pi / 2 - v / 4, 2 * v / 3
        angles = (
            0,
            v,
            v + bulk,
            v + 2 * bulk,
            v + 3 * bulk,
            2 * join,
            2 * join - v,
            2 * join - v - bulk,
            2 * join - v - 2 * bulk,
            2 * join - v - 3 * bulk,
            join,
        )
        impulse = declaration["form_direction"]
        phase_velocity = [
            sum(mp.mpf(impulse[i] - impulse[j]) for j in graph[i]) / graph.degree[i]
            for i in graph
        ]
        coefficient = declaration["source_coefficient"]

        def current(angle):
            sine = mp.sin(angle)
            return sine + coefficient * sine**3

        values = [
            mp.diff(
                lambda time: sum(
                    current(
                        angles[j]
                        - angles[i]
                        + time * (phase_velocity[j] - phase_velocity[i])
                    )
                    for j in graph[i]
                )
                / graph.degree[i],
                0,
            )
            for i in graph
        ]
        return [
            mp.nstr(value, declaration["response_record_digits"]) for value in values
        ]


def _fraction(record):
    evidence._require(
        isinstance(record, dict)
        and set(record) == {"numerator", "denominator"}
        and type(record["numerator"]) is int
        and type(record["denominator"]) is int
        and record["denominator"] > 0,
        "invalid projected exact fraction",
    )
    return Q(record["numerator"], record["denominator"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output
    protocol_path = output.with_suffix(".protocol.json")
    archive = output.with_suffix(".source.zip")
    prediction_path = output.with_suffix(".prediction.json")
    if output.exists():
        raise FileExistsError(f"retain existing response: {output}")
    evidence._verify_runtime_source()
    _, declaration = _read(DECLARATION)
    if args.stage == "prepare":
        if any(path.exists() for path in (protocol_path, archive, prediction_path)):
            raise FileExistsError("retain existing preparation or prediction")
        observation = prepare_observation(declaration)
        files = _source_files()
        evidence._archive(archive, files)
        _verify_archive(archive, evidence._manifest(files))
        evidence._write(
            protocol_path,
            {
                "schema": "tnfr.return-path-geometry-response-protocol.v1",
                "declaration": declaration,
                "observation": observation,
                "runtime": _runtime(),
                "source_sha256": evidence._manifest(files),
                "archive_scope": "all_tnfr_python_both_producers_pyproject_and_declaration_no_dependency_binaries",
            },
        )
        print(f"Prepared geometry: {protocol_path}")
        return 0
    protocol_bytes, protocol = _read(protocol_path)
    evidence._require(protocol["declaration"] == declaration, "declaration changed")
    evidence._require(protocol["runtime"] == _runtime(), "runtime changed")
    evidence._require(
        protocol["source_sha256"] == evidence._manifest(_source_files()),
        "source changed",
    )
    _verify_archive(archive, protocol["source_sha256"])
    provenance = {
        "protocol_sha256": _digest(protocol_bytes),
        "source_archive_sha256": _digest(archive.read_bytes()),
    }
    if args.stage == "predict":
        if prediction_path.exists():
            raise FileExistsError(
                "retain existing prediction, including unavailable results"
            )
        report = predict_response(
            _prediction_input(declaration), protocol["observation"]
        )
        evidence._write(
            prediction_path, {**provenance, **relational_report_to_dict(report)}
        )
        print(f"Recorded prediction: {prediction_path}")
        return 0
    prediction_bytes, prediction = _read(prediction_path)
    evidence._require(
        all(prediction.get(key) == value for key, value in provenance.items()),
        "prediction provenance mismatch",
    )
    # Re-admit the frozen geometric premises before consuming projected bounds.
    # This checks the saved prediction without replacing it or using a response.
    rebuilt = relational_report_to_dict(
        predict_response(_prediction_input(declaration), protocol["observation"])
    )
    evidence._require(
        evidence._encoded(prediction) == evidence._encoded({**provenance, **rebuilt}),
        "frozen prediction differs from admitted geometric premises",
    )
    report = prediction["report"]
    evidence._require(
        report["coefficient_status"] == "bounded",
        "prediction unavailable; do not evaluate response",
    )
    values = evaluate_response(declaration)
    bounds = report["response_acceleration_bounds"]
    evidence._require(len(bounds) == len(values) == 11, "response node order mismatch")
    contained = [
        _fraction(bound["lo"]) <= Q(value) <= _fraction(bound["hi"])
        for value, bound in zip(values, bounds)
    ]
    lower, upper = _fraction(report["coefficient_lower"]), _fraction(
        report["coefficient_upper"]
    )
    coefficient_contained = lower <= declaration["source_coefficient"] <= upper
    passed = coefficient_contained and lower > 0 and all(contained)
    evidence._write(
        output,
        {
            "schema": "tnfr.return-path-geometry-response-record.v1",
            **provenance,
            "prediction_sha256": _digest(prediction_bytes),
            "response_acceleration_decimal": values,
            "component_containment": contained,
            "coefficient_containment": coefficient_contained,
            "sine_baseline_excluded": lower > 0,
            "passed": passed,
            "scope": declaration["limitations"],
        },
    )
    print(f"Recorded response: {output}; passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

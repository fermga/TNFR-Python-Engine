"""Freeze and evaluate one two-direction full-state metric connection trial.

Use separate --prepare and evaluation invocations. Both horizons, the literal
rational source and the numerical method are declared before any flow. Every
complete or partial response is retained once, including noncertificates.
"""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction as Q
from pathlib import Path

from benchmarks import relational_seeded_response as evidence
from tnfr._exact_time import exact_or_represented_real
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._validated_metric import _admit_rate_search
from tnfr.physics.relational_sine_comparison import (
    _comparison_from_state,
    _sine_state_from_rows,
)
from tnfr.physics.relational_sine_corridor import prepare_sine_saddle_state
from tnfr.physics.relational_sine_forecast import _admit_support
from tnfr.physics.relational_sine_metric_connection import (
    forecast_sine_metric_connection,
)
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[1]


def _real(value, label):
    if isinstance(value, str):
        value = Q(value)
    return exact_or_represented_real(value, label)


def _source_files(declaration_path):
    files = evidence._source_files()
    for path in (Path(__file__).resolve(), Path(declaration_path).resolve()):
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return files


def _preparation(declaration):
    """Admit the literal rational recipe and fixed budgets without any flow."""
    expected = {
        "schema": "tnfr.sine-metric-connection-declaration.v1",
        "law": "normalized_sine_reciprocal_e0_w1_beta1",
        "support": "fixed_simple_connected_unit_undirected",
        "forcing": "none",
        "events": "none",
        "clock": "original_structural_t; tau=t/pi",
        "selection_rule": "first_forward_zero_winding_endpoint_and_first_backward_acute_winding_one_window",
    }
    evidence._require(
        all(declaration.get(key) == value for key, value in expected.items()),
        "unsupported complete-law or observation declaration",
    )
    evidence._require(
        declaration.get("directions") == [1, -1]
        and all(type(value) is int for value in declaration["directions"])
        and declaration.get("phase_radius", "missing") is None,
        "declare both directions and whole-tube metric growth, not a saddle phase cutoff",
    )
    nodes = declaration["nodes"]
    evidence._require(
        type(nodes) is list
        and nodes == list(range(10))
        and all(type(value) is int for value in nodes),
        "the literal preparation requires ten ordered integer nodes",
    )
    neighbors = [[] for _ in nodes]
    edges = declaration["edges"]
    evidence._require(type(edges) is list, "edges must be a list")
    for edge in edges:
        evidence._require(
            type(edge) is list
            and len(edge) == 2
            and all(type(i) is int and 0 <= i < 10 for i in edge),
            "invalid declared edge",
        )
        i, j = edge
        neighbors[i].append(j)
        neighbors[j].append(i)
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    neighbors, _ = _admit_support(neighbors, (Q(1),) * 9, model)

    def row(key):
        values = declaration[key]
        evidence._require(
            type(values) is list and len(values) == 10,
            f"{key} must retain ten literal coordinates",
        )
        return tuple(_real(value, key) for value in values)

    form, phase, capacity = row("form"), row("phase"), row("capacity")
    evidence._require(capacity == (Q(1),) * 10, "all ten held capacities must be one")
    cycle = declaration["cycle_indices"]
    evidence._require(
        type(cycle) is list and all(type(i) is int for i in cycle),
        "cycle indices must be exact integers",
    )
    source = _comparison_from_state(
        _sine_state_from_rows(
            tuple(nodes), tuple(map(tuple, edges)), form, phase, capacity, neighbors
        ),
        model,
    )
    preparation = prepare_sine_saddle_state(
        source, cycle=cycle, epsilon=_real(declaration["epsilon"], "epsilon")
    )
    evidence._require(
        preparation.preparation_certified,
        "rational near-saddle preparation is unavailable",
    )
    evidence._require(
        form == preparation.prepared_state.epi
        and phase == preparation.prepared_state.phase,
        "literal state differs from the exact declared rational preparation",
    )
    duration = _real(declaration["duration"], "duration")
    time_step = _real(declaration["time_step"], "time_step")
    radius = _real(
        declaration["initial_coordinate_radius"], "initial_coordinate_radius"
    )
    maximum, order, bisections = (
        declaration[key] for key in ("max_steps", "order", "growth_bisections")
    )
    growth = declaration["growth_rate_bounds"]
    evidence._require(
        type(growth) is list and len(growth) == 2,
        "growth_rate_bounds must contain two exact endpoints",
    )
    growth = _admit_rate_search(
        tuple(_real(value, "growth_rate_bounds") for value in growth), bisections
    )
    evidence._require(
        type(maximum) is int
        and 1 <= maximum <= 256
        and duration > 0
        and time_step > 0
        and duration / time_step <= maximum,
        "declared horizon exceeds its positive fixed step budget",
    )
    evidence._require(type(order) is int and 1 <= order <= 16, "invalid Taylor order")
    evidence._require(
        radius >= 0
        and growth[0] <= growth[1]
        and max(abs(value) for value in growth) * min(duration, time_step) <= 1,
        "invalid initial radius or scalar growth-step budget",
    )
    evidence._require(
        _real(declaration["retained_scaled_duration"], "retained_scaled_duration") == 1
        and _real(declaration["acute_margin"], "acute_margin") == 0,
        "the declared observation is strict acute winding one for one scaled time unit",
    )
    return (
        source,
        preparation,
        dict(
            cycle=tuple(cycle),
            duration=duration,
            time_step=time_step,
            initial_coordinate_radius=radius,
            growth_rate_bounds=growth,
            growth_bisections=bisections,
            order=order,
            max_steps=maximum,
        ),
    )


def prepare_protocol(declaration_path):
    evidence._verify_runtime_source()
    path = Path(declaration_path).resolve()
    declaration = json_loads(path.read_bytes())
    _, preparation, _ = _preparation(declaration)
    return {
        "schema": "tnfr.sine-metric-connection-protocol.v1",
        "declaration": declaration,
        "preparation": preparation.to_dict(),
        "runtime": evidence._runtime(),
        "declaration_path": path.relative_to(ROOT).as_posix(),
        "source_sha256": evidence._manifest(_source_files(path)),
        "archive_scope": "all src/tnfr Python, evidence transport, producer, declaration and pyproject; not dependency binaries or provenance authentication",
    }


def evaluate(declaration):
    source, _, options = _preparation(declaration)
    return forecast_sine_metric_connection(source, **options).to_dict()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    declaration, output = args.declaration.resolve(), args.output.resolve()
    protocol_path, archive = output.with_suffix(".protocol.json"), output.with_suffix(
        ".source.zip"
    )
    if output.exists():
        raise FileExistsError(f"retain existing response: {output}")
    if args.prepare:
        if protocol_path.exists() or archive.exists():
            raise FileExistsError("retain existing protocol and source archive")
        protocol = prepare_protocol(declaration)
        evidence._archive(archive, _source_files(declaration))
        _verify_archive(archive, protocol["source_sha256"])
        evidence._write(protocol_path, protocol)
        print(f"Frozen protocol: {protocol_path}", flush=True)
        return 0
    evidence._require(
        protocol_path.stat().st_size <= 32 * 1024**2,
        "protocol exceeds verification byte budget",
    )
    raw = protocol_path.read_bytes()
    protocol = json_loads(raw)
    evidence._require(
        evidence._encoded(protocol) == evidence._encoded(prepare_protocol(declaration)),
        "frozen source, runtime or declaration changed",
    )
    _verify_archive(archive, protocol["source_sha256"])
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(
        "Evaluating both fixed directions; no adaptive retry or target fitting",
        flush=True,
    )
    response, error, passed = None, None, False
    try:
        response = evaluate(json_loads(evidence._encoded(protocol["declaration"])))
        evidence._require(
            response.get("schema") == "tnfr.sine-metric-connection.v1",
            "unexpected connection response schema",
        )
        evidence._encoded(response)
        report = response["report"]
        passed = (
            report["same_orbit_connection_certified"] is True
            and report["declared_horizons_complete"] is True
        )
        evidence._verify_runtime_source()
        evidence._require(
            protocol["source_sha256"] == evidence._manifest(_source_files(declaration))
            and protocol["runtime"] == evidence._runtime()
            and protocol_path.read_bytes() == raw
            and hashlib.sha256(archive.read_bytes()).hexdigest() == archive_sha,
            "source, runtime or frozen evidence changed during evaluation",
        )
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)}
        passed = False
    record = {
        "schema": "tnfr.sine-metric-connection-response.v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "source_archive_sha256": archive_sha,
        "response": response,
        "evaluation_error": error,
        "passed": passed,
    }
    evidence._write(output, record)
    print(f"Retained response: {output}; passed={passed}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

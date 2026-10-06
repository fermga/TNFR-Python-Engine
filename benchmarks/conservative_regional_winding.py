"""Retain one analytic conservative acquisition certificate and its controls.

No trajectory or parameter search runs. A frozen declaration supplies the
environmental preparation and complete observation window; the shared owner
proves branch retention using global bounds on the full nonlinear law.
"""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx

from benchmarks import relational_seeded_response as evidence
from tnfr._exact_time import exact_or_represented_real
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_conservative_winding_entry
from tnfr.physics.relational_sine_symmetry import assess_sine_cycle_symmetry
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.sdk import relational_report_to_dict
from tnfr.sdk.relational_reports import _project
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/assets/conservative_regional_winding/declaration.json"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/research/conservative_regional_winding/certificate-v1.json"
)


def _source(declaration, *, quiet=False, return_edge=True):
    graph = nx.Graph()
    graph.add_nodes_from(declaration["nodes"])
    graph.add_edges_from(declaration["ring_edges"], weight=1)
    graph.add_edges_from(
        (
            edge
            for edge in declaration["connecting_edges"]
            if return_edge or edge != [1, 6]
        ),
        weight=1,
    )
    graph.graph["GAMMA"] = {"type": "none"}
    for node, form, phase in zip(
        declaration["nodes"], declaration["form"], declaration["phase"]
    ):
        graph.nodes[node].update(EPI=0 if quiet else form, theta=phase, nu_f=1)
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def assess_control(declaration):
    """Evaluate the fixed analytic certificate and two distinct controls."""
    source = _source(declaration)
    cycle = tuple(declaration["receiver_cycle"])
    window = tuple(map(Q, declaration["scaled_window"]))
    certificate = certify_sine_conservative_winding_entry(
        source,
        cycle=cycle,
        scaled_window=window,
        edge_turn_offsets=declaration["edge_turn_offsets"],
    )
    balance = source.regional_storage_balance(region=cycle)
    quiet = certify_sine_conservative_winding_entry(
        _source(declaration, quiet=True),
        cycle=cycle,
        scaled_window=window,
        edge_turn_offsets=(0,) * len(cycle),
    )
    single_port = assess_sine_cycle_symmetry(
        _source(declaration, return_edge=False),
        permutation_indices=(0, 1, 2, 3, 4, 5, 9, 8, 7, 6, 10),
        cycle=cycle,
    )
    passed = (
        certificate.acquisition_certified
        and certificate.certified_winding == declaration["declared_window_winding"]
        and quiet.certified_winding == 0
        and not quiet.acquisition_certified
        and all(value == 0 for value in quiet.initial_phase_velocity)
        and single_port.zero_winding_when_nonantipodal
    )
    return {
        "certificate": relational_report_to_dict(certificate),
        "initial_storage_balance": relational_report_to_dict(balance),
        "zero_environment_control": relational_report_to_dict(quiet),
        "single_port_control": relational_report_to_dict(single_port),
        "passed": passed,
        "scope": declaration["scope"],
    }


def _declared_scalar(value, label):
    """Admit exact declaration text or a represented real before conversion."""
    if isinstance(value, str):
        try:
            return Q(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{label} requires finite rational text") from exc
    return exact_or_represented_real(value, label)


def assess_source_geometry_control(declaration):
    """Compare finite acute entry with an equal-budget full-state reflection."""
    from tnfr.physics.relational_sine_entry import (
        analyze_sine_conservative_source_geometry,
    )

    evidence._require(
        declaration["schema"] == "tnfr.conservative-source-geometry-declaration.v1"
        and declaration["law"] == "normalized_sine_reciprocal_e0_w1_beta1"
        and declaration["support"] == "fixed_simple_connected_unit_undirected"
        and declaration["clock"] == "tau=t/pi"
        and declaration["forcing"] == "none"
        and declaration["events"] == "none",
        "unsupported complete-law source declaration",
    )
    nodes = declaration["nodes"]
    size = len(nodes)
    evidence._require(
        all(type(node) is int for node in nodes) and nodes == list(range(size)),
        "the declaration requires consecutive integer node labels",
    )
    edges = set()
    for field in ("ring_edges", "connecting_edges"):
        evidence._require(
            isinstance(declaration[field], (list, tuple)), "ordered edge rows required"
        )
        for edge in declaration[field]:
            evidence._require(
                isinstance(edge, (list, tuple))
                and len(edge) == 2
                and all(type(node) is int and 0 <= node < size for node in edge)
                and edge[0] != edge[1],
                "edges require distinct declared integer endpoints",
            )
            pair = tuple(sorted(edge))
            evidence._require(pair not in edges, "duplicate declared edge")
            edges.add(pair)
    for field in ("form", "control_form", "phase", "capacity"):
        evidence._require(
            isinstance(declaration[field], (list, tuple))
            and len(declaration[field]) == size,
            "incomplete source rows",
        )
        values = tuple(
            exact_or_represented_real(value, field) for value in declaration[field]
        )
        if field == "capacity":
            evidence._require(
                all(value == 1 for value in values), "unit capacities required"
            )
    cycle = declaration["receiver_cycle"]
    evidence._require(
        isinstance(cycle, (list, tuple))
        and all(type(node) is int and 0 <= node < size for node in cycle),
        "receiver cycle requires declared integer nodes",
    )
    raw_window = declaration["scaled_window"]
    evidence._require(
        isinstance(raw_window, (list, tuple)) and len(raw_window) == 2,
        "scaled_window requires two ordered endpoints",
    )
    window = tuple(_declared_scalar(value, "scaled_window") for value in raw_window)
    margin = _declared_scalar(
        declaration["minimum_acute_margin"], "minimum_acute_margin"
    )
    evidence._require(margin > 0, "positive declared acute margin required")
    source = _source(declaration)
    control = _source({**declaration, "form": declaration["control_form"]})
    cycle = tuple(declaration["receiver_cycle"])
    geometry = analyze_sine_conservative_source_geometry(source, receiver=cycle)
    control_geometry = analyze_sine_conservative_source_geometry(
        control, receiver=cycle
    )
    certificate = certify_sine_conservative_winding_entry(
        source,
        cycle=cycle,
        scaled_window=window,
        edge_turn_offsets=declaration["edge_turn_offsets"],
    )
    symmetry = assess_sine_cycle_symmetry(
        control,
        permutation_indices=declaration["control_reflection"],
        cycle=cycle,
    )
    positive_balance = source.regional_storage_balance(region=cycle)
    control_balance = control.regional_storage_balance(region=cycle)
    mean = lambda item: sum(
        (degree * form for degree, form in zip(item.degrees, item.epi)), Q(0)
    ) / sum(item.degrees)
    expected_storage = exact_or_represented_real(
        declaration["declared_total_storage"], "declared_total_storage"
    )
    expected_mean = exact_or_represented_real(
        declaration["declared_weighted_form_mean"], "declared_weighted_form_mean"
    )
    target_winding = declaration["declared_window_winding"]
    evidence._require(type(target_winding) is int, "nonboolean winding required")
    gates = {
        "finite_acute_acquisition": certificate.acute_acquisition_certified,
        "target_winding": certificate.certified_winding == target_winding,
        "declared_acute_margin": certificate.acute_margin_lower_bound is not None
        and certificate.acute_margin_lower_bound > margin,
        "control_acute_entry_excluded": symmetry.zero_winding_when_nonantipodal,
        "same_complete_support": source.edges == control.edges,
        "same_total_storage": source.form_storage
        == control.form_storage
        == expected_storage,
        "same_storage_partitions": all(
            getattr(positive_balance, name) == getattr(control_balance, name)
            for name in (
                "regional_form_storage",
                "complement_form_storage",
                "boundary_form_storage",
            )
        ),
        "same_conserved_form_mean": mean(source) == mean(control) == expected_mean,
    }
    return {
        "certificate": relational_report_to_dict(certificate),
        "source_geometry": relational_report_to_dict(geometry),
        "control_geometry": relational_report_to_dict(control_geometry),
        "reflection_control": relational_report_to_dict(symmetry),
        "initial_storage_balance": relational_report_to_dict(positive_balance),
        "control_storage_balance": relational_report_to_dict(control_balance),
        "weighted_form_means": _project((mean(source), mean(control))),
        "gates": gates,
        "passed": all(gates.values()),
        "scope": declaration["scope"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--declaration", type=Path, default=DECLARATION)
    args = parser.parse_args(argv)
    output, declaration_path = args.output, args.declaration.resolve()
    declaration_path.relative_to(ROOT)
    archive = output.with_suffix(".source.zip")
    if output.exists() or archive.exists():
        raise FileExistsError("retain existing analytic certificate and source archive")
    declaration_bytes = declaration_path.read_bytes()
    declaration = json_loads(declaration_bytes)
    evidence._require(
        declaration.get("schema")
        in (
            "tnfr.conservative-regional-winding-declaration.v1",
            "tnfr.conservative-source-geometry-declaration.v1",
        ),
        "unsupported analytic declaration schema",
    )
    evidence._verify_runtime_source()
    runtime = {**evidence._runtime(), "networkx": nx.__version__}
    files = evidence._source_files()
    producer_path = Path(__file__).resolve()
    files[producer_path.relative_to(ROOT).as_posix()] = producer_path.read_bytes()
    files[declaration_path.relative_to(ROOT).as_posix()] = declaration_bytes
    evidence._archive(archive, files)
    manifest = evidence._manifest(files)
    _verify_archive(archive, manifest)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if declaration.get("schema") == "tnfr.conservative-source-geometry-declaration.v1":
        result = assess_source_geometry_control(declaration)
        schema = "tnfr.conservative-source-geometry-control.v1"
    else:
        result = assess_control(declaration)
        schema = "tnfr.conservative-regional-winding-control.v1"
    evidence._verify_runtime_source()
    evidence._require(
        all((ROOT / name).read_bytes() == contents for name, contents in files.items())
        and hashlib.sha256(archive.read_bytes()).hexdigest() == archive_sha
        and runtime == {**evidence._runtime(), "networkx": nx.__version__},
        "analytic source, declaration, archive or runtime changed during evaluation",
    )
    evidence._write(
        output,
        {
            "schema": schema,
            "declaration_sha256": hashlib.sha256(declaration_bytes).hexdigest(),
            "source_archive_sha256": archive_sha,
            "source_sha256": manifest,
            "runtime": runtime,
            "evidence_kind": "analytic_global_enclosure_not_sampled_or_reserved_trajectory",
            **result,
        },
    )
    print(f"Analytic certificate recorded: {output}; passed={result['passed']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

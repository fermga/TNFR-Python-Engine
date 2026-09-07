"""Composite readouts share current fields without retaining graph state."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA
from tnfr.operators import apply_glyph
from tnfr.physics import conservation, unified, variational


READERS = (
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "compute_phase_current",
    "compute_dnfr_flux",
)
COMPOSITES = (
    unified.compute_unified_field_suite,
    variational.compute_variational_suite,
    variational.capture_lagrangian_snapshot,
    variational.translate_sectors,
    conservation.capture_conservation_snapshot,
)
GRAPH_KINDS = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


def _graph(kind=nx.Graph):
    graph = kind()
    nodes = ("a", ("b", 2), 3, "isolated")
    graph.add_nodes_from(nodes)
    graph.add_edges_from([("a", "a"), ("a", ("b", 2)), (("b", 2), 3)])
    if graph.is_multigraph():
        graph.add_edge("a", ("b", 2), weight=7.0)
    graph.graph["RANDOM_SEED"] = 17
    for node, phase, pressure in zip(
        nodes, [0.2, -0.5, 0.7, 0.1], [0.2, -0.3, 0.7, -0.1]
    ):
        set_attr(graph.nodes[node], ALIAS_THETA, phase)
        set_attr(graph.nodes[node], ALIAS_DNFR, pressure)
        graph.nodes[node].update(EPI=0.5, nu_f=1.0)
    return graph


@pytest.mark.parametrize("readout", COMPOSITES, ids=lambda f: f.__name__)
def test_composite_requests_each_base_field_once(monkeypatch, readout):
    """Cache hits still hash state; a suite must not request the same field again."""
    counts = Counter()
    for module in (unified, variational, conservation):
        for name in READERS:
            original = getattr(module, name)

            def counted(*args, _name=name, _original=original, **kwargs):
                counts[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(module, name, counted)
    readout(_graph())
    assert counts == Counter({name: 1 for name in READERS})


@pytest.mark.parametrize("kind", GRAPH_KINDS)
def test_unified_suite_matches_individual_fields_and_conservation_totals(kind):
    graph = _graph(kind)
    before = deepcopy(dict(graph.nodes(data=True)))
    result = unified.compute_unified_field_suite(graph)
    functions = {
        "chirality": unified.compute_chirality_field,
        "symmetry_breaking": unified.compute_symmetry_breaking_field,
        "coherence_coupling": unified.compute_coherence_coupling_field,
        "energy_density": unified.compute_energy_density,
        "action_density": unified.compute_action_density,
        "historical_q_density": unified.compute_historical_q_density,
        "topological_charge": unified.compute_topological_charge,
        "charge_density": conservation.compute_charge_density,
        "current_j_phi": unified.compute_phase_current,
        "current_j_dnfr": unified.compute_dnfr_flux,
    }
    for key, function in functions.items():
        assert result[key] == function(graph)
    assert result["historical_q_density"] is not result["topological_charge"]
    psi = unified.compute_complex_geometric_field(graph)
    assert result["psi_magnitude"] == unified.compute_field_magnitude(psi)
    assert result["psi_phase"] == unified.compute_field_phase(psi)
    metrics = result["conservation_metrics"]
    assert metrics["noether_charge"] == conservation.compute_noether_charge(graph)
    assert metrics["structural_energy"] == conservation.compute_energy_functional(graph)
    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("kind", GRAPH_KINDS)
def test_variational_suite_matches_standalone_results_and_energy_identity(kind):
    graph = _graph(kind)
    result = variational.compute_variational_suite(graph)
    snap = result["lagrangian_snapshot"]
    assert snap == variational.capture_lagrangian_snapshot(graph)
    assert result["critical_points"] == variational.analyze_potential_critical_points(
        graph
    )
    assert result["grammar_stationarity"] == variational.analyze_grammar_stationarity(
        graph
    )
    assert snap.kinetic == variational.compute_kinetic_density(graph)
    assert snap.potential == variational.compute_potential_density(graph)
    assert snap.interaction == variational.compute_interaction_density(graph)
    assert snap.hamiltonian == pytest.approx(
        variational.compute_hamiltonian_density(graph)
    )
    assert snap.total_hamiltonian == pytest.approx(
        conservation.compute_energy_functional(graph)
    )
    sectors = variational.translate_sectors(graph)
    assert sectors["variational"] == {"T": snap.kinetic, "V": snap.potential}
    assert sectors["consistency_check"] < 1e-12


@pytest.mark.parametrize("kind", GRAPH_KINDS)
def test_conservation_snapshot_divergence_uses_its_recorded_currents(kind):
    graph = _graph(kind)
    snap = conservation.capture_conservation_snapshot(graph)
    for node in graph:
        neighbors = list(graph.neighbors(node))
        expected = (
            sum(snap.j_phi[j] - snap.j_phi[node] for j in neighbors) / len(neighbors)
            + sum(snap.j_dnfr[j] - snap.j_dnfr[node] for j in neighbors)
            / len(neighbors)
            if neighbors else 0.0
        )
        assert snap.divergence[node] == expected
        assert snap.charge_density[node] == snap.phi_s[node] + snap.k_phi[node]
    assert conservation._energy_from_snapshot(
        snap
    ) == conservation.compute_energy_functional(graph)


@pytest.mark.parametrize("kind", GRAPH_KINDS)
def test_readout_refreshes_after_canonical_operator_without_changing_old_snapshots(
    kind,
):
    graph = _graph(kind)
    old_conservation = conservation.capture_conservation_snapshot(graph)
    old_variational = variational.capture_lagrangian_snapshot(graph)
    recorded_c, recorded_v = asdict(old_conservation), asdict(old_variational)
    apply_glyph(graph, ("b", 2), "coherence")
    current = unified.compute_unified_field_suite(graph)
    fresh = conservation.capture_conservation_snapshot(graph)
    assert fresh.phi_s != old_conservation.phi_s
    assert current["energy_density"] == unified.compute_energy_density(graph)
    assert fresh == conservation.capture_conservation_snapshot(graph)
    assert asdict(old_conservation) == recorded_c
    assert asdict(old_variational) == recorded_v


def test_snapshot_owns_maps_even_if_a_kernel_returns_a_shared_mapping(monkeypatch):
    graph = _graph()
    fields = {name: getattr(unified, name)(graph) for name in READERS}
    for name, values in fields.items():
        monkeypatch.setattr(unified, name, lambda graph, _values=values: _values)
    snap = conservation.capture_conservation_snapshot(graph)
    recorded = asdict(snap)
    for values in fields.values():
        values["a"] = 99.0
    assert asdict(snap) == recorded


def test_mutating_returned_maps_does_not_corrupt_a_later_readout():
    graph = _graph()
    first = unified.compute_unified_field_suite(graph)
    expected = deepcopy(first)
    first["current_j_phi"]["a"] = 99.0
    first["energy_density"]["a"] = 99.0
    assert unified.compute_unified_field_suite(graph) == expected


def test_empty_graph_preserves_empty_maps_and_zero_totals():
    graph = nx.Graph()
    result = unified.compute_unified_field_suite(graph)
    assert result.pop("conservation_metrics") == {
        "noether_charge": 0,
        "structural_energy": 0,
    }
    assert all(value == {} for value in result.values())
    suite = variational.compute_variational_suite(graph)
    assert suite["critical_points"] == []
    snap = suite["lagrangian_snapshot"]
    assert snap.kinetic == snap.potential == snap.interaction == {}
    assert snap.total_hamiltonian == snap.total_lagrangian == 0
    assert snap == variational.capture_lagrangian_snapshot(graph)
    assert variational.translate_sectors(graph)["consistency_check"] == 0
    assert all(
        value == {}
        for value in asdict(conservation.capture_conservation_snapshot(graph)).values()
    )


def test_correlation_helper_skips_complex_and_misaligned_maps():
    result = unified.analyze_field_correlations(
        {
            "constant": {0: 1.0, 1: 1.0},
            "varying": {0: 1.0, 1: 2.0},
            "complex_coordinate": {0: 1.0j, 1: 2.0j},
            "partial": {0: 3.0},
        }
    )
    assert result == {}


def test_correlation_helper_handles_empty_input():
    assert unified.analyze_field_correlations({}) == {}

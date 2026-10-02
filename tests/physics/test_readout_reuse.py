"""Composite readouts share current fields without retaining graph state."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tests.example_protocol_helpers import load_example
from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA
from tnfr.operators import apply_glyph
from tnfr.physics import conservation, fields, unified, variational

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
    fields.compute_emergent_fields,
    fields.compute_tensor_invariants,
)
GRAPH_KINDS = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)


def test_field_summary_preserves_finite_extreme_means_and_population_spread():
    stats = unified.summary_statistics(
        {
            "constant": {0: 1e308, 1: 1e308},
            "signed": {0: -1e308, 1: 1e308},
        }
    )
    assert stats["constant"] == {
        "mean": 1e308,
        "std": 0.0,
        "min": 1e308,
        "max": 1e308,
        "range": 0.0,
    }
    assert stats["signed"] == {
        "mean": 0.0,
        "std": 1e308,
        "min": -1e308,
        "max": 1e308,
        "range": None,
    }


def test_field_summary_includes_real_numpy_scalars_and_subnormal_samples():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    stats = unified.summary_statistics(
        {
            "mixed_real": {0: np.float32(2.0), 1: np.int64(4)},
            "tiny": {0: tiny, 1: tiny},
        }
    )
    assert stats["mixed_real"] == {
        "mean": 3.0,
        "std": 1.0,
        "min": 2.0,
        "max": 4.0,
        "range": 2.0,
    }
    assert stats["tiny"]["mean"] == tiny
    assert stats["tiny"]["std"] == 0.0


@pytest.mark.parametrize(
    "invalid", [True, "4", None, float("nan"), Fraction(1, 10**400)]
)
def test_field_summary_omits_invalid_whole_field_without_selecting_a_subset(invalid):
    stats = unified.summary_statistics(
        {
            "valid": {0: 2.0, 1: 4.0},
            "invalid": {0: 2.0, 1: invalid},
            "empty": {},
            "scalar_metadata": 7.0,
        }
    )
    assert set(stats) == {"valid"}
    assert stats["valid"]["mean"] == 3.0


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
            # Instrument existing call sites; a module need not re-export every
            # kernel used by the shared capture in unified.
            if name not in vars(module):
                continue
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
            if neighbors
            else 0.0
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


def _energy_maps(rows):
    return tuple(
        {node: row[index] for node, row in enumerate(rows)} for index in range(5)
    )


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(((1.1e-162,) * 5,), id="underflow-within-node"),
        pytest.param(
            ((1.1e-162, 0.0, 0.0, 0.0, 0.0),) * 8, id="underflow-across-nodes"
        ),
        pytest.param(((1.5e154, 0.0, 0.0, 0.0, 0.0),), id="raw-square-overflow"),
        pytest.param(((1e154, 0.0, 0.0, 0.0, 0.0),) * 2, id="raw-sum-overflow"),
    ],
)
def test_normalized_energy_recovers_representable_total_before_rounding(rows):
    exact = (
        sum((Fraction(value) ** 2 for row in rows for value in row), Fraction(0)) / 2
    )
    expected = float(exact)
    assert expected > 0.0
    assert unified._total_energy_from_fields(*_energy_maps(rows)) == expected


@pytest.mark.parametrize("value", [2e154, 1e-200])
def test_nonzero_energy_outside_represented_range_is_explicitly_unavailable(value):
    maps = _energy_maps(((value, 0.0, 0.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="normalized total energy"):
        unified._total_energy_from_fields(*maps)


@pytest.mark.parametrize("rows", [(), ((0.0,) * 5,)])
def test_zero_energy_remains_available_for_empty_and_zero_fields(rows):
    assert unified._total_energy_from_fields(*_energy_maps(rows)) == 0.0


def test_ordinary_energy_keeps_existing_arithmetic_without_exact_fallback(monkeypatch):
    rows = ((1.0, 0.25, -0.5, -2.0, 3.0), (0.125, 0.0, 2.0, 1.5, 0.0))

    def unexpected(*args, **kwargs):
        pytest.fail("ordinary energy should not require exceptional arithmetic")

    monkeypatch.setattr(unified, "exact_weighted_sum_ratio", unexpected)
    expected = 0.5 * sum(sum(value**2 for value in row) for row in rows)
    assert unified._total_energy_from_fields(*_energy_maps(rows)) == expected


def test_normalized_energy_does_not_redefine_the_raw_density_api():
    maps = _energy_maps(((1.1e-162,) * 5,))
    assert unified._energy_density_from_fields(*maps) == {0: 0.0}
    assert unified._total_energy_from_fields(*maps) == float.fromhex(
        "0x0.0000000000001p-1022"
    )


def test_normalized_energy_requires_matching_field_support():
    maps = _energy_maps(((1.0,) * 5,))
    maps[1]["extra"] = 0.0
    with pytest.raises(ValueError, match="identical node support"):
        unified._total_energy_from_fields(*maps)


def test_suite_live_and_snapshot_charge_share_a_cancelling_field_total():
    graph = nx.complete_graph(4)
    for node, pressure in enumerate((1e16, 1.0, 0.0, -1e16)):
        graph.nodes[node].update(delta_nfr=pressure, theta=0.0)
    # A finite-energy K4 counterpart of the extreme cancellation control:
    # represented potential/charge is (-1e16, 0, 1, +1e16), whose sum is 1.
    suite = unified.compute_unified_field_suite(graph)
    assert tuple(suite["charge_density"].values()) == (-1e16, 0.0, 1.0, 1e16)
    assert suite["conservation_metrics"]["noether_charge"] == 1.0
    assert conservation.compute_noether_charge(graph) == 1.0
    tracker = conservation.ConservationTracker(graph)
    tracker.record(0.0)
    tracker.record(1.0)
    assert tracker.report().total_charge == [1.0, 1.0]
    assert tracker.latest_balance.total_charge_after == 1.0


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


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_field_correlation_retains_paired_variation_at_extreme_scales(scale):
    left = {0: -scale, 1: 0.0, 2: scale}
    right = {2: -scale, 0: scale, 1: 0.0}
    correlations = unified.analyze_field_correlations(
        {"left": left, "right": right, "text": {n: str(v) for n, v in left.items()}}
    )
    assert correlations == pytest.approx({"left_vs_right": -1.0})


@pytest.mark.parametrize("kind", GRAPH_KINDS)
def test_array_facades_preserve_mixed_node_labels_and_declared_alignment(kind):
    graph = _graph(kind)
    before = deepcopy(dict(graph.nodes(data=True)))
    nodes = tuple(graph)
    complex_view = fields.compute_complex_geometric_field_arrays(graph)
    emergent_view = fields.compute_emergent_fields(graph)
    tensor_view = fields.compute_tensor_invariants(graph)
    psi = unified.compute_complex_geometric_field(graph)
    for view in (complex_view, emergent_view, tensor_view):
        assert view["nodes"] == nodes
        assert view["num_nodes"] == len(nodes)
    assert complex_view["psi_real"].tolist() == [psi[node].real for node in nodes]
    assert complex_view["psi_imag"].tolist() == [psi[node].imag for node in nodes]
    expected = unified.compute_unified_field_suite(graph)
    for key in ("chirality", "symmetry_breaking", "coherence_coupling"):
        assert emergent_view[key].tolist() == [expected[key][node] for node in nodes]
    for key, original in (
        ("energy_density", "energy_density"),
        ("topological_charge", "historical_q_density"),
        ("conservation_density", "charge_density"),
    ):
        assert tensor_view[key].tolist() == [expected[original][node] for node in nodes]
    assert dict(graph.nodes(data=True)) == before


def test_snapshot_conservation_is_unavailable_under_relabeling_and_on_empty_graph():
    graph = nx.path_graph(4)
    for node in graph:
        graph.nodes[node].update(theta=0.0, delta_nfr=float(2**node), EPI=0.5, nu_f=1.0)
    relabeled = nx.relabel_nodes(graph, {0: 0, 1: 3, 2: 1, 3: 2})
    observations = [
        fields.compute_tensor_invariants(g) for g in (graph, relabeled, nx.Graph())
    ]
    for result in observations:
        assert result["conservation_quality"] is None
        assert result["conservation_sample_available"] is False
        assert result["conservation_scope"] == "single_snapshot_no_temporal_balance"
    assert (
        observations[0]["conservation_density"].tolist()
        == observations[1]["conservation_density"].tolist()
    )
    # These same densities produced different fictitious scores when sorted by label.
    assert observations[1]["nodes"] != tuple(sorted(relabeled))


def test_full_field_facade_reuses_one_real_canonical_snapshot(monkeypatch):
    graph = _graph()
    original = fields.compute_structural_telemetry
    captured = []

    def snapshot(current):
        result = original(current)
        captured.append(deepcopy(result))
        return result

    def duplicate(*args, **kwargs):
        pytest.fail("a captured algebraic view must not request another base field")

    monkeypatch.setattr(fields, "compute_structural_telemetry", snapshot)
    monkeypatch.setattr(fields, "compute_extended_canonical_suite", duplicate)
    for name in READERS:
        monkeypatch.setattr(unified, name, duplicate)
    result = fields.compute_unified_telemetry(graph)
    assert len(captured) == 1
    assert result["canonical"] == captured[0]
    assert result["extended_canonical"] == {
        "phase_current": captured[0]["j_phi"],
        "dnfr_flux": captured[0]["j_dnfr"],
    }
    assert result["conservation"]["structural_energy"] == pytest.approx(
        0.5 * sum(result["tensor_invariants"]["energy_density"])
    )
    # The auxiliary block also consumes the same coordinates, without a graph recapture.
    assert result["symplectic_substrate"]["phase_space_dimension"] == 4 * len(graph)
    assert result["optional_sector_status"] == {
        "symplectic_substrate": {
            "available": True,
            "source": "tnfr.physics.symplectic_substrate",
            "scope": "auxiliary_symplectic_model",
            "error": None,
        },
        "pulse": {
            "available": True,
            "source": "tnfr.physics.structural_diffusion.compute_emergent_pulse",
            "scope": "auxiliary_graph_wave_spectrum",
            "error": None,
        },
        "resonance": {
            "available": True,
            "source": "tnfr.physics.structural_diffusion.compute_nodal_pulse",
            "scope": "stored_capacity_phase_readout",
            "error": None,
        },
    }
    result["extended_canonical"]["phase_current"].clear()
    assert result["canonical"]["j_phi"] == captured[0]["j_phi"]


@pytest.mark.parametrize(
    "sector, error_message",
    [
        ("symplectic_substrate", "n_nodes must be >= 1"),
        ("pulse", "This transport formula requires symmetric adjacency"),
        (
            "resonance",
            "local phase synchrony is unavailable: coherence is disabled",
        ),
    ],
)
def test_optional_telemetry_reports_real_domain_failure_per_sector(
    sector, error_message
):
    if sector == "symplectic_substrate":
        graph = nx.Graph()
    else:
        graph = nx.DiGraph([(0, 1)]) if sector == "pulse" else nx.path_graph(3)
        for node in graph:
            graph.nodes[node].update(theta=0.1 * node, delta_nfr=0.2 * node, nu_f=1.0)
        if sector == "resonance":
            graph.graph["COHERENCE"] = {"enabled": False}

    result = fields.compute_unified_telemetry(graph)
    assert result[sector] == {}
    status = result["optional_sector_status"]
    assert status[sector]["available"] is False
    assert status[sector]["error"] == {"type": "ValueError", "message": error_message}
    for name in status.keys() - {sector}:
        assert status[name]["available"] is True
        assert status[name]["error"] is None
        assert result[name]
    assert set(result["canonical"]["phi_s"]) == set(graph)
    assert result["tensor_invariants"]["conservation_sample_available"] is False


def test_optional_telemetry_capture_does_not_hide_required_field_errors(monkeypatch):
    from tnfr.physics import structural_diffusion

    graph = _graph()
    set_attr(graph.nodes["a"], ALIAS_DNFR, float("nan"))

    def unexpected(*args, **kwargs):
        pytest.fail("Required field rejection must precede optional readouts")

    monkeypatch.setattr(structural_diffusion, "compute_emergent_pulse", unexpected)
    with pytest.raises(ValueError):
        fields.compute_unified_telemetry(graph)


def test_phase_winding_facade_uses_shared_support_and_branch_admission():
    from tnfr.physics.emergent_particles import winding_number, winding_ring

    graph = winding_ring(5, 1)
    order = list(graph)
    assert (
        fields.compute_phase_winding(graph, order)
        == winding_number(graph, order=order)[0]
        == 1
    )
    graph.remove_edge(0, 1)
    for read in (
        fields.compute_phase_winding,
        lambda g, nodes: winding_number(g, order=nodes),
    ):
        with pytest.raises(ValueError, match="undefined"):
            read(graph, order)


def test_tensor_demo_reports_actual_density_instead_of_missing_key_zero(
    capsys, monkeypatch
):
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/02_physics_regimes/33_complex_field_unification.py"
    )
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(theta=0.0, delta_nfr=1.0)
    # Test display wiring on a supplied fixture, not its separate auxiliary flow.
    monkeypatch.setattr(example, "_build_graph", lambda *args: graph)
    monkeypatch.setattr(example, "_evolve_step", lambda graph: None)
    example.demo_tensor_invariants()
    rows = [
        row
        for row in capsys.readouterr().out.splitlines()
        if row.strip().startswith(("WS (N=40)", "BA (N=40)", "Grid (6x6)"))
    ]
    assert len(rows) == 3
    # rho = Phi_s + K = (5/4, 2, 5/4) for unit P3 and unit pressure.
    assert all(float(row.split()[-1]) == 1.5 for row in rows)


def test_supplied_snapshot_showcase_reuses_readouts_without_evolution(
    capsys, monkeypatch
):
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/08_emergent_geometry/unified_fields_showcase.py"
    )
    snapshots = example.build_snapshots(17)
    originals = deepcopy(
        {name: dict(graph.nodes(data=True)) for name, graph in snapshots.items()}
    )
    calls = []

    def read(graph):
        calls.append(graph)
        return fields.compute_unified_telemetry(graph)

    monkeypatch.setattr(example, "build_snapshots", lambda seed: snapshots)
    monkeypatch.setattr(example, "compute_unified_telemetry", read)
    reports = example.analyze_snapshots(17)
    example.print_snapshot_report(reports)
    output = capsys.readouterr().out
    assert output.count("Temporal conservation: unavailable (single snapshot)") == 3
    assert "validated" not in output
    assert calls == list(snapshots.values())
    assert {
        name: dict(graph.nodes(data=True)) for name, graph in snapshots.items()
    } == originals
    assert all("_t" not in graph.graph for graph in snapshots.values())

"""Branch-aware phase-winding telemetry and U3 boundary checks."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    Emission,
    Mutation,
    Silence,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.physics.emergent_particles import (
    WindingSector,
    classify_particle,
    classify_winding_sector,
    winding_number,
    winding_ring,
)
from tnfr.physics.winding_certificates import (
    certify_phase_winding,
    observe_winding_word,
)


def _executable_ring(nodes: int = 8, winding: float = 1.0) -> nx.Graph:
    graph = winding_ring(nodes, winding)
    for _, data in graph.nodes(data=True):
        data["νf"] = float(data["nu_f"])
        data["vf"] = float(data["nu_f"])
        data["ΔNFR"] = float(data["dnfr"])
        data["glyph_history"] = []
    graph.nodes[0]["epi_history"] = [0.0, 0.2, 1.0]
    return graph


def test_orientation_reversal_flips_only_signed_winding():
    graph = _executable_ring()
    forward = certify_phase_winding(graph, range(8))
    reverse = certify_phase_winding(graph, reversed(range(8)))
    assert forward.winding == 1
    assert reverse.winding == -1
    assert forward.absolute_winding == reverse.absolute_winding == 1


def test_relabeling_preserves_declared_oriented_winding():
    graph = _executable_ring()
    mapping = {node: f"node-{(3 * node) % 8}" for node in graph}
    relabelled = nx.relabel_nodes(graph, mapping)
    original = certify_phase_winding(graph, range(8))
    transformed = certify_phase_winding(
        relabelled, [mapping[node] for node in range(8)]
    )
    assert transformed.winding == original.winding
    assert transformed.raw_winding == pytest.approx(original.raw_winding)


def test_phase_perturbation_below_margin_preserves_winding():
    graph = _executable_ring()
    before = certify_phase_winding(graph, range(8))
    margin = min(before.minimum_u3_margin, before.minimum_branch_margin)
    perturbation = margin / 8
    for node in graph:
        graph.nodes[node]["theta"] += perturbation * math.sin(
            2 * math.pi * node / 8
        )
    after = certify_phase_winding(graph, range(8))
    assert after.winding == before.winding == 1
    assert after.u3_admissible
    assert after.minimum_branch_margin > 0.0


def test_branch_boundary_is_undefined_instead_of_rounded():
    graph = _executable_ring()
    epsilon = 1e-5
    states = []
    for phase in (math.pi - epsilon, math.pi, math.pi + epsilon):
        graph.nodes[1]["theta"] = phase
        states.append(certify_phase_winding(graph, range(8)))
    assert states[0].is_defined and states[0].winding == 1
    assert not states[1].is_defined and states[1].winding is None
    assert states[1].cycle_exists
    assert "branch boundary" in states[1].reason
    assert states[2].is_defined and states[2].winding == 0


def test_missing_cycle_is_an_undefined_observation():
    graph = _executable_ring()
    graph.remove_edge(3, 4)
    certificate = certify_phase_winding(graph, range(8))
    assert not certificate.is_defined
    assert not certificate.cycle_exists
    assert certificate.winding is None


def test_direct_certificate_does_not_fabricate_zero_for_missing_phases():
    certificate = certify_phase_winding(nx.cycle_graph(5), range(5))
    assert not certificate.is_defined
    assert certificate.cycle_exists
    assert certificate.winding is None
    assert certificate.raw_winding is None
    assert "phase is missing" in certificate.reason


@pytest.mark.parametrize("bad_phase", [None, "0.2", float("nan"), float("inf")])
def test_direct_certificate_rejects_invalid_phase_values(bad_phase):
    graph = winding_ring(5, 1)
    graph.nodes[2]["theta"] = bad_phase
    certificate = certify_phase_winding(graph, range(5))
    assert not certificate.is_defined
    assert certificate.cycle_exists
    assert certificate.winding is None
    assert "phase at cycle node 2" in certificate.reason


def test_finite_extreme_phases_are_reduced_before_differencing():
    graph = _executable_ring(nodes=5)
    graph.nodes[0]["theta"] = 1e308
    graph.nodes[1]["theta"] = -1e308

    certificate = certify_phase_winding(graph, range(5))

    assert certificate.is_defined
    assert isinstance(certificate.winding, int)
    assert math.isfinite(certificate.raw_winding)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"branch_tolerance": True},
        {"branch_tolerance": "0.1"},
        {"phase_gate": True},
        {"phase_gate": float("nan")},
    ],
)
def test_certificate_rejects_ambiguous_numeric_parameters(kwargs):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        certify_phase_winding(_executable_ring(), range(8), **kwargs)


@pytest.mark.parametrize("bad_phase", [None, float("nan")])
def test_winding_word_rejects_invalid_phase_before_mutation(bad_phase):
    graph = _executable_ring()
    if bad_phase is None:
        graph.nodes[2].pop("theta")
        graph.nodes[2].pop("phase")
    else:
        graph.nodes[2]["theta"] = bad_phase
        graph.nodes[2]["phase"] = bad_phase
    history_before = tuple(graph.nodes[0]["glyph_history"])
    epi_before = graph.nodes[0]["EPI"]

    with pytest.raises(ValueError, match="defined initial cycle"):
        observe_winding_word(
            graph,
            range(8),
            0,
            [Emission(), Coherence(), Silence()],
        )

    assert tuple(graph.nodes[0]["glyph_history"]) == history_before
    assert graph.nodes[0]["EPI"] == epi_before


def test_legacy_particle_api_returns_only_winding_sector_labels():
    results = {
        target: classify_particle(winding_ring(12, target))
        for target in (0, 1, 2)
    }
    labels = {target: result.winding_class for target, result in results.items()}
    assert labels == {
        0: "zero-winding class",
        1: "unit-winding defect class",
        2: "multi-winding defect class (|W|=2)",
    }
    assert all(
        term not in label.lower()
        for label in labels.values()
        for term in ("boson", "fermion", "matter", "particle")
    )
    assert all(
        result.particle_class == result.winding_class
        and result.as_dict()["winding_class"] == result.winding_class
        for result in results.values()
    )


def test_preferred_winding_sector_api_exposes_nonphysical_aliases():
    result = classify_winding_sector(winding_ring(12, -1))
    assert isinstance(result, WindingSector)
    assert result.winding_class == "unit-winding defect class"
    assert result.orientation_sign == -1
    assert result.q_density_mean == result.global_q_density_mean
    assert result.global_energy_density_mean == result.energy_density
    assert result.is_integral == result.is_quantized
    assert result.telemetry_available
    assert result.telemetry_scope == "whole_graph_snapshot"
    payload = result.as_dict()
    assert payload["orientation_sign"] == -1
    assert payload["q_density_mean"] == result.q_density_mean
    assert payload["is_integral"] is True


@pytest.mark.parametrize("n_nodes", [True, 3.0, 3.5, "3"])
def test_winding_ring_rejects_nonintegral_node_counts(n_nodes):
    with pytest.raises(TypeError, match="n_nodes must be an integer"):
        winding_ring(n_nodes, 1)


@pytest.mark.parametrize("winding", [True, "1", complex(1, 0)])
def test_winding_ring_rejects_nonreal_winding(winding):
    with pytest.raises(TypeError, match="winding must be a finite real number"):
        winding_ring(8, winding)


@pytest.mark.parametrize(
    "winding", [float("nan"), float("inf"), float("-inf"), 10**1000]
)
def test_winding_ring_rejects_nonfinite_winding(winding):
    with pytest.raises(ValueError, match="winding must be finite"):
        winding_ring(8, winding)


def test_winding_ring_handles_very_large_finite_winding_modularly():
    graph = winding_ring(8, 1e308)
    assert all(
        math.isfinite(graph.nodes[node]["theta"])
        and math.isfinite(graph.nodes[node]["phase"])
        for node in graph
    )


@pytest.mark.parametrize("base_dnfr", [True, "0.1", complex(0.1, 0)])
def test_winding_ring_rejects_nonreal_baseline(base_dnfr):
    with pytest.raises(TypeError, match="base_dnfr must be a finite real number"):
        winding_ring(8, 1, base_dnfr=base_dnfr)


@pytest.mark.parametrize("base_dnfr", [float("nan"), float("inf")])
def test_winding_ring_rejects_nonfinite_baseline(base_dnfr):
    with pytest.raises(ValueError, match="base_dnfr must be finite"):
        winding_ring(8, 1, base_dnfr=base_dnfr)


def test_default_traversal_supports_heterogeneous_labels_on_a_simple_cycle():
    graph = nx.relabel_nodes(
        winding_ring(5, 1),
        {0: "north", 1: 7, 2: ("south",), 3: 3.5, 4: frozenset({1})},
    )
    winding, raw = winding_number(graph)
    assert abs(winding) == 1
    assert abs(raw) == pytest.approx(1.0)


def test_noncycle_graph_requires_an_explicit_oriented_order():
    graph = winding_ring(4, 1)
    graph.add_edge(0, 2)
    with pytest.raises(ValueError, match="order is required"):
        winding_number(graph)
    assert winding_number(graph, order=(0, 1, 2, 3))[0] == 1


def test_subcycle_result_labels_supporting_telemetry_as_graph_global():
    graph = winding_ring(4, 1)
    graph.add_node(
        "outside",
        theta=0.2,
        phase=0.2,
        delta_nfr=0.4,
        dnfr=0.4,
        coherence=0.5,
        EPI=0.5,
        nu_f=1.0,
    )
    result = classify_winding_sector(graph, order=(0, 1, 2, 3))
    assert result.cycle_nodes == (0, 1, 2, 3)
    assert result.telemetry_scope == "whole_graph_snapshot"
    assert result.telemetry_available


def test_winding_requires_an_explicit_phase_on_every_cycle_node():
    graph = winding_ring(8, 1)
    del graph.nodes[3]["theta"]
    del graph.nodes[3]["phase"]
    with pytest.raises(ValueError, match="phase is missing at cycle node 3"):
        winding_number(graph)


@pytest.mark.parametrize(
    "bad_phase",
    [None, "0.2", float("nan"), float("inf"), float("-inf")],
)
def test_winding_rejects_invalid_cycle_phase(bad_phase):
    graph = winding_ring(8, 1)
    graph.nodes[3]["theta"] = bad_phase
    with pytest.raises(ValueError, match="phase at cycle node 3"):
        winding_number(graph)


def test_missing_dnfr_makes_supporting_telemetry_explicitly_unavailable():
    graph = winding_ring(8, 1)
    for _, data in graph.nodes(data=True):
        data.pop("delta_nfr")
        data.pop("dnfr")

    result = classify_winding_sector(graph)

    assert result.winding == 1
    assert result.energy_density is None
    assert result.q_density_mean is None
    assert not result.telemetry_available
    assert result.telemetry_scope == "whole_graph_snapshot"
    assert "delta_nfr" in result.telemetry_unavailable_reason
    payload = result.as_dict()
    assert payload["energy_density"] is None
    assert payload["q_density_mean"] is None
    assert payload["telemetry_unavailable_reason"]


def test_invalid_off_cycle_phase_marks_global_telemetry_unavailable():
    graph = winding_ring(4, 1)
    graph.add_node("outside", theta=float("nan"), delta_nfr=0.1)

    result = classify_winding_sector(graph, order=(0, 1, 2, 3))

    assert result.winding == 1
    assert not result.telemetry_available
    assert result.energy_density is None
    assert "phase at node 'outside' must be finite" == (
        result.telemetry_unavailable_reason
    )


def test_telemetry_failure_propagates_instead_of_fabricating_zero(monkeypatch):
    def fail(_graph):
        raise RuntimeError("telemetry failed")

    monkeypatch.setattr(
        "tnfr.physics.emergent_particles.compute_energy_density", fail
    )
    with pytest.raises(RuntimeError, match="telemetry failed"):
        classify_winding_sector(winding_ring(8, 1))


def test_legacy_winding_tuple_api_rejects_an_absent_cycle():
    graph = winding_ring(8, 1)
    graph.remove_edge(3, 4)
    with pytest.raises(ValueError, match="declared oriented cycle is absent"):
        winding_number(graph, order=list(range(8)))


def test_u3_rejects_inadmissible_high_winding_word():
    graph = _executable_ring(winding=3.0)
    certificate = certify_phase_winding(graph, range(8))
    assert certificate.winding == 3
    assert certificate.u3_admissible is False
    with pytest.raises(OperatorPreconditionError, match="U3 phase gate"):
        observe_winding_word(
            graph,
            range(8),
            0,
            [Emission(), Coupling(), Coherence(), Silence()],
        )
    assert tuple(graph.nodes[0]["glyph_history"]) == ("AL",)


def test_validated_words_record_actual_history_and_preserve_cycle():
    graph = _executable_ring()
    no_phase_change = observe_winding_word(
        graph, range(8), 0, [Emission(), Coherence(), Silence()]
    )
    assert no_phase_change.history_preserved
    assert no_phase_change.actual_history == ("AL", "IL", "SHA")
    assert all(not step.phase_changes for step in no_phase_change.steps)
    assert all(step.certificate.winding == 1 for step in no_phase_change.steps)

    graph = _executable_ring()
    coupled = observe_winding_word(
        graph,
        range(8),
        0,
        [Emission(), Coupling(), Coherence(), Silence()],
    )
    assert coupled.history_preserved
    assert coupled.steps[1].operator == "UM"
    assert coupled.steps[1].phase_changes
    assert coupled.steps[1].certificate.u3_admissible
    assert coupled.steps[-1].certificate.winding == 1

    graph = _executable_ring()
    mutated = observe_winding_word(
        graph,
        range(8),
        0,
        [
            Emission(),
            Coherence(),
            Dissonance(),
            Mutation(),
            Coherence(),
            Silence(),
        ],
    )
    assert mutated.history_preserved
    assert mutated.actual_history == ("AL", "IL", "OZ", "ZHIR", "IL", "SHA")
    assert mutated.steps[3].phase_changes
    assert mutated.steps[3].certificate.winding == 1
    assert mutated.steps[3].certificate.minimum_branch_margin > 0.0

"""The auxiliary Hermitian model is not a nonzero nodal pressure law."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.dnfr import compute_delta_nfr_hamiltonian
from tnfr.operators.hamiltonian import InternalHamiltonian, build_H_frequency


def _graph():
    graph = nx.path_graph(3)
    graph.graph["COHERENCE"] = {
        "enabled": True,
        "scope": "neighbors",
        "weights": {"phase": 1.0, "epi": 0.0, "vf": 0.0, "si": 0.0},
    }
    for node in graph:
        graph.nodes[node].update(
            EPI=0.125 * node,
            nu_f=0.5 + node,
            phase=0.25 * node,
            Si=0.8,
            delta_nfr=7.0,
        )
    return graph


def test_localized_projector_readout_is_zero_despite_nontrivial_matrix():
    graph = _graph()
    ham = InternalHamiltonian(graph, hbar_str=2.0)
    assert np.count_nonzero(ham.H_int - np.diag(np.diag(ham.H_int))) > 0
    for index, node in enumerate(ham.nodes):
        projector = np.zeros_like(ham.H_int)
        projector[index, index] = 1.0
        commutator = ham.H_int @ projector - projector @ ham.H_int
        assert commutator[index, index] == 0.0
        assert ham.compute_node_delta_nfr(node) == 0.0
    with pytest.raises(ValueError, match="not found"):
        ham.compute_node_delta_nfr("missing")

    compute_delta_nfr_hamiltonian(graph)
    assert all(get_attr(data, ALIAS_DNFR) == 0.0 for _, data in graph.nodes(data=True))
    assert "identically zero" in graph.graph["_DNFR_META"]["note"]


def test_legacy_matrix_sign_and_auxiliary_unitary_spectrum():
    ham = InternalHamiltonian(_graph(), hbar_str=2.0)
    positive_generator = ham.compute_delta_nfr_operator()
    np.testing.assert_array_equal(positive_generator, 0.5j * ham.H_int)
    np.testing.assert_array_equal(positive_generator.conj().T, -positive_generator)
    spectrum, modes = ham.get_spectrum()
    time = 0.125
    expected = modes @ np.diag(np.exp(-0.5j * spectrum * time)) @ modes.conj().T
    np.testing.assert_allclose(ham.time_evolution_operator(time), expected, atol=1e-14)


def test_wrapper_rebuilds_consumed_attributes_configuration_and_scale():
    graph = _graph()
    compute_delta_nfr_hamiltonian(graph)
    first = graph.graph["_hamiltonian_cache"]
    original_affinity = first.H_coh.copy()
    graph.nodes[0]["nu_f"] = 4.0
    graph.nodes[0]["phase"] = 1.5
    graph.graph["H_COUPLING_STRENGTH"] = 0.75
    graph.graph["HBAR_STR"] = 2.0

    compute_delta_nfr_hamiltonian(graph)

    current = graph.graph["_hamiltonian_cache"]
    assert current is not first
    assert current.H_freq[0, 0] == 4.0
    assert first.H_freq[0, 0] == 0.5
    assert not np.array_equal(current.H_coh, original_affinity)
    assert current.H_coupling[0, 1] == 0.75
    assert current.hbar_str == 2.0

    graph.remove_node(2)
    graph.add_node(9, nu_f=3.0, EPI=0.0, phase=0.0)
    graph.add_edge(1, 9)
    compute_delta_nfr_hamiltonian(graph)
    assert graph.graph["_hamiltonian_cache"].nodes == (0, 1, 9)
    assert get_attr(graph.nodes[9], ALIAS_DNFR) == 0.0
    compute_delta_nfr_hamiltonian(graph, cache_hamiltonian=False)
    assert "_hamiltonian_cache" not in graph.graph


@pytest.mark.parametrize("scale", [0.0, math.nan, math.inf, True, "1.0"])
def test_invalid_auxiliary_scale_fails_before_pressure_writes(scale):
    graph = _graph()
    with pytest.raises((TypeError, ValueError), match="hbar_str"):
        compute_delta_nfr_hamiltonian(graph, hbar_str=scale)
    assert all(get_attr(data, ALIAS_DNFR) == 7.0 for _, data in graph.nodes(data=True))
    assert "_hamiltonian_cache" not in graph.graph


def test_frequency_builder_and_constructor_reject_nonfinite_entries():
    graph = _graph()
    graph.graph["COHERENCE"] = {"enabled": False}
    graph.nodes[1]["nu_f"] = math.nan
    for build in (build_H_frequency, InternalHamiltonian):
        with pytest.raises(ValueError, match="nu_f must be finite"):
            build(graph)


def test_nonfinite_affinity_cannot_pass_hermiticity_check(monkeypatch):
    monkeypatch.setattr(
        InternalHamiltonian,
        "_build_H_coherence",
        lambda self: np.full((self.N, self.N), math.nan, dtype=complex),
    )
    with pytest.raises(ValueError, match="H_coh must contain only finite"):
        InternalHamiltonian(_graph())

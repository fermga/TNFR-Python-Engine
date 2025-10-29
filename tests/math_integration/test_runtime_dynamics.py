from __future__ import annotations

import importlib

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics import (
    HilbertSpace,
    MathematicalDynamicsEngine,
    NFRValidator,
    build_coherence_operator,
    build_delta_nfr,
    build_frequency_operator,
)
from tnfr.node import NodeNX
from tnfr.operators.definitions import Emission, Resonance
from tnfr.structural import create_nfr

from tests.helpers.mathematics import build_node_with_operators


@pytest.mark.parametrize("ops", [[Emission()], [Emission(), Resonance()]])
def test_run_sequence_with_validation_reports_metrics(ops):
    node, hilbert, validator = build_node_with_operators()
    result = node.run_sequence_with_validation(ops)

    assert set(result).issuperset({"pre", "post", "validation"})
    assert result["pre"]["metrics"]["normalized"]["passed"] is True
    assert result["post"]["metrics"]["normalized"]["passed"] is True
    frequency_summary = result["post"]["metrics"].get("frequency")
    assert isinstance(frequency_summary, dict)
    assert frequency_summary["passed"] is True

    validation = result["validation"]
    assert validation is not None
    assert validation["passed"] is True
    assert "report" in validation


def test_run_sequence_with_validation_respects_frequency_override():
    node, _, _ = build_node_with_operators()
    outcome = node.run_sequence_with_validation([], freq_op=None, enable_validation=False)
    assert "frequency" not in outcome["post"]["metrics"]
    assert outcome["validation"] is None


def test_mathematical_dynamics_engine_matches_analytic_solution():
    hilbert = HilbertSpace(2)
    generator = np.diag([1.0, -1.0])
    engine = MathematicalDynamicsEngine(generator, hilbert_space=hilbert, use_scipy=False)
    state = np.array([1.0 + 0j, 0.0 + 0j])
    evolved = engine.step(state, dt=np.pi)
    assert np.allclose(evolved, np.array([-1.0 + 0j, 0.0 + 0j]))

    trajectory = engine.evolve(state, steps=2, dt=np.pi / 2)
    assert trajectory.shape == (3, 2)
    assert np.allclose(trajectory[0], state)


def test_build_delta_nfr_variants_are_hermitian_and_reproducible():
    graph = nx.cycle_graph(4)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    laplacian_matrix = build_delta_nfr(graph, rng=rng1, noise_scale=0.1)
    adjacency_matrix = build_delta_nfr(graph, variant="adjacency")
    repeat_matrix = build_delta_nfr(graph, rng=rng2, noise_scale=0.1)

    assert np.allclose(laplacian_matrix, laplacian_matrix.conj().T)
    assert np.allclose(adjacency_matrix, adjacency_matrix.conj().T)
    assert np.allclose(laplacian_matrix, repeat_matrix)


def test_operator_factory_wiring_creates_valid_node():
    G, node_id = create_nfr("factory-node")
    coherence = np.array([[0.9, 0.1], [0.1, 0.8]])
    frequency = np.array([[0.5, 0.2], [0.2, 0.5]])
    hilbert = HilbertSpace(2)
    validator = NFRValidator(
        hilbert,
        build_coherence_operator(coherence),
        coherence_threshold=0.0,
        frequency_operator=build_frequency_operator(frequency),
    )
    node = NodeNX(
        G,
        node_id,
        hilbert_space=hilbert,
        coherence_operator=coherence,
        frequency_operator=frequency,
        coherence_threshold=0.0,
        validator=validator,
        enable_math_validation=True,
    )

    summary = node.run_sequence_with_validation([], enable_validation=True)
    assert summary["validation"]["passed"] is True


def test_new_modules_import_without_cycles():
    for module in (
        "tnfr.mathematics.runtime",
        "tnfr.mathematics.dynamics",
        "tnfr.mathematics.generators",
        "tnfr.mathematics.operators_factory",
    ):
        importlib.import_module(module)

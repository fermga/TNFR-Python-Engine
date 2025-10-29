from __future__ import annotations

import importlib

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics import (
    HilbertSpace,
    NFRValidator,
    build_coherence_operator,
    build_delta_nfr,
    build_frequency_operator,
)
from tnfr.node import NodeNX
from tnfr.operators.definitions import Coherence, Emission, Reception, Resonance, Transition
from tnfr.structural import create_nfr

from tests.helpers.compare_classical import (
    DEFAULT_ACCEPTANCE_OPS,
    math_sequence_summary,
)
from tests.helpers.mathematics import build_node_with_operators, make_dynamics_engine


@pytest.mark.parametrize(
    "ops",
    [
        [Emission(), Reception(), Coherence(), Resonance(), Transition()],
        [Emission(), Reception(), Coherence(), Resonance(), Transition(), Transition()],
    ],
)
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
    node, _, _ = build_node_with_operators(frequency_value=None)
    outcome = node.run_sequence_with_validation(
        list(DEFAULT_ACCEPTANCE_OPS), freq_op=None, enable_validation=False
    )
    assert "frequency" not in outcome["post"]["metrics"]
    assert outcome["validation"] is None


def test_run_sequence_validation_summary_aligns_with_metrics():
    node, _, _ = build_node_with_operators()
    result = node.run_sequence_with_validation(list(DEFAULT_ACCEPTANCE_OPS))

    validation = result["validation"]
    assert validation and validation["passed"] is True
    summary = validation["summary"]

    post_metrics = result["post"]["metrics"]
    assert summary["coherence"]["value"] == pytest.approx(
        post_metrics["coherence"]["value"]
    )
    freq_metric = post_metrics["frequency"]
    freq_summary = summary["frequency"]
    assert isinstance(freq_summary, dict)
    assert freq_summary["value"] == pytest.approx(freq_metric["value"])
    assert freq_summary["projection_passed"] is freq_metric["projection_passed"]


def test_mathematical_dynamics_engine_matches_analytic_solution():
    hilbert = HilbertSpace(2)
    generator = np.diag([1.0, -1.0])
    engine = make_dynamics_engine(generator, hilbert_space=hilbert, use_scipy=False)
    state = np.array([1.0 + 0j, 0.0 + 0j])
    evolved = engine.step(state, dt=np.pi)
    assert np.allclose(evolved, np.array([-1.0 + 0j, 0.0 + 0j]))

    trajectory = engine.evolve(state, steps=2, dt=np.pi / 2)
    assert trajectory.shape == (3, 2)
    assert np.allclose(trajectory[0], state)


def test_mathematical_dynamics_engine_reproducibility_without_rng():
    hilbert = HilbertSpace(3)
    generator = np.diag([0.2, -0.1, 0.05])
    engine = make_dynamics_engine(generator, hilbert_space=hilbert, use_scipy=False)
    state = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j])

    first = engine.evolve(state, steps=4, dt=0.3)
    second = engine.evolve(state, steps=4, dt=0.3)
    assert np.allclose(first, second)


def test_mathematical_dynamics_engine_matches_scipy_when_available():
    pytest.importorskip("scipy.linalg")
    hilbert = HilbertSpace(2)
    generator = np.array([[0.3, 0.0], [0.0, -0.3]], dtype=np.complex128)
    numpy_engine = make_dynamics_engine(generator, hilbert_space=hilbert, use_scipy=False)
    scipy_engine = make_dynamics_engine(generator, hilbert_space=hilbert, use_scipy=True)

    state = np.array([np.sqrt(0.5) + 0j, np.sqrt(0.5) + 0j])
    numpy_step = numpy_engine.step(state, dt=0.5)
    scipy_step = scipy_engine.step(state, dt=0.5)
    assert np.allclose(numpy_step, scipy_step)


def test_build_delta_nfr_variants_are_hermitian_and_reproducible():
    graph = nx.cycle_graph(4)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    laplacian_matrix = None
    repeat_matrix = None
    try:
        laplacian_matrix = build_delta_nfr(graph, rng=rng1, noise_scale=0.1)
        repeat_matrix = build_delta_nfr(graph, rng=rng2, noise_scale=0.1)
    except ModuleNotFoundError:
        laplacian_matrix = None
        repeat_matrix = None

    adjacency_matrix = build_delta_nfr(graph, variant="adjacency")
    assert np.allclose(adjacency_matrix, adjacency_matrix.conj().T)

    if laplacian_matrix is not None and repeat_matrix is not None:
        assert np.allclose(laplacian_matrix, laplacian_matrix.conj().T)
        assert np.allclose(laplacian_matrix, repeat_matrix)
    else:
        repeat_adjacency = build_delta_nfr(graph, variant="adjacency")
        assert np.allclose(adjacency_matrix, repeat_adjacency)


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

    summary = node.run_sequence_with_validation(list(DEFAULT_ACCEPTANCE_OPS), enable_validation=True)
    assert summary["validation"]["passed"] is True


def test_run_sequence_with_validation_is_reproducible_with_seed():
    summary_one, node_one = math_sequence_summary(DEFAULT_ACCEPTANCE_OPS, rng_seed=2024)
    summary_two, node_two = math_sequence_summary(DEFAULT_ACCEPTANCE_OPS, rng_seed=2024)
    summary_three, _node_three = math_sequence_summary(DEFAULT_ACCEPTANCE_OPS, rng_seed=2025)

    np.testing.assert_allclose(summary_one["pre"]["state"], summary_two["pre"]["state"])
    np.testing.assert_allclose(summary_one["post"]["state"], summary_two["post"]["state"])
    assert summary_one["post"]["metrics"] == summary_two["post"]["metrics"]
    assert summary_one["validation"] == summary_two["validation"]

    assert not np.allclose(summary_one["post"]["state"], summary_three["post"]["state"])

    graph_one = node_one.G
    graph_two = node_two.G
    assert graph_one is not graph_two
    assert set(graph_one.nodes) == set(graph_two.nodes)


def test_new_modules_import_without_cycles():
    for module in (
        "tnfr.mathematics.runtime",
        "tnfr.mathematics.dynamics",
        "tnfr.mathematics.generators",
        "tnfr.mathematics.operators_factory",
    ):
        importlib.import_module(module)

from __future__ import annotations

import importlib

import numpy as np
import pytest

from tnfr.config import context_flags
from tnfr.mathematics import (
    CoherenceOperator,
    HilbertSpace,
    MathematicalDynamicsEngine,
    make_frequency_operator,
)
from tnfr.validation import NFRValidator
from tnfr.node import NodeNX
from tnfr.operators.definitions import (
    Coherence,
    Emission,
    Reception,
    Resonance,
    Transition,
)
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

    assert set(result).issuperset(
        {"pre_state", "post_state", "pre_metrics", "post_metrics", "validation"}
    )
    assert result["pre_metrics"]["normalized"] is True
    assert result["post_metrics"]["normalized"] is True
    frequency_positive = result["post_metrics"].get("frequency_positive")
    if frequency_positive is not None:
        assert frequency_positive is True
        assert "frequency_expectation" in result["post_metrics"]

    validation = result["validation"]
    assert validation is not None
    assert validation["passed"] is True
    assert "report" in validation


def test_run_sequence_with_validation_respects_frequency_override():
    node, _, _ = build_node_with_operators(frequency_value=None)
    outcome = node.run_sequence_with_validation(
        list(DEFAULT_ACCEPTANCE_OPS), frequency_operator=None, enable_validation=False
    )
    assert "frequency_positive" not in outcome["post_metrics"]
    assert outcome["validation"] is None


def test_run_sequence_with_validation_logging_respects_flags(caplog):
    node, _, _ = build_node_with_operators()

    with context_flags(log_performance=True):
        with caplog.at_level("DEBUG", logger="tnfr.node"):
            node.run_sequence_with_validation(
                list(DEFAULT_ACCEPTANCE_OPS), enable_validation=False, log_metrics=False
            )
    node_records = [record for record in caplog.records if record.name == "tnfr.node"]
    assert not node_records

    caplog.clear()

    with context_flags(log_performance=True):
        with caplog.at_level("DEBUG", logger="tnfr.node"):
            node.run_sequence_with_validation(
                list(DEFAULT_ACCEPTANCE_OPS), enable_validation=False, log_metrics=True
            )

    node_records = [record for record in caplog.records if record.name == "tnfr.node"]
    assert len(node_records) == 2
    assert node_records[0].message.startswith("node_metrics.pre")
    assert node_records[1].message.startswith("node_metrics.post")


def test_run_sequence_validation_summary_aligns_with_metrics():
    node, _, _ = build_node_with_operators()
    result = node.run_sequence_with_validation(list(DEFAULT_ACCEPTANCE_OPS))

    validation = result["validation"]
    assert validation and validation["passed"] is True
    summary = validation["summary"]

    post_metrics = result["post_metrics"]
    assert summary["coherence"]["value"] == pytest.approx(post_metrics["coherence_expectation"])
    freq_metric = summary["frequency"]
    assert isinstance(freq_metric, dict)
    assert freq_metric["value"] == pytest.approx(post_metrics["frequency_expectation"])
    assert freq_metric["projection_passed"] is post_metrics["frequency_projection_passed"]


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


def test_mathematical_dynamics_engine_direct_instantiation_step():
    hilbert = HilbertSpace(2)
    generator = np.diag([1.0, -1.0])
    engine = MathematicalDynamicsEngine(generator, hilbert, use_scipy=False)
    state = np.array([1.0 + 0j, 0.0 + 0j])

    evolved = engine.step(state, dt=np.pi / 2)

    expected = np.array([-1.0j, 0.0 + 0j])
    assert np.allclose(evolved, expected)


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


def test_operator_factory_wiring_creates_valid_node():
    G, node_id = create_nfr("factory-node")
    coherence = np.array([[0.9, 0.1], [0.1, 0.8]])
    frequency = np.array([[0.5, 0.2], [0.2, 0.5]])
    hilbert = HilbertSpace(2)
    coherence_operator = CoherenceOperator(coherence)
    frequency_operator = make_frequency_operator(frequency)
    validator = NFRValidator(
        hilbert,
        coherence_operator,
        coherence_threshold=0.0,
        frequency_operator=frequency_operator,
    )
    node = NodeNX(
        G,
        node_id,
        hilbert_space=hilbert,
        coherence_operator=coherence_operator,
        frequency_operator=frequency_operator,
        coherence_threshold=0.0,
        validator=validator,
        enable_math_validation=True,
    )

    summary = node.run_sequence_with_validation(
        list(DEFAULT_ACCEPTANCE_OPS), enable_validation=True
    )
    assert summary["validation"]["passed"] is True


def test_run_sequence_with_validation_is_reproducible_with_seed():
    summary_one, node_one = math_sequence_summary(DEFAULT_ACCEPTANCE_OPS, rng_seed=2024)
    summary_two, node_two = math_sequence_summary(DEFAULT_ACCEPTANCE_OPS, rng_seed=2024)
    summary_three, _node_three = math_sequence_summary(DEFAULT_ACCEPTANCE_OPS, rng_seed=2025)

    np.testing.assert_allclose(summary_one["pre_state"], summary_two["pre_state"])
    np.testing.assert_allclose(summary_one["post_state"], summary_two["post_state"])
    assert summary_one["post_metrics"] == summary_two["post_metrics"]
    assert summary_one["validation"] == summary_two["validation"]

    assert not np.allclose(summary_one["post"]["state"], summary_three["post"]["state"])

    graph_one = node_one.G
    graph_two = node_two.G
    assert graph_one is not graph_two
    assert set(graph_one.nodes) == set(graph_two.nodes)


def test_run_sequence_with_validation_accepts_generator_instance():
    rng = np.random.default_rng(1337)
    summary_one, node_one = math_sequence_summary(DEFAULT_ACCEPTANCE_OPS, rng=rng)
    summary_two, node_two = math_sequence_summary(DEFAULT_ACCEPTANCE_OPS, rng=rng)

    np.testing.assert_allclose(summary_one["pre_state"], summary_two["pre_state"])
    np.testing.assert_allclose(summary_one["post_state"], summary_two["post_state"])
    assert summary_one["post_metrics"] == summary_two["post_metrics"]
    assert summary_one["validation"] == summary_two["validation"]

    assert node_one is not node_two


def test_new_modules_import_without_cycles():
    for module in (
        "tnfr.mathematics.runtime",
        "tnfr.mathematics.dynamics",
        "tnfr.mathematics.generators",
        "tnfr.mathematics.operators_factory",
    ):
        importlib.import_module(module)

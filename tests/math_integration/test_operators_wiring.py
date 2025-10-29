"""Integration tests for operator wiring on NodeNX."""
from __future__ import annotations

import numpy as np

from tnfr.mathematics import (
    CoherenceOperator,
    FrequencyOperator,
    HilbertSpace,
    make_frequency_operator,
)
from tnfr.node import NodeNX
from tnfr.operators.definitions import Coherence, Emission
from tnfr.structural import create_nfr

from tests.helpers.compare_classical import DEFAULT_ACCEPTANCE_OPS
from tests.helpers.mathematics import build_node_with_operators


def test_node_accepts_direct_operator_instances() -> None:
    G, node_id = create_nfr("direct-operators")
    hilbert = HilbertSpace(2)
    coherence_matrix = np.array([[0.8, 0.05], [0.05, 0.7]], dtype=np.complex128)
    frequency_matrix = np.array([[0.4, 0.0], [0.0, 0.6]], dtype=np.complex128)

    coherence_operator = CoherenceOperator(coherence_matrix)
    frequency_operator = make_frequency_operator(frequency_matrix)

    node = NodeNX(
        G,
        node_id,
        hilbert_space=hilbert,
        coherence_operator=coherence_operator,
        frequency_operator=frequency_operator,
        enable_math_validation=True,
    )

    assert node.coherence_operator is coherence_operator
    assert node.frequency_operator is frequency_operator

    summary = node.run_sequence_with_validation(
        [Emission(), Coherence()], enable_validation=True
    )
    assert summary["validation"] is not None
    assert summary["validation"]["passed"] is True


def test_node_constructs_operators_from_factory_parameters() -> None:
    G, node_id = create_nfr("factory-operators")
    hilbert = HilbertSpace(3)
    spectrum = np.array([0.3, 0.4, 0.5])
    frequency_matrix = np.diag([0.2, 0.25, 0.3]).astype(np.complex128)

    node = NodeNX(
        G,
        node_id,
        hilbert_space=hilbert,
        coherence_dim=3,
        coherence_spectrum=spectrum,
        coherence_c_min=0.25,
        frequency_matrix=frequency_matrix,
        enable_math_validation=False,
    )

    assert isinstance(node.coherence_operator, CoherenceOperator)
    assert isinstance(node.frequency_operator, FrequencyOperator)
    np.testing.assert_allclose(node.coherence_operator.spectrum().real, spectrum)
    np.testing.assert_allclose(node.frequency_operator.matrix, frequency_matrix)

    result = node.run_sequence_with_validation(
        list(DEFAULT_ACCEPTANCE_OPS), enable_validation=False
    )
    assert "coherence_expectation" in result["post_metrics"]
    assert "frequency_expectation" in result["post_metrics"]


def test_run_sequence_uses_factory_overrides() -> None:
    node, _, _ = build_node_with_operators(frequency_value=None, enable_validation=False)
    dim = node.hilbert_space.dimension
    new_spectrum = np.full(dim, 0.85)
    frequency_matrix = np.eye(dim) * 0.15

    outcome = node.run_sequence_with_validation(
        list(DEFAULT_ACCEPTANCE_OPS),
        coherence_dim=dim,
        coherence_spectrum=new_spectrum,
        coherence_c_min=0.8,
        frequency_matrix=frequency_matrix,
        enable_validation=False,
    )

    post_metrics = outcome["post_metrics"]
    assert "coherence_expectation" in post_metrics
    assert "frequency_positive" in post_metrics
    assert post_metrics["frequency_positive"] is True

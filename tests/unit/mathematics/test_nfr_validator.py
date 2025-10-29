from __future__ import annotations

import numpy as np

from tnfr.mathematics import CoherenceOperator, FrequencyOperator, HilbertSpace, NFRValidator


def _make_simple_validator(coherence_threshold: float = 0.25) -> NFRValidator:
    hilbert = HilbertSpace(dimension=2)
    coherence = CoherenceOperator([[0.5, 0.0], [0.0, 1.0]])
    frequency = FrequencyOperator([0.3, 0.6])
    return NFRValidator(
        hilbert_space=hilbert,
        coherence_operator=coherence,
        coherence_threshold=coherence_threshold,
        frequency_operator=frequency,
    )


def test_validator_successful_state_passes_all_checks() -> None:
    validator = _make_simple_validator(coherence_threshold=0.25)
    state = np.array([1.0 / np.sqrt(2.0), 1.0j / np.sqrt(2.0)], dtype=np.complex128)

    passed, summary = validator.validate_state(state)

    assert passed is True
    assert summary["normalized"] is True
    assert summary["coherence"]["passed"] is True
    assert summary["frequency"]["passed"] is True
    assert summary["unitary_stability"]["passed"] is True
    assert validator.report(summary) == "All validation checks passed."


def test_validator_detects_normalization_failure() -> None:
    validator = _make_simple_validator()
    state = np.array([2.0, 0.0], dtype=np.complex128)

    passed, summary = validator.validate_state(state)

    assert passed is False
    assert summary["normalized"] is False
    assert "normalization" in validator.report(summary)


def test_validator_detects_coherence_threshold_breach() -> None:
    validator = _make_simple_validator(coherence_threshold=0.9)
    state = np.array([1.0, 0.0], dtype=np.complex128)

    passed, summary = validator.validate_state(state)

    assert passed is False
    assert summary["coherence"]["passed"] is False
    assert "coherence threshold" in validator.report(summary)


def test_validator_frequency_positivity_optional() -> None:
    hilbert = HilbertSpace(dimension=2)
    coherence = CoherenceOperator(np.eye(2))
    validator = NFRValidator(
        hilbert_space=hilbert,
        coherence_operator=coherence,
        coherence_threshold=0.1,
    )
    state = np.array([1.0, 0.0], dtype=np.complex128)

    passed, summary = validator.validate_state(state, enforce_frequency_positivity=False)

    assert passed is True
    assert summary["frequency"] is None


def test_validator_frequency_negativity_flagged() -> None:
    validator = _make_simple_validator()
    negative_frequency = FrequencyOperator([-0.4, 0.1])
    validator.frequency_operator = negative_frequency
    state = np.array([1.0, 0.0], dtype=np.complex128)

    passed, summary = validator.validate_state(state)

    assert passed is False
    assert summary["frequency"]["passed"] is False
    assert "frequency positivity" in validator.report(summary)

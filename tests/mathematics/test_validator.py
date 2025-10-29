"""Tests for the NFR spectral validator."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics.operators import CoherenceOperator, FrequencyOperator
from tnfr.mathematics.spaces import HilbertSpace
from tnfr.mathematics.validators import NFRValidator


@pytest.fixture
def validator() -> NFRValidator:
    hilbert = HilbertSpace(dimension=2)
    coherence = CoherenceOperator([[1.0, 0.0], [0.0, 2.0]])
    frequency = FrequencyOperator([0.5, 1.0])
    return NFRValidator(
        hilbert_space=hilbert,
        coherence_operator=coherence,
        coherence_threshold=0.4,
        frequency_operator=frequency,
    )


def test_validator_accepts_structurally_sound_state(validator: NFRValidator) -> None:
    state = np.array([1.0 + 0.0j, 1.0 + 0.0j]) / np.sqrt(2.0)

    overall, summary = validator.validate_state(state)

    assert overall is True
    assert summary["normalized"] is True
    assert summary["coherence"]["passed"] is True
    assert summary["frequency"]["passed"] is True
    assert summary["unitary_stability"]["passed"] is True
    assert validator.report(summary) == "All validation checks passed."


def test_validator_flags_non_normalised_state(validator: NFRValidator) -> None:
    state = np.array([2.0 + 0.0j, 0.0 + 0.0j])

    overall, summary = validator.validate_state(state)

    assert overall is False
    assert summary["normalized"] is False
    assert summary["coherence"]["passed"] is True


def test_validator_frequency_failure() -> None:
    hilbert = HilbertSpace(dimension=2)
    coherence = CoherenceOperator([[1.0, 0.0], [0.0, 1.0]])
    frequency = FrequencyOperator([-0.1, 0.2])
    validator = NFRValidator(
        hilbert_space=hilbert,
        coherence_operator=coherence,
        coherence_threshold=-0.5,
        frequency_operator=frequency,
    )

    state = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    overall, summary = validator.validate_state(state)

    assert overall is False
    assert summary["frequency"]["passed"] is False
    assert "frequency positivity" in validator.report(summary)


def test_validator_requires_frequency_operator_for_enforcement(validator: NFRValidator) -> None:
    bare_validator = NFRValidator(
        hilbert_space=validator.hilbert_space,
        coherence_operator=validator.coherence_operator,
        coherence_threshold=validator.coherence_threshold,
    )

    state = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    with pytest.raises(ValueError):
        bare_validator.validate_state(state, enforce_frequency_positivity=True)


def test_validator_summary_structure(validator: NFRValidator) -> None:
    state = np.array([1.0 + 0.0j, 1.0 + 0.0j]) / np.sqrt(2.0)

    _, summary = validator.validate_state(state)
    assert set(summary.keys()) == {"normalized", "coherence", "frequency", "unitary_stability"}
    coherence = summary["coherence"]
    assert set(coherence.keys()) == {"passed", "value", "threshold"}

    frequency = summary["frequency"]
    assert set(frequency.keys()) == {"passed", "value", "enforced"}

    unitary = summary["unitary_stability"]
    assert set(unitary.keys()) == {"passed", "norm_after"}

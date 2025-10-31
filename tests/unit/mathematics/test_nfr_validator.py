from __future__ import annotations

import numpy as np

from tnfr.mathematics import CoherenceOperator, FrequencyOperator, HilbertSpace
from tnfr.mathematics.runtime import (
    coherence as runtime_coherence,
    frequency_positive as runtime_frequency_positive,
    normalized as runtime_normalized,
    stable_unitary as runtime_stable_unitary,
)
from tnfr.validation import ValidationOutcome
from tnfr.validation.spectral import NFRValidator


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

    outcome = validator.validate(state)
    assert isinstance(outcome, ValidationOutcome)
    assert outcome.passed is True
    summary = outcome.summary
    assert summary["normalized"] is True
    assert summary["coherence"]["passed"] is True
    assert summary["frequency"]["passed"] is True
    assert summary["unitary_stability"]["passed"] is True
    assert validator.report(outcome) == "All validation checks passed."
    assert validator.validate_state(state) == (True, summary)


def test_validator_detects_normalization_failure() -> None:
    validator = _make_simple_validator()
    state = np.array([2.0, 0.0], dtype=np.complex128)

    outcome = validator.validate(state)
    assert outcome.passed is False
    summary = outcome.summary
    assert summary["normalized"] is False
    assert "normalization" in validator.report(outcome)
    assert validator.validate_state(state) == (False, summary)


def test_validator_detects_coherence_threshold_breach() -> None:
    validator = _make_simple_validator(coherence_threshold=0.9)
    state = np.array([1.0, 0.0], dtype=np.complex128)

    outcome = validator.validate(state)
    assert outcome.passed is False
    summary = outcome.summary
    assert summary["coherence"]["passed"] is False
    assert "coherence threshold" in validator.report(outcome)
    assert validator.validate_state(state) == (False, summary)


def test_validator_frequency_positivity_optional() -> None:
    hilbert = HilbertSpace(dimension=2)
    coherence = CoherenceOperator(np.eye(2))
    validator = NFRValidator(
        hilbert_space=hilbert,
        coherence_operator=coherence,
        coherence_threshold=0.1,
    )
    state = np.array([1.0, 0.0], dtype=np.complex128)

    outcome = validator.validate(state, enforce_frequency_positivity=False)
    assert outcome.passed is True
    assert outcome.summary["frequency"] is None
    assert validator.validate_state(state, enforce_frequency_positivity=False) == (
        True,
        outcome.summary,
    )


def test_validator_frequency_negativity_flagged() -> None:
    validator = _make_simple_validator()
    negative_frequency = FrequencyOperator([-0.4, 0.1])
    validator.frequency_operator = negative_frequency
    state = np.array([1.0, 0.0], dtype=np.complex128)

    outcome = validator.validate(state)
    assert outcome.passed is False
    summary = outcome.summary
    assert summary["frequency"]["passed"] is False
    assert "frequency positivity" in validator.report(outcome)
    assert validator.validate_state(state) == (False, summary)


def test_validator_summary_matches_runtime_helpers() -> None:
    validator = _make_simple_validator()
    state = np.array([1.0 + 0.0j, 1.0j + 0.0j]) / np.sqrt(2.0)

    outcome = validator.validate(state)
    summary = outcome.summary

    normalised_passed, norm_value = runtime_normalized(
        state, validator.hilbert_space, atol=validator.atol
    )
    assert summary["normalized"] is normalised_passed

    normalised_state = state / norm_value

    coherence_passed, coherence_value = runtime_coherence(
        normalised_state,
        validator.coherence_operator,
        validator.coherence_threshold,
        normalise=False,
        atol=validator.atol,
    )
    assert summary["coherence"] == {
        "passed": coherence_passed,
        "value": coherence_value,
        "threshold": validator.coherence_threshold,
    }

    frequency_expected = runtime_frequency_positive(
        normalised_state,
        validator.frequency_operator,
        normalise=False,
        enforce=True,
        atol=validator.atol,
    )
    frequency_expected = {
        **frequency_expected,
        "enforced": frequency_expected["enforce"],
    }
    frequency_expected.pop("enforce", None)
    assert summary["frequency"] == frequency_expected

    unitary_passed, unitary_norm = runtime_stable_unitary(
        normalised_state,
        validator.coherence_operator,
        validator.hilbert_space,
        normalise=False,
        atol=validator.atol,
    )
    assert summary["unitary_stability"] == {
        "passed": unitary_passed,
        "norm_after": unitary_norm,
    }

    assert np.allclose(outcome.artifacts["normalised_state"], normalised_state)

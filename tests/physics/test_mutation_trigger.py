"""Mutation trigger evidence stays distinct from nodal-equation prediction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from tnfr.physics.mutation_trigger import (
    MutationTriggerInputError,
    certify_mutation_trigger,
)


def _certify(**overrides):
    values = {
        "current_epi": 0.2,
        "nu_f": 1.0,
        "delta_nfr": 0.3,
        "xi": 0.1,
    }
    values.update(overrides)
    return certify_mutation_trigger(**values)


def test_prediction_can_cross_while_observed_history_does_not():
    certificate = _certify(
        current_epi=0.4,
        epi_time_history=[(0.0, 0.5), (1.0, 0.4)],
    )

    assert certificate.predicted_depi_dt == pytest.approx(0.3)
    assert certificate.predicted_crossed is True
    assert certificate.observed_depi_dt == pytest.approx(-0.1)
    assert certificate.observed_crossed is False
    assert certificate.threshold_gate_satisfied is False
    assert certificate.rate_gap == pytest.approx(-0.4)


def test_observed_history_can_cross_while_prediction_does_not():
    certificate = _certify(
        delta_nfr=0.01,
        epi_time_history=[(0.0, 0.0), (1.0, 0.2)],
    )

    assert certificate.predicted_depi_dt == pytest.approx(0.01)
    assert certificate.predicted_crossed is False
    assert certificate.observed_depi_dt == pytest.approx(0.2)
    assert certificate.observed_crossed is True
    assert certificate.threshold_gate_satisfied is True
    assert certificate.capacity_active is True


def test_zero_capacity_is_valid_but_reported_inactive_separately_from_threshold():
    certificate = _certify(
        nu_f=0.0,
        epi_time_history=[(0.0, 0.0), (1.0, 0.2)],
    )

    assert certificate.capacity_active is False
    assert certificate.predicted_depi_dt == 0.0
    assert certificate.predicted_crossed is False
    assert certificate.observed_crossed is True
    assert certificate.threshold_gate_satisfied is True


@pytest.mark.parametrize(
    ("history", "expected_rate", "expected_crossed"),
    [
        ([(0.0, 0.0), (0.5, 0.2)], 0.4, True),
        ([(0.0, 0.0), (2.0, 0.2)], 0.1, False),
    ],
)
def test_physical_history_uses_its_actual_interval(
    history, expected_rate, expected_crossed
):
    certificate = _certify(epi_time_history=history)

    assert certificate.evidence_valid is True
    assert certificate.physical_time_resolved is True
    assert certificate.observed_depi_dt == pytest.approx(expected_rate)
    assert certificate.observed_crossed is expected_crossed
    assert certificate.source == "epi_time_history"
    assert certificate.time_basis == "physical_time"


@pytest.mark.parametrize("times", [(1.0, 1.0), (2.0, 1.0)])
def test_non_increasing_physical_time_is_invalid_and_not_a_false_observation(times):
    certificate = _certify(
        epi_time_history=[(times[0], 0.0), (times[1], 0.2)]
    )

    assert certificate.evidence_available is True
    assert certificate.evidence_valid is False
    assert certificate.observed_depi_dt is None
    assert certificate.observed_crossed is None
    assert certificate.threshold_gate_satisfied is False
    assert certificate.reason == "non_increasing_time"


def test_stale_physical_endpoint_is_invalid_but_preserves_past_rate_diagnostic():
    certificate = _certify(
        current_epi=0.3,
        epi_time_history=[(0.0, 0.0), (1.0, 0.2)],
    )

    assert certificate.evidence_available is True
    assert certificate.evidence_valid is False
    assert certificate.observed_depi_dt == pytest.approx(0.2)
    assert certificate.observed_crossed is None
    assert certificate.current_endpoint_matches_state is False
    assert certificate.threshold_gate_satisfied is False
    assert certificate.reason == "stale_physical_endpoint"
    assert certificate.rate_gap is None


def test_endpoint_tolerance_checks_freshness_without_weakening_threshold():
    certificate = _certify(
        current_epi=0.2000001,
        epi_time_history=[(0.0, 0.0), (2.0, 0.2)],
        endpoint_tolerance=1e-3,
    )

    assert certificate.evidence_valid is True
    assert certificate.current_endpoint_matches_state is True
    assert certificate.observed_depi_dt == pytest.approx(0.1)
    assert certificate.observed_crossed is False
    assert certificate.reason == "threshold_not_crossed"


@pytest.mark.parametrize(
    ("kwargs", "expected_source"),
    [
        ({"epi_history": [0.0, 0.2]}, "epi_history"),
        ({"legacy_epi_history": [0.0, 0.2]}, "_epi_history"),
        (
            {"epi_history": [], "legacy_epi_history": [0.0, 0.2]},
            "_epi_history",
        ),
    ],
)
def test_legacy_histories_preserve_unit_step_compatibility(kwargs, expected_source):
    certificate = _certify(current_epi=999.0, **kwargs)

    assert certificate.evidence_valid is True
    assert certificate.observed_depi_dt == pytest.approx(0.2)
    assert certificate.observed_crossed is True
    assert certificate.source == expected_source
    assert certificate.time_basis == "legacy_unit_operator_step"
    assert certificate.physical_time_resolved is False
    assert certificate.current_endpoint_matches_state is None
    assert certificate.rate_gap is None


def test_missing_history_is_unavailable_not_a_negative_observation():
    certificate = _certify()

    assert certificate.evidence is None
    assert certificate.evidence_available is False
    assert certificate.evidence_valid is False
    assert certificate.observed_depi_dt is None
    assert certificate.observed_crossed is None
    assert certificate.threshold_gate_satisfied is False
    assert certificate.reason == "missing_history"
    assert certificate.predicted_crossed is True


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (
            {"epi_time_history": [(0.0, 0.0), (1.0, float("nan"))]},
            "invalid_time_history_sample",
        ),
        (
            {"epi_time_history": [(0.0, 0.0), (True, 0.2)]},
            "invalid_time_history_sample",
        ),
        ({"epi_history": [0.0, "0.2"]}, "invalid_legacy_history_sample"),
        ({"epi_history": [0.0, float("inf")]}, "invalid_legacy_history_sample"),
    ],
)
def test_invalid_history_is_reported_without_mutation(kwargs, reason):
    before = deepcopy(kwargs)
    certificate = _certify(**kwargs)

    assert certificate.evidence_available is True
    assert certificate.evidence_valid is False
    assert certificate.observed_crossed is None
    assert certificate.reason == reason
    assert kwargs == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_epi", True),
        ("current_epi", "0.2"),
        ("nu_f", float("nan")),
        ("nu_f", float("inf")),
        ("nu_f", -0.1),
        ("delta_nfr", object()),
        ("xi", -0.1),
        ("xi", True),
        ("endpoint_tolerance", -1.0),
        ("endpoint_tolerance", float("nan")),
    ],
)
def test_invalid_scalar_inputs_raise_a_field_specific_error(field, value):
    with pytest.raises(MutationTriggerInputError, match=field):
        _certify(**{field: value})


def test_fraction_inputs_are_supported_without_accepting_boolean_subclasses():
    certificate = _certify(
        current_epi=Fraction(1, 5),
        nu_f=Fraction(1, 2),
        delta_nfr=Fraction(3, 5),
        xi=Fraction(1, 10),
        epi_time_history=[(Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(1, 5))],
    )

    assert certificate.predicted_depi_dt == pytest.approx(0.3)
    assert certificate.observed_depi_dt == pytest.approx(0.4)
    assert certificate.predicted_crossed is True
    assert certificate.observed_crossed is True


def test_certificate_and_evidence_are_immutable():
    certificate = _certify(epi_time_history=[(0.0, 0.0), (1.0, 0.2)])

    with pytest.raises(FrozenInstanceError):
        certificate.xi = 2.0  # type: ignore[misc]
    assert certificate.evidence is not None
    with pytest.raises(FrozenInstanceError):
        certificate.evidence.current_epi = 3.0  # type: ignore[misc]

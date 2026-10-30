"""Mutation trigger evidence stays distinct from nodal-equation prediction."""

from __future__ import annotations

import math
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
    certificate = _certify(epi_time_history=[(times[0], 0.0), (times[1], 0.2)])

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


@pytest.mark.parametrize(
    "field", ["current_epi", "nu_f", "delta_nfr", "xi", "endpoint_tolerance"]
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_nonzero_scalar_inputs_cannot_underflow_to_zero(field, sign):
    value = Fraction(sign, 10**400)

    with pytest.raises(MutationTriggerInputError, match="underflows") as caught:
        _certify(**{field: value})

    assert caught.value.field == field
    assert caught.value.value == value


@pytest.mark.parametrize("sample_index", [0, 1])
@pytest.mark.parametrize("coordinate_index", [0, 1], ids=["time", "epi"])
def test_underflowed_physical_samples_do_not_fall_back_to_legacy_history(
    sample_index, coordinate_index
):
    history = [[-1.0, -0.2], [1.0, 0.2]]
    history[sample_index][coordinate_index] = Fraction(1, 10**400)
    before = deepcopy(history)

    certificate = _certify(
        epi_time_history=history,
        epi_history=[0.0, 0.2],
        legacy_epi_history=[0.0, 0.3],
    )

    assert certificate.source == "epi_time_history"
    assert certificate.evidence_available is True
    assert certificate.evidence_valid is False
    assert certificate.observed_depi_dt is None
    assert certificate.observed_crossed is None
    assert certificate.threshold_gate_satisfied is False
    assert certificate.reason == "invalid_time_history_sample"
    assert history == before


@pytest.mark.parametrize("sample_index", [0, 1])
@pytest.mark.parametrize("canonical", [False, True])
def test_underflowed_legacy_samples_are_invalid_in_the_selected_source(
    sample_index, canonical
):
    history = [0.0, 0.2]
    history[sample_index] = Fraction(-1, 10**400)
    before = deepcopy(history)
    kwargs = {"legacy_epi_history": history}
    if canonical:
        kwargs = {"epi_history": history, "legacy_epi_history": [0.0, 0.3]}

    certificate = _certify(**kwargs)

    assert certificate.source == ("epi_history" if canonical else "_epi_history")
    assert certificate.evidence_available is True
    assert certificate.evidence_valid is False
    assert certificate.observed_crossed is None
    assert certificate.threshold_gate_satisfied is False
    assert certificate.reason == "invalid_legacy_history_sample"
    assert history == before


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (True, "must be a finite real scalar"),
        (float("inf"), "must be finite"),
        (10**400, "must be representable as a finite real scalar"),
    ],
)
def test_shared_admission_preserves_scalar_error_reasons(value, reason):
    with pytest.raises(MutationTriggerInputError) as caught:
        _certify(current_epi=value)

    assert caught.value.field == "current_epi"
    assert caught.value.reason == reason


def test_signed_samples_and_exact_zero_inputs_remain_valid():
    certificate = _certify(
        current_epi=Fraction(-1, 4),
        delta_nfr=Fraction(-1, 2),
        xi=Fraction(0),
        endpoint_tolerance=Fraction(0),
        epi_time_history=[
            (Fraction(-1), Fraction(-1, 2)),
            (Fraction(0), Fraction(-1, 4)),
        ],
    )

    assert certificate.evidence_valid is True
    assert certificate.predicted_depi_dt == -0.5
    assert certificate.observed_depi_dt == 0.25
    assert certificate.predicted_crossed is False
    assert certificate.observed_crossed is True


def test_representable_subnormal_inputs_are_not_rejected_as_underflow():
    smallest = float.fromhex("0x0.0000000000001p-1022")
    certificate = _certify(
        current_epi=smallest,
        delta_nfr=smallest,
        xi=0,
        epi_time_history=[(0, 0), (1, smallest)],
    )

    assert certificate.evidence_valid is True
    assert certificate.predicted_depi_dt == smallest
    assert certificate.observed_depi_dt == smallest
    assert certificate.predicted_crossed is True
    assert certificate.observed_crossed is True


@pytest.mark.parametrize("sign", [-1, 1])
def test_nonzero_predicted_product_cannot_underflow_to_zero(sign):
    smallest = float.fromhex("0x0.0000000000001p-1022")

    with pytest.raises(MutationTriggerInputError, match="underflows") as caught:
        _certify(nu_f=0.5, delta_nfr=sign * smallest, xi=0)

    assert caught.value.field == "predicted_depi_dt"


def test_nonfinite_predicted_product_preserves_its_error_reason():
    with pytest.raises(MutationTriggerInputError) as caught:
        _certify(nu_f=1e308, delta_nfr=2.0)

    assert caught.value.field == "predicted_depi_dt"
    assert caught.value.reason == (
        "must be finite for the represented nu_f * delta_nfr product"
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_nonzero_physical_secant_cannot_underflow_to_a_false_zero(sign):
    smallest = float.fromhex("0x0.0000000000001p-1022")
    certificate = _certify(
        current_epi=sign * smallest,
        epi_time_history=[(0, 0), (2, sign * smallest)],
        epi_history=[0.0, 0.2],
        xi=0,
    )

    assert certificate.source == "epi_time_history"
    assert certificate.evidence_available is True
    assert certificate.evidence_valid is False
    assert certificate.physical_time_resolved is True
    assert certificate.observed_depi_dt is None
    assert certificate.observed_crossed is None
    assert certificate.threshold_gate_satisfied is False
    assert certificate.reason == "underflowed_observed_rate"
    assert certificate.evidence is not None
    assert certificate.evidence.previous_epi == 0
    assert certificate.evidence.current_epi == sign * smallest
    assert certificate.evidence.sample_interval == 2


@pytest.mark.parametrize(
    "history",
    [
        [(0, -1e308), (1, 1e308)],
        [(0, 0), (float.fromhex("0x0.0000000000001p-1022"), 1)],
    ],
    ids=["difference_overflow", "quotient_overflow"],
)
def test_nonfinite_physical_secant_preserves_its_invalid_evidence_reason(history):
    certificate = _certify(current_epi=history[-1][1], epi_time_history=history)

    assert certificate.evidence_valid is False
    assert certificate.observed_depi_dt is None
    assert certificate.observed_crossed is None
    assert certificate.reason == "non_finite_observed_rate"


@pytest.mark.parametrize(("capacity", "pressure"), [(0, -1), (1, 0), (0, 0)])
def test_exact_zero_products_and_secants_remain_available(capacity, pressure):
    certificate = _certify(
        current_epi=0,
        nu_f=capacity,
        delta_nfr=pressure,
        xi=0,
        epi_time_history=[(0, 0), (1, 0)],
    )

    assert certificate.evidence_valid is True
    assert certificate.predicted_depi_dt == 0
    assert certificate.observed_depi_dt == 0
    assert certificate.observed_crossed is False
    assert certificate.predicted_crossed is False


@pytest.mark.parametrize("field", ["current_epi", "nu_f", "delta_nfr", "xi"])
def test_admitted_negative_zero_scalar_channels_preserve_their_sign(field):
    certificate = _certify(**{field: -0.0})

    assert math.copysign(1.0, getattr(certificate, field)) == -1.0
    if field == "delta_nfr":
        assert math.copysign(1.0, certificate.predicted_depi_dt) == -1.0


def test_admitted_negative_zero_history_coordinates_preserve_their_sign():
    certificate = _certify(
        current_epi=-0.0,
        epi_time_history=[(-0.0, -0.0), (1.0, -0.0)],
    )

    assert certificate.evidence_valid is True
    assert certificate.evidence is not None
    assert math.copysign(1.0, certificate.evidence.previous_time) == -1.0
    assert math.copysign(1.0, certificate.evidence.previous_epi) == -1.0
    assert math.copysign(1.0, certificate.evidence.current_epi) == -1.0


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

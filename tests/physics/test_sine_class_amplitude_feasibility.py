"""Response-free scale intervals, complete error budgets and carried work."""

import json
import subprocess
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_amplitude_feasibility as owner
from tnfr.physics import relational_sine_class_superposition as events
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    # Supplied, outward displayed coefficient premise; this fixture does not
    # validate its scientific provenance. A separate retained-evidence test does.
    return dict(
        amplitude_scale_lower=Q(4, 3),
        amplitude_scale_upper=Q(7, 5),
        base_cubic_lower=Q(-7013764940694, 10**42),
        base_cubic_upper=Q(-7013764940692, 10**42),
    )


@pytest.fixture(scope="module", autouse=True)
def no_coefficient_response_or_worker():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_cubic_response,
        relational_sine_class_readout,
        relational_sine_class_spatial_observation,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("amplitude arithmetic must not compute a response or coefficient")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                relational_sine_class_cubic_response,
                (
                    "bound_sine_class_cubic_response",
                    "_class_cubic_coefficients",
                    "_time_coefficients",
                ),
            ),
            (
                relational_sine_class_spatial_observation,
                ("bound_sine_class_spatial_observation", "_class_cubic_coefficients"),
            ),
            (_sine_flow, ("_full_sine_field",)),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (
                _validated_taylor,
                ("flow_jets", "picard_tube", "validated_box_taylor_step"),
            ),
            (relational_sine_class_readout, ("bound_sine_class_four_history_readout",)),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def report():
    return owner.bound_sine_class_amplitude_feasibility(**_arguments())


def test_four_scalar_primitives_and_explicit_conditional_coefficient():
    parameters = signature(owner.bound_sine_class_amplitude_feasibility).parameters
    assert set(parameters) == set(_arguments())
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for key in ("report", "source_state", "response", "sensor_error", "horizon"):
        with pytest.raises(TypeError):
            owner.bound_sine_class_amplitude_feasibility(**_arguments(), **{key: 0})


@pytest.mark.parametrize("key", tuple(_arguments()))
@pytest.mark.parametrize(
    "bad", (True, np.bool_(False), float("nan"), float("inf"), Decimal("1e-400"))
)
def test_invalid_scalar_admission_precedes_model_arithmetic(key, bad, monkeypatch):
    monkeypatch.setattr(owner, "_parameters", lambda *_: pytest.fail("admission order"))
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_amplitude_feasibility(**{**_arguments(), key: bad})


@pytest.mark.parametrize(
    "change",
    (
        {"amplitude_scale_lower": 0},
        {"amplitude_scale_lower": -1},
        {"amplitude_scale_upper": Q(4, 3) - Q(1, 2**400)},
        {"base_cubic_lower": 1, "base_cubic_upper": 0},
    ),
)
def test_order_and_positive_scale_domains(change):
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_amplitude_feasibility(**{**_arguments(), **change})


def test_declared_interval_has_all_uniform_sufficient_margins(report):
    assert report.status == "feasible_interval" and report.feasible_interval_certified
    assert report.response_bound_available and report.cauchy_admitted
    assert report.decision.null_excluded and report.decision.recorded_sign
    assert not report.decision.scalar_cancellation
    assert report.all_identities_certified and report.all_work_within_allowances
    assert report.unmet_sufficient_requirements == ()
    assert report.storage_barrier == Q(1, 388800)
    assert Q(-19334, 10**33) < report.decision.true_bounds[0]
    assert report.decision.true_bounds[1] < Q(-16537, 10**33)
    assert max(
        h.first_probe_work_upper_bound for h in report.upper_scale_history_bounds
    ) < Q(736, 10**9)
    assert max(
        h.second_probe_work_upper_bound for h in report.upper_scale_history_bounds
    ) < Q(1287, 10**9)
    assert max(
        h.after_second_excess_storage_upper_bound
        for h in report.upper_scale_history_bounds
    ) < Q(2022, 10**9)
    assert max(
        h.after_second_radius_squared_upper_bound
        for h in report.upper_scale_history_bounds
    ) < Q(5293, 10**8)


def test_homogeneity_keeps_the_base_interval_and_uses_uniform_upper_tail(report):
    lo, hi = report.amplitude_scale_lower, report.amplitude_scale_upper
    products = tuple(
        scale**3 * c
        for scale in (lo, hi)
        for c in (report.base_cubic_lower, report.base_cubic_upper)
    )
    assert report.scaled_cubic_contrast_bounds == (min(products), max(products))
    g, T, pulse = Q(1, 3000), Q(2), hi / 2000
    tail = tuple(
        256 * g**6 * A**5 * T / ((1 - 2 * g * g * T * T) * (1 - 4 * g * g * A * A))
        for A in (Q(0), pulse, pulse, 2 * pulse)
    )
    assert report.per_history_higher_amplitude_remainder_upper_bounds == tail
    assert report.higher_amplitude_contrast_error_upper_bound == 2 * sum(tail)
    source = 8 * report.endpoint_radius / (1 - 2 * report.gamma_bounds.hi * T)
    assert report.source_contrast_error_upper_bound == source
    error = 2 * sum(tail) + source
    assert report.decision.true_bounds == (min(products) - error, max(products) + error)
    assert (
        report.decision.null_separation_margin
        == -max(products) - error - 16 * report.readout_error_bound
    )


def test_delayed_pressure_keeps_heat_and_actual_source_defects(report):
    eps, g = report.endpoint_radius, Q(1, 3000)
    for history, bound in zip(
        report.upper_scale_history_bounds, report.upper_scale_delayed_laplacian_bounds
    ):
        p, q = history.first_probe_amplitude, history.second_probe_amplitude
        form = (p + eps + 2 * g * eps) / (1 - 2 * g * g)
        defect = eps + 2 * g * eps + 2 * g * g * form
        expected = (-6 * defect, Q(9, 8) * p + 6 * defect)
        assert bound == expected
        assert (
            history.second_probe_work_upper_bound == Q(3, 2) * q * q + q * expected[1]
        )
        assert (
            history.after_second_excess_storage_upper_bound
            == 22 * eps**2
            + history.first_probe_work_upper_bound
            + history.second_probe_work_upper_bound
        )
    assert report.upper_scale_delayed_laplacian_bounds[0][1] > 0


def test_uniform_means_are_not_the_upper_endpoint_means(report):
    eps, lo, hi = (
        report.endpoint_radius,
        report.amplitude_scale_lower,
        report.amplitude_scale_upper,
    )
    error = Q(4, 58) * eps
    assert report.phase_mean_bounds == (-error, error)
    for index, (first_count, total_count) in enumerate(
        ((0, 0), (1, 1), (0, 1), (1, 2))
    ):
        for count, bounds in (
            (first_count, report.first_form_mean_bounds[index]),
            (total_count, report.final_form_mean_bounds[index]),
        ):
            assert bounds == (
                Q(3, 58) * count * lo / 2000 - error,
                Q(3, 58) * count * hi / 2000 + error,
            )
    assert (
        report.final_form_mean_bounds[-1][0]
        < report.upper_scale_history_bounds[-1].final_form_mean_bounds.lo
    )


def test_lower_work_certificate_improves_a_bound_without_using_loss_credit(report):
    upper = report.amplitude_scale_upper / 2000
    values = dict(
        first_probe_amplitude=upper,
        second_probe_amplitude=upper,
        delay=1,
        total_duration=2,
        endpoint_radius=report.endpoint_radius,
        radius=report.radius,
        contact_work_allowance=report.contact_work_allowance,
        first_probe_work_allowance=report.first_probe_work_allowance,
        second_probe_work_allowance=report.second_probe_work_allowance,
    )
    old = events._probe_event_ledger(values, Q(1, 3000), (None,) * 4)
    assert (
        old.histories[-1].second_probe_work_upper_bound
        > report.second_probe_work_allowance
    )
    assert not old.histories[-1].identity_certified
    assert (
        report.upper_scale_history_bounds[-1].second_probe_work_upper_bound
        < report.second_probe_work_allowance
    )
    assert report.initial_excess_storage_upper_bound == old.initial_storage
    assert report.storage_barrier == old.barrier


@pytest.mark.parametrize(
    "scale,missing",
    (
        (Q(1), "separately_noisy_null_not_uniformly_excluded"),
        (Q(2), "uniform_work_allowances_not_certified"),
    ),
)
def test_failed_sufficient_guards_do_not_claim_actual_infeasibility(scale, missing):
    result = owner.bound_sine_class_amplitude_feasibility(
        **{
            **_arguments(),
            "amplitude_scale_lower": scale,
            "amplitude_scale_upper": scale,
        }
    )
    assert not result.feasible_interval_certified
    assert (
        result.response_bound_available
        and result.status == "requirements_not_certified"
    )
    assert missing in result.unmet_sufficient_requirements


def test_zero_supplied_cubic_does_not_invent_an_exact_full_response():
    result = owner.bound_sine_class_amplitude_feasibility(
        **{**_arguments(), "base_cubic_lower": 0, "base_cubic_upper": 0}
    )
    assert result.scaled_cubic_contrast_bounds == (Q(0), Q(0))
    assert result.higher_amplitude_contrast_error_upper_bound > 0
    assert result.decision.true_bounds[0] < 0 < result.decision.true_bounds[1]
    assert not result.feasible_interval_certified


@pytest.mark.parametrize("upper", (Q(1500000), Q(1500000) + Q(1, 2**400), Q(10**200)))
def test_unsupported_complex_domain_never_evaluates_negative_denominator(
    upper, monkeypatch
):
    monkeypatch.setattr(
        owner,
        "_higher_amplitude_remainder",
        lambda *_: pytest.fail("invalid Cauchy domain"),
    )
    result = owner.bound_sine_class_amplitude_feasibility(
        **{**_arguments(), "amplitude_scale_upper": upper}
    )
    assert result.cauchy_radius_margin <= 0 and not result.cauchy_admitted
    assert result.status == "unavailable" and result.decision is None
    assert result.higher_amplitude_contrast_error_upper_bound is None
    assert not result.feasible_interval_certified


def test_signed_and_tiny_exact_coefficient_intervals_are_not_coerced():
    tiny = Q(1, 10**400)
    result = owner.bound_sine_class_amplitude_feasibility(
        amplitude_scale_lower=0.5,
        amplitude_scale_upper=0.5,
        base_cubic_lower=-tiny,
        base_cubic_upper=tiny,
    )
    assert result.amplitude_scale_lower == Q(1, 2)
    assert result.scaled_cubic_contrast_bounds == (-tiny / 8, tiny / 8)
    assert result.base_cubic_lower == -tiny


@pytest.mark.parametrize(
    "bad",
    (
        (0,) * 3,
        (0,) * 5,
        (True, 0, 0, 0),
        (Q(-1), 0, 0, 0),
        (0.0, 0, 0, 0),
        (float("nan"), 0, 0, 0),
    ),
)
def test_optional_delayed_bounds_require_four_nonnegative_exact_values_first(bad):
    # Empty values would fail later; malformed authoritative bounds fail first.
    with pytest.raises((ValueError, TypeError)):
        events._probe_event_ledger(
            {}, Q(1, 3000), (None,) * 4, pre_second_laplacian_abs_bounds=bad
        )


@pytest.mark.parametrize(
    "a,b,s,t",
    (
        (Q(1, 500), Q(-1, 700), Q(1, 7), Q(3, 2)),
        (Q(0), Q(1, 500), Q(0), Q(2)),
        (Q(1, 2000), Q(1, 2000), Q(1), Q(2)),
    ),
)
def test_default_event_ledger_is_exactly_preserved(a, b, s, t):
    eps, g = Q(1, 10**32), Q(1, 3000)
    values = dict(
        first_probe_amplitude=a,
        second_probe_amplitude=b,
        delay=s,
        total_duration=t,
        endpoint_radius=eps,
        radius=Q(1, 12),
        contact_work_allowance=Q(1, 10**12),
        first_probe_work_allowance=Q(1, 500000),
        second_probe_work_allowance=Q(1, 500000),
    )
    old = events._probe_event_ledger(values, g, (None,) * 4)
    legacy_bounds = tuple(
        6 * (abs(first) + eps + 2 * g * s * eps) / (1 - 2 * g * g * s * s)
        for first in (Q(0), a, Q(0), a)
    )
    supplied = events._probe_event_ledger(
        values, g, (None,) * 4, pre_second_laplacian_abs_bounds=legacy_bounds
    )
    assert old == supplied
    for history in old.histories:
        p, q = history.first_probe_amplitude, history.second_probe_amplitude
        pre = (abs(p) + eps + 2 * g * s * eps) / (1 - 2 * g * g * s * s)
        assert history.second_probe_work_bounds == I(
            Q(3, 2) * q * q - 6 * abs(q) * pre, Q(3, 2) * q * q + 6 * abs(q) * pre
        )


@pytest.mark.parametrize(
    "first,eps,g",
    (
        (Q(7, 10000), Q(1, 10**32), Q(1, 3000)),
        (Q(3, 7), Q(2, 13), Q(1, 11)),
        (Q(0), Q(1, 2**500), Q(1, 3000)),
        (Q(1, 2**500), Q(0), Q(0)),
        (Q(0), Q(0), Q(7, 10)),
    ),
)
def test_shared_unit_delay_pressure_matches_independent_heat_defect(first, eps, g):
    form = (first + eps + 2 * g * eps) / (1 - 2 * g**2)
    defect = eps + 2 * g * eps + 2 * g**2 * form
    expected = -6 * defect, Q(9, 8) * first + 6 * defect
    actual = events._unit_delay_donor_laplacian_bounds(
        first_amplitude=first, endpoint_radius=eps, gamma_upper=g
    )
    assert actual == expected
    assert all(type(value) is Q for value in actual)
    assert actual[0] <= 0 <= actual[1]
    assert actual[1] + actual[0] == Q(9, 8) * first


@pytest.mark.parametrize("key", ("first_amplitude", "endpoint_radius", "gamma_upper"))
@pytest.mark.parametrize("bad", (True, np.bool_(False), 0.0, float("inf"), Q(-1)))
def test_shared_unit_delay_pressure_admits_exact_domains_before_envelope(
    key, bad, monkeypatch
):
    monkeypatch.setattr(
        events, "_probe_coordinate_envelope", lambda *_: pytest.fail("late admission")
    )
    values = dict(first_amplitude=Q(1), endpoint_radius=Q(0), gamma_upper=Q(1, 3000))
    with pytest.raises((TypeError, ValueError)):
        events._unit_delay_donor_laplacian_bounds(**(values | {key: bad}))


@pytest.mark.parametrize("g", (Q(3, 4), Q(1), Q(10**100)))
def test_shared_unit_delay_pressure_rejects_missing_bootstrap_before_envelope(
    g, monkeypatch
):
    monkeypatch.setattr(
        events, "_probe_coordinate_envelope", lambda *_: pytest.fail("late admission")
    )
    with pytest.raises(ValueError, match="requires"):
        events._unit_delay_donor_laplacian_bounds(
            first_amplitude=Q(1), endpoint_radius=Q(0), gamma_upper=g
        )


def test_amplitude_adapter_retains_the_original_pressure_and_consumes_shared_owner(
    monkeypatch,
):
    received = []
    shared = events._unit_delay_donor_laplacian_bounds

    def observe(**values):
        received.append(values)
        return shared(**values)

    monkeypatch.setattr(owner, "_unit_delay_donor_laplacian_bounds", observe)
    for amplitude in (Q(0), Q(3, 8000)):
        eps, g = Q(1, 10**32), Q(1, 3000)
        form = (amplitude + eps + 2 * g * eps) / (1 - 2 * g**2)
        defect = eps + 2 * g * eps + 2 * g**2 * form
        assert owner._delayed_laplacian_bounds(amplitude) == (
            -6 * defect,
            Q(9, 8) * amplitude + 6 * defect,
        )
    assert received == [
        dict(first_amplitude=a, endpoint_radius=Q(1, 10**32), gamma_upper=Q(1, 3000))
        for a in (Q(0), Q(3, 8000))
    ]


def test_exact_sdk_projection_and_unavailable_fields(report):
    encoded = relational_report_to_dict(report)
    assert encoded["report_type"] == "SineClassAmplitudeFeasibility"
    assert encoded["report"] == report.to_dict()["report"]
    assert report.to_dict()["schema"] == "tnfr.sine-class-amplitude-feasibility.v1"
    decoded = json_loads(json.dumps(encoded, allow_nan=False))
    assert decoded["report"]["amplitude_scale_lower"] == {
        "numerator": 4,
        "denominator": 3,
    }
    unavailable = owner.bound_sine_class_amplitude_feasibility(
        **{**_arguments(), "amplitude_scale_upper": 1500000}
    )
    assert unavailable.to_dict()["report"]["decision"] is None

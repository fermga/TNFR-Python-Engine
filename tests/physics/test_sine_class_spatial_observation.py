"""Admission, synthetic projection and fixed analytic observer controls."""

import json
import subprocess
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_cubic_response as cubic
from tnfr.physics import relational_sine_class_spatial_observation as owner
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    return dict(
        first_probe_amplitude=Q(1, 2000),
        second_probe_amplitude=Q(1, 2000),
        delay=Q(1),
        total_duration=Q(2),
        endpoint_radius=Q(1, 10**32),
        readout_error_bound=Q(1, 10**30),
        radius=Q(1, 12),
        contact_work_allowance=Q(1, 10**12),
        first_probe_work_allowance=Q(1, 500000),
        second_probe_work_allowance=Q(1, 500000),
    )


@pytest.fixture(scope="module", autouse=True)
def no_nonlinear_response_or_acquisition():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("observer controls must not acquire a source or nonlinear flow")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _validated_taylor,
                ("flow_jets", "picard_tube", "validated_box_taylor_step"),
            ),
            (_sine_flow, ("_full_sine_field",)),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (relational_sine_class_readout, ("bound_sine_class_four_history_readout",)),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(autouse=True)
def no_accidental_finite_coefficient(request, monkeypatch):
    if "finite_report" in request.fixturenames:
        return

    def forbidden(*args, **kwargs):
        pytest.fail("synthetic/admission controls must not evaluate a coefficient")

    for module, names in (
        (
            cubic,
            ("_class_cubic_coefficients", "_coefficient_segment", "_time_coefficients"),
        ),
        (owner, ("_class_cubic_coefficients",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.fixture(scope="module")
def finite_report():
    # This regression follows the retained first assessment. Shared immutable
    # class records also serve the other analytic suites in a combined run.
    return owner.bound_sine_class_spatial_observation(**_arguments())


@pytest.fixture
def synthetic_coefficients(monkeypatch):
    """Deliberately artificial level records test projection, not model truth."""

    def segment(label, left, right):
        rows = [[I(0)] * 54 for _ in range(3)]
        rows[1][23], rows[1][21] = I(left), I(right)
        # Unrelated coordinates and amplitude levels must not enter the readout.
        rows[1][22], rows[2][22] = I(10**40), I(-(10**40))
        levels = tuple(map(tuple, rows))
        return cubic._CubicSegment(
            label,
            None,
            Q(1),
            Q(2),
            Q(0),
            cubic._ZERO_LEVELS,
            (),
            (Q(0),) * 3,
            (Q(0),) * 3,
            levels,
        )

    parameters = {k: cubic._cubic_parameters(k) for k in (1, 2)}
    histories = {
        k: (
            segment("prefix", 0, 0),
            segment("first", 3, 1),
            segment("both", 11 if k == 1 else 7, 2),
            segment("second", 5, 2),
        )
        for k in (1, 2)
    }
    calls = []

    def coefficients(k, a, b, s, t):
        calls.append((k, a, b, s, t))
        # A bogus old central statistic is retained to catch accidental reuse.
        return parameters[k], histories[k], I(10**70)

    monkeypatch.setattr(owner, "_class_cubic_coefficients", coefficients)
    return calls, parameters, histories


def test_api_has_only_ten_admitted_primitives():
    parameters = signature(owner.bound_sine_class_spatial_observation).parameters
    assert set(parameters) == set(_arguments())
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for key in ("report", "reading_count", "observation", "order", "receiver_state"):
        with pytest.raises(TypeError):
            owner.bound_sine_class_spatial_observation(**_arguments(), **{key: 0})


@pytest.mark.parametrize("key", tuple(_arguments()))
@pytest.mark.parametrize("bad", (True, np.bool_(False), float("nan"), float("inf")))
def test_invalid_primitives_precede_parameters_and_cache(key, bad, monkeypatch):
    monkeypatch.setattr(owner, "_parameters", lambda *_: pytest.fail("admission order"))
    with pytest.raises((ValueError, TypeError)):
        owner.bound_sine_class_spatial_observation(**{**_arguments(), key: bad})


@pytest.mark.parametrize(
    "key,value",
    (
        ("delay", -1),
        ("delay", Q(2) + Q(1, 2**400)),
        ("total_duration", Q(2) + Q(1, 2**400)),
        ("endpoint_radius", -1),
        ("readout_error_bound", -1),
        ("radius", 0),
        ("radius", Q(1, 12) + Q(1, 2**400)),
        ("contact_work_allowance", -1),
        ("first_probe_work_allowance", -1),
        ("second_probe_work_allowance", -1),
        ("first_probe_amplitude", Decimal("1e-400")),
    ),
)
def test_domain_and_materialization_boundaries(key, value):
    with pytest.raises((ValueError, TypeError)):
        owner.bound_sine_class_spatial_observation(**{**_arguments(), key: value})


def test_two_node_second_level_projection_cancels_classes_before_gamma(
    synthetic_coefficients,
):
    calls, parameters, histories = synthetic_coefficients
    result = owner.bound_sine_class_spatial_observation(**_arguments())
    assert tuple(call[0] for call in calls) == (1, 2)
    assert result.class_segments == (histories[1], histories[2])
    assert result.class_parameters == (parameters[1], parameters[2])
    assert result.observation_node_weights == ((23, 1), (21, -1))
    assert result.reading_count == 16
    assert (
        result.coefficient_amplitude_degree == 2
        and result.coefficient_scale_gamma_power == 3
    )
    gamma = result.gamma_bounds
    expected = (4 * gamma.lo**3, 4 * gamma.hi**3)
    assert result.complete_quadratic_contrast_bounds == expected
    assert result.scaled_class_quadratic_bounds == (expected, (Q(0), Q(0)))
    assert result.coefficient_evaluated and result.response_bound_available


def test_cross_class_remainder_is_not_doubled_and_source_is_not_cancelled(
    synthetic_coefficients,
):
    result = owner.bound_sine_class_spatial_observation(**_arguments())
    a, b, t, g = Q(1, 2000), Q(1, 2000), Q(2), Q(1, 3000)
    expected = tuple(
        Q(4096, 3) * g**7 * A**4 * t**3 / (1 - 4 * g * g * A * A)
        for A in (Q(0), abs(a), abs(b), abs(a) + abs(b))
    )
    assert (
        result.per_history_higher_amplitude_contrast_remainder_upper_bounds == expected
    )
    assert result.higher_amplitude_contrast_error_upper_bound == sum(expected)
    source = 16 * result.endpoint_radius / (1 - 2 * result.gamma_bounds.hi * t)
    assert result.source_contrast_error_upper_bound == source
    lo, hi = result.complete_quadratic_contrast_bounds
    error = sum(expected) + source
    assert result.decision.true_bounds == (lo - error, hi + error)
    noise = 16 * result.readout_error_bound
    assert result.decision.recorded_bounds == (lo - error - noise, hi + error + noise)
    assert result.decision.noise_ceiling == (lo - error) / 16


@pytest.mark.parametrize(
    "multiple,field", ((16, "recorded_sign"), (32, "null_excluded"))
)
def test_strict_spatial_noise_thresholds(synthetic_coefficients, multiple, field):
    baseline = owner.bound_sine_class_spatial_observation(**_arguments())
    limit = baseline.decision.oriented_lower / multiple
    equal = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "readout_error_bound": limit}
    )
    assert not getattr(equal.decision, field)
    below = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "readout_error_bound": limit - Q(1, 10**100)}
    )
    assert getattr(below.decision, field)


def test_work_and_identity_policies_do_not_replace_observation(synthetic_coefficients):
    result = owner.bound_sine_class_spatial_observation(**_arguments())
    changed = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "radius": Q(1, 10**8), "first_probe_work_allowance": 0}
    )
    assert result.all_identities_certified and result.all_work_within_allowances
    assert (
        not changed.all_identities_certified and not changed.all_work_within_allowances
    )
    assert changed.decision == result.decision
    assert all(
        h.tangent_endpoint_discrepancy_upper_bound is None
        for h in changed.uniform_history_bounds
    )


@pytest.mark.parametrize(
    "change",
    (
        {"first_probe_amplitude": 0},
        {"second_probe_amplitude": 0},
        {"delay": 2},
        {"delay": 0, "total_duration": 0},
        {"first_probe_amplitude": 0, "second_probe_amplitude": 2000},
    ),
)
def test_exact_pairing_zero_needs_no_coefficients_or_cauchy_tail(change):
    result = owner.bound_sine_class_spatial_observation(**{**_arguments(), **change})
    assert result.exact_contrast_zero and result.response_bound_available
    assert not result.coefficient_evaluated
    assert result.complete_quadratic_contrast_bounds == (Q(0), Q(0))
    assert result.higher_amplitude_contrast_error_upper_bound is None
    assert result.decision.true_bounds == (Q(0), Q(0))
    assert result.decision.recorded_bounds == (
        -16 * result.readout_error_bound,
        16 * result.readout_error_bound,
    )


@pytest.mark.parametrize("amplitude", (Q(1499), Q(1499) + Q(1, 2**400), Q(10**300)))
def test_nonzero_unsupported_cauchy_domain_withholds_full_bounds(amplitude):
    result = owner.bound_sine_class_spatial_observation(
        **{
            **_arguments(),
            "first_probe_amplitude": amplitude,
            "second_probe_amplitude": 1,
        }
    )
    assert not result.cauchy_admitted and result.cauchy_radius_margin <= 0
    assert not result.response_bound_available and result.status == "unavailable"
    assert result.complete_quadratic_contrast_bounds is None and result.decision is None
    assert result.higher_amplitude_contrast_error_upper_bound is None
    assert result.class_segments == ()


def test_exact_small_signed_values_survive_admission():
    tiny = Q(1, 10**400)
    result = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "first_probe_amplitude": -tiny, "second_probe_amplitude": 0}
    )
    assert result.first_probe_amplitude == -tiny
    assert result.exact_contrast_zero


def test_real_unavailable_report_projection_has_exact_and_null_fields():
    result = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "first_probe_amplitude": 1499, "second_probe_amplitude": 1}
    )
    direct = result.to_dict()
    assert direct["schema"] == "tnfr.sine-class-spatial-observation.v1"
    assert direct["report"]["complete_quadratic_contrast_bounds"] is None
    assert direct["report"]["cauchy_radius_margin"] == {
        "numerator": 0,
        "denominator": 1,
    }
    assert json_loads(json.dumps(direct, allow_nan=False)) == direct
    generic = relational_report_to_dict(result)
    assert generic["report"] == direct["report"]
    assert generic["report_type"] == "SineClassSpatialObservation"


def test_shared_projection_preserves_the_original_central_operation(
    synthetic_coefficients,
):
    _, _, histories = synthetic_coefficients
    for history in histories.values():
        _, first, both, second = history
        old = (
            both.endpoint_levels[2][22]
            - first.endpoint_levels[2][22]
            - second.endpoint_levels[2][22]
        )
        assert cubic._mixed_form_projection(history, 2, ((22, 1),)) == old
    gamma = I(Q(1, 4000), Q(1, 3000))
    value = I(Q(-3, 7), Q(2, 9))
    expected = tuple(
        g**4 * v for g in (gamma.lo, gamma.hi) for v in (value.lo, value.hi)
    )
    assert cubic._scale_by_shared_gamma(value, gamma, 4) == (
        min(expected),
        max(expected),
    )


def test_fixed_observer_has_true_sign_but_not_recorded_sign(finite_report):
    result = finite_report
    lo, hi = result.complete_quadratic_contrast_bounds
    assert Q(7951, 10**33) < lo <= hi < Q(7952, 10**33)
    assert hi - lo < Q(1, 10**40)
    assert result.higher_amplitude_contrast_error_upper_bound < Q(5619, 10**36)
    assert result.source_contrast_error_upper_bound < Q(1603, 10**34)
    assert Q(7785, 10**33) < result.decision.true_bounds[0]
    assert result.decision.true_bounds[1] < Q(8117, 10**33)
    assert result.status == "true_sign_certified"
    assert result.decision.orientation == 1 and result.decision.true_sign
    assert not result.decision.recorded_sign and not result.decision.null_excluded
    assert result.decision.recorded_bounds[0] < 0 < result.decision.recorded_bounds[1]
    assert result.decision.scalar_cancellation
    assert result.all_identities_certified and result.all_work_within_allowances


def test_real_level_two_history_projection_and_source_sensitivity(finite_report):
    result = finite_report
    mixed = []
    for _, first, both, second in result.class_segments:

        def observation(segment):
            row = segment.endpoint_levels[1]
            return row[23] - row[21]

        mixed.append(observation(both) - observation(first) - observation(second))
    difference = mixed[0] - mixed[1]
    products = [
        g**3 * v
        for g in (result.gamma_bounds.lo, result.gamma_bounds.hi)
        for v in (difference.lo, difference.hi)
    ]
    assert result.complete_quadratic_contrast_bounds == (min(products), max(products))
    changed = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "endpoint_radius": Q(1, 10**20)}
    )
    assert changed.class_segments[0] is result.class_segments[0]
    assert (
        changed.complete_quadratic_contrast_bounds
        == result.complete_quadratic_contrast_bounds
    )
    assert changed.decision.true_bounds[0] < 0 < changed.decision.true_bounds[1]
    assert not changed.decision.true_sign and changed.status == "bounds_only"


@pytest.mark.parametrize(
    "multiple,field", ((16, "recorded_sign"), (32, "null_excluded"))
)
def test_real_exact_noise_boundary(finite_report, multiple, field):
    bound = finite_report.decision.oriented_lower / multiple
    equal = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "readout_error_bound": bound}
    )
    assert not getattr(equal.decision, field)
    below = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "readout_error_bound": bound - Q(1, 10**100)}
    )
    assert getattr(below.decision, field)


def test_real_source_and_sensor_errors_remain_distinct(finite_report):
    result = owner.bound_sine_class_spatial_observation(
        **{**_arguments(), "endpoint_radius": 0, "readout_error_bound": 0}
    )
    assert result.source_contrast_error_upper_bound == 0
    assert (
        result.higher_amplitude_contrast_error_upper_bound
        == finite_report.higher_amplitude_contrast_error_upper_bound
        > 0
    )
    assert result.decision.true_bounds == result.decision.recorded_bounds
    assert (
        result.decision.true_sign
        and result.decision.recorded_sign
        and result.decision.null_excluded
    )
    assert not result.decision.scalar_cancellation

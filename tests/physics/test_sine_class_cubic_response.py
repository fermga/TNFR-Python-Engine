"""Complete cubic coefficients, conditional bounds and honest observation policies."""

import json
import subprocess
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_cubic_response as owner
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
        relational_sine_formed_classes,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("analytic coefficient controls must not run acquisition or a flow")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _sine_formed_contact,
                ("_unprobed_handoff", "assess_sine_formed_class_pair"),
            ),
            (relational_sine_formed_classes, ("assess_sine_formed_class_pair",)),
            (_sine_flow, ("_full_sine_field",)),
            (
                _validated_taylor,
                ("validated_box_taylor_step", "flow_jets", "picard_tube"),
            ),
            (
                relational_sine_class_readout,
                ("bound_sine_class_four_history_readout", "_full_sine_field"),
            ),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def report():
    # The two immutable class entries also serve independent algebra controls
    # in the same pytest process. Every public decision remains freshly rebuilt.
    return owner.bound_sine_class_cubic_response(**_arguments())


def test_ten_primitive_inputs_and_fixed_method():
    parameters = signature(owner.bound_sine_class_cubic_response).parameters
    assert set(parameters) == set(_arguments())
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for name in (
        "order",
        "mediator_class",
        "source_report",
        "response",
        "second_state",
    ):
        with pytest.raises(TypeError):
            owner.bound_sine_class_cubic_response(**_arguments(), **{name: 0})


@pytest.mark.parametrize("key", tuple(_arguments()))
@pytest.mark.parametrize("bad", (True, np.bool_(False), float("nan"), float("inf")))
def test_invalid_primitives_are_rejected_before_parameters_or_cache(
    key, bad, monkeypatch
):
    def forbidden(*args):
        pytest.fail("primitive admission must precede coefficient/cache work")

    monkeypatch.setattr(owner, "_parameters", forbidden)
    monkeypatch.setattr(owner, "_class_cubic_coefficients", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_cubic_response(**{**_arguments(), key: bad})


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
def test_exact_domain_and_representation_boundaries(key, value, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_class_cubic_coefficients",
        lambda *_: pytest.fail("invalid input reached coefficient"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_cubic_response(**{**_arguments(), key: value})


def test_fixed_coefficient_resolves_true_sign_but_not_noisy_scalar(report):
    lo, hi = report.complete_cubic_contrast_bounds
    assert Q(-7014, 10**33) < lo <= hi < Q(-7013, 10**33)
    assert hi - lo < Q(1, 10**40)
    assert report.higher_amplitude_contrast_error_upper_bound < Q(16, 10**34)
    assert report.source_contrast_error_upper_bound < Q(81, 10**33)
    assert Q(-7096, 10**33) < report.decision.true_bounds[0]
    assert report.decision.true_bounds[1] < Q(-6932, 10**33)
    assert report.status == "true_sign_certified"
    assert report.decision.orientation == -1
    assert report.decision.true_sign and not report.decision.recorded_sign
    assert not report.decision.null_excluded
    assert report.decision.recorded_bounds[0] < 0 < report.decision.recorded_bounds[1]
    assert report.decision.scalar_cancellation
    assert report.coefficient_evaluated and report.response_bound_available
    assert report.all_identities_certified and report.all_work_within_allowances
    assert report.compared_classes == ((1, 1, 1), (1, 2, 1))


def test_full_eight_segment_carry_and_horner_evidence(report):
    assert report.time_polynomial_order == 64
    assert report.linear_majorant_rate == Q(201, 100)
    assert len(report.class_segments) == len(report.class_parameters) == 2
    for parameters, segments in zip(report.class_parameters, report.class_segments):
        prefix, first, both, second = segments
        assert tuple(s.label for s in segments) == (
            "first_prefix",
            "first_suffix",
            "both_suffix",
            "second_only_suffix",
        )
        assert tuple(s.parent_segment_index for s in segments) == (None, 0, 0, None)
        assert first.initial_levels == prefix.endpoint_levels
        assert both.initial_levels == owner._event_levels(
            prefix.endpoint_levels, Q(1, 2000)
        )
        assert second.initial_levels == owner._event_levels(
            owner._ZERO_LEVELS, Q(1, 2000)
        )
        # Initial coefficients of levels 2 and 3 survive the second event.
        assert both.initial_levels[1:] == prefix.endpoint_levels[1:]
        assert any(v.abs_max > 0 for row in both.initial_levels[1:] for v in row)
        for segment in segments:
            duration = segment.end_time - segment.start_time
            assert duration == 1
            assert len(segment.time_coefficients) == 3
            assert segment.initial_level_norm_upper_bounds == tuple(
                max(v.abs_max for v in row) for row in segment.initial_levels
            )
            for level, rows in enumerate(segment.time_coefficients):
                assert len(rows) == 65 and all(len(row) == 54 for row in rows)
                assert rows[0] == segment.initial_levels[level]
                for coordinate in range(54):
                    value = rows[-1][coordinate]
                    for row in reversed(rows[:-1]):
                        value = value * duration + row[coordinate]
                    tail = segment.time_tail_upper_bounds[level]
                    assert segment.endpoint_levels[level][coordinate] == value + I(
                        -tail, tail
                    )
            assert parameters.eta.hi < Q(1, 3000) ** 2


def test_class_subtraction_precedes_shared_gamma_fourth_enclosure(report):
    mixed = []
    for _, first, both, second in report.class_segments:
        mixed.append(
            both.endpoint_levels[2][22]
            - first.endpoint_levels[2][22]
            - second.endpoint_levels[2][22]
        )
    difference = mixed[0] - mixed[1]
    products = [
        g**4 * v
        for g in (report.gamma_bounds.lo, report.gamma_bounds.hi)
        for v in (difference.lo, difference.hi)
    ]
    assert report.complete_cubic_contrast_bounds == (min(products), max(products))
    assert report.complete_cubic_contrast_bounds[0] != I(min(products)).lo


def test_separate_higher_degree_and_actual_source_errors(report):
    g, t = Q(1, 3000), Q(2)
    amplitudes = (Q(0), Q(1, 2000), Q(1, 2000), Q(1, 1000))
    expected = tuple(
        256 * g**6 * a**5 * t / ((1 - 2 * g * g * t * t) * (1 - 4 * g * g * a * a))
        for a in amplitudes
    )
    assert report.per_history_higher_amplitude_remainder_upper_bounds == expected
    assert report.higher_amplitude_contrast_error_upper_bound == 2 * sum(expected)
    source = 8 * report.endpoint_radius / (1 - 2 * report.gamma_bounds.hi * t)
    assert report.source_contrast_error_upper_bound == source
    lo, hi = report.complete_cubic_contrast_bounds
    error = 2 * sum(expected) + source
    assert report.decision.true_bounds == (lo - error, hi + error)
    assert all(
        h.tangent_endpoint_discrepancy_upper_bound is None
        for h in report.uniform_history_bounds
    )


def test_bounded_cache_reuses_coefficients_but_not_policy_decisions(
    report, monkeypatch
):
    def forbidden(*args):
        pytest.fail("same admitted design must reuse immutable class coefficients")

    monkeypatch.setattr(owner, "_coefficient_segment", forbidden)
    before = owner._class_cubic_coefficients.cache_info()
    changed = owner.bound_sine_class_cubic_response(
        **{
            **_arguments(),
            "endpoint_radius": Q(0),
            "readout_error_bound": Q(0),
            "first_probe_work_allowance": Q(0),
        }
    )
    after = owner._class_cubic_coefficients.cache_info()
    assert after.maxsize == 2 and after.currsize == 2
    assert after.hits == before.hits + 2
    assert changed.class_segments[0] is report.class_segments[0]
    assert changed.source_contrast_error_upper_bound == 0
    assert changed.decision.recorded_sign and changed.decision.null_excluded
    assert not changed.all_work_within_allowances
    assert changed.decision is not report.decision


@pytest.mark.parametrize(
    "multiple,field", ((8, "recorded_sign"), (16, "null_excluded"))
)
def test_strict_observation_boundaries_use_exact_fractions(report, multiple, field):
    ceiling = report.decision.oriented_lower / multiple
    equal = owner.bound_sine_class_cubic_response(
        **{**_arguments(), "readout_error_bound": ceiling}
    )
    assert not getattr(equal.decision, field)
    just_below = owner.bound_sine_class_cubic_response(
        **{**_arguments(), "readout_error_bound": ceiling - Q(1, 10**100)}
    )
    assert getattr(just_below.decision, field)


def test_work_closed_boundary_and_identity_independent_of_observation(report):
    first = max(h.first_probe_work_upper_bound for h in report.uniform_history_bounds)
    second = max(h.second_probe_work_upper_bound for h in report.uniform_history_bounds)
    result = owner.bound_sine_class_cubic_response(
        **{
            **_arguments(),
            "contact_work_allowance": 8 * report.endpoint_radius**2,
            "first_probe_work_allowance": first,
            "second_probe_work_allowance": second,
            "radius": Q(1, 10**8),
        }
    )
    assert result.contact_work_within_allowance and result.all_work_within_allowances
    assert not result.all_identities_certified
    assert result.decision == report.decision
    assert result.response_bound_available


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
def test_exact_pairing_zero_skips_coefficient_and_cauchy_tail(change, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_class_cubic_coefficients",
        lambda *_: pytest.fail("exact identity needs no coefficient"),
    )
    result = owner.bound_sine_class_cubic_response(**{**_arguments(), **change})
    assert result.exact_contrast_zero and result.response_bound_available
    assert not result.coefficient_evaluated
    assert result.complete_cubic_contrast_bounds == (Q(0), Q(0))
    assert result.decision.true_bounds == (Q(0), Q(0))
    assert result.per_history_higher_amplitude_remainder_upper_bounds is None
    assert result.status == "exact_contrast_zero"
    assert result.decision.scalar_cancellation


@pytest.mark.parametrize("amplitude", (Q(1499), Q(1499) + Q(1, 2**400), Q(10**300)))
def test_nonzero_cauchy_domain_failure_abstains_before_coefficients(
    amplitude, monkeypatch
):
    monkeypatch.setattr(
        owner,
        "_class_cubic_coefficients",
        lambda *_: pytest.fail("unsupported Cauchy domain reached coefficient"),
    )
    result = owner.bound_sine_class_cubic_response(
        **{
            **_arguments(),
            "first_probe_amplitude": amplitude,
            "second_probe_amplitude": 1,
        }
    )
    assert result.cauchy_radius_margin <= 0
    assert not result.cauchy_admitted and not result.coefficient_evaluated
    assert result.status == "unavailable" and not result.response_bound_available
    assert result.decision is None and result.complete_cubic_contrast_bounds is None
    assert result.higher_amplitude_contrast_error_upper_bound is None
    assert result.class_segments == ()


def test_exact_tiny_rational_and_represented_float_preserved_without_zero_coercion():
    tiny = Q(1, 10**400)
    report = owner.bound_sine_class_cubic_response(
        **{
            **_arguments(),
            "first_probe_amplitude": 0,
            "second_probe_amplitude": -tiny,
            "endpoint_radius": 0.125,
        }
    )
    assert report.second_probe_amplitude == -tiny
    assert report.endpoint_radius == Q(1, 8)
    assert not report.joined_initial_identity_certified
    assert report.exact_contrast_zero


def test_real_unavailable_report_serializes_exact_values_and_nulls():
    report = owner.bound_sine_class_cubic_response(
        **{**_arguments(), "first_probe_amplitude": 1499, "second_probe_amplitude": 1}
    )
    encoded = relational_report_to_dict(report)
    assert encoded["schema"] == "tnfr.relational-report.v1"
    assert encoded["report_type"] == "SineClassCubicResponse"
    assert encoded["report"] == report.to_dict()["report"]
    loaded = json_loads(json.dumps(encoded, allow_nan=False))
    assert report.to_dict()["schema"] == "tnfr.sine-class-cubic-response.v1"
    body = loaded["report"]
    assert body["first_probe_amplitude"] == {"numerator": 1499, "denominator": 1}
    assert body["decision"] is None and body["complete_cubic_contrast_bounds"] is None
    assert body["cauchy_radius_margin"] == {"numerator": 0, "denominator": 1}

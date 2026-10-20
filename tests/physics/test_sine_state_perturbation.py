"""Independent full-law perturbation bounds without trajectories or events."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_support import (
    SineSupportExchangeComparison,
    bound_relational_sine_support_exchange,
)
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

FORM = (Q(-1, 4), Q(1, 8), Q(1, 2))
PHASE = (Q(0), Q(1, 4), Q(-1, 8))
FORM_ERRORS = (Q(1, 64), Q(1, 128), Q(1, 256))
PHASE_ERRORS = (Q(1, 128), Q(1, 256), Q(1, 512))


def _source(*, conductances=(2, 3), capacities=(2, 3, 5)):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node, (form, phase, capacity) in enumerate(zip(FORM, PHASE, capacities)):
        graph.add_node(node, EPI=float(form), theta=float(phase), nu_f=float(capacity))
    for edge, weight in zip(((0, 1), (1, 2)), conductances):
        graph.add_edge(*edge, weight=float(weight))
    return bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            2, epi_weight=0, phase_domain="regular"
        ),
    )


def _bound(source, *, duration=Q(1, 32), forms=FORM_ERRORS, phases=PHASE_ERRORS):
    return source.initial_state_perturbation_bounds(
        duration=duration, form_error_bounds=forms, phase_error_bounds=phases
    )


@pytest.fixture(scope="module")
def source():
    return _source()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.mark.parametrize("exchange", (Q(1), Q(3, 2)))
def test_independent_unequal_weight_capacity_and_channel_coefficients(
    source, mp, exchange
):
    model = copy(source.reference_model)
    object.__setattr__(model, "phase_weight", exchange)
    report = _bound(replace(source, reference_model=model))
    # Direct row sums: degrees(1,2,1), strengths(2,5,3), beta2.
    # These are two separate channel units, not one mixed form/angle norm.
    a_coefficients, b_coefficients = (4, 6, 10), (4, Q(15, 2), 15)
    for actuals, coefficients in (
        (report.form_phase_lipschitz_upper_bounds, a_coefficients),
        (report.phase_form_lipschitz_upper_bounds, b_coefficients),
    ):
        for actual, coefficient in zip(actuals, coefficients):
            expected = _mp(mp, exchange * coefficient) / mp.pi
            assert 0 <= _mp(mp, actual) - expected < mp.mpf("1e-30")
    a = report.max_form_phase_lipschitz_upper_bound
    b = report.max_phase_form_lipschitz_upper_bound
    h = Q(1, 32)
    denominator = 1 - a * b * h**2
    assert report.comparison_denominator == denominator > 0
    assert a == max(report.form_phase_lipschitz_upper_bounds)
    assert b == max(report.phase_form_lipschitz_upper_bounds)
    x = (Q(1, 64) + a * h / 128) / denominator
    theta = (Q(1, 128) + b * h / 64) / denominator
    assert report.form_error_supremum_upper_bound == x
    assert report.phase_error_supremum_upper_bound == theta
    assert report.whole_window_form_error_bounds == tuple(
        error + coefficient * h * theta
        for error, coefficient in zip(
            FORM_ERRORS, report.form_phase_lipschitz_upper_bounds
        )
    )
    assert report.whole_window_phase_error_bounds == tuple(
        error + coefficient * h * x
        for error, coefficient in zip(
            PHASE_ERRORS, report.phase_form_lipschitz_upper_bounds
        )
    )
    assert max(report.whole_window_form_error_bounds) <= x
    assert max(report.whole_window_phase_error_bounds) <= theta
    assert report.status == "enclosed" and report.clock == "t"
    assert report.comparison_method == "coupled_supremum_small_gain"
    assert report.source.reference_model.phase_weight == exchange
    assert report.source.reference_model.storage_scale == 2
    assert source.reference_model.phase_weight == 1
    with pytest.raises(FrozenInstanceError):
        report.duration = Q(2)


def test_initial_storage_box_retains_actual_weights_beta_and_phase_units(source, mp):
    report = _bound(source)
    form_upper, phase_upper = Q(0), mp.mpf(0)
    for (i, j), weight in zip(source.edges, source.conductances):
        form_upper += (
            weight * (abs(FORM[i] - FORM[j]) + FORM_ERRORS[i] + FORM_ERRORS[j]) ** 2 / 2
        )
        phase_upper += min(
            mp.mpf(2),
            1
            - mp.cos(_mp(mp, PHASE[j] - PHASE[i]))
            + _mp(mp, PHASE_ERRORS[i] + PHASE_ERRORS[j]),
        )
    assert report.initial_form_storage_upper_bound == form_upper
    assert (
        0
        <= _mp(mp, report.initial_phase_storage_upper_bound) - 2 * phase_upper
        < mp.mpf("1e-30")
    )
    assert (
        report.initial_storage_upper_bound
        == form_upper + report.initial_phase_storage_upper_bound
    )
    # One corner has opposite signed errors at each adjacent pair. Its
    # complete storage is computed directly, not from the report's verdict.
    forms = tuple(
        value + sign * error
        for value, sign, error in zip(FORM, (1, -1, 1), FORM_ERRORS)
    )
    phases = tuple(
        value + sign * error
        for value, sign, error in zip(PHASE, (-1, 1, -1), PHASE_ERRORS)
    )
    actual = sum(
        _mp(mp, weight * (forms[i] - forms[j]) ** 2 / 2)
        + 2 * (1 - mp.cos(_mp(mp, phases[j] - phases[i])))
        for (i, j), weight in zip(source.edges, source.conductances)
    )
    assert actual <= _mp(mp, report.initial_storage_upper_bound)


@pytest.mark.parametrize("duration", (Q(1, 64), Q(7)))
def test_zero_conductance_exact_nonlinear_solution_has_constant_phases(duration, mp):
    source = replace(
        _source(conductances=(0, 0), capacities=(2, 0, 3)), epi=(Q(100), Q(-100), Q(0))
    )
    report = _bound(source, duration=duration)
    assert report.max_phase_form_lipschitz_upper_bound == 0
    assert report.comparison_denominator == 1
    assert report.whole_window_phase_error_bounds == PHASE_ERRORS
    # With no weighted gradient, phases are fixed and form is affine for
    # the complete nonlinear law, including the supported bare sine edges.
    actual_form_error = (FORM_ERRORS[0], -FORM_ERRORS[1], FORM_ERRORS[2])
    actual_phase_error = (-PHASE_ERRORS[0], PHASE_ERRORS[1], -PHASE_ERRORS[2])
    perturbed_phase = tuple(
        value + error for value, error in zip(PHASE, actual_phase_error)
    )
    reference_current = [mp.mpf(0)] * 3
    perturbed_current = [mp.mpf(0)] * 3
    for i, j in source.edges:
        old = mp.sin(_mp(mp, PHASE[j] - PHASE[i]))
        new = mp.sin(_mp(mp, perturbed_phase[j] - perturbed_phase[i]))
        reference_current[i] += old
        reference_current[j] -= old
        perturbed_current[i] += new
        perturbed_current[j] -= new
    for i, (capacity, degree) in enumerate(zip((2, 0, 3), (1, 2, 1))):
        error = (
            _mp(mp, actual_form_error[i])
            + _mp(mp, duration * capacity / degree)
            * (perturbed_current[i] - reference_current[i])
            / mp.pi
        )
        assert abs(error) <= _mp(mp, report.whole_window_form_error_bounds[i])
    assert report.whole_window_form_error_bounds[1] == FORM_ERRORS[1]
    assert source.epi[:2] == (100, -100)  # No runtime clipping is part of this law.


@pytest.mark.parametrize("capacities", ((0, 0, 0), (2, 0, 5)))
def test_zero_capacity_keeps_both_error_rows_frozen(capacities):
    report = _bound(_source(capacities=capacities))
    for i, capacity in enumerate(capacities):
        if capacity == 0:
            assert report.source.pressure[i].abs_max > 0
            assert report.form_phase_lipschitz_upper_bounds[i] == 0
            assert report.phase_form_lipschitz_upper_bounds[i] == 0
            assert report.whole_window_form_error_bounds[i] == FORM_ERRORS[i]
            assert report.whole_window_phase_error_bounds[i] == PHASE_ERRORS[i]
    if not any(capacities):
        assert report.form_error_supremum_upper_bound == max(FORM_ERRORS)
        assert report.phase_error_supremum_upper_bound == max(PHASE_ERRORS)


def test_zero_initial_error_is_exact_at_any_finite_duration_by_uniqueness(source):
    report = _bound(source, duration=100, forms=(0, 0, 0), phases=(0, 0, 0))
    assert report.comparison_denominator < 0
    assert report.status == "enclosed"
    assert report.comparison_method == "zero_initial_error_uniqueness"
    assert (
        report.form_error_supremum_upper_bound
        == report.phase_error_supremum_upper_bound
        == 0
    )
    assert (
        report.whole_window_form_error_bounds
        == report.whole_window_phase_error_bounds
        == (0, 0, 0)
    )


def test_short_window_boundary_is_unavailable_not_instability():
    source = _source(conductances=(2, 2), capacities=(1, 1, 1))
    initial = _bound(source)
    a = initial.max_form_phase_lipschitz_upper_bound
    assert a == initial.max_phase_form_lipschitz_upper_bound
    boundary = 1 / a
    below = _bound(source, duration=(1 - Q(1, 2**40)) * boundary)
    assert below.comparison_denominator > 0 and below.status == "enclosed"
    for duration in (boundary, (1 + Q(1, 2**40)) * boundary):
        report = _bound(source, duration=duration)
        assert report.comparison_denominator <= 0
        assert report.status == "unavailable"
        assert report.unavailable_reasons
        assert report.form_error_supremum_upper_bound is None
        assert report.phase_error_supremum_upper_bound is None
        assert report.whole_window_form_error_bounds is None
        assert report.whole_window_phase_error_bounds is None
        assert report.initial_storage_upper_bound > 0


def test_large_phase_radius_uses_finite_cosine_storage_cap():
    source = replace(_source(conductances=(0, 0)), phase=(Q(0),) * 3)
    report = _bound(source, phases=(2, 2, 2), forms=(0, 0, 0))
    assert report.initial_form_storage_upper_bound == 0
    assert report.initial_phase_storage_upper_bound == 8
    assert report.initial_storage_upper_bound == 8  # Two bare edges, beta2.


def test_common_form_and_radian_translations_do_not_change_any_bound(source):
    original = _bound(source)
    moved = _bound(
        replace(
            source,
            epi=tuple(x + 7 for x in source.epi),
            phase=tuple(theta - Q(5, 8) for theta in source.phase),
        )
    )
    for field in (
        "form_phase_lipschitz_upper_bounds",
        "phase_form_lipschitz_upper_bounds",
        "comparison_denominator",
        "form_error_supremum_upper_bound",
        "phase_error_supremum_upper_bound",
        "whole_window_form_error_bounds",
        "whole_window_phase_error_bounds",
        "initial_form_storage_upper_bound",
        "initial_phase_storage_upper_bound",
        "initial_storage_upper_bound",
    ):
        assert getattr(moved, field) == getattr(original, field)


@pytest.mark.parametrize(
    "duration", (0, -1, True, float("nan"), float("inf"), complex(1, 0))
)
def test_positive_duration_admission(source, duration):
    with pytest.raises((TypeError, ValueError)):
        _bound(source, duration=duration)


@pytest.mark.parametrize(
    "field,values",
    (
        ("forms", (-1, 0, 0)),
        ("forms", (True, 0, 0)),
        ("forms", (0, 0)),
        ("phases", (0, float("inf"), 0)),
        ("phases", (False, 0, 0)),
        ("phases", (0, 0, 0, 0)),
    ),
)
def test_error_radius_admission_does_not_coerce_or_drop_coordinates(
    source, field, values
):
    with pytest.raises((TypeError, ValueError)):
        _bound(source, **{field: values})


def test_exact_rational_and_binary64_radii_remain_distinct(source):
    exact = _bound(source, duration=Q(1, 10), forms=(Q(1, 10), 0, 0))
    represented = _bound(source, duration=0.1, forms=(0.1, 0, 0))
    assert exact.duration == exact.form_error_bounds[0] == Q(1, 10)
    assert represented.duration == represented.form_error_bounds[0] == Q.from_float(0.1)
    assert exact.duration != represented.duration


@pytest.mark.parametrize(
    "field,value",
    (
        ("capacity", True),
        ("phase", float("nan")),
        ("epi", float("inf")),
        ("conductances", -1),
    ),
)
def test_invalid_primitive_source_cannot_use_favorable_cached_bounds(
    source, field, value
):
    forged = replace(
        source,
        **{field: (value,) + getattr(source, field)[1:]},
        form_rates=(I(0),) * 3,
        storage=I(0),
    )
    with pytest.raises((TypeError, ValueError)):
        _bound(forged)


def test_fresh_sdk_export_discards_all_cached_envelopes_and_preserves_atomic_failure(
    source, tmp_path
):
    report = _bound(source)
    poisoned_source = replace(
        source,
        degrees=(99,) * 3,
        form_gradient=(Q(0),) * 3,
        form_rates=(I(0),) * 3,
        phase_rates=(I(0),) * 3,
        pressure=(I(0),) * 3,
        storage=I(0),
    )
    assert _bound(poisoned_source) == report
    forged = replace(
        report,
        source=poisoned_source,
        form_phase_lipschitz_upper_bounds=(Q(0),) * 3,
        phase_form_lipschitz_upper_bounds=(Q(0),) * 3,
        max_form_phase_lipschitz_upper_bound=Q(0),
        max_phase_form_lipschitz_upper_bound=Q(0),
        comparison_denominator=Q(1),
        form_error_supremum_upper_bound=Q(0),
        phase_error_supremum_upper_bound=Q(0),
        whole_window_form_error_bounds=(Q(0),) * 3,
        whole_window_phase_error_bounds=(Q(0),) * 3,
        initial_form_storage_upper_bound=Q(0),
        initial_phase_storage_upper_bound=Q(0),
        initial_storage_upper_bound=Q(0),
        status="unavailable",
    )
    expected = relational_report_to_dict(report)
    assert relational_report_to_dict(forged) == expected
    path = tmp_path / "sine-perturbation.json"
    export_to_json(relational_report_to_dict(forged), path)
    assert json_loads(path.read_text(encoding="utf-8")) == expected
    before = path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(replace(forged, phase_error_bounds=(True, 0, 0))),
            path,
        )
    assert path.read_bytes() == before
    unavailable = _bound(source, duration=100)
    payload = relational_report_to_dict(unavailable)
    assert payload["report"]["status"] == "unavailable"
    assert payload["report"]["whole_window_form_error_bounds"] is None


def test_untrusted_source_cannot_dispatch_a_custom_perturbation_method(source):
    class Fake:
        def initial_state_perturbation_bounds(self, **kwargs):
            raise AssertionError("unadmitted perturbation dispatch")

    with pytest.raises(TypeError):
        SineSupportExchangeComparison.initial_state_perturbation_bounds(
            Fake(), duration=1, form_error_bounds=(0,), phase_error_bounds=(0,)
        )
    with pytest.raises(TypeError):
        replace(_bound(source), source=Fake()).to_dict()

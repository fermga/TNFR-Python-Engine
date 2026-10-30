"""Independent two-law comparison controls without trajectories or events."""

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
WEIGHT_ERRORS = (Q(1, 16), Q(3, 32))


def _source(*, weights=(2, 3), capacities=(2, 3, 5)):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node, (form, phase, capacity) in enumerate(zip(FORM, PHASE, capacities)):
        graph.add_node(node, EPI=float(form), theta=float(phase), nu_f=float(capacity))
    for edge, weight in zip(((0, 1), (1, 2)), weights):
        graph.add_edge(*edge, weight=float(weight))
    return bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            2, epi_weight=0, phase_domain="regular"
        ),
    )


def _bound(
    source,
    *,
    duration=Q(1, 32),
    forms=FORM_ERRORS,
    phases=PHASE_ERRORS,
    weights=WEIGHT_ERRORS,
):
    return source.state_and_conductance_perturbation_bounds(
        duration=duration,
        form_error_bounds=forms,
        phase_error_bounds=phases,
        conductance_error_bounds=weights,
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
def test_independent_weighted_lipschitz_channels_and_nonuniform_law_forcing(
    source, mp, exchange
):
    model = copy(source.reference_model)
    object.__setattr__(model, "phase_weight", exchange)
    report = _bound(replace(source, reference_model=model))
    h, beta = Q(1, 32), Q(2)
    capacities, degrees = (2, 3, 5), (1, 2, 1)
    upper_weights = (Q(33, 16), Q(99, 32))
    strengths = (upper_weights[0], sum(upper_weights), upper_weights[1])
    assert report.conductance_upper_bounds == upper_weights
    true_speeds = tuple(_mp(mp, exchange * capacity) / mp.pi for capacity in capacities)
    true_contrasts = tuple(
        _mp(mp, abs(FORM[i] - FORM[j])) + _mp(mp, h) * (true_speeds[i] + true_speeds[j])
        for i, j in ((0, 1), (1, 2))
    )
    for actual, expected in zip(report.reference_form_speed_upper_bounds, true_speeds):
        assert 0 <= _mp(mp, actual) - expected < mp.mpf("1e-30")
    for actual, expected in zip(
        report.reference_edge_form_contrast_upper_bounds, true_contrasts
    ):
        assert 0 <= _mp(mp, actual) - expected < mp.mpf("1e-30")
    forcing_sums = (
        _mp(mp, WEIGHT_ERRORS[0]) * true_contrasts[0],
        sum(
            _mp(mp, radius) * contrast
            for radius, contrast in zip(WEIGHT_ERRORS, true_contrasts)
        ),
        _mp(mp, WEIGHT_ERRORS[1]) * true_contrasts[1],
    )
    for node in range(3):
        expected_a = _mp(mp, 2 * exchange * capacities[node]) / mp.pi
        expected_b = (
            _mp(
                mp,
                2
                * exchange
                * capacities[node]
                * strengths[node]
                / (beta * degrees[node]),
            )
            / mp.pi
        )
        expected_f = (
            _mp(mp, exchange * capacities[node] / (beta * degrees[node]))
            * forcing_sums[node]
            / mp.pi
        )
        for actual, expected in (
            (report.form_phase_lipschitz_upper_bounds[node], expected_a),
            (report.phase_form_lipschitz_upper_bounds[node], expected_b),
            (report.phase_law_forcing_upper_bounds[node], expected_f),
        ):
            assert 0 <= _mp(mp, actual) - expected < mp.mpf("1e-30")
    a = max(report.form_phase_lipschitz_upper_bounds)
    b = max(report.phase_form_lipschitz_upper_bounds)
    forcing = max(report.phase_law_forcing_upper_bounds)
    denominator = 1 - a * b * h**2
    x = (
        max(FORM_ERRORS) + a * h * max(PHASE_ERRORS) + a * h**2 * forcing
    ) / denominator
    theta = (max(PHASE_ERRORS) + b * h * max(FORM_ERRORS) + h * forcing) / denominator
    assert report.comparison_denominator == denominator > 0
    assert report.form_error_supremum_upper_bound == x
    assert report.phase_error_supremum_upper_bound == theta
    assert report.max_phase_law_forcing_upper_bound == forcing > 0
    assert report.whole_window_form_error_bounds == tuple(
        error + coefficient * h * theta
        for error, coefficient in zip(
            FORM_ERRORS, report.form_phase_lipschitz_upper_bounds
        )
    )
    assert report.whole_window_phase_error_bounds == tuple(
        error + coefficient * h * x + h * force
        for error, coefficient, force in zip(
            PHASE_ERRORS,
            report.phase_form_lipschitz_upper_bounds,
            report.phase_law_forcing_upper_bounds,
        )
    )
    assert max(report.whole_window_form_error_bounds) <= x
    assert max(report.whole_window_phase_error_bounds) <= theta
    assert report.source.reference_model.phase_weight == exchange
    assert source.reference_model.phase_weight == 1
    assert report.status == "enclosed" and report.clock == "t"
    assert report.phase_error_units == "radians_in_declared_real_lifts"
    with pytest.raises(FrozenInstanceError):
        report.duration = Q(2)


def test_independent_initial_storage_uses_upper_weights_and_beta_scaled_bare_phase(
    source, mp
):
    report = _bound(source)
    form_upper, phase_upper = Q(0), mp.mpf(0)
    for (i, j), weight, error in zip(source.edges, source.conductances, WEIGHT_ERRORS):
        form_upper += (
            (weight + error)
            * (abs(FORM[i] - FORM[j]) + FORM_ERRORS[i] + FORM_ERRORS[j]) ** 2
            / 2
        )
        phase_upper += 2 * min(
            mp.mpf(2),
            1
            - mp.cos(_mp(mp, PHASE[j] - PHASE[i]))
            + _mp(mp, PHASE_ERRORS[i] + PHASE_ERRORS[j]),
        )
    assert report.initial_form_storage_upper_bound == form_upper
    assert (
        0
        <= _mp(mp, report.initial_phase_storage_upper_bound) - phase_upper
        < mp.mpf("1e-30")
    )
    assert (
        report.initial_storage_upper_bound
        == form_upper + report.initial_phase_storage_upper_bound
    )
    # The lower box face is truncated at zero, not rejected or treated as a
    # signed conductance. Storage still uses the positive upper face.
    wide = _bound(source, weights=(3, 4))
    assert wide.conductance_upper_bounds == (5, 7)
    assert wide.initial_form_storage_upper_bound > form_upper


def test_zero_conductance_error_reduces_to_shared_same_law_kernel(source):
    changed = _bound(source, weights=(0, 0))
    same = source.initial_state_perturbation_bounds(
        duration=Q(1, 32),
        form_error_bounds=FORM_ERRORS,
        phase_error_bounds=PHASE_ERRORS,
    )
    for field in same.__dataclass_fields__:
        if field != "scope":
            assert getattr(changed, field) == getattr(same, field)
    assert changed.phase_law_forcing_upper_bounds == (0, 0, 0)
    assert changed.conductance_upper_bounds == source.conductances


def test_zero_state_error_with_changed_weights_is_not_flow_equality(source, mp):
    nominal = replace(source, conductances=(Q(0), Q(0)), phase=(Q(0),) * 3)
    report = _bound(nominal, forms=(0, 0, 0), phases=(0, 0, 0))
    assert report.comparison_method == "coupled_supremum_small_gain"
    assert report.form_error_supremum_upper_bound > 0
    assert report.phase_error_supremum_upper_bound > 0
    assert report.max_phase_law_forcing_upper_bound > 0
    assert report.source.phase_rates == report.source.form_rates == (I(0),) * 3
    # An independently rebuilt candidate has the same x,theta and x_dot=0,
    # but its changed weighted phase rows immediately drive nonzero x_ddot.
    candidate = _source(weights=WEIGHT_ERRORS)
    candidate = bound_relational_sine_support_exchange(
        _graph_from_source(replace(candidate, phase=(Q(0),) * 3)),
        reference_model=nominal.reference_model,
    )
    assert candidate.form_rates == (I(0),) * 3
    gradient = [Q(0)] * 3
    for (i, j), weight in zip(candidate.edges, WEIGHT_ERRORS):
        gradient[i] += weight * (FORM[i] - FORM[j])
        gradient[j] += weight * (FORM[j] - FORM[i])
    phase_numerators = tuple(
        capacity * value / (2 * degree)
        for capacity, value, degree in zip((2, 3, 5), gradient, (1, 2, 1))
    )
    expected_first_acceleration = (
        _mp(mp, 2 * (phase_numerators[1] - phase_numerators[0])) / mp.pi**2
    )
    assert expected_first_acceleration != 0
    actual = candidate.form_acceleration_bounds()[0]
    assert _mp(mp, actual.lo) <= expected_first_acceleration <= _mp(mp, actual.hi)
    assert not actual.contains(0)


def _graph_from_source(source):
    graph = nx.Graph()
    graph.graph["GAMMA"] = {"type": "none"}
    for node, x, phase, capacity in zip(
        source.nodes, source.epi, source.phase, source.capacity
    ):
        graph.add_node(node, EPI=float(x), theta=float(phase), nu_f=float(capacity))
    for edge, weight in zip(source.edges, source.conductances):
        graph.add_edge(*edge, weight=float(weight))
    return graph


@pytest.mark.parametrize("capacities", ((0, 0, 0), (2, 0, 5)))
def test_zero_capacity_freezes_error_rows_even_with_conductance_uncertainty(capacities):
    report = _bound(
        _source(capacities=capacities),
        duration=100 if not any(capacities) else Q(1, 32),
    )
    for i, capacity in enumerate(capacities):
        if capacity == 0:
            assert report.source.pressure[i].abs_max > 0
            assert report.form_phase_lipschitz_upper_bounds[i] == 0
            assert report.phase_form_lipschitz_upper_bounds[i] == 0
            assert report.phase_law_forcing_upper_bounds[i] == 0
            assert report.whole_window_form_error_bounds[i] == FORM_ERRORS[i]
            assert report.whole_window_phase_error_bounds[i] == PHASE_ERRORS[i]
    assert report.status == "enclosed"


def test_zero_weights_without_weight_error_keep_phase_errors_constant_for_any_window(
    source,
):
    report = _bound(
        replace(source, conductances=(Q(0), Q(0))), duration=100, weights=(0, 0)
    )
    assert report.max_phase_form_lipschitz_upper_bound == 0
    assert report.max_phase_law_forcing_upper_bound == 0
    assert report.comparison_denominator == 1
    assert report.whole_window_phase_error_bounds == PHASE_ERRORS
    assert report.status == "enclosed"


def test_zero_state_and_weight_box_has_exact_uniqueness_at_arbitrary_horizon(source):
    report = _bound(
        source, duration=100, forms=(0, 0, 0), phases=(0, 0, 0), weights=(0, 0)
    )
    assert report.comparison_denominator < 0
    assert report.comparison_method == "zero_state_and_conductance_error_uniqueness"
    assert (
        report.whole_window_form_error_bounds
        == report.whole_window_phase_error_bounds
        == (0, 0, 0)
    )
    assert report.status == "enclosed"
    changed = _bound(source, duration=100, forms=(0, 0, 0), phases=(0, 0, 0))
    assert changed.status == "unavailable"
    assert changed.form_error_supremum_upper_bound is None


def test_exact_short_time_boundary_reports_unavailability_not_instability():
    source = _source(weights=(1, 1), capacities=(1, 1, 1))
    report = _bound(source, weights=(1, 1))
    a = report.max_form_phase_lipschitz_upper_bound
    assert a == report.max_phase_form_lipschitz_upper_bound
    for factor in (1, 1 + Q(1, 2**40)):
        result = _bound(source, duration=factor / a, weights=(1, 1))
        assert result.comparison_denominator <= 0
        assert result.status == "unavailable" and result.unavailable_reasons
        assert result.whole_window_form_error_bounds is None
        assert result.whole_window_phase_error_bounds is None
        assert result.phase_error_supremum_upper_bound is None
        assert result.initial_storage_upper_bound > 0
    below = _bound(source, duration=(1 - Q(1, 2**40)) / a, weights=(1, 1))
    assert below.status == "enclosed" and below.comparison_denominator > 0


def test_common_translation_and_permuted_edge_orientation_preserve_physical_bounds(
    source,
):
    original = _bound(source)
    moved = _bound(
        replace(
            source,
            epi=tuple(x + 8 for x in source.epi),
            phase=tuple(theta - 3 for theta in source.phase),
        )
    )
    reverse = _bound(
        replace(source, edges=((2, 1), (1, 0)), conductances=(Q(3), Q(2))),
        weights=WEIGHT_ERRORS[::-1],
    )
    for field in (
        "phase_law_forcing_upper_bounds",
        "form_phase_lipschitz_upper_bounds",
        "phase_form_lipschitz_upper_bounds",
        "comparison_denominator",
        "form_error_supremum_upper_bound",
        "phase_error_supremum_upper_bound",
        "initial_storage_upper_bound",
    ):
        assert getattr(moved, field) == getattr(original, field)
        assert getattr(reverse, field) == getattr(original, field)
    assert (
        reverse.reference_edge_form_contrast_upper_bounds
        == original.reference_edge_form_contrast_upper_bounds[::-1]
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("duration", True),
        ("duration", 0),
        ("duration", float("inf")),
        ("weights", (True, 0)),
        ("weights", (-1, 0)),
        ("weights", (0,)),
        ("weights", (0, 0, 0)),
        ("weights", (0, float("nan"))),
        ("forms", (0, False, 0)),
        ("phases", (0, -1, 0)),
    ),
)
def test_primitive_budget_admission(source, field, value):
    with pytest.raises((TypeError, ValueError)):
        _bound(source, **{field: value})


def test_exact_and_represented_weight_budgets_are_not_conflated(source):
    exact = _bound(source, weights=(Q(1, 10), 0))
    represented = _bound(source, weights=(0.1, 0))
    assert exact.conductance_error_bounds[0] == Q(1, 10)
    assert represented.conductance_error_bounds[0] == Q.from_float(0.1)
    assert exact.conductance_error_bounds != represented.conductance_error_bounds


@pytest.mark.parametrize(
    "field,value",
    (
        ("capacity", True),
        ("conductances", -1),
        ("phase", float("nan")),
        ("epi", float("inf")),
    ),
)
def test_invalid_source_primitives_cannot_use_favorable_cached_claims(
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


def test_sdk_rebuilds_cached_coefficients_envelopes_and_source_then_exports_atomically(
    source, tmp_path
):
    report = _bound(source)
    poisoned = replace(
        source,
        degrees=(99,) * 3,
        form_gradient=(Q(0),) * 3,
        form_rates=(I(0),) * 3,
        pressure=(I(0),) * 3,
        storage=I(0),
    )
    assert _bound(poisoned) == report
    forged = replace(
        report,
        source=poisoned,
        conductance_upper_bounds=(Q(0),) * 2,
        phase_law_forcing_upper_bounds=(Q(0),) * 3,
        max_phase_law_forcing_upper_bound=Q(0),
        reference_form_speed_upper_bounds=(Q(0),) * 3,
        reference_edge_form_contrast_upper_bounds=(Q(0),) * 2,
        comparison_denominator=Q(1),
        form_error_supremum_upper_bound=Q(0),
        phase_error_supremum_upper_bound=Q(0),
        initial_storage_upper_bound=Q(0),
        whole_window_form_error_bounds=(Q(0),) * 3,
        status="unavailable",
    )
    expected = relational_report_to_dict(report)
    assert relational_report_to_dict(forged) == expected
    path = tmp_path / "changed-conductance.json"
    export_to_json(relational_report_to_dict(forged), path)
    assert json_loads(path.read_text(encoding="utf-8")) == expected
    before = path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        export_to_json(
            relational_report_to_dict(
                replace(forged, conductance_error_bounds=(False, 0))
            ),
            path,
        )
    assert path.read_bytes() == before
    unavailable = relational_report_to_dict(_bound(source, duration=100))
    assert unavailable["report"]["status"] == "unavailable"
    assert unavailable["report"]["whole_window_form_error_bounds"] is None


def test_fake_source_cannot_dispatch_its_own_changed_law_reader(source):
    class Fake:
        def state_and_conductance_perturbation_bounds(self, **kwargs):
            raise AssertionError("unadmitted dispatch")

    with pytest.raises(TypeError):
        SineSupportExchangeComparison.state_and_conductance_perturbation_bounds(
            Fake(),
            duration=1,
            form_error_bounds=(0,),
            phase_error_bounds=(0,),
            conductance_error_bounds=(),
        )
    with pytest.raises(TypeError):
        replace(_bound(source), source=Fake()).to_dict()

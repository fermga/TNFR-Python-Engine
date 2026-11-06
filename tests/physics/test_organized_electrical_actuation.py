"""Static open-loop input and admission controls for one finite intervention.

These checks do not run the reserved source assessment or integrate its pulse.
The prescribed shift satisfies the linear phase row; its unremoved nonlinear
current requires the separate residual bound rather than exact feedback.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from types import SimpleNamespace

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_electrical_actuation as owner
from tnfr.physics._organized_outer_winding import (
    _derive_organized_outer_winding,
    _observe_organized_outer_winding,
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def arithmetic():
    context = mpmath.mp.clone()
    context.dps = 100
    laplacian = tuple(
        tuple(Q(2 if i == j else -1 if (i - j) % 5 in (1, 4) else 0) for j in range(5))
        for i in range(5)
    )
    inverse = exact_matrix_inverse(
        tuple(tuple(value + Q(1, 5) for value in row) for row in laplacian)
    )
    centered = tuple(tuple(value - Q(1, 5) for value in row) for row in inverse)
    return context, laplacian, centered


def _mp(context, value):
    return context.mpf(value.numerator) / value.denominator


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


def _action(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _pulse(**changes):
    arguments = dict(
        elapsed=Q(7, 12), duration=Q(7, 3), amplitude=Q(1, 10), marked_child=2
    )
    arguments.update(changes)
    return owner._evaluate_organized_electrical_pulse(**arguments)


def _bounds(**changes):
    arguments = dict(
        initial_form_bound=Q(1, 100),
        initial_hidden_form=Q(1, 10**5),
        initial_hidden_phase=Q(1, 10**6),
        outer_gap=Q(6, 5),
        amplitude=Q(1, 10),
        duration=Q(3, 4),
    )
    arguments.update(changes)
    return owner._bound_organized_electrical_pulse(**arguments)


@pytest.mark.parametrize("child", (0, 2, 4))
@pytest.mark.parametrize("position", (Q(0), Q(1, 4), Q(1, 2), Q(1)))
@pytest.mark.parametrize("amplitude", (Q(1, 10), Q(-1, 5)))
def test_prescribed_neutral_shift_satisfies_complete_linear_phase_row(
    arithmetic, child, position, amplitude
):
    context, laplacian, inverse = arithmetic
    duration = Q(7, 3)
    report = _pulse(
        marked_child=child,
        elapsed=position * duration,
        duration=duration,
        amplitude=amplitude,
    )
    direction = tuple(Q(int(i == child)) - Q(1, 5) for i in range(5))
    inverse_direction = _action(inverse, direction)
    assert report.direction == direction
    assert report.inverse_direction == inverse_direction
    assert _action(laplacian, inverse_direction) == direction
    assert sum(direction) == sum(inverse_direction) == 0
    gamma = 1 / (1023 * context.pi)
    u, a, d = (_mp(context, value) for value in (position, amplitude, duration))
    shape, slope, curvature = 3 * u**2 - 2 * u**3, 6 * u - 6 * u**2, 6 - 12 * u
    phase = tuple(a * _mp(context, value) * shape for value in direction)
    form = tuple(
        4 * a * _mp(context, value) * slope / (gamma * d) for value in inverse_direction
    )
    form_rate = tuple(
        4 * a * _mp(context, value) * curvature / (gamma * d**2)
        for value in inverse_direction
    )
    gradient = _action(laplacian, form)
    for node in range(5):
        phase_rate = a * _mp(context, direction[node]) * slope / d
        _contains(context, report.phase_offset_bounds[node], phase[node])
        _contains(context, report.form_offset_bounds[node], form[node])
        _contains(context, report.form_tau_rate_bounds[node], form_rate[node])
        _contains(context, report.phase_tau_rate_bounds[node], phase_rate)
        _contains(
            context,
            report.form_input_bounds[node],
            form_rate[node] + gradient[node] / 4,
        )
        assert context.almosteq(phase_rate, gamma * gradient[node] / 4)
    if position in (Q(0), Q(1)):
        assert all(value.lo == value.hi == 0 for value in report.form_offset_bounds)
        assert all(value.lo == value.hi == 0 for value in report.phase_tau_rate_bounds)


def test_open_loop_shift_does_not_cancel_the_nonlinear_phase_current(arithmetic):
    context, _, _ = arithmetic
    report = _pulse(elapsed=Q(7, 6), marked_child=0)
    phase = tuple(
        _mp(context, report.amplitude * value / 2) for value in report.direction
    )
    sine_current = tuple(
        context.sin(phase[(i - 1) % 5] - phase[i])
        + context.sin(phase[(i + 1) % 5] - phase[i])
        for i in range(5)
    )
    # Even at an initially stationary carrier, the moved phase produces a
    # nonzero current absent from the prescribed shift's form derivative.
    assert sine_current[0] < 0
    expected = -2 * context.sin(context.mpf(1) / 20)
    assert context.almosteq(sine_current[0], expected)
    assert any(current != 0 for current in sine_current)


def test_uniform_child_input_has_exact_fine_and_collective_work_pairing(arithmetic):
    context, laplacian, inverse = arithmetic
    raw = tuple(Q((7 * node) % 19 - 9, 100) for node in range(45))
    mean = sum(raw) / 45
    form = tuple(value - mean for value in raw)
    offsets = tuple(Q((11 * node) % 17 - 8, 70) for node in range(45))
    state = _observe_organized_outer_winding(form=form, phase_offsets=offsets)
    means = tuple(sum(form[9 * child : 9 * child + 9]) / 9 for child in range(5))
    mean_gradient = _action(laplacian, means)
    direction = tuple(Q(int(i == 2)) - Q(1, 5) for i in range(5))
    for pattern in (direction, _action(inverse, direction)):
        fine = sum(
            value * pattern[node // 9] for node, value in enumerate(state.form_gradient)
        )
        collective = 9 * sum(a * b for a, b in zip(mean_gradient, pattern))
        assert fine == collective
    pulse = _pulse()
    gamma = 1 / (1023 * context.pi)
    a, d = _mp(context, pulse.amplitude), _mp(context, pulse.duration)
    u = _mp(context, pulse.elapsed) / d
    forcing = tuple(
        4 * a * _mp(context, inverse_value) * (6 - 12 * u) / (gamma * d**2)
        + a * _mp(context, direction_value) * (6 * u - 6 * u**2) / (gamma * d)
        for inverse_value, direction_value in zip(
            pulse.inverse_direction, pulse.direction
        )
    )
    phase = tuple(
        2 * context.pi * (node % 9 - 4) / 9 + _mp(context, value)
        for node, value in enumerate(offsets)
    )
    currents = [context.mpf(0)] * 45
    for left, right, weight in state.geometry.weighted_edges:
        current = _mp(context, weight) * context.sin(phase[right] - phase[left])
        currents[left] += current
        currents[right] -= current
    direct = context.mpf(0)
    power = context.mpf(0)
    loss = context.mpf(0)
    for node, exact_gradient in enumerate(state.form_gradient):
        gradient = _mp(context, exact_gradient)
        dx = (-gradient + gamma * currents[node]) / 4 + forcing[node // 9]
        phase_rate = gamma * gradient / 4
        direct += gradient * dx - currents[node] * phase_rate
        power += gradient * forcing[node // 9]
        loss += gradient**2 / 4
    assert context.almosteq(direct, -loss + power, rel_eps=context.mpf("1e-90"))
    assert loss > 0  # The arbitrary fine deviations are not silently projected out.


@pytest.mark.parametrize("parameter", ("elapsed", "duration", "amplitude"))
@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), "0.1", 1j))
def test_waveform_rejects_invalid_physical_scalar_before_materialization(
    parameter, value
):
    with pytest.raises((ValueError, TypeError)):
        _pulse(**{parameter: value})


@pytest.mark.parametrize("value", (True, -1, 5, Q(2), 2.0, "2"))
def test_waveform_requires_an_ordinary_marked_child(value):
    with pytest.raises((ValueError, TypeError)):
        _pulse(marked_child=value)


@pytest.mark.parametrize(
    "changes",
    ({"duration": Q(0)}, {"duration": Q(-1)}, {"elapsed": Q(-1)}, {"elapsed": Q(3)}),
)
def test_waveform_rejects_nonpositive_duration_or_time_outside_pulse(changes):
    with pytest.raises(ValueError):
        _pulse(**changes)


@pytest.mark.parametrize(
    "parameter",
    (
        "initial_form_bound",
        "initial_hidden_form",
        "initial_hidden_phase",
        "outer_gap",
        "amplitude",
        "duration",
    ),
)
@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), "0.1", 1j))
def test_bound_primitives_are_admitted_before_arithmetic(parameter, value):
    with pytest.raises((ValueError, TypeError)):
        _bounds(**{parameter: value})


@pytest.mark.parametrize(
    "parameter",
    ("initial_form_bound", "initial_hidden_form", "initial_hidden_phase", "outer_gap"),
)
def test_state_bounds_cannot_be_negative(parameter):
    with pytest.raises(ValueError):
        _bounds(**{parameter: Q(-1)})


def test_unsupported_coupled_comparison_domain_rejects():
    with pytest.raises(ValueError):
        _bounds(duration=Q(2000))


def test_detached_reports_are_immutable():
    with pytest.raises(FrozenInstanceError):
        _pulse().duration = Q(0)
    with pytest.raises(FrozenInstanceError):
        _bounds().duration = Q(0)


def test_input_enclosures_fit_uniform_bounds_and_zero_input_is_unforced():
    bounds = _bounds(duration=Q(7, 3))
    for position in (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(1)):
        pulse = _pulse(elapsed=position * bounds.duration)
        # The reader outward-rounds to dyadics; compare its endpoint with the
        # same outward representation of the exact analytical bound.
        assert (
            max(value.abs_max for value in pulse.form_offset_bounds)
            <= I(bounds.form_offset_upper).hi
        )
        assert (
            max(value.abs_max for value in pulse.form_input_bounds)
            <= I(bounds.input_upper).hi
        )
    zero = _pulse(amplitude=Q(0))
    assert all(value.lo == value.hi == 0 for value in zero.form_input_bounds)
    assert all(value.lo == value.hi == 0 for value in zero.form_offset_bounds)
    unforced = _bounds(amplitude=Q(0))
    assert unforced.unforced_form_comparison_upper == 0
    assert unforced.unforced_phase_comparison_upper == 0
    assert unforced.phase_residual_upper > 0  # Actual background motion remains.


def test_uniform_forcing_does_not_create_a_hidden_state_bound_from_exact_carrier():
    report = _bounds(initial_hidden_form=Q(0), initial_hidden_phase=Q(0))
    assert report.hidden.form_upper == report.hidden.phase_upper == 0
    assert report.form_offset_upper > 0
    assert report.input_upper > 0
    assert report.unforced_form_comparison_upper > 0


@pytest.fixture
def synthetic_sources():
    """Invented positive histories for wiring, never assessed scientific sources."""
    recovery = SimpleNamespace(
        recovery_certified=True,
        endpoint_radius=Q(1, 10**8),
        endpoint_form_margin=Q(1, 10**9),
        endpoint_phase_margin=Q(1, 10**9),
    )
    outer = SimpleNamespace(
        geometry=_derive_organized_outer_winding(),
        primitive_entry=SimpleNamespace(
            endpoint_form_bounds=(I(-Q(1, 1000), Q(1, 1000)),) * 5,
            capture=SimpleNamespace(phase_edge_gap_bounds=(I(Q(1), Q(6, 5)),) * 5),
        ),
        full_form_error_upper_bound=Q(1, 10**8),
        full_phase_error_upper_bound=Q(2, 10**8),
        recovered_child_norm_bound=Q(1, 10**8),
        primitive_law_and_clock_certified=True,
        fine_error_certified=True,
        child_winding_retained_during_transit=True,
        outer_cell_certified=True,
        conditional_capture_certified=True,
        control_conditional_capture_certified=True,
        attachment_budget_certified=True,
        capture_margin=Q(1, 100),
        control_capture_margin=Q(1, 20),
        attachment_work_margin=Q(100),
        initial_storage_margin=Q(200),
    )
    return recovery, outer


@pytest.fixture
def assess(synthetic_sources, monkeypatch):
    def run(*, change=None):
        recovery, outer = deepcopy(synthetic_sources)
        if change is not None:
            change(recovery, outer)
        calls = []

        def recover(**arguments):
            calls.append(("recovery", arguments))
            return recovery

        def acquire(**arguments):
            calls.append(("outer", arguments))
            return outer

        monkeypatch.setattr(owner, "_assess_sine_recovery_resources", recover)
        monkeypatch.setattr(owner, "_derive_organized_outer_resource_bounds", acquire)
        report = owner._assess_organized_electrical_actuation()
        return report, calls

    return run


def test_source_consumer_rebuilds_only_precontact_history_and_keeps_prefix(assess):
    report, calls = assess()
    assert report.admitted and not report.unresolved_conditions
    assert [name for name, _ in calls] == ["recovery", "outer"]
    assert calls[0][1]["endpoint_radius"] == calls[1][1]["recovered_radius"]
    assert report.independent_child_count == 5
    assert report.prefix_duration == Q(1, 10**5)
    assert report.prefix_hidden.initial_form == report.outer.full_form_error_upper_bound
    assert (
        report.prefix_hidden.initial_phase == report.outer.full_phase_error_upper_bound
    )
    assert report.pulse.initial_hidden_form == report.prefix_hidden.form_upper
    assert report.pulse.initial_hidden_phase == report.prefix_hidden.phase_upper
    assert report.entry_form_upper > report.nominal_form_upper
    # No sequential observation or donor/contact report exists in this fixture.
    assert not hasattr(report.outer, "observation")
    assert all("contact" not in name for name, _ in report.prerequisite_margins)


@pytest.mark.parametrize("truthy", (False, 1))
@pytest.mark.parametrize(
    "field",
    (
        "recovery_certified",
        "primitive_law_and_clock_certified",
        "fine_error_certified",
        "child_winding_retained_during_transit",
        "outer_cell_certified",
        "conditional_capture_certified",
        "control_conditional_capture_certified",
        "attachment_budget_certified",
    ),
)
def test_source_flags_require_true_not_truthiness(assess, field, truthy):
    def change(recovery, outer):
        setattr(recovery if field == "recovery_certified" else outer, field, truthy)

    report, _ = assess(change=change)
    assert not report.admitted and report.unresolved_conditions


@pytest.mark.parametrize("value", (Q(0), Q(-1), None))
@pytest.mark.parametrize(
    "field",
    (
        "endpoint_form_margin",
        "endpoint_phase_margin",
        "capture_margin",
        "control_capture_margin",
        "attachment_work_margin",
        "initial_storage_margin",
    ),
)
def test_every_source_margin_is_present_and_strictly_positive(assess, field, value):
    def change(recovery, outer):
        setattr(recovery if field.startswith("endpoint_") else outer, field, value)

    report, _ = assess(change=change)
    assert not report.admitted and report.unresolved_conditions


def test_boolean_source_margin_cannot_pass_as_positive_evidence(assess):
    with pytest.raises((ValueError, TypeError)):
        assess(
            change=lambda recovery, outer: setattr(
                recovery, "endpoint_form_margin", True
            )
        )


@pytest.mark.parametrize("branch", ("forms", "gaps"))
@pytest.mark.parametrize("length", (0, 4, 6))
def test_primitive_source_bound_coordinates_require_all_five_children(
    assess, branch, length
):
    def change(recovery, outer):
        boxes = (I(Q(1, 100)),) * length
        if branch == "forms":
            outer.primitive_entry.endpoint_form_bounds = boxes
        else:
            outer.primitive_entry.capture.phase_edge_gap_bounds = boxes

    with pytest.raises((ValueError, TypeError)):
        assess(change=change)


@pytest.mark.parametrize("branch", ("forms", "gaps"))
def test_cached_abs_max_without_primitive_interval_is_not_admitted(assess, branch):
    def change(recovery, outer):
        boxes = (SimpleNamespace(abs_max=Q(1, 100)),) * 5
        if branch == "forms":
            outer.primitive_entry.endpoint_form_bounds = boxes
        else:
            outer.primitive_entry.capture.phase_edge_gap_bounds = boxes

    with pytest.raises((ValueError, TypeError)):
        assess(change=change)


@pytest.mark.parametrize("branch", ("recovery", "outer"))
def test_recovered_allowance_cannot_change_between_history_stages(assess, branch):
    def change(recovery, outer):
        if branch == "recovery":
            recovery.endpoint_radius = Q(1, 10**7)
        else:
            outer.recovered_child_norm_bound = Q(1, 10**7)

    report, _ = assess(change=change)
    assert not report.admitted
    assert "recovered_radius_matches" in report.unresolved_conditions


@pytest.mark.parametrize(
    "field,value",
    (
        ("clock_factor", Q(1, 2)),
        ("phase_domain", "acute"),
        ("held_capacities", (Q(2),) * 45),
    ),
)
def test_source_geometry_must_match_actual_complete_law(assess, field, value):
    def change(recovery, outer):
        outer.geometry = replace(outer.geometry, **{field: value})

    report, _ = assess(change=change)
    assert not report.admitted
    assert "complete_quarter_law" in report.unresolved_conditions


def test_missing_edge_cannot_be_replaced_by_a_cached_source_verdict(assess):
    def change(recovery, outer):
        outer.geometry = replace(
            outer.geometry, weighted_edges=outer.geometry.weighted_edges[:-1]
        )

    report, _ = assess(change=change)
    assert not report.admitted
    assert "precontact_product_support" in report.unresolved_conditions

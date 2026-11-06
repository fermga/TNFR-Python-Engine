"""Static contracts for independently evolved child-core error allowances."""

import math
from copy import deepcopy
from fractions import Fraction as Q

import mpmath
import pytest

from tests.physics.test_organized_source_child_relaxation import (
    SWAP,
    UNIT,
    ZERO,
    H,
    _contains,
    _fields,
    _mp,
    _state,
)
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_source_child_tracking as owner
from tnfr.physics._organized_source_child_quotient import (
    _restrict_organized_source_child_quotient,
)


def _bound(state=None, elapsed_time=Q(0), **overrides):
    form, offsets, capacities = _state() if state is None else state
    return owner._bound_organized_source_child_tracking(
        **{
            "form": form,
            "phase_offsets": offsets,
            "capacities": capacities,
            "elapsed_time": elapsed_time,
            **overrides,
        }
    )


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Child tracking controls cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module")
def initial():
    return _bound()


@pytest.fixture(scope="module")
def deadline(initial):
    return _bound(elapsed_time=10 / initial.error_decay_rate_lower)


def test_frozen_projected_initial_error_is_zero_but_its_form_rate_is_not(initial, mp):
    assert initial.certified and initial.reference_state.form == ZERO
    assert initial.reference_state.phase_offsets == ZERO
    assert initial.pointwise_metric_upper_bound == 0
    assert initial.pointwise_squared_error_upper_bound == 0
    assert initial.convolution_upper_bound == 0
    assert initial.error_decay_bounds == initial.forcing_decay_bounds == (Q(1), Q(1))
    fine_form, fine_phase, _ = _fields(mp, *_state())
    core_form, core_phase, _ = _fields(mp, ZERO, ZERO, UNIT)
    defect = (fine_form[9] + fine_form[18]) / 2 - core_form[9]
    gamma, alpha, h = 1 / (1023 * mp.pi), 2 * mp.pi / 9, _mp(mp, H)
    exact_defect = gamma * mp.sin(alpha) * (mp.cos(h) - 1) / 4
    assert mp.almosteq(defect, exact_defect)
    assert defect < 0
    _contains(mp, initial.initial_relaxation.orbit.form_tau_rate_bounds[9], defect)
    assert initial.initial_relaxation.orbit.form_tau_rate_bounds[9].hi < 0
    assert max(abs(value) for value in core_form + core_phase) < mp.mpf("1e-95")
    assert max(
        abs((fine_phase[i] + fine_phase[SWAP[i]]) / 2 - core_phase[i])
        for i in range(45)
    ) < mp.mpf("1e-95")


def test_frozen_uniform_and_pointwise_allowances_pass_without_response_evaluation(
    initial, deadline, mp
):
    assert initial.domain_certified and deadline.certified
    assert initial.full_gap_lower_bound == Q(4, 15)
    gamma = initial.initial_relaxation.gamma_bounds
    expected_rate = gamma.lo**2 / (60 * (1 + gamma.hi))
    assert initial.error_decay_rate_lower == expected_rate
    assert initial.hidden_decay_rate_lower == 15 * expected_rate
    assert initial.forcing_decay_rate == 30 * expected_rate
    assert initial.rate_separation == 29 * expected_rate
    assert 0 < initial.uniform_squared_error_upper_bound < H**2 / 128
    assert deadline.error_exponent == 10 and deadline.forcing_exponent == 300
    assert 0 < deadline.pointwise_squared_error_upper_bound < H**2 / 2**20
    assert (
        deadline.uniform_squared_error_upper_bound
        == initial.uniform_squared_error_upper_bound
    )
    # Independent analytic solution of the scalar forced inequality.
    convolution = (mp.exp(-10) - mp.exp(-300)) / (29 * _mp(mp, expected_rate))
    assert convolution <= _mp(mp, deadline.convolution_upper_bound)
    assert _mp(mp, deadline.convolution_upper_bound) - convolution < mp.mpf("1e-20")
    expected = (
        _mp(mp, deadline.forcing_coefficient)
        * convolution
        / _mp(mp, deadline.metric_lower)
    )
    assert expected <= _mp(mp, deadline.pointwise_squared_error_upper_bound)
    assert initial.metric_lower == 1 - gamma.hi
    assert initial.metric_upper == 1 + gamma.hi


def test_fixed_half_amplitude_has_exact_quartic_squared_allowances(initial, deadline):
    state = _state()
    half = tuple(tuple(x / 2 for x in row) for row in state[:2]) + (UNIT,)
    report = _bound(half, elapsed_time=deadline.elapsed_time)
    assert report.certified
    assert (
        report.hidden_norm_squared_coefficient
        == initial.hidden_norm_squared_coefficient / 4
    )
    assert report.forcing_coefficient == initial.forcing_coefficient / 16
    assert (
        report.uniform_squared_error_upper_bound
        == initial.uniform_squared_error_upper_bound / 16
    )
    assert (
        report.pointwise_squared_error_upper_bound
        == deadline.pointwise_squared_error_upper_bound / 16
    )


def test_reference_is_exact_projection_on_both_means_and_retains_orbit_masses(mp):
    form, offsets, capacities = _state()
    form = tuple(x + Q((7 * i) % 13 - 6, 8192) + Q(7, 3) for i, x in enumerate(form))
    offsets = tuple(
        x + Q((3 * i) % 7 - 3, 8192) + Q(5, 7) for i, x in enumerate(offsets)
    )
    report = _bound((form, offsets, capacities))
    projected = tuple(
        tuple((row[i] + row[SWAP[i]]) / 2 for i in range(45)) for row in (form, offsets)
    )
    assert report.certified and report.matched_means
    assert report.reference_state.form == projected[0]
    assert report.reference_state.phase_offsets == projected[1]
    assert report.initial_form_total == report.reference_form_total == sum(form, Q(0))
    assert (
        report.initial_phase_offset_total
        == report.reference_phase_offset_total
        == sum(offsets, Q(0))
    )
    reference = report.reference_state
    assert reference == _restrict_organized_source_child_quotient(
        form=projected[0], phase_offsets=projected[1], capacities=UNIT
    )
    for fine, coarse in (
        (reference.form, reference.form_coordinates),
        (reference.phase_offsets, reference.phase_offset_coordinates),
    ):
        assert sum((x**2 for x in fine), Q(0)) == sum(
            (mass * x**2 for mass, x in zip(reference.geometry.orbit_masses, coarse)),
            Q(0),
        )
        assert sum((x**2 for x in fine), Q(0)) != sum((x**2 for x in coarse), Q(0))
    dx, dphase, energy = _fields(mp, *projected, UNIT)
    _contains(mp, report.reference_excess_bounds, energy)
    for expected, intervals in (
        (dx, report.reference_observation.form_tau_rate_bounds),
        (dphase, report.reference_observation.phase_tau_rate_bounds),
    ):
        for value, interval in zip(expected, intervals):
            _contains(mp, interval, value)


def test_symmetry_tiny_hidden_data_and_tiny_time_preserve_exact_scaling(
    initial, deadline
):
    zero = _bound((ZERO, ZERO, UNIT), elapsed_time=deadline.elapsed_time)
    assert zero.certified
    assert zero.forcing_coefficient == zero.uniform_squared_error_upper_bound == 0
    assert zero.pointwise_squared_error_upper_bound == 0
    scale = Q(1, 2**600)
    state = _state()
    tiny = _bound(
        (tuple(scale * x for x in state[0]), tuple(scale * x for x in state[1]), UNIT),
        elapsed_time=deadline.elapsed_time,
    )
    assert tiny.certified
    assert (
        tiny.uniform_squared_error_upper_bound
        == scale**4 * initial.uniform_squared_error_upper_bound
        > 0
    )
    assert (
        tiny.pointwise_squared_error_upper_bound
        == scale**4 * deadline.pointwise_squared_error_upper_bound
        > 0
    )
    short = _bound(elapsed_time=scale)
    assert short.certified
    assert 0 < short.convolution_upper_bound <= scale
    assert (
        0
        < short.pointwise_squared_error_upper_bound
        < initial.uniform_squared_error_upper_bound
    )


def test_global_swap_and_common_offsets_preserve_comparison_bounds(initial):
    state = _state()
    exchanged = _bound(tuple(tuple(row[j] for j in SWAP) for row in state))
    shifted = _bound(
        form=tuple(x + Q(9, 7) for x in state[0]),
        phase_offsets=tuple(x + Q(11, 5) for x in state[1]),
    )
    for report in (exchanged, shifted):
        assert report.certified and report.matched_means
        assert (
            report.hidden_norm_squared_coefficient
            == initial.hidden_norm_squared_coefficient
        )
        assert report.forcing_coefficient == initial.forcing_coefficient
        assert (
            report.uniform_squared_error_upper_bound
            == initial.uniform_squared_error_upper_bound
        )
    assert shifted.reference_state.form == (Q(9, 7),) * 45
    assert shifted.reference_state.phase_offsets == (Q(11, 5),) * 45


def test_work_limit_preserves_uniform_information_and_skips_exponent_evaluation(
    initial, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported pointwise exponent reached the exponential evaluator")

    monkeypatch.setattr(owner, "_negative_exp_bounds", forbidden)
    report = _bound(elapsed_time=Q(4097) / initial.forcing_decay_rate)
    assert report.domain_certified and not report.exponential_work_admitted
    assert not report.certified and report.forcing_exponent == 4097
    assert report.uniform_metric_upper_bound == initial.uniform_metric_upper_bound
    assert (
        report.uniform_squared_error_upper_bound
        == initial.uniform_squared_error_upper_bound
    )
    assert report.error_decay_bounds is report.forcing_decay_bounds is None
    assert report.convolution_upper_bound is report.pointwise_metric_upper_bound is None
    assert report.pointwise_squared_error_upper_bound is None
    assert "negative_exponential_work_limit_unavailable" in report.unavailable_reasons


@pytest.mark.parametrize("failure", ["wrong_lift", "energy_boundary"])
def test_unadmitted_source_cannot_borrow_reference_domain(failure):
    if failure == "wrong_lift":
        offsets = list(_state()[1])
        offsets[9] += Q(2 * math.pi)
        report = _bound(phase_offsets=offsets)
        assert report.initial_relaxation.excess_bounds.hi < Q(1, 100)
        assert not report.initial_relaxation.in_acute_cell
    else:
        state = _state((Q(1, 20),) + (Q(0),) * 8, (Q(0),) * 9)
        report = _bound(state)
        assert report.reference_in_acute_cell and report.reference_below_energy_ceiling
        assert not report.initial_relaxation.below_energy_ceiling
    assert not report.domain_certified and not report.certified
    assert report.uniform_squared_error_upper_bound is None
    assert report.pointwise_squared_error_upper_bound is None
    assert report.unavailable_reasons


@pytest.mark.parametrize(
    "field,value",
    [
        ("elapsed_time", True),
        ("elapsed_time", math.nan),
        ("elapsed_time", Q(-1)),
        ("form", True),
        ("phase_offsets", math.inf),
        ("capacities", True),
        ("capacities", Q(1, 2)),
        ("capacities", Q(1) + Q(1, 2**600)),
    ],
)
def test_invalid_or_nonunit_primitives_reject_before_geometric_readers(
    field, value, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid primitives reached a geometric relaxation reader")

    monkeypatch.setattr(owner, "_bound_organized_source_child_relaxation", forbidden)
    if field != "elapsed_time":
        row = list(dict(zip(("form", "phase_offsets", "capacities"), _state()))[field])
        row[9] = value
        value = row
    with pytest.raises((TypeError, ValueError)):
        _bound(**{field: value})


def test_readers_rebuild_from_normalized_primitives_and_do_not_mutate_inputs(
    monkeypatch,
):
    state = tuple(list(row) for row in _state())
    before = deepcopy(state)
    calls = []
    relaxation = owner._bound_organized_source_child_relaxation
    quotient = owner._lift_organized_source_child_quotient

    def record_relaxation(**arguments):
        calls.append(("relaxation", arguments))
        return relaxation(**arguments)

    def record_quotient(**arguments):
        calls.append(("quotient", arguments))
        return quotient(**arguments)

    monkeypatch.setattr(
        owner, "_bound_organized_source_child_relaxation", record_relaxation
    )
    monkeypatch.setattr(owner, "_lift_organized_source_child_quotient", record_quotient)
    report = _bound(state, elapsed_time=0.1)
    assert report.certified and report.elapsed_time == Q(0.1) != Q(1, 10)
    assert state == before
    assert calls[0][0] == "relaxation" and calls[0][1]["elapsed_time"] == 0
    assert all(type(x) is Q for x in calls[0][1]["form"])
    assert calls[1][0] == "quotient"
    assert calls[1][1]["form_coordinates"] == report.reference_state.form_coordinates

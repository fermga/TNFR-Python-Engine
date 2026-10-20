"""Frozen second-order target response from opposed fine phase profiles.

Only the two declared zero-form states evaluate the research discriminator.
The separate nonzero-form contract control reuses the existing source-port
test coordinates to exercise the complete shared acceleration kernel. No
trajectory, acquired-source reconstruction, hidden-state closure or operator
realization is asserted. Individually marked interfaces are not in the
matched target observation, although all remain in the complete law.
"""

from fractions import Fraction as Q

import mpmath
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as owner
from tnfr.physics.relational_sine_entry import _unit_laplacian_action

ORIGIN = Q(1, 40)
EPSILON = Q(1, 100)
PROFILE = (Q(1), Q(-1)) + (Q(0),) * 7
ZERO = (Q(0),) * 45


@pytest.fixture(scope="module", autouse=True)
def no_response_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Fine response controls cannot execute or rebuild a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (capture, "_assess_organized_source_capture"),
            (capture, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def geometry():
    return owner._derive_organized_source_port()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, expected):
    assert _mp(mp, bounds.lo) <= expected <= _mp(mp, bounds.hi)


@pytest.fixture(scope="module")
def prepared_offsets(geometry):
    control = (ORIGIN,) * 45
    perturbed = tuple(
        ORIGIN
        + (
            EPSILON * PROFILE[slot]
            if child == 1
            else -EPSILON * PROFILE[slot] if child == 2 else 0
        )
        for child in range(5)
        for slot in range(9)
    )
    # Admission precedes either response evaluation. Both supports retain all
    # reacting coordinates, unit capacities and the quarter factor in both rows.
    assert geometry.node_order == tuple(range(45))
    assert len(geometry.weighted_edges) == 90
    assert all(weight == 1 for _, _, weight in geometry.weighted_edges)
    assert geometry.held_capacities == (Q(1),) * 45
    assert geometry.clock_factor == Q(1, 4)
    assert sum(ZERO) == 0
    alpha, half_pi = geometry.alpha_bounds, pi_interval() / 2
    for offsets in (control, perturbed):
        assert sum(offsets) == Q(9, 8)
        for child in range(5):
            values = offsets[9 * child : 9 * child + 9]
            assert sum(values) / 9 == ORIGIN
            differences = tuple(
                values[(slot + 1) % 9] - values[slot] for slot in range(9)
            )
            assert sum(differences) == 0
            for difference in differences:
                angle = alpha + difference
                assert 0 < angle.lo <= angle.hi < half_pi.lo
            # The fixed closing turn gives sum angles=9*alpha=2*pi exactly.
        for left, right, _ in geometry.weighted_edges:
            if left // 9 != right // 9:
                assert abs(offsets[right] - offsets[left]) < half_pi.lo
    return control, perturbed


@pytest.fixture(scope="module")
def observations(prepared_offsets):
    return tuple(
        owner._observe_organized_source_acceleration(form=ZERO, phase_offsets=offsets)
        for offsets in prepared_offsets
    )


def test_opposed_profile_identity_and_odd_cycle_zero_response_condition():
    alpha, forward, previous = sp.symbols("alpha forward previous", real=True)
    paired_current = (
        sp.sin(alpha + forward)
        + sp.sin(-alpha - previous)
        + sp.sin(alpha - forward)
        + sp.sin(-alpha + previous)
    )
    prediction = 2 * sp.sin(alpha) * (sp.cos(forward) - sp.cos(previous))
    assert sp.simplify(sp.expand_trig(paired_current - prediction)) == 0
    # The target response is the cyclic first difference of cos(d_j).
    # Its kernel is precisely constant cosines, and its mean always vanishes.
    cyclic_difference = sp.zeros(9)
    for row in range(9):
        cyclic_difference[row, row] = 1
        cyclic_difference[row, (row - 1) % 9] = -1
    assert sp.ones(1, 9) * cyclic_difference == sp.zeros(1, 9)
    assert cyclic_difference.nullspace() == [sp.ones(9, 1)]
    # When |d_j|<pi, cosine injectivity in |d_j| makes zero response require
    # a common magnitude d. If d>0, an odd count of signed increments cannot
    # telescope to zero: these ten counts cover every nine-sign assignment.
    signed_totals = tuple(2 * positive_count - 9 for positive_count in range(10))
    assert 0 not in signed_totals
    assert all(total % 2 for total in signed_totals)
    # Thus this exact family has no nonuniform zero-response profile on C9
    # in that domain. This is an algebraic corollary, not a sampled campaign.


def test_frozen_preparations_match_target_state_and_all_initial_net_rows(
    observations, mp
):
    control, perturbed = (report.state for report in observations)
    assert control.form == perturbed.form == ZERO
    assert control.phase_offsets[:9] == perturbed.phase_offsets[:9] == (ORIGIN,) * 9
    assert control.phase_lift_bounds[:9] == perturbed.phase_lift_bounds[:9]
    for state in (control, perturbed):
        assert state.conserved_form_total == 0
        assert state.conserved_phase_total == Q(1, 8)
        assert state.form_gradient == ZERO
        assert state.child_phase_offset_means == (ORIGIN,) * 5
        assert all(rate == I(0) for rate in state.phase_tau_rate_bounds)
        for node in range(9):
            assert state.phase_current_bounds[node].contains(0)
            assert state.form_tau_rate_bounds[node].contains(0)
            assert state.form_tau_rate_bounds[node].width < Q(1, 10**30)
        # Primitive slot-zero-minus-slot-one phase is -alpha, not zero.
        initial_contrast = state.phase_lift_bounds[0] - state.phase_lift_bounds[1]
        _contains(mp, initial_contrast, -2 * mp.pi / 9)
    # The marked child1/child2 interface currents already differ from control;
    # only their net contribution is matched. They are retained in the law.
    assert control.phase_offsets[9] - control.phase_offsets[0] == 0
    first = sin(I(perturbed.phase_offsets[9] - perturbed.phase_offsets[0]))
    second = sin(I(perturbed.phase_offsets[18] - perturbed.phase_offsets[0]))
    assert first.lo > 0 and second.hi < 0
    assert (first + second).contains(0)


def test_full_second_derivatives_resolve_a_contrast_hidden_from_the_target_means(
    observations, mp
):
    control, perturbed = observations
    state, geometry = perturbed.state, perturbed.state.geometry
    alpha, gamma, epsilon = 2 * mp.pi / 9, 1 / (1023 * mp.pi), _mp(mp, EPSILON)
    difference = mp.cos(epsilon) - mp.cos(2 * epsilon)
    edge = 1 - mp.cos(epsilon)
    profile = (-difference, difference, edge, 0, 0, 0, 0, 0, -edge)
    d = cos(I(EPSILON)) - cos(I(2 * EPSILON))
    e = 1 - cos(I(EPSILON))
    assert d.lo > 0 and e.lo > 0
    signs = (-1, 1, 1, 0, 0, 0, 0, 0, -1)
    gaps = tuple(EPSILON * (PROFILE[(j + 1) % 9] - PROFILE[j]) for j in range(9))
    assert all(abs(value) < pi_interval().lo for value in gaps)
    for slot in range(9):
        raw_pair_current = (
            state.phase_current_bounds[9 + slot] + state.phase_current_bounds[18 + slot]
        )
        independent_pair = 2 * mp.sin(alpha) * profile[slot]
        _contains(mp, raw_pair_current, independent_pair)
        cycle_prediction = (
            2
            * sin(geometry.alpha_bounds)
            * (cos(I(gaps[slot])) - cos(I(gaps[(slot - 1) % 9])))
        )
        _contains(mp, cycle_prediction, independent_pair)
        expected_form = gamma * mp.sin(alpha) * profile[slot] / 8
        expected_phase = -gamma * expected_form
        _contains(mp, perturbed.form_tau_acceleration_bounds[slot], expected_form)
        _contains(mp, perturbed.phase_tau_acceleration_bounds[slot], expected_phase)
        assert control.form_tau_acceleration_bounds[slot].contains(0)
        assert control.phase_tau_acceleration_bounds[slot].contains(0)
        if signs[slot] > 0:
            assert perturbed.form_tau_acceleration_bounds[slot].lo > 0
            assert perturbed.phase_tau_acceleration_bounds[slot].hi < 0
        elif signs[slot] < 0:
            assert perturbed.form_tau_acceleration_bounds[slot].hi < 0
            assert perturbed.phase_tau_acceleration_bounds[slot].lo > 0
        else:
            assert perturbed.form_tau_acceleration_bounds[slot].contains(0)
            assert perturbed.phase_tau_acceleration_bounds[slot].contains(0)
    form_contrast = (
        perturbed.form_tau_acceleration_bounds[0]
        - perturbed.form_tau_acceleration_bounds[1]
    )
    phase_contrast = (
        perturbed.phase_tau_acceleration_bounds[0]
        - perturbed.phase_tau_acceleration_bounds[1]
    )
    assert form_contrast.hi < 0 and phase_contrast.lo > 0
    _contains(mp, form_contrast, -gamma * mp.sin(alpha) * difference / 4)
    _contains(mp, phase_contrast, gamma**2 * mp.sin(alpha) * difference / 4)
    assert perturbed.child_form_tau_acceleration_bounds[0].contains(0)
    assert perturbed.child_phase_tau_acceleration_bounds[0].contains(0)
    assert perturbed.child_form_tau_acceleration_bounds[0].width < Q(1, 10**30)
    assert perturbed.child_phase_tau_acceleration_bounds[0].width < Q(1, 10**30)
    assert all(value == I(0) for value in perturbed.phase_current_tau_rate_bounds)
    # This frozen profile also has a symmetric value multiset, so its sine
    # values cancel. Zero mean alone would not give this stronger property.
    assert sorted(PROFILE) == sorted(-value for value in PROFILE)
    for child in range(5):
        assert state.child_form_tau_rate_bounds[child].contains(0)
        assert state.child_phase_tau_rate_bounds[child] == I(0)
        assert perturbed.child_form_tau_acceleration_bounds[child].contains(0)
        assert perturbed.child_phase_tau_acceleration_bounds[child].contains(0)


def test_actual_full_tangent_preserves_the_invisible_child_antisymmetric_subspace(
    geometry,
):
    edges = tuple((left, right) for left, right, _ in geometry.weighted_edges)
    cycles = tuple(edge for edge in edges if edge[0] // 9 == edge[1] // 9)
    matching = tuple(edge for edge in edges if edge[0] // 9 != edge[1] // 9)
    swap = tuple(
        9 * (2 if child == 1 else 1 if child == 2 else child) + slot
        for child in range(5)
        for slot in range(9)
    )
    for support in (cycles, matching):
        assert {
            tuple(sorted((swap[left], swap[right]))) for left, right in support
        } == {tuple(sorted(edge)) for edge in support}
    # At equilibrium H=cos(alpha)*(I x L_C9)+(K x I), while
    # L=(I x L_C9)+(K x I). Check a basis of the entire nine-slot odd space,
    # rather than only the frozen profile, using the actual admitted support.
    for slot in range(9):
        basis = tuple(Q(index == slot) for index in range(9))
        odd = (Q(0),) * 9 + basis + tuple(-value for value in basis) + (Q(0),) * 18
        cycle_profile = tuple(
            2 * basis[j] - basis[(j - 1) % 9] - basis[(j + 1) % 9] for j in range(9)
        )
        expected_cycle = (
            (Q(0),) * 9
            + cycle_profile
            + tuple(-value for value in cycle_profile)
            + (Q(0),) * 18
        )
        actual_cycle = _unit_laplacian_action(odd, cycles)
        actual_matching = _unit_laplacian_action(odd, matching)
        assert actual_cycle == expected_cycle
        assert actual_matching == tuple(2 * value for value in odd)
        for image in (actual_cycle, actual_matching):
            assert _unit_laplacian_action(image, edges)[:9] == (Q(0),) * 9
    assert 0 < cos(geometry.alpha_bounds).lo < 1
    # Both inputs of A=(1/4)[[-L,-gamma*H],[gamma*L,0]] stay in this odd
    # space. Target perturbations vanish for its entire linearized flow;
    # the second-order nonlinear contrast above is absent from that model.


def _independent_acceleration(mp, form, offsets, geometry):
    x = tuple(_mp(mp, value) for value in form)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value) for i, value in enumerate(offsets)
    )
    gradient, current = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
    for left, right, weight in geometry.weighted_edges:
        w = _mp(mp, weight)
        difference, sine = w * (x[left] - x[right]), w * mp.sin(
            theta[right] - theta[left]
        )
        gradient[left] += difference
        gradient[right] -= difference
        current[left] += sine
        current[right] -= sine
    gamma = 1 / (1023 * mp.pi)
    form_rate = tuple((-q + gamma * s) / 4 for q, s in zip(gradient, current))
    phase_rate = tuple(gamma * q / 4 for q in gradient)
    gradient_rate, current_rate = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
    for left, right, weight in geometry.weighted_edges:
        w = _mp(mp, weight)
        difference = w * (form_rate[left] - form_rate[right])
        sine_rate = (
            w
            * mp.cos(theta[right] - theta[left])
            * (phase_rate[right] - phase_rate[left])
        )
        gradient_rate[left] += difference
        gradient_rate[right] -= difference
        current_rate[left] += sine_rate
        current_rate[right] -= sine_rate
    return (
        tuple((-q + gamma * s) / 4 for q, s in zip(gradient_rate, current_rate)),
        tuple(gamma * q / 4 for q in gradient_rate),
        tuple(gradient_rate),
        tuple(current_rate),
    )


def test_shared_acceleration_retains_cosine_feedback_and_squared_clock_changes(
    geometry, mp
):
    # These are the existing general source-port contract coordinates, not a
    # third preparation in the frozen zero-form research comparison.
    form = tuple(Q((7 * i) % 19 - 9, 64) for i in range(45))
    offsets = tuple(Q((11 * i) % 23 - 11, 256) for i in range(45))
    report = owner._observe_organized_source_acceleration(
        form=form, phase_offsets=offsets
    )
    expected_form, expected_phase, gradient_rate, current_rate = (
        _independent_acceleration(mp, form, offsets, geometry)
    )
    assert report.state.form == form and report.state.phase_offsets == offsets
    for bounds, expected in zip(report.gradient_tau_rate_bounds, gradient_rate):
        _contains(mp, bounds, expected)
    for bounds, expected in zip(report.phase_current_tau_rate_bounds, current_rate):
        _contains(mp, bounds, expected)
    assert any(
        not bounds.contains(0) for bounds in report.phase_current_tau_rate_bounds
    )
    gamma = 1 / (1023 * mp.pi)
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("original", mp.mpf(1023) / 1024),
        ("slow", 2 / gamma**2),
    ):
        for channel, expected in (("form", expected_form), ("phase", expected_phase)):
            bounds = getattr(report, f"{channel}_{clock}_acceleration_bounds")
            for value, prediction in zip(bounds, expected):
                _contains(mp, value, factor**2 * prediction)
    for child in range(5):
        start = 9 * child
        _contains(
            mp,
            report.child_form_tau_acceleration_bounds[child],
            sum(expected_form[start : start + 9]) / 9,
        )
        _contains(
            mp,
            report.child_phase_tau_acceleration_bounds[child],
            sum(expected_phase[start : start + 9]) / 9,
        )


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, float("inf")))
def test_acceleration_rejects_invalid_raw_coordinates_before_geometry(
    field, bad, monkeypatch
):
    def forbidden():
        pytest.fail("Invalid raw coordinates reached geometry construction")

    monkeypatch.setattr(owner, "_derive_organized_source_port", forbidden)
    supplied = {"form": ZERO, "phase_offsets": (ORIGIN,) * 45}
    supplied[field] = supplied[field][:-1] + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_source_acceleration(**supplied)

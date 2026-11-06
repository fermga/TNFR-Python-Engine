"""Exact signed-reflection representation of the unchanged 45-node law.

The chart retains twenty form and twenty offset coordinates, reconstructing
the complete signed state. Its symmetry is sufficient for continuous mean
preservation, not a bare twenty-node law, a winding certificate or an operator
selector. Static field controls install no trajectory or source preparation.
"""

from fractions import Fraction as Q

import mpmath
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as owner

ORIGIN = Q(1, 40)
REPRESENTATIVES = (0, 2, 3, 4)
ZERO = (Q(0),) * 20
GENERIC_FORM = tuple(Q((7 * i) % 17 - 8, 128) for i in range(20))
GENERIC_PHASE = tuple(Q((11 * i) % 19 - 9, 256) for i in range(20))
DELAYED_PHASE = tuple(
    Q(1, 100) if i == 4 else -Q(1, 100) if i == 8 else Q(0) for i in range(20)
)
CASES = {
    "equilibrium": (ZERO, ZERO),
    "delayed": (ZERO, DELAYED_PHASE),
    "general": (GENERIC_FORM, GENERIC_PHASE),
}


@pytest.fixture(scope="module", autouse=True)
def no_response_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Reflection controls cannot execute or rebuild a source")

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


@pytest.fixture(scope="module")
def observations():
    return {
        name: owner._observe_organized_source_reflection(
            form_coordinates=form, phase_coordinates=phase
        )
        for name, (form, phase) in CASES.items()
    }


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, expected):
    assert _mp(mp, bounds.lo) <= expected <= _mp(mp, bounds.hi)


def _independent_lift(forms, phases):
    full_form, offsets = [Q(0)] * 45, [ORIGIN] * 45
    for child in range(5):
        for column, slot in enumerate(REPRESENTATIVES):
            coordinate = 4 * child + column
            partner = (1 - slot) % 9
            full_form[9 * child + slot] = forms[coordinate]
            full_form[9 * child + partner] = -forms[coordinate]
            offsets[9 * child + slot] += phases[coordinate]
            offsets[9 * child + partner] -= phases[coordinate]
    return tuple(full_form), tuple(offsets)


def test_exact_chart_metric_and_internal_laplacian_come_from_the_actual_cycle(geometry):
    embedding = sp.zeros(9, 4)
    for column, slot in enumerate(REPRESENTATIVES):
        embedding[slot, column] = 1
        embedding[(1 - slot) % 9, column] = -1
    projection = sp.eye(9)[REPRESENTATIVES, :]
    assert projection * embedding == sp.eye(4)
    assert embedding.T * embedding == 2 * sp.eye(4)
    assert sp.ones(1, 9) * embedding == sp.zeros(1, 4)
    assert embedding[5, :] == sp.zeros(1, 4)
    laplacian = sp.zeros(9)
    for left, right, weight in geometry.weighted_edges:
        if left < 9 and right < 9:
            assert weight == 1
            laplacian[left, left] += 1
            laplacian[right, right] += 1
            laplacian[left, right] -= 1
            laplacian[right, left] -= 1
    reduced = projection * laplacian * embedding
    assert reduced == sp.Matrix(
        ((3, 1, 0, 0), (1, 2, -1, 0), (0, -1, 2, -1), (0, 0, -1, 2))
    )
    assert laplacian * embedding == embedding * reduced
    child_laplacian, fine_laplacian = sp.zeros(5), sp.zeros(45)
    for left, right in geometry.collective_edges:
        child_laplacian[left, left] += 1
        child_laplacian[right, right] += 1
        child_laplacian[left, right] -= 1
        child_laplacian[right, left] -= 1
    for left, right, weight in geometry.weighted_edges:
        assert weight == 1
        fine_laplacian[left, left] += 1
        fine_laplacian[right, right] += 1
        fine_laplacian[left, right] -= 1
        fine_laplacian[right, left] -= 1
    full_embedding = sp.kronecker_product(sp.eye(5), embedding)
    full_projection = sp.kronecker_product(sp.eye(5), projection)
    reduced_full = sp.kronecker_product(
        child_laplacian, sp.eye(4)
    ) + sp.kronecker_product(sp.eye(5), reduced)
    assert full_projection * fine_laplacian * full_embedding == reduced_full
    assert fine_laplacian * full_embedding == full_embedding * reduced_full
    # All forty input axes retain their signed association and amplitudes;
    # these are pure chart evaluations, not additional response preparations.
    for axis in range(20):
        forms = tuple(Q(1, 8) if i == axis else Q(0) for i in range(20))
        phases = tuple(Q(1, 16) if i == axis else Q(0) for i in range(20))
        lifted = owner._lift_organized_source_reflection(
            form_coordinates=forms, phase_coordinates=phases
        )
        assert (lifted.form, lifted.phase_offsets) == _independent_lift(forms, phases)
        restored = owner._restrict_organized_source_reflection(
            form=lifted.form, phase_offsets=lifted.phase_offsets
        )
        assert restored.form_coordinates == forms
        assert restored.phase_coordinates == phases


def test_affine_primitive_reflection_and_full_support_preserve_both_rows(geometry):
    reflection = tuple(
        9 * child + (1 - slot) % 9 for child in range(5) for slot in range(9)
    )
    turns = tuple(-1 if i % 9 in (0, 1) else 0 for i in range(45))
    edges = {
        tuple(sorted((left, right))): weight
        for left, right, weight in geometry.weighted_edges
    }
    assert len(edges) == 90
    assert {
        tuple(sorted((reflection[left], reflection[right]))): weight
        for (left, right), weight in edges.items()
    } == edges
    assert geometry.held_capacities == (Q(1),) * 45
    assert geometry.clock_factor == Q(1, 4)
    alpha, m = 2 * sp.pi / 9, sp.Rational(1, 40)
    eta = sp.symbols("eta0:45", real=True)
    phases = tuple(alpha * (node % 9 - 4) + m + eta[node] for node in range(45))
    constant = 2 * m + 2 * alpha
    for node, partner in enumerate(reflection):
        assert reflection[partner] == node
        assert turns[partner] == turns[node]
        reflected = alpha * (node % 9 - 4) + m - eta[partner]
        assert (
            sp.simplify(
                reflected - (constant - phases[partner] + 2 * sp.pi * turns[node])
            )
            == 0
        )
    gap = sp.Symbol("gap", real=True)
    for left, right in edges:
        turn_difference = turns[right] - turns[left]
        assert (
            sp.simplify(sp.sin(-gap + 2 * sp.pi * turn_difference) + sp.sin(gap)) == 0
        )
    # Edge weights and support are permuted unchanged. The signed form
    # gradient and sine current therefore both transform by -R, proving
    # equivariance of both declared rows, not only of a mean observation.


@pytest.mark.parametrize("case", tuple(CASES))
def test_chart_roundtrip_retains_fine_state_and_each_child_mean(case, observations):
    forms, phases = CASES[case]
    report = observations[case]
    expected_form, expected_offsets = _independent_lift(forms, phases)
    assert report.state.form_coordinates == forms
    assert report.state.phase_coordinates == phases
    assert report.state.form == expected_form
    assert report.state.phase_offsets == expected_offsets
    assert report.fine.form == expected_form
    assert report.fine.phase_offsets == expected_offsets
    assert (
        owner._restrict_organized_source_reflection(
            form=expected_form, phase_offsets=expected_offsets
        )
        == report.state
    )
    assert report.fine.child_form_means == (Q(0),) * 5
    assert report.fine.child_phase_offset_means == (ORIGIN,) * 5
    assert report.fine.conserved_form_total == 0
    assert report.fine.conserved_phase_total == Q(1, 8)


def _independent_rates(mp, form, offsets, geometry):
    values = tuple(_mp(mp, value) for value in form)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value) for i, value in enumerate(offsets)
    )
    gradient, current = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
    for left, right, weight in geometry.weighted_edges:
        w = _mp(mp, weight)
        difference = w * (values[left] - values[right])
        sine = w * mp.sin(theta[right] - theta[left])
        gradient[left] += difference
        gradient[right] -= difference
        current[left] += sine
        current[right] -= sine
    gamma = 1 / (1023 * mp.pi)
    return tuple((-q + gamma * s) / 4 for q, s in zip(gradient, current)), tuple(
        gamma * q / 4 for q in gradient
    )


@pytest.mark.parametrize("case", tuple(CASES))
def test_shared_reduced_and_fine_rows_intertwine_in_all_three_clocks(
    case, observations, geometry, mp
):
    report = observations[case]
    form, offsets = _independent_lift(*CASES[case])
    predictions = _independent_rates(mp, form, offsets, geometry)
    gamma = 1 / (1023 * mp.pi)
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("original", mp.mpf(1023) / 1024),
        ("slow", 2 / gamma**2),
    ):
        for channel, expected in zip(("form", "phase"), predictions):
            reduced = getattr(report, f"{channel}_{clock}_rate_bounds")
            fine = getattr(report.fine, f"{channel}_{clock}_rate_bounds")
            assert len(reduced) == 20 and len(fine) == 45
            for child in range(5):
                for column, slot in enumerate(REPRESENTATIVES):
                    coordinate, node = 4 * child + column, 9 * child + slot
                    partner = 9 * child + (1 - slot) % 9
                    assert reduced[coordinate] == fine[node]
                    _contains(mp, reduced[coordinate], factor * expected[node])
                    _contains(mp, -reduced[coordinate], factor * expected[partner])
                assert fine[9 * child + 5].contains(0)
                for slot in range(9):
                    node = 9 * child + slot
                    _contains(mp, fine[node], factor * expected[node])
    for child in range(5):
        assert report.fine.child_form_tau_rate_bounds[child].contains(0)
        assert report.fine.child_phase_tau_rate_bounds[child].contains(0)
    if case == "general":
        assert any(not value.contains(0) for value in report.form_tau_rate_bounds)
        assert any(not value.contains(0) for value in report.phase_tau_rate_bounds)


def test_frozen_delayed_response_survives_the_exact_representation(observations, mp):
    state = observations["delayed"].state
    profile = (Q(1), Q(-1)) + (Q(0),) * 7
    for child in range(5):
        sign = 1 if child == 1 else -1 if child == 2 else 0
        assert state.phase_offsets[9 * child : 9 * child + 9] == tuple(
            ORIGIN + sign * Q(1, 100) * value for value in profile
        )
    acceleration = owner._observe_organized_source_acceleration(
        form=state.form, phase_offsets=state.phase_offsets
    )
    form_contrast = (
        acceleration.form_tau_acceleration_bounds[0]
        - acceleration.form_tau_acceleration_bounds[1]
    )
    phase_contrast = (
        acceleration.phase_tau_acceleration_bounds[0]
        - acceleration.phase_tau_acceleration_bounds[1]
    )
    assert form_contrast.hi < 0 and phase_contrast.lo > 0
    gamma, alpha, epsilon = 1 / (1023 * mp.pi), 2 * mp.pi / 9, mp.mpf(1) / 100
    difference = mp.cos(epsilon) - mp.cos(2 * epsilon)
    _contains(mp, form_contrast, -gamma * mp.sin(alpha) * difference / 4)
    _contains(mp, phase_contrast, gamma**2 * mp.sin(alpha) * difference / 4)
    assert all(
        value.contains(0) for value in acceleration.child_form_tau_acceleration_bounds
    )
    assert all(
        value.contains(0) for value in acceleration.child_phase_tau_acceleration_bounds
    )


def test_explicit_four_slot_current_and_inherited_storage_match_the_fine_law(
    observations, geometry, mp
):
    report = observations["general"]
    alpha, gamma = 2 * mp.pi / 9, 1 / (1023 * mp.pi)
    phase = tuple(_mp(mp, value) for value in GENERIC_PHASE)
    gradient, current = [], []
    for child in range(5):
        start = 4 * child
        u0, u1, u2, u3 = GENERIC_FORM[start : start + 4]
        gradient.extend(
            (3 * u0 + u1, u0 + 2 * u1 - u2, -u1 + 2 * u2 - u3, -u2 + 2 * u3)
        )
        a0, a1, a2, a3 = phase[start : start + 4]
        current.extend(
            (
                mp.sin(alpha - 2 * a0) - mp.sin(alpha + a0 + a1),
                mp.sin(alpha + a2 - a1) - mp.sin(alpha + a0 + a1),
                mp.sin(alpha + a3 - a2) - mp.sin(alpha + a2 - a1),
                mp.sin(alpha - a3) - mp.sin(alpha + a3 - a2),
            )
        )
    for left, right in geometry.collective_edges:
        for slot in range(4):
            a, b = 4 * left + slot, 4 * right + slot
            difference = GENERIC_FORM[a] - GENERIC_FORM[b]
            sine = mp.sin(phase[b] - phase[a])
            gradient[a] += difference
            gradient[b] -= difference
            current[a] += sine
            current[b] -= sine
    form_rate = tuple((-_mp(mp, q) + gamma * s) / 4 for q, s in zip(gradient, current))
    phase_rate = tuple(gamma * _mp(mp, q) / 4 for q in gradient)
    for coordinate in range(20):
        child, column = divmod(coordinate, 4)
        node = 9 * child + REPRESENTATIVES[column]
        assert report.fine.form_gradient[node] == gradient[coordinate]
        _contains(mp, report.fine.phase_current_bounds[node], current[coordinate])
        _contains(mp, report.form_tau_rate_bounds[coordinate], form_rate[coordinate])
        _contains(mp, report.phase_tau_rate_bounds[coordinate], phase_rate[coordinate])

    def potential(coordinates):
        offsets = [_mp(mp, ORIGIN)] * 45
        for child in range(5):
            for column, slot in enumerate(REPRESENTATIVES):
                value = coordinates[4 * child + column]
                offsets[9 * child + slot] += value
                offsets[9 * child + (1 - slot) % 9] -= value
        theta = tuple(
            alpha * (node % 9 - 4) + value for node, value in enumerate(offsets)
        )
        return sum(
            _mp(mp, weight) * (1 - mp.cos(theta[right] - theta[left]))
            for left, right, weight in geometry.weighted_edges
        )

    form_storage = sum(value * q for value, q in zip(GENERIC_FORM, gradient))
    _contains(
        mp, report.fine.joint_storage_bounds, _mp(mp, form_storage) + potential(phase)
    )
    exact_loss = -sum(value**2 for value in gradient) / 2
    assert report.fine.joint_storage_tau_rate == exact_loss < 0
    # Differentiate the original ninety-edge potential along the reduced phase
    # velocity independently of its documented gradient -2*j(a).
    potential_rate = mp.diff(
        lambda time: potential(
            tuple(a + time * rate for a, rate in zip(phase, phase_rate))
        ),
        0,
    )
    assert abs(
        potential_rate + 2 * sum(a * b for a, b in zip(current, phase_rate))
    ) < mp.mpf("1e-70")
    direct_storage_rate = (
        2 * sum(_mp(mp, q) * rate for q, rate in zip(gradient, form_rate))
        + potential_rate
    )
    assert abs(direct_storage_rate - _mp(mp, exact_loss)) < mp.mpf("1e-70")


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("defect", ("pair", "fixed_slot"))
def test_restriction_rejects_nonsector_state_without_projection(field, defect):
    form, offsets = _independent_lift(GENERIC_FORM, GENERIC_PHASE)
    supplied = {"form": list(form), "phase_offsets": list(offsets)}
    first, second = (0, 2) if defect == "pair" else (5, 0)
    supplied[field][first] += Q(1, 100)
    supplied[field][second] -= Q(1, 100)
    before = {name: tuple(values) for name, values in supplied.items()}
    # Even the child totals remain unchanged: they cannot authorize projection.
    assert sum(supplied["form"][:9]) == 0
    assert sum(supplied["phase_offsets"][:9]) == 9 * ORIGIN
    with pytest.raises(ValueError):
        owner._restrict_organized_source_reflection(**supplied)
    assert {name: tuple(values) for name, values in supplied.items()} == before


@pytest.mark.parametrize("field", ("form_coordinates", "phase_coordinates"))
@pytest.mark.parametrize("bad", (True, float("inf"), "short"))
def test_reduced_raw_admission_precedes_fine_observation(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Invalid reduced coordinates reached fine observation")

    monkeypatch.setattr(owner, "_observe_organized_source_port", forbidden)
    supplied = {"form_coordinates": ZERO, "phase_coordinates": ZERO}
    supplied[field] = ZERO[:-1] if bad == "short" else ZERO[:-1] + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_source_reflection(**supplied)


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
def test_full_raw_admission_is_not_replaced_by_sector_equalities(field):
    form, offsets = _independent_lift(ZERO, ZERO)
    supplied = {"form": form, "phase_offsets": offsets}
    supplied[field] = supplied[field][:-1] + (True,)
    with pytest.raises((TypeError, ValueError)):
        owner._restrict_organized_source_reflection(**supplied)


def test_binary64_origin_is_not_silently_identified_with_the_exact_origin():
    assert Q.from_float(0.025) != ORIGIN
    with pytest.raises(ValueError):
        owner._restrict_organized_source_reflection(
            form=(0.0,) * 45, phase_offsets=(0.025,) * 45
        )
    # Represented zero deviations are admissible: the chart then adds its exact
    # rational origin, rather than rejecting all floats or rounding fine data.
    state = owner._lift_organized_source_reflection(
        form_coordinates=(0.0,) * 20, phase_coordinates=(0.0,) * 20
    )
    assert state.phase_offsets == (ORIGIN,) * 45
    assert (
        owner._restrict_organized_source_reflection(
            form=state.form, phase_offsets=state.phase_offsets
        )
        == state
    )


def test_symbolic_sector_contains_a_stationary_zero_winding_state(geometry):
    # This is an exact symbolic boundary control, not an additional numerical
    # preparation and not a call admitting approximate multiples of 2*pi.
    alpha, m = 2 * sp.pi / 9, sp.Rational(1, 40)
    coordinates = (5 * alpha, 3 * alpha, 2 * alpha, alpha) * 5
    form, offsets = _independent_lift(ZERO, coordinates)
    theta = tuple(
        sp.simplify(alpha * (node % 9 - 4) + value)
        for node, value in enumerate(offsets)
    )
    assert form == (Q(0),) * 45
    for node, phase in enumerate(theta):
        turn = sp.simplify((phase - m - alpha) / (2 * sp.pi))
        assert turn == (-1 if node % 9 == 1 else 0)
    for left, right, _ in geometry.weighted_edges:
        gap = theta[right] - theta[left]
        assert sp.simplify(sp.sin(gap)) == 0
        assert sp.simplify(sp.cos(gap)) == 1
    # All circular phases coincide: principal edge increments and winding are
    # zero. Both rows vanish, although the real chart still retains its turns.


def test_exact_sector_admission_does_not_claim_an_acute_winding_domain(geometry):
    phases = (Q(2),) + ZERO[1:]
    lifted = owner._lift_organized_source_reflection(
        form_coordinates=ZERO, phase_coordinates=phases
    )
    restored = owner._restrict_organized_source_reflection(
        form=lifted.form, phase_offsets=lifted.phase_offsets
    )
    assert restored == lifted
    first_angle = (
        geometry.alpha_bounds + lifted.phase_offsets[1] - lifted.phase_offsets[0]
    )
    assert first_angle.hi < -pi_interval().hi
    # The unrestricted real-lift symmetry still admits this coordinate. No
    # acute-cell, winding or maintenance verdict follows from chart membership.

"""Static arithmetic for the declared positive weak-spoke contact law.

These controls check complete weighted rows, storage and a sufficient
barrier. They run no source producer, stationary search or trajectory.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import cos, pi_interval

_OLD_EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(6)
    for slot in range(9)
) + tuple(
    (9 * child + slot, 9 * ((child + 1) % 5) + slot)
    for child in range(5)
    for slot in range(9)
)
_SPOKES = tuple(
    (9 * child + slot, 45 + slot) for child in range(5) for slot in range(9)
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _weighted_edges(eta):
    return tuple((left, right, 1) for left, right in _OLD_EDGES) + tuple(
        (left, right, eta) for left, right in _SPOKES
    )


def _product_laplacian(mp, eta):
    """Assemble I6 tensor L9 + L_wheel(eta) tensor I9 by node rows."""
    laplacian = mp.zeros(54)
    for child in range(6):
        for slot in range(9):
            index = 9 * child + slot
            laplacian[index, index] = 4 + eta if child < 5 else 2 + 5 * eta
            for neighbor_slot in ((slot - 1) % 9, (slot + 1) % 9):
                laplacian[index, 9 * child + neighbor_slot] = -1
            if child < 5:
                for neighbor_child in ((child - 1) % 5, (child + 1) % 5):
                    laplacian[index, 9 * neighbor_child + slot] = -1
                laplacian[index, 45 + slot] = -eta
            else:
                for neighbor_child in range(5):
                    laplacian[index, 9 * neighbor_child + slot] = -eta
    return laplacian


@pytest.mark.parametrize("eta_value", (Q(1, 100000), Q(1)))
def test_weak_contact_complete_weighted_rows_and_nodal_loss(mp, eta_value):
    """Independent edge storage and node rows retain the same spoke weights."""
    tolerance = mp.mpf("1e-85")
    eta, gamma = _mp(mp, eta_value), 1 / (1023 * mp.pi)
    edges = _weighted_edges(eta)
    assert len(_OLD_EDGES) == 99
    assert len(_SPOKES) == 45
    assert len({tuple(sorted((a, b))) for a, b, _ in edges}) == 144

    raw_form = tuple(mp.mpf((7 * index) % 17 - 8) / 19 for index in range(54))
    form = mp.matrix([value - sum(raw_form) / 54 for value in raw_form])
    phase = mp.matrix([mp.mpf((11 * index) % 23 - 11) / 13 for index in range(54)])
    form_gradient, current = mp.zeros(54, 1), mp.zeros(54, 1)
    for left, right, weight in edges:
        exchange = weight * (form[left] - form[right])
        form_gradient[left] += exchange
        form_gradient[right] -= exchange
        pressure = weight * mp.sin(phase[right] - phase[left])
        current[left] += pressure
        current[right] -= pressure

    laplacian = _product_laplacian(mp, eta)
    assert max(abs(value) for value in laplacian * form - form_gradient) < tolerance
    assert max(abs(value) for value in laplacian * mp.ones(54, 1)) < tolerance
    assert abs(sum(form)) < tolerance
    assert abs(sum(current)) < tolerance
    form_rate = (-form_gradient + gamma * current) / 4
    phase_rate = gamma * laplacian * form / 4
    assert abs(sum(form_rate)) < tolerance
    assert abs(sum(phase_rate)) < tolerance

    def storage_on_tangent(parameter, velocity=phase_rate):
        forms = form + parameter * form_rate
        phases = phase + parameter * velocity
        return sum(
            weight
            * ((forms[b] - forms[a]) ** 2 / 2 + 1 - mp.cos(phases[b] - phases[a]))
            for a, b, weight in edges
        )

    loss = -(form_gradient.T * form_gradient)[0] / 4
    assert abs(mp.diff(storage_on_tangent, 0) - loss) < tolerance
    if eta_value != 1:
        # Leaving unit spokes in the phase row breaks the declared loss identity.
        unit_laplacian = _product_laplacian(mp, mp.mpf(1))
        wrong_velocity = gamma * unit_laplacian * form / 4
        defect = gamma * (current.T * (laplacian - unit_laplacian) * form)[0] / 4
        wrong_rate = mp.diff(lambda t: storage_on_tangent(t, wrong_velocity), 0)
        assert abs(defect) > mp.mpf("1e-10")
        assert abs(wrong_rate - loss - defect) < tolerance


@pytest.mark.parametrize("beta_turns", (Q(0), Q(1, 14)))
def test_weak_contact_reference_circle_energy_and_nonstationarity(mp, beta_turns):
    """The low-energy circle is a comparison set, not an equilibrium family."""
    tolerance = mp.mpf("1e-85")
    eta = mp.mpf(1) / 100000
    alpha, beta = 2 * mp.pi / 9, 2 * mp.pi * _mp(mp, beta_turns)
    origins = tuple(2 * mp.pi * (child - 2) / 5 for child in range(5)) + (beta,)
    phase = tuple(
        alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)
    )
    current = [mp.mpf(0)] * 54
    old_storage = sum(1 - mp.cos(phase[b] - phase[a]) for a, b in _OLD_EDGES)
    spoke_storage = sum(1 - mp.cos(phase[b] - phase[a]) for a, b in _SPOKES)
    for left, right, weight in _weighted_edges(eta):
        exchange = weight * mp.sin(phase[right] - phase[left])
        current[left] += exchange
        current[right] -= exchange
    v9, v5 = 9 * (1 - mp.cos(alpha)), 5 * (1 - mp.cos(2 * mp.pi / 5))
    assert abs(old_storage - (6 * v9 + 9 * v5)) < tolerance
    assert abs(spoke_storage - 45) < tolerance
    assert (
        abs(old_storage + eta * spoke_storage - (6 * v9 + 9 * v5 + 45 * eta))
        < tolerance
    )
    for child in range(5):
        for slot in range(9):
            assert (
                abs(current[9 * child + slot] - eta * mp.sin(beta - origins[child]))
                < tolerance
            )
    assert max(abs(value) for value in current[45:]) < tolerance
    assert max(abs(value) for value in current[:45]) > eta / 2


def test_weak_contact_old_cycle_faces_give_strict_rational_barrier(mp):
    """Exact outward arithmetic checks feasible faces and the spare gap."""
    pi = pi_interval()
    margins = []
    for size in (9, 5):
        ground = size * (1 - cos(2 * pi / size))
        positive_face = 1 + (size - 1) * (1 - cos(3 * pi / (2 * (size - 1))))
        margin = positive_face - ground
        margins.append(margin)
        # Independent saturated-face gaps sum to the same winding-one lift.
        plus_gaps = (mp.pi / 2,) + (3 * mp.pi / (2 * (size - 1)),) * (size - 1)
        minus_gaps = (-mp.pi / 2,) + (5 * mp.pi / (2 * (size - 1)),) * (size - 1)
        expected_ground = size * (1 - mp.cos(2 * mp.pi / size))
        expected_margin = sum(1 - mp.cos(gap) for gap in plus_gaps) - expected_ground
        assert abs(sum(plus_gaps) - 2 * mp.pi) < mp.mpf("1e-85")
        assert abs(sum(minus_gaps) - 2 * mp.pi) < mp.mpf("1e-85")
        assert _mp(mp, margin.lo) < expected_margin < _mp(mp, margin.hi)
        if size == 9:
            negative_face = 1 + (size - 1) * (1 - cos(5 * pi / (2 * (size - 1))))
            assert negative_face.lo > positive_face.hi
            assert all(abs(gap) <= mp.pi / 2 for gap in minus_gaps)
        else:
            # A negative C5 face cannot reach winding one with four acute gaps.
            assert -mp.pi / 2 + (size - 1) * mp.pi / 2 < 2 * mp.pi
            assert any(abs(gap) > mp.pi / 2 for gap in minus_gaps)
    assert margins[1].hi < margins[0].lo
    barrier_lower = min(value.lo for value in margins)
    barrier_upper = min(value.hi for value in margins)
    assert barrier_lower > Q(7, 500)
    assert 45 * Q(1, 100000) < barrier_lower
    # The sufficient weak-contact bound does not admit the original unit point.
    # Its failure is not a nonexistence or instability verdict at that point.
    assert 45 * Q(1) > barrier_upper


def test_weak_contact_source_budget_uses_only_retained_analytic_allowances():
    """Recombine exact majorants; do not run or replace their source producers."""
    gamma_upper, form_allowance, duration = Q(1, 3000), Q(1, 100), Q(1, 5000)
    reached_error, eta = Q(1, 10**70), Q(1, 100000)
    form_bound = form_allowance + gamma_upper * duration
    form_drift = (2 * form_bound + gamma_upper) * duration
    phase_drift = 2 * gamma_upper * form_bound * duration
    host_increase = 360 * form_bound * form_drift + 180 * phase_drift
    donor_excess = 4 * reached_error**2
    attachment = 45 * eta * ((form_bound + reached_error) ** 2 / 2 + 2)
    total_excess = Q(11, 1000) + host_increase + donor_excess + attachment
    assert host_increase < Q(1, 50000)
    assert attachment < Q(901, 10**6)
    assert total_excess < Q(3, 250)
    assert Q(7, 500) - total_excess > Q(1, 500)

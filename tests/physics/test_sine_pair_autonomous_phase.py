"""Static fine-row controls for the frozen autonomous phase corollary.

Only source states, local Cartesian jets and analytic remainder inequalities
are evaluated. No trajectory, operator, response fit or archived producer is
run. The protocol is owned by SINE_PAIR_INTERACTION.md section 36.8.
"""

from fractions import Fraction as Q
from math import factorial

import mpmath
import pytest

from tests.physics._sine_pair_oracles import (
    EDGES,
    NEIGHBORS,
    PAIRS,
    _contains,
    _energy,
    _fine_jets,
    _mp,
)
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics.relational_sine_comparison import (
    _sine_form_gradient,
    _sine_rate_evaluator,
)

AMPLITUDES = (Q(1, 1024), Q(1, 768))
HORIZON = Q(1, 2)
FORM = Q(5, 16)
ERROR = Q(1, 2**40)
RADIUS = Q(1, 16)
SOURCE = PAIRS[1]
READOUT = tuple(Q(int(i in PAIRS[0]) - int(i in PAIRS[2]), 2) for i in range(10))


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _phase_row(values):
    return tuple(
        values[i] - sum((values[j] for j in row), Q(0)) / len(row)
        for i, row in enumerate(NEIGHBORS)
    )


def _readout(values):
    return sum((weight * value for weight, value in zip(READOUT, values)), Q(0))


def _source(mp, amplitude, *, gamma=Q(0), winding=1):
    alpha = winding * 2 * mp.pi / 5
    phases = [_mp(mp, gamma) + (i // 2) * alpha for i in range(10)]
    phases[SOURCE[0]] += _mp(mp, amplitude)
    phases[SOURCE[1]] -= _mp(mp, amplitude)
    return (FORM,) * 10, phases


def _jets(mp, forms, phases):
    phasors = tuple((mp.cos(theta), mp.sin(theta)) for theta in phases)
    return _fine_jets(phasors, order=3, forms=forms)[0]


def _phase_coefficients(jets):
    # The phase row is linear in the *fine* forms. Integrate its exact local
    # Taylor coefficients; these are derivatives at zero, not a time step.
    return (Q(0),) + tuple(
        _readout(_phase_row(tuple(node[order] for node in jets))) / (order + 1)
        for order in range(4)
    )


@pytest.mark.parametrize("amplitude", (*AMPLITUDES, sum(AMPLITUDES) / 2))
def test_original_fine_jets_give_the_signed_relative_phase_acceleration(mp, amplitude):
    forms, phases = _source(mp, amplitude)
    jets = _jets(mp, forms, phases)
    coefficients = _phase_coefficients(jets)
    alpha = 2 * mp.pi / 5
    rate = mp.sin(alpha) * (1 - mp.cos(_mp(mp, amplitude))) / 2
    mean_rates = tuple((jets[left][1] + jets[right][1]) / 2 for left, right in PAIRS)
    expected = (-rate, 0, rate, 0, 0)
    assert max(
        abs(value - target) for value, target in zip(mean_rates, expected)
    ) < mp.mpf("1e-90")
    assert all(
        abs(_mp(mp, coefficients[index])) < mp.mpf("1e-90") for index in (0, 1, 3)
    )
    assert abs(coefficients[2] + rate) < mp.mpf("1e-90")
    assert coefficients[2] < 0
    assert abs(sum(node[1] for node in jets)) < mp.mpf("1e-90")
    # Initially identical mean forms and midpoint phases do not determine
    # this acceleration: the nonzero internal phase split is consumed.
    for pair, (left, right) in enumerate(PAIRS):
        assert forms[left] == forms[right] == FORM
        assert abs((phases[left] + phases[right]) / 2 - pair * alpha) < mp.mpf("1e-90")


def test_zero_split_pair_swap_and_gauge_are_actual_fine_law_controls(mp):
    forms, phases = _source(mp, Q(0))
    zero = _jets(mp, forms, phases)
    assert max(abs(node[order]) for node in zero for order in range(1, 4)) < mp.mpf(
        "1e-90"
    )
    forms, phases = _source(mp, AMPLITUDES[1])
    original = _jets(mp, forms, phases)
    swapped = phases.copy()
    swapped[SOURCE[0]], swapped[SOURCE[1]] = swapped[SOURCE[1]], swapped[SOURCE[0]]
    permuted = _jets(mp, forms, swapped)
    indices = list(range(10))
    indices[SOURCE[0]], indices[SOURCE[1]] = indices[SOURCE[1]], indices[SOURCE[0]]
    for i, counterpart in enumerate(indices):
        assert max(
            abs(_mp(mp, a - b)) for a, b in zip(permuted[i], original[counterpart])
        ) < mp.mpf("1e-90")
    shifted_forms = tuple(value + Q(7, 13) for value in forms)
    shifted_phases = [value + _mp(mp, Q(11, 17)) for value in phases]
    shifted = _jets(mp, shifted_forms, shifted_phases)
    assert max(
        abs(_mp(mp, a - b))
        for a, b in zip(_phase_coefficients(original), _phase_coefficients(shifted))
    ) < mp.mpf("1e-90")
    reversed_jets = _jets(mp, *_source(mp, AMPLITUDES[1], winding=-1))
    assert abs(
        _phase_coefficients(original)[2] + _phase_coefficients(reversed_jets)[2]
    ) < mp.mpf("1e-90")


def test_shared_zero_loss_rates_transform_both_rows_to_the_declared_clock(mp):
    model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
    evaluator = _sine_rate_evaluator(model, (4,) * 10, (Q(1),) * 10)
    alpha, pi = 2 * pi_interval() / 5, pi_interval()
    phases = tuple(
        (i // 2) * alpha
        + Q(11, 17)
        + (AMPLITUDES[0] if i == SOURCE[0] else -AMPLITUDES[0] if i == SOURCE[1] else 0)
        for i in range(10)
    )
    # A separate static state makes *both* clock transformations nonzero.
    # It is a kernel control, not a modified reserved preparation.
    forms = tuple(FORM + Q((i % 3) - 1, 4096) for i in range(10))
    currents = tuple(
        sum((sin(phases[j] - phases[i]) for j in row), I(0))
        for i, row in enumerate(NEIGHBORS)
    )
    rates = evaluator(_sine_form_gradient(forms, NEIGHBORS), currents)
    _, precise_phases = _source(mp, AMPLITUDES[0], gamma=Q(11, 17))
    independent = _jets(mp, forms, precise_phases)
    expected_phase = _phase_row(forms)
    assert any(value != 0 for value in expected_phase)
    for i in range(10):
        _contains(mp, pi * rates["form_rates"][i], independent[i][1])
        _contains(mp, pi * rates["phase_rates"][i], _mp(mp, expected_phase[i]))
        _contains(mp, rates["pressure"][i], independent[i][1] / mp.pi)


def _inherited_constants():
    # Recover the two fine/mean row norms and the selected-source fraction
    # from the original support, rather than importing a reported verdict.
    phase_norm = max(1 + sum(Q(1, len(row)) for _ in row) for row in NEIGHBORS)
    current_lipschitz = max(sum(Q(2, len(row)) for _ in row) for row in NEIGHBORS)
    source_fraction = sum(
        (
            Q(int(j in SOURCE), 2 * len(NEIGHBORS[i]))
            for i in PAIRS[0]
            for j in NEIGHBORS[i]
        ),
        Q(0),
    )
    internal_amplification = 1 / (1 - HORIZON**2 / factorial(2))
    forcing = source_fraction * internal_amplification**2 / 2
    feedback = current_lipschitz * phase_norm * HORIZON**2 / factorial(3)
    growth = forcing / (1 - feedback)
    rate_remainder = current_lipschitz * growth + forcing
    fine_weights = tuple(
        _readout(_phase_row(tuple(Q(int(j == i)) for j in range(10))))
        for i in range(10)
    )
    readout_norm = sum(map(abs, fine_weights), Q(0))
    # Twice integrating a quadratic rate bound gives 2!/4!. The actual
    # phase-observation row supplies norm 3, instead of a generic norm 4.
    phase_remainder = readout_norm * rate_remainder * Q(factorial(2), factorial(4))
    return fine_weights, phase_remainder


def test_fine_graph_and_integrated_majorants_certify_the_frozen_endpoint():
    fine_weights, remainder = _inherited_constants()
    coarse_weights = tuple(sum(fine_weights[i] for i in pair) for pair in PAIRS)
    assert coarse_weights == (1, 0, -1, Q(1, 2), Q(-1, 2))
    assert sum(map(abs, fine_weights)) == 3
    assert sin(2 * pi_interval() / 5).lo > Q(9, 10)
    # On |epsilon| <= 1, the alternating cosine remainder gives
    # 1-cos(epsilon) >= epsilon²*(1/2-1/24) > epsilon²/3.
    assert AMPLITUDES[1] < 1
    assert Q(1, 2) - Q(1, factorial(4)) > Q(1, 3)
    initial_coefficient = Q(9, 10) * Q(1, 3) / 2
    coefficient = initial_coefficient - remainder * HORIZON**2
    assert coefficient == Q(79, 980)

    # Independent branch errors are propagated in the phase channel. The
    # original two-channel row norms are two; factorial majorants enclose
    # cosh(2h) and sinh(2h), with no comparison-trajectory integration.
    even = 1 / (1 - (2 * HORIZON) ** 2 / factorial(2))
    odd = 2 * HORIZON / (1 - (2 * HORIZON) ** 2 / factorial(3))
    phase_error = ERROR * (even + odd)
    observation_norm = sum(map(abs, READOUT), Q(0))
    contrast_error = 2 * (observation_norm * phase_error + ERROR)
    upper = -coefficient * AMPLITUDES[0] ** 2 * HORIZON**2 + contrast_error
    assert contrast_error == Q(74, 5) * ERROR
    assert upper < -Q(1, 2**26)


def test_fine_storage_and_radius_retain_the_organized_source_family(mp):
    laplacian = mp.matrix(10)
    for i, row in enumerate(NEIGHBORS):
        laplacian[i, i] = len(row)
        for j in row:
            laplacian[i, j] = -1
    eigenvalues = list(mp.eigsy(laplacian, eigvals_only=True))
    assert abs(eigenvalues[1] - (5 - mp.sqrt(5))) < mp.mpf("1e-90")
    assert abs(eigenvalues[-1] - (5 + mp.sqrt(5))) < mp.mpf("1e-90")
    assert eigenvalues[-1] < 8
    alpha = 2 * pi_interval() / 5
    source_radius = sqrt(I(2)) * AMPLITUDES[1] + sqrt(I(20)) * ERROR
    assert source_radius.hi < Q(1, 512)
    coercivity = (5 - sqrt(I(5))) * cos(alpha + sqrt(I(2)) * RADIUS) / 2
    assert coercivity.lo > Q(1, 8)
    assert (4 * source_radius**2).hi < Q(1, 8) * RADIUS**2

    _, target_phases = _source(mp, Q(0))
    baseline = _energy(mp, [_mp(mp, FORM)] * 10, target_phases)
    for amplitude in AMPLITUDES:
        forms, phases = _source(mp, amplitude)
        energy = _energy(mp, [_mp(mp, value) for value in forms], phases)
        expected = 8 * mp.cos(2 * mp.pi / 5) * (1 - mp.cos(_mp(mp, amplitude)))
        assert abs(energy - baseline - expected) < mp.mpf("1e-90")
        assert energy > baseline
        # A full-coordinate perturbation leaves the nominal synchronized-tip
        # manifold; the bound retains it without imposing reflection symmetry.
        dx = [_mp(mp, ERROR) * (1 if i % 2 else -1) for i in range(10)]
        dt = [_mp(mp, ERROR) * (1 if i % 3 else -1) for i in range(10)]
        perturbed_forms = [_mp(mp, x) + error for x, error in zip(forms, dx)]
        perturbed_phases = [theta + error for theta, error in zip(phases, dt)]
        norm_squared = sum(value**2 for value in dx) + sum(
            (theta - target) ** 2
            for theta, target in zip(perturbed_phases, target_phases)
        )
        assert norm_squared < _mp(mp, source_radius.hi) ** 2
        assert (
            _energy(mp, perturbed_forms, perturbed_phases) - baseline
            <= 4 * norm_squared
        )
    assert len(EDGES) == 20

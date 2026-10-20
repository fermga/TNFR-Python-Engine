"""Exact quartic algebra and actual-source controls without trajectory replay."""

import builtins
from copy import deepcopy
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path

import mpmath
import pytest

from tnfr.dynamics import integrators
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.operators import network_stage, word_execution
from tnfr.research import weighted_star_contact, weighted_star_finite_episode
from tnfr.research.artifact_io import decode_exact_tree, read_bytes_bounded
from tnfr.research.weighted_star_constitutive_handoff import (
    compare_weighted_star_constitutive_handoff,
)
from tnfr.research.weighted_star_quartic import (
    _directional_derivative,
    _evaluate,
    _linear_field,
    _modular_pivots,
    _quadratic_polynomial,
    _solve_fixed_system,
    _verify_correction,
    assess_weighted_star_quartic_transfer,
    build_weighted_star_quartic_system,
)
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module", autouse=True)
def no_producer_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("the quartic audit must not replay a scientific producer")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (weighted_star_contact, ("contact_source", "execute_contact_word")),
            (
                weighted_star_finite_episode,
                ("prepare_source", "evaluate_protocol", "execute_contact_word"),
            ),
            (integrators, ("rk4_coupled_step",)),
            (word_execution, ("execute_network_operator_stage",)),
            (network_stage, ("execute_pointwise_stage", "execute_coupling_stage")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def preparation():
    path = Path(__file__).resolve().parents[2] / (
        "docs/assets/weighted_star_registration_episode/preparation.json"
    )
    return decode_exact_tree(json_loads(read_bytes_bounded(path, max_bytes=256 * 1024)))


@pytest.fixture(scope="module")
def result(preparation):
    pytest.importorskip("sympy")
    return assess_weighted_star_quartic_transfer(preparation)


@pytest.fixture(scope="module")
def source(preparation):
    return compare_weighted_star_constitutive_handoff(preparation).prefixes[-1].source


def test_frozen_parity_order_gauge_and_exact_full_rank(result):
    system = result.system
    unknowns, equations = system.unknown_monomials, system.equation_monomials
    assert len(unknowns) == 170 and len(equations) == 160
    assert tuple(sorted(unknowns, reverse=True)) == unknowns
    assert tuple(sorted(equations, reverse=True)) == equations
    assert all(sum(e) == 4 and sum(e[:4]) % 2 == 0 for e in unknowns)
    assert all(sum(e) == 4 and sum(e[:4]) % 2 == 1 for e in equations)
    assert system.certified_pivots == tuple(range(160))
    assert result.free_columns == tuple(range(160, 170))
    assert all(result.coefficients[i] == 0 for i in result.free_columns)
    prime = system.rank_prime
    assert all(prime % value for value in range(2, isqrt(prime) + 1))
    assert system.gauge_dimension == len(system.gauge_basis) == 10
    assert len(unknowns) - len(system.certified_pivots) == system.gauge_dimension
    for row, expected in zip(system.matrix, system.right_hand_side):
        assert sum((a * b for a, b in zip(row, result.coefficients)), Q(0)) == expected


def test_modular_rank_is_a_lower_bound_and_rejects_nonunit_denominators():
    assert _modular_pivots(((Q(1), Q(2)), (Q(2), Q(4)))) == (0,)
    with pytest.raises(ValueError, match="not invertible"):
        _modular_pivots(((Q(1, 65521),),))


def test_first_pivot_minor_has_independently_retained_modular_determinant(result):
    # Bareiss elimination checks the determinant independently of the owner's
    # normalized-row modular rank calculation.
    prime = 65521
    rows = [
        [
            value.numerator * pow(value.denominator, -1, prime) % prime
            for value in row[:160]
        ]
        for row in result.system.matrix
    ]
    previous, sign = 1, 1
    for column in range(159):
        pivot = next(i for i in range(column, 160) if rows[i][column])
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            sign = -sign
        value = rows[column][column]
        inverse = pow(previous, -1, prime)
        for i in range(column + 1, 160):
            multiplier = rows[i][column]
            for j in range(column + 1, 160):
                rows[i][j] = (
                    (value * rows[i][j] - multiplier * rows[column][j])
                    * inverse
                    % prime
                )
            rows[i][column] = 0
        previous = value
    assert sign * rows[-1][-1] % prime == 53426


@pytest.mark.parametrize("axis", range(4))
def test_cubic_phase_axes_have_correct_sign_mobility_and_clock(result, axis):
    system = result.system
    amplitude = Q(3, 7)
    state = (Q(0),) * 4 + tuple(amplitude if j == axis else Q(0) for j in range(4))
    actual = tuple(_evaluate(row, state) for row in system.cubic_field_pi)
    expected = (
        tuple(
            -system.quadratic.system.generator_pi[i][axis + 4] * amplitude**3 / 6
            for i in range(4)
        )
        + (Q(0),) * 4
    )
    assert actual == expected


def test_off_source_cubic_rows_use_actual_bare_fine_edges(result, source):
    state = tuple(Q(value, 17) for value in (-2, 3, 1, -4, 3, -2, 5, 1))
    block = (0, 0, 0, 0, 1, 2, 3, 3, 3, 4)
    phases = (Q(0),) + state[4:]
    lifted = tuple(phases[index] for index in block)
    raw = [Q(0)] * 10
    for i, j in source.edges:
        current = -((lifted[j] - lifted[i]) ** 3) / 6
        raw[i] += current
        raw[j] -= current
    rates = tuple(source.capacity[i] * raw[i] / source.degrees[i] for i in range(10))
    expected = tuple(rates[i] - rates[0] for i in (4, 5, 6, 9)) + (Q(0),) * 4
    assert (
        tuple(_evaluate(row, state) for row in result.system.cubic_field_pi) == expected
    )


@pytest.fixture(scope="module")
def independent_symbolic(result, source):
    import sympy as sp

    variables = sp.symbols("x_h x_u x_v x_d t_h t_u t_v t_d")
    phases = (sp.Integer(0),) + variables[4:]
    block = (0, 0, 0, 0, 1, 2, 3, 3, 3, 4)
    raw = [sp.Integer(0)] * 10
    energy_fourth = sp.Integer(0)
    for i, j in source.edges:
        gap = phases[block[j]] - phases[block[i]]
        raw[i] -= gap**3 / 6
        raw[j] += gap**3 / 6
        energy_fourth -= gap**4 / 12
    fine_rates = tuple(
        sp.Rational(source.capacity[i] / source.degrees[i]) * raw[i] for i in range(10)
    )
    cubic = tuple(fine_rates[i] - fine_rates[0] for i in (4, 5, 6, 9)) + (0,) * 4
    linear = sp.Matrix(result.system.quadratic.system.generator_pi) * sp.Matrix(
        variables
    )
    return sp, variables, cubic, linear, energy_fourth


def test_exact_solution_identity_rebuilt_with_independent_symbolic_fine_edges(
    result, independent_symbolic
):
    sp, variables, cubic, linear, _ = independent_symbolic
    polynomial = sum(
        sp.Rational(c) * sp.prod(x**n for x, n in zip(variables, exponent))
        for exponent, c in result.polynomial
    )
    quadratic = (
        sp.Matrix(variables).T
        * sp.Matrix(result.system.quadratic.invariant)
        * sp.Matrix(variables)
    )[0]
    identity = sum(
        sp.diff(polynomial, x) * linear[i] + sp.diff(quadratic, x) * cubic[i]
        for i, x in enumerate(variables)
    )
    assert sp.Poly(sp.expand(identity), *variables).is_zero


def test_known_energy_correction_independently_checks_homological_builder(
    result, independent_symbolic
):
    sp, variables, cubic, linear, energy_fourth = independent_symbolic
    twice_energy = (
        sp.Matrix(variables).T
        * sp.Matrix(result.system.quadratic.system.energy_metric)
        * sp.Matrix(variables)
    )[0]
    assert sp.Poly(
        sp.expand(
            sum(
                sp.diff(energy_fourth, x) * linear[i]
                + sp.diff(twice_energy, x) * cubic[i]
                for i, x in enumerate(variables)
            )
        ),
        *variables,
    ).is_zero
    expected = {
        exponent: Q(coefficient)
        for exponent, coefficient in sp.Poly(energy_fourth, *variables).terms()
    }
    image = _directional_derivative(
        expected, _linear_field(result.system.quadratic.system.generator_pi)
    )
    drift = _directional_derivative(
        _quadratic_polynomial(result.system.quadratic.system.energy_metric),
        result.system.cubic_field_pi,
    )
    assert image == {exponent: -coefficient for exponent, coefficient in drift.items()}


def test_actual_source_correction_and_full_sine_derivative_are_not_local_improvement(
    result, source
):
    mp = mpmath.mp.clone()
    mp.dps = 100

    def number(value):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator

    state = tuple(
        number(value) for value in result.system.quadratic.system.initial_state
    )
    polynomial = sum(
        number(c) * mp.fprod(x**n for x, n in zip(state, exponent))
        for exponent, c in result.polynomial
    )
    gradient = []
    for axis in range(8):
        value = mp.mpf(0)
        for exponent, coefficient in result.polynomial:
            if exponent[axis]:
                powers = list(exponent)
                powers[axis] -= 1
                value += (
                    number(coefficient)
                    * exponent[axis]
                    * mp.fprod(x**n for x, n in zip(state, powers))
                )
        gradient.append(value)
    raw = [[mp.mpf(0)] * 10 for _ in range(2)]
    for (i, j), weight in zip(source.edges, source.conductances):
        current = mp.sin(number(source.phase[j] - source.phase[i]))
        phase_rate = number(weight * (source.epi[i] - source.epi[j]))
        for row, value in zip(raw, (current, phase_rate)):
            row[i] += value
            row[j] -= value
    rates = tuple(
        tuple(
            number(source.capacity[i] / source.degrees[i]) * row[i] / mp.pi
            for i in range(10)
        )
        for row in raw
    )
    relative = tuple(row[i] - row[0] for row in rates for i in (4, 5, 6, 9))
    expected = sum(
        (
            gradient[i]
            + 2
            * sum(
                number(result.system.quadratic.invariant[i][j]) * state[j]
                for j in range(8)
            )
        )
        * relative[i]
        for i in range(8)
    )
    assert abs(polynomial - number(result.initial_correction)) < mp.mpf("1e-90")
    assert Q(29, 1000) < result.initial_correction < Q(3, 100)
    assert Q(328, 1000) < result.initial_corrected_witness < Q(329, 1000)
    interval = result.initial_corrected_derivative
    assert number(interval.lo) <= expected <= number(interval.hi)
    assert interval.lo > 20 * result.system.quadratic.sine_initial_derivative.hi > 0
    assert Q(29, 10000) < interval.lo < interval.hi < Q(3, 1000)


def test_domain_rebuilds_actual_storage_and_preserves_correlated_edge_bounds(
    result, source
):
    domain = result.domain
    mp = mpmath.mp.clone()
    mp.dps = 100

    def number(value):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator

    expected = sum(
        number(weight * (source.epi[i] - source.epi[j]) ** 2) / 2
        + 1
        - mp.cos(number(source.phase[i] - source.phase[j]))
        for (i, j), weight in zip(source.edges, source.conductances)
    )
    assert (
        number(domain.initial_sine_storage.lo)
        <= expected
        <= number(domain.initial_sine_storage.hi)
    )
    assert domain.initial_sine_storage.hi < Q(21, 50)
    energy = result.system.quadratic.system.energy_metric
    phase_inverse = exact_matrix_inverse(tuple(row[4:] for row in energy[4:]))
    block = (0, 0, 0, 0, 1, 2, 3, 3, 3, 4)
    resistances = []
    for i, j in source.edges:
        direction = tuple(
            Q(int(block[j] == k + 1) - int(block[i] == k + 1)) for k in range(4)
        )
        resistances.append(
            sum(
                direction[a] * phase_inverse[a][b] * direction[b]
                for a in range(4)
                for b in range(4)
            )
        )
    assert max(resistances) == Q(1, 3)
    assert Q(28, 97) < Q(3, 10)
    assert domain.phase_coercivity == Q(39, 40)
    inverse = exact_matrix_inverse(domain.ellipsoid_metric)
    for index, bound in enumerate(domain.coordinate_absolute_bounds):
        assert bound**2 >= 2 * domain.initial_sine_storage.hi * inverse[index][index]
    state = result.system.quadratic.system.initial_state
    actual_ellipsoid = sum(
        state[i] * domain.ellipsoid_metric[i][j] * state[j]
        for i in range(8)
        for j in range(8)
    )
    assert actual_ellipsoid < 2 * domain.initial_sine_storage.lo
    assert abs(result.initial_correction) < domain.correction_absolute_bound
    assert (
        result.initial_corrected_derivative.abs_max
        < domain.corrected_derivative_absolute_bound
    )
    assert not domain.absolute_bound_retains_threshold
    assert (
        domain.initial_observation_lower_bound
        == (result.initial_corrected_witness - domain.correction_absolute_bound)
        < Q(1, 4)
    )


def test_same_storage_target_rules_out_symmetric_transfer_beyond_box_looseness(result):
    domain = result.domain
    # Setting theta=0 and only relative x_H nonzero preserves the two
    # separately retained origins through the mean-fixed injection.
    energy = result.system.quadratic.system.energy_metric
    h = domain.initial_sine_storage
    coefficient = dict(result.polynomial)[(4,) + (0,) * 7]
    square_lo, square_hi = 2 * h.lo / energy[0][0], 2 * h.hi / energy[0][0]
    assert coefficient < 0
    target = domain.same_storage_target_correction
    assert target.contains(coefficient * square_lo**2)
    assert target.contains(coefficient * square_hi**2)
    assert target.hi < Q(-43, 100)
    assert domain.same_storage_target_observation.hi < Q(3, 50) < Q(1, 4)
    # Any symmetric absolute bound must contain this admitted same-H target.
    # Its required magnitude exceeds the fixed witness's initial margin.
    assert -target.hi > result.initial_corrected_witness - Q(1, 4)
    assert domain.correction_absolute_bound >= -target.lo


@pytest.mark.parametrize("change", ("invalid", "short", "gauge", "residual"))
def test_untrusted_candidate_rejected_by_admission_gauge_or_identity(result, change):
    candidate = list(result.coefficients)
    if change == "invalid":
        candidate[0] = True
    elif change == "short":
        candidate.pop()
    elif change == "gauge":
        candidate[-1] = 1
    else:
        candidate[0] += 1
    with pytest.raises((TypeError, ValueError)):
        _verify_correction(result.system, candidate)


def test_rechecks_source_before_backend_or_cached_candidate(preparation, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("source admission must precede a solve or cached answer")

    import tnfr.research.weighted_star_quartic as owner

    monkeypatch.setattr(owner, "_solve_fixed_system", forbidden)
    changed = deepcopy(preparation)
    changed["primitive"]["phase_lifts"] = (Q(0),) * 10
    with pytest.raises(ValueError, match="primitive source"):
        assess_weighted_star_quartic_transfer(changed)
    changed = deepcopy(preparation)
    changed["invented_cached_verdict"] = True
    with pytest.raises(
        ValueError, match="full retained preparation record association"
    ):
        build_weighted_star_quartic_system(changed)


def test_missing_optional_solver_is_unavailability_not_mathematical_obstruction(
    result, monkeypatch
):
    original = builtins.__import__

    def without_sympy(name, *args, **kwargs):
        if name == "sympy" or name.startswith("sympy."):
            raise ImportError("deliberately unavailable optional backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sympy)
    with pytest.raises(ImportError, match="optional sympy backend"):
        _solve_fixed_system.__wrapped__(
            result.system.matrix, result.system.right_hand_side
        )
    # A supplied mathematical candidate still receives complete verification.
    assert _verify_correction(result.system, result.coefficients) == result

"""Exact static reachability controls, without replaying events or trajectories."""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tnfr.dynamics import integrators
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics.krylov import exact_rank
from tnfr.operators import network_stage, word_execution
from tnfr.research import weighted_star_contact, weighted_star_finite_episode
from tnfr.research.artifact_io import decode_exact_tree, read_bytes_bounded
from tnfr.research.weighted_star_constitutive_handoff import (
    compare_weighted_star_constitutive_handoff,
)
from tnfr.research.weighted_star_reachability import (
    WITNESS_COEFFICIENTS,
    _sine_remainder,
    assess_weighted_star_quadratic_obstruction,
    build_weighted_star_relative_system,
)
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module", autouse=True)
def no_producer_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("the reachability audit must not replay a scientific producer")

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
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/weighted_star_registration_episode/preparation.json"
    )
    return decode_exact_tree(json_loads(read_bytes_bounded(path, max_bytes=256 * 1024)))


@pytest.fixture(scope="module")
def result(preparation):
    return assess_weighted_star_quadratic_obstruction(preparation)


@pytest.fixture(scope="module")
def source(preparation):
    return compare_weighted_star_constitutive_handoff(preparation).prefixes[-1].source


def _product(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _quadratic(matrix, vector):
    return sum((a * b for a, b in zip(vector, _product(matrix, vector))), Q(0))


def _lift(system, relative, origins=(Q(0), Q(0))):
    coarse = tuple(
        tuple(
            value + origins[channel]
            for value in _product(
                system.mean_fixed_injection, relative[channel * 4 : (channel + 1) * 4]
            )
        )
        for channel in range(2)
    )
    block_index = (0, 0, 0, 0, 1, 2, 3, 3, 3, 4)
    return tuple(tuple(values[block_index[i]] for i in range(10)) for values in coarse)


def test_chart_reconstructs_source_and_keeps_both_weighted_origins(result, source):
    system = result.system
    assert system.mean_weights == (Q(4), Q(32, 13), Q(5, 4), Q(9, 2), Q(5, 8))
    assert sum(system.mean_weights) == Q(1335, 104)
    assert system.initial_state == (
        Q(-63, 1024),
        0,
        0,
        Q(-1, 16),
        0,
        Q(201, 512),
        Q(201, 512),
        Q(201, 512),
    )
    assert _lift(system, system.initial_state, system.conserved_origins) == (
        source.epi,
        source.phase,
    )
    assert (
        system.initial_tangent_storage
        == _quadratic(system.energy_metric, system.initial_state) / 2
    )
    assert system.initial_tangent_storage < Q(1, 2)
    assert all(
        sum(
            system.mean_weights[i] * system.mean_fixed_injection[i][j] for i in range(5)
        )
        == 0
        for j in range(4)
    )
    # This injection has eight independent relative coordinates and two
    # separately retained origins, not a claimed reduction of arbitrary nodes.
    for j in range(4):
        assert tuple(
            system.mean_fixed_injection[i + 1][j] - system.mean_fixed_injection[0][j]
            for i in range(4)
        ) == tuple(Q(int(i == j)) for i in range(4))


@pytest.mark.parametrize("column", range(8))
def test_exact_generator_matches_all_full_tangent_basis_directions(
    result, source, column
):
    system = result.system
    relative = tuple(Q(int(i == column)) for i in range(8))
    form, phase = _lift(system, relative)
    fine_form, fine_phase = [Q(0)] * 10, [Q(0)] * 10
    for (i, j), weight in zip(source.edges, source.conductances):
        current = phase[j] - phase[i]
        gradient = weight * (form[i] - form[j])
        fine_form[i] += current
        fine_form[j] -= current
        fine_phase[i] += gradient
        fine_phase[j] -= gradient
    fine = tuple(
        tuple(source.capacity[i] * row[i] / source.degrees[i] for i in range(10))
        for row in (fine_form, fine_phase)
    )
    expected = tuple(row[node] - row[0] for row in fine for node in (4, 5, 6, 9))
    assert expected == tuple(row[column] for row in system.generator_pi)


def test_compact_mobility_energy_and_observation_match_independent_fine_sums(
    result, source
):
    system = result.system
    weights = dict(zip(source.edges, source.conductances))
    u, v = weights[4, 5], weights[4, 9]
    ex = (
        (8 + 4 * u + v, -u, -3 * u, -v),
        (-u, 11 + u, -9, -2),
        (-3 * u, -9, 15 + 3 * u, -6),
        (-v, -2, -6, 8 + v),
    )
    et = ((9, -1, -3, -1), (-1, 5, -3, -1), (-3, -3, 9, -3), (-1, -1, -3, 5))
    metric = tuple(
        tuple(Q(1, 4) + (value if i == j else 0) for j in range(4))
        for i, value in enumerate((Q(13, 32), Q(4, 5), Q(2, 9), Q(8, 5)))
    )
    upper, lower = exact_matrix_product(metric, et), exact_matrix_product(metric, ex)
    assert (
        tuple(tuple(-value for value in row[4:]) for row in system.generator_pi[:4])
        == upper
    )
    assert tuple(row[:4] for row in system.generator_pi[4:]) == lower
    relative = tuple(Q(i - 3, 16) for i in range(8))
    form, phase = _lift(system, relative, (Q(11, 4), Q(-17, 8)))
    q_value = (
        sum(
            (
                (row[i] - row[j]) ** 2
                for row in (form, phase)
                for i in range(10)
                for j in range(i + 1, 10)
            ),
            Q(0),
        )
        / 10
    )
    energy_twice = sum(
        (
            weight * (form[i] - form[j]) ** 2 + (phase[i] - phase[j]) ** 2
            for (i, j), weight in zip(source.edges, source.conductances)
        ),
        Q(0),
    )
    assert _quadratic(system.observation, relative) == q_value
    assert _quadratic(system.energy_metric, relative) == energy_twice


def test_four_polynomial_invariants_span_the_symmetric_lyapunov_kernel(result):
    system = result.system
    assert system.invariant_dimension == 4
    assert system.lyapunov_rank == 32
    assert 8 * 9 // 2 - system.lyapunov_rank == system.invariant_dimension
    for matrix in system.invariant_basis:
        assert matrix == tuple(zip(*matrix))
        product = exact_matrix_product(matrix, system.generator_pi)
        assert all(
            product[i][j] + product[j][i] == 0 for i in range(8) for j in range(8)
        )
    # A distinct observation quadratic is not automatically conserved.
    product = exact_matrix_product(system.observation, system.generator_pi)
    assert any(product[i][j] + product[j][i] != 0 for i in range(8) for j in range(8))


def test_sine_quadratic_obstruction_uses_block_structure_and_invertible_rows(result):
    system = result.system
    assert all(
        matrix[i][j] == 0
        for matrix in system.invariant_basis
        for i in range(8)
        for j in range(8)
        if (i < 4) != (j < 4)
    )
    b = tuple(tuple(-value for value in row[4:]) for row in system.generator_pi[:4])
    c = tuple(row[:4] for row in system.generator_pi[4:])
    assert exact_rank(b) == exact_rank(c) == 4
    # On each phase axis, the complete sine-minus-tangent field equals
    # B e_j (t-sin(t))/pi. Independent phase directions force S B=0 if
    # diag(S,T) were conserved for every form on an open state domain.
    mp = mpmath.mp.clone()
    mp.dps = 100
    amplitude = Q(1, 8)
    factor = (mp.mpf(1) / 8 - mp.sin(mp.mpf(1) / 8)) / mp.pi
    for axis in range(4):
        state = (Q(0),) * 4 + tuple(
            amplitude if index == axis else Q(0) for index in range(4)
        )
        residual = _sine_remainder(system, state)
        for row in range(4):
            expected = (
                factor * mp.mpf(b[row][axis].numerator) / b[row][axis].denominator
            )
            assert (
                mp.mpf(residual[row].lo.numerator) / residual[row].lo.denominator
                <= expected
                <= mp.mpf(residual[row].hi.numerator) / residual[row].hi.denominator
            )
        assert all(value == I(0) for value in residual[4:])


def test_factored_witness_strictly_dominates_the_unchanged_threshold(result):
    system = result.system
    assert result.coefficients == WITNESS_COEFFICIENTS
    square = exact_matrix_product(system.generator_pi, system.generator_pi)
    factored = system.energy_metric
    for denominator in (30, 50, 135):
        factor = tuple(
            tuple(Q(int(i == j)) + square[i][j] / denominator for j in range(8))
            for i in range(8)
        )
        factored = exact_matrix_product(factored, factor)
    factored = tuple(tuple(Q(3, 4) * value for value in row) for row in factored)
    assert result.invariant == factored
    assert result.domination_certified and result.tangent_exclusion_certified
    assert exact_symmetric_semidefinite(result.domination_margin, strict=True)
    assert result.initial_invariant == _quadratic(factored, system.initial_state)
    assert result.initial_invariant > Q(299, 1000) > Q(1, 4)
    assert result.threshold_margin == result.initial_invariant - Q(1, 4)


def test_nonlinear_drift_is_positive_at_the_actual_source(result, source):
    mp = mpmath.mp.clone()
    mp.dps = 100

    def value(number):
        number = Q(number)
        return mp.mpf(number.numerator) / number.denominator

    full_x, full_theta = [mp.mpf(0)] * 10, [mp.mpf(0)] * 10
    for (i, j), weight in zip(source.edges, source.conductances):
        current = mp.sin(value(source.phase[j] - source.phase[i]))
        gradient = value(weight * (source.epi[i] - source.epi[j]))
        full_x[i] += current
        full_x[j] -= current
        full_theta[i] += gradient
        full_theta[j] -= gradient
    rates = tuple(
        tuple(
            value(source.capacity[i] / source.degrees[i]) * row[i] / mp.pi
            for i in range(10)
        )
        for row in (full_x, full_theta)
    )
    relative_rate = tuple(row[node] - row[0] for row in rates for node in (4, 5, 6, 9))
    expected = 2 * sum(
        value(result.system.initial_state[i])
        * value(result.invariant[i][j])
        * relative_rate[j]
        for i in range(8)
        for j in range(8)
    )
    derivative = result.sine_initial_derivative
    assert value(derivative.lo) <= expected <= value(derivative.hi)
    assert Q(1, 10000) < derivative.lo < derivative.hi < Q(1, 5000)
    assert Q(-1, 20) < result.sine_initial_drift_coefficient < Q(-1, 25)
    assert result.nonlinear_invariance_excluded
    # Independently evaluated full remainder agrees with the factored source
    # formula, retaining the original structural clock divisor.
    remainder = _sine_remainder(result.system, result.system.initial_state)
    generic = 2 * sum(
        (
            result.system.initial_state[i] * result.invariant[i][j] * remainder[j]
            for i in range(8)
            for j in range(8)
        ),
        I(0),
    )
    assert generic.contains(derivative)


@pytest.mark.parametrize("coefficients", ((0, 0, 0, 0), (10, 0, 0, 0)))
def test_failed_quadratic_candidate_does_not_certify_exclusion(
    preparation, coefficients
):
    result = assess_weighted_star_quadratic_obstruction(
        preparation, coefficients=coefficients
    )
    assert not result.tangent_exclusion_certified
    if coefficients[0] == 0:
        assert result.domination_certified and result.initial_invariant == 0
        assert not result.nonlinear_invariance_excluded
    else:
        assert not result.domination_certified and result.initial_invariant > Q(1, 4)


@pytest.mark.parametrize(
    "coefficients",
    (
        (True, 0, 0, 0),
        (float("nan"), 0, 0, 0),
        (float("inf"), 0, 0, 0),
        (1, 2, 3),
        (1, 2, 3, 4, 5),
        "1234",
        {1, 2, 3, 4},
    ),
)
def test_invalid_coefficients_reject(preparation, coefficients):
    with pytest.raises((TypeError, ValueError)):
        assess_weighted_star_quadratic_obstruction(
            preparation, coefficients=coefficients
        )


def test_rebuilds_primitives_and_full_source_association_before_certifying(preparation):
    changed = deepcopy(preparation)
    changed["primitive"]["phase_lifts"] = (Q(0),) * 10
    with pytest.raises(ValueError, match="primitive source"):
        build_weighted_star_relative_system(changed)
    changed = deepcopy(preparation)
    changed["invented_cached_verdict"] = True
    with pytest.raises(
        ValueError, match="full retained preparation record association"
    ):
        assess_weighted_star_quadratic_obstruction(changed)

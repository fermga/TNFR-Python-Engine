"""Analytical relaxation controls, without sampled designs or trajectories."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import I, cos
from tnfr.physics._organized_law_observation import (
    _bound_organized_law_relaxation_obstruction,
    _bound_organized_observation_law,
)


@pytest.fixture(scope="module")
def report():
    return _bound_organized_law_relaxation_obstruction()


def test_witness_solves_independent_even_moment_cancellation_system(report):
    matrix = ((Q(9**2), Q(18**2)), (Q(9**4), Q(18**4)))
    solved = exact_matrix_product(exact_matrix_inverse(matrix), ((Q(-1),), (Q(-1),)))
    assert tuple(row[0] for row in solved) == report.uniform_edge_sine_coefficients[1:]
    assert report.uniform_edge_sine_coefficients[0] == 1
    assert report.second_moment == report.fourth_moment == 0
    sixth = sum(
        abs(a) * n**6
        for n, a in zip(report.harmonics, report.uniform_edge_sine_coefficients)
    )
    assert report.sixth_absolute_moment == sixth
    assert report.forcing_upper == sixth * report.command_gap_upper**6 / 720


@pytest.mark.parametrize("child", range(5))
@pytest.mark.parametrize("command", (Q(-1, 10), Q(0), Q(1, 20), Q(1, 10)))
def test_uniform_circulation_leaves_only_the_oriented_neighbor_dipole(
    report, child, command
):
    phase = tuple(command * (Q(int(i == child)) - Q(1, 5)) for i in range(5))
    field = [I(0)] * 5
    for tail in range(5):
        head = (tail + 1) % 5
        current = sum(
            (
                coefficient * cos(order * (phase[head] - phase[tail]))
                for order, coefficient in zip(
                    report.harmonics, report.uniform_edge_sine_coefficients
                )
            ),
            I(0),
        )
        field[tail] += current
        field[head] -= current
    forcing = sum(
        (
            coefficient * (cos(order * command) - 1)
            for order, coefficient in zip(
                report.harmonics, report.uniform_edge_sine_coefficients
            )
        ),
        I(0),
    )
    assert forcing.abs_max <= report.forcing_upper
    for node, value in enumerate(field):
        expected = (
            forcing
            if node == (child - 1) % 5
            else -forcing if node == (child + 1) % 5 else I(0)
        )
        difference = value - expected
        assert difference.contains(0)
        assert difference.width < Q(1, 10**30)


def test_declared_heat_bound_follows_from_exact_c5_spectral_control(report):
    laplacian = tuple(
        tuple(
            Q(2) if i == j else Q(-1) if (i - j) % 5 in (1, 4) else Q(0)
            for j in range(5)
        )
        for i in range(5)
    )
    squared = exact_matrix_product(laplacian, laplacian)
    # L^2-5L+5I is the all-ones matrix, so the nonzero eigenvalues satisfy
    # the independently derived polynomial r^2-5r+5=0.
    assert (
        tuple(
            tuple(squared[i][j] - 5 * laplacian[i][j] + 5 * Q(i == j) for j in range(5))
            for i in range(5)
        )
        == ((Q(1),) * 5,) * 5
    )
    relative_basis = tuple(
        tuple(Q(int(i == j)) - Q(int(i == 4)) for j in range(4)) for i in range(5)
    )
    shifted = tuple(
        tuple(laplacian[i][j] - Q(4, 3) * Q(i == j) for j in range(5)) for i in range(5)
    )
    compressed = exact_matrix_product(
        tuple(zip(*relative_basis)), exact_matrix_product(shifted, relative_basis)
    )
    assert exact_symmetric_semidefinite(compressed, strict=True)
    # Hence exp(-tL/4) contracts the zero-sum Euclidean subspace faster
    # than exp(-t/3). Its integrated factor is <=3 and ||dipole||=sqrt(2)<3/2.
    assert Q(3, 2) ** 2 > 2
    assert report.dipole_heat_factor == 3 * Q(3, 2)
    assert (
        report.sampled_response_upper
        == report.forcing_upper * report.dipole_heat_factor
    )


def test_own_law_errors_and_nonlinear_blocks_are_rebuilt_from_their_primitives(report):
    source = tuple(
        _bound_organized_observation_law(deformation=value) for value in (0, 1)
    )
    assert report.source_errors == tuple(
        item.source_form_error_upper for item in source
    )
    assert report.historical_pair_error == 2 * report.reading_error + sum(
        report.source_errors
    )
    g, duration, initial = (
        source[1].gamma_upper,
        source[1].duration,
        source[1].initial_form_upper,
    )
    drift = 4 * g * initial + 2 * g**2 * source[1].current_upper * duration
    beta = 6 * g**2 * duration
    assert report.edge_drift_upper == drift
    assert report.continuous_sample_margin == 1 - 2 * g**2 * duration
    assert report.remainder_record_factor * report.continuous_sample_margin == beta
    for harmonic, factor in zip(report.harmonics, report.remainder_block_factors):
        assert (
            factor
            == 18 * harmonic * drift
            + beta * 6 * (1 + harmonic * drift) / report.continuous_sample_margin
        )
    assert report.elementary_gap == 18 * drift - report.sampled_response_upper > 0
    assert (
        report.block_feedback_gap
        == sum(
            a * b for a, b in zip(report.remainder_block_factors, report.block_norms)
        )
        - report.sampled_response_upper
        > 0
    )
    assert report.obstruction_established


def test_block_feedback_supervector_obstructs_every_positive_rescaling(report):
    # The extremal scalar response allowance gives k=b/m and E=0. Its
    # positive rank-one feedback already has b as an expanding eigenvector.
    # The proof for arbitrary duals uses b<=Eb+k*m, not any assumed inverse.
    blocks = report.block_norms
    multiplier = (
        sum(a * b for a, b in zip(report.remainder_block_factors, blocks))
        / report.sampled_response_upper
    )
    assert multiplier > 1
    weights = (Q(1, 9), Q(7), Q(2, 3))
    scaled_blocks = tuple(weight * b for weight, b in zip(weights, blocks))
    feedback = tuple(
        tuple(
            weights[i]
            * blocks[i]
            * report.remainder_block_factors[j]
            / (report.sampled_response_upper * weights[j])
            for j in range(3)
        )
        for i in range(3)
    )
    image = exact_matrix_product(feedback, tuple((value,) for value in scaled_blocks))
    assert tuple(row[0] for row in image) == tuple(
        multiplier * value for value in scaled_blocks
    )
    # The witness is formal: alternative harmonic phasor radii must have
    # r18/r9=1/2, which these pure sine coefficient pairs do not satisfy.
    assert 2 * blocks[2] != blocks[1]

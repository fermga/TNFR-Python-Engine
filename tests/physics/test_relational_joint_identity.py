"""Exact geometry and full-field controls for joint pair identification.

The dynamical bound uses eight scaled real-lift coordinates. It does not
differentiate an embedded circular-state field or sample a reference orbit.
"""

import pytest


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("sympy")


def test_joint_distance_is_the_constant_reference_energy_separation(symbolic):
    s = symbolic
    xa, xb, pa, pb, gap, contrast = s.symbols("XA XB PA PB delta D", real=True)
    beta, nu, weight = s.symbols("beta nu w", positive=True)
    left = s.Matrix((xa, s.sqrt(beta) * s.cos(pa), s.sqrt(beta) * s.sin(pa)))
    right = s.Matrix((xb, s.sqrt(beta) * s.cos(pb), s.sqrt(beta) * s.sin(pb)))
    metric_squared = s.trigsimp((left - right).dot(left - right))
    joint = (xa - xb) ** 2 + 2 * beta * (1 - s.cos(pb - pa))
    assert s.trigsimp(metric_squared - joint) == 0

    separation = contrast**2 + 2 * beta * (1 - s.cos(gap))
    contrast_rate = 2 * weight * nu * s.sin(gap) / s.pi
    gap_rate = -2 * weight * nu * contrast / (beta * s.pi)
    derivative = (
        s.diff(separation, contrast) * contrast_rate
        + s.diff(separation, gap) * gap_rate
    )
    assert s.simplify(derivative) == 0
    coarse_storage = contrast**2 / 2 + beta * (1 - s.cos(gap))
    assert s.expand(separation - 2 * coarse_storage) == 0
    # Phase coincidence removes the phase-only distinction while retaining
    # the complete form contribution; the converse holds at equal form.
    assert separation.subs(gap, 0) == contrast**2
    assert separation.subs(contrast, 0) == 2 * beta * (1 - s.cos(gap))
    assert separation.subs({gap: 0, contrast: 0}) == 0


def test_scaled_lift_full_jacobian_has_the_claimed_symmetric_blocks(symbolic):
    s = symbolic
    edges = ((0, 2), (0, 3), (1, 2), (1, 3))
    forms = s.Matrix(s.symbols("x0:4", real=True))
    lifted_phases = s.Matrix(s.symbols("y0:4", real=True))
    beta, rate = s.symbols("beta c", positive=True)
    basis = s.eye(4)
    laplacian = s.zeros(4)
    cosine_laplacian = s.zeros(4)
    form_field = s.zeros(4, 1)
    for left, right in edges:
        incidence = basis[:, left] - basis[:, right]
        phase_gap = (lifted_phases[right] - lifted_phases[left]) / s.sqrt(beta)
        laplacian += incidence * incidence.T
        cosine_laplacian += s.cos(phase_gap) * incidence * incidence.T
        form_field += rate * s.sqrt(beta) * s.sin(phase_gap) * incidence / 2
    phase_field = rate * laplacian * forms / 2
    field = form_field.col_join(phase_field)
    jacobian = field.jacobian(forms.col_join(lifted_phases))
    expected = (
        s.zeros(4)
        .row_join(-rate * cosine_laplacian / 2)
        .col_join((rate * laplacian / 2).row_join(s.zeros(4)))
    )
    assert (jacobian - expected).applyfunc(s.simplify) == s.zeros(8)
    assert cosine_laplacian == cosine_laplacian.T
    assert laplacian.eigenvals() == {0: 1, 2: 2, 4: 1}

    # The signed-cosine block is dominated on both sides by L, because each
    # edge coefficient 1 +/- cos(gap) is nonnegative. No sampled spectrum is
    # used to replace this full-state bound.
    for sign in (-1, 1):
        edge_decomposition = s.zeros(4)
        for left, right in edges:
            incidence = basis[:, left] - basis[:, right]
            phase_gap = (lifted_phases[right] - lifted_phases[left]) / s.sqrt(beta)
            edge_decomposition += (
                (1 + sign * s.cos(phase_gap)) * incidence * incidence.T
            )
        residual = laplacian + sign * cosine_laplacian - edge_decomposition
        assert residual.applyfunc(s.expand) == s.zeros(4)
    # Orthogonal block placement gives ||DF||=c*max(||L/2||,||Lcos/2||),
    # rather than the sum of block norms.
    assert (expected.T * expected)[:4, 4:] == s.zeros(4)
    assert (expected.T * expected)[4:, :4] == s.zeros(4)


def test_embedding_error_and_pair_error_use_lift_norm_without_losing_factors(symbolic):
    s = symbolic
    phase, phase_error, form_error = s.symbols("theta h f", real=True)
    beta = s.symbols("beta", positive=True)
    embedded_difference = s.Matrix(
        (
            form_error,
            s.sqrt(beta) * (s.cos(phase + phase_error) - s.cos(phase)),
            s.sqrt(beta) * (s.sin(phase + phase_error) - s.sin(phase)),
        )
    )
    exact_error = form_error**2 + 2 * beta * (1 - s.cos(phase_error))
    assert s.trigsimp(embedded_difference.dot(embedded_difference) - exact_error) == 0
    assert (
        s.trigsimp(exact_error - form_error**2 - 4 * beta * s.sin(phase_error / 2) ** 2)
        == 0
    )

    first = s.Matrix(s.symbols("a0:3", real=True))
    second = s.Matrix(s.symbols("b0:3", real=True))
    pair_error_squared = (first - second).dot(first - second)
    bound_squared = 2 * (first.dot(first) + second.dot(second))
    residual = bound_squared - pair_error_squared
    assert s.expand(residual - (first + second).dot(first + second)) == 0
    # Opposite endpoint errors attain the factor sqrt(2), so omitting it
    # would silently invalidate an all-node preparation certificate.
    opposite = dict(zip(second, -first))
    assert s.expand(residual.subs(opposite)) == 0

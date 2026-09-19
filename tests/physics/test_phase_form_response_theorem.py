"""Conditional exact EPI response under identical prescribed phase inputs.

The common mean remains neutral between preparations. These identities do
not perturb the input, derive its clock, or certify an autonomous attractor.
No trajectory, numerical eigensolver, or new evolution API is used.
"""

import pytest

from tests.physics._internal_mode_fixture import Q_MODE, P, _exact_generator, _graph
from tnfr.physics.support_transport import observe_support_transport


def test_general_prescribed_source_has_exact_variation_of_constants_response():
    s = pytest.importorskip("sympy")
    e, weight = s.symbols("e weight", positive=True)
    time = s.Symbol("time", nonnegative=True)
    integration_time = s.Dummy("integration_time", real=True)
    initial_form, initial_mean = s.symbols("initial_form initial_mean")
    source, common = s.Function("source"), s.Function("common")
    response = s.exp(-e * time) * (
        initial_form
        - weight
        / s.pi
        * s.Integral(
            s.exp(e * integration_time) * source(integration_time),
            (integration_time, 0, time),
        )
    )
    mean = initial_mean + weight * s.Integral(
        common(integration_time), (integration_time, 0, time)
    )
    assert (
        s.simplify(s.diff(response, time) + e * response + weight * source(time) / s.pi)
        == 0
    )
    assert s.simplify(s.diff(mean, time) - weight * common(time)) == 0
    assert response.subs(time, 0).doit() == initial_form
    assert mean.subs(time, 0).doit() == initial_mean
    switched_off = s.exp(-e * time) * initial_form
    assert s.diff(switched_off, time) == -e * switched_off
    # source(t) is the independently supplied primitive contrast and
    # common(t) its full phasor mean. They are not fitted from this response.
    # Holding both at zero leaves exponential form decay and a fixed mean.


def test_same_source_repeated_sector_contracts_only_transverse_to_the_mean():
    s = pytest.importorskip("sympy")
    source = observe_support_transport(_graph())
    generator = s.Matrix(_exact_generator(source))
    basis = s.Matrix.hstack(
        s.ones(6, 1), s.Matrix(P * 2) / s.sqrt(2), s.Matrix(Q_MODE * 2) / s.sqrt(6)
    )
    assert generator * basis == basis * s.diag(0, -1, -1)
    assert 3 * basis.T * basis == s.diag(18, 6, 6)
    e = s.Symbol("e", positive=True)
    time = s.Symbol("time", nonnegative=True)
    mean, real, imaginary = s.symbols("mean real imaginary", real=True)
    coordinates = s.Matrix(
        [mean, s.exp(-e * time) * real, s.exp(-e * time) * imaginary]
    )
    difference = basis * coordinates
    assert (difference.diff(time) - e * generator * difference).applyfunc(
        s.simplify
    ) == s.zeros(6, 1)
    squared = (3 * difference.T * difference)[0]
    assert (
        s.simplify(
            squared - 18 * mean**2 - 6 * s.exp(-2 * e * time) * (real**2 + imaginary**2)
        )
        == 0
    )
    assert s.limit(squared, time, s.oo) == 18 * mean**2
    # Subtracting the nodal equations cancels the SAME complete source.
    # Source, phase-law, capacity, support, and metric perturbations are not
    # covered by this conditional comparison of two EPI preparations.


def test_full_prism_block_response_retains_all_same_source_preparation_errors():
    s = pytest.importorskip("sympy")
    source = observe_support_transport(_graph())
    generator = s.Matrix(_exact_generator(source))
    p, q = s.Matrix(P), s.Matrix(Q_MODE)
    basis = s.Matrix.hstack(
        s.ones(6, 1),
        s.Matrix([1, 1, 1, -1, -1, -1]),
        p.col_join(p) / s.sqrt(2),
        q.col_join(q) / s.sqrt(6),
        p.col_join(-p) / s.sqrt(2),
        q.col_join(-q) / s.sqrt(6),
    )
    rates = s.diag(0, -s.Rational(2, 3), -1, -1, -s.Rational(5, 3), -s.Rational(5, 3))
    assert generator * basis == basis * rates
    assert basis.det() != 0
    assert 3 * basis.T * basis == s.diag(18, 18, 6, 6, 6, 6)
    f0, f1, f2 = s.symbols("f0 f1 f2", real=True)
    repeated_force = s.Matrix([f0, f1, f2] * 2)
    projected_force = basis.inv() * repeated_force
    assert projected_force[0] == (f0 + f1 + f2) / 3
    assert projected_force[1] == projected_force[4] == projected_force[5] == 0
    assert s.simplify(projected_force[2] - (f0 - f1) / s.sqrt(2)) == 0
    assert s.simplify(projected_force[3] - (f0 + f1 - 2 * f2) / s.sqrt(6)) == 0
    # This derives the mu,d,Z,D rates from the actual fine conductance.
    # A repeated source has no d or D component, while equal prescribed
    # sources cancel completely in the difference of two preparations.
    e = s.Symbol("e", positive=True)
    time = s.Symbol("time", nonnegative=True)
    mu, d, zr, zi, dr, di = s.symbols("mu d zr zi dr di", real=True)
    initial = (mu, d, zr, zi, dr, di)
    evolved = s.Matrix(
        [value * s.exp(e * rates[i, i] * time) for i, value in enumerate(initial)]
    )
    difference = basis * evolved
    assert (difference.diff(time) - e * generator * difference).applyfunc(
        s.simplify
    ) == s.zeros(6, 1)
    expected = (
        18 * mu**2
        + 18 * s.exp(-4 * e * time / 3) * d**2
        + 6 * s.exp(-2 * e * time) * (zr**2 + zi**2)
        + 6 * s.exp(-10 * e * time / 3) * (dr**2 + di**2)
    )
    assert s.simplify((3 * difference.T * difference)[0] - expected) == 0
    assert s.limit(expected, time, s.oo) == 18 * mu**2
    # With a=exp(-2*e*time/3) in (0,1], nonuniform squared norm is bounded
    # by a^2 times its initial value. The factorization shows every gap term.
    a = s.Symbol("a", nonnegative=True)
    nonuniform_initial = 18 * d**2 + 6 * (zr**2 + zi**2 + dr**2 + di**2)
    nonuniform_later = (
        18 * a**2 * d**2 + 6 * a**3 * (zr**2 + zi**2) + 6 * a**5 * (dr**2 + di**2)
    )
    gap = 6 * a**2 * (1 - a) * (zr**2 + zi**2 + (1 + a + a**2) * (dr**2 + di**2))
    assert s.expand(a**2 * nonuniform_initial - nonuniform_later - gap) == 0
    # Norm contraction rate 2e/3 is attained by the intertriangle mean mode;
    # the ordinary mean is neutral. This does not make the whole driven
    # response an isolated attractor or establish autonomy of its source.

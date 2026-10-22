"""Exact driven form/phase geometry on the existing unit triangular prism.

These are projected identities and conditional curve controls. Primitive
phase and the angular orientation of internal EPI are distinct coordinates.
No new phase law, integration, amplitude normalization, or autonomous NFR is
introduced. Represented source capture and ideal circle algebra stay separate.
"""

import math
from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import (
    INDUCED,
    LIFT,
    NODES,
    PROJECTION,
    Q_MODE,
    P,
    _apply,
    _exact_generator,
    _graph,
    _inner,
    _phase_support,
    _unit_regional_budget,
)
from tnfr.mathematics.phasor_resultant import reduce_phasor_components
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.support_transport import observe_support_transport


def test_arbitrary_driven_internal_projections_resolve_radial_and_tangential_work():
    s = pytest.importorskip("sympy")
    graph = _graph((Q(1, 8), Q(1, 16), Q(-1, 16), Q(1, 8)))
    graph.graph["DNFR_WEIGHTS"] = dict(epi=0.5, phase=0.25, vf=0.25, topo=0)
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = (0, math.pi / 3, math.pi / 6)[i]
    captured = capture_non_epi_forcing(graph)
    coefficients = _apply(PROJECTION, captured.snapshot.epi)
    force = _apply(PROJECTION, captured.forcing)
    passive = _apply(INDUCED, coefficients)
    rates = tuple(
        captured.epi_weight * p + f for p, f in zip(passive, force, strict=True)
    )
    norm, rate, work = _unit_regional_budget(
        captured.snapshot, captured.forcing, e=captured.epi_weight
    )
    assert rate == 2 * sum(
        _inner(coefficients[i : i + 2], rates[i : i + 2]) for i in (0, 2)
    )
    assert work == 2 * sum(
        _inner(coefficients[i : i + 2], force[i : i + 2]) for i in (0, 2)
    )
    radial_work = s.Integer(0)
    for i in (0, 2):
        z = s.sqrt(2) * coefficients[i] + s.I * s.sqrt(6) * coefficients[i + 1]
        drive = s.sqrt(2) * force[i] + s.I * s.sqrt(6) * force[i + 1]
        velocity = s.sqrt(2) * rates[i] + s.I * s.sqrt(6) * rates[i + 1]
        radius = s.sqrt(s.expand_complex(s.conjugate(z) * z))
        radial = s.simplify(s.re(s.conjugate(z) * drive) / radius)
        tangent = s.simplify(s.im(s.conjugate(z) * drive) / radius)
        assert s.simplify(drive - (z / radius) * (radial + s.I * tangent)) == 0
        other = 2 - i
        neighbor = (
            s.sqrt(2) * coefficients[other] + s.I * s.sqrt(6) * coefficients[other + 1]
        )
        local_neighbor = s.conjugate(z / radius) * neighbor
        radius_rate = s.re(s.conjugate(z) * velocity) / radius
        angle_rate = s.im(s.conjugate(z) * velocity) / radius**2
        assert (
            s.simplify(
                radius_rate
                - captured.epi_weight * (s.re(local_neighbor) - 4 * radius) / 3
                - radial
            )
            == 0
        )
        assert (
            s.simplify(
                angle_rate
                - captured.epi_weight * s.im(local_neighbor) / (3 * radius)
                - tangent / radius
            )
            == 0
        )
        radial_work += 2 * radius * radial
    assert s.simplify(radial_work - work) == 0
    assert norm > 0

    # A purely tangential conditional source changes orientation but supplies
    # no instantaneous radial work. Use rational coefficient-space rotations.
    transverse = tuple(
        value for i in (0, 2) for value in (-3 * coefficients[i + 1], coefficients[i])
    )
    assert all(
        _inner(coefficients[i : i + 2], transverse[i : i + 2]) == 0 for i in (0, 2)
    )
    _, transverse_rate, transverse_work = _unit_regional_budget(
        captured.snapshot, _apply(LIFT, transverse), e=captured.epi_weight
    )
    assert transverse_work == 0 and transverse_rate < 0


def test_repeated_primitive_phase_contrast_drives_form_and_mean_independently():
    s = pytest.importorskip("sympy")
    graph, rows = _phase_support()
    u, v, phase_s, phase_t, mean, common = s.symbols(
        "u v phase_s phase_t mean common", real=True
    )
    e, weight = s.symbols("e weight", positive=True)
    p, q = s.Matrix(P * 2), s.Matrix(Q_MODE * 2)
    phase_contrast = phase_s * p + phase_t * q
    for row in rows:
        assert sorted(NODES[i][1] for i in row) == [0, 1, 2]
    # On the common regular lift, every row has the same resultant angle.
    # common=Arg(sum(exp(i*eta_j)))/pi; it must not be silently set to zero.
    phase_source = -phase_contrast / s.pi + common * s.ones(6, 1)
    field = mean * s.ones(6, 1) + u * p + v * q
    generator = s.Matrix(_exact_generator(observe_support_transport(graph)))
    velocity = e * generator * field + weight * phase_source
    mean_rate = (s.ones(1, 6) * velocity)[0] / 6
    projected_rate = s.Matrix(PROJECTION) * velocity
    z = s.sqrt(2) * u + s.I * s.sqrt(6) * v
    zeta = s.sqrt(2) * phase_s + s.I * s.sqrt(6) * phase_t
    z_rate = s.sqrt(2) * projected_rate[0] + s.I * s.sqrt(6) * projected_rate[1]
    assert s.simplify(mean_rate - weight * common) == 0
    assert s.simplify(z_rate + e * z + weight * zeta / s.pi) == 0
    assert projected_rate[:2] == projected_rate[2:]
    assert s.simplify(z_rate.subs({u: 0, v: 0}) + weight * zeta / s.pi) == 0
    assert z_rate.subs({u: 0, v: 0, phase_s: 1, phase_t: 0}) != 0
    zero = reduce_phasor_components(((0.0, 0.0),))
    assert zero.joint_zero and zero.angle is None
    # A Cartesian derivative can seed a zero mode, while its current angle
    # remains undefined. The source itself still needs a justified origin.


def test_mean_zero_geometry_has_three_lines_and_a_cubic_angular_dependence():
    s = pytest.importorskip("sympy")
    phase_s, phase_t, scale, angle = s.symbols("phase_s phase_t scale angle", real=True)
    rho = s.Symbol("rho", positive=True)
    eta = (phase_s + phase_t, -phase_s + phase_t, -2 * phase_t)
    assert sum(eta) == 0
    assert (
        s.trigsimp(
            sum(s.sin(value) for value in eta)
            + 4 * s.prod(s.sin(value / 2) for value in eta)
        )
        == 0
    )
    coordinate = s.Symbol("coordinate", real=True)
    assert s.solveset(
        s.sin(coordinate / 2), coordinate, domain=s.Interval.open(-s.pi / 3, s.pi / 3)
    ) == s.FiniteSet(0)
    # Centering plus phase spread<pi/2 bounds every eta_j in (-pi/3,pi/3).
    # Its real resultant is positive; Arg is zero iff the sine product is 0.
    product = s.factor(s.prod(eta))
    assert product == 2 * phase_t * (phase_s - phase_t) * (phase_s + phase_t)
    assert set(s.solve(product, phase_t)) == {0, -phase_s, phase_s}
    real = sum(s.cos(scale * value) for value in eta)
    imaginary = sum(s.sin(scale * value) for value in eta)
    mean_source = s.atan(imaginary / real) / s.pi
    cubic = s.series(mean_source, scale, 0, 5).removeO()
    assert s.simplify(cubic + scale**3 * product / (6 * s.pi)) == 0
    polar = {
        phase_s: rho * s.cos(angle) / s.sqrt(2),
        phase_t: rho * s.sin(angle) / s.sqrt(6),
    }
    moment = sum(value**3 for value in eta).subs(polar)
    assert (
        s.trigsimp(
            s.expand(s.expand_trig(moment - rho**3 * s.sin(3 * angle) / s.sqrt(6)))
        )
        == 0
    )
    assert (
        s.trigsimp(
            s.expand(
                s.expand_trig(
                    cubic.subs(polar)
                    + scale**3 * rho**3 * s.sin(3 * angle) / (18 * s.pi * s.sqrt(6))
                )
            )
        )
        == 0
    )


def test_constant_radius_requires_source_lag_and_fixed_mean_blocks_uniform_rotation():
    s = pytest.importorskip("sympy")
    e, weight, radius, omega = s.symbols("e weight radius omega", positive=True)
    psi = s.Symbol("psi", real=True)
    z = radius * (s.cos(psi) + s.I * s.sin(psi))
    required = -s.pi * (e + s.I * omega) * z / weight
    assert s.simplify(-e * z - weight * required / s.pi - s.I * omega * z) == 0
    # At a mean-zero source line zeta=rho is real, a uniform rotation has
    # eta=(a,-a,0), eta_dot=(b,b,-2b), b=omega*rho/sqrt(6).
    a, b = s.symbols("a b", positive=True)
    eta = (a, -a, s.Integer(0))
    eta_rate = (b, b, -2 * b)
    imaginary_rate = sum(
        s.cos(value) * rate for value, rate in zip(eta, eta_rate, strict=True)
    )
    assert s.simplify(imaginary_rate - 2 * b * (s.cos(a) - 1)) == 0
    assert s.calculus.util.function_range(
        s.cos(a), a, s.Interval.open(0, s.pi / 4)
    ) == s.Interval.open(s.sqrt(2) / 2, 1)
    # This nonzero imaginary rate leaves mean-zero immediately. Continuous
    # nonzero source contrast cannot switch between its three lines without
    # crossing their common zero; constant radius and e>0 exclude that zero.

    # Transient turning is NOT excluded: a fixed source line can accompany
    # psi_dot=-e*tan(psi) with a changing source magnitude (cos(psi)!=0).
    transient_rate = -e * s.tan(psi)
    transient_source = -s.pi * (e + s.I * transient_rate) * z / weight
    assert (
        s.trigsimp(
            s.expand_complex(transient_source)
            + s.pi * e * radius / (weight * s.cos(psi))
        )
        == 0
    )


def test_allowing_the_derived_mean_preserves_a_supplied_rotating_form_cycle():
    s = pytest.importorskip("sympy")
    rho = s.Symbol("rho", positive=True)
    angle = s.Symbol("angle", real=True)
    eta = s.Matrix(
        [
            rho * s.cos(angle) / s.sqrt(2) + rho * s.sin(angle) / s.sqrt(6),
            -rho * s.cos(angle) / s.sqrt(2) + rho * s.sin(angle) / s.sqrt(6),
            -2 * rho * s.sin(angle) / s.sqrt(6),
        ]
    )
    rotated = eta.subs(angle, angle + 2 * s.pi / 3)
    assert (rotated - s.Matrix([eta[2], eta[0], eta[1]])).applyfunc(
        s.expand_trig
    ).applyfunc(s.simplify) == s.zeros(3, 1)
    assert (eta.subs(angle, angle + s.pi) + eta).applyfunc(s.expand_trig).applyfunc(
        s.simplify
    ) == s.zeros(3, 1)
    # Permutation preserves the full phasor sum; half-turn conjugates it.
    # On Re Z>0, c=atan(Im Z/Re Z)/pi inherits period 2pi/3 and odd half-turn.
    real = s.Symbol("real", positive=True)
    imaginary = s.Symbol("imaginary", real=True)
    assert s.atan(-imaginary / real) == -s.atan(imaginary / real)
    basis = s.Matrix(
        [
            [1 / s.sqrt(2), 1 / s.sqrt(6)],
            [-1 / s.sqrt(2), 1 / s.sqrt(6)],
            [0, -2 / s.sqrt(6)],
        ]
    )
    for i, j in ((0, 1), (0, 2), (1, 2)):
        difference = basis.row(i) - basis.row(j)
        assert s.simplify((difference * difference.T)[0]) == 2
    assert s.simplify(s.sqrt(2) * (s.pi / (2 * s.sqrt(2)))) == s.pi / 2
    # Therefore rho<pi/(2sqrt(2)) keeps every rotated edge gap strictly U3
    # and gives a regular real-positive resultant throughout the supplied loop.
    # The two half-cycle integrals of c cancel exactly by conjugation, so
    # mu_dot=w*c integrates to zero over a full rotation. Permutation plus
    # conjugation also gives this cancellation over each 2pi/3 mean-source cycle.
    e, weight, omega, time = s.symbols("e weight omega time", positive=True)
    initial = s.Symbol("initial", real=True)
    # Specify the phase contrast first and derive its EPI response. The
    # input is not reconstructed retrospectively from a desired trajectory.
    prescribed = rho * s.exp(s.I * (omega * time + initial))
    z = -weight * prescribed * (e - s.I * omega) / (s.pi * (e**2 + omega**2))
    radius = weight * rho / (s.pi * s.sqrt(e**2 + omega**2))
    assert s.simplify(s.diff(z, time) + e * z + weight * prescribed / s.pi) == 0
    assert s.simplify(s.conjugate(z) * z - radius**2) == 0
    # Full instantaneous work retains the oscillating mean, even though
    # its net displacement cancels over a complete mean-source cycle.
    mean, mean_rate = s.symbols("mean mean_rate", real=True)
    p, q = s.Matrix(P * 2), s.Matrix(Q_MODE * 2)
    u, v = s.re(z) / s.sqrt(2), s.im(z) / s.sqrt(6)
    phase_s, phase_t = s.re(prescribed) / s.sqrt(2), s.im(prescribed) / s.sqrt(6)
    field = mean * s.ones(6, 1) + u * p + v * q
    force = mean_rate * s.ones(6, 1) - weight * (phase_s * p + phase_t * q) / s.pi
    velocity = mean_rate * s.ones(6, 1) + s.diff(u, time) * p + s.diff(v, time) * q
    generator = s.Matrix(_exact_generator(observe_support_transport(_graph())))
    assert (e * generator * field + force - velocity).applyfunc(s.simplify) == s.zeros(
        6, 1
    )
    assert s.simplify((-3 * field.T * generator * field)[0] / 2) == 3 * radius**2
    work_density = (3 * force.T * velocity)[0]
    action_density = (3 * velocity.T * velocity)[0]
    assert s.simplify(work_density - action_density) == 0
    assert (
        s.simplify(action_density - 6 * radius**2 * omega**2 - 18 * mean_rate**2) == 0
    )
    # The mean is allowed to move by its actual source, rather than being
    # reset. Source path and clock remain prescribed; this proves neither
    # autonomous generation nor attraction of the resulting periodic state.

"""Fiber obstructions for the signed polar observation z=x*exp(i*theta).

These are detached pressure and coordinate controls, not trajectories or a
new phase law. Exact circle components define observation equality; binary64
evaluation of exp(i*pi) is never identified with the exact component (-1, 0).
The pressure owner separately records its represented phase-gradient values.
"""

import math
from fractions import Fraction as Q

import networkx as nx

from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)

_QUARTER_WEIGHTS = dict.fromkeys(("phase", "epi", "vf", "topo"), Q(1, 4))
_PURE_EPI_WEIGHTS = {"phase": 0, "epi": 1, "vf": 0, "topo": 0}


def _capture(epi, capacity, phase, *, weights=None):
    graph = nx.path_graph(2)
    for node, x, nu, theta in zip(graph, epi, capacity, phase, strict=True):
        graph.nodes[node].update(EPI=float(x), nu_f=float(nu), theta=float(theta))
    graph.graph["DNFR_WEIGHTS"] = {
        key: float(value) for key, value in (weights or _QUARTER_WEIGHTS).items()
    }
    observation = capture_non_epi_forcing(graph)
    decompose_non_epi_forcing(observation)
    assert observation.kernel_pressure_defect == (0, 0)
    return observation


def _pressure(observation):
    return tuple(
        observation.epi_weight * gradient + source
        for gradient, source in zip(
            observation.snapshot.epi_gradient,
            observation.forcing,
            strict=True,
        )
    )


def _rate(observation):
    return tuple(
        nu * pressure
        for nu, pressure in zip(
            observation.snapshot.capacity, _pressure(observation), strict=True
        )
    )


def _squared_modulus_rate(observation):
    # |z_i|^2=x_i^2, so its derivative is independent of every real theta_dot.
    return tuple(
        2 * x * rate
        for x, rate in zip(observation.snapshot.epi, _rate(observation), strict=True)
    )


def _polar_readout(epi, unit_components):
    # Exact real/imaginary components, not a runtime complex-exponential API.
    assert all(real * real + imag * imag == 1 for real, imag in unit_components)
    return tuple(
        (x * real, x * imag)
        for x, (real, imag) in zip(epi, unit_components, strict=True)
    )


def test_global_sign_phase_fiber_loses_the_canonical_directed_response():
    before = _capture((Q(1, 4), Q(1, 2)), (1, 2), (0, 0))
    flipped = _capture((Q(-1, 4), Q(-1, 2)), (1, 2), (math.pi, math.pi))
    # The proof uses exact circle phases (0,0) and (pi,pi). Each production
    # snapshot separately has exact zero relative-phase reading and strict U3.
    assert before.phase_gradient == flipped.phase_gradient == (0, 0)
    assert before.snapshot.capacity == flipped.snapshot.capacity == (1, 2)
    assert before.snapshot.conductance == flipped.snapshot.conductance
    assert all(abs(x) <= 1 for state in (before, flipped) for x in state.snapshot.epi)
    assert before.forcing == flipped.forcing == (Q(1, 4), Q(-1, 4))
    assert (
        _polar_readout(before.snapshot.epi, ((1, 0), (1, 0)))
        == (_polar_readout(flipped.snapshot.epi, ((-1, 0), (-1, 0))))
        == ((Q(1, 4), 0), (Q(1, 2), 0))
    )
    assert _pressure(before) == (Q(5, 16), Q(-5, 16))
    assert _pressure(flipped) == (Q(3, 16), Q(-3, 16))
    assert _rate(before) == (Q(5, 16), Q(-5, 8))
    assert _rate(flipped) == (Q(3, 16), Q(-3, 8))
    difference = tuple(
        new - old
        for old, new in zip(
            _squared_modulus_rate(before),
            _squared_modulus_rate(flipped),
            strict=True,
        )
    )
    assert _squared_modulus_rate(before) == (Q(5, 32), Q(-5, 8))
    assert _squared_modulus_rate(flipped) == (Q(-3, 32), Q(3, 8))
    assert (
        difference
        == tuple(
            -4 * x * nu * source
            for x, nu, source in zip(
                before.snapshot.epi,
                before.snapshot.capacity,
                before.forcing,
                strict=True,
            )
        )
        == (Q(-1, 4), 1)
    )
    # Thus the observable's squared-modulus derivative is not fiber-constant,
    # whatever phase velocities a completion of the nodal equation supplies.


def test_pure_epi_global_sign_control_has_no_non_epi_source_defect():
    before = _capture((Q(1, 4), Q(1, 2)), (1, 2), (0, 0), weights=_PURE_EPI_WEIGHTS)
    flipped = _capture(
        (Q(-1, 4), Q(-1, 2)),
        (1, 2),
        (math.pi, math.pi),
        weights=_PURE_EPI_WEIGHTS,
    )
    assert before.forcing == flipped.forcing == (0, 0)
    assert _pressure(flipped) == tuple(-p for p in _pressure(before))
    assert _squared_modulus_rate(flipped) == _squared_modulus_rate(before)
    # This positive control concerns the common flip only. It does not prove
    # autonomy of z, supply theta_dot, or justify independent node sign flips.


def test_zero_amplitude_hides_phase_that_changes_a_neighbor_radial_rate():
    aligned = _capture((0, 1), (1, 1), (0, 0))
    shifted = _capture((0, 1), (1, 1), (math.pi / 4, 0))
    assert 0 < float(shifted.phase[0]) < math.pi / 2
    assert aligned.snapshot.epi == shifted.snapshot.epi == (0, 1)
    assert aligned.snapshot.capacity == shifted.snapshot.capacity == (1, 1)
    # z=(0,1) for every phase of the zero-amplitude node; exp(i*theta) is
    # multiplied by zero, so observation equality needs no trigonometric
    # rounding assertion. These are retained binary64 pressure coefficients,
    # not a claim that represented pi/4 is exactly a quarter of real pi.
    assert aligned.phase_gradient == (0, 0)
    assert shifted.phase_gradient == (Q(-1, 4), Q(1, 4))
    assert _pressure(aligned) == (Q(1, 4), Q(-1, 4))
    assert _pressure(shifted) == (Q(3, 16), Q(-3, 16))
    assert _squared_modulus_rate(aligned) == (0, Q(-1, 2))
    assert _squared_modulus_rate(shifted) == (0, Q(-3, 8))
    assert _squared_modulus_rate(aligned)[1] != _squared_modulus_rate(shifted)[1]


def test_nonnegative_amplitude_boundary_is_not_invariant_by_definition():
    boundary = _capture((0, Q(1, 2)), (2, 1), (0, 0))
    assert boundary.phase_gradient == (0, 0)
    assert _pressure(boundary) == (Q(-1, 8), Q(1, 8))
    assert _rate(boundary) == (Q(-1, 4), Q(1, 8))
    assert boundary.snapshot.epi[0] == 0 and _rate(boundary)[0] < 0
    # The boundary vector points outside x>=0. Declaring x>0 to remove the
    # polar sign ambiguity therefore still requires an invariance argument.


def test_polar_jacobian_singularity_and_faithful_cylinder_encoding():
    real, imag = Q(3, 5), Q(4, 5)
    assert real * real + imag * imag == 1
    for x in (Q(-2), Q(0), Q(3, 2)):
        # D(x*cos(theta), x*sin(theta)) has determinant x.
        determinant = real * (x * real) - (-x * imag) * imag
        assert determinant == x
        # Retaining u as well as z recovers signed x even at x=0, while the
        # unit component itself retains phase there. No angular chart is used.
        z = (x * real, x * imag)
        assert z[0] * real + z[1] * imag == x
        assert z[1] * real - z[0] * imag == 0
    # The cylinder embedding (x,cos(theta),sin(theta)) has tangent columns
    # (1,0,0), (0,-sin(theta),cos(theta)); their Gram matrix is the identity.
    form_tangent = (Q(1), Q(0), Q(0))
    phase_tangent = (Q(0), -imag, real)
    gram = tuple(
        tuple(
            sum(a * b for a, b in zip(left, right, strict=True))
            for right in (form_tangent, phase_tangent)
        )
        for left in (form_tangent, phase_tangent)
    )
    assert gram == ((1, 0), (0, 1))
    # A faithful encoding does not determine phase velocity: u_dot=i*omega*u
    # still has the free real coordinate omega until a phase law is supplied.

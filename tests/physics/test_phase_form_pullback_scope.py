"""Pullback source work for declared phase/form embeddings, not feedback laws.

For fixed D and F=w*g(theta), alpha=F.T*D*dx pulls back under
(x,theta)=(chi(z),Theta(z)). Its exterior derivative is determined by the
antisymmetric part of (Dchi).T*D*J_g*DTheta. Pointwise symmetry is only a
pointwise statement; symmetry throughout a simply connected regular chart
gives exactness. No global exactness across phase branches follows from it.
"""

from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import Q_MODE, P, _exact_generator, _graph
from tnfr.physics.phase_response import (
    derive_phase_response,
    observe_phase_source_geometry,
)
from tnfr.physics.support_transport import observe_support_transport


def _geometry():
    symbolic = pytest.importorskip("sympy")
    source = observe_support_transport(_graph())
    assert source.capacity == (1,) * 6
    assert all(
        sorted(source.nodes[j][1] for j in row) == [0, 1, 2]
        for row in source.support_neighbors
    )
    strengths = tuple(
        sum(weight for i, _, weight in source.conductance if i == row)
        for row in range(6)
    )
    assert strengths == (3,) * 6
    lift = symbolic.Matrix.hstack(symbolic.Matrix(P * 2), symbolic.Matrix(Q_MODE * 2))
    return symbolic, source, symbolic.diag(*strengths), lift


def test_shared_source_jacobian_distinguishes_raw_curl_from_pullback_symmetry():
    s, source, metric, lift = _geometry()
    # Exact phasors for (-a,a,0) repeated, cos(a)=4/5, sin(a)=3/5.
    # Since cos(2a)=7/25>0, this representative lies strictly inside U3.
    phasors = ((Q(4, 5), Q(-3, 5)), (Q(4, 5), Q(3, 5)), (Q(1), Q(0))) * 2
    gram = tuple(
        tuple(sum(a * b for a, b in zip(left, right, strict=True)) for right in phasors)
        for left in phasors
    )
    reference = derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=source.support_neighbors,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=0,
    )
    geometry = observe_phase_source_geometry(reference)
    source_jacobian = s.Matrix(geometry.scaled_source_jacobian) / s.pi
    assert source_jacobian * s.ones(6, 1) == s.zeros(6, 1)
    # The actual pressure derivative remains nonzero at zero stage factor.
    assert source_jacobian != s.zeros(6)
    raw = metric * source_jacobian
    assert raw[0, 2] - raw[2, 0] == 3 / (13 * s.pi)
    kappa = s.Symbol("kappa", positive=True)
    phase_map_jacobian = -s.pi * kappa * lift
    pulled = lift.T * metric * source_jacobian * phase_map_jacobian
    assert pulled == kappa * s.diag(12, 36)
    assert pulled == pulled.T
    # A state-dependent phase metric can make H_phi*g=-grad(V), while the
    # work metric D gives a different one-form. Neither raw curl nor this
    # pullback symmetry chooses a phase rate or certifies embedding tangency.


def test_repeated_triple_phase_slaving_has_an_exact_work_primitive_not_invariance():
    s, _, metric, lift = _geometry()
    u, v = s.symbols("u v", real=True)
    weight, kappa = s.symbols("weight kappa", positive=True)
    coordinates = s.Matrix([u, v])
    relative_form = lift * coordinates
    common_source = s.Function("c")(u, v)
    # Each neighbor set contains the same three phases. On a regular branch,
    # theta=beta-pi*kappa*y gives g=kappa*y+c(u,v)*1, where c=(Arg(S)-beta)/pi.
    phase_pressure = kappa * relative_form + common_source * s.ones(6, 1)
    coefficients = (weight * lift.T * metric * phase_pressure).applyfunc(s.simplify)
    primitive = weight * kappa * (6 * u**2 + 18 * v**2)
    assert coefficients == s.Matrix([s.diff(primitive, u), s.diff(primitive, v)])
    assert coefficients.jacobian(coordinates) == weight * kappa * s.diag(12, 36)

    angle = s.Symbol("angle", real=True)
    # This small ellipse stays in one regular open chart: every phase gap
    # is below pi/2. A primitive proves zero work on every closed loop here,
    # including the rank-one v=0 restriction; no traversal clock is needed.
    loop = {u: s.cos(angle) / (48 * kappa), v: s.sin(angle) / (48 * kappa)}
    density = sum(
        coefficient.subs(loop) * s.diff(loop[coordinate], angle)
        for coordinate, coefficient in zip(coordinates, coefficients, strict=True)
    )
    assert s.simplify(s.integrate(density, (angle, 0, 2 * s.pi))) == 0

    # Exactness of the work form does not establish invariance of the imposed
    # fixed-mean chart. At u=0,v=1/12,kappa=1 the phases are (-pi/12,-pi/12,pi/6).
    # Their common resultant has positive real part but nonzero imaginary part;
    # consequently c!=0 and positive phase weight changes the global EPI mean.
    resultant_real = 2 * s.cos(s.pi / 12) + s.cos(s.pi / 6)
    resultant_imaginary = -2 * s.sin(s.pi / 12) + s.sin(s.pi / 6)
    assert s.simplify(resultant_real).is_positive
    assert s.simplify(resultant_imaginary).is_negative
    assert s.simplify(sum(phase_pressure) - 6 * common_source) == 0

    # The simply connected chart condition cannot be dropped in a general
    # pullback theorem: d(arg) is closed on a punctured plane, yet not exact.
    angular_form = s.Matrix([-v / (u**2 + v**2), u / (u**2 + v**2)])
    assert s.simplify(
        angular_form.jacobian(coordinates) - angular_form.jacobian(coordinates).T
    ) == s.zeros(2)
    unit_circle = {u: s.cos(angle), v: s.sin(angle)}
    angular_density = sum(
        coefficient.subs(unit_circle) * s.diff(unit_circle[coordinate], angle)
        for coordinate, coefficient in zip(coordinates, angular_form, strict=True)
    )
    assert s.simplify(angular_density) == 1
    # Thus integral(d arg)=2*pi; this is a topology control, not a TNFR force.


def test_independent_phase_coordinate_allows_work_but_does_not_supply_a_trajectory():
    s, source, metric, lift = _geometry()
    q = lift[:, 0]
    generator = s.Matrix(_exact_generator(source))
    assert generator * q == -q
    u, a, angle = s.symbols("u a angle", real=True)
    weight = s.Symbol("weight", positive=True)
    x = s.ones(6, 1) / 2 + u * q
    # theta=(-a,a,0) repeated has S=1+2*cos(a)>0 and g=(a/pi)*q
    # for |a|<pi/4. The coordinates (u,a) are independent here.
    pressure = a * q / s.pi
    form = s.simplify(weight * x.jacobian([u, a]).T * metric * pressure)
    assert form == s.Matrix([12 * weight * a / s.pi, 0])
    differential = form.jacobian([u, a])
    assert differential[0, 1] - differential[1, 0] == 12 * weight / s.pi

    loop = {
        u: s.Rational(1, 16) + s.cos(angle) / 32,
        a: s.pi / 12 - s.pi * s.sin(angle) / 24,
    }
    # u stays in [1/32,3/32], a in [pi/24,pi/8], so EPI and the phase chart
    # remain regular. This orientation yields positive source/form work.
    loop_x, loop_pressure = x.subs(loop), pressure.subs(loop)
    density = (weight * loop_pressure.T * metric * loop_x.diff(angle))[0]
    assert s.simplify(s.integrate(density, (angle, 0, 2 * s.pi))) == s.pi * weight / 64

    # Positive circulation is insufficient. At angle=0 the EPI tangent is
    # zero, while the shared nodal EPI row is nonzero for e=1/2,w=1/4.
    # No finite positive change of clock repairs this pointwise mismatch.
    model_rate = generator * loop_x / 2 + loop_pressure / 4
    assert loop_x.diff(angle).subs(angle, 0) == s.zeros(6, 1)
    assert s.simplify(model_rate.subs(angle, 0)) == -s.Rational(5, 192) * q

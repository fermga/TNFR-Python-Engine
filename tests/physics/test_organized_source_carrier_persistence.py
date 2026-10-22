"""Finite algebraic controls for phase persistence on the exact carrier.

The six prescribed Lie-output differentials are evaluated only at equilibrium,
where they are powers of the actual linearization. No nonlinear Lie tower,
trajectory, acquired-source projection or additional derivative campaign is
evaluated. The local classification is an implicit-function theorem argument,
not a consequence of the sampled static family members alone.
"""

from fractions import Fraction as Q

import mpmath
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as owner

ORIGIN = Q(1, 40)
CASES = ((Q(0), Q(0)), (Q(1, 64), Q(0)), (Q(0), Q(1, 16)), (-Q(1, 64), -Q(1, 16)))


@pytest.fixture(scope="module", autouse=True)
def no_response_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Carrier persistence controls cannot execute or rebuild a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (capture, "_assess_organized_source_capture"),
            (capture, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def geometry():
    return owner._derive_organized_source_port()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, expected):
    assert _mp(mp, bounds.lo) <= expected <= _mp(mp, bounds.hi)


def _laplacian(edges):
    matrix = sp.zeros(5)
    for left, right in edges:
        matrix[left, left] += 1
        matrix[right, right] += 1
        matrix[left, right] -= 1
        matrix[right, left] -= 1
    return matrix


def _current(edges, phases):
    result = sp.zeros(5, 1)
    for left, right in edges:
        current = sp.sin(phases[right] - phases[left])
        result[left] += current
        result[right] -= current
    return result


def test_nonlinear_family_is_tangent_to_the_actual_carrier_field(geometry):
    u, a, gamma = sp.symbols("u a gamma", real=True)
    origin = sp.Rational(1, 40)
    direction = sp.Matrix((0, 1, -1, 0, 0))
    forms = u * direction
    phases = sp.ones(5, 1) * origin + a * direction
    edges = geometry.collective_edges
    swap = (0, 2, 1, 3, 4)
    assert {tuple(sorted((swap[i], swap[j]))) for i, j in edges} == {
        tuple(sorted(edge)) for edge in edges
    }
    matrix = _laplacian(edges)
    assert matrix * direction == 2 * direction
    gradient = matrix * forms
    current = _current(edges, phases)
    assert sp.simplify(current + 2 * sp.sin(a) * direction) == sp.zeros(5, 1)
    assert geometry.clock_factor == Q(1, 4)
    form_rate = (-gradient + gamma * current) / 4
    phase_rate = gamma * gradient / 4
    du, da = -(u + gamma * sp.sin(a)) / 2, gamma * u / 2
    assert sp.simplify(form_rate - du * direction) == sp.zeros(5, 1)
    assert sp.simplify(phase_rate - da * direction) == sp.zeros(5, 1)
    excess_storage = sp.Rational(9, 2) * forms.dot(gradient) + 9 * sum(
        1 - sp.cos(phases[right] - phases[left]) for left, right in edges
    )
    assert sp.simplify(excess_storage - 18 * u**2 - 36 * (1 - sp.cos(a))) == 0
    storage_rate = sp.diff(excess_storage, u) * du + sp.diff(excess_storage, a) * da
    assert sp.simplify(storage_rate + 18 * u**2) == 0
    assert sum(forms) == 0 and sum(phases) == sp.Rational(1, 8)
    for child in (0, 3, 4):
        assert forms[child] == form_rate[child] == phase_rate[child] == 0
        assert phases[child] == origin
    # Exact tangency gives an invariant two-parameter family. Its unchanged
    # target form excludes a positive Emission increment along this family.


@pytest.mark.parametrize("u,a", CASES)
def test_static_family_matches_shared_collective_and_all_fine_rows(u, a, mp):
    forms = (Q(0), u, -u, Q(0), Q(0))
    phases = (ORIGIN, ORIGIN + a, ORIGIN - a, ORIGIN, ORIGIN)
    collective = owner._observe_organized_source_collective(
        form_means=forms, phase_origins=phases
    )
    fine = owner._observe_organized_source_port(
        form=tuple(value for value in forms for _ in range(9)),
        phase_offsets=tuple(value for value in phases for _ in range(9)),
    )
    gamma = 1 / (1023 * mp.pi)
    du = -(_mp(mp, u) + gamma * mp.sin(_mp(mp, a))) / 2
    da = gamma * _mp(mp, u) / 2
    expected_form = (mp.mpf(0), du, -du, mp.mpf(0), mp.mpf(0))
    expected_phase = (mp.mpf(0), da, -da, mp.mpf(0), mp.mpf(0))
    assert collective.conserved_form_total == 0
    assert collective.conserved_phase_total == Q(1, 8)
    assert fine.conserved_form_total == 0
    assert fine.conserved_phase_total == Q(1, 8)
    assert sum(fine.phase_offsets) == Q(9, 8)
    assert collective.form_gradient == (Q(0), 2 * u, -2 * u, Q(0), Q(0))
    assert (
        collective.joint_storage_tau_rate == fine.joint_storage_tau_rate == -18 * u**2
    )
    for child in range(5):
        _contains(mp, collective.form_tau_rate_bounds[child], expected_form[child])
        _contains(mp, collective.phase_tau_rate_bounds[child], expected_phase[child])
        for slot in range(9):
            node = 9 * child + slot
            _contains(mp, fine.form_tau_rate_bounds[node], expected_form[child])
            _contains(mp, fine.phase_tau_rate_bounds[node], expected_phase[child])
    assert collective.host_port_form_tau_bounds.contains(0)
    assert collective.donor_port_form_tau_bounds.contains(0)
    assert collective.host_port_phase_tau_bounds.contains(0)
    assert collective.donor_port_phase_tau_bounds.contains(0)
    if u:
        assert not collective.phase_tau_rate_bounds[1].contains(0)
    elif a:
        assert not collective.form_tau_rate_bounds[1].contains(0)
    # Nonzero members can evolve internally while target and donor remain
    # stationary. This supplied family is not the previously acquired source.


def test_six_lie_output_differentials_have_exact_rank_six_on_the_carrier_leaf(
    geometry,
):
    gamma = sp.Symbol("gamma", positive=True)
    matrix = _laplacian(geometry.collective_edges)
    phases = sp.Matrix(sp.symbols("phi0:5", real=True))
    at_equilibrium = {value: sp.Rational(1, 40) for value in phases}
    current_jacobian = _current(geometry.collective_edges, phases).jacobian(phases)
    assert current_jacobian.subs(at_equilibrium) == -matrix

    # Coordinates (X0,Xs,X3,u) with Xs=(X1+X2)/2 and u=(X1-X2)/2;
    # the donor is -X0-2Xs-X3. Apply the same chart to phase displacements.
    chart = sp.Matrix(
        ((1, 0, 0, 0), (0, 1, 0, 1), (0, 1, 0, -1), (0, 0, 1, 0), (-1, -2, -1, 0))
    )
    inverse = sp.Matrix(
        (
            (1, 0, 0, 0, 0),
            (0, sp.Rational(1, 2), sp.Rational(1, 2), 0, 0),
            (0, 0, 0, 1, 0),
            (0, sp.Rational(1, 2), -sp.Rational(1, 2), 0, 0),
        )
    )
    assert sp.ones(1, 5) * chart == sp.zeros(1, 4)
    assert inverse * chart == sp.eye(4)
    reduced = inverse * matrix * chart
    assert matrix * chart == chart * reduced
    symmetric = sp.Matrix(((4, 0, 1), (-1, 2, -1), (0, -2, 2)))
    assert reduced == sp.diag(symmetric, sp.Matrix(((2,),)))
    zero = sp.zeros(4)
    linearization = (
        sp.BlockMatrix(
            ((-reduced, -gamma * reduced), (gamma * reduced, zero))
        ).as_explicit()
        / 4
    )
    row = sp.Matrix(((0, 0, 0, 0, 1, 0, 0, 0),))
    rows = []
    # At f(z*)=0, D(L_f^j h)(z*)=dh*A^j. Exactly the six frozen
    # differentials are needed; no nonlinear expressions or seventh output.
    for _ in range(6):
        row = (row * linearization).applyfunc(sp.expand)
        rows.append(row)
    outputs = sp.Matrix.vstack(*rows)
    assert outputs[:, 3] == sp.zeros(6, 1)
    assert outputs[:, 7] == sp.zeros(6, 1)
    symmetric_minor = outputs[:, (0, 1, 2, 4, 5, 6)]
    determinant = sp.factor(symmetric_minor.det(method="domain-ge"))
    polynomial = 4489 * gamma**6 - 2453 * gamma**4 + 435 * gamma**2 - 25
    assert determinant == 16000 * gamma**15 * polynomial / 4**21

    # The determinant is tiny at the actual gamma; factor its positive power
    # analytically rather than materializing it or asking floating-point rank.
    eta = geometry.gamma_bounds**2
    assert geometry.gamma_bounds.lo > 0
    assert 0 < eta.lo <= eta.hi < Q(1, 10**6)
    factored = 4489 * eta**3 - 2453 * eta**2 + 435 * eta - 25
    assert factored.hi < -24
    assert 4489 * Q(1, 10**6) ** 3 + 435 * Q(1, 10**6) < 1
    # The nonsingular six-column block and two exact zero columns establish
    # rank six and the two family tangents as the entire linear kernel.
    assert outputs.subs(gamma, 0) == sp.zeros(6, 8)

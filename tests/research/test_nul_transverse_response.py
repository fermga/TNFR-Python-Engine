"""Finite tangent-response controls on the unchanged moving sine contact.

The tangent normalization is not a finite perturbed source. Full linearized
rows, fixed-observation identities and analytic remainders are checked without
an event, a trajectory, a retained response or a parameter search.
"""

from fractions import Fraction as Q

import pytest

from tests.research import test_nul_cross_law_observation as cross_law
from tests.research.test_nul_causal_class_exit import HORIZON
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage

branches = cross_law.branches
NEIGHBORS = (0, 1, 6, 7)


@pytest.fixture(scope="module", autouse=True)
def no_event_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("transverse controls must not execute or prepare a runtime event")

    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "execute_pointwise_stage",
            "execute_operator_major_stage",
            "_propose_pointwise",
        ):
            patch.setattr(network_stage, name, forbidden)
        yield


@pytest.fixture(scope="module")
def full_tangent(branches):
    s = pytest.importorskip("sympy")
    graph = branches[0]
    forms = s.Matrix(s.symbols("form0:20", real=True))
    phases = s.Matrix(s.symbols("phase0:20", real=True))
    # Differentiate the actual two complete rows, before imposing symmetry.
    currents = s.Matrix(
        [sum(s.sin(phases[j] - phases[i]) for j in graph[i]) for i in graph]
    )
    gradients = s.Matrix(
        [
            sum(
                s.Rational(data["weight"]) * (forms[i] - forms[j])
                for j, data in graph[i].items()
            )
            for i in graph
        ]
    )
    current_jacobian = currents.jacobian(phases)
    form_jacobian = gradients.jacobian(forms)
    assert s.trigsimp((currents[8] - currents[9]).subs(phases[9], phases[8])) == 0
    assert s.expand((gradients[8] - gradients[9]).subs(forms[9], forms[8])) == 0
    injection = s.zeros(40, 2)
    injection[8, 0], injection[9, 0] = 1, -1
    injection[28, 1], injection[29, 1] = 1, -1
    return s, phases, current_jacobian, form_jacobian, injection


@pytest.mark.parametrize("branch", (0, 1))
def test_actual_pair_sector_is_invariant_in_all_forty_linearized_rows(
    branches, full_tangent, branch
):
    s, phases, current_jacobian, form_jacobian, injection = full_tangent
    graph = branches[branch]
    assert tuple(sorted(graph[8])) == tuple(sorted(graph[9])) == NEIGHBORS
    assert not graph.has_edge(8, 9)
    assert graph.degree[8] == graph.degree[9] == 4
    assert graph.nodes[8]["nu_f"] == graph.nodes[9]["nu_f"] == 1.0
    assert all(graph.nodes[8][key] == graph.nodes[9][key] for key in ("EPI", "theta"))
    assert all(
        graph.edges[pair, j]["weight"] == 1.0 for pair in (8, 9) for j in NEIGHBORS
    )
    mobility = s.diag(
        *(s.Rational(graph.nodes[i]["nu_f"]) / graph.degree[i] for i in graph)
    )
    # Keep the original pi clock in both rows. Arbitrary donor phase variables
    # remain live; only the preserved equality phase8=phase9 is imposed.
    upper = s.zeros(20).row_join(mobility * current_jacobian / s.pi)
    lower = (mobility * form_jacobian / s.pi).row_join(s.zeros(20))
    tangent = upper.col_join(lower).subs(phases[9], phases[8])
    stiffness = sum(s.cos(phases[8] - phases[j]) for j in NEIGHBORS)
    reduced = s.Matrix(((0, -stiffness / (4 * s.pi)), (1 / s.pi, 0)))
    residual = tangent * injection - injection * reduced
    assert residual.applyfunc(s.trigsimp) == s.zeros(40, 2)
    assert tangent.shape == (40, 40)
    assert len(stiffness.free_symbols) == 5
    assert (current_jacobian - current_jacobian.T).applyfunc(s.trigsimp) == s.zeros(20)
    assert form_jacobian == form_jacobian.T


def test_one_fixed_quadratic_has_the_declared_derivative_and_cubic_coefficient():
    s = pytest.importorskip("sympy")
    u, v, h, h0, h1, h2 = s.symbols("u v h h0 h1 h2", real=True)
    observation = (4 * u**2 + h0 * v**2) / 2

    def derivative(expression):
        return (
            s.diff(expression, u) * (-h * v / (4 * s.pi))
            + s.diff(expression, v) * u / s.pi
            + s.diff(expression, h) * h1
            + s.diff(expression, h1) * h2
        )

    first = s.expand(derivative(observation))
    assert s.simplify(first - (h0 - h) * u * v / s.pi) == 0
    second = derivative(first)
    third = derivative(second)
    initial = {u: 0, v: 1, h: h0}
    assert first.subs(initial) == second.subs(initial) == 0
    assert s.simplify(third.subs(initial) - h0 * h1 / (2 * s.pi**2)) == 0
    assert s.simplify(third.subs(initial) / 6 - h0 * h1 / (12 * s.pi**2)) == 0
    assert third.subs(initial).subs(h1, 0) == 0
    # h0 is the same fixed initial metric on both branches. Replacing it by
    # h(t) would add h'(t)*v^2/2 and define a different observation.
    moving = (4 * u**2 + h * v**2) / 2
    assert s.simplify(derivative(moving) - h1 * v**2 / 2) == 0


def test_initial_stiffness_and_its_derivative_use_the_actual_complete_image(branches):
    before, after, image = branches
    pi = pi_interval()
    amplitude = (1 - image.factor) / 2
    phases = tuple(Q(before.nodes[i]["theta"]) for i in before)
    assert phases == tuple(Q(after.nodes[i]["theta"]) for i in after)
    initial_coefficients = tuple(cos(I(phases[8] - phases[j])) for j in NEIGHBORS)
    assert all(coefficient.lo > Q(1, 6) for coefficient in initial_coefficients)
    h0 = sum(initial_coefficients, I(0))
    assert Q(2, 3) < h0.lo <= h0.hi < 4
    derivatives = []
    for graph in (before, after):
        numerators = cross_law._primitive_rows(graph, Q(0))[2]
        derivative = -sum(
            (
                sin(I(phases[8] - phases[j])) * (numerators[8] - numerators[j]) / pi
                for j in NEIGHBORS
            ),
            I(0),
        )
        derivatives.append(derivative)
    assert derivatives[0] == I(0)
    assert derivatives[1].lo > 0
    after_numerators = cross_law._primitive_rows(after, Q(0))[2]
    assert after_numerators[0] == -Q(27, 28) * image.factor * amplitude
    assert after_numerators[1] == amplitude / 8
    assert after_numerators[8] == after_numerators[9] == amplitude / 4
    assert after_numerators[6] == after_numerators[7] == 0
    bracket = -(
        sin(I(5)) * (Q(1, 4) + Q(27, 28) * image.factor)
        + sin(I(5) - I(phases[1])) / 8
        + sin(I(Q(5, 4))) / 2
    )
    expected = amplitude * bracket / pi
    assert (expected - derivatives[1]).contains(0)
    assert bracket.lo > Q(41, 128)
    assert amplitude > Q(5, 128) and pi.hi < 4
    assert Q(5, 128) * Q(41, 128) / 4 == Q(205, 65536)
    assert derivatives[1].lo > Q(205, 65536)
    # The zero baseline phase velocity does not freeze its background forms.
    before_form_rates = cross_law._primitive_rows(before, Q(0))[3]
    assert any(not rate.contains(0) for rate in before_form_rates)
    assert HORIZON == Q(1, 2**14)


@pytest.mark.parametrize("branch", (0, 1))
def test_actual_full_rows_supply_uniform_background_coefficient_derivatives(
    branches, branch
):
    graph = branches[branch]
    pi = pi_interval()
    forms = tuple(Q(graph.nodes[i]["EPI"]) for i in graph)
    capacities = tuple(Q(graph.nodes[i]["nu_f"]) for i in graph)
    amplitude = max(forms) - min(forms)
    # The bare sine form row gives |x'_i| <= nu_i/pi, hence the form diameter
    # is at most A+2t/pi. This holds on the full branch, not a frozen donor set.
    assert all(0 < capacity <= 1 for capacity in capacities)
    diameter_bound = amplitude + 2 * HORIZON / pi.lo
    assert diameter_bound < Q(1, 8)
    phase_velocity_coefficients = tuple(
        capacities[i]
        * sum((Q(data["weight"]) for data in graph[i].values()), Q(0))
        / graph.degree[i]
        for i in graph
    )
    # Differentiating q_i=sum_j w_ij(x_i-x_j) uses each actual capacity.
    phase_acceleration_coefficients = tuple(
        capacities[i]
        / graph.degree[i]
        * sum(
            (
                Q(data["weight"]) * (capacities[i] + capacities[j])
                for j, data in graph[i].items()
            ),
            Q(0),
        )
        for i in graph
    )
    assert all(0 < value <= 1 for value in phase_velocity_coefficients)
    assert all(0 < value <= 2 for value in phase_acceleration_coefficients)
    for i, j in graph.edges:
        velocity = (
            phase_velocity_coefficients[i] + phase_velocity_coefficients[j]
        ) * diameter_bound
        acceleration = (
            phase_acceleration_coefficients[i] + phase_acceleration_coefficients[j]
        )
        assert velocity < Q(1, 4)  # one original-clock pi divisor remains
        assert acceleration <= 4  # two original-clock pi divisors remain
    # h=sum four cos(g): h'=-sum sin(g)g' and
    # h''=-sum[cos(g)(g')^2+sin(g)g'']. No third derivative is needed.
    assert len(NEIGHBORS) == 4
    assert 4 * Q(1, 4) == 1
    assert 4 * (Q(1, 4) ** 2 + 4) == Q(65, 4)
    assert Q(65, 4) / pi.lo**2 < 2
    # Thus |h|<=4, |h'|<=1/pi, |h''|<2, and the linear stiffness remainder
    # is strictly less than t^2 for positive t, with equality zero at t=0.


def test_volterra_bounds_control_the_sector_and_its_first_product_remainder():
    s = pytest.importorskip("sympy")
    time, integration_time = s.symbols("time integration_time", nonnegative=True)
    pi = pi_interval()
    assert pi.lo > 3
    # Integrating v''=-h*v/(4pi^2), with v(0)=1 and v'(0)=0, bounds
    # sup|v| by 1 + t^2/(2pi^2)*sup|v|; the denominator is positive.
    feedback = HORIZON**2 / (2 * pi.lo**2)
    assert 0 < feedback < Q(1, 2)
    assert 1 / (1 - feedback) < 2
    assert s.integrate(
        2 * integration_time / s.pi**2, (integration_time, 0, time)
    ) == (time**2 / s.pi**2)
    # |v|<=2 and |h|<=4 give |u|<=2t/pi. The previous integral gives
    # |v-1|<=t^2/pi^2. Use |h-h0|<=t/pi and h0<=4 in the u remainder.
    u_error = s.integrate(
        (2 * integration_time / s.pi + 4 * integration_time**2 / s.pi**2) / (4 * s.pi),
        (integration_time, 0, time),
    )
    assert s.expand(u_error - time**2 / (4 * s.pi**2) - time**3 / (3 * s.pi**3)) == 0
    product_error = 2 * u_error + time / s.pi * time**2 / s.pi**2
    assert (
        s.expand(product_error - time**2 / (2 * s.pi**2) - 5 * time**3 / (3 * s.pi**3))
        == 0
    )
    assert Q(1, 18) + 5 * HORIZON / 81 < Q(1, 16)
    # Every bound is monotone on [0,T]. Therefore the product remainder is
    # <t^2/16 at each positive t, not merely at one computed endpoint.


def test_fixed_observation_remainder_is_uniform_and_vanishes_at_the_origin():
    s = pytest.importorskip("sympy")
    time, h0, h1, stiffness_error, product_error = s.symbols(
        "time h0 h1 stiffness_error product_error", real=True
    )
    derivative = (
        (-h1 * time - stiffness_error)
        * (-h0 * time / (4 * s.pi) + product_error)
        / s.pi
    )
    leading = h0 * h1 * time**2 / (4 * s.pi**2)
    residual = (
        h0 * stiffness_error * time / (4 * s.pi**2)
        - h1 * time * product_error / s.pi
        - stiffness_error * product_error / s.pi
    )
    assert s.expand(derivative - leading - residual) == 0
    # Insert |h1|<=1/pi, h0<=4, |stiffness_error|<=t^2,
    # |product_error|<=t^2/16, and pi>3. The t^4 term is bounded by T*t^3.
    coefficient = Q(17, 144) + HORIZON / 48
    assert coefficient < Q(1, 8)
    bound = coefficient * time**3
    integrated = s.integrate(bound, (time, 0, time))
    assert s.simplify(integrated - coefficient * time**4 / 4) == 0
    assert coefficient / 4 < Q(1, 32)
    assert bound.subs(time, 0) == integrated.subs(time, 0) == 0
    assert s.integrate(leading, (time, 0, time)) == h0 * h1 * time**3 / (12 * s.pi**2)


def test_two_finite_remainders_leave_the_frozen_strict_response_margin():
    # Use only the coarse analytic constants above, not sharper computed source
    # values. The two branch remainders have independent signs and both count.
    initial_stiffness_lower = Q(2, 3)
    derivative_difference_lower = Q(205, 65536)
    pi_upper = Q(4)
    cubic_lower = (
        initial_stiffness_lower * derivative_difference_lower / (12 * pi_upper**2)
    )
    assert cubic_lower == Q(205, 18874368)
    both_remainders = 2 * HORIZON**4 / 32
    finite_lower = cubic_lower * HORIZON**3 - both_remainders
    assert finite_lower == Q(133, 9 * 2**63)
    assert finite_lower > Q(1, 2**60) > 0
    # This is a finite positive-time comparison of the same fixed Q0 and
    # tangent normalization. At t=0 both responses coincide exactly.

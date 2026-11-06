"""Static controls for the principal reduction's positive-cosine exclusion.

The domain-wide proof belongs to the theory owner. These independent algebra
and interval checks protect its coefficients, sign margins and branch scope;
they do not infer exclusion from unsuccessful searches or sampled responses.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin, sqrt
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions import Coupling, SelfOrganization
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research import maintained_successor_principal as owner


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("sector controls must not execute an event or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (_coupling_stage_kernel, "propose_coupling_target"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
            (word_execution, "execute_network_operator_stage"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (maintained_parent_thol_execution, "execute_network_operator_stage"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "rk4_coupled_step"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
            (Coupling, "_execute"),
            (SelfOrganization, "_prepare_execution"),
            (SelfOrganization, "_prepare_bifurcation"),
        ):
            patch.setattr(module, name, forbidden)
        yield


def _core(coordinates):
    return owner._evaluate_core(
        coordinates, edges=owner.SELECTED_SUPPORT, shifts=owner.SELECTED_SHIFTS
    )


def _four(coordinates):
    return owner._evaluate_reduction(
        coordinates, edges=owner.SELECTED_SUPPORT, shifts=owner.SELECTED_SHIFTS
    )


def test_equal_half_spans_have_incompatible_fine_endpoint_currents():
    # A2 balances exactly at the familiar fifth-turn angle. The independently
    # summed fine B2 current remains nonzero because B1 has twice Q's weight.
    angle = 2 * pi_interval() / 5
    candidate = _four((angle, 0, angle, 0))
    fine = candidate.core.gradient
    assert fine[4].contains(0) and fine[5].contains(0)
    assert fine[14].lo > Q(9, 10) and fine[15].lo > Q(9, 10)
    assert (fine[14] - fine[4]).hull(sin(angle)).width < Q(1, 10**30)
    assert candidate.gradient[0].contains(0)
    assert candidate.gradient[2].lo > Q(18, 5)


def test_obtuse_stationary_endpoint_has_negative_chain_reduced_curvature():
    # This is an A2-current-balanced algebra probe, not a stationary state of
    # the whole graph: both upper A gaps are 2*pi/3 and the return gap pi/3.
    pi = pi_interval()
    candidate = _core((pi / 3, pi / 3, pi, 0, pi / 3, pi))
    assert candidate.gradient[4].contains(0)
    assert candidate.gradient[5].contains(0)
    assert candidate.core_gradient[2].contains(0)
    assert candidate.core_hessian[2][2].contains(Q(-4, 3))
    assert candidate.core_hessian[2][2].hi < -1
    assert all(margin.lo > 0 for margin in candidate.principal_margins)
    assert all(margin.lo > 0 for margin in candidate.chain_acute_margins)
    assert any(value.lo > Q(1, 2) for value in candidate.gradient)


@pytest.mark.parametrize("ratio", (Q(1, 4), Q(1), Q(3), Q(4)))
def test_argument_derivative_bound_has_exact_nonnegative_remainder(ratio):
    # With C=cos(m)>0 and r/C<=4, the bound follows from this exact identity.
    # Rational C values include the aligned endpoint and a nonzero-angle case.
    for cosine in (Q(3, 4), Q(7, 8), Q(1)):
        radius = ratio * cosine
        denominator = 1 + radius**2 + 2 * radius * cosine
        derivative = radius * (radius + cosine) / denominator
        bound = radius / (radius + cosine)
        remainder = radius * (1 - cosine**2) / (denominator * (radius + cosine))
        assert bound - derivative == remainder
        assert 0 <= derivative <= bound <= Q(4, 5)


def test_interval_clearances_cover_the_entire_b_upper_angle_range():
    pi = pi_interval()
    # The proof confines both B2 gaps to (2*pi/5,3*pi/5). Enclose that whole
    # interval, including its endpoints; no midpoint sample is used.
    angle = I((2 * pi / 5).lo, (3 * pi / 5).hi)
    sine = sin(angle)
    assert sine.lo > Q(2, 3)
    assert (sine + 2 * sine).lo > 2
    # Constants in the A-sector estimate have strict certified clearance.
    assert (pi**2 / 25).hi < Q(1, 2)
    assert (1 / sqrt(I(2))).lo > Q(1, 3)


@pytest.mark.parametrize("midpoint", (Q(-1, 4), Q(1, 4)))
def test_half_span_beyond_branch_bound_rejects_even_with_regular_argument(midpoint):
    # At u=2, cos(u)<-1/4 and the resultant is nonzero off the real cut.
    # Admission must fail on the original endpoint branches, not rewrap them.
    assert cos(I(2)).hi < Q(-1, 4)
    with pytest.raises(ValueError, match="principal margin"):
        _four((2, midpoint, 1, 0))


def test_negative_half_span_cosine_remains_an_admitted_candidate_observation():
    # The positive-sector theorem must not silently narrow the evaluator's
    # domain. Even after the full exclusion, regular nonacute observations
    # remain useful; admission itself is not a stationarity/minimum verdict.
    candidate = _four((Q(8, 5), 0, 1, 0))
    assert cos(candidate.parameters[0]).hi < 0
    assert candidate.resultants[0].lo > 0
    assert all(value.lo > 0 for value in candidate.core.principal_margins)
    assert not hasattr(candidate, "minimum_certified")
    assert not hasattr(candidate, "principal_minima_excluded")


def test_wide_b_stationary_endpoint_has_negative_full_chain_restriction():
    pi = pi_interval()
    parent = arg(sqrt(I(2)), I(1))
    candidate = _core((parent, parent, pi, 0, parent, pi))
    # Each B2 upper sine is 1/sqrt(3), so its actual 1:2 current balances
    # the pi/3 return chain, while the chain-restricted diagonal is negative.
    assert candidate.gradient[14].contains(0)
    assert candidate.gradient[15].contains(0)
    expected = Q(2, 3) - 2 * sqrt(I(6))
    diagonal = candidate.core_hessian[5][5]
    assert diagonal.hull(expected).width < Q(1, 10**30)
    assert diagonal.hi < -4
    assert all(margin.lo > 0 for margin in candidate.principal_margins)
    assert any(value.hi < Q(-1, 2) for value in candidate.gradient)


def test_zero_half_span_cosines_do_not_remove_endpoint_current_multiplicity():
    pi = pi_interval()
    candidate = _four((pi / 2, -pi / 6, pi / 2, -pi / 6))
    assert candidate.gradient[1].contains(0)
    assert candidate.gradient[3].contains(0)
    assert candidate.core.gradient[4].contains(0)
    assert candidate.core.gradient[5].contains(0)
    assert candidate.core.gradient[14].lo > Q(4, 5)
    assert candidate.core.gradient[15].lo > Q(4, 5)


def _wide_b_scalar_bounds():
    pi = pi_interval()
    contact = I(Q(34, 15))
    return {
        "contact_upper": sin(contact - 2 * pi / 3) - Q(1, 6),
        "middle_upper": sin(I(Q(21, 40))) - 4 * sin(I(Q(1, 8))),
        "small_b_current": sin(contact) + 2 * sin(pi - Q(61, 24)) - sqrt(I(3)),
        "large_b_current": cos(I(Q(1, 8))) * sin(I(Q(257, 120)))
        - sin(pi / 3 - Q(1, 12)),
        "upper_b": -I(Q(1, 4)) - cos(7 * pi / 12),
        "positive_beta_lower": pi - Q(61, 24),
        "sine_endpoint_order": sin(pi / 2 + Q(1, 8)) - sin(contact),
        "contact_angle_positive": contact - 2 * pi / 3,
        "contact_angle_acute": pi / 2 - (contact - 2 * pi / 3),
        "obtuse_contact_current": sin(3 * pi / 5) - Q(1, 6),
    }


@pytest.fixture(scope="module")
def wide_b_bounds():
    return _wide_b_scalar_bounds()


@pytest.mark.parametrize(
    "name",
    (
        "contact_upper",
        "middle_upper",
        "small_b_current",
        "large_b_current",
        "upper_b",
        "positive_beta_lower",
        "sine_endpoint_order",
        "contact_angle_positive",
        "contact_angle_acute",
        "obtuse_contact_current",
    ),
)
def test_wide_b_proof_scalar_inequalities_have_strict_interval_clearance(
    name, wide_b_bounds
):
    # Exact rational inputs and outward Machin-pi/trigonometry bounds certify
    # these signs; no rounded decimal target or response fixture is compared.
    assert wide_b_bounds[name].lo > 0


def test_mixed_sector_taylor_bounds_have_clearance_over_the_declared_ranges():
    pi = pi_interval()
    epsilon = I(0, Q(1, 5))
    squared_radius = I(0, Q(16, 25))
    # The low-e square-root comparison reduces to these positive polynomial
    # factors on entire intervals, rather than checks at selected states.
    assert (1 + 1 / sqrt(I(3)) - Q(58, 9) * epsilon).lo > 0
    assert (3 - 4 * squared_radius).lo > 0
    assert (1 - Q(2, 3) * squared_radius).lo > 0
    assert Q(146, 75) * Q(137, 175) > Q(3, 2)
    # A negative discriminant proves the B-sector torque polynomial positive
    # for all epsilon; the other bounds support the high-e curvature branch.
    assert ((2 * pi + 2) ** 2 - 80).hi < 0
    assert 4 * (Q(1, 5) - Q(1, 5) ** 3 / 6) > Q(3, 4)
    assert (sqrt(I(7)) / 4).hi < Q(2, 3)
    assert Q(161, 324) + Q(5, 6) * Q(485, 4374) == Q(7733, 13122)
    assert Q(7733, 13122) > Q(7, 12)
    assert -Q(2, 3) * Q(7, 12) - 6 * Q(7, 12) ** 2 == -Q(175, 72)
    assert (Q(1, 2) + sqrt(I(2))).hi < 2


@pytest.mark.parametrize(
    "coordinates",
    (
        (Q(17, 10), Q(-1, 5), Q(6, 5), Q(-1, 2)),
        (Q(9, 5), Q(-1, 5), Q(7, 5), Q(-2, 5)),
    ),
)
def test_common_lower_endpoint_direction_matches_all_53_fine_edges(coordinates):
    candidate = _four(coordinates)
    assert all(value.lo > 0 for value in candidate.core.principal_margins)
    mp = mpmath.mp.clone()
    mp.dps = 90
    a, m, b, n = (mp.mpf(value.numerator) / value.denominator for value in coordinates)
    left_a, right_a, left_b, right_b = m - a, m + a, n - b, n + b
    resultant_a = 1 + 2 * mp.exp(1j * left_a) + 2 * mp.exp(1j * right_a)
    resultant_b = 1 + 2 * mp.exp(1j * left_b) + 2 * mp.exp(1j * right_b)
    middle_a, middle_b = mp.arg(resultant_a), mp.arg(resultant_b)
    radius_a, radius_b = abs(resultant_a), abs(resultant_b)
    gap_a, gap_b = (2 * mp.pi - 2 * a) / 3, (2 * mp.pi - 2 * b) / 3
    phases = (
        left_a,
        left_a,
        mp.mpf(0),
        middle_a,
        right_a,
        right_a,
        right_a + gap_a,
        right_a + gap_a,
        right_a + 2 * gap_a,
        right_a + 2 * gap_a,
        left_b,
        left_b,
        middle_b,
        middle_b,
        right_b,
        right_b,
        right_b + gap_b,
        right_b + gap_b,
        right_b + 2 * gap_b,
        right_b + 2 * gap_b,
        mp.mpf(0),
    )
    assert all(
        abs(phases[j] - phases[i] - 2 * mp.pi * turn) < mp.pi
        for (i, j), turn in zip(
            owner.SELECTED_SUPPORT, owner.SELECTED_SHIFTS, strict=True
        )
    )
    # Project the actual middle phasor. Nonzero Q makes replacing cos(Q)
    # by one observably wrong, while the correct identity remains exact.
    projection = (
        mp.cos(middle_a)
        + 2 * mp.cos(left_a - middle_a)
        + 2 * mp.cos(right_a - middle_a)
    )
    assert abs(projection - radius_a) < mp.mpf("1e-80")
    assert 1 - mp.cos(middle_a) > mp.mpf("0.01")
    direction = [mp.mpf(0)] * 21
    for node in (0, 1, 10, 11):
        direction[node] = mp.mpf(1)
    direction[3] = 2 * mp.cos(left_a - middle_a) / radius_a
    for node in (12, 13):
        direction[node] = 2 * mp.cos(left_b - middle_b) / radius_b
    for node in (6, 7, 16, 17):
        direction[node] = mp.mpf(1) / 3
    for node in (8, 9, 18, 19):
        direction[node] = mp.mpf(2) / 3
    fine_curvature = mp.fsum(
        mp.cos(phases[j] - phases[i]) * (direction[j] - direction[i]) ** 2
        for i, j in owner.SELECTED_SUPPORT
    )
    cosine_a = mp.cos(left_a - middle_a)
    cosine_b = mp.cos(left_b - middle_b)
    diagonal_a = (
        2 * mp.cos(left_a)
        + 2 * cosine_a
        + 4 * mp.cos(gap_a) / 3
        - 4 * cosine_a**2 / radius_a
    )
    diagonal_b = (
        2 * mp.cos(left_b)
        + 4 * cosine_b
        + 4 * mp.cos(gap_b) / 3
        - 8 * cosine_b**2 / radius_b
    )
    assert abs(fine_curvature - diagonal_a - diagonal_b) < mp.mpf("1e-80")
    # These probes check the exact direction and its multiplicities; they are
    # not full equilibria, nor a finite substitute for the global theorem.

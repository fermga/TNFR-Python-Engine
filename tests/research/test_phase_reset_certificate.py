"""Independent phase-reset controls; retained-response review runs read-only."""

from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.research.phase_reset_certificate import bound_phase_reset


def _review(before=(Q(0), Q(1, 4)), after=(Q(1, 8), Q(1, 4)), **kwargs):
    options = {"coefficient": Q(1, 2), "budget": Q(1, 2**46)}
    options.update(kwargs)
    return bound_phase_reset(before, after, ((1,), (0,)), (0,), **options)[0]


def test_one_neighbor_exact_half_angle_gives_independent_zero_defect():
    result = _review()
    assert result.ideal_angle.contains(Q(1, 4))
    assert result.ideal_displacement.contains(Q(1, 8))
    assert result.represented_displacement.contains(Q(1, 8))
    assert result.defect.contains(0)
    assert result.nearest_lift_certified
    assert result.classification == "certified_within_budget"


def test_wrong_displacement_is_certified_excess_without_tolerance_relaxation():
    result = _review(after=(Q(1, 4), Q(1, 4)))
    assert result.defect.contains(Q(1, 8))
    assert result.classification == "certified_exceeds_budget"


def test_exact_boundary_is_unresolved_when_outward_rounding_straddles_it():
    result = _review(after=(Q(1, 4), Q(1, 4)), budget=Q(1, 8))
    assert result.defect.lo < result.budget < result.defect.hi
    assert result.classification == "unresolved"


def test_common_phase_translation_preserves_the_complete_certificate():
    result = _review()
    offset = Q(7, 3)
    shifted = _review(
        before=(offset, offset + Q(1, 4)),
        after=(offset + Q(1, 8), offset + Q(1, 4)),
    )
    assert shifted == result


def test_reflection_reverses_the_bound_and_preserves_the_verdict():
    result = _review()
    reflected = _review(before=(Q(0), Q(-1, 4)), after=(Q(-1, 8), Q(-1, 4)))
    assert reflected.defect.hull(-result.defect).abs_max <= result.budget
    assert reflected.classification == result.classification


def test_symmetric_two_neighbor_resultant_has_exact_midpoint_angle():
    before = (Q(0), Q(1, 4), Q(3, 4))
    result = bound_phase_reset(
        before,
        (Q(1, 4), *before[1:]),
        ((1, 2), (0,), (0,)),
        (0,),
        coefficient=Q(1, 2),
        budget=Q(1, 2**46),
    )[0]
    # exp(i/4)+exp(3i/4)=2*cos(1/4)*exp(i/2), with cos(1/4)>0.
    assert result.ideal_angle.contains(Q(1, 2))
    assert result.defect.contains(0)
    assert result.classification == "certified_within_budget"


def test_wrapped_endpoint_retains_nearest_lift_without_storing_a_false_raw_delta():
    two_pi = Q(float(2 * mp.pi))
    result = _review(after=(Q(1, 8) + two_pi, Q(1, 4)))
    assert result.represented_displacement.abs_max < Q(1, 4)
    assert result.classification == "certified_within_budget"


def test_opposite_branch_principal_lift_cannot_certify_a_violation():
    result = _review(before=(Q(0), Q(3)), after=(Q(-3), Q(3)), coefficient=Q(1))
    assert result.defect.contains(-6)
    assert not result.nearest_lift_certified
    assert result.classification == "unresolved"


def test_regular_two_neighbor_nonexact_case_contains_independent_100_digit_oracle():
    before = (Q(1, 7), Q(-2, 5), Q(4, 3))
    result = bound_phase_reset(
        before,
        (Q(2, 7), *before[1:]),
        ((1, 2), (0,), (0,)),
        (0,),
        coefficient=Q(3, 10),
        budget=Q(1, 2**46),
    )[0]
    with mp.workdps(100):
        phases = tuple(mp.mpf(v.numerator) / v.denominator for v in before)
        resultant = sum(mp.exp(mp.j * (phase - phases[0])) for phase in phases[1:])
        expected = mp.mpf(1) / 7 - mp.mpf(3) / 10 * mp.arg(resultant)
        point = Q(mp.nstr(expected, 100))
    assert result.defect.contains(point)


@pytest.mark.parametrize("invalid", [True, 0.3, float("nan"), "1/2"])
def test_invalid_physical_scalar_rejects_before_interval_construction(invalid):
    with pytest.raises((TypeError, ValueError)):
        _review(coefficient=invalid)
    with pytest.raises((TypeError, ValueError)):
        _review(before=(invalid, Q(1, 4)))
    with pytest.raises((TypeError, ValueError)):
        _review(budget=invalid)


@pytest.mark.parametrize(
    "coefficient,budget", [(Q(-1), Q(1)), (Q(2), Q(1)), (Q(0), Q(-1))]
)
def test_invalid_policy_domain_rejects(coefficient, budget):
    with pytest.raises(ValueError):
        _review(coefficient=coefficient, budget=budget)


@pytest.mark.parametrize(
    "neighbors", [((1, 1), (0,)), ((0,), (0,)), ((2,), (0,)), ((True,), (0,))]
)
def test_invalid_bare_support_reuses_shared_admission(neighbors):
    with pytest.raises(ValueError):
        bound_phase_reset(
            (Q(0), Q(1)), (Q(0), Q(1)), neighbors, (0,), coefficient=1, budget=0
        )


def test_zero_resultant_rejects_instead_of_inventing_an_angle():
    with pytest.raises(ValueError):
        bound_phase_reset((Q(0),), (Q(0),), ((),), (0,), coefficient=1, budget=0)


@pytest.mark.parametrize("targets", [(True,), (0, 0), (), (2,)])
def test_invalid_targets_reject(targets):
    with pytest.raises(ValueError):
        bound_phase_reset(
            (Q(0), Q(1)), (Q(0), Q(1)), ((1,), (0,)), targets, coefficient=1, budget=0
        )

"""Hand balances and domain controls for conditional C6 phase/EPI steps."""

from fractions import Fraction
import math

import pytest

from tnfr.physics.coupling_winding import (
    derive_c6_winding_joint_domain, observe_c6_winding_joint_domain,
)


F = Fraction
BEFORE = (F(1, 96), F(-1, 96)) * 3
AFTER = (F(1, 120), F(-1, 120)) * 3
EPI = (F(1, 2),) * 6


def _reference(**overrides):
    values = {
        "coupling_phase_factor": F(1, 2), "coherence_phase_factor": F(1, 3),
        "capacity": 1, "epi_weight": F(1, 2), "phase_weight": F(1, 4),
        "timestep": F(1, 2), "epi_lower": F(1, 4), "epi_upper": F(3, 4),
    }
    values.update(overrides)
    return derive_c6_winding_joint_domain(**values)


def test_hand_reference_uses_a_nonlinear_overlap_bound_not_the_tangent_energy_factor():
    reference = _reference()
    assert reference.max_phase_oscillation_pi == F(1, 12)
    assert reference.closed_mean_partial_lower == F(1, 10)
    assert reference.nonlinear_oscillation_factor == F(14, 15)
    assert reference.forcing_step_factor == F(1, 8)
    assert reference.epi_phase_budget_weight == F(7, 4)
    assert reference.phase_reference.quotient_energy_bound == F(25, 64)
    assert reference.nonlinear_oscillation_factor**2 != reference.phase_reference.quotient_energy_bound
    assert reference.nodal_euler_matrix[0] == (F(3, 4), F(1, 8), 0, 0, 0, F(1, 8))
    assert reference.epi_disagreement_factor == F(7, 8)
    assert reference.strict_epi_convergence


def test_independent_alternating_phase_fixture_has_exact_pressure_endpoint_and_reserves():
    result = observe_c6_winding_joint_domain(
        _reference(), phase_before_pi=BEFORE, phase_after_pi=AFTER, epi=EPI,
    )
    assert result.phase_oscillation_before == F(1, 48)
    assert result.phase_oscillation_after == F(1, 60)
    assert result.phase_contraction_slack == F(1, 360)
    assert result.phase_interval_nesting_slack == (F(1, 480), F(1, 480))
    assert result.phase_pressure == (F(-1, 60), F(1, 60)) * 3
    assert result.modeled_pressure == result.modeled_rate == (F(-1, 240), F(1, 240)) * 3
    assert result.epi_after == (F(239, 480), F(241, 480)) * 3
    assert result.lower_reserve_before == F(89, 192)
    assert result.upper_reserve_before == F(103, 192)
    assert result.lower_reserve_after == F(15, 32)
    assert result.upper_reserve_after == F(17, 32)
    assert result.lower_reserve_gain == result.upper_reserve_gain == F(1, 192)
    assert result.mean_before == result.mean_after == F(1, 2)
    assert result.mean_identity_residual == 0


def test_equality_in_the_nonlinear_bound_can_exhaust_both_reserve_gains_exactly():
    reference = _reference()
    after = tuple(reference.nonlinear_oscillation_factor * z for z in BEFORE)
    result = observe_c6_winding_joint_domain(
        reference, phase_before_pi=BEFORE, phase_after_pi=after, epi=EPI,
    )
    assert result.phase_contraction_slack == 0
    assert result.epi_after == (F(1, 2) - F(7, 2880), F(1, 2) + F(7, 2880)) * 3
    assert result.lower_reserve_gain == result.upper_reserve_gain == 0


def test_initial_reserves_may_equal_the_declared_positive_application_band():
    reference = _reference(epi_lower=F(89, 192), epi_upper=F(103, 192))
    result = observe_c6_winding_joint_domain(
        reference, phase_before_pi=BEFORE, phase_after_pi=AFTER, epi=EPI,
    )
    assert result.lower_reserve_before == reference.epi_lower
    assert result.upper_reserve_before == reference.epi_upper
    assert min(result.epi_after) > reference.epi_lower
    assert max(result.epi_after) < reference.epi_upper


@pytest.mark.parametrize("fraction", (F(0), F(1, 4), F(1, 2), F(1)))
def test_declared_held_affine_internal_points_keep_the_same_future_reserve(fraction):
    reference = _reference()
    after = tuple(reference.nonlinear_oscillation_factor * z for z in BEFORE)
    result = observe_c6_winding_joint_domain(
        reference, phase_before_pi=BEFORE, phase_after_pi=after, epi=EPI,
    )
    # These are points of one held-pressure interval, not refreshed substeps.
    internal = tuple(value + fraction * reference.timestep * rate
                     for value, rate in zip(EPI, result.modeled_rate, strict=True))
    reserve = reference.epi_phase_budget_weight * result.phase_oscillation_after
    assert min(internal) - reserve >= result.lower_reserve_before
    assert max(internal) + reserve <= result.upper_reserve_before


def test_common_phase_rotation_does_not_change_pressure_or_epi_reserves():
    reference = _reference()
    result = observe_c6_winding_joint_domain(
        reference, phase_before_pi=BEFORE, phase_after_pi=AFTER, epi=EPI,
    )
    shifted = observe_c6_winding_joint_domain(
        reference, phase_before_pi=tuple(12 + z for z in BEFORE),
        phase_after_pi=tuple(12 + z for z in AFTER), epi=EPI,
    )
    assert shifted.phase_pressure == result.phase_pressure
    assert shifted.epi_after == result.epi_after
    assert shifted.lower_reserve_after == result.lower_reserve_after
    assert shifted.upper_reserve_after == result.upper_reserve_after


@pytest.mark.parametrize("alpha", (F(0), F(1, 3), F(1)))
def test_nonlinear_phase_overlap_reserve_is_valid_independently_of_the_IL_factor(alpha):
    reference = _reference(coherence_phase_factor=alpha)
    assert reference.nonlinear_oscillation_factor == F(14, 15)
    result = observe_c6_winding_joint_domain(
        reference, phase_before_pi=BEFORE, phase_after_pi=AFTER, epi=EPI,
    )
    assert result.lower_reserve_gain == F(1, 192)


def test_exact_budget_identity_and_summable_forcing_coefficient():
    reference = _reference()
    rho = reference.nonlinear_oscillation_factor
    b, budget = reference.forcing_step_factor, reference.epi_phase_budget_weight
    assert (b + budget) * rho == budget
    # The tail starts after the first phase contraction, before its nodal step.
    assert budget == b * rho / (1 - rho)
    assert b * sum((rho**j for j in range(1, 4)), F(0)) + budget * rho**3 == budget


@pytest.mark.parametrize("override", (
    {"coupling_phase_factor": 0}, {"coupling_phase_factor": 2},
    {"coherence_phase_factor": -1}, {"capacity": 0}, {"capacity": -1},
    {"epi_weight": 0}, {"epi_weight": -1}, {"phase_weight": -1},
    {"timestep": -1}, {"timestep": 3}, {"epi_lower": 0},
    {"epi_lower": -1}, {"epi_upper": 2}, {"epi_lower": F(7, 8)},
    {"capacity": True}, {"phase_weight": math.nan}, {"timestep": math.inf},
))
def test_invalid_joint_source_coefficients_or_band_are_rejected(override):
    with pytest.raises((TypeError, ValueError)):
        _reference(**override)


@pytest.mark.parametrize("epi", ((F(1, 4),) * 6, (F(3, 4),) * 6))
def test_pointwise_inside_the_epi_band_is_insufficient_without_future_forcing_reserve(epi):
    with pytest.raises(ValueError, match="future phase-forcing reserve"):
        observe_c6_winding_joint_domain(
            _reference(), phase_before_pi=BEFORE, phase_after_pi=AFTER, epi=epi,
        )


def test_phase_box_without_contraction_is_rejected():
    with pytest.raises(ValueError, match="nesting or nonlinear contraction"):
        observe_c6_winding_joint_domain(
            _reference(), phase_before_pi=BEFORE, phase_after_pi=BEFORE, epi=EPI,
        )


def test_contraction_without_phase_interval_nesting_is_rejected():
    shifted = tuple(z / 2 + F(1, 10) for z in BEFORE)
    with pytest.raises(ValueError, match="nesting or nonlinear contraction"):
        observe_c6_winding_joint_domain(
            _reference(), phase_before_pi=BEFORE, phase_after_pi=shifted, epi=EPI,
        )


def test_phase_oscillation_above_the_exact_box_is_rejected():
    before = (F(1, 12), F(-1, 12)) * 3
    with pytest.raises(ValueError, match="declared winding box"):
        observe_c6_winding_joint_domain(
            _reference(), phase_before_pi=before, phase_after_pi=AFTER, epi=EPI,
        )


@pytest.mark.parametrize("field,value", (
    ("phase_before_pi", (0,) * 5), ("phase_after_pi", (0,) * 7),
    ("epi", set(range(6))), ("epi", (True,) * 6),
    ("phase_before_pi", "123456"), ("phase_after_pi", (math.nan,) * 6),
))
def test_joint_endpoint_order_dimensions_and_scalar_domain_are_checked(field, value):
    values = {"phase_before_pi": BEFORE, "phase_after_pi": AFTER, "epi": EPI}
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        observe_c6_winding_joint_domain(_reference(), **values)

"""Exact additive defects and signed finite C6 reserve telescopes."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import pytest

from tnfr.physics.coupling_winding import (
    bound_c6_winding_defect_prefix, derive_c6_winding_joint_domain,
    observe_c6_winding_defect, observe_c6_winding_joint_domain,
)


F = Fraction
ZERO = (F(0),) * 6
HALF = (F(1, 2),) * 6


def _reference(**overrides):
    values = {
        "coupling_phase_factor": F(1, 2), "coherence_phase_factor": F(1, 3),
        "capacity": 1, "epi_weight": F(1, 2), "phase_weight": F(1, 4),
        "timestep": F(1, 2), "epi_lower": F(1, 4), "epi_upper": F(3, 4),
    }
    values.update(overrides)
    return derive_c6_winding_joint_domain(**values)


def _hand_defect():
    return observe_c6_winding_defect(
        _reference(), phase_before_pi=ZERO,
        phase_after_pi=(F(1, 120), F(-1, 120)) * 3, epi_before=HALF,
        epi_after=(F(5993, 12000), F(6031, 12000)) * 3,
    )


def _uniform_shift(reference, before, shift):
    return observe_c6_winding_defect(
        reference, phase_before_pi=ZERO, phase_after_pi=ZERO,
        epi_before=(before,) * 6, epi_after=(before + shift,) * 6,
    )


def test_hand_phase_injection_and_epi_defect_are_separated_exactly():
    result = _hand_defect()
    assert result.phase_oscillation_before == 0
    assert result.phase_oscillation_after == result.phase_oscillation_defect == F(1, 60)
    assert result.phase_interval_expansion == F(1, 120)
    assert result.phase_pressure == (F(-1, 60), F(1, 60)) * 3
    assert result.modeled_pressure == result.modeled_rate == (F(-1, 240), F(1, 240)) * 3
    assert result.modeled_epi_after == (F(239, 480), F(241, 480)) * 3
    assert result.endpoint_defect == (F(3, 2000), F(1, 2000)) * 3
    assert result.mean_defect == F(1, 1000)
    assert result.centered_defect == (F(1, 2000), F(-1, 2000)) * 3
    assert result.centered_defect_oscillation == F(1, 1000)
    assert result.endpoint_defect_infinity == F(3, 2000)
    assert result.mean_before == F(1, 2)
    assert result.mean_after == F(501, 1000)
    assert result.mean_identity_residual == 0
    assert result.phase_box_before and result.phase_box_after


def test_signed_losses_bound_the_actual_reserve_changes_and_absolute_cost():
    result = _hand_defect()
    assert result.lower_reserve_before == result.upper_reserve_before == F(1, 2)
    assert result.lower_reserve_after == F(1881, 4000)
    assert result.upper_reserve_after == F(2127, 4000)
    assert result.lower_loss_bound == F(123, 4000)
    assert result.upper_loss_bound == F(131, 4000)
    assert result.lower_bound_slack == result.upper_bound_slack == F(1, 1000)
    assert result.absolute_defect_cost == F(131, 4000)


def test_original_defect_free_observer_and_new_model_share_the_same_nodal_endpoint():
    reference = _reference()
    before = (F(1, 96), F(-1, 96)) * 3
    after = (F(1, 120), F(-1, 120)) * 3
    original = observe_c6_winding_joint_domain(
        reference, phase_before_pi=before, phase_after_pi=after, epi=HALF,
    )
    result = observe_c6_winding_defect(
        reference, phase_before_pi=before, phase_after_pi=after,
        epi_before=HALF, epi_after=original.epi_after,
    )
    assert result.modeled_epi_after == original.epi_after
    assert result.modeled_pressure == original.modeled_pressure
    assert result.phase_oscillation_defect == result.absolute_defect_cost == 0
    assert result.endpoint_defect == result.centered_defect == ZERO
    assert result.lower_bound_slack == original.lower_reserve_gain
    assert result.upper_bound_slack == original.upper_reserve_gain


def test_finite_prefix_preserves_signed_mean_cancellation_instead_of_summing_absolute_drift():
    reference = _reference()
    eta = F(1, 100)
    first = _uniform_shift(reference, F(1, 2), eta)
    second = _uniform_shift(reference, F(1, 2) + eta, -eta)
    prefix = bound_c6_winding_defect_prefix(reference, observations=(first, second))
    assert first.lower_loss_bound == -eta
    assert second.upper_loss_bound == -eta
    assert prefix.mean_defect_prefix == (0, eta, 0)
    assert prefix.signed_lower_loss_prefix == (0, -eta, 0)
    assert prefix.signed_upper_loss_prefix == (0, eta, 0)
    assert prefix.absolute_defect_cost_prefix == (0, eta, 2 * eta)
    assert prefix.lower_reserve_bounds == prefix.upper_reserve_bounds == (F(1, 2), F(51, 100), F(1, 2))
    assert prefix.mean_prefix_bound == eta
    assert prefix.phase_oscillation_envelope == (0, 0, 0)
    assert prefix.phase_class_preserved and prefix.epi_reserve_preserved


def test_mean_drift_can_leave_positive_band_while_phase_and_centered_defects_are_zero():
    reference = _reference()
    first = _uniform_shift(reference, F(1, 2), F(1, 5))
    second = _uniform_shift(reference, F(7, 10), F(1, 5))
    prefix = bound_c6_winding_defect_prefix(reference, observations=(first, second))
    assert first.centered_defect == second.centered_defect == ZERO
    assert prefix.mean_defect_prefix == (0, F(1, 5), F(2, 5))
    assert prefix.phase_class_preserved
    assert not prefix.epi_reserve_preserved
    assert prefix.upper_reserve_bounds[-1] == F(9, 10)


def test_finite_phase_envelope_propagates_a_nonzero_defect_through_a_later_exact_step():
    first = _hand_defect()
    reference = first.reference
    z2 = tuple(z / 2 for z in first.phase_after_pi)
    second_model = observe_c6_winding_joint_domain(
        reference, phase_before_pi=first.phase_after_pi, phase_after_pi=z2, epi=first.epi_after,
    )
    second = observe_c6_winding_defect(
        reference, phase_before_pi=first.phase_after_pi, phase_after_pi=z2,
        epi_before=first.epi_after, epi_after=second_model.epi_after,
    )
    prefix = bound_c6_winding_defect_prefix(reference, observations=(first, second))
    assert second.phase_oscillation_defect == 0
    assert second.endpoint_defect == ZERO
    assert prefix.phase_oscillation_envelope == (0, F(1, 60), F(7, 450))
    assert second.phase_oscillation_after == F(1, 120)
    assert prefix.mean_defect_prefix == (0, F(1, 1000), F(1, 1000))
    assert prefix.absolute_defect_cost_prefix == (0, F(131, 4000), F(131, 4000))
    assert prefix.phase_class_preserved and prefix.epi_reserve_preserved


def test_phase_box_failure_is_retained_as_observation_instead_of_rejected_or_certified():
    result = observe_c6_winding_defect(
        _reference(), phase_before_pi=ZERO, phase_after_pi=(F(1, 12), F(-1, 12)) * 3,
        epi_before=HALF, epi_after=HALF,
    )
    assert result.phase_box_before and not result.phase_box_after
    assert result.phase_oscillation_defect == F(1, 6)
    prefix = bound_c6_winding_defect_prefix(result.reference, observations=(result,))
    assert not prefix.phase_class_preserved


def test_phase_common_rotation_is_recorded_but_does_not_invent_pressure_or_oscillation_cost():
    result = observe_c6_winding_defect(
        _reference(), phase_before_pi=ZERO, phase_after_pi=(F(1, 10),) * 6,
        epi_before=HALF, epi_after=HALF,
    )
    assert result.phase_interval_expansion == F(1, 10)
    assert result.phase_oscillation_defect == result.absolute_defect_cost == 0
    assert result.phase_pressure == ZERO
    # This endpoint pair is not thereby authenticated as an actual UM/IL step.


def test_observer_and_finite_prefix_rederive_forged_public_caches():
    reference = _reference()
    forged_reference = replace(reference, phase_reference=None, nodal_euler_matrix=(),
                               nonlinear_oscillation_factor=99, epi_phase_budget_weight=0)
    result = _uniform_shift(forged_reference, F(1, 2), F(1, 100))
    assert result.reference == reference
    forged_result = replace(result, phase_oscillation_defect=99, endpoint_defect=(99,) * 6,
                            mean_defect=99, lower_loss_bound=99, upper_loss_bound=99,
                            absolute_defect_cost=0)
    prefix = bound_c6_winding_defect_prefix(forged_reference, observations=(forged_result,))
    assert prefix.observations == (result,)
    assert prefix.mean_defect_prefix == (0, F(1, 100))
    assert prefix.absolute_defect_cost_prefix == (0, F(1, 100))


@pytest.mark.parametrize("mismatch", ("epi", "phase", "reference"))
def test_finite_chain_requires_actual_primitive_endpoint_adjacency_and_common_coefficients(mismatch):
    reference = _reference()
    first = _uniform_shift(reference, F(1, 2), 0)
    ref2 = _reference(timestep=F(1, 4)) if mismatch == "reference" else reference
    x2 = (F(3, 5),) * 6 if mismatch == "epi" else HALF
    z2 = (F(1, 10),) * 6 if mismatch == "phase" else ZERO
    second = observe_c6_winding_defect(
        ref2, phase_before_pi=z2, phase_after_pi=z2, epi_before=x2, epi_after=x2,
    )
    with pytest.raises(ValueError, match="adjacent endpoints|same joint-domain"):
        bound_c6_winding_defect_prefix(reference, observations=(first, second))


@pytest.mark.parametrize("observations", ((), {}, "records", (None,)))
def test_finite_prefix_rejects_empty_or_invalid_observation_sequences(observations):
    with pytest.raises((TypeError, ValueError)):
        bound_c6_winding_defect_prefix(_reference(), observations=observations)


def test_unordered_observations_are_not_a_finite_prefix():
    result = _uniform_shift(_reference(), F(1, 2), 0)
    with pytest.raises(TypeError, match="ordered sequence"):
        bound_c6_winding_defect_prefix(result.reference, observations={result})


@pytest.mark.parametrize("field,value", (
    ("epi_before", (0,) * 5), ("epi_after", (0,) * 7),
    ("phase_before_pi", {i: 0 for i in range(6)}),
    ("phase_after_pi", (math.nan,) * 6), ("epi_after", (True,) * 6),
))
def test_defect_observer_validates_exact_endpoint_dimensions_and_scalar_inputs(field, value):
    inputs = {"phase_before_pi": ZERO, "phase_after_pi": ZERO,
              "epi_before": HALF, "epi_after": HALF}
    inputs[field] = value
    with pytest.raises((TypeError, ValueError)):
        observe_c6_winding_defect(_reference(), **inputs)


def test_public_defect_and_prefix_records_are_immutable():
    result = _hand_defect()
    prefix = bound_c6_winding_defect_prefix(result.reference, observations=(result,))
    with pytest.raises(FrozenInstanceError):
        result.mean_defect = 0
    with pytest.raises(FrozenInstanceError):
        prefix.mean_prefix_bound = 0

"""Independent rational fixtures for the conditional P2 feedback model."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from tnfr.physics.capacity_feedback import (
    bound_p2_capacity_feedback, derive_p2_capacity_feedback,
    observe_p2_capacity_feedback_cycle,
)


F = Fraction


def _reference(**overrides):
    values = {
        "base_capacity": 1, "epi_weight": 1, "vf_weight": F(1, 2),
        "coupling_factor": F(1, 2), "timestep": F(1, 4),
        "max_capacity_gap": 1, "epi_lower": F(1, 4), "epi_upper": F(3, 4),
    }
    values.update(overrides)
    return derive_p2_capacity_feedback(**values)


def test_independent_first_cycle_retains_both_signed_mean_changes():
    ref = _reference()
    cycle = observe_p2_capacity_feedback_cycle(ref, capacity_gap=1, epi=(F(1, 4), F(3, 4)))
    assert ref.forcing_ratio == ref.capacity_retention == ref.contraction_factor == F(1, 2)
    assert ref.strict_disagreement_contraction
    assert cycle.capacity_gap_after == F(1, 2)
    assert cycle.capacity_after == (F(3, 2), 1)
    assert cycle.epi_after == (F(11, 32), F(11, 16))
    assert cycle.lifted_before == (F(5, 4), F(5, 4))
    assert cycle.lifted_after_event == (1, F(5, 4))
    assert cycle.lifted_after == (F(35, 32), F(19, 16))
    assert cycle.modeled_pressure == (F(1, 4), F(-1, 4))
    assert cycle.nodal_rate == (F(3, 8), F(-1, 4))
    assert cycle.euler_mix == (F(3, 8), F(1, 4))
    assert cycle.flow_multiplier == F(3, 8)
    assert cycle.disagreement_after == F(-11, 32)
    assert cycle.lifted_disagreement_before == 0
    assert cycle.lifted_disagreement_after_event == F(-1, 4)
    assert cycle.lifted_disagreement_after == F(-3, 32)
    assert cycle.arithmetic_mean_before == F(1, 2)
    assert cycle.arithmetic_mean_after == F(33, 64)
    assert cycle.arithmetic_mean_change == cycle.arithmetic_mean_flow_change == F(1, 64)
    assert cycle.metric_mean_before == F(7, 12)
    assert cycle.metric_mean_after_event == cycle.metric_mean_after == F(11, 20)
    assert cycle.metric_mean_change == cycle.metric_mean_reset_change == F(-1, 30)
    assert cycle.before_invariant_domain and cycle.after_event_invariant_domain
    assert cycle.after_invariant_domain and cycle.output_in_declared_interval
    assert all(getattr(cycle, name) == 0 for name in (
        "disagreement_identity_residual", "lifted_identity_residual",
        "arithmetic_mean_identity_residual", "metric_mean_identity_residual",
        "metric_flow_mean_identity_residual",
    ))


def test_second_cycle_shows_that_the_lifted_error_need_not_decrease_each_time():
    cycle = observe_p2_capacity_feedback_cycle(
        _reference(), capacity_gap=F(1, 2), epi=(F(11, 32), F(11, 16)),
    )
    assert cycle.capacity_gap_after == F(1, 4)
    assert cycle.lifted_disagreement_after_event == F(-7, 32)
    assert cycle.flow_multiplier == F(7, 16)
    assert cycle.lifted_disagreement_after == F(-49, 512)
    assert abs(cycle.lifted_disagreement_after) > abs(cycle.lifted_disagreement_before)
    assert cycle.disagreement_after == F(-113, 512)
    assert cycle.arithmetic_mean_after == F(535, 1024)
    assert cycle.arithmetic_mean_change == F(7, 1024)
    assert cycle.epi_after == (F(211, 512), F(81, 128))


def test_equal_beta_and_rho_use_the_exact_finite_convolution():
    bound = bound_p2_capacity_feedback(
        _reference(), capacity_gap=1, epi=(F(1, 4), F(3, 4)), cycles=2,
    )
    assert bound.geometric_convolution == 1
    assert bound.capacity_gap_after == F(1, 4)
    assert bound.lifted_disagreement_bound == F(1, 8)
    assert bound.disagreement_bound == F(1, 4)
    assert bound.arithmetic_mean_tail_bound == F(1, 64)
    assert bound.limiting_mean_interval == (F(7, 16), F(9, 16))
    assert bound.consensus_endpoint_bound == F(9, 64)
    assert bound.strict_disagreement_contraction


def test_distinct_beta_and_rho_have_the_independent_convolution_value():
    ref = _reference(coupling_factor=F(1, 4))
    bound = bound_p2_capacity_feedback(ref, capacity_gap=1, epi=(F(1, 4), F(3, 4)), cycles=2)
    assert ref.contraction_factor == F(1, 2)
    assert ref.capacity_retention == F(3, 4)
    assert bound.geometric_convolution == F(5, 4)
    assert bound.lifted_disagreement_bound == F(5, 64)
    assert bound.capacity_gap_after == F(9, 16)
    assert bound.disagreement_bound == F(23, 64)
    assert bound.arithmetic_mean_tail_bound == F(27, 256)


@pytest.mark.parametrize("cycles", (0, 1, 2, 5))
def test_zero_beta_handles_the_first_exact_consensus_step(cycles):
    ref = _reference(max_capacity_gap=0, timestep=F(1, 2))
    assert ref.contraction_factor == 0
    bound = bound_p2_capacity_feedback(ref, capacity_gap=0, epi=(F(1, 4), F(3, 4)), cycles=cycles)
    assert bound.lifted_disagreement_bound == (F(1, 2) if cycles == 0 else 0)
    assert bound.disagreement_bound == bound.lifted_disagreement_bound
    assert bound.arithmetic_mean_tail_bound == 0
    if cycles:
        assert bound.geometric_convolution == F(1, 2)**(cycles - 1)
        assert bound.consensus_endpoint_bound == 0
    cycle = observe_p2_capacity_feedback_cycle(ref, capacity_gap=0, epi=(F(1, 4), F(3, 4)))
    assert cycle.epi_after == (F(1, 2), F(1, 2))


def test_zero_cycles_keeps_the_initial_bound_without_a_negative_power():
    bound = bound_p2_capacity_feedback(
        _reference(), capacity_gap=1, epi=(F(1, 4), F(3, 4)), cycles=0,
    )
    assert bound.geometric_convolution == bound.lifted_disagreement_bound == 0
    assert bound.capacity_gap_after == 1
    assert bound.disagreement_bound == F(1, 2)
    assert bound.arithmetic_mean_tail_bound == F(1, 16)


def test_exact_repetition_stays_in_the_box_and_respects_the_finite_envelope():
    ref = _reference()
    initial, gap, epi = (F(1, 4), F(3, 4)), F(1), (F(1, 4), F(3, 4))
    initial_mean = sum(initial) / 2
    for count in range(1, 17):
        cycle = observe_p2_capacity_feedback_cycle(ref, capacity_gap=gap, epi=epi)
        bound = bound_p2_capacity_feedback(ref, capacity_gap=1, epi=initial, cycles=count)
        assert cycle.capacity_gap_after == bound.capacity_gap_after
        assert abs(cycle.lifted_disagreement_after) <= bound.lifted_disagreement_bound
        assert abs(cycle.disagreement_after) <= bound.disagreement_bound
        assert cycle.after_invariant_domain and cycle.output_in_declared_interval
        assert abs(cycle.arithmetic_mean_after - initial_mean) <= F(1, 16)
        gap, epi = cycle.capacity_gap_after, cycle.epi_after
    assert bound.disagreement_bound < F(1, 1000)
    assert cycle.arithmetic_mean_after != initial_mean


def test_convex_swap_boundary_is_invariant_without_disagreement_convergence():
    ref = _reference(max_capacity_gap=0, timestep=1)
    cycle = observe_p2_capacity_feedback_cycle(ref, capacity_gap=0, epi=(F(1, 4), F(3, 4)))
    assert ref.contraction_factor == 1
    assert not ref.strict_disagreement_contraction
    assert cycle.epi_after == (F(3, 4), F(1, 4))
    assert cycle.flow_multiplier == -1
    assert cycle.after_invariant_domain
    bound = bound_p2_capacity_feedback(ref, capacity_gap=0, epi=(F(1, 4), F(3, 4)), cycles=30)
    assert not bound.strict_disagreement_contraction
    assert bound.disagreement_bound == F(1, 2)
    assert bound.arithmetic_mean_tail_bound == 0
    assert bound.consensus_endpoint_bound == F(1, 4)


def test_zero_timestep_preserves_epi_while_capacity_changes():
    ref = _reference(timestep=0)
    cycle = observe_p2_capacity_feedback_cycle(ref, capacity_gap=1, epi=(F(1, 4), F(3, 4)))
    assert cycle.capacity_gap_after == F(1, 2)
    assert cycle.epi_after == (F(1, 4), F(3, 4))
    assert cycle.arithmetic_mean_change == 0
    assert cycle.metric_mean_change == F(-1, 30)
    assert not ref.strict_disagreement_contraction
    bound = bound_p2_capacity_feedback(ref, capacity_gap=1, epi=cycle.epi, cycles=4)
    assert bound.lifted_disagreement_bound == F(15, 32)
    assert bound.disagreement_bound == F(1, 2)
    assert bound.arithmetic_mean_tail_bound == 0


def test_zero_capacity_channel_reduces_to_the_expected_heterogeneous_diffusion():
    cycle = observe_p2_capacity_feedback_cycle(
        _reference(vf_weight=0), capacity_gap=1, epi=(F(1, 4), F(3, 4)),
    )
    assert cycle.epi_after == (F(7, 16), F(5, 8))
    assert cycle.lifted_disagreement_after == cycle.disagreement_after == F(-3, 16)
    assert cycle.arithmetic_mean_change == F(1, 32)


@pytest.mark.parametrize(("value", "expected"), (
    (F(1, 16), (F(-1, 8), F(3, 16))),
    (F(3, 16), (F(0), F(5, 16))),
))
def test_convex_flow_outside_the_box_can_lose_positive_or_nonzero_epi(value, expected):
    ref = _reference(timestep=F(1, 2), epi_lower=F(1, 16))
    cycle = observe_p2_capacity_feedback_cycle(ref, capacity_gap=1, epi=(value, value))
    assert cycle.euler_mix == (F(3, 4), F(1, 2))
    assert not cycle.before_invariant_domain
    assert not cycle.after_event_invariant_domain
    assert not cycle.after_invariant_domain
    assert cycle.epi_after == expected
    assert not cycle.output_in_declared_interval
    with pytest.raises(ValueError, match="initial invariant"):
        bound_p2_capacity_feedback(ref, capacity_gap=1, epi=(value, value), cycles=1)
    # Negative EPI alone does not refute UM's absolute-EPI admission gate.


def test_translation_preserves_disagreement_and_changes_the_mean_by_the_shift():
    base_ref = _reference()
    shifted_ref = _reference(epi_lower=F(5, 4), epi_upper=F(7, 4))
    base = observe_p2_capacity_feedback_cycle(base_ref, capacity_gap=1, epi=(F(1, 4), F(3, 4)))
    shifted = observe_p2_capacity_feedback_cycle(shifted_ref, capacity_gap=1, epi=(F(5, 4), F(7, 4)))
    assert shifted.epi_after == tuple(value + 1 for value in base.epi_after)
    for name in (
        "disagreement_after", "lifted_disagreement_after", "arithmetic_mean_change",
        "metric_mean_change", "modeled_pressure", "nodal_rate",
    ):
        assert getattr(shifted, name) == getattr(base, name)
    assert shifted.arithmetic_mean_after - base.arithmetic_mean_after == 1


@pytest.mark.parametrize("override", (
    {"base_capacity": 0}, {"epi_weight": 0}, {"vf_weight": -1},
    {"coupling_factor": 0}, {"coupling_factor": 1}, {"timestep": -1},
    {"max_capacity_gap": -1}, {"epi_lower": 0}, {"epi_upper": F(1, 8)},
    {"max_capacity_gap": 2}, {"timestep": 1}, {"epi_weight": float("nan")},
))
def test_invalid_or_uncertifiable_reference_domains_are_rejected(override):
    with pytest.raises((TypeError, ValueError)):
        _reference(**override)


@pytest.mark.parametrize(("gap", "epi"), (
    (-1, (F(1, 4), F(3, 4))), (2, (F(1, 4), F(3, 4))),
    (0, (1,)), (0, (1, 2, 3)), (0, (float("inf"), 1)),
))
def test_invalid_cycle_inputs_are_rejected(gap, epi):
    with pytest.raises((TypeError, ValueError)):
        observe_p2_capacity_feedback_cycle(_reference(), capacity_gap=gap, epi=epi)


@pytest.mark.parametrize("cycles", (-1, True, 2.5, "2"))
def test_envelope_rejects_noninteger_or_negative_cycle_counts(cycles):
    with pytest.raises(ValueError, match="nonnegative integer"):
        bound_p2_capacity_feedback(_reference(), capacity_gap=1, epi=(F(1, 4), F(3, 4)), cycles=cycles)


def test_public_reference_caches_are_rebuilt_and_outputs_are_frozen():
    reference = _reference()
    forged = replace(
        reference, forcing_ratio=99, capacity_retention=99, step_coefficient=99,
        contraction_factor=99, strict_disagreement_contraction=False,
    )
    expected = observe_p2_capacity_feedback_cycle(reference, capacity_gap=1, epi=(F(1, 4), F(3, 4)))
    actual = observe_p2_capacity_feedback_cycle(forged, capacity_gap=1, epi=(F(1, 4), F(3, 4)))
    assert actual == expected
    expected_bound = bound_p2_capacity_feedback(reference, capacity_gap=1, epi=actual.epi, cycles=2)
    assert bound_p2_capacity_feedback(forged, capacity_gap=1, epi=actual.epi, cycles=2) == expected_bound
    with pytest.raises(FrozenInstanceError):
        actual.epi_after = (0, 0)
    with pytest.raises(FrozenInstanceError):
        expected_bound.disagreement_bound = 0
    with pytest.raises(TypeError, match="reference"):
        observe_p2_capacity_feedback_cycle(None, capacity_gap=1, epi=(F(1, 4), F(3, 4)))

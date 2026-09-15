"""Independent cycle-gain and pre-REMESH history controls."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import numpy as np
import pytest

from tnfr.constants import DEFAULTS
from tnfr.physics.cycle_memory_relaxation import certify_cycle_memory_relaxation
from tnfr.physics.remesh_history_stability import (
    observe_uniform_remesh_history_transition,
)
from tnfr.utils import normalize_weights


F = Fraction


def _energy(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / 2


def _flow(values, reference):
    for coefficient in reference.euler_step_coefficients:
        values = tuple(
            (1 - coefficient) * value
            + coefficient * (values[i - 1] + values[(i + 1) % len(values)]) / 2
            for i, value in enumerate(values)
        )
    return values


def _advance(history, reference):
    transition = observe_uniform_remesh_history_transition(
        reference.remesh_certificate, history, (1,) * reference.node_count
    )
    post_remesh = transition.exact_next_field
    next_pre_remesh = _flow(post_remesh, reference)
    return (next_pre_remesh,) + history[:-1], post_remesh


def _augmented(history, reference):
    return sum(
        weight * _energy(row)
        for weight, row in zip(
            reference.remesh_certificate.stationary_distribution, history
        )
    )


def test_rational_gains_use_the_whole_partition_and_existing_sealed_owners():
    reference = certify_cycle_memory_relaxation(8, epi_weight=F(1, 2), alpha=F(1, 2))
    assert reference.spectral_gap_lower_bound == F(1, 8)
    assert reference.cycle_duration == F(1, 2)
    assert reference.euler_step_coefficients == (F(1, 8), F(1, 8))
    assert reference.euler_energy_gain_upper_bound == F(63, 64) ** 4
    assert reference.continuous_energy_gain_upper_bound == F(16, 17)
    for policy, gain in (
        (reference.euler_policy, reference.euler_energy_gain_upper_bound),
        (reference.continuous_policy, reference.continuous_energy_gain_upper_bound),
    ):
        assert policy.remesh_certificate is reference.remesh_certificate
        assert policy.policy_stability_certificate_certified
        assert policy.schedule_energy_gain_upper_bound == gain
        assert policy.exact_uniform_normalized_block_margin_lower_bound == 1 - gain
        assert not policy.binary64_runtime_stability_certified
        assert not policy.runtime_schedule_maps_verified


def test_default_weight_and_alpha_are_materialized_without_refitting():
    reference = certify_cycle_memory_relaxation(8)
    weights = normalize_weights(
        DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
    )
    assert reference.epi_weight == F.from_float(weights["epi"])
    assert reference.remesh_certificate.alpha == F.from_float(DEFAULTS["REMESH_ALPHA"])


@pytest.mark.parametrize("count", [3, 4, 8, 16])
def test_full_state_cycle_gain_contains_a_multimode_local_bump(count):
    reference = certify_cycle_memory_relaxation(
        count, epi_weight=1, step_sizes=(F(1, 4), F(1, 8))
    )
    initial = (F(2),) + (F(1),) * (count - 1)
    endpoint = _flow(initial, reference)
    assert sum(endpoint) == sum(initial)
    assert _energy(endpoint) <= (
        reference.euler_energy_gain_upper_bound * _energy(initial)
    )
    assert _energy(endpoint) < _energy(initial)
    assert reference.euler_energy_gain_upper_bound < 1


@pytest.mark.parametrize("alpha,history_length", [(0, 1), (F(1, 2), 4), (1, 2)])
def test_active_delays_and_finite_history_envelope_match_the_runtime_order(
    alpha, history_length,
):
    reference = certify_cycle_memory_relaxation(
        8, epi_weight=1, step_sizes=(F(1, 4),), alpha=alpha,
        tau_local=3, tau_global=1,
    )
    assert reference.euler_policy.history_length == history_length
    initial = history = tuple(
        tuple(F((i * i + 3 * lag) % 7 - 3, 5) for i in range(8))
        for lag in range(history_length)
    )
    initial_energy = _augmented(initial, reference)
    previous = initial_energy
    for count in range(1, 3 * history_length + 1):
        history, _ = _advance(history, reference)
        energy = _augmented(history, reference)
        bound = reference.euler_policy.exact_cycle_energy_gain_upper_bound(count)
        assert energy <= previous
        assert energy <= bound * initial_energy
        previous = energy


def test_next_history_head_is_scheduled_post_remesh_not_raw_post_remesh():
    reference = certify_cycle_memory_relaxation(
        4, epi_weight=1, step_sizes=(F(1, 4),), alpha=F(1, 2)
    )
    history = ((F(1), F(0), F(-1), F(0)), (F(2), F(0), F(-2), F(0)))
    new_history, raw = _advance(history, reference)
    assert raw == (F(7, 4), 0, F(-7, 4), 0)
    assert new_history[0] == (F(21, 16), 0, F(-21, 16), 0)
    assert new_history[1] == history[0]
    assert new_history[0] != raw


def test_pure_delay_damps_each_spatial_lineage_without_damping_uniform_means():
    reference = certify_cycle_memory_relaxation(
        4, epi_weight=1, step_sizes=(F(1, 4),), alpha=1,
        tau_local=3, tau_global=1,
    )
    uniform = ((F(1),) * 4, (F(3),) * 4)
    once, _ = _advance(uniform, reference)
    twice, _ = _advance(once, reference)
    assert once == uniform[::-1] and twice == uniform
    assert _augmented(uniform, reference) == 0
    mode = (F(1), 0, F(-1), 0)
    history = (mode, mode)
    once, _ = _advance(history, reference)
    twice, _ = _advance(once, reference)
    assert twice == tuple(tuple(F(3, 4) * v for v in row) for row in history)
    assert _augmented(twice, reference) == F(9, 16) * _augmented(history, reference)


def test_common_initial_mean_is_preserved_through_both_cycle_boundaries():
    reference = certify_cycle_memory_relaxation(8, alpha=1)
    row = (F(2),) + (F(1),) * 7
    history = (row, row[::-1])
    for _ in range(6):
        history, post = _advance(history, reference)
        assert sum(post) == 9
        assert all(sum(values) == 9 for values in history)


def test_positive_mode_memory_slows_the_declared_startup_but_still_relaxes():
    reference = certify_cycle_memory_relaxation(
        4, epi_weight=1, step_sizes=(F(1, 4),), alpha=F(1, 2)
    )
    mode = (F(1), 0, F(-1), 0)
    history = (mode, mode)
    previous = F(1)
    for count in range(1, 9):
        history, _ = _advance(history, reference)
        amplitude = history[0][0]
        assert F(3, 4) ** count <= amplitude <= previous
        previous = amplitude
    assert previous < 1
    assert history[0][0] > F(3, 4) ** 8


def test_memory_can_increase_current_energy_inside_a_decreasing_augmented_budget():
    reference = certify_cycle_memory_relaxation(
        4, epi_weight=1, step_sizes=(F(1, 4),), alpha=F(1, 2)
    )
    history = ((F(0),) * 4, (F(1), 0, F(-1), 0))
    after, raw = _advance(history, reference)
    assert _energy(raw) > _energy(history[0]) == 0
    assert _augmented(after, reference) < _augmented(history, reference)


def test_scalar_gain_bound_is_a_bound_not_an_exact_cycle_gap():
    reference = certify_cycle_memory_relaxation(
        4, epi_weight=1, step_sizes=(F(1, 4),)
    )
    # C4 has nonconstant eigenvalue 1; the rational generic lower bound is 1/2.
    mode = (F(1), F(0), F(-1), F(0))
    actual_gain = _energy(_flow(mode, reference)) / _energy(mode)
    assert reference.spectral_gap_lower_bound == F(1, 2)
    assert actual_gain == F(9, 16) < reference.euler_energy_gain_upper_bound


def test_detached_inputs_exact_reader_and_nested_seal():
    steps = [F(1, 4), F(1, 4)]
    result = certify_cycle_memory_relaxation(
        np.int64(8), capacity=F(np.int64(1)), epi_weight=F(1, 2),
        step_sizes=steps, alpha=np.float64(0.5), tau_local=np.int64(1),
    )
    steps[0] = 100
    assert result.step_sizes == (F(1, 4), F(1, 4))
    assert type(result.capacity.numerator) is int
    assert result.remesh_certificate.alpha == F(1, 2)
    changed = replace(result.euler_policy, schedule_energy_gain_upper_bound=F(1))
    assert not changed.policy_stability_certificate_certified
    with pytest.raises(FrozenInstanceError):
        result.capacity = F(0)


@pytest.mark.parametrize(
    "kwargs", [{"capacity": 0}, {"capacity": -1}, {"capacity": True},
               {"epi_weight": 0}, {"epi_weight": float("inf")},
               {"step_sizes": ()}, {"step_sizes": (0,)},
               {"step_sizes": (F(-1, 4),)}, {"step_sizes": {F(1, 4)}},
               {"epi_weight": 1, "step_sizes": (F(3, 4),)},
               {"alpha": F(-1, 2)}, {"alpha": F(3, 2)},
               {"tau_local": 0}, {"tau_global": True}],
)
def test_invalid_declared_domain_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        certify_cycle_memory_relaxation(8, **kwargs)


@pytest.mark.parametrize("count", [2, 0, True, 8.0])
def test_invalid_cycle_size_is_rejected(count):
    with pytest.raises((TypeError, ValueError)):
        certify_cycle_memory_relaxation(count)

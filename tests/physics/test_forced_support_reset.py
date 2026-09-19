"""Independent same-EPI controls for changing forced-support references."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_pattern,
    observe_forced_support_reset,
    observe_forced_support_state,
)
from tnfr.physics.support_transport import observe_support_transport

F = Fraction


def _reference(*, capacity=(1, 2), weight=1, forcing=(F(1, 4), F(1, 2))):
    graph = nx.path_graph(2)
    graph.edges[0, 1]["weight"] = weight
    for node, x, nu in zip(graph, (1, 0), capacity, strict=True):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: x,
                ALIAS_VF[0]: nu,
                ALIAS_DNFR[0]: 0,
            }
        )
    return derive_forced_support_balance(
        observe_support_transport(graph),
        epi_weight=F(1, 2),
        forcing=forcing,
    )


def _reset(before_reference, after_reference):
    return observe_forced_support_reset(
        before_reference,
        after_reference,
        before_reference.source,
        after_reference.source,
    )


def _pattern(reference, epi):
    return observe_forced_support_pattern(reference, nodes=(0, 1), epi=epi)


def test_new_profile_can_eliminate_error_without_recovering_old_pattern():
    old = _reference()
    new = _reference(forcing=(F(1, 2), F(-1, 2)))
    reset = _reset(old, new)
    assert old.relative_profile == (F(-1, 6), F(1, 3))
    assert new.relative_profile == (F(1, 3), F(-2, 3))
    assert reset.before.relative_error == (F(1, 2), -1)
    assert reset.after.relative_error == (0, 0)
    assert reset.mean_reweighting == 0
    assert reset.profile_shift == (F(1, 2), -1)
    assert reset.error_shift == (F(-1, 2), 1)
    assert reset.drift_change == F(-1, 2)
    assert reset.raw_support_reset.energy_change == 0
    assert reset.error_support_reset.energy_change == 0

    variance = reset.variance_budget
    assert variance.metric_term == 0
    assert variance.reference_cross_term == F(-3, 4)
    assert variance.reference_quadratic_term == F(3, 8)
    assert variance.energy_change == F(-3, 8)
    assert variance.identity_residual == 0
    energy = reset.dirichlet_budget
    assert energy.metric_term == 0
    assert energy.reference_cross_term == F(-9, 4)
    assert energy.reference_quadratic_term == F(9, 8)
    assert energy.energy_change == F(-9, 8)
    assert energy.identity_residual == 0

    fixed_before = _pattern(old, old.source.epi)
    fixed_after = _pattern(old, new.source.epi)
    assert fixed_before == fixed_after
    assert fixed_after.error_variance == F(3, 8)
    assert reset.after.error_variance == 0


def test_uniform_capacity_change_only_rescales_the_variance_metric():
    reset = _reset(_reference(), _reference(capacity=(2, 4)))
    assert reset.after_reference.metric_weights == (F(1, 2), F(1, 4))
    assert reset.after_reference.mean_drift == 1
    assert reset.mean_reweighting == 0
    assert reset.profile_shift == reset.error_shift == (0, 0)
    assert reset.variance_budget.metric_term == F(-3, 16)
    assert reset.variance_budget.energy_change == F(-3, 16)
    assert reset.variance_budget.reference_cross_term == 0
    assert reset.variance_budget.reference_quadratic_term == 0
    assert reset.dirichlet_budget.energy_change == 0
    assert reset.raw_support_reset.energy_change == 0


def test_uniform_conductance_change_scales_both_metrics_without_new_profile():
    reset = _reset(_reference(), _reference(weight=2))
    assert reset.profile_shift == reset.error_shift == (0, 0)
    assert reset.mean_reweighting == reset.drift_change == 0
    assert reset.variance_budget.metric_term == F(3, 8)
    assert reset.variance_budget.energy_change == F(3, 8)
    assert reset.dirichlet_budget.metric_term == F(9, 8)
    assert reset.dirichlet_budget.energy_change == F(9, 8)
    assert reset.raw_support_reset.energy_change == F(1, 2)
    assert reset.error_support_reset.energy_change == F(9, 8)
    assert reset.error_support_reset.before.epi == (F(1, 2), -1)
    assert reset.error_support_reset.after.epi == (F(1, 2), -1)


def test_heterogeneous_capacity_reset_has_signed_metric_and_profile_terms():
    reset = _reset(_reference(), _reference(capacity=(1, 1)))
    assert reset.after_reference.relative_profile == (F(-1, 8), F(1, 8))
    assert reset.mean_reweighting == F(-1, 6)
    assert reset.profile_shift == (F(1, 24), F(-5, 24))
    assert reset.error_shift == (F(1, 8), F(3, 8))
    variance = reset.variance_budget
    assert variance.metric_term == F(1, 4)
    assert variance.reference_cross_term == F(-5, 16)
    assert variance.reference_quadratic_term == F(5, 64)
    assert variance.energy_change == F(1, 64)
    assert variance.identity_residual == 0
    energy = reset.dirichlet_budget
    assert energy.metric_term == 0
    assert energy.reference_cross_term == F(-3, 8)
    assert energy.reference_quadratic_term == F(1, 32)
    assert energy.energy_change == F(-11, 32)
    assert energy.identity_residual == 0


def test_references_predating_event_use_actual_event_epi_for_every_budget():
    old, new = _reference(), _reference(capacity=(1, 1))
    before = replace(old.source, epi=(F(3), F(-1)))
    after = replace(new.source, epi=(F(3), F(-1)))
    reset = observe_forced_support_reset(old, new, before, after)
    assert old.source.epi == new.source.epi == (1, 0)
    assert reset.before.snapshot.epi == reset.after.snapshot.epi == (3, -1)
    assert reset.before.relative_error == (F(3, 2), -3)
    assert reset.after.relative_error == (F(17, 8), F(-17, 8))
    assert reset.mean_reweighting == F(-2, 3)
    assert reset.variance_budget.metric_term == F(9, 4)
    assert reset.variance_budget.reference_cross_term == F(-27, 16)
    assert reset.variance_budget.reference_quadratic_term == F(37, 64)
    assert reset.variance_budget.energy_change == F(73, 64)
    assert reset.dirichlet_budget.energy_change == F(-35, 32)


def test_forward_and_reverse_reference_reset_telescope_at_fixed_epi():
    old, new = _reference(), _reference(capacity=(1, 1), weight=2)
    forward, reverse = _reset(old, new), _reset(new, old)
    assert forward.mean_reweighting + reverse.mean_reweighting == 0
    for field in ("variance_budget", "dirichlet_budget", "raw_support_reset"):
        assert (
            getattr(forward, field).energy_change
            + getattr(
                reverse,
                field,
            ).energy_change
            == 0
        )


def test_pressure_only_write_does_not_change_either_profile_error():
    reference = _reference()
    after = replace(reference.source, stored_pressure=(F(7), F(-3)))
    reset = observe_forced_support_reset(reference, reference, reference.source, after)
    assert reset.before.pressure_defect != reset.after.pressure_defect
    assert reset.mean_reweighting == 0
    assert reset.error_shift == (0, 0)
    assert reset.variance_budget.energy_change == 0
    assert reset.dirichlet_budget.energy_change == 0


def test_fixed_pattern_readout_does_not_assert_the_old_held_dynamics():
    old, new = _reference(), _reference(capacity=(2, 4), weight=3)
    with pytest.raises(ValueError):
        observe_forced_support_state(old, new.source)
    pattern = _pattern(old, new.source.epi)
    assert pattern == _pattern(old, old.source.epi)
    assert pattern.error_variance == F(3, 8)
    assert pattern.error_dirichlet_energy == F(9, 8)


@pytest.mark.parametrize("shift", (F(-7, 3), F(11, 7)))
def test_fixed_pattern_is_invariant_under_a_uniform_epi_shift(shift):
    reference = _reference()
    initial = _pattern(reference, reference.source.epi)
    shifted = _pattern(reference, (1 + shift, shift))
    assert shifted.mean == initial.mean + shift
    assert shifted.relative_error == initial.relative_error
    assert shifted.error_variance == initial.error_variance


def test_public_reference_and_event_caches_are_rebuilt():
    old, new = _reference(), _reference(capacity=(1, 1))
    forged_old = replace(old, relative_profile=(F(999), F(999)), mean_drift=F(999))
    forged_new = replace(new, metric_weights=(F(999), F(999)))
    before = replace(old.source, epi_gradient=(F(999), F(999)))
    after = replace(new.source, dirichlet_energy=F(999))
    reset = observe_forced_support_reset(forged_old, forged_new, before, after)
    assert reset.before_reference == old
    assert reset.after_reference == new
    assert reset.variance_budget.energy_change == F(1, 64)
    assert reset.dirichlet_budget.energy_change == F(-11, 32)


@pytest.mark.parametrize(
    "field,value",
    (
        ("epi", (F(2), F(0))),
        ("nodes", (1, 0)),
    ),
)
def test_same_epi_and_node_order_are_required(field, value):
    old, new = _reference(), _reference(capacity=(1, 1))
    after = replace(new.source, **{field: value})
    with pytest.raises(ValueError):
        observe_forced_support_reset(old, new, old.source, after)


@pytest.mark.parametrize(
    "nodes,epi",
    (
        ((1, 0), (1, 0)),
        ((0, 1), (1,)),
        ({0, 1}, (1, 0)),
        ((0, 1), (float("nan"), 0)),
        ((0, 1), (True, 0)),
    ),
)
def test_invalid_pattern_coordinates_are_rejected(nodes, epi):
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_pattern(_reference(), nodes=nodes, epi=epi)


def test_reset_and_pattern_records_are_frozen():
    reference = _reference()
    reset = _reset(reference, reference)
    pattern = _pattern(reference, reference.source.epi)
    with pytest.raises(FrozenInstanceError):
        reset.mean_reweighting = F(1)
    with pytest.raises(FrozenInstanceError):
        pattern.mean = F(1)

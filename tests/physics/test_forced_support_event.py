"""Independent exact controls for EPI jumps and changing forced profiles."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.physics.forced_support import (
    ForcedSupportPattern,
    derive_forced_support_balance,
    observe_forced_support_event,
    observe_forced_support_pattern,
    observe_forced_support_reset,
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


def _event(old, new, *, x0=(F(1), F(0)), x1=(F(0), F(1))):
    return observe_forced_support_event(
        old,
        new,
        replace(old.source, epi=x0),
        replace(new.source, epi=x1),
    )


def test_pure_epi_jump_has_independent_signed_cross_and_quadratic_terms():
    reference = _reference()
    event = _event(reference, reference)
    assert event.epi_jump == (-1, 1)
    assert event.centered_epi_jump == (F(-2, 3), F(4, 3))
    assert event.mean_epi_jump == event.mean_change == F(-1, 3)
    assert event.mean_reweighting == 0
    assert event.before.relative_error == (F(1, 2), -1)
    assert event.after.relative_error == (F(-1, 6), F(1, 3))
    variance = event.variance_jump_budget
    assert variance.cross_term == -1
    assert variance.quadratic_term == F(2, 3)
    assert variance.energy_change == event.variance_change == F(-1, 3)
    energy = event.dirichlet_jump_budget
    assert energy.cross_term == -3
    assert energy.quadratic_term == 2
    assert energy.energy_change == event.dirichlet_change == -1
    assert event.variance_reset_budget.energy_change == 0
    assert event.dirichlet_reset_budget.energy_change == 0
    assert event.profile_shift == event.reference_error_shift == (0, 0)
    assert event.mean_identity_residual == 0
    assert event.error_identity_residual == (0, 0)
    assert variance.identity_residual == energy.identity_residual == 0
    assert event.variance_identity_residual == event.dirichlet_identity_residual == 0


def test_simultaneous_capacity_conductance_and_epi_changes_have_exact_budgets():
    event = _event(_reference(), _reference(capacity=(1, 1), weight=2))
    assert event.after_reference.metric_weights == (2, 2)
    assert event.after_reference.relative_profile == (F(-1, 8), F(1, 8))
    assert event.midpoint_pattern.error_variance == F(1, 24)
    assert event.after.error_variance == F(9, 32)
    assert event.after.error_dirichlet_energy == F(9, 16)
    assert event.mean_epi_jump == F(-1, 3)
    assert event.mean_reweighting == F(1, 6)
    assert event.mean_change == F(-1, 6)
    variance = event.variance_reset_budget
    assert variance.metric_term == F(7, 72)
    assert variance.reference_cross_term == F(7, 72)
    assert variance.reference_quadratic_term == F(13, 288)
    assert variance.energy_change == F(23, 96)
    energy = event.dirichlet_reset_budget
    assert energy.metric_term == F(1, 8)
    assert energy.reference_cross_term == F(1, 4)
    assert energy.reference_quadratic_term == F(1, 16)
    assert energy.energy_change == F(7, 16)
    assert event.variance_change == F(-3, 32)
    assert event.dirichlet_change == F(-9, 16)


def test_forcing_change_can_reverse_the_sign_of_the_full_event_budget():
    old = _reference()
    new = _reference(capacity=(1, 1), weight=2, forcing=(F(1, 2), F(-1, 2)))
    event = _event(old, new)
    assert event.after_reference.relative_profile == (F(1, 2), F(-1, 2))
    assert event.after_reference.mean_drift == 0
    assert event.after.relative_error == (-1, 1)
    assert event.variance_jump_budget.energy_change == F(-1, 3)
    assert event.variance_reset_budget.energy_change == F(47, 24)
    assert event.variance_change == F(13, 8)
    assert event.dirichlet_jump_budget.energy_change == -1
    assert event.dirichlet_reset_budget.energy_change == F(31, 8)
    assert event.dirichlet_change == F(23, 8)


def test_epi_coefficient_change_is_accounted_for_as_a_new_profile():
    old = _reference()
    new = derive_forced_support_balance(
        old.source,
        epi_weight=1,
        forcing=old.forcing,
    )
    event = _event(old, new)
    assert event.after_reference.relative_profile == (F(-1, 12), F(1, 6))
    assert event.after.relative_error == (F(-1, 4), F(1, 2))
    assert event.variance_change == F(-9, 32)
    assert event.dirichlet_change == F(-27, 32)


def test_current_profile_can_report_zero_error_while_original_pattern_worsens():
    old = _reference()
    new = _reference(capacity=(1, 1), forcing=(1, -1))
    event = _event(old, new, x1=(F(2), F(0)))
    fixed_after = observe_forced_support_pattern(old, nodes=(0, 1), epi=(2, 0))
    assert fixed_after == event.midpoint_pattern
    assert event.after_reference.relative_profile == (1, -1)
    assert event.after.error_variance == event.after.error_dirichlet_energy == 0
    assert event.before.error_variance == F(3, 8)
    assert fixed_after.error_variance == F(25, 24)
    assert event.variance_jump_budget.energy_change == F(2, 3)
    assert fixed_after.error_dirichlet_energy == F(25, 8)
    assert event.dirichlet_jump_budget.energy_change == 2
    assert event.variance_change == F(-3, 8)
    assert event.dirichlet_change == F(-9, 8)


def test_event_snapshots_override_earlier_reference_epi_for_all_terms():
    old, new = _reference(), _reference(capacity=(1, 1), weight=2)
    event = _event(old, new, x0=(F(3), F(-1)), x1=(F(2), F(0)))
    assert old.source.epi == new.source.epi == (1, 0)
    assert event.before.snapshot.epi == (3, -1)
    assert event.after.snapshot.epi == event.midpoint_pattern.epi == (2, 0)
    assert event.variance_jump_budget.energy_change == F(-7, 3)
    assert event.dirichlet_jump_budget.energy_change == -7
    assert event.variance_reset_budget.energy_change == F(143, 96)
    assert event.dirichlet_reset_budget.energy_change == F(31, 16)
    assert event.variance_change == F(-27, 32)
    assert event.dirichlet_change == F(-81, 16)
    assert event.mean_epi_jump == event.mean_reweighting == F(-1, 3)
    assert event.mean_change == F(-2, 3)


@pytest.mark.parametrize("shift", (F(-7, 3), F(11, 7)))
def test_uniform_epi_motion_changes_mean_without_profile_error(shift):
    reference = _reference()
    event = _event(reference, reference, x1=(1 + shift, shift))
    assert event.epi_jump == (shift, shift)
    assert event.centered_epi_jump == (0, 0)
    assert event.mean_epi_jump == event.mean_change == shift
    assert event.mean_reweighting == 0
    assert event.variance_change == event.dirichlet_change == 0
    for budget in (event.variance_jump_budget, event.dirichlet_jump_budget):
        assert budget.cross_term == budget.quadratic_term == 0


def test_pressure_only_write_is_retained_without_invented_epi_evolution():
    reference = _reference()
    after = replace(reference.source, stored_pressure=(F(7), F(-3)))
    event = observe_forced_support_event(
        reference,
        reference,
        reference.source,
        after,
    )
    assert event.before.pressure_defect != event.after.pressure_defect
    assert event.after.snapshot.stored_pressure == (7, -3)
    assert event.epi_jump == event.centered_epi_jump == (0, 0)
    assert event.mean_change == event.variance_change == event.dirichlet_change == 0
    assert type(event.midpoint_pattern) is ForcedSupportPattern
    assert not hasattr(event.midpoint_pattern, "stored_pressure")
    assert not hasattr(event.midpoint_pattern, "pressure_defect")
    assert not hasattr(event.midpoint_pattern, "snapshot")


def test_no_epi_jump_reuses_the_same_epi_observers_budgets():
    old, new = _reference(), _reference(capacity=(1, 1), weight=2)
    reset = observe_forced_support_reset(old, new, old.source, new.source)
    event = _event(old, new, x1=old.source.epi)
    assert event.variance_jump_budget.energy_change == 0
    assert event.dirichlet_jump_budget.energy_change == 0
    assert event.variance_reset_budget == reset.variance_budget
    assert event.dirichlet_reset_budget == reset.dirichlet_budget
    assert event.mean_reweighting == reset.mean_reweighting
    assert event.profile_shift == reset.profile_shift
    assert event.reference_error_shift == reset.error_shift
    with pytest.raises(ValueError, match="identical node order and EPI"):
        observe_forced_support_reset(
            old,
            new,
            old.source,
            replace(new.source, epi=(F(0), F(1))),
        )


def test_reversed_events_cancel_totals_but_not_individual_jump_terms():
    old, new = _reference(), _reference(capacity=(1, 1), weight=2)
    forward = _event(old, new)
    reverse = _event(new, old, x0=(F(0), F(1)), x1=(F(1), F(0)))
    for field in ("mean_change", "variance_change", "dirichlet_change"):
        assert getattr(forward, field) + getattr(reverse, field) == 0
    assert forward.variance_jump_budget.quadratic_term > 0
    assert reverse.variance_jump_budget.quadratic_term > 0


def test_three_reference_event_chain_telescopes_to_independent_endpoints():
    old = _reference()
    middle = _reference(capacity=(1, 1), weight=2)
    final = _reference(capacity=(1, 1), forcing=(1, -1))
    events = (
        _event(old, middle),
        _event(middle, old, x0=(F(0), F(1)), x1=(F(3), F(-1))),
        _event(old, final, x0=(F(3), F(-1)), x1=(F(2), F(0))),
    )
    assert sum(event.mean_change for event in events) == F(1, 3)
    assert sum(event.variance_change for event in events) == F(-3, 8)
    assert sum(event.dirichlet_change for event in events) == F(-9, 8)
    for left, right in zip(events, events[1:]):
        assert left.after == right.before
        assert left.after_reference == right.before_reference


def test_actual_support_addition_and_epi_jump_have_independent_three_node_energy():
    graph = nx.path_graph(3)
    for node, x in zip(graph, (1, 0, 0), strict=True):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: x,
                ALIAS_VF[0]: 1,
                ALIAS_DNFR[0]: 0,
            }
        )
    old = derive_forced_support_balance(
        observe_support_transport(graph),
        epi_weight=F(1, 2),
        forcing=(0, 0, 0),
    )
    graph.add_edge(0, 2)
    new = derive_forced_support_balance(
        observe_support_transport(graph),
        epi_weight=F(1, 2),
        forcing=(0, 0, 0),
    )
    event = _event(old, new, x0=(F(1), F(0), F(0)), x1=(F(0), F(1), F(0)))
    assert event.before.error_variance == F(3, 8)
    assert event.after.error_variance == F(2, 3)
    assert event.variance_jump_budget.energy_change == F(1, 8)
    assert event.variance_reset_budget.energy_change == F(1, 6)
    assert event.variance_change == F(7, 24)
    assert event.dirichlet_jump_budget.energy_change == F(1, 2)
    assert event.dirichlet_reset_budget.energy_change == 0
    assert event.dirichlet_change == F(1, 2)


def test_public_caches_are_rebuilt_and_inputs_stay_unchanged():
    old, new = _reference(), _reference(capacity=(1, 1), weight=2)
    forged_old = replace(old, relative_profile=(F(999), F(999)), mean_drift=F(999))
    forged_new = replace(new, metric_weights=(F(999), F(999)))
    before = replace(old.source, epi_gradient=(F(999), F(999)))
    after = replace(new.source, epi=(F(0), F(1)), dirichlet_energy=F(999))
    event = observe_forced_support_event(forged_old, forged_new, before, after)
    assert event.before_reference == old
    assert event.after_reference == new
    assert event.before.snapshot.epi_gradient == (-1, 1)
    assert event.after.snapshot.dirichlet_energy == 1
    assert event.variance_change == F(-3, 32)
    assert event.dirichlet_change == F(-9, 16)
    assert forged_old.mean_drift == after.dirichlet_energy == F(999)
    assert before.epi_gradient == (F(999), F(999))


@pytest.mark.parametrize(
    "observer", [observe_forced_support_event, observe_forced_support_reset]
)
def test_each_public_observation_rebuilds_each_reference_exactly_once(
    monkeypatch, observer
):
    from tnfr.physics import forced_support

    old, new = _reference(), _reference(capacity=(1, 1), weight=2)
    before, after = old.source, replace(new.source, epi=old.source.epi)
    expected = observer(old, new, before, after)
    forged_old = replace(old, relative_profile=(F(999), F(999)))
    forged_new = replace(new, metric_weights=(F(999), F(999)))
    inverse = forced_support.exact_matrix_inverse
    calls = []

    def counted_inverse(matrix):
        calls.append(matrix)
        return inverse(matrix)

    monkeypatch.setattr(forced_support, "exact_matrix_inverse", counted_inverse)
    # Reusing a public dataclass must not bypass reconstruction; repeating the
    # public call must not introduce a persistent cache of its derived fields.
    for invocation in (1, 2):
        assert observer(forged_old, forged_new, before, after) == expected
        assert len(calls) == 2 * invocation


def test_different_valid_reference_node_orders_are_rejected_before_comparison():
    old = _reference()
    new = derive_forced_support_balance(
        replace(old.source, nodes=(1, 0)),
        epi_weight=old.epi_weight,
        forcing=old.forcing,
    )
    with pytest.raises(ValueError, match="identical node count and order"):
        _event(old, new)


@pytest.mark.parametrize(
    "field,value",
    (
        ("epi", (F(0),)),
        ("epi", (float("nan"), 0)),
        ("epi", (True, 0)),
        ("nodes", (1, 0)),
        ("capacity", (1, 1)),
        ("conductance", ((0, 1, F(2)), (1, 0, F(2)))),
        ("support_neighbors", ((0, 1), (0,))),
    ),
)
def test_invalid_or_wrong_model_actual_endpoint_is_rejected(field, value):
    reference = _reference()
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_event(
            reference,
            reference,
            reference.source,
            replace(reference.source, **{field: value}),
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("epi_weight", 0),
        ("epi_weight", float("inf")),
        ("forcing", (F(1),)),
        ("forcing", (float("nan"), 0)),
    ),
)
def test_invalid_reference_inputs_are_revalidated(field, value):
    reference = _reference()
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_event(
            reference,
            replace(reference, **{field: value}),
            reference.source,
            reference.source,
        )


@pytest.mark.parametrize(
    "source",
    (
        {"capacity": (0, 1)},
        {"conductance": ()},
        {"conductance": ((0, 1, F(1)),)},
    ),
)
def test_invalid_reference_domain_cannot_be_hidden_in_a_cached_record(source):
    reference = _reference()
    forged = replace(reference, source=replace(reference.source, **source))
    with pytest.raises(ValueError):
        observe_forced_support_event(
            reference,
            forged,
            reference.source,
            forged.source,
        )


def test_event_and_jump_records_are_frozen():
    reference = _reference()
    event = _event(reference, reference)
    with pytest.raises(FrozenInstanceError):
        event.mean_change = F(1)
    with pytest.raises(FrozenInstanceError):
        event.variance_jump_budget.cross_term = F(1)


def test_new_public_api_is_exported_from_physics():
    from tnfr import physics
    from tnfr.physics import forced_support

    for name in (
        "ForcedSupportEvent",
        "ForcedSupportJumpEnergy",
        "observe_forced_support_event",
    ):
        assert getattr(physics, name) is getattr(forced_support, name)
        assert name in physics.__all__
        assert name in forced_support.__all__

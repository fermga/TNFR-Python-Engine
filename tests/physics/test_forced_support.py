"""Independent exact controls for held-support drift and relative profiles."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_state,
    observe_forced_support_step,
)
from tnfr.physics.support_transport import observe_support_transport


F = Fraction


def _snapshot(graph=None, *, epi=(1, 0), capacity=(1, 2), pressure=(-0.25, 1)):
    graph = nx.path_graph(2) if graph is None else graph
    for node, x, nu, p in zip(graph, epi, capacity, pressure, strict=True):
        graph.nodes[node].update({
            ALIAS_EPI[0]: x, ALIAS_VF[0]: nu, ALIAS_DNFR[0]: p,
        })
    return observe_support_transport(graph)


def _reference(snapshot=None, *, forcing=(F(1, 4), F(1, 2))):
    return derive_forced_support_balance(
        _snapshot() if snapshot is None else snapshot,
        epi_weight=F(1, 2), forcing=forcing,
    )


def test_heterogeneous_pair_has_nonzero_drift_and_exact_centered_profile():
    reference = _reference()
    assert reference.strengths == (1, 1)
    assert reference.metric_weights == (1, F(1, 2))
    assert reference.compatibility_residual == F(3, 4)
    assert reference.mean_drift == F(1, 2)
    assert reference.relative_profile == (F(-1, 6), F(1, 3))
    assert reference.profile_residual == (0, 0)
    assert reference.profile_center_residual == 0
    assert not reference.has_zero_pressure_equilibrium
    assert reference.max_convex_step == 1
    state = observe_forced_support_state(reference, reference.source)
    assert state.mean == F(2, 3)
    assert state.relative_error == (F(1, 2), -1)
    assert state.error_variance == F(3, 8)
    assert state.error_dirichlet_energy == F(9, 8)
    assert state.modeled_pressure == (F(-1, 4), 1)
    assert state.pressure_defect == (0, 0)


@pytest.mark.parametrize("mean", (F(-7, 3), F(0), F(11, 7)))
def test_compatible_profile_allows_every_initial_mean(mean):
    reference = _reference(forcing=(F(1, 4), F(-1, 4)))
    assert reference.has_zero_pressure_equilibrium
    assert reference.compatibility_residual == reference.mean_drift == 0
    assert reference.relative_profile == (F(1, 6), F(-1, 3))
    snapshot = replace(
        reference.source,
        epi=(mean + F(1, 6), mean - F(1, 3)), stored_pressure=(F(0), F(0)),
    )
    state = observe_forced_support_state(reference, snapshot)
    assert state.mean == mean
    assert state.relative_error == state.modeled_pressure == (0, 0)
    assert state.error_variance == state.error_dirichlet_energy == 0


def test_zero_forcing_profile_is_consensus_with_capacity_weighted_mean():
    reference = _reference(forcing=(0, 0))
    state = observe_forced_support_state(reference, reference.source)
    assert reference.relative_profile == (0, 0)
    assert reference.mean_drift == 0
    assert state.mean == F(2, 3)
    assert state.relative_error == (F(1, 3), F(-2, 3))


def test_drifting_profile_has_uniform_rate_and_constant_spatial_energy():
    reference = _reference()
    mean = F(11, 7)
    epi = tuple(mean + value for value in reference.relative_profile)
    before = replace(reference.source, epi=epi, stored_pressure=(F(1, 2), F(1, 4)))
    after = replace(before, epi=tuple(x + F(1, 8) for x in epi))
    step = observe_forced_support_step(reference, before, after, F(1, 4))
    assert step.before.modeled_pressure == (F(1, 2), F(1, 4))
    assert step.before.snapshot.rate == (F(1, 2), F(1, 2))
    assert step.before.relative_error == step.after.relative_error == (0, 0)
    assert step.before.mean == mean
    assert step.after.mean == mean + F(1, 8)
    assert step.support_budget.energy_change == 0
    assert step.support_budget.before.dirichlet_energy == F(1, 8)
    assert step.mean_identity_residual == 0


def test_refreshed_euler_matches_independent_pair_mode_and_energy_budget():
    reference = _reference()
    after = replace(reference.source, epi=(F(15, 16), F(1, 2)))
    step = observe_forced_support_step(reference, reference.source, after, F(1, 4))
    assert step.after.mean == F(19, 24)
    assert step.after.relative_error == (F(5, 16), F(-5, 8))
    assert step.after.error_variance == F(75, 512)
    assert step.after.error_dirichlet_energy == F(225, 512)
    assert step.mean_change == step.mean_model_change == F(1, 8)
    assert step.mean_pressure_defect == step.mean_step_defect == 0
    assert step.support_budget.state_defect == (0, 0)
    budget = step.relative_energy_budget
    assert budget.drift_term == F(-27, 32)
    assert budget.quadratic_term == F(81, 512)
    assert budget.defect_term == 0
    assert budget.energy_change == F(-351, 512)
    assert budget.identity_residual == step.mean_identity_residual == 0
    assert step.relative_recurrence_residual == (0, 0)
    assert step.convex_step_admissible


def test_mean_defects_can_cancel_while_centered_error_remains_nonzero():
    reference = _reference()
    before = replace(reference.source, stored_pressure=(F(-1, 8), F(15, 16)))
    after = replace(before, epi=(F(33, 32), F(5, 16)))
    step = observe_forced_support_step(reference, before, after, F(1, 4))
    assert step.before.pressure_defect == (F(1, 8), F(-1, 16))
    assert step.support_budget.state_defect == (F(1, 16), F(-5, 32))
    assert step.mean_pressure_defect == F(1, 96)
    assert step.mean_step_defect == F(-1, 96)
    assert step.mean_change == step.mean_model_change == F(1, 8)
    assert step.relative_energy_budget.state_defect == (F(3, 32), F(-3, 16))
    assert step.after.relative_error == (F(13, 32), F(-13, 16))
    assert step.relative_energy_budget.defect_term == F(621, 2048)
    assert step.relative_recurrence_residual == (0, 0)
    assert step.mean_identity_residual == 0


def test_declared_hard_clip_endpoint_keeps_nonzero_pressure_and_mean_defect():
    reference = _reference()
    before = replace(
        reference.source, epi=(F(3, 4), F(3, 4)),
        stored_pressure=(F(1, 4), F(1, 2)),
    )
    after = replace(before, epi=(F(1), F(1)))
    step = observe_forced_support_step(reference, before, after, 1)
    assert step.support_budget.expected_epi == (1, F(7, 4))
    assert step.support_budget.state_defect == (0, F(-3, 4))
    assert step.after.modeled_pressure == (F(1, 4), F(1, 2))
    assert step.mean_change == F(1, 4)
    assert step.mean_model_change == F(1, 2)
    assert step.mean_pressure_defect == 0
    assert step.mean_step_defect == F(-1, 4)
    assert step.mean_identity_residual == 0
    assert step.convex_step_admissible


def test_large_ideal_euler_step_can_increase_relative_energy():
    reference = _reference()
    after = replace(reference.source, epi=(F(5, 8), F(3)))
    step = observe_forced_support_step(reference, reference.source, after, F(3, 2))
    assert step.after.relative_error == (F(-5, 8), F(5, 4))
    assert step.after.error_dirichlet_energy == F(225, 128)
    assert step.relative_energy_budget.energy_change > 0
    assert step.relative_energy_budget.state_defect == (0, 0)
    assert not step.convex_step_admissible


def test_cached_reference_and_snapshot_fields_are_recomputed():
    reference = _reference()
    forged = replace(
        reference, mean_drift=F(999), relative_profile=(F(888), F(888)),
        metric_weights=(F(1), F(1)), max_convex_step=F(999),
    )
    before = replace(reference.source, epi_gradient=(F(999), F(999)),
                     rate=(F(999), F(999)), dirichlet_energy=F(999))
    after = replace(before, epi=(F(15, 16), F(1, 2)))
    result = observe_forced_support_step(forged, before, after, F(1, 4))
    assert result.reference == reference
    assert result.before.relative_error == (F(1, 2), -1)
    assert result.support_budget.state_defect == (0, 0)
    assert result.mean_identity_residual == 0


def test_observations_are_detached_and_frozen():
    graph = nx.path_graph(2)
    snapshot = _snapshot(graph)
    before = deepcopy(graph)
    forcing = [F(1, 4), F(1, 2)]
    reference = derive_forced_support_balance(snapshot, epi_weight=F(1, 2),
                                            forcing=forcing)
    forcing[0] = F(999)
    assert reference.forcing == (F(1, 4), F(1, 2))
    assert nx.utils.graphs_equal(graph, before)
    with pytest.raises(FrozenInstanceError):
        reference.mean_drift = F(0)


@pytest.mark.parametrize("epi_weight", (0, -1, float("nan"), True))
def test_invalid_epi_weight_is_rejected(epi_weight):
    with pytest.raises((TypeError, ValueError)):
        derive_forced_support_balance(_snapshot(), epi_weight=epi_weight,
                                      forcing=(0, 0))


@pytest.mark.parametrize("forcing", ((0,), (0, 0, 0), (0, float("inf")), (0, True)))
def test_invalid_forcing_is_rejected(forcing):
    with pytest.raises((TypeError, ValueError)):
        _reference(forcing=forcing)


def test_zero_capacity_and_zero_strength_are_outside_profile_scope():
    with pytest.raises(ValueError):
        _reference(_snapshot(capacity=(0, 2)))
    graph = nx.empty_graph(2)
    with pytest.raises(ValueError):
        _reference(_snapshot(graph))


def test_zero_weight_bridge_does_not_establish_transport_connectivity():
    graph = nx.path_graph(4)
    graph.edges[1, 2]["weight"] = 0
    snapshot = _snapshot(graph, epi=(1, 0, 1, 0), capacity=(1,) * 4,
                         pressure=(0,) * 4)
    assert nx.is_connected(graph)
    with pytest.raises(ValueError):
        _reference(snapshot, forcing=(0,) * 4)


@pytest.mark.parametrize("field,value", (
    ("nodes", (1, 0)),
    ("capacity", (F(1), F(3))),
    ("conductance", ((0, 1, F(2)), (1, 0, F(2)))),
    ("support_neighbors", ((0, 1), (0,))),
))
def test_changed_held_support_or_capacity_is_rejected(field, value):
    reference = _reference()
    changed = replace(reference.source, **{field: value})
    with pytest.raises(ValueError):
        observe_forced_support_state(reference, changed)


@pytest.mark.parametrize("dt", (-1, float("inf"), True))
def test_invalid_step_duration_is_rejected(dt):
    reference = _reference()
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_step(reference, reference.source, reference.source, dt)

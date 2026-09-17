"""Independent exact controls for target compatibility and its signed balance."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_target,
)
from tnfr.physics.support_transport import observe_support_transport


F = Fraction


def _reference(*, capacity=(1, 1), forcing=(0, 0), epi_weight=F(1, 2)):
    graph = nx.path_graph(len(capacity))
    for node, nu in zip(graph, capacity, strict=True):
        graph.nodes[node].update({
            ALIAS_EPI[0]: int(node == 0), ALIAS_VF[0]: nu,
            ALIAS_DNFR[0]: 0,
        })
    return derive_forced_support_balance(
        observe_support_transport(graph), epi_weight=epi_weight, forcing=forcing,
    )


def test_shape_compatibility_does_not_require_zero_pressure_or_zero_mean_drift():
    old = _reference()
    current = _reference(capacity=(1, 2), forcing=(1, F(1, 2)))
    result = observe_forced_support_target(old, current, current.source)
    assert result.target_rate == (1, 1)
    assert result.compatibility_residual == (0, 0)
    assert result.target_compatible
    assert result.limiting_pattern.relative_error == (0, 0)
    assert result.limiting_pattern.error_variance == 0
    assert result.reference.compatibility_residual == F(3, 2)
    assert result.reference.mean_drift == 1
    assert not result.reference.has_zero_pressure_equilibrium
    assert result.model_rate == (F(1, 2), 2)
    assert result.model_energy_rate == F(-3, 4)
    assert result.profile_identity_residual == (0, 0)


def test_compatible_three_node_model_can_increase_the_old_metric_energy():
    # This is a detached forced-model control, not default capacity forcing.
    old = _reference(capacity=(1, 2, 1), forcing=(0, 0, 0), epi_weight=1)
    current = _reference(
        capacity=(F(1, 8), F(1, 8), 4), forcing=(0, 0, 0), epi_weight=1,
    )
    snapshot = replace(
        current.source, epi=(F(1, 8), F(3, 4), F(5, 8)),
        stored_pressure=(F(5, 8), F(-3, 8), F(1, 8)),
    )
    result = observe_forced_support_target(old, current, snapshot)
    assert result.target_reference.metric_weights == (1, 1, 1)
    assert result.reference.metric_weights == (8, 16, F(1, 4))
    assert result.metric_proportionality is None
    assert result.target_compatible
    assert result.compatibility_residual == (0, 0, 0)
    assert result.pattern.mean == F(1, 2)
    assert result.pattern.relative_error == (F(-3, 8), F(1, 4), F(1, 8))
    assert result.pattern.error_variance == F(7, 64)
    assert result.model_rate == (F(5, 64), F(-3, 64), F(1, 2))
    assert result.homogeneous_energy_rate == F(11, 512)
    assert result.target_source_energy_rate == 0
    assert result.model_energy_rate == result.stored_nodal_energy_rate == F(11, 512)
    assert result.stored_pressure_energy_rate_defect == 0
    assert result.energy_rate_identity_residual == 0


def test_proportional_metrics_recover_the_exact_dirichlet_dissipation_balance():
    old = _reference(capacity=(1, 2), forcing=(F(1, 4), F(1, 2)))
    current = _reference(capacity=(2, 4), forcing=old.forcing)
    snapshot = replace(current.source, stored_pressure=(F(-1, 4), 1))
    result = observe_forced_support_target(old, current, snapshot)
    assert result.metric_proportionality == 2
    assert result.target_compatible
    assert result.target_rate == (1, 1)
    assert result.pattern.relative_error == (F(1, 2), -1)
    assert result.pattern.error_dirichlet_energy == F(9, 8)
    assert result.homogeneous_energy_rate == F(-9, 4)
    assert result.target_source_energy_rate == 0
    assert result.model_energy_rate == F(-9, 4)
    assert result.stored_pressure_energy_rate_defect == 0
    assert result.model_energy_rate == (
        -2 * current.epi_weight * result.metric_proportionality
        * result.pattern.error_dirichlet_energy
    )


def test_nonzero_target_residual_can_be_hidden_by_a_zero_instantaneous_energy_rate():
    old = _reference(forcing=(F(1, 2), F(-1, 2)))
    current = _reference(forcing=old.forcing, epi_weight=1)
    snapshot = replace(current.source, epi=(F(1, 2), F(-1, 2)))
    result = observe_forced_support_target(old, current, snapshot)
    assert result.pattern.relative_error == (0, 0)
    assert result.model_energy_rate == 0
    assert not result.target_compatible
    assert result.target_rate == result.compatibility_residual == (F(-1, 2), F(1, 2))
    assert result.compatibility_energy == F(1, 4)
    assert result.limiting_pattern.relative_error == (F(-1, 4), F(1, 4))
    assert result.limiting_pattern.error_variance == F(1, 16)
    assert result.profile_identity_residual == (0, 0)


def test_signed_source_term_and_stored_pressure_defect_are_separate():
    old = _reference()
    current = _reference(forcing=(F(1, 2), F(-1, 2)))
    snapshot = replace(current.source, stored_pressure=(2, -1))
    result = observe_forced_support_target(old, current, snapshot)
    assert result.model_rate == (0, 0)
    assert result.homogeneous_energy_rate == F(-1, 2)
    assert result.target_source_energy_rate == F(1, 2)
    assert result.model_energy_rate == 0
    assert result.state.pressure_defect == (2, -1)
    assert result.stored_pressure_energy_rate_defect == F(3, 2)
    assert result.stored_nodal_energy_rate == F(3, 2)
    assert result.energy_rate_identity_residual == 0


def test_actual_snapshot_overrides_old_and_current_reference_source_epi():
    reference = _reference()
    snapshot = replace(reference.source, epi=(3, -1), stored_pressure=(-2, 2))
    result = observe_forced_support_target(reference, reference, snapshot)
    assert result.target_reference.source.epi == result.reference.source.epi == (1, 0)
    assert result.state.snapshot.epi == result.pattern.epi == (3, -1)
    assert result.pattern.mean == 1
    assert result.pattern.relative_error == (2, -2)
    assert result.pattern.error_variance == 4
    assert result.model_rate == (-2, 2)
    assert result.model_energy_rate == -8
    assert result.limiting_pattern.epi == (0, 0)


def test_uniform_epi_translation_changes_only_the_fixed_pattern_mean():
    reference = _reference(capacity=(1, 2), forcing=(F(1, 4), F(1, 2)))
    base = observe_forced_support_target(reference, reference, reference.source)
    shifted = observe_forced_support_target(
        reference, reference, replace(reference.source, epi=(8, 7)),
    )
    assert shifted.pattern.mean - base.pattern.mean == 7
    for field in (
        "target_rate", "compatibility_residual", "compatibility_energy", "model_rate",
        "homogeneous_energy_rate", "target_source_energy_rate", "model_energy_rate",
        "stored_pressure_energy_rate_defect", "stored_nodal_energy_rate",
    ):
        assert getattr(shifted, field) == getattr(base, field)
    assert shifted.pattern.relative_error == base.pattern.relative_error


def test_nonzero_pressure_channels_can_cancel_exactly_in_target_rate():
    reference = _reference(forcing=(F(1, 2), F(-1, 2)))
    components = (("phase", (1, -1)), ("vf", (F(-1, 2), F(1, 2))))
    result = observe_forced_support_target(
        reference, reference, reference.source, forcing_components=components,
    )
    assert result.pressure_channels == (
        ("epi", (F(-1, 2), F(1, 2))), *components,
    )
    assert result.projected_rate_channels == result.pressure_channels
    assert result.channel_gram == (
        (F(1, 2), -1, F(1, 2)),
        (-1, 2, -1),
        (F(1, 2), -1, F(1, 2)),
    )
    assert result.target_compatible
    assert result.compatibility_energy == 0
    assert result.channel_energy_identity_residual == 0
    assert sum(result.channel_gram[i][i] for i in range(3)) == 3


def test_default_forcing_channel_keeps_heterogeneous_capacity_projection():
    old = _reference(capacity=(1, 2))
    current = _reference(capacity=(2, 1), forcing=(1, 0))
    result = observe_forced_support_target(old, current, current.source)
    assert result.pressure_channels == (("epi", (0, 0)), ("forcing", (1, 0)))
    assert result.projected_rate_channels == (
        ("epi", (0, 0)), ("forcing", (F(2, 3), F(-4, 3))),
    )
    assert result.target_rate == (2, 0)
    assert result.compatibility_residual == (F(2, 3), F(-4, 3))
    assert result.compatibility_energy == F(2, 3)
    assert result.channel_gram == ((0, 0), (0, F(4, 3)))
    assert result.profile_identity_residual == (0, 0)


@pytest.mark.parametrize("components", (
    (("phase", (1, 0)),),
    (("phase", (0,)),),
    (("phase", (0, 0, 0)),),
    (("phase", (0, 0)), ("phase", (0, 0))),
    (("epi", (0, 0)),),
))
def test_invalid_component_sums_dimensions_duplicates_or_reserved_name_fail(components):
    reference = _reference()
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_target(
            reference, reference, reference.source, forcing_components=components,
        )


@pytest.mark.parametrize("components", ({}, set(), "phase"))
def test_unordered_or_textual_component_containers_are_rejected(components):
    reference = _reference()
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_target(
            reference, reference, reference.source, forcing_components=components,
        )


def test_snapshot_must_match_current_reference_coefficients_and_node_order():
    reference = _reference()
    for snapshot in (
        replace(reference.source, capacity=(1, 2)),
        replace(reference.source, nodes=(1, 0)),
        replace(reference.source, conductance=((0, 1, F(2)), (1, 0, F(2)))),
    ):
        with pytest.raises(ValueError):
            observe_forced_support_target(reference, reference, snapshot)


def test_current_reference_and_snapshot_cannot_reorder_the_original_target():
    old = _reference()
    current = derive_forced_support_balance(
        replace(old.source, nodes=(1, 0)), epi_weight=old.epi_weight, forcing=old.forcing,
    )
    with pytest.raises(ValueError):
        observe_forced_support_target(old, current, current.source)


def test_public_cached_fields_are_rebuilt_for_both_references_and_snapshot():
    old = _reference()
    current = _reference(capacity=(1, 2), forcing=(F(1, 4), F(1, 2)))
    expected = observe_forced_support_target(old, current, current.source)
    bad_source = replace(
        current.source, epi_gradient=(99, 99), rate=(99, 99),
        dirichlet_energy=99, capacity_gradient=(99, 99),
    )
    actual = observe_forced_support_target(
        replace(old, relative_profile=(99, 99), metric_weights=(99, 99)),
        replace(
            current, source=bad_source, relative_profile=(99, 99),
            metric_weights=(99, 99), mean_drift=99,
        ),
        bad_source,
    )
    assert actual == expected


def test_target_observation_is_frozen_and_does_not_mutate_its_inputs():
    reference = _reference()
    snapshot = reference.source
    result = observe_forced_support_target(reference, reference, snapshot)
    with pytest.raises(FrozenInstanceError):
        result.target_compatible = False
    assert reference.source is snapshot
    assert snapshot.epi == (1, 0)
    assert snapshot.stored_pressure == (0, 0)

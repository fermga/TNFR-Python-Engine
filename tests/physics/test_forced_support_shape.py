"""Exact intrinsic shape rates of a declared held nodal diffusion model.

No trajectory is advanced. A modal orientation is a derived EPI coordinate,
not the independent canonical phase or a newly selected capacity law.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_shape,
)
from tnfr.physics.support_transport import observe_support_transport

F = Fraction


def _source(graph=None, *, epi=(1, F(1, 4), F(1, 2)), capacity=None):
    graph = nx.path_graph(3) if graph is None else graph
    capacity = (1,) * len(graph) if capacity is None else capacity
    for node, x, nu in zip(graph, epi, capacity, strict=True):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: x,
                ALIAS_VF[0]: nu,
                ALIAS_DNFR[0]: 0,
            }
        )
    return observe_support_transport(graph)


def _reference(source=None, *, epi_weight=1, forcing=None):
    source = _source() if source is None else source
    forcing = (0,) * len(source.nodes) if forcing is None else forcing
    return derive_forced_support_balance(
        source,
        epi_weight=epi_weight,
        forcing=forcing,
    )


def _observe(reference=None, snapshot=None):
    reference = _reference() if reference is None else reference
    snapshot = reference.source if snapshot is None else snapshot
    return observe_forced_support_shape(reference, snapshot)


def test_p3_mixed_modes_have_exact_shape_and_relaxation_rates():
    result = _observe()
    assert result.reference.metric_weights == (1, 2, 1)
    assert result.state.mean == F(1, 2)
    assert result.state.relative_error == (F(1, 2), F(-1, 4), 0)
    assert result.generator_image == (F(3, 4), F(-1, 2), F(1, 4))
    assert result.norm_squared == F(3, 8)
    assert result.state.error_dirichlet_energy == F(5, 16)
    assert result.relaxation_rate == F(5, 3)
    assert result.shape_tangent_scaled == (F(1, 12), F(1, 12), F(-1, 4))
    assert result.spectral_variance == F(2, 9)
    assert result.relaxation_rate_derivative == F(-4, 9)
    assert result.norm_squared_rate == F(-5, 4)
    assert result.stationary_shape is False
    assert result.radius_squared_balance_residual == 0
    assert result.rayleigh_balance_residual == 0
    assert result.tangent_orthogonality_residual == 0
    assert result.rate_identity_residual == 0


@pytest.mark.parametrize(
    "mode,eigenvalue",
    (
        ((1, 0, -1), F(1)),
        ((1, -1, 1), F(2)),
    ),
)
def test_eigenmode_shape_stays_fixed_while_amplitude_contracts(mode, eigenvalue):
    reference = _reference(_source(epi=mode))
    result = _observe(reference)
    assert result.relaxation_rate == eigenvalue
    assert result.generator_image == tuple(eigenvalue * value for value in mode)
    assert result.shape_tangent_scaled == (0, 0, 0)
    assert result.spectral_variance == result.relaxation_rate_derivative == 0
    assert result.norm_squared_rate == -2 * eigenvalue * result.norm_squared
    assert result.norm_squared_rate < 0
    assert result.stationary_shape is True


def test_two_mode_orientation_rate_follows_the_existing_generator():
    # u=(1,0,-1), v=(1,-1,1) have H-norms squared 2,4 and rates 1,2.
    # For y=a*u+b*v, r=b/a obeys r'=-r. This uses only supplied nodal
    # diffusion; the orthonormal angle obeys alpha'=-sin(alpha)cos(alpha).
    result = _observe()
    y, tangent = result.state.relative_error, result.shape_tangent_scaled
    a, b = (y[0] - y[2]) / 2, -y[1]
    tangent_a = (tangent[0] - tangent[2]) / 2
    tangent_b = -tangent[1]
    ratio_rate = (tangent_b * a - b * tangent_a) / a**2
    assert a == b == F(1, 4)
    assert ratio_rate == -b / a == -1
    assert result.relaxation_rate == (2 * a**2 + 8 * b**2) / (2 * a**2 + 4 * b**2)
    assert result.spectral_variance == 2 * a**2 * b**2 / (a**2 + 2 * b**2) ** 2
    assert result.relaxation_rate_derivative == (
        -2 * (result.relaxation_rate - 1) * (2 - result.relaxation_rate)
    )


@pytest.mark.parametrize("z", (F(1), F(1, 2), F(1, 7)))
def test_exact_two_mode_curve_agrees_with_derived_scalar_relaxation_law(z):
    # z=exp(-t), z'=-z: this analytic curve is never numerically integrated.
    # y=[z*(1,0,-1)+z^2*(1,-1,1)]/4 has exactly the existing mode rates 1,2.
    y = ((z + z**2) / 4, -(z**2) / 4, (-z + z**2) / 4)
    result = _observe(_reference(_source()), replace(_source(), epi=y))
    assert result.norm_squared == (z**2 + 2 * z**4) / 8
    assert result.relaxation_rate == (1 + 4 * z**2) / (1 + 2 * z**2)
    assert result.relaxation_rate_derivative == -4 * z**2 / (1 + 2 * z**2) ** 2
    assert result.relaxation_rate_derivative == (
        -2 * (result.relaxation_rate - 1) * (2 - result.relaxation_rate)
    )


@pytest.mark.parametrize(
    "shift,scale",
    (
        (F(7, 3), F(1)),
        (F(0), F(3, 2)),
        (F(-2), F(-3)),
    ),
)
def test_uniform_shift_and_nonzero_amplitude_scale_preserve_shape_rates(shift, scale):
    original = _observe()
    snapshot = replace(
        original.reference.source,
        epi=tuple(shift + scale * x for x in original.reference.source.epi),
    )
    changed = _observe(original.reference, snapshot)
    assert changed.norm_squared == scale**2 * original.norm_squared
    assert changed.norm_squared_rate == scale**2 * original.norm_squared_rate
    assert changed.generator_image == tuple(scale * x for x in original.generator_image)
    assert changed.shape_tangent_scaled == tuple(
        scale * x for x in original.shape_tangent_scaled
    )
    assert changed.relaxation_rate == original.relaxation_rate
    assert changed.spectral_variance == original.spectral_variance
    assert changed.relaxation_rate_derivative == original.relaxation_rate_derivative


def test_zero_relative_error_has_no_normalized_shape_or_relaxation_rate():
    reference = _reference()
    result = _observe(reference, replace(reference.source, epi=(F(3, 7),) * 3))
    assert result.norm_squared == result.norm_squared_rate == 0
    assert result.generator_image == (0, 0, 0)
    assert result.shape_tangent_scaled is None
    assert result.relaxation_rate is None
    assert result.spectral_variance is None
    assert result.relaxation_rate_derivative is None
    assert result.stationary_shape is None
    assert result.tangent_orthogonality_residual is None
    assert result.rate_identity_residual is None
    assert result.radius_squared_balance_residual == 0
    assert result.rayleigh_balance_residual == 0


def test_held_affine_profile_and_mean_drift_leave_a_relative_eigenmode():
    source = _source(nx.path_graph(2), epi=(1, 0), capacity=(1, 2))
    reference = _reference(source, epi_weight=F(1, 2), forcing=(F(1, 4), F(1, 2)))
    result = _observe(reference)
    assert reference.relative_profile == (F(-1, 6), F(1, 3))
    assert reference.mean_drift == F(1, 2)
    assert result.state.relative_error == (F(1, 2), -1)
    assert result.generator_image == (F(3, 4), F(-3, 2))
    assert result.norm_squared == F(3, 4)
    assert result.norm_squared_rate == F(-9, 4)
    assert result.relaxation_rate == F(3, 2)
    assert result.shape_tangent_scaled == (0, 0)
    assert result.spectral_variance == 0
    at_profile = replace(source, epi=tuple(2 + z for z in reference.relative_profile))
    equilibrium_shape = _observe(reference, at_profile)
    assert equilibrium_shape.relaxation_rate is None
    assert equilibrium_shape.norm_squared == 0
    assert equilibrium_shape.state.modeled_pressure == (F(1, 2), F(1, 4))


def test_stored_pressure_disagreement_remains_visible_without_changing_model_rates():
    result = _observe()
    assert result.state.snapshot.stored_pressure == (0, 0, 0)
    assert result.state.snapshot.rate == (0, 0, 0)
    assert result.state.pressure_defect == result.generator_image
    assert result.norm_squared_rate < 0
    refreshed = replace(
        result.reference.source,
        stored_pressure=(
            F(-3, 4),
            F(1, 2),
            F(-1, 4),
        ),
    )
    actual = _observe(result.reference, refreshed)
    assert actual.state.pressure_defect == (0, 0, 0)
    assert actual.norm_squared_rate == result.norm_squared_rate
    assert actual.relaxation_rate_derivative == result.relaxation_rate_derivative


def test_uniform_generator_scaling_changes_rate_without_changing_geometry():
    original = _observe()
    factor = F(7, 3)
    changed = _observe(_reference(epi_weight=factor))
    assert changed.state.relative_error == original.state.relative_error
    assert changed.relaxation_rate == factor * original.relaxation_rate
    assert changed.norm_squared_rate == factor * original.norm_squared_rate
    assert changed.spectral_variance == factor**2 * original.spectral_variance
    assert changed.relaxation_rate_derivative == (
        factor**2 * original.relaxation_rate_derivative
    )


def test_cached_reference_and_snapshot_arithmetic_is_rebuilt():
    original = _observe()
    forged_reference = replace(
        original.reference,
        metric_weights=(F(999),) * 3,
        relative_profile=(F(999),) * 3,
        mean_drift=F(999),
    )
    forged_snapshot = replace(
        original.reference.source,
        epi_gradient=(F(999),) * 3,
        rate=(F(999),) * 3,
        dirichlet_energy=F(999),
    )
    assert _observe(forged_reference, forged_snapshot) == original


@pytest.mark.parametrize("change", ("capacity", "conductance", "support"))
def test_changed_held_structure_is_rejected(change):
    reference = _reference()
    source = reference.source
    if change == "capacity":
        changed = replace(source, capacity=(F(2), F(1), F(1)))
    elif change == "conductance":
        changed = replace(
            source,
            conductance=tuple(
                (i, j, 2 * weight) for i, j, weight in source.conductance
            ),
        )
    else:
        changed = replace(source, support_neighbors=((0, 1), (0, 2), (1,)))
    with pytest.raises(ValueError, match="reference support and capacity"):
        _observe(reference, changed)


def test_shape_observation_is_detached_frozen_and_does_not_mutate_a_graph():
    graph = nx.path_graph(3)
    source = _source(graph)
    before = deepcopy(graph)
    result = _observe(_reference(source))
    assert nx.utils.graphs_equal(graph, before)
    with pytest.raises(FrozenInstanceError):
        result.norm_squared = F(999)
    graph.nodes[0][ALIAS_EPI[0]] = 99
    assert result.state.snapshot.epi == source.epi


def test_internal_shape_contraction_is_not_an_isolated_nodes_capacity_law():
    result = _observe()
    # A scalar aggregate with no neighbors has zero canonical EPI pressure.
    # Assigning the derived Rayleigh rate to its capacity cannot reconstruct
    # the nonzero fine internal-amplitude contraction through nu*pressure.
    aggregate = _source(
        nx.empty_graph(1),
        epi=(1,),
        capacity=(result.relaxation_rate,),
    )
    assert aggregate.epi_gradient == aggregate.rate == (0,)
    assert result.norm_squared_rate != 0

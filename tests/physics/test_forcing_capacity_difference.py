"""Detached capacity-pressure differences and their represented residuals."""

import math
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.physics import forcing_realization as realization
from tnfr.physics._cycle_algebra import dirichlet_energy, dot, laplacian_action
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    observe_forcing_capacity_difference,
)

F = Fraction


def _capture(
    capacity=(1.0, 2.0, 4.0),
    *,
    epi=(0.25, 0.5, 0.75),
    phase=(0.0, 0.0, 0.0),
    stored=(0.125, -0.25, 0.5),
    graph=None,
    weights=None,
):
    if graph is None:
        graph = nx.path_graph(3)
        graph.edges[0, 1]["weight"] = 1.0
        graph.edges[1, 2]["weight"] = 2.0
    else:
        graph = graph.copy()
    for node, nu, x, theta, pressure in zip(
        graph, capacity, epi, phase, stored, strict=True
    ):
        graph.nodes[node].update(EPI=x, nu_f=nu, theta=theta, delta_nfr=pressure)
    graph.graph["DNFR_WEIGHTS"] = weights or {
        "phase": 1.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 1.0,
    }
    return capture_non_epi_forcing(graph)


@pytest.fixture(scope="module")
def observations():
    before = _capture()
    after = _capture(
        (1.0, 3.0, 3.0),
        epi=(0.375, 0.625, 0.875),
        phase=(0.5, 0.5, 0.5),
        stored=(-0.25, 0.375, 0.0),
    )
    return before, after


def test_unequal_transport_weights_do_not_replace_the_unweighted_capacity_source(
    observations,
):
    before, after = observations
    result = observe_forcing_capacity_difference(before, after)
    assert result.epi_offset == F(1, 8)
    assert result.phase_offset == F(1, 2)
    assert result.capacity_change == (0, 1, -1)
    # The unique-neighbor difference of (0,1,-1) is (1,-3/2,2).
    assert result.capacity_pressure_change == (F(1, 4), F(-3, 8), F(1, 2))
    assert result.phase_realization_change == (0, 0, 0)
    assert result.modeled_pressure_before == (F(9, 16), F(-5, 48), F(-5, 16))
    assert result.modeled_pressure_after == (F(13, 16), F(-23, 48), F(3, 16))
    assert result.support_components == ((0, 1, 2),)
    assert result.component_capacity_offsets == (None,)
    assert result.identity_residual == result.stored_identity_residual == (0, 0, 0)


def test_fresh_and_stored_pressure_differences_retain_independent_exact_defects(
    observations,
):
    before, after = observations
    # These replaced primitive vectors are declared arithmetic controls,
    # not a claim that a production kernel generated these fresh pressures.
    before = replace(before, full_kernel_pressure=(F(5, 8), F(-5, 48), F(-7, 16)))
    after = replace(after, full_kernel_pressure=(F(13, 16), F(-17, 48), F(1, 4)))
    result = observe_forcing_capacity_difference(before, after)
    assert result.kernel_defect_change == (F(-1, 16), F(1, 8), F(3, 16))
    assert result.fresh_pressure_change == (F(3, 16), F(-1, 4), F(11, 16))
    assert result.stored_residual_change == (F(-9, 16), F(7, 8), F(-19, 16))
    assert result.stored_pressure_change == (F(-3, 8), F(5, 8), F(-1, 2))
    assert result.identity_residual == result.stored_identity_residual == (0, 0, 0)


@pytest.mark.parametrize("phase_container", (tuple, iter))
def test_uniform_raw_phase_offset_does_not_erase_a_supplied_realization_difference(
    observations,
    phase_container,
):
    before, after = observations
    # A raw phase shift alone cannot authenticate an exactly equivariant
    # trigonometric evaluation. Retain a declared represented discrepancy.
    phase_gradient = (F(1, 16), F(0), F(0))
    forcing = (after.forcing[0] + F(1, 64),) + after.forcing[1:]
    before = replace(before, phase_gradient=phase_container(before.phase_gradient))
    after = replace(
        after, phase_gradient=phase_container(phase_gradient), forcing=forcing
    )
    result = observe_forcing_capacity_difference(before, after)
    assert result.phase_offset == F(1, 2)
    assert result.phase_realization_change == (F(1, 64), 0, 0)
    assert result.modeled_pressure_after[0] == F(53, 64)
    assert result.identity_residual == result.stored_identity_residual == (0, 0, 0)


def test_a_uniform_capacity_increment_remains_in_the_support_kernel():
    before = _capture()
    after = _capture((3.0, 4.0, 6.0))
    result = observe_forcing_capacity_difference(before, after)
    assert result.capacity_change == (2, 2, 2)
    assert result.capacity_pressure_change == (0, 0, 0)
    assert result.component_capacity_offsets == (F(2),)
    assert result.modeled_pressure_before == result.modeled_pressure_after


def test_disconnected_support_and_isolate_have_independent_constant_offsets():
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from(((0, 1), (2, 3)))
    common = dict(graph=graph, epi=(0.25,) * 5, phase=(0.0,) * 5, stored=(0.0,) * 5)
    before = _capture((1.0, 2.0, 3.0, 4.0, 5.0), **common)
    after = _capture((2.0, 3.0, 5.0, 6.0, 8.0), **common)
    result = observe_forcing_capacity_difference(before, after)
    assert result.capacity_change == (1, 1, 2, 2, 3)
    assert result.capacity_pressure_change == (0,) * 5
    assert result.support_components == ((0, 1), (2, 3), (4,))
    assert result.component_capacity_offsets == (F(1), F(2), F(3))


def test_zero_conductance_edge_still_connects_the_capacity_support():
    graph = nx.path_graph(3)
    graph.edges[0, 1]["weight"] = 1.0
    graph.edges[1, 2]["weight"] = 0.0
    before = _capture(graph=graph)
    after = _capture((2.0, 3.0, 6.0), graph=graph)
    result = observe_forcing_capacity_difference(before, after)
    assert result.support_components == ((0, 1, 2),)
    assert result.capacity_pressure_change == (0, F(1, 8), F(-1, 4))
    assert result.component_capacity_offsets == (None,)


def test_disabled_capacity_channel_does_not_imply_constant_capacity_differences():
    weights = {"phase": 1.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
    before = _capture(weights=weights)
    after = _capture((1.0, 3.0, 3.0), weights=weights)
    result = observe_forcing_capacity_difference(before, after)
    assert result.capacity_change == (0, 1, -1)
    assert result.capacity_pressure_change == (0, 0, 0)
    assert result.component_capacity_offsets == (None,)


@pytest.mark.parametrize("closing_weight", (F(3, 4), F(1), F(5, 4)))
def test_capacity_smoothing_can_create_nonzero_signed_source_compatibility(
    closing_weight,
):
    """A declared convex map is neither writer admission nor a prepared lock."""
    graph = nx.cycle_graph(5)
    graph.edges[0, 4]["weight"] = float(closing_weight)
    width, mu = F(1, 4096), F(1, 2)
    capacity = (F(1), F(1), 1 + width, F(1), F(1))
    # Exact full-support half-averaging on one center bump. These are
    # independently supplied detached states; no policy gate is executed.
    following = (F(1), 1 + width / 4, 1 + width / 2, 1 + width / 4, F(1))
    common = dict(
        graph=graph,
        epi=(F(1, 8),) * 5,
        phase=(0.0,) * 5,
        stored=(0.0,) * 5,
        weights={"phase": 0.5, "epi": 0.25, "vf": 0.25, "topo": 0.0},
    )
    before, after = _capture(capacity, **common), _capture(following, **common)
    result = observe_forcing_capacity_difference(before, after)
    f = dict(before.normalized_weights)["vf"]
    assert f == F(1, 4)
    assert result.before.capacity == capacity and result.after.capacity == following
    assert result.epi_offset == result.phase_offset == 0
    assert not any(result.phase_realization_change)
    assert max(following) - min(following) == width / 2
    assert max(capacity) - min(capacity) == width
    assert dirichlet_energy(capacity) == width**2
    assert dirichlet_energy(following) == width**2 / 8

    # Endpoint strengths differ from the unique-support degree two. The
    # capacity source is zero-sum only in the latter, unweighted measure.
    strengths = tuple(
        sum(weight for i, _, weight in result.before.conductance if i == node)
        for node in range(5)
    )
    assert strengths == (1 + closing_weight, F(2), F(2), F(2), 1 + closing_weight)
    b_before = dot(strengths, result.modeled_pressure_before)
    b_after = dot(strengths, result.modeled_pressure_after)
    assert b_before == 0
    assert b_after == mu * f * (closing_weight - 1) * width / 2
    assert b_after - b_before == dot(strengths, result.capacity_pressure_change)
    assert result.identity_residual == result.stored_identity_residual == (0,) * 5
    if closing_weight == F(3, 4):
        assert b_after == -mu * f * width / 8 < 0
        assert abs(b_after) > abs(b_before)
    elif closing_weight == 1:
        assert b_after == 0
    else:
        assert b_after > 0


def test_masked_capacity_compatibility_change_uses_a_signed_geometry_covector():
    """The compatibility cross term differs from capacity-energy dissipation."""
    graph = nx.cycle_graph(5)
    graph.edges[0, 4]["weight"] = 0.75
    capacity = (F(1), F(9, 8), F(17, 16), F(1), F(5, 4))
    eligible, mu = {0, 2, 4}, F(1, 2)
    support_action = laplacian_action(capacity)
    masked_action = tuple(
        value if i in eligible else F(0) for i, value in enumerate(support_action)
    )
    following = tuple(
        value - mu * action
        for value, action in zip(capacity, masked_action, strict=True)
    )
    common = dict(
        graph=graph,
        epi=(0.125,) * 5,
        phase=(0.0,) * 5,
        stored=(0.0,) * 5,
        weights={"phase": 0.5, "epi": 0.25, "vf": 0.25, "topo": 0.0},
    )
    before, after = _capture(capacity, **common), _capture(following, **common)
    result = observe_forcing_capacity_difference(before, after)
    strengths = tuple(
        sum(weight for i, _, weight in result.before.conductance if i == node)
        for node in range(5)
    )
    assert strengths == (F(7, 4), F(2), F(2), F(2), F(7, 4))
    capacity_weight = dict(before.normalized_weights)["vf"]
    compatibility_change = dot(
        strengths,
        tuple(
            right - left
            for left, right in zip(
                result.modeled_pressure_before,
                result.modeled_pressure_after,
                strict=True,
            )
        ),
    )
    assert compatibility_change == mu * capacity_weight * dot(
        laplacian_action(strengths), masked_action
    )
    assert compatibility_change == dot(strengths, result.capacity_pressure_change)
    assert compatibility_change != 0
    assert result.epi_offset == result.phase_offset == 0
    assert not any(result.phase_realization_change)
    # The averaging mask and blend are mathematical inputs, not an observed
    # outcome of freshness, Si eligibility, counters or physical timing.
    assert result.identity_residual == result.stored_identity_residual == (0,) * 5


def test_snapshot_and_pressure_caches_are_rebuilt(observations):
    before, after = observations
    expected = observe_forcing_capacity_difference(before, after)

    def forge(value):
        bogus = (F(999),) * 3
        return replace(
            value,
            snapshot=replace(
                value.snapshot,
                epi_gradient=bogus,
                capacity_gradient=bogus,
                topology_gradient=bogus,
                dirichlet_gradient=bogus,
                rate=bogus,
                dirichlet_energy=F(999),
                energy_rate=F(999),
            ),
            kernel_pressure_defect=bogus,
            stored_pressure_residual=bogus,
        )

    assert observe_forcing_capacity_difference(forge(before), forge(after)) == expected


def test_observer_does_not_read_graphs_or_rerun_phase_kernels(
    observations, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("detached differences cannot recapture the graph")

    monkeypatch.setattr(realization, "capture_non_epi_forcing", forbidden)
    monkeypatch.setattr(realization, "observe_support_transport", forbidden)
    monkeypatch.setattr(
        realization.fused_dnfr, "compute_fused_gradients_symmetric", forbidden
    )
    result = observe_forcing_capacity_difference(*observations)
    assert result.capacity_pressure_change == (F(1, 4), F(-3, 8), F(1, 2))


@pytest.mark.parametrize("field", ("epi", "phase"))
def test_offsets_must_be_exactly_uniform_not_merely_close(observations, field):
    before, after = observations
    if field == "epi":
        altered = replace(
            after,
            snapshot=replace(
                after.snapshot,
                epi=(F(3, 8), F(math.nextafter(0.625, math.inf)), F(7, 8)),
            ),
        )
    else:
        altered = replace(
            after, phase=(F(1, 2), F(math.nextafter(0.5, math.inf)), F(1, 2))
        )
    with pytest.raises(ValueError):
        observe_forcing_capacity_difference(before, altered)


@pytest.mark.parametrize("field", ("nodes", "conductance", "weights"))
def test_comparison_rejects_changed_node_space_transport_or_coefficients(
    observations, field
):
    before, after = observations
    if field == "nodes":
        altered = replace(after, snapshot=replace(after.snapshot, nodes=(2, 1, 0)))
    elif field == "conductance":
        graph = nx.path_graph(3)
        altered = _capture(graph=graph)
    else:
        altered = _capture(weights={"phase": 2.0, "epi": 1.0, "vf": 1.0, "topo": 0.0})
    with pytest.raises(ValueError):
        observe_forcing_capacity_difference(before, altered)


def test_nonreciprocal_zero_weight_support_is_outside_the_component_theorem():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(((0, 1, 1), (1, 0, 1), (1, 2, 1), (2, 1, 1)))
    graph.add_edge(0, 2, weight=0.0)
    observation = _capture(graph=graph)
    with pytest.raises(ValueError):
        observe_forcing_capacity_difference(observation, observation)


@pytest.mark.parametrize("side", ("before", "after"))
def test_zero_capacity_does_not_supply_the_positive_capacity_stationary_gate(side):
    valid = _capture()
    zero = _capture((0.0, 2.0, 4.0))
    pair = (zero, valid) if side == "before" else (valid, zero)
    with pytest.raises(ValueError):
        observe_forcing_capacity_difference(*pair)


def test_empty_node_space_is_not_an_admitted_stationary_comparison():
    empty = _capture((), epi=(), phase=(), stored=(), graph=nx.Graph())
    with pytest.raises(ValueError):
        observe_forcing_capacity_difference(empty, empty)


@pytest.mark.parametrize("field", ("phase", "full_kernel_pressure"))
def test_complete_scalar_vectors_are_required(observations, field):
    before, after = observations
    with pytest.raises(ValueError):
        observe_forcing_capacity_difference(before, replace(after, **{field: (F(0),)}))


def test_comparison_requires_typed_detached_observations(observations):
    with pytest.raises(TypeError):
        observe_forcing_capacity_difference({}, observations[1])

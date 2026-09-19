"""Exact inherited counts and capacities from supplied transport snapshots."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.physics.epi_memory import observe_forced_support_closure
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.quotient_structure import observe_quotient_structure
from tnfr.physics.support_transport import _from_data


def _source(size, edges, *, capacity=None, zero_edges=()):
    directed, support = [], [set() for _ in range(size)]
    for i, j, weight in edges:
        directed.append((i, j, weight))
        if i != j:
            directed.append((j, i, weight))
        support[i].add(j)
        support[j].add(i)
    for i, j in zero_edges:
        support[i].add(j)
        support[j].add(i)
    return _from_data(
        tuple(range(size)),
        directed,
        tuple(tuple(sorted(row)) for row in support),
        tuple(F(i, 8) for i in range(size)),
        (F(1),) * size if capacity is None else capacity,
        (F(0),) * size,
    )


def _k23():
    return _source(
        5,
        [(i, j, 1) for i in range(2) for j in range(2, 5)],
        capacity=(2, 2, 3, 3, 3),
    )


def _triangle(*, capacity=(2, 3, 3)):
    return _source(3, ((0, 1, 1), (0, 2, 1), (1, 2, 1)), capacity=capacity)


def test_k23_inherits_asymmetric_counts_and_each_capacity_topology_channel():
    source = _k23()
    result = observe_quotient_structure(source, ((0, 1), (2, 3, 4)))
    assert result.node_blocks == (0, 0, 1, 1, 1)
    assert result.multiplicity == ((0, 3), (2, 0))
    assert result.block_degree == (3, 2)
    assert result.block_capacity == result.effective_capacity == (2, 3)
    assert result.aggregated_conductance == result.macro_conductance == ((0, 6), (6, 0))
    assert result.fine_strengths == (3, 3, 2, 2, 2)
    assert result.metric_weights == (F(3, 2),) * 2 + (F(2, 3),) * 3
    assert result.macro_metric_weights == (3, 2)
    assert result.macro_strengths == (6, 6)
    assert result.source_scale == (1, 1)
    assert result.capacity_gradient == (1, -1)
    assert result.topology_gradient == (-1, 1)
    assert (
        tuple(result.capacity_gradient[a] for a in result.node_blocks)
        == source.capacity_gradient
    )
    assert (
        tuple(result.topology_gradient[a] for a in result.node_blocks)
        == source.topology_gradient
    )
    # A two-node simple quotient has degree (1,1), hence misses this source.
    assert result.topology_gradient != (0, 0)


def test_internal_neighbors_require_retained_capacity_and_pressure_rescaling():
    result = observe_quotient_structure(_triangle(), ((0,), (1, 2)))
    assert result.multiplicity == ((0, 2), (1, 1))
    assert result.block_degree == (2, 2)
    assert result.aggregated_conductance == ((0, 2), (2, 2))
    assert result.macro_conductance == ((0, 2), (2, 0))
    assert result.macro_metric_weights == (1, F(4, 3))
    assert result.block_capacity == (2, 3)
    assert result.effective_capacity == (2, F(3, 2))
    assert result.source_scale == (1, 2)
    assert result.capacity_gradient == (1, F(-1, 2))
    assert result.topology_gradient == (0, 0)
    # Substituting effective capacity into the bare macro gradient reverses
    # this example's capacity-source signs. The observed fine channel is fixed.
    bare_gradient = (F(-1, 2), F(1, 2))
    assert result.capacity_gradient != bare_gradient
    supplied_pressure = (F(1, 7), F(-2, 5))
    for nu, effective, scale, pressure in zip(
        result.block_capacity,
        result.effective_capacity,
        result.source_scale,
        supplied_pressure,
        strict=True,
    ):
        assert effective * (scale * pressure) == nu * pressure


def test_unique_zero_weight_support_is_retained_separately_from_transport():
    edges = ((0, 1, 1), (1, 2, 1))
    source = _source(3, edges, zero_edges=((0, 2),))
    result = observe_quotient_structure(source, ((0,), (1, 2)))
    assert result.multiplicity == ((0, 2), (1, 1))
    assert result.block_degree == (2, 2)
    assert result.aggregated_conductance == ((0, 1), (1, 2))
    assert result.fine_strengths == (1, 2, 1)
    assert result.effective_capacity == (1, F(1, 3))
    assert result.source_scale == (1, 3)
    assert (0, 2) not in {(i, j) for i, j, _ in result.source.conductance}
    with pytest.raises(ValueError, match="equitable support"):
        observe_quotient_structure(_source(3, edges), ((0,), (1, 2)))


def test_exact_metadata_does_not_claim_weighted_epi_closure():
    source = _source(3, ((0, 1, 1), (0, 2, 2), (1, 2, 1)))
    blocks = ((0,), (1, 2))
    result = observe_quotient_structure(source, blocks)
    assert result.multiplicity == ((0, 2), (1, 1))
    reference = derive_forced_support_balance(source, epi_weight=1, forcing=(0, 0, 0))
    closure = observe_forced_support_closure(reference, blocks)
    assert closure.all_state_affine_closed is False
    assert result.macro_metric_weights == closure.macro_metric_weights
    assert result.macro_conductance == ((0, 3), (3, 0))
    assert "No weighted EPI closure" in result.scope


def test_self_neighbors_count_once_and_internal_edges_count_both_directions():
    source = _source(
        3,
        ((0, 1, 1), (0, 2, 1), (1, 2, 1), (0, 0, 2), (1, 1, 2), (2, 2, 2)),
    )
    result = observe_quotient_structure(source, ((0,), (1, 2)))
    assert result.multiplicity == ((1, 2), (1, 2))
    assert result.block_degree == (3, 3)
    assert result.aggregated_conductance == ((2, 2), (2, 6))
    assert result.macro_metric_weights == (4, 8)
    assert result.effective_capacity == (F(1, 2), F(1, 4))
    assert result.source_scale == (2, 4)
    assert result.capacity_gradient == result.topology_gradient == (0, 0)


def test_ordered_partition_is_materialized_once_without_mutation():
    source = _k23()
    original = deepcopy(source)
    blocks = [[4, 3, 2], [1, 0]]
    saved_blocks = deepcopy(blocks)
    result = observe_quotient_structure(source, (iter(row) for row in blocks))
    assert source == original and blocks == saved_blocks
    assert result.source is not source
    assert result.blocks == ((4, 3, 2), (1, 0))
    assert result.node_blocks == (1, 1, 0, 0, 0)
    assert result.multiplicity == ((0, 2), (3, 0))
    assert result.block_capacity == (3, 2)
    assert result.topology_gradient == (1, -1)
    blocks[0].append(99)
    assert result.blocks == ((4, 3, 2), (1, 0))
    with pytest.raises(FrozenInstanceError):
        result.source_scale = (1, 1)


@pytest.mark.parametrize(
    "field",
    (
        "epi_gradient",
        "capacity_gradient",
        "topology_gradient",
        "dirichlet_gradient",
        "rate",
        "dirichlet_energy",
        "energy_rate",
    ),
)
def test_forged_snapshot_caches_are_rejected(field):
    source = _triangle()
    value = getattr(source, field)
    forged = (value[0] + 1, *value[1:]) if isinstance(value, tuple) else value + 1
    with pytest.raises(ValueError, match="differs from rebuilt"):
        observe_quotient_structure(replace(source, **{field: forged}), ((0,), (1, 2)))


@pytest.mark.parametrize(
    "field", ("capacity", "stored_pressure", "rate", "energy_rate")
)
def test_boolean_numeric_payloads_are_not_accepted_as_exact_zero_or_one(field):
    source = _triangle()
    value = getattr(source, field)
    forged = (True, *value[1:]) if isinstance(value, tuple) else False
    with pytest.raises(TypeError, match="boolean"):
        observe_quotient_structure(replace(source, **{field: forged}), ((0,), (1, 2)))


def test_tiny_block_capacity_difference_cannot_be_promoted_by_tolerance():
    source = _triangle(capacity=(2, 3, F(3) + F(1, 2**80)))
    with pytest.raises(ValueError, match="exactly block-constant"):
        observe_quotient_structure(source, ((0,), (1, 2)))


def test_nonreciprocal_zero_support_and_disconnected_positive_transport_are_rejected():
    source = _source(3, ((0, 1, 1), (1, 2, 1)))
    asymmetric = _from_data(
        source.nodes,
        source.conductance,
        ((1, 2), (0, 2), (1,)),
        source.epi,
        source.capacity,
        source.stored_pressure,
    )
    with pytest.raises(ValueError, match="reciprocal unique support"):
        observe_quotient_structure(asymmetric, ((0,), (1, 2)))
    # A zero-weight support bridge does not connect the positive transport.
    disconnected = _source(4, ((0, 1, 1), (2, 3, 1)), zero_edges=((1, 2),))
    with pytest.raises(ValueError, match="connected positive transport"):
        observe_quotient_structure(disconnected, ((0, 1), (2, 3)))


@pytest.mark.parametrize(
    "blocks",
    (
        ((0, 1, 2),),
        ((0,), (1,), (2,)),
        ((0,), (1,)),
        ((0,), (1, 1, 2)),
        ((0,), (1, 9)),
        ((), (0, 1, 2)),
    ),
)
def test_invalid_or_nonreducing_partitions_are_rejected(blocks):
    with pytest.raises(ValueError):
        observe_quotient_structure(_triangle(), blocks)


def test_nonpositive_capacity_or_strength_is_outside_the_inherited_metric_domain():
    with pytest.raises(ValueError, match="positive strengths and capacities"):
        observe_quotient_structure(_triangle(capacity=(0, 3, 3)), ((0,), (1, 2)))
    isolated = _source(3, ((0, 1, 1),))
    with pytest.raises(ValueError, match="positive strengths and capacities"):
        observe_quotient_structure(isolated, ((0,), (1, 2)))

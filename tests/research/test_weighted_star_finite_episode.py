"""Static law admission and independent algebra, without a campaign replay."""

from copy import deepcopy
from fractions import Fraction as Q
from math import pi, sin

import pytest

from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._rational_interval import I
from tnfr.research.weighted_star_finite_episode import (
    MULTIPLICITIES,
    _admit_protocol,
    _metric,
    admit_preparation,
    frozen_protocol,
    prepare_fields,
    prepare_source,
    registration_bounds,
    storage_bounds,
)


@pytest.fixture(scope="module")
def admitted():
    preparation = prepare_source()
    return preparation, admit_preparation(preparation)


def test_preparation_and_full_tangent_metric(admitted):
    _, (source, partition, state) = admitted
    metric = _metric(source, partition)
    assert exact_symmetric_semidefinite(metric, strict=True)
    assert state.source_membership_certified and state.full_row_equality_certified
    assert storage_bounds(
        state.collective_form + state.collective_phase_radians,
        source,
        partition,
        tangent=True,
    ).hi < Q(1, 2)
    # Independently assemble pi*J from full fine rows and then restrict the
    # exact block injection. Cross-block counts include bare zero gradients.
    derivative = [[Q(0) for _ in range(10)] for _ in range(10)]
    for block, nodes in enumerate(partition.block_indices):
        i = nodes[0]
        mobility = source.capacity[i] / source.degrees[i]
        for edge, weight in zip(source.edges, source.conductances, strict=True):
            if i not in edge:
                continue
            j = edge[1] if edge[0] == i else edge[0]
            target = partition.node_blocks[j]
            derivative[block][5 + target] += mobility
            derivative[block][5 + block] -= mobility
            derivative[5 + block][block] += mobility * weight
            derivative[5 + block][target] -= mobility * weight
    product = [
        [sum(metric[i][k] * derivative[k][j] for k in range(10)) for j in range(10)]
        for i in range(10)
    ]
    assert all(product[i][j] + product[j][i] == 0 for i in range(10) for j in range(10))


@pytest.mark.parametrize("branch", ("nonlinear", "tangent"))
def test_block_fields_match_independent_full_rows(admitted, branch):
    _, (source, partition, _) = admitted
    interval_field, field = prepare_fields(source, partition, branch)
    coordinates = tuple(Q(i - 4, 17) for i in range(10))
    lifted = tuple(
        coordinates[channel * 5 + partition.node_blocks[i]]
        for channel in range(2)
        for i in range(10)
    )
    actual = field(7.0, tuple(map(float, coordinates)))
    intervals = interval_field(tuple(I(value) for value in coordinates))
    for i in range(10):
        gradient, current = 0.0, 0.0
        for (left, right), weight in zip(
            source.edges, source.conductances, strict=True
        ):
            if i not in (left, right):
                continue
            j = right if left == i else left
            gap = float(lifted[10 + j] - lifted[10 + i])
            current += sin(gap) if branch == "nonlinear" else gap
            gradient += float(weight * (lifted[i] - lifted[j]))
        mobility = float(source.capacity[i] / source.degrees[i])
        for channel, expected in enumerate(
            (mobility * current / pi, mobility * gradient / pi)
        ):
            k = channel * 5 + partition.node_blocks[i]
            assert actual[k] == pytest.approx(expected, abs=2e-16)
            assert float(intervals[k].midpoint) == pytest.approx(expected, abs=2e-16)


def test_registration_is_full_arithmetic_variance():
    values = tuple(Q(i * i - 3, 11) for i in range(10))
    expected = Q(0)
    for channel in range(2):
        fine = tuple(
            value
            for value, count in zip(
                values[channel * 5 : channel * 5 + 5], MULTIPLICITIES
            )
            for _ in range(count)
        )
        mean = sum(fine) / 10
        expected += sum((value - mean) ** 2 for value in fine)
    assert registration_bounds(values).contains(expected)
    shifted = tuple(value + (100 if i < 5 else -20) for i, value in enumerate(values))
    assert registration_bounds(shifted).contains(expected)


@pytest.mark.parametrize(
    "key,value",
    (
        ("horizon", True),
        ("certificate_order", True),
        ("certificate_bins", 16.0),
        ("horizon", Q(9)),
    ),
)
def test_policy_rejects_unadmitted_or_retuned_scalars(key, value):
    protocol = frozen_protocol()
    protocol[key] = value
    with pytest.raises(ValueError):
        _admit_protocol(protocol)


def test_primitive_signature_cannot_replace_graph(admitted):
    preparation, _ = admitted
    changed = deepcopy(preparation)
    changed["primitive"]["form"] = (Q(0),) * 10
    with pytest.raises(ValueError, match="primitive source"):
        admit_preparation(changed)


def test_observation_rejects_missing_coordinates_and_booleans():
    with pytest.raises(ValueError):
        registration_bounds((Q(0),) * 8)
    with pytest.raises(TypeError):
        registration_bounds((True,) + (Q(0),) * 9)

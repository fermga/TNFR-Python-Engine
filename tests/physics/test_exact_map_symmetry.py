"""Exact support, declared-map and source symmetries are separate claims.

Small detached rational fixtures only; no historical inputs or nodal execution.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tnfr.physics.equivariance import observe_exact_map_symmetries
from tnfr.physics.support_transport import _from_data

NODES = ("left", "child-a", "child-b", "right")
REGION = NODES[1:3]
IDENTITY = tuple(tuple(F(i == j) for j in range(4)) for i in range(4))
LAPLACIAN = (
    (1, -1, 0, 0),
    (F(-1, 2), 1, F(-1, 2), 0),
    (0, F(-1, 2), 1, F(-1, 2)),
    (0, 0, -1, 1),
)
REFLECTION = (3, 2, 1, 0)


def _snapshot(*, capacity=(1, 1, 1, 1), zero_edge=False, last_weight=F(1)):
    edges = ((0, 1, F(1)), (1, 2, F(1)), (2, 3, last_weight))
    support = [set() for _ in NODES]
    conductance = []
    for i, j, weight in edges:
        support[i].add(j)
        support[j].add(i)
        conductance.extend(((i, j, weight), (j, i, weight)))
    if zero_edge:
        support[0].add(2)
        support[2].add(0)
    return _from_data(
        NODES,
        conductance,
        tuple(tuple(sorted(s)) for s in support),
        (0,) * 4,
        capacity,
        (0,) * 4,
    )


def _observe(snapshot=None, **kwargs):
    options = {
        "metric_weights": (1, 2, 2, 1),
        "region": REGION,
        "operators": {"A": LAPLACIAN},
    }
    options.update(kwargs)
    return observe_exact_map_symmetries(
        _snapshot() if snapshot is None else snapshot, **options
    )


def _member(result, permutation=REFLECTION):
    return next(p for p in result.permutations if p.destinations == permutation)


def _product(a, b):
    return tuple(
        tuple(sum((a[i][k] * b[k][j] for k in range(4)), F(0)) for j in range(4))
        for i in range(4)
    )


def _sequential_reception():
    """Independent four-row half-mix example, composed in the stated order."""
    result = IDENTITY
    for i, neighbors in enumerate(((1,), (0, 2), (1, 3), (2,))):
        local = [list(row) for row in IDENTITY]
        local[i] = [F(0)] * 4
        local[i][i] = F(1, 2)
        for j in neighbors:
            local[i][j] = F(1, 2 * len(neighbors))
        result = _product(local, result)
    return result


def test_complete_path_reflection_and_fixed_environmental_space():
    result = _observe()
    assert result.support_group_order == 2
    assert result.admissible_indices == result.common_group_indices == (0, 1)
    assert result.common_operator_group == ((0, 1, 2, 3), REFLECTION)
    assert result.common_operator_orbits == ((0, 3), (1, 2))
    assert result.fixed_input_labels == ("region_constant", "outside_orbit_0")
    assert result.fixed_input_basis == ((0, 1, 1, 0), (1, 0, 0, 1))
    check = dict(_member(result).operator_checks)["A"]
    assert check.preserved and check.max_abs_defect == 0 and check.witness is None
    for vector in result.fixed_input_basis:
        assert tuple(vector[i] for i in REFLECTION) == vector


def test_sequential_reset_breaks_symmetry_preserved_by_nodal_generator():
    reset = _sequential_reception()
    dt = F(1, 4)
    transition = tuple(
        tuple(reset[i][j] - dt * LAPLACIAN[i][j] for j in range(4)) for i in range(4)
    )
    result = _observe(operators={"A": LAPLACIAN, "S": reset, "T": transition})
    assert len(dict(result.operator_group_indices)["A"]) == 2
    assert len(dict(result.operator_group_indices)["S"]) == 1
    assert len(dict(result.operator_group_indices)["T"]) == 1
    assert result.common_operator_group == ((0, 1, 2, 3),)
    assert result.common_operator_orbits == ((0,), (1,), (2,), (3,))
    assert result.fixed_input_basis == ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 1))
    defects = dict(_member(result).operator_checks)
    assert defects["S"] == defects["T"]
    i, j, difference = defects["S"].witness
    assert difference == reset[REFLECTION[i]][REFLECTION[j]] - reset[i][j] != 0
    assert defects["S"].max_abs_defect == max(
        abs(reset[REFLECTION[i]][REFLECTION[j]] - reset[i][j])
        for i in range(4)
        for j in range(4)
    )


@pytest.mark.parametrize(
    "capacity,metric,region,flags",
    [
        ((1, 1, 1, 2), (1, 2, 2, 1), REGION, (True, True, False)),
        ((1, 1, 1, 1), (1, 2, 3, 1), REGION, (True, False, True)),
        ((1, 1, 1, 1), (1, 2, 2, 1), NODES[:2], (False, True, True)),
    ],
)
def test_region_metric_capacity_filters_are_separate_from_support(
    capacity, metric, region, flags
):
    result = _observe(
        _snapshot(capacity=capacity), metric_weights=metric, region=region
    )
    assert result.support_group_order == 2
    reflection = _member(result)
    assert (
        reflection.region_preserved,
        reflection.metric_preserved,
        reflection.capacity_preserved,
    ) == flags
    assert dict(reflection.operator_checks)["A"].preserved
    assert len(result.admissible_indices) == len(result.common_group_indices) == 1


def test_zero_weight_support_edge_is_retained_in_automorphism_enumeration():
    original = _snapshot()
    extra = _snapshot(zero_edge=True)
    assert original.conductance == extra.conductance
    assert _observe(original).support_group_order == 2
    assert _observe(extra).support_group_order == 1


def test_exact_tiny_edge_weight_difference_breaks_support_symmetry():
    tiny = F(1, 2**1100)
    assert float(1 + tiny) == 1.0
    result = _observe(_snapshot(last_weight=1 + tiny), operators={"I": IDENTITY})
    assert result.support_group_order == 1


def test_exact_tiny_matrix_defect_is_not_rounded_to_zero():
    tiny = F(1, 2**1100)
    operator = [list(row) for row in IDENTITY]
    operator[0][0] += tiny
    result = _observe(operators={"declared": operator})
    defect = dict(_member(result).operator_checks)["declared"]
    assert not defect.preserved and defect.max_abs_defect == tiny
    assert defect.witness == (0, 0, -tiny)


def test_affine_source_and_particular_state_do_not_restrict_paired_matrix_group():
    result = _observe(
        fields={
            "c": (0, 0, 0, 0),
            "b": (1, 0, 0, 0),
            "EPI": (1, 2, 4, 8),
            "phase": (0, 1, 2, 3),
        }
    )
    assert len(result.common_operator_group) == 2
    counts = {name: len(indices) for name, indices in result.field_group_indices}
    assert counts == {"c": 2, "b": 1, "EPI": 1, "phase": 1}
    assert dict(_member(result).field_checks)["b"].witness == (0, F(-1))
    # A same-source paired difference can cancel b; this observer only reports
    # the supplied matrix and field conditions, never an actual cancellation.
    assert result.fields[1] == ("b", (1, 0, 0, 0))


def test_region_order_is_preserved_without_changing_its_set_stabilizer():
    result = _observe(region=REGION[::-1])
    assert result.region_indices == (2, 1)
    assert result.common_operator_group == ((0, 1, 2, 3), REFLECTION)


def test_cap_is_applied_to_complete_support_group_before_other_filters():
    with pytest.raises(ValueError, match="exceeds cap"):
        _observe(region=NODES[:1], cap=1)


@pytest.mark.parametrize("cap", [0, -1, True, 1.0, "2"])
def test_invalid_enumeration_caps_are_rejected(cap):
    with pytest.raises(ValueError, match="cap"):
        _observe(cap=cap)


def test_nonsymmetric_zero_conductance_support_is_rejected():
    snap = _snapshot()
    one_way = ((1, 2), (0, 2), (1, 3), (2,))
    snap = _from_data(
        snap.nodes,
        snap.conductance,
        one_way,
        snap.epi,
        snap.capacity,
        snap.stored_pressure,
    )
    with pytest.raises(ValueError, match="undirected support"):
        _observe(snap)


def test_tampered_derived_snapshot_is_rejected():
    snap = replace(_snapshot(), epi_gradient=(F(1), F(0), F(0), F(0)))
    with pytest.raises(ValueError, match="derived fields"):
        _observe(snap)


def test_nonsymmetric_conductance_snapshot_is_rejected():
    snap = _snapshot()
    asymmetric = tuple(edge for edge in snap.conductance if edge[:2] != (1, 0))
    with pytest.raises(ValueError, match="symmetric conductance"):
        _observe(replace(snap, conductance=asymmetric))


def test_zero_capacity_is_not_promoted_to_positive_mobility_symmetry():
    with pytest.raises(ValueError, match="positive capacity"):
        _observe(_snapshot(capacity=(0, 1, 1, 1)))


@pytest.mark.parametrize("region", [(), NODES, ("absent",), (NODES[0], NODES[0])])
def test_invalid_region_domains_are_rejected(region):
    with pytest.raises(ValueError, match="region"):
        _observe(region=region)


@pytest.mark.parametrize("region", ["left", set(REGION), {"left": 1}])
def test_unordered_or_scalar_region_is_rejected(region):
    with pytest.raises(TypeError, match="region"):
        _observe(region=region)


@pytest.mark.parametrize("metric", [(1, 2, 1), (1, 2, 0, 1), (1, 2, -1, 1)])
def test_metric_requires_positive_complete_coordinates(metric):
    with pytest.raises(ValueError, match="metric"):
        _observe(metric_weights=metric)


@pytest.mark.parametrize(
    "operators",
    [
        {},
        [],
        {"": IDENTITY},
        {1: IDENTITY},
        {"A": ((1, 0), (0, 1))},
        {"A": (IDENTITY[0],) * 3},
    ],
)
def test_operator_schema_and_dimensions_are_rejected(operators):
    with pytest.raises(ValueError):
        _observe(operators=operators)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_or_boolean_matrix_coefficients_are_rejected(value):
    operator = [list(row) for row in IDENTITY]
    operator[0][0] = value
    with pytest.raises((TypeError, ValueError)):
        _observe(operators={"A": operator})


@pytest.mark.parametrize(
    "fields", [[], {"": (0,) * 4}, {"x": (0,) * 3}, {"x": (0, float("nan"), 0, 0)}]
)
def test_invalid_field_schema_is_rejected(fields):
    with pytest.raises((TypeError, ValueError)):
        _observe(fields=fields)


def test_detached_results_do_not_alias_caller_owned_input_or_mutate_snapshot():
    snap = _snapshot()
    operator = [list(row) for row in IDENTITY]
    values = [0, 0, 0, 0]
    result = _observe(snap, operators={"I": operator}, fields={"x": values})
    operator[0][0] = 99
    values[0] = 99
    assert result.operators == (("I", IDENTITY),)
    assert result.fields == (("x", (0, 0, 0, 0)),)
    assert snap == _snapshot()
    with pytest.raises(FrozenInstanceError):
        result.support_group_order = 99


def test_only_detached_transport_snapshot_is_accepted():
    with pytest.raises(TypeError, match="SupportTransportSnapshot"):
        _observe(snapshot={"nodes": NODES})

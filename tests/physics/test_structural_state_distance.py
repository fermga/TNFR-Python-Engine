"""Exact structural-state distance on one graph-isomorphism class."""

import math

import networkx as nx
import pytest

from tnfr.physics import (
    StructuralChannelScales,
    circular_phase_distance,
    fixed_topology_structural_state_distance,
)


SCALES = StructuralChannelScales(
    epi=2.0,
    frequency=0.5,
    phase=math.pi,
    pressure=0.25,
    epi_rate=0.125,
    edge_conductance=1.0,
    edge_length=1.0,
)


def _populate(graph, offset=0.0):
    for index, node in enumerate(graph):
        frequency = 0.5 + 0.25 * index
        pressure = -0.3 + 0.2 * index
        graph.nodes[node].update(
            EPI=offset + 0.7 * index,
            nu_f=frequency,
            theta=0.2 + 0.9 * index,
            delta_nfr=pressure,
            dEPI_dt=frequency * pressure,
        )
    return graph


def _distance(left, right, **kwargs):
    return fixed_topology_structural_state_distance(
        left, right, scales=SCALES, **kwargs
    )


def test_identity_has_zero_nonnegative_components_and_exact_scope():
    graph = _populate(nx.path_graph(3))

    result = _distance(graph, graph.copy())

    assert result.distance == pytest.approx(0.0, abs=1e-15)
    assert result.admissible_isomorphism_count == 2
    assert result.node_count == 3
    assert result.exact_metric_on_isomorphism_classes
    assert "EXACT" in result.scope
    assert "cross-topology" in result.scope
    assert result.left_nodal_equation_residual_linf == pytest.approx(0.0)
    assert result.right_nodal_equation_residual_linf == pytest.approx(0.0)
    assert all(
        component >= 0.0
        for component in (
            result.epi_component,
            result.frequency_component,
            result.phase_component,
            result.pressure_component,
            result.epi_rate_component,
            result.edge_conductance_component,
            result.edge_length_component,
        )
    )


def test_zero_distance_is_invariant_under_relabeling_and_alias_choice():
    left = _populate(nx.path_graph(3))
    relabeling = {0: "gamma", 1: "alpha", 2: "beta"}
    right = nx.relabel_nodes(left, relabeling, copy=True)
    for node in right:
        data = right.nodes[node]
        data["νf"] = data.pop("nu_f")
        data["phase"] = data.pop("theta")
        data["ΔNFR"] = data.pop("delta_nfr")
        data["dEPI"] = data.pop("dEPI_dt")

    result = _distance(left, right)

    assert result.distance == pytest.approx(0.0, abs=1e-15)
    assert dict(result.minimizing_mapping) == relabeling


def test_graph_symmetry_can_identify_endpoint_swapped_states():
    left = _populate(nx.path_graph(3))
    right = left.copy()
    for key in ("EPI", "nu_f", "theta", "delta_nfr", "dEPI_dt"):
        right.nodes[0][key], right.nodes[2][key] = (
            right.nodes[2][key],
            right.nodes[0][key],
        )

    result = _distance(left, right)

    assert result.distance == pytest.approx(0.0, abs=1e-15)
    assert dict(result.minimizing_mapping) == {0: 2, 1: 1, 2: 0}


def test_equal_distance_breakdown_uses_insertion_invariant_lexicographic_rule():
    left = nx.empty_graph(2)
    right = nx.empty_graph(2)
    for graph in (left, right):
        for node in graph:
            graph.nodes[node].update(
                theta=0.0,
                delta_nfr=0.0,
                dEPI_dt=0.0,
            )
    left.nodes[0].update(EPI=0.0, nu_f=0.5)
    left.nodes[1].update(EPI=1.0, nu_f=0.75)
    right.nodes[0].update(EPI=0.0, nu_f=0.75)
    right.nodes[1].update(EPI=1.0, nu_f=0.5)

    reordered = nx.Graph()
    reordered.add_nodes_from(
        (node, dict(right.nodes[node])) for node in reversed(tuple(right))
    )

    first = _distance(left, right)
    second = _distance(left, reordered)

    assert first.minimizer_count == second.minimizer_count == 2
    assert first.admissible_isomorphism_count == 2
    assert first.epi_component == second.epi_component == 0.0
    assert first.frequency_component == second.frequency_component
    assert first.frequency_component == pytest.approx(math.sqrt(0.5))
    assert first.minimizing_mapping_unique
    assert second.minimizing_mapping_unique
    assert dict(first.minimizing_mapping) == {0: 0, 1: 1}
    assert dict(second.minimizing_mapping) == {0: 0, 1: 1}


def test_unresolved_mapping_tie_is_reported_and_not_arbitrarily_selected():
    left = nx.empty_graph(2)
    right = nx.empty_graph(2)
    for graph in (left, right):
        for node in graph:
            graph.nodes[node].update(
                EPI=0.0,
                nu_f=0.5,
                theta=0.0,
                delta_nfr=0.0,
                dEPI_dt=0.0,
            )

    result = _distance(left, right)

    assert result.minimizer_count == 2
    assert not result.minimizing_mapping_unique
    assert result.minimizing_mapping == ()
    assert "factorial worst case" in result.scope


def test_phase_uses_the_exact_circular_geodesic():
    left = _populate(nx.empty_graph(1))
    right = left.copy()
    left.nodes[0]["theta"] = 0.1
    right.nodes[0]["theta"] = 2.0 * math.pi - 0.1

    result = _distance(left, right)

    assert result.distance == pytest.approx(0.2 / math.pi)
    assert result.phase_component == pytest.approx(0.2 / math.pi)
    assert result.epi_component == 0.0
    right.nodes[0]["theta"] = left.nodes[0]["theta"] + 2.0 * math.pi
    assert _distance(left, right).distance == pytest.approx(0.0, abs=1e-15)


def test_symmetry_and_triangle_inequality_hold_on_finite_fixture():
    first = _populate(nx.path_graph(4), offset=0.0)
    second = _populate(nx.path_graph(4), offset=0.4)
    third = _populate(nx.path_graph(4), offset=-0.2)
    second.nodes[0]["theta"] = 2.0 * math.pi - 0.1
    third.nodes[3]["delta_nfr"] += 0.15
    third.nodes[3]["dEPI_dt"] = (
        third.nodes[3]["nu_f"] * third.nodes[3]["delta_nfr"]
    )
    first.edges[0, 1]["weight"] = 0.5
    second.edges[0, 1]["weight"] = 1.25
    third.edges[0, 1]["weight"] = 1.75

    distance_ab = _distance(first, second).distance
    distance_ba = _distance(second, first).distance
    distance_bc = _distance(second, third).distance
    distance_ac = _distance(first, third).distance

    assert distance_ab == pytest.approx(distance_ba)
    assert distance_ac <= distance_ab + distance_bc + 1e-12


def test_positive_distance_separates_non_equivalent_state_orbits():
    left = _populate(nx.path_graph(3))
    right = left.copy()
    right.nodes[1]["EPI"] += 0.5

    result = _distance(left, right)

    assert result.distance > 0.0
    assert result.epi_component == pytest.approx(0.25)


def test_declared_edge_labels_are_part_of_admissible_isomorphisms():
    left = _populate(nx.path_graph(3))
    right = left.copy()
    nx.set_edge_attributes(left, 1.0, "weight")
    nx.set_edge_attributes(right, 1.0, "weight")
    nx.set_edge_attributes(left, 1.0, "length")
    nx.set_edge_attributes(right, 1.0, "length")
    right.edges[0, 1]["weight"] = 2.0

    result = _distance(left, right)
    assert result.distance == pytest.approx(1.0)
    assert result.edge_conductance_component == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not isomorphic"):
        _distance(left, right, edge_label_attributes=("weight",))


def test_default_unit_conductance_and_weights_are_relabeling_invariant():
    left = _populate(nx.path_graph(3))
    left.edges[0, 1]["weight"] = 0.25
    right = nx.relabel_nodes(left, {0: "c", 1: "a", 2: "b"}, copy=True)

    result = _distance(left, right)

    assert result.distance == pytest.approx(0.0, abs=1e-15)
    assert result.edge_conductance_component == pytest.approx(0.0, abs=1e-15)
    assert result.edge_length_component == pytest.approx(0.0, abs=1e-15)


def test_structural_length_is_independent_from_transport_conductance():
    left = _populate(nx.path_graph(2))
    right = left.copy()
    left.edges[0, 1].update(weight=3.0, length=1.0)
    right.edges[0, 1].update(weight=3.0, length=2.0)

    result = _distance(left, right)

    assert result.edge_conductance_component == 0.0
    assert result.edge_length_component == pytest.approx(1.0)
    assert result.distance == pytest.approx(1.0)


def test_persistent_node_labels_can_expose_an_isomorphic_support_change():
    left = _populate(nx.path_graph(2))
    right = left.copy()
    for node in left:
        left.nodes[node]["persistent_id"] = node
        right.nodes[node]["persistent_id"] = node
    for key in ("EPI", "nu_f", "theta", "delta_nfr", "dEPI_dt"):
        right.nodes[0][key], right.nodes[1][key] = (
            right.nodes[1][key],
            right.nodes[0][key],
        )

    assert _distance(left, right).distance == pytest.approx(0.0)
    labelled = _distance(
        left, right, node_label_attributes=("persistent_id",)
    )
    assert labelled.distance > 0.0
    assert dict(labelled.minimizing_mapping) == {0: 0, 1: 1}


def test_nonisomorphic_and_mixed_direction_supports_are_rejected():
    path = _populate(nx.path_graph(4))
    cycle = _populate(nx.cycle_graph(4))
    directed = _populate(nx.DiGraph(path))

    with pytest.raises(ValueError, match="not isomorphic"):
        _distance(path, cycle)
    with pytest.raises(ValueError, match="directed and undirected"):
        _distance(path, directed)


def test_multigraph_scope_boundary_is_explicit():
    left = _populate(nx.MultiGraph(nx.path_graph(2)))
    right = left.copy()
    with pytest.raises(ValueError, match="multigraph"):
        _distance(left, right)


@pytest.mark.parametrize(
    "attribute,value,error",
    [
        ("EPI", float("nan"), "finite real"),
        ("theta", float("inf"), "finite real"),
        ("delta_nfr", "0.1", "finite real"),
        ("nu_f", -0.1, "nonnegative"),
    ],
)
def test_invalid_structural_state_is_rejected(attribute, value, error):
    left = _populate(nx.path_graph(2))
    right = left.copy()
    right.nodes[0][attribute] = value
    with pytest.raises(ValueError, match=error):
        _distance(left, right)


@pytest.mark.parametrize(
    "value,error",
    [
        (float("nan"), "finite real"),
        (float("inf"), "finite real"),
        (-0.1, "nonnegative"),
        (True, "finite real"),
        ("1.0", "finite real"),
    ],
)
def test_effective_edge_conductance_must_be_finite_and_nonnegative(value, error):
    left = _populate(nx.path_graph(2))
    right = left.copy()
    right.edges[0, 1]["weight"] = value

    with pytest.raises(ValueError, match=error):
        _distance(left, right)


def test_missing_structural_channel_is_not_silently_defaulted():
    left = _populate(nx.path_graph(2))
    right = left.copy()
    del right.nodes[0]["dEPI_dt"]
    with pytest.raises(ValueError, match="missing required EPI rate"):
        _distance(left, right)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epi", 0.0),
        ("frequency", -1.0),
        ("phase", float("nan")),
        ("pressure", float("inf")),
        ("epi_rate", True),
        ("edge_conductance", 0.0),
        ("edge_length", 0.0),
    ],
)
def test_every_channel_scale_must_be_explicitly_positive(field, value):
    values = dict(
        epi=1.0,
        frequency=1.0,
        phase=math.pi,
        pressure=1.0,
        epi_rate=1.0,
        edge_conductance=1.0,
        edge_length=1.0,
    )
    values[field] = value
    with pytest.raises(ValueError, match=f"{field} scale"):
        StructuralChannelScales(**values)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"node_label_attributes": "kind"}, "sequence"),
        ({"edge_label_attributes": ("weight", "weight")}, "duplicate"),
        ({"node_label_attributes": ("",)}, "non-empty"),
    ],
)
def test_label_contract_is_validated(kwargs, error):
    graph = _populate(nx.path_graph(2))
    with pytest.raises(ValueError, match=error):
        _distance(graph, graph.copy(), **kwargs)


def test_nodal_equation_residual_is_reported_without_changing_state():
    left = _populate(nx.path_graph(2))
    right = left.copy()
    right.nodes[1]["dEPI_dt"] += 0.75

    result = _distance(left, right)

    assert result.left_nodal_equation_residual_linf == pytest.approx(0.0)
    assert result.right_nodal_equation_residual_linf == pytest.approx(0.75)
    assert right.nodes[1]["dEPI_dt"] == pytest.approx(
        right.nodes[1]["nu_f"] * right.nodes[1]["delta_nfr"] + 0.75
    )


def test_repeated_evaluation_is_reproducible():
    left = _populate(nx.cycle_graph(4))
    right = left.copy()
    right.nodes[0]["theta"] += 0.4

    first = _distance(left, right)
    second = _distance(left, right)

    assert first == second


def test_empty_fixed_topology_has_the_unique_zero_distance():
    result = _distance(nx.Graph(), nx.Graph())
    assert result.distance == 0.0
    assert result.node_count == 0
    assert result.admissible_isomorphism_count == 1
    assert result.minimizer_count == 1
    assert result.minimizing_mapping_unique
    assert result.minimizing_mapping == ()


def test_extreme_finite_phase_representatives_do_not_overflow():
    distance = circular_phase_distance(1.0e308, -1.0e308)

    assert math.isfinite(distance)
    assert 0.0 <= distance <= math.pi
    assert distance == pytest.approx(
        circular_phase_distance(-1.0e308, 1.0e308)
    )


def test_unrepresentable_scaled_distance_reports_range_not_topology_failure():
    left = _populate(nx.empty_graph(1))
    right = left.copy()
    left.nodes[0]["persistent_id"] = 0
    right.nodes[0]["persistent_id"] = 0
    left.nodes[0]["EPI"] = 1.0e308
    right.nodes[0]["EPI"] = -1.0e308
    unit_epi_scale = StructuralChannelScales(
        epi=1.0,
        frequency=SCALES.frequency,
        phase=SCALES.phase,
        pressure=SCALES.pressure,
        epi_rate=SCALES.epi_rate,
        edge_conductance=SCALES.edge_conductance,
    )

    with pytest.raises(ValueError, match="EPI scaled difference exceeds"):
        fixed_topology_structural_state_distance(
            left,
            right,
            scales=unit_epi_scale,
            node_label_attributes=("persistent_id",),
        )


def test_nodal_equation_product_overflow_is_explicit():
    left = _populate(nx.empty_graph(1))
    right = left.copy()
    left.nodes[0].update(nu_f=1.0e308, delta_nfr=1.0e308)
    right.nodes[0].update(nu_f=1.0e308, delta_nfr=1.0e308)

    with pytest.raises(ValueError, match="nodal-equation product exceeds"):
        _distance(left, right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0"])
def test_circular_phase_distance_rejects_nonfinite_or_nonnumeric_values(value):
    with pytest.raises(ValueError, match="finite real"):
        circular_phase_distance(value, 0.0)


def test_invalid_explicit_structural_length_is_rejected():
    left = _populate(nx.path_graph(2))
    right = left.copy()
    right.edges[0, 1]["length"] = -1.0

    with pytest.raises(ValueError, match="structural length"):
        _distance(left, right)

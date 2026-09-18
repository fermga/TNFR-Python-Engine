r"""Tests for the R1 symmetry-sector observability (diffusion-sector base case).

Certifies, across the standard graph family (cycle, complete, star, path,
torus), that the canonical operator ``L_rw`` is equivariant under ``Aut(G)`` and
preserves the Reynolds sectors, that ``rank Q_Γ = #orbits``, and that the
observables are relabel-invariant and weight/direction aware.
"""

from __future__ import annotations

from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.equivariance import (
    sector_preservation_residual,
    verify_diffusion_equivariance,
)
from tnfr.physics.structural_diffusion import structural_diffusion_operator
from tnfr.physics.symmetry_sectors import (
    automorphism_orbits,
    automorphism_permutations,
    decompose_state,
    is_orbit_constant,
    orbit_count,
    permutation_matrix,
    reynolds_projector,
)

# (name, graph, expected vertex-orbit count)
GRAPHS = [
    ("cycle_C8", nx.cycle_graph(8), 1),
    ("complete_K6", nx.complete_graph(6), 1),
    ("star_K1_5", nx.star_graph(5), 2),
    ("path_P6", nx.path_graph(6), 3),
    ("torus_C3xC3", nx.cartesian_product(nx.cycle_graph(3), nx.cycle_graph(3)), 1),
]


@pytest.mark.parametrize("name,G,expected", GRAPHS)
def test_orbit_count(name, G, expected):
    assert orbit_count(G) == expected


@pytest.mark.parametrize("name,G,expected", GRAPHS)
def test_reynolds_projector_rank_equals_orbit_count(name, G, expected):
    Q = reynolds_projector(G)
    assert int(np.linalg.matrix_rank(Q, tol=1e-9)) == expected
    assert np.allclose(Q @ Q, Q, atol=1e-9)  # idempotent


@pytest.mark.parametrize("name,G,expected", GRAPHS)
def test_diffusion_operator_is_equivariant_and_sector_preserving(name, G, expected):
    cert = verify_diffusion_equivariance(G)
    assert cert.is_equivariant
    assert cert.equivariance_residual < cert.tolerance
    assert cert.sector_preservation_residual < cert.tolerance
    assert cert.orbit_count == expected


@pytest.mark.parametrize("name,G,expected", GRAPHS)
def test_relabel_invariance(name, G, expected):
    rng = np.random.default_rng(0)
    new_labels = rng.permutation(G.number_of_nodes())
    mapping = {old: int(new_labels[i]) for i, old in enumerate(G.nodes())}
    H = nx.relabel_nodes(G, mapping)
    assert orbit_count(H) == expected
    assert verify_diffusion_equivariance(H).is_equivariant


def test_decompose_state_symmetric_field_has_zero_perp():
    G = nx.cycle_graph(8)
    fixed, perp = decompose_state(G, np.ones(8))
    assert np.linalg.norm(perp) < 1e-12  # a constant field is already in Fix(Γ)
    assert np.allclose(fixed, np.ones(8), atol=1e-12)


def test_decompose_state_reconstructs_field():
    G = nx.path_graph(6)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(6)
    fixed, perp = decompose_state(G, v)
    assert np.allclose(fixed + perp, v, atol=1e-12)
    # the Fix(Γ) component is orbit-constant
    assert is_orbit_constant(G, fixed)


def test_orbit_constant_detects_non_orbit_constant_field():
    G = nx.star_graph(5)  # orbits: {center}, {leaves}
    assert is_orbit_constant(G, {n: (0.0 if n == 0 else 1.0) for n in G.nodes()})
    leaf_varying = {n: float(n) for n in G.nodes()}  # differs within the leaf orbit
    assert not is_orbit_constant(G, leaf_varying)


def test_weighted_automorphisms_reduce_symmetry():
    G = nx.cycle_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    assert orbit_count(G, weight="weight") == 1  # uniform weights: still 1 orbit
    G[0][1]["weight"] = 5.0  # a distinguished heavy edge breaks vertex-transitivity
    assert orbit_count(G, weight="weight") > 1


def test_directed_cycle_is_equivariant_single_orbit():
    G = nx.cycle_graph(5, create_using=nx.DiGraph)  # 0->1->..->4->0
    cert = verify_diffusion_equivariance(G)
    assert cert.is_equivariant
    assert cert.orbit_count == 1  # rotations act transitively


def test_star_operator_preserves_sectors_despite_asymmetric_operator():
    G = nx.star_graph(5)
    nodes, L = structural_diffusion_operator(G)
    Q = reynolds_projector(G, nodes=nodes)
    # L_rw of the star is non-symmetric, yet still commutes with the projector
    assert sector_preservation_residual(L, Q) < 1e-9
    assert len(automorphism_orbits(G, nodes=nodes)) == 2


def test_automatic_group_is_complete_at_cap_and_rejects_one_missing_map():
    graph = nx.cycle_graph(3)
    full = automorphism_permutations(graph, cap=6)
    assert len(full) == 6
    assert {tuple(mapping[i] for i in graph) for mapping in full} == set(
        permutations(range(3))
    )
    for owner in (automorphism_permutations, automorphism_orbits, reynolds_projector):
        with pytest.raises(ValueError, match="exceeds cap"):
            owner(graph, cap=5)


@pytest.mark.parametrize("cap", (True, False, 0, -1, 1.0, "6", None))
def test_automatic_group_requires_a_positive_nonboolean_integer_cap(cap):
    with pytest.raises(ValueError, match="positive nonboolean integer"):
        automorphism_permutations(nx.path_graph(2), cap=cap)


@pytest.mark.parametrize("graph_type", (nx.MultiGraph, nx.MultiDiGraph))
def test_automatic_multigraph_matching_requires_a_separate_contract(graph_type):
    graph = graph_type()
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(0, 1, weight=2)
    with pytest.raises(ValueError, match="multigraph"):
        automorphism_permutations(graph, weight="weight")
    # Explicit vertex actions do not claim multiedge/support authentication.
    assert np.array_equal(reynolds_projector(graph, permutations=[]), np.eye(2))


def test_near_but_unequal_weights_are_not_exact_graph_symmetries():
    graph = nx.path_graph(3)
    graph.edges[0, 1]["weight"] = 1.0
    graph.edges[1, 2]["weight"] = np.nextafter(1.0, 2.0)
    assert len(automorphism_permutations(graph)) == 2
    assert automorphism_permutations(graph, weight="weight") == [{0: 0, 1: 1, 2: 2}]
    assert np.array_equal(reynolds_projector(graph, weight="weight"), np.eye(3))


def test_missing_edge_weight_retains_exact_unit_default():
    graph = nx.path_graph(3)
    graph.edges[0, 1]["weight"] = 1.0
    assert len(automorphism_permutations(graph, weight="weight")) == 2


@pytest.mark.parametrize("value", (float("nan"), np.array([1.0, 2.0])))
def test_nonreflexive_or_nonscalar_weight_equality_is_rejected(value):
    graph = nx.path_graph(2)
    graph.edges[0, 1]["weight"] = value
    with pytest.raises(ValueError, match="scalar equality"):
        automorphism_permutations(graph, weight="weight")


@pytest.mark.parametrize("mapping", (
    {0: 0, 1: 1},                         # Missing source and destination.
    {0: 0, 1: 1, 2: 1},                   # Duplicate destination.
    {0: 0, 1: 1, 2: 3},                   # Foreign destination.
    {0: 0, 1: 1, 2: 2, 3: 3},             # Foreign source.
    {0: 0, 1: 1, 2: []},                  # Unhashable destination.
    (0, 1, 2),                            # No declared source mapping.
))
def test_every_permutation_consumer_rejects_incomplete_or_nonbijective_maps(mapping):
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match="permutation"):
        permutation_matrix(mapping, list(graph))
    for owner in (automorphism_orbits, reynolds_projector):
        with pytest.raises(ValueError, match="permutation"):
            owner(graph, permutations=[mapping])


@pytest.mark.parametrize("nodes", ([0, 0, 2], [0, 1], [0, 1, 3], {0, 1, 2}))
def test_orbit_and_projector_node_order_covers_the_entire_graph(nodes):
    graph = nx.path_graph(3)
    for owner in (automorphism_orbits, reynolds_projector):
        with pytest.raises(ValueError, match="node|vertices"):
            owner(graph, nodes=nodes, permutations=[])


def test_permutation_matrix_rejects_duplicate_or_unordered_node_domain():
    for nodes in ([0, 0], {0, 1}):
        with pytest.raises(ValueError, match="nodes"):
            permutation_matrix({0: 0, 1: 1}, nodes)


def test_incomplete_generator_list_produces_the_generated_group_projector():
    graph = nx.cycle_graph(3)
    identity = {0: 0, 1: 1, 2: 2}
    rotation = {0: 1, 1: 2, 2: 0}
    raw_average = (np.eye(3) + permutation_matrix(rotation, list(graph))) / 2
    assert not np.allclose(raw_average @ raw_average, raw_average)
    expected = np.full((3, 3), 1 / 3)
    for generators in ([rotation], [identity, rotation], [rotation] * 4):
        result = reynolds_projector(graph, permutations=generators)
        assert np.array_equal(result, expected)
        assert np.array_equal(result, result.T)
        assert np.allclose(result @ result, result, atol=1e-15, rtol=0)
        assert automorphism_orbits(graph, permutations=generators) == [(0, 1, 2)]


def test_full_group_average_and_generated_orbit_projector_agree():
    graph = nx.path_graph(4)
    # Two disjoint exchanges generate the Klein four-group. They intentionally
    # do not preserve path support: only the supplied vertex action is claimed.
    left = {0: 1, 1: 0, 2: 2, 3: 3}
    right = {0: 0, 1: 1, 2: 3, 3: 2}
    group = ({0: 0, 1: 1, 2: 2, 3: 3}, left, right,
             {0: 1, 1: 0, 2: 3, 3: 2})
    expected = sum(permutation_matrix(item, list(graph)) for item in group) / 4
    result = reynolds_projector(graph, permutations=(item for item in (left, right)))
    assert np.array_equal(result, expected)
    assert np.array_equal(result @ result, result)
    assert orbit_count(graph, permutations=[left, right]) == 2


def test_empty_generators_mean_identity_and_preserve_arbitrary_fields():
    graph = nx.path_graph(3)
    assert automorphism_orbits(graph, permutations=[]) == [(0,), (1,), (2,)]
    assert np.array_equal(reynolds_projector(graph, permutations=[]), np.eye(3))
    field = np.array([1.0, -3.0, 7.0])
    fixed, hidden = decompose_state(graph, field, permutations=[])
    assert np.array_equal(fixed, field)
    assert np.array_equal(hidden, np.zeros(3))


def test_reordered_domain_preserves_map_orientation_and_orbit_coordinates():
    graph = nx.path_graph(3)
    mapping = {0: 2, 1: 1, 2: 0}
    nodes = [2, 0, 1]
    matrix = permutation_matrix(mapping, nodes)
    assert np.array_equal(matrix @ np.array([10, 20, 30]), [20, 10, 30])
    result = reynolds_projector(graph, nodes=nodes, permutations=[mapping])
    assert np.array_equal(result, [[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])
    assert np.array_equal(reynolds_projector(graph, nodes=graph.nodes),
                          reynolds_projector(graph))


def test_empty_graph_has_a_well_defined_trivial_action():
    graph = nx.Graph()
    assert automorphism_permutations(graph, cap=1) == [{}]
    assert automorphism_orbits(graph) == []
    assert reynolds_projector(graph).shape == (0, 0)
    assert permutation_matrix({}, []).shape == (0, 0)

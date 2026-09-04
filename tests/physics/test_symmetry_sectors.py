r"""Tests for the R1 symmetry-sector observability (diffusion-sector base case).

Certifies, across the standard graph family (cycle, complete, star, path,
torus), that the canonical operator ``L_rw`` is equivariant under ``Aut(G)`` and
preserves the Reynolds sectors, that ``rank Q_Γ = #orbits``, and that the
observables are relabel-invariant and weight/direction aware.
"""

from __future__ import annotations

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
    decompose_state,
    is_orbit_constant,
    orbit_count,
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

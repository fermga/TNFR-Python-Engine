"""Fixed polyhedral graph controls do not select a geometry dynamically.

All five supplied skeletons share the connected pure-EPI diffusion theorem:
uniform form is the only stationary form, and nonuniform form dissipates.
With held uniform phase/capacity their regular degree also removes the
topology channel. An irregular graph supplies both the same unforced
equilibrium and a nonuniform equilibrium of the configured held-source law.
These finite identities do not prove a graph-selection law, autonomous joint
equilibrium, or an embedding as a regular Euclidean solid.
"""

from fractions import Fraction as F

import networkx as nx
import pytest

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.mathematics.krylov import exact_rank
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import capture_non_epi_forcing

PURE_EPI = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
ALL_CHANNELS = {"phase": 0.25, "epi": 0.25, "vf": 0.25, "topo": 0.25}
PLATONIC_CASES = (
    pytest.param(nx.tetrahedral_graph, 4, 3, id="tetrahedron"),
    pytest.param(nx.cubical_graph, 8, 3, id="cube"),
    pytest.param(nx.octahedral_graph, 6, 4, id="octahedron"),
    pytest.param(nx.dodecahedral_graph, 20, 3, id="dodecahedron"),
    pytest.param(nx.icosahedral_graph, 12, 5, id="icosahedron"),
)


def _prepared(graph, epi, weights):
    for node, value in zip(graph, epi, strict=True):
        graph.nodes[node].update(EPI=value, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.graph["DNFR_WEIGHTS"] = dict(weights)
    default_compute_delta_nfr(graph)
    return graph, capture_non_epi_forcing(graph)


def _apply(matrix, values):
    return tuple(
        sum((a * b for a, b in zip(row, values, strict=True)), F(0)) for row in matrix
    )


def _exact_diffusion_check(graph, capture):
    nodes = tuple(graph)
    degree = tuple(len(graph[node]) for node in nodes)
    # Independently assemble the exact random-walk generator from this
    # supplied simple unit graph, then compare its action to production.
    generator = tuple(
        tuple(
            F(int(i == j)) - F(int(graph.has_edge(i, j)), len(graph[i])) for j in nodes
        )
        for i in nodes
    )
    assert exact_rank(generator) == len(nodes) - 1
    assert _apply(generator, (F(1),) * len(nodes)) == (0,) * len(nodes)
    epi = capture.snapshot.epi
    model_rate = tuple(-value for value in _apply(generator, epi))
    assert model_rate == capture.snapshot.epi_gradient
    assert capture.epi_weight == 1 and capture.forcing == (0,) * len(nodes)
    assert sum((d * v for d, v in zip(degree, model_rate)), F(0)) == 0
    assert (
        tuple(
            p - defect
            for p, defect in zip(
                capture.full_kernel_pressure,
                capture.kernel_pressure_defect,
            )
        )
        == model_rate
    )
    assert capture.full_kernel_pressure == capture.snapshot.stored_pressure
    assert capture.stored_pressure_residual == (0,) * len(nodes)
    # The exact model dissipation is separate from binary64 pressure assembly.
    gradient = tuple(-d * value for d, value in zip(degree, model_rate))
    energy_rate = sum((g * v for g, v in zip(gradient, model_rate)), F(0))
    assert energy_rate == -sum((g * g / d for g, d in zip(gradient, degree)), F(0))
    assert energy_rate < 0
    return energy_rate


@pytest.mark.parametrize("builder,size,degree", PLATONIC_CASES)
def test_all_five_supplied_skeletons_share_diffusion_without_shape_selection(
    builder,
    size,
    degree,
):
    graph = builder()
    assert len(graph) == size and set(dict(graph.degree).values()) == {degree}
    graph, capture = _prepared(graph, (1.0,) + (0.0,) * (size - 1), PURE_EPI)
    assert _exact_diffusion_check(graph, capture) == -degree - 1
    assert capture.snapshot.dirichlet_energy == F(degree, 2)

    # Regular degree, held equal phase and held equal capacity make every
    # source channel zero. This is an EPI equilibrium on an already supplied
    # support; none of these reads evolves or selects that support.
    _, uniform = _prepared(builder(), (-2.0,) * size, ALL_CHANNELS)
    assert uniform.snapshot.epi_gradient == (0,) * size
    assert uniform.phase_gradient == uniform.snapshot.capacity_gradient == (0,) * size
    assert uniform.snapshot.topology_gradient == (0,) * size
    assert uniform.forcing == uniform.snapshot.stored_pressure == (0,) * size


def test_irregular_graph_has_the_same_unique_unforced_consensus_equilibrium():
    graph, capture = _prepared(nx.path_graph(4), (1.0, 0.0, 0.0, 0.0), PURE_EPI)
    assert tuple(dict(graph.degree).values()) == (1, 2, 2, 1)
    assert _exact_diffusion_check(graph, capture) == F(-3, 2)
    _, uniform = _prepared(nx.path_graph(4), (-2.0,) * 4, PURE_EPI)
    assert uniform.snapshot.topology_gradient == (1, F(-1, 2), F(-1, 2), 1)
    assert uniform.snapshot.stored_pressure == (0,) * 4


def test_irregular_support_also_has_a_nonuniform_exact_held_source_equilibrium():
    _, uniform = _prepared(nx.path_graph(4), (0.0,) * 4, ALL_CHANNELS)
    assert uniform.snapshot.topology_gradient == (1, F(-1, 2), F(-1, 2), 1)
    assert (
        uniform.forcing
        == uniform.snapshot.stored_pressure
        == (
            F(1, 4),
            F(-1, 8),
            F(-1, 8),
            F(1, 4),
        )
    )
    # This fixed nonuniform state cancels the independently known degree
    # source. It is not obtained by fitting stored pressure to a desired rate.
    _, stationary = _prepared(nx.path_graph(4), (1.0, 0.0, 0.0, 1.0), ALL_CHANNELS)
    assert stationary.forcing == uniform.forcing
    assert stationary.snapshot.epi_gradient == (-1, F(1, 2), F(1, 2), -1)
    assert stationary.snapshot.stored_pressure == (0,) * 4
    reference = derive_forced_support_balance(
        stationary.snapshot,
        epi_weight=stationary.epi_weight,
        forcing=stationary.forcing,
    )
    assert reference.compatibility_residual == 0
    assert reference.has_zero_pressure_equilibrium
    assert reference.relative_profile == (F(2, 3), F(-1, 3), F(-1, 3), F(2, 3))
    assert (
        tuple(
            x - z for x, z in zip(stationary.snapshot.epi, reference.relative_profile)
        )
        == (F(1, 3),) * 4
    )


def test_vertex_transitivity_does_not_make_every_laplacian_eigenspace_irreducible():
    # A fixed counterexample to the legacy "multiplicity = one irrep" claim.
    # The central involution splits each five-dimensional eigenspace into
    # two nonzero invariant subspaces. Exact rank, not spectral clustering
    # or a partial group average, establishes their dimensions.
    graph = nx.truncated_cube_graph()
    nodes = tuple(graph)
    mappings = nx.algorithms.isomorphism.GraphMatcher(graph, graph).isomorphisms_iter()
    permutations = tuple(tuple(mapping[i] for i in nodes) for mapping in mappings)
    assert len(permutations) == 48
    assert {permutation[0] for permutation in permutations} == set(nodes)
    central = tuple(
        p
        for p in permutations
        if p != nodes
        and all(
            tuple(p[q[i]] for i in nodes) == tuple(q[p[i]] for i in nodes)
            for q in permutations
        )
    )
    assert len(central) == 1
    p = central[0]
    assert all(p[p[i]] == i for i in nodes)
    for eigenvalue, expected in ((3, (3, 2)), (5, (2, 3))):
        eigenspace = tuple(
            tuple(
                F((3 - eigenvalue) * int(i == j) - int(graph.has_edge(i, j)))
                for j in nodes
            )
            for i in nodes
        )
        assert len(nodes) - exact_rank(eigenspace) == 5
        dimensions = []
        for sign in (-1, 1):
            parity = tuple(
                tuple(F(int(j == p[i]) - sign * int(i == j)) for j in nodes)
                for i in nodes
            )
            dimensions.append(len(nodes) - exact_rank((*eigenspace, *parity)))
        assert tuple(dimensions) == expected

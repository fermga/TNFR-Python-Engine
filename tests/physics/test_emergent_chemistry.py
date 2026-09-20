"""Tests for the assumption-explicit structural shell-model comparison."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.emergent_chemistry import (
    aufbau_subshell_order,
    classify_element,
    electron_configuration,
    emergent_magic_numbers,
    fibonacci_sphere_graph,
    shell_closure_distance,
    structural_eigenmodes,
    valence_delta_nfr,
)
from tnfr.physics.fields import classify_nodal_topology


def _solid_ball(n_shells: int = 4, base: int = 16, k: int = 8) -> nx.Graph:
    """A solid 3D ball: center point + concentric fibonacci shells, all wired
    by ONE 3D k-NN rule (deterministic; the center is not privileged by
    construction). Mirrors benchmarks/emergent_shell_ordering.solid_ball_graph.
    """
    pts: list[np.ndarray] = [np.zeros(3)]
    for s in range(1, n_shells + 1):
        npts = base * s * s
        idx = np.arange(npts, dtype=float)
        golden = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - 2.0 * (idx + 0.5) / npts
        ring = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        th = golden * idx
        pts.extend(
            np.stack([s * ring * np.cos(th), s * ring * np.sin(th), s * z], axis=1)
        )
    P = np.asarray(pts)
    G = nx.Graph()
    G.add_nodes_from(range(len(P)))
    for i in range(len(P)):
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            G.add_edge(i, int(j))
    return G


@pytest.fixture(scope="module")
def ball() -> nx.Graph:
    return _solid_ball()


class TestFiniteAngularClustering:
    """Finite spectral clustering on the constructed sphere graph."""

    def test_sphere_low_modes_are_2l_plus_1(self):
        shells = structural_eigenmodes(fibonacci_sphere_graph(162, 6), max_modes=16)
        assert [sh.multiplicity for sh in shells[:4]] == [1, 3, 5, 7]

    def test_angular_index_inferred_from_multiplicity(self):
        shells = structural_eigenmodes(fibonacci_sphere_graph(162, 6), max_modes=16)
        assert [sh.angular_index for sh in shells[:4]] == [0, 1, 2, 3]

    def test_even_cluster_has_no_invented_angular_index(self):
        shells = structural_eigenmodes(nx.path_graph(4), max_modes=4, gap_factor=1000.0)
        assert len(shells) == 1
        assert shells[0].multiplicity == 4
        assert shells[0].angular_index is None

    def test_clustering_can_merge_distinct_eigenvalues_and_depend_on_truncation(self):
        # K5 has normalized spectrum {0, 5/4, 5/4, 5/4, 5/4}. The retained
        # heuristic merges the zero mode with the positive eigenspace; its
        # angular labels therefore do not certify true spectral degeneracy.
        graph = nx.complete_graph(5)
        full = structural_eigenmodes(graph, max_modes=5)
        prefix = structural_eigenmodes(graph, max_modes=3)
        assert len(full) == len(prefix) == 1
        assert (full[0].multiplicity, full[0].angular_index) == (5, 2)
        assert (prefix[0].multiplicity, prefix[0].angular_index) == (3, 1)
        assert full[0].eigenvalue == pytest.approx(1.0, rel=0, abs=2e-15)
        assert prefix[0].eigenvalue == pytest.approx(5 / 6, rel=0, abs=2e-15)

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"max_modes": 0}, ValueError),
            ({"max_modes": True}, TypeError),
            ({"gap_factor": 0.0}, ValueError),
            ({"gap_factor": float("nan")}, ValueError),
        ],
    )
    def test_invalid_clustering_controls_are_rejected(self, kwargs, error):
        with pytest.raises(error):
            structural_eigenmodes(nx.path_graph(4), **kwargs)


class TestAufbauIsPostulated:
    """The aufbau (n+l) filling order is a hardcoded counting rule (Madelung),
    not a spectral derivation -- the honest boundary the audit flagged."""

    def test_aufbau_order_is_madelung_sort(self):
        order = aufbau_subshell_order(max_n=4)
        # ordered by non-decreasing (n + l) -- the Madelung counting rule
        keys = [n + l for n, l in order]
        assert keys == sorted(keys)
        assert order[:3] == [(1, 0), (2, 0), (2, 1)]

    def test_magic_numbers_are_noble_gases(self):
        # By construction (aufbau + np-closure), reproduces the noble gases.
        assert emergent_magic_numbers()[:6] == [2, 10, 18, 36, 54, 86]

    def test_free_product_spectrum_is_not_atomic(self):
        # This supplied product graph and finite clustering protocol do not
        # yield the selected closure list. The control does not establish a
        # normalized-Laplacian sum formula or exclude other graph models.
        prod = nx.cartesian_product(fibonacci_sphere_graph(80, 6), nx.path_graph(6))
        shells = structural_eigenmodes(prod, max_modes=30, gap_factor=4.0)
        cum, total = [], 0
        for sh in shells:
            total += 2 * sh.multiplicity
            cum.append(total)
        assert cum[:6] != [2, 10, 18, 36, 54, 86]


class TestEmergentChemistryAPI:
    """Pin the module's public API directly (was only tested via the SDK)."""

    def test_declared_closures_have_zero_model_distance(self):
        # The independently defined shell distance reuses the scalar
        # zero-pressure predicate; it does not share arithmetic dynamics.
        for z in (2, 10, 18):
            assert shell_closure_distance(z) == 0.0
            assert valence_delta_nfr(z) == 0.0
            assert classify_element(z).closed_shell is True
            assert classify_element(z).reactivity == 0.0

    def test_nonclosure_has_positive_model_distance(self):
        assert valence_delta_nfr(11) > 0.0  # Na: one past the Ne closure
        assert classify_element(11).closed_shell is False

    def test_electron_configuration_fills_to_Z(self):
        config = electron_configuration(10)
        assert sum(occ for _n, _l, occ in config) == 10

    @pytest.mark.parametrize("value", [True, 2.5, float("nan")])
    def test_non_integral_element_counts_are_rejected(self, value):
        with pytest.raises((TypeError, ValueError)):
            electron_configuration(value)

    @pytest.mark.parametrize(
        ("args", "error"),
        [
            ((5, 0), ValueError),
            ((5, -1), ValueError),
            ((5, 5), ValueError),
            ((True, 1), TypeError),
        ],
    )
    def test_invalid_sphere_controls_are_rejected(self, args, error):
        with pytest.raises(error):
            fibonacci_sphere_graph(*args)

    def test_disconnected_sphere_discretization_is_rejected(self):
        with pytest.raises(ValueError, match="disconnected graph"):
            fibonacci_sphere_graph(100, 1)

    @pytest.mark.parametrize("z", [119, 120, 132, 140, 148, 156])
    def test_closed_and_magic_use_one_closure_set(self, z):
        result = classify_element(z)
        assert result.closed_shell is result.magic_number
        assert result.closed_shell is False
        assert result.delta_nfr == result.closure_distance


class TestSuppliedGeometryClassification:
    """Calibrated topology labels on supplied graphs, without nucleus formation."""

    def test_star_is_radial(self):
        assert classify_nodal_topology(nx.star_graph(30))["topology"] == "radial"

    def test_ring_is_annular(self):
        assert classify_nodal_topology(nx.cycle_graph(30))["topology"] == "annular"

    def test_supplied_sphere_is_classified_annular(self):
        topo = classify_nodal_topology(fibonacci_sphere_graph(120, 6))
        assert topo["topology"] == "annular"

    def test_supplied_solid_ball_has_one_classified_center(self, ball):
        topo = classify_nodal_topology(ball)
        assert topo["topology"] == "radial"
        assert len(topo["centers"]) == 1


class TestBallSpectralCounting:
    """Finite ball clusters with assumed doubled occupancy, not atomic shells."""

    def test_first_shells_are_angular_multiplets(self, ball):
        shells = structural_eigenmodes(ball, max_modes=40, gap_factor=4.0)
        assert [sh.multiplicity for sh in shells[:3]] == [1, 3, 5]

    def test_cumulative_closures_start_2_8_18(self, ball):
        shells = structural_eigenmodes(ball, max_modes=40, gap_factor=4.0)
        cum, total = [], 0
        for sh in shells:
            total += 2 * sh.multiplicity
            cum.append(total)
        assert cum[:3] == [2, 8, 18]

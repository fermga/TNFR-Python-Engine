"""Tests for the structural-diffusion characterization of the nodal equation.

Verifies that the EPI channel of ΔNFR is the random-walk graph Laplacian
(the discrete diffusion operator), so the nodal equation
∂EPI/∂t = νf·ΔNFR is structurally a diffusion equation.
"""
from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.physics.structural_diffusion import (
    StructuralDiffusionCertificate,
    structural_diffusion_operator,
    structural_field,
    structural_diffusivity,
    relaxation_spectrum,
    degree_weighted_total,
    verify_structural_diffusion,
)


def _canonical_graph(n: int = 60, seed: int = 11) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 6, 0.2, seed=seed)
    for node in G.nodes():
        G.nodes[node]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        G.nodes[node]["EPI"] = rng.uniform(-0.4, 0.4)
        G.nodes[node]["nu_f"] = rng.uniform(0.5, 1.5)
    default_compute_delta_nfr(G)
    return G


class TestDiffusionOperator:
    """The structural diffusion operator is the random-walk Laplacian."""

    def test_operator_shape(self) -> None:
        G = _canonical_graph(40)
        nodes, lap = structural_diffusion_operator(G)
        assert len(nodes) == 40
        assert lap.shape == (40, 40)

    def test_rows_sum_to_zero(self) -> None:
        # L_rw = I − D⁻¹W has zero row sums (diffusion conserves locally).
        G = _canonical_graph(40)
        _, lap = structural_diffusion_operator(G)
        assert np.allclose(lap.sum(axis=1), 0.0)

    def test_diagonal_is_one(self) -> None:
        G = _canonical_graph(30)
        _, lap = structural_diffusion_operator(G)
        # connected (non-isolated) nodes have diagonal 1.0
        assert np.allclose(np.diag(lap), 1.0)


class TestNodalEquationIsDiffusion:
    """The canonical ΔNFR EPI channel equals −L_rw·EPI (machine precision)."""

    def test_dnfr_epi_channel_is_graph_laplacian(self) -> None:
        G = _canonical_graph(50)
        nodes, lap = structural_diffusion_operator(G)
        epi = structural_field(G, nodes)
        # isolate the EPI channel on a clean structural replica
        from tnfr.constants.aliases import ALIAS_VF

        g2 = nx.Graph()
        for node in nodes:
            g2.add_node(
                node,
                EPI=float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)),
                theta=float(G.nodes[node].get("theta", 0.0)),
                nu_f=float(get_attr(G.nodes[node], ALIAS_VF, 0.0)),
            )
        g2.add_edges_from(G.edges())
        g2.graph["DNFR_WEIGHTS"] = {
            "phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0
        }
        default_compute_delta_nfr(g2)
        dnfr = np.array(
            [float(get_attr(g2.nodes[n], ALIAS_DNFR, 0.0)) for n in nodes]
        )
        assert np.allclose(dnfr, -(lap @ epi), atol=1e-12)

    def test_certificate_confirms_laplacian(self) -> None:
        G = _canonical_graph(60)
        cert = verify_structural_diffusion(G)
        assert cert.dnfr_is_graph_laplacian
        assert cert.max_laplacian_residual < 1e-12


class TestDiffusionConservation:
    """Diffusion conserves the degree-weighted structural total."""

    def test_degree_weighted_total_conserved(self) -> None:
        G = _canonical_graph(60)
        cert = verify_structural_diffusion(G)
        assert cert.degree_weighted_conserved
        assert cert.max_conservation_drift < 1e-9

    def test_degree_weighted_total_value(self) -> None:
        import pytest

        G = _canonical_graph(40)
        # the helper matches a direct degree-weighted sum
        nodes, _ = structural_diffusion_operator(G)
        epi = structural_field(G, nodes)
        deg = np.array([G.degree(n) for n in nodes], dtype=float)
        assert degree_weighted_total(G) == pytest.approx(float(deg @ epi))


class TestDiffusionRelaxation:
    """The field relaxes to a uniform diffusive equilibrium."""

    def test_relaxes_to_uniform(self) -> None:
        G = _canonical_graph(60)
        cert = verify_structural_diffusion(G)
        assert cert.relaxes_to_uniform
        assert cert.final_field_std < 1e-3

    def test_spectrum_has_zero_mode(self) -> None:
        # λ₁ = 0 (the conserved uniform diffusion mode).
        G = _canonical_graph(50)
        spec = relaxation_spectrum(G)
        assert abs(float(spec[0])) < 1e-9
        assert float(spec[1]) > 0.0  # spectral gap positive

    def test_diffusivity_is_mean_vf(self) -> None:
        import pytest

        from tnfr.constants.aliases import ALIAS_VF

        G = _canonical_graph(40)
        nodes, _ = structural_diffusion_operator(G)
        vf = [float(get_attr(G.nodes[n], ALIAS_VF, 0.0)) for n in nodes]
        assert structural_diffusivity(G) == pytest.approx(
            float(np.mean(vf)), rel=1e-6
        )


class TestDiffusionCertificate:
    """The certificate aggregates the diffusion verdict."""

    def test_certificate_valid(self) -> None:
        G = _canonical_graph(60)
        cert = verify_structural_diffusion(G)
        assert isinstance(cert, StructuralDiffusionCertificate)
        assert cert.is_valid_diffusion
        assert "VALID" in cert.summary()
        assert cert.diffusivity > 0.0
        assert cert.spectral_gap > 0.0

    def test_graph_not_mutated(self) -> None:
        # verify runs the ΔNFR check on a replica; caller's graph is untouched.
        G = _canonical_graph(40)
        before = {n: float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0))
                  for n in G.nodes()}
        weights_before = G.graph.get("DNFR_WEIGHTS")
        verify_structural_diffusion(G)
        after = {n: float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0))
                 for n in G.nodes()}
        assert before == after
        assert G.graph.get("DNFR_WEIGHTS") == weights_before

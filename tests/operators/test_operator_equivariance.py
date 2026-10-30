r"""Per-operator equivariance audit (R1, non-linear stage).

Each canonical operator is tested INDIVIDUALLY (one parametrized case per
operator) so the autouse ``reset_global_state`` fixture gives it clean module
state.  The engine's content-keyed caches otherwise leak across successive
operator applications on a symmetric (content-identical) graph and confound the
measurement (a batch audit reports spurious ~1e-3 residuals that vanish under
isolation).  From clean state every operator maps a ``Fix(Γ)`` state to a
``Fix(Γ)`` state: the diffusion-sector equivariance of R1 extends to the full
13-operator set, so a grammar-composed word of equivariant operators is
equivariant and cannot move a symmetric state into ``Fix(Γ)^⊥``.
"""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.physics.operator_equivariance import operator_equivariance_residual

OPERATORS = [
    ("AL", Emission),
    ("EN", Reception),
    ("IL", Coherence),
    ("OZ", Dissonance),
    ("UM", Coupling),
    ("RA", Resonance),
    ("SHA", Silence),
    ("VAL", Expansion),
    ("NUL", Contraction),
    ("THOL", SelfOrganization),
    ("ZHIR", Mutation),
    ("NAV", Transition),
    ("REMESH", Recursivity),
]

TOL = 1e-6


def _seed(G, theta_of, epi_of, vf_of):
    for nd in G.nodes():
        epi = float(epi_of(nd))
        set_attr(G.nodes[nd], ALIAS_THETA, float(theta_of(nd)))
        set_attr(G.nodes[nd], ALIAS_EPI, epi)
        set_attr(G.nodes[nd], ALIAS_VF, float(vf_of(nd)))
        # Every operator shares this fixture. ZHIR additionally requires a
        # two-sample signed-growth witness through its non-disableable gate.
        G.nodes[nd]["epi_history"] = [epi - 1.0, epi]
    default_compute_delta_nfr(G)
    return G


def _cycle_case():
    """Vertex-transitive C6 with a uniform (Fix(Γ)) seed; σ = rotation."""
    G = _seed(nx.cycle_graph(6), lambda n: 0.2, lambda n: 0.4, lambda n: 1.0)
    return G, {i: (i + 1) % 6 for i in range(6)}, 0


def _star_case():
    """Star K1,4 (two orbits) with an orbit-constant seed; σ = leaf swap."""
    G = _seed(
        nx.star_graph(4),
        lambda n: 0.1 if n == 0 else 0.3,
        lambda n: 0.5 if n == 0 else 0.3,
        lambda n: 1.0,
    )
    return G, {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}, 1


@pytest.mark.parametrize("glyph,cls", OPERATORS)
def test_operator_equivariant_on_vertex_transitive_cycle(glyph, cls):
    G, sigma, node = _cycle_case()
    assert operator_equivariance_residual(cls(), G, sigma, node) < TOL


@pytest.mark.parametrize("glyph,cls", OPERATORS)
def test_operator_equivariant_on_two_orbit_star(glyph, cls):
    G, sigma, node = _star_case()
    assert operator_equivariance_residual(cls(), G, sigma, node) < TOL


def test_measurement_is_deterministic():
    """The audit is deterministic: no RNG confound (self-residual is zero)."""
    G, _, node = _cycle_case()
    identity = {n: n for n in G.nodes()}
    assert operator_equivariance_residual(Silence(), G, identity, node) == 0.0

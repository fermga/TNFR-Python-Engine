"""Canonicity regression tests: U3 is a hard invariant for UM and RA.

Locks in the 2026-09-04 audit fix for invariant #2: the phase gate
|φ_i − φ_j| ≤ Δφ_max (Δφ_max = π/2) is enforced on every Coupling/Resonance
application, independent of ``VALIDATE_OPERATOR_PRECONDITIONS``, and **raises**
before any state mutation (it is not a warning and cannot be disabled).
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Coupling, Resonance
from tnfr.operators.preconditions import OperatorPreconditionError


def _two_node_graph(theta_j: float):
    G = nx.Graph()
    G.add_node(0, EPI=0.6, vf=1.0, theta=0.0, dnfr=0.0)
    G.add_node(1, EPI=0.6, vf=1.0, theta=theta_j, dnfr=0.0)
    G.add_edge(0, 1)
    return G


def _snapshot(G):
    return {
        n: (
            get_attr(G.nodes[n], ALIAS_THETA, 0.0),
            get_attr(G.nodes[n], ALIAS_EPI, 0.0),
            get_attr(G.nodes[n], ALIAS_VF, 0.0),
            get_attr(G.nodes[n], ALIAS_DNFR, 0.0),
        )
        for n in G.nodes()
    }


def test_um_phase_gate_is_unconditional():
    """UM raises on an incompatible neighbour with default config (no flags)."""
    G = _two_node_graph(2.0 * math.pi / 3.0)  # 120° > 90°
    before = _snapshot(G)
    with pytest.raises(OperatorPreconditionError):
        Coupling()(G, 0)
    assert _snapshot(G) == before  # state untouched before mutation


def test_ra_phase_gate_is_unconditional():
    """RA raises (not warns) on an incompatible neighbour by default."""
    G = _two_node_graph(2.2)  # > π/2
    before = _snapshot(G)
    with pytest.raises(OperatorPreconditionError):
        Resonance()(G, 0)
    assert _snapshot(G) == before


def test_phase_gate_uses_pi_over_two_not_one_radian():
    """A 1.4 rad mismatch (< π/2, > legacy 1.0) is admissible for RA."""
    G = _two_node_graph(1.4)
    # Should not raise the U3 gate (1.4 < π/2 ≈ 1.5708).
    Resonance()(G, 0)


def test_phase_wrap_near_2pi():
    """Phase compatibility uses wrapped angular distance."""
    G = _two_node_graph(2.0 * math.pi - 0.01)  # wrapped Δφ ≈ 0.02
    # Compatible after wrapping -> no U3 rejection.
    Coupling()(G, 0)


def test_any_compatible_neighbor_allows_operator():
    """One compatible neighbour is enough to admit the operator."""
    G = nx.Graph()
    G.add_node(0, EPI=0.6, vf=1.0, theta=0.0, dnfr=0.0)
    G.add_node(1, EPI=0.6, vf=1.0, theta=2.5, dnfr=0.0)  # incompatible
    G.add_node(2, EPI=0.6, vf=1.0, theta=0.3, dnfr=0.0)  # compatible
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    Coupling()(G, 0)  # admitted via the compatible neighbour


def test_isolated_node_passes_phase_gate():
    """An isolated node has no phase conflict; the gate does not raise."""
    G = nx.Graph()
    G.add_node(0, EPI=0.6, vf=1.0, theta=0.0, dnfr=0.0)
    Coupling()(G, 0)

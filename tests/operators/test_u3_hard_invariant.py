"""Canonicity regression tests: U3 is a hard invariant for UM and RA.

Locks in the 2026-09-04 audit fix for invariant #2: the phase gate
|φ_i − φ_j| ≤ Δφ_max (Δφ_max = π/2) is enforced on every Coupling/Resonance
application, independent of ``VALIDATE_OPERATOR_PRECONDITIONS``, and **raises**
before any state mutation (it is not a warning and cannot be disabled).
"""

from __future__ import annotations

from copy import deepcopy
import math

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.node import NodeNX
from tnfr.operators import _op_UM, apply_glyph, get_glyph_factors
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


def _mutation_snapshot(G):
    """Capture all observable UM/RA mutation surfaces except gate inputs."""

    return {
        "nodes": deepcopy(dict(G.nodes(data=True))),
        "edges": deepcopy(tuple(G.edges(data=True))),
        "history": deepcopy(
            {node: G.nodes[node].get("glyph_history") for node in G.nodes}
        ),
        "operator_metrics": deepcopy(G.graph.get("operator_metrics")),
        "ra_metrics": deepcopy(G.graph.get("ra_metrics")),
        "ra_tracking": deepcopy(G.graph.get("_ra_c_tracking")),
        "has_node_cache": "_node_cache" in G.graph,
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
    """UM derives every synchronized channel from compatible neighbours only."""
    G = nx.Graph()
    G.add_node(0, EPI=0.6, frequency=1.0, theta=0.0, dnfr=1.0)
    G.add_node(
        1, EPI=0.6, frequency=100.0, theta=2.5, dnfr=7.0
    )  # incompatible
    G.add_node(
        2, EPI=0.6, frequency=3.0, theta=0.4, dnfr=5.0
    )  # compatible
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.graph["UM_FUNCTIONAL_LINKS"] = False
    factors = get_glyph_factors(NodeNX.from_graph(G, 0), "UM")
    k_theta = factors["UM_theta_push"]
    k_vf = factors["UM_vf_sync"]
    k_dnfr = factors["UM_dnfr_reduction"]

    apply_glyph(G, 0, "UM")

    consensus = 0.2
    expected_target_phase = k_theta * consensus
    expected_good_phase = 0.4 + k_theta * (consensus - 0.4)
    from tnfr.metrics.phase_compatibility import compute_phase_coupling_strength

    alignment = compute_phase_coupling_strength(
        expected_target_phase, expected_good_phase
    )
    assert get_attr(G.nodes[0], ALIAS_THETA, 0.0) == pytest.approx(
        expected_target_phase
    )
    assert get_attr(G.nodes[2], ALIAS_THETA, 0.0) == pytest.approx(
        expected_good_phase
    )
    assert get_attr(G.nodes[0], ALIAS_VF, 0.0) == pytest.approx(
        1.0 + k_vf * (3.0 - 1.0)
    )
    assert get_attr(G.nodes[0], ALIAS_DNFR, 0.0) == pytest.approx(
        1.0 * (1.0 - k_dnfr * alignment)
    )
    assert _snapshot(G)[1] == (2.5, 0.6, 100.0, 7.0)


@pytest.mark.parametrize("operator", [Coupling(), Resonance()])
def test_isolated_node_is_rejected_atomically(operator):
    """A concrete coupling/propagation must have an admissible edge."""
    G = nx.Graph()
    G.add_node(0, EPI=0.6, vf=1.0, theta=0.0, dnfr=0.0)
    before = _mutation_snapshot(G)
    with pytest.raises(OperatorPreconditionError, match="coupled neighbor"):
        operator(G, 0)
    assert _mutation_snapshot(G) == before


@pytest.mark.parametrize("glyph", ["UM", "RA"])
@pytest.mark.parametrize(
    "invalid_limit",
    [True, -0.1, math.pi / 2.0 + 1e-12, float("inf"), float("nan"), "bad"],
)
def test_apply_glyph_rejects_invalid_hard_phase_limits_before_adapter_cache(
    glyph, invalid_limit
):
    G = _two_node_graph(0.1)
    G.graph["DELTA_PHI_MAX"] = invalid_limit
    before = _mutation_snapshot(G)

    with pytest.raises(TNFRValueError, match="phase gate"):
        apply_glyph(G, 0, glyph)

    assert _mutation_snapshot(G) == before
    assert "_node_cache" not in G.graph


@pytest.mark.parametrize("invalid_limit", [True, -0.1, float("inf"), float("nan")])
def test_um_tightening_limit_must_be_finite_and_nonnegative(invalid_limit):
    G = _two_node_graph(0.1)
    G.graph["UM_MAX_PHASE_DIFF"] = invalid_limit
    before = _mutation_snapshot(G)
    with pytest.raises(TNFRValueError, match="phase gate"):
        apply_glyph(G, 0, "UM")
    assert _mutation_snapshot(G) == before


@pytest.mark.parametrize(
    ("neighbor_phase", "um_limit", "admissible"),
    [
        (0.2, 0.1, False),
        (2.0, math.pi, False),
        (1.4, math.pi, True),
    ],
)
def test_um_limit_can_tighten_but_never_weaken_hard_gate(
    neighbor_phase, um_limit, admissible
):
    G = _two_node_graph(neighbor_phase)
    G.graph.update(UM_MAX_PHASE_DIFF=um_limit, UM_FUNCTIONAL_LINKS=False)
    if admissible:
        apply_glyph(G, 0, "UM")
    else:
        before = _mutation_snapshot(G)
        with pytest.raises(TNFRValueError, match="phase gate"):
            apply_glyph(G, 0, "UM")
        assert _mutation_snapshot(G) == before


def test_direct_um_primitive_rejects_incompatible_neighbor_atomically():
    G = _two_node_graph(2.0)
    node = NodeNX.from_graph(G, 0)
    factors = get_glyph_factors(node, "UM")
    before = _mutation_snapshot(G)

    with pytest.raises(TNFRValueError, match="no compatible neighbor"):
        _op_UM(node, factors)

    assert _mutation_snapshot(G) == before


def test_isolated_um_does_not_create_a_phase_compatible_functional_link():
    G = nx.Graph()
    G.add_node(0, EPI=0.6, vf=1.0, theta=0.0, dnfr=0.0, Si=1.0)
    G.add_node(1, EPI=0.6, vf=1.0, theta=0.0, dnfr=0.0, Si=1.0)
    G.graph["UM_COMPAT_THRESHOLD"] = 0.0
    before = _mutation_snapshot(G)

    with pytest.raises(TNFRValueError, match="coupled neighbor"):
        apply_glyph(G, 0, "UM")

    assert _mutation_snapshot(G) == before
    assert not G.has_edge(0, 1)


def test_functional_link_candidates_are_filtered_by_the_hard_gate():
    G = nx.Graph()
    for node, phase in ((0, 0.0), (1, 0.1), (2, math.pi), (3, 0.2)):
        G.add_node(node, EPI=0.6, vf=1.0, theta=phase, dnfr=0.0, Si=1.0)
    G.add_edge(0, 1)
    G.graph.update(
        UM_COMPAT_THRESHOLD=0.0,
        UM_BIDIRECTIONAL=False,
        UM_SYNC_VF=False,
        UM_STABILIZE_DNFR=False,
    )

    apply_glyph(G, 0, "UM")

    assert not G.has_edge(0, 2)
    assert G.has_edge(0, 3)

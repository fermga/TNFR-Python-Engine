"""Tests for the classical interface-baseline suite.

Each baseline is verified on small hand-constructed graphs with known expected
values (Milestone 2 acceptance criterion).
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.validation.interface_baselines import (
    BASELINE_FORMULAS,
    compute_all_baselines,
    constant_baseline,
    degree_score,
    feature_deviation,
    graph_cut_contribution,
    graph_total_variation,
    label_propagation_residual,
    local_class_entropy,
    local_disagreement,
    mean_neighbour_distance,
    random_baseline,
)


def _alternating_path() -> nx.Graph:
    """Path 0-1-2 with labels a, b, a (node 1 is a lone boundary)."""
    G = nx.path_graph(3)
    G.nodes[0]["state"] = "a"
    G.nodes[1]["state"] = "b"
    G.nodes[2]["state"] = "a"
    return G


def _uniform_path() -> nx.Graph:
    G = nx.path_graph(4)
    for node in G.nodes():
        G.nodes[node]["state"] = "same"
    return G


# ---------------------------------------------------------------------------
# local_disagreement
# ---------------------------------------------------------------------------


def test_local_disagreement_alternating() -> None:
    G = _alternating_path()
    scores = local_disagreement(G, state_key="state")
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(2.0)
    assert scores[2] == pytest.approx(1.0)


def test_local_disagreement_uniform_is_zero() -> None:
    G = _uniform_path()
    scores = local_disagreement(G, state_key="state")
    assert set(scores.values()) == {0.0}


# ---------------------------------------------------------------------------
# graph_total_variation
# ---------------------------------------------------------------------------


def test_graph_total_variation_state_codes_matches_disagreement_binary() -> None:
    G = _alternating_path()
    tv = graph_total_variation(G, state_key="state")
    disagreement = local_disagreement(G, state_key="state")
    # For binary labels the two coincide (documented honest caveat).
    assert tv == pytest.approx(disagreement)


def test_graph_total_variation_numeric_value() -> None:
    G = nx.path_graph(3)
    for node, value in zip(G.nodes(), [0.0, 0.0, 5.0]):
        G.nodes[node]["val"] = value
    tv = graph_total_variation(G, value_key="val")
    assert tv[0] == pytest.approx(0.0)
    assert tv[1] == pytest.approx(5.0)
    assert tv[2] == pytest.approx(5.0)


def test_graph_total_variation_requires_a_signal() -> None:
    with pytest.raises(ValueError):
        graph_total_variation(nx.path_graph(2))


# ---------------------------------------------------------------------------
# local_class_entropy
# ---------------------------------------------------------------------------


def test_local_class_entropy_balanced_is_one() -> None:
    G = _alternating_path()
    entropy = local_class_entropy(G, state_key="state")
    # Node 0 closed neighbourhood {a, b} is perfectly balanced -> normalized 1.
    assert entropy[0] == pytest.approx(1.0)
    assert entropy[2] == pytest.approx(1.0)
    # Node 1 neighbourhood {b, a, a} is less balanced -> below 1.
    assert 0.0 < entropy[1] < 1.0


def test_local_class_entropy_uniform_is_zero() -> None:
    G = _uniform_path()
    entropy = local_class_entropy(G, state_key="state")
    assert set(entropy.values()) == {0.0}


# ---------------------------------------------------------------------------
# label_propagation_residual
# ---------------------------------------------------------------------------


def test_label_propagation_residual_boundary_higher_than_interior() -> None:
    # Chain a-a-a-a-b: node 4 (lone b) is the boundary, node 0 is deep interior.
    G = nx.path_graph(5)
    for node in range(4):
        G.nodes[node]["state"] = "a"
    G.nodes[4]["state"] = "b"
    residual = label_propagation_residual(G, state_key="state")
    assert residual[4] > residual[0]
    assert all(math.isfinite(v) for v in residual.values())


def test_label_propagation_residual_single_class_is_zero() -> None:
    G = _uniform_path()
    residual = label_propagation_residual(G, state_key="state")
    assert set(residual.values()) == {0.0}


# ---------------------------------------------------------------------------
# graph_cut_contribution
# ---------------------------------------------------------------------------


def test_graph_cut_contribution_unweighted_matches_disagreement() -> None:
    G = _alternating_path()  # no distance attr -> distance 0 -> weight 1.
    cut = graph_cut_contribution(G, state_key="state")
    disagreement = local_disagreement(G, state_key="state")
    assert cut == pytest.approx(disagreement)


def test_graph_cut_contribution_distance_weighting() -> None:
    G = _alternating_path()
    G.edges[0, 1]["distance"] = 1.0  # weight 1/(1+1) = 0.5
    G.edges[1, 2]["distance"] = 0.0  # weight 1.0
    cut = graph_cut_contribution(G, state_key="state")
    assert cut[0] == pytest.approx(0.5)
    assert cut[1] == pytest.approx(1.5)
    assert cut[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# topology / feature / controls
# ---------------------------------------------------------------------------


def test_degree_and_distance_and_constant() -> None:
    G = _alternating_path()
    for u, v in G.edges():
        G.edges[u, v]["distance"] = 2.0
    degree = degree_score(G)
    assert degree[1] == pytest.approx(2.0)
    distance = mean_neighbour_distance(G)
    assert distance[0] == pytest.approx(2.0)
    constant = constant_baseline(G)
    assert set(constant.values()) == {1.0}


def test_feature_deviation_flags_outlier() -> None:
    G = nx.path_graph(3)
    for node, value in zip(G.nodes(), [0.0, 0.0, 3.0]):
        G.nodes[node]["val"] = value
    deviation = feature_deviation(G, value_key="val")
    assert deviation[2] == max(deviation.values())
    assert all(math.isfinite(v) for v in deviation.values())


def test_random_baseline_is_deterministic_and_seed_sensitive() -> None:
    G = _alternating_path()
    first = random_baseline(G, seed=7)
    second = random_baseline(G, seed=7)
    third = random_baseline(G, seed=8)
    assert first == second
    assert first != third
    assert set(first) == set(G.nodes())


# ---------------------------------------------------------------------------
# aggregator
# ---------------------------------------------------------------------------


def test_compute_all_baselines_keys_and_finite() -> None:
    G = _alternating_path()
    for u, v in G.edges():
        G.edges[u, v]["distance"] = 1.0
    for node in G.nodes():
        G.nodes[node]["phase"] = 0.0 if G.nodes[node]["state"] == "a" else math.pi
    maps = compute_all_baselines(G, state_key="state")
    expected = {
        "local_disagreement",
        "graph_total_variation",
        "local_class_entropy",
        "label_propagation_residual",
        "graph_cut_contribution",
        "mean_neighbour_distance",
        "degree",
        "constant",
        "random",
    }
    assert set(maps) == expected
    for score_map in maps.values():
        assert set(score_map) == set(G.nodes())
        assert all(math.isfinite(v) for v in score_map.values())


def test_compute_all_baselines_optional_feature() -> None:
    G = _alternating_path()
    for node in G.nodes():
        G.nodes[node]["val"] = float(node)
    maps = compute_all_baselines(G, state_key="state", feature_key="val")
    assert "feature_deviation" in maps


def test_compute_all_baselines_empty_graph() -> None:
    maps = compute_all_baselines(nx.Graph(), state_key="state")
    assert all(score_map == {} for score_map in maps.values())


def test_baseline_formulas_documented() -> None:
    # Every aggregated baseline (except the optional feature) is documented.
    for name in (
        "local_disagreement",
        "graph_total_variation",
        "local_class_entropy",
        "label_propagation_residual",
        "graph_cut_contribution",
        "mean_neighbour_distance",
        "degree",
        "constant",
        "random",
        "feature_deviation",
    ):
        assert name in BASELINE_FORMULAS
        assert BASELINE_FORMULAS[name]

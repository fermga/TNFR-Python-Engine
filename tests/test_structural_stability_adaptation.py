"""Tests for the structural-stability frequency-adaptation gate."""

from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import pytest

import tnfr.dynamics.adaptation as adaptation_module
from tnfr.constants import inject_defaults
from tnfr.dynamics import (
    adapt_vf_after_structural_stability,
    adapt_vf_by_coherence,
)


def _stable_pair() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("left", "right")
    graph.nodes["left"].update(
        {"νf": 0.2, "Si": 0.9, "ΔNFR": 0.0, "stable_count": 1}
    )
    graph.nodes["right"].update(
        {"νf": 1.0, "Si": 0.9, "ΔNFR": 0.0, "stable_count": 1}
    )
    inject_defaults(graph)
    graph.graph["VF_ADAPT_TAU"] = 2
    graph.graph["VF_ADAPT_MU"] = 0.5
    graph.graph["SELECTOR_THRESHOLDS"] = {"si_hi": 0.8}
    return graph


@pytest.mark.parametrize(
    "adapter",
    [adapt_vf_after_structural_stability, adapt_vf_by_coherence],
)
def test_canonical_gate_and_compatibility_alias_share_snapshot_semantics(
    adapter,
) -> None:
    graph = _stable_pair()

    adapter(graph)

    assert graph.nodes["left"]["νf"] == pytest.approx(0.6)
    assert graph.nodes["right"]["νf"] == pytest.approx(0.6)
    assert graph.nodes["left"]["stable_count"] == 2
    assert graph.nodes["right"]["stable_count"] == 2
    assert "C_steps" not in graph.graph
    assert "coherence" not in graph.graph


def test_parallel_proposals_preserve_snapshot_result() -> None:
    graph = _stable_pair()

    adapt_vf_after_structural_stability(graph, n_jobs=2)

    assert graph.nodes["left"]["νf"] == pytest.approx(0.6)
    assert graph.nodes["right"]["νf"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("VF_ADAPT_TAU", True),
        ("VF_ADAPT_TAU", 2.0),
        ("VF_ADAPT_TAU", 0),
        ("VF_ADAPT_MU", True),
        ("VF_ADAPT_MU", float("nan")),
        ("VF_ADAPT_MU", -0.1),
        ("VF_ADAPT_MU", 1.1),
        ("EPS_DNFR_STABLE", float("inf")),
        ("EPS_DNFR_STABLE", -0.1),
        ("VF_MIN", -0.1),
        ("VF_MAX", float("nan")),
        ("VF_MAX", 10**400),
    ],
)
def test_invalid_gate_parameters_fail_before_mutation(key: str, value) -> None:
    graph = _stable_pair()
    graph.graph[key] = value
    before = copy.deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(ValueError):
        adapt_vf_after_structural_stability(graph)

    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("si_hi", [True, -0.1, 1.1, float("nan")])
def test_si_threshold_requires_a_finite_unit_interval_value(si_hi) -> None:
    graph = _stable_pair()
    graph.graph["SELECTOR_THRESHOLDS"] = {"si_hi": si_hi}
    before = copy.deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(ValueError, match="si_hi"):
        adapt_vf_after_structural_stability(graph)

    assert dict(graph.nodes(data=True)) == before


def test_fallback_si_threshold_is_validated_before_state_changes() -> None:
    graph = _stable_pair()
    graph.graph["SELECTOR_THRESHOLDS"] = {}
    graph.graph["GLYPH_THRESHOLDS"] = {"hi": float("inf")}
    before = copy.deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(ValueError, match="si_hi"):
        adapt_vf_after_structural_stability(graph)

    assert dict(graph.nodes(data=True)) == before

@pytest.mark.parametrize("key", ["SELECTOR_THRESHOLDS", "GLYPH_THRESHOLDS"])
def test_threshold_containers_must_be_mappings(key: str) -> None:
    graph = _stable_pair()
    graph.graph[key] = []

    with pytest.raises(ValueError, match=key):
        adapt_vf_after_structural_stability(graph)


@pytest.mark.parametrize("n_jobs", [True, 0, -1, 1.5, float("nan")])
def test_n_jobs_requires_a_positive_integer(n_jobs) -> None:
    graph = _stable_pair()

    with pytest.raises(ValueError, match="n_jobs"):
        adapt_vf_after_structural_stability(graph, n_jobs=n_jobs)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("νf", float("inf")),
        ("νf", 10**400),
        ("νf", -0.1),
        ("Si", float("nan")),
        ("Si", -0.1),
        ("ΔNFR", complex(0.0, 1.0)),
        ("stable_count", True),
        ("stable_count", 1.0),
        ("stable_count", -1),
    ],
)
def test_invalid_node_state_is_rejected_without_partial_counter_updates(
    attribute: str,
    value,
) -> None:
    graph = _stable_pair()
    graph.nodes["right"][attribute] = value
    before = copy.deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(ValueError):
        adapt_vf_after_structural_stability(graph)

    assert dict(graph.nodes(data=True)) == before


def test_failed_frequency_commit_rolls_back_counters_values_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _stable_pair()
    before_nodes = copy.deepcopy(dict(graph.nodes(data=True)))
    before_graph = copy.deepcopy(graph.graph)
    real_set_vf = adaptation_module.set_vf
    call_count = 0

    def failing_setter(graph_arg, node, value):
        nonlocal call_count
        call_count += 1
        real_set_vf(graph_arg, node, value)
        if call_count == 2:
            raise RuntimeError("injected setter failure")

    monkeypatch.setattr(adaptation_module, "set_vf", failing_setter)

    with pytest.raises(RuntimeError, match="injected setter failure"):
        adapt_vf_after_structural_stability(graph)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


def test_reversed_frequency_bounds_are_rejected() -> None:
    graph = _stable_pair()
    graph.graph["VF_MIN"] = 1.0
    graph.graph["VF_MAX"] = 0.5

    with pytest.raises(ValueError, match="VF_MAX"):
        adapt_vf_after_structural_stability(graph)

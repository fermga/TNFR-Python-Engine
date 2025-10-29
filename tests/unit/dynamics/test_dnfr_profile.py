"""ΔNFR profile instrumentation tests."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.constants import get_aliases
from tnfr.dynamics import default_compute_delta_nfr

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")

_PROFILE_KEYS = (
    "dnfr_cache_rebuild",
    "dnfr_neighbor_accumulation",
    "dnfr_neighbor_means",
    "dnfr_gradient_assembly",
    "dnfr_inplace_write",
)


def _seed_graph(node_count: int = 48) -> nx.Graph:
    graph = nx.cycle_graph(node_count)
    for index, node in enumerate(graph.nodes):
        data = graph.nodes[node]
        data[ALIAS_THETA[0]] = float(index) / node_count
        data[ALIAS_EPI[0]] = 0.25 + 0.05 * (index % 5)
        data[ALIAS_VF[0]] = 0.4 + 0.03 * (index % 7)
    return graph


def _assert_profile_timings(profile: dict[str, float | str], expected_path: str) -> None:
    assert profile.get("dnfr_path") == expected_path
    for key in _PROFILE_KEYS:
        value = profile.get(key)
        assert isinstance(value, (int, float))
        assert value > 0.0, f"expected {key} to accumulate positive duration"


def test_default_compute_dnfr_records_profile_vectorized() -> None:
    pytest.importorskip("numpy")
    graph = _seed_graph()
    profile: dict[str, float | str] = {}
    default_compute_delta_nfr(graph, profile=profile)
    _assert_profile_timings(profile, "vectorized")


def test_default_compute_dnfr_records_profile_fallback() -> None:
    graph = _seed_graph()
    graph.graph["vectorized_dnfr"] = False
    profile: dict[str, float | str] = {}
    default_compute_delta_nfr(graph, profile=profile)
    _assert_profile_timings(profile, "fallback")

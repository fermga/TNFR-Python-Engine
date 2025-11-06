"""Tests for maintaining the runtime EPI history buffer."""

from __future__ import annotations

from collections import deque

from tnfr.constants import EPI_PRIMARY, get_param
from tnfr.dynamics.runtime import _update_epi_hist
from tnfr.structural import create_nfr


def _build_graph():
    """Create a small graph with multiple TNFR nodes for history tests."""

    G, _ = create_nfr("n0", epi=0.0, vf=1.0)
    for idx in range(1, 4):
        create_nfr(f"n{idx}", epi=0.0, vf=1.0, graph=G)
    return G


def _set_epi_snapshot(G, base: float) -> dict[str, float]:
    """Assign deterministic EPI values and return the expected snapshot map."""

    snapshot: dict[str, float] = {}
    for offset, node in enumerate(G.nodes):
        value = base + offset * 0.5
        G.nodes[node][EPI_PRIMARY] = value
        snapshot[node] = value
    return snapshot


def _expected_maxlen(G) -> int:
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau = max(tau_g, tau_l)
    return max(2 * tau + 5, 64)


def test_update_epi_hist_rebuilds_history() -> None:
    """Ensure ``_update_epi_hist`` normalises the history buffer."""

    # (a) Missing history should be initialised with the latest snapshot.
    absent = _build_graph()
    absent_snapshot = _set_epi_snapshot(absent, base=1.0)
    _update_epi_hist(absent)
    absent_hist = absent.graph["_epi_hist"]
    expected_maxlen = _expected_maxlen(absent)
    assert isinstance(absent_hist, deque)
    assert absent_hist.maxlen == expected_maxlen
    assert list(absent_hist) == [absent_snapshot]

    # (b) Lists are promoted to a deque while preserving recent entries.
    legacy_list = _build_graph()
    legacy_entries = [{"step": i} for i in range(3)]
    legacy_list.graph["_epi_hist"] = legacy_entries[-2:]
    list_snapshot = _set_epi_snapshot(legacy_list, base=2.0)
    _update_epi_hist(legacy_list)
    list_hist = legacy_list.graph["_epi_hist"]
    assert isinstance(list_hist, deque)
    assert list_hist.maxlen == expected_maxlen
    assert list(list_hist) == legacy_entries[-2:] + [list_snapshot]

    # (c) Deques with the wrong maxlen are rebuilt and truncated to the latest data.
    legacy_deque = _build_graph()
    deque_entries = [{"step": i} for i in range(70)]
    legacy_deque.graph["_epi_hist"] = deque(deque_entries, maxlen=70)
    deque_snapshot = _set_epi_snapshot(legacy_deque, base=3.0)
    _update_epi_hist(legacy_deque)
    rebuilt_hist = legacy_deque.graph["_epi_hist"]
    assert isinstance(rebuilt_hist, deque)
    assert rebuilt_hist.maxlen == expected_maxlen
    assert len(rebuilt_hist) == expected_maxlen
    assert list(rebuilt_hist)[:-1] == deque_entries[-(expected_maxlen - 1) :]
    assert rebuilt_hist[-1] == deque_snapshot

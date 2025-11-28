"""Stress test coverage for the diagnosis observers on large graphs."""

from __future__ import annotations

import math
import time

import networkx as nx
import pytest

pytest.importorskip("numpy")

from tnfr.alias import set_attr
from tnfr.constants import (
    VF_KEY,
    get_aliases,
    inject_defaults,
    normalise_state_token,
)
from tnfr.glyph_history import ensure_history
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.diagnosis import _diagnosis_step

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_SI = get_aliases("SI")
ALIAS_DNFR = get_aliases("DNFR")

pytestmark = [pytest.mark.slow, pytest.mark.stress]


def _build_large_graph(*, seed: int, nodes: int, probability: float) -> nx.Graph:
    """Return a reproducible graph with canonical TNFR attributes."""

    graph = nx.gnp_random_graph(nodes, probability, seed=seed)
    inject_defaults(graph)

    metrics_cfg = dict(graph.graph.get("METRICS", {}))
    metrics_cfg["verbosity"] = "debug"
    graph.graph["METRICS"] = metrics_cfg
    graph.graph.setdefault("DIAGNOSIS", {}).setdefault("history_key", "nodal_diag")
    graph.graph["RANDOM_SEED"] = seed

    twopi = 2.0 * math.pi
    for node, data in graph.nodes(data=True):
        base = seed + int(node)
        theta = ((base * 0.017) % twopi) - math.pi
        epi = math.sin(base * 0.031)
        vf = 0.2 + 0.6 * ((base % 19) / 18.0)
        si = 0.5 + 0.5 * math.cos(base * 0.023)
        dnfr = 0.1 * math.sin(base * 0.011)

        set_attr(data, ALIAS_THETA, theta)
        set_attr(data, ALIAS_EPI, epi)
        set_attr(data, ALIAS_VF, vf)
        set_attr(data, ALIAS_SI, max(0.0, min(1.0, si)))
        set_attr(data, ALIAS_DNFR, dnfr)

    return graph


@pytest.mark.timeout(30)
def test_diagnosis_step_large_graph_runtime_and_history() -> None:
    """The diagnosis step must remain performant and structurally coherent."""

    node_count = 320
    edge_probability = 0.18
    seed = 9021

    graph = _build_large_graph(seed=seed, nodes=node_count, probability=edge_probability)
    register_metrics_callbacks(graph)
    ensure_history(graph)

    start = time.perf_counter()
    _diagnosis_step(graph, n_jobs=1)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0

    history = ensure_history(graph)
    diag_history = history.get("nodal_diag")

    assert isinstance(diag_history, list)
    assert diag_history, "diagnosis history must contain at least one snapshot"

    latest_snapshot = diag_history[-1]
    assert isinstance(latest_snapshot, dict)
    assert len(latest_snapshot) == node_count

    canonical_states = {"stable", "transition", "dissonant"}

    for node, payload in latest_snapshot.items():
        assert node in graph.nodes
        assert isinstance(payload, dict)

        Si = float(payload["Si"])
        EPI = float(payload["EPI"])
        VF = float(payload[VF_KEY])
        dnfr_norm = float(payload["dnfr_norm"])
        R_local = float(payload["R_local"])
        state = str(payload["state"])

        assert 0.0 <= Si <= 1.0
        assert math.isfinite(EPI)
        assert 0.0 <= VF <= 1.0
        assert 0.0 <= dnfr_norm <= 1.0
        assert math.isfinite(R_local)
        assert normalise_state_token(state) in canonical_states

        symmetry = payload.get("symmetry")
        if symmetry is not None:
            assert 0.0 <= float(symmetry) <= 1.0

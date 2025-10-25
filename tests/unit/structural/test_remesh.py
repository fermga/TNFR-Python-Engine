"""Remeshing tests."""

from collections import deque
from types import ModuleType

import pytest

from tnfr.alias import set_attr
from tnfr.callback_utils import CallbackEvent, callback_manager
from tnfr.constants import get_aliases, get_param, inject_defaults
from tnfr.glyph_history import ensure_history
from tnfr.operators import apply_remesh_if_globally_stable
from tnfr.operators.remesh import (
    _snapshot_epi,
    _snapshot_topology,
    apply_network_remesh,
)


class _MissingNumberOfNodesGraph:
    def number_of_edges(self) -> int:
        return 0

    def degree(self):  # pragma: no cover - simple stub iterator
        return []


class _AttrErrorNumberOfNodesGraph:
    def number_of_nodes(self) -> int:
        raise AttributeError("number_of_nodes not available")

    def number_of_edges(self) -> int:
        return 0

    def degree(self):  # pragma: no cover - simple stub iterator
        return []


class _NonNumericDegreesGraph:
    def number_of_nodes(self) -> int:
        return 2

    def number_of_edges(self) -> int:
        return 1

    def degree(self):  # pragma: no cover - simple stub iterator
        return [(0, 1 + 2j), (1, 1 - 2j)]

DEPRECATED_REMESH_KEYWORD = "legacy_stable_window"
DEPRECATED_REMESH_CONFIG = "REMESH_COOLDOWN_LEGACY"


def _prepare_graph_for_remesh(graph_canon, stable_steps: int = 3):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    G.graph["REMESH_REQUIRE_STABILITY"] = False

    hist = G.graph.setdefault("history", {})
    hist["stable_frac"] = [1.0] * stable_steps

    tau = G.graph["REMESH_TAU_GLOBAL"]
    maxlen = max(2 * tau + 5, 64)
    G.graph["_epi_hist"] = deque([{0: 0.0} for _ in range(tau + 1)], maxlen=maxlen)

    return G, hist


def test_get_networkx_modules_raises_without_networkx(monkeypatch):
    import tnfr.operators.remesh as remesh

    with monkeypatch.context() as mp:
        original = remesh.cached_import

        def fake_cached_import(name: str, *args, **kwargs):
            if name == "networkx":
                return None
            return original(name, *args, **kwargs)

        mp.setattr(remesh, "cached_import", fake_cached_import)

        with pytest.raises(
            ImportError,
            match="networkx is required for network operators; install 'networkx'",
        ):
            remesh._get_networkx_modules()


def test_get_networkx_modules_raises_without_networkx_community(monkeypatch):
    import tnfr.operators.remesh as remesh

    fake_networkx = ModuleType("networkx_stub")
    setattr(fake_networkx, "NetworkXError", Exception)

    with monkeypatch.context() as mp:
        original = remesh.cached_import

        def fake_cached_import(name: str, *args, **kwargs):
            if name == "networkx":
                return fake_networkx
            if name == "networkx.algorithms":
                return None
            return original(name, *args, **kwargs)

        mp.setattr(remesh, "cached_import", fake_cached_import)

        with pytest.raises(
            ImportError,
            match="networkx.algorithms.community is required for community-based",
        ):
            remesh._get_networkx_modules()


@pytest.mark.parametrize(
    "graph_cls",
    [
        _MissingNumberOfNodesGraph,
        _AttrErrorNumberOfNodesGraph,
        _NonNumericDegreesGraph,
    ],
    ids=[
        "missing_number_of_nodes",
        "attrerror_number_of_nodes",
        "non_numeric_degrees",
    ],
)
def test_snapshot_topology_gracefully_handles_invalid_graphs(graph_cls):
    nx = pytest.importorskip("networkx")
    bad_graph = graph_cls()

    assert _snapshot_topology(bad_graph, nx) is None


def test_snapshot_epi_returns_checksum_for_non_numeric_node_values(graph_canon):
    G = graph_canon()
    G.add_nodes_from(range(3))

    alias_epi = get_aliases("EPI")
    for node in G.nodes:
        G.nodes[node][alias_epi[0]] = "non-float-epi"

    mean_val, checksum = _snapshot_epi(G)

    assert mean_val == 0.0
    assert isinstance(checksum, str) and len(checksum) == 12


def test_apply_remesh_uses_custom_parameter(graph_canon):
    G, hist = _prepare_graph_for_remesh(graph_canon)

    # Without a custom parameter it should not trigger
    apply_remesh_if_globally_stable(G)
    assert "_last_remesh_step" not in G.graph

    # With the custom parameter it triggers after 3 stable steps
    apply_remesh_if_globally_stable(G, stable_step_window=3)
    assert G.graph["_last_remesh_step"] == len(hist["stable_frac"])


def test_apply_remesh_legacy_keyword_raises_typeerror(graph_canon):
    G, _ = _prepare_graph_for_remesh(graph_canon)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        apply_remesh_if_globally_stable(
            G,
            **{DEPRECATED_REMESH_KEYWORD: 3},
        )

    assert "_last_remesh_step" not in G.graph


def test_remesh_alpha_hard_ignores_glyph_factor(graph_canon):
    G, _ = _prepare_graph_for_remesh(graph_canon)
    G.graph["REMESH_ALPHA"] = 0.7
    G.graph["REMESH_ALPHA_HARD"] = True
    G.graph["GLYPH_FACTORS"]["REMESH_alpha"] = 0.1
    apply_remesh_if_globally_stable(G, stable_step_window=3)
    meta = G.graph.get("_REMESH_META", {})
    assert meta.get("alpha") == 0.7
    assert G.graph.get("_REMESH_ALPHA_SRC") == "REMESH_ALPHA"


def test_apply_network_remesh_triggers_callback(graph_canon):
    pytest.importorskip("networkx")

    G = graph_canon()
    nodes = [0, 1, 2]
    G.add_nodes_from(nodes)
    inject_defaults(G)

    alias_epi = get_aliases("EPI")
    for idx, node in enumerate(nodes):
        set_attr(G.nodes[node], alias_epi, float(idx))

    hist = ensure_history(G)
    hist.setdefault("C_steps", []).extend([0.0, 1.0])

    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau_req = max(tau_g, tau_l)

    snapshots = []
    for offset in range(tau_req + 1):
        snapshots.append({node: float(idx + offset) for idx, node in enumerate(nodes)})

    maxlen = max(tau_req + 5, tau_req + 1)
    G.graph["_epi_hist"] = deque(snapshots, maxlen=maxlen)

    triggered: list[dict] = []

    def on_remesh(graph, ctx):
        triggered.append(ctx)
        assert graph is G

    callback_manager.register_callback(G, CallbackEvent.ON_REMESH, on_remesh)

    apply_network_remesh(G)

    assert triggered, "The ON_REMESH callback should run"
    ctx = triggered[-1]
    assert ctx["tau_global"] == tau_g
    assert ctx["tau_local"] == tau_l
    assert "alpha" in ctx


def test_injected_defaults_include_cooldown_window_only(graph_canon):
    G, _ = _prepare_graph_for_remesh(graph_canon)

    assert "REMESH_COOLDOWN_WINDOW" in G.graph
    assert DEPRECATED_REMESH_CONFIG not in G.graph


def test_configured_cooldown_window_is_respected(graph_canon):
    G, hist = _prepare_graph_for_remesh(graph_canon)
    preferred_value = 1
    G.graph["REMESH_COOLDOWN_WINDOW"] = preferred_value

    apply_remesh_if_globally_stable(G, stable_step_window=3)

    hist["stable_frac"].append(1.0)
    apply_remesh_if_globally_stable(G, stable_step_window=3)

    events = ensure_history(G).get("remesh_events", [])
    assert len(events) == 2


@pytest.mark.parametrize(
    ("metric_sequences", "should_remesh"),
    [
        pytest.param(
            {
                "phase_sync": [0.83, 0.84, 0.849],
                "glyph_load_disr": deque([0.36, 0.355, 0.36], maxlen=3),
                "sense_sigma_mag": [0.49, 0.485, 0.49],
                "kuramoto_R": deque([0.79, 0.79, 0.795], maxlen=3),
                "Si_hi_frac": [0.49, 0.495, 0.49],
            },
            False,
            id="requires_stability_blocks_remesh",
        ),
        pytest.param(
            {
                "phase_sync": [0.86, 0.87, 0.88],
                "glyph_load_disr": deque([0.34, 0.335, 0.33], maxlen=3),
                "sense_sigma_mag": [0.5, 0.52, 0.53],
                "kuramoto_R": deque([0.8, 0.81, 0.82], maxlen=3),
                "Si_hi_frac": [0.51, 0.52, 0.53],
            },
            True,
            id="requires_stability_allows_remesh",
        ),
    ],
)
def test_apply_remesh_respects_stability_gating(graph_canon, metric_sequences, should_remesh):
    pytest.importorskip("networkx")

    G, hist = _prepare_graph_for_remesh(graph_canon)

    # Ensure a deterministic environment with ready-to-trigger stability window.
    hist["stable_frac"] = [1.0, 1.0, 1.0]
    hist.pop("remesh_events", None)
    for key, seq in metric_sequences.items():
        hist[key] = list(seq)

    G.graph.update(
        {
            "REMESH_COOLDOWN_WINDOW": 0,
            "REMESH_COOLDOWN_TS": 0.0,
            "_t": 100.0,
            "_last_remesh_ts": -1e6,
        }
    )
    G.graph["REMESH_REQUIRE_STABILITY"] = True

    apply_remesh_if_globally_stable(G, stable_step_window=3)

    if should_remesh:
        step_count = len(hist["stable_frac"])
        assert G.graph["_last_remesh_step"] == step_count
        assert G.graph["_last_remesh_ts"] == pytest.approx(G.graph["_t"])

        meta = G.graph.get("_REMESH_META")
        assert meta, "Remesh metadata should be recorded when gating passes"
        assert meta["tau_global"] == int(get_param(G, "REMESH_TAU_GLOBAL"))
        assert meta["tau_local"] == int(get_param(G, "REMESH_TAU_LOCAL"))
        assert meta["phase_sync_last"] == pytest.approx(metric_sequences["phase_sync"][-1])
        assert meta["glyph_disr_last"] == pytest.approx(
            metric_sequences["glyph_load_disr"][-1]
        )

        events = hist.get("remesh_events")
        assert events and events[-1]["tau_global"] == meta["tau_global"]
        assert events[-1]["epi_checksum_after"] == meta["epi_checksum_after"]
    else:
        assert "_last_remesh_step" not in G.graph
        assert "_REMESH_META" not in G.graph
        events = hist.get("remesh_events")
        assert not events

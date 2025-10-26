import networkx as nx

from tnfr.constants import METRIC_DEFAULTS
from tnfr.ontosim import prepare_network


def test_prepare_network_initializes_attrs_by_default():
    G = nx.path_graph(3)
    prepare_network(G)
    assert all("theta" in d for _, d in G.nodes(data=True))


def test_prepare_network_allows_disabling_init_attrs():
    G = nx.path_graph(3)
    prepare_network(G, init_attrs=False)
    assert all("theta" not in d for _, d in G.nodes(data=True))


def test_prepare_network_records_callback_error_when_observer_missing(monkeypatch):
    G = nx.path_graph(2)
    G.graph["ATTACH_STD_OBSERVER"] = True
    monkeypatch.setattr("tnfr.ontosim.cached_import", lambda *_, **__: None)

    prepare_network(G)

    assert G.graph["_callback_errors"] == [
        {"event": "attach_std_observer", "error": "ImportError"}
    ]


def test_prepare_network_attaches_standard_observer(monkeypatch):
    G = nx.path_graph(2)
    G.graph["ATTACH_STD_OBSERVER"] = True

    calls = []

    def _stub_attach_observer(graph):
        calls.append(graph)

    monkeypatch.setattr("tnfr.ontosim.cached_import", lambda *_, **__: _stub_attach_observer)

    returned = prepare_network(G)

    assert returned is G
    assert calls == [G]
    assert G.graph.get("_callback_errors", []) == []


def test_prepare_network_respects_override_defaults_flag():
    G = nx.path_graph(2)
    G.graph["PHASE_HISTORY_MAXLEN"] = 123
    G.graph["REMESH_TAU_GLOBAL"] = 42

    prepare_network(G, override_defaults=False, REMESH_TAU_GLOBAL=5)

    assert G.graph["PHASE_HISTORY_MAXLEN"] == 123
    assert G.graph["REMESH_TAU_GLOBAL"] == 5

    G_override = nx.path_graph(2)
    G_override.graph["PHASE_HISTORY_MAXLEN"] = 123
    G_override.graph["REMESH_TAU_GLOBAL"] = 42

    prepare_network(G_override, override_defaults=True, REMESH_TAU_GLOBAL=5)

    assert G_override.graph["PHASE_HISTORY_MAXLEN"] == METRIC_DEFAULTS["PHASE_HISTORY_MAXLEN"]
    assert G_override.graph["REMESH_TAU_GLOBAL"] == 5


def test_prepare_network_reuses_existing_history_state():
    G = nx.path_graph(3)
    prepare_network(G)
    history = G.graph["history"]
    history["custom_metric"] = ["alpha", "beta"]
    history["phase_state"].extend([1.0, 2.0])
    history_state_id_before = id(history["phase_state"])

    returned = prepare_network(G)

    assert returned is G
    assert G.graph["history"] is history
    assert history["custom_metric"] == ["alpha", "beta"]
    assert "phase_state" in history
    assert id(history["phase_state"]) == history_state_id_before
    assert list(history["phase_state"])[-2:] == [1.0, 2.0]

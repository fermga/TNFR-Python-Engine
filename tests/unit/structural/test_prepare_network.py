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

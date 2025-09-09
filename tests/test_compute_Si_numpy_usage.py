from tnfr.constants import ALIAS_THETA, ALIAS_VF, ALIAS_DNFR
from tnfr.metrics_utils import compute_Si
from tnfr.alias import set_attr


def test_compute_Si_calls_get_numpy_once_and_propagates(monkeypatch, graph_canon):
    calls = 0

    class DummyNP:
        def fromiter(self, iterable, dtype=float, count=-1):
            return list(iterable)

    sentinel = DummyNP()

    def fake_get_numpy():
        nonlocal calls
        calls += 1
        return sentinel

    captured = []

    def fake_neighbor_phase_mean_list(
        neigh, cos_th, sin_th, np=None, fallback=0.0
    ):
        captured.append(np)
        return 0.0

    monkeypatch.setattr("tnfr.metrics_utils.get_numpy", fake_get_numpy)
    monkeypatch.setattr(
        "tnfr.metrics_utils.neighbor_phase_mean_list",
        fake_neighbor_phase_mean_list,
    )

    G = graph_canon()
    G.add_edge(1, 2)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)
        set_attr(G.nodes[n], ALIAS_VF, 0.0)
        set_attr(G.nodes[n], ALIAS_DNFR, 0.0)

    compute_Si(G, inplace=False)

    assert calls == 1
    assert captured == [sentinel] * G.number_of_nodes()

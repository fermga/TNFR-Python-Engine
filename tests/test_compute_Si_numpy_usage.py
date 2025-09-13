from tnfr.constants import get_aliases
from tnfr.metrics_utils import compute_Si
from tnfr.alias import set_attr
from tnfr.import_utils import clear_optional_import_cache

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_compute_Si_uses_module_numpy_and_propagates(monkeypatch, graph_canon):
    class DummyNP:
        def fromiter(self, iterable, dtype=float, count=-1):
            return list(iterable)

    sentinel = DummyNP()

    captured = []

    def fake_neighbor_phase_mean_list(
        neigh, cos_th, sin_th, np=None, fallback=0.0
    ):
        captured.append(np)
        return 0.0

    clear_optional_import_cache()
    monkeypatch.setattr("tnfr.metrics_utils.get_numpy", lambda: sentinel)
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

    assert captured == [sentinel] * G.number_of_nodes()

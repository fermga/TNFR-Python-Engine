import math

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.sense_index import compute_Si
import tnfr.utils.init as utils_init

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_compute_Si_uses_module_numpy_and_propagates(monkeypatch, graph_canon):
    class DummyNP:
        def fromiter(self, iterable, dtype=float, count=-1):
            _ = dtype
            return list(iterable)

        def cos(self, arr):
            return [math.cos(x) for x in arr]

        def sin(self, arr):
            return [math.sin(x) for x in arr]

    sentinel = DummyNP()

    captured = []

    def fake_neighbor_phase_mean_list(
        _neigh, cos_th, sin_th, np=None, fallback=0.0
    ):
        captured.append(np)
        return 0.0

    monkeypatch.setattr(
        utils_init,
        "cached_import",
        lambda module, attr=None, **kwargs: sentinel if module == "numpy" else None,
    )
    monkeypatch.setattr(
        "tnfr.metrics.sense_index.neighbor_phase_mean_list",
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

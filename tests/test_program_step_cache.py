import threading
import networkx as nx
import tnfr.dynamics as dyn
import tnfr.program as program
from tnfr.program import _advance


def test_advance_caches_step(monkeypatch, graph_canon):
    G = graph_canon()
    G.add_node(0)
    calls = []

    def first_step(G):
        calls.append(1)

    def second_step(G):
        calls.append(2)

    monkeypatch.setattr(dyn, "step", first_step)
    program._load_step_fn.cache_clear()
    _advance(G)
    monkeypatch.setattr(dyn, "step", second_step)
    _advance(G)
    assert calls == [1, 1]
    program._load_step_fn.cache_clear()


def test_advance_thread_safe(monkeypatch, graph_canon):
    G = graph_canon()
    G.add_node(0)
    calls = []

    def first_step(G):
        calls.append(1)

    def second_step(G):
        calls.append(2)

    monkeypatch.setattr(dyn, "step", first_step)
    program._load_step_fn.cache_clear()

    barrier = threading.Barrier(5)

    def worker():
        barrier.wait()
        _advance(G)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    monkeypatch.setattr(dyn, "step", second_step)
    _advance(G)

    assert calls == [1] * 6
    program._load_step_fn.cache_clear()

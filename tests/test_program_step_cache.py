import threading
import networkx as nx
import tnfr.dynamics as dyn
from tnfr.program import _advance


def test_advance_caches_step(monkeypatch):
    G = nx.Graph()
    G.add_node(0)
    calls = []

    def first_step(G):
        calls.append(1)

    def second_step(G):
        calls.append(2)

    monkeypatch.setattr(dyn, "step", first_step)
    monkeypatch.setattr("tnfr.program._StepFnCache.step_fn", None)
    _advance(G)
    monkeypatch.setattr(dyn, "step", second_step)
    _advance(G)
    assert calls == [1, 1]
    monkeypatch.setattr("tnfr.program._StepFnCache.step_fn", None)


def test_advance_thread_safe(monkeypatch):
    G = nx.Graph()
    G.add_node(0)
    calls = []

    def first_step(G):
        calls.append(1)

    def second_step(G):
        calls.append(2)

    monkeypatch.setattr(dyn, "step", first_step)
    monkeypatch.setattr("tnfr.program._StepFnCache.step_fn", None)

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
    monkeypatch.setattr("tnfr.program._StepFnCache.step_fn", None)

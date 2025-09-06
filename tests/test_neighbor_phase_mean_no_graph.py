import pytest
import math
import time

from tnfr.helpers import neighbor_phase_mean


class DummyNeighbor:
    pass


class DummyNode:
    def __init__(self):
        self.theta = 0.5
        self._neigh = [DummyNeighbor()]

    def neighbors(self):
        return self._neigh


def test_neighbor_phase_mean_handles_missing_G():
    node = DummyNode()
    assert neighbor_phase_mean(node) == pytest.approx(node.theta)


@pytest.mark.slow
def test_neighbor_phase_mean_no_graph_performance():
    neigh = DummyNeighbor()
    node = DummyNode()
    node._neigh = [neigh] * 1000

    def naive(n: DummyNode):
        x = y = 0.0
        count = 0
        for v in n.neighbors():
            th = getattr(v, "theta", None)
            if th is None:
                continue
            x += math.cos(th)
            y += math.sin(th)
            count += 1
        if count == 0:
            return n.theta
        return math.atan2(y, x)

    start = time.perf_counter()
    for _ in range(50):
        neighbor_phase_mean(node)
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(50):
        naive(node)
    t_naive = time.perf_counter() - start

    assert t_opt <= t_naive

import pytest

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

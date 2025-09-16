import networkx as nx  # type: ignore[import-untyped]
import pytest

from tnfr.helpers.numeric import similarity_abs
from tnfr.observers import phase_sync


def test_phase_sync_statistics_fallback(monkeypatch):
    monkeypatch.setattr("tnfr.observers.get_numpy", lambda: None)

    G = nx.Graph()
    G.add_nodes_from(
        (
            (0, {"theta": 0.0}),
            (1, {"theta": 0.1}),
            (2, {"theta": -0.1}),
        )
    )

    # 0 variance would yield 1; this setup triggers the statistics branch.
    diffs = [0.0, 0.1, -0.1]
    expected_var = sum(d * d for d in diffs) / len(diffs)
    assert phase_sync(G, R=1.0, psi=0.0) == pytest.approx(1.0 / (1.0 + expected_var))


@pytest.mark.parametrize(
    "a, b, lo, hi, expected",
    [
        (0.5, 0.5, 0.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 1.0, 0.0),
        (1.0, 1.5, 1.0, 3.0, 0.75),
    ],
)
def test_similarity_abs_scales_difference(a, b, lo, hi, expected):
    assert similarity_abs(a, b, lo, hi) == pytest.approx(expected)


def test_similarity_abs_degenerate_range_returns_full_similarity():
    assert similarity_abs(1.0, 2.0, 1.0, 1.0) == pytest.approx(1.0)

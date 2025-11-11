import math

import networkx as nx  # type: ignore[import-untyped]
import pytest

from tnfr.utils import angle_diff, angle_diff_array, similarity_abs
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


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (math.pi, 0.0, -math.pi),
        (-math.pi, 0.0, -math.pi),
        (math.pi - 1e-9, 0.0, math.pi - 1e-9),
        (math.pi + 1e-9, 0.0, -math.pi + 1e-9),
        (-math.pi + 1e-9, 0.0, -math.pi + 1e-9),
        (-math.pi - 1e-9, 0.0, math.pi - 1e-9),
    ],
)
def test_angle_diff_wraps_boundary_extremes(a, b, expected):
    assert angle_diff(a, b) == pytest.approx(expected)


def test_angle_diff_array_matches_scalar():
    np = pytest.importorskip("numpy")

    angles_a = np.array([0.0, math.pi - 1e-9, -math.pi + 1e-6, math.tau + 0.5])
    angles_b = np.array([math.pi, -math.pi, 0.0, 0.25])
    out = angle_diff_array(angles_a, angles_b, np=np)
    expected = np.array(
        [angle_diff(a, b) for a, b in zip(angles_a, angles_b)], dtype=float
    )
    assert np.allclose(out, expected)


def test_angle_diff_array_preserves_masked_entries():
    np = pytest.importorskip("numpy")

    angles_a = np.array([0.0, math.pi / 2, -math.pi / 2])
    angles_b = np.array([math.pi, 0.0, 0.0])
    mask = np.array([True, False, True])
    out = np.full_like(angles_a, fill_value=42.0)
    angle_diff_array(angles_a, angles_b, np=np, out=out, where=mask)

    expected = np.array(
        [
            angle_diff(angles_a[0], angles_b[0]),
            42.0,
            angle_diff(angles_a[2], angles_b[2]),
        ]
    )
    assert np.allclose(out, expected)

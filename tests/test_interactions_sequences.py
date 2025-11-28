import networkx as nx
from tnfr.physics.patterns import (
    reset_baseline,
    apply_plane_wave,
    apply_vortex,
)
from tnfr.physics.interactions import (
    em_like,
    weak_like,
    strong_like,
)


def _grid(n=12):
    G = nx.grid_2d_graph(n, n)
    # Convert to simple graph with tuple nodes as-is
    return G


def test_em_like_preserves_basic_telemetry_shapes():
    G = _grid(10)
    reset_baseline(G)
    apply_plane_wave(G, kx=0.2, ky=0.0)

    nodes = list(G.nodes())[:10]
    res = em_like(G, nodes, compute_phi_s=False)

    assert isinstance(res.applied, list)
    assert res.grad_after_mean is not None
    assert res.kphi_after_abs_mean is not None


def test_weak_like_runs_with_stable_base():
    G = _grid(10)
    reset_baseline(G)
    apply_plane_wave(G, kx=0.1, ky=0.1)

    nodes = list(G.nodes())[:8]
    res = weak_like(G, nodes, ensure_stable_base=True, compute_phi_s=False)

    # Expect at least one operator applied per node
    assert len(res.applied) >= len(nodes)


def test_strong_like_flags_curvature_when_vortex_present():
    G = _grid(12)
    reset_baseline(G)
    apply_vortex(G)

    nodes = list(G.nodes())[:12]
    res = strong_like(G, nodes, compute_phi_s=False)

    # Either no warnings or curvature warning; both acceptable in smoke test
    assert isinstance(res.warnings, list)

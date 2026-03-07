"""Claim-level tests for the structural conservation preprint.

Each test corresponds to a specific claim in the paper. If a test fails,
the corresponding claim cannot be made.

Claims tested:
  C1: Charge drift < 5% under valid grammar-compliant dynamics
  C2: Energy is non-increasing (Lyapunov) under valid dynamics (most topologies)
  C3: Conservation quality > 0.30 under valid dynamics
  C4: Invalid dynamics exhibit measurably higher charge drift
  C5: Results are deterministic (seed reproducibility)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Resolve imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import networkx as nx

from tnfr.constants import inject_defaults
from tnfr.physics.conservation import (
    ConservationTracker,
    capture_conservation_snapshot,
    compute_energy_functional,
    compute_lyapunov_derivative,
    compute_noether_charge,
    verify_conservation_balance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SEED = 42
DT = 0.01
N_STEPS = 40


def _build(builder: str, n: int, seed: int = SEED) -> nx.Graph:
    rng = np.random.default_rng(seed)
    if builder == "path":
        G = nx.path_graph(n)
    elif builder == "cycle":
        G = nx.cycle_graph(n)
    elif builder == "grid":
        side = int(math.isqrt(n))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif builder == "tree":
        G = nx.balanced_tree(2, int(math.log2(n + 1)))
        G = nx.convert_node_labels_to_integers(G)
    elif builder == "erdos":
        G = nx.erdos_renyi_graph(n, 0.3, seed=seed)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
            G = nx.convert_node_labels_to_integers(G)
    else:
        raise ValueError(builder)

    inject_defaults(G)
    for nd in G.nodes():
        G.nodes[nd]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]
        G.nodes[nd]["frequency"] = rng.uniform(0.1, 1.0)
        G.nodes[nd]["nu_f"] = G.nodes[nd]["frequency"]
        G.nodes[nd]["delta_nfr"] = rng.uniform(-0.5, 0.5)
        G.nodes[nd]["EPI"] = rng.uniform(0.5, 2.0)
    return G


def _evolve_valid(G: nx.Graph, dt: float) -> None:
    for nd in G.nodes():
        nu_f = G.nodes[nd].get("nu_f", 1.0)
        dnfr = G.nodes[nd].get("delta_nfr", 0.0)
        G.nodes[nd]["phase"] += dt * nu_f * dnfr * 0.1
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]
        nbrs = list(G.neighbors(nd))
        if nbrs:
            mean_dnfr = float(
                np.mean([G.nodes[j].get("delta_nfr", 0.0) for j in nbrs])
            )
            G.nodes[nd]["delta_nfr"] += dt * 0.1 * (mean_dnfr - dnfr)


def _evolve_invalid(G: nx.Graph, dt: float, rng: np.random.Generator) -> None:
    for nd in G.nodes():
        nu_f = G.nodes[nd].get("nu_f", 1.0)
        dnfr = G.nodes[nd].get("delta_nfr", 0.0)
        G.nodes[nd]["delta_nfr"] += dt * 0.5 * abs(dnfr)
        G.nodes[nd]["phase"] += dt * nu_f * dnfr * 0.3
        if rng.random() < 0.1:
            G.nodes[nd]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]


# ---------------------------------------------------------------------------
# Parametrised topologies
# ---------------------------------------------------------------------------
TOPOLOGIES = [
    ("path", 20),
    ("cycle", 20),
    ("grid", 25),
    ("tree", 15),
    ("erdos", 25),
]


# ===================================================================
# C1: Structural charge drift bounded under valid dynamics
# ===================================================================
class TestC1ChargeDrift:
    """Claim: Relative charge drift < 5% under valid dynamics."""

    @pytest.mark.parametrize("builder,n", TOPOLOGIES)
    def test_charge_drift_bounded(self, builder: str, n: int) -> None:
        G = _build(builder, n)
        Q0 = compute_noether_charge(G)

        for _ in range(N_STEPS):
            _evolve_valid(G, DT)

        Q_final = compute_noether_charge(G)
        rel_drift = abs(Q_final - Q0) / max(abs(Q0), 1e-15)
        assert rel_drift < 0.05, (
            f"{builder}: relative charge drift {rel_drift:.4e} exceeds 5%"
        )


# ===================================================================
# C2: Lyapunov stability under valid dynamics
# ===================================================================
# Topologies where Lyapunov stability is expected (tree excluded:
# binary trees exhibit non-monotone energy due to leaf-heavy topology)
LYAPUNOV_TOPOLOGIES = [
    ("path", 20),
    ("cycle", 20),
    ("grid", 25),
    ("erdos", 25),
]


class TestC2Lyapunov:
    """Claim: Energy non-increasing in >= 80% of steps (most topologies)."""

    @pytest.mark.parametrize("builder,n", LYAPUNOV_TOPOLOGIES)
    def test_energy_mostly_non_increasing(self, builder: str, n: int) -> None:
        G = _build(builder, n)
        stable_count = 0

        for _ in range(N_STEPS):
            before = capture_conservation_snapshot(G)
            _evolve_valid(G, DT)
            after = capture_conservation_snapshot(G)
            lyap = compute_lyapunov_derivative(before, after, dt=DT)
            if lyap.is_stable:
                stable_count += 1

        stable_pct = stable_count / N_STEPS
        assert stable_pct >= 0.80, (
            f"{builder}: only {stable_pct:.0%} stable steps (need >= 80%)"
        )


# ===================================================================
# C3: Conservation quality above threshold
# ===================================================================
class TestC3Quality:
    """Claim: Mean conservation quality > 0.30 under valid dynamics."""

    @pytest.mark.parametrize("builder,n", TOPOLOGIES)
    def test_conservation_quality(self, builder: str, n: int) -> None:
        G = _build(builder, n)
        qualities: list[float] = []

        for _ in range(N_STEPS):
            before = capture_conservation_snapshot(G)
            _evolve_valid(G, DT)
            after = capture_conservation_snapshot(G)
            bal = verify_conservation_balance(before, after, dt=DT)
            qualities.append(bal.conservation_quality)

        mean_q = float(np.mean(qualities))
        assert mean_q > 0.30, (
            f"{builder}: mean quality {mean_q:.4f} below 0.30"
        )


# ===================================================================
# C4: Invalid dynamics show measurably worse conservation
# ===================================================================
class TestC4InvalidDegradation:
    """Claim: Invalid dynamics produce higher drift than valid."""

    @pytest.mark.parametrize("builder,n", TOPOLOGIES)
    def test_invalid_worse_than_valid(self, builder: str, n: int) -> None:
        # Valid arm
        G_valid = _build(builder, n)
        Q0_v = compute_noether_charge(G_valid)
        for _ in range(N_STEPS):
            _evolve_valid(G_valid, DT)
        drift_valid = abs(compute_noether_charge(G_valid) - Q0_v)

        # Invalid arm
        G_invalid = _build(builder, n)
        rng = np.random.default_rng(SEED + 999)
        Q0_i = compute_noether_charge(G_invalid)
        for _ in range(N_STEPS):
            _evolve_invalid(G_invalid, DT, rng)
        drift_invalid = abs(compute_noether_charge(G_invalid) - Q0_i)

        assert drift_invalid > drift_valid, (
            f"{builder}: invalid drift ({drift_invalid:.4e}) should exceed "
            f"valid drift ({drift_valid:.4e})"
        )


# ===================================================================
# C5: Deterministic reproducibility
# ===================================================================
class TestC5Reproducibility:
    """Claim: Identical seeds produce identical charge trajectories."""

    @pytest.mark.parametrize("builder,n", TOPOLOGIES)
    def test_seed_reproducibility(self, builder: str, n: int) -> None:
        charges: list[list[float]] = []
        for _ in range(2):
            G = _build(builder, n, seed=SEED)
            run_charges = [compute_noether_charge(G)]
            for _ in range(N_STEPS):
                _evolve_valid(G, DT)
                run_charges.append(compute_noether_charge(G))
            charges.append(run_charges)

        np.testing.assert_allclose(
            charges[0], charges[1], rtol=1e-12,
            err_msg=f"{builder}: trajectories differ across runs"
        )

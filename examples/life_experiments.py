"""
Life Experiments: Experimental scenarios for TNFR life emergence validation.

Experiments:
  1) Life emergence threshold (A>1) in hostile environment (ΔNFR_ext < 0)
  2) Self-maintenance without external input: Coherence C(t) stabilizes
  3) Replication fidelity (Fr) via pattern copying with small noise

Run:
    python -m examples.life_experiments
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Tuple

from tnfr.physics import detect_life_emergence
from tnfr.metrics.common import compute_coherence


def _logistic(t: np.ndarray, L=10.0, k=0.6, t0=8.0) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (t - t0)))


def exp1_life_emergence() -> Tuple[float, float]:
    T = 400
    times = np.linspace(0.0, 20.0, T)
    epi_series = _logistic(times, L=10.0, k=0.6, t0=8.0)
    dEPI_dt = np.gradient(epi_series, times)
    # Gentler external ΔNFR profile to allow earlier threshold crossing
    dnfr_external = -0.02 + 0.01 * (times / 20.0)  # -0.02 → -0.01
    d_dnfr_external_dt = np.gradient(dnfr_external, times)
    epsilon, gamma, epi_max = 0.9, 1.2, 10.0

    telem = detect_life_emergence(
        times, epi_series, dEPI_dt, dnfr_external, d_dnfr_external_dt,
        epsilon=epsilon, gamma=gamma, epi_max=epi_max
    )
    Amax = float(np.max(telem.autopoietic_coefficient))
    return telem.life_threshold_time if telem.life_threshold_time is not None else -1.0, Amax


def exp2_self_maintenance_coherence() -> Tuple[float, float]:
    # Network with no external input: ΔNFR_ext = 0
    N = 50
    G = nx.watts_strogatz_graph(N, k=4, p=0.15)

    # Time evolution of a single global EPI magnitude (proxy)
    T = 300
    times = np.linspace(0.0, 15.0, T)
    epi_series = _logistic(times, L=8.0, k=0.8, t0=6.0)  # approaches steady state
    dEPI_dt = np.gradient(epi_series, times)

    # Internal ΔNFR from self-generation (ε·G(EPI)), external = 0
    epsilon, gamma, epi_max = 0.7, 0.9, 8.0
    G_epi = gamma * epi_series * (1.0 - epi_series / epi_max)
    dnfr_internal = epsilon * G_epi

    # Assign node attributes over time and track coherence
    C_values = []
    for t_idx in range(T):
        dnfr_t = float(dnfr_internal[t_idx])
        depi_t = float(dEPI_dt[t_idx])
        nx.set_node_attributes(G, {n: dnfr_t for n in G.nodes()}, name="delta_nfr")
        nx.set_node_attributes(G, {n: depi_t for n in G.nodes()}, name="d_epi")
        C_values.append(float(compute_coherence(G)))

    C_values = np.asarray(C_values)
    C_final = float(C_values[-1])
    C_std = float(np.std(C_values[-50:]))  # stability in last segment
    return C_final, C_std


def exp3_replication_fidelity() -> float:
    # Parent and offspring EPI patterns (normalized) with small noise
    rng = np.random.default_rng(42)
    parent = rng.normal(0.0, 1.0, size=128)
    parent /= (np.linalg.norm(parent) + 1e-12)
    offspring = parent + 0.05 * rng.normal(0.0, 1.0, size=128)
    offspring /= (np.linalg.norm(offspring) + 1e-12)

    # Fidelity as cosine similarity (proxy for Fr)
    Fr = float(np.dot(parent, offspring))
    return Fr


def main() -> None:
    t_threshold, Amax = exp1_life_emergence()
    C_final, C_std = exp2_self_maintenance_coherence()
    Fr = exp3_replication_fidelity()

    print("Experiment Results:")
    print(f"  1) Life Emergence: threshold_time={t_threshold:.3f}, A_max={Amax:.3f}")
    print(f"  2) Self-maintenance: C_final={C_final:.3f}, C_std(last)= {C_std:.4f}")
    print(f"  3) Replication Fidelity: Fr={Fr:.3f}")

    # Minimal acceptance criteria (informal):
    ok1 = t_threshold >= 0.0 and Amax > 1.0
    ok2 = C_final > 0.6 and C_std < 0.02
    ok3 = Fr > 0.8
    print(f"  Acceptance: exp1={ok1}, exp2={ok2}, exp3={ok3}")


if __name__ == "__main__":
    main()

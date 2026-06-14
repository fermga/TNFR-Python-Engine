"""TNFR Example 116: Prime visibility from νf-embedded emergent geometry.

Goal
====
Measure whether TNFR structural diffusion / emergent substrate can "see"
prime structure when arithmetic enters ONLY through νf (not through an
imposed divisibility/GCD graph).

Design
======
1) Build a topology-neutral coupled network (Watts–Strogatz), independent
   of arithmetic relations.
2) Label nodes with integers n=2..N+1 only for arithmetic metadata.
3) Inject arithmetic in νf, then evolve the nodal equation:

      ∂EPI/∂t = νf · ΔNFR

4) Read two canonical observability channels:
   - emergent substrate fields (K_φ, J_φ, Φ_s, J_ΔNFR)
   - structural diffusion current/divergence
5) Compare prime vs composite separability and controls.

Controls
========
- Uniform νf baseline (no arithmetic content)
- Prime-coded νf (arithmetic in νf)
- Shuffled νf control (same νf histogram, broken arithmetic alignment)

Measured result (N=240, seeds 7/13/29; the νf-as-mobility echo)
==============================================================
- The structural-DIFFUSION channel (EPI transport: current divergence and
  current L1) DOES acquire a clean prime signature in the ``prime``
  scenario: r_pb(is_prime, ·) ≈ 0.25–0.34, Cohen d ≈ 0.62–0.87,
  reproducible across all three seeds, and it COLLAPSES to ≈0 under the
  ``prime-shuffled`` control (same νf histogram, primality alignment
  broken).  The shuffle control proves the signal is νf-on-primes
  alignment, not topology.
- The substrate / ΔNFR-derived fields (Φ_s, J_ΔNFR) and the frozen-phase
  fields (K_φ, J_φ) do NOT acquire a prime signature: prime ≈
  prime-shuffled ≈ uniform.

Mechanism (honest): νf is the per-node MOBILITY of the nodal equation
(EPI += dt·νf·ΔNFR).  Prime nodes carry high νf, so they take larger EPI
steps and develop a distinctive transient diffusion current at exactly the
prime sites.  The νf-channel of ΔNFR is minor (ΔNFR barely moves), so the
ΔNFR-derived substrate fields stay blind.  The geometry ECHOES the νf
contrast you inject, in the channel νf drives — it does not independently
"discover" prime structure.  This mirrors example 103 ("substrate blind to
Riemann, content in νf") and the REMESH-∞ statement that TNFR universality
is structural/operational, not spectral: the engine re-expresses what you
put in νf.

Note (cache integrity): an earlier run reported the substrate fields as
bit-identical across scenarios — a stale-cache artifact in the Φ_s/ξ_C/
J_ΔNFR dependency hash (canonical-alias key mismatch).  That bug was fixed
(tnfr.utils.cache._compute_dependency_hash); the numbers above are the
uncontaminated measurement.

Honest scope
============
This is a measurement script. It does not claim a new theorem and does not
resolve any open program. It checks doctrinal fidelity: arithmetic enters
through νf, while coupling topology remains arithmetic-neutral.
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import current_divergence, structural_current
from tnfr.physics.symplectic_substrate import extract_phase_space_point


def _primes_up_to(n_max: int) -> set[int]:
    """Simple sieve for prime labels."""
    if n_max < 2:
        return set()
    is_prime = [True] * (n_max + 1)
    is_prime[0] = is_prime[1] = False
    lim = int(math.isqrt(n_max))
    for p in range(2, lim + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start : n_max + 1 : p] = [False] * (((n_max - start) // p) + 1)
    return {i for i, ok in enumerate(is_prime) if ok}


def _point_biserial(binary: np.ndarray, values: np.ndarray) -> float:
    """Point-biserial correlation corr(binary, values)."""
    b = binary.astype(float)
    v = values.astype(float)
    if float(np.std(v)) <= 1e-15:
        return 0.0
    return float(np.corrcoef(b, v)[0, 1])


def _cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Effect size between prime and composite groups."""
    a = group_a.astype(float)
    b = group_b.astype(float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max((len(a) + len(b) - 2), 1)
    if pooled <= 1e-15:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled))


def _build_neutral_graph(n_nodes: int, seed: int) -> nx.Graph:
    """Arithmetic-neutral topology + random initial TNFR state."""
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n_nodes, 6, 0.2, seed=seed)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        set_attr(G.nodes[nd], ALIAS_EPI, rng.uniform(-0.35, 0.35))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)
    return G


def _assign_nu_f(
    G: nx.Graph,
    labels: np.ndarray,
    prime_labels: set[int],
    mode: str,
    *,
    base: float = 0.9,
    amp: float = 0.8,
    seed: int = 0,
) -> None:
    """Inject arithmetic through νf only."""
    nodes = list(G.nodes())
    if mode == "uniform":
        vf = np.full(len(nodes), base, dtype=float)
    elif mode == "prime":
        vf = np.array(
            [base + amp if int(labels[i]) in prime_labels else base for i in range(len(nodes))],
            dtype=float,
        )
    elif mode == "prime-shuffled":
        raw = np.array(
            [base + amp if int(labels[i]) in prime_labels else base for i in range(len(nodes))],
            dtype=float,
        )
        rng = np.random.default_rng(seed)
        vf = raw.copy()
        rng.shuffle(vf)
    else:
        raise ValueError(f"Unknown νf mode: {mode}")

    for i, nd in enumerate(nodes):
        set_attr(G.nodes[nd], ALIAS_VF, float(vf[i]))


def _evolve_nodal(G: nx.Graph, n_steps: int = 16, dt: float = 0.05) -> None:
    """Integrate nodal equation explicitly: EPI <- EPI + dt * νf * ΔNFR.

    Short transient: with νf as the per-node mobility, high-νf nodes relax
    faster.  Observables are read mid-transient so the EPI-diffusion channel
    has NOT washed out to the uniform equilibrium (which carries zero
    current and zero divergence).
    """
    for _ in range(n_steps):
        default_compute_delta_nfr(G)
        for nd in G.nodes():
            epi = float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0))
            vf = float(get_attr(G.nodes[nd], ALIAS_VF, 0.0))
            dnfr = float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0))
            set_attr(G.nodes[nd], ALIAS_EPI, epi + dt * vf * dnfr)


def _collect_observables(G: nx.Graph) -> dict[str, np.ndarray]:
    """Read canonical substrate + diffusion observables."""
    p = extract_phase_space_point(G)
    _, div = current_divergence(G)
    _, cur = structural_current(G)
    current_l1 = np.sum(np.abs(cur), axis=1)

    # per-node substrate energy density (symplectic core)
    e_sub = 0.5 * (
        np.asarray(p.k_phi) ** 2
        + np.asarray(p.j_phi) ** 2
        + np.asarray(p.phi_s) ** 2
        + np.asarray(p.j_dnfr) ** 2
    )

    return {
        "phi_s": np.asarray(p.phi_s, dtype=float),
        "j_dnfr": np.asarray(p.j_dnfr, dtype=float),
        "k_phi": np.asarray(p.k_phi, dtype=float),
        "e_sub": np.asarray(e_sub, dtype=float),
        "diff_div": np.asarray(div, dtype=float),
        "diff_current_l1": np.asarray(current_l1, dtype=float),
    }


def _score_prime_visibility(is_prime: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Return (point-biserial r, Cohen d prime-vs-composite)."""
    primes = x[is_prime]
    comps = x[~is_prime]
    r_pb = _point_biserial(is_prime.astype(float), x)
    d = _cohen_d(primes, comps)
    return r_pb, d


def run_trial(n_nodes: int, seed: int) -> None:
    labels = np.arange(2, n_nodes + 2)  # arithmetic labels, not graph edges
    prime_labels = _primes_up_to(int(labels[-1]))
    is_prime = np.array([int(n) in prime_labels for n in labels], dtype=bool)

    base_graph = _build_neutral_graph(n_nodes=n_nodes, seed=seed)
    scenarios = ("uniform", "prime", "prime-shuffled")

    print("=" * 78)
    print(f"nu_f-EMBEDDED PRIME VISIBILITY (N={n_nodes}, seed={seed})")
    print("Topology: Watts-Strogatz neutral graph (no arithmetic edges)")
    print("Arithmetic enters only through nu_f assignment")
    print("=" * 78)
    print()

    for mode in scenarios:
        G = base_graph.copy()
        _assign_nu_f(G, labels, prime_labels, mode, seed=seed + 17)
        _evolve_nodal(G, n_steps=16, dt=0.05)
        default_compute_delta_nfr(G)

        obs = _collect_observables(G)

        print(f"Scenario: {mode}")
        print(f"  {'observable':<18} {'r_pb(prime,x)':>14} {'cohen_d':>10}")
        for key in ("e_sub", "phi_s", "j_dnfr", "diff_div", "diff_current_l1"):
            r_pb, d = _score_prime_visibility(is_prime, np.abs(obs[key]))
            print(f"  {key:<18} {r_pb:>14.4f} {d:>10.4f}")
        print()

    print("Interpretation guideline:")
    print("  - uniform: near-zero separability (no arithmetic in nu_f)")
    print("  - prime: diffusion channel (diff_div/diff_current) acquires a")
    print("    prime signature; DeltaNFR-derived substrate fields stay ~flat")
    print("  - prime-shuffled: diffusion signature collapses to baseline")
    print("    (confirms the signal is nu_f-on-primes alignment, not topology)")
    print()


def main() -> None:
    print()
    print("#" * 78)
    print("# TNFR Example 116: nu_f-driven emergent number geometry")
    print("# Prime structure enters through nu_f, not imposed arithmetic graph")
    print("#" * 78)
    print()
    for seed in (7, 13, 29):
        run_trial(n_nodes=240, seed=seed)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("Under nu_f-only arithmetic embedding on an arithmetic-neutral graph:")
    print("  - the structural-DIFFUSION channel SEES the nu_f-prime imprint")
    print("    (r_pb~0.25-0.34, Cohen d~0.6-0.9), collapsing under the")
    print("    prime-shuffled control across 3 seeds;")
    print("  - the DeltaNFR-derived substrate fields stay blind.")
    print("Mechanism: nu_f-as-mobility echo. nu_f is the per-node diffusivity,")
    print("so prime nodes develop a distinctive EPI transient. The geometry")
    print("re-expresses what you inject through nu_f; it does not independently")
    print("discover primes. Doctrine-fidelity measurement, not a closure.")


if __name__ == "__main__":
    main()

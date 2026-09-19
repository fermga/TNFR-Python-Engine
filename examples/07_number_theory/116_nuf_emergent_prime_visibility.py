"""TNFR Example 116: Prime visibility from νf-embedded emergent geometry.

Goal
====
Measure whether TNFR structural diffusion / emergent substrate can "see"
arithmetic structure when arithmetic enters ONLY through νf (not through an
imposed divisibility/GCD graph) — and, decisively, whether any apparent
"prime visibility" is specific to primality or is just an echo of WHATEVER
νf contrast is injected.

Design
======
1) Build a topology-neutral coupled network (Watts–Strogatz), independent
   of arithmetic relations.
2) Label nodes with integers n=2..N+1 only for arithmetic metadata.
3) Inject arithmetic in νf via several carriers (prime, an arithmetic-free
   random set, Ω(n), log n), then evolve the nodal equation:

      ∂EPI/∂t = νf · ΔNFR

4) Read two canonical observability channels:
   - emergent substrate fields (K_φ, J_φ, Φ_s, J_ΔNFR)
   - structural diffusion current/divergence
5) Compare, carrier-agnostically, how strongly each channel correlates with
   the injected νf field; plus a prime-alignment shuffle control.

Controls (carriers, all matched in νf amplitude [base, base+amp])
================================================================
- uniform: no νf contrast (baseline)
- prime: νf high on the primes (binary)
- arbitrary: νf high on a RANDOM equal-size set, NO arithmetic meaning
  (the decisive carrier-agnostic control)
- omega: νf graded by Ω(n), prime-factor count with multiplicity (continuous)
- logn: νf graded by log n (continuous)
- prime-shuffled: prime νf histogram, alignment to primality broken

Measured result (N=240, seeds 7/13/29; the νf-as-mobility echo)
==============================================================
- Carrier-agnostic, Pearson(|diffusion channel|, injected νf): the binary
  carriers ``prime`` and ``arbitrary`` are EQUALLY visible in the diffusion
  channel — prime ≈ 0.25–0.34, arbitrary ≈ 0.27–0.29 across the three
  seeds.  A random equal-size set with NO arithmetic meaning echoes just as
  strongly as the primes, so primality is NOT special.
- The continuous carriers ``omega`` and ``logn`` echo too (above the
  uniform 0.0) but more weakly (≈ 0.02–0.19): the echo strength tracks the
  spatial SHARPNESS of the νf contrast (sharp binary > smoothly graded),
  not arithmetic content.
- The ΔNFR-derived substrate fields (Φ_s, J_ΔNFR) stay near zero (|r| ≲ 0.18,
  no consistent sign) for EVERY carrier — blind to all of them.
- Prime-alignment, point-biserial(is_prime, |diff_div|): the prime signature
  (r_pb ≈ 0.25–0.34, Cohen d ≈ 0.62–0.87) COLLAPSES to ≈0 under
  ``prime-shuffled`` (same νf histogram, alignment broken).  The diffusion
  field sits on the primes only because/when νf sits on the primes.

Mechanism (honest): νf is the per-node MOBILITY of the nodal equation
(EPI += dt·νf·ΔNFR).  Whichever nodes carry high νf take larger EPI steps
and develop a distinctive transient diffusion current at exactly those
sites — primes, an arbitrary set, or high-Ω(n) nodes alike.  The νf-channel
of ΔNFR is minor (ΔNFR barely moves), so the ΔNFR-derived substrate fields
stay blind.  The geometry ECHOES ANY νf contrast you inject, in the channel
νf drives — it does not "discover" prime structure.  This mirrors example
103 ("substrate blind to Riemann, content in νf") and the REMESH-∞
statement that TNFR universality is structural/operational, not spectral:
the engine re-expresses what you put in νf.

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
            [
                base + amp if int(labels[i]) in prime_labels else base
                for i in range(len(nodes))
            ],
            dtype=float,
        )
    elif mode == "prime-shuffled":
        raw = np.array(
            [
                base + amp if int(labels[i]) in prime_labels else base
                for i in range(len(nodes))
            ],
            dtype=float,
        )
        rng = np.random.default_rng(seed)
        vf = raw.copy()
        rng.shuffle(vf)
    else:
        raise ValueError(f"Unknown νf mode: {mode}")

    for i, nd in enumerate(nodes):
        set_attr(G.nodes[nd], ALIAS_VF, float(vf[i]))


def _omega(n: int) -> int:
    """Big-Omega: number of prime factors of n counted with multiplicity."""
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            n //= d
            count += 1
        d += 1
    if n > 1:
        count += 1
    return count


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; 0.0 if either side is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _carrier_vf(
    labels: np.ndarray,
    kind: str,
    prime_labels: set[int],
    *,
    base: float = 0.9,
    amp: float = 0.8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a nu_f field for an arithmetic carrier.

    Returns ``(vf, injected_contrast)`` with ``injected_contrast = vf - base``,
    the per-node nu_f deviation actually injected.  Every non-uniform carrier
    spans the SAME nu_f range ``[base, base+amp]``, so carriers are matched in
    nu_f amplitude and differ only in WHICH nodes carry high nu_f.
    """
    n = len(labels)
    if kind == "uniform":
        vf = np.full(n, base, dtype=float)
    elif kind == "prime":
        ind = np.array([int(labels[i]) in prime_labels for i in range(n)], dtype=float)
        vf = base + amp * ind
    elif kind == "arbitrary":
        # a random subset of the SAME cardinality as the primes, with NO
        # arithmetic meaning: the decisive carrier-agnostic control.
        k = int(sum(1 for x in labels if int(x) in prime_labels))
        rng = np.random.default_rng(seed)
        sel = np.zeros(n, dtype=float)
        sel[rng.choice(n, size=k, replace=False)] = 1.0
        vf = base + amp * sel
    elif kind == "omega":
        # graded by Omega(n) (prime-factor count): primes sit at the LOW end
        om = np.array([_omega(int(x)) for x in labels], dtype=float)
        vf = base + amp * (om - om.min()) / (om.max() - om.min() + 1e-12)
    elif kind == "logn":
        lg = np.log(labels.astype(float))
        vf = base + amp * (lg - lg.min()) / (lg.max() - lg.min() + 1e-12)
    else:
        raise ValueError(f"Unknown carrier: {kind}")
    return vf, vf - base


def _assign_vf_array(G: nx.Graph, vf: np.ndarray) -> None:
    """Write a precomputed nu_f array onto the graph (canonical alias)."""
    for nd, v in zip(G.nodes(), vf):
        set_attr(G.nodes[nd], ALIAS_VF, float(v))


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

    print("=" * 78)
    print(f"nu_f-EMBEDDED ARITHMETIC VISIBILITY (N={n_nodes}, seed={seed})")
    print("Topology: Watts-Strogatz neutral graph (no arithmetic edges)")
    print("Arithmetic enters only through nu_f assignment")
    print("=" * 78)
    print()

    # --- Section 1: carrier-agnostic nu_f echo --------------------------
    # Pearson(|channel|, injected nu_f contrast) for several arithmetic
    # carriers, ALL matched in nu_f amplitude.  If the diffusion channel
    # correlates with EVERY carrier comparably, the echo is carrier-
    # agnostic: it follows the injected nu_f field, not "prime structure".
    print("Section 1: carrier-agnostic echo  Pearson(|chan|, injected nu_f)")
    print(
        f"  {'carrier':<11} {'diff_div':>9} {'diff_curr':>10} "
        f"{'phi_s':>8} {'j_dnfr':>8}"
    )
    for kind in ("uniform", "prime", "arbitrary", "omega", "logn"):
        G = base_graph.copy()
        vf, contrast = _carrier_vf(labels, kind, prime_labels, seed=seed + 17)
        _assign_vf_array(G, vf)
        _evolve_nodal(G, n_steps=16, dt=0.05)
        default_compute_delta_nfr(G)
        obs = _collect_observables(G)
        r_div = _pearson(np.abs(obs["diff_div"]), contrast)
        r_cur = _pearson(np.abs(obs["diff_current_l1"]), contrast)
        r_phi = _pearson(np.abs(obs["phi_s"]), contrast)
        r_jd = _pearson(np.abs(obs["j_dnfr"]), contrast)
        print(
            f"  {kind:<11} {r_div:>9.4f} {r_cur:>10.4f} " f"{r_phi:>8.4f} {r_jd:>8.4f}"
        )
    print()

    # --- Section 2: prime-alignment control -----------------------------
    # Point-biserial(is_prime, |diff_div|): the diffusion field sits on the
    # PRIMES only when nu_f is on the primes; shuffling nu_f (same histogram,
    # alignment broken) collapses it -> the Section-1 prime row is not about
    # primality, it is about WHERE the high-nu_f nodes are.
    print("Section 2: prime-alignment  point-biserial(is_prime, |diff_div|)")
    print(f"  {'scenario':<16} {'r_pb':>8} {'cohen_d':>9}")
    for mode in ("prime", "prime-shuffled"):
        G = base_graph.copy()
        _assign_nu_f(G, labels, prime_labels, mode, seed=seed + 17)
        _evolve_nodal(G, n_steps=16, dt=0.05)
        default_compute_delta_nfr(G)
        obs = _collect_observables(G)
        r_pb, d = _score_prime_visibility(is_prime, np.abs(obs["diff_div"]))
        print(f"  {mode:<16} {r_pb:>8.4f} {d:>9.4f}")
    print()

    print("Interpretation:")
    print("  - Section 1: the diffusion channel correlates with EVERY nu_f")
    print("    carrier (prime, arbitrary, omega, logn) comparably; uniform")
    print("    gives ~0. The DeltaNFR substrate fields (phi_s, j_dnfr)")
    print("    stay ~0 for all carriers. -> carrier-agnostic nu_f echo.")
    print("  - Section 2: the diffusion field sits on the primes ONLY when")
    print("    nu_f is on the primes; the shuffle collapses it. Primality is")
    print("    not special -> nu_f-as-mobility echo, not emergence.")
    print()


def main() -> None:
    print()
    print("#" * 78)
    print("# TNFR Example 116: nu_f-driven emergent number geometry")
    print("# Arithmetic enters through nu_f, not an imposed arithmetic graph")
    print("#" * 78)
    print()
    for seed in (7, 13, 29):
        run_trial(n_nodes=240, seed=seed)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("Under nu_f-only arithmetic embedding on a neutral graph:")
    print("  - the structural-DIFFUSION channel correlates with EVERY nu_f")
    print("    carrier (prime, arbitrary random set, Omega(n), log n) at a")
    print("    comparable level; the DeltaNFR-derived substrate fields stay")
    print("    blind to all of them;")
    print("  - the prime-aligned diffusion signature collapses when nu_f is")
    print("    shuffled off the primes (Section 2).")
    print("Mechanism: nu_f-as-mobility echo. nu_f is the per-node mobility,")
    print("so whichever nodes carry high nu_f develop a distinctive EPI")
    print("transient. The geometry re-expresses ANY nu_f contrast you inject;")
    print("primality is not special. Doctrine-fidelity measurement, not a")
    print("closure.")


if __name__ == "__main__":
    main()

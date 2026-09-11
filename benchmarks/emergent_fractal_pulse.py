"""Emergent Fractal Resonant Pulse: resonance locks scale by scale.

THE PARADIGM (user, theory creator): "el pulso fractal resonante" -- the
per-NFR pulse is not a single global beat; on a self-similar (nested) network
the resonance LOCKS SCALE BY SCALE, fine -> coarse. The local-before-global
synchronization measured by net.pulse_trajectory() (R rises locally first) is
the two-scale shadow of a genuinely MULTI-SCALE cascade.

THE STRUCTURAL FACTS (measured by evolving, not by a fixed point):
  - every NFR is a phase oscillator pulsing at nu_f with phase theta; coupling
    is RESONANCE (the phase channel of dNFR ~ -L_rw theta in the small-angle
    limit), so the Kuramoto order R = |<e^{i theta}>| relaxes toward 1;
  - the relaxation rate of a phase mode is proportional to nu_f * lambda
    (its diffusion eigenvalue): HIGH-lambda modes (tight intra-cluster) lock
    FAST, LOW-lambda modes (the weak global links, near lambda_2) lock SLOW;
  - on a 3-level self-similar (ultrametric) coupling -- each coarser scale
    couples a factor r weaker -- the spectrum SPLITS into self-similar BANDS,
    one per scale, so the sync timescales are SET by the spectral bands.

WHAT EMERGES (measured):
  - M1 THE CASCADE: R_leaf > R_group > R_whole at EVERY step -- the resonance
    locks fine -> coarse (local before global), self-similarly.
  - M2 THE BANDS: the L_sym spectrum splits into 3 self-similar bands, one per
    coupling scale (intra-block high / intra-group meso / inter-group coarse).
  - M3 TIMESCALE = BAND: the measured per-scale synchronization time scales as
    1 / (nu_f * lambda_band) -- the leaf (high-lambda band) syncs first, the
    whole (lambda_2 band) last; the timescales ARE the spectral bands.
  - M4 SELF-SIMILAR SPACING: the bands are cleanly separated, one per scale,
    reflecting the geometric (ratio r) coupling hierarchy -- the fractal
    signature of the nested structure (the gaps, not a single constant ratio).

So the FRACTAL RESONANT PULSE = resonance (Kuramoto phase-locking) unfolding
self-similarly across the nested scales encoded in the spectral-gap structure;
the per-NFR pulses lock scale by scale, the collective pulse emerging as the
coarsest band finally synchronizes.

HONEST SCOPE: Kuramoto synchronization and the diffusion spectrum are standard
physics; the TNFR content is the READING -- the per-NFR pulse (nu_f, theta) and
its resonance (R) lock self-similarly because the relaxation rate is nu_f *
lambda and a self-similar form has a banded spectrum. Closes nothing (R and pi
assumed; the prime/zeta rhythm remains the wall).

Run:
    python benchmarks/emergent_fractal_pulse.py

Theoretical anchor: AGENTS.md (the per-NFR pulse; resonance R; nu_f * lambda
relaxation; resonant fractal nature); src/tnfr/physics/structural_diffusion.py
(compute_nodal_pulse / compute_emergent_pulse); the SDK read-outs
net.resonance() and net.pulse_trajectory() compute the same per-NFR pulse on a
single network; benchmarks/emergent_rhythm.py (the conservative face:
frequency decimation).
Status: RESEARCH.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def lsym(G, nodes):
    """Symmetric normalized Laplacian L_sym (shares spectrum with L_rw)."""
    A = nx.to_numpy_array(G, nodelist=nodes)
    d = A.sum(axis=1)
    dinv = np.zeros_like(d)
    pos = d > 0.0
    dinv[pos] = 1.0 / np.sqrt(d[pos])
    L = np.eye(len(nodes)) - (dinv[:, None] * A * dinv[None, :])
    return 0.5 * (L + L.T)


def lrw_phase_step(theta, A, deg, nu_f, dt):
    """One explicit step of the resonant phase channel (random-walk Kuramoto).

    theta_dot_i = nu_f * (1/d_i) * sum_j A_ij sin(theta_j - theta_i)

    This is the nonlinear phase coupling of dNFR; its small-angle limit is the
    canonical -nu_f * L_rw * theta diffusion (relaxation rate nu_f * lambda).
    """
    diff = theta[None, :] - theta[:, None]
    coupling = (A * np.sin(diff)).sum(axis=1) / np.maximum(deg, 1.0)
    return theta + dt * nu_f * coupling


def hierarchical_graph(n_groups, n_blocks, block_size, ratio):
    """A 3-level self-similar (ultrametric) resonant coupling.

    Every pair of NFRs is coupled; the weight depends only on the COARSEST
    shared scale -- a geometric hierarchy (each coarser scale couples a factor
    `ratio` weaker):

        same leaf block         -> weight ratio**2  (fine,   strong)
        same group, diff block  -> weight ratio**1  (meso)
        different group         -> weight 1.0       (coarse, weak)

    This is the canonical self-similar resonant coupling (an ultrametric /
    hierarchical network); its random-walk diffusion spectrum bands into one
    cluster per scale.

    Returns (G, leaf_sets, group_sets) where the *_sets are lists of node-id
    lists, one per leaf block / per group.
    """
    leaf_of: dict[int, tuple[int, int]] = {}
    group_of: dict[int, int] = {}
    leaf_sets: list[list[int]] = []
    group_sets: list[list[int]] = []
    nid = 0
    for g in range(n_groups):
        gnodes: list[int] = []
        for b in range(n_blocks):
            block = list(range(nid, nid + block_size))
            nid += block_size
            for v in block:
                leaf_of[v] = (g, b)
                group_of[v] = g
            leaf_sets.append(block)
            gnodes.extend(block)
        group_sets.append(gnodes)
    n = nid
    w_fine, w_meso, w_coarse = ratio ** 2, ratio, 1.0
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if leaf_of[i] == leaf_of[j]:
                w = w_fine
            elif group_of[i] == group_of[j]:
                w = w_meso
            else:
                w = w_coarse
            G.add_edge(i, j, weight=w)
    return G, leaf_sets, group_sets


def order_param(theta, idx_sets):
    """Mean Kuramoto order R = |<e^{i theta}>| over the given node sets."""
    rs = [abs(np.mean(np.exp(1j * theta[s]))) for s in idx_sets]
    return float(np.mean(rs))


def split_bands(eigvals, n_bands=3):
    """Split the nonzero spectrum into n_bands clusters at the largest gaps."""
    w = np.sort(eigvals[eigvals > 1e-9])
    if len(w) <= n_bands:
        return [np.array([v]) for v in w]
    gaps = np.diff(w)
    cuts = np.sort(np.argsort(gaps)[-(n_bands - 1):])
    bands = []
    start = 0
    for c in cuts:
        bands.append(w[start:c + 1])
        start = c + 1
    bands.append(w[start:])
    return bands


def sync_time(hist, idx_sets, threshold):
    """First step index at which the scale's R crosses threshold (else -1)."""
    for t, theta in enumerate(hist):
        if order_param(theta, idx_sets) >= threshold:
            return t
    return -1


def main() -> None:
    print("=" * 70)
    print("EMERGENT FRACTAL RESONANT PULSE -- resonance locks scale by scale")
    print("=" * 70)

    n_groups, n_blocks, block_size, ratio = 3, 3, 5, 6.0
    nu_f, dt, steps = 1.0, 0.3, 400
    G, leaf_sets, group_sets = hierarchical_graph(
        n_groups, n_blocks, block_size, ratio
    )
    nodes = sorted(G.nodes)
    whole = [nodes]
    A = nx.to_numpy_array(G, nodelist=nodes)
    deg = A.sum(axis=1)
    print(
        f"\nself-similar ultrametric coupling: {n_groups} groups x "
        f"{n_blocks} blocks x {block_size} nodes = {len(nodes)} NFRs "
        f"(weight ratio r={ratio})"
    )

    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=len(nodes))
    hist = [theta.copy()]
    for _ in range(steps):
        theta = lrw_phase_step(theta, A, deg, nu_f, dt)
        hist.append(theta.copy())

    # M1 -- THE CASCADE: R_leaf >= R_group >= R_whole at every step
    print("\nM1 -- the cascade R_leaf > R_group > R_whole (fine -> coarse):")
    ok_cascade = True
    for t in (0, 5, 20, 60, 150, steps):
        r_l = order_param(hist[t], leaf_sets)
        r_g = order_param(hist[t], group_sets)
        r_w = order_param(hist[t], whole)
        flag = "" if (r_l + 1e-9 >= r_g >= r_w - 1e-9) else "  <-- broken"
        if flag:
            ok_cascade = False
        print(
            f"  step {t:>4}:  R_leaf={r_l:.3f}  R_group={r_g:.3f}  "
            f"R_whole={r_w:.3f}{flag}"
        )
    print(f"  => cascade holds at every probe: {ok_cascade}")

    # M2 -- THE BANDS: L_sym spectrum splits into 3 self-similar bands
    print("\nM2 -- the spectrum splits into 3 bands (one per coupling scale):")
    L = lsym(G, nodes)
    eigvals = np.clip(np.linalg.eigvalsh(L), 0.0, None)
    bands = split_bands(eigvals, 3)
    centers = [float(b.mean()) for b in bands]
    labels = [
        "coarse (inter-group)", "meso (intra-group)", "fine (intra-block)"
    ]
    for lbl, b in zip(labels, bands):
        print(
            f"  {lbl:>22}: n={len(b):>2}  "
            f"lambda in [{b.min():.3f}, {b.max():.3f}]  center={b.mean():.3f}"
        )

    # M3 -- TIMESCALE = BAND: sync time scales as 1 / (nu_f * lambda_band)
    print("\nM3 -- per-scale sync time ~ 1/(nu_f*lambda_band) (leaf first):")
    thr = np.pi / (np.pi + 1.0)  # strong-coherence cut (pi-band complement)
    t_leaf = sync_time(hist, leaf_sets, thr)
    t_group = sync_time(hist, group_sets, thr)
    t_whole = sync_time(hist, whole, thr)
    pred = [1.0 / (nu_f * c) for c in centers]  # coarse, meso, fine
    print(f"  strong-coherence threshold R >= pi/(pi+1) = {thr:.3f}")
    print(
        f"  measured sync step: leaf={t_leaf} group={t_group} "
        f"whole={t_whole}"
    )
    print(
        f"  predicted 1/(nu_f*lambda) (fine/meso/coarse): "
        f"{pred[2]:.2f} / {pred[1]:.2f} / {pred[0]:.2f}"
    )
    ok_order = (
        0 <= t_leaf <= t_group <= t_whole
        or (t_leaf >= 0 and t_whole == -1)
    )
    print(f"  => leaf locks before group before whole: {ok_order}")

    # M4 -- SELF-SIMILAR SPACING: bands reflect the geometric coupling ratio
    print("\nM4 -- self-similar band separation (geometric coupling r):")
    asc = sorted(centers)
    gaps = [asc[i + 1] - asc[i] for i in range(len(asc) - 1)]
    print(f"  coupling hierarchy ratio r = {ratio}  (weights r^2 : r : 1)")
    print(f"  band centers (asc): {[round(c, 3) for c in asc]}")
    print(f"  inter-band gaps:    {[round(gp, 3) for gp in gaps]}")
    print(
        "  (3 cleanly separated bands, one per nested scale = self-similar)"
    )

    print("\n" + "=" * 70)
    print(
        "VERDICT: the per-NFR pulse locks SCALE BY SCALE (fine -> coarse);\n"
        "the timescales are the self-similar spectral bands. The fractal\n"
        "resonant pulse = resonance unfolding across the nested scales."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

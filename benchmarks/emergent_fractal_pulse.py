"""Auxiliary hierarchical phase relaxation and its graph spectrum.

The declared model is theta_dot_i = nu_f * sum_j W_ij sin(theta_j-theta_i)/d_i,
with held positive common nu_f and fixed symmetric nonnegative conductance.
Its small-spread linearization is -nu_f * L_rw * theta. This supplies a useful
comparison between spectral relaxation scales and nonlinear synchronization
on a prescribed hierarchy; it does not derive a primitive phase law from the
nodal EPI equation. The actual canonical phase pressure uses an unweighted
neighbor-phasor argument, not this weighted sine average.

The hierarchy, initial phases, capacity and explicit Euler step are supplied.
The experiment records scale-specific Kuramoto orders, threshold crossings and
spectral bands. For these equal-sized nested groups, R_leaf >= R_group >=
R_whole already follows from the triangle inequality at every snapshot; that
ordering alone is not evidence of a dynamical synchronization cascade.
The reference 1/(nu_f*lambda) is a linearized modal e-folding time, not an exact
nonlinear threshold-crossing prediction. Band grouping and the chosen
coherence threshold are diagnostics, not derived transition criteria.

This preserves the useful multiscale relaxation experiment, without claiming
a self-generated clock, sustained oscillation, or an autonomous TNFR pattern.

Run:
    python benchmarks/emergent_fractal_pulse.py

Related owners: src/tnfr/physics/structural_diffusion.py (shared Laplacian and
scoped pulse read-outs), theory/TNFR_VARIATIONAL_PRINCIPLE.md (phase pressure
and auxiliary laws), and benchmarks/emergent_rhythm.py (declared graph wave).
Status: auxiliary research experiment.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from tnfr.physics.structural_diffusion import symmetric_normalized_laplacian


def lsym(G, nodes):
    """Shared symmetric normalized Laplacian, including zero isolate rows."""
    return symmetric_normalized_laplacian(G, nodes)[1]


def lrw_phase_step(theta, A, deg, nu_f, dt):
    """One explicit Euler step of the supplied normalized Kuramoto model.

    theta_dot_i = nu_f * (1/d_i) * sum_j A_ij sin(theta_j - theta_i)

    Its small-angle limit is -nu_f * L_rw * theta; this phase evolution law
    is an auxiliary premise, not the canonical nodal pressure channel.
    A is a nonnegative matrix and deg its row sums; isolates remain fixed.
    """
    diff = theta[None, :] - theta[:, None]
    numerator = (A * np.sin(diff)).sum(axis=1)
    coupling = np.divide(numerator, deg, out=np.zeros_like(numerator), where=deg > 0)
    return theta + dt * nu_f * coupling


def hierarchical_graph(n_groups, n_blocks, block_size, ratio):
    """A 3-level self-similar (ultrametric) resonant coupling.

    Every pair of NFRs is coupled; the weight depends only on the COARSEST
    shared scale -- a geometric hierarchy (each coarser scale couples a factor
    `ratio` weaker):

        same leaf block         -> weight ratio**2  (fine,   strong)
        same group, diff block  -> weight ratio**1  (meso)
        different group         -> weight 1.0       (coarse, weak)

    This is a supplied hierarchical coupling, not a derived TNFR graph.
    The benchmark uses ratio > 1, making fine links stronger than coarse ones.

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
    w_fine, w_meso, w_coarse = ratio**2, ratio, 1.0
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
    cuts = np.sort(np.argsort(gaps)[-(n_bands - 1) :])
    bands = []
    start = 0
    for c in cuts:
        bands.append(w[start : c + 1])
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
    print("AUXILIARY HIERARCHICAL PHASE RELAXATION -- spectral scale comparison")
    print("=" * 70)

    n_groups, n_blocks, block_size, ratio = 3, 3, 5, 6.0
    nu_f, dt, steps = 1.0, 0.3, 400
    G, leaf_sets, group_sets = hierarchical_graph(n_groups, n_blocks, block_size, ratio)
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

    # Nested equal-size averages satisfy this ordering before any evolution.
    print("\nM1 -- nested order statistics R_leaf >= R_group >= R_whole:")
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
    print(f"  => aggregation inequality holds at every probe: {ok_cascade}")

    # M2 -- Group the measured spectrum at its largest gaps.
    print("\nM2 -- measured spectrum grouped into 3 bands:")
    L = lsym(G, nodes)
    eigvals = np.clip(np.linalg.eigvalsh(L), 0.0, None)
    bands = split_bands(eigvals, 3)
    centers = [float(b.mean()) for b in bands]
    labels = ["coarse (inter-group)", "meso (intra-group)", "fine (intra-block)"]
    for lbl, b in zip(labels, bands):
        print(
            f"  {lbl:>22}: n={len(b):>2}  "
            f"lambda in [{b.min():.3f}, {b.max():.3f}]  center={b.mean():.3f}"
        )

    # M3 -- Compare nonlinear threshold times to linear modal reference times.
    print("\nM3 -- threshold times and linearized modal e-folding scales:")
    thr = np.pi / (np.pi + 1.0)  # Selected diagnostic threshold.
    t_leaf = sync_time(hist, leaf_sets, thr)
    t_group = sync_time(hist, group_sets, thr)
    t_whole = sync_time(hist, whole, thr)
    pred = [1.0 / (nu_f * c) for c in centers]  # coarse, meso, fine
    print(f"  selected threshold R >= pi/(pi+1) = {thr:.3f}")
    print(f"  first crossing step: leaf={t_leaf} group={t_group} " f"whole={t_whole}")
    times = [None if t < 0 else t * dt for t in (t_leaf, t_group, t_whole)]
    print(f"  corresponding supplied-clock times (leaf/group/whole): {times}")
    print(
        f"  linear reference 1/(nu_f*lambda) (fine/meso/coarse): "
        f"{pred[2]:.2f} / {pred[1]:.2f} / {pred[0]:.2f}"
    )
    ok_order = 0 <= t_leaf <= t_group <= t_whole or (t_leaf >= 0 and t_whole == -1)
    print(f"  => threshold order consistent with aggregation: {ok_order}")

    # M4 -- SELF-SIMILAR SPACING: bands reflect the geometric coupling ratio
    print("\nM4 -- self-similar band separation (geometric coupling r):")
    asc = sorted(centers)
    gaps = [asc[i + 1] - asc[i] for i in range(len(asc) - 1)]
    print(f"  coupling hierarchy ratio r = {ratio}  (weights r^2 : r : 1)")
    print(f"  band centers (asc): {[round(c, 3) for c in asc]}")
    print(f"  inter-band gaps:    {[round(gp, 3) for gp in gaps]}")
    print("  (band gaps describe the supplied hierarchy at this configuration)")

    print("\n" + "=" * 70)
    print(
        "RESULT: recorded nonlinear phase relaxation on a supplied hierarchy,\n"
        "with modal reference scales and threshold crossings. Nested-order\n"
        "inequalities alone do not prove dynamical scale locking or a pulse."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

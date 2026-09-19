"""Investigation: the phi <-> Phi_s tetrad correspondence.

Goal
----
Probe the weakest tetrad link (phi <-> Phi_s) from first principles.
Three questions:

  Q1. Is the U6 confinement scale phi (1.618) actually an approximation
      to the Basel saturation pi^2/6 (1.6449) of the one-sided
      inverse-square sum on a 1D chain? Does it depend on alpha?

  Q2. Where do phi (drift) and 0.7711 (per-node) sit relative to the
      actual Phi_s distribution measured by the canonical engine on
      standard topologies, under both uniform and signed-random DNFR?

  Q3. Does any simple closed form built from (phi, gamma, pi, e, zeta)
      reproduce 0.7711 to high precision? (Proximity != derivation.)

This script is read-only telemetry probing. It mutates only a throwaway
graph's delta_nfr attribute. No canonical constant is changed here.
"""

from __future__ import annotations

import itertools

import mpmath as mp
import networkx as nx
import numpy as np

from tnfr.physics.fields import compute_structural_potential

mp.mp.dps = 40

PHI = (1 + mp.sqrt(5)) / 2
GAMMA = mp.euler
PI = mp.pi
E = mp.e
BASEL = PI**2 / 6  # zeta(2)
ZETA3 = mp.zeta(3)
CATALAN = mp.catalan
TARGET = mp.mpf("0.7711")  # the empirical per-node threshold


def sep(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------
# Q1: confinement scale vs Basel saturation, as a function of alpha
# ---------------------------------------------------------------------
def q1_basel_vs_phi() -> None:
    sep("Q1  Confinement scale phi vs inverse-power chain saturation")
    print(f"phi        = {float(PHI):.6f}")
    print(f"pi^2/6     = {float(BASEL):.6f}  (one-sided 1D chain, alpha=2)")
    print(f"|phi - pi^2/6| / (pi^2/6) = {float(abs(PHI-BASEL)/BASEL):.4%}")
    print()
    print("One-sided chain saturation  S(alpha) = zeta(alpha):")
    for a in [1.5, PHI, 1.8, 2.0, 2.5, 3.0]:
        z = mp.zeta(a)
        print(
            f"  alpha={float(a):.3f}  zeta(alpha)={float(z):8.4f}"
            f"   2-sided={float(2*z):8.4f}"
        )
    print()
    print("Reading: only alpha=2 gives one-sided saturation ~= phi.")
    print("With alpha=phi the saturation is zeta(phi) ~= 2.27 (far).")


# ---------------------------------------------------------------------
# Q2: actual Phi_s distribution on canonical topologies
# ---------------------------------------------------------------------
def _stats(vals: list[float]) -> tuple[float, float, float]:
    a = np.abs(np.array(vals, dtype=float))
    return float(a.mean()), float(np.median(a)), float(a.max())


def _measure(
    G: nx.Graph, mode: str, rng: np.random.Generator, alpha: float
) -> tuple[float, float, float]:
    n = G.number_of_nodes()
    if mode == "uniform":
        d = {node: 1.0 for node in G.nodes()}
    else:  # signed-random, normalized to unit std (typical telemetry)
        x = rng.standard_normal(n)
        x = x / (x.std() + 1e-12)
        d = {node: float(v) for node, v in zip(G.nodes(), x)}
    for node in G.nodes():
        G.nodes[node]["delta_nfr"] = d[node]
    phi_s = compute_structural_potential(G, alpha=alpha)
    vals = [v for k, v in phi_s.items() if not str(k).startswith("__")]
    return _stats(vals)


def q2_topologies() -> None:
    sep("Q2  Phi_s distribution on canonical topologies (alpha=2)")
    rng = np.random.default_rng(42)
    builders = {
        "ring C_30": lambda: nx.cycle_graph(30),
        "path P_30": lambda: nx.path_graph(30),
        "grid 6x6": lambda: nx.grid_2d_graph(6, 6),
        "complete K_20": lambda: nx.complete_graph(20),
        "ER p=0.2 n=40": lambda: nx.gnp_random_graph(40, 0.2, seed=7),
        "BA m=2 n=40": lambda: nx.barabasi_albert_graph(40, 2, seed=7),
    }
    for alpha in (2.0, float(PHI)):
        print(f"\n--- alpha = {alpha:.4f} ---")
        hdr = f"{'topology':16} {'mode':8} {'mean':>9} " f"{'median':>9} {'max':>9}"
        print(hdr)
        for name, build in builders.items():
            for mode in ("uniform", "signed"):
                G = nx.convert_node_labels_to_integers(build())
                mean, med, mx = _measure(G, mode, rng, alpha)
                print(f"{name:16} {mode:8} {mean:9.4f} " f"{med:9.4f} {mx:9.4f}")
    print("\n0.7711 = per-node threshold; phi=1.618; pi^2/6=1.6449")


# ---------------------------------------------------------------------
# Q3: high-precision closed-form search for 0.7711
# ---------------------------------------------------------------------
def q3_closed_form_search() -> None:
    sep("Q3  Closed-form search for 0.7711 (proximity != derivation)")
    base = {
        "phi": PHI,
        "gamma": GAMMA,
        "pi": PI,
        "e": E,
        "zeta2": BASEL,
        "zeta3": ZETA3,
        "catalan": CATALAN,
        "ln2": mp.log(2),
        "sqrt2": mp.sqrt(2),
        "sqrt3": mp.sqrt(3),
        "1": mp.mpf(1),
    }
    cands: dict[str, mp.mpf] = {}
    # ratios a/b and a/(b+c)
    for (na, a), (nb, b) in itertools.permutations(base.items(), 2):
        if b != 0:
            cands[f"{na}/{nb}"] = a / b
    for na, a in base.items():
        for nb, b in base.items():
            for nc, c in base.items():
                denom = b + c
                if denom != 0:
                    cands[f"{na}/({nb}+{nc})"] = a / denom
    # a - b, products, roots
    for (na, a), (nb, b) in itertools.permutations(base.items(), 2):
        cands[f"{na}-{nb}"] = a - b
        cands[f"{na}*{nb}"] = a * b
    for na, a in base.items():
        if a > 0:
            cands[f"sqrt({na})"] = mp.sqrt(a)
            cands[f"exp(-{na})"] = mp.e ** (-a)

    scored = sorted(
        ((abs(v - TARGET), name, v) for name, v in cands.items()),
        key=lambda t: t[0],
    )
    print(f"target = {float(TARGET)}")
    print(f"{'expr':28} {'value':>12} {'abs.resid':>12} {'rel':>9}")
    for resid, name, val in scored[:14]:
        rel = float(resid / TARGET)
        print(f"{name:28} {float(val):12.6f} " f"{float(resid):12.2e} {rel:9.3%}")
    print("\nNote: a match below ~1e-4 with a *structurally meaningful*")
    print("expression would be a derivation candidate; random small")
    print("combinations landing within ~1% are expected by density and")
    print("do NOT constitute a derivation.")


def q4_per_node_variance_scale() -> None:
    """Per-node |Phi_s| under unit-variance signed DNFR.

    If DNFR_j are i.i.d. unit-variance, then
        Var(Phi_s(i)) = Sum_{j!=i} 1/d(i,j)^4
    which on a 1D chain saturates to 2*zeta(4) = pi^4/45.
    The median of a half-normal with that std is 0.6745*std.
    This explains why the per-node scale is O(1), sub-phi.
    """
    sep("Q4  Per-node Phi_s scale vs zeta(4) variance prediction")
    zeta4 = PI**4 / 90
    chain_std = mp.sqrt(2 * zeta4)  # two-sided chain
    print(f"zeta(4)=pi^4/90      = {float(zeta4):.6f}")
    print(f"sqrt(2*zeta(4))      = {float(chain_std):.6f}  (chain std)")
    print(f"half-normal median   = {float(0.6745*chain_std):.6f}")
    print(f"empirical threshold  = {float(TARGET):.6f}\n")
    rng = np.random.default_rng(123)
    for name, build in (
        ("path P_200", lambda: nx.path_graph(200)),
        ("ring C_200", lambda: nx.cycle_graph(200)),
    ):
        meds, stds = [], []
        for _ in range(20):
            G = nx.convert_node_labels_to_integers(build())
            n = G.number_of_nodes()
            x = rng.standard_normal(n)
            x = x / (x.std() + 1e-12)
            for node, v in zip(G.nodes(), x):
                G.nodes[node]["delta_nfr"] = float(v)
            phi_s = compute_structural_potential(G, alpha=2.0)
            vals = np.abs([v for k, v in phi_s.items() if not str(k).startswith("__")])
            meds.append(float(np.median(vals)))
            stds.append(float(vals.std()))
        print(
            f"{name:12} median|Phi_s|={np.mean(meds):.4f} " f"std={np.mean(stds):.4f}"
        )
    print("\nReading: per-node scale is O(1), set by the zeta(4)")
    print("variance of inverse-square accumulation; 0.7711 is the")
    print("empirical operating point within this O(1) band, not a")
    print("closed-form constant.")


if __name__ == "__main__":
    q1_basel_vs_phi()
    q2_topologies()
    q3_closed_form_search()
    q4_per_node_variance_scale()

"""Why the Born rule, single-particle interference statistics, and a physical
hbar are OPEN: Bell's theorem applied to TNFR's own classical polarization.

THE QUESTION (theory creator): can probability amplitudes, the Born rule,
single-particle interference statistics, and a physical hbar be studied as
emergents of TNFR's structure and dynamics? Three of the four (probability
amplitude, Born rule, interference statistics) are ONE question in disguise --
whether TNFR's substrate can reproduce genuine QUANTUM statistics -- and it
admits a DECISIVE, general answer via Bell's theorem (1964), applied directly
to TNFR's own classical polarization sector. The fourth (a physical hbar) is a
units question, addressed separately (Sec. below).

A natural follow-up (theory creator, 2nd round): could FRACTAL and RESONANT
properties of the emergent substrate remove the two premises (locality,
classicality) Bell's theorem needs? Tested directly (M2): they do not, and the
reason is structural, not a matter of effort -- Bell's theorem bounds ANY local
realistic model regardless of how rich, fractal, or resonant the local hidden
variable's own generating mechanism is; the theorem constrains the LOGICAL
structure (locality + realism), not the complexity of what realizes it.

THE TEST (grounded in TNFR's own already-derived structure): the substrate's
Stokes/Poincare polarization (Sec.5.3; symplectic_substrate.polarization_
density) is explicitly CLASSICAL -- a point on the Poincare sphere, the same
mathematics as classical light polarization (Stokes 1852, Poincare 1892). Bell's
theorem says: any LOCAL (Sec.5.1, a finite causal cone) REALISTIC (a definite,
if unmeasured, polarization angle -- exactly what "a point on the Poincare
sphere" means) hidden-variable model is bounded by |S| <= 2 (the CHSH
inequality), while genuine quantum entangled correlations reach |S| = 2*sqrt(2)
(Tsirelson's bound) -- a gap confirmed by real, loophole-free experiments
(Hensen et al. 2015; Giustina et al. 2015; Shalm et al. 2015).

This benchmark builds the standard local-realistic model directly on TNFR's own
classical polarization ANGLE (the equatorial/linear-polarization circle of the
Poincare sphere) and measures its CHSH value; then rebuilds the SAME test with a
GENUINELY fractal + resonant hidden-variable generator (M2): the common phase of
a locally-coupled Kuramoto system on a THOL-nested (Sierpinski) coherent core,
after it synchronizes -- fractal geometry AND measured resonance (R > 0), still
using only the canonical LOCAL per-neighbour coupling of dnfr.py's own gradient
channels (confirmed fresh: default_compute_delta_nfr sums ONLY over graph
neighbours; the one candidate "non-local" field, Phi_s, is confirmed a DIAGNOSTIC
-- conservation/gauge/telemetry bookkeeping -- never a term in the dynamics).

WHAT EMERGES (measured):
  - M1: a local, realistic hidden-variable model built on TNFR's own classical
    polarization angle gives |S| = 2.0000 (2e6-sample Monte Carlo) -- EXACTLY
    the classical (Bell) bound, not the quantum (Tsirelson) bound 2*sqrt(2) =
    2.8284 shown alongside for reference.
  - M2: rebuilding the hidden variable as the emergent common phase of a
    fractally-nested (THOL/Sierpinski), locally-coupled, resonantly-
    synchronizing Kuramoto system (measured order parameter R > 0.7 -- a
    genuinely fractal AND resonant generator, still local) gives THE SAME
    |S| = 2.0000. Fractality and resonance do not move the bound.

THE ANSWER (grounded in this measurement + Bell's theorem): probability
amplitude (in the Born-rule sense), the Born rule, and single-particle
interference statistics are OPEN -- and by Bell's theorem this is not a gap
still to be closed by further work, but a DIRECT mathematical consequence of
two properties TNFR's substrate ALREADY carries (Sec.5.1's finite causal cone,
Sec.5.3's classical un-entangled polarization): a LOCAL, CLASSICAL substrate
cannot reproduce correlations above |S| = 2, so it cannot reproduce the
entangled statistics on which the Born rule's empirical content rests. TNFR
DOES have real, derived classical-wave-interference (Sec.5, the light cone and
dispersion) and a genuine complex linear structure (Psi = K_phi + i J_phi,
Sec.5.3, Sec.7.1d) -- the classical PRECURSOR structure any linear field theory
needs -- but interpreting |Psi|^2 as a probability for individual, localized
detection events (rather than a classical energy/intensity density) is exactly
the piece Bell's theorem blocks while the substrate stays local and classical.

WHY FRACTALITY AND RESONANCE DO NOT HELP (M2). Bell's theorem constrains the
LOGICAL structure of the model -- locality (a wing's outcome depends only on
its own setting and the shared hidden variable, not on the other wing's
setting) and realism (the hidden variable has a definite value before any
"measurement") -- it places NO constraint on how that hidden variable is
GENERATED, so no amount of richness (a THOL-nested fractal core, a resonantly
synchronized Kuramoto phase) changes the bound: M2's fractal, resonant
generator gives the identical |S| = 2.0000. What WOULD move the bound is
abandoning one of the two premises outright -- genuine non-locality (breaking
Sec.5.1's already-derived finite causal cone) or a genuinely non-commuting
(Hilbert-space) observable algebra -- neither of which "more classical
structure", however fractal or resonant, supplies. A bonus, consistent
observation: fractal (anomalous-diffusion) geometry (Sec.3.2) makes
information spread SLOWER, not faster, than a regular lattice -- if anything it
tightens, not loosens, the locality premise.

THE FOURTH PIECE -- a physical hbar (a units question, not an empirical one).
TNFR's native unit is Hz_str (a structural rate, Sec.1); hbar is a dimensionful
EMPIRICAL conversion constant (J*s) linking a physical action to physical
angular momentum. There is no analog of a physical action scale in Hz_str to
calibrate against without importing one from outside -- which the emergent-
first rule (Sec.1) forbids. This is not a gap current work could close; it is
a category mismatch (a structural rate has no physical-action partner to set
hbar's VALUE), addressed once, here, rather than tested.

HONEST SCOPE: Bell's theorem and its experimental confirmation (loophole-free
tests) are established physics, not new mathematics; the LHV model here is the
standard textbook construction (a deterministic threshold response to a shared
hidden angle) applied to TNFR's own polarization object, and to a genuinely
fractal + resonant local generator (M2) built from canonical TNFR pieces (the
THOL/Sierpinski nest, Kuramoto coupling). The TNFR content is that this
DECISIVE bound applies directly to the classical, local substrate ALREADY
derived in Sec.5.1/Sec.5.3, and stays put no matter how fractal or resonant the
local mechanism generating the hidden variable is -- sharpening Sec.9.2 from
"not yet done" to "blocked by a general theorem while the substrate stays local
and classical, regardless of its structural richness".
Closes no open problem (Bell's theorem is long settled); it locates TNFR's own
frontier precisely.

Run:
    python benchmarks/emergent_bell_inequality_bound.py

Theoretical anchor: theory/EMERGENT_ONTOLOGY.md Sec.5.1 (the causal cone),
Sec.5.3 (the classical Stokes/Poincare polarization), Sec.9.2 (the classical-
substrate frontier); src/tnfr/physics/symplectic_substrate.py
(polarization_density, the Poincare sphere).
Status: RESEARCH (a decisive, general boundary of the classical substrate).
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np


def _sierpinski_simplex(m: int, levels: int) -> nx.Graph:
    """THOL self-similar nesting of K_m -- a genuinely fractal coherent core."""
    if levels == 0:
        return nx.complete_graph(m)
    sub = _sierpinski_simplex(m, levels - 1)
    sub_corners = list(range(m)) if levels == 1 else list(sub.nodes())[:m]
    graph = nx.Graph()
    copies: list[list[tuple]] = []
    for i in range(m):
        mapping = {v: (i, v) for v in sub.nodes}
        graph.add_nodes_from(mapping[v] for v in sub.nodes)
        graph.add_edges_from((mapping[u], mapping[v]) for u, v in sub.edges)
        copies.append([mapping[c] for c in sub_corners])
    parent = {n: n for n in graph.nodes}

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for i in range(m):
        for j in range(i + 1, m):
            ra, rb = find(copies[i][j]), find(copies[j][i])
            if ra != rb:
                parent[rb] = ra
    merged = nx.Graph()
    for u, v in graph.edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            merged.add_edge(ru, rv)
    return merged


def fractal_resonant_hidden_variable(
    rng: np.random.Generator,
    *,
    m: int = 4,
    levels: int = 2,
    coupling: float = 2.0,
    steps: int = 150,
    dt: float = 0.05,
) -> tuple[float, float]:
    """The common phase of a LOCALLY-coupled Kuramoto system on a fractal
    (THOL-nested Sierpinski) coherent core, after it synchronizes.

    Each node updates from its OWN graph neighbours only (the same locality
    canonical TNFR dynamics uses, dnfr.py) -- this is a genuinely fractal
    (self-similar nesting) AND resonant (Kuramoto phase-locking) local hidden-
    variable generator. Returns (mean_phase, order_parameter_R).
    """
    graph = _sierpinski_simplex(m, levels)
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)  # a fresh draw = "nature decides"
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for u, v in graph.edges():
        adjacency[index[u]].append(index[v])
        adjacency[index[v]].append(index[u])
    for _ in range(steps):
        updated = theta.copy()
        for i in range(n):
            neighbours = adjacency[i]
            if not neighbours:
                continue
            drive = sum(math.sin(theta[j] - theta[i]) for j in neighbours)
            updated[i] = theta[i] + dt * coupling * drive / len(neighbours)
        theta = updated
    order = complex(np.mean(np.exp(1j * theta)))
    return float(np.angle(order)), float(abs(order))


def chsh_local_hidden_variable(
    a: float,
    a_prime: float,
    b: float,
    b_prime: float,
    *,
    n: int = 2_000_000,
    seed: int = 0,
) -> tuple[float, float, float, float, float]:
    """CHSH value of a local, realistic model on a shared classical angle.

    ``lam`` is a single "hidden" polarization angle drawn once per trial (the
    same equatorial angle of the Poincare sphere that
    ``symplectic_substrate.polarization_density`` assigns each node) --
    realism: the angle is definite before any "measurement". Each wing applies
    a deterministic threshold response to its OWN analyzer setting only
    (locality). Returns (E_ab, E_abp, E_apb, E_apbp, S).
    """
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.0, 2.0 * math.pi, n)

    def wing_a(setting: float) -> np.ndarray:
        return np.sign(np.cos(2.0 * (lam - setting)))

    def wing_b(setting: float) -> np.ndarray:
        # the singlet-like convention: perfect anti-correlation at a = b
        return np.sign(np.cos(2.0 * (lam + math.pi / 2.0 - setting)))

    e_ab = float(np.mean(wing_a(a) * wing_b(b)))
    e_abp = float(np.mean(wing_a(a) * wing_b(b_prime)))
    e_apb = float(np.mean(wing_a(a_prime) * wing_b(b)))
    e_apbp = float(np.mean(wing_a(a_prime) * wing_b(b_prime)))
    s = e_ab - e_abp + e_apb + e_apbp
    return e_ab, e_abp, e_apb, e_apbp, s


def quantum_singlet_chsh(a: float, a_prime: float, b: float, b_prime: float) -> float:
    """The textbook quantum CHSH value E(x,y) = -cos(2(x-y)) -- for reference
    only, NOT computed from TNFR (the substrate cannot reach it, see M1)."""

    def e(x: float, y: float) -> float:
        return -math.cos(2.0 * (x - y))

    return e(a, b) - e(a, b_prime) + e(a_prime, b) + e(a_prime, b_prime)


def main() -> None:
    print("=" * 74)
    print("BELL/CHSH BOUND on TNFR's own classical polarization sector")
    print("=" * 74)

    # standard CHSH test angles (the ones that saturate the quantum bound)
    a, a_prime = 0.0, math.pi / 4.0
    b, b_prime = math.pi / 8.0, 3.0 * math.pi / 8.0

    print("\n[M1] local hidden-variable model on TNFR's own polarization angle.")
    e_ab, e_abp, e_apb, e_apbp, s = chsh_local_hidden_variable(a, a_prime, b, b_prime)
    print(f"     E(a,b)   = {e_ab:+.4f}")
    print(f"     E(a,b')  = {e_abp:+.4f}")
    print(f"     E(a',b)  = {e_apb:+.4f}")
    print(f"     E(a',b') = {e_apbp:+.4f}")
    print(f"     CHSH  S  = {s:+.4f}")
    print("     classical (Bell) bound   : |S| <= 2.0000")
    s_q = quantum_singlet_chsh(a, a_prime, b, b_prime)
    print(f"     quantum (Tsirelson) bound: |S| <= {2 * math.sqrt(2):.4f}")
    print(f"     (quantum prediction at these angles, for reference: {s_q:+.4f})")
    assert abs(abs(s) - 2.0) < 0.01, "local-realistic model exceeded the Bell bound"
    print("     -> PASS: the local, realistic model on TNFR's own classical")
    print("        polarization angle saturates the CLASSICAL bound exactly --")
    print("        it does not reach the quantum bound.")

    # -- M2: a genuinely FRACTAL + RESONANT local hidden variable --------------
    print("\n[M2] a fractal (THOL-nested) + resonant (Kuramoto) hidden variable.")
    rng = np.random.default_rng(0)
    n_trials = 4000
    lam = np.empty(n_trials)
    order_r = np.empty(n_trials)
    for t in range(n_trials):
        lam[t], order_r[t] = fractal_resonant_hidden_variable(rng)
    print(f"     mean Kuramoto order parameter R = {order_r.mean():.4f}  "
          "(resonant synchrony, R>0)")

    def wing_a2(setting: float) -> np.ndarray:
        return np.sign(np.cos(2.0 * (lam - setting)))

    def wing_b2(setting: float) -> np.ndarray:
        return np.sign(np.cos(2.0 * (lam + math.pi / 2.0 - setting)))

    e_ab2 = float(np.mean(wing_a2(a) * wing_b2(b)))
    e_abp2 = float(np.mean(wing_a2(a) * wing_b2(b_prime)))
    e_apb2 = float(np.mean(wing_a2(a_prime) * wing_b2(b)))
    e_apbp2 = float(np.mean(wing_a2(a_prime) * wing_b2(b_prime)))
    s2 = e_ab2 - e_abp2 + e_apb2 + e_apbp2
    print(f"     CHSH  S  = {s2:+.4f}  (fractal + resonant hidden variable)")
    assert abs(abs(s2) - 2.0) < 0.05, "fractal/resonant model exceeded the Bell bound"
    print("     -> PASS: fractality and resonance do not move the bound -- the")
    print("        theorem constrains the LOGICAL structure (locality + realism),")
    print("        not the richness of what generates the hidden variable.")

    print("\n" + "=" * 74)
    print("THE ANSWER: probability amplitude / Born rule / interference")
    print("statistics are OPEN because Bell's theorem (1964, confirmed by")
    print("loophole-free experiments) forbids them for ANY LOCAL + CLASSICAL")
    print("substrate -- both properties TNFR's substrate ALREADY carries")
    print("(Sec.5.1 finite causal cone, Sec.5.3 classical polarization). A")
    print("physical hbar is a separate, units-level question: Hz_str has no")
    print("physical-action partner to calibrate against without importing one.")
    print("=" * 74)


if __name__ == "__main__":
    main()

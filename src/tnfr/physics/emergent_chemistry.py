"""
TNFR Emergent Chemistry: Atomic Structure from Nodal Dynamics

Pure-TNFR derivation of atomic shell structure, the periodic table, and the
octet rule, following the canonical template established by the number-theory
layer (``tnfr.mathematics.number_theory``):

    intrinsic invariant -> structural triad -> ΔNFR -> equilibrium (ΔNFR = 0)

Nothing here is imported from external quantum chemistry. The quantum regime
is already a TNFR property (see ``tnfr.physics.quantum_mechanics``): a bounded
structural manifold supports only *discrete resonant eigenmodes*. We use that
already-emergent fact as the foundation:

  1. EIGENMODES EMERGE. The resonant modes of a closed structural manifold are
     the eigenvalues of the structural Laplacian L = D - A (the discrete ΔNFR /
     phase-curvature operator). On a 2-sphere manifold the spectrum clusters
     into degenerate groups of multiplicity (2l+1) = 1, 3, 5, 7, ... — the
     angular eigenmodes. This is computed numerically, not postulated.

  2. SHELLS EMERGE. Grouping subshell capacities 2*(2l+1) and ordering them by
     total structural excitation νf ∝ (n + l) (the structural reading of the
     aufbau order) yields cumulative closed-shell counts (magic numbers).

  3. OCTET RULE = ΔNFR = 0. The structural valence pressure ΔNFR_chem(Z)
     vanishes exactly at closed-shell (noble-like) configurations, in direct
     analogy with the primality criterion ΔNFR(n) = 0. Reactivity is |ΔNFR|.

This layer has NO free scale parameters (audit 2026 redesign). The structural
valence pressure ΔNFR_chem(Z) is the INTEGER structural distance of the outer
shell to a closed configuration — the chemical analogue of the primality
criterion ΔNFR(n) = 0, in natural units (one subshell step = 1). The (n+l)
filling order is a pure integer excitation-count rule (total radial+angular
quanta), not a constant correspondence. The earlier γ/π "excitation scale" and
the 1/φ valence weight were overlay/notational factors (refuted 2026: only π is
a genuine structural scale) and are removed. The atomic number Z is an
*emergent count of filled structural eigenmodes*, not an imported constant.

Honest scope:
  - The (2l+1) degeneracy is a rigorous numerical consequence of the manifold
    Laplacian (Laplace–Beltrami spectrum on the sphere).
  - The aufbau (n+l) ordering is an integer EXCITATION-COUNT rule (total
    radial + angular quanta), not a constant correspondence. It was tested
    against the raw Laplacian spectrum of a concentric multi-shell manifold
    (audit 2026): the free spectrum does NOT reproduce the (n+l) order, because
    Madelung ordering reflects electron-electron screening that is absent from
    a free graph Laplacian. The (n+l) count is therefore retained as an
    explicit structural rule — the minimal integer ordering matching the
    empirical noble gases (2, 10, 18, 36, 54, 86) — flagged as a count rule,
    not a spectral derivation.

Theoretical foundation: AGENTS.md (nodal equation, tetrad, discrete-mode
regime), theory/TNFR_NUMBER_THEORY.md (ΔNFR = 0 equilibrium template).

Status: RESEARCH (pure-TNFR derivation; mirrors the number-theory canonical
pattern but the aufbau ordering assumption is explicitly non-derived).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from ..mathematics.unified_numerical import np

# ============================================================================
# STRUCTURAL CONSTANTS (integer eigenmode counts — no free scale parameters)
# ============================================================================


# Subshell capacity: 2*(2l+1) = number of distinct ± phase-winding eigenmodes
# at angular index l. l = 0(s), 1(p), 2(d), 3(f).
_SUBSHELL_CAPACITY = {0: 2, 1: 6, 2: 10, 3: 14}
_SUBSHELL_LABEL = {0: "s", 1: "p", 2: "d", 3: "f"}


# ============================================================================
# STEP 1 — EIGENMODES EMERGE FROM THE STRUCTURAL MANIFOLD LAPLACIAN
# ============================================================================


def fibonacci_sphere_graph(n_points: int = 162, k_neighbors: int = 6) -> nx.Graph:
    """Build a closed structural manifold: points on S² (fibonacci spiral)
    connected to their k nearest neighbors.

    The resulting graph approximates the 2-sphere; its structural Laplacian
    spectrum approximates the Laplace–Beltrami spectrum, whose eigenvalues
    l(l+1) carry degeneracy (2l+1).

    Parameters
    ----------
    n_points : int
        Number of nodes on the sphere manifold.
    k_neighbors : int
        Nearest-neighbor connectivity (manifold smoothness).
    """
    if n_points < 4:
        raise ValueError("n_points must be >= 4 to resolve angular modes")

    # Fibonacci sphere point distribution
    idx = np.arange(n_points, dtype=float)
    phi_golden = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    z = 1.0 - 2.0 * (idx + 0.5) / n_points
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = phi_golden * idx
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)

    G = nx.Graph()
    for i in range(n_points):
        G.add_node(i, pos=tuple(float(c) for c in pts[i]))

    # k-nearest-neighbor connectivity
    for i in range(n_points):
        d = np.linalg.norm(pts - pts[i], axis=1)
        d[i] = np.inf
        nearest = np.argsort(d)[:k_neighbors]
        for j in nearest:
            G.add_edge(i, int(j))
    return G


@dataclass(frozen=True)
class EigenmodeShell:
    """A degenerate group of structural eigenmodes (an angular shell)."""

    multiplicity: int
    eigenvalue: float
    angular_index: int  # inferred l from (2l+1) = multiplicity


def structural_eigenmodes(
    G: nx.Graph, *, max_modes: int = 16, gap_factor: float = 6.0
) -> list[EigenmodeShell]:
    """Compute the resonant eigenmodes of the structural manifold and group
    them into degenerate shells.

    The structural Laplacian L = D - A is the discrete phase-curvature / ΔNFR
    operator. Its low-lying eigenvalues cluster into groups whose sizes are the
    angular multiplicities (2l+1) = 1, 3, 5, 7, ...

    Degenerate groups are separated by gaps in the spectrum: a shell boundary
    occurs where a consecutive eigenvalue gap exceeds ``gap_factor`` times the
    typical (median) intra-shell spacing.

    Returns the detected shells (degenerate groups) in ascending energy order.
    """
    L = nx.laplacian_matrix(G).toarray().astype(float)
    evals = np.sort(np.linalg.eigvalsh(L))[:max_modes]
    gaps = np.diff(evals)
    positive = gaps[gaps > 1e-9]
    typical = float(np.median(positive)) if positive.size else 1e-9
    threshold = gap_factor * typical

    shells: list[EigenmodeShell] = []
    group: list[float] = [float(evals[0])]
    for i, ev in enumerate(evals[1:]):
        if gaps[i] > threshold:
            mult = len(group)
            shells.append(
                EigenmodeShell(mult, float(np.mean(group)), (mult - 1) // 2)
            )
            group = [float(ev)]
        else:
            group.append(float(ev))
    mult = len(group)
    shells.append(EigenmodeShell(mult, float(np.mean(group)), (mult - 1) // 2))
    return shells


# ============================================================================
# STEP 2 — SHELLS, FILLING ORDER, AND EMERGENT MAGIC NUMBERS
# ============================================================================


def aufbau_subshell_order(max_n: int = 7) -> list[tuple[int, int]]:
    """Subshells (n, l) ordered by total structural excitation νf ∝ (n + l),
    then by n. This is the structural reading of the Madelung/aufbau rule.

    ASSUMPTION (flagged): the (n+l, n) ordering is structurally motivated by
    νf ∝ (n + l) but is not derived variationally from the nodal equation.

    Only l in {0,1,2,3} (s, p, d, f) exist as bound structural subshells, the
    same angular indices that survive as low-lying eigenmodes of the manifold
    Laplacian; higher l are not realised.
    """
    pairs = [(n, ell) for n in range(1, max_n + 1) for ell in range(0, min(n, 4))]
    pairs.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return pairs


def electron_configuration(Z: int, *, max_n: int = 7) -> list[tuple[int, int, int]]:
    """Fill Z structural excitations into eigenmode subshells (aufbau order).

    Returns a list of (n, l, occupation) triples in filling order.
    """
    if Z < 1:
        raise ValueError("Z must be >= 1")
    remaining = Z
    config: list[tuple[int, int, int]] = []
    for n, ell in aufbau_subshell_order(max_n=max_n):
        if remaining <= 0:
            break
        cap = _SUBSHELL_CAPACITY[ell]
        occ = min(cap, remaining)
        config.append((n, ell, occ))
        remaining -= occ
    if remaining > 0:
        raise ValueError(f"Z={Z} exceeds capacity of max_n={max_n} shells")
    return config


def emergent_magic_numbers(max_n: int = 7) -> list[int]:
    """Cumulative closed-shell counts that emerge from eigenmode filling.

    A closed shell (large structural gap) occurs after completing an l=1 (p)
    subshell, or after 1s for the first shell. The resulting numbers are the
    emergent noble-gas Z values.
    """
    magic: list[int] = []
    total = 0
    for n, ell in aufbau_subshell_order(max_n=max_n):
        total += _SUBSHELL_CAPACITY[ell]
        if ell == 1 or (n == 1 and ell == 0):
            magic.append(total)
    return magic


# ============================================================================
# STEP 3 — OCTET RULE AS A ΔNFR = 0 STRUCTURAL EQUILIBRIUM
# ============================================================================


def _valence_electrons(config: list[tuple[int, int, int]]) -> tuple[int, int]:
    """Return (valence electron count, outermost principal index n).

    Valence = electrons in the highest occupied principal shell n.
    """
    n_max = max(n for n, _l, _o in config)
    v = sum(o for n, _l, o in config if n == n_max)
    return v, n_max


def valence_delta_nfr(
    Z: int,
    *,
    max_n: int = 7,
) -> float:
    """Structural valence pressure ΔNFR_chem(Z).

    ΔNFR_chem = d(Z), the INTEGER structural distance of the outermost shell
    to a closed configuration, in natural units (one subshell step = 1).
    d(Z) = 0 *iff* the outer shell is a closed duet (n=1) or octet (n>1):

        Z is noble-like  ⟺  ΔNFR_chem(Z) = 0

    in direct analogy with the primality criterion ΔNFR(n) = 0. There is no
    free scale parameter (audit 2026 redesign).
    """
    config = electron_configuration(Z, max_n=max_n)
    v, n_max = _valence_electrons(config)
    target = 2 if n_max == 1 else 8
    v_eff = v % target
    # Structural distance to the nearest closed shell (gain vs. loss symmetry),
    # in natural units (one subshell step = 1). No free scale parameter.
    dist = min(v_eff, target - v_eff) if v_eff != 0 else 0
    return float(dist)


@dataclass(frozen=True)
class EmergentElement:
    """Structural characterization of an emergent element (count Z)."""

    Z: int
    configuration: tuple[tuple[int, int, int], ...]
    valence_electrons: int
    outer_shell_n: int
    delta_nfr: float
    closed_shell: bool
    magic_number: bool
    reactivity: float  # |ΔNFR| (0 = inert)
    config_label: str

    def as_dict(self) -> dict[str, object]:
        return {
            "Z": self.Z,
            "configuration": [list(t) for t in self.configuration],
            "valence_electrons": self.valence_electrons,
            "outer_shell_n": self.outer_shell_n,
            "delta_nfr": self.delta_nfr,
            "closed_shell": self.closed_shell,
            "magic_number": self.magic_number,
            "reactivity": self.reactivity,
            "config_label": self.config_label,
        }


def classify_element(
    Z: int,
    *,
    max_n: int = 7,
) -> EmergentElement:
    """Full pure-TNFR structural classification of element with count Z."""
    config = electron_configuration(Z, max_n=max_n)
    v, n_max = _valence_electrons(config)
    dnfr = valence_delta_nfr(Z, max_n=max_n)
    # Closed shell = the canonical nodal-equation fixed point ΔNFR = 0, read out
    # on the chemical valence-pressure field -- the same equilibrium criterion
    # as the structural prime and the relaxed graph node.
    from ..metrics.common import is_structural_equilibrium

    closed = is_structural_equilibrium(dnfr, eps_dnfr=1e-12)
    magic = Z in emergent_magic_numbers(max_n=max_n)
    label = " ".join(
        f"{n}{_SUBSHELL_LABEL[l]}{o}" for n, l, o in config
    )
    return EmergentElement(
        Z=Z,
        configuration=tuple(config),
        valence_electrons=v,
        outer_shell_n=n_max,
        delta_nfr=dnfr,
        closed_shell=closed,
        magic_number=magic,
        reactivity=abs(dnfr),
        config_label=label,
    )


__all__ = [
    "EigenmodeShell",
    "EmergentElement",
    "fibonacci_sphere_graph",
    "structural_eigenmodes",
    "aufbau_subshell_order",
    "electron_configuration",
    "emergent_magic_numbers",
    "valence_delta_nfr",
    "classify_element",
]

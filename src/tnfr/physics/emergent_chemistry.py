"""Structural shell-model correspondence built from explicit assumptions.

This module combines three ingredients of different status:

1. ``fibonacci_sphere_graph`` constructs an externally chosen S² embedding and
   a nearest-neighbour graph. Its normalized-Laplacian spectrum is then grouped
   numerically into approximate degeneracy clusters.
2. capacities ``2(2l+1)``, the Madelung ``(n+l, n)`` ordering and the choice of
   duet/octet closures are declared shell-model rules. They are not derived
   from the TNFR nodal equation.
3. ``shell_closure_distance`` defines an integer distance to those selected closures
   and reuses the shared zero-pressure predicate. This is a domain-specific
   diagnostic, not a chemical force or a dynamical attractor theorem.

The code provides a reproducible comparison with familiar shell counts while
making each assumption inspectable. It does not derive atomic structure, the
periodic table, quantum mechanics, electron interactions or physical
reactivity. The arithmetic and chemical constructions share only the abstract
``ΔNFR = 0`` predicate; their state spaces and dynamics are different.

Theoretical foundation: AGENTS.md (nodal equation, tetrad, discrete-mode
regime), theory/TNFR_NUMBER_THEORY.md (ΔNFR = 0 equilibrium template).

Status: RESEARCH (assumption-explicit shell-model correspondence).
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..mathematics.unified_numerical import np

# ============================================================================
# DECLARED SHELL-MODEL COUNTS
# ============================================================================


# Conventional capacity 2*(2l+1); both the factor two and shell interpretation
# are model assumptions. l = 0(s), 1(p), 2(d), 3(f).
_SUBSHELL_CAPACITY = {0: 2, 1: 6, 2: 10, 3: 14}
_SUBSHELL_LABEL = {0: "s", 1: "p", 2: "d", 3: "f"}


def _require_integral_count(value: Any, *, name: str, minimum: int = 1) -> int:
    """Return an exact integer count and reject truth values or truncation."""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    normalized = int(normalized)
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def _require_positive_finite(value: Any, *, name: str) -> float:
    """Return a positive finite scalar without accepting truth values."""
    if isinstance(value, bool) or type(value).__name__ == "bool_" or isinstance(
        value, (str, bytes)
    ):
        raise TypeError(f"{name} must be a positive finite real scalar")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive finite real scalar") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return normalized


# ============================================================================
# STEP 1 — EIGENMODES OF A CHOSEN DISCRETE S² GRAPH
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
    n_points = _require_integral_count(n_points, name="n_points", minimum=4)
    k_neighbors = _require_integral_count(k_neighbors, name="k_neighbors")
    if k_neighbors >= n_points:
        raise ValueError("k_neighbors must be smaller than n_points")

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
    if not nx.is_connected(G):
        raise ValueError(
            "k_neighbors produces a disconnected graph; increase it for the S2 comparison"
        )
    return G


@dataclass(frozen=True)
class EigenmodeShell:
    """A numerical cluster of normalized-Laplacian eigenmodes."""

    multiplicity: int
    eigenvalue: float
    angular_index: int | None  # odd-multiplicity candidate; unresolved if even


def structural_eigenmodes(
    G: nx.Graph, *, max_modes: int = 16, gap_factor: float = 6.0
) -> list[EigenmodeShell]:
    """Cluster normalized-Laplacian modes of a supplied graph.

    ``fibonacci_sphere_graph`` supplies an S²-like geometry by construction.
    This function reads the spectrum through the symmetric normalized
    Laplacian and groups nearby ``sqrt(lambda)`` values using ``gap_factor``.
    The resulting multiplicities are finite-resolution measurements, and the
    reported angular index is inferred from a group's size rather than proved
    from the graph.

    Degenerate groups are separated by gaps in the frequency spectrum: a shell
    boundary occurs where a consecutive frequency gap exceeds ``gap_factor``
    times the typical (median) intra-shell frequency spacing.

    Returns the detected shells (degenerate groups) in ascending frequency order.
    """
    max_modes = _require_integral_count(max_modes, name="max_modes")
    gap_factor = _require_positive_finite(gap_factor, name="gap_factor")
    if G.number_of_nodes() == 0:
        raise ValueError("G must contain at least one node")

    # Symmetric normalized Laplacian L_sym; it shares the L_rw spectrum.
    from .structural_diffusion import symmetric_normalized_laplacian

    _, L = symmetric_normalized_laplacian(G)
    evals = np.sort(np.clip(np.linalg.eigvalsh(L), 0.0, None))[:max_modes]
    # Auxiliary standing-wave coordinate ω_k = sqrt(λ_k). Group boundaries are
    # a finite numerical protocol and depend on gap_factor and graph resolution.
    freqs = np.sqrt(evals)
    gaps = np.diff(freqs)
    positive = gaps[gaps > 1e-9]
    typical = float(np.median(positive)) if positive.size else 1e-9
    threshold = gap_factor * typical

    shells: list[EigenmodeShell] = []
    group: list[float] = [float(evals[0])]
    for i, ev in enumerate(evals[1:]):
        if gaps[i] > threshold:
            mult = len(group)
            angular = (mult - 1) // 2 if mult % 2 == 1 else None
            shells.append(EigenmodeShell(mult, float(np.mean(group)), angular))
            group = [float(ev)]
        else:
            group.append(float(ev))
    mult = len(group)
    angular = (mult - 1) // 2 if mult % 2 == 1 else None
    shells.append(EigenmodeShell(mult, float(np.mean(group)), angular))
    return shells


# ============================================================================
# STEP 2 — ASSUMED SHELL CAPACITIES, FILLING ORDER AND CLOSURES
# ============================================================================


def aufbau_subshell_order(max_n: int = 7) -> list[tuple[int, int]]:
    """Subshells (n, l) ordered by total structural excitation νf ∝ (n + l),
    then by n. This is the structural reading of the Madelung/aufbau rule.

    ASSUMPTION (flagged): the (n+l, n) ordering is structurally motivated by
    νf ∝ (n + l) but is not derived variationally from the nodal equation.

    This finite model truncates the declared labels to l in {0,1,2,3}
    (s, p, d, f). It makes no claim that higher angular sectors do not exist.
    """
    max_n = _require_integral_count(max_n, name="max_n")
    pairs = [(n, ell) for n in range(1, max_n + 1) for ell in range(0, min(n, 4))]
    pairs.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return pairs


def electron_configuration(Z: int, *, max_n: int = 7) -> list[tuple[int, int, int]]:
    """Fill Z count units into assumed subshell capacities (aufbau order).

    Returns a list of (n, l, occupation) triples in filling order.
    """
    Z = _require_integral_count(Z, name="Z")
    max_n = _require_integral_count(max_n, name="max_n")
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
    """Cumulative closures produced by the declared shell rules.

    By declared rule, a closure is recorded after completing an l=1 (p)
    subshell, or after 1s for the first shell. No spectral-gap calculation is
    performed by this count generator.
    """
    max_n = _require_integral_count(max_n, name="max_n")
    magic: list[int] = []
    total = 0
    for n, ell in aufbau_subshell_order(max_n=max_n):
        total += _SUBSHELL_CAPACITY[ell]
        if ell == 1 or (n == 1 and ell == 0):
            magic.append(total)
    return magic


# ============================================================================
# STEP 3 — SELECTED DUET/OCTET DISTANCE AS A ZERO-PRESSURE READ-OUT
# ============================================================================


def _valence_electrons(config: list[tuple[int, int, int]]) -> tuple[int, int]:
    """Return (valence electron count, outermost principal index n).

    Valence = electrons in the highest occupied principal shell n.
    """
    n_max = max(n for n, _l, _o in config)
    v = sum(o for n, _l, o in config if n == n_max)
    return v, n_max


def shell_closure_distance(
    Z: int,
    *,
    max_n: int = 7,
) -> float:
    """Count distance from ``Z`` to the nearest declared shell closure.

    ``ΔNFR_chem = d(Z)`` is the integer count distance to the nearest closure
    generated by :func:`emergent_magic_numbers`. Therefore ``d(Z) = 0`` iff
    ``Z`` is in that same declared closure set:

        Z is noble-like  ⟺  ΔNFR_chem(Z) = 0

    by definition of this domain model. Reusing the zero-pressure predicate is
    an analogy with the arithmetic criterion, not a shared dynamics.
    """
    Z = _require_integral_count(Z, name="Z")
    max_n = _require_integral_count(max_n, name="max_n")
    electron_configuration(Z, max_n=max_n)  # validate modeled capacity
    closures = emergent_magic_numbers(max_n=max_n)
    return float(min(abs(Z - closure) for closure in closures))


def valence_delta_nfr(
    Z: int,
    *,
    max_n: int = 7,
) -> float:
    """Compatibility alias for :func:`shell_closure_distance`."""
    return shell_closure_distance(Z, max_n=max_n)


@dataclass(frozen=True)
class EmergentElement:
    """Structural shell-model characterization for count ``Z``."""

    Z: int
    configuration: tuple[tuple[int, int, int], ...]
    valence_electrons: int
    outer_shell_n: int
    delta_nfr: float
    closed_shell: bool
    magic_number: bool
    reactivity: float  # legacy alias for closure_distance; no physical rate
    config_label: str

    @property
    def closure_distance(self) -> float:
        """Distance in count units to the nearest declared shell closure."""
        return abs(self.delta_nfr)

    def as_dict(self) -> dict[str, object]:
        return {
            "Z": self.Z,
            "configuration": [list(t) for t in self.configuration],
            "valence_electrons": self.valence_electrons,
            "outer_shell_n": self.outer_shell_n,
            "delta_nfr": self.delta_nfr,
            "closed_shell": self.closed_shell,
            "magic_number": self.magic_number,
            "closure_distance": self.closure_distance,
            "reactivity": self.reactivity,
            "config_label": self.config_label,
        }


def classify_element(
    Z: int,
    *,
    max_n: int = 7,
) -> EmergentElement:
    """Return the assumption-explicit shell-model classification for ``Z``."""
    Z = _require_integral_count(Z, name="Z")
    max_n = _require_integral_count(max_n, name="max_n")
    config = electron_configuration(Z, max_n=max_n)
    v, n_max = _valence_electrons(config)
    closures = emergent_magic_numbers(max_n=max_n)
    dnfr = shell_closure_distance(Z, max_n=max_n)
    # Apply the shared numerical zero-pressure predicate to this independently
    # defined shell-distance field. No chemical relaxation law is implied.
    from ..metrics.common import is_structural_equilibrium

    closed = is_structural_equilibrium(dnfr, eps_dnfr=1e-12)
    magic = Z in closures
    label = " ".join(f"{n}{_SUBSHELL_LABEL[l]}{o}" for n, l, o in config)
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


def classify_element_observation(Z: int, *, max_n: int = 7):
    """Return an element classification with explicit chemical provenance."""
    from ..metrics.observations import observe_emergent_element

    return observe_emergent_element(classify_element(Z, max_n=max_n))


__all__ = [
    "EigenmodeShell",
    "EmergentElement",
    "fibonacci_sphere_graph",
    "structural_eigenmodes",
    "aufbau_subshell_order",
    "electron_configuration",
    "emergent_magic_numbers",
    "shell_closure_distance",
    "valence_delta_nfr",
    "classify_element",
    "classify_element_observation",
]

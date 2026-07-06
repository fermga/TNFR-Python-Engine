r"""TNFR-Riemann canonical foundation: the prime-NFR nodal pulse.

This module re-founds the TNFR-Riemann attack on the **canonical emergent
nodal dynamics**, replacing the obsolete combinatorial-Laplacian Schrödinger
operator ``H(σ) = L_k + V_σ`` (a non-canonical graph-Laplacian prototype).

Canonical mapping (from the nodal equation ∂EPI/∂t = νf·ΔNFR)
============================================================
Each integer ``n`` is a fractal-resonant node (NFR) whose canonical structural
frequency is

.. math::

    \nu_{f,n} = \log n

(so ``νf`` is *additive* under multiplication, ``νf(pq)=νf(p)+νf(q)``, the
emergent image of the Euler product). On the coherence axis ``Re(s) = 1/2`` the
Riemann zeta function is the **superposition of the integer-NFR nodal pulses**

.. math::

    \zeta(\tfrac12 + iT) \;=\; \sum_{n\ge 1} n^{-1/2}\, e^{-i\,\nu_{f,n}\,T},

each term ``e^{-i νf_n T}`` being the nodal pulse of NFR ``n`` advancing at its
own structural frequency ``νf_n`` (phase ``φ_n(T) = νf_n·T``). The non-trivial
zeros are the heights ``T`` at which the integer-NFR pulses **totally
destructively interfere** (``|ζ| → 0``) — a coherence collapse of the
prime-pulse network.  RH is the statement that this collapse happens only on
the **coherence axis** ``Re(s) = 1/2`` — the fixed point of the
functional-equation reflection ``s ↔ 1-s`` (the ``ΔNFR = 0`` axis of a
reflection-symmetric NFR).

What is canonical here
----------------------
* The structural frequency ``νf_n = log n`` is the diagonal of the canonical
  internal Hamiltonian (see :mod:`tnfr.riemann.prime_ladder_hamiltonian`, P14);
  it is the *emergent* prime content, not a graph-Laplacian eigenvalue.
* The emergent structural operator on the prime graph is the random-walk
  Laplacian ``L_rw = I − D⁻¹W`` (the ΔNFR EPI channel; see
  :mod:`tnfr.physics.structural_diffusion`), **never** the imposed combinatorial
  ``D − A``.
* ``mpmath.zeta`` / :data:`KNOWN_RIEMANN_ZEROS` are used only as the numerical
  **oracle** against which the emergent pulse is verified.

Honest scope
------------
Reproducing the zeros as pulse-interference dips is a *structural picture* of
the classical Dirichlet/Riemann-Siegel behaviour; it does **not** prove RH
(G4). The residual is the oscillatory ``S(T) = (1/π) arg ζ(1/2+iT)`` — the open
arithmetic content (see the paused-program notes).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..mathematics.unified_numerical import np

try:  # networkx is optional; only the emergent prime-graph helper needs it
    import networkx as nx
except ImportError:  # pragma: no cover - networkx is a core dep in practice
    nx = None

__all__ = [
    "KNOWN_RIEMANN_ZEROS",
    "first_primes",
    "prime_structural_frequencies",
    "nodal_pulse",
    "nodal_pulse_magnitude",
    "detect_zeros_by_interference",
    "build_prime_nfr_graph",
    "NodalPulseCertificate",
    "verify_nodal_pulse",
]


# Canonical reference ordinates: imaginary parts of the first non-trivial
# Riemann zeros. The ORACLE against which the emergent nodal pulse is verified
# (moved here from the eliminated complex_extension module).
KNOWN_RIEMANN_ZEROS: tuple[float, ...] = (
    14.134725141734693,
    21.022039638771555,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
    48.005150881167160,
    49.773832477672302,
    52.970321477714460,
    56.446247697063394,
    59.347044002602353,
    60.831778524609809,
    65.112544048081607,
    67.079810529494174,
    69.546401711173980,
    72.067157674481907,
    75.704690699083933,
    77.144840068874805,
)


def first_primes(count: int) -> list[int]:
    """Return the first ``count`` prime numbers (canonical prime-NFR labels).

    Deliberately minimalist trial division; not optimised for large ``count``.
    Moved here from the eliminated combinatorial ``operator`` module so the
    canonical νf-based modules no longer depend on the obsolete track.
    """
    if count <= 0:
        return []
    primes: list[int] = []
    n = 2
    while len(primes) < count:
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
        n += 1
    return primes


def prime_structural_frequencies(count: int) -> list[float]:
    """Canonical structural frequencies ``νf = log p`` of the first ``count``
    prime NFRs."""
    return [float(np.log(p)) for p in first_primes(count)]


def _default_terms(t: float) -> int:
    """Canonical truncation length for the nodal pulse at height ``t``.

    A few times the Riemann-Siegel main-sum length ``√(t/2π)`` — enough integer
    NFRs for the destructive-interference dips to resolve the low zeros while
    staying short of the Dirichlet-partial-sum divergence.
    """
    rs = np.sqrt(max(t, 1.0) / (2.0 * np.pi))
    return int(max(10, round(3.0 * rs + 6.0)))


def nodal_pulse(t: float, n_terms: int | None = None) -> complex:
    r"""The integer-NFR nodal-pulse superposition at height ``t``.

    ``P(t) = Σ_{n=1}^{n_terms} n^{-1/2} e^{-i (log n) t}`` — the coherent
    superposition of the first ``n_terms`` integer-NFR pulses on the coherence
    axis ``Re(s) = 1/2`` (the truncated Dirichlet partial sum of
    ``ζ(1/2 + it)``). ``|P|`` collapses where the pulses destructively interfere.
    """
    if n_terms is None:
        n_terms = _default_terms(t)
    n = np.arange(1, int(n_terms) + 1, dtype=float)
    return complex(np.sum(n**-0.5 * np.exp(-1j * t * np.log(n))))


def nodal_pulse_magnitude(t: float, n_terms: int | None = None) -> float:
    """``|P(t)|`` — the prime-pulse coherence amplitude; dips at the zeros."""
    return float(abs(nodal_pulse(t, n_terms)))


def detect_zeros_by_interference(
    t_min: float,
    t_max: float,
    *,
    resolution: int = 4000,
    n_terms: int | None = None,
    threshold: float = 0.75,
) -> list[float]:
    r"""Locate zeros as **destructive-interference dips** of the nodal pulse.

    Scans ``|P(t)|`` on ``[t_min, t_max]`` and returns the heights of the local
    minima that fall below ``threshold`` — the values of ``t`` where the
    integer-NFR pulses cancel (``ζ → 0``). This is the emergent structural
    read-out of the non-trivial zeros; for high-precision ordinates use the
    ``mpmath`` oracle.
    """
    ts = np.linspace(t_min, t_max, resolution)
    mag = np.array([nodal_pulse_magnitude(float(t), n_terms) for t in ts])
    dips: list[float] = []
    for i in range(1, len(ts) - 1):
        if mag[i] < mag[i - 1] and mag[i] < mag[i + 1] and mag[i] < threshold:
            dips.append(float(ts[i]))
    return dips


def build_prime_nfr_graph(count: int):
    """Build the emergent prime-NFR path graph on the first ``count`` primes.

    Nodes carry the canonical triad seed (``EPI``, ``vf = log p``, ``theta``);
    the emergent structural operator on this graph is the random-walk Laplacian
    ``L_rw`` (:func:`tnfr.physics.structural_diffusion.structural_diffusion_operator`),
    never the imposed combinatorial ``D − A``.
    """
    if nx is None:
        raise ImportError("networkx is required for build_prime_nfr_graph")
    primes = first_primes(count)
    G = nx.Graph()
    for i, p in enumerate(primes):
        G.add_node(i, label=int(p), EPI=0.0, vf=float(np.log(p)), theta=0.0)
    for i in range(len(primes) - 1):
        gap = float(np.log(primes[i + 1]) - np.log(primes[i]))
        G.add_edge(i, i + 1, weight=1.0 / gap if gap > 0 else 1.0)
    return G


@dataclass(frozen=True)
class NodalPulseCertificate:
    """Certificate that the emergent nodal pulse reproduces the Riemann zeros."""

    n_zeros: int
    detected: tuple[float, ...]
    reference: tuple[float, ...]
    max_abs_error: float
    mean_abs_error: float
    all_matched: bool

    def summary(self) -> str:
        status = "PASS" if self.all_matched else "PARTIAL"
        return (
            f"NodalPulseCertificate[{status}]: {len(self.detected)}/{self.n_zeros} "
            f"zeros reproduced as interference dips; "
            f"max|Δ|={self.max_abs_error:.3f}, mean|Δ|={self.mean_abs_error:.3f}"
        )


def verify_nodal_pulse(
    n_zeros: int = 6, *, tol: float = 0.5, resolution: int = 8000
) -> NodalPulseCertificate:
    r"""Verify the emergent nodal pulse reproduces the first ``n_zeros`` zeros.

    Detects the destructive-interference dips of ``|P(t)|`` up to just past the
    ``n_zeros``-th known ordinate and matches each reference ``γ_n`` to its
    nearest dip. ``all_matched`` iff every matched error is within ``tol``.
    """
    n_zeros = min(n_zeros, len(KNOWN_RIEMANN_ZEROS))
    reference = KNOWN_RIEMANN_ZEROS[:n_zeros]
    t_max = reference[-1] + 3.0
    dips = detect_zeros_by_interference(
        8.0, t_max, resolution=resolution, threshold=0.9
    )
    errors: list[float] = []
    matched: list[float] = []
    for gamma in reference:
        if dips:
            nearest = min(dips, key=lambda d: abs(d - gamma))
            errors.append(abs(nearest - gamma))
            matched.append(nearest)
        else:  # pragma: no cover - dips is non-empty in practice
            errors.append(float("inf"))
            matched.append(float("nan"))
    max_err = max(errors) if errors else float("inf")
    mean_err = float(np.mean(errors)) if errors else float("inf")
    return NodalPulseCertificate(
        n_zeros=n_zeros,
        detected=tuple(matched),
        reference=tuple(reference),
        max_abs_error=float(max_err),
        mean_abs_error=mean_err,
        all_matched=bool(max_err <= tol),
    )

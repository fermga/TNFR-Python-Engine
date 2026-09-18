r"""Finite arithmetic pulse comparisons in the TNFR-Riemann program.

This declared arithmetic construction replaces the historical comparison
surface ``H(σ) = L_k + V_σ``. It does not derive prime labels, frequencies,
amplitudes or autonomous phase evolution from the scalar nodal equation.

Declared arithmetic inputs
==========================
The construction assigns to integer label ``n`` the frequency

.. math::

    \nu_{f,n} = \log n

(so ``νf(pq)=νf(p)+νf(q)`` by the logarithm identity), amplitude ``n**(-1/2)``
and phase evolution ``exp(-i * log(n) * T)``. The evaluated finite sum is

.. math::

    P_N(T) = \sum_{n=1}^{N} n^{-1/2}\,e^{-i(\log n)T}.

The ordinary infinite Dirichlet series defines ``zeta(s)`` for ``Re(s)>1``;
outside that domain its analytic continuation must be distinguished from
unregularized partial sums (https://dlmf.nist.gov/25.2). In particular, the
displayed finite sum is not an identity for ``zeta(1/2+iT)`` or a certified
convergent approximation as ``N`` grows. The default length is a numerical
policy, not a Riemann-Siegel formula with a remainder bound.

Implementation scope
====================
* P14 likewise assigns ``k*log(p)`` to explicitly supplied prime ladders;
  reading the same diagonal back is not independent emergence of primes.
* The optional graph helper uses the canonical EPI transport operator
  ``L_rw = I − D⁻¹W`` through :mod:`tnfr.physics.structural_diffusion`.
  This finite pulse evaluator does not execute that graph or a nodal integrator.
* :data:`KNOWN_RIEMANN_ZEROS` supplies the finite comparison oracle, scan
  endpoint and nearest-dip matches. This is not a blind zero-location test.

Honest scope
------------
Finite interference dips can be compared with known ordinates. They do not
certify analytic zeros, physical coherent entities, a ``DeltaNFR=0`` axis, or
RH (G4). The classical functional-equation reflection axis does not supply a
TNFR pressure law or prove that an engine trajectory approaches it. Historical
measurements remain scoped by ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md``.
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
    """Return the first ``count`` primes as explicit arithmetic input labels.

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
    """Assign ``νf = log p`` to the explicitly enumerated first ``count`` primes."""
    return [float(np.log(p)) for p in first_primes(count)]


def _default_terms(t: float) -> int:
    """Historical finite-length policy for the pulse at height ``t``.

    A few times the Riemann-Siegel main-sum length ``√(t/2π)`` — enough integer
    terms for the reported low-height comparisons. This heuristic supplies
    neither analytic continuation nor an approximation error bound.
    """
    rs = np.sqrt(max(t, 1.0) / (2.0 * np.pi))
    return int(max(10, round(3.0 * rs + 6.0)))


def nodal_pulse(t: float, n_terms: int | None = None) -> complex:
    r"""Evaluate the finite integer-label pulse at height ``t``.

    ``P(t) = Σ_{n=1}^{n_terms} n^{-1/2} e^{-i (log n) t}`` — the coherent
    superposition with prescribed amplitudes and logarithmic frequencies.
    This partial sum does not converge to ``ζ(1/2+it)`` without an appropriate
    continuation or summation construction, which this function does not add.
    """
    if n_terms is None:
        n_terms = _default_terms(t)
    n = np.arange(1, int(n_terms) + 1, dtype=float)
    return complex(np.sum(n**-0.5 * np.exp(-1j * t * np.log(n))))


def nodal_pulse_magnitude(t: float, n_terms: int | None = None) -> float:
    """Return ``|P(t)|``; local dips are not certificates of analytic zeros."""
    return float(abs(nodal_pulse(t, n_terms)))


def detect_zeros_by_interference(
    t_min: float,
    t_max: float,
    *,
    resolution: int = 4000,
    n_terms: int | None = None,
    threshold: float = 0.75,
) -> list[float]:
    r"""Return sampled interference minima of the declared finite pulse.

    Scans ``|P(t)|`` on ``[t_min, t_max]`` and returns the heights of the local
    minima that fall below ``threshold``. The historical function name does
    not imply that each minimum is a zeta zero or that every zero is found.
    """
    ts = np.linspace(t_min, t_max, resolution)
    mag = np.array([nodal_pulse_magnitude(float(t), n_terms) for t in ts])
    dips: list[float] = []
    for i in range(1, len(ts) - 1):
        if mag[i] < mag[i - 1] and mag[i] < mag[i + 1] and mag[i] < threshold:
            dips.append(float(ts[i]))
    return dips


def build_prime_nfr_graph(count: int):
    """Build a prescribed path graph on the first ``count`` enumerated primes.

    Nodes carry a chosen triad seed (``EPI``, ``vf = log p``, ``theta``), and
    edge weights are assigned from logarithmic gaps. The structural EPI
    transport operator on this graph is the random-walk Laplacian
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
    """Finite nearest-dip comparison receipt, not an analytic-zero certificate."""

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
    r"""Compare finite pulse minima with the first ``n_zeros`` known ordinates.

    Detects the destructive-interference dips of ``|P(t)|`` up to just past the
    ``n_zeros``-th known ordinate and matches each reference ``γ_n`` to its
    nearest dip. ``all_matched`` iff every matched error is within ``tol``;
    matches need not be distinct and extra dips are not penalized.
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

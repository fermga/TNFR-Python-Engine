r"""P4: Complex-s extension — non-Hermitian TNFR-Riemann operator.

Extends the discrete TNFR operator to complex structural parameter s in C:

    H^(k)(s) = L_k + (s - 1/2) diag(log p_1, ..., log p_k),  s in C.

When s = sigma + i*t with t != 0, H(s) is non-Hermitian and possesses
complex eigenvalues.  The Riemann zeros lie at s = 1/2 + i*t_n, so the
critical line s = 1/2 + i*t is the natural domain for investigating
connections between TNFR spectral structure and zeta-function zeros.

Key analyses
------------
1. **Critical line scan**: eigenvalue spectra along s = 1/2 + it.
2. **Eigenvalue zero-crossings**: t values where |lambda_j(s)| ~ 0.
3. **Pseudo-spectrum**: epsilon-pseudospectrum sigma_eps(H(s))
   characterizing non-normal sensitivity.
4. **Resolvent analysis**: ||(zI - H(s))^{-1}|| pole structure.
5. **Non-Hermiticity measures**: departure from self-adjointness.

TNFR physics basis
-------------------
The nodal equation dEPI/dt = nu_f * DELTA_NFR(t) supports complex
structural frequencies.  The imaginary part Im(s) encodes oscillatory
(quantum-like) dynamics — the regime where Riemann zeros reside.
The real Laplacian L_k captures the prime-graph topology, while the
complex potential (s - 1/2) V_1 introduces phase rotation proportional
to log(p_i).  This is a direct extension of the structural equilibrium
theorem (P1) into the oscillatory regime.

Status: EXPERIMENTAL — Research prototype for TNFR-Riemann P4 program.

References
----------
- AGENTS.md: TNFR-Riemann Program Overview
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md sec. 7-16
- src/tnfr/riemann/spectral_proof.py: P1 Hermitian spectral analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..mathematics.unified_numerical import np
from .operator import build_tridiagonal_h_tnfr_complex

# ---------------------------------------------------------------------------
# First 20 known Riemann zeta zeros (imaginary parts), high precision.
# Source: LMFDB / Odlyzko tables.
# ---------------------------------------------------------------------------

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

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Data structures
    "ComplexEigenResult",
    "CriticalLineScan",
    "PseudoSpectrumResult",
    "ResolventAnalysis",
    "ComplexPlaneAnalysis",
    # Constants
    "KNOWN_RIEMANN_ZEROS",
    # Core spectral
    "compute_complex_eigenspectrum",
    "compute_complex_eigensystem",
    # Critical line
    "scan_critical_line",
    "find_eigenvalue_zero_crossings",
    # Pseudo-spectrum
    "compute_pseudospectrum",
    "compute_resolvent_norm",
    # Non-Hermiticity
    "analyze_non_hermiticity",
    # Integrated
    "compare_with_riemann_zeros",
    "run_complex_plane_analysis",
]

# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True)
class ComplexEigenResult:
    r"""Complex eigenvalue result for H(s) at a single s value.

    Attributes
    ----------
    k : int
        Number of primes (graph size).
    s : complex
        Complex structural parameter.
    eigenvalues : np.ndarray
        Complex eigenvalues sorted by real part, shape (k,).
    min_abs_eigenvalue : float
        min_j |lambda_j(s)|.
    condition_number : float
        Condition number of the eigenvector matrix V (kappa(V)).
        Large values indicate high sensitivity to perturbation
        (characteristic of non-normal operators).
    non_hermiticity : float
        ||H - H^dag|| / ||H|| in Frobenius norm.  Zero for Hermitian.
    """

    k: int
    s: complex
    eigenvalues: np.ndarray
    min_abs_eigenvalue: float
    condition_number: float
    non_hermiticity: float


@dataclass(frozen=True)
class CriticalLineScan:
    r"""Results from scanning s = 1/2 + it along the critical line.

    At each t value, all k eigenvalues of H^(k)(1/2 + it) are computed.
    The scan reveals spectral flow in the oscillatory regime and
    identifies t values where eigenvalues approach zero.

    Attributes
    ----------
    k : int
        Number of primes.
    t_values : np.ndarray
        Imaginary parts scanned, shape (n_t,).
    eigenvalue_matrix : np.ndarray
        Complex eigenvalues at each t, shape (n_t, k).
    min_abs_eigenvalue : np.ndarray
        min_j |lambda_j| at each t, shape (n_t,).
    non_hermiticity : np.ndarray
        Non-Hermiticity measure at each t, shape (n_t,).
    local_minima_t : np.ndarray
        t values where min |lambda_j| has local minima (potential
        zero-crossing candidates).
    local_minima_val : np.ndarray
        Values of min |lambda_j| at those local minima.
    """

    k: int
    t_values: np.ndarray
    eigenvalue_matrix: np.ndarray
    min_abs_eigenvalue: np.ndarray
    non_hermiticity: np.ndarray
    local_minima_t: np.ndarray
    local_minima_val: np.ndarray


@dataclass(frozen=True)
class PseudoSpectrumResult:
    r"""Pseudo-spectrum of H(s) on a grid around the eigenvalues.

    The epsilon-pseudospectrum is:

        sigma_eps(H) = {z in C : sigma_min(zI - H) < epsilon}

    where sigma_min denotes the smallest singular value.  For normal
    operators, sigma_eps is the union of epsilon-discs around eigenvalues.
    For non-normal operators (Im(s) != 0), it can be much larger.

    Attributes
    ----------
    k : int
        Number of primes.
    s : complex
        Complex structural parameter at which H is constructed.
    z_real : np.ndarray
        Real axis grid, shape (n_grid,).
    z_imag : np.ndarray
        Imaginary axis grid, shape (n_grid,).
    sigma_min_grid : np.ndarray
        Minimum singular value of (zI - H) on 2D grid,
        shape (n_imag, n_real).
    eigenvalues : np.ndarray
        Eigenvalues of H(s) for reference, shape (k,).
    """

    k: int
    s: complex
    z_real: np.ndarray
    z_imag: np.ndarray
    sigma_min_grid: np.ndarray
    eigenvalues: np.ndarray


@dataclass(frozen=True)
class ResolventAnalysis:
    r"""Resolvent norm along the critical line at a fixed probe z.

    For each t, computes ||(zI - H(1/2 + it))^{-1}|| where z is a
    fixed probe point.  Peaks in the resolvent norm indicate that z
    is close to the spectrum of H(1/2 + it).

    Attributes
    ----------
    k : int
        Number of primes.
    z_probe : complex
        Fixed probe point in the complex plane.
    t_values : np.ndarray
        Imaginary parts scanned, shape (n_t,).
    resolvent_norms : np.ndarray
        ||R(z, 1/2 + it)|| at each t, shape (n_t,).
    peak_t_values : np.ndarray
        t values where resolvent norm has local maxima.
    peak_norms : np.ndarray
        Resolvent norm values at those peaks.
    """

    k: int
    z_probe: complex
    t_values: np.ndarray
    resolvent_norms: np.ndarray
    peak_t_values: np.ndarray
    peak_norms: np.ndarray


@dataclass
class ComplexPlaneAnalysis:
    r"""Integrated P4 complex-s plane analysis.

    Combines critical line scan, zero-crossing detection, resolvent
    analysis, and comparison with known Riemann zeros into a single
    assessment.

    Attributes
    ----------
    k : int
        Number of primes.
    critical_line : CriticalLineScan
        Full critical line scan.
    riemann_comparison : list[tuple[float, float, float]]
        (t_candidate, nearest_riemann_zero, |difference|) for each
        eigenvalue minimum near a known zero.
    non_hermiticity_at_half : float
        ||H - H^dag|| / ||H|| at s = 1/2 + it=0 (should be ~0).
    mean_non_hermiticity : float
        Average non-Hermiticity along critical line.
    zero_crossing_count : int
        Number of eigenvalue near-zero events detected.
    summary : str
        Human-readable summary.
    """

    k: int = 0
    critical_line: CriticalLineScan | None = None
    riemann_comparison: list[tuple[float, float, float]] = field(default_factory=list)
    non_hermiticity_at_half: float = 0.0
    mean_non_hermiticity: float = 0.0
    zero_crossing_count: int = 0
    summary: str = ""


# ============================================================================
# Core Spectral Computation (Non-Hermitian)
# ============================================================================


def _build_dense_from_tridiag(
    d: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    """Reconstruct dense matrix from tridiagonal components."""
    k = len(d)
    H = np.diag(d)
    if k >= 2:
        H += np.diag(e, 1) + np.diag(e, -1)
    return H


def compute_complex_eigenspectrum(
    k: int,
    s: complex = 0.5 + 0j,
    *,
    weight_by_log_gap: bool = True,
) -> np.ndarray:
    r"""Compute eigenvalues of H_TNFR^(k)(s) for complex s.

    Uses ``numpy.linalg.eig`` (general eigensolver) since H(s) is
    non-Hermitian when Im(s) != 0.

    Parameters
    ----------
    k : int
        Number of primes (>= 2).
    s : complex
        Complex structural parameter.
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    np.ndarray
        Complex eigenvalues sorted by real part, shape (k,).
    """
    d, e, _ = build_tridiagonal_h_tnfr_complex(
        k,
        s,
        weight_by_log_gap=weight_by_log_gap,
    )
    H = _build_dense_from_tridiag(d, e)
    eigenvalues = np.linalg.eig(H)[0]
    # Sort by real part, then imaginary part for determinism
    idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
    return eigenvalues[idx]


def compute_complex_eigensystem(
    k: int,
    s: complex = 0.5 + 0j,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute eigenvalues and right eigenvectors of H_TNFR^(k)(s).

    For a non-Hermitian matrix, left and right eigenvectors differ.
    This returns right eigenvectors V such that H V = V diag(lambda).

    Returns
    -------
    (eigenvalues, eigenvectors)
        eigenvalues sorted by real part; eigenvectors[:, j] corresponds
        to eigenvalues[j].  Both may be complex.
    """
    d, e, _ = build_tridiagonal_h_tnfr_complex(
        k,
        s,
        weight_by_log_gap=weight_by_log_gap,
    )
    H = _build_dense_from_tridiag(d, e)
    eigenvalues, eigenvectors = np.linalg.eig(H)
    idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
    return eigenvalues[idx], eigenvectors[:, idx]


def analyze_non_hermiticity(
    k: int,
    s: complex = 0.5 + 0j,
    *,
    weight_by_log_gap: bool = True,
) -> ComplexEigenResult:
    r"""Compute eigenvalues and non-Hermiticity measures for H(s).

    Measures:
    - Eigenvalue spectrum (complex).
    - Eigenvector condition number kappa(V) = ||V|| ||V^{-1}||.
    - Non-Hermiticity: ||H - H^dag||_F / ||H||_F.

    Large kappa(V) signals non-normality: the eigenvectors are nearly
    linearly dependent and the spectrum is sensitive to perturbations
    (relevant for pseudo-spectral analysis).
    """
    d, e, _ = build_tridiagonal_h_tnfr_complex(
        k,
        s,
        weight_by_log_gap=weight_by_log_gap,
    )
    H = _build_dense_from_tridiag(d, e)

    eigenvalues, V = np.linalg.eig(H)
    idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues = eigenvalues[idx]

    # Condition number of eigenvector matrix
    sv = np.linalg.svd(V[:, idx], compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")

    # Non-Hermiticity measure
    H_dag = H.conj().T
    norm_H = np.linalg.norm(H, "fro")
    nh = float(np.linalg.norm(H - H_dag, "fro") / norm_H) if norm_H > 0 else 0.0

    return ComplexEigenResult(
        k=k,
        s=s,
        eigenvalues=eigenvalues,
        min_abs_eigenvalue=float(np.min(np.abs(eigenvalues))),
        condition_number=cond,
        non_hermiticity=nh,
    )


# ============================================================================
# Critical Line Scan: s = 1/2 + it
# ============================================================================


def scan_critical_line(
    k: int,
    t_max: float = 50.0,
    n_points: int = 500,
    *,
    t_min: float = 0.0,
    weight_by_log_gap: bool = True,
) -> CriticalLineScan:
    r"""Scan eigenvalues of H^(k)(1/2 + it) along the critical line.

    Sweeps t from t_min to t_max in n_points steps, computing the full
    complex spectrum at each point.  Identifies local minima in
    min_j |lambda_j(t)| as candidates for eigenvalue zero-crossings.

    Parameters
    ----------
    k : int
        Number of primes.
    t_max : float
        Maximum imaginary part to scan.
    n_points : int
        Number of t values in the scan.
    t_min : float
        Minimum imaginary part (default 0).
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    CriticalLineScan
        Full scan results including eigenvalue matrix and local minima.
    """
    t_values = np.linspace(t_min, t_max, n_points)
    eigenvalue_matrix = np.zeros((n_points, k), dtype=complex)
    min_abs = np.zeros(n_points)
    nh_values = np.zeros(n_points)

    # Precompute Laplacian and potential vector (structure is constant)
    d_L, e, log_p = build_tridiagonal_h_tnfr_complex(
        k,
        0.5 + 0j,
        weight_by_log_gap=weight_by_log_gap,
    )
    H_laplacian = _build_dense_from_tridiag(d_L, e)
    V1 = np.diag(log_p)  # V_1 = diag(log p_i)

    for idx, t in enumerate(t_values):
        delta_s = complex(0, t)  # s - 1/2 = it
        H = H_laplacian + delta_s * V1
        evals = np.linalg.eig(H)[0]
        sort_idx = np.lexsort((evals.imag, evals.real))
        eigenvalue_matrix[idx] = evals[sort_idx]
        min_abs[idx] = float(np.min(np.abs(evals)))

        # Non-Hermiticity
        norm_H = np.linalg.norm(H, "fro")
        if norm_H > 0:
            nh_values[idx] = float(np.linalg.norm(H - H.conj().T, "fro") / norm_H)

    # Find local minima in min_abs
    local_min_indices = _find_local_minima(min_abs)
    local_minima_t = t_values[local_min_indices]
    local_minima_val = min_abs[local_min_indices]

    return CriticalLineScan(
        k=k,
        t_values=t_values,
        eigenvalue_matrix=eigenvalue_matrix,
        min_abs_eigenvalue=min_abs,
        non_hermiticity=nh_values,
        local_minima_t=local_minima_t,
        local_minima_val=local_minima_val,
    )


def find_eigenvalue_zero_crossings(
    k: int,
    t_max: float = 50.0,
    n_coarse: int = 500,
    n_refine: int = 50,
    *,
    threshold: float = 0.5,
    weight_by_log_gap: bool = True,
) -> list[tuple[float, float]]:
    r"""Find t values where eigenvalues of H(1/2 + it) approach zero.

    Two-pass algorithm:
    1. Coarse scan to identify candidate regions (min |lambda_j| < threshold).
    2. Fine-grained refinement around each candidate.

    Parameters
    ----------
    k : int
        Number of primes.
    t_max : float
        Maximum t to scan.
    n_coarse : int
        Coarse grid resolution.
    n_refine : int
        Points per refinement window.
    threshold : float
        Threshold for detecting candidates in the coarse pass.
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    list of (t_zero, min_abs_eigenvalue) tuples, sorted by t.
    """
    scan = scan_critical_line(
        k,
        t_max,
        n_coarse,
        weight_by_log_gap=weight_by_log_gap,
    )

    results: list[tuple[float, float]] = []

    for t_candidate, val in zip(scan.local_minima_t, scan.local_minima_val):
        if val > threshold:
            continue
        # Refine around this candidate
        dt = scan.t_values[1] - scan.t_values[0] if len(scan.t_values) > 1 else 1.0
        t_lo = max(0.0, t_candidate - 2 * dt)
        t_hi = t_candidate + 2 * dt
        refined = scan_critical_line(
            k,
            t_hi,
            n_refine,
            t_min=t_lo,
            weight_by_log_gap=weight_by_log_gap,
        )
        best_idx = int(np.argmin(refined.min_abs_eigenvalue))
        results.append(
            (
                float(refined.t_values[best_idx]),
                float(refined.min_abs_eigenvalue[best_idx]),
            )
        )

    # Deduplicate close results
    results.sort(key=lambda x: x[0])
    return _deduplicate_crossings(results, min_gap=1.0)


# ============================================================================
# Pseudo-Spectrum
# ============================================================================


def compute_pseudospectrum(
    k: int,
    s: complex = 0.5 + 0j,
    *,
    z_center: complex | None = None,
    z_radius: float = 5.0,
    n_grid: int = 80,
    weight_by_log_gap: bool = True,
) -> PseudoSpectrumResult:
    r"""Compute the pseudo-spectrum of H(s) on a complex z-grid.

    For each z in the grid, computes sigma_min(zI - H), the smallest
    singular value of (zI - H).  The epsilon-pseudospectrum is the
    set {z : sigma_min(zI - H) < epsilon}.

    For normal operators (e.g. Hermitian), the pseudo-spectrum is
    simply the union of epsilon-discs around eigenvalues.  For the
    non-Hermitian H(s) with Im(s) != 0, the pseudo-spectrum can be
    significantly larger, revealing spectral instability.

    Parameters
    ----------
    k : int
        Number of primes.
    s : complex
        Complex structural parameter.
    z_center : complex, optional
        Centre of the z-grid.  Defaults to the mean eigenvalue.
    z_radius : float
        Half-width of the z-grid.
    n_grid : int
        Grid resolution per axis (total grid is n_grid x n_grid).
    weight_by_log_gap : bool
        Use log-gap edge weights.
    """
    d, e, _ = build_tridiagonal_h_tnfr_complex(
        k,
        s,
        weight_by_log_gap=weight_by_log_gap,
    )
    H = _build_dense_from_tridiag(d, e)
    eigenvalues = np.linalg.eig(H)[0]

    if z_center is None:
        z_center = complex(np.mean(eigenvalues))

    z_real = np.linspace(z_center.real - z_radius, z_center.real + z_radius, n_grid)
    z_imag = np.linspace(z_center.imag - z_radius, z_center.imag + z_radius, n_grid)

    sigma_min_grid = np.zeros((n_grid, n_grid))
    I_mat = np.eye(k, dtype=complex)

    for i, zi in enumerate(z_imag):
        for j, zr in enumerate(z_real):
            z = complex(zr, zi)
            M = z * I_mat - H
            sv = np.linalg.svd(M, compute_uv=False)
            sigma_min_grid[i, j] = float(sv[-1])

    return PseudoSpectrumResult(
        k=k,
        s=s,
        z_real=z_real,
        z_imag=z_imag,
        sigma_min_grid=sigma_min_grid,
        eigenvalues=eigenvalues,
    )


def compute_resolvent_norm(
    k: int,
    s: complex,
    z: complex,
    *,
    weight_by_log_gap: bool = True,
) -> float:
    r"""Compute the resolvent norm ||(zI - H(s))^{-1}|| = 1/sigma_min(zI - H).

    Returns infinity if z is an eigenvalue of H(s).
    """
    d, e, _ = build_tridiagonal_h_tnfr_complex(
        k,
        s,
        weight_by_log_gap=weight_by_log_gap,
    )
    H = _build_dense_from_tridiag(d, e)
    M = z * np.eye(k, dtype=complex) - H
    sv = np.linalg.svd(M, compute_uv=False)
    smin = float(sv[-1])
    return 1.0 / smin if smin > 1e-15 else float("inf")


# ============================================================================
# Resolvent Analysis Along Critical Line
# ============================================================================


def analyze_resolvent_along_critical_line(
    k: int,
    z_probe: complex = 0.0 + 0j,
    t_max: float = 50.0,
    n_points: int = 300,
    *,
    weight_by_log_gap: bool = True,
) -> ResolventAnalysis:
    r"""Compute resolvent norm ||(zI - H(1/2+it))^{-1}|| along critical line.

    Peaks in ||R(z, s)|| at specific t values indicate that z is near
    the spectrum of H(s).  If the TNFR operator has structural
    connections to the Riemann zeta function, peaks should correlate
    with known Riemann zeros.

    Parameters
    ----------
    k : int
        Number of primes.
    z_probe : complex
        Fixed point z at which to evaluate the resolvent.
    t_max : float
        Maximum imaginary part.
    n_points : int
        Number of t values.
    """
    t_values = np.linspace(0, t_max, n_points)
    resolvent_norms = np.zeros(n_points)

    d_L, e, log_p = build_tridiagonal_h_tnfr_complex(
        k,
        0.5 + 0j,
        weight_by_log_gap=weight_by_log_gap,
    )
    H_laplacian = _build_dense_from_tridiag(d_L, e)
    V1 = np.diag(log_p)
    I_mat = np.eye(k, dtype=complex)

    for idx, t in enumerate(t_values):
        H = H_laplacian + complex(0, t) * V1
        M = z_probe * I_mat - H
        sv = np.linalg.svd(M, compute_uv=False)
        smin = float(sv[-1])
        resolvent_norms[idx] = 1.0 / smin if smin > 1e-15 else 1e15

    # Find peaks
    peak_indices = _find_local_maxima(resolvent_norms)
    peak_t = t_values[peak_indices]
    peak_vals = resolvent_norms[peak_indices]

    return ResolventAnalysis(
        k=k,
        z_probe=z_probe,
        t_values=t_values,
        resolvent_norms=resolvent_norms,
        peak_t_values=peak_t,
        peak_norms=peak_vals,
    )


# ============================================================================
# Comparison with Known Riemann Zeros
# ============================================================================


def compare_with_riemann_zeros(
    k: int,
    t_max: float = 50.0,
    n_points: int = 500,
    *,
    n_zeros: int = 10,
    threshold: float = 1.0,
    weight_by_log_gap: bool = True,
) -> list[tuple[float, float, float]]:
    r"""Compare eigenvalue minima along critical line with known Riemann zeros.

    Scans s = 1/2 + it and identifies local minima of min_j |lambda_j(s)|.
    Each minimum is matched to the nearest known Riemann zero t_n and the
    distance |t_min - t_n| is reported.

    Parameters
    ----------
    k : int
        Number of primes.
    t_max : float
        Maximum t to scan.
    n_points : int
        Scan resolution.
    n_zeros : int
        Number of Riemann zeros to compare with (max 20).
    threshold : float
        Only report minima with min |lambda_j| below this threshold.
    weight_by_log_gap : bool
        Use log-gap edge weights.

    Returns
    -------
    list of (t_candidate, nearest_riemann_zero, distance) tuples.
    """
    scan = scan_critical_line(
        k,
        t_max,
        n_points,
        weight_by_log_gap=weight_by_log_gap,
    )

    n_zeros = min(n_zeros, len(KNOWN_RIEMANN_ZEROS))
    zeros = np.array(KNOWN_RIEMANN_ZEROS[:n_zeros])

    results: list[tuple[float, float, float]] = []
    for t_min, val in zip(scan.local_minima_t, scan.local_minima_val):
        if val > threshold:
            continue
        # Find nearest known zero
        distances = np.abs(zeros - t_min)
        nearest_idx = int(np.argmin(distances))
        results.append(
            (
                float(t_min),
                float(zeros[nearest_idx]),
                float(distances[nearest_idx]),
            )
        )

    return results


# ============================================================================
# Integrated Analysis
# ============================================================================


def run_complex_plane_analysis(
    k: int,
    t_max: float = 50.0,
    n_points: int = 500,
    *,
    n_zeros: int = 10,
    weight_by_log_gap: bool = True,
) -> ComplexPlaneAnalysis:
    r"""Run integrated P4 complex-s analysis.

    Combines:
    1. Critical line scan (eigenvalue flow for s = 1/2 + it).
    2. Non-Hermiticity characterization.
    3. Comparison with known Riemann zeros.

    Parameters
    ----------
    k : int
        Number of primes.
    t_max : float
        Maximum imaginary part.
    n_points : int
        Scan resolution.
    n_zeros : int
        Number of Riemann zeros for comparison.
    weight_by_log_gap : bool
        Use log-gap edge weights.
    """
    # Critical line scan
    crit = scan_critical_line(
        k,
        t_max,
        n_points,
        weight_by_log_gap=weight_by_log_gap,
    )

    # Non-Hermiticity at s = 1/2 (should be ~0)
    nh_half = float(crit.non_hermiticity[0]) if len(crit.non_hermiticity) > 0 else 0.0
    mean_nh = float(np.mean(crit.non_hermiticity))

    # Compare with Riemann zeros
    comparison = compare_with_riemann_zeros(
        k,
        t_max,
        n_points,
        n_zeros=n_zeros,
        weight_by_log_gap=weight_by_log_gap,
    )

    # Count zero-crossings (min |lambda| < 0.5)
    zc_count = int(np.sum(crit.local_minima_val < 0.5))

    # Build summary
    lines = [
        f"P4 Complex-s Analysis: k={k}, t in [0, {t_max}]",
        f"  Eigenvalue minima detected: {len(crit.local_minima_t)}",
        f"  Near-zero crossings (|lambda| < 0.5): {zc_count}",
        f"  Non-Hermiticity at s=1/2: {nh_half:.6e}",
        f"  Mean non-Hermiticity: {mean_nh:.4f}",
    ]
    if comparison:
        lines.append(f"  Riemann zero matches: {len(comparison)}")
        for t_c, t_r, dist in comparison[:5]:
            lines.append(f"    t={t_c:.4f} -> nearest zero t={t_r:.4f} (d={dist:.4f})")
    else:
        lines.append("  No eigenvalue minima matched Riemann zeros.")

    return ComplexPlaneAnalysis(
        k=k,
        critical_line=crit,
        riemann_comparison=comparison,
        non_hermiticity_at_half=nh_half,
        mean_non_hermiticity=mean_nh,
        zero_crossing_count=zc_count,
        summary="\n".join(lines),
    )


# ============================================================================
# Private Helpers
# ============================================================================


def _find_local_minima(arr: np.ndarray) -> np.ndarray:
    """Find indices of local minima in a 1D array."""
    if len(arr) < 3:
        return np.array([], dtype=int)
    indices = []
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            indices.append(i)
    return np.array(indices, dtype=int)


def _find_local_maxima(arr: np.ndarray) -> np.ndarray:
    """Find indices of local maxima in a 1D array."""
    if len(arr) < 3:
        return np.array([], dtype=int)
    indices = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            indices.append(i)
    return np.array(indices, dtype=int)


def _deduplicate_crossings(
    crossings: list[tuple[float, float]],
    min_gap: float = 1.0,
) -> list[tuple[float, float]]:
    """Remove duplicate crossings closer than min_gap in t."""
    if not crossings:
        return []
    result = [crossings[0]]
    for t, val in crossings[1:]:
        if t - result[-1][0] >= min_gap:
            result.append((t, val))
        elif val < result[-1][1]:
            result[-1] = (t, val)
    return result

r"""TNFR Spectral Conservation Laws — Conservation in the Eigenvalue Domain.

Lifts the structural conservation theorem to the spectral (frequency) domain
by expanding conservation fields in the graph Laplacian eigenbasis via the
Graph Fourier Transform (GFT).

MAIN RESULT (Spectral Continuity Theorem):
==========================================
Given the discrete structural continuity equation:

    Δρ(i)/Δt + div J(i) = S_grammar(i)

Apply the GFT (projection onto Laplacian eigenvectors ψ_k):

    dρ̂_k/dt + λ_k · Ĵ_k = Ŝ_k

where:
    ρ̂_k = ⟨ψ_k | ρ⟩     (charge density in mode k)
    Ĵ_k = ⟨ψ_k | div J⟩  (current divergence in mode k)
    Ŝ_k = ⟨ψ_k | S⟩      (source term in mode k)
    λ_k                    (Laplacian eigenvalue = mode frequency)

PHYSICAL INTERPRETATION:
========================
- **Low-frequency modes** (small λ_k): Global coherence — near-exact
  conservation (Ŝ_k ≈ 0).  These modes correspond to U5 multi-scale
  structure and persist under grammar-compliant evolution.

- **High-frequency modes** (large λ_k): Local fluctuations — rapid
  equilibration via λ_k · Ĵ_k dissipation.  Grammar violations
  (S_grammar ≠ 0) manifest primarily in these modes.

- **Parseval conservation**: Energy in the spatial domain equals energy
  in the spectral domain:  ‖ρ‖² = Σ_k |ρ̂_k|².  Drift in this identity
  signals numerical or structural inconsistency.

SPECTRAL ENERGY CONSERVATION:
=============================
The Lyapunov energy E = ½Σ_i [Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²]
decomposes mode-by-mode:

    E_k = ½(|Φ̂_s_k|² + |∇̂φ_k|² + |K̂_φ_k|² + |Ĵ_φ_k|² + |Ĵ_ΔNFR_k|²)

Under grammar compliance (U2): dE_k/dt ≤ 0 for stabilizer-dominated modes.

DERIVATION:
===========
This module is derived via GFT of conservation.py equations.  The Laplacian
eigenbasis diagonalizes the diffusion operator, making mode-by-mode analysis
natural.  All results follow from the nodal equation ∂EPI/∂t = νf · ΔNFR(t)
and the structural continuity theorem (theory/STRUCTURAL_CONSERVATION_THEOREM.md §9).

STATUS: CANONICAL — Derived from spectral decomposition of proven conservation laws.

References
----------
- Structural conservation: src/tnfr/physics/conservation.py
- Spectral math: src/tnfr/mathematics/spectral.py (GFT, Laplacian)
- Theory: theory/STRUCTURAL_CONSERVATION_THEOREM.md §9.2
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t) [TNFR.pdf §2.1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..mathematics.unified_numerical import np
from ..mathematics.spectral import get_laplacian_spectrum, gft

from .conservation import (
    ConservationSnapshot,
    capture_conservation_snapshot,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralConservationBalance:
    r"""Two-snapshot spectral conservation verification.

    Verifies the mode-by-mode continuity equation:

        Δρ̂_k/Δt + λ_k · Ĵ_k ≈ 0

    for each Laplacian eigenmode k.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Laplacian eigenvalues λ_k sorted ascending.  Shape (N,).
    eigenvectors : np.ndarray
        Laplacian eigenvectors ψ_k as columns.  Shape (N, N).
    rho_spectrum_before : np.ndarray
        Charge density spectrum ρ̂_k at t_0.  Shape (N,).
    rho_spectrum_after : np.ndarray
        Charge density spectrum ρ̂_k at t_1.  Shape (N,).
    div_spectrum_mean : np.ndarray
        Mean current divergence spectrum (Ĵ_k_before + Ĵ_k_after)/2.
    mode_residuals : np.ndarray
        Per-mode unsigned residual |Δρ̂_k/Δt + λ_k · Ĵ_k|.  Shape (N,).
    mode_sources : np.ndarray
        Per-mode signed source Ŝ_k = Δρ̂_k/Δt + λ_k · Ĵ_k.  Shape (N,).
    parseval_before : float
        Spectral energy Σ|ρ̂_k|² at t_0.
    parseval_after : float
        Spectral energy Σ|ρ̂_k|² at t_1.
    parseval_drift : float
        Relative Parseval drift |(E_after - E_before)| / max(E_before, ε).
    spectral_gap : float
        λ_1 — first non-trivial eigenvalue.
    n_conserved_modes : int
        Number of modes with residual below tolerance.
    conservation_quality_by_band : Dict[str, float]
        Mean conservation quality per frequency band ('low', 'mid', 'high').
        Quality = 1 / (1 + mean_residual) ∈ [0, 1].
    overall_spectral_quality : float
        Global spectral conservation quality ∈ [0, 1].
    """

    eigenvalues: Any  # np.ndarray
    eigenvectors: Any  # np.ndarray
    rho_spectrum_before: Any  # np.ndarray
    rho_spectrum_after: Any  # np.ndarray
    div_spectrum_mean: Any  # np.ndarray
    mode_residuals: Any  # np.ndarray
    mode_sources: Any  # np.ndarray
    parseval_before: float
    parseval_after: float
    parseval_drift: float
    spectral_gap: float
    n_conserved_modes: int
    conservation_quality_by_band: Dict[str, float]
    overall_spectral_quality: float


@dataclass(frozen=True)
class SpectralWardIdentity:
    r"""Per-operator spectral conservation signature.

    Characterizes how a canonical operator redistributes charge across
    spectral modes:  Δρ̂_k = ρ̂_k(after) - ρ̂_k(before).

    Attributes
    ----------
    operator_name : str
        Name of the canonical operator.
    delta_rho_spectrum : np.ndarray
        Change in charge spectrum per mode.
    mode_energy_change : np.ndarray
        Change in per-mode energy |ρ̂_k|².
    total_spectral_energy_change : float
        Σ Δ(|ρ̂_k|²).
    affected_band : str
        Dominant affected frequency band: 'low', 'mid', or 'high'.
    spectral_character : str
        'conservative' (total energy preserved),
        'dissipative' (energy decreasing),
        'injective' (energy increasing).
    """

    operator_name: str
    delta_rho_spectrum: Any  # np.ndarray
    mode_energy_change: Any  # np.ndarray
    total_spectral_energy_change: float
    affected_band: str
    spectral_character: str


@dataclass(frozen=True)
class SpectralLyapunovResult:
    r"""Spectral Lyapunov stability — mode-by-mode energy analysis.

    Decomposes the Lyapunov energy E = ½Σ_i[Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²]
    into per-mode contributions via GFT.

    Attributes
    ----------
    mode_energies_before : np.ndarray
        Per-mode energy at t_0.
    mode_energies_after : np.ndarray
        Per-mode energy at t_1.
    mode_derivatives : np.ndarray
        dE_k/dt ≈ (E_k(t_1) - E_k(t_0)) / dt per mode.
    total_derivative : float
        Σ dE_k/dt — total spectral energy derivative.
    n_unstable_modes : int
        Number of modes where dE_k/dt > stability_threshold.
    stable_fraction : float
        Fraction of modes with dE_k/dt ≤ stability_threshold.
    is_spectrally_stable : bool
        True if total_derivative ≤ 0 (Lyapunov condition in spectral domain).
    """

    mode_energies_before: Any  # np.ndarray
    mode_energies_after: Any  # np.ndarray
    mode_derivatives: Any  # np.ndarray
    total_derivative: float
    n_unstable_modes: int
    stable_fraction: float
    is_spectrally_stable: bool


@dataclass(frozen=True)
class SpectralSectorDecomposition:
    r"""Two-sector (potential/geometric) spectral analysis.

    Decomposes ρ = Φ_s + K_φ into its two sectors in the Laplacian eigenbasis,
    revealing how global potential and local curvature distribute across
    structural frequencies.

    Attributes
    ----------
    phi_s_spectrum : np.ndarray
        GFT of structural potential Φ_s. Shape (N,).
    k_phi_spectrum : np.ndarray
        GFT of phase curvature K_φ. Shape (N,).
    potential_sector_energy : float
        Σ |Φ̂_s_k|²  (total energy in potential sector).
    geometric_sector_energy : float
        Σ |K̂_φ_k|²  (total energy in geometric sector).
    cross_sector_correlation : float
        Pearson correlation between Φ̂_s and K̂_φ spectra.
    sector_coupling_by_mode : np.ndarray
        Per-mode coupling strength: |Φ̂_s_k · K̂_φ_k|.
    dominant_sector : str
        'potential' or 'geometric' based on total energy.
    sector_ratio : float
        potential_energy / geometric_energy (or inf/0 edge cases).
    """

    phi_s_spectrum: Any  # np.ndarray
    k_phi_spectrum: Any  # np.ndarray
    potential_sector_energy: float
    geometric_sector_energy: float
    cross_sector_correlation: float
    sector_coupling_by_mode: Any  # np.ndarray
    dominant_sector: str
    sector_ratio: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Canonical 5 conservation fields used for Lyapunov energy computation
_LYAPUNOV_FIELDS: List[str] = ["phi_s", "grad_phi", "k_phi", "j_phi", "j_dnfr"]


def _snapshot_to_vectors(
    snapshot: ConservationSnapshot,
    nodes: Sequence[Any],
) -> Dict[str, np.ndarray]:
    """Extract ordered arrays from a conservation snapshot."""
    return {
        "rho": np.array([snapshot.charge_density[n] for n in nodes]),
        "div": np.array([snapshot.divergence[n] for n in nodes]),
        "phi_s": np.array([snapshot.phi_s[n] for n in nodes]),
        "k_phi": np.array([snapshot.k_phi[n] for n in nodes]),
        "j_phi": np.array([snapshot.j_phi[n] for n in nodes]),
        "j_dnfr": np.array([snapshot.j_dnfr[n] for n in nodes]),
        "grad_phi": np.array([snapshot.grad_phi[n] for n in nodes]),
    }


def _gft_fields(
    eigvecs: np.ndarray,
    fields: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Project multiple spatial signals onto the Laplacian eigenbasis via GFT.

    Uses the canonical ``gft()`` from ``mathematics.spectral`` which supports
    GPU acceleration for large matrices.
    """
    return {name: gft(signal, eigvecs) for name, signal in fields.items()}


def _compute_spectral_field_energies(
    vecs_before: Dict[str, np.ndarray],
    vecs_after: Dict[str, np.ndarray],
    eigvecs: np.ndarray,
    fields: List[str] = _LYAPUNOV_FIELDS,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
    r"""Per-mode Lyapunov energy from GFT of conservation fields.

    Computes E_k = ½ Σ_f |f̂_k|² for each mode k across the specified fields.

    Returns
    -------
    energy_before, energy_after : np.ndarray
        Per-mode energy arrays of shape (N,).
    per_field : Dict[str, Tuple[float, float]]
        Total spectral energy (before, after) for each field.
    """
    n = eigvecs.shape[0]
    energy_before = np.zeros(n)
    energy_after = np.zeros(n)
    per_field: Dict[str, Tuple[float, float]] = {}

    for field in fields:
        hat_0 = gft(vecs_before[field], eigvecs)
        hat_1 = gft(vecs_after[field], eigvecs)
        e0_sq = hat_0 ** 2
        e1_sq = hat_1 ** 2
        energy_before += e0_sq
        energy_after += e1_sq
        per_field[field] = (float(np.sum(e0_sq)), float(np.sum(e1_sq)))

    energy_before *= 0.5
    energy_after *= 0.5
    return energy_before, energy_after, per_field


def _classify_band(k: int, n: int) -> str:
    """Classify eigenmode index into frequency band."""
    if n <= 3:
        return "low"
    third = n / 3.0
    if k < third:
        return "low"
    elif k < 2 * third:
        return "mid"
    else:
        return "high"


def _band_quality(residuals: np.ndarray, n: int) -> Dict[str, float]:
    """Compute conservation quality per frequency band.

    Quality = 1/(1 + mean_residual) ∈ [0, 1].
    """
    bands: Dict[str, list] = {"low": [], "mid": [], "high": []}
    for k in range(n):
        bands[_classify_band(k, n)].append(float(residuals[k]))

    result: Dict[str, float] = {}
    for name, vals in bands.items():
        if vals:
            mean_r = sum(vals) / len(vals)
            result[name] = 1.0 / (1.0 + mean_r)
        else:
            result[name] = 1.0
    return result


# ---------------------------------------------------------------------------
# Core: Spectral conservation balance (two-snapshot)
# ---------------------------------------------------------------------------

def verify_spectral_conservation_balance(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    G: Any,
    dt: float = 1.0,
    tolerance: float = 1e-6,
) -> SpectralConservationBalance:
    r"""Verify the spectral continuity equation across two snapshots.

    For each Laplacian eigenmode k, computes:

        Ŝ_k = Δρ̂_k/Δt + λ_k · Ĵ_k

    where Ŝ_k is the spectral source term (should be ≈ 0 under grammar
    compliance).  This is the GFT of the spatial continuity equation
    ∂ρ/∂t + div(J) = S_grammar.

    Physics
    -------
    Low-frequency modes (global coherence) exhibit near-exact conservation.
    High-frequency modes (local fluctuations) show larger residuals due to
    rapid equilibration.  The spectral gap λ_1 determines the rate at which
    global modes relax.

    Parameters
    ----------
    before : ConservationSnapshot
        State at time t_0.
    after : ConservationSnapshot
        State at time t_1 = t_0 + dt.
    G : TNFRGraph
        The graph (needed for Laplacian eigenbasis).
    dt : float
        Time step Δt > 0.
    tolerance : float
        Residual threshold for classifying a mode as "conserved".

    Returns
    -------
    SpectralConservationBalance
    """
    nodes = sorted(before.charge_density.keys())
    n = len(nodes)

    eigvals, eigvecs = get_laplacian_spectrum(G)

    vecs_before = _snapshot_to_vectors(before, nodes)
    vecs_after = _snapshot_to_vectors(after, nodes)

    # GFT: project into eigenbasis
    rho_hat_0 = gft(vecs_before["rho"], eigvecs)
    rho_hat_1 = gft(vecs_after["rho"], eigvecs)
    div_hat_0 = gft(vecs_before["div"], eigvecs)
    div_hat_1 = gft(vecs_after["div"], eigvecs)

    # Mean divergence spectrum (trapezoidal-like average)
    div_hat_mean = 0.5 * (div_hat_0 + div_hat_1)

    # Mode-by-mode continuity residual
    drho_dt = (rho_hat_1 - rho_hat_0) / dt
    mode_sources = drho_dt + eigvals * div_hat_mean
    mode_residuals = np.abs(mode_sources)

    # Parseval identity: ‖ρ‖² = Σ|ρ̂_k|²
    parseval_0 = float(np.sum(rho_hat_0 ** 2))
    parseval_1 = float(np.sum(rho_hat_1 ** 2))
    denom = max(parseval_0, 1e-15)
    parseval_drift = abs(parseval_1 - parseval_0) / denom

    # Spectral gap
    spectral_gap = float(eigvals[1]) if n > 1 else 0.0

    # Conserved mode count
    n_conserved = int(np.sum(mode_residuals <= tolerance))

    # Band-resolved quality
    band_quality = _band_quality(mode_residuals, n)

    # Overall quality: 1/(1 + RMS residual)
    rms = float(np.sqrt(np.mean(mode_residuals ** 2))) if n > 0 else 0.0
    overall_quality = 1.0 / (1.0 + rms)

    return SpectralConservationBalance(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        rho_spectrum_before=rho_hat_0,
        rho_spectrum_after=rho_hat_1,
        div_spectrum_mean=div_hat_mean,
        mode_residuals=mode_residuals,
        mode_sources=mode_sources,
        parseval_before=parseval_0,
        parseval_after=parseval_1,
        parseval_drift=parseval_drift,
        spectral_gap=spectral_gap,
        n_conserved_modes=n_conserved,
        conservation_quality_by_band=band_quality,
        overall_spectral_quality=overall_quality,
    )


# ---------------------------------------------------------------------------
# Spectral Ward identity (per-operator)
# ---------------------------------------------------------------------------

def compute_spectral_ward_identity(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    operator_name: str,
    G: Any,
) -> SpectralWardIdentity:
    r"""Compute per-operator spectral conservation signature.

    Characterizes how a canonical operator redistributes structural charge
    across spectral modes.  Complements the spatial Ward identity in
    conservation.py by revealing *which frequency scales* the operator
    affects.

    Physics
    -------
    - Stabilizers (IL, THOL): Should primarily affect high-frequency modes
      (local smoothing), preserving low-frequency global structure.
    - Destabilizers (OZ, VAL): Inject energy into mid/high-frequency modes.
    - Generators (AL, NAV): Affect all bands (new structure creation).
    - Coupling (UM, RA): Redistribute energy across modes via phase sync.

    Parameters
    ----------
    before : ConservationSnapshot
        State before operator application.
    after : ConservationSnapshot
        State after operator application.
    operator_name : str
        Canonical operator name (e.g. 'IL', 'OZ', 'AL').
    G : TNFRGraph
        The graph.

    Returns
    -------
    SpectralWardIdentity
    """
    nodes = sorted(before.charge_density.keys())
    n = len(nodes)

    _, eigvecs = get_laplacian_spectrum(G)

    vecs_before = _snapshot_to_vectors(before, nodes)
    vecs_after = _snapshot_to_vectors(after, nodes)

    rho_hat_0 = gft(vecs_before["rho"], eigvecs)
    rho_hat_1 = gft(vecs_after["rho"], eigvecs)

    delta_rho = rho_hat_1 - rho_hat_0

    # Per-mode energy change: Δ(|ρ̂_k|²) = |ρ̂_k_after|² - |ρ̂_k_before|²
    energy_before = rho_hat_0 ** 2
    energy_after = rho_hat_1 ** 2
    mode_energy_change = energy_after - energy_before

    total_change = float(np.sum(mode_energy_change))

    # Determine which band is most affected (by absolute energy change)
    band_energy: Dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
    for k in range(n):
        band = _classify_band(k, n)
        band_energy[band] += abs(float(mode_energy_change[k]))

    affected_band = max(band_energy, key=band_energy.get)  # type: ignore[arg-type]

    # Classify spectral character
    eps = 1e-12
    if abs(total_change) < eps:
        spectral_character = "conservative"
    elif total_change < -eps:
        spectral_character = "dissipative"
    else:
        spectral_character = "injective"

    return SpectralWardIdentity(
        operator_name=operator_name,
        delta_rho_spectrum=delta_rho,
        mode_energy_change=mode_energy_change,
        total_spectral_energy_change=total_change,
        affected_band=affected_band,
        spectral_character=spectral_character,
    )


# ---------------------------------------------------------------------------
# Spectral Lyapunov stability
# ---------------------------------------------------------------------------

def compute_spectral_lyapunov(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    G: Any,
    dt: float = 1.0,
    stability_threshold: float = 1e-6,
) -> SpectralLyapunovResult:
    r"""Spectral decomposition of Lyapunov energy stability.

    Decomposes the structural energy functional E = ½Σ_i[Φ_s² + |∇φ|² + K_φ²
    + J_φ² + J_ΔNFR²] into per-mode contributions, then checks:

        dE_k/dt ≤ 0  (mode-stable)

    Under grammar compliance (U2: convergence), the total spectral energy
    derivative should be non-positive: Σ dE_k/dt ≤ 0.

    Parameters
    ----------
    before, after : ConservationSnapshot
        Two successive states.
    G : TNFRGraph
    dt : float
        Time step.
    stability_threshold : float
        Modes with dE_k/dt > threshold are classified as unstable.

    Returns
    -------
    SpectralLyapunovResult
    """
    nodes = sorted(before.charge_density.keys())
    n = len(nodes)

    _, eigvecs = get_laplacian_spectrum(G)

    vecs_before = _snapshot_to_vectors(before, nodes)
    vecs_after = _snapshot_to_vectors(after, nodes)

    # Per-mode Lyapunov energy via unified helper
    energy_before, energy_after, _ = _compute_spectral_field_energies(
        vecs_before, vecs_after, eigvecs, _LYAPUNOV_FIELDS,
    )

    derivatives = (energy_after - energy_before) / dt
    total_derivative = float(np.sum(derivatives))

    n_unstable = int(np.sum(derivatives > stability_threshold))
    stable_frac = 1.0 - n_unstable / max(n, 1)

    return SpectralLyapunovResult(
        mode_energies_before=energy_before,
        mode_energies_after=energy_after,
        mode_derivatives=derivatives,
        total_derivative=total_derivative,
        n_unstable_modes=n_unstable,
        stable_fraction=stable_frac,
        is_spectrally_stable=total_derivative <= stability_threshold,
    )


# ---------------------------------------------------------------------------
# Spectral sector decomposition
# ---------------------------------------------------------------------------

def decompose_spectral_sectors(
    G: Any,
    snapshot: Optional[ConservationSnapshot] = None,
) -> SpectralSectorDecomposition:
    r"""Decompose the two conservation sectors in spectral domain.

    The structural charge ρ = Φ_s + K_φ consists of:
    - **Potential sector** (Φ_s): Global ΔNFR-driven dynamics
    - **Geometric sector** (K_φ): Local phase-driven dynamics

    This function computes their GFT spectra and measures how the two sectors
    couple across frequency modes.  Strong coupling indicates the complex field
    Ψ = K_φ + i·J_φ is active.

    Parameters
    ----------
    G : TNFRGraph
    snapshot : ConservationSnapshot, optional
        If None, captured from G.

    Returns
    -------
    SpectralSectorDecomposition
    """
    if snapshot is None:
        snapshot = capture_conservation_snapshot(G)

    nodes = sorted(snapshot.charge_density.keys())
    n = len(nodes)

    _, eigvecs = get_laplacian_spectrum(G)

    vecs = _snapshot_to_vectors(snapshot, nodes)
    phi_s_hat = gft(vecs["phi_s"], eigvecs)
    k_phi_hat = gft(vecs["k_phi"], eigvecs)

    pot_energy = float(np.sum(phi_s_hat ** 2))
    geo_energy = float(np.sum(k_phi_hat ** 2))

    # Pearson correlation between spectra
    if n > 1:
        std_phi = float(np.std(phi_s_hat))
        std_kphi = float(np.std(k_phi_hat))
        if std_phi > 1e-15 and std_kphi > 1e-15:
            corr = float(np.corrcoef(phi_s_hat, k_phi_hat)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    coupling = np.abs(phi_s_hat * k_phi_hat)

    if geo_energy > 1e-15:
        ratio = pot_energy / geo_energy
    else:
        ratio = float("inf") if pot_energy > 1e-15 else 1.0

    dominant = "potential" if pot_energy >= geo_energy else "geometric"

    return SpectralSectorDecomposition(
        phi_s_spectrum=phi_s_hat,
        k_phi_spectrum=k_phi_hat,
        potential_sector_energy=pot_energy,
        geometric_sector_energy=geo_energy,
        cross_sector_correlation=corr,
        sector_coupling_by_mode=coupling,
        dominant_sector=dominant,
        sector_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Spectral energy conservation (Parseval-based)
# ---------------------------------------------------------------------------

def compute_spectral_energy_conservation(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    G: Any,
) -> Dict[str, float]:
    r"""Measure Parseval energy conservation across all five canonical fields.

    For each field f ∈ {Φ_s, |∇φ|, K_φ, J_φ, J_ΔNFR}, the Parseval identity
    guarantees ‖f‖² = Σ_k |f̂_k|².  This function measures the drift in
    total spectral energy between two snapshots:

        ΔE_f = |Σ|f̂_k(t1)|² - Σ|f̂_k(t0)|²| / max(Σ|f̂_k(t0)|², ε)

    Small ΔE_f indicates structural stability in the spectral domain.

    Parameters
    ----------
    before, after : ConservationSnapshot
    G : TNFRGraph

    Returns
    -------
    Dict[str, float]
        Keys: 'phi_s_drift', 'grad_phi_drift', 'k_phi_drift',
        'j_phi_drift', 'j_dnfr_drift', 'total_energy_before',
        'total_energy_after', 'total_drift'.
    """
    nodes = sorted(before.charge_density.keys())

    _, eigvecs = get_laplacian_spectrum(G)

    vecs_before = _snapshot_to_vectors(before, nodes)
    vecs_after = _snapshot_to_vectors(after, nodes)

    # Unified energy computation for all five canonical fields
    _, _, per_field = _compute_spectral_field_energies(
        vecs_before, vecs_after, eigvecs, _LYAPUNOV_FIELDS,
    )

    result: Dict[str, float] = {}
    total_e0 = 0.0
    total_e1 = 0.0

    for field in _LYAPUNOV_FIELDS:
        e0, e1 = per_field[field]
        denom = max(e0, 1e-15)
        result[f"{field}_drift"] = abs(e1 - e0) / denom
        total_e0 += e0
        total_e1 += e1

    result["total_energy_before"] = total_e0
    result["total_energy_after"] = total_e1
    denom = max(total_e0, 1e-15)
    result["total_drift"] = abs(total_e1 - total_e0) / denom

    return result


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------

def classify_spectral_modes(
    G: Any,
    snapshot: Optional[ConservationSnapshot] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    r"""Classify spectral modes by their conservation behavior.

    Each mode k is classified as:
    - 'conserved': |λ_k · Ĵ_k| < threshold (low transport rate)
    - 'dissipative': λ_k · Ĵ_k > 0 and above threshold
    - 'accumulative': λ_k · Ĵ_k < 0 and above threshold

    This mirrors the spatial conservation analysis but in the frequency
    domain, where the classification is mode-wise rather than node-wise.

    Parameters
    ----------
    G : TNFRGraph
    snapshot : ConservationSnapshot, optional
    threshold : float, optional
        Classification threshold.  Defaults to median |λ_k · Ĵ_k|.

    Returns
    -------
    Dict with keys:
        'mode_labels': list of str per mode
        'n_conserved': int
        'n_dissipative': int
        'n_accumulative': int
        'mode_transport_rates': np.ndarray (λ_k · Ĵ_k signed)
    """
    if snapshot is None:
        snapshot = capture_conservation_snapshot(G)

    nodes = sorted(snapshot.charge_density.keys())
    n = len(nodes)

    eigvals, eigvecs = get_laplacian_spectrum(G)

    vecs = _snapshot_to_vectors(snapshot, nodes)
    div_hat = gft(vecs["div"], eigvecs)

    transport_rates = eigvals * div_hat

    if threshold is None:
        threshold = float(np.median(np.abs(transport_rates))) if n > 0 else 0.0

    labels = []
    n_cons = n_diss = n_acc = 0
    for k in range(n):
        rate = float(transport_rates[k])
        if abs(rate) < threshold:
            labels.append("conserved")
            n_cons += 1
        elif rate > 0:
            labels.append("dissipative")
            n_diss += 1
        else:
            labels.append("accumulative")
            n_acc += 1

    return {
        "mode_labels": labels,
        "n_conserved": n_cons,
        "n_dissipative": n_diss,
        "n_accumulative": n_acc,
        "mode_transport_rates": transport_rates,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "SpectralConservationBalance",
    "SpectralWardIdentity",
    "SpectralLyapunovResult",
    "SpectralSectorDecomposition",
    # Core analysis
    "verify_spectral_conservation_balance",
    "compute_spectral_ward_identity",
    "compute_spectral_lyapunov",
    "decompose_spectral_sectors",
    "compute_spectral_energy_conservation",
    "classify_spectral_modes",
]

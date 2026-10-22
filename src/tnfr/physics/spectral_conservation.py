r"""TNFR spectral balance diagnostics in a Laplacian eigenbasis.

Expands the finite structural balance fields in a graph-Laplacian eigenbasis
via the Graph Fourier Transform (GFT). The transformation is algebraically
exact for a fixed orthonormal basis; whether a supplied trajectory satisfies a
conservation law remains an observed residual question.

Observations require nonempty support, complete finite real consumed maps and
a complete finite real orthonormal chart. Time intervals are finite and strictly
positive; policy tolerances are finite and nonnegative. Nonfinite result
arithmetic and detected nonzero product/rate loss reject before classification.
This sufficient numerical domain does not promise recovery of an aggregate
whose intermediate squared components cannot be represented. The legacy
sector-ratio infinity for a negligible denominator remains an explicit sentinel.

FIXED-BASIS BALANCE IDENTITY:
============================
Given the discrete structural continuity equation:

    Δρ(i)/Δt + div J(i) = S_grammar(i)

Apply the GFT (projection onto Laplacian eigenvectors ψ_k):

    Δρ̂_k/Δt + Ĵ_k = Ŝ_k

where:
    ρ̂_k = ⟨ψ_k | ρ⟩     (charge density in mode k)
    Ĵ_k = ⟨ψ_k | div J⟩  (current divergence in mode k)
    Ŝ_k = ⟨ψ_k | S⟩      (source term in mode k)
    λ_k                    (Laplacian spatial eigenvalue)

INTERPRETATION:
===============
- Small and large λ_k label smooth and oscillatory graph modes for the selected
  Laplacian. Their residual magnitudes are data; U5 or grammar compliance does
  not force a particular spectral band.

- **Parseval conservation**: Energy in the spatial domain equals energy
  in the spectral domain:  ‖ρ‖² = Σ_k |ρ̂_k|².  Drift in this identity
  signals numerical or structural inconsistency.

SPECTRAL STRUCTURAL ENERGY:
===========================
The nonnegative structural snapshot-energy candidate
E = ½Σ_i [Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²]
decomposes mode-by-mode:

    E_k = ½(|Φ̂_s_k|² + |∇̂φ_k|² + |K̂_φ_k|² + |Ĵ_φ_k|² + |Ĵ_ΔNFR_k|²)

The implementation reports each finite difference dE_k/dt. No sign follows
from U2 labels alone.

DERIVATION:
===========
This module applies the GFT to conservation.py diagnostics. The Laplacian
eigenbasis diagonalizes the diffusion operator, making mode-by-mode analysis
natural. Parseval identities follow from orthonormality; dynamic conservation
requires additional model hypotheses.

STATUS: CANONICAL DIAGNOSTIC INTERFACE.

References
----------
- Structural conservation: src/tnfr/physics/conservation.py
- Spectral math: src/tnfr/mathematics/spectral.py (GFT, Laplacian)
- Theory: theory/STRUCTURAL_CONSERVATION_THEOREM.md §9.2
- Nodal identity and complete-law premises: theory/FUNDAMENTAL_THEORY.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from ..mathematics import spectral as _spectral_math
from ..mathematics._complex_arrays import nonnegative_tolerance
from ..mathematics._neighbor_differences import mean_neighbor_difference
from ..mathematics.spectral import _gft_many, get_laplacian_spectrum, gft
from ..mathematics.unified_numerical import np
from ..metrics.common import finite_pearson_correlation, finite_population_std
from ._helpers import finite_real_scalar, finite_real_series
from .conservation import (
    ConservationSnapshot,
    _observed_secant,
    _positive_interval,
    _rms,
    capture_conservation_snapshot,
)

_CANONICAL_GFT = gft

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralConservationBalance:
    r"""Two-snapshot spectral conservation verification.

    Verifies the mode-by-mode continuity equation:

        Δρ̂_k/Δt + Ĵ_k ≈ 0

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
        Per-mode unsigned residual |Δρ̂_k/Δt + Ĵ_k|.  Shape (N,).
    mode_sources : np.ndarray
        Per-mode signed source Ŝ_k = Δρ̂_k/Δt + Ĵ_k.  Shape (N,).
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
    conservation_quality_by_band : dict[str, float | None]
        Mean conservation quality per frequency band ('low', 'mid', 'high').
        Quality = 1 / (1 + mean_residual) ∈ [0, 1]. Empty bands are None.
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
    conservation_quality_by_band: dict[str, float | None]
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
        'conservative' (absolute total energy change below the 1e-12 policy),
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
    r"""Legacy-named mode-by-mode structural snapshot-energy analysis.

    Decomposes the nonnegative candidate
    E = ½Σ_i[Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²] into per-mode
    contributions via GFT.  The class name is retained for compatibility; no
    monotonicity follows without a specified dynamics.

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
        True if the total finite difference is at most stability_threshold.
        This numerical tolerance flag is not a future stability theorem.
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

# Five fields used for the structural snapshot-energy candidate.
_STRUCTURAL_ENERGY_FIELDS: list[str] = ["phi_s", "grad_phi", "k_phi", "j_phi", "j_dnfr"]


def _observation_nodes(graph: Any) -> tuple[Any, ...]:
    nodes = tuple(graph.nodes())
    if not nodes:
        raise ValueError("spectral observations require nonempty node support")
    return nodes


def _observation_spectrum(
    graph: Any, nodes: Sequence[Any], *, consume_eigenvalues: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Admit the complete real orthonormal chart required by these observations."""
    eigenvalues, eigenvectors = get_laplacian_spectrum(graph)
    raw = (
        eigenvectors
        if type(eigenvectors) is np.ndarray
        else np.asarray(eigenvectors, dtype=object)
    )
    if raw.shape != (len(nodes), len(nodes)):
        raise ValueError("spectral observations require a complete square eigenbasis")
    order = "F" if raw.flags.f_contiguous else "C"
    basis = finite_real_series(
        raw.reshape(-1, order=order), "spectral eigenbasis"
    ).reshape(raw.shape, order=order)
    with np.errstate(over="ignore", invalid="ignore"):
        gram = basis.T @ basis
    if not np.allclose(gram, np.eye(len(nodes)), rtol=1e-10, atol=1e-12):
        raise ValueError("spectral observations require a real orthonormal eigenbasis")
    if consume_eigenvalues:
        eigenvalues = finite_real_series(
            eigenvalues, "spectral eigenvalues", nonnegative=True, nonempty=True
        )
        if eigenvalues.shape != (len(nodes),):
            raise ValueError("spectral eigenvalues must match the complete eigenbasis")
    return eigenvalues, basis


def _snapshot_to_vectors(
    snapshot: ConservationSnapshot,
    nodes: Sequence[Any],
    fields: Sequence[str],
) -> dict[str, np.ndarray]:
    """Extract consumed complete maps in the supplied Laplacian node order."""
    if not isinstance(snapshot, ConservationSnapshot):
        raise TypeError("expected a ConservationSnapshot")
    support = set(nodes)
    for field in fields:
        values = getattr(snapshot, field)
        if not isinstance(values, Mapping) or set(values) != support:
            raise ValueError(f"snapshot {field} must match the graph node support")
    return {
        field: finite_real_series(
            [getattr(snapshot, field)[node] for node in nodes],
            f"snapshot {field}",
            nonempty=True,
        )
        for field in fields
    }


def _finite_product(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = left * right
    result = finite_real_series(result, name)
    if np.any((left != 0.0) & (right != 0.0) & (result == 0.0)):
        raise ValueError(f"{name} is nonzero but underflows to represented zero")
    return result


def _finite_sum(values: np.ndarray, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return finite_real_scalar(np.sum(values), name)


def _finite_ratio(numerator: float, denominator: float, name: str) -> float:
    result = finite_real_scalar(numerator / denominator, name)
    if numerator != 0.0 and result == 0.0:
        raise ValueError(f"{name} is nonzero but underflows to represented zero")
    return result


def _modal_secants(before: np.ndarray, after: np.ndarray, dt: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = (after - before) / dt
    exceptional = ~np.isfinite(result) | ((result == 0.0) & (after != before))
    for index in np.flatnonzero(exceptional):
        result[index] = _observed_secant(before[index], after[index], dt, "modal rate")
    return result


def _mean_divergence(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = 0.5 * (before + after)
    exceptional = ~np.isfinite(result) | ((result == 0.0) & (before != -after))
    for index in np.flatnonzero(exceptional):
        result[index] = mean_neighbor_difference(0.0, (before[index], after[index]))
        if result[index] == 0.0 and before[index] != -after[index]:
            raise ValueError(
                "mean divergence is nonzero but underflows to represented zero"
            )
    return result


def _can_reuse_basis(eigvecs: np.ndarray) -> bool:
    """Reuse only the ordinary CPU route with no accelerator callbacks.

    Above the GFT's 100-node backend threshold, preserve sequential local
    transform calls and consumption: adapters may replace hooks or reuse
    output storage between calls. No basis is retained across observations.
    """
    return (
        gft is _CANONICAL_GFT
        and _spectral_math.gft is _CANONICAL_GFT
        and type(eigvecs) is np.ndarray
        and eigvecs.ndim == 2
        and eigvecs.dtype.kind in "iufc"
        and eigvecs.shape[0] <= 100
    )


def _ordinary_signals(signals: Sequence[np.ndarray]) -> bool:
    """Exclude signal conversions or arithmetic with user callbacks."""
    return all(
        type(signal) is np.ndarray and signal.dtype.kind in "iufc" for signal in signals
    )


def _project_vectors(
    signals: tuple[np.ndarray, ...], eigvecs: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Reuse one local basis validation unless the public transform was replaced."""
    if _can_reuse_basis(eigvecs) and _ordinary_signals(signals):
        return _gft_many(signals, eigvecs)
    return tuple(gft(signal, eigvecs) for signal in signals)


def _observed_coefficients(coefficients: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    result = finite_real_series(coefficients, "spectral coefficients", nonempty=True)
    if result.shape != (eigvecs.shape[1],):
        raise ValueError("spectral coefficients must match the complete eigenbasis")
    return result


def _observed_projections(
    signals: tuple[np.ndarray, ...], eigvecs: np.ndarray
) -> tuple[np.ndarray, ...]:
    return tuple(
        _observed_coefficients(coefficients, eigvecs)
        for coefficients in _project_vectors(signals, eigvecs)
    )


def _compute_spectral_field_energies(
    vecs_before: dict[str, np.ndarray],
    vecs_after: dict[str, np.ndarray],
    eigvecs: np.ndarray,
    fields: list[str] = _STRUCTURAL_ENERGY_FIELDS,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    r"""Per-mode structural snapshot energy from GFT of conservation fields.

    Computes E_k = ½ Σ_f |f̂_k|² for each mode k across the specified fields.

    Returns
    -------
    energy_before, energy_after : np.ndarray
        Per-mode energy arrays of shape (N,).
    per_field : dict[str, tuple[float, float]]
        Total spectral energy (before, after) for each field.
    """
    n = eigvecs.shape[0]
    energy_before = np.zeros(n)
    energy_after = np.zeros(n)
    per_field: dict[str, tuple[float, float]] = {}

    spectra = None
    if _can_reuse_basis(eigvecs):
        signals = tuple(
            signal
            for field in fields
            for signal in (vecs_before[field], vecs_after[field])
        )
        if _ordinary_signals(signals):
            spectra = iter(_gft_many(signals, eigvecs))

    for field in fields:
        if spectra is None:
            # Preserve callback order and effects between fields for overrides.
            hat_0 = gft(vecs_before[field], eigvecs)
            hat_1 = gft(vecs_after[field], eigvecs)
        else:
            hat_0, hat_1 = next(spectra), next(spectra)
        # Consume a callback's pair only after both transforms, as before.
        hat_0 = _observed_coefficients(hat_0, eigvecs)
        hat_1 = _observed_coefficients(hat_1, eigvecs)
        e0_sq = _finite_product(hat_0, hat_0, f"{field} squared spectrum before")
        e1_sq = _finite_product(hat_1, hat_1, f"{field} squared spectrum after")
        energy_before += e0_sq
        energy_after += e1_sq
        per_field[field] = (
            _finite_sum(e0_sq, f"{field} energy before"),
            _finite_sum(e1_sq, f"{field} energy after"),
        )

    energy_before = _finite_product(
        energy_before, np.full(n, 0.5), "modal energy before"
    )
    energy_after = _finite_product(energy_after, np.full(n, 0.5), "modal energy after")
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


def _band_quality(residuals: np.ndarray, n: int) -> dict[str, float | None]:
    """Compute conservation quality per frequency band.

    Quality = 1/(1 + mean_residual) ∈ [0, 1]; empty bands are unavailable.
    """
    bands: dict[str, list] = {"low": [], "mid": [], "high": []}
    for k in range(n):
        bands[_classify_band(k, n)].append(float(residuals[k]))

    result: dict[str, float | None] = {}
    for name, vals in bands.items():
        if vals:
            mean_r = mean_neighbor_difference(0.0, vals)
            result[name] = 1.0 / (1.0 + mean_r)
        else:
            result[name] = None
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

        Ŝ_k = Δρ̂_k/Δt + Ĵ_k

    Here Ĵ_k already transforms the observed divergence, so no additional
    eigenvalue factor is applied. This is the GFT of the two-snapshot spatial
    residual using the mean divergence. Its vanishing requires a matching
    complete law and observations; grammar compliance alone does not imply it.

    The supplied graph's iteration order defines the basis rows and the
    corresponding snapshot vectors. Each consumed snapshot map must have
    exactly that node support. Band quality is descriptive; no band ordering
    or relaxation rate follows without additional dynamics.

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
        Finite nonnegative residual tolerance; equality is included.

    Returns
    -------
    SpectralConservationBalance
    """
    dt = _positive_interval(dt)
    tolerance = nonnegative_tolerance(tolerance, "tolerance")
    nodes = _observation_nodes(G)
    n = len(nodes)

    eigvals, eigvecs = _observation_spectrum(G, nodes, consume_eigenvalues=True)

    vecs_before = _snapshot_to_vectors(before, nodes, ("charge_density", "divergence"))
    vecs_after = _snapshot_to_vectors(after, nodes, ("charge_density", "divergence"))

    # GFT: project into eigenbasis
    rho_hat_0, rho_hat_1, div_hat_0, div_hat_1 = _observed_projections(
        (
            vecs_before["charge_density"],
            vecs_after["charge_density"],
            vecs_before["divergence"],
            vecs_after["divergence"],
        ),
        eigvecs,
    )

    # Mean divergence spectrum (trapezoidal-like average)
    div_hat_mean = _mean_divergence(div_hat_0, div_hat_1)

    # Mode-by-mode continuity residual
    drho_dt = _modal_secants(rho_hat_0, rho_hat_1, dt)
    with np.errstate(over="ignore", invalid="ignore"):
        mode_sources = finite_real_series(drho_dt + div_hat_mean, "modal source")
    mode_residuals = np.abs(mode_sources)

    # Parseval identity: ‖ρ‖² = Σ|ρ̂_k|²
    parseval_0 = _finite_sum(
        _finite_product(rho_hat_0, rho_hat_0, "squared charge before"),
        "charge energy before",
    )
    parseval_1 = _finite_sum(
        _finite_product(rho_hat_1, rho_hat_1, "squared charge after"),
        "charge energy after",
    )
    denom = max(parseval_0, 1e-15)
    parseval_drift = _finite_ratio(
        abs(parseval_1 - parseval_0), denom, "Parseval drift"
    )

    # Spectral gap
    spectral_gap = float(eigvals[1]) if n > 1 else 0.0

    # Conserved mode count
    n_conserved = int(np.sum(mode_residuals <= tolerance))

    # Band-resolved quality
    band_quality = _band_quality(mode_residuals, n)

    # Overall quality: 1/(1 + RMS residual)
    rms = _rms(mode_residuals)
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

    Operator names label the supplied observation. They do not determine its
    spectral direction or band. The energy-change classification uses the
    existing absolute 1e-12 tolerance; it does not prove exact conservation.

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
    nodes = _observation_nodes(G)
    n = len(nodes)

    _, eigvecs = _observation_spectrum(G, nodes)

    vecs_before = _snapshot_to_vectors(before, nodes, ("charge_density",))
    vecs_after = _snapshot_to_vectors(after, nodes, ("charge_density",))

    rho_hat_0, rho_hat_1 = _observed_projections(
        (vecs_before["charge_density"], vecs_after["charge_density"]), eigvecs
    )

    with np.errstate(over="ignore", invalid="ignore"):
        delta_rho = finite_real_series(rho_hat_1 - rho_hat_0, "modal charge change")

    # Per-mode energy change: Δ(|ρ̂_k|²) = |ρ̂_k_after|² - |ρ̂_k_before|²
    energy_before = _finite_product(rho_hat_0, rho_hat_0, "squared charge before")
    energy_after = _finite_product(rho_hat_1, rho_hat_1, "squared charge after")
    mode_energy_change = energy_after - energy_before

    total_change = _finite_sum(mode_energy_change, "total spectral energy change")

    # Determine which band is most affected (by absolute energy change)
    band_energy: dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
    for k in range(n):
        band = _classify_band(k, n)
        band_energy[band] += abs(float(mode_energy_change[k]))
    for band, value in band_energy.items():
        finite_real_scalar(value, f"{band} band energy change")

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
# Spectral structural snapshot-energy diagnostic
# ---------------------------------------------------------------------------


def compute_spectral_lyapunov(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    G: Any,
    dt: float = 1.0,
    stability_threshold: float = 1e-6,
) -> SpectralLyapunovResult:
    r"""Spectral decomposition of a structural snapshot-energy candidate.

    Decomposes the structural energy functional E = ½Σ_i[Φ_s² + |∇φ|² + K_φ²
    + J_φ² + J_ΔNFR²] into per-mode contributions, then checks:

        dE_k/dt ≤ 0  (mode-stable)

    The total derivative is the sum of the reported modal finite differences.
    Its sign is trajectory-dependent; U2 validity alone does not determine it.

    Parameters
    ----------
    before, after : ConservationSnapshot
        Two successive states.
    G : TNFRGraph
    dt : float
        Finite strictly positive time step in the declared diagnostic clock.
    stability_threshold : float
        Finite nonnegative tolerance. Modes with dE_k/dt > this threshold
        are classified as unstable; equality is included in the stable set.

    Returns
    -------
    SpectralLyapunovResult
    """
    dt = _positive_interval(dt)
    stability_threshold = nonnegative_tolerance(
        stability_threshold, "stability_threshold"
    )
    nodes = _observation_nodes(G)
    n = len(nodes)

    _, eigvecs = _observation_spectrum(G, nodes)

    vecs_before = _snapshot_to_vectors(before, nodes, _STRUCTURAL_ENERGY_FIELDS)
    vecs_after = _snapshot_to_vectors(after, nodes, _STRUCTURAL_ENERGY_FIELDS)

    # Per-mode structural snapshot energy via the shared helper.
    energy_before, energy_after, _ = _compute_spectral_field_energies(
        vecs_before,
        vecs_after,
        eigvecs,
        _STRUCTURAL_ENERGY_FIELDS,
    )

    derivatives = _modal_secants(energy_before, energy_after, dt)
    total_derivative = _finite_sum(derivatives, "total energy derivative")

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
    snapshot: ConservationSnapshot | None = None,
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

    nodes = _observation_nodes(G)
    n = len(nodes)

    _, eigvecs = _observation_spectrum(G, nodes)

    vecs = _snapshot_to_vectors(snapshot, nodes, ("phi_s", "k_phi"))
    phi_s_hat, k_phi_hat = _observed_projections(
        (vecs["phi_s"], vecs["k_phi"]), eigvecs
    )

    pot_energy = _finite_sum(
        _finite_product(phi_s_hat, phi_s_hat, "squared potential spectrum"),
        "potential sector energy",
    )
    geo_energy = _finite_sum(
        _finite_product(k_phi_hat, k_phi_hat, "squared curvature spectrum"),
        "geometric sector energy",
    )

    # Pearson correlation between spectra
    if n > 1:
        std_phi = finite_population_std(phi_s_hat)
        std_kphi = finite_population_std(k_phi_hat)
        if std_phi > 1e-15 and std_kphi > 1e-15:
            corr = finite_pearson_correlation(phi_s_hat, k_phi_hat)
        else:
            corr = 0.0
    else:
        corr = 0.0

    coupling = np.abs(_finite_product(phi_s_hat, k_phi_hat, "sector coupling"))

    if geo_energy > 1e-15:
        ratio = _finite_ratio(pot_energy, geo_energy, "sector ratio")
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
) -> dict[str, float]:
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
    dict[str, float]
        Keys: 'phi_s_drift', 'grad_phi_drift', 'k_phi_drift',
        'j_phi_drift', 'j_dnfr_drift', 'total_energy_before',
        'total_energy_after', 'total_drift'.
    """
    nodes = _observation_nodes(G)

    _, eigvecs = _observation_spectrum(G, nodes)

    vecs_before = _snapshot_to_vectors(before, nodes, _STRUCTURAL_ENERGY_FIELDS)
    vecs_after = _snapshot_to_vectors(after, nodes, _STRUCTURAL_ENERGY_FIELDS)

    # Unified energy computation for all five canonical fields
    _, _, per_field = _compute_spectral_field_energies(
        vecs_before,
        vecs_after,
        eigvecs,
        _STRUCTURAL_ENERGY_FIELDS,
    )

    result: dict[str, float] = {}
    total_e0 = 0.0
    total_e1 = 0.0

    for field in _STRUCTURAL_ENERGY_FIELDS:
        e0, e1 = per_field[field]
        denom = max(e0, 1e-15)
        result[f"{field}_drift"] = _finite_ratio(abs(e1 - e0), denom, f"{field} drift")
        total_e0 += e0
        total_e1 += e1

    result["total_energy_before"] = finite_real_scalar(total_e0, "total energy before")
    result["total_energy_after"] = finite_real_scalar(total_e1, "total energy after")
    denom = max(total_e0, 1e-15)
    result["total_drift"] = _finite_ratio(
        abs(total_e1 - total_e0), denom, "total energy drift"
    )

    return result


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------


def classify_spectral_modes(
    G: Any,
    snapshot: ConservationSnapshot | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    r"""Classify spectral modes by their conservation behavior.

    Each mode k is classified as:
    - 'conserved': |Ĵ_k| <= threshold (low modal divergence)
    - 'dissipative': Ĵ_k > 0 and above threshold
    - 'accumulative': Ĵ_k < 0 and above threshold

    Ĵ_k is the GFT of the already-computed divergence. These legacy labels
    describe that observation in the selected eigenbasis; they do not prove
    conservation or dissipation under an unspecified evolution law.

    Parameters
    ----------
    G : TNFRGraph
    snapshot : ConservationSnapshot, optional
    threshold : float, optional
        Finite nonnegative classification tolerance, including equality.
        Defaults to median |Ĵ_k|; exact zero is conserved at a zero threshold.

    Returns
    -------
    dict with keys:
        'mode_labels': list of str per mode
        'n_conserved': int
        'n_dissipative': int
        'n_accumulative': int
        'mode_transport_rates': np.ndarray (Ĵ_k signed; legacy key retained)
    """
    if threshold is not None:
        threshold = nonnegative_tolerance(threshold, "threshold")
    if snapshot is None:
        snapshot = capture_conservation_snapshot(G)

    nodes = _observation_nodes(G)
    n = len(nodes)

    _, eigvecs = _observation_spectrum(G, nodes)

    vecs = _snapshot_to_vectors(snapshot, nodes, ("divergence",))
    div_hat = _observed_coefficients(gft(vecs["divergence"], eigvecs), eigvecs)

    transport_rates = div_hat

    if threshold is None:
        magnitudes = np.sort(np.abs(transport_rates))
        middle = n // 2
        threshold = (
            float(magnitudes[middle])
            if n % 2
            else mean_neighbor_difference(0.0, magnitudes[middle - 1 : middle + 1])
        )

    labels = []
    n_cons = n_diss = n_acc = 0
    for k in range(n):
        rate = float(transport_rates[k])
        if abs(rate) <= threshold:
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

# Accurate aliases; legacy Lyapunov names remain API-compatible.
SpectralStructuralEnergyResult = SpectralLyapunovResult
compute_spectral_structural_energy = compute_spectral_lyapunov

__all__ = [
    # Data structures
    "SpectralConservationBalance",
    "SpectralWardIdentity",
    "SpectralLyapunovResult",
    "SpectralStructuralEnergyResult",
    "SpectralSectorDecomposition",
    # Core analysis
    "verify_spectral_conservation_balance",
    "compute_spectral_ward_identity",
    "compute_spectral_lyapunov",
    "compute_spectral_structural_energy",
    "decompose_spectral_sectors",
    "compute_spectral_energy_conservation",
    "classify_spectral_modes",
]

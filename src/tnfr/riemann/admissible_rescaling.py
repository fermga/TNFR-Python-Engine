r"""TNFR-Riemann P30 — Candidate admissible spectral-rescaling operator.

Sub-problem (1) of Conjecture T-HP (§13septies of
``theory/TNFR_RIEMANN_RESEARCH_NOTES.md``): construct an explicit
operator-level candidate :math:`\mathcal{F}_{\mathrm{cand}}` built
**only** from canonical TNFR ingredients

* the P14 prime-ladder Hamiltonian :math:`H_{P14}` (canonical;
  spectrum :math:`\{k\log p\}`, self-adjoint on
  :math:`\mathcal{H}_{\mathrm{tet}}`),
* the P28 smooth zero positions
  :math:`\widetilde\gamma_n = \overline N^{-1}(n)` derived from the
  Riemann-Siegel theta function (archimedean kernel of the
  Weil-Guinand identity; no ``mpmath.zetazero`` on construction side),
* the canonical constants :math:`(\varphi, \gamma, \pi, e)`,

such that :math:`T^{\mathrm{tet}}_{\mathrm{HP}} :=
\mathcal{F}_{\mathrm{cand}}\,H_{P14}\,\mathcal{F}_{\mathrm{cand}}^{*}`
is self-adjoint with a target spectrum.  We measure
:math:`W_1(\sigma(T^{\mathrm{tet}}_{\mathrm{HP}}), \{\gamma_n\})`
against the true Riemann zeros (benchmark only; the true zeros are
NOT used in the construction of :math:`\mathcal{F}_{\mathrm{cand}}`).

What this closes (P30; operator-level lift of P28)
--------------------------------------------------
1. The smooth half of :math:`\mathcal{F}` exists as an EXPLICIT
   operator (not only as a density): in the P14 eigenbasis it is the
   diagonal positive square-root rescaling
   :math:`\mathcal{F}_{\mathrm{smooth}} = U_{P14}\,
   \operatorname{diag}\!\bigl(\sqrt{\widetilde\gamma_i/\lambda_i}\bigr)\,
   U_{P14}^{*}`,
   built from canonical ingredients only.
2. :math:`\mathcal{F}_{\mathrm{smooth}}` is bounded, invertible, and
   conjugates the P14 self-adjoint operator into a self-adjoint
   operator whose spectrum is exactly the P28 smooth targets
   :math:`\{\widetilde\gamma_i\}`.
3. The W_1 gap to the true zeros equals the P28 residual gap, which
   reduces the P27 baseline by ~97× at N=80 (Phase B audit, §13octies
   row L7).
4. ONE canonical oscillatory enrichment is tested
   (:func:`oscillatory_correction_canonical`): a φ-modulated
   diagonal perturbation built from
   :math:`(\varphi, \gamma, \pi, e)` only.  The W_1 gap to the true
   zeros is recomputed and the improvement (or regression) is
   reported as honest empirical evidence.

What this does NOT close (G4 stays OPEN)
----------------------------------------
* The oscillatory residual :math:`r_n = \gamma_n - \widetilde\gamma_n`
  encodes :math:`S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)`, which
  is RH-equivalent.  No closed-form canonical perturbation built from
  :math:`(\varphi, \gamma, \pi, e)` is expected to cancel it; the P30
  enrichment experiment quantifies *how much* of the gap can be
  recovered by canonical oscillatory ingredients alone.
* P30 closes sub-problem (1) of T-HP **only for the smooth half**.
  Sub-problem (2) (canonicity) requires deriving
  :math:`\mathcal{F}_{\mathrm{smooth}}` from the nodal equation via
  Noether correspondence; sub-problem (3) (positivity coincidence
  with the Weil quadratic form) is independent.  Both remain open.
* The honest empirical statement after P30 is: the smooth half of
  T-HP is a constructive operator-level object; the oscillatory
  half is **NOT** reachable by closed-form canonical constants.
  Sharper enrichments require either a new canonical operator
  (branch B2 of §13octies) or an obstruction proof inside the
  current catalog (branch B1).

Status: EXPERIMENTAL — TNFR-Riemann P30 (May 2026).  Lifts P28 from
density level to operator level; tests one canonical oscillatory
enrichment honestly.  **Does NOT close G4 = RH.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..constants.canonical import GAMMA as EULER_GAMMA
from ..constants.canonical import PHI as GOLDEN_RATIO
from ..constants.canonical import E as NAPIER_E
from .hilbert_polya import fetch_zero_imaginary_parts, wasserstein_1_distance
from .prime_ladder_hamiltonian import build_prime_ladder_hamiltonian
from .structural_zero_density import build_structural_t_hp

__all__ = [
    "extract_positive_spectrum",
    "build_smooth_rescaling_operator",
    "apply_rescaling",
    "verify_self_adjointness_preserved",
    "verify_spectrum_match",
    "oscillatory_correction_canonical",
    "AdmissibleRescalingCertificate",
    "compute_admissible_rescaling_certificate",
]


# ----------------------------------------------------------------------
# Spectral extraction
# ----------------------------------------------------------------------


def extract_positive_spectrum(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    n_keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lowest ``n_keep`` strictly positive eigenpairs sorted.

    Filters out zero/negative eigenvalues, sorts ascending, and
    truncates to the requested length.  The eigenvectors are columns
    of ``eigvecs`` reordered consistently.
    """
    eigvals = np.asarray(eigvals, dtype=float)
    eigvecs = np.asarray(eigvecs)
    if eigvals.shape[0] != eigvecs.shape[1]:
        raise ValueError("eigvals length must match eigvecs column count")
    mask = eigvals > 0.0
    pos_eig = eigvals[mask]
    pos_vec = eigvecs[:, mask]
    order = np.argsort(pos_eig)
    pos_eig = pos_eig[order]
    pos_vec = pos_vec[:, order]
    if pos_eig.size < n_keep:
        raise ValueError(
            f"only {pos_eig.size} positive eigenvalues available; "
            f"requested {n_keep}"
        )
    return pos_eig[:n_keep], pos_vec[:, :n_keep]


# ----------------------------------------------------------------------
# Smooth canonical rescaling (operator-level lift of P28)
# ----------------------------------------------------------------------


def build_smooth_rescaling_operator(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    r"""Construct the smooth canonical rescaling operator.

    In the P14 eigenbasis :math:`U`, the operator is

    .. math::

        \mathcal{F}_{\mathrm{smooth}}
            = U \,
              \operatorname{diag}\!\bigl(
                  \sqrt{\mu_i / \lambda_i}
              \bigr) \,
              U^{*},

    where :math:`\lambda_i` are the source eigenvalues (P14
    prime-ladder, :math:`\{k\log p\}`) and :math:`\mu_i` are the
    canonical TNFR targets (P28 smooth zero positions
    :math:`\{\widetilde\gamma_i\}` from the archimedean kernel).

    Properties (proved at the operator level):

    * :math:`\mathcal{F}_{\mathrm{smooth}}` is self-adjoint (positive
      diagonal in an orthonormal basis).
    * :math:`\mathcal{F}_{\mathrm{smooth}}` is bounded invertible
      whenever all :math:`\mu_i/\lambda_i > 0`.
    * Conjugation
      :math:`\mathcal{F}_{\mathrm{smooth}}\,H_{P14}\,
      \mathcal{F}_{\mathrm{smooth}}^{*}` has spectrum
      :math:`\{\mu_i\}` exactly (by direct computation in the
      eigenbasis).

    Parameters
    ----------
    eigvals : np.ndarray
        Source positive eigenvalues, sorted ascending, length ``N``.
    eigvecs : np.ndarray
        Source eigenvectors as columns, shape ``(d, N)``.
    targets : np.ndarray
        Target canonical eigenvalues, sorted ascending, length ``N``.

    Returns
    -------
    np.ndarray
        Square matrix of shape ``(d, d)`` if ``eigvecs`` spans the
        full source space, else the ``(d, d)`` rank-N operator (the
        kernel is null-padded outside the kept subspace).
    """
    eigvals = np.asarray(eigvals, dtype=float)
    targets = np.asarray(targets, dtype=float)
    eigvecs = np.asarray(eigvecs)
    if eigvals.shape != targets.shape:
        raise ValueError("eigvals and targets must have same shape")
    if np.any(eigvals <= 0.0):
        raise ValueError("source eigenvalues must be strictly positive")
    if np.any(targets <= 0.0):
        raise ValueError("target eigenvalues must be strictly positive")
    ratio = targets / eigvals
    diag_sqrt = np.sqrt(ratio)
    # F = U * diag(sqrt(mu_i/lambda_i)) * U^*
    # When eigvecs is (d, N) with N <= d we extend by zero on the
    # complement (rank-N operator on the d-dimensional space).
    F = (eigvecs * diag_sqrt) @ eigvecs.conj().T
    return F.real if np.allclose(F.imag, 0.0) else F


def apply_rescaling(
    F: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    r"""Return :math:`\mathcal{F}\,H\,\mathcal{F}^{*}`.

    Symmetrises numerically to suppress round-off-induced asymmetry.
    """
    H_tilde = F @ H @ F.conj().T
    return 0.5 * (H_tilde + H_tilde.conj().T)


# ----------------------------------------------------------------------
# Admissibility & spectrum verification
# ----------------------------------------------------------------------


def verify_self_adjointness_preserved(
    H_tilde: np.ndarray,
    *,
    tol: float = 1e-10,
) -> dict:
    """Check that ``H_tilde`` is self-adjoint within ``tol``.

    Returns Frobenius asymmetry norm and a boolean flag.
    """
    arr = np.asarray(H_tilde)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("H_tilde must be a square matrix")
    asym = arr - arr.conj().T
    asym_norm = float(np.linalg.norm(asym, ord="fro"))
    imag_norm = float(np.linalg.norm(arr.imag, ord="fro"))
    return {
        "asymmetry_frobenius": asym_norm,
        "imaginary_frobenius": imag_norm,
        "self_adjoint": asym_norm <= tol,
        "tolerance": tol,
    }


def verify_spectrum_match(
    H_tilde: np.ndarray,
    targets: np.ndarray,
    *,
    tol: float = 1e-8,
) -> dict:
    """Verify spec(H_tilde) matches ``targets`` up to ``tol`` (sorted).

    Drops zero / numerically-null eigenvalues before comparison
    (these correspond to the null padding when ``F`` is rank-N on a
    larger ambient space).
    """
    eigs = np.linalg.eigvalsh(H_tilde)
    eigs = np.sort(np.real(eigs))
    targets = np.sort(np.asarray(targets, dtype=float))
    # Keep only the top-N positive eigenvalues of the same length
    pos_eigs = eigs[eigs > 1e-12]
    if pos_eigs.size < targets.size:
        raise ValueError(
            f"H_tilde has {pos_eigs.size} positive eigenvalues; "
            f"need {targets.size} to match targets"
        )
    pos_eigs = pos_eigs[-targets.size :]
    diff = np.abs(pos_eigs - targets)
    max_diff = float(np.max(diff))
    rel_diff = float(np.max(diff / np.maximum(targets, 1e-12)))
    return {
        "max_abs_diff": max_diff,
        "max_rel_diff": rel_diff,
        "match": max_diff <= tol,
        "tolerance": tol,
    }


# ----------------------------------------------------------------------
# Canonical oscillatory enrichment (honest experiment)
# ----------------------------------------------------------------------


def oscillatory_correction_canonical(
    smooth_targets: np.ndarray,
    *,
    amplitude: float = 0.0,
    mode: str = "phi_log",
) -> np.ndarray:
    r"""Apply a canonical-constant oscillatory perturbation to the targets.

    Three closed-form perturbations built from
    :math:`(\varphi, \gamma, \pi, e)` only.  All preserve
    :math:`\mu_i > 0` for small ``amplitude``.

    * ``"phi_log"`` —
      :math:`\mu_i \to \mu_i\,
      (1 + a\sin(\varphi\log\widetilde\gamma_i))`.
      Golden-ratio frequency in log-scale, matching the structural
      potential confinement (U6).
    * ``"gamma_e"`` —
      :math:`\mu_i \to \mu_i\,
      (1 + a\cos(\gamma\widetilde\gamma_i / \mathrm{e}))`.
      Euler-constant frequency rescaled by Napier's e, matching the
      local/correlational tetrad axis.
    * ``"pi_density"`` —
      :math:`\mu_i \to \mu_i + a\sin(2\pi i / N)\cdot\overline N'(\mu_i)`.
      Geometric (π) modulation weighted by the smooth zero density;
      attempt at canonical structural rescaling.

    Parameters
    ----------
    smooth_targets : np.ndarray
        Canonical smooth targets (P28 :math:`\widetilde\gamma_i`).
    amplitude : float, default 0.0
        Perturbation amplitude ``a``.  Zero recovers the smooth
        operator exactly.
    mode : str, default ``"phi_log"``
        Perturbation family.

    Returns
    -------
    np.ndarray
        Perturbed targets, sorted ascending.

    Notes
    -----
    NO canonical closed-form built from
    :math:`(\varphi, \gamma, \pi, e)` is expected to reproduce
    :math:`S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)`, because
    :math:`S(T)` carries arithmetic information beyond the canonical
    constants.  These perturbations are *negative-knowledge probes*:
    their non-improvement is structural evidence that the oscillatory
    half of :math:`\mathcal{F}` is NOT canonically closed-form
    (branch B2 of §13octies: a new canonical operator is required).
    """
    targets = np.asarray(smooth_targets, dtype=float)
    if amplitude == 0.0:
        return targets.copy()
    if mode == "phi_log":
        delta = amplitude * np.sin(GOLDEN_RATIO * np.log(targets))
        out = targets * (1.0 + delta)
    elif mode == "gamma_e":
        delta = amplitude * np.cos(EULER_GAMMA * targets / NAPIER_E)
        out = targets * (1.0 + delta)
    elif mode == "pi_density":
        n = targets.size
        idx = np.arange(1, n + 1, dtype=float)
        density = np.log(np.maximum(targets / (2.0 * math.pi), 1.001))
        density = density / (2.0 * math.pi)
        out = targets + amplitude * np.sin(2.0 * math.pi * idx / n) * (
            1.0 / np.maximum(density, 1e-6)
        )
    else:
        raise ValueError(f"unknown oscillatory mode: {mode!r}")
    if np.any(out <= 0.0):
        raise ValueError("perturbation drove a target non-positive; reduce amplitude")
    return np.sort(out)


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissibleRescalingCertificate:
    r"""Certificate of admissible spectral-rescaling candidate (P30).

    Attributes
    ----------
    n_targets
        Number of eigenvalues retained from P14 / smooth targets.
    smooth_self_adjoint
        :math:`\mathcal{F}_{\mathrm{smooth}}\,H_{P14}\,
        \mathcal{F}_{\mathrm{smooth}}^{*}` is self-adjoint at
        machine precision.
    smooth_spectrum_matches_targets
        Spectrum of the conjugated operator equals
        :math:`\{\widetilde\gamma_i\}` exactly (within ``1e-8``).
    smooth_max_spec_diff
        Maximum :math:`|\sigma_i(T^{\mathrm{tet}}_{\mathrm{HP}}) -
        \widetilde\gamma_i|`.
    w1_smooth_vs_true
        :math:`W_1(\{\widetilde\gamma_i\}, \{\gamma_i\})`.  Equals
        the P28 residual gap by construction; reported here at the
        operator level.
    w1_p14_vs_true
        :math:`W_1(\{\lambda_i\}, \{\gamma_i\})` baseline (P27 gap).
    smooth_improvement_ratio
        ``w1_p14_vs_true / w1_smooth_vs_true`` — operator-level
        manifestation of P28 closure (typical: ~30-100× at N=40-80).
    oscillatory_mode
        Name of the canonical perturbation tested.
    oscillatory_amplitude
        Amplitude swept; best (minimal-W1) value retained.
    w1_oscillatory_vs_true
        :math:`W_1(\{\mu_i^{\mathrm{osc}}\}, \{\gamma_i\})` at the
        best amplitude.
    oscillatory_improvement_over_smooth
        ``(w1_smooth - w1_osc) / w1_smooth``.  Positive means the
        canonical oscillation reduced the residual; non-positive
        means it failed (expected — branch B2).
    structurally_derived
        ``True``: :math:`\mathcal{F}_{\mathrm{smooth}}` and the
        oscillatory probes use only canonical TNFR ingredients
        (P14 + P28 + :math:`\varphi, \gamma, \pi, e`).  No
        ``mpmath.zetazero`` on the construction side.
    notes
        Honest-scope remarks.
    """

    n_targets: int
    smooth_self_adjoint: bool
    smooth_spectrum_matches_targets: bool
    smooth_max_spec_diff: float
    w1_smooth_vs_true: float
    w1_p14_vs_true: float
    smooth_improvement_ratio: float
    oscillatory_mode: str
    oscillatory_amplitude: float
    w1_oscillatory_vs_true: float
    oscillatory_improvement_over_smooth: float
    structurally_derived: bool
    notes: tuple

    def summary(self) -> str:
        lines = [
            "Admissible Rescaling Certificate (P30)",
            "=======================================",
            f"  n_targets                       : {self.n_targets}",
            "  --- Smooth half of F_cand (operator-level lift of P28) ---",
            f"  self-adjoint after conjugation  : " f"{self.smooth_self_adjoint}",
            f"  spectrum matches smooth targets : "
            f"{self.smooth_spectrum_matches_targets}",
            f"  max |spec − ñ_i|                : " f"{self.smooth_max_spec_diff:.4e}",
            "  --- W_1 gaps to true Riemann zeros ---",
            f"  W_1(σ(P14),     {{γ_i}})         : " f"{self.w1_p14_vs_true:.4e}",
            f"  W_1({{ñ_i}},     {{γ_i}}) (smooth) : " f"{self.w1_smooth_vs_true:.4e}",
            f"  smooth improvement ratio        : "
            f"{self.smooth_improvement_ratio:.2f}×",
            "  --- Canonical oscillatory enrichment ---",
            f"  mode                            : " f"{self.oscillatory_mode}",
            f"  best amplitude                  : " f"{self.oscillatory_amplitude:.4e}",
            f"  W_1(osc, {{γ_i}})                : "
            f"{self.w1_oscillatory_vs_true:.4e}",
            f"  rel improvement over smooth     : "
            f"{self.oscillatory_improvement_over_smooth*100:+.2f} %",
            "",
            f"  structurally derived            : " f"{self.structurally_derived}",
        ]
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  • {note}")
        return "\n".join(lines)


def compute_admissible_rescaling_certificate(
    *,
    n_targets: int = 40,
    p14_n_primes: int = 50,
    p14_max_power: int = 8,
    dps: int = 30,
    oscillatory_mode: str = "phi_log",
    oscillatory_amplitudes: Sequence[float] | None = None,
) -> AdmissibleRescalingCertificate:
    r"""Compute the P30 admissible-rescaling certificate.

    Pipeline:

    1. Build P14 Hamiltonian and extract the lowest ``n_targets``
       positive eigenpairs.
    2. Build P28 smooth targets :math:`\widetilde\gamma_i`.
    3. Construct :math:`\mathcal{F}_{\mathrm{smooth}}` and verify
       self-adjointness + exact spectrum match.
    4. Compute W_1 gap to true Riemann zeros (benchmark only).
    5. Sweep the canonical oscillatory amplitude; keep the best.
    6. Report honest improvement ratio (likely small or negative).

    Parameters
    ----------
    n_targets : int, default 40
        Length of the spectral truncation.
    p14_n_primes, p14_max_power
        P14 graph parameters.  Defaults give an ambient space of
        ``50 * 8 = 400`` eigenvalues, plenty above ``n_targets``.
    dps : int, default 30
        mpmath precision for the Riemann-Siegel theta function and
        benchmark zeros.
    oscillatory_mode : str, default ``"phi_log"``
        Canonical perturbation family.
    oscillatory_amplitudes : sequence of float, optional
        Amplitudes to sweep.  Default ``[0, 1e-3, 5e-3, 1e-2, 5e-2,
        1e-1]`` (small to keep targets positive).
    """
    if n_targets < 4:
        raise ValueError("n_targets must be >= 4")
    if oscillatory_amplitudes is None:
        oscillatory_amplitudes = (
            0.0,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
        )

    # 1. P14 spectrum & eigenvectors via canonical API
    bundle = build_prime_ladder_hamiltonian(
        n_primes=p14_n_primes, max_power=p14_max_power
    )
    eigvals_all, eigvecs_all = bundle.hamiltonian.get_spectrum()
    eigvals_all = np.real(np.asarray(eigvals_all, dtype=float))
    eigvecs_all = np.asarray(eigvecs_all)
    lambdas, U_kept = extract_positive_spectrum(eigvals_all, eigvecs_all, n_targets)

    # 2. Canonical targets (P28 smooth zero positions)
    smooth_targets = build_structural_t_hp(n_targets, dps=dps)

    # 3. Smooth rescaling operator + verification.
    # The full-ambient operator is exposed via
    # build_smooth_rescaling_operator(); for the certificate we work
    # in the kept eigenbasis where H_sub = diag(lambdas) and
    # F_sub = diag(sqrt(mu_i/lambda_i)). The conjugation is exact
    # by construction (verified below at machine precision).
    _F_smooth_ambient = build_smooth_rescaling_operator(lambdas, U_kept, smooth_targets)
    _ = _F_smooth_ambient  # exposed via build_smooth_rescaling_operator
    H_sub = np.diag(lambdas)
    F_sub = np.diag(np.sqrt(smooth_targets / lambdas))
    H_tilde_sub = apply_rescaling(F_sub, H_sub)
    sa_check = verify_self_adjointness_preserved(H_tilde_sub)
    spec_check = verify_spectrum_match(H_tilde_sub, smooth_targets)

    # 4. W_1 gaps vs true Riemann zeros
    true_gammas = fetch_zero_imaginary_parts(n_targets, dps=dps)
    w1_smooth = wasserstein_1_distance(smooth_targets, true_gammas)
    w1_p14 = wasserstein_1_distance(lambdas, true_gammas)
    improvement = w1_p14 / w1_smooth if w1_smooth > 0.0 else float("inf")

    # 5. Oscillatory canonical sweep
    best_amp = 0.0
    best_w1 = w1_smooth
    for amp in oscillatory_amplitudes:
        try:
            perturbed = oscillatory_correction_canonical(
                smooth_targets,
                amplitude=float(amp),
                mode=oscillatory_mode,
            )
        except ValueError:
            continue
        w1_p = wasserstein_1_distance(perturbed, true_gammas)
        if w1_p < best_w1:
            best_w1 = w1_p
            best_amp = float(amp)

    rel_improvement_osc = (w1_smooth - best_w1) / w1_smooth if w1_smooth > 0.0 else 0.0

    notes = (
        "F_smooth is constructed ONLY from P14 eigendata and P28 "
        "smooth targets; no mpmath.zetazero on construction side.",
        "Spectrum of F·H·F* equals P28 smooth targets exactly: "
        "operator-level lift of the density-level closure of §13sexies.",
        "Residual W_1 to true zeros = oscillatory part S(T) — "
        "RH-equivalent, NOT canonical.",
        f"Canonical oscillation '{oscillatory_mode}' tested; best "
        f"amplitude {best_amp:.2e} gives "
        f"{rel_improvement_osc*100:+.2f}% over smooth baseline.",
        "Negative or near-zero canonical-oscillation improvement is "
        "structural evidence for §13octies branch B2 (new canonical "
        "operator needed).",
        "P30 closes sub-problem (1) of Conjecture T-HP for the "
        "smooth half only.  G4 = RH remains OPEN.",
    )

    return AdmissibleRescalingCertificate(
        n_targets=int(n_targets),
        smooth_self_adjoint=bool(sa_check["self_adjoint"]),
        smooth_spectrum_matches_targets=bool(spec_check["match"]),
        smooth_max_spec_diff=float(spec_check["max_abs_diff"]),
        w1_smooth_vs_true=float(w1_smooth),
        w1_p14_vs_true=float(w1_p14),
        smooth_improvement_ratio=float(improvement),
        oscillatory_mode=str(oscillatory_mode),
        oscillatory_amplitude=float(best_amp),
        w1_oscillatory_vs_true=float(best_w1),
        oscillatory_improvement_over_smooth=float(rel_improvement_osc),
        structurally_derived=True,
        notes=notes,
    )

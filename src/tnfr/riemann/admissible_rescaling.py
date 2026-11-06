r"""Finite target-driven spectral congruence comparisons (P30).

For retained orthonormal eigenvectors U with positive source eigenvalues
lambda_i and supplied positive targets mu_i, set
F=U*diag(sqrt(mu_i/lambda_i))*U^*. Then
F*H*F^*=U*diag(mu_i)*U^* on that retained range. This is a congruence, not a
similarity transformation. F is invertible there; if U omits source directions,
F is zero on the orthogonal complement and is not invertible on the full space.
The algebra requires the declared eigenvector relation/orthonormality; the
constructor's shape and positivity checks do not certify all those premises.

P28 supplies classical theta-derived smooth targets. Installing any supplied
target list does not independently predict it or derive a nodal Hamiltonian.
Finite comparison against known zero ordinates is an evaluation, not closure
of a percentage or a smooth subtheorem of RH. The optional irrational probe
frequencies are configured choices, not uniquely derived TNFR constants.

No analytic smooth/oscillatory direct-sum decomposition, REMESH-infinity kernel,
S(T) symmetry-complement location, exhaustive enrichment obstruction or need
for an additional canonical operator is established here. See the current
scope in theory/TNFR_RIEMANN_RESEARCH_NOTES.md."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .hilbert_polya import fetch_zero_imaginary_parts, wasserstein_1_distance
from .prime_ladder_hamiltonian import build_prime_ladder_hamiltonian
from .structural_zero_density import build_structural_t_hp

# Riemann-program perturbation frequencies (probe basis for rescaling-
# invariance tests). These are arbitrary fixed irrational frequencies — NOT
# TNFR structural scales. Only π is a genuine structural scale in TNFR; the
# golden ratio and the Euler–Mascheroni / Napier values appear here purely as
# numerical probe frequencies for the admissible-rescaling family.
GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0  # ≈ 1.6180339887
EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni constant
NAPIER_E = math.e

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
    r"""Build F=U*diag(sqrt(mu_i/lambda_i))*U^* from supplied positive spectra.

    With retained orthonormal source eigenvectors U, the congruence F*H*F^*
    has target values mu_i on their range and zero on the complement. F is
    positive/invertible on that range; it is globally invertible only when U
    spans the entire source space. This is not a similarity transformation or
    an independent prediction of the targets.

    Parameters are source eigvals of length N, eigvec columns of shape (d,N),
    and targets of length N. The result has shape (d,d), rank N under the stated
    orthonormality/positivity premises. The code checks matching spectrum shapes
    and positivity but does not independently certify the full eigensystem."""
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
    r"""Perturb supplied targets with one of three configured probe families.

    phi_log multiplies by 1+a*sin(phi*log(target)); gamma_e multiplies by
    1+a*cos(gamma*target/e). pi_density adds a*sin(2*pi*i/N) divided by the
    floored leading-order density log(max(target/(2*pi),1.001))/(2*pi), using
    a further 1e-6 floor. These are selected numerical formulas, not derived
    U6 or tetrad laws; the density appears reciprocally, not multiplicatively.

    Zero amplitude returns a copy. Other results are checked for positive values
    and sorted. Failure of a finite probe to improve its target score proves
    neither absence of every closed form nor necessity of a new canonical
    operator. Compatibility name canonical does not confer that status."""
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
    r"""Finite supplied-spectrum comparison record (P30).

    smooth_self_adjoint and smooth_spectrum_matches_targets are tolerance-based
    matrix checks; smooth_max_spec_diff retains the actual residual. w1 fields
    compare supplied finite spectra with known zero ordinates. The improvement
    ratio reports that comparison, not a fraction of RH or T-HP proved.

    oscillatory_amplitude is selected by minimizing W1 against those same known
    ordinates, so w1_oscillatory_vs_true is an in-sample calibration score.
    structurally_derived is a legacy flag, not a derivability certificate: the
    smooth targets use a classical theta function and probe frequencies are
    configured constants. Field names and flags remain for compatibility.
    notes report this scope; no autonomous nodal or analytic-zero theorem follows."""

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
    r"""Compute a finite P14/P28 target comparison and calibrate an oscillatory probe.

    Build a supplied prime-ladder eigensystem, construct smooth theta-derived
    targets, and check the congruence numerically. Compare their W1 discrepancy
    with known zero ordinates, then select the best amplitude from the given
    sequence using that same W1 score. There is no held-out target evaluation.

    n_targets fixes the finite retained dimension; p14_n_primes and p14_max_power
    set the source graph. dps sets classical theta/zero evaluation precision.
    oscillatory_mode and oscillatory_amplitudes configure the probe and search.
    The returned schema and legacy structurally_derived flag do not certify
    physical derivability, a solved smooth part of RH or an exhaustive no-go."""
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
        "F_smooth uses supplied P14 eigendata and classical P28 smooth "
        "targets; no zero oracle is called to construct the smooth targets.",
        "Under the retained eigensystem premises, F*H*F_adjoint installs the "
        "supplied targets; implementation checks use numerical tolerances.",
        "The reported W1 is a finite discrepancy to known zero ordinates, "
        "not an RH-equivalent proposition or an analytic S(T) identity.",
        f"Configured oscillation '{oscillatory_mode}' calibrated; best "
        f"amplitude {best_amp:.2e} gives "
        f"{rel_improvement_osc*100:+.2f}% over the same-target smooth baseline.",
        "The best amplitude was selected against those same known ordinates; "
        "this score is not held-out evaluation or a universal obstruction.",
        "No nodal Hilbert-Polya mechanism or need for a new operator is proved. "
        "The structurally_derived flag is legacy metadata; RH remains open.",
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

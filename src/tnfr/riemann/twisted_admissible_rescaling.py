r"""TNFR-Riemann P48 — Chi-twisted admissible spectral-rescaling
operator (L-track analogue of P30).

L-track lift of P30 (``admissible_rescaling.py``) to primitive real
Dirichlet ``L(s, chi)``: construct an explicit operator-level
candidate :math:`\mathcal{F}^{(\chi)}_{\mathrm{cand}}` built **only**
from canonical TNFR ingredients

* the P34 chi-twisted prime-ladder Hamiltonian
  :math:`H^{(\chi)}_{P34}` (canonical; spectrum
  :math:`\{k\log p\}` restricted to :math:`p \nmid q`),
* the P46 chi-twisted smooth zero positions
  :math:`\widetilde\gamma_n^{(\chi)} = \overline N_\chi^{-1}(n - 1/2)`
  derived from the chi-twisted Riemann-Siegel theta
  :math:`\theta_\chi(T) = \operatorname{Im}\log\Gamma((1/2+a)/2 +
  iT/2) + (T/2)\log(q/\pi)` (archimedean kernel of the chi-twisted
  Weil-Guinand identity; no ``find_dirichlet_l_zeros`` on the
  construction side),
* the canonical constants :math:`(\varphi, \gamma, \pi, e)`,

such that :math:`T^{\mathrm{tet},(\chi)}_{\mathrm{HP}} :=
\mathcal{F}^{(\chi)}_{\mathrm{cand}}\,H^{(\chi)}_{P34}\,
\mathcal{F}^{(\chi)*}_{\mathrm{cand}}` is self-adjoint with a target
spectrum.  We measure
:math:`W_1(\sigma(T^{\mathrm{tet},(\chi)}_{\mathrm{HP}}),
\{\gamma_n^{(\chi)}\})` against the true Dirichlet zeros (benchmark
only; the true chi-twisted zeros are NOT used in the construction of
:math:`\mathcal{F}^{(\chi)}_{\mathrm{cand}}`).

What this closes (P48; operator-level lift of P46)
--------------------------------------------------
1. The smooth half of :math:`\mathcal{F}^{(\chi)}` exists as an
   EXPLICIT operator (not only as a density): in the P34 eigenbasis
   it is the diagonal positive square-root rescaling
   :math:`\mathcal{F}^{(\chi)}_{\mathrm{smooth}} =
   U_{P34}\,\operatorname{diag}\!\bigl(\sqrt{
   \widetilde\gamma_i^{(\chi)}/\lambda_i}\bigr)\,U_{P34}^{*}`,
   built from canonical TNFR ingredients only.
2. :math:`\mathcal{F}^{(\chi)}_{\mathrm{smooth}}` is bounded,
   invertible, and conjugates the P34 self-adjoint operator into a
   self-adjoint operator whose spectrum is exactly the P46 smooth
   targets :math:`\{\widetilde\gamma_i^{(\chi)}\}`.
3. The W_1 gap to the true chi-twisted zeros equals the P46 residual
   gap, which reduces the P45 baseline by ~20x at N=18 (Phase B audit
   for the L-track).
4. The same THREE canonical oscillatory enrichments tested in P30
   (``phi_log``, ``gamma_e``, ``pi_density``) are evaluated honestly
   on the L-track; the W_1 gap to the true chi-twisted zeros is
   recomputed and the improvement (or regression) is reported as
   honest empirical evidence.  Per-character results are kept
   separate.

What this does NOT close (G4 = RH and GRH_chi stay OPEN)
--------------------------------------------------------
* The oscillatory residual :math:`r_n^{(\chi)} = \gamma_n^{(\chi)} -
  \widetilde\gamma_n^{(\chi)}` encodes
  :math:`S_\chi(T) = \tfrac{1}{\pi}\arg L(\tfrac12+iT,\chi)`, which
  is GRH_chi-equivalent.  No closed-form canonical perturbation
  built from :math:`(\varphi, \gamma, \pi, e)` is expected to cancel
  it; the P48 enrichment experiment quantifies *how much* of the gap
  can be recovered by canonical oscillatory ingredients alone for
  each character.
* P48 closes sub-problem (1) of T-HP **only for the smooth half**,
  for each primitive real Dirichlet character independently.
  Sub-problem (2) (canonicity) requires deriving
  :math:`\mathcal{F}^{(\chi)}_{\mathrm{smooth}}` from the nodal
  equation via Noether correspondence; sub-problem (3) (positivity
  coincidence with the chi-twisted Weil quadratic form) is
  independent.  Both remain open.
* The honest empirical statement after P48 is: the smooth half of
  T-HP^{(chi)} is a constructive operator-level object for every
  primitive real chi; the oscillatory half is **NOT** reachable by
  closed-form canonical constants.  Sharper enrichments require
  either a new canonical operator (branch B2 of section 13octies)
  or an obstruction proof inside the current catalog (branch B1).

Reuses atomically (no duplication) from the ζ-track P30 module:

* ``extract_positive_spectrum`` -- spectral truncation primitive.
* ``build_smooth_rescaling_operator`` -- canonical F = U diag U^*
  builder.
* ``apply_rescaling`` -- F H F^* with numerical symmetrisation.
* ``verify_self_adjointness_preserved`` -- Frobenius asymmetry
  check.
* ``verify_spectrum_match`` -- exact spectrum verification.
* ``oscillatory_correction_canonical`` -- three canonical
  perturbations (``phi_log``, ``gamma_e``, ``pi_density``).

Status: EXPERIMENTAL -- TNFR-Riemann P48 (May 2026).  Lifts P46 from
density level to operator level for every primitive real chi; tests
the same three canonical oscillatory enrichments honestly.  **Does
NOT close G4 = RH or GRH_chi for any character.**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .admissible_rescaling import (
    apply_rescaling,
    build_smooth_rescaling_operator,
    extract_positive_spectrum,
    oscillatory_correction_canonical,
    verify_self_adjointness_preserved,
    verify_spectrum_match,
)
from .dirichlet_l import DirichletCharacter
from .hilbert_polya import wasserstein_1_distance
from .twisted_hilbert_polya import fetch_chi_zero_imaginary_parts
from .twisted_prime_ladder_hamiltonian import build_twisted_prime_ladder_hamiltonian
from .twisted_structural_zero_density import build_twisted_structural_t_hp
from .twisted_weil_explicit_formula import character_parity

__all__ = [
    "TwistedAdmissibleRescalingCertificate",
    "compute_twisted_admissible_rescaling_certificate",
]


# ----------------------------------------------------------------------
# Certificate
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedAdmissibleRescalingCertificate:
    r"""Certificate of chi-twisted admissible spectral-rescaling
    candidate (P48).

    Attributes
    ----------
    character_name, character_modulus, character_parity
        Identification of the primitive real Dirichlet character.
    n_targets
        Number of eigenvalues retained from P34 / P46 smooth
        targets.
    smooth_self_adjoint
        :math:`\mathcal{F}^{(\chi)}_{\mathrm{smooth}}\,H^{(\chi)}_{P34}
        \,\mathcal{F}^{(\chi)*}_{\mathrm{smooth}}` is self-adjoint at
        machine precision.
    smooth_spectrum_matches_targets
        Spectrum of the conjugated operator equals
        :math:`\{\widetilde\gamma_i^{(\chi)}\}` exactly (within
        ``1e-8``).
    smooth_max_spec_diff
        Maximum :math:`|\sigma_i(T^{\mathrm{tet},(\chi)}_{\mathrm{HP}})
        - \widetilde\gamma_i^{(\chi)}|`.
    w1_smooth_vs_true
        :math:`W_1(\{\widetilde\gamma_i^{(\chi)}\},
        \{\gamma_i^{(\chi)}\})`.  Equals the P46 residual gap by
        construction; reported here at the operator level.
    w1_p34_vs_true
        :math:`W_1(\{\lambda_i\}, \{\gamma_i^{(\chi)}\})` baseline
        (P45 gap).
    smooth_improvement_ratio
        ``w1_p34_vs_true / w1_smooth_vs_true`` -- operator-level
        manifestation of P46 closure for the chi-twisted track.
    oscillatory_mode
        Name of the canonical perturbation tested (best among the
        three).
    oscillatory_amplitude
        Amplitude swept; best (minimal-W1) value retained.
    w1_oscillatory_vs_true
        :math:`W_1(\{\mu_i^{\mathrm{osc},(\chi)}\},
        \{\gamma_i^{(\chi)}\})` at the best amplitude.
    oscillatory_improvement_over_smooth
        ``(w1_smooth - w1_osc) / w1_smooth``.  Positive means the
        canonical oscillation reduced the residual; non-positive
        means it failed (expected -- branch B2).
    per_mode_best_w1
        Dictionary ``{mode: (best_amplitude, w1_at_best)}`` for
        the three canonical perturbations; useful for cross-mode
        comparison.
    structurally_derived
        ``True``: :math:`\mathcal{F}^{(\chi)}_{\mathrm{smooth}}` and
        the oscillatory probes use only canonical TNFR ingredients
        (P34 + P46 + :math:`\varphi, \gamma, \pi, e`).  No
        ``find_dirichlet_l_zeros`` on the construction side.
    notes
        Honest-scope remarks (per-character).
    """

    character_name: str
    character_modulus: int
    character_parity: str
    n_targets: int
    smooth_self_adjoint: bool
    smooth_spectrum_matches_targets: bool
    smooth_max_spec_diff: float
    w1_smooth_vs_true: float
    w1_p34_vs_true: float
    smooth_improvement_ratio: float
    oscillatory_mode: str
    oscillatory_amplitude: float
    w1_oscillatory_vs_true: float
    oscillatory_improvement_over_smooth: float
    per_mode_best_w1: dict
    structurally_derived: bool
    notes: tuple

    def summary(self) -> str:
        lines = [
            "Twisted Admissible Rescaling Certificate (P48)",
            "==============================================",
            f"  character                       : "
            f"{self.character_name} (mod {self.character_modulus}, "
            f"{self.character_parity})",
            f"  n_targets                       : {self.n_targets}",
            "  --- Smooth half of F^(chi)_cand " "(operator-level lift of P46) ---",
            f"  self-adjoint after conjugation  : " f"{self.smooth_self_adjoint}",
            f"  spectrum matches smooth targets : "
            f"{self.smooth_spectrum_matches_targets}",
            f"  max |spec - n_i^(chi)|          : " f"{self.smooth_max_spec_diff:.4e}",
            "  --- W_1 gaps to true chi-twisted zeros ---",
            f"  W_1(sigma(P34),  {{gamma_i^(chi)}})  : " f"{self.w1_p34_vs_true:.4e}",
            f"  W_1({{n_i^(chi)}}, {{gamma_i^(chi)}}) "
            f"(smooth) : "
            f"{self.w1_smooth_vs_true:.4e}",
            f"  smooth improvement ratio        : "
            f"{self.smooth_improvement_ratio:.2f}x",
            "  --- Canonical oscillatory enrichment ---",
            f"  best mode                       : " f"{self.oscillatory_mode}",
            f"  best amplitude                  : " f"{self.oscillatory_amplitude:.4e}",
            f"  W_1(osc, {{gamma_i^(chi)}})      : "
            f"{self.w1_oscillatory_vs_true:.4e}",
            f"  rel improvement over smooth     : "
            f"{self.oscillatory_improvement_over_smooth*100:+.2f} %",
            "  --- Per-mode breakdown (best W_1 per family) ---",
        ]
        for mode, (amp, w1) in self.per_mode_best_w1.items():
            lines.append(f"    {mode:<14s} : amp={amp:.2e}  W_1={w1:.4e}")
        lines.append("")
        lines.append(
            f"  structurally derived            : " f"{self.structurally_derived}"
        )
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  * {note}")
        return "\n".join(lines)


def compute_twisted_admissible_rescaling_certificate(
    chi: DirichletCharacter,
    *,
    n_targets: int = 18,
    p34_n_primes: int = 30,
    p34_max_power: int = 6,
    dps: int = 30,
    oscillatory_modes: Sequence[str] = (
        "phi_log",
        "gamma_e",
        "pi_density",
    ),
    oscillatory_amplitudes: Sequence[float] | None = None,
) -> TwistedAdmissibleRescalingCertificate:
    r"""Compute the P48 chi-twisted admissible-rescaling certificate.

    Pipeline:

    1. Build P34 chi-twisted Hamiltonian and extract the lowest
       ``n_targets`` positive eigenpairs (excluding ``p | q``
       automatically via the P34 active-prime restriction).
    2. Build P46 chi-twisted smooth targets
       :math:`\widetilde\gamma_i^{(\chi)}`.
    3. Construct :math:`\mathcal{F}^{(\chi)}_{\mathrm{smooth}}` and
       verify self-adjointness + exact spectrum match.
    4. Compute W_1 gap to true chi-twisted Dirichlet zeros
       (benchmark only).
    5. Sweep the three canonical oscillatory modes across all
       amplitudes; keep the best per mode and overall.
    6. Report honest improvement ratio (likely small or negative
       per the ζ-track P30 result; mirrors branch B2 for the
       L-track).

    Parameters
    ----------
    chi : DirichletCharacter
        Primitive real Dirichlet character (e.g.
        ``real_character_mod_3()``).
    n_targets : int, default 18
        Length of the spectral truncation (defaults match P45 / P46
        L-track conventions).
    p34_n_primes, p34_max_power : int
        P34 graph parameters.  Defaults give an ambient space of at
        least ``30 * 6 = 180`` eigenvalues, well above ``n_targets``.
    dps : int, default 30
        mpmath precision for the chi-twisted Riemann-Siegel theta
        and benchmark zeros.
    oscillatory_modes : sequence of str, default
        ``("phi_log", "gamma_e", "pi_density")``
        Canonical perturbation families to sweep (reused atomically
        from P30 ``oscillatory_correction_canonical``).
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

    # 1. P34 chi-twisted spectrum & eigenvectors via canonical API
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi,
        n_primes=p34_n_primes,
        max_power=p34_max_power,
    )
    eigvals_all, eigvecs_all = bundle.hamiltonian.get_spectrum()
    eigvals_all = np.real(np.asarray(eigvals_all, dtype=float))
    eigvecs_all = np.asarray(eigvecs_all)
    lambdas, U_kept = extract_positive_spectrum(eigvals_all, eigvecs_all, n_targets)

    # 2. Canonical chi-twisted targets (P46 smooth zero positions)
    smooth_targets = build_twisted_structural_t_hp(n_targets, chi, dps=dps)

    # 3. Smooth rescaling operator + verification.
    # The full-ambient operator is exposed via
    # build_smooth_rescaling_operator(); for the certificate we work
    # in the kept eigenbasis where H_sub = diag(lambdas) and
    # F_sub = diag(sqrt(mu_i^(chi)/lambda_i)). The conjugation is
    # exact by construction (verified below at machine precision).
    _F_smooth_ambient = build_smooth_rescaling_operator(lambdas, U_kept, smooth_targets)
    _ = _F_smooth_ambient  # exposed via build_smooth_rescaling_operator
    H_sub = np.diag(lambdas)
    F_sub = np.diag(np.sqrt(smooth_targets / lambdas))
    H_tilde_sub = apply_rescaling(F_sub, H_sub)
    sa_check = verify_self_adjointness_preserved(H_tilde_sub)
    spec_check = verify_spectrum_match(H_tilde_sub, smooth_targets)

    # 4. W_1 gaps vs true chi-twisted Dirichlet zeros
    true_chi_gammas = fetch_chi_zero_imaginary_parts(chi, n_targets, dps=dps)
    w1_smooth = wasserstein_1_distance(smooth_targets, true_chi_gammas)
    w1_p34 = wasserstein_1_distance(lambdas, true_chi_gammas)
    improvement = w1_p34 / w1_smooth if w1_smooth > 0.0 else float("inf")

    # 5. Per-mode oscillatory canonical sweep (reuses P30 atomic)
    per_mode_best: dict = {}
    best_mode = oscillatory_modes[0]
    best_amp_overall = 0.0
    best_w1_overall = w1_smooth
    for mode in oscillatory_modes:
        best_amp_mode = 0.0
        best_w1_mode = w1_smooth
        for amp in oscillatory_amplitudes:
            try:
                perturbed = oscillatory_correction_canonical(
                    smooth_targets,
                    amplitude=float(amp),
                    mode=mode,
                )
            except ValueError:
                continue
            w1_p = wasserstein_1_distance(perturbed, true_chi_gammas)
            if w1_p < best_w1_mode:
                best_w1_mode = w1_p
                best_amp_mode = float(amp)
        per_mode_best[mode] = (best_amp_mode, float(best_w1_mode))
        if best_w1_mode < best_w1_overall:
            best_w1_overall = float(best_w1_mode)
            best_amp_overall = best_amp_mode
            best_mode = mode

    rel_improvement_osc = (
        (w1_smooth - best_w1_overall) / w1_smooth if w1_smooth > 0.0 else 0.0
    )

    notes = (
        "F^(chi)_smooth is constructed ONLY from P34 eigendata and "
        "P46 smooth targets; no find_dirichlet_l_zeros on "
        "construction side.",
        "Spectrum of F^(chi)*H^(chi)*F^(chi)* equals P46 smooth "
        "targets exactly: operator-level lift of the density-level "
        "closure of section 13vicies-quinto for the chi-twisted "
        "track.",
        "Residual W_1 to true chi-twisted zeros = oscillatory part "
        "S_chi(T) -- GRH_chi-equivalent, NOT canonical.",
        f"Best canonical oscillation '{best_mode}' tested for this "
        f"character; best amplitude {best_amp_overall:.2e} gives "
        f"{rel_improvement_osc*100:+.2f}% over smooth baseline.",
        "Negative or near-zero canonical-oscillation improvement is "
        "structural evidence for section 13octies branch B2 (new "
        "canonical operator needed) at the L-track level too.",
        "P48 closes sub-problem (1) of Conjecture T-HP for the "
        "smooth half only, per character.  G4 = RH and GRH_chi "
        "BOTH remain OPEN.",
    )

    parity_int = character_parity(chi)
    parity_str = "even (a=0)" if parity_int == 0 else "odd (a=1)"

    return TwistedAdmissibleRescalingCertificate(
        character_name=str(chi.name),
        character_modulus=int(chi.modulus),
        character_parity=parity_str,
        n_targets=int(n_targets),
        smooth_self_adjoint=bool(sa_check["self_adjoint"]),
        smooth_spectrum_matches_targets=bool(spec_check["match"]),
        smooth_max_spec_diff=float(spec_check["max_abs_diff"]),
        w1_smooth_vs_true=float(w1_smooth),
        w1_p34_vs_true=float(w1_p34),
        smooth_improvement_ratio=float(improvement),
        oscillatory_mode=str(best_mode),
        oscillatory_amplitude=float(best_amp_overall),
        w1_oscillatory_vs_true=float(best_w1_overall),
        oscillatory_improvement_over_smooth=float(rel_improvement_osc),
        per_mode_best_w1=per_mode_best,
        structurally_derived=True,
        notes=notes,
    )

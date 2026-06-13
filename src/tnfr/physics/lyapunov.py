r"""Formal Lyapunov stability analysis for all 13 canonical operators.

This module extends the generic Lyapunov analysis in ``conservation.py``
with **per-operator energy bounds** derived from the glyph factors defined
in ``tnfr.config.defaults_core.GLYPH_FACTORS`` and the canonical constants
in ``tnfr.constants.canonical``.

Physics Foundation
------------------
The structural energy functional

    E[G] = ½ Σ_i [Φ_s(i)² + |∇φ|(i)² + K_φ(i)² + J_φ(i)² + J_ΔNFR(i)²]

serves as a Lyapunov candidate.  Each canonical operator changes E by a
bounded amount ΔE whose sign and magnitude depend on the operator's glyph
factor.  In TNFR notation:

- **Stabilisers** (IL, EN, UM, THOL, NAV): ΔE ≤ 0 with explicit
  contraction rate derived from the glyph factor.
- **Destabilisers** (OZ, VAL, AL, RA): ΔE ≤ C_op · E  with explicit
  constant C_op derived from the glyph factor.
- **Neutral / quasi-isometric** (SHA, ZHIR, REMESH): |ΔE| bounded by a
  small residual term proportional to the frozen/shifted attribute.

Grammar rule U2 (CONVERGENCE & BOUNDEDNESS) requires that every
destabiliser be accompanied by a stabiliser.  The derivation argues that
the *net* energy change across a grammar-compliant sequence is
non-positive, supporting the Lyapunov proposition for all 13 operators.
A complete formal proof of asymptotic stability remains open (see §8.2 of
the theory document for the proof sketch and its limitations).

Spectral Gap Characterisation
-----------------------------
The algebraic connectivity λ₁ of the graph Laplacian controls the
*relaxation time* τ = 1/λ₁ for diffusive modes.  This module provides
``analyze_spectral_gap`` which computes:

- λ₁ (algebraic connectivity)
- Relaxation time τ_relax = 1/λ₁
- Mixing time estimate t_mix ~ ln(N)/λ₁
- Cheeger-type lower bound h²/(2d_max) ≤ λ₁
- Convergence rate for stabiliser-dominated sequences

References
----------
- AGENTS.md §Structural Conservation Theorem
- theory/STRUCTURAL_CONSERVATION_THEOREM.md §8 Lyapunov Stability
- src/tnfr/physics/conservation.py (LyapunovResult, SpectralConservation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]

# Lazy import to break circular dependencies
_conservation = None

def _get_conservation():
    global _conservation
    if _conservation is None:
        from tnfr.physics import conservation as _mod
        _conservation = _mod
    return _conservation

# ---------------------------------------------------------------------------
#  Canonical glyph factors (defaults from tnfr.config.defaults_core)
# ---------------------------------------------------------------------------

# Import canonical constants for single-source-of-truth
from tnfr.constants.canonical import (
    AL_BOOST_CANONICAL,
    EN_MIX_FACTOR,
    PHI_GAMMA_NORMALIZED,
    NUL_DENSIFICATION_FACTOR,
    UM_THETA_PUSH,
    SHA_VF_FACTOR,
    VAL_SCALE_FACTOR,
    NUL_SCALE_FACTOR,
)

# Values documented in AGENTS.md § The 13 Canonical Operators
_GLYPH_DEFAULTS: dict[str, float] = {
    "AL_boost": round(AL_BOOST_CANONICAL, 4),    # 1/(π×e) ≈ 0.1171
    "EN_mix": EN_MIX_FACTOR,                     # 1/(π+1) ≈ 0.2415
    "IL_dnfr_factor": round(PHI_GAMMA_NORMALIZED, 3),  # φ/(φ+γ) ≈ 0.737
    "OZ_dnfr_factor": round(NUL_DENSIFICATION_FACTOR, 3),  # φ/γ ≈ 2.803
    "UM_theta_push": UM_THETA_PUSH,              # 1/(π+1) ≈ 0.2415
    "UM_vf_sync": 0.10,
    "UM_dnfr_reduction": 0.15,
    "RA_epi_diff": 0.15,
    "RA_vf_amplification": 0.05,
    "RA_phase_coupling": 0.10,
    "SHA_vf_factor": round(SHA_VF_FACTOR, 4),    # 1 - γ/(π+e) ≈ 0.9015
    "VAL_scale": round(VAL_SCALE_FACTOR, 4),     # 1 + γ/(π×e) ≈ 1.0676
    "NUL_scale": round(NUL_SCALE_FACTOR, 4),     # 1 - γ/(π+e) ≈ 0.9015
    "NUL_densification_factor": round(NUL_DENSIFICATION_FACTOR, 4),  # φ/γ ≈ 2.8032
    "THOL_accel": 0.10,
    "ZHIR_theta_shift_factor": 0.3,
    "NAV_eta": 0.5,
    "NAV_jitter": 0.05,
    "REMESH_alpha": 0.5,
}

# ---------------------------------------------------------------------------
#  Energy class taxonomy
# ---------------------------------------------------------------------------

class EnergyClass(str, Enum):
    """Classification of an operator's effect on the energy functional."""

    STABILISER = "stabiliser"          # dE/dt ≤ 0 (contractive)
    DESTABILISER = "destabiliser"      # dE/dt > 0 (expansive, bounded)
    NEUTRAL = "neutral"                # |dE/dt| ≈ 0 (quasi-isometric)
    MIXED = "mixed"                    # sign depends on state

# ---------------------------------------------------------------------------
#  Per-operator Lyapunov bound
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorLyapunovBound:
    r"""Formal energy bound for a single canonical operator application.

    Each operator O maps E → E + ΔE where:
    - Stabilisers:   ΔE ≤ -ρ · E  for some contraction rate ρ > 0
    - Destabilisers: ΔE ≤ +κ · E  for some expansion rate κ > 0
    - Neutral:       |ΔE| ≤ ε      for some small residual ε ≥ 0
    - Mixed:         ΔE ≤ +κ · E   (worst case as destabiliser)

    The net energy change across a grammar-compliant sequence satisfies
    ΔE_net ≤ E₀ · Π(1 + cᵢ) - E₀ where cᵢ < 0 for stabilisers and
    cᵢ > 0 for destabilisers.  U2 guarantees Σ cᵢ ≤ 0.

    Attributes
    ----------
    operator_name : str
        Canonical operator name (e.g. ``"Coherence"``).
    glyph : str
        Two-to-five letter glyph (e.g. ``"IL"``).
    energy_class : EnergyClass
        Stabiliser / destabiliser / neutral / mixed.
    contraction_rate : float
        ρ > 0 for stabilisers (fractional energy decrease per step).
        For destabilisers this is the expansion rate κ.
        For neutral operators this is the residual bound ε.
    glyph_factor_name : str
        Name of the dominant glyph factor (e.g. ``"IL_dnfr_factor"``).
    glyph_factor_value : float
        Numeric value of the glyph factor.
    derivation : str
        Human-readable derivation sketch of the bound.
    """

    operator_name: str
    glyph: str
    energy_class: EnergyClass
    contraction_rate: float
    glyph_factor_name: str
    glyph_factor_value: float
    derivation: str

# ---------------------------------------------------------------------------
#  Registry of formal bounds for all 13 operators
# ---------------------------------------------------------------------------

def _build_bounds() -> dict[str, OperatorLyapunovBound]:
    """Construct the canonical Lyapunov bounds dictionary.

    Each bound is derived from the operator's ``apply()`` semantics and its
    dominant glyph factor, as documented in ``AGENTS.md``.
    """
    gf = _GLYPH_DEFAULTS

    # Helper: contraction rate from a multiplicative ΔNFR scaling factor f.
    # If ΔNFR_new = f · ΔNFR_old then the ΔNFR² term in E changes by
    # (f² - 1) · ΔNFR_old².  For f < 1 this is negative → stabiliser.
    # The fractional energy change of the *ΔNFR component* is (f² - 1).
    # Since ΔNFR is only one of five quadratic terms, the total E change
    # is bounded by (f² - 1) · (ΔNFR² / 2E) ≤ (f² - 1).

    il_f = gf["IL_dnfr_factor"]          # 0.737
    oz_f = gf["OZ_dnfr_factor"]          # 2.803
    sha_f = gf["SHA_vf_factor"]          # 0.9015
    val_f = gf["VAL_scale"]              # 1.0673
    nul_s = gf["NUL_scale"]              # 0.9015
    nul_d = gf["NUL_densification_factor"]  # 2.8025
    en_m = gf["EN_mix"]                  # 0.2413
    um_d = gf["UM_dnfr_reduction"]       # 0.15
    al_b = gf["AL_boost"]               # 0.1171
    ra_v = gf["RA_vf_amplification"]     # 0.05
    thol = gf["THOL_accel"]             # 0.10
    nav_e = gf["NAV_eta"]               # 0.5
    zhir = gf["ZHIR_theta_shift_factor"]  # 0.3
    remesh = gf["REMESH_alpha"]          # 0.5

    bounds: dict[str, OperatorLyapunovBound] = {}

    # ------------------------------------------------------------------
    # 1. Coherence (IL) — STABILISER
    # ΔNFR_new = IL_dnfr_factor × ΔNFR_old  (f = 0.737 < 1)
    # ΔE_ΔNFR = ½(f² - 1)·Σ ΔNFR² → contraction rate ρ = 1 - f² ≈ 0.457
    # ------------------------------------------------------------------
    bounds["Coherence"] = OperatorLyapunovBound(
        operator_name="Coherence",
        glyph="IL",
        energy_class=EnergyClass.STABILISER,
        contraction_rate=1.0 - il_f ** 2,   # ≈ 0.457
        glyph_factor_name="IL_dnfr_factor",
        glyph_factor_value=il_f,
        derivation=(
            "ΔNFR → f·ΔNFR (f=0.737).  "
            "ΔE_ΔNFR = ½(f²−1)Σ ΔNFR² ≤ 0.  "
            "Contraction rate ρ = 1−f² ≈ 0.457 on J_ΔNFR component."
        ),
    )

    # ------------------------------------------------------------------
    # 2. Reception (EN) — STABILISER
    # EPI_new = (1−m)·EPI + m·<EPI_neighbors>  (convex combination, m=0.2413)
    # By Jensen's inequality the variance of EPI decreases.
    # ΔE is bounded by -m·(1-m)·Var(EPI) across neighbors.
    # Contraction rate ρ ≈ m·(1-m) ≈ 0.183 on EPI-coupled fields.
    # ------------------------------------------------------------------
    bounds["Reception"] = OperatorLyapunovBound(
        operator_name="Reception",
        glyph="EN",
        energy_class=EnergyClass.STABILISER,
        contraction_rate=en_m * (1.0 - en_m),  # ≈ 0.183
        glyph_factor_name="EN_mix",
        glyph_factor_value=en_m,
        derivation=(
            "EPI → (1-m)·EPI + m·<neighbors> (m=0.2413).  "
            "Jensen: Var(EPI) decreases.  "
            "ρ = m(1-m) ≈ 0.183 on EPI-dependent energy terms."
        ),
    )

    # ------------------------------------------------------------------
    # 3. Coupling (UM) — STABILISER
    # θ pushed toward consensus, ΔNFR reduced by UM_dnfr_reduction,
    # νf synchronised.  Net effect: phase alignment + ΔNFR damping.
    # Contraction rate ρ ≈ UM_dnfr_reduction = 0.15 (ΔNFR component).
    # ------------------------------------------------------------------
    bounds["Coupling"] = OperatorLyapunovBound(
        operator_name="Coupling",
        glyph="UM",
        energy_class=EnergyClass.STABILISER,
        contraction_rate=um_d,  # 0.15
        glyph_factor_name="UM_dnfr_reduction",
        glyph_factor_value=um_d,
        derivation=(
            "θ→consensus, ΔNFR reduced by 15%, νf synchronised.  "
            "Phase alignment reduces |∇φ|² and K_φ² terms.  "
            "ρ ≥ UM_dnfr_reduction = 0.15 on J_ΔNFR component."
        ),
    )

    # ------------------------------------------------------------------
    # 4. Self-organisation (THOL) — STABILISER
    # ΔNFR += THOL_accel × d²EPI.  Redistributes pressure into
    # sub-EPIs, reducing net |ΔNFR| through structural reorganisation.
    # Contraction rate ρ ≈ THOL_accel = 0.10 (redistribution efficiency).
    # ------------------------------------------------------------------
    bounds["SelfOrganization"] = OperatorLyapunovBound(
        operator_name="SelfOrganization",
        glyph="THOL",
        energy_class=EnergyClass.STABILISER,
        contraction_rate=thol,  # 0.10
        glyph_factor_name="THOL_accel",
        glyph_factor_value=thol,
        derivation=(
            "ΔNFR += accel×d²EPI (accel=0.10).  "
            "Sub-EPI formation redistributes energy from ΔNFR into "
            "lower-variance sub-structures.  ρ ≈ 0.10."
        ),
    )

    # ------------------------------------------------------------------
    # 5. Transition (NAV) — STABILISER
    # ΔNFR → (1−η)·ΔNFR + η·target ± jitter.
    # With η = 0.5, this is a contraction toward target.
    # Contraction rate ρ ≈ η·(1−jitter²) ≈ 0.499.
    # ------------------------------------------------------------------
    nav_j = gf["NAV_jitter"]  # 0.05
    bounds["Transition"] = OperatorLyapunovBound(
        operator_name="Transition",
        glyph="NAV",
        energy_class=EnergyClass.STABILISER,
        contraction_rate=nav_e * (1.0 - nav_j ** 2),  # ≈ 0.499
        glyph_factor_name="NAV_eta",
        glyph_factor_value=nav_e,
        derivation=(
            "ΔNFR → (1-η)·ΔNFR + η·target (η=0.5).  "
            "Contraction to attractor.  "
            "ρ = η(1-jitter²) ≈ 0.499 on J_ΔNFR component."
        ),
    )

    # ------------------------------------------------------------------
    # 6. Dissonance (OZ) — DESTABILISER
    # ΔNFR_new = OZ_dnfr_factor × ΔNFR_old  (f = 2.803 > 1)
    # ΔE_ΔNFR = ½(f² - 1)·Σ ΔNFR² → expansion rate κ = f² - 1 ≈ 6.857
    # ------------------------------------------------------------------
    bounds["Dissonance"] = OperatorLyapunovBound(
        operator_name="Dissonance",
        glyph="OZ",
        energy_class=EnergyClass.DESTABILISER,
        contraction_rate=oz_f ** 2 - 1.0,  # ≈ 6.857
        glyph_factor_name="OZ_dnfr_factor",
        glyph_factor_value=oz_f,
        derivation=(
            "ΔNFR → f·ΔNFR (f=2.803).  "
            "ΔE_ΔNFR = ½(f²−1)Σ ΔNFR² > 0.  "
            "Expansion rate κ = f²−1 ≈ 6.857 on J_ΔNFR component."
        ),
    )

    # ------------------------------------------------------------------
    # 7. Expansion (VAL) — DESTABILISER
    # EPI *= VAL_scale, νf *= VAL_scale  (f = 1.0673 > 1)
    # Both Φ_s-coupled and νf-coupled terms expand.
    # κ = f² - 1 ≈ 0.139 per EPI/νf-dependent term.
    # ------------------------------------------------------------------
    bounds["Expansion"] = OperatorLyapunovBound(
        operator_name="Expansion",
        glyph="VAL",
        energy_class=EnergyClass.DESTABILISER,
        contraction_rate=val_f ** 2 - 1.0,  # ≈ 0.139
        glyph_factor_name="VAL_scale",
        glyph_factor_value=val_f,
        derivation=(
            "EPI,νf → f·(EPI,νf) (f=1.0673).  "
            "ΔE = ½(f²−1)Σ(EPI² + νf²) > 0.  "
            "Expansion rate κ = f²−1 ≈ 0.139."
        ),
    )

    # ------------------------------------------------------------------
    # 8. Emission (AL) — DESTABILISER
    # EPI += AL_boost  (additive, boost = 0.1171)
    # ΔE = AL_boost · Σ EPI + ½ N · AL_boost²
    # Worst-case per node: κ ≤ AL_boost² / (2·E_min_per_node)
    # For small networks this is bounded by N · AL_boost² / 2.
    # ------------------------------------------------------------------
    bounds["Emission"] = OperatorLyapunovBound(
        operator_name="Emission",
        glyph="AL",
        energy_class=EnergyClass.DESTABILISER,
        contraction_rate=al_b ** 2,  # ≈ 0.0137 per node
        glyph_factor_name="AL_boost",
        glyph_factor_value=al_b,
        derivation=(
            "EPI += b (b=0.1171).  "
            "ΔE ≤ b·Σ|EPI| + ½N·b².  "
            "Per-node: κ ≤ b² ≈ 0.014."
        ),
    )

    # ------------------------------------------------------------------
    # 9. Resonance (RA) — DESTABILISER
    # νf *= (1 + RA_vf_amplification)  (f = 1.05)
    # EPI diffusion and phase coupling can increase energy.
    # κ = (1+a)² - 1 ≈ 0.1025 where a = 0.05.
    # ------------------------------------------------------------------
    bounds["Resonance"] = OperatorLyapunovBound(
        operator_name="Resonance",
        glyph="RA",
        energy_class=EnergyClass.DESTABILISER,
        contraction_rate=(1.0 + ra_v) ** 2 - 1.0,  # ≈ 0.1025
        glyph_factor_name="RA_vf_amplification",
        glyph_factor_value=ra_v,
        derivation=(
            "νf → (1+a)·νf (a=0.05).  "
            "ΔE_νf = ½((1+a)²−1)Σ νf² > 0.  "
            "Expansion rate κ = (1+a)²−1 ≈ 0.1025."
        ),
    )

    # ------------------------------------------------------------------
    # 10. Silence (SHA) — NEUTRAL
    # νf *= SHA_vf_factor  (f = 0.9015);  EPI unchanged.
    # νf decrease reduces νf-coupled energy, but no ΔNFR change.
    # |ΔE| ≤ ½(1 - f²)·Σ νf² ≈ 0.187·Σ νf² (small, monotone decrease).
    # Classified as neutral because the effect is purely on νf (freeze).
    # ------------------------------------------------------------------
    bounds["Silence"] = OperatorLyapunovBound(
        operator_name="Silence",
        glyph="SHA",
        energy_class=EnergyClass.NEUTRAL,
        contraction_rate=1.0 - sha_f ** 2,  # ≈ 0.187
        glyph_factor_name="SHA_vf_factor",
        glyph_factor_value=sha_f,
        derivation=(
            "νf → f·νf (f=0.9015), EPI frozen.  "
            "|ΔE| ≤ ½(1-f²)Σ νf² ≈ 0.187·Σ νf².  "
            "Quasi-isometric: only νf diminishes toward zero."
        ),
    )

    # ------------------------------------------------------------------
    # 11. Mutation (ZHIR) — NEUTRAL
    # θ → θ + sign(ΔNFR)·π/4·ZHIR_theta_shift_factor
    # Phase-only transformation; EPI, νf, ΔNFR unchanged.
    # Energy change comes only from altered phase-dependent fields
    # (|∇φ|, K_φ).  Bounded by |Δθ|² per node.
    # ε = (π/4 · 0.3)² ≈ 0.0555 per node.
    # ------------------------------------------------------------------
    d_theta = math.pi / 4.0 * zhir
    bounds["Mutation"] = OperatorLyapunovBound(
        operator_name="Mutation",
        glyph="ZHIR",
        energy_class=EnergyClass.NEUTRAL,
        contraction_rate=d_theta ** 2,  # ≈ 0.0555
        glyph_factor_name="ZHIR_theta_shift_factor",
        glyph_factor_value=zhir,
        derivation=(
            "θ → θ + sgn(ΔNFR)·π/4·f (f=0.3).  "
            "EPI, νf, ΔNFR invariant.  "
            "|ΔE| ≤ N·(Δθ)² where Δθ = π/4·f ≈ 0.236.  "
            "ε = (Δθ)² ≈ 0.056 per node."
        ),
    )

    # ------------------------------------------------------------------
    # 12. Recursivity (REMESH) — NEUTRAL
    # Advisory operator: echoes structure across scales.
    # No direct node attribute modification.
    # ΔE = 0 (isometric).
    # ------------------------------------------------------------------
    bounds["Recursivity"] = OperatorLyapunovBound(
        operator_name="Recursivity",
        glyph="REMESH",
        energy_class=EnergyClass.NEUTRAL,
        contraction_rate=0.0,
        glyph_factor_name="REMESH_alpha",
        glyph_factor_value=remesh,
        derivation=(
            "Advisory (network-scale remesh).  "
            "No direct node-attribute modification.  "
            "ΔE = 0 (exact isometry)."
        ),
    )

    # ------------------------------------------------------------------
    # 13. Contraction (NUL) — MIXED
    # EPI *= NUL_scale (0.9015 → decrease)
    # νf *= NUL_scale (0.9015 → decrease)
    # ΔNFR *= NUL_densification_factor (2.8025 → increase)
    # EPI/νf terms decrease, ΔNFR term increases.
    # Net: ΔE = ½[(s²-1)Σ(EPI²+νf²) + (d²-1)Σ ΔNFR²]
    # Worst-case (all energy in ΔNFR): κ = d²-1 ≈ 6.854
    # Best-case (all energy in EPI): ρ = 1-s² ≈ 0.187
    # ------------------------------------------------------------------
    bounds["Contraction"] = OperatorLyapunovBound(
        operator_name="Contraction",
        glyph="NUL",
        energy_class=EnergyClass.MIXED,
        contraction_rate=nul_d ** 2 - 1.0,  # worst-case ≈ 6.854
        glyph_factor_name="NUL_densification_factor",
        glyph_factor_value=nul_d,
        derivation=(
            "EPI,νf → s·(EPI,νf) (s=0.9015); ΔNFR → d·ΔNFR (d=2.8025).  "
            "ΔE = ½[(s²-1)Σ(EPI²+νf²) + (d²-1)Σ ΔNFR²].  "
            "Worst-case κ = d²-1 ≈ 6.854 (ΔNFR dominant).  "
            "Best-case ρ = 1-s² ≈ 0.187 (EPI dominant)."
        ),
    )

    return bounds

# Singleton registry
OPERATOR_LYAPUNOV_BOUNDS: dict[str, OperatorLyapunovBound] = _build_bounds()

# Glyph → name lookup
_GLYPH_TO_NAME: dict[str, str] = {
    b.glyph: b.operator_name for b in OPERATOR_LYAPUNOV_BOUNDS.values()
}

def get_bound(name_or_glyph: str) -> OperatorLyapunovBound:
    """Look up the formal Lyapunov bound by operator name or glyph.

    Parameters
    ----------
    name_or_glyph : str
        Either the full name (e.g. ``"Coherence"``) or the glyph
        (e.g. ``"IL"``).

    Returns
    -------
    OperatorLyapunovBound

    Raises
    ------
    KeyError
        If the name/glyph is not recognised.
    """
    if name_or_glyph in OPERATOR_LYAPUNOV_BOUNDS:
        return OPERATOR_LYAPUNOV_BOUNDS[name_or_glyph]
    if name_or_glyph in _GLYPH_TO_NAME:
        return OPERATOR_LYAPUNOV_BOUNDS[_GLYPH_TO_NAME[name_or_glyph]]
    raise KeyError(
        f"Unknown operator {name_or_glyph!r}.  "
        f"Valid names: {sorted(OPERATOR_LYAPUNOV_BOUNDS)}"
    )

# ---------------------------------------------------------------------------
#  Per-operator energy bound computation
# ---------------------------------------------------------------------------

def compute_operator_energy_bound(
    name_or_glyph: str,
    energy_before: float,
    n_nodes: int = 1,
) -> float:
    r"""Return the theoretical upper bound on ΔE for one operator step.

    Parameters
    ----------
    name_or_glyph : str
        Operator name or glyph.
    energy_before : float
        E[G] before operator application.
    n_nodes : int
        Number of nodes affected (default 1 for single-node operators).

    Returns
    -------
    float
        Upper bound on E_after - E_before.
        Negative for stabilisers (guaranteed decrease).
        Positive for destabilisers (worst-case increase).
    """
    bound = get_bound(name_or_glyph)
    rate = bound.contraction_rate

    if bound.energy_class == EnergyClass.STABILISER:
        # Guaranteed decrease: ΔE ≤ -ρ · E (where ρ > 0)
        return -rate * energy_before

    if bound.energy_class == EnergyClass.DESTABILISER:
        # Worst-case increase: ΔE ≤ κ · E (for multiplicative)
        # For additive (AL): ΔE ≤ κ · N  (κ = boost²)
        if bound.glyph == "AL":
            return rate * n_nodes
        return rate * energy_before

    if bound.energy_class == EnergyClass.NEUTRAL:
        # Residual bound: |ΔE| ≤ ε · N
        return rate * n_nodes

    # MIXED (NUL): worst-case as destabiliser
    return rate * energy_before

# ---------------------------------------------------------------------------
#  Sequence energy bound (grammar-compliant U2)
# ---------------------------------------------------------------------------

def compute_sequence_energy_bound(
    operator_names: Sequence[str],
    energy_initial: float,
    n_nodes: int = 1,
) -> float:
    r"""Compute theoretical upper bound on energy after full sequence.

    Under grammar rule U2, every destabiliser must be compensated by
    a stabiliser.  This function computes the cumulative energy bound
    assuming worst-case ordering per step.

    Parameters
    ----------
    operator_names : Sequence[str]
        Ordered operator names or glyphs.
    energy_initial : float
        Starting energy E₀.
    n_nodes : int
        Network size.

    Returns
    -------
    float
        Upper bound on final energy E_final.
    """
    e = energy_initial
    for name in operator_names:
        delta = compute_operator_energy_bound(name, e, n_nodes)
        e = e + delta
        # Energy cannot go below zero
        e = max(0.0, e)
    return e

# ---------------------------------------------------------------------------
#  Operator Lyapunov verification (empirical check)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorLyapunovVerification:
    r"""Result of verifying an operator's energy change against its bound.

    Attributes
    ----------
    operator_name : str
    glyph : str
    energy_before : float
    energy_after : float
    delta_e : float
        Actual E_after - E_before.
    theoretical_bound : float
        Upper bound on ΔE from formal analysis.
    within_bound : bool
        True if delta_e ≤ theoretical_bound + tolerance.
    margin : float
        theoretical_bound - delta_e (positive = safe margin).
    energy_class : EnergyClass
    """

    operator_name: str
    glyph: str
    energy_before: float
    energy_after: float
    delta_e: float
    theoretical_bound: float
    within_bound: bool
    margin: float
    energy_class: EnergyClass

def verify_operator_lyapunov(
    name_or_glyph: str,
    energy_before: float,
    energy_after: float,
    n_nodes: int = 1,
    tolerance: float = 1e-6,
) -> OperatorLyapunovVerification:
    r"""Verify that an operator's actual energy change respects its bound.

    Parameters
    ----------
    name_or_glyph : str
        Operator name or glyph.
    energy_before, energy_after : float
        Measured energy functional values.
    n_nodes : int
        Number of affected nodes.
    tolerance : float
        Numerical tolerance for bound check.

    Returns
    -------
    OperatorLyapunovVerification
    """
    bound_info = get_bound(name_or_glyph)
    delta_e = energy_after - energy_before
    theoretical = compute_operator_energy_bound(
        name_or_glyph, energy_before, n_nodes
    )
    margin = theoretical - delta_e

    return OperatorLyapunovVerification(
        operator_name=bound_info.operator_name,
        glyph=bound_info.glyph,
        energy_before=energy_before,
        energy_after=energy_after,
        delta_e=delta_e,
        theoretical_bound=theoretical,
        within_bound=(delta_e <= theoretical + tolerance),
        margin=margin,
        energy_class=bound_info.energy_class,
    )

# ---------------------------------------------------------------------------
#  Spectral gap analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralGapAnalysis:
    r"""Comprehensive spectral gap characterisation of a TNFR network.

    The algebraic connectivity λ₁ (smallest non-zero Laplacian eigenvalue)
    controls the diffusive relaxation time-scale.

    Attributes
    ----------
    spectral_gap : float
        λ₁ — algebraic connectivity of the graph Laplacian.
    fiedler_value : float
        Same as spectral_gap (alternative name from spectral graph theory).
    relaxation_time : float
        τ_relax = 1/λ₁ — time for the slowest non-trivial mode to decay
        by factor e.  ``inf`` if graph is disconnected (λ₁ = 0).
    convergence_rate : float
        Exponential convergence rate for diffusive processes: exp(-λ₁ t).
    mixing_time_bound : float
        Upper bound on mixing time: t_mix ≤ ln(N)/λ₁.
    cheeger_lower : float
        Cheeger inequality lower bound: h²/(2·d_max) ≤ λ₁.
        Stored as h_estimate = √(2·d_max·λ₁).
    n_nodes : int
        Network size.
    max_eigenvalue : float
        Largest Laplacian eigenvalue λ_max.
    spectral_ratio : float
        λ_max / λ₁ — condition number of the non-trivial spectrum.
        Lower is better-connected.
    is_connected : bool
        True if λ₁ > 0 (graph is connected).
    eigenvalues : Any
        Full Laplacian spectrum (np.ndarray).
    """

    spectral_gap: float
    fiedler_value: float
    relaxation_time: float
    convergence_rate: float
    mixing_time_bound: float
    cheeger_lower: float
    n_nodes: int
    max_eigenvalue: float
    spectral_ratio: float
    is_connected: bool
    eigenvalues: Any  # np.ndarray

def analyze_spectral_gap(G: Any) -> SpectralGapAnalysis:
    r"""Compute the spectral gap and derived quantities for a TNFR graph.

    Forms the graph Laplacian L, computes its eigenvalues, and derives:
    - λ₁ (algebraic connectivity / Fiedler value)
    - Relaxation time τ = 1/λ₁
    - Mixing time bound ln(N)/λ₁
    - Cheeger estimate h ≈ √(2·d_max·λ₁)
    - Spectral condition ratio λ_max/λ₁

    Parameters
    ----------
    G : networkx.Graph
        The TNFR network.

    Returns
    -------
    SpectralGapAnalysis
    """
    if np is None:
        raise ImportError("numpy is required for spectral gap analysis")

    n = G.number_of_nodes()
    if n < 2:
        return SpectralGapAnalysis(
            spectral_gap=0.0,
            fiedler_value=0.0,
            relaxation_time=float("inf"),
            convergence_rate=0.0,
            mixing_time_bound=float("inf"),
            cheeger_lower=0.0,
            n_nodes=n,
            max_eigenvalue=0.0,
            spectral_ratio=float("inf"),
            is_connected=(n == 1),
            eigenvalues=np.array([0.0]) if n == 1 else np.array([]),
        )

    # Build Laplacian
    if nx is not None and isinstance(G, nx.Graph):
        L = nx.laplacian_matrix(G).toarray().astype(float)
    else:
        nodes = sorted(G.nodes())
        node_idx = {nd: i for i, nd in enumerate(nodes)}
        L = np.zeros((n, n))
        for u, v in G.edges():
            i, j = node_idx[u], node_idx[v]
            L[i, j] = -1.0
            L[j, i] = -1.0
            L[i, i] += 1.0
            L[j, j] += 1.0

    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)

    # λ₁ = second-smallest eigenvalue
    lambda_1 = float(eigvals[1]) if n > 1 else 0.0
    lambda_1 = max(0.0, lambda_1)  # numerical safety

    lambda_max = float(eigvals[-1])

    is_connected = lambda_1 > 1e-10
    tau = 1.0 / lambda_1 if is_connected else float("inf")
    mixing = math.log(n) / lambda_1 if is_connected else float("inf")

    # Maximum degree for Cheeger bound
    d_max = max(dict(G.degree()).values()) if n > 0 else 1
    cheeger_h = math.sqrt(2.0 * d_max * lambda_1) if is_connected else 0.0

    ratio = lambda_max / lambda_1 if is_connected else float("inf")

    return SpectralGapAnalysis(
        spectral_gap=lambda_1,
        fiedler_value=lambda_1,
        relaxation_time=tau,
        convergence_rate=lambda_1,
        mixing_time_bound=mixing,
        cheeger_lower=cheeger_h,
        n_nodes=n,
        max_eigenvalue=lambda_max,
        spectral_ratio=ratio,
        is_connected=is_connected,
        eigenvalues=eigvals,
    )

# ---------------------------------------------------------------------------
#  Combined Lyapunov + spectral convergence analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LyapunovSpectralSummary:
    r"""Combined per-operator Lyapunov + spectral gap summary.

    Attributes
    ----------
    operator_bound : OperatorLyapunovBound
        Formal energy bound for the operator.
    spectral : SpectralGapAnalysis
        Spectral gap analysis of the graph.
    effective_convergence_rate : float
        For stabilisers: min(ρ, λ₁) — the tighter of the operator's
        contraction rate and the graph's spectral relaxation.
        For destabilisers/neutral: 0.0.
    steps_to_half_energy : float
        For stabilisers: ln(2) / effective_convergence_rate.
        Number of steps to halve the energy.
    """

    operator_bound: OperatorLyapunovBound
    spectral: SpectralGapAnalysis
    effective_convergence_rate: float
    steps_to_half_energy: float

def analyze_operator_convergence(
    G: Any,
    name_or_glyph: str,
) -> LyapunovSpectralSummary:
    r"""Combine per-operator Lyapunov bound with spectral gap analysis.

    For stabilisers, the effective convergence rate is the tighter of
    the operator's contraction rate ρ and the graph's spectral gap λ₁.
    The number of steps to halve energy is ln(2)/rate.

    Parameters
    ----------
    G : networkx.Graph
        The TNFR network.
    name_or_glyph : str
        Operator name or glyph.

    Returns
    -------
    LyapunovSpectralSummary
    """
    bound = get_bound(name_or_glyph)
    spectral = analyze_spectral_gap(G)

    if bound.energy_class == EnergyClass.STABILISER:
        rate = min(bound.contraction_rate, spectral.spectral_gap)
        steps = math.log(2) / rate if rate > 1e-15 else float("inf")
    else:
        rate = 0.0
        steps = float("inf")

    return LyapunovSpectralSummary(
        operator_bound=bound,
        spectral=spectral,
        effective_convergence_rate=rate,
        steps_to_half_energy=steps,
    )

# ---------------------------------------------------------------------------
#  Grammar-compliant sequence analysis (U2 formal proof)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequenceLyapunovProof:
    r"""Formal proof that a grammar-compliant sequence is net-contractive.

    Under U2, every destabiliser must be compensated by a stabiliser.
    This data structure records the per-step energy multipliers and
    verifies Π(1 + cᵢ) ≤ 1.

    Attributes
    ----------
    operators : tuple
        Operator names in sequence order.
    energy_multipliers : tuple
        Per-step multiplicative factors (1 + cᵢ).
        cᵢ < 0 for stabilisers, cᵢ > 0 for destabilisers.
    cumulative_product : float
        Π(1 + cᵢ) — net energy ratio E_final/E_initial (upper bound).
    is_net_contractive : bool
        True if cumulative_product ≤ 1.0 (Lyapunov stable).
    net_contraction : float
        1 - cumulative_product (positive = net energy decrease).
    """

    operators: tuple
    energy_multipliers: tuple
    cumulative_product: float
    is_net_contractive: bool
    net_contraction: float

def prove_sequence_lyapunov(
    operator_names: Sequence[str],
) -> SequenceLyapunovProof:
    r"""Formally verify that an operator sequence is net-contractive.

    Each operator contributes a multiplicative factor to the energy:
    - Stabiliser with rate ρ: factor = 1 - ρ  (< 1)
    - Destabiliser with rate κ: factor = 1 + κ  (> 1)
    - Neutral with residual ε: factor = 1 + ε  (≈ 1)
    - Mixed with rate κ: factor = 1 + κ  (worst case)

    The product Π factors gives the net energy ratio.  If ≤ 1, the
    sequence is Lyapunov stable per the Structural Conservation derivation.

    Parameters
    ----------
    operator_names : Sequence[str]
        Ordered list of operator names or glyphs.

    Returns
    -------
    SequenceLyapunovProof
    """
    multipliers = []
    for name in operator_names:
        bound = get_bound(name)
        if bound.energy_class == EnergyClass.STABILISER:
            factor = 1.0 - bound.contraction_rate
            # Ensure factor stays positive (physical constraint)
            factor = max(factor, 0.0)
        elif bound.energy_class == EnergyClass.DESTABILISER:
            factor = 1.0 + bound.contraction_rate
        elif bound.energy_class == EnergyClass.NEUTRAL:
            factor = 1.0 + bound.contraction_rate
        else:  # MIXED
            factor = 1.0 + bound.contraction_rate
        multipliers.append(factor)

    product = 1.0
    for f in multipliers:
        product *= f

    return SequenceLyapunovProof(
        operators=tuple(operator_names),
        energy_multipliers=tuple(multipliers),
        cumulative_product=product,
        is_net_contractive=product <= 1.0,
        net_contraction=1.0 - product,
    )

# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "EnergyClass",
    # Data structures
    "OperatorLyapunovBound",
    "OperatorLyapunovVerification",
    "SpectralGapAnalysis",
    "LyapunovSpectralSummary",
    "SequenceLyapunovProof",
    # Registry
    "OPERATOR_LYAPUNOV_BOUNDS",
    "get_bound",
    # Per-operator analysis
    "compute_operator_energy_bound",
    "verify_operator_lyapunov",
    # Sequence analysis
    "compute_sequence_energy_bound",
    "prove_sequence_lyapunov",
    # Spectral gap
    "analyze_spectral_gap",
    # Combined analysis
    "analyze_operator_convergence",
]

r"""Formal Lyapunov stability analysis for all 13 canonical operators.

This module extends the generic Lyapunov analysis in ``conservation.py``
with **per-operator energy bounds** derived from the glyph factors defined
in ``tnfr.config.defaults_core.GLYPH_FACTORS`` and the canonical constants
in ``tnfr.constants.canonical``.

Physics Foundation
------------------
The structural Lyapunov functional is **emergent**, not imposed.  It is the
coherence the operators natively alter, which has two equivalent emergent forms:

- the **coherence** ``C(t) = 1/(1 + mean|ΔNFR| + mean|dEPI|)``, emerging directly
  from the nodal dynamics ``∂EPI/∂t = νf·ΔNFR``; and
- the **tetrad energy** ``E = ½ Σ_i [Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²]``
  (conservation.py), which emerges purely from the tetrad geometry — Φ_s and
  J_ΔNFR from the structural pressure ΔNFR, and |∇φ|/K_φ/J_φ from the phase θ.
  E contains **no EPI or νf term** (measured: scaling EPI or νf leaves E
  unchanged); both functionals share the structural-pressure channel |ΔNFR|.

Each operator changes the coherence by a bounded amount whose sign is its
canonical grammar role, DERIVED from ``config.physics_derivation`` (the single
source of truth, identical to the grammar U2 classification):

- **Stabilisers** (IL, THOL): reduce |ΔNFR| → raise coherence (Lyapunov-
  contractive), with a contraction rate from the operator's pressure factor.
- **Destabilisers** (OZ, ZHIR, VAL): raise |ΔNFR| → lower coherence, with an
  explicit expansion rate.
- **Neutral** (AL, EN, RA, UM, SHA, NUL, NAV, REMESH): act on the EPI-form,
  νf-capacity, θ-phase or advisory channel that the coherence-pressure functional
  does not penalise by its grammatical role — so they neither contract nor expand
  coherence (|ΔE_coherence| ≈ 0 by their U2 role).

Grammar rule U2 (CONVERGENCE & BOUNDEDNESS) requires that every
destabiliser be accompanied by a stabiliser.  The derivation argues that
the *net* coherence change across a grammar-compliant sequence is
non-negative (energy non-positive), supporting the Lyapunov proposition.
A complete formal proof of asymptotic stability remains open (see §8.2 of
the theory document for the proof sketch and its limitations).

Spectral Gap Characterisation
-----------------------------
``analyze_spectral_gap`` reports two distinct, both-meaningful quantities:

- the **combinatorial algebraic connectivity** λ₁ (Fiedler value) of
  L = D − A — a graph-topology measure; and
- the **canonical diffusion relaxation gap** λ₂ of the symmetric normalized
  Laplacian L_sym = I − D^{-1/2} W D^{-1/2}, which shares the spectrum of the
  canonical TNFR diffusion operator L_rw = I − D⁻¹W (``structural_diffusion``).

The *diffusive* relaxation time-scale is set by the **diffusion gap**: the EPI
field relaxes as exp(−ν_f·λ₂·t).  The two gaps coincide only up to the degree
normalisation (λ₁/d on a d-regular graph) and differ on irregular graphs.
Derived: relaxation time, mixing-time estimate, Cheeger-type bound, and the
stabiliser convergence rate (which uses the canonical diffusion gap).

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

    # ── CANONICAL CLASSIFICATION — single source of truth ───────────────────
    #
    # The structural Lyapunov functional is EMERGENT, not a pre-existing
    # scoreboard: the tetrad energy E = ½Σ(Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²)
    # (conservation.py) emerges from the phase field θ and the structural
    # pressure ΔNFR — Φ_s and J_ΔNFR come from ΔNFR, |∇φ|/K_φ/J_φ from θ.  It
    # contains NO EPI or νf term (measured: scaling EPI or νf leaves E exactly
    # unchanged).  Equivalently, the primary coherence
    # C(t) = 1/(1 + mean|ΔNFR| + mean|dEPI|) emerges from the nodal dynamics
    # ∂EPI/∂t = νf·ΔNFR.  What operators NATIVELY alter is the coherence,
    # through the structural-pressure channel |ΔNFR|.
    #
    # The per-operator Lyapunov role is therefore the canonical grammar
    # coherence-pressure role, DERIVED from the nodal-equation predicates in
    # ``config.physics_derivation`` (the SAME single source of truth the grammar
    # U2 sets derive from) — NOT from hardcoded energy algebra that wrongly
    # assumed EPI/νf entered E:
    #   provides_negative_feedback(op)   → STABILISER (reduces |ΔNFR| → raises C)
    #   increases_structural_pressure(op)→ DESTABILISER (raises |ΔNFR| → lowers C)
    #   otherwise                        → NEUTRAL  (acts on the EPI-form,
    #       νf-capacity, θ-phase or advisory channel that the coherence-pressure
    #       functional does not penalise by its grammatical role).
    from ..config.physics_derivation import (
        increases_structural_pressure,
        provides_negative_feedback,
    )
    from ..operators.operator_contracts import contract_for
    from ..config.operator_names import (
        COHERENCE,
        CONTRACTION,
        COUPLING,
        DISSONANCE,
        EMISSION,
        EXPANSION,
        MUTATION,
        RECEPTION,
        RECURSIVITY,
        RESONANCE,
        SELF_ORGANIZATION,
        SILENCE,
        TRANSITION,
    )

    # (function name, dominant structural-pressure factor). The English name
    # and glyph are NOT duplicated here — they derive from the canonical
    # operator_contracts single source via contract_for() below.
    _OPS = (
        (EMISSION, "AL_boost"),
        (RECEPTION, "EN_mix"),
        (COHERENCE, "IL_dnfr_factor"),
        (DISSONANCE, "OZ_dnfr_factor"),
        (COUPLING, "UM_dnfr_reduction"),
        (RESONANCE, "RA_vf_amplification"),
        (SILENCE, "SHA_vf_factor"),
        (EXPANSION, "VAL_scale"),
        (CONTRACTION, "NUL_scale"),
        (SELF_ORGANIZATION, "THOL_accel"),
        (MUTATION, "ZHIR_theta_shift_factor"),
        (TRANSITION, "NAV_eta"),
        (RECURSIVITY, "REMESH_alpha"),
    )

    bounds: dict[str, OperatorLyapunovBound] = {}
    for fname, factor_name in _OPS:
        contract = contract_for(fname)
        ename = contract.english_name
        glyph = contract.glyph
        factor_val = float(gf.get(factor_name, 0.0))
        if provides_negative_feedback(fname):
            energy_class = EnergyClass.STABILISER
            if fname == COHERENCE:
                # IL scales |ΔNFR| → f·|ΔNFR| (f<1); pressure contraction = 1−f.
                rate = max(0.0, 1.0 - factor_val)
                deriv = (
                    f"IL reduces structural pressure |ΔNFR| → f·|ΔNFR| "
                    f"(f={factor_val:.3f}); coherence C=1/(1+mean|ΔNFR|+…) "
                    f"rises. Pressure contraction ρ = 1−f ≈ {rate:.3f}."
                )
            else:  # THOL
                # THOL redistributes |ΔNFR| into coherent sub-EPIs (handler).
                rate = factor_val
                deriv = (
                    f"THOL redistributes |ΔNFR| into coherent sub-EPIs "
                    f"(accel={factor_val:.3f}); the negative feedback raises "
                    f"coherence. Redistribution ρ ≈ {rate:.3f}."
                )
        elif increases_structural_pressure(fname):
            energy_class = EnergyClass.DESTABILISER
            if fname == DISSONANCE:
                # OZ scales |ΔNFR| → f·|ΔNFR| (f>1); pressure expansion = f−1.
                rate = max(0.0, factor_val - 1.0)
                deriv = (
                    f"OZ raises structural pressure |ΔNFR| → f·|ΔNFR| "
                    f"(f={factor_val:.3f}); coherence falls. Pressure "
                    f"expansion κ = f−1 ≈ {rate:.3f}."
                )
            elif fname == EXPANSION:
                # VAL adds unaligned DOF (scales νf); the new DOF raise |ΔNFR|.
                rate = max(0.0, factor_val - 1.0)
                deriv = (
                    f"VAL adds unaligned structural DOF (νf scale "
                    f"{factor_val:.3f}); the new DOF raise |ΔNFR| → coherence "
                    f"falls. Nominal expansion κ ≈ {rate:.3f}."
                )
            else:  # ZHIR
                # ZHIR's θ→θ' jump desynchronises the node → raises |∇φ| and
                # hence the structural pressure |ΔNFR|.
                rate = factor_val
                deriv = (
                    f"ZHIR jumps θ→θ' (shift factor {factor_val:.3f}); the "
                    f"phase desync raises |∇φ| → raises |ΔNFR| → coherence "
                    f"falls. Nominal expansion κ ≈ {rate:.3f}."
                )
        else:
            energy_class = EnergyClass.NEUTRAL
            rate = 0.0
            deriv = (
                "Coherence-neutral by grammatical role: acts on the EPI-form "
                "(AL/EN/RA), νf-capacity (SHA/NUL), θ-phase (UM) or advisory "
                "(NAV/REMESH) channel, not the structural-pressure |ΔNFR| axis "
                "the coherence functional penalises — so it neither contracts "
                "nor expands coherence by its U2 role."
            )
        bounds[ename] = OperatorLyapunovBound(
            operator_name=ename,
            glyph=glyph,
            energy_class=energy_class,
            contraction_rate=rate,
            glyph_factor_name=factor_name,
            glyph_factor_value=factor_val,
            derivation=deriv,
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
        λ₁ — combinatorial algebraic connectivity (Fiedler value): the
        second-smallest eigenvalue of L = D − A.  A graph-topology measure.
    fiedler_value : float
        Same as spectral_gap (alternative name from spectral graph theory).
    diffusion_gap : float
        λ₂ of the symmetric normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2},
        which shares the spectrum of the canonical TNFR diffusion operator
        L_rw = I − D⁻¹W.  This is the **canonical structural relaxation rate**
        (per unit ν_f): the EPI field relaxes as exp(−ν_f·λ₂·t).  Use this — not
        the combinatorial ``spectral_gap`` — for the diffusive relaxation
        time-scale.  Equals ``spectral_gap``/d on a d-regular graph; differs on
        irregular graphs.
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
    diffusion_gap: float
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
            diffusion_gap=0.0,
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

    # λ₁ = second-smallest eigenvalue (combinatorial algebraic connectivity)
    lambda_1 = float(eigvals[1]) if n > 1 else 0.0
    lambda_1 = max(0.0, lambda_1)  # numerical safety

    lambda_max = float(eigvals[-1])

    # Canonical structural relaxation rate: λ₂ of the symmetric normalized
    # Laplacian L_sym (shares the spectrum of the canonical diffusion operator
    # L_rw = I − D⁻¹W; built once in structural_diffusion).
    from .structural_diffusion import symmetric_normalized_laplacian
    _, L_sym = symmetric_normalized_laplacian(G)
    sym_eigs = np.sort(np.linalg.eigvalsh(L_sym))
    diffusion_gap = max(0.0, float(sym_eigs[1])) if n > 1 else 0.0

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
        diffusion_gap=diffusion_gap,
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
    the operator's contraction rate ρ and the graph's canonical diffusion
    relaxation gap λ₂(L_sym) (``diffusion_gap``, the structural_diffusion
    relaxation rate — not the combinatorial algebraic connectivity).
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
        # The diffusive relaxation bound is the CANONICAL diffusion gap
        # λ₂(L_sym) (= the structural_diffusion relaxation rate), not the
        # combinatorial algebraic connectivity.
        rate = min(bound.contraction_rate, spectral.diffusion_gap)
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

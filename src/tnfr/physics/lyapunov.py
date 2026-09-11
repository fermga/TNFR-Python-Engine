r"""U2 policy multipliers and independent graph-spectral diagnostics.

The public API in this module historically called its records *Lyapunov bounds*.
No such per-operator bounds have been derived.  The registry is a finite policy
model: it maps the U2 role of each canonical operator to a nominal multiplier
built from the canonical default glyph factors.  The multiplier is useful for
bookkeeping and comparison, but it is not an upper bound on a measured state
functional.

Two implemented state diagnostics are relevant and distinct:

* ``C(t) = 1/(1 + mean|DeltaNFR| + mean|dEPI|)`` is a coherence read-out;
* ``E = 1/2 sum(Phi_s^2 + |grad phi|^2 + K_phi^2 + J_phi^2
  + J_DeltaNFR^2)`` is a non-negative five-field energy candidate.

They are not equivalent, and an operator's U2 label does not determine the
finite change of either one.  In particular, a zero multiplier adjustment for
a U2-neutral operator means only "no U2 debt adjustment".  It does not predict
zero phase, pressure, coherence, or energy change.

The legacy names ``EnergyClass``, ``OperatorLyapunovBound``,
``compute_operator_energy_bound``, ``verify_operator_lyapunov`` and
``prove_sequence_lyapunov`` remain available.  Their results explicitly describe
the policy model.  New code should prefer the policy-named aliases and functions
defined alongside them.

Spectral Gap Characterisation
-----------------------------
``analyze_spectral_gap`` reports two distinct, both-meaningful quantities:

- the **combinatorial algebraic connectivity** λ₂ (Fiedler value) of
  L = D − A — a graph-topology measure; and
- the **canonical diffusion relaxation gap** λ₂ of the symmetric normalized
  Laplacian L_sym = I − D^{-1/2} W D^{-1/2}, which shares the spectrum of the
  canonical TNFR diffusion operator L_rw = I − D⁻¹W (``structural_diffusion``).

The *diffusive* relaxation time-scale is set by the **diffusion gap**: the EPI
field relaxes as exp(−ν_f·λ₂·t).  The two gaps coincide only up to the degree
normalisation (λ₂(D-W)/d on a d-regular graph) and differ on irregular graphs.
The normalized gap supplies a relaxation scale for homogeneous, fixed-graph,
pure-EPI diffusion.  It does not combine with a per-operation U2 multiplier to
produce a physical convergence rate; those quantities have different scopes and
time semantics.

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

# ---------------------------------------------------------------------------
#  Energy class taxonomy
# ---------------------------------------------------------------------------


class EnergyClass(str, Enum):
    """Legacy name for an operator's U2 bookkeeping role.

    The enum values do not classify the sign of the five-field energy change.
    ``MIXED`` is retained for compatibility; the current U2 partition does not
    assign it to a canonical operator.
    """

    STABILISER = "stabiliser"
    DESTABILISER = "destabiliser"
    NEUTRAL = "neutral"
    MIXED = "mixed"


U2PolicyRole = EnergyClass


# ---------------------------------------------------------------------------
#  Per-operator policy multiplier (legacy class name retained)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorLyapunovBound:
    r"""Nominal U2-role multiplier for one canonical operator.

    ``OperatorLyapunovBound`` is a compatibility name.  ``energy_class`` and
    ``contraction_rate`` likewise preserve the historical schema; canonically
    they mean ``policy_role`` and ``policy_rate``.  The model assigns multiplier
    ``1-rho`` to a stabilizer, ``1+kappa`` to a destabilizer, and ``1`` to a
    U2-neutral operator.  It makes no statement about a measured energy change.

    Attributes
    ----------
    operator_name : str
        Canonical operator name (e.g. ``"Coherence"``).
    glyph : str
        Two-to-five letter glyph (e.g. ``"IL"``).
    energy_class : EnergyClass
        U2 bookkeeping role.  This is not an observed energy-sign class.
    contraction_rate : float
        Legacy field containing the dimensionless policy-rate adjustment.
    glyph_factor_name : str
        Name of the representative glyph factor selected by this finite policy
        model (e.g. ``"IL_dnfr_factor"``).
    glyph_factor_value : float
        Shared-registry-validated canonical default.  Runtime graph overrides
        are intentionally outside this representative policy table.
    derivation : str
        Human-readable policy rationale and scope statement.
    """

    operator_name: str
    glyph: str
    energy_class: EnergyClass
    contraction_rate: float
    glyph_factor_name: str
    glyph_factor_value: float
    derivation: str

    @property
    def policy_role(self) -> EnergyClass:
        """U2 bookkeeping role (preferred name for ``energy_class``)."""
        return self.energy_class

    @property
    def policy_rate(self) -> float:
        """Dimensionless policy adjustment (preferred name for the legacy field)."""
        return self.contraction_rate

    @property
    def policy_multiplier(self) -> float:
        """Return the nominal multiplier assigned by the U2 policy model."""
        if self.energy_class == EnergyClass.STABILISER:
            return max(0.0, 1.0 - self.contraction_rate)
        if self.energy_class in {EnergyClass.DESTABILISER, EnergyClass.MIXED}:
            return 1.0 + self.contraction_rate
        return 1.0

    @property
    def verification_scope(self) -> str:
        """Machine-readable reminder that this record is not an analytic bound."""
        return "u2_policy_heuristic"

    @property
    def is_energy_bound(self) -> bool:
        """Whether this record certifies a bound on measured energy (always false)."""
        return False


# Preferred descriptive alias.  The original class object and constructor stay
# intact for callers importing ``OperatorLyapunovBound``.
OperatorPolicyMultiplier = OperatorLyapunovBound


# ---------------------------------------------------------------------------
#  Registry of policy multipliers for all 13 operators
# ---------------------------------------------------------------------------


def _build_bounds() -> dict[str, OperatorLyapunovBound]:
    """Construct the U2 policy-multiplier dictionary.

    Roles come from the canonical grammar predicates.  Numeric values come from
    validated canonical defaults in the shared glyph-factor registry.  Runtime
    graph overrides do not enter this representative/default-only table.  The
    mapping from one representative factor to a multiplier is a declared
    compatibility policy, not a dynamical derivation.
    """
    # Classification is centralized in physics_derivation.  These predicates
    # encode U2 composition roles; their names do not establish a sign for the
    # five-field energy or for coherence on every realized state.
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
    from ..config.physics_derivation import (
        increases_structural_pressure,
        provides_negative_feedback,
    )
    from ..operators.factor_contracts import (
        GLYPH_FACTOR_SPECS,
        canonical_glyph_factor_defaults,
        validate_glyph_factors,
    )
    from ..operators.operator_contracts import contract_for

    # Validate the complete canonical table before selecting one representative
    # factor per operator.  This prevents a non-representative invalid default
    # from being hidden by the intentionally partial U2 policy projection.
    glyph_defaults = validate_glyph_factors(
        canonical_glyph_factor_defaults(),
        preserve_unknown=False,
    )

    # (function name, representative canonical-default factor). English names
    # and glyphs derive from operator_contracts and are not duplicated here.
    representative_factors = (
        (EMISSION, "AL_boost"),
        (RECEPTION, "EN_mix"),
        (COHERENCE, "IL_dnfr_factor"),
        (DISSONANCE, "OZ_dnfr_factor"),
        (COUPLING, "UM_theta_push"),
        (RESONANCE, "RA_epi_diff"),
        (SILENCE, "SHA_vf_factor"),
        (EXPANSION, "VAL_scale"),
        (CONTRACTION, "NUL_scale"),
        (SELF_ORGANIZATION, "THOL_accel"),
        (MUTATION, "ZHIR_theta_shift_factor"),
        (TRANSITION, "NAV_eta"),
        (RECURSIVITY, "REMESH_alpha"),
    )

    bounds: dict[str, OperatorLyapunovBound] = {}
    for fname, factor_name in representative_factors:
        contract = contract_for(fname)
        ename = contract.english_name
        glyph = contract.glyph
        factor_spec = GLYPH_FACTOR_SPECS[factor_name]
        if not factor_spec.has_canonical_default:
            raise RuntimeError(
                f"U2 representative {factor_name!r} has no canonical default"
            )
        if factor_spec.glyph.value != glyph:
            raise RuntimeError(
                f"U2 representative {factor_name!r} belongs to "
                f"{factor_spec.glyph.value}, not {glyph}"
            )
        try:
            factor_val = glyph_defaults[factor_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Canonical default missing for U2 representative {factor_name!r}"
            ) from exc
        if provides_negative_feedback(fname):
            energy_class = EnergyClass.STABILISER
            if fname == COHERENCE:
                rate = max(0.0, 1.0 - factor_val)
                deriv = (
                    f"U2 stabilizer. The canonical default pressure-retention "
                    f"factor is {factor_val:.6g}, so the policy model assigns "
                    f"multiplier {factor_val:.6g}. This does not bound the "
                    "five-field energy or coherence change."
                )
            else:  # THOL
                # This unit ceiling belongs to the legacy multiplier encoding,
                # not to THOL's operator-factor domain (which only requires a
                # positive acceleration). A larger future canonical default is
                # legitimate for THOL but requires revising this policy model.
                if factor_val > 1.0:
                    raise ValueError(
                        "THOL_accel exceeds the unit-rate U2 policy encoding"
                    )
                rate = factor_val
                deriv = (
                    f"U2 stabilizer. The canonical default acceleration factor "
                    f"{factor_val:.6g} is reused as a nominal policy adjustment, "
                    "not as a derived contraction rate of a state functional."
                )
        elif increases_structural_pressure(fname):
            energy_class = EnergyClass.DESTABILISER
            if fname == DISSONANCE:
                rate = max(0.0, factor_val - 1.0)
                deriv = (
                    f"U2 destabilizer. The canonical default pressure factor "
                    f"{factor_val:.6g} is used directly as the policy "
                    "multiplier; it is not a global energy-gain bound."
                )
            elif fname == EXPANSION:
                rate = max(0.0, factor_val - 1.0)
                deriv = (
                    f"U2 destabilizer. The canonical default capacity scale "
                    f"{factor_val:.6g} is reused as a policy multiplier. "
                    "Changing capacity alone does not determine energy change."
                )
            else:  # ZHIR
                rate = factor_val
                deriv = (
                    f"U2 destabilizer. The canonical default phase-shift factor "
                    f"{factor_val:.6g} is reused as a nominal multiplier "
                    "adjustment; no universal phase-energy gain follows."
                )
        else:
            energy_class = EnergyClass.NEUTRAL
            rate = 0.0
            deriv = (
                "U2-neutral bookkeeping role: the policy multiplier is one. "
                "Neutrality here means no stabilizer/destabilizer debt; it does "
                "not predict a zero state, coherence, or energy change."
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


# Singleton registry.  The policy-named object is canonical; the historical
# Lyapunov name points to the same dictionary for source compatibility.
OPERATOR_LYAPUNOV_BOUNDS: dict[str, OperatorLyapunovBound] = _build_bounds()
OPERATOR_POLICY_MULTIPLIERS = OPERATOR_LYAPUNOV_BOUNDS

# Glyph → name lookup
_GLYPH_TO_NAME: dict[str, str] = {
    b.glyph: b.operator_name for b in OPERATOR_LYAPUNOV_BOUNDS.values()
}


def get_bound(name_or_glyph: str) -> OperatorLyapunovBound:
    """Look up a policy multiplier by operator name or glyph.

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


def get_policy_multiplier(name_or_glyph: str) -> OperatorPolicyMultiplier:
    """Preferred policy-named alias for :func:`get_bound`."""
    return get_bound(name_or_glyph)


# ---------------------------------------------------------------------------
#  Policy score computation (legacy energy-bound name retained)
# ---------------------------------------------------------------------------


def _validate_policy_score(value: float, name: str) -> float:
    """Return a finite non-negative score or raise a clear error."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative scalar") from exc
    if not math.isfinite(score) or score < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return score


def _validate_node_count(n_nodes: int) -> int:
    """Validate the compatibility-only node-count argument."""
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes < 1:
        raise ValueError("n_nodes must be a positive integer")
    return n_nodes


def _operator_sequence(operator_names: Sequence[str]) -> tuple[str, ...]:
    """Materialize and validate an operator-name sequence once."""
    if isinstance(operator_names, (str, bytes)):
        raise TypeError("operator_names must be a sequence of operator names")
    try:
        operators = tuple(operator_names)
    except TypeError as exc:
        raise TypeError("operator_names must be a sequence of operator names") from exc
    if any(not isinstance(name, str) for name in operators):
        raise TypeError("each operator name or glyph must be a string")
    return operators


def compute_operator_policy_delta(
    name_or_glyph: str,
    score_before: float,
    n_nodes: int = 1,
) -> float:
    r"""Return one nominal change in an abstract non-negative policy score.

    ``n_nodes`` is retained because it was part of the historical energy-bound
    API.  The current multiplier model is scale-free, so the value does not enter
    the calculation after validation.

    Returns
    -------
    float
        ``(policy_multiplier - 1) * score_before``.  This is neither a prediction
        nor an upper bound on the five-field energy.
    """
    score = _validate_policy_score(score_before, "score_before")
    _validate_node_count(n_nodes)
    policy = get_policy_multiplier(name_or_glyph)
    return (policy.policy_multiplier - 1.0) * score


def compute_operator_energy_bound(
    name_or_glyph: str,
    energy_before: float,
    n_nodes: int = 1,
) -> float:
    r"""Compatibility wrapper for :func:`compute_operator_policy_delta`.

    The return value is a nominal policy-score change.  Despite the historical
    function name, it is not a mathematical bound on ``energy_before`` or on an
    observed TNFR energy change.
    """
    return compute_operator_policy_delta(name_or_glyph, energy_before, n_nodes)


# ---------------------------------------------------------------------------
#  Sequence multiplier composition (grammar-role model)
# ---------------------------------------------------------------------------


def compute_sequence_policy_score(
    operator_names: Sequence[str],
    score_initial: float,
    n_nodes: int = 1,
) -> float:
    r"""Compose U2 multipliers on an abstract non-negative policy score."""
    score = _validate_policy_score(score_initial, "score_initial")
    _validate_node_count(n_nodes)
    for name in _operator_sequence(operator_names):
        score *= get_policy_multiplier(name).policy_multiplier
    return score


def compute_sequence_energy_bound(
    operator_names: Sequence[str],
    energy_initial: float,
    n_nodes: int = 1,
) -> float:
    r"""Compatibility wrapper returning the nominal final policy score.

    ``energy_initial`` is interpreted as the initial abstract score.  The result
    is not an upper bound on the measured five-field energy.
    """
    return compute_sequence_policy_score(operator_names, energy_initial, n_nodes)


# ---------------------------------------------------------------------------
#  Comparison of measured energy with the independent policy model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorLyapunovVerification:
    r"""Measured energy change compared with a nominal policy-score change.

    This compatibility record does not verify a Lyapunov theorem.  The fields
    ``theoretical_bound`` and ``within_bound`` retain their names; they represent
    the model delta and a one-sided policy screen, respectively.

    Attributes
    ----------
    operator_name : str
    glyph : str
    energy_before : float
    energy_after : float
    delta_e : float
        Actual E_after - E_before.
    theoretical_bound : float
        Nominal policy-score delta (legacy field name).
    within_bound : bool
        Result of the legacy one-sided comparison.  It is not a certificate.
    margin : float
        Nominal policy delta minus measured energy delta.
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

    @property
    def policy_delta(self) -> float:
        """Preferred name for ``theoretical_bound``."""
        return self.theoretical_bound

    @property
    def policy_screen_passed(self) -> bool:
        """Preferred name for the legacy one-sided comparison result."""
        return self.within_bound

    @property
    def policy_residual(self) -> float:
        """Measured energy change minus nominal policy-score change."""
        return -self.margin

    @property
    def policy_multiplier(self) -> float:
        """Nominal score multiplier reconstructed from the compatibility fields."""
        if self.energy_before == 0.0:
            return 1.0
        return 1.0 + self.policy_delta / self.energy_before

    @property
    def observed_energy_ratio(self) -> float:
        """Measured ``energy_after / energy_before``, or ``nan`` at zero baseline."""
        if self.energy_before == 0.0:
            return float("nan")
        return self.energy_after / self.energy_before

    @property
    def multiplier_residual(self) -> float:
        """Observed energy ratio minus policy multiplier, when defined."""
        ratio = self.observed_energy_ratio
        return ratio - self.policy_multiplier if math.isfinite(ratio) else float("nan")

    @property
    def verification_scope(self) -> str:
        return "measured_energy_vs_u2_policy_heuristic"

    @property
    def is_lyapunov_certificate(self) -> bool:
        return False


OperatorPolicyComparison = OperatorLyapunovVerification


def compare_operator_energy_to_policy(
    name_or_glyph: str,
    energy_before: float,
    energy_after: float,
    n_nodes: int = 1,
    tolerance: float = 1e-6,
) -> OperatorLyapunovVerification:
    r"""Compare a measured energy change with the nominal U2 policy delta.

    Parameters
    ----------
    name_or_glyph : str
        Operator name or glyph.
    energy_before, energy_after : float
        Measured energy functional values.
    n_nodes : int
        Number of affected nodes.
    tolerance : float
        Non-negative relative tolerance for the legacy one-sided screen.

    Returns
    -------
    OperatorLyapunovVerification
    """
    before = _validate_policy_score(energy_before, "energy_before")
    after = _validate_policy_score(energy_after, "energy_after")
    _validate_node_count(n_nodes)
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be finite and non-negative") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    bound_info = get_policy_multiplier(name_or_glyph)
    delta_e = after - before
    theoretical = compute_operator_policy_delta(name_or_glyph, before, n_nodes)
    margin = theoretical - delta_e
    scale = max(1.0, before, after, abs(theoretical), abs(delta_e))

    return OperatorLyapunovVerification(
        operator_name=bound_info.operator_name,
        glyph=bound_info.glyph,
        energy_before=before,
        energy_after=after,
        delta_e=delta_e,
        theoretical_bound=theoretical,
        within_bound=(delta_e <= theoretical + tolerance * scale),
        margin=margin,
        energy_class=bound_info.energy_class,
    )


def verify_operator_lyapunov(
    name_or_glyph: str,
    energy_before: float,
    energy_after: float,
    n_nodes: int = 1,
    tolerance: float = 1e-6,
) -> OperatorLyapunovVerification:
    r"""Compatibility wrapper for :func:`compare_operator_energy_to_policy`."""
    return compare_operator_energy_to_policy(
        name_or_glyph,
        energy_before,
        energy_after,
        n_nodes=n_nodes,
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
#  Spectral gap analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralGapAnalysis:
    r"""Combinatorial and normalized spectral read-outs for one graph.

    Attributes
    ----------
    spectral_gap : float
        Combinatorial algebraic connectivity (Fiedler value): the
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
        Per-unit-capacity pure-EPI scale ``1/diffusion_gap``.  A physical time
        additionally needs a homogeneous capacity and a fixed graph.
    convergence_rate : float
        Legacy field containing ``diffusion_gap`` per unit capacity.
    mixing_time_bound : float
        Legacy ``log(N)/diffusion_gap`` topology scale.  It is not a universal
        total-variation mixing bound on irregular weighted graphs.
    cheeger_lower : float
        ``diffusion_gap/2``, the standard normalized-Cheeger lower expression
        for conductance under the usual reversible-graph convention.
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
    r"""Compute independent combinatorial and normalized graph gaps.

    The returned pure-EPI relaxation scale uses the normalized diffusion gap.
    It assumes a fixed connected symmetric graph and unit homogeneous capacity.
    Heterogeneous capacities require the generalized certificate in
    :mod:`structural_diffusion`.

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

    # Second-smallest eigenvalue (combinatorial algebraic connectivity).
    combinatorial_gap = float(eigvals[1]) if n > 1 else 0.0
    combinatorial_gap = max(0.0, combinatorial_gap)  # numerical safety

    lambda_max = float(eigvals[-1])

    # Canonical structural relaxation rate: λ₂ of the symmetric normalized
    # Laplacian L_sym (shares the spectrum of the canonical diffusion operator
    # L_rw = I − D⁻¹W; built once in structural_diffusion).
    from .structural_diffusion import symmetric_normalized_laplacian

    _, L_sym = symmetric_normalized_laplacian(G)
    sym_eigs = np.sort(np.linalg.eigvalsh(L_sym))
    diffusion_gap = max(0.0, float(sym_eigs[1])) if n > 1 else 0.0

    # Normalization removes a harmless global conductance scale, so it is the
    # more reliable connectivity signal for very small or very large weights.
    is_connected = diffusion_gap > 1e-10
    if not is_connected:
        combinatorial_gap = 0.0
        diffusion_gap = 0.0
    tau = 1.0 / diffusion_gap if is_connected else float("inf")
    mixing = math.log(n) / diffusion_gap if is_connected else float("inf")
    cheeger_lower = 0.5 * diffusion_gap if is_connected else 0.0

    ratio = (
        lambda_max / combinatorial_gap
        if is_connected and combinatorial_gap > 0.0
        else float("inf")
    )

    return SpectralGapAnalysis(
        spectral_gap=combinatorial_gap,
        fiedler_value=combinatorial_gap,
        diffusion_gap=diffusion_gap,
        relaxation_time=tau,
        convergence_rate=diffusion_gap,
        mixing_time_bound=mixing,
        cheeger_lower=cheeger_lower,
        n_nodes=n,
        max_eigenvalue=lambda_max,
        spectral_ratio=ratio,
        is_connected=is_connected,
        eigenvalues=eigvals,
    )


# ---------------------------------------------------------------------------
#  Side-by-side policy and spectral context (legacy names retained)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LyapunovSpectralSummary:
    r"""Side-by-side U2 policy and graph-spectral read-outs.

    A policy multiplier acts per operator position.  ``diffusion_gap`` acts per
    unit continuous time for a restricted pure-EPI model.  No canonical mapping
    between those clocks is available, so this record does not combine them into
    an effective physical rate.

    Attributes
    ----------
    operator_bound : OperatorLyapunovBound
        Nominal U2-role multiplier for the operator.
    spectral : SpectralGapAnalysis
        Spectral gap analysis of the graph.
    effective_convergence_rate : float
        Deprecated compatibility field.  Always ``nan`` because the combined
        rate is undefined without an explicit operator-time dynamical model.
    steps_to_half_energy : float
        Deprecated compatibility field.  Equals ``policy_half_steps`` and says
        nothing about measured energy.
    policy_multiplier : float
        Nominal multiplier per operator position.
    policy_half_steps : float
        Number of repeated nominal multipliers needed to halve the abstract
        score; ``inf`` unless ``0 < policy_multiplier < 1``.
    diffusion_relaxation_time : float
        Independent per-unit-capacity pure-EPI relaxation scale.
    combination_defined : bool
        Always false for this API because the required bridge is absent.
    """

    operator_bound: OperatorLyapunovBound
    spectral: SpectralGapAnalysis
    effective_convergence_rate: float
    steps_to_half_energy: float
    policy_multiplier: float = 1.0
    policy_half_steps: float = float("inf")
    diffusion_relaxation_time: float = float("inf")
    combination_defined: bool = False
    verification_scope: str = "independent_policy_and_pure_epi_spectral_readouts"


OperatorPolicySpectralContext = LyapunovSpectralSummary


def analyze_operator_policy_context(
    G: Any,
    name_or_glyph: str,
) -> LyapunovSpectralSummary:
    r"""Report policy multiplier and pure-EPI spectral scale side by side.

    The result deliberately leaves ``effective_convergence_rate`` undefined.
    Taking ``min(policy_rate, diffusion_gap)`` would mix a dimensionless
    per-operation adjustment with a continuous-time eigenvalue and would not be
    a theorem about IL, THOL, or any other realized operator.

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
    bound = get_policy_multiplier(name_or_glyph)
    spectral = analyze_spectral_gap(G)
    multiplier = bound.policy_multiplier
    policy_half_steps = (
        math.log(0.5) / math.log(multiplier)
        if 0.0 < multiplier < 1.0
        else float("inf")
    )

    return LyapunovSpectralSummary(
        operator_bound=bound,
        spectral=spectral,
        effective_convergence_rate=float("nan"),
        steps_to_half_energy=policy_half_steps,
        policy_multiplier=multiplier,
        policy_half_steps=policy_half_steps,
        diffusion_relaxation_time=spectral.relaxation_time,
        combination_defined=False,
        verification_scope="independent_policy_and_pure_epi_spectral_readouts",
    )


def analyze_operator_convergence(
    G: Any,
    name_or_glyph: str,
) -> LyapunovSpectralSummary:
    r"""Compatibility wrapper for :func:`analyze_operator_policy_context`."""
    return analyze_operator_policy_context(G, name_or_glyph)


# ---------------------------------------------------------------------------
#  Sequence analysis for the legacy U2 multiplier model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceLyapunovProof:
    r"""Product of nominal U2 policy multipliers for a supplied sequence.

    This records the product of policy multipliers and whether that product is
    at most one.  It is not a proof that U2 implies tetrad-energy contraction.

    Attributes
    ----------
    operators : tuple
        Operator names in sequence order.
    energy_multipliers : tuple
        Legacy name for the per-position policy multipliers.
    cumulative_product : float
        Product of nominal policy multipliers.
    is_net_contractive : bool
        True if cumulative_product ≤ 1.0 in the nominal model.
    net_contraction : float
        1 - cumulative_product (positive = nominal contraction).
    """

    operators: tuple
    energy_multipliers: tuple
    cumulative_product: float
    is_net_contractive: bool
    net_contraction: float

    @property
    def policy_multipliers(self) -> tuple:
        """Preferred name for ``energy_multipliers``."""
        return self.energy_multipliers

    @property
    def policy_product_at_most_one(self) -> bool:
        """Preferred name for ``is_net_contractive``."""
        return self.is_net_contractive

    @property
    def verification_scope(self) -> str:
        return "u2_policy_multiplier_product"

    @property
    def is_lyapunov_proof(self) -> bool:
        return False


SequencePolicyEvaluation = SequenceLyapunovProof


def evaluate_sequence_policy(
    operator_names: Sequence[str],
) -> SequenceLyapunovProof:
    r"""Evaluate the nominal U2 multiplier product for an operator sequence.

    Each operator contributes a multiplier to an abstract policy score:
    - Stabiliser with rate ρ: factor = 1 - ρ  (< 1)
    - Destabiliser with rate κ: factor = 1 + κ  (> 1)
    - Neutral with residual ε: factor = 1 + ε  (≈ 1)
    - Mixed with rate κ: factor = 1 + κ  (worst case)

    The function does not validate grammar and does not inspect a trajectory.
    A product at most one is only a property of this finite multiplier model.

    Parameters
    ----------
    operator_names : Sequence[str]
        Ordered list of operator names or glyphs.

    Returns
    -------
    SequenceLyapunovProof
    """
    operators = _operator_sequence(operator_names)
    multipliers = [get_policy_multiplier(name).policy_multiplier for name in operators]

    product = 1.0
    for f in multipliers:
        product *= f

    return SequenceLyapunovProof(
        operators=operators,
        energy_multipliers=tuple(multipliers),
        cumulative_product=product,
        is_net_contractive=product <= 1.0,
        net_contraction=1.0 - product,
    )


def prove_sequence_lyapunov(
    operator_names: Sequence[str],
) -> SequenceLyapunovProof:
    r"""Compatibility wrapper for :func:`evaluate_sequence_policy`.

    The historical name does not turn the returned multiplier product into a
    Lyapunov proof.
    """
    return evaluate_sequence_policy(operator_names)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "U2PolicyRole",
    "EnergyClass",
    # Data structures
    "OperatorPolicyMultiplier",
    "OperatorPolicyComparison",
    "OperatorPolicySpectralContext",
    "SequencePolicyEvaluation",
    "OperatorLyapunovBound",
    "OperatorLyapunovVerification",
    "SpectralGapAnalysis",
    "LyapunovSpectralSummary",
    "SequenceLyapunovProof",
    # Registry
    "OPERATOR_POLICY_MULTIPLIERS",
    "OPERATOR_LYAPUNOV_BOUNDS",
    "get_policy_multiplier",
    "get_bound",
    # Per-operator analysis
    "compute_operator_policy_delta",
    "compare_operator_energy_to_policy",
    "compute_operator_energy_bound",
    "verify_operator_lyapunov",
    # Sequence analysis
    "compute_sequence_policy_score",
    "evaluate_sequence_policy",
    "compute_sequence_energy_bound",
    "prove_sequence_lyapunov",
    # Spectral gap
    "analyze_spectral_gap",
    # Combined analysis
    "analyze_operator_policy_context",
    "analyze_operator_convergence",
]

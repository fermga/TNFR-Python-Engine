"""TNFR structural conservation diagnostics.

The module records the balance

    Δρ/Δt + div J = S,  ρ = Φ_s + K_φ,
    J = (J_φ, J_ΔNFR),

between two observed graph states.  ``S`` is the measured source/residual of
that finite-interval balance.  This identity follows from the definition of
``S``; grammar-valid operator labels alone do not prove that it vanishes or is
small.  Likewise, the non-negative quadratic structural energy exposed here is
a Lyapunov *candidate*: monotonicity requires trajectory evidence or a
model-specific proof.

The exact conservation result owned elsewhere is the degree-weighted EPI total
for fixed symmetric pure diffusion under its stated capacity assumptions.  It
must not be conflated with the tetrad charge reported by this module.

STATUS
======
CANONICAL DIAGNOSTIC INTERFACE.  Residuals, charge drift, and energy change are
observations of the supplied trajectory, not consequences of U1-U6 by label.

References
----------
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t) [TNFR.pdf §2.1]
- Grammar U1-U6: theory/UNIFIED_GRAMMAR_RULES.md
- Structural fields: src/tnfr/physics/canonical.py
- Extended fields: src/tnfr/physics/extended.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..mathematics.unified_numerical import np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from ..constants.canonical import (
    K_PHI_CANONICAL_THRESHOLD,
    PI,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)
from .canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from .extended import compute_dnfr_flux, compute_phase_current
from .unified import _capture_structural_fields, _energy_density_from_fields
from .unified import compute_energy_density as _raw_energy_density

# ---------------------------------------------------------------------------
# Conservation diagnostic alert levels
# ---------------------------------------------------------------------------
_BALANCE_RMS_ALERT = 1.0
_SECTOR_IMBALANCE_RATIO = 1.5

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ConservationAlertLevels(dict[str, float]):
    """Backward-compatible numeric alert mapping with explicit scope metadata.

    The historical API returned a plain ``dict[str, float]`` from
    :func:`compute_grammar_conservation_bounds`.  This subclass preserves that
    mapping behaviour and its legacy keys while exposing why the values must
    not be read as mathematical bounds or as grammar-rule classifiers.
    """

    scope = "finite_structural_balance_alerts"
    thresholds_are_proven_bounds = False
    grammar_validation_applicable = False
    applicable_grammar_rules: tuple[str, ...] = ()
    u6_drift_requires_reference = True

    @property
    def metadata(self) -> dict[str, Any]:
        """Return detached scope metadata without changing numeric iteration."""
        return {
            "scope": self.scope,
            "thresholds_are_proven_bounds": self.thresholds_are_proven_bounds,
            "grammar_validation_applicable": self.grammar_validation_applicable,
            "applicable_grammar_rules": self.applicable_grammar_rules,
            "u6_drift_requires_reference": self.u6_drift_requires_reference,
            "legacy_pi_scaled_alerts": True,
        }


@dataclass(frozen=True)
class ConservationSnapshot:
    """Single-time snapshot of structural balance fields at every node.

    Attributes
    ----------
    charge_density : dict[Any, float]
        ρ(i) = Φ_s(i) + K_φ(i) at each node.
    phi_s : dict[Any, float]
        Structural potential Φ_s per node.
    k_phi : dict[Any, float]
        Phase curvature K_φ per node.
    j_phi : dict[Any, float]
        Phase current J_φ per node.
    j_dnfr : dict[Any, float]
        ΔNFR flux J_ΔNFR per node.
    grad_phi : dict[Any, float]
        Phase gradient |∇φ| per node.
    divergence : dict[Any, float]
        Discrete divergence div(J) at each node.
    """

    charge_density: dict[Any, float]
    phi_s: dict[Any, float]
    k_phi: dict[Any, float]
    j_phi: dict[Any, float]
    j_dnfr: dict[Any, float]
    grad_phi: dict[Any, float]
    divergence: dict[Any, float]


@dataclass
class ConservationBalance:
    """Result of the continuity equation verification across two snapshots.

    Uses **Crank-Nicolson (trapezoidal)** discretization for O(Δt²)
    accuracy:

    * ``residual[i] = Δρ(i)/Δt + ½[div J_before(i) + div J_after(i)]``
      — a finite-interval balance diagnostic, not a grammar validator.
    * ``mean_residual``, ``max_residual`` — aggregate diagnostics.
    * ``conservation_quality`` — scalar in [0, 1]; 1 means zero measured RMS
      residual for this finite interval.
    * ``grammar_violation_index`` — legacy name for the mean absolute residual;
      zero does not certify grammar compliance.
    """

    residual: dict[Any, float]
    delta_rho: dict[Any, float]
    divergence_after: dict[Any, float]  # trapezoidal average (before+after)/2
    mean_residual: float
    std_residual: float
    max_residual: float
    rms_residual: float
    conservation_quality: float
    grammar_violation_index: float
    total_charge_before: float
    total_charge_after: float
    charge_drift: float
    diagnostic_scope: str = "two_snapshot_structural_balance"
    grammar_validation_applicable: bool = False
    assessed_grammar_rules: tuple[str, ...] = ()

    @property
    def balance_alert_index(self) -> float:
        """Return the mean absolute residual under an accurate public name."""
        return self.grammar_violation_index


@dataclass
class ConservationTimeSeries:
    """Full time-series of conservation diagnostics.

    Built incrementally via :meth:`ConservationTracker.record`.
    """

    times: list[float] = field(default_factory=list)
    total_charge: list[float] = field(default_factory=list)
    mean_residuals: list[float] = field(default_factory=list)
    rms_residuals: list[float] = field(default_factory=list)
    conservation_quality: list[float] = field(default_factory=list)
    grammar_violation_index: list[float] = field(default_factory=list)
    charge_drift: list[float] = field(default_factory=list)

    @property
    def is_conserved(self) -> bool:
        """Return the legacy finite-series quality alert classification.

        This alias neither proves a conservation law nor validates grammar.
        Prefer :attr:`aggregate_balance_within_alert` in new code.
        """
        return self.aggregate_balance_within_alert

    @property
    def aggregate_balance_within_alert(self) -> bool:
        """Whether measured interval quality reaches the legacy alert cut.

        The synthetic baseline stored for the first snapshot has no preceding
        interval and is excluded.
        """
        if len(self.conservation_quality) <= 1:
            return False
        return self.sampled_mean_quality >= 0.9

    @property
    def grammar_validation_applicable(self) -> bool:
        """Grammar cannot be inferred from this residual time series."""
        return False

    @property
    def sampled_mean_quality(self) -> float:
        """Average quality over actual two-snapshot balance intervals."""
        if len(self.conservation_quality) <= 1:
            return 0.0
        return float(np.mean(self.conservation_quality[1:]))

    @property
    def mean_quality(self) -> float:
        """Compatibility name for :attr:`sampled_mean_quality`."""
        return self.sampled_mean_quality

    @property
    def mean_quality_including_baseline(self) -> float:
        """Return the historical mean including the synthetic first value."""
        if not self.conservation_quality:
            return 0.0
        return float(np.mean(self.conservation_quality))


# ---------------------------------------------------------------------------
# Core computation: structural charge density and divergence
# ---------------------------------------------------------------------------


def compute_charge_density(G: Any) -> dict[Any, float]:
    r"""Compute structural charge density ρ(i) = Φ_s(i) + K_φ(i).

    This is the Noether-like charge-density diagnostic used by TNFR:
    - Φ_s captures global structural potential (long-range coupling)
    - K_φ captures local geometric curvature (short-range confinement)

    Their sum combines global and local structural fields. Conservation of its
    total must be checked on the actual trajectory or derived for a specified
    auxiliary model.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[node, float]
        ρ(i) per node.
    """
    phi_s = compute_structural_potential(G)
    k_phi = compute_phase_curvature(G)
    return _charge_density_from_fields(phi_s, k_phi)


def _charge_density_from_fields(
    phi_s: dict[Any, float], k_phi: dict[Any, float]
) -> dict[Any, float]:
    """Shared charge definition for live and captured field maps."""
    return {n: phi_s[n] + k_phi.get(n, 0.0) for n in phi_s}


def compute_current_divergence(G: Any) -> dict[Any, float]:
    r"""Compute discrete divergence of structural current div J(i).

    The structural current is J = (J_φ, J_ΔNFR).  On a graph, the
    stored quantity at node i uses the legacy neighbor-minus-center sign:

        div J(i) = (1/|N(i)|) Σ_{j∈N(i)} [
            (J_φ(j) - J_φ(i)) + (J_ΔNFR(j) - J_ΔNFR(i))
        ]

    Thus it is ``-L_rw`` applied to each current component for an unweighted
    graph, conventionally an inward rather than outward flux. The public name
    is retained for compatibility; all balance routines use this same sign.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    dict[node, float]
        div J(i) per node.
    """
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _current_divergence_from_fields(G, j_phi, j_dnfr)


def _current_divergence_from_fields(
    G: Any, j_phi: dict[Any, float], j_dnfr: dict[Any, float]
) -> dict[Any, float]:
    """Apply the existing neighbor-mean divergence to recorded currents."""
    nodes = list(G.nodes())

    divergence: dict[Any, float] = {}
    for i in nodes:
        neighbors = list(G.neighbors(i))
        if not neighbors:
            divergence[i] = 0.0
            continue

        deg = len(neighbors)
        # Legacy neighbor-minus-center (inward-flux / -L_rw) convention.
        div_j_phi = sum(j_phi.get(j, 0.0) - j_phi.get(i, 0.0) for j in neighbors) / deg
        div_j_dnfr = (
            sum(j_dnfr.get(j, 0.0) - j_dnfr.get(i, 0.0) for j in neighbors) / deg
        )
        divergence[i] = div_j_phi + div_j_dnfr

    return divergence


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------


def capture_conservation_snapshot(G: Any) -> ConservationSnapshot:
    """Capture structural charge, currents, and fields at one instant.

    This is a *read-only* operation that never mutates EPI.

    The caller must hold graph state fixed during the complete call, including
    the topology-dependent divergence. Returned maps are detached; capture is
    not atomic with concurrent evolution.

    Parameters
    ----------
    G : TNFRGraph
        Network in its current state.

    Returns
    -------
    ConservationSnapshot
    """
    fields = _capture_structural_fields(G)
    phi_s, k_phi, grad_phi = fields.phi_s, fields.k_phi, fields.grad_phi
    j_phi, j_dnfr = fields.j_phi, fields.j_dnfr

    charge = _charge_density_from_fields(phi_s, k_phi)
    div_j = _current_divergence_from_fields(G, j_phi, j_dnfr)

    return ConservationSnapshot(
        charge_density=charge,
        phi_s=phi_s,
        k_phi=k_phi,
        j_phi=j_phi,
        j_dnfr=j_dnfr,
        grad_phi=grad_phi,
        divergence=div_j,
    )


# ---------------------------------------------------------------------------
# Conservation balance (two-snapshot comparison)
# ---------------------------------------------------------------------------


def verify_conservation_balance(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    dt: float = 1.0,
) -> ConservationBalance:
    r"""Verify the structural continuity equation between two snapshots.

    Computes the residual of the continuity equation using a
    **Crank-Nicolson (trapezoidal)** discretization:

        Δρ(i)/Δt + ½[div J_before(i) + div J_after(i)] ≈ 0

    For sufficiently smooth evolution on fixed support, the trapezoidal
    approximation has O(Δt²) accuracy, compared to the O(Δt) of a
    right-endpoint scheme. A small residual records agreement with this
    balance over the sampled interval; it does not prove sequence-wide
    conservation or grammar compliance. Large residuals can reflect numerical
    error, changing topology or node support, source terms, or a failure of the
    assumed balance. Their cause requires separate investigation.

    Parameters
    ----------
    before : ConservationSnapshot
        State before operator application.
    after : ConservationSnapshot
        State after operator application.
    dt : float
        Effective time step between snapshots (default 1.0).

    Returns
    -------
    ConservationBalance
        Finite-interval structural-balance diagnostics.
    """
    nodes = list(after.charge_density.keys())

    delta_rho: dict[Any, float] = {}
    residual: dict[Any, float] = {}

    for n in nodes:
        rho_before = before.charge_density.get(n, 0.0)
        rho_after = after.charge_density.get(n, 0.0)
        d_rho = (rho_after - rho_before) / dt
        delta_rho[n] = d_rho

        # Crank-Nicolson: trapezoidal average of divergence at both endpoints
        div_j = 0.5 * (before.divergence.get(n, 0.0) + after.divergence.get(n, 0.0))
        # Continuity: ∂ρ/∂t + div J = S  →  residual = ∂ρ/∂t + div J
        residual[n] = d_rho + div_j

    residual_vals = np.array(list(residual.values()))

    mean_res = float(np.mean(residual_vals)) if len(residual_vals) > 0 else 0.0
    std_res = float(np.std(residual_vals)) if len(residual_vals) > 0 else 0.0
    max_res = float(np.max(np.abs(residual_vals))) if len(residual_vals) > 0 else 0.0
    rms_res = (
        float(np.sqrt(np.mean(residual_vals**2))) if len(residual_vals) > 0 else 0.0
    )

    # Conservation quality: 1/(1 + RMS) maps [0, ∞) → (0, 1]
    quality = 1.0 / (1.0 + rms_res)

    # Legacy field name. This is only the mean absolute balance residual and
    # has no rule-classification semantics.
    gvi = float(np.mean(np.abs(residual_vals))) if len(residual_vals) > 0 else 0.0

    # Total charge tracking
    q_before = sum(before.charge_density.values())
    q_after = sum(after.charge_density.values())
    charge_drift = abs(q_after - q_before)

    return ConservationBalance(
        residual=residual,
        delta_rho=delta_rho,
        divergence_after={
            n: 0.5 * (before.divergence.get(n, 0.0) + after.divergence.get(n, 0.0))
            for n in nodes
        },
        mean_residual=mean_res,
        std_residual=std_res,
        max_residual=max_res,
        rms_residual=rms_res,
        conservation_quality=quality,
        grammar_violation_index=gvi,
        total_charge_before=q_before,
        total_charge_after=q_after,
        charge_drift=charge_drift,
    )


# ---------------------------------------------------------------------------
# Conservation tracker (multi-step)
# ---------------------------------------------------------------------------


class ConservationTracker:
    """Track finite structural-balance diagnostics across an operator sequence.

    Usage
    -----
    >>> tracker = ConservationTracker(G)
    >>> tracker.record(t=0.0)           # initial snapshot
    >>> Emission()(G, node)
    >>> tracker.record(t=1.0)           # after operator
    >>> Coherence()(G, node)
    >>> tracker.record(t=2.0)
    >>> report = tracker.report()
    >>> print(f"Within alert: {report.aggregate_balance_within_alert}")
    """

    def __init__(self, G: Any) -> None:
        self._G = G
        self._snapshots: list[tuple[float, ConservationSnapshot]] = []
        self._series = ConservationTimeSeries()

    def record(self, t: float = 0.0) -> ConservationSnapshot:
        """Capture current state and compute balance against previous snapshot.

        Parameters
        ----------
        t : float
            Structural time stamp for this snapshot.

        Returns
        -------
        ConservationSnapshot
            The captured snapshot (also stored internally).
        """
        snap = capture_conservation_snapshot(self._G)
        self._snapshots.append((t, snap))

        if len(self._snapshots) >= 2:
            t_prev, snap_prev = self._snapshots[-2]
            dt = t - t_prev if t != t_prev else 1.0
            balance = verify_conservation_balance(snap_prev, snap, dt=dt)

            self._series.times.append(t)
            self._series.total_charge.append(balance.total_charge_after)
            self._series.mean_residuals.append(balance.mean_residual)
            self._series.rms_residuals.append(balance.rms_residual)
            self._series.conservation_quality.append(balance.conservation_quality)
            self._series.grammar_violation_index.append(
                balance.grammar_violation_index
            )
            self._series.charge_drift.append(balance.charge_drift)
        else:
            # First snapshot — record initial charge only
            q_total = sum(snap.charge_density.values())
            self._series.times.append(t)
            self._series.total_charge.append(q_total)
            self._series.mean_residuals.append(0.0)
            self._series.rms_residuals.append(0.0)
            self._series.conservation_quality.append(1.0)
            self._series.grammar_violation_index.append(0.0)
            self._series.charge_drift.append(0.0)

        return snap

    def report(self) -> ConservationTimeSeries:
        """Return the accumulated time-series diagnostics."""
        return self._series

    @property
    def latest_balance(self) -> ConservationBalance | None:
        """Return the most recent balance check, or None."""
        if len(self._snapshots) < 2:
            return None
        t_prev, snap_prev = self._snapshots[-2]
        t_curr, snap_curr = self._snapshots[-1]
        dt = t_curr - t_prev if t_curr != t_prev else 1.0
        return verify_conservation_balance(snap_prev, snap_curr, dt=dt)


# ---------------------------------------------------------------------------
# Decomposed analysis: which field component contributes most to residual
# ---------------------------------------------------------------------------


def decompose_conservation_residual(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    dt: float = 1.0,
) -> dict[str, dict[Any, float]]:
    r"""Decompose the continuity residual into Φ_s and K_φ contributions.

    The charge density ρ = Φ_s + K_φ, so:

        Δρ/Δt = ΔΦ_s/Δt + ΔK_φ/Δt

    This function separates potential-field and curvature-field contributions
    to the measured residual.  Their magnitudes do not classify U2 or U3, and
    potential drift evaluates the U6 policy only when compared with the
    canonical two-snapshot threshold through a dedicated U6 checker.

    Divergence is evaluated using the **Crank-Nicolson (trapezoidal)**
    average of the before and after snapshots for O(Δt²) accuracy.

    Returns
    -------
    dict with keys:
        'phi_s_drift'  : per-node ΔΦ_s/Δt
        'k_phi_drift'  : per-node ΔK_φ/Δt
        'j_phi_div'    : per-node div(J_φ) contribution
        'j_dnfr_div'   : per-node div(J_ΔNFR) contribution
        'potential_residual' : ΔΦ_s/Δt + div(J_ΔNFR)  [potential sector]
        'geometric_residual' : ΔK_φ/Δt + div(J_φ)     [geometric sector]
    """
    nodes = list(after.phi_s.keys())

    phi_s_drift: dict[Any, float] = {}
    k_phi_drift: dict[Any, float] = {}
    j_phi_div: dict[Any, float] = {}
    j_dnfr_div: dict[Any, float] = {}
    potential_residual: dict[Any, float] = {}
    geometric_residual: dict[Any, float] = {}

    for n in nodes:
        # Rate of change of each charge component
        d_phi_s = (after.phi_s.get(n, 0.0) - before.phi_s.get(n, 0.0)) / dt
        d_k_phi = (after.k_phi.get(n, 0.0) - before.k_phi.get(n, 0.0)) / dt
        phi_s_drift[n] = d_phi_s
        k_phi_drift[n] = d_k_phi

        # Crank-Nicolson: trapezoidal average of divergence
        div_j = 0.5 * (before.divergence.get(n, 0.0) + after.divergence.get(n, 0.0))
        # Approximate split: use field magnitudes as proxy (averaged)
        j_phi_n = 0.5 * (before.j_phi.get(n, 0.0) + after.j_phi.get(n, 0.0))
        j_dnfr_n = 0.5 * (before.j_dnfr.get(n, 0.0) + after.j_dnfr.get(n, 0.0))
        total_j = abs(j_phi_n) + abs(j_dnfr_n) + 1e-15
        j_phi_fraction = abs(j_phi_n) / total_j
        j_dnfr_fraction = abs(j_dnfr_n) / total_j

        j_phi_div[n] = div_j * j_phi_fraction
        j_dnfr_div[n] = div_j * j_dnfr_fraction

        # Sector residuals (the key physics insight):
        # Potential sector: Φ_s is driven by ΔNFR distribution → coupled to J_ΔNFR
        # Geometric sector: K_φ is driven by phase dynamics → coupled to J_φ
        potential_residual[n] = d_phi_s + j_dnfr_div[n]
        geometric_residual[n] = d_k_phi + j_phi_div[n]

    return {
        "phi_s_drift": phi_s_drift,
        "k_phi_drift": k_phi_drift,
        "j_phi_div": j_phi_div,
        "j_dnfr_div": j_dnfr_div,
        "potential_residual": potential_residual,
        "geometric_residual": geometric_residual,
    }


# ---------------------------------------------------------------------------
# Legacy pi-scaled conservation alert levels
# ---------------------------------------------------------------------------


def compute_grammar_conservation_bounds(G: Any) -> ConservationAlertLevels:
    r"""Compute legacy grammar-scaled diagnostic alert levels.

    The legacy alert construction combines:

    - U2's stabilizer/debt role as qualitative context;
    - the U6 drift policy ``ΔPhi_s < pi/2`` as a numeric scale;
    - the exact trigonometric bound ``|J_phi| <= 1``.

    These values combine historical pi-scaled policies into monitoring alert
    levels. They are not derived U2/U3/U6 bounds, do not evaluate any grammar
    rule, and must not be used as a conservation or grammar certificate. The
    returned object remains a ``dict`` subclass containing only numeric legacy
    keys; its :attr:`ConservationAlertLevels.metadata` property exposes this
    scope without breaking callers that iterate the numeric mapping.

    Returns
    -------
    ConservationAlertLevels
        'max_charge_density'  : legacy policy-scaled alert level for |ρ|
        'max_current_magnitude' : legacy alert level for |J|
        'max_allowed_residual' : legacy alert level for |Δρ/Δt + div J|
        'phi_s_confinement'   : legacy key for the π/2 U6 *drift alert scale*
        'k_phi_hotspot'       : 0.9×π ≈ 2.8274 (curvature hotspot threshold)
    """
    n_nodes = G.number_of_nodes()

    # Legacy key/value retained. U6 constrains two-snapshot mean |ΔΦ_s|;
    # it does not bound the magnitude |Φ_s| in one graph.
    phi_s_alert = U6_STRUCTURAL_POTENTIAL_LIMIT

    # K_φ is bounded by π (wrapped angle difference)
    k_phi_bound = PI

    # Historical composite alert, not a maximum charge-density theorem.
    max_charge = phi_s_alert + k_phi_bound

    # J_φ = mean(sin(Δθ)), bounded by 1
    j_phi_bound = 1.0

    # Legacy J_ΔNFR alert proxy. U2/U6 do not bound pressure spread.
    j_dnfr_alert = 2.0 * phi_s_alert

    # Maximum current magnitude
    max_current = math.sqrt(j_phi_bound**2 + j_dnfr_alert**2)

    # Historical divergence alert derived from the composite current scale.
    avg_degree = 2.0 * G.number_of_edges() / max(n_nodes, 1)
    max_div = 2.0 * max_current

    # Historical residual alert, not a maximum allowed by grammar.
    max_residual = max_charge + max_div

    return ConservationAlertLevels(
        {
            "max_charge_density": max_charge,
            "max_current_magnitude": max_current,
            "max_allowed_residual": max_residual,
            "phi_s_confinement": phi_s_alert,
            "k_phi_hotspot": K_PHI_CANONICAL_THRESHOLD,
            "average_degree": avg_degree,
        }
    )


# ---------------------------------------------------------------------------
# Legacy grammar-named entry point for balance alerts
# ---------------------------------------------------------------------------


def detect_grammar_violations_from_conservation(
    balance: ConservationBalance,
    bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    r"""Report legacy-scaled balance alerts without classifying grammar.

    Residuals and charge drift contain no operator-history evidence and no
    direct phase-admissibility evidence. They therefore cannot classify U2,
    U3, or U6, and cannot validate U1-U6. Numerical discretization, topology
    changes, source terms, or an external pressure law can all produce the
    same values for grammatically different histories.

    The function name and the legacy ``violations_*`` keys remain for API
    compatibility. Those keys now report that no grammar verdict was made.
    New ``alerts_*`` keys contain the actual finite-balance alerts.

    Parameters
    ----------
    balance : ConservationBalance
        Result from verify_conservation_balance.
    bounds : dict[str, float], optional
        Legacy alert levels from :func:`compute_grammar_conservation_bounds`.

    Returns
    -------
    dict with:
        ``violations_detected`` is always ``False`` because grammar is not
        assessed; ``alerts_detected``, ``alert_types``, ``severity`` and
        ``nodes_alerted`` expose the legacy-scaled observations. Scope and
        applicability metadata are included explicitly.
    """
    threshold = U6_STRUCTURAL_POTENTIAL_LIMIT
    if bounds is not None:
        threshold = bounds.get("max_allowed_residual", U6_STRUCTURAL_POTENTIAL_LIMIT)

    alert_types: list[str] = []
    nodes_alerted: list[Any] = []

    for node, res in balance.residual.items():
        if abs(res) > threshold:
            nodes_alerted.append(node)

    if nodes_alerted:
        alert_types.append("balance_residual_above_legacy_alert")

    if balance.charge_drift > U6_STRUCTURAL_POTENTIAL_LIMIT:
        alert_types.append("charge_drift_above_legacy_pi_alert")

    if balance.rms_residual > _BALANCE_RMS_ALERT:
        alert_types.append("balance_rms_above_legacy_alert")

    if balance.max_residual > 2 * threshold:
        alert_types.append("balance_peak_above_legacy_alert")

    severity = min(1.0, balance.rms_residual / max(threshold, 1e-10))

    return {
        # Backward-compatible grammar-shaped fields: no grammar assessment was
        # made, so these must never carry inferred U-rule failures.
        "violations_detected": False,
        "violation_count": 0,
        "violation_types": [],
        "nodes_violating": [],
        # Accurate finite-balance alert surface.
        "alerts_detected": bool(alert_types),
        "alert_count": len(alert_types),
        "alert_types": alert_types,
        "severity": severity,
        "nodes_alerted": nodes_alerted,
        "alert_threshold": threshold,
        # Applicability metadata.
        "diagnostic_scope": "two_snapshot_structural_balance_alerts",
        "grammar_validation_applicable": False,
        "grammar_validated": False,
        "grammar_rules_assessed": (),
        "thresholds_are_proven_bounds": False,
        "u6_drift_requires_phi_s_reference": True,
    }


# ---------------------------------------------------------------------------
# Structural charge (historical Noether-like name)
# ---------------------------------------------------------------------------


def compute_noether_charge(G: Any) -> float:
    r"""Compute the historically named tetrad charge candidate.

    ``Q = Σ_i ρ(i) = Σ_i [Φ_s(i) + K_φ(i)]``.

    The charge Q integrates global (potential) and local (geometric)
    structural information into a single scalar.  Its drift must be measured
    along the actual trajectory; grammar compliance is neither sufficient nor
    inferred from a small drift.

    This tetrad charge candidate is distinct from the EPI-channel
    degree-weighted total ``Σ_i deg(i)·EPI(i)``
    (:func:`tnfr.physics.structural_diffusion.degree_weighted_total`). The
    latter has an exact restricted conservation theorem for fixed symmetric
    random-walk diffusion; this function provides no corresponding theorem for
    the full tetrad dynamics.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    float
        Total structural-charge candidate.
    """
    charge = compute_charge_density(G)
    return sum(charge.values())


def compute_energy_functional(G: Any) -> float:
    r"""Compute the TNFR structural energy functional.

    E = (1/2) Σ_i ℰ(i)

    where ℰ(i) = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR² is the raw
    energy density from :func:`unified.compute_energy_density`
    (CANONICAL SOURCE).

    **Single source of truth**: delegates to
    ``unified.compute_energy_density`` for the per-node quadratic form,
    then applies the ½ normalisation and sums.

    **Consistency contracts**:
        ``E == sum(variational.compute_hamiltonian_density(G).values())``
        ``E == 0.5 * sum(unified.compute_energy_density(G).values())``

    The result is non-negative.  It is a Lyapunov candidate only: U2 validity
    alone does not prove ``dE/dt <= 0`` for an arbitrary pressure law or
    operator sequence.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    float
        Total structural energy.

    See Also
    --------
    unified.compute_energy_density : Raw ℰ(i) per node.
    variational.compute_hamiltonian_density : H(i) = ½·ℰ(i) per node.
    """
    raw = _raw_energy_density(G)
    return 0.5 * sum(raw.values())


def _energy_from_snapshot(snapshot: ConservationSnapshot) -> float:
    r"""E = ½ Σ_i (Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²) from a captured snapshot.

    Snapshot-field counterpart of :func:`compute_energy_functional` (which reads
    the live graph): both evaluate the same canonical quadratic form, so the
    energy-density definition lives in one place rather than being inlined.
    """
    raw = _energy_density_from_fields(
        snapshot.phi_s, snapshot.grad_phi, snapshot.k_phi,
        snapshot.j_phi, snapshot.j_dnfr,
    )
    return 0.5 * sum(raw.values())


# ---------------------------------------------------------------------------
# Sector coupling analysis (the deep physics insight)
# ---------------------------------------------------------------------------


def analyze_sector_coupling(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    dt: float = 1.0,
) -> dict[str, float]:
    r"""Analyze coupling between potential and geometric conservation sectors.

    The measured balance decomposes into two residual sectors:

    **Potential sector** (global, ΔNFR-driven):
        ∂Φ_s/∂t + div(J_ΔNFR) ≈ 0
        - Records pressure-field transport and source mismatch
        - Does not classify the U2 stabilizer/debt rule
        - U6 needs a two-snapshot mean |ΔΦ_s| comparison

    **Geometric sector** (local, phase-driven):
        ∂K_φ/∂t + div(J_φ) ≈ 0
        - Records phase-curvature transport and source mismatch
        - Does not assess U3 edge-phase admissibility
        - The 0.9π hotspot cut is a selected alert inside the exact π wrap

    The cross-correlation summarizes co-variation between the two residual
    channels. It does not establish a causal coupling or derive the complex
    field ``Psi = K_phi + i*J_phi``.

    Parameters
    ----------
    before, after : ConservationSnapshot
        Snapshots before and after an operator sequence.
    dt : float
        Time step.

    Returns
    -------
    dict with:
        'potential_sector_residual' : RMS residual of potential sector
        'geometric_sector_residual' : RMS residual of geometric sector
        'cross_coupling_strength' : Correlation between sector residuals
        'dominant_sector' : 'potential' | 'geometric' | 'balanced'
        'sector_asymmetry' : Ratio of dominant to subdominant residual
    """
    decomp = decompose_conservation_residual(before, after, dt=dt)
    nodes = list(decomp["potential_residual"].keys())

    pot_res = np.array([decomp["potential_residual"][n] for n in nodes])
    geo_res = np.array([decomp["geometric_residual"][n] for n in nodes])

    rms_pot = float(np.sqrt(np.mean(pot_res**2)))
    rms_geo = float(np.sqrt(np.mean(geo_res**2)))

    # Cross-coupling: correlation between sector residuals
    if len(nodes) > 2 and np.std(pot_res) > 1e-15 and np.std(geo_res) > 1e-15:
        cross_corr = float(np.corrcoef(pot_res, geo_res)[0, 1])
    else:
        cross_corr = 0.0

    # Determine dominant sector
    if rms_pot > _SECTOR_IMBALANCE_RATIO * rms_geo:
        dominant = "potential"
    elif rms_geo > _SECTOR_IMBALANCE_RATIO * rms_pot:
        dominant = "geometric"
    else:
        dominant = "balanced"

    asymmetry = max(rms_pot, rms_geo) / (min(rms_pot, rms_geo) + 1e-15)

    return {
        "potential_sector_residual": rms_pot,
        "geometric_sector_residual": rms_geo,
        "cross_coupling_strength": cross_corr,
        "dominant_sector": dominant,
        "sector_asymmetry": asymmetry,
    }


# ---------------------------------------------------------------------------
# Ward-like diagnostics: per-operator finite-step signatures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WardIdentity:
    """Conservation signature of a single operator application.

    This legacy-named object records finite before/after observables for one
    reported operator application. It does not prove a symmetry or validate
    the operator sequence. For operator O_k at step k:

        ΔQ_k/(N Δt_k) + mean_i[(div J_before + div J_after)/2]_i
            = mean_i[S_k(i)]

    Thus ``delta_charge`` is a total structural-charge-candidate change while
    ``mean_source`` is a per-node rate residual. They are distinct quantities.

    Attributes
    ----------
    operator_name : str
        Name of the applied operator (e.g. "AL", "IL", "OZ").
    delta_charge : float
        Total structural-charge-candidate change ΔQ = Q_after - Q_before.
    delta_energy : float
        Energy functional change ΔE = E_after - E_before.
    mean_source : float
        Network-averaged source term ⟨S⟩ (continuity residual).
    conservation_quality : float
        Balance quality for this single step.
    charge_character : str
        Finite-difference label: 'source' (ΔQ > ε), 'sink' (ΔQ < -ε),
        'transport' (|ΔQ| < ε), or legacy 'exact' (both changes within ε).
        The last label means threshold-neutral, not exact conservation.
    energy_character : str
        'dissipative' (ΔE < -ε), 'injective' (ΔE > ε), or 'neutral'.
    """

    operator_name: str
    delta_charge: float
    delta_energy: float
    mean_source: float
    conservation_quality: float
    charge_character: str
    energy_character: str


def compute_ward_identity(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    operator_name: str,
    G_before: Any = None,
    G_after: Any = None,
    dt: float = 1.0,
    threshold: float = 0.01,
) -> WardIdentity:
    r"""Compute a legacy-named Ward diagnostic for one observed step.

    Measures how the supplied snapshots changed structural charge and the
    energy candidate. The labels are finite threshold classifications and
    contain no grammar verdict.

    Parameters
    ----------
    before, after : ConservationSnapshot
        Network state before and after operator application.
    operator_name : str
        Name/glyph of the operator that was applied (e.g. "AL", "IL").
    G_before, G_after : TNFRGraph, optional
        Graph objects for energy computation.  If ``None``, energy change
        is estimated from snapshot fields.
    dt : float
        Time step between snapshots (default 1.0).
    threshold : float
        Minimum |ΔQ| or |ΔE| to classify as non-neutral (default 0.01).

    Returns
    -------
    WardIdentity
    """
    if not isinstance(operator_name, str) or not operator_name.strip():
        raise ValueError("operator_name must be a non-empty string")
    if isinstance(threshold, bool):
        raise TypeError("threshold must be a finite positive real number")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and strictly positive")
    balance = verify_conservation_balance(before, after, dt=dt)

    q_before = balance.total_charge_before
    q_after = balance.total_charge_after
    delta_q = q_after - q_before

    # Energy from graphs if available, else estimate from snapshots
    if G_before is not None and G_after is not None:
        e_before = compute_energy_functional(G_before)
        e_after = compute_energy_functional(G_after)
    else:
        e_before = _energy_from_snapshot(before)
        e_after = _energy_from_snapshot(after)
    delta_e = e_after - e_before

    # Classify charge character
    if abs(delta_q) < threshold and abs(delta_e) < threshold:
        charge_char = "exact"
    elif delta_q > threshold:
        charge_char = "source"
    elif delta_q < -threshold:
        charge_char = "sink"
    else:
        charge_char = "transport"

    # Classify energy character
    if delta_e < -threshold:
        energy_char = "dissipative"
    elif delta_e > threshold:
        energy_char = "injective"
    else:
        energy_char = "neutral"

    return WardIdentity(
        operator_name=operator_name,
        delta_charge=delta_q,
        delta_energy=delta_e,
        mean_source=balance.mean_residual,
        conservation_quality=balance.conservation_quality,
        charge_character=charge_char,
        energy_character=energy_char,
    )


def verify_sequence_ward_identity(
    identities: Sequence[WardIdentity],
    *,
    alert_level: float | None = None,
) -> dict[str, Any]:
    r"""Measure the sequence Ward residual ``Σ_k <S_k>``.

    The legacy ``sequence_conserved`` flag applies a finite-sequence alert
    level. It is retained as an alias for ``aggregate_balance_within_alert``;
    it does not infer grammar validity or prove a conservation law. When no
    alert level is supplied, the historical pi-scaled value is retained for
    compatibility and is explicitly reported as a legacy alert, not a bound.

    Parameters
    ----------
    identities : Sequence[WardIdentity]
        Ordered Ward identities for each operator in the sequence.
    alert_level : float, optional
        Finite aggregate-source alert. Defaults to the historical
        ``(pi/2) / n_steps`` scale.

    Returns
    -------
    dict with:
        'total_source' : float — Σ⟨S_k⟩ (should be ≈ 0)
        'total_charge_change' : float — net ΔQ
        'total_energy_change' : float — net ΔE
        'sequence_conserved' : legacy alias for a threshold comparison
        'operator_summary' : dict[str, int] — count by charge_character
    """
    total_source = sum(w.mean_source for w in identities)
    total_dq = sum(w.delta_charge for w in identities)
    total_de = sum(w.delta_energy for w in identities)

    summary: dict[str, int] = {}
    for w in identities:
        summary[w.charge_character] = summary.get(w.charge_character, 0) + 1

    n_steps = max(len(identities), 1)
    if alert_level is None:
        threshold = U6_STRUCTURAL_POTENTIAL_LIMIT / n_steps
    else:
        if isinstance(alert_level, bool):
            raise TypeError("alert_level must be a finite positive real number")
        threshold = float(alert_level)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("alert_level must be finite and strictly positive")
    within_alert = abs(total_source) < threshold

    return {
        "total_source": total_source,
        "total_charge_change": total_dq,
        "total_energy_change": total_de,
        "sequence_conserved": within_alert,
        "aggregate_balance_within_alert": within_alert,
        "alert_threshold": threshold,
        "thresholds_are_proven_bounds": False,
        "grammar_validation_applicable": False,
        "grammar_validated": False,
        "diagnostic_scope": "finite_sequence_balance_aggregate",
        "operator_summary": summary,
    }


# ---------------------------------------------------------------------------
# Lyapunov stability analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LyapunovResult:
    """Result of Lyapunov stability analysis for an operator step.

    The energy functional E = ½Σ(Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²) is a
    Lyapunov candidate.  This result reports whether one observed finite step
    decreases it; grammar validity alone does not determine that sign.

    Attributes
    ----------
    energy_before : float
        E[G] before the operator application.
    energy_after : float
        E[G] after the operator application.
    energy_derivative : float
        (E_after - E_before) / dt — approximation of dE/dt.
    dissipation : float
        D[G] = max(0, -dE/dt) — structural dissipation rate.
    is_stable : bool
        True when dE/dt ≤ 0 (energy non-increasing).
    is_strongly_stable : bool
        True when dE/dt < -ε (energy strictly decreasing).
    """

    energy_before: float
    energy_after: float
    energy_derivative: float
    dissipation: float
    is_stable: bool
    is_strongly_stable: bool


def compute_lyapunov_derivative(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    dt: float = 1.0,
    stability_threshold: float = 1e-6,
) -> LyapunovResult:
    r"""Compute the Lyapunov derivative dE/dt between two snapshots.

    The structural energy functional:
        E = ½ Σ_i [Φ_s(i)² + |∇φ|(i)² + K_φ(i)² + J_φ(i)² + J_ΔNFR(i)²]

    The structural dissipation readout is ``D[G] = max(0, -dE/dt)``.  A
    non-increasing observation supports stability for that step; this function
    does not prove a general Lyapunov theorem from grammar labels.

    Parameters
    ----------
    before, after : ConservationSnapshot
        Network state before and after operator application.
    dt : float
        Time step (default 1.0).
    stability_threshold : float
        Minimum |dE/dt| to classify as "strongly stable" (default 1e-6).

    Returns
    -------
    LyapunovResult
    """
    e_before = _energy_from_snapshot(before)
    e_after = _energy_from_snapshot(after)

    de_dt = (e_after - e_before) / dt
    dissipation = max(0.0, -de_dt)

    return LyapunovResult(
        energy_before=e_before,
        energy_after=e_after,
        energy_derivative=de_dt,
        dissipation=dissipation,
        is_stable=de_dt <= stability_threshold,
        is_strongly_stable=de_dt < -stability_threshold,
    )


# ---------------------------------------------------------------------------
# Spectral conservation analysis (graph Laplacian decomposition)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralConservation:
    r"""Spectral decomposition of the conservation fields.

    Expands charge density ρ and current divergence in the eigenbasis of the
    symmetric normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2}, whose
    spectrum is that of the canonical TNFR diffusion operator L_rw = I − D⁻¹W
    (the EPI channel of the nodal equation): ρ(i) = Σ_k ρ̂_k ψ_k(i).

    The continuity equation mode-by-mode reads:
        dρ̂_k/dt + λ_k Ĵ_k = Ŝ_k

    Eigenvalue ordering supplies graph scales only. A static snapshot cannot
    infer conservation at low frequency, relaxation at high frequency, U5
    compliance, or the omitted time derivative.

    Attributes
    ----------
    eigenvalues : np.ndarray
        L_sym eigenvalues λ_k (sorted ascending) — the canonical L_rw
        relaxation spectrum.
    rho_spectrum : np.ndarray
        Charge density coefficients ρ̂_k in the eigenbasis.
    div_spectrum : np.ndarray
        Current divergence coefficients in the eigenbasis.
    modal_divergence_magnitude : np.ndarray
        Static activity ``|div_hat_k|`` of the already-computed divergence.
        No second Laplacian factor is applied.
    low_divergence_activity_modes : int
        Number of modes with divergence magnitude at or below its median.
    spectral_gap : float
        λ_1 — gap between zero mode and first non-trivial mode.
    node_order : tuple
        Node order used to build both field vectors and the Laplacian.
    """

    eigenvalues: Any  # np.ndarray
    rho_spectrum: Any  # np.ndarray
    div_spectrum: Any  # np.ndarray
    conservation_by_mode: Any  # np.ndarray
    dominant_conservation_modes: int
    spectral_gap: float
    node_order: tuple[Any, ...] = ()

    @property
    def modal_divergence_magnitude(self) -> Any:
        """Accurate name for the legacy stored per-mode activity field."""
        return self.conservation_by_mode

    @property
    def low_divergence_activity_modes(self) -> int:
        """Accurate name for the legacy stored median-split count."""
        return self.dominant_conservation_modes


def compute_spectral_conservation(
    G: Any,
    snapshot: ConservationSnapshot | None = None,
) -> SpectralConservation:
    r"""Decompose conservation fields in the normalized Laplacian eigenbasis.

    Represents one charge-density and current-divergence snapshot in the
    ``L_sym`` eigenbasis. ``modal_divergence_magnitude=|div_hat_k|`` is a static
    activity readout. The divergence has already been computed in node space,
    so multiplying it by ``lambda_k`` again would apply an unintended second
    graph derivative. The compatibility field ``conservation_by_mode`` exposes
    the same array. With no ``d rho_hat / dt``, neither name is a per-mode
    conservation residual or a U5 assessment.

    Parameters
    ----------
    G : TNFRGraph
        The TNFR network.
    snapshot : ConservationSnapshot, optional
        Pre-computed snapshot.  If ``None``, captured from *G*.

    Returns
    -------
    SpectralConservation
    """
    if snapshot is None:
        snapshot = capture_conservation_snapshot(G)

    # Build the symmetric normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2},
    # whose spectrum is that of the canonical TNFR diffusion operator
    # L_rw = I − D⁻¹W (the EPI channel; see ``structural_diffusion``).  This is
    # The field vectors must use exactly the node order returned with L.
    from .structural_diffusion import symmetric_normalized_laplacian

    nodes, L = symmetric_normalized_laplacian(G)
    n = len(nodes)

    # Eigendecomposition (orthonormal eigenbasis of L_sym)
    eigvals, eigvecs = np.linalg.eigh(L)

    # Project charge density and divergence into eigenbasis
    rho_vec = np.array([snapshot.charge_density[nd] for nd in nodes])
    div_vec = np.array([snapshot.divergence[nd] for nd in nodes])

    rho_hat = eigvecs.T @ rho_vec  # coefficients in eigenbasis
    div_hat = eigvecs.T @ div_vec

    # Static per-mode divergence magnitude. ``div_vec`` is already a graph
    # divergence, so an additional eigenvalue factor would apply L twice.
    divergence_activity = np.abs(div_hat)

    # Median split is descriptive only; it is not a conservation verdict.
    median_res = float(np.median(divergence_activity)) if n > 0 else 0.0
    n_low_activity = int(np.sum(divergence_activity <= median_res + 1e-15))

    # Spectral gap
    sorted_eigs = np.sort(eigvals)
    spectral_gap = float(sorted_eigs[1]) if n > 1 else 0.0

    return SpectralConservation(
        eigenvalues=eigvals,
        rho_spectrum=rho_hat,
        div_spectrum=div_hat,
        conservation_by_mode=divergence_activity,
        dominant_conservation_modes=n_low_activity,
        spectral_gap=spectral_gap,
        node_order=tuple(nodes),
    )


# ---------------------------------------------------------------------------
# Historical conservation-quality scaling fit
# ---------------------------------------------------------------------------


def compute_conservation_scaling(
    topologies: Sequence[tuple[Any, str]],
    dt: float = 0.01,
    n_steps: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    r"""Measure conservation quality scaling with network size.

    Fits the historical finite-sample ansatz:

        q(N) ~ 1 - C/√N

    The returned fit and R² describe only the supplied graph family and
    evolution procedure. They do not prove a continuum limit or exact
    conservation as ``N -> infinity``.

    Parameters
    ----------
    topologies : Sequence[tuple[graph, label]]
        list of (graph, label) pairs at different sizes.
    dt : float
        Integration time step per evolution step.
    n_steps : int
        Number of evolution steps per graph.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with:
        'sizes' : list[int]
        'qualities' : list[float]
        'labels' : list[str]
        'fit_C' : float — estimated C from q(N) ≈ 1 - C/√N
        'fit_R2' : float — goodness of fit
    """
    rng = np.random.default_rng(seed)
    sizes: list[int] = []
    qualities: list[float] = []
    labels: list[str] = []

    for G, label in topologies:
        n = G.number_of_nodes()
        # Ensure canonical attributes
        for nd in G.nodes():
            if "phase" not in G.nodes[nd]:
                G.nodes[nd]["phase"] = rng.uniform(0, 2 * math.pi)
            if "delta_nfr" not in G.nodes[nd]:
                G.nodes[nd]["delta_nfr"] = rng.uniform(-0.5, 0.5)
            if "frequency" not in G.nodes[nd]:
                G.nodes[nd]["frequency"] = rng.uniform(0.1, 1.0)

        tracker = ConservationTracker(G)
        tracker.record(t=0.0)

        # Simple nodal evolution (phase + ΔNFR diffusion)
        for step in range(n_steps):
            for nd in G.nodes():
                nu_f = G.nodes[nd].get("frequency", 1.0)
                dnfr = G.nodes[nd].get("delta_nfr", 0.0)
                G.nodes[nd]["phase"] += dt * nu_f * dnfr * 0.1
                nbrs = list(G.neighbors(nd))
                if nbrs:
                    mean_dnfr = float(
                        np.mean([G.nodes[j].get("delta_nfr", 0.0) for j in nbrs])
                    )
                    G.nodes[nd]["delta_nfr"] += dt * 0.1 * (mean_dnfr - dnfr)
            tracker.record(t=(step + 1) * dt)

        report = tracker.report()
        sizes.append(n)
        qualities.append(report.mean_quality)
        labels.append(label)

    # Fit q(N) ≈ 1 - C/√N  →  (1 - q) ≈ C/√N
    # Linear regression: y = C * x where y = 1-q, x = 1/√N
    x = np.array([1.0 / math.sqrt(s) for s in sizes])
    y = np.array([1.0 - q for q in qualities])

    # Least-squares: C = Σ(x·y) / Σ(x²)
    xx = float(np.sum(x * x))
    xy = float(np.sum(x * y))
    fit_C = xy / xx if xx > 1e-15 else 0.0

    # R² goodness of fit
    y_pred = fit_C * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit_R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return {
        "sizes": sizes,
        "qualities": qualities,
        "labels": labels,
        "fit_C": fit_C,
        "fit_R2": fit_R2,
    }


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "ConservationSnapshot",
    "ConservationBalance",
    "ConservationTimeSeries",
    "ConservationAlertLevels",
    "WardIdentity",
    "LyapunovResult",
    "SpectralConservation",
    # Core computations
    "compute_charge_density",
    "compute_current_divergence",
    "capture_conservation_snapshot",
    "verify_conservation_balance",
    # Tracking
    "ConservationTracker",
    # Analysis
    "decompose_conservation_residual",
    "analyze_sector_coupling",
    "compute_grammar_conservation_bounds",
    "detect_grammar_violations_from_conservation",
    # Conserved quantities
    "compute_noether_charge",
    "compute_energy_functional",
    # Ward identities
    "compute_ward_identity",
    "verify_sequence_ward_identity",
    # Lyapunov stability
    "compute_lyapunov_derivative",
    # Spectral conservation
    "compute_spectral_conservation",
    # Scaling analysis
    "compute_conservation_scaling",
]

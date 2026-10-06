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

These sums use the engine's normalized numeric field convention. With raw
units, pressure has EPI units, the inverse-square potential also carries
inverse distance squared, and phase discrepancies are angles. They cannot be
added without declared reference scales. The neighbor-difference divergence
also supplies no time-rate factor: ``dt`` and the current normalization must
share a declared time convention before this diagnostic can represent a
physical continuity equation. The implementation does not derive those scales.

The exact conservation result owned elsewhere is the degree-weighted EPI total
for fixed symmetric pure diffusion under its stated capacity assumptions.  It
must not be conflated with the tetrad charge reported by this module.

STATUS
======
CANONICAL DIAGNOSTIC INTERFACE.  Residuals, charge drift, and energy change are
observations of the supplied trajectory, not consequences of U1-U6 by label.

References
----------
- Nodal identity and complete-law premises: theory/FUNDAMENTAL_THEORY.md
- Grammar U1-U6: theory/UNIFIED_GRAMMAR_RULES.md
- Structural fields: src/tnfr/physics/canonical.py
- Extended fields: src/tnfr/physics/extended.py
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from fractions import Fraction
from typing import Any, Sequence

from .._exact_time import finite_represented_real
from ..mathematics._exact_weighted import exact_weighted_sum_ratio
from ..mathematics._neighbor_differences import mean_neighbor_difference
from ..mathematics.unified_numerical import np
from ..metrics.common import (
    finite_mean_absolute,
    finite_pearson_correlation,
    finite_population_std,
)

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from ..constants.canonical import (
    K_PHI_CANONICAL_THRESHOLD,
    PI,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)
from ._helpers import finite_real_scalar
from .canonical import compute_phase_curvature, compute_structural_potential
from .extended import compute_dnfr_flux, compute_phase_current
from .unified import (
    _capture_structural_fields,
    _total_charge_from_density,
    _total_energy_from_fields,
)

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
    divergence_phi, divergence_dnfr : dict or None
        Captured applications of that same graph operator to each current.
        Legacy manually constructed snapshots may omit them; their sector
        decomposition is then unavailable, rather than inferred from magnitudes.
    """

    charge_density: dict[Any, float]
    phi_s: dict[Any, float]
    k_phi: dict[Any, float]
    j_phi: dict[Any, float]
    j_dnfr: dict[Any, float]
    grad_phi: dict[Any, float]
    divergence: dict[Any, float]
    divergence_phi: dict[Any, float] | None = None
    divergence_dnfr: dict[Any, float] | None = None


def _observed_scalar(value: Any, name: str) -> float:
    """Admit represented evidence, retaining this module's ValueError API."""
    try:
        return finite_represented_real(value, name)[0]
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _positive_interval(dt: float) -> float:
    value = _observed_scalar(dt, "dt")
    if value <= 0.0:
        raise ValueError("dt must be strictly positive")
    return value


def _observed_secant(before: float, after: float, dt: float, name: str) -> float:
    """Compute a finite represented secant without false zero or range loss.

    Ordinary differences retain their historical rounding. Exceptional
    subtraction/division uses the shared exact difference kernel and rounds
    only the final quotient; a nonzero unrepresentable rate is unavailable.
    """
    before = _observed_scalar(before, name)
    after = _observed_scalar(after, name)
    result = (after - before) / dt
    if math.isfinite(result) and (result != 0.0 or before == after):
        return result
    numerator, denominator = exact_weighted_sum_ratio(
        (1.0,), (after,), center=before, normalize=False
    )
    time_num, time_den = dt.as_integer_ratio()
    return _observed_scalar(
        Fraction(numerator * time_den, denominator * time_num), name
    )


def _snapshot_nodes(snapshot: ConservationSnapshot, fields: Sequence[str]) -> tuple:
    """Admit complete finite maps without fabricating missing observations."""
    if not isinstance(snapshot, ConservationSnapshot):
        raise TypeError("expected a ConservationSnapshot")
    nodes = tuple(snapshot.charge_density)
    if not nodes:
        raise ValueError("conservation observations require nonempty node support")
    support = set(nodes)
    for name in fields:
        values = getattr(snapshot, name)
        if not isinstance(values, Mapping) or set(values) != support:
            raise ValueError(f"snapshot {name} must match the complete node support")
        for node in nodes:
            _observed_scalar(values[node], f"snapshot {name}[{node!r}]")
    return nodes


def _snapshot_pair_nodes(before, after, fields: Sequence[str]) -> tuple:
    nodes = _snapshot_nodes(before, fields)
    if set(nodes) != set(_snapshot_nodes(after, fields)):
        raise ValueError("snapshot comparison requires identical node support")
    return nodes


def _rms(values: Sequence[float]) -> float:
    """Reduce finite residuals without squaring their absolute scale."""
    scale = max(map(abs, values), default=0.0)
    if scale == 0.0:
        return 0.0
    return scale * math.sqrt(
        math.fsum((value / scale) ** 2 for value in values) / len(values)
    )


@dataclass
class ConservationBalance:
    """Result of the continuity equation verification across two snapshots.

    Uses **Crank-Nicolson (trapezoidal)** discretization for O(Δt²)
    accuracy:

    * ``residual[i] = Δρ(i)/Δt + ½[div J_before(i) + div J_after(i)]``
      — a finite-interval balance diagnostic, not a grammar validator.
    * ``mean_residual``, ``max_residual`` — aggregate diagnostics.
    * ``conservation_quality`` — represented ``1/(1+RMS)`` in [0, 1].
      Rounding can produce one for a small nonzero residual; inspect the RMS
      itself when testing represented zero.
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

    This is a normalized graph diagnostic, with no metric-length divisor or
    transport-rate factor. Summing its phase and pressure components assumes
    declared field reference scales, not equality of their raw physical units.

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
    phase, pressure = _current_divergence_components_from_fields(G, j_phi, j_dnfr)
    return {node: phase[node] + pressure[node] for node in phase}


def _current_divergence_components_from_fields(G, j_phi, j_dnfr):
    """Read each linear current channel on the actual graph neighborhood."""
    nodes = tuple(G.nodes())
    phase, pressure = {}, {}
    for i in nodes:
        neighbors = list(G.neighbors(i))
        # Legacy neighbor-minus-center (inward-flux / -L_rw) convention.
        phase[i] = mean_neighbor_difference(j_phi[i], [j_phi[j] for j in neighbors])
        pressure[i] = mean_neighbor_difference(
            j_dnfr[i], [j_dnfr[j] for j in neighbors]
        )
    return phase, pressure


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
    div_phi, div_dnfr = _current_divergence_components_from_fields(G, j_phi, j_dnfr)
    div_j = {node: div_phi[node] + div_dnfr[node] for node in div_phi}

    return ConservationSnapshot(
        charge_density=charge,
        phi_s=phi_s,
        k_phi=k_phi,
        j_phi=j_phi,
        j_dnfr=j_dnfr,
        grad_phi=grad_phi,
        divergence=div_j,
        divergence_phi=div_phi,
        divergence_dnfr=div_dnfr,
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

    Requires nonempty identical node support, complete finite charge/divergence
    maps and a finite positive interval. A support change needs explicit source
    and correspondence accounting outside this observer; it is never filled
    with missing-node zeros. All inputs use a common declared normalization
    and time convention. The
    routine does not convert the raw units of potential, curvature, currents
    or ``dt``; a small numeric residual alone does not validate those units.
    A nonzero rate outside binary64 range raises instead of becoming a zero
    residual. Recoverable intermediate subtraction overflow uses an exact
    represented difference before the final quotient is rounded.

    Parameters
    ----------
    before : ConservationSnapshot
        State before operator application.
    after : ConservationSnapshot
        State after operator application.
    dt : float
        Step in the declared diagnostic time coordinate (default 1.0).

    Returns
    -------
    ConservationBalance
        Finite-interval structural-balance diagnostics.
    """
    dt = _positive_interval(dt)
    nodes = _snapshot_pair_nodes(before, after, ("charge_density", "divergence"))

    delta_rho: dict[Any, float] = {}
    residual: dict[Any, float] = {}
    average_divergence: dict[Any, float] = {}

    for n in nodes:
        rho_before = before.charge_density[n]
        rho_after = after.charge_density[n]
        d_rho = _observed_secant(rho_before, rho_after, dt, "charge rate")
        delta_rho[n] = d_rho

        # Crank-Nicolson: trapezoidal average of divergence at both endpoints
        div_j = mean_neighbor_difference(
            0.0, (before.divergence[n], after.divergence[n])
        )
        average_divergence[n] = div_j
        # Continuity: ∂ρ/∂t + div J = S  →  residual = ∂ρ/∂t + div J
        residual[n] = finite_real_scalar(d_rho + div_j, "balance residual")

    residual_vals = tuple(residual.values())
    mean_res = mean_neighbor_difference(0.0, residual_vals)
    std_res = finite_population_std(residual_vals, name="balance residual")
    max_res = max(map(abs, residual_vals))
    rms_res = _rms(residual_vals)

    # Conservation quality: 1/(1 + RMS) maps [0, ∞) → (0, 1]
    quality = 1.0 / (1.0 + rms_res)

    # Legacy field name. This is only the mean absolute balance residual and
    # has no rule-classification semantics.
    gvi = finite_mean_absolute(residual_vals, name="balance residual")

    # Total charge tracking
    q_before = _total_charge_from_density(before.charge_density)
    q_after = _total_charge_from_density(after.charge_density)
    charge_drift = finite_real_scalar(abs(q_after - q_before), "charge drift")

    return ConservationBalance(
        residual=residual,
        delta_rho=delta_rho,
        divergence_after=average_divergence,
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

    def record(
        self, t: float = 0.0, *, _require_energy: bool = False
    ) -> ConservationSnapshot:
        """Capture current state and compute balance against previous snapshot.

        Parameters
        ----------
        t : float
            Finite timestamp, strictly greater than the previous timestamp.
            Equal timestamps do not imply an invented unit-duration step.

        Returns
        -------
        ConservationSnapshot
            A detached copy of the captured snapshot. Editing it cannot alter
            the tracker's retained evidence.

        Rejected samples leave the retained snapshots and series unchanged.
        The private energy flag lets the SDK admit its combined balance/energy
        report before commit; pure balance tracking does not require energy.
        """
        t = _observed_scalar(t, "snapshot timestamp")
        if self._snapshots:
            t_prev, snap_prev = self._snapshots[-1]
            dt = _positive_interval(t - t_prev)
        snap = capture_conservation_snapshot(self._G)
        _snapshot_nodes(snap, ("charge_density", "divergence"))

        if self._snapshots:
            balance = verify_conservation_balance(snap_prev, snap, dt=dt)
            if _require_energy:
                compute_lyapunov_derivative(snap_prev, snap, dt=dt)
            # Commit only after timestamp, capture and interval admission pass.
            self._snapshots.append((t, snap))
            self._series.times.append(t)
            self._series.total_charge.append(balance.total_charge_after)
            self._series.mean_residuals.append(balance.mean_residual)
            self._series.rms_residuals.append(balance.rms_residual)
            self._series.conservation_quality.append(balance.conservation_quality)
            self._series.grammar_violation_index.append(balance.grammar_violation_index)
            self._series.charge_drift.append(balance.charge_drift)
        else:
            # First snapshot — record initial charge only
            if _require_energy:
                _energy_from_snapshot(snap)
            q_total = _total_charge_from_density(snap.charge_density)
            self._snapshots.append((t, snap))
            self._series.times.append(t)
            self._series.total_charge.append(q_total)
            self._series.mean_residuals.append(0.0)
            self._series.rms_residuals.append(0.0)
            self._series.conservation_quality.append(1.0)
            self._series.grammar_violation_index.append(0.0)
            self._series.charge_drift.append(0.0)

        # Copy maps without copying node identifiers: labels may use object
        # identity, and they must still address the caller's original nodes.
        return replace(
            snap,
            **{
                descriptor.name: dict(values)
                for descriptor in dataclass_fields(snap)
                if (values := getattr(snap, descriptor.name)) is not None
            },
        )

    def report(self) -> ConservationTimeSeries:
        """Return detached accumulated diagnostics, not editable retained evidence."""
        return deepcopy(self._series)

    @property
    def latest_balance(self) -> ConservationBalance | None:
        """Return the most recent balance check, or None."""
        if len(self._snapshots) < 2:
            return None
        t_prev, snap_prev = self._snapshots[-2]
        t_curr, snap_curr = self._snapshots[-1]
        dt = t_curr - t_prev
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
    Its sector components are captured independently on the graph. Local
    current magnitudes cannot recover their divergences or even their signs.
    Snapshots missing this component evidence must be recaptured.

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
    dt = _positive_interval(dt)
    if any(
        getattr(snapshot, field) is None
        for snapshot in (before, after)
        for field in ("divergence_phi", "divergence_dnfr")
    ):
        raise ValueError("sector divergences are unavailable; recapture both snapshots")
    nodes = _snapshot_pair_nodes(
        before,
        after,
        ("phi_s", "k_phi", "divergence_phi", "divergence_dnfr"),
    )

    phi_s_drift: dict[Any, float] = {}
    k_phi_drift: dict[Any, float] = {}
    j_phi_div: dict[Any, float] = {}
    j_dnfr_div: dict[Any, float] = {}
    potential_residual: dict[Any, float] = {}
    geometric_residual: dict[Any, float] = {}

    for n in nodes:
        # Rate of change of each charge component
        d_phi_s = _observed_secant(
            before.phi_s[n], after.phi_s[n], dt, "potential rate"
        )
        d_k_phi = _observed_secant(
            before.k_phi[n], after.k_phi[n], dt, "curvature rate"
        )
        phi_s_drift[n] = d_phi_s
        k_phi_drift[n] = d_k_phi

        j_phi_div[n] = mean_neighbor_difference(
            0.0, (before.divergence_phi[n], after.divergence_phi[n])
        )
        j_dnfr_div[n] = mean_neighbor_difference(
            0.0, (before.divergence_dnfr[n], after.divergence_dnfr[n])
        )
        potential_residual[n] = finite_real_scalar(
            d_phi_s + j_dnfr_div[n], "potential residual"
        )
        geometric_residual[n] = finite_real_scalar(
            d_k_phi + j_phi_div[n], "geometric residual"
        )

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
    threshold = _observed_scalar(threshold, "max_allowed_residual")
    if threshold <= 0.0:
        raise ValueError("max_allowed_residual must be strictly positive")

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
    return _total_charge_from_density(charge)


def compute_energy_functional(G: Any) -> float:
    r"""Compute the TNFR structural energy functional.

    E = (1/2) Σ_i ℰ(i)

    where ℰ(i) = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR² is the raw
    energy density defined by :func:`unified.compute_energy_density`.

    **Single source of truth**: delegates to
    the shared normalized-total helper in ``unified`` for the same quadratic
    form and ½ normalization. It avoids materializing unrepresentable squares
    when the final total remains representable.

    **Mathematical normalization**:
        ``E = sum(variational.compute_hamiltonian_density(G).values())``
        ``E = 0.5 * sum(unified.compute_energy_density(G).values())``

    These equalities identify the real quadratic form, not bitwise equality
    after independent per-node rounding. Raw densities can underflow/overflow
    even when the normalized network total is representable.

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
    fields = _capture_structural_fields(G)
    return _total_energy_from_fields(
        fields.phi_s, fields.grad_phi, fields.k_phi, fields.j_phi, fields.j_dnfr
    )


def _energy_from_snapshot(snapshot: ConservationSnapshot) -> float:
    r"""E = ½ Σ_i (Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²) from a captured snapshot.

    Snapshot-field counterpart of :func:`compute_energy_functional` (which reads
    the live graph): both evaluate the same canonical quadratic form, so the
    energy-density definition lives in one place rather than being inlined.
    """
    _snapshot_nodes(snapshot, ("phi_s", "grad_phi", "k_phi", "j_phi", "j_dnfr"))
    return _total_energy_from_fields(
        snapshot.phi_s,
        snapshot.grad_phi,
        snapshot.k_phi,
        snapshot.j_phi,
        snapshot.j_dnfr,
    )


# ---------------------------------------------------------------------------
# Sector coupling analysis (the deep physics insight)
# ---------------------------------------------------------------------------


def analyze_sector_coupling(
    before: ConservationSnapshot,
    after: ConservationSnapshot,
    dt: float = 1.0,
) -> dict[str, Any]:
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
    field ``Psi = K_phi + i*J_phi``. The legacy three-sample and 1e-15
    dispersion cuts are retained as observation policies, not physical bounds.
    An unavailable correlation keeps its compatibility zero but is explicitly
    tagged; a nonrepresentable asymmetry ratio is returned as None.

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

    pot_res = [float(decomp["potential_residual"][n]) for n in nodes]
    geo_res = [float(decomp["geometric_residual"][n]) for n in nodes]

    rms_pot = _rms(pot_res)
    rms_geo = _rms(geo_res)

    # Cross-coupling: correlation between sector residuals
    if len(nodes) <= 2:
        correlation_status = "insufficient_samples"
        cross_corr = None
    elif (
        finite_population_std(pot_res) <= 1e-15
        or finite_population_std(geo_res) <= 1e-15
    ):
        correlation_status = "below_legacy_dispersion_cut"
        cross_corr = None
    else:
        cross_corr = finite_pearson_correlation(pot_res, geo_res)
        correlation_status = (
            "available" if cross_corr is not None else "constant_sample"
        )

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
        "cross_coupling_strength": cross_corr if cross_corr is not None else 0.0,
        "cross_coupling_available": cross_corr is not None,
        "cross_coupling_status": correlation_status,
        "dominant_sector": dominant,
        "sector_asymmetry": asymmetry if math.isfinite(asymmetry) else None,
        "sector_asymmetry_available": math.isfinite(asymmetry),
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

    Measures structural-charge change from the supplied snapshots. Energy
    also comes from those snapshots unless both optional graphs are supplied.
    The labels are finite threshold classifications and contain no grammar
    verdict.

    Parameters
    ----------
    before, after : ConservationSnapshot
        Network state before and after operator application.
    operator_name : str
        Name/glyph of the operator that was applied (e.g. "AL", "IL").
    G_before, G_after : TNFRGraph, optional
        If both are supplied, they provide the energy endpoints. Correspondence
        of those graph states to the charge snapshots is the caller's unverified
        responsibility; neither support nor field equality is checked. Supplying
        only one graph retains the legacy snapshot-energy fallback. Snapshot
        energy evaluates the same declared quadratic form, not an estimate of
        an independently supplied graph.
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
    if isinstance(threshold, (bool, np.bool_)):
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
    total_source = finite_real_scalar(
        math.fsum(finite_real_scalar(w.mean_source, "mean source") for w in identities),
        "total source",
    )
    total_dq = finite_real_scalar(
        math.fsum(
            finite_real_scalar(w.delta_charge, "charge change") for w in identities
        ),
        "total charge change",
    )
    total_de = finite_real_scalar(
        math.fsum(
            finite_real_scalar(w.delta_energy, "energy change") for w in identities
        ),
        "total energy change",
    )

    summary: dict[str, int] = {}
    for w in identities:
        summary[w.charge_character] = summary.get(w.charge_character, 0) + 1

    n_steps = max(len(identities), 1)
    if alert_level is None:
        threshold = U6_STRUCTURAL_POTENTIAL_LIMIT / n_steps
    else:
        if isinstance(alert_level, (bool, np.bool_)):
            raise TypeError("alert_level must be a finite positive real number")
        threshold = float(alert_level)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("alert_level must be finite and strictly positive")
    within_alert = bool(identities) and abs(total_source) < threshold

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
        "sample_available": bool(identities),
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
        Compatibility alert: dE/dt <= the configured nonnegative tolerance.
        Use ``energy_nonincreasing`` for the captured endpoint-energy comparison.
    is_strongly_stable : bool
        True when dE/dt < -ε (energy strictly decreasing).
    """

    energy_before: float
    energy_after: float
    energy_derivative: float
    dissipation: float
    is_stable: bool
    is_strongly_stable: bool

    @property
    def energy_nonincreasing(self) -> bool:
        """Compare the captured endpoint energies, independently of rate rounding."""
        return self.energy_after <= self.energy_before


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
        Nonnegative numerical allowance for ``is_stable`` and the minimum
        decrease for ``is_strongly_stable`` (default 1e-6). It is not a theorem.

    Returns
    -------
    LyapunovResult
    """
    dt = _positive_interval(dt)
    stability_threshold = _observed_scalar(stability_threshold, "stability_threshold")
    if stability_threshold < 0.0:
        raise ValueError("stability_threshold must be nonnegative")
    _snapshot_pair_nodes(before, after, ("charge_density",))
    e_before = _energy_from_snapshot(before)
    e_after = _energy_from_snapshot(after)

    de_dt = _observed_secant(e_before, e_after, dt, "energy derivative")
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
]

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

from ..constants.canonical import K_PHI_CANONICAL_THRESHOLD, PI, U6_STRUCTURAL_POTENTIAL_LIMIT
from .canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from .extended import compute_dnfr_flux, compute_phase_current
from .unified import _capture_structural_fields, _energy_density_from_fields
from .unified import compute_energy_density as _raw_energy_density

# ---------------------------------------------------------------------------
# Conservation diagnostic thresholds
# ---------------------------------------------------------------------------
_BALANCE_RMS_ALERT = 1.0
_SECTOR_IMBALANCE_RATIO = 1.5

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConservationSnapshot:
    """Single-time snapshot of all conserved quantities at every node.

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
    * ``conservation_quality`` — scalar in [0, 1]; 1 = perfect conservation.
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
        """Return the legacy finite-series quality classification."""
        if not self.conservation_quality:
            return False
        return float(np.mean(self.conservation_quality)) >= 0.9

    @property
    def mean_quality(self) -> float:
        """Average conservation quality across all recorded steps."""
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
    divergence at node i is approximated by the net outward flux:

        div J(i) = (1/|N(i)|) Σ_{j∈N(i)} [
            (J_φ(j) - J_φ(i)) + (J_ΔNFR(j) - J_ΔNFR(i))
        ]

    This is the discrete Laplacian applied to each current component,
    consistent with the graph-theoretic divergence used in the Φ_s and
    K_φ definitions.

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
        # Divergence = mean outward flux of both current components
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
    """Capture all conserved quantities at the current instant.

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
        Comprehensive diagnostics of the conservation law.
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

    # Grammar violation index: proportional to mean |residual|
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
    """Track conservation law compliance across a full operator sequence.

    Usage
    -----
    >>> tracker = ConservationTracker(G)
    >>> tracker.record(t=0.0)           # initial snapshot
    >>> Emission()(G, node)
    >>> tracker.record(t=1.0)           # after operator
    >>> Coherence()(G, node)
    >>> tracker.record(t=2.0)
    >>> report = tracker.report()
    >>> print(f"Conserved: {report.is_conserved}")
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
            self._series.grammar_violation_index.append(balance.grammar_violation_index)
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

    This function separates the two contributions to identify whether
    the residual comes from potential drift (global, grammar-U6 related)
    or curvature drift (local, phase dynamics related).

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
# Theoretical bounds from grammar constraints
# ---------------------------------------------------------------------------


def compute_grammar_conservation_bounds(G: Any) -> dict[str, float]:
    r"""Compute legacy grammar-scaled diagnostic alert levels.

    The legacy alert construction combines:

    - U2's stabilizer/debt role as qualitative context;
    - the U6 drift policy ``ΔPhi_s < pi/2`` as a numeric scale;
    - the exact trigonometric bound ``|J_phi| <= 1``.

    These values combine configured U3/U6 scales into monitoring thresholds.
    They are not proved upper bounds on the residual of every grammar-valid
    trajectory, so callers must not use them as a conservation certificate.

    Returns
    -------
    dict[str, float]
        'max_charge_density'  : legacy policy-scaled alert level for |ρ|
        'max_current_magnitude' : legacy alert level for |J|
        'max_allowed_residual' : legacy alert level for |Δρ/Δt + div J|
        'phi_s_confinement'   : π/2 (the U6 confinement bound)
        'k_phi_hotspot'       : 0.9×π ≈ 2.8274 (curvature hotspot threshold)
    """
    n_nodes = G.number_of_nodes()

    # U6: |Φ_s| < π/2
    phi_s_bound = U6_STRUCTURAL_POTENTIAL_LIMIT

    # K_φ is bounded by π (wrapped angle difference)
    k_phi_bound = PI

    # Maximum charge density
    max_charge = phi_s_bound + k_phi_bound

    # J_φ = mean(sin(Δθ)), bounded by 1
    j_phi_bound = 1.0

    # J_ΔNFR = mean(ΔNFR_j - ΔNFR_i), bounded by max|ΔNFR| spread
    # Legacy alert proxy. U2/U6 alone do not bound pressure spread.
    j_dnfr_bound = 2.0 * phi_s_bound

    # Maximum current magnitude
    max_current = math.sqrt(j_phi_bound**2 + j_dnfr_bound**2)

    # Maximum divergence scales with max_current / connectivity
    avg_degree = 2.0 * G.number_of_edges() / max(n_nodes, 1)
    max_div = 2.0 * max_current  # upper bound on discrete divergence

    # Maximum allowed residual (charge rate + divergence)
    max_residual = max_charge + max_div

    return {
        "max_charge_density": max_charge,
        "max_current_magnitude": max_current,
        "max_allowed_residual": max_residual,
        "phi_s_confinement": phi_s_bound,
        "k_phi_hotspot": K_PHI_CANONICAL_THRESHOLD,  # 0.9×π ≈ 2.8274 (canonical curvature hotspot threshold)
        "average_degree": avg_degree,
    }


# ---------------------------------------------------------------------------
# Grammar violation detection via conservation analysis
# ---------------------------------------------------------------------------


def detect_grammar_violations_from_conservation(
    balance: ConservationBalance,
    bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    r"""Flag residual patterns associated with possible grammar violations.

    High residuals are heuristic alerts, not a grammar validator: numerical
    discretization, topology changes, or an external pressure law can produce
    them even for a valid operator history.  Validate U1-U6 independently.

    Parameters
    ----------
    balance : ConservationBalance
        Result from verify_conservation_balance.
    bounds : dict[str, float], optional
        Bounds from compute_grammar_conservation_bounds.

    Returns
    -------
    dict with:
        'violations_detected' : bool
        'violation_count' : int
        'violation_types' : list[str]
        'severity' : float  (0 = none, 1 = extreme)
        'nodes_violating' : list  (nodes with |residual| above threshold)
    """
    threshold = U6_STRUCTURAL_POTENTIAL_LIMIT  # U6 structural-potential bound (π/2)
    if bounds is not None:
        threshold = bounds.get("max_allowed_residual", U6_STRUCTURAL_POTENTIAL_LIMIT)

    violation_types: list[str] = []
    nodes_violating: list[Any] = []

    for node, res in balance.residual.items():
        if abs(res) > threshold:
            nodes_violating.append(node)

    # Classify violation types
    if balance.charge_drift > U6_STRUCTURAL_POTENTIAL_LIMIT:
        violation_types.append("U6_confinement_breach")

    if balance.rms_residual > _BALANCE_RMS_ALERT:
        violation_types.append("U2_convergence_failure")

    if balance.max_residual > 2 * threshold:
        violation_types.append("U3_phase_incompatibility")

    severity = min(1.0, balance.rms_residual / max(threshold, 1e-10))

    return {
        "violations_detected": len(violation_types) > 0,
        "violation_count": len(violation_types),
        "violation_types": violation_types,
        "severity": severity,
        "nodes_violating": nodes_violating,
    }


# ---------------------------------------------------------------------------
# Noether charge: total conserved quantity
# ---------------------------------------------------------------------------


def compute_noether_charge(G: Any) -> float:
    r"""Compute the total Noether charge Q = Σ_i ρ(i) = Σ_i [Φ_s(i) + K_φ(i)].

    The charge Q integrates global (potential) and local (geometric)
    structural information into a single scalar.  Its drift must be measured
    along the actual trajectory; grammar compliance is neither sufficient nor
    inferred from a small drift.

    This **tetrad** charge is **distinct** from the EPI-channel degree-weighted
    total Σ_i deg(i)·EPI(i)
    (:func:`tnfr.physics.structural_diffusion.degree_weighted_total`), the
    conserved quantity of the random-walk diffusion: TNFR carries two distinct
    conservation laws, on the tetrad fields and on the EPI field respectively
    (see STRUCTURAL_CONSERVATION_THEOREM §8.7).

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    float
        Total structural Noether charge.
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

    The full conservation law decomposes into TWO coupled sectors:

    **Potential sector** (global, ΔNFR-driven):
        ∂Φ_s/∂t + div(J_ΔNFR) ≈ 0
        - Conserves when ΔNFR redistributes without creation/destruction
        - Violated by unconstrained destabilizers (grammar U2)
        - Monitored by grammar U6 (|Φ_s| < π/2)

    **Geometric sector** (local, phase-driven):
        ∂K_φ/∂t + div(J_φ) ≈ 0
        - Conserves when phase curvature transports without source terms
        - Violated by phase-incompatible operations (grammar U3)
        - Monitored by curvature hotspot detection (|K_φ| < 2.8274)

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
# Ward identities: per-operator conservation signatures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WardIdentity:
    """Conservation signature of a single operator application.

    A Ward identity constrains the expectation value of observables between
    operator applications.  For operator O_k at step k:

        ⟨Δρ⟩_k + ⟨div J⟩_k = ⟨S_k⟩

    Attributes
    ----------
    operator_name : str
        Name of the applied operator (e.g. "AL", "IL", "OZ").
    delta_charge : float
        Total Noether charge change ΔQ = Q_after - Q_before.
    delta_energy : float
        Energy functional change ΔE = E_after - E_before.
    mean_source : float
        Network-averaged source term ⟨S⟩ (continuity residual).
    conservation_quality : float
        Balance quality for this single step.
    charge_character : str
        Classification: 'source' (ΔQ > ε), 'sink' (ΔQ < -ε),
        'transport' (|ΔQ| < ε), or 'exact' (ΔQ ≈ 0 and ΔE ≈ 0).
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
    r"""Compute the Ward identity for a single operator application.

    Measures how the operator changed conserved quantities and classifies
    its conservation character.

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
) -> dict[str, Any]:
    r"""Measure the sequence Ward residual ``Σ_k <S_k>``.

    The legacy ``sequence_conserved`` flag applies a configured finite-sequence
    threshold.  It does not infer grammar validity or prove that every valid
    sequence has a vanishing source.

    Parameters
    ----------
    identities : Sequence[WardIdentity]
        Ordered Ward identities for each operator in the sequence.

    Returns
    -------
    dict with:
        'total_source' : float — Σ⟨S_k⟩ (should be ≈ 0)
        'total_charge_change' : float — net ΔQ
        'total_energy_change' : float — net ΔE
        'sequence_conserved' : bool — True if |total_source| < threshold
        'operator_summary' : dict[str, int] — count by charge_character
    """
    total_source = sum(w.mean_source for w in identities)
    total_dq = sum(w.delta_charge for w in identities)
    total_de = sum(w.delta_energy for w in identities)

    summary: dict[str, int] = {}
    for w in identities:
        summary[w.charge_character] = summary.get(w.charge_character, 0) + 1

    n_steps = max(len(identities), 1)
    threshold = U6_STRUCTURAL_POTENTIAL_LIMIT / n_steps  # Scale threshold with sequence length

    return {
        "total_source": total_source,
        "total_charge_change": total_dq,
        "total_energy_change": total_de,
        "sequence_conserved": abs(total_source) < threshold,
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

    Low-frequency modes (small λ_k) → conservation regime.
    High-frequency modes (large λ_k) → rapid relaxation.

    Attributes
    ----------
    eigenvalues : np.ndarray
        L_sym eigenvalues λ_k (sorted ascending) — the canonical L_rw
        relaxation spectrum.
    rho_spectrum : np.ndarray
        Charge density coefficients ρ̂_k in the eigenbasis.
    div_spectrum : np.ndarray
        Current divergence coefficients in the eigenbasis.
    conservation_by_mode : np.ndarray
        Per-mode residual |dρ̂_k/dt + λ_k Ĵ_k| (lower = better conservation).
    dominant_conservation_modes : int
        Number of modes with residual below median.
    spectral_gap : float
        λ_1 — gap between zero mode and first non-trivial mode.
    """

    eigenvalues: Any  # np.ndarray
    rho_spectrum: Any  # np.ndarray
    div_spectrum: Any  # np.ndarray
    conservation_by_mode: Any  # np.ndarray
    dominant_conservation_modes: int
    spectral_gap: float


def compute_spectral_conservation(
    G: Any,
    snapshot: ConservationSnapshot | None = None,
) -> SpectralConservation:
    r"""Decompose conservation fields in the normalized Laplacian eigenbasis.

    Connects TNFR conservation to spectral graph theory.  The L_sym
    eigenvalues determine at which structural scales conservation holds
    most precisely:

    - Global modes (k = 0, 1): total charge Q is most conserved
    - Mesoscale modes: sector-level conservation with cross-coupling
    - Local modes (k → N): rapid equilibration, sources/sinks active

    This spectral hierarchy mirrors the U5 multi-scale coherence principle.

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

    nodes = sorted(snapshot.charge_density.keys())
    n = len(nodes)

    # Build the symmetric normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2},
    # whose spectrum is that of the canonical TNFR diffusion operator
    # L_rw = I − D⁻¹W (the EPI channel; see ``structural_diffusion``).  This is
    # consistent with the §9.1 normalized divergence (∇·J = L_rw·J).
    from .structural_diffusion import symmetric_normalized_laplacian

    _, L = symmetric_normalized_laplacian(G)

    # Eigendecomposition (orthonormal eigenbasis of L_sym)
    eigvals, eigvecs = np.linalg.eigh(L)

    # Project charge density and divergence into eigenbasis
    rho_vec = np.array([snapshot.charge_density[nd] for nd in nodes])
    div_vec = np.array([snapshot.divergence[nd] for nd in nodes])

    rho_hat = eigvecs.T @ rho_vec  # coefficients in eigenbasis
    div_hat = eigvecs.T @ div_vec

    # Per-mode "conservation residual": for static snapshot,
    # this is |λ_k · Ĵ_k| (transport rate per mode)
    conservation_modes = np.abs(eigvals * div_hat)

    # Number of well-conserved modes (below median residual)
    median_res = float(np.median(conservation_modes)) if n > 0 else 0.0
    n_conserved = int(np.sum(conservation_modes <= median_res + 1e-15))

    # Spectral gap
    sorted_eigs = np.sort(eigvals)
    spectral_gap = float(sorted_eigs[1]) if n > 1 else 0.0

    return SpectralConservation(
        eigenvalues=eigvals,
        rho_spectrum=rho_hat,
        div_spectrum=div_hat,
        conservation_by_mode=conservation_modes,
        dominant_conservation_modes=n_conserved,
        spectral_gap=spectral_gap,
    )


# ---------------------------------------------------------------------------
# Conservation scaling: q(N) ~ 1 - C/√N  verification
# ---------------------------------------------------------------------------


def compute_conservation_scaling(
    topologies: Sequence[tuple[Any, str]],
    dt: float = 0.01,
    n_steps: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    r"""Measure conservation quality scaling with network size.

    Verifies the theoretical prediction:

        q(N) ~ 1 - C/√N

    where C is a topology-dependent constant.  In the continuum limit
    (N → ∞), q → 1 (exact conservation).

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

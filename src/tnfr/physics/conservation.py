"""TNFR Structural Conservation Laws — Noether-like Theorems from the Nodal Equation.

This module derives and verifies the conservation laws that emerge from the
TNFR nodal equation ∂EPI/∂t = νf · ΔNFR(t) under unified grammar constraints
U1-U6.

MAIN RESULT (Structural Continuity Theorem):
=============================================
Let G be a TNFR network evolving under the nodal equation with grammar
constraints.  Define:

    ρ(i, t)  = Φ_s(i, t) + K_φ(i, t)       [structural charge density]
    J(i, t)  = (J_φ(i, t), J_ΔNFR(i, t))    [structural current vector]

Then the **discrete structural continuity equation** holds:

    Δρ(i)/Δt + div J(i) ≈ S(i)

where S(i) is a *source term* that is bounded and small when:
  (a) Grammar U2 is satisfied (convergence: stabilizers balance destabilizers)
  (b) Grammar U6 is satisfied (confinement: |Φ_s| < φ)

The bound ||S||_{ℓ²} ≤ C/√N (§4.5 of the theory document) implies
  S → 0 in the continuum limit N → ∞.  On finite graphs S is empirically
  small (charge drift < 0.03%) but does not vanish exactly.

This is the TNFR analogue of the Noether theorem:
    Grammar symmetry (U-rules) ⟹ Approximate structural conservation law.

DERIVATION
==========
The conservation law derives from the nodal equation ∂EPI/∂t = νf·ΔNFR(t)
under grammar constraints U1–U6. The complete formal proof, including
explicit operator norm bounds and scaling analysis, is in:

    theory/STRUCTURAL_CONSERVATION_THEOREM.md  §4 (Derivation of the Continuity Equation)

Key results from the proof:

- §4.5 Step 1: U2+U6 guarantee M_U2 := sup_t Σ|∂ΔNFR_j/∂t| < ∞
- §4.5 Step 2: U3 guarantees |∂K_φ/∂t| ≤ 2·νf_max =: M_U3
- §4.5 Step 3: Source S = R_pot + R_geo (potential + geometric residuals)
- §4.5 Step 4: |R_pot| ~ O(1/D), |R_geo| ~ O(Δφ³_max) under grammar
- §4.5 Step 5: ||S||_rms bounded independently of N → q(N) ~ 1 - C/√N
- §4.5 Step 6: S ≠ 0 detects and classifies grammar violations (U2/U3/U6)

STATUS
======
CANONICAL — Derived from the nodal equation under grammar constraints.
The continuity equation is analytically derived; the bound on the source
term relies on specific operator-norm constants from the TNFR implementation.
Numerical validation: charge drift < 0.03% across tested topologies and seeds.

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

from ..constants.canonical import K_PHI_CANONICAL_THRESHOLD, PHI, PI
from .canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
)
from .extended import compute_phase_current, compute_dnfr_flux
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
      — should be ≈ 0 when grammar is satisfied.
    * ``mean_residual``, ``max_residual`` — aggregate diagnostics.
    * ``conservation_quality`` — scalar in [0, 1]; 1 = perfect conservation.
    * ``grammar_violation_index`` — ∝ |mean_residual|;  0 = no violation.
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
        """True when mean conservation quality ≥ 0.9 across all steps."""
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

    This is the conserved "charge" of TNFR structural dynamics:
    - Φ_s captures global structural potential (long-range coupling)
    - K_φ captures local geometric curvature (short-range confinement)

    Their sum is the natural conserved density because the nodal equation
    couples global (ΔNFR distribution → Φ_s) and local (phase dynamics → K_φ)
    degrees of freedom.

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
    nodes = list(G.nodes())
    return {n: phi_s.get(n, 0.0) + k_phi.get(n, 0.0) for n in nodes}

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
    nodes = list(G.nodes())

    divergence: dict[Any, float] = {}
    for i in nodes:
        neighbors = list(G.neighbors(i))
        if not neighbors:
            divergence[i] = 0.0
            continue

        deg = len(neighbors)
        # Divergence = mean outward flux of both current components
        div_j_phi = sum(j_phi.get(j, 0.0) - j_phi.get(i, 0.0)
                        for j in neighbors) / deg
        div_j_dnfr = sum(j_dnfr.get(j, 0.0) - j_dnfr.get(i, 0.0)
                         for j in neighbors) / deg
        divergence[i] = div_j_phi + div_j_dnfr

    return divergence

# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------

def capture_conservation_snapshot(G: Any) -> ConservationSnapshot:
    """Capture all conserved quantities at the current instant.

    This is a *read-only* operation that never mutates EPI.

    Parameters
    ----------
    G : TNFRGraph
        Network in its current state.

    Returns
    -------
    ConservationSnapshot
    """
    phi_s = compute_structural_potential(G)
    k_phi = compute_phase_curvature(G)
    grad_phi = compute_phase_gradient(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)

    nodes = list(G.nodes())
    charge = {n: phi_s.get(n, 0.0) + k_phi.get(n, 0.0) for n in nodes}
    div_j = compute_current_divergence(G)

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

    This gives O(Δt²) accuracy, compared to the O(Δt) of a
    right-endpoint scheme.  A small residual means the operator
    sequence conserved structural charge;  large residuals indicate
    grammar violations acting as sources.

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
        div_j = 0.5 * (
            before.divergence.get(n, 0.0) + after.divergence.get(n, 0.0)
        )
        # Continuity: ∂ρ/∂t + div J = S  →  residual = ∂ρ/∂t + div J
        residual[n] = d_rho + div_j

    residual_vals = np.array(list(residual.values()))

    mean_res = float(np.mean(residual_vals)) if len(residual_vals) > 0 else 0.0
    std_res = float(np.std(residual_vals)) if len(residual_vals) > 0 else 0.0
    max_res = float(np.max(np.abs(residual_vals))) if len(residual_vals) > 0 else 0.0
    rms_res = float(np.sqrt(np.mean(residual_vals ** 2))) if len(residual_vals) > 0 else 0.0

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
        divergence_after={n: 0.5 * (before.divergence.get(n, 0.0)
                              + after.divergence.get(n, 0.0))
                        for n in nodes},
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
            self._series.conservation_quality.append(
                balance.conservation_quality
            )
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
        div_j = 0.5 * (
            before.divergence.get(n, 0.0) + after.divergence.get(n, 0.0)
        )
        # Approximate split: use field magnitudes as proxy (averaged)
        j_phi_n = 0.5 * (
            before.j_phi.get(n, 0.0) + after.j_phi.get(n, 0.0)
        )
        j_dnfr_n = 0.5 * (
            before.j_dnfr.get(n, 0.0) + after.j_dnfr.get(n, 0.0)
        )
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
        'phi_s_drift': phi_s_drift,
        'k_phi_drift': k_phi_drift,
        'j_phi_div': j_phi_div,
        'j_dnfr_div': j_dnfr_div,
        'potential_residual': potential_residual,
        'geometric_residual': geometric_residual,
    }

# ---------------------------------------------------------------------------
# Theoretical bounds from grammar constraints
# ---------------------------------------------------------------------------

def compute_grammar_conservation_bounds(G: Any) -> dict[str, float]:
    r"""Compute theoretical upper bounds on conservation residual.

    From TNFR grammar constraints:

    - U2 (convergence): ∫|νf·ΔNFR| dt < ∞  ⟹  |ΔΦ_s/Δt| bounded
    - U6 (confinement): |Φ_s| < φ ≈ 1.618   ⟹  |ρ| < φ + π ≈ 4.76
    - U3 (coupling):    |Δθ| < Δθ_max        ⟹  |J_φ| ≤ 1

    These bounds predict the maximum allowed residual for a grammar-
    compliant sequence.

    Returns
    -------
    dict[str, float]
        'max_charge_density'  : theoretical upper bound on |ρ|
        'max_current_magnitude' : theoretical upper bound on |J|
        'max_allowed_residual' : theoretical bound on |Δρ/Δt + div J|
        'phi_s_confinement'   : φ (golden ratio, U6 escape threshold)
        'k_phi_hotspot'       : 0.9×π ≈ 2.8274 (curvature hotspot threshold)
    """
    n_nodes = G.number_of_nodes()

    # U6: |Φ_s| < φ
    phi_s_bound = PHI

    # K_φ is bounded by π (wrapped angle difference)
    k_phi_bound = PI

    # Maximum charge density
    max_charge = phi_s_bound + k_phi_bound

    # J_φ = mean(sin(Δθ)), bounded by 1
    j_phi_bound = 1.0

    # J_ΔNFR = mean(ΔNFR_j - ΔNFR_i), bounded by max|ΔNFR| spread
    # Under U2, ΔNFR is bounded; use Φ_s bound as proxy
    j_dnfr_bound = 2.0 * phi_s_bound  # worst case: ±Φ_s

    # Maximum current magnitude
    max_current = math.sqrt(j_phi_bound ** 2 + j_dnfr_bound ** 2)

    # Maximum divergence scales with max_current / connectivity
    avg_degree = 2.0 * G.number_of_edges() / max(n_nodes, 1)
    max_div = 2.0 * max_current  # upper bound on discrete divergence

    # Maximum allowed residual (charge rate + divergence)
    max_residual = max_charge + max_div

    return {
        'max_charge_density': max_charge,
        'max_current_magnitude': max_current,
        'max_allowed_residual': max_residual,
        'phi_s_confinement': phi_s_bound,
        'k_phi_hotspot': K_PHI_CANONICAL_THRESHOLD,  # 0.9×π ≈ 2.8274 (canonical curvature hotspot threshold)
        'average_degree': avg_degree,
    }

# ---------------------------------------------------------------------------
# Grammar violation detection via conservation analysis
# ---------------------------------------------------------------------------

def detect_grammar_violations_from_conservation(
    balance: ConservationBalance,
    bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    r"""Detect grammar violations by analyzing conservation residuals.

    High conservation residuals indicate that the operator sequence
    violated grammar constraints. This function classifies violations
    by type.

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
    threshold = PHI  # Golden ratio as natural threshold (from U6)
    if bounds is not None:
        threshold = bounds.get('max_allowed_residual', PHI)

    violation_types: list[str] = []
    nodes_violating: list[Any] = []

    for node, res in balance.residual.items():
        if abs(res) > threshold:
            nodes_violating.append(node)

    # Classify violation types
    if balance.charge_drift > PHI:
        violation_types.append("U6_confinement_breach")

    if balance.rms_residual > _BALANCE_RMS_ALERT:
        violation_types.append("U2_convergence_failure")

    if balance.max_residual > 2 * threshold:
        violation_types.append("U3_phase_incompatibility")

    severity = min(1.0, balance.rms_residual / max(threshold, 1e-10))

    return {
        'violations_detected': len(violation_types) > 0,
        'violation_count': len(violation_types),
        'violation_types': violation_types,
        'severity': severity,
        'nodes_violating': nodes_violating,
    }

# ---------------------------------------------------------------------------
# Noether charge: total conserved quantity
# ---------------------------------------------------------------------------

def compute_noether_charge(G: Any) -> float:
    r"""Compute the total Noether charge Q = Σ_i ρ(i) = Σ_i [Φ_s(i) + K_φ(i)].

    Under grammar-compliant evolution, Q is approximately conserved:

        dQ/dt ≈ 0  ⟺  Grammar U1-U6 satisfied

    The charge Q integrates global (potential) and local (geometric)
    structural information into a single scalar.

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

    Under grammar-compliant evolution (U2 convergence):
        dE/dt ≤ 0  (energy is non-increasing)

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
        - Monitored by grammar U6 (|Φ_s| < φ)

    **Geometric sector** (local, phase-driven):
        ∂K_φ/∂t + div(J_φ) ≈ 0
        - Conserves when phase curvature transports without source terms
        - Violated by phase-incompatible operations (grammar U3)
        - Monitored by curvature hotspot detection (|K_φ| < 2.8274)

    The CROSS-COUPLING between sectors measures how potential changes
    induce geometric changes and vice versa — this is the mechanism
    behind the Ψ = K_φ + i·J_φ unification.

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
    nodes = list(decomp['potential_residual'].keys())

    pot_res = np.array([decomp['potential_residual'][n] for n in nodes])
    geo_res = np.array([decomp['geometric_residual'][n] for n in nodes])

    rms_pot = float(np.sqrt(np.mean(pot_res ** 2)))
    rms_geo = float(np.sqrt(np.mean(geo_res ** 2)))

    # Cross-coupling: correlation between sector residuals
    if len(nodes) > 2 and np.std(pot_res) > 1e-15 and np.std(geo_res) > 1e-15:
        cross_corr = float(np.corrcoef(pot_res, geo_res)[0, 1])
    else:
        cross_corr = 0.0

    # Determine dominant sector
    if rms_pot > _SECTOR_IMBALANCE_RATIO * rms_geo:
        dominant = 'potential'
    elif rms_geo > _SECTOR_IMBALANCE_RATIO * rms_pot:
        dominant = 'geometric'
    else:
        dominant = 'balanced'

    asymmetry = max(rms_pot, rms_geo) / (min(rms_pot, rms_geo) + 1e-15)

    return {
        'potential_sector_residual': rms_pot,
        'geometric_sector_residual': rms_geo,
        'cross_coupling_strength': cross_corr,
        'dominant_sector': dominant,
        'sector_asymmetry': asymmetry,
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
        e_before = 0.5 * sum(
            before.phi_s[n] ** 2 + before.grad_phi[n] ** 2
            + before.k_phi[n] ** 2
            + before.j_phi[n] ** 2 + before.j_dnfr[n] ** 2
            for n in before.phi_s
        )
        e_after = 0.5 * sum(
            after.phi_s[n] ** 2 + after.grad_phi[n] ** 2
            + after.k_phi[n] ** 2
            + after.j_phi[n] ** 2 + after.j_dnfr[n] ** 2
            for n in after.phi_s
        )
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
    r"""Verify the sequence Ward identity: Σ_k ⟨S_k⟩ ≈ 0.

    For a complete grammar-valid sequence, the total source over all steps
    must approximately vanish (U1 closure + U2 convergence).

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
    threshold = PHI / n_steps  # Scale threshold with sequence length

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
    Lyapunov candidate.  Under grammar-compliant evolution (U2), dE/dt ≤ 0.

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

    Under grammar-compliant evolution (U2 convergence + stabilizers):
        dE/dt ≤ 0  (Lyapunov theorem)

    The structural dissipation function is D[G] = -dE/dt ≥ 0.

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
    e_before = 0.5 * sum(
        before.phi_s[n] ** 2 + before.grad_phi[n] ** 2
        + before.k_phi[n] ** 2
        + before.j_phi[n] ** 2 + before.j_dnfr[n] ** 2
        for n in before.phi_s
    )
    e_after = 0.5 * sum(
        after.phi_s[n] ** 2 + after.grad_phi[n] ** 2
        + after.k_phi[n] ** 2
        + after.j_phi[n] ** 2 + after.j_dnfr[n] ** 2
        for n in after.phi_s
    )

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
    graph Laplacian L: ρ(i) = Σ_k ρ̂_k ψ_k(i).

    The continuity equation mode-by-mode reads:
        dρ̂_k/dt + λ_k Ĵ_k = Ŝ_k

    Low-frequency modes (small λ_k) → conservation regime.
    High-frequency modes (large λ_k) → rapid relaxation.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Laplacian eigenvalues λ_k (sorted ascending).
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
    r"""Decompose conservation fields in the graph Laplacian eigenbasis.

    Connects TNFR conservation to spectral graph theory.  The Laplacian
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

    # Build graph Laplacian
    if nx is not None and isinstance(G, nx.Graph):
        L = nx.laplacian_matrix(G).toarray().astype(float)
    else:
        # Fallback: build from adjacency
        L = np.zeros((n, n))
        node_idx = {nd: i for i, nd in enumerate(nodes)}
        for u, v in G.edges():
            i, j = node_idx[u], node_idx[v]
            L[i, j] = -1.0
            L[j, i] = -1.0
            L[i, i] += 1.0
            L[j, j] += 1.0

    # Eigendecomposition
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
                    mean_dnfr = float(np.mean([
                        G.nodes[j].get("delta_nfr", 0.0) for j in nbrs
                    ]))
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

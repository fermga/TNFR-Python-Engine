r"""TNFR variational diagnostics and a specified quadratic substrate model.

IMPLEMENTED FUNCTIONALS
=======================
For canonical field readouts, this module computes

    T = 0.5 * (J_phi**2 + J_DNFR**2),
    V = 0.5 * (Phi_s**2 + |grad_phi|**2 + K_phi**2),
    L = T - V,  H = T + V,  S = sum(dt * L).

The full H equals conservation.compute_energy_functional; its density and
bilinear interaction delegate to physics.unified. The conjugate-coordinate
model uses (K_phi, J_phi) and (Phi_s, J_DNFR), with an isotropic harmonic flow.
Its algebra and local Jacobian checks are valid for that specified model.

CERTIFICATE SCOPE
==================
- Snapshot products and covariance statistics are legacy diagnostics, not
  symplectic volumes or Poisson brackets. Symplectic preservation requires a
  supplied tangent map; snapshot-only calls return an inconclusive result.
- Nonzero telemetry thresholds are regular points of V=0.5*x**2, whose only
  critical point is x=0. Proximity to a threshold is reported separately.
- Finite recorded action and energy changes do not prove infinite-horizon
  convergence or grammar compliance. The grammar-labelled stationarity
  readouts are heuristic comparisons, not replacement grammar validators.
- A derivation of the full nodal equation from this V remains unresolved.
  In particular, DeltaNFR=-dV/dEPI is not an identity of the implemented field
  definitions: on one edge in the pure EPI channel, DeltaNFR=[-1,1] at
  EPI=[1,0], whereas the negative gradient of this V is [-2,2].
- The graph-wave overdamped limit uses q''+gamma*q'+L_rw*q=0; the isotropic
  substrate flow instead has q''=-q. An explicit coordinate/metric/damping
  bridge between these models has not been established here. In the nodal
  equation nu_f is mobility, not a derived inverse inertial mass.

See theory/TNFR_VARIATIONAL_PRINCIPLE.md and
theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md for assumptions and scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..constants.canonical import PI, U6_STRUCTURAL_POTENTIAL_LIMIT
from ..mathematics.unified_numerical import np

# ---------------------------------------------------------------------------
# Critical point classification
# ---------------------------------------------------------------------------
_THRESHOLD_PROXIMITY_FRACTION = 0.1

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from .canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from .extended import compute_dnfr_flux, compute_phase_current
from .unified import (
    _StructuralFieldReadout,
    _action_density_from_fields,
    _capture_structural_fields,
    _energy_density_from_fields,
)
from .unified import compute_action_density as _action_density
from .unified import compute_energy_density as _raw_energy_density

# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConjugatePair:
    """A canonical conjugate pair (q, p) in the TNFR phase space.

    Attributes
    ----------
    sector : str
        ``'geometric'`` for (K_φ, J_φ) or ``'potential'`` for (Φ_s, J_ΔNFR).
    q : dict[Any, float]
        Configuration field values (generalised coordinate) per node.
    p : dict[Any, float]
        Transport field values (conjugate momentum) per node.
    """

    sector: str
    q: dict[Any, float]
    p: dict[Any, float]


@dataclass(frozen=True)
class LagrangianSnapshot:
    """Complete Lagrangian analysis at a single instant.

    Attributes
    ----------
    kinetic : dict[Any, float]
        T(i) = ½[J_φ(i)² + J_ΔNFR(i)²] per node.
    potential : dict[Any, float]
        V(i) = ½[Φ_s(i)² + |∇φ|(i)² + K_φ(i)²] per node.
    lagrangian : dict[Any, float]
        ℒ(i) = T(i) − V(i) per node.
    hamiltonian : dict[Any, float]
        H(i) = T(i) + V(i) per node (= energy density / 2).
    interaction : dict[Any, float]
        𝒜(i) = Φ_s·|∇φ| + K_φ·J_φ + |∇φ|·J_ΔNFR per node (bilinear coupling).
    total_lagrangian : float
        L = Σ_i ℒ(i).
    total_hamiltonian : float
        H = Σ_i H(i) (total energy).
    total_kinetic : float
        T = Σ_i T(i).
    total_potential : float
        V = Σ_i V(i).
    conjugate_geometric : ConjugatePair
        (K_φ, J_φ) sector.
    conjugate_potential : ConjugatePair
        (Φ_s, J_ΔNFR) sector.
    """

    kinetic: dict[Any, float]
    potential: dict[Any, float]
    lagrangian: dict[Any, float]
    hamiltonian: dict[Any, float]
    interaction: dict[Any, float]
    total_lagrangian: float
    total_hamiltonian: float
    total_kinetic: float
    total_potential: float
    conjugate_geometric: ConjugatePair
    conjugate_potential: ConjugatePair


@dataclass(frozen=True)
class EulerLagrangeResidual:
    r"""Residual of the specified harmonic momentum equations.

    For each conjugate sector, R=dp/dt+q. A small value checks this equation
    only; it does not establish the configuration equation q'=p, stationarity
    of the full graph-field action, or equivalence with the nodal equation.

    Attributes
    ----------
    residual : dict[Any, float]
        EL residual per node.
    mean_residual : float
    rms_residual : float
    max_residual : float
    is_stationary : bool
        True when rms_residual < threshold.
    stationarity_quality : float
        1/(1 + rms_residual) ∈ (0, 1].
    """

    residual: dict[Any, float]
    mean_residual: float
    rms_residual: float
    max_residual: float
    is_stationary: bool
    stationarity_quality: float


@dataclass(frozen=True)
class SymplecticCheck:
    r"""Local Jacobian check with legacy snapshot statistics.

    Symplecticity requires ``D(F).T @ omega @ D(F) = omega``. Snapshots alone
    do not determine D(F); without a supplied Jacobian the result is explicitly
    inconclusive. Even a passing Jacobian check concerns that supplied tangent
    map, not a proof for the operator at every state.

    Attributes
    ----------
    operator_name : str
    symplectic_ratio_geometric : float
        Legacy ratio of sum |q*p| in the geometric sector; not a 2-form ratio.
    symplectic_ratio_potential : float
        Legacy ratio of sum |q*p| in the potential sector.
    is_canonical : bool or None
        True/False for the supplied Jacobian, None without tangent evidence.
    phase_space_volume_before : float
    phase_space_volume_after : float
    volume_ratio : float
        Legacy ratio of snapshot products; not a transported volume ratio.
    classification : str
        ``'canonical'``, ``'non_symplectic'``, or ``'inconclusive'``.
    heuristic_classification : str
        The former product-ratio classification, retained as a statistic only.
    verification_method : str
        ``'snapshot_only'`` or ``'provided_jacobian'``.
    symplectic_residual : float or None
        Maximum absolute entry of D(F).T @ omega @ D(F) - omega.
    """

    operator_name: str
    symplectic_ratio_geometric: float
    symplectic_ratio_potential: float
    is_canonical: bool | None
    phase_space_volume_before: float
    phase_space_volume_after: float
    volume_ratio: float
    classification: str
    heuristic_classification: str = "unknown"
    verification_method: str = "snapshot_only"
    symplectic_residual: float | None = None


@dataclass(frozen=True)
class GrammarStationarityAnalysis:
    r"""Heuristic field comparisons labelled by related grammar rules.

    These do not validate operator sequences or establish stationarity of an
    action. Use the operator grammar validator for U1-U5 and drift telemetry
    for U6. The legacy ``is_satisfied`` field refers to this heuristic only.

    Attributes
    ----------
    rule : str
        Grammar rule identifier (e.g. ``'U1a'``).
    variational_interpretation : str
        How the rule maps to a variational condition.
    is_satisfied : bool
    diagnostic_value : float
        Quantitative measure of (non-)satisfaction.
    """

    rule: str
    variational_interpretation: str
    is_satisfied: bool
    diagnostic_value: float
    verification_scope: str = "heuristic"


@dataclass(frozen=True)
class CriticalPointAnalysis:
    r"""Quadratic-potential derivatives at a telemetry threshold.

    Attributes
    ----------
    field_name : str
        Name of the field analysed.
    threshold_value : float
        Theoretical TNFR threshold.
    gradient_at_threshold : float
        ∂V/∂field at the threshold value.
    is_critical : bool
        True when gradient ≈ 0 (extremum or saddle).
    curvature_at_threshold : float
        ∂²V/∂field² — positive = minimum, negative = maximum.
    critical_type : str
        ``'minimum'``, ``'maximum'``, ``'saddle'``, or ``'regular'``.
    near_threshold_count : int
        Observed nodes within the threshold-proximity band. Proximity does
        not imply a stationary point of V.
    """

    field_name: str
    threshold_value: float
    gradient_at_threshold: float
    is_critical: bool
    curvature_at_threshold: float
    critical_type: str
    near_threshold_count: int = 0


@dataclass
class VariationalTimeSeries:
    """Time-series of variational diagnostics across operator sequence.

    Built incrementally via :meth:`VariationalTracker.record`.
    """

    times: list[float] = field(default_factory=list)
    total_lagrangian: list[float] = field(default_factory=list)
    total_hamiltonian: list[float] = field(default_factory=list)
    total_kinetic: list[float] = field(default_factory=list)
    total_potential: list[float] = field(default_factory=list)
    el_rms_residual: list[float] = field(default_factory=list)
    stationarity_quality: list[float] = field(default_factory=list)
    action_accumulated: list[float] = field(default_factory=list)

    @property
    def is_action_finite(self) -> bool:
        """True when accumulated action remains bounded (U2 compliance)."""
        if not self.action_accumulated:
            return True
        return all(math.isfinite(a) for a in self.action_accumulated)

    @property
    def mean_stationarity(self) -> float:
        """Average stationarity quality across all recorded steps."""
        if not self.stationarity_quality:
            return 0.0
        return float(np.mean(self.stationarity_quality))


# ---------------------------------------------------------------------------
#  Core Lagrangian computations
# ---------------------------------------------------------------------------


def compute_kinetic_density(G: Any) -> dict[Any, float]:
    r"""Compute transport kinetic energy density per node.

    T(i) = ½ [J_φ(i)² + J_ΔNFR(i)²]

    The transport fields (J_φ, J_ΔNFR) act as generalised velocities in
    the TNFR phase space, carrying the temporal evolution of the
    geometric and potential sectors, respectively.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    dict[node, float]
    """
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    return _kinetic_density_from_fields(j_phi, j_dnfr)


def _kinetic_density_from_fields(
    j_phi: dict[Any, float], j_dnfr: dict[Any, float]
) -> dict[Any, float]:
    return {n: 0.5 * (j_phi[n] ** 2 + j_dnfr[n] ** 2) for n in j_phi}


def compute_potential_density(G: Any) -> dict[Any, float]:
    r"""Compute configuration potential energy density per node.

    V(i) = ½ [Φ_s(i)² + |∇φ|(i)² + K_φ(i)²]

    These configuration-field statistics define the recorded potential.
    Their negative EPI gradient is not generally the canonical pressure.
    The separate EPI-only Dirichlet balance is implemented by
    ``structural_diffusion.compute_diffusion_energy``.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    dict[node, float]
    """
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    return _potential_density_from_fields(phi_s, grad_phi, k_phi)


def _potential_density_from_fields(
    phi_s: dict[Any, float], grad_phi: dict[Any, float], k_phi: dict[Any, float]
) -> dict[Any, float]:
    return {
        n: 0.5 * (phi_s[n] ** 2 + grad_phi[n] ** 2 + k_phi[n] ** 2)
        for n in phi_s
    }


def compute_lagrangian_density(G: Any) -> dict[Any, float]:
    r"""Compute TNFR Lagrangian density per node.

    ℒ(i) = T(i) − V(i)
          = ½[J_φ² + J_ΔNFR²] − ½[Φ_s² + |∇φ|² + K_φ²]

    Positive ℒ indicates transport-dominated dynamics (kinetic regime).
    Negative ℒ indicates configuration-dominated dynamics (potential regime).

    This defines the recorded action density. A derivation of the full nodal
    equation from this particular field potential remains unresolved.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    dict[node, float]
    """
    T = compute_kinetic_density(G)
    V = compute_potential_density(G)
    return {n: T[n] - V[n] for n in G.nodes()}


def compute_hamiltonian_density(G: Any) -> dict[Any, float]:
    r"""Compute TNFR Hamiltonian density per node.

    H(i) = T(i) + V(i)
          = ½[Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²]
          = ½ · ℰ(i)

    where ℰ(i) is the raw energy density from
    :func:`unified.compute_energy_density` (CANONICAL SOURCE).

    **Consistency contract**:
        ``H(i) == 0.5 * unified.compute_energy_density(G)[i]``

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    dict[node, float]
    """
    raw = _raw_energy_density(G)
    return {n: 0.5 * raw[n] for n in raw}


def compute_interaction_density(G: Any) -> dict[Any, float]:
    r"""Compute cross-sector interaction (bilinear coupling) per node.

    𝒜(i) = Φ_s·|∇φ| + K_φ·J_φ + |∇φ|·J_ΔNFR

    This is the **interaction Lagrangian** coupling the geometric and
    potential sectors.

    **Single source of truth**: delegates to
    :func:`unified.compute_action_density`.

    In the full Lagrangian with interactions:
        ℒ_full = T − V − 𝒜

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    dict[node, float]
    """
    return _action_density(G)


# ---------------------------------------------------------------------------
#  Sector decomposition translation
# ---------------------------------------------------------------------------


def translate_sectors(G: Any) -> dict[str, Any]:
    r"""Translate between the variational and conservation sector decompositions.

    The **same five configuration and transport fields** admit two
    decompositions of the recorded energy. Coherence length ξ_C does not
    enter these algebraic expressions and is not computed here:

    +-------------------+---------------------------------------------------------+
    | Decomposition     | Fields                                                  |
    +===================+=========================================================+
    | **Variational**   | Kinetic T = ½[J_φ² + J_ΔNFR²]  (transport)             |
    | (T / V split)     | Potential V = ½[Φ_s² + |∇φ|² + K_φ²]  (configuration)  |
    +-------------------+---------------------------------------------------------+
    | **Conservation**  | Charge ρ = Φ_s + K_φ  (scalar density)                  |
    | (ρ / J split)     | Current J = (J_φ, J_ΔNFR)  (vector transport)           |
    +-------------------+---------------------------------------------------------+
    | **Unified**       | Complex field Ψ = K_φ + i·J_φ  (geometry-transport)     |
    | (Ψ unification)   | Orthogonal to both T/V and ρ/J                          |
    +-------------------+---------------------------------------------------------+

    **Why they differ**: the variational split groups by *temporal role*
    (kinetic = time-derivatives, potential = configuration); the
    conservation split groups by *physical role* (charge = what is
    conserved, current = how it flows); the unified representation
    groups by *dual structure* (K_φ and J_φ as real and imaginary parts).

    **Consistency identity**:
        ``T(i) + V(i) == ½·ℰ(i)``  (both projections sum to the same energy)

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    dict[str, Any]
        - ``variational``: {'T': dict, 'V': dict}
        - ``conservation``: {'rho': dict, 'J_phi': dict, 'J_dnfr': dict}
        - ``unified_psi``: dict[node, complex]  (K_φ + i·J_φ)
        - ``energy_density``: dict[node, float]  (raw ℰ from unified.py)
        - ``consistency_check``: float  (max |T+V − ½ℰ| across nodes, should be ~0)
    """
    from .conservation import _charge_density_from_fields
    from .unified import _complex_geometric_field

    fields = _capture_structural_fields(G)
    phi_s, grad_phi, k_phi = fields.phi_s, fields.grad_phi, fields.k_phi
    j_phi, j_dnfr = fields.j_phi, fields.j_dnfr
    T = _kinetic_density_from_fields(j_phi, j_dnfr)
    V = _potential_density_from_fields(phi_s, grad_phi, k_phi)
    raw = _energy_density_from_fields(phi_s, grad_phi, k_phi, j_phi, j_dnfr)

    rho = _charge_density_from_fields(phi_s, k_phi)
    psi = _complex_geometric_field(k_phi, j_phi)

    # Consistency: T(i) + V(i) must equal ½·ℰ(i)
    max_err = (
        max(abs((T[n] + V[n]) - 0.5 * raw[n]) for n in phi_s)
        if phi_s
        else 0.0
    )

    return {
        "variational": {"T": T, "V": V},
        "conservation": {"rho": rho, "J_phi": j_phi, "J_dnfr": j_dnfr},
        "unified_psi": psi,
        "energy_density": raw,
        "consistency_check": max_err,
    }


# ---------------------------------------------------------------------------
#  Conjugate pairs and phase space
# ---------------------------------------------------------------------------


def identify_conjugate_pairs(G: Any) -> tuple[ConjugatePair, ConjugatePair]:
    r"""Identify the canonical conjugate pairs in the TNFR phase space.

    The conservation law structure (Noether theorem) reveals two sectors:

    - **Geometric sector**: (q, p) = (K_φ, J_φ)
      ``∂K_φ/∂t + div(J_φ) ≈ 0``

    - **Potential sector**: (q, p) = (Φ_s, J_ΔNFR)
      ``∂Φ_s/∂t + div(J_ΔNFR) ≈ 0``

    These are the natural conjugate pairs from the symplectic structure
    of the TNFR action.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    tuple[ConjugatePair, ConjugatePair]
        (geometric_pair, potential_pair).
    """
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    phi_s = compute_structural_potential(G)
    j_dnfr = compute_dnfr_flux(G)

    geometric = ConjugatePair(sector="geometric", q=k_phi, p=j_phi)
    potential = ConjugatePair(sector="potential", q=phi_s, p=j_dnfr)
    return geometric, potential


def compute_phase_space_volume(pair: ConjugatePair) -> float:
    r"""Return the legacy snapshot product statistic sum |q(i)*p(i)|.

    Ω = Σ_i |q(i)·p(i)|

    The historical function name is retained for compatibility. This is not a
    symplectic volume: a canonical rotation can change it from zero to nonzero.
    It must not be used to certify a map or invoke Liouville's theorem.

    Parameters
    ----------
    pair : ConjugatePair

    Returns
    -------
    float
    """
    nodes = list(pair.q.keys())
    if not nodes:
        return 0.0
    return float(sum(abs(pair.q[n] * pair.p[n]) for n in nodes))


def compute_poisson_bracket_estimate(
    pair: ConjugatePair,
) -> float:
    r"""Return a legacy covariance-determinant statistic for node samples.

    Despite its historical name, this is not a Poisson bracket. Field sample
    variances do not determine the Poisson tensor. Use
    :func:`tnfr.physics.symplectic_substrate.poisson_bracket` with observable
    gradients when a bracket on the specified substrate is required.

    Parameters
    ----------
    pair : ConjugatePair

    Returns
    -------
    float
        Square root of the nonnegative part of the sample statistic.
    """
    nodes = list(pair.q.keys())
    if len(nodes) < 2:
        return 0.0
    q_arr = np.array([pair.q[n] for n in nodes])
    p_arr = np.array([pair.p[n] for n in nodes])

    var_q = float(np.var(q_arr))
    var_p = float(np.var(p_arr))
    if var_q < 1e-30 or var_p < 1e-30:
        return 0.0
    cov_qp = float(np.cov(q_arr, p_arr)[0, 1])
    det = var_q * var_p - cov_qp**2
    return float(np.sqrt(max(det, 0.0)))


# ---------------------------------------------------------------------------
#  Lagrangian snapshot (complete analysis at one instant)
# ---------------------------------------------------------------------------


def capture_lagrangian_snapshot(G: Any) -> LagrangianSnapshot:
    """Capture complete Lagrangian analysis of the current graph state.

    The caller must hold graph state fixed during the call. Returned maps are
    detached; capture is not atomic with concurrent evolution.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    LagrangianSnapshot
    """
    return _lagrangian_snapshot_from_fields(_capture_structural_fields(G))


def _lagrangian_snapshot_from_fields(
    fields: _StructuralFieldReadout,
) -> LagrangianSnapshot:
    """Derive every snapshot component from the same owned base maps."""
    T = _kinetic_density_from_fields(fields.j_phi, fields.j_dnfr)
    V = _potential_density_from_fields(fields.phi_s, fields.grad_phi, fields.k_phi)

    nodes = fields.phi_s
    lagrangian = {n: T[n] - V[n] for n in nodes}
    hamiltonian = {n: T[n] + V[n] for n in nodes}
    interaction = _action_density_from_fields(
        fields.phi_s, fields.grad_phi, fields.k_phi, fields.j_phi, fields.j_dnfr
    )

    total_T = sum(T.values())
    total_V = sum(V.values())

    geo = ConjugatePair(sector="geometric", q=fields.k_phi, p=fields.j_phi)
    pot = ConjugatePair(sector="potential", q=fields.phi_s, p=fields.j_dnfr)

    return LagrangianSnapshot(
        kinetic=T,
        potential=V,
        lagrangian=lagrangian,
        hamiltonian=hamiltonian,
        interaction=interaction,
        total_lagrangian=total_T - total_V,
        total_hamiltonian=total_T + total_V,
        total_kinetic=total_T,
        total_potential=total_V,
        conjugate_geometric=geo,
        conjugate_potential=pot,
    )


# ---------------------------------------------------------------------------
#  Euler-Lagrange residual
# ---------------------------------------------------------------------------


def compute_euler_lagrange_residual(
    before: LagrangianSnapshot,
    after: LagrangianSnapshot,
    dt: float = 1.0,
    stationarity_threshold: float = 0.1,
) -> EulerLagrangeResidual:
    r"""Compute the Euler-Lagrange residual between two snapshots.

    The EL equation for the TNFR action is:

        d/dt(∂ℒ/∂q̇_i) − ∂ℒ/∂q_i = 0

    In the conjugate-pair formulation, this becomes (for each sector):

        dp_i/dt + ∂V/∂q_i = 0    (Hamilton's equation for momentum)

    The **residual** measures departure from stationarity:

        R(i) = Δp_i/Δt + [∂V/∂q_i]_{mean}

    A small residual concerns these harmonic momentum equations only. It does
    not validate the full nodal equation, grammar, or configuration equations.

    Parameters
    ----------
    before, after : LagrangianSnapshot
    dt : float
    stationarity_threshold : float

    Returns
    -------
    EulerLagrangeResidual
    """
    nodes = list(after.lagrangian.keys())
    residual: dict[Any, float] = {}

    for n in nodes:
        # Geometric sector: dp/dt = ΔJ_φ/Δt,  ∂V/∂q ≈ K_φ (from V = ½K_φ²)
        dj_phi = (
            after.conjugate_geometric.p.get(n, 0.0)
            - before.conjugate_geometric.p.get(n, 0.0)
        ) / dt
        k_phi_avg = 0.5 * (
            before.conjugate_geometric.q.get(n, 0.0)
            + after.conjugate_geometric.q.get(n, 0.0)
        )
        r_geo = dj_phi + k_phi_avg

        # Potential sector: dp/dt = ΔJ_ΔNFR/Δt,  ∂V/∂q ≈ Φ_s
        dj_dnfr = (
            after.conjugate_potential.p.get(n, 0.0)
            - before.conjugate_potential.p.get(n, 0.0)
        ) / dt
        phi_s_avg = 0.5 * (
            before.conjugate_potential.q.get(n, 0.0)
            + after.conjugate_potential.q.get(n, 0.0)
        )
        r_pot = dj_dnfr + phi_s_avg

        # Total residual per node (RMS of both sectors)
        residual[n] = float(np.sqrt(r_geo**2 + r_pot**2))

    res_arr = np.array(list(residual.values()))
    mean_r = float(np.mean(res_arr)) if len(res_arr) > 0 else 0.0
    rms_r = float(np.sqrt(np.mean(res_arr**2))) if len(res_arr) > 0 else 0.0
    max_r = float(np.max(res_arr)) if len(res_arr) > 0 else 0.0

    return EulerLagrangeResidual(
        residual=residual,
        mean_residual=mean_r,
        rms_residual=rms_r,
        max_residual=max_r,
        is_stationary=rms_r < stationarity_threshold,
        stationarity_quality=1.0 / (1.0 + rms_r),
    )


# ---------------------------------------------------------------------------
#  Action functional
# ---------------------------------------------------------------------------


def compute_action_functional(
    snapshots: Sequence[LagrangianSnapshot],
    dt: float = 1.0,
) -> float:
    r"""Compute the discrete TNFR action functional from a time series.

    S = Σ_n Δt · L(t_n)  where L(t_n) = Σ_i ℒ(i, t_n)

    A finite recorded sum does not prove infinite-horizon convergence or U2.

    Parameters
    ----------
    snapshots : Sequence[LagrangianSnapshot]
    dt : float

    Returns
    -------
    float
        Sampled action over the supplied finite trajectory; this does not
        certify U2 compliance or convergence over an infinite time horizon.
    """
    return dt * sum(s.total_lagrangian for s in snapshots)


# ---------------------------------------------------------------------------
#  Symplectic structure and canonical transformation checks
# ---------------------------------------------------------------------------


def check_symplectic_preservation(
    before: LagrangianSnapshot,
    after: LagrangianSnapshot,
    operator_name: str = "unknown",
    tolerance: float = 0.3,
    *,
    jacobian: Any | None = None,
    jacobian_tolerance: float = 1e-9,
) -> SymplecticCheck:
    r"""Check a supplied local Jacobian, or report inconclusive snapshots.

    A pair of snapshots cannot distinguish the identity from a non-symplectic
    map fixing the same point. The original positional arguments and product
    ratios remain available, but only a Jacobian can set ``is_canonical``.

    Parameters
    ----------
    before, after : LagrangianSnapshot
    operator_name : str
    tolerance : float
        Tolerance for the legacy product-ratio classification only.
    jacobian : array-like, optional
        Caller-supplied derivative of the map, of shape (4N, 4N). Input/output
        coordinates use each snapshot's geometric-q node order and interleave
        (K_phi, J_phi, Phi_s, J_DNFR) per node. A passing result certifies this
        tangent map only. Dimension-changing maps are not supported here.
    jacobian_tolerance : float
        Absolute pullback-residual tolerance, independent of the legacy ratio
        tolerance. Must be finite and nonnegative.

    Returns
    -------
    SymplecticCheck
    """
    if not math.isfinite(jacobian_tolerance) or jacobian_tolerance < 0.0:
        raise ValueError("jacobian_tolerance must be finite and nonnegative")
    vol_geo_before = compute_phase_space_volume(before.conjugate_geometric)
    vol_geo_after = compute_phase_space_volume(after.conjugate_geometric)
    vol_pot_before = compute_phase_space_volume(before.conjugate_potential)
    vol_pot_after = compute_phase_space_volume(after.conjugate_potential)

    def _ratio(a: float, b: float) -> float:
        if b < 1e-30:
            return 1.0 if a < 1e-30 else float("inf")
        return a / b

    ratio_geo = _ratio(vol_geo_after, vol_geo_before)
    ratio_pot = _ratio(vol_pot_after, vol_pot_before)

    total_before = vol_geo_before + vol_pot_before
    total_after = vol_geo_after + vol_pot_after
    vol_ratio = _ratio(total_after, total_before)

    is_canonical_geo = abs(ratio_geo - 1.0) < tolerance
    is_canonical_pot = abs(ratio_pot - 1.0) < tolerance
    ratios_match = is_canonical_geo and is_canonical_pot

    # Classification based on volume change
    if ratios_match:
        heuristic = "canonical"
    elif vol_ratio < 1.0 - tolerance:
        heuristic = "dissipative"
    elif vol_ratio > 1.0 + tolerance:
        heuristic = "expansive"
    else:
        heuristic = "mixed"

    is_canonical = None
    classification = "inconclusive"
    residual = None
    method = "snapshot_only"
    if jacobian is not None:
        from .symplectic_substrate import symplectic_pullback_residual

        n_nodes = len(before.conjugate_geometric.q)
        for snapshot in (before, after):
            nodes = set(snapshot.conjugate_geometric.q)
            if len(nodes) != n_nodes or any(
                set(values) != nodes
                for values in (
                    snapshot.conjugate_geometric.p,
                    snapshot.conjugate_potential.q,
                    snapshot.conjugate_potential.p,
                )
            ):
                raise ValueError("Jacobian checks require matching fixed-size coordinate fields")
        residual = symplectic_pullback_residual(jacobian, n_nodes)
        is_canonical = residual <= jacobian_tolerance
        classification = "canonical" if is_canonical else "non_symplectic"
        method = "provided_jacobian"

    return SymplecticCheck(
        operator_name=operator_name,
        symplectic_ratio_geometric=ratio_geo,
        symplectic_ratio_potential=ratio_pot,
        is_canonical=is_canonical,
        phase_space_volume_before=total_before,
        phase_space_volume_after=total_after,
        volume_ratio=vol_ratio,
        classification=classification,
        heuristic_classification=heuristic,
        verification_method=method,
        symplectic_residual=residual,
    )


# ---------------------------------------------------------------------------
#  Grammar rules as variational/stationarity conditions
# ---------------------------------------------------------------------------


def analyze_grammar_stationarity(
    G: Any,
    snapshots: Sequence[LagrangianSnapshot] | None = None,
    dt: float = 1.0,
) -> list[GrammarStationarityAnalysis]:
    r"""Return heuristic field comparisons associated with grammar labels.

    Their thresholds test sampled energy/interaction statistics. They do not
    check operator history, phase admissibility, nested identities, or U6 drift,
    and must not replace the canonical grammar and confinement validators.

    Parameters
    ----------
    G : NetworkX graph
        Current state.
    snapshots : Sequence[LagrangianSnapshot], optional
        Time series for temporal checks (U2, U5).  If *None*, only
        instantaneous checks are performed.
    dt : float

    Returns
    -------
    list[GrammarStationarityAnalysis]
    """
    return _grammar_stationarity_from_snapshot(
        capture_lagrangian_snapshot(G), snapshots, dt
    )


def _grammar_stationarity_from_snapshot(
    snap: LagrangianSnapshot,
    snapshots: Sequence[LagrangianSnapshot] | None = None,
    dt: float = 1.0,
) -> list[GrammarStationarityAnalysis]:
    """Evaluate existing heuristic comparisons without recapturing fields."""
    results: list[GrammarStationarityAnalysis] = []

    # --- U1a: Initiation = boundary condition on S at t=0 ------------------
    # Diagnostic only: nonzero density does not establish generator history.
    lag_vals = list(snap.lagrangian.values())
    has_nontrivial = any(abs(v) > 1e-12 for v in lag_vals)
    results.append(
        GrammarStationarityAnalysis(
            rule="U1a",
            variational_interpretation=(
                "Heuristic: at least one sampled Lagrangian density is nonzero; "
                "generator history is not checked."
            ),
            is_satisfied=has_nontrivial,
            diagnostic_value=float(np.max(np.abs(lag_vals))) if lag_vals else 0.0,
        )
    )

    # --- U1b: Closure = boundary condition on S at t_f --------------------
    # Potential dominance alone does not establish an attractor or closure.
    potential_dominant = snap.total_potential > snap.total_kinetic
    results.append(
        GrammarStationarityAnalysis(
            rule="U1b",
            variational_interpretation=(
                "Heuristic: potential energy exceeds kinetic energy (V > T); "
                "closure and attraction are not certified."
            ),
            is_satisfied=potential_dominant,
            diagnostic_value=snap.total_lagrangian,
        )
    )

    # --- U2: Convergence = finite action requirement -----------------------
    if snapshots and len(snapshots) >= 2:
        S = compute_action_functional(snapshots, dt=dt)
        is_finite = math.isfinite(S)
        results.append(
            GrammarStationarityAnalysis(
                rule="U2",
                variational_interpretation=(
                    "Heuristic: the recorded finite-horizon action sum is finite; "
                    "this does not establish U2 convergence."
                ),
                is_satisfied=is_finite,
                diagnostic_value=S if is_finite else float("inf"),
            )
        )
    else:
        # Instantaneous: check that Lagrangian density is bounded
        max_L = float(np.max(np.abs(lag_vals))) if lag_vals else 0.0
        results.append(
            GrammarStationarityAnalysis(
                rule="U2",
                variational_interpretation=(
                    "Bounded Lagrangian density: |ℒ(i)| < ∞ at each node "
                    "(necessary for finite action integral)."
                ),
                is_satisfied=math.isfinite(max_L),
                diagnostic_value=max_L,
            )
        )

    # --- U3: Resonant coupling = regularity of coupling terms ------
    # A finite interaction statistic is not a phase-compatibility test.
    interaction_vals = list(snap.interaction.values())
    max_interaction = (
        float(np.max(np.abs(interaction_vals))) if interaction_vals else 0.0
    )
    interaction_bounded = max_interaction < 16.0  # generous operational bound
    results.append(
        GrammarStationarityAnalysis(
            rule="U3",
            variational_interpretation=(
                "Heuristic: sampled interaction magnitude is below 16; "
                "the U3 phase gate is not evaluated."
            ),
            is_satisfied=interaction_bounded,
            diagnostic_value=max_interaction,
        )
    )

    # --- U4: Bifurcation = Morse-theory constraints at critical points ----
    # Energy partition does not determine the Hessian or a bifurcation.
    if snap.total_potential > 1e-12:
        tv_ratio = snap.total_kinetic / snap.total_potential
    else:
        tv_ratio = float("inf")
    results.append(
        GrammarStationarityAnalysis(
            rule="U4",
            variational_interpretation=(
                "Advisory heuristic: kinetic/potential energy ratio; "
                "bifurcation and handler history are not checked."
            ),
            is_satisfied=True,  # advisory
            diagnostic_value=tv_ratio,
        )
    )

    # --- U5: Multi-scale coherence = hierarchical action factorisation ----
    # Check: energy is distributed across nodes (not concentrated).
    h_vals = list(snap.hamiltonian.values())
    if h_vals:
        h_arr = np.array(h_vals)
        h_mean = float(np.mean(h_arr))
        h_std = float(np.std(h_arr))
        cv = h_std / h_mean if h_mean > 1e-12 else 0.0
        well_distributed = cv < 2.0
    else:
        cv = 0.0
        well_distributed = True
    results.append(
        GrammarStationarityAnalysis(
            rule="U5",
            variational_interpretation=(
                "Heuristic: node energy CV is below 2; nested identity and "
                "per-level stabilizers are not checked."
            ),
            is_satisfied=well_distributed,
            diagnostic_value=cv,
        )
    )

    # --- U6: Structural confinement = bounded potential sector ------------
    phi_s_vals = list(snap.conjugate_potential.q.values())
    if phi_s_vals:
        max_phi_s = float(np.max(np.abs(phi_s_vals)))
        confined = max_phi_s < U6_STRUCTURAL_POTENTIAL_LIMIT
    else:
        max_phi_s = 0.0
        confined = True
    results.append(
        GrammarStationarityAnalysis(
            rule="U6",
            variational_interpretation=(
                "Heuristic: absolute potential magnitude is below π/2; "
                "the U6 change from a reference state is not evaluated."
            ),
            is_satisfied=confined,
            diagnostic_value=max_phi_s,
        )
    )

    return results


# ---------------------------------------------------------------------------
#  Threshold analysis — critical points of V
# ---------------------------------------------------------------------------


def analyze_potential_critical_points(G: Any) -> list[CriticalPointAnalysis]:
    r"""Evaluate quadratic-potential derivatives at telemetry thresholds.

    The TNFR potential per node is:
        V(i) = ½[Φ_s² + |∇φ|² + K_φ²]

    For each field, V has the form ½x², so V'=x and V''=1. Its only critical
    point is x=0; the nonzero thresholds below are regular points. Counts of
    nearby observations are reported separately from criticality:

    - Φ_s threshold at π/2 ≈ 1.571: configured comparison level.
    - |∇φ| threshold at 0.9π ≈ 2.827: phase-wrap confinement limit. |∇φ| is
      a mean of WRAPPED angles, so |∇φ| ≤ π — the SAME bound as K_φ (audit
      2026: π scales the whole phase sector). The earlier |∇φ| early-warning level was an overlay,
      not a derived bound (measured sync-onset ≈ 0.29, σ-dependent).
    - K_φ threshold at 0.9π ≈ 2.827: phase-wrap confinement limit (same bound)

    No nonlinear constrained effective potential is specified by this function,
    so no saddle point or change in restoring-force sign is inferred.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    list[CriticalPointAnalysis]
    """
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    k_phi = compute_phase_curvature(G)
    return _potential_critical_points_from_fields(phi_s, grad_phi, k_phi)


def _potential_critical_points_from_fields(
    phi_s: dict[Any, float], grad_phi: dict[Any, float], k_phi: dict[Any, float]
) -> list[CriticalPointAnalysis]:
    """Apply the existing threshold readout to already captured field maps."""
    results: list[CriticalPointAnalysis] = []

    # Canonical thresholds. Both phase derivatives (|∇φ|, K_φ) are means of
    # WRAPPED angles bounded by π (audit 2026: π scales the whole phase
    # sector), so they share the SAME 0.9π wrap-margin threshold. Φ_s uses the
    # configured U6 comparison value (π/2). This evaluates magnitude, not the
    # actual U6 drift. The earlier |∇φ| early-warning level was an overlay, not a derived
    # bound: the measured sync-onset is ≈ 0.29 and σ-dependent.
    thresholds = [
        ("Phi_s", U6_STRUCTURAL_POTENTIAL_LIMIT, phi_s),
        ("grad_phi", 0.9 * PI, grad_phi),
        ("K_phi", 0.9 * PI, k_phi),
    ]

    for name, threshold, field_vals in thresholds:
        vals = np.array(list(field_vals.values()))
        if len(vals) == 0:
            continue

        # Measure proximity of field values to the threshold
        at_threshold = vals[
            np.abs(np.abs(vals) - threshold) < _THRESHOLD_PROXIMITY_FRACTION * threshold
        ]

        gradient_at_thresh = threshold
        curvature = 1.0
        is_critical_point = threshold == 0.0
        ctype = "minimum" if is_critical_point else "regular"

        results.append(
            CriticalPointAnalysis(
                field_name=name,
                threshold_value=threshold,
                gradient_at_threshold=gradient_at_thresh,
                is_critical=is_critical_point,
                curvature_at_threshold=curvature,
                critical_type=ctype,
                near_threshold_count=len(at_threshold),
            )
        )

    return results


# ---------------------------------------------------------------------------
#  Operator canonical classification
# ---------------------------------------------------------------------------

# Expected canonical properties of the 13 operators.
# Each operator has a type (canonical, dissipative, or expansive) and
# its effect on Hamiltonian (energy).
_OPERATOR_CANONICAL_MAP = {
    "AL": {"type": "generating", "dH": "increase", "symplectic": "expansive"},
    "EN": {"type": "canonical", "dH": "neutral", "symplectic": "canonical"},
    "IL": {"type": "dissipative", "dH": "decrease", "symplectic": "dissipative"},
    "OZ": {"type": "generating", "dH": "increase", "symplectic": "expansive"},
    "UM": {"type": "canonical", "dH": "neutral", "symplectic": "canonical"},
    "RA": {"type": "canonical", "dH": "neutral", "symplectic": "canonical"},
    "SHA": {"type": "canonical", "dH": "neutral", "symplectic": "canonical"},
    "VAL": {"type": "generating", "dH": "increase", "symplectic": "expansive"},
    "NUL": {"type": "dissipative", "dH": "decrease", "symplectic": "dissipative"},
    "THOL": {"type": "canonical", "dH": "neutral", "symplectic": "canonical"},
    "ZHIR": {"type": "generating", "dH": "increase", "symplectic": "expansive"},
    "NAV": {"type": "canonical", "dH": "neutral", "symplectic": "canonical"},
    "REMESH": {"type": "canonical", "dH": "neutral", "symplectic": "canonical"},
}


def classify_operator_canonical(
    before: LagrangianSnapshot,
    after: LagrangianSnapshot,
    operator_name: str,
    tolerance: float = 0.3,
    *,
    jacobian: Any | None = None,
    jacobian_tolerance: float = 1e-9,
) -> dict[str, Any]:
    r"""Report energy changes and an optional local symplectic check.

    Energy increase/decrease and symplecticity are independent: a canonical
    transformation need not preserve a given Hamiltonian. The historical
    ``expected_*`` table and ``consistent_with_theory`` key retain their
    energy-heuristic meaning, not a proof of operator or grammar compliance.
    Without a Jacobian, the nested symplectic check is inconclusive.

    Parameters
    ----------
    before, after : LagrangianSnapshot
    operator_name : str
    tolerance : float
        Energy and legacy snapshot-statistic tolerance.
    jacobian, jacobian_tolerance
        Optional tangent evidence passed to :func:`check_symplectic_preservation`.

    Returns
    -------
    dict[str, Any]
        Classification results.
    """
    symp = check_symplectic_preservation(
        before, after, operator_name, tolerance,
        jacobian=jacobian, jacobian_tolerance=jacobian_tolerance,
    )

    dH = after.total_hamiltonian - before.total_hamiltonian
    dT = after.total_kinetic - before.total_kinetic
    dV = after.total_potential - before.total_potential

    # Energy classification
    if abs(dH) < tolerance * max(abs(before.total_hamiltonian), 1e-6):
        energy_class = "neutral"
    elif dH > 0:
        energy_class = "generating"
    else:
        energy_class = "dissipative"

    # Look up theoretical expectation
    expected = _OPERATOR_CANONICAL_MAP.get(operator_name, {})

    return {
        "operator": operator_name,
        "symplectic_check": symp,
        "energy_change": dH,
        "kinetic_change": dT,
        "potential_change": dV,
        "energy_classification": energy_class,
        "expected_type": expected.get("type", "unknown"),
        "expected_dH": expected.get("dH", "unknown"),
        "expected_symplectic": expected.get("symplectic", "unknown"),
        "consistent_with_theory": (
            energy_class == expected.get("type", energy_class)
            or energy_class == "neutral"  # neutral is always acceptable
        ),
    }


# ---------------------------------------------------------------------------
#  Variational Tracker (time-series)
# ---------------------------------------------------------------------------


class VariationalTracker:
    """Track variational principle compliance across an operator sequence.

    Usage
    -----
    >>> tracker = VariationalTracker(G)
    >>> tracker.record(t=0.0)
    >>> Emission()(G, node)
    >>> tracker.record(t=1.0)
    >>> Coherence()(G, node)
    >>> tracker.record(t=2.0)
    >>> report = tracker.report()
    >>> print(f"Action finite: {report.is_action_finite}")
    """

    def __init__(self, G: Any, dt: float = 1.0) -> None:
        self._G = G
        self._dt = dt
        self._snapshots: list[tuple[float, LagrangianSnapshot]] = []
        self._series = VariationalTimeSeries()
        self._cumulative_action = 0.0

    def record(self, t: float = 0.0) -> LagrangianSnapshot:
        """Capture current Lagrangian state.

        Parameters
        ----------
        t : float
            Structural time stamp.

        Returns
        -------
        LagrangianSnapshot
        """
        snap = capture_lagrangian_snapshot(self._G)
        self._snapshots.append((t, snap))

        self._series.times.append(t)
        self._series.total_lagrangian.append(snap.total_lagrangian)
        self._series.total_hamiltonian.append(snap.total_hamiltonian)
        self._series.total_kinetic.append(snap.total_kinetic)
        self._series.total_potential.append(snap.total_potential)

        if len(self._snapshots) >= 2:
            t_prev, snap_prev = self._snapshots[-2]
            dt = t - t_prev if t != t_prev else self._dt
            el = compute_euler_lagrange_residual(snap_prev, snap, dt=dt)
            self._series.el_rms_residual.append(el.rms_residual)
            self._series.stationarity_quality.append(el.stationarity_quality)

            # Accumulate action
            self._cumulative_action += dt * snap.total_lagrangian
            self._series.action_accumulated.append(self._cumulative_action)
        else:
            self._series.el_rms_residual.append(0.0)
            self._series.stationarity_quality.append(1.0)
            self._series.action_accumulated.append(0.0)

        return snap

    def report(self) -> VariationalTimeSeries:
        """Return collected time-series data."""
        return self._series

    @property
    def latest_snapshot(self) -> LagrangianSnapshot | None:
        """Return most recent snapshot, or *None*."""
        if not self._snapshots:
            return None
        return self._snapshots[-1][1]

    @property
    def action(self) -> float:
        """Accumulated action to date."""
        return self._cumulative_action

    @property
    def all_snapshots(self) -> list[LagrangianSnapshot]:
        """All recorded snapshots (for action computation)."""
        return [s for _, s in self._snapshots]


# ---------------------------------------------------------------------------
#  Comprehensive variational analysis
# ---------------------------------------------------------------------------


def compute_variational_suite(G: Any) -> dict[str, Any]:
    """Compute the variational analysis from one local collection of fields.

    The caller must hold graph state fixed during the call. Returned maps are
    detached; capture is not atomic with concurrent evolution.

    Returns
    -------
    dict[str, Any]
        - ``lagrangian_snapshot``: full :class:`LagrangianSnapshot`
        - ``critical_points``: threshold analysis
        - ``grammar_stationarity``: U1-U6 variational interpretation
        - ``poisson_bracket_geometric``: {K_φ, J_φ} estimate
        - ``poisson_bracket_potential``: {Φ_s, J_ΔNFR} estimate
        - ``virial_ratio``: T/V (= 1 at virialisation)
    """
    fields = _capture_structural_fields(G)
    snap = _lagrangian_snapshot_from_fields(fields)
    crit = _potential_critical_points_from_fields(
        fields.phi_s, fields.grad_phi, fields.k_phi
    )
    grammar = _grammar_stationarity_from_snapshot(snap)

    pb_geo = compute_poisson_bracket_estimate(snap.conjugate_geometric)
    pb_pot = compute_poisson_bracket_estimate(snap.conjugate_potential)

    virial = (
        snap.total_kinetic / snap.total_potential
        if snap.total_potential > 1e-12
        else float("inf")
    )

    return {
        "lagrangian_snapshot": snap,
        "critical_points": crit,
        "grammar_stationarity": grammar,
        "poisson_bracket_geometric": pb_geo,
        "poisson_bracket_potential": pb_pot,
        "virial_ratio": virial,
    }


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

__all__ = [
    #  Data structures
    "ConjugatePair",
    "LagrangianSnapshot",
    "EulerLagrangeResidual",
    "SymplecticCheck",
    "GrammarStationarityAnalysis",
    "CriticalPointAnalysis",
    "VariationalTimeSeries",
    # Core Lagrangian
    "compute_kinetic_density",
    "compute_potential_density",
    "compute_lagrangian_density",
    "compute_hamiltonian_density",
    "compute_interaction_density",
    # Sector translation
    "translate_sectors",
    # Phase space
    "identify_conjugate_pairs",
    "compute_phase_space_volume",
    "compute_poisson_bracket_estimate",
    # Snapshot & tracking
    "capture_lagrangian_snapshot",
    "compute_euler_lagrange_residual",
    "compute_action_functional",
    "VariationalTracker",
    # Symplectic & canonical checks
    "check_symplectic_preservation",
    "classify_operator_canonical",
    # Grammar as stationarity
    "analyze_grammar_stationarity",
    # Critical points
    "analyze_potential_critical_points",
    # Comprehensive suite
    "compute_variational_suite",
]

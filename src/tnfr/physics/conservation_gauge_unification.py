r"""Finite-snapshot conservation/gauge compatibility diagnostics.

The module aggregates algebraic action-energy consistency, covariance of the
auxiliary local pure-gauge coordinates, a separate global oscillator-rotation
check in the ambient substrate, and selected graph-field measurements. These
checks are complementary diagnostics. They do not derive
the nodal equation from the auxiliary action, prove energy conservation along
engine trajectories, or validate grammar U1-U6.

``compute_grammar_symmetry_mapping`` retains its historical public name and
six-entry layout. Each entry now states whether the available evidence can
actually assess the rule. A graph snapshot can measure current edge-phase
compatibility for U3. U1, U2, U4, and U5 require operator history or hierarchy
context. U3 also requires explicit finite phase data on every current edge.
U6 requires a reference structural-potential field to measure drift.

``is_unified`` is retained for compatibility as an alias for a finite
aggregate-diagnostic pass. It is never a universal unification theorem or a
grammar-validity certificate.

STATUS: CANONICAL DIAGNOSTIC INTERFACE — scoped finite observations.

References
----------
- Action functional: src/tnfr/physics/variational.py
- Conservation laws: src/tnfr/physics/conservation.py
- Gauge structure: src/tnfr/physics/gauge.py
- Nodal-pulse foundation: src/tnfr/riemann/nodal_pulse.py
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)  [TNFR.pdf §2.1]
- Grammar: theory/UNIFIED_GRAMMAR_RULES.md (U1-U6)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ..mathematics.unified_numerical import np

from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA
from ..constants.canonical import DELTA_PHI_MAX, U6_STRUCTURAL_POTENTIAL_LIMIT

# Canonical fields
from .canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)

# Conservation layer
from .conservation import (
    ConservationSnapshot,
    compute_energy_functional,
    compute_noether_charge,
)
from .extended import compute_dnfr_flux, compute_phase_current

# Gauge layer
from .gauge import (
    GaugeInvarianceResult,
    capture_gauge_snapshot,
    compute_covariant_derivative_magnitude,
    compute_yang_mills_action,
    verify_gauge_invariance,
)

# Variational layer
from .variational import (
    capture_lagrangian_snapshot,
    compute_phase_space_volume,
    compute_poisson_bracket_estimate,
    identify_conjugate_pairs,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "GrammarSymmetryMapping",
    "ActionEnergyConsistency",
    "NoetherGaugeDecomposition",
    "GaugeConservationCoupling",
    "SymplecticGaugeCompatibility",
    "ConservationGaugeUnification",
    # Functions
    "compute_grammar_symmetry_mapping",
    "verify_action_energy_consistency",
    "compute_noether_gauge_decomposition",
    "compute_gauge_conservation_coupling",
    "verify_symplectic_gauge_compatibility",
    "run_conservation_gauge_unification",
]


def _as_finite_real(value: Any) -> float | None:
    """Return a finite real scalar, rejecting booleans and overflow."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        scalar = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _finite_node_field_issue(
    G: Any, aliases: tuple[str, ...], field_name: str
) -> str:
    """Return why a required per-node scalar field is unavailable."""
    for node, data in G.nodes(data=True):
        alias = next((key for key in aliases if key in data), None)
        if alias is None or _as_finite_real(data[alias]) is None:
            return f"finite {field_name} is required at node {node!r}"
    return ""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrammarSymmetryMapping:
    """Historical rule/correspondence row with explicit applicability.

    ``is_satisfied`` is retained for compatibility. It is meaningful only
    when ``is_applicable`` is true; unassessed rows set it to false and expose
    ``assessment_status='not_assessed'``.

    Attributes
    ----------
    rule : str
        Grammar rule identifier (e.g. 'U1', 'U2', ..., 'U6').
    symmetry_type : str
        Historical correspondence label ('boundary', 'stability', 'gauge',
        'topological', 'hierarchical', or 'confinement').
    conservation_law : str
        Historical correspondence, now phrased without implying a theorem.
    variational_role : str
        Role in the variational formulation.
    is_satisfied : bool
        Result of the scoped diagnostic when applicable; false otherwise.
    diagnostic_value : float
        Quantitative measure of (non-)satisfaction. 0 = perfect.
    is_applicable : bool
        Whether the supplied snapshot/context can assess this rule.
    assessment_status : str
        ``'pass'``, ``'fail'``, or ``'not_assessed'``.
    assessment_scope : str
        Evidence boundary for the row.
    required_evidence : str
        Missing evidence needed when the row is not assessed.
    """

    rule: str
    symmetry_type: str
    conservation_law: str
    variational_role: str
    is_satisfied: bool
    diagnostic_value: float
    is_applicable: bool = True
    assessment_status: str = "pass"
    assessment_scope: str = "current_graph_snapshot"
    required_evidence: str = ""


@dataclass(frozen=True)
class ActionEnergyConsistency:
    """Verify the algebraic equality of two same-snapshot energy read-outs.

    Both implementations sum the same five squared fields. Their agreement is
    an implementation identity, not evidence that the engine follows the
    auxiliary action or conserves this energy over time.

    Attributes
    ----------
    hamiltonian_variational : float
        H = Σ_i [T(i) + V(i)] from variational.py.
    energy_conservation : float
        E = ½Σ_i ℰ(i) from conservation.py.
    relative_error : float
        |H - E| / max(|H|, |E|, ε).
    is_consistent : bool
        True if relative_error < tolerance.
    total_kinetic : float
        T = Σ_i T(i) (transport sector energy).
    total_potential : float
        V = Σ_i V(i) (configuration sector energy).
    kinetic_fraction : float
        T / H (virial ratio).
    """

    hamiltonian_variational: float
    energy_conservation: float
    relative_error: float
    is_consistent: bool
    total_kinetic: float
    total_potential: float
    kinetic_fraction: float


@dataclass(frozen=True)
class NoetherGaugeDecomposition:
    """Decompose one snapshot into structural-charge and U(1) field read-outs.

    The historical Noether/gauge names identify the intended correspondence.
    This object does not prove time-translation symmetry or conservation of Q.

    Attributes
    ----------
    noether_charge : float
        Q = Σ_i [Φ_s(i) + K_φ(i)] (structural charge, NOT gauge-invariant).
    energy_functional : float
        E = ½Σ_i ℰ(i) (gauge-invariant total energy).
    gauge_invariant_energy : float
        Same as energy_functional, emphasising gauge invariance.
    mean_psi_magnitude : float
        Mean magnitude of the diagnostic complex coordinate.
    mean_gauge_curvature : float
        Legacy name for the mean absolute cycle-closure residual. The bundled
        connection is pure gauge, so this is numerical closure error rather
        than an independent curvature field.
    yang_mills_action : float
        Legacy name for the squared cycle-closure residual penalty returned by
        :func:`compute_yang_mills_action`.
    matter_action : float
        Legacy name for the finite covariant-difference energy ``Σ |D Ψ|²``.
    noether_gauge_ratio : float
        Historical snapshot coordinate ``|Q| / E``; no sector-separation
        theorem follows from it.
    decomposition_quality : float
        Legacy name for ``1/(1 + std(energy_density)/mean(energy_density))``.
    """

    noether_charge: float
    energy_functional: float
    gauge_invariant_energy: float
    mean_psi_magnitude: float
    mean_gauge_curvature: float
    yang_mills_action: float
    matter_action: float
    noether_gauge_ratio: float
    decomposition_quality: float

    @property
    def mean_cycle_closure_residual(self) -> float:
        """Accurate alias for ``mean_gauge_curvature``."""
        return self.mean_gauge_curvature

    @property
    def squared_cycle_closure_penalty(self) -> float:
        """Accurate alias for the legacy ``yang_mills_action`` field."""
        return self.yang_mills_action

    @property
    def covariant_difference_energy(self) -> float:
        """Accurate alias for the legacy ``matter_action`` field."""
        return self.matter_action

    @property
    def energy_density_uniformity_score(self) -> float:
        """Accurate alias for the legacy ``decomposition_quality`` field."""
        return self.decomposition_quality


@dataclass(frozen=True)
class GaugeConservationCoupling:
    """Quantify shared-field and U(1)-rotation diagnostics.

    The gauge sector (Ψ = K_φ + iJ_φ) and the conservation sector
    (ρ = Φ_s + K_φ) share the K_φ field.  This coupling means:

    A constant rotation of the geometric coordinate changes the historical
    structural-charge snapshot while preserving its quadratic norm. A separate
    seeded check tests the node-dependent pure-gauge coordinate covariance.
    Neither calculation is a temporal conservation or Ward-identity test.

    Attributes
    ----------
    shared_field_fraction : float
        Legacy name for the mean ratio ``|K_φ|/|ρ|`` where ``|ρ|>0``. The
        value can exceed one when ``Φ_s`` and ``K_φ`` cancel.
    gauge_charge_sensitivity : float
        ``|ΔQ|`` under one constant coordinate rotation.
    energy_gauge_invariance : float
        Maximum energy-density deviation in the seeded local pure-gauge
        coordinate-covariance check.
    geometric_sector_energy : float
        E_geo = ½Σ_i |Ψ(i)|² (geometric sector contribution to H).
    potential_sector_energy : float
        E_pot = ½Σ_i [Φ_s² + |∇φ|² + J_ΔNFR²] (potential sector).
    sector_coupling_parameter : float
        κ = E_geo / (E_geo + E_pot) — normalised geometric sector weight.
    ward_gauge_consistency : float
        Legacy name for the gauge-invariance pass score. It contains no Ward
        residual and does not couple gauge invariance to conservation.
    """

    shared_field_fraction: float
    gauge_charge_sensitivity: float
    energy_gauge_invariance: float
    geometric_sector_energy: float
    potential_sector_energy: float
    sector_coupling_parameter: float
    ward_gauge_consistency: float

    @property
    def shared_field_ratio(self) -> float:
        """Accurate alias for the legacy ``shared_field_fraction`` field."""
        return self.shared_field_fraction

    @property
    def local_pure_gauge_invariance_score(self) -> float:
        """Accurate alias for the legacy ``ward_gauge_consistency`` field."""
        return self.ward_gauge_consistency


@dataclass(frozen=True)
class SymplecticGaugeCompatibility:
    """Check a constant oscillator rotation in the auxiliary substrate.

    The symplectic 2-form:
        ω = Σ_i [dK_φ(i) ∧ dJ_φ(i) + dΦ_s(i) ∧ dJ_ΔNFR(i)]

    Under one constant angle, the geometric pair rotates and
    ``ω_geo = Σ dK_φ ∧ dJ_φ`` is invariant. This is a global oscillator
    symmetry of the declared harmonic model. It is distinct from, and does not
    assess, the node-dependent pure-gauge rephasing used by ``gauge.py``.

    Attributes
    ----------
    geometric_volume : float
        Legacy snapshot-product statistic ``Σ_i |K_φ(i) J_φ(i)|``; not a
        symplectic volume.
    potential_volume : float
        Legacy snapshot-product statistic ``Σ_i |Φ_s(i) J_ΔNFR(i)|``.
    total_volume : float
        Sum of the two legacy snapshot-product statistics.
    geometric_poisson : float
        Legacy normalized snapshot covariance; not a Poisson bracket.
    potential_poisson : float
        Legacy normalized snapshot covariance; not a Poisson bracket.
    gauge_volume_invariance : float
        Legacy field name for the symplectic pullback residual of the constant
        two-coordinate rotation.
    is_compatible : bool
        True if that global rotation preserves the auxiliary two-form.
    snapshot_product_change : float
        Relative change of ``Σ K_φ J_φ``. This coordinate statistic is
        allowed to change and is not used for ``is_compatible``.
    transformation_scope : str
        Explicitly identifies the tested constant global rotation.
    local_gauge_assessed : bool
        False; local pure-gauge covariance is tested elsewhere.
    """

    geometric_volume: float
    potential_volume: float
    total_volume: float
    geometric_poisson: float
    potential_poisson: float
    gauge_volume_invariance: float
    is_compatible: bool
    snapshot_product_change: float = 0.0
    transformation_scope: str = "global_constant_oscillator_rotation"
    local_gauge_assessed: bool = False

    @property
    def geometric_snapshot_product(self) -> float:
        """Accurate alias for the legacy ``geometric_volume`` field."""
        return self.geometric_volume

    @property
    def potential_snapshot_product(self) -> float:
        """Accurate alias for the legacy ``potential_volume`` field."""
        return self.potential_volume

    @property
    def geometric_normalized_covariance(self) -> float:
        """Accurate alias for the legacy ``geometric_poisson`` field."""
        return self.geometric_poisson

    @property
    def potential_normalized_covariance(self) -> float:
        """Accurate alias for the legacy ``potential_poisson`` field."""
        return self.potential_poisson


@dataclass(frozen=True)
class ConservationGaugeUnification:
    """Aggregate finite-snapshot conservation/gauge diagnostic result.

    This is the primary output of ``run_conservation_gauge_unification()``.

    Attributes
    ----------
    grammar_symmetry : list[GrammarSymmetryMapping]
        Six historical correspondence rows with applicability metadata.
    action_consistency : ActionEnergyConsistency
        H_variational = E_conservation verification.
    noether_gauge : NoetherGaugeDecomposition
        Historical charge and pure-gauge snapshot read-outs.
    gauge_conservation : GaugeConservationCoupling
        Shared-field and coordinate-rotation diagnostics.
    symplectic_gauge : SymplecticGaugeCompatibility
        Global oscillator-rotation check; local gauge is not assessed here.
    gauge_invariance : GaugeInvarianceResult
        Full gauge invariance verification.
    is_unified : bool
        Legacy alias for ``aggregate_diagnostic_passed``. It is not a
        universal proof or grammar certificate.
    unification_quality : float
        Aggregate finite-diagnostic score in [0, 1].
    summary : dict[str, Any]
        Human-readable summary of key results.
    aggregate_diagnostic_passed : bool
        Whether all applicable finite checks passed the aggregate policy.
    grammar_validated : bool
        Always false: this pipeline does not validate an operator word.
    assessed_grammar_rules, unassessed_grammar_rules : tuple[str, ...]
        Explicit coverage of the six historical mapping rows.
    diagnostic_scope : str
        Scope of the aggregate result.
    """

    grammar_symmetry: list[GrammarSymmetryMapping]
    action_consistency: ActionEnergyConsistency
    noether_gauge: NoetherGaugeDecomposition
    gauge_conservation: GaugeConservationCoupling
    symplectic_gauge: SymplecticGaugeCompatibility
    gauge_invariance: GaugeInvarianceResult
    is_unified: bool
    unification_quality: float
    summary: dict[str, Any]
    aggregate_diagnostic_passed: bool = False
    grammar_validated: bool = False
    assessed_grammar_rules: tuple[str, ...] = ()
    unassessed_grammar_rules: tuple[str, ...] = ()
    diagnostic_scope: str = "finite_snapshot_aggregate"


# ---------------------------------------------------------------------------
# 1. Scoped grammar/correspondence mapping
# ---------------------------------------------------------------------------


def compute_grammar_symmetry_mapping(
    G: Any,
    *,
    reference_graph: Any | None = None,
    reference_snapshot: ConservationSnapshot | None = None,
) -> list[GrammarSymmetryMapping]:
    """Return six legacy correspondence rows with scoped rule assessments.

    A current graph snapshot with explicit finite edge phases can assess U3
    edge-phase compatibility. It
    cannot assess U1 sequence boundaries, U2 debt/history, U4 trigger/handler
    context, or U5 nesting. U6 is assessed only when ``reference_graph`` or
    ``reference_snapshot`` supplies the earlier structural potential and the
    required current/reference fields are explicit and finite.

    This function is a diagnostic mapper, not a grammar validator. Use the
    grammar module with an operator word/history for grammar validation.

    Parameters
    ----------
    G : TNFRGraph
        Graph with canonical TNFR attributes.
    reference_graph : TNFRGraph, optional
        Earlier graph state used only to measure mean absolute U6 potential
        drift. Its node set must match ``G``.
    reference_snapshot : ConservationSnapshot, optional
        Earlier captured fields used instead of ``reference_graph`` for U6.
        Supplying both reference forms raises ``ValueError``.

    Returns
    -------
    list[GrammarSymmetryMapping]
        One entry per grammar rule.
    """
    if reference_graph is not None and reference_snapshot is not None:
        raise ValueError("provide at most one U6 reference source")

    # Phase differences for U3. A missing, non-real, or non-finite endpoint
    # phase makes this snapshot diagnostic unavailable rather than compatible.
    edge_list = list(G.edges())
    phase_nodes = {node for edge in edge_list for node in edge[:2]}
    phase_values: dict[Any, float] = {}
    u3_issue = ""
    for node in phase_nodes:
        data = G.nodes[node]
        present_alias = next((key for key in ALIAS_THETA if key in data), None)
        if present_alias is None:
            u3_issue = f"finite phase is required at edge node {node!r}"
            break
        phase = _as_finite_real(data[present_alias])
        if phase is None:
            u3_issue = f"finite phase is required at edge node {node!r}"
            break
        phase_values[node] = phase

    delta_phi_raw = G.graph.get("delta_phi_max", DELTA_PHI_MAX)
    parsed_delta_phi = _as_finite_real(delta_phi_raw)
    if not u3_issue:
        if parsed_delta_phi is None or parsed_delta_phi < 0.0:
            u3_issue = "delta_phi_max must be a finite nonnegative real number"
    delta_phi_max = (
        parsed_delta_phi if not u3_issue else float(DELTA_PHI_MAX)
    )

    max_phase_diff = 0.0
    for u, v in edge_list if not u3_issue else ():
        phi_u = phase_values[u]
        phi_v = phase_values[v]
        diff = abs(
            (phi_u - phi_v + math.pi) % (2 * math.pi) - math.pi
        )
        max_phase_diff = max(max_phase_diff, diff)

    # U6 requires a reference field because the policy concerns drift, not
    # the magnitude of Phi_s in one state.
    current_phi_s: dict[Any, float] | None = None
    reference_phi_s: dict[Any, float] | None = None
    reference_kind = ""
    u6_issue = ""
    if reference_snapshot is not None or reference_graph is not None:
        u6_issue = _finite_node_field_issue(G, ALIAS_DNFR, "delta_nfr")
    if reference_snapshot is not None and not u6_issue:
        candidate_reference = reference_snapshot.phi_s
        if not candidate_reference or any(
            _as_finite_real(value) is None
            for value in candidate_reference.values()
        ):
            u6_issue = "reference snapshot requires a nonempty finite phi_s field"
        else:
            current_phi_s = compute_structural_potential(G)
            reference_phi_s = candidate_reference
            reference_kind = "conservation_snapshot"
    elif reference_graph is not None and not u6_issue:
        u6_issue = _finite_node_field_issue(
            reference_graph, ALIAS_DNFR, "reference delta_nfr"
        )
        if not u6_issue:
            current_phi_s = compute_structural_potential(G)
            reference_phi_s = compute_structural_potential(reference_graph)
            reference_kind = "reference_graph"

    for field_name, field_values in (
        ("current phi_s", current_phi_s),
        ("reference phi_s", reference_phi_s),
    ):
        if field_values is not None and any(
            _as_finite_real(value) is None for value in field_values.values()
        ):
            u6_issue = f"{field_name} must contain only finite real values"
            break

    u6_applicable = bool(
        not u6_issue
        and current_phi_s
        and reference_phi_s is not None
        and set(current_phi_s) == set(reference_phi_s)
    )
    if (
        not u6_issue
        and (reference_snapshot is not None or reference_graph is not None)
        and not u6_applicable
    ):
        u6_issue = "matching nonempty current and reference phi_s fields are required"
    if u6_applicable:
        u6_drift = (
            float(
                np.mean(
                    [
                        abs(current_phi_s[node] - reference_phi_s[node])
                        for node in current_phi_s
                    ]
                )
            )
            if current_phi_s
            else 0.0
        )
        u6_satisfied = u6_drift < U6_STRUCTURAL_POTENTIAL_LIMIT
    else:
        u6_drift = 0.0
        u6_satisfied = False

    mappings = []

    # U1 needs the operator word and its execution context.
    mappings.append(
        GrammarSymmetryMapping(
            rule="U1",
            symmetry_type="boundary",
            conservation_law="No conservation inference from a graph snapshot",
            variational_role="Historical boundary-condition correspondence",
            is_satisfied=False,
            diagnostic_value=0.0,
            is_applicable=False,
            assessment_status="not_assessed",
            assessment_scope="operator_history_required",
            required_evidence="ordered operator word and execution context",
        )
    )

    # A finite energy at one time cannot assess U2 debt or convergence.
    mappings.append(
        GrammarSymmetryMapping(
            rule="U2",
            symmetry_type="stability",
            conservation_law=(
                "No Lyapunov or convergence inference from one energy value"
            ),
            variational_role="Historical stability correspondence",
            is_satisfied=False,
            diagnostic_value=0.0,
            is_applicable=False,
            assessment_status="not_assessed",
            assessment_scope="operator_history_required",
            required_evidence="operator roles, debt history, and trajectory",
        )
    )

    # U3 phase compatibility is directly observable on current graph edges.
    u3_applicable = not u3_issue
    u3_sat = u3_applicable and max_phase_diff <= delta_phi_max
    mappings.append(
        GrammarSymmetryMapping(
            rule="U3",
            symmetry_type="gauge",
            conservation_law="Current-edge phase-admissibility diagnostic",
            variational_role="Historical gauge-connection correspondence",
            is_satisfied=u3_sat,
            diagnostic_value=(
                max(0.0, max_phase_diff - delta_phi_max)
                if u3_applicable
                else 0.0
            ),
            is_applicable=u3_applicable,
            assessment_status=(
                "pass" if u3_sat else "fail"
            ) if u3_applicable else "not_assessed",
            assessment_scope=(
                "current_graph_edge_phase_compatibility"
                if u3_applicable
                else "finite_edge_phase_data_required"
            ),
            required_evidence=u3_issue,
        )
    )

    # U4 needs trigger/handler and transformer recency context.
    mappings.append(
        GrammarSymmetryMapping(
            rule="U4",
            symmetry_type="topological",
            conservation_law="No topological conservation inference from a snapshot",
            variational_role="Historical bifurcation/topology correspondence",
            is_satisfied=False,
            diagnostic_value=0.0,
            is_applicable=False,
            assessment_status="not_assessed",
            assessment_scope="operator_history_required",
            required_evidence="trigger, handler, transformer, and recency history",
        )
    )

    # A flat graph snapshot has no declared parent/child hierarchy for U5.
    mappings.append(
        GrammarSymmetryMapping(
            rule="U5",
            symmetry_type="hierarchical",
            conservation_law="No hierarchy conservation inference from a flat snapshot",
            variational_role="Historical scale-factorisation correspondence",
            is_satisfied=False,
            diagnostic_value=0.0,
            is_applicable=False,
            assessment_status="not_assessed",
            assessment_scope="hierarchy_context_required",
            required_evidence="declared parent/child EPI hierarchy and normalization",
        )
    )

    # U6 is the finite two-snapshot mean |Delta Phi_s| policy.
    mappings.append(
        GrammarSymmetryMapping(
            rule="U6",
            symmetry_type="confinement",
            conservation_law="Finite mean absolute structural-potential drift policy",
            variational_role=(
                "Historical confinement correspondence; not an energy bound"
            ),
            is_satisfied=u6_satisfied,
            diagnostic_value=u6_drift,
            is_applicable=u6_applicable,
            assessment_status=(
                ("pass" if u6_satisfied else "fail")
                if u6_applicable
                else "not_assessed"
            ),
            assessment_scope=(
                f"two_snapshot_phi_s_drift:{reference_kind}"
                if u6_applicable
                else "phi_s_reference_required"
            ),
            required_evidence=(
                "" if u6_applicable
                else u6_issue or "matching reference graph or conservation snapshot"
            ),
        )
    )

    return mappings


# ---------------------------------------------------------------------------
# 2. Action-Energy consistency: H_variational = E_conservation
# ---------------------------------------------------------------------------


def verify_action_energy_consistency(
    G: Any,
    *,
    tolerance: float = 1e-10,
) -> ActionEnergyConsistency:
    """Compare two algebraically equivalent same-snapshot energy read-outs.

    Both implementations sum the same fields:
    - H = Σ_i [T(i) + V(i)] (variational Hamiltonian)
    - E = ½Σ_i ℰ(i) (structural energy diagnostic)

    They agree because T(i)+V(i) = ½ℰ(i) by construction. This check does not
    establish an Euler-Lagrange bridge or temporal energy conservation.

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Maximum acceptable relative error.

    Returns
    -------
    ActionEnergyConsistency
    """
    snap = capture_lagrangian_snapshot(G)
    E_cons = compute_energy_functional(G)

    H_var = snap.total_hamiltonian
    T = snap.total_kinetic
    V = snap.total_potential

    denom = max(abs(H_var), abs(E_cons), 1e-15)
    rel_err = abs(H_var - E_cons) / denom

    return ActionEnergyConsistency(
        hamiltonian_variational=H_var,
        energy_conservation=E_cons,
        relative_error=rel_err,
        is_consistent=rel_err < tolerance,
        total_kinetic=T,
        total_potential=V,
        kinetic_fraction=T / max(H_var, 1e-15),
    )


# ---------------------------------------------------------------------------
# 3. Historical Noether/gauge-named snapshot decomposition
# ---------------------------------------------------------------------------


def compute_noether_gauge_decomposition(G: Any) -> NoetherGaugeDecomposition:
    """Compute structural-charge and U(1)-invariant snapshot diagnostics.

    ``Q`` retains its historical Noether-like name, while its conservation
    must be measured along a declared trajectory. U(1) invariance of the
    quadratic field norms is an algebraic same-snapshot statement.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    NoetherGaugeDecomposition
    """
    Q = compute_noether_charge(G)
    E = compute_energy_functional(G)

    # Gauge snapshot for internal sector
    gauge = capture_gauge_snapshot(G)

    psi_mags = list(gauge.psi_magnitude.values())
    mean_psi = float(np.mean(psi_mags)) if psi_mags else 0.0

    closure_values = list(gauge.curvature.values())
    mean_closure_residual = (
        float(np.mean([abs(value) for value in closure_values]))
        if closure_values
        else 0.0
    )

    # Historical API name: a square penalty on pure-gauge cycle closure error.
    squared_closure_penalty = compute_yang_mills_action(G)

    # Historical API name: finite covariant-difference energy.
    cov_mag = compute_covariant_derivative_magnitude(G)
    covariant_difference_energy = sum(m**2 for m in cov_mag.values())

    # Historical snapshot ratio; no decomposition theorem is inferred.
    ratio = abs(Q) / max(E, 1e-15)

    # Legacy decomposition-quality field: inverse normalized dispersion of
    # the snapshot energy-density map. It is not a sector-separation measure.
    energy_vals = list(gauge.energy_density.values())
    energy_std = float(np.std(energy_vals)) if energy_vals else 0.0
    energy_mean = float(np.mean(energy_vals)) if energy_vals else 1e-15
    quality = 1.0 / (1.0 + energy_std / max(energy_mean, 1e-15))

    return NoetherGaugeDecomposition(
        noether_charge=Q,
        energy_functional=E,
        gauge_invariant_energy=E,  # E is gauge-invariant by construction
        mean_psi_magnitude=mean_psi,
        mean_gauge_curvature=mean_closure_residual,
        yang_mills_action=squared_closure_penalty,
        matter_action=covariant_difference_energy,
        noether_gauge_ratio=ratio,
        decomposition_quality=quality,
    )


# ---------------------------------------------------------------------------
# 4. Gauge-Conservation coupling
# ---------------------------------------------------------------------------


def compute_gauge_conservation_coupling(
    G: Any,
    *,
    gauge_angle: float = 0.1,
    seed: int = 42,
) -> GaugeConservationCoupling:
    """Quantify shared-field and gauge-rotation snapshot diagnostics.

    ``K_φ`` appears in both the historical structural-charge density
    ``ρ = Φ_s + K_φ`` and the complex coordinate ``Ψ = K_φ + iJ_φ``.
    The charge-sensitivity field uses one constant oscillator rotation. The
    energy-invariance field comes from a separate seeded local pure-gauge
    coordinate check. This algebra proves no engine symmetry or conservation.

    Parameters
    ----------
    G : TNFRGraph
    gauge_angle : float
        Angle (radians) for gauge sensitivity test.
    seed : int
        RNG seed for gauge invariance verification.

    Returns
    -------
    GaugeConservationCoupling
    """
    if not math.isfinite(gauge_angle):
        raise ValueError("gauge_angle must be finite")

    phi_s = compute_structural_potential(G)
    k_phi = compute_phase_curvature(G)
    j_phi = compute_phase_current(G)
    j_dnfr = compute_dnfr_flux(G)
    grad_phi = compute_phase_gradient(G)

    nodes = list(G.nodes())
    # Shared field fraction: |K_φ| / |ρ| averaged over nodes
    shared_fracs = []
    for n in nodes:
        rho_n = abs(phi_s.get(n, 0.0) + k_phi.get(n, 0.0))
        kphi_n = abs(k_phi.get(n, 0.0))
        if rho_n > 1e-15:
            shared_fracs.append(kphi_n / rho_n)
    shared_frac = float(np.mean(shared_fracs)) if shared_fracs else 0.0

    # Gauge charge sensitivity: ΔQ under rotation by gauge_angle
    # Under rotation: K_φ' = K_φ cos α − J_φ sin α
    # ΔK_φ = K_φ(cos α − 1) − J_φ sin α
    # ΔQ = Σ_i ΔK_φ(i) (Φ_s doesn't change)
    delta_Q = 0.0
    for n in nodes:
        kn = k_phi.get(n, 0.0)
        jn = j_phi.get(n, 0.0)
        delta_kphi = kn * (math.cos(gauge_angle) - 1.0) - jn * math.sin(gauge_angle)
        delta_Q += delta_kphi

    # Energy gauge invariance check
    gauge_result = verify_gauge_invariance(G, seed=seed)
    energy_dev = gauge_result.energy_max_deviation

    # Sector energies
    e_geo = 0.0  # geometric sector: ½Σ |Ψ|² = ½Σ(K_φ² + J_φ²)
    e_pot = 0.0  # potential sector: ½Σ(Φ_s² + |∇φ|² + J_ΔNFR²)
    for n in nodes:
        kn = k_phi.get(n, 0.0)
        jn = j_phi.get(n, 0.0)
        e_geo += 0.5 * (kn**2 + jn**2)

        fn = phi_s.get(n, 0.0)
        gn = grad_phi.get(n, 0.0)
        dn = j_dnfr.get(n, 0.0)
        e_pot += 0.5 * (fn**2 + gn**2 + dn**2)

    total_e = e_geo + e_pot
    kappa = e_geo / max(total_e, 1e-15)

    # Legacy field name: this is a gauge-invariance pass score only. No Ward
    # residual or temporal conservation observation enters this function.
    ward_gauge = 1.0 if gauge_result.is_invariant else 0.5

    return GaugeConservationCoupling(
        shared_field_fraction=shared_frac,
        gauge_charge_sensitivity=abs(delta_Q),
        energy_gauge_invariance=energy_dev,
        geometric_sector_energy=e_geo,
        potential_sector_energy=e_pot,
        sector_coupling_parameter=kappa,
        ward_gauge_consistency=ward_gauge,
    )


# ---------------------------------------------------------------------------
# 5. Global oscillator rotation (legacy symplectic-gauge API name)
# ---------------------------------------------------------------------------


def verify_symplectic_gauge_compatibility(
    G: Any,
    *,
    gauge_angle: float = 0.1,
    tolerance: float = 1e-8,
) -> SymplecticGaugeCompatibility:
    """Check a constant global oscillator rotation against the auxiliary form.

    The symplectic 2-form ω = Σ dK_φ ∧ dJ_φ + dΦ_s ∧ dJ_ΔNFR
    splits into:
    - ω_geo = Σ dK_φ ∧ dJ_φ (the gauge-active sector)
    - ω_pot = Σ dΦ_s ∧ dJ_ΔNFR (gauge-singlet sector)

    Under ``Ψ → e^{iα}Ψ`` with one constant ``α``, the geometric pair
    rotates and ``dK ∧ dJ`` is preserved by SO(2). This is a symmetry of
    the harmonic oscillator coordinates, not a test of local gauge dynamics.

    Parameters
    ----------
    G : TNFRGraph
    gauge_angle : float
        Rotation angle for invariance test.
    tolerance : float
        Maximum acceptable symplectic pullback residual.

    Returns
    -------
    SymplecticGaugeCompatibility
    """
    if not math.isfinite(gauge_angle):
        raise ValueError("gauge_angle must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    geo_pair, pot_pair = identify_conjugate_pairs(G)

    vol_geo = compute_phase_space_volume(geo_pair)
    vol_pot = compute_phase_space_volume(pot_pair)

    pb_geo = compute_poisson_bracket_estimate(geo_pair)
    pb_pot = compute_poisson_bracket_estimate(pot_pair)

    # The product sum is retained only as a legacy snapshot statistic. It is
    # not a symplectic volume and need not be invariant under rotation.
    signed_geo_before = 0.0
    for n in list(G.nodes()):
        q_n = geo_pair.q.get(n, 0.0)
        p_n = geo_pair.p.get(n, 0.0)
        signed_geo_before += q_n * p_n

    # After rotation
    signed_geo_after = 0.0
    ca, sa = math.cos(gauge_angle), math.sin(gauge_angle)
    for n in list(G.nodes()):
        q_n = geo_pair.q.get(n, 0.0)
        p_n = geo_pair.p.get(n, 0.0)
        q_rot = q_n * ca - p_n * sa
        p_rot = q_n * sa + p_n * ca
        signed_geo_after += q_rot * p_rot

    delta_signed = abs(signed_geo_after - signed_geo_before)
    product_change = delta_signed / max(abs(signed_geo_before), 1e-15)

    rotation = np.array([[ca, -sa], [sa, ca]], dtype=float)
    omega = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)
    pullback_residual = float(np.linalg.norm(rotation.T @ omega @ rotation - omega))
    determinant_residual = abs(float(np.linalg.det(rotation)) - 1.0)
    symplectic_residual = max(pullback_residual, determinant_residual)

    return SymplecticGaugeCompatibility(
        geometric_volume=vol_geo,
        potential_volume=vol_pot,
        total_volume=vol_geo + vol_pot,
        geometric_poisson=pb_geo,
        potential_poisson=pb_pot,
        gauge_volume_invariance=symplectic_residual,
        is_compatible=symplectic_residual <= tolerance,
        snapshot_product_change=product_change,
        transformation_scope="global_constant_oscillator_rotation",
        local_gauge_assessed=False,
    )


# ---------------------------------------------------------------------------
# 6. Aggregate diagnostic pipeline
# ---------------------------------------------------------------------------


def run_conservation_gauge_unification(
    G: Any,
    *,
    gauge_seed: int = 42,
    tolerance: float = 1e-10,
    reference_graph: Any | None = None,
    reference_snapshot: ConservationSnapshot | None = None,
) -> ConservationGaugeUnification:
    """Run the aggregate finite-snapshot conservation/gauge diagnostics.

    The legacy name and ``is_unified`` field are retained for compatibility.
    The result combines applicable snapshot checks; it does not validate a
    grammar word or prove a universal conservation-gauge unification.

    Parameters
    ----------
    G : TNFRGraph
        Graph with canonical TNFR attributes.
    gauge_seed : int
        RNG seed for gauge invariance verification.
    tolerance : float
        Tolerance for consistency checks.
    reference_graph, reference_snapshot : optional
        Mutually exclusive earlier state used to make the U6 mean-potential-
        drift observation applicable. See
        :func:`compute_grammar_symmetry_mapping`.

    Returns
    -------
    ConservationGaugeUnification
        Aggregate diagnostics with explicit scope and coverage metadata.
    """
    # 1. Historical grammar/correspondence rows with applicability metadata.
    grammar = compute_grammar_symmetry_mapping(
        G,
        reference_graph=reference_graph,
        reference_snapshot=reference_snapshot,
    )
    assessed_rules = tuple(m.rule for m in grammar if m.is_applicable)
    unassessed_rules = tuple(m.rule for m in grammar if not m.is_applicable)
    n_satisfied = sum(1 for m in grammar if m.is_applicable and m.is_satisfied)

    # 2. Action-Energy consistency
    action_cons = verify_action_energy_consistency(G, tolerance=tolerance)

    # 3. Historical charge plus pure-gauge snapshot coordinates
    noether_gauge = compute_noether_gauge_decomposition(G)

    # 4. Shared-field and coordinate-rotation diagnostics
    gauge_cons = compute_gauge_conservation_coupling(G, seed=gauge_seed)

    # 5. Separate global oscillator-rotation check
    symp_gauge = verify_symplectic_gauge_compatibility(G, tolerance=tolerance)

    # 6. Local pure-gauge coordinate covariance
    gauge_inv = verify_gauge_invariance(G, seed=gauge_seed)

    # Aggregate finite diagnostics. Applicable U3/U6 observations can affect
    # this aggregate, but they never turn it into full grammar validation.
    applicable_rule_checks = [m.is_satisfied for m in grammar if m.is_applicable]
    checks = [
        action_cons.is_consistent,  # H_var = E_cons
        gauge_inv.is_invariant,  # gauge invariance
        symp_gauge.is_compatible,  # global oscillator rotation
        *applicable_rule_checks,
    ]
    quality_scores = [
        1.0 - min(action_cons.relative_error * 1e8, 1.0),  # action consistency
        1.0 if gauge_inv.is_invariant else 0.5,  # gauge invariance
        1.0 if symp_gauge.is_compatible else 0.5,  # symplectic
        noether_gauge.decomposition_quality,  # energy-density uniformity
        gauge_cons.ward_gauge_consistency,  # legacy local-covariance pass score
        *[1.0 if passed else 0.0 for passed in applicable_rule_checks],
    ]
    quality = float(np.mean(quality_scores))
    aggregate_passed = all(checks) and quality > 0.8
    diagnostic_scope = (
        "two_snapshot_aggregate_with_u6_reference"
        if "U6" in assessed_rules
        else "current_snapshot_aggregate"
    )

    # Summary
    summary = {
        "grammar_rules_satisfied": f"{n_satisfied}/{len(assessed_rules)} assessed",
        "grammar_rules_assessed": assessed_rules,
        "grammar_rules_unassessed": unassessed_rules,
        "grammar_validation_applicable": False,
        "grammar_validated": False,
        "H_variational": action_cons.hamiltonian_variational,
        "E_conservation": action_cons.energy_conservation,
        "H_E_relative_error": action_cons.relative_error,
        "T_kinetic": action_cons.total_kinetic,
        "V_potential": action_cons.total_potential,
        "kinetic_fraction": action_cons.kinetic_fraction,
        "historical_structural_charge_snapshot": noether_gauge.noether_charge,
        "noether_charge_Q": noether_gauge.noether_charge,
        "gauge_invariant_energy": noether_gauge.gauge_invariant_energy,
        "mean_cycle_closure_residual": noether_gauge.mean_cycle_closure_residual,
        "squared_cycle_closure_penalty": (
            noether_gauge.squared_cycle_closure_penalty
        ),
        "covariant_difference_energy": noether_gauge.covariant_difference_energy,
        "energy_density_uniformity_score": (
            noether_gauge.energy_density_uniformity_score
        ),
        "yang_mills_action": noether_gauge.yang_mills_action,
        "mean_psi_magnitude": noether_gauge.mean_psi_magnitude,
        "geometric_sector_energy": gauge_cons.geometric_sector_energy,
        "potential_sector_energy": gauge_cons.potential_sector_energy,
        "sector_coupling_kappa": gauge_cons.sector_coupling_parameter,
        "shared_K_phi_fraction": gauge_cons.shared_field_fraction,
        "shared_K_phi_ratio": gauge_cons.shared_field_ratio,
        "gauge_charge_sensitivity": gauge_cons.gauge_charge_sensitivity,
        "energy_gauge_invariance_dev": gauge_cons.energy_gauge_invariance,
        "local_pure_gauge_invariance_score": (
            gauge_cons.local_pure_gauge_invariance_score
        ),
        "symplectic_volume_geo": symp_gauge.geometric_volume,
        "symplectic_volume_pot": symp_gauge.potential_volume,
        "poisson_bracket_geo": symp_gauge.geometric_poisson,
        "poisson_bracket_pot": symp_gauge.potential_poisson,
        "geometric_snapshot_product": symp_gauge.geometric_snapshot_product,
        "potential_snapshot_product": symp_gauge.potential_snapshot_product,
        "geometric_normalized_covariance": (
            symp_gauge.geometric_normalized_covariance
        ),
        "potential_normalized_covariance": (
            symp_gauge.potential_normalized_covariance
        ),
        "global_oscillator_symplectic_residual": (
            symp_gauge.gauge_volume_invariance
        ),
        "global_oscillator_snapshot_product_change": (
            symp_gauge.snapshot_product_change
        ),
        "global_oscillator_scope": symp_gauge.transformation_scope,
        "local_gauge_assessed_by_symplectic_check": symp_gauge.local_gauge_assessed,
        "unification_quality": quality,
        "is_unified": aggregate_passed,
        "aggregate_diagnostic_passed": aggregate_passed,
        "diagnostic_scope": diagnostic_scope,
        "narrative": (
            "Aggregate finite diagnostics passed; grammar was not validated"
            if aggregate_passed
            else "One or more aggregate diagnostics failed; grammar was not validated"
        ),
    }

    return ConservationGaugeUnification(
        grammar_symmetry=grammar,
        action_consistency=action_cons,
        noether_gauge=noether_gauge,
        gauge_conservation=gauge_cons,
        symplectic_gauge=symp_gauge,
        gauge_invariance=gauge_inv,
        is_unified=aggregate_passed,
        unification_quality=quality,
        summary=summary,
        aggregate_diagnostic_passed=aggregate_passed,
        grammar_validated=False,
        assessed_grammar_rules=assessed_rules,
        unassessed_grammar_rules=unassessed_rules,
        diagnostic_scope=diagnostic_scope,
    )

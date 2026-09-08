"""Simple TNFR SDK — Maximum Power, Minimum Complexity.

The simplified TNFR API designed for 90% of use cases, now with full
Structural Field Tetrad, finite structural-balance monitoring, grammar-aware
dynamics, and research-grade telemetry.

**DESIGN PRINCIPLE**:
- **Intuitive**: Natural method names that read like English
- **Chainable**: Fluent interface for rapid prototyping
- **Integrated**: Core TNFR dynamics and diagnostics behind one interface
- **Research-grade**: Structural Field Tetrad + scoped balance diagnostics

**USAGE EXAMPLES**::

    # Instant network creation
    net = TNFR.create(20)  # 20 nodes

    # Chain operations
    results = TNFR.create(10).ring().evolve(5).results()

    # Auto-optimization
    optimized = TNFR.create(15).random(0.3).auto_optimize()

    # Full Structural Field Tetrad
    tetrad = net.tetrad()
    # -> {'phi_s': {...}, 'grad_phi': {...}, 'k_phi': {...}, 'xi_c': float, ...}

    # Finite structural-balance monitoring
    conservation = net.conservation()
    # -> structural charge, candidate energy, sampled balance and legacy aliases

    # Unified diagnostic fields and invariant read-outs
    telemetry = net.telemetry()

    # Grammar-aware evolution
    net.evolve_grammar_aware(steps=10)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..alias import get_attr
from ..constants import DEFAULTS
from ..constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..constants.canonical import HIGH_COHERENCE_THRESHOLD as COHERENCE_STRONG
from ..constants.canonical import ZHIR_THRESHOLD_XI_CANONICAL
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..metrics.coherence import compute_coherence
from ..metrics.common import (
    finite_mean_absolute,
    is_structural_equilibrium,
    structural_coherence,
)
from ..metrics.sense_index import compute_Si
from ..operators.nodal_equation import compute_d2epi_dt2
from ..physics.mutation_trigger import (
    MutationTriggerCertificate,
    MutationTriggerInputError,
    certify_mutation_trigger,
)
from ..types import BEPIProtocol, scalarize_epi

# TNFR core imports
from ..structural import create_nfr
from ._topology import grid_edges, nonnegative_integer, probability as validate_probability
from ._topology import ring_edges, small_world_edges

# Canonical telemetry marks (AGENTS.md §7) -- heuristic cuts, not fitted:
#   C(t) > COHERENCE_STRONG (π/(π+1) ~0.7585, selected telemetry cut)
#   Si   > SENSE_INDEX_EXCELLENT (~0.8)               -> excellent sense index
SENSE_INDEX_EXCELLENT = 0.8

# Canonical graph-node equilibrium tolerance (EPS_DNFR_STABLE ~ 1e-3): the same
# cut the engine's stability tracker uses (metrics/coherence). The per-node
# micro-NFR equilibrium (nodal_state) and the region macro-NFR equilibrium
# (Network.nfr) share this single canonical tolerance.
_EPS_DNFR_STABLE_DEFAULT = float(DEFAULTS["EPS_DNFR_STABLE"])


def _raw_alias_value(data: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    """Read the first present alias without coercing certificate inputs."""

    for key in aliases:
        if key in data:
            return data[key]
    return None


def _certificate_epi_value(value: Any) -> Any:
    """Scalarize canonical BEPI form while preserving invalid scalar inputs."""

    serialized_bepi = isinstance(value, Mapping) and {
        "continuous",
        "discrete",
        "grid",
    }.issubset(value)
    if isinstance(value, BEPIProtocol) or serialized_bepi:
        return scalarize_epi(value)
    return value


def _certify_sdk_mutation_trigger(
    *,
    node: Any | None,
    current_epi: Any,
    nu_f: Any,
    delta_nfr: Any,
    xi: Any,
    epi_time_history: Any = None,
    epi_history: Any = None,
    legacy_epi_history: Any = None,
) -> MutationTriggerCertificate:
    """Build the pure trigger certificate and translate input errors for the SDK."""

    try:
        return certify_mutation_trigger(
            current_epi=_certificate_epi_value(current_epi),
            nu_f=nu_f,
            delta_nfr=delta_nfr,
            xi=xi,
            epi_time_history=epi_time_history,
            epi_history=epi_history,
            legacy_epi_history=legacy_epi_history,
        )
    except MutationTriggerInputError as exc:
        context: dict[str, Any] = {
            "field": exc.field,
            "value": repr(exc.value),
            "reason": exc.reason,
        }
        if node is not None:
            context["node"] = node
        raise TNFRValueError(
            "Invalid nodal Mutation-trigger input.",
            context=context,
            suggestion=(
                "Provide finite real EPI, nu_f, DeltaNFR channels and a "
                "finite nonnegative Mutation threshold."
            ),
        ) from exc


from ..operators.word_execution import (
    preflight_network_mutation_sequence as _preflight_sdk_mutation_sequence,
    run_network_sequence as _run_network_sequence,
)
# Private aliases are retained for downstream SDK compatibility; new code should
# import the neutral public functions from tnfr.operators.


def _first_interpolated_crossing(
    samples: list[float], threshold: float
) -> float | None:
    """Estimate the first upward threshold crossing between sampled steps.

    Returning a fractional step prevents two distinct crossings within the
    same sampling interval from being reported as simultaneous. No crossing
    is inferred beyond the observed trajectory.
    """

    if not samples:
        return None
    if samples[0] >= threshold:
        return 0.0
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]
        if previous < threshold <= current:
            fraction = (threshold - previous) / (current - previous)
            return float(index - 1) + fraction
    return None


try:
    # Availability probe for the self-optimization engine (capability flag
    # only; auto_optimize uses the grammar-aware stabilizer path directly).
    import tnfr.dynamics.self_optimizing_engine  # noqa: F401

    _HAS_OPTIMIZATION = True
except ImportError:
    _HAS_OPTIMIZATION = False

# Structural Field Tetrad (CANONICAL)
try:
    from ..physics.fields import (
        compute_complex_geometric_field_arrays,
        compute_dnfr_flux,
        compute_emergent_fields,
        compute_phase_current,
        compute_phase_curvature,
        compute_phase_gradient,
        compute_structural_potential,
        compute_tensor_invariants,
        compute_unified_telemetry,
        estimate_coherence_length,
    )

    _HAS_FIELDS = True
except ImportError:
    _HAS_FIELDS = False

# Finite structural-balance diagnostics (historically Noether-like)
try:
    from ..physics.conservation import (
        ConservationTracker,
        compute_energy_functional,
        compute_lyapunov_derivative,
        compute_noether_charge,
    )

    _HAS_CONSERVATION = True
except ImportError:
    _HAS_CONSERVATION = False

# Auxiliary ambient symplectic model initialized from extracted graph fields
try:
    from ..physics.symplectic_substrate import (
        background_potential,
        extract_phase_space_point,
        substrate_hamiltonian,
        verify_canonical_structure,
    )

    _HAS_SUBSTRATE = True
except ImportError:
    _HAS_SUBSTRATE = False

# Structural Integrity Monitor
try:
    from ..physics.integrity import MonitorMode, StructuralIntegrityMonitor

    _HAS_INTEGRITY = True
except ImportError:
    _HAS_INTEGRITY = False

# Grammar-aware dynamics
try:
    from ..operators.grammar_dynamics import filter_candidates

    _HAS_GRAMMAR_DYNAMICS = True
except ImportError:
    _HAS_GRAMMAR_DYNAMICS = False


# Canonical factorization bridge (optional dependency path)
try:
    from ..factorization import factorize as canonical_factorize

    _HAS_FACTORIZATION = True
except Exception:
    _HAS_FACTORIZATION = False


# Canonical primality bridge (optional dependency path)
try:
    from ..primality import analyze as canonical_primality_analyze

    _HAS_PRIMALITY = True
except Exception:
    _HAS_PRIMALITY = False


@dataclass
class TetradSnapshot:
    """Structural Field Tetrad snapshot — four canonical fields.

    Captures the complete four-field diagnostic read-out
    (Phi_s, |grad_phi|, K_phi, xi_C) at a single point in time. These lossy
    summaries do not reconstruct the complete TNFR graph state.

    Attributes
    ----------
    phi_s : dict[Any, float]
        Structural potential per node.
    grad_phi : dict[Any, float]
        Phase gradient per node.
    k_phi : dict[Any, float]
        Phase curvature per node.
    xi_c : float
        Coherence length (global scalar).
    j_phi : dict[Any, float]
        Phase current per node.
    j_dnfr : dict[Any, float]
        DNFR flux per node.
    """

    phi_s: dict[Any, float] = field(default_factory=dict)
    grad_phi: dict[Any, float] = field(default_factory=dict)
    k_phi: dict[Any, float] = field(default_factory=dict)
    xi_c: float = float("nan")
    j_phi: dict[Any, float] = field(default_factory=dict)
    j_dnfr: dict[Any, float] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary of tetrad fields."""
        n = len(self.phi_s)
        if n == 0:
            return "Tetrad: empty"
        phi_s_mean = sum(self.phi_s.values()) / n
        grad_mean = sum(self.grad_phi.values()) / n
        k_mean = sum(abs(v) for v in self.k_phi.values()) / n
        return (
            f"Phi_s={phi_s_mean:.4f}, |grad_phi|={grad_mean:.4f}, "
            f"|K_phi|={k_mean:.4f}, xi_C={self.xi_c:.4f} (N={n})"
        )

    def is_safe(self) -> dict[str, bool]:
        """Check canonical safety thresholds for all tetrad fields.

        Returns dict with keys: phi_s_safe, grad_phi_safe, k_phi_safe,
        xi_c_safe, and overall.
        """
        from ..constants.canonical import (
            GRAD_PHI_CANONICAL_THRESHOLD,
            K_PHI_CANONICAL_THRESHOLD,
            PHI_S_VON_KOCH_THRESHOLD,
        )

        phi_s_safe = (
            all(abs(v) < PHI_S_VON_KOCH_THRESHOLD for v in self.phi_s.values())
            if self.phi_s
            else True
        )
        grad_safe = (
            all(v < GRAD_PHI_CANONICAL_THRESHOLD for v in self.grad_phi.values())
            if self.grad_phi
            else True
        )
        k_safe = (
            all(abs(v) < K_PHI_CANONICAL_THRESHOLD for v in self.k_phi.values())
            if self.k_phi
            else True
        )
        xi_safe = not np.isnan(self.xi_c) if np.isfinite(self.xi_c) else True
        return {
            "phi_s_safe": phi_s_safe,
            "grad_phi_safe": grad_safe,
            "k_phi_safe": k_safe,
            "xi_c_safe": xi_safe,
            "overall": phi_s_safe and grad_safe and k_safe,
        }


@dataclass
class ConservationReport:
    """Finite-trajectory structural conservation diagnostics.

    Captures the historically named tetrad charge candidate, energy candidate,
    observed finite-step energy change, and balance-quality telemetry.
    """

    noether_charge: float = 0.0
    energy: float = 0.0
    lyapunov_stable: bool = True
    lyapunov_derivative: float = 0.0
    conservation_quality: float = 0.0
    sample_available: bool = False

    @property
    def structural_charge(self) -> float:
        """Accurately named view of the historical ``noether_charge`` field."""
        return self.noether_charge

    @property
    def candidate_energy(self) -> float:
        """Structural energy evaluated as a Lyapunov candidate."""
        return self.energy

    @property
    def candidate_energy_nonincreasing(self) -> bool | None:
        """Finite-step energy result, or ``None`` before an interval exists."""
        if not self.sample_available:
            return None
        if not np.isfinite(self.lyapunov_derivative):
            return None
        return self.lyapunov_derivative <= 0.0

    @property
    def candidate_energy_within_numerical_tolerance(self) -> bool | None:
        """Historical Lyapunov classification, including its tolerance."""
        return self.lyapunov_stable if self.sample_available else None

    @property
    def candidate_energy_derivative(self) -> float | None:
        """Sampled candidate-energy derivative, unavailable on first capture."""
        return self.lyapunov_derivative if self.sample_available else None

    @property
    def balance_quality(self) -> float | None:
        """Finite-step balance quality, unavailable on first capture."""
        return self.conservation_quality if self.sample_available else None

    def summary(self) -> str:
        """One-line scoped balance and candidate-energy summary."""
        if not self.sample_available:
            return (
                f"Q_candidate={self.noether_charge:.4f}, "
                f"E_candidate={self.energy:.4f}, interval=UNSAMPLED"
            )
        exact_trend = self.candidate_energy_nonincreasing
        if exact_trend is None:
            trend = "UNDEFINED"
        else:
            trend = "NON-INCREASING" if exact_trend else "INCREASING"
        return (
            f"Q_candidate={self.noether_charge:.4f}, "
            f"E_candidate={self.energy:.4f}, "
            f"candidate dE/dt={self.lyapunov_derivative:.4f} ({trend}), "
            f"balance_quality={self.conservation_quality:.3f}"
        )


def _empty_balance_alert_report(diagnostic_scope: str) -> dict[str, Any]:
    """Build one consistent unsampled balance-alert result."""
    return {
        "sample_available": False,
        "alerts_detected": False,
        "alert_count": 0,
        "alert_types": [],
        "nodes_alerted": [],
        "severity": 0.0,
        # Backward-compatible grammar-shaped keys remain deliberately empty.
        "violations_detected": False,
        "violation_count": 0,
        "violation_types": [],
        "nodes_violating": [],
        "diagnostic_scope": diagnostic_scope,
        "grammar_validation_applicable": False,
        "grammar_validated": False,
        "grammar_rules_assessed": (),
    }


@dataclass
class SymplecticReport:
    """Diagnostics for the auxiliary ambient symplectic substrate model.

    Extracted graph fields initialize a point in the ambient phase space
    ``R^(4N)``. The report evaluates that model's harmonic Hamiltonian,
    held-fixed configuration background, Liouville divergence, and canonical
    symplectic structure. It does not derive this model from the nodal flow.
    """

    phase_space_dimension: int = 0
    hamiltonian: float = 0.0
    background_potential: float = 0.0
    liouville_divergence: float = 0.0
    is_valid_manifold: bool = False

    def summary(self) -> str:
        """One-line symplectic substrate summary."""
        valid_str = "VALID" if self.is_valid_manifold else "INVALID"
        return (
            f"dim={self.phase_space_dimension}, "
            f"H_sub={self.hamiltonian:.4f}, "
            f"U={self.background_potential:.4f}, "
            f"div(X_H)={self.liouville_divergence:.2e} ({valid_str})"
        )


@dataclass
class FactorizationReport:
    """SDK report for canonical TNFR factorization.

    Bridges `tnfr.factorization.factorize()` with SDK-level telemetry and
    an optional, explicitly heuristic comparison to a caller-supplied network.
    """

    n: int
    modulus: int
    candidate_factors: list[int] = field(default_factory=list)
    tnfr_certified_factors: list[int] = field(default_factory=list)
    coherence_score: float = 0.0
    arithmetic_delta_nfr: float = 0.0
    arithmetic_epi: float = 0.0
    arithmetic_nu_f: float = 0.0
    certificate_path: str | None = None
    partition_manifest_path: str | None = None
    operator_strategy_plan: dict[str, Any] | None = None
    spectral: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    network_synergy: dict[str, Any] | None = None  # legacy field name

    def summary(self) -> str:
        certified = len(self.tnfr_certified_factors)
        return (
            f"n={self.n}, candidates={len(self.candidate_factors)}, "
            f"certified={certified}, coherence={self.coherence_score:.3f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to plain dict for JSON/reporting pipelines."""
        return {
            "n": int(self.n),
            "modulus": int(self.modulus),
            "candidate_factors": list(self.candidate_factors),
            "tnfr_certified_factors": list(self.tnfr_certified_factors),
            "coherence_score": float(self.coherence_score),
            "arithmetic_delta_nfr": float(self.arithmetic_delta_nfr),
            "arithmetic_epi": float(self.arithmetic_epi),
            "arithmetic_nu_f": float(self.arithmetic_nu_f),
            "certificate_path": self.certificate_path,
            "partition_manifest_path": self.partition_manifest_path,
            "operator_strategy_plan": self.operator_strategy_plan,
            "spectral": self.spectral,
            "telemetry": self.telemetry,
            "network_synergy": self.network_synergy,
        }


@dataclass
class PrimalityReport:
    """SDK report for canonical TNFR primality analysis.

    Bridges `tnfr.primality.analyze()` with SDK-level telemetry and
    an optional, explicitly heuristic comparison to a caller-supplied network.
    """

    n: int
    is_prime: bool
    delta_nfr: float
    tolerance: float
    components: dict[str, Any] = field(default_factory=dict)
    triad: dict[str, Any] = field(default_factory=dict)
    network_synergy: dict[str, Any] | None = None  # legacy field name

    def summary(self) -> str:
        status = "prime" if self.is_prime else "not-prime"
        return f"n={self.n}, status={status}, delta_nfr={self.delta_nfr:.6g}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to plain dict for JSON/reporting pipelines."""
        return {
            "n": int(self.n),
            "is_prime": bool(self.is_prime),
            "delta_nfr": float(self.delta_nfr),
            "tolerance": float(self.tolerance),
            "components": self.components,
            "triad": self.triad,
            "network_synergy": self.network_synergy,
        }


@dataclass
class NodalStateReport:
    """Node-level nodal dynamics snapshot based on ∂EPI/∂t = νf·ΔNFR.

    ``expected_depi_dt`` is the nodal-equation prediction.  The retained
    ``near_bifurcation`` name is a legacy alias of ``predicted_crossed``; it is
    not an observed Mutation gate.  ``mutation_threshold_satisfied`` mirrors
    only the observed strict threshold gate and does not assess U4b grammar or
    operator execution readiness.
    """

    node: Any
    epi: float
    nu_f: float
    delta_nfr: float
    coherence: float
    phase: float
    expected_depi_dt: float
    d2epi_dt2: float
    degree: int
    equilibrium: bool
    active: bool
    near_bifurcation: bool
    observed_depi_dt: float | None = None
    predicted_crossed: bool | None = None
    observed_crossed: bool | None = None
    evidence_available: bool = False
    evidence_valid: bool = False
    source: str | None = None
    time_basis: str | None = None
    physical_time_resolved: bool = False
    current_endpoint_matches_state: bool | None = None
    reason: str | None = None
    rate_gap: float | None = None
    mutation_threshold_satisfied: bool = False

    def __post_init__(self) -> None:
        """Keep the legacy prediction alias coherent for direct construction."""

        if self.predicted_crossed is None:
            self.predicted_crossed = bool(self.near_bifurcation)
        else:
            self.predicted_crossed = bool(self.predicted_crossed)
            self.near_bifurcation = self.predicted_crossed

    def summary(self) -> str:
        state = "active" if self.active else "inactive"
        eq = "equilibrium" if self.equilibrium else "driven"
        return (
            f"node={self.node}, {state}, {eq}, "
            f"∂EPI/∂t={self.expected_depi_dt:.4g}, ∂²EPI/∂t²={self.d2epi_dt2:.4g}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "epi": float(self.epi),
            "nu_f": float(self.nu_f),
            "delta_nfr": float(self.delta_nfr),
            "coherence": float(self.coherence),
            "phase": float(self.phase),
            "expected_depi_dt": float(self.expected_depi_dt),
            "d2epi_dt2": float(self.d2epi_dt2),
            "degree": int(self.degree),
            "equilibrium": bool(self.equilibrium),
            "active": bool(self.active),
            "near_bifurcation": bool(self.near_bifurcation),
            "observed_depi_dt": (
                None
                if self.observed_depi_dt is None
                else float(self.observed_depi_dt)
            ),
            "predicted_crossed": bool(self.predicted_crossed),
            "observed_crossed": self.observed_crossed,
            "evidence_available": bool(self.evidence_available),
            "evidence_valid": bool(self.evidence_valid),
            "source": self.source,
            "time_basis": self.time_basis,
            "physical_time_resolved": bool(self.physical_time_resolved),
            "current_endpoint_matches_state": self.current_endpoint_matches_state,
            "reason": self.reason,
            "rate_gap": None if self.rate_gap is None else float(self.rate_gap),
            "mutation_threshold_satisfied": bool(
                self.mutation_threshold_satisfied
            ),
        }


@dataclass
class NodalDynamicsReport:
    """Global nodal-equation prediction report for study and diagnostics.

    Its total coherence aggregates pressure and predicted EPI-rate magnitudes
    before applying the nonlinear constitutive kernel. ``mean_local_coherence``
    remains available as a distinct descriptive statistic.
    """

    nodes: dict[Any, NodalStateReport] = field(default_factory=dict)
    equilibrium_tolerance: float = _EPS_DNFR_STABLE_DEFAULT
    bifurcation_threshold: float = ZHIR_THRESHOLD_XI_CANONICAL

    def _channel_means(self) -> tuple[float, float]:
        """Return mean pressure and predicted-rate magnitudes for this scan."""

        values = self.nodes.values()
        mean_pressure = finite_mean_absolute(
            (state.delta_nfr for state in values), name="scan dnfr"
        )
        mean_rate = finite_mean_absolute(
            (state.expected_depi_dt for state in self.nodes.values()),
            name="scan predicted depi",
        )
        return mean_pressure, mean_rate

    def total_coherence(self) -> float:
        """Return canonical C over this scan's predicted nodal channels."""

        if not self.nodes:
            return 0.0
        mean_pressure, mean_rate = self._channel_means()
        return float(structural_coherence(mean_pressure, mean_rate))

    def mean_local_coherence(self) -> float:
        """Return the distinct mean of already-reduced local coherences."""

        if not self.nodes:
            return 0.0
        return finite_mean_absolute(
            (state.coherence for state in self.nodes.values()),
            name="scan local coherence",
        )

    def summary(self) -> str:
        n = len(self.nodes)
        if n == 0:
            return "Nodal dynamics: empty"
        values = list(self.nodes.values())
        active = sum(1 for state in values if state.active)
        equilibrium = sum(1 for state in values if state.equilibrium)
        bifurcation = sum(1 for state in values if state.near_bifurcation)
        _, mean_abs_rate = self._channel_means()
        coherence = self.total_coherence()
        return (
            f"NodalDynamics(N={n}, active={active}, equilibrium={equilibrium}, "
            f"bifurcation={bifurcation}, C={coherence:.3f}, "
            f"mean|∂EPI/∂t|={mean_abs_rate:.4g})"
        )

    def top_pressure_nodes(self, k: int = 5) -> list[NodalStateReport]:
        """Nodes with largest |∂EPI/∂t| (highest nodal drive)."""
        ranked = sorted(
            self.nodes.values(), key=lambda s: abs(s.expected_depi_dt), reverse=True
        )
        return ranked[: max(int(k), 0)]

    def near_equilibrium_nodes(self) -> list[NodalStateReport]:
        """Nodes whose pressure and predicted rate meet the fixed-point cut."""
        return [state for state in self.nodes.values() if state.equilibrium]

    def to_dict(self) -> dict[str, Any]:
        values = list(self.nodes.values())
        n = len(values)
        mean_abs_pressure, mean_abs_rate = self._channel_means()
        max_abs_rate = max(
            (abs(state.expected_depi_dt) for state in values), default=0.0
        )
        return {
            "equilibrium_tolerance": float(self.equilibrium_tolerance),
            "bifurcation_threshold": float(self.bifurcation_threshold),
            "nodes": {
                str(node): report.to_dict() for node, report in self.nodes.items()
            },
            "aggregate": {
                "count": n,
                "active_count": sum(1 for s in values if s.active),
                "equilibrium_count": sum(1 for s in values if s.equilibrium),
                "bifurcation_count": sum(
                    1 for state in values if state.near_bifurcation
                ),
                "coherence": self.total_coherence(),
                "mean_local_coherence": self.mean_local_coherence(),
                "mean_abs_dnfr": float(mean_abs_pressure),
                "mean_abs_depi_dt": float(mean_abs_rate),
                "max_abs_depi_dt": float(max_abs_rate),
                "depi_dt_source": "nodal_equation_prediction",
            },
        }


def _build_factorization_report(
    raw_result: Any,
    network: "Network | None" = None,
) -> FactorizationReport:
    """Convert canonical factorization output into SDK-native report."""
    candidate_factors = list(getattr(raw_result, "candidate_factors", []) or [])
    certified = list(getattr(raw_result, "tnfr_certified_factors", []) or [])
    coherence_score = float(getattr(raw_result, "coherence_score", 0.0) or 0.0)
    delta_nfr = float(getattr(raw_result, "arithmetic_delta_nfr", 0.0) or 0.0)
    arithmetic_epi = float(getattr(raw_result, "arithmetic_epi", 0.0) or 0.0)
    arithmetic_nu_f = float(getattr(raw_result, "arithmetic_nu_f", 0.0) or 0.0)
    n = int(getattr(raw_result, "n", 0) or 0)
    modulus = int(getattr(raw_result, "modulus", 0) or 0)

    telemetry = {
        "phi_s": float(getattr(raw_result, "phi_s", 0.0) or 0.0),
        "phase_gradient": float(getattr(raw_result, "phase_gradient", 0.0) or 0.0),
        "phase_curvature": float(getattr(raw_result, "phase_curvature", 0.0) or 0.0),
        "coherence_length": float(getattr(raw_result, "coherence_length", 0.0) or 0.0),
        "coherence_score": coherence_score,
        "delta_nfr": delta_nfr,
        "epi": arithmetic_epi,
        "nu_f": arithmetic_nu_f,
    }

    spectral = {
        "laplacian_gap": float(getattr(raw_result, "laplacian_gap", 0.0) or 0.0),
        "fft_backend": getattr(raw_result, "fft_backend", None),
        "node_count": int(getattr(raw_result, "node_count", 0) or 0),
        "edge_count": int(getattr(raw_result, "edge_count", 0) or 0),
        "notes": getattr(raw_result, "notes", ""),
    }

    network_synergy: dict[str, Any] | None = None
    if network is not None:
        net_coherence = float(network.coherence())
        net_si = float(network.sense_index())
        coherence_alignment = max(0.0, 1.0 - abs(net_coherence - coherence_score))
        nodal_drive = abs(arithmetic_nu_f * delta_nfr)
        drive_score = nodal_drive / (1.0 + nodal_drive)
        # Exploratory equal-weight comparison. No canonical map relates the
        # caller-supplied graph to this arithmetic factorization instance.
        alignment_score = (coherence_alignment + drive_score) / 2.0
        size_divisible_candidates = [
            int(f)
            for f in candidate_factors
            if f > 1 and len(network.G.nodes()) % int(f) == 0
        ]
        network_synergy = {
            "status": "heuristic_alignment",
            "scope": (
                "independent network supplied by caller; no canonical "
                "cross-domain map"
            ),
            "weights": {"coherence_alignment": 0.5, "drive_score": 0.5},
            "network_nodes": len(network.G.nodes()),
            "network_coherence": net_coherence,
            "network_sense_index": net_si,
            "coherence_alignment": coherence_alignment,
            "nodal_drive": nodal_drive,
            "drive_score": drive_score,
            "alignment_score": alignment_score,
            "size_divisible_candidates": sorted(set(size_divisible_candidates)),
            # Backward-compatible names; neither value certifies resonance.
            "synergy_index": alignment_score,
            "topology_resonance_factors": sorted(set(size_divisible_candidates)),
            "legacy_name_warning": (
                "synergy_index/topology_resonance_factors are historical names"
            ),
        }

    return FactorizationReport(
        n=n,
        modulus=modulus,
        candidate_factors=candidate_factors,
        tnfr_certified_factors=certified,
        coherence_score=coherence_score,
        arithmetic_delta_nfr=delta_nfr,
        arithmetic_epi=arithmetic_epi,
        arithmetic_nu_f=arithmetic_nu_f,
        certificate_path=getattr(raw_result, "certificate_path", None),
        partition_manifest_path=getattr(raw_result, "partition_manifest_path", None),
        operator_strategy_plan=getattr(raw_result, "operator_strategy_plan", None),
        spectral=spectral,
        telemetry=telemetry,
        network_synergy=network_synergy,
    )


def _build_primality_report(
    n: int,
    raw_result: dict[str, Any],
    *,
    tolerance: float,
    network: "Network | None" = None,
) -> PrimalityReport:
    """Convert canonical primality output into SDK-native report."""
    is_prime = bool(raw_result.get("is_prime", False))
    delta_nfr = float(raw_result.get("delta_nfr", float("inf")))
    components = dict(raw_result.get("components", {}) or {})
    triad = dict(raw_result.get("triad", {}) or {})

    network_synergy: dict[str, Any] | None = None
    if network is not None:
        net_coherence = float(network.coherence())
        net_si = float(network.sense_index())
        local_coherence = float(triad.get("local_coherence", 0.0) or 0.0)
        coherence_alignment = max(0.0, 1.0 - abs(net_coherence - local_coherence))
        pressure_ratio = abs(delta_nfr) / max(float(tolerance), 1e-15)
        pressure_score = 1.0 / (1.0 + pressure_ratio)
        # Exploratory equal-weight comparison. It deliberately excludes the
        # old label-conditioned ``prime_resonance`` term, which injected the
        # primality answer into its own score.
        alignment_score = (coherence_alignment + pressure_score) / 2.0
        network_synergy = {
            "status": "heuristic_alignment",
            "scope": (
                "independent network supplied by caller; no canonical "
                "cross-domain map"
            ),
            "weights": {"coherence_alignment": 0.5, "pressure_score": 0.5},
            "network_nodes": len(network.G.nodes()),
            "network_coherence": net_coherence,
            "network_sense_index": net_si,
            "local_coherence": local_coherence,
            "coherence_alignment": coherence_alignment,
            "pressure_ratio": pressure_ratio,
            "pressure_score": pressure_score,
            "alignment_score": alignment_score,
            "synergy_index": alignment_score,
            "legacy_name_warning": "synergy_index is a historical field name",
        }

    return PrimalityReport(
        n=int(n),
        is_prime=is_prime,
        delta_nfr=delta_nfr,
        tolerance=float(tolerance),
        components=components,
        triad=triad,
        network_synergy=network_synergy,
    )


@dataclass
class Results:
    """TNFR Results with full structural field support.

    Contains essential metrics plus optional structural field tetrad,
    conservation diagnostics, and unified telemetry for research-grade
    analysis.
    """

    coherence: float
    sense_index: float
    nodes: int
    edges: int
    density: float
    avg_phase: float
    tetrad: TetradSnapshot | None = None
    conservation: ConservationReport | None = None
    unified_fields: dict[str, Any] | None = None

    def summary(self) -> str:
        """One-line summary of results."""
        coherence = (
            float(self.coherence) if hasattr(self.coherence, "item") else self.coherence
        )
        sense_index = (
            float(self.sense_index)
            if hasattr(self.sense_index, "item")
            else self.sense_index
        )
        density = float(self.density) if hasattr(self.density, "item") else self.density

        return (
            f"C={coherence:.3f}, Si={sense_index:.3f}, "
            f"N={self.nodes}, E={self.edges}, rho={density:.3f}"
        )

    def full_summary(self) -> str:
        """Multi-line summary including tetrad and conservation."""
        lines = [self.summary()]
        if self.tetrad is not None:
            lines.append(f"  Tetrad: {self.tetrad.summary()}")
            safety = self.tetrad.is_safe()
            if not safety["overall"]:
                unsafe = [k for k, v in safety.items() if k != "overall" and not v]
                lines.append(f"  WARNING: Unsafe fields: {', '.join(unsafe)}")
        if self.conservation is not None:
            lines.append(f"  Balance diagnostics: {self.conservation.summary()}")
        return "\n".join(lines)

    def is_coherent(self) -> bool:
        """Quick coherence check (C(t) > strong mark ~0.75; AGENTS.md §7)."""
        return self.coherence > COHERENCE_STRONG

    def is_stable(self) -> bool:
        """Quick stability check (Si > excellent mark ~0.8; AGENTS.md §7)."""
        return self.sense_index > SENSE_INDEX_EXCELLENT

    def to_dict(self) -> dict[str, Any]:
        """Serialize results to a plain dictionary."""
        d: dict[str, Any] = {
            "coherence": float(self.coherence),
            "sense_index": float(self.sense_index),
            "nodes": self.nodes,
            "edges": self.edges,
            "density": float(self.density),
            "avg_phase": float(self.avg_phase),
        }
        if self.tetrad is not None:
            d["tetrad"] = {
                "phi_s": {str(k): float(v) for k, v in self.tetrad.phi_s.items()},
                "grad_phi": {str(k): float(v) for k, v in self.tetrad.grad_phi.items()},
                "k_phi": {str(k): float(v) for k, v in self.tetrad.k_phi.items()},
                "xi_c": (
                    float(self.tetrad.xi_c) if np.isfinite(self.tetrad.xi_c) else None
                ),
            }
        if self.conservation is not None:
            d["conservation"] = {
                "structural_charge": float(self.conservation.structural_charge),
                "candidate_energy": float(self.conservation.candidate_energy),
                "candidate_energy_nonincreasing": (
                    self.conservation.candidate_energy_nonincreasing
                ),
                "candidate_energy_within_numerical_tolerance": (
                    self.conservation.candidate_energy_within_numerical_tolerance
                ),
                "candidate_energy_derivative": (
                    self.conservation.candidate_energy_derivative
                ),
                "balance_quality": self.conservation.balance_quality,
                "sample_available": self.conservation.sample_available,
                # Backward-compatible historical keys.
                "noether_charge": float(self.conservation.noether_charge),
                "energy": float(self.conservation.energy),
                "lyapunov_stable": self.conservation.lyapunov_stable,
            }
        return d


class Network:
    """Core TNFR Network — Essential Operations + Advanced Telemetry.

    Simplified interface to TNFR networks with structural field tetrad,
    finite structural-balance diagnostics, grammar-aware dynamics, and
    operator-contract monitoring.
    """

    def __init__(self, graph: nx.Graph, name: str = "network", seed: int | None = None):
        """Initialize with a NetworkX graph.

        Parameters
        ----------
        graph : nx.Graph
            Graph whose nodes carry the canonical TNFR triad (EPI, νf, θ).
        name : str
            Network label.
        seed : int, optional
            Seed governing the stochastic topology builders (``random``,
            ``small_world``, ``scale_free``) so identical seeds reproduce
            identical trajectories (canonical invariant #6).
        """
        self.G = graph
        self.name = name
        self._seed = seed
        self._tracker: Any = None  # ConservationTracker (lazy)
        self._monitor: Any = None  # StructuralIntegrityMonitor (lazy)

    # === TOPOLOGY BUILDERS ===

    def ring(self) -> Network:
        """Connect nodes in a ring (each node to its two neighbours)."""
        nodes = list(self.G.nodes())
        self.G.add_edges_from(ring_edges(nodes))
        return self

    def complete(self) -> Network:
        """Connect every node to every other node (complete graph)."""
        nodes = list(self.G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                self.G.add_edge(u, v)
        return self

    def random(self, probability: float = 0.3, seed: int | None = None) -> Network:
        """Add random edges with given probability (Erdos-Renyi).

        Reproducible when *seed* (or the network seed set at ``create``) is
        supplied, so identical seeds reproduce identical topologies
        (canonical invariant #6).
        """
        probability = validate_probability(probability)
        s = seed if seed is not None else self._seed
        rng = np.random.RandomState(s)
        nodes = list(self.G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                if rng.random_sample() < probability:
                    self.G.add_edge(u, v)
        return self

    def star(self, center: Any = None) -> Network:
        """Connect every node to a single central hub (star topology)."""
        nodes = list(self.G.nodes())
        if center is None:
            if not nodes:
                return self
            center = nodes[0]
        if center not in self.G:
            raise TNFRValueError("star center must be an existing node")

        for node in nodes:
            if node != center:
                self.G.add_edge(center, node)
        return self

    def small_world(
        self, k: int = 4, p: float = 0.3, seed: int | None = None
    ) -> Network:
        """Create Watts-Strogatz small-world topology.

        Parameters
        ----------
        k : int
            Each node is connected to *k* nearest neighbours in ring.
        p : float
            Probability of rewiring each edge.
        seed : int, optional
            Random seed for reproducibility.
        """
        nodes = list(self.G)
        self.G.add_edges_from(small_world_edges(
            nodes, k, p, seed if seed is not None else self._seed,
        ))
        return self

    def scale_free(self, m: int = 2, seed: int | None = None) -> Network:
        """Create Barabasi-Albert scale-free topology.

        Parameters
        ----------
        m : int
            Number of edges to attach from a new node to existing nodes.
        seed : int, optional
            Random seed for reproducibility.
        """
        nodes = list(self.G)
        m = nonnegative_integer(m, "m")
        ba = nx.barabasi_albert_graph(
            len(nodes), m, seed=seed if seed is not None else self._seed,
        )
        self.G.add_edges_from((nodes[u], nodes[v]) for u, v in ba.edges())
        return self

    def grid(self, rows: int | None = None, cols: int | None = None) -> Network:
        """Create 2-D grid (lattice) topology.

        Insertion order fills a near-square rectangle when dimensions are
        omitted; the final row may be incomplete. An explicit dimension infers
        the other. Two explicit dimensions must accommodate every node. Like
        the other topology builders, this adds edges to existing support.
        """
        self.G.add_edges_from(grid_edges(list(self.G), rows, cols))
        return self

    def path(self) -> Network:
        """Connect nodes in a linear path (no closing edge)."""
        nodes = list(self.G.nodes())
        for i in range(len(nodes) - 1):
            self.G.add_edge(nodes[i], nodes[i + 1])
        return self

    # === EVOLUTION ===

    def evolve(
        self,
        steps: int = 5,
        sequence: str = "basic_activation",
        record: bool = False,
    ) -> Network:
        """Evolve the network by applying a canonical operator sequence.

        The named *sequence* is resolved to canonical structural operators.
        Execution is operator-major: each operator reaches every node before
        the next operator begins. When grammar retains the requested glyph,
        EN/IL/OZ/UM/RA/THOL and AL/SHA/VAL/NUL/ZHIR/NAV derive every target
        proposal from one immutable stage snapshot and commit atomically
        (two-phase Jacobi). IL binds pressure contraction and phase locking to
        that snapshot; OZ reduces local and propagated pressure there; THOL
        validates and merges child support and hierarchy; NAV binds per-node
        RNG progress, transactional seed resolution and one shared latency
        instant. REMESH and grammar replacements retain Gauss-Seidel reads in
        graph iteration order inside a complete stage rollback boundary.

        Every registered SDK word is resolved from NAMED_SEQUENCES and applied
        through the shared runner. In words that open with Emission, AL sources
        EPI through its operator contract rather than an ad hoc assignment.
        See tnfr.operators.run_network_sequence for the executable scheduling
        contract.

        Parameters
        ----------
        steps : int
            Number of times the sequence is applied to the whole network.
        sequence : str
            Name of a registered complete word in
            tnfr.sdk.fluent.NAMED_SEQUENCES, the executable source of truth.
            The default is "basic_activation" =
            [Emission, Reception, Coherence, Expansion, Resonance, Silence].
        record : bool
            When True, sample the canonical metrics step after each cycle so
            the per-step rhythm series (the pulse in motion: ``kuramoto_R``,
            ``C_steps``, ``phase_sync``, ``Si_mean``) are recorded into the
            graph history, readable via :meth:`history`. Default False (the
            fast path, no recording overhead).

        Returns
        -------
        Network
            self (for chaining).

        Raises
        ------
        ValueError
            If *steps* is not a non-negative integer. Boolean and real-valued
            counts are rejected rather than coerced.
        TNFRValueError
            If *sequence* is not a known canonical sequence; an invalid
            sequence is rejected (by name lookup or grammar validation)
            rather than silently ignored.
        """
        from .fluent import NAMED_SEQUENCES

        step_count = nonnegative_integer(steps, "steps")

        operator_names = NAMED_SEQUENCES.get(sequence)
        if operator_names is None:
            available = ", ".join(sorted(NAMED_SEQUENCES))
            raise TNFRValueError(
                f"Unknown sequence '{sequence}'.",
                context={
                    "requested": sequence,
                    "available": list(NAMED_SEQUENCES),
                },
                suggestion=f"Choose from: {available}",
            )
        # Network evolution from the shared SDK primitive.  The sub-threshold
        # birth phase (neighbour EPI below the centralized operational source
        # threshold) makes Reception correctly find no sources; that transient
        # warning is silenced on the high-level SDK surface.
        if record:
            # cycle-by-cycle so the canonical metrics step samples the rhythm
            # (kuramoto_R, C_steps, ...) after each cycle: the pulse in motion
            from ..metrics.core import _metrics_step

            for _ in range(step_count):
                _run_network_sequence(
                    self.G,
                    operator_names,
                    cycles=1,
                    suppress_birth_warnings=True,
                )
                _metrics_step(self.G)
        else:
            _run_network_sequence(
                self.G,
                operator_names,
                cycles=step_count,
                suppress_birth_warnings=True,
            )
        return self

    def auto_optimize(self) -> Network:
        """Auto-optimize: drive the network toward coherence.

        Self-optimization selects grammar-valid low-risk candidates. Coherence
        (IL) is the canonical U2 stabilizer; Reception, Resonance, and
        Coupling remain form/phase operations whose realized telemetry is
        measured rather than assumed to establish convergence.

        This delegates to grammar-aware evolution with a stabilizer-leaning
        candidate set. It does not claim a general gradient-descent or
        asymptotic-convergence theorem for arbitrary pressure laws.

        Returns
        -------
        Network
            self (for chaining).
        """
        return self.evolve_grammar_aware(
            steps=3,
            candidates=["coherence", "reception", "resonance", "coupling"],
        )

    def trajectory(
        self, cycles: int = 3, sequence: str = "basic_activation"
    ) -> list[dict[str, Any]]:
        """Record canonical metrics after EACH operator (fine-grained dynamics).

        Applies *sequence* with the shared operator-major mixed schedule for
        *cycles* repetitions and records C(t) and Si after every stage.
        When grammar retains the requested glyph, EN/IL/OZ/UM/RA/THOL and
        AL/SHA/VAL/NUL/ZHIR/NAV use atomic two-phase Jacobi stages; REMESH and
        grammar replacements use the documented Gauss-Seidel schedule.
        The trace exposes the implemented intra-sequence transient through
        the shared executor callback.

        Parameters
        ----------
        cycles : int
            Number of times the sequence is repeated.
        sequence : str
            Canonical sequence name (see
            :data:`tnfr.sdk.fluent.NAMED_SEQUENCES`).

        Returns
        -------
        list[dict]
            One snapshot per applied operator with keys ``step``,
            ``operator`` (name), ``coherence`` (C(t)) and ``sense_index``
            (Si).

        Raises
        ------
        ValueError
            If *cycles* is not a non-negative integer. Boolean and real-valued
            counts are rejected rather than coerced.
        TNFRValueError
            If *sequence* is not a registered complete word.

        Examples
        --------
        >>> hist = TNFR.create(12, seed=1).random(0.3).trajectory(2)
        >>> [(h["operator"], round(h["coherence"], 3)) for h in hist]  # doctest: +SKIP
        """
        from .fluent import NAMED_SEQUENCES

        cycle_count = nonnegative_integer(cycles, "cycles")

        operator_names = NAMED_SEQUENCES.get(sequence)
        if operator_names is None:
            available = ", ".join(sorted(NAMED_SEQUENCES))
            raise TNFRValueError(
                f"Unknown sequence '{sequence}'.",
                context={
                    "requested": sequence,
                    "available": list(NAMED_SEQUENCES),
                },
                suggestion=f"Choose from: {available}",
            )
        history: list[dict[str, Any]] = []

        def _record(operator_name: str) -> None:
            history.append(
                {
                    "step": len(history) + 1,
                    "operator": operator_name,
                    "coherence": self.coherence(),
                    "sense_index": self.sense_index(),
                }
            )

        _run_network_sequence(
            self.G,
            operator_names,
            cycles=cycle_count,
            suppress_birth_warnings=True,
            on_step=_record,
        )
        return history

    def history(self) -> dict[str, list[float]]:
        """The recorded pulse in motion: the canonical per-step rhythm series.

        After :meth:`evolve` with ``record=True`` the engine's metrics step
        records the canonical temporal series in the graph history. This
        surfaces the rhythm ones -- the collective resonance ``kuramoto_R``,
        the coherence ``C_steps``, the phase synchrony ``phase_sync`` and the
        mean sense index ``Si_mean`` -- as the engine actually recorded them
        over the run (the resonance forming, not a snapshot). Empty lists if
        the network was not evolved with ``record=True``.

        Returns
        -------
        dict[str, list[float]]
            ``kuramoto_R``, ``C_steps``, ``phase_sync``, ``Si_mean``.
        """
        hist = self.G.graph.get("history", {})
        keys = ("kuramoto_R", "C_steps", "phase_sync", "Si_mean")
        return {
            k: [float(x) for x in hist.get(k, [])] for k in keys
        }

    # === METRICS ===

    def coherence(self) -> float:
        """Current network coherence C(t) in [0,1]."""
        result = compute_coherence(self.G)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)

    def sense_index(self) -> float:
        """Current sense index Si in [0,1+]."""
        result = compute_Si(self.G)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)

    def density(self) -> float:
        """Network density [0,1]."""
        n = len(self.G.nodes())
        if n < 2:
            return 0.0
        return 2 * len(self.G.edges()) / (n * (n - 1))

    def avg_phase(self) -> float:
        """Average node phase [0, 2π]."""
        if not self.G.nodes():
            return 0.0
        phases = [get_attr(self.G.nodes[n], ALIAS_THETA, 0.0) for n in self.G.nodes()]
        result = np.mean(phases)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)

    # === NODAL DYNAMICS ===

    def nodal_state(
        self,
        node: Any,
        *,
        equilibrium_tolerance: float = _EPS_DNFR_STABLE_DEFAULT,
        bifurcation_threshold: float | None = None,
    ) -> NodalStateReport:
        """Return node-level state from the canonical nodal equation.

        Computes local triad state and derived dynamics without modifying the
        graph:

        - ``expected_depi_dt = nu_f * DeltaNFR`` is the instantaneous model
          prediction;
        - ``observed_depi_dt`` is a tri-state two-sample observation;
        - ``d2epi_dt2`` is read from EPI history by a pure finite difference.

        Mutation threshold evidence remains separate from U4b grammar and
        operator execution readiness.
        """
        if node not in self.G:
            raise TNFRValueError(
                f"Node '{node}' not found in network.",
                context={"node": node, "nodes": list(self.G.nodes())[:10]},
                suggestion="Use an existing node id from network.G.nodes().",
            )

        nd = self.G.nodes[node]
        epi_raw = _raw_alias_value(nd, ALIAS_EPI)
        nu_f_raw = _raw_alias_value(nd, ALIAS_VF)
        delta_nfr_raw = _raw_alias_value(nd, ALIAS_DNFR)
        phase = float(get_attr(nd, ALIAS_THETA, 0.0) or 0.0)

        xi_raw = (
            bifurcation_threshold
            if bifurcation_threshold is not None
            else self.G.graph.get("ZHIR_THRESHOLD_XI", ZHIR_THRESHOLD_XI_CANONICAL)
        )
        trigger = _certify_sdk_mutation_trigger(
            node=node,
            current_epi=epi_raw,
            nu_f=nu_f_raw,
            delta_nfr=delta_nfr_raw,
            xi=xi_raw,
            epi_time_history=nd.get("epi_time_history"),
            epi_history=nd.get("epi_history"),
            legacy_epi_history=nd.get("_epi_history"),
        )
        epi = trigger.current_epi
        nu_f = trigger.nu_f
        delta_nfr = trigger.delta_nfr
        d2epi_dt2 = float(compute_d2epi_dt2(self.G, node, store=False))

        return NodalStateReport(
            node=node,
            epi=epi,
            nu_f=nu_f,
            delta_nfr=delta_nfr,
            coherence=structural_coherence(
                delta_nfr, trigger.predicted_depi_dt
            ),
            phase=phase,
            expected_depi_dt=trigger.predicted_depi_dt,
            d2epi_dt2=d2epi_dt2,
            degree=int(self.G.degree(node)),
            equilibrium=is_structural_equilibrium(
                delta_nfr,
                trigger.predicted_depi_dt,
                eps_dnfr=float(equilibrium_tolerance),
                eps_depi=float(equilibrium_tolerance),
            ),
            active=trigger.capacity_active,
            near_bifurcation=trigger.predicted_crossed,
            observed_depi_dt=trigger.observed_depi_dt,
            predicted_crossed=trigger.predicted_crossed,
            observed_crossed=trigger.observed_crossed,
            evidence_available=trigger.evidence_available,
            evidence_valid=trigger.evidence_valid,
            source=trigger.source,
            time_basis=trigger.time_basis,
            physical_time_resolved=trigger.physical_time_resolved,
            current_endpoint_matches_state=trigger.current_endpoint_matches_state,
            reason=trigger.reason,
            rate_gap=trigger.rate_gap,
            mutation_threshold_satisfied=trigger.threshold_gate_satisfied,
        )

    def nodal_scan(
        self,
        nodes: list[Any] | None = None,
        *,
        equilibrium_tolerance: float = _EPS_DNFR_STABLE_DEFAULT,
        bifurcation_threshold: float | None = None,
    ) -> NodalDynamicsReport:
        """Scan nodal dynamics over a subset (or all) nodes.

        Useful for research diagnostics, bifurcation watch, and pressure maps.
        """
        xi_raw = (
            bifurcation_threshold
            if bifurcation_threshold is not None
            else self.G.graph.get("ZHIR_THRESHOLD_XI", ZHIR_THRESHOLD_XI_CANONICAL)
        )
        # Validate the shared threshold even for an empty scan.  The zero
        # channels are only a carrier for the certificate's scalar-domain
        # validation and do not create observational evidence.
        xi = _certify_sdk_mutation_trigger(
            node=None,
            current_epi=0.0,
            nu_f=0.0,
            delta_nfr=0.0,
            xi=xi_raw,
        ).xi

        target_nodes = list(self.G.nodes()) if nodes is None else list(nodes)
        report_nodes: dict[Any, NodalStateReport] = {}
        for node in target_nodes:
            if node not in self.G:
                continue
            report_nodes[node] = self.nodal_state(
                node,
                equilibrium_tolerance=equilibrium_tolerance,
                bifurcation_threshold=xi,
            )

        return NodalDynamicsReport(
            nodes=report_nodes,
            equilibrium_tolerance=float(equilibrium_tolerance),
            bifurcation_threshold=xi,
        )

    def nodal_profile(
        self,
        node: Any,
        *,
        equilibrium_tolerance: float = _EPS_DNFR_STABLE_DEFAULT,
        bifurcation_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Convenience dict profile for notebooks/reporting pipelines."""
        return self.nodal_state(
            node,
            equilibrium_tolerance=equilibrium_tolerance,
            bifurcation_threshold=bifurcation_threshold,
        ).to_dict()

    def results(self) -> Results:
        """Get comprehensive results including tetrad and balance diagnostics."""
        tetrad = self.tetrad() if _HAS_FIELDS else None
        cons = self.conservation() if _HAS_CONSERVATION else None
        unified = self.telemetry() if _HAS_FIELDS else None
        return Results(
            coherence=self.coherence(),
            sense_index=self.sense_index(),
            nodes=len(self.G.nodes()),
            edges=len(self.G.edges()),
            density=self.density(),
            avg_phase=self.avg_phase(),
            tetrad=tetrad,
            conservation=cons,
            unified_fields=unified,
        )

    def summary(self) -> str:
        """Quick network summary."""
        return self.results().summary()

    # === STRUCTURAL FIELD TETRAD ===

    def tetrad(self) -> TetradSnapshot:
        """Compute the Structural Field Tetrad (Phi_s, |grad_phi|, K_phi, xi_C).

        Returns a TetradSnapshot with per-node canonical fields plus
        extended transport fields (J_phi, J_DNFR).  Use .is_safe() to
        check canonical thresholds.

        Returns
        -------
        TetradSnapshot
            Phi_s, |grad_phi|, K_phi per node; xi_C scalar; J_phi, J_DNFR per node.
        """
        if not _HAS_FIELDS:
            return TetradSnapshot()
        phi_s = compute_structural_potential(self.G)
        grad_phi = compute_phase_gradient(self.G)
        k_phi = compute_phase_curvature(self.G)
        xi_c = estimate_coherence_length(self.G)
        j_phi = compute_phase_current(self.G)
        j_dnfr = compute_dnfr_flux(self.G)
        return TetradSnapshot(
            phi_s=phi_s,
            grad_phi=grad_phi,
            k_phi=k_phi,
            xi_c=xi_c,
            j_phi=j_phi,
            j_dnfr=j_dnfr,
        )

    def tetrad_observation(self):
        """Return the tetrad in the optional cross-domain provenance envelope."""
        from ..metrics.observations import observe_graph_tetrad

        return observe_graph_tetrad(self.tetrad())

    def fields(self) -> dict[str, dict[str, float]]:
        """Compute all canonical + extended fields as flat per-node dicts.

        Returns
        -------
        dict[str, dict]
            Keys: 'phi_s', 'grad_phi', 'k_phi', 'j_phi', 'j_dnfr';
            each maps node -> float.  Plus 'xi_c' -> float.
        """
        snap = self.tetrad()
        return {
            "phi_s": snap.phi_s,
            "grad_phi": snap.grad_phi,
            "k_phi": snap.k_phi,
            "xi_c": snap.xi_c,
            "j_phi": snap.j_phi,
            "j_dnfr": snap.j_dnfr,
        }

    # === STRUCTURAL-BALANCE DIAGNOSTICS ===

    def _sample_structural_balance(self, dt: float = 1.0) -> Any:
        """Record one read-only snapshot and return the latest interval.

        ``ConservationTracker`` stores timestamp/snapshot pairs. Centralizing
        sampling here keeps :meth:`conservation` and :meth:`balance_alerts`
        consistent and prevents diagnostic methods from modifying graph state.
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise TNFRValueError("dt must be finite and positive")
        if self._tracker is None:
            self._tracker = ConservationTracker(self.G)
        if self._tracker._snapshots:
            sample_time = float(self._tracker._snapshots[-1][0]) + dt
        else:
            sample_time = 0.0
        self._tracker.record(t=sample_time)
        return self._tracker.latest_balance

    def conservation(self, dt: float = 1.0) -> ConservationReport:
        """Compute finite-trajectory conservation diagnostics.

        Computes a Noether-like tetrad charge, a non-negative energy candidate,
        and an observed finite-step balance between the two most recent
        diagnostic snapshots. ``dt`` is the declared time between consecutive
        samples from :meth:`conservation` or :meth:`balance_alerts`. The first
        call establishes a baseline and
        reports ``sample_available=False``. Grammar labels alone do not imply
        a zero residual or a monotone candidate-energy trajectory.

        Returns
        -------
        ConservationReport
            Accurate properties ``structural_charge``, ``candidate_energy``,
            ``candidate_energy_nonincreasing`` and ``balance_quality`` plus
            their backward-compatible historical fields.
        """
        if not _HAS_CONSERVATION:
            return ConservationReport()
        Q = compute_noether_charge(self.G)
        E = compute_energy_functional(self.G)
        # Candidate-energy change and balance require two snapshots.
        lyap_stable = True
        lyap_deriv = 0.0
        quality = 1.0
        balance = self._sample_structural_balance(dt=dt)
        sample_available = balance is not None
        if balance is not None:
            quality = balance.conservation_quality
            _, previous = self._tracker._snapshots[-2]
            _, current = self._tracker._snapshots[-1]
            lyap = compute_lyapunov_derivative(previous, current, dt=dt)
            lyap_stable = lyap.is_stable
            lyap_deriv = lyap.energy_derivative
        return ConservationReport(
            noether_charge=Q,
            energy=E,
            lyapunov_stable=lyap_stable,
            lyapunov_derivative=lyap_deriv,
            conservation_quality=quality,
            sample_available=sample_available,
        )

    def symplectic_substrate(self) -> SymplecticReport:
        """Evaluate the auxiliary ambient symplectic substrate model.

        The extracted fields initialize two canonical pairs per graph node,
        ``(K_phi, J_phi)`` and ``(Phi_s, J_dnfr)``, in ``P = R^(4N)``. The
        model's exact harmonic flow is symplectic. This readout neither derives
        that ambient flow from the nodal equation nor certifies any engine
        operator as a symplectomorphism.

        Returns
        -------
        SymplecticReport
            phase_space_dimension, hamiltonian H_sub, background_potential U
            (with H_sub + U equal to the evaluated auxiliary energy),
            liouville_divergence, and is_valid_manifold for the ambient model.
        """
        if not _HAS_SUBSTRATE:
            return SymplecticReport()
        cert = verify_canonical_structure(self.G)
        pt = extract_phase_space_point(self.G)
        return SymplecticReport(
            phase_space_dimension=cert.dimension,
            hamiltonian=substrate_hamiltonian(pt),
            background_potential=background_potential(pt),
            liouville_divergence=cert.liouville_divergence,
            is_valid_manifold=cert.is_valid_symplectic_manifold,
        )

    # === UNIFIED TELEMETRY ===

    def telemetry(self) -> dict[str, Any]:
        """Compute the unified diagnostic telemetry mapping.

        Aggregates the complete four-field diagnostic read-out, complex
        geometric field Psi, emergent fields (chirality, symmetry-breaking,
        coherence coupling), and tensor invariants (energy density,
        topological charge). These diagnostics do not reconstruct the complete
        TNFR graph state or its evolution.

        Returns
        -------
        dict[str, Any]
            Unified diagnostic fields and invariant read-outs. Empty dict if
            the fields module is unavailable.
        """
        if not _HAS_FIELDS:
            return {}
        return compute_unified_telemetry(self.G)

    def tensor_invariants(self) -> dict[str, Any]:
        """Compute tensor invariants (energy density, topological charge).

        Returns
        -------
        dict[str, Any]
            energy_density, topological_charge, conservation_density,
            conservation_quality, num_nodes.
        """
        if not _HAS_FIELDS:
            return {}
        return compute_tensor_invariants(self.G)

    def emergent_fields(self) -> dict[str, Any]:
        """Compute emergent composite fields.

        Returns
        -------
        dict[str, Any]
            chirality, symmetry_breaking, coherence_coupling, num_nodes.
        """
        if not _HAS_FIELDS:
            return {}
        return compute_emergent_fields(self.G)

    # === Structural winding and phase read-outs ===

    def winding(self, order: list[Any] | None = None) -> dict[str, Any]:
        """Measure the phase-winding sector of a declared closed node order.

        The integer ``W`` is the wrapped phase circulation on that loop. The
        read-out distinguishes neutral, unit-winding and higher-winding sectors
        and reports the sign as orientation. It does not identify physical
        particle species, show that the engine created the winding, or equate
        this invariant with an arithmetic or chemical zero-pressure state.

        Parameters
        ----------
        order : list, optional
            Node order defining the closed loop along which the winding is
            measured. When omitted, a traversal is derived only if the entire
            graph is one simple cycle; otherwise an explicit order is required.

        Returns
        -------
        dict
            Preferred fields include ``winding``, ``raw_winding``,
            ``orientation_sign``, ``winding_class``, ``is_integral``,
            ``cycle_nodes`` and explicitly scoped whole-graph snapshot
            telemetry. Historical particle/charge/chirality names remain as
            compatibility aliases.
        """
        from ..physics.emergent_particles import classify_winding_sector

        return classify_winding_sector(self.G, order=order).as_dict()

    def particle(self, order: list[Any] | None = None) -> dict[str, Any]:
        """Compatibility alias for :meth:`winding`.

        The legacy method name does not assert that a winding sector is a
        physical particle species.
        """
        return self.winding(order=order)

    def phase(self) -> dict[str, Any]:
        """Return the network's operational structural-phase label.

        Signed global structural-field imbalances select the compatibility
        labels ``non_life``, ``critical`` and ``life``.  Here ``critical``
        means order imbalance without chirality imbalance; it does not certify
        proximity to a phase transition.  The underlying standardized spatial
        imbalance is a deterministic classifier input, not a significance
        test for correlated graph nodes. Most informative after :meth:`evolve`.

        Returns
        -------
        dict
            The operational ``phase`` label (``"non_life"`` / ``"critical"`` /
            ``"life"``),
            ``is_life`` (bool), the order parameter ``order_parameter`` (<S>),
            ``chirality_mean`` (<chi>), ``coherence_length`` (xi_C) and
            ``has_homochirality`` (bool).
        """
        from ..physics.phase_transition import Phase, capture_phase_snapshot

        snap = capture_phase_snapshot(self.G)
        return {
            "phase": snap.phase.value,
            "is_life": snap.phase is Phase.LIFE,
            "order_parameter": float(snap.order_parameter),
            "chirality_mean": float(snap.chirality_mean),
            "coherence_length": float(snap.coherence_length),
            "has_homochirality": bool(snap.has_homochirality),
        }

    def gauge(self) -> dict[str, Any]:
        """Return legacy auxiliary U(1) field-coordinate diagnostics.

        The complex diagnostic field Psi = K_phi + i*J_phi admits a local U(1)
        coordinate rotation. Its derived connection A=d(arg Psi) is pure gauge,
        so cycle values are floating-point closure residuals rather than
        physical curvature. The historical interaction-regime labels are
        heuristic compatibility fields computed from arg(Psi), residuals and
        structural potential; they are not operator or force classifications.

        Returns
        -------
        dict
            ``per_node``, ``regime_distribution``, ``dominant_regime``,
            accurately named pure-gauge residual aliases, and historical
            ``mean_gauge_curvature`` / ``gauge_flatness`` compatibility keys.
        """
        from ..physics.gauge import classify_network_regimes

        return classify_network_regimes(self.G)

    def spectrum(self) -> dict[str, Any]:
        """Structural relaxation spectrum of the diffusion operator.

        For symmetric adjacency, nodal decay rates are eigenvalues of
        diag(nu_f)*L_rw; only a common frequency gives nu_f*lambda_k.
        Geometry supplies the separate topology-only proxy
        `1/sqrt(lambda_2)`; fitted xi_C remains state-dependent.
        Multiple stationary modes give a zero gap and an infinite proxy.
        Asymmetric adjacency has no symmetric geometry basis; its damping
        rates can be read directly with physics.relaxation_spectrum.

        Returns
        -------
        dict
            ``diffusivity`` (mean nu_f), ``relaxation_rates``
            (actual nodal decay rates ascending), ``spectral_gap`` (second
            nodal decay rate), ``structural_rank`` (distinct geometry modes) and
            ``coherence_length`` (the topology-only `1/sqrt(lambda_2)`
            proxy, independent of the nu_f clock scale).
        """
        from ..physics.structural_diffusion import (
            relaxation_spectrum,
            structural_diffusivity,
            structural_eigenmodes,
            structural_frequency_rank,
        )

        rates = [float(r) for r in relaxation_spectrum(self.G)]
        gap = rates[1] if len(rates) > 1 and rates[1] > 1e-12 else 0.0
        eigenvalues, _ = structural_eigenmodes(self.G)
        geometry_gap = (
            float(eigenvalues[1])
            if len(eigenvalues) > 1 and eigenvalues[1] > 1e-12 else 0.0
        )
        xi_c = (
            1.0 / float(np.sqrt(geometry_gap))
            if geometry_gap > 1e-12
            else float("inf")
        )
        return {
            "diffusivity": float(structural_diffusivity(self.G)),
            "relaxation_rates": rates,
            "spectral_gap": gap,
            "structural_rank": int(structural_frequency_rank(self.G)),
            "coherence_length": xi_c,
        }

    def nfr(self) -> dict[str, Any]:
        """Characterize the network as a Fractal-Resonant Node (NFR).

        Per TNFR.pdf section 1.4.1, an NFR is "a region of structural coherence
        coupled to a network", defined by the triad (EPI, nu_f, phase) with a
        nodal topology and the multiescalar (fractal) + autopoietic properties.
        This surfaces the NFR as the joint read-out of its three emergent
        facets, each from canonical quantities:

        - RESONANT: measured proximity to the zero-pressure fixed-point set;
          dynamic equilibrium additionally requires a recorded dEPI value
          (:func:`~tnfr.metrics.common.is_structural_equilibrium`).
        - GEOMETRIC: the nodal topology radial / annular / multinodal
          (:func:`~tnfr.physics.fields.classify_nodal_topology`, read from the
          structural-potential geometry).
        - FRACTAL: the multi-scale coherence range xi_C (region size).

        Uniform attraction is proved only for the restricted fixed, connected
        pure-EPI diffusion model. Missing nodal telemetry is reported as
        unavailable rather than interpreted as zero. Most informative after
        :meth:`evolve`.

        Returns
        -------
        dict
            ``topology``, ``centers``, ``concentration`` (geometry);
            ``coherence`` (aggregate-then-reduce C(t)),
            ``mean_local_coherence`` (a distinct descriptive mean),
            ``mean_abs_dnfr``, ``mean_abs_depi_dt``,
            ``zero_pressure_fraction`` and ``equilibrium_fraction``
            (resonance, when available);
            ``depi_dt_source`` identifies whether the rate was recorded or
            evaluated from the nodal equation;
            ``coherence_length`` (xi_C, fractal/region scale); ``triad`` (mean
            EPI, nu_f and the Kuramoto phase synchrony); ``n_nodes``.
        """
        from ..physics.fields import classify_nodal_topology

        topo = classify_nodal_topology(self.G)
        nodes = list(self.G.nodes())
        n = len(nodes)
        missing = object()

        def complete_values(aliases: tuple[str, ...]) -> list[float] | None:
            values: list[float] = []
            for node in nodes:
                raw = get_attr(self.G.nodes[node], aliases, missing)
                if raw is missing or raw is None:
                    return None
                try:
                    value = float(raw)
                except (TypeError, ValueError, OverflowError):
                    return None
                if not math.isfinite(value):
                    return None
                values.append(value)
            return values

        dnfr = complete_values(ALIAS_DNFR) if n else None
        recorded_depi = complete_values(ALIAS_DEPI) if n else None
        epis = complete_values(ALIAS_EPI) if n else None
        frequencies = complete_values(ALIAS_VF) if n else None
        thetas = complete_values(ALIAS_THETA) if n else None

        pressure_available = dnfr is not None
        if recorded_depi is not None and dnfr is not None:
            depi = recorded_depi
            depi_dt_source = "recorded"
        elif dnfr is not None and frequencies is not None:
            # Evaluate the missing rate from the canonical nodal equation.
            depi = [vf * pressure for vf, pressure in zip(frequencies, dnfr)]
            depi_dt_source = "nodal_equation"
        else:
            depi = None
            depi_dt_source = None
        dynamic_available = dnfr is not None and depi is not None
        triad_available = (
            epis is not None and frequencies is not None and thetas is not None
        )
        zero_pressure_fraction = (
            sum(1 for value in dnfr if is_structural_equilibrium(value)) / n
            if pressure_available and n
            else None
        )
        equilibrium_fraction = (
            sum(
                1
                for pressure, rate in zip(dnfr, depi)
                if is_structural_equilibrium(pressure, rate)
            )
            / n
            if dynamic_available and n
            else None
        )
        if dynamic_available and n:
            mean_abs_dnfr = finite_mean_absolute(dnfr, name="nfr dnfr")
            mean_abs_depi = finite_mean_absolute(depi, name="nfr depi")
            coherence = structural_coherence(mean_abs_dnfr, mean_abs_depi)
            mean_local_coherence = finite_mean_absolute(
                (
                    structural_coherence(pressure, rate)
                    for pressure, rate in zip(dnfr, depi)
                ),
                name="nfr local coherence",
            )
        else:
            mean_abs_dnfr = None
            mean_abs_depi = None
            coherence = None
            mean_local_coherence = None
        epi_mean = sum(epis) / n if epis is not None and n else None
        vf_mean = sum(frequencies) / n if frequencies is not None and n else None
        phase_sync = (
            float(abs(np.mean(np.exp(1j * np.asarray(thetas)))))
            if thetas is not None and n and np is not None
            else None
        )
        try:
            # Topology-only spectral comparison/fallback 1/sqrt(lambda_2).
            # The primary fitted xi_C is state-dependent and may be undefined
            # at a uniform DeltaNFR=0 equilibrium.
            xi_c = float(self.spectrum()["coherence_length"])
        except Exception:
            xi_c = float("nan")
        return {
            "topology": topo["topology"],
            "centers": topo["centers"],
            "concentration": topo["concentration"],
            "coherence": coherence,
            "mean_local_coherence": mean_local_coherence,
            "mean_abs_dnfr": mean_abs_dnfr,
            "mean_abs_depi_dt": mean_abs_depi,
            "zero_pressure_fraction": zero_pressure_fraction,
            "equilibrium_fraction": equilibrium_fraction,
            "pressure_telemetry_available": pressure_available,
            "dynamic_telemetry_available": dynamic_available,
            "depi_dt_source": depi_dt_source,
            "triad_available": triad_available,
            "coherence_length": xi_c,
            "triad": {
                "epi_mean": epi_mean,
                "vf_mean": vf_mean,
                "phase_sync": phase_sync,
            },
            "n_nodes": n,
        }

    def nfr_observation(self):
        """Return the graph NFR readout with explicit observation provenance."""
        from ..metrics.observations import StructuralObservation

        return StructuralObservation(
            domain="graph",
            pressure_realization="graph_coupled_delta_nfr",
            aggregation="global_network_nfr",
            derivative_kind="read_only_snapshot",
            equilibrium_tolerance=_EPS_DNFR_STABLE_DEFAULT,
            scope="graph NFR observation",
            value=self.nfr(),
        )

    def rhythm(self) -> dict[str, Any]:
        """The emergent pulse: the resonant rhythm the substrate plays.

        TNFR is a substrate that *vibrates and keeps a rhythm* -- the
        conservative face of the nodal dynamics. Every structural mode
        oscillates at omega_k = sqrt(lambda_k), and the equilibria (the
        dNFR = 0 coherence states) are the BEATS the vibration passes through.
        Where :meth:`nfr` is the dissipative read-out (the relaxed state),
        this is its conservative twin -- the resonant spectrum, the dominant
        beat and the self-similar (fractal) signature -- closed-form from the
        structural spectrum (no time integration).

        Returns
        -------
        dict
            ``resonant_spectrum`` (leading omega_k), ``fundamental``,
            ``dominant_beat``, ``spectral_multiplicity`` (fractal signature),
            ``vibration_energy``, ``n_modes``.
        """
        from ..physics.structural_diffusion import compute_emergent_pulse

        return compute_emergent_pulse(self.G)

    def resonance(self) -> dict[str, Any]:
        """The per-NFR pulse and the resonance that couples the NFRs.

        Where :meth:`rhythm` is the collective network pulse, this is its
        *source*: each NFR is itself a phase oscillator (the single-node
        reduction of the nodal equation) pulsing at its own structural
        frequency nu_f with phase phi. Resonance -- the local phase synchrony
        per NFR and the global Kuramoto order R -- couples those pulses, and
        the collective rhythm emerges as they lock (R -> 1). The local face
        of the rhythm, from canonical per-node quantities. Most informative
        after :meth:`evolve`.

        Returns
        -------
        dict
            ``mean_frequency``, ``frequency_spread`` (the per-NFR pulse
            rates nu_f), ``phase_coherence`` (collective Kuramoto R),
            ``mean_local_resonance`` (mean per-NFR resonance),
            ``resonance_gate`` (Delta phi_max), ``n_pulsing``, ``n_nodes``.
        """
        from ..physics.structural_diffusion import compute_nodal_pulse

        return compute_nodal_pulse(self.G)

    def pulse_trajectory(
        self, steps: int = 8, sequence: str = "basic_activation"
    ) -> dict[str, Any]:
        """The pulse IN MOTION: the rhythm forming as the NFR pulses resonate.

        The snapshot read-outs (:meth:`rhythm`, :meth:`resonance`) see a single
        instant; the interesting structure appears only when the dynamics
        *runs*. This evolves a **copy** of the network ``steps`` times (so the
        caller's network is untouched) and records the canonical rhythm
        trajectory at each step -- the collective resonance ``R(t)`` (Kuramoto
        order), the coherence ``C(t)``, and the mean per-NFR local resonance --
        then reports how the collective rhythm emerges from the resonating
        per-NFR pulses.

        Two grounded facts shape it: (1) the collective topological pulse
        ``omega_k = sqrt(lambda_k)`` is **invariant** under evolution on a fixed
        graph, so it is computed once (not per step); (2) the per-NFR pulses
        typically lock with their neighbours (local resonance) **before** the
        global rhythm forms -- ``local_leads_global`` records that cascade
        (clusters lock, then merge). Threshold crossings are linearly
        interpolated between samples so two crossings within one evolution
        step retain their observed order. Most informative from a perturbed /
        off-equilibrium state.

        Returns
        -------
        dict
            ``phase_coherence`` (R(t)), ``coherence`` (C(t)),
            ``local_resonance`` (mean per-NFR local resonance per step);
            ``synchronizing`` (bool, R rises overall), ``delta_R`` (net change),
            ``asymptotic_R`` (final R), ``local_leads_global`` (bool: the
            interpolated local-0.9 crossing precedes the interpolated R-0.5
            crossing); ``collective_pulse`` (the invariant fundamental +
            dominant beat), ``steps``.
        """
        from ..gamma import kuramoto_R_psi
        from ..physics.structural_diffusion import (
            compute_emergent_pulse,
            compute_nodal_pulse,
        )

        probe = Network(self.G.copy(), name=f"{self.name}:pulse")
        r_t: list[float] = []
        c_t: list[float] = []
        local_t: list[float] = []
        for step in range(max(1, steps)):
            if step:
                probe.evolve(1, sequence=sequence)
            r = float(kuramoto_R_psi(probe.G)[0])
            c = float(probe.coherence())
            local = float(compute_nodal_pulse(probe.G)["mean_local_resonance"])
            r_t.append(r)
            c_t.append(c)
            local_t.append(local)
        # the collective topological pulse is invariant under evolution on a
        # fixed graph -> compute it once, not per step
        pulse = compute_emergent_pulse(self.G)
        delta_r = r_t[-1] - r_t[0]
        t_local = _first_interpolated_crossing(local_t, 0.9)
        t_global = _first_interpolated_crossing(r_t, 0.5)
        local_leads = t_local is not None and (
            t_global is None or t_local < t_global
        )
        return {
            "phase_coherence": r_t,
            "coherence": c_t,
            "local_resonance": local_t,
            "synchronizing": delta_r > 0.0,
            "delta_R": delta_r,
            "asymptotic_R": r_t[-1],
            "local_leads_global": local_leads,
            "collective_pulse": {
                "fundamental": pulse["fundamental"],
                "dominant_beat": pulse["dominant_beat"],
            },
            "steps": len(r_t),
        }

    # === COMPLEX FIELD & EXTENDED ACCESS ===

    def complex_field(self) -> dict[str, Any]:
        """Compute the unified complex geometric field Psi = K_phi + i*J_phi.

        Returns
        -------
        dict[str, Any]
            psi_real (K_phi), psi_imag (J_phi), magnitude, phase arrays
            keyed by node.
        """
        if not _HAS_FIELDS:
            return {}
        arrays = compute_complex_geometric_field_arrays(self.G)
        return arrays

    def j_phi(self) -> dict:
        """Phase current J_phi per node (transport companion to K_phi)."""
        if not _HAS_FIELDS:
            return {}
        return compute_phase_current(self.G)

    def j_dnfr(self) -> dict:
        """DELTA_NFR flux J_DELTA_NFR per node."""
        if not _HAS_FIELDS:
            return {}
        return compute_dnfr_flux(self.G)

    def structural_charge(self) -> float:
        """Tetrad charge candidate ``sum_i(Phi_s(i) + K_phi(i))``.

        Its conservation must be tested on a declared trajectory; U1--U6
        validity alone does not make it an exact Noether charge.
        """
        if not _HAS_CONSERVATION:
            return 0.0
        return compute_noether_charge(self.G)

    def noether_charge(self) -> float:
        """Backward-compatible alias for :meth:`structural_charge`."""
        return self.structural_charge()

    def candidate_energy(self) -> float:
        """Non-negative structural energy evaluated as a Lyapunov candidate."""
        if not _HAS_CONSERVATION:
            return 0.0
        return compute_energy_functional(self.G)

    def energy(self) -> float:
        """Backward-compatible alias for :meth:`candidate_energy`."""
        return self.candidate_energy()

    def balance_alerts(self, dt: float = 1.0) -> dict[str, Any]:
        """Sample read-only finite-balance alerts between consecutive calls.

        The first call records a baseline and returns ``sample_available=False``.
        Later calls compare the current graph with the preceding diagnostic
        snapshot using the declared ``dt``. This method never evolves or
        mutates the graph.
        Residual alerts do not validate or classify U1--U6.

        Returns
        -------
        A balance-alert mapping with explicit grammar-inapplicability metadata.
        Legacy ``violations_*`` keys remain false/empty for compatibility.
        """
        if not _HAS_CONSERVATION:
            return _empty_balance_alert_report("conservation_module_unavailable")
        from ..physics.conservation import detect_grammar_violations_from_conservation

        balance = self._sample_structural_balance(dt=dt)
        if balance is None:
            return _empty_balance_alert_report("baseline_only")
        result = detect_grammar_violations_from_conservation(balance)
        result["sample_available"] = True
        return result

    def grammar_violations(self, dt: float = 1.0) -> dict[str, Any]:
        """Backward-compatible alias for :meth:`balance_alerts`.

        The legacy name is retained for callers, but the returned mapping
        always states that grammar was not assessed. Use canonical grammar
        validators on actual operator history for U1--U6 decisions.
        """
        return self.balance_alerts(dt=dt)

    # === GRAMMAR-AWARE DYNAMICS ===

    def evolve_grammar_aware(
        self,
        steps: int = 5,
        candidates: list[str] | None = None,
    ) -> Network:
        """Evolve network with proactive grammar validation (U1-U6).

        Each step selects from grammar-valid operators only, preventing
        violations before they corrupt graph state.

        Parameters
        ----------
        steps : int
            Number of evolution steps.
        candidates : list[str] | None
            Operator NAMES to consider (the canonical public identifiers,
            AGENTS.md §5). Defaults to the stabilizer-leaning set
            ['coherence', 'reception', 'resonance', 'dissonance', 'coupling'].
            Legacy glyph codes ('IL', 'EN', ...) are still accepted.

        Returns
        -------
        Network
            self (for chaining).
        """
        if not _HAS_GRAMMAR_DYNAMICS:
            return self.evolve(steps)
        if candidates is None:
            candidates = [
                "coherence",
                "reception",
                "resonance",
                "dissonance",
                "coupling",
            ]
        # Public API speaks operator NAMES; the grammar machinery
        # (filter_candidates/apply_glyph) operates on glyph codes. Translate
        # names -> glyphs here (legacy codes pass through unchanged).
        from ..operators import apply_glyph
        from ..operators.grammar_types import function_name_to_glyph

        glyphs = [function_name_to_glyph(c, default=c) for c in candidates]
        for _step in range(steps):
            for node in self.G.nodes():
                valid = filter_candidates(self.G, node, glyphs)
                if not valid:
                    continue
                glyph_code = valid[0]  # safest first
                try:
                    apply_glyph(self.G, node, glyph_code)
                except Exception:
                    continue
        return self

    # === OPERATOR-CONTRACT AND BALANCE-ALERT MONITORING ===

    def integrity_check(self, operator_name: str = "coherence") -> dict[str, Any]:
        """Return recorded monitor evidence for one operator name.

        Calling this method attaches a monitor if needed. Evidence is produced
        only by later operator applications, where real before/after states are
        available. The method never fabricates a postcondition result from the
        current state alone.

        Parameters
        ----------
        operator_name : str
            Canonical operator name or glyph used to filter recorded reports.

        Returns
        -------
        dict[str, Any]
            Recorded postcondition results, finite balance alerts and sampled
            candidate-energy changes. ``grammar_validated`` is false because
            monitor telemetry does not replace history/state-aware validators.
            Returns an empty dict if the integrity module is unavailable.
        """
        if not _HAS_INTEGRITY:
            return {}
        if self._monitor is None:
            self._monitor = StructuralIntegrityMonitor.get(self.G)
        if self._monitor is None:
            self._monitor = StructuralIntegrityMonitor(mode=MonitorMode.OBSERVE)
            self._monitor.attach(self.G)
        from ..operators.grammar_types import glyph_function_name

        target_name = glyph_function_name(operator_name)
        reports: list[dict[str, Any]] = []
        matching = [
            report
            for report in self._monitor.summary.reports
            if glyph_function_name(report.operator) == target_name
        ]
        for report in matching[-10:]:
            reports.append(
                {
                    "node": report.node,
                    "within_monitor_policy": report.within_monitor_policy,
                    "balance_quality": report.balance_quality,
                    "balance_alerts": list(report.balance_alerts),
                    "candidate_energy_derivative": (
                        report.candidate_energy_derivative
                    ),
                    "candidate_energy_nonincreasing": (
                        report.candidate_energy_nonincreasing
                    ),
                    "candidate_energy_within_numerical_tolerance": (
                        report.candidate_energy_within_numerical_tolerance
                    ),
                    "structural_charge_drift": report.structural_charge_drift,
                    "postcondition_evaluated": report.postcondition_evaluated,
                    "postcondition_ok": report.postcondition_ok,
                    "diagnostic_follow_up": report.diagnostic_follow_up,
                    # Backward-compatible result key.
                    "passed": report.is_healthy,
                    "details": str(report),
                }
            )
        passed_count = sum(1 for r in reports if r.get("passed", False))
        return {
            "operator": operator_name,
            "operator_name": target_name,
            "evidence_available": bool(reports),
            "nodes_checked": len(reports),
            "passed": passed_count,
            "failed": len(reports) - passed_count,
            "pass_rate": passed_count / max(len(reports), 1),
            "grammar_validated": False,
            "reports": reports,
        }

    def audit_operators(self) -> dict[str, Any]:
        """Proactively MEASURE all 13 operator-contract fidelities.

        Unlike :meth:`integrity_check` (which inspects the current network
        state), this applies each of the 13 canonical operators in its
        correct canonical context and measures whether its postcondition
        contract (AGENTS.md §Operators) is satisfied — the measured-not-
        asserted operator-fidelity audit.

        Returns
        -------
        dict[str, Any]
            ``all_satisfied`` (bool), ``n_satisfied``/``n_operators`` (int),
            ``operators`` (per-operator list of glyph/contract/context/
            satisfied/detail), and ``summary`` (str).  Empty dict if the
            integrity module is unavailable.
        """
        if not _HAS_INTEGRITY:
            return {}
        from ..physics.integrity import audit_operator_contracts

        audit = audit_operator_contracts()
        return {
            "all_satisfied": audit.all_satisfied,
            "n_operators": audit.n_operators,
            "n_satisfied": audit.n_satisfied,
            "operators": [
                {
                    "glyph": r.glyph,
                    "operator": r.operator,
                    "contract": r.contract,
                    "context": r.context,
                    "satisfied": r.satisfied,
                    "detail": r.detail,
                }
                for r in audit.results
            ],
            "summary": audit.summary(),
        }

    # === ANALYSIS ===

    def info(self) -> dict[str, Any]:
        """Detailed network information including feature availability."""
        return {
            "name": self.name,
            "nodes": len(self.G.nodes()),
            "edges": len(self.G.edges()),
            "density": self.density(),
            "coherence": self.coherence(),
            "sense_index": self.sense_index(),
            "avg_phase": self.avg_phase(),
            "is_connected": nx.is_connected(self.G),
            "has_tnfr_props": all("EPI" in self.G.nodes[n] for n in self.G.nodes()),
            "features": {
                "fields": _HAS_FIELDS,
                "conservation": _HAS_CONSERVATION,
                "integrity": _HAS_INTEGRITY,
                "grammar_dynamics": _HAS_GRAMMAR_DYNAMICS,
                "optimization": _HAS_OPTIMIZATION,
            },
        }

    # === FACTORIZATION BRIDGE ===

    def factorize(
        self,
        n: int,
        *,
        modulus: int | None = None,
        trace_certificates: bool = False,
        certificate_dir: str | None = None,
    ) -> FactorizationReport:
        """Run canonical TNFR factorization and attach network synergy diagnostics.

        This creates an explicit bridge between factorization-lab dynamics and
        SDK network telemetry, enabling direct cross-module analysis.
        """
        return TNFR.factorize(
            n,
            modulus=modulus,
            trace_certificates=trace_certificates,
            certificate_dir=certificate_dir,
            network=self,
        )

    def primality(
        self,
        n: int,
        *,
        tolerance: float = 1e-10,
    ) -> PrimalityReport:
        """Run canonical TNFR primality analysis with network synergy diagnostics."""
        return TNFR.primality(n, tolerance=tolerance, network=self)

    def is_prime(
        self,
        n: int,
        *,
        tolerance: float = 1e-10,
    ) -> bool:
        """Convenience boolean primality check fused with SDK bridge."""
        return self.primality(n, tolerance=tolerance).is_prime


class TNFR:
    """Static factory for instant TNFR networks.

    Main entry point for the simplified TNFR SDK.
    All methods are static for maximum convenience.

    **PHILOSOPHY**: Start creating networks immediately with zero boilerplate.
    """

    @staticmethod
    def create(
        num_nodes: int, name: str = "network", seed: int | None = None
    ) -> Network:
        """Create a TNFR network of nodes in structural vacuum.

        Each node is anchored via :func:`create_nfr` with the canonical
        triad initialised to the structural vacuum: EPI = 0, νf = 1 Hz_str,
        θ = 0. **Form (EPI) is not assigned here** -- it emerges canonically
        from the Emission generator the first time :meth:`evolve` runs a
        sequence (invariant #1; grammar U1). The *seed* governs the
        stochastic topology builders for reproducibility (invariant #6).

        Parameters
        ----------
        num_nodes : int
            Number of nodes to anchor.
        name : str
            Optional network label.
        seed : int, optional
            Seed propagated to ``random``/``small_world``/``scale_free`` so
            identical seeds reproduce identical trajectories.

        Returns
        -------
        Network
            Network of nodes in vacuum, ready for topology and evolution.

        Examples
        --------
        >>> net = TNFR.create(10, seed=7).ring().evolve(5)
        """
        num_nodes = nonnegative_integer(num_nodes, "num_nodes")
        G = nx.Graph()
        for i in range(num_nodes):
            create_nfr(i, graph=G, epi=0.0, vf=1.0, theta=0.0)
        return Network(G, name, seed=seed)

    @staticmethod
    def operators(name: str | None = None) -> Any:
        """Return the canonical contract catalog of the 13 structural operators.

        Reads straight from the contract source of truth
        (:mod:`tnfr.operators.operator_contracts`). Each entry exposes the
        operator's public name, internal glyph, nodal-equation channel
        (EPI/nu_f/theta/dNFR), scale (NODE/NETWORK), grammar role(s),
        canonical purpose and postcondition, and TNFR.pdf anchor -- the
        canonical structure for understanding the operator algebra.

        Parameters
        ----------
        name : str, optional
            An operator name (``"emission"``) or glyph (``"AL"``). If given,
            return only that operator's contract; otherwise all 13
            (channel-ordered).

        Returns
        -------
        dict | list[dict]
            One contract dict or the list of all 13.

        Examples
        --------
        >>> TNFR.operators("emission")["channel"]  # doctest: +SKIP
        'EPI'
        """
        from ..operators.grammar_types import (
            CLOSURES,
            COUPLING_RESONANCE,
            DESTABILIZERS,
            GENERATORS,
            STABILIZERS,
            TRANSFORMERS,
        )
        from ..operators.operator_contracts import contract_for, iter_contracts

        def _roles(n: str) -> list[str]:
            roles: list[str] = []
            if n in GENERATORS:
                roles.append("generator")
            if n in CLOSURES:
                roles.append("closure")
            if n in STABILIZERS:
                roles.append("stabilizer")
            if n in DESTABILIZERS:
                roles.append("destabilizer")
            if n in TRANSFORMERS:
                roles.append("transformer")
            if n in COUPLING_RESONANCE:
                roles.append("coupling/resonance")
            return roles

        def _to_dict(c: Any) -> dict[str, Any]:
            return {
                "name": c.english_name,
                "glyph": c.glyph,
                "channel": c.primary_channel.value,
                "scale": c.scale.value,
                "roles": _roles(c.name),
                "purpose": c.purpose,
                "postcondition": c.postcondition,
                "pdf_reference": c.pdf_reference,
            }

        if name is not None:
            return _to_dict(contract_for(name))
        return [_to_dict(c) for c in iter_contracts()]

    @staticmethod
    def explain_sequence(operators: list[str]) -> dict[str, Any]:
        """Validate an operator sequence and explain its canonical grammar.

        A teaching/diagnostic aid: reports each operator's grammar role and
        whether the whole sequence satisfies the unified grammar (U1-U6) --
        why a structural "word" is or is not canonical. Accepts operator
        names or glyph codes.

        Parameters
        ----------
        operators : list[str]
            Operator names (``["emission", "coherence", "silence"]``) or
            glyph codes (``["AL", "IL", "SHA"]``).

        Returns
        -------
        dict
            ``valid`` (bool), ``operators`` (canonical names), ``roles``
            (per-operator), U1 flags ``starts_with_generator`` /
            ``ends_with_closure``, U2 flags ``has_destabilizer`` /
            ``has_stabilizer``, and a human-readable ``message``.
        """
        from ..operators.grammar_types import (
            CLOSURES,
            COUPLING_RESONANCE,
            DESTABILIZERS,
            GENERATORS,
            STABILIZERS,
            TRANSFORMERS,
        )
        from ..operators.operator_contracts import contract_for
        from ..validation import validate_sequence

        def _roles(n: str) -> list[str]:
            roles: list[str] = []
            if n in GENERATORS:
                roles.append("generator")
            if n in CLOSURES:
                roles.append("closure")
            if n in STABILIZERS:
                roles.append("stabilizer")
            if n in DESTABILIZERS:
                roles.append("destabilizer")
            if n in TRANSFORMERS:
                roles.append("transformer")
            if n in COUPLING_RESONANCE:
                roles.append("coupling/resonance")
            return roles

        contracts = [contract_for(op) for op in operators]
        names = [c.name for c in contracts]
        roles = [
            {"name": c.english_name, "glyph": c.glyph, "roles": _roles(c.name)}
            for c in contracts
        ]
        try:
            outcome = validate_sequence(names)
            valid = bool(getattr(outcome, "passed", bool(outcome)))
            summary = getattr(outcome, "summary", {}) or {}
            default_msg = "valid sequence" if valid else "invalid sequence"
            message = summary.get("message", default_msg)
        except Exception as exc:  # validation raised -> invalid
            valid = False
            message = str(exc)
        return {
            "valid": valid,
            "operators": [c.english_name for c in contracts],
            "roles": roles,
            "starts_with_generator": bool(names) and names[0] in GENERATORS,
            "ends_with_closure": bool(names) and names[-1] in CLOSURES,
            "has_destabilizer": any(n in DESTABILIZERS for n in names),
            "has_stabilizer": any(n in STABILIZERS for n in names),
            "message": message,
        }

    @staticmethod
    def template(template_name: str) -> Network:
        """Create network from pre-configured template.

        Available templates:
        - 'small': 5 nodes, ring topology
        - 'medium': 15 nodes, small-world topology
        - 'large': 50 nodes, random topology
        - 'molecule': 8 nodes, molecular-like structure
        - 'star': 10 nodes, star topology
        - 'complete': 6 nodes, complete graph

        Args:
            template_name: Template to use

        Returns:
            Pre-configured network ready to use

        Example:
            >>> mol = TNFR.template('molecule')
        """
        templates = {
            "small": lambda: TNFR.create(5).ring(),
            "medium": lambda: TNFR.create(15).ring().random(0.1),  # Small-world-like
            "large": lambda: TNFR.create(50).random(0.08),
            "molecule": lambda: TNFR.create(8).ring().random(0.2),
            "star": lambda: TNFR.create(10).star(),
            "complete": lambda: TNFR.create(6).complete(),
        }

        if template_name not in templates:
            available = ", ".join(templates.keys())
            raise TNFRValueError(
                f"Unknown template '{template_name}'.",
                context={
                    "requested": template_name,
                    "available": list(templates.keys()),
                },
                suggestion=f"Choose from: {available}",
            )

        return templates[template_name]()

    @staticmethod
    def compare(*networks: Network) -> dict[str, Any]:
        """Compare multiple networks including tetrad and conservation.

        Args:
            *networks: Networks to compare

        Returns:
            Comparison results with rankings and structural field comparison.

        Example:
            >>> comparison = TNFR.compare(net1, net2, net3)
            >>> print(comparison['ranking'])
        """
        if not networks:
            return {}

        results = []
        for i, net in enumerate(networks):
            result = net.results()
            entry: dict[str, Any] = {
                "name": net.name,
                "index": i,
                "coherence": result.coherence,
                "sense_index": result.sense_index,
                "nodes": result.nodes,
                "edges": result.edges,
                "density": result.density,
            }
            if result.conservation is not None:
                entry["structural_charge"] = result.conservation.structural_charge
                entry["candidate_energy"] = result.conservation.candidate_energy
                entry["candidate_energy_nonincreasing"] = (
                    result.conservation.candidate_energy_nonincreasing
                )
                entry["candidate_energy_within_numerical_tolerance"] = (
                    result.conservation.candidate_energy_within_numerical_tolerance
                )
                # Backward-compatible historical keys.
                entry["noether_charge"] = result.conservation.noether_charge
                entry["energy"] = result.conservation.energy
                entry["lyapunov_stable"] = result.conservation.lyapunov_stable
            results.append(entry)

        # Rank by coherence
        ranking = sorted(results, key=lambda x: x["coherence"], reverse=True)

        return {
            "results": results,
            "ranking": ranking,
            "best": ranking[0] if ranking else None,
            "worst": ranking[-1] if ranking else None,
            "count": len(networks),
        }

    @staticmethod
    def analyze(network: Network) -> dict[str, Any]:
        """One-shot comprehensive structural analysis.

        Computes coherence, sense index, full tetrad, conservation diagnostics,
        tensor invariants, emergent fields, and integrity check in a single call.

        Args:
            network: Network to analyze.

        Returns:
            Complete analysis dictionary with all available metrics.

        Example:
            >>> analysis = TNFR.analyze(net)
            >>> print(analysis['coherence'], analysis['tetrad'].summary())
        """
        result: dict[str, Any] = {
            "coherence": network.coherence(),
            "sense_index": network.sense_index(),
            "nodes": len(network.G.nodes()),
            "edges": len(network.G.edges()),
            "density": network.density(),
            "avg_phase": network.avg_phase(),
            "nodal_dynamics": network.nodal_scan(),
        }
        try:
            result["nfr"] = network.nfr()
        except Exception:
            pass
        if _HAS_FIELDS:
            result["tetrad"] = network.tetrad()
            result["tensor_invariants"] = network.tensor_invariants()
            result["emergent_fields"] = network.emergent_fields()
        if _HAS_CONSERVATION:
            result["conservation"] = network.conservation()
        if _HAS_SUBSTRATE:
            result["symplectic_substrate"] = network.symplectic_substrate()
        if _HAS_INTEGRITY:
            result["integrity"] = network.integrity_check()
        result["features"] = {
            "fields": _HAS_FIELDS,
            "conservation": _HAS_CONSERVATION,
            "symplectic_substrate": _HAS_SUBSTRATE,
            "integrity": _HAS_INTEGRITY,
            "grammar_dynamics": _HAS_GRAMMAR_DYNAMICS,
            "optimization": _HAS_OPTIMIZATION,
        }
        return result

    @staticmethod
    def factorize(
        n: int,
        *,
        modulus: int | None = None,
        trace_certificates: bool = False,
        certificate_dir: str | None = None,
        network: Network | None = None,
    ) -> FactorizationReport:
        """Canonical TNFR factorization, optionally fused with network telemetry.

        Factorization is the **spectral** sector (theory
        TNFR_NUMBER_THEORY.md §9.5): the factor signal of a semiprime
        ``n = p*q`` appears as a *coset / Fourier mode* of the emergent
        structural-diffusion spectrum ``L_rw = I - D^{-1} W`` (the canonical
        ``ΔNFR`` EPI channel) on the residue/Paley graph -- a non-circular
        spectral diagnostic under that declared graph family, not the symbolic
        per-node ``ΔNFR`` (which is blind to the cosets). Honest scope: the residue graph is regular, so ``L_rw``
        shares eigenvectors with the classical Laplacian and the coset signal
        is the CRT structure *re-expressed*, not added by the emergent framing;
        TNFR factorization is **not** a speedup over classical factoring.

        Parameters
        ----------
        n : int
            Integer to factor (>1).
        modulus : int | None
            Optional Paley modulus override.
        trace_certificates : bool
            Whether to emit operator/partition certificate artifacts.
        certificate_dir : str | None
            Optional output directory for certificate artifacts.
        network : Network | None
            If provided, computes an explicitly heuristic alignment between
            the factorization telemetry and this independently supplied graph.
        """
        if not _HAS_FACTORIZATION:
            raise TNFRValueError(
                "Factorization bridge is unavailable.",
                context={"feature": "sdk.factorize", "available": False},
                suggestion=(
                    "Ensure the canonical factorization module is present "
                    "(tnfr.factorization + factorization-lab in this repository)."
                ),
            )

        kwargs: dict[str, Any] = {
            "modulus": modulus,
            "trace_certificates": trace_certificates,
        }
        if certificate_dir is not None:
            from pathlib import Path

            kwargs["certificate_dir"] = Path(certificate_dir)

        raw_result = canonical_factorize(n, **kwargs)
        return _build_factorization_report(raw_result, network=network)

    @staticmethod
    def primality(
        n: int,
        *,
        tolerance: float = 1e-10,
        network: Network | None = None,
    ) -> PrimalityReport:
        """Canonical primality analysis with optional heuristic graph alignment.

        Reads the arithmetic equilibrium ``ΔNFR_arith(n) = 0`` (**sector A** --
        an exact but *circular* re-expression that consumes ``n``'s
        divisibility). A separate non-circular spectral construction is exposed
        through :meth:`factorize`; its finite coverage and hypotheses are
        documented in theory §9.5.
        """
        if not _HAS_PRIMALITY:
            raise TNFRValueError(
                "Primality bridge is unavailable.",
                context={"feature": "sdk.primality", "available": False},
                suggestion=(
                    "Ensure the canonical primality module is present "
                    "(tnfr.primality + primality-test in this repository)."
                ),
            )
        raw = canonical_primality_analyze(n, tolerance=tolerance)
        return _build_primality_report(n, raw, tolerance=tolerance, network=network)

    @staticmethod
    def is_prime(
        n: int,
        *,
        tolerance: float = 1e-10,
    ) -> bool:
        """Convenience boolean primality check from canonical SDK bridge."""
        return TNFR.primality(n, tolerance=tolerance).is_prime

    @staticmethod
    def primes(max_number: int = 100) -> dict[str, Any]:
        """Read structural primes off the arithmetic equilibrium field.

        A number ``n`` is structurally prime when its arithmetic reorganization
        pressure vanishes, ``ΔNFR_arith(n) = 0`` -- the exact nodal-equation
        equilibrium predicate
        (:func:`tnfr.metrics.common.is_structural_equilibrium`). Other domain
        models can reuse that numerical predicate without sharing a state
        space or evolution law.

        Emergence status (theory/TNFR_NUMBER_THEORY.md §9.5). This method is
        **sector A**: the arithmetic ``ΔNFR`` is an *exact but circular*
        re-expression that **consumes** ``n``'s divisibility ``(Ω, τ, σ)``.
        A separate **sector B** studies the spectral Paley/residue Fiedler gap
        (input only
        ``x^2 mod n``, primes-OUT, non-circular), carried by the spectral
        factorizer (:meth:`factorize`), not by this arithmetic read-out. These
        are distinct constructions with different inputs and dynamics.

        Parameters
        ----------
        max_number : int
            Largest integer to include in the arithmetic network.

        Returns
        -------
        dict
            ``max_number``, ``primes`` (sorted list) and ``count``.

        Examples
        --------
        >>> TNFR.primes(30)["primes"]  # doctest: +SKIP
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        """
        from ..mathematics.number_theory import ArithmeticTNFRNetwork

        net = ArithmeticTNFRNetwork(max_number)
        candidates = net.detect_prime_candidates()
        primes = sorted(int(n) for n, _delta in candidates)
        return {
            "max_number": net.max_number,
            "primes": primes,
            "count": len(primes),
        }

    @staticmethod
    def magic_numbers(max_n: int = 7) -> list[int]:
        """Closure counts from the assumption-explicit structural shell model.

        The closed-shell counts (2, 10, 18, 36, 54, 86, ...) combine two
        ingredients of *different* status:

        - a constructed S² graph numerically approximates multiplicities
          ``2l+1``;
        - occupation capacities ``2(2l+1)`` are assumed, including the factor
          two;
        - the ``(n+l)`` filling order is an **assumed** integer count rule
          (Madelung), retained because the free manifold spectrum does not by
          itself reproduce it.

        A closed shell is ``ΔNFR_chem = 0``
        (:func:`tnfr.metrics.common.is_structural_equilibrium`): the chemical
        zero-pressure state of this shell-filling model. It shares an abstract
        predicate with arithmetic equilibrium but not its state space or
        dynamics.

        Parameters
        ----------
        max_n : int
            Highest principal shell index to fill.

        Returns
        -------
        list[int]
            The model closure counts used for noble-gas comparison.
        """
        from ..physics.emergent_chemistry import emergent_magic_numbers

        return [int(z) for z in emergent_magic_numbers(max_n=max_n)]

    @staticmethod
    def element(Z: int, *, max_n: int = 7) -> dict[str, Any]:
        """Structural characterization of the element with count Z.

        Structural shell-model characterization: configuration, valence count
        and distance ``ΔNFR_chem`` to a declared closure. The constructed S²
        graph supplies a numerical ``2l+1`` comparison; capacities
        ``2(2l+1)``, the ``(n+l)`` order and duet/octet closures are assumed.
        A closed shell is ``ΔNFR_chem(Z) = 0``
        within the chemical model. The arithmetic criterion
        ``ΔNFR_arith(n) = 0`` uses the same shared numerical predicate on a
        different state space; this does not identify the two dynamics.

        Parameters
        ----------
        Z : int
            Positive integer count allocated through the declared shell capacities.
        max_n : int
            Highest principal shell index available.

        Returns
        -------
        dict
            Z, configuration, valence_electrons, outer_shell_n, delta_nfr,
            closure_distance, closed_shell, magic_number, the legacy
            reactivity alias, and config_label.
        """
        from ..physics.emergent_chemistry import classify_element

        return classify_element(Z, max_n=max_n).as_dict()

    @staticmethod
    def guide() -> str:
        """Print and return a theory-to-code discovery map.

        Lists every major SDK method alongside the TNFR theory document
        and example that demonstrates it, enabling quick navigation from
        code to physics and back.

        Returns:
            Formatted guide string (also printed to stdout).

        Example:
            >>> TNFR.guide()
        """
        lines = [
            "TNFR SDK — Theory-to-Code Guide",
            "=" * 50,
            "",
            "SDK Method                     Theory                                        Example",
            "-" * 100,
            "TNFR.create(n).ring()          FUNDAMENTAL_THEORY.md                         01-03, 05-06, 08",
            ".small_world(k, p)             FUNDAMENTAL_THEORY.md                         31, 34",
            ".scale_free(m)                 FUNDAMENTAL_THEORY.md                         34",
            ".grid(rows, cols)              FUNDAMENTAL_THEORY.md                         34",
            ".path()                        FUNDAMENTAL_THEORY.md                         —",
            ".tetrad()                      EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md      20, unified_fields_showcase",
            ".conservation()                STRUCTURAL_CONSERVATION_THEOREM.md             17, 24, 34",
            ".evolve_grammar_aware(steps)   UNIFIED_GRAMMAR_RULES.md                      04, 07",
            ".integrity_check()             STRUCTURAL_STABILITY_AND_DYNAMICS.md           29",
            ".complex_field()               EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md      33",
            ".j_phi() / .j_dnfr()          EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md      33",
            ".tensor_invariants()           EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md      20, 33",
            ".emergent_fields()             EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md      33",
            ".structural_charge()           STRUCTURAL_CONSERVATION_THEOREM.md             17, 34",
            ".candidate_energy()            STRUCTURAL_CONSERVATION_THEOREM.md             17, 34",
            ".noether_charge() [alias]      STRUCTURAL_CONSERVATION_THEOREM.md             34",
            ".energy() [alias]              STRUCTURAL_CONSERVATION_THEOREM.md             34",
            ".nodal_state(node)             TNFR.pdf §2.1 (nodal equation)                 04, 05",
            ".nodal_scan()                  TNFR.pdf §2.1 + U4 bifurcation diagnostics      07",
            ".nodal_profile(node)           TNFR nodal telemetry bridge                     10",
            ".nfr()                         TNFR.pdf §1.4.1 (NFR region of coherence)        —",
            ".balance_alerts()              STRUCTURAL_CONSERVATION_THEOREM.md ss12        17, 36",
            ".grammar_violations() [alias]  STRUCTURAL_CONSERVATION_THEOREM.md ss12        36",
            ".telemetry()                   FUNDAMENTAL_THEORY.md                         10",
            ".auto_optimize()               AGENTS.md § Self-Optimizing Dynamics           30",
            "TNFR.factorize(n)              TNFR_NUMBER_THEORY.md                          40",
            "Network.factorize(n)           TNFR_NUMBER_THEORY.md + SDK telemetry bridge   40",
            "TNFR.primality(n)              TNFR_NUMBER_THEORY.md                          40",
            "Network.primality(n)           TNFR_NUMBER_THEORY.md + SDK telemetry bridge   40",
            "TNFR.analyze(net)              APPLIED_STRUCTURAL_ANALYSIS.md                 10",
            "",
            "New theory-experiment links (v0.0.3.2):",
            "  31 — Structural scale basis (π)",
            "  32 — Spiral attractors (golden spiral, KAM)",
            "  33 — Complex field unification (Psi = K_phi + i*J_phi)",
            "  34 — Conservation protocol suite (Noether, Lyapunov)",
            "  35 — Tetrad diagnostic complementarity (finite probes)",
            "  36 — Balance-alert limits and independent grammar validation",
            "",
            "Riemann program:               TNFR_RIEMANN_RESEARCH_NOTES.md                16, 18-23, 25",
            "Classical/Quantum regimes:      PHYSICAL_REGIME_CORRESPONDENCES.md             11-15",
            "Variational formulation:        TNFR_VARIATIONAL_PRINCIPLE.md                  27",
            "Dissipative systems:            DISSIPATIVE_AND_OPEN_SYSTEMS.md                28",
            "Gauge structure:                GAUGE_SYMMETRY_AND_UNIFICATION.md              26",
            "",
            "All theory docs: theory/README.md | All examples: examples/README.md",
        ]
        text = "\n".join(lines)
        print(text)
        return text


# === CONVENIENT ALIASES ===

# Short aliases for power users
T = TNFR  # Even shorter: T.create(10).ring()
Net = Network  # type alias

# Export main API
__all__ = [
    "TNFR",
    "Network",
    "Results",
    "TetradSnapshot",
    "ConservationReport",
    "SymplecticReport",
    "FactorizationReport",
    "PrimalityReport",
    "NodalStateReport",
    "NodalDynamicsReport",
    "T",
    "Net",
]

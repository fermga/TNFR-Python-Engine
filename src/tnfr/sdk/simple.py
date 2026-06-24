"""Simple TNFR SDK — Maximum Power, Minimum Complexity.

The simplified TNFR API designed for 90% of use cases, now with full
Structural Field Tetrad, conservation law monitoring, grammar-aware
dynamics, and research-grade telemetry.

**DESIGN PRINCIPLE**:
- **Intuitive**: Natural method names that read like English
- **Chainable**: Fluent interface for rapid prototyping
- **Complete**: Full TNFR physics under the hood
- **Research-grade**: Structural Field Tetrad + conservation laws

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

    # Conservation law monitoring
    conservation = net.conservation()
    # -> {'noether_charge': Q, 'energy': E, 'lyapunov_stable': bool, ...}

    # Unified telemetry (all fields + invariants)
    telemetry = net.telemetry()

    # Grammar-aware evolution
    net.evolve_grammar_aware(steps=10)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx

from ..alias import get_attr, set_attr
from ..constants import DEFAULTS
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..constants.canonical import MIN_BUSINESS_COHERENCE_CANONICAL as COHERENCE_STRONG
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..metrics.coherence import compute_coherence
from ..metrics.common import is_structural_equilibrium, structural_coherence
from ..metrics.sense_index import compute_Si
from ..operators.nodal_equation import compute_d2epi_dt2, compute_expected_depi_dt

# TNFR core imports
from ..structural import create_nfr

# Canonical telemetry marks (AGENTS.md §7) -- heuristic cuts, not fitted:
#   C(t) > COHERENCE_STRONG ((e*phi)/(pi+e) ~ 0.7506) -> strong coherence
#   Si   > SENSE_INDEX_EXCELLENT (~0.8)               -> excellent sense index
SENSE_INDEX_EXCELLENT = 0.8

# Canonical mutation/bifurcation threshold xi. ZHIR (Mutation) is the
# structural bifurcation operator: it transforms phase when the structural
# change rate crosses xi (AGENTS.md §5: "ZHIR transforms theta when
# dEPI/dt > xi"). Mirrors the engine default ZHIR_THRESHOLD_XI.
_ZHIR_THRESHOLD_XI_DEFAULT = 0.1

# Canonical graph-node equilibrium tolerance (EPS_DNFR_STABLE ~ 1e-3): the same
# cut the engine's stability tracker uses (metrics/coherence). The per-node
# micro-NFR equilibrium (nodal_state) and the region macro-NFR equilibrium
# (Network.nfr) share this single canonical tolerance.
_EPS_DNFR_STABLE_DEFAULT = float(DEFAULTS["EPS_DNFR_STABLE"])


def _run_network_sequence(
    G: nx.Graph,
    operator_names: list[str],
    *,
    cycles: int = 1,
    validate: bool = True,
    suppress_birth_warnings: bool = False,
    context: dict[str, Any] | None = None,
    on_step: Callable[[str], None] | None = None,
) -> None:
    """Evolve all nodes synchronously (lock-step) by an operator sequence.

    Canonical network-evolution primitive shared by the SDK surfaces. Each
    operator in *operator_names* is applied to EVERY node before advancing to
    the next, honouring the temporal simultaneity of the nodal equation
    dEPI/dt = nu_f * dNFR(t): the coupling operators (Reception, Resonance,
    Coupling) see neighbours at the same time step. A row-major schedule
    (whole sequence per node) would fracture coupling symmetry.

    The grammar (U1-U6) is validated once when *validate* is True; every node
    then receives the same validated trajectory, interleaved across the
    lock-step. Form (EPI) is created from the structural vacuum by the
    Emission generator that opens canonical sequences -- never by direct
    assignment (invariant #1; grammar U1).
    """
    from ..operators.registry import get_operator_class
    from ..validation import validate_sequence

    names = list(operator_names)
    if not names:
        return
    if validate:
        validate_sequence(names, context=context)
    ops = [get_operator_class(n)() for n in names]
    compute = G.graph.get("compute_delta_nfr")
    nodes = list(G.nodes())
    with warnings.catch_warnings():
        if suppress_birth_warnings:
            warnings.filterwarnings("ignore", message=r".*has no sources.*")
        for _ in range(cycles):
            for op in ops:
                for node in nodes:
                    G._last_operator_applied = op.name
                    op(G, node)
                if callable(compute):
                    compute(G)
                if on_step is not None:
                    on_step(op.name)


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

# Conservation laws (Noether-like)
try:
    from ..physics.conservation import (
        ConservationTracker,
        capture_conservation_snapshot,
        compute_energy_functional,
        compute_lyapunov_derivative,
        compute_noether_charge,
        verify_conservation_balance,
    )

    _HAS_CONSERVATION = True
except ImportError:
    _HAS_CONSERVATION = False

# Emergent symplectic substrate (geometry the nodal dynamics generates)
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

    Captures the complete state of the TNFR Structural Field Tetrad
    (Phi_s, |grad_phi|, K_phi, xi_C) at a single point in time.

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
    """Conservation law diagnostics from structural continuity theorem.

    Captures Noether charge Q, energy functional E, Lyapunov stability,
    and conservation quality metrics.
    """

    noether_charge: float = 0.0
    energy: float = 0.0
    lyapunov_stable: bool = True
    lyapunov_derivative: float = 0.0
    conservation_quality: float = 0.0

    def summary(self) -> str:
        """One-line conservation summary."""
        stable_str = "STABLE" if self.lyapunov_stable else "UNSTABLE"
        return (
            f"Q={self.noether_charge:.4f}, E={self.energy:.4f}, "
            f"dE/dt={self.lyapunov_derivative:.4f} ({stable_str}), "
            f"quality={self.conservation_quality:.3f}"
        )


@dataclass
class SymplecticReport:
    """Emergent symplectic substrate diagnostics.

    Captures the geometry the nodal dynamics generates from itself: phase
    space dimension 4N, the substrate Hamiltonian H_sub, the configuration
    background U, the Liouville divergence (≈0), and whether the structure
    is a valid symplectic manifold.
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
    """Unified SDK report for canonical TNFR factorization.

    Bridges `tnfr.factorization.factorize()` with SDK-level telemetry and
    optional network-coupled synergy diagnostics.
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
    network_synergy: dict[str, Any] | None = None

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
    """Unified SDK report for canonical TNFR primality analysis.

    Bridges `tnfr.primality.analyze()` with SDK-level telemetry and
    optional network-coupled synergy diagnostics.
    """

    n: int
    is_prime: bool
    delta_nfr: float
    tolerance: float
    components: dict[str, Any] = field(default_factory=dict)
    triad: dict[str, Any] = field(default_factory=dict)
    network_synergy: dict[str, Any] | None = None

    def summary(self) -> str:
        status = "prime" if self.is_prime else "composite"
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
    """Node-level nodal dynamics snapshot based on ∂EPI/∂t = νf·ΔNFR."""

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
        }


@dataclass
class NodalDynamicsReport:
    """Global nodal-dynamics report for study and diagnostics."""

    nodes: dict[Any, NodalStateReport] = field(default_factory=dict)
    equilibrium_tolerance: float = _EPS_DNFR_STABLE_DEFAULT
    bifurcation_threshold: float = _ZHIR_THRESHOLD_XI_DEFAULT

    def summary(self) -> str:
        n = len(self.nodes)
        if n == 0:
            return "Nodal dynamics: empty"
        values = list(self.nodes.values())
        active = sum(1 for s in values if s.active)
        equilibrium = sum(1 for s in values if s.equilibrium)
        bif = sum(1 for s in values if s.near_bifurcation)
        mean_abs_rate = sum(abs(s.expected_depi_dt) for s in values) / n
        mean_coh = sum(s.coherence for s in values) / n
        return (
            f"NodalDynamics(N={n}, active={active}, equilibrium={equilibrium}, "
            f"bifurcation={bif}, C={mean_coh:.3f}, mean|∂EPI/∂t|={mean_abs_rate:.4g})"
        )

    def top_pressure_nodes(self, k: int = 5) -> list[NodalStateReport]:
        """Nodes with largest |∂EPI/∂t| (highest nodal drive)."""
        ranked = sorted(
            self.nodes.values(), key=lambda s: abs(s.expected_depi_dt), reverse=True
        )
        return ranked[: max(int(k), 0)]

    def near_equilibrium_nodes(self) -> list[NodalStateReport]:
        """Nodes with |ΔNFR| <= equilibrium tolerance."""
        return [s for s in self.nodes.values() if s.equilibrium]

    def to_dict(self) -> dict[str, Any]:
        values = list(self.nodes.values())
        n = len(values)
        mean_abs_rate = sum(abs(s.expected_depi_dt) for s in values) / max(n, 1)
        max_abs_rate = max((abs(s.expected_depi_dt) for s in values), default=0.0)
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
                "bifurcation_count": sum(1 for s in values if s.near_bifurcation),
                "mean_abs_depi_dt": float(mean_abs_rate),
                "max_abs_depi_dt": float(max_abs_rate),
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
        # Equal-weight aggregate of the canonical components (coherence
        # alignment and the nodal-drive |nu_f * dNFR| score) -- equipartition,
        # avoiding arbitrary weighting.
        synergy_index = (coherence_alignment + drive_score) / 2.0
        topology_resonance = [
            int(f)
            for f in candidate_factors
            if f > 1 and len(network.G.nodes()) % int(f) == 0
        ]
        network_synergy = {
            "network_nodes": len(network.G.nodes()),
            "network_coherence": net_coherence,
            "network_sense_index": net_si,
            "coherence_alignment": coherence_alignment,
            "nodal_drive": nodal_drive,
            "drive_score": drive_score,
            "synergy_index": synergy_index,
            "topology_resonance_factors": sorted(set(topology_resonance)),
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
        if is_prime:
            prime_resonance = net_si
        else:
            prime_resonance = max(0.0, 1.0 - net_si)
        # Equal-weight aggregate of the canonical components (coherence
        # alignment, dNFR pressure score, Si-based prime resonance) --
        # equipartition, avoiding arbitrary weighting.
        synergy_index = (coherence_alignment + pressure_score + prime_resonance) / 3.0
        network_synergy = {
            "network_nodes": len(network.G.nodes()),
            "network_coherence": net_coherence,
            "network_sense_index": net_si,
            "local_coherence": local_coherence,
            "coherence_alignment": coherence_alignment,
            "pressure_ratio": pressure_ratio,
            "pressure_score": pressure_score,
            "prime_resonance": prime_resonance,
            "synergy_index": synergy_index,
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
            lines.append(f"  Conservation: {self.conservation.summary()}")
        return "\n".join(lines)

    def is_coherent(self) -> bool:
        """Quick coherence check (C(t) > strong mark ~0.7506; AGENTS.md §7)."""
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
                "noether_charge": float(self.conservation.noether_charge),
                "energy": float(self.conservation.energy),
                "lyapunov_stable": self.conservation.lyapunov_stable,
            }
        return d


class Network:
    """Core TNFR Network — Essential Operations + Advanced Telemetry.

    Simplified interface to TNFR networks with structural field tetrad,
    conservation law monitoring, grammar-aware dynamics, and integrity
    checks for research-grade analysis.
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
        edges = [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]
        self.G.add_edges_from(edges)
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
        s = seed if seed is not None else self._seed
        rng = np.random.RandomState(s)
        nodes = list(self.G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                if rng.random_sample() < probability:
                    self.G.add_edge(u, v)
        return self

    def star(self, center: int | None = None) -> Network:
        """Connect every node to a single central hub (star topology)."""
        nodes = list(self.G.nodes())
        if center is None:
            center = nodes[0]

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
        n = len(self.G.nodes())
        ws = nx.watts_strogatz_graph(n, k, p, seed=seed)
        self.G.add_edges_from(ws.edges())
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
        n = len(self.G.nodes())
        ba = nx.barabasi_albert_graph(n, m, seed=seed)
        self.G.add_edges_from(ba.edges())
        return self

    def grid(self, rows: int | None = None, cols: int | None = None) -> Network:
        """Create 2-D grid (lattice) topology.

        If *rows* and *cols* are omitted the closest square layout is used.
        """
        n = len(self.G.nodes())
        if rows is None:
            rows = int(n**0.5)
        if cols is None:
            cols = rows
        g2d = nx.grid_2d_graph(rows, cols)
        mapping = {node: i for i, node in enumerate(g2d.nodes())}
        g2d = nx.relabel_nodes(g2d, mapping)
        for u, v in g2d.edges():
            if u in self.G and v in self.G:
                self.G.add_edge(u, v)
        return self

    def path(self) -> Network:
        """Connect nodes in a linear path (no closing edge)."""
        nodes = list(self.G.nodes())
        for i in range(len(nodes) - 1):
            self.G.add_edge(nodes[i], nodes[i + 1])
        return self

    # === EVOLUTION ===

    def evolve(self, steps: int = 5, sequence: str = "basic_activation") -> Network:
        """Evolve the network by applying a canonical operator sequence.

        The named *sequence* is resolved to canonical structural operators
        and applied to the whole network in **lock-step** (synchronously):
        each operator is applied to *every* node before advancing to the
        next, honouring the temporal simultaneity of the nodal equation
        ``dEPI/dt = vf * dNFR(t)`` -- the coupling operators (Resonance,
        Coupling) see neighbours at the *same* time step. Form (EPI) is
        created from the structural vacuum by the **Emission** generator that
        opens every canonical sequence -- never by direct assignment
        (invariant #1; the grammar U1 initiation rule).

        Lock-step is the canonical network evolution. A row-major schedule
        (whole sequence per node) would let Reception fire before any
        neighbour has emitted, fracturing the coupling symmetry of a
        symmetric graph (a measurable, non-physical artifact).

        Parameters
        ----------
        steps : int
            Number of times the sequence is applied to the whole network.
        sequence : str
            Name of a canonical sequence (see
            :data:`tnfr.sdk.fluent.NAMED_SEQUENCES`). Defaults to
            ``"basic_activation"`` =
            ``[Emission, Reception, Coherence, Resonance, Silence]``.

        Returns
        -------
        Network
            self (for chaining).

        Raises
        ------
        TNFRValueError
            If *sequence* is not a known canonical sequence; an invalid
            sequence is rejected (by name lookup or grammar validation)
            rather than silently ignored.
        """
        from .fluent import NAMED_SEQUENCES

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
        # Lock-step network evolution from the canonical primitive. The
        # sub-threshold birth phase (neighbour EPI < ACTIVE_EMISSION_THRESHOLD
        # ~ 0.464) makes Reception correctly find no sources; that transient
        # warning is silenced on the high-level SDK surface.
        _run_network_sequence(
            self.G,
            operator_names,
            cycles=steps,
            suppress_birth_warnings=True,
        )
        return self

    def auto_optimize(self) -> Network:
        """Auto-optimize: drive the network toward coherence.

        Self-optimization in TNFR is *gradient descent on the structural
        manifold* (AGENTS.md §Self-Optimizing Dynamics): apply grammar-valid
        **stabilizers** (coherence IL, reception EN, resonance RA, coupling
        UM) per node, which monotonically reduce |ΔNFR| and raise C(t).

        This delegates to the grammar-aware evolution restricted to a
        stabilizer-only candidate set, so it never destabilizes (the full
        glyph selector would explore/destabilize fragile nodes, which is the
        general dynamics, not coherence optimization).

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

        Applies *sequence* in lock-step for *cycles* repetitions and records a
        snapshot of C(t) and Si after every operator is applied to the whole
        network, so the intra-sequence transient -- how each structural
        operator moves the metrics -- can be studied, not just the post-cycle
        fixed point. Reuses the canonical lock-step primitive via a per-step
        callback.

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

        Examples
        --------
        >>> hist = TNFR.create(12, seed=1).random(0.3).trajectory(2)
        >>> [(h["operator"], round(h["coherence"], 3)) for h in hist]  # doctest: +SKIP
        """
        from .fluent import NAMED_SEQUENCES

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
            cycles=int(cycles),
            suppress_birth_warnings=True,
            on_step=_record,
        )
        return history

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

        Computes local triad state and derived dynamics:
        - expected_depi_dt = νf·ΔNFR
        - d2epi_dt2 from EPI history (finite differences)
        """
        if node not in self.G:
            raise TNFRValueError(
                f"Node '{node}' not found in network.",
                context={"node": node, "nodes": list(self.G.nodes())[:10]},
                suggestion="Use an existing node id from network.G.nodes().",
            )

        nd = self.G.nodes[node]
        epi = float(get_attr(nd, ALIAS_EPI, 0.0) or 0.0)
        nu_f = float(get_attr(nd, ALIAS_VF, 0.0) or 0.0)
        delta_nfr = float(get_attr(nd, ALIAS_DNFR, 0.0) or 0.0)
        phase = float(get_attr(nd, ALIAS_THETA, 0.0) or 0.0)
        expected_depi_dt = float(compute_expected_depi_dt(self.G, node))
        d2epi_dt2 = float(compute_d2epi_dt2(self.G, node))

        # ZHIR (Mutation) is the canonical bifurcation operator: a node nears
        # bifurcation when its structural change rate dEPI/dt = nu_f * dNFR
        # crosses the mutation threshold xi (AGENTS.md §5).
        xi = float(
            bifurcation_threshold
            if bifurcation_threshold is not None
            else self.G.graph.get("ZHIR_THRESHOLD_XI", _ZHIR_THRESHOLD_XI_DEFAULT)
        )

        return NodalStateReport(
            node=node,
            epi=epi,
            nu_f=nu_f,
            delta_nfr=delta_nfr,
            coherence=structural_coherence(delta_nfr),
            phase=phase,
            expected_depi_dt=expected_depi_dt,
            d2epi_dt2=d2epi_dt2,
            degree=int(self.G.degree(node)),
            equilibrium=is_structural_equilibrium(
                delta_nfr, eps_dnfr=float(equilibrium_tolerance)
            ),
            active=nu_f > 0.0,
            near_bifurcation=abs(expected_depi_dt) > xi,
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
        target_nodes = list(self.G.nodes()) if nodes is None else list(nodes)
        report_nodes: dict[Any, NodalStateReport] = {}
        for node in target_nodes:
            if node not in self.G:
                continue
            report_nodes[node] = self.nodal_state(
                node,
                equilibrium_tolerance=equilibrium_tolerance,
                bifurcation_threshold=bifurcation_threshold,
            )

        xi = float(
            bifurcation_threshold
            if bifurcation_threshold is not None
            else self.G.graph.get("ZHIR_THRESHOLD_XI", _ZHIR_THRESHOLD_XI_DEFAULT)
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
        """Get comprehensive results including tetrad and conservation."""
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

    # === CONSERVATION LAWS ===

    def conservation(self) -> ConservationReport:
        """Compute conservation law diagnostics (Noether charge, energy, Lyapunov).

        Uses the Structural Conservation Theorem (Noether-like) to compute:
        - Noether charge Q = sum(Phi_s + K_phi)
        - Energy functional E = 0.5 * sum(energy_density)
        - Lyapunov stability between the two most recent snapshots

        Returns
        -------
        ConservationReport
            noether_charge, energy, lyapunov_stable, lyapunov_derivative,
            conservation_quality.
        """
        if not _HAS_CONSERVATION:
            return ConservationReport()
        Q = compute_noether_charge(self.G)
        E = compute_energy_functional(self.G)
        # Lyapunov: requires two snapshots
        lyap_stable = True
        lyap_deriv = 0.0
        quality = 1.0
        if self._tracker is None:
            self._tracker = ConservationTracker(self.G)
        snap = capture_conservation_snapshot(self.G)
        if self._tracker._snapshots:
            prev = self._tracker._snapshots[-1]
            balance = verify_conservation_balance(prev, snap)
            quality = balance.conservation_quality
            lyap = compute_lyapunov_derivative(prev, snap)
            lyap_stable = lyap.is_stable
            lyap_deriv = lyap.energy_derivative
        self._tracker._snapshots.append(snap)
        return ConservationReport(
            noether_charge=Q,
            energy=E,
            lyapunov_stable=lyap_stable,
            lyapunov_derivative=lyap_deriv,
            conservation_quality=quality,
        )

    def symplectic_substrate(self) -> SymplecticReport:
        """Diagnose the emergent symplectic substrate.

        The TNFR nodal dynamics generates its own geometry: a symplectic
        phase space P = R^{4N} with canonical conjugate pairs (K_phi, J_phi)
        and (Phi_s, J_dnfr), on which the energy functional is the
        Hamiltonian and the 13 operators are symplectomorphisms.

        Returns
        -------
        SymplecticReport
            phase_space_dimension, hamiltonian H_sub, background_potential U
            (with H_sub + U = energy functional), liouville_divergence, and
            is_valid_manifold.
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
        """Compute full unified telemetry (all fields + invariants).

        Aggregates the complete Structural Field Tetrad, complex geometric
        field Psi, emergent fields (chirality, symmetry-breaking, coherence
        coupling), and tensor invariants (energy density, topological charge).

        Returns
        -------
        dict[str, Any]
            Complete unified telemetry suite.  Empty dict if fields module
            is unavailable.
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

    # === EMERGENT ONTOLOGY (particle / phase from graph state) ===

    def particle(self, order: list[Any] | None = None) -> dict[str, Any]:
        """Classify this network's coherent mode as an emergent particle.

        The class is an OUTPUT of the measured quantized topological winding
        number W of the phase field along a closed loop -- not an imposed
        label: ``|W| = 0`` -> boson/scalar-like, ``|W| = 1`` -> fermion-like,
        ``|W| > 1`` -> composite; ``sign(W)`` -> chirality (matter /
        antimatter-like). The integer ``W`` emerges *directly* as a topological
        invariant of the dynamics -- this is the **physical-layer** read-out of
        the same ``ΔNFR = 0`` fixed point whose **symbolic-layer** shadow is
        the structural prime (:meth:`TNFR.primes`). Energy- and charge-density
        telemetry support the integer invariant. Most informative after
        :meth:`evolve`.

        Parameters
        ----------
        order : list, optional
            Node order defining the closed loop along which the winding is
            measured. Defaults to the graph's node order.

        Returns
        -------
        dict
            winding, raw_winding, chirality, energy_density,
            charge_density_mean, particle_class, is_quantized.
        """
        from ..physics.emergent_particles import classify_particle

        return classify_particle(self.G, order=order).as_dict()

    def phase(self) -> dict[str, Any]:
        """Classify the network's emergent structural phase.

        Second-order symmetry breaking of the structural fields sets the
        phase: ``non_life`` (symmetric, <S> ~ 0, |<chi>| ~ 0), ``critical``
        (near the transition), or ``life`` (broken symmetry <S> != 0 with
        non-zero chirality chi). Uses a sampling z-score significance test --
        no magic constants. Most informative after :meth:`evolve`.

        Returns
        -------
        dict
            ``phase`` (``"non_life"`` / ``"critical"`` / ``"life"``),
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
        """Classify the network's emergent gauge-interaction regimes.

        The complex geometric field Psi = K_phi + i*J_phi carries an emergent
        U(1) gauge connection on edges and curvature F_C on plaquettes. Each
        node is classified into an interaction regime (em_like / weak_like /
        strong_like / gravity_like) from arg(Psi) and the structural
        potential. Most informative after :meth:`evolve`.

        Returns
        -------
        dict
            ``per_node``, ``regime_distribution``, ``dominant_regime``,
            ``mean_gauge_curvature`` (mean |F_C|) and ``gauge_flatness``.
        """
        from ..physics.gauge import classify_network_regimes

        return classify_network_regimes(self.G)

    def spectrum(self) -> dict[str, Any]:
        """Structural relaxation spectrum of the diffusion operator.

        The EPI channel of the nodal equation is exactly a graph diffusion
        dEPI/dt = -nu_f * L_rw * EPI. Its eigenmodes relax as
        exp(-nu_f*lambda_k*t): lambda_1 = 0 is the conserved uniform mode and
        the spectral gap lambda_2 (the Fiedler value) sets the slowest
        relaxation, the synchronization tendency, and -- via the coherence
        length xi_C ~ 1/sqrt(lambda_2) -- the structural correlation range.

        Returns
        -------
        dict
            ``diffusivity`` (mean nu_f), ``relaxation_rates``
            (nu_f*lambda_k ascending), ``spectral_gap`` (nu_f*lambda_2),
            ``structural_rank`` (distinct frequencies) and
            ``coherence_length`` (xi_C ~ 1/sqrt(spectral_gap)).
        """
        from ..physics.structural_diffusion import (
            relaxation_spectrum,
            structural_diffusivity,
            structural_frequency_rank,
        )

        rates = [float(r) for r in relaxation_spectrum(self.G)]
        gap = next((r for r in rates if r > 1e-12), 0.0)
        xi_c = 1.0 / float(np.sqrt(gap)) if gap > 1e-12 else float("inf")
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

        - RESONANT: proximity to the dNFR = 0 coherence attractor
          (:func:`~tnfr.metrics.common.is_structural_equilibrium`).
        - GEOMETRIC: the nodal topology radial / annular / multinodal
          (:func:`~tnfr.physics.fields.classify_nodal_topology`, read from the
          structural-potential geometry).
        - FRACTAL: the multi-scale coherence range xi_C (region size).

        A fully relaxed network (dNFR -> 0) is one uniform NFR; off-equilibrium
        the geometry differentiates sub-NFRs. Most informative after
        :meth:`evolve`.

        Returns
        -------
        dict
            ``topology``, ``centers``, ``concentration`` (geometry);
            ``coherence``, ``equilibrium_fraction`` (resonance);
            ``coherence_length`` (xi_C, fractal/region scale); ``triad`` (mean
            EPI, nu_f and the Kuramoto phase synchrony); ``n_nodes``.
        """
        from ..physics.fields import classify_nodal_topology

        topo = classify_nodal_topology(self.G)
        nodes = list(self.G.nodes())
        n = len(nodes)
        if n:
            dnfr = [
                float(get_attr(self.G.nodes[k], ALIAS_DNFR, 0.0) or 0.0) for k in nodes
            ]
            eq_frac = sum(1 for d in dnfr if is_structural_equilibrium(d)) / n
            epi_mean = (
                sum(
                    float(get_attr(self.G.nodes[k], ALIAS_EPI, 0.0) or 0.0)
                    for k in nodes
                )
                / n
            )
            vf_mean = (
                sum(
                    float(get_attr(self.G.nodes[k], ALIAS_VF, 0.0) or 0.0)
                    for k in nodes
                )
                / n
            )
            thetas = [
                float(get_attr(self.G.nodes[k], ALIAS_THETA, 0.0) or 0.0) for k in nodes
            ]
            if np is not None:
                phase_sync = float(abs(np.mean(np.exp(1j * np.asarray(thetas)))))
            else:
                phase_sync = 0.0
        else:
            eq_frac = epi_mean = vf_mean = phase_sync = 0.0
        try:
            # Spectral coherence length xi_C ~ 1/sqrt(nu_f*lambda_2): the
            # structural region scale, robust at the uniform dNFR=0 equilibrium
            # (where the correlation-fit xi_C is undefined).
            xi_c = float(self.spectrum()["coherence_length"])
        except Exception:
            xi_c = float("nan")
        return {
            "topology": topo["topology"],
            "centers": topo["centers"],
            "concentration": topo["concentration"],
            "coherence": self.coherence(),
            "equilibrium_fraction": eq_frac,
            "coherence_length": xi_c,
            "triad": {
                "epi_mean": epi_mean,
                "vf_mean": vf_mean,
                "phase_sync": phase_sync,
            },
            "n_nodes": n,
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

    def noether_charge(self) -> float:
        """Total Noether charge Q = sum_i [Phi_s(i) + K_phi(i)]."""
        if not _HAS_CONSERVATION:
            return 0.0
        return compute_noether_charge(self.G)

    def energy(self) -> float:
        """Total structural energy E = 0.5 * sum_i energy_density(i)."""
        if not _HAS_CONSERVATION:
            return 0.0
        return compute_energy_functional(self.G)

    def grammar_violations(self, dt: float = 1.0) -> dict[str, Any]:
        """Detect grammar violations via conservation residuals.

        Requires at least two snapshots.  Takes snapshots before and
        after a single compliant evolution step (dt) and checks the
        conservation balance.

        Returns
        -------
        dict with violations_detected, violation_types, severity, etc.
        """
        if not _HAS_CONSERVATION:
            return {
                "violations_detected": False,
                "violation_types": [],
                "severity": 0.0,
            }
        from ..physics.conservation import detect_grammar_violations_from_conservation

        snap_before = capture_conservation_snapshot(self.G)
        # Small stabilisation step for delta measurement
        for n in self.G.nodes():
            neighbors = list(self.G.neighbors(n))
            if neighbors:
                mean_ph = float(
                    np.mean(
                        [
                            get_attr(self.G.nodes[nb], ALIAS_THETA, 0.0)
                            for nb in neighbors
                        ]
                    )
                )
                cur = get_attr(self.G.nodes[n], ALIAS_THETA, 0.0)
                set_attr(
                    self.G.nodes[n],
                    ALIAS_THETA,
                    cur + dt * 0.05 * (mean_ph - cur),
                )
        snap_after = capture_conservation_snapshot(self.G)
        balance = verify_conservation_balance(snap_before, snap_after, dt=dt * 0.05)
        return detect_grammar_violations_from_conservation(balance)

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

    # === INTEGRITY MONITORING ===

    def integrity_check(self, operator_name: str = "coherence") -> dict[str, Any]:
        """Run structural integrity check via postcondition monitor.

        Parameters
        ----------
        operator_name : str
            Operator NAME whose postconditions are verified (the canonical
            public identifier, AGENTS.md §5; default 'coherence'). Glyph
            codes are accepted but only names resolve a postcondition check.

        Returns
        -------
        dict[str, Any]
            Integrity report with conservation_quality, lyapunov status,
            charge drift, and per-node diagnostics.  Returns empty dict
            if integrity module is unavailable.
        """
        if not _HAS_INTEGRITY:
            return {}
        if self._monitor is None:
            self._monitor = StructuralIntegrityMonitor(mode=MonitorMode.OBSERVE)
        reports: list[dict[str, Any]] = []
        for node in list(self.G.nodes())[: min(10, len(self.G.nodes()))]:
            try:
                report = self._monitor.after_operator(self.G, node, operator_name)
                reports.append(
                    {
                        "node": node,
                        "passed": report.is_healthy,
                        "details": str(report),
                    }
                )
            except Exception:
                continue
        passed_count = sum(1 for r in reports if r.get("passed", False))
        return {
            "operator": operator_name,
            "nodes_checked": len(reports),
            "passed": passed_count,
            "failed": len(reports) - passed_count,
            "pass_rate": passed_count / max(len(reports), 1),
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
        if num_nodes < 0:
            raise TNFRValueError(
                "num_nodes must be non-negative.",
                context={"num_nodes": num_nodes},
            )
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
        ``ΔNFR`` EPI channel) on the residue/Paley graph -- a *partially
        emergent* read-out, not the symbolic per-node ``ΔNFR`` (which is blind
        to the cosets). Honest scope: the residue graph is regular, so ``L_rw``
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
            If provided, computes synergy metrics between the factorization
            telemetry and this network's structural state.
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
        """Canonical primality analysis via SDK, optionally fused with telemetry.

        Reads the arithmetic equilibrium ``ΔNFR_arith(n) = 0`` (**sector A** --
        an exact but *circular* re-expression that consumes ``n``'s
        divisibility; the genuinely emergent primality is the spectral
        sector B, see :meth:`primes` and theory §9.5).
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
        equilibrium (:func:`tnfr.metrics.common.is_structural_equilibrium`)
        that a relaxed graph node and a noble-gas atom also satisfy.

        Emergence status (theory/TNFR_NUMBER_THEORY.md §9.5). This method is
        **sector A**: the arithmetic ``ΔNFR`` is an *exact but circular*
        re-expression that **consumes** ``n``'s divisibility ``(Ω, τ, σ)`` --
        the symbolic shadow of the fixed point whose *physical*-layer read-out
        is the directly-emergent particle winding ``W``
        (:meth:`Network.particle`). The genuinely **emergent** primality is
        **sector B** -- the spectral Paley/residue Fiedler gap (input only
        ``x^2 mod n``, primes-OUT, non-circular), carried by the spectral
        factorizer (:meth:`factorize`), not by this arithmetic read-out. One
        fixed point, a spectrum of emergence.

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

        net = ArithmeticTNFRNetwork(int(max_number))
        candidates = net.detect_prime_candidates()
        primes = sorted(int(n) for n, _delta in candidates)
        return {
            "max_number": int(max_number),
            "primes": primes,
            "count": len(primes),
        }

    @staticmethod
    def magic_numbers(max_n: int = 7) -> list[int]:
        """Noble-gas atomic numbers from the structural shell-filling equilibrium.

        The closed-shell counts (2, 10, 18, 36, 54, 86, ...) combine two
        ingredients of *different* status:

        - the subshell capacities ``2l+1`` **emerge** from the degenerate
          eigenmodes of a closed structural manifold (its phase-curvature /
          Laplace-Beltrami spectrum) -- a genuine dynamical emergence;
        - the ``(n+l)`` filling order is an **assumed** integer count rule
          (Madelung), retained because the free manifold spectrum does not by
          itself reproduce it.

        A closed shell is ``ΔNFR_chem = 0``
        (:func:`tnfr.metrics.common.is_structural_equilibrium`): the chemical
        read-out of the same fixed point as the structural prime -- another
        symbolic-layer shadow of the equilibrium process whose physical
        read-out is the particle winding.

        Parameters
        ----------
        max_n : int
            Highest principal shell index to fill.

        Returns
        -------
        list[int]
            The noble-gas atomic numbers.
        """
        from ..physics.emergent_chemistry import emergent_magic_numbers

        return [int(z) for z in emergent_magic_numbers(max_n=int(max_n))]

    @staticmethod
    def element(Z: int, *, max_n: int = 7) -> dict[str, Any]:
        """Structural characterization of the element with count Z.

        Pure-TNFR atomic structure: electron configuration, valence count,
        structural valence pressure ``ΔNFR_chem`` and reactivity ``|ΔNFR|``.
        The subshell capacities ``2l+1`` emerge from the manifold eigenmode
        degeneracies; the ``(n+l)`` aufbau order is an *assumed* integer count
        rule, not a spectral derivation. A closed shell is ``ΔNFR_chem(Z) = 0``
        -- the chemical read-out of the *same* nodal-equation fixed point
        (:func:`tnfr.metrics.common.is_structural_equilibrium`) as the
        primality criterion ``ΔNFR_arith(n) = 0``: the symbolic-layer shadow of
        the equilibrium process whose physical read-out is the particle
        (:meth:`Network.particle`).

        Parameters
        ----------
        Z : int
            Atomic number (count of filled structural eigenmodes).
        max_n : int
            Highest principal shell index available.

        Returns
        -------
        dict
            Z, configuration, valence_electrons, outer_shell_n, delta_nfr,
            closed_shell, magic_number, reactivity, config_label.
        """
        from ..physics.emergent_chemistry import classify_element

        return classify_element(int(Z), max_n=int(max_n)).as_dict()

    @staticmethod
    def weyl_spectrum(k: int = 1) -> dict[str, Any]:
        """Weyl spectral asymptotics of the k-th TNFR-Riemann operator.

        Research diagnostic for the TNFR-Riemann program (theory/
        TNFR_RIEMANN_RESEARCH_NOTES.md): the eigenvalue counting function
        N(lambda) = #{lambda_j <= lambda} ~ A * lambda^alpha of the
        prime-weighted structural Laplacian. The exponent alpha encodes the
        spectral dimension (alpha = 1/2 for a uniform 1D chain; prime-gap
        weights modify it). The operator *consumes* the prime support (``νf``
        weighted by the primes) as input, so this is a symbolic-layer probe of
        the equilibrium structure -- a NUMERICAL diagnostic of the
        sigma_c -> 1/2 program, NOT a direct emergence and NOT a proof of the
        Riemann Hypothesis.

        Parameters
        ----------
        k : int
            Operator index in the TNFR-Riemann spectral family.

        Returns
        -------
        dict
            ``k``, ``alpha`` (Weyl exponent), ``A_coeff`` (prefactor),
            ``r_squared`` (log-log fit quality) and ``n_eigenvalues``.
        """
        from ..riemann.zeta_bridge import compute_weyl_asymptotic

        w = compute_weyl_asymptotic(int(k))
        return {
            "k": int(w.k),
            "alpha": float(w.alpha),
            "A_coeff": float(w.A_coeff),
            "r_squared": float(w.r_squared),
            "n_eigenvalues": int(len(w.eigenvalues)),
        }

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
            ".noether_charge()              STRUCTURAL_CONSERVATION_THEOREM.md             34",
            ".energy()                      STRUCTURAL_CONSERVATION_THEOREM.md             34",
            ".nodal_state(node)             TNFR.pdf §2.1 (nodal equation)                 04, 05",
            ".nodal_scan()                  TNFR.pdf §2.1 + U4 bifurcation diagnostics      07",
            ".nodal_profile(node)           TNFR nodal telemetry bridge                     10",
            ".nfr()                         TNFR.pdf §1.4.1 (NFR region of coherence)        —",
            ".grammar_violations()          STRUCTURAL_CONSERVATION_THEOREM.md ss12        36",
            ".telemetry()                   FUNDAMENTAL_THEORY.md                         10",
            ".auto_optimize()               AGENTS.md § Self-Optimizing Dynamics           30",
            "TNFR.factorize(n)              TNFR_NUMBER_THEORY.md                          40",
            "Network.factorize(n)           TNFR_NUMBER_THEORY.md + SDK telemetry bridge   40",
            "TNFR.primality(n)              TNFR_NUMBER_THEORY.md                          40",
            "Network.primality(n)           TNFR_NUMBER_THEORY.md + SDK telemetry bridge   40",
            "TNFR.analyze(net)              APPLIED_STRUCTURAL_ANALYSIS.md                 10",
            "",
            "New theory-experiment links (v0.0.3.2):",
            "  31 — Mathematical constants basis (phi, gamma, pi, e)",
            "  32 — Spiral attractors (golden spiral, KAM)",
            "  33 — Complex field unification (Psi = K_phi + i*J_phi)",
            "  34 — Conservation protocol suite (Noether, Lyapunov)",
            "  35 — Tetrad irreducibility (blind spot verification)",
            "  36 — Grammar violation detector (conservation residuals)",
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

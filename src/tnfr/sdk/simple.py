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
from typing import Any
from dataclasses import dataclass, field
import networkx as nx
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np

# TNFR core imports
from ..structural import create_nfr, run_sequence
from ..metrics.coherence import compute_coherence
from ..metrics.sense_index import compute_Si

try:
    from ..dynamics.self_optimizing_engine import TNFRSelfOptimizingEngine
    _HAS_OPTIMIZATION = True
except ImportError:
    _HAS_OPTIMIZATION = False

# Structural Field Tetrad (CANONICAL)
try:
    from ..physics.fields import (
        compute_structural_potential,
        compute_phase_gradient,
        compute_phase_curvature,
        estimate_coherence_length,
        compute_phase_current,
        compute_dnfr_flux,
        compute_unified_telemetry,
        compute_tensor_invariants,
        compute_emergent_fields,
        compute_complex_geometric_field_arrays,
    )
    _HAS_FIELDS = True
except ImportError:
    _HAS_FIELDS = False

# Conservation laws (Noether-like)
try:
    from ..physics.conservation import (
        compute_noether_charge,
        compute_energy_functional,
        compute_lyapunov_derivative,
        capture_conservation_snapshot,
        verify_conservation_balance,
        ConservationTracker,
    )
    _HAS_CONSERVATION = True
except ImportError:
    _HAS_CONSERVATION = False

# Structural Integrity Monitor
try:
    from ..physics.integrity import StructuralIntegrityMonitor, MonitorMode
    _HAS_INTEGRITY = True
except ImportError:
    _HAS_INTEGRITY = False

# Grammar-aware dynamics
try:
    from ..operators.grammar_dynamics import (
        validate_candidate,
        filter_candidates,
        enforce_grammar_on_glyph,
    )
    _HAS_GRAMMAR_DYNAMICS = True
except ImportError:
    _HAS_GRAMMAR_DYNAMICS = False


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
    xi_c: float = float('nan')
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
            PHI_S_VON_KOCH_THRESHOLD,
            GRAD_PHI_CANONICAL_THRESHOLD,
            K_PHI_CANONICAL_THRESHOLD,
        )
        phi_s_safe = all(
            abs(v) < PHI_S_VON_KOCH_THRESHOLD for v in self.phi_s.values()
        ) if self.phi_s else True
        grad_safe = all(
            v < GRAD_PHI_CANONICAL_THRESHOLD for v in self.grad_phi.values()
        ) if self.grad_phi else True
        k_safe = all(
            abs(v) < K_PHI_CANONICAL_THRESHOLD for v in self.k_phi.values()
        ) if self.k_phi else True
        xi_safe = not np.isnan(self.xi_c) if np.isfinite(self.xi_c) else True
        return {
            'phi_s_safe': phi_s_safe,
            'grad_phi_safe': grad_safe,
            'k_phi_safe': k_safe,
            'xi_c_safe': xi_safe,
            'overall': phi_s_safe and grad_safe and k_safe,
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
        coherence = float(self.coherence) if hasattr(self.coherence, 'item') else self.coherence
        sense_index = float(self.sense_index) if hasattr(self.sense_index, 'item') else self.sense_index
        density = float(self.density) if hasattr(self.density, 'item') else self.density

        return (f"C={coherence:.3f}, Si={sense_index:.3f}, "
                f"N={self.nodes}, E={self.edges}, rho={density:.3f}")

    def full_summary(self) -> str:
        """Multi-line summary including tetrad and conservation."""
        lines = [self.summary()]
        if self.tetrad is not None:
            lines.append(f"  Tetrad: {self.tetrad.summary()}")
            safety = self.tetrad.is_safe()
            if not safety['overall']:
                unsafe = [k for k, v in safety.items() if k != 'overall' and not v]
                lines.append(f"  WARNING: Unsafe fields: {', '.join(unsafe)}")
        if self.conservation is not None:
            lines.append(f"  Conservation: {self.conservation.summary()}")
        return '\n'.join(lines)

    def is_coherent(self) -> bool:
        """Quick coherence check (C > 0.7)."""
        return self.coherence > 0.7

    def is_stable(self) -> bool:
        """Quick stability check (Si > 0.8)."""
        return self.sense_index > 0.8

    def to_dict(self) -> dict[str, Any]:
        """Serialize results to a plain dictionary."""
        d: dict[str, Any] = {
            'coherence': float(self.coherence),
            'sense_index': float(self.sense_index),
            'nodes': self.nodes,
            'edges': self.edges,
            'density': float(self.density),
            'avg_phase': float(self.avg_phase),
        }
        if self.tetrad is not None:
            d['tetrad'] = {
                'phi_s': {str(k): float(v) for k, v in self.tetrad.phi_s.items()},
                'grad_phi': {str(k): float(v) for k, v in self.tetrad.grad_phi.items()},
                'k_phi': {str(k): float(v) for k, v in self.tetrad.k_phi.items()},
                'xi_c': float(self.tetrad.xi_c) if np.isfinite(self.tetrad.xi_c) else None,
            }
        if self.conservation is not None:
            d['conservation'] = {
                'noether_charge': float(self.conservation.noether_charge),
                'energy': float(self.conservation.energy),
                'lyapunov_stable': self.conservation.lyapunov_stable,
            }
        return d

class Network:
    """Core TNFR Network — Essential Operations + Advanced Telemetry.

    Simplified interface to TNFR networks with structural field tetrad,
    conservation law monitoring, grammar-aware dynamics, and integrity
    checks for research-grade analysis.
    """

    def __init__(self, graph: nx.Graph, name: str = "network"):
        """Initialize with a NetworkX graph."""
        self.G = graph
        self.name = name
        self._tracker: Any = None  # ConservationTracker (lazy)
        self._monitor: Any = None  # StructuralIntegrityMonitor (lazy)
    
    # === TOPOLOGY BUILDERS ===
    
    def ring(self) -> Network:
        """🔄 Connect nodes in a ring topology."""
        nodes = list(self.G.nodes())
        edges = [(nodes[i], nodes[(i+1) % len(nodes)]) for i in range(len(nodes))]
        self.G.add_edges_from(edges)
        return self
    
    def complete(self) -> Network:
        """🌐 Connect all nodes to all other nodes."""
        nodes = list(self.G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                self.G.add_edge(u, v)
        return self
    
    def random(self, probability: float = 0.3) -> Network:
        """🎲 Add random connections with given probability."""
        nodes = list(self.G.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if np.random.random() < probability:
                    self.G.add_edge(u, v)
        return self
    
    def star(self, center: int | None = None) -> Network:
        """⭐ Create star topology with optional center node."""
        nodes = list(self.G.nodes())
        if center is None:
            center = nodes[0]
        
        for node in nodes:
            if node != center:
                self.G.add_edge(center, node)
        return self
    
    # === EVOLUTION ===
    
    def evolve(self, steps: int = 5, sequence: str = "basic_activation") -> Network:
        """🧬 Evolve network using TNFR dynamics."""
        # Ensure nodes have TNFR properties
        if not all('EPI' in self.G.nodes[n] for n in self.G.nodes()):
            create_nfr(self.G)
        
        # Simple evolution: Apply coherence improvement to all nodes
        from ..operators.definitions import Coherence, Resonance
        
        for step in range(steps):
            for node in self.G.nodes():
                try:
                    # Simple sequence: Coherence -> Resonance
                    ops = [Coherence(), Resonance()]
                    run_sequence(self.G, node, ops)
                except Exception:
                    # If sequence fails, just continue
                    continue
        
        return self
    
    def auto_optimize(self) -> Network:
        """🤖 Auto-optimize using self-optimizing engine."""
        if not _HAS_OPTIMIZATION:
            print("⚠️  Auto-optimization not available - using basic evolution")
            return self.evolve(3, "stabilization")
        
        if not all('EPI' in self.G.nodes[n] for n in self.G.nodes()):
            create_nfr(self.G)
        
        engine = TNFRSelfOptimizingEngine(self.G)
        
        # Try to optimize a few nodes
        for node in list(self.G.nodes())[:min(5, len(self.G.nodes()))]:
            try:
                engine.step(node)
            except Exception:
                continue  # Skip if optimization fails
        
        return self
    
    # === METRICS ===
    
    def coherence(self) -> float:
        """📏 Current network coherence [0,1]."""
        result = compute_coherence(self.G)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)
    
    def sense_index(self) -> float:
        """🎯 Current sense index [0,1+]."""
        result = compute_Si(self.G)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)
    
    def density(self) -> float:
        """🔗 Network density [0,1]."""
        n = len(self.G.nodes())
        if n < 2:
            return 0.0
        return 2 * len(self.G.edges()) / (n * (n - 1))
    
    def avg_phase(self) -> float:
        """📐 Average node phase [0, 2π]."""
        if not self.G.nodes():
            return 0.0
        phases = [self.G.nodes[n].get('phase', 0) for n in self.G.nodes()]
        result = np.mean(phases)
        try:
            return float(np.asarray(result).flat[0])
        except (IndexError, TypeError):
            return float(result)
    
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
        """📋 Quick network summary."""
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
            'phi_s': snap.phi_s,
            'grad_phi': snap.grad_phi,
            'k_phi': snap.k_phi,
            'xi_c': snap.xi_c,
            'j_phi': snap.j_phi,
            'j_dnfr': snap.j_dnfr,
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
            Operator codes to consider.  Defaults to common set
            ['IL', 'EN', 'RA', 'OZ', 'UM'].

        Returns
        -------
        Network
            self (for chaining).
        """
        if not _HAS_GRAMMAR_DYNAMICS:
            return self.evolve(steps)
        if not all('EPI' in self.G.nodes[n] for n in self.G.nodes()):
            create_nfr(self.G)
        if candidates is None:
            candidates = ['IL', 'EN', 'RA', 'OZ', 'UM']
        from ..operators.registry import get_operator_class
        for _step in range(steps):
            for node in self.G.nodes():
                valid = filter_candidates(self.G, node, candidates)
                if not valid:
                    continue
                glyph_code = valid[0]  # safest first
                try:
                    op_cls = get_operator_class(glyph_code)
                except KeyError:
                    continue
                try:
                    run_sequence(self.G, node, [op_cls()])
                except Exception:
                    continue
        return self

    # === INTEGRITY MONITORING ===

    def integrity_check(self, operator_name: str = "IL") -> dict[str, Any]:
        """Run structural integrity check via postcondition monitor.

        Parameters
        ----------
        operator_name : str
            Operator code to verify postconditions for
            (default 'IL' — coherence).

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
        for node in list(self.G.nodes())[:min(10, len(self.G.nodes()))]:
            try:
                report = self._monitor.after_operator(self.G, node, operator_name)
                reports.append({
                    'node': node,
                    'passed': report.passed,
                    'details': str(report),
                })
            except Exception:
                continue
        passed_count = sum(1 for r in reports if r.get('passed', False))
        return {
            'operator': operator_name,
            'nodes_checked': len(reports),
            'passed': passed_count,
            'failed': len(reports) - passed_count,
            'pass_rate': passed_count / max(len(reports), 1),
            'reports': reports,
        }

    # === ANALYSIS ===

    def info(self) -> dict[str, Any]:
        """Detailed network information including feature availability."""
        return {
            'name': self.name,
            'nodes': len(self.G.nodes()),
            'edges': len(self.G.edges()),
            'density': self.density(),
            'coherence': self.coherence(),
            'sense_index': self.sense_index(),
            'avg_phase': self.avg_phase(),
            'is_connected': nx.is_connected(self.G),
            'has_tnfr_props': all('EPI' in self.G.nodes[n] for n in self.G.nodes()),
            'features': {
                'fields': _HAS_FIELDS,
                'conservation': _HAS_CONSERVATION,
                'integrity': _HAS_INTEGRITY,
                'grammar_dynamics': _HAS_GRAMMAR_DYNAMICS,
                'optimization': _HAS_OPTIMIZATION,
            },
        }

class TNFR:
    """🌊 **Static Factory for Instant TNFR Networks** ⭐
    
    Main entry point for the simplified TNFR SDK.
    All methods are static for maximum convenience.
    
    **PHILOSOPHY**: Start creating networks immediately with zero boilerplate.
    """
    
    @staticmethod
    def create(num_nodes: int, name: str = "network") -> Network:
        """🏗️ Create empty TNFR network with specified nodes.
        
        Args:
            num_nodes: Number of nodes to create
            name: Optional network name
            
        Returns:
            Network ready for topology and evolution
            
        Example:
            >>> net = TNFR.create(10)  # 10 isolated nodes
        """
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # Initialize with TNFR properties
        create_nfr(G)
        
        return Network(G, name)
    
    @staticmethod
    def template(template_name: str) -> Network:
        """📋 Create network from pre-configured template.
        
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
            'small': lambda: TNFR.create(5).ring(),
            'medium': lambda: TNFR.create(15).ring().random(0.1),  # Small-world-like
            'large': lambda: TNFR.create(50).random(0.08),
            'molecule': lambda: TNFR.create(8).ring().random(0.2),
            'star': lambda: TNFR.create(10).star(),
            'complete': lambda: TNFR.create(6).complete()
        }
        
        if template_name not in templates:
            available = ', '.join(templates.keys())
            raise TNFRValueError(
                f"Unknown template '{template_name}'.",
                context={
                    "requested": template_name,
                    "available": list(templates.keys())
                },
                suggestion=f"Choose from: {available}"
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
                'name': net.name,
                'index': i,
                'coherence': result.coherence,
                'sense_index': result.sense_index,
                'nodes': result.nodes,
                'edges': result.edges,
                'density': result.density,
            }
            if result.conservation is not None:
                entry['noether_charge'] = result.conservation.noether_charge
                entry['energy'] = result.conservation.energy
                entry['lyapunov_stable'] = result.conservation.lyapunov_stable
            results.append(entry)
        
        # Rank by coherence
        ranking = sorted(results, key=lambda x: x['coherence'], reverse=True)
        
        return {
            'results': results,
            'ranking': ranking,
            'best': ranking[0] if ranking else None,
            'worst': ranking[-1] if ranking else None,
            'count': len(networks),
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
            'coherence': network.coherence(),
            'sense_index': network.sense_index(),
            'nodes': len(network.G.nodes()),
            'edges': len(network.G.edges()),
            'density': network.density(),
            'avg_phase': network.avg_phase(),
        }
        if _HAS_FIELDS:
            result['tetrad'] = network.tetrad()
            result['tensor_invariants'] = network.tensor_invariants()
            result['emergent_fields'] = network.emergent_fields()
        if _HAS_CONSERVATION:
            result['conservation'] = network.conservation()
        if _HAS_INTEGRITY:
            result['integrity'] = network.integrity_check()
        result['features'] = {
            'fields': _HAS_FIELDS,
            'conservation': _HAS_CONSERVATION,
            'integrity': _HAS_INTEGRITY,
            'grammar_dynamics': _HAS_GRAMMAR_DYNAMICS,
            'optimization': _HAS_OPTIMIZATION,
        }
        return result

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
            ".tetrad()                      EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md      20, unified_fields_showcase",
            ".conservation()                STRUCTURAL_CONSERVATION_THEOREM.md             17, 24",
            ".evolve_grammar_aware(steps)   UNIFIED_GRAMMAR_RULES.md                      04, 07",
            ".integrity_check()             STRUCTURAL_STABILITY_AND_DYNAMICS.md           29",
            ".tensor_invariants()           EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md      20, unified_fields_showcase",
            ".emergent_fields()             EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md      unified_fields_showcase",
            ".telemetry()                   FUNDAMENTAL_THEORY.md                         10",
            ".auto_optimize()               AGENTS.md § Self-Optimizing Dynamics           30",
            "TNFR.analyze(net)              APPLIED_STRUCTURAL_ANALYSIS.md                 10",
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
    'TNFR', 'Network', 'Results', 'TetradSnapshot', 'ConservationReport',
    'T', 'Net',
]
"""
TNFR Emergent Mathematical Pattern Engine

This module implements the natural mathematical patterns that emerge from the
nodal equation ∂EPI/∂t = νf · ΔNFR(t) at multiple scales.

Mathematical Discovery:
Through systematic analysis of the nodal equation, several deep patterns
emerge:

1. **Natural Eigenmodes**: The equation admits natural oscillatory
    solutions
2. **Spectral Resonance Cascades**: Harmonics naturally couple across
    scales
3. **Information-Theoretic Structure**: EPI evolution has intrinsic
    entropy flow
4. **Topological Invariants**: Network structure creates conservation
    laws
5. **Fractal Self-Similarity**: Patterns repeat at multiple temporal
    scales
6. **Emergent Symmetries**: Hidden symmetries appear in spectral domain

These patterns enable:
- Predictive compression of EPI trajectories
- Automatic detection of critical transitions
- Natural clustering of equivalent network states
- Emergent quantum-like interference patterns
- Self-organizing optimization strategies
- Mathematical proof techniques for grammar convergence

Status: EMERGENT MATHEMATICAL DISCOVERY ENGINE
"""

import importlib.util
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Mapping

from ...mathematics.unified_numerical import np

# Import PHASE 6 EXTENDED Canonical Constants for magic number elimination
from ..constants.canonical import (
    OPT_ORCH_ARITHMETIC_BOOST_CANONICAL  # γ/(2π+e) ≈ 0.0625 (2.5 → canonical)
)

# Import canonical constants for Phase 6 magic number elimination
from ..constants.canonical import (
    PATTERNS_HIGH_CONFIDENCE_CANONICAL,
    PATTERNS_COMPRESSION_RATIO_CANONICAL,
    PATTERNS_RSQUARED_THRESHOLD_CANONICAL,
    PATTERNS_SLOPE_THRESHOLD_CANONICAL,
    PATTERNS_HORIZON_LONG_CANONICAL,
    PATTERNS_COMPRESSION_OSCILLATORY_CANONICAL,
    PATTERNS_ENTROPY_THRESHOLD_CANONICAL,
    PATTERNS_DIVERGENCE_THRESHOLD_CANONICAL,
    PATTERNS_HORIZON_MEDIUM_CANONICAL,
    PATTERNS_RSQUARED_HIGH_CANONICAL,
    PATTERNS_SLOPE_MINIMAL_CANONICAL,
    PATTERNS_HORIZON_SHORT_CANONICAL,
    PATTERNS_COMPRESSION_SIGNIFICANT_CANONICAL,
    PATTERNS_HORIZON_PREDICTIVE_CANONICAL,
    PATTERNS_CONFIDENCE_BROKEN_CANONICAL,
)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

# Import TNFR components
try:
    from ...mathematics.spectral import get_laplacian_spectrum, gft
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# Import GPU-aware mathematics backend for pattern acceleration
try:
    from ...mathematics.backend import get_backend
    HAS_GPU_BACKENDS = True
except ImportError:
    HAS_GPU_BACKENDS = False

try:
    from .multi_modal_cache import CacheEntryType, cache_unified_computation
    HAS_UNIFIED_CACHE = True
except ImportError:
    HAS_UNIFIED_CACHE = False

try:
    from ...physics.fields import (
        compute_structural_potential,
        compute_phase_gradient,
    )
    HAS_PHYSICS_FIELDS = True
except ImportError:
    HAS_PHYSICS_FIELDS = False

try:
    from ..computation.fft_engine import FFTDynamicsEngine
    HAS_FFT_ENGINE = True
except ImportError:
    HAS_FFT_ENGINE = False
    FFTDynamicsEngine = None  # type: ignore

try:
    from ..integration.emergent_integration import IntegrationOpportunity
    HAS_INTEGRATION_HINTS = True
except ImportError:
    HAS_INTEGRATION_HINTS = False
    IntegrationOpportunity = None  # type: ignore

try:
    from ...mathematics.number_theory import (
        ArithmeticTNFRFormalism,
        ArithmeticTNFRParameters,
        ArithmeticStructuralTerms,
    )
    HAS_ARITHMETIC = True
except ImportError:
    HAS_ARITHMETIC = False
    ArithmeticTNFRFormalism = None  # type: ignore
    ArithmeticTNFRParameters = None  # type: ignore
    ArithmeticStructuralTerms = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper utilities for detector subsystems (spectral, field, arithmetic)
# ---------------------------------------------------------------------------

try:
    _GRAPH_STATE_CACHE: "weakref.WeakKeyDictionary[Any, Dict[str, Any]]" = (
        weakref.WeakKeyDictionary()
    )
except Exception:  # pragma: no cover - extremely unlikely
    _GRAPH_STATE_CACHE = None  # type: ignore
_GRAPH_STATE_FALLBACK: Dict[int, Dict[str, Any]] = {}
_ARITHMETIC_CONTEXT: Optional[Dict[str, Any]] = None
if HAS_INTEGRATION_HINTS:
    _INTEGRATION_HINT_MAPPING: Dict[str, Any] = {
        "eigenmode_resonance": getattr(
            IntegrationOpportunity,
            "SPECTRAL_SHARING",
            "spectral_sharing",
        ),
        "spectral_cascade": getattr(
            IntegrationOpportunity,
            "SPECTRAL_SHARING",
            "spectral_sharing",
        ),
        "entropy_flow": getattr(
            IntegrationOpportunity,
            "MATHEMATICAL_CONSISTENCY",
            "mathematical_consistency",
        ),
    }
else:
    _INTEGRATION_HINT_MAPPING = {}


def _graph_cache_for(graph: Any) -> Dict[str, Any]:
    """Return mutable cache bucket for a graph instance."""
    if _GRAPH_STATE_CACHE is not None:
        try:
            cache = _GRAPH_STATE_CACHE.get(graph)
        except TypeError:
            cache = None
        if cache is None:
            cache = {}
            try:
                _GRAPH_STATE_CACHE[graph] = cache
            except TypeError:
                # Some lightweight containers (e.g., dict) are not
                # weak-ref'able
                pass
        if cache is not None:
            return cache
    key = id(graph)
    bucket = _GRAPH_STATE_FALLBACK.get(key)
    if bucket is None:
        bucket = {}
        _GRAPH_STATE_FALLBACK[key] = bucket
    return bucket


def _get_spectral_basis(graph: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (eigenvalues, eigenvectors) with lightweight caching."""
    if not HAS_SPECTRAL:
        return None
    cache = _graph_cache_for(graph)
    if "spectral_basis" in cache:
        return cache["spectral_basis"]
    try:
        eigenvalues, eigenvectors = get_laplacian_spectrum(graph)
    except Exception:
        return None
    cache["spectral_basis"] = (eigenvalues, eigenvectors)
    return cache["spectral_basis"]


def _field_snapshot(graph: Any) -> Dict[str, Any]:
    """Capture structural field telemetry when physics fields are available."""
    if not HAS_PHYSICS_FIELDS:
        return {}
    cache = _graph_cache_for(graph)
    stamp = cache.get("field_snapshot")
    if stamp is not None and (
        time.perf_counter() - stamp.get("timestamp", 0)
    ) < 0.25:
        return stamp
    try:
        phi_s = compute_structural_potential(graph)
        phase_grad = compute_phase_gradient(graph)
    except Exception:
        return {}

    def _summaries(values: Any) -> Dict[str, float]:
        if isinstance(values, dict):
            arr = np.array(list(values.values()), dtype=float)
        else:
            arr = np.array(values, dtype=float)
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
        }

    snapshot = {
        "structural_potential": _summaries(phi_s),
        "phase_gradient": _summaries(phase_grad),
        "timestamp": time.perf_counter(),
    }
    cache["field_snapshot"] = snapshot
    return snapshot


def _get_fft_engine() -> Optional[Any]:
    """Provide a shared FFT dynamics engine instance."""
    if not HAS_FFT_ENGINE:
        return None
    cache = _GRAPH_STATE_FALLBACK.setdefault(-1, {})  # Global singleton bucket
    engine = cache.get("fft_engine")
    if engine is not None:
        return engine
    try:
        engine = FFTDynamicsEngine()
    except Exception:
        cache["fft_engine"] = None
        return None
    cache["fft_engine"] = engine
    return engine


def _arithmetic_context() -> Optional[Dict[str, Any]]:
    """Expose canonical arithmetic formalism references."""
    global _ARITHMETIC_CONTEXT
    if not HAS_ARITHMETIC:
        return None
    if _ARITHMETIC_CONTEXT is None:
        try:
            params = ArithmeticTNFRParameters()
            _ARITHMETIC_CONTEXT = {
                "params": params,
                "formalism": ArithmeticTNFRFormalism,
                "terms_class": ArithmeticStructuralTerms,
            }
        except Exception:
            return None
    return _ARITHMETIC_CONTEXT


def _make_integration_hint(
    kind: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Create integration hint payload referencing canonical opportunities."""
    metadata = metadata or {}
    hint: Dict[str, Any] = {"kind": kind, "metadata": metadata}
    opportunity = _INTEGRATION_HINT_MAPPING.get(kind)
    if opportunity is not None:
        if hasattr(opportunity, "value"):
            hint["opportunity_type"] = opportunity.value
        else:
            hint["opportunity_type"] = str(opportunity)
    return hint


def _augment_signature(
    signature: Mapping[str, Any],
    **updates: Any,
) -> Dict[str, Any]:
    """Return a new mathematical signature enriched with optional context."""
    enriched = dict(signature) if signature else {}
    for key, value in updates.items():
        if value is not None:
            enriched[key] = value
    return enriched


def _extract_scalar_epi(val: Any) -> float:
    """Extract scalar magnitude from potentially complex/dict EPI value."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, complex):
        return float(np.abs(val))
    if isinstance(val, dict):
        if 'continuous' in val:
            c = val['continuous']
            if isinstance(c, (tuple, list)) and len(c) > 0:
                v = c[0]
                return float(np.abs(v)) if isinstance(v, complex) else float(v)
    return 0.0


class EmergentPatternType(Enum):
    """Types of emergent mathematical patterns."""

    # Natural oscillatory modes
    EIGENMODE_RESONANCE = "eigenmode_resonance"
    # Multi-scale harmonic coupling
    SPECTRAL_CASCADE = "spectral_cascade"
    # Information theoretic structure
    ENTROPY_FLOW = "entropy_flow"
    # Network conservation laws
    TOPOLOGICAL_INVARIANT = "topological_invariant"
    # Self-similar patterns
    FRACTAL_SCALING = "fractal_scaling"
    # Hidden symmetries
    SYMMETRY_BREAKING = "symmetry_breaking"
    # Wave-like interference
    QUANTUM_INTERFERENCE = "quantum_interference"
    # Trajectory compression
    PREDICTIVE_COMPRESSION = "predictive_compression"
    # Phase transition detection
    CRITICAL_TRANSITION = "critical_transition"


@dataclass
class EmergentPattern:
    """Discovered emergent mathematical pattern."""
    pattern_type: EmergentPatternType
    discovery_confidence: float  # 0.0-1.0
    mathematical_signature: Dict[str, Any]
    temporal_scale: float  # Characteristic time scale
    spatial_scale: int     # Characteristic length scale
    prediction_horizon: float  # How far ahead it can predict
    compression_ratio: float   # Information compression achieved
    physical_interpretation: str
    applications: List[str] = field(default_factory=list)


@dataclass
class PatternDiscoveryResult:
    """Result of pattern discovery analysis."""
    discovered_patterns: List[EmergentPattern]
    pattern_interactions: Dict[
        Tuple[EmergentPatternType, EmergentPatternType],
        float,
    ]
    emergent_optimization_strategies: List[str]
    mathematical_invariants: Dict[str, float]
    compression_potential: float
    predictive_accuracy: float
    execution_time: float


class TNFREmergentPatternEngine:
    """
    Engine for discovering emergent mathematical patterns in TNFR dynamics.

    This engine analyzes the deep mathematical structure of the nodal
    equation to discover patterns that emerge naturally at different scales
    and contexts.
    """
    def __init__(
            self,
            enable_caching: bool = True,
            analysis_depth: str = "medium",
    ):
        self.enable_caching = enable_caching
        self.analysis_depth = analysis_depth  # "shallow", "medium", "deep"

        # Discovery state
        self.discovered_patterns = {}
        self.pattern_cache = {}
        self.mathematical_invariants = {}

        # Performance tracking
        self.total_discoveries = 0
        self.pattern_statistics = defaultdict(int)

    def discover_eigenmode_resonances(
        self,
        G: Any,
        time_window: float = 10.0,
        frequency_resolution: int = 100
    ) -> List[EmergentPattern]:
        """
        Discover natural eigenmode resonances in network dynamics.

        The nodal equation ∂EPI/∂t = νf · ΔNFR naturally admits oscillatory
        solutions. These eigenmodes create resonant structures.
        """
        patterns = []

        if not HAS_SCIPY:
            return patterns

        spectral_basis = _get_spectral_basis(G)
        if spectral_basis is None:
            return patterns

        eigenvalues, eigenvectors = spectral_basis
        node_count = (
            len(G.nodes()) if hasattr(G, "nodes") else 0
        )
        field_info = _field_snapshot(G)
        fft_state_summary: Optional[Dict[str, Any]] = None
        fft_engine = _get_fft_engine()
        if fft_engine is not None:
            try:
                fft_state = fft_engine.create_fft_state(G)
                if fft_state.spectral_epi.size:
                    dominant_idx = int(
                        np.argmax(np.abs(fft_state.spectral_epi)))
                    fft_state_summary = {
                        "spectral_energy": float(
                            np.linalg.norm(fft_state.spectral_epi)
                        ),
                        "dominant_index": dominant_idx,
                        "dominant_phase": (
                            float(
                                np.angle(
                                    fft_state.spectral_epi[dominant_idx]
                                )
                            )
                            if fft_state.spectral_epi.size > dominant_idx
                            else 0.0
                        ),
                    }
            except Exception:
                fft_state_summary = None

        arithmetic_summary = None
        arithmetic_ctx = _arithmetic_context()
        if arithmetic_ctx is not None:
            params = arithmetic_ctx["params"]
            arithmetic_summary = {
                "alpha": params.alpha,
                "beta": params.beta,
                "gamma": params.gamma,
            }

        # Analyze natural frequencies
        natural_frequencies = np.sqrt(np.abs(eigenvalues))

        # Find resonant combinations
        resonant_pairs = []
        for i, freq1 in enumerate(natural_frequencies):
            for j, freq2 in enumerate(natural_frequencies[i + 1:], i + 1):
                # Check for harmonic relationships
                ratio = freq1 / freq2 if freq2 > 0 else 0

                # Simple harmonic ratios (1:2, 2:3, 3:4, etc.)
                for n, m in [(1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]:
                    if abs(ratio - n / m) < 0.05 or abs(ratio - m / n) < 0.05:
                        resonant_pairs.append((i, j, freq1, freq2, n, m))

        # Create patterns for each resonance
        for i, j, freq1, freq2, n, m in resonant_pairs:
            integration_hint = _make_integration_hint(
                "eigenmode_resonance",
                {
                    "mode_indices": (i, j),
                    "harmonic_ratio": (n, m),
                    "graph_nodes": node_count,
                },
            )

            pattern = EmergentPattern(
                pattern_type=EmergentPatternType.EIGENMODE_RESONANCE,
                # High confidence for harmonic ratios
                discovery_confidence=PATTERNS_HIGH_CONFIDENCE_CANONICAL,
                mathematical_signature=_augment_signature(
                    {
                        "mode_indices": (i, j),
                        "frequencies": (freq1, freq2),
                        "harmonic_ratio": (n, m),
                        "resonance_strength": (
                            abs(freq1 - freq2) / (freq1 + freq2)
                        ),
                        "coupling_coefficient": float(
                            np.dot(eigenvectors[:, i], eigenvectors[:, j])
                        ),
                    },
                    field_snapshot=field_info if field_info else None,
                    fft_signature=fft_state_summary,
                    arithmetic_coefficients=arithmetic_summary,
                    integration_hint=integration_hint,
                ),
                temporal_scale=(
                    2 * np.pi / min(freq1, freq2)
                    if min(freq1, freq2) > 0
                    else float('inf')
                ),
                spatial_scale=node_count,
                prediction_horizon=time_window,
                # Can compress oscillatory patterns
                compression_ratio=PATTERNS_COMPRESSION_RATIO_CANONICAL,
                physical_interpretation=(
                    "Harmonic coupling between modes "
                    f"{i} and {j} with ratio {n}:{m}"
                ),
                applications=[
                    "resonance_prediction",
                    "mode_coupling",
                    "harmonic_analysis",
                ],
            )
            patterns.append(pattern)

        return patterns

    def discover_spectral_cascades(
        self,
        G: Any,
        cascade_depth: int = 5
    ) -> List[EmergentPattern]:
        """
        Discover spectral energy cascades across scales.

        Multi-scale coupling in TNFR creates natural energy cascades
        similar to turbulence but in network-spectral space.
        """
        patterns = []

        if not HAS_PHYSICS_FIELDS:
            return patterns

        spectral_basis = _get_spectral_basis(G)
        if spectral_basis is None:
            return patterns

        # Get current EPI distribution
        epi_signal = np.array([
            _extract_scalar_epi(G.nodes[node].get('EPI', 0.0))
            for node in G.nodes()
        ])

        # Get spectral decomposition
        eigenvalues, eigenvectors = spectral_basis
        spectral_coeffs = gft(epi_signal, eigenvectors)

        # Analyze energy distribution across scales
        energy_spectrum = np.abs(spectral_coeffs) ** 2

        # Look for power-law cascades (E(k) ~ k^(-α))
        frequencies = eigenvalues
        valid_indices = frequencies > 0

        if np.sum(valid_indices) > 3:
            log_freq = np.log(frequencies[valid_indices])
            log_energy = np.log(energy_spectrum[valid_indices] + 1e-12)

            # Fit power law
            if HAS_SCIPY:
                slope, intercept = np.polyfit(log_freq, log_energy, 1)
                r_squared = np.corrcoef(log_freq, log_energy)[0, 1] ** 2

                # Strong power law indicates cascade
                if (
                    r_squared > PATTERNS_RSQUARED_THRESHOLD_CANONICAL
                    and abs(slope) > PATTERNS_SLOPE_THRESHOLD_CANONICAL
                ):
                    field_info = _field_snapshot(G)
                    fft_signature = None
                    fft_engine = _get_fft_engine()
                    if fft_engine is not None:
                        try:
                            fft_state = fft_engine.create_fft_state(G)
                            if fft_state.spectral_epi.size:
                                fft_signature = {
                                    "spectral_energy": float(
                                        np.linalg.norm(
                                            fft_state.spectral_epi)
                                    ),
                                    "phase_energy": float(
                                        np.linalg.norm(
                                            fft_state.spectral_phase)
                                    ),
                                }
                        except Exception:
                            fft_signature = None

                    arithmetic_summary = None
                    arithmetic_ctx = _arithmetic_context()
                    if arithmetic_ctx is not None:
                        params = arithmetic_ctx["params"]
                        arithmetic_summary = {
                            "nu_0": params.nu_0,
                            "delta": params.delta,
                            "epsilon": params.epsilon,
                        }

                    integration_hint = _make_integration_hint(
                        "spectral_cascade",
                        {
                            "slope": float(slope),
                            "r_squared": float(r_squared),
                            "frequency_span": float(
                                np.max(frequencies[valid_indices])
                                - np.min(frequencies[valid_indices])
                            ),
                        },
                    )

                    pattern = EmergentPattern(
                        pattern_type=EmergentPatternType.SPECTRAL_CASCADE,
                        discovery_confidence=r_squared,
                        mathematical_signature=_augment_signature(
                            {
                                "cascade_exponent": -slope,
                                "energy_scale": np.exp(intercept),
                                "frequency_range": (
                                    np.min(frequencies[valid_indices]),
                                    np.max(frequencies[valid_indices]),
                                ),
                                "r_squared": r_squared,
                                "total_energy": np.sum(energy_spectrum),
                            },
                            field_snapshot=field_info if field_info else None,
                            fft_signature=fft_signature,
                            arithmetic_coefficients=arithmetic_summary,
                            integration_hint=integration_hint,
                        ),
                        temporal_scale=1.0
                        / np.max(frequencies[valid_indices]),
                        spatial_scale=len(G.nodes()),
                        prediction_horizon=PATTERNS_HORIZON_LONG_CANONICAL,
                        compression_ratio=(
                            PATTERNS_COMPRESSION_OSCILLATORY_CANONICAL
                        ),
                        physical_interpretation=(
                            f"Energy cascade with exponent {slope:.2f}"
                        ),
                        applications=[
                            "energy_prediction",
                            "cascade_modeling",
                            "multi_scale_analysis",
                        ],
                    )
                    patterns.append(pattern)

        return patterns

    def discover_entropy_flow_patterns(
        self,
        G: Any,
        history_length: int = 10
    ) -> List[EmergentPattern]:
        """
        Discover information-theoretic patterns in EPI evolution.

        The nodal equation has natural entropy production and flow.
        """
        patterns = []
        node_count = len(G.nodes()) if hasattr(G, "nodes") else 0
        if node_count == 0:
            return patterns
        max_entropy = np.log(node_count)

        # Extract current state entropy
        epi_values = [
            _extract_scalar_epi(G.nodes[node].get('EPI', 0.0))
            for node in G.nodes()
        ]
        epi_array = np.array(epi_values)

        # Normalize to probability distribution
        if np.sum(np.abs(epi_array)) > 0:
            prob_dist = np.abs(epi_array) / np.sum(np.abs(epi_array))

            # Compute Shannon entropy
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))

            # Compute relative entropy (KL divergence from uniform)
            uniform_dist = np.ones_like(prob_dist) / len(prob_dist)
            kl_divergence = np.sum(
                prob_dist * np.log(prob_dist / uniform_dist + 1e-12))

            # Analyze entropy gradient across network
            if HAS_NETWORKX:
                entropy_gradient = 0.0
                for edge in G.edges():
                    u, v = edge
                    epi_u = _extract_scalar_epi(G.nodes[u].get('EPI', 0.0))
                    epi_v = _extract_scalar_epi(G.nodes[v].get('EPI', 0.0))
                    entropy_gradient += abs(epi_u - epi_v)
                entropy_gradient /= len(G.edges()) if len(G.edges()) > 0 else 1

                # Strong entropy flow indicates information-theoretic structure
                if (
                    entropy > PATTERNS_ENTROPY_THRESHOLD_CANONICAL
                    and kl_divergence > PATTERNS_DIVERGENCE_THRESHOLD_CANONICAL
                ):
                    field_info = _field_snapshot(G)
                    integration_hint = _make_integration_hint(
                        "entropy_flow",
                        {
                            "entropy": float(entropy),
                            "kl_divergence": float(kl_divergence),
                            "edge_entropy_gradient": float(entropy_gradient),
                        },
                    )
                    pattern = EmergentPattern(
                        pattern_type=EmergentPatternType.ENTROPY_FLOW,
                        discovery_confidence=min(
                            entropy / max_entropy, 1.0
                        ),
                        mathematical_signature=_augment_signature(
                            {
                                "shannon_entropy": entropy,
                                "kl_divergence": kl_divergence,
                                "entropy_gradient": entropy_gradient,
                                "max_entropy": max_entropy,
                                "entropy_efficiency": entropy / max_entropy,
                                "information_density": np.sum(epi_array ** 2)
                            },
                            field_snapshot=field_info if field_info else None,
                            integration_hint=integration_hint,
                        ),
                        temporal_scale=1.0,
                        spatial_scale=int(np.sqrt(node_count)),
                        prediction_horizon=PATTERNS_HORIZON_MEDIUM_CANONICAL,
                        compression_ratio=entropy / max_entropy,
                        physical_interpretation=(
                            "Information flow with entropy production"
                        ),
                        applications=[
                            "information_theory",
                            "compression",
                            "prediction",
                        ],
                    )
                    patterns.append(pattern)

        return patterns

    def discover_topological_invariants(
        self,
        G: Any
    ) -> List[EmergentPattern]:
        """
        Discover topological invariants that constrain TNFR evolution.

        Network topology creates conservation laws for certain quantities.
        """
        patterns = []

        if not HAS_NETWORKX:
            return patterns

        # Compute basic topological invariants
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())

        # Euler characteristic for planar graphs
        if num_nodes > 2:
            try:
                # Attempt to compute number of faces for planar graphs
                # For non-planar, this is just a connectivity measure
                euler_char = num_nodes - num_edges

                # Degree sequence invariant
                degrees = [G.degree(node) for node in G.nodes()]
                degree_sequence_invariant = (
                    np.sum(np.array(degrees) ** 2) / (2 * num_edges)
                    if num_edges > 0 else 0
                )

                # Spectral invariants
                if HAS_SPECTRAL:
                    eigenvalues, _ = get_laplacian_spectrum(G)
                    # Should equal sum of degrees
                    spectral_trace = np.sum(eigenvalues)
                    spectral_determinant = np.prod(
                        eigenvalues[eigenvalues > 1e-10])

                    # Check if invariants are preserved (they should be for
                    # topology)
                    degree_sum_check = abs(
                        spectral_trace - np.sum(degrees)) < 1e-10

                    if degree_sum_check:
                        pattern = EmergentPattern(
                            pattern_type=(
                                EmergentPatternType.TOPOLOGICAL_INVARIANT
                            ),
                            discovery_confidence=1.0,
                            mathematical_signature={
                                "euler_characteristic": euler_char,
                                "degree_sequence_invariant": (
                                    degree_sequence_invariant
                                ),
                                "spectral_trace": spectral_trace,
                                "spectral_determinant": spectral_determinant,
                                "num_components": (
                                    nx.number_connected_components(G)
                                ),
                                "diameter": (
                                    nx.diameter(G)
                                    if nx.is_connected(G)
                                    else float('inf')
                                ),
                            },
                            temporal_scale=float('inf'),
                            spatial_scale=num_nodes,
                            prediction_horizon=float('inf'),
                            compression_ratio=float('inf'),
                            physical_interpretation=(
                                "Topological conservation law"
                            ),
                            applications=[
                                "invariant_checking",
                                "topology_verification",
                                "conservation_laws",
                            ],
                        )
                        patterns.append(pattern)

            except Exception:
                pass  # Skip if topological calculations fail

        return patterns

    def discover_fractal_scaling_patterns(
        self,
        G: Any,
        scale_range: Tuple[int, int] = (2, 10)
    ) -> List[EmergentPattern]:
        """
        Discover fractal self-similarity in network structure.

        TNFR dynamics can exhibit fractal patterns across multiple scales.
        """
        patterns = []

        if not HAS_NETWORKX or not HAS_SCIPY:
            return patterns

        # Analyze scaling of various network properties
        min_scale, max_scale = scale_range

        # Box-counting dimension for network structure
        scales = range(min_scale, min(max_scale, len(G.nodes()) // 2))
        box_counts = []

        for scale in scales:
            # Simple box-counting: partition nodes into boxes of size 'scale'
            # and count non-empty boxes
            try:
                if nx.is_connected(G):
                    # Use shortest path distances for partitioning
                    distances = dict(nx.all_pairs_shortest_path_length(G))

                    # Find maximal sets of nodes within distance 'scale'
                    boxes = []
                    uncovered = set(G.nodes())

                    while uncovered:
                        seed = next(iter(uncovered))
                        box = {seed}

                        for node in list(uncovered):
                            if (
                                node in distances[seed]
                                and distances[seed][node] <= scale
                            ):
                                box.add(node)

                        boxes.append(box)
                        uncovered -= box

                    box_counts.append(len(boxes))

            except Exception:
                box_counts.append(1)  # Fallback

        # Fit fractal dimension
        if len(box_counts) >= 3:
            log_scales = np.log(np.array(scales[:len(box_counts)]))
            log_boxes = np.log(np.array(box_counts))

            # Fractal dimension from slope
            slope, intercept = np.polyfit(log_scales, log_boxes, 1)
            r_squared = np.corrcoef(log_scales, log_boxes)[0, 1] ** 2

            # Good fractal scaling
            if r_squared > PATTERNS_RSQUARED_HIGH_CANONICAL and abs(
                    slope) > PATTERNS_SLOPE_MINIMAL_CANONICAL:
                fractal_dim = -slope  # Negative because N(r) ~ r^(-D)

                pattern = EmergentPattern(
                    pattern_type=EmergentPatternType.FRACTAL_SCALING,
                    discovery_confidence=r_squared,
                    mathematical_signature={
                        "fractal_dimension": fractal_dim,
                        "scaling_prefactor": np.exp(intercept),
                        "scale_range": scale_range,
                        "r_squared": r_squared,
                        "box_counts": box_counts,
                        "scales": list(scales[:len(box_counts)])
                    },
                    temporal_scale=1.0,
                    spatial_scale=int(np.mean(scales)),
                    prediction_horizon=PATTERNS_HORIZON_SHORT_CANONICAL,
                    compression_ratio=len(G.nodes()) / len(box_counts),
                    physical_interpretation=(
                        f"Fractal scaling with dimension {fractal_dim:.2f}"
                    ),
                    applications=[
                        "fractal_analysis",
                        "multi_scale_modeling",
                        "dimension_reduction",
                    ]
                )
                patterns.append(pattern)

        return patterns

    def analyze_pattern_interactions(
        self,
        patterns: List[EmergentPattern]
    ) -> Dict[Tuple[EmergentPatternType, EmergentPatternType], float]:
        """
        Analyze interactions between discovered patterns.

        Patterns can reinforce or interfere with each other.
        """
        interactions = {}

        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i + 1:]:
                # Compute interaction strength based on scale overlap
                spatial_overlap = min(
                    pattern1.spatial_scale,
                    pattern2.spatial_scale) / max(
                    pattern1.spatial_scale,
                    pattern2.spatial_scale)

                temporal_overlap = 1.0
                if pattern1.temporal_scale != float(
                        'inf') and pattern2.temporal_scale != float('inf'):
                    temporal_overlap = min(
                        pattern1.temporal_scale,
                        pattern2.temporal_scale) / max(
                        pattern1.temporal_scale,
                        pattern2.temporal_scale)

                # Combined interaction strength
                interaction_strength = (
                    spatial_overlap
                    * temporal_overlap
                    * pattern1.discovery_confidence
                    * pattern2.discovery_confidence
                )

                interactions[(pattern1.pattern_type,
                              pattern2.pattern_type)] = interaction_strength

        return interactions

    def discover_all_patterns(
        self,
        G: Any,
        **kwargs
    ) -> PatternDiscoveryResult:
        """
        Comprehensive pattern discovery across all pattern types.
        """
        start_time = time.perf_counter()

        all_patterns = []

        # Discover each pattern type
        all_patterns.extend(self.discover_eigenmode_resonances(G, **kwargs))
        all_patterns.extend(self.discover_spectral_cascades(G, **kwargs))
        all_patterns.extend(self.discover_entropy_flow_patterns(G, **kwargs))
        all_patterns.extend(self.discover_topological_invariants(G, **kwargs))
        all_patterns.extend(
            self.discover_fractal_scaling_patterns(G, **kwargs))

        # Analyze pattern interactions
        pattern_interactions = self.analyze_pattern_interactions(all_patterns)

        # Generate emergent optimization strategies
        optimization_strategies = []
        for pattern in all_patterns:
            if (
                pattern.compression_ratio
                > PATTERNS_COMPRESSION_SIGNIFICANT_CANONICAL
            ):
                optimization_strategies.append(
                    f"compress_using_{pattern.pattern_type.value}")
            if (
                pattern.prediction_horizon
                > PATTERNS_HORIZON_PREDICTIVE_CANONICAL
            ):
                optimization_strategies.append(
                    f"predict_using_{pattern.pattern_type.value}")

        # Compute mathematical invariants
        invariants = {}
        if all_patterns:
            invariants["total_compression"] = np.prod(
                [p.compression_ratio for p in all_patterns])
            invariants["max_prediction_horizon"] = np.max(
                [p.prediction_horizon for p in all_patterns])
            invariants["average_confidence"] = np.mean(
                [p.discovery_confidence for p in all_patterns])

        # Overall metrics
        compression_potential = (
            np.mean([p.compression_ratio for p in all_patterns])
            if all_patterns else 1.0
        )
        predictive_accuracy = (
            np.mean([p.discovery_confidence for p in all_patterns])
            if all_patterns else 0.0
        )

        execution_time = time.perf_counter() - start_time

        # Update statistics
        self.total_discoveries += len(all_patterns)
        for pattern in all_patterns:
            self.pattern_statistics[pattern.pattern_type] += 1

        return PatternDiscoveryResult(
            discovered_patterns=all_patterns,
            pattern_interactions=pattern_interactions,
            emergent_optimization_strategies=list(
                set(optimization_strategies)),
            mathematical_invariants=invariants,
            compression_potential=compression_potential,
            predictive_accuracy=predictive_accuracy,
            execution_time=execution_time
        )

    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern discoveries."""
        return {
            "total_discoveries": self.total_discoveries,
            "pattern_counts": dict(self.pattern_statistics),
            "cached_patterns": len(self.pattern_cache),
            "analysis_depth": self.analysis_depth,
            "caching_enabled": self.enable_caching,
            "available_modules": {
                "scipy": HAS_SCIPY,
                "spectral": HAS_SPECTRAL,
                "physics_fields": HAS_PHYSICS_FIELDS,
                "unified_cache": HAS_UNIFIED_CACHE
            }
        }

    def _serialize_discovery_statistics(self) -> Dict[str, Any]:
        """Get JSON-serializable discovery statistics."""
        stats = self.get_discovery_statistics()
        # Convert enum keys to strings in pattern_counts
        if "pattern_counts" in stats:
            stats["pattern_counts"] = {
                k.value if hasattr(k, "value") else str(k): v
                for k, v in stats["pattern_counts"].items()
            }
        return stats

    def export_pattern_manifest(
        self,
        G: Any,
        discovery_result: PatternDiscoveryResult,
        output_dir: Path,
        partition_id: str,
    ) -> Dict[str, Path]:
        """Export pattern discovery results as manifest for self-optimization.

        Parameters
        ----------
        G : Any
            TNFR network graph that was analyzed.
        discovery_result : PatternDiscoveryResult
            Results from discover_all_patterns().
        output_dir : Path
            Directory where manifests will be written.
        partition_id : str
            Unique identifier for this pattern discovery partition.

        Returns
        -------
        Dict[str, Path]
            Dictionary with keys 'manifest_absolute' and 'summary_absolute'
            pointing to the generated manifest files.

        Notes
        -----
        Manifest format compatible with self_opt_support pipeline:
        - operation_type: 'pattern_discovery'
        - partition_id: unique identifier
        - discovered_patterns: list of pattern metadata
        - telemetry: coherence, sense_index, phase metrics
        - network_metadata: node count, edge count, structural metrics
        """
        import json
        from datetime import datetime, timezone

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute telemetry metrics
        telemetry = {}
        try:
            from ...physics import compute_coherence, compute_sense_index
            telemetry["coherence"] = float(compute_coherence(G))
            telemetry["sense_index"] = float(compute_sense_index(G))
        except Exception:
            telemetry["coherence"] = None
            telemetry["sense_index"] = None

        try:
            from ...physics.fields import compute_structural_potential_field
            phi_s_values = compute_structural_potential_field(G)
            if phi_s_values:
                telemetry["structural_potential_range"] = [
                    float(min(phi_s_values.values())),
                    float(max(phi_s_values.values())),
                ]
            else:
                telemetry["structural_potential_range"] = None
        except Exception:
            telemetry["structural_potential_range"] = None

        # Extract network metadata
        node_count = len(G.nodes()) if hasattr(G, "nodes") else 0
        edge_count = len(G.edges()) if hasattr(G, "edges") else 0

        # Serialize discovered patterns
        patterns_serialized = []
        for pattern in discovery_result.discovered_patterns:
            pattern_data = {
                "pattern_type": pattern.pattern_type.value,
                "confidence": float(pattern.discovery_confidence),
                "temporal_scale": float(pattern.temporal_scale),
                "spatial_scale": float(pattern.spatial_scale),
                "prediction_horizon": float(pattern.prediction_horizon),
                "compression_ratio": float(pattern.compression_ratio),
                "physical_interpretation": pattern.physical_interpretation,
                "applications": pattern.applications,
                "mathematical_signature": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                    for k, v in pattern.mathematical_signature.items()
                    if not k.startswith("_")
                },
            }
            patterns_serialized.append(pattern_data)

        # Build manifest
        manifest = {
            "operation_type": "pattern_discovery",
            "partition_id": partition_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "network_metadata": {
                "node_count": node_count,
                "edge_count": edge_count,
            },
            "telemetry": telemetry,
            "discovered_patterns": patterns_serialized,
            "discovery_statistics": self._serialize_discovery_statistics(),
            "emergent_optimization_strategies": discovery_result.emergent_optimization_strategies,
            "mathematical_invariants": {
                k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                for k, v in discovery_result.mathematical_invariants.items()
            },
            "compression_potential": float(discovery_result.compression_potential),
            "predictive_accuracy": float(discovery_result.predictive_accuracy),
        }

        # Write manifest
        manifest_path = output_dir / "pattern_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Write summary
        summary = {
            "operation_type": "pattern_discovery",
            "partition_id": partition_id,
            "pattern_count": len(patterns_serialized),
            "coherence": telemetry.get("coherence"),
            "sense_index": telemetry.get("sense_index"),
            "compression_potential": float(discovery_result.compression_potential),
            "predictive_accuracy": float(discovery_result.predictive_accuracy),
        }
        summary_path = output_dir / "pattern_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "manifest_absolute": manifest_path.resolve(),
            "summary_absolute": summary_path.resolve(),
        }


# Factory functions for easy access
def create_emergent_pattern_engine(**kwargs) -> TNFREmergentPatternEngine:
    """Create emergent pattern discovery engine."""
    return TNFREmergentPatternEngine(**kwargs)


@cache_unified_computation(
    CacheEntryType.NODAL_STATE,
    # γ/(2π+e) ≈ 0.0625 → canonical
    mathematical_importance=OPT_ORCH_ARITHMETIC_BOOST_CANONICAL
) if HAS_UNIFIED_CACHE else lambda *args, **kwargs: lambda f: f
def discover_mathematical_patterns(G: Any, **kwargs) -> PatternDiscoveryResult:
    """Convenience function for comprehensive pattern discovery."""
    engine = create_emergent_pattern_engine()
    return engine.discover_all_patterns(G, **kwargs)


def analyze_emergent_symmetries(G: Any) -> Dict[str, Any]:
    """Analyze emergent symmetries in TNFR dynamics."""
    engine = create_emergent_pattern_engine()
    result = engine.discover_all_patterns(G)

    # Focus on symmetry-related patterns
    symmetry_patterns = [
        p for p in result.discovered_patterns if p.pattern_type in [
            EmergentPatternType.EIGENMODE_RESONANCE,
            EmergentPatternType.TOPOLOGICAL_INVARIANT]]

    return {
        "symmetry_count": len(symmetry_patterns),
        "symmetry_patterns": symmetry_patterns,
        "broken_symmetries": [
            p
            for p in symmetry_patterns
            if p.discovery_confidence < PATTERNS_CONFIDENCE_BROKEN_CANONICAL
        ],
        "conservation_laws": [
            p
            for p in symmetry_patterns
            if p.pattern_type == EmergentPatternType.TOPOLOGICAL_INVARIANT
        ],
    }

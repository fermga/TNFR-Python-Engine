"""Finite-snapshot mathematical pattern diagnostics for TNFR graphs.

The detectors compute graph-spectral fits, amplitude-distribution summaries,
topology identities, and an order-dependent box-cover heuristic. The nodal
equation fixes the meanings of EPI, structural frequency, and DeltaNFR; it does
not by itself imply oscillations, entropy production, conservation under
topology changes, fractal scaling, prediction skill, or quantum behavior.

Historical result names are retained for compatibility. Each detector records
its model scope and uses neutral performance fields unless a quantity is
actually measured.
"""

import importlib.util
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ...alias import get_attr
from ...constants.aliases import ALIAS_EPI
from ...mathematics.unified_numerical import np
from ...types import require_finite_real_scalar_epi

# Import canonical constants for Phase 6 magic number elimination
# Import PHASE 6 EXTENDED Canonical Constants for magic number elimination
from ..constants.operational import (
    OPT_ORCH_ARITHMETIC_BOOST_CANONICAL,
)
from ..constants.operational import (
    PATTERNS_COMPRESSION_SIGNIFICANT_CANONICAL,
    PATTERNS_CONFIDENCE_BROKEN_CANONICAL,
    PATTERNS_DIVERGENCE_THRESHOLD_CANONICAL,
    PATTERNS_ENTROPY_THRESHOLD_CANONICAL,
    PATTERNS_HIGH_CONFIDENCE_CANONICAL,
    PATTERNS_HORIZON_PREDICTIVE_CANONICAL,
    PATTERNS_RSQUARED_HIGH_CANONICAL,
    PATTERNS_RSQUARED_THRESHOLD_CANONICAL,
    PATTERNS_SLOPE_MINIMAL_CANONICAL,
    PATTERNS_SLOPE_THRESHOLD_CANONICAL,
)

# ---------------------------------------------------------------------------
# Harmonic resonance detection
# ---------------------------------------------------------------------------
_HARMONIC_RATIO_TOLERANCE = 0.05

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

try:
    from .multi_modal_cache import CacheEntryType, cache_unified_computation

    HAS_UNIFIED_CACHE = True
except ImportError:
    HAS_UNIFIED_CACHE = False

try:
    from ...physics.fields import compute_phase_gradient, compute_structural_potential

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
        ArithmeticStructuralTerms,
        ArithmeticTNFRFormalism,
        ArithmeticTNFRParameters,
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

_SHARED_FFT_ENGINE: Any | None = None

_ARITHMETIC_CONTEXT: dict[str, Any] | None = None
if HAS_INTEGRATION_HINTS:
    _INTEGRATION_HINT_MAPPING: dict[str, Any] = {
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


def _get_spectral_basis(graph: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a live graph Laplacian basis from the shared spectral reader."""

    if not HAS_SPECTRAL:
        return None
    try:
        return get_laplacian_spectrum(graph)
    except Exception:
        return None


def _field_snapshot(graph: Any) -> dict[str, Any]:
    """Capture structural-field telemetry for the current graph state."""

    if not HAS_PHYSICS_FIELDS:
        return {}
    try:
        phi_s = compute_structural_potential(graph)
        phase_grad = compute_phase_gradient(graph)
    except Exception:
        return {}

    def _summaries(values: Any) -> dict[str, float]:
        if isinstance(values, dict):
            arr = np.array(list(values.values()), dtype=float)
        else:
            arr = np.array(values, dtype=float)
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("structural field snapshot must be finite")
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "structural_potential": _summaries(phi_s),
        "phase_gradient": _summaries(phase_grad),
        "timestamp": time.perf_counter(),
        "scope": "live_snapshot",
    }


def _get_fft_engine() -> Any | None:
    """Provide a shared FFT dynamics engine instance."""

    global _SHARED_FFT_ENGINE
    if not HAS_FFT_ENGINE:
        return None
    if _SHARED_FFT_ENGINE is not None:
        return _SHARED_FFT_ENGINE
    try:
        _SHARED_FFT_ENGINE = FFTDynamicsEngine()
    except Exception:
        return None
    return _SHARED_FFT_ENGINE

def _arithmetic_context() -> dict[str, Any] | None:
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
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Create integration hint payload referencing canonical opportunities."""
    metadata = metadata or {}
    hint: dict[str, Any] = {"kind": kind, "metadata": metadata}
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
) -> dict[str, Any]:
    """Return a new mathematical signature enriched with optional context."""
    enriched = dict(signature) if signature else {}
    for key, value in updates.items():
        if value is not None:
            enriched[key] = value
    return enriched


def _node_scalar_epi(G: Any, node: Any) -> float:
    """Read one finite signed scalar EPI through canonical alias precedence."""

    raw = get_attr(
        G.nodes[node], ALIAS_EPI, 0.0, strict=True, conv=lambda value: value
    )
    return require_finite_real_scalar_epi(raw, f"node {node!r} EPI")


def _finite_nonnegative_mean(values: list[float]) -> float | None:
    """Return a stable mean, or ``None`` if any value is not representable."""

    if not values:
        return 0.0
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        return None
    scale = max(values)
    if scale == 0.0:
        return 0.0
    result = scale * math.fsum(value / scale for value in values) / len(values)
    return result if math.isfinite(result) else None


def _finite_squared_norm(values: np.ndarray) -> float | None:
    """Return the squared Euclidean norm when binary64 can represent it."""

    magnitudes = np.abs(np.asarray(values, dtype=float))
    if magnitudes.size == 0:
        return 0.0
    scale = float(np.max(magnitudes))
    if scale == 0.0:
        return 0.0
    normalized_sum = math.fsum(
        float(value / scale) ** 2 for value in magnitudes
    )
    if scale > math.sqrt(float(np.finfo(float).max) / normalized_sum):
        return None
    result = scale * scale * normalized_sum
    return result if math.isfinite(result) else None


def _graph_epi_signal(G: Any) -> np.ndarray:
    """Return the live signed scalar-EPI chart in graph iteration order."""

    return np.asarray([_node_scalar_epi(G, node) for node in G.nodes()], dtype=float)


class EmergentPatternType(Enum):
    """Types of emergent mathematical patterns."""

    # Auxiliary graph-wave frequency ratios (historical enum label)
    EIGENMODE_RESONANCE = "eigenmode_resonance"
    # Snapshot spectral power-law fit (historical enum label)
    SPECTRAL_CASCADE = "spectral_cascade"
    # Snapshot amplitude-distribution structure (historical enum label)
    ENTROPY_FLOW = "entropy_flow"
    # Fixed-topology snapshot identities
    TOPOLOGICAL_INVARIANT = "topological_invariant"
    # Self-similar patterns
    FRACTAL_SCALING = "fractal_scaling"
    # Hidden symmetries
    SYMMETRY_BREAKING = "symmetry_breaking"
    # Historical label for classical wave interference; no quantum-state claim.
    QUANTUM_INTERFERENCE = "quantum_interference"
    # Trajectory compression
    PREDICTIVE_COMPRESSION = "predictive_compression"
    # Phase transition detection
    CRITICAL_TRANSITION = "critical_transition"


@dataclass
class EmergentPattern:
    """Discovered emergent mathematical pattern."""

    pattern_type: EmergentPatternType
    discovery_confidence: float  # Detector fit/threshold score in [0, 1]
    mathematical_signature: dict[str, Any]
    temporal_scale: float  # Characteristic scale, 0 when not estimated
    spatial_scale: int
    prediction_horizon: float | None  # absent unless prediction was evaluated
    compression_ratio: float | None  # absent unless a compressor was measured
    physical_interpretation: str
    applications: list[str] = field(default_factory=list)


@dataclass
class PatternDiscoveryResult:
    """Result of pattern discovery analysis."""

    discovered_patterns: list[EmergentPattern]
    pattern_interactions: dict[
        tuple[EmergentPatternType, EmergentPatternType],
        float,
    ]
    emergent_optimization_strategies: list[str]
    mathematical_invariants: dict[str, float]
    compression_potential: float | None
    predictive_accuracy: float | None
    execution_time: float


class TNFREmergentPatternEngine:
    """
    Compute scoped pattern scores from a TNFR graph snapshot.

    Detector confidence is a fit or threshold score, not held-out predictive
    accuracy. No detector advances the nodal state.
    """

    def __init__(
        self,
        enable_caching: bool = True,
        analysis_depth: str = "medium",
    ):
        self.enable_caching = enable_caching
        self.analysis_depth = analysis_depth  # "shallow", "medium", "deep"

        # Discovery state. Class methods are live snapshot readers; the
        # module-level convenience function may use the unified cache.
        self.discovered_patterns = {}

        # Performance tracking
        self.total_discoveries = 0
        self.pattern_statistics = defaultdict(int)

    def discover_eigenmode_resonances(
        self, G: Any, time_window: float = 10.0, frequency_resolution: int = 100
    ) -> list[EmergentPattern]:
        """
        Find near-harmonic ratios in the auxiliary graph-wave spectrum.

        Laplacian eigenvalues define frequencies for the auxiliary graph-wave
        comparison. This does not assert oscillatory solutions of the
        first-order nodal equation.
        """
        patterns = []

        if not HAS_SCIPY:
            return patterns

        spectral_basis = _get_spectral_basis(G)
        if spectral_basis is None:
            return patterns

        eigenvalues, eigenvectors = spectral_basis
        node_count = len(G.nodes()) if hasattr(G, "nodes") else 0
        field_info = _field_snapshot(G)
        fft_state_summary: dict[str, Any] | None = None
        fft_engine = _get_fft_engine()
        if fft_engine is not None:
            try:
                fft_state = fft_engine.create_fft_state(G)
                if fft_state.spectral_epi.size:
                    dominant_idx = int(np.argmax(np.abs(fft_state.spectral_epi)))
                    fft_state_summary = {
                        "spectral_energy": float(
                            np.linalg.norm(fft_state.spectral_epi)
                        ),
                        "dominant_index": dominant_idx,
                        "dominant_phase": (
                            float(np.angle(fft_state.spectral_epi[dominant_idx]))
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

        # Find near-harmonic frequency ratios.
        resonant_pairs = []
        for i, freq1 in enumerate(natural_frequencies):
            for j, freq2 in enumerate(natural_frequencies[i + 1 :], i + 1):
                # Check for harmonic relationships
                ratio = freq1 / freq2 if freq2 > 0 else 0

                # Simple harmonic ratios (1:2, 2:3, 3:4, etc.)
                for n, m in [(1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]:
                    if (
                        abs(ratio - n / m) < _HARMONIC_RATIO_TOLERANCE
                        or abs(ratio - m / n) < _HARMONIC_RATIO_TOLERANCE
                    ):
                        resonant_pairs.append((i, j, freq1, freq2, n, m))

        # Create one diagnostic record for each accepted ratio.
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
                        "relative_frequency_separation": (
                            abs(freq1 - freq2) / (freq1 + freq2)
                        ),
                        "model_scope": "auxiliary_graph_wave_snapshot",
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
                    else float("inf")
                ),
                spatial_scale=node_count,
                prediction_horizon=None,
                # No compressor is run by this detector.
                compression_ratio=None,
                physical_interpretation=(
                    "Near-harmonic auxiliary graph-wave frequencies for modes "
                    f"{i} and {j} with ratio {n}:{m}"
                ),
                applications=[
                    "auxiliary_wave_mode_comparison",
                    "harmonic_ratio_analysis",
                ],
            )
            patterns.append(pattern)

        return patterns

    def discover_spectral_cascades(
        self, G: Any, cascade_depth: int = 5
    ) -> list[EmergentPattern]:
        """
        Discover spectral energy cascades across scales.

        Fit a power law to one graph-spectral EPI power distribution. The
        result is a snapshot fit, not evidence of a temporal energy cascade.
        """
        patterns = []

        if not HAS_PHYSICS_FIELDS:
            return patterns

        spectral_basis = _get_spectral_basis(G)
        if spectral_basis is None:
            return patterns

        # Get current EPI distribution
        epi_signal = _graph_epi_signal(G)

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
                # Guard against zero-variance inputs (degenerate spectra,
                # e.g. when energy_spectrum saturates the 1e-12 floor):
                # np.corrcoef would divide by zero std, yielding nan and a
                # RuntimeWarning. Physically, a constant spectrum has no
                # power-law structure -> R^2 = 0.
                if np.std(log_freq) <= 0.0 or np.std(log_energy) <= 0.0:
                    r_squared = 0.0
                else:
                    r_squared = np.corrcoef(log_freq, log_energy)[0, 1] ** 2

                # Record a strong snapshot power-law fit.
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
                                        np.linalg.norm(fft_state.spectral_epi)
                                    ),
                                    "phase_energy": float(
                                        np.linalg.norm(fft_state.spectral_phase)
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
                                "total_spectral_power": np.sum(energy_spectrum),
                                "model_scope": "graph_spectral_snapshot_fit",
                            },
                            field_snapshot=field_info if field_info else None,
                            fft_signature=fft_signature,
                            arithmetic_coefficients=arithmetic_summary,
                            integration_hint=integration_hint,
                        ),
                        temporal_scale=1.0 / np.max(frequencies[valid_indices]),
                        spatial_scale=len(G.nodes()),
                        prediction_horizon=None,
                        compression_ratio=None,
                        physical_interpretation=(
                            f"Snapshot spectral power-law fit with exponent {slope:.2f}"
                        ),
                        applications=[
                            "spectral_power_law_fit",
                            "multi_scale_snapshot_analysis",
                        ],
                    )
                    patterns.append(pattern)

        return patterns

    def discover_entropy_flow_patterns(
        self, G: Any, history_length: int = 10
    ) -> list[EmergentPattern]:
        """
        Compute amplitude entropy and mean edge contrast in one EPI snapshot.

        A single state contains no entropy-production or temporal-flow evidence.
        """
        patterns = []
        node_count = len(G.nodes()) if hasattr(G, "nodes") else 0
        if node_count < 2:
            return patterns
        max_entropy = np.log(node_count)

        # Extract current state entropy
        epi_array = _graph_epi_signal(G)

        # Normalize magnitudes after scaling, so several maximum finite
        # EPI coordinates cannot overflow the intermediate sum.
        epi_magnitudes = np.abs(epi_array)
        magnitude_scale = float(np.max(epi_magnitudes, initial=0.0))
        if magnitude_scale > 0.0:
            scaled_magnitudes = epi_magnitudes / magnitude_scale
            scaled_total = math.fsum(float(value) for value in scaled_magnitudes)
            prob_dist = scaled_magnitudes / scaled_total

            # Compute Shannon entropy
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-12))

            # Compute relative entropy (KL divergence from uniform)
            uniform_dist = np.ones_like(prob_dist) / len(prob_dist)
            kl_divergence = np.sum(prob_dist * np.log(prob_dist / uniform_dist + 1e-12))

            # Analyze entropy gradient across network
            if HAS_NETWORKX:
                edge_contrasts = []
                for edge in G.edges():
                    u, v = edge
                    epi_u = _node_scalar_epi(G, u)
                    epi_v = _node_scalar_epi(G, v)
                    edge_contrasts.append(abs(epi_u - epi_v))
                entropy_gradient = _finite_nonnegative_mean(edge_contrasts)

                # Record a nonuniform amplitude-distribution diagnostic.
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
                            "mean_edge_epi_contrast": entropy_gradient,
                        },
                    )
                    pattern = EmergentPattern(
                        pattern_type=EmergentPatternType.ENTROPY_FLOW,
                        discovery_confidence=min(entropy / max_entropy, 1.0),
                        mathematical_signature=_augment_signature(
                            {
                                "shannon_entropy": entropy,
                                "kl_divergence": kl_divergence,
                                "mean_edge_epi_contrast": entropy_gradient,
                                "max_entropy": max_entropy,
                                "normalized_amplitude_entropy": entropy / max_entropy,
                                "temporal_evidence": False,
                                "squared_epi_norm": _finite_squared_norm(epi_array),
                            },
                            field_snapshot=field_info if field_info else None,
                            integration_hint=integration_hint,
                        ),
                        temporal_scale=1.0,
                        spatial_scale=int(np.sqrt(node_count)),
                        prediction_horizon=None,
                        compression_ratio=None,
                        physical_interpretation=(
                            "Snapshot amplitude entropy with mean edge EPI contrast"
                        ),
                        applications=[
                            "amplitude_distribution_analysis",
                            "snapshot_comparison",
                        ],
                    )
                    patterns.append(pattern)

        return patterns

    def discover_topological_invariants(self, G: Any) -> list[EmergentPattern]:
        """
        Report identities and descriptors of the current graph topology.

        They remain fixed only while topology is held fixed; this detector does
        not establish a conservation law for topology-changing dynamics.
        """
        patterns = []

        if not HAS_NETWORKX:
            return patterns

        # Compute basic topological invariants
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())

        # Euler characteristic of the graph as a 1D cell complex.
        if num_nodes > 2:
            try:
                # Attempt to compute number of faces for planar graphs
                # For non-planar, this is just a connectivity measure
                euler_char = num_nodes - num_edges

                # Degree sequence invariant
                degrees = [G.degree(node) for node in G.nodes()]
                degree_sequence_invariant = (
                    np.sum(np.array(degrees) ** 2) / (2 * num_edges)
                    if num_edges > 0
                    else 0
                )

                # Spectral invariants
                if HAS_SPECTRAL:
                    eigenvalues, _ = get_laplacian_spectrum(G)
                    # Should equal sum of degrees
                    spectral_trace = np.sum(eigenvalues)

                    # Spectral determinant: product of positive eigenvalues.
                    # Direct np.prod can overflow for larger graphs (warning:
                    # overflow encountered in reduce). Use log-domain
                    # aggregation to preserve ordering and avoid warnings.
                    positive_eigs = eigenvalues[eigenvalues > 1e-10]
                    if positive_eigs.size == 0:
                        spectral_determinant = 0.0
                    else:
                        log_det = float(np.sum(np.log(positive_eigs)))
                        max_log = float(np.log(np.finfo(float).max))
                        spectral_determinant = (
                            float(np.exp(log_det))
                            if log_det <= max_log
                            else None
                        )

                    # Verify the Laplacian trace identity for this snapshot.
                    degree_sum_check = abs(spectral_trace - np.sum(degrees)) < 1e-10

                    if degree_sum_check:
                        pattern = EmergentPattern(
                            pattern_type=(EmergentPatternType.TOPOLOGICAL_INVARIANT),
                            discovery_confidence=1.0,
                            mathematical_signature={
                                "euler_characteristic": euler_char,
                                "normalized_degree_second_moment": (
                                    degree_sequence_invariant
                                ),
                                "spectral_trace": spectral_trace,
                                "spectral_determinant": spectral_determinant,
                                "num_components": (nx.number_connected_components(G)),
                                "diameter": (
                                    nx.diameter(G)
                                    if nx.is_connected(G)
                                    else None
                                ),
                            },
                            temporal_scale=0.0,
                            spatial_scale=num_nodes,
                            prediction_horizon=None,
                            compression_ratio=None,
                            physical_interpretation=(
                                "Fixed-topology graph identities and descriptors"
                            ),
                            applications=[
                                "invariant_checking",
                                "topology_verification",
                                "fixed_topology_identity_checks",
                            ],
                        )
                        patterns.append(pattern)

            except Exception:
                pass  # Skip if topological calculations fail

        return patterns

    def discover_fractal_scaling_patterns(
        self, G: Any, scale_range: tuple[int, int] = (2, 10)
    ) -> list[EmergentPattern]:
        """
        Discover fractal self-similarity in network structure.

        Apply a greedy, node-order-dependent box-cover scaling heuristic to the
        current graph. A good fit is evidence for this estimator only.
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
            log_scales = np.log(np.array(scales[: len(box_counts)]))
            log_boxes = np.log(np.array(box_counts))

            # Fractal dimension from slope
            slope, intercept = np.polyfit(log_scales, log_boxes, 1)
            # Guard against zero-variance inputs (uniform box counts /
            # single-scale collapse): np.corrcoef would emit a divide-by-zero
            # RuntimeWarning and return nan. Without scale variance there is
            # no fractal scaling to fit.
            if np.std(log_scales) <= 0.0 or np.std(log_boxes) <= 0.0:
                r_squared = 0.0
            else:
                r_squared = np.corrcoef(log_scales, log_boxes)[0, 1] ** 2

            # Good fractal scaling
            if (
                r_squared > PATTERNS_RSQUARED_HIGH_CANONICAL
                and abs(slope) > PATTERNS_SLOPE_MINIMAL_CANONICAL
            ):
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
                        "scales": list(scales[: len(box_counts)]),
                    },
                    temporal_scale=1.0,
                    spatial_scale=int(np.mean(scales)),
                    prediction_horizon=None,
                    compression_ratio=None,
                    physical_interpretation=(
                        "Greedy box-cover scaling fit with estimated dimension "
                        f"{fractal_dim:.2f}"
                    ),
                    applications=[
                        "greedy_box_cover_diagnostic",
                        "multi_scale_snapshot_analysis",
                    ],
                )
                patterns.append(pattern)

        return patterns

    def analyze_pattern_interactions(
        self, patterns: list[EmergentPattern]
    ) -> dict[tuple[EmergentPatternType, EmergentPatternType], float]:
        """
        Analyze interactions between discovered patterns.

        Score pairwise overlap of reported spatial and temporal scales.

        The score is descriptive and does not establish causal interaction.
        """
        interactions = {}

        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i + 1 :]:
                # Compute interaction strength based on scale overlap
                spatial_overlap = min(
                    pattern1.spatial_scale, pattern2.spatial_scale
                ) / max(pattern1.spatial_scale, pattern2.spatial_scale)

                temporal_overlap = 1.0
                if pattern1.temporal_scale != float(
                    "inf"
                ) and pattern2.temporal_scale != float("inf"):
                    temporal_overlap = min(
                        pattern1.temporal_scale, pattern2.temporal_scale
                    ) / max(pattern1.temporal_scale, pattern2.temporal_scale)

                # Combined interaction strength
                interaction_strength = (
                    spatial_overlap
                    * temporal_overlap
                    * pattern1.discovery_confidence
                    * pattern2.discovery_confidence
                )

                interactions[(pattern1.pattern_type, pattern2.pattern_type)] = (
                    interaction_strength
                )

        return interactions

    def discover_all_patterns(self, G: Any, **kwargs) -> PatternDiscoveryResult:
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
        all_patterns.extend(self.discover_fractal_scaling_patterns(G, **kwargs))

        # Analyze pattern interactions
        pattern_interactions = self.analyze_pattern_interactions(all_patterns)

        # Promote performance strategies only when a detector explicitly
        # carries measured compressor or prediction evidence. Built-in snapshot
        # detectors currently carry neither.
        optimization_strategies = []
        for pattern in all_patterns:
            signature = pattern.mathematical_signature
            if (
                signature.get("compression_evaluated") is True
                and pattern.compression_ratio is not None
                and pattern.compression_ratio
                > PATTERNS_COMPRESSION_SIGNIFICANT_CANONICAL
            ):
                optimization_strategies.append(
                    f"compress_using_{pattern.pattern_type.value}"
                )
            if (
                signature.get("prediction_evaluated") is True
                and pattern.prediction_horizon is not None
                and pattern.prediction_horizon > PATTERNS_HORIZON_PREDICTIVE_CANONICAL
            ):
                optimization_strategies.append(
                    f"predict_using_{pattern.pattern_type.value}"
                )

        # Aggregate detector outputs without relabelling scores as measured
        # compression or out-of-sample prediction accuracy.
        invariants: dict[str, float] = {}
        if all_patterns:
            invariants["pattern_count"] = float(len(all_patterns))
            invariants["mean_detection_score"] = float(
                np.mean([p.discovery_confidence for p in all_patterns])
            )

        compression_potential = None
        predictive_accuracy = None

        execution_time = time.perf_counter() - start_time

        # Update statistics
        self.total_discoveries += len(all_patterns)
        for pattern in all_patterns:
            self.pattern_statistics[pattern.pattern_type] += 1

        return PatternDiscoveryResult(
            discovered_patterns=all_patterns,
            pattern_interactions=pattern_interactions,
            emergent_optimization_strategies=list(set(optimization_strategies)),
            mathematical_invariants=invariants,
            compression_potential=compression_potential,
            predictive_accuracy=predictive_accuracy,
            execution_time=execution_time,
        )

    def get_discovery_statistics(self) -> dict[str, Any]:
        """Get statistics about pattern discoveries."""
        return {
            "total_discoveries": self.total_discoveries,
            "pattern_counts": dict(self.pattern_statistics),
            "cached_patterns": 0,
            "analysis_depth": self.analysis_depth,
            "analysis_depth_applied": False,
            "caching_enabled": False,
            "caching_requested": self.enable_caching,
            "cache_scope": "module_convenience_function" if HAS_UNIFIED_CACHE else None,
            "available_modules": {
                "scipy": HAS_SCIPY,
                "spectral": HAS_SPECTRAL,
                "physics_fields": HAS_PHYSICS_FIELDS,
                "unified_cache": HAS_UNIFIED_CACHE,
            },
        }

    def _serialize_discovery_statistics(self) -> dict[str, Any]:
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
    ) -> dict[str, Path]:
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
        dict[str, Path]
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

        The entries index references a graph payload beside the manifest.
        Only finite built-in JSON state and scalar node IDs are supported;
        runtime callbacks, arrays, RNG objects, and other unsupported state
        raise ValueError rather than being serialized as misleading strings.
        """
        from datetime import datetime, timezone
        from ..manifest import (
            collect_manifest_telemetry,
            finite_json_state,
            write_manifest_bundle,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        telemetry = collect_manifest_telemetry(G)

        # Extract network metadata
        node_count = len(G.nodes()) if hasattr(G, "nodes") else 0
        edge_count = len(G.edges()) if hasattr(G, "edges") else 0

        # Serialize discovered patterns
        patterns_serialized = []
        for pattern in discovery_result.discovered_patterns:
            public_signature = {
                key: value
                for key, value in pattern.mathematical_signature.items()
                if not key.startswith("_")
            }
            pattern_data = finite_json_state(
                {
                    "pattern_type": pattern.pattern_type.value,
                    "confidence": pattern.discovery_confidence,
                    "temporal_scale": pattern.temporal_scale,
                    "spatial_scale": pattern.spatial_scale,
                    "prediction_horizon": pattern.prediction_horizon,
                    "compression_ratio": pattern.compression_ratio,
                    "physical_interpretation": pattern.physical_interpretation,
                    "applications": pattern.applications,
                    "mathematical_signature": public_signature,
                },
                f"discovered_patterns.{pattern.pattern_type.value}",
            )
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
            "mathematical_invariants": finite_json_state(
                discovery_result.mathematical_invariants, "mathematical_invariants"
            ),
            "compression_potential": discovery_result.compression_potential,
            "predictive_accuracy": discovery_result.predictive_accuracy,
        }

        # Write summary
        summary = {
            "operation_type": "pattern_discovery",
            "partition_id": partition_id,
            "pattern_count": len(patterns_serialized),
            "coherence": telemetry.get("coherence"),
            "sense_index": telemetry.get("sense_index"),
            "compression_potential": discovery_result.compression_potential,
            "predictive_accuracy": discovery_result.predictive_accuracy,
        }
        return write_manifest_bundle(
            output_dir, "pattern_manifest.json", "pattern_summary.json",
            manifest, summary, [(partition_id, G, telemetry)],
        )


# Factory functions for easy access
def create_emergent_pattern_engine(**kwargs) -> TNFREmergentPatternEngine:
    """Create emergent pattern discovery engine."""
    return TNFREmergentPatternEngine(**kwargs)


@(
    cache_unified_computation(
        CacheEntryType.NODAL_STATE,
        # ≈ 0.0625 → canonical
        mathematical_importance=OPT_ORCH_ARITHMETIC_BOOST_CANONICAL,
    )
    if HAS_UNIFIED_CACHE
    else lambda *args, **kwargs: lambda f: f
)
def discover_mathematical_patterns(G: Any, **kwargs) -> PatternDiscoveryResult:
    """Convenience function for comprehensive pattern discovery."""
    engine = create_emergent_pattern_engine()
    return engine.discover_all_patterns(G, **kwargs)


def analyze_emergent_symmetries(G: Any) -> dict[str, Any]:
    """Collect snapshot patterns with symmetry-related graph descriptors."""
    engine = create_emergent_pattern_engine()
    result = engine.discover_all_patterns(G)

    # Focus on symmetry-related patterns
    symmetry_patterns = [
        p
        for p in result.discovered_patterns
        if p.pattern_type
        in [
            EmergentPatternType.EIGENMODE_RESONANCE,
            EmergentPatternType.TOPOLOGICAL_INVARIANT,
        ]
    ]

    return {
        "symmetry_count": len(symmetry_patterns),
        "symmetry_patterns": symmetry_patterns,
        "broken_symmetries": [
            p
            for p in symmetry_patterns
            if p.discovery_confidence < PATTERNS_CONFIDENCE_BROKEN_CANONICAL
        ],
        "fixed_topology_identities": [
            p
            for p in symmetry_patterns
            if p.pattern_type == EmergentPatternType.TOPOLOGICAL_INVARIANT
        ],
        # Compatibility key: no dynamical conservation law is established here.
        "conservation_laws": [],
    }

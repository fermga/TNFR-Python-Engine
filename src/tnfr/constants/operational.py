#!/usr/bin/env python3
"""
TNFR Operational Engine-Tuning Constants
========================================

Operational knobs for the engine subsystems (optimization orchestrator, caches,
FFT engines, pattern discovery, self-optimization, integration estimates,
operator pattern scoring). These carry **no nodal-physics meaning** and must NOT
be cited as first-principles TNFR results.

This module is the home for every value that is purely an engine-tuning
parameter (cache sizes, speedup estimates, confidence thresholds, scoring
weights, performance/memory baselines). The genuine TNFR structural constants
— the π phase-wrap bounds, the spectral-gap ξ_C, the coherence band, the
operator gains, and the tetrad / phase / frequency scales — live in
``tnfr.constants.canonical`` and are the single source of structural truth.

Separation rationale (audit 2026): keeping these knobs out of ``canonical.py``
makes that file a pure statement of TNFR physics. A few values here equal π
incidentally (decorative "importance" weights); π is imported from canonical
for those, but that does NOT make them structural — they remain free tunables.

Author: TNFR Research Team
Date: 2026-06 (operational-knob relocation out of canonical.py)
"""

from __future__ import annotations

from .canonical import PI

# ============================================================================
# CYCLE DETECTION (operational balance rails — operators/cycle_detection.py)
# ============================================================================
# NOTE: the structural balance target CYCLE_OPTIMAL_BALANCE_CANONICAL = 1/(π+1)
# stays in canonical.py; only these tuning rails live here.
CYCLE_BALANCE_RANGE_LOW_CANONICAL = -0.1  # operational tuning (not TNFR physics)
CYCLE_BALANCE_RANGE_HIGH_CANONICAL = 0.49  # operational tuning (not TNFR physics)
CYCLE_BALANCE_MULTIPLIER_CANONICAL = 1.6  # operational tuning (not TNFR physics)
CYCLE_FALLBACK_SCORE_CANONICAL = 0.54  # operational tuning (not TNFR physics)
CYCLE_MIN_HEALTH_CANONICAL = 0.49  # operational tuning (not TNFR physics)

# ============================================================================
# PATTERN WEIGHTS (operators/patterns.py — domain scoring multipliers)
# ============================================================================
PATTERN_BASE_WEIGHT_CANONICAL = 1.0  # 1.0 (unit base weight)
PATTERN_THERAPEUTIC_WEIGHT_CANONICAL = 2.8  # operational tuning (not TNFR physics)
PATTERN_EDUCATIONAL_WEIGHT_CANONICAL = 0.74  # operational tuning (not TNFR physics)
PATTERN_ORGANIZATIONAL_WEIGHT_CANONICAL = 0.16  # operational tuning (not TNFR physics)
PATTERN_CREATIVE_WEIGHT_CANONICAL = 0.74  # operational tuning (not TNFR physics)
PATTERN_REGENERATIVE_WEIGHT_CANONICAL = 0.6  # operational tuning (not TNFR physics)
PATTERN_BOOTSTRAP_WEIGHT_CANONICAL = 1.07  # operational tuning (not TNFR physics)
PATTERN_EXPLORE_WEIGHT_CANONICAL = (
    PATTERN_BOOTSTRAP_WEIGHT_CANONICAL  # Same as bootstrap
)
PATTERN_STABILIZE_WEIGHT_CANONICAL = 0.62  # operational tuning (not TNFR physics)
PATTERN_COMPLEX_WEIGHT_CANONICAL = (
    PATTERN_STABILIZE_WEIGHT_CANONICAL  # Same as stabilize
)
PATTERN_COMPRESS_WEIGHT_CANONICAL = 0.93  # operational tuning (not TNFR physics)
PATTERN_LINEAR_WEIGHT_CANONICAL = 0.18  # operational tuning (not TNFR physics)

# ============================================================================
# ALGEBRAIC TOLERANCES (operators/algebra.py — operational precision)
# ============================================================================
ALGEBRA_EPI_TOLERANCE_CANONICAL = 0.04  # operational tuning (not TNFR physics)
ALGEBRA_VF_TOLERANCE_CANONICAL = 0.1  # operational tuning (not TNFR physics)
ALGEBRA_COMBINED_TOLERANCE_CANONICAL = 0.04  # operational tuning (not TNFR physics)

# ============================================================================
# PHYSICS NETWORK-STUDY CALIBRATION (physics/calibration.py — Watts–Strogatz
# regression coefficients; empirical study fit, not nodal physics)
# ============================================================================
PHYSICS_EXPECTED_CORRELATION_WS_CANONICAL = 0.1  # operational tuning (not TNFR physics)
PHYSICS_N_NODES_DEPENDENCY_CANONICAL = 0.04  # operational tuning (not TNFR physics)
PHYSICS_K_DEGREE_DEPENDENCY_CANONICAL = -0.16  # operational tuning (not TNFR physics)
PHYSICS_P_REWIRE_DEPENDENCY_CANONICAL = 0.09  # operational tuning (not TNFR physics)

# ============================================================================
# OPTIMIZATION ORCHESTRATOR (dynamics/optimization_orchestrator.py)
# ============================================================================
OPT_ORCH_DENSITY_THRESHOLD_CANONICAL = 0.1  # operational tuning (not TNFR physics)
OPT_ORCH_FFT_BOOST_CANONICAL = 1.16  # operational tuning (not TNFR physics)
OPT_ORCH_SMALL_PENALTY_CANONICAL = 0.34  # operational tuning (not TNFR physics)
OPT_ORCH_VECTORIZED_BOOST_CANONICAL = 0.6  # operational tuning (not TNFR physics)
OPT_ORCH_ARITHMETIC_BOOST_CANONICAL = 0.06  # operational tuning (not TNFR physics)
OPT_ORCH_DENSE_BOOST_CANONICAL = 0.37  # operational tuning (not TNFR physics)
OPT_ORCH_BEST_THRESHOLD_CANONICAL = 0.7  # operational tuning (not TNFR physics)
OPT_ORCH_VECTORIZED_SPEEDUP_CANONICAL = 0.93  # operational tuning (not TNFR physics)
OPT_ORCH_FFT_SPEEDUP_CANONICAL = 2.14  # operational tuning (not TNFR physics)
OPT_ORCH_CACHE_SPEEDUP_CANONICAL = PI  # π (incidental cache-hit speedup estimate)

# ============================================================================
# MULTI-MODAL CACHE (dynamics/multi_modal_cache.py)
# ============================================================================
MULTIMODAL_CACHE_TARGET_FRACTION_CANONICAL = 0.74  # operational tuning (not TNFR physics)
MULTIMODAL_CACHE_SPECTRAL_IMPORTANCE_CANONICAL = 1.16  # operational tuning (not TNFR physics)
MULTIMODAL_CACHE_TETRAD_IMPORTANCE_CANONICAL = (
    PI  # π (incidental tetrad-computation importance weight)
)
MULTIMODAL_CACHE_TARGET_CANONICAL = 0.74  # operational tuning (not TNFR physics)

# ============================================================================
# FFT ENGINES (dynamics/advanced_fft_arithmetic.py, fft_engine.py,
# fft_cache_coordinator.py, cache_aware_fft_engine.py)
# ============================================================================
FFT_ARITHMETIC_IMPORTANCE_CANONICAL = PI  # π (incidental mathematical importance weight)
FFT_LOW_CUTOFF_CANONICAL = 0.34  # operational tuning (not TNFR physics)
FFT_HIGH_CUTOFF_CANONICAL = 0.6  # operational tuning (not TNFR physics)
FFT_BANDWIDTH_CANONICAL = 0.1  # operational tuning (not TNFR physics)
FFT_COHERENT_THRESHOLD_CANONICAL = 0.62  # operational tuning (not TNFR physics)
FFT_CACHE_IMPORTANCE_HIGH_CANONICAL = PI  # π (incidental high-importance weight)
FFT_ENGINE_COUPLING_CANONICAL = 0.1  # operational tuning (not TNFR physics)
FFT_OPT_SEQUENTIAL_IMPROVEMENT_CANONICAL = 0.11  # operational tuning (not TNFR physics)
ARITHMETIC_FFT_ENHANCEMENT_CANONICAL = 1.06  # operational tuning (not TNFR physics)

# ============================================================================
# EMERGENT MATHEMATICAL PATTERNS (engines/pattern_discovery/mathematical_patterns.py)
# ============================================================================
PATTERNS_HIGH_CONFIDENCE_CANONICAL = 0.75  # operational tuning (not TNFR physics)
PATTERNS_COMPRESSION_RATIO_CANONICAL = 1.16  # operational tuning (not TNFR physics)
PATTERNS_RSQUARED_THRESHOLD_CANONICAL = 0.44  # operational tuning (not TNFR physics)
PATTERNS_SLOPE_THRESHOLD_CANONICAL = 0.34  # operational tuning (not TNFR physics)
PATTERNS_HORIZON_LONG_CANONICAL = 0.93  # operational tuning (not TNFR physics)
PATTERNS_COMPRESSION_OSCILLATORY_CANONICAL = 0.6  # operational tuning (not TNFR physics)
PATTERNS_ENTROPY_THRESHOLD_CANONICAL = 1.04  # operational tuning (not TNFR physics)
PATTERNS_DIVERGENCE_THRESHOLD_CANONICAL = 0.34  # operational tuning (not TNFR physics)
PATTERNS_HORIZON_MEDIUM_CANONICAL = PI  # π (incidental medium prediction horizon)
PATTERNS_RSQUARED_HIGH_CANONICAL = 0.44  # operational tuning (not TNFR physics)
PATTERNS_SLOPE_MINIMAL_CANONICAL = 0.1  # operational tuning (not TNFR physics)
PATTERNS_HORIZON_SHORT_CANONICAL = 1.16  # operational tuning (not TNFR physics)
PATTERNS_COMPRESSION_SIGNIFICANT_CANONICAL = 0.6  # operational tuning (not TNFR physics)
PATTERNS_HORIZON_PREDICTIVE_CANONICAL = 1.16  # operational tuning (not TNFR physics)
PATTERNS_CONFIDENCE_BROKEN_CANONICAL = 0.74  # operational tuning (not TNFR physics)

# ============================================================================
# SELF-OPTIMIZING ENGINE (dynamics/self_optimizing_engine.py)
# ============================================================================
SELF_OPT_CACHE_SIZE_CANONICAL = 2**8  # 256 (cache size MB)
SELF_OPT_COMPRESSION_HIGH_CANONICAL = 1.16  # operational tuning (not TNFR physics)
SELF_OPT_HORIZON_HIGH_CANONICAL = PI  # π (incidental high prediction horizon)
SELF_OPT_CHIRALITY_THRESHOLD_CANONICAL = 0.62  # operational tuning (not TNFR physics)
SELF_OPT_SYMMETRY_THRESHOLD_CANONICAL = 0.34  # operational tuning (not TNFR physics)
SELF_OPT_COUPLING_LOW_CANONICAL = 0.49  # operational tuning (not TNFR physics)
SELF_OPT_ENERGY_HIGH_CANONICAL = 1.16  # operational tuning (not TNFR physics)
SELF_OPT_DENSITY_SPARSE_CANONICAL = 0.16  # operational tuning (not TNFR physics)
SELF_OPT_DENSITY_DENSE_CANONICAL = 0.74  # operational tuning (not TNFR physics)
SELF_OPT_IMPROVEMENT_SIGNIFICANT_CANONICAL = 0.7  # operational tuning (not TNFR physics)
SELF_OPT_CACHE_LOW_FRACTION_CANONICAL = 0.34  # operational tuning (not TNFR physics)
SELF_OPT_SPEEDUP_HIGH_CANONICAL = 1.16  # operational tuning (not TNFR physics)
SELF_OPT_CACHE_EXPANSION_CANONICAL = 0.7  # operational tuning (not TNFR physics)
SELF_OPT_CACHE_HIGH_FRACTION_CANONICAL = 0.74  # operational tuning (not TNFR physics)
SELF_OPT_SPEEDUP_LOW_CANONICAL = 0.6  # operational tuning (not TNFR physics)
SELF_OPT_CACHE_CONTRACTION_CANONICAL = 0.74  # operational tuning (not TNFR physics)

# ============================================================================
# EMERGENT CENTRALIZATION (dynamics/emergent_centralization.py,
# cache_aware_fft_engine.py, propagation.py, physics/signatures.py)
# ============================================================================
EMERGENT_CENTRALITY_THRESHOLD_CANONICAL = 0.74  # operational tuning (not TNFR physics)
EMERGENT_COORDINATION_THRESHOLD_CANONICAL = 0.56  # operational tuning (not TNFR physics)
EMERGENT_STABILITY_THRESHOLD_CANONICAL = 0.59  # operational tuning (not TNFR physics)
EMERGENT_COUPLING_STRENGTH_CANONICAL = 0.44  # operational tuning (not TNFR physics)
EMERGENT_FREQ_BALANCE_CANONICAL = 0.46  # operational tuning (not TNFR physics)
EMERGENT_EFFICIENCY_GAIN_CANONICAL = 0.18  # operational tuning (not TNFR physics)
EMERGENT_COORDINATION_BOOST_CANONICAL = 1.03  # operational tuning (not TNFR physics)

# ============================================================================
# ADVANCED CACHE OPTIMIZER (dynamics/advanced_cache_optimizer.py — memory/time
# estimates in MB and seconds)
# ============================================================================
CACHE_OPT_PREFETCH_TIME_CANONICAL = 0.1  # operational tuning (not TNFR physics)
CACHE_OPT_SHARED_MEMORY_CANONICAL = 0.62  # operational tuning (not TNFR physics)
CACHE_OPT_FIELD_MEMORY_CANONICAL = 0.1  # operational tuning (not TNFR physics)
CACHE_OPT_HIGH_PRIORITY_CANONICAL = 0.74  # operational tuning (not TNFR physics)
CACHE_OPT_MEDIUM_PRIORITY_CANONICAL = 0.62  # operational tuning (not TNFR physics)
CACHE_OPT_LOW_PRIORITY_CANONICAL = 0.16  # operational tuning (not TNFR physics)
CACHE_OPT_PRESERVED_MEMORY_CANONICAL = 0.16  # operational tuning (not TNFR physics)
CACHE_OPT_MAX_EVICTION_CANONICAL = 0.34  # operational tuning (not TNFR physics)
CACHE_OPT_COMPRESSION_BASE_CANONICAL = 0.6  # operational tuning (not TNFR physics)
CACHE_OPT_COMPRESSION_SCALE_CANONICAL = 0.1  # operational tuning (not TNFR physics)
CACHE_OPT_COMPRESSION_MAX_CANONICAL = PI  # π (incidental max-compression estimate)
CACHE_OPT_LOCALITY_BASE_CANONICAL = 50  # 50 (base access count)
CACHE_OPT_LOCALITY_MAX_CANONICAL = 1.16  # operational tuning (not TNFR physics)
CACHE_OPT_LOCALITY_TIME_CANONICAL = 0.1  # operational tuning (not TNFR physics)
CACHE_OPT_LOCALITY_MEMORY_CANONICAL = 0.05  # operational tuning (not TNFR physics)
CACHE_OPT_LOCALITY_HIT_CANONICAL = 0.22  # operational tuning (not TNFR physics)
CACHE_OPT_SPECTRAL_TIME_CANONICAL = 1.8e-05  # operational tuning (not TNFR physics)
CACHE_OPT_SPECTRAL_MEMORY_CANONICAL = 0.00147  # operational tuning (not TNFR physics)

# ============================================================================
# UNIFIED MATHEMATICAL CACHE ORCHESTRATOR
# (dynamics/unified_mathematical_cache_orchestrator.py)
# ============================================================================
UNIFIED_CACHE_MIN_COHERENCE_CANONICAL = 0.62  # operational tuning (not TNFR physics)

# ============================================================================
# EMERGENT INTEGRATION ENGINE (dynamics/emergent_integration_engine.py —
# confidence levels and performance/memory baselines)
# ============================================================================
INTEGRATION_COMPUTATION_REDUCTION_CANONICAL = 0.34  # operational tuning (not TNFR physics)
INTEGRATION_MEMORY_SAVINGS_CANONICAL = 0.49  # operational tuning (not TNFR physics)
INTEGRATION_CACHE_EFFICIENCY_CANONICAL = 0.16  # operational tuning (not TNFR physics)
INTEGRATION_CONFIDENCE_HIGH_CANONICAL = 0.75  # operational tuning (not TNFR physics)
INTEGRATION_CENTRALITY_THRESHOLD_CANONICAL = 0.74  # operational tuning (not TNFR physics)
INTEGRATION_HIT_RATE_IMPROVE_CANONICAL = 0.02  # operational tuning (not TNFR physics)
INTEGRATION_MEMORY_REDUCE_CANONICAL = 0.22  # operational tuning (not TNFR physics)
INTEGRATION_ACCESS_TIME_CANONICAL = 0.16  # operational tuning (not TNFR physics)
INTEGRATION_CONFIDENCE_MEDIUM_CANONICAL = 0.43  # operational tuning (not TNFR physics)
INTEGRATION_SPEEDUP_CANONICAL = 0.49  # operational tuning (not TNFR physics)
INTEGRATION_EFFICIENCY_CANONICAL = 0.16  # operational tuning (not TNFR physics)
INTEGRATION_CPU_UTIL_CANONICAL = 0.34  # operational tuning (not TNFR physics)
INTEGRATION_CONFIDENCE_LOW_CANONICAL = 0.37  # operational tuning (not TNFR physics)
INTEGRATION_PRECOMPUTE_SUCCESS_CANONICAL = 0.3  # operational tuning (not TNFR physics)
INTEGRATION_COMPUTATION_AVOID_CANONICAL = 0.34  # operational tuning (not TNFR physics)
INTEGRATION_RESPONSE_TIME_CANONICAL = 0.02  # operational tuning (not TNFR physics)
INTEGRATION_CONFIDENCE_MINIMAL_CANONICAL = 0.34  # operational tuning (not TNFR physics)
INTEGRATION_SYNC_THRESHOLD_CANONICAL = 0.74  # operational tuning (not TNFR physics)
INTEGRATION_PREFETCH_ACCURACY_CANONICAL = 0.62  # operational tuning (not TNFR physics)
INTEGRATION_CACHE_EFF_CANONICAL = 0.16  # operational tuning (not TNFR physics)
INTEGRATION_SYNC_PREDICTION_CANONICAL = 0.34  # operational tuning (not TNFR physics)
INTEGRATION_CONFIDENCE_SYNC_CANONICAL = 0.28  # operational tuning (not TNFR physics)
INTEGRATION_MEMORY_MB_CANONICAL = 8.5  # operational tuning (not TNFR physics)
INTEGRATION_COMPUTATION_TIME_CANONICAL = 0.001352  # operational tuning (not TNFR physics)
INTEGRATION_COMPUTATION_BASELINE_CANONICAL = 0.1  # operational tuning (not TNFR physics)
INTEGRATION_MEMORY_BASELINE_CANONICAL = 50.0  # 50.0 MB (baseline)
INTEGRATION_CACHE_HIT_BASELINE_CANONICAL = 0.3  # operational tuning (not TNFR physics)
INTEGRATION_CPU_BASELINE_CANONICAL = 0.49  # operational tuning (not TNFR physics)
INTEGRATION_CONFIDENCE_THRESHOLD_CANONICAL = 0.74  # operational tuning (not TNFR physics)

# ============================================================================
# NODAL OPTIMIZER (dynamics/nodal_optimizer.py + shared coupling default)
# ============================================================================
NODAL_OPT_COUPLING_CANONICAL = 0.1  # operational tuning (not TNFR physics)
NODAL_OPT_TARGET_DT_CANONICAL = 0.1  # operational tuning (not TNFR physics)
NODAL_OPT_VECTORIZED_SPEEDUP_CANONICAL = 0.6  # operational tuning (not TNFR physics)
NODAL_OPT_PARALLEL_SPEEDUP_CANONICAL = 1.16  # operational tuning (not TNFR physics)
NODAL_OPT_CACHE_SPEEDUP_CANONICAL = 0.7  # operational tuning (not TNFR physics)
NODAL_OPT_ADAPTIVE_SPEEDUP_CANONICAL = 0.34  # operational tuning (not TNFR physics)

# ============================================================================
# STRUCTURAL CACHE (dynamics/structural_cache.py)
# ============================================================================
STRUCT_CACHE_INTERPOLATE_CANONICAL = 0.1  # operational tuning (not TNFR physics)
STRUCT_CACHE_EVICTION_CANONICAL = 0.74  # operational tuning (not TNFR physics)

# ============================================================================
# OPERATOR PATTERN SCORING (operators/patterns.py — structural-complexity and
# domain-suitability weights; references EMERGENT_FREQ_BALANCE above)
# ============================================================================
OPERATORS_PATTERN_UNIQUE_WEIGHT_CANONICAL = 0.49  # operational tuning (not TNFR physics)
OPERATORS_PATTERN_TRANSITION_WEIGHT_CANONICAL = EMERGENT_FREQ_BALANCE_CANONICAL  # = 0.46 (operational transition weight)
OPERATORS_PATTERN_DESTABILIZER_WEIGHT_CANONICAL = 0.16  # operational tuning (not TNFR physics)
OPERATORS_PATTERN_STABILIZER_WEIGHT_CANONICAL = 0.07  # operational tuning (not TNFR physics)
OPERATORS_THERAPEUTIC_HIGH_CANONICAL = 0.49  # operational tuning (not TNFR physics)
OPERATORS_EDUCATIONAL_HIGH_CANONICAL = 0.3  # operational tuning (not TNFR physics)
OPERATORS_CREATIVE_BASE_CANONICAL = 0.6  # operational tuning (not TNFR physics)
OPERATORS_ORGANIZATIONAL_CANONICAL = 0.37  # operational tuning (not TNFR physics)

# ============================================================================
# BUSINESS HEALTH (config/defaults_core.py, sdk health read-outs)
# ============================================================================
MIN_BUSINESS_COHERENCE_CANONICAL = 0.75  # operational tuning (not TNFR physics)

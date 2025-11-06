"""TNFR Structural Patterns - Compositional Building Blocks.

This module demonstrates specialized structural patterns for specific purposes,
completing the coverage of all patterns detectable by TNFR Grammar 2.0:

- BOOTSTRAP: Rapid system initialization with minimal operators
- EXPLORE: Controlled exploration with safe return to baseline
- STABILIZE: Consolidation and closure patterns
- RESONATE: Amplification and propagation through coupling
- COMPRESS: Concentration and simplification sequences
- COMPLEX: Multi-pattern combinations for advanced scenarios

Each pattern is optimized for its specific structural purpose and validated
using TNFR's canonical health metrics (coherence, balance, sustainability).

Examples
--------
>>> from structural_patterns import get_bootstrap_pattern
>>> sequence = get_bootstrap_pattern()
>>> from tnfr.operators.grammar import validate_sequence_with_health
>>> result = validate_sequence_with_health(sequence)
>>> result.passed
True
>>> result.detected_pattern
<StructuralPattern.BOOTSTRAP: 'bootstrap'>
"""

from tnfr.config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)


# =============================================================================
# BOOTSTRAP PATTERNS - Rapid Initialization
# =============================================================================


def get_bootstrap_pattern():
    """Bootstrap pattern - rapid system initialization with minimal operators.
    
    Context: Minimal initialization sequence for quickly establishing a stable
    structural foundation. Optimized for speed and effectiveness with the
    fewest possible operators while maintaining structural coherence.
    
    Note: Includes canonical RECEPTION→COHERENCE segment required by TNFR grammar.
    The BOOTSTRAP pattern signature (AL→UM→IL) is detected within the valid sequence.
    
    Structural flow:
    1. EMISSION (AL): Initialize system, seed initial coherence
    2. RECEPTION (EN): Gather initial environmental context
    3. COUPLING (UM): Establish basic network connections
    4. COHERENCE (IL): Achieve stable configuration
    5. SILENCE (SHA): Lock initial state
    
    Expected metrics:
    - Health score: > 0.65 (good structural quality despite minimalism)
    - Pattern: BOOTSTRAP (initialization sequence detected)
    - Efficiency: Very high (minimal operators for purpose)
    - Time to stability: Rapid (5 operators)
    
    Use cases:
    - Quick system startup
    - Network initialization
    - Minimal viable structure
    - Foundation for further elaboration
    - Rapid prototyping scenarios
    
    Returns
    -------
    list[str]
        Validated operator sequence for bootstrap initialization
    """
    return [
        EMISSION,      # AL: Initialize/start system
        RECEPTION,     # EN: Gather initial context (canonical requirement)
        COUPLING,      # UM: Establish basic connections
        COHERENCE,     # IL: Stable configuration
        SILENCE        # SHA: Lock initial state (valid ending)
    ]


def get_bootstrap_extended_pattern():
    """Extended bootstrap pattern with additional stabilization.
    
    Context: Bootstrap pattern with extra consolidation for higher initial
    stability. Useful when the initialized system needs stronger foundation
    before further operations. Adds resonance for harmonic reinforcement.
    
    Structural flow:
    1. EMISSION (AL): Initialize system
    2. RECEPTION (EN): Gather initial context from environment
    3. COUPLING (UM): Establish connections
    4. COHERENCE (IL): Stabilize configuration
    5. RESONANCE (RA): Amplify stable elements
    6. COHERENCE (IL): Reinforce stability
    7. SILENCE (SHA): Consolidate initial state
    
    Expected metrics:
    - Health score: > 0.75 (higher than minimal bootstrap)
    - Pattern: BOOTSTRAP or STABILIZE (longer sequence)
    - Sustainability: Higher than minimal bootstrap
    
    Use cases:
    - Critical systems requiring robust initialization
    - Complex networks needing strong foundation
    - Scenarios where initial stability is paramount
    
    Returns
    -------
    list[str]
        Extended bootstrap sequence with additional stabilization
    """
    return [
        EMISSION,      # AL: Initialize system
        RECEPTION,     # EN: Gather initial context
        COUPLING,      # UM: Establish connections
        COHERENCE,     # IL: Stabilize configuration
        RESONANCE,     # RA: Amplify stable elements
        COHERENCE,     # IL: Reinforce stability
        SILENCE        # SHA: Consolidate initial state
    ]


# =============================================================================
# EXPLORE PATTERNS - Controlled Exploration
# =============================================================================


def get_explore_pattern():
    """Explore pattern - controlled exploration with safe return to baseline.
    
    Context: Safe exploration sequence that ventures into structural
    alternatives while maintaining ability to return to stable baseline.
    Essential for discovery without risk of catastrophic destabilization.
    
    Structural flow:
    1. EMISSION (AL): Seed exploration system (canonical start)
    2. RECEPTION (EN): Establish baseline context (canonical requirement)
    3. COHERENCE (IL): Establish stable baseline (anchor point)
    4. DISSONANCE (OZ): Introduce controlled tension for exploration
    5. MUTATION (ZHIR): Explore alternative structural states
    6. COHERENCE (IL): Return to stability with new insights
    7. SILENCE (SHA): Consolidate discoveries
    
    Expected metrics:
    - Health score: > 0.70 (balanced exploration/stability)
    - Pattern: EXPLORE (exploration with return)
    - Balance: Good equilibrium between destabilizers and stabilizers
    - Return-to-baseline: Ensured by final coherence operators
    
    Use cases:
    - Safe system experimentation
    - Hypothesis testing in stable environments
    - Adaptive learning with safety constraints
    - Innovation with controlled risk
    - A/B testing structural alternatives
    
    Returns
    -------
    list[str]
        Validated operator sequence for controlled exploration
    """
    return [
        EMISSION,      # AL: Seed exploration (canonical start)
        RECEPTION,     # EN: Establish baseline context
        COHERENCE,     # IL: Stable baseline (anchor point)
        DISSONANCE,    # OZ: Introduce controlled tension
        MUTATION,      # ZHIR: Explore alternative states
        COHERENCE,     # IL: Return to stability with insights
        SILENCE        # SHA: Consolidate discoveries
    ]


def get_explore_deep_pattern():
    """Deep exploration pattern with extended search space.
    
    Context: Extended exploration for complex search spaces requiring
    multiple exploration cycles before consolidation. Maintains safety
    through periodic coherence checks.
    
    Structural flow:
    1. EMISSION (AL): Seed exploration (canonical start)
    2. RECEPTION (EN): Gather context (canonical requirement)
    3. COHERENCE (IL): Establish baseline
    4. EXPANSION (VAL): Increase exploration volume
    5. COHERENCE (IL): Stabilize expansion
    6. DISSONANCE (OZ): First exploration wave
    7. MUTATION (ZHIR): Explore first alternative
    8. COHERENCE (IL): Checkpoint stability
    9. DISSONANCE (OZ): Second exploration wave
    10. MUTATION (ZHIR): Explore second alternative
    11. COHERENCE (IL): Final consolidation
    12. SILENCE (SHA): Lock in discoveries
    
    Expected metrics:
    - Health score: > 0.68 (moderate due to extended exploration)
    - Pattern: EXPLORE or COMPLEX (multiple cycles)
    - Complexity: Higher than basic explore
    
    Use cases:
    - Multi-hypothesis testing
    - Complex optimization landscapes
    - Research and development scenarios
    
    Returns
    -------
    list[str]
        Extended exploration sequence with multiple cycles
    """
    return [
        EMISSION,      # AL: Seed exploration (canonical start)
        RECEPTION,     # EN: Gather context (canonical requirement)
        COHERENCE,     # IL: Establish baseline
        EXPANSION,     # VAL: Increase exploration volume
        COHERENCE,     # IL: Stabilize expansion
        DISSONANCE,    # OZ: First exploration wave
        MUTATION,      # ZHIR: Explore first alternative
        COHERENCE,     # IL: Checkpoint stability
        DISSONANCE,    # OZ: Second exploration wave
        MUTATION,      # ZHIR: Explore second alternative
        COHERENCE,     # IL: Final consolidation
        SILENCE        # SHA: Lock in discoveries
    ]


# =============================================================================
# STABILIZE PATTERNS - Consolidation and Closure
# =============================================================================


def get_stabilize_pattern():
    """Stabilize pattern - robust consolidation with high sustainability.
    
    Context: Consolidation sequence for achieving maximum structural
    stability. Used to lock in gains, establish equilibrium, and prepare
    for long-term maintenance. Emphasizes multiple stabilization layers.
    
    Structural flow:
    1. EMISSION (AL): Seed stabilization process (canonical start)
    2. RECEPTION (EN): Gather current state information (canonical requirement)
    3. COHERENCE (IL): Organize and structure information
    4. RESONANCE (RA): Amplify stable elements
    5. COHERENCE (IL): Reinforce stability
    6. SILENCE (SHA): Final consolidation pause
    
    Expected metrics:
    - Health score: > 0.80 (excellent stability)
    - Pattern: STABILIZE (consolidation ending detected)
    - Sustainability: Very high (multiple stabilizers)
    - Balance: Strongly favors stabilization
    
    Use cases:
    - System shutdown/hibernation preparation
    - Checkpoint creation
    - Long-term state preservation
    - Post-operation consolidation
    - Ensuring robustness before handoff
    
    Returns
    -------
    list[str]
        Validated operator sequence for robust stabilization
    """
    return [
        EMISSION,      # AL: Seed stabilization (canonical start)
        RECEPTION,     # EN: Gather current state information
        COHERENCE,     # IL: Organize and structure
        RESONANCE,     # RA: Amplify stable elements
        COHERENCE,     # IL: Reinforce stability
        SILENCE        # SHA: Final consolidation
    ]


def get_stabilize_recursive_pattern():
    """Recursive stabilization pattern for nested structures.
    
    Context: Stabilization that propagates through nested structural
    levels using recursivity. Ensures stability across all fractal scales.
    
    Structural flow:
    1. EMISSION (AL): Seed recursive stabilization (canonical start)
    2. RECEPTION (EN): Gather multi-scale context (canonical requirement)
    3. COHERENCE (IL): Top-level stabilization
    4. RECURSIVITY (REMESH): Propagate to nested structures
    5. RESONANCE (RA): Amplify stability at all levels
    6. COHERENCE (IL): Final consolidation
    7. SILENCE (SHA): Lock in multi-scale stability
    
    Expected metrics:
    - Health score: > 0.78 (very good with recursivity)
    - Pattern: STABILIZE or FRACTAL (recursive structure)
    - Sustainability: High across scales
    
    Use cases:
    - Hierarchical system stabilization
    - Nested network consolidation
    - Multi-scale coherence preservation
    
    Returns
    -------
    list[str]
        Recursive stabilization sequence for nested structures
    """
    return [
        EMISSION,      # AL: Seed recursive stabilization (canonical start)
        RECEPTION,     # EN: Gather multi-scale context
        COHERENCE,     # IL: Top-level stabilization
        RECURSIVITY,   # REMESH: Propagate to nested structures
        RESONANCE,     # RA: Amplify stability at all levels
        COHERENCE,     # IL: Final consolidation
        SILENCE        # SHA: Lock in multi-scale stability
    ]


# =============================================================================
# RESONATE PATTERNS - Amplification and Propagation
# =============================================================================


def get_resonate_pattern():
    """Resonate pattern - structural amplification through coupling.
    
    Context: Amplification sequence for propagating structural patterns
    through network coupling while preserving pattern identity. Creates
    harmonic reinforcement through synchronized resonance.
    
    Structural flow:
    1. EMISSION (AL): Seed resonance pattern (canonical start)
    2. RECEPTION (EN): Establish network context (canonical requirement)
    3. COHERENCE (IL): Establish base frequency/structure
    4. RESONANCE (RA): First amplification wave
    5. COUPLING (UM): Synchronize amplifications across network
    6. RESONANCE (RA): Second amplification (harmonic)
    7. COHERENCE (IL): Stabilize amplified state
    8. SILENCE (SHA): Lock resonant state
    
    Expected metrics:
    - Health score: > 0.75 (good with balanced amplification)
    - Pattern: RESONATE (amplification through coupling)
    - Frequency harmony: High (multiple resonance operators)
    - Propagation: Effective (coupling synchronizes)
    
    Use cases:
    - Pattern broadcasting through networks
    - Collective synchronization
    - Signal amplification
    - Harmonic reinforcement
    - Spreading stable configurations
    
    Returns
    -------
    list[str]
        Validated operator sequence for resonant amplification
    """
    return [
        EMISSION,      # AL: Seed resonance pattern (canonical start)
        RECEPTION,     # EN: Establish network context
        COHERENCE,     # IL: Base frequency/structure
        RESONANCE,     # RA: First amplification
        COUPLING,      # UM: Synchronize amplifications
        RESONANCE,     # RA: Second amplification (harmonic)
        COHERENCE,     # IL: Stabilize amplified state
        SILENCE        # SHA: Lock resonant state
    ]


def get_resonate_cascade_pattern():
    """Cascading resonance pattern for network-wide propagation.
    
    Context: Extended resonance cascade that propagates through multiple
    network layers with self-organization. Creates emergent harmonic
    structures through cascading amplification.
    
    Structural flow:
    1. EMISSION (AL): Seed initial pattern (canonical start)
    2. RECEPTION (EN): Establish network baseline (canonical requirement)
    3. COHERENCE (IL): Base structure
    4. RESONANCE (RA): Initial amplification
    5. COUPLING (UM): First layer synchronization
    6. RESONANCE (RA): Second wave amplification
    7. COUPLING (UM): Second layer synchronization
    8. COHERENCE (IL): Consolidate resonance
    9. DISSONANCE (OZ): Enable emergent reorganization
    10. SELF_ORGANIZATION (THOL): Emergent harmonic structures
    11. COHERENCE (IL): Stabilize emergence
    12. SILENCE (SHA): Lock cascaded state
    
    Expected metrics:
    - Health score: > 0.72 (good with complexity)
    - Pattern: RESONATE or COMPLEX (cascading structure)
    - Emergent properties: Present (self-organization)
    
    Use cases:
    - Network-wide information spreading
    - Viral pattern propagation
    - Collective behavior emergence
    - Distributed synchronization
    
    Returns
    -------
    list[str]
        Cascading resonance sequence for network propagation
    """
    return [
        EMISSION,            # AL: Seed initial pattern (canonical start)
        RECEPTION,           # EN: Establish network baseline
        COHERENCE,           # IL: Base structure
        RESONANCE,           # RA: Initial amplification
        COUPLING,            # UM: First layer synchronization
        RESONANCE,           # RA: Second wave amplification
        COUPLING,            # UM: Second layer synchronization
        COHERENCE,           # IL: Consolidate resonance
        DISSONANCE,          # OZ: Enable emergent reorganization
        SELF_ORGANIZATION,   # THOL: Emergent harmonic structures
        COHERENCE,           # IL: Stabilize emergence
        SILENCE              # SHA: Lock cascaded state
    ]


# =============================================================================
# COMPRESS PATTERNS - Concentration and Simplification
# =============================================================================


def get_compress_pattern():
    """Compress pattern - concentration and essentialization.
    
    Context: Simplification sequence for concentrating complex structures
    toward their essential core. Removes redundancy and extracts minimal
    viable structural representation. Note: EXPANSION must be followed by
    COHERENCE before CONTRACTION per canonical rules.
    
    Structural flow:
    1. EMISSION (AL): Seed compression process (canonical start)
    2. RECEPTION (EN): Gather full state (canonical requirement)
    3. COHERENCE (IL): Organize initial state
    4. EXPANSION (VAL): Start from expanded/complex state
    5. COHERENCE (IL): Stabilize expansion (canonical requirement)
    6. CONTRACTION (NUL): Concentrate toward essence
    7. COHERENCE (IL): Organize concentrated form
    8. CONTRACTION (NUL): Further refinement
    9. COHERENCE (IL): Stabilize essential form
    10. SILENCE (SHA): Pure essential state
    
    Expected metrics:
    - Health score: > 0.65 (good despite multiple transitions)
    - Pattern: COMPRESS or LINEAR (simplification sequence)
    - Complexity efficiency: High (value-to-complexity ratio)
    - Information density: High (concentrated essence)
    
    Use cases:
    - Data compression analogues
    - System simplification
    - Essential feature extraction
    - Minimal representation discovery
    - Computational efficiency optimization
    
    Returns
    -------
    list[str]
        Validated operator sequence for structural compression
    """
    return [
        EMISSION,      # AL: Seed compression (canonical start)
        RECEPTION,     # EN: Gather full state
        COHERENCE,     # IL: Organize initial state
        EXPANSION,     # VAL: Start from expanded/complex state
        COHERENCE,     # IL: Stabilize expansion (required after VAL)
        CONTRACTION,   # NUL: Concentrate toward essence
        COHERENCE,     # IL: Organize concentrated form
        CONTRACTION,   # NUL: Further refinement
        COHERENCE,     # IL: Stabilize essential form
        SILENCE        # SHA: Pure essential state
    ]


def get_compress_adaptive_pattern():
    """Adaptive compression with exploration of compression paths.
    
    Context: Compression that explores multiple simplification paths
    before selecting the most coherent. Uses mutation to test different
    compression strategies. Note: EXPANSION must be followed by COHERENCE
    before DISSONANCE per canonical rules.
    
    Structural flow:
    1. EMISSION (AL): Seed adaptive compression (canonical start)
    2. RECEPTION (EN): Gather initial complexity (canonical requirement)
    3. COHERENCE (IL): Organize initial state
    4. EXPANSION (VAL): Begin with full complexity
    5. COHERENCE (IL): Stabilize expansion (canonical requirement)
    6. DISSONANCE (OZ): Explore compression tensions
    7. CONTRACTION (NUL): First compression attempt
    8. MUTATION (ZHIR): Try alternative compression path
    9. CONTRACTION (NUL): Select best compression
    10. COHERENCE (IL): Stabilize compressed form
    11. SILENCE (SHA): Lock essential state
    
    Expected metrics:
    - Health score: > 0.65 (moderate with exploration)
    - Pattern: COMPRESS, EXPLORE, or LINEAR (adaptive compression)
    - Efficiency: High after exploration
    
    Use cases:
    - Optimal compression path discovery
    - Adaptive dimensionality reduction
    - Feature selection with exploration
    
    Returns
    -------
    list[str]
        Adaptive compression sequence with exploration
    """
    return [
        EMISSION,      # AL: Seed adaptive compression (canonical start)
        RECEPTION,     # EN: Gather initial complexity
        COHERENCE,     # IL: Organize initial state
        EXPANSION,     # VAL: Begin with full complexity
        COHERENCE,     # IL: Stabilize expansion (required after VAL)
        DISSONANCE,    # OZ: Explore compression tensions
        CONTRACTION,   # NUL: First compression attempt
        COHERENCE,     # IL: Stabilize compression (required after NUL)
        MUTATION,      # ZHIR: Try alternative compression path
        COHERENCE,     # IL: Stabilize mutation (required after ZHIR)
        CONTRACTION,   # NUL: Select best compression
        COHERENCE,     # IL: Stabilize compressed form
        SILENCE        # SHA: Lock essential state
    ]


# =============================================================================
# COMPLEX PATTERNS - Multi-Pattern Combinations
# =============================================================================


def get_complex_pattern():
    """Complex pattern - integration of multiple structural patterns.
    
    Context: Advanced sequence combining multiple pattern components
    (BOOTSTRAP, EXPLORE, STABILIZE) for comprehensive structural evolution.
    Demonstrates how compositional patterns can be combined while
    maintaining overall coherence.
    
    Structural flow:
    1-5. BOOTSTRAP component: AL→EN→UM→IL→SHA (initialize)
    6-9. EXPLORE component: OZ→ZHIR→IL→SHA (explore alternatives)
    10-12. STABILIZE component: RA→IL→SHA (consolidate results)
    
    Expected metrics:
    - Health score: > 0.75 (good despite complexity)
    - Pattern: COMPLEX (multiple patterns detected)
    - Components: bootstrap, explore, stabilize
    - Overall balance: Maintained through pattern composition
    
    Use cases:
    - Complete lifecycle operations
    - Full system evolution cycles
    - Comprehensive workflows
    - Multi-stage processes
    - End-to-end transformations
    
    Returns
    -------
    list[str]
        Multi-pattern sequence demonstrating pattern composition
    """
    return [
        # BOOTSTRAP component
        EMISSION,      # AL: Initialize
        RECEPTION,     # EN: Gather context (canonical requirement)
        COUPLING,      # UM: Connect
        COHERENCE,     # IL: Stabilize initialization
        # EXPLORE component
        DISSONANCE,    # OZ: Explore
        MUTATION,      # ZHIR: Transform
        COHERENCE,     # IL: Consolidate exploration
        # STABILIZE component
        RESONANCE,     # RA: Amplify stable elements
        COHERENCE,     # IL: Final coherence
        SILENCE        # SHA: Lock results
    ]


def get_complex_full_cycle_pattern():
    """Full cycle complex pattern with regenerative elements.
    
    Context: Complete structural evolution cycle incorporating initialization,
    exploration, transformation, self-organization, and regenerative closure.
    Demonstrates maximum compositional complexity while maintaining coherence.
    Note: Follows all canonical operator compatibility rules.
    
    Structural flow:
    1. EMISSION (AL): Seed system
    2. RECEPTION (EN): Gather context (canonical requirement)
    3. COUPLING (UM): Establish network
    4. COHERENCE (IL): Initial stability
    5. EXPANSION (VAL): Expand search space
    6. COHERENCE (IL): Stabilize expansion (required after VAL)
    7. DISSONANCE (OZ): Explore tensions
    8. MUTATION (ZHIR): Transform structure
    9. COHERENCE (IL): Stabilize transformation (required after ZHIR)
    10. DISSONANCE (OZ): Enable emergent reorganization
    11. SELF_ORGANIZATION (THOL): Emergent reorganization
    12. COHERENCE (IL): Consolidate emergence
    13. RESONANCE (RA): Amplify success
    14. TRANSITION (NAV): Navigate to final state
    15. COHERENCE (IL): Final stability
    16. RECURSIVITY (REMESH): Fractal consolidation
    17. SILENCE (SHA): Complete closure
    
    Expected metrics:
    - Health score: > 0.68 (good given high complexity)
    - Pattern: COMPLEX (multiple high-level patterns)
    - Completeness: Maximum (full cycle)
    - All structural phases: Represented
    
    Use cases:
    - Complete system lifecycle
    - Full transformational journeys
    - Comprehensive evolution cycles
    - Maximum structural sophistication scenarios
    
    Returns
    -------
    list[str]
        Full cycle sequence with all major structural components
    """
    return [
        EMISSION,            # AL: Seed system
        RECEPTION,           # EN: Gather context
        COUPLING,            # UM: Establish network
        COHERENCE,           # IL: Initial stability
        EXPANSION,           # VAL: Expand search space
        COHERENCE,           # IL: Stabilize expansion (required after VAL)
        DISSONANCE,          # OZ: Explore tensions
        MUTATION,            # ZHIR: Transform structure
        COHERENCE,           # IL: Stabilize transformation (required after ZHIR)
        DISSONANCE,          # OZ: Enable emergent reorganization
        SELF_ORGANIZATION,   # THOL: Emergent reorganization
        COHERENCE,           # IL: Consolidate emergence
        RESONANCE,           # RA: Amplify success
        TRANSITION,          # NAV: Navigate to final state
        COHERENCE,           # IL: Final stability
        RECURSIVITY,         # REMESH: Fractal consolidation
        COHERENCE,           # IL: Stabilize recursion (required after REMESH)
        SILENCE              # SHA: Complete closure
    ]


# =============================================================================
# Pattern Catalog Export
# =============================================================================


def get_all_patterns():
    """Get all structural patterns as a catalog.
    
    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping pattern names to operator sequences
    """
    return {
        # Bootstrap patterns
        "bootstrap": get_bootstrap_pattern(),
        "bootstrap_extended": get_bootstrap_extended_pattern(),
        
        # Explore patterns
        "explore": get_explore_pattern(),
        "explore_deep": get_explore_deep_pattern(),
        
        # Stabilize patterns
        "stabilize": get_stabilize_pattern(),
        "stabilize_recursive": get_stabilize_recursive_pattern(),
        
        # Resonate patterns
        "resonate": get_resonate_pattern(),
        "resonate_cascade": get_resonate_cascade_pattern(),
        
        # Compress patterns
        "compress": get_compress_pattern(),
        "compress_adaptive": get_compress_adaptive_pattern(),
        
        # Complex patterns
        "complex": get_complex_pattern(),
        "complex_full_cycle": get_complex_full_cycle_pattern(),
    }


if __name__ == "__main__":
    """Validate all structural patterns."""
    from tnfr.operators.grammar import validate_sequence_with_health
    
    patterns = get_all_patterns()
    
    print("TNFR Structural Patterns Validation")
    print("=" * 60)
    
    for name, sequence in patterns.items():
        result = validate_sequence_with_health(sequence)
        health = result.health_metrics.overall_health if result.passed and result.health_metrics else 0.0
        pattern = result.health_metrics.dominant_pattern if result.passed and result.health_metrics else "unknown"
        
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{status} {name}")
        print(f"  Pattern: {pattern}")
        print(f"  Health: {health:.3f}")
        print(f"  Operators: {' → '.join(sequence)}")
        
        if not result.passed:
            print(f"  Error: {result.message}")
    
    print("\n" + "=" * 60)
    passed = sum(1 for name in patterns if validate_sequence_with_health(patterns[name]).passed)
    print(f"Results: {passed}/{len(patterns)} patterns passed validation")

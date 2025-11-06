"""TNFR Pattern Library - Curated Collection of Tested Sequences.

This module provides a comprehensive library of pre-validated operator
sequences organized by pattern type and use case. All sequences have been
tested and validated against TNFR canonical grammar.

Purpose
-------
- Quick reference for common patterns
- Starting points for custom sequences
- Variations demonstrating different approaches
- Validated examples for learning TNFR

Organization
------------
Sequences are organized by pattern family:
- BOOTSTRAP: Initialization patterns
- EXPLORE: Exploration and discovery patterns
- STABILIZE: Consolidation and closure patterns
- RESONATE: Amplification and propagation patterns
- COMPRESS: Simplification patterns
- COMPLEX: Multi-pattern combinations

Each entry includes:
- Operator sequence
- Expected health score range
- Detected pattern type
- Primary use case
- Key characteristics

Examples
--------
>>> from pattern_library import PATTERN_LIBRARY
>>> bootstrap_seq = PATTERN_LIBRARY['bootstrap']['minimal']
>>> print(bootstrap_seq['sequence'])
['emission', 'reception', 'coupling', 'coherence', 'silence']
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
# BOOTSTRAP PATTERNS - System Initialization
# =============================================================================


BOOTSTRAP_PATTERNS = {
    "minimal": {
        "sequence": [EMISSION, RECEPTION, COUPLING, COHERENCE, SILENCE],
        "health_range": (0.65, 0.70),
        "detected_pattern": "bootstrap/activation",
        "use_case": "Rapid system startup with minimal overhead",
        "characteristics": ["Fastest initialization", "Basic connectivity", "Stable foundation"],
    },
    "enhanced": {
        "sequence": [EMISSION, RECEPTION, COUPLING, COHERENCE, RESONANCE, COHERENCE, SILENCE],
        "health_range": (0.67, 0.72),
        "detected_pattern": "bootstrap/activation",
        "use_case": "Initialization with harmonic reinforcement",
        "characteristics": ["Enhanced stability", "Resonant amplification", "Stronger foundation"],
    },
    "networked": {
        "sequence": [EMISSION, RECEPTION, COUPLING, COHERENCE, COUPLING, RESONANCE, COHERENCE, SILENCE],
        "health_range": (0.68, 0.73),
        "detected_pattern": "bootstrap/activation",
        "use_case": "Network-focused initialization",
        "characteristics": ["Multi-layer connections", "Network synchronization", "Distributed stability"],
    },
}


# =============================================================================
# EXPLORE PATTERNS - Controlled Exploration
# =============================================================================


EXPLORE_PATTERNS = {
    "basic": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE],
        "health_range": (0.80, 0.85),
        "detected_pattern": "explore/activation",
        "use_case": "Safe single-hypothesis testing",
        "characteristics": ["Single exploration cycle", "Safe return", "High stability"],
    },
    "dual_hypothesis": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE],
        "health_range": (0.82, 0.87),
        "detected_pattern": "explore/activation",
        "use_case": "Testing multiple alternatives",
        "characteristics": ["Multiple exploration cycles", "Extended search space", "Controlled bifurcation"],
    },
    "deep_search": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, EXPANSION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE],
        "health_range": (0.78, 0.83),
        "detected_pattern": "explore/complex",
        "use_case": "Extensive exploration with dimensional expansion",
        "characteristics": ["Maximum search depth", "Multiple expansions", "Comprehensive coverage"],
    },
    "conservative": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE, MUTATION, COHERENCE, SILENCE],
        "health_range": (0.75, 0.80),
        "detected_pattern": "explore/activation",
        "use_case": "Minimal exploration with extra safety",
        "characteristics": ["Stabilization before mutation", "Extra coherence checkpoints", "Maximum safety"],
    },
}


# =============================================================================
# STABILIZE PATTERNS - Consolidation and Closure
# =============================================================================


STABILIZE_PATTERNS = {
    "standard": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, COHERENCE, SILENCE],
        "health_range": (0.65, 0.70),
        "detected_pattern": "stabilize/activation",
        "use_case": "Basic consolidation for checkpointing",
        "characteristics": ["Quick consolidation", "Resonant reinforcement", "Stable closure"],
    },
    "robust": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, COHERENCE, RESONANCE, COHERENCE, SILENCE],
        "health_range": (0.68, 0.73),
        "detected_pattern": "stabilize/activation",
        "use_case": "High-reliability stabilization",
        "characteristics": ["Multiple stabilization layers", "Double resonance", "Maximum sustainability"],
    },
    "fractal": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, RECURSIVITY, RESONANCE, COHERENCE, SILENCE],
        "health_range": (0.71, 0.76),
        "detected_pattern": "fractal/regenerative",
        "use_case": "Multi-scale stabilization",
        "characteristics": ["Recursive propagation", "Scale-invariant", "Nested stability"],
    },
    "transitional": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, TRANSITION, COHERENCE, RECURSIVITY, COHERENCE, SILENCE],
        "health_range": (0.70, 0.75),
        "detected_pattern": "regenerative/cyclic",
        "use_case": "Stabilization with handoff preparation",
        "characteristics": ["State transition", "Prepared for continuation", "Regenerative closure"],
    },
}


# =============================================================================
# RESONATE PATTERNS - Amplification and Propagation
# =============================================================================


RESONATE_PATTERNS = {
    "harmonic": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, COUPLING, RESONANCE, COHERENCE, SILENCE],
        "health_range": (0.67, 0.72),
        "detected_pattern": "resonate/activation",
        "use_case": "Network-wide pattern broadcasting",
        "characteristics": ["Harmonic amplification", "Network synchronization", "Pattern propagation"],
    },
    "cascade": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, COUPLING, RESONANCE, COUPLING, COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE, SILENCE],
        "health_range": (0.73, 0.78),
        "detected_pattern": "therapeutic/hierarchical",
        "use_case": "Emergent harmonic structures",
        "characteristics": ["Cascading amplification", "Emergent organization", "Multi-layer propagation"],
    },
    "triple": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, COUPLING, RESONANCE, COUPLING, RESONANCE, COHERENCE, SILENCE],
        "health_range": (0.69, 0.74),
        "detected_pattern": "resonate/activation",
        "use_case": "Triple harmonic reinforcement",
        "characteristics": ["Three resonance waves", "Maximum amplification", "Strong propagation"],
    },
}


# =============================================================================
# COMPRESS PATTERNS - Simplification and Extraction
# =============================================================================


COMPRESS_PATTERNS = {
    "standard": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, CONTRACTION, COHERENCE, CONTRACTION, COHERENCE, SILENCE],
        "health_range": (0.74, 0.79),
        "detected_pattern": "compress/activation",
        "use_case": "Standard compression with dual contraction",
        "characteristics": ["Balanced compression", "Dual refinement", "High efficiency"],
    },
    "adaptive": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, DISSONANCE, CONTRACTION, COHERENCE, MUTATION, COHERENCE, CONTRACTION, COHERENCE, SILENCE],
        "health_range": (0.76, 0.81),
        "detected_pattern": "compress/explore",
        "use_case": "Exploration of compression paths",
        "characteristics": ["Adaptive strategy", "Mutation explores alternatives", "Optimized compression"],
    },
    "aggressive": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, CONTRACTION, COHERENCE, CONTRACTION, COHERENCE, CONTRACTION, COHERENCE, SILENCE],
        "health_range": (0.72, 0.77),
        "detected_pattern": "compress/activation",
        "use_case": "Maximum compression with triple contraction",
        "characteristics": ["Aggressive simplification", "Maximum essence extraction", "Minimal representation"],
    },
}


# =============================================================================
# COMPLEX PATTERNS - Multi-Pattern Integration
# =============================================================================


COMPLEX_PATTERNS = {
    "standard": {
        "sequence": [EMISSION, RECEPTION, COUPLING, COHERENCE, DISSONANCE, MUTATION, COHERENCE, RESONANCE, COHERENCE, SILENCE],
        "health_range": (0.76, 0.81),
        "detected_pattern": "complex/activation",
        "use_case": "Bootstrap + Explore + Stabilize integration",
        "characteristics": ["Three pattern components", "Balanced composition", "Complete mini-cycle"],
    },
    "full_lifecycle": {
        "sequence": [
            EMISSION, RECEPTION, COUPLING, COHERENCE,
            EXPANSION, COHERENCE, DISSONANCE, MUTATION, COHERENCE,
            DISSONANCE, SELF_ORGANIZATION, COHERENCE,
            RESONANCE, TRANSITION, COHERENCE, RECURSIVITY, COHERENCE, SILENCE
        ],
        "health_range": (0.79, 0.84),
        "detected_pattern": "therapeutic/complex",
        "use_case": "Complete system lifecycle",
        "characteristics": ["All structural phases", "Maximum completeness", "Full transformation cycle"],
    },
    "regenerative": {
        "sequence": [
            EMISSION, RECEPTION, COUPLING, COHERENCE,
            RESONANCE, EXPANSION, COHERENCE, TRANSITION,
            COHERENCE, RECURSIVITY, COHERENCE, SILENCE
        ],
        "health_range": (0.74, 0.79),
        "detected_pattern": "regenerative/cyclic",
        "use_case": "Self-sustaining cycles",
        "characteristics": ["Regenerative closure", "Cyclic structure", "Self-sustaining"],
    },
}


# =============================================================================
# SPECIALIZED PATTERNS - Domain-Specific Applications
# =============================================================================


SPECIALIZED_PATTERNS = {
    "therapeutic_healing": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE, SILENCE],
        "health_range": (0.72, 0.77),
        "detected_pattern": "therapeutic",
        "use_case": "Crisis resolution and healing",
        "characteristics": ["Controlled crisis", "Emergent reorganization", "Healing closure"],
    },
    "educational_learning": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, DISSONANCE, MUTATION, TRANSITION, COHERENCE, SILENCE],
        "health_range": (0.77, 0.82),
        "detected_pattern": "educational",
        "use_case": "Transformative learning cycles",
        "characteristics": ["Knowledge expansion", "Cognitive transition", "Phase shift learning"],
    },
    "creative_emergence": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, SILENCE, EMISSION, EXPANSION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SELF_ORGANIZATION, COHERENCE, SILENCE],
        "health_range": (0.75, 0.80),
        "detected_pattern": "creative",
        "use_case": "Artistic creation processes",
        "characteristics": ["Contemplative start", "Creative breakthrough", "Emergent artistry"],
    },
}


# =============================================================================
# COMPLETE PATTERN LIBRARY
# =============================================================================


PATTERN_LIBRARY = {
    "bootstrap": BOOTSTRAP_PATTERNS,
    "explore": EXPLORE_PATTERNS,
    "stabilize": STABILIZE_PATTERNS,
    "resonate": RESONATE_PATTERNS,
    "compress": COMPRESS_PATTERNS,
    "complex": COMPLEX_PATTERNS,
    "specialized": SPECIALIZED_PATTERNS,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_pattern(family: str, variant: str):
    """Get a specific pattern from the library.
    
    Parameters
    ----------
    family : str
        Pattern family (e.g., 'bootstrap', 'explore')
    variant : str
        Pattern variant (e.g., 'minimal', 'enhanced')
        
    Returns
    -------
    dict or None
        Pattern information or None if not found
    """
    return PATTERN_LIBRARY.get(family, {}).get(variant)


def list_patterns_by_family(family: str):
    """List all pattern variants in a family.
    
    Parameters
    ----------
    family : str
        Pattern family name
        
    Returns
    -------
    list
        List of variant names
    """
    return list(PATTERN_LIBRARY.get(family, {}).keys())


def get_all_patterns_flat():
    """Get all patterns in a flat dictionary.
    
    Returns
    -------
    dict
        Dictionary mapping 'family.variant' to pattern info
    """
    flat = {}
    for family, variants in PATTERN_LIBRARY.items():
        for variant, pattern in variants.items():
            key = f"{family}.{variant}"
            flat[key] = pattern
    return flat


def search_patterns(use_case_keyword: str = None, min_health: float = None, max_length: int = None):
    """Search patterns by criteria.
    
    Parameters
    ----------
    use_case_keyword : str, optional
        Keyword to search in use case description
    min_health : float, optional
        Minimum health score (lower bound)
    max_length : int, optional
        Maximum sequence length
        
    Returns
    -------
    list
        List of matching pattern keys ('family.variant')
    """
    matches = []
    flat = get_all_patterns_flat()
    
    for key, pattern in flat.items():
        # Check use case keyword
        if use_case_keyword and use_case_keyword.lower() not in pattern['use_case'].lower():
            continue
        
        # Check min health
        if min_health is not None and pattern['health_range'][0] < min_health:
            continue
        
        # Check max length
        if max_length is not None and len(pattern['sequence']) > max_length:
            continue
        
        matches.append(key)
    
    return matches


if __name__ == "__main__":
    """Display pattern library overview."""
    from tnfr.operators.grammar import validate_sequence_with_health
    
    print("=" * 70)
    print("TNFR Pattern Library - Overview")
    print("=" * 70)
    print()
    
    # Count patterns
    total = sum(len(variants) for variants in PATTERN_LIBRARY.values())
    print(f"Total Patterns: {total}")
    print()
    
    # List families
    print("Pattern Families:")
    for family in PATTERN_LIBRARY.keys():
        count = len(PATTERN_LIBRARY[family])
        print(f"  - {family}: {count} variants")
    print()
    
    # Show example from each family
    print("Sample Patterns (one from each family):")
    print("-" * 70)
    for family in PATTERN_LIBRARY.keys():
        variants = list(PATTERN_LIBRARY[family].keys())
        if variants:
            variant = variants[0]
            pattern = PATTERN_LIBRARY[family][variant]
            result = validate_sequence_with_health(pattern['sequence'])
            health = result.health_metrics.overall_health if result.passed and result.health_metrics else 0.0
            
            print(f"\n{family}.{variant}:")
            print(f"  Use case: {pattern['use_case']}")
            print(f"  Sequence length: {len(pattern['sequence'])} operators")
            print(f"  Expected health: {pattern['health_range'][0]:.2f}-{pattern['health_range'][1]:.2f}")
            print(f"  Actual health: {health:.3f}")
            print(f"  Pattern: {pattern['detected_pattern']}")
    
    print()
    print("=" * 70)
    print("Use get_pattern(family, variant) to access specific patterns")
    print("Use search_patterns() to find patterns by criteria")
    print("=" * 70)

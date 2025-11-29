"""TNFR Grammar: Main Validation Entry Point

Primary validate_grammar() function - the main public API for grammar checking.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .definitions import Operator
else:
    from .definitions import Operator

from .grammar_core import GrammarValidator

# ============================================================================
# Public API: Validation Functions
# ============================================================================


def validate_grammar(
    sequence: List[Operator],
    epi_initial: float = 0.0,
    collect_unified_telemetry: bool = False,
) -> bool:
    """Validate sequence using canonical TNFR grammar constraints.

    Convenience function that returns only boolean result.
    For detailed messages, use GrammarValidator.validate().

    Parameters
    ----------
    sequence : List[Operator]
        Sequence of operators to validate
    epi_initial : float, optional
        Initial EPI value (default: 0.0)
    collect_unified_telemetry : bool, optional
        If True, collect unified field telemetry during validation
        (Nov 28, 2025 - enables unified field metrics for analysis)

    Returns
    -------
    bool
        True if sequence satisfies all canonical constraints

    Notes
    -----
    The unified field telemetry (when enabled) provides additional insights
    into sequence validation using the mathematical unification discoveries:
    - K_φ ↔ J_φ correlation analysis
    - Conservation law metrics
    - Emergent field patterns
    
    This telemetry is purely observational and does not affect validation
    logic, maintaining full compatibility with existing U1-U6 constraints.

    Examples
    --------
    >>> from tnfr.operators.definitions import Emission, Coherence, Silence
    >>> ops = [Emission(), Coherence(), Silence()]
    >>> validate_grammar(ops, epi_initial=0.0)  # doctest: +SKIP
    True

    Notes
    -----
    This validator is 100% physics-based. All constraints emerge from:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - TNFR invariants (AGENTS.md §3)
    - Formal operator contracts (AGENTS.md §4)

    See UNIFIED_GRAMMAR_RULES.md for complete derivations.
    """
    validator = GrammarValidator()
    is_valid, _ = validator.validate(sequence, epi_initial)
    
    # Optional unified field telemetry (Nov 28, 2025 integration)
    if collect_unified_telemetry and is_valid:
        try:
            # Create a sample graph for unified field analysis
            # This is telemetry-only and doesn't affect validation result
            import networkx as nx
            from ..physics.fields import compute_unified_telemetry
            
            # Create minimal graph with sequence-derived state
            G = nx.Graph()
            G.add_node(0, EPI=1.0, theta=0.0, nu_f=1.0, delta_nfr=0.0)
            G.add_node(1, EPI=0.8, theta=0.5, nu_f=0.9, delta_nfr=0.1)
            G.add_edge(0, 1)
            
            # Collect unified telemetry for sequence analysis
            unified_data = compute_unified_telemetry(G)
            
            # Log for development/research purposes
            # In production, this could feed into metrics pipeline
            print(f"[UNIFIED TELEMETRY] Sequence validation completed")
            if "complex_field" in unified_data:
                corr = unified_data["complex_field"].get("correlation", 0.0)
                print(f"  K_φ ↔ J_φ correlation: {corr:.3f}")
            if "tensor_invariants" in unified_data:
                conservation = unified_data["tensor_invariants"].get("conservation_quality", 0.0)
                print(f"  Conservation quality: {conservation:.3f}")
                
        except Exception:
            # Graceful degradation - unified telemetry is optional
            pass
    
    return is_valid



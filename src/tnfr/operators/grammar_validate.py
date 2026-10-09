"""TNFR Grammar: Main Validation Entry Point

Primary validate_grammar() function - the main public API for grammar checking.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from .definitions import Operator
from .grammar_core import GrammarValidator

# ============================================================================
# Public API: Validation Functions
# ============================================================================


def validate_grammar(
    sequence: list[Operator],
    epi_initial: float = 0.0,
) -> bool:
    """Validate sequence using canonical TNFR grammar constraints.

    Convenience function that returns only boolean result.
    For detailed messages, use GrammarValidator.validate().

    Parameters
    ----------
    sequence : list[Operator]
        Sequence of operators to validate
    epi_initial : float, optional
        Initial EPI value (default: 0.0)

    Returns
    -------
    bool
        True if the available canonical word constraints pass; no live phase
        gate, operator postcondition or U6 trajectory observation is certified

    Examples
    --------
    >>> from tnfr.operators.definitions import Emission, Coherence, Silence
    >>> ops = [Emission(), Coherence(), Silence()]
    >>> validate_grammar(ops, epi_initial=0.0)  # doctest: +SKIP
    True

    Notes
    -----
    The nodal equation motivates the supported operator contracts and policies.
    Their mathematical limits are in DIAGNOSTIC_AND_GRAMMAR_SCOPE.md.
    This function receives no graph and produces no field observations.
    Use ``diagnose_network(actual_graph)`` or the field readers on explicitly
    supplied state for diagnostics; word acceptance cannot supply that state.
    """
    validator = GrammarValidator()
    is_valid, _ = validator.validate(sequence, epi_initial)
    return is_valid

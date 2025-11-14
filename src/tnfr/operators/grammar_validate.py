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

    Returns
    -------
    bool
        True if sequence satisfies all canonical constraints

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
    return is_valid



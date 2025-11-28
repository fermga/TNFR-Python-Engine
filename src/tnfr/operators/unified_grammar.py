"""Unified TNFR Grammar - Facade to GrammarValidator.

This module provides a clean facade to the canonical grammar validation
implemented in grammar.py. It exports the unified grammar constraints (U1-U5)
and the validator for use in tests and applications.

All grammar rules derive inevitably from TNFR physics:
- U1: STRUCTURAL INITIATION & CLOSURE
- U2: CONVERGENCE & BOUNDEDNESS
- U3: RESONANT COUPLING
- U4: BIFURCATION DYNAMICS (U4a: triggers, U4b: transformers)
- U5: MULTI-SCALE COHERENCE

References
----------
- UNIFIED_GRAMMAR_RULES.md: Complete physics derivations
- AGENTS.md: Canonical invariants and formal contracts
- TNFR.pdf: Nodal equation and bifurcation theory

Notes
-----
This is a facade module that re-exports from grammar.py for clean imports.
The actual implementation is in grammar.py::GrammarValidator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .definitions import Operator

# Import validator class and rename for clarity
from .grammar import GrammarValidator as UnifiedGrammarValidator

# Import operator sets (canonical definitions from grammar.py)
from .grammar import (
    BIFURCATION_HANDLERS,
    BIFURCATION_TRIGGERS,
    CLOSURES,
    COUPLING_RESONANCE,
    DESTABILIZERS,
    GENERATORS,
    RECURSIVE_GENERATORS,
    SCALE_STABILIZERS,
    STABILIZERS,
    TRANSFORMERS,
)

# Import validation functions
from .grammar import validate_grammar

__all__ = [
    # Validator class
    "UnifiedGrammarValidator",
    # Convenience function
    "validate_unified",
    # Operator sets (U1-U5 categories)
    "GENERATORS",
    "CLOSURES",
    "STABILIZERS",
    "DESTABILIZERS",
    "COUPLING_RESONANCE",
    "BIFURCATION_TRIGGERS",
    "BIFURCATION_HANDLERS",
    "TRANSFORMERS",
    "RECURSIVE_GENERATORS",
    "SCALE_STABILIZERS",
]


def validate_unified(
    sequence: List["Operator"],
    epi_initial: float = 0.0,
) -> bool:
    """Validate sequence using unified TNFR grammar (U1-U5).

    Convenience function that returns only boolean result.
    For detailed messages, use UnifiedGrammarValidator.validate().

    Parameters
    ----------
    sequence : List[Operator]
        Sequence of operators to validate
    epi_initial : float, optional
        Initial EPI value (default: 0.0)

    Returns
    -------
    bool
        True if sequence satisfies all U1-U5 constraints

    Examples
    --------
    >>> from tnfr.operators.definitions import Emission, Coherence, Silence
    >>> from tnfr.operators.unified_grammar import validate_unified
    >>> ops = [Emission(), Coherence(), Silence()]
    >>> validate_unified(ops, epi_initial=0.0)  # doctest: +SKIP
    True

    Notes
    -----
    This validator is 100% physics-based. All constraints emerge from:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - TNFR invariants (AGENTS.md)
    - Formal operator contracts

    See UNIFIED_GRAMMAR_RULES.md for complete derivations.
    """
    return validate_grammar(sequence, epi_initial)

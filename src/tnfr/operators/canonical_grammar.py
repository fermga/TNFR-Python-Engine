"""TNFR Canonical Grammar (LEGACY - Use unified_grammar instead).

⚠️  DEPRECATION NOTICE
=======================
This module is maintained for backward compatibility only.
All functionality has been consolidated into unified_grammar.py.

This module will be removed in version 8.0.0.
Please migrate to:

    from tnfr.operators.unified_grammar import UnifiedGrammarValidator

Old Canonical Grammar (RC1-RC4) → Unified Grammar (U1-U4)
----------------------------------------------------------
RC1: Initialization      → U1a: Initiation
RC2: Convergence         → U2:  CONVERGENCE & BOUNDEDNESS  
RC3: Phase Verification  → U3:  RESONANT COUPLING
RC4: Bifurcation Limits  → U4a: Bifurcation Triggers

Additional unified constraints:
U1b: Closure (restores physical basis of removed RNC1)
U4b: Transformer Context (graduated destabilization)

Historical Context
------------------
Previous versions of this module implemented RC1-RC4 constraints derived from
TNFR physics. These have been consolidated into the unified grammar system
(U1-U4) which provides:
- Single source of truth (no duplication)
- Complete physics derivations in docstrings
- Additional constraints based on structural physics
- Better alignment with TNFR.pdf and AGENTS.md

References
----------
- UNIFIED_GRAMMAR_RULES.md: Complete derivations and mappings
- unified_grammar.py: Canonical implementation
- AGENTS.md: Invariants and formal contracts

Warnings
--------
Importing from this module will emit DeprecationWarning.
All new code should use unified_grammar.py instead.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List

# Import all exports from unified grammar
from .unified_grammar import (
    UnifiedGrammarValidator,
    validate_unified,
    GENERATORS,
    CLOSURES,
    STABILIZERS,
    DESTABILIZERS,
    COUPLING_RESONANCE,
    BIFURCATION_TRIGGERS,
    BIFURCATION_HANDLERS,
    TRANSFORMERS,
)

if TYPE_CHECKING:
    from ..types import NodeId
    from .definitions import Operator

# Emit warning on module import
warnings.warn(
    "canonical_grammar is deprecated and will be removed in version 8.0.0. "
    "Use unified_grammar instead. See UNIFIED_GRAMMAR_RULES.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    # Legacy validators (deprecated)
    'CanonicalGrammarValidator',
    'validate_canonical_only',
    'validate_with_conventions',
    # Re-exports from unified grammar
    'UnifiedGrammarValidator',
    'validate_unified',
    # Legacy operator set names
    'GENERATOR_OPS',
    'STABILIZER_OPS',
    'DESTABILIZER_OPS',
    'COUPLING_OPS',
    'BIFURCATION_TRIGGER_OPS',
    'BIFURCATION_HANDLER_OPS',
    # New in unified
    'CLOSURE_OPS',
    'TRANSFORMER_OPS',
]


# =============================================================================
# MIGRATION GUIDE: canonical_grammar → unified_grammar
# =============================================================================
#
# Old (canonical_grammar.py):              | New (unified_grammar.py):
# -----------------------------------------|----------------------------------
# from tnfr.operators.canonical_grammar    | from tnfr.operators.unified_grammar
# import CanonicalGrammarValidator         | import UnifiedGrammarValidator
#                                          |
# valid, msgs = CanonicalGrammarValidator  | valid, msgs = UnifiedGrammarValidator
#     .validate(seq)                       |     .validate(seq)
# -----------------------------------------|----------------------------------
# validate_canonical_only(seq)             | validate_unified(seq)
# -----------------------------------------|----------------------------------
# CanonicalGrammarValidator                | UnifiedGrammarValidator
#     .validate_initialization(seq)        |     .validate_initiation(seq)
# -----------------------------------------|----------------------------------
# CanonicalGrammarValidator                | UnifiedGrammarValidator
#     .validate_convergence(seq)           |     .validate_convergence(seq)
# -----------------------------------------|----------------------------------
# CanonicalGrammarValidator                | UnifiedGrammarValidator
#     .validate_phase_compatibility(seq)   |     .validate_resonant_coupling(seq)
# -----------------------------------------|----------------------------------
# CanonicalGrammarValidator                | UnifiedGrammarValidator
#     .validate_bifurcation_limits(seq)    |     .validate_bifurcation_triggers(seq)
# -----------------------------------------|----------------------------------
#
# Benefits of migration:
# 1. Single source of truth (no duplication)
# 2. Complete physics derivations in docstrings
# 3. Detailed validation messages
# 4. Additional constraints (U1b: Closure, U4b: Transformer Context)
# 5. Better alignment with TNFR.pdf and AGENTS.md
#
# See UNIFIED_GRAMMAR_RULES.md for complete migration guide.
# =============================================================================


# Legacy operator set names - map to unified names
GENERATOR_OPS = GENERATORS  # RC1 → U1a
STABILIZER_OPS = STABILIZERS  # RC2 → U2
DESTABILIZER_OPS = DESTABILIZERS  # RC2 → U2
COUPLING_OPS = COUPLING_RESONANCE  # RC3 → U3
BIFURCATION_TRIGGER_OPS = BIFURCATION_TRIGGERS  # RC4 → U4a
BIFURCATION_HANDLER_OPS = BIFURCATION_HANDLERS  # RC4 → U4a

# NEW in unified grammar (not in original RC system)
CLOSURE_OPS = CLOSURES  # U1b
TRANSFORMER_OPS = TRANSFORMERS  # U4b


class CanonicalGrammarValidator:
    """DEPRECATED: Use UnifiedGrammarValidator from unified_grammar instead.
    
    Legacy validator maintained for backward compatibility.
    All methods delegate to UnifiedGrammarValidator.
    """

    @staticmethod
    def validate_initialization(
        sequence: List[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, str]:
        """DEPRECATED: Use UnifiedGrammarValidator.validate_initiation().
        
        RC1 → U1a: Initiation requirement.
        """
        warnings.warn(
            "CanonicalGrammarValidator.validate_initialization is deprecated. "
            "Use UnifiedGrammarValidator.validate_initiation().",
            DeprecationWarning,
            stacklevel=2
        )
        return UnifiedGrammarValidator.validate_initiation(sequence, epi_initial)

    @staticmethod
    def validate_convergence(sequence: List[Operator]) -> tuple[bool, str]:
        """DEPRECATED: Use UnifiedGrammarValidator.validate_convergence().
        
        RC2 → U2: Convergence & Boundedness.
        """
        warnings.warn(
            "CanonicalGrammarValidator.validate_convergence is deprecated. "
            "Use UnifiedGrammarValidator.validate_convergence().",
            DeprecationWarning,
            stacklevel=2
        )
        return UnifiedGrammarValidator.validate_convergence(sequence)

    @staticmethod
    def validate_phase_compatibility(sequence: List[Operator]) -> tuple[bool, str]:
        """DEPRECATED: Use UnifiedGrammarValidator.validate_resonant_coupling().
        
        RC3 → U3: Resonant Coupling (requires phase check per Invariant #5).
        """
        warnings.warn(
            "CanonicalGrammarValidator.validate_phase_compatibility is deprecated. "
            "Use UnifiedGrammarValidator.validate_resonant_coupling().",
            DeprecationWarning,
            stacklevel=2
        )
        return UnifiedGrammarValidator.validate_resonant_coupling(sequence)

    @staticmethod
    def validate_bifurcation_limits(sequence: List[Operator]) -> tuple[bool, str]:
        """DEPRECATED: Use UnifiedGrammarValidator.validate_bifurcation_triggers().
        
        RC4 → U4a: Bifurcation Triggers need handlers.
        """
        warnings.warn(
            "CanonicalGrammarValidator.validate_bifurcation_limits is deprecated. "
            "Use UnifiedGrammarValidator.validate_bifurcation_triggers().",
            DeprecationWarning,
            stacklevel=2
        )
        return UnifiedGrammarValidator.validate_bifurcation_triggers(sequence)

    @classmethod
    def validate(
        cls,
        sequence: List[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, List[str]]:
        """DEPRECATED: Use UnifiedGrammarValidator.validate().
        
        Validates using canonical constraints (RC1-RC4 → U1-U4).
        
        Note: This now uses the unified grammar which includes U1b (Closure)
        and U4b (Transformer Context) constraints that were not in the
        original RC1-RC4 system.
        """
        warnings.warn(
            "CanonicalGrammarValidator.validate is deprecated. "
            "Use UnifiedGrammarValidator.validate().",
            DeprecationWarning,
            stacklevel=2
        )
        return UnifiedGrammarValidator.validate(sequence, epi_initial)


def validate_canonical_only(
    sequence: List[Operator],
    epi_initial: float = 0.0,
) -> bool:
    """DEPRECATED: Use validate_unified() from unified_grammar.
    
    Validates using canonical constraints (RC1-RC4 → U1-U4).
    
    Note: This now uses the unified grammar which includes U1b (Closure)
    and U4b (Transformer Context) constraints that were not in the
    original RC1-RC4 system.
    """
    warnings.warn(
        "validate_canonical_only is deprecated. Use validate_unified() from unified_grammar.",
        DeprecationWarning,
        stacklevel=2
    )
    return validate_unified(sequence, epi_initial)


def validate_with_conventions(
    sequence: List[Operator],
    epi_initial: float = 0.0,
) -> tuple[bool, List[str]]:
    """DEPRECATED: Use UnifiedGrammarValidator.validate() from unified_grammar.
    
    Historical Note: This function previously enforced RNC1 (terminator
    convention), but that was removed. Now delegates to unified grammar.
    
    Returns detailed validation messages using unified U1-U4 constraints.
    """
    warnings.warn(
        "validate_with_conventions is deprecated. "
        "Use UnifiedGrammarValidator.validate() from unified_grammar.",
        DeprecationWarning,
        stacklevel=2
    )
    return UnifiedGrammarValidator.validate(sequence, epi_initial)

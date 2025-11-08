"""Canonical grammar validator - Pure physics from nodal equation.

This module implements grammar validation that emerges EXCLUSIVELY from
the nodal equation ∂EPI/∂t = νf · ΔNFR(t), without organizational conventions.

Canonical Rules (Inevitable from Physics)
------------------------------------------
RC1: Initialization - If EPI=0, sequence must start with generator
     Reason: ∂EPI/∂t undefined at EPI=0
     
RC2: Convergence - If sequence has destabilizers, must include stabilizer
     Reason: ∫νf·ΔNFR dt must converge (convergence theorem)

Non-Canonical Rules (Organizational Conventions)
-------------------------------------------------
RNC1: Termination - Sequence must end with specific terminators
      Reason: Code organization, NOT physics
      
RNC2: Specific composition restrictions
      Reason: High-level semantics, NOT nodal equation

References
----------
See CANONICAL_GRAMMAR_DERIVATION.md for complete mathematical derivation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..types import NodeId
    from .definitions import Operator

__all__ = [
    "CanonicalGrammarValidator",
    "validate_canonical_only",
    "validate_with_conventions",
]


# Canonical operator sets (derived from physics)
GENERATORS = frozenset({'emission', 'transition', 'recursivity'})
STABILIZERS = frozenset({'coherence', 'self_organization'})
DESTABILIZERS = frozenset({
    'dissonance',
    'mutation',
    'expansion',
    # Note: Some operators have destabilizing components
})

# Conventional terminators (NOT canonical - organizational only)
CONVENTIONAL_TERMINATORS = frozenset({
    'silence',
    'dissonance',
    'transition',
    'recursivity',
})


class CanonicalGrammarValidator:
    """Validates sequences using ONLY physics-derived rules.
    
    This validator implements RC1 and RC2, which emerge inevitably from
    the nodal equation ∂EPI/∂t = νf · ΔNFR(t). It does NOT enforce
    organizational conventions like required terminators.
    
    Use this for testing algebraic properties where you want to validate
    pure physics without implementation conventions.
    """
    
    @staticmethod
    def validate_initialization(
        sequence: List[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, str]:
        """Validate RC1: Initialization requirement.
        
        Physical basis: If EPI=0, then ∂EPI/∂t is undefined or zero.
        Cannot evolve structure that doesn't exist.
        
        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate
        epi_initial : float, optional
            Initial EPI value (default: 0.0)
        
        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        if epi_initial > 0.0:
            # Already initialized, no generator required
            return True, "EPI>0: initialization not required"
        
        if not sequence:
            return False, "RC1 violated: Empty sequence with EPI=0"
        
        first_op = getattr(sequence[0], 'canonical_name', sequence[0].name.lower())
        
        if first_op not in GENERATORS:
            return (
                False,
                f"RC1 violated: EPI=0 requires generator (got '{first_op}'). "
                f"Valid: {sorted(GENERATORS)}"
            )
        
        return True, f"RC1 satisfied: starts with generator '{first_op}'"
    
    @staticmethod
    def validate_convergence(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate RC2: Convergence requirement.
        
        Physical basis: Without stabilizers, ∫νf·ΔNFR dt → ∞ (diverges).
        Convergence theorem requires negative feedback.
        
        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate
        
        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        # Check if sequence contains destabilizers
        destabilizers_present = [
            getattr(op, 'canonical_name', op.name.lower())
            for op in sequence
            if getattr(op, 'canonical_name', op.name.lower()) in DESTABILIZERS
        ]
        
        if not destabilizers_present:
            # No destabilizers = no divergence risk
            return True, "RC2 not applicable: no destabilizers present"
        
        # Check for stabilizers
        stabilizers_present = [
            getattr(op, 'canonical_name', op.name.lower())
            for op in sequence
            if getattr(op, 'canonical_name', op.name.lower()) in STABILIZERS
        ]
        
        if not stabilizers_present:
            return (
                False,
                f"RC2 violated: destabilizers {destabilizers_present} present "
                f"without stabilizer. Integral ∫νf·ΔNFR dt may diverge. "
                f"Add: {sorted(STABILIZERS)}"
            )
        
        return (
            True,
            f"RC2 satisfied: stabilizers {stabilizers_present} "
            f"bound destabilizers {destabilizers_present}"
        )
    
    @classmethod
    def validate(
        cls,
        sequence: List[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, List[str]]:
        """Validate sequence using ONLY canonical rules (RC1, RC2).
        
        This validates pure physics without organizational conventions.
        
        Parameters
        ----------
        sequence : List[Operator]
            Sequence to validate
        epi_initial : float, optional
            Initial EPI value (default: 0.0)
        
        Returns
        -------
        tuple[bool, List[str]]
            (is_valid, messages)
            is_valid: True if all canonical rules satisfied
            messages: List of validation messages
        """
        messages = []
        all_valid = True
        
        # RC1: Initialization
        valid_init, msg_init = cls.validate_initialization(sequence, epi_initial)
        messages.append(f"RC1: {msg_init}")
        all_valid = all_valid and valid_init
        
        # RC2: Convergence
        valid_conv, msg_conv = cls.validate_convergence(sequence)
        messages.append(f"RC2: {msg_conv}")
        all_valid = all_valid and valid_conv
        
        return all_valid, messages


def validate_canonical_only(
    sequence: List[Operator],
    epi_initial: float = 0.0,
) -> bool:
    """Validate sequence using only physics-derived canonical rules.
    
    This function validates ONLY:
    - RC1: Initialization (if EPI=0, use generator)
    - RC2: Convergence (if destabilizers, use stabilizer)
    
    It does NOT validate:
    - Terminator requirements (organizational convention)
    - Specific composition restrictions (high-level semantics)
    
    Use this when testing algebraic properties where you want pure physics
    validation without implementation conventions.
    
    Parameters
    ----------
    sequence : List[Operator]
        Sequence of operators to validate
    epi_initial : float, optional
        Initial EPI value (default: 0.0)
    
    Returns
    -------
    bool
        True if sequence satisfies canonical physics rules
    
    Examples
    --------
    >>> from tnfr.operators.definitions import Emission, Coherence
    >>> ops = [Emission(), Coherence()]
    >>> validate_canonical_only(ops, epi_initial=0.0)  # doctest: +SKIP
    True
    
    Notes
    -----
    This validator is 100% physics-based. All rules emerge inevitably from
    the nodal equation ∂EPI/∂t = νf · ΔNFR(t).
    """
    is_valid, messages = CanonicalGrammarValidator.validate(sequence, epi_initial)
    return is_valid


def validate_with_conventions(
    sequence: List[Operator],
    epi_initial: float = 0.0,
) -> tuple[bool, List[str]]:
    """Validate sequence with both canonical rules and conventions.
    
    This validates:
    - RC1, RC2: Canonical physics rules
    - RNC1: Terminator convention (organizational, NOT physics)
    
    Parameters
    ----------
    sequence : List[Operator]
        Sequence to validate
    epi_initial : float, optional
        Initial EPI value (default: 0.0)
    
    Returns
    -------
    tuple[bool, List[str]]
        (is_valid, messages)
    """
    messages = []
    all_valid = True
    
    # First validate canonical rules
    valid_canonical, canonical_msgs = CanonicalGrammarValidator.validate(
        sequence, epi_initial
    )
    messages.extend(canonical_msgs)
    all_valid = all_valid and valid_canonical
    
    # Then check conventions
    if sequence:
        last_op = getattr(sequence[-1], 'canonical_name', sequence[-1].name.lower())
        if last_op not in CONVENTIONAL_TERMINATORS:
            messages.append(
                f"RNC1 (CONVENTION): Sequence should end with terminator "
                f"(got '{last_op}'). Valid: {sorted(CONVENTIONAL_TERMINATORS)}. "
                f"Note: This is organizational convention, NOT physics."
            )
            all_valid = False
        else:
            messages.append(
                f"RNC1 (CONVENTION): ends with terminator '{last_op}' "
                f"(organizational convention satisfied)"
            )
    
    return all_valid, messages

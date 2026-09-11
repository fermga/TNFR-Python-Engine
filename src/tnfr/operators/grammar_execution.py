"""Immutable context for execution of a canonically validated operator word.

Future handlers can satisfy U4a only inside such a word. The live U2 debt,
U3 phase gate and U4b stable-base requirements remain independent checks.
No context is written to the graph, and completed prefixes are not rolled back
if a later operation fails before its promised handler executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..errors import TNFRValueError
from .grammar_core import GrammarValidator
from .grammar_types import BIFURCATION_HANDLERS, glyph_function_name


@dataclass(frozen=True, init=False)
class ValidatedSequence:
    """Validate actual operator instances once, preserving their metadata."""

    names: tuple[str, ...]

    def __init__(self, operators: Iterable[Any], *, context: Mapping[str, Any] | None = None):
        sequence = list(operators)
        initialized = bool(context and context.get("initial_epi_nonzero", False))
        valid, messages = GrammarValidator().validate(
            sequence, epi_initial=1.0 if initialized else 0.0,
        )
        if not valid:
            raise TNFRValueError(
                "Invalid canonical sequence: " + "; ".join(messages),
                context={"sequence": [operator.name for operator in sequence]},
            )
        object.__setattr__(self, "names", tuple(operator.name for operator in sequence))

    def step(self, index: int) -> ValidatedSequenceStep:
        """Bind the current operator and the remaining suffix of this word."""
        if not 0 <= index < len(self.names):
            raise IndexError("sequence step is outside the validated word")
        return ValidatedSequenceStep(self, index)


@dataclass(frozen=True)
class ValidatedSequenceStep:
    """A step in a validated word; no past operators are assumed executed."""

    sequence: ValidatedSequence
    index: int

    def has_future_handler(self, candidate: Any) -> bool:
        """Read U4a's future handler only for this step's actual candidate."""
        if (
            not isinstance(self.sequence, ValidatedSequence)
            or not 0 <= self.index < len(self.sequence.names)
            or glyph_function_name(candidate) != self.sequence.names[self.index]
        ):
            raise TNFRValueError("Operator does not match its validated sequence context")
        return any(
            name in BIFURCATION_HANDLERS
            for name in self.sequence.names[self.index + 1:]
        )

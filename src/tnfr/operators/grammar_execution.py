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
from .grammar_context import _normalize_sequence_context
from .grammar_core import GrammarValidator
from .grammar_types import BIFURCATION_HANDLERS, glyph_function_name


@dataclass(frozen=True, init=False)
class ValidatedSequence:
    """Validate actual instances once under core or explicit legacy policy.

    Instance metadata remains visible to U5. Diagnostic-only fragment waivers
    do not grant executable admission, and live operator gates remain separate.
    """

    names: tuple[str, ...]
    compatibility_profile: str

    def __init__(
        self, operators: Iterable[Any], *, context: Mapping[str, Any] | None = None
    ):
        sequence = list(operators)
        context = _normalize_sequence_context(context)
        if context.get("diagnostic", False) and [
            operator.name for operator in sequence
        ] == ["dissonance", "mutation"]:
            raise TNFRValueError(
                "The diagnostic OZ/ZHIR waiver does not admit an executable word"
            )
        initialized = bool(context and context.get("initial_epi_nonzero", False))
        checks = GrammarValidator().validate_checks(
            sequence,
            epi_initial=1.0 if initialized else 0.0,
        )
        if any(check.blocking and not check.passed for check in checks):
            raise TNFRValueError(
                "Invalid canonical sequence: "
                + "; ".join(
                    check.legacy_message()
                    for check in checks
                    if check.blocking and not check.passed
                ),
                context={"sequence": [operator.name for operator in sequence]},
            )
        profile = context["compatibility_profile"]
        if profile == "legacy":
            from .grammar_patterns import _legacy_word_preference_error

            error = _legacy_word_preference_error(
                [operator.name for operator in sequence]
            )
            if error is not None:
                raise TNFRValueError(
                    "Invalid legacy sequence preference: " + error[1],
                    context={"sequence": [operator.name for operator in sequence]},
                )
        object.__setattr__(self, "names", tuple(operator.name for operator in sequence))
        object.__setattr__(self, "compatibility_profile", profile)

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
            raise TNFRValueError(
                "Operator does not match its validated sequence context"
            )
        return any(
            name in BIFURCATION_HANDLERS
            for name in self.sequence.names[self.index + 1 :]
        )

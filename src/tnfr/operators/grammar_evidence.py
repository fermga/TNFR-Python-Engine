"""Derived finite EPI evidence, separate from operator-word grammar policy.

The existing event executor owns the observations, represented affine maps,
common metric and exact gain product.  This adapter makes those results usable
by grammar reporting without assigning stability to an operator label or a
debt counter.  It does not choose a subsequent operator or admit a candidate.

Only an intact, canonical ``OperatorEventExecutionResult`` is accepted.  Its
own composition is used; a detached composition, mapping or caller-supplied
claim cannot replace the execution evidence.  The result describes the finite
historical trace, without identifying it with any current live graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, NamedTuple

from ..utils._structural_signature import proof_stamps_are_identical

if TYPE_CHECKING:
    from .event_runtime import OperatorEventExecutionResult

__all__ = (
    "StructuralGrammarEvidence",
    "assess_structural_grammar_evidence",
)

_PROOF_VERSION = "structural_grammar_execution_evidence_v1"
_SCOPE = (
    "Finite historical evidence for the represented affine EPI maps of one "
    "sealed event execution, in its declared common disagreement metric. "
    "The exact upper bound determines whether nonincrease or contraction is "
    "certified. A bound above one does not prove expansion. This is not "
    "prospective grammar admission, an autonomous selection law, a global "
    "executable-map gain, full multichannel stability, a trajectory theorem "
    "for the structural tetrad (potential, phase gradient, phase curvature "
    "and coherence length), solver accuracy, a "
    "future or repeated-schedule guarantee, or a binding to a current graph."
)


def _require_execution(execution: Any) -> None:
    """Delegate nested proof validation to the sole execution owner."""
    if execution is None:
        return
    from .event_runtime import OperatorEventExecutionResult

    if type(execution) is not OperatorEventExecutionResult:
        raise TypeError(
            "execution must be a canonical OperatorEventExecutionResult or None"
        )
    if not OperatorEventExecutionResult._proof_fields_are_intact(execution):
        raise ValueError("execution evidence proof fields are not intact")


def _execution_link(execution: Any) -> tuple[Any, ...]:
    """Retain the exact source identity and its existing proof stamp."""
    if execution is None:
        return (_PROOF_VERSION, None)
    return (
        _PROOF_VERSION,
        id(execution),
        object.__getattribute__(execution, "_proof_stamp"),
    )


class _Assessment(NamedTuple):
    available: bool
    gain: Fraction | None
    nonincrease: bool | None
    contraction: bool | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuralGrammarEvidence:
    """A source-bound readout; certification fields are never caller inputs.

    ``available`` means that the exact represented-map gain is available, even
    when it exceeds one.  A false nonincrease/contraction result only says that
    the supplied upper bound does not establish that claim.  Missing evidence
    instead produces ``None`` for both conclusions.
    """

    execution: OperatorEventExecutionResult | None = field(repr=False)
    _proof_stamp: tuple[Any, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_execution(self.execution)
        expected = _execution_link(self.execution)
        try:
            retained = object.__getattribute__(self, "_proof_stamp")
        except AttributeError:
            object.__setattr__(self, "_proof_stamp", expected)
        else:
            if not proof_stamps_are_identical(retained, expected):
                raise ValueError("grammar evidence source link is not intact")

    def _proof_fields_are_intact(self) -> bool:
        if type(self) is not StructuralGrammarEvidence:
            return False
        try:
            execution = object.__getattribute__(self, "execution")
            _require_execution(execution)
            return proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                _execution_link(execution),
            )
        except BaseException:
            return False

    def _assessment(self) -> _Assessment:
        if not StructuralGrammarEvidence._proof_fields_are_intact(self):
            return _Assessment(
                False, None, None, None, ("execution_evidence_integrity_failed",)
            )
        if self.execution is None:
            return _Assessment(
                False, None, None, None, ("execution_evidence_not_supplied",)
            )
        composition = self.execution.represented_epi_schedule_composition
        if composition is None:
            return _Assessment(
                False,
                None,
                None,
                None,
                ("represented_composition_not_recorded",),
            )
        if not composition.represented_affine_composition_gain_certified:
            return _Assessment(
                False, None, None, None, composition.failed_conditions
            )
        gain = composition.exact_energy_gain_upper_bound
        # The execution owner already checks this identity. Keep the adapter's
        # boundary explicit without introducing a second gain calculation.
        if type(gain) is not Fraction or gain < 0:
            return _Assessment(
                False, None, None, None, ("exact_represented_gain_unavailable",)
            )
        return _Assessment(True, gain, gain <= 1, gain < 1, ())

    @property
    def available(self) -> bool:
        return self._assessment().available

    @property
    def exact_energy_gain_upper_bound(self) -> Fraction | None:
        return self._assessment().gain

    @property
    def represented_nonincrease_certified(self) -> bool | None:
        return self._assessment().nonincrease

    @property
    def represented_contraction_certified(self) -> bool | None:
        return self._assessment().contraction

    @property
    def unavailable_reasons(self) -> tuple[str, ...]:
        return self._assessment().reasons

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def prospective_admission_certified(self) -> bool:
        return False

    @property
    def runtime_schedule_global_gain_certified(self) -> bool:
        return False

    @property
    def full_multichannel_stability_certified(self) -> bool:
        return False

    @property
    def tetrad_trajectory_certified(self) -> bool:
        return False

    @property
    def future_or_repeated_schedule_stability_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-ready reporting data, not transferable proof evidence."""
        facts = self._assessment()
        return {
            "available": facts.available,
            "exact_energy_gain_upper_bound": (
                None if facts.gain is None else str(facts.gain)
            ),
            "represented_nonincrease_certified": facts.nonincrease,
            "represented_contraction_certified": facts.contraction,
            "unavailable_reasons": list(facts.reasons),
            "prospective_admission_certified": False,
            "runtime_schedule_global_gain_certified": False,
            "full_multichannel_stability_certified": False,
            "tetrad_trajectory_certified": False,
            "future_or_repeated_schedule_stability_certified": False,
            "solver_accuracy_certified": False,
            "scope": _SCOPE,
        }

    def as_record(self) -> dict[str, Any]:
        """Return the same reporting projection as :meth:`as_dict`."""
        return self.as_dict()


def assess_structural_grammar_evidence(
    execution: OperatorEventExecutionResult | None,
) -> StructuralGrammarEvidence:
    """Assess a complete observed execution without deriving a new controller.

    ``None`` or a genuine execution lacking complete supported evidence is
    reported as unavailable.  A malformed or altered proof source is rejected.
    Fixed debt, history windows and operator priorities do not enter the result.
    """
    return StructuralGrammarEvidence(execution)

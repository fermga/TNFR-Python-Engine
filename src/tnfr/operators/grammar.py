"""Canonical grammar and sequence validation for structural operators."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Iterable, MutableMapping
from copy import deepcopy
from enum import Enum
from importlib import resources
from json import JSONDecodeError
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from ..node import NodeProtocol
    from .health_analyzer import SequenceHealthMetrics

from ..compat.dataclass import dataclass
from ..config.operator_names import (
    CONTRACTION,
    COHERENCE,
    COUPLING,
    DISSONANCE,
    EMISSION,
    INTERMEDIATE_OPERATORS,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SELF_ORGANIZATION_CLOSURES,
    SILENCE,
    TRANSITION,
    EXPANSION,
    VALID_END_OPERATORS,
    VALID_START_OPERATORS,
    DESTABILIZERS,
    TRANSFORMERS,
    BIFURCATION_WINDOW,
    canonical_operator_name,
    operator_display_name,
)
from ..constants import DEFAULTS, get_param
from ..types import Glyph, NodeId, TNFRGraph
from ..validation import ValidationOutcome, rules as _rules
from ..validation.soft_filters import soft_grammar_filters
from ..utils import get_logger
from .registry import OPERATORS

try:  # pragma: no cover - optional dependency import
    from jsonschema import Draft7Validator
    from jsonschema import exceptions as _jsonschema_exceptions
except Exception:  # pragma: no cover - jsonschema optional
    Draft7Validator = None  # type: ignore[assignment]
    _jsonschema_exceptions = None  # type: ignore[assignment]

__all__ = [
    "GrammarContext",
    "GrammarConfigurationError",
    "StructuralGrammarError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "MutationWithoutDissonanceError",
    "IncompatibleSequenceError",
    "IncompleteEncapsulationError",
    "MissingStabilizerError",
    "SequenceSyntaxError",
    "SequenceValidationResult",
    "StructuralPattern",
    "record_grammar_violation",
    "_gram_state",
    "apply_glyph_with_grammar",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "parse_sequence",
    "validate_sequence",
    "validate_sequence_with_health",
    "FUNCTION_TO_GLYPH",
    "GLYPH_TO_FUNCTION",
    "STRUCTURAL_FREQUENCIES",
    "FREQUENCY_TRANSITIONS",
    "glyph_function_name",
    "function_name_to_glyph",
    "validate_frequency_transition",
    "REGENERATORS",
    "MIN_CYCLE_LENGTH",
    "MAX_CYCLE_LENGTH",
    "CycleType",
]

logger = get_logger(__name__)

_SCHEMA_LOAD_ERROR: str | None = None
_SOFT_VALIDATOR: Draft7Validator | None = None
_CANON_VALIDATOR: Draft7Validator | None = None

# Structural mapping ---------------------------------------------------------

# NOTE: Glyph comments describe the structural function following TNFR canon
# so downstream modules can reason in terms of operator semantics rather than
# internal glyph codes. This mapping is the single source of truth for
# translating between glyph identifiers and the structural operators defined
# in :mod:`tnfr.config.operator_names`.
GLYPH_TO_FUNCTION: dict[Glyph, str] = {
    Glyph.AL: EMISSION,  # Emission — seeds coherence outward from the node.
    Glyph.EN: RECEPTION,  # Reception — anchors inbound energy into the EPI.
    Glyph.IL: COHERENCE,  # Coherence — compresses ΔNFR drift to stabilise C(t).
    Glyph.OZ: DISSONANCE,  # Dissonance — injects controlled tension for probes.
    Glyph.UM: COUPLING,  # Coupling — synchronises bidirectional coherence links.
    Glyph.RA: RESONANCE,  # Resonance — amplifies aligned structural frequency.
    Glyph.SHA: SILENCE,  # Silence — suspends reorganisation while preserving form.
    Glyph.VAL: EXPANSION,  # Expansion — dilates the structure to explore volume.
    Glyph.NUL: CONTRACTION,  # Contraction — concentrates trajectories into the core.
    Glyph.THOL: SELF_ORGANIZATION,  # Self-organisation — spawns autonomous cascades.
    Glyph.ZHIR: MUTATION,  # Mutation — pivots the node across structural thresholds.
    Glyph.NAV: TRANSITION,  # Transition — guides controlled regime hand-offs.
    Glyph.REMESH: RECURSIVITY,  # Recursivity — echoes patterns across nested EPIs.
}

FUNCTION_TO_GLYPH: dict[str, Glyph] = {
    name: glyph for glyph, name in GLYPH_TO_FUNCTION.items()
}


def glyph_function_name(
    val: Glyph | str | None, *, default: str | None = None
) -> str | None:
    """Return the structural operator name corresponding to ``val``.

    Parameters
    ----------
    val:
        Glyph enumeration, glyph code or structural operator identifier.
    default:
        Value returned when ``val`` cannot be translated.
    """

    if val is None:
        return default
    if isinstance(val, Glyph):
        return GLYPH_TO_FUNCTION.get(val, default)
    try:
        glyph = Glyph(str(val))
    except (TypeError, ValueError):
        canon = canonical_operator_name(str(val))
        return canon if canon in FUNCTION_TO_GLYPH else default
    else:
        return GLYPH_TO_FUNCTION.get(glyph, default)


def function_name_to_glyph(
    val: str | Glyph | None, *, default: Glyph | None = None
) -> Glyph | None:
    """Return the :class:`Glyph` associated with the structural identifier ``val``."""

    if val is None:
        return default
    if isinstance(val, Glyph):
        return val
    try:
        return Glyph(str(val))
    except (TypeError, ValueError):
        canon = canonical_operator_name(str(val))
        return FUNCTION_TO_GLYPH.get(canon, default)


@dataclass(slots=True)
class GrammarContext:
    """Shared context for grammar helpers."""

    G: TNFRGraph
    cfg_soft: dict[str, Any]
    cfg_canon: dict[str, Any]
    norms: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_grammar_configs(self)

    @classmethod
    def from_graph(cls, G: TNFRGraph) -> "GrammarContext":
        """Create a context pulling graph overrides or isolated defaults.

        When a graph omits grammar configuration, copies of
        :data:`tnfr.constants.DEFAULTS` are materialised so each context can
        mutate its settings without leaking state into other graphs.
        """

        def _copy_default(key: str) -> dict[str, Any]:
            default = DEFAULTS.get(key, {})
            if isinstance(default, Mapping):
                return {k: deepcopy(v) for k, v in default.items()}
            if isinstance(default, Iterable) and not isinstance(default, (str, bytes)):
                return dict(default)
            return {}

        cfg_soft = G.graph.get("GRAMMAR")
        if cfg_soft is None:
            cfg_soft = _copy_default("GRAMMAR")

        cfg_canon = G.graph.get("GRAMMAR_CANON")
        if cfg_canon is None:
            cfg_canon = _copy_default("GRAMMAR_CANON")

        return cls(
            G=G,
            cfg_soft=cfg_soft,
            cfg_canon=cfg_canon,
            norms=G.graph.get("_sel_norms") or {},
        )


def _gram_state(nd: dict[str, Any]) -> dict[str, Any]:
    return nd.setdefault("_GRAM", {"thol_open": False, "thol_len": 0})


class GrammarConfigurationError(ValueError):
    """Raised when grammar configuration violates the bundled JSON schema."""

    __slots__ = ("section", "messages", "details")

    def __init__(
        self,
        section: str,
        messages: Sequence[str],
        *,
        details: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        msg = "; ".join(messages)
        super().__init__(f"invalid {section} configuration: {msg}")
        self.section = section
        self.messages = tuple(messages)
        self.details = tuple(details or ())


def _validation_env_flag() -> bool | None:
    flag = os.environ.get("TNFR_GRAMMAR_VALIDATE")
    if flag is None:
        return None
    normalised = flag.strip().lower()
    if normalised in {"0", "false", "off", "no"}:
        return False
    if normalised in {"1", "true", "on", "yes"}:
        return True
    return None


def _ensure_schema_validators() -> (
    tuple[Draft7Validator | None, Draft7Validator | None] | None
):
    global _SCHEMA_LOAD_ERROR, _SOFT_VALIDATOR, _CANON_VALIDATOR
    if _SOFT_VALIDATOR is not None or _CANON_VALIDATOR is not None:
        return _SOFT_VALIDATOR, _CANON_VALIDATOR
    if _SCHEMA_LOAD_ERROR is not None:
        return None
    if Draft7Validator is None:
        _SCHEMA_LOAD_ERROR = "jsonschema package is not installed"
        return None
    try:
        schema_text = (
            resources.files("tnfr.schemas").joinpath("grammar.json").read_text("utf-8")
        )
    except FileNotFoundError:
        _SCHEMA_LOAD_ERROR = "grammar schema resource not found"
        return None
    try:
        schema = json.loads(schema_text)
    except JSONDecodeError as exc:
        _SCHEMA_LOAD_ERROR = f"unable to decode grammar schema: {exc}"
        return None
    definitions = schema.get("definitions")
    if not isinstance(definitions, Mapping):
        _SCHEMA_LOAD_ERROR = "grammar schema missing definitions"
        return None
    soft_schema = definitions.get("cfg_soft")
    canon_schema = definitions.get("cfg_canon")
    if soft_schema is None or canon_schema is None:
        _SCHEMA_LOAD_ERROR = "grammar schema missing cfg_soft/cfg_canon definitions"
        return None
    if not isinstance(soft_schema, Mapping) or not isinstance(canon_schema, Mapping):
        _SCHEMA_LOAD_ERROR = "grammar schema definitions must be objects"
        return None
    soft_payload = dict(soft_schema)
    canon_payload = dict(canon_schema)
    soft_payload.setdefault("definitions", definitions)
    canon_payload.setdefault("definitions", definitions)
    try:
        _SOFT_VALIDATOR = Draft7Validator(soft_payload)
        _CANON_VALIDATOR = Draft7Validator(canon_payload)
    except Exception as exc:  # pragma: no cover - delegated to jsonschema
        if _jsonschema_exceptions is not None and isinstance(
            exc, _jsonschema_exceptions.SchemaError
        ):
            _SCHEMA_LOAD_ERROR = f"invalid grammar schema: {exc.message}"
        else:
            _SCHEMA_LOAD_ERROR = f"unable to construct grammar validators: {exc}"
        _SOFT_VALIDATOR = None
        _CANON_VALIDATOR = None
        return None
    return _SOFT_VALIDATOR, _CANON_VALIDATOR


def _format_validation_error(err: Any) -> str:
    path = "".join(
        f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path
    )
    path = path.lstrip(".") or "<root>"
    return f"{path}: {err.message}"


def _validate_grammar_configs(ctx: GrammarContext) -> None:
    flag = _validation_env_flag()
    if flag is False:
        logger.debug("TNFR_GRAMMAR_VALIDATE=0 → skipping grammar schema validation")
        return
    validators = _ensure_schema_validators()
    if validators is None:
        if flag is True:
            reason = _SCHEMA_LOAD_ERROR or "validators unavailable"
            raise RuntimeError(
                "grammar schema validation requested but unavailable: " f"{reason}"
            )
        if _SCHEMA_LOAD_ERROR is not None:
            logger.debug("Skipping grammar schema validation: %s", _SCHEMA_LOAD_ERROR)
        return
    soft_validator, canon_validator = validators
    issues: list[tuple[str, str]] = []
    if soft_validator is not None:
        for err in soft_validator.iter_errors(ctx.cfg_soft):  # type: ignore[union-attr]
            issues.append(("cfg_soft", _format_validation_error(err)))

    canon_cfg: Mapping[str, Any] | None
    if isinstance(ctx.cfg_canon, Mapping):
        canon_cfg = ctx.cfg_canon
    else:
        canon_cfg = None
        issues.append(
            (
                "cfg_canon",
                "GRAMMAR_CANON must be a mapping"
                f" (received {type(ctx.cfg_canon).__name__})",
            )
        )

    if canon_validator is not None:
        for err in canon_validator.iter_errors(ctx.cfg_canon):  # type: ignore[union-attr]
            issues.append(("cfg_canon", _format_validation_error(err)))

    cfg_for_lengths: Mapping[str, Any] = canon_cfg or {}
    min_len = cfg_for_lengths.get("thol_min_len")
    max_len = cfg_for_lengths.get("thol_max_len")
    if isinstance(min_len, int) and isinstance(max_len, int) and max_len < min_len:
        issues.append(
            (
                "cfg_canon",
                "thol_max_len must be greater than or equal to thol_min_len",
            )
        )
    if not issues:
        return
    section = issues[0][0]
    raise GrammarConfigurationError(
        section,
        [msg for _, msg in issues],
        details=issues,
    )


class SequenceSyntaxError(ValueError):
    """Raised when an operator sequence violates the canonical grammar."""

    __slots__ = ("index", "token", "message")

    def __init__(self, index: int, token: object, message: str) -> None:
        super().__init__(message)
        self.index = index
        self.token = token
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - delegated to ValueError
        return self.message


class SequenceValidationResult(ValidationOutcome[tuple[str, ...]]):
    """Structured report emitted by :func:`validate_sequence`."""

    __slots__ = ("tokens", "canonical_tokens", "message", "metadata", "error", "health_metrics")

    def __init__(
        self,
        *,
        tokens: Sequence[str],
        canonical_tokens: Sequence[str],
        passed: bool,
        message: str,
        metadata: Mapping[str, object],
        error: SequenceSyntaxError | None = None,
        health_metrics: Optional["SequenceHealthMetrics"] = None,
    ) -> None:
        tokens_tuple = tuple(tokens)
        canonical_tuple = tuple(canonical_tokens)
        metadata_dict = dict(metadata)
        metadata_view = MappingProxyType(metadata_dict)
        summary: dict[str, object] = {
            "message": message,
            "passed": passed,
            "tokens": tokens_tuple,
        }
        if metadata_dict:
            summary["metadata"] = metadata_view
        if error is not None:
            summary["error"] = {"index": error.index, "token": error.token}
        super().__init__(
            subject=canonical_tuple,
            passed=passed,
            summary=summary,
            artifacts={"canonical_tokens": canonical_tuple},
        )
        self.tokens = tokens_tuple
        self.canonical_tokens = canonical_tuple
        self.message = message
        self.metadata = metadata_view
        self.error = error
        self.health_metrics = health_metrics


_CANONICAL_START = tuple(sorted(VALID_START_OPERATORS))
_CANONICAL_INTERMEDIATE = tuple(sorted(INTERMEDIATE_OPERATORS))
_CANONICAL_END = tuple(sorted(VALID_END_OPERATORS))


def _format_token_group(tokens: Sequence[str]) -> str:
    return ", ".join(operator_display_name(token) for token in tokens)


class _SequenceAutomaton:
    __slots__ = (
        "_canonical",
        "_found_reception",
        "_found_coherence",
        "_seen_intermediate",
        "_open_thol",
        "_unknown_tokens",
        "_found_dissonance",
        "_found_stabilizer",
        "_detected_pattern",
        "_bifurcation_context",
    )

    def __init__(self) -> None:
        self._canonical: list[str] = []
        self._found_reception = False
        self._found_coherence = False
        self._seen_intermediate = False
        self._open_thol = False
        self._unknown_tokens: list[tuple[int, str]] = []
        self._found_dissonance = False  # Track OZ for R4
        self._found_stabilizer = False  # Track IL or THOL for R2
        self._detected_pattern: StructuralPattern = StructuralPattern.UNKNOWN
        # Use deque with maxlen for efficient O(1) append and automatic size management
        self._bifurcation_context: deque[tuple[str, int]] = deque(maxlen=BIFURCATION_WINDOW)

    def run(self, names: Sequence[str]) -> None:
        if not names:
            raise SequenceSyntaxError(index=-1, token=None, message="empty sequence")
        for index, token in enumerate(names):
            self._consume(token, index)
        self._finalize(names)

    def _consume(self, token: str, index: int) -> None:
        if not isinstance(token, str):
            raise SequenceSyntaxError(
                index=index, token=token, message="tokens must be str"
            )
        canonical = canonical_operator_name(token)
        self._canonical.append(canonical)
        if canonical not in OPERATORS:
            self._unknown_tokens.append((index, token))

        # R1: Validate start (already implemented)
        if index == 0:
            if canonical not in VALID_START_OPERATORS:
                expected = _format_token_group(_CANONICAL_START)
                raise SequenceSyntaxError(
                    index=index, token=token, message=f"must start with {expected}"
                )

        # Track state for various rules
        if canonical == RECEPTION and not self._found_reception:
            self._found_reception = True
        elif (
            self._found_reception
            and canonical == COHERENCE
            and not self._found_coherence
        ):
            self._found_coherence = True
        elif self._found_coherence and canonical in INTERMEDIATE_OPERATORS:
            self._seen_intermediate = True

        # R2: Track stabilizers (IL or THOL)
        if canonical in {COHERENCE, SELF_ORGANIZATION}:
            self._found_stabilizer = True

        # R4: Track destabilizers for bifurcation control (expanded from just OZ)
        if canonical in DESTABILIZERS:
            # Deque automatically maintains window size (BIFURCATION_WINDOW)
            self._bifurcation_context.append((canonical, index))
        
        # R4: Validate transformers (ZHIR/THOL) require recent destabilizer
        if canonical in TRANSFORMERS:
            if not self._has_recent_destabilizer(index):
                destab_names = ", ".join(operator_display_name(d) for d in sorted(DESTABILIZERS))
                raise SequenceSyntaxError(
                    index=index,
                    token=token,
                    message=(
                        f"{operator_display_name(canonical)} requires destabilizer "
                        f"({destab_names}) in previous {BIFURCATION_WINDOW} operators"
                    ),
                )
        
        # Legacy: Keep tracking dissonance for backward compatibility with existing code
        if canonical == DISSONANCE:
            self._found_dissonance = True

        # Track THOL state
        if canonical == SELF_ORGANIZATION:
            self._open_thol = True
        elif self._open_thol and canonical in SELF_ORGANIZATION_CLOSURES:
            self._open_thol = False

        # Validate sequential compatibility if not first token
        # Only validate if both prev and current are known operators
        if index > 0 and canonical in OPERATORS:
            self._validate_transition(
                self._canonical[index - 1], canonical, index, token
            )

    def _validate_transition(
        self, prev: str, curr: str, index: int, token: str
    ) -> None:
        """Validate that curr is compatible after prev using graduated compatibility levels.

        Uses the graduated compatibility matrix to provide more nuanced feedback:
        - EXCELLENT and GOOD: Pass silently
        - CAUTION: Log warning but allow transition
        - AVOID: Raise SequenceSyntaxError

        Note: Import is done inside the method to avoid circular dependency between
        grammar and compatibility modules.
        """
        from ..validation.compatibility import CompatibilityLevel, get_compatibility_level

        # Only validate if prev is also a known operator
        if prev not in OPERATORS:
            return

        # R5: Validate structural frequency transitions (warnings only)
        freq_valid, freq_msg = validate_frequency_transition(prev, curr)
        if not freq_valid:
            logger.warning(
                "R5 frequency transition warning at position %d: %s",
                index,
                freq_msg,
            )

        # Get graduated compatibility level
        level = get_compatibility_level(prev, curr)

        if level == CompatibilityLevel.AVOID:
            raise SequenceSyntaxError(
                index=index,
                token=token,
                message=f"{operator_display_name(curr)} incompatible after {operator_display_name(prev)}",
            )
        elif level == CompatibilityLevel.CAUTION:
            logger.warning(
                "Caution: %s → %s transition at position %d requires careful context validation",
                operator_display_name(prev),
                operator_display_name(curr),
                index,
            )

    def _has_recent_destabilizer(self, current_index: int) -> bool:
        """Check if a destabilizer exists within the bifurcation window.
        
        Parameters
        ----------
        current_index : int
            Current position in the sequence
            
        Returns
        -------
        bool
            True if a destabilizer was found in the previous BIFURCATION_WINDOW operators
            
        Notes
        -----
        The `_bifurcation_context` deque stores only destabilizers (not all operators),
        so we must check that stored destabilizers are within BIFURCATION_WINDOW steps
        of the current index.
        """
        window_start = max(0, current_index - BIFURCATION_WINDOW)
        return any(
            window_start <= dest_index < current_index
            for _, dest_index in self._bifurcation_context
        )

    def _finalize(self, names: Sequence[str]) -> None:
        if self._unknown_tokens:
            ordered = ", ".join(sorted({token for _, token in self._unknown_tokens}))
            first_index, first_token = self._unknown_tokens[0]
            raise SequenceSyntaxError(
                index=first_index,
                token=first_token,
                message=f"unknown tokens: {ordered}",
            )

        # R2: Must contain at least one stabilizer
        if not self._found_stabilizer:
            raise SequenceSyntaxError(
                index=-1,
                token=None,
                message=f"missing required stabilizer ({operator_display_name(COHERENCE)} or {operator_display_name(SELF_ORGANIZATION)})",
            )

        if not (self._found_reception and self._found_coherence):
            raise SequenceSyntaxError(
                index=-1,
                token=None,
                message=f"missing {RECEPTION}→{COHERENCE} segment",
            )
        # NOTE: Intermediate operator check removed - COHERENCE provides sufficient
        # structural transformation for valid TNFR sequences. The requirement for
        # explicit DISSONANCE/COUPLING/RESONANCE was overly restrictive.

        # R3: Must end with terminator
        if self._canonical[-1] not in VALID_END_OPERATORS:
            cierre = _format_token_group(_CANONICAL_END)
            raise SequenceSyntaxError(
                index=len(names) - 1,
                token=names[-1],
                message=f"sequence must end with {cierre}",
            )

        if self._open_thol:
            raise SequenceSyntaxError(
                index=len(names) - 1,
                token=names[-1],
                message=f"{operator_display_name(SELF_ORGANIZATION)} block without closure",
            )

        # Detect structural pattern
        self._detected_pattern = self._detect_pattern()
        
        # R5: Validate regenerative cycles if pattern is REGENERATIVE
        if self._detected_pattern == StructuralPattern.REGENERATIVE:
            self._validate_regenerative_cycle()

    def _validate_regenerative_cycle(self) -> None:
        """Validate regenerative cycle structural requirements (R5).
        
        Uses CycleDetector to ensure the sequence meets minimum standards
        for self-sustaining regenerative behavior.
        """
        from .cycle_detection import CycleDetector
        
        detector = CycleDetector()
        analysis = detector.analyze_full_cycle(self._canonical)
        
        if not analysis.is_valid_regenerative:
            # Build informative error message
            reason_messages = {
                "too_short": f"cycle too short (minimum {MIN_CYCLE_LENGTH} operators)",
                "too_long": f"cycle too long (maximum {MAX_CYCLE_LENGTH} operators)",
                "no_regenerator": f"no regenerator found ({', '.join(REGENERATORS)})",
                "no_stabilization": "missing stabilizers before and/or after regenerator",
                "low_health_score": f"cycle health score {analysis.health_score:.2f} below threshold {CycleDetector.MIN_HEALTH_SCORE}",
                "no_valid_cycle": "no valid regenerative cycle structure detected",
            }
            
            message = reason_messages.get(
                analysis.reason,
                f"invalid regenerative cycle: {analysis.reason}"
            )
            
            raise SequenceSyntaxError(
                index=-1,
                token=None,
                message=f"R5 regenerative cycle validation failed: {message}",
            )

    def _detect_pattern(self) -> StructuralPattern:
        """Detect the structural pattern type of the sequence.
        
        Uses advanced pattern detection to identify domain-specific and
        meta-patterns, falling back to basic pattern detection when needed.
        """
        from .patterns import AdvancedPatternDetector
        
        detector = AdvancedPatternDetector()
        return detector.detect_pattern(self._canonical)

    @property
    def canonical(self) -> tuple[str, ...]:
        return tuple(self._canonical)

    @property
    def detected_pattern(self) -> StructuralPattern:
        return self._detected_pattern

    def metadata(self) -> Mapping[str, object]:
        return {
            "has_reception": self._found_reception,
            "has_coherence": self._found_coherence,
            "has_intermediate": self._seen_intermediate,
            "open_thol": self._open_thol,
            "unknown_tokens": frozenset(token for _, token in self._unknown_tokens),
            "has_dissonance": self._found_dissonance,
            "has_stabilizer": self._found_stabilizer,
            "detected_pattern": self._detected_pattern.value,
        }


_MISSING = object()


class StructuralGrammarError(RuntimeError):
    """Raised when canonical grammar invariants are violated."""

    __slots__ = (
        "rule",
        "candidate",
        "window",
        "threshold",
        "order",
        "context",
        "message",
    )

    def __init__(
        self,
        *,
        rule: str,
        candidate: str,
        message: str,
        window: int | None = None,
        threshold: float | None = None,
        order: Sequence[str] | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.rule = rule
        self.candidate = candidate
        self.message = message
        self.window = window
        self.threshold = threshold
        self.order = tuple(order) if order is not None else None
        self.context: dict[str, object] = dict(context or {})

    def attach_context(self, **context: object) -> "StructuralGrammarError":
        """Return ``self`` after updating contextual metadata."""

        for key, value in context.items():
            if value is not None:
                self.context[key] = value
        return self

    def to_payload(self) -> dict[str, object]:
        """Return a structured payload suitable for telemetry sinks."""

        payload: dict[str, object] = {
            "rule": self.rule,
            "candidate": self.candidate,
            "message": self.message,
        }
        if self.window is not None:
            payload["window"] = self.window
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        if self.order is not None:
            payload["order"] = self.order
        if self.context:
            payload["context"] = dict(self.context)
        return payload


class RepeatWindowError(StructuralGrammarError):
    """Repeated glyph within the configured hysteresis window."""


class MutationPreconditionError(StructuralGrammarError):
    """Mutation attempted without satisfying canonical dissonance requirements."""


class TholClosureError(StructuralGrammarError):
    """THOL block reached closure conditions without a canonical terminator."""


class TransitionCompatibilityError(StructuralGrammarError):
    """Transition attempted that violates canonical compatibility tables."""


class MutationWithoutDissonanceError(StructuralGrammarError):
    """ZHIR applied without OZ precedent (R4 violation)."""


class IncompatibleSequenceError(StructuralGrammarError):
    """Sequence violates canonical compatibility rules."""


class IncompleteEncapsulationError(StructuralGrammarError):
    """THOL without valid internal sequence."""


class MissingStabilizerError(StructuralGrammarError):
    """Sequence missing required stabilizer (IL or THOL) - R2 violation."""


class StructuralPattern(Enum):
    """Unified typology of structural patterns in operator sequences.
    
    All patterns are equal - no hierarchies or priorities. Detection uses
    pattern signatures to identify the best match based on specificity
    and coverage of the sequence.
    """

    # Fundamental structural patterns
    LINEAR = "linear"  # Simple progression: AL → IL → RA → SHA
    HIERARCHICAL = "hierarchical"  # Self-organization: THOL[...]
    FRACTAL = "fractal"  # Recursive structure: NAV → IL → UM → NAV
    CYCLIC = "cyclic"  # Regenerative loops: multiple NAV
    BIFURCATED = "bifurcated"  # Branching: OZ → {ZHIR | NUL}
    
    # Domain-specific applied patterns
    THERAPEUTIC = "therapeutic"  # Healing: EN→AL→IL→OZ→THOL→IL
    EDUCATIONAL = "educational"  # Learning: EN→AL→IL→VAL→OZ→ZHIR
    ORGANIZATIONAL = "organizational"  # Evolution: NAV→AL→EN→UM→RA→OZ→THOL
    CREATIVE = "creative"  # Artistic: SHA→AL→VAL→OZ→ZHIR→THOL
    REGENERATIVE = "regenerative"  # Self-sustaining: IL→RA→VAL→SHA→NAV→AL
    
    # Compositional patterns
    BOOTSTRAP = "bootstrap"  # Initialization: AL→UM→IL
    EXPLORE = "explore"  # Exploration: OZ→ZHIR→IL
    STABILIZE = "stabilize"  # Consolidation: *→IL→{SHA|RA}
    RESONATE = "resonate"  # Amplification: RA→UM→RA
    COMPRESS = "compress"  # Simplification: NUL→IL→SHA
    
    # Adaptive learning patterns (AL + T'HOL canonical sequences)
    BASIC_LEARNING = "basic_learning"  # Simple learning: AL→EN→IL
    DEEP_LEARNING = "deep_learning"  # Deep learning with crisis: AL→EN→OZ→THOL→IL
    EXPLORATORY_LEARNING = "exploratory_learning"  # Exploration learning: AL→OZ→THOL→RA→IL
    CONSOLIDATION_CYCLE = "consolidation_cycle"  # Memory consolidation: IL→SHA→REMESH
    ADAPTIVE_MUTATION = "adaptive_mutation"  # Transformative learning: EN→OZ→THOL→ZHIR→NAV→IL
    
    # Structural complexity
    COMPLEX = "complex"  # Multiple patterns combined
    MINIMAL = "minimal"  # Single or very few operators
    UNKNOWN = "unknown"  # Unclassified


# Structural frequency matrix (νf): Hz_str categories per operator
# R5: HARMONIC_FREQUENCIES validation (now active)
# Maps each operator to its structural frequency level following TNFR canonical theory.
# - high: Operators that initiate, amplify, concentrate, or pivot structure
# - medium: Operators that capture, stabilize, couple, or organize  
# - zero: Operators that suspend reorganization while preserving form
STRUCTURAL_FREQUENCIES: dict[str, str] = {
    EMISSION: "high",  # AL: initiation/reorganization
    RECEPTION: "medium",  # EN: structural capture
    COHERENCE: "medium",  # IL: stabilization
    DISSONANCE: "high",  # OZ: tension
    COUPLING: "medium",  # UM: coupling
    RESONANCE: "high",  # RA: amplification
    SILENCE: "zero",  # SHA: pause (suspended)
    EXPANSION: "medium",  # VAL: volumetric exploration
    CONTRACTION: "high",  # NUL: concentration
    SELF_ORGANIZATION: "medium",  # THOL: autonomous cascades
    MUTATION: "high",  # ZHIR: threshold pivot
    TRANSITION: "medium",  # NAV: controlled hand-off
    RECURSIVITY: "medium",  # REMESH: fractal echo
}

# Frequency compatibility: operators with harmonic frequencies can transition
# Valid transitions preserve structural coherence:
# - high ↔ medium: High energy can stabilize or stabilized can amplify
# - medium ↔ zero: Stabilized can pause or resume from pause  
# - high ↔ high: High energy can chain directly
# Invalid: zero → high (cannot jump from pause to high without intermediate stabilization)
FREQUENCY_TRANSITIONS: dict[str, set[str]] = {
    "high": {"high", "medium"},
    "medium": {"high", "medium", "zero"},
    "zero": {"medium"},  # Must transition through medium before high
}


def validate_frequency_transition(prev_operator: str, next_operator: str) -> tuple[bool, str]:
    """Validate structural frequency transition between consecutive operators (R5 rule).

    Parameters
    ----------
    prev_operator : str
        Previous operator in canonical form (e.g., "emission", "coherence").
    next_operator : str
        Next operator in canonical form (e.g., "dissonance", "resonance").

    Returns
    -------
    tuple[bool, str]
        (is_valid, message) where is_valid indicates if transition respects frequency
        harmonics, and message provides context when invalid.

    Notes
    -----
    Structural frequency transitions follow TNFR canonical rules:
    - High ↔ Medium: Bidirectional, natural energy exchange
    - Medium ↔ Zero: Stabilization can pause, pause can resume
    - High ↔ High: High energy operators can chain directly
    - **Invalid**: Zero → High without Medium intermediary

    This validation generates warnings (not errors) to maintain flexibility while
    alerting users to potentially incoherent structural transitions.
    """
    # Get frequency levels for both operators
    prev_freq = STRUCTURAL_FREQUENCIES.get(prev_operator)
    next_freq = STRUCTURAL_FREQUENCIES.get(next_operator)

    # If either operator is unknown, skip validation (compatibility handles this)
    if prev_freq is None or next_freq is None:
        return True, ""

    # Check if transition is allowed
    allowed_targets = FREQUENCY_TRANSITIONS.get(prev_freq, set())
    if next_freq not in allowed_targets:
        prev_display = operator_display_name(prev_operator)
        next_display = operator_display_name(next_operator)
        return (
            False,
            f"Incoherent frequency transition: {prev_display} ({prev_freq}) → {next_display} ({next_freq}). "
            f"Valid transitions from {prev_freq}: {', '.join(sorted(allowed_targets))}",
        )

    return True, ""


# R5: Regenerative cycle constants and types (from cycle_detection module)
from .cycle_detection import (
    REGENERATORS,
    MIN_CYCLE_LENGTH,
    MAX_CYCLE_LENGTH,
    CycleType,
)


def _record_grammar_violation(
    G: TNFRGraph, node: NodeId, error: StructuralGrammarError, *, stage: str
) -> None:
    """Store ``error`` telemetry on ``G`` and emit a structured log."""

    telemetry = G.graph.setdefault("telemetry", {})
    if not isinstance(telemetry, MutableMapping):
        telemetry = {}
        G.graph["telemetry"] = telemetry
    channel = telemetry.setdefault("grammar_errors", [])
    if not isinstance(channel, list):
        channel = []
        telemetry["grammar_errors"] = channel
    payload = {"node": node, "stage": stage, **error.to_payload()}
    channel.append(payload)
    logger.warning(
        "grammar violation on node %s during %s: %s",
        node,
        stage,
        payload,
        exc_info=error,
    )


def record_grammar_violation(
    G: TNFRGraph, node: NodeId, error: StructuralGrammarError, *, stage: str
) -> None:
    """Public shim for recording grammar violations with telemetry hooks intact."""

    _record_grammar_violation(G, node, error, stage=stage)


def _analyse_sequence(names: Iterable[str]) -> SequenceValidationResult:
    names_list = list(names)
    automaton = _SequenceAutomaton()
    try:
        automaton.run(names_list)
        error: SequenceSyntaxError | None = None
        message = "ok"
    except SequenceSyntaxError as exc:
        error = exc
        message = exc.message
    return SequenceValidationResult(
        tokens=tuple(names_list),
        canonical_tokens=automaton.canonical,
        passed=error is None,
        message=message,
        metadata=automaton.metadata(),
        error=error,
    )


def validate_sequence(
    names: Iterable[str] | object = _MISSING, **kwargs: object
) -> ValidationOutcome[tuple[str, ...]]:
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"validate_sequence() got unexpected keyword argument(s): {unexpected}"
        )
    if names is _MISSING:
        raise TypeError("validate_sequence() missing required argument: 'names'")
    return _analyse_sequence(names)


def validate_sequence_with_health(
    names: Iterable[str] | object = _MISSING, **kwargs: object
) -> SequenceValidationResult:
    """Validate a sequence and include structural health metrics.
    
    This is an enhanced version of :func:`validate_sequence` that additionally
    computes comprehensive health metrics for valid sequences. The metrics
    provide quantitative assessment of sequence quality: coherence, balance,
    sustainability, and efficiency.
    
    Parameters
    ----------
    names : Iterable[str]
        Sequence of operator names to validate and analyze.
    
    Returns
    -------
    SequenceValidationResult
        Validation result with ``health_metrics`` field populated for valid sequences.
    
    Examples
    --------
    >>> from tnfr.operators.grammar import validate_sequence_with_health
    >>> result = validate_sequence_with_health(["emission", "reception", "coherence", "silence"])
    >>> result.passed
    True
    >>> result.health_metrics.overall_health > 0.7
    True
    >>> result.health_metrics.dominant_pattern
    'activation'
    
    Notes
    -----
    Health metrics are only computed for sequences that pass validation.
    Invalid sequences will have ``health_metrics=None``.
    """
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"validate_sequence_with_health() got unexpected keyword argument(s): {unexpected}"
        )
    if names is _MISSING:
        raise TypeError("validate_sequence_with_health() missing required argument: 'names'")
    
    # First, perform standard validation
    result = _analyse_sequence(names)
    
    # If validation passed, compute health metrics
    if result.passed:
        from .health_analyzer import SequenceHealthAnalyzer
        
        analyzer = SequenceHealthAnalyzer()
        health = analyzer.analyze_health(list(names))
        
        # Create new result with health metrics
        result = SequenceValidationResult(
            tokens=result.tokens,
            canonical_tokens=result.canonical_tokens,
            passed=result.passed,
            message=result.message,
            metadata=result.metadata,
            error=result.error,
            health_metrics=health,
        )
    
    return result


def parse_sequence(names: Iterable[str]) -> SequenceValidationResult:
    result = _analyse_sequence(names)
    if not result.passed:
        if result.error is not None:
            raise result.error
        raise SequenceSyntaxError(index=-1, token=None, message=result.message)
    return result


def enforce_canonical_grammar(
    G: TNFRGraph,
    n: NodeId,
    cand: Glyph | str,
    ctx: Optional[GrammarContext] = None,
) -> Glyph | str:
    """Validate ``cand`` against canonical grammar rules and preserve structural identifiers.

    When callers provide textual operator identifiers (for example ``"emission"``), the
    returned value mirrors the canonical structural token instead of the raw glyph code.
    This keeps downstream traces aligned with TNFR operator semantics while still
    permitting glyph inputs for internal workflows.
    """
    if ctx is None:
        ctx = GrammarContext.from_graph(G)

    nd = ctx.G.nodes[n]
    st = _gram_state(nd)

    raw_cand = cand
    cand = _rules.coerce_glyph(cand)
    input_was_str = isinstance(raw_cand, str)

    if not isinstance(cand, Glyph):
        translated = function_name_to_glyph(raw_cand if input_was_str else cand)
        if translated is None and cand is not raw_cand:
            translated = function_name_to_glyph(cand)
        if translated is not None:
            cand = translated

    from ..validation.compatibility import CANON_COMPAT

    if not isinstance(cand, Glyph) or cand not in CANON_COMPAT:
        return raw_cand if input_was_str else cand

    cand = soft_grammar_filters(ctx, n, cand)
    cand = _rules._check_oz_to_zhir(ctx, n, cand)
    cand = _rules._check_thol_closure(ctx, n, cand, st)
    cand = _rules._check_compatibility(ctx, n, cand)

    coerced_final = _rules.coerce_glyph(cand)
    if input_was_str:
        resolved = glyph_function_name(coerced_final)
        if resolved is None:
            resolved = glyph_function_name(cand)
        if resolved is not None:
            return resolved
        if isinstance(coerced_final, Glyph):
            return coerced_final.value
        return str(cand)
    return coerced_final if isinstance(coerced_final, Glyph) else cand


def on_applied_glyph(G: TNFRGraph, n: NodeId, applied: Glyph | str) -> None:
    nd = G.nodes[n]
    st = _gram_state(nd)
    glyph = function_name_to_glyph(applied)

    if glyph is Glyph.THOL:
        st["thol_open"] = True
        st["thol_len"] = 0
    elif glyph in (Glyph.SHA, Glyph.NUL):
        st["thol_open"] = False
        st["thol_len"] = 0


def apply_glyph_with_grammar(
    G: TNFRGraph,
    nodes: Optional[Iterable[NodeId | "NodeProtocol"]],
    glyph: Glyph | str,
    window: Optional[int] = None,
) -> None:
    # Convert Glyph enum instances to string values early for consistent processing
    if isinstance(glyph, Glyph):
        glyph = glyph.value

    if window is None:
        window = get_param(G, "GLYPH_HYSTERESIS_WINDOW")

    g_str = glyph if isinstance(glyph, str) else str(glyph)
    iter_nodes = G.nodes() if nodes is None else nodes
    ctx = GrammarContext.from_graph(G)
    from . import apply_glyph as _apply_glyph

    for node_ref in iter_nodes:
        node_id = node_ref.n if hasattr(node_ref, "n") else node_ref
        try:
            g_eff = enforce_canonical_grammar(G, node_id, g_str, ctx)
        except StructuralGrammarError as err:
            nd = G.nodes[node_id]

            def _structural_history(value: Glyph | str) -> str:
                default = value.value if isinstance(value, Glyph) else str(value)
                resolved = glyph_function_name(value, default=default)
                return default if resolved is None else resolved

            history = tuple(
                _structural_history(item) for item in nd.get("glyph_history", ())
            )
            err.attach_context(node=node_id, stage="apply_glyph", history=history)
            _record_grammar_violation(G, node_id, err, stage="apply_glyph")
            raise
        telemetry_token = g_eff
        glyph_obj = g_eff if isinstance(g_eff, Glyph) else function_name_to_glyph(g_eff)
        if glyph_obj is None:
            coerced = _rules.coerce_glyph(g_eff)
            glyph_obj = coerced if isinstance(coerced, Glyph) else None
        if glyph_obj is None:
            raise ValueError(f"unknown glyph: {g_eff}")

        _apply_glyph(G, node_id, glyph_obj, window=window)
        on_applied_glyph(G, node_id, telemetry_token)

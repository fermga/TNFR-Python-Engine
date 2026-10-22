"""TNFR Grammar: Sequence Pattern Recognition

Sequence validation, parsing, pattern recognition, and optimization helpers.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability

The exact [dissonance, mutation] probe has a retained explicit diagnostic
waiver when existing form is declared. Its metadata records that canonical
word validation was waived. This grants no live operator preconditions,
threshold crossing, stability or fragmentation conclusion. Ordinary words
reuse the core's causal debt and transformer-history checks; adjacency and
THOL-terminal restrictions are additional compatibility policies.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ..config.operator_names import (
    CANONICAL_OPERATOR_NAMES,
    INTERMEDIATE_OPERATORS,
    SELF_ORGANIZATION,
    SELF_ORGANIZATION_CLOSURES,
)
from ..types import Glyph
from ..validation.compatibility import CompatibilityLevel, get_compatibility_level
from .grammar_context import _normalize_sequence_context
from .grammar_types import (
    STABILIZERS,
    SequenceSyntaxError,
    SequenceValidationResult,
    StructuralPattern,
)

# --- State classification thresholds for IL sequence suggestion ---
_INACTIVE_EPI_THRESHOLD = 0.1  # EPI below this → node inactive
_HIGH_DNFR_THRESHOLD = 0.8  # ΔNFR above this → high pressure
_MODERATE_DNFR_LOW = 0.3  # lower bound of moderate ΔNFR range
_MODERATE_DNFR_HIGH = 0.7  # upper bound of moderate ΔNFR range

__all__ = [
    "validate_sequence",
    "parse_sequence",
    "SequenceValidationResultWithHealth",
    "validate_sequence_with_health",
]

# ============================================================================


def _canonicalize_tokens(names: Sequence[str]) -> tuple[list[str], list[int]]:
    canonical: list[str] = []
    non_str_indices: list[int] = []
    for idx, tok in enumerate(names):
        if not isinstance(tok, str):
            non_str_indices.append(idx)
            canonical.append(str(tok))
        else:
            canonical.append(tok)
    return canonical, non_str_indices


def _compute_metadata(tokens: list[str]) -> dict[str, object]:
    from .pattern_detection import detect_pattern

    meta: dict[str, object] = {}
    meta["unknown_tokens"] = frozenset(
        t for t in tokens if t not in CANONICAL_OPERATOR_NAMES
    )
    meta["has_intermediate"] = any(t in INTERMEDIATE_OPERATORS for t in tokens)
    meta["has_reception"] = "reception" in tokens
    meta["has_coherence"] = "coherence" in tokens
    meta["has_dissonance"] = "dissonance" in tokens
    meta["has_stabilizer"] = any(t in STABILIZERS for t in tokens)
    try:
        pattern = detect_pattern(tokens)
        meta["detected_pattern"] = getattr(pattern, "value", str(pattern))
    except Exception:
        meta["detected_pattern"] = StructuralPattern.UNKNOWN.value
    return meta


def _check_thol_closure(tokens: list[str]) -> tuple[bool, str | None]:
    if SELF_ORGANIZATION in tokens and tokens[-1] not in SELF_ORGANIZATION_CLOSURES:
        return (
            False,
            ("self_organization requires terminal closure " "(silence or contraction)"),
        )
    return True, None


def _check_adjacent_compatibility(
    tokens: list[str],
) -> tuple[bool, int | None, str | None]:
    # Check for therapeutic patterns overriding compatibility rules
    if _is_legacy_therapeutic_pattern(tokens):
        return True, None, None

    prev = tokens[0]
    for i in range(1, len(tokens)):
        cur = tokens[i]
        level = get_compatibility_level(prev, cur)
        if level == CompatibilityLevel.AVOID:
            if prev == "silence":
                if cur == "silence":
                    msg = (
                        f"consecutive silence operations excluded by the retained "
                        f"adjacency policy: {prev} → {cur}"
                    )
                elif cur == "dissonance":
                    msg = (
                        "silence → dissonance is excluded by the retained "
                        "adjacency policy; low capacity alone does not "
                        "prevent a pressure change"
                    )
                else:
                    msg = f"invalid after silence: {prev} → {cur}"
            elif cur == "mutation":
                msg = (
                    f"Transition {prev} → {cur} is excluded by the retained "
                    f"adjacency policy; U4b history is checked separately"
                )
            else:
                msg = f"operator transition {prev} → {cur} fails the adjacency policy"
            return False, i, msg
        prev = cur
    return True, None, None


def _is_legacy_therapeutic_pattern(tokens: list[str]) -> bool:
    """Retain the historical named recipe's adjacency-policy exception.

    The recipe name establishes no therapeutic effect or core grammar rule.
    """
    # CONTAINED_CRISIS: emission,reception,coherence,dissonance,silence
    if len(tokens) == 5 and tokens == [
        "emission",
        "reception",
        "coherence",
        "dissonance",
        "silence",
    ]:
        return True

    return False


def _legacy_word_preference_error(tokens: list[str]) -> tuple[int, str] | None:
    """Apply only the explicitly requested historical word preferences.

    Callers first admit the shared core contracts. These preferences contain
    no live state evidence and never replace U3 or an operator precondition.
    """
    ok, message = _check_thol_closure(tokens)
    if not ok:
        return len(tokens) - 1, message or "legacy THOL closure preference"
    ok, index, message = _check_adjacent_compatibility(tokens)
    if not ok:
        return index or 1, message or "legacy adjacency preference"
    return None


def _build_result(
    *,
    names: Sequence[str],
    canonical: Sequence[str],
    passed: bool,
    message: str,
    metadata: Mapping[str, object],
    error: SequenceSyntaxError | None = None,
) -> SequenceValidationResult:
    return SequenceValidationResult(
        tokens=tuple(names),
        canonical_tokens=tuple(canonical),
        passed=passed,
        message=message,
        metadata=metadata,
        summary={
            "message": message,
            "tokens": tuple(canonical),
            "metadata": dict(metadata),
            **(
                {
                    "error": {
                        "index": error.index,
                        "token": error.token,
                        "message": error.message,
                    }
                }
                if error is not None
                else {}
            ),
        },
        artifacts={
            "tokens": tuple(names),
            "canonical_tokens": tuple(canonical),
        },
        error=error,
    )


def validate_sequence(
    names: Any,
    *,
    context: Mapping[str, Any] | None = None,
    compatibility_profile: str | None = None,
    **kwargs: Any,
) -> SequenceValidationResult:
    """Validate shared core word contracts, with explicit legacy preferences.

    Optional context keys:
    - initial_epi_nonzero: bool -> if True, permits non-generator start
      because EPI birth already occurred outside this sequence.
    - diagnostic: bool -> the exact two-token OZ/ZHIR probe may waive word
      requirements; this never supplies runtime preconditions or admission.

    Ordinary words delegate initialization, closure, debt, transformer history
    and depth-default checks to the same core validator as typed execution.
    The default ``core`` profile retains U1-U5 contracts, including calibrated
    U2/U4 policies. Explicit ``legacy`` adds historical adjacency/THOL-terminal
    preferences; these are not consequences of U1-U6 alone. Context may carry
    the profile through execution; an explicit argument overrides that value.
    Neither profile certifies live preconditions or autonomous dynamics.

    Any other unexpected keyword raises TypeError (legacy guard).
    """
    if kwargs:
        bad = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"unexpected keyword argument(s): {bad}")
    context = _normalize_sequence_context(
        context, compatibility_profile=compatibility_profile
    )
    compatibility_profile = context["compatibility_profile"]

    # type checks and canonicalization
    if not isinstance(names, (list, tuple)):
        try:
            names = list(names)  # type: ignore[assignment]
        except Exception:
            names = [names]  # type: ignore[assignment]
    canon_list, non_str = _canonicalize_tokens(names)  # type: ignore[arg-type]
    if non_str:
        idx = non_str[0]
        err = SequenceSyntaxError(idx, names[idx], "tokens must be str")
        meta = _compute_metadata([str(t) for t in names])
        meta["compatibility_profile"] = compatibility_profile
        return _build_result(
            names=names,  # type: ignore[arg-type]
            canonical=canon_list,
            passed=False,
            message="tokens must be str",
            metadata=meta,
            error=err,
        )

    tokens = [t for t in canon_list]
    meta = _compute_metadata(tokens)
    meta["compatibility_profile"] = compatibility_profile

    if not tokens:
        return _build_result(
            names=names,  # type: ignore[arg-type]
            canonical=tokens,
            passed=False,
            message="empty sequence",
            metadata=meta,
        )

    # Unknown tokens
    for i, t in enumerate(tokens):
        if t not in CANONICAL_OPERATOR_NAMES:
            err = SequenceSyntaxError(i, t, f"unknown tokens: {t}")
            return _build_result(
                names=names,  # type: ignore[arg-type]
                canonical=tokens,
                passed=False,
                message="unknown tokens",
                metadata=meta,
                error=err,
            )

    # The exact diagnostic fragment is descriptive only. It still requires
    # declared prior initialization and never constructs an executable word.
    diagnostic_probe = (
        context.get("diagnostic", False)
        and context.get("initial_epi_nonzero", False)
        and tokens == ["dissonance", "mutation"]
    )
    meta["diagnostic_probe"] = diagnostic_probe
    meta["canonical_word_checked"] = not diagnostic_probe
    if not diagnostic_probe:
        from .grammar_core import GrammarValidator

        initialized = context.get("initial_epi_nonzero", False)
        checks = GrammarValidator().validate_checks(
            [SimpleNamespace(name=token) for token in tokens],
            epi_initial=1.0 if initialized else 0.0,
        )
        failures = [check for check in checks if check.blocking and not check.passed]
        if failures:
            message = "; ".join(check.legacy_message() for check in failures)
            # Local endpoint attribution follows the shared rule identifier,
            # not an additional admission condition or a parsed message.
            index = len(tokens) - 1 if failures[0].rule == "U1b" else 0
            return _build_result(
                names=names,
                canonical=tokens,
                passed=False,
                message=message,
                metadata=meta,
                error=SequenceSyntaxError(index, tokens[index], message),
            )

    if compatibility_profile == "legacy":
        preference_error = _legacy_word_preference_error(tokens)
        if preference_error is not None:
            index, message = preference_error
            return _build_result(
                names=names,
                canonical=tokens,
                passed=False,
                message=message,
                metadata=meta,
                error=SequenceSyntaxError(index, tokens[index], message),
            )

    # All good
    return _build_result(
        names=names,  # type: ignore[arg-type]
        canonical=tokens,
        passed=True,
        message="ok",
        metadata=meta,
    )


def parse_sequence(
    names: Sequence[str],
    *,
    context: Mapping[str, Any] | None = None,
    compatibility_profile: str | None = None,
) -> SequenceValidationResult:
    """Use the same word checks as validate_sequence, raising on rejection."""
    result = validate_sequence(
        names,
        context=context,
        compatibility_profile=compatibility_profile,
    )
    if result.passed:
        return result
    if result.error is not None:
        raise result.error
    tokens = result.canonical_tokens
    raise SequenceSyntaxError(0, tokens[0] if tokens else "", result.message)


class SequenceValidationResultWithHealth:
    """Validation result wrapper that includes health metrics."""

    def __init__(self, validation_result, health_metrics=None):
        self._validation_result = validation_result
        self.health_metrics = health_metrics

    def __getattr__(self, name):
        """Delegate attribute access to the underlying validation result."""
        return getattr(self._validation_result, name)

    @property
    def passed(self):
        """Whether validation passed."""
        return self._validation_result.passed

    @property
    def tokens(self):
        """Original tokens."""
        return self._validation_result.tokens

    @property
    def canonical_tokens(self):
        """Canonical tokens."""
        return self._validation_result.canonical_tokens

    @property
    def message(self):
        """Validation message."""
        return self._validation_result.message

    @property
    def metadata(self):
        """Validation metadata."""
        return self._validation_result.metadata

    @property
    def error(self):
        """Validation error."""
        return self._validation_result.error


def validate_sequence_with_health(sequence):
    """Validate sequence and compute health metrics.

    This wrapper combines validation with health analysis.

    Parameters
    ----------
    sequence : Iterable[str]
        Sequence of operator names

    Returns
    -------
    result : SequenceValidationResultWithHealth
        Validation result with health_metrics attribute
    """
    # Import here to avoid circular dependency
    try:
        from ..operators.health_analyzer import SequenceHealthAnalyzer
    except ImportError:
        # If health analyzer not available, just validate
        result = validate_sequence(sequence)
        return SequenceValidationResultWithHealth(result, None)

    # Validate the sequence
    result = validate_sequence(sequence)

    # Add health metrics if validation passed
    health_metrics = None
    if result.passed:
        try:
            analyzer = SequenceHealthAnalyzer()
            health_metrics = analyzer.analyze_health(sequence)
        except Exception:
            # If health analysis fails, set to None
            health_metrics = None

    return SequenceValidationResultWithHealth(result, health_metrics)


# Compatibility: Canonical IL sequences and helpers

# Minimal registry for tests that import canonical IL sequences. These
# definitions are educational shims; the canonical grammar remains
# physics‑first.
CANONICAL_IL_SEQUENCES: Mapping[str, Mapping[str, object]] = {
    "EMISSION_COHERENCE": {
        "name": "safe_activation",
        "pattern": ["emission", "coherence"],
        "glyphs": [Glyph.AL, Glyph.IL],
        "optimization": "can_fuse",
        "description": "Emission stabilized by coherence",
    },
    "RECEPTION_COHERENCE": {
        "name": "stable_integration",
        "pattern": ["reception", "coherence"],
        "glyphs": [Glyph.EN, Glyph.IL],
        "optimization": "can_fuse",
        "description": "Reception consolidated into coherent form",
    },
    "DISSONANCE_COHERENCE": {
        "name": "creative_resolution",
        "pattern": ["dissonance", "coherence"],
        "glyphs": [Glyph.OZ, Glyph.IL],
        "optimization": "preserve",
        "description": "Dissonance resolved by stabilizer",
    },
    "RESONANCE_COHERENCE": {
        "name": "resonance_consolidation",
        "pattern": ["resonance", "coherence"],
        "glyphs": [Glyph.RA, Glyph.IL],
        "optimization": "preserve",
        "description": "Propagated coherence locked by IL",
    },
    "COHERENCE_MUTATION": {
        "name": "stable_transformation",
        "pattern": ["coherence", "mutation"],
        "glyphs": [Glyph.IL, Glyph.ZHIR],
        "optimization": "preserve",
        "description": "Stable base enabling phase transformation",
        "structural_effect": "Phase transformation from stable base",
    },
}

IL_ANTIPATTERNS: Mapping[str, Mapping[str, object]] = {
    "COHERENCE_SILENCE": {
        "severity": "info",
        "warning": "coherence → silence is valid but often redundant",
        "alternative": None,
        "alternative_glyphs": None,
    },
    "COHERENCE_COHERENCE": {
        "severity": "warning",
        "warning": "repeated coherence has limited structural effect",
        "alternative": None,
        "alternative_glyphs": None,
    },
    "SILENCE_COHERENCE": {
        "severity": "error",
        "warning": (
            "silence → coherence is non-canonical; "
            "use silence → emission → coherence"
        ),
        "alternative": ["silence", "emission", "coherence"],
        "alternative_glyphs": [Glyph.SHA, Glyph.AL, Glyph.IL],
    },
}

# Hot-path normalization map for grammar telemetry/runtime tracking.
# Hoisted to module scope to avoid per-call dict allocation in
# recognize_il_sequences().
_OPERATOR_NAME_TO_GLYPH: dict[str, Glyph] = {
    "emission": Glyph.AL,
    "reception": Glyph.EN,
    "coherence": Glyph.IL,
    "dissonance": Glyph.OZ,
    "coupling": Glyph.UM,
    "resonance": Glyph.RA,
    "silence": Glyph.SHA,
    "expansion": Glyph.VAL,
    "contraction": Glyph.NUL,
    "self_organization": Glyph.THOL,
    "mutation": Glyph.ZHIR,
    "transition": Glyph.NAV,
    "recursivity": Glyph.REMESH,
}


def recognize_il_sequences(
    glyphs: Sequence[Glyph],
) -> list[Mapping[str, object]]:
    """Recognize canonical two-step IL-related sequences.

    Returns matches with names/positions; antipatterns flagged.

    Note
    ----
    Pure detection function: classifies sequences without emitting
    warnings. User-facing warning emission belongs to the runtime
    layer (``grammar_application.on_glyph_applied``) to avoid
    duplicate notifications. This separation preserves the
    detection/emission boundary and matches the TNFR principle that
    grammar telemetry must not couple to side-effects.
    """
    # Handle string names by converting to Glyphs
    processed_glyphs = []
    for g in glyphs:
        if isinstance(g, str):
            processed_glyphs.append(_OPERATOR_NAME_TO_GLYPH.get(g.lower(), g))
        else:
            processed_glyphs.append(g)

    # Build quick lookup of patterns by glyph tuple
    pattern_by_glyphs = {
        tuple(v["glyphs"]): v["name"] for v in CANONICAL_IL_SEQUENCES.values()
    }

    results: list[Mapping[str, object]] = []
    for i in range(len(processed_glyphs) - 1):
        pair = (processed_glyphs[i], processed_glyphs[i + 1])
        name = pattern_by_glyphs.get(pair)
        if name:
            results.append(
                {
                    "pattern_name": name,
                    "position": i,
                    "is_antipattern": False,
                }
            )
        # Detect antipatterns
        elif pair == (Glyph.IL, Glyph.SHA):
            anti_info = IL_ANTIPATTERNS["COHERENCE_SILENCE"]
            results.append(
                {
                    "pattern_name": "coherence_silence_info",
                    "position": i,
                    "is_antipattern": True,
                    "severity": anti_info["severity"],
                    "warning": anti_info["warning"],
                    "alternative": anti_info.get("alternative"),
                    "alternative_glyphs": anti_info.get("alternative_glyphs"),
                }
            )
        elif pair == (Glyph.IL, Glyph.IL):
            anti_info = IL_ANTIPATTERNS["COHERENCE_COHERENCE"]
            results.append(
                {
                    "pattern_name": "coherence_coherence_antipattern",
                    "position": i,
                    "is_antipattern": True,
                    "severity": anti_info["severity"],
                    "warning": anti_info["warning"],
                    "alternative": anti_info.get("alternative"),
                    "alternative_glyphs": anti_info.get("alternative_glyphs"),
                }
            )
        elif pair == (Glyph.SHA, Glyph.IL):
            anti_info = IL_ANTIPATTERNS["SILENCE_COHERENCE"]
            results.append(
                {
                    "pattern_name": "silence_coherence_antipattern",
                    "position": i,
                    "is_antipattern": True,
                    "severity": anti_info["severity"],
                    "warning": anti_info["warning"],
                    "alternative": anti_info.get("alternative"),
                    "alternative_glyphs": anti_info.get("alternative_glyphs"),
                }
            )
    return results


def optimize_il_sequence(
    pattern: Sequence[Glyph], allow_fusion: bool = True
) -> Sequence[Glyph]:
    """Return optimization hint for a 2-step pattern."""
    if not allow_fusion:
        return pattern

    lookup = {
        tuple(v["glyphs"]): v["optimization"] for v in CANONICAL_IL_SEQUENCES.values()
    }
    opt = lookup.get(tuple(pattern), "preserve")
    if opt == "preserve":
        return pattern
    return pattern  # For now just return original


def suggest_il_sequence(
    current: Mapping[str, float], goal: Mapping[str, object] = None
) -> list[str]:
    """Suggest canonical 2-step IL sequence for a starting state."""
    if goal is None:
        goal = {}

    epi = current.get("epi", 0.0)
    dnfr = current.get("dnfr", 0.0)

    # Inactive node needs activation (low EPI but functioning vf)
    if epi < _INACTIVE_EPI_THRESHOLD:
        if goal.get("reactivate", False) or goal.get("consolidate", False):
            return ["emission", "coherence"]

    # High ΔNFR needs reduction
    if dnfr > _HIGH_DNFR_THRESHOLD:
        if goal.get("dnfr_target") == "low":
            return ["dissonance", "coherence"]

    # Moderate ΔNFR, direct coherence
    if _MODERATE_DNFR_LOW < dnfr < _MODERATE_DNFR_HIGH:
        if goal.get("dnfr_target") == "low":
            return ["coherence"]

    # Phase transformation goal
    if goal.get("phase_change", False):
        return ["coherence", "mutation"]

    # Consolidation goal
    if goal.get("consolidate", False):
        return ["coherence"]

    # Default fallback - but need to match test case logic
    if epi < _INACTIVE_EPI_THRESHOLD and goal.get("consolidate", False):
        # For very low EPI with consolidate goal, suggest activation first
        return ["emission", "coherence"]

    return ["emission", "coherence"]


# Duplicate functions removed - main implementations above

# Extend __all__ with compatibility symbols
__all__ += [
    "CANONICAL_IL_SEQUENCES",
    "IL_ANTIPATTERNS",
    "recognize_il_sequences",
    "optimize_il_sequence",
    "suggest_il_sequence",
]

# Grammar Validator Class
# ============================================================================

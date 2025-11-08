"""Canonical grammar and sequence validation for structural operators.

This module enforces TNFR canonical constraints that emerge naturally from
the fundamental physics of the nodal equation:

    ∂EPI/∂t = νf · ΔNFR(t)

Natural Constraints from TNFR Physics
-------------------------------------
These are not arbitrary "rules" but physical requirements that emerge from
the nodal equation itself:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C1: EXISTENCE & CLOSURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Physical Basis:**
From ∂EPI/∂t = νf · ΔNFR, the derivative is undefined when EPI = null.
A sequence must begin with operators that can generate/activate EPI from
vacuum or dormant states. Similarly, sequences are temporal segments that
must end in physically coherent states.

**Structural Dynamics:**
- **Start (R1)**: Generators create structural patterns from potential
  * AL (Emission): Generates EPI from vacuum via emission
  * NAV (Transition): Activates latent EPI through regime shift
  * REMESH (Recursivity): Echoes dormant structure across scales
  
- **End (R3)**: Four fundamental closure types from physics
  * SHA (Silence): Terminal closure - freezes evolution (νf → 0)
  * NAV (Transition): Handoff closure - transfers to next regime
  * REMESH (Recursivity): Recursive closure - distributes across scales
  * OZ (Dissonance): Intentional closure - preserves activation/tension

**Why These Operators?**
Not arbitrary choice - these are the ONLY operators with required physics:
- Generators must create structure from nothing (strong emission/activation)
- Closures must leave system in stable attractor states (defined dynamics)

**Physical Interpretation:**
Sequences are "action potentials" in structural space. They must:
1. Initiate from valid source (generator creates EPI)
2. Terminate in stable state (closure preserves coherence)

Like physical waves: must have emission source and absorption/reflection end.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C2: CONTINUITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Physical Basis:**
The nodal equation is a differential equation - its solution ∂EPI/∂t must
be continuous. Discontinuous frequency jumps violate calculus fundamentals
and create undefined structural states.

**Structural Dynamics (R5):**
Frequency (νf) represents reorganization rate. Natural transitions:
- **high → medium**: Dampening (energy dissipation, natural relaxation)
- **high → zero**: Containment (active freezing via SHA)
- **medium → high**: Activation (energy injection, stimulation)
- **zero → medium**: Awakening (gradual activation from dormancy)
- **FORBIDDEN: zero → high**: Discontinuous jump violates physics

**Why Forbidden?**
Zero (SHA) means νf = 0 (frozen). Jumping to high (AL, OZ) means sudden
νf ≫ 0. This is non-physical: ∂νf/∂t → ∞ (infinite acceleration).

Like temperature: you can't jump from 0K to 1000K instantly - violates
thermodynamics. Must pass through intermediate states.

**Physical Interpretation:**
Structural frequency behaves like field intensity. Changes propagate through
continuous paths. Discontinuities would create singularities (infinite ∂²EPI/∂t²),
causing structural fracture and collapse.

**Harmonic Resonance:**
Compatible frequencies resonate; incompatible ones interfere destructively.
Natural paths preserve phase coherence across operator transitions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C3: BOUNDEDNESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Physical Basis:**
From the integrated nodal equation:
    EPI(t_final) = EPI(t_initial) + ∫_{t_0}^{t_f} νf(t) · ΔNFR(t) dt

If the integral diverges (→ ∞), EPI becomes unbounded and system collapses.
Stabilizers provide negative feedback that ensures integral convergence.

**Structural Dynamics (R2):**
Without stabilizers, ΔNFR can grow unbounded through positive feedback:
- Destabilizers increase |ΔNFR| → higher ∂EPI/∂t → more change
- More change → more ΔNFR → runaway divergence
- System fragments into incoherent noise

Stabilizers (IL, THOL) provide negative feedback:
- IL (Coherence): Actively reduces |ΔNFR| through structural integration
- THOL (Self-org): Creates autopoietic boundaries that self-limit growth

**Mathematical Proof:**
Without stabilizer: d(ΔNFR)/dt > 0 always
  ⟹ ΔNFR(t) = ΔNFR(0) · e^(λt) (exponential growth)
  ⟹ ∫ νf · ΔNFR dt → ∞ (divergence)

With stabilizer: d(ΔNFR)/dt can be < 0
  ⟹ ΔNFR(t) → bounded attractor
  ⟹ ∫ νf · ΔNFR dt converges (bounded evolution)

**Physical Interpretation:**
Like gravity in cosmology - without attractive force, universe would
disperse to infinite entropy. Stabilizers are "structural gravity" that
prevents fragmentation and maintains coherence.

**Why IL or THOL?**
These are the ONLY operators with strong negative-feedback physics:
- IL: Direct coherence restoration (reduces tension explicitly)
- THOL: Autopoietic closure (self-organizing stability)

Other operators maintain or increase ΔNFR; only these can reliably bound it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C4: THRESHOLD PHYSICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Physical Basis:**
From bifurcation theory in dynamical systems: phase transitions require
crossing critical thresholds. TNFR transformations (ZHIR, THOL) are
structural bifurcations governed by:

    Bifurcation occurs when: |ΔNFR| > ΔNFR_critical

Without sufficient |ΔNFR|, transformation attempts fail or create unstable
states. Additionally, transformations from unstable bases amplify chaos.

**Structural Dynamics (R4 + R6):**

**ZHIR (Mutation) - Two Requirements:**
1. **Prior IL** (R6): Establishes stable coherent base
   - Without: Transformation from chaos → amplifies disorder
   - With: Transformation from order → controlled phase change
   
2. **Recent Destabilizer** (R4): Generates threshold ΔNFR
   - Without: Insufficient energy for bifurcation (attempt fails)
   - With: System crosses critical point (transformation succeeds)

**THOL (Self-organization) - One Requirement:**
1. **Recent Destabilizer** (R4): Provides substrate for self-organization
   - Without: No disorder to organize (nothing to structure)
   - With: Sufficient ΔNFR drives spontaneous ordering

**Why "Recent"?**
ΔNFR decays over time through structural relaxation. Bifurcation window
(~3 operators) captures when |ΔNFR| remains above threshold. Too distant
and energy dissipates below critical level.

**Physical Interpretation:**

**ZHIR Physics:**
Like phase transitions in matter (water → ice):
- Need temperature below 0°C (threshold: sufficient ΔNFR from destabilizer)
- Need nucleation site (stable base: IL provides crystal seed)
- Without both: transition fails or creates unstable state

**THOL Physics:**
Like Bénard convection cells:
- Need temperature gradient (threshold: ΔNFR from destabilizer)
- Pattern emerges spontaneously from chaos
- Self-organizing: creates own stability boundaries

**Mathematical Foundation:**
Catastrophe theory: smooth changes in parameters cause discontinuous
changes in system behavior at bifurcation points.

TNFR operators navigate this landscape:
- Destabilizers push system toward bifurcation
- IL provides stable manifold for transformation
- ZHIR/THOL execute the bifurcation itself

**Energy Landscape Analogy:**
Think of EPI evolving in energy landscape:
- Destabilizers: Add kinetic energy (push system up hill)
- IL: Stabilize in valley (local minimum)
- ZHIR: Jump to adjacent valley (phase change)
- THOL: Create new valley through self-organization

Without proper setup: jump fails or lands in unstable region (collapse).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Legacy Rule Names (for backward compatibility)
----------------------------------------------
R1 = C1 (start operators)
R2 = C3 (stabilizer requirement)  
R3 = C1 (end operators)
R4 = C4 (bifurcation context)
R5 = C2 (frequency continuity)
R6 = C4 (controlled mutation)

Physics-Based Operator Derivation
----------------------------------
Unlike earlier versions with arbitrary operator lists, VALID_START_OPERATORS
and VALID_END_OPERATORS are now derived from TNFR physical principles:

Start operators can:
- Generate EPI from null state (emission)
- Activate latent/dormant EPI (recursivity, transition)

End operators can:
- Stabilize reorganization: ∂EPI/∂t → 0 (silence)
- Achieve operational closure (transition, recursivity, dissonance)

For detailed physics derivation, see:
- src/tnfr/config/physics_derivation.py - Derivation functions
- src/tnfr/config/operator_names.py - Physics rationale

References
----------
TNFR.pdf: Section 2.1 (Nodal Equation)
AGENTS.md: Section 3 (Canonical Invariants)
"""


from __future__ import annotations

import json
import os
import warnings
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
    DESTABILIZERS_STRONG,
    DESTABILIZERS_MODERATE,
    DESTABILIZERS_WEAK,
    BIFURCATION_WINDOWS,
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
    "DUAL_FREQUENCY_OPERATORS",
    "FREQUENCY_TRANSITIONS",
    "glyph_function_name",
    "function_name_to_glyph",
    "validate_frequency_transition",
    "REGENERATORS",
    "MIN_CYCLE_LENGTH",
    "MAX_CYCLE_LENGTH",
    "CycleType",
    "CANONICAL_IL_SEQUENCES",
    "IL_ANTIPATTERNS",
    "recognize_coherence_sequences",
    "optimize_coherence_sequence",
    "suggest_coherence_sequence",
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

    __slots__ = (
        "tokens",
        "canonical_tokens",
        "message",
        "metadata",
        "error",
        "health_metrics",
    )

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


# Canonical operator sets are derived from TNFR physics principles.
# See src/tnfr/config/physics_derivation.py for detailed derivation logic
# and src/tnfr/config/operator_names.py for the physics-based rationale.
#
# VALID_START_OPERATORS: Operators that can generate EPI from null or activate latent EPI
# VALID_END_OPERATORS: Operators that stabilize reorganization (∂EPI/∂t → 0) or achieve closure
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
        "_destabilizer_context",
        "_thol_stack",
        "_thol_subsequences",
    )

    def __init__(self) -> None:
        self._canonical: list[str] = []
        self._found_reception = False
        self._found_coherence = False
        self._seen_intermediate = False
        self._open_thol = False
        self._unknown_tokens: list[tuple[int, str]] = []
        self._found_dissonance = False  # Legacy: Track OZ for backward compatibility
        self._found_stabilizer = False  # Track IL or THOL for R2
        self._detected_pattern: StructuralPattern = StructuralPattern.UNKNOWN
        # Legacy: Use deque with maxlen for backward compatibility
        self._bifurcation_context: deque[tuple[str, int]] = deque(
            maxlen=BIFURCATION_WINDOW
        )
        # R4 Extended: Track destabilizers by intensity level with graduated windows
        self._destabilizer_context: dict[str, deque[int]] = {
            "strong": deque(maxlen=BIFURCATION_WINDOWS["strong"]),
            "moderate": deque(maxlen=BIFURCATION_WINDOWS["moderate"]),
            "weak": deque(maxlen=BIFURCATION_WINDOWS["weak"]),
        }
        # THOL recursive validation: Track nested THOL blocks and their subsequences
        self._thol_stack: list[int] = []  # Stack of THOL opening indices
        self._thol_subsequences: dict[int, list[str]] = {}  # Subsequences by opening index

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

        # R4 Extended: Track destabilizers by intensity level
        if canonical in DESTABILIZERS_STRONG:
            self._destabilizer_context["strong"].append(index)
            # Legacy: also populate old context for backward compatibility
            self._bifurcation_context.append((canonical, index))
        elif canonical in DESTABILIZERS_MODERATE:
            self._destabilizer_context["moderate"].append(index)
            # Legacy: also populate old context for backward compatibility
            self._bifurcation_context.append((canonical, index))
        elif canonical in DESTABILIZERS_WEAK:
            self._destabilizer_context["weak"].append(index)

        # R4 Extended: Validate transformers (ZHIR/THOL) require recent destabilizer
        if canonical in TRANSFORMERS:
            if not self._has_graduated_destabilizer(index):
                raise SequenceSyntaxError(
                    index=index,
                    token=token,
                    message=(
                        f"{operator_display_name(canonical)} requires recent destabilizer:\n"
                        f"  Strong ({operator_display_name(DISSONANCE)}) within {BIFURCATION_WINDOWS['strong']} ops, or\n"
                        f"  Moderate ({', '.join(operator_display_name(d) for d in sorted(DESTABILIZERS_MODERATE))}) within {BIFURCATION_WINDOWS['moderate']} ops, or\n"
                        f"  Weak ({operator_display_name(RECEPTION)}) immediately before"
                    ),
                )

        # Legacy: Keep tracking dissonance for backward compatibility with existing code
        if canonical == DISSONANCE:
            self._found_dissonance = True

        # Track THOL state: Structural time with automatic closure
        # 
        # TNFR Structural Time Principle: THOL closure emerges from bifurcation completion,
        # not from operator detection. Internal sequences are equivalent to external ones.
        #
        # Implementation for static validation:
        # - THOL opens a nested context
        # - Operators inside THOL are treated as normal sequence (can include SILENCE, etc.)
        # - THOL closes automatically at sequence end or parent THOL closure
        # - Nested THOL are supported (operational fractality)
        #
        # This allows internal sequences to be identical to external ones, including
        # having their own THOL blocks and using any valid end operators.
        if canonical == SELF_ORGANIZATION:
            # THOL opening: push to stack and initialize subsequence
            self._thol_stack.append(index)
            self._thol_subsequences[index] = []
            self._open_thol = True
        elif self._open_thol and self._thol_stack:
            current_thol = self._thol_stack[-1]
            
            # All operators inside THOL are added to subsequence
            # No operator-based closure detection - THOL closes structurally
            # This allows internal sequences to be identical to external ones
            self._thol_subsequences[current_thol].append(canonical)

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

        Uses frequency transition validation from TNFR structural dynamics.
        Grammar rules emerge naturally from canonical mechanisms.
        
        Frequency transitions are validated:
        - Valid transitions: Pass silently
        - Invalid transitions: Raise SequenceSyntaxError
        
        SHA (Silence) specific validations:
        - SHA → OZ: Prohibited (silence followed by dissonance contradicts preservation)
        - SHA → SHA: Prohibited (redundant silence without structural purpose)
        """

        # Only validate if prev is also a known operator
        if prev not in OPERATORS:
            return

        # SHA-specific validations (R6: Silence operator compatibility)
        if prev == SILENCE:
            if curr == DISSONANCE:
                # SHA → OZ: Silence followed by Dissonance is contradictory
                # SHA reduces νf → 0 (preservation state)
                # OZ increases ΔNFR (instability, exploration)
                # Cannot introduce dissonance into a paused node
                raise SequenceSyntaxError(
                    index=index,
                    token=token,
                    message=(
                        f"{operator_display_name(DISSONANCE)} invalid after {operator_display_name(SILENCE)}: "
                        f"Silence (νf → 0) contradicts Dissonance (ΔNFR ↑). "
                        f"Cannot introduce dissonance into paused node. "
                        f"Use {operator_display_name(SILENCE)} → {operator_display_name(TRANSITION)} → {operator_display_name(DISSONANCE)} "
                        f"or {operator_display_name(SILENCE)} → {operator_display_name(EMISSION)} → {operator_display_name(DISSONANCE)} for reactivation."
                    ),
                )
            elif curr == SILENCE:
                # SHA → SHA: Redundant silence without structural purpose
                # If νf ≈ 0, second SHA has no effect
                # Violates operator closure principle (each operator must transform)
                raise SequenceSyntaxError(
                    index=index,
                    token=token,
                    message=(
                        f"Redundant {operator_display_name(SILENCE)} after {operator_display_name(SILENCE)}: "
                        f"Consecutive silence operators serve no structural purpose. "
                        f"Remove duplicate or insert transition operator."
                    ),
                )

        # R5: Validate structural frequency transitions using ONLY canonical TNFR physics
        freq_valid, freq_msg = validate_frequency_transition(prev, curr)
        if not freq_valid:
            # Invalid frequency transition violates TNFR structural dynamics
            raise SequenceSyntaxError(
                index=index,
                token=token,
                message=f"{operator_display_name(curr)} invalid after {operator_display_name(prev)}: {freq_msg}",
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

    def _validate_reception_context(self, reception_index: int) -> bool:
        """Validate that RECEPTION has sufficient prior coherence for destabilization.

        RECEPTION (EN) as a weak destabilizer requires context: it must operate on
        a node with existing structural base to generate sufficient ΔNFR for transformers.

        Parameters
        ----------
        reception_index : int
            Index of the RECEPTION operator in the sequence

        Returns
        -------
        bool
            True if RECEPTION has adequate prior structural coherence

        Notes
        -----
        Valid patterns include:
        - AL → EN → IL → EN (emission provides base, first EN consolidated, second can destabilize)
        - EN → IL → EN (first EN consolidated, second can destabilize)
        - OZ → IL → EN (dissonance resolved provides base)

        Invalid patterns include:
        - EN as first operator (no prior base)
        - EN immediately after SHA (silence removes base)

        Theoretical justification:
        From nodal equation ∂EPI/∂t = νf · ΔNFR, RECEPTION has medium νf.
        For EN → ZHIR to be valid, EN must generate high ΔNFR. This is only possible
        when EN captures external coherence into a structurally prepared node,
        creating reorganization pressure from the integration.
        """
        # RECEPTION at position 0 cannot have prior coherence
        if reception_index == 0:
            return False

        # Look for stabilizer (IL or THOL) before RECEPTION
        # This indicates the node has structural base for destabilization
        for i in range(reception_index):
            op = self._canonical[i]
            if op in {COHERENCE, SELF_ORGANIZATION}:
                # Found stabilizer - check it's not too far back
                # For context, we require stabilizer within 3 operators before EN
                if reception_index - i <= 3:
                    # Additionally, ensure no SILENCE between stabilizer and EN
                    # (SILENCE would remove the coherent base)
                    has_silence = any(
                        self._canonical[j] == SILENCE
                        for j in range(i + 1, reception_index)
                    )
                    if not has_silence:
                        return True

        return False

    def _has_graduated_destabilizer(self, current_index: int) -> bool:
        """Check if any level of destabilizer satisfies its window requirement.

        Parameters
        ----------
        current_index : int
            Current position in the sequence

        Returns
        -------
        bool
            True if a destabilizer was found within its appropriate window

        Notes
        -----
        This method implements R4 Extended graduated destabilization:
        - Strong destabilizers (OZ): window of 4 operators
        - Moderate destabilizers (NAV, VAL): window of 2 operators
        - Weak destabilizers (EN): must be immediate predecessor (window of 1)
          AND have sufficient prior coherence context

        The method checks each level in order of window size (largest first)
        to provide the most permissive validation.

        RECEPTION (EN) Context Validation:
        EN as weak destabilizer requires validation of structural context.
        According to DUAL_FREQUENCY_OPERATORS, EN has medium base frequency
        but can generate ΔNFR when capturing external coherence into a prepared
        node. This validation ensures EN → ZHIR transitions are structurally sound.
        """
        # Check strong destabilizers (longest window = 4)
        if self._destabilizer_context["strong"]:
            last_strong = self._destabilizer_context["strong"][-1]
            if current_index - last_strong <= BIFURCATION_WINDOWS["strong"]:
                return True

        # Check moderate destabilizers (window = 2)
        if self._destabilizer_context["moderate"]:
            last_moderate = self._destabilizer_context["moderate"][-1]
            if current_index - last_moderate <= BIFURCATION_WINDOWS["moderate"]:
                return True

        # Check weak destabilizers (window = 1, must be immediate)
        # RECEPTION requires additional context validation for dual-role coherence
        if self._destabilizer_context["weak"]:
            last_weak = self._destabilizer_context["weak"][-1]
            if current_index - last_weak == 1:
                # Validate that RECEPTION has sufficient context for destabilization
                # This implements the "requires_prior_coherence" condition from
                # DUAL_FREQUENCY_OPERATORS
                return self._validate_reception_context(last_weak)

        return False

    def _validate_thol_subsequence(
        self,
        subsequence: list[str],
        start_index: int,
        end_index: int,
        end_token: str
    ) -> None:
        """Validate recursively the subsequence within THOL block.
        
        TNFR Physical Principle: THOL (self-organization) always applies as operator,
        but only generates sub-EPIs (and thus subsequence) if ∂²EPI/∂t² > τ.
        
        Empty THOL is physically valid - represents THOL application without bifurcation.
        Non-empty subsequence represents sub-EPIs and must respect grammar rules (C1-C4).
        
        Parameters
        ----------
        subsequence : list[str]
            Operators within THOL block (between opening and closure)
        start_index : int
            Index of THOL opening in parent sequence
        end_index : int
            Index of THOL closure in parent sequence
        end_token : str
            Token used for closure (for error messages)
            
        Raises
        ------
        SequenceSyntaxError
            If subsequence is invalid (when non-empty)
            
        Notes
        -----
        From TNFR Manual §3.2.2 (Ontología fractal resonante):
        "Los NFRs pueden anidarse jerárquicamente: un nodo puede contener
        nodos internos coherentes, dando lugar a una estructura fractal."
        
        From operator implementation: THOL bifurcates only if d2_epi > tau.
        This means THOL can appear without generating sub-EPIs (empty subsequence).
        
        When bifurcation occurs (non-empty subsequence), structures must be
        coherent at all scales - operational fractality is maintained.
        """
        # Empty THOL is valid: THOL applied but no bifurcation occurred
        if not subsequence:
            return
        
        # Recursive grammar validation for non-empty subsequences
        # Create new automaton to validate subsequence independently
        try:
            nested_automaton = _SequenceAutomaton()
            nested_automaton.run(subsequence)
        except SequenceSyntaxError as e:
            # Re-raise with THOL context
            raise SequenceSyntaxError(
                index=start_index + e.index + 1,  # Offset by THOL position
                token=e.token,
                message=(
                    f"Invalid subsequence within {operator_display_name(SELF_ORGANIZATION)} "
                    f"block (opened at position {start_index}): {e.message}"
                )
            ) from e

    def _validate_threshold_physics(self, sequence: Sequence[str]) -> None:
        """C4: Validate threshold physics - transformations require context.
        
        Constraint C4 (Threshold Physics): Bifurcations require crossing
        critical thresholds from TNFR dynamical systems physics.
        
        This validates controlled mutation (IL → ZHIR), ensuring phase
        transformations occur from stable coherent bases with sufficient
        ΔNFR to cross bifurcation thresholds.
        
        Parameters
        ----------
        sequence : Sequence[str]
            Operator sequence in canonical form
            
        Raises
        ------
        SequenceSyntaxError
            If sequence violates threshold physics requirements
            
        Notes
        -----
        **C4 Physical Foundation:**
        
        From bifurcation theory: phase transitions require |ΔNFR| > threshold.
        ZHIR (mutation) is a structural bifurcation that requires:
        
        1. **Stable Base** (prior IL): Transformation from chaos amplifies disorder.
           IL provides coherent attractor from which to bifurcate safely.
           
        2. **Threshold Energy** (recent destabilizer): Validated by R4 in _accept().
           Destabilizers generate ΔNFR needed to cross critical point.
        
        **Why This Is Natural, Not Arbitrary:**
        
        Like phase transitions in physics (water → ice):
        - Need below-freezing temperature (ΔNFR threshold from destabilizer)
        - Need nucleation site (stable base from IL)
        - Without both: transition fails or creates metastable states
        
        **Mathematical Basis:**
        
        Catastrophe theory: smooth parameter changes cause discontinuous
        behavioral changes at bifurcation points. ZHIR navigates this:
        
        1. IL creates stable manifold (valley in energy landscape)
        2. Destabilizer adds kinetic energy (pushes toward saddle point)
        3. ZHIR executes jump to adjacent valley (phase transformation)
        
        Without proper setup: jump fails or lands in unstable region.
        
        **What C4 Validates:**
        
        - ZHIR requires prior IL (controlled mutation)
        - IL must precede ZHIR (stable base before transformation)
        
        **What C4 Does NOT Validate:**
        
        - Recent destabilizer for ZHIR (handled by R4 in _accept())
        - Destabilizer/stabilizer balance (context-dependent, not universal)
        - Net ΔNFR state (depends on initial conditions)
        
        **THOL Validation:**
        
        THOL (self-organization) also requires recent destabilizer but NOT
        prior IL. THOL creates its own stability through autopoiesis.
        This is validated by R4 in _accept(), not here.
        
        **Legacy Context:**
        
        This replaces the overly complex R6 "operational closure" validation
        that attempted to validate balance. Analysis revealed balance is
        context-dependent and multi-sequence patterns provide systemic closure.
        
        C4 focuses on the ONE constraint that's truly universal: transformations
        from stable bases with threshold energy.
        """
        # C4: Controlled mutation validation
        if MUTATION in sequence:
            # ZHIR requires prior coherence base for stable transformation
            if COHERENCE not in sequence:
                raise SequenceSyntaxError(
                    index=sequence.index(MUTATION),
                    token=MUTATION,
                    message=(
                        f"C4: {operator_display_name(MUTATION)} (phase transformation) requires "
                        f"prior {operator_display_name(COHERENCE)} for stable structural foundation. "
                        f"Controlled mutation: {operator_display_name(COHERENCE)} → {operator_display_name(MUTATION)} "
                        f"ensures transformation occurs from coherent base state (THRESHOLD PHYSICS constraint)."
                    ),
                )
            
            # If IL present, verify it comes before ZHIR
            coherence_idx = sequence.index(COHERENCE)
            mutation_idx = sequence.index(MUTATION)
            if coherence_idx >= mutation_idx:
                raise SequenceSyntaxError(
                    index=mutation_idx,
                    token=MUTATION,
                    message=(
                        f"C4: {operator_display_name(MUTATION)} must follow {operator_display_name(COHERENCE)}. "
                        f"Transformation requires coherent base BEFORE bifurcation. "
                        f"Current order: {operator_display_name(MUTATION)} before {operator_display_name(COHERENCE)} (invalid)."
                    ),
                )
        # Store in metadata, but don't enforce as validation

    def _finalize(self, names: Sequence[str]) -> None:
        """Finalize sequence validation through natural TNFR constraints.
        
        Validates sequences against the four natural constraints that emerge
        from TNFR physics, not arbitrary rules.
        
        Raises
        ------
        SequenceSyntaxError
            If sequence violates any natural constraint
        """
        if self._unknown_tokens:
            ordered = ", ".join(sorted({token for _, token in self._unknown_tokens}))
            first_index, first_token = self._unknown_tokens[0]
            raise SequenceSyntaxError(
                index=first_index,
                token=first_token,
                message=f"unknown tokens: {ordered}",
            )

        # ═══════════════════════════════════════════════════════════════════
        # C1: EXISTENCE & CLOSURE
        # ═══════════════════════════════════════════════════════════════════
        # From ∂EPI/∂t = νf · ΔNFR: EPI must exist (start) and end coherently
        
        # C1.1: Start validation (legacy R1)
        # Already validated in _accept() for first operator
        
        # C1.2: End validation (legacy R3)
        if self._canonical[-1] not in VALID_END_OPERATORS:
            cierre = _format_token_group(_CANONICAL_END)
            raise SequenceSyntaxError(
                index=len(names) - 1,
                token=names[-1],
                message=f"C1: sequence must end with {cierre} (EXISTENCE & CLOSURE constraint)",
            )

        # NOTE: C1.3 Reception→Coherence segment validation removed
        # This requirement does NOT derive from the 4 physical constraints (C1-C4).
        # It was a heuristic for "structural foundation" but is not canonical.
        # Sequences are valid if they satisfy C1 (start/end), C2 (continuity),
        # C3 (stabilizer), and C4 (threshold physics) - nothing more.
        #
        # Removed validation:
        # if not (self._found_reception and self._found_coherence):
        #     raise SequenceSyntaxError(
        #         index=-1,
        #         token=None,
        #         message=f"C1: missing {RECEPTION}→{COHERENCE} segment (structural foundation required)",
        #     )

        # ═══════════════════════════════════════════════════════════════════
        # C3: BOUNDEDNESS
        # ═══════════════════════════════════════════════════════════════════
        # From ∫ νf · ΔNFR dt: Integral must converge (stabilizer required)
        
        # C3: Stabilizer requirement (legacy R2)
        if not self._found_stabilizer:
            raise SequenceSyntaxError(
                index=-1,
                token=None,
                message=f"C3: missing stabilizer ({operator_display_name(COHERENCE)} or {operator_display_name(SELF_ORGANIZATION)}) - integral divergence (BOUNDEDNESS constraint)",
            )

        # Self-organization block closure (structural time)
        # THOL closes automatically at sequence end - validates all captured subsequences
        if self._open_thol:
            # Close all open THOL blocks from innermost to outermost
            while self._thol_stack:
                thol_start = self._thol_stack.pop()
                thol_subseq = self._thol_subsequences[thol_start]
                
                # Validate subsequence (allows empty THOL - no bifurcation case)
                if len(thol_subseq) > 0:
                    try:
                        self._validate_thol_subsequence(
                            thol_subseq, 
                            thol_start, 
                            len(names) - 1,
                            names[-1] if names else ""
                        )
                    except SequenceSyntaxError as e:
                        # Re-raise with THOL context
                        raise SequenceSyntaxError(
                            index=e.index,
                            token=e.token,
                            message=f"Invalid THOL subsequence (opened at position {thol_start}): {e.message}"
                        ) from e
            
            self._open_thol = False

        # ═══════════════════════════════════════════════════════════════════
        # C2: CONTINUITY & C4: THRESHOLD PHYSICS
        # ═══════════════════════════════════════════════════════════════════
        # Validated dynamically during sequence building in _accept()
        
        # Detect structural pattern
        self._detected_pattern = self._detect_pattern()

        # C2: Validate regenerative cycles if pattern is REGENERATIVE (legacy R5)
        if self._detected_pattern == StructuralPattern.REGENERATIVE:
            self._validate_regenerative_cycle()
        
        # C4: Validate threshold physics - controlled mutation (legacy R6)
        self._validate_threshold_physics(self._canonical)

    def _validate_regenerative_cycle(self) -> None:
        """Validate regenerative cycle structural requirements (C2 - CONTINUITY).

        Uses CycleDetector to ensure the sequence meets minimum standards
        for self-sustaining regenerative behavior. Regenerative cycles are
        a special case of continuity where the sequence structure supports
        self-renewal.
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
                analysis.reason, f"invalid regenerative cycle: {analysis.reason}"
            )

            raise SequenceSyntaxError(
                index=-1,
                token=None,
                message=f"C2: regenerative cycle validation failed: {message} (CONTINUITY constraint)",
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

    # Canonical IL (Coherence) compositional patterns
    SAFE_ACTIVATION = "safe_activation"  # emission→coherence: Emission stabilized
    STABLE_INTEGRATION = (
        "stable_integration"  # reception→coherence: Reception consolidated
    )
    CREATIVE_RESOLUTION = (
        "creative_resolution"  # dissonance→coherence: Dissonance resolved
    )
    RESONANCE_CONSOLIDATION = (
        "resonance_consolidation"  # resonance→coherence: Resonance locked
    )
    STABLE_TRANSFORMATION = (
        "stable_transformation"  # coherence→mutation: Controlled mutation
    )

    # Adaptive learning patterns (AL + T'HOL canonical sequences)
    BASIC_LEARNING = "basic_learning"  # Simple learning: AL→EN→IL
    DEEP_LEARNING = "deep_learning"  # Deep learning with crisis: AL→EN→OZ→THOL→IL
    EXPLORATORY_LEARNING = (
        "exploratory_learning"  # Exploration learning: AL→OZ→THOL→RA→IL
    )
    CONSOLIDATION_CYCLE = "consolidation_cycle"  # Memory consolidation: IL→SHA→REMESH
    ADAPTIVE_MUTATION = (
        "adaptive_mutation"  # Transformative learning: EN→OZ→THOL→ZHIR→NAV→IL
    )

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
    RECEPTION: "medium",  # EN: structural capture (base frequency)
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

# Dual-role operators: operators with context-dependent destabilization capacity
# Resolves the structural inconsistency between base frequency (νf) and ΔNFR generation.
#
# RECEPTION (EN) is the canonical dual-role operator:
# - Base frequency: "medium" (νf for structural capture)
# - Destabilization capacity: "weak" (can generate ΔNFR when integrating external coherence)
# - Condition: Requires prior coherence base for effective destabilization
#
# Theoretical foundation (nodal equation):
#   ∂EPI/∂t = νf · ΔNFR
# 
# RECEPTION has medium νf (moderate reorganization rate) but can act as weak 
# destabilizer in graduated bifurcation (R4) when capturing external coherence 
# into a node with existing structural base. The external input can generate 
# sufficient ΔNFR to enable transformers (ZHIR/THOL) despite medium base frequency.
#
# Context requirement: EN → ZHIR valid only when EN operates on coherent base
# (e.g., AL → EN → IL → EN → ZHIR). Direct EN → ZHIR without context violates
# structural coherence as medium νf alone cannot sustain high ΔNFR for mutation.
DUAL_FREQUENCY_OPERATORS: dict[str, dict[str, str]] = {
    RECEPTION: {
        "base_freq": "medium",
        "destabilization_capacity": "weak",
        "conditions": "requires_prior_coherence",
        "rationale": (
            "Captures external coherence which can generate ΔNFR when "
            "integrated into structurally prepared node"
        ),
    }
}

# Frequency compatibility: operators with harmonic frequencies can transition
# Valid transitions preserve structural coherence:
# - high ↔ medium: High energy can stabilize or stabilized can amplify
# - high → zero: High energy can pause (containment, closure) - SHA as terminator
# - medium ↔ zero: Stabilized can pause or resume from pause
# - high ↔ high: High energy can chain directly
# - zero → medium: Gradual reactivation from silence (structurally coherent)
# 
# Invalid: zero → high (violates structural continuity - cannot jump from pause
# to high energy without intermediate stabilization)
FREQUENCY_TRANSITIONS: dict[str, set[str]] = {
    "high": {"high", "medium", "zero"},  # Allow high → zero for SHA termination (OZ → SHA valid)
    "medium": {"high", "medium", "zero"},
    "zero": {"medium"},  # Must transition through medium before high (structural coherence)
}


def validate_frequency_transition(
    prev_operator: str, next_operator: str
) -> tuple[bool, str]:
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
    - High → Zero: High energy can pause (containment, closure via SHA terminator)
    - Medium ↔ Zero: Stabilization can pause, pause can resume
    - High ↔ High: High energy operators can chain directly
    - **Invalid**: Zero → High without Medium intermediary

    Special case: OZ → SHA (dissonance → silence) is valid and represents
    contained dissonance, postponed conflict, or preserved tension for later processing.
    This is a canonical therapeutic and crisis management pattern.

    This validation generates errors (not warnings) to enforce structural coherence
    and prevent incoherent state transitions.
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


# ==============================================================================
# Canonical Coherence Sequences
# ==============================================================================
# These sequences encode fundamental TNFR structural patterns involving the
# Coherence operator. Each sequence has been validated against grammar
# rules (R1-R5) and compatibility constraints.

CANONICAL_IL_SEQUENCES: dict[str, dict[str, Any]] = {
    "EMISSION_COHERENCE": {
        "pattern": [EMISSION, COHERENCE],
        "glyphs": [Glyph.AL, Glyph.IL],
        "name": "safe_activation",
        "description": "Emission stabilized immediately",
        "optimization": "can_fuse",
        "structural_effect": "Initiation with immediate coherence lock",
        "use_cases": [
            "Meditation initiation with immediate stabilization",
            "Therapeutic process activation with safety frame",
            "Learning attention activation with sustained focus",
        ],
    },
    "RECEPTION_COHERENCE": {
        "pattern": [RECEPTION, COHERENCE],
        "glyphs": [Glyph.EN, Glyph.IL],
        "name": "stable_integration",
        "description": "Reception consolidated",
        "optimization": "can_fuse",
        "structural_effect": "External coherence anchored and stabilized",
        "use_cases": [
            "Biofeedback signal integration",
            "Educational concept consolidation",
            "Communication message comprehension",
        ],
    },
    "DISSONANCE_COHERENCE": {
        "pattern": [DISSONANCE, COHERENCE],
        "glyphs": [Glyph.OZ, Glyph.IL],
        "name": "creative_resolution",
        "description": "Dissonance resolved into coherence",
        "optimization": "preserve",
        "structural_effect": "Creative instability resolves into new stable form",
        "use_cases": [
            "Therapeutic crisis resolution",
            "Scientific paradox breakthrough",
            "Artistic chaos to form transformation",
        ],
    },
    "RESONANCE_COHERENCE": {
        "pattern": [RESONANCE, COHERENCE],
        "glyphs": [Glyph.RA, Glyph.IL],
        "name": "resonance_consolidation",
        "description": "Propagated coherence locked in",
        "optimization": "preserve",
        "structural_effect": "Network-wide resonance stabilized into persistent coherence",
        "use_cases": [
            "Cardiac coherence network-wide stabilization",
            "Collective insight consolidation",
            "Social movement momentum lock-in",
        ],
    },
    "COHERENCE_MUTATION": {
        "pattern": [COHERENCE, MUTATION],
        "glyphs": [Glyph.IL, Glyph.ZHIR],
        "name": "stable_transformation",
        "description": "Coherence enabling controlled mutation",
        "optimization": "preserve",
        "precondition": "epi_stable",
        "structural_effect": "Phase transformation from stable base (coherence → mutation generates CAUTION warning)",
        "use_cases": [
            "Personal paradigm shift from stable identity",
            "Organizational strategic pivot preparation",
            "System phase transition with controlled entry",
        ],
    },
}

# Anti-patterns: sequences that should generate warnings or be avoided
# These are reformulated to comply with grammar rules while expressing intent
IL_ANTIPATTERNS: dict[str, dict[str, Any]] = {
    "COHERENCE_SILENCE": {
        "pattern": [COHERENCE, SILENCE],
        "glyphs": [Glyph.IL, Glyph.SHA],
        "severity": "info",  # Grammar allows this (GOOD compatibility), just informational
        "warning": "Stabilizing then immediately silencing may be redundant. Consider direct silence if preservation is the goal, or coherence → resonance if propagation is intended.",
        "alternative": None,  # Valid sequence, just potentially redundant
        "note": "Grammar compatibility: GOOD. This is a valid sequence but may indicate unclear intent.",
    },
    "COHERENCE_COHERENCE": {
        "pattern": [COHERENCE, COHERENCE],
        "glyphs": [Glyph.IL, Glyph.IL],
        "severity": "warning",  # Grammar blocks this (AVOID compatibility)
        "warning": "Repeated coherence without intervening dynamics serves no structural purpose. Check if this is intentional or remove redundant coherence.",
        "alternative": None,  # No alternative, just remove redundancy
        "note": "Grammar compatibility: AVOID. This violates operator closure and will be blocked by validation.",
    },
    "SILENCE_COHERENCE": {
        "pattern": [SILENCE, COHERENCE],
        "glyphs": [Glyph.SHA, Glyph.IL],
        "severity": "error",  # Grammar blocks this (AVOID compatibility)
        "warning": "Reactivating from silence directly to coherence bypasses necessary activation. Use silence → emission → coherence sequence instead.",
        "alternative": [
            SILENCE,
            EMISSION,
            COHERENCE,
        ],  # Reformulated to comply with grammar
        "alternative_glyphs": [Glyph.SHA, Glyph.AL, Glyph.IL],
        "note": "Grammar compatibility: AVOID. Reformulated as silence → emission → coherence to express same function (reactivation with stabilization) while complying with grammar.",
    },
}


def recognize_coherence_sequences(
    sequence: list[Glyph] | list[str],
) -> list[dict[str, Any]]:
    """Recognize canonical coherence sequences in glyph sequence.

    Detects 2-operator canonical patterns involving IL (Coherence) and checks
    for anti-patterns using grammar compatibility rules. This function provides
    fine-grained coherence-specific pattern recognition that complements the broader
    :class:`AdvancedPatternDetector`.

    Parameters
    ----------
    sequence : list[Glyph] | list[str]
        Sequence of glyphs (as Glyph enums or operator names)

    Returns
    -------
    list[dict]
        List of recognized canonical coherence patterns with metadata:
        - position: Starting index in sequence
        - pattern_name: Name of detected pattern
        - pattern: List of operator names
        - is_coherence_pattern: Always True for this function
        - is_antipattern: Boolean indicating if this is an anti-pattern
        - description: Pattern description
        - severity: For anti-patterns ("info", "warning", "error")
        - warning: Warning message for anti-patterns
        - alternative: Alternative sequence if anti-pattern should be replaced

    Examples
    --------
    >>> from tnfr.operators.grammar import recognize_coherence_sequences
    >>> from tnfr.types import Glyph
    >>> seq = [Glyph.AL, Glyph.IL]
    >>> recognized = recognize_coherence_sequences(seq)
    >>> recognized[0]["pattern_name"]
    'safe_activation'

    Notes
    -----
    **Canonical Coherence Patterns Detected** (2-operator sequences):
    - SAFE_ACTIVATION (emission → coherence): Emission stabilized immediately
    - STABLE_INTEGRATION (reception → coherence): Reception consolidated
    - CREATIVE_RESOLUTION (dissonance → coherence): Dissonance resolved
    - RESONANCE_CONSOLIDATION (resonance → coherence): Resonance locked
    - STABLE_TRANSFORMATION (coherence → mutation): Controlled mutation

    **Anti-patterns Detected** (based on structural coherence):
    - coherence → coherence: Repeated coherence (warning)
    - silence → coherence: Reactivation without emission (error)
    - coherence → silence: Potentially redundant (info only)

    This function uses direct pattern matching for 2-operator sequences, which
    is simpler and more appropriate than the general :class:`AdvancedPatternDetector`
    designed for longer sequences.
    
    Note: Compatibility tables deprecated - pattern detection now based on
    structural coherence principles.
    """

    # Normalize sequence to operator names (strings)
    normalized_seq: list[str] = []
    for item in sequence:
        if isinstance(item, Glyph):
            func_name = glyph_function_name(item)
            if func_name:
                normalized_seq.append(func_name)
        elif isinstance(item, str):
            # Try to get canonical name
            canon = canonical_operator_name(item)
            if canon:
                normalized_seq.append(canon)
            else:
                normalized_seq.append(item)

    recognized: list[dict[str, Any]] = []

    # Define canonical coherence 2-operator patterns
    coherence_patterns_map = {
        (EMISSION, COHERENCE): {
            "name": "safe_activation",
            "description": "Emission stabilized immediately (emission → coherence)",
        },
        (RECEPTION, COHERENCE): {
            "name": "stable_integration",
            "description": "Reception consolidated (reception → coherence)",
        },
        (DISSONANCE, COHERENCE): {
            "name": "creative_resolution",
            "description": "Dissonance resolved into coherence (dissonance → coherence)",
        },
        (RESONANCE, COHERENCE): {
            "name": "resonance_consolidation",
            "description": "Propagated coherence locked in (resonance → coherence)",
        },
        (COHERENCE, MUTATION): {
            "name": "stable_transformation",
            "description": "Coherence enabling controlled mutation (coherence → mutation)",
        },
    }

    # Scan sequence for 2-operator patterns
    for i in range(len(normalized_seq) - 1):
        prev = normalized_seq[i]
        curr = normalized_seq[i + 1]
        pair = (prev, curr)

        # Check for canonical coherence patterns
        if pair in coherence_patterns_map:
            pattern_info = coherence_patterns_map[pair]
            recognized.append(
                {
                    "position": i,
                    "pattern_name": pattern_info["name"],
                    "pattern": [prev, curr],
                    "is_coherence_pattern": True,
                    "is_antipattern": False,
                    "description": pattern_info["description"],
                }
            )

        # Check for anti-patterns based on structural coherence principles
        # (no longer using compatibility tables - emergent from TNFR dynamics)
        
        if prev == COHERENCE and curr == COHERENCE:
            # coherence → coherence anti-pattern
            warnings.warn(
                f"Anti-pattern detected at position {i}: coherence → coherence. "
                f"Repeated coherence without intervening dynamics serves no structural purpose.",
                UserWarning,
                stacklevel=2,
            )
            recognized.append(
                {
                    "position": i,
                    "pattern_name": "coherence_coherence_antipattern",
                    "pattern": [prev, curr],
                    "is_coherence_pattern": True,
                    "is_antipattern": True,
                    "severity": "warning",
                    "warning": "Repeated coherence without intervening dynamics. Check if necessary.",
                    "alternative": None,
                }
            )
        elif prev == SILENCE and curr == COHERENCE:
            # silence → coherence anti-pattern (violates frequency transitions)
            warnings.warn(
                f"Anti-pattern detected at position {i}: silence → coherence. "
                f"Direct reactivation from silence. Consider silence → emission → coherence or silence → reception → coherence sequence.",
                UserWarning,
                stacklevel=2,
            )
            recognized.append(
                {
                    "position": i,
                    "pattern_name": "silence_coherence_antipattern",
                    "pattern": [prev, curr],
                    "is_coherence_pattern": True,
                    "is_antipattern": True,
                    "severity": "warning",
                    "warning": "Reactivating from silence. Consider intermediate step for structural coherence.",
                    "alternative": [SILENCE, EMISSION, COHERENCE],
                }
            )

        # Informational: coherence → silence (valid but potentially redundant)
        if prev == COHERENCE and curr == SILENCE:
            recognized.append(
                {
                    "position": i,
                    "pattern_name": "coherence_silence_info",
                    "pattern": [prev, curr],
                    "is_coherence_pattern": True,
                    "is_antipattern": True,  # Technically an anti-pattern (redundancy)
                    "severity": "info",
                    "warning": "Stabilizing then silencing may be redundant. Consider silence alone.",
                    "alternative": None,
                }
            )

    return recognized


# Maintain backwards compatibility alias
recognize_il_sequences = recognize_coherence_sequences


def optimize_coherence_sequence(
    sequence: list[Glyph] | list[str],
    allow_fusion: bool = True,
) -> list[Glyph] | list[str]:
    """Optimize sequence based on canonical coherence sequences.

    Future-ready function for sequence optimization including operator fusion.
    Currently performs pattern recognition without transformation, as operator
    fusion would require composite glyph definitions.

    Parameters
    ----------
    sequence : list[Glyph] | list[str]
        Original glyph sequence
    allow_fusion : bool, default=True
        Whether to fuse compatible operators (e.g., AL+IL → AL_IL composite).
        Currently not implemented; reserved for future optimization.

    Returns
    -------
    list[Glyph] | list[str]
        Optimized sequence (currently unchanged from input)

    Examples
    --------
    >>> from tnfr.operators.grammar import optimize_coherence_sequence
    >>> from tnfr.types import Glyph
    >>> seq = [Glyph.AL, Glyph.IL, Glyph.SHA]
    >>> optimized = optimize_coherence_sequence(seq)
    >>> optimized == seq  # Currently no transformation
    True

    Notes
    -----
    **Future Optimization Capabilities** (when composite glyphs are defined):

    - **Fusion**: Combine compatible operators into composite glyphs
      - `emission + coherence → EMISSION_COHERENCE` (safe_activation composite)
      - `reception + coherence → RECEPTION_COHERENCE` (stable_integration composite)

    - **Anti-pattern Resolution**: Automatically fix detected anti-patterns
      - `silence + coherence → silence + emission + coherence` (reactivation with proper grammar)

    - **Redundancy Elimination**: Remove unnecessary repetitions
      - `coherence + coherence → coherence` (single coherence sufficient)

    **Current Behavior**:
    This function currently performs pattern recognition only via
    :func:`recognize_coherence_sequences` and returns the original sequence unchanged.
    This preserves backward compatibility while establishing the API for future
    optimization features.
    """
    if not allow_fusion:
        return sequence

    # Recognize patterns (for logging/telemetry, though not used for transformation yet)
    _ = recognize_coherence_sequences(sequence)

    # Future: Implement fusion logic here when composite glyphs are defined
    # For now, return sequence unchanged
    return sequence


# Maintain backwards compatibility alias
optimize_il_sequence = optimize_coherence_sequence


def suggest_coherence_sequence(
    current_state: dict[str, float],
    goal_state: dict[str, Any],
) -> list[str]:
    """Suggest glyph sequence to reach goal involving coherence.

    Provides intelligent sequence suggestions based on current node state and
    desired outcome. Suggests canonical coherence sequences when appropriate for the
    structural transformation required.

    Parameters
    ----------
    current_state : dict[str, float]
        Current node state with keys:
        - "epi": Primary Information Structure magnitude
        - "dnfr": Internal reorganization gradient
        - "vf": Structural frequency (Hz_str)
        - "theta": Phase (optional)
    goal_state : dict[str, Any]
        Desired state properties with keys:
        - "dnfr_target": Target ΔNFR level ("low", "moderate", "high")
        - "phase_change": Boolean indicating phase transformation needed
        - "consolidate": Boolean indicating need for coherence lock
        - "reactivate": Boolean indicating reactivation from silence

    Returns
    -------
    list[str]
        Suggested glyph sequence as operator names

    Examples
    --------
    >>> from tnfr.operators.grammar import suggest_coherence_sequence
    >>> current = {"epi": 0.2, "dnfr": 0.15, "vf": 0.85}
    >>> goal = {"dnfr_target": "low", "consolidate": True}
    >>> suggest_coherence_sequence(current, goal)
    ['emission', 'coherence']

    >>> current = {"epi": 0.5, "dnfr": 0.8, "vf": 1.0}
    >>> goal = {"dnfr_target": "low"}
    >>> suggest_coherence_sequence(current, goal)
    ['dissonance', 'coherence']

    >>> current = {"epi": 0.0, "dnfr": 0.0, "vf": 0.02}
    >>> goal = {"reactivate": True, "dnfr_target": "low"}
    >>> suggest_coherence_sequence(current, goal)
    ['emission', 'coherence']

    Notes
    -----
    **Suggestion Logic**:

    1. **Activation Check**: If EPI ≈ 0, suggests emission first
    2. **ΔNFR Management**: Suggests sequences based on current ΔNFR
       - High ΔNFR → OZ + IL (creative resolution)
       - Moderate → Direct IL (consolidation)
       - Low → Consider expansion before stabilization
    3. **Phase Changes**: If goal requires phase transformation, suggests IL → ZHIR
    4. **Reactivation**: From silence, suggests AL → IL (not SHA → IL anti-pattern)

    **Canonical Coherence Sequences Used**:
    - emission → coherence (safe_activation): Node initialization with stabilization
    - reception → coherence (stable_integration): External input consolidation
    - dissonance → coherence (creative_resolution): High ΔNFR resolution
    - resonance → coherence (resonance_consolidation): Network coherence lock
    - coherence → mutation (stable_transformation): Prepared phase change
    """
    sequence: list[str] = []

    # Check for reactivation from silence (anti-pattern silence → coherence)
    if goal_state.get("reactivate", False):
        # Use emission → coherence instead of silence → coherence (grammar-compliant reformulation)
        sequence.append(EMISSION)
        if goal_state.get("consolidate", True):
            sequence.append(COHERENCE)
        return sequence

    # If node is inactive or very low EPI, start with activation
    epi = current_state.get("epi", 0)
    if epi < 0.1:
        sequence.append(EMISSION)

    # Determine ΔNFR management strategy
    dnfr = current_state.get("dnfr", 0)
    dnfr_target = goal_state.get("dnfr_target", "moderate")

    if dnfr_target == "low":
        # Goal is low ΔNFR (stability)
        if dnfr > 0.7:
            # Very high ΔNFR: use dissonance → coherence (creative resolution)
            sequence.extend([DISSONANCE, COHERENCE])
        elif dnfr > 0.3:
            # Moderate ΔNFR: direct coherence (consolidation)
            sequence.append(COHERENCE)
        else:
            # Already low: may need expansion before stabilization
            if goal_state.get("expand_before_stabilize", False):
                sequence.extend([EXPANSION, COHERENCE])
            else:
                sequence.append(COHERENCE)

    elif dnfr_target == "moderate":
        # Maintain moderate ΔNFR
        if dnfr < 0.2:
            # Too low: introduce controlled challenge
            sequence.extend([DISSONANCE, COHERENCE])
        else:
            # Already moderate: just stabilize
            sequence.append(COHERENCE)

    # If goal includes phase transformation
    if goal_state.get("phase_change", False):
        # Use coherence → mutation pattern (stable_transformation)
        # This generates CAUTION warning but is grammar-compliant
        if COHERENCE not in sequence:
            sequence.append(COHERENCE)
        sequence.append(MUTATION)

    # If goal is consolidation without further operations
    if goal_state.get("consolidate", False) and not sequence:
        sequence.append(COHERENCE)

    return sequence


# Maintain backwards compatibility alias
suggest_il_sequence = suggest_coherence_sequence


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
        raise TypeError(
            "validate_sequence_with_health() missing required argument: 'names'"
        )

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

    # Validate glyph is known (compatibility tables deprecated)
    if not isinstance(cand, Glyph) or cand not in GLYPH_TO_FUNCTION:
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
    """Apply glyph with grammar-based validation and coherence sequence recognition.

    This function is the canonical entry point for applying structural operators through
    the grammar layer. It enforces TNFR grammar rules, validates operator preconditions,
    and tracks operator sequences for telemetry.

    Enhanced to recognize and log canonical coherence sequences when they
    occur in the glyph history. This enables pattern detection, telemetry, and
    future optimization of common structural patterns.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing TNFR nodes
    nodes : Optional[Iterable[NodeId | NodeProtocol]]
        Target nodes (all nodes if None)
    glyph : Glyph | str
        Glyph to apply (e.g., Glyph.SHA for Silence operator)
    window : Optional[int]
        Hysteresis window (uses graph default if None)

    Notes
    -----
    **Structural Operator Effects:**

    This function delegates to operator-specific implementations that enforce
    TNFR canonical behavior. Key operator effects include:

    - **SHA (Silence)**: Reduces νf → νf_min ≈ 0, causing ∂EPI/∂t → 0
      (structural evolution freezes). EPI preserved intact regardless of ΔNFR.
      Implements structural silence for memory consolidation and protective latency.

    - **AL (Emission)**: Increases νf and initiates positive ΔNFR for pattern activation

    - **IL (Coherence)**: Reduces |ΔNFR| and stabilizes EPI form

    - **OZ (Dissonance)**: Increases |ΔNFR| for exploration and bifurcation

    See operator definitions in ``tnfr.operators.definitions`` for complete
    canonical behavior specifications.

    **Coherence Sequence Recognition:**

    When a glyph is applied, the function checks if it forms a canonical coherence
    sequence with the previous glyph in the node's history. Recognized patterns
    are logged in ``G.graph["recognized_coherence_patterns"]`` for:

    - Telemetry and analysis
    - Pattern-based optimization hints
    - Structural sequence quality assessment
    - Anti-pattern detection and warnings

    Canonical coherence sequences recognized:
    - emission → coherence (safe_activation)
    - reception → coherence (stable_integration)
    - dissonance → coherence (creative_resolution)
    - resonance → coherence (resonance_consolidation)
    - coherence → mutation (stable_transformation, generates CAUTION warning)

    Anti-patterns detected:
    - coherence → silence (info: potentially redundant)
    - coherence → coherence (warning: no structural purpose)
    - silence → coherence (error: use silence → emission → coherence instead)
    """
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

        # Get node's glyph history before applying new glyph
        nd = G.nodes[node_id]
        glyph_history = nd.get("glyph_history", ())

        # Check if current glyph forms canonical IL sequence with previous glyph
        if glyph_history:
            # Get last applied glyph
            last_glyph = glyph_history[-1]

            # Convert to Glyph enum for pattern matching
            current_glyph_obj = function_name_to_glyph(g_str)
            last_glyph_obj = (
                last_glyph
                if isinstance(last_glyph, Glyph)
                else function_name_to_glyph(last_glyph)
            )

            if current_glyph_obj is not None and last_glyph_obj is not None:
                # Build 2-element sequence for recognition
                recent_sequence = [last_glyph_obj, current_glyph_obj]
                recognized = recognize_coherence_sequences(recent_sequence)

                if recognized:
                    # Log recognized patterns in graph metadata
                    if "recognized_coherence_patterns" not in G.graph:
                        G.graph["recognized_coherence_patterns"] = []

                    for pattern_info in recognized:
                        # Add node and timestamp information
                        pattern_info["node"] = node_id
                        pattern_info["timestamp"] = len(
                            glyph_history
                        )  # Position in history
                        G.graph["recognized_coherence_patterns"].append(pattern_info)

                        # Log recognized canonical sequence (not anti-patterns, those warn separately)
                        if not pattern_info.get("is_antipattern", False):
                            logger.info(
                                f"Recognized coherence sequence '{pattern_info['pattern_name']}' "
                                f"at node {node_id}: {pattern_info['description']}"
                            )

        try:
            g_eff = enforce_canonical_grammar(G, node_id, g_str, ctx)
        except StructuralGrammarError as err:

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

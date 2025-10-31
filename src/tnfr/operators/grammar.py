"""Canonical grammar and sequence validation for structural operators."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..config.operator_names import (
    COHERENCE,
    INTERMEDIATE_OPERATORS,
    RECEPTION,
    SELF_ORGANIZATION,
    SELF_ORGANIZATION_CLOSURES,
    VALID_END_OPERATORS,
    VALID_START_OPERATORS,
    canonical_operator_name,
    operator_display_name,
)
from ..constants import DEFAULTS, get_param
from ..types import Glyph, NodeId, TNFRGraph
from ..validation import rules as _rules
from ..validation.compatibility import CANON_COMPAT
from ..validation.soft_filters import soft_grammar_filters
from .registry import OPERATORS

__all__ = [
    "GrammarContext",
    "SequenceSyntaxError",
    "SequenceValidationResult",
    "_gram_state",
    "apply_glyph_with_grammar",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "parse_sequence",
    "validate_sequence",
]


@dataclass(slots=True)
class GrammarContext:
    """Shared context for grammar helpers."""

    G: TNFRGraph
    cfg_soft: dict[str, Any]
    cfg_canon: dict[str, Any]
    norms: dict[str, Any]

    @classmethod
    def from_graph(cls, G: TNFRGraph) -> "GrammarContext":
        return cls(
            G=G,
            cfg_soft=G.graph.get("GRAMMAR", DEFAULTS.get("GRAMMAR", {})),
            cfg_canon=G.graph.get("GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {})),
            norms=G.graph.get("_sel_norms") or {},
        )


def _gram_state(nd: dict[str, Any]) -> dict[str, Any]:
    return nd.setdefault("_GRAM", {"thol_open": False, "thol_len": 0})


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


@dataclass(slots=True)
class SequenceValidationResult:
    """Structured report emitted by :func:`validate_sequence`."""

    tokens: tuple[str, ...]
    canonical_tokens: tuple[str, ...]
    passed: bool
    message: str
    metadata: Mapping[str, object]
    error: SequenceSyntaxError | None = None

    @property
    def summary(self) -> Mapping[str, object]:
        base: dict[str, object] = {
            "message": self.message,
            "passed": self.passed,
            "tokens": self.tokens,
        }
        if self.metadata:
            base["metadata"] = self.metadata
        if self.error is not None:
            base["error"] = {"index": self.error.index, "token": self.error.token}
        return base

    @property
    def artifacts(self) -> Mapping[str, object]:
        return {"canonical_tokens": self.canonical_tokens}

    @property
    def subject(self) -> tuple[str, ...]:
        return self.canonical_tokens


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
    )

    def __init__(self) -> None:
        self._canonical: list[str] = []
        self._found_reception = False
        self._found_coherence = False
        self._seen_intermediate = False
        self._open_thol = False
        self._unknown_tokens: set[str] = set()

    def run(self, names: Sequence[str]) -> None:
        if not names:
            raise SequenceSyntaxError(index=-1, token=None, message="empty sequence")
        for index, token in enumerate(names):
            self._consume(token, index)
        self._finalize(names)

    def _consume(self, token: str, index: int) -> None:
        if not isinstance(token, str):
            raise SequenceSyntaxError(index=index, token=token, message="tokens must be str")
        canonical = canonical_operator_name(token)
        self._canonical.append(canonical)
        if canonical not in OPERATORS:
            self._unknown_tokens.add(token)
        if index == 0:
            if canonical not in VALID_START_OPERATORS:
                expected = _format_token_group(_CANONICAL_START)
                raise SequenceSyntaxError(index=index, token=token, message=f"must start with {expected}")
        if canonical == RECEPTION and not self._found_reception:
            self._found_reception = True
        elif self._found_reception and canonical == COHERENCE and not self._found_coherence:
            self._found_coherence = True
        elif self._found_coherence and canonical in INTERMEDIATE_OPERATORS:
            self._seen_intermediate = True
        if canonical == SELF_ORGANIZATION:
            self._open_thol = True
        elif self._open_thol and canonical in SELF_ORGANIZATION_CLOSURES:
            self._open_thol = False

    def _finalize(self, names: Sequence[str]) -> None:
        if self._unknown_tokens:
            ordered = ", ".join(sorted(self._unknown_tokens))
            raise SequenceSyntaxError(
                index=len(names) - 1,
                token=names[-1],
                message=f"unknown tokens: {ordered}",
            )
        if not (self._found_reception and self._found_coherence):
            raise SequenceSyntaxError(
                index=-1,
                token=None,
                message=f"missing {RECEPTION}→{COHERENCE} segment",
            )
        if not self._seen_intermediate:
            intermediate = _format_token_group(_CANONICAL_INTERMEDIATE)
            raise SequenceSyntaxError(
                index=-1,
                token=None,
                message=f"missing {intermediate} segment",
            )
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
                message="THOL block without closure",
            )

    @property
    def canonical(self) -> tuple[str, ...]:
        return tuple(self._canonical)

    def metadata(self) -> Mapping[str, object]:
        return {
            "has_reception": self._found_reception,
            "has_coherence": self._found_coherence,
            "has_intermediate": self._seen_intermediate,
            "open_thol": self._open_thol,
            "unknown_tokens": frozenset(self._unknown_tokens),
        }


_MISSING = object()


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


def validate_sequence(names: Iterable[str] | object = _MISSING, **kwargs: object) -> SequenceValidationResult:
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"validate_sequence() got unexpected keyword argument(s): {unexpected}"
        )
    if names is _MISSING:
        raise TypeError("validate_sequence() missing required argument: 'names'")
    return _analyse_sequence(names)


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
    if ctx is None:
        ctx = GrammarContext.from_graph(G)

    nd = ctx.G.nodes[n]
    st = _gram_state(nd)

    raw_cand = cand
    cand = _rules.coerce_glyph(cand)
    input_was_str = isinstance(raw_cand, str)

    if not isinstance(cand, Glyph) or cand not in CANON_COMPAT:
        return raw_cand if input_was_str else cand

    cand = soft_grammar_filters(ctx, n, cand)
    cand = _rules._check_oz_to_zhir(ctx, n, cand)
    cand = _rules._check_thol_closure(ctx, n, cand, st)
    cand = _rules._check_compatibility(ctx, n, cand)

    coerced_final = _rules.coerce_glyph(cand)
    if input_was_str:
        if isinstance(coerced_final, Glyph):
            return coerced_final.value
        return str(cand)
    return coerced_final if isinstance(coerced_final, Glyph) else cand


def on_applied_glyph(G: TNFRGraph, n: NodeId, applied: Glyph | str) -> None:
    nd = G.nodes[n]
    st = _gram_state(nd)
    try:
        glyph = applied if isinstance(applied, Glyph) else Glyph(str(applied))
    except ValueError:
        glyph = None

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
    if window is None:
        window = get_param(G, "GLYPH_HYSTERESIS_WINDOW")

    g_str = glyph.value if isinstance(glyph, Glyph) else str(glyph)
    iter_nodes = G.nodes() if nodes is None else nodes
    ctx = GrammarContext.from_graph(G)
    from . import apply_glyph as _apply_glyph

    for node_ref in iter_nodes:
        node_id = node_ref.n if hasattr(node_ref, "n") else node_ref
        g_eff = enforce_canonical_grammar(G, node_id, g_str, ctx)
        _apply_glyph(G, node_id, g_eff, window=window)
        on_applied_glyph(G, node_id, g_eff)

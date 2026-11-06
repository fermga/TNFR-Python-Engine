"""Passive observations of one supplied native decision occasion.

These frozen records are reports, not authenticated execution certificates.
They retain actual evaluated branches, not uncomputed counterfactual choices.
Opaque node identities and custom tokens are retained as supplied objects;
freezing the record does not recursively freeze those external objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NativeSelectorObservation:
    """Consumed branch of a builtin selector, or an opaque custom proposal."""

    kind: str
    selected: Any = None
    normalized_metrics: tuple[float, float, float] | None = None
    thresholds: tuple[tuple[str, float], ...] = ()
    base_candidate: Any = None
    hysteresis_candidate: Any = None
    margin: float | None = None
    score: float | None = None
    override_candidate: Any = None
    soft_filter_candidate: Any = None
    acceleration_source: str | None = None
    acceleration_window: tuple[Any, Any] | None = None


@dataclass(frozen=True, slots=True)
class NativeOperatorDecision:
    """One target's observed proposal, admission and direct-glyph outcome.

    ``applied_glyph`` is populated only after successful primitive dispatch. A
    failed dispatch can have partial side effects; ``None`` does not certify no
    effect. ``not_reached`` may retain a previously computed proposal. Lag counts
    are the once-incremented counts consumed to choose the request. Preflight
    failures before admitted targets produce no target observations. ``abstained``
    identifies a gate-blocked ZHIR request under the explicit skip policy;
    ``skipped`` instead identifies a selector that supplied no request.
    """

    node: Any
    index: int
    selector_kind: str
    status: str = "not_reached"
    selector_evaluated: bool | None = None
    selection: NativeSelectorObservation | None = None
    lag_reason: str | None = None
    since_al: int | None = None
    since_en: int | None = None
    al_max: int | None = None
    en_max: int | None = None
    requested_glyph: Any = None
    mutation_certificate: Any = None
    mutation_abstention_reason: str | None = None
    mutation_occurrence_policy: str | None = None
    mutation_interval: tuple[str, float, float] | None = None
    mutation_interval_previously_consumed: bool | None = None
    mutation_interval_committed: bool = False
    physical_history_invalidated: bool | None = None
    post_gate_glyph: Any = None
    grammar_decision: Any = None
    applied_glyph: Any = None
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    mutation_abstention_policy: str | None = None

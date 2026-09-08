"""Immutable advisory proposal shared by direct and staged Recursivity.

The canonical node-level REMESH glyph is an advisory. It records one
graph-level request per telemetry step and leaves every structural channel
unchanged. Explicit delayed EPI mixing remains the separate
apply_network_remesh network operation.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .. import glyph_history
from ..errors import TNFRValueError

REMESH_ADVISORY_MESSAGE = (
    "REMESH operates at network scale. Use apply_remesh_if_globally_"
    "stable(G) or apply_network_remesh(G)."
)
REMESH_WARNING_STEP_KEY = "_remesh_warn_step"


def _graph_data(subject: Any) -> MutableMapping[str, Any]:
    graph_data = getattr(subject, "graph", subject)
    if not isinstance(graph_data, MutableMapping):
        raise TNFRValueError("Recursivity requires mutable graph metadata")
    return graph_data


def _read_warning_step(
    graph_data: Mapping[str, Any],
) -> tuple[bool, int | None]:
    if REMESH_WARNING_STEP_KEY not in graph_data:
        return False, None
    raw = graph_data[REMESH_WARNING_STEP_KEY]
    if isinstance(raw, bool) or not isinstance(raw, Integral) or raw < 0:
        raise TNFRValueError(
            "Recursivity warning step must be a nonnegative integer",
            context={REMESH_WARNING_STEP_KEY: repr(raw)},
        )
    return True, int(raw)


@dataclass(frozen=True, slots=True)
class RecursivityAdvisoryProposal:
    """One immutable graph-level REMESH advisory transition."""

    step: int
    had_warning_step: bool
    warning_step_before: int | None
    message: str = REMESH_ADVISORY_MESSAGE

    @property
    def emits_advisory(self) -> bool:
        """Whether this proposal adds the step-level advisory event."""

        return (
            not self.had_warning_step
            or self.warning_step_before != self.step
        )


def propose_recursivity_advisory(subject: Any) -> RecursivityAdvisoryProposal:
    """Read one graph snapshot and propose its deduplicated advisory."""

    graph_data = _graph_data(subject)
    had_step, prior_step = _read_warning_step(graph_data)
    return RecursivityAdvisoryProposal(
        step=glyph_history.current_step_idx(subject),
        had_warning_step=had_step,
        warning_step_before=prior_step,
    )


def commit_recursivity_advisory(
    subject: Any, proposal: RecursivityAdvisoryProposal
) -> None:
    """Commit a non-stale advisory proposal exactly once."""

    graph_data = _graph_data(subject)
    had_step, prior_step = _read_warning_step(graph_data)
    if (
        had_step != proposal.had_warning_step
        or prior_step != proposal.warning_step_before
    ):
        raise RuntimeError("stale Recursivity advisory proposal")
    if not proposal.emits_advisory:
        return

    history = glyph_history.ensure_history(subject)
    if not isinstance(history, MutableMapping):
        raise TNFRValueError(
            "Recursivity requires graph history to be a mutable mapping"
        )
    glyph_history.append_metric(
        history,
        "events",
        (
            "warn",
            {"step": proposal.step, "node": None, "msg": proposal.message},
        ),
    )
    graph_data[REMESH_WARNING_STEP_KEY] = proposal.step


__all__ = [
    "REMESH_ADVISORY_MESSAGE",
    "REMESH_WARNING_STEP_KEY",
    "RecursivityAdvisoryProposal",
    "commit_recursivity_advisory",
    "propose_recursivity_advisory",
]
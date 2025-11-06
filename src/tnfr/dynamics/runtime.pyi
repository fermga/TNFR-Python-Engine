from __future__ import annotations

from ..types import HistoryState, TNFRGraph
from .aliases import (
    ALIAS_DNFR as ALIAS_DNFR,
    ALIAS_EPI as ALIAS_EPI,
    ALIAS_SI as ALIAS_SI,
    ALIAS_VF as ALIAS_VF,
)
from collections.abc import Mapping
from typing import Any

__all__ = [
    "ALIAS_VF",
    "ALIAS_DNFR",
    "ALIAS_EPI",
    "ALIAS_SI",
    "_normalize_job_overrides",
    "_resolve_jobs_override",
    "_prepare_dnfr",
    "_update_nodes",
    "_update_epi_hist",
    "_maybe_remesh",
    "_run_validators",
    "_run_before_callbacks",
    "_run_after_callbacks",
    "step",
    "run",
]

def _normalize_job_overrides(
    job_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]: ...
def _resolve_jobs_override(
    overrides: Mapping[str, Any],
    key: str,
    graph_value: Any,
    *,
    allow_non_positive: bool,
) -> int | None: ...
def _run_before_callbacks(
    G: TNFRGraph, *, step_idx: int, dt: float | None, use_Si: bool, apply_glyphs: bool
) -> None: ...
def _prepare_dnfr(
    G: TNFRGraph, *, use_Si: bool, job_overrides: Mapping[str, Any] | None = None
) -> None: ...
def _update_nodes(
    G: TNFRGraph,
    *,
    dt: float | None,
    use_Si: bool,
    apply_glyphs: bool,
    step_idx: int,
    hist: HistoryState,
    job_overrides: Mapping[str, Any] | None = None,
) -> None: ...
def _update_epi_hist(G: TNFRGraph) -> None: ...
def _maybe_remesh(G: TNFRGraph) -> None: ...
def _run_validators(G: TNFRGraph) -> None: ...
def _run_after_callbacks(G, *, step_idx: int) -> None: ...
def step(
    G: TNFRGraph,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
    n_jobs: Mapping[str, Any] | None = None,
) -> None: ...
def run(
    G: TNFRGraph,
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
    n_jobs: Mapping[str, Any] | None = None,
) -> None: ...

from __future__ import annotations

from collections import deque
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..types import GraphLike

__all__ = (
    "NuFWindow",
    "NuFSnapshot",
    "NuFTelemetryAccumulator",
    "ensure_nu_f_telemetry",
    "record_nu_f_window",
)

@dataclass
class NuFWindow:
    reorganisations: int
    duration: float
    start: float | None = ...
    end: float | None = ...

    def __post_init__(self) -> None: ...
    @classmethod
    def from_bounds(cls, reorganisations: int, start: float, end: float) -> NuFWindow: ...
    def as_payload(self) -> Mapping[str, float | int | None]: ...

@dataclass
class NuFSnapshot:
    windows: Sequence[NuFWindow]
    total_reorganisations: int
    total_duration: float
    rate_hz_str: float | None
    rate_hz: float | None
    variance_hz_str: float | None
    variance_hz: float | None
    confidence_level: float | None
    ci_lower_hz_str: float | None
    ci_upper_hz_str: float | None
    ci_lower_hz: float | None
    ci_upper_hz: float | None

    def as_payload(self) -> dict[str, Any]: ...

class NuFTelemetryAccumulator:
    _windows: deque[NuFWindow]

    def __init__(
        self,
        *,
        confidence_level: float = ...,
        history_limit: int | None = ...,
        window_limit: int | None = ...,
        graph: GraphLike | MutableMapping[str, Any] | None = ...,
    ) -> None: ...
    @property
    def confidence_level(self) -> float: ...
    @property
    def history_limit(self) -> int | None: ...
    @property
    def window_limit(self) -> int | None: ...
    def attach_graph(self, graph: GraphLike | MutableMapping[str, Any] | None) -> None: ...
    def record_window(
        self,
        window: NuFWindow,
        *,
        graph: GraphLike | MutableMapping[str, Any] | None = ...,
    ) -> NuFSnapshot: ...
    def record_counts(
        self,
        reorganisations: int,
        duration: float,
        *,
        start: float | None = ...,
        end: float | None = ...,
        graph: GraphLike | MutableMapping[str, Any] | None = ...,
    ) -> NuFSnapshot: ...
    def reset(self) -> None: ...
    def snapshot(
        self,
        *,
        graph: GraphLike | MutableMapping[str, Any] | None = ...,
    ) -> NuFSnapshot: ...

def ensure_nu_f_telemetry(
    graph: GraphLike,
    *,
    confidence_level: float | None = ...,
    history_limit: int | None = ...,
    window_limit: int | None = ...,
) -> NuFTelemetryAccumulator: ...
def record_nu_f_window(
    graph: GraphLike,
    reorganisations: int,
    duration: float,
    *,
    start: float | None = ...,
    end: float | None = ...,
    confidence_level: float | None = ...,
    history_limit: int | None = ...,
    window_limit: int | None = ...,
) -> NuFSnapshot: ...

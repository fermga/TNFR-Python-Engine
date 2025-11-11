"""Structural frequency (νf) telemetry estimators.

This module aggregates discrete reorganisation counts observed over
time windows and exposes Poisson maximum likelihood estimators (MLE) for
the structural frequency νf.  Results are provided both in canonical
``Hz_str`` and converted ``Hz`` using :mod:`tnfr.units`, allowing callers
to surface telemetry without duplicating conversion logic.

Snapshots emitted by :class:`NuFTelemetryAccumulator` are appended to the
``G.graph["telemetry"]["nu_f_history"]`` channel so downstream observers
and structured logging hooks can consume them without interfering with
runtime summaries stored under ``G.graph["telemetry"]["nu_f"]``.
"""

from __future__ import annotations

import math
import weakref
from collections import deque
from collections.abc import MutableMapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Deque, Mapping

from ..types import GraphLike
from ..units import get_hz_bridge, hz_str_to_hz

__all__ = (
    "NuFWindow",
    "NuFSnapshot",
    "NuFTelemetryAccumulator",
    "ensure_nu_f_telemetry",
    "record_nu_f_window",
)


@dataclass(frozen=True)
class NuFWindow:
    """Discrete reorganisation observations captured over a time window."""

    reorganisations: int
    """Number of reorganisations counted within the window."""

    duration: float
    """Duration of the window expressed in structural time units."""

    start: float | None = None
    """Optional inclusive window start timestamp."""

    end: float | None = None
    """Optional exclusive window end timestamp."""

    def __post_init__(self) -> None:
        reorganisations = int(self.reorganisations)
        duration = float(self.duration)
        object.__setattr__(self, "reorganisations", reorganisations)
        object.__setattr__(self, "duration", duration)
        if reorganisations < 0:
            raise ValueError("reorganisations must be non-negative")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be a positive finite number")
        if self.start is not None and self.end is not None:
            start = float(self.start)
            end = float(self.end)
            object.__setattr__(self, "start", start)
            object.__setattr__(self, "end", end)
            if end < start:
                raise ValueError("end must be greater than or equal to start")
            window = end - start
            if window <= 0.0:
                raise ValueError("start and end must describe a non-empty window")
            # Allow minor numerical discrepancies when duration is supplied
            # independently from ``start``/``end``.
            if not math.isclose(window, duration, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    "duration does not match the difference between start and end",
                )

    @classmethod
    def from_bounds(cls, reorganisations: int, start: float, end: float) -> "NuFWindow":
        """Construct a window inferring the duration from ``start``/``end``."""

        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            raise ValueError("end must be greater than start")
        return cls(
            reorganisations=int(reorganisations),
            duration=end_f - start_f,
            start=start_f,
            end=end_f,
        )

    def as_payload(self) -> Mapping[str, float | int | None]:
        """Return a JSON-serialisable representation of the window."""

        return {
            "reorganisations": int(self.reorganisations),
            "duration": float(self.duration),
            "start": float(self.start) if self.start is not None else None,
            "end": float(self.end) if self.end is not None else None,
        }


@dataclass(frozen=True)
class NuFSnapshot:
    """Aggregate νf estimates computed from recorded windows."""

    windows: tuple[NuFWindow, ...]
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

    def as_payload(self) -> dict[str, Any]:
        """Return a structured representation suitable for telemetry sinks."""

        return {
            "windows": [window.as_payload() for window in self.windows],
            "total_reorganisations": self.total_reorganisations,
            "total_duration": self.total_duration,
            "rate_hz_str": self.rate_hz_str,
            "rate_hz": self.rate_hz,
            "variance_hz_str": self.variance_hz_str,
            "variance_hz": self.variance_hz,
            "confidence_level": self.confidence_level,
            "ci_lower_hz_str": self.ci_lower_hz_str,
            "ci_upper_hz_str": self.ci_upper_hz_str,
            "ci_lower_hz": self.ci_lower_hz,
            "ci_upper_hz": self.ci_upper_hz,
        }


class NuFTelemetryAccumulator:
    """Accumulate reorganisation telemetry and produce νf estimates."""

    def __init__(
        self,
        *,
        confidence_level: float = 0.95,
        history_limit: int | None = 128,
        window_limit: int | None = None,
        graph: GraphLike | MutableMapping[str, Any] | None = None,
    ) -> None:
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be in the open interval (0, 1)")
        if history_limit is not None and history_limit <= 0:
            raise ValueError("history_limit must be positive when provided")
        if window_limit is not None and window_limit <= 0:
            raise ValueError("window_limit must be positive when provided")

        self._confidence_level = float(confidence_level)
        self._history_limit = history_limit
        self._window_limit = window_limit
        self._windows: Deque[NuFWindow] = deque()
        self._total_reorganisations = 0
        self._total_duration = 0.0
        self._graph_ref: weakref.ReferenceType[GraphLike | MutableMapping[str, Any]] | None = None
        self.attach_graph(graph)

    @property
    def confidence_level(self) -> float:
        """Return the configured confidence level for interval estimation."""

        return self._confidence_level

    @property
    def history_limit(self) -> int | None:
        """Return the maximum number of snapshots retained on the graph."""

        return self._history_limit

    @property
    def window_limit(self) -> int | None:
        """Return the maximum number of windows stored in memory."""

        return self._window_limit

    def attach_graph(self, graph: GraphLike | MutableMapping[str, Any] | None) -> None:
        """Attach ``graph`` for unit conversions and telemetry persistence."""

        if graph is None:
            return
        try:
            self._graph_ref = weakref.ref(graph)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover - mapping instances are not weakrefable
            self._graph_ref = None

    def _resolve_graph(
        self,
    ) -> GraphLike | MutableMapping[str, Any] | None:
        return self._graph_ref() if self._graph_ref is not None else None

    def _coerce_window(self, window: NuFWindow) -> None:
        if self._window_limit is not None and len(self._windows) >= self._window_limit:
            removed = self._windows.popleft()
            self._total_reorganisations -= removed.reorganisations
            self._total_duration -= removed.duration
        self._windows.append(window)
        self._total_reorganisations += window.reorganisations
        self._total_duration += window.duration

    def record_window(
        self,
        window: NuFWindow,
        *,
        graph: GraphLike | MutableMapping[str, Any] | None = None,
    ) -> NuFSnapshot:
        """Record ``window`` and return the updated telemetry snapshot."""

        self._coerce_window(window)
        graph_obj = graph or self._resolve_graph()
        snapshot = self.snapshot(graph=graph_obj)
        self._persist_snapshot(snapshot, graph_obj)
        return snapshot

    def record_counts(
        self,
        reorganisations: int,
        duration: float,
        *,
        start: float | None = None,
        end: float | None = None,
        graph: GraphLike | MutableMapping[str, Any] | None = None,
    ) -> NuFSnapshot:
        """Record a window described by ``reorganisations`` and ``duration``."""

        window = NuFWindow(
            reorganisations=int(reorganisations),
            duration=float(duration),
            start=float(start) if start is not None else None,
            end=float(end) if end is not None else None,
        )
        return self.record_window(window, graph=graph)

    def reset(self) -> None:
        """Clear accumulated windows and totals."""

        self._windows.clear()
        self._total_reorganisations = 0
        self._total_duration = 0.0

    def _normal_dist(self) -> NormalDist:
        return NormalDist()

    def _graph_mapping(
        self, graph: GraphLike | MutableMapping[str, Any] | None
    ) -> MutableMapping[str, Any] | None:
        if graph is None:
            return None
        if isinstance(graph, MutableMapping):
            return graph
        graph_data = getattr(graph, "graph", None)
        return graph_data if isinstance(graph_data, MutableMapping) else None

    def snapshot(
        self,
        *,
        graph: GraphLike | MutableMapping[str, Any] | None = None,
    ) -> NuFSnapshot:
        """Return a νf telemetry snapshot without mutating internal state."""

        total_duration = self._total_duration
        total_reorganisations = self._total_reorganisations
        windows = tuple(self._windows)

        if total_duration <= 0.0:
            rate_hz_str = None
            variance_hz_str = None
            ci_lower_str = None
            ci_upper_str = None
            confidence_level: float | None = None
        else:
            rate_hz_str = total_reorganisations / total_duration
            variance_hz_str = rate_hz_str / total_duration
            std_error = math.sqrt(variance_hz_str)
            z = self._normal_dist().inv_cdf(0.5 + (self._confidence_level / 2.0))
            ci_lower_str = max(rate_hz_str - z * std_error, 0.0)
            ci_upper_str = rate_hz_str + z * std_error
            confidence_level = self._confidence_level

        graph_obj = graph or self._resolve_graph()
        rate_hz = variance_hz = ci_lower_hz = ci_upper_hz = None
        if rate_hz_str is not None and graph_obj is not None:
            if not isinstance(graph_obj, MutableMapping):
                bridge = get_hz_bridge(graph_obj)
                rate_hz = hz_str_to_hz(rate_hz_str, graph_obj)
                if variance_hz_str is not None:
                    variance_hz = variance_hz_str * (bridge**2)
                if ci_lower_str is not None and ci_upper_str is not None:
                    ci_lower_hz = hz_str_to_hz(ci_lower_str, graph_obj)
                    ci_upper_hz = hz_str_to_hz(ci_upper_str, graph_obj)

        return NuFSnapshot(
            windows=windows,
            total_reorganisations=total_reorganisations,
            total_duration=total_duration,
            rate_hz_str=rate_hz_str,
            rate_hz=rate_hz,
            variance_hz_str=variance_hz_str,
            variance_hz=variance_hz,
            confidence_level=confidence_level,
            ci_lower_hz_str=ci_lower_str,
            ci_upper_hz_str=ci_upper_str,
            ci_lower_hz=ci_lower_hz,
            ci_upper_hz=ci_upper_hz,
        )

    def _persist_snapshot(
        self,
        snapshot: NuFSnapshot,
        graph: GraphLike | MutableMapping[str, Any] | None,
    ) -> None:
        mapping = self._graph_mapping(graph)
        if mapping is None:
            return

        telemetry = mapping.setdefault("telemetry", {})
        if not isinstance(telemetry, MutableMapping):
            telemetry = {}
            mapping["telemetry"] = telemetry
        payload = snapshot.as_payload()
        history_key = "nu_f_history"
        history = telemetry.get(history_key)
        if not isinstance(history, list):
            legacy_history = telemetry.get("nu_f")
            if isinstance(legacy_history, list):
                history = legacy_history
            else:
                history = []
            telemetry[history_key] = history
        history.append(payload)
        if self._history_limit is not None and len(history) > self._history_limit:
            del history[: len(history) - self._history_limit]


_ACCUMULATOR_KEY = "_tnfr_nu_f_accumulator"


def ensure_nu_f_telemetry(
    graph: GraphLike,
    *,
    confidence_level: float | None = None,
    history_limit: int | None = 128,
    window_limit: int | None = None,
) -> NuFTelemetryAccumulator:
    """Ensure ``graph`` exposes a :class:`NuFTelemetryAccumulator`.

    When ``confidence_level`` is ``None`` the existing accumulator is preserved
    and new accumulators default to ``0.95``.
    """

    mapping = getattr(graph, "graph", None)
    if not isinstance(mapping, MutableMapping):
        raise TypeError("graph.graph must be a mutable mapping for telemetry storage")

    accumulator = mapping.get(_ACCUMULATOR_KEY)
    replace = False
    if isinstance(accumulator, NuFTelemetryAccumulator):
        if (
            (
                confidence_level is not None
                and abs(accumulator.confidence_level - confidence_level) > 1e-12
            )
            or (history_limit is not None and accumulator.history_limit != history_limit)
            or (window_limit is not None and accumulator.window_limit != window_limit)
        ):
            replace = True
    if not isinstance(accumulator, NuFTelemetryAccumulator) or replace:
        requested_confidence = 0.95 if confidence_level is None else confidence_level
        accumulator = NuFTelemetryAccumulator(
            confidence_level=requested_confidence,
            history_limit=history_limit,
            window_limit=window_limit,
            graph=graph,
        )
        mapping[_ACCUMULATOR_KEY] = accumulator
    else:
        accumulator.attach_graph(graph)
    return accumulator


def record_nu_f_window(
    graph: GraphLike,
    reorganisations: int,
    duration: float,
    *,
    start: float | None = None,
    end: float | None = None,
    confidence_level: float | None = None,
    history_limit: int | None = None,
    window_limit: int | None = None,
) -> NuFSnapshot:
    """Record a νf observation for ``graph`` and persist the snapshot."""

    kwargs: dict[str, Any] = {}
    if confidence_level is not None:
        kwargs["confidence_level"] = confidence_level
    if history_limit is not None:
        kwargs["history_limit"] = history_limit
    if window_limit is not None:
        kwargs["window_limit"] = window_limit

    accumulator = ensure_nu_f_telemetry(graph, **kwargs)
    return accumulator.record_counts(
        reorganisations,
        duration,
        start=start,
        end=end,
        graph=graph,
    )

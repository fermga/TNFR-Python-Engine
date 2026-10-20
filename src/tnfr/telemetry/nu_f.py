"""Structural frequency (νf) telemetry estimators.

This module aggregates discrete reorganisation counts observed over
time windows and exposes count-rate estimators under a supplied Poisson
observation model, with approximate normal confidence intervals. These
observations do not replace the capacity consumed by a nodal law. Results are
provided both in canonical
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
from fractions import Fraction
from statistics import NormalDist
from typing import Any, Deque, Mapping

from .._exact_time import finite_represented_real
from ..errors import TNFRValueError
from ..types import GraphLike
from ..units import get_hz_bridge, hz_str_to_hz
from ..validation.window import validate_window

__all__ = (
    "NuFWindow",
    "NuFSnapshot",
    "NuFTelemetryAccumulator",
    "ensure_nu_f_telemetry",
    "record_nu_f_window",
)


def _finite_real(value: Any, name: str) -> float:
    """Use shared admission while preserving telemetry's value-error type."""
    try:
        return finite_represented_real(value, name)[0]
    except ValueError as exc:
        raise TNFRValueError(str(exc)) from exc


def _confidence(value: float) -> float:
    value = _finite_real(value, "confidence_level")
    if not 0.0 < value < 1.0:
        raise TNFRValueError("confidence_level must be in the open interval (0, 1)")
    return value


@dataclass(frozen=True)
class NuFWindow:
    """Integer counts over positive represented time, with optional finite bounds."""

    reorganisations: int
    """Number of reorganisations counted within the window."""

    duration: float
    """Duration of the window expressed in structural time units."""

    start: float | None = None
    """Optional inclusive window start timestamp."""

    end: float | None = None
    """Optional exclusive window end timestamp."""

    def __post_init__(self) -> None:
        reorganisations = validate_window(self.reorganisations)
        duration = _finite_real(self.duration, "duration")
        object.__setattr__(self, "reorganisations", reorganisations)
        object.__setattr__(self, "duration", duration)
        if duration <= 0.0:
            raise TNFRValueError(
                "duration must be a positive finite number",
                context={"duration": duration},
            )
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_real(value, name))
        if self.start is not None and self.end is not None:
            start, end = self.start, self.end
            if end < start:
                raise TNFRValueError(
                    "end must be greater than or equal to start",
                    context={"start": start, "end": end},
                )
            window = _finite_real(end - start, "window duration")
            if window <= 0.0:
                raise TNFRValueError(
                    "start and end must describe a non-empty window",
                    context={"window": window},
                )
            # Allow minor numerical discrepancies when duration is supplied
            # independently from ``start``/``end``.
            if not math.isclose(window, duration, rel_tol=1e-9, abs_tol=0.0):
                raise TNFRValueError(
                    "duration does not match the difference between start and end",
                    context={"duration": duration, "window": window},
                )

    @classmethod
    def from_bounds(cls, reorganisations: int, start: float, end: float) -> "NuFWindow":
        """Construct a window inferring the duration from ``start``/``end``."""

        start_f = _finite_real(start, "start")
        end_f = _finite_real(end, "end")
        if end_f <= start_f:
            raise TNFRValueError(
                "end must be greater than start",
                context={"start": start_f, "end": end_f},
            )
        return cls(
            reorganisations=reorganisations,
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
    """Accumulate admitted counts with exact sums of represented durations.

    Invalid evidence or unrepresentable estimates reject before changing the
    observation buffer or history. Window overlap and Poisson independence
    remain the caller's observation premises, not conclusions of this reducer.
    """

    def __init__(
        self,
        *,
        confidence_level: float = 0.95,
        history_limit: int | None = 128,
        window_limit: int | None = None,
        graph: GraphLike | MutableMapping[str, Any] | None = None,
    ) -> None:
        self._confidence_level = _confidence(confidence_level)
        self._history_limit = (
            validate_window(history_limit, positive=True)
            if history_limit is not None
            else None
        )
        self._window_limit = (
            validate_window(window_limit, positive=True)
            if window_limit is not None
            else None
        )
        self._windows: Deque[NuFWindow] = deque()
        self._total_reorganisations = 0
        self._total_duration = Fraction(0)
        self._graph_ref: (
            weakref.ReferenceType[GraphLike | MutableMapping[str, Any]] | None
        ) = None
        self._graph_mapping_ref: MutableMapping[str, Any] | None = None
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
            self._graph_mapping_ref = None
        except TypeError:  # pragma: no cover - mapping instances are not weakrefable
            self._graph_ref = None
            self._graph_mapping_ref = (
                graph if isinstance(graph, MutableMapping) else None
            )

    def _resolve_graph(
        self,
    ) -> GraphLike | MutableMapping[str, Any] | None:
        return (
            self._graph_ref()
            if self._graph_ref is not None
            else self._graph_mapping_ref
        )

    def record_window(
        self,
        window: NuFWindow,
        *,
        graph: GraphLike | MutableMapping[str, Any] | None = None,
    ) -> NuFSnapshot:
        """Record ``window`` and return the updated telemetry snapshot."""

        if not isinstance(window, NuFWindow):
            raise TypeError("window must be a NuFWindow")
        # Re-admit primitive evidence, even for a manually altered frozen record.
        window = NuFWindow(
            window.reorganisations, window.duration, window.start, window.end
        )
        windows = tuple(self._windows)
        total_count = self._total_reorganisations + window.reorganisations
        total_duration = self._total_duration + Fraction.from_float(window.duration)
        if self._window_limit is not None and len(windows) >= self._window_limit:
            removed = windows[0]
            windows = windows[1:]
            total_count -= removed.reorganisations
            total_duration -= Fraction.from_float(removed.duration)
        windows += (window,)
        graph_obj = graph if graph is not None else self._resolve_graph()
        # Validate all result arithmetic before changing the buffer or history.
        snapshot = self._snapshot(windows, total_count, total_duration, graph_obj)
        self._persist_snapshot(snapshot, graph_obj)
        if self._window_limit is not None and len(self._windows) >= self._window_limit:
            self._windows.popleft()
        self._windows.append(window)
        self._total_reorganisations = total_count
        self._total_duration = total_duration
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
            reorganisations=reorganisations,
            duration=duration,
            start=start,
            end=end,
        )
        return self.record_window(window, graph=graph)

    def reset(self) -> None:
        """Clear accumulated windows and totals."""

        self._windows.clear()
        self._total_reorganisations = 0
        self._total_duration = Fraction(0)

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

        return self._snapshot(
            tuple(self._windows),
            self._total_reorganisations,
            self._total_duration,
            graph if graph is not None else self._resolve_graph(),
        )

    def _snapshot(
        self,
        windows: tuple[NuFWindow, ...],
        total_reorganisations: int,
        exact_duration: Fraction,
        graph: GraphLike | MutableMapping[str, Any] | None,
    ) -> NuFSnapshot:
        total_duration = _finite_real(exact_duration, "total duration")

        if total_duration <= 0.0:
            rate_hz_str = None
            variance_hz_str = None
            ci_lower_str = None
            ci_upper_str = None
            confidence_level: float | None = None
        else:
            exact_rate = total_reorganisations / exact_duration
            exact_variance = exact_rate / exact_duration
            rate_hz_str = _finite_real(exact_rate, "estimated rate")
            variance_hz_str = _finite_real(exact_variance, "estimated variance")
            std_error = math.sqrt(variance_hz_str)
            # The lower tail remains representable when confidence is near one.
            z = -self._normal_dist().inv_cdf((1.0 - self._confidence_level) / 2.0)
            ci_lower_str = max(rate_hz_str - z * std_error, 0.0)
            ci_upper_str = _finite_real(
                rate_hz_str + z * std_error, "interval upper bound"
            )
            confidence_level = self._confidence_level

        rate_hz = variance_hz = ci_lower_hz = ci_upper_hz = None
        if rate_hz_str is not None and graph is not None:
            if not isinstance(graph, MutableMapping):
                bridge = get_hz_bridge(graph)
                rate_hz = hz_str_to_hz(rate_hz_str, graph)
                if variance_hz_str is not None:
                    variance_hz = _finite_real(
                        Fraction.from_float(variance_hz_str)
                        * Fraction.from_float(bridge) ** 2,
                        "converted variance",
                    )
                if ci_lower_str is not None and ci_upper_str is not None:
                    ci_lower_hz = hz_str_to_hz(ci_lower_str, graph)
                    ci_upper_hz = hz_str_to_hz(ci_upper_str, graph)

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
_UNSPECIFIED_HISTORY = object()


def _configured_accumulator(
    graph: GraphLike,
    *,
    confidence_level: float | None = None,
    history_limit: Any = _UNSPECIFIED_HISTORY,
    window_limit: int | None = None,
) -> NuFTelemetryAccumulator:
    """Select or prepare an accumulator without installing it on the graph."""

    mapping = getattr(graph, "graph", None)
    if not isinstance(mapping, MutableMapping):
        raise TypeError("graph.graph must be a mutable mapping for telemetry storage")

    if confidence_level is not None:
        confidence_level = _confidence(confidence_level)
    history_specified = history_limit is not _UNSPECIFIED_HISTORY
    if history_specified and history_limit is not None:
        history_limit = validate_window(history_limit, positive=True)
    if window_limit is not None:
        window_limit = validate_window(window_limit, positive=True)
    accumulator = mapping.get(_ACCUMULATOR_KEY)
    replace = False
    if isinstance(accumulator, NuFTelemetryAccumulator):
        if (
            (
                confidence_level is not None
                and accumulator.confidence_level != confidence_level
            )
            or (
                history_specified
                and history_limit is not None
                and accumulator.history_limit != history_limit
            )
            or (window_limit is not None and accumulator.window_limit != window_limit)
        ):
            replace = True
    if not isinstance(accumulator, NuFTelemetryAccumulator) or replace:
        previous = (
            accumulator if isinstance(accumulator, NuFTelemetryAccumulator) else None
        )
        requested_confidence = (
            confidence_level
            if confidence_level is not None
            else previous.confidence_level if previous is not None else 0.95
        )
        requested_history = (
            previous.history_limit
            if previous is not None and (not history_specified or history_limit is None)
            else history_limit if history_specified else 128
        )
        requested_window = (
            previous.window_limit
            if previous is not None and window_limit is None
            else window_limit
        )
        accumulator = NuFTelemetryAccumulator(
            confidence_level=requested_confidence,
            history_limit=requested_history,
            window_limit=requested_window,
            graph=graph,
        )
    return accumulator


def ensure_nu_f_telemetry(
    graph: GraphLike,
    *,
    confidence_level: float | None = None,
    history_limit: int | None = 128,
    window_limit: int | None = None,
) -> NuFTelemetryAccumulator:
    """Ensure ``graph`` exposes a :class:`NuFTelemetryAccumulator`.

    When ``confidence_level`` is ``None`` the existing accumulator is preserved
    and new accumulators default to ``0.95``. Explicit configuration changes
    replace its accumulated windows.
    """
    accumulator = _configured_accumulator(
        graph,
        confidence_level=confidence_level,
        history_limit=history_limit,
        window_limit=window_limit,
    )
    accumulator.attach_graph(graph)
    graph.graph[_ACCUMULATOR_KEY] = accumulator
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

    window = NuFWindow(reorganisations, duration, start, end)
    kwargs: dict[str, Any] = {}
    if confidence_level is not None:
        kwargs["confidence_level"] = confidence_level
    if history_limit is not None:
        kwargs["history_limit"] = history_limit
    if window_limit is not None:
        kwargs["window_limit"] = window_limit

    accumulator = _configured_accumulator(graph, **kwargs)
    snapshot = accumulator.record_window(window, graph=graph)
    accumulator.attach_graph(graph)
    graph.graph[_ACCUMULATOR_KEY] = accumulator
    return snapshot

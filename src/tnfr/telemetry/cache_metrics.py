"""Cache telemetry publishers for structured observability channels."""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import Any, MutableMapping, TYPE_CHECKING

from ..utils import (
    _graph_cache_manager,
    CacheManager,
    CacheStatistics,
    get_logger,
    json_dumps,
)

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from networkx import Graph

    from ..types import TNFRGraph

__all__ = (
    "CacheMetricsSnapshot",
    "CacheTelemetryPublisher",
    "ensure_cache_metrics_publisher",
    "publish_graph_cache_metrics",
)


@dataclass(frozen=True)
class CacheMetricsSnapshot:
    """Structured cache metrics enriched with ratios and latency estimates."""

    cache: str
    hits: int
    misses: int
    evictions: int
    total_time: float
    timings: int
    hit_ratio: float | None
    miss_ratio: float | None
    avg_latency: float | None

    @classmethod
    def from_statistics(cls, name: str, stats: CacheStatistics) -> "CacheMetricsSnapshot":
        """Build a snapshot computing ratios from :class:`CacheStatistics`."""

        hits = int(stats.hits)
        misses = int(stats.misses)
        evictions = int(stats.evictions)
        total_time = float(stats.total_time)
        timings = int(stats.timings)
        requests = hits + misses
        hit_ratio = (hits / requests) if requests else None
        miss_ratio = (misses / requests) if requests else None
        avg_latency = (total_time / timings) if timings else None
        return cls(
            cache=name,
            hits=hits,
            misses=misses,
            evictions=evictions,
            total_time=total_time,
            timings=timings,
            hit_ratio=hit_ratio,
            miss_ratio=miss_ratio,
            avg_latency=avg_latency,
        )

    def as_payload(self) -> dict[str, Any]:
        """Return a dictionary suitable for structured logging."""

        return {
            "cache": self.cache,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_time": self.total_time,
            "timings": self.timings,
            "hit_ratio": self.hit_ratio,
            "miss_ratio": self.miss_ratio,
            "avg_latency": self.avg_latency,
        }


class CacheTelemetryPublisher:
    """Metrics publisher broadcasting cache counters to observability channels."""

    def __init__(
        self,
        *,
        graph: "TNFRGraph | Graph | MutableMapping[str, Any] | None" = None,
        logger: logging.Logger | None = None,
        hit_ratio_alert: float = 0.5,
        latency_alert: float = 0.1,
    ) -> None:
        self._logger = logger or get_logger("tnfr.telemetry.cache")
        self._graph_ref: (
            weakref.ReferenceType["TNFRGraph | Graph | MutableMapping[str, Any]"] | None
        ) = None
        self._hit_ratio_alert = float(hit_ratio_alert)
        self._latency_alert = float(latency_alert)
        self.attach_graph(graph)

    @property
    def logger(self) -> logging.Logger:
        """Logger used for structured cache telemetry."""

        return self._logger

    def attach_graph(self, graph: "TNFRGraph | Graph | MutableMapping[str, Any] | None") -> None:
        """Attach ``graph`` so observability callbacks receive metrics."""

        if graph is None:
            return
        try:
            self._graph_ref = weakref.ref(graph)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover - defensive path for exotic graphs
            self._graph_ref = None

    def _resolve_graph(
        self,
    ) -> "TNFRGraph | Graph | MutableMapping[str, Any] | None":
        return self._graph_ref() if self._graph_ref is not None else None

    def __call__(self, name: str, stats: CacheStatistics) -> None:
        """Emit structured telemetry and invoke observability hooks."""

        snapshot = CacheMetricsSnapshot.from_statistics(name, stats)
        payload = snapshot.as_payload()
        message = json_dumps({"event": "cache_metrics", **payload}, sort_keys=True)
        self._logger.info(message)

        if (
            snapshot.hit_ratio is not None
            and snapshot.hit_ratio < self._hit_ratio_alert
            and snapshot.misses > 0
        ):
            warning = json_dumps(
                {
                    "event": "cache_metrics.low_hit_ratio",
                    "cache": name,
                    "hit_ratio": snapshot.hit_ratio,
                    "threshold": self._hit_ratio_alert,
                    "requests": snapshot.hits + snapshot.misses,
                },
                sort_keys=True,
            )
            self._logger.warning(warning)

        if (
            snapshot.avg_latency is not None
            and snapshot.avg_latency > self._latency_alert
            and snapshot.timings > 0
        ):
            warning = json_dumps(
                {
                    "event": "cache_metrics.high_latency",
                    "cache": name,
                    "avg_latency": snapshot.avg_latency,
                    "threshold": self._latency_alert,
                    "timings": snapshot.timings,
                },
                sort_keys=True,
            )
            self._logger.warning(warning)

        graph = self._resolve_graph()
        if graph is not None:
            from ..utils import CallbackEvent, callback_manager

            ctx = {"cache": name, "metrics": payload}
            callback_manager.invoke_callbacks(graph, CallbackEvent.CACHE_METRICS, ctx)


_PUBLISHER_ATTR = "_tnfr_cache_metrics_publisher"


def ensure_cache_metrics_publisher(
    manager: CacheManager,
    *,
    graph: "TNFRGraph | Graph | MutableMapping[str, Any] | None" = None,
    logger: logging.Logger | None = None,
    hit_ratio_alert: float = 0.5,
    latency_alert: float = 0.1,
) -> CacheTelemetryPublisher:
    """Attach a :class:`CacheTelemetryPublisher` to ``manager`` if missing."""

    publisher = getattr(manager, _PUBLISHER_ATTR, None)
    if not isinstance(publisher, CacheTelemetryPublisher):
        publisher = CacheTelemetryPublisher(
            graph=graph,
            logger=logger,
            hit_ratio_alert=hit_ratio_alert,
            latency_alert=latency_alert,
        )
        manager.register_metrics_publisher(publisher)
        setattr(manager, _PUBLISHER_ATTR, publisher)
    else:
        if graph is not None:
            publisher.attach_graph(graph)
    return publisher


def publish_graph_cache_metrics(
    graph: "TNFRGraph | Graph | MutableMapping[str, Any]",
    *,
    manager: CacheManager | None = None,
    hit_ratio_alert: float = 0.5,
    latency_alert: float = 0.1,
) -> None:
    """Publish cache metrics for ``graph`` using the shared manager."""

    if manager is None:
        manager = _graph_cache_manager(getattr(graph, "graph", graph))
    ensure_cache_metrics_publisher(
        manager,
        graph=graph,
        hit_ratio_alert=hit_ratio_alert,
        latency_alert=latency_alert,
    )
    manager.publish_metrics()

from __future__ import annotations

import logging
from ..types import TNFRGraph
from ..utils import CacheManager, CacheStatistics
from dataclasses import dataclass
from networkx import Graph
from typing import Any, MutableMapping

__all__ = [
    "CacheMetricsSnapshot",
    "CacheTelemetryPublisher",
    "ensure_cache_metrics_publisher",
    "publish_graph_cache_metrics",
]

@dataclass(frozen=True)
class CacheMetricsSnapshot:
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
    def from_statistics(cls, name: str, stats: CacheStatistics) -> CacheMetricsSnapshot: ...
    def as_payload(self) -> dict[str, Any]: ...

class CacheTelemetryPublisher:
    def __init__(
        self,
        *,
        graph: TNFRGraph | Graph | MutableMapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
        hit_ratio_alert: float = 0.5,
        latency_alert: float = 0.1,
    ) -> None: ...
    @property
    def logger(self) -> logging.Logger: ...
    def attach_graph(self, graph: TNFRGraph | Graph | MutableMapping[str, Any] | None) -> None: ...
    def __call__(self, name: str, stats: CacheStatistics) -> None: ...

def ensure_cache_metrics_publisher(
    manager: CacheManager,
    *,
    graph: TNFRGraph | Graph | MutableMapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
    hit_ratio_alert: float = 0.5,
    latency_alert: float = 0.1,
) -> CacheTelemetryPublisher: ...
def publish_graph_cache_metrics(
    graph: TNFRGraph | Graph | MutableMapping[str, Any],
    *,
    manager: CacheManager | None = None,
    hit_ratio_alert: float = 0.5,
    latency_alert: float = 0.1,
) -> None: ...

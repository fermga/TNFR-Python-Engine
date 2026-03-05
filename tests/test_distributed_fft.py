"""Tests for the distributed FFT backend shim."""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx

from tnfr.dynamics.distributed_fft import DistributedFFTEngine
from tnfr.dynamics.fft_dispatchers import ThreadedQueueDispatcher
from tnfr.dynamics.fft_workers import default_fft_worker


def test_distributed_fft_falls_back_when_no_dispatcher() -> None:
    engine = DistributedFFTEngine(dispatcher=None)

    G = nx.path_graph(4)
    state = engine.get_spectral_state(G)

    assert state.eigenvalues.shape[0] == 4

    caps = engine.get_capabilities()
    assert caps.supports_distributed is True
    assert caps.extra["has_dispatcher"] is False


def test_distributed_fft_queue_dispatcher_executes_remote_path() -> None:
    call_count = {"count": 0}

    def worker(action: str, payload: dict[str, Any]) -> Any:
        call_count["count"] += 1
        return default_fft_worker(action, payload)

    queue_dispatcher = ThreadedQueueDispatcher(worker=worker, timeout=5.0)
    engine = DistributedFFTEngine(dispatcher=queue_dispatcher.dispatch)

    graph = nx.cycle_graph(8)
    state = engine.get_spectral_state(graph)

    assert state.eigenvalues.shape[0] == graph.number_of_nodes()
    assert call_count["count"] >= 1


def test_threaded_queue_dispatcher_serialization_hooks() -> None:
    encoded_requests: Dict[str, Any] = {}

    def request_serializer(payload: Dict[str, Any]) -> Dict[str, Any]:
        wrapper = {"wrapped": payload}
        encoded_requests["last"] = wrapper
        return wrapper

    def request_deserializer(wrapper: Dict[str, Any]) -> Dict[str, Any]:
        return wrapper["wrapped"]

    def response_serializer(result: Any) -> Dict[str, Any]:
        return {"result": result}

    def response_deserializer(wrapper: Dict[str, Any]) -> Any:
        return wrapper["result"]

    dispatcher = ThreadedQueueDispatcher(
        max_workers=2,
        request_serializer=request_serializer,
        request_deserializer=request_deserializer,
        response_serializer=response_serializer,
        response_deserializer=response_deserializer,
    )

    graph = nx.path_graph(6)
    engine = DistributedFFTEngine(dispatcher=dispatcher.dispatch)
    state = engine.get_spectral_state(graph)

    assert state.eigenvalues.shape[0] == graph.number_of_nodes()
    assert "last" in encoded_requests

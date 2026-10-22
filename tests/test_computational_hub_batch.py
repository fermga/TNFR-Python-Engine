"""Bounded public dispatch and measured elapsed-time contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tnfr.dynamics.computational_hub as module


class _ObservedHub:
    def __init__(self, *, timeout_id: str | None = None) -> None:
        self.pending: list[str] = []
        self.submitted: list[str] = []
        self.collected: list[str] = []
        self.maximum_pending = 0
        self.timeout_id = timeout_id

    def submit_computation(self, request: module.ComputationRequest) -> str:
        self.pending.append(request.request_id)
        self.submitted.append(request.request_id)
        self.maximum_pending = max(self.maximum_pending, len(self.pending))
        return request.request_id

    def get_result(
        self, request_id: str, timeout: float | None = None
    ) -> module.ComputationResult | None:
        assert timeout == 300.0
        self.collected.append(request_id)
        if request_id == self.timeout_id:
            return None
        self.pending.remove(request_id)
        return module.ComputationResult(
            request_id=request_id,
            engine_used=module.EngineType.FFT_ENGINE,
            operation="supplied_operation",
            success=request_id != "failure",
            error_message="adapter failure" if request_id == "failure" else None,
        )


@pytest.mark.parametrize("limit", [1, 2, np.int64(3), 10])
def test_batch_bounds_outstanding_requests_and_keeps_order(monkeypatch, limit):
    hub = _ObservedHub()
    monkeypatch.setattr(module, "get_computational_hub", lambda: hub)
    request_ids = ["last", "failure", "first", "middle", "end"]
    requests = [module.ComputationRequest(request_id=name) for name in request_ids]

    results = module.batch_execute_computations(requests, max_parallel=limit)

    assert hub.maximum_pending == min(int(limit), len(requests))
    assert hub.submitted == hub.collected == request_ids
    assert list(results) == request_ids
    assert not hub.pending
    assert results["failure"].success is False
    assert results["failure"].error_message == "adapter failure"
    assert all(results[name].success for name in request_ids if name != "failure")


@pytest.mark.parametrize("invalid", [True, np.bool_(False), 0, -1, 1.0, "2", None])
def test_invalid_batch_limit_rejects_before_hub_access_even_empty(monkeypatch, invalid):
    def unexpected_hub():
        pytest.fail("invalid batch configuration must not initialize the hub")

    monkeypatch.setattr(module, "get_computational_hub", unexpected_hub)
    with pytest.raises((TypeError, ValueError)):
        module.batch_execute_computations([], max_parallel=invalid)


def test_batch_timeout_does_not_submit_more_or_cancel_pending(monkeypatch):
    hub = _ObservedHub(timeout_id="first")
    monkeypatch.setattr(module, "get_computational_hub", lambda: hub)
    requests = [
        module.ComputationRequest(request_id=name)
        for name in ["first", "second", "third"]
    ]

    with pytest.raises(TimeoutError, match="submitted work remains active"):
        module.batch_execute_computations(requests, max_parallel=2)

    assert hub.submitted == hub.pending == ["first", "second"]
    assert hub.collected == ["first"]


@pytest.mark.parametrize("operation", ["system_status", "supplied_operation"])
def test_execution_duration_uses_one_monotonic_clock(monkeypatch, operation):
    elapsed_clock = iter([10.0, 10.125])
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(
            time=lambda: 1_700_000_000.0,
            perf_counter=lambda: next(elapsed_clock),
        ),
    )
    hub = object.__new__(module.TNFRComputationalHub)
    hub._engine_performance = {module.EngineType.FFT_ENGINE: []}
    monkeypatch.setattr(hub, "get_system_status", lambda: {"fixture": True})
    monkeypatch.setattr(
        hub, "_select_optimal_engine", lambda request: module.EngineType.FFT_ENGINE
    )
    monkeypatch.setattr(hub, "_execute_fft_engine", lambda request: {"fixture": True})

    result = hub._execute_computation(module.ComputationRequest(operation=operation))

    assert result.success is True
    assert result.execution_time == 0.125
    if operation != "system_status":
        assert hub._engine_performance[module.EngineType.FFT_ENGINE] == [0.125]

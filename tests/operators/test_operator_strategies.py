from __future__ import annotations

from collections.abc import Iterator

import pytest

from tnfr.operators.strategies import (
    ensure_default_strategies_registered,
    OperationResult,
    ResourceEstimate,
    StrategyContext,
    StrategyRegistrationError,
    StrategyRegistry,
    StructuralFields,
)


def _structural_fields() -> StructuralFields:
    return StructuralFields(
        phi_s=0.5,
        phase_gradient=0.1,
        phase_curvature=0.2,
        coherence_length=1.5,
    )


def _resource_estimate() -> ResourceEstimate:
    return ResourceEstimate(
        memory_bytes=1024,
        time_ms=1.5,
        delta_nfr=0.1,
        phi_s_drift=0.01,
        failure_risk="low",
    )


def _make_context(**overrides: object) -> StrategyContext:
    payload = dict(
        partition_id="p1",
        operator_sequence_position=3,
        structural_fields=_structural_fields(),
        dispatcher_capabilities={"gpu": True},
        backend="cpu",
        block_size=2048,
        boundary_overlap=8,
        seed=7,
    )
    payload.update(overrides)
    return StrategyContext(**payload)  # type: ignore[arg-type]


class _DummyStrategy:
    operator = "AL"

    def supports(self, ctx: StrategyContext) -> bool:  # pragma: no cover - trivial
        return ctx.block_size > 0

    def resource_estimate(self, ctx: StrategyContext) -> ResourceEstimate:
        return _resource_estimate()

    def prepare(self, ctx: StrategyContext, block: str) -> str:
        return f"prepared:{block}:{ctx.partition_id}"

    def apply(self, prepared: str) -> OperationResult:
        return OperationResult(block=prepared, telemetry={"prepared": prepared}, proof_hash="hash")

    def cleanup(self, prepared: str) -> None:  # pragma: no cover - no side effects
        return None


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    StrategyRegistry.clear()
    yield
    StrategyRegistry.clear()


def test_strategy_registry_register_and_create() -> None:
    StrategyRegistry.register(operator="AL", name="dummy", factory=_DummyStrategy)

    strategy = StrategyRegistry.create(operator="AL", name="dummy")

    assert isinstance(strategy, _DummyStrategy)
    ctx = _make_context()
    assert strategy.supports(ctx)
    prepared = strategy.prepare(ctx, "block")
    result = strategy.apply(prepared)
    assert result.block == prepared
    assert result.telemetry["prepared"].startswith("prepared")


def test_strategy_registry_rejects_duplicate_names() -> None:
    StrategyRegistry.register(operator="AL", name="dummy", factory=_DummyStrategy)

    with pytest.raises(StrategyRegistrationError):
        StrategyRegistry.register(operator="AL", name="dummy", factory=_DummyStrategy)


def test_strategy_registry_available_lists_registered() -> None:
    StrategyRegistry.register(operator="AL", name="dummy", factory=_DummyStrategy)
    StrategyRegistry.register(operator="IL", name="dummy-il", factory=_DummyStrategy)

    available = StrategyRegistry.available()

    assert available["AL"] == ["dummy"]
    assert available["IL"] == ["dummy-il"]
    assert available["RA"] == []


def test_strategy_context_exposes_structural_fields() -> None:
    ctx = _make_context(partition_id="p7")

    assert ctx.partition_id == "p7"
    assert ctx.structural_fields.phi_s == pytest.approx(0.5)
    assert ctx.structural_fields.phase_gradient == pytest.approx(0.1)
    assert ctx.structural_fields.phase_curvature == pytest.approx(0.2)
    assert ctx.structural_fields.coherence_length == pytest.approx(1.5)


def test_strategy_registry_rejects_empty_name() -> None:
    with pytest.raises(StrategyRegistrationError):
        StrategyRegistry.register(operator="AL", name="  ", factory=_DummyStrategy)


def test_default_strategies_available() -> None:
    ensure_default_strategies_registered()
    available = StrategyRegistry.available()
    assert "cpu-default" in available["AL"]
    assert "cpu-default" in available["IL"]

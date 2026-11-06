"""Discrete-domain contracts for the simplified SDK evolution methods."""

from __future__ import annotations

import pytest

from tnfr.sdk import TNFR


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("steps", [True, False, 1.0, 1.5])
def test_evolve_rejects_coerced_cycle_counts(
    steps,
    record: bool,
) -> None:
    network = TNFR.create(3, seed=5).ring()

    with pytest.raises(ValueError, match="steps must be a non-negative integer"):
        network.evolve(steps=steps, record=record)


@pytest.mark.parametrize("cycles", [True, False, 1.0, 1.5])
def test_trajectory_rejects_coerced_cycle_counts(cycles) -> None:
    network = TNFR.create(3, seed=5).ring()

    with pytest.raises(ValueError, match="cycles must be a non-negative integer"):
        network.trajectory(cycles=cycles)


def test_zero_cycle_counts_are_valid_no_ops() -> None:
    direct = TNFR.create(3, seed=5).ring()
    recorded = TNFR.create(3, seed=5).ring()
    traced = TNFR.create(3, seed=5).ring()

    assert direct.evolve(steps=0, record=False) is direct
    assert recorded.evolve(steps=0, record=True) is recorded
    assert recorded.history() == {
        "kuramoto_R": [],
        "C_steps": [],
        "phase_sync": [],
        "Si_mean": [],
    }
    assert traced.trajectory(cycles=0) == []

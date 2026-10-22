"""Finite arithmetic must precede dissipative diagnostic verdicts."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tnfr.physics import dissipative_conservation as dc


def _damping(scale):
    return np.array([[0.0, scale], [0.0, 0.0]])


def _excited():
    return np.diag([0.0, 1.0])


@pytest.mark.parametrize("consumer", ["action", "bound", "purity_bound", "rate"])
def test_finite_collapse_entries_do_not_admit_nonfinite_outputs(consumer):
    operators = [_damping(1e154)]
    with pytest.raises(ValueError):
        if consumer == "action":
            dc.compute_dissipator_action(_excited(), operators)
        elif consumer == "bound":
            dc.compute_dissipation_bound(operators, 1.0)
        elif consumer == "purity_bound":
            dc.compute_purity_decay_bound(operators, _excited())
        else:
            dc.compute_instantaneous_purity_rate(operators, _excited())


def test_infinite_bound_and_action_norm_cannot_produce_a_passing_balance():
    snapshot = dc.capture_dissipative_snapshot(_excited())
    with pytest.raises(ValueError, match="dissipation bound"):
        dc.verify_dissipative_balance(
            snapshot, snapshot, collapse_operators=[_damping(7e153)] * 2
        )


def test_large_representable_action_norm_and_rate_are_preserved():
    snapshot = dc.capture_dissipative_snapshot(_excited())
    scale = 7e153
    rate = scale**2
    balance = dc.verify_dissipative_balance(
        snapshot, snapshot, collapse_operators=[_damping(scale)]
    )
    assert balance.actual_dissipation == pytest.approx(
        math.hypot(rate, rate), rel=1e-14
    )
    assert balance.dissipation_bound == pytest.approx(2 * rate, rel=1e-14)
    assert balance.instantaneous_purity_change_rate == pytest.approx(
        -2 * rate, rel=1e-14
    )
    assert balance.dissipation_bound_satisfied
    assert not balance.unital_dissipator


def test_nonfinite_custom_bound_is_rejected_before_comparison(monkeypatch):
    snapshot = dc.capture_dissipative_snapshot(_excited())
    monkeypatch.setattr(dc, "compute_dissipation_bound", lambda *_: float("inf"))
    with pytest.raises(ValueError, match="dissipation bound"):
        dc.verify_dissipative_balance(
            snapshot, snapshot, collapse_operators=[_damping(0.5)]
        )


def test_unitality_requires_a_representable_comparison_threshold():
    with pytest.raises(ValueError, match="unitality threshold"):
        dc.is_unital_dissipator([1e154 * np.eye(2)] * 2)


def test_nonzero_collapse_squared_norm_cannot_silently_underflow():
    with pytest.raises(ValueError, match="squared norm underflows"):
        dc.compute_dissipation_bound([_damping(1e-200)], 1.0)


def test_observed_rates_reject_tiny_positive_time_overflow():
    before = dc.capture_dissipative_snapshot(_excited())
    after = dc.capture_dissipative_snapshot(np.eye(2) / 2)
    with pytest.raises(ValueError, match="purity rate"):
        dc.verify_dissipative_balance(before, after, dt=np.nextafter(0.0, 1.0))


def test_identical_snapshots_have_zero_rates_at_the_smallest_positive_time():
    snapshot = dc.capture_dissipative_snapshot(_excited())
    result = dc.verify_dissipative_balance(
        snapshot, snapshot, dt=np.nextafter(0.0, 1.0)
    )
    assert result.purity_decay_rate == 0.0
    assert result.entropy_production_rate == 0.0
    assert result.state_change_rate == 0.0
    assert result.charge_leak_rate == 0.0
    assert result.dissipation_bound_satisfied is None
    assert math.isnan(result.dissipation_bound)


@pytest.mark.parametrize("elapsed", [1e100, 1e200])
def test_small_observed_state_change_survives_norm_squaring(elapsed):
    before = dc.capture_dissipative_snapshot(np.eye(2) / 2)
    after = dc.capture_dissipative_snapshot(np.array([[0.5, 1e-200], [1e-200, 0.5]]))
    if elapsed == 1e200:
        with pytest.raises(ValueError, match="state-change rate"):
            dc.verify_dissipative_balance(before, after, dt=elapsed)
    else:
        result = dc.verify_dissipative_balance(before, after, dt=elapsed)
        assert result.state_change_rate == pytest.approx(
            math.hypot(1e-200, 1e-200) / elapsed, rel=1e-14, abs=0.0
        )


def test_missing_contractivity_denominator_retains_explicit_infinity():
    before = dc.capture_dissipative_snapshot(np.eye(2) / 2)
    after = dc.capture_dissipative_snapshot(_excited())
    result = dc.verify_dissipative_balance(before, after, steady_state=before.density)
    assert result.contractivity_gap == float("inf")
    assert result.contractivity_evaluated
    assert not result.is_contractive


@pytest.mark.parametrize("consumer", ["tracker", "spectrum", "steady_state"])
def test_overflowing_generator_scale_cannot_certify_a_stationary_mode(consumer):
    generator = np.full((4, 4), 1e308)
    with pytest.raises(ValueError, match="generator spectral norm"):
        if consumer == "tracker":
            engine = SimpleNamespace(
                generator=generator, hilbert_space=SimpleNamespace(dimension=2)
            )
            dc.DissipativeConservationTracker(engine, steady_state=np.eye(2) / 2)
        elif consumer == "spectrum":
            dc.analyze_dissipation_rates(generator, dim=2)
        else:
            dc.steady_state_from_generator(generator, dim=2)


def test_rejected_record_does_not_poison_tracker_history():
    engine = SimpleNamespace(
        generator=np.zeros((4, 4)), hilbert_space=SimpleNamespace(dimension=2)
    )
    tracker = dc.DissipativeConservationTracker(engine)
    tracker.record(_excited(), t=0.0)
    with pytest.raises(ValueError, match="purity rate"):
        tracker.record(np.eye(2) / 2, t=np.nextafter(0.0, 1.0))
    assert len(tracker._snapshots) == 1
    assert tracker._series.times == [0.0]
    tracker.record(np.eye(2) / 2, t=1.0)
    assert len(tracker._snapshots) == 2
    assert tracker._series.times == [0.0, 1.0]
    assert tracker._series.purity_change_rate[-1] == pytest.approx(-0.5)


def test_tolerant_density_admission_still_requires_finite_purity():
    with pytest.raises(ValueError, match="density purity"):
        dc.capture_dissipative_snapshot(
            np.array([[0.5, 1e155], [1e155, 0.5]]), atol=1e200
        )

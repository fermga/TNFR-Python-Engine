"""Regression checks for the operator-event relaxation example."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import pytest

import tnfr.physics as physics
import tnfr.physics.event_duration as event_duration


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_physics_regimes"
    / "165_operator_event_relaxation.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "operator_event_relaxation_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_and_protocol():
    example = _load_example()
    return example, example.run_protocol()


def test_event_duration_exports_are_available_from_physics() -> None:
    names = set(event_duration.__all__)

    assert names == {
        "ContinuousRelaxationDurationDiagnostic",
        "diagnose_continuous_relaxation_duration",
    }
    assert names <= set(physics.__all__)
    for name in names:
        assert getattr(physics, name) is getattr(event_duration, name)


def test_schedule_uses_m_plus_one_exact_flow_intervals(
    example_and_protocol,
) -> None:
    _, protocol = example_and_protocol
    schedule = protocol["schedule"]

    assert schedule.event_count == 2
    assert len(schedule.intervals) == schedule.event_count + 1
    assert schedule.flow_durations[1:] == (0.0, 0.0)
    assert schedule.exact_total_flow_duration == (
        schedule.intervals[0].exact_duration
    )
    assert [event.event_index for event in schedule.events] == [0, 1]
    assert schedule.events[0].exact_event_time == (
        schedule.events[1].exact_event_time
    )
    assert schedule.events[0].event_time == schedule.events[1].event_time
    assert schedule.event_history_channel == "hybrid_event_log"
    assert schedule.coincident_event_order == "event_index"
    assert not schedule.event_timestamps_feed_epi_time_history


def test_relaxation_uses_exact_duration_instead_of_timestamp_delta(
    example_and_protocol,
) -> None:
    _, protocol = example_and_protocol
    schedule = protocol["schedule"]
    interval = schedule.intervals[0]
    result = protocol["relaxation"]

    assert interval.end_time - interval.start_time == 1.0
    assert interval.duration != 1.0
    assert interval.exact_duration == Fraction.from_float(interval.duration)
    assert result.exact_flow_duration == interval.exact_duration
    assert result.exact_energy_decay_rate_lower_bound is not None
    assert result.exact_log_target_upper_bound is not None
    assert (
        result.exact_energy_decay_rate_lower_bound
        * result.exact_flow_duration
        >= result.exact_log_target_upper_bound
    )
    assert result.duration_reaches_target is True
    assert result.certified_decay_factor_upper_bound is not None
    assert result.certified_decay_factor_upper_bound <= result.target_fraction
    assert not result.spectral_estimate_is_proof_input


def test_report(example_and_protocol) -> None:
    example, protocol = example_and_protocol
    report = example.build_report(protocol)
    encoded = json.dumps(report, allow_nan=False, separators=(",", ":"))

    assert len(encoded) < 7000
    assert report["schedule"]["m_plus_one_flow_intervals"]
    assert report["schedule"]["timestamp_role"] == (
        "binary64_representation_only"
    )
    assert not report["schedule"][
        "event_timestamps_feed_epi_time_history"
    ]
    assert report["relaxation"]["duration_reaches_target"]
    assert not report["relaxation"]["spectral_estimate_is_proof_input"]
    assert not report["scope"]["operator_events_executed"]
    assert not report["scope"]["runtime_integration_certified"]
    assert not report["scope"]["adaptive_u2_u4_policy"]


def test_main_executes_and_emits_finite_json(
    example_and_protocol,
    capsys,
) -> None:
    example, _ = example_and_protocol

    example.main()
    report = json.loads(capsys.readouterr().out)

    assert report["claim"] == (
        "operator-event physical-time and fixed-flow relaxation boundary"
    )
    assert [event["event_index"] for event in report["hybrid_event_log"]] == [
        0,
        1,
    ]

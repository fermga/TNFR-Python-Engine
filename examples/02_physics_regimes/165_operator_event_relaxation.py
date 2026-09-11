"""Expose the exact physical-time boundary for a finite operator schedule.

The example places a positive pure-EPI flow before two coincident, zero-duration
operator events.  A schedule with ``m`` events has ``m + 1`` flow intervals.
The materialized binary64 durations define exact rational offsets; absolute
binary64 timestamps are display representations and are never subtracted to
recover duration or written to ``epi_time_history``.

The relaxation diagnostic certifies only the initial frozen, connected,
symmetric, positive-capacity pure-EPI flow.  It does not execute either event,
identify an operator gain, certify a runtime integration, or adapt U2/U4.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

import networkx as nx

from tnfr.operators import build_operator_event_schedule
from tnfr.physics import diagnose_continuous_relaxation_duration


OPERATOR_WORD = ("emission", "silence")
START_TIME = float(2**52)
TARGET_FRACTION = 0.25


def fixed_diffusion_graph() -> nx.Graph:
    """Build a deterministic homogeneous complete graph in theorem scope."""

    graph = nx.complete_graph(4)
    for index, node in enumerate(graph):
        graph.nodes[node].update(
            EPI=float(index),
            nu_f=1.0,
            theta=0.0,
        )
    return graph


def run_protocol() -> dict[str, Any]:
    """Construct a schedule at its sufficient certified flow duration."""

    graph = fixed_diffusion_graph()
    probe = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=0.0,
        target_fraction=TARGET_FRACTION,
    )
    required_duration = probe.required_flow_duration
    if probe.abstained or required_duration is None:
        raise RuntimeError("the fixed diffusion fixture must provide a duration")
    schedule = build_operator_event_schedule(
        OPERATOR_WORD,
        start_time=START_TIME,
        flow_durations=(required_duration, 0.0, 0.0),
    )
    relaxation = diagnose_continuous_relaxation_duration(
        graph,
        flow_duration=required_duration,
        target_fraction=TARGET_FRACTION,
    )
    return {
        "schedule": schedule,
        "relaxation": relaxation,
    }


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def build_report(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return a finite JSON-compatible record of the two scoped contracts."""

    schedule = protocol["schedule"]
    relaxation = protocol["relaxation"]
    first_interval = schedule.intervals[0]
    return {
        "claim": "operator-event physical-time and fixed-flow relaxation boundary",
        "schedule": {
            "operator_word": list(schedule.operator_names),
            "event_count": schedule.event_count,
            "flow_interval_count": len(schedule.intervals),
            "m_plus_one_flow_intervals": (
                len(schedule.intervals) == schedule.event_count + 1
            ),
            "flow_durations": list(schedule.flow_durations),
            "exact_total_flow_duration": _fraction_text(
                schedule.exact_total_flow_duration
            ),
            "duration_and_offsets_are_authoritative": (
                schedule.duration_and_offsets_are_authoritative
            ),
            "timestamp_role": schedule.timestamp_role,
            "event_history_channel": schedule.event_history_channel,
            "coincident_event_order": schedule.coincident_event_order,
            "event_timestamps_feed_epi_time_history": (
                schedule.event_timestamps_feed_epi_time_history
            ),
        },
        "first_flow_interval": {
            "duration": first_interval.duration,
            "exact_duration": _fraction_text(first_interval.exact_duration),
            "represented_timestamp_delta": (
                first_interval.end_time - first_interval.start_time
            ),
        },
        "hybrid_event_log": [
            {
                "event_index": event.event_index,
                "operator_name": event.operator_name,
                "glyph": event.glyph.value,
                "event_time": event.event_time,
                "exact_event_offset": _fraction_text(event.event_offset),
            }
            for event in schedule.events
        ],
        "relaxation": {
            "target_fraction": relaxation.target_fraction,
            "exact_rate_lower_bound": _fraction_text(
                relaxation.exact_energy_decay_rate_lower_bound
            ),
            "certified_rate_display": (
                relaxation.certified_energy_decay_rate_lower_bound
            ),
            "spectral_rate_estimate": (
                relaxation.spectral_energy_decay_rate_estimate
            ),
            "spectral_estimate_provenance": (
                relaxation.spectral_estimate_provenance
            ),
            "spectral_estimate_is_proof_input": (
                relaxation.spectral_estimate_is_proof_input
            ),
            "exact_log_target_upper_bound": _fraction_text(
                relaxation.exact_log_target_upper_bound
            ),
            "exact_required_duration_upper_bound": _fraction_text(
                relaxation.exact_required_flow_duration_upper_bound
            ),
            "required_duration_display": relaxation.required_flow_duration,
            "required_duration_estimate": (
                relaxation.required_flow_duration_estimate
            ),
            "certified_decay_factor_upper_bound": (
                relaxation.certified_decay_factor_upper_bound
            ),
            "decay_factor_estimate": relaxation.decay_factor_estimate,
            "duration_reaches_target": relaxation.duration_reaches_target,
        },
        "scope": {
            "operator_events_executed": False,
            "runtime_integration_certified": False,
            "adaptive_u2_u4_policy": False,
            "flow_model": (
                "frozen connected symmetric positive-capacity pure-EPI"
            ),
        },
    }


def main() -> None:
    print(json.dumps(build_report(run_protocol()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Observed count rates retain raw domains, rolling evidence and finite output."""

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.telemetry.nu_f import (
    NuFTelemetryAccumulator,
    NuFWindow,
    ensure_nu_f_telemetry,
    record_nu_f_window,
)


@pytest.mark.parametrize("count", [True, False, 1.9, "2", -1])
@pytest.mark.parametrize("route", ["direct", "bounds", "counts", "graph"])
def test_count_admission_precedes_integer_conversion(count, route):
    with pytest.raises((TypeError, ValueError)):
        if route == "direct":
            NuFWindow(count, 1)
        elif route == "bounds":
            NuFWindow.from_bounds(count, 0, 1)
        elif route == "counts":
            NuFTelemetryAccumulator().record_counts(count, 1)
        else:
            graph = nx.Graph()
            record_nu_f_window(graph, count, 1)
    if route == "graph":
        assert graph.graph == {}


@pytest.mark.parametrize("name", ["duration", "start", "end"])
@pytest.mark.parametrize("value", [True, "1", math.nan, math.inf, Fraction(1, 10**400)])
def test_each_clock_is_admitted_even_when_only_one_bound_is_present(name, value):
    kwargs = {"reorganisations": 1, "duration": 1, name: value}
    with pytest.raises((TypeError, ValueError)):
        NuFWindow(**kwargs)


@pytest.mark.parametrize("value", [True, "0", Fraction(1, 10**400)])
def test_bounds_factory_preserves_raw_clock_domain(value):
    with pytest.raises((TypeError, ValueError)):
        NuFWindow.from_bounds(1, value, 2)


def test_small_windows_do_not_gain_an_absolute_duration_tolerance():
    with pytest.raises(ValueError, match="does not match"):
        NuFWindow(1, 1e-100, start=0, end=1e-12)
    window = NuFWindow(1, 1e-12, start=0, end=1e-12)
    assert window.duration == 1e-12


def test_independent_count_rate_and_approximate_interval():
    graph = nx.Graph()
    graph.graph["HZ_STR_BRIDGE"] = 2
    accumulator = NuFTelemetryAccumulator(graph=graph)
    accumulator.record_counts(np.int64(3), 2, start=-2, end=0)
    result = accumulator.record_counts(5, 2, start=0, end=2)
    assert result.total_reorganisations == 8
    assert result.total_duration == 4
    assert result.rate_hz_str == 2
    assert result.variance_hz_str == 0.5
    assert result.rate_hz == 4
    assert result.variance_hz == 2
    # Independent standard-normal 97.5th percentile, not the producer's helper.
    half_width = 1.959963984540054 * math.sqrt(0.5)
    assert result.ci_lower_hz_str == pytest.approx(2 - half_width)
    assert result.ci_upper_hz == pytest.approx(2 * (2 + half_width))
    assert len(graph.graph["telemetry"]["nu_f_history"]) == 2


def test_rolling_eviction_does_not_erase_small_retained_durations():
    accumulator = NuFTelemetryAccumulator(window_limit=2)
    for duration in (1e16, 1, 1):
        result = accumulator.record_counts(1, duration)
    assert result.total_duration == 2
    assert result.total_reorganisations == 2
    assert result.rate_hz_str == 1
    assert result.variance_hz_str == 0.5
    assert [window.duration for window in result.windows] == [1, 1]
    accumulator.reset()
    assert accumulator.snapshot().rate_hz_str is None
    assert accumulator.record_counts(2, 1).rate_hz_str == 2


@pytest.mark.parametrize("count,duration", [(10**400, 1), (1, 5e-324), (1, 1e308)])
def test_unrepresentable_estimate_rejects_before_buffer_or_history_changes(
    count, duration
):
    graph = nx.Graph()
    accumulator = NuFTelemetryAccumulator(graph=graph, window_limit=1)
    before = accumulator.record_counts(3, 2)
    history = graph.graph["telemetry"]["nu_f_history"][:]
    with pytest.raises(ValueError):
        accumulator.record_counts(count, duration)
    assert accumulator.snapshot() == before
    assert graph.graph["telemetry"]["nu_f_history"] == history


def test_total_duration_overflow_does_not_commit():
    accumulator = NuFTelemetryAccumulator()
    before = accumulator.record_counts(0, 1e308)
    with pytest.raises(ValueError, match="total duration"):
        accumulator.record_counts(0, 1e308)
    assert accumulator.snapshot() == before


def test_large_bridge_squared_need_not_overflow_a_representable_variance():
    graph = nx.Graph()
    graph.graph["HZ_STR_BRIDGE"] = 1e160
    result = NuFTelemetryAccumulator(graph=graph).record_counts(1, 1e160)
    assert result.rate_hz == pytest.approx(1)
    assert result.variance_hz == pytest.approx(1, rel=2e-5)
    assert math.isfinite(result.ci_upper_hz)


def test_conversion_failure_does_not_commit():
    graph = nx.Graph()
    accumulator = NuFTelemetryAccumulator(graph=graph)
    before = accumulator.record_counts(1, 1)
    graph.graph["HZ_STR_BRIDGE"] = 1e308
    with pytest.raises(ValueError):
        accumulator.record_counts(1, 1)
    graph.graph["HZ_STR_BRIDGE"] = 1
    assert accumulator.snapshot() == before
    assert len(graph.graph["telemetry"]["nu_f_history"]) == 1


def test_failed_graph_record_does_not_install_reconfigured_accumulator():
    graph = nx.Graph()
    before = record_nu_f_window(graph, 3, 2)
    accumulator = graph.graph["_tnfr_nu_f_accumulator"]
    history = graph.graph["telemetry"]["nu_f_history"][:]
    with pytest.raises(ValueError):
        record_nu_f_window(graph, 10**400, 1, confidence_level=0.9)
    assert graph.graph["_tnfr_nu_f_accumulator"] is accumulator
    assert accumulator.snapshot() == before
    assert graph.graph["telemetry"]["nu_f_history"] == history
    result = record_nu_f_window(graph, 4, 1, confidence_level=0.9)
    assert graph.graph["_tnfr_nu_f_accumulator"] is not accumulator
    assert result.total_reorganisations == 4
    assert result.confidence_level == 0.9


def test_failed_first_graph_record_keeps_graph_metadata_untouched():
    graph = nx.Graph()
    with pytest.raises(ValueError):
        record_nu_f_window(graph, 10**400, 1)
    assert graph.graph == {}


def test_graph_record_preserves_unspecified_nondefault_history_configuration():
    graph = nx.Graph()
    record_nu_f_window(graph, 3, 2, history_limit=2)
    accumulator = graph.graph["_tnfr_nu_f_accumulator"]
    result = record_nu_f_window(graph, 1, 1)
    assert graph.graph["_tnfr_nu_f_accumulator"] is accumulator
    assert accumulator.history_limit == 2
    assert result.total_reorganisations == 4
    assert result.total_duration == 3
    default_graph = nx.Graph()
    record_nu_f_window(default_graph, 1, 1)
    assert default_graph.graph["_tnfr_nu_f_accumulator"].history_limit == 128


def test_reconfiguration_changes_only_supplied_settings():
    graph = nx.Graph()
    record_nu_f_window(graph, 3, 2, history_limit=2, window_limit=3)
    result = record_nu_f_window(graph, 1, 1, confidence_level=0.9)
    accumulator = graph.graph["_tnfr_nu_f_accumulator"]
    assert accumulator.history_limit == 2
    assert accumulator.window_limit == 3
    assert result.confidence_level == 0.9
    assert result.total_reorganisations == 1


@pytest.mark.parametrize("name", ["history_limit", "window_limit"])
@pytest.mark.parametrize("value", [True, 1.5, "1", math.nan, 0])
def test_limit_domain_is_validated_before_reusing_an_accumulator(name, value):
    graph = nx.Graph()
    accumulator = ensure_nu_f_telemetry(graph, history_limit=1, window_limit=1)
    with pytest.raises((TypeError, ValueError)):
        ensure_nu_f_telemetry(graph, **{name: value})
    assert graph.graph["_tnfr_nu_f_accumulator"] is accumulator


@pytest.mark.parametrize("value", [math.nan, True, "0.95", Fraction(1, 10**400)])
def test_confidence_domain_is_validated_before_reusing_an_accumulator(value):
    graph = nx.Graph()
    accumulator = ensure_nu_f_telemetry(graph)
    with pytest.raises((TypeError, ValueError)):
        ensure_nu_f_telemetry(graph, confidence_level=value)
    assert graph.graph["_tnfr_nu_f_accumulator"] is accumulator


def test_confidence_next_to_one_uses_representable_tail():
    result = NuFTelemetryAccumulator(
        confidence_level=math.nextafter(1.0, 0.0)
    ).record_counts(1, 1)
    assert 9 < result.ci_upper_hz_str < 10


def test_explicit_empty_graph_overrides_attached_graph():
    attached = nx.path_graph(2)
    explicit = nx.Graph()
    explicit.graph["HZ_STR_BRIDGE"] = 3
    accumulator = NuFTelemetryAccumulator(graph=attached)
    result = accumulator.record_counts(2, 1, graph=explicit)
    assert result.rate_hz == 6
    assert "telemetry" not in attached.graph
    assert explicit.graph["telemetry"]["nu_f_history"][-1] == result.as_payload()


def test_plain_mapping_attachment_is_retained_without_invented_unit_conversion():
    mapping = {}
    accumulator = NuFTelemetryAccumulator(graph=mapping)
    result = accumulator.record_counts(2, 1)
    assert result.rate_hz_str == 2
    assert result.rate_hz is None
    assert mapping["telemetry"]["nu_f_history"][-1] == result.as_payload()


def test_altered_record_is_readmitted_before_aggregation():
    window = NuFWindow(1, 1)
    object.__setattr__(window, "duration", True)
    accumulator = NuFTelemetryAccumulator()
    with pytest.raises(TypeError):
        accumulator.record_window(window)
    assert accumulator.snapshot().windows == ()

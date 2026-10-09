"""Buffered event evidence survives batching, caller mutation and I/O failures."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tnfr.telemetry import unified_telemetry_system as telemetry


def _sink(path, **kwargs):
    return telemetry.TNFRUnifiedTelemetrySystem(
        telemetry.TelemetryConfiguration(
            output_directory=path, async_emission=False, **kwargs
        )
    )


def _emit(sink, channel, **kwargs):
    if channel == "structural":
        return sink.emit_structural_event(coherence=0.5, **kwargs)
    if channel == "performance":
        return sink.emit_performance_event("observed operation", 1.0, **kwargs)
    return sink.emit_failure_event("validation", "declared failure", **kwargs)


def _records(path, file_format="jsonl"):
    result = []
    for filename in path.glob(f"*.{file_format}"):
        content = filename.read_text(encoding="utf-8")
        result.extend(
            json.loads(content)
            if file_format == "json"
            else [json.loads(line) for line in content.splitlines()]
        )
    return result


@pytest.mark.parametrize("channel", ["structural", "performance", "failure"])
@pytest.mark.parametrize("file_format", ["json", "jsonl"])
def test_batches_in_same_second_never_overwrite_earlier_events(
    tmp_path, monkeypatch, channel, file_format
):
    monkeypatch.setattr(telemetry.time, "time", lambda: 1000.0)
    sink = _sink(tmp_path, batch_size=2, file_format=file_format)
    expected = {
        _emit(sink, channel, metadata={"description": "phase φ"}) for _ in range(4)
    }
    sink.cleanup()
    records = _records(tmp_path, file_format)
    assert len(records) == 4
    assert {record["event_id"] for record in records} == expected
    assert all(record["metadata"] == {"description": "phase φ"} for record in records)
    assert sink.get_statistics()["bytes_emitted"] == sum(
        filename.stat().st_size for filename in tmp_path.glob(f"*.{file_format}")
    )


@pytest.mark.parametrize("channel", ["structural", "performance", "failure"])
def test_global_and_channel_switches_suppress_collection(tmp_path, channel):
    path = tmp_path / "disabled"
    global_off = _sink(path, enable_telemetry=False)
    assert _emit(global_off, channel) == ""
    assert global_off.start_correlation("disabled") == ""
    global_off.cleanup()
    assert global_off.get_statistics()["total_events"] == 0
    assert not path.exists()
    channel_off = _sink(path, **{f"enable_{channel}_telemetry": False})
    assert _emit(channel_off, channel) == ""
    channel_off.cleanup()
    assert not path.exists()


def test_failed_write_retains_detached_batch_and_successful_retry(
    tmp_path, monkeypatch
):
    sink = _sink(tmp_path)
    metadata = {"steps": [1], "unavailable": None}
    event_id = _emit(sink, "structural", metadata=metadata)
    metadata["steps"].append(2)
    original = telemetry.safe_write

    def fail_write(*args, **kwargs):
        raise OSError("unavailable destination")

    monkeypatch.setattr(telemetry, "safe_write", fail_write)
    with pytest.raises(OSError, match="destination"):
        sink.flush_all()
    stats = sink.get_statistics()
    assert stats["buffer_sizes"]["structural"] == 1
    assert stats["bytes_emitted"] == 0
    assert not _records(tmp_path)
    monkeypatch.setattr(telemetry, "safe_write", original)
    sink.flush_all()
    sink.cleanup()
    records = _records(tmp_path)
    assert len(records) == 1 and records[0]["event_id"] == event_id
    assert records[0]["metadata"] == {"steps": [1], "unavailable": None}


def test_atomic_replace_failure_keeps_batch_and_leaves_no_partial_file(
    tmp_path, monkeypatch
):
    from tnfr.utils import io

    sink = _sink(tmp_path)
    _emit(sink, "performance")
    original = io.os.replace

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        sink.cleanup()
    assert sink.get_statistics()["buffer_sizes"]["performance"] == 1
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(io.os, "replace", original)
    sink.cleanup()
    assert len(_records(tmp_path)) == 1
    with pytest.raises(RuntimeError, match="closed"):
        _emit(sink, "performance")


@pytest.mark.parametrize(
    "metadata", [{"value": float("nan")}, {"value": object()}, {"value": "\ud800"}]
)
def test_unserializable_payload_rejects_before_counting_or_poisoning_buffer(
    tmp_path, metadata
):
    sink = _sink(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        _emit(sink, "structural", metadata=metadata)
    assert sink.get_statistics()["total_events"] == 0
    assert sink.get_statistics()["buffer_sizes"]["structural"] == 0
    sink.cleanup()
    assert not _records(tmp_path)


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_unsupported_format_rejects_instead_of_discarding_events(tmp_path, file_format):
    with pytest.raises(ValueError, match="file_format"):
        _sink(tmp_path, file_format=file_format)


def test_concurrent_channel_flushes_preserve_exact_event_inventory(tmp_path):
    sink = _sink(tmp_path, batch_size=3)
    context = {"steps": [1]}
    correlation = sink.start_correlation("parallel", context)
    context["steps"].append(2)

    def produce(index):
        event_id = _emit(sink, "structural", correlation_id=correlation)
        if index % 2 == 0:
            sink.flush_all()
        return event_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        expected = set(pool.map(produce, range(20)))
    sink.cleanup()
    records = _records(tmp_path)
    assert len(records) == len(expected) == 20
    assert {record["event_id"] for record in records} == expected
    summary = sink.end_correlation(correlation)
    assert summary["event_count"] == 20
    assert summary["context"] == {"steps": [1]}


def test_cleanup_prevents_late_timer_callback_from_restarting_collection(
    tmp_path, monkeypatch
):
    timers = []

    class Timer:
        def __init__(self, interval, callback):
            self.callback = callback
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(telemetry.threading, "Timer", Timer)
    sink = telemetry.TNFRUnifiedTelemetrySystem(
        telemetry.TelemetryConfiguration(output_directory=tmp_path)
    )
    _emit(sink, "structural")
    sink.cleanup()
    assert timers[0].cancelled
    timers[0].callback()  # A callback already dispatched when cancellation occurred.
    assert len(timers) == 1
    assert len(_records(tmp_path)) == 1


@pytest.mark.parametrize(
    "option,value",
    [("batch_size", True), ("batch_size", 0), ("enable_telemetry", "false")],
)
def test_consumed_configuration_rejects_invalid_values(tmp_path, option, value):
    with pytest.raises((TypeError, ValueError)):
        _sink(tmp_path, **{option: value})


@pytest.mark.parametrize("channel", ["structural", "performance", "failure"])
def test_invalid_correlation_cannot_partially_enqueue_event(tmp_path, channel):
    sink = _sink(tmp_path)
    with pytest.raises(TypeError, match="correlation_id"):
        _emit(sink, channel, correlation_id=["invalid"])
    assert sink.get_statistics()["total_events"] == 0
    assert not any(sink.get_statistics()["buffer_sizes"].values())
    sink.cleanup()
    assert not _records(tmp_path)


def test_event_metadata_reuses_sdk_json_name_collision_boundary(tmp_path):
    from tnfr.sdk.utils import export_to_json

    metadata = {"nested": [{1: "numeric key", "1": "text key"}]}
    sink = _sink(tmp_path)
    with pytest.raises(ValueError, match="keys collide"):
        _emit(sink, "structural", metadata=metadata)
    with pytest.raises(ValueError, match="keys collide"):
        export_to_json(metadata, tmp_path / "report.json")
    assert sink.get_statistics()["total_events"] == 0
    sink.cleanup()
    assert list(tmp_path.iterdir()) == []

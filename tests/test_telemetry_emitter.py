"""Partial canonical telemetry and JSON emission without runtime execution."""

import json
import math
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from tnfr.metrics.telemetry import TelemetryEmitter
from tnfr.physics.fields import (
    UndefinedPhaseCurvatureError,
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)
from tnfr.utils.cache import reset_global_cache


@pytest.fixture(autouse=True)
def clear_field_caches():
    reset_global_cache()
    yield
    reset_global_cache()


def _star(*, cancellation):
    graph = nx.star_graph(4)
    phases = (
        (0.3, 0.0, 0.0, math.pi, -math.pi)
        if cancellation
        else (0.3, 0.1, 0.2, 0.4, 0.6)
    )
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node].update(EPI=0.25, nu_f=1.0, theta=phase, delta_nfr=0.125)
    return graph


@pytest.mark.parametrize("include_extended", (False, True))
def test_safe_record_preserves_available_tetrad_when_curvature_is_undefined(
    tmp_path, include_extended
):
    graph = _star(cancellation=True)
    nodes_before = deepcopy(dict(graph.nodes(data=True)))
    path = tmp_path / "partial.jsonl"
    event = TelemetryEmitter(path, safe=True, include_extended=include_extended).record(
        graph, step=7
    )
    assert event.metrics["phi_s"] == compute_structural_potential(graph)
    assert event.metrics["phase_grad"] == compute_phase_gradient(graph)
    assert event.metrics["xi_c"] == estimate_coherence_length(graph)
    assert "phase_curv" not in event.metrics
    assert "unified_fields" not in event.metrics
    assert ("phase_current" in event.metrics) is include_extended
    assert ("dnfr_flux" in event.metrics) is include_extended
    failed_fields = (
        ("phase_curv", "unified_fields") if include_extended else ("phase_curv",)
    )
    for field in failed_fields:
        error = event.metrics["field_errors"][field]
        assert error["type"] == "UndefinedPhaseCurvatureError"
        assert "represented" in error["message"]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["step"] == 7
    assert saved["metrics"]["phase_grad"]["0"] == event.metrics["phase_grad"][0]
    assert saved["metrics"]["field_errors"] == event.metrics["field_errors"]
    assert dict(graph.nodes(data=True)) == nodes_before


def test_individual_tetrad_mode_does_not_call_extended_or_unified_suites(
    tmp_path, monkeypatch
):
    from tnfr.metrics import telemetry
    from tnfr.physics import fields

    def forbidden(_graph):
        raise AssertionError("the individual mode must not call a composite suite")

    monkeypatch.setattr(telemetry, "compute_extended_canonical_suite", forbidden)
    monkeypatch.setattr(telemetry, "compute_unified_telemetry", forbidden)
    monkeypatch.setattr(fields, "compute_unified_telemetry", forbidden)
    event = TelemetryEmitter(
        tmp_path / "individual.jsonl", include_extended=False, safe=False
    ).record(
        _star(cancellation=False),
    )
    assert all(
        name in event.metrics for name in ("phi_s", "phase_grad", "phase_curv", "xi_c")
    )
    assert "field_errors" not in event.metrics
    assert "unified_fields" not in event.metrics


def test_strict_record_propagates_curvature_error_without_an_event(tmp_path):
    path = tmp_path / "strict.jsonl"
    emitter = TelemetryEmitter(path, safe=False)
    with pytest.raises(UndefinedPhaseCurvatureError, match="represented"):
        emitter.record(_star(cancellation=True))
    assert emitter._buffer == []
    assert not path.exists()


@pytest.mark.parametrize("safe", (False, True))
def test_ordinary_record_roundtrips_actual_unified_numpy_fields(tmp_path, safe):
    graph = _star(cancellation=False)
    path = tmp_path / "ordinary.jsonl"
    event = TelemetryEmitter(path, safe=safe).record(
        graph,
        step=3,
        operator="IL",
        extra={"seed": np.int64(12), "trace": np.array([0.0, 0.5])},
    )
    assert "field_errors" not in event.metrics
    assert event.metrics["phase_curv"] == compute_phase_curvature(graph)
    saved = json.loads(path.read_text(encoding="utf-8"))
    actual = event.metrics["unified_fields"]["complex_field"]["psi_real"]
    assert isinstance(actual, np.ndarray)
    assert (
        saved["metrics"]["unified_fields"]["complex_field"]["psi_real"]
        == actual.tolist()
    )
    assert saved["extra"] == {"seed": 12, "trace": [0.0, 0.5]}
    assert saved["operator"] == "IL"


@pytest.mark.parametrize("safe", (False, True))
def test_unsupported_extra_is_not_stringified_or_written_as_success(tmp_path, safe):
    path = tmp_path / "unsupported.jsonl"
    emitter = TelemetryEmitter(path, safe=safe)
    with pytest.raises(TypeError, match="not JSON serializable"):
        emitter.record(_star(cancellation=False), extra={"opaque": object()})
    assert not path.exists()
    assert len(emitter._buffer) == 1  # Retained for caller correction/retry.


def test_partial_core_metrics_have_a_mirror_without_duplicate_json_on_flush(
    tmp_path, monkeypatch
):
    from tnfr.metrics import telemetry

    def unavailable(_graph):
        raise ValueError("core metric unavailable")

    monkeypatch.setattr(telemetry, "compute_coherence", unavailable)
    monkeypatch.setattr(telemetry, "sense_index", unavailable)
    path = tmp_path / "partial_core.jsonl"
    emitter = TelemetryEmitter(
        path, safe=True, include_extended=False, human_mirror=True
    )
    emitter.record(_star(cancellation=False), step=4)
    assert emitter._buffer == []
    emitter.flush()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    mirror = path.with_suffix(".log").read_text(encoding="utf-8")
    assert "C=unavailable" in mirror and "Si=unavailable" in mirror
    assert len(mirror.splitlines()) == 1


def test_successful_unified_collection_reuses_its_extended_suite(tmp_path, monkeypatch):
    from tnfr.metrics import telemetry
    from tnfr.physics import fields

    original = fields.compute_extended_canonical_suite
    calls = []

    def counted(graph):
        calls.append(tuple(graph))
        return original(graph)

    monkeypatch.setattr(fields, "compute_extended_canonical_suite", counted)
    monkeypatch.setattr(telemetry, "compute_extended_canonical_suite", counted)
    event = TelemetryEmitter(tmp_path / "reuse.jsonl", safe=False).record(
        _star(cancellation=False)
    )
    assert len(calls) == 1
    extended = event.metrics["unified_fields"]["extended_canonical"]
    assert event.metrics["phase_current"] == extended["phase_current"]
    assert event.metrics["dnfr_flux"] == extended["dnfr_flux"]


def test_mirror_format_failure_precedes_either_file_append(tmp_path):
    path = tmp_path / "bad_mirror.jsonl"
    emitter = TelemetryEmitter(
        path, include_extended=False, human_mirror=True, flush_interval=2
    )
    event = emitter.record(_star(cancellation=False))
    # JSON can encode this caller-edited metric, but numeric mirror formatting
    # cannot. A failed flush must not append the JSON record before finding out.
    event.metrics["coherence_total"] = "not a numeric metric"
    with pytest.raises(ValueError, match="format code"):
        emitter.flush()
    assert not path.exists()
    assert not path.with_suffix(".log").exists()
    assert len(emitter._buffer) == 1

"""Frozen control transport and wiring without evaluating its research source."""

import hashlib
import zipfile
from fractions import Fraction as Q

import pytest

from benchmarks import conservative_regional_organization as owner
from tnfr.mathematics._rational_interval import I, cos, pi_interval
from tnfr.utils.io import json_loads


def test_preparation_does_not_consume_the_reserved_response(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("preparation must not evaluate the full trajectory")

    monkeypatch.setattr(owner, "bound_sine_flow", forbidden)
    protocol = owner.prepare_protocol()
    declaration = protocol["declaration"]
    prepared = owner._preparation(declaration)
    assert prepared["end_time"] / prepared["time_step"] == 256
    assert prepared["model"].effective_weights == (0, 1)
    assert len(prepared["initial"]) == 23
    assert prepared["initial"][1].lo == -3
    assert prepared["initial"][10].hi == 3
    assert declaration["minimum_duration"] == "1/4"
    assert declaration["acute_margin"] == "1/16"


def test_evaluation_uses_shared_forecast_and_whole_time_observer(monkeypatch):
    declaration = json_loads(owner.DECLARATION.read_bytes())
    calls, forecast = [], object()

    def run(**kwargs):
        calls.append(kwargs)
        assert kwargs == owner._preparation(declaration)
        return forecast

    class Report:
        def to_dict(self):
            return {"report": {"outcome": "unresolved"}}

    def assess(actual, **kwargs):
        assert actual is forecast
        assert kwargs == dict(
            cycle_indices=(5, 6, 7, 8, 9),
            minimum_duration=Q(1, 4),
            acute_margin=Q(1, 16),
        )
        return Report()

    monkeypatch.setattr(owner, "bound_sine_flow", run)
    monkeypatch.setattr(owner, "assess_sine_regional_organization", assess)
    assert owner.evaluate(declaration)["report"]["outcome"] == "unresolved"
    assert len(calls) == 1


@pytest.mark.parametrize("outcome", ("unresolved", "acute_winding_excluded_on_horizon"))
def test_failed_target_records_are_retained_without_retry(
    monkeypatch, tmp_path, outcome
):
    monkeypatch.setattr(owner, "_files", lambda: {"declared.py": b"unchanged source"})
    calls = []

    def evaluate(declaration):
        calls.append(declaration)
        return {"report": {"outcome": outcome}}

    monkeypatch.setattr(owner, "evaluate", evaluate)
    destination = tmp_path / "response.json"
    assert owner.main(["--prepare", "--output", str(destination)]) == 0
    assert not calls and not destination.exists()
    assert owner.main(["--output", str(destination)]) == 1
    record = json_loads(destination.read_bytes())
    assert record["passed"] is False and record["evaluation_error"] is None
    assert record["response"]["report"]["outcome"] == outcome
    assert len(calls) == 1
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        owner.main(["--output", str(destination)])
    assert destination.read_bytes() == before and len(calls) == 1


def test_changed_source_is_rejected_before_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(owner, "_files", lambda: {"declared.py": b"old"})
    destination = tmp_path / "response.json"
    owner.main(["--prepare", "--output", str(destination)])
    monkeypatch.setattr(owner, "_files", lambda: {"declared.py": b"new"})

    def forbidden(declaration):
        pytest.fail("a changed source must not evaluate the reserved response")

    monkeypatch.setattr(owner, "evaluate", forbidden)
    with pytest.raises(ValueError, match="frozen source"):
        owner.main(["--output", str(destination)])
    assert not destination.exists()


def test_retained_response_excludes_winding_from_complete_primitive_tubes(monkeypatch):
    """Check saved enclosure implications and binding, without replay/authentication."""

    def forbidden(*args, **kwargs):
        pytest.fail("retained evidence must not rerun its scientific producer")

    monkeypatch.setattr(owner, "bound_sine_flow", forbidden)
    monkeypatch.setattr(owner, "evaluate", forbidden)
    output = owner.DECLARATION.with_name("response-v1.json")
    protocol_bytes = output.with_suffix(".protocol.json").read_bytes()
    protocol = json_loads(protocol_bytes)
    record = json_loads(output.read_bytes())
    archive = output.with_suffix(".source.zip")
    owner._verify_archive(archive, protocol["source_sha256"])
    assert record["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert (
        record["source_archive_sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert record["protocol"] == protocol
    declaration = protocol["declaration"]
    with zipfile.ZipFile(archive) as bundle:
        archived_declaration = bundle.read(
            owner.DECLARATION.relative_to(owner.ROOT).as_posix()
        )
    assert json_loads(archived_declaration) == declaration

    def rational(value):
        return Q(value["numerator"], value["denominator"])

    def interval(value):
        return I(rational(value["lo"]), rational(value["hi"]))

    report = record["response"]["report"]
    forecast = report["forecast"]
    cycle = tuple(declaration["receiver_cycle_indices"])
    size = len(declaration["nodes"])
    assert len(cycle) == 5
    at, end, duration = (
        Q(declaration[key]) for key in ("observation_time", "end_time", "time_step")
    )
    assert len(forecast["steps"]) * duration == end - at
    assert len(report["steps"]) == len(forecast["steps"])
    pi = pi_interval()
    for step, observed in zip(forecast["steps"], report["steps"]):
        assert rational(step["time"]) == at
        assert rational(step["duration"]) == duration
        phases = tuple(map(interval, step["tube"][size : 2 * size]))
        gaps = tuple(
            phases[j] - phases[i] for i, j in zip(cycle, cycle[1:] + cycle[:1])
        )
        # These are already principal differences. Their exact underlying
        # lifted-node differences telescope, so the winding is zero at every
        # time in the whole tube, irrespective of cached report verdicts.
        assert all(-pi.lo < gap.lo <= gap.hi < pi.lo for gap in gaps)
        assert tuple(map(interval, observed["cycle_raw_gap_bounds"])) == gaps
        assert observed["edge_turn_offsets"] == [0] * len(cycle)
        assert observed["winding"] == 0
        # In an acute five-cycle with winding +/-1 every two-hop sum lies
        # strictly between pi/2 and pi in magnitude, hence has negative cosine.
        # A nonnegative two-hop cosine independently rules that target out.
        assert any(
            cos(phases[cycle[(k + 2) % 5]] - phases[cycle[k]]).lo >= 0 for k in range(5)
        )
        at += duration
    assert at == end == rational(forecast["validated_end_time"])

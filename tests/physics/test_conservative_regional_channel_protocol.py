"""Historical enclosure transport and retrospective wiring, without IVP replay."""

import hashlib
import shutil
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from benchmarks import conservative_regional_organization as owner
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_regional as reader
from tnfr.physics.relational_sine_forecast import SineForecast
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module")
def retained_source():
    return owner.DECLARATION.with_name("response-v1.json")


@pytest.fixture(scope="module")
def projected_forecast(retained_source):
    return json_loads(retained_source.read_bytes())["response"]["report"]["forecast"]


@pytest.fixture(autouse=True)
def no_trajectory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("retrospective transport must not evaluate another trajectory")

    monkeypatch.setattr(owner, "bound_sine_flow", forbidden)
    monkeypatch.setattr(owner, "evaluate", forbidden)


def test_historical_loader_preserves_complete_exact_evidence(
    retained_source, projected_forecast, monkeypatch
):
    def forbidden():
        pytest.fail("historical source must not be compared with today's source")

    monkeypatch.setattr(owner, "_files", forbidden)
    forecast, cycle, metadata = owner._load_channel_source(retained_source)
    assert isinstance(forecast, SineForecast)
    assert cycle == (5, 6, 7, 8, 9)
    assert len(forecast.steps) == len(projected_forecast["steps"]) == 256
    assert forecast.validated_end_time == forecast.end_time == 12
    assert forecast.time_step == Q(3, 64)
    assert forecast.endpoint == forecast.steps[-1].endpoint
    for actual, saved in zip(forecast.steps, projected_forecast["steps"]):
        assert actual.tube == tuple(map(owner._interval_projection, saved["tube"]))
        assert actual.endpoint == tuple(
            map(owner._interval_projection, saved["endpoint"])
        )
        assert actual.local_remainder_bounds == tuple(
            map(owner._interval_projection, saved["local_remainder_bounds"])
        )
    assert (
        metadata["source_response_sha256"]
        == hashlib.sha256(retained_source.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "change",
    (
        lambda v: v["model"].update(phase_weight=True),
        lambda v: v["model"].update(phase_weight=2),
        lambda v: v.update(order=True),
        lambda v: v["neighbors"][0].__setitem__(0, True),
        lambda v: v["visible_capacity"][0].update(numerator=True),
        lambda v: v["visible_capacity"].__setitem__(0, 1.0),
        lambda v: v["visible_capacity"][0].update(numerator=2, denominator=2),
        lambda v: v["steps"][0].pop("local_remainder_bounds"),
        lambda v: v["steps"][0]["tube"][0]["lo"].update(numerator=99, denominator=1),
        lambda v: v.update(prior_admission={}),
        lambda v: v.update(unconsumed_state=0),
    ),
)
def test_private_projection_decoder_rejects_lossy_or_incomplete_state(
    projected_forecast, change
):
    candidate = deepcopy(projected_forecast)
    change(candidate)
    with pytest.raises((TypeError, ValueError)):
        owner._forecast_projection(candidate)


@pytest.fixture
def copied_source(retained_source, tmp_path):
    destination = tmp_path / "historical.json"
    for suffix in (".json", ".protocol.json", ".source.zip"):
        shutil.copyfile(
            retained_source.with_suffix(suffix), destination.with_suffix(suffix)
        )
    return destination


@pytest.mark.parametrize("changed", ("protocol", "archive", "initial", "clock"))
def test_loader_rejects_broken_binding_or_changed_preparation(copied_source, changed):
    if changed == "archive":
        with copied_source.with_suffix(".source.zip").open("ab") as stream:
            stream.write(b"changed archive")
    elif changed == "protocol":
        with copied_source.with_suffix(".protocol.json").open("ab") as stream:
            stream.write(b"\n")
    else:
        record = json_loads(copied_source.read_bytes())
        forecast = record["response"]["report"]["forecast"]
        if changed == "initial":
            forecast["initial_box"][0] = {
                "lo": {"numerator": 1, "denominator": 1},
                "hi": {"numerator": 1, "denominator": 1},
            }
        else:
            forecast["end_time"] = {"numerator": 24, "denominator": 1}
        copied_source.write_text(owner.evidence._encoded(record), encoding="utf-8")
    with pytest.raises(ValueError):
        owner._load_channel_source(copied_source)


@pytest.mark.parametrize("failure", (None, "source_change", "reader_failure"))
def test_analysis_is_separate_write_once_and_uses_only_the_shared_reader(
    retained_source, monkeypatch, tmp_path, failure
):
    source_files = {"analysis.py": b"original retrospective reader"}
    monkeypatch.setattr(owner, "_files", lambda: dict(source_files))
    before = retained_source.read_bytes()
    calls = []

    class Report:
        def to_dict(self):
            return {
                "schema": "tnfr.sine-regional-channel-history.v1",
                "report": {"fixture": "reader wiring only; no scientific verdict"},
            }

    def assess(forecast, *, cycle_indices):
        calls.append(forecast)
        assert isinstance(forecast, SineForecast)
        assert cycle_indices == (5, 6, 7, 8, 9)
        assert forecast.initial_box[1] == I(-3)
        assert len(forecast.steps) == 256
        if failure == "reader_failure":
            raise ValueError("injected inadmissible supplied enclosure")
        if failure == "source_change":
            source_files["analysis.py"] = b"changed during analysis"
        return Report()

    monkeypatch.setattr(reader, "assess_sine_regional_channels", assess)
    output = tmp_path / "analysis.json"
    args = ["--analyze-channels", str(retained_source), "--output", str(output)]
    assert owner.main(args) == (0 if failure is None else 1)
    record = json_loads(output.read_bytes())
    assert record["schema"] == "tnfr.conservative-regional-channel-analysis.v1"
    assert record["source_response_sha256"] == hashlib.sha256(before).hexdigest()
    archive = output.with_suffix(".source.zip")
    owner._verify_archive(archive, record["analysis_source_sha256"])
    assert (
        record["analysis_source_archive_sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert (record["analysis_error"] is None) == (failure is None)
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        owner.main(args)
    assert output.read_bytes() == saved and retained_source.read_bytes() == before
    assert len(calls) == 1


def test_retrospective_mode_requires_an_explicit_destination(retained_source):
    with pytest.raises(SystemExit):
        owner.main(["--analyze-channels", str(retained_source)])

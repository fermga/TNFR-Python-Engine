"""Synthetic transport controls; never acquire a registration response.

The real producer is replaced before either explicit command is called. These
tests check freeze consistency, single-attempt execution and durable failure
prefixes independently of the scientific certificate's acceptance policy.
"""

import importlib.util
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/assets/weighted_star_registration_episode/produce.py"
)


@pytest.fixture
def transport(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "weighted_star_episode_transport", SCRIPT
    )
    instrument = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instrument)
    output = tmp_path / "evidence"
    output.mkdir()
    member = tmp_path / "owner.py"
    member.write_bytes(b"# synthetic archived source\n")
    monkeypatch.setattr(instrument, "ROOT", tmp_path)
    monkeypatch.setattr(instrument, "OUT", output)
    monkeypatch.setattr(instrument, "_source_paths", lambda: (member,))
    calls = []
    protocol = {"horizon": Q(8), "order": 1, "scope": "synthetic transport only"}

    def prepare_source():
        calls.append("prepare")
        return {"state": (Q(1, 4), Q(201, 512)), "scope": "synthetic"}

    def evaluate_protocol(preparation, frozen, *, progress):
        calls.append("evaluate")
        assert frozen == protocol
        assert preparation["state"] == (Q(1, 4), Q(201, 512))
        progress("nonlinear", 0, {"cell": 0, "value": Q(3, 8)})
        progress("tangent", 0, {"cell": 0, "value": Q(5, 8)})
        return {"branches": ("unavailable", "unavailable"), "value": Q(7, 16)}

    def summarize_response(response, frozen):
        calls.append("summarize")
        assert frozen == protocol
        assert type(response["value"]) is Q
        assert response["value"] == Q(7, 16)
        return {"status": "unavailable", "episode_certified": False}

    owner = SimpleNamespace(
        frozen_protocol=lambda: dict(protocol),
        prepare_source=prepare_source,
        evaluate_protocol=evaluate_protocol,
        summarize_response=summarize_response,
    )
    monkeypatch.setattr(instrument, "_owner", lambda: owner)
    return instrument, owner, calls, member


def test_command_requires_explicit_action_without_preparing_any_source(transport):
    instrument, _, calls, _ = transport
    with pytest.raises(SystemExit) as exc:
        instrument.main([])
    assert exc.value.code == 2
    assert calls == []
    assert list(instrument.OUT.iterdir()) == []


def test_prepare_archives_identical_bytes_and_never_evaluates(transport):
    instrument, _, calls, member = transport
    assert instrument.prepare() == 0
    assert calls == ["prepare"]
    manifest = instrument._read("source-manifest.json")["files"]
    assert manifest == (instrument._receipt(instrument.ROOT, "owner.py"),)
    with ZipFile(instrument.OUT / "source.zip") as archive:
        assert archive.namelist() == ["owner.py"]
        assert archive.read("owner.py") == member.read_bytes()
    freeze = instrument._read("freeze.json")
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert {row["path"] for row in freeze["artifacts"]} == set(instrument.FROZEN_NAMES)
    for row in freeze["artifacts"]:
        assert instrument._receipt(instrument.OUT, row["path"]) == row
    assert not (instrument.OUT / "attempt.json").exists()
    with pytest.raises(FileExistsError):
        instrument.prepare()
    assert calls == ["prepare"]


def test_evaluate_once_keeps_scientific_unavailability_separate_from_completion(
    transport,
):
    instrument, _, calls, _ = transport
    instrument.prepare()
    assert instrument.evaluate() == 0
    assert calls == ["prepare", "evaluate", "summarize"]
    attempt, result = (
        instrument._read(name) for name in ("attempt.json", "result.json")
    )
    assert attempt["attempt_number"] == 1
    assert attempt["freeze"] == instrument._receipt(instrument.OUT, "freeze.json")
    assert result["status"] == "evaluated" and result["failure"] is None
    assert result["scientific_summary"] == {
        "status": "unavailable",
        "episode_certified": False,
    }
    assert {row["path"] for row in result["artifacts"]} == {
        "records/nonlinear-0000.json",
        "records/tangent-0000.json",
        "response.json",
    }
    for row in result["artifacts"]:
        assert instrument._receipt(instrument.OUT, row["path"]) == row
    with pytest.raises(FileExistsError):
        instrument.evaluate()
    assert calls == ["prepare", "evaluate", "summarize"]


@pytest.mark.parametrize("changed", ("source", "protocol", "archive"))
def test_changed_frozen_input_rejects_before_exclusive_attempt(transport, changed):
    instrument, _, calls, member = transport
    instrument.prepare()
    target = {
        "source": member,
        "protocol": instrument.OUT / "protocol.json",
        "archive": instrument.OUT / "source.zip",
    }[changed]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen content differs"):
        instrument.evaluate()
    assert calls == ["prepare"]
    assert not (instrument.OUT / "attempt.json").exists()


def test_protocol_comparison_does_not_equate_boolean_and_integer(transport):
    instrument, owner, calls, _ = transport
    instrument.prepare()
    original = owner.frozen_protocol
    owner.frozen_protocol = lambda: {**original(), "order": True}
    with pytest.raises(ValueError, match="protocol differs"):
        instrument.evaluate()
    assert calls == ["prepare"]
    assert not (instrument.OUT / "attempt.json").exists()


def test_interrupt_keeps_raw_prefix_and_blocks_retry(transport):
    instrument, owner, calls, _ = transport
    instrument.prepare()

    def interrupted(preparation, protocol, *, progress):
        calls.append("evaluate")
        progress("nonlinear", 0, {"partial": Q(5, 16)})
        raise KeyboardInterrupt("synthetic interruption")

    owner.evaluate_protocol = interrupted
    assert instrument.evaluate() == 1
    result = instrument._read("result.json")
    assert result["status"] == "failed"
    assert result["failure"]["type"] == "KeyboardInterrupt"
    assert result["scientific_summary"] is None
    assert instrument._read("records/nonlinear-0000.json") == {"partial": Q(5, 16)}
    assert tuple(row["path"] for row in result["artifacts"]) == (
        "records/nonlinear-0000.json",
    )
    assert not (instrument.OUT / "response.json").exists()
    with pytest.raises(FileExistsError):
        instrument.evaluate()
    assert calls == ["prepare", "evaluate"]


def test_duplicate_progress_cannot_replace_the_first_record(transport):
    instrument, owner, _, _ = transport
    instrument.prepare()

    def duplicate(preparation, protocol, *, progress):
        progress("nonlinear", 0, {"first": True})
        progress("nonlinear", 0, {"first": False})

    owner.evaluate_protocol = duplicate
    assert instrument.evaluate() == 1
    assert instrument._read("records/nonlinear-0000.json") == {"first": True}
    assert instrument._read("result.json")["failure"]["type"] == "FileExistsError"


def test_source_edit_during_preparation_blocks_freeze_and_silent_retry(transport):
    instrument, owner, _, member = transport
    original = owner.prepare_source

    def changing_source():
        result = original()
        member.write_bytes(b"# changed during synthetic preparation\n")
        return result

    owner.prepare_source = changing_source
    with pytest.raises(ValueError, match="frozen content differs"):
        instrument.prepare()
    assert not (instrument.OUT / "freeze.json").exists()
    assert (
        instrument._read("preparation-failure.json")["failure"]["type"] == "ValueError"
    )
    with pytest.raises(FileExistsError):
        instrument.prepare()

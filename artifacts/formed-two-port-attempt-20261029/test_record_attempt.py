"""Bounded recorder/wrapper controls with fake subprocesses only."""

import importlib.util
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from tnfr.research.artifact_io import file_receipt
from tnfr.utils.io import json_loads

HERE = Path(__file__).resolve().parent


@pytest.fixture
def recorder(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "synthetic_record_attempt", HERE / "record_attempt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Main guard must prevent any execution.

    def forbidden(*args, **kwargs):
        pytest.fail("Recorder controls must not create any real subprocess")

    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    return module


def caller(**changes):
    result = {
        "schema": "tnfr.formed-two-port-caller.v1",
        "record_stage": "post_caller_write_result",
        "execution_passed": True,
        "caller_write_within_total_budget": True,
        "caller_elapsed_seconds": 1.0,
    }
    result.update(changes)
    return json.dumps(result).encode()


class Pipe(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.requested, self.observed = [], 0

    def read(self, count):
        self.requested.append(count)
        data = super().read(count)
        self.observed += len(data)
        return data


class Process:
    def __init__(self, stdout, stderr=b"", code=0, error=False):
        self.stdout, self.stderr = Pipe(stdout), Pipe(stderr)
        self.code, self.error = code, error
        self.returncode, self.killed = None, False

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        if self.error:
            raise OSError("synthetic wait failure")
        if not self.killed:
            self.returncode = self.code
        return self.returncode


def capture(recorder, monkeypatch, stdout, stderr=b"", code=0):
    process = Process(stdout, stderr, code)
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        assert kwargs["bufsize"] == 0
        return process

    monkeypatch.setattr(recorder.subprocess, "Popen", launch)
    captured = recorder.capture_wrapper(["inert-fake"], HERE)
    assert len(calls) == 1
    return captured, process


def test_valid_observed_exit_and_return_form_one_small_envelope(recorder, monkeypatch):
    captured, process = capture(recorder, monkeypatch, caller())
    data, accepted = recorder.envelope_bytes(captured)
    record = json_loads(data)
    assert accepted and not process.killed
    assert record["wrapper_exit_code"] == 0 and record["output_error"] is None
    assert record["stdout_complete"] and record["stderr_complete"]
    assert record["stdout_hash_scope"] == "complete stream"
    assert len(data) <= 32 * 1024
    assert "stdout" not in record and "stderr" not in record


@pytest.mark.parametrize(
    "body,code",
    [
        (caller(execution_passed=False), 0),
        (caller(), 2),
        (caller(caller_write_within_total_budget=False), 0),
        (caller(caller_elapsed_seconds=3601), 0),
        (caller(caller_elapsed_seconds=True), 0),
        (caller(record_stage="pre_caller_write_snapshot_not_final_verdict"), 0),
        (b"{}", 0),
        (b'{"partial"', 0),
    ],
)
def test_code_zero_cannot_promote_invalid_or_failed_return(
    recorder, monkeypatch, body, code
):
    captured, _ = capture(recorder, monkeypatch, body, code=code)
    data, accepted = recorder.envelope_bytes(captured)
    assert not accepted and len(data) <= 32 * 1024


def test_stdout_overflow_kills_and_hashes_only_observed_prefix(recorder, monkeypatch):
    captured, process = capture(recorder, monkeypatch, b"x" * (40 << 10))
    assert process.killed and captured["stdout_bytes"] <= (31 << 10) + 1
    assert len(captured["stdout"]) <= 31 << 10
    assert max(process.stdout.requested) <= 4096
    assert captured["stdout_hash_scope"] == "observed prefix only"
    assert not recorder.envelope_bytes(captured)[1]


def test_any_stderr_kills_with_at_most_1025_observed_bytes(recorder, monkeypatch):
    captured, process = capture(recorder, monkeypatch, caller(), b"e" * (10 << 10))
    assert process.killed and 0 < captured["stderr_bytes"] <= 1025
    assert len(captured["stderr"]) <= 1024
    assert captured["stderr_hash_scope"] == "observed prefix only"
    assert not recorder.envelope_bytes(captured)[1]


@pytest.mark.parametrize("kind", ["launch", "wait", "read", "timeout"])
def test_capture_failures_are_retained_and_never_retry(recorder, monkeypatch, kind):
    process = Process(caller(), error=kind == "wait")
    calls = []
    if kind == "read":

        def fail_read(count):
            raise OSError("synthetic pipe failure")

        process.stdout.read = fail_read
    if kind == "timeout":
        monkeypatch.setattr(recorder, "WRAPPER_SECONDS", 0)

    def launch(*args, **kwargs):
        calls.append(1)
        if kind == "launch":
            raise OSError("synthetic unavailable launch")
        return process

    monkeypatch.setattr(recorder.subprocess, "Popen", launch)
    captured = recorder.capture_wrapper(["inert-fake"], HERE)
    data, accepted = recorder.envelope_bytes(captured)
    assert len(calls) == 1 and not accepted
    assert json_loads(data)["output_error"]
    assert captured["capture_errors"] and len(data) <= 32 << 10
    if kind != "launch":
        assert process.killed


def test_envelope_overflow_retains_diagnostic_without_invalid_success(
    recorder, monkeypatch
):
    captured, _ = capture(recorder, monkeypatch, caller(padding="x" * 2000))
    monkeypatch.setattr(recorder, "LIMIT", 1200)
    data, accepted = recorder.envelope_bytes(captured)
    assert not accepted and len(data) <= 1200
    record = json_loads(data)
    assert record["launcher_return"] is None and "exceeds" in record["output_error"]


@pytest.mark.parametrize("passed", [True, False])
def test_single_persisted_record_uses_same_acceptance_for_exit(
    recorder, monkeypatch, tmp_path, passed
):
    captured, _ = capture(recorder, monkeypatch, caller(execution_passed=passed))
    monkeypatch.setattr(recorder, "OUT", tmp_path)
    monkeypatch.setattr(recorder, "ROOT", tmp_path)
    monkeypatch.setattr(recorder, "capture_wrapper", lambda *args: captured)
    evidence = tmp_path / "frozen"
    evidence.write_bytes(b"synthetic preflight bytes")
    row = file_receipt(evidence, label="frozen")
    protocol = {"sources": [], "archive": row, "reviews": [], "controls": row}
    (tmp_path / "producing-protocol.json").write_text(json.dumps(protocol))
    assert recorder.main() == (0 if passed else 2)
    original = (tmp_path / "caller-result.json").read_bytes()
    assert len(original) <= 32 << 10
    with pytest.raises(FileExistsError):
        recorder.main()
    assert (tmp_path / "caller-result.json").read_bytes() == original


@pytest.mark.parametrize("outcome", ["success", "failure", "exception"])
def test_frozen_wrapper_emits_bounded_return_and_matching_exit(monkeypatch, outcome):
    from tnfr.research import formed_two_port_launch

    def launch(*args):
        if outcome == "exception":
            raise OSError("synthetic caller failure")
        return json.loads(caller(execution_passed=outcome == "success"))

    monkeypatch.setattr(formed_two_port_launch, "launch_formed_two_port", launch)

    class Output:
        buffer = io.BytesIO()

    output = Output()
    monkeypatch.setattr(sys, "stdout", output)
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(str(HERE / "launch_once.py"), run_name="__main__")
    assert exit.value.code == (0 if outcome == "success" else 2)
    data = output.buffer.getvalue()
    assert len(data) <= 31 << 10
    assert json_loads(data)["execution_passed"] is (outcome == "success")

"""Synthetic caller records only: no process, field, target or transit."""

import json
from pathlib import Path

import pytest

from tnfr.research import formed_two_port_launch as owner
from tnfr.research.artifact_io import file_receipt
from tnfr.utils.io import json_loads


@pytest.fixture(autouse=True)
def no_process_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Caller controls may not launch a real process or reference")

    monkeypatch.setattr(owner, "run_bounded_worker", forbidden)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    protocol, script, directory = (
        root / "protocol.json",
        root / "worker.py",
        root / "attempt",
    )
    _json(protocol, {"synthetic": "never an execution protocol"})
    script.write_text("Inert text; never imported or executed", encoding="utf-8")
    case = {
        "root": root,
        "protocol": protocol,
        "script": script,
        "directory": directory,
        "calls": 0,
        "alter": lambda *args: None,
    }

    def supervisor(actual_script, **options):
        case["calls"] += 1
        assert actual_script == script and options["cwd"] == root
        assert options["attempt_dir"] == directory
        assert options["wall_seconds"] == 3600
        assert options["process_memory_bytes"] == 1 << 30
        assert options["log_limit_bytes"] == 1 << 20
        assert options["arguments"] == (
            "--root",
            str(root),
            "--protocol",
            str(protocol),
            "--output",
            str(directory),
        )
        directory.mkdir()
        _json(directory / "attempt.json", {"synthetic": True})
        (directory / "worker.log").write_bytes(b"synthetic metadata\n")
        _json(directory / "steps/4095.json", {"synthetic": "not a certificate"})
        protocol_receipt = file_receipt(protocol, label="protocol.json")
        worker = {
            "status": "certified_capture",
            "completed_steps": 4096,
            "conditional_on_successful_post_write_supervisor_return": True,
            "endpoint": {
                "certified": True,
                "reasons": [],
                "conditional_on_acute_reference_prefix": True,
            },
            "source": {
                "protocol": protocol_receipt,
                "archive": {"path": "source.zip", "bytes": 0, "sha256": "0" * 64},
            },
            "last_complete_step": file_receipt(
                directory / "steps/4095.json", label="steps/4095.json"
            ),
        }
        _json(directory / "worker-result.json", worker)
        snapshot = {
            "record_stage": "pre_write_snapshot_not_final_attempt_verdict",
            "passed": True,
            "status": "success",
            "exit_code": 0,
            "job_assigned_before_resume": True,
            "resumed": True,
            "cleanup_complete": True,
            "worker_stopped_observed": True,
            "terminal_write_deadline_verified": False,
            "total_budget_exhausted": False,
            "watchdog_error": None,
            "error": None,
            "wall_seconds_limit": 3600,
            "process_memory_limit_bytes": 1 << 30,
            "closing_reserve_seconds": 10.0,
            "worker_seconds_budget": 3590.0,
            "supervisor_elapsed_seconds": 1.0,
            "attempt": file_receipt(directory / "attempt.json", label="attempt.json"),
            "log": file_receipt(directory / "worker.log", label="worker.log"),
        }
        returned = dict(
            snapshot,
            record_stage="post_write_result_required_for_attempt_admission",
            terminal_write_deadline_verified=True,
            supervisor_elapsed_seconds=1.1,
        )
        _json(directory / "execution.json", snapshot)
        case["alter"](worker, snapshot, returned)
        return returned

    monkeypatch.setattr(owner, "run_bounded_worker", supervisor)
    case["launch"] = lambda: owner.launch_formed_two_port(
        root, protocol, script, directory
    )
    return case


def test_post_write_return_is_preserved_but_cannot_admit_capture(synthetic):
    result = synthetic["launch"]()
    assert synthetic["calls"] == 1
    assert result["execution_passed"] and result["caller_write_within_total_budget"]
    assert result["record_stage"] == "post_caller_write_result"
    assert result["reference_success_admitted"] is False
    assert result["documentary_audit_required"] is True
    assert result["status"] == "execution_completed_pending_documentary_audit"
    assert set(result["receipts"]) == {
        "attempt",
        "execution_snapshot",
        "log",
        "worker_result",
    }
    path = synthetic["directory"] / "attempt-verdict.json"
    snapshot = json_loads(path.read_bytes())
    assert snapshot["record_stage"] == "pre_caller_write_snapshot_not_final_verdict"
    assert snapshot["caller_write_within_total_budget"] is False
    assert snapshot["supervisor_return"] == result["supervisor_return"]
    assert result["caller_verdict"] == file_receipt(path, label="attempt-verdict.json")
    assert path.stat().st_size <= 32 << 10
    assert len(owner.caller_return_bytes(result)) <= 31 << 10
    # No full prefix exists in this fixture: success here is deliberately only
    # execution-metadata eligibility, never documentary or mathematical capture.
    assert len(list((synthetic["directory"] / "steps").iterdir())) == 1


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "partial",
        "failed",
        "boolean_steps",
        "wrong_protocol",
        "missing_last",
        "truncated_last",
        "failed_parent",
        "changed_snapshot",
    ],
)
def test_missing_failed_partial_or_unassociated_evidence_is_unavailable(
    synthetic, kind
):
    directory = synthetic["directory"]

    def alter(worker, snapshot, returned):
        if kind == "missing":
            (directory / "worker-result.json").unlink()
        elif kind == "partial":
            (directory / "worker-result.json").write_bytes(b'{"unfinished"')
        elif kind in ("failed", "boolean_steps", "wrong_protocol"):
            if kind == "failed":
                worker["status"] = "unavailable"
            elif kind == "boolean_steps":
                worker["completed_steps"] = True
            else:
                worker["source"]["protocol"]["sha256"] = "f" * 64
            _json(directory / "worker-result.json", worker)
        elif kind == "missing_last":
            (directory / "steps/4095.json").unlink()
        elif kind == "truncated_last":
            (directory / "steps/4095.json").write_bytes(b"{")
        elif kind == "failed_parent":
            returned.update(
                passed=False, status="timeout_during_close", total_budget_exhausted=True
            )
        else:
            snapshot["error"] = "altered after return"
            _json(directory / "execution.json", snapshot)

    synthetic["alter"] = alter
    result = synthetic["launch"]()
    assert synthetic["calls"] == 1
    assert (
        result["execution_passed"] is False
        and result["reference_success_admitted"] is False
    )
    assert result["status"] == "unavailable" and result["errors"]
    assert (directory / "attempt-verdict.json").is_file()
    if kind == "partial":
        assert (directory / "worker-result.json").read_bytes() == b'{"unfinished"'
        assert result["receipts"]["worker_result"]["bytes"] == len(b'{"unfinished"')


def test_caller_write_overrun_revokes_snapshot_without_mutating_it(
    synthetic, monkeypatch
):
    original, monotonic = owner.safe_write, owner.time.monotonic
    offset = 0

    def clock():
        return monotonic() + offset

    def delayed_write(path, callback, **kwargs):
        nonlocal offset
        original(path, callback, **kwargs)
        offset = 3601

    monkeypatch.setattr(owner.time, "monotonic", clock)
    monkeypatch.setattr(owner, "safe_write", delayed_write)
    result = synthetic["launch"]()
    snapshot = json_loads(
        (synthetic["directory"] / "attempt-verdict.json").read_bytes()
    )
    assert snapshot["execution_passed"] is True
    assert result["execution_passed"] is False
    assert result["caller_write_within_total_budget"] is False
    assert result["status"] == "caller_close_unavailable"
    assert snapshot["record_stage"] == "pre_caller_write_snapshot_not_final_verdict"


def test_attempt_cannot_be_reused_even_when_caller_was_unavailable(synthetic):
    result = synthetic["launch"]()
    original = (synthetic["directory"] / "attempt-verdict.json").read_bytes()
    with pytest.raises(FileExistsError):
        synthetic["launch"]()
    assert synthetic["calls"] == 1
    assert (synthetic["directory"] / "attempt-verdict.json").read_bytes() == original
    assert result["reference_success_admitted"] is False


def test_oversized_verdict_fails_before_open_preserving_worker_evidence(synthetic):
    def alter(worker, snapshot, returned):
        snapshot["extra"] = returned["extra"] = "x" * (40 << 10)
        _json(synthetic["directory"] / "execution.json", snapshot)

    synthetic["alter"] = alter
    with pytest.raises(ValueError, match="32-KiB"):
        synthetic["launch"]()
    assert not (synthetic["directory"] / "attempt-verdict.json").exists()
    assert (synthetic["directory"] / "worker-result.json").is_file()
    assert synthetic["calls"] == 1


def test_oversized_return_is_rejected_before_wrapper_output():
    with pytest.raises(ValueError, match="31-KiB"):
        owner.caller_return_bytes({"oversized": "x" * (32 << 10)})


def test_interrupted_caller_write_stays_partial_and_prevents_retry(
    synthetic, monkeypatch
):
    def partial(path, callback, **kwargs):
        with Path(path).open("xb") as stream:
            stream.write(b'{"')
        raise OSError("synthetic interrupted caller write")

    monkeypatch.setattr(owner, "safe_write", partial)
    with pytest.raises(OSError, match="interrupted caller write"):
        synthetic["launch"]()
    assert (synthetic["directory"] / "attempt-verdict.json").read_bytes() == b'{"'
    with pytest.raises(FileExistsError):
        synthetic["launch"]()
    assert synthetic["calls"] == 1

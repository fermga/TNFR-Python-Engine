"""Fixed formed-pair launcher and bounded post-write caller evidence.

No mathematical producer is imported here. Source/target admission and all
numerical work remain inside the unchanged Windows supervisor's one worker.
The caller only inspects bounded terminal metadata. Full prefix association is
a later read-only documentary audit, never an extension of numerical time.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from ..utils.io import json_dumps, json_loads, safe_write
from .artifact_io import (
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
)
from .bounded_worker import run_bounded_worker

__all__ = ["launch_formed_two_port", "caller_return_bytes"]

SCHEMA = "tnfr.formed-two-port-caller.v1"
CALLER_VERDICT_BYTES = 32 * 1024
CALLER_RETURN_BYTES = 31 * 1024  # Leave 1 KiB for the external exit-code envelope.
_WALL = 3600
_MEMORY = 1 << 30
_LOG = 1 << 20
_SUPERVISOR_REVOCABLE = {
    "record_stage",
    "passed",
    "status",
    "supervisor_elapsed_seconds",
    "terminal_write_deadline_verified",
    "total_budget_exhausted",
}


def _json_bytes(value):
    return (
        json_dumps(encode_exact_tree(value), indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def caller_return_bytes(result: dict) -> bytes:
    """Prepare the exact wrapper stdout, reserving its external envelope space.

    The recorder must retain the decoded result and actual process exit code in
    one external file <=32 KiB. Together with the <=32-KiB in-attempt snapshot,
    these consume the runner's 64-KiB caller reserve from the same auxiliary cap.
    The outer persistence is documentary; it adds no scientific execution time.
    """
    data = _json_bytes(result)
    if len(data) > CALLER_RETURN_BYTES:
        raise ValueError("caller return exceeds its 31-KiB output allowance")
    return data


def _receipt_shape(row, name, limit):
    if (
        not isinstance(row, dict)
        or set(row) != {"path", "bytes", "sha256"}
        or row["path"] != name
        or type(row["bytes"]) is not int
        or not 0 <= row["bytes"] <= limit
        or not isinstance(row["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
    ):
        raise ValueError(f"invalid retained receipt: {name}")


def _read_json(path, limit, receipts, key):
    data = read_bytes_bounded(path, max_bytes=limit)
    receipts[key] = {
        "path": path.name,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    return json_loads(data)


def _successful_supervisor(result, snapshot, receipts):
    if not isinstance(result, dict) or not isinstance(snapshot, dict):
        raise ValueError("missing structured supervisor outcome")
    if (
        result.get("record_stage") != "post_write_result_required_for_attempt_admission"
        or result.get("passed") is not True
        or result.get("status") != "success"
        or type(result.get("exit_code")) is not int
        or result["exit_code"] != 0
        or result.get("job_assigned_before_resume") is not True
        or result.get("resumed") is not True
        or result.get("cleanup_complete") is not True
        or result.get("worker_stopped_observed") is not True
        or result.get("terminal_write_deadline_verified") is not True
        or result.get("total_budget_exhausted") is not False
        or result.get("watchdog_error") is not None
        or result.get("error") is not None
        or type(result.get("wall_seconds_limit")) is not int
        or result["wall_seconds_limit"] != _WALL
        or type(result.get("process_memory_limit_bytes")) is not int
        or result["process_memory_limit_bytes"] != _MEMORY
        or result.get("closing_reserve_seconds") != 10
        or result.get("worker_seconds_budget") != 3590
    ):
        raise ValueError("supervisor did not return a complete within-budget success")
    if (
        snapshot.get("record_stage") != "pre_write_snapshot_not_final_attempt_verdict"
        or snapshot.get("passed") is not True
        or snapshot.get("status") != "success"
        or snapshot.get("terminal_write_deadline_verified") is not False
        or snapshot.get("total_budget_exhausted") is not False
        or set(snapshot) != set(result)
        or {k: v for k, v in snapshot.items() if k not in _SUPERVISOR_REVOCABLE}
        != {k: v for k, v in result.items() if k not in _SUPERVISOR_REVOCABLE}
        or result.get("attempt") != receipts["attempt"]
        or result.get("log") != receipts["log"]
    ):
        raise ValueError("supervisor snapshot/returned bytes are not associated")


def _successful_worker(worker, directory, protocol_receipt):
    if (
        not isinstance(worker, dict)
        or worker.get("status") != "certified_capture"
        or type(worker.get("completed_steps")) is not int
        or worker["completed_steps"] != 4096
        or worker.get("conditional_on_successful_post_write_supervisor_return")
        is not True
        or not isinstance(worker.get("endpoint"), dict)
        or worker["endpoint"].get("certified") is not True
        or worker["endpoint"].get("reasons") != []
        or worker["endpoint"].get("conditional_on_acute_reference_prefix") is not True
        or not isinstance(worker.get("source"), dict)
    ):
        raise ValueError("worker has no complete conditional capture terminal record")
    source = worker["source"]
    _receipt_shape(source.get("protocol"), "protocol.json", 1 << 20)
    _receipt_shape(source.get("archive"), "source.zip", 64 << 20)
    if any(
        source["protocol"][key] != protocol_receipt[key] for key in ("bytes", "sha256")
    ):
        raise ValueError("terminal source differs from the invoked protocol bytes")
    last = worker.get("last_complete_step")
    _receipt_shape(last, "steps/4095.json", 256 << 20)
    path = directory / last["path"]
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve() != path
        or path.stat().st_size != last["bytes"]
    ):
        raise ValueError("the claimed final complete step is missing or truncated")
    # This is intentionally not a prefix audit or endpoint recomputation.
    # The documentary auditor must verify every receipt, source and relation.


def launch_formed_two_port(root, protocol_path, worker_script, attempt_dir) -> dict:
    """Launch the one fixed attempt and preserve its actual supervisor return.

    No retries or scientific options are exposed. The call starts its elapsed
    accounting immediately before invoking the supervisor. The intended final
    caller write and its receipt check must also finish within 3600 seconds;
    a late close returns failure, while preserving the immutable pre-write
    snapshot. Missing/failed/partial worker evidence is unavailable, not a
    refutation of capture. No reference success is admitted by this function.

    The frozen wrapper must emit ``caller_return_bytes(result)`` exactly and exit
    zero only if ``execution_passed is True``. Its external recorder binds that
    post-caller-write outcome to the real process exit code. The later bounded
    documentary auditor consumes both records and all 4096 retained steps.
    No record authenticates its own future writing or its review chronology.
    """
    root, protocol, script, directory = (
        Path(path).resolve()
        for path in (root, protocol_path, worker_script, attempt_dir)
    )
    if not root.is_dir() or not protocol.is_file() or not script.is_file():
        raise ValueError("selected root, protocol and frozen worker must exist")
    if not all(path.is_relative_to(root) for path in (protocol, script, directory)):
        raise ValueError(
            "formed-pair attempt paths must remain in the selected repository"
        )
    if directory.exists():
        raise FileExistsError("formed-pair attempt already exists; no retry")
    arguments = (
        "--root",
        str(root),
        "--protocol",
        str(protocol),
        "--output",
        str(directory),
    )
    started = time.monotonic()
    errors = []
    supervisor = None
    try:
        supervisor = run_bounded_worker(
            script,
            arguments=arguments,
            cwd=root,
            attempt_dir=directory,
            wall_seconds=_WALL,
            process_memory_bytes=_MEMORY,
            log_limit_bytes=_LOG,
        )
    except FileExistsError:
        raise  # Never append a caller verdict to another or older attempt.
    except Exception as exc:
        if not directory.is_dir():
            raise
        errors.append(f"supervisor exception: {type(exc).__name__}: {exc}"[:512])

    receipts = dict.fromkeys(("attempt", "execution_snapshot", "log", "worker_result"))
    records = {}
    for key, name, limit in (
        ("attempt", "attempt.json", 512 << 10),
        ("execution_snapshot", "execution.json", 512 << 10),
        ("worker_result", "worker-result.json", 1 << 20),
    ):
        try:
            path = directory / name
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("redirected caller evidence")
            records[key] = _read_json(path, limit, receipts, key)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}"[:512])
    try:
        path = directory / "worker.log"
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("redirected caller log")
        receipts["log"] = file_receipt(path, label="worker.log", max_bytes=_LOG)
    except Exception as exc:
        errors.append(f"log: {type(exc).__name__}: {exc}"[:512])
    protocol_receipt = None
    try:
        protocol_receipt = file_receipt(
            protocol, label=str(protocol), max_bytes=1 << 20
        )
        _successful_supervisor(supervisor, records.get("execution_snapshot"), receipts)
        _successful_worker(records.get("worker_result"), directory, protocol_receipt)
    except Exception as exc:
        errors.append(f"terminal admission: {type(exc).__name__}: {exc}"[:512])
    elapsed = time.monotonic() - started
    if elapsed >= _WALL:
        errors.append("caller metadata preparation exhausted the total wall budget")
    worker = records.get("worker_result")
    snapshot = {
        "schema": SCHEMA,
        "record_stage": "pre_caller_write_snapshot_not_final_verdict",
        "attempt_directory": str(directory),
        "protocol_receipt": protocol_receipt,
        "execution_passed": not errors,
        "status": (
            "execution_completed_pending_documentary_audit"
            if not errors
            else "unavailable"
        ),
        "reference_success_admitted": False,
        "documentary_audit_required": True,
        "supervisor_return": supervisor,
        "receipts": receipts,
        "worker_status": worker.get("status") if isinstance(worker, dict) else None,
        "completed_steps": (
            worker.get("completed_steps") if isinstance(worker, dict) else None
        ),
        "caller_elapsed_seconds": elapsed,
        "caller_total_seconds_limit": _WALL,
        "caller_write_within_total_budget": False,
        "errors": errors,
    }
    data = _json_bytes(snapshot)
    if len(data) > CALLER_VERDICT_BYTES:
        raise ValueError("caller verdict exceeds its 32-KiB immutable allowance")
    path = directory / "attempt-verdict.json"
    safe_write(
        path,
        lambda stream: stream.write(data),
        mode="xb",
        atomic=False,
        sync=True,
        base_dir=directory,
    )
    receipt = file_receipt(
        path, label="attempt-verdict.json", max_bytes=CALLER_VERDICT_BYTES
    )
    if receipt["bytes"] != len(data) or receipt["sha256"] != sha256_bytes(data):
        raise ValueError("persisted caller verdict differs from prepared bytes")
    result = dict(snapshot)
    result["record_stage"] = "post_caller_write_result"
    result["caller_verdict"] = receipt
    result["caller_elapsed_seconds"] = time.monotonic() - started
    result["caller_write_within_total_budget"] = (
        result["caller_elapsed_seconds"] < _WALL
    )
    if not result["caller_write_within_total_budget"]:
        result["execution_passed"] = False
        result["status"] = "caller_close_unavailable"
    caller_return_bytes(result)  # Fail before the wrapper emits oversized metadata.
    return result

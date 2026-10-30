"""One bounded documentary observation of the fixed wrapper, never a retry.

Killing the wrapper closes the supervisor's kill-on-close Job. Digests of
partial streams describe observed prefixes only, never unseen output.
"""

from pathlib import Path
import hashlib
import math
import subprocess
import sys
import threading
import time

from tnfr.research.artifact_io import file_receipt, read_bytes_bounded, sha256_bytes
from tnfr.research.formed_two_port_run import _json_bytes
from tnfr.utils.io import json_loads, safe_write

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LIMIT = 32 * 1024
STDOUT_LIMIT = 31 * 1024
STDERR_LIMIT = 1024
WRAPPER_SECONDS = 3630


def capture_wrapper(command, root):
    """Fixed bounded pipes; cap+1 observed bytes per pipe, no numerical work.

    Two daemon readers prevent pipe deadlock. Overflow, stderr or read failure
    immediately requests wrapper termination. Cleanup waits consume at most ten
    seconds of the remaining outer allowance. An OS stall leaves unavailable
    evidence, never permission for further computation or another attempt.
    """
    started = time.monotonic()
    deadline = started + WRAPPER_SECONDS
    errors, lock, abort = [], threading.Lock(), threading.Event()
    process = None
    timed_out = False
    rows = {
        name: {
            "data": bytearray(),
            "bytes": 0,
            "digest": hashlib.sha256(),
            "complete": False,
            "done": threading.Event(),
        }
        for name in ("stdout", "stderr")
    }
    threads = []

    def fail(reason):
        with lock:
            if len(errors) < 4:
                errors.append(reason[:512])
        abort.set()
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass  # It may have exited; wait/return code determines that.

    def read_pipe(name, stream, limit):
        row = rows[name]
        try:
            while not abort.is_set():
                count = min(4096, limit - row["bytes"] + 1)
                chunk = stream.read(count)
                if type(chunk) is not bytes:
                    raise TypeError("pipe returned non-byte data")
                if not chunk:
                    with lock:
                        row["complete"] = True
                    break
                if len(chunk) > count:
                    raise ValueError("pipe exceeded requested bounded read")
                with lock:
                    row["bytes"] += len(chunk)
                    row["digest"].update(chunk)
                    row["data"].extend(chunk[: max(0, limit - len(row["data"]))])
                if name == "stderr":
                    fail("unexpected wrapper stderr")
                    break
                if row["bytes"] > limit:
                    fail("wrapper stdout exceeds 31-KiB contract")
                    break
        except Exception as exc:
            fail(f"{name} read failure: {type(exc).__name__}: {exc}")
        finally:
            try:
                stream.close()
            except OSError:
                pass
            row["done"].set()

    try:
        process = subprocess.Popen(
            command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        for name, limit in (("stdout", STDOUT_LIMIT), ("stderr", STDERR_LIMIT)):
            thread = threading.Thread(
                target=read_pipe,
                args=(name, getattr(process, name), limit),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        while not abort.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                fail("outer wrapper deadline exhausted")
                break
            try:
                process.wait(timeout=min(0.05, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except Exception as exc:
        fail(f"wrapper launch/wait failure: {type(exc).__name__}: {exc}")
    finally:
        cleanup_deadline = min(deadline, time.monotonic() + 10)
        if process is not None:
            if abort.is_set():
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=max(0, cleanup_deadline - time.monotonic()))
            except Exception as exc:
                fail(f"wrapper termination not observed: {type(exc).__name__}: {exc}")
        for thread in threads:
            thread.join(timeout=max(0, cleanup_deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            fail("documentary pipe cleanup incomplete")
    with lock:
        captured = {
            "wrapper_exit_code": None if process is None else process.returncode,
            "wrapper_timed_out": timed_out,
            "wrapper_wall_seconds": time.monotonic() - started,
            "capture_errors": list(errors),
        }
        for name, row in rows.items():
            captured[name] = bytes(row["data"])
            captured[name + "_bytes"] = row["bytes"]
            captured[name + "_sha256"] = row["digest"].hexdigest()
            captured[name + "_complete"] = row["complete"] and row["done"].is_set()
            captured[name + "_hash_scope"] = (
                "complete stream"
                if captured[name + "_complete"]
                else "observed prefix only"
            )
    return captured


def envelope_bytes(captured):
    """One <=32-KiB external envelope and one acceptance predicate."""
    stdout, stderr = captured["stdout"], captured["stderr"]
    result, error = None, None
    try:
        if (
            captured["capture_errors"]
            or not captured["stdout_complete"]
            or not captured["stderr_complete"]
        ):
            raise ValueError("incomplete or failed wrapper capture")
        if len(stdout) > STDOUT_LIMIT or stderr:
            raise ValueError("wrapper output differs from its bounded contract")
        result = json_loads(stdout)
        if (
            not isinstance(result, dict)
            or result.get("schema") != "tnfr.formed-two-port-caller.v1"
            or result.get("record_stage") != "post_caller_write_result"
            or type(result.get("execution_passed")) is not bool
            or type(result.get("caller_write_within_total_budget")) is not bool
            or type(result.get("caller_elapsed_seconds")) not in (int, float)
            or not math.isfinite(result["caller_elapsed_seconds"])
            or result["caller_elapsed_seconds"] < 0
        ):
            raise ValueError("missing well-formed post-caller-write return")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:512]
    envelope = {k: v for k, v in captured.items() if k not in ("stdout", "stderr")}
    envelope.update(launcher_return=result, output_error=error)
    if error:
        envelope["stdout_prefix"] = stdout[:1024].decode("utf-8", errors="replace")
        envelope["stderr_prefix"] = stderr[:1024].decode("utf-8", errors="replace")
    data = _json_bytes(envelope)
    if len(data) > LIMIT:
        envelope.pop("stdout_prefix", None)
        envelope.pop("stderr_prefix", None)
        envelope.update(
            launcher_return=None,
            output_error="External return envelope exceeds 32-KiB reserve",
        )
        data = _json_bytes(envelope)
    if len(data) > LIMIT:
        raise ValueError("bounded diagnostic envelope unexpectedly exceeds reserve")
    retained = envelope["launcher_return"]
    accepted = (
        type(envelope["wrapper_exit_code"]) is int
        and envelope["wrapper_exit_code"] == 0
        and envelope["wrapper_timed_out"] is False
        and not envelope["output_error"]
        and not envelope["capture_errors"]
        and isinstance(retained, dict)
        and retained.get("execution_passed") is True
        and retained.get("caller_write_within_total_budget") is True
        and retained.get("caller_elapsed_seconds", 3600) < 3600
    )
    return data, accepted


def main():
    target = OUT / "caller-result.json"
    if target.exists() or (OUT / "attempt").exists():
        raise FileExistsError("No retry or output replacement")
    protocol = json_loads(
        read_bytes_bounded(OUT / "producing-protocol.json", max_bytes=1 << 20)
    )
    for row in [
        *protocol["sources"],
        protocol["archive"],
        *protocol["reviews"],
        protocol["controls"],
    ]:
        if (
            file_receipt(ROOT / row["path"], label=row["path"], max_bytes=row["bytes"])
            != row
        ):
            raise ValueError(f"producing bytes changed: {row['path']}")
    captured = capture_wrapper(
        [sys.executable, "-E", str(OUT / "launch_once.py")], ROOT
    )
    data, accepted = envelope_bytes(captured)
    safe_write(
        target,
        lambda stream: stream.write(data),
        mode="xb",
        atomic=False,
        sync=True,
        base_dir=OUT,
    )
    if file_receipt(target, label=target.name, max_bytes=LIMIT)[
        "sha256"
    ] != sha256_bytes(data):
        raise ValueError("retained external envelope differs from prepared bytes")
    print(
        {
            "wrapper_exit_code": captured["wrapper_exit_code"],
            "execution_passed": accepted,
            "caller_result": str(target),
            "documentary_envelope_bytes": len(data),
        }
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())

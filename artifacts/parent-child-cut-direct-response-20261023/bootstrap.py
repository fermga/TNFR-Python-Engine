"""Wait for actual-interpreter/job admission before importing the target.

Only the standard library is used before acknowledgement. These bounded JSON
records are process-control messages, not TNFR scientific evidence readers.
"""

import json
import math
import os
from pathlib import Path
import sys
from time import monotonic, sleep

MAX_CONTROL_BYTES = 65536


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate process-control key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite process-control number: " + value)


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite process-control number")
    return result


def read_control(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CONTROL_BYTES + 1)
    if len(raw) > MAX_CONTROL_BYTES:
        raise ValueError("process-control record exceeds its byte budget")
    return json.loads(
        raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float
    )


def write_control(path, value):
    path = Path(path)
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    if len(raw) > MAX_CONTROL_BYTES:
        raise ValueError("process-control record exceeds its byte budget")
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # Hard-link publication is atomic and refuses an existing final path.
    os.link(pending, path)
    pending.unlink()


def same_path(left, right):
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(
        os.path.realpath(right)
    )


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--request":
        raise ValueError("one explicit bootstrap request is required")
    request_path = Path(sys.argv[2]).resolve()
    request = read_control(request_path)
    if request.get("schema") != "tnfr.bootstrap-request.v1":
        raise ValueError("unsupported bootstrap request")
    nonce = request["nonce"]
    if (
        type(nonce) is not str
        or len(nonce) != 64
        or any(c not in "0123456789abcdef" for c in nonce)
    ):
        raise ValueError("invalid bootstrap nonce")
    timeout = request["handshake_timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive finite handshake timeout required")
    paths = {
        name: Path(request[name]).resolve()
        for name in ("ready_path", "ack_path", "exit_path")
    }
    if any(path.parent != request_path.parent for path in paths.values()):
        raise ValueError("bootstrap control paths must share their request folder")
    if not same_path(sys.prefix, request["expected_prefix"]) or not same_path(
        sys.executable, request["expected_executable"]
    ):
        raise ValueError("bootstrap interpreter environment differs")
    if not same_path(sys._base_executable, request["expected_base_executable"]):
        raise ValueError("bootstrap base interpreter differs")
    if any(name == "tnfr" or name.startswith("tnfr.") for name in sys.modules):
        raise ValueError("TNFR was imported before bootstrap acknowledgement")
    worker = Path(request["worker"]).resolve()
    arguments = request["worker_args"]
    if type(arguments) is not list or any(
        type(value) is not str for value in arguments
    ):
        raise ValueError("worker arguments must be ordinary strings")
    ready = {
        "schema": "tnfr.bootstrap-ready.v1",
        "nonce": nonce,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "executable": sys.executable,
        "base_executable": sys._base_executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "version": sys.version,
        "tnfr_imported": False,
    }
    write_control(paths["ready_path"], ready)
    started = monotonic()
    while not paths["ack_path"].exists():
        if monotonic() - started >= timeout:
            raise TimeoutError("bootstrap acknowledgement timed out")
        sleep(0.025)
    ack = read_control(paths["ack_path"])
    if (
        ack.get("schema") != "tnfr.bootstrap-ack.v1"
        or ack.get("nonce") != nonce
        or ack.get("worker_pid") != os.getpid()
        or ack.get("job_verified") is not True
        or ack.get("supervisor_pid") != request["supervisor_pid"]
    ):
        raise ValueError("bootstrap acknowledgement association differs")
    for name, expected in (
        ("worker", worker),
        ("executable", sys.executable),
        ("prefix", sys.prefix),
    ):
        if not same_path(ack[name], expected):
            raise ValueError("bootstrap acknowledgement changed " + name)

    # Import this stdlib-only local owner after ACK and independently verify
    # current-process membership/creation identity before target imports.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tnfr_admitted_process_supervision", Path(__file__).with_name("supervision.py")
    )
    supervision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervision)
    marker = {
        "schema": "tnfr.bootstrap-admitted.v1",
        "nonce": nonce,
        "worker_pid": os.getpid(),
        "supervisor_pid": request["supervisor_pid"],
        "acknowledgement_path": str(paths["ack_path"]),
        "request_path": str(request_path),
        "worker": str(worker),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_executable": sys._base_executable,
    }
    supervision.validate_bootstrap_admission(
        marker,
        expected_prefix=request["expected_prefix"],
        expected_executable=request["expected_executable"],
    )
    os.environ["TNFR_BOOTSTRAP_NONCE"] = nonce
    os.environ["TNFR_BOOTSTRAP_RECEIPT"] = str(paths["ack_path"])
    sys.argv = [str(worker), *arguments]
    # All target imports are downstream of acknowledged OS job admission.
    import runpy

    code, error = 0, None
    try:
        runpy.run_path(
            str(worker),
            run_name="__main__",
            init_globals={"__tnfr_supervision__": marker},
        )
    except SystemExit as outcome:
        code = (
            outcome.code
            if type(outcome.code) is int
            else (0 if outcome.code is None else 1)
        )
        error = None if code == 0 else str(outcome.code)[:1000]
        if code:
            raise
    except BaseException as outcome:
        code, error = 1, f"{type(outcome).__name__}: {outcome}"[:1000]
        raise
    finally:
        write_control(
            paths["exit_path"],
            {
                "schema": "tnfr.bootstrap-exit.v1",
                "nonce": nonce,
                "worker_pid": os.getpid(),
                "target_authorized": True,
                "worker": str(worker),
                "exit_code": code,
                "error": error,
            },
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())

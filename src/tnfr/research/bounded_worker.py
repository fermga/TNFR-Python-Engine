"""One Windows Python worker under an OS-enforced committed-memory limit.

No mathematical owner is imported here. Source/target admission belongs in the
worker, after job assignment and inside the deadline. This is a cooperative
research runner, not a security sandbox; artifact quotas belong to its writer.

Windows 10+ JOB_LIST associates the process during creation, before its initial
thread runs. CREATE_SUSPENDED additionally permits verification before resume:
https://devblogs.microsoft.com/oldnewthing/20230209-00/?p=107812
Memory means committed virtual memory, not RSS or Python allocations:
https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information
The CPython venv redirector's environment/argv convention is reproduced to
avoid a second launcher process under the one-process limit:
https://github.com/python/cpython/blob/v3.13.6/PC/venvlauncher.c
"""

from __future__ import annotations

import ctypes as C
import math
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes as W
from pathlib import Path

__all__ = ["run_bounded_worker", "require_current_job_limits"]


class _BasicLimits(C.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", C.c_longlong),
        ("PerJobUserTimeLimit", C.c_longlong),
        ("LimitFlags", W.DWORD),
        ("MinimumWorkingSetSize", C.c_size_t),
        ("MaximumWorkingSetSize", C.c_size_t),
        ("ActiveProcessLimit", W.DWORD),
        ("Affinity", C.c_size_t),
        ("PriorityClass", W.DWORD),
        ("SchedulingClass", W.DWORD),
    ]


class _IOCounters(C.Structure):
    _fields_ = [
        (name, C.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimits(C.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IOCounters),
        ("ProcessMemoryLimit", C.c_size_t),
        ("JobMemoryLimit", C.c_size_t),
        ("PeakProcessMemoryUsed", C.c_size_t),
        ("PeakJobMemoryUsed", C.c_size_t),
    ]


class _Security(C.Structure):
    _fields_ = [
        ("nLength", W.DWORD),
        ("lpSecurityDescriptor", C.c_void_p),
        ("bInheritHandle", W.BOOL),
    ]


class _Startup(C.Structure):
    _fields_ = [
        ("cb", W.DWORD),
        ("lpReserved", W.LPWSTR),
        ("lpDesktop", W.LPWSTR),
        ("lpTitle", W.LPWSTR),
        *[
            (name, W.DWORD)
            for name in (
                "dwX",
                "dwY",
                "dwXSize",
                "dwYSize",
                "dwXCountChars",
                "dwYCountChars",
                "dwFillAttribute",
                "dwFlags",
            )
        ],
        ("wShowWindow", W.WORD),
        ("cbReserved2", W.WORD),
        ("lpReserved2", C.POINTER(C.c_ubyte)),
        ("hStdInput", W.HANDLE),
        ("hStdOutput", W.HANDLE),
        ("hStdError", W.HANDLE),
    ]


class _StartupEx(C.Structure):
    _fields_ = [("StartupInfo", _Startup), ("lpAttributeList", C.c_void_p)]


class _Process(C.Structure):
    _fields_ = [
        ("hProcess", W.HANDLE),
        ("hThread", W.HANDLE),
        ("dwProcessId", W.DWORD),
        ("dwThreadId", W.DWORD),
    ]


# ACTIVE_PROCESS, PROCESS_MEMORY, JOB_MEMORY, KILL_ON_JOB_CLOSE. No breakaway.
_LIMIT_FLAGS = 0x8 | 0x100 | 0x200 | 0x2000
_BREAKAWAY_FLAGS = 0x800 | 0x1000
_INVALID_HANDLE = C.c_void_p(-1).value
_MAX_LOG_BYTES = 1 << 20
_MAX_RECORD_BYTES = 1 << 19  # Two records, hence at most 1 MiB in total.


def _winapi():
    """Bind pointer-sized Win32 signatures; fail before launching if absent."""
    if os.name != "nt":
        raise OSError("Windows Job Objects are required; no soft-limit fallback")
    api = C.WinDLL("kernel32", use_last_error=True)
    ptr = C.c_void_p
    definitions = {
        "CreateJobObjectW": ([ptr, W.LPCWSTR], W.HANDLE),
        "SetInformationJobObject": ([W.HANDLE, C.c_int, ptr, W.DWORD], W.BOOL),
        "QueryInformationJobObject": (
            [W.HANDLE, C.c_int, ptr, W.DWORD, C.POINTER(W.DWORD)],
            W.BOOL,
        ),
        "CreatePipe": (
            [C.POINTER(W.HANDLE), C.POINTER(W.HANDLE), ptr, W.DWORD],
            W.BOOL,
        ),
        "SetHandleInformation": ([W.HANDLE, W.DWORD, W.DWORD], W.BOOL),
        "CreateFileW": (
            [W.LPCWSTR, W.DWORD, W.DWORD, ptr, W.DWORD, W.DWORD, W.HANDLE],
            W.HANDLE,
        ),
        "InitializeProcThreadAttributeList": (
            [ptr, W.DWORD, W.DWORD, C.POINTER(C.c_size_t)],
            W.BOOL,
        ),
        "UpdateProcThreadAttribute": (
            [ptr, W.DWORD, C.c_size_t, ptr, C.c_size_t, ptr, ptr],
            W.BOOL,
        ),
        "DeleteProcThreadAttributeList": ([ptr], None),
        "CreateProcessW": (
            [
                W.LPCWSTR,
                W.LPWSTR,
                ptr,
                ptr,
                W.BOOL,
                W.DWORD,
                ptr,
                W.LPCWSTR,
                ptr,
                C.POINTER(_Process),
            ],
            W.BOOL,
        ),
        "IsProcessInJob": ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
        "GetCurrentProcess": ([], W.HANDLE),
        "DuplicateHandle": (
            [
                W.HANDLE,
                W.HANDLE,
                W.HANDLE,
                C.POINTER(W.HANDLE),
                W.DWORD,
                W.BOOL,
                W.DWORD,
            ],
            W.BOOL,
        ),
        "ResumeThread": ([W.HANDLE], W.DWORD),
        "WaitForSingleObject": ([W.HANDLE, W.DWORD], W.DWORD),
        "GetExitCodeProcess": ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL),
        "PeekNamedPipe": (
            [W.HANDLE, ptr, W.DWORD, ptr, C.POINTER(W.DWORD), ptr],
            W.BOOL,
        ),
        "ReadFile": ([W.HANDLE, ptr, W.DWORD, C.POINTER(W.DWORD), ptr], W.BOOL),
        "TerminateJobObject": ([W.HANDLE, W.UINT], W.BOOL),
        "CloseHandle": ([W.HANDLE], W.BOOL),
    }
    for name, (arguments, result) in definitions.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result
    return api


def _require(ok, operation):
    if not ok:
        raise OSError(C.get_last_error(), operation)


class _DeadlineGuard:
    """Parent watchdog independent of stdout/file drainage, owning one handle."""

    def __init__(self, api, job, deadline):
        self.api, self.deadline = api, deadline
        self.cancel = threading.Event()
        self.fired = False
        self.error = None
        duplicate = W.HANDLE()
        current = api.GetCurrentProcess()
        _require(
            api.DuplicateHandle(current, job, current, C.byref(duplicate), 0, False, 2),
            "Duplicate bounded Job for watchdog",
        )
        self.job = duplicate.value
        self.thread = threading.Thread(target=self._watch, daemon=True)
        try:
            self.thread.start()
        except BaseException:
            api.CloseHandle(self.job)
            raise

    def _watch(self):
        try:
            if not self.cancel.wait(max(0, self.deadline - time.monotonic())):
                self.fired = True
                if not self.api.TerminateJobObject(self.job, 1):
                    self.error = f"TerminateJobObject: {C.get_last_error()}"
        finally:
            self.api.CloseHandle(self.job)

    def stop(self, total_deadline):
        self.cancel.set()
        self.thread.join(timeout=max(0, total_deadline - time.monotonic()))
        return not self.thread.is_alive()


def _required_limits(api, job, process_memory_bytes):
    actual = _ExtendedLimits()
    _require(
        api.QueryInformationJobObject(job, 9, C.byref(actual), C.sizeof(actual), None),
        "Query job limits",
    )
    basic = actual.BasicLimitInformation
    if (
        basic.LimitFlags & _LIMIT_FLAGS != _LIMIT_FLAGS
        or basic.LimitFlags & _BREAKAWAY_FLAGS
        or basic.ActiveProcessLimit != 1
        or actual.ProcessMemoryLimit != process_memory_bytes
        or actual.JobMemoryLimit != process_memory_bytes
    ):
        raise OSError("Windows job does not have the exact required hard limits")
    return actual


def require_current_job_limits(process_memory_bytes=1 << 30):
    """Fail closed unless the current process has the required immediate Job.

    Call before source/target admission in every worker entry point. Windows
    queries the immediate job for a NULL handle, including under nested jobs:
    https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject

    This verifies OS limits, not supervisor identity or an external wall clock.
    The reviewed supervisor still supplies the deadline; this is not a sandbox
    or authentication mechanism. A stronger ancestor limit may cause failure
    earlier, never relax the admitted cap. The pseudo process handle is not
    owned and must not be closed.
    """
    if type(process_memory_bytes) is not int or not 0 < process_memory_bytes <= 1 << 30:
        raise ValueError("process_memory_bytes must be a positive integer <= 1 GiB")
    api = _winapi()
    belongs = W.BOOL()
    _require(
        api.IsProcessInJob(api.GetCurrentProcess(), None, C.byref(belongs)),
        "IsProcessInJob for current process",
    )
    if not belongs.value:
        raise OSError("Current process is outside a bounded Windows Job")
    actual = _required_limits(api, None, process_memory_bytes)
    return {
        "current_process_in_job": True,
        "queried_job": "immediate",
        "limit_flags": actual.BasicLimitInformation.LimitFlags,
        "active_process_limit": actual.BasicLimitInformation.ActiveProcessLimit,
        "process_memory_limit_bytes": actual.ProcessMemoryLimit,
        "job_memory_limit_bytes": actual.JobMemoryLimit,
        "kill_on_close": True,
        "breakaway_allowed": False,
        "wall_deadline_verified": False,
    }


_BOOTSTRAP = """import os, sys
prefix, executable, script, *args = sys.argv[1:]
same = lambda a,b: os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
if not same(sys.prefix, prefix) or not same(sys.executable, executable):
    raise RuntimeError('The bounded worker did not retain the admitted interpreter environment')
sys.argv = [script, *args]
sys.path.insert(0, os.path.dirname(script))
import runpy
runpy.run_path(script, run_name='__main__')
"""


def _launch(api, job, script, arguments, cwd, process):
    """Create suspended; caller owns process handles even if verification fails."""
    handles = []
    attributes = None
    try:
        sa = _Security(C.sizeof(_Security), None, True)
        reader, writer = W.HANDLE(), W.HANDLE()
        _require(
            api.CreatePipe(C.byref(reader), C.byref(writer), C.byref(sa), 4096),
            "CreatePipe",
        )
        handles.extend([reader.value, writer.value])
        _require(api.SetHandleInformation(reader, 1, 0), "Protect pipe reader")
        null = api.CreateFileW("NUL", 0x80000000, 3, C.byref(sa), 3, 0x80, None)
        if null in (None, _INVALID_HANDLE):
            raise OSError(C.get_last_error(), "Open NUL")
        handles.append(null)
        size = C.c_size_t()
        api.InitializeProcThreadAttributeList(None, 2, 0, C.byref(size))
        if not size.value:
            raise OSError("Attribute list sizing unavailable")
        storage = C.create_string_buffer(size.value)
        _require(
            api.InitializeProcThreadAttributeList(storage, 2, 0, C.byref(size)),
            "InitializeProcThreadAttributeList",
        )
        attributes = storage
        inherited = (W.HANDLE * 2)(writer.value, null)
        jobs = (W.HANDLE * 1)(job)
        for key, value in ((0x20002, inherited), (0x2000D, jobs)):
            _require(
                api.UpdateProcThreadAttribute(
                    storage, 0, key, value, C.sizeof(value), None, None
                ),
                "Set handle/job attribute",
            )
        startup = _StartupEx()
        startup.StartupInfo.cb = C.sizeof(startup)
        startup.StartupInfo.dwFlags = 0x100  # STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = null
        startup.StartupInfo.hStdOutput = writer.value
        startup.StartupInfo.hStdError = writer.value
        startup.lpAttributeList = C.cast(storage, C.c_void_p)
        # Directly launch the actual CPython image, not the venv redirector.
        image = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        command = [
            sys.executable,
            "-E",
            "-u",
            "-c",
            _BOOTSTRAP,
            sys.prefix,
            sys.executable,
            str(script),
            *arguments,
        ]
        environment = dict(os.environ)
        environment.pop("PYTHONEXECUTABLE", None)
        environment["__PYVENV_LAUNCHER__"] = sys.executable
        block = C.create_unicode_buffer(
            "\0".join(
                f"{key}={value}"
                for key, value in sorted(
                    environment.items(), key=lambda item: item[0].upper()
                )
            )
            + "\0\0"
        )
        # Suspended, Unicode environment, extended startup, no visible window.
        flags = 0x4 | 0x400 | 0x80000 | 0x8000000
        _require(
            api.CreateProcessW(
                str(image),
                C.create_unicode_buffer(subprocess.list2cmdline(command)),
                None,
                None,
                True,
                flags,
                block,
                str(cwd),
                C.byref(startup),
                C.byref(process),
            ),
            "CreateProcessW with JOB_LIST",
        )
        belongs = W.BOOL()
        _require(
            api.IsProcessInJob(process.hProcess, job, C.byref(belongs)),
            "IsProcessInJob",
        )
        if not belongs.value:
            raise OSError("New suspended worker is not in the bounded job")
        result = (reader.value, str(image))
        handles.remove(reader.value)
        return result
    except BaseException:
        if process.hProcess:
            api.TerminateJobObject(job, 1)
            # Do not discard the only process handle before the caller has
            # observed termination under its remaining cleanup budget.
        raise
    finally:
        if attributes is not None:
            api.DeleteProcThreadAttributeList(attributes)
        for handle in handles:
            api.CloseHandle(handle)


def _drain(api, pipe, log, written, limit):
    """Read a bounded chunk without a blocking stdout reader or unbounded RAM."""
    available = W.DWORD()
    if not api.PeekNamedPipe(pipe, None, 0, None, C.byref(available), None):
        if C.get_last_error() == 109:  # ERROR_BROKEN_PIPE: all writers closed.
            return written, False, True
        raise OSError(C.get_last_error(), "PeekNamedPipe")
    count = min(available.value, 4096, limit - written + 1)
    if not count:
        return written, False, False
    buffer = C.create_string_buffer(count)
    received = W.DWORD()
    _require(api.ReadFile(pipe, buffer, count, C.byref(received), None), "ReadFile")
    data = buffer.raw[: received.value]
    retained = data[: max(0, limit - written)]
    log.write(retained)
    log.flush()
    return written + len(retained), len(retained) != len(data), False


def _write_record(path, payload):
    """Admit the exact shared serialization size before its immutable write."""
    from ..utils.io import json_dumps
    from .artifact_io import encode_exact_tree, write_json_once

    text = json_dumps(encode_exact_tree(payload), indent=2, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > _MAX_RECORD_BYTES:
        raise ValueError("Supervisor metadata exceeds its 512-KiB per-record quota")
    write_json_once(path, payload)


def run_bounded_worker(
    script,
    *,
    arguments=(),
    cwd,
    attempt_dir,
    wall_seconds=3600,
    process_memory_bytes=1 << 30,
    log_limit_bytes=1 << 20,
):
    """Run one current-interpreter worker, or fail closed with immutable evidence.

    The monotonic total budget begins on entry. All source/target readmission and
    scientific imports must be inside ``script``. ``attempt_dir`` must be new;
    neither a partial attempt nor a prior result may be reused. The caller's
    artifact ledger must reserve 2 MiB: log <=1 MiB and exactly two supervisor
    JSON records, each <=512 KiB. Oversized metadata fails before its write.

    A Windows Job denies committed memory above the process/job cap, prevents
    another process, and kills the child if the supervisor loses its handle.
    Allocation denial may produce MemoryError or a normal application failure;
    this function never infers its cause just from an exit code. Unsupported
    OS/API/job nesting fails closed. No soft RSS polling fallback is used.

    Reserve min(10 seconds, one quarter of the total) for termination and
    supervisor evidence. An independent parent thread terminates the Job at
    the earlier worker deadline even if log drainage stalls. No wait extends
    beyond the remaining total budget. OS scheduling and blocked filesystem
    operations are not hard real-time guarantees: an overrun is a failure,
    never extra computation time. The worker's own writes are inside its
    earlier deadline. Terminal JSON is an immutable pre-write snapshot;
    the returned dictionary additionally checks time after that write and may
    report timeout_during_close without overwriting the snapshot.

    Returns an execution dictionary. Nonzero exit, timeout, log overflow or
    enforcement failure has ``passed=False``. Invalid parameters and existing
    attempt directories raise before launch. An interrupted or partial attempt
    cannot retry. Cleanup failures and total-budget exhaustion are explicit.
    """
    start = time.monotonic()
    if (
        type(wall_seconds) not in (int, float)
        or not math.isfinite(wall_seconds)
        or not 0 < wall_seconds <= 3600
    ):
        raise ValueError("wall_seconds must be finite and in (0, 3600]")
    for name, value, ceiling in (
        ("process_memory_bytes", process_memory_bytes, 1 << 30),
        ("log_limit_bytes", log_limit_bytes, _MAX_LOG_BYTES),
    ):
        if type(value) is not int or not 0 < value <= ceiling:
            raise ValueError(f"{name} must be a positive integer <= {ceiling}")
    if type(arguments) not in (tuple, list):
        raise TypeError("arguments must be a tuple/list of strings")
    if len(arguments) > 64:
        raise ValueError("At most 64 worker arguments are admitted")
    arguments = tuple(arguments)
    if any(type(value) is not str or "\0" in value for value in arguments):
        raise ValueError("arguments must contain NUL-free strings")
    script, cwd, output = map(lambda p: Path(p).resolve(), (script, cwd, attempt_dir))
    if not script.is_file() or not cwd.is_dir():
        raise ValueError("script and cwd must exist")
    output.mkdir()  # Exclusive reservation: even an incomplete attempt cannot retry.
    from .artifact_io import file_receipt

    total_deadline = start + wall_seconds
    closing_reserve = min(10.0, wall_seconds / 4)
    deadline = total_deadline - closing_reserve
    _write_record(
        output / "attempt.json",
        {
            "script": str(script),
            "arguments": arguments,
            "cwd": str(cwd),
            "interpreter": sys.executable,
            "prefix": sys.prefix,
            "wall_seconds": wall_seconds,
            "closing_reserve_seconds": closing_reserve,
            "worker_seconds_budget": wall_seconds - closing_reserve,
            "process_memory_bytes": process_memory_bytes,
            "memory_metric": "Windows committed virtual memory per process and job",
            "log_limit_bytes": log_limit_bytes,
            "one_worker": True,
        },
    )
    api = job = reader = process = guard = None
    status, error, code, image = "enforcement_unavailable", None, None, None
    assigned = resumed = False
    worker_stopped = True
    cleanup_complete = True
    written, peak = 0, None
    log_path = output / "worker.log"
    try:
        with log_path.open("xb") as log:
            api = _winapi()
            if sys.implementation.name != "cpython":
                raise OSError("Only the admitted CPython environment is supported")
            job = api.CreateJobObjectW(None, None)
            _require(job, "CreateJobObjectW")
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = _LIMIT_FLAGS
            limits.BasicLimitInformation.ActiveProcessLimit = 1
            limits.ProcessMemoryLimit = limits.JobMemoryLimit = process_memory_bytes
            _require(
                api.SetInformationJobObject(job, 9, C.byref(limits), C.sizeof(limits)),
                "Set job limits",
            )
            _required_limits(api, job, process_memory_bytes)
            if time.monotonic() >= deadline:
                status = "timeout"
            else:
                guard = _DeadlineGuard(api, job, deadline)
                process = _Process()
                reader, image = _launch(api, job, script, arguments, cwd, process)
                assigned = True
                worker_stopped = False
                if time.monotonic() >= deadline:
                    status = "timeout"
                else:
                    if api.ResumeThread(process.hThread) == 0xFFFFFFFF:
                        raise OSError(C.get_last_error(), "ResumeThread")
                    resumed, status = True, "running"
                while status == "running":
                    if time.monotonic() >= deadline:
                        status = "timeout"
                        break
                    written, overflow, _ = _drain(
                        api, reader, log, written, log_limit_bytes
                    )
                    if time.monotonic() >= deadline or guard.fired:
                        status = "timeout"
                        break
                    if overflow:
                        status = "log_limit"
                        break
                    waited = api.WaitForSingleObject(process.hProcess, 0)
                    if waited == 0:
                        # Drain the finite remainder; no descendant can retain stdout.
                        available = W.DWORD()
                        more = api.PeekNamedPipe(
                            reader, None, 0, None, C.byref(available), None
                        )
                        if not more and C.get_last_error() != 109:
                            raise OSError(
                                C.get_last_error(), "PeekNamedPipe after exit"
                            )
                        if more and available.value:
                            continue
                        exit_code = W.DWORD()
                        _require(
                            api.GetExitCodeProcess(
                                process.hProcess, C.byref(exit_code)
                            ),
                            "GetExitCodeProcess",
                        )
                        code = exit_code.value
                        worker_stopped = True
                        status = "success" if code == 0 else "worker_failed"
                    elif waited == 258:
                        time.sleep(min(0.01, max(0, deadline - time.monotonic())))
                    else:
                        raise OSError(C.get_last_error(), "WaitForSingleObject")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:2048]
        status = "enforcement_unavailable" if not resumed else "supervisor_failed"
    finally:
        if api is not None:
            if job:
                if (
                    process is not None
                    and process.hProcess
                    and status not in ("success", "worker_failed")
                ):
                    api.TerminateJobObject(job, 1)
                    milliseconds = max(
                        0, int((total_deadline - time.monotonic()) * 1000)
                    )
                    worker_stopped = (
                        api.WaitForSingleObject(process.hProcess, milliseconds) == 0
                    )
                if guard is not None:
                    cleanup_complete = guard.stop(total_deadline)
                actual = _ExtendedLimits()
                if api.QueryInformationJobObject(
                    job, 9, C.byref(actual), C.sizeof(actual), None
                ):
                    peak = actual.PeakProcessMemoryUsed
                api.CloseHandle(job)  # KILL_ON_JOB_CLOSE, also on exceptions.
            if reader:
                api.CloseHandle(reader)
            if process is not None:
                if process.hThread:
                    api.CloseHandle(process.hThread)
                if process.hProcess:
                    api.CloseHandle(process.hProcess)
    cleanup_complete = cleanup_complete and worker_stopped
    if guard is not None and guard.error is not None:
        status, error = "supervisor_failed", guard.error
    if not cleanup_complete and status == "success":
        status = "cleanup_incomplete"
    if time.monotonic() >= total_deadline:
        status = "timeout_during_close"
    result = {
        "record_stage": "pre_write_snapshot_not_final_attempt_verdict",
        "passed": status == "success",
        "status": status,
        "exit_code": code,
        "error": error,
        "job_assigned_before_resume": assigned,
        "resumed": resumed,
        "worker_pid": (
            None if process is None or not process.hProcess else process.dwProcessId
        ),
        "image": image,
        "wall_seconds_limit": wall_seconds,
        "closing_reserve_seconds": closing_reserve,
        "worker_seconds_budget": wall_seconds - closing_reserve,
        "watchdog_fired": guard is not None and guard.fired,
        "watchdog_error": None if guard is None else guard.error,
        "worker_stopped_observed": worker_stopped,
        "cleanup_complete": cleanup_complete,
        "total_budget_exhausted": time.monotonic() >= total_deadline,
        "terminal_write_deadline_verified": False,
        "supervisor_elapsed_seconds": time.monotonic() - start,
        "process_memory_limit_bytes": process_memory_bytes,
        "peak_process_commit_bytes": peak,
        "log_bytes": written,
        "attempt": file_receipt(output / "attempt.json", label="attempt.json"),
        "log": file_receipt(log_path, label="worker.log"),
    }
    # Receipts and serialization take time too. Never commit a success snapshot
    # when its pre-write check has already exhausted the total budget.
    if time.monotonic() >= total_deadline:
        result.update(
            passed=False, status="timeout_during_close", total_budget_exhausted=True
        )
    _write_record(output / "execution.json", result)
    result["record_stage"] = "post_write_result_required_for_attempt_admission"
    result["supervisor_elapsed_seconds"] = time.monotonic() - start
    result["terminal_write_deadline_verified"] = True
    result["total_budget_exhausted"] = time.monotonic() >= total_deadline
    if result["total_budget_exhausted"]:
        result.update(passed=False, status="timeout_during_close")
    return result

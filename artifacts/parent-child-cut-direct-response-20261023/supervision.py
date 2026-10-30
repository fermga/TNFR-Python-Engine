"""Windows job supervision of the actual acknowledged Python interpreter.

The venv redirector and actual worker are associated before ACK. A frozen,
stdlib-only bootstrap waits without spawning children; pre-ACK lineage scans
also attach an interpreter born before launcher job assignment. Descendants
created after ACK inherit a no-breakaway, kill-on-close job. RSS is sampled
across job members and the actual worker's OS peak is additionally checked;
this is not an operating-system hard RSS allocation limit.
"""

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
from threading import Event, Thread
from time import monotonic, sleep

MAX_CONTROL_BYTES = 65536
MAX_JOB_PROCESSES = 64
MAX_EXCLUDED_LINEAGE_DIAGNOSTICS = 16
CONTROL_NAMES = (
    "supervision-start.json",
    "bootstrap-request.json",
    "bootstrap-ready.json",
    "bootstrap-ack.json",
    "bootstrap-exit.json",
    "supervision-result.json",
    "worker.log",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "duplicate process-control key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite process-control number: " + value)


def _float(value):
    result = float(value)
    _require(math.isfinite(result), "nonfinite process-control number")
    return result


def read_control(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CONTROL_BYTES + 1)
    _require(
        len(raw) <= MAX_CONTROL_BYTES, "process-control record exceeds its byte budget"
    )
    return json.loads(
        raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float
    )


def write_control(path, value):
    path = Path(path)
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    _require(
        len(raw) <= MAX_CONTROL_BYTES, "process-control record exceeds its byte budget"
    )
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def _same_path(left, right):
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(
        os.path.realpath(right)
    )


def _verify_pre_ack_inventory(
    launcher_pid, worker_pid, member_ids, identities, console_image
):
    """Admit the real interpreter and owned native console helpers only."""
    members = set(member_ids)
    _require(
        len(members) == len(member_ids) and {launcher_pid, worker_pid} <= members,
        "pre-ACK job omits launcher or actual worker",
    )
    _require(launcher_pid in identities, "launcher handle identity is unavailable")
    for pid in members:
        _require(
            pid in identities and identities[pid].get("job_verified") is True,
            "pre-ACK job member lacks an admitted OS handle",
        )
        if pid not in {launcher_pid, worker_pid}:
            _require(
                _same_path(identities[pid]["image"], console_image),
                "unexpected pre-ACK helper executable",
            )
        current, visited = pid, set()
        while current != launcher_pid:
            _require(
                current not in visited and current in identities,
                "pre-ACK helper lineage is cyclic or incomplete",
            )
            visited.add(current)
            child = identities[current]
            parent = child["parent_pid"]
            _require(
                parent in identities
                and child["creation_time"] >= identities[parent]["creation_time"],
                "pre-ACK helper is outside the owned creation lineage",
            )
            current = parent


def _attach_owned_lineage(api, launcher_pid, handles, identities, admit, excluded):
    """Grow ownership through live, creation-checked immediate parent edges.

    A Toolhelp parent PID alone is not an ownership proof: an older process
    may retain the PID of a terminated parent whose number is now reused.
    Such a candidate and its raw subtree are never admitted or expanded.
    Opened identities with unresolved parent changes remain fail-closed.
    """
    _require(
        launcher_pid in handles and launcher_pid in identities,
        "retained launcher identity is required before lineage discovery",
    )
    parents = api.parents()
    frontier, visited = [launcher_pid], set()

    def exclude(pid, parent, reason, identity=None):
        identity = {} if identity is None else identity
        image = identity.get("image")
        row = {
            "pid": pid,
            "snapshot_parent_pid": parent["pid"],
            "actual_parent_pid": identity.get("parent_pid"),
            "creation_time": identity.get("creation_time"),
            "parent_creation_time": parent["creation_time"],
            "image": None if image is None else image[:512],
            "image_truncated": image is not None and len(image) > 512,
            "reason": reason,
        }
        if row in excluded["records"]:
            return
        if len(excluded["records"]) < MAX_EXCLUDED_LINEAGE_DIAGNOSTICS:
            excluded["records"].append(row)
        else:
            excluded["omitted_observations"] += 1

    while frontier:
        parent_pid = frontier.pop()
        if parent_pid in visited:
            continue
        visited.add(parent_pid)
        parent = identities[parent_pid]
        _require(
            not api.exited(handles[parent_pid]),
            "retained lineage parent exited before discovery",
        )
        for pid in sorted(
            pid for pid, ancestor in parents.items() if ancestor == parent_pid
        ):
            _require(
                pid != launcher_pid,
                "pre-ACK launcher appears in its own descendant tree",
            )
            if pid in handles:
                identity = identities[pid]
                _require(
                    identity["parent_pid"] == parent_pid
                    and identity["creation_time"] >= parent["creation_time"],
                    "retained pre-ACK child has inconsistent immediate-parent identity",
                )
                frontier.append(pid)
                continue
            handle = None
            try:
                try:
                    handle = api.open_process(pid)
                except OSError:
                    if pid not in api.parents():
                        exclude(pid, parent, "candidate_disappeared_before_open")
                        continue
                    raise
                if api.exited(handle):
                    exclude(pid, parent, "candidate_exited_before_identity")
                    continue
                # identity(handle) takes a fresh parent snapshot. Check the
                # same handle again afterwards so an exited/reused PID is not
                # paired with the current process's parent relationship.
                identity = api.identity(handle)
                _require(identity["pid"] == pid, "opened lineage PID differs")
                if api.exited(handle):
                    exclude(pid, parent, "candidate_exited_during_identity", identity)
                    continue
                if identity["creation_time"] < parent["creation_time"]:
                    exclude(pid, parent, "stale_parent_pid_creation", identity)
                    continue
                _require(
                    not api.exited(handles[parent_pid]),
                    "retained lineage parent exited during identity admission",
                )
                _require(
                    identity["parent_pid"] == parent_pid,
                    "potentially owned lineage parent changed after snapshot",
                )
                _require(
                    len(handles) < MAX_JOB_PROCESSES,
                    "admitted pre-ACK lineage exceeds process allowance",
                )
                # admit stores the checked handle before assigning its job.
                # If assignment fails, cleanup still owns this exact handle.
                admit(pid, handle, identity)
                _require(
                    handles.get(pid) == handle,
                    "lineage admission did not retain its handle",
                )
                frontier.append(pid)
            finally:
                if handle is not None and handles.get(pid) != handle:
                    api.close(handle)


def _settle_owned_processes(
    api, job, handles, *, timeout_seconds=10, clock=monotonic, pause=sleep
):
    """Wait for asynchronous job termination before any individual fallback.

    A failed termination call is only a call diagnostic if the same retained
    handle becomes signalled. Unresolved handles or job membership always fail
    cleanup, independently of the primary reason for stopping the worker.
    """
    deadline = clock() + timeout_seconds
    errors, calls, terminated = [], [], set()
    try:
        initial_members = set(api.members(job))
    except BaseException as error:
        initial_members = set()
        errors.append(f"initial_job_inventory:{type(error).__name__}:{error}"[:1000])
    job_requested = False
    try:
        api.terminate_job(job)
        job_requested = True
        terminated.update(initial_members)
    except BaseException as error:
        calls.append({"kind": "job", "error": f"{type(error).__name__}:{error}"[:1000]})

    def state():
        try:
            empty = not api.members(job)
        except BaseException:
            empty = False
        signalled = {}
        for pid, handle in handles.items():
            try:
                signalled[pid] = api.exited(handle)
            except BaseException:
                signalled[pid] = False
        return empty, signalled

    # TerminateJobObject is asynchronous. Its members need a wait, not an
    # immediate second termination request while Windows is tearing them down.
    if job_requested:
        job_wait_deadline = min(deadline, clock() + 1)
        while True:
            empty, signalled = state()
            if empty and all(signalled.get(pid, True) for pid in initial_members):
                break
            if clock() >= job_wait_deadline:
                break
            pause(min(0.025, max(0, job_wait_deadline - clock())))

    for pid, handle in handles.items():
        try:
            if api.exited(handle) or (job_requested and pid in initial_members):
                continue
            terminated.add(pid)
            api.terminate_process(handle)
        except BaseException as error:
            calls.append(
                {
                    "kind": "process",
                    "pid": pid,
                    "error": f"{type(error).__name__}:{error}"[:1000],
                }
            )
    while True:
        empty, signalled = state()
        if empty and all(signalled.values()):
            break
        if clock() >= deadline:
            break
        pause(min(0.025, max(0, deadline - clock())))
    if not empty:
        errors.append("job_still_has_active_members_or_inventory_unavailable")
    errors.extend(
        f"owned_process_not_signalled:{pid}"
        for pid, done in signalled.items()
        if not done
    )
    return {
        "job_empty": empty,
        "signalled": signalled,
        "errors": errors,
        "termination_call_diagnostics": calls,
        "terminated_process_ids": sorted(terminated),
        "job_termination_requested": job_requested,
    }


class WindowsProcesses:
    """Small handle-owning Win32 adapter; methods neither import nor run TNFR."""

    def __init__(self):
        _require(os.name == "nt", "registered supervision requires Windows")
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.p = ctypes.WinDLL("psapi", use_last_error=True)
        size = ctypes.c_size_t

        class Limits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_longlong),
                ("job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD),
                ("minimum_ws", size),
                ("maximum_ws", size),
                ("active_processes", wintypes.DWORD),
                ("affinity", size),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class Io(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "read_ops",
                    "write_ops",
                    "other_ops",
                    "read_bytes",
                    "write_bytes",
                    "other_bytes",
                )
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Limits),
                ("io", Io),
                ("process_memory", size),
                ("job_memory", size),
                ("peak_process_memory", size),
                ("peak_job_memory", size),
            ]

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("faults", wintypes.DWORD),
                ("peak_ws", size),
                ("ws", size),
                ("peak_paged", size),
                ("paged", size),
                ("peak_nonpaged", size),
                ("nonpaged", size),
                ("pagefile", size),
                ("peak_pagefile", size),
            ]

        class Entry(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("usage", wintypes.DWORD),
                ("pid", wintypes.DWORD),
                ("heap", size),
                ("module", wintypes.DWORD),
                ("threads", wintypes.DWORD),
                ("parent_pid", wintypes.DWORD),
                ("priority", wintypes.LONG),
                ("flags", wintypes.DWORD),
                ("exe", wintypes.WCHAR * 260),
            ]

        self.Extended, self.Counters, self.Entry = Extended, Counters, Entry
        signatures = {
            "GetSystemDirectoryW": ([wintypes.LPWSTR, wintypes.UINT], wintypes.UINT),
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "OpenJobObjectW": (
                [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR],
                wintypes.HANDLE,
            ),
            "SetInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            "QueryInformationJobObject": (
                [
                    wintypes.HANDLE,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                ],
                wintypes.BOOL,
            ),
            "AssignProcessToJobObject": (
                [wintypes.HANDLE, wintypes.HANDLE],
                wintypes.BOOL,
            ),
            "IsProcessInJob": (
                [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)],
                wintypes.BOOL,
            ),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "TerminateProcess": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "OpenProcess": (
                [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
                wintypes.HANDLE,
            ),
            "GetProcessId": ([wintypes.HANDLE], wintypes.DWORD),
            "GetExitCodeProcess": (
                [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)],
                wintypes.BOOL,
            ),
            "GetProcessTimes": (
                [
                    wintypes.HANDLE,
                    ctypes.POINTER(wintypes.FILETIME),
                    ctypes.POINTER(wintypes.FILETIME),
                    ctypes.POINTER(wintypes.FILETIME),
                    ctypes.POINTER(wintypes.FILETIME),
                ],
                wintypes.BOOL,
            ),
            "QueryFullProcessImageNameW": (
                [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    wintypes.LPWSTR,
                    ctypes.POINTER(wintypes.DWORD),
                ],
                wintypes.BOOL,
            ),
            "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            "CreateToolhelp32Snapshot": (
                [wintypes.DWORD, wintypes.DWORD],
                wintypes.HANDLE,
            ),
            "Process32FirstW": (
                [wintypes.HANDLE, ctypes.POINTER(Entry)],
                wintypes.BOOL,
            ),
            "Process32NextW": ([wintypes.HANDLE, ctypes.POINTER(Entry)], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result
        self.p.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        self.p.GetProcessMemoryInfo.restype = wintypes.BOOL

    @staticmethod
    def checked(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def close(self, handle):
        if handle:
            self.k.CloseHandle(handle)

    def open_process(self, pid):
        return self.checked(
            self.k.OpenProcess(
                0x100000 | 0x1000 | 0x0400 | 0x0010 | 0x0100 | 0x0001, False, pid
            )
        )

    def create_job(self, name):
        job = self.checked(self.k.CreateJobObjectW(None, name))
        limits = self.Extended()
        limits.basic.flags = 0x2000 | 0x0008  # kill on close; finite process count
        limits.basic.active_processes = MAX_JOB_PROCESSES
        try:
            self.checked(
                self.k.SetInformationJobObject(
                    job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                )
            )
        except BaseException:
            self.close(job)
            raise
        return job

    def member(self, job, process):
        value = wintypes.BOOL()
        self.checked(self.k.IsProcessInJob(process, job, ctypes.byref(value)))
        return bool(value.value)

    def assign(self, job, process):
        if not self.member(job, process):
            self.checked(self.k.AssignProcessToJobObject(job, process))
        _require(self.member(job, process), "process job association failed")

    def console_image(self):
        directory = ctypes.create_unicode_buffer(32768)
        count = self.k.GetSystemDirectoryW(directory, len(directory))
        self.checked(count)
        _require(count < len(directory), "Windows system directory exceeds its buffer")
        return str(Path(directory.value) / "conhost.exe")

    def terminate_job(self, job):
        self.checked(self.k.TerminateJobObject(job, 1))

    def terminate_process(self, process):
        self.checked(self.k.TerminateProcess(process, 1))

    def members(self, job):
        buffer = ctypes.create_string_buffer(
            8 + MAX_JOB_PROCESSES * ctypes.sizeof(ctypes.c_size_t)
        )
        self.checked(
            self.k.QueryInformationJobObject(job, 3, buffer, len(buffer), None)
        )
        assigned = ctypes.c_uint32.from_buffer(buffer, 0).value
        count = ctypes.c_uint32.from_buffer(buffer, 4).value
        _require(
            count == assigned <= MAX_JOB_PROCESSES, "job member inventory incomplete"
        )
        values = (ctypes.c_size_t * count).from_buffer(buffer, 8)
        return tuple(int(value) for value in values)

    def parents(self):
        snapshot = self.k.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = self.Entry()
            entry.size = ctypes.sizeof(entry)
            present = self.k.Process32FirstW(snapshot, ctypes.byref(entry))
            result = {}
            while present:
                result[int(entry.pid)] = int(entry.parent_pid)
                present = self.k.Process32NextW(snapshot, ctypes.byref(entry))
            return result
        finally:
            self.close(snapshot)

    def identity(self, handle, parents=None):
        pid = int(self.k.GetProcessId(handle))
        self.checked(pid)
        creation, exit_time, kernel_time, user_time = (
            wintypes.FILETIME() for _ in range(4)
        )
        self.checked(
            self.k.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            )
        )
        image = ctypes.create_unicode_buffer(32768)
        length = wintypes.DWORD(len(image))
        self.checked(
            self.k.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(length))
        )
        return {
            "pid": pid,
            "parent_pid": (self.parents() if parents is None else parents).get(pid),
            "image": image.value,
            "creation_time": (int(creation.dwHighDateTime) << 32)
            | int(creation.dwLowDateTime),
        }

    def exited(self, process):
        result = self.k.WaitForSingleObject(process, 0)
        _require(result in (0, 258), "process wait failed")
        return result == 0

    def exit_code(self, process):
        value = wintypes.DWORD()
        self.checked(self.k.GetExitCodeProcess(process, ctypes.byref(value)))
        return int(value.value)

    def rss(self, process):
        value = self.Counters()
        value.cb = ctypes.sizeof(value)
        self.checked(
            self.p.GetProcessMemoryInfo(process, ctypes.byref(value), value.cb)
        )
        return int(value.ws), int(value.peak_ws)


def validate_bootstrap_admission(marker, *, expected_prefix, expected_executable):
    """Check the current actual interpreter and named job, without TNFR imports."""
    _require(
        type(marker) is dict and marker.get("schema") == "tnfr.bootstrap-admitted.v1",
        "admitted bootstrap marker required",
    )
    _require(
        type(marker.get("worker_pid")) is int and marker["worker_pid"] == os.getpid(),
        "bootstrap worker PID differs",
    )
    _require(
        _same_path(sys.prefix, expected_prefix)
        and _same_path(sys.executable, expected_executable),
        "actual interpreter environment differs",
    )
    _require(
        _same_path(marker["prefix"], sys.prefix)
        and _same_path(marker["executable"], sys.executable),
        "marker interpreter association differs",
    )
    request = read_control(marker["request_path"])
    ack = read_control(marker["acknowledgement_path"])
    _require(
        ack.get("schema") == "tnfr.bootstrap-ack.v1"
        and ack.get("job_verified") is True,
        "job-verified bootstrap acknowledgement required",
    )
    _require(
        ack["nonce"] == marker["nonce"] == request["nonce"]
        and ack["worker_pid"] == os.getpid()
        and ack["supervisor_pid"]
        == marker["supervisor_pid"]
        == request["supervisor_pid"],
        "bootstrap nonce or process association differs",
    )
    _require(
        _same_path(ack["worker"], marker["worker"])
        and _same_path(request["worker"], marker["worker"]),
        "worker script association differs",
    )
    api = WindowsProcesses()
    job, process = None, None
    try:
        job = api.checked(api.k.OpenJobObjectW(0x0004, False, ack["job_name"]))
        process = api.open_process(os.getpid())
        identity = api.identity(process)
        _require(
            api.member(job, process)
            and identity["creation_time"] == ack["worker_creation_time"],
            "current interpreter is not the acknowledged process in its job",
        )
        _require(
            _same_path(identity["image"], request["expected_base_executable"]),
            "actual worker OS image differs",
        )
    finally:
        api.close(process)
        api.close(job)
    return dict(ack)


def run_supervised(
    *,
    interpreter,
    worker,
    folder,
    budget,
    worker_args=(),
    bootstrap_path=None,
    expected_prefix=None,
    response_bytes=None,
    stop_requested=None,
    before_ack=None,
):
    """Run one acknowledged worker, retain every failure, terminate the job.

    Test callbacks inspect harmless process state only. Production wrappers
    omit before_ack/stop_requested. No retry, alternate executable or fallback
    launcher is selected after observing a failed admission.
    """
    interpreter, worker, folder = map(
        lambda value: Path(value).resolve(), (interpreter, worker, folder)
    )
    bootstrap_path = Path(
        bootstrap_path or Path(__file__).with_name("bootstrap.py")
    ).resolve()
    expected_prefix = str(Path(expected_prefix or sys.prefix).resolve())
    _require(
        folder.is_dir()
        and interpreter.is_file()
        and worker.is_file()
        and bootstrap_path.is_file(),
        "explicit existing interpreter/worker/control folder required",
    )
    _require(
        all(type(value) is str for value in worker_args),
        "worker arguments require strings",
    )
    for name in CONTROL_NAMES:
        _require(
            not (folder / name).exists()
            and not (folder / (name + ".pending")).exists(),
            "existing supervision evidence forbids another launch",
        )
    wall = budget["watchdog_wall_budget_seconds"]
    memory_limit, output_limit = (
        budget["maximum_rss_bytes"],
        budget["maximum_response_bytes"],
    )
    handshake_timeout = budget.get("handshake_timeout_seconds", 30)
    poll_seconds = budget.get("poll_seconds", 0.1)
    log_limit = budget.get("maximum_log_bytes", 1024**2)
    reserve = budget.get("diagnostic_reserve_bytes", 1024**2)
    _require(
        all(
            type(value) in (int, float) and math.isfinite(value) and value > 0
            for value in (wall, handshake_timeout, poll_seconds)
        ),
        "positive finite process time budgets required",
    )
    _require(
        all(
            type(value) is int and value > 0
            for value in (memory_limit, output_limit, log_limit, reserve)
        )
        and reserve < output_limit,
        "positive ordinary byte budgets required",
    )
    _require(
        log_limit <= 1024**2 and poll_seconds <= 0.1,
        "log or sampling policy exceeds fixed supervision limits",
    )
    if response_bytes is None:

        def response_bytes():
            return sum(
                path.stat().st_size for path in folder.rglob("*") if path.is_file()
            )

    started = monotonic()
    nonce = secrets.token_hex(32)
    job_name = f"Local\\TNFR-{os.getpid()}-{nonce[:16]}"
    request = {
        "schema": "tnfr.bootstrap-request.v1",
        "nonce": nonce,
        "supervisor_pid": os.getpid(),
        "worker": str(worker),
        "worker_args": list(worker_args),
        "ready_path": str(folder / "bootstrap-ready.json"),
        "ack_path": str(folder / "bootstrap-ack.json"),
        "exit_path": str(folder / "bootstrap-exit.json"),
        "expected_prefix": expected_prefix,
        "expected_executable": str(interpreter),
        "expected_base_executable": str(Path(sys._base_executable).resolve()),
        "handshake_timeout_seconds": handshake_timeout,
    }
    write_control(
        folder / "supervision-start.json",
        {
            "schema": "tnfr.windows-job-supervision-start.v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "supervisor_pid": os.getpid(),
            "nonce": nonce,
            "job_name": job_name,
            "interpreter": str(interpreter),
            "worker": str(worker),
            "budget": budget,
            "maximum_job_processes": MAX_JOB_PROCESSES,
            "scope": "One venv launch and actual-interpreter handshake; no TNFR target before ACK; no retry.",
        },
    )
    write_control(folder / "bootstrap-request.json", request)
    api = WindowsProcesses()
    job = launcher = pump = None
    handles, observed, terminated = {}, {}, set()
    launcher_identity = worker_identity = ready = None
    worker_pid = worker_handle = None
    reason, detail = None, None
    handshake_verified = job_verified = acknowledged = False
    peak_aggregate = peak_worker = samples = 0
    worker_exit = launcher_exit = None
    job_empty = worker_waited = launcher_waited = False
    all_owned_waited = False
    cleanup_errors = []
    termination_call_diagnostics = []
    excluded_lineage = {"records": [], "omitted_observations": 0}
    log_exceeded, log_failed = Event(), Event()
    log = None

    def admit_lineage_handle(pid, handle, identity):
        handles[pid] = handle
        observed[pid] = dict(identity, peak_working_set_bytes=0, job_verified=False)
        # Keep an owned, identity-checked handle even if job assignment
        # fails, so failure cleanup can terminate that same process.
        api.assign(job, handle)
        observed[pid]["job_verified"] = True

    def retain(pid, parents=None, identity_check=None):
        if pid not in handles:
            handle = api.open_process(pid)
            try:
                identity = api.identity(handle, parents)
                if identity_check is not None:
                    identity_check(identity)
            except BaseException:
                api.close(handle)
                raise
            admit_lineage_handle(pid, handle, identity)
        elif identity_check is not None:
            identity_check(api.identity(handles[pid], parents))
        return handles[pid]

    def attach_pre_ack_lineage():
        if launcher is None:
            return
        _attach_owned_lineage(
            api,
            launcher.pid,
            handles,
            observed,
            admit_lineage_handle,
            excluded_lineage,
        )

    def sample():
        nonlocal peak_aggregate, peak_worker, samples
        current_total = 0
        parents = api.parents()
        for pid in api.members(job):
            try:
                handle = retain(pid, parents)
                current, peak = api.rss(handle)
            except OSError:
                if pid in api.members(job):
                    raise
                continue
            current_total += current
            observed[pid]["peak_working_set_bytes"] = max(
                observed[pid]["peak_working_set_bytes"], peak
            )
            if pid == worker_pid:
                peak_worker = max(peak_worker, peak)
        peak_aggregate = max(peak_aggregate, current_total)
        samples += 1
        return current_total

    def limit_reason(current_rss):
        if monotonic() - started >= wall:
            return "watchdog_wall_budget_exceeded"
        if current_rss > memory_limit or peak_worker > memory_limit:
            return "aggregate_rss_budget_exceeded"
        if log_exceeded.is_set():
            return "worker_log_budget_exceeded"
        if log_failed.is_set():
            return "worker_log_capture_failed"
        if response_bytes() > output_limit - reserve:
            return "response_byte_budget_exceeded"
        return None

    try:
        job = api.create_job(job_name)
        log = (folder / "worker.log").open("xb")
        environment = dict(os.environ)
        for name in (
            "TNFR_CUT_RESOURCE_PARENT",
            "TNFR_BOOTSTRAP_NONCE",
            "TNFR_BOOTSTRAP_RECEIPT",
        ):
            environment.pop(name, None)
        launcher = subprocess.Popen(
            [
                str(interpreter),
                str(bootstrap_path),
                "--request",
                str(folder / "bootstrap-request.json"),
            ],
            cwd=worker.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        def drain():
            count = 0
            try:
                while chunk := launcher.stdout.read1(65536):
                    remaining = max(0, log_limit - count)
                    if len(chunk) > remaining:
                        log_exceeded.set()
                    value = chunk[:remaining]
                    log.write(value)
                    log.flush()
                    count += len(value)
                os.fsync(log.fileno())
            except BaseException:
                log_failed.set()

        pump = Thread(target=drain, name="tnfr-supervised-log", daemon=True)
        pump.start()
        launcher_handle = retain(launcher.pid)
        launcher_identity = api.identity(launcher_handle)
        _require(
            _same_path(launcher_identity["image"], interpreter),
            "launcher OS image differs from selected executable",
        )
        handshake_started = monotonic()
        while True:
            attach_pre_ack_lineage()
            reason = limit_reason(sample())
            if reason:
                break
            if Path(request["ready_path"]).exists():
                try:
                    ready = read_control(request["ready_path"])
                    _require(
                        ready.get("schema") == "tnfr.bootstrap-ready.v1"
                        and ready.get("nonce") == nonce,
                        "bootstrap schema or nonce differs",
                    )
                    _require(
                        type(ready.get("pid")) is int
                        and ready["pid"] > 0
                        and type(ready.get("ppid")) is int
                        and ready.get("tnfr_imported") is False,
                        "bootstrap PID or no-TNFR premise differs",
                    )
                    for key, expected in (
                        ("executable", interpreter),
                        ("base_executable", request["expected_base_executable"]),
                        ("prefix", expected_prefix),
                    ):
                        _require(
                            _same_path(ready[key], expected),
                            "bootstrap environment differs: " + key,
                        )
                    candidate_pid = ready["pid"]

                    def worker_association(identity):
                        _require(
                            identity["parent_pid"] == ready["ppid"],
                            "worker OS parent differs from handshake",
                        )
                        _require(
                            candidate_pid == launcher.pid
                            or identity["parent_pid"] == launcher.pid,
                            "worker is not the actual launched interpreter",
                        )
                        _require(
                            identity["creation_time"]
                            >= launcher_identity["creation_time"]
                            and _same_path(
                                identity["image"], request["expected_base_executable"]
                            ),
                            "worker creation or OS image association differs",
                        )

                    candidate_handle = retain(
                        candidate_pid, identity_check=worker_association
                    )
                    worker_pid, worker_handle = candidate_pid, candidate_handle
                    worker_identity = api.identity(worker_handle)
                    handshake_verified = True
                    attach_pre_ack_lineage()
                    _verify_pre_ack_inventory(
                        launcher.pid,
                        worker_pid,
                        api.members(job),
                        observed,
                        api.console_image(),
                    )
                    _require(
                        api.member(job, worker_handle)
                        and api.member(job, launcher_handle),
                        "actual interpreter or launcher lost job membership",
                    )
                    job_verified = True
                except BaseException as error:
                    reason, detail = (
                        "handshake_rejected",
                        f"{type(error).__name__}: {error}"[:1000],
                    )
                    break
                if before_ack is not None:
                    try:
                        before_ack(dict(ready))
                    except BaseException as error:
                        reason, detail = (
                            "before_ack_rejected",
                            f"{type(error).__name__}: {error}"[:1000],
                        )
                        break
                reason = limit_reason(sample())
                if reason:
                    break
                write_control(
                    Path(request["ack_path"]),
                    {
                        "schema": "tnfr.bootstrap-ack.v1",
                        "nonce": nonce,
                        "worker_pid": worker_pid,
                        "supervisor_pid": os.getpid(),
                        "job_name": job_name,
                        "worker_creation_time": worker_identity["creation_time"],
                        "job_verified": True,
                        "worker": str(worker),
                        "executable": str(interpreter),
                        "prefix": expected_prefix,
                    },
                )
                acknowledged = True
                break
            if monotonic() - handshake_started >= handshake_timeout:
                reason = "handshake_timeout"
                break
            if api.exited(launcher_handle):
                reason = "worker_nonzero_exit"
                break
            sleep(poll_seconds)
        while acknowledged and reason is None:
            reason = limit_reason(sample())
            if reason:
                break
            if stop_requested is not None and stop_requested():
                reason = "stop_requested"
                break
            if api.exited(worker_handle):
                worker_exit = api.exit_code(worker_handle)
                if worker_exit:
                    reason = "worker_nonzero_exit"
                break
            sleep(poll_seconds)
    except BaseException as error:
        reason = reason or "monitor_error"
        detail = detail or f"{type(error).__name__}: {error}"[:1000]
    finally:
        if job is not None:
            try:
                if not acknowledged:
                    attach_pre_ack_lineage()
            except BaseException as error:
                reason = reason or "cleanup_lineage_error"
                detail = detail or f"{type(error).__name__}: {error}"[:1000]
                cleanup_errors.append(
                    "lineage:" + f"{type(error).__name__}: {error}"[:1000]
                )
            try:
                remaining = api.members(job)
                for pid in remaining:
                    try:
                        retain(pid)
                    except OSError:
                        if pid in api.members(job):
                            raise
            except BaseException as error:
                reason = reason or "cleanup_roster_error"
                detail = detail or f"{type(error).__name__}: {error}"[:1000]
                cleanup_errors.append(
                    "cleanup_roster:" + f"{type(error).__name__}: {error}"[:1000]
                )
            try:
                settled = _settle_owned_processes(api, job, handles)
                job_empty = settled["job_empty"]
                cleanup_errors.extend(settled["errors"])
                termination_call_diagnostics.extend(
                    settled["termination_call_diagnostics"]
                )
                terminated.update(settled["terminated_process_ids"])
            except BaseException as error:
                reason = reason or "cleanup_error"
                detail = detail or f"{type(error).__name__}: {error}"[:1000]
                cleanup_errors.append(
                    "process_cleanup:" + f"{type(error).__name__}: {error}"[:1000]
                )
            finally:
                # Final handle evidence is independent of any intermediate
                # termination-call result, including a racy AccessDenied.
                if worker_handle is not None:
                    try:
                        worker_waited = api.exited(worker_handle)
                        worker_exit = (
                            api.exit_code(worker_handle) if worker_waited else None
                        )
                    except BaseException as error:
                        cleanup_errors.append(
                            "worker_final_wait:"
                            + f"{type(error).__name__}:{error}"[:1000]
                        )
                if launcher is not None:
                    try:
                        launcher.wait(timeout=2)
                        launcher_waited, launcher_exit = True, launcher.returncode
                    except BaseException as error:
                        cleanup_errors.append(
                            "launcher_final_wait:"
                            + f"{type(error).__name__}:{error}"[:1000]
                        )
                # This last handle owns kill-on-close even if an earlier
                # cleanup operation failed; retained process handles allow
                # post-close wait evidence where available.
                api.close(job)
                job = None
        if launcher is not None and launcher.poll() is None:
            try:
                launcher.kill()
                launcher.wait(timeout=2)
                launcher_waited, launcher_exit = True, launcher.returncode
            except BaseException as error:
                cleanup_errors.append(
                    "launcher_cleanup:" + f"{type(error).__name__}: {error}"[:1000]
                )
        if pump is not None:
            pump.join(timeout=2)
            if pump.is_alive():
                reason = reason or "worker_log_drain_failed"
                cleanup_errors.append("worker_log_thread_not_joined")
        if log_failed.is_set():
            cleanup_errors.append("log_capture_failed")
        if log is not None:
            log.close()
        all_owned_waited = all(
            api.k.WaitForSingleObject(handle, 0) == 0 for handle in handles.values()
        )
        if not all_owned_waited:
            cleanup_errors.append("one_or_more_owned_process_handles_not_signalled")
        for handle in handles.values():
            api.close(handle)
    cleanup_complete = (
        job_empty
        and all_owned_waited
        and not cleanup_errors
        and (launcher is None or launcher_waited)
        and (pump is None or not pump.is_alive())
    )
    if reason is None:
        if not (
            handshake_verified
            and job_verified
            and acknowledged
            and cleanup_complete
            and worker_waited
            and launcher_waited
            and worker_exit == 0
        ):
            reason = "incomplete_supervision_admission"
        elif log_exceeded.is_set():
            reason = "worker_log_budget_exceeded"
        elif log_failed.is_set():
            reason = "worker_log_capture_failed"
    if reason is None:
        try:
            completion = read_control(Path(request["exit_path"]))
            _require(
                completion.get("schema") == "tnfr.bootstrap-exit.v1"
                and completion.get("nonce") == nonce
                and completion.get("worker_pid") == worker_pid
                and completion.get("exit_code") == 0
                and completion.get("target_authorized") is True
                and _same_path(completion["worker"], worker),
                "bootstrap completion association differs",
            )
        except BaseException as error:
            reason, detail = (
                "bootstrap_completion_missing",
                f"{type(error).__name__}: {error}"[:1000],
            )
    result = {
        "schema": "tnfr.windows-job-supervision-result.v1",
        "reason": reason,
        "detail": detail,
        "elapsed_seconds": monotonic() - started,
        "worker_exit_code": worker_exit,
        "worker_pid": worker_pid,
        "launcher_pid": None if launcher is None else launcher.pid,
        "launch_pid": None if launcher is None else launcher.pid,
        "launcher_exit_code": launcher_exit,
        "handshake_verified": handshake_verified,
        "job_verified": job_verified,
        "acknowledged": acknowledged,
        "target_authorized": acknowledged,
        "job_empty_after_cleanup": job_empty,
        "worker_waited": worker_waited,
        "launcher_waited": launcher_waited,
        "all_owned_processes_waited": all_owned_waited,
        "cleanup_complete": cleanup_complete,
        "cleanup_errors": cleanup_errors,
        "termination_call_diagnostics": termination_call_diagnostics,
        "excluded_lineage": excluded_lineage,
        "cleanup_timeout_seconds": 10,
        "terminated_process_ids": sorted(terminated),
        "worker_identity": worker_identity,
        "launcher_identity": launcher_identity,
        "observed_members": [observed[pid] for pid in sorted(observed)],
        "peak_aggregate_working_set_bytes": peak_aggregate,
        "peak_worker_working_set_bytes": peak_worker,
        "rss_samples": samples,
        "handshake": ready,
        "resource_scope": "Whole-job aggregate RSS sampled at most every100ms; actual-worker OS peak additionally checked. Shared pages may be counted more than once. No hard OS RSS allocation limit is claimed.",
        "maximum_job_processes": MAX_JOB_PROCESSES,
        "retry_permitted": False,
    }
    write_control(folder / "supervision-result.json", result)
    return result

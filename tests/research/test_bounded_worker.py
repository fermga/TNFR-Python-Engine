"""Short synthetic OS controls, never a scientific producer or benchmark.

Run only under the reviewed non-scientific control protocol. Native tests use
96 MiB and short deadlines. Children import only the standard library and, for
the Job inspector control, this infrastructure module by its explicit path.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from tnfr.research import bounded_worker as owner

WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Windows Job Object control")
MEMORY = 96 << 20


def _script(tmp_path, text):
    path = tmp_path / "synthetic.py"
    path.write_text(text, encoding="utf-8")
    return path


def _run(tmp_path, text, **limits):
    options = dict(wall_seconds=10, process_memory_bytes=MEMORY, log_limit_bytes=4096)
    options.update(limits)
    return owner.run_bounded_worker(
        _script(tmp_path, text),
        cwd=tmp_path,
        attempt_dir=tmp_path / "attempt",
        **options,
    )


@WINDOWS
def test_single_worker_preserves_admitted_venv_and_immutable_evidence(tmp_path):
    result = _run(
        tmp_path, "import json,sys\nprint(json.dumps([sys.prefix,sys.executable]))"
    )
    assert result["passed"] and result["status"] == "success"
    assert result["job_assigned_before_resume"] and result["resumed"]
    prefix, executable = json.loads((tmp_path / "attempt/worker.log").read_text())
    assert Path(prefix).resolve() == Path(sys.prefix).resolve()
    assert Path(executable).resolve() == Path(sys.executable).resolve()
    assert 0 < result["peak_process_commit_bytes"] <= MEMORY
    assert result["worker_pid"] != os.getpid()
    original = (tmp_path / "attempt/execution.json").read_bytes()
    with pytest.raises(FileExistsError):
        owner.run_bounded_worker(
            tmp_path / "synthetic.py",
            cwd=tmp_path,
            attempt_dir=tmp_path / "attempt",
            process_memory_bytes=MEMORY,
        )
    assert (tmp_path / "attempt/execution.json").read_bytes() == original


@WINDOWS
def test_worker_can_inspect_its_effective_immediate_job(tmp_path):
    script = (
        "import importlib.util,json\n"
        f"s=importlib.util.spec_from_file_location('bounded_worker',{owner.__file__!r})\n"
        "m=importlib.util.module_from_spec(s)\ns.loader.exec_module(m)\n"
        f"print(json.dumps(m.require_current_job_limits({MEMORY})))\n"
    )
    result = _run(tmp_path, script)
    assert result["passed"], result
    actual = json.loads((tmp_path / "attempt/worker.log").read_text())
    assert actual["current_process_in_job"] and actual["kill_on_close"]
    assert not actual["breakaway_allowed"] and not actual["wall_deadline_verified"]
    assert actual["active_process_limit"] == 1
    assert actual["process_memory_limit_bytes"] == MEMORY
    assert actual["job_memory_limit_bytes"] == MEMORY


def test_inspector_rejects_a_process_outside_any_job(monkeypatch):
    class OutsideJob:
        def GetCurrentProcess(self):
            return -1

        def IsProcessInJob(self, process, job, belongs):
            assert job is None
            owner.C.cast(belongs, owner.C.POINTER(owner.W.BOOL))[0] = False
            return 1

        def QueryInformationJobObject(self, *args):
            pytest.fail("An unassigned process must be rejected before query")

    monkeypatch.setattr(owner, "_winapi", lambda: OutsideJob())
    with pytest.raises(OSError, match="outside a bounded"):
        owner.require_current_job_limits(MEMORY)


@pytest.mark.parametrize(
    "change",
    [
        {"flags": owner._LIMIT_FLAGS & ~0x100},
        {"flags": owner._LIMIT_FLAGS & ~0x200},
        {"flags": owner._LIMIT_FLAGS & ~0x2000},
        {"flags": owner._LIMIT_FLAGS | 0x800},
        {"flags": owner._LIMIT_FLAGS | 0x1000},
        {"active": 2},
        {"process_memory": MEMORY + 1},
        {"job_memory": MEMORY - 1},
    ],
)
def test_inspector_rejects_incomplete_or_different_limits(monkeypatch, change):
    class DifferentJob:
        def GetCurrentProcess(self):
            return -1

        def IsProcessInJob(self, process, job, belongs):
            owner.C.cast(belongs, owner.C.POINTER(owner.W.BOOL))[0] = True
            return 1

        def QueryInformationJobObject(self, job, kind, output, size, length):
            assert job is None and kind == 9
            actual = owner.C.cast(
                output, owner.C.POINTER(owner._ExtendedLimits)
            ).contents
            actual.BasicLimitInformation.LimitFlags = change.get(
                "flags", owner._LIMIT_FLAGS
            )
            actual.BasicLimitInformation.ActiveProcessLimit = change.get("active", 1)
            actual.ProcessMemoryLimit = change.get("process_memory", MEMORY)
            actual.JobMemoryLimit = change.get("job_memory", MEMORY)
            return 1

    monkeypatch.setattr(owner, "_winapi", lambda: DifferentJob())
    with pytest.raises(OSError, match="exact required hard limits"):
        owner.require_current_job_limits(MEMORY)


@WINDOWS
def test_nonzero_exit_and_stderr_are_retained(tmp_path):
    result = _run(
        tmp_path, "import sys\nprint('expected failure',file=sys.stderr)\nsys.exit(7)"
    )
    assert not result["passed"] and result["status"] == "worker_failed"
    assert result["exit_code"] == 7
    assert b"expected failure" in (tmp_path / "attempt/worker.log").read_bytes()


@WINDOWS
def test_external_deadline_stops_worker_without_a_later_marker(tmp_path):
    result = _run(
        tmp_path,
        "import time\nfrom pathlib import Path\n"
        "print('prefix',flush=True)\ntime.sleep(5)\n"
        "Path('forbidden-late-marker').write_text('late')\n",
        wall_seconds=1,
    )
    assert not result["passed"] and result["status"] in (
        "timeout",
        "timeout_during_close",
    )
    assert result["closing_reserve_seconds"] == 0.25
    assert result["worker_seconds_budget"] == 0.75
    assert not (tmp_path / "forbidden-late-marker").exists()
    assert (tmp_path / "attempt/execution.json").is_file()


@WINDOWS
def test_watchdog_stops_worker_when_parent_log_drain_stalls(tmp_path, monkeypatch):
    original = owner._drain
    stalled = False

    def delayed_drain(*args):
        nonlocal stalled
        if not stalled:
            stalled = True
            time.sleep(1.25)  # Deliberately exceed the total synthetic budget.
        return original(*args)

    monkeypatch.setattr(owner, "_drain", delayed_drain)
    result = _run(
        tmp_path,
        "import time\nfrom pathlib import Path\n"
        "print('prefix',flush=True)\ntime.sleep(.95)\n"
        "Path('forbidden-late-marker').write_text('late')\n",
        wall_seconds=1,
    )
    assert stalled and result["watchdog_fired"]
    assert not result["passed"] and result["status"] == "timeout_during_close"
    assert result["total_budget_exhausted"]
    assert not (tmp_path / "forbidden-late-marker").exists()


@WINDOWS
def test_terminal_write_overrun_cannot_return_success(tmp_path, monkeypatch):
    original = owner._write_record
    monotonic = owner.time.monotonic
    offset = 0

    def controlled_clock():
        return monotonic() + offset

    def delayed_record(path, payload):
        nonlocal offset
        original(path, payload)
        if path.name == "execution.json":
            # The child and watchdog are already stopped. Cross the total
            # budget deterministically, without a load-sensitive short launch.
            offset = 20

    monkeypatch.setattr(owner.time, "monotonic", controlled_clock)
    monkeypatch.setattr(owner, "_write_record", delayed_record)
    result = _run(tmp_path, "print('synthetic completed')")
    assert not result["passed"] and result["status"] == "timeout_during_close"
    assert (
        result["total_budget_exhausted"] and result["terminal_write_deadline_verified"]
    )
    snapshot = json.loads((tmp_path / "attempt/execution.json").read_text())
    assert snapshot["passed"] and snapshot["status"] == "success"
    assert snapshot["record_stage"] == "pre_write_snapshot_not_final_attempt_verdict"
    assert not snapshot["terminal_write_deadline_verified"]


@WINDOWS
def test_os_denies_a_commit_larger_than_the_process_limit(tmp_path):
    # No Python allocation probe or memory polling: ask Windows to commit more
    # than the entire admitted cap. Correct enforcement denies that request.
    script = """import ctypes as C
k=C.WinDLL('kernel32',use_last_error=True)
k.VirtualAlloc.argtypes=[C.c_void_p,C.c_size_t,C.c_ulong,C.c_ulong]
k.VirtualAlloc.restype=C.c_void_p
p=k.VirtualAlloc(None,256*1024*1024,0x3000,4)
if p:
    raise RuntimeError('Job committed-memory limit failed')
print('Windows denied excess commitment',flush=True)
"""
    result = _run(tmp_path, script)
    assert result["passed"], result
    assert b"denied excess commitment" in (tmp_path / "attempt/worker.log").read_bytes()
    assert result["peak_process_commit_bytes"] <= MEMORY


@WINDOWS
def test_job_prevents_a_second_process(tmp_path):
    script = """import subprocess,sys
try:
    subprocess.run([sys.executable,'-E','-c','print("forbidden")'],check=True)
except OSError:
    print('second process denied')
else:
    raise RuntimeError('another process escaped the one-worker limit')
"""
    result = _run(tmp_path, script)
    assert result["passed"], result
    assert b"second process denied" in (tmp_path / "attempt/worker.log").read_bytes()


@WINDOWS
def test_output_quota_terminates_without_growing_the_log(tmp_path):
    result = _run(
        tmp_path,
        "import sys\nsys.stdout.write('X'*1000000)\nsys.stdout.flush()",
        log_limit_bytes=64,
    )
    assert not result["passed"] and result["status"] == "log_limit"
    assert (tmp_path / "attempt/worker.log").stat().st_size == 64
    assert result["log_bytes"] == 64


@WINDOWS
def test_failed_job_verification_never_resumes_user_script(tmp_path, monkeypatch):
    native = owner._winapi()

    class RejectMembership:
        def IsProcessInJob(self, *args):
            return 0

        def __getattr__(self, name):
            return getattr(native, name)

    monkeypatch.setattr(owner, "_winapi", lambda: RejectMembership())
    result = _run(
        tmp_path, "from pathlib import Path\nPath('forbidden-entry').write_text('ran')"
    )
    assert not result["passed"] and result["status"] == "enforcement_unavailable"
    assert not result["resumed"]
    assert result["worker_pid"] is not None
    assert result["worker_stopped_observed"] and result["cleanup_complete"]
    assert not (tmp_path / "forbidden-entry").exists()


def test_unavailable_hard_enforcement_fails_closed(tmp_path, monkeypatch):
    def unavailable():
        raise OSError("synthetic unavailable Windows enforcement")

    monkeypatch.setattr(owner, "_winapi", unavailable)
    result = _run(tmp_path, "raise RuntimeError('must not execute')")
    assert not result["passed"] and result["status"] == "enforcement_unavailable"
    assert not result["resumed"] and result["worker_pid"] is None
    assert result["log_bytes"] == 0


@pytest.mark.parametrize(
    "limits",
    [
        {"wall_seconds": True},
        {"wall_seconds": float("nan")},
        {"wall_seconds": float("inf")},
        {"wall_seconds": 0},
        {"wall_seconds": 3601},
        {"process_memory_bytes": True},
        {"process_memory_bytes": 0},
        {"process_memory_bytes": (1 << 30) + 1},
        {"log_limit_bytes": 0},
        {"log_limit_bytes": (1 << 20) + 1},
    ],
)
def test_invalid_limits_cannot_reserve_or_launch(tmp_path, limits):
    with pytest.raises(ValueError):
        _run(tmp_path, "raise RuntimeError('must not execute')", **limits)
    assert not (tmp_path / "attempt").exists()


def test_nonsequence_arguments_rejected_before_attempt(tmp_path):
    with pytest.raises(TypeError):
        owner.run_bounded_worker(
            _script(tmp_path, "pass"),
            arguments="not-a-sequence",
            cwd=tmp_path,
            attempt_dir=tmp_path / "attempt",
        )
    assert not (tmp_path / "attempt").exists()


def test_metadata_quota_rejects_before_launch(tmp_path, monkeypatch):
    def forbidden():
        pytest.fail("Oversized metadata must not reach process admission")

    monkeypatch.setattr(owner, "_winapi", forbidden)
    with pytest.raises(ValueError, match="metadata exceeds"):
        owner.run_bounded_worker(
            _script(tmp_path, "pass"),
            arguments=("x" * (600 << 10),),
            cwd=tmp_path,
            attempt_dir=tmp_path / "attempt",
        )
    assert not (tmp_path / "attempt/attempt.json").exists()
    assert not (tmp_path / "attempt/worker.log").exists()

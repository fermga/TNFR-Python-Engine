"""Source-independent controls of the new actual-inclusion execution policy."""

from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
import sys
import zipfile

import pytest

from tnfr.research import formed_two_port_arrival_run as owner
from tnfr.research.artifact_io import decode_exact_tree, file_receipt
from tnfr.utils.io import json_loads


@dataclass(frozen=True)
class SyntheticModel:
    synthetic: bool = True


@dataclass(frozen=True)
class SyntheticSource:
    time: Q = Q(1)
    center: tuple = (Q(0),)
    radius: Q = Q(1, 100)
    evidence_receipts: tuple = ()


@dataclass(frozen=True)
class SyntheticTarget:
    evidence_receipts: tuple = ()


@dataclass(frozen=True)
class SyntheticStep:
    endpoint_center: tuple
    endpoint_radius: Q


@dataclass(frozen=True)
class SyntheticDecision:
    status: str
    time: Q


@pytest.fixture(autouse=True)
def forbid_scientific_response(monkeypatch):
    from tnfr.research import formed_two_port_arrival as model
    from tnfr.mathematics import _validated_metric_inclusion as kernel

    def forbidden(*args, **kwargs):
        pytest.fail("Runner controls must not evaluate the actual scientific response")

    for name in (
        "build_formed_two_port_arrival_model",
        "admit_formed_two_port_arrival_source",
        "admit_formed_two_port_arrival_target",
        "assess_formed_two_port_arrival_terminal",
    ):
        monkeypatch.setattr(model, name, forbidden)
    monkeypatch.setattr(kernel, "validated_metric_inclusion_step", forbidden)


def wire(monkeypatch, *, terminal_at=None, verdict="certified_capture", fail_at=None):
    from tnfr.research import bounded_worker, formed_two_port_arrival as model

    calls = {"steps": [], "decisions": []}
    monkeypatch.setattr(bounded_worker, "require_current_job_limits", lambda **k: {})
    monkeypatch.setattr(owner, "admit_execution_inputs", lambda *a: {
        "prefix_evidence": [], "target_evidence": [], "synthetic": True,
    })
    monkeypatch.setattr(model, "build_formed_two_port_arrival_model", SyntheticModel)
    monkeypatch.setattr(model, "admit_formed_two_port_arrival_source",
                        lambda *a: SyntheticSource())
    monkeypatch.setattr(model, "admit_formed_two_port_arrival_target",
                        lambda *a: SyntheticTarget())

    def decide(model, target, center, radius, time):
        index = len(calls["decisions"])
        calls["decisions"].append((center, radius, time))
        return SyntheticDecision(verdict if index == terminal_at else "unavailable", time)

    def advance(model, center, radius, time):
        index = len(calls["steps"])
        calls["steps"].append((center, radius, time))
        if index == fail_at:
            return None, None, "synthetic_unavailable"
        return SyntheticStep((Q(index+1),), radius + Q(1, 1000)), None, None

    monkeypatch.setattr(model, "assess_formed_two_port_arrival_terminal", decide)
    monkeypatch.setattr(owner, "_advance", advance)
    return calls


def read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


@pytest.mark.parametrize("verdict", ["certified_capture", "certified_nonarrival"])
@pytest.mark.parametrize("terminal_at", [0, 3])
def test_either_declared_proof_stops_at_first_checkpoint(
    tmp_path, monkeypatch, verdict, terminal_at
):
    calls = wire(monkeypatch, terminal_at=terminal_at, verdict=verdict)
    assert owner.run_arrival_attempt(tmp_path, tmp_path / "unused", tmp_path) == 0
    result = read(tmp_path / "worker-result.json")
    assert result["status"] == verdict
    assert result["completed_steps"] == terminal_at
    assert result["endpoint_time"] == 1 + terminal_at * Q(1, 16)
    assert len(calls["steps"]) == terminal_at
    assert len(calls["decisions"]) == terminal_at + 1
    assert len(list((tmp_path / "steps").glob("*.json"))) == terminal_at
    for index in range(terminal_at):
        record = read(tmp_path / "steps" / f"{index:04d}.json")
        assert record["index"] == index
        assert record["time"] == 1 + index * Q(1, 16)
        assert record["endpoint_time"] == 1 + (index + 1) * Q(1, 16)
        if index == 0:
            assert record["previous_complete_step"] is None
        else:
            assert record["previous_complete_step"]["path"] == f"steps/{index-1:04d}.json"
        assert calls["steps"][index][0] == ((Q(0),) if index == 0 else (Q(index),))


def test_unavailable_step_preserves_prefix_and_cannot_be_retried(tmp_path, monkeypatch):
    calls = wire(monkeypatch, fail_at=2)
    assert owner.run_arrival_attempt(tmp_path, tmp_path / "unused", tmp_path) == 2
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = read(tmp_path / "worker-result.json")
    assert result["status"] == "unavailable"
    assert result["completed_steps"] == 2
    assert result["nonarrival_not_proved"] is True
    assert len(calls["decisions"]) == 3
    failed = read(tmp_path / "failed-step.json")
    assert failed["index"] == 2 and failed["time"] == Q(9, 8)
    assert failed["initial_center"] == (Q(2),)
    with pytest.raises(FileExistsError, match="no resume"):
        owner.run_arrival_attempt(tmp_path, tmp_path / "unused", tmp_path)
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_full_synthetic_schedule_is_fixed_and_horizon_is_inconclusive(tmp_path, monkeypatch):
    calls = wire(monkeypatch)
    assert owner.run_arrival_attempt(tmp_path, tmp_path / "unused", tmp_path) == 2
    result = read(tmp_path / "worker-result.json")
    assert result["status"] == "inconclusive"
    assert result["completed_steps"] == 496
    assert result["endpoint_time"] == 32
    assert len(calls["steps"]) == 496 and len(calls["decisions"]) == 497
    assert calls["decisions"][-1][0] == (Q(496),)
    assert calls["decisions"][-1][1] == Q(1, 100) + 496 * Q(1, 1000)


def test_job_gate_precedes_writes_or_science(tmp_path, monkeypatch):
    from tnfr.research import bounded_worker

    def fail(**kwargs):
        raise RuntimeError("synthetic missing job")

    monkeypatch.setattr(bounded_worker, "require_current_job_limits", fail)
    with pytest.raises(RuntimeError, match="missing job"):
        owner.run_arrival_attempt(tmp_path, tmp_path / "unused", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unknown_verdict_does_not_become_success(tmp_path, monkeypatch):
    wire(monkeypatch, terminal_at=0, verdict="looks_like_capture")
    assert owner.run_arrival_attempt(tmp_path, tmp_path / "unused", tmp_path) == 2
    result = read(tmp_path / "worker-result.json")
    assert result["status"] == "unavailable"
    assert result["reason"] == "unknown mathematical terminal verdict"


def test_policy_binds_fixed_clock_budget_and_no_parameter_override():
    policy = owner.declared_policy()
    assert policy["start_slow_time"] + policy["maximum_steps"] * policy["step_size"] == 32
    assert policy["step_bytes"] + policy["auxiliary_bytes"] == 320 * 1024**2
    assert policy["growth_bisections"] == 12
    assert policy["growth_rate_bounds"] == (Q(-1, 64), Q(2))
    assert policy["adaptive_parameters"] is policy["resume"] is False
    assert policy["execution_revision"] == "chunked_artifact_reads_and_durable_phase_diagnostics_v2"


@pytest.mark.parametrize("where,phase", [
    ("source", "source_admission_started"),
    ("target", "target_admission_started"),
    ("initial", "initial_decision_started"),
    ("serialization", "admission_write_started"),
])
def test_memory_failure_retains_exact_entered_phase_and_bounded_stack(
    tmp_path, monkeypatch, capsys, where, phase
):
    from tnfr.research import formed_two_port_arrival as model

    wire(monkeypatch)
    def fail(*args, **kwargs):
        raise MemoryError("synthetic allocation failure")
    names = {
        "source": "admit_formed_two_port_arrival_source",
        "target": "admit_formed_two_port_arrival_target",
        "initial": "assess_formed_two_port_arrival_terminal",
    }
    if where in names:
        monkeypatch.setattr(model, names[where], fail)
    else:
        original = owner.EvidenceWriter.write
        def failed_write(self, name, *args, **kwargs):
            if name == "admission.json":
                fail()
            return original(self, name, *args, **kwargs)
        monkeypatch.setattr(owner.EvidenceWriter, "write", failed_write)
    assert owner.run_arrival_attempt(tmp_path, tmp_path / "unused", tmp_path) == 2
    result = read(tmp_path / "worker-result.json")
    assert result["status"] == "unavailable" and result["completed_steps"] == 0
    assert result["failure_phase"] == phase and result["exception_type"] == "MemoryError"
    assert 1 <= len(result["traceback_frames"]) <= 12
    assert all(set(row) == {"file", "function", "line"} for row in result["traceback_frames"])
    journal = [read(path) for path in sorted(tmp_path.glob("phase-*.json"))]
    assert journal[-1]["phase"] == phase
    assert [row["ordinal"] for row in journal] == list(range(len(journal)))
    assert not (tmp_path / "admission.json").exists()
    diagnostic = json_loads(capsys.readouterr().err.strip())
    assert diagnostic["failure_phase"] == phase


def test_partial_terminal_is_preserved_and_supervisor_log_retains_failure(
    tmp_path, monkeypatch, capsys
):
    from tnfr.research import formed_two_port_run as old_owner

    wire(monkeypatch, terminal_at=0)
    original = old_owner.safe_write
    def fail_terminal(path, *args, **kwargs):
        if Path(path).name == "worker-result.json":
            Path(path).write_bytes(b"{")
            raise MemoryError("synthetic partial terminal write")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(old_owner, "safe_write", fail_terminal)
    assert owner.run_arrival_attempt(tmp_path, tmp_path / "unused", tmp_path) == 2
    assert (tmp_path / "worker-result.json").read_bytes() == b"{"
    diagnostic = json_loads(capsys.readouterr().err.strip())
    assert diagnostic["failure_phase"] == "terminal_write_started"
    assert diagnostic["exception_type"] == "MemoryError"


def test_exception_locations_have_fixed_count_without_locals():
    def descend(depth):
        if depth:
            return descend(depth-1)
        raise RuntimeError("synthetic")
    try:
        descend(30)
    except RuntimeError as exc:
        rows, truncated = owner._exception_frames(exc)
    assert len(rows) == 12 and truncated is True
    assert all(set(row) == {"file", "function", "line"} for row in rows)


def test_advance_wires_forcing_and_structured_growth_without_scientific_field(monkeypatch):
    from tnfr.mathematics import _validated_metric_inclusion as kernel
    calls = {}

    class Model:
        metric = ((Q(1),),)
        def flow(self, values):
            pytest.fail("wiring must not evaluate a field")
        domain = flow
        def field_norm_upper(self, tube):
            calls["forcing_tube"] = tube
            return Q(1, 10)
        def certify_growth(self, tube, **kwargs):
            calls["growth"] = tube, kwargs
            return "synthetic proof"

    def step(*args, **kwargs):
        assert args[:4] == ((Q(0),), Q(1, 10), ((Q(1),),), Q(1, 16))
        assert kwargs["time"] == 1 and kwargs["order"] == 8
        assert kwargs["forcing_bound"] == Q(1, 7000)
        assert kwargs["forcing_refinement"]("synthetic tube") == Q(1, 200000)
        assert kwargs["growth_certifier"]("synthetic tube") == "synthetic proof"
        return None, None, "synthetic"

    monkeypatch.setattr(kernel, "validated_metric_inclusion_step", step)
    assert owner._advance(Model(), (Q(0),), Q(1, 10), Q(1)) == (None, None, "synthetic")
    assert calls["growth"] == ("synthetic tube", {
        "rate_bounds": (Q(-1, 64), Q(2)), "bisections": 12,
    })


def synthetic_protocol(tmp_path, monkeypatch):
    """A tiny frozen file tree; contains no real scientific input or code."""
    import tnfr

    root = (tmp_path / "repo").resolve()
    root.mkdir()
    names = owner._OWNERS | owner._PREFIX_FILES | owner._TARGET_FILES | {
        "src/tnfr/__init__.py", "worker.py",
    }
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic source only\n")
    rows = [file_receipt(root / name, label=name) for name in sorted(names)]
    archive = root / "source.zip"
    with zipfile.ZipFile(archive, "w") as out:
        for name in sorted(names):
            out.write(root / name, name)
    reviews = []
    for index in range(2):
        path = root / f"review-{index}.md"
        path.write_bytes(b"synthetic review fixture; no scientific authorization\n")
        reviews.append(file_receipt(path, label=path.name))
    controls = root / "controls.json"
    controls.write_bytes(owner._json_bytes({"passed": True, "reserved_actual_steps": 0}))
    protocol = root / "protocol.json"
    entry = root / "worker.py"
    monkeypatch.setattr(tnfr, "__file__", str(root / "src/tnfr/__init__.py"))
    monkeypatch.setattr(sys, "argv", [str(entry)])
    record = {
        "schema": owner.SCHEMA, "policy": owner.declared_policy(), "head": "a"*40,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_prefix": str(Path(sys.prefix).resolve()), "python_version": sys.version,
            "package_root": str((root / "src/tnfr").resolve()),
        },
        "sources": rows, "archive": file_receipt(archive, label=archive.name),
        "reviews": reviews, "controls": file_receipt(controls, label=controls.name),
        "worker_entry": file_receipt(entry, label=entry.name),
    }
    output = root / "attempt"
    output.mkdir()
    attempt = {
        "wall_seconds": 3600, "process_memory_bytes": 1024**3,
        "log_limit_bytes": 1024**2, "one_worker": True, "script": str(entry),
        "cwd": str(root), "arguments": ["--root", str(root), "--protocol", str(protocol),
                                        "--output", str(output)],
        "interpreter": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
    }
    (output / "attempt.json").write_bytes(owner._json_bytes(attempt))
    protocol.write_bytes(owner._json_bytes(record))
    return root, protocol, output, record


def test_gate_associates_exact_synthetic_sources_and_copies_immutable_inputs(tmp_path, monkeypatch):
    root, protocol, output, record = synthetic_protocol(tmp_path, monkeypatch)
    writer = owner.EvidenceWriter(output)
    result = owner.admit_execution_inputs(root, protocol, writer)
    assert len(result["prefix_evidence"]) == 8
    assert len(result["target_evidence"]) == 4
    assert (output / "protocol.json").read_bytes() == protocol.read_bytes()
    assert (output / "source.zip").read_bytes() == (root / "source.zip").read_bytes()
    assert (output / "review-00.md").read_bytes() == (root / "review-0.md").read_bytes()


@pytest.mark.parametrize("change,message", [
    ("policy", "selected inclusion policy"),
    ("source", "frozen source receipt differs"),
    ("controls", "source-independent method controls"),
    ("duplicate_review", "duplicate independent review"),
    ("runtime", "reviewed interpreter/source"),
    ("supervisor", "supervisor declaration differs"),
    ("missing_source", "incomplete or duplicate"),
])
def test_gate_refuses_tampered_or_unreviewed_execution_before_science(
    tmp_path, monkeypatch, change, message
):
    root, protocol, output, record = synthetic_protocol(tmp_path, monkeypatch)
    if change == "policy":
        record["policy"]["maximum_steps"] += 1
    elif change == "source":
        (root / "src/tnfr/__init__.py").write_bytes(b"different bytes\n")
    elif change == "controls":
        path = root / "controls.json"
        path.write_bytes(owner._json_bytes({"passed": True, "reserved_actual_steps": 1}))
        record["controls"] = file_receipt(path, label=path.name)
    elif change == "duplicate_review":
        record["reviews"][1] = record["reviews"][0]
    elif change == "runtime":
        record["runtime"]["python_version"] = "different interpreter"
    elif change == "supervisor":
        path = output / "attempt.json"
        attempt = json_loads(path.read_bytes())
        attempt["wall_seconds"] = 3601
        path.write_bytes(owner._json_bytes(attempt))
    elif change == "missing_source":
        record["sources"] = [r for r in record["sources"] if r["path"] != "src/tnfr/__init__.py"]
    protocol.write_bytes(owner._json_bytes(record))
    writer = owner.EvidenceWriter(output)
    with pytest.raises(ValueError, match=message):
        owner.admit_execution_inputs(root, protocol, writer)
    assert not (output / "protocol.json").exists()

"""Synthetic immutable lifecycle controls; never evaluate the reserved response."""

import zipfile
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_localized_mobile_response as owner
from tnfr.mathematics._rational_interval import I as Interval
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.utils.io import json_loads


@dataclass(frozen=True)
class _SyntheticReport:
    status: str = "synthetic_unavailable"
    interval: Interval = Interval(Q(1, 8), Q(1, 4))
    tiny_source_value: Q = Q(1, 10**500)
    observation: None = None


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    root = tmp_path / "synthetic"
    for name, data in {
        "src/tnfr/control.py": b"control = 'complete source'\n",
        owner.RESPONSE_SOURCE: b"# Synthetic response owner; never imported.\n",
        "src/tnfr/untracked.py": b"untracked = 'retained'\n",
        "src/tnfr/__pycache__/ignored.pyc": b"not source",
        owner.DRIVER: Path(owner.__file__).read_bytes(),
        "pyproject.toml": b"[project]\nname = 'synthetic'\n",
        owner.PROOF: b"Current implementation proof differs from pinned protocol.\n",
        owner.RESPONSE_PROOF: b"Numerical method proof before response evaluation.\n",
        **{
            name: b"# Supporting static synthetic control.\n"
            for name in owner.SUPPORTING_TESTS
        },
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    proof = b"Independent frozen protocol at the declared Git revision.\n"

    def git(where, *arguments):
        assert where == root
        if arguments == ("show", f"{owner.PROTOCOL_REVISION}:{owner.PROOF}"):
            return proof
        if arguments == ("rev-parse", "HEAD"):
            return b"a" * 40 + b"\n"
        assert arguments[:3] == ("status", "--porcelain=v1", "--untracked-files=all")
        return b"?? src/tnfr/untracked.py\n"

    monkeypatch.setattr(owner, "_git", git)
    monkeypatch.setattr(owner, "_runtime", lambda: {"python": "synthetic"})
    monkeypatch.setattr(
        owner,
        "_imported_sources",
        lambda where: {"tnfr.control": "src/tnfr/control.py"},
    )
    calls = []
    output = root / owner.EVIDENCE

    def evaluator(progress):
        assert (output / "freeze.json").is_file()
        assert (output / "attempt.json").is_file()
        assert not (output / "report.json").exists()
        calls.append("synthetic_response")
        progress({"panel": 1, "status": "synthetic"})
        return _SyntheticReport()

    monkeypatch.setattr(owner, "_evaluate_response", evaluator)
    return root, output, calls, proof


def test_freeze_and_audit_retain_complete_source_without_response(harness):
    root, output, calls, pinned = harness
    result = owner.freeze(root)
    assert calls == []
    assert result["source_or_execution_authenticated"] is False
    assert result["existing_outcomes"] == ()
    assert (output / "protocol-proof.md").read_bytes() == pinned
    manifest = _read(output / "source-manifest.json")
    names = {item["path"] for item in manifest["files"]}
    assert "src/tnfr/untracked.py" in names
    assert not any("__pycache__" in name for name in names)
    with zipfile.ZipFile(output / "source.zip") as archive:
        for name in names:
            assert archive.read(name) == (root / name).read_bytes()
        assert archive.read("protocol-proof.md") != archive.read(owner.PROOF)
    assert owner.audit(root, require_current=True) == result
    assert calls == []
    with pytest.raises(FileExistsError):
        owner.freeze(root)


def test_first_synthetic_report_follows_attempt_and_preserves_exact_bounds(harness):
    root, output, calls, _ = harness
    owner.freeze(root)
    owner.evaluate(root)
    record = _read(output / "report.json")
    assert record["report"]["status"] == "synthetic_unavailable"
    assert record["report"]["interval"]["lo"] == Q(1, 8)
    assert record["report"]["interval"]["hi"] == Q(1, 4)
    assert record["report"]["tiny_source_value"] == Q(1, 10**500)
    assert record["report"]["observation"] is None
    assert record["attempt"]["path"] == "attempt.json"
    assert calls == ["synthetic_response"]
    assert owner.audit(root)["existing_outcomes"] == ("attempt.json", "report.json")
    with pytest.raises(FileExistsError):
        owner.evaluate(root)
    assert calls == ["synthetic_response"]


@pytest.mark.parametrize("name", owner.OUTCOMES)
def test_any_existing_attempt_outcome_or_export_failure_blocks_before_audit(
    harness, name
):
    root, output, calls, _ = harness
    output.mkdir(parents=True)
    (output / name).write_bytes(b"Retain even an incomplete record.")
    with pytest.raises(FileExistsError):
        owner.evaluate(root)
    assert calls == []
    assert (output / name).read_bytes() == b"Retain even an incomplete record."


@pytest.mark.parametrize("change", ("modified", "additional", "runtime"))
def test_changed_source_or_runtime_blocks_before_first_attempt(
    harness, monkeypatch, change
):
    root, output, calls, _ = harness
    owner.freeze(root)
    if change == "runtime":
        monkeypatch.setattr(owner, "_runtime", lambda: {"python": "different"})
    elif change == "additional":
        (root / "src/tnfr/later.py").write_bytes(b"later = 1\n")
    else:
        (root / "src/tnfr/control.py").write_bytes(b"changed = 1\n")
    # Historical association can still be read without certifying this runtime.
    owner.audit(root)
    with pytest.raises(ValueError, match="source or runtime"):
        owner.evaluate(root)
    assert calls == []
    assert not (output / "attempt.json").exists()


def test_source_change_during_attempt_is_retained_as_failure(harness, monkeypatch):
    root, output, calls, _ = harness
    owner.freeze(root)

    def changed_response(progress):
        calls.append("synthetic_response")
        (root / "src/tnfr/control.py").write_bytes(b"changed = 1\n")
        return _SyntheticReport()

    monkeypatch.setattr(owner, "_evaluate_response", changed_response)
    with pytest.raises(ValueError, match="source or runtime"):
        owner.evaluate(root)
    assert (output / "attempt.json").is_file()
    assert _read(output / "failure.json")["no_retry"] is True
    assert not (output / "report.json").exists()
    with pytest.raises(FileExistsError):
        owner.evaluate(root)
    assert len(calls) == 1


def test_external_loaded_module_blocks_before_first_attempt(harness, monkeypatch):
    root, output, calls, _ = harness
    owner.freeze(root)

    def outside(where):
        raise ValueError("loaded TNFR module is outside the frozen tree")

    monkeypatch.setattr(owner, "_imported_sources", outside)
    with pytest.raises(ValueError, match="outside the frozen tree"):
        owner.evaluate(root)
    assert not (output / "attempt.json").exists()
    assert calls == []


def test_loaded_module_origin_checks_actual_paths(tmp_path, monkeypatch):
    monkeypatch.setitem(
        owner.sys.modules,
        "tnfr.synthetic_external",
        SimpleNamespace(__file__=str(tmp_path / "elsewhere.py")),
    )
    with pytest.raises(ValueError, match="tnfr.synthetic_external"):
        owner._imported_sources(owner.ROOT)


def test_first_computation_failure_is_not_retried(harness, monkeypatch):
    root, output, calls, _ = harness
    owner.freeze(root)

    def failed_response(progress):
        calls.append("synthetic_failure")
        raise ArithmeticError("synthetic precision limit")

    monkeypatch.setattr(owner, "_evaluate_response", failed_response)
    with pytest.raises(ArithmeticError):
        owner.evaluate(root)
    failure = _read(output / "failure.json")
    assert failure["exception"] == "ArithmeticError"
    assert failure["message"] == "synthetic precision limit"
    assert failure["no_retry"] is True
    owner.audit(root)
    with pytest.raises(FileExistsError):
        owner.evaluate(root)
    assert calls == ["synthetic_failure"]


def test_partial_export_remains_with_separate_error_and_blocks_retry(
    harness, monkeypatch
):
    root, output, calls, _ = harness
    owner.freeze(root)
    real_writer = owner.write_json_once

    def fail_export(path, payload, **kwargs):
        if path.name == "report.json":
            path.write_bytes(b'{"partial":')
            raise OSError("synthetic disk failure")
        return real_writer(path, payload, **kwargs)

    monkeypatch.setattr(owner, "write_json_once", fail_export)
    with pytest.raises(OSError):
        owner.evaluate(root)
    assert (output / "report.json").read_bytes() == b'{"partial":'
    assert _read(output / "export-error.json")["exception"] == "OSError"
    with pytest.raises(FileExistsError):
        owner.evaluate(root)
    assert calls == ["synthetic_response"]


def test_changed_archive_is_rejected_without_response(harness):
    root, output, calls, _ = harness
    owner.freeze(root)
    with (output / "source.zip").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="artifact bytes"):
        owner.evaluate(root)
    assert calls == []
    assert not (output / "attempt.json").exists()


def test_incomplete_freeze_is_preserved_without_evaluation(harness, monkeypatch):
    root, output, calls, _ = harness
    original = owner._write_bytes_once

    def fail_archive(path, data):
        if path.name == "source.zip":
            path.write_bytes(b"incomplete archive")
            raise OSError("synthetic freeze failure")
        return original(path, data)

    monkeypatch.setattr(owner, "_write_bytes_once", fail_archive)
    with pytest.raises(OSError):
        owner.freeze(root)
    assert not (output / "freeze.json").exists()
    assert (output / "source.zip").read_bytes() == b"incomplete archive"
    with pytest.raises(FileExistsError):
        owner.freeze(root)
    assert calls == []

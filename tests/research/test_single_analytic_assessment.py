"""Frozen one-assessment lifecycle exercised only with synthetic reports."""

from dataclasses import dataclass
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from benchmarks import organized_interface_factorization as driver
from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I as Interval
from tnfr.research import _single_analytic_assessment as owner
from tnfr.research.artifact_io import decode_exact_tree, file_receipt, sha256_bytes
from tnfr.utils.io import json_loads


@dataclass(frozen=True)
class _FakeReport:
    admitted: bool
    determinant_bounds: Interval = Interval(Q(1, 8), Q(1, 4))
    unresolved_conditions: tuple[str, ...] = ()


def _read(path):
    return decode_exact_tree(json_loads(path.read_text(encoding="utf-8")))


@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    def forbidden():
        pytest.fail("lifecycle test attempted the actual factorization assessment")

    monkeypatch.setattr(driver, "_assess", forbidden)
    with forbid_sine_regeneration():
        yield


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.chdir(driver.ROOT)
    source = {"source_revision": "fixture-revision", "source_tree": "fixture-tree"}
    monkeypatch.setattr(owner, "_source_snapshot", lambda *args: dict(source))
    monkeypatch.setattr(
        owner, "_imported_sources", lambda root: {"tnfr": "src/tnfr/__init__.py"}
    )
    return SimpleNamespace(output=tmp_path / "evidence", source=source, calls=[])


@pytest.mark.parametrize("admitted", [True, False])
def test_driver_records_one_no_argument_assessment_after_attempt(
    harness, monkeypatch, admitted
):
    def assess():
        attempt = _read(harness.output / "attempt.json")
        assert attempt["parameters"] == {}
        assert attempt["evaluation_status_at_start"] == "not_evaluated"
        assert attempt["maximum_assessments"] == 1
        assert attempt["declaration"] == (
            "theory/nodal/ORGANIZED_INTERFACE_FACTORIZATION.md"
        )
        assert attempt["evaluator"] == "benchmarks/organized_interface_factorization.py"
        harness.calls.append(1)
        return _FakeReport(
            admitted, unresolved_conditions=() if admitted else ("premise",)
        )

    monkeypatch.setattr(driver, "_assess", assess)
    assert driver.main(["--output", str(harness.output)]) == 0
    result = _read(harness.output / "result.json")
    assert len(harness.calls) == result["assessment_count"] == 1
    assert result["verdict"] == (
        "finite_scalar_factorization_excluded"
        if admitted
        else "specified_factorization_test_unavailable"
    )
    assert result["report"]["determinant_bounds"] == {"lo": Q(1, 8), "hi": Q(1, 4)}
    assert result["report"]["unresolved_conditions"] == (
        () if admitted else ("premise",)
    )
    assert result["attempt"] == file_receipt(
        harness.output / "attempt.json", label="attempt.json"
    )
    before = {path.name: path.read_bytes() for path in harness.output.iterdir()}
    with pytest.raises(FileExistsError):
        driver.main(["--output", str(harness.output)])
    assert len(harness.calls) == 1
    assert before == {path.name: path.read_bytes() for path in harness.output.iterdir()}


def test_failed_assessment_keeps_attempt_and_prevents_retry(harness, monkeypatch):
    def failed():
        harness.calls.append(1)
        raise ArithmeticError("fixture failure")

    monkeypatch.setattr(driver, "_assess", failed)
    with pytest.raises(ArithmeticError, match="fixture failure"):
        driver.main(["--output", str(harness.output)])
    assert not (harness.output / "result.json").exists()
    failure = _read(harness.output / "incomplete.json")
    assert failure["exception"] == "ArithmeticError"
    assert failure["report_if_serialized"] is None
    with pytest.raises(FileExistsError):
        driver.main(["--output", str(harness.output)])
    assert len(harness.calls) == 1


def test_source_change_preserves_response_without_success(harness, monkeypatch):
    def changed():
        harness.source["source_revision"] = "changed-fixture-revision"
        return _FakeReport(True)

    monkeypatch.setattr(driver, "_assess", changed)
    with pytest.raises(RuntimeError, match="source changed"):
        driver.main(["--output", str(harness.output)])
    assert not (harness.output / "result.json").exists()
    failure = _read(harness.output / "incomplete.json")
    assert failure["source_revision"] == "fixture-revision"
    assert failure["report_if_serialized"]["admitted"]


@pytest.mark.parametrize("invalid", [1, "yes", None])
def test_truthy_metadata_cannot_select_scientific_verdict(
    harness, monkeypatch, invalid
):
    monkeypatch.setattr(driver, "_assess", lambda: _FakeReport(invalid))
    with pytest.raises(ValueError, match="ordinary Boolean"):
        driver.main(["--output", str(harness.output)])
    assert not (harness.output / "result.json").exists()
    assert (harness.output / "incomplete.json").exists()


@pytest.mark.parametrize("target", ["attempt.json", "result.json"])
def test_partial_write_prevents_repeat(harness, monkeypatch, target):
    real_write = owner.write_json_once

    def partial(path, payload):
        if path.name == target:
            path.write_text("partial", encoding="utf-8")
            raise OSError("fixture interrupted write")
        real_write(path, payload)

    def assess():
        harness.calls.append(1)
        return _FakeReport(True)

    monkeypatch.setattr(owner, "write_json_once", partial)
    monkeypatch.setattr(driver, "_assess", assess)
    with pytest.raises(OSError, match="interrupted"):
        driver.main(["--output", str(harness.output)])
    assert len(harness.calls) == int(target == "result.json")
    assert (harness.output / target).read_text(encoding="utf-8") == "partial"
    with pytest.raises(FileExistsError):
        driver.main(["--output", str(harness.output)])
    assert len(harness.calls) == int(target == "result.json")


def test_help_and_parameter_override_do_not_evaluate(harness):
    with pytest.raises(SystemExit) as stopped:
        driver.main(["--help"])
    assert stopped.value.code == 0
    with pytest.raises(SystemExit) as stopped:
        driver.main(["--output", str(harness.output), "--horizon", "1"])
    assert stopped.value.code == 2
    assert not harness.output.exists()


def test_working_directory_is_declared(harness, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="repository root"):
        driver.main(["--output", str(harness.output)])
    assert not harness.output.exists()


def test_dirty_source_rejects_before_inventory(monkeypatch):
    monkeypatch.setattr(
        owner,
        "current_git_source_provenance",
        lambda *args: ("revision", True, "digest"),
    )
    monkeypatch.setattr(
        owner.subprocess,
        "check_output",
        lambda *args, **kw: pytest.fail("queried dirty source"),
    )
    with pytest.raises(RuntimeError, match="commit the declaration"):
        owner._source_snapshot(driver.ROOT, "declaration.md", "benchmarks/evaluator.py")


def test_shared_source_inventory_and_committed_caller_are_bound(monkeypatch):
    calls = []

    def provenance(root, scope):
        calls.append((root, scope))
        return "revision", False, None

    listing = (
        b"100644 blob abc\tsrc/tnfr/research/shared.py\0"
        b"100644 blob def\tdeclaration.md\0"
        b"100644 blob ghi\tbenchmarks/evaluator.py\0"
    )
    monkeypatch.setattr(owner, "current_git_source_provenance", provenance)
    monkeypatch.setattr(
        owner.subprocess,
        "check_output",
        lambda args, **kw: listing if args[1] == "ls-tree" else b"tree\n",
    )
    source = owner._source_snapshot(
        driver.ROOT, "declaration.md", "benchmarks/evaluator.py"
    )
    assert calls == [
        (
            driver.ROOT,
            ("src/tnfr", "benchmarks/evaluator.py", "declaration.md", "pyproject.toml"),
        )
    ]
    assert source["source_inventory_sha256"] == sha256_bytes(listing)
    assert source["source_inventory"]["src/tnfr/research/shared.py"]["blob"] == "abc"
    assert source["source_revision"] == "revision" and source["source_tree"] == "tree"


@pytest.mark.parametrize("absent", ["declaration.md", "benchmarks/evaluator.py"])
def test_missing_committed_declaration_or_evaluator_rejects(monkeypatch, absent):
    monkeypatch.setattr(
        owner, "current_git_source_provenance", lambda *args: ("revision", False, None)
    )
    present = (
        "benchmarks/evaluator.py" if absent == "declaration.md" else "declaration.md"
    )
    listing = f"100644 blob abc\t{present}\0".encode()
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *args, **kw: listing)
    with pytest.raises(RuntimeError, match="must be committed"):
        owner._source_snapshot(driver.ROOT, "declaration.md", "benchmarks/evaluator.py")


def test_foreign_imported_source_rejects(monkeypatch, tmp_path):
    monkeypatch.setitem(
        owner.sys.modules,
        "tnfr.fixture_foreign_module",
        SimpleNamespace(__file__=str(tmp_path / "foreign.py")),
    )
    with pytest.raises(RuntimeError, match="outside this tree"):
        owner._imported_sources(driver.ROOT)

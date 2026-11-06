"""Exclusive evidence lifecycle with a fake assessor, never the new corner."""

from dataclasses import dataclass
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from benchmarks import organized_joint_resources as owner
from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I as Interval
from tnfr.research.artifact_io import decode_exact_tree, file_receipt, sha256_bytes
from tnfr.utils.io import json_loads


@dataclass(frozen=True)
class _FakeReport:
    admitted: bool
    retained_exact_value: Q = Q(1, 10**70)
    retained_interval: Interval = Interval(-1, 2)


def _read(path):
    return decode_exact_tree(json_loads(path.read_text(encoding="utf-8")))


@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("evaluator wiring test called the scientific corner assessor")

    monkeypatch.setattr(owner, "_assess_organized_joint_resources", forbidden)
    with forbid_sine_regeneration():
        yield


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.chdir(owner.ROOT)
    source = {"source_revision": "fixture-revision", "source_tree": "fixture-tree"}
    monkeypatch.setattr(owner, "_source_snapshot", lambda: dict(source))
    monkeypatch.setattr(
        owner, "_imported_sources", lambda: {"tnfr": "src/tnfr/__init__.py"}
    )
    return SimpleNamespace(output=tmp_path / "evidence", source=source, calls=[])


@pytest.mark.parametrize("admitted", [True, False])
def test_attempt_precedes_one_exact_corner_and_result(harness, monkeypatch, admitted):
    def assess(**kwargs):
        attempt = _read(harness.output / "attempt.json")
        assert attempt["parameters"] == kwargs
        assert attempt["evaluation_status_at_start"] == "not_evaluated"
        assert attempt["maximum_assessments"] == 1
        assert kwargs == dict(
            form_error_bounds=(Q(1, 10**6),) * 6,
            phase_error_bounds=(Q(1, 10**6),) * 6,
            quarter_dwell=Q(8 * 10**10),
            recovered_radius=Q(1, 10**8),
            decay_power=80,
            sequential_readout_error=Q(1, 10**14),
            contact_readout_error=Q(1, 10**9),
        )
        harness.calls.append(kwargs)
        return _FakeReport(admitted)

    monkeypatch.setattr(owner, "_assess_organized_joint_resources", assess)
    assert owner.main(["--output", str(harness.output)]) == 0
    result = _read(harness.output / "result.json")
    assert len(harness.calls) == result["assessment_count"] == 1
    assert result["source_revision"] == harness.source["source_revision"]
    assert result["verdict"] == (
        "joint_resource_box_admitted"
        if admitted
        else "specified_joint_resource_box_unavailable"
    )
    assert result["report"]["retained_exact_value"] == Q(1, 10**70)
    assert result["report"]["retained_interval"] == {"lo": Q(-1), "hi": Q(2)}
    assert result["attempt"] == file_receipt(
        harness.output / "attempt.json", label="attempt.json"
    )
    before = {path.name: path.read_bytes() for path in harness.output.iterdir()}
    with pytest.raises(FileExistsError, match="not empty"):
        owner.main(["--output", str(harness.output)])
    assert len(harness.calls) == 1
    assert before == {path.name: path.read_bytes() for path in harness.output.iterdir()}


def test_failed_assessment_keeps_attempt_and_blocks_retry(harness, monkeypatch):
    def failed(**kwargs):
        harness.calls.append(kwargs)
        raise ArithmeticError("deliberate fixture failure")

    monkeypatch.setattr(owner, "_assess_organized_joint_resources", failed)
    with pytest.raises(ArithmeticError, match="fixture failure"):
        owner.main(["--output", str(harness.output)])
    assert len(harness.calls) == 1
    assert not (harness.output / "result.json").exists()
    failure = _read(harness.output / "incomplete.json")
    assert failure["exception"] == "ArithmeticError"
    assert failure["report_if_serialized"] is None
    with pytest.raises(FileExistsError):
        owner.main(["--output", str(harness.output)])
    assert len(harness.calls) == 1


def test_source_change_retains_evaluated_report_without_success(harness, monkeypatch):
    def changed(**kwargs):
        harness.source["source_revision"] = "changed-fixture-revision"
        return _FakeReport(True)

    monkeypatch.setattr(owner, "_assess_organized_joint_resources", changed)
    with pytest.raises(RuntimeError, match="source changed"):
        owner.main(["--output", str(harness.output)])
    assert not (harness.output / "result.json").exists()
    failure = _read(harness.output / "incomplete.json")
    assert failure["source_revision"] == "fixture-revision"
    assert failure["report_if_serialized"]["admitted"]


def test_partial_attempt_write_never_reaches_assessment(harness, monkeypatch):
    def partial(path, payload):
        path.write_text("partial", encoding="utf-8")
        raise OSError("fixture interrupted write")

    monkeypatch.setattr(owner, "write_json_once", partial)
    with pytest.raises(OSError, match="interrupted"):
        owner.main(["--output", str(harness.output)])
    assert (harness.output / "attempt.json").read_text(encoding="utf-8") == "partial"
    with pytest.raises(FileExistsError):
        owner.main(["--output", str(harness.output)])


def test_help_and_unknown_configuration_never_assess(harness):
    with pytest.raises(SystemExit) as stopped:
        owner.main(["--help"])
    assert stopped.value.code == 0
    with pytest.raises(SystemExit) as stopped:
        owner.main(["--output", str(harness.output), "--decay-power", "81"])
    assert stopped.value.code == 2
    assert not harness.output.exists()


def test_runtime_must_use_declared_repository(harness, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="repository root"):
        owner.main(["--output", str(harness.output)])
    assert not harness.output.exists()


def test_dirty_source_rejects_before_inventory(monkeypatch):
    monkeypatch.setattr(
        owner,
        "current_git_source_provenance",
        lambda *args: ("revision", True, "digest"),
    )
    monkeypatch.setattr(
        owner, "_git", lambda *args: pytest.fail("dirty source queried")
    )
    with pytest.raises(RuntimeError, match="commit the declaration"):
        owner._source_snapshot()


def test_source_inventory_binds_shared_dependencies_and_declaration(monkeypatch):
    scope_calls = []

    def provenance(root, scope):
        scope_calls.append((root, scope))
        return "revision", False, None

    listing = (
        b"100644 blob abc\tsrc/tnfr/shared_dependency.py\0"
        + b"100644 blob def\ttheory/nodal/ORGANIZED_JOINT_RESOURCES.md\0"
    )
    monkeypatch.setattr(owner, "current_git_source_provenance", provenance)
    monkeypatch.setattr(
        owner, "_git", lambda *args: listing if args[0] == "ls-tree" else b"tree\n"
    )
    source = owner._source_snapshot()
    assert scope_calls == [(owner.ROOT, owner._SOURCE_SCOPE)]
    assert "src/tnfr" in source["source_scope"]
    assert source["source_inventory"]["src/tnfr/shared_dependency.py"] == {
        "mode": "100644",
        "blob": "abc",
    }
    assert source["source_inventory_sha256"] == sha256_bytes(listing)
    assert source["source_revision"] == "revision" and source["source_tree"] == "tree"


def test_uncommitted_declaration_rejects(monkeypatch):
    monkeypatch.setattr(
        owner, "current_git_source_provenance", lambda *args: ("revision", False, None)
    )
    monkeypatch.setattr(
        owner, "_git", lambda *args: b"100644 blob abc\tsrc/tnfr/a.py\0"
    )
    with pytest.raises(RuntimeError, match="declaration is not committed"):
        owner._source_snapshot()


def test_foreign_imported_source_rejects(monkeypatch, tmp_path):
    monkeypatch.setitem(
        owner.sys.modules,
        "tnfr.fixture_foreign_module",
        SimpleNamespace(__file__=str(tmp_path / "foreign.py")),
    )
    with pytest.raises(RuntimeError, match="outside this tree"):
        owner._imported_sources()

"""Frozen response plumbing; no control evaluates the reserved trajectory."""

import hashlib
import zipfile
from copy import deepcopy
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from benchmarks import relational_seeded_response as owner
from tnfr.utils.io import json_loads


@pytest.fixture
def plumbing(monkeypatch):
    files = {"fixture.py": b"# Protocol plumbing only.\n"}
    declaration = {"schema": "fixture.protocol", "reserved": "unchanged"}
    monkeypatch.setattr(owner, "_source_files", lambda: dict(files))
    monkeypatch.setattr(
        owner, "prepare_relational_seeded_response", lambda: deepcopy(declaration)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("protocol plumbing must never evaluate the reserved trajectory")

    monkeypatch.setattr(owner, "evaluate_relational_seeded_response", forbidden)
    return files, declaration


def _prepare(tmp_path):
    output = tmp_path / "response.json"
    arguments = ["--output", str(output)]
    assert owner.main(["--prepare", *arguments]) == 0
    return output, arguments


@pytest.mark.parametrize("passed", (False, True))
def test_separate_preparation_and_immutable_response(
    plumbing, monkeypatch, tmp_path, passed
):
    files, declaration = plumbing
    output, arguments = _prepare(tmp_path)
    frozen, archive = output.with_suffix(".protocol.json"), output.with_suffix(
        ".source.zip"
    )
    assert not output.exists()
    original = frozen.read_bytes(), archive.read_bytes()
    protocol = json_loads(original[0])
    assert protocol["protocol"] == declaration
    assert "correction" not in protocol
    assert "mathematical pi" in protocol["runtime"]["precision"]
    assert "128" in protocol["runtime"]["precision"]
    assert "no RNG" in protocol["runtime"]["randomness"]
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(files)
        for name, content in files.items():
            assert bundle.read(name) == content
            assert (
                protocol["source_sha256"][name] == hashlib.sha256(content).hexdigest()
            )
    with pytest.raises(FileExistsError):
        owner.main(["--prepare", *arguments])
    calls = []

    def response(received):
        calls.append(deepcopy(received))
        return {"passed": passed, "evidence": "synthetic wiring control"}

    monkeypatch.setattr(owner, "evaluate_relational_seeded_response", response)
    assert owner.main(arguments) == (0 if passed else 1)
    saved = output.read_bytes()
    record = json_loads(saved)
    assert calls == [declaration]
    assert record["passed"] is passed
    assert record["response"]["evidence"] == "synthetic wiring control"
    assert record["evaluation_error"] is None
    assert record["protocol"] == protocol
    assert record["protocol_sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert record["source_archive_sha256"] == hashlib.sha256(original[1]).hexdigest()
    for flags in (arguments, ["--prepare", *arguments]):
        with pytest.raises(FileExistsError):
            owner.main(flags)
    assert calls == [declaration]
    assert output.read_bytes() == saved
    assert (frozen.read_bytes(), archive.read_bytes()) == original


@pytest.mark.parametrize("change", ("protocol", "source", "runtime"))
def test_changed_frozen_inputs_reject_before_evaluation(
    plumbing, monkeypatch, tmp_path, change
):
    files, _ = plumbing
    output, arguments = _prepare(tmp_path)
    if change == "source":
        files["fixture.py"] = b"changed source\n"
    elif change == "runtime":
        monkeypatch.setattr(owner.platform, "python_version", lambda: "different")
    else:
        frozen = output.with_suffix(".protocol.json")
        payload = json_loads(frozen.read_bytes())
        payload["protocol"]["reserved"] = "changed"
        frozen.write_text(owner._encoded(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol/source/runtime mismatch"):
        owner.main(arguments)
    assert not output.exists()


@pytest.mark.parametrize("change", ("content", "inventory", "compression", "duplicate"))
def test_archive_rejects_changed_members_before_evaluation(plumbing, tmp_path, change):
    files, _ = plumbing
    output, arguments = _prepare(tmp_path)
    archive = output.with_suffix(".source.zip")
    compression = zipfile.ZIP_BZIP2 if change == "compression" else zipfile.ZIP_STORED
    with zipfile.ZipFile(archive, "w", compression=compression) as bundle:
        bundle.writestr(
            "other.py" if change == "inventory" else "fixture.py",
            b"changed member" if change == "content" else files["fixture.py"],
        )
        if change == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                bundle.writestr("fixture.py", files["fixture.py"])
    with pytest.raises(ValueError, match="archive|archived"):
        owner.main(arguments)
    assert not output.exists()


@pytest.mark.parametrize(
    "failure", ("exception", "source", "runtime", "frozen", "verdict", "projection")
)
def test_failed_evaluation_retains_available_evidence_once(
    plumbing, monkeypatch, tmp_path, failure
):
    files, _ = plumbing
    output, arguments = _prepare(tmp_path)

    def response(protocol):
        if failure == "exception":
            raise ValueError("injected evaluation failure")
        if failure == "source":
            files["fixture.py"] = b"changed while evaluating\n"
        elif failure == "runtime":
            monkeypatch.setattr(owner.platform, "python_version", lambda: "changed")
        elif failure == "frozen":
            with output.with_suffix(".protocol.json").open("ab") as stream:
                stream.write(b" ")
        elif failure == "projection":
            return {"passed": True, "unsupported": object()}
        return {"passed": 1 if failure == "verdict" else True, "partial": "retained"}

    monkeypatch.setattr(owner, "evaluate_relational_seeded_response", response)
    assert owner.main(arguments) == 1
    saved = output.read_bytes()
    record = json_loads(saved)
    assert record["passed"] is False
    assert record["protocol_sha256"] and record["source_archive_sha256"]
    assert record["evaluation_error"] is not None
    if failure in ("exception", "projection"):
        assert record["response"] is None
    else:
        assert record["response"]["partial"] == "retained"
    with pytest.raises(FileExistsError):
        owner.main(arguments)
    assert output.read_bytes() == saved


def test_foreign_loaded_source_rejects_preparation(plumbing, monkeypatch, tmp_path):
    monkeypatch.setitem(
        owner.sys.modules,
        "tnfr.foreign_seeded_control",
        SimpleNamespace(__file__=str(tmp_path / "foreign.py")),
    )
    output = tmp_path / "response.json"
    with pytest.raises(ValueError, match="outside the declared checkout"):
        owner.main(["--prepare", "--output", str(output)])
    assert not output.exists()
    assert not output.with_suffix(".protocol.json").exists()
    assert not output.with_suffix(".source.zip").exists()


def test_duplicate_protocol_keys_reject_before_evaluation(plumbing, tmp_path):
    output, arguments = _prepare(tmp_path)
    output.with_suffix(".protocol.json").write_text(
        '{"schema":"one","schema":"two"}', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        owner.main(arguments)
    assert not output.exists()


def test_source_inventory_includes_nested_engine_and_producer_bytes(
    monkeypatch, tmp_path
):
    expected = {
        "src/tnfr/alpha.py": b"# alpha\n",
        "src/tnfr/nested/beta.py": b"# beta\n",
        "benchmarks/relational_seeded_response.py": b"# producer\n",
        "pyproject.toml": b"# configuration\n",
    }
    for name, content in expected.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "src/tnfr/ignored.bin").write_bytes(b"outside the declared scope")
    monkeypatch.setattr(owner, "ROOT", tmp_path)
    monkeypatch.setattr(
        owner, "__file__", str(tmp_path / "benchmarks/relational_seeded_response.py")
    )
    assert owner._source_files() == expected


def _prepare_correction(tmp_path):
    prior = tmp_path / "response-v1.json"
    prior.write_text(
        owner._encoded(
            {
                "schema": "tnfr.relational-seeded-response-record.v1",
                "passed": False,
                "response": {"partial": "original inconclusive response"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "response-v2.json"
    arguments = ["--output", str(output), "--correction-of", str(prior)]
    assert owner.main(["--prepare", *arguments]) == 0
    return prior, output, arguments


def test_numerical_correction_retains_prior_hash_and_explicit_scope(
    plumbing, monkeypatch, tmp_path
):
    prior, output, arguments = _prepare_correction(tmp_path)
    original = prior.read_bytes()
    protocol = json_loads(output.with_suffix(".protocol.json").read_bytes())
    correction = protocol["correction"]
    assert correction["prior_record"] == prior.name
    assert correction["prior_record_sha256"] == hashlib.sha256(original).hexdigest()
    assert correction["prior_passed"] is False
    assert correction["kind"] == "separately_identified_numerical_correction"
    assert (
        correction["scope"] == "numerical_correction_not_independent_blind_replication"
    )
    monkeypatch.setattr(
        owner, "evaluate_relational_seeded_response", lambda protocol: {"passed": True}
    )
    assert owner.main(arguments) == 0
    record = json_loads(output.read_bytes())
    assert record["protocol"]["correction"] == correction
    assert record["passed"] is True
    assert prior.read_bytes() == original


@pytest.mark.parametrize("change", ("prior_bytes", "omitted_argument"))
def test_correction_inputs_reject_changes_before_evaluation(plumbing, tmp_path, change):
    prior, output, arguments = _prepare_correction(tmp_path)
    if change == "prior_bytes":
        with prior.open("ab") as stream:
            stream.write(b" ")
    else:
        arguments = ["--output", str(output)]
    with pytest.raises(ValueError, match="protocol/source/runtime mismatch"):
        owner.main(arguments)
    assert not output.exists()


def test_prior_change_during_correction_retains_failed_verdict(
    plumbing, monkeypatch, tmp_path
):
    prior, output, arguments = _prepare_correction(tmp_path)
    original_hash = hashlib.sha256(prior.read_bytes()).hexdigest()

    def response(protocol):
        with prior.open("ab") as stream:
            stream.write(b" ")
        return {"passed": True, "partial": "synthetic response"}

    monkeypatch.setattr(owner, "evaluate_relational_seeded_response", response)
    assert owner.main(arguments) == 1
    record = json_loads(output.read_bytes())
    assert record["passed"] is False
    assert record["response"]["passed"] is True
    assert "prior response changed" in record["evaluation_error"]["error"]
    assert record["protocol"]["correction"]["prior_record_sha256"] == original_hash
    with pytest.raises(FileExistsError):
        owner.main(arguments)


@pytest.mark.parametrize("invalid", ({"passed": False}, {"schema": "wrong"}, []))
def test_correction_requires_a_retained_response_record(plumbing, tmp_path, invalid):
    prior = tmp_path / "unrelated.json"
    prior.write_text(owner._encoded(invalid), encoding="utf-8")
    output = tmp_path / "response-v2.json"
    with pytest.raises(ValueError, match="retained seeded response"):
        owner.main(
            ["--prepare", "--output", str(output), "--correction-of", str(prior)]
        )
    assert not output.with_suffix(".protocol.json").exists()
    assert not output.with_suffix(".source.zip").exists()


def test_target_budget_study_binds_declaration_and_evaluation_keyword(
    plumbing, monkeypatch, tmp_path
):
    calls = []

    def prepare(*, study="short"):
        calls.append(("prepare", study))
        return {"schema": "fixture.protocol", "study": study}

    def evaluate(protocol, *, study="short"):
        calls.append(("evaluate", study))
        assert protocol["study"] == study == "target-budget"
        return {"passed": True, "study": study}

    monkeypatch.setattr(owner, "prepare_relational_seeded_response", prepare)
    output = tmp_path / "target-budget-v1.json"
    arguments = ["--study", "target-budget", "--output", str(output)]
    assert owner.main(["--prepare", *arguments]) == 0
    assert calls == [("prepare", "target-budget")]
    assert not output.exists()
    protocol = json_loads(output.with_suffix(".protocol.json").read_bytes())
    assert protocol["protocol"]["study"] == "target-budget"
    monkeypatch.setattr(owner, "evaluate_relational_seeded_response", evaluate)
    assert owner.main(arguments) == 0
    assert calls == [
        ("prepare", "target-budget"),
        ("prepare", "target-budget"),
        ("evaluate", "target-budget"),
    ]
    assert json_loads(output.read_bytes())["response"]["study"] == "target-budget"


@pytest.mark.parametrize("prepared_study", ("short", "target-budget"))
def test_changed_study_rejects_before_evaluation(
    plumbing, monkeypatch, tmp_path, prepared_study
):
    monkeypatch.setattr(
        owner,
        "prepare_relational_seeded_response",
        lambda *, study="short": {"schema": "fixture.protocol", "study": study},
    )
    output = tmp_path / "study.json"
    arguments = ["--output", str(output)]
    assert owner.main(["--prepare", "--study", prepared_study, *arguments]) == 0
    different = "target-budget" if prepared_study == "short" else "short"
    with pytest.raises(ValueError, match="protocol/source/runtime mismatch"):
        owner.main(["--study", different, *arguments])
    assert not output.exists()


@pytest.mark.parametrize("prepare", (False, True))
def test_target_budget_requires_explicit_output_before_any_preparation(
    plumbing, monkeypatch, tmp_path, prepare
):
    default = tmp_path / "untouched-default.json"
    monkeypatch.setattr(owner, "DEFAULT_OUTPUT", default)

    def forbidden(*args, **kwargs):
        pytest.fail("missing output must reject before preparing a declaration")

    monkeypatch.setattr(owner, "prepare_relational_seeded_response", forbidden)
    arguments = ["--study", "target-budget"]
    if prepare:
        arguments.insert(0, "--prepare")
    with pytest.raises(SystemExit) as error:
        owner.main(arguments)
    assert error.value.code == 2
    assert not tuple(tmp_path.iterdir())


def test_default_short_study_retains_default_output_and_no_keyword_calls(
    plumbing, monkeypatch, tmp_path
):
    output = tmp_path / "short-default.json"
    monkeypatch.setattr(owner, "DEFAULT_OUTPUT", output)
    # The shared fixture's no-keyword prepare stub must remain sufficient.
    assert owner.main(["--prepare"]) == 0
    monkeypatch.setattr(
        owner, "evaluate_relational_seeded_response", lambda protocol: {"passed": True}
    )
    assert owner.main([]) == 0
    assert json_loads(output.read_bytes())["passed"] is True


def _horizon_declaration(*, study="short", horizon=None):
    value = Q(1) if horizon is None else horizon
    return {
        "schema": "fixture.protocol",
        "study": study,
        "horizon": {"numerator": value.numerator, "denominator": value.denominator},
    }


def test_explicit_horizon_is_forwarded_as_exact_fraction_and_frozen(
    plumbing, monkeypatch, tmp_path
):
    calls = []

    def prepare(*, study, horizon):
        calls.append(("prepare", study, horizon))
        return _horizon_declaration(study=study, horizon=horizon)

    def evaluate(protocol, *, study, horizon):
        calls.append(("evaluate", study, horizon))
        assert horizon == Q(9, 8) and type(horizon) is Q
        assert protocol == _horizon_declaration(study=study, horizon=horizon)
        return {"passed": True}

    monkeypatch.setattr(owner, "prepare_relational_seeded_response", prepare)
    output = tmp_path / "target-budget-9over8.json"
    arguments = [
        "--study",
        "target-budget",
        "--horizon",
        "9/8",
        "--output",
        str(output),
    ]
    assert owner.main(["--prepare", *arguments]) == 0
    assert calls == [("prepare", "target-budget", Q(9, 8))]
    protocol = json_loads(output.with_suffix(".protocol.json").read_bytes())
    assert protocol["protocol"]["horizon"] == {"numerator": 9, "denominator": 8}
    monkeypatch.setattr(owner, "evaluate_relational_seeded_response", evaluate)
    assert owner.main(arguments) == 0
    assert calls[-1] == ("evaluate", "target-budget", Q(9, 8))


@pytest.mark.parametrize("evaluation_horizon", (None, "5/4"))
def test_changed_horizon_rejects_before_evaluation(
    plumbing, monkeypatch, tmp_path, evaluation_horizon
):
    monkeypatch.setattr(
        owner, "prepare_relational_seeded_response", _horizon_declaration
    )
    output = tmp_path / "target-budget.json"
    arguments = ["--study", "target-budget", "--output", str(output)]
    assert owner.main(["--prepare", *arguments, "--horizon", "9/8"]) == 0
    if evaluation_horizon is not None:
        arguments += ["--horizon", evaluation_horizon]
    with pytest.raises(ValueError, match="protocol/source/runtime mismatch"):
        owner.main(arguments)
    assert not output.exists()


def test_horizon_bounds_are_admitted_by_the_shared_preparation_owner(
    plumbing, monkeypatch, tmp_path
):
    received = []

    def reject(*, study, horizon):
        received.append((study, horizon))
        raise ValueError("shared owner rejected horizon")

    monkeypatch.setattr(owner, "prepare_relational_seeded_response", reject)
    for text in ("0", "-1/2", "5"):
        output = tmp_path / f"rejected-{len(received)}.json"
        with pytest.raises(ValueError, match="shared owner rejected"):
            owner.main(
                [
                    "--prepare",
                    "--study",
                    "target-budget",
                    f"--horizon={text}",
                    "--output",
                    str(output),
                ]
            )
        assert received[-1] == ("target-budget", Q(text))
        assert not output.with_suffix(".protocol.json").exists()
        assert not output.with_suffix(".source.zip").exists()


@pytest.mark.parametrize(
    "arguments",
    (
        ("--horizon", "9/8"),
        ("--study", "target-budget", "--horizon", "1/0"),
        ("--study", "target-budget", "--horizon", "nan"),
    ),
)
def test_invalid_horizon_syntax_or_short_override_rejects_before_preparation(
    plumbing, monkeypatch, tmp_path, arguments
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid horizon must reject before shared preparation")

    monkeypatch.setattr(owner, "prepare_relational_seeded_response", forbidden)
    with pytest.raises(SystemExit) as error:
        owner.main(["--prepare", *arguments, "--output", str(tmp_path / "unused.json")])
    assert error.value.code == 2
    assert not tuple(tmp_path.iterdir())

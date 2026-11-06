"""Frozen K3 response plumbing; no test creates a reserved scientific response."""

import hashlib
import zipfile
from copy import deepcopy
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from benchmarks import relational_capacity_sampling_response as owner
from tnfr.physics.relational_observations import (
    bound_relational_rate_contrast,
    bound_relational_rate_from_samples,
)
from tnfr.sdk import relational_report_to_dict
from tnfr.utils.io import json_loads


@pytest.fixture
def plumbing(monkeypatch):
    """Use a declaration stub and forbid the actual response evaluator."""
    files = {"fixture.py": b"# Protocol plumbing only.\n"}
    declaration = {"schema": "fixture.protocol", "report": {"reserved": "unchanged"}}
    typed = SimpleNamespace(to_dict=lambda: deepcopy(declaration))
    monkeypatch.setattr(owner, "_source_files", lambda: dict(files))
    monkeypatch.setattr(owner, "prepare_relational_capacity_response", lambda: typed)

    def forbidden(*args, **kwargs):
        pytest.fail("protocol plumbing must never evaluate the reserved response")

    monkeypatch.setattr(owner, "evaluate_relational_capacity_response", forbidden)
    return files, typed


def _prepare(tmp_path):
    output = tmp_path / "response.json"
    arguments = ["--output", str(output)]
    assert owner.main(["--prepare", *arguments]) == 0
    return output, arguments


def _fake_response(*, passed):
    payload = {
        "schema": "fixture.response",
        "report": {"completed": passed, "passed": passed, "cases": ["retained"]},
    }
    return SimpleNamespace(passed=passed, to_dict=lambda: deepcopy(payload))


def test_prepare_is_separate_and_negative_response_cannot_be_overwritten(
    plumbing, monkeypatch, tmp_path
):
    files, typed = plumbing
    output, arguments = _prepare(tmp_path)
    frozen, archive = output.with_suffix(".protocol.json"), output.with_suffix(
        ".source.zip"
    )
    assert not output.exists()
    original = frozen.read_bytes(), archive.read_bytes()
    protocol = json_loads(original[0])
    assert protocol["protocol"] == typed.to_dict()
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

    def negative(declaration):
        calls.append(declaration)
        return _fake_response(passed=False)

    monkeypatch.setattr(owner, "evaluate_relational_capacity_response", negative)
    assert owner.main(arguments) == 1
    saved = output.read_bytes()
    record = json_loads(saved)
    assert calls == [typed]
    assert record["passed"] is False
    assert record["response"]["report"]["cases"] == ["retained"]
    assert record["evaluation_error"] is None
    assert record["protocol"] == protocol
    assert record["protocol_sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert record["source_archive_sha256"] == hashlib.sha256(original[1]).hexdigest()
    for flags in (arguments, ["--prepare", *arguments]):
        with pytest.raises(FileExistsError):
            owner.main(flags)
    assert calls == [typed]
    assert output.read_bytes() == saved
    assert (frozen.read_bytes(), archive.read_bytes()) == original


@pytest.mark.parametrize("change", ("protocol", "source", "runtime"))
def test_changed_frozen_inputs_reject_before_any_response(
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
        payload["protocol"]["report"]["reserved"] = "changed"
        frozen.write_text(owner._encoded(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol/source/runtime mismatch"):
        owner.main(arguments)
    assert not output.exists()


@pytest.mark.parametrize("change", ("content", "inventory", "compression"))
def test_archive_rejects_changed_members_before_any_response(
    plumbing, tmp_path, change
):
    files, _ = plumbing
    output, arguments = _prepare(tmp_path)
    archive = output.with_suffix(".source.zip")
    compression = zipfile.ZIP_BZIP2 if change == "compression" else zipfile.ZIP_STORED
    with zipfile.ZipFile(archive, "w", compression=compression) as bundle:
        bundle.writestr(
            "other.py" if change == "inventory" else "fixture.py",
            b"changed member" if change == "content" else files["fixture.py"],
        )
    with pytest.raises(ValueError, match="archive"):
        owner.main(arguments)
    assert not output.exists()


@pytest.mark.parametrize("after_response", (False, True))
def test_failure_persists_provenance_and_any_returned_partial_evidence(
    plumbing, monkeypatch, tmp_path, after_response
):
    files, _ = plumbing
    output, arguments = _prepare(tmp_path)

    def failure(protocol):
        if not after_response:
            raise ValueError("injected pre-response failure")
        files["fixture.py"] = b"changed while evaluating\n"
        return _fake_response(passed=True)

    monkeypatch.setattr(owner, "evaluate_relational_capacity_response", failure)
    assert owner.main(arguments) == 1
    record = json_loads(output.read_bytes())
    assert record["passed"] is False
    assert record["protocol_sha256"] and record["source_archive_sha256"]
    assert record["evaluation_error"]["error_type"] == "ValueError"
    if after_response:
        assert "source changed" in record["evaluation_error"]["error"]
        assert record["response"]["report"]["cases"] == ["retained"]
        assert record["response"]["report"]["passed"] is True
    else:
        assert record["response"] is None
        assert record["evaluation_error"]["error"] == "injected pre-response failure"
    with pytest.raises(FileExistsError):
        owner.main(arguments)


def test_foreign_runtime_source_rejects_before_preparation(
    plumbing, monkeypatch, tmp_path
):
    monkeypatch.setitem(
        owner.sys.modules,
        "tnfr.foreign_protocol_control",
        SimpleNamespace(__file__=str(tmp_path / "foreign.py")),
    )
    output = tmp_path / "response.json"
    with pytest.raises(ValueError, match="outside the declared checkout"):
        owner.main(["--prepare", "--output", str(output)])
    assert not output.exists()
    assert not output.with_suffix(".protocol.json").exists()
    assert not output.with_suffix(".source.zip").exists()


def _exact(value):
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            return Q(value["numerator"], value["denominator"])
        return {key: _exact(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(map(_exact, value))
    return value


def _encloses(outer, inner):
    assert outer[0] <= inner[0] <= inner[1] <= outer[1]


def test_retained_record_reconstructs_samples_observations_and_original_verdict(
    monkeypatch,
):
    """Audit retained algebra/bytes only; this does not authenticate execution."""
    output = owner.DEFAULT_OUTPUT
    if not output.is_file():
        pytest.skip("retained response absent; never regenerate it in a test")

    def forbidden(*args, **kwargs):
        pytest.fail("a retained-record audit must not prepare or evaluate a response")

    monkeypatch.setattr(owner, "prepare_protocol", forbidden)
    monkeypatch.setattr(owner, "prepare_relational_capacity_response", forbidden)
    monkeypatch.setattr(owner, "evaluate_relational_capacity_response", forbidden)
    paths = (
        output,
        output.with_suffix(".protocol.json"),
        output.with_suffix(".source.zip"),
    )
    saved = tuple(path.read_bytes() for path in paths)
    record, frozen = json_loads(saved[0]), json_loads(saved[1])
    assert record["protocol"] == frozen
    assert record["protocol_sha256"] == hashlib.sha256(saved[1]).hexdigest()
    assert record["source_archive_sha256"] == hashlib.sha256(saved[2]).hexdigest()
    with zipfile.ZipFile(paths[2]) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert set(names) == set(frozen["source_sha256"])
        for name, digest in frozen["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest

    if record["response"] is None:
        assert record["evaluation_error"] is not None
        assert record["passed"] is False
        assert tuple(path.read_bytes() for path in paths) == saved
        return

    response = _exact(record["response"]["report"])
    protocol = _exact(frozen["protocol"]["report"])
    assert response["protocol"] == protocol
    admission = protocol["sampling"]
    times = (Q(0), admission["sample_step"], admission["horizon"])
    source_cases = admission["cases"]
    assert len(response["cases"]) <= len(source_cases)
    rates = {}
    for index, case in enumerate(response["cases"]):
        specification = source_cases[index]
        for key in ("law", "arm", "capacity"):
            assert case[key] == specification[key]
        coefficients = case["initial_coefficients"]
        seventh = case["remainder_coefficients"]
        if coefficients:
            assert len(coefficients) == 6
            assert all(len(row) == protocol["taylor_order"] + 1 for row in coefficients)
            assert tuple(row[0] for row in coefficients) == admission["initial_bounds"]
        assert len(case["samples"]) <= 3
        for ordinal, sample in enumerate(case["samples"]):
            time = sample["time"]
            assert time == times[ordinal]
            assert len(seventh) == 6
            for coordinate in range(6):
                # Independent power sum; do not call the producer's Horner helper.
                polynomial = tuple(
                    sum(
                        row[endpoint] * time**degree
                        for degree, row in enumerate(coefficients[coordinate])
                    )
                    for endpoint in (0, 1)
                )
                _encloses(sample["polynomial_bounds"][coordinate], polynomial)
                remainder = tuple(
                    value * time ** (protocol["taylor_order"] + 1)
                    for value in seventh[coordinate]
                )
                _encloses(sample["remainder_bounds"][coordinate], remainder)
                _encloses(
                    sample["state_bounds"][coordinate],
                    tuple(
                        sample["polynomial_bounds"][coordinate][side]
                        + sample["remainder_bounds"][coordinate][side]
                        for side in (0, 1)
                    ),
                )
            state = sample["state_bounds"]
            contrast = sample["phase_contrast_bounds"]
            _encloses(contrast, (state[3][0] - state[5][1], state[3][1] - state[5][0]))
            assert sample["midpoint"] == (contrast[0] + contrast[1]) / 2
            assert sample["radius"] == (contrast[1] - contrast[0]) / 2
            assert sample["admitted"] is (
                sample["radius"] <= admission["sample_error_bound"]
            )
            if not sample["admitted"]:
                assert ordinal == len(case["samples"]) - 1
                assert case["status"] == "stopped"
        if case["rate_observation"] is None:
            assert case["status"] == "stopped"
        else:
            assert len(case["samples"]) == 3 and all(
                sample["admitted"] for sample in case["samples"]
            )
            rate = bound_relational_rate_from_samples(
                tuple(sample["midpoint"] for sample in case["samples"]),
                sample_step=admission["sample_step"],
                sample_error_bound=admission["sample_error_bound"],
                third_derivative_bound=admission["third_derivative_bound"],
            )
            assert (
                _exact(relational_report_to_dict(rate)["report"])
                == case["rate_observation"]
            )
            rates[(case["law"], case["arm"])] = rate

    predictions = dict(protocol["predictions"])
    intervals = []
    for analysis in response["analyses"]:
        law = analysis["law"]
        if analysis["contrast"] is None:
            assert not analysis["passed"] and analysis["unavailable_reasons"]
            continue
        before, after = rates[(law, "before")], rates[(law, "after")]
        contrast = bound_relational_rate_contrast(
            before_bounds=before.rate_bounds, after_bounds=after.rate_bounds
        )
        assert (
            _exact(relational_report_to_dict(contrast)["report"])
            == analysis["contrast"]
        )
        observed = contrast.normalized_change_bounds
        assert observed is not None
        intervals.append(observed)
        corners = tuple(
            a / b - 1 for a in after.rate_bounds for b in before.rate_bounds
        )
        hull = min(corners), max(corners)
        inflation = hull[0] - observed[0], observed[1] - hull[1]
        assert analysis["exact_corner_hull"] == hull
        assert analysis["arithmetic_inflation"] == inflation
        assert analysis["baseline_lower_bound"] == before.rate_bounds[0]
        own = predictions[law]
        other = next(value for name, value in predictions.items() if name != law)
        checks = dict(
            resolved_baseline=before.rate_bounds[0] > 0,
            operational_rate_precision=200
            * max(before.rate_error_bound, after.rate_error_bound)
            < before.rate_bounds[0],
            arithmetic_inflation_budget=all(
                0 <= value <= protocol["arithmetic_allowance"] for value in inflation
            ),
            contains_own_prediction=observed[0] <= own[0] <= own[1] <= observed[1],
            excludes_other_prediction=observed[1] < other[0] or other[1] < observed[0],
            within_own_envelope=own[0] - protocol["envelope_radius"] <= observed[0]
            and observed[1] <= own[1] + protocol["envelope_radius"],
        )
        assert dict(analysis["checks"]) == checks
        assert analysis["passed"] is all(checks.values())
    complete = len(response["cases"]) == len(source_cases) and all(
        case["status"] == "completed" for case in response["cases"]
    )
    disjoint = len(intervals) == 2 and (
        intervals[0][1] < intervals[1][0] or intervals[1][1] < intervals[0][0]
    )
    assert response["completed"] is complete
    assert response["contrasts_disjoint"] is disjoint
    assert response["passed"] is (
        complete and disjoint and all(item["passed"] for item in response["analyses"])
    )
    assert record["passed"] is (
        response["passed"] and record["evaluation_error"] is None
    )
    assert tuple(path.read_bytes() for path in paths) == saved

"""Role separation and immutable evidence; no reserved trajectory is replayed."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from benchmarks import relational_sine_prior_forecast as owner
from tnfr.research.relational_sine_prior_forecast import (
    _prior_inference,
    prepare_sine_prior_forecast,
)
from tnfr.research.relational_sine_prior_source import prepare_sine_prior_source
from tnfr.utils.io import json_loads


def forbidden(*args, **kwargs):
    pytest.fail("reserved response or source role accessed prematurely")


@pytest.fixture
def plumbing(monkeypatch):
    files = {"fixture.py": b"# Information-flow fixture only.\n"}
    prior = {"earlier": "public evidence"}
    source = {"hidden": "source-role only"}
    monkeypatch.setattr(owner, "_source_files", lambda: dict(files))
    monkeypatch.setattr(
        owner, "_source_preparation", lambda: (deepcopy(prior), deepcopy(source))
    )
    monkeypatch.setattr(
        owner, "prepare_sine_prior_forecast", lambda p: {"prior": p, "fixed": "policy"}
    )
    monkeypatch.setattr(owner, "predict_sine_prior_forecast", forbidden)
    monkeypatch.setattr(owner, "_source_response", forbidden)
    return files, prior, source


def _prepare(tmp_path):
    output = tmp_path / "response.json"
    args = ["--output", str(output)]
    assert owner.main(["--prepare", *args]) == 0
    return output, args


def test_prior_joint_admission_without_future_access(monkeypatch):
    from tnfr.physics import relational_sine_forecast as flow

    monkeypatch.setattr(flow, "bound_sine_flow", forbidden)
    prior, _ = prepare_sine_prior_source()
    protocol = prepare_sine_prior_forecast(prior)
    capacity = _prior_inference(protocol["prior"])
    admission = flow.admit_sine_prior(capacity)
    assert admission.admitted
    assert all(
        row.contains(value)
        for row, value in zip(admission.initial_box, admission.joint_witness)
    )
    assert admission.initial_box[-1].width > 0
    assert admission.initial_box[-2].width > 0
    assert "source" not in protocol
    changed = deepcopy(prior)
    changed["hidden_capacity"] = 1
    with pytest.raises(ValueError, match="unexpected prior fields"):
        prepare_sine_prior_forecast(changed)


def test_three_stages_and_predictor_cannot_read_source(plumbing, monkeypatch, tmp_path):
    _, prior, source = plumbing
    output, args = _prepare(tmp_path)
    frozen = output.with_suffix(".protocol.json")
    archive = output.with_suffix(".source.zip")
    originals = frozen.read_bytes(), archive.read_bytes()
    source_path = output.with_suffix(".source-state.json")
    original_source = source_path.read_bytes()
    real_read = owner._read

    def read(path):
        if path == source_path:
            forbidden()
        return real_read(path)

    calls = []

    def predict(declaration):
        calls.append(deepcopy(declaration))
        assert declaration == {"prior": prior, "fixed": "policy"}
        return {"passed": True, "issued": "prior-only bounds"}

    monkeypatch.setattr(owner, "_source_preparation", forbidden)
    monkeypatch.setattr(owner, "_read", read)
    monkeypatch.setattr(owner, "predict_sine_prior_forecast", predict)
    assert owner.main(["--predict", *args]) == 0
    prediction = output.with_suffix(".prediction.json")
    issued = prediction.read_bytes()
    assert not output.exists()
    assert source_path.read_bytes() == original_source
    with pytest.raises(FileExistsError):
        owner.main(["--predict", *args])

    monkeypatch.setattr(owner, "_read", real_read)
    monkeypatch.setattr(
        owner, "_source_response", lambda received: (received, "reserved")
    )

    def assess(predicted, response):
        assert predicted == {"passed": True, "issued": "prior-only bounds"}
        assert response == (source, "reserved")
        return {"passed": True, "observed": "reserved once"}

    monkeypatch.setattr(owner, "assess_sine_prior_response", assess)
    assert owner.main(args) == 0
    record = json_loads(output.read_bytes())
    assert record["passed"] is True
    assert record["prediction_sha256"] == owner._digest(issued)
    assert len(calls) == 1
    assert prediction.read_bytes() == issued
    assert (frozen.read_bytes(), archive.read_bytes()) == originals
    for flags in (args, ["--prepare", *args], ["--predict", *args]):
        with pytest.raises(FileExistsError):
            owner.main(flags)


def test_response_requires_prior_prediction(plumbing, tmp_path):
    _, args = _prepare(tmp_path)
    with pytest.raises(FileNotFoundError):
        owner.main(args)


@pytest.mark.parametrize(
    "failure", ("negative", "exception", "source_change", "nonboolean", "projection")
)
def test_failed_prediction_is_retained_and_blocks_response(
    plumbing, monkeypatch, tmp_path, failure
):
    files, _, _ = plumbing
    output, args = _prepare(tmp_path)

    def predict(_):
        if failure == "exception":
            raise ValueError("injected prediction failure")
        if failure == "source_change":
            files["fixture.py"] = b"changed during prediction"
        if failure == "projection":
            return {"passed": True, "bad": object()}
        return {"passed": 1 if failure == "nonboolean" else failure != "negative"}

    monkeypatch.setattr(owner, "predict_sine_prior_forecast", predict)
    assert owner.main(["--predict", *args]) == 1
    path = output.with_suffix(".prediction.json")
    original = path.read_bytes()
    assert json_loads(original)["passed"] is False
    with pytest.raises(ValueError):
        owner.main(args)
    assert path.read_bytes() == original
    assert not output.exists()


def test_changed_sealed_source_rejects_before_response(plumbing, monkeypatch, tmp_path):
    output, args = _prepare(tmp_path)
    monkeypatch.setattr(
        owner, "predict_sine_prior_forecast", lambda _: {"passed": True}
    )
    assert owner.main(["--predict", *args]) == 0
    with output.with_suffix(".source-state.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="source preparation changed"):
        owner.main(args)
    assert not output.exists()


def test_failed_response_is_immutable(plumbing, monkeypatch, tmp_path):
    output, args = _prepare(tmp_path)
    monkeypatch.setattr(
        owner, "predict_sine_prior_forecast", lambda _: {"passed": True}
    )
    assert owner.main(["--predict", *args]) == 0

    def fail(_):
        raise ArithmeticError("injected enclosure failure")

    monkeypatch.setattr(owner, "_source_response", fail)
    assert owner.main(args) == 1
    saved = output.read_bytes()
    record = json_loads(saved)
    assert record["passed"] is False
    assert record["execution_error"]["error_type"] == "ArithmeticError"
    with pytest.raises(FileExistsError):
        owner.main(args)
    assert output.read_bytes() == saved


@pytest.mark.parametrize(
    "change",
    (
        {"observation_time": Q(1, 64)},
        {"end_time": Q(1, 32)},
        {"time_step": Q(1, 64)},
        {"order": 5},
        {"neighbors": ((1,), (0, 2), (1,))},
        {"visible_capacity": (Q(1), Q(1))},
        {"freeze_hidden": True},
    ),
)
def test_assessment_rejects_another_response_contract(change):
    from tnfr.dynamics.relational import RelationalExchangeModel
    from tnfr.mathematics._rational_interval import I
    from tnfr.physics.relational_sine_forecast import SineForecast
    from tnfr.research.relational_sine_prior_forecast import assess_sine_prior_response

    # Metadata-only fixture; it makes no claim to certify an evaluated flow.
    report = SineForecast(
        model=RelationalExchangeModel(1, phase_domain="regular"),
        neighbors=((2,), (2,), (0, 1)),
        visible_capacity=(Q(1), Q(2)),
        initial_box=(I(0),) * 7,
        observation_time=Q(0),
        end_time=Q(1, 16),
        time_step=Q(1, 128),
        order=6,
        steps=(),
        validated_end_time=Q(0),
        endpoint=(I(0),) * 7,
        failed_tube=None,
        status="unavailable",
        reasons=("metadata_fixture",),
    )
    with pytest.raises(
        ValueError, match="frozen law, support, clock or numerical budget"
    ):
        assess_sine_prior_response({}, replace(report, **change))

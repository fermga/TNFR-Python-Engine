"""Prospective record wiring; these tests never evaluate the reserved response."""

from pathlib import Path

import pytest

from benchmarks import return_path_geometry_response as producer
from tnfr.utils.io import json_loads


@pytest.fixture
def frozen_transport(monkeypatch, tmp_path):
    # A small archive exercises the transport contract, not source coverage.
    monkeypatch.setattr(producer, "_source_files", lambda: {"fixture.py": b"pass\n"})

    def forbidden(*args, **kwargs):
        pytest.fail("record wiring must not evaluate the reserved response")

    monkeypatch.setattr(producer, "evaluate_response", forbidden)
    return tmp_path / "response.json"


def test_geometry_and_prediction_are_retained_before_any_response(
    frozen_transport, monkeypatch
):
    output = frozen_transport
    assert producer.main(["prepare", "--output", str(output)]) == 0
    protocol_path = output.with_suffix(".protocol.json")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json_loads(protocol_bytes)
    assert protocol["observation"]["special_turn_bounds"] == [
        "1101/8000",
        "68813/500000",
    ]
    assert not output.exists()
    original = producer.predict_response

    def restricted_inputs(declaration, observation):
        assert "source_coefficient" not in declaration
        assert "reserved_observable" not in declaration
        return original(declaration, observation)

    monkeypatch.setattr(producer, "predict_response", restricted_inputs)
    assert producer.main(["predict", "--output", str(output)]) == 0
    prediction_path = output.with_suffix(".prediction.json")
    prediction_bytes = prediction_path.read_bytes()
    prediction = json_loads(prediction_bytes)
    assert prediction["report"]["coefficient_status"] == "bounded"
    assert prediction["protocol_sha256"] == producer._digest(protocol_bytes)
    assert not output.exists()
    with pytest.raises(FileExistsError, match="retain existing prediction"):
        producer.main(["predict", "--output", str(output)])
    assert prediction_path.read_bytes() == prediction_bytes
    assert protocol_path.read_bytes() == protocol_bytes


def test_source_changes_reject_before_prediction_or_response(
    frozen_transport, monkeypatch
):
    output = frozen_transport
    producer.main(["prepare", "--output", str(output)])
    monkeypatch.setattr(producer, "_source_files", lambda: {"fixture.py": b"changed\n"})
    for stage in ("predict", "evaluate"):
        with pytest.raises(ValueError, match="source changed"):
            producer.main([stage, "--output", str(output)])


def test_response_requires_prior_prediction(frozen_transport):
    output = frozen_transport
    producer.main(["prepare", "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        producer.main(["evaluate", "--output", str(output)])
    assert not Path(output).exists()


def test_edited_prediction_cannot_certify_response(frozen_transport):
    output = frozen_transport
    producer.main(["prepare", "--output", str(output)])
    producer.main(["predict", "--output", str(output)])
    prediction_path = output.with_suffix(".prediction.json")
    record = json_loads(prediction_path.read_bytes())
    record["report"]["response_acceleration_bounds"][0]["hi"]["numerator"] = 10**100
    prediction_path.write_text(producer.evidence._encoded(record), encoding="utf-8")
    with pytest.raises(ValueError, match="admitted geometric premises"):
        producer.main(["evaluate", "--output", str(output)])
    assert not output.exists()


def test_retained_records_bind_original_prediction_and_source_without_replay():
    output = (
        producer.ROOT / "docs/assets/return_path_geometry_response/response-v1.json"
    )
    protocol_bytes, protocol = producer._read(output.with_suffix(".protocol.json"))
    prediction_bytes, prediction = producer._read(
        output.with_suffix(".prediction.json")
    )
    _, response = producer._read(output)
    archive = output.with_suffix(".source.zip")
    producer._verify_archive(archive, protocol["source_sha256"])
    for record in (prediction, response):
        assert record["protocol_sha256"] == producer._digest(protocol_bytes)
        assert record["source_archive_sha256"] == producer._digest(archive.read_bytes())
    assert response["prediction_sha256"] == producer._digest(prediction_bytes)
    report = prediction["report"]
    values = response["response_acceleration_decimal"]
    contained = [
        producer._fraction(bound["lo"])
        <= producer.Q(value)
        <= producer._fraction(bound["hi"])
        for bound, value in zip(report["response_acceleration_bounds"], values)
    ]
    assert len(values) == len(contained) == len(protocol["declaration"]["nodes"])
    assert response["component_containment"] == contained
    assert response["passed"] is True and all(contained)

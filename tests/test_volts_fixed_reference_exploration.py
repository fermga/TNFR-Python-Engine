"""Synthetic-only checks of the fixed, within-acquisition Volts experiment."""

import hashlib
import importlib.util
import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tnfr.validation import NodalMeasurementRun
from tnfr.validation.nodal_prediction import _digest

_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/volts_fixed_reference_exploration.py"
)
_SPEC = importlib.util.spec_from_file_location("volts_exploration_test", _PATH)
BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BENCH
_SPEC.loader.exec_module(BENCH)


@pytest.fixture(autouse=True)
def forbid_real_data_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("these tests must never load the real Volts data")

    monkeypatch.setattr(BENCH, "load_volts", forbidden)


@pytest.fixture
def run():
    # Nonzero time origin detects using absolute time instead of elapsed time.
    times = tuple(10.0 + i * 0.02 for i in range(50))
    return NodalMeasurementRun(
        run_id="synthetic-trace",
        acquisition_id="one-synthetic-preparation",
        channel_ids=("Voltage",),
        timestamps=times,
        samples=(tuple(3 * math.exp(-1.25 * (t - times[0])) for t in times),),
        value_unit="V",
        time_unit="s",
    )


def prefix_of(run):
    return replace(run, timestamps=run.timestamps[:25], samples=(run.samples[0][:25],))


def issue(run):
    return BENCH.build_continuation(prefix_of(run), run.timestamps[24:])


def test_zero_reference_decay_predicts_same_run_without_factor_two(run):
    forecast = issue(run)
    retained_hash = _digest(forecast)
    result = BENCH.score_continuation(
        run,
        forecast,
        expected_forecast_hash=retained_hash,
    )
    assert forecast["capacity_display"] == pytest.approx((1.25, 1.25))
    assert forecast["predicted_voltage"] == pytest.approx(run.samples[0][24:])
    assert forecast["affine_ar1"] == pytest.approx(run.samples[0][24:])
    assert result["continuation_rows"] == 25
    assert result["nodal"]["max_absolute_error_volts"] < 1e-13
    assert result["persistence"]["rmse_volts"] > 0.1
    assert result["forecast_hash"] == retained_hash
    assert result["observation_hash"] == run.content_hash
    for record in (forecast, result):
        assert record["physical_status"] == "not_admitted"
        assert record["measurement_verdict"] == "not_assessed"
        assert record["split_scope"] == "within_single_acquisition"
    for field in (
        "sensor_uncertainty",
        "clock_uncertainty",
        "physical_acceptance_threshold",
    ):
        assert forecast[field] is None
    assert forecast["nominal_reference"] == 0


def test_changed_suffix_is_scored_without_refitting_or_physical_rejection(
    run, monkeypatch
):
    forecast = issue(run)
    retained_hash = _digest(forecast)
    changed_values = run.samples[0][:25] + tuple(v + 1.0 for v in run.samples[0][25:])
    changed = replace(run, samples=(changed_values,))
    assert issue(changed) == forecast

    def no_refit(*args, **kwargs):
        raise AssertionError("scoring must not estimate a new rate or baseline")

    monkeypatch.setattr(BENCH, "bound_fixed_reference_capacity", no_refit)
    monkeypatch.setattr(BENCH.np.linalg, "lstsq", no_refit)
    result = BENCH.score_continuation(
        changed,
        forecast,
        expected_forecast_hash=retained_hash,
    )
    assert result["nodal"]["signed_error_volts"] == pytest.approx((1.0,) * 25)
    assert result["nodal"]["rmse_volts"] == pytest.approx(1.0)
    assert result["measurement_verdict"] == "not_assessed"
    assert result["observation_hash"] != run.content_hash
    assert _digest(forecast) == retained_hash


@pytest.mark.parametrize(
    "field,value",
    [
        ("predicted_voltage", (0.0,) * 26),
        ("physical_status", "admitted"),
        ("sensor_uncertainty", 0),
        ("nominal_reference", 1),
    ],
)
def test_issued_hash_rejects_prediction_and_scope_tampering(run, field, value):
    forecast = issue(run)
    retained_hash = _digest(forecast)
    changed = deepcopy(forecast)
    changed[field] = value
    with pytest.raises(ValueError, match="issued continuation hash"):
        BENCH.score_continuation(run, changed, expected_forecast_hash=retained_hash)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: replace(r, run_id="renamed-trace"),
        lambda r: replace(r, acquisition_id="claimed-independent-preparation"),
        lambda r: replace(r, value_unit="mV"),
        lambda r: replace(r, time_unit="ms"),
        lambda r: replace(r, samples=((r.samples[0][0] + 1.0,) + r.samples[0][1:],)),
        lambda r: replace(r, timestamps=r.timestamps[:-1] + (r.timestamps[-1] + 0.01,)),
        lambda r: replace(
            r, timestamps=r.timestamps[:-1], samples=(r.samples[0][:-1],)
        ),
    ],
)
def test_scoring_binds_prefix_schedule_units_and_acquisition(run, change):
    forecast = issue(run)
    with pytest.raises(ValueError, match="observed trace"):
        BENCH.score_continuation(
            change(run), forecast, expected_forecast_hash=_digest(forecast)
        )


def test_json_roundtrip_preserves_issued_identity(run):
    forecast = issue(run)
    retained_hash = _digest(forecast)
    decoded = json.loads(json.dumps(forecast, allow_nan=False))
    assert _digest(decoded) == retained_hash
    assert (
        BENCH.score_continuation(
            run,
            decoded,
            expected_forecast_hash=retained_hash,
        )[
            "nodal"
        ]["max_absolute_error_volts"]
        < 1e-13
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda p: replace(
            p, timestamps=p.timestamps[:-1], samples=(p.samples[0][:-1],)
        ),
        lambda p: replace(p, channel_ids=("Other",)),
        lambda p: replace(p, value_unit="mV"),
        lambda p: replace(p, time_unit="ms"),
    ],
)
def test_prediction_rejects_wrong_prefix_domain(run, change):
    with pytest.raises(ValueError, match="25-row"):
        BENCH.build_continuation(change(prefix_of(run)), run.timestamps[24:])


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t[1:],
        lambda t: (t[0] - 0.001,) + t[1:],
        lambda t: (t[0], t[0]) + t[2:],
        lambda t: t[:-1] + (float("nan"),),
    ],
)
def test_prediction_rejects_invalid_future_schedule(run, change):
    with pytest.raises((ValueError, TypeError)):
        BENCH.build_continuation(prefix_of(run), change(run.timestamps[24:]))


@pytest.mark.parametrize("end", [3.0, 4.0, 0.0, -1.0])
def test_nondecay_or_unresolved_zero_crossing_abstains(run, end):
    prefix = prefix_of(run)
    changed = replace(prefix, samples=(prefix.samples[0][:-1] + (end,),))
    with pytest.raises(ValueError):
        BENCH.build_continuation(changed, run.timestamps[24:])


def test_nominal_model_uses_only_predeclared_endpoint_calibration(run):
    prefix = prefix_of(run)
    altered_values = (
        (prefix.samples[0][0],)
        + tuple(v + 0.05 for v in prefix.samples[0][1:-1])
        + (prefix.samples[0][-1],)
    )
    altered = replace(prefix, samples=(altered_values,))
    original = issue(run)
    changed = BENCH.build_continuation(altered, run.timestamps[24:])
    assert (
        changed["nominal_capacity_enclosure"] == original["nominal_capacity_enclosure"]
    )
    assert changed["predicted_voltage"] == original["predicted_voltage"]
    assert changed["prefix_hash"] != original["prefix_hash"]
    assert changed["affine_ar1"] != original["affine_ar1"]


def test_affine_control_does_not_fit_an_asymptote_into_nodal_model():
    values = tuple(2.0 + 3.0 * 0.9**i for i in range(50))
    shifted = NodalMeasurementRun(
        run_id="synthetic-offset",
        acquisition_id="synthetic-offset-preparation",
        channel_ids=("Voltage",),
        timestamps=tuple(i * 0.02 for i in range(50)),
        samples=(values,),
        value_unit="V",
        time_unit="s",
    )
    forecast = issue(shifted)
    assert forecast["nominal_reference"] == 0
    assert forecast["affine_ar1"] == pytest.approx(values[24:])
    result = BENCH.score_continuation(
        shifted,
        forecast,
        expected_forecast_hash=_digest(forecast),
    )
    assert result["affine_ar1"]["rmse_volts"] < 1e-12
    assert result["nodal"]["rmse_volts"] > 0.1
    assert result["measurement_verdict"] == "not_assessed"


def test_nonuniform_schedule_keeps_nodal_forecast_but_abstains_ar1(run):
    times = run.timestamps[24:-1] + (run.timestamps[-1] + 0.01,)
    forecast = BENCH.build_continuation(prefix_of(run), times)
    assert len(forecast["predicted_voltage"]) == 26
    assert forecast["affine_ar1"] is None
    assert "nonuniform" in forecast["affine_ar1_status"]


def test_optional_ar1_failure_does_not_erase_nodal_forecast(run, monkeypatch):
    def failure(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic decomposition failure")

    monkeypatch.setattr(BENCH.np.linalg, "lstsq", failure)
    forecast = issue(run)
    assert forecast["predicted_voltage"] == pytest.approx(run.samples[0][24:])
    assert forecast["affine_ar1"] is None
    assert any(
        word in forecast["affine_ar1_status"].lower()
        for word in ("fail", "unavailable", "error")
    )


@pytest.fixture
def specification(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_bytes(b"Synthetic fixed reference protocol; no real observations.\n")
    spec = deepcopy(BENCH.POLICY)
    spec["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    return path, protocol, hashlib.sha256(path.read_bytes()).hexdigest()


def test_specification_binds_exact_protocol_and_fixed_design(specification):
    path, protocol, expected_hash = specification
    actual = BENCH.verify_specification(path, protocol, expected_hash)
    assert actual["sensor_uncertainty"] is None
    assert actual["calibration_indices_inclusive"] == [0, 24]
    assert actual["continuation_indices_inclusive"] == [25, 49]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "0" * 64),
        ("expected_rows", 49),
        ("reference", 1),
        ("sensor_uncertainty", 0),
        ("clock_uncertainty", 0),
        ("physical_acceptance_threshold", 0.01),
        ("calibration_indices_inclusive", [0, 30]),
        ("continuation_indices_inclusive", [31, 49]),
        ("split_scope", "independent_preparations"),
        ("physical_status", "admitted"),
    ],
)
def test_rehashed_specification_still_cannot_change_fixed_policy(
    specification, field, value
):
    path, protocol, _ = specification
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec[field] = value
    path.write_text(json.dumps(spec), encoding="utf-8")
    new_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="fixed exploratory design"):
        BENCH.verify_specification(path, protocol, new_hash)


@pytest.mark.parametrize("target", ["spec", "protocol"])
def test_retained_byte_hashes_detect_even_nonsemantic_tampering(specification, target):
    path, protocol, expected_hash = specification
    altered = path if target == "spec" else protocol
    altered.write_bytes(altered.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        BENCH.verify_specification(path, protocol, expected_hash)


def test_synthetic_study_issues_before_scoring_and_admits_real_evidence(
    run, specification, tmp_path, monkeypatch
):
    spec_path, protocol_path, spec_hash = specification
    raw_path = tmp_path / "synthetic-only.rda"
    raw_path.write_bytes(b"Synthetic loader seam; this is not RData or Volts.\n")
    source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output = tmp_path / "synthetic-study"

    # The real loader's pinned-byte admission has its own synthetic tests.
    # This explicit seam supplies a synthetic observation without decoding data.
    monkeypatch.setattr(
        BENCH,
        "load_volts",
        lambda path: (
            run,
            {"source_sha256": source_hash, "rdata_version": "1.1.0"},
        ),
    )
    original_score = BENCH.score_continuation

    def require_issued_artifact(observed, forecast, *, expected_forecast_hash):
        saved = json.loads((output / "forecast.json").read_text(encoding="utf-8"))
        assert saved["content_hash"] == expected_forecast_hash
        assert _digest(saved["forecast"]) == expected_forecast_hash
        assert not (output / "result.json").exists()
        return original_score(
            observed, forecast, expected_forecast_hash=expected_forecast_hash
        )

    monkeypatch.setattr(BENCH, "score_continuation", require_issued_artifact)
    result = BENCH.execute_study(
        raw_path,
        spec_path=spec_path,
        protocol_path=protocol_path,
        expected_spec_hash=spec_hash,
        output=output,
    )
    evidence = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
    assert result["physical_status"] == "not_admitted"
    assert result["measurement_verdict"] == "not_assessed"
    assert evidence["observation_context"]["whole_file_decoded_before_issue"] is True
    assert evidence["observation_context"]["external_trusted_chronology"] is False
    assert evidence["observation_context"]["future_voltage_used_by_predictor"] is False
    assert evidence["numerical_context"] == {
        "sensor_error_bound": None,
        "clock_error_bound": None,
    }
    assert evidence["graph_context"]["support_status"].startswith("hypothetical")
    assert evidence["cost_context"]["automatic_refits"] == 0
    assert evidence["artifact_hashes"]
    for name, expected in evidence["artifact_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError, match="overwrite"):
        BENCH.execute_study(
            raw_path,
            spec_path=spec_path,
            protocol_path=protocol_path,
            expected_spec_hash=spec_hash,
            output=output,
        )


def test_changed_source_between_ingestion_and_copy_aborts_before_prediction(
    run, specification, tmp_path, monkeypatch
):
    spec_path, protocol_path, spec_hash = specification
    raw_path = tmp_path / "synthetic-only.rda"
    raw_path.write_bytes(b"Synthetic bytes before ingestion")
    initial_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output = tmp_path / "changed-input-study"

    def changed_during_ingestion(path):
        Path(path).write_bytes(b"Different synthetic bytes after ingestion")
        return run, {"source_sha256": initial_hash, "rdata_version": "1.1.0"}

    monkeypatch.setattr(BENCH, "load_volts", changed_during_ingestion)
    with pytest.raises(ValueError, match="retained input changed"):
        BENCH.execute_study(
            raw_path,
            spec_path=spec_path,
            protocol_path=protocol_path,
            expected_spec_hash=spec_hash,
            output=output,
        )
    assert not (output / "forecast.json").exists()
    assert not (output / "evidence.json").exists()

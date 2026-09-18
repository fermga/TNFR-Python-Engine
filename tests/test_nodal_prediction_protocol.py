"""Adversarial engineering checks, not laboratory TNFR validation."""

from dataclasses import FrozenInstanceError, asdict, replace
import json

import networkx as nx
import numpy as np
import pytest

from tnfr.validation import nodal_prediction as p


def _run(run_id="calibration", *, mean=1.0, amplitude=0.7, rate=0.4, times=None):
    times = np.arange(9, dtype=float) / 8 if times is None else np.asarray(times)
    # Independent two-node Euler eigenmode formula; synthetic fixture only.
    factors = np.r_[1.0, np.cumprod(1 - 2 * rate * np.diff(times))]
    return p.NodalMeasurementRun(
        run_id, ("left", "right"), times,
        [mean + amplitude*factors, mean - amplitude*factors],
        "fixture_units", "s", "acquisition-" + run_id,
    )


def _calibrate(run=None, graph=None, **kwargs):
    graph = nx.Graph([("left", "right")]) if graph is None else graph
    return p.calibrate_nodal_prediction(
        [_run() if run is None else run], graph=graph,
        offsets=kwargs.pop("offsets", (0.0, 0.0)), scales=(1.0, 1.0),
        structural_time_per_unit=kwargs.pop("structural_time_per_unit", 1.0),
        support_provenance="independently declared fixture edge",
        measurement_provenance="synthetic unit map, not instrument calibration", **kwargs)


def _forecast(calibration, run=None, **kwargs):
    run = _run("reserved", mean=2.0, amplitude=-0.5) if run is None else run
    return p.forecast_nodal_response(
        calibration, evaluation_run_id=run.run_id,
        evaluation_acquisition_id=run.acquisition_id,
        initial_measurement=[row[0] for row in run.samples], timestamps=run.timestamps,
        absolute_error_bound=kwargs.pop("absolute_error_bound", 1e-12),
        max_structural_step=kwargs.pop("max_structural_step", 0.125),
        max_steps=kwargs.pop("max_steps", 16), **kwargs)


def test_calibration_then_reserved_forecast_recovers_independent_eigenmode(monkeypatch):
    calls = []
    original = p.update_epi_via_nodal_equation

    def traced(graph, **kwargs):
        calls.append(kwargs["dt"])
        original(graph, **kwargs)

    monkeypatch.setattr(p, "update_epi_via_nodal_equation", traced)
    calibration = _calibrate()
    held_out = _run("reserved", mean=2.0, amplitude=-0.5)
    forecast = _forecast(calibration, held_out)
    result = p.score_nodal_forecast(
        forecast, calibration, held_out, expected_forecast_hash=forecast.content_hash)
    assert calibration.capacity == pytest.approx(0.4)
    assert len(calls) == forecast.steps_executed == 8
    assert np.asarray(forecast.epi).mean(axis=0) == pytest.approx(2.0)
    assert result.max_absolute_error < 1e-14
    assert result.meets_declared_error_bound
    assert result.physical_status == "not_admitted_by_this_score"
    assert result.persistence_mse > result.mean_squared_error


def test_graph_and_input_mutations_cannot_change_frozen_calibration():
    graph = nx.Graph([("left", "right")])
    offsets = [0.0, 0.0]
    calibration = _calibrate(graph=graph, offsets=offsets)
    before = _forecast(calibration)
    graph.remove_edge("left", "right")
    offsets[0] = 999
    assert _forecast(calibration) == before
    with pytest.raises(FrozenInstanceError):
        calibration.capacity = 99
    with pytest.raises(TypeError):
        calibration.conductance[0][1] = 99
    restored = p.FrozenNodalCalibration(**json.loads(json.dumps(asdict(calibration))))
    assert restored == calibration


def test_held_out_suffix_changes_score_but_never_the_issued_forecast(tmp_path):
    calibration = _calibrate()
    observation = _run("reserved", mean=2.0, amplitude=-0.5)
    forecast = _forecast(calibration, observation)
    path = p.write_nodal_forecast(forecast, tmp_path / "forecast.json")
    before = path.read_bytes()
    samples = np.asarray(observation.samples).copy()
    samples[:, 4:] += 10
    changed = replace(observation, samples=samples)
    assert not p.score_nodal_forecast(
        forecast, calibration, changed,
        expected_forecast_hash=forecast.content_hash).meets_declared_error_bound
    assert _forecast(calibration, changed) == forecast
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        p.write_nodal_forecast(forecast, path)


@pytest.mark.parametrize("rate,reason", [(-0.4, "negative_capacity"), (0, "inactive_capacity")])
def test_invalid_capacity_is_not_clipped_or_reinterpreted(rate, reason):
    with pytest.raises(p.NodalCalibrationError) as failure:
        _calibrate(_run(rate=rate))
    assert failure.value.reason == reason


def test_uniform_state_cannot_identify_capacity():
    with pytest.raises(p.NodalCalibrationError, match="capacity_unidentifiable"):
        _calibrate(_run(amplitude=0))


def test_irregular_timestamps_are_used_not_renumbered():
    times = (0.0, 0.125, 0.375, 0.5, 0.875)
    calibration = _calibrate(_run(times=times))
    observation = _run("reserved", mean=2, times=times)
    forecast = _forecast(calibration, observation, max_structural_step=0.5)
    assert calibration.capacity == pytest.approx(0.4)
    assert p.score_nodal_forecast(
        forecast, calibration, observation,
        expected_forecast_hash=forecast.content_hash).meets_declared_error_bound


@pytest.mark.parametrize("override", [
    {"run_id": "calibration"}, {"value_unit": "other"},
    {"time_unit": "ms"}, {"channel_ids": ("right", "left")},
    {"timestamps": tuple(np.arange(9)/4)},
])
def test_scoring_rejects_overlap_or_coordinate_mismatch(override):
    calibration = _calibrate()
    observation = _run("reserved", mean=2, amplitude=-0.5)
    forecast = _forecast(calibration, observation)
    with pytest.raises(ValueError):
        p.score_nodal_forecast(
            forecast, calibration, replace(observation, **override),
            expected_forecast_hash=forecast.content_hash)


def test_renaming_calibration_data_does_not_make_it_held_out():
    calibration = _calibrate()
    renamed = replace(_run(), run_id="renamed", acquisition_id="renamed-acquisition")
    forecast = _forecast(calibration, renamed)
    with pytest.raises(ValueError, match="overlap"):
        p.score_nodal_forecast(
            forecast, calibration, renamed, expected_forecast_hash=forecast.content_hash)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 0.0, -1.0])
def test_invalid_clock_bridge_is_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        _calibrate(structural_time_per_unit=value)


@pytest.mark.parametrize("times", [(0, 0, 1), (0, 2, 1), (0, float("nan"), 2)])
def test_invalid_measurement_time_is_not_silently_repaired(times):
    with pytest.raises(ValueError):
        p.NodalMeasurementRun("bad", ("a",), times, [(1, 2, 3)], "u", "s", "acq")


def test_missing_sample_is_not_deleted_with_its_timestamp():
    with pytest.raises(ValueError, match="finite"):
        replace(_run(), samples=np.full((2, 9), float("nan")))


def test_disconnected_or_reordered_support_is_not_admitted():
    with pytest.raises(ValueError, match="connected"):
        _calibrate(graph=nx.empty_graph(["left", "right"]))
    with pytest.raises(ValueError, match="order"):
        _calibrate(graph=nx.Graph([("right", "left")]))


def test_budget_and_fitted_model_mismatch_are_not_silent():
    calibration = _calibrate()
    with pytest.raises(ValueError, match="budget"):
        _forecast(calibration, max_steps=1)
    forecast = _forecast(calibration)
    with pytest.raises(ValueError, match="hash mismatch"):
        p.score_nodal_forecast(
            forecast, replace(calibration, capacity=0.9), _run("reserved"),
            expected_forecast_hash=forecast.content_hash)


def test_shared_clock_scaling_changes_capacity_not_forecast():
    ordinary = _calibrate()
    rescaled = _calibrate(structural_time_per_unit=2)
    assert rescaled.capacity == pytest.approx(ordinary.capacity / 2)
    assert np.asarray(_forecast(rescaled, max_structural_step=0.25).epi) == pytest.approx(
        np.asarray(_forecast(ordinary).epi))


def test_renamed_window_from_same_acquisition_is_not_independent():
    calibration = _calibrate()
    run = _run()
    cropped = replace(run, run_id="new-window", timestamps=run.timestamps[:5],
                      samples=tuple(row[:5] for row in run.samples))
    with pytest.raises(ValueError, match="acquisition overlaps"):
        _forecast(calibration, cropped)


def test_prediction_and_acceptance_budget_cannot_change_after_issue():
    calibration = _calibrate()
    observation = _run("reserved", mean=2, amplitude=-0.5)
    forecast = _forecast(calibration, observation)
    retained = forecast.content_hash
    replaced = replace(forecast, epi=observation.samples, absolute_error_bound=0)
    with pytest.raises(ValueError, match="issued forecast hash"):
        p.score_nodal_forecast(
            replaced, calibration, observation, expected_forecast_hash=retained)

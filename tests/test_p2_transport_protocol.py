"""Independent continuous-model controls for the P2 interval protocol.

All observations here are Decimal-generated fixtures, not laboratory evidence.
The nonzero declared sensor bound also covers their binary64 representation.
"""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json

import networkx as nx
import pytest

from tnfr.validation.nodal_prediction import (
    NodalMeasurementRun,
    calibrate_nodal_prediction,
)
from tnfr.validation import p2_transport as p


def _run(run_id="calibration", *, rate="0.4", mean="1", amplitude="0.7",
         times=(0.0, 0.125, 0.25, 0.5, 1.0), acquisition_id=None):
    with localcontext() as context:
        context.prec = 80
        rate, mean, amplitude = map(Decimal, (rate, mean, amplitude))
        shape = [amplitude * (-2 * rate * Decimal.from_float(time)).exp() for time in times]
        samples = (tuple(float(mean + value) for value in shape),
                   tuple(float(mean - value) for value in shape))
    return NodalMeasurementRun(
        run_id=run_id, channel_ids=("left", "right"), timestamps=times, samples=samples,
        value_unit="fixture_temperature_unit", time_unit="s",
        acquisition_id=acquisition_id or "acquisition-" + run_id,
    )


def _measurement(**changes):
    values = dict(offsets=(0.0, 0.0), scales=(1.0, 1.0),
                  epi_error=(1e-10, 1e-10), timestamp_error=0.0,
                  structural_time_per_unit=(1.0, 1.0),
                  provenance="independently declared synthetic bounds; no apparatus")
    values.update(changes)
    return p.P2MeasurementBounds(**values)


def _calibrate(runs=None, *, graph=None, measurement=None):
    return p.calibrate_p2_transport(
        [_run()] if runs is None else runs,
        graph=nx.Graph([("left", "right")]) if graph is None else graph,
        measurement=_measurement() if measurement is None else measurement,
        support_provenance="known synthetic passive P2 support",
    )


def _forecast(calibration, observation=None):
    observation = _run("reserved", mean="2", amplitude="-0.5") if observation is None else observation
    return p.forecast_p2_transport(
        calibration, evaluation_run_id=observation.run_id,
        evaluation_acquisition_id=observation.acquisition_id,
        initial_measurement=tuple(row[0] for row in observation.samples),
        timestamps=observation.timestamps,
    )


def _score(forecast, calibration, observation, expected=None):
    return p.score_p2_transport(
        forecast, calibration, observation,
        expected_forecast_hash=forecast.content_hash if expected is None else expected,
    )


def _endpoints(interval):
    return interval


def _contains(interval, value):
    low, high = _endpoints(interval)
    assert isinstance(low, Fraction) and isinstance(high, Fraction)
    assert low <= value <= high


@pytest.mark.parametrize("amplitude", ["0.7", "-0.7"])
def test_continuous_capacity_and_each_forecast_channel_are_enclosed(amplitude):
    calibration = _calibrate([_run(amplitude=amplitude)])
    _contains(calibration.capacity, Fraction(2, 5))
    low, high = _endpoints(calibration.capacity)
    assert high - low < Fraction(1, 1_000_000)
    reserved = _run("reserved", mean="2", amplitude="-0.5")
    forecast = _forecast(calibration, reserved)
    assert len(forecast.tube.samples) == len(reserved.timestamps)
    for index, sample in enumerate(forecast.tube.samples):
        for channel, interval in enumerate(sample.epi):
            _contains(interval, Fraction.from_float(reserved.samples[channel][index]))
        _contains(sample.mean, Fraction(2))
        represented_contrast = (Fraction.from_float(reserved.samples[0][index])
                                - Fraction.from_float(reserved.samples[1][index]))
        _contains(sample.contrast, represented_contrast)
    score = _score(forecast, calibration, reserved)
    assert score.status == "not_falsified_by_enclosures"
    assert score.physical_status == "not_admitted_by_this_score"


def test_continuous_fit_does_not_silently_reuse_biased_finite_euler_fit():
    run = _run(times=(0.0, 0.25, 0.5, 0.75, 1.0))
    graph = nx.Graph([("left", "right")])
    continuous = _calibrate([run], graph=graph)
    discrete = calibrate_nodal_prediction(
        [run], graph=graph, offsets=(0, 0), scales=(1, 1), structural_time_per_unit=1,
        support_provenance="known fixture graph", measurement_provenance="fixture mapping",
    )
    low, _ = _endpoints(continuous.capacity)
    assert Fraction.from_float(discrete.capacity) < low
    _contains(continuous.capacity, Fraction(2, 5))


def test_only_declared_calibration_endpoints_constrain_capacity():
    run = _run()
    changed = replace(run, samples=tuple(
        tuple(value if index in (0, len(row) - 1) else value + 10
              for index, value in enumerate(row)) for row in run.samples
    ))
    first, second = _calibrate([run]), _calibrate([changed])
    assert first.capacity == second.capacity
    assert first.content_hash != second.content_hash


def test_multiple_calibration_runs_intersect_outer_capacity_constraints():
    calibration = _calibrate([_run(), _run("calibration-2", mean="3", amplitude="-1.1")])
    _contains(calibration.capacity, Fraction(2, 5))
    with pytest.raises(ValueError):
        _calibrate([_run(), _run("inconsistent-rate", rate="0.8")])


def test_clock_uncertainty_is_propagated_instead_of_refitted():
    calibration = _calibrate(measurement=_measurement(structural_time_per_unit=(0.9, 1.1)))
    _contains(calibration.capacity, Fraction(4, 11))
    _contains(calibration.capacity, Fraction(4, 9))
    rescaled = _calibrate(measurement=_measurement(structural_time_per_unit=(2.0, 2.0)))
    _contains(rescaled.capacity, Fraction(1, 5))
    with pytest.raises(ValueError):
        _calibrate(measurement=_measurement(timestamp_error=0.6))


def test_sensor_scale_and_offset_map_precedes_the_epi_error_bound():
    measurement = _measurement(offsets=(10, -4), scales=(2, 3))

    def measured(run):
        return replace(run, samples=tuple(
            tuple(value * scale + offset for value in row)
            for row, scale, offset in zip(run.samples, (2, 3), (10, -4))
        ))

    calibration = _calibrate([measured(_run())], measurement=measurement)
    _contains(calibration.capacity, Fraction(2, 5))
    reserved = measured(_run("reserved", mean="2", amplitude="-0.5"))
    forecast = _forecast(calibration, reserved)
    assert _score(forecast, calibration, reserved).status == "not_falsified_by_enclosures"
    for sample in forecast.tube.samples:
        _contains(sample.mean, Fraction(2))


def test_timestamp_error_encloses_opposite_endpoint_clock_errors():
    true = _run()
    times = list(true.timestamps)
    times[0] += 2e-6
    times[-1] -= 2e-6
    measured = replace(true, timestamps=times)
    calibration = _calibrate([measured], measurement=_measurement(timestamp_error=3e-6))
    _contains(calibration.capacity, Fraction(2, 5))


def test_endpoint_fit_ignores_unused_early_time_but_forecast_requires_it_resolved():
    run = _run(times=(0.0, 0.001, 1.0))
    calibration = _calibrate([run], measurement=_measurement(timestamp_error=0.01))
    _contains(calibration.capacity, Fraction(2, 5))
    reserved = _run("reserved", mean="2", amplitude="-0.5", times=run.timestamps)
    with pytest.raises(ValueError, match="does not resolve duration"):
        _forecast(calibration, reserved)


def test_mean_drift_rejects_even_when_the_contrast_is_unchanged():
    calibration = _calibrate()
    observed = _run("reserved", mean="2", amplitude="-0.5")
    forecast = _forecast(calibration, observed)
    shifted = replace(observed, samples=tuple(
        tuple(value if index == 0 else value + 0.01 for index, value in enumerate(row))
        for row in observed.samples
    ))
    assert _score(forecast, calibration, shifted).status == "incompatible_with_declared_bounds"
    # The issued forecast depends on the initialization, not the unseen suffix.
    assert _forecast(calibration, shifted) == forecast


def test_graph_and_measurement_inputs_are_detached_and_frozen():
    graph = nx.Graph([("left", "right")])
    offsets = [0.0, 0.0]
    measurement = _measurement(offsets=offsets)
    calibration = _calibrate(graph=graph, measurement=measurement)
    before = _forecast(calibration)
    offsets[0] = 50
    graph.remove_edge("left", "right")
    assert _forecast(calibration) == before
    with pytest.raises(FrozenInstanceError):
        calibration.capacity = (Fraction(1), Fraction(2))


def test_forecast_hash_is_required_independently_of_the_payload():
    calibration = _calibrate()
    observed = _run("reserved", mean="2", amplitude="-0.5")
    forecast = _forecast(calibration, observed)
    with pytest.raises(ValueError):
        _score(forecast, calibration, observed, expected="sha256:" + "0" * 64)


def test_writer_preserves_exact_rational_endpoints_and_does_not_overwrite(tmp_path):
    calibration = _calibrate()
    forecast = _forecast(calibration)
    destination = p.write_p2_transport_forecast(forecast, tmp_path / "issued.json")
    before = destination.read_bytes()
    payload = json.loads(before)
    assert payload["content_hash"] == forecast.content_hash
    for original, encoded in zip(forecast.tube.samples, payload["forecast"]["tube"]["samples"]):
        for pair, encoded_pair in zip(original.epi, encoded["epi"]):
            restored = tuple(Fraction(int(item["numerator_hex"], 16), int(item["denominator_hex"], 16))
                             for item in encoded_pair)
            assert restored == pair
    with pytest.raises(FileExistsError):
        p.write_p2_transport_forecast(forecast, destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"run_id": "another-run"}, {"acquisition_id": "another-acquisition"},
    {"value_unit": "other"}, {"time_unit": "ms"},
    {"channel_ids": ("right", "left")},
    {"timestamps": (0.0, 0.25, 0.5, 1.0, 2.0)},
])
def test_reserved_identity_units_and_time_cannot_change_at_scoring(change):
    calibration = _calibrate()
    observed = _run("reserved", mean="2", amplitude="-0.5")
    forecast = _forecast(calibration, observed)
    with pytest.raises(ValueError):
        _score(forecast, calibration, replace(observed, **change))


def test_reserved_initialization_cannot_change_after_forecast():
    calibration = _calibrate()
    observed = _run("reserved", mean="2", amplitude="-0.5")
    forecast = _forecast(calibration, observed)
    changed = replace(observed, samples=tuple((row[0] + 1,) + row[1:] for row in observed.samples))
    with pytest.raises(ValueError):
        _score(forecast, calibration, changed)


def test_renamed_subwindow_of_calibration_retains_acquisition_overlap():
    calibration = _calibrate()
    original = _run()
    renamed = replace(original, run_id="renamed", timestamps=original.timestamps[1:],
                      samples=tuple(row[1:] for row in original.samples))
    with pytest.raises(ValueError):
        _forecast(calibration, renamed)


@pytest.mark.parametrize("change", [
    {"rate": "0"}, {"rate": "-0.4"}, {"amplitude": "0"},
])
def test_unresolved_constant_or_growing_contrast_is_not_admitted(change):
    with pytest.raises(ValueError):
        _calibrate([_run(**change)])


def test_contrast_sign_change_and_calibration_mean_shift_are_not_admitted():
    original = _run()
    swapped = replace(original, samples=(
        original.samples[0][:-1] + (original.samples[1][-1],),
        original.samples[1][:-1] + (original.samples[0][-1],),
    ))
    with pytest.raises(ValueError):
        _calibrate([swapped])
    shifted = replace(original, samples=tuple(row[:-1] + (row[-1] + 0.01,) for row in original.samples))
    with pytest.raises(ValueError):
        _calibrate([shifted])


@pytest.mark.parametrize("kind", ["directed", "loop", "disconnected", "third-node", "zero", "negative"])
def test_only_declared_passive_two_node_graph_domain_is_accepted(kind):
    graph = nx.DiGraph() if kind == "directed" else nx.Graph()
    graph.add_nodes_from(("left", "right"))
    if kind != "disconnected":
        graph.add_edge("left", "right", weight=-1 if kind == "negative" else (0 if kind == "zero" else 1))
    if kind == "loop":
        graph.add_edge("left", "left", weight=1)
    if kind == "third-node":
        graph.add_edge("right", "extra", weight=1)
    with pytest.raises((ValueError, TypeError)):
        _calibrate(graph=graph)


@pytest.mark.parametrize("change", [
    {"epi_error": (-1e-10, 0)}, {"epi_error": (float("nan"), 0)},
    {"epi_error": (True, 0)}, {"timestamp_error": -0.1},
    {"timestamp_error": float("inf")}, {"timestamp_error": True},
    {"scales": (0, 1)}, {"scales": (-1, 1)},
    {"structural_time_per_unit": (0, 1)},
    {"structural_time_per_unit": (1, 0.9)},
    {"structural_time_per_unit": (1, float("inf"))},
    {"provenance": ""},
])
def test_measurement_bound_domains_are_explicit(change):
    with pytest.raises((ValueError, TypeError)):
        _measurement(**change)

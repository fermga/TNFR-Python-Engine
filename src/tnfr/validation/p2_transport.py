"""Continuous P2 interval calibration and reserved observation comparisons.

This is the numerical part of the passive-transport measurement annex, not
apparatus admission. All uncertainty bounds are supplied independently and
must hold jointly for the complete finite record. They are not confidence
intervals inferred here. The pressure law is the fixed pure-EPI P2 channel;
continuous references reuse rational log/exp enclosures, never fitted Euler
increments. No live graph is evolved or assigned a state by this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from .._exact_time import exact_or_represented_real as _rational
from .._exact_time import fraction_lower_float, fraction_upper_float
from ..physics._conductance import read_conductance
from ..physics.p2_transport_reference import (
    P2TransportTube,
    bound_p2_capacity,
    bound_p2_transport,
)
from ..utils.io import json_dumps, safe_write
from .nodal_prediction import NodalMeasurementRun, _digest, _labels, _times, _vector

__all__ = [
    "P2MeasurementBounds",
    "P2IntervalCalibration",
    "P2IntervalForecast",
    "P2IntervalComparison",
    "calibrate_p2_transport",
    "forecast_p2_transport",
    "score_p2_transport",
    "write_p2_transport_forecast",
]


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _pair(values, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a pair")
    result = tuple(_rational(value, name) for value in values)
    if len(result) != 2:
        raise ValueError(f"{name} must be a pair")
    return result


def _positive_interval(values, name):
    result = _pair(values, name)
    if not 0 < result[0] <= result[1]:
        raise ValueError(f"{name} requires ordered strictly positive bounds")
    return result


def _payload(value):
    """Encode exact endpoints without float conversion or integer digit limits."""
    if isinstance(value, Fraction):
        return {
            "numerator_hex": hex(value.numerator),
            "denominator_hex": hex(value.denominator),
        }
    if isinstance(value, dict):
        return {key: _payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    return value


def _hash(value):
    return _digest(_payload(asdict(value)))


@dataclass(frozen=True)
class P2MeasurementBounds:
    """Independent sensor/clock specification with deterministic joint bounds.

    ``epi_error`` bounds total coordinate error AFTER the fixed offset/scale
    conversion, including their calibration uncertainty. ``timestamp_error``
    bounds each timestamp in the declared instrument unit; subtracting two
    timestamps therefore costs twice that bound. The clock bridge interval
    maps that physical duration to structural time. These bounds have no
    implicit statistical coverage or apparatus-validity claim.
    Each sensor pair must represent one common latent time. Acquisition skew
    needs synchronization evidence or an independently bounded contribution
    inside ``epi_error``; timestamp error alone does not align two channels.
    """

    offsets: tuple[Fraction, Fraction]
    scales: tuple[Fraction, Fraction]
    epi_error: tuple[Fraction, Fraction]
    timestamp_error: Fraction
    structural_time_per_unit: tuple[Fraction, Fraction]
    provenance: str

    def __post_init__(self):
        for name in ("offsets", "scales", "epi_error"):
            object.__setattr__(self, name, _pair(getattr(self, name), name))
        if min(self.scales) <= 0 or min(self.epi_error) < 0:
            raise ValueError("scales must be positive and errors nonnegative")
        error = _rational(self.timestamp_error, "timestamp_error")
        if error < 0:
            raise ValueError("timestamp_error must be nonnegative")
        object.__setattr__(self, "timestamp_error", error)
        object.__setattr__(
            self,
            "structural_time_per_unit",
            _positive_interval(
                self.structural_time_per_unit, "structural_time_per_unit"
            ),
        )
        _text(self.provenance, "provenance")


def _coordinates(pair, measurement):
    values = _pair(pair, "measurement")
    return tuple(
        ((value - offset) / scale - error, (value - offset) / scale + error)
        for value, offset, scale, error in zip(
            values,
            measurement.offsets,
            measurement.scales,
            measurement.epi_error,
            strict=True,
        )
    )


def _elapsed(timestamps, measurement):
    origin = _rational(timestamps[0], "origin")
    result = [(Fraction(0), Fraction(0))]
    for time in timestamps[1:]:
        duration = _rational(time, "timestamp") - origin
        lower = duration - 2 * measurement.timestamp_error
        if lower <= 0:
            raise ValueError("timestamp uncertainty does not resolve duration")
        result.append(
            (
                lower * measurement.structural_time_per_unit[0],
                (duration + 2 * measurement.timestamp_error)
                * measurement.structural_time_per_unit[1],
            )
        )
    return tuple(result)


def _identity_sequence(values, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_text(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


@dataclass(frozen=True)
class P2IntervalCalibration:
    """Detached declared calibration record, not a sealed physical certificate."""

    channel_ids: tuple[str, str]
    measurement: P2MeasurementBounds
    capacity: tuple[Fraction, Fraction]
    calibration_run_ids: tuple[str, ...]
    calibration_acquisition_ids: tuple[str, ...]
    calibration_hashes: tuple[str, ...]
    value_unit: str
    time_unit: str
    support_provenance: str

    def __post_init__(self):
        labels = _labels(self.channel_ids)
        if len(labels) != 2:
            raise ValueError("P2 requires exactly two channels")
        object.__setattr__(self, "channel_ids", labels)
        if not isinstance(self.measurement, P2MeasurementBounds):
            raise TypeError("measurement must be P2MeasurementBounds")
        object.__setattr__(
            self, "capacity", _positive_interval(self.capacity, "capacity")
        )
        for name in (
            "calibration_run_ids",
            "calibration_acquisition_ids",
            "calibration_hashes",
        ):
            object.__setattr__(
                self, name, _identity_sequence(getattr(self, name), name)
            )
        n = len(self.calibration_run_ids)
        if (
            len(set(self.calibration_run_ids)) != n
            or len(set(self.calibration_hashes)) != n
            or len(self.calibration_acquisition_ids) != n
            or len(self.calibration_hashes) != n
        ):
            raise ValueError("calibration identities must match unique runs")
        for name in ("value_unit", "time_unit", "support_provenance"):
            _text(getattr(self, name), name)

    @property
    def content_hash(self):
        return _hash(self)


def calibrate_p2_transport(runs, *, graph, measurement, support_provenance):
    """Enclose common continuous capacity from fixed first/last observations.

    Selection of run duration and endpoints must precede held-out evaluation.
    Interior calibration samples are not fitted by this endpoint estimator.
    Nonempty intersections are necessary outer consistency checks, not proof
    that one latent trajectory realizes all calibration uncertainty boxes.
    Capacity endpoints are rounded outward to binary64 to bound subsequent
    rational computation size. No physical parameter is silently clipped.
    """
    runs = tuple(runs)
    if not runs or any(not isinstance(run, NodalMeasurementRun) for run in runs):
        raise ValueError("supply nonempty calibration runs")
    if not isinstance(measurement, P2MeasurementBounds):
        raise TypeError("measurement must be P2MeasurementBounds")
    first = runs[0]
    if (
        len(first.channel_ids) != 2
        or tuple(graph) != first.channel_ids
        or graph.is_directed()
        or graph.is_multigraph()
        or graph.number_of_edges() != 1
        or not graph.has_edge(*first.channel_ids)
    ):
        raise ValueError("requires known undirected P2 without self loops")
    weights = read_conductance(graph, symmetric=True).dense()
    if weights[0, 1] <= 0:
        raise ValueError("P2 conductance must be positive")
    enclosures = []
    for run in runs:
        if (run.channel_ids, run.value_unit, run.time_unit) != (
            first.channel_ids,
            first.value_unit,
            first.time_unit,
        ):
            raise ValueError("calibration coordinates and units must match")
        enclosures.append(
            bound_p2_capacity(
                _coordinates(tuple(row[0] for row in run.samples), measurement),
                _coordinates(tuple(row[-1] for row in run.samples), measurement),
                elapsed_time=_elapsed(
                    (run.timestamps[0], run.timestamps[-1]), measurement
                )[-1],
            ).capacity
        )
    lower, upper = max(x[0] for x in enclosures), min(x[1] for x in enclosures)
    if lower > upper:
        raise ValueError("calibration capacity enclosures do not intersect")
    lo, hi = fraction_lower_float(lower), fraction_upper_float(upper)
    if lo <= 0 or not math.isfinite(hi):
        raise ValueError("capacity bounds have no finite positive representation")
    return P2IntervalCalibration(
        first.channel_ids,
        measurement,
        (Fraction(lo), Fraction(hi)),
        tuple(run.run_id for run in runs),
        tuple(run.acquisition_id for run in runs),
        tuple(run.content_hash for run in runs),
        first.value_unit,
        first.time_unit,
        support_provenance,
    )


@dataclass(frozen=True)
class P2IntervalForecast:
    """Pre-evaluation conditional model tube; no future samples are consumed."""

    calibration_hash: str
    evaluation_run_id: str
    evaluation_acquisition_id: str
    initial_measurement: tuple[float, float]
    timestamps: tuple[float, ...]
    tube: P2TransportTube

    def __post_init__(self):
        for name in (
            "calibration_hash",
            "evaluation_run_id",
            "evaluation_acquisition_id",
        ):
            _text(getattr(self, name), name)
        initial = _vector(self.initial_measurement, "initial_measurement")
        if len(initial) != 2:
            raise ValueError("two initial measurements are required")
        object.__setattr__(self, "initial_measurement", initial)
        object.__setattr__(self, "timestamps", _times(self.timestamps))
        if not isinstance(self.tube, P2TransportTube) or len(self.tube.samples) != len(
            self.timestamps
        ):
            raise ValueError("tube must cover all declared timestamps")

    @property
    def content_hash(self):
        return _hash(self)


def forecast_p2_transport(
    calibration,
    *,
    evaluation_run_id,
    evaluation_acquisition_id,
    initial_measurement,
    timestamps,
):
    """Issue the conditional continuous tube before reading reserved values."""
    if (
        evaluation_run_id in calibration.calibration_run_ids
        or evaluation_acquisition_id in calibration.calibration_acquisition_ids
    ):
        raise ValueError("evaluation acquisition overlaps calibration")
    times = _times(timestamps)
    initial = _vector(initial_measurement, "initial_measurement")
    tube = bound_p2_transport(
        _coordinates(initial, calibration.measurement),
        capacity=calibration.capacity,
        elapsed_times=_elapsed(times, calibration.measurement),
    )
    return P2IntervalForecast(
        calibration.content_hash,
        evaluation_run_id,
        evaluation_acquisition_id,
        initial,
        times,
        tube,
    )


@dataclass(frozen=True)
class P2IntervalComparison:
    """Outer-enclosure comparison, not existence or statistical acceptance."""

    forecast_hash: str
    observation_hash: str
    status: str
    incompatible_nodes: tuple[tuple[int, int], ...]
    incompatible_mean_samples: tuple[int, ...]
    incompatible_contrast_samples: tuple[int, ...]
    physical_status: str = "not_admitted_by_this_score"


def score_p2_transport(forecast, calibration, observation, *, expected_forecast_hash):
    """Reject disjoint boxes; overlapping outer boxes do not prove model fit."""
    if forecast.content_hash != expected_forecast_hash:
        raise ValueError("issued forecast hash mismatch")
    if forecast.calibration_hash != calibration.content_hash:
        raise ValueError("forecast/calibration hash mismatch")
    if (
        observation.run_id in calibration.calibration_run_ids
        or observation.acquisition_id in calibration.calibration_acquisition_ids
        or observation.content_hash in calibration.calibration_hashes
    ):
        raise ValueError("evaluation observations overlap calibration")
    if (
        observation.run_id != forecast.evaluation_run_id
        or observation.acquisition_id != forecast.evaluation_acquisition_id
        or observation.channel_ids != calibration.channel_ids
        or observation.timestamps != forecast.timestamps
        or (observation.value_unit, observation.time_unit)
        != (calibration.value_unit, calibration.time_unit)
    ):
        raise ValueError("reserved identity, coordinates, units or times differ")
    if tuple(row[0] for row in observation.samples) != forecast.initial_measurement:
        raise ValueError("reserved initialization differs from issued forecast")
    nodes, means, contrasts = [], [], []

    def disjoint(a, b):
        return max(a[0], b[0]) > min(a[1], b[1])

    for k, sample in enumerate(forecast.tube.samples[1:], 1):
        pair = _coordinates(
            tuple(row[k] for row in observation.samples), calibration.measurement
        )
        for node in range(2):
            if disjoint(pair[node], sample.epi[node]):
                nodes.append((k, node))
        mean = ((pair[0][0] + pair[1][0]) / 2, (pair[0][1] + pair[1][1]) / 2)
        contrast = (pair[0][0] - pair[1][1], pair[0][1] - pair[1][0])
        if disjoint(mean, sample.mean):
            means.append(k)
        if disjoint(contrast, sample.contrast):
            contrasts.append(k)
    status = (
        "incompatible_with_declared_bounds"
        if nodes or means or contrasts
        else "not_falsified_by_enclosures"
    )
    return P2IntervalComparison(
        forecast.content_hash,
        observation.content_hash,
        status,
        tuple(nodes),
        tuple(means),
        tuple(contrasts),
    )


def write_p2_transport_forecast(forecast, path):
    """Save exact endpoints without overwriting a previously issued record."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("do not overwrite an issued forecast")
    payload = {
        "forecast": _payload(asdict(forecast)),
        "content_hash": forecast.content_hash,
    }
    safe_write(
        destination,
        lambda stream: stream.write(
            json_dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        ),
    )
    return destination

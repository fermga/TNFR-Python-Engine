"""Frozen calibration and reserved forecasts for restricted pure-EPI transport.

This is an engineering evaluation boundary, not a physical measurement admission.
Coordinates, clock conversion and known conductance are supplied independently.
Only calibration runs estimate the common positive capacity and baseline. The
finite-increment fit describes refreshed Euler dynamics; it does not identify a
continuous physical rate without a separate discretization/measurement budget.
Forecasts consume an initial observation and declared timestamps, never future
observations. Every EPI evolution step uses the shared nodal integrator.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

from .._exact_time import finite_represented_real
from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..dynamics.integrators import update_epi_via_nodal_equation
from ..mathematics.unified_numerical import np
from ..physics._conductance import read_conductance
from ..physics.structural_diffusion import structural_diffusion_operator
from ..utils.io import json_dumps, safe_write

__all__ = [
    "NodalMeasurementRun",
    "FrozenNodalCalibration",
    "NodalForecast",
    "NodalForecastScore",
    "NodalCalibrationError",
    "calibrate_nodal_prediction",
    "forecast_nodal_response",
    "score_nodal_forecast",
    "write_nodal_forecast",
]


def _digest(payload: Any) -> str:
    encoded = json_dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _real(value: Any, label: str, *, positive: bool = False) -> float:
    number = finite_represented_real(value, label)[0]
    if positive and number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _vector(values: Any, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a numeric sequence")
    return tuple(_real(value, label) for value in values)


def _times(values: Any) -> tuple[float, ...]:
    times = _vector(values, "timestamps")
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("at least two strictly increasing timestamps are required")
    if not all(math.isfinite(b - a) for a, b in zip(times, times[1:])):
        raise ValueError("timestamp intervals must be representable")
    return times


def _labels(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("channel_ids must be a sequence")
    labels = tuple(values)
    if (
        not labels
        or any(not isinstance(s, str) or not s.strip() for s in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("channel_ids must be unique nonempty strings")
    return labels


@dataclass(frozen=True)
class NodalMeasurementRun:
    """Detached channel-major observations with explicit time and unit identity.

    Missing samples must be handled explicitly before admission; this boundary
    rejects them instead of deleting timestamps or silently joining gaps.
    ``acquisition_id`` is the independently reserved preparation/subject/run
    unit. Cropped windows retain it. Its truth is a declared protocol obligation,
    not something inferred from sample values or a renamed file.
    """

    run_id: str
    channel_ids: tuple[str, ...]
    timestamps: tuple[float, ...]
    samples: tuple[tuple[float, ...], ...]
    value_unit: str
    time_unit: str
    acquisition_id: str

    def __post_init__(self) -> None:
        for key in ("run_id", "value_unit", "time_unit", "acquisition_id"):
            if (
                not isinstance(getattr(self, key), str)
                or not getattr(self, key).strip()
            ):
                raise ValueError(f"{key} must be nonempty text")
        object.__setattr__(self, "channel_ids", _labels(self.channel_ids))
        object.__setattr__(self, "timestamps", _times(self.timestamps))
        rows = tuple(_vector(row, "samples") for row in self.samples)
        if len(rows) != len(self.channel_ids) or any(
            len(row) != len(self.timestamps) for row in rows
        ):
            raise ValueError("samples must have shape (channels, timestamps)")
        object.__setattr__(self, "samples", rows)

    @property
    def content_hash(self) -> str:
        """Hash measured content independently of a caller-renamed run ID."""
        payload = asdict(self)
        payload.pop("run_id")
        payload.pop("acquisition_id")
        return _digest(payload)


class NodalCalibrationError(ValueError):
    """An explicit non-admission of a fitted nodal parameter."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _graph(labels: tuple[str, ...], conductance: Any) -> Any:
    matrix = np.asarray(conductance, dtype=float)
    if (
        matrix.shape != (len(labels), len(labels))
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or not np.array_equal(matrix, matrix.T)
    ):
        raise ValueError("conductance must be finite, symmetric and nonnegative")
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    for i, source in enumerate(labels):
        for j in range(i, len(labels)):
            if matrix[i, j] > 0:
                graph.add_edge(source, labels[j], weight=float(matrix[i, j]))
    if len(labels) < 2 or not nx.is_connected(graph):
        raise ValueError("the admitted transport domain requires connected support")
    return graph


@dataclass(frozen=True)
class FrozenNodalCalibration:
    """An immutable fitted model; no physical uncertainty claim is implied."""

    channel_ids: tuple[str, ...]
    conductance: tuple[tuple[float, ...], ...]
    calibration_run_ids: tuple[str, ...]
    calibration_hashes: tuple[str, ...]
    calibration_acquisition_ids: tuple[str, ...]
    offsets: tuple[float, ...]
    scales: tuple[float, ...]
    structural_time_per_unit: float
    capacity: float
    baseline_rates: tuple[float, ...]
    baseline_offsets: tuple[float, ...]
    value_unit: str
    time_unit: str
    support_provenance: str
    measurement_provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_ids", _labels(self.channel_ids))
        for field in ("offsets", "scales", "baseline_rates", "baseline_offsets"):
            values = _vector(getattr(self, field), field)
            if len(values) != len(self.channel_ids):
                raise ValueError(f"{field} must match channel_ids")
            object.__setattr__(self, field, values)
        if any(value <= 0 for value in self.scales):
            raise ValueError("scales must be positive")
        for field in ("structural_time_per_unit", "capacity"):
            object.__setattr__(
                self, field, _real(getattr(self, field), field, positive=True)
            )
        matrix = tuple(_vector(row, "conductance") for row in self.conductance)
        _graph(self.channel_ids, matrix)
        object.__setattr__(self, "conductance", matrix)
        for field in (
            "calibration_run_ids",
            "calibration_hashes",
            "calibration_acquisition_ids",
        ):
            if isinstance(getattr(self, field), (str, bytes)):
                raise TypeError(f"{field} must be a sequence")
            values = tuple(getattr(self, field))
            if not values or any(not isinstance(x, str) or not x for x in values):
                raise ValueError(f"{field} must contain nonempty strings")
            if field != "calibration_acquisition_ids" and len(set(values)) != len(
                values
            ):
                raise ValueError(f"duplicate {field}")
            object.__setattr__(self, field, values)
        if len(self.calibration_hashes) != len(self.calibration_run_ids):
            raise ValueError("calibration hashes must identify every run")
        if len(self.calibration_acquisition_ids) != len(self.calibration_run_ids):
            raise ValueError("acquisition identities must identify every run")
        if any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in self.calibration_hashes
        ):
            raise ValueError("calibration_hashes must be SHA-256 digests")
        for field in (
            "value_unit",
            "time_unit",
            "support_provenance",
            "measurement_provenance",
        ):
            if (
                not isinstance(getattr(self, field), str)
                or not getattr(self, field).strip()
            ):
                raise ValueError(f"{field} must be declared")

    @property
    def content_hash(self) -> str:
        return _digest(asdict(self))


def _coordinates(samples: Any, offsets: Any, scales: Any) -> Any:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = (np.asarray(samples) - np.asarray(offsets)[:, None]) / np.asarray(
            scales
        )[:, None]
    if not np.isfinite(result).all():
        raise ValueError("EPI conversion is not finite")
    return result


def calibrate_nodal_prediction(
    runs: Sequence[NodalMeasurementRun],
    *,
    graph: Any,
    offsets: Sequence[float],
    scales: Sequence[float],
    structural_time_per_unit: float,
    support_provenance: str,
    measurement_provenance: str,
) -> FrozenNodalCalibration:
    """Fit one common capacity on calibration-only increments and known support.

    ``EPI=(measurement-offset)/scale`` and the time bridge are supplied sensor
    mappings, never per-window standardizations. Irregular intervals retain
    their measured duration. Their Euler approximation error remains a P2 gate.
    """
    runs = tuple(runs)
    if not runs or any(not isinstance(run, NodalMeasurementRun) for run in runs):
        raise ValueError("supply nonempty calibration runs")
    first = runs[0]
    if tuple(graph) != first.channel_ids:
        raise ValueError("graph order must equal declared channel order")
    for run in runs:
        if (run.channel_ids, run.value_unit, run.time_unit) != (
            first.channel_ids,
            first.value_unit,
            first.time_unit,
        ):
            raise ValueError("calibration runs must share channels and units")
    snapshot = read_conductance(graph, symmetric=True)
    conductance = tuple(tuple(float(x) for x in row) for row in snapshot.dense())
    local = _graph(first.channel_ids, conductance)
    _, laplacian = structural_diffusion_operator(local)
    offsets, scales = _vector(offsets, "offsets"), _vector(scales, "scales")
    if len(offsets) != len(first.channel_ids) or len(scales) != len(offsets):
        raise ValueError("coordinate maps must match channels")
    if any(value <= 0 for value in scales):
        raise ValueError("scales must be positive")
    bridge = _real(structural_time_per_unit, "structural_time_per_unit", positive=True)
    previous, increments, durations, directions = [], [], [], []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for run in runs:
            x = _coordinates(run.samples, offsets, scales)
            dt = np.diff(run.timestamps) * bridge
            if not np.isfinite(dt).all() or np.any(dt <= 0):
                raise ValueError("structural durations must be finite and positive")
            previous.append(x[:, :-1])
            increments.append(np.diff(x, axis=1))
            durations.append(dt)
            directions.append(-(laplacian @ x[:, :-1]) * dt)
        x, r, dt, d = (
            np.concatenate(seq, axis=-1)
            for seq in (previous, increments, durations, directions)
        )
        denominator = float(np.sum(d * d))
        if denominator <= 0:
            raise NodalCalibrationError("capacity_unidentifiable")
        capacity = float(np.sum(d * r) / denominator)
    if not math.isfinite(capacity):
        raise NodalCalibrationError("capacity_unresolved")
    if capacity <= 0:
        raise NodalCalibrationError(
            "negative_capacity" if capacity < 0 else "inactive_capacity"
        )
    # Positivity-preserving Euler is a numerical domain, not a physical law.
    if np.any(capacity * dt > 1.0):
        raise NodalCalibrationError("calibration_step_outside_convex_euler_domain")
    baseline = [
        np.linalg.lstsq(np.column_stack((row * dt, dt)), inc, rcond=None)[0]
        for row, inc in zip(x, r)
    ]
    return FrozenNodalCalibration(
        first.channel_ids,
        conductance,
        tuple(run.run_id for run in runs),
        tuple(run.content_hash for run in runs),
        tuple(run.acquisition_id for run in runs),
        offsets,
        scales,
        bridge,
        capacity,
        tuple(float(item[0]) for item in baseline),
        tuple(float(item[1]) for item in baseline),
        first.value_unit,
        first.time_unit,
        support_provenance,
        measurement_provenance,
    )


@dataclass(frozen=True)
class NodalForecast:
    """A finite open-loop prediction issued without a reserved response."""

    calibration_hash: str
    evaluation_run_id: str
    evaluation_acquisition_id: str
    channel_ids: tuple[str, ...]
    timestamps: tuple[float, ...]
    initial_measurement: tuple[float, ...]
    epi: tuple[tuple[float, ...], ...]
    affine_baseline: tuple[tuple[float, ...], ...]
    absolute_error_bound: float
    max_structural_step: float
    steps_executed: int
    max_update_residual: float
    scope: str = "restricted refreshed-Euler prediction; physical admission separate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_ids", _labels(self.channel_ids))
        object.__setattr__(self, "timestamps", _times(self.timestamps))
        initial = _vector(self.initial_measurement, "initial_measurement")
        if len(initial) != len(self.channel_ids):
            raise ValueError("initial_measurement must match channels")
        object.__setattr__(self, "initial_measurement", initial)
        for field in ("epi", "affine_baseline"):
            rows = tuple(_vector(row, field) for row in getattr(self, field))
            if len(rows) != len(initial) or any(
                len(row) != len(self.timestamps) for row in rows
            ):
                raise ValueError(f"{field} shape must match channels and timestamps")
            object.__setattr__(self, field, rows)
        for field in (
            "absolute_error_bound",
            "max_update_residual",
            "max_structural_step",
        ):
            value = _real(
                getattr(self, field), field, positive=field == "max_structural_step"
            )
            if value < 0:
                raise ValueError(f"{field} must be nonnegative")
            object.__setattr__(self, field, value)
        if (
            isinstance(self.steps_executed, bool)
            or not isinstance(self.steps_executed, int)
            or self.steps_executed < len(self.timestamps) - 1
        ):
            raise ValueError("steps_executed must cover the forecast intervals")
        for field in (
            "calibration_hash",
            "evaluation_run_id",
            "evaluation_acquisition_id",
            "scope",
        ):
            if (
                not isinstance(getattr(self, field), str)
                or not getattr(self, field).strip()
            ):
                raise ValueError(f"{field} must be nonempty text")

    @property
    def content_hash(self) -> str:
        return _digest(asdict(self))


def forecast_nodal_response(
    calibration: FrozenNodalCalibration,
    *,
    evaluation_run_id: str,
    evaluation_acquisition_id: str,
    initial_measurement: Sequence[float],
    timestamps: Sequence[float],
    absolute_error_bound: float,
    max_structural_step: float,
    max_steps: int,
) -> NodalForecast:
    """Advance only the initial observation through the shared nodal integrator.

    The fixed numerical budget and error decision are frozen into the forecast.
    No automatic horizon/mesh expansion follows a poor evaluation result.
    """
    if not isinstance(evaluation_run_id, str) or not evaluation_run_id.strip():
        raise ValueError("evaluation_run_id must be nonempty")
    if evaluation_run_id in calibration.calibration_run_ids:
        raise ValueError("evaluation run overlaps calibration")
    if (
        not isinstance(evaluation_acquisition_id, str)
        or not evaluation_acquisition_id.strip()
    ):
        raise ValueError("evaluation_acquisition_id must be nonempty")
    if evaluation_acquisition_id in calibration.calibration_acquisition_ids:
        raise ValueError("evaluation acquisition overlaps calibration")
    times = _times(timestamps)
    initial = _vector(initial_measurement, "initial_measurement")
    if len(initial) != len(calibration.channel_ids):
        raise ValueError("initial measurement must match channels")
    bound = _real(absolute_error_bound, "absolute_error_bound")
    if bound < 0:
        raise ValueError("absolute_error_bound must be nonnegative")
    max_step = _real(max_structural_step, "max_structural_step", positive=True)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    durations = tuple(
        _real(
            (b - a) * calibration.structural_time_per_unit,
            "structural duration",
            positive=True,
        )
        for a, b in zip(times, times[1:])
    )
    counts = tuple(
        math.ceil(_real(span / max_step, "subdivision ratio", positive=True))
        for span in durations
    )
    if sum(counts) > max_steps:
        raise ValueError("forecast exceeds the declared step budget")
    graph = _graph(calibration.channel_ids, calibration.conductance)
    _, laplacian = structural_diffusion_operator(graph)
    graph.graph.update(
        DT_MIN=0.0,
        EPI_MIN=-float(np.finfo(float).max),
        EPI_MAX=float(np.finfo(float).max),
        CLIP_MODE="hard",
        GAMMA={"type": "none"},
    )
    x = _coordinates(
        np.asarray(initial)[:, None], calibration.offsets, calibration.scales
    )[:, 0]
    for node, value in zip(graph, x):
        set_attr(graph.nodes[node], ALIAS_EPI, float(value))  # initial preparation
        set_attr(graph.nodes[node], ALIAS_VF, calibration.capacity)
    history, baseline_history = [x.copy()], [x.copy()]
    baseline = x.copy()
    residual, t = 0.0, 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for span, count in zip(durations, counts):
            h = _real(span / count, "Euler duration", positive=True)
            if h * calibration.capacity > 1.0:
                raise ValueError("forecast step outside convex Euler domain")
            for _ in range(count):
                pressure = -(laplacian @ x)
                for node, value in zip(graph, pressure):
                    set_attr(graph.nodes[node], ALIAS_DNFR, float(value))
                previous = x.copy()
                update_epi_via_nodal_equation(graph, dt=h, t=t, method="euler")
                x = np.asarray(
                    [get_attr(graph.nodes[node], ALIAS_EPI) for node in graph]
                )
                defect = x - previous - h * (calibration.capacity * pressure)
                residual = max(residual, float(np.max(np.abs(defect))))
                baseline = baseline + h * (
                    np.asarray(calibration.baseline_rates) * baseline
                    + np.asarray(calibration.baseline_offsets)
                )
                if not np.isfinite(x).all() or not np.isfinite(baseline).all():
                    raise ValueError("nonfinite forecast or comparison baseline")
                t += h
            history.append(x.copy())
            baseline_history.append(baseline.copy())
    return NodalForecast(
        calibration.content_hash,
        evaluation_run_id,
        evaluation_acquisition_id,
        calibration.channel_ids,
        times,
        initial,
        tuple(tuple(float(v) for v in row) for row in np.asarray(history).T),
        tuple(tuple(float(v) for v in row) for row in np.asarray(baseline_history).T),
        bound,
        max_step,
        sum(counts),
        residual,
    )


@dataclass(frozen=True)
class NodalForecastScore:
    """Reserved numerical errors; meeting a budget is not physical confirmation."""

    forecast_hash: str
    observation_hash: str
    max_absolute_error: float
    mean_squared_error: float
    persistence_mse: float
    affine_baseline_mse: float
    meets_declared_error_bound: bool
    physical_status: str = "not_admitted_by_this_score"


def score_nodal_forecast(
    forecast: NodalForecast,
    calibration: FrozenNodalCalibration,
    observation: NodalMeasurementRun,
    *,
    expected_forecast_hash: str,
) -> NodalForecastScore:
    """Score against a forecast digest retained before evaluation was opened.

    The caller/protocol owns the retained digest and chronological evidence.
    This check detects changed forecasts or decision bounds relative to it.
    """
    if expected_forecast_hash != forecast.content_hash:
        raise ValueError("issued forecast hash mismatch")
    if (
        forecast.calibration_hash != calibration.content_hash
        or forecast.channel_ids != calibration.channel_ids
    ):
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
        or observation.channel_ids != forecast.channel_ids
        or observation.timestamps != forecast.timestamps
        or (observation.value_unit, observation.time_unit)
        != (calibration.value_unit, calibration.time_unit)
    ):
        raise ValueError(
            "reserved run identity, coordinates, units or timestamps differ"
        )
    if tuple(row[0] for row in observation.samples) != forecast.initial_measurement:
        raise ValueError("reserved initialization differs from issued forecast")
    actual = _coordinates(observation.samples, calibration.offsets, calibration.scales)
    with np.errstate(over="raise", invalid="raise"):
        residual = actual[:, 1:] - np.asarray(forecast.epi)[:, 1:]
        maximum = float(np.max(np.abs(residual)))
        mse = float(np.mean(residual**2))
        persistence = float(np.mean((actual[:, 1:] - actual[:, :1]) ** 2))
        baseline = float(
            np.mean((actual[:, 1:] - np.asarray(forecast.affine_baseline)[:, 1:]) ** 2)
        )
    return NodalForecastScore(
        forecast.content_hash,
        observation.content_hash,
        maximum,
        mse,
        persistence,
        baseline,
        maximum <= forecast.absolute_error_bound,
    )


def write_nodal_forecast(forecast: NodalForecast, path: str | Path) -> Path:
    """Save prediction content for a protocol to bind before reading outcomes.

    This records content, not a trusted timestamp or proof of human chronology.
    The protocol/evidence envelope must retain its pre-evaluation digest.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("do not overwrite an issued forecast")
    payload = {"forecast": asdict(forecast), "content_hash": forecast.content_hash}
    safe_write(
        destination,
        lambda stream: stream.write(
            json_dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        ),
    )
    return destination

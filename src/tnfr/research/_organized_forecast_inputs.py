"""Read-only inputs for the declared archived collective forecast demonstration.

Content hashes associate two fixed archives with their declared source. The
compact witnesses retain the original conditional Taylor derivative premises;
this reader rebuilds propagation, sample association and dual arithmetic. It
does not evaluate a producer, authenticate execution, or supply a new response.
The record interface deliberately exposes only the thirty five-form readings.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from io import BytesIO
from math import factorial
from pathlib import Path
from typing import Any

from ..mathematics._rational_interval import I
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    exact_record,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)

_LIMIT = 128 * 1024**2
_ETA = Q(1, 10**14)


@dataclass(frozen=True)
class _Archive:
    name: str
    revision: str
    source_tree: str
    inventory_sha256: str
    attempt_sha256: str
    archive_sha256: str
    archive_bytes: int
    result_sha256: str
    result_bytes: int


_TRAJECTORY = _Archive(
    "organized_computable_prediction",
    "8f5a28c58f23ab1586cc91ac36d49e6ddd9bbce0",
    "2a01f5abe9c5c60e3e0ec9648cd108126ee066bd",
    "0b32bf625135fda29906696b5cee1fce2d5aa6f3e03945b4832a500c3f69b4f1",
    "8e96c986bae7feda649576241e23bd8679090bf5d665f1ee9344c306e8491d8c",
    "c4fca529593dbfdf445d314508b78f3f188cd3fa3dce3388fa52df3a1581e4d3",
    1452983,
    "8ccabeecabb0dbaf7ba146dc3034ad8cf496e2e258348ccca016b83619273f1f",
    24548218,
)
_GAIN = _Archive(
    "organized_observed_prediction",
    "de5b500f70b1626d75f12484af0e6381fc9d93af",
    "aba2bc499bcdc54dcfeda1b70a34d660187ca644",
    "205a650debe8e22dbe9fe27fe6aa2f704fefa9845e480b8634f31501a43fba91",
    "8dce8c4e9d8af4e9b4cf6927b255a20009226af71b86cfc27a068a6725753b60",
    "49febee9897b4acf738d803d37128d5e9652a19e59cb082c284de780a66e3592",
    448095,
    "29cd3e2c738d884dde68a57db231581bab4e4eea7683ce86a9c6dbd05a9baffe",
    11097988,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _decode(raw: bytes) -> Any:
    return decode_exact_tree(json_loads(raw))


def _sequence(value: Any, length: int | None = None) -> tuple[Any, ...]:
    _require(isinstance(value, (tuple, list)), "expected an ordered sequence")
    _require(length is None or len(value) == length, "sequence dimension differs")
    return tuple(value)


def _coordinates(value: Any, length: int) -> tuple[Q, ...]:
    return tuple(exact_record(item) for item in _sequence(value, length))


def _interval(value: Any) -> I:
    _require(
        isinstance(value, Mapping) and set(value) == {"lo", "hi"},
        "invalid interval record",
    )
    lower, upper = exact_record(value["lo"]), exact_record(value["hi"])
    interval = I(lower, upper)
    _require(
        (interval.lo, interval.hi) == (lower, upper),
        "retained interval endpoints must already be outward dyadics",
    )
    return interval


def _matrix(value: Any, rows: int, columns: int) -> tuple[tuple[I, ...], ...]:
    return tuple(
        tuple(_interval(item) for item in _sequence(row, columns))
        for row in _sequence(value, rows)
    )


@dataclass(frozen=True)
class ArchiveAssociation:
    name: str
    source_revision: str
    source_tree: str
    source_inventory_sha256: str
    attempt_sha256: str
    archive_sha256: str
    result_sha256: str
    proof_scope: str = "conditional_on_retained_Taylor_derivative_and_source_premises"


def _association(spec: _Archive) -> ArchiveAssociation:
    return ArchiveAssociation(
        spec.name,
        spec.revision,
        spec.source_tree,
        spec.inventory_sha256,
        spec.attempt_sha256,
        spec.archive_sha256,
        spec.result_sha256,
    )


def _read_archive(root: Path, spec: _Archive) -> dict[str, Any]:
    folder = Path(root) / "docs/assets" / spec.name
    attempt_raw = read_bytes_bounded(folder / "attempt.json", max_bytes=2**22)
    _require(
        sha256_bytes(attempt_raw) == spec.attempt_sha256,
        "frozen attempt digest differs",
    )
    attempt = _decode(attempt_raw)
    _require(
        attempt["source_revision"] == spec.revision
        and attempt["source_tree"] == spec.source_tree
        and attempt["source_inventory_sha256"] == spec.inventory_sha256,
        "frozen source association differs",
    )
    _require(
        attempt["schema"] == spec.name + "_attempt_v1"
        and type(attempt["maximum_assessments"]) is int
        and attempt["maximum_assessments"] == 1
        and attempt["evaluation_status_at_start"] == "not_evaluated",
        "frozen attempt schema differs",
    )
    receipt = _decode(read_bytes_bounded(folder / "receipt.json", max_bytes=2**20))
    expected_archive = {
        "path": "response.zip",
        "bytes": spec.archive_bytes,
        "sha256": spec.archive_sha256,
    }
    expected_result = {
        "path": "result.json",
        "bytes": spec.result_bytes,
        "sha256": spec.result_sha256,
    }
    _require(
        receipt == {"response_archive": expected_archive, "result": expected_result},
        "frozen receipt differs",
    )
    raw = read_bytes_bounded(folder / "response.zip", max_bytes=_LIMIT)
    _require(
        len(raw) == spec.archive_bytes and sha256_bytes(raw) == spec.archive_sha256,
        "frozen archive bytes differ",
    )
    verify_archive_members(
        BytesIO(raw), {"result.json": spec.result_sha256}, max_bytes=_LIMIT
    )
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        result_raw = archive.read("result.json")
    _require(len(result_raw) == spec.result_bytes, "frozen result size differs")
    result = _decode(result_raw)
    _require(
        result["source_revision"] == spec.revision
        and result["schema"] == spec.name + "_result_v1"
        and type(result["assessment_count"]) is int
        and result["assessment_count"] == 1,
        "frozen result schema differs",
    )
    _require(
        result["attempt"]
        == {
            "path": "attempt.json",
            "bytes": len(attempt_raw),
            "sha256": spec.attempt_sha256,
        },
        "result attempt association differs",
    )
    inventory = attempt["source_inventory"]
    for record in (attempt, result):
        _require(
            set(record["imported_sources"].values()) <= set(inventory),
            "imported source lies outside the frozen inventory",
        )
    return result


def _sample_times() -> tuple[Q, ...]:
    from ..physics._organized_computable_prediction import _forecast_stages

    return tuple(
        time
        for stage in _forecast_stages()
        if stage.name == "plateau"
        for time in (stage.start, stage.start + stage.duration)
    )


def _rebuild_design(raw: Mapping[str, Any]) -> Any:
    """Rebuild coefficient balls and their common sample/future matrices."""
    from ..physics._organized_computable_prediction import _forecast_stages
    from ..physics._organized_observed_prediction import _StaticDesign

    observation = _matrix(raw["observation_matrix"], 150, 15)
    future = _matrix(raw["future_matrix"], 5, 15)
    witnesses = _sequence(raw["steps"], 15 * 143)
    centers = [tuple(Q(i == j) for i in range(5)) for j in range(15)]
    radii = [Q(0)] * 15
    rows: list[tuple[I, ...]] = []
    times: list[Q] = []
    position = 0
    maximum_local = Q(0)

    def snapshot() -> tuple[tuple[I, ...], ...]:
        return tuple(
            tuple(
                I(centers[j][i] - radii[j], centers[j][i] + radii[j]) for j in range(15)
            )
            for i in range(5)
        )

    for stage in _forecast_stages()[1:]:
        time, end = stage.start, stage.start + stage.duration
        if stage.name == "plateau":
            times.append(time - 200)
            rows.extend(snapshot())
        while time < end:
            duration = min(Q(1, 8), end - time)
            for column in range(15):
                step = witnesses[position]
                position += 1
                _require(
                    type(step["column"]) is int and step["column"] == column,
                    "coefficient column order differs",
                )
                _require(
                    exact_record(step["time"]) == time
                    and exact_record(step["duration"]) == duration,
                    "coefficient step clock differs",
                )
                polynomial = tuple(map(_interval, _sequence(step["polynomial"], 5)))
                remainder = tuple(map(_interval, _sequence(step["remainder"], 5)))
                local = I(
                    max(p.radius + r.abs_max for p, r in zip(polynomial, remainder))
                ).hi
                centers[column] = tuple(p.midpoint for p in polynomial)
                radii[column] = I(radii[column] + local).hi
                _require(
                    _coordinates(step["endpoint_center"], 5) == centers[column]
                    and exact_record(step["endpoint_radius"]) == radii[column],
                    "coefficient endpoint propagation differs",
                )
                maximum_local = max(maximum_local, local)
            time += duration
        if stage.name == "plateau":
            times.append(time - 200)
            rows.extend(snapshot())
    _require(
        tuple(rows) == observation and snapshot() == future,
        "coefficient samples do not share the retained step history",
    )
    _require(
        _coordinates(raw["observation_times"], 30)
        == tuple(times)
        == tuple(t - 200 for t in _sample_times()),
        "coefficient sample clock differs",
    )
    radius = max(radii)
    _require(radius <= Q(1, 10**12), "static coefficient budget exceeded")
    return _StaticDesign(
        tuple(times), observation, future, 143, radius, maximum_local, True, None
    )


@dataclass(frozen=True)
class ForecastGainEvidence:
    carrier_future_gain: Q
    source_error: Q
    historical_pair_error: Q
    observation_matrix: tuple[tuple[I, ...], ...]
    provenance: ArchiveAssociation


def load_archived_prediction_gain(root: Path) -> ForecastGainEvidence:
    """Re-admit the saved matrix/weights and rebuild the exact gain and source."""
    from ..physics._organized_observed_prediction import (
        _certify_dual,
        _observation_budget,
    )

    report = _read_archive(root, _GAIN)["report"]
    design = _rebuild_design(report["design"])
    saved = report["certificate"]
    _require(isinstance(saved, Mapping), "archived gain certificate unavailable")
    phase = tuple(
        _coordinates(row, 150) for row in _sequence(saved["phase_weights"], 10)
    )
    future = tuple(
        _coordinates(row, 150) for row in _sequence(saved["future_weights"], 5)
    )
    certificate = _certify_dual(design, phase, future)
    _require(
        certificate.admitted and certificate.carrier_future_gain is not None,
        "rebuilt gain is unavailable",
    )
    budget = _observation_budget()
    return ForecastGainEvidence(
        certificate.carrier_future_gain,
        budget.source_error,
        budget.historical_pair_error,
        design.observation_matrix,
        _association(_GAIN),
    )


@lru_cache(maxsize=8)
def _growth_reference_upper(duration: Q) -> Q:
    exponent = duration / 3000
    _require(0 < exponent < Q(1, 10000), "unsupported compact step duration")
    # An independent rational series upper-bounds exp(duration/3000).
    partial = sum((exponent**k / factorial(k) for k in range(13)), Q(0))
    return partial + exponent**13 / factorial(13) / (1 - exponent / 14)


def _audit_metric_step(
    step: Mapping[str, Any],
    center: tuple[Q, ...],
    radius: Q,
    *,
    time: Q,
    duration: Q,
) -> tuple[tuple[Q, ...], Q]:
    """Rebuild compact-ball propagation under retained derivative premises."""
    _require(
        exact_record(step["time"]) == time
        and exact_record(step["duration"]) == duration,
        "trajectory step clock differs",
    )
    tube = tuple(map(_interval, _sequence(step["tube"], 10)))
    polynomial = tuple(map(_interval, _sequence(step["polynomial"], 10)))
    remainder = tuple(map(_interval, _sequence(step["remainder"], 10)))
    _require(
        radius >= 0
        and all(
            bound.lo < value - radius <= value + radius < bound.hi
            for value, bound in zip(center, tube)
        ),
        "initial ball is outside the retained tube",
    )
    _require(exact_record(step["picard_margin"]) > 0, "missing strict Picard witness")
    local = tuple(p + r for p, r in zip(polynomial, remainder))
    next_center = tuple(value.midpoint for value in local)
    _require(
        _coordinates(step["endpoint_center"], 10) == next_center,
        "trajectory endpoint center differs",
    )
    error = exact_record(step["local_error"])
    _require(
        error >= 0 and error**2 >= sum((value.radius**2 for value in local), Q(0)),
        "local metric error does not enclose all coordinates",
    )
    growth = exact_record(step["growth"])
    _require(growth >= _growth_reference_upper(duration), "growth bound is too small")
    next_radius = I(I(growth * radius).hi + error).hi
    _require(
        exact_record(step["endpoint_radius"]) == next_radius
        and next_radius <= Q(1, 10**15),
        "trajectory radius propagation differs",
    )
    return next_center, next_radius


@dataclass(frozen=True)
class SampleAssociation:
    time: Q
    branch: str
    endpoint_step: int | None
    numerical_radius: Q


@dataclass(frozen=True)
class ArchivedFormRecord:
    times: tuple[Q, ...]
    values: tuple[tuple[Q, ...], ...]
    extraction_errors: tuple[Q, ...]
    samples: tuple[SampleAssociation, ...]
    provenance: ArchiveAssociation
    record_kind: str = "archived_synthetic_software_demonstration"


def _extract_record(report: Mapping[str, Any]) -> ArchivedFormRecord:
    from ..physics._organized_computable_prediction import (
        _forecast_stages,
        _source_and_suffix_bounds,
    )

    stages = _forecast_stages()
    source, margins, _ = _source_and_suffix_bounds()
    _require(min(margins) > 0, "source identity extension unavailable")
    _require(
        report["clock"] == "tau=(1023/1024)*t"
        and report["coordinate_order"] == "Y[0:5],Psi[0:5]; exact command removed"
        and report["branch"] == "outer_one",
        "trajectory model association differs",
    )
    _require(
        report["numerical_method"]
        == "prescribed_time_identity_metric_Taylor16_dyadic128"
        and exact_record(report["growth_upper"]) == Q(1, 3000)
        and exact_record(report["maximum_step"]) == Q(1, 8)
        and report["source_scope"]
        == "conditional_on_original_independent_preparation_and_zero_sums",
        "trajectory numerical or source premises differ",
    )
    formation = report["formation"]
    center = _coordinates(formation["initial_center"], 10)
    radius = exact_record(formation["initial_radius"])
    _require(
        center == tuple(Q(4092 * (c - 2)) for c in range(5)) + (Q(0),) * 5
        and radius == 0,
        "source preparation differs",
    )
    for index, step in enumerate(_sequence(formation["steps"], 1600)):
        center, radius = _audit_metric_step(
            step, center, radius, time=Q(index, 8), duration=Q(1, 8)
        )
    _require(
        _coordinates(formation["endpoint_center"], 10) == center
        and exact_record(formation["endpoint_radius"]) == radius,
        "formation endpoint differs",
    )
    observed = report["observed"]
    _require(
        observed["name"] == "finite_observation"
        and exact_record(observed["start"]) == 200,
        "selected branch is not the original observation branch",
    )
    _require(
        _coordinates(observed["initial_center"], 10) == center
        and exact_record(observed["initial_radius"]) == radius,
        "observation initial state does not continue the formation",
    )
    times = _sample_times()
    selected = set(times)
    values: list[tuple[Q, ...]] = []
    errors: list[Q] = []
    associations: list[SampleAssociation] = []

    def sample(time: Q, step: int | None) -> None:
        if time in selected:
            error = radius + source
            _require(error <= _ETA, "archived extraction exceeds the reading allowance")
            values.append(center[:5])
            errors.append(error)
            associations.append(SampleAssociation(time, "observed", step, radius))

    sample(Q(200), None)
    index = 0
    steps = _sequence(observed["steps"], 143)
    for stage in stages[1:]:
        time, end = stage.start, stage.start + stage.duration
        while time < end and time < times[-1]:
            duration = min(Q(1, 8), end - time)
            center, radius = _audit_metric_step(
                steps[index], center, radius, time=time, duration=duration
            )
            time += duration
            sample(time, index)
            index += 1
        if time >= times[-1]:
            break
    _require(
        tuple(item.time for item in associations) == times,
        "record is not the complete thirty-sample prefix",
    )
    return ArchivedFormRecord(
        times,
        tuple(values),
        tuple(errors),
        tuple(associations),
        _association(_TRAJECTORY),
    )


def extract_archived_form_record(root: Path) -> ArchivedFormRecord:
    """Extract the frozen prefix only; return no phases or reserved endpoint."""
    return _extract_record(_read_archive(root, _TRAJECTORY)["report"])

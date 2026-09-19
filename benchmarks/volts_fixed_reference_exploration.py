"""One predeclared within-trace Volts continuation, without physical admission.

The declared model is pure-EPI P2 with capacities (nu, 0) and reference zero.
Only rows 0 and 24 identify the nominal continuous rate; the prefix-only
affine AR-1 control uses rows 0..24. Rows 25..49 never update either model.
Unknown instrument/clock errors remain unknown. Numerical enclosures do not
serve as measurement confidence intervals or physical acceptance thresholds.

The optional ``research-data`` dependency reads the pinned RData file without
R execution. The command performs no downloads and refuses an existing output
directory. Supply the retained specification hash from before value decoding.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402

from tnfr._exact_time import fraction_lower_float, fraction_upper_float  # noqa: E402
from tnfr.physics.p2_transport_reference import (  # noqa: E402
    bound_fixed_reference_capacity,
    bound_fixed_reference_transport,
)
from tnfr.research import (  # noqa: E402
    ClaimStatus,
    CoreExperimentManifest,
    EvidenceSidecar,
    current_git_source_provenance,
)
from tnfr.utils.io import json_dumps, safe_write  # noqa: E402
from tnfr.validation.nodal_prediction import (  # noqa: E402
    NodalMeasurementRun,
    _digest,
    _times,
)
from tnfr.validation.p2_transport import _payload  # noqa: E402
from benchmarks.volts_data import load_volts, SOURCE_SHA256  # noqa: E402

POLICY = {
    "schema": "tnfr-volts-exploration-spec-v1",
    "source_sha256": SOURCE_SHA256,
    "source_bytes": 767,
    "expanded_cap_bytes": 65536,
    "expected_rows": 50,
    "columns": ["Voltage", "Time"],
    "calibration_indices_inclusive": [0, 24],
    "continuation_indices_inclusive": [25, 49],
    "reference": 0,
    "offset": 0,
    "scale": 1,
    "structural_time_per_reported_second": 1,
    "baselines": ["persistence", "calibration-only affine AR-1 when identifiable"],
    "sensor_uncertainty": None,
    "clock_uncertainty": None,
    "physical_acceptance_threshold": None,
    "split_scope": "within_single_acquisition",
    "physical_status": "not_admitted",
    "measurement_verdict": "not_assessed",
}


def verify_specification(spec_path, protocol_path, expected_spec_hash):
    """Bind the exact retained bytes and fixed design before decoding values."""
    content = Path(spec_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_spec_hash:
        raise ValueError("predeclared specification hash mismatch")
    spec = json.loads(content)
    if any(_digest(spec.get(key)) != _digest(value) for key, value in POLICY.items()):
        raise ValueError("specification does not match the fixed exploratory design")
    if hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest() != spec.get(
        "protocol_sha256"
    ):
        raise ValueError("predeclared protocol hash mismatch")
    return spec


def _prefix(run):
    return NodalMeasurementRun(
        run_id=run.run_id,
        acquisition_id=run.acquisition_id,
        channel_ids=run.channel_ids,
        timestamps=run.timestamps[:25],
        samples=(run.samples[0][:25],),
        value_unit=run.value_unit,
        time_unit=run.time_unit,
    )


def build_continuation(prefix, timestamps):
    """Issue from only 25 prefix values and a declared future time schedule."""
    if (
        len(prefix.timestamps) != 25
        or prefix.channel_ids != ("Voltage",)
        or (prefix.value_unit, prefix.time_unit) != ("V", "s")
    ):
        raise ValueError("requires the fixed 25-row voltage/time prefix")
    times = _times(timestamps)
    if len(times) != 26 or times[0] != prefix.timestamps[-1]:
        raise ValueError("continuation must start at row 24 and include 25 later times")
    start, end = map(Fraction, (prefix.samples[0][0], prefix.samples[0][-1]))
    duration = Fraction(prefix.timestamps[-1]) - Fraction(prefix.timestamps[0])
    rate = bound_fixed_reference_capacity(
        (start, start), (end, end), reference=(0, 0), elapsed_time=(duration, duration)
    ).capacity
    # Outward materialization bounds rational growth, not instrument error.
    lower, upper = fraction_lower_float(rate[0]), fraction_upper_float(rate[1])
    if lower <= 0 or not np.isfinite(upper):
        raise ValueError("nominal rate lacks a finite positive represented interval")
    rate = Fraction(lower), Fraction(upper)
    elapsed = tuple(Fraction(t) - Fraction(times[0]) for t in times)
    tube = bound_fixed_reference_transport(
        (end, end),
        reference=(0, 0),
        capacity=rate,
        elapsed_times=tuple((t, t) for t in elapsed),
    )
    prediction = tuple(float(sum(sample.epi) / 2) for sample in tube.samples)
    values = np.asarray(prefix.samples[0])
    steps = np.diff(prefix.timestamps + times[1:])
    baseline, reason = None, "nonuniform reported steps; AR-1 not admitted"
    if np.allclose(steps, steps[0], rtol=1e-12, atol=1e-15):
        matrix = np.column_stack((values[:-1], np.ones(24)))
        try:
            coefficients, _, rank, _ = np.linalg.lstsq(matrix, values[1:], rcond=None)
        except np.linalg.LinAlgError:
            coefficients, rank = (), -1
        if rank == 2 and np.isfinite(coefficients).all():
            a, b = map(float, coefficients)
            predicted = [float(end)]
            for _ in times[1:]:
                predicted.append(a * predicted[-1] + b)
            if np.isfinite(predicted).all():
                baseline, reason = tuple(predicted), "calibration-only affine AR-1"
            else:
                reason = "affine AR-1 continuation is nonfinite"
        elif rank == -1:
            reason = "affine AR-1 numerical fit failure"
        else:
            reason = "affine AR-1 prefix is rank-deficient or nonfinite"
    return {
        "schema": "tnfr-volts-nominal-continuation-v1",
        "prefix_hash": prefix.content_hash,
        "run_id": prefix.run_id,
        "acquisition_id": prefix.acquisition_id,
        "timestamps": times,
        "nominal_reference": 0,
        "nominal_capacity_enclosure": _payload(rate),
        "capacity_display": (lower, upper),
        "capacity_units": "per reported second; structural bridge one is a convention",
        "predicted_voltage": prediction,
        "nominal_arithmetic_tube": _payload(asdict(tube)),
        "persistence": tuple(float(end) for _ in times),
        "affine_ar1": baseline,
        "affine_ar1_status": reason,
        "sensor_uncertainty": None,
        "clock_uncertainty": None,
        "physical_acceptance_threshold": None,
        "split_scope": "within_single_acquisition",
        "physical_status": "not_admitted",
        "measurement_verdict": "not_assessed",
    }


def score_continuation(run, forecast, *, expected_forecast_hash):
    """Describe fixed suffix errors; no physical pass/fail rule is supplied."""
    if _digest(forecast) != expected_forecast_hash:
        raise ValueError("issued continuation hash mismatch")
    if (
        len(run.timestamps) != 50
        or forecast["prefix_hash"] != _prefix(run).content_hash
        or forecast["run_id"] != run.run_id
        or forecast["acquisition_id"] != run.acquisition_id
        or tuple(forecast["timestamps"]) != run.timestamps[24:]
    ):
        raise ValueError("observed trace does not match the issued continuation")
    actual = np.asarray(run.samples[0][25:])

    def errors(values):
        if values is None:
            return None
        with np.errstate(over="raise", invalid="raise"):
            residual = actual - np.asarray(values[1:])
            return {
                "rmse_volts": float(np.sqrt(np.mean(residual**2))),
                "max_absolute_error_volts": float(np.max(np.abs(residual))),
                "signed_error_volts": tuple(float(x) for x in residual),
            }

    return {
        "forecast_hash": expected_forecast_hash,
        "observation_hash": run.content_hash,
        "acquisition_id": run.acquisition_id,
        "continuation_rows": 25,
        "nodal": errors(forecast["predicted_voltage"]),
        "persistence": errors(forecast["persistence"]),
        "affine_ar1": errors(forecast["affine_ar1"]),
        "affine_ar1_status": forecast["affine_ar1_status"],
        "split_scope": "within_single_acquisition",
        "physical_status": "not_admitted",
        "measurement_verdict": "not_assessed",
        "scope": "One specified conditional continuation; no independent preparation, instrument bounds or physical acceptance threshold.",
    }


def _write_json(path, value):
    safe_write(
        path,
        lambda stream: stream.write(
            json_dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ),
    )


def execute_study(data_path, *, spec_path, protocol_path, expected_spec_hash, output):
    """Run once, retaining exact inputs and issued predictions before scores."""
    spec = verify_specification(spec_path, protocol_path, expected_spec_hash)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refuse to overwrite an existing study directory")
    run, ingestion = load_volts(data_path)
    destination.mkdir(parents=True)
    for source, name, expected in (
        (data_path, "Volts.rda", ingestion["source_sha256"]),
        (spec_path, "spec.json", expected_spec_hash),
        (protocol_path, "protocol.md", spec["protocol_sha256"]),
    ):
        safe_write(
            destination / name,
            lambda stream, p=source: stream.write(Path(p).read_bytes()),
            mode="wb",
            encoding=None,
        )
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise ValueError("retained input changed after verification: " + name)
    _write_json(destination / "ingestion.json", ingestion)
    try:
        forecast = build_continuation(_prefix(run), run.timestamps[24:])
    except (ValueError, ArithmeticError) as exc:
        result = {
            "status": "specified_model_unavailable",
            "reason": str(exc),
            "physical_status": "not_admitted",
            "measurement_verdict": "not_assessed",
            "split_scope": "within_single_acquisition",
        }
        _write_json(destination / "result.json", result)
        return result
    issued_hash = _digest(forecast)
    _write_json(
        destination / "forecast.json",
        {"forecast": forecast, "content_hash": issued_hash},
    )
    result = score_continuation(run, forecast, expected_forecast_hash=issued_hash)
    result["specification_hash"] = expected_spec_hash
    _write_json(destination / "result.json", result)
    revision, dirty, dirty_hash = current_git_source_provenance(
        _ROOT, ("src/tnfr", "benchmarks")
    )
    names = (
        "Volts.rda",
        "spec.json",
        "protocol.md",
        "ingestion.json",
        "forecast.json",
        "result.json",
    )
    manifest = CoreExperimentManifest(
        claim_id="P2-Volts-exploratory-fixed-reference-continuation",
        git_sha=revision,
        source_dirty=dirty,
        dirty_source_hash=dirty_hash,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdata": ingestion["rdata_version"],
        },
        graph_construction="hypothetical fixed P2; active voltage and reference zero",
        capacity_specification="prefix endpoint nominal continuous rate, reference capacity zero",
        solver="existing rational exponential reference; no live graph evolution",
        result_status=ClaimStatus.MEASURED,
        telemetry=("continuation_rmse", "max_absolute_error"),
        controls=("persistence", "prefix-only affine AR-1"),
        artifacts=names,
    )
    sidecar = EvidenceSidecar(
        manifest=manifest,
        artifact="result.json",
        model="conditional pure-EPI fixed-reference P2",
        norm="nominal voltage continuation residuals",
        distance_convention="one hypothetical positive edge; no geometric claim",
        clock="reported seconds; structural bridge one, clock error unknown",
        finite_horizon=run.timestamps[-1] - run.timestamps[24],
        tail_status="UNASSESSED_FINITE_WINDOW",
        provenance={
            "uses_future_samples": False,
            "uses_outcome_derived_wiring": False,
            "fits_on_evaluation_data": False,
            "uses_evaluation_labels": False,
            "uses_postselection": False,
        },
        claim_statement="One predeclared within-trace continuation was computed and compared descriptively",
        claim_status="measured",
        scope="exploratory; physical admission missing",
        assumptions=(
            "unverified zero reference and pure-EPI observation map",
            "single acquisition; no independent preparation",
            "unknown instrument and clock uncertainty",
        ),
        outcome="nominal residuals reported; physical measurement verdict not assessed",
        source_imports=("tnfr.physics.p2_transport_reference",),
        dirty_source_hash=dirty_hash or "",
        graph_context={
            "nodes": ["Voltage", "reference"],
            "edges": [[0, 1]],
            "support_status": "hypothetical; source wiring unverified",
            "active_pressure_channel": "EPI only",
        },
        state_context={
            "initial_voltage": run.samples[0][24],
            "reference": 0,
            "reference_capacity": 0,
            "prefix_hash": forecast["prefix_hash"],
            "nominal_capacity_display": forecast["capacity_display"],
        },
        observation_context={
            "split_scope": spec["split_scope"],
            "future_time_schedule_available": True,
            "whole_file_decoded_before_issue": True,
            "future_voltage_used_by_predictor": False,
            "external_trusted_chronology": False,
        },
        numerical_context={"sensor_error_bound": None, "clock_error_bound": None},
        cost_context={
            "record_rows": 50,
            "calibration_rows": 25,
            "continuation_rows": 25,
            "automatic_refits": 0,
        },
        artifact_hashes={
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in names
        },
    )
    sidecar.write_admitted(destination / "evidence.json", root_dir=destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "spec", "protocol", "output", "expected-spec-hash"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = execute_study(
        args.data,
        spec_path=args.spec,
        protocol_path=args.protocol,
        expected_spec_hash=args.expected_spec_hash,
        output=args.output,
    )
    print(json_dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

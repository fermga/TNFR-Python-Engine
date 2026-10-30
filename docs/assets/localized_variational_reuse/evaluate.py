"""Single frozen input-cost pilot; revalidate an already retained candidate.

No candidate generator, old producer, or nonlinear response is executed.
The protocol/source receipts must exist before this script can run.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from tnfr.mathematics._interval_chebyshev import _norm_upper_admitted
from tnfr.physics._localized_variational_panel_inputs import (
    _build_localized_variational_panel_inputs,
)
from tnfr.physics._localized_variational_panels import _bound_cycle_variational_panel
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
    write_json_once,
)
from tnfr.utils.io import json_dumps, json_loads

OUTPUT = Path(__file__).resolve().parent
PRIOR = OUTPUT.parent / "localized_variational_panel"


def _read(path):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(path, max_bytes=16 * 1024**2))
    )


def _check_files():
    freeze = _read(OUTPUT / "freeze.json")
    for receipt in freeze["artifacts"]:
        if file_receipt(OUTPUT / receipt["path"], label=receipt["path"]) != receipt:
            raise ValueError("frozen pilot artifact differs")
    manifest = _read(OUTPUT / "source-manifest.json")
    for receipt in manifest["files"]:
        if file_receipt(ROOT / receipt["path"], label=receipt["path"]) != receipt:
            raise ValueError("current runtime differs from frozen pilot source")
    for receipt in freeze["prior_artifacts"]:
        if file_receipt(PRIOR / receipt["path"], label=receipt["path"]) != receipt:
            raise ValueError("prior evidence differs")
    return freeze


def _retained_candidate():
    report = _read(PRIOR / "report.json")
    receipts = {item["path"]: item for item in report["response_members"]}
    verify_archive_members(
        PRIOR / "response.zip",
        {name: item["sha256"] for name, item in receipts.items()},
        max_bytes=128 * 1024**2,
    )
    with ZipFile(PRIOR / "response.zip") as archive:
        info = archive.getinfo("candidate.json")
        if (
            info.file_size != receipts["candidate.json"]["bytes"]
            or info.file_size > 32 * 1024**2
        ):
            raise ValueError("retained candidate exceeds its byte budget")
        with archive.open(info) as stream:
            data = stream.read(32 * 1024**2 + 1)
    if sha256_bytes(data) != receipts["candidate.json"]["sha256"]:
        raise ValueError("retained candidate bytes differ")
    record = decode_exact_tree(json_loads(data))

    def table(name):
        rows = record[name]
        if type(rows) is not tuple or len(rows) != 1024:
            raise ValueError("retained candidate requires 1024 rows")
        if any(type(row) is not tuple or len(row) != 97 for row in rows):
            raise ValueError("retained candidate requires degree96 rows")
        return tuple(tuple(exact_record(value) for value in row) for row in rows)

    return table("form_coefficients"), table("phase_coefficients"), report


def _record(archive, name, value):
    data = json_dumps(encode_exact_tree(value), sort_keys=True).encode("utf-8")
    archive.writestr(name, data)
    return {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}


def main():
    _check_files()
    write_json_once(
        OUTPUT / "attempt.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": file_receipt(OUTPUT / "freeze.json")["sha256"],
            "maximum_evaluations": 1,
            "python": sys.version,
            "platform": platform.platform(),
            "numeric_threads": 1,
        },
    )
    started = perf_counter()
    try:
        form, phase, prior = _retained_candidate()
        print(
            "Building the same 97 canonical inputs with reused spatial arithmetic.",
            flush=True,
        )
        input_started = perf_counter()
        inputs = _build_localized_variational_panel_inputs(start=0, duration=16)
        input_seconds = perf_counter() - input_started
        print(
            f"Inputs completed in {input_seconds:.3f}s; checking the retained polynomial.",
            flush=True,
        )
        validation_started = perf_counter()
        bounds = _bound_cycle_variational_panel(
            form,
            phase,
            inputs.source_form_coefficients,
            inputs.source_phase_coefficients,
            inputs.edge_coefficient_polynomials,
            sigma=inputs.sigma_bounds,
            duration=16,
            generator_error=Q(1, 2**114),
            source_error=Q(1, 2**109),
        )
        validation_seconds = perf_counter() - validation_started
        initial = _norm_upper_admitted(bounds.initial_form + bounds.initial_phase)
        exact_zero = all(
            sum(((-1) ** k * value for k, value in enumerate(row)), Q(0)) == 0
            for table in (form, phase)
            for row in table
        )
        checks = {
            "candidate_bound": bounds.candidate_norm_upper <= 2**35,
            "polynomial_defect": bounds.polynomial_defect_norm_upper <= Q(1, 2**54),
            "initial_enclosure": initial <= Q(1, 2**70),
            "exact_initial_zero": exact_zero,
            "input_cost": input_seconds <= 180,
            "validation_cost": validation_seconds <= 120,
        }
        with ZipFile(
            OUTPUT / "response.zip", "x", ZIP_DEFLATED, compresslevel=6
        ) as archive:
            members = [
                _record(archive, "inputs.json", asdict(inputs)),
                _record(archive, "validation.json", asdict(bounds)),
            ]
        _check_files()
        panels = -(-inputs.horizon_bounds.hi // 16)
        scenario_seconds = panels * (
            input_seconds + validation_seconds + prior["seconds"]["proposal"]
        )
        scenario_bytes = panels * prior["response"]["bytes"]
        report = {
            "schema": "tnfr.localized-variational-input-reuse-result.v1",
            "scope": "Input implementation/resource pilot; same archived candidate revalidated",
            "panel": [0, 16],
            "checks": checks,
            "candidate_norm_upper": bounds.candidate_norm_upper,
            "polynomial_defect_norm_upper": bounds.polynomial_defect_norm_upper,
            "conditional_integrated_defect_upper": bounds.conditional_integrated_defect_upper,
            "initial_enclosure_norm_upper": initial,
            "local_finite_input_correction_error_upper": 512
            * (initial + bounds.conditional_integrated_defect_upper),
            "seconds": {
                "inputs": input_seconds,
                "factory": inputs.factory_seconds,
                "sampling": inputs.sampling_seconds,
                "transform": inputs.transform_seconds,
                "validation": validation_seconds,
                "total_with_artifact_io": perf_counter() - started,
            },
            "input_speed_ratio_to_retained_attempt": prior["seconds"]["inputs"]
            / input_seconds,
            "resource_scenario": {
                "panel_count": panels,
                "sequential_seconds_excluding_io": scenario_seconds,
                "archive_bytes_at_prior_panel_size": scenario_bytes,
                "time_policy_passed": scenario_seconds <= 86400,
                "storage_policy_passed": scenario_bytes <= 8 * 1024**3,
                "scope": "Mixed measured-cost extrapolation, not a bound or full-run admission",
            },
            "response": file_receipt(OUTPUT / "response.zip", label="response.zip"),
            "response_members": members,
            "prior_report": file_receipt(PRIOR / "report.json", label="report.json"),
            "candidate_regenerated": False,
            "new_variational_response_evaluated": False,
            "whole_interval_evaluated": False,
            "nonlinear_maintenance_established": False,
        }
        write_json_once(OUTPUT / "report.json", encode_exact_tree(report))
        print(json_dumps({"checks": checks, "seconds": report["seconds"]}), flush=True)
    except BaseException as exc:
        write_json_once(
            OUTPUT / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "retry_authorized": False,
            },
        )
        raise


if __name__ == "__main__":
    main()

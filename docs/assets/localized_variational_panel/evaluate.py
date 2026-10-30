"""Frozen first-panel evaluator; never imported by routine research tests.

Run from the repository root after writing protocol, source archive and freeze
receipts. An exclusive attempt record prevents repeating this assessment.
All input/candidate coefficients and rebuilt validation bounds are retained.
"""

from __future__ import annotations

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

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

import numpy as np
import scipy

from tnfr.mathematics._interval_chebyshev import _norm_upper_admitted
from tnfr.physics._localized_variational_candidate import (
    _construct_zero_initial_cycle_candidate,
)
from tnfr.physics._localized_variational_panel_inputs import (
    _build_localized_variational_panel_inputs,
)
from tnfr.physics._localized_variational_panels import _bound_cycle_variational_panel
from tnfr.research.artifact_io import (
    encode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    write_json_once,
)
from tnfr.utils.io import json_dumps, json_loads

OUTPUT = Path(__file__).resolve().parent


def _read(path):
    return json_loads(read_bytes_bounded(path, max_bytes=16 * 1024**2))


def _check_files():
    freeze = _read(OUTPUT / "freeze.json")
    for receipt in freeze["artifacts"]:
        actual = file_receipt(OUTPUT / receipt["path"], label=receipt["path"])
        if actual != receipt:
            raise ValueError("frozen artifact bytes differ")
    manifest = _read(OUTPUT / "source-manifest.json")
    for receipt in manifest["files"]:
        actual = file_receipt(ROOT / receipt["path"], label=receipt["path"])
        if actual != receipt:
            raise ValueError("current source differs from the frozen evaluator")
    return freeze


def _record(archive, name, value):
    data = json_dumps(encode_exact_tree(value), sort_keys=True).encode("utf-8")
    archive.writestr(name, data)
    return {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}


def main():
    freeze = _check_files()
    write_json_once(
        OUTPUT / "attempt.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_sha256": file_receipt(OUTPUT / "freeze.json")["sha256"],
            "maximum_evaluations": 1,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "numeric_threads": 1,
        },
    )
    started = perf_counter()
    members = []
    try:
        with ZipFile(
            OUTPUT / "response.zip", "x", ZIP_DEFLATED, compresslevel=6
        ) as archive:
            print("Constructing the fixed 97-sample canonical input panel.", flush=True)
            inputs_started = perf_counter()
            inputs = _build_localized_variational_panel_inputs(start=0, duration=16)
            input_seconds = perf_counter() - inputs_started
            members.append(_record(archive, "inputs.json", asdict(inputs)))
            print(
                f"Inputs enclosed in {input_seconds:.3f} seconds; constructing proposal.",
                flush=True,
            )
            candidate = _construct_zero_initial_cycle_candidate(
                inputs.source_form_coefficients,
                inputs.source_phase_coefficients,
                inputs.edge_coefficient_polynomials,
                sigma=inputs.sigma_bounds,
                duration=16,
                degree=96,
            )
            members.append(_record(archive, "candidate.json", asdict(candidate)))
            print(
                "Proposal retained; rebuilding the complete interval defect.",
                flush=True,
            )
            validation_started = perf_counter()
            bounds = _bound_cycle_variational_panel(
                candidate.form_coefficients,
                candidate.phase_coefficients,
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
            exact_initial_zero = all(
                sum(((-1) ** k * value for k, value in enumerate(row)), Q(0)) == 0
                for table in (candidate.form_coefficients, candidate.phase_coefficients)
                for row in table
            )
            policy = {
                "candidate_bound": bounds.candidate_norm_upper <= 2**35,
                "polynomial_defect": bounds.polynomial_defect_norm_upper <= Q(1, 2**54),
                "initial_enclosure": initial <= Q(1, 2**70),
                "exact_initial_zero": exact_initial_zero,
                "input_cost": input_seconds <= 600,
                "proposal_cost": candidate.total_seconds <= 180,
                "validation_cost": validation_seconds <= 120,
            }
            members.append(_record(archive, "validation.json", asdict(bounds)))
        _check_files()
        report = {
            "schema": "tnfr.localized-variational-first-panel-result.v1",
            "scope": "One canonical finite-input linear correction panel; no nonlinear response",
            "panel": [0, 16],
            "checks": policy,
            "panel_numerical_policy_passed": all(
                policy[name]
                for name in (
                    "candidate_bound",
                    "polynomial_defect",
                    "initial_enclosure",
                    "exact_initial_zero",
                )
            ),
            "local_resource_policy_passed": all(
                policy[name]
                for name in ("input_cost", "proposal_cost", "validation_cost")
            ),
            "candidate_norm_upper": bounds.candidate_norm_upper,
            "polynomial_defect_norm_upper": bounds.polynomial_defect_norm_upper,
            "conditional_integrated_defect_upper": bounds.conditional_integrated_defect_upper,
            "initial_enclosure_norm_upper": initial,
            "local_finite_input_correction_error_upper": 512
            * (initial + bounds.conditional_integrated_defect_upper),
            "source_coefficient_radius_norm_sum": inputs.source_coefficient_radius_norm_sum,
            "edge_coefficient_radius_operator_upper": inputs.edge_coefficient_radius_operator_upper,
            "horizon_bounds": asdict(inputs.horizon_bounds),
            "seconds": {
                "inputs": input_seconds,
                "factory": inputs.factory_seconds,
                "sampling": inputs.sampling_seconds,
                "transform": inputs.transform_seconds,
                "proposal": candidate.total_seconds,
                "factorization": candidate.factorization_seconds,
                "feedback": candidate.feedback_seconds,
                "validation": validation_seconds,
                "total_with_artifact_io": perf_counter() - started,
            },
            "whole_interval_evaluated": False,
            "nonlinear_maintenance_established": False,
            "response": file_receipt(OUTPUT / "response.zip", label="response.zip"),
            "response_members": members,
            "freeze_schema": freeze["schema"],
        }
        write_json_once(OUTPUT / "report.json", encode_exact_tree(report))
        print(json_dumps({"checks": policy, "seconds": report["seconds"]}), flush=True)
    except BaseException as exc:
        write_json_once(
            OUTPUT / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": perf_counter() - started,
                "retry_authorized": False,
            },
        )
        raise


if __name__ == "__main__":
    main()

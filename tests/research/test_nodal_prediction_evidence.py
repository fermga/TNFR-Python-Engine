"""Public forecast-to-evidence integration; synthetic software control only."""

from dataclasses import FrozenInstanceError, asdict
import hashlib
import json
from pathlib import Path
import platform

import networkx as nx
import numpy as np
import pytest

from tnfr.research import (
    ClaimStatus,
    CoreExperimentManifest,
    EvidenceAdmissionError,
    EvidenceSidecar,
    current_git_source_provenance,
)
from tnfr.utils.io import json_dumps, safe_write
from tnfr.validation import (
    FrozenNodalCalibration,
    NodalMeasurementRun,
    calibrate_nodal_prediction,
    forecast_nodal_response,
    score_nodal_forecast,
    write_nodal_forecast,
)


def _save_json(path, payload):
    safe_write(path, lambda stream: stream.write(
        json_dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    ))


def test_reserved_prediction_bundle_admits_only_its_actual_saved_bytes(tmp_path):
    times = tuple(i / 8 for i in range(9))
    # Independent P2 refreshed-Euler eigenmode formula, not experimental data.
    factor = 1 - 2 * 0.4 / 8
    amplitudes = 0.7 * factor ** np.arange(9)
    training = NodalMeasurementRun(
        run_id="calibration", acquisition_id="software-preparation-A",
        channel_ids=("left", "right"), timestamps=times,
        samples=(1 + amplitudes, 1 - amplitudes),
        value_unit="fixture_units", time_unit="fixture_seconds",
    )
    calibration = calibrate_nodal_prediction(
        [training], graph=nx.Graph([("left", "right")]),
        offsets=(0, 0), scales=(1, 1), structural_time_per_unit=1,
        support_provenance="independently specified software P2 edge",
        measurement_provenance="synthetic unit map; no laboratory admission",
    )
    calibration_path = tmp_path / "calibration.json"
    _save_json(calibration_path, asdict(calibration))
    restored = FrozenNodalCalibration(**json.loads(calibration_path.read_text()))
    assert restored == calibration
    calibration_hash = restored.content_hash
    with pytest.raises(FrozenInstanceError):
        restored.capacity = 999

    forecast = forecast_nodal_response(
        restored, evaluation_run_id="reserved",
        evaluation_acquisition_id="software-preparation-B",
        initial_measurement=(1.5, 2.5), timestamps=times,
        absolute_error_bound=1e-12, max_structural_step=0.125, max_steps=8,
    )
    issued_hash = forecast.content_hash
    forecast_path = write_nodal_forecast(forecast, tmp_path / "forecast.json")
    issued_bytes = forecast_path.read_bytes()
    # Construct the reserved response only after freezing and saving prediction.
    reserved_amplitudes = -0.5 * factor ** np.arange(9)
    observation = NodalMeasurementRun(
        run_id="reserved", acquisition_id="software-preparation-B",
        channel_ids=("left", "right"), timestamps=times,
        samples=(2 + reserved_amplitudes, 2 - reserved_amplitudes),
        value_unit="fixture_units", time_unit="fixture_seconds",
    )
    score = score_nodal_forecast(
        forecast, restored, observation, expected_forecast_hash=issued_hash,
    )
    assert score.meets_declared_error_bound
    assert score.physical_status == "not_admitted_by_this_score"
    assert restored.content_hash == calibration_hash
    assert forecast_path.read_bytes() == issued_bytes
    _save_json(tmp_path / "result.json", asdict(score))

    repository = Path(__file__).resolve().parents[2]
    revision, dirty, dirty_hash = current_git_source_provenance(
        repository, ("src/tnfr/validation", "src/tnfr/research",
                     "tests/research/test_nodal_prediction_evidence.py"),
    )
    names = ("calibration.json", "forecast.json", "result.json")
    manifest = CoreExperimentManifest(
        claim_id="P1-software-forecast-evidence-integration",
        git_sha=revision, source_dirty=dirty, dirty_source_hash=dirty_hash,
        versions={"python": platform.python_version(), "numpy": np.__version__},
        graph_construction="fixed undirected two-node unit-conductance graph",
        capacity_specification="common positive capacity from calibration only",
        solver="shared refreshed-Euler nodal integrator; eight finite steps",
        result_status=ClaimStatus.MEASURED, timestep=0.125,
        telemetry=("max_absolute_error", "issued_forecast_hash"),
        controls=("independent analytic eigenmode", "modified artifact bytes"),
        artifacts=names,
    )
    sidecar = EvidenceSidecar(
        manifest=manifest, artifact="result.json",
        model="restricted pure-EPI software P2 transport",
        norm="maximum absolute EPI coordinate error",
        distance_convention="unit-conductance P2, ordered left/right channels",
        clock="declared fixture seconds; structural bridge one",
        finite_horizon=1.0, tail_status="UNASSESSED_FINITE_WINDOW",
        provenance={
            "uses_future_samples": False,
            "uses_outcome_derived_wiring": False,
            "fits_on_evaluation_data": False,
            "uses_evaluation_labels": False,
            "uses_postselection": False,
        },
        claim_statement="issued software prediction meets its fixed error budget",
        claim_status="measured", scope="synthetic integration control only",
        assumptions=("fixed P2", "declared independent acquisition identities",
                     "no physical measurement admission or trusted chronology"),
        outcome="finite software residual within 1e-12",
        source_imports=("tnfr.validation", "tnfr.research.evidence_sidecar"),
        dirty_source_hash=dirty_hash or "",
        graph_context={"nodes": ["left", "right"], "edges": [[0, 1]]},
        state_context={"initial_measurement": [1.5, 2.5]},
        numerical_context={"absolute_error_bound": 1e-12, "max_steps": 8},
        observation_context={"issued_hash": issued_hash,
                             "physical_status": score.physical_status},
        cost_context={"steps_executed": forecast.steps_executed},
        artifact_hashes={
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in names
        },
    )
    sidecar.validate_for_admission(root_dir=tmp_path)
    path = sidecar.write_admitted(tmp_path / "evidence.json", root_dir=tmp_path)
    saved = json.loads(path.read_text())
    assert saved["manifest"]["claim_id"] == manifest.claim_id
    assert saved["artifact_hashes"] == dict(sidecar.artifact_hashes)
    assert saved["observation_context"]["issued_hash"] == issued_hash

    # Even harmless JSON whitespace changes invalidate a previously bound file.
    forecast_path.write_bytes(issued_bytes + b" ")
    with pytest.raises(EvidenceAdmissionError, match="SHA-256 mismatch: forecast"):
        sidecar.validate_for_admission(root_dir=tmp_path)
    with pytest.raises(EvidenceAdmissionError, match="SHA-256 mismatch: forecast"):
        sidecar.write_admitted(tmp_path / "altered-evidence.json", root_dir=tmp_path)
    assert not (tmp_path / "altered-evidence.json").exists()

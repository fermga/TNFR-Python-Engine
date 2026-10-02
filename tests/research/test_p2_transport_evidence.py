"""Continuous P2 forecast/evidence integration, without physical admission."""

import hashlib
import json
import platform
from dataclasses import asdict
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import networkx as nx
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
    NodalMeasurementRun,
    P2MeasurementBounds,
    calibrate_p2_transport,
    forecast_p2_transport,
    score_p2_transport,
    write_p2_transport_forecast,
)


def _continuous_run(run_id, acquisition_id, times, *, mean, amplitude):
    """Independent analytic fixture; no shared interval owner generates data."""
    with localcontext() as context:
        context.prec = 80
        mean, amplitude = Decimal(mean), Decimal(amplitude)
        contrast_half = [
            amplitude * (-Decimal("0.8") * Decimal.from_float(time)).exp()
            for time in times
        ]
        samples = (
            tuple(float(mean + value) for value in contrast_half),
            tuple(float(mean - value) for value in contrast_half),
        )
    return NodalMeasurementRun(
        run_id=run_id,
        acquisition_id=acquisition_id,
        channel_ids=("left", "right"),
        timestamps=times,
        samples=samples,
        value_unit="fixture_units",
        time_unit="fixture_seconds",
    )


def test_continuous_p2_bundle_binds_issued_exact_forecast_and_result(tmp_path):
    times = (0.0, 0.125, 0.25, 0.5, 1.0)
    measurement = P2MeasurementBounds(
        offsets=(0, 0),
        scales=(1, 1),
        epi_error=(Fraction(1, 10**10), Fraction(1, 10**10)),
        timestamp_error=0,
        structural_time_per_unit=(1, 1),
        provenance="joint synthetic rounding bound; exact paired times; no apparatus",
    )
    calibration = calibrate_p2_transport(
        [
            _continuous_run(
                "calibration", "preparation-A", times, mean="1", amplitude="0.7"
            )
        ],
        graph=nx.Graph([("left", "right")]),
        measurement=measurement,
        support_provenance="declared fixed passive P2 software fixture",
    )
    calibration_hash = calibration.content_hash
    assert calibration.capacity[0] <= Fraction(2, 5) <= calibration.capacity[1]
    forecast = forecast_p2_transport(
        calibration,
        evaluation_run_id="reserved",
        evaluation_acquisition_id="preparation-B",
        initial_measurement=(1.5, 2.5),
        timestamps=times,
    )
    issued_hash = forecast.content_hash
    forecast_path = write_p2_transport_forecast(forecast, tmp_path / "forecast.json")
    issued_bytes = forecast_path.read_bytes()
    encoded = json.loads(issued_bytes)
    assert encoded["content_hash"] == issued_hash
    for sample, saved in zip(
        forecast.tube.samples, encoded["forecast"]["tube"]["samples"]
    ):
        for interval, saved_interval in zip(sample.epi, saved["epi"]):
            assert (
                tuple(
                    Fraction(
                        int(item["numerator_hex"], 16), int(item["denominator_hex"], 16)
                    )
                    for item in saved_interval
                )
                == interval
            )

    # No reserved future samples exist until after the forecast is persisted.
    observed = _continuous_run(
        "reserved", "preparation-B", times, mean="2", amplitude="-0.5"
    )
    comparison = score_p2_transport(
        forecast,
        calibration,
        observed,
        expected_forecast_hash=issued_hash,
    )
    assert comparison.status == "not_falsified_by_enclosures"
    assert comparison.physical_status == "not_admitted_by_this_score"
    assert calibration.content_hash == calibration_hash
    assert forecast_path.read_bytes() == issued_bytes
    safe_write(
        tmp_path / "result.json",
        lambda stream: stream.write(
            json_dumps(asdict(comparison), sort_keys=True, allow_nan=False) + "\n"
        ),
    )

    revision, dirty, dirty_hash = current_git_source_provenance(
        Path(__file__).resolve().parents[2],
        (
            "src/tnfr/validation",
            "src/tnfr/research",
            "src/tnfr/_exact_time.py",
            "src/tnfr/physics/p2_transport_reference.py",
            "src/tnfr/physics/reversible_eigenmode_reference.py",
            "tests/research/test_p2_transport_evidence.py",
        ),
    )
    names = ("forecast.json", "result.json")
    manifest = CoreExperimentManifest(
        claim_id="P2-continuous-software-evidence-integration",
        git_sha=revision,
        source_dirty=dirty,
        dirty_source_hash=dirty_hash,
        versions={"python": platform.python_version()},
        graph_construction="fixed undirected P2; no self loops",
        capacity_specification="calibration-only common positive interval",
        solver="rational continuous log/exp enclosure; no numerical trajectory",
        result_status=ClaimStatus.MEASURED,
        telemetry=("conditional interval compatibility", "issued forecast hash"),
        controls=("independent Decimal fixture", "altered artifact bytes"),
        artifacts=names,
    )
    sidecar = EvidenceSidecar(
        manifest=manifest,
        artifact="result.json",
        model="conditional common-capacity pure-EPI P2",
        norm="coordinate, mean and contrast interval disjointness",
        distance_convention="ordered two-node unit-conductance support",
        clock="exact fixture timestamps; structural bridge one",
        finite_horizon=1.0,
        tail_status="UNASSESSED_FINITE_WINDOW",
        provenance={
            "uses_future_samples": False,
            "uses_outcome_derived_wiring": False,
            "fits_on_evaluation_data": False,
            "uses_evaluation_labels": False,
            "uses_postselection": False,
        },
        claim_statement="synthetic reserved observations do not falsify outer boxes",
        claim_status="measured",
        scope="software integration only",
        assumptions=(
            "joint synthetic error bounds",
            "same latent time per pair",
            "shared-parameter existence is not established",
            "no physical admission or trusted chronology claim",
        ),
        outcome=comparison.status,
        source_imports=("tnfr.validation", "tnfr.research.evidence_sidecar"),
        dirty_source_hash=dirty_hash or "",
        graph_context={"nodes": ["left", "right"], "edges": [[0, 1]]},
        state_context={"initial_measurement": [1.5, 2.5]},
        numerical_context={
            "epi_error": "1/10000000000",
            "calibration_hash": calibration_hash,
        },
        observation_context={
            "issued_hash": issued_hash,
            "physical_status": comparison.physical_status,
        },
        cost_context={"requested_samples": len(times), "trajectory_steps": 0},
        artifact_hashes={
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in names
        },
    )
    sidecar.validate_for_admission(root_dir=tmp_path)
    saved_sidecar = sidecar.write_admitted(
        tmp_path / "evidence.json", root_dir=tmp_path
    )
    assert json.loads(saved_sidecar.read_text())["outcome"] == comparison.status

    forecast_path.write_bytes(issued_bytes + b" ")
    with pytest.raises(EvidenceAdmissionError, match="SHA-256 mismatch: forecast"):
        sidecar.validate_for_admission(root_dir=tmp_path)
    with pytest.raises(EvidenceAdmissionError, match="SHA-256 mismatch: forecast"):
        sidecar.write_admitted(tmp_path / "altered-evidence.json", root_dir=tmp_path)
    assert not (tmp_path / "altered-evidence.json").exists()

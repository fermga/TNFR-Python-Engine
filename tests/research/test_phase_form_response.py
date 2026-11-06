"""Independent modal, matrix and provenance controls for the fixed G3 study."""

import hashlib
import json
import math

import numpy as np
import pytest

from tnfr.research import phase_form_response as response


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    output = tmp_path_factory.mktemp("g3_phase_response")
    original = response._run_case
    issued = []

    def require_issued_declaration(declaration, model, spread):
        # This check runs before each actual trajectory, not after the report.
        path = output / "declaration.json"
        assert path.is_file()
        assert not (output / "result.json").exists()
        issued.append(hashlib.sha256(path.read_bytes()).hexdigest())
        return original(declaration, model, spread)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(response, "_run_case", require_issued_declaration)
        result = response.run_phase_form_response(output)
    assert len(issued) == 4
    assert set(issued) == {result["declaration_sha256"]}
    return output, result


def test_paired_response_has_the_independently_predicted_sign_and_solver_error(
    produced,
):
    _, report = produced
    mean, first, second = math.pi / 8, math.pi / 16, 3 * math.pi / 16
    source_difference = (
        math.sin(mean) * (math.cos(second) - math.cos(first)) / (4 * math.pi)
    )
    continuous = source_difference * (1 - math.exp(-1))
    euler = source_difference * (1 - (255 / 256) ** 256)

    assert report["accepted"]
    assert all(report["checks"].values())
    assert report["paired"]["argument"]["observed"] == pytest.approx(0, abs=1e-11)
    pair = report["paired"]["sine_current"]
    assert pair["observed"] < -0.002
    assert pair["observed"] == pytest.approx(euler, abs=1e-11)
    assert pair["continuous_prediction"] == pytest.approx(continuous, abs=2e-16)
    assert pair["analytic_integration_defect"] == pytest.approx(
        euler - continuous, abs=2e-16
    )
    assert 3e-6 < abs(pair["total_continuous_error"]) < 5e-6
    assert abs(pair["engine_realization_defect"]) < 1e-11
    assert report["physical_status"] == "not_admitted_by_this_software_comparison"


@pytest.mark.parametrize(
    "index",
    range(4),
    ids=("argument-narrow", "argument-wide", "current-narrow", "current-wide"),
)
def test_actual_trajectory_matches_an_independent_augmented_matrix_power(
    produced, index
):
    _, report = produced
    case = report["cases"][index]
    phase = np.asarray(case["phases"])
    neighbors = ((1,), (0, 2), (1,))
    phase_source = np.array(
        [
            np.mean(
                [
                    (
                        phase[j] - phase[i]
                        if case["model"] == "argument"
                        else math.sin(phase[j] - phase[i])
                    )
                    for j in row
                ]
            )
            / math.pi
            for i, row in enumerate(neighbors)
        ]
    )
    laplacian = np.array([[1, -1, 0], [-0.5, 1, -0.5], [0, -1, 1]])
    source = phase_source / 4 + np.array([1, -1, 1]) / 8
    augmented = np.eye(4)
    augmented[:3, :3] -= laplacian / 512
    augmented[:3, 3] = source / 256
    expected = (np.linalg.matrix_power(augmented, 256) @ np.array([0.5, 0.5, 0.5, 1]))[
        :3
    ]
    np.testing.assert_allclose(case["trace"][-1]["epi"], expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        case["trace"][1]["pressure_used"], source, rtol=0, atol=2e-15
    )
    assert case["trace"][1]["pressure_used"] != case["trace"][-1]["pressure_used"]
    assert case["held_coordinates_verified"]
    assert case["unclipped_domain_verified"]
    assert len(case["trace"]) == 257
    assert case["nodal_integrator_calls"] == 256
    for sample in case["trace"]:
        assert all(0.25 < value < 0.75 for value in sample["epi"])
        assert np.dot([0.25, 0.5, 0.25], sample["epi"]) == pytest.approx(0.5, abs=2e-14)


def test_full_mixture_and_nonzero_topology_source_are_retained(produced):
    output, report = produced
    declared = json.loads((output / "declaration.json").read_text())
    assert declared["normalized_weights"] == {
        "phase": 0.25,
        "epi": 0.5,
        "vf": 0.125,
        "topo": 0.125,
    }
    np.testing.assert_allclose(
        report["cases"][0]["trace"][1]["pressure_used"],
        [7 / 64, -3 / 32, 5 / 64],
        rtol=0,
        atol=2e-15,
    )
    # Positive central phase input coexists with negative total pressure because
    # the topology term is included; a phase-only replacement would fail this.
    assert report["cases"][0]["initial_arg_phase_source"][1] > 0
    assert report["cases"][0]["trace"][1]["pressure_used"][1] < 0


def test_evidence_binds_issued_declaration_results_and_working_source(produced):
    output, report = produced
    evidence = json.loads((output / "evidence.json").read_text())
    for name in ("declaration.json", "result.json"):
        assert (
            evidence["artifact_hashes"][name]
            == hashlib.sha256((output / name).read_bytes()).hexdigest()
        )
    assert (
        evidence["observation_context"]["declaration_sha256"]
        == report["declaration_sha256"]
    )
    assert type(evidence["manifest"]["source_dirty"]) is bool
    if evidence["manifest"]["source_dirty"]:
        assert evidence["dirty_source_hash"].startswith("sha256:")
    else:
        assert evidence["dirty_source_hash"] == ""
    assert evidence["cost_context"] == {
        "trajectories": 4,
        "nodal_integrator_calls": 1024,
    }
    assert not any(evidence["provenance"].values())
    assert evidence["tail_status"] == "UNASSESSED_FINITE_WINDOW"


def test_existing_frozen_artifact_cannot_be_overwritten(produced, monkeypatch):
    output, _ = produced
    monkeypatch.setattr(
        response, "_run_case", lambda *_: pytest.fail("must reject before evolution")
    )
    with pytest.raises(FileExistsError, match="fresh output"):
        response.run_phase_form_response(output)


def test_changed_declaration_rejects_evidence_instead_of_refitting(
    tmp_path, monkeypatch
):
    def tamper(*_):
        path = tmp_path / "declaration.json"
        path.write_bytes(path.read_bytes() + b" ")
        return {}

    monkeypatch.setattr(response, "_run_case", tamper)
    with pytest.raises(RuntimeError, match="frozen declaration changed"):
        response.run_phase_form_response(tmp_path)
    assert not (tmp_path / "result.json").exists()

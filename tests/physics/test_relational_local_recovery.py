"""Independent local-law checks and retained finite recovery observations.

The saved two-grid experiment is read once; tests do not repeat its 6144
steps. Static differentiation, an actual frozen boundary step, and CLI
admission are independent controls of the declared research owner.
"""

import json
import math
from copy import deepcopy
from fractions import Fraction as Q

import numpy as np
import pytest

from benchmarks import relational_local_recovery as campaign
from tnfr.dynamics.relational import evaluate_relational_exchange


@pytest.fixture(scope="module")
def retained():
    directory = campaign.ROOT / "docs/assets/relational_local_recovery"
    return (
        json.loads((directory / "result.prediction.json").read_text(encoding="utf-8")),
        json.loads((directory / "result.json").read_text(encoding="utf-8")),
    )


def test_c5_native_linearization_has_the_independent_joint_cycle_blocks():
    graph = campaign._graph()
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=campaign.REFERENCE_PHASE[node])
    baseline = evaluate_relational_exchange(graph, model=campaign.MODEL)
    assert max(map(abs, baseline.pressure)) < 1e-15
    assert baseline.phase_rate == (0.0,) * 5
    laplacian = np.zeros((5, 5))
    for i, j in campaign.EDGES:
        laplacian[i, i] += 1
        laplacian[j, j] += 1
        laplacian[i, j] -= 1
        laplacian[j, i] -= 1
    weighted = np.diag(campaign.CAPACITY) @ laplacian
    cosine = math.cos(math.tau / 5)
    expected = np.block(
        [
            [-weighted / 4, -weighted / (4 * math.pi)],
            [weighted / (4 * math.pi * cosine), np.zeros((5, 5))],
        ]
    )
    # This differentiates the production field, not the benchmark's observer.
    epsilon = 2**-18
    observed = np.zeros((10, 10))
    for column in range(10):
        attribute = "EPI" if column < 5 else "theta"
        node = column % 5
        values = []
        for sign in (-1, 1):
            sample = deepcopy(graph)
            sample.nodes[node][attribute] += sign * epsilon
            field = evaluate_relational_exchange(sample, model=campaign.MODEL)
            values.append(np.array(field.form_rate + field.phase_rate))
        observed[:, column] = (values[1] - values[0]) / (2 * epsilon)
    assert observed == pytest.approx(expected, abs=1e-9)
    eigenvalues = np.linalg.eigvals(expected)
    neutral = [value for value in eigenvalues if abs(value) < 1e-10]
    transverse = [value for value in eigenvalues if abs(value) >= 1e-10]
    assert len(neutral) == 2 and len(transverse) == 8
    bound = 0.75 * (2 - 2 * cosine) / 8
    assert all(value.real <= -bound + 1e-12 for value in transverse)


def test_zero_capacity_suppresses_both_rows_without_removing_pressure():
    boundary = campaign._zero_capacity_control()
    assert boundary["state_frozen"] and boundary["clock_advanced"]
    assert boundary["nonzero_pressure_retained"]
    assert boundary["form_rate"] == boundary["phase_rate"] == (0.0,) * 5
    assert boundary["epi_after"][0] == 1 / 256
    assert boundary["epi_after"][1] == -1 / 256


def test_quotient_observation_removes_common_offsets_and_retains_their_values():
    graph = campaign._graph()
    field = evaluate_relational_exchange(graph, model=campaign.MODEL)
    before = campaign._observation(graph, field, time=0.0)
    for node in graph:
        graph.nodes[node]["EPI"] += 2.0
        graph.nodes[node]["theta"] += 0.5
    field = evaluate_relational_exchange(graph, model=campaign.MODEL)
    after = campaign._observation(graph, field, time=0.0)
    assert after["quotient_state"] == pytest.approx(before["quotient_state"], abs=1e-15)
    assert after["form_mean"] == before["form_mean"] + 2.0
    assert after["phase_error_mean"] == pytest.approx(before["phase_error_mean"] + 0.5)
    assert after["winding"] == before["winding"] == 1


def test_preparation_and_changed_protocol_never_execute_a_trace(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unfrozen trajectory execution")

    monkeypatch.setattr(campaign, "_trace", forbidden)
    prediction = campaign.prepare_prediction()
    changed = deepcopy(prediction)
    changed["epsilon"] *= 2
    with pytest.raises(ValueError, match="frozen protocol"):
        campaign.evaluate_prediction(changed)
    changed = deepcopy(prediction)
    changed["capacity"] = (True,) + changed["capacity"][1:]
    with pytest.raises(ValueError, match="frozen protocol"):
        campaign.evaluate_prediction(changed)
    prediction["model"]["epi_weight"] = 0.0
    assert campaign.prepare_prediction()["model"]["epi_weight"] == 0.5


def test_cli_requires_matching_freeze_and_exclusive_response(tmp_path, monkeypatch):
    output = tmp_path / "result.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("trajectory was not admitted")

    monkeypatch.setattr(campaign, "evaluate_prediction", forbidden)
    with pytest.raises(FileNotFoundError):
        campaign.main(["--output", str(output)])
    assert campaign.main(["--prepare", "--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        campaign.main(["--prepare", "--output", str(output)])
    output.write_text("retained", encoding="utf-8")
    with pytest.raises(FileExistsError, match="retained"):
        campaign.main(["--output", str(output)])
    output.unlink()
    output.with_suffix(".prediction.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="freeze"):
        campaign.main(["--output", str(output)])


def test_retained_recovery_uses_full_quotient_and_preserves_mean_evidence(retained):
    prediction, report = retained
    assert report["prediction"] == prediction
    assert set(report["traces"]) == {str(count) for count in campaign.STEP_COUNTS}
    for count in campaign.STEP_COUNTS:
        trace = report["traces"][str(count)]
        assert trace["observed_state_count"] == count + 1
        assert set(trace["checkpoints"]) == {
            "0",
            str(count // 4),
            str(count // 2),
            str(count),
        }
        for index, frame in trace["checkpoints"].items():
            assert frame["time"] == int(index) * trace["dt"]
            assert frame["capacity"] == list(campaign.CAPACITY)
            form_mean = math.fsum(frame["epi"]) / 5
            phase_error = [
                a - b
                for a, b in zip(frame["phase"], campaign.REFERENCE_PHASE, strict=True)
            ]
            phase_mean = math.fsum(phase_error) / 5
            vector = [value - form_mean for value in frame["epi"]]
            vector += [value - phase_mean for value in phase_error]
            assert frame["form_mean"] == form_mean
            assert frame["phase_error_mean"] == phase_mean
            assert frame["quotient_state"] == vector
            assert frame["quotient_distance"] == math.hypot(*vector)
            assert frame["winding_defined"] and frame["winding"] == 1
            assert frame["acute_margin"] >= trace["minimum_acute_margin"]
        initial = trace["checkpoints"]["0"]["quotient_distance"]
        endpoint = trace["checkpoints"][str(count)]["quotient_distance"]
        assert trace["final_to_initial_distance"] == endpoint / initial
        assert endpoint <= initial / 2
    assert report["passed"] and all(report["checks"].values())


def test_retained_two_grid_difference_is_recounted_without_an_error_claim(retained):
    _, report = retained
    endpoint = [
        report["traces"][str(count)]["checkpoints"][str(count)]["quotient_state"]
        for count in campaign.STEP_COUNTS
    ]
    assert report["endpoint_grid_difference"] == math.dist(*endpoint)
    initial = report["traces"][str(campaign.STEP_COUNTS[0])]["checkpoints"]["0"][
        "quotient_distance"
    ]
    assert (
        report["endpoint_grid_difference_to_initial_distance"]
        == math.dist(*endpoint) / initial
    )
    assert report["endpoint_grid_difference"] <= initial / 64
    assert "not_an_ODE_error_certificate" in report["scope"]


def test_retained_checkpoints_replay_detached_fields_without_a_trajectory(retained):
    _, report = retained
    for trace in report["traces"].values():
        for frame in trace["checkpoints"].values():
            graph = campaign._graph()
            for node in graph:
                graph.nodes[node].update(
                    EPI=frame["epi"][node],
                    theta=frame["phase"][node],
                    nu_f=frame["capacity"][node],
                )
            before = deepcopy(dict(graph.nodes(data=True)))
            field = evaluate_relational_exchange(graph, model=campaign.MODEL)
            assert dict(graph.nodes(data=True)) == before
            for attribute in ("pressure", "form_rate", "phase_rate"):
                assert getattr(field, attribute) == pytest.approx(
                    frame[attribute], abs=1e-12
                )
            for rate, capacity, pressure in zip(
                frame["form_rate"], frame["capacity"], frame["pressure"], strict=True
            ):
                assert abs(Q(rate) - Q(capacity) * Q(pressure)) < Q(1, 10**14)
            winding = (
                math.fsum(
                    math.remainder(
                        frame["phase"][(i + 1) % 5] - frame["phase"][i], math.tau
                    )
                    for i in range(5)
                )
                / math.tau
            )
            assert winding == pytest.approx(1.0, abs=1e-15)

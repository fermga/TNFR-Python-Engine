"""Frozen protocol, detached checkpoints and honest stopped-run accounting.

No test regenerates the bounded continuation. Preparation and a deliberately
rejected step are checked before execution; retained evidence is read once.
"""

import json
import math
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from benchmarks import relational_formation_response as campaign
from tnfr.dynamics.relational import evaluate_relational_exchange
from tnfr.physics.winding_certificates import certify_phase_winding


def _exact(item):
    if set(item) == {"numerator", "denominator"}:
        return Q(item["numerator"], item["denominator"])
    return item


@pytest.fixture(scope="module")
def retained():
    directory = campaign.ROOT / "docs/assets/relational_formation_response"
    return tuple(
        json.loads((directory / name).read_text(encoding="utf-8"), object_hook=_exact)
        for name in ("result.prediction.json", "result.json")
    )


def test_preparation_is_regular_winding_zero_with_the_fixed_transverse_form():
    graph = campaign._graph()
    before = campaign._owned_state(graph)
    field = evaluate_relational_exchange(graph, model=campaign._model())
    geometry = campaign._geometry(graph, field)
    assert campaign._owned_state(graph) == before
    assert geometry["windings"] == (0, 0)
    assert not geometry["strict_acute_edges"]
    assert min(field.resultant_real_lower_bounds) > 0
    assert field.phase_rate[1] - field.phase_rate[0] < 0
    assert field.phase_rate[6] - field.phase_rate[5] < 0
    assert field.epi == campaign.INITIAL_FORM
    assert field.phase == campaign.INITIAL_PHASE
    assert field.capacity == (1.0,) * 10


def test_frozen_protocol_rejects_changed_scalars_and_wrong_loaded_source(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unfrozen temporal execution")

    monkeypatch.setattr(campaign, "_trace", forbidden)
    prediction = campaign.prepare_prediction()
    for key, value in (("capacity", (True,) * 10), ("horizon", Q(32))):
        changed = deepcopy(prediction)
        changed[key] = value
        with pytest.raises(ValueError, match="frozen protocol"):
            campaign.evaluate_prediction(changed)
    monkeypatch.setattr(
        campaign.relational_owner, "__file__", str(campaign.ROOT / "unrelated.py")
    )
    with pytest.raises(ValueError, match="workspace source"):
        campaign.prepare_prediction()


def test_rejected_step_retains_last_state_and_unavailable_proposal_without_retry(
    monkeypatch,
):
    calls = []

    def rejected(graph, *, model, dt):
        calls.append(dt)
        raise ValueError("synthetic regular-chamber admission rejection")

    monkeypatch.setattr(campaign, "step_relational_exchange", rejected)
    prediction = campaign.prepare_prediction()
    report = campaign._trace(campaign.STEP_COUNTS[0], prediction)
    assert calls == [1 / 64]
    assert report["status"] == "stopped_by_shared_owner"
    assert report["completed_steps"] == 0
    assert report["last_admitted_checkpoint"] == "0"
    assert set(report["checkpoints"]) == {"0"}
    assert report["stop"]["owned_state_unchanged"]
    assert report["stop"]["rejected_proposal"] is None
    assert report["stop"]["attempted_clock_target"] == Q(1, 64)
    assert "synthetic regular-chamber" in report["stop"]["reason"]
    assert report["final_horizon_basin_indicator"] is None
    assert not any(report["short_prediction_checks"].values())


def test_cli_freeze_and_output_are_exclusive_and_preserve_failure_record(
    tmp_path, monkeypatch
):
    output = tmp_path / "result.json"

    def failed(prediction):
        raise RuntimeError("synthetic evaluation error")

    monkeypatch.setattr(campaign, "evaluate_prediction", failed)
    with pytest.raises(FileNotFoundError):
        campaign.main(["--output", str(output)])
    assert campaign.main(["--prepare", "--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        campaign.main(["--prepare", "--output", str(output)])
    with pytest.raises(RuntimeError, match="synthetic evaluation"):
        campaign.main(["--output", str(output)])
    failed_record = json.loads(output.read_text(encoding="utf-8"))
    assert "synthetic evaluation error" in failed_record["error"]
    with pytest.raises(FileExistsError, match="retained"):
        campaign.main(["--output", str(output)])


def test_retained_outcomes_are_bound_to_preparation_and_stopped_time(retained):
    prediction, report = retained
    assert report["prediction"] == prediction
    assert set(report["traces"]) == {str(count) for count in campaign.STEP_COUNTS}
    short_checks = []
    completed = []
    for count in campaign.STEP_COUNTS:
        trace = report["traces"][str(count)]
        initial = trace["checkpoints"]["0"]
        assert initial["epi"] == list(campaign.INITIAL_FORM)
        assert initial["phase"] == list(campaign.INITIAL_PHASE)
        assert initial["capacity"] == [1.0] * 10
        assert trace["initial_winding_zero"] == (
            initial["geometry"]["windings"] == [0, 0]
        )
        assert trace["last_admitted_checkpoint"] == str(trace["completed_steps"])
        for index, frame in trace["checkpoints"].items():
            assert frame["time"] == int(index) * trace["dt"]
            assert frame["capacity"] == [1.0] * 10
            assert frame["nodes"] == list(campaign.NODES)
            assert frame["edges"] == [list(edge) for edge in campaign.EDGES]
            assert min(frame["resultant_real_lower_bounds"]) > 0
            graph = campaign._graph()
            for node in graph:
                graph.nodes[node]["theta"] = frame["phase"][node]
            winding = tuple(
                certify_phase_winding(graph, cycle) for cycle in campaign.CYCLES
            )
            assert frame["geometry"]["windings"] == [item.winding for item in winding]
            assert frame["geometry"]["winding_defined"] == [
                item.is_defined for item in winding
            ]
            gaps = tuple(
                math.remainder(
                    float(Q(frame["phase"][j]) - Q(frame["phase"][i])), math.tau
                )
                for i, j in campaign.EDGES
            )
            assert frame["geometry"]["strict_acute_edges"] == all(
                abs(gap) < math.pi / 2 for gap in gaps
            )
            assert frame["geometry"]["minimum_acute_margin"] == min(
                math.pi / 2 - abs(gap) for gap in gaps
            )
        for time in prediction["short_prediction"]["check_times"]:
            index = int(time / Q(trace["dt"]))
            frame = trace["checkpoints"].get(str(index))
            expected = frame is not None and frame["geometry"]["windings"] == [1, 1]
            assert trace["short_prediction_checks"][str(time)] == expected
        short_checks.append(
            trace["initial_winding_zero"]
            and all(trace["short_prediction_checks"].values())
        )
        completed.append(trace["completed_steps"] == count)
        if trace["status"] == "stopped_by_shared_owner":
            assert trace["stop"]["attempted_step"] == trace["completed_steps"] + 1
            assert trace["stop"]["owned_state_unchanged"]
            assert trace["stop"]["rejected_proposal"] is None
            assert trace["final_horizon_basin_indicator"] is None
        else:
            assert trace["status"] == "completed" and trace["completed_steps"] == count
            assert trace["stop"] is None
    assert report["short_prediction_passed"] == all(short_checks)
    assert (report["continuation_status"] == "all_grids_completed") == all(completed)
    assert report["numerical_work_policy_admitted"] == all(
        trace["maximum_actual_balance_residual"]
        <= prediction["gates"]["maximum_actual_balance_residual"]
        for trace in report["traces"].values()
    )


def test_retained_last_checkpoints_replay_shared_observer_without_continuation(
    retained,
):
    prediction, report = retained
    for trace in report["traces"].values():
        saved = trace["checkpoints"][trace["last_admitted_checkpoint"]]
        graph = campaign._graph()
        graph.graph["_t"] = saved["time"]
        for i in graph:
            graph.nodes[i].update(
                EPI=saved["epi"][i], theta=saved["phase"][i], nu_f=saved["capacity"][i]
            )
        replay = campaign._snapshot(graph, campaign._model(), prediction)
        for key in ("pressure", "form_rate", "phase_rate", "phase_source"):
            assert replay[key] == pytest.approx(saved[key], abs=1e-10)
        for current, original in zip(replay["regions"], saved["regions"], strict=True):
            assert current["form_mean"] == original["form_mean"]
            assert current["phase_error_mean"] == original["phase_error_mean"]
            assert current["centered_form"] == tuple(original["centered_form"])
            assert current["centered_phase_error"] == tuple(
                original["centered_phase_error"]
            )
        whole = saved["regions"][-1]
        squared = whole["form_norm_squared"] + whole["phase_norm_squared"]
        assert squared == saved["numerical_basin"]["quotient_squared_norm"]
        policy = prediction["numerical_basin_indicator"]
        excess = float(saved["storage"]) - policy["target_phase_storage_estimate"]
        expected = (
            saved["geometry"]["windings"] == [1, 1]
            and saved["geometry"]["strict_acute_edges"]
            and squared < policy["quotient_squared_threshold"]
            and 0 <= excess < policy["energy_threshold"]
        )
        assert saved["numerical_basin"]["eligible"] == expected
        assert "not_rigorous" in saved["numerical_basin"]["scope"]


def test_retained_grid_comparison_only_claims_common_observed_times(retained):
    _, report = retained
    expected = campaign._comparison(report["traces"])
    assert expected == report["grid_comparisons"]

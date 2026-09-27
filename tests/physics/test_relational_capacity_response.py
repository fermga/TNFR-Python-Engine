"""Prospective capacity intervention and bounded response wiring.

Static identities and one-step execution are checked independently of the
retained trajectory record. CLI controls use stubs and never regenerate a
scientific campaign or replace an earlier frozen response.
"""

import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from benchmarks import relational_capacity_audit as retained_audit
from benchmarks import relational_capacity_response as campaign
from tnfr.alias import get_attr, get_theta_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.dynamics import relational
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)


def _initial(name):
    graph = campaign._graph(name)
    field = evaluate_relational_exchange(
        graph, model=RelationalExchangeModel(storage_scale=1.0)
    )
    return graph, field


def _chi(phase):
    return phase[0] - (phase[1] + phase[2]) / 2


def test_initial_counterpart_has_independent_k3_rate_and_work_balance():
    _, field = _initial("baseline")
    root_three = math.sqrt(3)
    assert field.epi == pytest.approx((1 / 3, 0, -1 / 3))
    assert field.phase == pytest.approx((0, math.pi / 6, -math.pi / 6))
    assert field.capacity == (1, 1, 1)
    assert field.form_gradient == pytest.approx((1, 0, -1))
    assert field.phase_metric == pytest.approx(
        (math.pi * root_three, 2 * (1 + root_three), 2 * (1 + root_three))
    )
    assert field.pressure == pytest.approx((-1 / 4, -1 / 8, 3 / 8))
    actual = campaign.counterpart_phase_rate(field)
    extra = tuple(a - b for a, b in zip(actual, field.phase_rate, strict=True))
    expected = tuple(
        value / 128 for value in (1 + root_three, -3 - root_three, -3 - root_three)
    )
    assert extra == pytest.approx(expected, abs=2e-15)
    local_work = tuple(
        Q(a) * Q(z) for a, z in zip(field.phase_gradient, extra, strict=True)
    )
    assert any(value != 0 for value in local_work)
    assert abs(sum(local_work)) < Q(1, 10**14)
    complete_work = sum(
        Q(q) * Q(dx) + Q(a) * Q(omega)
        for q, dx, a, omega in zip(
            field.form_gradient,
            field.form_rate,
            field.phase_gradient,
            actual,
            strict=True,
        )
    )
    assert abs(complete_work + field.continuous_loss) < Q(1, 10**14)


def test_capacity_intervention_has_predicted_relative_phase_and_form_acceleration():
    _, baseline = _initial("baseline")
    _, intervened = _initial("intervened")
    assert intervened.capacity == (1, 0.5, 1)
    assert intervened.pressure == baseline.pressure
    assert intervened.phase_rate == baseline.phase_rate
    base_extra = tuple(
        value - original
        for value, original in zip(
            campaign.counterpart_phase_rate(baseline), baseline.phase_rate, strict=True
        )
    )
    changed_extra = tuple(
        value - original
        for value, original in zip(
            campaign.counterpart_phase_rate(intervened),
            intervened.phase_rate,
            strict=True,
        )
    )
    assert changed_extra == pytest.approx(
        tuple(2 * value / 3 for value in base_extra), abs=2e-15
    )
    difference = tuple(b - a for a, b in zip(base_extra, changed_extra, strict=True))
    slope = -(2 + math.sqrt(3)) / 192
    assert _chi(difference) == pytest.approx(slope, abs=2e-15)
    # On the regular triangle chart, Dg=(A/2-I)/pi. Capacity and
    # instantaneous form response match between the two candidate laws;
    # the double intervention isolates this source-response difference.
    source_difference_at_zero = (
        (difference[1] + difference[2]) / 2 - difference[0]
    ) / math.pi
    acceleration = 0.5 * source_difference_at_zero
    expected = (2 + math.sqrt(3)) / (384 * math.pi)
    assert acceleration == pytest.approx(expected, abs=2e-15)
    assert acceleration > 0 and slope < 0


def test_changed_protocol_rejects_before_a_trajectory(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("trajectory executed before frozen-protocol admission")

    monkeypatch.setattr(campaign, "_trace", forbidden)
    prediction = campaign.prepare_prediction()
    changed = deepcopy(prediction)
    changed["unregistered_change"] = True
    with pytest.raises(ValueError, match="frozen|protocol|source"):
        campaign.evaluate_prediction(changed)
    assert prediction["capacities"] is not campaign.CAPACITIES
    original = dict(campaign.CAPACITIES)
    prediction["capacities"]["baseline"] = (1.0, 0.0, 1.0)
    assert campaign.CAPACITIES == original
    assert campaign.prepare_prediction()["capacities"] == original
    with pytest.raises(ValueError, match="frozen|protocol|source"):
        campaign.evaluate_prediction(prediction)


def test_one_counterpart_step_keeps_the_same_nodal_form_row():
    graph, before = _initial("intervened")
    duration = 1 / 4096
    phase_rate = campaign.counterpart_phase_rate(before)
    campaign._counterpart_step(graph, dt=duration, t=0.0)
    observed_form = tuple(get_attr(graph.nodes[node], ALIAS_EPI) for node in graph)
    observed_phase = tuple(get_theta_attr(graph.nodes[node]) for node in graph)
    expected_form = tuple(
        value + duration * rate
        for value, rate in zip(before.epi, before.form_rate, strict=True)
    )
    expected_phase = tuple(
        value + duration * rate
        for value, rate in zip(before.phase, phase_rate, strict=True)
    )
    assert observed_form == expected_form
    assert observed_phase == expected_phase
    # Different phase feedback is the sole change to this candidate. It is
    # not implemented by retroactively reconstructing another pressure row.
    assert before.form_rate == pytest.approx((-1 / 4, -1 / 16, 3 / 8))


def test_counterpart_rejects_unsupported_support_before_writing():
    graph, _ = _initial("baseline")
    graph.edges[0, 1]["weight"] = 2.0
    node_state = deepcopy(dict(graph.nodes(data=True)))
    metadata = deepcopy(graph.graph)
    with pytest.raises(ValueError, match="unit|conductance|support"):
        campaign._counterpart_step(graph, dt=1 / 4096, t=0.0)
    assert dict(graph.nodes(data=True)) == node_state
    assert graph.graph == metadata


def test_prepare_is_exclusive_and_failure_is_retained_without_a_run(
    tmp_path, monkeypatch
):
    def failed(*args, **kwargs):
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(campaign, "evaluate_prediction", failed)
    output = tmp_path / "response.json"
    assert campaign.main(["--prepare", "--output", str(output)]) == 0
    frozen = output.with_suffix(".prediction.json")
    original = frozen.read_bytes()
    assert not output.exists()
    with pytest.raises(FileExistsError):
        campaign.main(["--prepare", "--output", str(output)])
    assert frozen.read_bytes() == original
    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        campaign.main(["--output", str(output)])
    retained = json.loads(output.read_text(encoding="utf-8"))
    assert retained["passed"] is False
    assert "injected evaluation failure" in retained["error"]
    assert frozen.read_bytes() == original
    with pytest.raises(FileExistsError):
        campaign.main(["--output", str(output)])


@pytest.fixture(scope="module")
def retained_records():
    owner = campaign.ROOT / "docs/assets/relational_capacity_response"
    return (
        retained_audit.load_exact_json(owner / "result.prediction.json"),
        retained_audit.load_exact_json(owner / "result.json"),
    )


def test_read_only_audit_cli_works_from_outside_the_repository(tmp_path):
    records = tuple(
        retained_audit.DEFAULT_DIRECTORY / name
        for name in ("result.prediction.json", "result.json")
    )
    original = tuple(path.read_bytes() for path in records)
    completed = subprocess.run(
        [
            sys.executable,
            str(campaign.ROOT / "benchmarks/relational_capacity_audit.py"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["record_consistent"] is True
    assert report["recorded_passed"] is True
    assert report["checkpoint_count"] == 36
    assert tuple(path.read_bytes() for path in records) == original
    assert not tuple(tmp_path.iterdir())


@pytest.fixture
def prohibit_trajectory_replay(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a retained-evidence audit must not evolve a state")

    monkeypatch.setattr(campaign, "_trace", forbidden)
    monkeypatch.setattr(campaign, "_counterpart_step", forbidden)
    monkeypatch.setattr(campaign, "evaluate_prediction", forbidden)
    monkeypatch.setattr(campaign, "step_relational_exchange", forbidden)
    monkeypatch.setattr(relational, "step_relational_exchange", forbidden)


def test_shared_auditor_checks_the_retained_response_without_trajectory_replay(
    retained_records, prohibit_trajectory_replay
):
    prediction, report = retained_records
    original = deepcopy(report)
    result = retained_audit.audit_retained(prediction, report)
    assert result["record_consistent"] is True
    assert result["recorded_passed"] is report["passed"]
    assert result["checkpoint_count"] == 36
    available = result["source_compatible"] and result["runtime_compatible"]
    assert result["replay_available"] is available
    assert result["replayed_checkpoints"] == (36 if available else 0)
    if available:
        assert type(result["replay_exact"]) is bool
        assert bool(result["replay_mismatches"]) is not result["replay_exact"]
    else:
        assert result["replay_exact"] is None
        assert not result["replay_mismatches"]
    assert result["maximum_independent_pressure_defect"] <= Q(1, 10**12)
    receiver = result["receiver_identity_defects"]
    assert set(receiver) == {"128", "256", "512"}
    for checkpoints in receiver.values():
        assert set(checkpoints) == {"initial", "midpoint", "final"}
        assert all(type(value) is Q for value in checkpoints.values())
        assert max(map(abs, checkpoints.values())) <= Q(1, 10**12)
    assert report == original


def test_audit_rejects_a_different_initial_phase_frame(
    retained_records, prohibit_trajectory_replay
):
    prediction, report = retained_records
    changed = deepcopy(report)
    initial = changed["trajectories"]["relational"]["baseline"]["128"]["checkpoints"][
        "initial"
    ]
    # Relative geometry is unchanged, but the recorded preparation must still
    # match the independently frozen initial state and shared phase frame.
    initial["phase"] = [value + 0.125 for value in initial["phase"]]
    with pytest.raises((ValueError, TypeError)):
        retained_audit.audit_retained(prediction, changed)


def test_audit_rejects_a_wrapped_equivalent_midpoint_outside_the_common_lift(
    retained_records, prohibit_trajectory_replay
):
    prediction, report = retained_records
    changed = deepcopy(report)
    state = changed["trajectories"]["counterpart"]["intervened"]["256"]["checkpoints"][
        "midpoint"
    ]
    # The wrapped phase and all circular observations are unchanged. The
    # frozen lifted observable cannot acquire an independent full turn.
    state["phase"][0] += 2 * math.pi
    with pytest.raises((ValueError, TypeError)):
        retained_audit.audit_retained(prediction, changed)


def test_audit_rejects_fabricated_pressure_even_when_the_nodal_product_matches(
    retained_records, prohibit_trajectory_replay, tmp_path
):
    prediction, report = retained_records
    changed = deepcopy(report)
    state = changed["trajectories"]["counterpart"]["intervened"]["256"]["checkpoints"][
        "midpoint"
    ]
    state["pressure"][0] += 1 / 64
    state["form_rate"][0] = state["capacity"][0] * state["pressure"][0]
    assert state["form_rate"][0] == state["capacity"][0] * state["pressure"][0]
    with pytest.raises((ValueError, TypeError)):
        retained_audit.audit_retained(prediction, changed, root=tmp_path)


@pytest.mark.parametrize("quantity", ("metric", "work"))
def test_audit_rejects_fabricated_metric_or_self_consistent_work_summary(
    retained_records, prohibit_trajectory_replay, quantity, tmp_path
):
    prediction, report = retained_records
    changed = deepcopy(report)
    state = changed["trajectories"]["counterpart"]["baseline"]["512"]["checkpoints"][
        "final"
    ]
    if quantity == "metric":
        state["phase_metric"][0] *= 2
    else:
        state["actual_storage_rate"] += Q(1, 64)
        state["actual_balance_residual"] = (
            state["actual_storage_rate"] + state["continuous_loss"]
        )
    with pytest.raises((ValueError, TypeError)):
        retained_audit.audit_retained(prediction, changed, root=tmp_path)


def test_audit_rejects_boolean_state_before_numerical_coercion(
    retained_records, prohibit_trajectory_replay
):
    prediction, report = retained_records
    changed = deepcopy(report)
    state = changed["trajectories"]["relational"]["intervened"]["128"]["checkpoints"][
        "midpoint"
    ]
    state["epi"][0] = True
    with pytest.raises((ValueError, TypeError)):
        retained_audit.audit_retained(prediction, changed)


def test_named_protocol_rejects_coherently_replaced_scientific_metadata(
    retained_records, prohibit_trajectory_replay, monkeypatch
):
    prediction, report = (deepcopy(value) for value in retained_records)
    prediction["storage_scale"] = 2.0
    report["prediction"] = deepcopy(prediction)

    def forbidden(*args, **kwargs):
        raise AssertionError("a changed named protocol must reject before field replay")

    monkeypatch.setattr(retained_audit, "evaluate_relational_exchange", forbidden)
    with pytest.raises((ValueError, TypeError)):
        retained_audit.audit_retained(prediction, report)


def test_unavailable_current_sources_do_not_rewrite_the_historical_verdict(
    retained_records, prohibit_trajectory_replay, tmp_path
):
    prediction, report = retained_records
    original = deepcopy(report)
    result = retained_audit.audit_retained(prediction, report, root=tmp_path)
    assert result["record_consistent"] is True
    assert result["recorded_passed"] is report["passed"]
    assert result["checkpoint_count"] == 36
    assert result["source_compatible"] is False
    assert result["source_mismatches"]
    assert result["replay_available"] is False
    assert result["replayed_checkpoints"] == 0
    assert result["replay_exact"] is None
    assert report == original


def test_current_replay_drift_is_separate_from_historical_record_consistency(
    retained_records, prohibit_trajectory_replay, monkeypatch
):
    prediction, report = retained_records
    original = deepcopy(report)
    evaluate = retained_audit.evaluate_relational_exchange

    def one_bit_drift(*args, **kwargs):
        field = evaluate(*args, **kwargs)
        metric = list(field.phase_metric)
        metric[0] = math.nextafter(metric[0], math.inf)
        return replace(field, phase_metric=tuple(metric))

    # Exercise a listed-compatible replay even on a CI host whose installed
    # dependencies differ. Compatibility alone cannot authenticate libm.
    monkeypatch.setattr(retained_audit, "_compatibility", lambda *args: ({}, {}))
    monkeypatch.setattr(retained_audit, "evaluate_relational_exchange", one_bit_drift)
    result = retained_audit.audit_retained(prediction, report)
    assert result["record_consistent"] is True
    assert result["recorded_passed"] is report["passed"]
    assert result["replay_available"] is True
    assert result["replayed_checkpoints"] == 36
    assert result["replay_exact"] is False
    assert result["replay_mismatches"]
    assert report == original

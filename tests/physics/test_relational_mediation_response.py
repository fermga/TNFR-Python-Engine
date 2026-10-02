"""Prospective mediation instrument checks without reserved native traces."""

import hashlib
import json
import zipfile
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest

from benchmarks import relational_mediation_response as study


@pytest.fixture(scope="module")
def prepared():
    return study.prepare_prediction()


def test_preparation_never_accesses_native_field_or_step(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("prediction accessed the reserved native response")

    monkeypatch.setattr(study, "evaluate_relational_exchange", forbidden)
    monkeypatch.setattr(study, "step_relational_exchange", forbidden)
    prediction = study.prepare_prediction()
    assert prediction["step_counts"] == (64, 128)
    assert prediction["mediator_capacities"] == (0, 1, 2)
    assert prediction["horizon"] == study.Q(1, 4)
    assert prediction["epsilon"] == study.Q(1, 64)


@pytest.mark.parametrize("mu", (0, 1, 2))
def test_split_prediction_matches_independently_assembled_full_euler(mu):
    # Derive the actual block formula using dense B and cosine Hessian, with
    # an independent simultaneous full-state Euler recurrence. No native
    # field, Jacobian finite difference or reserved trajectory is accessed.
    adjacency = np.zeros((11, 11))
    stiffness = np.zeros((11, 11))
    for i, j in study.EDGES:
        adjacency[i, j] = adjacency[j, i] = 1
        stiffness[i, j] = stiffness[j, i] = 1 if 10 in (i, j) else np.cos(2 * np.pi / 5)
    degrees, strengths = adjacency.sum(axis=1), stiffness.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    hessian = np.diag(strengths) - stiffness
    capacity = np.array([1.0] * 10 + [float(mu)])
    e, w, beta = (
        study.MODEL.epi_weight,
        study.MODEL.phase_weight,
        study.MODEL.storage_scale,
    )
    full = np.block(
        [
            [
                -e * (capacity / degrees)[:, None] * laplacian,
                -w / np.pi * (capacity / strengths)[:, None] * hessian,
            ],
            [
                w / (beta * np.pi) * (capacity / strengths)[:, None] * laplacian,
                np.zeros((11, 11)),
            ],
        ]
    )
    state = np.array(study.INITIAL_FORM + (0.0,) * 11)
    steps, dt = 64, float(study.HORIZON / 64)
    for _ in range(steps):
        state = state + dt * (full @ state)
    prediction = study.predict(mu, steps)
    assert prediction["state_deviation"] == pytest.approx(state, rel=0, abs=2e-15)
    assert prediction["receiver_port"] == pytest.approx(
        state[[5, 16]], rel=0, abs=2e-15
    )
    assert prediction["mediator"] == pytest.approx(state[[10, 21]], rel=0, abs=2e-15)
    assert prediction["hidden_initial"] == (0.0, 0.0)
    assert prediction["zero_memory_receiver_port"] == (0.0, 0.0)


def test_hidden_update_is_simultaneous_and_quasistatic_control_changes_preparation(
    monkeypatch,
):
    # An auxiliary one-step recurrence is enough to detect consumption of a
    # just-updated mediator; it does not alter the actual frozen protocol.
    monkeypatch.setattr(study, "STEP_COUNTS", (1,))
    prediction = study.predict(1, 1)
    assert (
        prediction["receiver_port"]
        == prediction["zero_memory_receiver_port"]
        == (0.0, 0.0)
    )
    assert prediction["mediator"][0] > 0 > prediction["mediator"][1]
    assert prediction["quasistatic_hidden_initial"] == (float(study.EPSILON) / 2, 0.0)
    assert prediction["quasistatic_receiver_port"][0] != 0


def test_quasistatic_prediction_cannot_resolve_positive_capacity_change(prepared):
    for count in ("64", "128"):
        first, second = (prepared["forecasts"][key][count] for key in ("1", "2"))
        assert first["quasistatic_visible"] == second["quasistatic_visible"]
        assert first["quasistatic_receiver_port"] == second["quasistatic_receiver_port"]
        assert first["receiver_port"] != second["receiver_port"]
        assert prepared["forecasts"]["0"][count]["quasistatic_visible"] is None


@pytest.mark.parametrize("part", ("gate", "forecast", "source", "capacity"))
def test_changed_protocol_rejects_before_reserved_access(prepared, monkeypatch, part):
    changed = deepcopy(prepared)
    if part == "gate":
        changed["gates"]["maximum_relative_receiver_error"] = 0.9
    elif part == "forecast":
        changed["forecasts"]["1"]["64"]["receiver_port"] = (1.0, 1.0)
    elif part == "source":
        changed["source_sha256"][next(iter(changed["source_sha256"]))] = "0" * 64
    else:
        changed["mediator_capacities"] = (0, 1, 3)
    monkeypatch.setattr(study, "prepare_prediction", lambda: prepared)
    monkeypatch.setattr(
        study, "_reference_probe", lambda *args: pytest.fail("native response accessed")
    )
    monkeypatch.setattr(
        study, "_trace", lambda *args: pytest.fail("native response accessed")
    )
    with pytest.raises(ValueError, match="current protocol"):
        study.evaluate_prediction(changed)


def test_workspace_binding_rejects_a_different_imported_engine(monkeypatch):
    monkeypatch.setattr(study.engine_owner, "__file__", "elsewhere/relational.py")
    with pytest.raises(ValueError, match="workspace owners"):
        study.prepare_prediction()


def test_freeze_archives_sources_and_refuses_overwriting_evidence(
    prepared, monkeypatch, tmp_path
):
    monkeypatch.setattr(study, "prepare_prediction", lambda: prepared)
    monkeypatch.setattr(
        study,
        "evaluate_prediction",
        lambda *args: pytest.fail("native response accessed"),
    )
    output = tmp_path / "response.json"
    assert study.main(["--prepare", "--output", str(output)]) == 0
    assert not output.exists()
    assert output.with_suffix(".prediction.json").read_text(
        encoding="utf-8"
    ) == study._encoded(prepared)
    with zipfile.ZipFile(output.with_suffix(".sources.zip")) as bundle:
        assert set(bundle.namelist()) == set(prepared["source_sha256"])
        for path, digest in prepared["source_sha256"].items():
            assert hashlib.sha256(bundle.read(path)).hexdigest() == digest
    with pytest.raises(FileExistsError, match="replace"):
        study.main(["--prepare", "--output", str(output)])


def test_tampered_archive_rejects_before_reserved_execution(
    prepared, monkeypatch, tmp_path
):
    monkeypatch.setattr(study, "prepare_prediction", lambda: prepared)
    monkeypatch.setattr(
        study,
        "evaluate_prediction",
        lambda *args: pytest.fail("native response accessed"),
    )
    output = tmp_path / "response.json"
    study.main(["--prepare", "--output", str(output)])
    with zipfile.ZipFile(output.with_suffix(".sources.zip"), "a") as bundle:
        bundle.writestr("unlisted.py", "not fingerprinted")
    with pytest.raises(ValueError, match="membership"):
        study.main(["--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("mu,steps", ((True, 64), (3, 64), (1, True), (1, 256)))
def test_predictions_do_not_silently_expand_the_protocol(mu, steps):
    with pytest.raises(ValueError, match="protocol"):
        study.predict(mu, steps)


def test_acceptance_uses_absolute_slack_and_independent_signal_resolution():
    gates = dict(study.GATES)
    assert study._comparison((0.0, 0.0), (0.0, 0.0), gates)["passed"] is False
    assert study._comparison((1e-7, -2e-7), (1e-7, -2e-7), gates)["passed"] is True
    assert study._comparison((1e-7, 0.0), (1e-7, -2e-7), gates)["passed"] is False


@pytest.fixture(scope="module")
def retained():
    """Read the historical record directly, without rebuilding its producer."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/relational_mediation_response/result.json"
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _retained_state(frame, reference):
    return tuple(map(Q, frame["epi"])) + tuple(
        Q(phase) - Q(origin)
        for phase, origin in zip(frame["phase"], reference, strict=True)
    )


def _retained_error(actual, predicted):
    return float(max(abs(Q(a) - Q(b)) for a, b in zip(actual, predicted, strict=True)))


def test_retained_record_is_bound_to_its_frozen_prediction_and_source_archive(retained):
    path, report = retained
    prediction = json.loads(
        path.with_suffix(".prediction.json").read_text(encoding="utf-8")
    )
    assert report["prediction"] == prediction
    with zipfile.ZipFile(path.with_suffix(".sources.zip")) as bundle:
        assert len(bundle.namelist()) == len(prediction["source_sha256"])
        assert set(bundle.namelist()) == set(prediction["source_sha256"])
        for name, digest in prediction["source_sha256"].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest
    # The manifest authenticates archived bytes, not today's changed workspace.
    assert report["passed"] and all(report["checks"].values())


def test_retained_endpoint_errors_preserve_port_mediator_and_full_state_scope(retained):
    _, report = retained
    prediction = report["prediction"]
    reference, gates = prediction["reference_phase"], prediction["gates"]
    for mu, traces in report["traces"].items():
        for count, trace in traces.items():
            assert trace["completed_steps"] == trace["requested_steps"] == int(count)
            assert trace["stop"] is None
            state = _retained_state(trace["endpoint"], reference)
            assert tuple(map(Q, trace["endpoint"]["state_deviation"])) == state
            receiver, mediator = (state[5], state[16]), (state[10], state[21])
            assert tuple(map(Q, trace["endpoint"]["receiver_port"])) == receiver
            assert tuple(map(Q, trace["endpoint"]["mediator"])) == mediator
            forecast = prediction["forecasts"][mu][count]
            comparison = report["comparisons"][mu][count]
            assert tuple(map(Q, comparison["actual"])) == receiver
            assert comparison["predicted"] == forecast["receiver_port"]
            port_error = _retained_error(receiver, forecast["receiver_port"])
            full_error = _retained_error(state, forecast["state_deviation"])
            assert comparison["error_linf"] == pytest.approx(
                port_error, rel=2e-15, abs=0
            )
            assert comparison["mediator_error_linf"] == pytest.approx(
                _retained_error(mediator, forecast["mediator"]), rel=2e-15, abs=0
            )
            assert comparison["full_state_error_linf"] == pytest.approx(
                full_error, rel=2e-15, abs=0
            )
            assert full_error >= port_error
            signal = max(map(abs, forecast["receiver_port"]))
            assert comparison["predicted_linf"] == signal
            if mu == "0":
                assert (
                    max(map(abs, receiver)) <= gates["maximum_zero_capacity_receiver"]
                )
                assert not comparison["signal_resolved"]
            else:
                assert signal > 100 * gates["numerical_slack"]
                assert (
                    port_error
                    <= gates["maximum_relative_receiver_error"] * signal
                    + gates["numerical_slack"]
                )
                # This record's much smaller port error is not promoted to
                # the same order or accuracy claim for every fine coordinate.
                assert full_error > port_error
            assert comparison["passed"]
            assert trace["capacity_support_and_path_held"]
            assert trace["minimum_acute_margin"] >= gates["minimum_acute_margin"]
            balance = trace["maximum_defects"]["balance_residual"]
            assert (
                Q(balance["numerator"], balance["denominator"])
                <= gates["maximum_actual_balance_residual"]
            )


def test_retained_capacity_intervention_is_resolved_and_predicted_independently(
    retained,
):
    _, report = retained
    prediction, gates = report["prediction"], report["prediction"]["gates"]
    for count, comparison in report["capacity_differences"].items():
        states = tuple(
            _retained_state(
                report["traces"][mu][count]["endpoint"], prediction["reference_phase"]
            )
            for mu in ("1", "2")
        )
        actual = tuple(float(states[1][i] - states[0][i]) for i in (5, 16))
        forecasts = tuple(
            prediction["forecasts"][mu][count]["receiver_port"] for mu in ("1", "2")
        )
        expected = tuple(
            float(Q(second) - Q(first))
            for first, second in zip(*forecasts, strict=True)
        )
        error = _retained_error(actual, expected)
        signal = max(map(abs, expected))
        assert comparison["actual"] == list(actual)
        assert comparison["predicted"] == list(expected)
        assert comparison["error_linf"] == pytest.approx(error, rel=2e-15, abs=0)
        assert comparison["predicted_linf"] == signal
        assert (
            signal
            > gates["minimum_predicted_signal_in_slack_units"]
            * gates["numerical_slack"]
        )
        assert (
            error
            <= gates["maximum_relative_receiver_error"] * signal
            + gates["numerical_slack"]
        )
        assert comparison["signal_resolved"] and comparison["passed"]


def test_retained_ablation_and_quasistatic_controls_do_not_reproduce_capacity_response(
    retained,
):
    _, report = retained
    prediction = report["prediction"]
    for count in map(str, prediction["step_counts"]):
        first, second = (prediction["forecasts"][mu][count] for mu in ("1", "2"))
        assert first["quasistatic_visible"] == second["quasistatic_visible"]
        assert first["quasistatic_receiver_port"] == second["quasistatic_receiver_port"]
        for mu, forecast in (("1", first), ("2", second)):
            comparison = report["comparisons"][mu][count]
            assert forecast["zero_memory_receiver_port"] == [0.0, 0.0]
            assert (
                comparison["zero_memory_prediction"]
                == forecast["zero_memory_receiver_port"]
            )
            assert (
                comparison["quasistatic_prediction"]
                == forecast["quasistatic_receiver_port"]
            )
            for name in ("zero_memory", "quasistatic"):
                expected_error = _retained_error(
                    comparison["actual"], forecast[name + "_receiver_port"]
                )
                assert comparison[name + "_error_linf"] == pytest.approx(
                    expected_error, rel=2e-15, abs=0
                )
                assert expected_error > comparison["error_linf"]
        difference = report["capacity_differences"][count]
        assert difference["quasistatic_prediction"] == [0.0, 0.0]
        assert difference["quasistatic_error_linf"] == max(
            map(abs, difference["actual"])
        )
        assert difference["quasistatic_error_linf"] > difference["error_linf"]

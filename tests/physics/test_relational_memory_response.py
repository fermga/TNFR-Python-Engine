"""Prospective protocol and retained evidence checks without trajectory replay."""

import hashlib
import json
import zipfile
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from benchmarks import relational_memory_response as study


@pytest.fixture(scope="module")
def prepared():
    return study.prepare_prediction()


def test_prepare_never_evolves_or_observes_a_graph(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must be independent of the reserved engine")

    monkeypatch.setattr(study, "step_relational_exchange", forbidden)
    monkeypatch.setattr(study, "evaluate_relational_exchange", forbidden)
    prediction = study.prepare_prediction()
    assert prediction["step_counts"] == (64, 128, 256)
    assert len(prediction["forecasts"]) == 3


def test_changed_protocol_rejects_before_native_execution(prepared, monkeypatch):
    changed = deepcopy(prepared)
    changed["gates"]["maximum_memory_to_each_control_error"] = 0.9
    monkeypatch.setattr(study, "prepare_prediction", lambda: prepared)
    monkeypatch.setattr(study, "_trace", lambda *args: pytest.fail("response accessed"))
    with pytest.raises(ValueError, match="current protocol"):
        study.evaluate_prediction(changed)


def test_freeze_archives_source_and_refuses_overwrite(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "prepare_prediction", lambda: prepared)
    output = tmp_path / "result.json"
    assert study.main(["--prepare", "--output", str(output)]) == 0
    assert not output.exists()
    with zipfile.ZipFile(output.with_suffix(".sources.zip")) as archive:
        assert set(archive.namelist()) == set(prepared["source_sha256"])
        for path, digest in prepared["source_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    with pytest.raises(FileExistsError, match="replace"):
        study.main(["--prepare", "--output", str(output)])


def test_workspace_import_binding_is_required(monkeypatch):
    monkeypatch.setattr(study.engine_owner, "__file__", "elsewhere/relational.py")
    with pytest.raises(ValueError, match="workspace owners"):
        study.prepare_prediction()


def _rational(value):
    if isinstance(value, dict):
        return Q(value["numerator"], value["denominator"])
    return Q(value)


@pytest.fixture(scope="module")
def retained():
    path = study.ROOT / "docs/assets/relational_memory_response/result.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_retained_prediction_and_archive_are_bound_without_current_source_replay(
    retained,
):
    path, report = retained
    prediction = json.loads(
        path.with_suffix(".prediction.json").read_text(encoding="utf-8")
    )
    assert report["prediction"] == prediction
    with zipfile.ZipFile(path.with_suffix(".sources.zip")) as archive:
        assert set(archive.namelist()) == set(prediction["source_sha256"])
        for name, digest in prediction["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_retained_endpoint_coordinates_and_errors_have_independent_accounting(retained):
    _, report = retained
    prediction = report["prediction"]
    c = float.fromhex(
        dict(prediction["forecasts"]["64"]["binary64"]["coefficient_constants"])["c"]
    )

    def project(values):
        left, right = sum(values[:5]) / 5, sum(values[5:]) / 5
        return (
            left - right,
            values[0] - left,
            values[5] - right,
            (values[1] + values[4] - values[2] - values[3]) / 2,
            (values[6] + values[9] - values[7] - values[8]) / 2,
        )

    def represented_project(values):
        # Independent reconstruction of materialized 1/5 and 4/5 entries.
        fifth = Q(float(Q(1, 5)))
        left, right = sum(values[:5]), sum(values[5:])
        return (
            fifth * (left - right),
            fifth * (5 * values[0] - left),
            fifth * (5 * values[5] - right),
        ) + project(values)[3:]

    def lift(values):
        mu, p, q, s, t = values

        def ring(mean, port, shape):
            near, far = mean - port / 4 + shape / 2, mean - port / 4 - shape / 2
            return (mean + port, near, far, far, near)

        return ring(mu / 2, p, s) + ring(-mu / 2, q, t)

    def energy_norm(values):
        form, phase = lift(values[:5]), lift(values[5:])
        square = sum(
            (
                (form[i] - form[j]) ** 2
                + (1 if (i, j) == (0, 5) else Q(c)) * (phase[i] - phase[j]) ** 2
            )
            / 2
            for i, j in prediction["edges"]
        )
        return float(square) ** 0.5

    def hidden_project(values):
        return tuple(
            (values[a] - values[b]) / 2 for a, b in ((1, 4), (2, 3), (6, 9), (7, 8))
        )

    def hidden_norm(values):
        return (
            float(
                sum(
                    (1 if i < 4 else Q(c))
                    * (
                        2 * values[i] ** 2
                        - 2 * values[i] * values[i + 1]
                        + 3 * values[i + 1] ** 2
                    )
                    for i in (0, 2, 4, 6)
                )
            )
            ** 0.5
        )

    for key, trace in report["traces"].items():
        assert trace["completed_steps"] == int(key)
        endpoint = trace["checkpoints"][key]
        x = tuple(map(Q, endpoint["epi"]))
        v = tuple(
            Q(a) - Q(b)
            for a, b in zip(
                endpoint["phase"], prediction["reference_phase"], strict=True
            )
        )
        visible = project(x) + project(v)
        recorded = tuple(map(_rational, endpoint["visible"]))
        assert recorded == represented_project(x) + represented_project(v)
        hidden = hidden_project(x) + hidden_project(v)
        assert hidden == tuple(map(_rational, endpoint["hidden"]))
        hidden_forecast = prediction["forecasts"][key]["binary64"][
            "hidden_second_order"
        ]
        hidden_error = hidden_norm(
            tuple(a - Q(b) for a, b in zip(hidden, hidden_forecast, strict=True))
        )
        assert report["errors"][key]["hidden_error"] == pytest.approx(
            hidden_error, rel=5e-15
        )
        assert report["errors"][key]["hidden_signal"] == pytest.approx(
            hidden_norm(hidden), rel=5e-15
        )
        assert hidden_error < prediction["gates"][
            "maximum_hidden_relative_error"
        ] * hidden_norm(hidden)
        projection_error = tuple(a - b for a, b in zip(visible, recorded, strict=True))
        # The producer evaluates its represented projection matrix exactly;
        # binary64 1/5 and 4/5 differ from these ideal rational means. Every
        # coefficient is below one, so half an ulp is at most 2**-54.
        assert max(map(abs, projection_error)) <= Q(2) ** -54 * sum(map(abs, x + v))
        for name in ("linear", "direct", "memory"):
            expected = prediction["forecasts"][key]["binary64"][name + "_visible"]
            error = tuple(a - Q(b) for a, b in zip(visible, expected, strict=True))
            assert report["errors"][key][name] == pytest.approx(
                energy_norm(error),
                rel=5e-15,
                abs=2 * energy_norm(projection_error) + 1e-25,
            )
        # Assert the frozen scientific result, not an arbitrary literal fixture.
        assert report["errors"][key]["memory"] < 0.1 * min(
            report["errors"][key]["linear"], report["errors"][key]["direct"]
        )
    assert report["passed"] and all(report["checks"].values())

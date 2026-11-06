"""Optional read-only audit of the retained first sine prior/future record.

This imports no producer or forecast solver. It verifies saved bytes, static
initial equations, time chains and the original verdict without regenerating
a prediction or response. File consistency does not authenticate chronology.
"""

import hashlib
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "artifacts/research/relational_sine_prior_forecast/response-v1.json"


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            assert type(value["numerator"]) is type(value["denominator"]) is int
            assert value["denominator"] > 0
            return Q(value["numerator"], value["denominator"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(map(_decode, value))
    return value


def _pair(value):
    assert set(value) == {"lo", "hi"}
    assert value["lo"] <= value["hi"]
    return value["lo"], value["hi"]


def _inside(outer, inner):
    lo, hi = _pair(outer)
    assert lo <= inner[0] <= inner[1] <= hi


def _width(value):
    lo, hi = _pair(value)
    return hi - lo


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _difference(left, right):
    left_lo, left_hi = _pair(left)
    right_lo, right_hi = _pair(right)
    return left_lo - right_hi, left_hi - right_lo


def _static_derivatives(initial, neighbors, visible_capacity, model):
    """Independent forward edge sums and their chain rule at one saved point."""
    size = len(neighbors)
    form, phase = initial[:size], initial[size : 2 * size]
    capacity = visible_capacity + (initial[-1],)
    e = Q(model["epi_weight"])
    w = Q(model["phase_weight"])
    beta = Q(model["storage_scale"])
    pi = pi_interval()
    gradients = [I(0) for _ in neighbors]
    currents = [I(0) for _ in neighbors]
    for i, row in enumerate(neighbors):
        for j in row:
            if i < j:
                gradient = I(form[i] - form[j])
                current = sin(I(phase[j] - phase[i]))
                gradients[i] += gradient
                gradients[j] -= gradient
                currents[i] += current
                currents[j] -= current
    velocity = tuple(
        capacity[i] * (-e * gradients[i] + w * currents[i] / pi) / len(row)
        for i, row in enumerate(neighbors)
    )
    omega = tuple(
        capacity[i] * w * gradients[i] / (beta * pi * len(row))
        for i, row in enumerate(neighbors)
    )
    form_acceleration, phase_acceleration = [], []
    for i, row in enumerate(neighbors):
        gradient_rate = sum((velocity[i] - velocity[j] for j in row), I(0))
        current_rate = sum(
            (cos(I(phase[j] - phase[i])) * (omega[j] - omega[i]) for j in row), I(0)
        )
        form_acceleration.append(
            capacity[i] * (-e * gradient_rate + w * current_rate / pi) / len(row)
        )
        phase_acceleration.append(
            capacity[i] * w * gradient_rate / (beta * pi * len(row))
        )
    return (
        velocity + omega + (I(0),),
        tuple(form_acceleration) + tuple(phase_acceleration) + (I(0),),
    )


def _audit_steps(report, declaration):
    initial = previous = report["initial_box"]
    at = declaration["prior"]["observation_time"]
    assert report["observation_time"] == at
    assert report["end_time"] == declaration["horizon"]
    assert report["time_step"] == declaration["time_step"]
    assert report["order"] == declaration["order"]
    assert len(report["steps"]) == 8
    assert len(initial) == 7
    assert report["neighbors"] == ((2,), (2,), (0, 1))
    assert report["visible_capacity"] == declaration["visible_capacity"]
    assert report["model"] == {
        key: declaration["model"][key]
        for key in ("epi_weight", "phase_weight", "storage_scale")
    } | {"phase_domain": "regular"}
    held = (6, 2, 5) if _pair(initial[-1]) == (0, 0) else (6,)
    for step in report["steps"]:
        assert step["time"] == at
        assert step["duration"] == declaration["time_step"]
        assert step["picard_interior_margin"] > 0
        assert all(margin > 0 for margin in step["domain_lower_bounds"])
        assert all(radius >= 0 for radius in step["propagated_initial_radii"])
        assert len(step["local_remainder_bounds"]) == 7
        for before, tube, after in zip(previous, step["tube"], step["endpoint"]):
            _inside(tube, _pair(before))
            _inside(tube, _pair(after))
        for coordinate in held:
            assert _pair(step["endpoint"][coordinate]) == _pair(initial[coordinate])
        previous = step["endpoint"]
        at += step["duration"]
    assert at == report["validated_end_time"] == declaration["horizon"]
    assert report["endpoint"] == previous
    assert report["status"] == "admitted" and report["failed_tube"] is None
    assert report["steps"][0]["time"] < declaration["prior"]["forecast_start"]


def test_retained_sine_prior_response_hashes_initial_equations_and_original_verdict():
    paths = (
        RECORD,
        RECORD.with_suffix(".protocol.json"),
        RECORD.with_suffix(".prediction.json"),
        RECORD.with_suffix(".source-state.json"),
        RECORD.with_suffix(".source.zip"),
    )
    if not all(path.is_file() for path in paths):
        pytest.skip("retained sine evidence absent; never regenerate it in a test")
    original = tuple(path.read_bytes() for path in paths)
    response, frozen, issued, source = (
        _decode(json_loads(data)) for data in original[:4]
    )
    for record in (issued, response):
        assert record["protocol_sha256"] == _digest(original[1])
        assert record["source_archive_sha256"] == _digest(original[4])
        assert record["execution_error"] is None
    assert response["prediction_sha256"] == _digest(original[2])
    assert response["source_state_sha256"] == _digest(original[3])
    assert frozen["source_state_sha256"] == _digest(original[3])
    _verify_archive(paths[4], frozen["source_sha256"])

    declaration = frozen["declaration"]
    prior = declaration["prior"]
    prediction_result, response_result = issued["result"], response["result"]
    admission = prediction_result["admission"]["report"]
    prediction = prediction_result["prediction"]["report"]
    control = prediction_result["control"]["report"]
    observed = response_result["response"]["report"]
    assert admission["status"] == "admitted"
    assert prior["evidence_window"][1] < prior["forecast_start"]
    assert prediction["prior_admission"] == control["prior_admission"] == admission
    assert prediction["initial_box"] == admission["initial_box"]
    assert control["initial_box"][:-1] == admission["initial_box"][:-1]
    assert _pair(control["initial_box"][-1]) == (0, 0)
    assert control["freeze_hidden"] is True and prediction["freeze_hidden"] is False
    assert admission["initial_box"][-1]["hi"] > admission["initial_box"][-1]["lo"] > 0

    for point in (source["initial"], admission["joint_witness"]):
        for coordinate, box in zip(point, admission["initial_box"]):
            _inside(box, (coordinate, coordinate))
        first, second = _static_derivatives(
            point, source["neighbors"], source["visible_capacity"], prediction["model"]
        )
        for i, port in enumerate(declaration["ports"]):
            for field, row in (
                ("form_rate_bounds", first[i]),
                ("phase_rate_bounds", first[3 + i]),
                ("form_acceleration_bounds", second[i]),
                ("phase_acceleration_bounds", second[3 + i]),
            ):
                lo, hi = prior[field][port]
                assert lo <= row.lo <= row.hi <= hi
    for coordinate, box in zip(source["initial"], observed["initial_box"]):
        assert _pair(box) == (coordinate, coordinate)
    for report in (prediction, control, observed):
        _audit_steps(report, declaration)

    index = declaration["observable_index"]
    forecast, comparison, actual = (
        report["endpoint"][index] for report in (prediction, control, observed)
    )
    assert prediction_result["prediction_bounds"] == forecast
    assert prediction_result["control_bounds"] == comparison
    assert response_result["reserved_observation"] == actual
    prediction_gap = _difference(forecast, comparison)
    response_gap = _difference(actual, comparison)
    _inside(prediction_result["separation_bounds"], prediction_gap)
    _inside(response_result["response_control_separation"], response_gap)
    _inside(forecast, _pair(actual))
    width_limit = declaration["max_endpoint_width"]
    gap_limit = declaration["min_prediction_control_gap"]
    prediction_gates = {
        "prediction_horizon": prediction["validated_end_time"]
        == declaration["horizon"],
        "control_horizon": control["validated_end_time"] == declaration["horizon"],
        "prediction_width": max(map(_width, prediction["endpoint"])) <= width_limit,
        "control_width": max(map(_width, control["endpoint"])) <= width_limit,
        "reserved_observable_separation": prediction_gap[0] > gap_limit,
    }
    response_gates = {
        "response_horizon": observed["validated_end_time"] == declaration["horizon"],
        "response_width": max(map(_width, observed["endpoint"])) <= width_limit,
        "reserved_response_inside_issued_prediction": forecast["lo"]
        <= actual["lo"]
        <= actual["hi"]
        <= forecast["hi"],
        "reserved_response_separated_from_control": response_gap[0] > gap_limit,
    }
    assert prediction_result["gates"] == prediction_gates
    assert response_result["gates"] == response_gates
    assert (
        issued["passed"]
        is prediction_result["passed"]
        is all(prediction_gates.values())
    )
    assert (
        response["passed"] is response_result["passed"] is all(response_gates.values())
    )
    assert tuple(path.read_bytes() for path in paths) == original

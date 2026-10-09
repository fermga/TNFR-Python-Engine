"""Read-only audit of the frozen four-reading software experiment.

Stored derivative enclosures remain a validation premise. Rechecking retained
arithmetic and Picard inclusion neither reruns the producer nor authenticates
execution chronology or physical acquisition.
"""

import ast
import hashlib
import json
import re
import subprocess
import zipfile
from fractions import Fraction as Q
from pathlib import Path, PurePosixPath

import pytest

from tests.physics import test_sine_formed_evidence as retained_helpers
from tnfr.mathematics._rational_interval import I, cos, sin
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.utils.io import json_loads

ROOT = Path(__file__).parents[2]
DIRECTORY = ROOT / "docs/assets/sine_formed_classes"
STEM = "curvature-inference-v1"
HELPERS = "build/curvature-inference-freeze"
PROOF = "theory/nodal/SINE_FINITE_CURVATURE_INFERENCE.md"
PUBLIC_KEYS = {
    "bulk_angle_bounds",
    "receiver_short_angle_bounds",
    "form_radius",
    "phase_radius",
    "phase_increments",
    "probe_duration",
    "recorded_reading_bounds",
    "readout_error_bound",
    "readout_gain_bounds",
    "clock_rate_bounds",
}
_decode = retained_helpers._inference_decode
_interval = retained_helpers._inference_interval


@pytest.fixture(scope="module", autouse=True)
def no_evidence_execution():
    from tnfr.mathematics import _validated_metric, _validated_taylor
    from tnfr.physics import (
        relational_sine_clock_inference,
        relational_sine_curvature_inference,
        relational_sine_two_port_capture,
        relational_sine_two_port_compatibility,
        relational_sine_two_port_dipole,
        relational_sine_two_port_inference,
        relational_sine_two_port_probe,
        relational_sine_two_port_readout,
        relational_sine_two_pulse_inference,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("retained evidence audit must not execute a producer or inverse")

    with pytest.MonkeyPatch.context() as patch:
        for module in (
            relational_sine_clock_inference,
            relational_sine_curvature_inference,
            relational_sine_two_port_capture,
            relational_sine_two_port_compatibility,
            relational_sine_two_port_dipole,
            relational_sine_two_port_inference,
            relational_sine_two_port_probe,
            relational_sine_two_port_readout,
            relational_sine_two_pulse_inference,
        ):
            for name in vars(module):
                if name.startswith(("infer_sine_", "assess_sine_", "bound_sine_")) or (
                    name in {"validated_box_taylor_step", "_full_sine_field"}
                ):
                    patch.setattr(module, name, forbidden)
        for name in (
            "validated_box_taylor_step",
            "validated_taylor_step",
            "flow_jets",
            "picard_tube",
        ):
            patch.setattr(_validated_taylor, name, forbidden)
        patch.setattr(_validated_metric, "validated_metric_taylor_step", forbidden)
        patch.setattr(subprocess, "run", forbidden)
        patch.setattr(subprocess, "Popen", forbidden)
        yield


@pytest.fixture(scope="module")
def evidence():
    names = tuple(
        STEM + suffix
        for suffix in (
            ".json",
            ".protocol.json",
            ".source.zip",
            ".attempt.json",
            ".manifest.json",
        )
    )
    content = {name: (DIRECTORY / name).read_bytes() for name in names}
    raw = json_loads(content[STEM + ".json"])
    protocol = _decode(json_loads(content[STEM + ".protocol.json"]))
    saved = _decode(raw)
    manifest = json_loads(content[STEM + ".manifest.json"])
    yield protocol, saved, raw, manifest, content
    assert all(
        (DIRECTORY / name).read_bytes() == value for name, value in content.items()
    )


def test_archive_inventory_attempt_and_outer_hash_associations(evidence):
    protocol, saved, _, manifest, content = evidence
    entries = manifest["artifacts"]
    assert len(entries) == 4
    assert {entry["file"] for entry in entries} == set(content) - {
        STEM + ".manifest.json"
    }
    for entry in entries:
        value = content[entry["file"]]
        assert type(entry["bytes"]) is int and entry["bytes"] == len(value)
        assert entry["sha256"] == hashlib.sha256(value).hexdigest()
    attempt = json_loads(content[STEM + ".attempt.json"])
    assert attempt["schema"] == "tnfr.sine-curvature-first-attempt.v1"
    assert (
        attempt["protocol_sha256"]
        == hashlib.sha256(content[STEM + ".protocol.json"]).hexdigest()
    )
    assert (
        attempt["source_archive_sha256"]
        == hashlib.sha256(content[STEM + ".source.zip"]).hexdigest()
    )
    with zipfile.ZipFile(DIRECTORY / (STEM + ".source.zip")) as archive:
        source_bytes = archive.read("source-manifest.json")
        source = json_loads(source_bytes)
        expected = {
            PROOF,
            f"docs/assets/sine_formed_classes/{STEM}.protocol.json",
            "src/tnfr/physics/relational_sine_curvature_inference.py",
            "src/tnfr/physics/relational_sine_clock_inference.py",
            "src/tnfr/physics/relational_sine_two_pulse_inference.py",
            "src/tnfr/physics/relational_sine_two_port_readout.py",
            "src/tnfr/mathematics/_validated_taylor.py",
            "src/tnfr/mathematics/_rational_interval.py",
            "tests/physics/test_sine_curvature_inference.py",
            "tests/physics/test_sine_two_port_readout.py",
            "tests/mathematics/test_validated_box_taylor.py",
        } | {
            f"{HELPERS}/{name}.py"
            for name in (
                "prepare_protocol",
                "evaluate_inference",
                "invert_public_packet",
                "archive_source",
                "experiment_support",
                "write_manifest",
            )
        }
        assert len(source["files"]) == len(expected) == 17
        assert {entry["path"] for entry in source["files"]} == expected
        digests = {"source-manifest.json": hashlib.sha256(source_bytes).hexdigest()}
        for entry in source["files"]:
            path = PurePosixPath(entry["path"])
            assert not path.is_absolute() and ".." not in path.parts
            value = archive.read(entry["path"])
            assert len(value) == entry["bytes"]
            digests[entry["path"]] = entry["sha256"]
        assert archive.read(
            f"docs/assets/sine_formed_classes/{STEM}.protocol.json"
        ) == (content[STEM + ".protocol.json"])
    _verify_archive(DIRECTORY / (STEM + ".source.zip"), digests)
    assert source["runtime_overlays"] == manifest["runtime_overlays"] == []
    base = protocol["source_base_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", base)
    assert source["source_base_commit"] == manifest["source_base_commit"] == base
    assert attempt["source_base_commit"] == base
    assert manifest["status"] == saved["report"]["status"]
    assert manifest["frozen_stopping_rule"] == saved["frozen_stopping_rule"]
    assert (
        manifest["frozen_stopping_rule_passed"] is saved["frozen_stopping_rule_passed"]
    )


def test_prospective_prefix_public_worker_and_first_attempt_guard():
    with zipfile.ZipFile(DIRECTORY / (STEM + ".source.zip")) as archive:
        prospective = archive.read(PROOF)
        suffix = retained_helpers._documentation_suffix(
            (ROOT / PROOF).read_bytes(), prospective
        )
        assert suffix is not None
        assert b"sine-curvature-reserved-protocol" in prospective
        assert b"sine-curvature-inference-result" not in prospective
        assert b"sine-curvature-inference-result" in suffix
        helper = ast.parse(archive.read(f"{HELPERS}/experiment_support.py"))
        keys = next(
            node.value
            for node in helper.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PUBLIC_KEYS"
        )
        assert set(ast.literal_eval(keys)) == PUBLIC_KEYS
        tree = ast.parse(archive.read(f"{HELPERS}/invert_public_packet.py"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "infer_sine_geometry_gain_clock_curvature"
        ]
        assert len(calls) == 1
        call = calls[0]
        assert not call.args and len(call.keywords) == 1
        assert call.keywords[0].arg is None
        mapping = call.keywords[0].value
        assert isinstance(mapping, ast.Dict) and mapping.keys == [None, None]
        assert [value.id for value in mapping.values] == ["inputs", "changes"]
        variants = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "variants"
        )
        expected = {
            "primary": None,
            "false_angle": ("bulk_angle_bounds", "false_bulk_angle_bounds"),
            "false_gain": ("readout_gain_bounds", "false_gain_bounds"),
            "false_clock": ("clock_rate_bounds", "false_clock_bounds"),
            "equal_inputs": ("phase_increments", "equal_phase_increments"),
        }
        assert {key.value for key in variants.keys} == set(expected)
        for key, mapping in zip(variants.keys, variants.values):
            if expected[key.value] is None:
                assert mapping.keys == mapping.values == []
            else:
                target_key, source_key = expected[key.value]
                assert len(mapping.keys) == 1 and mapping.keys[0].value == target_key
                assert mapping.values[0].value.id == "controls"
                assert mapping.values[0].slice.value == source_key
        imports = [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        assert "tnfr.physics.relational_sine_curvature_inference" in imports
        assert not any(
            module and any(word in module for word in ("readout", "capture", "dipole"))
            for module in imports
        )
        evaluator = ast.parse(archive.read(f"{HELPERS}/evaluate_inference.py"))
        lock = next(
            node
            for node in evaluator.body
            if isinstance(node, ast.With)
            and isinstance(node.items[0].context_expr, ast.Call)
            and ast.unparse(node.items[0].context_expr.func) == "attempt.open"
        )
        assert lock.items[0].context_expr.args[0].value == "x"
        attempt = next(node for node in evaluator.body if isinstance(node, ast.Try))
        assert lock.lineno < attempt.lineno
        assert len(attempt.handlers) == 1
        assert any(
            isinstance(node, ast.Call) and ast.unparse(node.func) == "export_to_json"
            for node in ast.walk(attempt.handlers[0])
        )
        guard = next(
            node
            for node in evaluator.body
            if isinstance(node, ast.If) and "attempt.exists()" in ast.unparse(node.test)
        )
        assert "target.exists()" in ast.unparse(guard.test)
        assert any(isinstance(node, ast.Raise) for node in guard.body)


def _source(protocol, case, index):
    edges = tuple(
        sorted(
            {tuple(sorted((o + j, o + (j + 1) % 9))) for o in (0, 9) for j in range(9)}
            | {(0, 9), (1, 10)}
        )
    )
    degrees = tuple(sum(i in edge for edge in edges) for i in range(18))
    assert protocol["support"]["nodes"] == tuple(range(18))
    assert protocol["support"]["edges"] == edges
    assert protocol["support"]["degrees"] == degrees
    expected = (
        (Q(45, 32), Q(3, 4), Q(3, 2), Q(1), 0),
        (Q(45, 32), Q(3, 4), Q(6, 5), Q(5, 4), 0),
        (Q(47, 32), Q(11, 12), Q(7, 4), Q(7, 8), 2),
    )
    assert (
        tuple(
            case[key]
            for key in (
                "bulk_angle",
                "receiver_short_angle",
                "hidden_gain",
                "hidden_clock_rate",
                "recipe_index",
            )
        )
        == expected[index]
    )
    recipe = case["recipe_index"]
    assert case["common_form"] == Q(recipe + 2, 11)
    assert case["common_phase"] == -Q(recipe + 2, 13)
    assert case["hidden_offset"] == Q(2 * index + 3, 11)
    assert case["reading_errors"] == tuple(
        Q(value, 2**91) for value in (index - 1, 1 - index, (-1) ** index, index - 1)
    )
    raw_x = tuple(Q((i + 2) * (recipe + 2), 2**62) for i in range(18))
    raw_phase = tuple(Q((5 * i + 3 * recipe) % 23 + 1, 2**62) for i in range(18))
    for row, label in ((raw_x, "form"), (raw_phase, "phase")):
        assert case[f"raw_{label}_residual"] == row
        mean = sum(d * value for d, value in zip(degrees, row)) / 40
        residual = tuple(value - mean for value in row)
        assert case[f"{label}_residual"] == residual and all(residual)
        assert sum(d * value for d, value in zip(degrees, residual)) == 0
        assert sum(d * value**2 for d, value in zip(degrees, residual)) < Q(1, 2**96)
    values = retained_helpers._materialized_inference_source(degrees, case)
    assert values[-2] == case["bulk_angle"] - Q(5, 2**65)
    return values


@pytest.mark.parametrize("index", range(3))
def test_three_retained_segments_carry_all_state_through_passive_observation(
    evidence, index
):
    protocol, saved, _, _, _ = evidence
    case, entry = protocol["hidden_cases"][index], saved["cases"][index]
    degrees, form, phase, x2, y2, actual, actual_box = _source(protocol, case, index)
    public, numerics = (
        protocol["public_inputs_without_observation"],
        protocol["numerics"],
    )
    h, rate = public["probe_duration"], case["hidden_clock_rate"]
    assert (
        entry["source_audit"]["full_form_norm_squared_upper"]
        == x2
        < public["form_radius"] ** 2
    )
    assert (
        entry["source_audit"]["full_phase_norm_squared_upper"]
        == y2
        < public["phase_radius"] ** 2
    )
    assert entry["source_audit"]["actual_pre_probe_arc_mean"] == actual
    assert _interval(entry["actual_arc_mean_source_box"]) == actual_box
    times = (Q(0), h / 2, h, 2 * h)
    assert protocol["observation"]["times"] == entry["observed_reading_times"] == times
    assert entry["structural_reading_times"] == tuple(rate * value for value in times)
    assert entry["structural_segment_starts"] == tuple(
        rate * value for value in times[:3]
    )
    assert numerics["observed_segment_durations"] == (h / 2, h / 2, h)
    assert (
        numerics["phase_increments"]
        == entry["actual_phase_jumps"]
        == (Q(1, 4), Q(0), Q(1, 2))
    )
    assert len(entry["responses"]) == numerics["segments"] == 3
    assert numerics["steps_per_segment"] == 1 and numerics["dimension"] == 36
    carried = form + phase
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in range(18))
    prior = None
    for j, (wrapper, duration, jump) in enumerate(
        zip(entry["responses"], (h / 2, h / 2, h), (Q(1, 4), Q(0), Q(1, 2)))
    ):
        assert wrapper["schema"] == "tnfr.sine-two-port-readout.v1"
        report = wrapper["report"]
        assert tuple(map(_interval, report["initial_form_bounds"])) == carried[:18]
        assert tuple(map(_interval, report["initial_phase_bounds"])) == carried[18:]
        post = carried[:18] + tuple(
            value + jump * direction for value, direction in zip(carried[18:], q)
        )
        assert tuple(map(_interval, report["post_event_initial_box"])) == post
        assert report["phase_increment"] == jump
        assert report["probe_duration"] == rate * duration
        assert report["degrees"] == degrees and report["dipole"] == q
        assert report["geometry"]["nodes"] == protocol["support"]["nodes"]
        assert report["geometry"]["edges"] == protocol["support"]["edges"]
        assert report["capacity"] == (Q(1),) * 18
        assert report["law"] == protocol["complete_model"]["law"]
        assert report["clock"] == "tau=e*t"
        assert report["state_order"] == tuple(f"x_{i}" for i in range(18)) + tuple(
            f"theta_{i}" for i in range(18)
        )
        for field, expected in (
            ("storage_scale", Q(1)),
            ("epi_weight", Q(1023, 1024)),
            ("phase_weight", Q(1, 1024)),
        ):
            value = report["reference_model"][field]
            assert type(value) is float and Q.from_float(value) == expected
        assert report["reference_model"]["phase_domain"] == "regular"
        assert (
            report["step"]["method"] == "direct_source_box_Picard_Taylor_dyadic128_v1"
        )
        retained_helpers._assert_retained_full_box_taylor(
            report, protocol, rate * duration
        )
        if prior is not None:
            assert report["baseline_readout_bounds"] == prior["endpoint_readout_bounds"]
        if j == 1:
            assert post == carried  # No artificial jump at the half-time reading.
        carried = tuple(map(_interval, report["step"]["endpoint"]))
        prior = report


def _audit_clock(inputs, result):
    assert {key: result[key] for key in inputs} == inputs
    rates, gain = inputs["clock_rate_bounds"], inputs["readout_gain_bounds"]
    h = inputs["probe_duration"]
    normalized = {
        key: value for key, value in inputs.items() if key != "clock_rate_bounds"
    }
    normalized["probe_duration"] = rates[1] * h
    normalized["readout_gain_bounds"] = (gain[0] * rates[0] / rates[1], gain[1])
    base = result["constraint_envelope"]
    status = retained_helpers._audit_two_pulse_inverse(normalized, base)
    assert result["structural_probe_duration_bounds"] == (rates[0] * h, rates[1] * h)
    assert result["auxiliary_gain_prior_bounds"] == normalized["readout_gain_bounds"]
    assert result["effective_gain_prior_bounds"] == (
        gain[0] * rates[0],
        gain[1] * rates[1],
    )
    assert (
        result["finite_remainder_over_clock_upper_bound"]
        == base["finite_remainder_upper_bound"] / rates[1]
    )
    assert result["rank_certified"] is base["rank_certified"]
    assert result["rank_deficient"] is base["rank_deficient"]
    if status == "bounded_candidate":
        auxiliary = _interval(base["readout_gain_outer_bounds"])
        lo, hi = rates[1] * auxiliary.lo, rates[1] * auxiliary.hi
        projected_gain = I(max(gain[0], lo / rates[1]), min(gain[1], hi / rates[0]))
        projected_rate = I(max(rates[0], lo / gain[1]), min(rates[1], hi / gain[0]))
        assert _interval(result["effective_gain_outer_bounds"]) == I(lo, hi)
        assert _interval(result["readout_gain_outer_bounds"]) == projected_gain
        assert _interval(result["clock_rate_outer_bounds"]) == projected_rate
        assert (
            result["nominal_bulk_angle_outer_bounds"]
            == base["nominal_bulk_angle_outer_bounds"]
        )
        assert (
            result["actual_long_arc_mean_outer_bounds"]
            == base["actual_long_arc_mean_outer_bounds"]
        )
    else:
        assert result["effective_gain_outer_bounds"] is None
        assert result["readout_gain_outer_bounds"] is None
        assert result["clock_rate_outer_bounds"] is None
    assert result["status"] == status
    return status


def _quotient(numerator, denominator):
    assert 0 < denominator[0] <= denominator[1]
    values = tuple(value / divisor for value in numerator for divisor in denominator)
    return I(min(values), max(values))


def _audit_curvature(inputs, result):
    assert {key: result[key] for key in inputs} == inputs
    selected = dict(inputs)
    selected["recorded_reading_bounds"] = tuple(
        inputs["recorded_reading_bounds"][i] for i in (0, 2, 3)
    )
    child = result["clock_envelope"]
    status = _audit_clock(selected, child)
    base = child["constraint_envelope"]
    x, y, h = (inputs[key] for key in ("form_radius", "phase_radius", "probe_duration"))
    rates, gain = inputs["clock_rate_bounds"], inputs["readout_gain_bounds"]
    a1, g = inputs["phase_increments"][0], Q(1, 3069)
    qmax = base["whole_window_form_norm_candidate"]
    s = min(Q(7), 4 + 4 * a1 + 2 * y + 8 * g * rates[1] * h * qmax)
    eps2 = 4 * (1 + g * g) * x + 4 * g * y
    m3 = 8 * (1 + 2 * g * g) * qmax + 4 * g * (1 + g * g) * s + 8 * g**3 * qmax * qmax
    error = rates[1] * eps2 + rates[1] ** 2 * h * m3 / 2
    assert result["first_window_sine_norm_candidate"] == s
    assert result["initial_curvature_error_candidate"] == eps2
    assert result["third_derivative_bound_candidate"] == m3
    assert (
        result["normalized_curvature_error_candidate"]
        == result["normalized_curvature_error_upper_bound"]
        == error
    )
    centers = tuple((lo + hi) / 2 for lo, hi in inputs["recorded_reading_bounds"])
    radii = tuple((hi - lo) / 2 for lo, hi in inputs["recorded_reading_bounds"])
    midpoint = centers[2] - 2 * centers[1] + centers[0]
    radius = radii[0] + 2 * radii[1] + radii[2] + 4 * inputs["readout_error_bound"]
    assert result["reading_midpoints"] == centers and result["reading_radii"] == radii
    assert result["curvature_midpoint"] == midpoint
    assert result["curvature_observation_radius"] == radius
    assert result["curvature_reading_coefficients"] == (1, -2, 1, 0)
    assert _interval(result["sensor_corrected_curvature_bounds"]) == I(
        midpoint - radius, midpoint + radius
    )
    assert result["source_admitted"] and result["finite_curvature_bound_certified"]
    if base["transformed_reading_coefficients"] is None:
        assert result["embedded_inverse_reading_coefficients"] is None
    else:
        rows = tuple(
            tuple(map(_interval, row))
            for row in base["transformed_reading_coefficients"]
        )
        embedded = tuple((row[0], I(0), row[1], row[2]) for row in rows)
        assert (
            tuple(
                tuple(map(_interval, row))
                for row in result["embedded_inverse_reading_coefficients"]
            )
            == embedded
        )
    if status == "bounded_candidate":
        effective = _interval(child["effective_gain_outer_bounds"])
        denominator = (
            max(effective.lo, gain[0] * rates[0]),
            min(effective.hi, gain[1] * rates[1]),
        )
        angle = _interval(child["nominal_bulk_angle_outer_bounds"])
        coefficient = _interval(base["gamma_bounds"]) * (
            2 * sin(I(a1 / 2)) ** 2 * (1 + 3 * cos(I(a1))) * sin(angle)
            + sin(I(a1)) * (2 + 3 * cos(I(a1))) * cos(angle)
        )
        assert coefficient.lo > 0 and result["curvature_coefficient_positive"]
        assert result["effective_gain_denominator_bounds"] == denominator
        assert _interval(result["curvature_coefficient_bounds"]) == coefficient
        normalized = _quotient(
            (4 * (midpoint - radius) / h**2, 4 * (midpoint + radius) / h**2),
            denominator,
        )
        constraint = _quotient(
            (normalized.lo - error, normalized.hi + error),
            (coefficient.lo, coefficient.hi),
        )
        assert (
            _interval(result["normalized_curvature_observation_bounds"]) == normalized
        )
        assert _interval(result["clock_rate_constraint_bounds"]) == constraint
        coarse_rate = _interval(child["clock_rate_outer_bounds"])
        lo, hi = max(rates[0], coarse_rate.lo, constraint.lo), min(
            rates[1], coarse_rate.hi, constraint.hi
        )
        if lo > hi:
            status = "incompatible"
        else:
            coarse_gain = _interval(child["readout_gain_outer_bounds"])
            gain_lo = max(gain[0], coarse_gain.lo, denominator[0] / hi)
            gain_hi = min(gain[1], coarse_gain.hi, denominator[1] / lo)
            assert gain_lo <= gain_hi
            assert _interval(result["clock_rate_outer_bounds"]) == I(lo, hi)
            assert _interval(result["readout_gain_outer_bounds"]) == I(gain_lo, gain_hi)
            for key in (
                "nominal_bulk_angle_outer_bounds",
                "actual_long_arc_mean_outer_bounds",
                "effective_gain_outer_bounds",
            ):
                assert result[key] == child[key]
    if status != "bounded_candidate":
        for key in (
            "nominal_bulk_angle_outer_bounds",
            "actual_long_arc_mean_outer_bounds",
            "effective_gain_outer_bounds",
            "readout_gain_outer_bounds",
            "clock_rate_outer_bounds",
        ):
            assert result[key] is None
    assert result["status"] == status
    assert result["curvature_refinement_available"] is (status == "bounded_candidate")
    assert result["inverse_enclosure_available"] is (status == "bounded_candidate")
    return status


def test_public_constraints_readings_and_all_stopping_predicates(evidence):
    protocol, saved, raw, manifest, _ = evidence
    assert protocol["schema"] == "tnfr.sine-curvature-inference-protocol.v1"
    assert saved["schema"] == "tnfr.sine-curvature-inference-experiment.v1"
    public = protocol["public_inputs_without_observation"]
    assert set(public) == PUBLIC_KEYS - {"recorded_reading_bounds"}
    assert set(protocol["public_packet"]["inverse_keys"]) == PUBLIC_KEYS
    h, noise = public["probe_duration"], public["readout_error_bound"]
    assert h == Q(1, 2**24) and noise == Q(1, 2**90)
    assert public["form_radius"] == public["phase_radius"] == Q(1, 2**48)
    assert protocol["numerics"]["max_recorded_reading_halfwidth"] == noise
    all_stops = {}
    for index, (case, entry) in enumerate(
        zip(protocol["hidden_cases"], saved["cases"])
    ):
        _, form, phase, x2, y2, actual, actual_box = _source(protocol, case, index)
        gain, rate = case["hidden_gain"], case["hidden_clock_rate"]
        reports = tuple(wrapper["report"] for wrapper in entry["responses"])
        raw_readings = (_interval(reports[0]["baseline_readout_bounds"]),) + tuple(
            _interval(report["endpoint_readout_bounds"]) for report in reports
        )
        readings = tuple(
            gain * value + case["hidden_offset"] + error
            for value, error in zip(raw_readings, case["reading_errors"])
        )
        assert (
            tuple(map(_interval, entry["readings"]["raw_readout_bounds"]))
            == raw_readings
        )
        assert (
            tuple(map(_interval, entry["readings"]["recorded_reading_bounds"]))
            == readings
        )
        increments = (readings[2] - readings[0], readings[3] - readings[2])
        curvature = readings[2] - 2 * readings[1] + readings[0]
        assert (
            tuple(map(_interval, entry["readings"]["recorded_increment_bounds"]))
            == increments
        )
        assert _interval(entry["readings"]["recorded_curvature_bounds"]) == curvature
        packet = entry["public_packet"]
        assert set(packet) == {"schema", "inputs", "controls"}
        assert set(packet["inputs"]) == PUBLIC_KEYS
        assert packet["inputs"] == {
            **public,
            "recorded_reading_bounds": tuple(
                (value.lo, value.hi) for value in readings
            ),
        }
        assert packet["controls"] == protocol["controls"]
        assert packet["schema"] == protocol["public_packet"]["schema"]
        request = json.dumps(
            raw["cases"][index]["public_packet"], sort_keys=True, allow_nan=False
        ).encode("utf-8")
        outputs = entry["inverse_outputs"]
        assert set(outputs) == {
            "request_sha256",
            "primary",
            "false_angle",
            "false_gain",
            "false_clock",
            "equal_inputs",
        }
        changes = {
            "primary": {},
            "false_angle": {
                "bulk_angle_bounds": protocol["controls"]["false_bulk_angle_bounds"]
            },
            "false_gain": {
                "readout_gain_bounds": protocol["controls"]["false_gain_bounds"]
            },
            "false_clock": {
                "clock_rate_bounds": protocol["controls"]["false_clock_bounds"]
            },
            "equal_inputs": {
                "phase_increments": protocol["controls"]["equal_phase_increments"]
            },
        }
        for name, changed in changes.items():
            assert outputs[name]["schema"] == "tnfr.sine-curvature-inference.v1"
            _audit_curvature({**packet["inputs"], **changed}, outputs[name]["report"])
        primary = outputs["primary"]["report"]
        false_angle, false_gain, false_clock, equal = (
            outputs[name]["report"]
            for name in ("false_angle", "false_gain", "false_clock", "equal_inputs")
        )
        stops = {
            "source_admitted": x2 < public["form_radius"] ** 2
            and y2 < public["phase_radius"] ** 2
            and public["bulk_angle_bounds"][0]
            <= case["bulk_angle"]
            <= public["bulk_angle_bounds"][1]
            and public["receiver_short_angle_bounds"][0]
            <= case["receiver_short_angle"]
            <= public["receiver_short_angle_bounds"][1]
            and all(case["form_residual"])
            and all(case["phase_residual"])
            and actual_box.contains(actual),
        }
        carried = form + phase
        for j, (report, duration, jump) in enumerate(
            zip(reports, (h / 2, h / 2, h), (Q(1, 4), Q(0), Q(1, 2)))
        ):
            stops[f"segment_{j+1}_source_association"] = (
                tuple(
                    map(
                        _interval,
                        report["initial_form_bounds"] + report["initial_phase_bounds"],
                    )
                )
                == carried
            )
            stops[f"segment_{j+1}_phase_only_or_zero_event"] = tuple(
                map(_interval, report["post_event_initial_box"])
            ) == carried[:18] + tuple(
                value + jump * q for value, q in zip(carried[18:], report["dipole"])
            )
            stops[f"segment_{j+1}_complete_horizon"] = (
                report["status"] == "admitted"
                and report["step"]["time"] == 0
                and report["step"]["duration"] == rate * duration
                and len(report["step"]["endpoint"]) == 36
                and report["step"]["picard_interior_margin"] > 0
            )
            if j:
                stops[f"segment_{j+1}_shared_baseline"] = (
                    reports[j - 1]["endpoint_readout_bounds"]
                    == report["baseline_readout_bounds"]
                )
            carried = tuple(map(_interval, report["step"]["endpoint"]))
        stops.update(
            {
                "passive_half_window_reading": reports[1]["phase_increment"] == 0
                and reports[1]["post_event_initial_box"]
                == reports[0]["step"]["endpoint"],
                "held_sensor_and_clock_admitted": public["readout_gain_bounds"][0]
                <= gain
                <= public["readout_gain_bounds"][1]
                and public["clock_rate_bounds"][0]
                <= rate
                <= public["clock_rate_bounds"][1]
                and max(map(abs, case["reading_errors"])) <= noise,
                "numerical_reading_halfwidths": all(
                    value.radius <= noise for value in readings
                ),
                "public_only_packet": outputs["request_sha256"]
                == hashlib.sha256(request).hexdigest(),
                "primary_bounded_candidate": primary["status"] == "bounded_candidate",
                "primary_certificates": primary["curvature_refinement_available"]
                and primary["finite_curvature_bound_certified"]
                and primary["curvature_coefficient_positive"]
                and primary["clock_envelope"]["rank_certified"],
                "false_angle_prior_excluded": false_angle["status"] == "incompatible",
                "false_gain_coarse_available": false_gain["clock_envelope"][
                    "inverse_enclosure_available"
                ]
                and false_gain["clock_envelope"]["status"] == "bounded_candidate",
                "false_gain_curvature_excluded": false_gain["status"] == "incompatible",
                "false_clock_coarse_available": false_clock["clock_envelope"][
                    "inverse_enclosure_available"
                ]
                and false_clock["clock_envelope"]["status"] == "bounded_candidate",
                "false_clock_curvature_excluded": false_clock["status"]
                == "incompatible",
                "equal_amplitude_rank_abstention": equal["status"] == "unavailable"
                and equal["clock_envelope"]["rank_deficient"]
                and not equal["clock_envelope"]["rank_certified"],
            }
        )
        truths = {
            "nominal_angle": case["bulk_angle"],
            "actual_angle": actual_box,
            "effective_gain": gain * rate,
            "gain": gain,
            "clock": rate,
        }
        fields = {
            "nominal_angle": "nominal_bulk_angle_outer_bounds",
            "actual_angle": "actual_long_arc_mean_outer_bounds",
            "effective_gain": "effective_gain_outer_bounds",
            "gain": "readout_gain_outer_bounds",
            "clock": "clock_rate_outer_bounds",
        }
        for name, field in fields.items():
            bound = _interval(primary[field])
            stops[f"{name}_covered"] = bound is not None and bound.contains(
                truths[name]
            )
            if name != "nominal_angle":
                stops[f"{name}_resolution"] = (
                    bound is not None
                    and bound.width < protocol["thresholds"][f"primary_{name}_width"]
                )
        heat = (
            2
            * public["readout_gain_bounds"][1]
            * public["clock_rate_bounds"][1]
            * h
            * public["form_radius"]
            + 2 * noise
        )
        assert entry["phase_blind_recorded_bound"] == heat
        for j, value in enumerate(increments):
            stops[f"window_{j+1}_phase_blind_excluded"] = (
                value.hi < -heat or value.lo > heat
            )
        assert len(stops) == 35 and all(stops.values())
        assert entry["source_audit"]["admitted"] is stops["source_admitted"]
        assert entry["stopping_rule"] == stops
        all_stops.update(
            {f"{entry['id']}.{key}": value for key, value in stops.items()}
        )
    first, second = protocol["hidden_cases"][:2]
    source_fields = (
        "bulk_angle",
        "receiver_short_angle",
        "common_form",
        "common_phase",
        "form_residual",
        "phase_residual",
    )
    ca, cb = (
        _interval(entry["readings"]["recorded_curvature_bounds"])
        for entry in saved["cases"][:2]
    )
    paired = {
        "identical_complete_source": all(
            first[key] == second[key] for key in source_fields
        ),
        "equal_effective_gain": first["hidden_gain"] * first["hidden_clock_rate"]
        == second["hidden_gain"] * second["hidden_clock_rate"]
        == Q(3, 2),
        "distinct_held_gain_and_clock": first["hidden_gain"] != second["hidden_gain"]
        and first["hidden_clock_rate"] != second["hidden_clock_rate"],
        "recorded_finite_curvature_separated": ca.hi < cb.lo or cb.hi < ca.lo,
    }
    assert saved["paired_control"] == paired and all(paired.values())
    all_stops.update({f"paired.{key}": value for key, value in paired.items()})
    assert len(saved["cases"]) == 3 and len(all_stops) == 109
    assert (
        saved["frozen_stopping_rule"] == manifest["frozen_stopping_rule"] == all_stops
    )
    assert saved["frozen_stopping_rule_passed"] is all(all_stops.values()) is True
    assert saved["report"]["status"] == "certified_reserved_curvature_inference"

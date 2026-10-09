"""Read-only audit of the frozen nonconstant-clock software experiment.

Stored full-state derivative enclosures remain a validation premise. Rebuilding
their arithmetic, primitive exposure association and inverse constraints neither
reruns a producer nor authenticates chronology or physical acquisition.
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

from tests.physics import test_sine_curvature_evidence as curvature_audit
from tests.physics import test_sine_formed_evidence as retained_helpers
from tnfr.mathematics._rational_interval import I, cos, sin
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.utils.io import json_loads

ROOT = Path(__file__).parents[2]
DIRECTORY = ROOT / "docs/assets/sine_formed_classes"
STEM = "clock-drift-inference-v1"
HELPERS = "build/clock-drift-inference-freeze"
PROOF = "theory/nodal/SINE_CLOCK_DRIFT_INFERENCE.md"
PUBLIC_KEYS = curvature_audit.PUBLIC_KEYS | {"clock_rate_derivative_bound"}
_decode = retained_helpers._inference_decode
_interval = retained_helpers._inference_interval


@pytest.fixture(scope="module", autouse=True)
def no_evidence_execution():
    from tnfr.mathematics import _validated_metric, _validated_taylor
    from tnfr.physics import (
        relational_sine_clock_drift_inference,
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
        pytest.fail("retained audit must not execute a producer, inverse or worker")

    with pytest.MonkeyPatch.context() as patch:
        for module in (
            relational_sine_clock_drift_inference,
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


def test_archive_attempt_and_outer_hash_associations(evidence):
    protocol, saved, _, manifest, content = evidence
    assert protocol["schema"] == "tnfr.sine-clock-drift-inference-protocol.v1"
    assert saved["schema"] == "tnfr.sine-clock-drift-inference-experiment.v1"
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
    assert attempt["schema"] == "tnfr.sine-clock-drift-first-attempt.v1"
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
        paths = [entry["path"] for entry in source["files"]]
        expected = {
            PROOF,
            f"docs/assets/sine_formed_classes/{STEM}.protocol.json",
            "src/tnfr/physics/relational_sine_clock_drift_inference.py",
            "src/tnfr/physics/relational_sine_curvature_inference.py",
            "src/tnfr/physics/relational_sine_clock_inference.py",
            "src/tnfr/physics/relational_sine_two_pulse_inference.py",
            "src/tnfr/physics/relational_sine_two_port_readout.py",
            "src/tnfr/mathematics/_validated_taylor.py",
            "src/tnfr/mathematics/_rational_interval.py",
            "tests/physics/test_sine_clock_drift_inference.py",
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
        assert len(paths) == len(set(paths)) == len(expected) == 19
        assert set(paths) == expected
        digests = {"source-manifest.json": hashlib.sha256(source_bytes).hexdigest()}
        for entry in source["files"]:
            path = PurePosixPath(entry["path"])
            assert not path.is_absolute() and ".." not in path.parts
            value = archive.read(entry["path"])
            assert len(value) == entry["bytes"]
            digests[entry["path"]] = entry["sha256"]
        assert (
            archive.read(f"docs/assets/sine_formed_classes/{STEM}.protocol.json")
            == content[STEM + ".protocol.json"]
        )
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


def test_prospective_prefix_exclusive_attempt_and_public_worker():
    with zipfile.ZipFile(DIRECTORY / (STEM + ".source.zip")) as archive:
        prospective = archive.read(PROOF)
        suffix = retained_helpers._documentation_suffix(
            (ROOT / PROOF).read_bytes(), prospective
        )
        assert suffix is not None
        assert b"sine-clock-drift-reserved-protocol" in prospective
        assert b"sine-clock-drift-result" not in prospective
        assert b"sine-clock-drift-result" in suffix
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
            and node.func.id == "infer_sine_geometry_gain_clock_drift"
        ]
        assert len(calls) == 1
        call = calls[0]
        assert (
            not call.args and len(call.keywords) == 1 and call.keywords[0].arg is None
        )
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
        assert "tnfr.physics.relational_sine_clock_drift_inference" in imports
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


def _audit_drift(inputs, result):
    """Rebuild the transfer and retained inverse constraints from primitives."""
    assert set(inputs) == PUBLIC_KEYS
    assert {key: result[key] for key in inputs} == inputs
    h, x = inputs["probe_duration"], inputs["form_radius"]
    lo, hi = inputs["clock_rate_bounds"]
    gain = inputs["readout_gain_bounds"]
    drift, g = inputs["clock_rate_derivative_bound"], Q(1, 3069)
    assert 0 < lo <= hi and drift >= 0 and 0 < h and hi * h <= Q(1, 2)
    total = 2 * hi * h
    q0 = x + 7 * g * total
    speed = 2 * q0 + 7 * g
    exposure = (
        Q(0),
        min(drift * h**2 / 8, (hi - lo) * h / 4),
        Q(0),
        min(drift * h**2, (hi - lo) * h),
    )
    discrepancy = tuple(gain[1] * speed * value for value in exposure)
    comparison = tuple(
        (a - radius, b + radius)
        for (a, b), radius in zip(inputs["recorded_reading_bounds"], discrepancy)
    )
    selected = {
        key: value
        for key, value in inputs.items()
        if key != "clock_rate_derivative_bound"
    }
    selected["recorded_reading_bounds"] = comparison
    reference = result["reference_envelope"]
    status = curvature_audit._audit_curvature(selected, reference)
    assert result["observation_times"] == (Q(0), h / 2, h, 2 * h)
    assert result["structural_duration_upper_bound"] == total
    assert result["whole_window_form_norm_candidate"] == q0
    assert result["structural_readout_speed_candidate"] == speed
    assert result["exposure_discrepancy_bounds"] == exposure
    assert result["recorded_discrepancy_candidates"] == discrepancy
    assert result["comparison_reading_bounds"] == comparison
    admitted = reference["source_admitted"]
    assert result["source_admitted"] is admitted
    assert result["clock_drift_transfer_certified"] is admitted
    assert result["whole_window_form_norm_upper_bound"] == (q0 if admitted else None)
    assert result["structural_readout_speed_upper_bound"] == (
        speed if admitted else None
    )
    assert result["recorded_discrepancy_upper_bounds"] == (
        discrepancy if admitted else None
    )
    for key in (
        "nominal_bulk_angle_outer_bounds",
        "actual_long_arc_mean_outer_bounds",
        "effective_gain_outer_bounds",
        "readout_gain_outer_bounds",
    ):
        assert result[key] == reference[key]
    assert (
        result["first_window_mean_clock_rate_outer_bounds"]
        == reference["clock_rate_outer_bounds"]
    )
    assert (
        result["finite_curvature_bound_certified"]
        is reference["finite_curvature_bound_certified"]
    )
    for key in ("rank_certified", "rank_deficient"):
        assert result[key] is reference["clock_envelope"][key]
    for key in ("status", "unavailable_reasons", "incompatibility_reasons"):
        assert result[key] == reference[key]
    assert result["inverse_enclosure_available"] is (status == "bounded_candidate")
    assert result["effective_gain_relation"] == "J=G*rho_bar_1"
    assert "clock_rate_outer_bounds" not in result
    return status


def _signed_budget(public, source):
    """Check the prospective sign margin without using a response derivative."""
    h = public["probe_duration"]
    noise = public["readout_error_bound"]
    drift = public["clock_rate_derivative_bound"]
    x, y = public["form_radius"], public["phase_radius"]
    b, gain = source["bulk_angle"], source["hidden_gain"]
    assert (h, drift, noise, x, y) == (
        Q(1, 2**24),
        Q(1, 2**22),
        Q(1, 2**90),
        Q(1, 2**48),
        Q(1, 2**48),
    )
    assert b == Q(23, 16) and gain == Q(7, 5)
    assert public["phase_increments"] == (Q(1, 4), Q(3, 4))
    assert public["clock_rate_bounds"] == (Q(1, 2), Q(2))
    assert public["readout_gain_bounds"] == (Q(1), Q(2))
    z = Q(21, 16)
    assert 1 - z**2 / 2 + z**4 / 24 - z**6 / 720 > Q(1, 4)
    assert Q(3, 8) - Q(3, 8) ** 3 / 6 > Q(1, 3)
    assert Q(9, 8) - Q(9, 8) ** 3 / 6 > Q(1, 2)
    assert 1 - Q(17, 16) ** 2 / 2 == Q(223, 512) > Q(1, 6)
    for amplitude in public["phase_increments"]:
        forcing = 2 * sin(I(3 * amplitude / 2)) * cos(I(b - amplitude / 2))
        assert forcing.lo > Q(1, 6)
    g, total = Q(1, 3069), 4 * h
    q0 = x + 7 * g * total
    phase = y + 2 * g * total * q0
    slope = Q(1, 6 * 3216) - 2 * q0 - 2 * g * phase
    assert slope > Q(1, 20000)
    half_gap = gain * drift * h**2 / (16 * 20000)
    assert half_gap / noise == Q(14336, 3125) > 4
    assert 8 * half_gap > 4 * noise
    assert 4 * half_gap / noise == Q(57344, 3125) > 16
    speed = 2 * q0 + 7 * g
    half_band = 2 * speed * drift * h**2 / 8
    assert half_band / noise == 2**18 * speed > 500
    assert 4 * noise < Q(13, 20) * half_band
    return q0, phase, slope, half_band


def _audit_history(protocol, entry, source_state, durations):
    """Read three full retained certificates and both exact endpoint handoffs."""
    degrees, form, phase = source_state[:3]
    public, numerics = (
        protocol["public_inputs_without_observation"],
        protocol["numerics"],
    )
    h = public["probe_duration"]
    times = (Q(0), h / 2, h, 2 * h)
    structural = (Q(0), durations[0], durations[0] + durations[1], sum(durations))
    jumps = (Q(1, 4), Q(0), Q(1, 2))
    assert entry["observed_reading_times"] == times
    assert entry["structural_reading_times"] == structural
    assert entry["structural_segment_starts"] == structural[:3]
    assert entry["actual_phase_jumps"] == jumps
    assert len(entry["responses"]) == numerics["segments"] == 3
    assert numerics["steps_per_segment"] == 1 and numerics["dimension"] == 36
    carried = form + phase
    dipole = tuple(Q(int(i == 4) - int(i == 5)) for i in range(18))
    previous = None
    for wrapper, duration, jump in zip(entry["responses"], durations, jumps):
        assert duration > 0 and wrapper["schema"] == "tnfr.sine-two-port-readout.v1"
        report = wrapper["report"]
        assert tuple(map(_interval, report["initial_form_bounds"])) == carried[:18]
        assert tuple(map(_interval, report["initial_phase_bounds"])) == carried[18:]
        post = carried[:18] + tuple(
            value + jump * direction for value, direction in zip(carried[18:], dipole)
        )
        assert tuple(map(_interval, report["post_event_initial_box"])) == post
        assert report["phase_increment"] == jump
        assert report["probe_duration"] == duration
        assert report["degrees"] == degrees and report["dipole"] == dipole
        for field in ("nodes", "edges"):
            assert report["geometry"][field] == protocol["support"][field]
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
        retained_helpers._assert_retained_full_box_taylor(report, protocol, duration)
        if previous is not None:
            assert (
                report["baseline_readout_bounds"] == previous["endpoint_readout_bounds"]
            )
        if jump == 0:
            assert post == carried
        carried = tuple(map(_interval, report["step"]["endpoint"]))
        previous = report


@pytest.fixture(scope="module")
def source_state(evidence):
    protocol, _, _, _, _ = evidence
    source = protocol["hidden_source"]
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
    assert protocol["support"]["weights"] == "unit"
    assert protocol["support"]["original_periods"] == (2, 1, 0)
    law = protocol["complete_model"]
    assert law["capacity"] == (Q(1),) * 18
    assert (law["beta"], law["loss"], law["exchange"]) == (
        Q(1),
        Q(1023, 1024),
        Q(1, 1024),
    )
    assert law["gamma"] == "1/(1023*pi)"
    assert law["law"] == "normalized_sine_reciprocal_exchange"
    assert source["bulk_angle"] == Q(23, 16)
    assert source["receiver_short_angle"] == Q(4, 5)
    assert source["recipe_index"] == 4
    assert source["common_form"] == Q(6, 11)
    assert source["common_phase"] == -Q(6, 13)
    raw_x = tuple(Q((i + 2) * 6, 2**62) for i in range(18))
    raw_phase = tuple(Q((5 * i + 12) % 23 + 1, 2**62) for i in range(18))
    assert Q(40 * 114**2, 2**124) < Q(1, 2**96)
    for row, label in ((raw_x, "form"), (raw_phase, "phase")):
        assert source[f"raw_{label}_residual"] == row
        mean = sum(d * value for d, value in zip(degrees, row)) / 40
        residual = tuple(value - mean for value in row)
        assert source[f"{label}_residual"] == residual and all(residual)
        assert sum(d * value for d, value in zip(degrees, residual)) == 0
        assert sum(d * value**2 for d, value in zip(degrees, residual)) < Q(1, 2**96)
    values = retained_helpers._materialized_inference_source(degrees, source)
    assert values[-2] == source["bulk_angle"] - Q(5, 2**65)
    return values


def _linear_profile(profile, h, drift, slope):
    """Rebuild the exact three integrals from their closed-form exposure rule."""
    mean = Q(9, 8)
    assert profile["first_window_mean"] == mean and profile["slope"] == slope
    durations = (
        mean * h / 2 - slope * h**2 / 8,
        mean * h / 2 + slope * h**2 / 8,
        mean * h + slope * h**2,
    )
    cumulative = (Q(0), durations[0], durations[0] + durations[1], sum(durations))
    sampled = (
        mean - slope * h / 2,
        mean,
        mean + slope * h / 2,
        mean + 3 * slope * h / 2,
    )
    lo, hi = min(sampled[0], sampled[-1]), max(sampled[0], sampled[-1])
    assert Q(1, 2) < lo <= hi < 2 and abs(slope) <= drift
    assert all(value > 0 for value in durations)
    assert cumulative[2] == mean * h
    return {
        "definition": "rho(s)=first_window_mean+slope*(s-H/2)",
        "profile_id": profile["id"],
        "first_window_mean": mean,
        "slope": slope,
        "observed_reading_times": (Q(0), h / 2, h, 2 * h),
        "sampled_rates": sampled,
        "global_rate_bounds": (lo, hi),
        "derivative_upper_bound": abs(slope),
        "structural_reading_times": cumulative,
        "structural_segment_durations": durations,
        "admitted": True,
        "first_window_mean_identity": True,
        "signed_half_exposure_difference": -slope * h**2 / 8,
        "signed_final_exposure_difference": slope * h**2,
    }


@pytest.fixture(scope="module")
def clock_records(evidence):
    protocol, _, _, _, _ = evidence
    public = protocol["public_inputs_without_observation"]
    h, drift = public["probe_duration"], public["clock_rate_derivative_bound"]
    profiles = protocol["actual_clock_profiles"]
    assert len(profiles) == 2
    assert tuple(profile["id"] for profile in profiles) == (
        "linear-positive",
        "linear-negative",
    )
    reference = protocol["constant_mean_reference"]
    assert reference["id"] == "constant-mean-reference"
    rows = tuple(
        _linear_profile(profile, h, drift, slope)
        for profile, slope in zip(
            profiles + (reference,), (drift / 2, -drift / 2, Q(0))
        )
    )
    assert protocol["preparation_clock_checks"] == rows
    return rows


def test_primitive_source_profiles_and_prospective_sign_budget(
    evidence, source_state, clock_records
):
    protocol, _, _, _, _ = evidence
    public, sensor = (
        protocol["public_inputs_without_observation"],
        protocol["hidden_sensor"],
    )
    assert set(public) == PUBLIC_KEYS - {"recorded_reading_bounds"}
    assert set(protocol["public_packet"]["inverse_keys"]) == PUBLIC_KEYS
    assert sensor == {
        "gain": Q(7, 5),
        "offset": Q(5, 13),
        "reading_errors": tuple(Q(value, 2**91) for value in (-1, 1, -1, 1)),
    }
    assert max(map(abs, sensor["reading_errors"])) <= public["readout_error_bound"]
    assert source_state[3] < public["form_radius"] ** 2
    assert source_state[4] < public["phase_radius"] ** 2
    _signed_budget(public, {**protocol["hidden_source"], "hidden_gain": sensor["gain"]})
    companion = protocol["equal_exposure_companion"]
    h, drift = public["probe_duration"], public["clock_rate_derivative_bound"]
    epsilon = drift * h / 32
    assert companion["associated_history_id"] == "linear-positive"
    assert companion["epsilon"] == epsilon > 0
    assert companion["sampled_cosine_turns"] == (0, 1, 2, 4)
    # Integer turns make every sine-antiderivative endpoint exactly zero.
    # The same cosine equals one at each sample, changing all sampled rates.
    assert drift / 2 + 4 * Q(22, 7) * epsilon / h < drift
    radius = Q(25, 32) * drift * h
    assert Q(1, 2) < Q(9, 8) - radius < Q(9, 8) + radius < 2
    assert clock_records[0]["signed_half_exposure_difference"] != 0
    assert clock_records[0]["signed_final_exposure_difference"] != 0


@pytest.mark.parametrize("index", range(3))
def test_all_nine_full_state_steps_and_exact_handoffs(
    evidence, source_state, clock_records, index
):
    protocol, saved, _, _, _ = evidence
    histories = saved["cases"] + (saved["reference"],)
    assert len(histories) == 3
    record, entry = clock_records[index], histories[index]
    assert entry["id"] == record["profile_id"]
    assert entry["clock_audit"] == record
    assert protocol["numerics"]["total_step_cap"] == 9
    assert saved["full_flow_steps_attempted"] == 9
    _audit_history(
        protocol, entry, source_state, record["structural_segment_durations"]
    )
    audit = saved["source_audit"]
    _, form, phase, x2, y2, actual, actual_box = source_state
    assert audit["initial_form_bounds"] == tuple((v.lo, v.hi) for v in form)
    assert audit["initial_phase_bounds"] == tuple((v.lo, v.hi) for v in phase)
    assert audit["full_form_norm_squared_upper"] == x2
    assert audit["full_phase_norm_squared_upper"] == y2
    assert audit["actual_pre_probe_arc_mean"] == actual
    assert _interval(saved["actual_arc_mean_source_box"]) == actual_box


def _history_readings_and_stops(protocol, entry, source_state, clock):
    public, sensor = (
        protocol["public_inputs_without_observation"],
        protocol["hidden_sensor"],
    )
    reports = tuple(wrapper["report"] for wrapper in entry["responses"])
    raw = (_interval(reports[0]["baseline_readout_bounds"]),) + tuple(
        _interval(report["endpoint_readout_bounds"]) for report in reports
    )
    readings = tuple(
        sensor["gain"] * value + sensor["offset"] + error
        for value, error in zip(raw, sensor["reading_errors"])
    )
    increments = (readings[2] - readings[0], readings[3] - readings[2])
    curvature = readings[2] - 2 * readings[1] + readings[0]
    assert tuple(map(_interval, entry["readings"]["raw_readout_bounds"])) == raw
    assert (
        tuple(map(_interval, entry["readings"]["recorded_reading_bounds"])) == readings
    )
    assert (
        tuple(map(_interval, entry["readings"]["recorded_increment_bounds"]))
        == increments
    )
    assert _interval(entry["readings"]["recorded_curvature_bounds"]) == curvature
    stops = {
        "clock_profile_admitted": clock["admitted"],
        "first_window_mean_identity": clock["first_window_mean_identity"],
    }
    carried = source_state[1] + source_state[2]
    for index, (report, duration, jump) in enumerate(
        zip(reports, clock["structural_segment_durations"], (Q(1, 4), Q(0), Q(1, 2)))
    ):
        initial = tuple(
            map(
                _interval,
                report["initial_form_bounds"] + report["initial_phase_bounds"],
            )
        )
        stops[f"segment_{index+1}_source_association"] = initial == carried
        stops[f"segment_{index+1}_phase_only_or_zero_event"] = tuple(
            map(_interval, report["post_event_initial_box"])
        ) == carried[:18] + tuple(
            value + jump * direction
            for value, direction in zip(carried[18:], report["dipole"])
        )
        stops[f"segment_{index+1}_complete_horizon"] = (
            report["status"] == "admitted"
            and report["step"]["time"] == 0
            and report["step"]["duration"] == duration
            and len(report["step"]["endpoint"]) == 36
            and report["step"]["picard_interior_margin"] > 0
        )
        if index:
            stops[f"segment_{index+1}_shared_baseline"] = (
                reports[index - 1]["endpoint_readout_bounds"]
                == report["baseline_readout_bounds"]
            )
        carried = tuple(map(_interval, report["step"]["endpoint"]))
    stops["passive_half_window_reading"] = (
        reports[1]["phase_increment"] == 0
        and reports[1]["post_event_initial_box"] == reports[0]["step"]["endpoint"]
    )
    stops["held_sensor_admitted"] = (
        public["readout_gain_bounds"][0]
        <= sensor["gain"]
        <= public["readout_gain_bounds"][1]
        and max(map(abs, sensor["reading_errors"])) <= public["readout_error_bound"]
    )
    assert (
        protocol["numerics"]["max_recorded_reading_halfwidth"]
        == public["readout_error_bound"]
    )
    stops["numerical_reading_halfwidths"] = all(
        value.radius <= public["readout_error_bound"] for value in readings
    )
    return readings, increments, curvature, stops


def test_public_inverse_transfer_companion_and_all_stops(
    evidence, source_state, clock_records
):
    protocol, saved, raw, manifest, _ = evidence
    public, sensor = (
        protocol["public_inputs_without_observation"],
        protocol["hidden_sensor"],
    )
    h, noise, drift = (
        public["probe_duration"],
        public["readout_error_bound"],
        public["clock_rate_derivative_bound"],
    )
    assert public["bulk_angle_bounds"] == (Q(11, 8), Q(3, 2))
    assert public["receiver_short_angle_bounds"] == (Q(2, 3), Q(1))
    assert protocol["controls"] == {
        "false_bulk_angle_bounds": (Q(11, 8), Q(353, 256)),
        "false_gain_bounds": (Q(31, 16), Q(2)),
        "false_clock_bounds": (Q(25, 32), Q(13, 16)),
        "equal_phase_increments": (Q(1, 4), Q(1, 4)),
    }
    assert protocol["thresholds"] == {
        "primary_actual_angle_width": Q(1, 1024),
        "primary_effective_gain_width": Q(1, 2048),
        "primary_gain_width": Q(1, 16),
        "primary_mean_clock_width": Q(1, 80),
    }
    _, form, phase, x2, y2, actual, actual_box = source_state
    source_ok = (
        x2 < public["form_radius"] ** 2
        and y2 < public["phase_radius"] ** 2
        and actual_box.contains(actual)
    )
    assert saved["source_audit"]["admitted"] is source_ok
    all_stops = {"shared.source_admitted": source_ok}
    reference = saved["reference"]
    reference_readings, _, _, reference_stops = _history_readings_and_stops(
        protocol, reference, source_state, clock_records[2]
    )
    assert reference_stops == reference["stopping_rule"] and len(reference_stops) == 16
    assert "public_packet" not in reference and "inverse_outputs" not in reference
    all_stops.update(
        {f"reference.{key}": value for key, value in reference_stops.items()}
    )
    curvatures = []
    for index, entry in enumerate(saved["cases"]):
        clock = clock_records[index]
        readings, increments, curvature, stops = _history_readings_and_stops(
            protocol, entry, source_state, clock
        )
        curvatures.append(curvature)
        packet = entry["public_packet"]
        assert set(packet) == {"schema", "inputs", "controls"}
        assert (
            packet["schema"]
            == protocol["public_packet"]["schema"]
            == "tnfr.sine-clock-drift-public-inference.v1"
        )
        assert packet["inputs"] == {
            **public,
            "recorded_reading_bounds": tuple(
                (value.lo, value.hi) for value in readings
            ),
        }
        assert set(packet["inputs"]) == PUBLIC_KEYS
        assert packet["controls"] == protocol["controls"]
        request = json.dumps(
            raw["cases"][index]["public_packet"], sort_keys=True, allow_nan=False
        ).encode("utf-8")
        outputs = entry["inverse_outputs"]
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
        assert set(outputs) == set(changes) | {"request_sha256"}
        for name, changed in changes.items():
            assert outputs[name]["schema"] == "tnfr.sine-clock-drift-inference.v1"
            _audit_drift({**packet["inputs"], **changed}, outputs[name]["report"])
        primary, false_angle, false_gain, false_clock, equal = (
            outputs[name]["report"]
            for name in (
                "primary",
                "false_angle",
                "false_gain",
                "false_clock",
                "equal_inputs",
            )
        )
        stops["public_only_packet"] = (
            outputs["request_sha256"] == hashlib.sha256(request).hexdigest()
        )
        stops["primary_bounded_candidate"] = primary["status"] == "bounded_candidate"
        stops["primary_certificates"] = (
            primary["clock_drift_transfer_certified"]
            and primary["finite_curvature_bound_certified"]
            and primary["rank_certified"]
            and primary["inverse_enclosure_available"]
        )
        for name, field, truth in (
            ("nominal_angle", "nominal_bulk_angle_outer_bounds", Q(23, 16)),
            ("actual_angle", "actual_long_arc_mean_outer_bounds", actual_box),
            ("effective_gain", "effective_gain_outer_bounds", Q(63, 40)),
            ("gain", "readout_gain_outer_bounds", Q(7, 5)),
            ("mean_clock", "first_window_mean_clock_rate_outer_bounds", Q(9, 8)),
        ):
            bound = None if primary[field] is None else _interval(primary[field])
            stops[f"{name}_covered"] = bound is not None and bound.contains(truth)
            if name != "nominal_angle":
                stops[f"{name}_resolution"] = (
                    bound is not None
                    and bound.width < protocol["thresholds"][f"primary_{name}_width"]
                )
        stops["false_angle_prior_excluded"] = false_angle["status"] == "incompatible"
        for name, control in (("gain", false_gain), ("clock", false_clock)):
            coarse = control["reference_envelope"]["clock_envelope"]
            stops[f"false_{name}_coarse_available"] = (
                coarse["inverse_enclosure_available"]
                and coarse["status"] == "bounded_candidate"
            )
            stops[f"false_{name}_curvature_excluded"] = (
                control["status"] == "incompatible"
            )
        stops["equal_amplitude_rank_abstention"] = (
            equal["status"] == "unavailable"
            and equal["rank_deficient"]
            and not equal["rank_certified"]
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
        for i, increment in enumerate(increments):
            stops[f"window_{i+1}_phase_blind_excluded"] = (
                increment.hi < -heat or increment.lo > heat
            )
        g = Q(1, 3069)
        q0 = public["form_radius"] + 14 * g * public["clock_rate_bounds"][1] * h
        speed = 2 * q0 + 7 * g
        rate_range = public["clock_rate_bounds"][1] - public["clock_rate_bounds"][0]
        allowances = tuple(
            public["readout_gain_bounds"][1] * speed * value
            for value in (
                Q(0),
                min(drift * h**2 / 8, rate_range * h / 4),
                Q(0),
                min(drift * h**2, rate_range * h),
            )
        )
        comparison = entry["reference_comparison"]
        differences = tuple(
            value - base for value, base in zip(readings, reference_readings)
        )
        assert comparison["reference_history_id"] == reference["id"]
        assert (
            tuple(map(_interval, comparison["recorded_difference_bounds"]))
            == differences
        )
        assert comparison["reconstructed_recorded_discrepancy_bounds"] == allowances
        stops["transfer_bounds_match_public_primitives"] = (
            primary["recorded_discrepancy_upper_bounds"] == allowances
        )
        positive = clock["slope"] > 0
        stops["half_reading_signed_separation"] = (
            differences[1].lo > 0 if positive else differences[1].hi < 0
        )
        stops["final_reading_signed_separation"] = (
            differences[3].hi < 0 if positive else differences[3].lo > 0
        )
        for i, label in ((1, "half"), (3, "final")):
            stops[f"{label}_full_difference_within_transfer"] = (
                -allowances[i]
                <= differences[i].lo
                <= differences[i].hi
                <= allowances[i]
            )
        for key, j, location in (
            ("H_full_state_enclosures_intersect", 1, "endpoint"),
            (
                "H_post_event_full_state_enclosures_intersect",
                2,
                "post_event_initial_box",
            ),
        ):
            left_report, right_report = (
                entry["responses"][j]["report"],
                reference["responses"][j]["report"],
            )
            left = (
                left_report["step"][location]
                if location == "endpoint"
                else left_report[location]
            )
            right = (
                right_report["step"][location]
                if location == "endpoint"
                else right_report[location]
            )
            pairs = tuple(zip(map(_interval, left), map(_interval, right)))
            assert len(pairs) == 36
            stops[key] = all(max(a.lo, b.lo) <= min(a.hi, b.hi) for a, b in pairs)
        assert stops == entry["stopping_rule"] and len(stops) == 43
        all_stops.update(
            {f"{entry['id']}.{key}": value for key, value in stops.items()}
        )
    paired = {
        "actual_curvatures_signed_separation": curvatures[0].hi < curvatures[1].lo
    }
    assert saved["paired_control"] == paired
    all_stops.update({f"paired.{key}": value for key, value in paired.items()})
    companion = saved["equal_exposure_companion"]
    declared = protocol["equal_exposure_companion"]
    positive = saved["cases"][0]
    clock = clock_records[0]
    epsilon = drift * h / 32
    bounds = (
        clock["global_rate_bounds"][0] - epsilon,
        clock["global_rate_bounds"][1] + epsilon,
    )
    derivative = drift / 2 + 16 * epsilon / h
    turns = tuple(2 * time / h for time in protocol["observation"]["times"])
    assert turns == (0, 1, 2, 4) and all(value.denominator == 1 for value in turns)
    sampled = tuple(value + epsilon for value in clock["sampled_rates"])
    assert (
        companion["associated_history_id"]
        == declared["associated_history_id"]
        == positive["id"]
    )
    assert companion["definition"] == declared["definition"]
    assert companion["epsilon"] == epsilon
    assert companion["global_rate_bounds"] == bounds
    assert companion["derivative_upper_bound_using_pi_less_than_four"] == derivative
    assert (
        companion["sampled_rates"] == sampled
        and companion["sampled_rate_differences"] == (epsilon,) * 4
    )
    assert (
        companion["cosine_turns"] == turns
        and companion["exact_exposure_differences"] == (Q(0),) * 4
    )
    assert companion["structural_reading_times"] == clock["structural_reading_times"]
    assert (
        companion["structural_segment_durations"]
        == clock["structural_segment_durations"]
    )
    assert (
        companion["source_association"] == "hidden_source"
        and companion["sensor_association"] == "hidden_sensor"
    )
    assert "responses" not in companion and "inverse_outputs" not in companion
    initial = positive["responses"][0]["report"]
    companion_stops = {
        "associated_positive_history": positive["id"]
        == declared["associated_history_id"],
        "nonzero_companion_amplitude": epsilon > 0,
        "global_positive_rate_admitted": 0
        < public["clock_rate_bounds"][0]
        <= bounds[0]
        <= bounds[1]
        <= public["clock_rate_bounds"][1],
        "derivative_admitted_by_pi_less_than_four": derivative <= drift,
        "integer_cosine_turns_at_all_readings": turns
        == declared["sampled_cosine_turns"]
        and all(value.denominator == 1 for value in turns),
        "distinct_sampled_rates": all(
            value != base for value, base in zip(sampled, clock["sampled_rates"])
        ),
        "same_all_three_segment_exposures": companion["structural_segment_durations"]
        == clock["structural_segment_durations"]
        and all(value > 0 for value in clock["structural_segment_durations"]),
        "same_complete_source_and_events": tuple(
            map(
                _interval,
                initial["initial_form_bounds"] + initial["initial_phase_bounds"],
            )
        )
        == form + phase
        and positive["actual_phase_jumps"] == protocol["numerics"]["phase_increments"],
    }
    assert companion["stopping_rule"] == companion_stops
    all_stops.update(
        {f"companion.{key}": value for key, value in companion_stops.items()}
    )
    assert len(all_stops) == 112 and all(value is True for value in all_stops.values())
    assert (
        saved["frozen_stopping_rule"] == manifest["frozen_stopping_rule"] == all_stops
    )
    assert (
        saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is True
    )
    assert saved["report"]["status"] == "certified_reserved_clock_drift_inference"

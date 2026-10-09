"""Read-only arithmetic audit of the frozen finite-aperture experiment.

Stored higher derivative enclosures remain an explicit evidence premise.
No producer, inverse, worker or archived helper is replayed. Hash associations
and an exclusive attempt record do not authenticate physical acquisition.
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

# Import modules as namespaces; do not import their autouse fixtures into this
# module, where pytest would collect them and leak unrelated patch contexts.
from tests.physics import test_sine_curvature_evidence as curvature_audit
from tests.physics import test_sine_formed_evidence as retained_helpers
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.research.relational_acquisition import _verify_archive
from tnfr.utils.io import json_loads

ROOT = Path(__file__).parents[2]
DIRECTORY = ROOT / "docs/assets/sine_formed_classes"
STEM = "aperture-inference-v1"
HELPERS = "build/aperture-inference-freeze"
PROOF = "theory/nodal/SINE_APERTURE_INFERENCE.md"
PUBLIC_KEYS = (curvature_audit.PUBLIC_KEYS - {"recorded_reading_bounds"}) | {
    "averaged_reading_bounds",
    "clock_rate_derivative_bound",
}
_decode = retained_helpers._inference_decode
_interval = retained_helpers._inference_interval
MOMENTS = (
    (Q(11, 6), Q(-7, 6), Q(1, 3), Q(0)),
    (Q(-1, 24), Q(13, 12), Q(-1, 24), Q(0)),
    (Q(1, 3), Q(-7, 6), Q(11, 6), Q(0)),
    (Q(-1, 3), Q(7, 6), Q(-11, 6), Q(2)),
)


@pytest.fixture(scope="module", autouse=True)
def no_evidence_execution():
    from tnfr.mathematics import _validated_metric, _validated_taylor
    from tnfr.physics import (
        relational_sine_aperture_inference,
        relational_sine_aperture_readout,
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
            relational_sine_aperture_inference,
            relational_sine_aperture_readout,
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
                if name.startswith(
                    ("infer_sine_", "assess_sine_", "bound_sine_")
                ) or name in {
                    "validated_box_taylor_step",
                    "_full_sine_field",
                    "_affine_clock_sine_field",
                }:
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


def test_archive_inventory_first_attempt_and_outer_associations(evidence):
    protocol, saved, _, manifest, content = evidence
    assert protocol["schema"] == "tnfr.sine-aperture-inference-protocol.v1"
    assert saved["schema"] == "tnfr.sine-aperture-inference-experiment.v1"
    assert manifest["schema"] == "tnfr.sine-aperture-inference-evidence.v1"
    assert len(manifest["artifacts"]) == 4
    assert {v["file"] for v in manifest["artifacts"]} == set(content) - {
        STEM + ".manifest.json"
    }
    for entry in manifest["artifacts"]:
        value = content[entry["file"]]
        assert type(entry["bytes"]) is int and entry["bytes"] == len(value)
        assert entry["sha256"] == hashlib.sha256(value).hexdigest()
    attempt = json_loads(content[STEM + ".attempt.json"])
    assert attempt["schema"] == "tnfr.sine-aperture-first-attempt.v1"
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
            *(
                "src/tnfr/physics/" + name + ".py"
                for name in (
                    "relational_sine_aperture_inference",
                    "relational_sine_aperture_readout",
                    "relational_sine_curvature_inference",
                    "relational_sine_clock_inference",
                    "relational_sine_two_pulse_inference",
                    "relational_sine_two_port_inference",
                    "relational_sine_two_port_readout",
                    "relational_sine_two_port_compatibility",
                    "relational_sine_comparison",
                    "_sine_admission",
                    "phase_cycle_geometry",
                    "relational_observations",
                )
            ),
            "src/tnfr/dynamics/relational.py",
            "src/tnfr/_exact_time.py",
            *(
                "src/tnfr/mathematics/" + name + ".py"
                for name in (
                    "_validated_taylor",
                    "_interval_taylor",
                    "_rational_interval",
                    "_exact_linear_algebra",
                )
            ),
            *(
                "tests/physics/test_sine_" + name + ".py"
                for name in (
                    "aperture_inference",
                    "aperture_readout",
                    "curvature_inference",
                )
            ),
            "tests/mathematics/test_validated_box_taylor.py",
            *(
                f"{HELPERS}/{name}.py"
                for name in (
                    "experiment_support",
                    "prepare_protocol",
                    "evaluate_inference",
                    "invert_public_packet",
                    "archive_source",
                    "write_manifest",
                )
            ),
        }
        paths = [entry["path"] for entry in source["files"]]
        assert len(paths) == len(set(paths)) == len(expected) == 30
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
    base = protocol["source_base_commit"]
    assert base == "035965355a8066622eebabd6fe08eedee9c7aef5"
    assert re.fullmatch(r"[0-9a-f]{40}", base)
    assert (
        source["source_base_commit"]
        == manifest["source_base_commit"]
        == attempt["source_base_commit"]
        == base
    )
    assert source["runtime_overlays"] == manifest["runtime_overlays"] == []
    assert manifest["status"] == saved["report"]["status"]
    assert manifest["frozen_stopping_rule"] == saved["frozen_stopping_rule"]
    assert (
        manifest["frozen_stopping_rule_passed"] is saved["frozen_stopping_rule_passed"]
    )


def test_prospective_prefix_public_worker_and_exclusive_attempt():
    with zipfile.ZipFile(DIRECTORY / (STEM + ".source.zip")) as archive:
        prospective = archive.read(PROOF)
        suffix = retained_helpers._documentation_suffix(
            (ROOT / PROOF).read_bytes(), prospective
        )
        assert suffix is not None
        assert b"sine-aperture-reserved-protocol" in prospective
        assert b"sine-aperture-result" not in prospective
        assert b"sine-aperture-result" in suffix
        helper = ast.parse(archive.read(f"{HELPERS}/experiment_support.py"))
        keys = next(
            node.value
            for node in helper.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PUBLIC_KEYS"
        )
        assert (
            set(ast.literal_eval(keys)) == PUBLIC_KEYS
            and len(ast.literal_eval(keys)) == 11
        )
        tree = ast.parse(archive.read(f"{HELPERS}/invert_public_packet.py"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "infer_sine_geometry_gain_clock_aperture"
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
        for key, value in zip(variants.keys, variants.values):
            if expected[key.value] is None:
                assert value.keys == value.values == []
            else:
                target, origin = expected[key.value]
                assert len(value.keys) == 1 and value.keys[0].value == target
                assert (
                    value.values[0].value.id == "controls"
                    and value.values[0].slice.value == origin
                )
        imports = [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        assert "tnfr.physics.relational_sine_aperture_inference" in imports
        assert not any(
            module and any(word in module for word in ("readout", "capture", "dipole"))
            for module in imports
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"open", "read_bytes", "read_text"}
            for node in ast.walk(tree)
        )
        evaluator = ast.parse(archive.read(f"{HELPERS}/evaluate_inference.py"))
        lock = next(
            node
            for node in evaluator.body
            if isinstance(node, ast.With)
            and ast.unparse(node.items[0].context_expr.func) == "attempt.open"
        )
        assert lock.items[0].context_expr.args[0].value == "x"
        run = next(node for node in evaluator.body if isinstance(node, ast.Try))
        assert lock.lineno < run.lineno and len(run.handlers) == 1
        assert any(
            isinstance(node, ast.Call) and ast.unparse(node.func) == "export_to_json"
            for node in ast.walk(run.handlers[0])
        )
        guard = next(
            node
            for node in evaluator.body
            if isinstance(node, ast.If) and "attempt.exists()" in ast.unparse(node.test)
        )
        assert "target.exists()" in ast.unparse(guard.test)
        assert any(isinstance(node, ast.Raise) for node in guard.body)


def _audit_aperture(inputs, result):
    """Rebuild raw-to-virtual constraints before rechecking the retained child."""
    assert set(inputs) == PUBLIC_KEYS
    assert {key: result[key] for key in inputs} == inputs
    h, x = inputs["probe_duration"], inputs["form_radius"]
    lo, hi = inputs["clock_rate_bounds"]
    gain, drift, noise = (
        inputs[k]
        for k in (
            "readout_gain_bounds",
            "clock_rate_derivative_bound",
            "readout_error_bound",
        )
    )
    g, hs = Q(1, 3069), hi * h
    q0 = x + 14 * g * hs
    speed = 2 * q0 + 7 * g
    m2 = 4 * (1 + g * g) * q0 + 14 * g
    m3 = 8 * (1 + 2 * g * g) * q0 + 28 * g * (1 + g * g) + 8 * g**3 * q0**2
    exposure = tuple(
        min(c * drift * h**2, 2 * c * (hi - lo) * h)
        for c in (Q(7, 108), Q(13, 108), Q(7, 108))
    ) + (
        min(5 * drift * h**2 / 12, (hi - lo) * h / 2),
    )
    clock_average = tuple(gain[1] * speed * value for value in exposure)
    centers = tuple((a + b) / 2 for a, b in inputs["averaged_reading_bounds"])
    radii = tuple((b - a) / 2 for a, b in inputs["averaged_reading_bounds"])

    def project(values):
        return tuple(
            sum(abs(w) * value for w, value in zip(row, values)) for row in MOMENTS
        )

    virtual = tuple(sum(w * c for w, c in zip(row, centers)) for row in MOMENTS)
    numerical, sensor, clock = (
        project(radii),
        project((noise,) * 4),
        project(clock_average),
    )
    end = gain[1] * m3 * hs**3 / 108
    reconstruction = (
        end,
        gain[1] * m3 * hs**3 / 2304,
        end,
        end + gain[1] * m2 * hs**2 / 6,
    )
    point_radii = tuple(
        sum(parts) for parts in zip(numerical, sensor, clock, reconstruction)
    )
    bands = tuple((c - radius, c + radius) for c, radius in zip(virtual, point_radii))
    selected = {
        k: v
        for k, v in inputs.items()
        if k not in {"averaged_reading_bounds", "clock_rate_derivative_bound"}
    }
    selected.update(recorded_reading_bounds=bands, readout_error_bound=Q(0))
    child = result["reference_envelope"]
    status = curvature_audit._audit_curvature(selected, child)
    assert result["reconstruction_matrix"] == MOMENTS
    assert result["aperture_windows"] == (
        (0, h / 3),
        (h / 3, 2 * h / 3),
        (2 * h / 3, h),
        (h, 2 * h),
    )
    assert result["virtual_observation_times"] == (0, h / 2, h, 2 * h)
    assert result["structural_duration_upper_bound"] == 2 * hs
    assert (
        result["averaged_reading_midpoints"] == centers
        and result["averaged_reading_radii"] == radii
    )
    assert (
        result["virtual_reading_midpoints"] == virtual
        and result["point_reference_reading_bounds"] == bands
    )
    admitted = child["source_admitted"]
    for name, expected in (
        ("whole_window_form_norm", q0),
        ("structural_readout_speed", speed),
        ("second_derivative_bound", m2),
        ("third_derivative_bound", m3),
    ):
        assert result[name + "_candidate"] == expected
        assert result[name + "_upper_bound"] == (expected if admitted else None)
    assert result["averaged_exposure_discrepancy_bounds"] == exposure
    for name, expected in (
        ("averaged_clock_discrepancy", clock_average),
        ("projected_clock_discrepancy", clock),
        ("reconstruction_error", reconstruction),
    ):
        assert result[name + "_candidates"] == expected
        assert result[name + "_upper_bounds"] == (expected if admitted else None)
    assert result["projected_numerical_radii"] == numerical
    assert result["projected_sensor_error_radii"] == sensor
    curvature_row = tuple(
        MOMENTS[0][j] - 2 * MOMENTS[1][j] + MOMENTS[2][j] for j in range(4)
    )
    assert (
        result["original_curvature_average_coefficients"]
        == curvature_row
        == (Q(9, 4), Q(-9, 2), Q(9, 4), Q(0))
    )
    if child["embedded_inverse_reading_coefficients"] is None:
        assert result["embedded_inverse_average_coefficients"] is None
    else:
        rows = tuple(
            tuple(map(_interval, row))
            for row in child["embedded_inverse_reading_coefficients"]
        )
        embedded = tuple(
            tuple(
                sum((row[k] * MOMENTS[k][j] for k in range(4)), I(0)) for j in range(4)
            )
            for row in rows
        )
        assert (
            tuple(
                tuple(map(_interval, row))
                for row in result["embedded_inverse_average_coefficients"]
            )
            == embedded
        )
    for key in (
        "source_admitted",
        "clock_drift_transfer_certified",
        "aperture_reconstruction_certified",
    ):
        assert result[key] is admitted
    assert (
        result["finite_curvature_bound_certified"]
        is child["finite_curvature_bound_certified"]
    )
    for key in ("rank_certified", "rank_deficient"):
        assert result[key] is child["clock_envelope"][key]
    for key in (
        "nominal_bulk_angle_outer_bounds",
        "actual_long_arc_mean_outer_bounds",
        "effective_gain_outer_bounds",
        "readout_gain_outer_bounds",
    ):
        assert result[key] == child[key]
    assert (
        result["first_window_mean_clock_rate_outer_bounds"]
        == child["clock_rate_outer_bounds"]
    )
    assert "clock_rate_outer_bounds" not in result
    assert result["effective_gain_relation"] == "J=G*rho_bar_1"
    for key in ("status", "unavailable_reasons", "incompatibility_reasons"):
        assert result[key] == child[key]
    assert result["inverse_enclosure_available"] is (status == "bounded_candidate")
    return status


@pytest.fixture(scope="module")
def source_state(evidence):
    protocol, saved, _, _, _ = evidence
    source, public = (
        protocol["hidden_source"],
        protocol["public_inputs_without_observation"],
    )
    edges = tuple(
        sorted(
            {tuple(sorted((o + j, o + (j + 1) % 9))) for o in (0, 9) for j in range(9)}
            | {(0, 9), (1, 10)}
        )
    )
    degrees = tuple(sum(i in edge for edge in edges) for i in range(18))
    assert all(type(v) is int for v in protocol["support"]["nodes"])
    assert protocol["support"]["nodes"] == tuple(range(18))
    assert (
        protocol["support"]["edges"] == edges
        and protocol["support"]["degrees"] == degrees
    )
    assert protocol["support"]["weights"] == "unit" and protocol["support"][
        "original_periods"
    ] == (2, 1, 0)
    law = protocol["complete_model"]
    assert law["capacity"] == (Q(1),) * 18
    assert all(type(v) is Q for v in law["capacity"])
    assert (law["beta"], law["loss"], law["exchange"]) == (
        Q(1),
        Q(1023, 1024),
        Q(1, 1024),
    )
    assert (
        law["law"] == "normalized_sine_reciprocal_exchange"
        and law["gamma"] == "1/(1023*pi)"
    )
    assert (
        law["observed_augmented_rows"]
        == "x_s=rho*(-A*x+gamma*f(theta)); theta_s=rho*gamma*A*x; rho_s=slope; z_s=x4-x5"
    )
    assert source["bulk_angle"] == Q(23, 16) and source["receiver_short_angle"] == Q(
        4, 5
    )
    assert type(source["recipe_index"]) is int and source["recipe_index"] == 4
    assert source["common_form"] == Q(6, 11) and source["common_phase"] == -Q(6, 13)
    raw_x = tuple(Q((i + 2) * 6, 2**62) for i in range(18))
    raw_phase = tuple(Q((5 * i + 12) % 23 + 1, 2**62) for i in range(18))
    assert Q(40 * 114**2, 2**124) < Q(1, 2**96)
    for raw, name in ((raw_x, "form"), (raw_phase, "phase")):
        assert source[f"raw_{name}_residual"] == raw
        mean = sum(d * v for d, v in zip(degrees, raw)) / 40
        residual = tuple(v - mean for v in raw)
        assert source[f"{name}_residual"] == residual and all(residual)
        assert sum(d * v for d, v in zip(degrees, residual)) == 0
        assert sum(d * v * v for d, v in zip(degrees, residual)) < Q(1, 2**96)
    values = retained_helpers._materialized_inference_source(degrees, source)
    _, form, phase, x2, y2, actual, actual_box = values
    assert actual == source["bulk_angle"] - Q(5, 2**65)
    assert x2 < public["form_radius"] ** 2 and y2 < public["phase_radius"] ** 2
    audit = saved["source_audit"]
    assert audit["initial_form_bounds"] == tuple((v.lo, v.hi) for v in form)
    assert audit["initial_phase_bounds"] == tuple((v.lo, v.hi) for v in phase)
    assert (
        audit["full_form_norm_squared_upper"] == x2
        and audit["full_phase_norm_squared_upper"] == y2
    )
    assert audit["actual_pre_probe_arc_mean"] == actual and audit["admitted"] is True
    assert _interval(saved["actual_arc_mean_source_box"]) == actual_box
    return values


def _clock_record(profile, public):
    h, mean, slope = (
        public["probe_duration"],
        profile["first_window_mean"],
        profile["slope"],
    )
    initial = mean - slope * h / 2
    times = (Q(0), h / 3, 2 * h / 3, h, 2 * h)
    windows = tuple(zip(times, times[1:]))
    rates = tuple(initial + slope * time for time in times)
    exposure = tuple(initial * time + slope * time**2 / 2 for time in times)
    segments = tuple(b - a for a, b in zip(exposure, exposure[1:]))

    def primitive(t):
        return slope * (t**3 / 6 - h * t**2 / 4)

    signed = tuple((primitive(b) - primitive(a)) / (b - a) for a, b in windows)
    lower, upper = min(rates[0], rates[-1]), max(rates[0], rates[-1])
    admitted = (
        public["clock_rate_bounds"][0]
        <= lower
        <= upper
        <= public["clock_rate_bounds"][1]
        and lower > 0
        and upper * h <= Q(1, 2)
        and abs(slope) <= public["clock_rate_derivative_bound"]
        and min(segments) > 0
    )
    return {
        "definition": "rho(s)=first_window_mean+slope*(s-H/2)",
        "profile_id": profile["id"],
        "first_window_mean": mean,
        "initial_clock_rate": initial,
        "slope": slope,
        "observed_window_boundaries": times,
        "aperture_windows": windows,
        "sampled_rates": rates,
        "global_rate_bounds": (lower, upper),
        "derivative_upper_bound": abs(slope),
        "cumulative_structural_exposures": exposure,
        "structural_segment_durations": segments,
        "signed_averaged_exposure_differences": signed,
        "admitted": admitted,
        "first_window_mean_identity": exposure[3] == mean * h,
    }


def test_primitive_clock_and_response_free_separation_budget(evidence, source_state):
    protocol, _, _, _, _ = evidence
    public, source, sensor = (
        protocol[k]
        for k in ("public_inputs_without_observation", "hidden_source", "hidden_sensor")
    )
    h, drift, noise = (
        public[k]
        for k in (
            "probe_duration",
            "clock_rate_derivative_bound",
            "readout_error_bound",
        )
    )
    assert (h, drift, noise) == (Q(1, 2**24), Q(1, 2**22), Q(1, 2**90))
    assert public["form_radius"] == public["phase_radius"] == Q(1, 2**48)
    assert public["phase_increments"] == (Q(1, 4), Q(3, 4))
    assert public["clock_rate_bounds"] == (Q(1, 2), Q(2))
    assert public["readout_gain_bounds"] == (Q(1), Q(2))
    assert (sensor["gain"], sensor["offset"]) == (Q(7, 5), Q(5, 13))
    assert sensor["reading_errors"] == tuple(
        sign * noise / 2 for sign in (-1, 1, -1, 1)
    )
    profiles = protocol["actual_clock_profiles"] + (
        protocol["constant_mean_reference"],
    )
    assert len(profiles) == 3
    for profile, slope, retained in zip(
        profiles, (drift / 2, -drift / 2, Q(0)), protocol["preparation_clock_checks"]
    ):
        assert profile["first_window_mean"] == Q(9, 8) and profile["slope"] == slope
        assert retained == _clock_record(profile, public)
        assert retained["admitted"] and retained["first_window_mean_identity"]
    g, total = Q(1, 3069), 4 * h
    q0 = public["form_radius"] + 7 * g * total
    phase = public["phase_radius"] + 2 * g * total * q0
    for amplitude in public["phase_increments"]:
        forcing = (
            2 * sin(I(3 * amplitude / 2)) * cos(I(source["bulk_angle"] - amplitude / 2))
        )
        assert forcing.lo > Q(1, 6)
    assert (1 / (1023 * pi_interval())).lo > Q(1, 3216)
    m = Q(1, 6 * 3216) - 2 * q0 - 2 * g * phase
    assert m > Q(1, 20000)
    m2 = 4 * (1 + g * g) * q0 + 14 * g
    assert m2 * total < Q(1, 10**8)
    coefficients = (Q(7, 108), Q(13, 108), Q(7, 108), Q(5, 12))
    lower = tuple(sensor["gain"] * drift * h**2 * c / 40000 for c in coefficients)
    assert (
        tuple(v / noise for v in lower)
        == protocol["prospective_separation"][
            "true_average_separation_lower_in_delta_units"
        ]
    )
    radius = protocol["numerics"]["max_recorded_average_halfwidth"]
    assert radius == noise / 16 and min(lower) > 4 * radius
    curvature = sensor["gain"] * drift * h**2 * (Q(1, 80000) - Q(5, 6) * m2 * total)
    conservative = sensor["gain"] * drift * h**2 / 100000
    assert curvature > conservative > 36 * radius
    assert (
        conservative / noise
        == protocol["prospective_separation"][
            "true_negative_minus_positive_curvature_lower_in_delta_units"
        ]
    )
    speed = 2 * q0 + 7 * g
    allowances = tuple(
        public["readout_gain_bounds"][1] * speed * drift * h**2 * c
        for c in coefficients
    )
    assert min(allowances) / noise > 300
    assert sensor["gain"] / (2 * public["readout_gain_bounds"][1]) == Q(7, 20)
    assert all(Q(7, 20) * value + 4 * radius < value for value in allowances)
    assert (
        sensor["gain"] * h / 120000
        > 2 * public["readout_gain_bounds"][1] * public["form_radius"]
        + 4 * noise
        + 4 * radius
    )
    # The first three moment rows reproduce degree-two polynomials exactly.
    for power in range(3):
        averages = tuple(
            (b ** (power + 1) - a ** (power + 1)) / ((power + 1) * (b - a))
            for a, b in ((Q(0), Q(1, 3)), (Q(1, 3), Q(2, 3)), (Q(2, 3), Q(1)))
        )
        for row, time in zip(MOMENTS[:3], (Q(0), Q(1, 2), Q(1))):
            assert sum(w * value for w, value in zip(row, averages)) == time**power


def _audit_step(step, initial, protocol, start, duration, slope):
    """Check retained 38D arithmetic and independently assemble tube rates."""
    assert step["time"] == start and step["duration"] == duration
    assert step["order"] == protocol["numerics"]["order"] == 4
    assert protocol["numerics"]["interval_bits"] == 128
    assert step["method"] == "direct_source_box_Picard_Taylor_dyadic128_v1"
    assert tuple(map(_interval, step["initial_box"])) == initial
    tube = tuple(map(_interval, step["tube"]))
    series = tuple(tuple(map(_interval, row)) for row in step["series"])
    remainder = tuple(map(_interval, step["local_remainder_bounds"]))
    assert len(initial) == len(tube) == len(series) == len(remainder) == 38
    assert all(len(row) == 5 and row[0] == value for row, value in zip(series, initial))
    assert series[36][1:] == (I(slope), I(0), I(0), I(0))
    for k in range(1, 5):
        assert series[37][k] == (series[4][k - 1] - series[5][k - 1]) / k
    assert remainder[36] == I(0)
    rebuilt = []
    for row, tail in zip(series, remainder):
        polynomial = row[4]
        for coefficient in (row[3], row[2], row[1]):
            polynomial = polynomial * duration + coefficient
        rebuilt.append(polynomial * duration + tail)
    increment = tuple(rebuilt)
    assert increment == tuple(map(_interval, step["increment"]))
    endpoint = tuple(
        I(max((value + change).lo, whole.lo), min((value + change).hi, whole.hi))
        for value, change, whole in zip(initial, increment, tube)
    )
    assert endpoint == tuple(map(_interval, step["endpoint"]))

    def complete_rates(box):
        gradient, current = [I(0) for _ in range(18)], [I(0) for _ in range(18)]
        for i, j in protocol["support"]["edges"]:
            difference = box[i] - box[j]
            gradient[i] += difference
            gradient[j] -= difference
            force = sin(box[18 + j] - box[18 + i])
            current[i] += force
            current[j] -= force
        gamma = 1 / (1023 * pi_interval())
        degrees = protocol["support"]["degrees"]
        values = tuple(
            box[36] * (-a + gamma * f) / d
            for a, f, d in zip(gradient, current, degrees)
        )
        values += tuple(box[36] * gamma * a / d for a, d in zip(gradient, degrees))
        return values + (I(slope), box[4] - box[5])

    # Independently grouped interval arithmetic need not have identical
    # endpoints; its first-rate enclosure must intersect the retained one.
    # Higher nodal derivatives still depend on the retained Taylor execution.
    initial_rates = complete_rates(initial)
    assert all(
        max(rate.lo, row[1].lo) <= min(rate.hi, row[1].hi)
        for rate, row in zip(initial_rates, series)
    )
    rates = complete_rates(tube)
    images = tuple(value + I(0, duration) * rate for value, rate in zip(initial, rates))
    assert (
        min(
            min(image.lo - whole.lo, whole.hi - image.hi)
            for image, whole in zip(images, tube)
        )
        > 0
    )
    assert step["picard_interior_margin"] > 0 and step["domain_lower_bounds"] == (Q(1),)
    return endpoint, increment


def _history(protocol, entry, profile, source_state):
    public, sensor = (
        protocol["public_inputs_without_observation"],
        protocol["hidden_sensor"],
    )
    h = public["probe_duration"]
    clock = _clock_record(profile, public)
    assert entry["id"] == profile["id"] and entry["clock_audit"] == clock
    wrapper = entry["response"]
    assert wrapper["schema"] == "tnfr.sine-aperture-readout.v1"
    report = wrapper["report"]
    degrees, form, phase, _, _, _, _ = source_state
    source = form + phase
    assert (
        tuple(
            map(
                _interval,
                report["initial_form_bounds"] + report["initial_phase_bounds"],
            )
        )
        == source
    )
    assert report["geometry"]["nodes"] == protocol["support"]["nodes"]
    assert report["geometry"]["edges"] == protocol["support"]["edges"]
    assert report["degrees"] == degrees and report["capacity"] == (Q(1),) * 18
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in range(18))
    assert report["dipole"] == q
    assert report["law"] == protocol["complete_model"]["law"]
    assert report["clock"] == "d_tau/d_s=rho(s)=initial_clock_rate+clock_slope*s"
    assert report["state_order"] == tuple(f"x_{i}" for i in range(18)) + tuple(
        f"theta_{i}" for i in range(18)
    ) + ("rho", "cumulative_readout_integral")
    assert report["complete_observed_rows"] == (
        "dx/ds=rho*(-A*x+gamma*f(theta))",
        "dtheta/ds=rho*gamma*A*x",
        "drho/ds=clock_slope",
        "dz/ds=x_4-x_5",
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
        report["probe_duration"] == h
        and report["phase_increments"] == public["phase_increments"]
    )
    assert report["order"] == protocol["numerics"]["order"]
    carried = source + (I(clock["initial_clock_rate"]), I(0))
    assert tuple(map(_interval, report["initial_augmented_box"])) == carried
    assert _interval(report["baseline_readout_bounds"]) == form[4] - form[5]
    assert (
        report["initial_clock_rate"] == clock["initial_clock_rate"]
        and report["clock_slope"] == profile["slope"]
    )
    assert (
        report["clock_rate_bounds"] == clock["global_rate_bounds"]
        and report["clock_positivity_certified"] is True
    )
    assert (
        report["cumulative_structural_exposures"]
        == clock["cumulative_structural_exposures"]
    )
    assert (
        report["aperture_windows"]
        == protocol["observation"]["aperture_windows"]
        == clock["aperture_windows"]
    )
    assert (
        report["phase_event_increments"]
        == protocol["numerics"]["phase_increments"]
        == (Q(1, 4), Q(0), Q(0), Q(1, 2))
    )
    stops = {
        "clock_profile_admitted": clock["admitted"],
        "first_window_mean_identity": clock["first_window_mean_identity"],
        "complete_source_association": True,
        "affine_clock_association": True,
        "fixed_apertures_and_events": True,
        "initial_integral_zero_and_full_state": True,
    }
    averages = []
    assert (
        len(report["steps"])
        == len(report["window_initial_boxes"])
        == len(report["averaged_readout_bounds"])
        == 4
    )
    for index, (step, window, jump) in enumerate(
        zip(
            report["steps"],
            report["aperture_windows"],
            report["phase_event_increments"],
        )
    ):
        start, end = window
        post = (
            carried[:18]
            + tuple(
                value + jump * direction for value, direction in zip(carried[18:36], q)
            )
            + carried[36:]
        )
        assert tuple(map(_interval, report["window_initial_boxes"][index])) == post
        carried, increment = _audit_step(
            step, post, protocol, start, end - start, profile["slope"]
        )
        assert carried[36].contains(
            clock["initial_clock_rate"] + profile["slope"] * end
        )
        average = I(increment[37].lo / (end - start), increment[37].hi / (end - start))
        assert average == _interval(report["averaged_readout_bounds"][index])
        averages.append(average)
        for suffix in (
            "complete_certificate",
            "full_carry_and_phase_only_event",
            "average_from_same_integral_increment",
            "exact_clock_enclosed",
        ):
            stops[f"window_{index+1}_{suffix}"] = True
    assert report["status"] == "admitted" and report["completed_window_count"] == 4
    assert report["completed_observed_time"] == 2 * h
    assert tuple(map(_interval, report["final_state_bounds"])) == carried
    assert tuple(map(_interval, report["completed_endpoint_box"])) == carried
    assert report["failed_window_index"] is None and report["failed_tube"] is None
    assert report["unavailable_reasons"] == ()
    stops["full_history_available"] = True
    raw = tuple(averages)
    readings = tuple(
        sensor["gain"] * value + sensor["offset"] + error
        for value, error in zip(raw, sensor["reading_errors"])
    )
    contrasts = tuple(
        readings[a] - readings[b] for a, b in protocol["observation"]["contrast_pairs"]
    )
    curvature = sum(
        (
            weight * value
            for weight, value in zip(
                protocol["observation"]["curvature_coefficients"], readings
            )
        ),
        I(0),
    )
    assert (
        tuple(map(_interval, entry["readings"]["raw_averaged_readout_bounds"])) == raw
    )
    assert (
        tuple(map(_interval, entry["readings"]["averaged_reading_bounds"])) == readings
    )
    assert (
        tuple(map(_interval, entry["readings"]["recorded_average_contrast_bounds"]))
        == contrasts
    )
    assert _interval(entry["readings"]["recorded_curvature_bounds"]) == curvature
    stops["held_sensor_admitted"] = (
        public["readout_gain_bounds"][0]
        <= sensor["gain"]
        <= public["readout_gain_bounds"][1]
        and max(map(abs, sensor["reading_errors"])) <= public["readout_error_bound"]
    )
    stops["numerical_recorded_average_halfwidths"] = all(
        value.radius <= protocol["numerics"]["max_recorded_average_halfwidth"]
        for value in readings
    )
    return readings, contrasts, curvature, stops


@pytest.fixture(scope="module")
def histories(evidence, source_state):
    protocol, saved, _, _, _ = evidence
    entries = (saved["reference"],) + saved["cases"]
    profiles = (protocol["constant_mean_reference"],) + protocol[
        "actual_clock_profiles"
    ]
    return tuple(
        _history(protocol, entry, profile, source_state)
        for entry, profile in zip(entries, profiles)
    )


@pytest.mark.parametrize("index", range(3))
def test_three_complete_augmented_histories(evidence, histories, index):
    protocol, saved, _, _, _ = evidence
    entries = (saved["reference"],) + saved["cases"]
    assert len(entries) == 3
    assert protocol["numerics"]["dimension"] == 38
    assert protocol["numerics"]["windows_per_history"] == 4
    assert protocol["numerics"]["total_step_cap"] == 12
    assert protocol["numerics"]["producer_call_cap"] == 3
    stops = histories[index][3]
    assert all(
        entries[index]["stopping_rule"][key] is value for key, value in stops.items()
    )
    if index == 0:
        assert entries[index]["stopping_rule"] == stops and len(stops) == 25
        assert (
            "inverse_outputs" not in entries[index]
            and "public_packet" not in entries[index]
        )


def test_public_averages_inverse_constraints_and_all_fixed_stops(
    evidence, source_state, histories
):
    protocol, saved, raw_saved, _, _ = evidence
    public, sensor, controls = (
        protocol[k]
        for k in ("public_inputs_without_observation", "hidden_sensor", "controls")
    )
    h, drift = public["probe_duration"], public["clock_rate_derivative_bound"]
    assert protocol["observation"]["contrast_pairs"] == ((1, 0), (3, 2))
    assert protocol["observation"]["curvature_coefficients"] == (
        Q(9, 4),
        Q(-9, 2),
        Q(9, 4),
        Q(0),
    )
    assert controls == {
        "false_bulk_angle_bounds": (Q(11, 8), Q(353, 256)),
        "false_gain_bounds": (Q(31, 16), Q(2)),
        "false_clock_bounds": (Q(25, 32), Q(13, 16)),
        "equal_phase_increments": (Q(1, 4), Q(1, 4)),
    }
    assert protocol["thresholds"] == {
        "primary_actual_angle_width": Q(1, 1024),
        "primary_effective_gain_width": Q(1, 2048),
        "primary_gain_width": Q(1, 16),
        "primary_mean_clock_width": Q(1, 64),
    }
    assert set(protocol["public_packet"]["inverse_keys"]) == PUBLIC_KEYS
    assert protocol["public_packet"]["worker"] == f"{HELPERS}/invert_public_packet.py"
    reference = saved["reference"]
    reference_readings = histories[0][0]
    rebuilt = {"shared.source_admitted": True}
    rebuilt.update(
        {f"reference.{key}": value for key, value in histories[0][3].items()}
    )
    variants = {
        "primary": {},
        "false_angle": {"bulk_angle_bounds": controls["false_bulk_angle_bounds"]},
        "false_gain": {"readout_gain_bounds": controls["false_gain_bounds"]},
        "false_clock": {"clock_rate_bounds": controls["false_clock_bounds"]},
        "equal_inputs": {"phase_increments": controls["equal_phase_increments"]},
    }
    for index, (entry, profile) in enumerate(
        zip(saved["cases"], protocol["actual_clock_profiles"])
    ):
        readings, contrasts, _, original_stops = histories[index + 1]
        stops = dict(original_stops)
        packet = entry["public_packet"]
        assert set(packet) == {"schema", "inputs", "controls"}
        assert (
            packet["schema"]
            == protocol["public_packet"]["schema"]
            == "tnfr.sine-aperture-public-inference.v1"
        )
        assert set(packet["inputs"]) == PUBLIC_KEYS and packet["controls"] == controls
        expected_inputs = {
            **public,
            "averaged_reading_bounds": tuple((v.lo, v.hi) for v in readings),
        }
        assert packet["inputs"] == expected_inputs
        request = json.dumps(
            raw_saved["cases"][index]["public_packet"], sort_keys=True, allow_nan=False
        ).encode("utf-8")
        outputs = entry["inverse_outputs"]
        assert set(outputs) == {"request_sha256", *variants}
        stops["public_only_packet"] = (
            outputs["request_sha256"] == hashlib.sha256(request).hexdigest()
        )
        reports = {}
        for name, changes in variants.items():
            assert outputs[name]["schema"] == "tnfr.sine-aperture-inference.v1"
            reports[name] = outputs[name]["report"]
            _audit_aperture({**expected_inputs, **changes}, reports[name])
        primary = reports["primary"]
        stops["primary_bounded_candidate"] = primary["status"] == "bounded_candidate"
        stops["primary_certificates"] = all(
            primary[key]
            for key in (
                "source_admitted",
                "clock_drift_transfer_certified",
                "aperture_reconstruction_certified",
                "finite_curvature_bound_certified",
                "rank_certified",
                "inverse_enclosure_available",
            )
        )
        truths = {
            "nominal_angle": protocol["hidden_source"]["bulk_angle"],
            "actual_angle": source_state[-1],
            "effective_gain": sensor["gain"] * profile["first_window_mean"],
            "gain": sensor["gain"],
            "mean_clock": profile["first_window_mean"],
        }
        for name, field in (
            ("nominal_angle", "nominal_bulk_angle_outer_bounds"),
            ("actual_angle", "actual_long_arc_mean_outer_bounds"),
            ("effective_gain", "effective_gain_outer_bounds"),
            ("gain", "readout_gain_outer_bounds"),
            ("mean_clock", "first_window_mean_clock_rate_outer_bounds"),
        ):
            bound = _interval(primary[field]) if primary[field] is not None else None
            stops[name + "_covered"] = bound is not None and bound.contains(
                truths[name]
            )
            if name != "nominal_angle":
                stops[name + "_resolution"] = (
                    bound is not None
                    and bound.width < protocol["thresholds"][f"primary_{name}_width"]
                )
        stops["false_angle_prior_excluded"] = (
            reports["false_angle"]["status"] == "incompatible"
        )
        for name in ("gain", "clock"):
            control = reports["false_" + name]
            coarse = control["reference_envelope"]["clock_envelope"]
            stops[f"false_{name}_coarse_available"] = (
                coarse["inverse_enclosure_available"]
                and coarse["status"] == "bounded_candidate"
            )
            stops[f"false_{name}_curvature_excluded"] = (
                control["status"] == "incompatible"
            )
        equal = reports["equal_inputs"]
        stops["equal_amplitude_rank_abstention"] = (
            equal["status"] == "unavailable"
            and equal["rank_deficient"]
            and not equal["rank_certified"]
        )
        heat = (
            2 * public["readout_gain_bounds"][1] * public["form_radius"]
            + 2 * public["readout_error_bound"]
        )
        assert entry["phase_blind_recorded_average_contrast_bound"] == heat
        assert (
            protocol["phase_blind_alternative"]["structural_rows"]
            == "x_tau=-A*x; theta_tau=gamma*A*x"
        )
        for j, contrast in enumerate(contrasts):
            stops[f"contrast_{j+1}_phase_blind_excluded"] = (
                contrast.hi < -heat or contrast.lo > heat
            )
        # Recompute clock transfer from the public primitives, not the report.
        g = Q(1, 3069)
        q0 = public["form_radius"] + 14 * g * public["clock_rate_bounds"][1] * h
        speed = 2 * q0 + 7 * g
        span = public["clock_rate_bounds"][1] - public["clock_rate_bounds"][0]
        exposures = tuple(
            min(c * drift * h**2, 2 * c * span * h)
            for c in (Q(7, 108), Q(13, 108), Q(7, 108))
        ) + (
            min(5 * drift * h**2 / 12, span * h / 2),
        )
        allowances = tuple(
            public["readout_gain_bounds"][1] * speed * value for value in exposures
        )
        differences = tuple(
            value - baseline for value, baseline in zip(readings, reference_readings)
        )
        comparison = entry["reference_comparison"]
        assert comparison["reference_history_id"] == reference["id"]
        assert (
            tuple(map(_interval, comparison["recorded_average_difference_bounds"]))
            == differences
        )
        assert comparison["reconstructed_averaged_exposure_bounds"] == exposures
        assert (
            comparison["reconstructed_recorded_average_discrepancy_bounds"]
            == allowances
        )
        stops["transfer_bounds_match_public_primitives"] = (
            primary["averaged_exposure_discrepancy_bounds"] == exposures
            and primary["averaged_clock_discrepancy_upper_bounds"] == allowances
        )
        signs = (1, 1, 1, -1) if profile["slope"] > 0 else (-1, -1, -1, 1)
        name = (
            "positive_minus_reference_signs"
            if profile["slope"] > 0
            else "negative_minus_reference_signs"
        )
        assert protocol["prospective_separation"][name] == signs
        for j, (difference, allowance, sign) in enumerate(
            zip(differences, allowances, signs)
        ):
            stops[f"average_{j+1}_signed_reference_separation"] = (
                difference.lo > 0 if sign > 0 else difference.hi < 0
            )
            stops[f"average_{j+1}_full_difference_within_transfer"] = (
                -allowance <= difference.lo <= difference.hi <= allowance
            )
        actual_report, reference_report = (
            entry["response"]["report"],
            reference["response"]["report"],
        )
        for name, left_raw, right_raw in (
            (
                "H_nodal_enclosures_intersect",
                actual_report["steps"][2]["endpoint"][:36],
                reference_report["steps"][2]["endpoint"][:36],
            ),
            (
                "H_post_event_nodal_enclosures_intersect",
                actual_report["window_initial_boxes"][3][:36],
                reference_report["window_initial_boxes"][3][:36],
            ),
        ):
            left, right = tuple(map(_interval, left_raw)), tuple(
                map(_interval, right_raw)
            )
            stops[name] = len(left) == len(right) == 36 and all(
                max(a.lo, b.lo) <= min(a.hi, b.hi) for a, b in zip(left, right)
            )
        assert entry["stopping_rule"] == stops and len(stops) == 56
        rebuilt.update({f"{entry['id']}.{key}": value for key, value in stops.items()})
    plus, minus = histories[1][2], histories[2][2]
    paired = {"actual_composed_curvatures_signed_separation": plus.hi < minus.lo}
    assert saved["paired_control"] == paired
    rebuilt.update({f"paired.{key}": value for key, value in paired.items()})
    assert len(rebuilt) == 139 and saved["frozen_stopping_rule"] == rebuilt
    assert saved["work_budget"] == {
        "producer_calls_attempted": 3,
        "full_flow_step_budget_reserved": 12,
        "full_flow_steps_completed": 12,
        "full_flow_steps_attempted_known": 12,
        "producer_call_pending": False,
    }
    assert saved["frozen_stopping_rule_passed"] is all(rebuilt.values())
    expected_status = (
        "certified_reserved_aperture_inference"
        if all(rebuilt.values())
        else "unavailable"
    )
    assert saved["report"]["status"] == expected_status
    assert "equal_exposure_companion" not in saved

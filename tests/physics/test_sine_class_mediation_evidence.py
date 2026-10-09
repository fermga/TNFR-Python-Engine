"""Read-only integrity and decision audit of the class-mediation certificate.

Stored formation and analytic bound evaluations remain evidence premises. This
audit checks their source association, arithmetic consistency and fixed stopping
inequalities; it neither reruns them nor authenticates execution or chronology.
"""

import ast
import hashlib
import re
import subprocess
import zipfile
from fractions import Fraction as Q
from pathlib import Path, PurePosixPath

import pytest

from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "docs/assets/sine_formed_classes"
STEM = "class-mediation-v1"
PROOF = "theory/nodal/SINE_CLASS_MEDIATED_RESPONSE.md"
PRODUCER = "build/class-mediation-freeze/evaluate_class_mediation.py"
OVERLAYS = {
    "src/tnfr/physics/relational_sine_class_mediation.py",
    "src/tnfr/sdk/relational_reports.py",
}


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            assert type(value["numerator"]) is int
            assert type(value["denominator"]) is int and value["denominator"] > 0
            return Q(value["numerator"], value["denominator"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    return value


def _interval(value):
    assert set(value) == {"lo", "hi"}
    assert isinstance(value["lo"], Q) and isinstance(value["hi"], Q)
    assert value["lo"] <= value["hi"]
    return value["lo"], value["hi"]


@pytest.fixture(scope="module", autouse=True)
def no_frozen_execution():
    from tnfr.physics import (
        _sine_formed_contact,
        relational_sine_class_mediation,
        relational_sine_formed_classes,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("saved evidence audit must not execute an assessor or producer")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                relational_sine_class_mediation,
                ("assess_sine_class_mediation", "_unprobed_handoff"),
            ),
            (
                _sine_formed_contact,
                ("_unprobed_handoff", "assess_sine_formed_class_pair"),
            ),
            (relational_sine_formed_classes, ("assess_sine_formed_class_pair",)),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        for name in ("run", "Popen", "check_output"):
            patch.setattr(subprocess, name, forbidden)
        yield


@pytest.fixture(scope="module")
def evidence():
    content = {
        suffix: (DIRECTORY / (STEM + suffix)).read_bytes()
        for suffix in (".protocol.json", ".source.zip", ".json", ".manifest.json")
    }
    protocol, saved, manifest = (
        _decode(json_loads(content[suffix]))
        for suffix in (".protocol.json", ".json", ".manifest.json")
    )
    yield protocol, saved, manifest, content
    assert all(
        (DIRECTORY / (STEM + suffix)).read_bytes() == value
        for suffix, value in content.items()
    )


def test_outer_and_archive_hashes_protocol_and_base_overlay_association(evidence):
    protocol, saved, manifest, content = evidence
    assert protocol["schema"] == "tnfr.sine-class-mediation-protocol.v1"
    assert saved["schema"] == "tnfr.sine-class-mediation-assessment.v1"
    assert manifest["schema"] == "tnfr.sine-class-mediation-evidence.v1"
    assert {item["file"] for item in manifest["artifacts"]} == {
        STEM + suffix for suffix in (".protocol.json", ".source.zip", ".json")
    }
    assert len(manifest["artifacts"]) == 3
    for item in manifest["artifacts"]:
        value = content[item["file"][len(STEM) :]]
        assert type(item["bytes"]) is int and item["bytes"] == len(value)
        assert item["sha256"] == hashlib.sha256(value).hexdigest()
    assert (
        saved["protocol_sha256"]
        == hashlib.sha256(content[".protocol.json"]).hexdigest()
    )
    assert (
        saved["source_archive_sha256"]
        == hashlib.sha256(content[".source.zip"]).hexdigest()
    )
    base = protocol["source_base_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", base)
    assert manifest["source_base_commit"] == base
    assert (
        set(protocol["runtime_overlays"])
        == set(manifest["runtime_overlays"])
        == OVERLAYS
    )
    with zipfile.ZipFile(DIRECTORY / (STEM + ".source.zip")) as archive:
        source = json_loads(archive.read("source-manifest.json"))
        expected = OVERLAYS | {
            PROOF,
            PRODUCER,
            f"docs/assets/sine_formed_classes/{STEM}.protocol.json",
        }
        assert source["source_base_commit"] == base
        assert len(source["files"]) == len(expected)
        assert {item["path"] for item in source["files"]} == expected
        assert len(archive.namelist()) == len(expected) + 1
        assert set(archive.namelist()) == expected | {"source-manifest.json"}
        for item in source["files"]:
            path = PurePosixPath(item["path"])
            assert not path.is_absolute() and ".." not in path.parts
            value = archive.read(item["path"])
            assert type(item["bytes"]) is int and item["bytes"] == len(value)
            assert item["sha256"] == hashlib.sha256(value).hexdigest()
        assert (
            archive.read(f"docs/assets/sine_formed_classes/{STEM}.protocol.json")
            == content[".protocol.json"]
        )


def test_prospective_prefix_and_single_archived_assessment_call():
    with zipfile.ZipFile(DIRECTORY / (STEM + ".source.zip")) as archive:
        prospective = archive.read(PROOF).decode("utf-8").replace("\r\n", "\n")
        current = (ROOT / PROOF).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert current.startswith(prospective)
        assert 'id="sine-class-mediated-protocol"' in prospective
        assert 'id="sine-class-mediated-result"' not in prospective
        assert 'id="sine-class-mediated-result"' in current[len(prospective) :]
        # Current runtime may legitimately evolve. The strict source checks
        # concern the archived overlay bytes, not equality with live source.
        tree = ast.parse(archive.read(PRODUCER))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assess_sine_class_mediation"
        ]
        assert len(calls) == 1
        call = calls[0]
        assert not call.args and len(call.keywords) == 1
        assert call.keywords[0].arg is None and call.keywords[0].value.id == "inputs"
        writer = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_once"
        )
        assert isinstance(writer.body[0], ast.If)
        assert any(isinstance(node, ast.Raise) for node in ast.walk(writer.body[0]))


def test_original_source_family_handoff_and_actual_support_are_retained(evidence):
    protocol, saved, _, _ = evidence
    report = saved["response"]["report"]
    inputs = protocol["inputs"]
    assert all(
        report[name] == value and type(report[name]) is type(value)
        for name, value in inputs.items()
    )
    assert report["classes"] == protocol["cases"] == ((1, 1, 1), (1, 2, 1))
    assert report["nodes"] == protocol["support"]["nodes"] == tuple(range(27))
    assert report["edges"] == protocol["support"]["edges"]
    degrees = tuple(
        sum(node in edge for edge in report["edges"]) for node in report["nodes"]
    )
    assert len(report["edges"]) == 29 and sum(degrees) == 58
    assert tuple(degrees[i] for i in (4, 13, 22)) == (3, 4, 3)
    assert all(degrees[i] == 2 for i in range(27) if i not in (4, 13, 22))
    assert report["phase_origins"] == (0, 0, 0)
    assert report["held_capacities"] == (1,) * 27
    handoff = report["source_handoff"]
    formation = handoff["formation_certificate"]
    scale = Q(2046, 9) * Q(355, 113) ** 2
    assert formation["initial_forms_by_class"] == tuple(
        tuple(k * scale * (j - 4) for j in range(9)) for k in (1, 2)
    )
    assert formation["initial_form_storage_by_class"] == tuple(
        36 * k**2 * scale**2 for k in (1, 2)
    )
    assert formation["common_source_storage_budget"] == 10**9
    assert formation["scaled_time"] == inputs["formation_time"]
    assert formation["form_error_bound"] == inputs["form_error_bound"]
    assert formation["phase_error_bound"] == inputs["phase_error_bound"]
    assert formation["preparation_dimension"] == 16
    assert formation["full_relative_preparation"] is True
    assert formation["initial_zero_winding_certified"] is True
    assert formation["formation_certified_by_class"] == (True, True)
    assert handoff["handoff_certified_by_class"] == (True, True)
    assert handoff["exact_decay_upper_bound"] == Q(1, 2 ** inputs["decay_power"])
    for name in (
        "endpoint_form_norm_squared_upper_bounds",
        "endpoint_phase_norm_squared_upper_bounds",
    ):
        assert len(handoff[name]) == 2
        assert all(
            0 <= value <= inputs["endpoint_radius"] ** 2 for value in handoff[name]
        )


def test_saved_interval_inflation_work_and_all_nine_stops(evidence):
    protocol, saved, manifest, _ = evidence
    report = saved["response"]["report"]
    inputs = protocol["inputs"]
    eps, h, a, r = (
        inputs[name]
        for name in ("endpoint_radius", "contact_duration", "probe_amplitude", "radius")
    )
    assert report["common_diffusion_walk_coefficient"] == Q(1, 12)
    assert report["mediator_walk_coefficient"] == Q(1, 24)
    assert report["first_class_dependent_derivative_order"] == 3
    preparation = 4 * eps / (1 - 3 * h)
    noise = 4 * inputs["readout_error_bound"]
    assert report["preparation_contrast_error_upper_bound"] == preparation
    assert report["readout_contrast_error_upper_bound"] == noise
    leading = _interval(report["leading_contrast_bounds"])
    recorded = _interval(report["recorded_contrast_bounds"])
    blind = _interval(report["phase_blind_recorded_contrast_bounds"])
    error = (
        report["linear_tail_upper_bound"]
        + report["nonlinear_contrast_error_upper_bound"]
        + preparation
        + noise
    )
    assert recorded[0] <= leading[0] - error
    assert recorded[1] >= leading[1] + error
    assert blind[0] <= -preparation - noise <= preparation + noise <= blind[1]
    assert recorded[0] > protocol["contrast_threshold"] > 0
    assert recorded[0] > blind[1]
    assert _interval(report["phase_blind_exclusion_margin_bounds"])[0] > 0
    joined = report["joined_bounds"]
    assert joined["contact_work_upper_bound"] == 8 * eps**2
    assert 0 <= joined["contact_work_upper_bound"] <= inputs["contact_work_allowance"]
    work = _interval(report["probe_work_bounds"])
    work_low, work_high = Q(3, 2) * a**2 - 6 * a * eps, Q(3, 2) * a**2 + 6 * a * eps
    assert (
        0
        < work[0]
        <= work_low
        <= work_high
        <= work[1]
        <= inputs["probe_work_allowance"]
    )
    assert report["probe_work_margin"] == inputs["probe_work_allowance"] - work_high
    radius_squared = 6 * eps**2 + 2 * a * eps + Q(26, 27) * a**2
    energy = 22 * eps**2 + 6 * a * eps + Q(3, 2) * a**2
    assert report["post_probe_radius_squared_upper_bound"] == radius_squared < r**2
    assert (
        report["post_probe_excess_storage_upper_bound"]
        == energy
        < joined["joined_barrier_lower_bound"]
    )
    assert report["post_probe_radius_margin"] == r**2 - radius_squared
    assert (
        report["post_probe_storage_margin"]
        == joined["joined_barrier_lower_bound"] - energy
    )
    assert report["probe_form_mean_shift"] == 3 * a / 58
    expected = {
        "report_certified": report["status"] == "certified_class_mediation",
        "fresh_source_handoff": report["source_handoff_certified"],
        "positive_recorded_response": report["response_certified"] and recorded[0] > 0,
        "baseline_identity": report["baseline_identity_certified"]
        and joined["identity_certified"],
        "probe_identity": report["probe_identity_certified"]
        and radius_squared < r**2
        and energy < joined["joined_barrier_lower_bound"],
        "contact_work_within_allowance": report["contact_work_within_allowance"]
        and joined["contact_work_upper_bound"] <= inputs["contact_work_allowance"],
        "probe_work_within_allowance": report["probe_work_within_allowance"]
        and work_high <= inputs["probe_work_allowance"],
        "recorded_contrast_above_threshold": recorded[0]
        > protocol["contrast_threshold"],
        "phase_blind_excluded": recorded[0] > blind[1],
    }
    assert len(expected) == 9 and all(value is True for value in expected.values())
    assert saved["checks"] == expected
    assert saved["evaluation_error"] is None and saved["passed"] is True
    assert report["unavailable_reasons"] == ()
    assert manifest["frozen_stopping_rule_passed"] is True
    assert (
        saved["evaluation_kind"]
        == protocol["evaluation_kind"]
        == "conditional_analytic_complete_family_no_trajectory"
    )

"""Read-only content and stopping-rule checks for retained formed-C9 evidence.

No producing assessor or archived script is executed. Content consistency is
not independent authentication of chronology or physical acquisition. Runtime
source may legitimately evolve after the retained snapshot; full Git history
is not required to check these archived bytes and their revision references.
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

from tnfr.research.relational_acquisition import _exact, _verify_archive
from tnfr.utils.io import json_loads

DIRECTORY = Path(__file__).parents[2] / "docs/assets/sine_formed_classes"
ARCHIVED = (
    "maintenance-v1",
    "contact-v1",
    "reduced-ports-v1",
    "port-composition-v1",
    "port-relaxation-v1",
    "port-form-tracking-v1",
    "two-port-compatibility-v1",
    "two-port-capture-v1",
    "two-port-probe-v1",
    "two-port-dipole-v1",
    "two-port-inference-v1",
)
MANIFESTS = ("evidence.manifest.json",) + tuple(
    f"{stem}.manifest.json" for stem in ARCHIVED
)


@pytest.fixture(scope="module", autouse=True)
def no_evidence_producers():
    from tnfr.mathematics import _validated_metric, _validated_taylor
    from tnfr.physics import (
        _sine_formed_contact,
        relational_sine_formed_class_contact,
        relational_sine_formed_class_maintenance,
        relational_sine_formed_classes,
        relational_sine_port_composition,
        relational_sine_port_form_tracking,
        relational_sine_port_relaxation,
        relational_sine_reduced_class_ports,
        relational_sine_two_port_capture,
        relational_sine_two_port_compatibility,
        relational_sine_two_port_dipole,
        relational_sine_two_port_inference,
        relational_sine_two_port_probe,
        relational_sine_two_port_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("retained evidence audit must not execute a producer")

    with pytest.MonkeyPatch.context() as patch:
        for module in (
            _sine_formed_contact,
            relational_sine_formed_class_contact,
            relational_sine_formed_class_maintenance,
            relational_sine_formed_classes,
            relational_sine_port_composition,
            relational_sine_port_form_tracking,
            relational_sine_port_relaxation,
            relational_sine_reduced_class_ports,
            relational_sine_two_port_capture,
            relational_sine_two_port_compatibility,
            relational_sine_two_port_dipole,
            relational_sine_two_port_inference,
            relational_sine_two_port_probe,
            relational_sine_two_port_readout,
        ):
            for name in vars(module):
                if (
                    name.startswith(
                        (
                            "assess_sine_formed",
                            "assess_sine_reduced",
                            "evaluate_sine_reduced",
                            "assess_sine_port_composition",
                            "evaluate_sine_port_composition",
                            "assess_sine_port_relaxation",
                            "assess_sine_port_form_tracking",
                            "assess_sine_two_port_compatibility",
                            "assess_sine_two_port_capture",
                            "assess_sine_two_port_transit",
                            "assess_sine_two_port_probe",
                            "assess_sine_two_port_dipole",
                            "bound_sine_two_port_readout",
                            "infer_sine_two_port_geometry",
                            "validated_box_taylor_step",
                            "validated_metric_taylor_step",
                        )
                    )
                    or name == "_unprobed_handoff"
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
def retained():
    manifests = {
        name: json_loads((DIRECTORY / name).read_bytes()) for name in MANIFESTS
    }
    content = {
        item["file"]: (DIRECTORY / item["file"]).read_bytes()
        for manifest in manifests.values()
        for item in manifest["artifacts"]
    }
    yield manifests, content
    assert all(
        (DIRECTORY / name).read_bytes() == data for name, data in content.items()
    )


def test_all_retained_artifact_sizes_and_hashes(retained):
    manifests, content = retained
    expected = (
        {"pair-v1.json", "response-v1.json"}
        | {
            stem + suffix
            for stem in ARCHIVED
            for suffix in (".json", ".protocol.json", ".source.zip")
        }
        | {
            "two-port-probe-v1.first-attempt.json",
            "two-port-probe-v1.export-recovery.py.txt",
        }
    )
    names = [
        item["file"]
        for manifest in manifests.values()
        for item in manifest["artifacts"]
    ]
    assert len(names) == len(set(names)) == 37 and set(names) == expected
    for manifest in manifests.values():
        for item in manifest["artifacts"]:
            data = content[item["file"]]
            assert type(item["bytes"]) is int and len(data) == item["bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            if item["file"].endswith(".json"):
                assert isinstance(json_loads(data), dict)


@pytest.mark.parametrize("stem", ARCHIVED)
def test_archived_inventory_protocol_and_base_revision_are_consistent(retained, stem):
    manifests, content = retained
    outer = manifests[f"{stem}.manifest.json"]
    protocol = json_loads(content[f"{stem}.protocol.json"])
    archive_path = DIRECTORY / f"{stem}.source.zip"
    # The outer digest binds the embedded manifest too. Its own observed digest
    # is added only to adapt the existing exact-inventory verifier, not as an
    # independent self-authentication claim.
    assert hashlib.sha256(content[archive_path.name]).hexdigest() == next(
        item["sha256"]
        for item in outer["artifacts"]
        if item["file"] == archive_path.name
    )
    with zipfile.ZipFile(archive_path) as archive:
        source_bytes = archive.read("source-manifest.json")
        source = json_loads(source_bytes)
        entries = source["files"]
        paths = [entry["path"] for entry in entries]
        assert len(paths) == len(set(paths)) and "source-manifest.json" not in paths
        assert all(
            not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts
            for name in paths
        )
        digests = {entry["path"]: entry["sha256"] for entry in entries}
        digests["source-manifest.json"] = hashlib.sha256(source_bytes).hexdigest()
        _verify_archive(archive_path, digests)
        for entry in entries:
            assert type(entry["bytes"]) is int
            assert len(archive.read(entry["path"])) == entry["bytes"]
        protocol_entries = [
            name
            for name in paths
            if PurePosixPath(name).name == f"{stem}.protocol.json"
        ]
        assert len(protocol_entries) == 1
        assert archive.read(protocol_entries[0]) == content[f"{stem}.protocol.json"]
    base = outer["source_base_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", base)
    assert source["source_base_commit"] == protocol["source_base_commit"] == base
    assert source["runtime_overlays"] == outer["runtime_overlays"]
    assert set(source["runtime_overlays"]) == {
        name for name in paths if name.startswith("src/")
    }
    assert protocol["source_overlay_archive"] == archive_path.name
    assert json_loads(content[f"{stem}.json"])["report"]["status"] == outer["status"]


def test_original_pair_and_response_do_not_acquire_an_evaluation_snapshot(retained):
    manifests, _ = retained
    manifest = manifests["evidence.manifest.json"]
    assert manifest["original_evaluated_source_snapshot_archived"] is False
    assert (
        manifest["reviewed_source"]["kind"]
        == "reviewed_consolidation_worktree_not_evaluation_snapshot"
    )
    assert {item["file"] for item in manifest["artifacts"]} == {
        "pair-v1.json",
        "response-v1.json",
    }
    assert all(
        item["source_kind"] == "original_saved_analytic_report"
        for item in manifest["artifacts"]
    )
    assert manifest["provenance_limitations"]


def test_reduced_stopping_rule_follows_saved_exact_fields(retained):
    manifests, content = retained
    manifest = manifests["reduced-ports-v1.manifest.json"]
    protocol = json_loads(content["reduced-ports-v1.protocol.json"])
    report = json_loads(content["reduced-ports-v1.json"])["report"]
    for name, value in protocol["inputs"].items():
        assert _exact(report[name]) == _exact(value)
    assert report["receiver_class"] == 2 and report["donor_classes"] == [1, 2]
    assert (
        report["joined_coordinate_count"]
        == 20
        < report["full_joined_coordinate_count"]
        == 36
    )
    total = sum(
        _exact(report[name])
        for name in (
            "reduced_semigroup_tail_upper_bound",
            "reduced_nonlinear_remainder_upper_bound",
            "surrogate_full_discrepancy_upper_bound",
            "preparation_response_error_upper_bound",
            "readout_contrast_error_upper_bound",
        )
    )
    assert total == _exact(report["total_error_upper_bound"])
    lo, hi = (_exact(report["recorded_contrast_bounds"][end]) for end in ("lo", "hi"))
    leading_lo, leading_hi = (
        _exact(report["ideal_leading_contrast_bounds"][end]) for end in ("lo", "hi")
    )
    assert lo <= leading_lo - total <= leading_hi + total <= hi
    threshold = _exact(
        protocol["prediction"]["recorded_class_two_minus_one_lower_threshold"]
    )
    fraction = _exact(protocol["prediction"]["fraction_of_certified_full_gap"])
    assert _exact(manifest["contrast_threshold"]) == threshold
    assert (
        _exact(manifest["error_fraction"])
        == _exact(report["error_fraction"])
        == fraction
    )
    assert lo > threshold > 0 and 0 <= total < fraction * lo
    assert _exact(report["error_ratio_upper_bound"]) == total / lo
    margin = report["error_fraction_margin_bounds"]
    assert 0 < _exact(margin["lo"]) <= fraction * lo - total <= _exact(margin["hi"])
    handoff = report["unprobed_handoff"]
    assert handoff["formation_certificate"]["status"] == "certified_two_formed_classes"
    assert handoff["handoff_certified_by_class"] == [True, True]
    power = report["decay_power"]
    assert power <= _exact(handoff["decay_exponent"]) <= 4096
    assert _exact(handoff["exact_decay_upper_bound"]) == Q(1, 2**power)
    eps = _exact(report["endpoint_radius"])
    for name in (
        "endpoint_form_norm_squared_upper_bounds",
        "endpoint_phase_norm_squared_upper_bounds",
    ):
        assert all(0 <= _exact(value) <= eps**2 for value in handoff[name])
    joined = report["joined_contact_bounds"]
    for name in ("joined_radius_margin_bounds", "joined_storage_margin_bounds"):
        assert _exact(joined[name]["lo"]) > 0
    phi = _exact(report["phase_origin_difference"])
    work_upper = 2 * eps**2 + (phi + 2 * eps) ** 2 / 2
    assert work_upper <= _exact(report["work_allowance"])
    assert (
        joined["identity_certified"] is True and joined["work_within_allowance"] is True
    )
    assert all(
        report[name] is True
        for name in (
            "response_certified",
            "approximation_certified",
            "identity_certified",
            "work_within_allowance",
        )
    )
    assert (
        report["status"] == "certified_reduced_class_ports"
        and report["unavailable_reasons"] == []
    )
    assert manifest["frozen_stopping_rule_passed"] is True


def test_composition_stopping_rule_follows_saved_exact_fields(retained):
    manifests, content = retained
    manifest = manifests["port-composition-v1.manifest.json"]
    protocol = json_loads(content["port-composition-v1.protocol.json"])
    saved = json_loads(content["port-composition-v1.json"])
    report = saved["report"]
    for name, value in protocol["inputs"].items():
        if name in ("classes", "contacts"):
            assert report[name] == value
        elif name == "phase_origins":
            assert tuple(map(_exact, report[name])) == tuple(map(_exact, value))
        else:
            assert _exact(report[name]) == _exact(value)
    geometry = report["geometry"]
    assert geometry["component_count"] == 3
    assert geometry["contact_degrees"] == [1, 2, 1]
    assert geometry["contact_diameter"] == 2 and geometry["connected"] is True
    assert tuple(map(_exact, geometry["layer_masses"])) == (
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        3,
        4,
        4,
        4,
        4,
    )
    handoff = report["unprobed_handoff"]
    assert handoff["formation_certificate"]["status"] == "certified_two_formed_classes"
    assert handoff["handoff_certified_by_class"] == [True, True]
    power = report["decay_power"]
    assert power <= _exact(handoff["decay_exponent"]) <= 4096
    assert _exact(handoff["exact_decay_upper_bound"]) == Q(1, 2**power)
    eps = _exact(report["endpoint_radius"])
    for name in (
        "endpoint_form_norm_squared_upper_bounds",
        "endpoint_phase_norm_squared_upper_bounds",
    ):
        assert all(0 <= _exact(value) <= eps**2 for value in handoff[name])
    h = _exact(report["contact_duration"])
    prep = _exact(report["preparation_error_upper_bound"])
    defect = _exact(report["ideal_surrogate_discrepancy_upper_bound"])
    total = _exact(report["total_approximation_error_upper_bound"])
    allowance = _exact(report["approximation_allowance"])
    assert prep == eps / (1 - 3 * h) and defect > 0
    assert total == prep + defect < allowance == 2 * eps
    margin = report["approximation_margin_bounds"]
    assert 0 < _exact(margin["lo"]) <= allowance - total <= _exact(margin["hi"])
    phi = _exact(report["phase_origins"][1])
    radius = _exact(report["radius"])
    assert _exact(report["joined_gap_lower_bound"]) == Q(2, 135)
    assert (
        _exact(report["joined_radius_squared_upper_bound"]) == 6 * eps**2 + 6 * phi**2
    )
    assert (
        _exact(report["joined_excess_storage_upper_bound"])
        == 18 * eps**2 + (phi + 2 * eps) ** 2
    )
    assert (
        _exact(report["contact_work_upper_bound"]) == 4 * eps**2 + (phi + 2 * eps) ** 2
    )
    assert _exact(report["contact_work_upper_bound"]) <= _exact(
        report["work_allowance"]
    )
    assert _exact(report["joined_excess_storage_upper_bound"]) < _exact(
        report["joined_barrier_lower_bound"]
    )
    assert _exact(report["joined_radius_squared_upper_bound"]) < radius**2
    for name in ("joined_radius_margin_bounds", "joined_storage_margin_bounds"):
        assert _exact(report[name]["lo"]) > 0
    control = saved["algebraic_control"]
    assert _exact(control["middle_form_rate"]) == -1
    assert _exact(control["naive_middle_form_rate"]) == -Q(4, 3)
    assert _exact(control["weighted_form_charge_rate"]) == 0
    assert _exact(control["naive_weighted_form_charge_rate"]) == -Q(4, 3)
    assert _exact(control["form_storage"]) == 2
    assert (
        _exact(control["storage_rate"]) == -_exact(control["dissipation"]) == -Q(17, 3)
    )
    assert all(
        report[name] is True
        for name in (
            "approximation_certified",
            "identity_certified",
            "work_within_allowance",
        )
    )
    assert report["status"] == "certified_sine_port_composition"
    assert report["unavailable_reasons"] == []
    assert (
        control["passed"]
        is saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is True
    )


def test_relaxation_preserves_independent_channel_outcomes(retained):
    manifests, content = retained
    manifest = manifests["port-relaxation-v1.manifest.json"]
    protocol = json_loads(content["port-relaxation-v1.protocol.json"])
    saved = json_loads(content["port-relaxation-v1.json"])
    report = saved["report"]
    for name, value in protocol["inputs"].items():
        if name in ("classes", "contacts"):
            assert report[name] == value
        elif name == "phase_origins":
            assert tuple(map(_exact, report[name])) == tuple(map(_exact, value))
        else:
            assert _exact(report[name]) == _exact(value)
    gamma_lo, gamma_hi = (_exact(report["gamma_bounds"][key]) for key in ("lo", "hi"))
    eps = _exact(report["endpoint_radius"])
    handoff = report["unprobed_handoff"]
    assert handoff["formation_certificate"]["status"] == "certified_two_formed_classes"
    assert handoff["handoff_certified_by_class"] == [True, True]
    assert _exact(handoff["exact_decay_upper_bound"]) == Q(
        1, 2 ** report["decay_power"]
    )
    for name in (
        "endpoint_form_norm_squared_upper_bounds",
        "endpoint_phase_norm_squared_upper_bounds",
    ):
        assert all(0 <= _exact(value) <= eps**2 for value in handoff[name])
    joined = report["joined_bounds"]
    assert joined["identity_certified"] is joined["work_within_allowance"] is True
    assert (
        report["normalized_gap_certified"] is report["refined_chart_certified"] is True
    )
    assert _exact(report["normalized_gap_lower_bound"]) == Q(3, 100)
    assert _exact(report["refined_cosine_bounds"]["lo"]) >= Q(1, 6)
    assert _exact(report["edge_disagreement_norm_squared_upper_bound"]) == 12 * _exact(
        joined["joined_excess_storage_upper_bound"]
    )
    for name in ("odd_bounds", "even_bounds"):
        box = report[name]
        gap = _exact(box["gap_lower_bound"])
        forcing = _exact(box["forcing_upper_bound"])
        a, b = _exact(box["phase_damping"]), _exact(box["scaled_form_damping"])
        determinant = _exact(box["determinant"])
        assert a == gap / 6 and b == gap / gamma_hi**2 - 2
        assert determinant == a * b - 4 > 0
        initial = _exact(box["initial_norm_upper_bound"])
        y0 = _exact(box["initial_joint_norm_upper_bound"])
        z0 = _exact(box["initial_scaled_form_norm_upper_bound"])
        assert y0 == (1 + gamma_hi) * initial and z0 == gamma_hi * initial
        p = max(Q(0), -a * y0 + 2 * z0)
        q = max(Q(0), 2 * y0 - b * z0)
        assert _exact(box["joint_initial_correction"]) == p
        assert _exact(box["scaled_form_initial_correction"]) == q
        y = _exact(box["joint_norm_upper_bound"])
        z = _exact(box["scaled_form_norm_upper_bound"])
        assert y == y0 + (b * (forcing + p) + 2 * (forcing + q)) / determinant
        assert z == z0 + (2 * (forcing + p) + a * (forcing + q)) / determinant
        assert y >= y0 and z >= z0
        assert -a * y + 2 * z + forcing <= 0
        assert 2 * y - b * z + forcing <= 0
        assert _exact(box["phase_norm_upper_bound"]) == y + z
        assert box["certified"] is True
    assert (
        _exact(report["form_mean_error_floor"])
        == _exact(report["phase_mean_error_floor"])
        == 2 * eps / 29
    )
    span = _exact(report["origin_span"])
    phase_error = _exact(report["all_time_phase_error_upper_bound"])
    form_error = _exact(report["all_time_form_error_upper_bound"])
    phase_allowance = _exact(report["phase_allowance"])
    allowance = report["form_allowance_bounds"]
    assert phase_allowance == span / 2 == Q(1, 2000)
    assert (
        _exact(allowance["lo"])
        <= gamma_lo * span / 2
        <= gamma_hi * span / 2
        <= _exact(allowance["hi"])
    )
    assert phase_error < phase_allowance and form_error > _exact(allowance["hi"])
    assert _exact(report["phase_resolution_margin_bounds"]["lo"]) > 0
    assert _exact(report["form_resolution_margin_bounds"]["hi"]) < 0
    assert (
        report["all_time_envelopes_certified"]
        is report["phase_resolution_certified"]
        is True
    )
    assert (
        report["form_resolution_certified"]
        is report["joint_resolution_certified"]
        is False
    )
    assert report["status"] == "phase_only" and report["unavailable_reasons"] == []
    assert report["resolution_limitations"] == ["form_resolution_not_certified"]
    assert (
        saved["frozen_stopping_rule"]
        == manifest["frozen_stopping_rule"]
        == {
            "envelopes_admitted": True,
            "phase_resolution_passed": True,
            "form_resolution_passed": False,
            "joint_resolution_passed": False,
        }
    )
    assert (
        saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is False
    )


def test_form_tracking_retains_prior_result_and_independent_heat_gains(retained):
    manifests, content = retained
    manifest = manifests["port-form-tracking-v1.manifest.json"]
    protocol = json_loads(content["port-form-tracking-v1.protocol.json"])
    previous_protocol = json_loads(content["port-relaxation-v1.protocol.json"])
    saved = json_loads(content["port-form-tracking-v1.json"])
    previous = json_loads(content["port-relaxation-v1.json"])
    report = saved["report"]
    baseline = report["baseline_certificate"]
    # The separate proof method consumes the same primitives, including both
    # declared resolution policies. It never revises the first partial result.
    assert protocol["inputs"] == previous_protocol["inputs"]
    assert baseline == previous["report"]
    assert previous["frozen_stopping_rule_passed"] is False
    assert baseline["status"] == "phase_only"
    gamma = _exact(baseline["gamma_bounds"]["hi"])
    b2 = _exact(baseline["edge_disagreement_norm_squared_upper_bound"])
    assert _exact(report["bridge_hessian_variation_upper_bound"]) == b2
    bridge = b2 * _exact(baseline["even_bounds"]["phase_norm_upper_bound"])
    assert _exact(report["even_bridge_forcing_upper_bound"]) == bridge
    form_norms = []
    for sector in ("odd", "even"):
        box = report[f"{sector}_heat_bounds"]
        old = baseline[f"{sector}_bounds"]
        gap = _exact(old["gap_lower_bound"])
        initial = _exact(old["initial_norm_upper_bound"])
        forcing = _exact(old["forcing_upper_bound"])
        if sector == "even":
            forcing += bridge
        assert _exact(box["gap_lower_bound"]) == gap
        assert _exact(box["initial_norm_upper_bound"]) == initial
        assert _exact(box["forcing_upper_bound"]) == forcing
        lower = _exact(box["spectral_lower_bound"])
        upper = _exact(box["spectral_upper_bound"])
        assert lower == gap / 6 and upper == 2
        ratio = upper / lower
        assert _exact(box["spectral_ratio_upper_bound"]) == ratio
        bands = box["dyadic_band_count"]
        assert type(bands) is int and bands >= 0
        assert ratio <= 2**bands
        assert bands == 0 or 2 ** (bands - 1) < ratio
        heat = _exact(box["heat_integral_upper_bound"])
        gain = _exact(box["derivative_filter_gain_upper_bound"])
        assert heat == 1 + Q(21, 80) * bands and gain == 1 + heat
        loop = gamma**2 * upper * gain / gap
        assert _exact(box["loop_gain_upper_bound"]) == loop
        assert _exact(box["loop_margin"]) == 1 - loop > 0
        initial_exchange = gamma * upper * (1 + gamma) * initial / gap
        forced = gamma * gain * forcing / gap
        assert _exact(box["initial_form_contribution_upper_bound"]) == initial
        assert _exact(box["initial_joint_contribution_upper_bound"]) == initial_exchange
        assert _exact(box["forced_contribution_upper_bound"]) == forced
        norm = _exact(box["form_norm_upper_bound"])
        assert (1 - loop) * norm == initial + initial_exchange + forced
        assert norm >= initial and box["certified"] is True
        form_norms.append(norm)
    assert report["form_mean_error_floor"] == baseline["form_mean_error_floor"]
    assert report["phase_mean_error_floor"] == baseline["phase_mean_error_floor"]
    assert (
        report["all_time_phase_error_upper_bound"]
        == baseline["all_time_phase_error_upper_bound"]
    )
    assert (
        report["phase_resolution_margin_bounds"]
        == baseline["phase_resolution_margin_bounds"]
    )
    form = _exact(report["all_time_form_error_upper_bound"])
    floor = _exact(report["form_mean_error_floor"])
    # Check outward restoration independently of the producer's sqrt routine.
    assert form >= floor
    assert 2 * (form - floor) ** 2 >= sum(value**2 for value in form_norms)
    allowance = baseline["form_allowance_bounds"]
    margin = report["form_resolution_margin_bounds"]
    assert 0 < _exact(margin["lo"]) <= _exact(allowance["lo"]) - form
    assert _exact(allowance["hi"]) - form <= _exact(margin["hi"])
    assert report["status"] == "full"
    assert report["unavailable_reasons"] == report["resolution_limitations"] == []
    assert all(
        report[name] is True
        for name in (
            "all_time_envelopes_certified",
            "form_resolution_certified",
            "phase_resolution_certified",
            "joint_resolution_certified",
        )
    )
    assert (
        saved["frozen_stopping_rule"]
        == manifest["frozen_stopping_rule"]
        == {
            "envelopes_admitted": True,
            "phase_resolution_passed": True,
            "form_resolution_passed": True,
            "joint_resolution_passed": True,
        }
    )
    assert (
        saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is True
    )


def test_two_port_record_preserves_correlated_equilibria_and_null_control(retained):
    from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
    from tnfr.mathematics._rational_interval import I, pi_interval, sin

    manifests, content = retained
    manifest = manifests["two-port-compatibility-v1.manifest.json"]
    protocol = json_loads(content["two-port-compatibility-v1.protocol.json"])
    saved = json_loads(content["two-port-compatibility-v1.json"])
    primary, control = saved["report"], saved["matched_control"]["report"]
    expected_edges = sorted(
        {
            tuple(sorted((offset + j, offset + (j + 1) % 9)))
            for offset in (0, 9)
            for j in range(9)
        }
        | {(0, 9), (1, 10)}
    )
    pi = pi_interval()

    def interval(value):
        return I(_exact(value["lo"]), _exact(value["hi"]))

    def h(k, turn):
        return sin(2 * pi * ((k - turn) / 8)) - sin(2 * pi * turn)

    for inputs, report in (
        (protocol["inputs"], primary),
        (protocol["matched_control_inputs"], control),
    ):
        for name, value in inputs.items():
            assert report[name] == value
        geometry = report["geometry"]
        assert geometry["nodes"] == list(range(18))
        edges = list(map(tuple, geometry["edges"]))
        assert edges == expected_edges and geometry["cycle_rank"] == 3
        degrees = [sum(i in edge for edge in edges) for i in range(18)]
        assert report["degrees"] == degrees
        assert tuple(map(_exact, report["invariant_weights"])) == tuple(degrees)
        assert _exact(report["weighted_coordinate_mass"]) == sum(degrees) == 40
        assert all(_exact(value) == 0 for value in report["target_epi"])
        assert all(_exact(value) == 1 for value in report["capacity"])
        assert _exact(report["weighted_form_mean"]) == 0
        assert _exact(report["weighted_phase_mean"]) == 0
        nodal = tuple(
            tuple(map(_exact, row)) for row in report["nodal_turn_affine_coefficients"]
        )
        edge_affine = tuple(
            tuple(map(_exact, row)) for row in report["edge_turn_affine_coefficients"]
        )
        for column in range(3):
            assert sum(degree * row[column] for degree, row in zip(degrees, nodal)) == 0
        for index, (left, right) in enumerate(edges):
            offset = report["edge_integer_offsets"][index]
            assert edge_affine[index] == tuple(
                nodal[right][column]
                - nodal[left][column]
                - (offset if column == 0 else 0)
                for column in range(3)
            )
        assert report["named_cycles"] == [
            list(range(9)),
            list(range(9, 18)),
            [0, 9, 10, 1],
        ]
        assert report["named_cycle_periods"] == [*report["classes"], 0]
        for cycle, winding in zip(
            report["named_cycles"], report["named_cycle_periods"]
        ):
            total = [Q(0)] * 3
            for left, right in zip(cycle, cycle[1:] + cycle[:1]):
                index = edges.index(tuple(sorted((left, right))))
                sign = 1 if left < right else -1
                total = [a + sign * b for a, b in zip(total, edge_affine[index])]
            assert total == [winding, 0, 0]
        # Rebuild both the fine incidence identity and the full-support gap.
        symbols = tuple(
            tuple(map(_exact, row)) for row in report["edge_sine_symbol_coefficients"]
        )
        equations = tuple(
            tuple(map(_exact, row))
            for row in report["balance_equation_sine_coefficients"]
        )
        for node in range(18):
            row = tuple(
                sum(
                    (int(node == left) - int(node == right)) * value[column]
                    for (left, right), value in zip(edges, symbols)
                )
                for column in range(5)
            )
            assert row == tuple(
                map(_exact, report["nodal_sine_symbol_coefficients"][node])
            )
            factors = tuple(map(_exact, report["nodal_balance_multipliers"][node]))
            assert row == tuple(
                sum(
                    factor * equation[column]
                    for factor, equation in zip(factors, equations)
                )
                for column in range(5)
            )
        laplacian = tuple(
            tuple(
                Q(degrees[i] if i == j else -int(tuple(sorted((i, j))) in edges))
                for j in range(18)
            )
            for i in range(18)
        )
        assert laplacian == tuple(
            tuple(map(_exact, row)) for row in report["laplacian"]
        )
        gap = _exact(report["laplacian_gap_lower_bound"])
        assert gap == Q(2, 81)
        shifted = tuple(
            tuple(laplacian[i][j] - gap * (int(i == j) - Q(1, 18)) for j in range(18))
            for i in range(18)
        )
        assert exact_symmetric_semidefinite(shifted)
        cosine_lower = min(
            _exact(value["lo"]) for value in report["edge_cosine_bounds"]
        )
        assert _exact(report["minimum_cosine_lower_bound"]) == cosine_lower > 0
        assert _exact(report["phase_hessian_gap_lower_bound"]) == cosine_lower * gap > 0
        assert interval(report["acute_margin_turns_bounds"]).lo > 0
        assert all(
            interval(value).contains(0)
            for name in (
                "nodal_current_residual_bounds",
                "target_form_rate_bounds",
                "target_phase_rate_bounds",
            )
            for value in report[name]
        )
        assert (
            report["status"] == "certified_compatible"
            and report["unavailable_reasons"] == []
        )
        assert all(
            report[name] is True
            for name in (
                "implicit_equilibrium_certified",
                "full_nodal_residuals_consistent",
                "acute_geometry_certified",
                "local_attraction_certified",
            )
        )

    # Re-admit saved strict root evidence without calling a root solver.
    outer = primary["canonical_root_turn_bracket"]
    assert outer["refinements"] == protocol["inputs"]["outer_refinements"] == 32
    lower, upper = _exact(outer["lower"]), _exact(outer["upper"])
    assert upper - lower == Q(1, 9 * 2**32)
    assert (
        interval(outer["lower_residual"]).lo > 0 > interval(outer["upper_residual"]).hi
    )
    for index, endpoint in enumerate((lower, upper)):
        inner = primary["inner_root_brackets_at_outer_endpoints"][index]
        c_low, c_high = _exact(inner["lower"]), _exact(inner["upper"])
        assert inner["refinements"] == protocol["inputs"]["inner_refinements"] == 64
        assert c_high - c_low == Q(1, 9 * 2**64)
        assert (h(1, c_low) + h(2, endpoint)).lo > 0
        assert (h(1, c_high) + h(2, endpoint)).hi < 0
        value = h(2, endpoint) - sin(pi * (endpoint - I(c_low, c_high)))
        assert value.lo > 0 if index == 0 else value.hi < 0
    current = interval(primary["edge_current_bounds"][expected_edges.index((0, 9))])
    assert current.lo > 0
    assert primary["uniform_pair_excluded"] is True
    assert primary["uniform_pair_compatible"] is False
    assert control["canonical_root_turn_bracket"] is None
    assert control["uniform_pair_compatible"] is True
    assert control["uniform_pair_excluded"] is False
    for edge in ((0, 9), (1, 10)):
        current = interval(control["edge_current_bounds"][expected_edges.index(edge)])
        assert current.lo == current.hi == 0
    assert saved["frozen_stopping_rule"] == manifest["frozen_stopping_rule"]
    assert all(value is True for value in saved["frozen_stopping_rule"].values())
    assert (
        saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is True
    )


def test_capture_record_retains_source_target_and_radian_metric(retained):
    from tnfr._exact_time import exact_or_represented_real

    _, content = retained
    protocol = json_loads(content["two-port-capture-v1.protocol.json"])
    report = json_loads(content["two-port-capture-v1.json"])["report"]
    source, folded = report["preparation"], report["folded_reference"]
    target = report["target"]
    for name in ("form_error_radius", "phase_error_radius"):
        assert _exact(source[name]) == _exact(protocol["inputs"][name]) == Q(1, 65536)
    for name in ("reference_duration", "time_step"):
        assert _exact(report[name]) == _exact(protocol["inputs"][name])
    for name in ("order", "max_steps"):
        assert type(report[name]) is int and report[name] == protocol["inputs"][name]
    assert (
        source["geometry"]["nodes"] == protocol["support"]["nodes"] == list(range(18))
    )
    edges = source["geometry"]["edges"]
    assert edges == protocol["support"]["edges"]
    degrees = tuple(sum(i in edge for edge in edges) for i in range(18))
    assert source["degrees"] == list(degrees)
    assert sum(degrees) == protocol["support"]["degree_mass"] == 40
    nominal = tuple(
        Q(2 * i, 9) - Q(229, 360) if i < 9 else Q(1, 18) + Q(i - 9, 9) - Q(229, 360)
        for i in range(18)
    )
    assert tuple(map(_exact, source["nominal_phase_turns"])) == nominal
    assert sum(d * t for d, t in zip(degrees, nominal)) == 0
    offsets = tuple(
        2 if edge == [0, 8] else 1 if edge == [9, 17] else 0 for edge in edges
    )
    assert tuple(source["edge_integer_offsets"]) == offsets
    edge_turns = tuple(
        nominal[right] - nominal[left] - offset
        for (left, right), offset in zip(edges, offsets)
    )
    assert tuple(map(_exact, source["nominal_edge_turns"])) == edge_turns
    cycles = [list(range(9)), list(range(9, 18)), [0, 9, 10, 1]]
    assert source["named_cycles"] == protocol["support"]["cycles"] == cycles
    for cycle, period in zip(cycles, (2, 1, 0)):
        assert (
            sum(
                (1 if left < right else -1)
                * edge_turns[edges.index(sorted((left, right)))]
                for left, right in zip(cycle, cycle[1:] + cycle[:1])
            )
            == period
        )
    assert all(_exact(value) == 0 for value in source["nominal_epi"])
    assert all(_exact(value) == 1 for value in source["capacity"])
    for name, value in (
        ("storage_scale", Q(1)),
        ("epi_weight", Q(1023, 1024)),
        ("phase_weight", Q(1, 1024)),
    ):
        assert exact_or_represented_real(source["reference_model"][name], name) == value
    assert source["reference_model"]["phase_domain"] == "regular"
    assert source["named_cycle_periods"] == protocol["support"]["periods"] == [2, 1, 0]
    permutation = tuple(9 * (i // 9) + (1 - i % 9) % 9 for i in range(18))
    representatives = protocol["reference"]["representatives"]
    assert folded["permutation"] == list(permutation)
    assert folded["representatives"] == representatives == [0, 2, 3, 4, 9, 11, 12, 13]
    reconstruction = tuple(
        tuple(Q(int(i == r) - int(i == permutation[r])) for r in representatives)
        for i in range(18)
    )
    assert (
        tuple(tuple(map(_exact, row)) for row in folded["reconstruction_matrix"])
        == reconstruction
    )
    metric = tuple(tuple(map(_exact, row)) for row in folded["metric"])
    for i in range(8):
        for j in range(8):
            assert metric[i][j] == sum(
                degree * row[i] * row[j] for degree, row in zip(degrees, reconstruction)
            )
    # The existing record's strict-root and complete-nodal audit above checks
    # these target primitives independently. Equality here checks fresh-source
    # consistency, not root existence or provenance authentication by equality.
    assert target == json_loads(content["two-port-compatibility-v1.json"])["report"]
    assert report["root_outer_refinements"] == 32
    assert report["root_inner_refinements"] == 64


def test_capture_retained_reference_chain_and_full_edge_margins(retained):
    from tnfr.mathematics._rational_interval import pi_interval

    _, content = retained
    report = json_loads(content["two-port-capture-v1.json"])["report"]
    source, folded = report["preparation"], report["folded_reference"]
    reconstruction = tuple(
        tuple(map(_exact, row)) for row in folded["reconstruction_matrix"]
    )
    weights = tuple(_exact(folded["metric"][i][i]) for i in range(8))
    edge_turns = tuple(map(_exact, source["nominal_edge_turns"]))
    center, radius, elapsed = (Q(0),) * 8, Q(0), Q(0)
    margin = None
    pi = pi_interval()
    step_size = _exact(report["time_step"])
    horizon = _exact(report["reference_duration"])
    threshold = _exact(report["reference_margin_threshold"])
    assert threshold == Q(1, 2048)
    steps = report["reference_steps"]
    assert len(steps) <= report["max_steps"]
    for step in steps:
        assert _exact(step["time"]) == elapsed
        delta = _exact(step["duration"])
        assert delta == min(step_size, horizon - elapsed) > 0
        assert tuple(map(_exact, step["initial_center"])) == center
        assert _exact(step["initial_radius"]) == radius
        tube = tuple(
            (_exact(value["lo"]), _exact(value["hi"])) for value in step["tube"]
        )
        assert len(tube) == len(center) == 8
        for value, (lower, upper), weight in zip(center, tube, weights):
            assert lower <= value <= upper
            assert weight * (value - lower) ** 2 >= radius**2
            assert weight * (upper - value) ** 2 >= radius**2
        assert _exact(step["picard_interior_margin"]) > 0
        bounds = tuple(map(_exact, step["domain_lower_bounds"]))
        assert len(bounds) == 20 and min(bounds) > 0
        for bound, (left, right), turn in zip(
            bounds, source["geometry"]["edges"], edge_turns
        ):
            coefficients = tuple(
                b - a for a, b in zip(reconstruction[left], reconstruction[right])
            )
            lower = min(2 * pi.lo * turn, 2 * pi.hi * turn)
            upper = max(2 * pi.lo * turn, 2 * pi.hi * turn)
            for coefficient, (lo, hi) in zip(coefficients, tube):
                lower += min(coefficient * lo, coefficient * hi)
                upper += max(coefficient * lo, coefficient * hi)
            # Rebuild a valid margin directly from primitive affine edge gaps.
            # This checks compact evidence; it is not a replay of Taylor jets.
            assert bound <= pi.lo / 2 - max(abs(lower), abs(upper)) - threshold
        observed = min(bounds) + threshold
        margin = observed if margin is None else min(margin, observed)
        local = _exact(step["local_metric_error_upper_bound"])
        assert local >= 0
        expected_radius = radius + local
        scale = 1 << 128
        rounded = Q(
            -((-expected_radius.numerator * scale) // expected_radius.denominator),
            scale,
        )
        radius = _exact(step["endpoint_radius"])
        assert radius == rounded >= expected_radius
        center = tuple(map(_exact, step["endpoint_center"]))
        assert len(center) == 8
        elapsed += delta
    assert _exact(report["validated_reference_duration"]) == elapsed
    assert tuple(map(_exact, report["reference_endpoint_center"])) == center
    assert _exact(report["reference_endpoint_radius"]) == radius
    assert (
        None
        if report["reference_minimum_acute_margin"] is None
        else _exact(report["reference_minimum_acute_margin"])
    ) == margin
    assert report["reference_validated"] is (elapsed == horizon)
    if elapsed < horizon:
        assert report["status"] == "unavailable"
        assert report["unavailable_reasons"]
        assert report["capture_certified"] is False


def test_capture_stopping_rule_rebuilt_from_exact_handoff_evidence(retained):
    from tnfr.mathematics._rational_interval import pi_interval

    manifests, content = retained
    saved = json_loads(content["two-port-capture-v1.json"])
    report = saved["report"]
    manifest = manifests["two-port-capture-v1.manifest.json"]
    source = report["preparation"]
    rx, rt = (
        _exact(source[name]) for name in ("form_error_radius", "phase_error_radius")
    )
    energy0 = Q(10, 81) + 40 * rt + 40 * rx**2
    assert _exact(report["initial_excess_storage_upper_bound"]) == energy0 < Q(1, 8)
    z0 = 7 * rx / 3069
    joint = 7 * rt + z0 + Q(24, 100000)
    z = (z0 + Q(1, 100000) * (Q(5, 12) + 2 * joint)) / (1 - Q(1, 50000))
    error = joint + z
    assert _exact(report["joint_error_candidate"]) == joint
    assert _exact(report["scaled_form_norm_candidate"]) == z
    assert _exact(report["phase_error_candidate"]) == error < Q(1, 2048)
    assert _exact(report["comparison_margin"]) == Q(1, 2048) - error > 0
    assert _exact(report["full_slow_horizon"]) == 1025
    assert _exact(report["full_horizon_pi_squared_coefficient"]) == 1025 * 1023 * 1024
    pi = pi_interval()
    target_margin = _exact(report["target_acute_margin_lower_bound"])
    assert (
        Q(1, 8)
        < target_margin
        <= 2 * pi.lo * _exact(report["target"]["acute_margin_turns_bounds"]["lo"])
    )
    assert report["preparation_admitted"] is (energy0 < Q(1, 8) and error < Q(1, 2048))
    assert report["target_admitted"] is True
    assert report["full_comparison_certified"] is (
        report["preparation_admitted"]
        and report["target_admitted"]
        and report["reference_validated"]
        and report["reference_endpoint_certified"]
    )
    assert _exact(report["analytic_tail_slow_duration"]) == 1
    assert _exact(report["fast_tail_decay_upper_bound"]) == Q(1, 2**512)
    assert _exact(report["capture_barrier_lower_bound"]) == Q(1, 648000)
    if report["reference_validated"]:
        center = tuple(map(_exact, report["reference_endpoint_center"]))
        radius = _exact(report["reference_endpoint_radius"])
        distance = _exact(report["reference_target_distance_upper_bound"])
        pi = pi_interval()
        square = Q(0)
        for index, value in enumerate(center):
            node = report["folded_reference"]["representatives"][index]
            target = report["target"]["target_phase_bounds"][node]
            turn = _exact(source["nominal_phase_turns"][node])
            lower = (
                value - _exact(target["hi"]) + min(2 * pi.lo * turn, 2 * pi.hi * turn)
            )
            upper = (
                value - _exact(target["lo"]) + max(2 * pi.lo * turn, 2 * pi.hi * turn)
            )
            weight = _exact(report["folded_reference"]["metric"][index][index])
            square += weight * max(abs(lower), abs(upper)) ** 2
        assert distance >= radius and (distance - radius) ** 2 >= square
        assert report["reference_endpoint_certified"] is (distance <= Q(1, 2048))
    else:
        assert report["reference_target_distance_upper_bound"] is None
        assert report["reference_endpoint_certified"] is False
    if report["full_comparison_certified"]:
        phase = distance + error
        form = 3216 * (z / 2**512 + Q(1, 100000) * (Q(1, 1024) + 2 * error))
        energy = phase**2 + form**2
        assert (
            _exact(report["endpoint_phase_distance_upper_bound"]) == phase < Q(1, 1024)
        )
        assert (
            _exact(report["endpoint_relative_form_norm_upper_bound"])
            == form
            < Q(1, 8192)
        )
        assert (
            _exact(report["endpoint_excess_storage_upper_bound"])
            == energy
            < Q(65, 67108864)
        )
        assert _exact(report["capture_storage_margin"]) == Q(1, 648000) - energy > 0
        assert energy < Q(1, 12) ** 2
        assert report["capture_certified"] is True
    else:
        assert all(
            report[name] is None
            for name in (
                "endpoint_phase_distance_upper_bound",
                "endpoint_relative_form_norm_upper_bound",
                "endpoint_excess_storage_upper_bound",
                "capture_storage_margin",
            )
        )
        assert report["capture_certified"] is False
    stopping = {
        "complete_preparation_admitted": energy0 < Q(1, 8) and error < Q(1, 2048),
        "fresh_implicit_target_admitted": report["target_admitted"],
        "all_frozen_reference_steps_validated": len(report["reference_steps"]) == 4096
        and _exact(report["validated_reference_duration"]) == 1024,
        "strict_reference_tube_margins": report["reference_minimum_acute_margin"]
        is not None
        and _exact(report["reference_minimum_acute_margin"]) > Q(1, 2048),
        "reference_endpoint_target_distance": report["reference_endpoint_certified"],
        "full_family_comparison": report["full_comparison_certified"],
        "original_form_handoff": report["endpoint_relative_form_norm_upper_bound"]
        is not None
        and _exact(report["endpoint_relative_form_norm_upper_bound"]) < Q(1, 8192),
        "phase_handoff": report["endpoint_phase_distance_upper_bound"] is not None
        and _exact(report["endpoint_phase_distance_upper_bound"]) < Q(1, 1024),
        "strict_capture_storage_margin": report["capture_storage_margin"] is not None
        and _exact(report["capture_storage_margin"]) > 0,
        "full_law_convergence": report["capture_certified"]
        and not report["unavailable_reasons"],
    }
    assert saved["frozen_stopping_rule"] == manifest["frozen_stopping_rule"] == stopping
    assert (
        saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is all(stopping.values())
    )


def test_probe_retains_failed_export_and_separate_same_protocol_recovery(retained):
    manifests, content = retained
    stem = "two-port-probe-v1"
    saved = json_loads(content[f"{stem}.json"])
    protocol = json_loads(content[f"{stem}.protocol.json"])
    attempt = json_loads(content[f"{stem}.first-attempt.json"])
    history = saved["evaluation_history"]
    assert history == manifests[f"{stem}.manifest.json"]["evaluation_history"]
    assert (
        history["original_attempt"]
        == attempt["status"]
        == "assessment_completed_but_export_failed_without_retained_report"
    )
    assert attempt["retained_primary_report"] is False
    assert attempt["exception_type"] == "TypeError"
    assert attempt["original_evaluator_unchanged"] is True
    assert attempt["scientific_inputs_or_runtime_changed"] is False
    assert history["scientific_inputs_or_runtime_changed"] is False
    assert history["prior_capture_producer_replayed"] is False
    assert (
        history["retained_assessment_kind"]
        == "separately_frozen_export_recovery_recomputation"
    )
    assert history["failure_record"] == f"{stem}.first-attempt.json"
    assert history["recovery_source"] == attempt["recovery_source"]["file"]
    recovery = content[history["recovery_source"]]
    assert len(recovery) == attempt["recovery_source"]["bytes"]
    assert hashlib.sha256(recovery).hexdigest() == attempt["recovery_source"]["sha256"]
    assert (
        hashlib.sha256(content[f"{stem}.source.zip"]).hexdigest()
        == attempt["frozen_source_sha256"]
    )
    with zipfile.ZipFile(DIRECTORY / f"{stem}.source.zip") as archive:
        original = archive.read(
            "build/two-port-probe-freeze/evaluate_two_port_probe.py"
        )
    assert (
        original.count(b'saved["original_control_identity"] = _project(control)') == 1
    )
    assert b"{name: _project(value) for name, value in control.items()}" in recovery
    assert saved["prior_capture_artifacts"] == protocol["prior_capture_artifacts"]
    assert {item["file"] for item in protocol["prior_capture_artifacts"]} == {
        "two-port-capture-v1" + suffix
        for suffix in (".json", ".protocol.json", ".source.zip", ".manifest.json")
    }
    for item in protocol["prior_capture_artifacts"]:
        data = (DIRECTORY / item["file"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]


def test_probe_fixed_support_and_control_use_actual_degree_observations(retained):
    from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite

    _, content = retained
    saved = json_loads(content["two-port-probe-v1.json"])
    report = saved["report"]
    protocol = json_loads(content["two-port-probe-v1.protocol.json"])
    edges = {
        tuple(sorted((offset + j, offset + (j + 1) % 9)))
        for offset in (0, 9)
        for j in range(9)
    } | {(0, 9), (1, 10)}
    assert {tuple(edge) for edge in protocol["joined_support"]["edges"]} == edges
    assert {tuple(edge) for edge in report["geometry"]["edges"]} == edges
    degrees = tuple(sum(node in edge for edge in edges) for node in range(18))
    assert tuple(report["degrees"]) == degrees
    assert sum(degrees) == 40 and sum(degrees[9:]) == 20
    receiver = tuple(Q(degrees[i] * int(i >= 9), 20) for i in range(18))
    assert tuple(map(_exact, report["receiver_weights"])) == receiver
    control_edges = edges - {(0, 9), (1, 10)}
    assert {tuple(edge) for edge in report["disconnected_edges"]} == control_edges
    assert {
        tuple(edge) for edge in protocol["unjoined_control"]["edges"]
    } == control_edges
    assert report["disconnected_degrees"] == [2] * 18
    control_weights = tuple(map(_exact, report["disconnected_receiver_weights"]))
    assert control_weights == tuple(Q(int(i >= 9), 9) for i in range(18))
    assert receiver != control_weights
    donor = tuple(Q(i < 9) for i in range(18))
    assert tuple(map(_exact, report["donor_mask"])) == donor
    # Both complete control rows cancel edgewise in the receiver mean; the
    # same pulse has no edge difference and therefore no control work.
    assert all(
        control_weights[i] == control_weights[j] and donor[i] == donor[j]
        for i, j in control_edges
    )
    assert tuple(map(_exact, report["disconnected_increment_bounds"])) == (0, 0)
    assert tuple(map(_exact, report["disconnected_work_bounds"])) == (0, 0)
    amplitude = _exact(report["pulse_amplitude"])
    assert _exact(report["joined_form_mean_increment"]) == amplitude / 2
    assert tuple(map(_exact, report["disconnected_form_mean_increments"])) == (
        amplitude,
        0,
    )

    control = saved["original_control_identity"]
    ring = {edge for edge in control_edges if edge[1] < 9}
    gap = Q(1, 18)
    slack = tuple(
        tuple(
            Q(2 if i == j else -int(tuple(sorted((i, j))) in ring))
            - gap * (2 * int(i == j) - Q(2, 9))
            for j in range(9)
        )
        for i in range(9)
    )
    assert (
        tuple(tuple(map(_exact, row)) for row in control["normalized_gap_slack_matrix"])
        == slack
    )
    assert exact_symmetric_semidefinite(slack)
    rx = Q(protocol["preparation"]["original_form_error_radius"])
    ry = Q(protocol["preparation"]["original_phase_error_radius_radians"])
    initial = 18 * (rx**2 + ry**2)
    barrier = gap * Q(1, 25) * Q(1, 12) ** 2 / 2
    assert _exact(control["initial_combined_norm_squared_upper_bound"]) == initial
    assert _exact(control["initial_excess_storage_upper_bound"]) == initial
    assert _exact(control["barrier_lower_bound"]) == barrier
    assert _exact(control["storage_margin"]) == barrier - initial > 0
    assert _exact(control["radius_margin"]) == Q(1, 144) - initial > 0


@pytest.fixture(scope="module")
def capture_handoff_audit():
    from tnfr.research.sine_two_port_handoff import audit_sine_two_port_capture_handoff

    return audit_sine_two_port_capture_handoff(DIRECTORY)


def test_probe_response_work_and_stopping_rebuilt_from_primitive_evidence(
    retained, capture_handoff_audit
):
    manifests, content = retained
    saved = json_loads(content["two-port-probe-v1.json"])
    report = saved["report"]
    protocol = json_loads(content["two-port-probe-v1.protocol.json"])
    manifest = manifests["two-port-probe-v1.manifest.json"]
    values = {key: _exact(value) for key, value in protocol["inputs"].items()}
    assert values == {
        "form_radius": Q(1, 8192),
        "phase_radius": Q(1, 1024),
        "pulse_amplitude": Q(1, 2048),
        "probe_duration": Q(1, 4),
        "readout_error_bound": Q(1, 67108864),
        "contrast_threshold": Q(1, 262144),
        "work_allowance": Q(1, 2000000),
    }
    assert all(_exact(report[key]) == value for key, value in values.items())
    # Rebuild source-to-endpoint evidence without replaying any producer;
    # equality of old/new summary labels is not source admission.
    handoff = capture_handoff_audit
    assert saved["source_handoff"] == handoff.to_dict()
    assert handoff.numerical_execution_replayed is False
    assert handoff.provenance_authenticated is False
    old_target = json_loads(content["two-port-capture-v1.json"])["report"]["target"]
    assert report["target"] == old_target
    assert _exact(report["target_acute_margin_lower_bound"]) > Q(1, 8)
    x, y = values["form_radius"], values["phase_radius"]
    a, h = values["pulse_amplitude"], values["probe_duration"]
    noise = values["readout_error_bound"]
    q0 = x + Q(19, 6) * a
    c = Q(2, 3069) * h
    qmax = (q0 + c * y) / (1 - c**2)
    pmax = y + c * qmax
    background = Q(7, 120) * h * x
    sine_error = h * pmax / 9207
    lower = a * h * (1 - h) / 10 - background - sine_error
    upper = a * h / 10 + background + sine_error
    assert _exact(report["whole_window_form_norm_upper_bound"]) == qmax
    assert _exact(report["whole_window_phase_norm_upper_bound"]) == pmax
    assert _exact(report["response_error_upper_bound"]) == background + sine_error
    assert tuple(map(_exact, report["joined_increment_bounds"])) == (lower, upper)
    assert tuple(map(_exact, report["recorded_joined_increment_bounds"])) == (
        lower - 2 * noise,
        upper + 2 * noise,
    )
    assert tuple(map(_exact, report["recorded_disconnected_increment_bounds"])) == (
        -2 * noise,
        2 * noise,
    )
    assert tuple(map(_exact, report["recorded_contrast_bounds"])) == (
        lower - 4 * noise,
        upper + 4 * noise,
    )
    response_margin = lower - 4 * noise - values["contrast_threshold"]
    assert _exact(report["response_margin"]) == response_margin > 0
    work = (a * a - Q(7, 6) * a * x, a * a + Q(7, 6) * a * x)
    assert tuple(map(_exact, report["joined_work_bounds"])) == work
    assert (
        _exact(report["work_allowance_margin"])
        == values["work_allowance"] - work[1]
        > 0
    )
    energy = x * x + y * y + work[1]
    radius_margin = Q(1, 144) - q0 * q0 - y * y
    storage_margin = Q(1, 648000) - energy
    assert _exact(report["post_probe_excess_storage_upper_bound"]) == energy
    assert _exact(report["post_probe_radius_margin"]) == radius_margin > 0
    assert _exact(report["capture_storage_margin"]) == storage_margin > 0
    identity = (
        handoff.target_acute_margin_lower_bound > Q(1, 8)
        and radius_margin > 0
        and storage_margin > 0
    )
    control = saved["original_control_identity"]
    stopping = {
        "prior_capture_content_association": True,
        "complete_source_handoff": handoff.endpoint_form_radius < x
        and handoff.endpoint_phase_radius < y,
        "fresh_implicit_target_admitted": handoff.target_acute_margin_lower_bound
        > Q(1, 8),
        "strict_recorded_response_margin": response_margin > 0,
        "exact_unjoined_null_and_zero_work": all(
            _exact(value) == 0
            for name in ("disconnected_increment_bounds", "disconnected_work_bounds")
            for value in report[name]
        ),
        "strictly_positive_supplied_work": work[0] > 0,
        "supplied_work_within_allowance": work[1] <= values["work_allowance"],
        "joined_identity_retained": identity,
        "joined_recovery": identity,
        "original_unjoined_identity_retained": _exact(control["radius_margin"]) > 0
        and _exact(control["storage_margin"]) > 0,
    }
    assert saved["frozen_stopping_rule"] == manifest["frozen_stopping_rule"] == stopping
    assert (
        saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is all(stopping.values())
    )
    assert report["status"] == "certified_probe" and not report["unavailable_reasons"]


def test_dipole_original_source_and_finite_metric_warmup(
    retained, capture_handoff_audit
):
    _, content = retained
    saved = json_loads(content["two-port-dipole-v1.json"])
    report = saved["report"]
    protocol = json_loads(content["two-port-dipole-v1.protocol.json"])
    inputs = {key: _exact(value) for key, value in protocol["inputs"].items()}
    assert inputs == {
        "warmup_duration": Q(285934809600000),
        "form_radius": Q(1, 2**40),
        "phase_radius": Q(1, 2**40),
        "phase_increment": Q(1, 2**12),
        "probe_duration": Q(1, 2**10),
        "readout_error_bound": Q(1, 2**50),
        "contrast_threshold": Q(1, 2**38),
        "work_allowance": Q(1, 2**21),
    }
    assert all(_exact(report[key]) == value for key, value in inputs.items())
    assert saved["source_handoff"] == capture_handoff_audit.to_dict()
    assert saved["prior_capture_artifacts"] == protocol["prior_capture_artifacts"]
    for item in protocol["prior_capture_artifacts"]:
        data = (DIRECTORY / item["file"]).read_bytes()
        assert type(item["bytes"]) is int and len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    assert {item["file"] for item in protocol["prior_capture_artifacts"]} == {
        "two-port-capture-v1" + suffix
        for suffix in (".json", ".protocol.json", ".source.zip", ".manifest.json")
    }
    control = saved["original_control_handoff"]
    rx = Q(protocol["preparation"]["original_form_error_radius"])
    ry = Q(protocol["preparation"]["original_phase_error_radius_radians"])
    ring_energy = 18 * (rx**2 + ry**2)
    assert _exact(control["initial_ring_excess_storage_upper_bound"]) == ring_energy
    assert (
        _exact(control["initial_ring_combined_norm_squared_upper_bound"]) == ring_energy
    )
    assert ring_energy < _exact(control["ring_capture_barrier"]) == Q(1, 129600)
    assert _exact(control["total_excess_storage_upper_bound"]) == 2 * ring_energy
    assert (
        _exact(control["full_combined_norm_squared_upper_bound"]) == 9000 * ring_energy
    )
    assert 9000 * ring_energy < _exact(control["source_radius_squared"]) == Q(1, 144)
    assert control["admitted"] is True

    # Rebuild the modified-energy coefficients and returns directly. The
    # original-source trapping is independently required above, not inferred
    # from a product of norm caps or from a previous warmup flag.
    gap, maximum, cosine = Q(1, 90), Q(2), Q(1, 25)
    eta_lo, eta_hi = Q(1, 11000000), Q(1, 9000000)
    epsilon = gap / 4
    mu, stiffness = eta_lo * cosine * gap**2, eta_hi * maximum**2
    amin = mu / 2 + gap**2 / 16
    amax = stiffness / 2 + epsilon * maximum / 2 + epsilon**2
    kappa = min(4 * (gap - epsilon) / 3, epsilon * mu / amax)
    assert _exact(report["lyapunov_decay_rate"]) == kappa
    assert _exact(report["lyapunov_position_lower_coefficient"]) == amin
    assert _exact(report["lyapunov_position_upper_coefficient"]) == amax
    assert kappa * inputs["warmup_duration"] == 128
    assert report["warmup_decay_power"] == 128
    decay = Q(1, 2**128)
    assert _exact(report["warmup_decay_upper_bound"]) == decay
    initial = (3 * eta_hi * maximum / 4 + amax / gap) * Q(1, 12) ** 2
    assert _exact(report["initial_lyapunov_upper_bound"]) == initial
    returned = initial * decay
    x2, y2 = 4 * returned / (eta_lo * gap), maximum * returned / amin
    assert (
        _exact(report["warmup_form_norm_squared_upper_bound"])
        == x2
        < inputs["form_radius"] ** 2
    )
    assert (
        _exact(report["warmup_phase_norm_squared_upper_bound"])
        == y2
        < inputs["phase_radius"] ** 2
    )
    assert _exact(report["warmup_form_margin"]) == inputs["form_radius"] ** 2 - x2 > 0
    assert _exact(report["warmup_phase_margin"]) == inputs["phase_radius"] ** 2 - y2 > 0
    assert capture_handoff_audit.numerical_execution_replayed is False
    assert capture_handoff_audit.provenance_authenticated is False


def test_dipole_fixed_local_readout_retains_each_full_support(retained):
    from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite

    _, content = retained
    report = json_loads(content["two-port-dipole-v1.json"])["report"]
    protocol = json_loads(content["two-port-dipole-v1.protocol.json"])
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in range(18))
    assert tuple(map(_exact, report["dipole"])) == q == tuple(protocol["input"]["q"])
    for model, key in enumerate(("joined_edges", "unjoined_edges")):
        edges = {tuple(edge) for edge in protocol["support"][key]}
        stored = (
            report["geometry"]["edges"] if model == 0 else report["disconnected_edges"]
        )
        assert {tuple(edge) for edge in stored} == edges
        degrees = tuple(sum(i in edge for edge in edges) for i in range(18))
        assert tuple(report["degrees_by_model"][model]) == degrees
        assert sum(degrees[i] * q[i] ** 2 for i in range(18)) == 4
        assert sum(q[i] ** 2 / degrees[i] for i in range(18)) == 1
        assert sum((q[i] - q[j]) ** 2 for i, j in edges) == 6
        assert {j if i == 4 else i for i, j in edges if 4 in (i, j)} == {3, 5}
        assert {j if i == 5 else i for i, j in edges if 5 in (i, j)} == {4, 6}
        components = (
            (tuple(range(18)),)
            if model == 0
            else (tuple(range(9)), tuple(range(9, 18)))
        )
        assert all(sum(degrees[i] * q[i] for i in piece) == 0 for piece in components)
        laplacian = tuple(
            tuple(
                Q(degrees[i] if i == j else -int(tuple(sorted((i, j))) in edges))
                for j in range(18)
            )
            for i in range(18)
        )
        assert (
            tuple(tuple(map(_exact, row)) for row in report["laplacians"][model])
            == laplacian
        )
        lower = tuple(
            tuple(
                laplacian[i][j]
                - Q(1, 90)
                * (
                    degrees[i] * int(i == j)
                    - sum(
                        (
                            Q(degrees[i] * degrees[j], sum(degrees[k] for k in piece))
                            for piece in components
                            if i in piece and j in piece
                        ),
                        Q(0),
                    )
                )
                for j in range(18)
            )
            for i in range(18)
        )
        assert (
            tuple(
                tuple(map(_exact, row))
                for row in report["normalized_gap_slack_matrices"][model]
            )
            == lower
        )
        assert exact_symmetric_semidefinite(lower)
    assert report["laplacians"][0] != report["laplacians"][1]


def test_dipole_finite_response_work_heat_and_stops_from_primitives(
    retained, capture_handoff_audit
):
    from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin

    manifests, content = retained
    saved = json_loads(content["two-port-dipole-v1.json"])
    r = saved["report"]
    protocol = json_loads(content["two-port-dipole-v1.protocol.json"])
    values = {key: _exact(value) for key, value in protocol["inputs"].items()}
    x, y, a, h, noise = (
        values[key]
        for key in (
            "form_radius",
            "phase_radius",
            "phase_increment",
            "probe_duration",
            "readout_error_bound",
        )
    )

    def interval(raw):
        return I(_exact(raw["lo"]), _exact(raw["hi"]))

    def equal_interval(raw, value):
        assert (_exact(raw["lo"]), _exact(raw["hi"])) == (value.lo, value.hi)

    old_target = json_loads(content["two-port-capture-v1.json"])["report"]["target"]
    assert r["target"] == old_target
    pi = pi_interval()
    gamma, g = 1 / (1023 * pi), Q(1, 3069)
    b = 2 * pi * interval(old_target["bulk_arc_turn_bounds"][0])
    b0 = 4 * pi / 9
    cosine_gap = cos(b0 - a / 2) - cos(b - a / 2)
    equal_interval(r["cosine_gap_bounds"], cosine_gap)
    assert cosine_gap.lo > Q(1, 32)
    c = 2 * g * h
    qmax = (x + c * (y + 2 * a)) / (1 - c * c)
    error = 2 * h * x + 4 * g * a * h * h + 2 * g * h * y + 4 * g * g * h * h * qmax
    assert _exact(r["finite_remainder_upper_bound"]) == error
    ideal = 2 * gamma * h * sin(I(Q(3, 2) * a)) * cosine_gap
    recorded = ideal + I(-2 * error - 4 * noise, 2 * error + 4 * noise)
    equal_interval(r["ideal_correlated_contrast_bounds"], ideal)
    equal_interval(r["recorded_contrast_bounds"], recorded)
    assert (
        _exact(r["response_margin"]) == recorded.lo - values["contrast_threshold"] > 0
    )
    # A separate coarse analytic witness still passes; it uses neither the
    # displayed contrast endpoints nor the fresh target's numerical gap.
    coarse_lower = 5 * a * h / (64 * 3216) - 2 * error - 4 * noise
    assert recorded.lo > coarse_lower > values["contrast_threshold"]
    works = []
    for model, angle in enumerate((b, b0)):
        work = (
            3 * cos(angle)
            - 2 * cos(angle + a)
            - cos(angle - 2 * a)
            + I(-4 * a * y, 4 * a * y)
        )
        works.append(work)
        equal_interval(r["work_bounds_by_model"][model], work)
        assert 0 < 3 * a * a / 25 - 4 * a * y <= work.lo
        assert work.hi <= 3 * a * a + 4 * a * y < values["work_allowance"]
        assert (
            _exact(r["work_allowance_margins"][model])
            == values["work_allowance"] - work.hi
        )
        assert (
            _exact(r["capture_storage_margins"][model])
            == Q(1, 648000) - x * x - y * y - work.hi
            > 0
        )
    radius_margin = Q(1, 144) - x * x - (y + 2 * a) ** 2
    assert _exact(r["post_probe_radius_margin"]) == radius_margin > 0
    # The alternative uses its own heat contraction from its original source,
    # rather than treating the sine warmup or a causal null as a raw null.
    heat_exponent = Q(1, 90) * values["warmup_duration"]
    assert _exact(r["heat_warmup_exponent"]) == heat_exponent >= 128
    heat_form = Q(7, 65536 * 2**128)
    assert _exact(r["heat_warmup_form_norm_upper_bound"]) == heat_form < x
    heat_bound = 4 * h * x + 4 * noise
    assert _exact(r["phase_blind_recorded_contrast_upper_bound"]) == heat_bound > 0
    assert _exact(r["phase_blind_exclusion_margin"]) == recorded.lo - heat_bound > 0
    handoff = capture_handoff_audit
    control = saved["original_control_handoff"]
    stopping = {
        "prior_capture_content_association": True,
        "complete_joined_source_handoff": handoff.endpoint_form_radius**2
        + handoff.endpoint_phase_radius**2
        < Q(1, 144)
        and handoff.endpoint_excess_storage_upper_bound < Q(1, 648000),
        "original_unjoined_trapping_and_full_norm": _exact(
            control["initial_ring_excess_storage_upper_bound"]
        )
        < Q(1, 129600)
        and _exact(control["full_combined_norm_squared_upper_bound"]) < Q(1, 144),
        "fresh_implicit_target_admitted": handoff.target_acute_margin_lower_bound
        > Q(1, 8),
        "finite_complete_law_warmup": _exact(r["warmup_form_margin"]) > 0
        and _exact(r["warmup_phase_margin"]) > 0,
        "strict_acquired_geometry_cosine_gap": cosine_gap.lo > Q(1, 32),
        "strict_recorded_response_margin": recorded.lo > values["contrast_threshold"],
        "phase_blind_original_source_warmup": heat_exponent >= 128 and heat_form <= x,
        "phase_blind_response_excluded": recorded.lo > heat_bound,
        "positive_supplied_work_both_models": all(work.lo > 0 for work in works),
        "supplied_work_within_both_allowances": all(
            work.hi <= values["work_allowance"] for work in works
        ),
        "both_identities_retained": radius_margin > 0
        and all(x * x + y * y + work.hi < Q(1, 648000) for work in works),
        "both_models_recover": radius_margin > 0
        and all(x * x + y * y + work.hi < Q(1, 648000) for work in works),
    }
    manifest = manifests["two-port-dipole-v1.manifest.json"]
    assert saved["frozen_stopping_rule"] == manifest["frozen_stopping_rule"] == stopping
    assert (
        saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is all(stopping.values())
    )
    assert r["status"] == "certified_dipole" and not r["unavailable_reasons"]


def _inference_decode(value):
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            return _exact(value)
        return {key: _inference_decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_inference_decode(item) for item in value)
    return value


def _inference_interval(value):
    from tnfr.mathematics._rational_interval import I

    return None if value is None else I(value["lo"], value["hi"])


def _documentation_suffix(current, prospective):
    """Compare maintained text across checkout EOLs, not archived byte hashes."""
    current = current.replace(b"\r\n", b"\n")
    prospective = prospective.replace(b"\r\n", b"\n")
    return current[len(prospective) :] if current.startswith(prospective) else None


@pytest.mark.parametrize("current_eol", (b"\n", b"\r\n"))
@pytest.mark.parametrize("prospective_eol", (b"\n", b"\r\n"))
def test_documentation_prefix_accepts_only_checkout_newline_changes(
    current_eol, prospective_eol
):
    prefix = b"# Prospective proof\n\nUnchanged mathematics.\n"
    suffix = b"\n# Retained result\n"
    current = (prefix + suffix).replace(b"\n", current_eol)
    prospective = prefix.replace(b"\n", prospective_eol)
    assert _documentation_suffix(current, prospective) == suffix
    assert (
        _documentation_suffix(current.replace(b"Unchanged", b"Changed"), prospective)
        is None
    )
    assert (
        _documentation_suffix(
            current, prospective.replace(b"mathematics", b"math\rematics")
        )
        is None
    )


@pytest.fixture(scope="module")
def inference_evidence(retained):
    manifests, content = retained
    protocol = _inference_decode(
        json_loads(content["two-port-inference-v1.protocol.json"])
    )
    raw = json_loads(content["two-port-inference-v1.json"])
    return (
        protocol,
        _inference_decode(raw),
        raw,
        manifests["two-port-inference-v1.manifest.json"],
    )


def test_inference_archive_retains_prospective_proof_and_separate_public_worker(
    retained,
):
    manifests, content = retained
    with zipfile.ZipFile(DIRECTORY / "two-port-inference-v1.source.zip") as archive:
        source = json_loads(archive.read("source-manifest.json"))
        assert set(source["runtime_overlays"]) == {
            "src/tnfr/mathematics/_validated_taylor.py",
            "src/tnfr/physics/relational_sine_two_port_readout.py",
            "src/tnfr/sdk/relational_reports.py",
        }
        proof_path = "theory/nodal/SINE_TWO_PORT_INFERENCE.md"
        prospective = archive.read(proof_path)
        current = (Path(__file__).parents[2] / proof_path).read_bytes()
        suffix = _documentation_suffix(current, prospective)
        assert suffix is not None
        assert b"sine-two-port-inference-protocol" in prospective
        assert b"sine-two-port-inference-result" not in prospective
        assert b"sine-two-port-inference-result" in suffix
        # Parse archived worker source, never execute it. Its only inference
        # calls consume the admitted public inputs and two declared overrides.
        worker_path = "build/two-port-inference-freeze/invert_public_packet.py"
        tree = ast.parse(archive.read(worker_path))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "infer_sine_two_port_geometry"
        ]
        assert len(calls) == 3
        assert all(
            not call.args and len(call.keywords) == 1 and call.keywords[0].arg is None
            for call in calls
        )
        direct = [
            call.keywords[0].value
            for call in calls
            if isinstance(call.keywords[0].value, ast.Name)
        ]
        assert len(direct) == 1 and direct[0].id == "inputs"
        changed = [
            call.keywords[0].value
            for call in calls
            if isinstance(call.keywords[0].value, ast.Dict)
        ]
        assert len(changed) == 2
        assert {node.keys[1].value for node in changed} == {
            "readout_gain_bounds",
            "bulk_angle_bounds",
        }
        assert all(
            node.keys[0] is None
            and isinstance(node.values[0], ast.Name)
            and node.values[0].id == "inputs"
            and len(node.keys) == 2
            for node in changed
        )
        imports = [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        assert "tnfr.physics.relational_sine_two_port_inference" in imports
        assert not any(
            name and ("readout" in name or "capture" in name) for name in imports
        )
    assert hashlib.sha256(
        content["two-port-inference-v1.source.zip"]
    ).hexdigest() == next(
        item["sha256"]
        for item in manifests["two-port-inference-v1.manifest.json"]["artifacts"]
        if item["file"].endswith(".source.zip")
    )


def test_inference_calibration_is_rebuilt_only_from_separate_reference_readings(
    inference_evidence,
):
    protocol, saved, _, _ = inference_evidence
    declared, retained_calibration = protocol["calibration"], saved["calibration"]
    references = declared["reference_values"]
    gain, offset = declared["hidden_gain"], declared["hidden_offset"]
    errors, delta = (
        declared["hidden_reading_errors"],
        declared["per_reading_error_bound"],
    )
    assert references == (-1, 1)
    assert gain == Q(3, 2) and offset == Q(5, 7)
    assert delta == Q(1, 2**14)
    assert errors == (delta / 2, -delta / 4)
    readings = tuple(gain * z + offset + error for z, error in zip(references, errors))
    assert retained_calibration["readings"] == readings
    center = (readings[1] - readings[0]) / (references[1] - references[0])
    uncertainty = 2 * delta / (references[1] - references[0])
    gain_bounds = (center - uncertainty, center + uncertainty)
    assert retained_calibration["gain_bounds"] == gain_bounds
    assert gain_bounds[0] > 1 and gain_bounds[0] < gain < gain_bounds[1] < 2
    assert gain_bounds[1] - gain_bounds[0] == Q(1, 8192)
    assert retained_calibration["reference_values"] == references
    assert retained_calibration["per_reading_error_bound"] == delta
    assert retained_calibration["admitted"] is all(
        abs(error) <= delta for error in errors
    )


def _inference_source(protocol, case, index):
    from tnfr.mathematics._rational_interval import I, pi_interval

    edges = tuple(
        sorted(
            {tuple(sorted((o + j, o + (j + 1) % 9))) for o in (0, 9) for j in range(9)}
            | {(0, 9), (1, 10)}
        )
    )
    degrees = tuple(sum(i in edge for edge in edges) for i in range(18))
    assert protocol["support"]["edges"] == edges
    assert protocol["support"]["degrees"] == degrees
    assert protocol["support"]["nodes"] == tuple(range(18))
    assert sum(degrees) == 40
    expected = ((Q(45, 32), Q(3, 4)), (Q(23, 16), Q(5, 6)), (Q(47, 32), Q(11, 12)))
    b, c = expected[index]
    assert (case["bulk_angle"], case["receiver_short_angle"]) == (b, c)
    assert case["common_form"] == Q(index + 1, 7)
    assert case["common_phase"] == -Q(index + 1, 5)
    raw_x = tuple(Q((i + 1) * (index + 1), 2**52) for i in range(18))
    raw_theta = tuple(Q((7 * i + 5 * index) % 19 + 1, 2**52) for i in range(18))
    for raw, label in ((raw_x, "form"), (raw_theta, "phase")):
        assert case[f"raw_{label}_residual"] == raw
        mean = sum(d * value for d, value in zip(degrees, raw)) / 40
        centered = tuple(value - mean for value in raw)
        assert case[f"{label}_residual"] == centered
        assert all(centered)
        assert sum(d * value for d, value in zip(degrees, centered)) == 0
        assert sum(d * value**2 for d, value in zip(degrees, centered)) < Q(1, 2**80)
    pi = pi_interval()
    short, bulk = 4 * pi - 8 * b, (2 * pi - c) / 8
    delta = (short - c) / 2
    raw = (I(0),) + tuple(short + (j - 1) * b for j in range(1, 9))
    raw += (delta,) + tuple(delta + c + (j - 1) * bulk for j in range(1, 9))
    mean = sum((d * value for d, value in zip(degrees, raw)), I(0)) / 40
    nominal = tuple(value - mean for value in raw)
    form = tuple(I(case["common_form"] + value) for value in case["form_residual"])
    phase = tuple(
        value + case["common_phase"] + error
        for value, error in zip(nominal, case["phase_residual"])
    )
    x2 = sum(
        d * (value - case["common_form"]).abs_max ** 2
        for d, value in zip(degrees, form)
    )
    y2 = sum(
        d * (value - base - case["common_phase"]).abs_max ** 2
        for d, value, base in zip(degrees, phase, nominal)
    )
    actual = b - (case["phase_residual"][1] - case["phase_residual"][0]) / 8
    assert actual == b - Q(7, 2**55)
    actual_box = (4 * pi - (phase[1] - phase[0])) / 8
    assert actual_box.contains(actual)
    return degrees, form, phase, x2, y2, actual, actual_box


@pytest.mark.parametrize("index", range(3))
def test_inference_complete_sources_are_rebuilt_without_reset(
    inference_evidence, index
):
    protocol, saved, _, _ = inference_evidence
    case, entry = protocol["hidden_cases"][index], saved["cases"][index]
    degrees, form, phase, x2, y2, actual, actual_box = _inference_source(
        protocol, case, index
    )
    public = protocol["public_inputs_without_observation_or_gain"]
    response = entry["response"]["report"]
    assert entry["id"] == case["id"] == f"case-{index + 1}"
    assert tuple(map(_inference_interval, response["initial_form_bounds"])) == form
    assert tuple(map(_inference_interval, response["initial_phase_bounds"])) == phase
    assert response["degrees"] == degrees
    audit = entry["source_audit"]
    assert audit["full_form_norm_squared_upper"] == x2 < public["form_radius"] ** 2
    assert audit["full_phase_norm_squared_upper"] == y2 < public["phase_radius"] ** 2
    assert audit["actual_pre_probe_arc_mean"] == actual
    assert _inference_interval(entry["actual_arc_mean_source_box"]) == actual_box
    assert audit["admitted"] is (
        x2 < public["form_radius"] ** 2 and y2 < public["phase_radius"] ** 2
    )
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in range(18))
    assert response["dipole"] == q
    post_event = form + tuple(
        value + public["phase_increment"] * coefficient
        for value, coefficient in zip(phase, q)
    )
    assert (
        tuple(map(_inference_interval, response["post_event_initial_box"]))
        == post_event
    )
    assert (
        response["capacity"] == protocol["complete_model"]["capacity"] == (Q(1),) * 18
    )
    model = response["reference_model"]
    assert model["epi_weight"] == protocol["complete_model"]["loss"] == Q(1023, 1024)
    assert model["phase_weight"] == protocol["complete_model"]["exchange"] == Q(1, 1024)
    assert model["storage_scale"] == protocol["complete_model"]["beta"] == 1
    assert (
        response["law"]
        == protocol["complete_model"]["law"]
        == "normalized_sine_reciprocal_exchange"
    )
    assert response["clock"] == "tau=e*t"


@pytest.mark.parametrize("index", range(3))
def test_inference_retained_taylor_arithmetic_and_independent_picard_inclusion(
    inference_evidence, index
):
    from tnfr.mathematics._rational_interval import I, pi_interval, sin

    protocol, saved, _, _ = inference_evidence
    response = saved["cases"][index]["response"]["report"]
    step = response["step"]
    h = protocol["public_inputs_without_observation_or_gain"]["probe_duration"]
    assert h == Q(1, 2**16)
    assert step["time"] == 0 and step["duration"] == h
    assert response["order"] == step["order"] == protocol["numerics"]["order"] == 4
    assert (
        protocol["numerics"]["steps"] == 1
        and protocol["numerics"]["picard_iteration_cap"] == 16
    )
    assert protocol["numerics"]["interval_bits"] == 128
    initial = tuple(map(_inference_interval, step["initial_box"]))
    tube = tuple(map(_inference_interval, step["tube"]))
    series = tuple(tuple(map(_inference_interval, row)) for row in step["series"])
    remainder = tuple(map(_inference_interval, step["local_remainder_bounds"]))
    assert len(initial) == len(tube) == len(series) == len(remainder) == 36
    assert initial == tuple(
        map(_inference_interval, response["post_event_initial_box"])
    )
    assert all(len(row) == 5 and row[0] == value for row, value in zip(series, initial))
    rebuilt = []
    for row, tail in zip(series, remainder):
        polynomial = row[4]
        for coefficient in (row[3], row[2], row[1]):
            polynomial = polynomial * h + coefficient
        rebuilt.append(polynomial * h + tail)
    increment = tuple(rebuilt)
    assert increment == tuple(map(_inference_interval, step["increment"]))
    endpoint = tuple(
        I(max((value + change).lo, whole.lo), min((value + change).hi, whole.hi))
        for value, change, whole in zip(initial, increment, tube)
    )
    assert endpoint == tuple(map(_inference_interval, step["endpoint"]))
    assert (
        _inference_interval(response["true_increment_bounds"])
        == increment[4] - increment[5]
    )
    assert (
        _inference_interval(response["baseline_readout_bounds"])
        == initial[4] - initial[5]
    )
    assert (
        _inference_interval(response["endpoint_readout_bounds"])
        == endpoint[4] - endpoint[5]
    )
    assert step["picard_interior_margin"] > 0 and step["domain_lower_bounds"] == (Q(1),)
    # Re-admit the retained tube using independently assembled complete rows.
    # This does not recompute the Taylor jets/remainders or authenticate the
    # recorded execution; those validated derivative enclosures remain premises.
    edges, degrees = protocol["support"]["edges"], protocol["support"]["degrees"]
    gradient, currents = [I(0) for _ in range(18)], [I(0) for _ in range(18)]
    for i, j in edges:
        difference = tube[i] - tube[j]
        gradient[i] += difference
        gradient[j] -= difference
        current = sin(tube[18 + j] - tube[18 + i])
        currents[i] += current
        currents[j] -= current
    gamma = 1 / (1023 * pi_interval())
    rates = tuple((-a + gamma * f) / d for a, f, d in zip(gradient, currents, degrees))
    rates += tuple(gamma * a / d for a, d in zip(gradient, degrees))
    images = tuple(value + I(0, h) * rate for value, rate in zip(initial, rates))
    assert (
        min(
            min(image.lo - whole.lo, whole.hi - image.hi)
            for image, whole in zip(images, tube)
        )
        > 0
    )
    assert response["status"] == "admitted" and response["failed_tube"] is None


def test_inference_readings_public_packets_and_all_stops_are_rebuilt(
    inference_evidence,
):
    from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin

    protocol, saved, raw, manifest = inference_evidence
    keys = {
        "bulk_angle_bounds",
        "receiver_short_angle_bounds",
        "form_radius",
        "phase_radius",
        "phase_increment",
        "probe_duration",
        "recorded_increment_bounds",
        "readout_error_bound",
        "readout_gain_bounds",
        "refinements",
    }
    assert set(protocol["public_packet"]["inverse_keys"]) == keys
    public = protocol["public_inputs_without_observation_or_gain"]
    assert set(public) == keys - {"recorded_increment_bounds", "readout_gain_bounds"}
    gain, offset = (
        protocol["calibration"][key] for key in ("hidden_gain", "hidden_offset")
    )
    gain_bounds = saved["calibration"]["gain_bounds"]
    all_stops = {
        "calibration_admitted": gain_bounds[0] > 0
        and gain_bounds[0] <= gain <= gain_bounds[1]
        and all(
            abs(v) <= protocol["calibration"]["per_reading_error_bound"]
            for v in protocol["calibration"]["hidden_reading_errors"]
        )
    }
    x, y, a, h, noise = (
        public[key]
        for key in (
            "form_radius",
            "phase_radius",
            "phase_increment",
            "probe_duration",
            "readout_error_bound",
        )
    )
    assert x == y == Q(1, 2**40) and a == Q(1, 1024) and h == Q(1, 65536)
    assert noise == Q(1, 2**60) and public["refinements"] == 48
    for index, (case, entry, original) in enumerate(
        zip(protocol["hidden_cases"], saved["cases"], raw["cases"])
    ):
        _, _, _, x2, y2, actual, actual_box = _inference_source(protocol, case, index)
        response, step = (
            entry["response"]["report"],
            entry["response"]["report"]["step"],
        )
        error_before, error_after = (
            case["reading_error_before"],
            case["reading_error_after"],
        )
        assert error_before == (index - 1) * noise / 2
        assert error_after == (1 - 2 * index) * noise / 4
        baseline = _inference_interval(response["baseline_readout_bounds"])
        endpoint = _inference_interval(response["endpoint_readout_bounds"])
        change = _inference_interval(response["true_increment_bounds"])
        assert baseline.contains(-Q(index + 1, 2**52))
        before, after = (
            gain * baseline + offset + error_before,
            gain * endpoint + offset + error_after,
        )
        assert _inference_interval(entry["readings"]["before"]) == before
        assert _inference_interval(entry["readings"]["after"]) == after
        correlated = gain * change + (error_after - error_before)
        separate = after - before
        recorded = I(max(correlated.lo, separate.lo), min(correlated.hi, separate.hi))
        assert _inference_interval(entry["readings"]["recorded_increment"]) == recorded
        packet = entry["public_packet"]
        assert set(packet) == {"schema", "inputs", "controls"}
        assert packet["schema"] == protocol["public_packet"]["schema"]
        assert set(packet["inputs"]) == keys
        assert packet["inputs"] == {
            **public,
            "readout_gain_bounds": gain_bounds,
            "recorded_increment_bounds": (recorded.lo, recorded.hi),
        }
        assert packet["controls"] == protocol["controls"]
        request = json.dumps(
            original["public_packet"], sort_keys=True, allow_nan=False
        ).encode("utf-8")
        outputs = entry["inverse_outputs"]
        assert outputs["request_sha256"] == hashlib.sha256(request).hexdigest()
        primary, broad, false = (
            outputs[key]["report"] for key in ("primary", "broad_gain", "false_prior")
        )
        for output, override in (
            (primary, {}),
            (broad, {"readout_gain_bounds": protocol["controls"]["broad_gain_bounds"]}),
            (
                false,
                {"bulk_angle_bounds": protocol["controls"]["false_bulk_angle_bounds"]},
            ),
        ):
            expected = {**packet["inputs"], **override}
            assert {key: output[key] for key in keys} == expected
            assert output["source_admitted"] and output["finite_response_certified"]
        nominal = _inference_interval(primary["nominal_bulk_angle_outer_bounds"])
        actual_primary = _inference_interval(
            primary["actual_long_arc_mean_outer_bounds"]
        )
        actual_broad = _inference_interval(broad["actual_long_arc_mean_outer_bounds"])
        assert actual_primary == nominal + I(-y / 8, y / 8)
        assert (
            false["nominal_bulk_angle_outer_bounds"]
            is false["actual_long_arc_mean_outer_bounds"]
            is None
        )
        assert actual_primary.contains(actual) and actual_primary.contains(actual_box)
        # Rebuild the necessary false-prior exclusion without running inverse
        # refinement or using its cached verdict as a premise.
        g = Q(1, 3069)
        qmax = (x + g * h * (4 + 4 * a + 2 * y)) / (1 - 4 * g * g * h * h)
        error = (
            2 * h * x
            + 4 * g * h**4 / 3
            + 4 * g * a * h * h
            + 2 * g * h * y
            + 4 * g * g * h * h * qmax
        )
        assert primary["finite_remainder_upper_bound"] == error
        scale = 2 / (1023 * pi_interval()) * h * sin(I(3 * a / 2))
        wrong = protocol["controls"]["false_bulk_angle_bounds"]
        wrong_cosine = I(cos(I(wrong[1] - a / 2)).lo, cos(I(wrong[0] - a / 2)).hi)
        wrong_recorded = I(*gain_bounds) * (
            -scale * wrong_cosine + I(-error, error)
        ) + I(-2 * noise, 2 * noise)
        false_excluded = (
            recorded.hi < wrong_recorded.lo or recorded.lo > wrong_recorded.hi
        )
        assert false_excluded
        heat = 2 * gain_bounds[1] * h * x + 2 * noise
        assert entry["phase_blind_recorded_bound"] == heat
        stops = {
            "source_admitted": x2 < x * x
            and y2 < y * y
            and public["bulk_angle_bounds"][0]
            <= case["bulk_angle"]
            <= public["bulk_angle_bounds"][1]
            and public["receiver_short_angle_bounds"][0]
            <= case["receiver_short_angle"]
            <= public["receiver_short_angle_bounds"][1]
            and all(case["form_residual"])
            and all(case["phase_residual"])
            and actual_box.contains(actual),
            "complete_horizon": response["status"] == "admitted"
            and step["time"] == 0
            and step["duration"] == h
            and len(step["endpoint"]) == 36
            and step["picard_interior_margin"] > 0,
            "reading_errors_admitted": max(abs(error_before), abs(error_after))
            <= noise,
            "numerical_readout_width": recorded.width
            <= protocol["numerics"]["max_recorded_increment_width"],
            "public_only_packet": outputs["request_sha256"]
            == hashlib.sha256(request).hexdigest(),
            "primary_bounded_candidate": primary["status"] == "bounded_candidate",
            "nominal_angle_covered": nominal.contains(case["bulk_angle"]),
            "actual_angle_covered": actual_primary.contains(actual_box),
            "primary_resolution": actual_primary.width
            < protocol["thresholds"]["primary_actual_angle_width"],
            "broad_gain_bounded_candidate": broad["status"] == "bounded_candidate",
            "broad_gain_contains_primary": actual_broad.contains(actual_primary),
            "broad_gain_loses_resolution": actual_broad.width
            > protocol["thresholds"]["broad_gain_angle_width"],
            "false_prior_excluded": false["status"] == "incompatible"
            and false_excluded,
            "phase_blind_alternative_excluded": recorded.hi < -heat
            or recorded.lo > heat,
        }
        assert entry["stopping_rule"] == stops and len(stops) == 14
        all_stops.update({f"{case['id']}.{key}": value for key, value in stops.items()})
    assert len(saved["cases"]) == 3 and len(all_stops) == 43
    assert (
        saved["frozen_stopping_rule"] == manifest["frozen_stopping_rule"] == all_stops
    )
    assert (
        saved["frozen_stopping_rule_passed"]
        is manifest["frozen_stopping_rule_passed"]
        is all(all_stops.values())
    )
    assert (
        saved["report"]["status"]
        == manifest["status"]
        == "certified_reserved_inference"
    )

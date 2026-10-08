"""Read-only content and stopping-rule checks for retained formed-C9 evidence.

No producing assessor or archived script is executed. Content consistency is
not independent authentication of chronology or physical acquisition. Runtime
source may legitimately evolve after the retained snapshot; full Git history
is not required to check these archived bytes and their revision references.
"""

import hashlib
import re
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
)
MANIFESTS = ("evidence.manifest.json",) + tuple(
    f"{stem}.manifest.json" for stem in ARCHIVED
)


@pytest.fixture(scope="module", autouse=True)
def no_evidence_producers():
    from tnfr.physics import (
        _sine_formed_contact,
        relational_sine_formed_class_contact,
        relational_sine_formed_class_maintenance,
        relational_sine_formed_classes,
        relational_sine_port_composition,
        relational_sine_port_form_tracking,
        relational_sine_port_relaxation,
        relational_sine_reduced_class_ports,
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
                        )
                    )
                    or name == "_unprobed_handoff"
                ):
                    patch.setattr(module, name, forbidden)
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
    expected = {"pair-v1.json", "response-v1.json"} | {
        stem + suffix
        for stem in ARCHIVED
        for suffix in (".json", ".protocol.json", ".source.zip")
    }
    names = [
        item["file"]
        for manifest in manifests.values()
        for item in manifest["artifacts"]
    ]
    assert len(names) == len(set(names)) == 20 and set(names) == expected
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

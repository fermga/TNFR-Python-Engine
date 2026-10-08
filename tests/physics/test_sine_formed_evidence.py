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
ARCHIVED = ("maintenance-v1", "contact-v1", "reduced-ports-v1")
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
            relational_sine_reduced_class_ports,
        ):
            for name in vars(module):
                if (
                    name.startswith(
                        (
                            "assess_sine_formed",
                            "assess_sine_reduced",
                            "evaluate_sine_reduced",
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


def test_all_eleven_retained_artifact_sizes_and_hashes(retained):
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
    assert len(names) == len(set(names)) == 11 and set(names) == expected
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

"""Portable saved-map fixtures; no historical admission or runtime execution."""

import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F

import pytest

from benchmarks import thol_regional_map_symmetry as study
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.regional_response import observe_regional_response
from tnfr.physics.support_transport import _from_data
from tnfr.research.claims import ClaimStatus
from tnfr.research.core_manifests import CoreExperimentManifest


def _manifest(claim=study.INPUT_CLAIM):
    return CoreExperimentManifest(
        claim_id=claim,
        git_sha="a" * 40,
        source_dirty=False,
        versions={"python": "fixture"},
        graph_construction="Detached four-node symmetric support",
        capacity_specification="Uniform positive fixture capacities",
        solver="Exact arithmetic fixture",
        timestep=None,
        seed=None,
        result_status=ClaimStatus.DERIVED,
        operator_sequence=("reception",),
        telemetry=("map symmetry",),
        controls=("synthetic retained fixture",),
        artifacts=("fixture.json",),
    ).to_dict()


def _fixture(*, ordered=True):
    nodes, children = ("p0", "p1", "c0", "c1"), ("c0", "c1")
    region, metric = (2, 3), (F(2), F(2), F(1), F(1))
    support = ((1, 2), (0, 3), (0,), (1,))
    edges = tuple((i, j, F(1)) for i, row in enumerate(support) for j in row)
    capacities, pressure = (F(1),) * 4, (F(0),) * 4
    epi_weight, dt = F(1, 2), F(1, 4)
    identity = tuple(tuple(F(i == j) for j in range(4)) for i in range(4))
    a = tuple(
        tuple(
            epi_weight * (F(i == j) - F(j in support[i], len(support[i])))
            for j in range(4)
        )
        for i in range(4)
    )
    s = identity
    local_rows = []
    for i, neighbors in enumerate(support):
        coefficients = tuple(
            (
                (F(i == j) / 2 + F(j in neighbors, 2 * len(neighbors)))
                if ordered
                else F(i == j)
            )
            for j in range(4)
        )
        local_rows.append({"node": nodes[i], "row": coefficients, "offset": F(0)})
    if ordered:
        for i, neighbors in enumerate(support):
            changed = tuple(
                s[i][j] / 2
                + sum((s[k][j] for k in neighbors), F(0)) / (2 * len(neighbors))
                for j in range(4)
            )
            s = s[:i] + (changed,) + s[i + 1 :]
    t = tuple(
        tuple(x - dt * y for x, y in zip(sr, ar, strict=True))
        for sr, ar in zip(s, a, strict=True)
    )
    c = pressure
    common = {
        "A": a,
        "S": s,
        "T": t,
        "c": c,
        "metric_weights": metric,
        "dt": dt,
        "paired_b_cancels": True,
        "cross_experiment_b_equal": False,
    }
    original = _from_data(nodes, edges, support, (1, 2, 3, 4), capacities, pressure)
    reference = {
        "source": asdict(original),
        "metric_weights": metric,
        "strengths": metric,
        "epi_weight": epi_weight,
    }
    pairs, summarized = {}, {}
    for number, name in enumerate(study.geometry.WITNESSES):
        pair = {}
        source = tuple(F(i + number) for i in range(4))
        control = tuple(F(i + 1) for i in range(4))
        difference = tuple(F((i == 2) if number else (i in region)) for i in range(4))
        for side, x in (
            ("control", control),
            (
                "perturbed",
                tuple(x + d for x, d in zip(control, difference, strict=True)),
            ),
        ):
            snapshot = _from_data(nodes, edges, support, x, capacities, pressure)
            pair[side] = {
                "snapshot": asdict(snapshot),
                "A": a,
                "S": s,
                "c": c,
                "b": source,
                "rows": deepcopy(local_rows),
                "vectors": {"x0": x},
                "observation": {
                    "snapshot": asdict(snapshot),
                    "epi_weight": epi_weight,
                    "forcing": source,
                    "phase": tuple(F(i, 8) for i in range(4)),
                },
            }
        pair["paired_source_difference"] = pressure
        pairs[name] = pair
        stages = {
            label: asdict(observe_regional_response(matrix, metric, region, difference))
            for label, matrix in (("reception", s), ("held_pressure_interval", t))
        }
        summarized[name] = {"stages": stages, "source_difference": pressure}
    return study._payload(
        {
            "manifest": _manifest(),
            "nodes": nodes,
            "children": children,
            "region_indices": region,
            "common_coefficients": common,
            "admission": {
                "original_reference": reference,
                "children": children,
                "common_coefficients": deepcopy(common),
                "witnesses": pairs,
            },
            "witnesses": summarized,
        }
    )


def _write_fixture(tmp_path, retained=None):
    path = tmp_path / "criterion.json"
    path.write_text(
        json.dumps(_fixture() if retained is None else retained), encoding="utf-8"
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _cli(monkeypatch, path, digest, output):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "symmetry",
            "--input",
            str(path),
            "--expected-sha256",
            digest,
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        study, "current_git_source_provenance", lambda *a: ("a" * 40, False, None)
    )


def test_ordered_map_can_break_support_symmetry_and_restrict_no_inputs():
    result = study.analyze_retained(_fixture())
    symmetry = result["symmetry"]
    groups = dict(symmetry.operator_group_indices)
    assert symmetry.support_group_order == len(symmetry.admissible_indices) == 2
    assert len(groups["A"]) == 2
    assert (
        len(groups["S"]) == len(groups["T"]) == len(symmetry.common_group_indices) == 1
    )
    assert len(symmetry.fixed_input_basis) == 3
    for row in result["common_group_environment"].values():
        assert row["same_input_basis_as_unrestricted"]
        assert row["fixed_input_rank"] == row["unrestricted_input_rank"] == 1
        assert row["fixed_input_protected_dimension"] == 0
    fields = dict(symmetry.field_group_indices)
    assert len(fields["reception_offset_c"]) == 2
    assert len(fields["cohort_source_rate_b"]) == 1
    assert result["cross_experiment_source_equal"] is False
    assert (
        result["native_calls"]
        == result["kernel_calls"]
        == result["new_trajectories"]
        == 0
    )
    assert result["historical_admission_calls"] == 0
    assert not result["complete_runtime_equivariance_certified"]
    local = result["local_reception_family"]
    assert len(local["group_indices"]) == 2
    assert all(row["preserved"] and row["witness"] is None for row in local["checks"])
    assert local["ordered_product_reconstructed"]


def test_symmetric_maps_restrict_algebraic_inputs_even_if_captured_fields_are_asymmetric():
    result = study.analyze_retained(_fixture(ordered=False))
    symmetry = result["symmetry"]
    assert len(symmetry.common_group_indices) == 2
    assert len(symmetry.fixed_input_basis) == 2
    assert symmetry.common_operator_orbits == ((0, 1), (2, 3))
    assert len(dict(symmetry.field_group_indices)["cohort_source_rate_b"]) == 1
    held = result["common_group_environment"]["held_pressure_interval"]
    assert held["unrestricted_input_rank"] == 1
    assert held["fixed_input_rank"] == 0
    assert held["fixed_input_protected_dimension"] == 1
    assert not held["same_input_basis_as_unrestricted"]
    assert not result["input_reachability_certified"]
    assert not result["repeated_invariance_certified"]


def test_fixed_image_columns_are_recomputed_from_shared_centering_and_maps():
    result = study.analyze_retained(_fixture(ordered=False))
    matrices = dict(result["symmetry"].operators)
    for row in result["common_group_environment"].values():
        matrix = matrices[row["transition_key"]]
        expected = tuple(
            tuple(
                dot(center, tuple(dot(mrow, value) for mrow in matrix))
                for value in row["fixed_input_basis"]
            )
            for center in row["centering"]
        )
        assert row["fixed_input_map"] == expected


def test_every_captured_epi_phase_and_affine_offset_is_named_and_bound():
    retained = _fixture()
    result = study.analyze_retained(retained)
    fields = dict(result["symmetry"].fields)
    assert len(fields) == 13
    for name in study.geometry.WITNESSES:
        for side in ("control", "perturbed"):
            raw = retained["admission"]["witnesses"][name][side]
            assert fields[f"{name}_{side}_generation_epi"] == tuple(
                map(F, raw["snapshot"]["epi"])
            )
            assert fields[f"{name}_{side}_generation_phase_coordinates"] == tuple(
                map(F, raw["observation"]["phase"])
            )
        b = fields[f"{name}_source_rate_b"]
        assert fields[f"{name}_held_affine_offset_c_plus_hb"] == tuple(v / 4 for v in b)


@pytest.mark.parametrize("name", study.geometry.WITNESSES)
@pytest.mark.parametrize("side", ("control", "perturbed"))
@pytest.mark.parametrize(
    "change",
    (
        "A",
        "S",
        "c",
        "b",
        "forcing",
        "epi",
        "observation_snapshot",
        "epi_weight",
        "local_coefficient",
        "local_offset",
    ),
)
def test_each_saved_side_must_match_common_domain_and_source(name, side, change):
    retained = _fixture()
    row = retained["admission"]["witnesses"][name][side]
    if change in ("A", "S"):
        row[change][0][0] = "99"
    elif change in ("c", "b"):
        row[change][0] = "99"
    elif change == "forcing":
        row["observation"]["forcing"][0] = "99"
    elif change == "epi":
        row["vectors"]["x0"][0] = "99"
    elif change == "observation_snapshot":
        raw = row["observation"]["snapshot"]
        raw["epi"][0] = "99"
    elif change == "epi_weight":
        row["observation"]["epi_weight"] = "99"
    elif change == "local_coefficient":
        row["rows"][0]["row"][0] = "99"
    else:
        row["rows"][0]["offset"] = "99"
    with pytest.raises(ValueError, match="mismatch|differs"):
        study.analyze_retained(retained)


@pytest.mark.parametrize(
    "change",
    (
        "metric",
        "nodal_A",
        "strengths",
        "paired_assertion",
        "cross_assertion",
        "cross_boolean",
        "paired_difference",
        "summary_difference",
        "missing_witness",
    ),
)
def test_original_nodal_bindings_and_pair_assertions_are_checked(change):
    retained = _fixture()
    common = retained["common_coefficients"]
    reference = retained["admission"]["original_reference"]
    if change == "metric":
        # Both retained copies agree; H=d/nu still has to be independently true.
        common["metric_weights"][0] = "99"
        reference["metric_weights"][0] = "99"
    elif change == "nodal_A":
        common["A"][0][0] = "99"
        common["T"][0][0] = str(F(common["S"][0][0]) - F(99, 4))
    elif change == "strengths":
        reference["strengths"][0] = "99"
    elif change == "paired_assertion":
        common["paired_b_cancels"] = False
    elif change == "cross_assertion":
        common["cross_experiment_b_equal"] = True
    elif change == "cross_boolean":
        common["cross_experiment_b_equal"] = 0
    elif change == "paired_difference":
        retained["admission"]["witnesses"]["cohort"]["paired_source_difference"][
            0
        ] = "99"
    elif change == "summary_difference":
        retained["witnesses"]["cohort"]["source_difference"][0] = "99"
    else:
        del retained["admission"]["witnesses"]["cohort"]
    retained["admission"]["common_coefficients"] = deepcopy(common)
    with pytest.raises(ValueError):
        study.analyze_retained(retained)


def test_original_capture_fields_do_not_get_silently_truncated():
    retained = _fixture()
    retained["admission"]["witnesses"]["cohort"]["control"]["observation"][
        "phase"
    ].pop()
    with pytest.raises(ValueError, match="match the full node space"):
        study.analyze_retained(retained)


@pytest.mark.parametrize(
    "change", ("order", "count", "row_dimension", "all_coefficients", "all_offsets")
)
def test_local_rows_reconstruct_only_the_declared_order_and_offset(change):
    retained = _fixture()
    for witness in retained["admission"]["witnesses"].values():
        for side in ("control", "perturbed"):
            rows = witness[side]["rows"]
            if change == "order":
                rows.reverse()
            elif change == "count":
                rows.pop()
            elif change == "row_dimension":
                rows[0]["row"].pop()
            elif change == "all_coefficients":
                rows[0]["row"][0] = "99"
            else:
                rows[0]["offset"] = "99"
    with pytest.raises(ValueError, match="Reception|product|offset"):
        study.analyze_retained(retained)


def test_cap_exhaustion_does_not_return_a_partial_report():
    with pytest.raises(ValueError, match="cap"):
        study.analyze_retained(_fixture(), cap=1)


def test_real_manifest_and_cli_serialize_all_exact_owner_evidence(
    tmp_path, monkeypatch
):
    path, digest = _write_fixture(tmp_path)
    output = tmp_path / "symmetry.json"
    _cli(monkeypatch, path, digest, output)

    def forbidden(*args, **kwargs):
        raise AssertionError("no historical admission, runtime or scalar kernel calls")

    monkeypatch.setattr(study.reset, "audit_reset_step", forbidden)
    monkeypatch.setattr(study.reset, "load_evidence", forbidden)
    monkeypatch.setattr(study.reset, "neighbor_epi_blend_value", forbidden)
    monkeypatch.setattr(study.reset, "neighbor_epi_represented_affine_row", forbidden)
    study.main()
    report = json.loads(output.read_bytes())
    CoreExperimentManifest(**report["manifest"]).validate_for_admission()
    assert report["manifest"]["claim_id"] == "O3.a-regional-map-symmetry"
    assert report["historical_inputs"]["criterion"]["sha256"] == digest
    assert report["source_scope"] == ["src/tnfr", "benchmarks"]
    assert report["symmetry"]["support_group_order"] == 2
    assert report["symmetry"]["permutations"][1]["operator_checks"]
    assert report["symmetry"]["fields"]
    assert (
        report["common_group_environment"]["held_pressure_interval"]["fixed_input_rank"]
        == 1
    )


@pytest.mark.parametrize("damage", ("digest", "claim", "manifest"))
def test_authentication_failure_precedes_analysis(tmp_path, monkeypatch, damage):
    retained = _fixture()
    if damage == "claim":
        retained["manifest"]["claim_id"] = "other-claim"
    elif damage == "manifest":
        retained["manifest"]["timestep"] = -1
    path, digest = _write_fixture(tmp_path, retained)
    if damage == "digest":
        digest = "0" * 64
    output = tmp_path / "symmetry.json"
    _cli(monkeypatch, path, digest, output)

    def forbidden(*args, **kwargs):
        raise AssertionError("no arithmetic before authentication")

    monkeypatch.setattr(study, "analyze_retained", forbidden)
    with pytest.raises(ValueError):
        study.main()
    assert not output.exists()


def test_input_overwrite_refused_before_reading(tmp_path, monkeypatch):
    path, digest = _write_fixture(tmp_path)
    before = path.read_bytes()
    _cli(monkeypatch, path, digest, path)

    def forbidden(*args, **kwargs):
        raise AssertionError("overwrite must be rejected before read")

    monkeypatch.setattr(study.reset, "_load", forbidden)
    with pytest.raises(ValueError, match="overwrite"):
        study.main()
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ("input", "source"))
def test_postanalysis_changes_prevent_output(tmp_path, monkeypatch, change):
    path, digest = _write_fixture(tmp_path)
    output = tmp_path / "symmetry.json"
    _cli(monkeypatch, path, digest, output)
    original = study.analyze_retained

    def analyze(retained):
        result = original(retained)
        if change == "input":
            path.write_text("{}", encoding="utf-8")
        else:
            monkeypatch.setattr(
                study,
                "current_git_source_provenance",
                lambda *a: ("b" * 40, False, None),
            )
        return result

    monkeypatch.setattr(study, "analyze_retained", analyze)
    with pytest.raises(RuntimeError, match="changed"):
        study.main()
    assert not output.exists()

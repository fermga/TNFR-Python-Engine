"""Static cell obstructions preserve the saved state and their evidence lineage."""

import hashlib
import json
import sys
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_temporal_compatibility as campaign

DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts/research"
INPUTS = tuple(DIRECTORY / name for name in campaign.INPUT_NAMES)


@pytest.fixture(scope="module")
def parents():
    if any(not path.is_file() for path in INPUTS):
        pytest.skip("the retained B40 through B26 evidence chain is unavailable")
    return tuple(json.loads(path.read_bytes()) for path in INPUTS)


@pytest.fixture(scope="module")
def report(parents):
    return campaign.analyze_c6_winding_temporal_compatibility(*parents)


def test_saved_endpoint_and_static_pressure_rows_keep_separate_provenance(
    report, parents
):
    source = report["source"]
    assert source["source_report_replayed"] is True
    assert (
        campaign._payload(source["saved_B40_state"])
        == parents[0]["B40_continuation"]["endpoint"]
    )
    rows = tuple(
        tuple(point["state"]["epi"])
        for point in parents[0]["B38_static_pressure_balance"]["points"]
    )
    assert source["B38_visible_rows"] == rows
    assert source["phase"] == tuple(parents[0]["source"]["phase"])
    assert any(source["saved_B40_state"]["remainder"])
    assert report["new_saved_trajectory_steps"] == 0
    assert report["static_cell_witness_is_saved_continuation"] is False


def test_complete_cell_obstruction_uses_exact_uniform_negative_step_bound(report):
    section = report["B41_complete_cell_obstruction"]
    delta = F(1, 2**54)
    assert section["grid_quantum"] == F(1, 2**3222)
    assert section["minimum_cell_grid_width"] == delta / 2 - 2 * section["grid_quantum"]
    assert section["maximum_negative_increment"] / delta == F(
        5757725321284133, 36028797018963968
    )
    assert (
        0 < section["maximum_negative_increment"] < section["minimum_cell_grid_width"]
    )
    assert section["finite_complete_cell_union_invariance_excluded"] is True
    for flag in (
        "correlated_carry_region_excluded",
        "saved_trajectory_escape_certified",
        "future_runtime_certified",
    ):
        assert section[flag] is False
    assert "closure" not in section


def test_hypothetical_witness_changes_carry_without_claiming_a_saved_continuation(
    report,
):
    witness = report["B41_hypothetical_cell_witness"]
    saved = report["source"]["saved_B40_state"]
    assert len(witness["epi_states"]) == 8 and witness["selected_index"] == 7
    assert witness["state"]["epi"] == saved["epi"] == witness["epi_states"][7]
    assert witness["state"]["remainder"] != saved["remainder"]
    assert witness["epi_states"][:7] == report["source"]["B38_visible_rows"]
    assert witness["saved_trajectory_escape_certified"] is False
    assert witness["family_invariance_excluded"] is True
    assert witness["band_failure"] is False
    before = tuple(
        F(x) + r
        for x, r in zip(
            witness["state"]["epi"], witness["state"]["remainder"], strict=True
        )
    )
    assert witness["exact_increment"] == tuple(
        F(value) / 16 for value in witness["pressure"]
    )
    assert witness["exact_candidate"] == tuple(
        x + a for x, a in zip(before, witness["exact_increment"], strict=True)
    )
    assert witness["endpoint"]["before"] == witness["state"]
    assert witness["endpoint"]["nodal_balance_residual"] == (F(0),) * 6
    assert witness["endpoint"]["after"]["epi"] not in witness["epi_states"]
    assert witness["endpoint_visible_sum"] - witness["source_visible_sum"] == F(
        1, 2**52
    )
    assert "obstruction" not in witness


def test_seven_cell_deadline_binds_exactly_the_original_seven_pressures(
    report, parents
):
    section = report["B42_finite_cell_graph"]
    assert section["epi_states"] == report["source"]["B38_visible_rows"]
    pressures = tuple(
        tuple(point["observation"]["pressure"])
        for point in parents[0]["B38_static_pressure_balance"]["points"]
    )
    assert section["pressures"] == pressures
    edges = tuple(
        (i, j)
        for i, row in enumerate(section["adjacency"])
        for j, edge in enumerate(row)
        if edge
    )
    assert edges == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (5, 5), (6, 6))
    assert tuple(item["first_exit_step"] for item in section["maximal_residence"]) == (
        77,
        51,
        39,
        51,
        40,
        30,
        50,
    )
    assert section["family_exit_step_bounds"] == (77, 51, 39, 51, 40, 70, 50)
    assert section["maximum_steps_until_family_exit_or_band_failure"] == 77
    assert (
        section["finite_family_escape_certified"]
        is section["carried_cycle_in_family_excluded"]
        is True
    )
    assert "reference" not in section
    for flag in (
        "whole_band_exit_certified",
        "future_runtime_certified",
        "reachable_from_saved_state_certified",
    ):
        assert section[flag] is False


def test_static_campaign_never_enters_the_runtime_branch_or_mutates_inputs(
    parents, monkeypatch
):
    def forbidden():
        raise AssertionError(
            "the static cell study must not invoke a graph preparation"
        )

    monkeypatch.setattr(campaign, "_runtime_bridge", forbidden)
    original = deepcopy(parents)
    result = campaign.analyze_c6_winding_temporal_compatibility(*parents)
    assert parents == original
    assert result["B43_original_phase_tail_runtime"] is None
    assert result["new_saved_trajectory_steps"] == 0


def test_runtime_branch_receives_no_saved_state_and_does_not_promote_its_reachability(
    parents, monkeypatch
):
    calls = []
    detached = {"separate_preparation_marker": "mocked branch, no runtime evidence"}

    def bridge():
        calls.append(())
        return detached

    monkeypatch.setattr(campaign, "_runtime_bridge", bridge)
    result = campaign.analyze_c6_winding_temporal_compatibility(
        *parents, include_runtime=True
    )
    assert calls == [()]
    assert result["B43_original_phase_tail_runtime"] is detached
    assert result["new_saved_trajectory_steps"] == 0
    assert result["B40_origin_reachability_certified"] is False
    assert result["static_cell_witness_is_saved_continuation"] is False


def test_scope_flags_and_exact_payload_remain_json_serializable(report):
    for flag in (
        "infinite_band_invariance_certified",
        "full_runtime_stability_certified",
        "B40_origin_reachability_certified",
    ):
        assert report[flag] is False
    assert report["B43_original_phase_tail_runtime"] is None
    encoded = campaign._payload(report)
    json.dumps(encoded, allow_nan=False)
    assert (
        encoded["B42_finite_cell_graph"][
            "maximum_steps_until_family_exit_or_band_failure"
        ]
        == 77
    )


@pytest.mark.parametrize(
    "change",
    (
        "claim",
        "scope",
        "carry",
        "phase",
        "balance_weight",
        "flag",
        "ancestor_pressure",
        "lineage",
    ),
)
def test_tampered_parent_or_ancestor_cannot_certify_a_new_temporal_result(
    parents, change
):
    values = deepcopy(parents)
    parent = values[0]
    if change == "claim":
        parent["manifest"]["claim_id"] = "unrelated"
    elif change == "scope":
        parent["source_scope"] = ["benchmarks"]
    elif change == "carry":
        parent["B40_continuation"]["endpoint"]["remainder"] = ["0"] * 6
    elif change == "phase":
        parent["source"]["phase"][4] = 0.0
    elif change == "balance_weight":
        parent["B38_static_pressure_balance"]["weights"][0] = "0"
    elif change == "flag":
        parent["original_tail_reachability_certified"] = True
    elif change == "ancestor_pressure":
        values[2]["continuation"]["steps"][0]["pressure"][4] = 0.0
    else:
        parent["input_evidence"][0]["producer_manifest"]["claim_id"] = "unrelated"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_temporal_compatibility(*values)


@pytest.mark.parametrize("value", (None, 0, 1, "yes"))
def test_runtime_request_requires_explicit_boolean_before_any_history_work(value):
    with pytest.raises(TypeError, match="boolean"):
        campaign.analyze_c6_winding_temporal_compatibility(include_runtime=value)


def _cli(tmp_path, monkeypatch):
    paths = tuple(tmp_path / name for name in campaign.INPUT_NAMES)
    for path, source in zip(paths, INPUTS, strict=True):
        if not source.is_file():
            pytest.skip("the retained B40 through B26 evidence chain is unavailable")
        path.write_bytes(source.read_bytes())
    output = tmp_path / "temporal.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temporal",
            "--history-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--skip-runtime",
        ],
    )
    return paths, output


def test_cli_binds_eight_historical_hashes_and_respects_static_execution(
    tmp_path, monkeypatch, report
):
    paths, output = _cli(tmp_path, monkeypatch)
    original = tuple(path.read_bytes() for path in paths)
    calls = []

    def analysis(*args, include_runtime):
        calls.append((len(args), include_runtime))
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_temporal_compatibility", analysis)
    provenance = ("a" * 40, True, "sha256:" + "b" * 64)
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *args: provenance
    )
    campaign.main()
    result = json.loads(output.read_bytes())
    assert calls == [(8, False)]
    assert result["manifest"]["claim_id"] == "O3.a-C6-temporal-cell-compatibility"
    assert result["manifest"]["git_sha"] == provenance[0]
    assert len(result["input_evidence"]) == 8
    for evidence, data in zip(result["input_evidence"], original, strict=True):
        assert evidence["sha256"] == hashlib.sha256(data).hexdigest()
        assert evidence["producer_manifest"] != result["manifest"]
    assert tuple(path.read_bytes() for path in paths) == original


@pytest.mark.parametrize("changed_index", range(1, 8))
def test_cli_checks_all_ancestral_byte_links_before_analysis(
    tmp_path, monkeypatch, changed_index
):
    paths, output = _cli(tmp_path, monkeypatch)
    paths[changed_index].write_bytes(paths[changed_index].read_bytes() + b"\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("historical byte corruption must fail before analysis")

    monkeypatch.setattr(
        campaign, "analyze_c6_winding_temporal_compatibility", forbidden
    )
    with pytest.raises(ValueError, match="lineage hashes"):
        campaign.main()
    assert not output.exists()


@pytest.mark.parametrize("change", ("source", "input"))
def test_cli_aborts_on_source_or_input_changes_during_analysis(
    tmp_path, monkeypatch, report, change
):
    paths, output = _cli(tmp_path, monkeypatch)
    original = ("a" * 40, True, "sha256:" + "b" * 64)
    later = ("a" * 40, True, "sha256:" + "c" * 64) if change == "source" else original
    values = iter((original, later))
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *args: next(values)
    )

    def analysis(*args, **kwargs):
        if change == "input":
            paths[0].write_bytes(paths[0].read_bytes() + b"\n")
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_temporal_compatibility", analysis)
    with pytest.raises(RuntimeError, match="changed during"):
        campaign.main()
    assert not output.exists()


def test_cli_cannot_overwrite_the_saved_parent(tmp_path, monkeypatch):
    paths, _ = _cli(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temporal",
            "--history-dir",
            str(tmp_path),
            "--output",
            str(paths[0]),
            "--skip-runtime",
        ],
    )
    original = paths[0].read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert paths[0].read_bytes() == original

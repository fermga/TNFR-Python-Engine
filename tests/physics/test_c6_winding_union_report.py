"""Whole-target, provenance and resource controls for exact return unions."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks import c6_winding_invariant_region as campaign


@pytest.fixture(scope="module")
def union_inputs():
    root = Path(__file__).resolve().parents[2]
    names = (
        "artifacts/research/c6_winding_relay_handoff.json",
        "artifacts/research/c6_winding_relay_budget.json",
        "artifacts/research/c6_winding_temporal_compatibility.json",
        "artifacts/research/c6_winding_forward_envelope.json",
        "artifacts/research/c6_winding_region_exclusions.json",
        "artifacts/research/c6_winding_excursion_exclusion.json",
        "benchmarks/c6_winding_mode_excursion_candidates.json",
        "artifacts/research/c6_winding_mode_excursions.json",
        "artifacts/research/c6_winding_return_regions.json",
    )
    if any(not (root / name).exists() for name in names):
        pytest.skip("the retained C6 research evidence is not available")
    return tuple((root / name).read_bytes() for name in names)


def arguments(raws):
    return dict(
        zip(
            (
                "parent_bytes",
                "relay_bytes",
                "historical_bytes",
                "envelope_bytes",
                "region_bytes",
                "excursion_bytes",
                "candidates_bytes",
                "mode_bytes",
                "return_bytes",
            ),
            raws,
            strict=True,
        )
    )


@pytest.fixture(scope="module")
def previous_returns(union_inputs):
    kwargs = arguments(union_inputs)
    kwargs.pop("return_bytes")
    return campaign.analyze_c6_winding_return_regions(
        json.loads(union_inputs[0]), **kwargs
    )


@pytest.fixture(scope="module")
def union_report(union_inputs, previous_returns):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            campaign,
            "analyze_c6_winding_return_regions",
            lambda *_a, **_k: deepcopy(previous_returns),
        )
        return campaign.analyze_c6_winding_return_unions(
            json.loads(union_inputs[0]), **arguments(union_inputs)
        )


@pytest.mark.slow
def test_exact_union_closes_mask_zero_without_shrinking_original_target(
    union_report, previous_returns
):
    result = union_report
    assert result["excluded_first_exit_slabs"] == 39
    assert result["additional_first_exit_slabs_excluded"] == 1
    assert len(result["B55_remaining_first_exit_facets"]) == 17
    assert result["B55_additional_first_exit_facets"] == (
        dict(mask=0, node=0, direction="lower"),
    )
    original = {
        tuple(item["target"].values()): item["query"]["target_regions"]
        for item in previous_returns["B54_region_queries"]
    }
    for item in result["B55_union_queries"]:
        assert (
            item["query"]["target_regions"] == original[tuple(item["target"].values())]
        )
    positives = [
        r["query"]
        for r in result["B55_union_queries"]
        if r["query"]["origin_path_within_domain_excluded"]
    ]
    assert len(positives) == 1
    positive = positives[0]
    assert (
        positive["status"] == "empty_complete_layer"
        and positive["completed_depth"] == 4
    )
    assert all(not item["origin_present"] for item in positive["iterations"])
    assert (
        result["new_conditional_trajectory_steps"]
        == result["new_live_graph_steps"]
        == 0
    )
    assert result["primary_proof_origin_remains_B47"]
    assert result["B55_reconstruction"]["prior_excursion_prefix_steps_replayed"] == 56
    assert (
        not result["indefinite_trapping_certified"]
        and not result["future_runtime_certified"]
    )
    assert not result["B55_return_envelope"]["clipped_forward_inclusion_certified"]


@pytest.mark.parametrize("index", range(8))
def test_union_audit_rejects_any_changed_historical_input(union_inputs, index):
    changed = list(union_inputs)
    changed[index] += b" "
    with pytest.raises(ValueError, match="lineage"):
        campaign.analyze_c6_winding_return_unions(
            json.loads(changed[0]), **arguments(changed)
        )


@pytest.mark.slow
@pytest.mark.parametrize("mutation", ("result", "origin", "guard", "target", "claim"))
def test_union_audit_rebuilds_stored_return_claims(
    monkeypatch, union_inputs, previous_returns, mutation
):
    old = json.loads(union_inputs[-1])
    if mutation == "result":
        old["excluded_first_exit_slabs"] += 1
    elif mutation == "origin":
        old["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "guard":
        old["B54_return_envelope"]["return_relation"][0]["source_guard"][0][6] += 1
    elif mutation == "target":
        old["B54_region_queries"][0]["query"]["target_regions"][0]["bounds"][0][6] += 1
    else:
        old["manifest"]["claim_id"] = "invented"
    changed = (*union_inputs[:-1], json.dumps(old).encode())
    monkeypatch.setattr(
        campaign,
        "analyze_c6_winding_return_regions",
        lambda *_a, **_k: deepcopy(previous_returns),
    )
    with pytest.raises(ValueError, match="lineage|reconstruction"):
        campaign.analyze_c6_winding_return_unions(
            json.loads(changed[0]), **arguments(changed)
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "limit",
    (
        "max_intersections",
        "query_max_intersections",
        "query_max_subsumptions",
        "query_max_zones",
    ),
)
def test_union_resource_guards_do_not_promote_exclusion(
    monkeypatch, union_inputs, previous_returns, limit
):
    monkeypatch.setattr(
        campaign,
        "analyze_c6_winding_return_regions",
        lambda *_a, **_k: deepcopy(previous_returns),
    )
    result = campaign.analyze_c6_winding_return_unions(
        json.loads(union_inputs[0]),
        **arguments(union_inputs),
        **{limit: 1},
    )
    assert result["excluded_first_exit_slabs"] == 38
    assert result["additional_first_exit_slabs_excluded"] == 0
    assert len(result["B55_remaining_first_exit_facets"]) == 18
    assert not result["indefinite_trapping_certified"]


def cli_args(monkeypatch, paths, output):
    flags = (
        "--input",
        "--relay-input",
        "--historical-input",
        "--envelope-input",
        "--region-input",
        "--excursion-input",
        "--mode-candidates-input",
        "--mode-input",
        "--return-input",
    )
    argv = [
        "c6_winding_invariant_region.py",
        "--method",
        "unions",
        "--output",
        str(output),
    ]
    for flag, path in zip(flags, paths, strict=True):
        argv.extend((flag, str(path)))
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.slow
def test_union_cli_binds_nine_inputs(tmp_path, monkeypatch, union_inputs, union_report):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(9))
    for path, raw in zip(paths, union_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        campaign,
        "analyze_c6_winding_return_unions",
        lambda *_a, **_k: deepcopy(union_report),
    )
    monkeypatch.setattr(
        campaign,
        "current_git_source_provenance",
        lambda *_: ("a" * 40, True, "sha256:" + "b" * 64),
    )
    cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-return-unions"
    assert [r["sha256"] for r in result["input_evidence"]] == [
        hashlib.sha256(raw).hexdigest() for raw in union_inputs
    ]
    assert (
        result["excluded_first_exit_slabs"] == 39
        and not result["indefinite_trapping_certified"]
    )


@pytest.mark.parametrize("index", range(9))
def test_union_cli_cannot_overwrite_any_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(9))
    for path in paths:
        path.write_text("{}")
    cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.slow
@pytest.mark.parametrize("which", (*range(9), "source"))
def test_union_cli_rejects_changed_input_or_source(
    tmp_path, monkeypatch, union_inputs, union_report, which
):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(9))
    for path, raw in zip(paths, union_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    provenance = iter(
        (
            ("a" * 40, True, "sha256:" + "b" * 64),
            ("a" * 40, True, "sha256:" + ("c" if which == "source" else "b") * 64),
        )
    )
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *_: next(provenance)
    )

    def analyze(*_a, **_k):
        if which != "source":
            paths[which].write_text("{}")
        return deepcopy(union_report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_return_unions", analyze)
    cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()

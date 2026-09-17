"""Whole-target and provenance boundaries of the safe-memory campaign."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks import c6_winding_invariant_region as campaign


@pytest.fixture(scope="module")
def inputs():
    root = Path(__file__).resolve().parents[2]
    names = (
        "artifacts/research/c6_winding_relay_handoff.json", "artifacts/research/c6_winding_relay_budget.json",
        "artifacts/research/c6_winding_temporal_compatibility.json", "artifacts/research/c6_winding_forward_envelope.json",
        "artifacts/research/c6_winding_region_exclusions.json", "artifacts/research/c6_winding_excursion_exclusion.json",
        "benchmarks/c6_winding_mode_excursion_candidates.json", "artifacts/research/c6_winding_mode_excursions.json",
        "artifacts/research/c6_winding_return_regions.json", "artifacts/research/c6_winding_return_unions.json",
        "benchmarks/c6_winding_return_count_candidates.json", "artifacts/research/c6_winding_return_histories.json",
        "artifacts/research/c6_winding_return_memory.json", "artifacts/research/c6_winding_memory_cuts.json",
    )
    if any(not (root / name).is_file() for name in names):
        pytest.skip("the retained C6 research evidence is unavailable")
    return tuple((root / name).read_bytes() for name in names)


def arguments(raws):
    return dict(zip(("parent_bytes", "relay_bytes", "historical_bytes", "envelope_bytes", "region_bytes",
                     "excursion_bytes", "candidates_bytes", "mode_bytes", "return_bytes", "union_bytes",
                     "history_candidates_bytes", "history_bytes", "memory_bytes", "cuts_bytes"), raws, strict=True))


@pytest.fixture(scope="module")
def previous(inputs):
    # The production campaign replays all B58 ancestors. Isolate B59 admission
    # and recompute its partition from the unchanged B57 source here.
    retained = json.loads(inputs[13])
    for key in ("manifest", "source_scope", "input_evidence"):
        retained.pop(key)
    return retained


def patch_previous(monkeypatch, previous):
    monkeypatch.setattr(campaign, "analyze_c6_winding_memory_cuts", lambda *_a, **_k: deepcopy(previous))


@pytest.fixture(scope="module")
def report(inputs, previous):
    with pytest.MonkeyPatch.context() as patch:
        patch_previous(patch, previous)
        return campaign.analyze_c6_winding_safe_partition(
            json.loads(inputs[0]), **arguments(inputs), max_partition_work=120_000,
            query_max_intersections=1,
        )


def test_safe_partition_preserves_canonical_origin_and_rechecks_the_proposed_cut(inputs, report):
    parent, baseline = json.loads(inputs[13]), json.loads(inputs[12])
    assert report["source"] == parent["source"]
    assert campaign._payload(report["B59_return_envelope"]["state"]) == baseline["B57_return_envelope"]["state"]
    assert campaign._payload(report["B59_return_envelope"]["domain_zones"]) == baseline["B57_return_envelope"]["domain_zones"]
    assert report["B59_exclusion_proposals"] == tuple(baseline["B57_additional_first_exit_facets"])
    assert report["B59_exclusion_proposals"] == (dict(mask=30, node=3, direction="upper"),)
    assert report["B59_reconstruction"]["prior_B58_completely_rebuilt"]
    assert report["B59_reconstruction"]["prior_first_exit_slabs_reverified"] == 40
    assert report["B59_reconstruction"]["prior_cuts_sha256"] == hashlib.sha256(inputs[13]).hexdigest()
    partition = report["B59_partition"]
    assert partition["exclusion_query_relation_complete"]
    assert len(partition["excluded_target_queries"]) == 1
    assert partition["excluded_target_queries"][0]["status"] in ("empty_complete_layer", "stationary_complete_layer")
    assert report["primary_proof_origin_remains_B47"] and not report["contract"]["physical_dynamics_changed"]
    assert report["new_conditional_trajectory_steps"] == report["new_live_graph_steps"] == 0
    for key in ("indefinite_trapping_certified", "whole_band_exit_certified", "future_runtime_certified",
                "asymptotic_convergence_certified", "all_first_exit_slabs_excluded"):
        assert report[key] is False
    json.dumps(campaign._payload(report), allow_nan=False)


def test_safe_partition_queries_keep_all_full_original_residual_targets(inputs, report):
    parent, baseline = json.loads(inputs[13]), json.loads(inputs[12])
    labels = parent["B58_remaining_first_exit_facets"]
    queries = report["B59_partition_queries"]
    assert len(queries) == len(labels) == 16
    assert [item["target"] for item in queries] == labels
    for item in queries:
        expected = next(query["query"]["target_regions"] for query in baseline["B57_memory_queries"]
                        if query["target"] == item["target"])
        assert campaign._payload(item["query"]["target_regions"]) == expected
        assert not item["query"]["actual_origin_reachability_certified"]
    assert report["B59_original_cube_geometry"] == parent["B58_original_cube_geometry"]
    assert report["B59_outgoing_facets"] == parent["B58_outgoing_facets"]


def test_incomplete_residual_query_graph_cannot_add_any_exclusion(report):
    assert not report["B59_query_relation_complete"]
    assert report["B59_query_relation_intersections"] <= 1
    assert report["excluded_first_exit_slabs"] == 40 and report["additional_first_exit_slabs_excluded"] == 0
    assert len(report["B59_remaining_first_exit_facets"]) == 16
    assert all(not item["query"]["initialization_complete"] for item in report["B59_partition_queries"])
    assert all(not item["query"]["origin_path_within_domain_excluded"] for item in report["B59_partition_queries"])


def test_unverified_cut_proposal_has_explicit_no_partition_fallback(monkeypatch, inputs, previous):
    patch_previous(monkeypatch, previous)
    result = campaign.analyze_c6_winding_safe_partition(
        json.loads(inputs[0]), **arguments(inputs), max_memory_work=1,
        exclusion_query_max_intersections=1, query_max_intersections=1,
    )
    partition = result["B59_partition"]
    assert partition["status"] == "exclusion_not_verified"
    assert not partition["initialization_complete"] and not partition["relation_complete"]
    assert not partition["seed_zones"] and not partition["partition_arcs"]
    assert not result["B59_query_relation_complete"]
    assert result["excluded_first_exit_slabs"] == 40 and result["additional_first_exit_slabs_excluded"] == 0
    assert all(not item["query"]["origin_path_within_domain_excluded"] for item in result["B59_partition_queries"])


@pytest.mark.parametrize("index", range(13))
def test_safe_partition_rejects_every_altered_ancestor(inputs, index):
    changed = list(inputs)
    changed[index] += b" "
    with pytest.raises(ValueError, match="lineage"):
        campaign.analyze_c6_winding_safe_partition(json.loads(changed[0]), **arguments(changed))


@pytest.mark.parametrize("mutation", ("claim", "scope", "origin", "positive_count", "guard", "target", "equivalence"))
def test_safe_partition_reconstructs_untrusted_parent_claims(monkeypatch, inputs, previous, mutation):
    parent = json.loads(inputs[13])
    if mutation == "claim":
        parent["manifest"]["claim_id"] = "invented"
    elif mutation == "scope":
        parent["source_scope"] = []
    elif mutation == "origin":
        parent["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "positive_count":
        parent["excluded_first_exit_slabs"] += 1
    elif mutation == "guard":
        parent["B58_return_envelope"]["return_relation"][0]["source_guard"][0][6] += 1
    elif mutation == "target":
        parent["B58_original_target_groups"][0][0]["bounds"][0][6] += 1
    else:
        parent["B58_query_equivalence"]["all_target_operators_and_initial_conditions_identical"] = False
    changed = (*inputs[:13], json.dumps(parent).encode())
    patch_previous(monkeypatch, previous)
    with pytest.raises(ValueError, match="lineage|reconstruction"):
        campaign.analyze_c6_winding_safe_partition(json.loads(changed[0]), **arguments(changed))


def cli_args(monkeypatch, paths, output):
    flags = ("--input", "--relay-input", "--historical-input", "--envelope-input", "--region-input",
             "--excursion-input", "--mode-candidates-input", "--mode-input", "--return-input",
             "--union-input", "--history-candidates-input", "--history-input", "--memory-input", "--cuts-input")
    argv = ["c6_winding_invariant_region.py", "--method", "safe-partition", "--output", str(output)]
    for flag, path in zip(flags, paths, strict=True):
        argv.extend((flag, str(path)))
    monkeypatch.setattr(sys, "argv", argv)


def test_safe_partition_cli_binds_all_fourteen_inputs(tmp_path, monkeypatch, inputs, report):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(14))
    for path, raw in zip(paths, inputs, strict=True):
        path.write_bytes(raw)
    output, received = tmp_path / "result.json", []

    def analyze(parent, **kwargs):
        received.append((parent, kwargs))
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_safe_partition", analyze)
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert received[0][0] == json.loads(inputs[0])
    assert all(received[0][1][key] == raw for key, raw in arguments(inputs).items())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-safe-partition"
    assert [item["sha256"] for item in result["input_evidence"]] == [hashlib.sha256(raw).hexdigest() for raw in inputs]
    assert not result["indefinite_trapping_certified"]
    assert [path.read_bytes() for path in paths] == list(inputs)


@pytest.mark.parametrize("index", range(14))
def test_safe_partition_cli_cannot_overwrite_any_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(14))
    for path in paths:
        path.write_text("{}")
    cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.parametrize("which", (*range(14), "source"))
def test_safe_partition_cli_rejects_midflight_input_changes(tmp_path, monkeypatch, inputs, report, which):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(14))
    for path, raw in zip(paths, inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    provenance = iter((("a" * 40, True, "sha256:" + "b" * 64),
                       ("a" * 40, True, "sha256:" + ("c" if which == "source" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(provenance))

    def analyze(*_a, **_k):
        if which != "source":
            paths[which].write_text("{}")
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_safe_partition", analyze)
    cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()

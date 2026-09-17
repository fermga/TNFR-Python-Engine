"""Depth-specific provenance and scope of exact safe predecessor campaigns."""
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
    retained = json.loads(inputs[13])
    for key in ("manifest", "source_scope", "input_evidence"):
        retained.pop(key)
    return retained


def patch_previous(monkeypatch, previous):
    monkeypatch.setattr(campaign, "analyze_c6_winding_memory_cuts", lambda *_a, **_k: deepcopy(previous))


@pytest.fixture(scope="module")
def report(inputs, previous):
    # Rebuild B60's canonical owner, while the expensive ancestor chain is
    # independently covered by existing admission tests and production replay.
    with pytest.MonkeyPatch.context() as patch:
        patch_previous(patch, previous)
        return campaign.analyze_c6_winding_safe_partition(
            json.loads(inputs[0]), **arguments(inputs), excluded_predecessor_depth=1,
            max_partition_work=120_000, query_max_intersections=1,
        )


def test_depth_one_report_retains_original_source_and_only_verified_proposals(inputs, report):
    baseline, ancestor = json.loads(inputs[12]), json.loads(inputs[13])
    assert report["contract"]["excluded_predecessor_depth"] == 1
    assert not any(key.startswith("B59_") for key in report)
    assert report["source"] == ancestor["source"]
    assert campaign._payload(report["B60_return_envelope"]["state"]) == baseline["B57_return_envelope"]["state"]
    assert campaign._payload(report["B60_return_envelope"]["domain_zones"]) == baseline["B57_return_envelope"]["domain_zones"]
    assert report["B60_exclusion_proposals"] == tuple(baseline["B57_additional_first_exit_facets"])
    assert report["B60_reconstruction"]["prior_cuts_sha256"] == hashlib.sha256(inputs[13]).hexdigest()
    partition = report["B60_partition"]
    assert partition["excluded_predecessor_depth"] == 1 and partition["predecessor_complete"]
    assert len(partition["predecessor_layers"]) == 1
    assert partition["predecessor_layers"][0]["depth"] == 1
    assert partition["exclusion_query_relation_complete"]
    assert partition["excluded_target_queries"][0]["status"] in ("empty_complete_layer", "stationary_complete_layer")
    assert report["primary_proof_origin_remains_B47"] and not report["contract"]["physical_dynamics_changed"]
    assert report["new_conditional_trajectory_steps"] == report["new_live_graph_steps"] == 0
    for key in ("indefinite_trapping_certified", "whole_band_exit_certified", "future_runtime_certified",
                "asymptotic_convergence_certified", "all_first_exit_slabs_excluded"):
        assert report[key] is False
    json.dumps(campaign._payload(report), allow_nan=False)


def test_depth_one_keeps_all_sixteen_original_targets_without_promoting_incomplete_queries(inputs, report):
    ancestor, baseline = json.loads(inputs[13]), json.loads(inputs[12])
    labels = ancestor["B58_remaining_first_exit_facets"]
    queries = report["B60_partition_queries"]
    assert len(queries) == len(labels) == 16
    assert [item["target"] for item in queries] == labels
    for item in queries:
        original = next(row["query"]["target_regions"] for row in baseline["B57_memory_queries"] if row["target"] == item["target"])
        assert campaign._payload(item["query"]["target_regions"]) == original
        assert not item["query"]["initialization_complete"]
        assert not item["query"]["origin_path_within_domain_excluded"]
        assert not item["query"]["actual_origin_reachability_certified"]
    assert not report["B60_query_relation_complete"]
    assert report["excluded_first_exit_slabs"] == 40 and report["additional_first_exit_slabs_excluded"] == 0
    assert report["B60_original_cube_geometry"] == ancestor["B58_original_cube_geometry"]
    assert report["B60_outgoing_facets"] == ancestor["B58_outgoing_facets"]


@pytest.mark.parametrize("index", range(13))
def test_depth_one_rejects_each_cross_lineage_ancestor(inputs, index):
    changed = list(inputs)
    changed[index] += b" "
    with pytest.raises(ValueError, match="lineage"):
        campaign.analyze_c6_winding_safe_partition(
            json.loads(changed[0]), **arguments(changed), excluded_predecessor_depth=1,
        )


@pytest.mark.parametrize("mutation", ("origin", "target", "claim"))
def test_depth_one_reconstructs_parent_evidence_instead_of_trusting_it(monkeypatch, inputs, previous, mutation):
    parent = json.loads(inputs[13])
    if mutation == "origin":
        parent["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "target":
        parent["B58_original_target_groups"][0][0]["bounds"][0][6] += 1
    else:
        parent["manifest"]["claim_id"] = "invented"
    changed = (*inputs[:13], json.dumps(parent).encode())
    patch_previous(monkeypatch, previous)
    with pytest.raises(ValueError, match="lineage|reconstruction"):
        campaign.analyze_c6_winding_safe_partition(json.loads(changed[0]), **arguments(changed), excluded_predecessor_depth=1)


@pytest.mark.parametrize("depth", (-1, True, 1.5, "1", None))
def test_invalid_campaign_depth_is_rejected_before_reading_or_replaying_ancestors(depth):
    with pytest.raises((TypeError, ValueError), match="depth"):
        campaign.analyze_c6_winding_safe_partition({}, **arguments((b"not json",) * 14), excluded_predecessor_depth=depth)


def cli_args(monkeypatch, paths, output=None, depth=None, method="safe-partition"):
    flags = ("--input", "--relay-input", "--historical-input", "--envelope-input", "--region-input",
             "--excursion-input", "--mode-candidates-input", "--mode-input", "--return-input",
             "--union-input", "--history-candidates-input", "--history-input", "--memory-input", "--cuts-input")
    argv = ["c6_winding_invariant_region.py", "--method", method]
    if output is not None:
        argv.extend(("--output", str(output)))
    if depth is not None:
        argv.extend(("--excluded-predecessor-depth", str(depth)))
    for flag, path in zip(flags, paths, strict=True):
        argv.extend((flag, str(path)))
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("depth,custom", ((None, False), (1, False), (1, True)))
def test_cli_default_b59_and_depth_one_b60_have_separate_paths_claims_and_full_lineage(
        tmp_path, monkeypatch, inputs, report, depth, custom):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(14))
    for path, raw in zip(paths, inputs, strict=True):
        path.write_bytes(raw)
    expected = (tmp_path / "custom.json" if custom else tmp_path / "artifacts/research" /
                ("c6_winding_safe_predecessors.json" if depth else "c6_winding_safe_partition.json"))
    received = []

    def analyze(parent, **kwargs):
        received.append((parent, kwargs))
        result = deepcopy(report)
        if kwargs.get("excluded_predecessor_depth", 0) == 0:
            result = {key.replace("B60_", "B59_", 1) if key.startswith("B60_") else key: value for key, value in result.items()}
            result["contract"].pop("excluded_predecessor_depth")
        return result

    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "analyze_c6_winding_safe_partition", analyze)
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    cli_args(monkeypatch, paths, expected if custom else None, depth)
    campaign.main()
    result = json.loads(expected.read_bytes())
    assert received[0][1].get("excluded_predecessor_depth", 0) == (depth or 0)
    assert all(received[0][1][key] == raw for key, raw in arguments(inputs).items())
    expected_claim = "O3.a-C6-carried-safe-predecessors" if depth else "O3.a-C6-carried-safe-partition"
    assert result["manifest"]["claim_id"] == expected_claim
    assert [item["sha256"] for item in result["input_evidence"]] == [hashlib.sha256(raw).hexdigest() for raw in inputs]
    assert [path.read_bytes() for path in paths] == list(inputs)
    assert not result["indefinite_trapping_certified"]


@pytest.mark.parametrize("method", ("backward", "modes", "memory-cuts"))
def test_cli_positive_depth_rejects_unrelated_methods_before_input_reads(tmp_path, monkeypatch, method):
    paths = tuple(tmp_path / f"missing{i}.json" for i in range(14))
    cli_args(monkeypatch, paths, depth=1, method=method)
    with pytest.raises(ValueError, match="safe-partition"):
        campaign.main()


def test_cli_negative_depth_rejected_before_input_reads(tmp_path, monkeypatch):
    paths = tuple(tmp_path / f"missing{i}.json" for i in range(14))
    cli_args(monkeypatch, paths, depth=-1)
    with pytest.raises(ValueError, match="depth"):
        campaign.main()


@pytest.mark.parametrize("index", (0, 12, 13))
def test_depth_one_cli_cannot_overwrite_primary_or_latest_evidence(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(14))
    for path in paths:
        path.write_text("{}")
    cli_args(monkeypatch, paths, paths[index], depth=1)
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()


@pytest.mark.parametrize("which", (0, 13, "source"))
def test_depth_one_cli_rejects_midflight_primary_latest_or_source_changes(tmp_path, monkeypatch, inputs, report, which):
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
    cli_args(monkeypatch, paths, output, depth=1)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()

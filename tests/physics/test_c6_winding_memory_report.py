"""Whole-target and retained-input scope of the last-return memory campaign."""
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
    )
    if any(not (root / name).exists() for name in names):
        pytest.skip("the retained C6 research evidence is unavailable")
    return tuple((root / name).read_bytes() for name in names)


def arguments(raws):
    return dict(zip(("parent_bytes", "relay_bytes", "historical_bytes", "envelope_bytes", "region_bytes",
                     "excursion_bytes", "candidates_bytes", "mode_bytes", "return_bytes", "union_bytes",
                     "history_candidates_bytes", "history_bytes"), raws, strict=True))


@pytest.fixture(scope="module")
def previous(inputs):
    # The complete expensive B55/B56 reconstruction is checked by its own tests
    # and the production campaign. These tests isolate B57's boundary checks.
    retained = json.loads(inputs[11])
    for key in ("manifest", "source_scope", "input_evidence"):
        retained.pop(key)
    return retained


def patch_previous(monkeypatch, previous):
    monkeypatch.setattr(campaign, "analyze_c6_winding_return_histories", lambda *_a, **_k: deepcopy(previous))


@pytest.fixture(scope="module")
def report(inputs, previous):
    with pytest.MonkeyPatch.context() as patch:
        patch_previous(patch, previous)
        return campaign.analyze_c6_winding_return_memory(
            json.loads(inputs[0]), **arguments(inputs), max_memory_work=120_000,
            query_max_intersections=30_000,
        )


def test_memory_report_keeps_original_origin_and_conditional_scope(inputs, report):
    parent = json.loads(inputs[11])
    assert report["source"] == parent["source"]
    assert campaign._payload(report["B57_return_envelope"]["state"]) == parent["B56_return_envelope"]["state"]
    assert report["primary_proof_origin_remains_B47"]
    assert report["new_conditional_trajectory_steps"] == report["new_live_graph_steps"] == 0
    assert not report["contract"]["physical_dynamics_changed"]
    assert not report["indefinite_trapping_certified"] and not report["whole_band_exit_certified"]
    assert not report["future_runtime_certified"] and not report["asymptotic_convergence_certified"]
    assert report["B57_reconstruction"]["prior_first_exit_slabs_reverified"] == 39
    assert report["excluded_first_exit_slabs"] + len(report["B57_remaining_first_exit_facets"]) == 56


def test_memory_report_preserves_every_whole_original_target(inputs, report):
    parent = json.loads(inputs[11])
    b55 = json.loads(inputs[9])
    observations = report["B57_target_observations"]
    assert len(observations) == len(parent["B56_remaining_first_exit_facets"]) == 17
    assert [item["target"] for item in observations] == parent["B56_remaining_first_exit_facets"]
    for item in observations:
        original = next(q["query"]["target_regions"][0]["bounds"]
                        for q in b55["B55_union_queries"] if q["target"] == item["target"])
        assert campaign._payload(item["original_target_bounds"]) == original
        assert item["observation_complete"]
        assert not item["actual_origin_reachability_certified"]
    # Mask30 is a transient cell: observing only completed return endpoints
    # would incorrectly lose its entire possible terminal population.
    transient = next(item for item in observations if item["target"] == dict(mask=30, node=3, direction="upper"))
    assert not transient["direct_memory_indices"]
    assert transient["terminal_transient_memory_edges"]
    env = report["B57_return_envelope"]
    target_row = env["epi_states"][30]
    for memory_index, edge_index in transient["terminal_transient_memory_edges"]:
        edge = env["intermediate_transitions"][edge_index]
        source_row = env["state"]["epi"] if memory_index == -1 else env["return_relation"][memory_index]["target_epi"]
        assert edge["source_epi"] == source_row and edge["target_epi"] == target_row


def test_memory_queries_keep_full_targets_and_complete_layers(inputs, report):
    parent = json.loads(inputs[11])
    original_queries = json.loads(inputs[9])["B55_union_queries"]
    queries = report["B57_memory_queries"]
    assert [item["target"] for item in queries] == parent["B56_remaining_first_exit_facets"]
    assert report["B57_query_relation_complete"]
    assert report["B57_query_relation_intersections"] <= 30_000
    count = len(report["B57_return_envelope"]["return_relation"]) + 1
    for item in queries:
        query = item["query"]
        original = next(q["query"]["target_regions"] for q in original_queries if q["target"] == item["target"])
        assert campaign._payload(query["target_regions"]) == original
        assert query["initialization_complete"]
        assert len(query["initial_endpoint_zones"]) == len(query["retained_endpoint_zones"]) == count
        assert query["intersections"] <= 30_000
        assert not query["actual_origin_reachability_certified"]
        assert [step["depth"] for step in query["iterations"]] == list(range(len(query["iterations"])))
        if query["status"] == "resource_limit":
            assert not query["origin_path_within_domain_excluded"]
        if query["origin_path_within_domain_excluded"]:
            assert item["target"] in report["B57_additional_first_exit_facets"]


@pytest.mark.parametrize("index", range(11))
def test_memory_rejects_each_altered_ancestor(inputs, index):
    changed = list(inputs)
    changed[index] += b" "
    with pytest.raises(ValueError, match="lineage"):
        campaign.analyze_c6_winding_return_memory(json.loads(changed[0]), **arguments(changed))


@pytest.mark.parametrize("mutation", ("claim", "result", "origin", "guard", "target", "word", "count"))
def test_memory_reconstructs_untrusted_parent_claims(monkeypatch, inputs, previous, mutation):
    parent = json.loads(inputs[11])
    if mutation == "claim":
        parent["manifest"]["claim_id"] = "invented"
    elif mutation == "result":
        parent["excluded_first_exit_slabs"] += 1
    elif mutation == "origin":
        parent["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "guard":
        parent["B56_return_envelope"]["return_relation"][0]["source_guard"][0][6] += 1
    elif mutation == "target":
        parent["B56_remaining_first_exit_facets"].pop()
    elif mutation == "word":
        parent["B56_word_budgets"][0]["proof"]["maximum_repetitions"] += 1
    else:
        parent["B56_count_relaxation"]["coordinate_divisors"][0] += 1
    changed = (*inputs[:11], json.dumps(parent).encode())
    patch_previous(monkeypatch, previous)
    with pytest.raises(ValueError, match="lineage|reconstruction"):
        campaign.analyze_c6_winding_return_memory(json.loads(changed[0]), **arguments(changed))


def test_memory_incomplete_initialization_cannot_exclude(monkeypatch, inputs, previous):
    patch_previous(monkeypatch, previous)
    result = campaign.analyze_c6_winding_return_memory(
        json.loads(inputs[0]), **arguments(inputs), max_memory_work=1,
        query_max_intersections=1,
    )
    assert not result["B57_memory_envelope"]["initialization_complete"]
    assert result["excluded_first_exit_slabs"] == 39
    assert result["additional_first_exit_slabs_excluded"] == 0
    assert len(result["B57_remaining_first_exit_facets"]) == 17
    assert all(not item["observation_complete"] for item in result["B57_target_observations"])
    assert all(not item["origin_path_before_original_cube_exit_excluded"] for item in result["B57_target_observations"])
    assert not result["B57_query_relation_complete"]
    assert all(not item["query"]["initialization_complete"] for item in result["B57_memory_queries"])
    assert all(not item["query"]["origin_path_within_domain_excluded"] for item in result["B57_memory_queries"])


def cli_args(monkeypatch, paths, output):
    flags = ("--input", "--relay-input", "--historical-input", "--envelope-input", "--region-input",
             "--excursion-input", "--mode-candidates-input", "--mode-input", "--return-input",
             "--union-input", "--history-candidates-input", "--history-input")
    argv = ["c6_winding_invariant_region.py", "--method", "memory", "--output", str(output)]
    for flag, path in zip(flags, paths, strict=True):
        argv.extend((flag, str(path)))
    monkeypatch.setattr(sys, "argv", argv)


def test_memory_cli_binds_all_twelve_inputs_in_order(tmp_path, monkeypatch, inputs, report):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(12))
    for path, raw in zip(paths, inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    received = []

    def analyze(parent, **kwargs):
        received.append((parent, kwargs))
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_return_memory", analyze)
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert received[0][0] == json.loads(inputs[0])
    for key, raw in arguments(inputs).items():
        assert received[0][1][key] == raw
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-return-memory"
    assert [item["sha256"] for item in result["input_evidence"]] == [hashlib.sha256(raw).hexdigest() for raw in inputs]
    assert not result["indefinite_trapping_certified"]
    assert [path.read_bytes() for path in paths] == list(inputs)


@pytest.mark.parametrize("index", range(12))
def test_memory_cli_cannot_overwrite_any_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(12))
    for path in paths:
        path.write_text("{}")
    cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()


@pytest.mark.parametrize("which", (*range(12), "source"))
def test_memory_cli_rejects_midflight_input_changes(tmp_path, monkeypatch, inputs, report, which):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(12))
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

    monkeypatch.setattr(campaign, "analyze_c6_winding_return_memory", analyze)
    cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()

"""Provenance and chronological scope of the C6 return-history campaign."""
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
        "benchmarks/c6_winding_return_count_candidates.json",
    )
    if any(not (root / name).exists() for name in names):
        pytest.skip("the retained C6 research evidence is unavailable")
    return tuple((root / name).read_bytes() for name in names)


def arguments(raws):
    return dict(zip(("parent_bytes", "relay_bytes", "historical_bytes", "envelope_bytes", "region_bytes",
                     "excursion_bytes", "candidates_bytes", "mode_bytes", "return_bytes", "union_bytes",
                     "history_candidates_bytes"), raws, strict=True))


@pytest.fixture(scope="module")
def previous(inputs):
    kwargs = arguments(inputs)
    kwargs.pop("union_bytes")
    kwargs.pop("history_candidates_bytes")
    return campaign.analyze_c6_winding_return_unions(json.loads(inputs[0]), **kwargs)


@pytest.fixture(scope="module")
def report(inputs, previous):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "analyze_c6_winding_return_unions", lambda *_a, **_k: deepcopy(previous))
        return campaign.analyze_c6_winding_return_histories(json.loads(inputs[0]), **arguments(inputs))


def test_history_report_distinguishes_counts_from_jointly_admissible_words(report):
    assert report["excluded_first_exit_slabs"] == 39
    assert report["additional_first_exit_slabs_excluded"] == 0
    assert len(report["B56_remaining_first_exit_facets"]) == 17
    counts = report["B56_count_relaxation"]
    assert counts["coordinate_divisors"] == (1, 1, 4, 8, 8, 8)
    assert counts["nonnegative_counts_cover_coordinate_cosets"]
    assert counts["abstract_mode_walk_exists_for_each_coordinate_coset_point"]
    assert not counts["joint_guard_satisfaction_certified"] and not counts["actual_origin_reachability_certified"]
    assert len(report["B56_target_count_witnesses"]) == 17
    assert all(item["count_witness"]["minimum_edge_count"] > 0 for item in report["B56_target_count_witnesses"])
    words = [item["proof"] for item in report["B56_word_budgets"]]
    assert [word["maximum_repetitions"] for word in words] == [1, 2]
    assert [word["first_empty_repetition"] for word in words] == [2, 3]
    assert all(word["status"] == "finite_repetition_budget" for word in words)
    assert report["primary_proof_origin_remains_B47"]
    assert report["new_conditional_trajectory_steps"] == report["new_live_graph_steps"] == 0
    assert not report["indefinite_trapping_certified"] and not report["future_runtime_certified"]


@pytest.mark.parametrize("index", range(9))
def test_history_rejects_altered_lineage(inputs, index):
    changed = list(inputs)
    changed[index] += b" "
    with pytest.raises(ValueError, match="lineage"):
        campaign.analyze_c6_winding_return_histories(json.loads(changed[0]), **arguments(changed))


@pytest.mark.parametrize("mutation", ("claim", "result", "origin", "guard", "target"))
def test_history_reconstructs_parent_claims(monkeypatch, inputs, previous, mutation):
    parent = json.loads(inputs[9])
    if mutation == "claim":
        parent["manifest"]["claim_id"] = "invented"
    elif mutation == "result":
        parent["excluded_first_exit_slabs"] += 1
    elif mutation == "origin":
        parent["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "guard":
        parent["B55_return_envelope"]["return_relation"][0]["source_guard"][0][6] += 1
    else:
        parent["B55_union_queries"][0]["query"]["target_regions"][0]["bounds"][0][6] += 1
    changed = (*inputs[:9], json.dumps(parent).encode(), inputs[10])
    monkeypatch.setattr(campaign, "analyze_c6_winding_return_unions", lambda *_a, **_k: deepcopy(previous))
    with pytest.raises(ValueError, match="lineage|reconstruction"):
        campaign.analyze_c6_winding_return_histories(json.loads(changed[0]), **arguments(changed))


@pytest.mark.parametrize("mutation", ("relation", "circulation", "generator", "duplicate", "endpoint", "missing", "terminal", "word"))
def test_history_checks_untrusted_integer_and_word_proposals(monkeypatch, inputs, previous, mutation):
    proposal = json.loads(inputs[10])
    if mutation == "relation":
        proposal["return_relation_sha256"] = "0" * 64
    elif mutation == "circulation":
        proposal["positive_circulation"][0] += 1
    elif mutation == "generator":
        proposal["coordinate_generators"][0][0][1] += 1
    elif mutation == "duplicate":
        proposal["coordinate_generators"][0].append(proposal["coordinate_generators"][0][0])
    elif mutation == "endpoint":
        proposal["remaining_target_proposals"][0]["endpoint_grid_coordinates"][0] = 10**100
    elif mutation == "missing":
        proposal["remaining_target_proposals"].pop()
    elif mutation == "terminal":
        next(p for p in proposal["remaining_target_proposals"] if p["terminal_transient_edge_index"] is not None)["terminal_transient_edge_index"] = -1
    else:
        proposal["closed_return_words"][0]["edge_indices"] = [884, 884]
    changed = (*inputs[:10], json.dumps(proposal).encode())
    monkeypatch.setattr(campaign, "analyze_c6_winding_return_unions", lambda *_a, **_k: deepcopy(previous))
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_return_histories(json.loads(changed[0]), **arguments(changed))


def test_word_resource_stop_cannot_promote_a_repeat_bound(monkeypatch, inputs, previous):
    monkeypatch.setattr(campaign, "analyze_c6_winding_return_unions", lambda *_a, **_k: deepcopy(previous))
    result = campaign.analyze_c6_winding_return_histories(json.loads(inputs[0]), **arguments(inputs), max_word_work_items=1)
    assert all(item["proof"]["status"] == "word_resource_limit" for item in result["B56_word_budgets"])
    assert all(item["proof"]["maximum_repetitions"] is None for item in result["B56_word_budgets"])
    assert result["excluded_first_exit_slabs"] == 39 and not result["indefinite_trapping_certified"]


def cli_args(monkeypatch, paths, output):
    flags = ("--input", "--relay-input", "--historical-input", "--envelope-input", "--region-input",
             "--excursion-input", "--mode-candidates-input", "--mode-input", "--return-input",
             "--union-input", "--history-candidates-input")
    argv = ["c6_winding_invariant_region.py", "--method", "histories", "--output", str(output)]
    for flag, path in zip(flags, paths, strict=True):
        argv.extend((flag, str(path)))
    monkeypatch.setattr(sys, "argv", argv)


def test_history_cli_binds_eleven_input_artifacts(tmp_path, monkeypatch, inputs, report):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(11))
    for path, raw in zip(paths, inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    monkeypatch.setattr(campaign, "analyze_c6_winding_return_histories", lambda *_a, **_k: deepcopy(report))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-return-histories"
    assert [item["sha256"] for item in result["input_evidence"]] == [hashlib.sha256(raw).hexdigest() for raw in inputs]
    assert result["excluded_first_exit_slabs"] == 39 and not result["indefinite_trapping_certified"]


@pytest.mark.parametrize("index", range(11))
def test_history_cli_cannot_overwrite_any_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(11))
    for path in paths:
        path.write_text("{}")
    cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()


@pytest.mark.parametrize("which", (*range(11), "source"))
def test_history_cli_detects_midflight_input_changes(tmp_path, monkeypatch, inputs, report, which):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(11))
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

    monkeypatch.setattr(campaign, "analyze_c6_winding_return_histories", analyze)
    cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()

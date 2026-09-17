"""Scope, provenance and whole-region controls for the C6 return campaign."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks import c6_winding_invariant_region as campaign


@pytest.fixture(scope="module")
def return_inputs():
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
    )
    if any(not (root / name).exists() for name in names):
        pytest.skip("the retained C6 research evidence is not available")
    return tuple((root / name).read_bytes() for name in names)


def arguments(raws):
    return dict(zip(("parent_bytes", "relay_bytes", "historical_bytes", "envelope_bytes", "region_bytes",
                     "excursion_bytes", "candidates_bytes", "mode_bytes"), raws, strict=True))


@pytest.fixture(scope="module")
def previous_modes(return_inputs):
    kwargs = arguments(return_inputs)
    kwargs.pop("mode_bytes")
    return campaign.analyze_c6_winding_mode_excursions(json.loads(return_inputs[0]), **kwargs)


@pytest.fixture(scope="module")
def return_report(return_inputs, previous_modes):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "analyze_c6_winding_mode_excursions", lambda *_a, **_k: deepcopy(previous_modes))
        return campaign.analyze_c6_winding_return_regions(json.loads(return_inputs[0]), **arguments(return_inputs))


def test_full_return_report_excludes_only_two_additional_complete_slabs(return_report):
    result = return_report
    assert result["excluded_first_exit_slabs"] == 38
    assert result["additional_first_exit_slabs_excluded"] == 2
    assert len(result["B54_remaining_first_exit_facets"]) == 18
    assert result["B54_additional_first_exit_facets"] == (
        dict(mask=4, node=0, direction="lower"), dict(mask=12, node=0, direction="lower"),
    )
    envelope = result["B54_return_envelope"]
    assert envelope["status"] == "fixed_point" and envelope["relation_complete"]
    assert len(envelope["return_relation"]) == 1265
    assert envelope["ordinary_transition_count"] == 830
    assert not envelope["clipped_forward_inclusion_certified"]
    positive = [r["query"] for r in result["B54_region_queries"] if r["query"]["origin_path_within_domain_excluded"]]
    assert len(positive) == 2
    assert all(q["status"] == "empty_complete_layer" and q["completed_depth"] == 7 for q in positive)
    assert all(not item["origin_present"] for q in positive for item in q["iterations"])
    assert result["new_conditional_trajectory_steps"] == result["new_live_graph_steps"] == 0
    assert result["primary_proof_origin_remains_B47"]
    assert result["B54_reconstruction"]["prior_excursion_prefix_steps_replayed"] == 56
    assert not result["indefinite_trapping_certified"] and not result["future_runtime_certified"]


@pytest.mark.parametrize("index", range(7))
def test_return_audit_rejects_any_changed_historical_input(return_inputs, index):
    changed = list(return_inputs)
    changed[index] += b" "
    with pytest.raises(ValueError, match="lineage"):
        campaign.analyze_c6_winding_return_regions(json.loads(changed[0]), **arguments(changed))


@pytest.mark.parametrize("mutation", ("result", "origin", "domain", "claim"))
def test_return_audit_rebuilds_stored_mode_claims(monkeypatch, return_inputs, previous_modes, mutation):
    old = json.loads(return_inputs[-1])
    if mutation == "result":
        old["excluded_first_exit_slabs"] += 1
    elif mutation == "origin":
        old["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "domain":
        old["B53_post_exclusion_reachable_envelope"]["retained_zones"][0]["bounds"][0][6] += 1
    else:
        old["manifest"]["claim_id"] = "invented"
    changed = (*return_inputs[:-1], json.dumps(old).encode())
    monkeypatch.setattr(campaign, "analyze_c6_winding_mode_excursions", lambda *_a, **_k: deepcopy(previous_modes))
    with pytest.raises(ValueError, match="lineage|reconstruction"):
        campaign.analyze_c6_winding_return_regions(json.loads(changed[0]), **arguments(changed))


@pytest.mark.parametrize("limit", ("max_intersections", "query_max_intersections"))
def test_return_resource_guard_does_not_promote_exclusion(monkeypatch, return_inputs, previous_modes, limit):
    monkeypatch.setattr(campaign, "analyze_c6_winding_mode_excursions", lambda *_a, **_k: deepcopy(previous_modes))
    result = campaign.analyze_c6_winding_return_regions(
        json.loads(return_inputs[0]), **arguments(return_inputs), **{limit: 1},
    )
    assert result["excluded_first_exit_slabs"] == 36
    assert result["additional_first_exit_slabs_excluded"] == 0
    assert len(result["B54_remaining_first_exit_facets"]) == 20
    assert not result["indefinite_trapping_certified"]


def cli_args(monkeypatch, paths, output):
    flags = ("--input", "--relay-input", "--historical-input", "--envelope-input", "--region-input",
             "--excursion-input", "--mode-candidates-input", "--mode-input")
    argv = ["c6_winding_invariant_region.py", "--method", "returns", "--output", str(output)]
    for flag, path in zip(flags, paths, strict=True):
        argv.extend((flag, str(path)))
    monkeypatch.setattr(sys, "argv", argv)


def test_return_cli_binds_eight_inputs(tmp_path, monkeypatch, return_inputs, return_report):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(8))
    for path, raw in zip(paths, return_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    monkeypatch.setattr(campaign, "analyze_c6_winding_return_regions", lambda *_a, **_k: deepcopy(return_report))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-return-regions"
    assert [r["sha256"] for r in result["input_evidence"]] == [hashlib.sha256(raw).hexdigest() for raw in return_inputs]
    assert result["excluded_first_exit_slabs"] == 38 and not result["indefinite_trapping_certified"]


@pytest.mark.parametrize("index", range(8))
def test_return_cli_cannot_overwrite_any_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(8))
    for path in paths:
        path.write_text("{}")
    cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.parametrize("which", (*range(8), "source"))
def test_return_cli_rejects_changed_input_or_source(tmp_path, monkeypatch, return_inputs, return_report, which):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(8))
    for path, raw in zip(paths, return_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    provenance = iter((("a" * 40, True, "sha256:" + "b" * 64),
                       ("a" * 40, True, "sha256:" + ("c" if which == "source" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(provenance))

    def analyze(*_a, **_k):
        if which != "source":
            paths[which].write_text("{}")
        return deepcopy(return_report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_return_regions", analyze)
    cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()

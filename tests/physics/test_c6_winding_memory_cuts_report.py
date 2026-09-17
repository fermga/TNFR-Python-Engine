"""Exact query-operator comparison after proved C6 slab cuts."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks import c6_winding_invariant_region as campaign
from tnfr.dynamics._euler_kernel import NodalRemainderState
from tnfr.physics import c6_carried_return as owner
from tnfr.physics.c6_carried_viability import C6CarriedForwardZone


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
        "artifacts/research/c6_winding_return_memory.json",
    )
    if any(not (root / name).is_file() for name in names):
        pytest.skip("the retained C6 research evidence is unavailable")
    return tuple((root / name).read_bytes() for name in names)


def arguments(raws):
    return dict(zip(("parent_bytes", "relay_bytes", "historical_bytes", "envelope_bytes", "region_bytes",
                     "excursion_bytes", "candidates_bytes", "mode_bytes", "return_bytes", "union_bytes",
                     "history_candidates_bytes", "history_bytes", "memory_bytes"), raws, strict=True))


@pytest.fixture(scope="module")
def previous(inputs):
    # Full B57 ancestor reconstruction belongs to the production campaign and
    # its existing tests. Isolate B58 admission and exact geometry here.
    retained = json.loads(inputs[12])
    for key in ("manifest", "source_scope", "input_evidence"):
        retained.pop(key)
    return retained


def patch_previous(monkeypatch, previous):
    monkeypatch.setattr(campaign, "analyze_c6_winding_return_memory", lambda *_a, **_k: deepcopy(previous))


@pytest.fixture(scope="module")
def report(inputs, previous):
    with pytest.MonkeyPatch.context() as patch:
        patch_previous(patch, previous)
        return campaign.analyze_c6_winding_memory_cuts(
            json.loads(inputs[0]), **arguments(inputs), max_memory_work=120_000,
            query_max_intersections=30_000,
        )


def test_cut_report_preserves_nodal_source_origin_and_scope(inputs, report):
    previous = json.loads(inputs[12])
    assert report["source"] == previous["source"]
    assert campaign._payload(report["B58_return_envelope"]["state"]) == previous["B57_return_envelope"]["state"]
    assert report["primary_proof_origin_remains_B47"]
    assert report["new_conditional_trajectory_steps"] == report["new_live_graph_steps"] == 0
    assert not report["contract"]["physical_dynamics_changed"]
    assert report["B58_reconstruction"]["prior_B57_completely_rebuilt"]
    assert report["B58_reconstruction"]["prior_first_exit_slabs_reverified"] == 40
    assert report["B58_reconstruction"]["prior_memory_sha256"] == hashlib.sha256(inputs[12]).hexdigest()
    assert report["excluded_first_exit_slabs"] == 40
    assert report["additional_first_exit_slabs_excluded"] == 0
    for key in ("indefinite_trapping_certified", "whole_band_exit_certified", "future_runtime_certified",
                "asymptotic_convergence_certified", "all_first_exit_slabs_excluded"):
        assert report[key] is False
    json.dumps(campaign._payload(report), allow_nan=False)


def test_cut_report_keeps_all_sixteen_full_original_targets(inputs, report):
    previous, original = json.loads(inputs[12]), json.loads(inputs[9])
    labels = previous["B57_remaining_first_exit_facets"]
    assert len(labels) == 16
    assert report["B58_remaining_first_exit_facets"] == labels
    assert report["B58_outgoing_facets"] == previous["B57_outgoing_facets"]
    assert report["B58_exit_groups"] == previous["B57_exit_groups"]
    assert report["B58_original_cube_geometry"] == original["B55_cube_geometry"]
    expected = [item["query"]["target_regions"] for item in original["B55_union_queries"] if item["target"] in labels]
    assert campaign._payload(report["B58_original_target_groups"]) == expected
    initializations = report["B58_query_equivalence"]["target_initializations"]
    assert [item["target"] for item in initializations] == labels
    expected_cuts = (json.loads(inputs[8])["B54_additional_first_exit_facets"]
                     + original["B55_additional_first_exit_facets"] + previous["B57_additional_first_exit_facets"])
    assert list(report["B58_proved_cuts"]) == expected_cuts


def test_cut_report_cannot_promote_matching_counts_to_equivalence(report):
    comparison = report["B58_query_equivalence"]
    assert comparison["canonical_source_equal"]
    assert not comparison["actual_origin_reachability_certified"]
    assert not comparison["conditional_boundedness_certified"]
    conditions = (
        comparison["canonical_source_equal"], comparison["return_record_identities_equal"],
        comparison["retained_endpoint_arrays_equal"], comparison["origin_images_equal"],
        comparison["complete_query_graph_equal"], all(item["equal"] for item in comparison["target_initializations"]),
    )
    assert comparison["all_target_operators_and_initial_conditions_identical"] == all(conditions)
    assert report["same_query_geometry_repetition_avoided"] == all(conditions)
    for item in comparison["target_initializations"]:
        if item["equal"]:
            assert item["complete"] and item["before_sha256"] == item["after_sha256"]
        else:
            assert not report["same_query_geometry_repetition_avoided"]


@pytest.mark.parametrize("index", range(12))
def test_cut_audit_rejects_each_altered_ancestor(inputs, index):
    changed = list(inputs)
    changed[index] += b" "
    with pytest.raises(ValueError, match="lineage"):
        campaign.analyze_c6_winding_memory_cuts(json.loads(changed[0]), **arguments(changed))


@pytest.mark.parametrize("mutation", ("claim", "scope", "origin", "positive_count", "positive_flag", "guard",
                                      "target", "endpoint", "budget"))
def test_cut_audit_reconstructs_untrusted_parent_claims(monkeypatch, inputs, previous, mutation):
    parent = json.loads(inputs[12])
    if mutation == "claim":
        parent["manifest"]["claim_id"] = "invented"
    elif mutation == "scope":
        parent["source_scope"] = []
    elif mutation == "origin":
        parent["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "positive_count":
        parent["excluded_first_exit_slabs"] += 1
    elif mutation == "positive_flag":
        query = next(item["query"] for item in parent["B57_memory_queries"]
                     if not item["query"]["origin_path_within_domain_excluded"])
        query["origin_path_within_domain_excluded"] = True
    elif mutation == "guard":
        parent["B57_return_envelope"]["return_relation"][0]["source_guard"][0][6] += 1
    elif mutation == "target":
        parent["B57_memory_queries"][0]["query"]["target_regions"][0]["bounds"][0][6] += 1
    elif mutation == "endpoint":
        zone = next(zone for zone in parent["B57_memory_envelope"]["retained_endpoint_zones"] if zone is not None)
        zone[0][6] += 1
    else:
        parent["contract"]["max_memory_work"] += 1
    changed = (*inputs[:12], json.dumps(parent).encode())
    patch_previous(monkeypatch, previous)
    with pytest.raises(ValueError, match="lineage|reconstruction"):
        campaign.analyze_c6_winding_memory_cuts(json.loads(changed[0]), **arguments(changed))


def _hull(values):
    points = tuple((value, 0, 0, 0, 0, 0, 0) for value in values)
    return tuple(tuple(max(point[i] - point[j] for point in points) for j in range(7)) for i in range(7))


@pytest.fixture(scope="module")
def small_memory():
    # A private integer transition fixture, not a canonical pressure claim.
    # Its unreachable interval-valued self-return detects changed DBMs even
    # when every edge count, mode identity and nonempty-array count is fixed.
    rows = ((.4,) * 6, (.5,) * 6, (.6,) * 6)
    specs = ((0, 1, (0,), 1), (1, 0, (1,), 1), (0, 0, (2,), 0), (0, 0, (10, 11, 12), 0))
    edges = tuple(owner.C6CarriedReturnTransition(
        rows[a], None, rows[b], _hull(values), None, _hull(value + shift for value in values),
        (shift, 0, 0, 0, 0, 0),
    ) for a, b, values, shift in specs)
    zones = tuple(C6CarriedForwardZone(row, _hull(values)) for row, values in zip(
        rows, (range(13), (1,), (3, 11, 12, 13)), strict=True,
    ))
    terminals = tuple(owner.C6CarriedReturnTransition(
        rows[0], None, rows[2], _hull(values), None, _hull(value + 1 for value in values),
        (1, 0, 0, 0, 0, 0),
    ) for values in ((2,), (10, 11, 12)))
    state = NodalRemainderState(rows[0], (F(0),) * 6, .375, .625)
    envelope = owner.C6CarriedReturnEnvelope(
        None, state, 1., rows, ((0.,) * 6,) * 3, F(1), state.exact_epi,
        (rows[2],), zones, zones, edges, terminals, 6, True, True, (), "fixed_point", 0, 0, 1, 3,
    )
    return owner._derive_return_memory_envelope(envelope, max_memory_work=1000, max_memory_arcs=100)


def _small_groups(memory):
    return ((C6CarriedForwardZone(memory.return_envelope.epi_states[2], _hull((3, 11, 12, 13))),),)


def test_complete_operator_comparison_admits_only_exact_geometry(small_memory):
    result = campaign._compare_c6_memory_query_geometry(small_memory, small_memory, _small_groups(small_memory), 1000)
    assert result["all_target_operators_and_initial_conditions_identical"]
    assert result["before_arc_count"] == result["after_arc_count"] > 0
    assert result["before_graph_sha256"] == result["after_graph_sha256"]
    assert result["target_initializations"][0]["equal"]
    assert not result["actual_origin_reachability_certified"] and not result["conditional_boundedness_certified"]


@pytest.mark.parametrize("mutation", ("guard", "endpoint", "initial"))
def test_equal_counts_do_not_hide_changed_guard_endpoint_or_initial_dbm(small_memory, mutation):
    if mutation == "guard":
        changed = replace(small_memory, source_guards=(*small_memory.source_guards[:3], _hull((10, 11))))
    elif mutation == "endpoint":
        changed = replace(small_memory, retained_endpoint_zones=(*small_memory.retained_endpoint_zones[:3], _hull((10, 11))))
    else:
        envelope = small_memory.return_envelope
        terminal = replace(envelope.intermediate_transitions[-1], source_guard=_hull((10, 11)))
        changed = replace(small_memory, return_envelope=replace(
            envelope, intermediate_transitions=(*envelope.intermediate_transitions[:-1], terminal),
        ))
    groups = _small_groups(small_memory)
    before_initial, before_complete, _ = owner._return_memory_query_initialization(small_memory, groups[0], 1000)
    after_initial, after_complete, _ = owner._return_memory_query_initialization(changed, groups[0], 1000)
    assert before_complete and after_complete
    assert sum(zone is not None for zone in before_initial) == sum(zone is not None for zone in after_initial)
    result = campaign._compare_c6_memory_query_geometry(small_memory, changed, groups, 1000)
    assert result["before_graph_complete"] and result["after_graph_complete"]
    assert result["before_arc_count"] == result["after_arc_count"] > 0
    assert result["return_record_identities_equal"]
    assert not result["all_target_operators_and_initial_conditions_identical"]
    if mutation == "guard":
        assert result["retained_endpoint_arrays_equal"]
        assert not result["complete_query_graph_equal"]
        assert result["before_graph_sha256"] != result["after_graph_sha256"]
    elif mutation == "endpoint":
        assert not result["retained_endpoint_arrays_equal"]
    else:
        assert result["retained_endpoint_arrays_equal"] and result["complete_query_graph_equal"]
        item = result["target_initializations"][0]
        assert item["complete"] and not item["equal"]
        assert item["before_sha256"] != item["after_sha256"]


def test_tiny_comparison_budget_never_promotes_two_partial_results(small_memory):
    result = campaign._compare_c6_memory_query_geometry(small_memory, small_memory, _small_groups(small_memory), 1)
    assert not result["before_graph_complete"] and not result["after_graph_complete"]
    assert result["before_arc_count"] == result["after_arc_count"] == 0
    assert result["before_graph_sha256"] == result["after_graph_sha256"]
    assert not result["complete_query_graph_equal"]
    assert not result["all_target_operators_and_initial_conditions_identical"]
    assert all(not item["complete"] and not item["equal"] for item in result["target_initializations"])


def cli_args(monkeypatch, paths, output):
    flags = ("--input", "--relay-input", "--historical-input", "--envelope-input", "--region-input",
             "--excursion-input", "--mode-candidates-input", "--mode-input", "--return-input",
             "--union-input", "--history-candidates-input", "--history-input", "--memory-input")
    argv = ["c6_winding_invariant_region.py", "--method", "memory-cuts", "--output", str(output)]
    for flag, path in zip(flags, paths, strict=True):
        argv.extend((flag, str(path)))
    monkeypatch.setattr(sys, "argv", argv)


def test_cut_cli_binds_all_thirteen_inputs_in_order(tmp_path, monkeypatch, inputs, report):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(13))
    for path, raw in zip(paths, inputs, strict=True):
        path.write_bytes(raw)
    output, received = tmp_path / "result.json", []

    def analyze(parent, **kwargs):
        received.append((parent, kwargs))
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_memory_cuts", analyze)
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert received[0][0] == json.loads(inputs[0])
    assert all(received[0][1][key] == raw for key, raw in arguments(inputs).items())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-memory-cuts"
    assert [item["sha256"] for item in result["input_evidence"]] == [hashlib.sha256(raw).hexdigest() for raw in inputs]
    assert not result["indefinite_trapping_certified"]
    assert [path.read_bytes() for path in paths] == list(inputs)


@pytest.mark.parametrize("index", range(13))
def test_cut_cli_cannot_overwrite_any_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(13))
    for path in paths:
        path.write_text("{}")
    cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.parametrize("which", (*range(13), "source"))
def test_cut_cli_rejects_midflight_input_changes(tmp_path, monkeypatch, inputs, report, which):
    paths = tuple(tmp_path / f"input{i}.json" for i in range(13))
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

    monkeypatch.setattr(campaign, "analyze_c6_winding_memory_cuts", analyze)
    cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()

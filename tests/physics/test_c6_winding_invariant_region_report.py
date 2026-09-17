"""An unfinished universal-set search must never become a stability claim."""

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks import c6_winding_invariant_region as campaign

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts/research"


@pytest.fixture(scope="module")
def inputs():
    paths = tuple(ARTIFACTS / name for name in
                  (campaign.INPUT_NAME, campaign.RELAY_NAME, campaign.HISTORICAL_NAME))
    if not all(path.is_file() for path in paths):
        pytest.skip("the retained B43/B46/B47 lineage is unavailable")
    return tuple(path.read_bytes() for path in paths)


@pytest.fixture(scope="module")
def report(inputs):
    raw, relay, historical = inputs
    return campaign.analyze_c6_winding_invariant_region(
        json.loads(raw), relay_bytes=relay, historical_bytes=historical,
    )


@pytest.fixture(scope="module")
def forward_report(inputs):
    raw, relay, historical = inputs
    return campaign.analyze_c6_winding_forward_envelope(
        json.loads(raw), relay_bytes=relay, historical_bytes=historical,
    )


def test_forward_envelope_reuses_actual_lineage_without_advancing_it(forward_report, inputs):
    result = forward_report
    candidate = result["B49_profile_candidate"]
    parent = json.loads(inputs[0])
    assert candidate["actual_endpoint_mask"] == 57 and len(candidate["visible_rows"]) == 64
    assert campaign._payload(result["source"]["retained_B47_endpoint"]) == parent["B47_conditional_first_exit"]["endpoint"]
    assert result["source"]["historical_nodal_steps_replayed"] == 356
    assert result["source"]["previous_conditional_steps_replayed"] == 204
    assert not result["source"]["live_execution_seal_recreated"]
    assert result["new_conditional_trajectory_steps"] == result["new_live_graph_steps"] == 0


def test_actual_forward_layers_tighten_but_leave_outgoing_facets(forward_report):
    proof = forward_report["B49_relational_forward_envelope"]
    assert proof["grid_quantum"] == F(1, 2**113)
    assert len(proof["pair_barriers"]) == 15
    assert len(proof["initial_zones"]) == len(proof["retained_zones"]) == 64
    assert proof["status"] == "resource_limit" and proof["intersections"] == 250_000
    assert proof["iterations"][-1]["ordinal"] == 259
    assert proof["iterations"][0]["outgoing_facets"] == 72
    assert proof["iterations"][-1]["outgoing_facets"] == 56
    assert all(row["origin_retained"] for row in proof["iterations"])
    assert proof["entry_state"] is None and proof["entry_steps"] == ()
    for flag in ("conditional_invariance_certified", "conditional_boundedness_certified",
                 "future_runtime_certified", "asymptotic_convergence_certified"):
        assert proof[flag] is False
    for flag in ("indefinite_trapping_certified", "whole_band_exit_certified",
                 "future_runtime_certified", "all_correlated_candidates_excluded"):
        assert forward_report[flag] is False


def test_forward_envelope_preserves_input_and_discards_partial_layer(inputs):
    raw, relay, historical = inputs
    parent = json.loads(raw)
    before = deepcopy(parent)
    result = campaign.analyze_c6_winding_forward_envelope(
        parent, relay_bytes=relay, historical_bytes=historical, max_intersections=1,
    )
    assert parent == before
    proof = result["B49_relational_forward_envelope"]
    assert len(proof["iterations"]) == 1 and proof["initial_zones"] == proof["retained_zones"]
    assert proof["status"] == "resource_limit"
    json.dumps(campaign._payload(result), allow_nan=False)


def test_forward_cli_uses_distinct_claim_and_preserves_lineage(tmp_path, monkeypatch, inputs, forward_report):
    paths = tuple(tmp_path / name for name in ("b47.json", "b46.json", "b43.json"))
    for path, raw in zip(paths, inputs):
        path.write_bytes(raw)
    output = tmp_path / "forward.json"
    monkeypatch.setattr(campaign, "analyze_c6_winding_forward_envelope", lambda *_a, **_k: deepcopy(forward_report))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    _cli_args(monkeypatch, paths, output)
    monkeypatch.setattr(sys, "argv", sys.argv + ["--method", "forward"])
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-forward-envelope"
    assert "B48_exact_correlated_viability" not in result
    assert tuple(item["sha256"] for item in result["input_evidence"]) == tuple(hashlib.sha256(raw).hexdigest() for raw in inputs)
    assert result["manifest"]["dirty_source_hash"] == "sha256:" + "b" * 64


def test_candidate_is_derived_from_profile_and_keeps_actual_endpoint(report, inputs):
    parent = json.loads(inputs[0])
    candidate = report["B48_profile_candidate"]
    assert candidate["actual_endpoint_mask"] == 57
    assert len(candidate["visible_rows"]) == 64
    assert tuple(candidate["visible_rows"][57]) == tuple(parent["B47_conditional_first_exit"]["endpoint"]["epi"])
    for target, pair in zip(candidate["exact_targets"], candidate["coordinate_pairs"]):
        assert F(pair[0]) < target < F(pair[1])
    assert campaign._payload(report["source"]["retained_B47_endpoint"]) == parent["B47_conditional_first_exit"]["endpoint"]
    assert report["source"]["historical_nodal_steps_replayed"] == 356
    assert report["source"]["previous_conditional_steps_replayed"] == 204
    assert not report["source"]["live_execution_seal_recreated"]


def test_actual_pressure_has_exact_negative_quadratic_direction(report):
    control = report["B48_common_quadratic_controls"]
    matrix, metric = control["affine_matrix"], control["symmetrizing_diagonal"]
    vector = control["negative_direction"]
    assert all(value > 0 for value in metric)
    assert all(metric[i] * matrix[i][j] == metric[j] * matrix[j][i] for i in range(6) for j in range(6))
    value = sum((vector[i] * metric[i] * matrix[i][j] * vector[j] for i in range(6) for j in range(6)), F(0))
    assert value == control["negative_quadratic_value"] == -F(4124249100202121, 274949940013475)
    assert control["mixed_bounds"] == ((0, 1), (0, 0), (-8, 0), (-8, 0), (-8, 0), (0, 0))
    assert any(control["mixed_remainders"][mask][i] for mask in range(64) for i in range(6))


def test_positive_dual_identity_binds_actual_nonlinear_rows(report):
    control = report["B48_common_quadratic_controls"]
    unit = control["increment_unit"]
    weights = control["positive_dual_weights"]
    assert len(weights) == 21 and all(weight > 0 for weight in weights) and sum(weights) == 1
    assert control["dual_identity_residual"] == (F(0),) * 21
    # Independent evaluation at H=I: S=(Q^T Q)^-1 has trace25/6.
    scale = abs(control["affine_matrix"][1][0])
    oriented = sum((weight * (1 if mask >> node & 1 else -1)
                    * control["exact_nodal_areas"][mask][node] / unit / scale
                    for weight, (mask, node) in zip(weights, control["positive_dual_support"])), F(0))
    assert oriented == control["positive_trace_multiplier"] * F(25, 6) > 0
    assert control["all_orthant_inward_common_quadratic_criterion_excluded"]
    assert not control["all_quadratic_invariants_excluded"]
    assert not control["piecewise_potentials_excluded"]
    assert not control["actual_trajectory_instability_certified"]


def test_complete_descents_preserve_state_but_do_not_prove_future(report):
    proof = report["B48_exact_correlated_viability"]
    assert proof["status"] == "resource_limit"
    assert tuple(row["box_count"] for row in proof["iterations"]) == (64, 64, 416, 4285)
    assert all(row["origin_retained"] for row in proof["iterations"])
    counts = tuple(row["point_count"] for row in proof["iterations"])
    assert all(a > b for a, b in zip(counts, counts[1:]))
    assert len(proof["retained_boxes"]) == 4285
    assert proof["coordinate_spacings"] == tuple(F(value, 2**113) for value in (1, 1, 4, 8, 8, 8))
    assert proof["exclusion_step_bound"] is None
    for flag in ("conditional_invariance_certified", "conditional_boundedness_certified", "origin_exit_certified",
                 "future_runtime_certified", "asymptotic_convergence_certified"):
        assert proof[flag] is False
    assert report["new_conditional_trajectory_steps"] == report["new_live_graph_steps"] == 0
    assert not report["indefinite_trapping_certified"] and not report["candidate_exit_certified"]
    assert not report["all_correlated_candidates_excluded"]
    assert not report["whole_band_exit_certified"] and not report["future_runtime_certified"]


def test_report_keeps_input_objects_unchanged_and_serializes_exactly(inputs):
    raw, relay, historical = inputs
    parent = json.loads(raw)
    before = deepcopy(parent)
    small = campaign.analyze_c6_winding_invariant_region(
        parent, relay_bytes=relay, historical_bytes=historical, max_work_items=1,
    )
    assert parent == before
    assert len(small["B48_exact_correlated_viability"]["iterations"]) == 1
    assert small["B48_exact_correlated_viability"]["status"] == "resource_limit"
    json.dumps(campaign._payload(small), allow_nan=False)


def _cli_args(monkeypatch, paths, output):
    monkeypatch.setattr(sys, "argv", [
        "campaign", "--input", str(paths[0]), "--relay-input", str(paths[1]),
        "--historical-input", str(paths[2]), "--output", str(output),
    ])


def test_cli_retains_all_three_input_hashes_and_current_source(tmp_path, monkeypatch, inputs, report):
    paths = tuple(tmp_path / name for name in ("b47.json", "b46.json", "b43.json"))
    for path, raw in zip(paths, inputs):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    monkeypatch.setattr(campaign, "analyze_c6_winding_invariant_region", lambda *_a, **_k: deepcopy(report))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    _cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert tuple(item["sha256"] for item in result["input_evidence"]) == tuple(hashlib.sha256(raw).hexdigest() for raw in inputs)
    assert result["manifest"]["dirty_source_hash"] == "sha256:" + "b" * 64


@pytest.mark.parametrize("index", range(3))
def test_cli_cannot_overwrite_any_lineage_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"{i}.json" for i in range(3))
    for path in paths:
        path.write_text("{}")
    _cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.parametrize("which", (0, 1, 2, "source"))
def test_cli_rejects_changed_source_or_any_retained_input(tmp_path, monkeypatch, inputs, report, which):
    paths = tuple(tmp_path / f"{i}.json" for i in range(3))
    for path, raw in zip(paths, inputs):
        path.write_bytes(raw)
    output = tmp_path / "result.json"
    provenance = iter((("a" * 40, True, "sha256:" + "b" * 64),
                       ("a" * 40, True, "sha256:" + ("c" if which == "source" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(provenance))

    def analyze(*_args, **_kwargs):
        if which != "source":
            paths[which].write_text("{}")
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_invariant_region", analyze)
    _cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()


@pytest.fixture(scope="module")
def predecessor_inputs(inputs):
    paths = tuple(ARTIFACTS / name for name in (
        "c6_winding_forward_envelope.json", "c6_winding_forward_envelope.validation.json",
    ))
    if not all(path.is_file() for path in paths):
        pytest.skip("the retained B49 envelope and selected exact witnesses are unavailable")
    return inputs + tuple(path.read_bytes() for path in paths)


def _analyze_predecessors(raw_inputs, **kwargs):
    parent, relay, historical, envelope, witnesses = raw_inputs
    return campaign.analyze_c6_winding_temporal_predecessors(
        json.loads(parent), relay_bytes=relay, historical_bytes=historical,
        envelope_bytes=envelope, witness_bytes=witnesses, **kwargs,
    )


@pytest.fixture(scope="module")
def predecessor_report(predecessor_inputs, forward_report):
    # The existing fixture already recomputes all 259 B49 image layers. Reuse
    # that result while retaining the new producer's independent B47 replay.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "analyze_c6_winding_forward_envelope",
                      lambda *_a, **_k: deepcopy(forward_report))
        return _analyze_predecessors(predecessor_inputs)


@pytest.fixture
def cached_forward(monkeypatch, forward_report):
    monkeypatch.setattr(campaign, "analyze_c6_winding_forward_envelope",
                        lambda *_a, **_k: deepcopy(forward_report))


def test_predecessor_report_preserves_lineage_and_rebinds_selected_outward_points(predecessor_report, predecessor_inputs):
    result = predecessor_report
    binding = result["B50_B49_reconstruction"]
    assert binding["complete_envelope_rebuilt"]
    assert binding["completed_layers"] == 259 and binding["outgoing_facets"] == 56
    assert binding["clipped_forward_inclusion_verified"]
    assert binding["envelope_sha256"] == hashlib.sha256(predecessor_inputs[3]).hexdigest()
    assert binding["witness_sha256"] == hashlib.sha256(predecessor_inputs[4]).hexdigest()
    parent = json.loads(predecessor_inputs[0])
    assert campaign._payload(result["source"]["retained_B47_endpoint"]) == parent["B47_conditional_first_exit"]["endpoint"]
    assert result["source"]["historical_nodal_steps_replayed"] == 356
    assert result["source"]["previous_conditional_steps_replayed"] == 204
    assert not result["source"]["live_execution_seal_recreated"]
    records = result["B50_selected_point_pasts"]
    assert tuple(item["mask"] for item in records) == (8, 12, 16, 20, 24, 28, 49, 51, 57, 59)
    origin = result["source"]["retained_B47_endpoint"]
    original_sum = sum(F(x) + F(r) for x, r in zip(origin["epi"], origin["remainder"], strict=True))
    for record in records:
        target = record["target"]
        assert sum(F(x) + F(r) for x, r in zip(target["epi"], target["remainder"], strict=True)) == original_sum
        assert record["outgoing_step"]["before"] == target
        assert set(record["domains"]) == {"retained_R259", "original_cube"}
        assert sum(record["grid_indices"]) == 0


def test_default_predecessor_outcomes_remain_point_and_domain_specific(predecessor_report):
    records = predecessor_report["B50_selected_point_pasts"]
    retained = tuple(item["domains"]["retained_R259"] for item in records)
    original = tuple(item["domains"]["original_cube"] for item in records)
    assert sum(proof["status"] == "past_excluded" for proof in retained) == 9
    assert sum(proof["status"] == "past_excluded" for proof in original) == 5
    assert {item["mask"]: item["domains"]["retained_R259"]["past_exclusion_depth"]
            for item in records if item["domains"]["retained_R259"]["status"] == "past_excluded"} == {
                8: 12, 12: 9, 16: 3, 20: 1, 24: 3, 49: 1, 51: 1, 57: 1, 59: 4,
            }
    assert {item["mask"]: item["domains"]["original_cube"]["past_exclusion_depth"]
            for item in records if item["domains"]["original_cube"]["status"] == "past_excluded"} == {
                20: 42, 49: 91, 51: 120, 57: 91, 59: 40,
            }
    assert {item["mask"]: item["domains"]["original_cube"]["completed_depth"]
            for item in records if item["domains"]["original_cube"]["status"] == "resource_limit"} == {
                8: 11, 12: 9, 16: 32, 24: 32, 28: 13,
            }
    remaining = next(item["domains"]["retained_R259"] for item in records if item["mask"] == 28)
    assert remaining["status"] == "resource_limit" and remaining["completed_depth"] == 21
    for record in records:
        for proof in record["domains"].values():
            assert proof["max_depth"] == 128 and proof["max_row_checks"] == 32_768
            assert proof["frontier_counts"][0] == 1
            assert proof["frontier_counts"] == tuple(len(layer["states"]) for layer in proof["layers"])
            assert proof["completed_depth"] == proof["layers"][-1]["depth"]
            assert proof["origin_reachability_depths"] == ()
            assert not proof["finite_origin_reachability_certified"]
            assert not proof["whole_domain_exclusion_certified"]
            assert not proof["conditional_boundedness_certified"] and not proof["future_runtime_certified"]
            if proof["status"] == "past_excluded":
                assert proof["frontier_counts"][-1] == 0
                assert proof["past_exclusion_depth"] == proof["completed_depth"]
                assert proof["maximum_compatible_past_depth"] == proof["completed_depth"] - 1
                assert proof["origin_path_within_domain_excluded"]
            else:
                assert proof["status"] in ("depth_limit", "resource_limit")
                assert proof["past_exclusion_depth"] is None
                assert proof["maximum_compatible_past_depth"] is None
                assert not proof["origin_path_within_domain_excluded"]
        assert record["selected_point_cannot_cause_first_cube_exit"] == (record["mask"] != 28)
    for flag in ("selected_points_are_actual_reached_states", "all_outgoing_facets_excluded",
                 "indefinite_trapping_certified", "whole_band_exit_certified", "future_runtime_certified"):
        assert predecessor_report[flag] is False
    assert predecessor_report["new_conditional_trajectory_steps"] == predecessor_report["new_live_graph_steps"] == 0
    json.dumps(campaign._payload(predecessor_report), allow_nan=False)


@pytest.mark.parametrize("mutation", ("carry", "zone", "claim"))
def test_predecessors_reject_altered_retained_B49_data(predecessor_inputs, cached_forward, mutation):
    data = json.loads(predecessor_inputs[3])
    if mutation == "carry":
        state = data["B49_relational_forward_envelope"]["state"]
        state["remainder"][0] = str(F(state["remainder"][0]) + F(1, 2**3222))
    elif mutation == "zone":
        data["B49_relational_forward_envelope"]["retained_zones"][0]["bounds"][0][6] += 1
    else:
        data["manifest"]["claim_id"] = "O3.a-C6-carried-invariant-region"
    altered = predecessor_inputs[:3] + (json.dumps(data).encode(), predecessor_inputs[4])
    with pytest.raises(ValueError, match="retained B49|canonical reconstruction"):
        _analyze_predecessors(altered, max_depth=0)


def test_predecessors_reject_a_witness_file_for_different_B49_bytes(predecessor_inputs, cached_forward):
    validation = json.loads(predecessor_inputs[4])
    validation["input_sha256"] = "0" * 64
    altered = predecessor_inputs[:4] + (json.dumps(validation).encode(),)
    with pytest.raises(ValueError, match="do not bind"):
        _analyze_predecessors(altered, max_depth=0)


@pytest.mark.parametrize("mutation,message", (
    ("changed_mean", "preserve the exact original mean"),
    ("bool_index", "six exact integer indices"),
    ("invalid_node", "valid row, facet"),
    ("wrong_facet", "does not leave its declared cube facet"),
    ("duplicate", "must be distinct"),
    ("empty", "nonempty exact target witnesses"),
))
def test_predecessors_validate_selected_points_before_admitting_past_claims(
        predecessor_inputs, cached_forward, mutation, message):
    validation = json.loads(predecessor_inputs[4])
    selected = [item for item in validation["outgoing_slab_sum_projections"]
                if item["hypothetical_original_mean_affine_coset_witness"] is not None]
    first = selected[0]
    if mutation == "changed_mean":
        first["hypothetical_original_mean_affine_coset_witness"][0] += 1
    elif mutation == "bool_index":
        first["hypothetical_original_mean_affine_coset_witness"][0] = True
    elif mutation == "invalid_node":
        first["node"] = 6
    elif mutation == "wrong_facet":
        first["direction"] = "upper"
    elif mutation == "duplicate":
        validation["outgoing_slab_sum_projections"] = [first, deepcopy(first)]
    else:
        validation["outgoing_slab_sum_projections"] = []
    altered = predecessor_inputs[:4] + (json.dumps(validation).encode(),)
    with pytest.raises(ValueError, match=message):
        _analyze_predecessors(altered, max_depth=0)


def _predecessor_cli_args(monkeypatch, paths, output):
    _cli_args(monkeypatch, paths, output)
    monkeypatch.setattr(sys, "argv", sys.argv + [
        "--method", "predecessors", "--envelope-input", str(paths[3]),
        "--witness-input", str(paths[4]),
    ])


def test_predecessor_cli_binds_all_five_inputs_and_distinct_claim(
        tmp_path, monkeypatch, predecessor_inputs, predecessor_report):
    paths = tuple(tmp_path / f"input_{i}.json" for i in range(5))
    for path, raw in zip(paths, predecessor_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "predecessors.json"
    captured = {}

    def analyze(parent, **kwargs):
        captured.update(parent=parent, **kwargs)
        return deepcopy(predecessor_report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_temporal_predecessors", analyze)
    monkeypatch.setattr(campaign, "current_git_source_provenance",
                        lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    _predecessor_cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-temporal-predecessors"
    assert result["manifest"]["dirty_source_hash"] == "sha256:" + "b" * 64
    assert tuple(item["sha256"] for item in result["input_evidence"]) == tuple(
        hashlib.sha256(raw).hexdigest() for raw in predecessor_inputs
    )
    assert captured["envelope_bytes"] == predecessor_inputs[3]
    assert captured["witness_bytes"] == predecessor_inputs[4]
    assert captured["max_depth"] == 128 and captured["max_row_checks"] == 32_768
    assert len(result["B50_selected_point_pasts"]) == 10
    assert "B48_exact_correlated_viability" not in result
    assert "B49_relational_forward_envelope" not in result


@pytest.mark.parametrize("index", range(5))
def test_predecessor_cli_cannot_overwrite_any_of_its_five_inputs(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"input_{i}.json" for i in range(5))
    for path in paths:
        path.write_text("{}")
    _predecessor_cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.parametrize("which", (0, 1, 2, 3, 4, "source"))
def test_predecessor_cli_rejects_changed_source_or_any_input(
        tmp_path, monkeypatch, predecessor_inputs, predecessor_report, which):
    paths = tuple(tmp_path / f"input_{i}.json" for i in range(5))
    for path, raw in zip(paths, predecessor_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "predecessors.json"
    provenance = iter((("a" * 40, True, "sha256:" + "b" * 64),
                       ("a" * 40, True, "sha256:" + ("c" if which == "source" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(provenance))

    def analyze(*_args, **_kwargs):
        if which != "source":
            paths[which].write_text("{}")
        return deepcopy(predecessor_report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_temporal_predecessors", analyze)
    _predecessor_cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()


@pytest.fixture(scope="module")
def region_inputs(inputs):
    path = ARTIFACTS / "c6_winding_forward_envelope.json"
    if not path.is_file():
        pytest.skip("the retained B49 forward envelope is unavailable")
    return inputs + (path.read_bytes(),)


def _analyze_regions(raw_inputs, **kwargs):
    parent, relay, historical, envelope = raw_inputs
    return campaign.analyze_c6_winding_region_exclusions(
        json.loads(parent), relay_bytes=relay, historical_bytes=historical,
        envelope_bytes=envelope, **kwargs,
    )


@pytest.fixture(scope="module")
def region_report(region_inputs, forward_report):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "analyze_c6_winding_forward_envelope",
                      lambda *_a, **_k: deepcopy(forward_report))
        return _analyze_regions(region_inputs)


def test_region_report_binds_full_lineage_and_complete_RN_geometry(region_report, region_inputs):
    from tnfr.physics.nodal_remainder import derive_nodal_remainder_itinerary

    result = region_report
    binding = result["B51_B49_reconstruction"]
    assert binding["complete_envelope_rebuilt"] and binding["clipped_forward_inclusion_verified"]
    assert binding["envelope_sha256"] == hashlib.sha256(region_inputs[3]).hexdigest()
    state = result["source"]["retained_B47_endpoint"]
    assert campaign._payload(state) == json.loads(region_inputs[0])["B47_conditional_first_exit"]["endpoint"]
    assert result["source"]["historical_nodal_steps_replayed"] == 356
    assert result["source"]["previous_conditional_steps_replayed"] == 204
    assert not result["source"]["live_execution_seal_recreated"]
    geometry = result["B51_cube_geometry"]
    assert geometry["grid_quantum"] == F(1, 2**113)
    original = tuple(F(x) + F(r) for x, r in zip(state["epi"], state["remainder"], strict=True))
    assert geometry["affine_origin"] == original
    assert len(geometry["complete_RN_cells"]) == 64
    minima, maxima = [], []
    for cell in geometry["complete_RN_cells"]:
        row = cell["epi"]
        coordinates = derive_nodal_remainder_itinerary(
            epi_states=(row, row), timesteps=(0.,), capacities=((1.,) * 6,), pressures=((0.,) * 6,),
            epi_lower=state["epi_lower"], epi_upper=state["epi_upper"],
        ).coordinates
        low = tuple(((F(coordinate.first_grid_index, 2**3222) - initial) / geometry["grid_quantum"]).__ceil__()
                    for coordinate, initial in zip(coordinates, original, strict=True))
        high = tuple(((F(coordinate.last_grid_index, 2**3222) - initial) / geometry["grid_quantum"]).__floor__()
                     for coordinate, initial in zip(coordinates, original, strict=True))
        assert tuple(-cell["bounds"][6][i] for i in range(6)) == low
        assert tuple(cell["bounds"][i][6] for i in range(6)) == high
        minima.append(low)
        maxima.append(high)
    assert geometry["lower_grid"] == tuple(min(row[i] for row in minima) for i in range(6))
    assert geometry["upper_grid"] == tuple(max(row[i] for row in maxima) for i in range(6))


def test_region_report_covers_every_unsafe_slab_of_the_retained_domain(region_report, forward_report):
    from tnfr.physics.c6_carried_viability import _dbm_close

    proof = region_report["B51_region_predecessors"]
    facets = region_report["B51_outgoing_facets"]
    assert len(facets) == len(proof["queries"]) == 56
    assert len({(f["mask"], f["node"], f["direction"]) for f in facets}) == 56
    geometry = region_report["B51_cube_geometry"]
    source = forward_report["B49_relational_forward_envelope"]
    zones = {item["epi"]: item["bounds"] for item in source["retained_zones"]}
    expected = {}
    for mask, (row, pressure) in enumerate(zip(source["epi_states"], source["pressures"], strict=True)):
        added = tuple(F(source["timestep"]) * F(value) / geometry["grid_quantum"] for value in pressure)
        assert all(value.denominator == 1 for value in added)
        bounds = zones[row]
        for node in range(6):
            if -bounds[6][node] + added[node] < geometry["lower_grid"][node]:
                expected[(mask, node, "lower")] = int(geometry["lower_grid"][node] - added[node] - 1)
            if bounds[node][6] + added[node] > geometry["upper_grid"][node]:
                expected[(mask, node, "upper")] = -int(geometry["upper_grid"][node] - added[node] + 1)
    assert set(expected) == {(f["mask"], f["node"], f["direction"]) for f in facets}
    for facet, query in zip(facets, proof["queries"], strict=True):
        identity = facet["mask"], facet["node"], facet["direction"]
        target, = query["target_regions"]
        row = source["epi_states"][facet["mask"]]
        assert target["epi"] == row
        bounds = [list(values) for values in zones[row]]
        first, second = (facet["node"], 6) if facet["direction"] == "lower" else (6, facet["node"])
        bounds[first][second] = min(bounds[first][second], expected[identity])
        assert target["bounds"] == _dbm_close(bounds)


def test_region_report_excludes_24_whole_slabs_without_promoting_the_remaining_32(region_report):
    proof = region_report["B51_region_predecessors"]
    assert proof["intersections"] == proof["max_intersections"] == 250_000
    assert proof["common_grid_relaxes_coordinate_cosets"] is True
    for flag in ("conditional_invariance_certified", "conditional_boundedness_certified",
                 "future_runtime_certified", "asymptotic_convergence_certified"):
        assert proof[flag] is False
    facets, queries = region_report["B51_outgoing_facets"], proof["queries"]
    assert sum(query["status"] == "empty_complete_layer" for query in queries) == 24
    assert sum(query["status"] == "resource_limit" for query in queries) == 32
    expected_depths = {
        (6, 1): 2, (7, 1): 2, (14, 1): 2, (15, 1): 3, (22, 1): 3, (23, 1): 8,
        (30, 1): 2, (31, 1): 8, (38, 1): 2, (39, 1): 2, (46, 1): 3, (47, 1): 3,
        (49, 5): 6, (51, 5): 6, (53, 5): 6, (54, 1): 5, (55, 1): 6, (55, 5): 6,
        (57, 5): 6, (59, 5): 6, (61, 5): 6, (62, 1): 5, (63, 1): 8, (63, 5): 6,
    }
    for facet, query in zip(facets, queries, strict=True):
        assert query["actual_origin_reachability_certified"] is False
        assert all(not layer["origin_present"] for layer in query["iterations"])
        assert query["completed_depth"] == query["iterations"][-1]["depth"]
        assert query["origin_path_within_domain_excluded"] == (query["status"] == "empty_complete_layer")
        if query["status"] == "empty_complete_layer":
            assert facet["direction"] == "upper" and facet["node"] in (1, 5)
            assert query["past_exclusion_depth"] == expected_depths[(facet["mask"], facet["node"])]
            assert query["iterations"][-1]["zone_count"] == 0 and query["retained_zones"] == ()
        else:
            assert query["completed_depth"] in (33, 34)
            assert query["past_exclusion_depth"] is None and query["retained_zones"]
    assert {(item["node"], item["direction"]): (item["total"], item["excluded"])
            for item in region_report["B51_exit_groups"]} == {
                (0, "lower"): (8, 0), (1, "upper"): (16, 16), (2, "lower"): (8, 0),
                (3, "upper"): (8, 0), (4, "upper"): (8, 0), (5, "upper"): (8, 8),
            }
    assert region_report["excluded_first_exit_slabs"] == 24
    for flag in ("all_first_exit_slabs_excluded", "indefinite_trapping_certified",
                 "whole_band_exit_certified", "future_runtime_certified"):
        assert region_report[flag] is False
    assert region_report["new_conditional_trajectory_steps"] == region_report["new_live_graph_steps"] == 0
    json.dumps(campaign._payload(region_report), allow_nan=False)


def test_region_report_resource_interruption_preserves_complete_initial_regions(region_inputs, cached_forward):
    before = tuple(region_inputs)
    result = _analyze_regions(region_inputs, max_intersections=1)
    assert region_inputs == before
    proof = result["B51_region_predecessors"]
    assert proof["intersections"] == 1
    assert len(proof["queries"]) == len(result["B51_outgoing_facets"]) == 56
    first = proof["queries"][0]
    assert first["status"] == "resource_limit" and first["completed_depth"] == 0
    assert first["retained_zones"] == first["target_regions"]
    assert not result["all_first_exit_slabs_excluded"] and not result["indefinite_trapping_certified"]


@pytest.mark.parametrize("mutation", ("carry", "zone", "claim"))
def test_regions_reject_altered_B49_evidence_before_excluding_whole_slabs(region_inputs, cached_forward, mutation):
    data = json.loads(region_inputs[3])
    if mutation == "carry":
        state = data["B49_relational_forward_envelope"]["state"]
        state["remainder"][0] = str(F(state["remainder"][0]) + F(1, 2**3222))
    elif mutation == "zone":
        data["B49_relational_forward_envelope"]["retained_zones"][0]["bounds"][0][6] += 1
    else:
        data["manifest"]["claim_id"] = "O3.a-C6-carried-temporal-predecessors"
    altered = region_inputs[:3] + (json.dumps(data).encode(),)
    with pytest.raises(ValueError, match="retained B49|canonical reconstruction"):
        _analyze_regions(altered, max_intersections=1)


def _region_cli_args(monkeypatch, paths, output):
    _cli_args(monkeypatch, paths, output)
    monkeypatch.setattr(sys, "argv", sys.argv + [
        "--method", "regions", "--envelope-input", str(paths[3]),
    ])


def test_regions_cli_uses_four_inputs_and_a_distinct_regional_claim(
        tmp_path, monkeypatch, region_inputs, region_report):
    paths = tuple(tmp_path / f"region_input_{i}.json" for i in range(4))
    for path, raw in zip(paths, region_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "regions.json"
    captured = {}

    def analyze(parent, **kwargs):
        captured.update(parent=parent, **kwargs)
        return deepcopy(region_report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_region_exclusions", analyze)
    monkeypatch.setattr(campaign, "current_git_source_provenance",
                        lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    _region_cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-region-exclusions"
    assert result["manifest"]["dirty_source_hash"] == "sha256:" + "b" * 64
    assert tuple(item["sha256"] for item in result["input_evidence"]) == tuple(
        hashlib.sha256(raw).hexdigest() for raw in region_inputs
    )
    assert captured["envelope_bytes"] == region_inputs[3] and captured["max_intersections"] == 250_000
    assert "witness_bytes" not in captured
    assert result["excluded_first_exit_slabs"] == 24
    assert "B50_selected_point_pasts" not in result


@pytest.mark.parametrize("index", range(4))
def test_regions_cli_cannot_overwrite_any_of_its_four_inputs(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"region_input_{i}.json" for i in range(4))
    for path in paths:
        path.write_text("{}")
    _region_cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.parametrize("which", (0, 1, 2, 3, "source"))
def test_regions_cli_rejects_changed_source_or_any_retained_input(
        tmp_path, monkeypatch, region_inputs, region_report, which):
    paths = tuple(tmp_path / f"region_input_{i}.json" for i in range(4))
    for path, raw in zip(paths, region_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "regions.json"
    provenance = iter((("a" * 40, True, "sha256:" + "b" * 64),
                       ("a" * 40, True, "sha256:" + ("c" if which == "source" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(provenance))

    def analyze(*_args, **_kwargs):
        if which != "source":
            paths[which].write_text("{}")
        return deepcopy(region_report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_region_exclusions", analyze)
    _region_cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()


@pytest.fixture(scope="module")
def excursion_inputs(region_inputs, region_report):
    public = campaign._payload(region_report)
    public["manifest"] = {"claim_id": "O3.a-C6-carried-region-exclusions"}
    public["input_evidence"] = [dict(path=f"input_{i}.json", sha256=hashlib.sha256(raw).hexdigest())
                                for i, raw in enumerate(region_inputs)]
    return region_inputs + (json.dumps(public).encode(),)


def _analyze_excursion(raw_inputs, **kwargs):
    parent, relay, historical, envelope, region = raw_inputs
    return campaign.analyze_c6_winding_excursion_exclusion(
        json.loads(parent), relay_bytes=relay, historical_bytes=historical,
        envelope_bytes=envelope, region_bytes=region, **kwargs,
    )


@pytest.fixture(scope="module")
def excursion_report(excursion_inputs, forward_report):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "analyze_c6_winding_forward_envelope",
                      lambda *_a, **_k: deepcopy(forward_report))
        return _analyze_excursion(excursion_inputs)


def test_excursion_adds_eight_whole_slabs_from_the_same_primary_origin(excursion_report, excursion_inputs):
    result = excursion_report
    assert result["B52_reconstruction"]["prior_positive_queries_reverified"] == 24
    assert result["B52_reconstruction"]["prior_query_verification_intersections"] == 668
    assert result["B52_reconstruction"]["prior_region_sha256"] == hashlib.sha256(excursion_inputs[-1]).hexdigest()
    assert result["excluded_first_exit_slabs"] == 32 and result["additional_first_exit_slabs_excluded"] == 8
    assert result["new_conditional_trajectory_steps"] == 56 and result["new_live_graph_steps"] == 0
    assert result["primary_proof_origin_remains_B47"] is True
    assert campaign._payload(result["source"]["retained_B47_endpoint"]) == json.loads(
        excursion_inputs[0])["B47_conditional_first_exit"]["endpoint"]
    assert result["B52_excursion_exclusion"]["origin_path_within_domain_excluded"]
    assert {(g["node"], g["direction"]): g["excluded"] for g in result["B52_exit_groups"]} == {
        (0, "lower"): 0, (1, "upper"): 16, (2, "lower"): 8,
        (3, "upper"): 0, (4, "upper"): 0, (5, "upper"): 8,
    }
    for key in ("all_first_exit_slabs_excluded", "indefinite_trapping_certified",
                "whole_band_exit_certified", "future_runtime_certified"):
        assert result[key] is False
    json.dumps(campaign._payload(result), allow_nan=False)


def test_excursion_coordinate_is_the_primitive_nodal_poisson_contrast():
    weights = campaign._c6_poisson_contrast_weights()
    assert weights == (-2, 0, 2, 1, 0, -1) and sum(weights) == 0
    image = tuple(F(value) - F(weights[(i - 1) % 6] + weights[(i + 1) % 6], 2)
                  for i, value in enumerate(weights))
    assert image == (F(-3, 2), F(0), F(3, 2), F(0), F(0), F(0))


def test_excursion_short_resource_guard_does_not_promote_the_unfinished_prefix(excursion_inputs, cached_forward):
    result = _analyze_excursion(excursion_inputs, max_prefix_steps=1)
    assert result["new_conditional_trajectory_steps"] == 1
    assert result["excluded_first_exit_slabs"] == 24 and result["additional_first_exit_slabs_excluded"] == 0
    assert not result["B52_excursion_exclusion"]["origin_path_within_domain_excluded"]


@pytest.mark.parametrize("mutation", ("geometry", "carry", "target", "positive", "input_hash", "claim"))
def test_excursion_rejects_tampered_regional_evidence(excursion_inputs, cached_forward, mutation):
    prior = json.loads(excursion_inputs[-1])
    if mutation == "geometry":
        prior["B51_cube_geometry"]["upper_grid"][2] += 1
    elif mutation == "carry":
        prior["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "target":
        positive = next(q for q in prior["B51_region_predecessors"]["queries"] if q["origin_path_within_domain_excluded"])
        positive["target_regions"][0]["bounds"][0][6] += 1
    elif mutation == "positive":
        prior["excluded_first_exit_slabs"] += 1
    elif mutation == "input_hash":
        prior["input_evidence"][1]["sha256"] = "0" * 64
    else:
        prior["manifest"]["claim_id"] = "O3.a-C6-carried-forward-envelope"
    with pytest.raises(ValueError, match="regional|prior regional"):
        _analyze_excursion(excursion_inputs[:-1] + (json.dumps(prior).encode(),), max_prefix_steps=1)


def _excursion_cli_args(monkeypatch, paths, output):
    _cli_args(monkeypatch, paths, output)
    monkeypatch.setattr(sys, "argv", sys.argv + [
        "--method", "excursion", "--envelope-input", str(paths[3]), "--region-input", str(paths[4]),
    ])


def test_excursion_cli_binds_five_inputs_and_the_derived_prefix(
        tmp_path, monkeypatch, excursion_inputs, excursion_report):
    paths = tuple(tmp_path / f"excursion_{i}.json" for i in range(5))
    for path, raw in zip(paths, excursion_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "excursion.json"
    monkeypatch.setattr(campaign, "analyze_c6_winding_excursion_exclusion", lambda *_a, **_k: deepcopy(excursion_report))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    _excursion_cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-excursion-exclusion"
    assert tuple(x["sha256"] for x in result["input_evidence"]) == tuple(hashlib.sha256(raw).hexdigest() for raw in excursion_inputs)
    assert result["new_conditional_trajectory_steps"] == 56


@pytest.mark.parametrize("index", range(5))
def test_excursion_cli_cannot_overwrite_any_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"excursion_{i}.json" for i in range(5))
    for path in paths:
        path.write_text("{}")
    _excursion_cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.parametrize("which", (0, 1, 2, 3, 4, "source"))
def test_excursion_cli_rejects_changed_source_or_input(
        tmp_path, monkeypatch, excursion_inputs, excursion_report, which):
    paths = tuple(tmp_path / f"excursion_{i}.json" for i in range(5))
    for path, raw in zip(paths, excursion_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "excursion.json"
    provenance = iter((("a" * 40, True, "sha256:" + "b" * 64),
                       ("a" * 40, True, "sha256:" + ("c" if which == "source" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(provenance))

    def analyze(*_args, **_kwargs):
        if which != "source":
            paths[which].write_text("{}")
        return deepcopy(excursion_report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_excursion_exclusion", analyze)
    _excursion_cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()


@pytest.fixture(scope="module")
def mode_inputs(excursion_inputs, excursion_report):
    prior = campaign._payload(excursion_report)
    prior["manifest"] = {"claim_id": "O3.a-C6-carried-excursion-exclusion"}
    prior["source_scope"] = list(campaign.SOURCE_SCOPE)
    prior["input_evidence"] = [dict(path=f"mode_input_{i}.json", sha256=hashlib.sha256(raw).hexdigest())
                               for i, raw in enumerate(excursion_inputs)]
    candidates = (campaign.ROOT / "benchmarks/c6_winding_mode_excursion_candidates.json").read_bytes()
    return excursion_inputs + (json.dumps(prior).encode(), candidates)


@pytest.fixture(scope="module")
def mode_context(inputs, forward_report):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "analyze_c6_winding_forward_envelope", lambda *_a, **_k: deepcopy(forward_report))
        return campaign._rebuild_c6_outgoing_regions(
            json.loads(inputs[0]), relay_bytes=inputs[1], historical_bytes=inputs[2],
            envelope_bytes=(ARTIFACTS / "c6_winding_forward_envelope.json").read_bytes(),
        )


@pytest.fixture(scope="module")
def mode_closure_cache():
    # Every distinct complete domain and guard is still evaluated by its owner.
    # Repeated coefficient/admission tests need not repeat 265 identical layers.
    from tnfr.physics.c6_carried_viability import derive_c6_carried_reachable_envelope

    cache = {}

    def derive(reference, **kwargs):
        key = repr((reference, tuple(sorted(kwargs.items()))))
        if key not in cache:
            cache[key] = derive_c6_carried_reachable_envelope(reference, **kwargs)
        return cache[key]

    return derive


def _analyze_modes(raw_inputs, **kwargs):
    parent, relay, historical, envelope, region, excursion, candidates = raw_inputs
    return campaign.analyze_c6_winding_mode_excursions(
        json.loads(parent), parent_bytes=parent, relay_bytes=relay, historical_bytes=historical,
        envelope_bytes=envelope, region_bytes=region, excursion_bytes=excursion,
        candidates_bytes=candidates, **kwargs,
    )


@pytest.fixture
def cached_mode_context(monkeypatch, mode_context, mode_closure_cache):
    from tnfr.physics import c6_carried_viability

    monkeypatch.setattr(campaign, "_rebuild_c6_outgoing_regions", lambda *_a, **_k: mode_context)
    monkeypatch.setattr(c6_carried_viability, "derive_c6_carried_reachable_envelope", mode_closure_cache)


@pytest.fixture(scope="module")
def mode_report(mode_inputs, mode_context, mode_closure_cache):
    from tnfr.physics import c6_carried_viability

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "_rebuild_c6_outgoing_regions", lambda *_a, **_k: mode_context)
        patch.setattr(c6_carried_viability, "derive_c6_carried_reachable_envelope", mode_closure_cache)
        return _analyze_modes(mode_inputs)


def test_modes_reverify_all_priors_and_exclude_four_whole_regions(mode_report, mode_inputs):
    report = mode_report
    audit = report["B53_reconstruction"]
    assert audit["complete_envelope_rebuilt"] and audit["prior_B52_completely_rebuilt"]
    assert audit["prior_first_exit_slabs_reverified"] == 32
    assert audit["prior_excursion_prefix_steps_replayed"] == 56
    assert audit["prior_regional_verification"]["prior_query_verification_intersections"] == 668
    assert audit["prior_excursion_sha256"] == hashlib.sha256(mode_inputs[5]).hexdigest()
    assert audit["candidate_bytes_sha256"] == hashlib.sha256(mode_inputs[6]).hexdigest()
    assert report["excluded_first_exit_slabs"] == 36 and report["additional_first_exit_slabs_excluded"] == 4
    assert report["new_conditional_trajectory_steps"] == report["new_live_graph_steps"] == 0
    assert report["primary_proof_origin_remains_B47"]
    assert campaign._payload(report["source"]["retained_B47_endpoint"]) == json.loads(
        mode_inputs[0])["B47_conditional_first_exit"]["endpoint"]
    assert {(r["node"], r["direction"]): r["excluded"] for r in report["B53_exit_groups"]} == {
        (0, "lower"): 0, (1, "upper"): 16, (2, "lower"): 8,
        (3, "upper"): 4, (4, "upper"): 0, (5, "upper"): 8,
    }
    remaining = report["B53_remaining_first_exit_facets"]
    assert len(remaining) == 20
    assert [r["mask"] for r in remaining if r["node"] == 3] == [28, 30, 60, 62]
    for flag in ("all_first_exit_slabs_excluded", "indefinite_trapping_certified",
                 "whole_band_exit_certified", "future_runtime_certified", "asymptotic_convergence_certified"):
        assert report[flag] is False
    json.dumps(campaign._payload(report), allow_nan=False)


def test_modes_use_complete_origin_injected_hulls_and_exact_per_target_bounds(mode_report):
    before = mode_report["B53_origin_injected_reachable_envelope"]
    after = mode_report["B53_post_exclusion_reachable_envelope"]
    assert before["status"] == after["status"] == "fixed_point"
    assert before["iterations"][-1]["ordinal"] == 265 and before["intersections"] == 232802
    assert after["iterations"][-1]["ordinal"] == 1 and after["intersections"] == 830
    for envelope in (before, after):
        assert all(layer["origin_retained"] for layer in envelope["iterations"])
        assert envelope["domain_confined_paths_covered"] and envelope["clipped_forward_inclusion_certified"]
        assert not envelope["conditional_boundedness_certified"]
    records = mode_report["B53_mode_excursion_proofs"]
    assert [record["target"]["mask"] for record in records] == [29, 31, 61, 63]
    for record in records:
        assert record["normalization_scale"] > 0
        proof = record["proof"]
        assert proof["status"] == "initially_above_targets"
        assert proof["origin_path_within_domain_excluded"]
        assert proof["initial_potential"] > proof["target_upper"]
        assert proof["ingress_lower"] > proof["target_upper"]
        assert proof["minimum_drift"] > 0
        assert len(proof["active_transitions"]) == 242 and len(proof["ingress"]) == 189
        assert proof["prefix_deadline"] == 0 and proof["observed_prefix_steps"] == 0
        assert all(type(value) is int for value in proof["weights"])


@pytest.mark.parametrize("position", (3, 4, 5))
def test_modes_reject_cross_lineage_input_hashes(mode_inputs, position):
    blobs = list(mode_inputs)
    data = json.loads(blobs[position])
    data["input_evidence"][0]["sha256"] = "0" * 64
    blobs[position] = json.dumps(data).encode()
    with pytest.raises(ValueError, match="lineage"):
        _analyze_modes(blobs)


def test_modes_reject_parent_objects_not_matching_the_retained_bytes(mode_inputs):
    parent = json.loads(mode_inputs[0])
    parent["B47_conditional_first_exit"]["endpoint"]["remainder"][0] = "0"
    with pytest.raises(ValueError, match="unchanged retained B47"):
        campaign.analyze_c6_winding_mode_excursions(
            parent, parent_bytes=mode_inputs[0], relay_bytes=mode_inputs[1], historical_bytes=mode_inputs[2],
            envelope_bytes=mode_inputs[3], region_bytes=mode_inputs[4], excursion_bytes=mode_inputs[5],
            candidates_bytes=mode_inputs[6],
        )


def test_mode_integer_normalization_uses_one_positive_scale_for_all_coordinates(mode_inputs, mode_context):
    fresh, _profile, _state, _zones, _geometry, facets, _targets = mode_context
    candidates = campaign._c6_mode_candidates(
        mode_inputs[-1], facets, fresh["B49_relational_forward_envelope"]["epi_states"],
    )
    for original, (_index, _active, weights, offsets, scale) in zip(
            json.loads(mode_inputs[-1])["candidates"], candidates, strict=True):
        assert scale > 0
        assert weights + offsets == tuple(F(value) * scale for value in original["weights"] + original["offsets"])


@pytest.mark.parametrize("mutation", ("carry", "positive_count", "target", "claim"))
def test_modes_reconstruct_b52_instead_of_trusting_its_claims(mode_inputs, cached_mode_context, mutation):
    prior = json.loads(mode_inputs[5])
    if mutation == "carry":
        prior["source"]["retained_B47_endpoint"]["remainder"][0] = "0"
    elif mutation == "positive_count":
        prior["excluded_first_exit_slabs"] += 1
    elif mutation == "target":
        prior["B52_excursion_exclusion"]["target_regions"][0]["bounds"][0][6] += 1
    else:
        prior["manifest"]["claim_id"] = "O3.a-C6-carried-region-exclusions"
    altered = mode_inputs[:5] + (json.dumps(prior).encode(), mode_inputs[6])
    with pytest.raises(ValueError, match="B52"):
        _analyze_modes(altered)


def test_modes_changed_candidate_coordinates_cannot_inherit_success(mode_inputs, cached_mode_context):
    data = json.loads(mode_inputs[6])
    data["candidates"] = data["candidates"][:1]
    candidate = data["candidates"][0]
    candidate["weights"] = ["0"] * 6
    candidate["offsets"] = ["0"] * len(candidate["active_masks"])
    candidate["origin_path_within_domain_excluded"] = True
    result = _analyze_modes(mode_inputs[:6] + (json.dumps(data).encode(),))
    assert result["excluded_first_exit_slabs"] == 32
    assert result["additional_first_exit_slabs_excluded"] == 0
    assert result["B53_mode_excursion_proofs"][0]["status"] == "nonpositive_drift"
    assert not result["indefinite_trapping_certified"]


def test_modes_resource_guard_retains_only_complete_outer_domains(mode_inputs, cached_mode_context):
    result = _analyze_modes(mode_inputs, max_intersections=1, max_prefix_steps=1)
    for name in ("B53_origin_injected_reachable_envelope", "B53_post_exclusion_reachable_envelope"):
        proof = result[name]
        assert proof["status"] == "resource_limit" and proof["intersections"] == 1
        assert len(proof["iterations"]) == 1
        assert proof["domain_zones"] == proof["retained_zones"]
    assert result["excluded_first_exit_slabs"] <= 36
    assert not result["indefinite_trapping_certified"]


@pytest.mark.parametrize("mutation", ("schema", "duplicate", "float", "offset_count", "mask", "facet", "zero_denominator", "object"))
def test_mode_candidate_shape_is_strict(mode_inputs, mode_context, mutation):
    data = json.loads(mode_inputs[-1])
    candidate = data["candidates"][0]
    if mutation == "schema":
        data["schema"] = "unknown"
    elif mutation == "duplicate":
        data["candidates"].append(deepcopy(candidate))
    elif mutation == "float":
        candidate["weights"][0] = 0.5
    elif mutation == "offset_count":
        candidate["offsets"].pop()
    elif mutation == "mask":
        candidate["active_masks"][0] = True
    elif mutation == "facet":
        candidate["target"]["node"] = 99
    elif mutation == "zero_denominator":
        candidate["weights"][0] = "1/0"
    else:
        data["candidates"][0] = None
    fresh, _profile, _state, _zones, _geometry, facets, _targets = mode_context
    with pytest.raises(ValueError, match="mode|Mode"):
        campaign._c6_mode_candidates(json.dumps(data).encode(), facets, fresh["B49_relational_forward_envelope"]["epi_states"])


def _mode_cli_args(monkeypatch, paths, output):
    _cli_args(monkeypatch, paths, output)
    monkeypatch.setattr(sys, "argv", sys.argv + [
        "--method", "modes", "--envelope-input", str(paths[3]), "--region-input", str(paths[4]),
        "--excursion-input", str(paths[5]), "--mode-candidates-input", str(paths[6]),
    ])


def test_modes_cli_binds_seven_inputs_without_promoting_clipped_closure(tmp_path, monkeypatch, mode_inputs, mode_report):
    paths = tuple(tmp_path / f"mode_{i}.json" for i in range(7))
    for path, raw in zip(paths, mode_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "modes.json"
    monkeypatch.setattr(campaign, "analyze_c6_winding_mode_excursions", lambda *_a, **_k: deepcopy(mode_report))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    _mode_cli_args(monkeypatch, paths, output)
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-carried-mode-excursions"
    assert tuple(x["sha256"] for x in result["input_evidence"]) == tuple(hashlib.sha256(raw).hexdigest() for raw in mode_inputs)
    assert result["new_conditional_trajectory_steps"] == 0 and not result["indefinite_trapping_certified"]


@pytest.mark.parametrize("index", range(7))
def test_modes_cli_cannot_overwrite_any_input(tmp_path, monkeypatch, index):
    paths = tuple(tmp_path / f"mode_{i}.json" for i in range(7))
    for path in paths:
        path.write_text("{}")
    _mode_cli_args(monkeypatch, paths, paths[index])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert all(path.read_text() == "{}" for path in paths)


@pytest.mark.parametrize("which", (0, 1, 2, 3, 4, 5, 6, "source"))
def test_modes_cli_rejects_changed_source_or_any_input(tmp_path, monkeypatch, mode_inputs, mode_report, which):
    paths = tuple(tmp_path / f"mode_{i}.json" for i in range(7))
    for path, raw in zip(paths, mode_inputs, strict=True):
        path.write_bytes(raw)
    output = tmp_path / "modes.json"
    provenance = iter((("a" * 40, True, "sha256:" + "b" * 64),
                       ("a" * 40, True, "sha256:" + ("c" if which == "source" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(provenance))

    def analyze(*_args, **_kwargs):
        if which != "source":
            paths[which].write_text("{}")
        return deepcopy(mode_report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_mode_excursions", analyze)
    _mode_cli_args(monkeypatch, paths, output)
    with pytest.raises(RuntimeError, match="source or retained input changed"):
        campaign.main()
    assert not output.exists()

"""One native composite step, with portable retained inputs and exact ledgers."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from unittest.mock import patch

import pytest

from benchmarks.thol_pressure_feedback import _payload
from tests.physics.test_forced_epi_closure import _apply
from tests.physics.test_forced_epi_prediction import _assert_forecast
from tests.physics.test_thol_full_state_response import _metric_readout, _prediction, _reference
from tnfr.research.core_manifests import CoreExperimentManifest


F = Fraction


def _write(path, payload):
    raw = (json.dumps(_payload(payload), allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _epi(record):
    return tuple(map(F, record["state"]["epi"]))


def _rate(reference, epi):
    """Direct conductance sum, independent of the benchmark matrix helpers."""
    source = reference["source"]
    strengths = [F(0)]*len(epi)
    differences = [F(0)]*len(epi)
    for i, j, weight in source["conductance"]:
        strengths[i] += weight
        differences[i] += weight*(epi[j]-epi[i])
    return tuple(nu*(reference["epi_weight"]*gradient/d+force)
                 for nu, gradient, d, force in zip(
                     source["capacity"], differences, strengths, reference["forcing"], strict=True))


@pytest.fixture(scope="module")
def execution(tmp_path_factory):
    import benchmarks.thol_full_state_response as prior
    import benchmarks.thol_native_runtime_response as benchmark

    # Exactly two real preparations. The portable envelope tests admission;
    # independent historical replay is separately exercised by the official CLI.
    sources = {name: prior.replay_response_branch(name) for name in prior.BRANCHES}
    with patch.object(prior, "run_response_branch", side_effect=lambda name: sources[name][1]):
        retained_study = prior.run_study()
    path = tmp_path_factory.mktemp("native_response") / "prior.json"
    with patch.object(prior, "run_study", return_value=retained_study), patch(
        "sys.argv", ["thol_full_state_response", "--output", str(path)],
    ):
        prior.main()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    predictions, chronology = [], []
    predictor, step = benchmark.predict_forced_support_realization_euler, benchmark.dynamics.step

    def predict(*args, **kwargs):
        value = predictor(*args, **kwargs)
        predictions.append(value)
        chronology.append(("forecast", len(predictions)-1))
        return value

    def native(graph, **kwargs):
        assert chronology[-1] == ("forecast", len(predictions)-1)
        assert predictions[-1].frames[0].epi == tuple(map(F, benchmark._state(graph)["epi"]))
        assert graph.graph["_t"] == 1.5
        assert kwargs == {"dt": 0.25, "use_Si": True, "apply_glyphs": True}
        chronology.append(("native", len(predictions)-1))
        return step(graph, **kwargs)

    def replay(name):
        chronology.append(("replay", prior.BRANCHES.index(name)))
        return sources[name]

    with patch.object(benchmark, "replay_response_branch", side_effect=replay), patch.object(
        benchmark, "predict_forced_support_realization_euler", predict,
    ), patch.object(benchmark.dynamics, "step", native):
        study = benchmark.run_study(path, expected_prior_sha256=digest)
    return {"study": study, "sources": sources, "path": path, "digest": digest,
            "prior": json.loads(path.read_bytes()), "predictions": predictions, "chronology": chronology}


@pytest.fixture(scope="module")
def study(execution):
    return execution["study"]


def test_complete_prior_payloads_live_endpoints_and_fixed_native_call_budget(execution):
    study = execution["study"]
    assert execution["chronology"] == [
        ("replay", 0), ("replay", 1), ("forecast", 0), ("native", 0), ("forecast", 1), ("native", 1),
    ]
    assert study["scientific_prior_replays_equal"] and study["complete_common_source_equal"]
    assert all(row["scientific_payload_equal"] for row in study["prior_replay_checks"])
    assert study["prior_evidence"]["sha256"] == execution["digest"]
    assert study["prior_evidence"]["historical_manifest"] == execution["prior"]["manifest"]
    assert study["protocol"]["native_calls_per_branch"] == 1
    assert study["protocol"]["dt"] == F(1, 4)
    assert study["protocol"]["fallback"] is None
    for branch, previous, archived in zip(study["branches"], study["replayed_prior_reports"],
                                          execution["prior"]["branches"], strict=True):
        assert _payload(previous) == archived
        assert branch["before"]["state"] == previous["endpoint"]
        assert branch["native_trace"]["native_calls"] == 1
        assert branch["status"] == "executed"
        assert branch["endpoint"]["state"]["time"] == 1.75
    assert not study["autonomous_maintenance_certified"]


@pytest.mark.parametrize("index", (0, 1))
def test_original_full_coordinate_counterfactual_is_frozen_before_the_native_step(study, index):
    branch, prior = study["branches"][index], study["replayed_prior_reports"][index]
    frozen = branch["held_prediction"]
    prediction = _prediction(frozen["payload"])
    reference = _reference(prior["original_reference"])
    _assert_forecast(prediction, reference, prior["lineage"]["parent_children"], (F(1, 4),), _epi(branch["before"]))
    assert frozen["time"] == 1.5 and frozen["frozen_before_native_step"]
    digest = hashlib.sha256(json.dumps(_payload(frozen["payload"]), sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert frozen["sha256"] == digest
    assert frozen["payload"]["realization"]["closure"]["reference"] == prior["original_reference"]
    assert branch["before"]["state"]["nodes"] == branch["endpoint"]["state"]["nodes"]


@pytest.mark.parametrize("index", (0, 1))
def test_native_stage_chronology_retains_real_selector_gates_history_and_outer_owners(study, index):
    branch = study["branches"][index]
    trace = branch["native_trace"]
    rows = trace["boundaries"]
    names = [row["boundary"] for row in rows]
    required = ("_run_before_callbacks", "_prepare_dnfr", "_refresh_delta_nfr", "compute_Si",
                "_apply_selector", "_apply_glyphs", "integrate", "coordinate_global_local_phase",
                "adapt_vf_after_structural_stability", "_advance_math_engine", "_update_epi_hist",
                "_maybe_remesh", "_run_validators", "_run_after_callbacks", "publish_graph_cache_metrics")
    assert [names.index(name) for name in required] == sorted(names.index(name) for name in required)
    assert all(names.count(name) == 1 for name in required)
    assert names.count("_record_mutation_flow_boundary") == 3
    assert all(row["outcome"] == "completed" for row in rows)
    prepared = next(row for row in rows if row["boundary"] == "_prepare_dnfr")
    glyph_stage = next(row for row in rows if row["boundary"] == "_apply_glyphs")
    assert prepared["after"]["state"] == glyph_stage["before"]["state"]
    assert prepared["after"]["stored_Si"] == glyph_stage["before"]["stored_Si"]
    assert all(value is not None for value in glyph_stage["before"]["stored_Si"])
    requests, calls = trace["selector_requests"], trace["actual_glyph_calls"]
    nodes = branch["before"]["state"]["nodes"]
    assert tuple(row["node"] for row in requests) == nodes
    assert tuple(row["node"] for row in calls) == nodes
    for request in requests:
        row = rows[request["trace_ordinal"]]
        assert row["boundary"] == "_resolve_preselected_glyph"
        assert row["requested_glyph"] == request["glyph"] and row["node"] == request["node"]
    for call in calls:
        row = rows[call["ordinal"]]
        assert row["boundary"] == "apply_glyph" and row["glyph"] == call["glyph"]
        assert row["outcome"] == call["outcome"] == "completed"
        node = call["node"]
        assert row["after"]["state"]["glyph_history"][node] == (
            row["before"]["state"]["glyph_history"][node] + (call["glyph"],))
        assert row["before"]["state"]["physical_epi_history"][node]
    integration = next(row for row in rows if row["boundary"] == "integrate")
    assert integration["before"]["state"]["time"] == 1.5
    assert integration["after"]["state"]["time"] == 1.75
    assert integration["effective_arguments"]["dt"] == ("binary64", float(0.25).hex())


@pytest.mark.parametrize("index", (0, 1))
def test_exact_changed_model_ledger_separates_actions_drives_integration_and_later_writes(study, index):
    branch, prior = study["branches"][index], study["replayed_prior_reports"][index]
    trace, ledger = branch["native_trace"], branch["runtime_ledger"]
    assert ledger["available"]
    row = next(row for row in trace["boundaries"] if row["boundary"] == "integrate")
    x0, xg, xi, xf = map(_epi, (branch["before"], row["before"], row["after"], branch["endpoint"]))
    h, reference = F(1, 4), prior["original_reference"]
    jump = tuple(b-a for a, b in zip(x0, xg, strict=True))
    old_rate = _rate(reference, xg)
    state = row["before"]["state"]
    stored_rate = tuple(F(nu)*F(p) for nu, p in zip(state["capacity"], state["pressure"], strict=True))
    matrix = branch["held_prediction"]["payload"]["fine_euler_matrices"][0]
    propagated = _apply(matrix, jump)
    changed = tuple(h*(a-b) for a, b in zip(stored_rate, old_rate, strict=True))
    remainder = tuple(y-x-h*r for y, x, r in zip(xi, xg, stored_rate, strict=True))
    post = tuple(a-b for a, b in zip(xf, xi, strict=True))
    ideal = branch["held_prediction"]["payload"]["frames"][-1]["epi"]
    actual = tuple(a-b for a, b in zip(xf, ideal, strict=True))
    assert ledger["preintegration_jump"] == jump
    assert ledger["transported_jump"] == propagated
    assert ledger["old_model_drive_at_integration_entry"] == old_rate
    assert ledger["actual_stored_nodal_rate"] == stored_rate
    assert ledger["changed_consumed_drive"] == changed
    assert ledger["integration_remainder"] == remainder
    assert ledger["postintegration_epi_change"] == post
    assert ledger["actual_minus_held_forecast"] == actual
    assert actual == tuple(sum(parts) for parts in zip(propagated, changed, remainder, post, strict=True))
    capture = trace["captures"]["integrator_entry"]
    assert capture["available"] and ledger["refined_drive"]["available"]
    observation = capture["payload"]["observation"]
    snapshot = observation["snapshot"]
    current_rate = _rate({"source": snapshot, "epi_weight": observation["epi_weight"],
                          "forcing": observation["forcing"]}, xg)
    kernel_rate = tuple(nu*p for nu, p in zip(snapshot["capacity"], observation["full_kernel_pressure"], strict=True))
    refinement = ledger["refined_drive"]
    assert refinement["current_reference_drive"] == current_rate
    for key, left, right in (("model_change", current_rate, old_rate),
                             ("stored_minus_fresh_kernel", stored_rate, kernel_rate),
                             ("fresh_kernel_minus_exact_reference", kernel_rate, current_rate)):
        assert refinement[key] == tuple(h*(a-b) for a, b in zip(left, right, strict=True))
    assert not any(ledger["identity_residual"])
    assert not any(refinement["identity_residual"])
    assert "not identified as rounding" in ledger["scope"]


def test_original_target_and_paired_metric_are_not_refitted_after_coefficient_changes(study):
    for branch, prior in zip(study["branches"], study["replayed_prior_reports"], strict=True):
        reference = prior["original_reference"]
        metric, z = reference["metric_weights"], reference["relative_profile"]
        for field, record in (("old_target_before", branch["before"]), ("old_target_endpoint", branch["endpoint"])):
            observed = branch[field]
            assert observed["available"]
            epi = _epi(record)
            mean = _metric_readout(epi, metric)[0]
            error = tuple(value-mean-profile for value, profile in zip(epi, z, strict=True))
            assert observed["payload"]["epi"] == epi
            assert observed["payload"]["mean"] == mean
            assert observed["payload"]["relative_error"] == error
            assert observed["payload"]["error_variance"] == sum(h*u*u for h, u in zip(metric, error, strict=True))/2
        comparison = branch["endpoint_coefficient_comparison"]
        for key in ("nodes", "capacity", "phase", "edges"):
            assert comparison[key] == (branch["before"]["state"][key] == branch["endpoint"]["state"][key])
    h = study["replayed_prior_reports"][0]["original_reference"]["metric_weights"]
    for key in ("before", "endpoint"):
        x, y = (_epi(branch[key]) for branch in study["branches"])
        delta = tuple(b-a for a, b in zip(x, y, strict=True))
        mean, centered, full, energy = _metric_readout(delta, h)
        row = study["paired_original_metric_response"][key]
        assert row["epi_difference"] == delta and row["weighted_mean_offset"] == mean
        assert row["centered_epi_difference"] == centered
        assert row["centered_H_energy"] == energy and row["full_H_energy"] == full


def test_changed_node_space_abstains_from_old_target_readout(execution):
    from benchmarks.thol_native_runtime_response import _pattern

    reference = _reference(execution["study"]["replayed_prior_reports"][0]["original_reference"])
    record = deepcopy(execution["study"]["branches"][0]["endpoint"])
    record["state"]["nodes"] = tuple(reversed(record["state"]["nodes"]))
    assert _pattern(reference, record) == {"available": False, "reason": "Ordered node space changed"}


@pytest.mark.parametrize("digest", (None, True, "0"*63, "A"*64, "z"*64, "0"*64))
def test_prior_evidence_rejects_bad_or_incorrect_digest(execution, digest):
    from benchmarks.thol_native_runtime_response import load_prior_evidence

    with pytest.raises(ValueError):
        load_prior_evidence(execution["path"], expected_sha256=digest)


@pytest.mark.parametrize("mutation", ("claim", "branch", "manifest"))
def test_prior_admission_rejects_wrong_claim_branch_or_invalid_manifest(execution, tmp_path, mutation):
    from benchmarks.thol_native_runtime_response import load_prior_evidence

    payload = deepcopy(execution["prior"])
    if mutation == "claim":
        payload["manifest"]["claim_id"] = "unrelated"
    elif mutation == "branch":
        payload["branches"].reverse()
    else:
        payload["manifest"]["git_sha"] = "invalid"
    path = tmp_path / "bad.json"
    digest = _write(path, payload)
    with pytest.raises(ValueError):
        load_prior_evidence(path, expected_sha256=digest)


def test_payload_mismatch_refuses_before_any_native_call(execution, tmp_path, monkeypatch):
    import benchmarks.thol_native_runtime_response as benchmark

    payload = deepcopy(execution["prior"])
    payload["branches"][0]["endpoint"]["epi"][0] += 0.01
    path = tmp_path / "changed.json"
    digest = _write(path, payload)
    monkeypatch.setattr(benchmark, "replay_response_branch", lambda name: execution["sources"][name])
    monkeypatch.setattr(benchmark, "run_native_branch", lambda *args: pytest.fail("mismatched source executed"))
    with pytest.raises(ValueError, match="complete scientific replay differs"):
        benchmark.run_study(path, expected_prior_sha256=digest)


def test_only_consistent_child_invocation_utc_labels_may_differ(execution):
    from benchmarks.thol_native_runtime_response import _admit_replay

    prior = execution["study"]["replayed_prior_reports"][1]
    retained = deepcopy(execution["prior"]["branches"][1])
    child = prior["lineage"]["children"][0]
    attrs = dict(retained["event"]["after_node_attributes"])[child]
    for key in ("emission_timestamp", "_emission_origin"):
        attrs[key] = "2001-02-03T04:05:06+00:00"
    attrs["_structural_lineage"]["origin"] = "2001-02-03T04:05:06+00:00"
    before = deepcopy(retained)
    result = _admit_replay(prior, retained)
    assert result["scientific_payload_equal"] and not result["full_payload_equal"]
    assert retained == before
    assert any(row["utc_origin"] == "2001-02-03T04:05:06+00:00" for row in result["utc_origin_records"])
    assert len(result["exempt_paths"]) == 3


@pytest.mark.parametrize("mutation", ("epi", "physical_time", "history", "activation_count",
                                      "nonchild_timestamp", "inconsistent_alias", "non_utc"))
def test_replay_metadata_exception_does_not_hide_scientific_or_lifecycle_changes(execution, mutation):
    from benchmarks.thol_native_runtime_response import _admit_replay

    prior = execution["study"]["replayed_prior_reports"][1]
    retained = deepcopy(execution["prior"]["branches"][1])
    attrs = dict(retained["event"]["after_node_attributes"])
    child, parent = prior["lineage"]["children"][0], prior["lineage"]["parents"][0]
    if mutation == "epi":
        retained["event"]["after"]["epi"][0] += 0.01
    elif mutation == "physical_time":
        retained["event"]["after"]["time"] += 0.25
    elif mutation == "history":
        retained["event"]["after"]["physical_epi_history"][str(child)].append([99.0, 0.5])
    elif mutation == "activation_count":
        attrs[child]["_structural_lineage"]["activation_count"] += 1
    elif mutation == "nonchild_timestamp":
        attrs[parent]["emission_timestamp"] = "2001-02-03T04:05:06+00:00"
    elif mutation == "inconsistent_alias":
        attrs[child]["_emission_origin"] = "2001-02-03T04:05:06+00:00"
    else:
        value = "2001-02-03T04:05:06+01:00"
        attrs[child]["emission_timestamp"] = attrs[child]["_emission_origin"] = value
        attrs[child]["_structural_lineage"]["origin"] = value
    with pytest.raises(ValueError):
        _admit_replay(prior, retained)


@pytest.mark.parametrize("field", ("nodes", "epi", "capacity", "stored_pressure", "phase"))
def test_refined_ledger_requires_capture_identity_not_only_a_cancelling_sum(execution, field):
    from benchmarks.thol_native_runtime_response import _runtime_ledger

    branch = execution["study"]["branches"][0]
    trace = deepcopy(branch["native_trace"])
    observation = trace["captures"]["integrator_entry"]["payload"]["observation"]
    container = observation if field == "phase" else observation["snapshot"]
    values = list(container[field])
    values[0] = "forged-node" if field == "nodes" else values[0]+F(1, 7)
    container[field] = tuple(values)
    with pytest.raises(ValueError, match="does not bind the actual integrator entry"):
        _runtime_ledger(execution["predictions"][0], branch["before"], branch["endpoint"], trace)


@pytest.mark.parametrize("error_type", ("known", "unexpected"))
def test_native_refusal_is_one_attempt_with_partial_trace_and_no_claimed_rollback(monkeypatch, error_type):
    import benchmarks.thol_native_runtime_response as benchmark
    from benchmarks.thol_birth_transport import _birth_graph
    from tnfr.operators.preconditions import OperatorPreconditionError

    graph = _birth_graph("attached")
    original_time = graph.graph.get("_t", 0.0)
    history_before = tuple(tuple(data.get("epi_time_history", ())) for _, data in graph.nodes(data=True))
    called = []

    def fail(graph, node, glyph, **kwargs):
        called.append(node)
        if error_type == "known":
            raise OperatorPreconditionError("Synthetic", "Explicit bounded refusal control")
        raise RuntimeError("Unexpected execution defect")

    monkeypatch.setattr(benchmark.selectors, "apply_glyph", fail)
    if error_type == "unexpected":
        with pytest.raises(RuntimeError, match="Unexpected execution defect"):
            benchmark._trace_step(graph)
    else:
        result = benchmark._trace_step(graph)
        assert result["status"] == "refused" and result["native_calls"] == 1
        assert result["failure"]["type"] == "OperatorPreconditionError"
        names = [row["boundary"] for row in result["boundaries"]]
        assert "compute_Si" in names and "apply_glyph" in names
        assert "integrate" not in names and "coordinate_global_local_phase" not in names
        assert result["actual_glyph_calls"][0]["outcome"] == "raised"
        assert "no simultaneous-stage claim or whole-step rollback" in result["scope"]
    assert len(called) == 1 and graph.graph.get("_t", 0.0) == original_time
    history_after = tuple(tuple(data.get("epi_time_history", ())) for _, data in graph.nodes(data=True))
    assert history_after != history_before
    assert benchmark.selectors.apply_glyph is fail


def test_cli_serializes_complete_trace_and_refuses_overwriting_prior_evidence(execution, tmp_path, monkeypatch):
    import benchmarks.thol_native_runtime_response as benchmark

    output = tmp_path / "native.json"
    monkeypatch.setattr(benchmark, "run_study", lambda *args, **kwargs: execution["study"])
    monkeypatch.setattr("sys.argv", ["native", "--prior-input", str(execution["path"]),
                                     "--expected-prior-sha256", execution["digest"], "--output", str(output)])
    benchmark.main()
    report = json.loads(output.read_bytes())
    CoreExperimentManifest(**report["manifest"]).validate_for_admission()
    assert report["experimental_status"] == "No empirical correspondence tested"
    assert not report["autonomous_maintenance_certified"]
    assert all(branch["native_trace"]["boundaries"] for branch in report["branches"])
    before = execution["path"].read_bytes()
    monkeypatch.setattr("sys.argv", ["native", "--prior-input", str(execution["path"]),
                                     "--output", str(execution["path"])])
    with pytest.raises(ValueError, match="must not overwrite"):
        benchmark.main()
    assert execution["path"].read_bytes() == before

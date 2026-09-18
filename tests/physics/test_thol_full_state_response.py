"""Full-coordinate matched-response forecasts and exact runtime defects."""

from copy import deepcopy
from collections import deque
from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import pytest

from benchmarks.thol_pressure_feedback import _payload
from tests.physics.test_forced_epi_closure import _apply
from tests.physics.test_forced_epi_prediction import _assert_forecast
from tests.physics.test_thol_distributed_target import (
    _assert_step, _assert_target, _dot, _energy, _laplacian,
)
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.support_transport import SupportTransportSnapshot
from tnfr.research.core_manifests import CoreExperimentManifest


F = Fraction


def _reference(raw):
    return derive_forced_support_balance(
        SupportTransportSnapshot(**raw["source"]), epi_weight=raw["epi_weight"], forcing=raw["forcing"],
    )


def _prediction(raw):
    return SimpleNamespace(**{
        **raw,
        "realization": SimpleNamespace(**raw["realization"]),
        "frames": tuple(SimpleNamespace(**frame) for frame in raw["frames"]),
    })


def _metric_readout(vector, metric):
    mass = sum(metric, F(0))
    mean = sum(h*x for h, x in zip(metric, vector, strict=True))/mass
    centered = tuple(x-mean for x in vector)
    full = sum(h*x*x for h, x in zip(metric, vector, strict=True))/2
    energy = sum(h*x*x for h, x in zip(metric, centered, strict=True))/2
    assert full == energy + mass*mean*mean/2
    return mean, centered, full, energy


def _admission_graph():
    graph = nx.Graph([(0, 1)])
    for node in graph:
        graph.nodes[node].update(EPI=0.25, nu_f=0.95, theta=0.0, delta_nfr=0.0,
                                 glyph_history=["IL", "OZ", "THOL"])
    graph.nodes[0]["sub_nodes"] = [1]
    graph.nodes[1]["parent_node"] = 0
    graph.graph.update(hierarchy={0: [1]}, _node_sample=(0, 1), _t=1.0)
    return graph


@pytest.fixture(scope="module")
def execution():
    import benchmarks.thol_full_state_response as benchmark
    import tnfr.physics.epi_memory as memory

    prepare = benchmark.prepare_distributed_transport_support
    flow = benchmark._physical_flow
    forecast = memory.predict_forced_support_realization_euler
    graphs, chronology, predictions = [], [], []

    def fresh(*args, **kwargs):
        graph, prefix = prepare(*args, **kwargs)
        graphs.append(graph)
        return graph, prefix

    def predict(*args, **kwargs):
        result = forecast(*args, **kwargs)
        predictions.append(result)
        chronology.append((len(graphs)-1, "prediction", graphs[-1].graph["_t"]))
        return result

    def physical(graph, *args, **kwargs):
        time = graph.graph["_t"]
        chronology.append((len(graphs)-1, "flow", time))
        if time == 1.0:
            assert len(predictions) == len(graphs)
            assert predictions[-1].frames[0].epi == tuple(
                F(value) for value in benchmark._state(graph)["epi"]
            )
        return flow(graph, *args, **kwargs)

    with patch.object(benchmark, "prepare_distributed_transport_support", fresh), patch.object(
        benchmark, "_physical_flow", physical,
    ), patch.object(memory, "predict_forced_support_realization_euler", predict):
        study = benchmark.run_study()
    return study, graphs, chronology, predictions


@pytest.fixture(scope="module")
def study(execution):
    return execution[0]


def test_fresh_complete_sources_match_and_each_prediction_precedes_its_native_continuation(execution):
    import benchmarks.thol_full_state_response as benchmark

    study, graphs, chronology, _ = execution
    assert len(graphs) == 2 and graphs[0] is not graphs[1]
    assert chronology == [
        (0, "flow", 0.5), (0, "prediction", 1.0), (0, "flow", 1.0),
        (1, "flow", 0.5), (1, "prediction", 1.0), (1, "flow", 1.0),
    ]
    control, changed = study["branches"]
    assert tuple(branch["branch"] for branch in study["branches"]) == ("control", "child_emission")
    assert study["complete_common_source_equal"]
    for field in benchmark.COMMON_FIELDS:
        assert control[field] == changed[field]
    assert control["before_event"]["time"] == 1.0
    assert len(control["before_event"]["nodes"]) == 16
    assert len(control["lineage"]["children"]) == 8
    assert control["realization"]["dimension"] == control["realization"]["full_state_dimension"] == 16
    assert study["protocol"]["fallback_operator"] is None
    assert not study["coefficient_tuning"] and not study["target_rewrite"]


@pytest.mark.parametrize("index", (0, 1))
def test_frozen_full_coordinate_prediction_matches_exact_fine_model_and_hash(study, index):
    branch = study["branches"][index]
    original = _reference(branch["original_reference"])
    frozen = branch["frozen_prediction"]
    prediction = _prediction(frozen["payload"])
    initial = branch["post_event_target"]["state"]["snapshot"]["epi"]
    _assert_forecast(prediction, original, branch["lineage"]["parent_children"], (F(1, 4),)*2, initial)
    assert frozen["frozen_before_continuation"] and frozen["physical_time_at_freeze"] == 1.0
    digest = hashlib.sha256(json.dumps(_payload(frozen["payload"]), sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert frozen["sha256"] == digest
    for name in ("observation", "right_inverse", "reduced_generator", "output_map", "reduced_source"):
        assert frozen["payload"]["realization"][name] == branch["realization"][name]
    assert branch["continuation_flow"]["partition"]["segment_durations"] == (0.25, 0.25)
    assert branch["endpoint"]["time"] == 1.5


@pytest.mark.parametrize("index", (0, 1))
def test_every_measured_pressure_and_integrator_defect_propagates_in_all_coordinates(study, index):
    branch = study["branches"][index]
    prediction = branch["frozen_prediction"]["payload"]
    n = len(branch["before_event"]["nodes"])
    pressure_error = integrator_error = (F(0),)*n
    c = prediction["realization"]["observation"]
    t = prediction["realization"]["right_inverse"]
    for key in ("baseline_target", "post_event_target", "endpoint_target"):
        _assert_target(branch[key], branch["original_reference"])
        assert branch[key]["target_compatible"]
    for step in branch["baseline_steps"]:
        _assert_step(step)
    for step, matrix, ideal, record in zip(
        branch["continuation_steps"], prediction["fine_euler_matrices"],
        prediction["frames"][1:], branch["forecast_errors"]["frames"], strict=True,
    ):
        _assert_step(step)
        before, after = step["before"]["snapshot"], step["after"]["snapshot"]
        h = step["dt"]
        local_pressure = tuple(h*nu*p for nu, p in zip(before["capacity"], step["before"]["pressure_defect"], strict=True))
        local_integrator = tuple(y-x-h*nu*p for y, x, nu, p in zip(
            after["epi"], before["epi"], before["capacity"], before["stored_pressure"], strict=True,
        ))
        pressure_error = tuple(a+b for a, b in zip(_apply(matrix, pressure_error), local_pressure, strict=True))
        integrator_error = tuple(a+b for a, b in zip(_apply(matrix, integrator_error), local_integrator, strict=True))
        expected = tuple(a+b for a, b in zip(pressure_error, integrator_error, strict=True))
        actual = tuple(a-b for a, b in zip(after["epi"], ideal["epi"], strict=True))
        assert expected == actual == record["actual_minus_ideal_epi"]
        assert record["local_pressure_defect_impulse"] == local_pressure
        assert record["local_integrator_state_defect"] == local_integrator
        assert record["propagated_pressure_error"] == pressure_error
        assert record["propagated_integrator_error"] == integrator_error
        assert record["actual_reduced_state"] == _apply(c, after["epi"])
        assert record["actual_minus_ideal_reduced"] == _apply(c, actual)
        assert _apply(t, record["actual_minus_ideal_reduced"]) == actual
        assert record["maximum_absolute_epi_error"] == max(map(abs, actual))
        assert record["identity_residual"] == (0,)*n
    assert branch["forecast_errors"]["all_exact_identities_pass"]
    assert not branch["forecast_errors"]["zero_runtime_error_claimed"]
    for key in ("post_event_model_checks", "continuation_capture_checks"):
        assert all(value for field, value in branch[key].items() if field != "scope")


def test_actual_emission_targets_all_children_once_and_retains_native_jump_and_metadata(study):
    control, branch = study["branches"]
    assert control["event"] is control["admission"] is None
    event, admission = branch["event"], branch["admission"]
    children = branch["lineage"]["children"]
    assert admission["targets"] == children and admission["allowed"]
    assert admission["configured_gate_switch_unchanged"]
    assert tuple(row["node"] for row in admission["rows"]) == children
    assert all(row["grammar"]["allowed"] and row["strict_emission_allowed"] for row in admission["rows"])
    assert event["stage_result"]["glyph"] == "AL"
    assert event["stage_result"]["schedule"] == "two_phase_jacobi"
    assert event["stage_result"]["nodes_processed"] == 8
    before, after = event["before"], event["after"]
    assert before == branch["before_event"]
    for key in ("nodes", "capacity", "phase", "pressure", "edges", "time"):
        assert before[key] == after[key]
    expected = tuple(F(b)-F(a) for a, b in zip(before["epi"], after["epi"], strict=True))
    assert event["actual_epi_jump"] == expected
    assert event["any_epi_jump"] and event["native_jump_value_seal_verified"]
    native = event["native_epi_jump"]
    conditions = dict(native["runtime_epi_realization_conditions"])
    assert not conditions["all_graph_nodes_targeted_once"]
    assert not native["runtime_epi_realization_certified"]
    assert native["proposal_builder_replayed_from_declared_inputs"]
    assert set(native["target_nodes"]) == set(children) and len(native["target_nodes"]) == len(children)
    assert not native["clip_intervention_nodes"]
    partial = event["partial_event_evidence"]
    permutation = tuple(before["nodes"].index(node) for node in native["nodes"])
    native_before = tuple(F(before["epi"][index]) for index in permutation)
    native_after = tuple(F(after["epi"][index]) for index in permutation)
    assert partial["native_to_state_indices"] == permutation
    assert partial["native_order_before_epi"] == native_before
    assert partial["native_order_after_epi"] == native_after
    assert partial["actual_endpoints_bound_by_node"] and partial["proposal_and_gate_replayed"]
    assert partial["affine_residual_identity_verified"]
    represented = tuple(value+offset for value, offset in zip(
        _apply(native["exact_represented_linear_map"], native_before),
        native["exact_represented_offset"], strict=True,
    ))
    residual = tuple(actual-ideal for actual, ideal in zip(native_after, represented, strict=True))
    assert residual == native["exact_runtime_minus_represented_affine"]
    assert residual == partial["exact_runtime_affine_residual"]
    assert represented == partial["represented_affine_after"]
    assert native["runtime_matches_represented_affine_exactly"] == (not any(residual))
    for i, node in enumerate(before["nodes"]):
        assert (expected[i] > 0) if node in children else (expected[i] == 0)
        assert after["glyph_history"][node] == before["glyph_history"][node] + (("AL",) if node in children else ())
    attributes = dict(event["after_node_attributes"])
    for child in children:
        assert attributes[child]["_emission_activated"] is True
        assert attributes[child]["_structural_lineage"]["activation_count"] >= 1
    for phase in branch["activity"].values():
        assert phase["positive_capacity_at_every_node"]
        assert phase["minimum_capacity"] > 0
        assert phase["nonzero_fresh_rate_nodes"]


def test_paired_response_keeps_mean_offset_and_full_energy_distinct_from_shape(study):
    control, changed = study["branches"]
    metric = control["original_reference"]["metric_weights"]
    paired = study["paired_response"]
    for key, source in (("after_event", "post_event_target"), ("endpoint", "endpoint_target")):
        left, right = control[source]["pattern"]["epi"], changed[source]["pattern"]["epi"]
        delta = tuple(b-a for a, b in zip(left, right, strict=True))
        mean, centered, full, energy = _metric_readout(delta, metric)
        row = paired[key]
        assert row["epi_difference"] == delta
        assert row["weighted_mean_offset"] == mean
        assert row["centered_epi_difference"] == centered
        assert row["centered_H_energy"] == energy
        assert row["full_H_energy"] == full
        assert row["full_epi_equal"] == (not any(delta))
        assert row["shape_equal_modulo_uniform_offset"] == (not any(centered))
    start, end = paired["after_event"], paired["endpoint"]
    assert paired["mean_offset_change"] == end["weighted_mean_offset"]-start["weighted_mean_offset"]
    assert paired["centered_response_energy_change"] == end["centered_H_energy"]-start["centered_H_energy"]
    assert paired["full_epi_return_observed"] == (not start["full_epi_equal"] and end["full_epi_equal"])
    assert paired["centered_response_decreased"] == (end["centered_H_energy"] < start["centered_H_energy"])
    # The homogeneous exact model retains the perturbation's mean; affine F cancels.
    left, right = (branch["frozen_prediction"]["payload"] for branch in study["branches"])
    ideal_delta = tuple(b-a for a, b in zip(left["frames"][0]["epi"], right["frames"][0]["epi"], strict=True))
    for matrix in left["fine_euler_matrices"]:
        ideal_delta = _apply(matrix, ideal_delta)
    assert ideal_delta == tuple(b-a for a, b in zip(left["frames"][-1]["epi"], right["frames"][-1]["epi"], strict=True))
    assert _metric_readout(ideal_delta, metric)[0] == start["weighted_mean_offset"]
    error_difference = tuple(b-a for a, b in zip(
        control["forecast_errors"]["frames"][-1]["actual_minus_ideal_epi"],
        changed["forecast_errors"]["frames"][-1]["actual_minus_ideal_epi"], strict=True,
    ))
    assert tuple(a+b for a, b in zip(ideal_delta, error_difference, strict=True)) == end["epi_difference"]


def test_target_damage_classification_requires_positive_damage_and_uses_original_target(study):
    for branch in study["branches"]:
        base = branch["baseline_target"]["pattern"]["error_variance"]
        post = branch["post_event_target"]["pattern"]["error_variance"]
        end = branch["endpoint_target"]["pattern"]["error_variance"]
        response = branch["original_target_response"]
        assert response["event_variance_change"] == post-base
        assert response["target_damage_present"] == (post > base)
        assert response["continuation_variance_change"] == end-post
        assert response["partial_target_recovery_observed"] == (post > base and end < post)
        assert response["returned_to_pre_event_target_score"] == (post > base and end <= base)
        assert branch["event_target_budget"]["before_reference"] == branch["original_reference"]
        assert branch["event_target_budget"]["after_reference"] == branch["original_reference"]
        event = branch["event_target_budget"]
        metric = branch["original_reference"]["metric_weights"]
        conductance = branch["original_reference"]["source"]["conductance"]
        before = branch["baseline_target"]["pattern"]
        after = branch["post_event_target"]["pattern"]
        delta = tuple(b-a for a, b in zip(before["epi"], after["epi"], strict=True))
        mean, centered, _, energy = _metric_readout(delta, metric)
        assert event["epi_jump"] == delta and event["centered_epi_jump"] == centered
        assert event["mean_epi_jump"] == event["mean_change"] == mean
        assert event["mean_reweighting"] == event["mean_identity_residual"] == 0
        cross = _dot(metric, tuple(u*q for u, q in zip(before["relative_error"], centered, strict=True)))
        variance = event["variance_jump_budget"]
        assert variance["cross_term"] == cross
        assert variance["quadratic_term"] == energy
        assert variance["energy_change"] == post-base == cross+energy
        dirichlet = event["dirichlet_jump_budget"]
        assert dirichlet["cross_term"] == _dot(_laplacian(conductance, before["relative_error"]), centered)
        assert dirichlet["quadratic_term"] == _energy(conductance, centered)
        assert dirichlet["energy_change"] == dirichlet["cross_term"]+dirichlet["quadratic_term"]
        assert event["variance_reset_budget"]["energy_change"] == 0
        assert event["dirichlet_reset_budget"]["energy_change"] == 0
        assert event["variance_identity_residual"] == event["dirichlet_identity_residual"] == 0
    control = study["branches"][0]
    assert not control["original_target_response"]["target_damage_present"]
    assert not control["original_target_response"]["partial_target_recovery_observed"]


def test_strict_emission_observation_is_read_only_and_does_not_enable_global_gates():
    from benchmarks.thol_full_state_response import _emission_admission

    graph = _admission_graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
    before = deepcopy((graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))))
    result = _emission_admission(graph, (1,))
    assert result["allowed"] and result["rows"][0]["strict_emission_allowed"]
    assert graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] is False
    assert before == (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))


@pytest.mark.parametrize("field,value", (("EPI", 0.5), ("nu_f", 0.0)))
def test_explicit_strict_refusal_is_not_a_silent_operator_substitution(field, value):
    from benchmarks.thol_full_state_response import _emission_admission

    graph = _admission_graph()
    graph.nodes[1][field] = value
    result = _emission_admission(graph, (1,))
    assert not result["allowed"]
    assert not result["rows"][0]["strict_emission_allowed"]
    assert result["rows"][0]["strict_emission_refusal"]


@pytest.mark.parametrize("owner,exception", (("validate_candidate", TypeError), ("validate_emission_strict", RuntimeError)))
def test_unexpected_admission_errors_propagate(owner, exception, monkeypatch):
    import benchmarks.thol_full_state_response as benchmark

    def broken(*args, **kwargs):
        raise exception("unexpected gate failure")

    monkeypatch.setattr(benchmark, owner, broken)
    with pytest.raises(exception, match="unexpected gate failure"):
        benchmark._emission_admission(_admission_graph(), (1,))


def test_uniform_offset_is_not_full_return_even_when_centered_error_is_zero():
    from benchmarks.thol_full_state_response import _paired_delta

    result = _paired_delta((F(1), F(2)), (F(4), F(5)), (F(1), F(2)))
    assert result["shape_equal_modulo_uniform_offset"]
    assert not result["full_epi_equal"]
    assert result["weighted_mean_offset"] == 3
    assert result["centered_H_energy"] == 0
    assert result["full_H_energy"] == F(27, 2)


def test_nested_bepi_metadata_preserves_complex_component_bits_and_history_retention():
    from benchmarks.thol_preparation_policy import _literal
    from tnfr.mathematics.epi import BEPIElement

    values = (complex(-0.0, 0.0), complex(0.25, -0.0))
    element = BEPIElement(values, (complex(-0.5, 0.125),), (0.0, 1.0))
    state = element.__getstate__()
    source = {"bepi": state, "history": deque([state], maxlen=3)}
    encoded = _literal(source)
    expected = tuple(("complex", ("binary64", value.real.hex()), ("binary64", value.imag.hex()))
                     for value in values)
    assert encoded["bepi"]["continuous"] == expected
    assert encoded["history"] == ("deque", 3, (encoded["bepi"],))
    assert expected[0][1][1] == "-0x0.0p+0"
    assert expected[1][2][1] == "-0x0.0p+0"
    json.dumps(encoded, allow_nan=False)
    state["continuous"] = (complex(7.0, 8.0),)
    assert encoded["bepi"]["continuous"] == expected


@pytest.mark.parametrize("mutation", ("horizon", "start", "duration", "adjacency", "defect"))
def test_forecast_audit_rejects_mismatched_or_corrupted_observed_segments(execution, mutation):
    from benchmarks.thol_full_state_response import _forecast_error_accounting

    study, _, _, predictions = execution
    steps = deepcopy(list(study["branches"][0]["continuation_steps"]))
    if mutation == "horizon":
        steps.pop()
    elif mutation == "start":
        steps[0]["before"]["snapshot"]["epi"] = (F(99),)*16
    elif mutation == "duration":
        steps[0]["dt"] = F(1, 2)
    elif mutation == "adjacency":
        steps[1]["before"]["snapshot"]["epi"] = (F(99),)*16
    else:
        steps[0]["support_budget"]["state_defect"] = (F(99),)*16
    with pytest.raises((ValueError, RuntimeError)):
        _forecast_error_accounting(predictions[0], steps)


def test_refused_real_branch_stops_before_emission_prediction_or_continuation(monkeypatch):
    import benchmarks.thol_full_state_response as benchmark
    import tnfr.physics.epi_memory as memory
    from tnfr.errors import TNFRValueError

    flow = benchmark._physical_flow
    calls = []

    def actual_flow(graph, *args, **kwargs):
        calls.append(graph.graph["_t"])
        return flow(graph, *args, **kwargs)

    def refuse(*args, **kwargs):
        raise TNFRValueError("Explicit synthetic refusal control")

    monkeypatch.setattr(benchmark, "_physical_flow", actual_flow)
    monkeypatch.setattr(benchmark, "validate_emission_strict", refuse)
    with patch.object(benchmark, "_emission_event", side_effect=AssertionError("refused event executed")), patch.object(
        memory, "predict_forced_support_realization_euler", side_effect=AssertionError("refused forecast issued"),
    ):
        branch = benchmark.run_response_branch("child_emission")
    assert calls == [0.5]
    assert branch["status"] == "refused"
    assert branch["event"] is branch["frozen_prediction"] is branch["continuation_flow"] is None
    assert len(branch["admission"]["rows"]) == 8
    assert not branch["admission"]["allowed"]


def test_cli_json_keeps_frozen_forecasts_native_jump_and_limited_claims(study, tmp_path, monkeypatch):
    import benchmarks.thol_full_state_response as benchmark

    output = tmp_path / "response.json"
    monkeypatch.setattr(benchmark, "run_study", lambda: study)
    monkeypatch.setattr("sys.argv", ["thol_full_state_response", "--output", str(output)])
    benchmark.main()
    payload = json.loads(output.read_bytes())
    CoreExperimentManifest(**payload["manifest"]).validate_for_admission()
    assert payload["experimental_status"] == "No empirical correspondence tested"
    assert not payload["autonomous_maintenance_certified"]
    assert len(payload["branches"]) == 2
    assert all(row["frozen_prediction"]["frozen_before_continuation"] for row in payload["branches"])
    event = payload["branches"][1]["event"]
    assert event["native_jump_value_seal_verified"]
    assert not event["native_epi_jump"]["runtime_epi_realization_certified"]

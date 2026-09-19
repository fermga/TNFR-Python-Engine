"""Actual UM/IL phase amplification is distinct from pressure contraction."""

import inspect
import json
import math
from fractions import Fraction

import networkx as nx
import pytest

from benchmarks import antipodal_region_phase_response as campaign
from tnfr.operators._coherence_stage_kernel import (
    DEFAULT_PHASE_LOCKING_COEFFICIENT,
    propose_coherence_stage,
)
from tnfr.operators.definitions import Coherence
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.operators.preconditions.coherence import (
    coherence_precondition_warnings,
    validate_coherence_strict,
)

F = Fraction
NONZERO = campaign.EPSILONS[1:]
# The finite scalar comparison allows several circular-chart rounding steps.
# This is a test tolerance, not a uniform libm error theorem.
ANGLE_TOLERANCE = 8 * math.ulp(math.tau)


@pytest.fixture(scope="module")
def cases():
    return {
        epsilon: campaign.run_antipodal_phase_case(epsilon)
        for epsilon in campaign.EPSILONS
    }


def _readiness_graph(event):
    """Reconstruct the recorded IL input for read-only precondition checks."""
    graph = nx.Graph()
    graph.add_nodes_from(
        (node, dict(data)) for node, data in event["before"]["node_attributes"].items()
    )
    graph.graph.update(event["before"]["configured_controls"])
    snapshot = event["before_capture"]["snapshot"]
    for node, neighbors in zip(
        snapshot["nodes"], snapshot["support_neighbors"], strict=True
    ):
        graph.add_edges_from((node, neighbor) for neighbor in neighbors)
    return graph


def test_predeclared_controls_share_preparation_and_full_word(cases):
    assert campaign.EPSILONS == (0.0, 2.0**-12, -(2.0**-12), 2.0**-16, -(2.0**-16))
    assert tuple(cases) == campaign.EPSILONS
    references = []
    for epsilon, record in cases.items():
        assert record["epsilon"] == F(epsilon)
        capture = record["initial_capture"]
        assert capture["snapshot"]["nodes"] == tuple(range(6))
        assert capture["snapshot"]["epi"] == (F(1, 2),) * 6
        assert capture["snapshot"]["capacity"] == (1, 1, 1, 2, 2, 2)
        expected = tuple(
            F((base + (epsilon if node < 3 else -epsilon)) % math.tau)
            for node, base in enumerate(campaign.BASE_PHASE)
        )
        assert capture["phase"] == expected
        assert record["initial"]["state"]["time"] == 0.0
        assert record["initial"]["random_provenance"]["resolved_base_seed"] == 17
        assert record["word"]["names"] == ("coupling", "coherence", "silence")
        assert record["word"]["string_validator_passed"]
        assert record["word"]["instance_validator_passed"]
        assert record["word"]["context"] == {"initial_epi_nonzero": True}
        assert "not an observed ancestor" in record["comparison_scope"]
        references.append(record["nominal_reference"])
    assert all(reference == references[0] for reference in references)


@pytest.mark.parametrize("epsilon", campaign.EPSILONS)
@pytest.mark.parametrize(
    "name,glyph,history", (("um", "UM", ("UM",)), ("il", "IL", ("UM", "IL")))
)
def test_actual_six_target_stages_bind_unsealed_predictions_and_metrics(
    cases, epsilon, name, glyph, history
):
    event = cases[epsilon][name]
    assert event["targets"] == tuple(range(6))
    assert len(event["admissions"]) == 6
    assert all(
        item["allowed"] and item["candidate"] == glyph for item in event["admissions"]
    )
    stage = event["stage_result"]
    assert stage["schedule"] == "two_phase_jacobi"
    assert stage["glyph"] == glyph and stage["nodes_processed"] == 6
    assert all(event["prediction_matches_actual"].values())
    assert "not executor-retained proposals" in event["prediction_scope"]
    assert len(event["actual_operator_metrics"]) == 6
    assert all(item["glyph"] == glyph for item in event["actual_operator_metrics"])
    assert all(
        event["raw_state"]["state"]["glyph_history"][node] == history
        for node in range(6)
    )
    before = event["before_capture"]["snapshot"]
    for capture in (event["raw_capture"], event["after_capture"]):
        for field in ("nodes", "epi", "capacity", "conductance", "support_neighbors"):
            assert capture["snapshot"][field] == before[field]
    support = event["compatible_support_after"]
    assert support["components"] == ((0, 1, 2), (3, 4, 5))
    assert support["excluded_edges"] == ((2, 3),) and support["blocked_targets"] == ()
    assert event["phase_after"]["bridge_separation"] > support["effective_phase_limit"]
    budget = event["event_budget"]
    assert budget["epi_jump"] == (0,) * 6
    assert budget["mean_identity_residual"] == 0
    assert budget["error_identity_residual"] == (0,) * 6
    assert (
        budget["variance_identity_residual"]
        == budget["dirichlet_identity_residual"]
        == 0
    )


def test_strict_il_readiness_is_recorded_independently_of_optional_runtime_flag(cases):
    for record in cases.values():
        assert record["um"]["independent_strict_il_readiness"] is None
        event = record["il"]
        graph = _readiness_graph(event)
        assert not graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False)
        readiness = event["independent_strict_il_readiness"]
        assert tuple(item["node"] for item in readiness) == tuple(range(6))
        assert all(value > 0 for value in event["before_capture"]["snapshot"]["epi"])
        assert all(
            value > 0 for value in event["before_capture"]["snapshot"]["capacity"]
        )
        for item in readiness:
            node = item["node"]
            assert item["passed"]
            assert validate_coherence_strict(graph, node, emit_warnings=False) is None
            assert item["warnings"] == coherence_precondition_warnings(graph, node)
            assert bool(item["warnings"]) == (node in (0, 1, 4, 5))


def test_shared_il_default_binds_direct_proposal_and_actual_stage(cases):
    alpha = DEFAULT_PHASE_LOCKING_COEFFICIENT
    assert alpha == 0.3
    assert (
        inspect.signature(propose_coherence_stage)
        .parameters["phase_locking_coefficient"]
        .default
        == alpha
    )
    defaults = canonical_glyph_factor_defaults()
    for record in cases.values():
        event = record["il"]
        graph = _readiness_graph(event)
        assert event["resolved_factors"]["IL_dnfr_factor"] == defaults["IL_dnfr_factor"]
        assert record["phase_model"]["coherence_phase_factor"] == F(alpha)
        assert record["phase_model"]["coupling_phase_factor"] == F(
            defaults["UM_theta_push"]
        )
        for node, recorded in enumerate(event["independent_prediction"]):
            direct = Coherence()._build_proposal(graph, node)
            pure = propose_coherence_stage(graph, node, defaults["IL_dnfr_factor"])
            assert direct == pure
            assert direct.phase.coefficient == recorded["phase"]["coefficient"] == alpha
            assert F(direct.phase.theta_after) == event["raw_capture"]["phase"][node]
            assert (
                F(direct.dnfr_after)
                == event["raw_capture"]["snapshot"]["stored_pressure"][node]
            )


@pytest.mark.parametrize("epsilon", NONZERO)
def test_signed_bridge_response_matches_independent_finite_scalar_formula(
    cases, epsilon
):
    record = cases[epsilon]
    alpha = DEFAULT_PHASE_LOCKING_COEFFICIENT
    # For phase-uniform triangles UM is the identity. IL bridge neighbors sum
    # to cos(epsilon) + 3i sin(epsilon) in the mathematical antipodal chart.
    bridge = epsilon + alpha * (math.atan(3 * math.tan(epsilon)) - epsilon)
    expected = (epsilon, epsilon, bridge, -bridge, -epsilon, -epsilon)
    actual = tuple(float(value) for value in record["phase_after"]["represented_lift"])
    assert actual == pytest.approx(expected, rel=0, abs=ANGLE_TOLERANCE)
    assert (
        record["um"]["before_capture"]["phase"]
        == record["um"]["after_capture"]["phase"]
    )
    assert abs(actual[2]) > abs(epsilon) and actual[2] * epsilon > 0
    assert abs(actual[3]) > abs(epsilon) and actual[3] * epsilon < 0
    assert actual[2] / epsilon == pytest.approx(1.6, rel=0, abs=2e-7)
    before_energy = record["phase_before"]["energy"]
    after_energy = record["phase_after"]["energy"]
    assert before_energy > 0 and after_energy > before_energy
    gain = record["observed_phase_energy_gain"]
    assert gain == after_energy / before_energy
    finite_gain = (2 + (bridge / epsilon) ** 2) / 3
    assert float(gain) == pytest.approx(finite_gain, rel=0, abs=1e-9)
    assert float(gain) == pytest.approx(1.52, rel=0, abs=2e-7)


def test_declared_linear_action_and_binary64_residuals_remain_separate(cases):
    alpha = F(DEFAULT_PHASE_LOCKING_COEFFICIENT)
    for epsilon, record in cases.items():
        e = F(epsilon)
        linear = record["linear_response"]
        assert linear["before"] == linear["after_coupling"] == (e, e)
        assert linear["after_coherence"] == (e, (1 + 2 * alpha) * e)
        assert linear["energy_before"] == 3 * e**2
        assert linear["energy_after"] == (2 + (1 + 2 * alpha) ** 2) * e**2
        assert record["preparation_lift_residual"] == tuple(
            a - b
            for a, b in zip(
                record["phase_before"]["represented_lift"], linear["embedded_before"]
            )
        )
        assert record["observed_minus_linear"] == tuple(
            a - b
            for a, b in zip(
                record["phase_after"]["represented_lift"], linear["embedded_after"]
            )
        )
    assert any(cases[-(2.0**-12)]["preparation_lift_residual"])
    null = cases[0.0]
    assert null["phase_before"]["energy"] == 0
    assert null["observed_phase_energy_gain"] is None
    assert all(
        abs(float(value)) <= ANGLE_TOLERANCE for value in null["observed_minus_linear"]
    )
    # The midpoint path resolves phase pressure below the old phasor/wrap
    # cancellation scale. The nominal null is not a full numeric fixed point.
    before = dict(null["forcing_components_before"])
    after = dict(null["forcing_components_after"])
    assert not any(before["phase"])
    assert any(after["phase"])
    assert max(map(abs, after["phase"])) <= ANGLE_TOLERANCE
    assert before["vf"] == after["vf"] and before["topo"] == after["topo"]


def test_raw_pressure_contraction_does_not_claim_refreshed_coherence_improvement(cases):
    for epsilon, record in cases.items():
        for name in ("um", "il"):
            event = record[name]
            assert event["coherence_raw"] > event["coherence_before"]
            assert event["coherence_refreshed"] < event["coherence_raw"]
            before = event["before_capture"]["snapshot"]["stored_pressure"]
            raw = event["raw_capture"]["snapshot"]["stored_pressure"]
            assert all(abs(a) <= abs(b) for a, b in zip(raw, before))
            assert any(abs(a) < abs(b) for a, b in zip(raw, before))
            assert any(event["raw_capture"]["stored_pressure_residual"])
            assert event["after_capture"]["stored_pressure_residual"] == (0,) * 6
            assert event["raw_capture"]["phase"] == event["after_capture"]["phase"]
        il = record["il"]
        retention = il["resolved_factors"]["IL_dnfr_factor"]
        assert il["raw_capture"]["snapshot"]["stored_pressure"] == tuple(
            F(retention * float(value))
            for value in il["before_capture"]["snapshot"]["stored_pressure"]
        )
        before, after = dict(record["forcing_components_before"]), dict(
            record["forcing_components_after"]
        )
        assert before["vf"] == after["vf"] and before["topo"] == after["topo"]
        if epsilon:
            assert before["phase"] != after["phase"]
            assert all(before["phase"][node] == 0 for node in (0, 1, 4, 5))
            assert all(after["phase"][node] != 0 for node in (0, 1, 4, 5))
            assert il["coherence_refreshed"] < il["coherence_before"]
            assert il["before_capture"]["forcing"] != il["after_capture"]["forcing"]


@pytest.mark.parametrize("epsilon", campaign.EPSILONS)
def test_one_held_nodal_interval_moves_epi_with_phase_and_capacity_held(cases, epsilon):
    record = cases[epsilon]
    before, after = record["post_il_capture"], record["final_capture"]
    assert before["phase"] == after["phase"]
    assert before["snapshot"]["capacity"] == after["snapshot"]["capacity"]
    assert before["snapshot"]["conductance"] == after["snapshot"]["conductance"]
    assert before["snapshot"]["epi"] != after["snapshot"]["epi"]
    flow = record["flow"]
    assert flow["duration"] == F(1, 4)
    assert flow["before"] == record["il"]["after_refresh"]["state"]
    assert flow["after_refresh"] == record["final_before_closure"]["state"]
    assert all(flow["frozen_input_checks"].values())
    evidence = flow["executor_evidence"]
    assert evidence["integrator_provenance_certified"]
    assert evidence["resolved_method"] == "euler" and evidence["resolved_substeps"] == 4
    assert evidence["gamma_is_none"] and not evidence["extended_dynamics_requested"]
    assert not evidence["clipping_applied"]
    assert all(evidence["left_binding"].values()) and all(
        evidence["right_binding"].values()
    )
    budget = flow["regime_step_budget"]
    assert budget["before"]["snapshot"] == before["snapshot"]
    assert budget["after"]["snapshot"] == after["snapshot"]
    assert budget["mean_identity_residual"] == 0
    assert budget["relative_recurrence_residual"] == (0,) * 6
    assert budget["support_budget"]["identity_residual"] == 0
    assert budget["relative_energy_budget"]["identity_residual"] == 0
    state = record["final_before_closure"]["state"]
    assert state["time"] == 0.25
    for node, value in zip(state["nodes"], state["epi"], strict=True):
        assert state["physical_epi_history"][node][-1] == (0.25, value)


def test_sha_is_after_measurement_and_local_expansion_does_not_predict_future_gates(
    cases,
):
    for record in cases.values():
        model = record["phase_model"]
        assert (
            model["det_identity_minus_product"] < 0
            and model["strict_expansion_certificate"]
        )
        assert "no Lyapunov claim" in record["gain_scope"]
        assert "Finite probes do not prove" in record["scope"]
        assert "eventual U3 crossing" in record["scope"]
        assert (
            "complete-runtime instability or autonomous preparation" in record["scope"]
        )
        closure = record["closure_after_measurement"]
        assert closure["stage_result"]["glyph"] == "SHA"
        assert closure["stage_result"]["schedule"] == "two_phase_jacobi"
        assert closure["stage_result"]["nodes_processed"] == 6
        assert (
            len(closure["admissions"]) == len(closure["actual_operator_metrics"]) == 6
        )
        assert all(
            item["allowed"] and item["candidate"] == "SHA"
            for item in closure["admissions"]
        )
        before, after = (
            record["final_before_closure"]["state"],
            closure["after"]["state"],
        )
        assert before["epi"] == after["epi"] and before["time"] == after["time"] == 0.25
        assert all(a < b for a, b in zip(after["capacity"], before["capacity"]))
        assert all(
            after["glyph_history"][node] == ("UM", "IL", "SHA") for node in range(6)
        )


def test_strict_artifact_json_preserves_scoped_infinity_and_phase_defects(cases):
    payload = campaign._artifact_payload({"cases": list(cases.values())})
    decoded = json.loads(json.dumps(payload, allow_nan=False))
    assert len(decoded["cases"]) == 5
    for original, encoded in zip(cases.values(), decoded["cases"], strict=True):
        assert encoded["epsilon"] == str(original["epsilon"])
        assert encoded["observed_minus_linear"] == [
            str(value) for value in original["observed_minus_linear"]
        ]
        assert len(encoded["il"]["actual_operator_metrics"]) == 6
        for metric in encoded["closure_after_measurement"]["actual_operator_metrics"]:
            assert metric["time_to_collapse"] == {"numeric_kind": "positive_infinity"}
    payload["unrelated_nonfinite"] = math.inf
    with pytest.raises(ValueError, match="Out of range float"):
        json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("epsilon", (0, True, 0.01, math.nan, math.inf))
def test_undeclared_or_wrongly_typed_preparations_are_rejected(epsilon):
    with pytest.raises(ValueError, match="predeclared float controls"):
        campaign.prepare_phase_response(epsilon)

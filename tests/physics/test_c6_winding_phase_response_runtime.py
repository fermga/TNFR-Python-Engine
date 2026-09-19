"""Default all-target C6 phase response, winding and nodal-flow evidence."""

import inspect
import json
import math
from fractions import Fraction

import networkx as nx
import pytest

from benchmarks import antipodal_region_phase_response as antipodal
from benchmarks import c6_winding_phase_response as campaign
from tnfr.constants.aliases import ALIAS_THETA
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators._phase_gate import resolve_u3_phase_limits
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.operators.preconditions.coherence import (
    coherence_precondition_warnings,
    validate_coherence_strict,
)
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils import angle_diff

F = Fraction
ANGLE_TOLERANCE = 8 * math.ulp(math.tau)
NONZERO = campaign.CASES[1:]


@pytest.fixture(scope="module")
def cases():
    return {case: campaign.run_c6_phase_case(*case) for case in campaign.CASES}


def _recorded_graph(capture, controls):
    graph = nx.Graph()
    graph.graph.update(controls)
    snapshot = capture["snapshot"]
    for node, phase in zip(snapshot["nodes"], capture["phase"], strict=True):
        graph.add_node(node, **{ALIAS_THETA[0]: float(phase)})
    for node, neighbors in zip(
        snapshot["nodes"], snapshot["support_neighbors"], strict=True
    ):
        graph.add_edges_from((node, neighbor) for neighbor in neighbors)
    return graph


def _finite_phase_response(direction, t, alpha):
    """Independent mathematical C6 phasors, receiver mean and IL midpoint."""
    root_three_half = math.sqrt(3) / 2
    local_means = []
    for i, value in enumerate(direction):
        left, right = direction[(i - 1) % 6], direction[(i + 1) % 6]
        real = (
            math.cos(value)
            + (math.cos(left) + math.cos(right)) / 2
            + root_three_half * (math.sin(left) - math.sin(right))
        )
        imag = (
            math.sin(value)
            + (math.sin(left) + math.sin(right)) / 2
            + root_three_half * (math.cos(right) - math.cos(left))
        )
        local_means.append(math.atan2(imag, real))
    coupled = tuple(
        (1 - t) * value
        + t * (local_means[(i - 1) % 6] + local_means[i] + local_means[(i + 1) % 6]) / 3
        for i, value in enumerate(direction)
    )
    return coupled, tuple(
        (1 - alpha) * value + alpha * (coupled[(i - 1) % 6] + coupled[(i + 1) % 6]) / 2
        for i, value in enumerate(coupled)
    )


def test_five_declared_inputs_keep_canonical_defaults_and_one_null(cases):
    assert campaign.CASES == (
        ("null", 0.0),
        ("k1", 2.0**-12),
        ("k1", -(2.0**-12)),
        ("k3", 2.0**-12),
        ("k3", -(2.0**-12)),
    )
    assert tuple(cases) == campaign.CASES
    assert campaign.DIRECTIONS["k1"] == (1, F(1, 2), F(-1, 2), -1, F(-1, 2), F(1, 2))
    defaults = canonical_glyph_factor_defaults()
    for (mode, epsilon), record in cases.items():
        capture = record["initial_capture"]
        assert record["mode"] == mode and record["epsilon"] == F(epsilon)
        assert capture["snapshot"]["nodes"] == tuple(range(6))
        assert capture["snapshot"]["epi"] == (F(1, 2),) * 6
        assert capture["snapshot"]["capacity"] == (1,) * 6
        assert record["declared_tangent"] == tuple(
            F(epsilon) * v for v in campaign.DIRECTIONS[mode]
        )
        assert capture["phase"] == tuple(
            F((base + epsilon * float(v)) % math.tau)
            for base, v in zip(campaign.BASE_PHASE, campaign.DIRECTIONS[mode])
        )
        assert record["word"]["names"] == ("coupling", "coherence", "silence")
        assert (
            record["word"]["string_validator_passed"]
            and record["word"]["instance_validator_passed"]
        )
        assert record["initial"]["random_provenance"]["resolved_base_seed"] == 17
        controls = record["initial"]["configured_controls"]
        assert controls.get("UM_BIDIRECTIONAL", True)
        assert controls.get("UM_FUNCTIONAL_LINKS", True)
        assert controls.get("UM_SYNC_VF", True) and controls.get(
            "UM_STABILIZE_DNFR", True
        )
        assert controls.get("UM_CANDIDATE_COUNT", 0) == 0
        assert (
            record["um"]["resolved_factors"]["UM_theta_push"]
            == defaults["UM_theta_push"]
        )
        assert record["um"]["resolved_factors"]["UM_vf_sync"] == defaults["UM_vf_sync"]
        assert (
            record["il"]["resolved_factors"]["IL_dnfr_factor"]
            == defaults["IL_dnfr_factor"]
        )
        assert all(
            p["phase"]["coefficient"] == DEFAULT_PHASE_LOCKING_COEFFICIENT
            for p in record["il"]["independent_prediction"]
        )


@pytest.mark.parametrize("case", campaign.CASES)
def test_all_measured_cycles_keep_oriented_winding_and_nonedge_exclusion(cases, case):
    record = cases[case]
    controls = record["initial"]["configured_controls"]
    boundaries = (
        (record["initial_capture"], record["initial_winding"]),
        (record["um"]["raw_capture"], record["um_winding"]),
        (record["um"]["after_capture"], record["um_winding"]),
        (record["il"]["raw_capture"], record["il_winding"]),
        (record["post_il_capture"], record["il_winding"]),
        (record["final_capture"], record["final_winding"]),
    )
    for capture, retained in boundaries:
        graph = _recorded_graph(capture, controls)
        assert graph.number_of_edges() == 6 and nx.is_connected(graph)
        assert all(degree == 2 for _, degree in graph.degree())
        _, gate = resolve_u3_phase_limits(graph.graph, operator_code="UM")
        certificate = certify_phase_winding(graph, range(6), phase_gate=gate)
        assert certificate.is_defined and certificate.winding == 1
        assert (
            certificate.u3_admissible
            and certificate.quantization_residual <= 8 * math.ulp(1.0)
        )
        assert certificate.minimum_u3_margin > 0.5
        assert certificate.minimum_branch_margin > 2.0
        assert retained["certificate"]["winding"] == certificate.winding
        assert (
            retained["certificate"]["minimum_u3_margin"]
            == certificate.minimum_u3_margin
        )
        assert (
            retained["certificate"]["minimum_branch_margin"]
            == certificate.minimum_branch_margin
        )
        reverse = certify_phase_winding(
            graph, tuple(reversed(range(6))), phase_gate=gate
        )
        assert reverse.is_defined and reverse.winding == -1
        assert reverse.minimum_branch_margin == pytest.approx(
            certificate.minimum_branch_margin
        )
        nonedges = tuple(
            (i, j)
            for i in range(6)
            for j in range(i + 1, 6)
            if not graph.has_edge(i, j)
        )
        margins = tuple(
            abs(angle_diff(float(capture["phase"][i]), float(capture["phase"][j])))
            - gate
            for i, j in nonedges
        )
        assert len(margins) == 9
        assert margins == retained["nonedge_exclusion_margins"]
        assert min(margins) == retained["minimum_nonedge_exclusion_margin"] > 0.5
        support = retained["support"]
        assert support["components"] == (tuple(range(6)),)
        assert support["excluded_edges"] == support["blocked_targets"] == ()
        assert all(len(row) == 2 for row in support["compatible_neighbors"])
        assert (
            support["balance"]["is_fixed"] and support["balance"]["energy_before"] == 0
        )


@pytest.mark.parametrize("case", campaign.CASES)
def test_shared_observed_stages_bind_actual_glyphs_and_c6_readout(cases, case):
    record = cases[case]
    assert (
        inspect.signature(antipodal._observed_stage).parameters["phase_readout"].default
        is antipodal._phase_readout
    )
    for name, glyph, history in (("um", "UM", ("UM",)), ("il", "IL", ("UM", "IL"))):
        event = record[name]
        assert event["targets"] == tuple(range(6))
        assert len(event["admissions"]) == len(event["actual_operator_metrics"]) == 6
        assert all(
            a["allowed"] and a["candidate"] == glyph for a in event["admissions"]
        )
        assert all(m["glyph"] == glyph for m in event["actual_operator_metrics"])
        assert event["stage_result"]["schedule"] == "two_phase_jacobi"
        assert (
            event["stage_result"]["glyph"] == glyph
            and event["stage_result"]["nodes_processed"] == 6
        )
        assert all(event["prediction_matches_actual"].values())
        assert "not executor-retained proposals" in event["prediction_scope"]
        assert all(
            event["raw_state"]["state"]["glyph_history"][node] == history
            for node in range(6)
        )
        assert (
            "centered_energy" in event["phase_after"]
            and "bridge" not in event["phase_after"]
        )
        before = event["before_capture"]["snapshot"]
        for capture in (event["raw_capture"], event["after_capture"]):
            for field in (
                "nodes",
                "epi",
                "capacity",
                "conductance",
                "support_neighbors",
            ):
                assert capture["snapshot"][field] == before[field]
        assert event["raw_capture"]["phase"] == event["after_capture"]["phase"]
        assert event["after_capture"]["stored_pressure_residual"] == (0,) * 6
        budget = event["event_budget"]
        assert budget["epi_jump"] == budget["error_identity_residual"] == (0,) * 6
        assert budget["mean_identity_residual"] == 0
        assert (
            budget["variance_identity_residual"]
            == budget["dirichlet_identity_residual"]
            == 0
        )
    prediction = record["um"]["independent_prediction"]
    assert prediction["edges"] == ()
    assert all(
        p["link_candidates"] == () and len(p["phase_proposals"]) == 3
        for p in prediction["target_proposals"]
    )
    for node in range(6):
        contributors = [
            p
            for target in prediction["target_proposals"]
            for p in target["phase_proposals"]
            if p["node"] == node
        ]
        assert len(contributors) == 3


def test_recorded_il_readiness_is_checked_again_without_executing(cases):
    for record in cases.values():
        event = record["il"]
        graph = _recorded_graph(
            event["before_capture"], event["before"]["configured_controls"]
        )
        nx.set_node_attributes(graph, event["before"]["node_attributes"])
        assert not graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False)
        readiness = event["independent_strict_il_readiness"]
        assert tuple(item["node"] for item in readiness) == tuple(range(6))
        for item in readiness:
            assert item["passed"]
            assert (
                validate_coherence_strict(graph, item["node"], emit_warnings=False)
                is None
            )
            assert item["warnings"] == coherence_precondition_warnings(
                graph, item["node"]
            )


@pytest.mark.parametrize("case", NONZERO)
def test_finite_mode_response_and_centered_gain_have_independent_predictions(
    cases, case
):
    record = cases[case]
    mode, epsilon = case
    t = record["um"]["resolved_factors"]["UM_theta_push"]
    alpha = DEFAULT_PHASE_LOCKING_COEFFICIENT
    direction = tuple(epsilon * float(v) for v in campaign.DIRECTIONS[mode])
    coupled, final = _finite_phase_response(direction, t, alpha)
    um_lift = tuple(float(v) for v in record["um"]["phase_after"]["represented_lift"])
    il_lift = tuple(float(v) for v in record["phase_after"]["represented_lift"])
    assert um_lift == pytest.approx(coupled, rel=0, abs=ANGLE_TOLERANCE)
    assert il_lift == pytest.approx(final, rel=0, abs=ANGLE_TOLERANCE)
    exact_t, exact_alpha = F(t), F(alpha)
    multiplier = (
        (1 - exact_t / 2) * (1 - exact_alpha / 2)
        if mode == "k1"
        else (1 - exact_t) * (1 - 2 * exact_alpha)
    )
    tangent = record["tangent_observation"]
    assert tangent["after_coherence"] == tuple(
        multiplier * value for value in record["declared_tangent"]
    )
    gain = record["observed_centered_energy_gain"]
    assert 0 < gain < 1
    assert (
        gain
        == record["phase_after"]["centered_energy"]
        / record["phase_before"]["centered_energy"]
    )
    assert float(gain) == pytest.approx(float(multiplier**2), rel=0, abs=1e-7)
    if mode == "k3":
        # The alternating phasor identity is finite, not merely a derivative.
        expected = tuple(
            float(multiplier * value) for value in record["declared_tangent"]
        )
        assert il_lift == pytest.approx(expected, rel=0, abs=ANGLE_TOLERANCE)
        assert (
            max(abs(float(v)) for v in record["observed_minus_linear"])
            < ANGLE_TOLERANCE
        )


def test_neutral_rotation_null_residue_and_linear_defects_are_not_erased(cases):
    for record in cases.values():
        for phase in (record["phase_before"], record["phase_after"]):
            lift, mean = phase["represented_lift"], phase["mean_shift"]
            centered = phase["centered_lift"]
            assert mean == sum(lift) / 6
            assert centered == tuple(value - mean for value in lift)
            assert sum(centered) == 0
            assert phase["centered_energy"] == sum(value**2 for value in centered) / 2
            assert (
                sum(value**2 for value in lift) / 2
                == phase["centered_energy"] + 3 * mean**2
            )
        assert record["preparation_lift_residual"] == tuple(
            a - b
            for a, b in zip(
                record["phase_before"]["represented_lift"], record["declared_tangent"]
            )
        )
        assert record["observed_minus_linear"] == tuple(
            a - b
            for a, b in zip(
                record["phase_after"]["represented_lift"],
                record["tangent_observation"]["after_coherence"],
            )
        )
    null = cases[("null", 0.0)]
    assert null["phase_before"]["centered_energy"] == 0
    assert null["observed_centered_energy_gain"] is None
    assert all(abs(float(v)) < ANGLE_TOLERANCE for v in null["observed_minus_linear"])
    assert all(
        abs(float(v)) < ANGLE_TOLERANCE
        for v in null["realized_phase_pressure_pi_scaled"]
    )


@pytest.mark.parametrize("case", campaign.CASES)
def test_midpoint_phase_pressure_is_bound_to_actual_endpoint_and_forcing(cases, case):
    record = cases[case]
    lift = record["phase_after"]["represented_lift"]
    midpoint = tuple(
        (lift[(i - 1) % 6] + lift[(i + 1) % 6]) / 2 - lift[i] for i in range(6)
    )
    scaled = tuple(F(math.pi) * v for v in record["post_il_capture"]["phase_gradient"])
    assert record["endpoint_midpoint_pressure_pi_scaled"] == midpoint
    assert record["realized_phase_pressure_pi_scaled"] == scaled
    assert record["midpoint_pressure_residual"] == tuple(
        a - b for a, b in zip(scaled, midpoint)
    )
    assert (
        max(abs(float(v)) for v in record["midpoint_pressure_residual"])
        < ANGLE_TOLERANCE
    )
    assert record["linear_pressure_residual"] == tuple(
        a - b
        for a, b in zip(
            scaled, record["tangent_observation"]["phase_pressure_pi_scaled"]
        )
    )
    components = dict(record["forcing_components"])
    assert components["vf"] == components["topo"] == (0,) * 6
    assert components["phase"] == record["post_il_capture"]["forcing"]
    if case[0] != "null":
        assert any(components["phase"])
    il = record["il"]
    assert il["coherence_raw"] >= il["coherence_before"]
    assert il["raw_capture"]["snapshot"]["stored_pressure"] == tuple(
        F(il["resolved_factors"]["IL_dnfr_factor"] * float(v))
        for v in il["before_capture"]["snapshot"]["stored_pressure"]
    )


@pytest.mark.parametrize("case", campaign.CASES)
def test_one_nodal_interval_holds_phase_and_binds_positive_capacity_evolution(
    cases, case
):
    record = cases[case]
    before, after = record["post_il_capture"], record["final_capture"]
    assert before["phase"] == after["phase"]
    assert before["snapshot"]["capacity"] == after["snapshot"]["capacity"] == (1,) * 6
    assert before["snapshot"]["conductance"] == after["snapshot"]["conductance"]
    if case[0] != "null":
        assert before["snapshot"]["epi"] != after["snapshot"]["epi"]
    flow, evidence = record["flow"], record["flow"]["executor_evidence"]
    assert flow["duration"] == F(1, 4)
    assert flow["before"] == record["il"]["after_refresh"]["state"]
    assert flow["after_refresh"] == record["final_before_closure"]["state"]
    assert all(flow["frozen_input_checks"].values())
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


def test_terminal_closure_preserves_winding_without_claiming_autonomous_preparation(
    cases,
):
    for record in cases.values():
        closure = record["closure_after_measurement"]
        assert closure["stage_result"]["glyph"] == "SHA"
        assert closure["stage_result"]["schedule"] == "two_phase_jacobi"
        assert closure["stage_result"]["nodes_processed"] == 6
        assert (
            len(closure["admissions"]) == len(closure["actual_operator_metrics"]) == 6
        )
        assert all(
            a["allowed"] and a["candidate"] == "SHA" for a in closure["admissions"]
        )
        before, after = (
            record["final_before_closure"]["state"],
            closure["after"]["state"],
        )
        assert before["epi"] == after["epi"] and before["phase"] == after["phase"]
        assert before["time"] == after["time"] == 0.25
        assert all(0 < a < b for a, b in zip(after["capacity"], before["capacity"]))
        assert all(
            after["glyph_history"][node] == ("UM", "IL", "SHA") for node in range(6)
        )
        graph = _recorded_graph(
            record["final_capture"], closure["after"]["configured_controls"]
        )
        nx.set_node_attributes(graph, closure["after"]["node_attributes"])
        _, gate = resolve_u3_phase_limits(graph.graph, operator_code="UM")
        assert certify_phase_winding(graph, range(6), phase_gate=gate).winding == 1
        assert "No autonomous formation" in record["scope"]
        assert (
            "binary64 asymptotic convergence or future complete-word admission"
            in record["scope"]
        )
        assert "not inferred from rational tangent sums" in record["scope"]


def test_strict_json_keeps_exact_residuals_and_named_sha_infinity(cases):
    payload = campaign._artifact_payload({"cases": list(cases.values())})
    encoded = json.loads(json.dumps(payload, allow_nan=False))
    assert len(encoded["cases"]) == 5
    for original, row in zip(cases.values(), encoded["cases"], strict=True):
        assert row["midpoint_pressure_residual"] == [
            str(v) for v in original["midpoint_pressure_residual"]
        ]
        assert row["observed_minus_linear"] == [
            str(v) for v in original["observed_minus_linear"]
        ]
        assert row["final_winding"]["certificate"]["winding"] == 1
        assert all(
            metric["time_to_collapse"] == {"numeric_kind": "positive_infinity"}
            for metric in row["closure_after_measurement"]["actual_operator_metrics"]
        )


@pytest.mark.parametrize(
    "case", (("null", 0), ("null", 2.0**-12), ("k2", 2.0**-12), ("k1", math.nan))
)
def test_unplanned_or_wrongly_typed_preparations_are_rejected(case):
    with pytest.raises(ValueError, match="five declared C6 controls"):
        campaign.prepare_c6_phase_response(*case)

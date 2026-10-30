"""Distributed birth/attachment observations, without a persistence theorem."""

import json
from dataclasses import asdict, replace
from fractions import Fraction

import pytest

from benchmarks.thol_pressure_feedback import _payload

CASES = ("attached", "links_disabled", "stale_sample")


@pytest.fixture(scope="module")
def study():
    from benchmarks.thol_distributed_transport import run_study

    return run_study()


@pytest.fixture(scope="module")
def cases(study):
    return {case["case"]: case for case in study["cases"]}


def _energy(conductance, values):
    """Independent directed-edge Dirichlet convention on exact inputs."""
    return (
        sum(
            (weight * (values[i] - values[j]) ** 2 for i, j, weight in conductance),
            Fraction(0),
        )
        / 4
    )


def _laplacian(conductance, values):
    result = [Fraction(0) for _ in values]
    for i, j, weight in conductance:
        result[i] += weight * (values[i] - values[j])
    return tuple(result)


def _assert_euler_budget(budget):
    before, after = budget["before"], budget["after"]
    assert before["nodes"] == after["nodes"]
    assert before["conductance"] == after["conductance"]
    assert before["capacity"] == after["capacity"]
    h = budget["dt"]
    edges = before["conductance"]
    x, observed = before["epi"], after["epi"]
    rate = tuple(
        nu * pressure
        for nu, pressure in zip(
            before["capacity"],
            before["stored_pressure"],
            strict=True,
        )
    )
    expected = tuple(
        value + h * velocity for value, velocity in zip(x, rate, strict=True)
    )
    defect = tuple(
        value - ideal for value, ideal in zip(observed, expected, strict=True)
    )
    drift = h * sum(
        (
            gradient * velocity
            for gradient, velocity in zip(
                _laplacian(edges, x),
                rate,
                strict=True,
            )
        ),
        Fraction(0),
    )
    quadratic = h**2 * _energy(edges, rate)
    defect_term = sum(
        (
            gradient * error
            for gradient, error in zip(
                _laplacian(edges, expected),
                defect,
                strict=True,
            )
        ),
        Fraction(0),
    ) + _energy(edges, defect)
    change = _energy(edges, observed) - _energy(edges, x)
    assert budget["expected_epi"] == expected
    assert budget["state_defect"] == defect
    assert budget["drift_term"] == drift
    assert budget["quadratic_term"] == quadratic
    assert budget["defect_term"] == defect_term
    assert budget["energy_change"] == change
    assert budget["identity_residual"] == change - drift - quadratic - defect_term == 0


def test_all_controls_replay_the_unchanged_causal_preparation_and_public_births(cases):
    assert tuple(cases) == CASES
    baseline = cases["attached"]
    for case in cases.values():
        assert case["preparation"] == baseline["preparation"]
        assert case["birth"] == baseline["birth"]
        preparation = case["preparation"]
        assert preparation["preparation"] == "all_nodes"
        assert preparation["initial"]["epi"] == (2.0, 0.5) * 4
        assert preparation["initial"]["capacity"] == (1.0,) * 8
        assert preparation["physical_steps"] == (0.25, 0.25)
        assert preparation["default_birth_threshold"] == 0.1
        assert preparation["before_birth"] == case["birth"]["before"]
        assert tuple(row["glyph"] for row in preparation["actual_prefix"]) == (
            "IL",
            "OZ",
        )
        assert all(
            row["targets"] == tuple(range(8)) for row in preparation["actual_prefix"]
        )


@pytest.mark.parametrize("name", CASES)
def test_all_eight_actual_parents_birth_isolated_children_without_rewriting_old_epi(
    cases, name
):
    birth = cases[name]["birth"]
    before, after = birth["before"], birth["after"]
    pairs = birth["parent_children"]
    assert tuple(parent for parent, _ in pairs) == tuple(range(8))
    children = tuple(child for _, child in pairs)
    assert len(set(children)) == 8
    assert before["nodes"] == tuple(range(8))
    assert after["nodes"] == (*before["nodes"], *children)
    assert before["time"] == after["time"] == 0.5
    assert after["edges"] == before["edges"]
    assert after["epi"][:8] == before["epi"]
    assert after["phase"][:8] == before["phase"]
    assert after["capacity"][:8] == (1.0,) * 8
    assert after["capacity"][8:] == (0.95,) * 8
    assert after["pressure"][8:] == (0.0,) * 8
    assert birth["stage_result"]["schedule"] == "two_phase_jacobi"
    assert birth["stage_result"]["nodes_processed"] == 8
    assert birth["policy"] == "all_eligible_once"
    assert birth["eligibility"]["eligible_nodes"] == tuple(range(8))
    assert all(
        row["eligible"] and row["proposal_valid"] and row["birth_proposed"]
        for row in birth["eligibility"]["candidates"]
    )
    for record in birth["children"]:
        assert record["degree"] == 0
        assert record["node_data"]["parent_node"] == record["parent"]
    for parent, child in pairs:
        assert before["glyph_history"][parent] == ("IL", "OZ")
        assert after["glyph_history"][parent] == ("IL", "OZ", "THOL")
        assert after["glyph_history"][child] == ()
        assert after["physical_epi_history"][child] == ()
    assert "no same-dimensional support-reset certificate" in birth["scope"]


@pytest.mark.parametrize("name", CASES)
def test_simultaneous_um_replays_complete_proposal_and_commits_actual_outputs(
    cases, name
):
    coupling = cases[name]["coupling"]
    proposal = coupling["kernel_proposal"]
    before, raw = coupling["before"], coupling["after_raw"]
    assert coupling["targets"] == proposal["targets"] == tuple(range(8))
    assert len(proposal["target_proposals"]) == 8
    assert coupling["stage_result"]["nodes_processed"] == 8
    assert coupling["stage_result"]["schedule"] == "two_phase_jacobi"
    assert tuple(row["node"] for row in coupling["admissions"]) == tuple(range(8))
    assert all(
        row["allowed"] and row["candidate"] == "UM" for row in coupling["admissions"]
    )
    assert coupling["bidirectional"] and coupling["candidate_limit"] == 0
    assert coupling["candidate_mode"] == "sample"
    assert before == cases[name]["birth"]["after"]
    assert raw["epi"] == before["epi"]
    assert raw["nodes"] == before["nodes"]
    expected = {
        field: list(before[field]) for field in ("phase", "capacity", "pressure")
    }
    indices = {node: index for index, node in enumerate(raw["nodes"])}
    for update in proposal["node_updates"]:
        for key, field in (
            ("theta_after", "phase"),
            ("vf_after", "capacity"),
            ("dnfr_after", "pressure"),
        ):
            if update[key] is not None:
                expected[field][indices[update["node"]]] = update[key]
    for field, values in expected.items():
        assert raw[field] == tuple(values)
    actual = {
        frozenset((u, v)): attrs["weight"] for u, v, attrs in coupling["new_edges"]
    }
    proposed = {
        frozenset((edge["left"], edge["right"])): edge["weight"]
        for edge in proposal["edges"]
    }
    assert actual == proposed
    old = {frozenset((u, v)): attrs for u, v, attrs in before["edges"]}
    current = {frozenset((u, v)): attrs for u, v, attrs in raw["edges"]}
    assert all(current[edge] == attrs for edge, attrs in old.items())
    assert set(current) - set(old) == set(actual)
    for parent, child in cases[name]["birth"]["parent_children"]:
        assert raw["glyph_history"][parent] == ("IL", "OZ", "THOL", "UM")
        assert raw["glyph_history"][child] == ()


def test_fresh_sample_selects_the_observed_sixteen_edges_not_eight_prescribed_links(
    cases,
):
    normal = cases["attached"]
    coupling = normal["coupling"]
    children = dict(normal["birth"]["parent_children"])
    assert coupling["functional_links"] and coupling["sampling_refresh_executed"]
    assert coupling["node_sample_before"] == tuple(range(8))
    assert coupling["node_sample_used"] == normal["birth"]["after"]["nodes"]
    expected = {frozenset((parent, child)) for parent, child in children.items()}
    expected.update(
        frozenset((neighbor, children[parent]))
        for parent in range(0, 8, 2)
        for neighbor in ((parent - 1) % 8, (parent + 1) % 8)
    )
    assert len(expected) == 16
    assert {frozenset((u, v)) for u, v, _ in coupling["new_edges"]} == expected
    assert tuple(
        row["child_response"]["degree_after_um"] for row in normal["responses"]
    ) == (
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
    )
    assert all(0 < attrs["weight"] < 1 for _, _, attrs in coupling["new_edges"])
    for parent, child in children.items():
        item = next(row for row in normal["responses"] if row["parent"] == parent)
        assert parent in item["child_response"]["neighbors_after_um"]
        assert item["child_response"]["refreshed_pressure_after_um"] > 0
        assert item["child_response"]["node"] == child


def test_disabled_links_and_old_sample_are_distinct_inputs_with_the_same_observed_isolation(
    cases,
):
    disabled = cases["links_disabled"]["coupling"]
    stale = cases["stale_sample"]["coupling"]
    assert not disabled["functional_links"] and disabled["sampling_refresh_executed"]
    assert stale["functional_links"] and not stale["sampling_refresh_executed"]
    assert len(disabled["node_sample_used"]) == 16
    assert stale["node_sample_used"] == stale["node_sample_before"] == tuple(range(8))
    assert disabled["new_edges"] == stale["new_edges"] == ()
    assert disabled["after_raw"] == stale["after_raw"]
    assert disabled["after_refreshed"] == stale["after_refreshed"]
    for name in ("links_disabled", "stale_sample"):
        case = cases[name]
        assert case["coupling"]["after_refreshed"]["pressure"][8:] == (0.0,) * 8
        for row in case["responses"]:
            child = row["child_response"]
            assert child["degree_after_um"] == 0
            assert child["exact_epi_change"] == 0
            assert child["endpoint_pressure"] == 0
        assert case["endpoint"]["capacity"][8:] == (0.95,) * 8


@pytest.mark.parametrize("name", CASES)
def test_support_attachment_budget_uses_the_same_postbirth_dimension_and_exact_edges(
    cases, name
):
    coupling = cases[name]["coupling"]
    budget = coupling["support_reset_budget"]
    before, after = budget["before"], budget["after"]
    assert before["nodes"] == after["nodes"] == cases[name]["birth"]["after"]["nodes"]
    assert len(before["nodes"]) == 16
    assert before["epi"] == after["epi"]
    expected = _energy(after["conductance"], after["epi"]) - _energy(
        before["conductance"],
        before["epi"],
    )
    assert budget["energy_change"] == budget["edge_energy_change"] == expected
    assert budget["identity_residual"] == 0
    indices = {node: index for index, node in enumerate(before["nodes"])}
    by_added_edges = sum(
        (
            Fraction.from_float(attrs["weight"])
            * (before["epi"][indices[u]] - before["epi"][indices[v]]) ** 2
            / 2
            for u, v, attrs in coupling["new_edges"]
        ),
        Fraction(0),
    )
    assert by_added_edges == expected
    assert (expected > 0) == (name == "attached")
    refreshed = coupling["pressure_refresh_reset_budget"]
    assert refreshed["energy_change"] == refreshed["edge_energy_change"] == 0
    assert refreshed["identity_residual"] == 0


@pytest.mark.parametrize("name", CASES)
def test_fresh_postgrowth_executor_owns_exactly_two_segments_and_three_pressure_boundaries(
    cases, name
):
    case = cases[name]
    flow = case["physical_flow"]
    assert flow["before"] == case["coupling"]["after_refreshed"]
    assert flow["after"] == case["endpoint"]
    assert flow["partition"]["segment_durations"] == (0.25, 0.25)
    intervals = flow["partition"]["segments"]
    assert tuple((row["start_time"], row["end_time"]) for row in intervals) == (
        (0.5, 0.75),
        (0.75, 1.0),
    )
    assert flow["pressure_refresh_callback_invocations"] == 3
    assert flow["physical_pressure_reevaluated_partition_established"]
    assert flow["all_segment_binary64_replays_identified"]
    assert tuple(row["time"] for row in flow["boundaries"]) == (0.5, 0.75, 1.0)
    assert len(flow["segments"]) == 2
    for boundary in flow["boundaries"]:
        assert (
            boundary["callback_completed"] and boundary["nonpressure_state_preserved"]
        )
        assert len(boundary["before"]["nodes"]) == 16
        for field in ("nodes", "epi", "exact_epi", "nu_f", "exact_nu_f", "conductance"):
            assert boundary["before"][field] == boundary["after"][field]
        assert boundary["refresh_reset_budget"]["energy_change"] == 0
    for field in ("nodes", "edges", "capacity", "phase", "glyph_history"):
        assert flow["before"][field] == flow["after"][field]
    assert flow["before"]["time"] == 0.5
    assert flow["after"]["time"] == 1.0
    for _, child in case["birth"]["parent_children"]:
        history = case["endpoint"]["physical_epi_history"][child]
        assert tuple(t for t, _ in history) == (0.5, 0.75, 1.0)
        index = case["endpoint"]["nodes"].index(child)
        assert history[0][1] == case["birth"]["after"]["epi"][index]
        assert history[-1][1] == case["endpoint"]["epi"][index]


@pytest.mark.parametrize("name", CASES)
def test_held_rate_and_refreshed_endpoint_pressure_are_distinct_in_exact_euler_accounting(
    cases, name
):
    flow = cases[name]["physical_flow"]
    for index, segment in enumerate(flow["segments"]):
        assert segment["before"] == flow["boundaries"][index]["after"]
        assert segment["held_after"] == flow["boundaries"][index + 1]["before"]
        assert segment["before"]["delta_nfr"] == segment["held_after"]["delta_nfr"]
        assert segment["pressure_unchanged"]
        assert segment["integrator_provenance_certified"]
        assert segment["binary64_held_pressure_interval_identified"]
        assert segment["method"] == "euler" and not segment["clipping_applied"]
        assert segment["interval"]["duration"] == 0.25
        _assert_euler_budget(segment["exact_euler_budget"])
        assert (
            segment["before_support"]["stored_pressure"]
            == segment["after_support"]["stored_pressure"]
        )
        next_pressure = flow["boundaries"][index + 1]["after"]["exact_delta_nfr"]
        assert next_pressure != segment["after_support"]["stored_pressure"]
    telescope = sum(
        (
            segment["exact_euler_budget"]["energy_change"]
            for segment in flow["segments"]
        ),
        Fraction(0),
    )
    assert (
        flow["exact_energy_change"]
        == flow["exact_segment_energy_change_sum"]
        == telescope
    )
    assert flow["energy_telescope_residual"] == 0


def test_report_serialization_keeps_finite_execution_separate_from_future_or_physical_claims(
    study,
):
    encoded = json.dumps(_payload(study), allow_nan=False)
    payload = json.loads(encoded)
    assert len(payload["cases"]) == 3
    for case in payload["cases"]:
        flow = case["physical_flow"]
        for flag in (
            "solver_accuracy_certified",
            "mesh_convergence_certified",
            "future_or_repeated_behavior_certified",
        ):
            assert flow[flag] is False
        for segment in flow["segments"]:
            assert segment["pure_epi_diffusion_eligible"] is False
            assert segment["exact_affine_map_identified"] is False
            assert segment["global_disagreement_contraction_certified"] is False
        assert "no final SHA" in case["scope"]
        assert "empirical validation" in case["scope"]
        assert "not one transaction" in flow["scope"]


@pytest.mark.parametrize("change", ("node_order", "conductance", "capacity", "raw_epi"))
def test_detached_flow_support_bridge_rejects_mismatched_authoritative_channels(
    cases, change
):
    from benchmarks.thol_distributed_transport import _support_from_flow_state
    from tnfr.physics.runtime_flow_stability import NodalFlowStateSnapshot
    from tnfr.physics.support_transport import SupportTransportSnapshot

    case = cases["attached"]
    reference = SupportTransportSnapshot(
        **case["coupling"]["after_support"]["snapshot"]
    )
    state = NodalFlowStateSnapshot(**case["physical_flow"]["segments"][0]["before"])
    if change == "node_order":
        changed = replace(state, nodes=state.nodes[1:] + state.nodes[:1])
    elif change == "conductance":
        dense = [list(row) for row in state.conductance]
        dense[0][1] += 1
        changed = replace(state, conductance=tuple(tuple(row) for row in dense))
    elif change == "capacity":
        changed = replace(state, exact_nu_f=(Fraction(2), *state.exact_nu_f[1:]))
    else:
        changed = replace(state, epi=(state.epi[0] + 1, *state.epi[1:]))
    with pytest.raises(ValueError):
        _support_from_flow_state(reference, changed)


def test_detached_bridge_rebuilds_derived_energy_instead_of_trusting_supplied_cache(
    cases,
):
    from benchmarks.thol_distributed_transport import _support_from_flow_state
    from tnfr.physics.runtime_flow_stability import NodalFlowStateSnapshot
    from tnfr.physics.support_transport import SupportTransportSnapshot

    case = cases["attached"]
    reference = SupportTransportSnapshot(
        **case["coupling"]["after_support"]["snapshot"]
    )
    segment = case["physical_flow"]["segments"][0]
    state = NodalFlowStateSnapshot(**segment["before"])
    forged_cache = replace(
        reference, dirichlet_energy=Fraction(-123), energy_rate=Fraction(456)
    )
    assert (
        asdict(_support_from_flow_state(forged_cache, state))
        == segment["before_support"]
    )


def test_connected_generated_support_keeps_all_children_in_its_exact_capacity_source_budget(
    cases,
):
    attached = cases["attached"]
    coupling = attached["coupling"]
    snapshot = coupling["after_support"]["snapshot"]
    inventory = coupling["support_inventory"]
    assert inventory["positive_conductance_components"] == (snapshot["nodes"],)
    assert inventory["directed_positive_conductance_entries"] == 48
    assert inventory["directed_unique_support_entries"] == 48
    assert inventory["minimum_capacity"] == Fraction.from_float(0.95)
    budget = attached["weighted_source_budget"]
    assert budget["available"]
    assert snapshot == coupling["refreshed_forcing"]["observation"]["snapshot"]
    nu = snapshot["capacity"]
    assert nu == (Fraction(1),) * 8 + (Fraction.from_float(0.95),) * 8
    strength = [Fraction(0) for _ in nu]
    child_counts = [0] * 8
    child_strengths = [Fraction(0) for _ in range(8)]
    for i, j, weight in snapshot["conductance"]:
        strength[i] += weight
        if i < 8 and j >= 8:
            child_counts[i] += 1
            child_strengths[i] += weight
    assert tuple(child_counts) == (1, 3, 1, 3, 1, 3, 1, 3)
    assert budget["strengths"] == tuple(strength)
    metric = tuple(
        degree / capacity for degree, capacity in zip(strength, nu, strict=True)
    )
    assert all(value > 0 for value in metric)
    assert budget["metric_weights"] == metric
    captured = coupling["refreshed_forcing"]
    observation = captured["observation"]
    w_vf = dict(observation["normalized_weights"])["vf"]
    vf_gradient = tuple(
        sum((nu[j] - nu[i] for j in neighbors), Fraction(0)) / len(neighbors)
        for i, neighbors in enumerate(snapshot["support_neighbors"])
    )
    actual = sum(
        (
            d * w_vf * gradient
            for d, gradient in zip(strength, vf_gradient, strict=True)
        ),
        Fraction(0),
    )
    closed = (
        2
        * w_vf
        * (1 - nu[8])
        * sum(
            (
                (s - count) / (2 + count)
                for s, count in zip(child_strengths, child_counts, strict=True)
            ),
            Fraction(0),
        )
    )
    assert actual == closed < 0
    assert (
        budget["capacity_weighted_source"]
        == budget["capacity_closed_form_source"]
        == actual
    )
    assert budget["capacity_identity_residual"] == 0
    for i, incidence in enumerate(budget["parent_child_incidence"]):
        assert incidence == {
            "parent": i,
            "child_count": child_counts[i],
            "child_strength": child_strengths[i],
        }
    components = dict(captured["components"])
    components["epi"] = tuple(
        observation["epi_weight"] * g for g in snapshot["epi_gradient"]
    )
    weighted = {
        name: sum((d * f for d, f in zip(strength, values, strict=True)), Fraction(0))
        for name, values in components.items()
    }
    assert budget["weighted_source_by_channel"] == weighted
    assert weighted["epi"] == 0
    mass = sum(metric, Fraction(0))
    assert budget["mean_rate_by_channel"] == {
        name: value / mass for name, value in weighted.items()
    }
    kernel_defect = (
        sum(
            (
                d * f
                for d, f in zip(
                    strength,
                    observation["kernel_pressure_defect"],
                    strict=True,
                )
            ),
            Fraction(0),
        )
        / mass
    )
    stored_defect = (
        sum(
            (
                d * f
                for d, f in zip(
                    strength,
                    observation["stored_pressure_residual"],
                    strict=True,
                )
            ),
            Fraction(0),
        )
        / mass
    )
    observed_rate = (
        sum(
            (
                h * capacity * pressure
                for h, capacity, pressure in zip(
                    metric,
                    nu,
                    snapshot["stored_pressure"],
                    strict=True,
                )
            ),
            Fraction(0),
        )
        / mass
    )
    assert budget["kernel_mean_rate_defect"] == kernel_defect
    assert budget["stored_pressure_mean_rate_defect"] == stored_defect
    assert budget["represented_weighted_mean_rate"] == observed_rate
    assert (
        observed_rate
        == sum(weighted.values(), Fraction(0)) / mass + kernel_defect + stored_defect
    )
    assert budget["mean_rate_identity_residual"] == 0


@pytest.mark.parametrize("name", ("links_disabled", "stale_sample"))
def test_disconnected_controls_do_not_discard_children_to_claim_positive_metric(
    cases, name
):
    case = cases[name]
    snapshot = case["coupling"]["after_support"]["snapshot"]
    inventory = case["coupling"]["support_inventory"]
    assert inventory["positive_conductance_components"] == (
        tuple(range(8)),
        *((child,) for _, child in case["birth"]["parent_children"]),
    )
    assert inventory["directed_positive_conductance_entries"] == 16
    assert inventory["directed_unique_support_entries"] == 16
    assert all(not row for row in snapshot["support_neighbors"][8:])
    assert case["weighted_source_budget"]["available"] is False
    assert "disconnected" in case["weighted_source_budget"]["reason"]


@pytest.mark.parametrize("name", CASES)
def test_all_recorded_parent_and_child_responses_match_real_endpoints(cases, name):
    case = cases[name]
    nodes = case["endpoint"]["nodes"]
    assert len(case["responses"]) == 8
    for pair in case["responses"]:
        for kind in ("parent", "child"):
            node = pair[kind]
            row = pair[kind + "_response"]
            index = nodes.index(node)
            assert row["node"] == node
            initial = Fraction.from_float(case["birth"]["after"]["epi"][index])
            final = Fraction.from_float(case["endpoint"]["epi"][index])
            assert row["epi_after_birth"] == initial
            assert row["epi_endpoint"] == final
            assert row["exact_epi_change"] == final - initial
        if name == "attached":
            assert pair["child_response"]["exact_epi_change"] > 0
    assert case["physical_flow"]["exact_energy_change"] < 0

"""Causal birth needs admitted, actually selected support before transport."""

from fractions import Fraction

import pytest

from benchmarks.thol_birth_transport import (
    CASES, INITIAL_EPI, STEPS, WORD, run_birth_transport_case,
)
from tnfr.constants.canonical import SHA_VF_FACTOR, UM_COMPAT_THRESHOLD


@pytest.fixture(scope="module")
def cases():
    return {name: run_birth_transport_case(name) for name in CASES}


def test_causal_checkerboard_crosses_default_threshold_with_model_residual(cases):
    preparation = cases["attached"]["preparation"]
    observed = preparation["physical_acceleration"]
    reference = preparation["checkerboard_reference"]
    assert preparation["initial"]["epi"] == INITIAL_EPI == (2.0, 0.5) * 4
    assert preparation["initial"]["capacity"] == (1.0,) * 8
    assert preparation["physical_steps"] == STEPS == (0.25, 0.25)
    assert tuple(time for time, _ in observed["physical_samples"]) == (0, 0.25, 0.5)
    assert observed["physical_samples"][-1][1] == preparation["before_birth"][
        "epi"
    ][0]
    assert preparation["segment_methods"] == ("euler", "euler")
    assert preparation["clipping_applied"] == (False, False)
    assert all(row["pressure_only_refresh"] for row in preparation[
        "physical_boundaries"
    ])
    e = reference["represented_epi_weight"]
    assert reference["exact_model_acceleration"] == 3 * e**2
    assert reference["exact_model_threshold_margin"] > Fraction(6, 10_000)
    assert preparation["default_birth_threshold"] == 0.1
    assert observed["observed_acceleration"] > 0.1
    assert abs(reference["exact_observed_acceleration_residual"]) < 1e-14
    assert max(map(abs, reference["exact_endpoint_residual"])) < 1e-15
    expected = tuple(
        Fraction(5, 4) + (1 if i % 2 == 0 else -1) * Fraction(3, 4)
        * (1 - e / 2)**2 for i in range(8)
    )
    assert reference["exact_model_endpoint"] == expected


@pytest.mark.parametrize("name", CASES)
def test_birth_is_executed_and_initially_has_no_transport_edge(cases, name):
    preparation = cases[name]["preparation"]
    before = preparation["before_birth"]
    raw = preparation["raw_after_birth"]
    refreshed = preparation["after_birth_refresh"]
    assert all(item["allowed"] for item in preparation["actual_admissions"])
    assert before["glyph_history"][0] == ("IL", "OZ")
    assert raw["glyph_history"][0] == ("IL", "OZ", "THOL")
    assert len(preparation["children"]) == 1
    assert len(raw["nodes"]) == 9
    assert preparation["child_degrees_after_birth"] == (0,)
    assert raw["edges"] == before["edges"]
    assert raw["epi"][:8] == before["epi"]
    assert raw["capacity"][:8] == before["capacity"]
    assert raw["phase"][:8] == before["phase"]
    assert raw["capacity"][-1] == 0.95
    assert raw["pressure"][0] != before["pressure"][0]
    assert refreshed["pressure"][:8] == before["pressure"]
    assert refreshed["pressure"][-1] == 0


def test_real_sample_refresh_and_compatibility_select_the_newborn(cases):
    coupling = cases["attached"]["coupling"]
    child = cases["attached"]["preparation"]["children"][0]
    assert coupling["target"] == 0
    assert coupling["functional_links"] and coupling["bidirectional"]
    assert coupling["candidate_limit"] == 0
    assert coupling["candidate_mode"] == "sample"
    assert coupling["sampling_refresh_executed"]
    assert child not in coupling["sample_before_optional_refresh"]
    assert coupling["actual_candidate_sample"] == coupling["before"]["nodes"]
    target = coupling["readonly_kernel_proposal"]["target_proposals"][0]
    assert child in tuple(item["target"] for item in target["link_candidates"])
    assert coupling["compatible_existing_neighbors"] == (1, 7)
    assert coupling["compatibility_threshold"] == UM_COMPAT_THRESHOLD
    assert len(coupling["actual_new_edges"]) == 1
    u, v, data = coupling["actual_new_edges"][0]
    assert (u, v) == (0, child)
    assert UM_COMPAT_THRESHOLD < data["weight"] < 1
    assert coupling["actual_new_edge_phase_separations"][0] < (
        coupling["effective_phase_limit"]
    )


@pytest.mark.parametrize("name", CASES)
def test_public_coupling_commits_the_replayed_kernel_outputs(cases, name):
    coupling = cases[name]["coupling"]
    assert coupling["actual_admission"]["allowed"]
    before = coupling["before"]
    raw = coupling["raw_after_coupling"]
    proposal = coupling["readonly_kernel_proposal"]
    assert proposal["targets"] == (0,)
    assert raw["epi"] == before["epi"]
    index = {node: i for i, node in enumerate(raw["nodes"])}
    expected = {key: list(before[key]) for key in ("phase", "capacity", "pressure")}
    for update in proposal["node_updates"]:
        i = index[update["node"]]
        for field, key in (
            ("theta_after", "phase"), ("vf_after", "capacity"),
            ("dnfr_after", "pressure"),
        ):
            if update[field] is not None:
                expected[key][i] = update[field]
    for key, values in expected.items():
        assert raw[key] == tuple(values)
    assert tuple(
        (edge["left"], edge["right"], {"weight": edge["weight"]})
        for edge in proposal["edges"]
    ) == coupling["actual_new_edges"]
    assert raw["glyph_history"][0] == ("IL", "OZ", "THOL", "UM")


def test_stale_sample_is_a_distinct_no_attachment_control(cases):
    attached = cases["attached"]["coupling"]
    disabled = cases["links_disabled"]["coupling"]
    stale = cases["stale_sample"]["coupling"]
    child = cases["attached"]["preparation"]["children"][0]
    assert stale["functional_links"] and not disabled["functional_links"]
    assert not stale["sampling_refresh_executed"]
    assert child not in stale["actual_candidate_sample"]
    assert child in disabled["actual_candidate_sample"]
    assert stale["actual_new_edges"] == disabled["actual_new_edges"] == ()
    for key in ("epi", "capacity", "phase", "pressure"):
        assert stale["raw_after_coupling"][key] == disabled[
            "raw_after_coupling"
        ][key] == attached["raw_after_coupling"][key]
    assert stale["after_refresh"] == disabled["after_refresh"]


def test_attachment_has_exact_support_energy_cost_before_any_flow(cases):
    attached = cases["attached"]["coupling"]
    budget = attached["support_reset"]
    x = budget["before"]["epi"]
    weight = Fraction.from_float(attached["actual_new_edges"][0][2]["weight"])
    expected = weight * (x[0] - x[-1])**2 / 2
    assert budget["energy_change"] == budget["edge_energy_change"] == expected
    assert expected > 0
    assert budget["identity_residual"] == 0
    for name in ("links_disabled", "stale_sample"):
        control = cases[name]["coupling"]["support_reset"]
        assert control["energy_change"] == control["identity_residual"] == 0


def test_child_pressure_uses_singleton_epi_and_unweighted_capacity_channels(cases):
    coupling = cases["attached"]["coupling"]
    support = coupling["after_support"]
    state = support["snapshot"]
    weights = support["normalized_channel_weights"]
    expected = (
        weights["epi"] * (state["epi"][0] - state["epi"][-1])
        + weights["vf"] * (state["capacity"][0] - state["capacity"][-1])
    )
    assert state["support_neighbors"][-1] == (0,)
    assert support["exact_nonphase_pressure"][-1] == expected
    assert abs(support["exact_stored_minus_nonphase_pressure"][-1]) < 1e-15
    assert coupling["after_refresh"]["pressure"][-1] > 0.27
    # Default bidirectional UM also changes phases at the two old neighbors.
    assert abs(support["exact_stored_minus_nonphase_pressure"][1]) > 0.04
    for name in ("links_disabled", "stale_sample"):
        assert cases[name]["coupling"]["after_refresh"]["pressure"][-1] == 0


@pytest.mark.parametrize("name", CASES)
def test_actual_shared_euler_has_fixed_support_and_exact_energy_budget(cases, name):
    case = cases[name]
    previous = case["coupling"]["after_refresh"]
    for segment in case["postbirth_flow"]:
        before, after = segment["before"], segment["after_refresh"]
        assert before == previous
        assert after["time"] - before["time"] == segment["duration"] == 0.25
        for key in ("nodes", "edges", "capacity", "phase"):
            assert before[key] == after[key]
        assert segment["method"] == "euler"
        assert not segment["clipping_applied"]
        budget = segment["exact_euler_budget"]
        assert budget["identity_residual"] == 0
        assert budget["energy_change"] == (
            budget["drift_term"] + budget["quadratic_term"] + budget["defect_term"]
        )
        assert max(map(abs, budget["state_defect"])) < 2e-16
        assert budget["energy_change"] < 0
        previous = after
    assert previous == case["measured_endpoint_before_closure"]
    assert previous["time"] == 1.0
    initial_child = case["preparation"]["raw_after_birth"]["epi"][-1]
    if name == "attached":
        assert previous["epi"][-1] > initial_child + 0.12
    else:
        assert previous["epi"][-1] == initial_child


@pytest.mark.parametrize("name", CASES)
def test_complete_word_closes_after_measured_transport(cases, name):
    case = cases[name]
    assert case["word"] == WORD
    assert case["whole_word_validation"]["both_validators_passed"]
    assert case["closure"]["admission"]["allowed"]
    measured = case["measured_endpoint_before_closure"]
    closed = case["closure"]["after"]
    assert closed["glyph_history"][0] == ("IL", "OZ", "THOL", "UM", "SHA")
    assert closed["epi"] == measured["epi"]
    assert closed["time"] == measured["time"] == 1.0
    assert closed["capacity"][0] == measured["capacity"][0] * SHA_VF_FACTOR
    assert closed["capacity"][1:] == measured["capacity"][1:]

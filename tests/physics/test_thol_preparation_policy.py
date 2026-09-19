"""Finite causal preparation controls; no autonomous target-selection theorem."""

import json
import math
from copy import deepcopy
from fractions import Fraction

import pytest

from benchmarks.thol_birth_transport import prepare_birth_selection_source
from benchmarks.thol_pressure_feedback import _payload


@pytest.fixture(scope="module")
def study():
    from benchmarks.thol_preparation_policy import run_study

    return run_study()


def _case(study, mode, rotation=0):
    return next(
        case
        for case in study["cases"]
        if case["preparation"] == mode and case["rotation"] == rotation
    )


def _rename_state(source, mapping):
    """Transport declared node references, never numeric state coordinates."""
    result = deepcopy(source)

    def rename(node):
        return mapping[node]

    result["nodes"] = tuple(rename(node) for node in source["nodes"])
    for field in (
        "glyph_history",
        "physical_epi_history",
        "node_attributes",
        "ordered_neighbors",
    ):
        if field in source:
            result[field] = {
                rename(node): values for node, values in source[field].items()
            }
    if "ordered_neighbors" in source:
        result["ordered_neighbors"] = {
            rename(node): tuple(rename(neighbor) for neighbor in neighbors)
            for node, neighbors in source["ordered_neighbors"].items()
        }
    result["children"] = {
        rename(node): tuple(rename(child) for child in children)
        for node, children in source["children"].items()
    }
    result["hierarchy"] = {
        rename(node): [rename(child) for child in children]
        for node, children in source["hierarchy"].items()
    }
    result["edges"] = tuple(
        (rename(u), rename(v), attributes) for u, v, attributes in source["edges"]
    )
    if "node_sample" in source:
        result["node_sample"] = tuple(rename(node) for node in source["node_sample"])
    return result


@pytest.mark.parametrize(
    "options",
    (
        {"preparation": "all"},
        {"preparation": True},
        {"preparation": None},
        {"rotation": True},
        {"rotation": -1},
        {"rotation": 8},
        {"rotation": 1.0},
        {"rotation": "3"},
    ),
)
def test_invalid_preparation_does_not_silently_choose_a_control(options):
    with pytest.raises(ValueError):
        prepare_birth_selection_source(**options)


def test_declared_study_has_exactly_the_six_preparation_rotation_controls(study):
    assert tuple(
        (case["preparation"], case["rotation"]) for case in study["cases"]
    ) == (
        ("none", 0),
        ("none", 3),
        ("single_parent", 0),
        ("single_parent", 3),
        ("all_nodes", 0),
        ("all_nodes", 3),
    )


@pytest.mark.parametrize("mode", ("none", "single_parent", "all_nodes"))
@pytest.mark.parametrize("rotation", (0, 3))
def test_each_prefix_is_actual_public_execution_before_the_same_physical_partition(
    study,
    mode,
    rotation,
):
    branch = _case(study, mode, rotation)["explicit"]
    prep = branch["preparation_record"]
    order = tuple((node + rotation) % 8 for node in range(8))
    targets = () if mode == "none" else order[:1] if mode == "single_parent" else order
    assert prep["projection_node_order"] == prep["initial"]["nodes"] == order
    assert prep["initial_candidate_sample"] == order
    assert prep["initial"]["epi"] == (2.0, 0.5) * 4
    assert prep["initial"]["capacity"] == (1.0,) * 8
    assert prep["initial"]["phase"] == tuple(node * math.pi / 4 for node in range(8))
    assert all(not history for history in prep["initial"]["glyph_history"].values())
    assert all(
        not history for history in prep["initial"]["physical_epi_history"].values()
    )
    assert prep["prep_targets"] == targets
    assert prep["physical_steps"] == (0.25, 0.25)
    assert prep["segment_methods"] == ("euler", "euler")
    assert prep["clipping_applied"] == (False, False)
    assert prep["default_birth_threshold"] == 0.1
    assert branch["before"] == prep["before_birth"]
    assert branch["before"]["time"] == 0.5
    assert branch["policy"]["external_targets"] == targets
    assert branch["policy"]["external_target"] == (
        rotation if len(targets) == 1 else None
    )
    assert branch["policy"]["external_prefix"] == (() if not targets else ("IL", "OZ"))
    assert tuple(row["glyph"] for row in prep["actual_prefix"]) == (
        () if not targets else ("IL", "OZ")
    )
    for step in prep["actual_prefix"]:
        assert step["targets"] == targets
        assert all(admission["allowed"] for admission in step["admissions"])
        if mode == "single_parent":
            assert step["route"] == "direct_public"
            assert step["stage_result"] is None
        else:
            assert step["route"] == "simultaneous_public"
            assert step["stage_result"]["schedule"] == "two_phase_jacobi"
            assert step["stage_result"]["nodes_processed"] == 8
    for node in order:
        expected = ("IL", "OZ") if node in targets else ()
        assert prep["after_prefix"]["glyph_history"][node] == expected
        assert branch["before"]["glyph_history"][node] == expected
        history = branch["before"]["physical_epi_history"][node]
        assert tuple(row[0] for row in history) == (0.0, 0.25, 0.5)
        assert history[-1][1] == branch["before"]["epi"][order.index(node)]


@pytest.mark.parametrize("mode", ("none", "single_parent", "all_nodes"))
def test_rotation_transports_all_node_attributes_neighbors_histories_and_marks(
    study, mode
):
    original = _case(study, mode)["explicit"]
    moved = _case(study, mode, 3)["explicit"]
    mapping = {node: (node + 3) % 8 for node in range(8)}
    assert (
        _rename_state(original["source_projection"], mapping)
        == moved["source_projection"]
    )
    expected = deepcopy(original["eligibility"])
    expected["eligible_nodes"] = tuple(
        (node + 3) % 8 for node in expected["eligible_nodes"]
    )
    expected["candidates"] = tuple(
        {**row, "node": (row["node"] + 3) % 8} for row in expected["candidates"]
    )
    assert expected == moved["eligibility"]
    for checkpoint in ("initial", "after_prefix", "before_birth"):
        left = original["preparation_record"][checkpoint]
        right = moved["preparation_record"][checkpoint]
        for channel in ("epi", "phase", "pressure", "capacity", "cached_acceleration"):
            assert left[channel] == right[channel]
        for history in ("glyph_history", "physical_epi_history"):
            assert {
                (node + 3) % 8: value for node, value in left[history].items()
            } == right[history]


@pytest.mark.parametrize("mode", ("none", "single_parent", "all_nodes"))
def test_each_finite_continuation_transports_birth_lineage_or_runtime_state_exactly(
    study, mode
):
    original = _case(study, mode)
    moved = _case(study, mode, 3)
    mapping = {node: (node + 3) % 8 for node in range(8)}
    children = dict(moved["explicit"]["parent_children"])
    birth_mapping = {
        **mapping,
        **{
            child: children[mapping[parent]]
            for parent, child in original["explicit"]["parent_children"]
        },
    }
    assert (
        _rename_state(original["explicit"]["after"], birth_mapping)
        == moved["explicit"]["after"]
    )
    for left, right in zip(original["runtime"], moved["runtime"], strict=True):
        assert left["selector"] == right["selector"]
        assert {
            mapping[node]: glyph
            for node, glyph in left["actual_selector_proposals"].items()
        } == right["actual_selector_proposals"]
        for checkpoint in ("dispatch_boundary", "endpoint"):
            assert _rename_state(left[checkpoint], mapping) == right[checkpoint]
    control = next(
        row for row in study["rotation_controls"] if row["preparation"] == mode
    )
    assert control["source_projection_equal"] and control["eligibility_equal"]
    assert control["explicit_dispatch_state_equal"]
    assert all(
        row["proposals_equal"]
        and row["dispatch_state_equal"]
        and row["endpoint_state_equal"]
        for row in control["runtime"]
    )


@pytest.mark.parametrize("mode", ("none", "single_parent", "all_nodes"))
@pytest.mark.parametrize("rotation", (0, 3))
def test_full_eligibility_separates_observed_acceleration_from_actual_grammar_marks(
    study,
    mode,
    rotation,
):
    branch = _case(study, mode, rotation)["explicit"]
    observation = branch["eligibility"]
    expected = branch["preparation_record"]["prep_targets"]
    assert observation["eligible_nodes"] == expected
    assert observation["joint_stage_viable"] and observation["joint_error"] is None
    assert branch["dispatch_eligibility"] == observation
    assert branch["observer_state_projection_unchanged"]
    assert branch["observer_node_attributes_unchanged"]
    assert len(observation["candidates"]) == 8
    assert (
        tuple(row["node"] for row in observation["candidates"])
        == branch["before"]["nodes"]
    )
    for row in observation["candidates"]:
        assert row["grammar_allowed"] == (row["node"] in expected)
        assert row["eligible"] == row["grammar_allowed"]
        assert (
            row["proposal_valid"] and row["birth_proposed"] and row["threshold_crossed"]
        )
        assert (
            row["application_preconditions_passed"] and not row["optional_gate_enabled"]
        )
        assert row["depth_limit_reached"] is False
        acceleration = row["acceleration"]
        assert (
            acceleration["available"] and acceleration["source"] == "epi_time_history"
        )
        assert acceleration["time_basis"] == "physical_time"
        assert acceleration["current_endpoint_matches_state"] is True
        samples = tuple(
            tuple(Fraction.from_float(value) for value in pair)
            for pair in acceleration["samples"]
        )
        (t0, x0), (t1, x1), (t2, x2) = samples
        exact = 2 * ((x2 - x1) / (t2 - t1) - (x1 - x0) / (t1 - t0)) / (t2 - t0)
        assert abs(exact) > Fraction.from_float(observation["tau"])
        assert acceleration["value"] == pytest.approx(float(exact), abs=1e-14)


@pytest.mark.parametrize("mode", ("none", "single_parent", "all_nodes"))
@pytest.mark.parametrize("rotation", (0, 3))
def test_explicit_policy_dispatches_every_eligible_parent_without_a_tie_break(
    study, mode, rotation
):
    branch = _case(study, mode, rotation)["explicit"]
    parents = branch["eligibility"]["eligible_nodes"]
    pairs = branch["parent_children"]
    assert tuple(parent for parent, _ in pairs) == parents
    assert len(pairs) == {"none": 0, "single_parent": 1, "all_nodes": 8}[mode]
    children = tuple(child for _, child in pairs)
    assert len(set(children)) == len(children)
    assert set(branch["after"]["nodes"]) == set(branch["before"]["nodes"]) | set(
        children
    )
    assert branch["after"]["edges"] == branch["before"]["edges"]
    assert branch["after"]["time"] == branch["before"]["time"] == 0.5
    if not parents:
        assert branch["stage_result"] is None
        assert branch["after"] == branch["before"]
    else:
        assert branch["stage_result"]["nodes_processed"] == len(parents)
        assert branch["stage_result"]["schedule"] == "two_phase_jacobi"
        for record in branch["children"]:
            assert record["degree"] == 0
            assert record["node_data"]["parent_node"] == record["parent"]
        for parent, child in pairs:
            assert branch["after"]["children"][parent] == (child,)
            assert branch["after"]["glyph_history"][parent] == ("IL", "OZ", "THOL")


@pytest.mark.parametrize("mode", ("none", "single_parent", "all_nodes"))
@pytest.mark.parametrize("rotation", (0, 3))
def test_both_unchanged_selectors_replay_source_but_do_not_dispatch_thol(
    study, mode, rotation
):
    case = _case(study, mode, rotation)
    explicit = case["explicit"]
    assert tuple(row["selector"] for row in case["runtime"]) == (
        "default",
        "parametric",
    )
    for runtime in case["runtime"]:
        assert runtime["source_projection"] == explicit["source_projection"]
        assert runtime["eligibility"] == explicit["eligibility"]
        runtime_policy = deepcopy(explicit["policy"])
        runtime_policy["declared_graph_settings"]["GLYPH_SELECTOR_N_JOBS"] = 1
        assert runtime["policy"] == runtime_policy
        assert runtime["actual_selector_proposals"] == {
            node: "IL" for node in explicit["before"]["nodes"]
        }
        assert runtime["new_nodes_at_dispatch"] == ()
        boundary = runtime["dispatch_boundary"]
        assert boundary["nodes"] == explicit["before"]["nodes"]
        assert boundary["time"] == explicit["after"]["time"] == 0.5
        assert runtime["endpoint"]["time"] == 0.75
        for node, history in explicit["before"]["glyph_history"].items():
            assert boundary["glyph_history"][node] == history + ("IL",)
        assert "No equal-horizon endpoint comparison" in runtime["boundary_scope"]


def test_real_prefix_phase_effect_is_recorded_instead_of_assuming_unchanged_state(
    study,
):
    preparations = {
        mode: _case(study, mode)["explicit"]["preparation_record"]
        for mode in ("none", "single_parent", "all_nodes")
    }
    none = preparations["none"]
    assert none["after_prefix"]["phase"] == none["initial"]["phase"]
    for mode in ("single_parent", "all_nodes"):
        actual = preparations[mode]
        assert actual["initial"]["phase"][0] == 0.0
        assert actual["after_prefix"]["phase"][0] == 2 * math.pi
        assert actual["after_prefix"]["phase"] != actual["initial"]["phase"]
        # This finite result does not assert that arbitrary IL/OZ prefixes are flow-neutral.
        assert actual["before_birth"]["epi"] == none["before_birth"]["epi"]
        assert (
            actual["before_birth"]["physical_epi_history"]
            == none["before_birth"]["physical_epi_history"]
        )


def test_study_serializes_finite_evidence_without_claiming_spontaneous_selection(study):
    encoded = json.dumps(_payload(study), allow_nan=False)
    payload = json.loads(encoded)
    assert len(payload["cases"]) == 6
    limitations = " ".join(payload["limitations"])
    assert "supplied inputs" in limitations
    assert "not spontaneous generation" in limitations
    assert (
        "Initial checkerboard EPI, winding phase, support and capacity are supplied"
        in limitations
    )
    assert (
        "No UM, post-birth physical flow, restoration or sustained pattern"
        in limitations
    )

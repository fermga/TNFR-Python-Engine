"""A causal trigger is not sufficient for selector-driven public birth."""

from fractions import Fraction

import pytest

from benchmarks.selection_birth_closure import (
    SELECTORS, run_dispatch_case, run_selector_case, run_study,
)
from benchmarks.thol_birth_transport import prepare_birth_selection_source
from benchmarks.thol_pressure_feedback import _state
from tnfr.dynamics.runtime import step


@pytest.fixture(scope="module")
def study():
    return run_study()


def _control(study, dispatch, refresh):
    return next(row for row in study["dispatch_controls"]
                if row["dispatch"] == dispatch and row["refresh_before_flow"] == refresh)


def test_all_branches_replay_the_same_causal_source_without_synthetic_history(study):
    branches = (*study["selectors"], *study["dispatch_controls"], *study["transport_controls"])
    source = branches[0]["preparation"]["before_birth"]
    assert len(source["nodes"]) == 8
    assert source["glyph_history"] == {0: ("IL", "OZ"), **{i: () for i in range(1, 8)}}
    assert not any(source["children"].values())
    for branch in branches:
        prep = branch["preparation"]
        assert prep["before_birth"] == source
        assert prep["physical_steps"] == (0.25, 0.25)
        assert prep["physical_acceleration"]["observed_acceleration"] > (
            prep["default_birth_threshold"]
        )
        assert tuple(t for t, _ in source["physical_epi_history"][0]) == (0, 0.25, 0.5)


@pytest.mark.parametrize("index", (0, 1))
def test_actual_builtin_choices_and_commits_are_coherence_despite_trigger(study, index):
    row = study["selectors"][index]
    assert len(row["selection_contexts"]) == len(row["integration"]) == 1
    context = row["selection_contexts"][0]
    assert all(si > row["policy"]["resolved_selector_thresholds"]["si_hi"]
               for si in context["sense_index"])
    assert row["actual_selector_proposals"] == {i: "IL" for i in range(8)}
    assert context["preselection"]["base_choices"] == row["actual_selector_proposals"]
    assert len(row["endpoint"]["nodes"]) == 8
    assert not any(row["endpoint"]["children"].values())
    for node in range(8):
        assert row["endpoint"]["glyph_history"][node] == (
            row["preparation"]["before_birth"]["glyph_history"][node] + ("IL",)
        )
    assert row["decision_history_after_step"]["since_AL"] == {i: 1 for i in range(8)}
    assert row["decision_history_after_step"]["since_EN"] == {i: 1 for i in range(8)}
    assert row["integration"][0]["held_proposals_strictly_inside_hard_bounds"]


@pytest.mark.parametrize("name", tuple(SELECTORS))
def test_recording_delegate_preserves_unwrapped_runtime_endpoint(study, name):
    graph, _ = prepare_birth_selection_source()
    graph.graph.update(glyph_selector=SELECTORS[name](), GLYPH_SELECTOR_N_JOBS=1,
                       INTEGRATOR_METHOD="euler")
    step(graph, dt=0.25, use_Si=True, apply_glyphs=True)
    observed = next(row for row in study["selectors"] if row["selector"] == name)
    assert _state(graph) == observed["endpoint"]


def test_primitive_and_public_thol_share_pressure_but_only_public_creates_child(study):
    primitive = _control(study, "primitive", False)
    public = _control(study, "public", False)
    assert len(primitive["raw_after_dispatch"]["nodes"]) == 8
    raw = public["raw_after_dispatch"]
    assert len(raw["nodes"]) == 9
    assert len(raw["children"][0]) == 1
    assert raw["edges"] == public["before"]["edges"]
    assert raw["pressure"][-1] == 0
    assert raw["capacity"][-1] == 0.95
    for field in ("epi", "capacity", "phase", "pressure"):
        assert primitive["raw_after_dispatch"][field] == raw[field][:8]
    assert raw["glyph_history"][0] == ("IL", "OZ", "THOL")
    acceleration = Fraction.from_float(public["preparation"]["physical_acceleration"][
        "observed_acceleration"
    ])
    factor = Fraction.from_float(public["policy"]["resolved_thol_factors"]["THOL_accel"])
    actual = Fraction.from_float(raw["pressure"][0]) - Fraction.from_float(public["before"][
        "pressure"
    ][0])
    assert actual > 0
    assert abs(actual - factor * acceleration) < Fraction(1, 10**16)


@pytest.mark.parametrize("dispatch", ("primitive", "public"))
def test_only_held_pressure_retains_thol_increment_in_the_nodal_step(study, dispatch):
    baseline = _control(study, "none", False)
    held = _control(study, dispatch, False)
    refreshed = _control(study, dispatch, True)
    assert held["raw_after_dispatch"] == refreshed["raw_after_dispatch"]
    assert held["integration"]["before"]["pressure"] == held["raw_after_dispatch"]["pressure"]
    assert refreshed["integration"]["before"]["pressure"][:8] == baseline["before"]["pressure"]
    assert refreshed["integration"]["after"]["epi"][:8] == baseline["integration"]["after"]["epi"]
    assert held["integration"]["after"]["epi"][0] > baseline["integration"]["after"]["epi"][0]
    for row in (held, refreshed):
        assert max(map(abs, row["integration"]["exact_held_input_euler_residual"])) < 2e-16
        assert row["integration"]["held_proposals_strictly_inside_hard_bounds"]
        if dispatch == "public":
            assert row["integration"]["after"]["epi"][-1] == row["raw_after_dispatch"]["epi"][-1]


def test_positive_transport_control_requires_real_edge_and_candidate_inventory(study):
    controls = {row["case"]: row for row in study["transport_controls"]}
    attached = controls["attached"]
    assert len(attached["coupling"]["actual_new_edges"]) == 1
    assert attached["coupling"]["after_refresh"]["pressure"][-1] > 0
    assert attached["measured_endpoint_before_closure"]["epi"][-1] > (
        attached["preparation"]["raw_after_birth"]["epi"][-1]
    )
    for name in ("links_disabled", "stale_sample"):
        row = controls[name]
        assert row["coupling"]["actual_new_edges"] == ()
        assert row["coupling"]["after_refresh"]["pressure"][-1] == 0
        assert row["measured_endpoint_before_closure"]["epi"][-1] == (
            row["preparation"]["raw_after_birth"]["epi"][-1]
        )


def test_exact_symmetry_examples_do_not_promote_marked_runtime_to_unmarked_case(study):
    examples = study["detached_symmetry_examples"]
    uniform = examples["uniform_unmarked"]
    assert uniform["unique_equivariant_selection_obstructed"]
    assert uniform["candidate_orbits"] == (tuple(range(8)),)
    assert uniform["fixed_candidates"] == ()
    marked = examples["marked_parent"]
    assert not marked["unique_equivariant_selection_obstructed"]
    assert marked["fixed_candidates"] == (0, 4)  # Absence of obstruction is not uniqueness.
    assert len(marked["stabilizer_permutations"]) == 2


def test_invalid_branch_names_are_rejected():
    with pytest.raises(ValueError, match="selector"):
        run_selector_case("invented")
    with pytest.raises(ValueError, match="dispatch"):
        run_dispatch_case("invented", refresh=True)

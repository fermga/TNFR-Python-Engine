"""Joint-condition and prefix wiring controls; no retained response is read."""

from fractions import Fraction as Q
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.research import maintained_child_joint_window as owner
from tnfr.research.maintained_parent_thol import _signature

MINIMUM = Q.from_float(0.2)
THRESHOLD = Q.from_float(0.1)


@pytest.fixture(autouse=True)
def forbid_retained_response_read(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unit controls cannot read the retained trajectory")

    monkeypatch.setattr(owner, "load_sha_retention", forbidden)


def classify(**changes):
    arguments = dict(
        form=I(Q(1, 2)),
        pressure=I(Q(1, 4)),
        acceleration=I(2),
        minimum_form=MINIMUM,
        threshold=THRESHOLD,
        principal_margins=(Q(1), Q(2)),
    )
    arguments.update(changes)
    return owner.classify_joint_conditions(**arguments)


@pytest.mark.parametrize("acceleration", (I(1, 2), I(-2, -1)))
def test_absolute_acceleration_admits_either_strict_sign(acceleration):
    result = classify(acceleration=acceleration)
    assert result == owner.JointConditions(
        "admitted", "admitted", "admitted", "admitted", True
    )


def test_represented_form_equality_passes_without_replacing_it_by_decimal_value():
    assert MINIMUM > Q(1, 5)
    assert classify(form=I(MINIMUM)).joint
    below = classify(form=I(Q(1, 5)))
    assert below.form == "excluded" and not below.joint
    crossing = classify(form=I(MINIMUM - Q(1, 32), MINIMUM))
    assert crossing.form == "unresolved" and not crossing.joint


@pytest.mark.parametrize("acceleration", (I(THRESHOLD), I(-THRESHOLD)))
def test_acceleration_equality_is_excluded(acceleration):
    result = classify(acceleration=acceleration)
    assert result.acceleration == "excluded" and not result.joint


@pytest.mark.parametrize(
    "acceleration",
    (
        I(THRESHOLD, THRESHOLD + Q(1, 32)),
        I(-THRESHOLD - Q(1, 32), -THRESHOLD),
        I(-1, 1),
    ),
)
def test_large_magnitude_somewhere_does_not_admit_a_whole_acceleration_interval(
    acceleration,
):
    result = classify(acceleration=acceleration)
    assert acceleration.abs_max > THRESHOLD
    assert result.acceleration == "unresolved" and not result.joint


def test_acceleration_contained_in_closed_threshold_band_is_excluded():
    result = classify(acceleration=I(-THRESHOLD, THRESHOLD))
    assert result.acceleration == "excluded" and not result.joint
    zero = classify(threshold=0, acceleration=I(0))
    assert zero.acceleration == "excluded" and not zero.joint


@pytest.mark.parametrize(
    "pressure,status",
    ((I(0), "excluded"), (I(-1, 0), "excluded"), (I(0, 1), "unresolved")),
)
def test_positive_pressure_is_strict_over_the_whole_interval(pressure, status):
    result = classify(pressure=pressure)
    assert result.pressure == status and not result.joint


@pytest.mark.parametrize("margin", (Q(0), Q(-1)))
def test_failed_sufficient_principal_margin_abstains_instead_of_proving_exit(margin):
    result = classify(principal_margins=(Q(2), margin))
    assert result.identity == "unresolved" and not result.joint
    assert result.form == result.pressure == result.acceleration == "admitted"


@pytest.mark.parametrize(
    "argument,value",
    (
        ("form", True),
        ("pressure", False),
        ("acceleration", float("nan")),
        ("form", 0.25),
        ("minimum_form", True),
        ("threshold", False),
        ("threshold", float("inf")),
        ("minimum_form", "0.2"),
        ("principal_margins", (True,)),
        ("principal_margins", (float("nan"),)),
    ),
)
def test_invalid_authoritative_scalars_reject_before_classification(argument, value):
    with pytest.raises((TypeError, ValueError)):
        classify(**{argument: value})


@pytest.mark.parametrize(
    "changes",
    (
        {"minimum_form": -1},
        {"threshold": -1},
        {"principal_margins": ()},
    ),
)
def test_negative_policies_or_absent_original_margins_cannot_certify(changes):
    with pytest.raises(ValueError):
        classify(**changes)


def _synthetic_retained():
    """Minimal loader result for orchestration, deliberately no scientific data."""
    graph = nx.Graph()
    graph.add_nodes_from(range(20), EPI=0.4)
    graph.add_node(owner.CHILD, EPI=0.4, parent_node=2, glyph_history=[])
    graph.add_edge(2, owner.CHILD)
    graph.graph["_t"] = Q(7)
    prediction = SimpleNamespace(
        nodes=tuple(range(20)) + (owner.CHILD,),
        capacities=(Q(1),) * 21,
        edges=((2, 20),),
        conductances=(Q(1),),
        shifts=(0,),
        bounds=SimpleNamespace(horizon_remaining_margins=(I(-1),)),
    )
    steps = tuple(
        SimpleNamespace(
            index=index,
            time=Q(index, 512),
            duration=Q(1, 512),
            # Index sentinels identify the caller's complete tube. There is
            # intentionally no endpoint field to substitute for that tube.
            tube=(I(index),) * 42,
        )
        for index in range(256)
    )
    return SimpleNamespace(
        admitted=SimpleNamespace(graph=graph),
        prediction=prediction,
        summary={"status": "completed"},
        steps=steps,
        receipts=("synthetic receipt association",),
    )


def _wire_synthetic(monkeypatch, retained, observations):
    calls = []

    def loader(path, *, repository_root):
        assert path == "synthetic-evidence" and repository_root == "synthetic-root"
        return retained

    def bound(**arguments):
        index = int(arguments["epi"][0].lo)
        calls.append(index)
        assert arguments["epi"] == retained.steps[index].tube[:21]
        assert arguments["phases"] == retained.steps[index].tube[21:]
        for name in ("capacities", "edges", "conductances", "shifts"):
            assert arguments[name] == getattr(retained.prediction, name)
        assert arguments["target_index"] == 20
        form, pressure, acceleration, margin = observations(index)
        return SimpleNamespace(
            form=form,
            pressure=pressure,
            form_acceleration=acceleration,
            principal_margins=(margin,),
        )

    monkeypatch.setattr(owner, "load_sha_retention", loader)
    monkeypatch.setattr(owner, "bound_node_history_box", bound)
    return calls


def test_selection_stops_at_first_joint_whole_cell_and_keeps_child_grammar(monkeypatch):
    retained = _synthetic_retained()
    signature = _signature(retained.admitted.graph)
    observations = (
        (I(0), I(1), I(2), Q(1)),
        (I(1), I(0), I(2), Q(1)),
        (I(1), I(1), I(-2, -1), Q(1)),
    )
    calls = _wire_synthetic(monkeypatch, retained, lambda index: observations[index])
    result = owner.inspect_child_joint_window(
        "synthetic-evidence", repository_root="synthetic-root"
    )
    assert calls == [0, 1, 2]
    assert result.status == "joint_window_admitted"
    assert result.selected_cell is result.inspected_cells[-1]
    assert result.selected_cell.index == 2
    assert result.selected_cell.elapsed_start == Q(2, 512)
    assert result.selected_cell.elapsed_end == Q(3, 512)
    assert result.retained_cells == 256
    assert result.source_grammar_error_rules == ("U4b",)
    assert result.source_receipts == retained.receipts
    assert result.retrospective_observation
    assert not result.acquired_history
    assert not result.acquired_u4b_context
    assert not result.executed_event
    assert not result.full_thol_admission_established
    assert _signature(retained.admitted.graph) == signature


def test_favorable_different_cells_cannot_be_combined_into_joint_admission(monkeypatch):
    retained = _synthetic_retained()

    def observations(index):
        if index % 2 == 0:
            return I(1), I(0), I(2), Q(1)
        return I(0), I(1), I(2), Q(1)

    calls = _wire_synthetic(monkeypatch, retained, observations)
    result = owner.inspect_child_joint_window(
        "synthetic-evidence", repository_root="synthetic-root"
    )
    assert calls == list(range(256))
    assert result.status == "no_joint_cell_certified"
    assert result.selected_cell is None
    assert len(result.inspected_cells) == 256
    assert all(not cell.conditions.joint for cell in result.inspected_cells)


def test_nonpositive_box_and_global_margins_prevent_joint_selection(monkeypatch):
    retained = _synthetic_retained()
    _wire_synthetic(monkeypatch, retained, lambda index: (I(1), I(1), I(2), Q(0)))
    result = owner.inspect_child_joint_window(
        "synthetic-evidence", repository_root="synthetic-root"
    )
    assert result.selected_cell is None
    assert all(
        cell.conditions.identity == "unresolved" for cell in result.inspected_cells
    )


@pytest.mark.parametrize("incomplete", ("count", "status", "parent"))
def test_incomplete_or_different_source_rejects_before_observation(
    monkeypatch, incomplete
):
    retained = _synthetic_retained()
    if incomplete == "count":
        retained.steps = retained.steps[:-1]
    elif incomplete == "status":
        retained.summary["status"] = "resource_budget_exhausted"
    else:
        retained.admitted.graph.nodes[owner.CHILD]["parent_node"] = 3
    calls = _wire_synthetic(
        monkeypatch, retained, lambda index: pytest.fail("unexpected box observation")
    )
    with pytest.raises(ValueError, match="complete actual certificate"):
        owner.inspect_child_joint_window(
            "synthetic-evidence", repository_root="synthetic-root"
        )
    assert calls == []

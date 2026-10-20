"""Independent pre-event bounds; these tests never execute the reserved reset."""

from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp
import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.mathematics._rational_interval import I
from tnfr.operators import metabolism
from tnfr.operators.definitions import SelfOrganization
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.research.maintained_parent_thol_prediction import (
    NUMERICAL_ALLOWANCE,
    _metabolic_prediction,
    _unit_clip,
    predict_thol_reset,
)
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.sdk._state import copy_graph_state
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_event_or_metabolic_proposal(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("prospective bounds must not evaluate a THOL proposal")

    monkeypatch.setattr(SelfOrganization, "_prepare_execution", forbidden)
    monkeypatch.setattr(metabolism, "capture_network_signals", forbidden)
    monkeypatch.setattr(metabolism, "compose_subepi_amplitude", forbidden)


@pytest.fixture
def graph():
    response = decode_exact_tree(
        json_loads(
            (ROOT / "docs/assets/local_capacity_history/response.json").read_bytes()
        )
    )
    return restore_recorded_graph_state(response["endpoint"]).graph


def _mpq(value):
    return mp.mpf(value.numerator) / value.denominator


def _encloses(interval, value):
    assert _mpq(interval.lo) <= value <= _mpq(interval.hi)


def test_independent_metabolic_trigonometry_bounds_high_precision_reference():
    parent, phase = Q(3, 8), Q(1, 2)
    forms, phases = (Q(1, 4), Q(1, 2)), (Q(0), Q(1))
    result = _metabolic_prediction(parent, phase, forms, phases)
    assert result.mean_neighbor_epi == Q(3, 8)
    assert result.epi_gradient == 0
    with mp.workdps(95):
        variance = 1 - mp.cos(mp.mpf(1) / 2)
        _encloses(result.phase_variance, variance)
        expected = _mpq(parent) * _mpq(Q(0.3)) + variance * _mpq(
            Q(1 / (4 * float(mp.pi)))
        )
        _encloses(result.child_form, expected)
        _encloses(result.mean_cos, (1 + mp.cos(1)) / 2)
        _encloses(result.mean_sin, mp.sin(1) / 2)


def test_unit_clip_is_the_monotone_image_at_both_rails():
    assert _unit_clip(I(-2, -1)) == I(0)
    assert _unit_clip(I(2, 3)) == I(1)
    assert _unit_clip(I(-1, Q(1, 4))) == I(0, Q(1, 4))
    assert _unit_clip(I(Q(3, 4), 2)) == I(Q(3, 4), 1)


@pytest.mark.parametrize(
    "parent,phase,forms,phases",
    [
        (True, 0, (0,), (0,)),
        (Q(1, 3), 0, (0,), (0,)),
        (0, 0, (), ()),
        (0, 0, (0,), (0, 1)),
        (0, 0, (0,), (4,)),
        (0, 0, (-1,), (0,)),
        (0, 0, (0,), (float("nan"),)),
    ],
)
def test_metabolic_admission_precedes_arithmetic(parent, phase, forms, phases):
    with pytest.raises((TypeError, ValueError)):
        _metabolic_prediction(parent, phase, forms, phases)


def test_retained_endpoint_prediction_is_detached_and_has_zero_edge_work(graph):
    before = copy_graph_state(graph)
    result = predict_thol_reset(graph)
    assert graph.graph == before.graph
    assert dict(graph.nodes) == dict(before.nodes)
    assert dict(graph.edges) == dict(before.edges)
    assert result.successor_nodes == (*tuple(range(20)), "2_sub_0")
    assert result.expected_child_count == 1 and result.child_degree == 0
    assert result.source_edges == result.successor_edges
    assert len(result.successor_edges) == 45
    assert result.child_capacity == Q(121.6)
    assert result.child_phase == result.parent_phase
    assert result.child_pressure == 0
    assert result.child_hierarchy_path == (2,)
    assert result.child_record_operator_step == 2
    assert result.expected_parent_glyph_history == ("VAL", "THOL")
    assert result.expected_parent_u2_debt == 0
    assert result.event_storage_jump == 0
    assert result.old_component_mean_jumps == (0, 0)
    assert result.global_positive_mean_metric_available is False
    assert result.metabolic.child_form.lo > Q(1, 8)
    assert result.metabolic.child_form.hi < Q(1, 6)
    assert result.child_form_bounds.lo == (
        result.metabolic.child_form.lo - NUMERICAL_ALLOWANCE
    )
    assert result.child_form_bounds.hi == (
        result.metabolic.child_form.hi + NUMERICAL_ALLOWANCE
    )
    assert result.pressure_square_storage_jump > 0
    assert result.raw_pressure_ideal == (
        result.parent_pressure + result.pressure_gain * result.observed_acceleration
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("THOL_METABOLIC_ENABLED", False),
        ("BIFURCATION_THRESHOLD_TAU", 0.01),
        ("VALIDATE_OPERATOR_PRECONDITIONS", 1),
        ("EDGE_AWARE_ENABLED", 0),
        ("GRAMMAR_REJECTION_MODE", "substitute"),
        ("_t", 7.0),
        ("EPI_MIN", True),
        ("hierarchy", {2: []}),
        ("GLYPH_FACTORS", {"VAL_scale": 128.0, "THOL_accel": 1.0}),
    ],
)
def test_selected_configuration_cannot_silently_change(graph, field, value):
    graph.graph[field] = value
    with pytest.raises((TypeError, ValueError)):
        predict_thol_reset(graph)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sub_nodes", ["absent"]),
        ("sub_epis", [{}]),
        ("_bifurcation_level", True),
        ("_operator_step", True),
        ("_grammar_u2_debt", True),
        ("glyph_history", ["OZ"]),
    ],
)
def test_no_hierarchy_or_prefix_substitution(graph, field, value):
    graph.nodes[2][field] = value
    with pytest.raises((TypeError, ValueError)):
        predict_thol_reset(graph)


def test_invalid_authoritative_pressure_does_not_fall_back(graph):
    set_attr(graph.nodes[2], ALIAS_DNFR, float("nan"))
    graph.nodes[2]["dnfr"] = 0.1
    with pytest.raises(ValueError):
        predict_thol_reset(graph)


def test_replaced_history_times_are_rejected_even_with_live_form_endpoint(graph):
    graph.nodes[2]["epi_time_history"] = tuple(
        (time - 1, form) for time, form in graph.nodes[2]["epi_time_history"]
    )
    with pytest.raises(ValueError, match="original sample times"):
        predict_thol_reset(graph)

"""Admission and whole-window semantics of conservative regional winding."""

from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_conservative_winding_entry


def _source(*, reversed_order=False, label=lambda i: i):
    graph = nx.Graph()
    graph.add_nodes_from(
        label(i) for i in (reversed(range(11)) if reversed_order else range(11))
    )
    for offset in (0, 5):
        graph.add_edges_from(
            (label(offset + i), label(offset + (i + 1) % 5)) for i in range(5)
        )
    graph.add_edges_from((label(i), label(j)) for i, j in ((0, 10), (10, 5), (1, 6)))
    graph.graph["GAMMA"] = {"type": "none"}
    for i in range(11):
        graph.nodes[label(i)].update(
            EPI=24 if i == 10 else -24 if i == 1 else 0, theta=0, nu_f=1
        )
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def _assess(source, **kwargs):
    defaults = dict(
        cycle=(5, 6, 7, 8, 9),
        scaled_window=(Q(1, 4), Q(1, 3)),
        edge_turn_offsets=(1, 0, 0, 0, 0),
    )
    return certify_sine_conservative_winding_entry(source, **(defaults | kwargs))


@pytest.fixture(scope="module")
def source():
    return _source()


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def test_cached_source_fields_do_not_certify_the_whole_window(source, report):
    forged = replace(
        source, form_rates=(I(999),) * 11, phase_rates=(I(999),) * 11, storage=I(0)
    )
    actual = _assess(forged)
    assert actual.initial_total_storage == report.initial_total_storage
    assert actual.initial_phase_velocity == report.initial_phase_velocity
    assert actual.cycle_raw_gap_bounds == report.cycle_raw_gap_bounds


def test_common_origins_leave_entry_evidence_unchanged(source, report):
    shifted = replace(
        source, epi=tuple(x + Q(7, 3) for x in source.epi), phase=(Q(8, 7),) * 11
    )
    actual = _assess(shifted)
    assert actual.acquisition_certified
    assert actual.initial_total_storage == report.initial_total_storage
    assert actual.cycle_principal_gap_bounds == report.cycle_principal_gap_bounds
    for speed, before, after in zip(
        report.initial_phase_velocity,
        report.whole_window_phase_bounds,
        actual.whole_window_phase_bounds,
    ):
        endpoints = tuple(speed * time for time in report.scaled_window)
        for exact in (min(endpoints) - Q(1, 9), max(endpoints) + Q(1, 9)):
            assert before.contains(exact)
            assert after.contains(exact + Q(8, 7))


def test_cycle_orientation_changes_the_period_sign(source, report):
    reversed_cycle = _assess(
        source, cycle=(5, 9, 8, 7, 6), edge_turn_offsets=(0, 0, 0, 0, -1)
    )
    assert reversed_cycle.certified_winding == -report.certified_winding
    assert reversed_cycle.acquisition_certified


def test_labels_and_node_order_are_not_a_source_of_orientation(report):
    mapped = _source(reversed_order=True, label=lambda i: f"node-{i}")
    actual = _assess(mapped, cycle=tuple(f"node-{i}" for i in range(5, 10)))
    assert actual.cycle_raw_gap_bounds == report.cycle_raw_gap_bounds
    assert actual.certified_winding == report.certified_winding
    assert (
        tuple(reversed(actual.initial_phase_velocity)) == report.initial_phase_velocity
    )


def test_failed_branch_inclusion_abstains_without_claiming_failed_dynamics(source):
    actual = _assess(source, scaled_window=(Q(1, 4), 1))
    assert actual.status == "unavailable"
    assert actual.certified_winding is None
    assert not actual.acquisition_certified
    assert "whole_window_strict_branch_inclusion_unresolved" in actual.unresolved


def test_declared_integer_offsets_must_pass_whole_time_branch_check(source):
    actual = _assess(source, edge_turn_offsets=(0, 0, 0, 0, 0))
    assert actual.status == "unavailable"
    assert actual.certified_winding is None


def test_zero_period_is_not_nonzero_acquisition(source):
    flat = replace(source, epi=(Q(0),) * 11)
    actual = _assess(flat, edge_turn_offsets=(0,) * 5)
    assert actual.initial_phase_velocity == (0,) * 11
    assert actual.certified_winding == 0
    assert not actual.acquisition_certified
    assert actual.status == "unavailable"


@pytest.mark.parametrize(
    "window",
    [(True, 1), (0, 1), (1, 1), (2, 1), (Q(1, 4), float("inf")), ("1/4", 1), (1,)],
)
def test_invalid_clocks_reject_before_bounds(source, window):
    with pytest.raises((ValueError, TypeError)):
        _assess(source, scaled_window=window)


@pytest.mark.parametrize("offsets", [(True, 0, 0, 0, 0), (1, 0), (1.0, 0, 0, 0, 0)])
def test_offset_admission_rejects_coercion(source, offsets):
    with pytest.raises(ValueError):
        _assess(source, edge_turn_offsets=offsets)


@pytest.mark.parametrize(
    "field,value",
    [
        ("capacity", (True,) * 11),
        ("capacity", (Q(0),) + (Q(1),) * 10),
        ("epi", (True,) + (Q(0),) * 10),
        ("phase", (Q(1),) + (Q(0),) * 10),
        ("degrees", (1,) * 11),
    ],
)
def test_source_primitives_reject_invalid_or_unsupported_premises(source, field, value):
    with pytest.raises((ValueError, TypeError)):
        _assess(replace(source, **{field: value}))


def test_receiver_local_seed_is_outside_this_environmental_contract(source):
    forms = list(source.epi)
    forms[7] = Q(1)
    with pytest.raises(ValueError, match="constant initial form"):
        _assess(replace(source, epi=tuple(forms)))


def test_positive_loss_cannot_reuse_the_conservative_certificate(source):
    model = RelationalExchangeModel(
        1, epi_weight=Q(1, 2), phase_weight=Q(1, 2), phase_domain="regular"
    )
    with pytest.raises(ValueError, match="zero loss"):
        _assess(replace(source, reference_model=model))


@pytest.mark.parametrize("cycle", [(5, 6, 5), (5, 7, 9), (5, 6, "missing")])
def test_receiver_cycle_must_be_a_simple_existing_chain(source, cycle):
    with pytest.raises(ValueError):
        _assess(source, cycle=cycle)

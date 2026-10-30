"""Exact topology/root admission and independent native equilibrium checks."""

import math
from fractions import Fraction as Q

import networkx as nx
import pytest

from benchmarks import relational_return_geometry as study
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.mathematics._rational_interval import I
from tnfr.physics.phase_cycle_geometry import reconstruct_phase_cycle_state


@pytest.fixture(scope="module")
def report():
    return study.analyze_return_geometry()


def test_root_enclosure_has_strict_certified_signs_and_requested_width(report):
    root = report.opposite_root
    assert Q(1, 12) < root.lower < root.upper < Q(1, 8)
    assert root.lower_residual.lo > 0 > root.upper_residual.hi
    assert root.upper - root.lower == Q(1, 24 * 2**root.refinements)
    assert study.root_residual(I(root.lower, root.upper)).contains(0)


def test_added_return_changes_cycle_space_without_changing_nodes(report):
    extension = report.extension
    assert extension.before.nodes == extension.after.nodes == tuple(range(11))
    assert (extension.before.cycle_rank, extension.after.cycle_rank) == (2, 3)
    assert extension.after.edges[extension.added_edge_index] == (1, 6)
    assert len(extension.before.bridge_edge_indices) == 2
    assert extension.after.bridge_edge_indices == ()


@pytest.mark.parametrize("sector,right_period", (("same", 1), ("opposite", -1)))
def test_independent_winding_and_sine_balance_from_reconstructed_phases(
    report, sector, right_period
):
    witness = getattr(report, sector)
    turns = witness.state.nodal_turns
    cycles = ((0, 1, 2, 3, 4, 0), (5, 6, 7, 8, 9, 5), (0, 10, 5, 6, 1, 0))
    periods = []
    for cycle in cycles:
        deltas = [(turns[j] - turns[i]) % 1 for i, j in zip(cycle, cycle[1:])]
        periods.append(sum(delta - (delta > Q(1, 2)) for delta in deltas))
    assert tuple(periods) == witness.named_periods == (1, right_period, 0)
    sine = [0.0] * 11
    for i, j in witness.state.geometry.edges:
        term = math.sin(math.tau * float(turns[j] - turns[i]))
        sine[i] += term
        sine[j] -= term
    assert max(map(abs, sine)) < 4e-13
    assert witness.maximum_native_rate < 1e-13
    assert witness.native_field.phase_rate == (0.0,) * 11
    assert witness.native_field.form_storage == 0


def test_opposite_root_redistributes_geometry_but_not_mixed_winding(report):
    same, opposite = report.same, report.opposite
    assert same.connecting_turns == (0, 0, 0)
    assert opposite.connecting_turns == (2 * report.opposite_root.midpoint / 3,) * 3
    assert opposite.minimum_acute_margin_turns == report.opposite_root.midpoint / 4
    assert same.state.sine_balance_status == "proved_by_odd_cancellation"
    assert opposite.state.sine_balance_status == "unresolved"
    assert opposite.status == "rational_midpoint_not_an_exact_equilibrium"
    assert opposite.native_field.phase_storage > same.native_field.phase_storage
    assert float(same.native_field.phase_storage) == pytest.approx(
        10 * (1 - math.cos(math.tau / 5)), abs=3e-14
    )
    excess = report.opposite_phase_storage_excess
    assert excess.lo > 0
    assert excess.contains(
        opposite.native_field.phase_storage - same.native_field.phase_storage
    )


def test_old_opposite_twists_cannot_be_copied_to_the_return_support(report):
    turns = tuple(
        Q(node % 5, 5) * (1 if node < 5 else -1) if node < 10 else Q(0)
        for node in range(11)
    )
    geometry = report.extension.after
    edges = tuple(
        (turns[j] - turns[i] + Q(1, 2)) % 1 - Q(1, 2) for i, j in geometry.edges
    )
    assert abs(edges[geometry.edges.index((1, 6))]) == Q(2, 5)
    with pytest.raises(ValueError, match="strictly acute"):
        reconstruct_phase_cycle_state(geometry, edge_turns=edges)
    native = nx.Graph()
    native.add_nodes_from(geometry.nodes)
    native.add_edges_from(geometry.edges, weight=1.0)
    for node in native:
        native.nodes[node].update(
            EPI=0.0, nu_f=1.0, theta=math.tau * float(turns[node])
        )
    for domain, reason in (("acute", "acute"), ("positive_resultant", "positive")):
        with pytest.raises(ValueError, match=reason):
            evaluate_relational_exchange(
                native, model=RelationalExchangeModel(1.0, phase_domain=domain)
            )


@pytest.mark.parametrize("value", (True, False, 0, -1, 65, 2.0, "40", None))
def test_refinement_admission_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="refinements"):
        study.enclose_opposite_root(refinements=value)


def test_uncertain_midpoint_sign_does_not_silently_choose_a_half(monkeypatch):
    actual = study.root_residual

    def unresolved(turns):
        return actual(turns) if turns in (Q(1, 12), Q(1, 8)) else I(-1, 1)

    monkeypatch.setattr(study, "root_residual", unresolved)
    with pytest.raises(ArithmeticError, match="refinement sign is unresolved"):
        study.enclose_opposite_root(refinements=1)

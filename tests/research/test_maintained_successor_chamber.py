"""Independent branch, cycle and retained-source controls without execution."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_VF
from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions import Coupling, SelfOrganization
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research import maintained_successor_chamber as owner
from tnfr.research.artifact_io import file_receipt
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_thol"
RING = ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))


def _ring_phases():
    return tuple(2 * pi_interval() * index / 5 for index in range(5))


def _incidence(edges, coefficients, count):
    result = [0] * count
    for (left, right), value in zip(edges, coefficients, strict=True):
        result[left] -= value
        result[right] += value
    return tuple(result)


def test_wound_five_cycle_has_oriented_full_branch_inventory():
    report = owner._certify_chamber(_ring_phases(), RING)
    assert report.shifts == (0, 1, 0, 0, 0)
    assert report.minimum_acute_margin.contains(pi_interval() / 10)
    tree, cycles = owner._cycle_inventory(5, RING, report.shifts)
    assert tree == (0, 1, 2, 3)
    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle.chord_edge_index == 4
    assert cycle.edge_coefficients == (1, -1, 1, 1, 1)
    assert _incidence(RING, cycle.edge_coefficients, 5) == (0,) * 5
    assert cycle.branch_sum == -1
    assert cycle.winding == 1
    gap_sum = sum(
        coefficient * difference
        for coefficient, difference in zip(
            cycle.edge_coefficients, report.differences, strict=True
        )
    )
    assert gap_sum.contains(2 * pi_interval())


def test_integer_lift_changes_change_edge_shifts_but_preserve_cycle_class():
    phases = _ring_phases()
    lifts = (7, -4, 0, 3, -2)
    original = owner._certify_chamber(phases, RING)
    lifted = owner._certify_chamber(
        tuple(value + 2 * pi_interval() * lift for value, lift in zip(phases, lifts)),
        RING,
    )
    assert lifted.shifts == tuple(
        shift + lifts[right] - lifts[left]
        for (left, right), shift in zip(RING, original.shifts, strict=True)
    )
    before = owner._cycle_inventory(5, RING, original.shifts)[1]
    after = owner._cycle_inventory(5, RING, lifted.shifts)[1]
    assert tuple(cycle.winding for cycle in after) == tuple(
        cycle.winding for cycle in before
    )


def test_acute_branch_resolves_large_integer_lifts_without_float_conversion():
    report = owner._certify_chamber((I(0), 400 * pi_interval() + Q(1, 10)), ((0, 1),))
    assert report.shifts == (200,)
    assert report.differences[0].contains(Q(1, 10))
    assert report.minimum_acute_margin.lo > 1


@pytest.mark.parametrize(
    "phase",
    (pi_interval() / 2, -pi_interval() / 2, pi_interval(), I(Q(3, 2), Q(8, 5))),
)
def test_nonacute_or_unresolved_entire_interval_rejects(phase):
    with pytest.raises(ValueError, match="acute"):
        owner._certify_chamber((I(0), phase), ((0, 1),))


@pytest.mark.parametrize("phase", (True, 0.25, float("nan")))
def test_private_interval_kernel_requires_exact_or_admitted_coordinates(phase):
    with pytest.raises(TypeError):
        owner._certify_chamber((I(0), phase), ((0, 1),))


@pytest.mark.parametrize(
    "edges",
    (
        ((1, 0),),
        ((0, 0),),
        ((0, True),),
        ((0, 3),),
        ((0, 1), (0, 1)),
        ((1, 2), (0, 1)),
    ),
)
def test_support_rejects_invalid_aliasing_and_noncanonical_order(edges):
    with pytest.raises(ValueError, match="edges"):
        owner._certify_chamber((I(0), I(0), I(0)), edges)


def test_cycle_inventory_rejects_disconnected_support():
    with pytest.raises(ValueError, match="connected"):
        owner._cycle_inventory(3, ((0, 1),), (0,))


@pytest.mark.parametrize("shifts", ((), (False,), (Q(0),)))
def test_cycle_inventory_requires_every_exact_integer_branch(shifts):
    with pytest.raises(ValueError, match="exact integer"):
        owner._cycle_inventory(2, ((0, 1),), shifts)


def test_fundamental_cycles_are_independent_by_their_distinct_chord_columns():
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    tree, cycles = owner._cycle_inventory(4, edges, (0,) * 6)
    assert len(tree) == 3
    assert len(cycles) == 3
    for row, cycle in enumerate(cycles):
        assert _incidence(edges, cycle.edge_coefficients, 4) == (0,) * 4
        assert tuple(
            cycle.edge_coefficients[other.chord_edge_index] for other in cycles
        ) == tuple(int(column == row) for column in range(3))


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("chamber admission must not execute events, producers or flow")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (_coupling_stage_kernel, "propose_coupling_target"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
            (word_execution, "execute_network_operator_stage"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (maintained_parent_thol_execution, "execute_network_operator_stage"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "rk4_coupled_step"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
            (Coupling, "_execute"),
            (SelfOrganization, "_prepare_execution"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def study():
    paths = tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    source = load_thol_successor(EVIDENCE, repository_root=ROOT)
    signature = _signature(source.graph)
    report = owner.predict_successor_chamber(source.graph)
    yield source, report
    assert _signature(source.graph) == signature
    assert tuple(file_receipt(path) for path in paths) == receipts


def test_retained_source_has_all_53_acute_edges_and_33_independent_cycles(study):
    source, report = study
    assert len(source.graph.edges) == 45
    assert source.graph.degree("2_sub_0") == 0
    assert report.reset.nodes == tuple(range(20)) + ("2_sub_0",)
    assert len(report.edges) == 53
    assert report.edges == tuple(sorted(set(report.edges)))
    assert len(report.tree_edge_indices) == 20
    assert report.cycle_rank == 53 - 21 + 1 == 33
    assert report.ideal.minimum_acute_margin.lo > Q(1, 100)
    assert report.conditional.minimum_acute_margin.lo > Q(1, 100)
    assert report.ideal.shifts == report.conditional.shifts
    assert all(value.lo > 0 for value in report.conditional_conductances)
    assert all(value.lo > 0 for value in report.conditional_capacities)
    for row, cycle in enumerate(report.cycles):
        assert _incidence(report.edges, cycle.edge_coefficients, 21) == (0,) * 21
        assert tuple(
            cycle.edge_coefficients[other.chord_edge_index] for other in report.cycles
        ) == tuple(int(column == row) for column in range(33))
        assert cycle.branch_sum == sum(
            coefficient * shift
            for coefficient, shift in zip(
                cycle.edge_coefficients, report.ideal.shifts, strict=True
            )
        )
        assert cycle.winding == -cycle.branch_sum


def test_conditional_envelope_widens_only_changed_outputs(study):
    _, report = study
    epsilon = report.numerical_allowance
    assert epsilon == Q(1, 2**40)
    for index, (ideal, conditional) in enumerate(
        zip(report.ideal.phases, report.conditional.phases, strict=True)
    ):
        if index in report.reset.phase_changed_indices:
            assert conditional.lo == ideal.lo - epsilon
            assert conditional.hi == ideal.hi + epsilon
        else:
            assert conditional == ideal
    for index, (ideal, conditional) in enumerate(
        zip(report.reset.capacity_after, report.conditional_capacities, strict=True)
    ):
        if index in report.reset.capacity_changed_indices:
            assert conditional.lo == ideal.lo - epsilon
            assert conditional.hi == ideal.hi + epsilon
        else:
            assert conditional == ideal
    for edge, ideal, conditional in zip(
        report.edges,
        report.conductances,
        report.conditional_conductances,
        strict=True,
    ):
        if edge in report.reset.new_edges:
            assert conditional.lo == ideal.lo - epsilon
            assert conditional.hi == ideal.hi + epsilon
        else:
            assert conditional == ideal


def test_child_is_a_leaf_on_zero_branch_and_full_inventory_is_immutable(study):
    _, report = study
    child_edges = tuple(edge for edge in report.edges if 20 in edge)
    assert child_edges == ((2, 20),)
    index = report.edges.index((2, 20))
    assert report.ideal.shifts[index] == report.conditional.shifts[index] == 0
    with pytest.raises(FrozenInstanceError):
        report.ideal.shifts = ()


def test_cached_verdict_cannot_replace_live_primitive_readmission(study):
    source, report = study
    changed = deepcopy(source.graph)
    changed.graph["_successor_chamber"] = report
    set_attr(changed.nodes[2], ALIAS_VF, True)
    with pytest.raises((TypeError, ValueError)):
        owner.predict_successor_chamber(changed)

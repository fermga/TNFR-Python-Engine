"""Finite cell adjacency is stronger than a static pressure convex balance."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import (
    NODAL_REMAINDER_DENOMINATOR_BITS, NodalRemainderState, advance_nodal_remainder,
)
from tnfr.physics.c6_carried_cell_graph import derive_c6_carried_cell_graph
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice
from tnfr.physics.nodal_remainder import (
    derive_nodal_remainder_cell_horizon, derive_nodal_remainder_itinerary,
)

PHASE = tuple(map(float.fromhex, (
    "0x1.0b8fb3e3956cbp-55", "0x1.0c152382d7365p+0", "0x1.0c152382d7365p+1",
    "0x1.921fb54442d18p+1", "0x1.0c152382d7365p+2", "0x1.4f1a6c638d03fp+2",
)))
OFFSETS = (
    (-6, -1, 2, 2, 8, -7), (-5, 2, 4, -2, 6, -7),
    (-4, -2, 0, 2, 6, -4), (-4, -1, 2, -2, 8, -5),
    (-3, 0, 2, -2, 8, -7), (-2, -1, 4, -2, 6, -7),
    (-2, 2, 0, -2, 6, -6),
)
ROWS = tuple(tuple(float(F(1, 2) + F(v, 2**54)) for v in row) for row in OFFSETS)


@pytest.fixture(scope="module")
def source():
    return derive_c6_pressure_lattice(
        phase=PHASE, epi_weight=CHANNEL_WEIGHT_SECONDARY, phase_weight=CHANNEL_WEIGHT_PRIMARY,
    )


@pytest.fixture(scope="module")
def graph(source):
    return derive_c6_carried_cell_graph(source, epi_states=ROWS, timestep=.0625)


def test_seven_static_compensators_cannot_form_a_closed_temporal_class(graph):
    edges = tuple((i, j) for i, row in enumerate(graph.adjacency) for j, edge in enumerate(row) if edge)
    assert edges == ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (5, 5), (6, 6))
    assert graph.nonself_topological_order == (0, 1, 2, 3, 5, 4, 6)
    assert tuple(item.max_unchanged_steps for item in graph.maximal_residence) == (76, 50, 38, 50, 39, 29, 49)
    assert graph.family_exit_step_bounds == (77, 51, 39, 51, 40, 70, 50)
    assert graph.maximum_steps_until_family_exit_or_band_failure == 77
    assert graph.finite_family_escape_certified and graph.carried_cycle_in_family_excluded
    assert not graph.whole_band_exit_certified
    assert not graph.future_runtime_certified
    assert not graph.reachable_from_saved_state_certified
    with pytest.raises(FrozenInstanceError):
        graph.timestep = 0.


def test_every_existential_edge_replays_with_its_own_legal_carry(graph):
    for i, row in enumerate(graph.adjacency):
        for j, edge in enumerate(row):
            witness = graph.edge_witnesses[i][j]
            if edge:
                assert witness is not None
                observed = advance_nodal_remainder(
                    witness, timestep=graph.timestep, capacity=(1.,) * 6, pressure=graph.pressures[i],
                )
                assert witness.epi == ROWS[i]
                assert observed.after.epi == ROWS[j]
                assert graph.infeasible_coordinate[i][j] is None
            else:
                assert witness is None
                assert 0 <= graph.infeasible_coordinate[i][j] < 6


def test_directional_carry_maximizes_residence_over_all_cell_corners(graph):
    scale = 2**NODAL_REMAINDER_DENOMINATOR_BITS
    for i, row in enumerate(ROWS):
        pressure = graph.pressures[i]
        domain = derive_nodal_remainder_itinerary(
            epi_states=(row, row), timesteps=(0.,), capacities=((1.,) * 6,),
            pressures=(pressure,), epi_lower=.375, epi_upper=.625,
        )
        for side in product((0, 1), repeat=6):
            indices = tuple(cell.last_grid_index if bit else cell.first_grid_index
                            for cell, bit in zip(domain.coordinates, side, strict=True))
            state = NodalRemainderState(
                row, tuple(F(index, scale) - F(value) for index, value in zip(indices, row, strict=True)),
                .375, .625,
            )
            horizon = derive_nodal_remainder_cell_horizon(
                state=state, timestep=graph.timestep, capacity=(1.,) * 6, pressure=pressure,
            )
            assert horizon.max_unchanged_steps <= graph.maximal_residence[i].max_unchanged_steps


def test_first_exit_extremal_witness_is_sharp_for_each_single_cell(graph):
    for row, horizon in zip(ROWS, graph.maximal_residence, strict=True):
        assert tuple(map(float, horizon.unchanged_endpoint)) == row
        assert tuple(map(float, horizon.first_exit_exact)) != row
        assert horizon.first_exit_step == horizon.max_unchanged_steps + 1


def test_nonself_cycle_is_explicitly_undecided_without_a_carry_cycle_claim():
    source = derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=.5, phase_weight=1.)
    graph = derive_c6_carried_cell_graph(source, epi_states=((.4, .6) * 3, (.6, .4) * 3), timestep=2.)
    assert graph.adjacency == ((False, True), (True, False))
    assert graph.nonself_topological_order is None
    assert graph.family_exit_step_bounds is None
    assert not graph.finite_family_escape_certified
    assert not graph.carried_cycle_in_family_excluded


def test_stationary_pressure_abstains_from_finite_escape_even_on_a_dag():
    source = derive_c6_pressure_lattice(phase=(0.,) * 6, epi_weight=.5, phase_weight=1.)
    graph = derive_c6_carried_cell_graph(source, epi_states=((.5,) * 6,), timestep=.0625)
    assert graph.adjacency == ((True,),)
    assert graph.nonself_topological_order == (0,)
    assert graph.maximal_residence[0].max_unchanged_steps is None
    assert graph.family_exit_step_bounds is None
    assert not graph.finite_family_escape_certified


def test_pressure_and_sign_caches_cannot_forge_a_stationary_row(source, graph):
    forged = replace(source, sources=(0.,) * 6, every_pressure_has_positive=False, no_cartesian_trap=False)
    rebuilt = derive_c6_carried_cell_graph(forged, epi_states=ROWS, timestep=.0625)
    assert rebuilt == graph


def test_supplied_cell_order_only_relabels_the_certificate(source, graph):
    reverse = derive_c6_carried_cell_graph(source, epi_states=ROWS[::-1], timestep=.0625)
    assert reverse.adjacency == tuple(row[::-1] for row in graph.adjacency[::-1])
    assert reverse.family_exit_step_bounds == graph.family_exit_step_bounds[::-1]
    assert reverse.maximum_steps_until_family_exit_or_band_failure == 77


@pytest.mark.parametrize("states", ((), [], [ROWS[0]], (ROWS[0], ROWS[0])))
def test_missing_or_duplicate_cell_families_are_rejected(source, states):
    with pytest.raises((TypeError, ValueError)):
        derive_c6_carried_cell_graph(source, epi_states=states, timestep=.0625)


@pytest.mark.parametrize("row", (
    (.5,) * 5, [.5] * 6, (.5,) * 5 + (1,), (.5,) * 5 + (float("nan"),), (.25,) * 6,
))
def test_malformed_visible_rows_are_rejected(source, row):
    with pytest.raises((TypeError, ValueError)):
        derive_c6_carried_cell_graph(source, epi_states=(row,), timestep=.0625)


@pytest.mark.parametrize("step", (0., -1., True, 1, float("nan"), float("inf")))
def test_timestep_requires_a_finite_positive_binary64(source, step):
    with pytest.raises((TypeError, ValueError)):
        derive_c6_carried_cell_graph(source, epi_states=(ROWS[0],), timestep=step)


def test_primitive_source_changes_are_recomputed_instead_of_trusting_derived_fields(source, graph):
    changed = replace(source, source=replace(source.source, phase=(0.,) * 6))
    result = derive_c6_carried_cell_graph(changed, epi_states=ROWS, timestep=.0625)
    assert result.pressures != graph.pressures
    assert result.reference.source.phase == (0.,) * 6


def test_bad_reference_is_not_admitted():
    with pytest.raises(TypeError, match="C6PressureLatticeReference"):
        derive_c6_carried_cell_graph(object(), epi_states=(ROWS[0],), timestep=.0625)

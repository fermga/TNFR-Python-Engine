"""Exact row-space oracles for minimal fixed affine EPI observations."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from tests.physics.test_forced_epi_closure import (
    _apply, _geometry, _identity, _product, _rank,
)
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.support_transport import _from_data


F = Fraction


def _reference(size=5, *, capacities=None, undirected=None, forcing=None):
    nodes = tuple(range(size))
    if capacities is None:
        capacities = (F(1),) * size
    if undirected is None:
        undirected = tuple((i, i+1, F(1)) for i in range(size-1))
    directed = tuple((i, j, F(w)) for left, right, w in undirected
                     for i, j in ((left, right), (right, left)))
    support = tuple(tuple(sorted(j for i, j, _ in directed if i == node)) for node in nodes)
    source = _from_data(nodes, directed, support, tuple(F(i-2, i+1) for i in nodes),
                        capacities, (F(17),)*size)
    if forcing is None:
        forcing = tuple(F(i+1, i+2) for i in nodes)
    return derive_forced_support_balance(source, epi_weight=F(2, 5), forcing=forcing)


def _row_space_oracle(reference, blocks):
    """Full powers through n, using fresh elimination for each candidate row."""
    geometry = _geometry(reference, blocks)
    r, a = geometry["R"], geometry["A"]
    retained, positions = [], []
    level_ranks = []
    power = r
    for level in range(len(a)+1):
        previous_rank = len(retained)
        for ordinal, row in enumerate(power):
            candidate = (*retained, row)
            if _rank(candidate) > len(retained):
                retained.append(row)
                positions.append((level, ordinal))
        level_ranks.append(len(retained))
        if level and previous_rank == len(retained):
            break
        power = _product(power, a)
    assert _rank((*retained, *_product(tuple(retained), a))) == len(retained)
    return geometry, tuple(retained), tuple(positions), tuple(level_ranks)


def _assert_realization(result, reference, blocks):
    geometry, expected, labels, levels = _row_space_oracle(reference, blocks)
    c, t = result.observation, result.right_inverse
    g, d = result.reduced_generator, result.output_map
    assert c == expected
    assert result.selected_row_labels == labels
    assert result.rank_progression == levels[:len(result.rank_progression)]
    assert _rank(c) == len(c)
    assert len(c) == len(reference.source.nodes) or result.rank_progression[-1] == result.rank_progression[-2]
    assert _product(c, t) == _identity(len(c))
    assert _product(c, geometry["A"]) == _product(g, c)
    assert _product(d, c) == geometry["R"]
    assert g == _product(_product(c, geometry["A"]), t)
    assert d == _product(geometry["R"], t)
    assert result.reduced_source == _apply(c, geometry["b"])
    assert result.reduced_state == _apply(c, result.epi)
    rate = tuple(b-a for b, a in zip(geometry["b"], _apply(geometry["A"], result.epi), strict=True))
    assert result.reduced_rate == _apply(c, rate)
    assert result.reduced_rate == tuple(b-a for b, a in zip(
        result.reduced_source, _apply(g, result.reduced_state), strict=True,
    ))
    assert result.projected_state == _apply(geometry["R"], result.epi)
    assert result.projected_rate == _apply(geometry["R"], rate)
    assert result.reconstructed_projected_state == _apply(d, result.reduced_state) == result.projected_state
    assert result.reconstructed_projected_rate == _apply(d, result.reduced_rate) == result.projected_rate
    columns = []
    for j in range(len(reference.source.nodes)):
        candidate = tuple(tuple(row[k] for k in (*columns, j)) for row in c)
        if _rank(candidate) > len(columns):
            columns.append(j)
    assert result.pivot_columns == tuple(columns)
    assert result.dimension == len(c)
    assert result.full_state_dimension == len(reference.source.nodes)
    assert result.extra_coordinates == len(c)-len(blocks)
    assert result.completed_levels == len(result.level_records) == len(result.rank_progression)
    assert result.stabilization_power == result.completed_levels-1
    for power, level in enumerate(result.level_records):
        assert level.power == power and level.candidate_rows == len(blocks)
        assert level.rank_before == (0 if power == 0 else result.rank_progression[power-1])
        assert level.rank_after == result.rank_progression[power]
        assert level.selected_row_indices == tuple(i for k, i in labels if k == power)
    assert result.level_records[-1].rank_before == result.level_records[-1].rank_after
    assert result.rank_calls == len(blocks)*result.completed_levels + result.pivot_columns[-1]+2
    for i, row in enumerate(t):
        if i not in columns:
            assert not any(row)


def test_closed_equitable_quotient_requires_no_additional_observed_coordinates():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    reference = _reference(4)
    blocks = ((0, 3), (1, 2))
    with patch("tnfr.physics.epi_memory.matrix_exponential", side_effect=AssertionError("no exponential")):
        result = observe_forced_support_realization(reference, blocks)
    _assert_realization(result, reference, blocks)
    assert result.closure.all_state_affine_closed
    assert result.observation == result.closure.projection
    assert len(result.observation) == 2
    assert result.selected_row_labels == ((0, 0), (0, 1))
    assert result.rank_progression == (2, 2)


def test_p5_requires_exactly_one_hidden_linear_coordinate():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    reference = _reference()
    blocks = ((0, 4), (1, 2, 3))
    result = observe_forced_support_realization(reference, blocks)
    _assert_realization(result, reference, blocks)
    assert not result.closure.all_state_affine_closed
    assert len(result.observation) == 3
    assert result.rank_progression == (2, 3, 3)
    assert result.selected_row_labels == ((0, 0), (0, 1), (1, 0))


def test_minimal_closed_observation_can_require_the_full_microstate_dimension():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    reference = _reference(4, capacities=(F(1), F(2), F(3), F(4)))
    blocks = ((0, 1), (2, 3))
    result = observe_forced_support_realization(reference, blocks)
    _assert_realization(result, reference, blocks)
    assert len(result.observation) == 4
    assert not result.closure.all_state_affine_closed
    assert result.selected_row_labels[-1][0] == 2


def test_whole_level_is_processed_when_its_first_seed_row_adds_no_direction():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    reference = _reference(
        5, capacities=(F(1), F(1), F(1), F(1), F(2)),
        undirected=((0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 4, 1)),
    )
    blocks = ((0,), (1, 2), (3, 4))
    geometry = _geometry(reference, blocks)
    first_level = _product(geometry["R"], geometry["A"])
    assert _rank((*geometry["R"], first_level[0])) == 3
    assert _rank((*geometry["R"], first_level[1])) == 4
    result = observe_forced_support_realization(reference, blocks)
    _assert_realization(result, reference, blocks)
    assert len(result.observation) == 5
    assert result.selected_row_labels == ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))


def test_rank_is_exact_even_when_binary64_erases_a_tiny_rational_symmetry_break():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    epsilon = F(1, 2**100)
    capacities = (1+epsilon, F(1), F(1), F(1))
    assert float(capacities[0]) == 1.0
    reference = _reference(4, capacities=capacities)
    blocks = ((0, 3), (1, 2))
    result = observe_forced_support_realization(reference, blocks)
    _assert_realization(result, reference, blocks)
    assert len(result.observation) == 4
    assert np.linalg.matrix_rank(np.asarray(result.observation, dtype=float)) == 2
    symmetric = observe_forced_support_realization(_reference(4), blocks)
    assert len(symmetric.observation) == 2


def test_affine_source_changes_the_reduced_forcing_without_changing_linear_minimality():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    reference = _reference()
    blocks = ((0, 4), (1, 2, 3))
    epi = tuple(F(i*i-3, i+1) for i in range(5))
    forced = observe_forced_support_realization(reference, blocks, epi=epi)
    unforced = observe_forced_support_realization(replace(reference, forcing=(F(0),)*5), blocks, epi=epi)
    _assert_realization(forced, reference, blocks)
    for field in ("observation", "right_inverse", "reduced_generator", "output_map",
                  "selected_row_labels", "pivot_columns", "rank_progression", "reduced_state"):
        assert getattr(forced, field) == getattr(unforced, field)
    assert any(forced.reduced_source) and not any(unforced.reduced_source)
    assert tuple(a-b for a, b in zip(forced.reduced_rate, unforced.reduced_rate, strict=True)) == forced.reduced_source


def test_guard_exhaustion_returns_no_partial_realization_and_input_remains_unchanged():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    reference = _reference()
    blocks = [[0, 4], [1, 2, 3]]
    epi = [F(i) for i in range(5)]
    before = deepcopy((asdict(reference), blocks, epi))
    complete = observe_forced_support_realization(reference, blocks, epi=epi)
    assert complete.rank_calls > 1
    for maximum in (1, complete.rank_calls-1):
        with pytest.raises(ValueError):
            observe_forced_support_realization(reference, blocks, epi=epi, max_rank_calls=maximum)
    assert (asdict(reference), blocks, epi) == before
    exact_budget = observe_forced_support_realization(
        reference, blocks, epi=epi, max_rank_calls=complete.rank_calls,
    )
    assert exact_budget.rank_calls == complete.rank_calls
    assert exact_budget.observation == complete.observation


@pytest.mark.parametrize("maximum", (False, True, 0, -1, 2.0, "20", None, F(20)))
def test_rank_call_guard_requires_an_explicit_positive_integer(maximum):
    from tnfr.physics.epi_memory import observe_forced_support_realization

    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_realization(_reference(), ((0, 4), (1, 2, 3)), max_rank_calls=maximum)


@pytest.mark.parametrize("blocks", ((), ((0, 1, 2, 3, 4),), ((0, 4), (1, 2)),
                                    ((0, 4), (1, 2, 4)), ({0, 4}, (1, 2, 3)),
                                    {"a": (0, 4), "b": (1, 2, 3)}))
def test_invalid_partitions_cannot_seed_a_reduced_model(blocks):
    from tnfr.physics.epi_memory import observe_forced_support_realization

    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_realization(_reference(), blocks)


@pytest.mark.parametrize("epi", ((F(1),), (True, 0, 0, 0, 0),
                                 (float("nan"), 0, 0, 0, 0), (float("inf"), 0, 0, 0, 0),
                                 {0, 1, 2, 3, 4}))
def test_invalid_current_epi_cannot_enter_the_realization(epi):
    from tnfr.physics.epi_memory import observe_forced_support_realization

    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_realization(_reference(), ((0, 4), (1, 2, 3)), epi=epi)


def test_cached_reference_fields_cannot_forge_observability_or_pressure_forcing():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    reference = _reference()
    blocks = ((0, 4), (1, 2, 3))
    expected = observe_forced_support_realization(reference, blocks)
    corrupted = replace(reference, strengths=(F(99),)*5, metric_weights=(F(99),)*5,
                        relative_profile=(F(99),)*5, profile_residual=(F(99),)*5,
                        mean_drift=F(99))
    assert observe_forced_support_realization(corrupted, blocks) == expected
    with pytest.raises(TypeError):
        observe_forced_support_realization(asdict(reference), blocks)
    with pytest.raises(ValueError):
        observe_forced_support_realization(replace(reference, epi_weight=F(0)), blocks)
    altered_pressure = replace(reference, source=replace(reference.source, stored_pressure=(F(999),)*5))
    second = observe_forced_support_realization(altered_pressure, blocks)
    for field in ("observation", "reduced_source", "reduced_state", "reduced_rate"):
        assert getattr(second, field) == getattr(expected, field)


def test_inputs_are_detached_and_outputs_immutable_with_deterministic_pivots():
    from tnfr.physics.epi_memory import observe_forced_support_realization

    reference = _reference()
    blocks, epi = [[0, 4], [1, 2, 3]], [F(i) for i in range(5)]
    result = observe_forced_support_realization(reference, blocks, epi=epi)
    assert result == observe_forced_support_realization(reference, blocks, epi=epi)
    before = asdict(result)
    blocks.reverse()
    epi[0] = F(999)
    assert asdict(result) == before
    with pytest.raises(FrozenInstanceError):
        result.reduced_source = ()

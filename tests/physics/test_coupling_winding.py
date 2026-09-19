"""Independent exact controls for the target-only UM gap companion."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.constants.canonical import DELTA_PHI_MAX, UM_THETA_PUSH
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.coupling_winding import observe_coupling_gap_step

F = Fraction


def _variance(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / 2


@pytest.mark.parametrize("target", [0, 1, 4])
def test_single_target_moves_only_its_incident_gaps(target):
    gaps = (F(1), F(1, 2), F(-1, 4), F(-1), F(3, 4))
    result = observe_coupling_gap_step(gaps, target=target, eta=F(2, 3))
    left = (target - 1) % len(gaps)
    displacement = (gaps[target] - gaps[left]) / 3
    expected = list(gaps)
    expected[left] += displacement
    expected[target] -= displacement
    assert result.output_gaps == tuple(expected)
    assert result.mode == "single_target"
    assert result.target == target
    assert result.sum_residual == 0
    assert result.invariant_interval_preserved
    assert result.spread_before == _variance(gaps)
    assert result.spread_after == _variance(expected)
    assert result.spread_drop == F(2, 9) * (gaps[target] - gaps[left]) ** 2


@pytest.mark.parametrize("target", [None, 0, 3])
def test_gap_map_is_doubly_stochastic_and_symmetric(target):
    result = observe_coupling_gap_step((1, 0, -1, F(1, 2)), target=target)
    matrix = result.matrix
    assert all(sum(row) == 1 for row in matrix)
    assert all(sum(column) == 1 for column in zip(*matrix))
    assert matrix == tuple(zip(*matrix))
    assert all(value >= 0 for row in matrix for value in row)
    assert result.eta == F.from_float(UM_THETA_PUSH)
    assert result.phase_gate == F.from_float(DELTA_PHI_MAX)


def test_jacobi_gap_transport_matches_existing_canonical_epi_pressure():
    gaps = (F(1), F(-1, 2), F(0), F(1, 2), F(-1))
    graph = nx.cycle_graph(len(gaps))
    graph.graph.update(
        DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
        GAMMA={"type": "none"},
    )
    for node, value in enumerate(gaps):
        graph.nodes[node].update(EPI=float(value), theta=0.0, nu_f=1.0)
    default_compute_delta_nfr(graph)
    pressure = tuple(
        F.from_float(float(get_attr(graph.nodes[node], ALIAS_DNFR, None)))
        for node in graph
    )
    result = observe_coupling_gap_step(gaps, eta=F(1, 2))
    assert result.output_gaps == tuple(
        value + p / 2 for value, p in zip(gaps, pressure)
    )
    assert result.mode == "all_targets"
    assert result.gap_sum == 0
    assert result.mean_gap == 0


def test_jacobi_dissipation_matches_the_independent_jensen_identity():
    result = observe_coupling_gap_step((1, F(-1, 2), 0, F(1, 4), -1))
    expected = F(0)
    for row in result.matrix:
        for left in range(len(row)):
            for right in range(left + 1, len(row)):
                expected += (
                    row[left]
                    * row[right]
                    * (result.input_gaps[left] - result.input_gaps[right]) ** 2
                    / 2
                )
    assert result.spread_drop == result.expected_spread_drop == expected
    assert expected > 0


def test_even_cycle_eta_one_has_a_noncontracting_checkerboard():
    gaps = (F(3, 4), F(1, 4)) * 4
    first = observe_coupling_gap_step(gaps, eta=1)
    second = observe_coupling_gap_step(first.output_gaps, eta=1)
    assert first.output_gaps == tuple(reversed(gaps))
    assert first.spread_drop == 0
    assert first.spread_after > 0
    assert second.output_gaps == gaps


def test_half_mix_annihilates_checkerboard_gap_spread():
    result = observe_coupling_gap_step((1, -1) * 4, eta=F(1, 2))
    assert result.output_gaps == (0,) * 8
    assert result.spread_before == result.spread_drop == 4
    assert result.spread_after == 0


def test_one_target_does_not_imply_global_gap_convergence():
    gaps = (0, 0, 1, -1)
    result = observe_coupling_gap_step(gaps, target=1)
    assert result.output_gaps == gaps
    assert result.spread_drop == 0
    assert result.spread_after > 0


@pytest.mark.parametrize("gaps", [(F(1, 2),) * 8, (F(-1, 3),) * 7, (0,) * 5])
def test_uniform_gap_fields_are_fixed_without_inferring_circle_closure(gaps):
    result = observe_coupling_gap_step(gaps)
    assert result.output_gaps == gaps
    assert result.gap_sum == sum(gaps)
    assert result.spread_after == result.spread_before == 0
    assert not hasattr(result, "winding")


def test_finite_mixed_update_sequence_preserves_the_original_interval():
    gaps = tuple(F(value, 8) for value in (1, 5, 3, 7, 4, 2, 6, 4))
    total, lower, upper = sum(gaps), min(gaps), max(gaps)
    previous = _variance(gaps)
    for target, eta in ((0, F(1, 2)), (None, F(2, 3)), (4, F(1)), (None, F(1))):
        result = observe_coupling_gap_step(gaps, target=target, eta=eta)
        assert result.gap_sum == total
        assert lower <= result.minimum_after <= result.maximum_after <= upper
        assert result.spread_after <= previous
        gaps, previous = result.output_gaps, result.spread_after


def test_orientation_reversal_and_cyclic_reindexing_transport_jacobi_output():
    gaps = (F(1), F(1, 2), F(-1, 4), F(-1), F(3, 4))
    original = observe_coupling_gap_step(gaps)
    reversed_result = observe_coupling_gap_step(tuple(-g for g in reversed(gaps)))
    rotated_result = observe_coupling_gap_step(gaps[2:] + gaps[:2])
    assert reversed_result.output_gaps == tuple(
        -g for g in reversed(original.output_gaps)
    )
    assert rotated_result.output_gaps == (
        original.output_gaps[2:] + original.output_gaps[:2]
    )
    assert reversed_result.spread_after == rotated_result.spread_after


def test_exact_rationals_and_materialized_reals_have_explicit_semantics():
    tiny = F(1, 10**500)
    exact = observe_coupling_gap_step((tiny, -tiny, 0))
    assert exact.input_gaps == (tiny, -tiny, 0)
    assert exact.spread_before == tiny**2
    floating = observe_coupling_gap_step((0.1, 0.2, -0.3))
    assert floating.input_gaps == tuple(F.from_float(v) for v in (0.1, 0.2, -0.3))


@pytest.mark.parametrize(
    "gaps",
    [
        np.array([1, 0, -1], dtype=np.int64),
        (F(np.int64(1)), F(np.int64(0)), F(np.int64(-1))),
    ],
)
def test_numpy_rational_inputs_are_promoted_to_python_integer_arithmetic(gaps):
    result = observe_coupling_gap_step(gaps)
    assert result.input_gaps == (1, 0, -1)
    assert all(type(value.numerator) is int for value in result.input_gaps)
    assert result.spread_drop > 0


def test_output_is_immutable_and_detached_from_caller_coordinates():
    gaps = [0, 1, -1]
    result = observe_coupling_gap_step(gaps)
    gaps[0] = 99
    assert result.input_gaps == (0, 1, -1)
    with pytest.raises(FrozenInstanceError):
        result.spread_drop = F(0)
    with pytest.raises(TypeError):
        result.output_gaps[0] = F(0)


@pytest.mark.parametrize(
    "gaps",
    [
        (),
        (0, 0),
        "000",
        {0, 1, 2},
        {0: 0, 1: 1, 2: -1},
        (True, 0, 0),
        ("0", 0, 0),
        (1j, 0, 0),
        (float("nan"), 0, 0),
        (float("inf"), 0, 0),
    ],
)
def test_invalid_gap_coordinates_are_rejected(gaps):
    with pytest.raises((TypeError, ValueError)):
        observe_coupling_gap_step(gaps)


@pytest.mark.parametrize("eta", [0, -1, F(3, 2), True, "0.5", float("inf")])
def test_eta_is_a_positive_canonical_unit_interval_factor(eta):
    with pytest.raises((TypeError, ValueError)):
        observe_coupling_gap_step((0, 1, -1), eta=eta)


@pytest.mark.parametrize("gate", [0, -1, 2, True, "1", float("nan")])
def test_gate_cannot_weaken_the_canonical_limit(gate):
    with pytest.raises((TypeError, ValueError)):
        observe_coupling_gap_step((0, 0, 0), phase_gate=gate)


@pytest.mark.parametrize("target", [True, F(1, 2), "1", -1, 3])
def test_target_must_identify_one_declared_cycle_node(target):
    with pytest.raises((TypeError, ValueError)):
        observe_coupling_gap_step((0, 1, -1), target=target)


@pytest.mark.parametrize("gap", [F(1), F(-1), F(5, 4)])
def test_strict_gate_excludes_boundary_and_outside_gaps(gap):
    with pytest.raises(ValueError, match="strictly inside"):
        observe_coupling_gap_step((gap, 0, 0), phase_gate=1)

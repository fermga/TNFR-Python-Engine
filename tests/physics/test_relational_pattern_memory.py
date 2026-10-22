"""Static controls for the joint-law memory split and its nonlinear parity.

Rational matrices test the declared coefficient family. Native field probes
check independent formulas at admitted snapshots, not trajectory error bounds.
"""

import math
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from benchmarks.relational_local_composition import (
    analyze_local_memory,
    analyze_state_rate_obstruction,
)
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)


def _mv(matrix, vector):
    return tuple(
        sum((a * b for a, b in zip(row, vector, strict=True)), Q(0)) for row in matrix
    )


def _quadratic(matrix, vector):
    return sum((a * b for a, b in zip(vector, _mv(matrix, vector))), Q(0))


@pytest.fixture(scope="module")
def split():
    return analyze_local_memory(ring_cosine=Q(1, 3), inverse_pi=Q(2, 7))


def test_joint_coordinates_reconstruct_exactly_only_modulo_common_offsets(split):
    state = tuple(Q(i * i - 5 * i, 64) for i in range(20))
    y, h = _mv(split.visible_projection, state), _mv(split.hidden_projection, state)
    reconstructed = tuple(
        a + b for a, b in zip(_mv(split.visible_lift, y), _mv(split.hidden_lift, h))
    )
    assert len(y) == 10 and len(h) == 8
    expected = tuple(
        state[i] - sum(state[10 * (i // 10) : 10 * (i // 10) + 10]) / 10
        for i in range(20)
    )
    assert reconstructed == expected == _mv(split.quotient_projector, state)
    assert h == tuple(
        (state[a] - state[b]) / 2
        for a, b in (
            (1, 4),
            (2, 3),
            (6, 9),
            (7, 8),
            (11, 14),
            (12, 13),
            (16, 19),
            (17, 18),
        )
    )


def test_hidden_block_is_joint_damped_exchange_not_a_diffusion_kernel(split):
    spatial = (
        (2, -1, 0, 0),
        (-1, 3, 0, 0),
        (0, 0, 2, -1),
        (0, 0, -1, 3),
    )
    # c=1/3, inverse_pi=2/7, e=w=1/2, beta=1.
    expected = tuple(
        tuple(-Q(v, 4) for v in row) + tuple(-Q(v, 14) for v in row) for row in spatial
    ) + tuple(tuple(Q(3 * v, 14) for v in row) + (Q(0),) * 4 for row in spatial)
    assert split.hidden_generator == expected
    energy = tuple(tuple(map(Q, row)) + (Q(0),) * 4 for row in spatial) + tuple(
        (Q(0),) * 4 + tuple(Q(v, 3) for v in row) for row in spatial
    )
    assert split.hidden_energy == energy
    hidden = tuple(Q(i - 3, 16) for i in range(8))
    expected_loss = sum((v * v for v in _mv(spatial, hidden[:4])), Q(0)) / 2
    assert _quadratic(split.hidden_energy_rate, hidden) == -expected_loss
    assert _quadratic(split.hidden_dissipation, hidden) == expected_loss


def test_visible_and_hidden_energy_split_matches_independent_graph_edges(split):
    state = tuple(Q((i * 7) % 11 - 5, 32) for i in range(20))
    y, h = _mv(split.visible_projection, state), _mv(split.hidden_projection, state)
    edges = [(r + i, r + (i + 1) % 5) for r in (0, 5) for i in range(5)] + [(0, 5)]
    expected = sum(
        (
            (state[i] - state[j]) ** 2
            + (Q(1) if (i, j) == (0, 5) else Q(1, 3))
            * (state[10 + i] - state[10 + j]) ** 2
        )
        / 2
        for i, j in edges
    )
    assert (
        _quadratic(split.visible_energy, y) + _quadratic(split.hidden_energy, h)
        == expected
    )
    assert _quadratic(split.visible_energy_rate, y) == -_quadratic(
        split.visible_dissipation, y
    )
    assert _quadratic(split.visible_energy_rate, y) <= 0


def test_lossless_hidden_block_does_not_claim_forgetting():
    report = analyze_local_memory(ring_cosine=Q(1, 3), inverse_pi=Q(2, 7), epi_weight=0)
    assert all(v == 0 for row in report.hidden_energy_rate for v in row)
    assert any(v != 0 for row in report.hidden_generator for v in row)


def test_memory_split_retains_the_initial_source_of_the_acceleration_obstruction(split):
    witness = analyze_state_rate_obstruction(
        ring_cosine=Q(1, 3), ring_sine=Q(2, 5), inverse_pi=Q(2, 7)
    )
    hidden = []
    for state, rate in (
        (witness.state_minus, witness.rate_minus),
        (witness.state_plus, witness.rate_plus),
    ):
        y, h = _mv(split.visible_projection, state), _mv(split.hidden_projection, state)
        hidden.append(h)
        # At this snapshot the nonlinear remainder is zero, but its future
        # derivative can depend on the retained hidden initialization.
        assert _mv(split.visible_projection, rate) == _mv(split.visible_generator, y)
        assert _mv(split.hidden_projection, rate) == _mv(split.hidden_generator, h)
    assert hidden[0] == (-Q(1, 64),) + (Q(0),) * 7
    assert hidden[1] == (Q(1, 64),) + (Q(0),) * 7
    assert witness.projected_rate_minus == witness.projected_rate_plus
    assert witness.scalar_gap_actual < 0


def _graph(form, deviation):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edge(0, 5)
    for node in graph:
        graph.nodes[node].update(
            EPI=form[node], theta=(node % 5) * math.tau / 5 + deviation[node], nu_f=1.0
        )
    return graph


def _rates(form, deviation):
    field = evaluate_relational_exchange(
        _graph(form, deviation), model=RelationalExchangeModel(1)
    )
    return np.array(field.form_rate + field.phase_rate)


def test_native_joint_parity_separates_prediction_from_hidden_reconstruction(split):
    reflection = (0, 4, 3, 2, 1, 5, 9, 8, 7, 6)
    form = (0, 3 / 128, -1 / 128, -3 / 128, 1 / 128) + (0,) * 5
    phase = (1 / 64, 1 / 256, 0, 0, -1 / 256) + (0,) * 5
    original = _rates(form, phase)
    inverted = _rates(
        tuple(-form[i] for i in reflection), tuple(-phase[i] for i in reflection)
    )
    c, p = np.array(split.visible_projection, dtype=float), np.array(
        split.hidden_projection, dtype=float
    )
    np.testing.assert_allclose(c @ inverted, -(c @ original), atol=3e-15, rtol=0)
    np.testing.assert_allclose(p @ inverted, p @ original, atol=3e-15, rtol=0)
    pure_odd = _rates(
        (0, 1 / 64, 0, 0, -1 / 64) + (0,) * 5, (0, 1 / 128, 0, 0, -1 / 128) + (0,) * 5
    )
    np.testing.assert_allclose(c @ pure_odd, 0, atol=3e-15, rtol=0)
    assert np.linalg.norm(p @ pure_odd) > 1e-3


def test_prepared_even_state_generates_hidden_response_and_cubic_visible_residual(
    split,
):
    epsilon, displacement = 1 / 64, 1 / 128
    form = tuple(epsilon * v for v in (0, 1, -1, -1, 1)) + (0,) * 5
    phase = (displacement,) + (0,) * 9
    state = np.array(form + phase)
    rate = _rates(form, phase)
    crows, prows = np.array(split.visible_projection, dtype=float), np.array(
        split.hidden_projection, dtype=float
    )
    np.testing.assert_array_equal(prows @ state, np.zeros(8))
    c, s, k = math.cos(math.tau / 5), math.sin(math.tau / 5), 0.5
    hidden_expected = (
        -3
        * k
        * epsilon
        * s
        * displacement
        / (4 * math.pi * (c * c - math.sin(displacement / 2) ** 2))
    )
    assert (prows @ rate)[4] == pytest.approx(hidden_expected, rel=2e-12, abs=3e-16)
    assert hidden_expected < 0
    # The output is theta_dot0 - mean(theta_dot_right). Its linear prediction
    # uses the true trigonometric metric, not the rational probe's generator.
    actual = (crows @ rate)[5] + (crows @ rate)[6]
    linear = -2 * k * epsilon / (math.pi * (1 + 2 * c))
    sinc = math.sin(displacement) / displacement
    remainder = linear * (1 / sinc - 1)
    assert actual - linear == pytest.approx(remainder, rel=2e-9, abs=3e-16)
    leading = -k * epsilon * displacement**2 / (3 * math.pi * (1 + 2 * c))
    assert remainder == pytest.approx(leading, rel=2e-5, abs=0)

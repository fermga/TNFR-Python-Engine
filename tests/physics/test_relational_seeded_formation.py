"""Static storage/domain obstructions, not simulated pattern formation."""

import math

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.mathematics._rational_interval import cos, pi_interval


def _source_and_uniform_receiver(ports, offset):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from((i, i + 5) for i in range(ports))
    for node in graph:
        phase = math.tau * node / 5 if node < 5 else offset
        graph.nodes[node].update(EPI=0.0, theta=phase, nu_f=1.0)
    return graph


def test_exact_radical_storage_deficits_are_positive_for_both_interfaces():
    s = pytest.importorskip("sympy")
    c = s.cos(2 * s.pi / 5)
    assert s.simplify(c - (s.sqrt(5) - 1) / 4) == 0
    source = 5 * (1 - c)
    expected = ((23 - 7 * s.sqrt(5)) / 4, (21 - 9 * s.sqrt(5)) / 4)
    for bridges, radical in enumerate(expected, start=1):
        deficit = 2 * source - (source + bridges * (1 + 2 * c))
        assert s.simplify(deficit - radical) == 0
        assert bool(radical > 0)
    # Jensen applies to the strictly acute target, because the phase-edge
    # potential's second derivative is positive there. Winding alone does
    # not supply this convexity hypothesis.
    delta = s.symbols("delta", real=True)
    assert s.diff(1 - s.cos(delta), delta, 2) == s.cos(delta)


@pytest.mark.parametrize("ports", (1, 2))
@pytest.mark.parametrize("offset", (0.0, 2 * math.pi / 3))
def test_native_positive_resultant_field_respects_the_interface_budget(ports, offset):
    graph = _source_and_uniform_receiver(ports, offset)
    field = evaluate_relational_exchange(
        graph, model=RelationalExchangeModel(1, phase_domain="positive_resultant")
    )
    c = math.cos(math.tau / 5)
    bridge_cost = 0.0
    for i in range(ports):
        gap = offset - math.tau * i / 5
        bridge_cost += 1 - math.cos(gap)
        assert field.relative_resultant[i][0] == pytest.approx(2 * c + math.cos(gap))
        assert field.relative_resultant[i + 5][0] == pytest.approx(2 + math.cos(gap))
    assert field.form_storage == 0
    assert float(field.storage) == pytest.approx(5 * (1 - c) + bridge_cost)
    assert float(field.storage) < 5 * (1 - c) + ports * (1 + 2 * c)
    assert float(field.storage) < 10 * (1 - c)
    # Equal form implies zero initial phase rate and work, but attachment
    # can still start a nonzero form response. Lack of storage is the
    # obstruction to the specified target, not absence of interaction.
    assert field.phase_rate == (0.0,) * 10
    if ports == 1 and offset == 0:
        assert max(map(abs, field.form_rate)) < 1e-14
    else:
        assert any(abs(value) > 1e-3 for value in field.form_rate)
    assert sum(field.work.dissipation) == 0


def test_wider_regular_two_port_state_is_not_a_universal_energy_obstruction():
    s = pytest.importorskip("sympy")
    c, d = s.cos(2 * s.pi / 5), s.cos(s.pi / 5)
    source_real, source_imaginary = 2 * c - d, s.sin(s.pi / 5)
    assert bool(source_real < 0) and bool(source_imaginary > 0)
    # The two source arguments are +/-3pi/5, away from the excluded
    # negative-real axis. Their existing full-regular phase metric is finite
    # and positive even though production's positive-resultant gate rejects.
    assert s.simplify(source_real - 2 * c * s.cos(3 * s.pi / 5)) == 0
    assert s.simplify(source_imaginary - 2 * c * s.sin(3 * s.pi / 5)) == 0
    metric = s.pi * source_imaginary / (3 * s.pi / 5)
    assert s.simplify(metric - 5 * source_imaginary / 3) == 0
    assert bool(metric > 0) and bool(2 - d > 0)
    excess = 2 + 2 * d - 5 * (1 - c)
    assert s.simplify(excess - (7 * s.sqrt(5) - 15) / 4) == 0
    assert bool(excess > 0)
    graph = _source_and_uniform_receiver(2, 6 * math.pi / 5)
    with pytest.raises(ValueError, match="positive real part"):
        evaluate_relational_exchange(
            graph, model=RelationalExchangeModel(1, phase_domain="positive_resultant")
        )
    # This is an initial-domain counterexample only: positive storage excess
    # does not certify a regular route, target reachability or later capture.


def test_nonacute_winding_one_does_not_imply_the_twist_storage_lower_bound():
    from fractions import Fraction as Q

    gaps = (Q(7, 8), *(Q(9, 32) for _ in range(4)))
    assert sum(gaps) == 2
    assert all(0 < value < 1 for value in gaps) and max(gaps) > Q(1, 2)
    pi = pi_interval()
    potential_upper = sum(1 - cos(value * pi).lo for value in gaps)
    twist_lower = 5 * (1 - cos(2 * pi / 5).hi)
    assert potential_upper < twist_lower

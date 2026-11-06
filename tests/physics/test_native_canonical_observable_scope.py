"""Static observation controls, without a native trajectory or source search.

Equal phase diameter and resultant do not retain target-associated pressure.
The exact recurrence controls distinguish finite nonzero spread, a positive
maintenance margin, and strict complete-call growth. Their scalar budgets
are comparison examples, not native phase writers or occurrence witnesses.

Existing ``test_native_mutation_renewal.py`` owns the admitted narrowing
Mutation prefix and later live rejection; ``test_mutation_trigger.py`` owns
observed-history admission. Neither calculation is replayed here.
"""

import math
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr import operators
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics import integrators, runtime
from tnfr.dynamics.dnfr import _resolve_dnfr_weights, default_compute_delta_nfr

PHASE_GAP = Q(1, 4)
PHASES = (
    (Q(0), PHASE_GAP, Q(0), Q(0), Q(0)),
    (Q(0), Q(0), PHASE_GAP, Q(0), Q(0)),
)


@pytest.fixture(scope="module", autouse=True)
def no_native_execution():
    """Allow static pressure preparation, never events or integrated responses."""

    def forbidden(*args, **kwargs):
        pytest.fail("Observable-scope controls must not execute native dynamics")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (runtime, "step"),
            (runtime, "run"),
            (operators, "apply_glyph"),
            (operators, "apply_glyph_obj"),
            (integrators.DefaultIntegrator, "integrate"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


def _doubled_cycle(phases):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for block in range(5):
        for left in (2 * block, 2 * block + 1):
            for right in (2 * ((block + 1) % 5), 2 * ((block + 1) % 5) + 1):
                graph.add_edge(left, right, weight=1.0)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=1.0,
            theta=float(phases[node // 2]),
            epi_kind="seed",
        )
    return graph


def _resultant_squared(phases):
    # Component sums retain the supplied node order. Equal phase multisets
    # give equal ideal resultants regardless of their geometric arrangement.
    real = math.fsum(math.cos(float(value)) for value in phases)
    imag = math.fsum(math.sin(float(value)) for value in phases)
    return (real * real + imag * imag) / len(phases) ** 2


def test_equal_diameter_and_resultant_do_not_determine_target_pressure():
    pressures = []
    supports = []
    states = []
    for phases in PHASES:
        graph = _doubled_cycle(phases)
        nodes = tuple(graph)
        assert nodes == tuple(range(10))
        assert all(graph.degree(node) == 4 for node in nodes)
        assert all(graph.edges[edge]["weight"] == 1 for edge in graph.edges)
        supports.append(frozenset(frozenset(edge) for edge in graph.edges))
        states.append(
            tuple(
                (node, graph.nodes[node]["EPI"], graph.nodes[node]["nu_f"])
                for node in nodes
            )
        )

        # Resolve the configured law before preparation. Homogeneous form,
        # capacity and degree annihilate their three channels independently.
        weights = _resolve_dnfr_weights(graph)
        phase_weight = Q(weights["phase"])
        assert phase_weight > 0
        expected = tuple(
            phase_weight
            * ((phases[(block - 1) % 5] + phases[(block + 1) % 5]) / 2 - phase)
            / Q(math.pi)
            for block, phase in enumerate(phases)
        )
        primitive_before = tuple(dict(graph.nodes[node]) for node in nodes)
        default_compute_delta_nfr(graph, cache_size=0, n_jobs=1)
        actual = tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in nodes)
        for node, observed in zip(nodes, actual, strict=True):
            # This tolerance checks represented trig arithmetic, not an exact
            # circular theorem or a native transition certificate.
            assert observed == pytest.approx(
                float(expected[node // 2]), rel=0, abs=2e-15
            )
            assert all(
                graph.nodes[node][key] == value
                for key, value in primitive_before[node].items()
            )
        pressures.append(actual)

    assert supports[0] == supports[1]
    assert states[0] == states[1]
    assert sorted(PHASES[0]) == sorted(PHASES[1])
    assert max(PHASES[0]) - min(PHASES[0]) == max(PHASES[1]) - min(PHASES[1])
    assert _resultant_squared(PHASES[0]) == _resultant_squared(PHASES[1])
    # The marked target is fixed at block zero in both preparations. Its
    # neighboring phase changes; relabeling the target would change the task.
    assert PHASES[0][0] == PHASES[1][0] == 0
    assert pressures[0][0] > 0.001
    assert abs(pressures[1][0]) < 2e-15
    assert pressures[0][0] != pressures[1][0]


@pytest.mark.parametrize(
    "margin,crossing_call,prior_equality",
    ((Q(1, 64), 10, False), (Q(9, 64), 3, True)),
)
def test_positive_finite_spread_can_cross_a_fixed_maintenance_margin(
    margin, crossing_call, prior_equality
):
    # A fixed exact convex map supplies this abstract counterexample. It is
    # not a claim that a complete native call realizes these two coordinates.
    q, initial_diameter = Q(3, 4), Q(1, 4)
    assert 0 < q < 1 and 0 < margin < initial_diameter
    coordinates = (-initial_diameter / 2, initial_diameter / 2)
    observed = [max(coordinates) - min(coordinates)]
    for _ in range(crossing_call):
        coordinates = tuple(q * value for value in coordinates)
        observed.append(max(coordinates) - min(coordinates))

    assert observed == [
        q**index * initial_diameter for index in range(crossing_call + 1)
    ]
    assert all(value > 0 for value in observed)
    assert all(value >= margin for value in observed[:-1])
    assert (observed[-2] == margin) is prior_equality
    assert observed[-1] < margin
    # A closed identity condition D >= margin includes exact equality;
    # exclusion requires the strict upper-bound crossing at the next call.
    # Induction gives D_n=q^n D_0>0 for every finite n, yet a positive
    # lower maintenance margin has already failed at this explicit call.


def test_balanced_abstract_spread_budget_needs_no_strict_complete_growth():
    q, maintained = Q(3, 4), Q(1, 16)
    recharge = (1 / q - 1) * maintained
    assert recharge > 0
    previous = maintained
    for _ in range(7):
        # Saturate D_next <= q*D + q*r with zero realization defect.
        # This is an abstract comparison-budget sequence, not an admitted
        # event, source of recharge or native maintenance witness.
        before_coordination = previous + recharge
        after_coordination = q * before_coordination
        assert before_coordination > previous
        assert after_coordination == maintained
        assert after_coordination == previous  # No strict complete growth.
        previous = after_coordination

    zero_recharge_upper = q * maintained
    assert zero_recharge_upper < maintained

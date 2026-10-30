"""Finite common-chart formation controls under existing execution owners.

The initial path, homogeneous capacity, default UM policies and small declared
sine steps are supplied inputs. Exact chart observations concern represented
endpoint phases; high-precision comparisons retain runtime rounding error.
They do not certify arbitrary repetitions or derive autonomous link selection.
"""

import math
from fractions import Fraction

import mpmath as mp
import networkx as nx
import pytest

from tests.joint_phase_helpers import (
    configure,
    execute_coupling_cycle_birth,
    execute_joint_step,
)
from tnfr.alias import set_theta
from tnfr.constants.aliases import ALIAS_SI
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step
from tnfr.operators.definitions import Coupling
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    _detached_stage_graph,
    execute_coupling_stage,
)
from tnfr.physics import phase_chart
from tnfr.physics.phase_chart import observe_common_phase_chart
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.types import Glyph


def _prepared_path(*, phases=(-0.30, -0.12, 0.04, 0.16, 0.29), capacities=None):
    graph = nx.path_graph(len(phases))
    configure(graph)
    graph.graph.update(
        RANDOM_SEED=17,
        UM_FUNCTIONAL_LINKS=True,
        UM_CANDIDATE_COUNT=0,
        UM_CANDIDATE_MODE="sample",
    )
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    capacities = (1.0,) * len(phases) if capacities is None else capacities
    for node, phase, capacity in zip(graph, phases, capacities, strict=True):
        graph.nodes[node].update(
            EPI=0.125,
            theta=phase % math.tau,
            nu_f=float(capacity),
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=["AL"],
        )
        graph.nodes[node][ALIAS_SI[0]] = 0.8
    default_compute_delta_nfr(graph)
    return graph


@pytest.fixture(scope="module")
def overlapping_stages():
    executions = []
    for targets in ((1, 3), (3, 1)):
        graph = _prepared_path()
        before = _detached_stage_graph(graph)
        factors = resolve_runtime_operator_factors(None, Glyph.UM, graph.graph)
        result = execute_coupling_stage(graph, Coupling(), targets)
        executions.append((before, graph, factors, result))
    return tuple(executions)


def _high_precision_principal_phases(graph):
    return tuple(
        mp.mpf(graph.nodes[node]["theta"])
        - (2 * mp.pi if graph.nodes[node]["theta"] > math.pi else 0)
        for node in graph
    )


def test_overlapping_actual_um_creates_cycles_without_nonzero_winding(
    overlapping_stages,
):
    before, graph, _, result = overlapping_stages[0]
    initial, final = map(observe_common_phase_chart, (before, graph))
    assert initial.admitted and final.admitted
    assert initial.geometry.cycle_rank == 0
    assert set(graph.edges) - set(before.edges) == {(0, 3), (1, 3), (1, 4)}
    assert final.geometry.cycle_rank == 3
    assert initial.zero_cycle_winding is True and initial.cycle_periods == ()
    assert final.zero_cycle_winding is True and final.cycle_periods == (0, 0, 0)
    assert final.semicircle_margin_enclosure[0] > 0
    assert final.diameter_enclosure[1] < initial.diameter_enclosure[0]
    # Stored coordinates straddle zero; a raw min/max would use the wrong chart.
    assert max(initial.phase) - min(initial.phase) > 6
    for cycle in final.geometry.fundamental_cycles:
        nodes = tuple(final.nodes[index] for index in cycle)
        observed = certify_phase_winding(graph, nodes)
        assert observed.winding == 0 and observed.u3_admissible
        assert observed.quantization_residual < 1e-14
    assert result.schedule == TWO_PHASE_JACOBI and result.nodes_processed == 2
    assert tuple(graph.nodes[node]["nu_f"] for node in graph) == (1.0,) * 5
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (0.125,) * 5
    assert graph.graph["_t"] == before.graph["_t"] == 0.0
    assert graph.graph["RANDOM_SEED"] == 17
    assert "GLYPH_FACTORS" not in graph.graph
    assert "UM_COMPAT_THRESHOLD" not in graph.graph
    for node in graph:
        expected = ("AL", "UM") if node in (1, 3) else ("AL",)
        assert tuple(graph.nodes[node]["glyph_history"]) == expected


def test_actual_overlapping_writes_match_independent_circular_convex_merge(
    overlapping_stages,
):
    before, graph, factors, _ = overlapping_stages[0]
    with mp.workdps(80):
        initial = _high_precision_principal_phases(before)
        push = mp.mpf(factors["UM_theta_push"])
        expected_contributions = [[] for _ in before]
        for target in (1, 3):
            participants = (target, *before.neighbors(target))
            real = sum(mp.cos(initial[node]) for node in participants)
            imag = sum(mp.sin(initial[node]) for node in participants)
            center = mp.atan2(imag, real)
            assert min(initial) < center < max(initial)
            for node in participants:
                expected_contributions[node].append(
                    (1 - push) * initial[node] + push * center
                )
        assert len(expected_contributions[2]) == 2
        expected = tuple(
            sum(contributions) / len(contributions)
            for contributions in expected_contributions
        )
        actual = _high_precision_principal_phases(graph)
        assert max(abs(left - right) for left, right in zip(actual, expected)) < (
            mp.mpf("2e-15")
        )
        assert min(actual) > min(initial) and max(actual) < max(initial)
        # The finite binary64 endpoint is close to this exact-real calculation,
        # not asserted identical to an unrounded convex operation.
        assert any(left != right for left, right in zip(actual, expected))


def test_reverse_target_order_preserves_fixed_rank_endpoints(overlapping_stages):
    forward, reverse = (case[1] for case in overlapping_stages)
    assert tuple(forward.nodes) == tuple(reverse.nodes) == tuple(range(5))
    assert set(forward.edges) == set(reverse.edges)
    for edge in forward.edges:
        assert forward.edges[edge] == reverse.edges[edge]
    for field in ("theta", "nu_f", "EPI", "delta_nfr", "dEPI", "glyph_history"):
        assert tuple(forward.nodes[node][field] for node in forward) == tuple(
            reverse.nodes[node][field] for node in reverse
        )


def test_small_shared_joint_steps_retain_observed_chart_and_zero_periods(
    overlapping_stages,
):
    graph = _detached_stage_graph(overlapping_stages[0][1])
    default_compute_delta_nfr(graph)
    initial = observe_common_phase_chart(graph)
    dt, coupling = 0.125, 0.5
    assert dt * coupling <= 1
    previous_width = initial.diameter_enclosure[1]
    for step in range(4):
        before = _detached_stage_graph(graph)
        execution = execute_joint_step(
            graph, dt=dt, coupling_strength=coupling, time=step * dt
        )
        observed = observe_common_phase_chart(graph)
        assert observed.admitted and observed.zero_cycle_winding is True
        assert observed.cycle_periods == (0, 0, 0)
        assert observed.diameter_enclosure[1] < previous_width
        assert tuple(graph.nodes[node]["nu_f"] for node in graph) == (1.0,) * 5
        assert set(graph.edges) == set(before.edges)
        with mp.workdps(80):
            incoming = _high_precision_principal_phases(before)
            lower, upper = min(incoming), max(incoming)
            # The shared producer includes free angular advance. Removing its
            # supplied common rate isolates the finite convex-hull comparison.
            outgoing = tuple(
                value - mp.mpf(dt) for value in _high_precision_principal_phases(graph)
            )
            assert min(outgoing) >= lower - mp.mpf("2e-15")
            assert max(outgoing) <= upper + mp.mpf("2e-15")
        assert execution.phase_after == tuple(
            graph.nodes[node]["theta"] for node in graph
        )
        previous_width = observed.diameter_enclosure[1]


def test_heterogeneous_free_rates_can_escape_despite_small_coupling_step():
    graph = _prepared_path(phases=(0.0, 0.0, 0.0), capacities=(1.0, 2.0, 3.0))
    initial = observe_common_phase_chart(graph)
    assert initial.admitted and initial.diameter_enclosure == (0, 0)
    proposal = propose_u3_gated_phase_step(
        graph,
        tuple(graph),
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        dt=2.0,
        coupling_strength=0.5,
    )
    assert tuple(proposal) == (2.0, 4.0, 6.0)
    assert tuple(graph.nodes[node]["theta"] for node in graph) == (0.0,) * 3
    for node, value in zip(graph, proposal, strict=True):
        set_theta(graph, node, float(value))
    final = observe_common_phase_chart(graph)
    assert final.status == "excluded" and not final.admitted
    assert final.zero_cycle_winding is None and final.cycle_periods is None
    # This tree has no cycles; exclusion proves no common semicircle contains these phases,
    # not that a nonzero winding or an autonomous support change occurred.
    assert final.geometry.cycle_rank == 0


def test_prepared_winding_birth_lies_outside_common_chart_obstruction():
    case = execute_coupling_cycle_birth()
    initial = observe_common_phase_chart(case["before"])
    final = observe_common_phase_chart(case["graph"])
    assert initial.status == final.status == "excluded"
    assert initial.geometry.cycle_rank == 0 and final.geometry.cycle_rank == 1
    assert initial.zero_cycle_winding is None and final.zero_cycle_winding is None
    actual = certify_phase_winding(case["graph"], range(5))
    assert actual.winding == 1 and actual.u3_admissible


def _phase_only_graph(phases):
    graph = nx.complete_graph(len(phases))
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node]["theta"] = phase
    return graph


def test_pure_geometry_repeated_phase_and_represented_pi_boundary():
    constant = observe_common_phase_chart(_phase_only_graph((0.375,) * 3))
    assert constant.admitted and constant.diameter_enclosure == (0, 0)
    assert constant.diameter_affine == (0, 0)
    assert constant.cycle_periods == (0,) and constant.zero_cycle_winding is True

    pair = observe_common_phase_chart(_phase_only_graph((0.0, math.pi)))
    assert pair.admitted
    assert pair.diameter_affine == (Fraction.from_float(math.pi), 0)
    lower, upper = pair.semicircle_margin_enclosure
    assert 0 < lower <= upper < Fraction(1, 10**15)
    with mp.workdps(80):
        independent_margin = mp.pi - mp.mpf(math.pi)
        assert mp.mpf(lower.numerator) / lower.denominator < independent_margin
        assert independent_margin < mp.mpf(upper.numerator) / upper.denominator

    # Binary64 pi lies below true pi, its successor above; neither float alone
    # is an exact antipode. Together with zero they exclude a common semicircle.
    triple = observe_common_phase_chart(
        _phase_only_graph((0.0, math.pi, math.nextafter(math.pi, math.inf)))
    )
    assert triple.status == "excluded"
    assert triple.cycle_periods is None and triple.zero_cycle_winding is None
    assert triple.lift_turns is None and triple.diameter_enclosure is None


def test_common_chart_is_detached_and_independent_of_node_coordinates():
    graph = _phase_only_graph((math.tau - 0.25, 0.125, 0.375))
    before_nodes = tuple((node, dict(data)) for node, data in graph.nodes(data=True))
    before_edges = tuple(graph.edges(data=True))
    observed = observe_common_phase_chart(graph)
    renamed = nx.Graph()
    for node in reversed(tuple(graph)):
        renamed.add_node(f"node-{node}", **graph.nodes[node])
    for left, right in graph.edges:
        renamed.add_edge(f"node-{left}", f"node-{right}", weight=0)
    reordered = observe_common_phase_chart(renamed)
    assert observed.admitted and reordered.admitted
    assert observed.nodes != reordered.nodes
    assert observed.diameter_affine == reordered.diameter_affine
    assert observed.diameter_enclosure == reordered.diameter_enclosure
    assert observed.semicircle_margin_enclosure == reordered.semicircle_margin_enclosure
    assert observed.cycle_periods == reordered.cycle_periods == (0,)
    assert tuple((node, dict(data)) for node, data in graph.nodes(data=True)) == (
        before_nodes
    )
    assert tuple(graph.edges(data=True)) == before_edges
    graph.nodes[0]["theta"] = 1.0
    assert observed.phase[0] == Fraction.from_float(math.tau - 0.25)


def test_insufficient_pi_enclosure_abstains_without_zero_winding(monkeypatch):
    graph = _phase_only_graph((0.0, 3.1, 3.2))
    assert observe_common_phase_chart(graph).status == "excluded"
    monkeypatch.setattr(phase_chart, "_pi_bounds", lambda: (Fraction(3), Fraction(4)))
    unresolved = observe_common_phase_chart(graph)
    assert unresolved.status == "unresolved" and not unresolved.admitted
    assert unresolved.zero_cycle_winding is None and unresolved.cycle_periods is None
    assert unresolved.lift_turns is None and unresolved.diameter_enclosure is None


@pytest.mark.parametrize("phase", [True, math.nan, -0.125, math.tau])
def test_observer_rejects_invalid_or_noncanonical_phase_without_normalizing(phase):
    graph = _phase_only_graph((0.0, phase))
    with pytest.raises((TypeError, ValueError)):
        observe_common_phase_chart(graph)

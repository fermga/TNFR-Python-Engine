"""Three canonical pressure channels share a derived strict cycle coordinate."""

from fractions import Fraction
import math

import pytest

from tnfr.alias import get_attr, set_attr, set_theta
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.metrics.common import merge_and_normalize_weights
from tnfr.operators.word_execution import run_network_sequence
from tnfr.physics.cycle_support_dynamics import observe_cycle_support_balance
from tnfr.physics.emergent_particles import winding_ring
from tnfr.physics.winding_certificates import certify_phase_winding


F = Fraction


def _prepared_joint_balance(winding, vectorized):
    graph = winding_ring(8, winding)
    graph.graph.update(
        vectorized_dnfr=vectorized,
        UM_BIDIRECTIONAL=False, UM_FUNCTIONAL_LINKS=False,
        compute_delta_nfr=default_compute_delta_nfr,
    )
    weights = merge_and_normalize_weights(
        graph, "DNFR_WEIGHTS", ("phase", "epi", "vf", "topo"), default=0.0
    )
    r = F(weights["vf"]) / F(weights["epi"])
    t = F(weights["phase"]) / F(weights["epi"])
    offsets = tuple(F(v, 32) for v in (1, 0, -1, 0, 1, 0, -1, 0))
    capacity = (F(7, 8),) + (F(1),) * 7
    for node in graph:
        phase = math.tau * winding * node / 8 + math.pi * float(offsets[node])
        set_theta(graph, node, phase)
        set_attr(graph.nodes[node], ALIAS_VF, float(capacity[node]))
        # A declared balanced initial condition, not a solver update.
        set_attr(graph.nodes[node], ALIAS_EPI,
                 float(1 - r * capacity[node] - t * offsets[node]))
    default_compute_delta_nfr(graph)
    return graph, offsets, weights


def _values(graph, alias):
    return tuple(get_attr(graph.nodes[node], alias) for node in graph)


@pytest.mark.parametrize("winding", [-1, 0, 1])
@pytest.mark.parametrize("vectorized", [False, True])
def test_phase_capacity_and_epi_cancel_in_the_same_strict_chart(winding, vectorized):
    graph, offsets, weights = _prepared_joint_balance(winding, vectorized)
    observed = observe_cycle_support_balance(
        _values(graph, ALIAS_EPI), _values(graph, ALIAS_VF), offsets,
        epi_weight=weights["epi"], vf_weight=weights["vf"],
        phase_weight=weights["phase"],
    )
    certificate = certify_phase_winding(graph, tuple(graph))
    assert certificate.winding == winding
    assert certificate.minimum_u3_margin > 0.5
    assert max(map(abs, observed.pressure)) < F(1, 10**15)
    assert _values(graph, ALIAS_DNFR) == pytest.approx(
        tuple(map(float, observed.pressure)), abs=2e-15, rel=0
    )
    assert max(_values(graph, ALIAS_EPI)) > min(_values(graph, ALIAS_EPI))
    assert len(set(graph.nodes[n]["theta"] for n in graph)) > 1


@pytest.mark.parametrize("winding", [0, 1])
def test_admitted_capacity_and_phase_updates_release_a_joint_preparation(winding):
    graph, _, _ = _prepared_joint_balance(winding, True)
    before_epi = _values(graph, ALIAS_EPI)
    before_phase = tuple(graph.nodes[n]["theta"] for n in graph)
    before_capacity = _values(graph, ALIAS_VF)
    run_network_sequence(
        graph, ["coupling", "silence"],
        context={"initial_epi_nonzero": all(before_epi)},
    )
    assert _values(graph, ALIAS_EPI) == before_epi
    assert _values(graph, ALIAS_VF) != before_capacity
    assert tuple(graph.nodes[n]["theta"] for n in graph) != before_phase
    assert max(map(abs, _values(graph, ALIAS_DNFR))) > 1e-4
    assert certify_phase_winding(graph, tuple(graph)).winding == winding
    assert all(tuple(graph.nodes[n]["glyph_history"])[-2:] == ("UM", "SHA")
               for n in graph)

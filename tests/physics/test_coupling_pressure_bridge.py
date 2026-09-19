"""The restricted cycle gap law reads the canonical phase-pressure channel."""

import math

import pytest

from tnfr.alias import get_attr, set_theta
from tnfr.constants import DEFAULTS
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.dynamics.dnfr import default_compute_delta_nfr, dnfr_phase_only
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.physics.emergent_particles import winding_ring
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils.numeric import angle_diff

# Finite libm/backend comparisons, not a certified roundoff enclosure.
_ATOL = 8 * math.ulp(1.0)


@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("winding", [-1, 0, 1])
@pytest.mark.parametrize("vectorized", [False, True])
def test_gap_divergence_matches_canonical_pressure_and_nodal_euler(
    size, winding, vectorized
):
    graph = winding_ring(size, winding)
    graph.graph["vectorized_dnfr"] = vectorized
    for node in graph:
        set_theta(
            graph,
            node,
            graph.nodes[node]["theta"]
            + math.pi / 16 * math.sin(math.tau * node / size),
        )
    phases = [graph.nodes[node]["theta"] for node in graph]
    gaps = [angle_diff(phases[(i + 1) % size], phases[i]) for i in graph]
    expected_phase = [(gaps[i] - gaps[i - 1]) / math.tau for i in graph]
    dnfr_phase_only(graph)
    assert [get_attr(graph.nodes[i], ALIAS_DNFR) for i in graph] == pytest.approx(
        expected_phase, abs=_ATOL, rel=0
    )

    # Constant EPI, capacity and degree remove the other gradient channels.
    # Restore the canonical default mixture after the phase-only hook.
    graph.graph["DNFR_WEIGHTS"] = dict(DEFAULTS["DNFR_WEIGHTS"])
    default_compute_delta_nfr(graph)
    phase_weight = DEFAULTS["DNFR_WEIGHTS"]["phase"]
    expected_pressure = [phase_weight * value for value in expected_phase]
    actual_pressure = [get_attr(graph.nodes[i], ALIAS_DNFR) for i in graph]
    assert actual_pressure == pytest.approx(expected_pressure, abs=_ATOL, rel=0)

    before = [get_attr(graph.nodes[i], ALIAS_EPI) for i in graph]
    update_epi_via_nodal_equation(graph, dt=0.125, t=0.0, method="euler")
    assert [get_attr(graph.nodes[i], ALIAS_EPI) for i in graph] == pytest.approx(
        [value + 0.125 * pressure for value, pressure in zip(before, actual_pressure)],
        abs=_ATOL,
        rel=0,
    )
    assert [graph.nodes[i]["theta"] for i in graph] == phases
    assert certify_phase_winding(graph, range(size)).winding == winding


@pytest.mark.parametrize("size,winding", [(8, 1), (16, 1), (16, 2), (16, -2)])
def test_nonzero_winding_can_have_zero_canonical_pressure(size, winding):
    graph = winding_ring(size, winding)
    default_compute_delta_nfr(graph)
    certificate = certify_phase_winding(graph, range(size))
    assert certificate.winding == winding
    assert certificate.u3_admissible
    pressure_max = max(
        abs(get_attr(data, ALIAS_DNFR)) for _, data in graph.nodes(data=True)
    )
    assert pressure_max < _ATOL
    # Zero pressure is compatible with a twist; it does not mean equal phases.
    assert certificate.minimum_u3_margin > 0

"""Finite mobility distinguishes frozen form from pressure equilibrium."""

from fractions import Fraction

import pytest

from tnfr.alias import get_attr
from tnfr.constants import DEFAULTS
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.constants.canonical import SHA_VF_FACTOR
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.operators.word_execution import run_network_sequence
from tnfr.physics.emergent_particles import winding_ring
from tnfr.physics.reversible_eigenmode_reference import _negative_exp_bounds
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils import normalize_weights


def _channel(graph, aliases):
    return tuple(get_attr(graph.nodes[node], aliases) for node in graph)


@pytest.mark.parametrize("winding", [0, 1])
def test_uniform_silence_slows_a_bump_without_removing_its_pressure(winding):
    graph = winding_ring(8, winding)
    for node in graph:
        graph.nodes[node]["EPI"] = 0.75 if node == 0 else 0.5
    graph.graph["compute_delta_nfr"] = default_compute_delta_nfr
    default_compute_delta_nfr(graph)
    epi = _channel(graph, ALIAS_EPI)
    phases = _channel(graph, ALIAS_THETA)
    pressure = _channel(graph, ALIAS_DNFR)
    assert max(map(abs, pressure)) > 0

    run_network_sequence(
        graph, ["silence"], context={"initial_epi_nonzero": True}
    )
    assert _channel(graph, ALIAS_EPI) == epi
    assert _channel(graph, ALIAS_THETA) == phases
    assert _channel(graph, ALIAS_DNFR) == pressure
    assert _channel(graph, ALIAS_VF) == (SHA_VF_FACTOR,) * 8
    assert certify_phase_winding(graph, range(8)).winding == winding
    rates = tuple(nu * delta for nu, delta in zip(
        _channel(graph, ALIAS_VF), pressure, strict=True
    ))
    assert max(map(abs, rates)) == SHA_VF_FACTOR * max(map(abs, pressure))
    assert all(tuple(graph.nodes[node]["glyph_history"]) == ("SHA",)
               for node in graph)


def test_geometric_capacity_clock_leaves_a_nonzero_exact_diffusion_mode():
    # A C8 mode (1,0,-1,0,1,0,-1,0) has exact L_rw eigenvalue one.
    # Each declared exact flow of duration h precedes a scalar SHA reset.
    # This reference neither executes nor certifies an infinite runtime word.
    q = Fraction.from_float(SHA_VF_FACTOR)
    weights = normalize_weights(
        DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
    )
    e = Fraction.from_float(weights["epi"])
    h, nu0 = Fraction(1, 8), Fraction(1)
    complete_clock = h * nu0 / (1 - q)
    prefix_clock = sum((h * nu0 * q**k for k in range(8)), Fraction(0))
    tail_clock = h * nu0 * q**8 / (1 - q)
    assert prefix_clock + tail_clock == complete_clock
    limit_lower, limit_upper = _negative_exp_bounds(e * complete_clock)
    prefix_lower, _ = _negative_exp_bounds(e * prefix_clock)
    assert 0 < limit_lower <= limit_upper < prefix_lower < 1
    # Remaining pressure amplitude e*exp(-e*clock_infinity) is also nonzero,
    # although the limiting nodal rate vanishes as the capacity tends to zero.
    assert e * limit_lower > 0

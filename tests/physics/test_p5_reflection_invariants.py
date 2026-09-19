"""Exact P5 reflection invariants and their retained diagnostic scope.

These tests identify reflected fine states without discarding their two
continuous hidden coordinates. All graph inputs are declared snapshots;
there is no trajectory, fitted dynamics, or emergent-reflection claim.
"""

from copy import deepcopy
from fractions import Fraction as F
from math import log

import networkx as nx
import pytest

from tests.physics.test_forced_epi_closure import _apply
from tnfr.mathematics.krylov import exact_rank
from tnfr.physics.canonical import compute_structural_potential
from tnfr.physics.geometry_realization import observe_forced_support_tetrad_dependencies
from tnfr.physics.p5_reduction import (
    decode_p5_reflection_invariants,
    observe_p5_reflection_invariants,
    p5_reduction_geometry,
    reduce_p5_state,
)

ORBIT = (F(3, 8), F(-1, 4), F(5, 8))
R_ROW = (F(1, 2), 0, 0, 0, F(-1, 2))
S_ROW = (0, F(1, 2), 0, F(-1, 2), 0)
BLOCKS = ((0, 4), (1, 3), (2,))


def _epi(orbit, hidden):
    a, middle, center = orbit
    r, s = hidden
    return (a + r, middle + s, center, middle - s, a - r)


@pytest.mark.parametrize(
    "hidden", [(F(1, 2), F(-3, 4)), (0, F(3, 4)), (F(1, 2), 0), (0, 0)]
)
def test_invariant_rates_are_the_fine_nodal_chain_rule_and_tangent_to_the_cone(hidden):
    epi = _epi(ORBIT, hidden)
    nu = F(7, 4)
    observed = observe_p5_reflection_invariants(epi, capacity=nu)
    geometry = p5_reduction_geometry(nu)
    fine_rate = tuple(-value for value in _apply(geometry.micro_generator, epi))
    r, s = hidden
    r_rate, s_rate = _apply((R_ROW, S_ROW), fine_rate)
    assert observed.geometry == geometry
    assert observed.reduced == reduce_p5_state(epi)
    assert observed.orbit_epi == ORBIT
    assert observed.hidden_epi == hidden
    assert observed.orbit_rate == _apply(geometry.orbit_projection, fine_rate)
    assert observed.hidden_rate == (r_rate, s_rate)
    assert observed.quadratic_invariants == (r * r, r * s, s * s)
    assert observed.invariant_rate == (
        2 * r * r_rate,
        s * r_rate + r * s_rate,
        2 * s * s_rate,
    )
    u, v, w = observed.quadratic_invariants
    u_rate, v_rate, w_rate = observed.invariant_rate
    assert u >= 0 and w >= 0 and u * w == v * v
    assert u_rate * w + u * w_rate - 2 * v * v_rate == 0

    reflected = observe_p5_reflection_invariants(epi[::-1], capacity=nu)
    assert reflected.orbit_epi == observed.orbit_epi
    assert reflected.quadratic_invariants == observed.quadratic_invariants
    assert reflected.orbit_rate == observed.orbit_rate
    assert reflected.invariant_rate == observed.invariant_rate
    assert reflected.hidden_rate == tuple(-value for value in observed.hidden_rate)
    representative = decode_p5_reflection_invariants(
        ORBIT, observed.quadratic_invariants
    )
    assert representative == observed.representative_epi == reflected.representative_epi
    assert representative in (epi, epi[::-1])


@pytest.mark.parametrize("hidden", [(2, 3), (0, 2), (2, 0), (0, 0)])
def test_reflection_identification_keeps_five_generic_continuous_dimensions(hidden):
    r, s = map(F, hidden)
    observed = observe_p5_reflection_invariants(_epi(ORBIT, (r, s)))
    jacobian = (
        *observed.geometry.orbit_projection,
        tuple(2 * r * value for value in R_ROW),
        tuple(s * left + r * right for left, right in zip(R_ROW, S_ROW, strict=True)),
        tuple(2 * s * value for value in S_ROW),
    )
    assert exact_rank(jacobian) == observed.jacobian_rank == (5 if r or s else 3)
    # In independent (r,s) coordinates, the two relevant 2x2 minors are
    # 2*r^2 and 2*s^2. One is nonzero at every nonzero hidden vector.
    first_minor = (2 * r) * r
    second_minor = s * (2 * s)
    assert bool(first_minor or second_minor) == bool(r or s)
    # The singular derivative at hidden zero is not a three-dimensional
    # neighborhood: arbitrarily small nonzero rational inputs have rank five.


def test_diagonal_squares_lose_the_sign_product_needed_for_pressure_and_rates():
    same_sign = observe_p5_reflection_invariants(_epi((0, 0, 0), (1, 1)))
    opposite_sign = observe_p5_reflection_invariants(_epi((0, 0, 0), (1, -1)))
    assert same_sign.quadratic_invariants == (1, 1, 1)
    assert opposite_sign.quadratic_invariants == (1, -1, 1)
    assert same_sign.invariant_rate[0] == 0
    assert opposite_sign.invariant_rate[0] == -4
    first_pressure = tuple(
        -value
        for value in _apply(same_sign.geometry.micro_generator, same_sign.reduced.epi)
    )
    second_pressure = tuple(
        -value
        for value in _apply(
            opposite_sign.geometry.micro_generator, opposite_sign.reduced.epi
        )
    )
    assert first_pressure == (0, F(-1, 2), 0, F(1, 2), 0)
    assert second_pressure == (-2, F(3, 2), 0, F(-3, 2), 2)
    first_product = 1 / ((1 + abs(first_pressure[0])) * (1 + abs(first_pressure[1])))
    second_product = 1 / ((1 + abs(second_pressure[0])) * (1 + abs(second_pressure[1])))
    assert first_product == F(2, 3)
    assert second_product == F(2, 15)


def _fields(epi):
    graph = nx.path_graph(5)
    pressure = tuple(
        -value for value in _apply(p5_reduction_geometry().micro_generator, epi)
    )
    for node, x, p in zip(graph, epi, pressure, strict=True):
        graph.nodes[node].update(EPI=x, nu_f=1, theta=0, delta_nfr=p)
    for edge in graph.edges:
        graph.edges[edge].update(weight=1, length=1)
    graph.graph["DNFR_WEIGHTS"] = {"phase": 0, "epi": 1, "vf": 0, "topo": 0}
    before = deepcopy(graph)
    fields = observe_forced_support_tetrad_dependencies(graph, BLOCKS)
    potential_map = compute_structural_potential(graph)
    potential = tuple(potential_map[i] for i in graph)
    assert graph.graph == before.graph
    assert tuple(graph.nodes(data=True)) == tuple(before.nodes(data=True))
    assert tuple(graph.edges(data=True)) == tuple(before.edges(data=True))
    return fields, potential


def test_complete_quadratic_data_preserves_fields_up_to_the_declared_reflection():
    # Nonzero orbit form avoids a trivial zero averaged-potential comparison.
    original = _epi((1, 0, 0), (-2, -1))
    observed = observe_p5_reflection_invariants(original)
    representative = decode_p5_reflection_invariants(
        observed.orbit_epi, observed.quadratic_invariants
    )
    assert representative == original[::-1] != original
    first, first_potential = _fields(original)
    reflected, reflected_potential = _fields(original[::-1])
    decoded, decoded_potential = _fields(representative)
    assert first.geometry.model_pressure == reflected.geometry.model_pressure[::-1]
    assert first.geometry.model_potential == reflected.geometry.model_potential[::-1]
    assert first_potential == pytest.approx(
        reflected_potential[::-1], rel=2e-14, abs=1e-15
    )
    assert decoded_potential == reflected_potential
    assert (
        first.geometry.model_output
        == reflected.geometry.model_output
        == decoded.geometry.model_output
    )
    assert (
        first.geometry.realization.output_state
        == reflected.geometry.realization.output_state
    )
    assert first.mean_phase_gradient == first.mean_phase_curvature == (0,) * 3
    assert (
        decoded.mean_phase_gradient
        == reflected.mean_phase_gradient
        == first.mean_phase_gradient
    )
    assert (
        decoded.mean_phase_curvature
        == reflected.mean_phase_curvature
        == first.mean_phase_curvature
    )
    fit = first.observed_coherence_length
    assert (
        fit == reflected.observed_coherence_length == decoded.observed_coherence_length
    )
    assert fit.method == "autocorrelation_fit" and fit.fit_available
    # The three usable distance-bin means have endpoints 5/9 and 4/9.
    assert fit.value == pytest.approx(2 / log(F(5, 4)), rel=2e-14)


@pytest.mark.parametrize(
    "orbit, quadratic, error",
    [
        ((0, 0), (1, 0, 0), ValueError),
        ((0, 0, 0), (1, 0), ValueError),
        ((True, 0, 0), (1, 0, 0), TypeError),
        ((0, 0, 0), (True, 0, 0), TypeError),
        ((float("nan"), 0, 0), (1, 0, 0), ValueError),
        ((0, 0, 0), (1, float("inf"), 0), ValueError),
        ((0, 0, 0), (-1, 0, 0), ValueError),
        ((0, 0, 0), (0, 0, -1), ValueError),
        ((0, 0, 0), (1, 0, 1), ValueError),
        ((0, 0, 0), (0, 1, 0), ValueError),
    ],
)
def test_decoder_rejects_malformed_or_non_rank_one_positive_data(
    orbit, quadratic, error
):
    with pytest.raises(error):
        decode_p5_reflection_invariants(orbit, quadratic)


@pytest.mark.parametrize("quadratic", [(2, 0, 0), (0, 0, 2), (2, 2, 2)])
def test_real_rank_one_cones_without_rational_lifts_are_explicitly_out_of_scope(
    quadratic,
):
    u, v, w = quadratic
    assert u >= 0 and w >= 0 and u * w == v * v
    with pytest.raises(ValueError, match="rational"):
        decode_p5_reflection_invariants((0, 0, 0), quadratic)


@pytest.mark.parametrize("capacity", [0, -1, True, float("nan"), float("inf")])
def test_observer_rejects_invalid_capacity(capacity):
    with pytest.raises(TypeError if capacity is True else ValueError):
        observe_p5_reflection_invariants((0,) * 5, capacity=capacity)


@pytest.mark.parametrize(
    "epi", [(0,) * 4, (0, 0, True, 0, 0), (0, 0, float("inf"), 0, 0)]
)
def test_observer_rejects_invalid_fine_form(epi):
    with pytest.raises(TypeError if epi[2] is True else ValueError):
        observe_p5_reflection_invariants(epi)


def test_decoder_and_observer_detach_ordered_rational_inputs():
    epi = list(_epi(ORBIT, (F(-2, 3), F(4, 5))))
    before = tuple(epi)
    observed = observe_p5_reflection_invariants(iter(epi))
    orbit = list(observed.orbit_epi)
    quadratic = list(observed.quadratic_invariants)
    decoded = decode_p5_reflection_invariants(iter(orbit), iter(quadratic))
    assert decoded == before[::-1]
    assert observed.representative_epi == decoded
    epi[0] = 99
    orbit[0] = 99
    quadratic[0] = 99
    assert observed.reduced.epi == before
    assert observed.representative_epi == decoded

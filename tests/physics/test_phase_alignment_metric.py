"""Instantaneous pair-cost geometry of the canonical phase-pressure channel.

The exact-real identities do not install a phase law or a potential-energy
coefficient. Numeric current, curvature and pressure retain separate rounding;
zero resultants and the half-turn cut lie outside the positive metric chart.
"""

from fractions import Fraction as Q
from math import atan2, isfinite, pi, sin, sqrt

import networkx as nx
import pytest

from tnfr.mathematics.phasor_resultant import reduce_phasor_components
from tnfr.physics.canonical import compute_phase_curvature, observe_phase_curvature
from tnfr.physics.extended import compute_phase_current
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_curvature import UndefinedPhaseCurvatureError
from tnfr.physics.phase_response import derive_phase_response


def _state(graph, phases):
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0,
        "epi": 0.0,
        "vf": 0.0,
        "topo": 0.0,
    }
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node].update(EPI=0.5, nu_f=1.0, theta=phase, delta_nfr=0.0)
    return graph


def test_pair_cost_gradient_and_positive_metric_have_the_exact_regular_identity():
    s = pytest.importorskip("sympy")
    graph = nx.path_graph(3)
    theta = s.symbols("theta0:3", real=True)
    cost = sum(1 - s.cos(theta[j] - theta[i]) for i, j in graph.edges)
    for i in graph:
        gradient = s.diff(cost, theta[i])
        torque = sum(s.sin(theta[j] - theta[i]) for j in graph.neighbors(i))
        assert s.simplify(gradient + torque) == 0
    delta = s.Symbol("delta", real=True)
    magnitude = s.Symbol("magnitude", positive=True)
    mobility = delta / (s.pi * magnitude * s.sin(delta))
    gradient = -magnitude * s.sin(delta)
    assert s.simplify(-mobility * gradient - delta / s.pi) == 0
    assert s.limit(mobility, delta, 0) == 1 / (s.pi * magnitude)
    assert s.limit(mobility, delta, s.pi, dir="-") == s.oo
    # On -pi<delta<pi, delta/sin(delta)>0 with the removable zero limit.
    # This is a representation of the given g, not a choice theta'=g.


def test_regular_path_links_actual_current_curvature_and_pressure_with_residuals():
    graph = _state(nx.path_graph(3), (0.0, pi / 6, pi / 3))
    current = compute_phase_current(graph)
    curvature = compute_phase_curvature(graph)
    captured = capture_non_epi_forcing(graph)
    ideal_gradient = (Q(-1, 2), Q(0), Q(1, 2))
    ideal_pressure = (Q(1, 6), Q(0), Q(-1, 6))
    degrees = (1, 2, 1)
    # V=2-sqrt(3), |S|=(1,sqrt(3),1), delta=(pi/6,0,-pi/6).
    mobility = (1 / 3, 1 / (pi * sqrt(3)), 1 / 3)
    assert all(
        abs(graph.nodes[i]["theta"] - graph.nodes[j]["theta"]) < pi / 2
        for i, j in graph.edges
    )
    assert tuple(-d * current[i] for i, d in enumerate(degrees)) == pytest.approx(
        ideal_gradient, abs=1e-15
    )
    assert tuple(map(float, captured.phase_gradient)) == pytest.approx(
        ideal_pressure, abs=1e-15
    )
    assert tuple(
        -m * gradient for m, gradient in zip(mobility, ideal_gradient, strict=True)
    ) == pytest.approx(ideal_pressure, abs=1e-15)
    assert curvature == pytest.approx({0: -pi / 6, 1: 0.0, 2: pi / 6}, abs=1e-15)
    assert current[0] * curvature[0] < 0 and current[2] * curvature[2] < 0
    assert tuple(-curvature[i] / pi for i in graph) == pytest.approx(
        tuple(map(float, captured.phase_gradient)), abs=1e-15
    )
    # Record represented-minus-ideal pressure defects explicitly. A binary64
    # endpoint cannot equal 1/6 exactly, even on this well-conditioned chart.
    residual = tuple(
        actual - ideal
        for actual, ideal in zip(captured.phase_gradient, ideal_pressure, strict=True)
    )
    assert residual[0] != 0 and residual[2] != 0
    assert max(map(abs, residual)) < Q(1, 2**49)
    assert captured.full_kernel_pressure == captured.phase_gradient


def test_state_dependent_metric_coexists_with_the_fixed_metric_star_obstruction():
    alpha = atan2(4, 3)
    graph = _state(nx.star_graph(3), (0.0, 0.0, 0.0, alpha))
    captured = capture_non_epi_forcing(graph)
    current = compute_phase_current(graph)
    vectors = ((Q(1), Q(0)),) * 3 + ((Q(3, 5), Q(4, 5)),)
    gram = tuple(
        tuple(sum(a * b for a, b in zip(left, right, strict=True)) for right in vectors)
        for left in vectors
    )
    reference = derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=((1, 2, 3), (0,), (0,), (0,)),
        receiver_sources=((0,), (1,), (2,), (3,)),
        phase_factor=1,
    )
    assert all(gram[0][j] > 0 for j in (1, 2, 3))
    assert reference.mean_resultant_squared == (Q(37, 5), 1, 1, 1)
    assert 3 * reference.mean_response[0][1] - reference.mean_response[1][0] == Q(2, 37)
    delta = atan2(4, 13)
    ideal_gradient = (Q(-4, 5), 0, 0, Q(4, 5))
    mobility = (5 * delta / (4 * pi), 1 / pi, 1 / pi, 5 * alpha / (4 * pi))
    assert all(value > 0 for value in mobility)
    assert mobility[0] != mobility[1] / 3
    assert tuple(-d * current[i] for i, d in enumerate((3, 1, 1, 1))) == pytest.approx(
        ideal_gradient, abs=1e-15
    )
    assert tuple(
        -m * gradient for m, gradient in zip(mobility, ideal_gradient, strict=True)
    ) == pytest.approx(tuple(map(float, captured.phase_gradient)), abs=1e-15)
    # Dg=-M Hess(V)-(DM) grad(V): pointwise positive M does not assert
    # symmetry of diag(3,1,1,1)*Dg throughout a neighborhood.


def test_half_turn_zero_ideal_torque_is_not_zero_canonical_phase_pressure():
    graph = _state(nx.path_graph(2), (0.0, pi))
    current = compute_phase_current(graph)
    captured = capture_non_epi_forcing(graph)
    # Exact antipodal phasors have V=2, gradient V=0 and |S|=1.
    resultant = reduce_phasor_components(((-1.0, 0.0),))
    assert not resultant.joint_zero and resultant.angle == pi
    assert captured.phase_gradient == (Q(-1), Q(-1))
    assert all(0 < abs(value) < 2e-16 for value in current.values())
    # sin(represented pi) is not exactly zero; wrapping both endpoint gaps
    # to -pi produces tiny signed currents, not the ideal zero torque.
    assert all(isfinite(value) for value in current.values())
    assert max(abs(float(value)) for value in captured.phase_gradient) == 1
    # The regular positive metric excludes this cut: finite mobility cannot
    # multiply an exact zero gradient into a nonzero canonical pressure.


def test_zero_resultant_blocks_the_metric_even_when_current_is_finite():
    graph = _state(nx.star_graph(4), (0.3, 0.0, 0.0, pi, -pi))
    observed = observe_phase_curvature(graph)
    center = observed.rows[0]
    reduced = reduce_phasor_components(observed.components[1:])
    assert reduced.joint_zero and reduced.angle is None
    assert center.resultant == reduced
    assert (
        center.curvature is None and center.status == "undefined_represented_resultant"
    )
    with pytest.raises(UndefinedPhaseCurvatureError):
        compute_phase_curvature(graph)
    current = compute_phase_current(graph)
    assert all(isfinite(value) for value in current.values())
    assert abs(current[0]) < 1e-15
    # The existing fused pressure branch still produces a number at this
    # represented cancellation. That does not authenticate a regular chart.
    captured = capture_non_epi_forcing(graph)
    assert all(isfinite(float(value)) for value in captured.phase_gradient)
    assert "represented_zero_does_not_certify_exact_real_trigonometric_zero" in (
        observed.scope
    )


def test_pair_cost_uses_unique_support_not_parallel_multiplicity_or_conductance():
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edge(0, 0, weight=5.0)
    graph.add_edge(0, 1, weight=0.0)
    graph.add_edge(0, 1, weight=7.0)
    graph.add_edge(1, 2, weight=0.0)
    _state(graph, (0.0, pi / 3, pi / 3))
    captured = capture_non_epi_forcing(graph)
    current = compute_phase_current(graph)
    degrees = tuple(len(row) for row in captured.snapshot.support_neighbors)
    assert degrees == (2, 2, 1)
    assert tuple(dict(graph.degree()).values()) != degrees
    assert captured.snapshot.epi_gradient == (0,) * 3
    # Unique non-loop pairs are (0,1),(1,2): V=1/2; the loop cost is zero.
    unique_pairs = tuple(
        (i, j)
        for i, row in enumerate(captured.snapshot.support_neighbors)
        for j in row
        if i < j
    )
    assert unique_pairs == ((0, 1), (1, 2))
    ideal_gradient = (-sqrt(3) / 2, sqrt(3) / 2, 0.0)
    assert tuple(-d * current[i] for i, d in enumerate(degrees)) == pytest.approx(
        ideal_gradient, abs=1e-15
    )
    assert tuple(map(float, captured.phase_gradient)) == pytest.approx(
        (1 / 6, -1 / 6, 0.0), abs=1e-15
    )
    # A self-loop alters |S| and mean degree, but contributes no sine torque.
    # Summing MultiGraph edges or weighting this V by conductance is a different cost.


def test_nonreciprocal_support_current_is_not_the_gradient_of_an_undirected_pair_cost():
    graph = _state(nx.DiGraph(((0, 1),)), (0.0, pi / 3))
    current = compute_phase_current(graph)
    assert current[0] == pytest.approx(sqrt(3) / 2, abs=1e-15)
    assert current[1] == 0.0
    # V=1-cos(theta1-theta0) has the nonzero second partial below, while
    # successor-only current at node 1 is zero by its isolate convention.
    pair_gradient_at_one = sin(pi / 3)
    assert pair_gradient_at_one > 0
    assert pair_gradient_at_one != -len(tuple(graph.neighbors(1))) * current[1]
    # Reciprocal support is a hypothesis of the metric identity, not a
    # property silently imposed on this more general diagnostic API.

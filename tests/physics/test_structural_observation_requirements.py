"""Fixed task-dependent observation witnesses, without trajectory evaluation.

Exact pressure/phase calculations are independent mathematical oracles.
Binary64 field comparisons check only the declared small supported fixtures;
they do not turn a sampled rank or trigonometric residual into a global proof.
"""

import cmath
import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tests.physics.test_coherence_observation_scope import _radial_star
from tests.physics.test_tetrad_observability import _state_with_uniform_capacity
from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.canonical import compute_canonical_nodal_derivative
from tnfr.physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    observe_coherence_profile,
)
from tnfr.physics.observability import (
    linear_observability_certificate,
    minimal_distinguishing_channels,
    observation_signature,
    tetrad_observation_vector,
)
from tnfr.utils.cache import reset_global_cache

Q = Fraction
REL = 2e-13
ABS = 2e-14
KINDS = ("path", "cycle", "disconnected")


@pytest.fixture(autouse=True)
def _clear_field_cache():
    reset_global_cache()
    yield
    reset_global_cache()


def _directed_pressure_star(center_pressure):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(13))
    for node in graph:
        radius = 0 if node == 0 else 1 + (node - 1) // 3
        pressure = center_pressure if node == 0 else 2**radius - 1
        set_attr(graph.nodes[node], ALIAS_DNFR, pressure)
        set_attr(graph.nodes[node], ALIAS_EPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_VF, 1.0)
        set_attr(graph.nodes[node], ALIAS_THETA, 0.0)
        if node:
            graph.add_edge(0, node, length=radius, weight=1.0)
    return graph


def _held_pressure_rates(graph):
    """Evaluate the complete fixture's initial nodal row; no time step."""
    return tuple(
        compute_canonical_nodal_derivative(
            get_attr(graph.nodes[node], ALIAS_VF),
            get_attr(graph.nodes[node], ALIAS_DNFR),
        ).derivative
        for node in graph
    )


def test_directed_kernel_hides_center_pressure_from_full_tetrad_but_not_profile():
    first, second = (_directed_pressure_star(value) for value in (0, 1))
    exact_center = 3 * sum(
        (Q(2**radius - 1, radius**2) for radius in range(1, 5)), Q(0)
    )
    assert exact_center == Q(499, 48)

    for graph, center_coherence in ((first, Q(1)), (second, Q(1, 2))):
        potential = compute_structural_potential(graph)
        assert potential[0] == pytest.approx(float(exact_center), rel=REL, abs=ABS)
        assert tuple(potential[node] for node in range(1, 13)) == (0.0,) * 12
        assert tuple(compute_phase_gradient(graph).values()) == (0.0,) * 13
        assert tuple(compute_phase_curvature(graph).values()) == (0.0,) * 13
        profile = observe_coherence_profile(graph)
        assert profile.positive_pair_count == 12
        assert tuple(row.pair_count for row in profile.bins) == (3,) * 4
        assert tuple(row.product_mean for row in profile.bins) == pytest.approx(
            tuple(float(center_coherence / 2**radius) for radius in range(1, 5)),
            rel=REL,
            abs=ABS,
        )
        assert profile.fit.value == pytest.approx(1 / math.log(2), rel=REL, abs=ABS)
        assert profile.fit.intercept == pytest.approx(
            math.log(float(center_coherence)), rel=REL, abs=ABS
        )
    np.testing.assert_allclose(
        tetrad_observation_vector(first, representation="full"),
        tetrad_observation_vector(second, representation="full"),
        rtol=REL,
        atol=ABS,
    )
    rates_first, rates_second = _held_pressure_rates(first), _held_pressure_rates(
        second
    )
    assert (rates_first[0], rates_second[0]) == (0.0, 1.0)
    assert rates_first[1:] == rates_second[1:]
    assert observe_coherence_profile(first) != observe_coherence_profile(second)


def test_directed_pressure_kernel_has_an_exact_linear_completion():
    symbolic = pytest.importorskip("sympy")
    # This is the analytic outgoing distance kernel of the same fixed star.
    kernel = symbolic.zeros(13)
    for node in range(1, 13):
        radius = 1 + (node - 1) // 3
        kernel[0, node] = symbolic.Rational(1, radius**2)
    assert kernel.rank() == 1
    center = symbolic.eye(13)[:, 0]
    assert kernel * center == symbolic.zeros(13, 1)

    # Keep all pressure entries except leaf 1. Its nonzero coefficient in
    # the potential supplies the one remaining coordinate. These twelve
    # independent scalar additions exactly match the kernel dimension.
    coordinates = symbolic.eye(13).extract((0,) + tuple(range(2, 13)), range(13))
    assert coordinates.rows == 13 - kernel.rank()
    assert kernel.col_join(coordinates).rank() == 13
    assert kernel.col_join(coordinates[:-1, :]).rank() == 12


def test_capacity_extension_resolves_the_same_clock_held_pressure_rate():
    slow, fast = (_state_with_uniform_capacity(value) for value in (1.0, 2.0))
    slow_signature = observation_signature(slow, tetrad_representation="full")
    fast_signature = observation_signature(fast, tetrad_representation="full")
    assert slow_signature == fast_signature
    slow_observed = observation_signature(
        slow, tetrad_representation="full", include_capacity=True
    )
    fast_observed = observation_signature(
        fast, tetrad_representation="full", include_capacity=True
    )
    assert minimal_distinguishing_channels(slow_observed, fast_observed) == (
        "capacity",
    )
    assert _held_pressure_rates(slow) == (-1.0, 0.5, 0.5, 0.0)
    assert _held_pressure_rates(fast) == (-2.0, 1.0, 1.0, 0.0)


def test_existing_form_primitive_resolves_the_same_static_profile_case():
    first, second = _radial_star(0), _radial_star(1)
    assert observe_coherence_profile(first) == observe_coherence_profile(second)
    options = {"tetrad_representation": "full", "include_pressure": True}
    assert observation_signature(first, **options) == observation_signature(
        second, **options
    )
    observed = tuple(
        observation_signature(graph, include_form=True, **options)
        for graph in (first, second)
    )
    assert minimal_distinguishing_channels(*observed) == ("form",)
    assert observed[0].form == (0.0,) * 7
    assert observed[1].form == (1.0,) * 7
    # The existing coherence-scope test proves the exact xi tangent for the
    # supplied oscillator. This control checks the added primitive channel,
    # without duplicating that proof or performing a new evolution.
    assert _held_pressure_rates(first) == _held_pressure_rates(second)


def _phase_fixture(kind):
    graph = nx.cycle_graph(4) if kind == "cycle" else nx.path_graph(4)
    if kind == "disconnected":
        graph.remove_edge(1, 2)
    nx.set_edge_attributes(graph, 1.0, "length")
    nx.set_edge_attributes(graph, 1.0, "weight")
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_THETA, node * math.pi / 4)
    # One reversed measurement checks the inverse phasor convention.
    edges = ((0, 1), (2, 3)) if kind == "disconnected" else ((0, 1), (2, 1), (2, 3))
    forest = nx.Graph()
    forest.add_nodes_from(graph)
    forest.add_edges_from(edges)
    return graph, forest, edges


def _exact_phasors(symbolic):
    root_half = symbolic.sqrt(2) / 2
    return (
        symbolic.Integer(1),
        root_half * (1 + symbolic.I),
        symbolic.I,
        root_half * (-1 + symbolic.I),
    )


@pytest.mark.parametrize("kind", KINDS)
def test_exact_spanning_forest_reconstruction_and_component_rotation_scope(kind):
    symbolic = pytest.importorskip("sympy")
    graph, forest, edges = _phase_fixture(kind)
    phase = _exact_phasors(symbolic)
    measured = {
        edge: symbolic.simplify(phase[edge[1]] * symbolic.conjugate(phase[edge[0]]))
        for edge in edges
    }
    components = tuple(nx.connected_components(graph))
    assert len(edges) == len(graph) - len(components)
    decoded = {}
    rotated = list(phase)
    for index, component in enumerate(components):
        root = min(component)
        rotation = symbolic.I if index == 0 else -1
        for node in component:
            path = nx.shortest_path(forest, root, node)
            factors = (
                (
                    measured[(left, right)]
                    if (left, right) in measured
                    else symbolic.conjugate(measured[(right, left)])
                )
                for left, right in zip(path, path[1:])
            )
            decoded[node] = symbolic.simplify(symbolic.prod(factors))
            assert symbolic.simplify(decoded[node] - phase[node] / phase[root]) == 0
            assert symbolic.simplify(decoded[node] * phase[root] - phase[node]) == 0
            rotated[node] = symbolic.simplify(rotation * phase[node])
        assert symbolic.simplify(rotated[root] - phase[root]) != 0
    for left, right in edges:
        after = rotated[right] * symbolic.conjugate(rotated[left])
        assert symbolic.simplify(after - measured[(left, right)]) == 0
    if kind == "cycle":
        # The unmeasured closing edge is fixed by the forest decoder, and
        # its complete cycle product is one; it is a consistency condition.
        closing = phase[0] * symbolic.conjugate(phase[3])
        assert (
            symbolic.simplify(decoded[0] * symbolic.conjugate(decoded[3]) - closing)
            == 0
        )
        product = symbolic.prod(
            phase[right] * symbolic.conjugate(phase[left])
            for left, right in ((0, 1), (1, 2), (2, 3), (3, 0))
        )
        assert symbolic.simplify(product) == 1


@pytest.mark.parametrize("kind", KINDS)
def test_omitting_a_forest_edge_leaves_an_extra_relative_rotation(kind):
    symbolic = pytest.importorskip("sympy")
    graph, forest, edges = _phase_fixture(kind)
    phase = _exact_phasors(symbolic)
    removed = edges[0]
    forest.remove_edge(*removed)
    shifted_component = nx.node_connected_component(forest, removed[1])
    shifted = tuple(
        symbolic.I * value if node in shifted_component else value
        for node, value in enumerate(phase)
    )
    assert removed[0] not in shifted_component
    assert nx.has_path(graph, *removed)
    for left, right in edges[1:]:
        before = phase[right] * symbolic.conjugate(phase[left])
        after = shifted[right] * symbolic.conjugate(shifted[left])
        assert symbolic.simplify(after - before) == 0
    left, right = removed
    before = phase[right] * symbolic.conjugate(phase[left])
    after = shifted[right] * symbolic.conjugate(shifted[left])
    assert symbolic.simplify(after - before) != 0
    # For C4 this observes only the selected incomplete forest. Observing
    # the closing edge instead would reconnect the measurement graph.


@pytest.mark.parametrize("kind", KINDS)
def test_forest_decoding_preserves_existing_phase_readouts_with_declared_rounding(kind):
    graph, forest, edges = _phase_fixture(kind)
    angles = tuple(node * math.pi / 4 for node in graph)
    measured = {
        (left, right): cmath.rect(1, angles[right] - angles[left])
        for left, right in edges
    }
    relative = graph.copy()
    for component in nx.connected_components(graph):
        root = min(component)
        for node in component:
            decoded = complex(1)
            path = nx.shortest_path(forest, root, node)
            for left, right in zip(path, path[1:]):
                decoded *= (
                    measured[(left, right)]
                    if (left, right) in measured
                    else measured[(right, left)].conjugate()
                )
            assert decoded == pytest.approx(
                cmath.rect(1, angles[node] - angles[root]), rel=REL, abs=ABS
            )
            assert decoded * cmath.rect(1, angles[root]) == pytest.approx(
                cmath.rect(1, angles[node]), rel=REL, abs=ABS
            )
            set_attr(relative.nodes[node], ALIAS_THETA, cmath.phase(decoded))
    expected_gradient = (math.pi / 4,) * 4
    expected_curvature = (-math.pi / 4, 0, 0, math.pi / 4)
    if kind == "cycle":
        expected_gradient = (math.pi / 2, math.pi / 4, math.pi / 4, math.pi / 2)
        expected_curvature = (-math.pi / 2, 0, 0, math.pi / 2)
    elif kind == "disconnected":
        expected_curvature = (-math.pi / 4, math.pi / 4, -math.pi / 4, math.pi / 4)
    for current in (graph, relative):
        assert tuple(compute_phase_gradient(current).values()) == pytest.approx(
            expected_gradient, rel=REL, abs=ABS
        )
        assert tuple(compute_phase_curvature(current).values()) == pytest.approx(
            expected_curvature, rel=REL, abs=ABS
        )


def test_known_oscillator_pressure_is_temporally_observable_without_first_order_closure():
    # One coordinate block of the already supplied radial-star oscillator.
    generator = np.array(((0.0, 1.0), (-1.0, 0.0)))
    observer = np.array(((0.0, 1.0),))
    certificate = linear_observability_certificate(
        generator, observer, scope="one supplied radial-star oscillator block"
    )
    assert certificate.rank == 2
    assert certificate.is_state_observable
    # A hidden form difference lies in ker(C), but its pressure derivative
    # does not: ker(C) is not A-invariant, so no y'=M*y law descends.
    hidden = np.array((1.0, 0.0))
    np.testing.assert_array_equal(observer @ hidden, (0.0,))
    np.testing.assert_array_equal(observer @ generator @ hidden, (-1.0,))
    np.testing.assert_array_equal(
        np.vstack((observer, observer @ generator)), ((0, 1), (-1, 0))
    )

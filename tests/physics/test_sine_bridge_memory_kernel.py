"""Independent causal-kernel controls on the admitted two-C6 nodal tangent."""

from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_matrix_product as product
from tnfr.mathematics.krylov import exact_rank
from tnfr.physics.relational_sine_bridge_memory import assess_sine_bridge_memory
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

HIDDEN_SPATIAL = (
    (Q(7, 6), Q(-1, 2), Q(0)),
    (Q(-1, 2), Q(1), Q(-1, 2)),
    (Q(0), Q(-1, 2), Q(3, 2)),
)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _matrix(rows):
    return mp.matrix([[_mp(value) for value in row] for row in rows])


@pytest.fixture(scope="module")
def witness():
    graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
    graph.add_edge(0, 6)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    source = bound_relational_sine_exchange(graph, reference_model=model)
    report = assess_sine_bridge_memory(
        source,
        left_cycle=tuple(range(6)),
        right_cycle=tuple(range(6, 12)),
        target_phase_turns=tuple(Q(i % 6, 6) for i in range(12)),
    )
    return graph, report


def _full_nodal_generator(graph):
    """Build both nodal rows directly from neighbors and exact target cosines."""
    result = [[Q(0) for _ in range(24)] for _ in range(24)]
    for i in range(12):
        for j in graph[i]:
            mobility = Q(1, graph.degree[i])
            cosine = Q(1) if {i, j} == {0, 6} else Q(1, 2)
            result[i][12 + i] -= mobility * cosine
            result[i][12 + j] += mobility * cosine
            result[12 + i][i] += mobility
            result[12 + i][j] -= mobility
    return tuple(tuple(row) for row in result)


def _trigonometric_functions():
    """Diagonalize the independent three-mode energy generator only once."""
    frequencies, modes = mp.eigsy(_matrix(HIDDEN_SPATIAL) / mp.sqrt(2))

    def matrices(time):
        cosine = (
            modes * mp.diag([mp.cos(rate * time) for rate in frequencies]) * modes.T
        )
        sine = modes * mp.diag([mp.sin(rate * time) for rate in frequencies]) * modes.T
        return cosine, sine

    def kernel(time):
        cosine, sine = matrices(time)
        c, s = cosine[0, 0], sine[0, 0]
        return mp.matrix(
            [
                [-2 * c / 9, 2 * mp.sqrt(2) * s / 9],
                [-2 * mp.sqrt(2) * s / 9, -4 * c / 9],
            ]
        )

    def source(time, hidden):
        cosine, sine = matrices(time)
        form = hidden[:3, :]
        phase = hidden[3:, :] / mp.sqrt(2)
        return mp.matrix(
            [
                mp.sqrt(2) * (sine * form + cosine * phase)[0] / 3,
                -2 * (cosine * form - sine * phase)[0] / 3,
            ]
        )

    return kernel, source


def test_analytic_memory_and_hidden_source_match_exact_partition(witness):
    _graph, report = witness
    memory = report.coordinate_memory
    with mp.workdps(65):
        kernel, forcing = _trigonometric_functions()
        hidden = mp.matrix([1, -2, 3, -1, 2, -3])
        b = _matrix(memory.hidden_to_visible)
        c = _matrix(memory.visible_to_hidden)
        d = _matrix(memory.hidden_generator)
        for time in (mp.mpf(0), mp.mpf(1) / 3, mp.mpf(2)):
            propagation = mp.expm(time * d)
            assert mp.norm(b * propagation * c - kernel(time)) < mp.mpf("1e-60")
            assert mp.norm(b * propagation * hidden - forcing(time, hidden)) < mp.mpf(
                "1e-60"
            )
        # The sine off-diagonal is part of the causal response, not an optional
        # correction that can be removed after reading the zero-lag matrix.
        assert abs(kernel(mp.mpf(1) / 3)[0, 1]) > mp.mpf("0.01")
        assert memory.kernel_at_zero == ((Q(-2, 9), 0), (0, Q(-4, 9)))


def test_finite_convolution_reconstructs_a_full_nodal_response_with_hidden_state(
    witness,
):
    graph, report = witness
    nodal = _full_nodal_generator(graph)
    initial = tuple(Q((5 * i) % 13 - 6, 7) for i in range(24))
    with mp.workdps(65):
        full = _matrix(nodal)
        z0 = mp.matrix([_mp(value) for value in initial])
        projection = _matrix(report.projection_rows)
        reduced = _matrix(report.coordinate_generator)
        coordinate0 = projection * z0
        hidden0 = mp.matrix(
            [coordinate0[i] for i in report.coordinate_memory.hidden_indices]
        )
        observation = mp.matrix(2, 8)
        observation[0, 0] = observation[1, 4] = 1
        kernel, forcing = _trigonometric_functions()
        time = mp.mpf(2) / 5
        nodal_endpoint = mp.expm(time * full) * z0
        observed = mp.matrix(
            [
                nodal_endpoint[6] - nodal_endpoint[0],
                nodal_endpoint[18] - nodal_endpoint[12],
            ]
        )
        reduced_endpoint = mp.expm(time * reduced) * coordinate0
        assert mp.norm(observed - observation * reduced_endpoint) < mp.mpf("1e-60")
        derivative = observation * reduced * reduced_endpoint
        roots, weights = mp.gauss_quadrature(24, "legendre")
        convolution = mp.matrix(2, 1)
        for root, weight in zip(roots, weights):
            lag = time * (root + 1) / 2
            past = observation * mp.expm((time - lag) * reduced) * coordinate0
            convolution += time * weight / 2 * kernel(lag) * past
        visible = _matrix(report.coordinate_memory.visible_generator)
        hidden_source = forcing(time, hidden0)
        reconstructed = visible * observed + hidden_source + convolution
        # Independent finite quadrature checks the proven identity; it is not
        # a validated trajectory enclosure or a prediction of a reserved run.
        assert mp.norm(derivative - reconstructed) < mp.mpf("1e-50")
        assert mp.norm(hidden_source) > mp.mpf("0.01")
        assert mp.norm(derivative - visible * observed - convolution) > mp.mpf("0.01")


def test_hidden_spectrum_is_finite_fully_visible_and_not_uniformly_fast(witness):
    _graph, report = witness
    square = tuple(
        tuple(value / 2 for value in row)
        for row in product(HIDDEN_SPATIAL, HIDDEN_SPATIAL)
    )
    fourth = product(square, square)
    sixth = product(fourth, square)
    # Cayley-Hamilton for S^2; no numerical roots certify the polynomial.
    for i in range(3):
        for j in range(3):
            assert 1152 * sixth[i][j] - 3232 * fourth[i][j] + 2130 * square[i][
                j
            ] == 169 * int(i == j)
    first = ((1,), (0,), (0,))
    second = product(HIDDEN_SPATIAL, first)
    third = product(HIDDEN_SPATIAL, second)
    assert (
        exact_rank(tuple((first[i][0], second[i][0], third[i][0]) for i in range(3)))
        == 3
    )
    # The endpoint is cyclic: none of these three internal modes is absent
    # from its spectral measure. A Rayleigh witness gives a hidden frequency
    # below the direct bridge exchange, without choosing a relaxation time.
    rayleigh = sum((value for row in square for value in row), Q(0)) / 3
    direct_exchange = report.coordinate_memory.visible_generator[1][0]
    assert rayleigh == Q(13, 54) < direct_exchange**2
    shifted = tuple(
        tuple(value - Q(int(i == j), 16) for j, value in enumerate(row))
        for i, row in enumerate(square)
    )
    a, b, c = shifted[0]
    d, e, f = shifted[1]
    g, h, i = shifted[2]
    leading_minors = (
        a,
        a * e - b * d,
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g),
    )
    assert leading_minors == (Q(107, 144), Q(167, 768), Q(1543, 36864))
    assert all(value > 0 for value in leading_minors)
    # Sylvester's criterion gives S^2>I/16 and hence omega_min>1/4.


def test_zero_frequency_elimination_renormalizes_exchange_without_selecting_loss(
    witness,
):
    _graph, report = witness
    with mp.workdps(65):
        memory = report.coordinate_memory
        a = _matrix(memory.visible_generator)
        b = _matrix(memory.hidden_to_visible)
        c = _matrix(memory.visible_to_hidden)
        d = _matrix(memory.hidden_generator)
        static = mp.matrix([[0, -mp.mpf(2) / 13], [mp.mpf(2) / 13, 0]])
        rate = mp.diag([mp.mpf(309) / 169, mp.mpf(449) / 169])
        assert mp.norm(a - b * d**-1 * c - static) < mp.mpf("1e-60")
        assert mp.norm(mp.eye(2) + b * d**-2 * c - rate) < mp.mpf("1e-60")
        assert mp.norm(_matrix(report.static_visible_generator) - static) < mp.mpf(
            "1e-60"
        )
        assert mp.norm(_matrix(report.low_frequency_rate_matrix) - rate) < mp.mpf(
            "1e-60"
        )
        # The explicit bound uses omega_min>1/4 and energy-coordinate coupling
        # norm 2/3. Real Laplace arguments do not test imaginary frequencies
        # or imply a time-domain approximation with discarded hidden state.
        for s in (mp.mpf("0.001"), mp.mpf("0.1")):
            effective = s * mp.eye(2) - a - b * (s * mp.eye(6) - d) ** -1 * c
            error = mp.svd(effective - (s * rate - static), compute_uv=False)[0]
            assert 0 < error <= mp.mpf(256) * s**2 / 9

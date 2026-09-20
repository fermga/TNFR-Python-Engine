"""Analytic scope controls for the production four-channel pressure.

These small regular-chart cases compare represented evaluation with an
independent exact-model formula. They neither certify all backends nor
derive a phase/capacity law, conservation requirement or future trajectory.
"""

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.extended import compute_phase_current


def _read_pressure(graph, epi, capacity, phase):
    for node, x, nu, theta in zip(graph, epi, capacity, phase, strict=True):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: float(x),
                ALIAS_VF[0]: float(nu),
                ALIAS_THETA[0]: float(theta),
                ALIAS_DNFR[0]: 0.0,
            }
        )
    graph.graph["vectorized_dnfr"] = True
    default_compute_delta_nfr(graph)
    pressure = np.array([get_attr(graph.nodes[i], ALIAS_DNFR, 0.0) for i in graph])
    weights = {
        name: Fraction(value) for name, value in graph.graph["_dnfr_weights"].items()
    }
    return pressure, weights


@pytest.mark.parametrize("factory", [nx.path_graph, nx.cycle_graph], ids=["P4", "C4"])
def test_low_degree_common_chart_is_one_support_difference(factory):
    graph = factory(4)
    graph.graph["DNFR_WEIGHTS"] = {"phase": 1, "epi": 2, "vf": 3, "topo": 4}
    turns_of_pi = (Fraction(-1, 8), Fraction(0), Fraction(1, 16), Fraction(1, 8))
    epi = (Fraction(1, 4), Fraction(1, 2), Fraction(-1, 4), Fraction(3, 4))
    capacity = (Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2))

    pressure, weights = _read_pressure(
        graph, epi, capacity, [float(value) * math.pi for value in turns_of_pi]
    )

    composite = tuple(
        weights["epi"] * epi[i]
        + weights["phase"] * turns_of_pi[i]
        + weights["vf"] * capacity[i]
        + weights["topo"] * graph.degree[i]
        for i in graph
    )
    expected = tuple(
        sum((composite[j] - composite[i] for j in graph[i]), Fraction(0))
        / graph.degree[i]
        for i in graph
    )
    np.testing.assert_allclose(pressure, list(map(float, expected)), rtol=0, atol=2e-15)
    assert sum((graph.degree[i] * expected[i] for i in graph), Fraction(0)) == 0
    assert sum(graph.degree[i] * pressure[i] for i in graph) == pytest.approx(
        0, abs=4e-15
    )


def test_regular_k4_phase_source_is_not_degree_conservative():
    graph = nx.complete_graph(4)
    beta = math.atan(math.sqrt(3) / 5)

    pressure, weights = _read_pressure(
        graph, (0.25,) * 4, (1.0,) * 4, (0.0, 0.0, 0.0, math.pi / 3)
    )

    expected = (
        float(weights["phase"]) * np.array([beta, beta, beta, -math.pi / 3]) / math.pi
    )
    np.testing.assert_allclose(pressure, expected, rtol=0, atol=2e-15)
    # Constant EPI/capacity and regular degree make all other channels zero.
    assert weights["phase"] > 0
    assert weights["topo"] == 0
    assert sum(pressure) < -0.005
    assert sum(graph.degree[i] * pressure[i] for i in graph) < -0.015


def _read_phase_pressure(graph, phases):
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0,
        "epi": 0.0,
        "vf": 0.0,
        "topo": 0.0,
    }
    pressure, _weights = _read_pressure(
        graph, (0.25,) * len(graph), (1.0,) * len(graph), phases
    )
    return pressure


def test_equal_mean_different_spread_separates_argument_pressure_and_sine_current():
    center, mean = math.pi / 4, math.pi / 8
    arg_readings, sine_readings = [], []
    for spread in (math.pi / 16, 3 * math.pi / 16):
        graph = nx.path_graph(3)
        phases = (center + mean - spread, center, center + mean + spread)

        pressure = _read_phase_pressure(graph, phases)
        # This existing current is a read-out, not a replacement pressure law.
        matched_sine = compute_phase_current(graph)[1] / math.pi

        assert all(0 <= phase < math.tau for phase in phases)
        assert all(abs(phases[i] - phases[j]) < math.pi / 2 for i, j in graph.edges)
        assert pressure[1] == pytest.approx(1 / 8, rel=0, abs=2e-15)
        assert matched_sine == pytest.approx(
            math.sin(mean) * math.cos(spread) / math.pi, rel=0, abs=2e-15
        )
        arg_readings.append(pressure[1])
        sine_readings.append(matched_sine)

    assert arg_readings[0] == pytest.approx(arg_readings[1], rel=0, abs=2e-15)
    assert sine_readings[0] > sine_readings[1]


def test_one_sided_antipodal_pressure_reads_opposite_regular_branches():
    distance = math.pi / 4096
    readings = []
    for side in (-1, 1):
        graph = nx.path_graph(2)
        pressure = _read_phase_pressure(graph, (0.0, math.pi + side * distance))
        expected = -side * (1 - 1 / 4096)

        np.testing.assert_allclose(pressure, (expected, -expected), rtol=0, atol=2e-15)
        readings.append(pressure[0])

    # A finite check of the two analytic branches, not a sampled proof of
    # continuity failure or U3 admission. Neither input is exactly antipodal.
    assert readings[0] - readings[1] == pytest.approx(
        2 * (1 - 1 / 4096), rel=0, abs=4e-15
    )


def test_zero_neighbor_resultant_has_distinct_one_sided_pressure_limits():
    distance = math.pi / 4096
    readings = []
    for side in (-1, 1):
        graph = nx.path_graph(3)
        pressure = _read_phase_pressure(graph, (0.0, 0.0, math.pi + side * distance))
        # Arg(1+exp(i*(pi-distance)))=(pi-distance)/2; the other
        # approach has the opposite direction. The center's displacement
        # approaches +/-pi/2, rather than its own antipodal branch boundary.
        expected = -side * (1 / 2 - 1 / 8192)
        assert pressure[1] == pytest.approx(expected, rel=0, abs=2e-13)
        readings.append(pressure[1])

    # The larger tolerance retains cancellation conditioning. These are
    # nonzero represented resultants, not ideal-zero or U3 certificates.
    assert readings[0] - readings[1] == pytest.approx(1 - 1 / 4096, rel=0, abs=4e-13)


def test_general_near_coherent_cubic_jet_distinguishes_argument_and_sine():
    s = pytest.importorskip("sympy")
    epsilon, m1, m2, m3 = s.symbols("epsilon m1 m2 m3", real=True)
    # m_j are raw moments of fixed neighbor offsets a, with gaps epsilon*a.
    # Re(mean(exp(i*gap))) is positive near zero, so Arg=atan(Im/Re).
    real = 1 - m2 * epsilon**2 / 2
    imaginary = m1 * epsilon - m3 * epsilon**3 / 6
    argument_jet = s.series(s.atan(imaginary / real), epsilon, 0, 4).removeO()
    variance = m2 - m1**2
    centered_third = m3 - 3 * m1 * m2 + 2 * m1**3

    assert (
        s.expand(argument_jet - (m1 * epsilon - centered_third * epsilon**3 / 6)) == 0
    )
    assert (
        s.expand(
            argument_jet - imaginary - (m1**3 + 3 * m1 * variance) * epsilon**3 / 6
        )
        == 0
    )
    # Divide both jets by pi for matching linear pressure gain. Oddness
    # removes the fourth-order term; the documented remainder is O(epsilon^5).

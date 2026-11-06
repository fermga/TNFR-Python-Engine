"""Fixed analytical witnesses for the information retained by static xi_C.

The supplied oscillator below is a complete, nondimensional comparison law,
not the default engine pressure law. Only its initial derivative is evaluated;
no trajectory or empirical correlation model is inferred from these fixtures.
"""

import math
from copy import deepcopy
from fractions import Fraction
from itertools import combinations

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
    observe_coherence_profile,
)
from tnfr.physics.extended import compute_phase_current
from tnfr.physics.observability import tetrad_observation_vector
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils.cache import reset_global_cache

Q = Fraction
RADII = (0, 1, 1, 2, 2, 3, 3)
COHERENCE = tuple(Q(1, 2 ** (radius + 1)) for radius in RADII)


@pytest.fixture(autouse=True)
def _clear_field_cache():
    reset_global_cache()
    yield
    reset_global_cache()


def _radial_star(form=0):
    graph = nx.star_graph(6)
    for node, radius in enumerate(RADII):
        attributes = graph.nodes[node]
        set_attr(attributes, ALIAS_DNFR, float(1 / COHERENCE[node] - 1))
        set_attr(attributes, ALIAS_EPI, float(form))
        set_attr(attributes, ALIAS_THETA, 0.0)
        set_attr(attributes, ALIAS_VF, 1.0)
        if node:
            graph.edges[0, node].update(length=float(radius), weight=1.0)
    return graph


def _pairs_at(distance):
    """Use the star's analytic distance, independently of graph path code."""
    return tuple(
        (left, right)
        for left, right in combinations(range(7), 2)
        if RADII[left] + RADII[right] == distance
    )


def _mean(values):
    values = tuple(values)
    return sum(values, Q(0)) / len(values)


def _exact_moments(distance):
    pairs = _pairs_at(distance)
    mean = _mean(COHERENCE)
    endpoint_mean = _mean(
        value for left, right in pairs for value in (COHERENCE[left], COHERENCE[right])
    )
    product = _mean(COHERENCE[left] * COHERENCE[right] for left, right in pairs)
    second = _mean(
        value**2
        for left, right in pairs
        for value in (COHERENCE[left], COHERENCE[right])
    )
    centered = _mean(
        (COHERENCE[left] - mean) * (COHERENCE[right] - mean) for left, right in pairs
    )
    semivariogram = _mean(
        (COHERENCE[left] - COHERENCE[right]) ** 2 / 2 for left, right in pairs
    )
    return {
        "product_mean": product,
        "endpoint_mean": endpoint_mean,
        "endpoint_second_moment": second,
        "centered_product": centered,
        "endpoint_covariance": product - endpoint_mean**2,
        "semivariogram": semivariogram,
    }


def test_radial_profile_matches_exact_endpoint_moments_and_signed_centering():
    profile = observe_coherence_profile(_radial_star())
    exact = tuple(_exact_moments(distance) for distance in range(1, 7))

    assert tuple(row.pair_count for row in profile.bins) == (2, 3, 6, 5, 4, 1)
    assert tuple(row.distance for row in profile.bins) == (1, 2, 3, 4, 5, 6)
    assert profile.global_mean == pytest.approx(float(Q(11, 56)), abs=1e-16)
    assert profile.global_variance == pytest.approx(float(Q(129, 6272)), abs=1e-16)
    # Independent rational targets make the sign-changing centered profile
    # explicit even though every raw product lies on one exact exponential.
    assert tuple(row["centered_product"] for row in exact) == (
        Q(51, 3136),
        Q(-127, 9408),
        Q(-101, 6272),
        Q(-37, 7840),
        Q(15, 1568),
        Q(225, 12544),
    )
    for row, oracle in zip(profile.bins, exact, strict=True):
        assert oracle["product_mean"] == Q(1, 2 ** (int(row.distance) + 2))
        for name, value in oracle.items():
            assert getattr(row, name) == pytest.approx(float(value), abs=1e-16)
        assert row.source_mean == row.target_mean == row.endpoint_mean
        assert (
            row.source_second_moment
            == row.target_second_moment
            == row.endpoint_second_moment
        )
    assert profile.fit.selected_distances == (1, 2, 3, 4, 5)
    assert profile.fit.value == pytest.approx(1 / math.log(2), rel=2e-14)
    assert profile.fit.slope == pytest.approx(-math.log(2), rel=2e-14)
    assert profile.fit.intercept == pytest.approx(-2 * math.log(2), rel=2e-14)
    assert profile.fit.r_squared == pytest.approx(1.0, abs=1e-14)
    assert not profile.bins[-1].fit_included


def test_semivariogram_uses_bin_endpoint_moments_not_global_variance():
    profile = observe_coherence_profile(_radial_star())
    global_mean = _mean(COHERENCE)
    global_variance = _mean((value - global_mean) ** 2 for value in COHERENCE)
    first = _exact_moments(1)
    assert first["semivariogram"] == Q(1, 32)
    assert global_variance - first["centered_product"] == Q(27, 6272)
    assert global_variance - first["centered_product"] != first["semivariogram"]

    for row in profile.bins:
        oracle = _exact_moments(int(row.distance))
        assert oracle["semivariogram"] == (
            oracle["endpoint_second_moment"] - oracle["product_mean"]
        )
        assert oracle["centered_product"] == (
            oracle["product_mean"]
            - 2 * global_mean * oracle["endpoint_mean"]
            + global_mean**2
        )
        assert row.semivariogram == pytest.approx(
            row.endpoint_second_moment - row.product_mean, abs=1e-16
        )
    # A long-distance bin can have a positive globally centered product while
    # its own endpoint covariance and semivariogram are both exactly zero.
    assert profile.bins[-1].centered_product > 0
    assert profile.bins[-1].endpoint_covariance == 0
    assert profile.bins[-1].semivariogram == 0


def test_pressure_sign_is_lost_by_entire_coherence_profile_but_not_potential():
    positive = _radial_star()
    reflected = deepcopy(positive)
    set_attr(reflected.nodes[1], ALIAS_DNFR, -3.0)

    assert observe_coherence_profile(positive) == observe_coherence_profile(reflected)
    assert estimate_coherence_length_with_provenance(positive) == (
        estimate_coherence_length_with_provenance(reflected)
    )
    pressures = tuple(1 / value - 1 for value in COHERENCE)
    signed = tuple(
        -value if node == 1 else value for node, value in enumerate(pressures)
    )
    initial_product = _mean(
        pressures[left] * pressures[right] for left, right in _pairs_at(1)
    )
    flipped_product = _mean(
        signed[left] * signed[right] for left, right in _pairs_at(1)
    )
    assert initial_product == 3
    assert flipped_product == 0
    before = compute_structural_potential(positive)
    after = compute_structural_potential(reflected)
    # This is extra information relative to xi alone; the full potential
    # already resolves this particular pressure-sign ambiguity.
    assert after[0] - before[0] == pytest.approx(-6.0, abs=1e-14)


def _marked_twist(sign):
    graph = nx.cycle_graph(8)
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, node / 8)
        set_attr(graph.nodes[node], ALIAS_EPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_VF, 1.0)
        set_attr(graph.nodes[node], ALIAS_THETA, sign * math.pi * node / 4)
    return graph


def test_marked_twists_share_full_tetrad_and_canceled_currents_but_not_winding():
    positive, negative = _marked_twist(1), _marked_twist(-1)
    left = tetrad_observation_vector(positive, representation="full")
    right = tetrad_observation_vector(negative, representation="full")
    np.testing.assert_allclose(left, right, rtol=0, atol=2e-15)
    first = estimate_coherence_length_with_provenance(positive)
    second = estimate_coherence_length_with_provenance(negative)
    assert first == second
    assert first.method in {"autocorrelation_fit", "spectral_gap"}
    assert math.isfinite(first.value)

    for graph, sign in ((positive, 1), (negative, -1)):
        assert tuple(compute_phase_gradient(graph).values()) == pytest.approx(
            (math.pi / 4,) * 8, abs=2e-15
        )
        assert tuple(compute_phase_curvature(graph).values()) == pytest.approx(
            (0.0,) * 8, abs=2e-15
        )
        assert tuple(compute_phase_current(graph).values()) == pytest.approx(
            (0.0,) * 8, abs=2e-15
        )
        certificate = certify_phase_winding(graph, range(8))
        assert certificate.is_defined and certificate.u3_admissible
        assert certificate.winding == sign
        assert certificate.minimum_branch_margin == pytest.approx(3 * math.pi / 4)
        assert certificate.minimum_u3_margin == pytest.approx(math.pi / 4)
        # Orientation reversal is a convention change, not a new preparation.
        assert certify_phase_winding(graph, reversed(range(8))).winding == -sign
    # Distinct pressure labels forbid identifying the two preparations by a
    # reflection that leaves the declared pressure/port association unchanged.
    pressure = tuple(node / 8 for node in range(8))
    assert len(set(pressure)) == 8
    assert tuple(pressure[(-node) % 8] for node in range(8)) != pressure


def test_same_supplied_oscillator_law_has_equal_tetrads_but_different_xi_tangents():
    zero_form, unit_form = _radial_star(0), _radial_star(1)
    np.testing.assert_array_equal(
        tetrad_observation_vector(zero_form, representation="full"),
        tetrad_observation_vector(unit_form, representation="full"),
    )
    profile = observe_coherence_profile(unit_form)
    assert observe_coherence_profile(zero_form) == profile

    # Complete supplied law: x_dot=p, p_dot=-x, nu=1; phase/support fixed.
    # Initial p>0 gives c_dot=x*c^2. Both preparations have the same x_dot,
    # but different independently specified p_dot, without a time integration.
    pressure = tuple(1 / value - 1 for value in COHERENCE)
    zero_velocity = tuple((value, Q(0)) for value in pressure)
    unit_velocity = tuple((value, Q(-1)) for value in pressure)
    assert tuple(row[0] for row in zero_velocity) == tuple(
        row[0] for row in unit_velocity
    )

    slopes = []
    for form in (0, 1):
        coherence_rate = tuple(form * value**2 for value in COHERENCE)
        log_product_rates = []
        for distance in range(1, 6):
            pairs = _pairs_at(distance)
            product = _mean(COHERENCE[left] * COHERENCE[right] for left, right in pairs)
            rate = _mean(
                coherence_rate[left] * COHERENCE[right]
                + COHERENCE[left] * coherence_rate[right]
                for left, right in pairs
            )
            log_product_rates.append(rate / product)
        # Unweighted least-squares slope on distances 1..5 has mean 3 and
        # centered square sum 10. All admissions persist in a local interval.
        slopes.append(
            sum(
                (distance - 3) * rate
                for distance, rate in enumerate(log_product_rates, 1)
            )
            / 10
        )
    assert slopes == [Q(0), Q(-169, 1200)]
    # xi=-1/slope and slope=-ln2 at this preparation. The rich profile's
    # endpoint means recover this analytic jet without measuring a response.
    profile_slope_rate = (
        sum(
            (row.distance - 3) * 2 * row.endpoint_mean
            for row in profile.bins
            if row.fit_included
        )
        / 10
    )
    assert profile_slope_rate == pytest.approx(float(Q(-169, 1200)), abs=1e-16)
    assert profile_slope_rate / profile.fit.slope**2 == pytest.approx(
        -169 / (1200 * math.log(2) ** 2), rel=2e-14
    )

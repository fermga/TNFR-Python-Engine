"""Static quadratic memory identities, family admission and native readouts."""

import math
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics._cycle_algebra import laplacian_matrix
from tnfr.physics.relational_capture import certify_relational_cycle_capture
from tnfr.physics.relational_cycle_memory import bound_relational_cycle_memory
from tnfr.physics.relational_observations import observe_relational_pattern

FORM = (1, -1, 0, 0, 0)
PHASE = (0, 1, -1, 0, 0)
MODEL = RelationalExchangeModel(storage_scale=1)


def _bounds(**kwargs):
    arguments = dict(
        model=MODEL,
        form_direction=FORM,
        phase_direction=PHASE,
        capacity=Q(1),
        amplitude_radius=Q(1, 64),
    )
    arguments.update(kwargs)
    return bound_relational_cycle_memory(**arguments)


def test_cyclic_pairing_integrates_the_quadratic_mean_rate_without_a_solver():
    s = pytest.importorskip("sympy")
    # B is reused from the shared cycle owner; D is independently oriented.
    B = 2 * s.Matrix(laplacian_matrix(5))
    D = s.zeros(5)
    for i in range(5):
        D[i, (i + 1) % 5] = 1
        D[i, (i - 1) % 5] = -1
    assert B * D == D * B and (B * D).T == -B * D
    a, b, d, coefficient = s.symbols("a b d coefficient", positive=True)
    zero = s.zeros(5)
    generator = (-a * B).row_join(-b * B).col_join((d * B).row_join(zero))
    pairing = zero.row_join(D / 2).col_join((-D / 2).row_join(zero))
    rate = zero.row_join(coefficient * B * D / 2).col_join(
        (-coefficient * B * D / 2).row_join(zero)
    )
    integral = coefficient * pairing / a
    assert generator.T * integral + integral * generator == -rate
    vector = s.Matrix(FORM + PHASE)
    assert (vector.T * pairing * vector)[0] == 2 == _bounds().cyclic_pairing
    assert (vector.T * rate * vector)[0] == 5 * coefficient
    # sinc has zero first derivative, so only the oriented cosine factor
    # contributes to the quadratic common-phase drift at the twist.
    epsilon, kappa, mean, oriented = s.symbols("epsilon kappa mean oriented", real=True)
    metric = (
        2
        * s.pi
        * s.cos(kappa + epsilon * oriented)
        * s.sin(epsilon * mean)
        / (epsilon * mean)
    )
    assert (
        s.simplify(
            s.limit(s.diff(1 / metric, epsilon), epsilon, 0)
            - oriented * s.sin(kappa) / (2 * s.pi * s.cos(kappa) ** 2)
        )
        == 0
    )


def test_coefficients_enclose_independent_exact_twist_and_contact_expressions():
    s = pytest.importorskip("sympy")
    report = _bounds()
    kappa = 2 * s.pi / 5
    # Here P=2 and e=w=1/2, beta=nu=1. These are independent expressions,
    # not the midpoint of the reported interval or a fitted response.
    shift = s.sin(kappa) / (5 * s.pi * s.cos(kappa) ** 2)
    contact = -shift / (2 * s.pi * (1 + 2 * s.cos(kappa)))
    for expression, interval in (
        (shift, report.quadratic_phase_shift_coefficient_bounds),
        (contact, report.quadratic_left_port_form_rate_coefficient_bounds),
    ):
        reference = Q(str(s.N(expression, 80)))
        assert interval[0] < reference < interval[1]
    # For the real cos/sin Fourier pair of harmonic one, P=5*sin(kappa).
    assert s.simplify(s.tan(kappa) ** 2 - 5 - 2 * s.sqrt(5)) == 0
    assert report.basin_admitted and report.unavailable_reasons == ()
    assert report.quadratic_phase_shift_coefficient_bounds[0] > 0
    assert report.quadratic_left_port_form_rate_coefficient_bounds[1] < 0
    assert report.remainder_order == 3 and report.remainder_bound is None


def test_orientation_capacity_and_exact_zero_pairing_retain_distinct_meanings():
    baseline = _bounds()
    reflected = _bounds(target_sector=-1)
    reversed_form = _bounds(form_direction=tuple(-value for value in FORM))
    for name in (
        "quadratic_phase_shift_coefficient_bounds",
        "quadratic_left_port_form_rate_coefficient_bounds",
    ):
        lower, upper = getattr(baseline, name)
        assert (
            getattr(reflected, name) == getattr(reversed_form, name) == (-upper, -lower)
        )
    assert reflected.storage_upper_bound == baseline.storage_upper_bound
    accelerated = _bounds(capacity=Q(3, 2))
    assert accelerated.quadratic_phase_shift_coefficient_bounds == (
        baseline.quadratic_phase_shift_coefficient_bounds
    )
    assert accelerated.quadratic_left_port_form_rate_coefficient_bounds == tuple(
        Q(3, 2) * value
        for value in baseline.quadratic_left_port_form_rate_coefficient_bounds
    )
    same_direction = _bounds(phase_direction=FORM)
    assert same_direction.cyclic_pairing == 0
    assert same_direction.quadratic_phase_shift_coefficient_bounds == (0, 0)
    assert same_direction.quadratic_left_port_form_rate_coefficient_bounds == (0, 0)
    # A zero quadratic coefficient does not bound or erase the higher terms.
    assert same_direction.remainder_bound is None


def test_sufficient_family_basin_is_separate_from_asymptotic_coefficient():
    small = _bounds()
    large = _bounds(amplitude_radius=Q(1, 4))
    assert small.basin_admitted and not large.basin_admitted
    assert small.quadratic_phase_shift_coefficient_bounds == (
        large.quadratic_phase_shift_coefficient_bounds
    )
    assert large.unavailable_reasons == (
        "strict_family_acute_margin_not_certified",
        "strict_family_cycle_energy_barrier_not_certified",
    )
    # Both direction energies are 3, from their three changed cycle edges.
    assert (
        small.storage_upper_bound == small.reference_storage_bounds[1] + 6 / Q(64) ** 2
    )
    assert small.acute_margin_lower_bound == small.pi_bounds[0] / 10 - Q(1, 32)
    assert small.storage_margin_lower_bound == (
        small.capture_barrier_bounds[0] - small.storage_upper_bound
    )
    only_form = _bounds(
        form_direction=tuple(100 * value for value in FORM), phase_direction=(0,) * 5
    )
    assert only_form.acute_margin_lower_bound > 0
    assert only_form.unavailable_reasons == (
        "strict_family_cycle_energy_barrier_not_certified",
    )


def test_tiny_model_denominators_and_exact_capacity_are_not_lost_in_intervals():
    tiny = math.ulp(0.0)
    model = RelationalExchangeModel(storage_scale=tiny, epi_weight=tiny, phase_weight=1)
    report = _bounds(
        model=model, capacity=Q(1, 2**1500), amplitude_radius=Q(1, 2**1500)
    )
    baseline = _bounds()
    factor = Q(model.phase_weight) / (Q(model.epi_weight) * Q(model.storage_scale))
    assert report.quadratic_phase_shift_coefficient_bounds == tuple(
        factor * value for value in baseline.quadratic_phase_shift_coefficient_bounds
    )
    assert report.capacity == Q(1, 2**1500)
    assert report.quadratic_left_port_form_rate_coefficient_bounds[1] < 0
    assert report.basin_admitted


@pytest.mark.parametrize(
    "kwargs, error, match",
    (
        ({"form_direction": (1, 0, 0, 0, 0)}, ValueError, "zero sum"),
        ({"phase_direction": (0, 0, 0, 0)}, ValueError, "five"),
        ({"phase_direction": (False, 0, 0, 0, 0)}, TypeError, "boolean"),
        ({"phase_direction": (math.nan, 0, 0, 0, 0)}, ValueError, "finite"),
        ({"capacity": 0}, ValueError, "positive"),
        ({"amplitude_radius": -1}, ValueError, "positive"),
        ({"target_sector": True}, TypeError, "nonboolean integer"),
        ({"target_sector": 0}, ValueError, "-1 or 1"),
        ({"model": object()}, TypeError, "RelationalExchangeModel"),
    ),
)
def test_invalid_inputs_cannot_be_projected_into_an_admitted_family(
    kwargs, error, match
):
    with pytest.raises(error, match=match):
        _bounds(**kwargs)


def test_lossless_boundary_has_no_dissipative_asymptotic_coefficient():
    with pytest.raises(ValueError, match="positive epi_weight"):
        _bounds(model=RelationalExchangeModel(storage_scale=1, epi_weight=0))


def test_native_full_cycle_observer_retains_the_covariance_mean_rate():
    graph = nx.cycle_graph(5)
    reference = {i: (i - 2) * 2 * math.pi / 5 for i in graph}
    epsilon = Q(1, 64)
    for i in graph:
        graph.nodes[i].update(
            EPI=float(epsilon * FORM[i]),
            theta=reference[i] + float(epsilon * PHASE[i]),
            nu_f=1,
        )
    graph.graph["GAMMA"] = {"type": "none"}
    observed = observe_relational_pattern(
        graph, model=MODEL, reference_phase=reference, regions=(tuple(graph),)
    )
    certificate = certify_relational_cycle_capture(
        graph, model=MODEL, cycle=tuple(graph)
    )
    response = observed.regions[0].phase_response
    assert certificate.admitted
    assert response.mean_mobility_boundary_rate == 0
    assert response.covariance_rate > 0
    assert response.mean_rate == sum(map(Q, observed.field.phase_rate)) / 5
    assert (
        response.mean_rate
        == (response.covariance_rate + response.rounding_residual) / 5
    )
    assert response.identity_residual == 0
    assert sum(Q(value) for value in observed.field.phase_rate) != 0
    # This current represented readout neither computes the limiting phase
    # shift nor supplies the static coefficient's missing cubic error bound.
    assert _bounds().remainder_bound is None

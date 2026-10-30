"""Independent local word/flow algebra, without events or trajectories.

The phase preparation is the exact mathematical carrier. High-precision
phasors below check its analytic directional derivative; they do not replace
the carrier with a floating-point phase snapshot or integrate a response.
"""

from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics import _organized_source_timing as owner

# Declare the support independently of every geometry/report consumer.
EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(5)
    for slot in range(9)
) + tuple(
    (9 * a + slot, 9 * b + slot)
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
    for slot in range(9)
)
NEIGHBORS = tuple(
    tuple(b if a == node else a for a, b in EDGES if node in (a, b))
    for node in range(45)
)
ZERO = (Q(0),) * 45
MAIN = (Q(1, 64), -Q(1, 64)) + ZERO[2:]
ASYMMETRIC = ZERO[:2] + (Q(1, 1000), -Q(1, 1000)) + ZERO[4:]


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("A local timing proof may not execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def factors():
    registered = canonical_glyph_factor_defaults()
    return {
        "reception_mix": Q(registered["EN_mix"]),
        "phase_coefficient": Q(DEFAULT_PHASE_LOCKING_COEFFICIENT),
        "silence_capacity_factor": Q(registered["SHA_vf_factor"]),
    }


@pytest.fixture(scope="module")
def report(factors):
    return owner._assess_organized_source_carrier_timing(MAIN, **factors)


def _gradient(values):
    result = [Q(0)] * 45
    for a, b in EDGES:
        result[a] += values[a] - values[b]
        result[b] += values[b] - values[a]
    return tuple(result)


def _energy(values):
    return sum(((values[b] - values[a]) ** 2 / 2 for a, b in EDGES), Q(0))


def _reception(values, beta):
    result = list(values)
    for target in (0, 1):
        result[target] = (1 - beta) * values[target] + beta * sum(
            (values[j] for j in NEIGHBORS[target]), Q(0)
        ) / len(NEIGHBORS[target])
    return tuple(result)


def _mp(context, value):
    return context.mpf(value.numerator) / int(value.denominator)


def _contains(context, bounds, value):
    assert _mp(context, bounds.lo) <= value <= _mp(context, bounds.hi)


def test_actual_support_and_independent_complete_form_derivative(report, factors):
    geometry = report.geometry
    assert len(EDGES) == len(geometry.weighted_edges) == 90
    assert {frozenset(edge) for edge in EDGES} == {
        frozenset((a, b)) for a, b, _ in geometry.weighted_edges
    }
    assert all(weight == 1 for _, _, weight in geometry.weighted_edges)
    assert set(NEIGHBORS[0]) == {1, 8, 9, 18, 36}
    assert set(NEIGHBORS[1]) == {0, 2, 10, 19, 37}
    assert geometry.clock_factor == Q(1, 4)
    assert report.form == MAIN
    q = _gradient(MAIN)
    successor = _reception(MAIN, factors["reception_mix"])
    successor_q = _gradient(successor)
    capacities = (factors["silence_capacity_factor"],) * 2 + (Q(1),) * 43
    assert report.form_gradient == q
    assert report.successor_form == successor
    assert report.successor_form_gradient == successor_q
    assert report.successor_capacities == capacities
    assert report.energy_excess == _energy(MAIN) == Q(3, 2048)
    # Apply the independently stated map to the pre-flow tangent, then
    # subtract the actual changed-capacity successor tangent.
    transported = _reception(tuple(-value / 4 for value in q), factors["reception_mix"])
    expected = tuple(
        value + nu * gradient / 4
        for value, nu, gradient in zip(transported, capacities, successor_q)
    )
    assert report.form_commutator == expected
    assert (
        report.form_commutator[9] - report.form_commutator[10]
        == (3 * factors["reception_mix"] / (5 * 64))
        > 0
    )
    # The theorem fixes the original child-zero port contrast, whereas the
    # preceding matching-child contrast is an independent observable control.
    contrast = report.form_commutator[8] - report.form_commutator[2]
    assert contrast == 3 * factors["reception_mix"] / (5 * 64)
    assert contrast == Q(815555754282159, 360287970189639680) > 0
    assert report.phase_commutator_base[8] - report.phase_commutator_base[2] == (
        -contrast
    )
    for index in (8, 2):
        assert report.phase_commutator_matching[index] == 0
        assert report.phase_commutator_cycle[index] == 0
    assert report.initial_form_mean == report.successor_form_mean == 0


def test_open_joint_box_has_independent_edge_energy_and_mean_margins(report):
    h, radius = Q(1, 64), Q(1, 100000)
    absolute_form_gaps = sum((abs(MAIN[b] - MAIN[a]) for a, b in EDGES), Q(0))
    assert absolute_form_gaps == 10 * h
    # Each endpoint moves by at most radius, so an edge difference changes
    # by at most 2*radius. Expanding its quadratic form cost gives this
    # linear term and the per-edge quadratic remainder.
    form_bound = _energy(MAIN) + 2 * radius * absolute_form_gaps
    form_bound += len(EDGES) * (2 * radius) ** 2 / 2
    # On the exact carrier each node has one forward and one backward cycle
    # neighbor; matching neighbors have zero carrier gap. Thus its sine
    # gradient cancels. The potential Hessian per edge is bounded above by
    # one, giving the same quadratic remainder with no first-order term.
    for node, neighbors in enumerate(NEIGHBORS):
        cycle_slots = sorted(
            (neighbor - node) % 9
            for neighbor in neighbors
            if neighbor // 9 == node // 9
        )
        assert cycle_slots == [1, 8]
        assert all(
            neighbor % 9 == node % 9
            for neighbor in neighbors
            if neighbor // 9 != node // 9
        )
    phase_bound = len(EDGES) * (2 * radius) ** 2 / 2
    assert form_bound + phase_bound == 6 * h**2 + 20 * h * radius + 360 * radius**2
    assert form_bound + phase_bound < Q(1, 100)
    assert abs(report.initial_form_mean) + radius < Q(1, 30)


@pytest.mark.parametrize("form", (MAIN, ASYMMETRIC))
@pytest.mark.parametrize("coefficient", (Q(0), Q(0.3), Q(1)))
def test_complete_phase_derivative_matches_independent_phasors(
    factors, form, coefficient
):
    result = owner._assess_organized_source_carrier_timing(
        form, **{**factors, "phase_coefficient": coefficient}
    )
    mp = mpmath.mp.clone()
    mp.dps = 100
    gamma = 1 / (1023 * mp.pi)
    carrier = tuple(2 * mp.pi * (node % 9 - 4) / 9 for node in range(45))
    cosine = mp.cos(2 * mp.pi / 9)
    q = _gradient(form)
    successor_q = _gradient(_reception(form, factors["reception_mix"]))
    capacity = (factors["silence_capacity_factor"],) * 2 + (Q(1),) * 43
    before = tuple(gamma * _mp(mp, value) / 4 for value in q)
    transported = list(before)
    for target in (0, 1):
        phasors = tuple(
            mp.exp(mp.j * (carrier[j] - carrier[target])) for j in NEIGHBORS[target]
        )
        resultant = mp.fsum(phasors)
        assert abs(resultant.imag) < mp.mpf("1e-95")
        assert resultant.real > 4
        derivative = (
            mp.fsum(
                phasor * (before[j] - before[target])
                for phasor, j in zip(phasors, NEIGHBORS[target])
            )
            / resultant
        )
        transported[target] += _mp(mp, coefficient) * derivative.real
    expected = tuple(
        value - gamma * _mp(mp, nu * gradient) / 4
        for value, nu, gradient in zip(transported, capacity, successor_q)
    )
    _contains(mp, result.gamma_bounds, gamma)
    _contains(mp, result.carrier_cosine_bounds, cosine)
    for index, value in enumerate(expected):
        _contains(mp, result.phase_commutator_bounds[index], value)
        symbolic = gamma * (
            _mp(mp, result.phase_commutator_base[index])
            + (
                _mp(mp, result.phase_commutator_matching[index])
                + _mp(mp, result.phase_commutator_cycle[index]) * cosine
            )
            / (3 + 2 * cosine)
        )
        assert abs(symbolic - value) < mp.mpf("1e-90")
    if coefficient:
        # IL fixes the carrier itself, but its derivative on a moving form
        # state is nontrivial and must not be silently replaced by identity.
        base = tuple(
            gamma * _mp(mp, before_q - nu * after_q) / 4
            for before_q, nu, after_q in zip(q, capacity, successor_q)
        )
        assert any(abs(expected[i] - base[i]) > mp.mpf("1e-12") for i in (0, 1))
    else:
        assert result.phase_commutator_matching == ZERO
        assert result.phase_commutator_cycle == ZERO


def test_asymmetric_input_retains_changed_weighted_mean(factors):
    result = owner._assess_organized_source_carrier_timing(ASYMMETRIC, **factors)
    beta, kappa = factors["reception_mix"], factors["silence_capacity_factor"]
    assert result.energy_excess == _energy(ASYMMETRIC) == Q(3, 500000)
    assert result.initial_form_mean == 0
    weighted_mean = sum(
        (x / nu for x, nu in zip(result.successor_form, result.successor_capacities)),
        Q(0),
    ) / sum((1 / nu for nu in result.successor_capacities), Q(0))
    assert (
        result.successor_form_mean == weighted_mean == beta / (5000 * (43 * kappa + 2))
    )
    assert weighted_mean != sum(result.successor_form, Q(0)) / 45
    assert result.form_commutator[10] - result.form_commutator[11] == -beta / 20000


def test_common_form_shift_preserves_full_commutator_but_moves_both_means(
    report, factors
):
    shift = Q(1, 100)
    shifted = owner._assess_organized_source_carrier_timing(
        tuple(value + shift for value in MAIN), **factors
    )
    for field in (
        "energy_excess",
        "form_gradient",
        "successor_form_gradient",
        "form_commutator",
        "phase_commutator_base",
        "phase_commutator_matching",
        "phase_commutator_cycle",
        "phase_commutator_bounds",
    ):
        assert getattr(shifted, field) == getattr(report, field)
    assert shifted.successor_form == tuple(
        value + shift for value in report.successor_form
    )
    assert shifted.initial_form_mean == report.initial_form_mean + shift
    assert shifted.successor_form_mean == report.successor_form_mean + shift


@pytest.mark.parametrize("level", (Q(0), Q(1, 100), -Q(1, 100)))
def test_constant_form_is_a_zero_local_control(factors, level):
    form = (level,) * 45
    result = owner._assess_organized_source_carrier_timing(form, **factors)
    assert result.form == result.successor_form == form
    assert result.energy_excess == 0
    assert result.initial_form_mean == result.successor_form_mean == level
    for field in (
        "form_gradient",
        "successor_form_gradient",
        "form_commutator",
        "phase_commutator_base",
        "phase_commutator_matching",
        "phase_commutator_cycle",
    ):
        assert getattr(result, field) == ZERO
    assert all(bound.lo == bound.hi == 0 for bound in result.phase_commutator_bounds)


def test_fixed_width_fraction_components_are_normalized_before_products(
    report, factors
):
    form = tuple(Q(np.int64(x.numerator), np.int64(x.denominator)) for x in MAIN)
    converted = {
        name: Q(np.int64(value.numerator), np.int64(value.denominator))
        for name, value in factors.items()
    }
    result = owner._assess_organized_source_carrier_timing(form, **converted)
    assert result == report
    assert all(
        type(value.numerator) is type(value.denominator) is int for value in result.form
    )


@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), 1j, "0"))
def test_invalid_form_values_reject_before_geometry(monkeypatch, factors, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid form reached geometric construction")

    monkeypatch.setattr(owner, "_derive_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_carrier_timing((invalid,) + ZERO[1:], **factors)


@pytest.mark.parametrize("form", (ZERO[:44], ZERO + (Q(0),), "0" * 45))
def test_wrong_form_container_or_length_rejects(factors, form):
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_carrier_timing(form, **factors)


@pytest.mark.parametrize("level", (Q(1, 30), -Q(1, 30), Q(1), -Q(1)))
def test_strict_mean_domain_is_independent_of_zero_energy(factors, level):
    assert _energy((level,) * 45) == 0
    with pytest.raises(ValueError):
        owner._assess_organized_source_carrier_timing((level,) * 45, **factors)


def test_exact_energy_boundary_rejects_and_an_interior_state_is_admitted(factors):
    boundary = tuple(Q(1, 20) if i in (9, 11) else Q(0) for i in range(45))
    assert _energy(boundary) == Q(1, 100)
    assert abs(sum(boundary, Q(0)) / 45) < Q(1, 30)
    with pytest.raises(ValueError):
        owner._assess_organized_source_carrier_timing(boundary, **factors)
    interior = tuple(value * Q(999, 1000) for value in boundary)
    result = owner._assess_organized_source_carrier_timing(interior, **factors)
    assert result.energy_excess == _energy(interior) < Q(1, 100)


@pytest.mark.parametrize(
    "field,invalid",
    (
        ("reception_mix", 0),
        ("reception_mix", Q(5, 4)),
        ("reception_mix", True),
        ("reception_mix", float("nan")),
        ("reception_mix", Q(1, 3)),
        ("phase_coefficient", -1),
        ("phase_coefficient", Q(5, 4)),
        ("phase_coefficient", True),
        ("phase_coefficient", float("inf")),
        ("phase_coefficient", Q(1, 3)),
        ("silence_capacity_factor", 0),
        ("silence_capacity_factor", 1),
        ("silence_capacity_factor", -Q(1, 2)),
        ("silence_capacity_factor", True),
        ("silence_capacity_factor", float("nan")),
        ("silence_capacity_factor", Q(1, 3)),
    ),
)
def test_invalid_or_unrepresented_factors_reject_before_geometry(
    monkeypatch, factors, field, invalid
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid factor reached geometric construction")

    monkeypatch.setattr(owner, "_derive_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_carrier_timing(
            MAIN, **{**factors, field: invalid}
        )

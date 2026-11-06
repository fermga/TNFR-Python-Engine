"""Static hybrid retention controls on the fixed reflected 45-node source.

The frozen source has x0=-x1=1/64 and the opposed child-1/2 phase-offset
profile m=1/40, epsilon=1/100. The candidate acute-cell excess is <1/100.
Only source-free observers, exact algebra and rational enclosures are used:
there is no trajectory, source acquisition, operator execution or event replay.
The separately materialized scalar endpoint is a declared static state, not
evidence of an executed Reception event or future binary64 invariance.
"""

from fractions import Fraction as Q

import mpmath
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions import Reception
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.physics import _organized_source_capture as owner
from tnfr.physics import _organized_source_port as port
from tnfr.types import Glyph

ORIGIN, EPSILON, AMPLITUDE = Q(1, 40), Q(1, 100), Q(1, 64)
CEILING = Q(1, 100)
FORM = (AMPLITUDE, -AMPLITUDE) + (Q(0),) * 43
OFFSETS = tuple(
    ORIGIN
    + (EPSILON if c == 1 else -EPSILON if c == 2 else 0)
    * (1 if j == 0 else -1 if j == 1 else 0)
    for c in range(5)
    for j in range(9)
)
REPRESENTATIVES = (0, 2, 3, 4)


@pytest.fixture(scope="module", autouse=True)
def no_response_event_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Retention controls cannot execute flow, events or source producers"
        )

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Reception, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (word_execution, "execute_network_operator_stage"),
            (owner, "_assess_organized_source_capture"),
            (owner, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def geometry():
    return port._derive_organized_source_port()


@pytest.fixture(scope="module")
def report():
    return owner._assess_organized_source_reflection_retention(
        form=FORM, phase_offsets=OFFSETS
    )


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _independent_sector(coordinates):
    result = [0] * 45
    for child in range(5):
        for k, j in enumerate(REPRESENTATIVES):
            result[9 * child + j] = coordinates[4 * child + k]
            result[9 * child + (1 - j) % 9] = -coordinates[4 * child + k]
    return tuple(result)


def test_frozen_source_excess_has_independent_cycle_and_matching_budget(
    geometry, report, mp
):
    alpha, epsilon = sp.symbols("alpha epsilon", real=True)
    deviations = _independent_sector(
        tuple(epsilon if i == 4 else -epsilon if i == 8 else 0 for i in range(20))
    )
    cycle_excess, matching = 0, 0
    form_cost = Q(0)
    for a, b, weight in geometry.weighted_edges:
        assert weight == 1
        form_cost += (FORM[a] - FORM[b]) ** 2 / 2
        difference = deviations[b] - deviations[a]
        if a // 9 == b // 9:
            assert (b - a) % 9 == 1
            cycle_excess += sp.cos(alpha) - sp.cos(alpha + difference)
        else:
            matching += 1 - sp.cos(difference)
    expected_cycle = (
        2 * sp.cos(alpha) * (1 - sp.cos(2 * epsilon) + 2 * (1 - sp.cos(epsilon)))
    )
    assert sp.trigsimp(sp.expand_trig(cycle_excess - expected_cycle)) == 0
    assert sp.trigsimp(matching - 8 * (1 - sp.cos(epsilon))) == 0
    assert form_cost == 6 * AMPLITUDE**2
    # 1-cos(t)<=t^2/2 and cos(alpha)<=1 supply this a priori ceiling.
    rational_upper = 6 * AMPLITUDE**2 + 10 * EPSILON**2
    assert rational_upper == Q(631, 256000) < CEILING
    a, e = 2 * mp.pi / 9, _mp(mp, EPSILON)
    expected = (
        _mp(mp, form_cost)
        + 2 * mp.cos(a) * (1 - mp.cos(2 * e) + 2 * (1 - mp.cos(e)))
        + 8 * (1 - mp.cos(e))
    )
    _contains(mp, report.excess_bounds, expected)
    assert 0 < report.excess_bounds.lo <= report.excess_bounds.hi < rational_upper
    assert report.state.form == FORM and report.state.phase_offsets == OFFSETS
    assert report.observation.conserved_form_total == 0
    assert report.observation.child_phase_offset_means == (ORIGIN,) * 5
    assert report.energy_ceiling == CEILING
    assert report.in_acute_cell and report.below_energy_ceiling and report.admitted
    assert report.retained_windings == (1,) * 5
    assert report.form_squared_upper_bound == Q(2, 25)
    assert report.form_absolute_upper_bound == Q(3, 10)
    assert not report.unresolved_conditions


def test_fixed_branch_angles_and_existing_full_boundary_cover_the_same_support(
    geometry, report, mp
):
    expected = []
    a = 2 * mp.pi / 9
    for left, right, _ in geometry.weighted_edges:
        delta = _mp(mp, OFFSETS[right] - OFFSETS[left])
        expected.append(a + delta if left // 9 == right // 9 else delta)
    assert len(expected) == len(report.edge_angle_bounds) == 90
    for bounds, angle in zip(report.edge_angle_bounds, expected, strict=True):
        _contains(mp, bounds, angle)
        assert -pi_interval().lo / 2 < bounds.lo <= bounds.hi < pi_interval().lo / 2
    closing = geometry.weighted_edges.index((8, 0, Q(1)))
    _contains(mp, report.edge_angle_bounds[closing], a)
    primitive_gap = (
        report.observation.phase_lift_bounds[0]
        - report.observation.phase_lift_bounds[8]
    )
    assert primitive_gap.hi < -pi_interval().hi
    barrier = report.barrier
    assert barrier.geometry == geometry and barrier.barrier_certified
    assert (
        barrier.signed_internal_boundary_faces
        == barrier.signed_matching_boundary_faces
        == 90
    )
    assert barrier.boundary_energy_lower_bound > Q(6, 25) > CEILING
    assert (
        report.domain_boundary_margin == barrier.boundary_energy_lower_bound - CEILING
    )


def test_fixed_slot_paths_give_the_declared_form_rail_bound(geometry):
    coordinates = sp.symbols("u:20", real=True)
    form = _independent_sector(coordinates)
    support = {frozenset((a, b)) for a, b, _ in geometry.weighted_edges}
    maximum_length = 0
    for child in range(5):
        assert form[9 * child + 5] == 0
        for slot in range(9):
            forward = (5 - slot) % 9
            direction = 1 if forward <= 4 else -1
            length = min(forward, 9 - forward)
            path = tuple(
                9 * child + (slot + direction * k) % 9 for k in range(length + 1)
            )
            assert path[-1] == 9 * child + 5
            assert all(frozenset(edge) in support for edge in zip(path, path[1:]))
            differences = tuple(form[a] - form[b] for a, b in zip(path, path[1:]))
            assert sp.expand(sum(differences) - form[9 * child + slot]) == 0
            cauchy_slack = sum(
                (differences[i] - differences[j]) ** 2
                for i in range(length)
                for j in range(i + 1, length)
            )
            assert (
                sp.expand(
                    length * sum(d**2 for d in differences)
                    - form[9 * child + slot] ** 2
                    - cauchy_slack
                )
                == 0
            )
            maximum_length = max(maximum_length, length)
    assert maximum_length == 4
    # Every path edge is a distinct summand of 2*E_form, and acute-cell Jensen
    # gives E_form<=H-H*. Thus |x_i|^2<=8*(H-H*) on the admitted sector.
    assert 2 * maximum_length * CEILING == Q(2, 25) < Q(3, 10) ** 2 < 1


def test_ideal_pair_work_and_composition_follow_from_actual_fine_quadratic_form(
    geometry,
):
    beta, second_beta = sp.symbols("beta second_beta", positive=True)
    source = _independent_sector(sp.symbols("u:20", real=True))

    def energy(form):
        return sum(
            sp.Rational(weight.numerator, 2 * weight.denominator)
            * (form[a] - form[b]) ** 2
            for a, b, weight in geometry.weighted_edges
        )

    def gradient_zero(form):
        return sum(
            form[0] - form[b if a == 0 else a]
            for a, b, _ in geometry.weighted_edges
            if 0 in (a, b)
        )

    q = gradient_zero(source)
    first = list(source)
    first[0] -= beta * q / 5
    first[1] += beta * q / 5
    first_work = -2 * beta * (1 - 3 * beta / 5) * q**2 / 5
    assert sp.expand(energy(first) - energy(source) - first_work) == 0
    q_next = gradient_zero(first)
    assert sp.expand(q_next - (1 - 6 * beta / 5) * q) == 0
    second = list(first)
    second[0] -= second_beta * q_next / 5
    second[1] += second_beta * q_next / 5
    second_work = -2 * second_beta * (1 - 3 * second_beta / 5) * q_next**2 / 5
    assert sp.expand(energy(second) - energy(source) - first_work - second_work) == 0
    for coefficient in (beta, second_beta):
        # On 0<coefficient<=1, both displayed terms are nonnegative and the
        # first is positive. Independent continuous loss adds another debit.
        positive_factor = 2 * coefficient * (1 - 3 * coefficient / 5) / 5
        assert (
            sp.expand(
                positive_factor
                - 4 * coefficient / 25
                - 6 * coefficient * (1 - coefficient) / 25
            )
            == 0
        )


def test_separately_materialized_static_endpoint_stays_in_domain(report):
    mix = resolve_runtime_operator_factors(None, Glyph.EN, {})["EN_mix"]
    h = float(AMPLITUDE)
    amplitude = Q((1.0 - mix) * h + mix * (-h / 5))
    assert 0 < amplitude < AMPLITUDE
    endpoint = (amplitude, -amplitude) + FORM[2:]
    after = owner._assess_organized_source_reflection_retention(
        form=endpoint, phase_offsets=OFFSETS
    )
    jump = port._observe_organized_source_form_jump(
        form_before=FORM, form_after=endpoint, phase_offsets=OFFSETS
    )
    assert jump.storage_jump == 6 * (amplitude**2 - AMPLITUDE**2) < 0
    assert after.admitted and after.retained_windings == (1,) * 5
    assert after.excess_bounds.hi < report.excess_bounds.lo
    # This input was materialized without calling Reception. Its assessment
    # proves conditional membership, not execution or inherited future gates.


def test_acute_in_rail_high_form_is_not_mistaken_for_the_small_storage_domain():
    form = (Q(1, 4), -Q(1, 4)) + FORM[2:]
    failed = owner._assess_organized_source_reflection_retention(
        form=form, phase_offsets=OFFSETS
    )
    assert failed.in_acute_cell and max(abs(value) for value in form) < 1
    assert failed.excess_bounds.lo > CEILING
    assert not failed.below_energy_ceiling and not failed.admitted
    assert failed.form_squared_upper_bound is failed.form_absolute_upper_bound is None
    assert failed.retained_windings is None and failed.unresolved_conditions


def test_small_physical_storage_does_not_replace_the_fixed_acute_lift_admission():
    a = Q(2, 3)
    deviations = _independent_sector((5 * a, 3 * a, 2 * a, a) * 5)
    offsets = tuple(ORIGIN + value for value in deviations)
    failed = owner._assess_organized_source_reflection_retention(
        form=(Q(0),) * 45, phase_offsets=offsets
    )
    assert failed.state.phase_offsets == offsets
    assert failed.below_energy_ceiling and failed.excess_bounds.hi < 0
    assert not failed.in_acute_cell and not failed.admitted
    assert any(
        bounds.lo > pi_interval().hi / 2 or bounds.hi < -pi_interval().hi / 2
        for bounds in failed.edge_angle_bounds
    )
    assert failed.retained_windings is None
    assert failed.form_squared_upper_bound is failed.form_absolute_upper_bound is None


@pytest.mark.parametrize("defect", ("off_sector", "wrong_origin"))
def test_raw_sector_rejection_precedes_the_barrier_and_fine_observation(
    defect, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Off-sector state reached a geometry or fine observation")

    form, offsets = list(FORM), OFFSETS
    if defect == "off_sector":
        form[2] += Q(1, 1024)
        form[3] -= Q(1, 1024)
        assert sum(form) == 0
    else:
        offsets = tuple(value + Q(1, 1000) for value in OFFSETS)
    monkeypatch.setattr(owner, "_derive_organized_source_capture_barrier", forbidden)
    monkeypatch.setattr(port, "_observe_organized_source_port", forbidden)
    with pytest.raises(ValueError, match="reflection"):
        owner._assess_organized_source_reflection_retention(
            form=tuple(form), phase_offsets=offsets
        )


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, float("inf"), "short"))
def test_invalid_primitive_admission_precedes_geometry(field, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid primitive reached geometry")

    arguments = dict(form=FORM, phase_offsets=OFFSETS)
    values = arguments[field]
    arguments[field] = values[:-1] if bad == "short" else (bad,) + values[1:]
    monkeypatch.setattr(owner, "_derive_organized_source_capture_barrier", forbidden)
    monkeypatch.setattr(port, "_derive_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_reflection_retention(**arguments)

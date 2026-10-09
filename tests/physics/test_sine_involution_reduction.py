"""Exact signed families and independent full-row controls, without trajectories."""

from dataclasses import FrozenInstanceError, dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import (
    _comparison_from_state,
    _sine_state_from_rows,
)
from tnfr.physics.relational_sine_regional import assess_sine_cycle_barrier
from tnfr.physics.relational_sine_symmetry import (
    assess_sine_cycle_symmetry,
    assess_sine_involution_reduction,
)

EDGES = tuple((i, (i + 1) % 5) for i in range(5)) + tuple((i, 5 + i) for i in range(5))
REFLECTION = (4, 3, 2, 1, 0, 9, 8, 7, 6, 5)


def _source(*, form=(0,) * 10, phase=(0,) * 10, capacity=(1,) * 10, model=None):
    rows = [[] for _ in range(10)]
    for i, j in EDGES:
        rows[i].append(j)
        rows[j].append(i)
    return _comparison_from_state(
        _sine_state_from_rows(
            tuple(range(10)),
            EDGES,
            tuple(map(Q, form)),
            tuple(map(Q, phase)),
            tuple(map(Q, capacity)),
            tuple(map(tuple, rows)),
        ),
        model
        or RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def _assess(source, **kwargs):
    return assess_sine_involution_reduction(
        source, **({"permutation_indices": REFLECTION} | kwargs)
    )


def _odd(values):
    a, b, c, d = values
    return a, b, Q(0), -b, -a, c, d, Q(0), -d, -c


def test_family_and_captured_membership_have_distinct_meanings():
    source = _source(form=(0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
    report = _assess(source)
    assert report.family_invariance_certified and report.status == "certified"
    assert report.representative_indices == (0, 1, 5, 6)
    assert report.orbits == ((0, 4), (1, 3), (2,), (5, 9), (6, 8), (7,))
    assert not report.source_membership_certified
    assert not report.source_trajectory_reduction_certified
    assert report.form_reconstruction_residuals[2] == -1
    state = report.evaluate((0,) * 4, (0,) * 4)
    assert state.comparison.epi == (0,) * 10
    assert source.epi[2] == 1 and report.source.epi[2] == 1
    with pytest.raises(FrozenInstanceError):
        report.sign = 1


@pytest.mark.parametrize("sign", (-1, 1))
def test_full_nonlinear_rows_and_storage_match_independent_fine_equations(sign):
    model = RelationalExchangeModel(
        Q(3, 2), epi_weight=2, phase_weight=3, phase_domain="regular"
    )
    capacity = (2, 3, 11, 3, 2, 5, 7, 13, 7, 5)
    report = _assess(_source(model=model, capacity=capacity), sign=sign)
    count = len(report.representative_indices)
    forms = tuple(Q(i - 2, i + 3) for i in range(count))
    phases = tuple(Q(2 * i - 3, i + 5) for i in range(count))
    state = report.evaluate(forms, phases)
    assert state.full_row_equality_certified
    assert state.clock == "original_structural_t"
    assert state.comparison.capacity == capacity
    fine_x, fine_theta = {}, {}
    for representative, x, theta in zip(report.representative_indices, forms, phases):
        fine_x[representative], fine_theta[representative] = x, theta
        other = REFLECTION[representative]
        fine_x[other], fine_theta[other] = sign * x, sign * theta
    for i in range(10):
        fine_x.setdefault(i, Q(0))
        fine_theta.setdefault(i, Q(0))
    assert state.comparison.epi == tuple(fine_x[i] for i in range(10))
    assert state.comparison.phase == tuple(fine_theta[i] for i in range(10))

    def number(value):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator

    with mp.workdps(90):
        e, w = map(number, model.effective_weights)
        beta = number(model.storage_scale)
        x = tuple(number(fine_x[i]) for i in range(10))
        theta = tuple(number(fine_theta[i]) for i in range(10))
        for i in range(10):
            adjacent = tuple(
                j if i == left else left for left, j in EDGES if i in (left, j)
            )
            q = mp.fsum(x[i] - x[j] for j in adjacent)
            s = mp.fsum(mp.sin(theta[j] - theta[i]) for j in adjacent)
            x_rate = number(capacity[i]) / len(adjacent) * (-e * q + w * s / mp.pi)
            phase_rate = number(capacity[i]) / len(adjacent) * w * q / (beta * mp.pi)
            for expected, enclosure in (
                (x_rate, state.comparison.form_rates[i]),
                (phase_rate, state.comparison.phase_rates[i]),
            ):
                assert number(enclosure.lo) <= expected <= number(enclosure.hi)
        storage = mp.fsum(
            (x[j] - x[i]) ** 2 / 2 + beta * (1 - mp.cos(theta[j] - theta[i]))
            for i, j in EDGES
        )
        assert (
            number(state.comparison.storage.lo)
            <= storage
            <= number(state.comparison.storage.hi)
        )
    for residual in (
        *state.form_rate_reconstruction_residual_bounds,
        *state.phase_rate_reconstruction_residual_bounds,
    ):
        assert residual.contains(0)


def test_negative_sign_preserves_nonzero_winding_not_the_ordinary_reflection_obstruction():
    phase = tuple((i - 2) * Q(1256637, 10**6) for i in range(5)) * 2
    source = _source(phase=phase)
    signed = _assess(source)
    assert signed.source_trajectory_reduction_certified
    assert assess_sine_cycle_barrier(source, cycle=range(5)).initial_winding == 1
    ordinary = assess_sine_cycle_symmetry(
        source, permutation_indices=REFLECTION, cycle=range(5)
    )
    assert ordinary.cycle_orientation_reversed
    assert not ordinary.trajectory_symmetry_certified
    assert not ordinary.zero_winding_when_nonantipodal
    shifted = _assess(
        replace(source, phase=tuple(value + Q(1, 7) for value in source.phase))
    )
    assert shifted.family_invariance_certified
    assert not shifted.source_membership_certified
    assert shifted.phase_lift_reconstruction_residuals[2] == -Q(1, 7)


def test_zero_dimensional_negative_identity_and_unreduced_positive_identity():
    source = _source()
    zero = _assess(source, permutation_indices=tuple(range(10)))
    assert zero.representative_indices == ()
    assert zero.reconstruction_matrix == ((),) * 10
    state = zero.evaluate((), ())
    assert state.comparison.epi == state.comparison.phase == (0,) * 10
    assert state.form_rates == state.phase_rates == ()
    assert all(value == I(0) for value in state.comparison.form_rates)
    identity = _assess(source, permutation_indices=tuple(range(10)), sign=1)
    forms, phases = tuple(Q(i, 7) for i in range(10)), tuple(
        Q(-i, 9) for i in range(10)
    )
    direct = identity.evaluate(forms, phases)
    assert direct.comparison.epi == forms and direct.comparison.phase == phases
    assert len(direct.form_rates) == 10


def test_capacity_and_support_are_full_family_premises_even_for_a_fixed_source():
    base = _source()
    for source, permutation, reason in (
        (
            replace(base, capacity=(Q(2),) + base.capacity[1:]),
            REFLECTION,
            "held_capacity_not_preserved",
        ),
        (base, (1, 0, 2, 3, 4, 5, 6, 7, 8, 9), "full_support_not_preserved"),
    ):
        report = _assess(source, permutation_indices=permutation)
        assert report.source_membership_certified
        assert not report.family_invariance_certified
        assert not report.source_trajectory_reduction_certified
        assert reason in report.reasons and report.status == "unavailable"
        with pytest.raises(ValueError, match="certified invariant family"):
            report.evaluate(
                (0,) * len(report.representative_indices),
                (0,) * len(report.representative_indices),
            )
    inactive = _assess(_source(capacity=(0,) * 10)).evaluate((1, 2, 3, 4), (4, 3, 2, 1))
    assert all(
        value == I(0)
        for value in (*inactive.comparison.form_rates, *inactive.comparison.phase_rates)
    )


def test_evaluation_rebuilds_every_consumed_cache_and_retains_actual_source_primitives():
    report = _assess(_source())
    coordinates = (Q(1, 2), Q(-1, 3), Q(2, 5), Q(3, 7))
    expected = report.evaluate(coordinates, coordinates)
    poisoned = replace(
        report,
        reconstruction_matrix=((Q(99),),),
        representative_indices=(9,),
        family_invariance_certified=False,
        source_membership_certified=False,
        source=replace(
            report.source,
            storage=I(-99),
            form_rates=(),
            phase_rates=(),
            form_gradient=(),
            relative_resultant=(),
        ),
    ).evaluate(coordinates, coordinates)
    assert poisoned.comparison == expected.comparison
    assert poisoned.reduction.reconstruction_matrix == report.reconstruction_matrix
    assert poisoned.form_rates == expected.form_rates
    for field in ("epi", "phase", "capacity"):
        bad_source = replace(
            report.source, **{field: (False,) + getattr(report.source, field)[1:]}
        )
        with pytest.raises((TypeError, ValueError)):
            replace(report, source=bad_source).evaluate(coordinates, coordinates)


@pytest.mark.parametrize("sign", (True, False, 0, 2, Q(-1), -1.0, "-1"))
def test_sign_admission_is_exact_and_nonboolean(sign):
    with pytest.raises((TypeError, ValueError)):
        _assess(_source(), sign=sign)


@pytest.mark.parametrize(
    "permutation",
    ((0, 1), (0,) * 10, (True,) + REFLECTION[1:], (1, 2, 0, 3, 4, 5, 6, 7, 8, 9)),
)
def test_permutation_requires_an_exact_full_involution(permutation):
    with pytest.raises((TypeError, ValueError)):
        _assess(_source(), permutation_indices=permutation)


@pytest.mark.parametrize(
    "coordinates",
    ((0, 1, 2), (True, 1, 2, 3), (float("nan"), 1, 2, 3), "0000", {0, 1, 2, 3}),
)
def test_reduced_coordinates_have_shared_real_and_ordered_admission(coordinates):
    with pytest.raises((TypeError, ValueError)):
        _assess(_source()).evaluate(coordinates, (0,) * 4)


def test_coordinate_iterator_has_a_bounded_dimension_admission():
    def excessive():
        for _ in range(6):
            yield 0
        pytest.fail("coordinate admission consumed an unbounded iterator")

    with pytest.raises(ValueError):
        _assess(_source()).evaluate(excessive(), (0,) * 4)


def test_source_order_and_relabeling_retain_the_actual_fixed_point_family():
    source = _source(
        form=_odd((Q(1, 2), Q(1, 3), Q(1, 4), Q(1, 5))),
        phase=_odd((Q(2, 3), Q(-3, 4), Q(4, 5), Q(-5, 6))),
    )
    original = _assess(source)
    order = (8, 0, 5, 3, 6, 9, 1, 7, 4, 2)
    reordered = replace(
        source,
        nodes=tuple(f"n{i}" for i in order),
        edges=tuple((f"n{i}", f"n{j}") for i, j in EDGES),
        epi=tuple(source.epi[i] for i in order),
        phase=tuple(source.phase[i] for i in order),
        capacity=tuple(source.capacity[i] for i in order),
        degrees=tuple(source.degrees[i] for i in order),
    )
    permutation = tuple(order.index(REFLECTION[i]) for i in order)
    mapped = _assess(reordered, permutation_indices=permutation)
    assert mapped.source_trajectory_reduction_certified
    result = mapped.evaluate(
        mapped.source_form_coordinates, mapped.source_phase_coordinates
    )
    baseline = original.evaluate(
        original.source_form_coordinates, original.source_phase_coordinates
    )
    assert result.comparison.epi == reordered.epi
    assert result.comparison.phase == reordered.phase
    assert result.comparison.storage == baseline.comparison.storage
    assert result.comparison.form_rates == tuple(
        baseline.comparison.form_rates[i] for i in order
    )


def test_sdk_exports_exact_reconstruction_and_checks_nested_labels():
    from tnfr.sdk import relational_report_to_dict

    report = _assess(_source())
    state = report.evaluate((1, 2, 3, 4), (Q(1, 7), Q(-1, 9), Q(2, 11), Q(-2, 13)))
    assert relational_report_to_dict(report)["report_type"] == "SineInvolutionReduction"
    assert report.to_dict()["schema"] == "tnfr.sine-involution-reduction.v1"
    assert relational_report_to_dict(state)["report_type"] == "SineInvolutionState"
    assert state.to_dict()["schema"] == "tnfr.sine-involution-state.v1"

    @dataclass(frozen=True)
    class Opaque:
        index: int

    invalid = replace(
        report,
        source=replace(report.source, nodes=(Opaque(0),) + report.source.nodes[1:]),
    )
    for value in (invalid, replace(state, reduction=invalid)):
        with pytest.raises(TypeError):
            relational_report_to_dict(value)

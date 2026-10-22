"""Adversarial source, chart and directional admission for a finite corridor.

These controls only evaluate instantaneous state and analytical bounds. They
do not integrate a trajectory or substitute an uncertain box for an exact
member of the sign/reflection family.
"""

from copy import copy
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from tests.physics.test_sine_cycle_barrier import _state
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_corridor import assess_sine_saddle_corridor


@pytest.fixture(scope="module")
def source():
    a, b, c, d = Q(1, 6), Q(1, 6), Q(1, 4), Q(5, 24)
    phase = (Q(-9, 5), Q(-9, 10), Q(0), Q(9, 10), Q(9, 5)) * 2
    return _state(epi=(a, b, 0, -b, -a, c, d, 0, -d, -c), phase=phase)


def _assess(source, **changes):
    arguments = dict(cycle=range(5), lower_phase=Q(-2), upper_phase=Q(-3, 2))
    return assess_sine_saddle_corridor(source, **(arguments | changes))


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def test_complete_source_is_rebuilt_before_using_derived_fields(source, report):
    assert report.directed_exit_certified
    forged = replace(
        source,
        storage=I(-1000),
        form_rates=(),
        phase_rates=(I(999),) * 10,
        relative_resultant=(),
        form_gradient=(),
    )
    rebuilt = _assess(forged)
    for field in (
        "full_storage_bounds",
        "initial_momentum",
        "initial_momentum_rate_bounds",
        "initial_reaction_velocity",
        "force_lower_bound",
        "left_exit_momentum_upper_bound",
        "residence_time_upper_bound",
    ):
        assert getattr(rebuilt, field) == getattr(report, field)
    assert rebuilt.directed_exit_certified


@pytest.mark.parametrize("field", ("epi", "phase"))
def test_nonzero_transverse_error_is_not_discarded_as_nominal_symmetry(source, field):
    values = list(getattr(source, field))
    values[4] += Q(1, 2**150)
    actual = _assess(replace(source, **{field: tuple(values)}))
    assert not actual.source_family_certified
    assert not actual.directed_exit_certified
    assert actual.exit_winding is None
    assert actual.residence_time_upper_bound is None
    assert "exact_signed_source_membership_not_certified" in actual.reasons


def test_unavailable_source_reaction_velocity_still_uses_its_actual_full_row(source):
    forms = list(source.epi)
    forms[4] += Q(1, 7)
    actual = _assess(replace(source, epi=tuple(forms)))
    expected = sum((forms[0] - forms[j] for j in (1, 4, 5)), Q(0)) / 3
    assert not actual.source_family_certified
    assert actual.initial_reaction_velocity == expected


@pytest.mark.parametrize("field", ("epi", "phase"))
def test_common_origin_is_not_silently_removed_to_obtain_a_signed_source(source, field):
    shifted = replace(
        source, **{field: tuple(x + Q(1, 7) for x in getattr(source, field))}
    )
    actual = _assess(shifted)
    assert not actual.source_family_certified
    assert not actual.directed_exit_certified


def test_negative_momentum_does_not_pass_the_squared_directional_inequality(
    source, report
):
    reversed_source = replace(source, epi=tuple(-x for x in source.epi))
    actual = _assess(reversed_source)
    assert actual.full_storage_bounds == report.full_storage_bounds
    assert actual.initial_momentum == -report.initial_momentum < 0
    assert actual.initial_momentum**2 > actual.left_exit_momentum_upper_bound**2
    assert actual.positive_force_certified
    assert not actual.left_exit_excluded and not actual.directed_exit_certified
    assert "lower_exit_momentum_exclusion_not_certified" in actual.reasons


def test_real_path_lifts_are_required_even_inside_the_signed_family(source):
    phases = list(source.phase)
    phases[1] += 7
    phases[3] -= 7
    phases[6] += 7
    phases[8] -= 7
    actual = _assess(replace(source, phase=tuple(phases)))
    assert actual.source_family_certified
    assert actual.source_strip_certified
    assert not actual.initial_branches_certified
    assert not actual.directed_exit_certified
    assert "initial_winding_one_path_lifts_not_certified" in actual.reasons


@pytest.mark.parametrize(
    "field,value",
    (
        ("lower_phase", True),
        ("upper_phase", False),
        ("lower_phase", float("nan")),
        ("upper_phase", float("inf")),
        ("lower_phase", "-2"),
        ("lower_phase", Q(-3)),
        ("lower_phase", Q(-3, 2)),
        ("upper_phase", Q(-2)),
        ("upper_phase", Q(0)),
    ),
)
def test_invalid_strip_or_scalar_cannot_reuse_a_source_certificate(
    source, field, value
):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, **{field: value})


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
def test_boolean_primitive_is_rejected_before_numeric_equalities(source, field):
    values = list(getattr(source, field))
    values[2] = True if field == "capacity" else False
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, **{field: tuple(values)}))


@pytest.mark.parametrize("field", ("epi_weight", "phase_weight", "storage_scale"))
def test_boolean_law_coefficients_cannot_impersonate_the_conservative_model(
    source, field
):
    model = copy(source.reference_model)
    object.__setattr__(model, field, False if field == "epi_weight" else True)
    with pytest.raises((TypeError, ValueError)):
        _assess(replace(source, reference_model=model))


def test_original_order_and_opaque_compatible_labels_do_not_define_the_reduced_chart(
    source, report
):
    order = (8, 0, 7, 2, 9, 4, 6, 3, 5, 1)
    changed = _state(
        epi=source.epi,
        phase=source.phase,
        order=order,
        label=lambda i: ("label", i),
    )
    actual = _assess(changed, cycle=tuple(("label", i) for i in range(5)))
    assert actual.directed_exit_certified
    for field in (
        "form_coordinates",
        "phase_coordinates",
        "corridor_coordinates",
        "initial_momentum",
        "full_storage_bounds",
        "residence_time_upper_bound",
    ):
        assert getattr(actual, field) == getattr(report, field)
    assert actual.cycle_indices == tuple(order.index(i) for i in range(5))

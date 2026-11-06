"""Constructed image controls for joint represented state/capacity boxes."""

import math
from fractions import Fraction as Q
from itertools import product

import pytest

from tnfr.operators._scale_operator_kernel import propose_scale_operator
from tnfr.research import nul_reset as owner
from tnfr.types import Glyph


def _image(**overrides):
    arguments = dict(
        factor=0.5,
        capacity_bounds=(0.5, 1.0, 2.0),
        form_bounds=(-3.0, 0.25, 3.0),
        pressure_bounds=(-0.25, 0.125, 0.5),
    )
    arguments.update(overrides)
    return owner.bound_nul_capacity_reset(**arguments)


def test_complete_cartesian_grid_is_inside_actual_three_channel_image():
    image = _image()
    assert image.image_capacity == owner.NulCoordinateBounds(Q(1, 4), Q(1, 2), Q(1))
    assert image.image_form == owner.NulCoordinateBounds(Q(-1), Q(1, 8), Q(1))
    assert image.image_stored_pressure == owner.NulCoordinateBounds(
        Q(-1, 2), Q(1, 4), Q(1)
    )
    for capacity, form, pressure in product(
        (0.5, 0.75, 1.0, 1.5, 2.0),
        (-3.0, -0.5, 0.25, 0.5, 3.0),
        (-0.25, 0.0, 0.125, 0.25, 0.5),
    ):
        proposal = propose_scale_operator(
            glyph=Glyph.NUL,
            factor=0.5,
            vf_before=capacity,
            dnfr_before=pressure,
            edge_aware_enabled=True,
            epi_before=form,
        )
        for value, bounds in (
            (proposal.vf_after, image.image_capacity),
            (proposal.epi_after, image.image_form),
            (proposal.dnfr_after, image.image_stored_pressure),
        ):
            assert bounds.lower <= Q(value) <= bounds.upper


def test_fixed_capacity_api_retains_every_numerical_field_and_diagnostic():
    joint = _image(capacity_bounds=(2.5, 2.5, 2.5), factor=0.37)
    fixed = owner.bound_nul_reset(
        factor=0.37,
        capacity=2.5,
        form_bounds=(-3.0, 0.25, 3.0),
        pressure_bounds=(-0.25, 0.125, 0.5),
    )
    assert fixed.capacity_before == joint.source_capacity.nominal
    assert fixed.capacity_after == joint.image_capacity.nominal
    for field in (
        "factor",
        "source_form",
        "image_form",
        "source_stored_pressure",
        "image_stored_pressure",
        "proposals",
        "endpoint_stored_product_defects",
    ):
        assert getattr(fixed, field) == getattr(joint, field)
    assert joint.endpoint_capacity_ratios == (fixed.actual_capacity_ratio,) * 3
    assert fixed.actual_capacity_ratio != joint.factor


def test_subnormal_rounding_uses_actual_capacity_image_not_ideal_factor():
    unit = math.ulp(0.0)
    image = _image(capacity_bounds=(2 * unit, 3 * unit, 5 * unit))
    assert image.image_capacity == owner.NulCoordinateBounds(
        Q(unit), 2 * Q(unit), 2 * Q(unit)
    )
    assert image.endpoint_capacity_ratios == (Q(1, 2), Q(2, 3), Q(2, 5))
    assert image.image_capacity.nominal != image.factor * image.source_capacity.nominal


def test_admissible_nominal_capacity_does_not_hide_lower_member_underflow():
    unit = math.ulp(0.0)
    assert (
        owner.bound_nul_reset(
            factor=0.5,
            capacity=2 * unit,
            form_bounds=(0.0, 0.0, 0.0),
            pressure_bounds=(0.0, 0.0, 0.0),
        ).capacity_after
        > 0
    )
    with pytest.raises(ValueError, match="positive represented successor"):
        _image(capacity_bounds=(unit, 2 * unit, 3 * unit))


@pytest.mark.parametrize(
    "bounds",
    (
        (0.0, 1.0, 2.0),
        (-1.0, 1.0, 2.0),
        (2.0, 1.0, 3.0),
        (0.5, True, 2.0),
        (0.5, 1, 2.0),
        (0.5, 1.0, math.inf),
        (-0.0, 1.0, 2.0),
        (0.5, 1.0),
    ),
)
def test_invalid_capacity_endpoints_reject_before_any_proposal(bounds, monkeypatch):
    def forbidden(**arguments):
        pytest.fail("capacity admission must precede numerical proposals")

    monkeypatch.setattr(owner, "propose_scale_operator", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _image(capacity_bounds=bounds)

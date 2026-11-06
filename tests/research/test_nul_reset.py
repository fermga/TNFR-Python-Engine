"""Constructed numerical controls for one detached hard-projection NUL image."""

import math
import sys
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators._scale_operator_kernel import propose_scale_operator
from tnfr.operators.preconditions import OperatorPreconditionError, validate_contraction
from tnfr.research import nul_reset
from tnfr.types import Glyph


def _image(**overrides):
    arguments = dict(
        factor=0.5,
        capacity=1.0,
        form_bounds=(-0.5, 0.25, 0.75),
        pressure_bounds=(-0.25, 0.125, 0.5),
    )
    arguments.update(overrides)
    return nul_reset.bound_nul_reset(**arguments)


def test_dyadic_image_preserves_both_asymmetric_error_directions():
    image = _image()
    assert image.image_form == nul_reset.NulCoordinateBounds(Q(-1, 4), Q(1, 8), Q(3, 8))
    assert image.image_form.lower_error == Q(3, 8)
    assert image.image_form.upper_error == Q(1, 4)
    assert image.image_form.radius == Q(3, 8)
    assert image.image_stored_pressure == nul_reset.NulCoordinateBounds(
        Q(-1, 2), Q(1, 4), Q(1)
    )
    assert image.capacity_before == 1
    assert image.capacity_after == image.actual_capacity_ratio == Q(1, 2)
    assert image.endpoint_stored_product_defects == (0, 0, 0)


def test_hard_saturation_crossing_encloses_the_cartesian_input_box():
    image = _image(form_bounds=(-3.0, 0.5, 3.0))
    assert image.image_form == nul_reset.NulCoordinateBounds(Q(-1), Q(1, 4), Q(1))
    assert image.image_form.lower_error == Q(5, 4)
    assert image.image_form.upper_error == Q(3, 4)
    # Interior combinations include off-diagonal form/pressure choices: only
    # channel independence and monotonicity justify the endpoint rectangle.
    for form in (-2.5, -0.5, 0.0, 1.0, 2.5):
        for pressure in (-0.125, 0.0, 0.375):
            proposal = propose_scale_operator(
                glyph=Glyph.NUL,
                factor=0.5,
                vf_before=1.0,
                dnfr_before=pressure,
                edge_aware_enabled=True,
                epi_before=form,
            )
            assert (
                image.image_form.lower
                <= Q(proposal.epi_after)
                <= image.image_form.upper
            )
            assert image.image_stored_pressure.lower <= Q(proposal.dnfr_after)
            assert Q(proposal.dnfr_after) <= image.image_stored_pressure.upper


def test_positive_lower_rail_preserves_the_actual_reversed_form_change():
    image = _image(form_bounds=(0.125, 0.25, 0.5), epi_min=0.5, epi_max=1.0)
    assert image.source_form.nominal == Q(1, 4)
    assert image.image_form == nul_reset.NulCoordinateBounds(Q(1, 2), Q(1, 2), Q(1, 2))
    assert image.proposals[1].raw_epi_after == 0.125
    assert image.proposals[1].clip_delta == 0.375
    # Numerical image success deliberately supplies no live admission verdict.
    assert not hasattr(image, "admitted")


def test_degenerate_rail_is_a_constant_form_image():
    image = _image(epi_min=-0.25, epi_max=-0.25)
    assert image.image_form == nul_reset.NulCoordinateBounds(
        Q(-1, 4), Q(-1, 4), Q(-1, 4)
    )


def test_power_of_two_neighbor_widths_do_not_become_symmetric():
    image = _image(
        form_bounds=(math.nextafter(1.0, 0.0), 1.0, math.nextafter(1.0, math.inf))
    )
    assert image.image_form.nominal == Q(1, 2)
    assert image.image_form.lower_error == Q(1, 2**54)
    assert image.image_form.upper_error == Q(1, 2**53)


def test_subnormal_form_ties_use_the_shared_represented_endpoints():
    unit = math.ulp(0.0)
    image = _image(form_bounds=(unit, 2 * unit, 3 * unit))
    assert image.image_form == nul_reset.NulCoordinateBounds(Q(0), Q(unit), 2 * Q(unit))
    assert image.image_form.lower != image.factor * image.source_form.lower
    assert image.image_form.upper != image.factor * image.source_form.upper
    negative = _image(form_bounds=(-3 * unit, -2 * unit, -unit))
    assert negative.image_form == nul_reset.NulCoordinateBounds(
        -2 * Q(unit), -Q(unit), Q(0)
    )
    assert negative.proposals[-1].raw_epi_after == 0.0
    assert math.copysign(1.0, negative.proposals[-1].raw_epi_after) == 1.0
    assert math.copysign(1.0, negative.proposals[-1].epi_after) == 1.0


def test_common_density_admission_uses_the_worst_box_corner_not_the_nominal():
    graph = nx.Graph(
        GLYPH_FACTORS={"NUL_scale": 0.5},
        EDGE_AWARE_ENABLED=True,
        CLIP_MODE="hard",
        EPI_MIN=0.0,
        EPI_MAX=1.0,
        NUL_MIN_VF=0.1,
        NUL_MIN_EPI=0.1,
        NUL_MAX_DENSITY=2.0,
    )
    graph.add_node("target", EPI=0.25, nu_f=1.0, delta_nfr=0.5)
    # The largest absolute pressure is 0.5 and the smallest denominator is
    # 0.25. Their density 2 equals the inclusive configured maximum.
    validate_contraction(graph, "target")
    for form in (0.25, 0.5, 0.75):
        for pressure in (-0.25, 0.125, 0.5):
            graph.nodes["target"].update(EPI=form, delta_nfr=pressure)
            validate_contraction(graph, "target")

    graph.graph["NUL_MAX_DENSITY"] = 1.5
    graph.nodes["target"].update(EPI=0.5, delta_nfr=0.125)
    validate_contraction(graph, "target")
    graph.nodes["target"].update(EPI=0.25, delta_nfr=0.5)
    with pytest.raises(OperatorPreconditionError, match="density proxy"):
        validate_contraction(graph, "target")


def test_capacity_ratio_and_product_defect_use_actual_rounded_endpoints():
    image = _image(
        factor=0.37,
        capacity=2.5,
        pressure_bounds=(-0.8, -0.8, -0.8),
    )
    assert image.capacity_after == Q(0.925)
    assert image.actual_capacity_ratio != image.factor
    assert image.proposals[1].binary64_inverse_product_residual == 0.0
    defect = Q(-486875635391405, 5070602400912917605986812821504)
    assert image.endpoint_stored_product_defects == (defect, defect, defect)


def test_helper_uses_exactly_three_shared_proposals_without_graph_execution(
    monkeypatch,
):
    calls = []

    def observe(**kwargs):
        calls.append(kwargs)
        return propose_scale_operator(**kwargs)

    monkeypatch.setattr(nul_reset, "propose_scale_operator", observe)
    image = _image()
    assert len(calls) == len(image.proposals) == 3
    assert [(item["epi_before"], item["dnfr_before"]) for item in calls] == [
        (-0.5, -0.25),
        (0.25, 0.125),
        (0.75, 0.5),
    ]
    assert all(item["edge_aware_enabled"] is True for item in calls)
    assert all(item["clip_mode"] == "hard" for item in calls)


def test_report_and_retained_proposals_are_immutable():
    image = _image()
    with pytest.raises(FrozenInstanceError):
        image.capacity_after = Q(0)
    with pytest.raises(FrozenInstanceError):
        image.image_form.lower = Q(-10)
    with pytest.raises(FrozenInstanceError):
        image.proposals[0].vf_after = 0.0


@pytest.mark.parametrize("value", [True, 1, Q(1, 2), "0.5", None])
@pytest.mark.parametrize("field", ["factor", "capacity", "epi_min", "epi_max"])
def test_nonbinary64_scalars_are_rejected_before_proposals(value, field, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("proposal called before binary64 scalar admission")

    monkeypatch.setattr(nul_reset, "propose_scale_operator", forbidden)
    with pytest.raises(TypeError, match="binary64 float"):
        _image(**{field: value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.0])
@pytest.mark.parametrize("field", ["form_bounds", "pressure_bounds"])
def test_invalid_represented_endpoints_are_not_materialized(value, field):
    with pytest.raises(ValueError):
        _image(**{field: (-0.25, value, 0.5)})


@pytest.mark.parametrize("field", ["form_bounds", "pressure_bounds"])
@pytest.mark.parametrize("values", [(0.5, 0.0, 1.0), (-1.0, 1.0, 0.5)])
def test_unordered_bounds_reject(field, values):
    with pytest.raises(ValueError, match="lower <= nominal <= upper"):
        _image(**{field: values})


@pytest.mark.parametrize("values", [(), (0.0, 1.0), iter((0.0, 0.5, 1.0)), "abc"])
def test_endpoint_shape_requires_a_finite_three_value_sequence(values):
    with pytest.raises(TypeError, match="three ordered"):
        _image(form_bounds=values)


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.0, 2.0])
def test_factor_domain_is_rejected_by_shared_owner(factor):
    with pytest.raises(ValueError, match="NUL_scale"):
        _image(factor=factor)


@pytest.mark.parametrize("capacity", [0.0, -0.25])
def test_positive_successor_theorem_requires_positive_input_capacity(capacity):
    with pytest.raises(ValueError, match="positive initial capacity"):
        _image(capacity=capacity)


def test_positive_source_capacity_underflow_is_an_explicit_successor_obstruction():
    with pytest.raises(ValueError, match="positive represented successor capacity"):
        _image(capacity=math.ulp(0.0))


@pytest.mark.parametrize(
    "overrides",
    [
        {"factor": math.ulp(0.0)},
        {"pressure_bounds": (0.0, 1.0, sys.float_info.max)},
        {"epi_min": 1.0, "epi_max": -1.0},
    ],
)
def test_shared_reciprocal_overflow_pressure_overflow_and_rails_reject(overrides):
    with pytest.raises(TNFRValueError):
        _image(**overrides)


def test_ieee_assumption_is_checked_before_any_proposal(monkeypatch):
    monkeypatch.setattr(nul_reset, "uses_ieee_binary64_rounding", lambda: False)

    def forbidden(**kwargs):
        pytest.fail("proposal called outside the IEEE rounding domain")

    monkeypatch.setattr(nul_reset, "propose_scale_operator", forbidden)
    with pytest.raises(ValueError, match="IEEE binary64"):
        _image()

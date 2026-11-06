"""Independent work/source and strict observation controls without coefficients."""

from fractions import Fraction as Q

import pytest

from tnfr.research.sine_class_collective_protocol import (
    assess_collective_prediction,
    collective_prediction_admission,
    collective_prediction_inputs,
)


def test_changed_port_has_its_own_work_and_mean():
    source = collective_prediction_inputs(2)
    q = source["port_impulse"][1]
    eps = source["initial_form_bounds"][0][1]
    # Independent graph count: mediator has two ring and two contact neighbors.
    degree, total_mass = 2 + 2, 3 * 9 * 2 + 2 * 2
    pressure_row_norm = 2 * degree
    expected = (
        degree * q**2 / 2 - pressure_row_norm * q * eps,
        degree * q**2 / 2 + pressure_row_norm * q * eps,
    )
    admission = collective_prediction_admission()
    assert admission["event_work_bounds"] == expected
    assert admission["form_mean_increment"] == degree * q / total_mass
    assert admission["form_mean_increment"] != 3 * q / total_mass
    assert expected[1] < Q(1, 10**6)
    assert (
        admission["post_event_excess_storage_upper_bound"] == 22 * eps**2 + expected[1]
    )
    assert admission["identity_admitted"] and admission["work_admitted"]


def test_apriori_prediction_precedes_numerical_response():
    admission = collective_prediction_admission()
    assert admission["nominal_gap_strict_lower_bound"] > Q(1, 40000)
    assert admission["conservative_recorded_gap_strict_lower_bound"] > Q(1, 40000)
    assert admission["linear_source_allowance_per_model"] > Q(1, 10**32)


@pytest.mark.parametrize("bad", (True, 1.0, 0, 3, "1"))
def test_class_admission_is_not_python_equality(bad):
    with pytest.raises(ValueError):
        collective_prediction_inputs(bad)


def _assess(**changes):
    return assess_collective_prediction(
        **(
            dict(
                full_model_bounds=(Q(2, 10**8), Q(1)),
                comparator_bounds=(Q(-1), Q(0)),
                full_numerical_radius=Q(1, 10**12),
                comparator_numerical_radius=Q(0),
            )
            | changes
        )
    )


@pytest.mark.parametrize("side", (-1, 0, 1))
def test_each_model_contributes_its_own_reading_error(side):
    step = Q(1, 2**600)
    report = _assess(full_model_bounds=(Q(2, 10**8) + side * step, Q(1)))
    assert report["separation_margin"] == side * step
    assert report["prediction_separates"] is (side > 0)
    assert report["numerical_budgets_met"]


def test_wide_numerics_do_not_certify_an_apparent_separation():
    report = _assess(
        full_model_bounds=(Q(1), Q(2)),
        comparator_numerical_radius=Q(1, 10**12) + Q(1, 2**600),
    )
    assert report["separation_margin"] > 0
    assert not report["numerical_budgets_met"]
    assert report["status"] == "bounds_only"


@pytest.mark.parametrize(
    "changes",
    (
        {"full_model_bounds": (True, 1)},
        {"comparator_bounds": (0.0, 1)},
        {"full_model_bounds": (2, 1)},
        {"full_numerical_radius": True},
        {"comparator_numerical_radius": float("inf")},
        {"full_numerical_radius": -1},
    ),
)
def test_supplied_enclosures_retain_primitive_admission(changes):
    with pytest.raises(ValueError):
        _assess(**changes)

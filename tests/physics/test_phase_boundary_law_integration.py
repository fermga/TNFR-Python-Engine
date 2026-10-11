"""Match generic boundary information to the existing complete-law kernels.

These are exact static algebra controls, not a record or source evaluation.
The law families retain their separate preparation and composition premises.
"""

from fractions import Fraction as Q

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.physics._composition_compatible_constitutive import (
    _derive_composition_compatible_law,
    _observe_composition_compatible_edge,
)
from tnfr.physics._phase_boundary_information import (
    _derive_boundary_information,
    _evaluate_boundary_response,
)
from tnfr.physics._sine_class_constitutive_comparison import _edge_law


PHASORS = (
    (Q(3, 5), Q(4, 5)),
    (Q(-3, 5), Q(4, 5)),
    (Q(5, 13), Q(-12, 13)),
    (Q(0), Q(-1)),
    (Q(1), Q(0)),
)
QUERIES = ((1, 0), (0, 1), (Q(12, 13), Q(5, 13)))


@pytest.fixture(scope="module", autouse=True)
def no_response_or_source_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _relative_to_query(phase, query):
    cosine, sine = phase
    c, s = query
    return cosine * c + sine * s, sine * c - cosine * s


@pytest.mark.parametrize("deformation", (Q(0), Q(1, 2), Q(1)))
@pytest.mark.parametrize("query", QUERIES)
def test_boundary_decoder_matches_the_owned_ninth_harmonic_current(deformation, query):
    # Expand the derivative of the owned factored potential. The independent
    # comparison below uses its Chebyshev/factored evaluator, not this list.
    coefficients = ((1, Q(1)), (9, deformation / 2187), (18, -deformation / 4374))
    report = _evaluate_boundary_response(
        coefficients, gap_phasors=PHASORS, query_phasor=query
    )
    expected = sum(
        (
            _edge_law(*_relative_to_query(phase, query), deformation)[1][0]
            for phase in PHASORS
        ),
        Q(0),
    )
    assert report.current == expected
    assert report.degree == len(PHASORS)
    information = _derive_boundary_information(coefficients)
    assert information.minimum_regular_dimension == (6 if deformation else 2)


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(0), Q(1, 1000), Q(1, 324)))
@pytest.mark.parametrize("query", QUERIES)
def test_composition_control_uses_its_owned_coefficients_and_current(
    deformation, query
):
    law = _derive_composition_compatible_law(deformation=deformation)
    report = _evaluate_boundary_response(
        law.current_coefficients, gap_phasors=PHASORS, query_phasor=query
    )
    expected = sum(
        (
            _observe_composition_compatible_edge(
                deformation=deformation,
                relative_phase_phasor=_relative_to_query(phase, query),
            ).current_derivatives[0]
            for phase in PHASORS
        ),
        Q(0),
    )
    assert report.current == expected
    assert law.potential_relative_lower_bound > 0
    information = _derive_boundary_information(law.current_coefficients)
    assert information.minimum_regular_dimension == (10 if deformation else 2)


def test_a_single_current_reading_does_not_answer_another_registration():
    sine = ((1, 1),)
    left, right = ((1, 0),), ((-1, 0),)
    initial = tuple(
        _evaluate_boundary_response(sine, gap_phasors=gaps, query_phasor=(1, 0))
        for gaps in (left, right)
    )
    shifted = tuple(
        _evaluate_boundary_response(sine, gap_phasors=gaps, query_phasor=(0, 1))
        for gaps in (left, right)
    )
    assert initial[0].current == initial[1].current == 0
    assert shifted[0].current == -1
    assert shifted[1].current == 1


def test_exact_information_dimension_does_not_give_a_finite_noise_margin():
    # The added current is uniformly <= 3*lambda/4374 per incidence.
    # Arbitrarily small nonzero coefficients retain exact rank, not resolution.
    deformation = Q(1, 10**400)
    coefficients = ((1, 1), (9, deformation / 2187), (18, -deformation / 4374))
    information = _derive_boundary_information(coefficients)
    assert information.minimum_regular_dimension == 6
    high_current_bound = deformation / 1458
    assert 0 < high_current_bound < Q(1, 10**300)
    alternative = _evaluate_boundary_response(
        coefficients, gap_phasors=PHASORS, query_phasor=QUERIES[-1]
    )
    sine = _evaluate_boundary_response(
        ((1, 1),), gap_phasors=PHASORS, query_phasor=QUERIES[-1]
    )
    assert abs(alternative.current - sine.current) <= len(PHASORS) * high_current_bound

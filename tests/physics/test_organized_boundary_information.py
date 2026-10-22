"""Static contact filtering on an ideal C5 host, without source transfer.

These controls use the actual marked 45-edge support and owned edge laws.
No supplied ideal state is asserted acquired or retained for positive time.
"""

from fractions import Fraction as Q

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _organized_external_registration as registration
from tnfr.physics._composition_compatible_constitutive import (
    _derive_composition_compatible_law,
    _edge_potential_current_bounds,
)
from tnfr.physics._phase_boundary_information import _evaluate_boundary_response
from tnfr.physics._sine_class_constitutive_comparison import _edge_law


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    def forbidden(*args, **kwargs):
        pytest.fail("static contact control attempted source evaluation")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(external, "_assess_organized_external_interaction", forbidden)
        patch.setattr(registration, "_assess_organized_external_interaction", forbidden)
        patch.setattr(
            registration, "_assess_organized_external_registration", forbidden
        )
        yield


def _marked_contact_mean(current, relative_origin, *, middle_child_shift=Q(0)):
    """Sum the owned edges from full ideal lifts, retaining same-slot identity."""
    geometry = external._derive_organized_external_interaction()
    common_origin = Q(2, 7)
    phases = tuple(
        geometry.alpha_bounds * (node % 9 - 4)
        + common_origin
        + (
            2 * pi_interval() * Q(node // 9 - 2, 5)
            + (middle_child_shift if node // 9 == 2 else 0)
            if node < 45
            else relative_origin
        )
        for node in geometry.node_order
    )
    assert len(geometry.contact_edges) == 45
    assert all(a % 9 == b % 9 for a, b, _ in geometry.contact_edges)
    return (
        sum(
            (
                weight * current(phases[b] - phases[a])
                for a, b, weight in geometry.contact_edges
            ),
            I(0),
        )
        / 9
    )


@pytest.mark.parametrize("deformation", (Q(0), Q(1)))
@pytest.mark.parametrize("relative_origin", (Q(0), Q(1, 7), Q(-2, 3)))
def test_ideal_c5_cancels_sine_and_ninth_harmonic_family(deformation, relative_origin):
    response = _marked_contact_mean(
        lambda gap: _edge_law(cos(gap), sin(gap), deformation)[1][0],
        relative_origin,
    )
    assert response.contains(0)
    assert response.width < Q(1, 10**25)


@pytest.mark.parametrize("deformation", (Q(-1, 324), Q(1, 1000), Q(1, 324)))
@pytest.mark.parametrize("sign", (-1, 1))
def test_composition_family_has_an_exact_nonzero_tenth_harmonic_contact(
    deformation, sign
):
    law = _derive_composition_compatible_law(deformation=deformation)
    response = _marked_contact_mean(
        lambda gap: _edge_potential_current_bounds(gap, law)[1],
        sign * pi_interval() / 20,
    )
    # C5 kills orders 1, 8, 17 and 19. At this registration the remaining
    # sin(10*beta) is exactly +/-1; five children give 5*10*deformation.
    expected = sign * 50 * deformation
    assert response.contains(expected)
    assert response.width < Q(1, 10**25)
    assert not response.contains(0)


def test_a_perturbed_child_breaks_the_ideal_sine_cancellation():
    shift = Q(1, 100)
    response = _marked_contact_mean(sin, Q(0), middle_child_shift=shift)
    # Only the middle child's nine equal contacts change, from 0 to -sin(h).
    residual = response + sin(I(shift))
    assert residual.contains(0)
    assert residual.width < Q(1, 10**25)
    assert response.hi < -Q(9, 1000)


def test_rational_boundary_reader_retains_marked_contact_associations():
    geometry = external._derive_organized_external_interaction()
    choices = ((Q(1), Q(0)), (Q(0), Q(1)), (Q(-1), Q(0)), (Q(3, 5), Q(4, 5)))
    host = tuple(choices[(node % 9 + node // 9) % 4] for node in range(45))
    donor = tuple(choices[node % 3] for node in range(9))

    def read(donor_order):
        phases = host + donor_order
        gaps = tuple(
            (ca * cb + sa * sb, ca * sb - sa * cb)
            for a, b, _ in geometry.contact_edges
            for (ca, sa), (cb, sb) in ((phases[a], phases[b]),)
        )
        result = _evaluate_boundary_response(
            ((1, 1),), gap_phasors=gaps, query_phasor=(1, 0)
        )
        independent = sum(
            (
                phases[a][0] * phases[b][1] - phases[a][1] * phases[b][0]
                for a, b, _ in geometry.contact_edges
            ),
            Q(0),
        )
        assert result.degree == 45
        assert result.current == independent
        return result.current / 9

    # Marginal phase lists agree; reversing the donor changes marked matches.
    assert read(donor) != read(tuple(reversed(donor)))

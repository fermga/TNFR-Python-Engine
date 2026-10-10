"""Detached slot-permuted contacts with the full reciprocal sine law.

The host keeps its C9-by-C5 support and the donor keeps its internal C9.
Each host child has one declared cyclic slot shift at its nine donor edges.
This owner supplies static geometry and fields, not source formation,
support-event selection, a finite response, or indefinite identity retention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I
from ._organized_external_interaction import (
    _OrganizedExternalInteractionGeometry,
    _derive_organized_external_interaction,
    _observe_admitted_external_interaction,
)
from ._organized_external_registration import (
    _OrganizedExternalRegistrationObservation,
    _quadrature_norm,
)
from ._repeated_organized_composition import _bounded_sine_edge_fields
from ._sine_preparation import _prepared_coordinate_vector as _vector
from .relational_observations import _ordered


def _admit_slot_offsets(slot_offsets):
    """Admit five literal discrete labels without modular coercion."""
    slots = _ordered(slot_offsets, "slot_offsets", limit=6)
    if len(slots) != 5:
        raise ValueError("slot_offsets requires five ordered entries")
    if any(type(slot) is not int or slot not in range(9) for slot in slots):
        raise ValueError("slot_offsets entries require ordinary integers in 0..8")
    return slots


@dataclass(frozen=True)
class _OrganizedSlotRegistrationGeometry(_OrganizedExternalInteractionGeometry):
    slot_offsets: tuple[int, ...] = (0,) * 5


def _derive_organized_slot_registration(*, slot_offsets):
    """Build the fixed internal supports and the supplied cyclic matchings."""
    slots = _admit_slot_offsets(slot_offsets)
    base = _derive_organized_external_interaction()
    contact = tuple(
        (9 * child + slot, 45 + (slot + shift) % 9, Q(1))
        for child, shift in enumerate(slots)
        for slot in range(9)
    )
    return _OrganizedSlotRegistrationGeometry(
        **(
            vars(base)
            | dict(
                contact_edges=contact,
                weighted_edges=base.internal_edges + base.outer_edges + contact,
                slot_offsets=slots,
                scope=(
                    "fixed_host_C9_by_C5_and_external_C9_with_cyclic_slot_matchings",
                    "nine_unit_contact_edges_per_child_preserve_both_support_degrees",
                    "all_fifty_four_forms_and_phases_retained_off_the_carrier",
                    "external_pi_over_two_origin_supplied_from_primitive_preparation",
                    "geometry_installs_no_support_event_or_occurrence_law",
                ),
            )
        )
    )


def _slot_moments_from_admitted_offsets(offsets, geometry):
    """Return beta-free mean exp(i*(theta_D-theta_H-beta)) components.

    The carrier contribution on j -> (j+k) mod 9 is +alpha*k modulo
    a full turn. The opposite host-minus-donor convention is its conjugate.
    The shared edge kernel retains actual wrapped slot indices and all
    fine correlations, rather than substituting the carrier formula.
    """
    _, potential, currents = _bounded_sine_edge_fields(
        phase_offsets=offsets,
        weighted_edges=geometry.contact_edges,
        alpha_bounds=geometry.alpha_bounds,
    )
    return (45 - sum(potential, I(0))) / 9, sum(currents, I(0)) / 9


def _bounded_slot_registration_moments(*, slot_offsets, phase_offsets):
    """Enclose beta-free contact quadratures from primitive phase boxes.

    Exactly 54 offsets are relative to alpha*(j-4); donor offsets exclude
    its supplied common origin. Bounds may be shared immutable intervals
    or admitted real scalars. No incoming geometry or source verdict is
    consumed; a caller must justify inclusion of its actual source.
    """
    slots = _admit_slot_offsets(slot_offsets)
    raw = _ordered(phase_offsets, "phase_offsets", limit=55)
    if len(raw) != 54:
        raise ValueError("phase_offsets requires 54 ordered entries")
    offsets = tuple(
        value if type(value) is I else _exact(value, f"phase_offsets[{index}]")
        for index, value in enumerate(raw)
    )
    geometry = _derive_organized_slot_registration(slot_offsets=slots)
    return _slot_moments_from_admitted_offsets(offsets, geometry)


def _observe_organized_slot_registration(
    *, slot_offsets, relative_phase_origin, form, phase_offsets
):
    """Observe a complete detached state after all primitive admission."""
    slots = _admit_slot_offsets(slot_offsets)
    beta = _exact(relative_phase_origin, "relative_phase_origin")
    values = _vector(form, "form", 54)
    offsets = _vector(phase_offsets, "phase_offsets", 54)
    base = _derive_organized_slot_registration(slot_offsets=slots)
    geometry = replace(
        base,
        external_phase_origin_bounds=I(beta),
        scope=tuple(
            (
                "external_origin_supplied_from_primitive_preparation"
                if item
                == "external_pi_over_two_origin_supplied_from_primitive_preparation"
                else item
            )
            for item in base.scope
        ),
    )
    interaction = _observe_admitted_external_interaction(values, offsets, geometry)
    interaction = replace(
        interaction,
        scope=tuple(
            (
                "raw_rational_offsets_relative_to_carrier_and_declared_external_origin"
                if item
                == "raw_rational_offsets_relative_to_carrier_and_symbolic_external_pi_over_two"
                else item
            )
            for item in interaction.scope
        ),
    )
    cosine, sine = _slot_moments_from_admitted_offsets(offsets, geometry)
    return _OrganizedExternalRegistrationObservation(
        relative_phase_origin=beta,
        interaction=interaction,
        boundary_cosine_bounds=cosine,
        boundary_sine_bounds=sine,
        boundary_amplitude_bounds=_quadrature_norm(cosine, sine),
    )

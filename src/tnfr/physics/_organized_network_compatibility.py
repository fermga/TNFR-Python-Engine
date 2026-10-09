"""Detached complete laws for an oriented triangle of registered C9 objects.

ORGANIZED_NETWORK_COMPATIBILITY.md owns this fixed 27-node weighted model.
Shared contact coefficients do not carry a pair's degree, clock, source or
identity certificate into a network. Chart verdicts use exact primitive
conditions, never rounded cycle angles or overlapping magnitude intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ._oriented_interface_constitutive import (
    _derive_oriented_contact,
    _mixing,
    _OrientedContactFactors,
)
from .phase_response import _admit_relative_phasors
from .relational_observations import _ordered

_CONTACTS = ((0, 1), (1, 2), (2, 0))
_INTERNAL = tuple(
    (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(3) for j in range(9)
)


def _mixings(values):
    raw = _ordered(values, "mixing", limit=4)
    if len(raw) != 3:
        raise ValueError("mixing requires three weights in contact order 01,12,20")
    return tuple(_mixing(value) for value in raw)


@dataclass(frozen=True)
class _OrganizedTriangle:
    """Actual support, retained boundary data and scoped global chart verdicts.

    A common amplitude chart may exist while phase compatibility fails. The
    comparison permits positive nodewise affine form charts, constant phase
    origins and one common clock/reciprocal coefficient. It does not test all
    nonlinear conjugacies or provide a source, equilibrium or stability proof.
    """

    mixing: tuple[Q, Q, Q]
    contacts: tuple[_OrientedContactFactors, ...]
    weighted_edges: tuple[tuple[int, int, Q], ...]
    weighted_degrees: tuple[Q, ...]
    support_degrees: tuple[int, ...]
    gamma_bounds: I
    internal_storage_bounds: I
    cycle_phase_bounds: I
    cycle_phasor_bounds: tuple[I, I]
    tree_phase_origins_bounds: tuple[I, I, I]
    phase_compatible: bool
    amplitude_compatible: bool
    common_network_chart: bool
    common_form_scale_bounds: I | None
    common_gamma_bounds: I | None
    aligned_attachment_work_bounds: I
    clock_factor: Q = Q(1, 4)
    nodes: tuple[int, ...] = tuple(range(27))
    contact_order: tuple[tuple[int, int], ...] = _CONTACTS
    clock: str = "tau=(1023/1024)*t; sigma=tau/4"
    scope: tuple[str, ...] = (
        "three_registered_winding_one_C9s_with_supplied_cyclic_contacts",
        "held_unit_capacity_and_actual_weighted_degree_four",
        "sine_weighted_diffusion_current_phase_row_and_storage",
        "exact_six_coordinate_invariant_restriction_not_generic_quotient",
        "cycle_phase_and_magnitudes_retained_as_boundary_data",
        "chart_obstruction_does_not_imply_loss_of_identity",
        "no_transferred_pair_acquisition_or_contact_certificate",
    )


def _derive_organized_triangle(*, mixing):
    """Rebuild fixed support and contact data from three admitted weights."""
    values = _mixings(mixing)
    contacts = tuple(_derive_oriented_contact(mixing=s) for s in values)
    cross = tuple(
        (9 * a + j, 9 * b + (j + offset) % 9, weight)
        for (a, b), s in zip(_CONTACTS, values)
        for j in range(9)
        for offset, weight in ((0, 1 - s), (1, s))
        if weight > 0
    )
    edges = _INTERNAL + cross
    degrees = tuple(
        sum((w for a, b, w in edges if node in (a, b)), Q(0)) for node in range(27)
    )
    neighbors = tuple(sum(node in (a, b) for a, b, _ in edges) for node in range(27))
    phases = tuple(contact.phase_origin_shift_bounds for contact in contacts)
    real, imaginary = I(1), I(0)
    for contact in contacts:
        c, d = contact.phase_origin_phasor_bounds
        real, imaginary = real * c - imaginary * d, imaginary * c + real * d
    phase_ok = all(s == 0 for s in values)
    amplitude_keys = tuple(s * (1 - s) for s in values)
    amplitude_ok = all(value == amplitude_keys[0] for value in amplitude_keys)
    scale = contacts[0].form_scale_bounds if amplitude_ok else None
    gamma = Q(1, 1023) / pi_interval()
    cosine = contacts[0].alpha_cosine_bounds
    return _OrganizedTriangle(
        mixing=values,
        contacts=contacts,
        weighted_edges=edges,
        weighted_degrees=degrees,
        support_degrees=neighbors,
        gamma_bounds=gamma,
        internal_storage_bounds=27 * (1 - cosine),
        cycle_phase_bounds=sum(phases, I(0)),
        cycle_phasor_bounds=(real, imaginary),
        tree_phase_origins_bounds=(I(0), phases[0], phases[0] + phases[1]),
        phase_compatible=phase_ok,
        amplitude_compatible=amplitude_ok,
        common_network_chart=phase_ok and amplitude_ok,
        common_form_scale_bounds=scale,
        common_gamma_bounds=gamma * scale if scale is not None else None,
        aligned_attachment_work_bounds=9 * sum(values, Q(0)) * (1 - cosine),
    )


@dataclass(frozen=True)
class _OrganizedTriangleObservation:
    """Static complete collective state; all edge phases share one assignment."""

    geometry: _OrganizedTriangle
    form: tuple[Q, Q, Q]
    phase_phasors: tuple[tuple[Q, Q], ...]
    relative_phase_phasors: tuple[tuple[Q, Q], ...]
    form_gradient: tuple[Q, ...]
    edge_current_bounds: tuple[I, ...]
    phase_current_sums_bounds: tuple[I, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    edge_potential_bounds: tuple[I, ...]
    form_storage: Q
    phase_storage_bounds: I
    full_storage_bounds: I
    attachment_work_bounds: I
    loss: Q
    storage_rate: Q
    storage_rate_from_rows_bounds: I
    tree_phase_phasors_bounds: tuple[tuple[I, I], ...]
    collective_form_rate_bounds: tuple[I, ...]
    collective_phase_rate_bounds: tuple[I, ...]


def _observe_organized_triangle(*, mixing, form, phase_phasors):
    """Admit joint primitive coordinates before any graph or interval work."""
    weights = _mixings(mixing)
    raw_form = _ordered(form, "form", limit=4)
    raw_phase = _ordered(phase_phasors, "phase_phasors", limit=4)
    if len(raw_form) != 3 or len(raw_phase) != 3:
        raise ValueError(
            "three component forms and absolute phase phasors are required"
        )
    values = tuple(_exact(value, f"form[{r}]") for r, value in enumerate(raw_form))
    phases = _admit_relative_phasors(raw_phase, "phase_phasors")
    geometry = _derive_organized_triangle(mixing=weights)
    relatives, currents, potentials = [], [], []
    gradient, current_sums = [Q(0)] * 3, [I(0)] * 3
    form_energy = Q(0)
    for (a, b), contact in zip(_CONTACTS, geometry.contacts):
        ca, sa = phases[a]
        cb, sb = phases[b]
        cp, sp = ca * cb + sa * sb, ca * sb - sa * cb
        c, d = contact.real_multiplier_bounds, contact.imaginary_multiplier_bounds
        current, potential = c * sp + d * cp, 1 - c * cp + d * sp
        u = values[b] - values[a]
        gradient[a] -= u
        gradient[b] += u
        current_sums[a] = current_sums[a] + current
        current_sums[b] = current_sums[b] - current
        form_energy += Q(9, 2) * u**2
        relatives.append((cp, sp))
        currents.append(current)
        potentials.append(potential)
    gamma, k = geometry.gamma_bounds, geometry.clock_factor
    dx = tuple(k * (-q + gamma * s) for q, s in zip(gradient, current_sums))
    dphase = tuple(k * gamma * q for q in gradient)
    loss = Q(9, 4) * sum((q**2 for q in gradient), Q(0))
    rate = sum(
        (
            9 * q * x - 9 * s * theta
            for q, s, x, theta in zip(gradient, current_sums, dx, dphase)
        ),
        I(0),
    )
    contact_energy = 9 * sum(potentials, I(0))
    phase_energy = geometry.internal_storage_bounds + contact_energy
    tree_phases = []
    for (cp, sp), beta in zip(phases, geometry.tree_phase_origins_bounds):
        cb, sb = cos(beta), sin(beta)
        tree_phases.append((cp * cb - sp * sb, sp * cb + cp * sb))
    return _OrganizedTriangleObservation(
        geometry=geometry,
        form=values,
        phase_phasors=phases,
        relative_phase_phasors=tuple(relatives),
        form_gradient=tuple(gradient),
        edge_current_bounds=tuple(currents),
        phase_current_sums_bounds=tuple(current_sums),
        form_rate_bounds=dx,
        phase_rate_bounds=dphase,
        edge_potential_bounds=tuple(potentials),
        form_storage=form_energy,
        phase_storage_bounds=phase_energy,
        full_storage_bounds=form_energy + phase_energy,
        attachment_work_bounds=form_energy + contact_energy,
        loss=loss,
        storage_rate=-loss,
        storage_rate_from_rows_bounds=rate,
        tree_phase_phasors_bounds=tuple(tree_phases),
        collective_form_rate_bounds=tuple(value / k for value in dx),
        collective_phase_rate_bounds=tuple(value / k for value in dphase),
    )

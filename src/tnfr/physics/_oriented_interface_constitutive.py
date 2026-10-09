"""Detached weighted C9 interface laws and conditional profile inheritance.

ORIENTED_INTERFACE_PROFILE_INHERITANCE.md owns the complete weighted model.
Weights enter both evolution rows and storage, with held unit capacities and
weighted degree three. This is not the native phase-neighbor convention.
Every observation rebuilds its primitive law and geometry; no source report,
trajectory, event selector or universal NFR profile requirement is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, arg, cos, pi_interval, sin, sqrt
from ._composition_compatible_constitutive import (
    _amplitude,
    _CompositionCompatibleLaw,
    _derive_composition_compatible_law,
    _registered_edge_components,
)
from ._sine_constitutive_inheritance import _INTERNAL
from .phase_response import _admit_relative_phasors
from .relational_observations import _ordered


def _mixing(value):
    mixing = _exact(value, "mixing")
    if not 0 <= mixing <= 1:
        raise ValueError("mixing must be in [0,1]; endpoints are separate controls")
    return mixing


@dataclass(frozen=True)
class _OrientedInterface:
    """Fixed oriented contact geometry; zero weights are absent support."""

    mixing: Q
    weighted_edges: tuple[tuple[int, int, Q], ...]
    weighted_degrees: tuple[Q, ...]
    support_degrees: tuple[int, ...]
    real_multiplier_bounds: I
    imaginary_multiplier_bounds: I
    amplitude_bounds: I
    phase_origin_shift_bounds: I
    phase_origin_phasor_bounds: tuple[I, I]
    form_scale_bounds: I
    gamma_bounds: I
    effective_gamma_bounds: I
    internal_storage_bounds: I
    aligned_attachment_work_bounds: I
    clock_factor: Q = Q(1, 3)
    nodes: tuple[int, ...] = tuple(range(18))
    clock: str = "tau=(1023/1024)*t; sigma=tau/3"
    scope: tuple[str, ...] = (
        "undirected_weighted_edges_with_marked_component_orientation",
        "internal_cycle_weights_one_cross_weights_1_minus_s_and_s",
        "weighted_degree_three_differs_from_unique_neighbor_count",
        "weights_consumed_by_diffusion_current_and_storage",
        "fixed_mixing_parameter_not_a_time_varying_input_law",
        "phase_chart_keeps_marked_port_registration",
        "no_support_occurrence_or_acquired_state_certificate",
    )


def _derive_oriented_interface(*, mixing):
    """Derive actual support and enclosed geometry from the exact weight."""
    s = _mixing(mixing)
    cross = tuple(
        (j, 9 + (j + offset) % 9, weight)
        for j in range(9)
        for offset, weight in ((0, 1 - s), (1, s))
        if weight > 0
    )
    edges = tuple((i, j, Q(1)) for i, j in _INTERNAL) + cross
    degrees = tuple(
        sum((w for i, j, w in edges if node in (i, j)), Q(0)) for node in range(18)
    )
    neighbors = tuple(sum(node in (i, j) for i, j, _ in edges) for node in range(18))
    pi = pi_interval()
    alpha = 2 * pi / 9
    cosine, sine = cos(alpha), sin(alpha)
    real, imaginary = 1 - s + s * cosine, s * sine
    # This form keeps the known unit modulus exact at both endpoint controls.
    amplitude = sqrt(1 - 2 * s * (1 - s) * (1 - cosine))
    beta = I(0) if s == 0 else alpha if s == 1 else arg(real, imaginary)
    scale, gamma = sqrt(amplitude), Q(1, 1023) / pi
    return _OrientedInterface(
        mixing=s,
        weighted_edges=edges,
        weighted_degrees=degrees,
        support_degrees=neighbors,
        real_multiplier_bounds=real,
        imaginary_multiplier_bounds=imaginary,
        amplitude_bounds=amplitude,
        phase_origin_shift_bounds=beta,
        phase_origin_phasor_bounds=(real / amplitude, imaginary / amplitude),
        form_scale_bounds=scale,
        gamma_bounds=gamma,
        effective_gamma_bounds=gamma * scale,
        internal_storage_bounds=18 * (1 - cosine),
        aligned_attachment_work_bounds=9 * s * (1 - cosine),
    )


@dataclass(frozen=True)
class _OrientedConstitutivePair:
    """Complete static restriction, with oriented boundary data retained.

    The chart uses the first-harmonic contact angle and a positive form scale
    for every admitted member. It does not restore a nonsine profile. Its
    contact potential is U/rho, including the full additive baseline, and
    need not vanish at chart angle zero. Interval coefficients are enclosures;
    exact profile verdicts follow the theorem, never a numerical tolerance.
    """

    law: _CompositionCompatibleLaw
    geometry: _OrientedInterface
    form: tuple[Q, Q]
    relative_phase_phasor: tuple[Q, Q]
    current_bounds: I
    contact_potential_bounds: I
    form_rate_bounds: tuple[I, I]
    phase_rate_bounds: tuple[I, I]
    chart_form_bounds: tuple[I, I]
    chart_relative_phase_phasor_bounds: tuple[I, I]
    chart_form_rate_bounds: tuple[I, I]
    chart_phase_rate_bounds: tuple[I, I]
    chart_current_bounds: I
    chart_contact_potential_bounds: I
    inherited_harmonic_coefficients: tuple[tuple[int, I, I], ...]
    full_storage_bounds: I
    attachment_work_bounds: I
    loss: Q
    storage_rate: Q
    storage_rate_from_rows_bounds: I
    profile_preserved: bool
    profile_preserved_over_open_interfaces: bool


def _observe_oriented_constitutive_pair(
    *, mixing, deformation, form, relative_phase_phasor
):
    """Rebuild the weighted law after admitting every primitive coordinate."""
    s, e = _mixing(mixing), _amplitude(deformation)
    raw = _ordered(form, "form", limit=3)
    if len(raw) != 2:
        raise ValueError("form requires two registered component forms")
    values = tuple(_exact(value, f"form[{i}]") for i, value in enumerate(raw))
    phase = _admit_relative_phasors((relative_phase_phasor,), "relative_phase_phasor")[
        0
    ]
    law = _derive_composition_compatible_law(deformation=e)
    geometry = _derive_oriented_interface(mixing=s)
    potential, jet, companion, primitive = _registered_edge_components(phase, law)
    c, d = geometry.real_multiplier_bounds, geometry.imaginary_multiplier_bounds
    rho, scale, gamma = (
        geometry.amplitude_bounds,
        geometry.form_scale_bounds,
        geometry.gamma_bounds,
    )
    current = c * jet[0] + d * companion
    contact_potential = 1 - c + c * potential + d * primitive
    u, k = values[1] - values[0], geometry.clock_factor
    dx, dphase = k * (u + gamma * current), -k * gamma * u
    attachment = 9 * (Q(1, 2) * u**2 + contact_potential)
    loss = 6 * u**2
    beta = geometry.phase_origin_shift_bounds
    coefficients = []
    for n, b in law.current_coefficients:
        sign = 1 if n % 9 == 1 else -1
        cb, sb = cos(n * beta), sin(n * beta)
        coefficients.append(
            (n, b * (c * cb + sign * d * sb) / rho, b * (-c * sb + sign * d * cb) / rho)
        )
    cp, sp = phase
    return _OrientedConstitutivePair(
        law=law,
        geometry=geometry,
        form=values,
        relative_phase_phasor=phase,
        current_bounds=current,
        contact_potential_bounds=contact_potential,
        form_rate_bounds=(dx, -dx),
        phase_rate_bounds=(dphase, -dphase),
        chart_form_bounds=tuple(I(value) / scale for value in values),
        chart_relative_phase_phasor_bounds=(
            (cp * c - sp * d) / rho,
            (sp * c + cp * d) / rho,
        ),
        chart_form_rate_bounds=(dx / (k * scale), -dx / (k * scale)),
        chart_phase_rate_bounds=(dphase / k, -dphase / k),
        chart_current_bounds=current / rho,
        chart_contact_potential_bounds=contact_potential / rho,
        inherited_harmonic_coefficients=tuple(coefficients),
        full_storage_bounds=geometry.internal_storage_bounds + attachment,
        attachment_work_bounds=attachment,
        loss=loss,
        storage_rate=-loss,
        storage_rate_from_rows_bounds=-18 * u * dx - 18 * current * dphase,
        profile_preserved=e == 0 or s in (0, 1),
        profile_preserved_over_open_interfaces=e == 0,
    )

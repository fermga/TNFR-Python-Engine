"""Static analytic counterfamily for profile-preserving C9 composition.

The proof owner is COMPOSITION_COMPATIBLE_CONSTITUTIVE_LAWS.md. This family
preserves the two named interfaces and the sine target jets, but changes the
complete nonlinear current. No source, capture, trajectory or event selector
is inferred. The admitted amplitude is a sufficient nonnegative-storage domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I
from ._sine_constitutive_inheritance import (
    _ConstitutiveInheritance,
    _derive_constitutive_inheritance,
    _interface,
)
from .phase_response import _admit_relative_phasors
from .relational_observations import _ordered


def _amplitude(value):
    coefficient = _exact(value, "deformation")
    if abs(coefficient) > Q(1, 324):
        raise ValueError("deformation requires the proved interval [-1/324,1/324]")
    return coefficient


@dataclass(frozen=True)
class _CompositionCompatibleLaw:
    """Exact finite Fourier law; target moments do not certify acquisition.

    Moments are storage, individual edge current, tangent and quadratic field,
    in that order. Edge-current matching is stronger than the symmetric target
    equilibrium requirement, where incident odd currents already cancel. This
    explicit family satisfies all four; the general classification needs only
    the other three moments plus independent potential nonnegativity.
    """

    deformation: Q
    current_coefficients: tuple[tuple[int, Q], ...]
    target_moments: tuple[Q, Q, Q, Q]
    potential_relative_lower_bound: Q
    current_derivative_bounds: tuple[Q, ...]
    scope: tuple[str, ...] = (
        "odd_analytic_periodic_current_with_even_nonnegative_own_potential",
        "V=1-cos(delta)+epsilon*cos(delta)*(1-cos(9*delta))**2",
        "same_profile_for_matching_and_three_offset_C9_interfaces_only",
        "target_storage_current_tangent_and_quadratic_field_match_sine",
        "complete_rows_require_derived_clock_form_and_reciprocal_maps",
        "no_transferred_source_recovery_capture_or_identity_certificate",
    )


def _derive_composition_compatible_law(*, deformation):
    """Admit and expose one member of the proved finite counterfamily."""
    e = _amplitude(deformation)
    coefficients = (
        (1, 1 - Q(3, 2) * e),
        (8, 8 * e),
        (10, 10 * e),
        (17, -Q(17, 4) * e),
        (19, -Q(19, 4) * e),
    )
    signs = tuple(1 if n % 9 == 1 else -1 for n, _ in coefficients)
    moments = (
        sum((b / n for n, b in coefficients), Q(0)),
        sum((s * b for s, (_, b) in zip(signs, coefficients)), Q(0)),
        sum((n * b for n, b in coefficients), Q(0)),
        sum((s * n**2 * b for s, (n, b) in zip(signs, coefficients)), Q(0)),
    )
    bounds = tuple(
        sum((abs(b) * n**order for n, b in coefficients), Q(0)) for order in range(4)
    )
    return _CompositionCompatibleLaw(
        deformation=e,
        current_coefficients=coefficients,
        target_moments=moments,
        potential_relative_lower_bound=1 - 162 * abs(e),
        current_derivative_bounds=bounds,
    )


def _edge_values(phasor, law):
    """Evaluate finite harmonics exactly on an already admitted unit phasor."""
    cosine, sine = phasor
    powers = [(Q(1), Q(0))]
    for _ in range(19):
        c, s = powers[-1]
        powers.append((c * cosine - s * sine, s * cosine + c * sine))
    potential = sum(
        (b * (1 - powers[n][0]) / n for n, b in law.current_coefficients), Q(0)
    )
    jet = tuple(
        sum(
            (
                b
                * n**order
                * (powers[n][1], powers[n][0], -powers[n][1], -powers[n][0])[order]
                for n, b in law.current_coefficients
            ),
            Q(0),
        )
        for order in range(4)
    )
    return potential, jet


@dataclass(frozen=True)
class _CompositionCompatibleEdge:
    """Static edge potential and current jets in its own constitutive law."""

    deformation: Q
    phasor: tuple[Q, Q]
    potential: Q
    current_derivatives: tuple[Q, ...]


def _observe_composition_compatible_edge(*, deformation, relative_phase_phasor):
    """Re-admit the amplitude and circular observation before arithmetic."""
    e = _amplitude(deformation)
    phase = _admit_relative_phasors((relative_phase_phasor,), "relative_phase_phasor")[
        0
    ]
    law = _derive_composition_compatible_law(deformation=e)
    potential, jet = _edge_values(phase, law)
    return _CompositionCompatibleEdge(e, phase, potential, jet)


@dataclass(frozen=True)
class _CompositionCompatiblePair:
    """Both rows on the registered invariant family, with full storage.

    Geometry is the rebuilt zero-deformation sine reference: only its fixed
    support, clock, chart, reciprocal coefficient and target baseline are
    reused. The current and potential come from ``law``, not that reference.
    This is a supplied static family state, not an acquired-object report.
    """

    law: _CompositionCompatibleLaw
    geometry: _ConstitutiveInheritance
    form: tuple[Q, Q]
    relative_phase_phasor: tuple[Q, Q]
    current_bounds: I
    form_rate_bounds: tuple[I, I]
    phase_rate_bounds: tuple[I, I]
    chart_form_bounds: tuple[I, I]
    chart_form_rate_bounds: tuple[I, I]
    chart_phase_rate_bounds: tuple[I, I]
    full_storage_bounds: I
    loss: Q
    storage_rate: Q
    storage_rate_from_rows_bounds: I


def _observe_composition_compatible_pair(
    *, interface, deformation, form, relative_phase_phasor
):
    """Compute the own-law restriction without integrating or attaching nodes."""
    name, e = _interface(interface), _amplitude(deformation)
    raw = _ordered(form, "form", limit=3)
    if len(raw) != 2:
        raise ValueError("form requires two registered component forms")
    values = tuple(_exact(value, f"form[{i}]") for i, value in enumerate(raw))
    phase = _admit_relative_phasors((relative_phase_phasor,), "relative_phase_phasor")[
        0
    ]
    law = _derive_composition_compatible_law(deformation=e)
    geometry = _derive_constitutive_inheritance(interface=name, deformation=Q(0))
    potential, jet = _edge_values(phase, law)
    r, gamma = geometry.first_harmonic_multiplier_bounds, geometry.gamma_bounds
    k, q, scale = (
        geometry.collective_clock_factor,
        geometry.cross_degree,
        geometry.form_chart_scale_bounds,
    )
    u = values[1] - values[0]
    current = r * jet[0]
    dx, dphase = k * (u + gamma * current), -k * gamma * u
    energy = geometry.aligned_target_storage_bounds + 9 * q * (
        Q(1, 2) * u**2 + r * potential
    )
    loss = 18 * q * k * u**2
    rate = -18 * q * u * dx - 18 * q * current * dphase
    return _CompositionCompatiblePair(
        law=law,
        geometry=geometry,
        form=values,
        relative_phase_phasor=phase,
        current_bounds=current,
        form_rate_bounds=(dx, -dx),
        phase_rate_bounds=(dphase, -dphase),
        chart_form_bounds=tuple(I(value) / scale for value in values),
        chart_form_rate_bounds=(dx / (k * scale), -dx / (k * scale)),
        chart_phase_rate_bounds=(dphase / k, -dphase / k),
        full_storage_bounds=energy,
        loss=loss,
        storage_rate=-loss,
        storage_rate_from_rows_bounds=rate,
    )

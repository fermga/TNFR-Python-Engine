"""Detached energy-guard sensitivity at an exact organized carrier crossing.

ORGANIZED_SOURCE_GUARD_SENSITIVITY.md owns the local hitting-time and complete
hybrid-response theorem. This reader reuses the complete carrier timing reader;
it selects neither a guard nor an event and computes no trajectory or hit time.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I
from ._organized_source_timing import (
    _assess_organized_source_carrier_timing,
    _OrganizedSourceCarrierTiming,
)
from ._sine_preparation import _prepared_form_costs


@dataclass(frozen=True)
class _OrganizedSourceCarrierGuardSensitivity:
    """Exact carrier energy jets, crossing covectors and event correction.

    ``energy_loss_rate`` is the positive magnitude ``d=-E_tau=||Lx||^2/4``.
    The source is the supplied exact-carrier crossing state. Its energy is
    reported as ``guard_energy_level`` without selecting a level for any
    other source or certifying a future first crossing.

    Write ``K=(timing.form_commutator,timing.phase_commutator)`` for the
    complete continuous timing difference. The saltation factorization is
    ``S=DW+K outer (event_time_form_covector,event_time_phase_covector)``.
    The fixed-observation-time response to a level change is ``-K/d`` at
    this crossing, before any subsequent variational-flow transport.

    With ``c=cos(2*pi/9)``, the level phase derivative divided by gamma is
    ``level_phase_base+(level_phase_matching+c*level_phase_cycle)/(3+2*c)``.
    The rational coefficients exclude gamma; ``level_phase_bounds`` includes
    it. Capacity and structural-clock event-time corrections are zero.

    For the same supplied point regarded as an initial state, the energy-drop
    inverse has ``tau(eta)=eta/d+drop_time_quadratic_coefficient*eta**2+O(eta**3)``
    as the positive drop eta tends to zero. The comparison level here is
    ``E(initial)-eta``; this coefficient is not a fixed-level initial-state
    derivative or an evaluated future hit. The phase current vanishes at the
    carrier evaluation, while the complete phase rate remains in the law.
    """

    timing: _OrganizedSourceCarrierTiming
    energy_loss_rate: Q
    energy_second_time_derivative: Q
    drop_time_quadratic_coefficient: Q
    guard_energy_level: Q
    event_time_form_covector: tuple[Q, ...]
    event_time_phase_covector: tuple[Q, ...]
    level_time_derivative: Q
    level_form_derivative: tuple[Q, ...]
    level_phase_base: tuple[Q, ...]
    level_phase_matching: tuple[Q, ...]
    level_phase_cycle: tuple[Q, ...]
    level_phase_bounds: tuple[I, ...]
    scope: tuple[str, ...] = (
        "primitive_admission_and_complete_carrier_word_tangent_reused_from_timing",
        "supplied_exact_carrier_crossing_state_or_zero_delay_local_limit",
        "positive_energy_loss_is_required_for_transversality",
        "complete_law_energy_acceleration_uses_initial_carrier_current_cancellation",
        "phase_rate_is_retained_and_phase_is_not_held_fixed",
        "energy_drop_inverse_coefficient_is_local_at_one_fixed_initial_state",
        "guard_level_is_the_supplied_crossing_energy_not_an_occurrence_choice",
        "event_time_covectors_are_with_respect_to_pre_event_continuous_state",
        "full_saltation_is_reset_derivative_plus_factored_timing_correction",
        "both_form_and_phase_level_sensitivities_and_changed_capacities_retained",
        "capacity_and_structural_clock_timing_corrections_are_zero",
        "pre_and_post_flow_variational_transport_are_separate_obligations",
        "fixed_level_first_crossing_urgency_and_grammar_context_are_premises",
        "no_event_hit_forecast_finite_horizon_margin_or_runtime_selector",
    )


def _assess_organized_source_carrier_guard_sensitivity(
    form, *, reception_mix, phase_coefficient, silence_capacity_factor
):
    """Rebuild transverse carrier event-time, level and energy derivatives.

    At the exact winding-one carrier, the energy differential is ``(Lx,0)``
    and the unit-capacity incoming law gives ``E_tau=-||Lx||^2/4``. Thus the
    local event-time state covector is ``(Lx/d,0)``, while its derivative
    with respect to the fixed guard level is ``-1/d``. Multiplication by the
    complete word commutator gives the common-time hybrid corrections.

    Differentiating the complete loss law gives
    ``E_tautau=(q dot Lq-gamma*q dot L*S)/8``. At this initial carrier ``S=0``,
    so the exact second derivative is ``q dot Lq/8`` even though the phase
    rate ``gamma*q/4`` is nonzero. The local inverse-energy-drop coefficient
    is ``E_tautau/(2*d**3)``; no drop or finite observation time is selected.

    All primitive state, factor, support, energy and mean admission is owned
    by the reused timing reader. A zero gradient is rejected rather than
    dividing by zero or inferring a crossing from energy nonincrease alone.
    """
    timing = _assess_organized_source_carrier_timing(
        form,
        reception_mix=reception_mix,
        phase_coefficient=phase_coefficient,
        silence_capacity_factor=silence_capacity_factor,
    )
    gradient = timing.form_gradient
    loss = sum((value**2 for value in gradient), Q(0)) / 4
    if loss == 0:
        raise ValueError(
            "energy guard requires a transverse crossing with positive loss"
        )
    _, gradient_laplacian = _prepared_form_costs(
        form=gradient, weighted_edges=timing.geometry.weighted_edges
    )
    energy_acceleration = (
        sum(
            (
                value * laplacian
                for value, laplacian in zip(gradient, gradient_laplacian)
            ),
            Q(0),
        )
        / 8
    )
    phase_base = tuple(-value / loss for value in timing.phase_commutator_base)
    phase_matching = tuple(-value / loss for value in timing.phase_commutator_matching)
    phase_cycle = tuple(-value / loss for value in timing.phase_commutator_cycle)
    # Divide exact coefficients before interval construction: a positive loss
    # can lie below the interval grid, and old tiny rate bounds can lose sign.
    cosine = timing.carrier_cosine_bounds
    resultant = 3 + 2 * cosine
    phase_bounds = tuple(
        timing.gamma_bounds * (base + (matching + cycle * cosine) / resultant)
        for base, matching, cycle in zip(phase_base, phase_matching, phase_cycle)
    )
    return _OrganizedSourceCarrierGuardSensitivity(
        timing=timing,
        energy_loss_rate=loss,
        energy_second_time_derivative=energy_acceleration,
        drop_time_quadratic_coefficient=energy_acceleration / (2 * loss**3),
        guard_energy_level=timing.energy_excess,
        event_time_form_covector=tuple(value / loss for value in gradient),
        event_time_phase_covector=(Q(0),) * len(gradient),
        level_time_derivative=-1 / loss,
        level_form_derivative=tuple(-value / loss for value in timing.form_commutator),
        level_phase_base=phase_base,
        level_phase_matching=phase_matching,
        level_phase_cycle=phase_cycle,
        level_phase_bounds=phase_bounds,
    )

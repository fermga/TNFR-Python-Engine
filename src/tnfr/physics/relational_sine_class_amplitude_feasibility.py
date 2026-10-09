"""Response-free common-amplitude feasibility under a supplied cubic enclosure.

The base interval is a declared coefficient premise, including its finite-time
arithmetic error. This calculator neither validates it nor consumes a saved
verdict. It combines homogeneity with independently derived error and event
bounds for the fixed source, two impulses, clock, support and observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import I
from ._sine_class_contrast import _contrast_decision, _ContrastDecision
from .relational_sine_class_cubic_response import (
    _CAUCHY_GAMMA,
    _higher_amplitude_remainder,
)
from .relational_sine_class_mediation import _CONTACTS, _EDGES, _NODES
from .relational_sine_class_superposition import (
    _ClassProbeHistoryBound,
    _probe_event_ledger,
    _unit_delay_donor_laplacian_bounds,
)
from .relational_sine_port_composition import _parameters

__all__ = ("SineClassAmplitudeFeasibility", "bound_sine_class_amplitude_feasibility")

_BASE_AMPLITUDE = Q(1, 2000)
_DELAY, _TOTAL = Q(1), Q(2)
_EPSILON, _DELTA = Q(1, 10**32), Q(1, 10**30)
_RADIUS = Q(1, 12)
_CONTACT_ALLOWANCE = Q(1, 10**12)
_PROBE_ALLOWANCE = Q(1, 500000)


def _delayed_laplacian_bounds(first_amplitude):
    """Fixed-delay heat pressure plus the complete-law/source defect."""
    return _unit_delay_donor_laplacian_bounds(
        first_amplitude=first_amplitude,
        endpoint_radius=_EPSILON,
        gamma_upper=_CAUCHY_GAMMA,
    )


@dataclass(frozen=True)
class SineClassAmplitudeFeasibility:
    """Uniform sufficient scale bounds, conditional on a supplied base coefficient.

    The upper-scale ledger supplies monotone work, norm and storage envelopes.
    Its individual mean shifts describe that endpoint only. Separate exact
    mean intervals below cover every scale in the declared positive interval.
    Failure to certify a requirement is not proof that the actual model fails it.
    """

    amplitude_scale_lower: Q
    amplitude_scale_upper: Q
    base_cubic_lower: Q
    base_cubic_upper: Q
    gamma_bounds: I
    scaled_cubic_contrast_bounds: tuple[Q, Q]
    cauchy_bootstrap_margin: Q
    cauchy_radius_margin: Q
    cauchy_admitted: bool
    per_history_higher_amplitude_remainder_upper_bounds: tuple[Q, ...] | None
    higher_amplitude_contrast_error_upper_bound: Q | None
    source_contrast_error_upper_bound: Q
    decision: _ContrastDecision | None
    response_bound_available: bool
    upper_scale_delayed_laplacian_bounds: tuple[tuple[Q, Q], ...]
    upper_scale_history_bounds: tuple[_ClassProbeHistoryBound, ...]
    first_form_mean_bounds: tuple[tuple[Q, Q], ...]
    final_form_mean_bounds: tuple[tuple[Q, Q], ...]
    phase_mean_bounds: tuple[Q, Q]
    contact_work_upper_bound: Q
    initial_excess_storage_upper_bound: Q
    storage_barrier: Q
    joined_initial_identity_certified: bool
    contact_work_within_allowance: bool
    all_identities_certified: bool
    all_work_within_allowances: bool
    feasible_interval_certified: bool
    status: str
    unmet_sufficient_requirements: tuple[str, ...]
    base_first_probe_amplitude: Q = _BASE_AMPLITUDE
    base_second_probe_amplitude: Q = _BASE_AMPLITUDE
    delay: Q = _DELAY
    total_duration: Q = _TOTAL
    endpoint_radius: Q = _EPSILON
    readout_error_bound: Q = _DELTA
    radius: Q = _RADIUS
    contact_work_allowance: Q = _CONTACT_ALLOWANCE
    first_probe_work_allowance: Q = _PROBE_ALLOWANCE
    second_probe_work_allowance: Q = _PROBE_ALLOWANCE
    work_gamma_upper_bound: Q = _CAUCHY_GAMMA
    cauchy_gamma_upper_bound: Q = _CAUCHY_GAMMA
    compared_classes: tuple[tuple[int, ...], ...] = ((1, 1, 1), (1, 2, 1))
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    contacts: tuple[tuple[int, int], ...] = _CONTACTS
    reading_count: int = 8
    clock: str = "tau=e*t; e=1023/1024"
    method: str = "cubic_homogeneity_interval_with_fixed_delay_heat_work_certificate_v1"
    scope: tuple[str, ...] = (
        "base_cubic_interval_is_a_supplied_conditional_premise_not_validated_here",
        "base_interval_must_enclose_the_complete_fixed_design_cubic_including_time_error",
        "no_coefficient_or_nonlinear_response_is_evaluated_and_no_scale_search_is_performed",
        "positive_common_scale_is_applied_to_both_carried_form_events_without_a_reset",
        "cubic_homogeneity_retains_the_entire_base_interval_before_complete_error_expansion",
        "higher_amplitude_error_and_uniform_event_bounds_use_the_upper_scale",
        "eight_reading_errors_and_sixteen_for_a_separately_noisy_zero_contrast_alternative",
        "delayed_work_retains_the_carried_donor_laplacian_with_no_continuous_loss_credit",
        "upper_scale_ledger_means_are_endpoint_values_separate_mean_intervals_cover_all_scales",
        "failed_sufficient_guards_do_not_prove_physical_infeasibility_or_nonidentifiability",
        "source_acquisition_and_laboratory_preparation_are_not_established",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-amplitude-feasibility.v1",
            "report": _project(self),
        }


def bound_sine_class_amplitude_feasibility(
    *,
    amplitude_scale_lower,
    amplitude_scale_upper,
    base_cubic_lower,
    base_cubic_upper,
) -> SineClassAmplitudeFeasibility:
    """Bound a declared positive scale interval without evaluating any response.

    All four scalars use shared exact-or-represented admission. Scales must be
    positive and ordered; the supplied base coefficient endpoints must be
    ordered. This function applies the coefficient premise, not its proof or
    numerical provenance. It admits the complete interval with uniform bounds.
    """
    values = {
        key: exact_or_represented_real(value, key)
        for key, value in dict(
            amplitude_scale_lower=amplitude_scale_lower,
            amplitude_scale_upper=amplitude_scale_upper,
            base_cubic_lower=base_cubic_lower,
            base_cubic_upper=base_cubic_upper,
        ).items()
    }
    lower, upper, base_lower, base_upper = (
        values[key]
        for key in (
            "amplitude_scale_lower",
            "amplitude_scale_upper",
            "base_cubic_lower",
            "base_cubic_upper",
        )
    )
    if not 0 < lower <= upper:
        raise ValueError("amplitude scales must be positive and ordered")
    if base_lower > base_upper:
        raise ValueError("base cubic coefficient endpoints must be ordered")
    gamma, _ = _parameters((1, 2, 1))
    products = tuple(
        scale**3 * coefficient
        for scale in (lower, upper)
        for coefficient in (base_lower, base_upper)
    )
    scaled = min(products), max(products)
    g = _CAUCHY_GAMMA
    amplitude = _BASE_AMPLITUDE * upper
    bootstrap = 1 - 16 * g**2 * _TOTAL**2
    radius_margin = 1 - 2 * g * (2 * amplitude)
    admitted = bootstrap > 0 and radius_margin > 0
    remainders = higher = decision = None
    source_error = 8 * _EPSILON / (1 - 2 * gamma.hi * _TOTAL)
    if admitted:
        remainders = tuple(
            _higher_amplitude_remainder(A, _TOTAL)
            for A in (Q(0), amplitude, amplitude, 2 * amplitude)
        )
        higher = 2 * sum(remainders, Q(0))
        decision = _contrast_decision(scaled, higher + source_error, _DELTA)
    first_amplitudes = (Q(0), amplitude, Q(0), amplitude)
    laplacian = tuple(_delayed_laplacian_bounds(a) for a in first_amplitudes)
    ledger_values = dict(
        first_probe_amplitude=amplitude,
        second_probe_amplitude=amplitude,
        delay=_DELAY,
        total_duration=_TOTAL,
        endpoint_radius=_EPSILON,
        readout_error_bound=_DELTA,
        radius=_RADIUS,
        contact_work_allowance=_CONTACT_ALLOWANCE,
        first_probe_work_allowance=_PROBE_ALLOWANCE,
        second_probe_work_allowance=_PROBE_ALLOWANCE,
    )
    ledger = _probe_event_ledger(
        ledger_values,
        g,
        (None,) * 4,
        pre_second_laplacian_abs_bounds=tuple(
            max(map(abs, bound)) for bound in laplacian
        ),
    )
    mean_error = Q(4, 58) * _EPSILON

    def mean_bounds(pulse_count):
        return (
            Q(3, 58) * pulse_count * _BASE_AMPLITUDE * lower - mean_error,
            Q(3, 58) * pulse_count * _BASE_AMPLITUDE * upper + mean_error,
        )

    first_means = tuple(mean_bounds(count) for count in (0, 1, 0, 1))
    final_means = tuple(mean_bounds(count) for count in (0, 1, 1, 2))
    identity = all(h.identity_certified for h in ledger.histories)
    work = all(h.work_within_allowances for h in ledger.histories)
    unmet = []
    if not admitted:
        unmet.append("complex_amplitude_domain_not_admitted")
    if decision is not None and not decision.null_excluded:
        unmet.append("separately_noisy_null_not_uniformly_excluded")
    if not identity:
        unmet.append("uniform_identity_not_certified")
    if not work:
        unmet.append("uniform_work_allowances_not_certified")
    feasible = (
        admitted
        and decision is not None
        and decision.null_excluded
        and identity
        and work
    )
    return SineClassAmplitudeFeasibility(
        **values,
        gamma_bounds=gamma,
        scaled_cubic_contrast_bounds=scaled,
        cauchy_bootstrap_margin=bootstrap,
        cauchy_radius_margin=radius_margin,
        cauchy_admitted=admitted,
        per_history_higher_amplitude_remainder_upper_bounds=remainders,
        higher_amplitude_contrast_error_upper_bound=higher,
        source_contrast_error_upper_bound=source_error,
        decision=decision,
        response_bound_available=decision is not None,
        upper_scale_delayed_laplacian_bounds=laplacian,
        upper_scale_history_bounds=ledger.histories,
        first_form_mean_bounds=first_means,
        final_form_mean_bounds=final_means,
        phase_mean_bounds=(-mean_error, mean_error),
        contact_work_upper_bound=ledger.contact_work,
        initial_excess_storage_upper_bound=ledger.initial_storage,
        storage_barrier=ledger.barrier,
        joined_initial_identity_certified=ledger.initial_identity,
        contact_work_within_allowance=ledger.contact_allowed,
        all_identities_certified=identity,
        all_work_within_allowances=work,
        feasible_interval_certified=feasible,
        status=(
            "feasible_interval"
            if feasible
            else "unavailable" if decision is None else "requirements_not_certified"
        ),
        unmet_sufficient_requirements=tuple(unmet),
    )

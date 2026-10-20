"""Detached five-child electrical protocol arithmetic.

``ORGANIZED_ELECTRICAL_PROTOCOL.md`` owns the conditional C5 model and its
observation scope. These helpers admit primitives and rebuild conservative
bounds; they perform no acquired-source assessment, trajectory, experiment,
calibration fit or physical identification. The capacitor support is fixed:
five nonadjacent branches are positive and five adjacent branches are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from ._organized_electrical_identification import (
    _intersection,
    _positive_box,
    _positive_interval,
    _positive_vector,
)

# Ordered input coordinates for this fixed five-child protocol.
_C5_PAIRS = tuple((a, b) for a in range(5) for b in range(a + 1, 5))
_C5_RING_PAIRS = tuple((a, b) for a, b in _C5_PAIRS if b - a in (1, 4))
_C5_CAPACITOR_PAIRS = tuple((a, b) for a, b in _C5_PAIRS if b - a in (2, 3))
_C5_SCOPE = (
    "fixed_five_child_ring_with_nine_slots_per_child",
    "five_positive_nonadjacent_mutual_capacitors_five_adjacent_branches_absent",
    "ten_positive_mutual_conductances_and_five_supplied_ring_sine_junctions",
    "floating_neutral_charge_q_equals_nine_kappa_times_U_minus_mean_U",
    "voltage_equals_voltage_scale_times_C5_laplacian_U",
    "same_circular_phase_with_supplied_positive_josephson_constant",
    "fixed_gamma_equals_one_over_1023_pi_and_one_positive_clock_scale",
    "primitive_component_intervals_require_independent_calibration",
    "overlap_is_only_necessary_not_joint_consistency_or_physical_admission",
    "no_source_preparation_observation_actuation_or_experimental_admission",
)


@dataclass(frozen=True)
class _C5ElectricalDeparture:
    initial_hidden_form: Q
    initial_hidden_phase: Q
    duration: Q
    collective_form_upper: Q
    gamma_bounds: I
    hidden_coupling: Q
    hidden_domain_margin: Q
    hidden_form_upper: Q
    hidden_phase_upper: Q
    mean_current_defect_upper: Q
    comparison_coupling: Q
    comparison_domain_margin: Q
    mean_form_error_upper: Q
    mean_phase_error_upper: Q
    storage_remainder_lower: Q
    storage_remainder_upper: Q
    integrated_hidden_loss_upper: Q
    integrated_hidden_work_upper: Q
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_C9_by_C5_quarter_sine_model_with_held_unit_capacity",
        "lifted_fine_deviations_have_zero_mean_in_every_child",
        "the_supplied_input_is_constant_across_each_child_so_hidden_input_vanishes",
        "ideal_C5_reference_starts_at_actual_collective_means",
        "supplied_collective_form_bound_includes_the_entire_known_input_offset",
        "signed_hidden_storage_is_not_a_nonnegative_reserve",
        "conditional_primitive_bounds_not_source_inclusion_or_physical_evidence",
    )


def _bound_c5_electrical_departure(
    *, initial_hidden_form, initial_hidden_phase, duration, collective_form_upper
):
    """Bound fixed-support mean closure and signed hidden accounting.

    Form and phase primitives are Euclidean norms of the 45 fine residuals;
    ``collective_form_upper`` is a supremum of each actual child mean's
    absolute form over the complete interval, including any known pulse
    offset. All four inputs are nonnegative exact/represented reals. This
    function does not certify that a supplied trajectory satisfies them.

    The shared hidden comparison uses ``lambda <= 8``. Its sufficient domain
    is ``2*gamma_upper*duration < 1``; the mean comparison additionally uses
    ``gamma_upper*duration < 1``. The closure defect is an l1 bound, so it
    also bounds its infinity norm. Storage and hidden loss use full fine
    normalization. The hidden-work bound instead bounds integral (L5*U).R
    in H5 units: multiply by nine in the full fine balance, and by 9*K for
    electrical energy. The signed storage remainder is not an energy reserve.
    """
    # Delay the source owner's import: pure component admission below has no
    # acquired-source dependency. Only this parameterized majorant is reused.
    from ._organized_electrical_source import _transport_hidden_deviations

    names = (
        "initial_hidden_form",
        "initial_hidden_phase",
        "duration",
        "collective_form_upper",
    )
    values = tuple(
        _exact(value, name)
        for name, value in zip(
            names,
            (
                initial_hidden_form,
                initial_hidden_phase,
                duration,
                collective_form_upper,
            ),
        )
    )
    if min(values) < 0:
        raise ValueError("C5 departure bounds require nonnegative primitives")
    a, z, time, form = values
    pi = pi_interval()
    gamma = I(1 / (1023 * pi.hi), 1 / (1023 * pi.lo))
    hidden = _transport_hidden_deviations(
        initial_form=a,
        initial_phase=z,
        gamma_upper=gamma.hi,
        laplacian_upper=8,
        duration=time,
    )
    A, B = hidden.form_upper, hidden.phase_upper
    rho = gamma.hi * B**2 / 9
    k = gamma.hi * time
    margin = 1 - k**2
    if margin <= 0:
        raise ValueError("C5 mean comparison requires gamma*duration < 1")
    form_error = time * rho / margin
    return _C5ElectricalDeparture(
        initial_hidden_form=a,
        initial_hidden_phase=z,
        duration=time,
        collective_form_upper=form,
        gamma_bounds=gamma,
        hidden_coupling=hidden.coupling,
        hidden_domain_margin=hidden.domain_margin,
        hidden_form_upper=A,
        hidden_phase_upper=B,
        mean_current_defect_upper=rho,
        comparison_coupling=k,
        comparison_domain_margin=margin,
        mean_form_error_upper=form_error,
        mean_phase_error_upper=k * form_error,
        storage_remainder_lower=-4 * B**2,
        storage_remainder_upper=4 * (A**2 + B**2),
        integrated_hidden_loss_upper=16 * time * A**2,
        integrated_hidden_work_upper=4 * form * time * rho,
    )


@dataclass(frozen=True)
class _C5ElectricalCalibration:
    capacitance_bounds: tuple[I, ...]
    conductance_bounds: tuple[I, ...]
    josephson_constant_bounds: I
    gamma_bounds: I
    capacitance_scale_bounds: I | None
    conductance_scale_bounds: I | None
    charge_scale_bounds: I | None
    clock_scale_bounds: I | None
    predicted_sine_amplitude_bounds: I | None
    voltage_scale_bounds: I | None
    energy_scale_bounds: I | None
    status: str
    exclusions: tuple[str, ...]
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = _C5_SCOPE


def _calibrate_c5_electrical_components(
    *, capacitances, conductances, josephson_constant
):
    """Enclose scales for the stipulated passive C5 reference.

    Five positive capacitances use ``_C5_CAPACITOR_PAIRS``; the five ring
    capacitors are exactly absent as a structural premise, not tested by
    this input. Ten positive conductances use ``_C5_PAIRS``. Values are
    admitted by the shared scalar/interval contract. Common branch values
    are ``c/5`` and ``g/5``. Necessary interval intersections cannot establish
    independence, realizability, observed equality or the entire law.

    This fixed dictionary predicts ``b=4g/c``, ``kappa=a0*gamma*g/9``,
    ``j=a0*gamma**2*g**2/c``, ``v*=a0*gamma*g/c`` and
    ``K=a0**2*gamma**2*g**2/(9c)``. Here ``V=v*L5*U`` and the collective
    electrical storage is ``9*K*H5``. It excludes the fixed internal
    carrier constant and signed fine-state storage remainder. Derived
    endpoints use exact rational expressions before one outward rounding.
    """
    caps = _positive_vector(capacitances, "capacitances", 5)
    conductors = _positive_vector(conductances, "conductances", 10)
    a0 = _positive_box(josephson_constant, "josephson_constant")
    pi = pi_interval()
    gamma = I(1 / (1023 * pi.hi), 1 / (1023 * pi.lo))
    c = _intersection(tuple(I(5 * value.lo, 5 * value.hi) for value in caps))
    g = _intersection(tuple(I(5 * value.lo, 5 * value.hi) for value in conductors))
    exclusions = tuple(
        label
        for missing, label in (
            (c is None, "capacitance_shape_intersection_empty"),
            (g is None, "conductance_shape_intersection_empty"),
        )
        if missing
    )
    charge = clock = predicted = voltage = energy = None
    if not exclusions:
        charge = _positive_interval(
            a0.lo * gamma.lo * g.lo / 9,
            a0.hi * gamma.hi * g.hi / 9,
            "charge_scale",
        )
        clock = _positive_interval(4 * g.lo / c.hi, 4 * g.hi / c.lo, "clock_scale")
        predicted = _positive_interval(
            a0.lo * gamma.lo**2 * g.lo**2 / c.hi,
            a0.hi * gamma.hi**2 * g.hi**2 / c.lo,
            "predicted_sine_amplitude",
        )
        voltage = _positive_interval(
            a0.lo * gamma.lo * g.lo / c.hi,
            a0.hi * gamma.hi * g.hi / c.lo,
            "voltage_scale",
        )
        energy = _positive_interval(
            a0.lo**2 * gamma.lo**2 * g.lo**2 / (9 * c.hi),
            a0.hi**2 * gamma.hi**2 * g.hi**2 / (9 * c.lo),
            "energy_scale",
        )
    return _C5ElectricalCalibration(
        capacitance_bounds=caps,
        conductance_bounds=conductors,
        josephson_constant_bounds=a0,
        gamma_bounds=gamma,
        capacitance_scale_bounds=c,
        conductance_scale_bounds=g,
        charge_scale_bounds=charge,
        clock_scale_bounds=clock,
        predicted_sine_amplitude_bounds=predicted,
        voltage_scale_bounds=voltage,
        energy_scale_bounds=energy,
        status=(
            "component_shape_excluded" if exclusions else "component_shape_not_excluded"
        ),
        exclusions=exclusions,
    )


@dataclass(frozen=True)
class _C5ElectricalSineComparison:
    calibration: _C5ElectricalCalibration
    sine_amplitude_bounds: tuple[I, ...]
    common_sine_amplitude_bounds: I | None
    dimensionless_ratio_bounds: I | None
    status: str
    exclusions: tuple[str, ...]
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = _C5_SCOPE


def _compare_c5_electrical_sine_amplitudes(
    *, capacitances, conductances, josephson_constant, sine_amplitudes
):
    """Check the unfitted five-ring amplitude consequence from primitives.

    Five positive amplitudes use ``_C5_RING_PAIRS`` and need independently
    evaluated uncertainty. The provided values are never used to calibrate
    any scale. A nonempty intersection only avoids this necessary exclusion;
    it neither admits a full sinusoidal current law nor confirms TNFR.
    """
    amplitudes = _positive_vector(sine_amplitudes, "sine_amplitudes", 5)
    calibration = _calibrate_c5_electrical_components(
        capacitances=capacitances,
        conductances=conductances,
        josephson_constant=josephson_constant,
    )
    common = _intersection(amplitudes)
    exclusions = calibration.exclusions
    if common is None:
        exclusions += ("sine_amplitude_intersection_empty",)
    ratio = None
    if calibration.exclusions:
        status = "component_shape_excluded"
    elif common is None:
        status = "sine_amplitude_uniformity_excluded"
    else:
        c = calibration.capacitance_scale_bounds
        g = calibration.conductance_scale_bounds
        a0 = calibration.josephson_constant_bounds
        ratio = _positive_interval(
            c.lo * common.lo / (a0.hi * g.hi**2),
            c.hi * common.hi / (a0.lo * g.lo**2),
            "dimensionless_ratio",
        )
        if _intersection((common, calibration.predicted_sine_amplitude_bounds)) is None:
            status = "sine_amplitude_relation_excluded"
            exclusions += ("predicted_sine_amplitude_intersection_empty",)
        else:
            status = "necessary_relation_not_excluded"
    return _C5ElectricalSineComparison(
        calibration=calibration,
        sine_amplitude_bounds=amplitudes,
        common_sine_amplitude_bounds=common,
        dimensionless_ratio_bounds=ratio,
        status=status,
        exclusions=exclusions,
    )

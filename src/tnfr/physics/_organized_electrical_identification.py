"""Detached necessary component conditions for the fixed collective circuit.

The proof and observational scope belong to
``theory/nodal/ORGANIZED_ELECTRICAL_IDENTIFICATION.md``. Supplied component
intervals are neither independently calibrated by this arithmetic nor evidence
of a realized preparation. No source, state, trajectory or physical constants
are inferred. Interval overlap is only a necessary condition, never a verdict
of mathematical equivalence or empirical agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from .relational_observations import _ordered

# These orders specify the input coordinates, not a second topology catalog.
_MUTUAL_PAIRS = tuple((a, b) for a in range(6) for b in range(a + 1, 6))
_WHEEL_PAIRS = tuple((a, b) for a, b in _MUTUAL_PAIRS if b == 5 or (b - a) in (1, 4))
_CAPACITANCE_WEIGHTS = tuple(
    Q(11 if b == 5 else (5 if (b - a) in (1, 4) else 41), 396) for a, b in _MUTUAL_PAIRS
)
_SCOPE = (
    "six_marked_children_with_five_child_rim_and_child_five_hub",
    "fifteen_positive_mutual_capacitors_and_fifteen_positive_mutual_conductances",
    "fixed_floating_neutral_charge_map_q_c_equals_nine_kappa_times_X_c_minus_r",
    "same_circular_collective_phase_and_one_positive_common_clock_scale",
    "supplied_sine_junction_law_and_supplied_positive_josephson_constant",
    "fixed_supplied_gamma_equals_one_over_1023_pi_not_a_fitted_parameter",
    "input_intervals_require_independent_calibration_and_complete_uncertainty",
    "overlap_is_only_necessary_not_a_joint_consistency_or_equivalence_certificate",
    "no_source_preparation_work_switching_or_physical_identification_admission",
)


def _positive_box(value, label):
    """Admit primitive positivity before outward materialization."""
    if type(value) is I:
        if value.lo <= 0:
            raise ValueError(
                f"{label} interval must have a strictly positive lower bound"
            )
        return value
    normalized = _exact(value, label)
    if normalized <= 0:
        raise ValueError(f"{label} must be strictly positive")
    return _positive_interval(normalized, normalized, label)


def _positive_interval(lower, upper, label):
    result = I(lower, upper)
    if result.lo <= 0:
        raise ValueError(
            f"{label}: unsupported interval domain; outward positive lower bound "
            "is not resolved on the fixed dyadic grid"
        )
    return result


def _positive_vector(values, label, size):
    raw = _ordered(values, label, limit=size + 1)
    if len(raw) != size:
        raise ValueError(f"{label} requires exactly {size} ordered branch values")
    return tuple(_positive_box(value, f"{label}[{i}]") for i, value in enumerate(raw))


def _intersection(values):
    lower = max(value.lo for value in values)
    upper = min(value.hi for value in values)
    return None if lower > upper else I(lower, upper)


@dataclass(frozen=True)
class _OrganizedElectricalCalibration:
    capacitance_bounds: tuple[I, ...]
    conductance_bounds: tuple[I, ...]
    josephson_constant_bounds: I
    gamma_bounds: I
    capacitance_scale_bounds: I | None
    conductance_scale_bounds: I | None
    charge_scale_bounds: I | None
    clock_scale_bounds: I | None
    predicted_sine_amplitude_bounds: I | None
    status: str
    exclusions: tuple[str, ...]
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = _SCOPE


@dataclass(frozen=True)
class _OrganizedElectricalSineComparison:
    calibration: _OrganizedElectricalCalibration
    sine_amplitude_bounds: tuple[I, ...]
    common_sine_amplitude_bounds: I | None
    dimensionless_ratio_bounds: I | None
    status: str
    exclusions: tuple[str, ...]
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = _SCOPE


def _calibrate_organized_electrical_components(
    *, capacitances, conductances, josephson_constant
):
    """Enclose bridge scales after supplied independent component calibration.

    Both vectors contain fifteen positive mutual branch values in
    ``_MUTUAL_PAIRS`` order: lexicographic pairs ``0 <= a < b <= 5``. The
    first five nodes form the rim, and node five is the hub. Capacitances
    require one common multiplier of the wheel inverse's ``5:41:11``
    mutual entries; conductances require one common value ``g/6``. Shape
    checks use every branch before deriving any scale, without least-squares
    fitting or trace-based admission.

    A scalar is admitted by the shared exact/represented-real contract;
    an exact shared ``I`` must have strictly positive lower endpoint. An
    unresolved positive lower endpoint on the fixed dyadic grid raises an
    explicit unsupported-domain error rather than changing precision.

    Conditional on the fixed map, ``kappa=a0*gamma*g/9``, ``b=4*g/c`` and
    ``j=a0*gamma**2*g**2/c``. Bounds consume only component primitives, not
    the later sine-amplitude response. They do not establish the assumed
    physical meaning, independence or joint realization of those primitives.
    """
    caps = _positive_vector(capacitances, "capacitances", 15)
    conductors = _positive_vector(conductances, "conductances", 15)
    a0 = _positive_box(josephson_constant, "josephson_constant")
    pi = pi_interval()
    gamma = I(1 / (1023 * pi.hi), 1 / (1023 * pi.lo))
    # Rational endpoint expressions are rounded only at their final boundary.
    # Intermediate products in physical units may lie below the dyadic grid
    # even when the requested final scale has a resolved positive enclosure.
    c = _intersection(
        tuple(
            I(value.lo / weight, value.hi / weight)
            for value, weight in zip(caps, _CAPACITANCE_WEIGHTS)
        )
    )
    g = _intersection(tuple(I(6 * value.lo, 6 * value.hi) for value in conductors))
    exclusions = tuple(
        label
        for missing, label in (
            (c is None, "capacitance_shape_intersection_empty"),
            (g is None, "conductance_shape_intersection_empty"),
        )
        if missing
    )
    kappa = clock = predicted = None
    if not exclusions:
        kappa = _positive_interval(
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
    return _OrganizedElectricalCalibration(
        capacitance_bounds=caps,
        conductance_bounds=conductors,
        josephson_constant_bounds=a0,
        gamma_bounds=gamma,
        capacitance_scale_bounds=c,
        conductance_scale_bounds=g,
        charge_scale_bounds=kappa,
        clock_scale_bounds=clock,
        predicted_sine_amplitude_bounds=predicted,
        status=(
            "component_shape_excluded" if exclusions else "component_shape_not_excluded"
        ),
        exclusions=exclusions,
    )


def _compare_organized_electrical_sine_amplitudes(
    *, capacitances, conductances, josephson_constant, sine_amplitudes
):
    """Test necessary fixed-map relations using freshly admitted primitives.

    The ten positive sine amplitudes use ``_WHEEL_PAIRS``: lexicographic
    pairs from the five-cycle rim and its five hub spokes. These supplied
    amplitudes must have been evaluated independently of the component
    calibration and its prediction; this function does not authenticate that
    provenance or admit a nonsinusoidal current law.

    Empty shape, common-amplitude or prediction intersections are distinct
    exclusions. Nonempty interval overlap remains only ``not_excluded``:
    correlations, approximation error, observation and other premises require
    separate admission. No incoming calibration report or cached verdict is
    consumed, and no amplitude is used to adjust the fixed gamma or scales.
    """
    amplitudes = _positive_vector(sine_amplitudes, "sine_amplitudes", 10)
    calibration = _calibrate_organized_electrical_components(
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
        prediction = calibration.predicted_sine_amplitude_bounds
        if _intersection((common, prediction)) is None:
            status = "sine_amplitude_relation_excluded"
            exclusions += ("predicted_sine_amplitude_intersection_empty",)
        else:
            status = "necessary_relation_not_excluded"
    return _OrganizedElectricalSineComparison(
        calibration=calibration,
        sine_amplitude_bounds=amplitudes,
        common_sine_amplitude_bounds=common,
        dimensionless_ratio_bounds=ratio,
        status=status,
        exclusions=exclusions,
    )

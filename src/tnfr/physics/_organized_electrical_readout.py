"""Primitive voltage adapter for the fixed finite C5 observation word.

The voltage/charge dictionary is a supplied observation hypothesis, not an
independent measurement of charge. Timing, source tracking and electronics
errors require separately justified allowances. No sensor or source is
installed, calibrated or authenticated by this detached inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ._organized_electrical_protocol import (
    _C5ElectricalCalibration,
    _calibrate_c5_electrical_components,
)
from ._organized_finite_observation import (
    _FiniteObservationWindingInference,
    _fixed_rows,
    _infer_finite_observation_winding,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector


_FROZEN_FORM_ERROR = Q(1, 10**14)


@dataclass(frozen=True)
class _C5VoltageWindingInference:
    calibration: _C5ElectricalCalibration
    voltage_readings: tuple
    voltage_error: Q
    mean_form: Q
    mean_form_error: Q
    additional_form_error: Q
    transformed_voltage_readings: tuple | None
    inverse_voltage_scale_endpoints: tuple[Q, Q] | None
    form_readings: tuple | None
    form_readout_error_upper: Q | None
    inference: _FiniteObservationWindingInference | None
    identified_winding: int | None
    status: str
    unresolved_conditions: tuple[str, ...]
    frozen_form_error_allowance: Q = _FROZEN_FORM_ERROR
    scope: tuple[str, ...] = (
        "same_150_endpoint_readings_and_finite_input_word_as_frozen_protocol",
        "supplied_voltage_dictionary_V_equals_voltage_scale_times_L5_U",
        "constant_floating_mean_form_is_supplied_not_inferred_from_voltage",
        "exact_common_voltage_gauge_cancellation",
        "component_scale_uncertainty_and_arbitrary_correlated_readout_errors_retained",
        "additional_source_timing_and_tracking_errors_require_independent_bounds",
        "all_source_law_clock_preparation_and_sensor_premises_remain_conditional",
        "no_observed_charge_or_phase_and_no_physical_identification",
    )


def _infer_c5_voltage_winding(
    *,
    capacitances,
    conductances,
    josephson_constant,
    voltages,
    voltage_error,
    mean_form,
    mean_form_error=0,
    additional_form_error=0,
):
    """Rebuild form-readout error before the unchanged finite-word inverse.

    ``voltages`` has shape (5,3,2,5), with the same marked block, plateau,
    endpoint and node order as the original 150 form records. Each scalar
    error is bounded by ``voltage_error``; arbitrary error correlations and
    arbitrary common voltage offsets at each endpoint are permitted.
    The constant mean form and its uncertainty must be separately supplied.

    With ``w=L5+ V``, inverse-scale endpoints ``alpha_lo,alpha_hi`` and
    their midpoint/radius, use ``Uhat=mean_form+alpha_mid*w``. The maximum
    absolute row sum of L5+ is 4/5, so one uniform raw form error is
    ``mean_error + alpha_hi*4*voltage_error/5 + alpha_radius*max(abs(w))
    + additional_form_error``. The additional bound is independently
    supplied; it is never inferred from the records or an earlier verdict.

    Component-shape exclusions and errors above the frozen 1e-14 allowance
    return explicit unavailable results after all reading/error admission.
    This adapter does not revise that frozen budget or validate the physical
    clock, input delivery, initial state, waveform or measurement model.
    """
    normalized = []
    for c, block in enumerate(_fixed_rows(voltages, "voltages", 5)):
        plateaus = _fixed_rows(block, f"voltages[{c}]", 3)
        normalized.append(
            tuple(
                tuple(
                    _vector(row, f"voltages[{c}][{j}][{e}]", 5)
                    for e, row in enumerate(
                        _fixed_rows(plateau, f"voltages[{c}][{j}]", 2)
                    )
                )
                for j, plateau in enumerate(plateaus)
            )
        )
    values = tuple(normalized)
    voltage_error, mean_form, mean_form_error, additional_form_error = (
        _exact(value, name)
        for name, value in (
            ("voltage_error", voltage_error),
            ("mean_form", mean_form),
            ("mean_form_error", mean_form_error),
            ("additional_form_error", additional_form_error),
        )
    )
    if min(voltage_error, mean_form_error, additional_form_error) < 0:
        raise ValueError("voltage, mean-form and additional errors must be nonnegative")
    calibration = _calibrate_c5_electrical_components(
        capacitances=capacitances,
        conductances=conductances,
        josephson_constant=josephson_constant,
    )
    transformed = scale = forms = error = inference = winding = None
    unresolved = calibration.exclusions
    status = "unavailable_component_shape"
    if not unresolved:
        v = calibration.voltage_scale_bounds
        scale = (1 / v.hi, 1 / v.lo)
        midpoint, radius = (scale[0] + scale[1]) / 2, (scale[1] - scale[0]) / 2
        transformed = tuple(
            tuple(
                tuple(
                    tuple(
                        (2 * row[i] - row[(i + 2) % 5] - row[(i + 3) % 5]) / 5
                        for i in range(5)
                    )
                    for row in plateau
                )
                for plateau in block
            )
            for block in values
        )
        largest = max(
            abs(item)
            for block in transformed
            for plateau in block
            for endpoint in plateau
            for item in endpoint
        )
        forms = tuple(
            tuple(
                tuple(
                    tuple(mean_form + midpoint * item for item in row) for row in pair
                )
                for pair in block
            )
            for block in transformed
        )
        error = (
            mean_form_error
            + scale[1] * Q(4, 5) * voltage_error
            + radius * largest
            + additional_form_error
        )
        if error > _FROZEN_FORM_ERROR:
            status = "unavailable_frozen_readout_allowance"
            unresolved = ("reconstructed_form_error_exceeds_frozen_allowance",)
        else:
            inference = _infer_finite_observation_winding(
                readings=forms, raw_readout_error=error
            )
            winding = inference.identified_winding
            status = inference.status
            unresolved = inference.unresolved_conditions
    return _C5VoltageWindingInference(
        calibration=calibration,
        voltage_readings=values,
        voltage_error=voltage_error,
        mean_form=mean_form,
        mean_form_error=mean_form_error,
        additional_form_error=additional_form_error,
        transformed_voltage_readings=transformed,
        inverse_voltage_scale_endpoints=scale,
        form_readings=forms,
        form_readout_error_upper=error,
        inference=inference,
        identified_winding=winding,
        status=status,
        unresolved_conditions=unresolved,
    )

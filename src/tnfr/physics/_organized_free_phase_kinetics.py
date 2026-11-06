"""Independent circular phase checks of the free probe's kinetic mean row.

Primitive phases are observed, never reconstructed from form. This detached
reader tests a necessary consequence of ``delta_s = gamma*(X_A-X_B)`` in an
independently declared observation clock s. It neither supplies that clock
nor admits a physical measurement bridge, formation or retained identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval
from ._organized_collective_observation import (
    _MarkedC9PairObservation,
    _observe_marked_c9_pair,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector
from .relational_observations import _ordered

_MAX_SNAPSHOTS = 128


@dataclass(frozen=True)
class _FreePhaseKineticCell:
    sample_times: tuple[Q, Q]
    integrated_velocity_bounds: I
    product_phase_increment_bounds: I
    product_circular_residual_bounds: I
    product_squared_chord_defect_bounds: I
    phase_increment_bounds: I | None
    circular_residual_bounds: I | None
    squared_chord_defect_bounds: I | None
    mean_increment_available: bool
    mean_lift_margin_lower_bounds: tuple[Q, Q] | None
    mean_increment_unavailable_reason: str | None
    status: str


@dataclass(frozen=True)
class _OrganizedFreePhaseKineticRecord:
    """Necessary circular consistency checks, never a verified kinetic law.

    The product-phase defect is global: it tests exp(i*sum(theta_A-theta_B))
    against nine times the form integral. It needs no intersample collar, but
    discards mean increments modulo 2*pi/9. The finer mean-phase defect is
    eligible for a verdict only with an independently certified no-slip bound.
    Otherwise its fields are snapshot-chart diagnostics, not temporal lifts.
    A separated eligible defect rejects the joint collective kinetic and
    observation premises; every other result remains inconclusive. Eighteen
    primitive inputs do not turn this collective test into a check of all
    nodal phase rows: zero-mean internal phase motion may also be invisible.
    """

    sample_times: tuple[Q, ...]
    form_snapshots: tuple[tuple[Q, ...], ...]
    phase_snapshots: tuple[tuple[Q, ...], ...]
    form_error_bound: Q
    phase_error_bound: Q
    exposure_time_error_bound: Q
    velocity_lipschitz_bound: Q
    primitive_phase_speed_bound: Q | None
    gamma_bounds: I
    effective_phase_error_bound: Q
    velocity_sample_error_upper_bound: Q
    snapshots: tuple[_MarkedC9PairObservation, ...]
    velocity_bounds: tuple[I, ...]
    relative_phase_bounds: tuple[I | None, ...]
    cells: tuple[_FreePhaseKineticCell, ...]
    status: str
    interval_method: str = INTERVAL_METHOD
    conditional_on_observation_bounds: bool = True
    scope: tuple[str, ...] = (
        "eighteen_primitive_forms_and_phases_in_fixed_marked_child_order",
        "independently_declared_observation_clock_s_on_zero_to_one",
        "all_actual_exposures_are_assumed_to_remain_in_zero_to_one",
        "per_node_form_and_circular_phase_readout_errors_are_supplied_bounds",
        "velocity_Lipschitz_and_primitive_phase_speed_bounds_are_independent_premises",
        "speed_bounds_hold_on_the_whole_nominal_and_actual_exposure_window",
        "timestamp_jitter_does_not_supply_an_unknown_clock_rate",
        "snapshot_collars_do_not_certify_intersample_marked_child_identity",
        "product_phase_test_is_global_and_retains_mean_aliases_modulo_two_pi_over_nine",
        "mean_phase_verdict_requires_independently_certified_no_internal_slip",
        "circular_defects_discard_complete_turns_without_temporal_unwrapping",
        "zero_enclosed_defect_is_inconclusive_not_law_verification",
        "collective_phase_row_only_not_a_certificate_of_the_full_nodal_phase_law",
        "zero_mean_internal_phase_motion_can_remain_unobserved_by_these_defects",
        "no_pressure_or_phase_reconstruction_from_the_tested_kinetic_row",
        "no_preparation_retention_response_or_physical_observation_authentication",
    )


def _read_organized_free_phase_kinetic_record(
    *,
    sample_times,
    form_snapshots,
    phase_snapshots,
    form_error_bound,
    phase_error_bound,
    exposure_time_error_bound,
    velocity_lipschitz_bound,
    primitive_phase_speed_bound=None,
) -> _OrganizedFreePhaseKineticRecord:
    """Check independent primitive phases against the integrated kinetic row.

    Admit 2..128 strictly increasing nominal times in [0,1], each with eighteen
    finite signed scalar forms and eighteen finite primitive phase scalars.
    Error/speed bounds are nonnegative exact or represented reals. A primitive
    phase speed bound is optional only when timestamp jitter is exactly zero.
    These bounds are not inferred from samples or from the tested equation.

    With per-node reading errors eps_x and eta, timestamp jitter xi, and
    independent bounds |v_s|<=Lv and |theta_node,s|<=Ltheta, nominal samples
    have errors 2*gamma*eps_x+Lv*xi and eta+Ltheta*xi respectively. Correlation
    is allowed; the conservative boxes assert no statistical independence.
    For a Lipschitz velocity the trapezoid error on a cell of length h is
    at most Lv*h**2/4. This uses only |v_s|<=Lv almost everywhere, not a
    pressure law, fine-state certificate or phase evolution hypothesis.

    The global product-phase defect is always computed, including when a
    marked collar is unresolved. The finer arithmetic-mean diagnostic may
    determine a verdict only if both endpoint collars and a whole-cell
    no-internal-slip bound are independently certified using Ltheta. Neither
    a sampled collar nor the kinetic equation itself supplies this bound.
    Unavailable local phases remain None; they never hide a separated global
    contradiction. Every nonseparated result is inconclusive.
    """
    raw_times = _ordered(sample_times, "sample_times", limit=_MAX_SNAPSHOTS + 1)
    if not 2 <= len(raw_times) <= _MAX_SNAPSHOTS:
        raise ValueError("sample_times requires between 2 and 128 snapshots")
    times = tuple(
        _exact(value, f"sample_times[{i}]") for i, value in enumerate(raw_times)
    )
    if (
        times[0] < 0
        or times[-1] > 1
        or any(left >= right for left, right in zip(times, times[1:]))
    ):
        raise ValueError("sample_times must strictly increase within [0,1]")
    count = len(times)

    def snapshots(values, label):
        rows = _ordered(values, label, limit=count + 1)
        if len(rows) != count:
            raise ValueError(f"{label} must match the sample count")
        return tuple(_vector(row, f"{label}[{i}]", 18) for i, row in enumerate(rows))

    forms = snapshots(form_snapshots, "form_snapshots")
    phases = snapshots(phase_snapshots, "phase_snapshots")
    eps_x, eta, xi, lv = tuple(
        _exact(value, name)
        for name, value in (
            ("form_error_bound", form_error_bound),
            ("phase_error_bound", phase_error_bound),
            ("exposure_time_error_bound", exposure_time_error_bound),
            ("velocity_lipschitz_bound", velocity_lipschitz_bound),
        )
    )
    ltheta = (
        None
        if primitive_phase_speed_bound is None
        else _exact(primitive_phase_speed_bound, "primitive_phase_speed_bound")
    )
    if min(eps_x, eta, xi, lv) < 0 or (ltheta is not None and ltheta < 0):
        raise ValueError(
            "observation errors and independent speed bounds must be nonnegative"
        )
    if xi and ltheta is None:
        raise ValueError("nonzero timestamp error requires primitive_phase_speed_bound")

    gamma = 1 / (1023 * pi_interval())
    phase_error = eta + (ltheta or Q(0)) * xi
    velocity_error = 2 * gamma.hi * eps_x + lv * xi
    observations = tuple(
        _observe_marked_c9_pair(
            form=form,
            phase=tuple(I(value - phase_error, value + phase_error) for value in phase),
        )
        for form, phase in zip(forms, phases)
    )
    velocities = tuple(
        gamma * (frame.children[0].form_mean - frame.children[1].form_mean)
        + I(-velocity_error, velocity_error)
        for frame in observations
    )
    relative_phases = tuple(
        (
            frame.children[0].phase_mean_lift_bounds
            - frame.children[1].phase_mean_lift_bounds
            if frame.available
            else None
        )
        for frame in observations
    )
    product_phases = tuple(
        sum(frame.primitive_phase_bounds[:9], I(0))
        - sum(frame.primitive_phase_bounds[9:], I(0))
        for frame in observations
    )
    cells = []
    for i, (left, right) in enumerate(zip(times, times[1:])):
        h = right - left
        truncation = lv * h**2 / 4
        area = h * (velocities[i] + velocities[i + 1]) / 2 + I(-truncation, truncation)
        product_increment = product_phases[i + 1] - product_phases[i]
        product_residual = product_increment - 9 * area
        product_defect = 2 * (1 - cos(product_residual))
        margins = None
        mean_available = False
        if relative_phases[i] is None or relative_phases[i + 1] is None:
            increment = residual = defect = None
            reason = "endpoint_marked_collar_unresolved"
        else:
            increment = relative_phases[i + 1] - relative_phases[i]
            residual = increment - area
            defect = 2 * (1 - cos(residual))
            if ltheta is None:
                reason = "independent_phase_speed_not_supplied"
            else:
                margins = tuple(
                    pi_interval().lo
                    - max(
                        (value - child.centered_phase_bounds[0]).abs_max
                        for value in child.centered_phase_bounds
                    )
                    - 2 * ltheta * h
                    for child in observations[i].children
                )
                mean_available = min(margins) > 0
                reason = (
                    None if mean_available else "intersample_internal_slip_unresolved"
                )
        status = (
            "incompatible_kinetic_record"
            if product_defect.lo > 0 or (mean_available and defect.lo > 0)
            else "inconclusive"
        )
        cells.append(
            _FreePhaseKineticCell(
                sample_times=(left, right),
                integrated_velocity_bounds=area,
                product_phase_increment_bounds=product_increment,
                product_circular_residual_bounds=product_residual,
                product_squared_chord_defect_bounds=product_defect,
                phase_increment_bounds=increment,
                circular_residual_bounds=residual,
                squared_chord_defect_bounds=defect,
                mean_increment_available=mean_available,
                mean_lift_margin_lower_bounds=margins,
                mean_increment_unavailable_reason=reason,
                status=status,
            )
        )
    statuses = {cell.status for cell in cells}
    status = (
        "incompatible_kinetic_record"
        if "incompatible_kinetic_record" in statuses
        else "inconclusive"
    )
    return _OrganizedFreePhaseKineticRecord(
        sample_times=times,
        form_snapshots=forms,
        phase_snapshots=phases,
        form_error_bound=eps_x,
        phase_error_bound=eta,
        exposure_time_error_bound=xi,
        velocity_lipschitz_bound=lv,
        primitive_phase_speed_bound=ltheta,
        gamma_bounds=gamma,
        effective_phase_error_bound=phase_error,
        velocity_sample_error_upper_bound=velocity_error,
        snapshots=observations,
        velocity_bounds=velocities,
        relative_phase_bounds=relative_phases,
        cells=tuple(cells),
        status=status,
    )

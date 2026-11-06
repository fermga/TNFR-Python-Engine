"""Independent full-law finite readout from a supplied two-port source box.

All eighteen forms and eighteen continuous phase lifts evolve after the
supplied phase dipole, or without an event when its amplitude is zero.
Shared Picard/Taylor certificates enclose the complete flow; no
inverse-response approximation or equilibrium target is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._interval_taylor import MAX_ORDER
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ..mathematics._validated_taylor import (
    ValidatedBoxTaylorStep,
    validated_box_taylor_step,
)
from ._sine_flow import _full_sine_field
from .phase_cycle_geometry import PhaseCycleGeometry, _derive
from .relational_observations import _interval, _ordered
from .relational_sine_two_port_compatibility import _EDGES, _NODES

__all__ = ("SineTwoPortReadout", "bound_sine_two_port_readout")


@dataclass(frozen=True)
class SineTwoPortReadout:
    """One independently generated finite local increment and its full audit.

    Source boxes are primitive enclosures, not an acquired-source assertion.
    Phase entries are continuous lifts in radians. The complete smooth sine
    law is evolved in fast time tau=e*t; a positive kick changes only phase,
    while zero amplitude means no event and preserves the full source box.
    The readout is q-transpose-x with q=e4-e5, with its pre-event baseline
    retained separately. The true increment removes the initial coordinate
    symbolically before interval evaluation. No sensor gain or noise is used.
    """

    initial_form_bounds: tuple[I, ...]
    initial_phase_bounds: tuple[I, ...]
    phase_increment: Q
    probe_duration: Q
    order: int
    reference_model: RelationalExchangeModel
    geometry: PhaseCycleGeometry
    degrees: tuple[int, ...]
    dipole: tuple[Q, ...]
    post_event_initial_box: tuple[I, ...]
    baseline_readout_bounds: I
    step: ValidatedBoxTaylorStep | None
    true_increment_bounds: I | None
    endpoint_readout_bounds: I | None
    failed_tube: tuple[I, ...] | None
    status: str
    unavailable_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    law: str = "normalized_sine_reciprocal_exchange"
    clock: str = "tau=e*t"
    capacity: tuple[Q, ...] = (Q(1),) * 18
    state_order: tuple[str, ...] = tuple(f"x_{i}" for i in range(18)) + tuple(
        f"theta_{i}" for i in range(18)
    )
    scope: tuple[str, ...] = (
        "fixed_two_port_C9_support_and_full_thirty_six_coordinate_source_box",
        "complete_shared_sine_rates_with_both_rows_transformed_to_fast_clock",
        "supplied_phase_dipole_then_unforced_continuous_flow",
        "zero_phase_increment_means_no_event_and_preserves_all_source_coordinates",
        "one_fixed_Picard_Taylor_step_without_retry_or_adaptive_budget",
        "direct_source_box_coefficients_and_whole_tube_derivative_remainder",
        "readout_increment_cancels_the_same_initial_form_coordinate_symbolically",
        "independent_of_inverse_error_bound_target_root_or_forward_verdict",
        "global_smooth_domain_is_not_acute_chart_or_identity_retention",
        "primitive_source_enclosure_is_not_acquisition_or_calibration_evidence",
        "no_sensor_gain_noise_realization_or_physical_measurement",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-two-port-readout.v1", "report": _project(self)}


def bound_sine_two_port_readout(
    *,
    initial_form_bounds,
    initial_phase_bounds,
    phase_increment,
    probe_duration,
    order,
) -> SineTwoPortReadout:
    """Generate a finite full-flow readout from eighteen primitive bound pairs.

    All endpoints undergo shared exact/represented-real admission before
    interval arithmetic. Increment lies in [0,1], with zero denoting no
    phase event; duration lies in (0,1]. Order is an ordinary integer in
    1..16. A failed fixed step returns unavailable with
    its last tube. No trajectory target, inverse enclosure or cached report
    can substitute for the supplied complete source box.
    """
    channels = []
    for raw, label in (
        (initial_form_bounds, "initial_form_bounds"),
        (initial_phase_bounds, "initial_phase_bounds"),
    ):
        rows = _ordered(raw, label, limit=19)
        if len(rows) != 18:
            raise ValueError(f"{label} must contain exactly eighteen endpoint pairs")
        channels.append(
            tuple(_interval(row, f"{label}[{i}]") for i, row in enumerate(rows))
        )
    form, phase = channels
    amplitude = exact_or_represented_real(phase_increment, "phase_increment")
    horizon = exact_or_represented_real(probe_duration, "probe_duration")
    if not 0 <= amplitude <= 1:
        raise ValueError("phase_increment must lie in [0,1]")
    if not 0 < horizon <= 1:
        raise ValueError("probe_duration must lie in (0,1]")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("order must be an ordinary integer in 1..16")
    geometry = _derive(_NODES, _EDGES)
    degrees = tuple(sum(i in edge for edge in geometry.edges) for i in _NODES)
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    q = tuple(Q(int(i == 4) - int(i == 5)) for i in _NODES)
    initial = form + tuple(
        value + amplitude * coefficient for value, coefficient in zip(phase, q)
    )
    flow, domain = _full_sine_field(model, geometry, degrees)
    step, failed, reason = validated_box_taylor_step(
        initial, horizon, flow, domain, order=order
    )
    return SineTwoPortReadout(
        initial_form_bounds=form,
        initial_phase_bounds=phase,
        phase_increment=amplitude,
        probe_duration=horizon,
        order=order,
        reference_model=model,
        geometry=geometry,
        degrees=degrees,
        dipole=q,
        post_event_initial_box=initial,
        baseline_readout_bounds=form[4] - form[5],
        step=step,
        true_increment_bounds=step.increment[4] - step.increment[5] if step else None,
        endpoint_readout_bounds=step.endpoint[4] - step.endpoint[5] if step else None,
        failed_tube=failed,
        status="admitted" if step else "unavailable",
        unavailable_reasons=() if reason is None else (reason,),
    )

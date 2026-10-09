"""Static finite-time boundary-access counterexample on two joined C5 rings.

The named boundary state is outside the native phase domain. Its four consumed
row pairs have smooth limits in the reflection restriction, enabling a local
backward-flow existence proof. This is not an execution adapter, trajectory,
or continuation rule for an undefined full-state field.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I, arg, pi_interval
from ._relational_reflected_flow import _reflected_rate_rows

__all__ = (
    "RelationalReflectedBoundaryExitCertificate",
    "certify_relational_reflected_boundary_exit",
)


@dataclass(frozen=True)
class RelationalReflectedBoundaryExitCertificate:
    """Exact named boundary and interval-enclosed limiting algebra.

    Resultants use ``row_order``; rates use the eight reflected coordinates.
    The negative central-resultant derivative proves local transverse access
    from the regular side. Existence does not specify an initial state or a
    hitting time. The nonextension slope belongs to different full-state
    approaches, not to continuation of the reflected solution.
    """

    model: RelationalExchangeModel
    form_amplitude: Q
    coordinates: tuple[I, ...]
    limiting_resultants: tuple[tuple[I, I], ...]
    limiting_rates: tuple[I, ...]
    limiting_resultant_rates: tuple[tuple[I, I], ...]
    storage: I
    continuous_loss: I
    seven_beta_storage_gap: Q
    below_seven_beta: bool
    full_state_phase_rate_slope: I
    row_order: tuple[int, ...] = (0, 4, 3, 5, 9, 8)
    boundary_node: int = 3
    scope: tuple[str, ...] = (
        "two_unit_C5_rings_adjacent_matching_bridges_unit_held_capacity",
        "regular_unforced_reference_law_positive_phase_weight",
        "ideal_boundary_state_not_admitted_for_native_execution",
        "limiting_consumed_rates_do_not_define_the_full_boundary_field",
        "local_backward_flow_proves_some_acute_initial_states_hit_zero_in_finite_time",
        "no_explicit_initial_state_hitting_time_or_frozen_response_prediction",
        "low_energy_counterexample_requires_positive_seven_beta_storage_gap",
        "full_state_approach_parameter_c_changes_phase_rate_by_c_times_slope",
        "no_continuous_full_state_extension_even_from_positive_resultants",
        "no_boundary_continuation_event_law_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-reflected-boundary-exit.v1",
            "report": _project(self),
        }


def certify_relational_reflected_boundary_exit(*, model, form_amplitude=1):
    """Certify local finite-time access to a genuine zero-resultant boundary.

    The ideal state is ``(0,r,0,0,0,pi/2,0,0)`` with exact positive ``r``.
    Analytic identities, shared Arg enclosures and the shared consumed rate
    algebra prove transversality. Native six-resultant admission is unchanged.
    Energy below ``7*beta`` is recorded only when its exact gap is positive.
    """
    if (
        not isinstance(model, RelationalExchangeModel)
        or model.phase_domain != "regular"
    ):
        raise ValueError("an explicit regular relational model is required")
    if type(form_amplitude) is not int and not isinstance(form_amplitude, Q):
        raise TypeError("form_amplitude requires an exact positive Fraction or integer")
    rho = Q(form_amplitude)
    if rho <= 0:
        raise ValueError("form_amplitude must be positive")
    e, w = map(Q, model.effective_weights)
    beta, pi = Q(model.storage_scale), pi_interval()
    zero, one = I(0), I(1)
    h = arg(I(2), one) / pi
    speed = w * rho / beta
    state = (zero, I(rho), zero, zero, zero, pi / 2, zero, zero)
    resultants = (
        (I(2), one),
        (zero, I(-2)),
        (zero, zero),
        (I(3), zero),
        (I(2), zero),
        (I(2), zero),
    )
    rates = _reflected_rate_rows(
        (I(-rho), I(2 * rho), zero, zero),
        (h, I(Q(-1, 2)), zero, zero),
        (h, I(Q(1, 4)), 1 / (3 * pi), 1 / (2 * pi)),
        epi_weight=e,
        phase_weight=w,
        storage_scale=beta,
    )
    resultant_rates = (
        (-speed * (h + Q(1, 2)), 3 * speed * h),
        (-speed * (h + 1), zero),
        (I(-speed), zero),
        (zero, -speed * h),
        (zero, zero),
        (zero, zero),
    )
    storage = I(2 * rho**2 + 4 * beta)
    gap = 3 * beta - 2 * rho**2
    if rates[5].lo <= 0 or resultant_rates[2][0].hi >= 0:
        raise ArithmeticError("boundary transversality enclosure is unresolved")
    return RelationalReflectedBoundaryExitCertificate(
        model=model,
        form_amplitude=rho,
        coordinates=state,
        limiting_resultants=resultants,
        limiting_rates=rates,
        limiting_resultant_rates=resultant_rates,
        storage=storage,
        continuous_loss=I(Q(14, 3) * e * rho**2),
        seven_beta_storage_gap=gap,
        below_seven_beta=gap > 0,
        full_state_phase_rate_slope=w / (beta * pi),
    )

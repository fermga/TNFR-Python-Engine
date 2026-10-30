"""Fixed changed-input policy and conditional admission, without a response run.

The original acquired families, complete law and common origins are premises.
The grounded tangent port truncation has its own visible initialization and
recording error; it is not assigned the complete model's storage or mean law.
"""

from fractions import Fraction as Q

from ..physics._sine_class_collective_interface import _bound_collective_interface
from .artifact_io import exact_record


def collective_prediction_inputs(mediator_class):
    """Primitive residual covers for the fixed, previously untested port word.

    These covers contain the original component balls and zero-sum constraints.
    They do not assert independent acquisition, identical residuals between
    classes/models, or that every point in the boxes belongs to the family.
    """
    if type(mediator_class) is not int or mediator_class not in (1, 2):
        raise ValueError("mediator class must be ordinary integer one or two")
    eps = Q(1, 10**32)
    return dict(
        mediator_class=mediator_class,
        initial_form_bounds=((-eps, eps),) * 27,
        initial_phase_bounds=((-eps, eps),) * 27,
        comparator_initial_bounds=((-eps, eps),) * 6,
        port_impulse=(Q(0), Q(7, 10000), Q(0)),
        horizon=Q(1),
    )


def collective_prediction_admission():
    """Rebuild the selected word's work, identity and a priori gap bounds.

    The gap uses positive heat walks and a full-sine/heat error bound, not a
    coefficient evaluation. It is a sufficient bound for this named alternative
    and does not establish nonlinear necessity or class discrimination.
    """
    inputs = collective_prediction_inputs(1)
    q, horizon = inputs["port_impulse"][1], inputs["horizon"]
    eps = inputs["initial_form_bounds"][0][1]
    g, delta, numerical = Q(1, 3000), Q(1, 10**8), Q(1, 10**12)
    ell, denominator = 1 - 2 * g * horizon, 1 - 2 * g**2 * horizon**2
    fidelity = _bound_collective_interface(
        total_input_variation=q, horizon=horizon, endpoint_radius=eps
    )
    work = (2 * q**2 - 8 * q * eps, 2 * q**2 + 8 * q * eps)
    storage = 22 * eps**2 + work[1]
    radius = 27 * ((q + eps) ** 2 + eps**2)
    # exp(-1)>1/3 and the hidden two-step return weight is 1/4.
    nominal_gap = q * (Q(1, 24) - 4 * g**2 / denominator)
    source = eps / ell
    conservative_gap = (
        nominal_gap
        - 2 * source
        - 2 * delta
        # Each retained endpoint can lie a full interval width from truth.
        - 4 * numerical
        # Full nominal -> cubic -> returned full band uses the tail twice.
        - 2 * fidelity.nominal_fifth_order_error_upper_bound
        - fidelity.nonlinear_initialization_error_upper_bound
    )
    return dict(
        event_work_bounds=work,
        event_work_ceiling=Q(2, 10**6),
        post_event_excess_storage_upper_bound=storage,
        storage_barrier_lower_bound=Q(1, 388800),
        post_event_euclidean_quotient_radius_squared_upper_bound=radius,
        identity_radius_squared=Q(1, 144),
        form_mean_increment=Q(2, 29) * q,
        phase_mean_increment=Q(0),
        linear_source_allowance_per_model=source,
        reading_error_per_model=delta,
        numerical_radius_ceiling_per_model=numerical,
        nominal_gap_strict_lower_bound=nominal_gap,
        conservative_recorded_gap_strict_lower_bound=conservative_gap,
        work_admitted=work[1] < Q(2, 10**6),
        identity_admitted=storage < Q(1, 388800) and radius < Q(1, 144),
        prospective_separation_sufficient=conservative_gap > 0,
    )


def _pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("bounds require two exact endpoints")
    lo, hi = map(exact_record, value)
    if lo > hi:
        raise ValueError("bounds must be ordered")
    return lo, hi


def assess_collective_prediction(
    *,
    full_model_bounds,
    comparator_bounds,
    full_numerical_radius,
    comparator_numerical_radius,
):
    """Assess independently established source-inclusive endpoint enclosures.

    This scalar consumer authenticates neither their source nor their numerical
    derivation. An evidence reader must reconstruct the consumed endpoints and
    error bounds from admitted primitive inputs and coefficient evidence first.
    """
    full, comparator = _pair(full_model_bounds), _pair(comparator_bounds)
    radii = tuple(
        map(exact_record, (full_numerical_radius, comparator_numerical_radius))
    )
    if any(radius < 0 for radius in radii):
        raise ValueError("numerical radii must be nonnegative")
    admission = collective_prediction_admission()
    delta = admission["reading_error_per_model"]
    budget = admission["numerical_radius_ceiling_per_model"]
    recorded_full = (full[0] - delta, full[1] + delta)
    recorded_comparator = (comparator[0] - delta, comparator[1] + delta)
    margin = recorded_full[0] - recorded_comparator[1]
    complete = all(radius <= budget for radius in radii)
    return dict(
        recorded_full_model_bounds=recorded_full,
        recorded_comparator_bounds=recorded_comparator,
        separation_margin=margin,
        numerical_budgets_met=complete,
        prediction_separates=complete and margin > 0,
        status=(
            "conditional_prediction_separates"
            if complete and margin > 0
            else "bounds_only"
        ),
    )

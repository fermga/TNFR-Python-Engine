"""Conditional form-resource bounds for finite reciprocal contact networks.

The complete law has symmetric nonnegative locally integrable edge weights in
both uniform quarter rows, held unit capacities and fixed gamma. The weights
may change without form resets. This detached instrument consumes all initial
form readings; it evaluates no source, phase response or trajectory.
Proof: RECIPROCAL_FORM_RESOURCE_ENVELOPE.md#reciprocal-form-resource-envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import ceil

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._organized_free_phase_probe import (
    _CHILD_SIZE,
    _INITIAL_SCALED_FORM,
    _free_probe_current_bounds,
)
from .relational_observations import _ordered
from .relational_sine_port_composition import _parameters

_MAX_INITIAL_NODES = 4096


@dataclass(frozen=True)
class _ReciprocalFormResourceBound:
    """Necessary contrast envelope conditional on a complete initial resource.

    ``initial_forms`` and the two marked groups retain input position labels.
    Sorted values are only the derivation of the centered cut resource; they
    never redefine those marks. The common error bound is coordinatewise,
    with no correlation restriction assumed. All nodes, including any source
    organizations, must occur in the input.

    ``scaled_contrast_upper_bound`` bounds gamma*|X_A-X_B| at every time.
    ``not_excluded`` is not a reachability or identity certificate. The optional
    node-count lower bound applies only to exactly uniform initial forms and
    uses the upper gamma enclosure; it is conservative, not an asserted sharp
    threshold. The implementation cap is separate from the finite-N theorem.
    ``current_coefficients`` records the lambda=1 endpoint used for the uniform
    majorant; it does not identify the unknown actual constitutive coefficient.
    """

    initial_forms: tuple[Q, ...]
    marked_a: tuple[int, ...]
    marked_b: tuple[int, ...]
    form_error_bound: Q
    node_count: int
    initial_form_mean: Q
    centered_forms: tuple[Q, ...]
    decreasing_centered_forms: tuple[Q, ...]
    centered_top_sums: tuple[Q, ...]
    initial_resource: Q
    resource_error_upper_bound: Q
    initial_resource_upper_bound: Q
    initial_marked_contrast: Q
    initial_marked_contrast_error_bound: Q
    current_coefficients: tuple[tuple[int, Q], ...]
    current_bound: Q
    gamma_bounds: I
    invariant_resource_upper_bound: Q
    scaled_contrast_upper_bound: Q
    target_margin: Q
    status: str
    necessary_node_count_lower_bound: int | None
    target_scaled_form: Q = _INITIAL_SCALED_FORM
    instrument_node_limit: int = _MAX_INITIAL_NODES
    interval_method: str = INTERVAL_METHOD
    conditional_on_complete_law_and_initial_bounds: bool = True
    scope: tuple[str, ...] = (
        "all_initial_forms_of_one_fixed_finite_labeled_network_are_supplied",
        "two_disjoint_nine_node_groups_are_fixed_by_input_positions",
        "symmetric_nonnegative_locally_integrable_weights_in_both_quarter_rows",
        "held_unit_capacities_fixed_gamma_and_structural_tau_clock",
        "uniform_over_the_retained_constitutive_family_zero_to_one",
        "no_form_inputs_form_jumps_node_birth_or_capacity_changes",
        "state_preserving_contact_changes_need_not_be_passive",
        "all_fine_forms_and_phases_remain_in_the_underlying_complete_model",
        "global_current_majorant_requires_no_phase_or_identity_tube",
        "initial_coordinate_error_bounds_are_supplied_not_authenticated",
        "support_work_is_separate_from_the_form_resource_envelope",
        "resource_bound_is_necessary_for_reachability_not_sufficient",
        "no_source_execution_formation_autonomy_or_physical_sensor_claim",
    )


def _marked_positions(raw, label, node_count):
    positions = _ordered(raw, label, limit=_CHILD_SIZE + 1)
    if len(positions) != _CHILD_SIZE:
        raise ValueError(f"{label} must contain exactly {_CHILD_SIZE} positions")
    if any(type(position) is not int for position in positions):
        raise TypeError(f"{label} positions must be ordinary Python integers")
    if len(set(positions)) != _CHILD_SIZE:
        raise ValueError(f"{label} positions must be distinct")
    if any(position < 0 or position >= node_count for position in positions):
        raise ValueError(f"{label} positions must index initial_forms")
    return positions


def _bound_reciprocal_form_resources(
    *, initial_forms, marked_a, marked_b, form_error_bound=0
) -> _ReciprocalFormResourceBound:
    """Rebuild a necessary marked contrast bound from primitive initial forms.

    Admit 18..4096 ordered exact or represented real readings and two ordered
    nine-element groups of distinct ordinary Python integer positions. The
    groups must be disjoint. A zero per-node error bound declares exact initial
    readings; no cached report, inferred missing source or phase is consumed.
    Iterables are read at most one entry beyond each declared count limit.

    For the centered decreasing values, let S_k be their first k sum and
    q=max_k 2*S_k/[k*(N-k)]. Coordinatewise error bounds of radius eps add
    exactly 4*eps/N to the worst possible q. The centered cut polytope of
    radius rho=max(q_upper,gamma*J0) is invariant under the declared law:
    every saturated cut has each outgoing form gap at least rho, and the
    bounded odd current cannot drive that facet outward. Any two marked
    nine-node means therefore differ by at most rho*(N-9). This uses no
    collective closure, contact schedule, storage reserve or trajectory.
    """
    raw = _ordered(initial_forms, "initial_forms", limit=_MAX_INITIAL_NODES + 1)
    node_count = len(raw)
    if not 2 * _CHILD_SIZE <= node_count <= _MAX_INITIAL_NODES:
        raise ValueError(
            f"initial_forms requires {2 * _CHILD_SIZE}..{_MAX_INITIAL_NODES} "
            "entries within this instrument's work limit"
        )
    forms = tuple(_exact(value, f"initial_forms[{i}]") for i, value in enumerate(raw))
    a = _marked_positions(marked_a, "marked_a", node_count)
    b = _marked_positions(marked_b, "marked_b", node_count)
    if set(a).intersection(b):
        raise ValueError("marked_a and marked_b must be disjoint")
    error = _exact(form_error_bound, "form_error_bound")
    if error < 0:
        raise ValueError("form_error_bound must be nonnegative")

    mean = sum(forms, Q(0)) / node_count
    centered = tuple(value - mean for value in forms)
    decreasing = tuple(sorted(centered, reverse=True))
    prefix, top_sums = Q(0), []
    for value in decreasing[:-1]:
        prefix += value
        top_sums.append(prefix)
    resource = max(
        2 * value / (k * (node_count - k)) for k, value in enumerate(top_sums, 1)
    )
    resource_error = 4 * error / node_count
    resource_upper = resource + resource_error
    coefficients, derivative_bounds = _free_probe_current_bounds(Q(1))
    current_bound = derivative_bounds[0]
    gamma, _ = _parameters(())
    invariant = max(resource_upper, gamma.hi * current_bound)
    upper = gamma.hi * invariant * (node_count - _CHILD_SIZE)
    margin = _INITIAL_SCALED_FORM - upper
    necessary_nodes = None
    if resource == 0 and error == 0:
        necessary_nodes = ceil(
            _CHILD_SIZE + _INITIAL_SCALED_FORM / (gamma.hi**2 * current_bound)
        )
    return _ReciprocalFormResourceBound(
        initial_forms=forms,
        marked_a=a,
        marked_b=b,
        form_error_bound=error,
        node_count=node_count,
        initial_form_mean=mean,
        centered_forms=centered,
        decreasing_centered_forms=decreasing,
        centered_top_sums=tuple(top_sums),
        initial_resource=resource,
        resource_error_upper_bound=resource_error,
        initial_resource_upper_bound=resource_upper,
        initial_marked_contrast=(sum(forms[i] for i in a) - sum(forms[i] for i in b))
        / _CHILD_SIZE,
        initial_marked_contrast_error_bound=2 * error,
        current_coefficients=coefficients,
        current_bound=current_bound,
        gamma_bounds=gamma,
        invariant_resource_upper_bound=invariant,
        scaled_contrast_upper_bound=upper,
        target_margin=margin,
        status="excluded_by_resource_bound" if margin > 0 else "not_excluded",
        necessary_node_count_lower_bound=necessary_nodes,
    )

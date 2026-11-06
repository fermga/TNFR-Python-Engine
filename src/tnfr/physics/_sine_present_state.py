"""Present relative state under the declared four-node quarter-sine family.

The exact linear observation owner supplies the anchor closure. This adapter
checks its common-family identities and realizes the nonlinear relative field
through the existing full field. Present initialization and the common held
coupling remain independent premises; no historical packet is decoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import scalar_affine_upper
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._interval_taylor import Jet
from ..mathematics.linear_observation import (
    LinearObservation,
    derive_linear_observation,
)
from ._sine_contact_information import _admit_contact_support_information
from ._sine_flow import _held_quarter_sine_field
from ._sine_history_interface import _INTERFACE_GAMMA_BOUNDS
from .relational_observations import _ordered

_PRESENT_SUPPORTS = (None, (0, 1), (0, 3))
_PRESENT_LIPSCHITZ = 2 * (1 + _INTERFACE_GAMMA_BOUNDS[1])
_PRESENT_ERROR_HORIZON = Q(1, 4)


@dataclass(frozen=True)
class _PreparedSinePresentObservation:
    """A verified common tangent observation, not acquired present data.

    anchor_observation is freshly derived from the old-support complete
    tangent at gamma=1/2 and the four primitive boundary outputs. The same
    row space intertwines every support at both coupling endpoints. Since
    the matrices depend affinely on gamma, these exact identities extend
    to the entire interval. Every common invariant row space retaining the
    outputs contains the minimal anchor space, which is itself common.

    relative_projection defines (a1,a2,a3,b1,b2,b3); relative_lift sets the
    representative reference coordinates to zero. The two basis-change
    matrices relate this canonical chart to the anchor's selected rows.
    The six-dimensional linear minimality concerns all-state exact output
    preservation. A nonlinear statement needs the separate derivative-map
    proof; neither statement is a finite-precision encoding lower bound.
    """

    anchor_observation: LinearObservation
    boundary_observation: tuple
    relative_projection: tuple
    relative_lift: tuple
    output_from_relative: tuple
    relative_to_anchor: tuple
    anchor_to_relative: tuple
    family_labels: tuple
    full_generators: tuple
    relative_generators: tuple
    anchor_reduced_generators: tuple
    exact_identity_checks: tuple[str, ...]
    dimension: int
    nodes: tuple[int, ...] = (0, 1, 2, 3)
    supports: tuple = _PRESENT_SUPPORTS
    gamma_bounds: tuple[Q, Q] = _INTERFACE_GAMMA_BOUNDS
    anchor_gamma: Q = Q(1, 2)
    full_state_dimension: int = 8
    boundary_coordinate_count: int = 4
    relative_coordinate_labels: tuple = (
        (1, "relative_form"),
        (2, "relative_form"),
        (3, "relative_form"),
        (1, "relative_phase"),
        (2, "relative_phase"),
        (3, "relative_phase"),
    )
    scope: tuple[str, ...] = (
        "old_and_two_unit_contact_supports_with_complete_joint_tangent",
        "existing_minimal_anchor_observation_and_exact_common_family_verification",
        "affine_gamma_dependence_extends_endpoint_intertwiners_to_whole_interval",
        "canonical_relative_state_removes_only_common_form_and_phase_shifts",
        "same_unknown_held_gamma_remains_an_associated_law_parameter",
        "exact_current_initialization_is_supplied_not_acquired_or_packet_decoded",
        "linear_minimality_is_not_nonlinear_or_finite_tolerance_minimality",
    )


@dataclass(frozen=True)
class _SinePresentInitializationBound:
    """Conditional error lifetime for two evolutions with the SAME gamma.

    initial_error_bound is a supplied infinity-norm radius in the six current
    relative coordinates. Both solutions follow the same prescribed support
    sequence. The bound covers arbitrary finite state-continuous switches
    inside the family, over the stated total duration. It includes no solver
    defect, new measurement, packet-reconstruction or coupling-mismatch error.
    The receiver factor uses equal conserved form means within each matched
    on/off comparison, so their common offset cancels.
    """

    initial_error_bound: Q
    elapsed_time: Q
    state_error_upper_bound: Q
    matched_receiver_error_upper_bound: Q
    lipschitz_bound: Q = _PRESENT_LIPSCHITZ
    horizon: Q = _PRESENT_ERROR_HORIZON
    gamma_bounds: tuple[Q, Q] = _INTERFACE_GAMMA_BOUNDS
    relative_coordinate_count: int = 6
    scope: tuple[str, ...] = (
        "supplied_current_relative_initialization_radius",
        "actual_and_reference_share_one_gamma_and_the_same_support_sequence",
        "uniform_parameter_family_not_one_unknown_gamma_point_forecast",
        "finite_prescribed_switches_are_state_continuous_without_error_reset",
        "relative_infinity_lipschitz_bound_from_complete_both_row_field",
        "receiver_excess_uses_shared_conserved_mean_and_two_branch_error_bounds",
        "no_numerical_solver_defect_or_new_current_measurement_supplied",
    )


def _prepared_present_support_matrices():
    """Obtain the three actual nodal generators from shared support admission."""
    contacts = []
    for contact in _PRESENT_SUPPORTS[1:]:
        contacts.append(
            _admit_contact_support_information(
                nodes=(0, 1, 2, 3),
                old_edges=((1, 2), (2, 3)),
                old_conductances=(Q(1), Q(1)),
                capacity=(Q(1),) * 4,
                contact=contact,
                contact_weight=Q(1),
                retained_nodes=contact,
                observation_nodes=(0,),
                elapsed_time=Q(0),
                mean_kind="arithmetic",
                reading_error_bound=Q(0),
                tolerance=Q(0),
            )
        )
    return (contacts[0].off_generator,) + tuple(p.on_generator for p in contacts)


def _prepared_present_joint_tangent(generator, gamma):
    """The full quarter-law blocks [[G,gamma*G],[-gamma*G,0]]."""
    size = len(generator)
    return tuple(
        tuple(
            (
                generator[i][j]
                if i < size and j < size
                else (
                    gamma * generator[i][j - size]
                    if i < size
                    else -gamma * generator[i - size][j] if j < size else Q(0)
                )
            )
            for j in range(2 * size)
        )
        for i in range(2 * size)
    )


def _prepared_sine_present_observation(
    *, max_rank_calls=4096
) -> _PreparedSinePresentObservation:
    """Derive one existing anchor closure and verify the common family.

    The fixed output is (a1,a3,b1,b3). No generic common-closure algorithm,
    external observation certificate or represented state is consumed.
    Invalid rank budgets are rejected before constructing the fixed model.
    Failure of a required exact identity raises instead of reporting closure.
    """
    if type(max_rank_calls) is not int:
        raise TypeError("max_rank_calls must be a positive non-boolean integer")
    if max_rank_calls < 1:
        raise ValueError("max_rank_calls must be positive")
    coordinates = (1, 2, 3, 5, 6, 7)
    relative = tuple(
        tuple(Q(j == i) - Q(j == (0 if i < 4 else 4)) for j in range(8))
        for i in coordinates
    )
    lift = tuple(tuple(Q(i == j) for j in coordinates) for i in range(8))
    boundary_indices = (0, 2, 3, 5)
    boundary = tuple(relative[i] for i in boundary_indices)
    output = tuple(tuple(Q(i == j) for j in range(6)) for i in boundary_indices)
    nodal_generators = _prepared_present_support_matrices()
    anchor = derive_linear_observation(
        _prepared_present_joint_tangent(nodal_generators[0], Q(1, 2)),
        boundary,
        max_rank_calls=max_rank_calls,
    )
    if anchor.dimension != 6:
        raise RuntimeError("declared present boundary does not yield six anchor rows")
    c, t = anchor.observation, anchor.right_inverse
    to_anchor = exact_matrix_product(c, lift)
    to_relative = exact_matrix_product(relative, t)
    identity = tuple(tuple(Q(i == j) for j in range(6)) for i in range(6))
    checks = {
        "R E=I6": exact_matrix_product(relative, lift) == identity,
        "O=Y R": exact_matrix_product(output, relative) == boundary,
        "C=(C E) R": exact_matrix_product(to_anchor, relative) == c,
        "R=(R T) C": exact_matrix_product(to_relative, c) == relative,
        "(C E)(R T)=I6": exact_matrix_product(to_anchor, to_relative) == identity,
        "(R T)(C E)=I6": exact_matrix_product(to_relative, to_anchor) == identity,
    }
    labels, full, reduced, anchor_reduced = [], [], [], []
    for support, nodal in zip(_PRESENT_SUPPORTS, nodal_generators):
        for gamma in _INTERFACE_GAMMA_BOUNDS:
            generator = _prepared_present_joint_tangent(nodal, gamma)
            cj = exact_matrix_product(c, generator)
            rj = exact_matrix_product(relative, generator)
            g = exact_matrix_product(cj, t)
            k = exact_matrix_product(rj, lift)
            label = (support, gamma)
            checks[f"C J=G C at {label}"] = exact_matrix_product(g, c) == cj
            checks[f"R J=K R at {label}"] = exact_matrix_product(k, relative) == rj
            checks[f"K=(R T) G (C E) at {label}"] = (
                exact_matrix_product(exact_matrix_product(to_relative, g), to_anchor)
                == k
            )
            labels.append(label)
            full.append(generator)
            reduced.append(k)
            anchor_reduced.append(g)
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"present-state common identities failed: {failed}")
    return _PreparedSinePresentObservation(
        anchor_observation=anchor,
        boundary_observation=boundary,
        relative_projection=relative,
        relative_lift=lift,
        output_from_relative=output,
        relative_to_anchor=to_anchor,
        anchor_to_relative=to_relative,
        family_labels=tuple(labels),
        full_generators=tuple(full),
        relative_generators=tuple(reduced),
        anchor_reduced_generators=tuple(anchor_reduced),
        exact_identity_checks=tuple(checks),
        dimension=anchor.dimension,
    )


def _held_sine_present_field(*, contact=None, gamma):
    """Compile six inherited relative rates for one supplied coupling member.

    contact is None, (0,1) or (0,3). Gamma is required and belongs to the
    declared interval; it is never filled by a nominal value. The returned
    field accepts exactly (a1,a2,a3,b1,b2,b3), including intervals or common-
    order jets. It lifts a representative, evaluates ALL original rates and
    subtracts the two reference rates. Reference node zero is not frozen in
    the physical dynamics. This function installs no event or integrator.
    """
    if contact is not None:
        contact = _ordered(contact, "contact", limit=3)
        if (
            len(contact) != 2
            or any(type(value) is not int for value in contact)
            or contact not in _PRESENT_SUPPORTS[1:]
        ):
            raise ValueError("contact must be None, (0,1) or (0,3)")
    gamma = exact_or_represented_real(gamma, "gamma")
    if not _INTERFACE_GAMMA_BOUNDS[0] <= gamma <= _INTERFACE_GAMMA_BOUNDS[1]:
        raise ValueError("gamma must lie in [49/100,51/100]")
    edges = ((1, 2, Q(1)), (2, 3, Q(1)))
    if contact is not None:
        edges += ((*contact, Q(1)),)
    full_field = _held_quarter_sine_field(
        capacity=(Q(1),) * 4, gamma=gamma, weighted_edges=edges
    )

    def field(state):
        values = _ordered(state, "relative state", limit=7)
        if len(values) != 6:
            raise ValueError("relative state must contain exactly six coordinates")
        jets = tuple(isinstance(value, Jet) for value in values)
        if any(jets):
            if not all(jets) or len({value.order for value in values}) != 1:
                raise ValueError("relative jets must have one common order")
            zero = Jet.constant(0, values[0].order)
        else:
            zero = Q(0)
        lifted = (zero, *values[:3], zero, *values[3:])
        rates = full_field(lifted)
        return tuple(rates[i] - rates[0] for i in (1, 2, 3)) + tuple(
            rates[i] - rates[4] for i in (5, 6, 7)
        )

    return field


def _bound_sine_present_initialization(
    *, initial_error_bound, elapsed_time
) -> _SinePresentInitializationBound:
    """Bound same-member initialization error without selecting unknown gamma.

    The relative field has global infinity-norm Lipschitz bound 151/50 for
    the fixed support/coupling family. A supplied radius rho consequently
    grows by at most exp(K*t); a matched receiver excess loses at most 3/2
    of that bound. The admitted total duration is [0,1/4]. No numerical
    evolution or truncation defect is silently included. Exact zero time
    retains rho, and zero radius stays zero, after complete input admission.
    """
    radius = exact_or_represented_real(initial_error_bound, "initial_error_bound")
    time = exact_or_represented_real(elapsed_time, "elapsed_time")
    if radius < 0 or not 0 <= time <= _PRESENT_ERROR_HORIZON:
        raise ValueError("initial radius must be nonnegative and time lie in [0,1/4]")
    error = (
        radius
        if not time or not radius
        else scalar_affine_upper(radius, _PRESENT_LIPSCHITZ, time, Q(0))
    )
    return _SinePresentInitializationBound(
        initial_error_bound=radius,
        elapsed_time=time,
        state_error_upper_bound=error,
        matched_receiver_error_upper_bound=Q(3, 2) * error,
    )

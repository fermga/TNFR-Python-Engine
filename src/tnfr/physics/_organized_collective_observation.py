"""Detached circular child means on marked C9 organizations.

The strict coherent collar is the observation domain in equations (27)-(28)
of ``theory/nodal/ORGANIZED_ELECTRICAL_REALIZATION.md``. Outward enclosures
can leave this domain unresolved; they never replace a missing angle by zero.
No source, law, support event or trajectory is certified by this observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    atan,
    cos,
    pi_interval,
    sin,
)
from ._sine_preparation import _prepared_coordinate_vector as _vector
from .relational_observations import _ordered


def _phase_box(value, label):
    """Admit immutable shared bounds or a finite exact/represented scalar."""
    return value if type(value) is I else I(_exact(value, label))


@dataclass(frozen=True)
class _OrganizedChildObservation:
    nodes: tuple[int, ...]
    form_mean: Q
    centered_form: tuple[Q, ...]
    rotated_resultant_real_bounds: I
    rotated_resultant_imaginary_bounds: I
    phase_anchor_lift_bounds: I | None
    phase_mean_lift_bounds: I | None
    phase_mean_cosine_bounds: I | None
    phase_mean_sine_bounds: I | None
    centered_phase_bounds: tuple[I, ...] | None
    collar_cosine_lower_bound: Q | None
    available: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class _OrganizedCollectiveObservation:
    form: tuple[Q, ...]
    phase_offset_bounds: tuple[I, ...]
    relative_phase_origin_bounds: I
    children: tuple[_OrganizedChildObservation, ...]
    available: bool
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_child_major_node_order_9_c_plus_j_for_six_marked_C9_children",
        "phase_offsets_relative_to_alpha_times_j_minus_four_with_alpha_two_pi_over_nine",
        "supplied_relative_phase_origin_added_only_to_the_sixth_child",
        "circular_mean_uses_resultant_anchor_then_arithmetic_mean_wrapped_residual",
        "lift_anchored_at_each_first_marked_node_not_a_global_principal_argument",
        "strict_collar_requires_all_residual_cosines_proved_positive",
        "outward_bounds_may_abstain_without_proving_an_actual_domain_failure",
        "no_source_formation_law_trajectory_or_physical_identification_certificate",
    )


@dataclass(frozen=True)
class _MarkedC9PairObservation:
    """Two independently observed marked children, without a temporal lift."""

    form: tuple[Q, ...]
    primitive_phase_bounds: tuple[I, ...]
    phase_offset_bounds: tuple[I, ...]
    children: tuple[_OrganizedChildObservation, ...]
    available: bool
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fixed_node_order_9_c_plus_j_for_two_marked_C9_children",
        "primitive_phases_demodulated_by_the_fixed_centered_winding_one_template",
        "arithmetic_collar_mean_reuses_the_shared_marked_child_observer",
        "snapshot_lifts_are_not_a_continuous_temporal_lift",
        "no_intersample_identity_retention_or_dynamical_law_is_certified",
    )


def _observe_marked_c9_pair(*, form, phase):
    """Read eighteen signed forms and primitive circular phase observations.

    Each phase may be an exact/represented scalar or an outward interval.
    Subtracting the fixed template ``2*pi*(j-4)/9`` exposes each child's
    coherent collar even though a raw winding-one resultant can vanish.
    No fine residual, common origin or dynamic reconstruction is supplied.
    All primitive values are admitted before any trigonometric evaluation.
    """
    values = _vector(form, "form", 18)
    raw = _ordered(phase, "phase", limit=19)
    if len(raw) != 18:
        raise ValueError("phase requires 18 registered coordinates")
    phases = tuple(_phase_box(value, f"phase[{i}]") for i, value in enumerate(raw))
    alpha = 2 * pi_interval() / 9
    offsets = tuple(value - alpha * (i % 9 - 4) for i, value in enumerate(phases))
    children = tuple(
        _observe_child(
            child=child,
            values=values[9 * child : 9 * child + 9],
            offsets=offsets[9 * child : 9 * child + 9],
            origin=I(0),
        )
        for child in range(2)
    )
    return _MarkedC9PairObservation(
        form=values,
        primitive_phase_bounds=phases,
        phase_offset_bounds=offsets,
        children=children,
        available=all(child.available for child in children),
    )


def _observe_child(*, child, values, offsets, origin):
    nodes = tuple(range(9 * child, 9 * child + 9))
    mean = sum(values, Q(0)) / 9
    # The origin is one common variable, so it cancels exactly here. Rotating
    # by the first marked phasor also avoids the global Arg display cut.
    relative = (I(0),) + tuple(value - offsets[0] for value in offsets[1:])
    real = sum((cos(value) for value in relative), I(0))
    imaginary = sum((sin(value) for value in relative), I(0))
    common = dict(
        nodes=nodes,
        form_mean=mean,
        centered_form=tuple(value - mean for value in values),
        rotated_resultant_real_bounds=real,
        rotated_resultant_imaginary_bounds=imaginary,
    )

    def unavailable(reason):
        return _OrganizedChildObservation(
            **common,
            phase_anchor_lift_bounds=None,
            phase_mean_lift_bounds=None,
            phase_mean_cosine_bounds=None,
            phase_mean_sine_bounds=None,
            centered_phase_bounds=None,
            collar_cosine_lower_bound=None,
            available=False,
            unavailable_reason=reason,
        )

    # A nonzero resultant lying strictly within pi/2 of its first phasor
    # necessarily has positive rotated real part. Failure to prove that
    # condition includes a zero resultant and an unresolved first residual.
    if real.lo <= 0:
        return unavailable("resultant_or_first_marked_collar_unresolved")
    anchor_relative = atan(imaginary / real)
    about_anchor = tuple(value - anchor_relative for value in relative)
    cosine = tuple(cos(value) for value in about_anchor)
    if any(value.lo <= 0 for value in cosine):
        return unavailable("strict_resultant_collar_unresolved")
    # Positive cosine identifies the unique wrapped residual in (-pi/2,pi/2).
    # It avoids guessing integer turns from a rounded representation of pi.
    residual = tuple(
        atan(sin(value) / real_part) for value, real_part in zip(about_anchor, cosine)
    )
    residual_mean = sum(residual, I(0)) / 9
    anchor = offsets[0] + origin + anchor_relative
    phase_mean = anchor + residual_mean
    return _OrganizedChildObservation(
        **common,
        phase_anchor_lift_bounds=anchor,
        phase_mean_lift_bounds=phase_mean,
        phase_mean_cosine_bounds=cos(phase_mean),
        phase_mean_sine_bounds=sin(phase_mean),
        centered_phase_bounds=tuple(value - residual_mean for value in residual),
        collar_cosine_lower_bound=min(value.lo for value in cosine),
        available=True,
        unavailable_reason=None,
    )


def _observe_organized_collective_observation(
    *, form, phase_offsets, relative_phase_origin
):
    """Observe supplied signed forms and marked circular phase coordinates.

    Exactly 54 ordered entries use node ``9*c+j``. The actual fine phase is
    ``alpha*(j-4) + phase_offsets[9*c+j]``, plus the supplied common origin
    on child five. Thus the reported collective phase is relative to the
    existing centered carrier template, including that actual donor origin.

    Forms use the shared exact signed-real coordinate admission. Phase data
    may additionally be immutable outward intervals, for example mathematical
    pi origins or phase boxes; no correlations besides the declared common
    origin and first-node subtraction are assumed. Unavailable child phases
    stay ``None`` while their independently available signed form means remain.

    The returned phase lift is anchored at each first marked coordinate.
    Independent full turns preserve its circular value and centered residuals,
    although an added first-node turn changes its displayed lift. It is not a
    continuous temporal lift without a separate continuity argument.
    """
    values = _vector(form, "form", 54)
    raw = _ordered(phase_offsets, "phase_offsets", limit=55)
    if len(raw) != 54:
        raise ValueError("phase_offsets requires 54 registered coordinates")
    offsets = tuple(
        _phase_box(value, f"phase_offsets[{index}]") for index, value in enumerate(raw)
    )
    origin = _phase_box(relative_phase_origin, "relative_phase_origin")
    children = tuple(
        _observe_child(
            child=child,
            values=values[9 * child : 9 * child + 9],
            offsets=offsets[9 * child : 9 * child + 9],
            origin=origin if child == 5 else I(0),
        )
        for child in range(6)
    )
    return _OrganizedCollectiveObservation(
        form=values,
        phase_offset_bounds=offsets,
        relative_phase_origin_bounds=origin,
        children=children,
        available=all(child.available for child in children),
    )

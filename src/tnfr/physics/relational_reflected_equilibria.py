"""Analytic regular equilibria and full-network stability on two joined C5s.

The named families specify exact equilibria, not a tolerance-based recognition
of supplied graphs. Interval coordinates enclose their ideal definitions.
Classification is complete only in the inherited reflection lift; stability
includes all twenty nodal coordinates, modulo two common offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I, cos, pi_interval
from ._relational_reflected_flow import evaluate_reflected_regular_flow
from .relational_reflected_transit import _lift_margins

__all__ = (
    "RelationalReflectedEquilibriumCertificate",
    "certify_relational_reflected_equilibrium",
)


def _opposite_polynomial(value):
    return 16 * value**3 - 8 * value + 1


@lru_cache(maxsize=1)
def _opposite_root_and_angle():
    """Isolate the unique root in (1/8,1/7), then enclose 2*acos(root).

    The polynomial is strictly decreasing on this interval. Angle bisection
    uses shared certified cosine, with no new inverse-trigonometric primitive.
    An unresolved comparison keeps the last valid bracket, never its midpoint.
    """
    lo, hi = Q(1, 8), Q(1, 7)
    for _ in range(96):
        middle = (lo + hi) / 2
        value = _opposite_polynomial(middle)
        if value > 0:
            lo = middle
        elif value < 0:
            hi = middle
        else:
            lo = hi = middle
            break
    root = I(lo, hi)
    pi = pi_interval()
    lower_angle, upper_angle = Q(0), Q(1, 2)
    for _ in range(96):
        middle = (lower_angle + upper_angle) / 2
        cosine = cos(pi * middle)
        if cosine.lo > root.hi:
            lower_angle = middle
        elif cosine.hi < root.lo:
            upper_angle = middle
        else:
            break
    return root, 2 * pi * I(lower_angle, upper_angle)


@dataclass(frozen=True)
class RelationalReflectedEquilibriumCertificate:
    family: str
    orientation: int
    model: RelationalExchangeModel
    coordinates: tuple[I, ...]
    root_cosine: I | None
    storage: I
    cycle_port_cosine: I
    cycle_path_cosine: I
    bridge_cosine: I
    domain_lower_bounds: tuple[Q, ...]
    field_rate_enclosures: tuple[I, ...]
    winding: tuple[int, int]
    phase_hessian_inertia: tuple[int, int, int]
    quotient_modes: tuple[int, int, int]
    stability: str
    scope: tuple[str, ...] = (
        "ideal_analytic_equilibrium_not_inferred_from_a_small_residual",
        "coordinate_box_encloses_one_named_ideal_point_not_a_box_of_equilibria",
        "two_unit_C5_rings_adjacent_matching_bridges_unit_held_capacity",
        "regular_unforced_reference_law_positive_epi_and_phase_weights",
        "complete_equilibrium_list_only_in_inherited_reflection_lift",
        "hessian_inertia_positive_negative_zero_full_ten_node_phase_space",
        "quotient_modes_stable_unstable_neutral_full_joint_eighteen_dimensional_quotient",
        "two_common_offsets_are_neutral_in_full_twenty_coordinate_state",
        "no_formation_global_regular_continuation_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.relational-reflected-equilibrium.v1",
            "report": _project(self),
        }


def certify_relational_reflected_equilibrium(family, *, model, orientation=1):
    """Enclose one of seven analytically selected regular equilibria.

    Families are consensus, aligned_twist, aligned_saddle and opposite_twist.
    The latter three have orientations +1 and -1. Positive edge stiffness
    proves full quotient recovery for the five minima. At either saddle an
    exact ring-exchange decomposition proves one negative phase direction;
    the symmetric quadratic eigenvalue pencil gives one growing real mode.
    No eigensolver, trajectory or projected graph supplies those verdicts.
    """
    if not isinstance(family, str) or family not in (
        "consensus",
        "aligned_twist",
        "aligned_saddle",
        "opposite_twist",
    ):
        raise ValueError("unknown analytic equilibrium family")
    if type(orientation) is not int or orientation not in (-1, 1):
        raise ValueError("orientation must be exactly +1 or -1")
    if family == "consensus" and orientation != 1:
        raise ValueError("consensus has one orientation-independent representative")
    if (
        not isinstance(model, RelationalExchangeModel)
        or model.phase_domain != "regular"
    ):
        raise ValueError("an explicit regular relational model is required")
    e, w = map(Q, model.effective_weights)
    if e <= 0:
        raise ValueError("the stability certificate requires positive epi_weight")
    beta, pi = Q(model.storage_scale), pi_interval()
    root = None
    if family == "consensus":
        a = capital_a = I(0)
        port = path = bridge = I(1)
        winding = (0, 0)
    elif family == "opposite_twist":
        root, angle = _opposite_root_and_angle()
        a, capital_a = orientation * angle, -orientation * angle
        port = bridge = 8 * root**4 - 8 * root**2 + 1
        path = root
        winding = (orientation, -orientation)
    else:
        saddle = family == "aligned_saddle"
        a = capital_a = orientation * pi * (Q(2, 3) if saddle else Q(4, 5))
        port = I(Q(-1, 2)) if saddle else cos(2 * pi / 5)
        path = I(Q(1, 2)) if saddle else port
        bridge = I(1)
        winding = (orientation, orientation)
    state = (I(0),) * 4 + (a, a / 2, capital_a, capital_a / 2)
    field = evaluate_reflected_regular_flow(
        state, epi_weight=e, phase_weight=w, storage_scale=beta
    )
    margins = field.resultant_margin_lower_bounds + _lift_margins(state)
    if min(margins) <= 0 or any(not rate.contains(0) for rate in field.rates):
        raise ArithmeticError(
            "analytic equilibrium enclosure failed its consistency checks"
        )
    saddle = family == "aligned_saddle"
    if not saddle and min(port.lo, path.lo, bridge.lo) <= 0:
        raise ArithmeticError("positive equilibrium stiffness enclosure is unresolved")
    return RelationalReflectedEquilibriumCertificate(
        family,
        orientation,
        model,
        state,
        root,
        field.storage,
        port,
        path,
        bridge,
        margins,
        field.rates,
        winding,
        (8, 1, 1) if saddle else (9, 0, 1),
        (17, 1, 0) if saddle else (18, 0, 0),
        (
            "hyperbolic_saddle_modulo_offsets"
            if saddle
            else "locally_exponentially_stable_modulo_offsets"
        ),
    )

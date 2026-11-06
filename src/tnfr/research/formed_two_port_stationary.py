"""Static rational instruments on the formed two-port eighteen-node support.

No flow, root search, trigonometric sign admission or branch enumeration is
performed. Circulation, realizable circular phase geometry and Hessian signs
have separate admission contracts; none selects a basin for the actual source.
The support and its original edge orientation come from the reference owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from .formed_two_port_reference import _ORIENTED_EDGES

__all__ = (
    "FormedTwoPortCirculation",
    "FormedTwoPortRationalPhases",
    "FormedTwoPortHessianCondensation",
    "assign_formed_two_port_circulation",
    "admit_formed_two_port_rational_phases",
    "condense_formed_two_port_hessian",
)

_PORTS = (
    _ORIENTED_EDGES[-2][0],
    _ORIENTED_EDGES[-1][0],
    _ORIENTED_EDGES[-2][1],
    _ORIENTED_EDGES[-1][1],
)
_SHORT = (
    _ORIENTED_EDGES.index(_PORTS[:2]),
    _ORIENTED_EDGES.index(_PORTS[2:]),
)
_BULK = tuple(
    tuple(i for i in range(start, start + 9) if i != short)
    for start, short in zip((0, 9), _SHORT, strict=True)
)


def _rational(value, label):
    if type(value) not in (int, Q):
        raise TypeError(f"{label} must be an exact integer or Fraction, not boolean")
    return exact_or_represented_real(value, label)


def _row(values, length, label):
    if type(values) not in (tuple, list):
        raise TypeError(f"{label} must be an ordered tuple or list")
    if len(values) != length:
        raise ValueError(f"{label} requires exactly {length} entries")
    return tuple(values)


def _divergence(currents):
    values = [Q(0)] * 18
    for (left, right), current in zip(_ORIENTED_EDGES, currents, strict=True):
        values[left] += current
        values[right] -= current
    return tuple(values)


@dataclass(frozen=True)
class FormedTwoPortCirculation:
    """A divergence-free current assignment, without admitted phase closure."""

    parameters: tuple[Q, Q, Q]
    edge_currents: tuple[Q, ...]
    nodal_divergence: tuple[Q, ...]
    scope: tuple[str, ...] = (
        "fixed_original_twenty_oriented_unit_edges",
        "all_five_current_magnitudes_at_most_one",
        "phase_closure_not_established",
        "no_stationary_phase_state_or_source_basin_selected",
    )


def assign_formed_two_port_circulation(p, q, j):
    """Assign bulk currents p,q and contact currents j,-j in original order.

    Short edges 4 and 13 carry p-j and q+j. The five independent bounds
    |p|,|q|,|j|,|p-j|,|q+j| <= 1 only admit sine-current magnitudes; unit
    phasors and cycle products are additional, independent requirements.
    """
    p, q, j = (_rational(value, name) for value, name in zip((p, q, j), "pqj"))
    if any(abs(value) > 1 for value in (p, q, j, p - j, q + j)):
        raise ValueError("the five current magnitudes must be at most one")
    currents = [p] * 9 + [q] * 9 + [j, -j]
    currents[_SHORT[0]], currents[_SHORT[1]] = p - j, q + j
    currents = tuple(currents)
    return FormedTwoPortCirculation((p, q, j), currents, _divergence(currents))


def _multiply(left, right):
    a, b = left
    c, d = right
    return a * c - b * d, a * d + b * c


def _conjugate(value):
    return value[0], -value[1]


def _product(values):
    result = (Q(1), Q(0))
    for value in values:
        result = _multiply(result, value)
    return result


@dataclass(frozen=True)
class FormedTwoPortRationalPhases:
    """Admitted rational edge/node phasors, with phase criticality checked.

    Node zero has unit phasor. No continuous lifts, winding at a cut, form
    row, acquired-source association or nonrational target are admitted.
    """

    edge_phasors: tuple[tuple[Q, Q], ...]
    node_phasors: tuple[tuple[Q, Q], ...]
    cycle_products: tuple[tuple[Q, Q], ...]
    nodal_divergence: tuple[Q, ...]
    phase_stationary: bool
    phase_potential: Q
    scope: tuple[str, ...] = (
        "supplied_rational_unit_phasors_and_three_exact_cycle_closures",
        "phase_stationary_iff_all_nodal_sine_currents_vanish",
        "full_A2_equilibrium_also_requires_uniform_form",
        "not_a_complete_stationary_catalog_or_source_basin_certificate",
    )


def admit_formed_two_port_rational_phases(edge_phasors):
    """Check twenty rational (cos,sin) pairs and all three cycle products.

    The cycle basis comprises the two oriented rings and the four-edge
    contact cycle 4->5->14->13->4. Unit edge phasors alone do not suffice.
    The resulting phase geometry may be nonstationary; inspect the separate
    phase_stationary flag. All checks use exact rational arithmetic.
    """
    phasors = tuple(
        tuple(_rational(value, f"phasor[{i}]") for value in _row(row, 2, "phasor"))
        for i, row in enumerate(_row(edge_phasors, 20, "edge phasors"))
    )
    if any(c * c + s * s != 1 for c, s in phasors):
        raise ValueError("each edge phasor must have exactly unit norm")
    cycles = (
        _product(phasors[:9]),
        _product(phasors[9:18]),
        _product(
            (
                phasors[_SHORT[0]],
                phasors[19],
                _conjugate(phasors[_SHORT[1]]),
                _conjugate(phasors[18]),
            )
        ),
    )
    if any(value != (Q(1), Q(0)) for value in cycles):
        raise ValueError("all three independent cycle products must equal one")
    nodes = {0: (Q(1), Q(0))}
    # Propagate on the fixed connected support, then check every edge. This
    # also verifies that the cycle-basis admission reconstructs a common state.
    while len(nodes) < 18:
        for (left, right), value in zip(_ORIENTED_EDGES, phasors, strict=True):
            if left in nodes and right not in nodes:
                nodes[right] = _multiply(nodes[left], value)
            elif right in nodes and left not in nodes:
                nodes[left] = _multiply(nodes[right], _conjugate(value))
    if any(
        _multiply(_conjugate(nodes[left]), nodes[right]) != value
        for (left, right), value in zip(_ORIENTED_EDGES, phasors, strict=True)
    ):
        raise ArithmeticError("cycle closure failed reconstruction on fixed support")
    divergence = _divergence(tuple(value[1] for value in phasors))
    return FormedTwoPortRationalPhases(
        phasors,
        tuple(nodes[i] for i in range(18)),
        cycles,
        divergence,
        not any(divergence),
        sum((1 - value[0] for value in phasors), Q(0)),
    )


@dataclass(frozen=True)
class FormedTwoPortHessianCondensation:
    """Exact static quadratic-form signs under the stated weight premises."""

    edge_cosines: tuple[Q, ...]
    port_order: tuple[int, ...]
    bulk_conductances: tuple[Q, Q]
    combined_short_weights: tuple[Q, Q]
    contact_weight: Q
    four_port_schur: tuple[tuple[Q, ...], ...]
    difference_matrix: tuple[tuple[Q, ...], ...]
    relative_positive_semidefinite: bool
    relative_positive_definite: bool
    scope: tuple[str, ...] = (
        "supplied_edge_coefficients_do_not_establish_phase_feasibility_or_stationarity",
        "sixteen_positive_bulk_weights_and_equal_positive_contact_weights",
        "fourteen_interior_coordinates_eliminated_by_exact_static_Schur_complement",
        "positive_definiteness_is_relative_to_the_single_common_phase_mode",
        "not_a_four_node_dynamic_closure_or_source_basin_certificate",
    )


def condense_formed_two_port_hessian(edge_cosines):
    """Condense supplied edge weights, allowing uneven positive long paths.

    Each eight-edge path has conductance 1/sum(1/c_i). Add its parallel short
    edge to obtain A_D,A_R. In port means m_D,m_R and differences d_D,d_R,
    the remaining form is 2*kappa*(m_R-m_D)^2 + d.T*M*d, where M has
    diagonal A_D+kappa/2,A_R+kappa/2 and off-diagonal -kappa/2.
    These identities require no stationary-state or acute-short-edge premise.
    """
    values = tuple(
        _rational(value, f"edge cosine[{i}]")
        for i, value in enumerate(_row(edge_cosines, 20, "edge cosines"))
    )
    if any(abs(value) > 1 for value in values):
        raise ValueError("edge cosine coefficients must lie in [-1,1]")
    if any(values[i] <= 0 for path in _BULK for i in path):
        raise ValueError("all sixteen bulk coefficients must be strictly positive")
    kappa = values[18]
    if kappa <= 0 or values[19] != kappa:
        raise ValueError("the two contact coefficients must be equal and positive")
    conductances = tuple(1 / sum((1 / values[i] for i in path), Q(0)) for path in _BULK)
    a, b = tuple(
        conductance + values[short]
        for conductance, short in zip(conductances, _SHORT, strict=True)
    )
    zero = Q(0)
    schur = (
        (a + kappa, -a, -kappa, zero),
        (-a, a + kappa, zero, -kappa),
        (-kappa, zero, b + kappa, -b),
        (zero, -kappa, -b, b + kappa),
    )
    reduced = ((a + kappa / 2, -kappa / 2), (-kappa / 2, b + kappa / 2))
    return FormedTwoPortHessianCondensation(
        values,
        _PORTS,
        conductances,
        (a, b),
        kappa,
        schur,
        reduced,
        exact_symmetric_semidefinite(reduced),
        exact_symmetric_semidefinite(reduced, strict=True),
    )

"""Exact local response jets on the declared ideal organized-contact carrier.

The finite algebra is Q[zeta_20][gamma,z,z^-1], z=exp(i*beta), with
Phi_20(t)=t^8-t^6+t^4-t^2+1 and i=zeta_20^5. It computes the complete
six-form/six-phase wheel restriction of the 54-node quarter-rate law.
The fixed order limit is twelve, stopping at the first nonzero donor row.
No floating point, trajectory, finite response or acquired source is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial

from .._exact_time import exact_or_represented_real as _exact
from ._sine_class_constitutive_comparison import _current_fourier_coefficients

_Key = tuple[int, int, int]
_Polynomial = dict[_Key, Q]
_Terms = tuple[tuple[int, int, int, Q], ...]
_MAX_ORDER = 12
_EDGES = tuple((c, (c + 1) % 5) for c in range(5)) + tuple((c, 5) for c in range(5))


@lru_cache(maxsize=20)
def _root_reduction(power: int) -> tuple[tuple[int, int], ...]:
    """Canonical Q-basis of a twentieth-root power (private integer input)."""
    exponent = power % 20
    coefficients = {exponent: 1}
    for degree in range(exponent, 7, -1):
        value = coefficients.pop(degree, 0)
        if value:
            for shift, sign in ((6, 1), (4, -1), (2, 1), (0, -1)):
                target = degree - 8 + shift
                coefficients[target] = coefficients.get(target, 0) + sign * value
    return tuple(
        sorted((degree, value) for degree, value in coefficients.items() if value)
    )


def _monomial(
    coefficient: Q = Q(1), *, gamma: int = 0, beta: int = 0, root: int = 0
) -> _Polynomial:
    return {
        (gamma, beta, degree): coefficient * scale
        for degree, scale in _root_reduction(root % 20)
        if coefficient * scale
    }


def _add(*polynomials: _Polynomial) -> _Polynomial:
    result: _Polynomial = {}
    for polynomial in polynomials:
        for key, value in polynomial.items():
            total = result.get(key, Q(0)) + value
            if total:
                result[key] = total
            else:
                result.pop(key, None)
    return result


def _scale(polynomial: _Polynomial, coefficient: Q) -> _Polynomial:
    return {
        key: value * coefficient for key, value in polynomial.items() if coefficient
    }


def _multiply(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    result: _Polynomial = {}
    for (ga, ba, ra), a in left.items():
        for (gb, bb, rb), b in right.items():
            value = a * b
            for root, scale in _root_reduction((ra + rb) % 20):
                key = ga + gb, ba + bb, root
                total = result.get(key, Q(0)) + value * scale
                if total:
                    result[key] = total
                else:
                    result.pop(key, None)
    return result


def _shift(
    polynomial: _Polynomial, *, gamma: int = 0, beta: int = 0, root: int = 0
) -> _Polynomial:
    return _multiply(polynomial, _monomial(gamma=gamma, beta=beta, root=root))


def _terms(polynomial: _Polynomial) -> _Terms:
    return tuple((*key, value) for key, value in sorted(polynomial.items()))


def _laplacian(values: list[_Polynomial]) -> list[_Polynomial]:
    result: list[_Polynomial] = [{} for _ in range(6)]
    for a, b in _EDGES:
        difference = _add(values[a], _scale(values[b], Q(-1)))
        result[a] = _add(result[a], difference)
        result[b] = _add(result[b], _scale(difference, Q(-1)))
    return result


def _initial_edge_phasor(
    a: int, b: int, harmonic: int, sign: int, origin_root_power: int | None
) -> _Polynomial:
    roots = tuple(4 * (c - 2) for c in range(5)) + (0,)
    root = sign * harmonic * (roots[b] - roots[a])
    beta = sign * harmonic * (int(b == 5) - int(a == 5))
    if origin_root_power is not None:
        root += beta * origin_root_power
        beta = 0
    return _monomial(beta=beta, root=root)


@dataclass(frozen=True)
class _ContactCancellationJet:
    """Exact coefficients of tau^k, including the separately labeled derivatives.

    A term (g,b,r,c) denotes c*gamma^g*exp(i*b*beta)*zeta_20^r.
    All r lie in 0..7. Empty tuples mean exactly zero in this algebra.
    Phase rows are displacements from the supplied initial phase origins.
    """

    deformation: Q
    origin_root_power: int | None
    computed_order: int
    first_nonzero_order: int | None
    donor_taylor_coefficients: tuple[_Terms, ...]
    donor_derivatives: tuple[_Terms, ...]
    form_taylor_coefficients: tuple[tuple[_Terms, ...], ...]
    phase_offset_taylor_coefficients: tuple[tuple[_Terms, ...], ...]
    all_preceding_donor_derivatives_zero: bool
    status: str
    maximum_order: int = _MAX_ORDER
    node_order: tuple[int, ...] = tuple(range(6))
    edges: tuple[tuple[int, int], ...] = _EDGES
    initial_form: tuple[Q, ...] = (Q(0),) * 6
    initial_host_phase_turns: tuple[Q, ...] = tuple(Q(c - 2, 5) for c in range(5))
    clock_factor: Q = Q(1, 4)
    coefficient_basis: str = (
        "Q[zeta20][gamma,z,z^-1]; z=exp(i*beta); Phi20=zeta20^8-zeta20^6+zeta20^4-zeta20^2+1"
    )
    gamma_definition: str = "gamma=1/(1023*pi)>0; tau=(1023/1024)*t"
    scope: tuple[str, ...] = (
        "ideal_twelve_coordinate_carrier_of_full_54_node_144_edge_law",
        "six_independent_forms_and_six_phase_offsets_retained_in_every_coefficient",
        "all_rim_and_contact_edges_include_reciprocal_reaction",
        "exact_coefficients_at_supplied_ideal_initial_state_not_a_reached_source",
        "first_nonzero_polynomial_requires_its_stated_origin_domain",
        "finite_jet_zeros_do_not_prove_an_invariant_sector",
        "no_trajectory_finite_horizon_response_fit_or_physical_identification",
    )


def _derive_contact_cancellation_jet(
    *, deformation: object, origin_root_power: object = None
) -> _ContactCancellationJet:
    """Compute only the declared law and generic or twentieth-root origin.

    The fixed twelve-order ceiling cannot be increased by the caller. Each
    branch stops on its first nonzero donor coefficient. Special roots are
    exact separately supplied registrations, not samples of a beta scan.
    """
    coefficient = _exact(deformation, "deformation")
    if coefficient not in (Q(0), Q(1)):
        raise ValueError("dynamic cancellation admits only deformation zero or one")
    if origin_root_power is not None and (
        type(origin_root_power) is not int or not 0 <= origin_root_power < 20
    ):
        raise ValueError(
            "origin_root_power must be None or an ordinary integer in 0..19"
        )
    root_power: int | None = origin_root_power  # type: ignore[assignment]
    currents = _current_fourier_coefficients(coefficient)
    forms: list[list[_Polynomial]] = [[{}] for _ in range(6)]
    phases: list[list[_Polynomial]] = [[{}] for _ in range(6)]
    exponentials = {
        (edge, harmonic, sign): [_initial_edge_phasor(a, b, harmonic, sign, root_power)]
        for edge, (a, b) in enumerate(_EDGES)
        for harmonic, _ in currents
        for sign in (1, -1)
    }
    first = None
    for order in range(_MAX_ORDER):
        force: list[_Polynomial] = [{} for _ in range(6)]
        for edge, (a, b) in enumerate(_EDGES):
            current: _Polynomial = {}
            for harmonic, weight in currents:
                pair = []
                for sign in (1, -1):
                    history = exponentials[edge, harmonic, sign]
                    if order:
                        convolution = _add(
                            *(
                                _scale(
                                    _multiply(
                                        _add(
                                            phases[b][degree],
                                            _scale(phases[a][degree], Q(-1)),
                                        ),
                                        history[order - degree],
                                    ),
                                    Q(degree),
                                )
                                for degree in range(1, order + 1)
                            )
                        )
                        history.append(
                            _scale(
                                _shift(convolution, root=5), Q(sign * harmonic, order)
                            )
                        )
                    pair.append(history[order])
                # 1/(2i)=-i/2. Harmonics are supplied independently by the law.
                sine = _scale(
                    _shift(_add(pair[0], _scale(pair[1], Q(-1))), root=5), Q(-1, 2)
                )
                current = _add(current, _scale(sine, weight))
            force[a] = _add(force[a], current)
            force[b] = _add(force[b], _scale(current, Q(-1)))
        laplacian = _laplacian([row[order] for row in forms])
        divisor = Q(1, 4 * (order + 1))
        for node in range(6):
            forms[node].append(
                _scale(
                    _add(_scale(laplacian[node], Q(-1)), _shift(force[node], gamma=1)),
                    divisor,
                )
            )
            phases[node].append(_scale(_shift(laplacian[node], gamma=1), divisor))
        if forms[5][-1]:
            first = order + 1
            break
    computed = len(forms[5]) - 1
    donor = tuple(_terms(polynomial) for polynomial in forms[5])
    derivatives = tuple(
        _terms(_scale(polynomial, Q(factorial(order))))
        for order, polynomial in enumerate(forms[5])
    )
    return _ContactCancellationJet(
        deformation=coefficient,
        origin_root_power=root_power,
        computed_order=computed,
        first_nonzero_order=first,
        donor_taylor_coefficients=donor,
        donor_derivatives=derivatives,
        form_taylor_coefficients=tuple(
            tuple(_terms(polynomial) for polynomial in row) for row in forms
        ),
        phase_offset_taylor_coefficients=tuple(
            tuple(_terms(polynomial) for polynomial in row) for row in phases
        ),
        all_preceding_donor_derivatives_zero=(
            all(not polynomial for polynomial in forms[5][:first])
            if first is not None
            else all(not polynomial for polynomial in forms[5])
        ),
        status=(
            "first_nonzero_donor_derivative"
            if first is not None
            else "unresolved_zero_jet_at_order_twelve"
        ),
    )


@dataclass(frozen=True)
class _ContactDynamicCancellation:
    deformation: Q
    generic: _ContactCancellationJet
    exceptional: _ContactCancellationJet | None
    protocol_revision: str = "173265a3"
    maximum_order: int = _MAX_ORDER
    protected_origins: str = "beta=k*pi/5; complete-law reflection proof required"
    exception_orbit: str = (
        "beta=(2k+1)*pi/10 is +/-pi/10 modulo 2*pi/5 for deformation one"
    )
    scope: tuple[str, ...] = (
        "generic_polynomial_then_one_symmetry_distinct_exceptional_origin",
        "reflection_sector_protection_belongs_to_complete_law_proof_not_finite_jet",
        "local_regeneration_does_not_replace_the_finite_actual_source_envelope",
    )


def _derive_organized_contact_dynamic_cancellation(
    *, deformation: object
) -> _ContactDynamicCancellation:
    """Derive the two bounded branches for one of the two frozen laws."""
    coefficient = _exact(deformation, "deformation")
    if coefficient not in (Q(0), Q(1)):
        raise ValueError("dynamic cancellation admits only deformation zero or one")
    generic = _derive_contact_cancellation_jet(deformation=coefficient)
    exceptional = (
        _derive_contact_cancellation_jet(deformation=coefficient, origin_root_power=1)
        if coefficient
        else None
    )
    return _ContactDynamicCancellation(coefficient, generic, exceptional)

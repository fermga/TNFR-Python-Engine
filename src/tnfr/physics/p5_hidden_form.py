"""Certified absolute output errors for odd-form omission on fixed unit P5.

The exact reflection means (a,p,c) retain their autonomous EPI diffusion.
Discarding (r,s) instead loses fine form, pressure and structural potential.
This observer bounds that loss without evolving a graph or adding a solver.
It does not bound relative pattern identity or unqualified fit/fallback xi.
Proof and scope: theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import networkx as nx

from .._exact_time import exact_or_represented_real as _rational
from ._cycle_algebra import Matrix, Vector, dot
from .geometry_realization import ExactPotentialGeometry, _potential_geometry
from .hybrid_operator_stability import _exact_matrix_product
from .p5_reduction import (
    P5ReducedState,
    P5ReductionGeometry,
    _diagonal,
    _matrix,
    _transpose,
    _vector,
    p5_reduction_geometry,
    reduce_p5_state,
)
from .reversible_eigenmode_reference import _negative_exp_bounds
from .structural_diffusion import _exact_inverse_norm_gap_lower_bound

__all__ = [
    "P5HiddenFormSample",
    "P5HiddenFormBound",
    "bound_p5_hidden_form",
]

# Evaluation resource ceiling, not a finite-horizon theorem hypothesis.
# The decay exponent is <= 2*capacity*time, within the shared exp limit 4096.
_MAX_SCALED_TIME = Fraction(2048)


@dataclass(frozen=True)
class P5HiddenFormSample:
    """Rational upper bounds for squared *absolute* errors at one model time.

    Hidden energy is the degree norm squared, without a factor of one half.
    Output bounds concern fresh model pressure and its exact inverse-square
    potential, not stored runtime fields. They include conservative spectral
    and exponential enclosure slack; binary64 execution defects are excluded.
    """

    time: Fraction
    decay_factor_upper: Fraction
    hidden_energy_upper: Fraction
    pressure_squared_upper: Vector
    potential_squared_upper: Vector

    def within_tolerances(self, *, pressure, potential) -> bool:
        """Sufficient nodewise absolute-error test, not a dynamical controller.

        False means these bounds do not establish the requested tolerances;
        it does not establish an actual violation. Tolerances specify a user
        observation budget and never enter the nodal law.
        """
        pressure = _rational(pressure, "pressure tolerance")
        potential = _rational(potential, "potential tolerance")
        if pressure < 0 or potential < 0:
            raise ValueError("output tolerances must be nonnegative")
        return bool(
            max(self.pressure_squared_upper) <= pressure**2
            and max(self.potential_squared_upper) <= potential**2
        )


@dataclass(frozen=True)
class P5HiddenFormBound:
    """Detached theorem evaluation for fixed support, not a runtime certificate.

    ``hidden_generator`` uses the positive convention ``z'=-A_hidden*z``.
    ``hidden_metric`` is induced by the degree metric, i.e. capacity times
    the existing reversible metric. ``decay_rate_lower`` is a proved rational
    lower bound on capacity*(1-1/sqrt(2)), not a fitted physical constant.
    Squared row factors multiply the same hidden-energy envelope. Initial
    signed errors mean full model minus its lifted reflection reduction.
    """

    reduced: P5ReducedState
    geometry: P5ReductionGeometry
    potential_geometry: ExactPotentialGeometry
    hidden_metric: Vector
    hidden_generator: Matrix
    decay_rate_lower: Fraction
    initial_hidden_energy: Fraction
    initial_pressure_error: Vector
    initial_potential_error: Vector
    pressure_squared_factors: Vector
    potential_squared_factors: Vector
    samples: tuple[P5HiddenFormSample, ...]
    exact_identity_checks: tuple[str, ...]
    scope: str


def bound_p5_hidden_form(initial_epi, *, times, capacity=1) -> P5HiddenFormBound:
    """Bound the loss from setting r=s=0 in fixed pure-EPI P5 diffusion.

    Domain: unit undirected path 0--1--2--3--4, unit structural edge lengths,
    constant positive common capacity, fresh pressure -L_rw*EPI, no forcing
    or hybrid events. Both full and reduced exact-real trajectories start
    from the supplied form and its reflection means, respectively. No state
    is evolved by this function. The even contrast u=c-p remains retained;
    this is not truncation of the separate two-output memory equation.

    Rational inputs remain exact; other admitted real scalars denote their
    finite binary64 materializations. Times are a nonempty ordered collection
    of nonnegative values. Each capacity*time must be <= 2048 for bounded
    rational exponential evaluation; the analytical theorem holds for all
    nonnegative times. No claim covers runtime rounding, relative identity,
    xi estimator continuity or evolving support, phase or capacity.
    """
    reduced = reduce_p5_state(initial_epi)
    geometry = p5_reduction_geometry(capacity)
    nu = geometry.capacity
    sample_times = _vector(times, "times")
    if any(time < 0 for time in sample_times):
        raise ValueError("times must be nonnegative")
    if any(nu * time > _MAX_SCALED_TIME for time in sample_times):
        raise ValueError("rational P5 evaluation requires capacity*time <= 2048")

    product = _exact_matrix_product
    lift = _matrix(((1, 0), (0, 1), (0, 0), (0, -1), (-1, 0)))
    degree_metric = tuple(nu * value for value in geometry.micro_metric)
    weighted_lift = product(_transpose(lift), _diagonal(degree_metric))
    gram = product(weighted_lift, lift)
    hidden_metric = tuple(gram[i][i] for i in range(2))
    generator_lift = product(geometry.micro_generator, lift)
    hidden_generator = generator_lift[:2]
    dissipation = product(weighted_lift, generator_lift)
    rate_lower = _exact_inverse_norm_gap_lower_bound(dissipation, gram)

    # Pressure is -A/nu, rather than the EPI rate -A. The potential uses
    # the shared path/length kernel, with no separately implemented geometry.
    pressure_rows = tuple(tuple(-value / nu for value in row) for row in generator_lift)
    path = nx.path_graph(5)
    nx.set_edge_attributes(path, 1, "length")
    potential_geometry = _potential_geometry(path, tuple(path.nodes))
    potential_rows = product(potential_geometry.kernel, pressure_rows)
    zero_macro_rows = _matrix(((0, 0),) * 3)
    if (
        gram != _diagonal(hidden_metric)
        or generator_lift != product(lift, hidden_generator)
        or dissipation != _transpose(dissipation)
        or rate_lower <= 0
        or rate_lower > nu
        or product(geometry.orbit_projection, pressure_rows) != zero_macro_rows
        or product(geometry.orbit_projection, potential_rows) != zero_macro_rows
    ):
        raise RuntimeError("fixed P5 hidden-form identities are inconsistent")

    def squared_factors(rows: Matrix) -> Vector:
        return tuple(
            sum(
                (
                    value**2 / weight
                    for value, weight in zip(row, hidden_metric, strict=True)
                ),
                Fraction(0),
            )
            for row in rows
        )

    pressure_factors = squared_factors(pressure_rows)
    potential_factors = squared_factors(potential_rows)
    hidden = reduced.discarded_epi[:2]
    energy = sum(
        (
            weight * value**2
            for weight, value in zip(hidden_metric, hidden, strict=True)
        ),
        Fraction(0),
    )
    factors = {
        time: _negative_exp_bounds(2 * rate_lower * time)[1]
        for time in dict.fromkeys(sample_times)
    }
    samples = tuple(
        P5HiddenFormSample(
            time=time,
            decay_factor_upper=factors[time],
            hidden_energy_upper=energy * factors[time],
            pressure_squared_upper=tuple(
                value * energy * factors[time] for value in pressure_factors
            ),
            potential_squared_upper=tuple(
                value * energy * factors[time] for value in potential_factors
            ),
        )
        for time in sample_times
    )
    return P5HiddenFormBound(
        reduced=reduced,
        geometry=geometry,
        potential_geometry=potential_geometry,
        hidden_metric=hidden_metric,
        hidden_generator=hidden_generator,
        decay_rate_lower=rate_lower,
        initial_hidden_energy=energy,
        initial_pressure_error=tuple(dot(row, hidden) for row in pressure_rows),
        initial_potential_error=tuple(dot(row, hidden) for row in potential_rows),
        pressure_squared_factors=pressure_factors,
        potential_squared_factors=potential_factors,
        samples=samples,
        exact_identity_checks=(
            "odd subspace invariant under the shared P5 generator",
            "positive rational decay rate certified in the degree metric",
            "orbit-mean pressure and potential errors vanish identically",
            "nodewise squared factors derived by metric Cauchy-Schwarz",
        ),
        scope=(
            "fixed unit P5 pure-EPI exact-real diffusion with constant positive "
            "common capacity; absolute fine pressure/potential omission bounds; "
            "no runtime rounding, relative identity or xi continuity certificate"
        ),
    )

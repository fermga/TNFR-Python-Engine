"""Optional finite-width observations of the live coherence pair population.

Widths and query radii are supplied observation policies. This readout neither
replaces the historical scalar xi_C fit nor certifies robustness in floating
arithmetic. The conditional perturbation theorem has a separate theory owner.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ..mathematics._exact_weighted import exact_weighted_sum_ratio
from ..mathematics._neighbor_differences import mean_neighbor_difference
from ._coherence_fit import CoherenceProfile
from ._helpers import finite_real_scalar, finite_real_series

__all__ = [
    "CoherenceResolution",
    "CoherenceResolutionPoint",
    "observe_coherence_resolution",
]


@dataclass(frozen=True)
class CoherenceResolutionPoint:
    """One normalized triangular-kernel observation and its actual support.

    The mass sums one weight per observed pair, including pairs from exact
    bins excluded by the historical fit. ``normalized_mass`` divides by the
    live profile's positive pair count; it is unavailable for an empty pair
    population. Pair counts and masses are not independent statistical samples.
    Empty kernel support has zero mass/numerator and unavailable moments.
    A positive weighted numerator that underflows is None with its own status;
    the independently normalized product can still be representable. A stored
    zero bin product retains the underlying profile's rounding interpretation.
    """

    radius: float
    available: bool
    status: str
    weight_mass: float
    normalized_mass: float | None
    contributing_pair_count: int
    product_numerator: float | None
    product_numerator_status: str
    product_mean: float | None
    endpoint_mean: float | None
    endpoint_second_moment: float | None
    source_mean: float | None
    target_mean: float | None
    source_second_moment: float | None
    target_second_moment: float | None
    centered_product: float | None
    endpoint_covariance: float | None
    semivariogram: float | None


@dataclass(frozen=True)
class CoherenceResolution:
    """Detached resolution report retaining the shared live profile provenance.

    ``profile`` retains all original bins, source/node associations, omitted
    pairs and geometry/numeric conventions. Each point uses the same supplied
    bandwidth. Its floating values are descriptive, not outward enclosures.
    """

    profile: CoherenceProfile
    bandwidth: float
    points: tuple[CoherenceResolutionPoint, ...]
    weighting: str
    interpretation: str


def _triangular_weight(distance: float, radius: float, bandwidth: float) -> float:
    separation = abs(distance - radius)
    if separation >= bandwidth:
        return 0.0
    # Subtract before division to retain small represented positive support
    # near the boundary. No division of a distant value by a tiny width occurs.
    return (bandwidth - separation) / bandwidth


def _observe_point(
    profile: CoherenceProfile, radius: float, bandwidth: float
) -> CoherenceResolutionPoint:
    weighted = tuple(
        (row, row.pair_count * weight)
        for row in profile.bins
        if (weight := _triangular_weight(row.distance, radius, bandwidth)) > 0.0
    )
    mass = math.fsum(weight for _, weight in weighted)
    products = tuple(row.product_mean for row, _ in weighted)
    weights = tuple(weight for _, weight in weighted)
    contributions = tuple(value * weight for value, weight in zip(products, weights))
    if any(
        value > 0.0 and contribution < sys.float_info.min
        for value, contribution in zip(products, contributions)
    ):
        exact_numerator, exact_denominator = exact_weighted_sum_ratio(
            weights, products, normalize=False
        )
        numerator = float(Fraction(exact_numerator, exact_denominator))
        if exact_numerator > 0 and numerator == 0.0:
            numerator = None
    else:
        numerator = math.fsum(contributions)
    normalized_mass = (
        mass / profile.positive_pair_count if profile.positive_pair_count else None
    )

    def mean(attribute):
        if not mass:
            return None
        # Normalize first, retaining the shared exact fallback for range-limited
        # contributions and signed cancellation. Dividing an already-underflowed
        # raw numerator by its mass cannot recover a representable mean.
        return mean_neighbor_difference(
            0.0, tuple(getattr(row, attribute) for row, _ in weighted), weights
        )

    source_mean = mean("source_mean")
    target_mean = mean("target_mean")
    # The covariance of a mixture includes variation in its component means.
    # Averaging bin covariances alone would discard this between-bin term.
    covariance = (
        mean_neighbor_difference(
            0.0,
            tuple(
                row.endpoint_covariance
                + (row.source_mean - source_mean) * (row.target_mean - target_mean)
                for row, _ in weighted
            ),
            weights,
        )
        if mass
        else None
    )
    return CoherenceResolutionPoint(
        radius=radius,
        available=mass > 0.0,
        status="observed" if mass else "no_positive_kernel_mass",
        weight_mass=mass,
        normalized_mass=normalized_mass,
        contributing_pair_count=sum(row.pair_count for row, _ in weighted),
        product_numerator=numerator,
        product_numerator_status=(
            "empty_support"
            if not mass
            else "positive_underflow" if numerator is None else "represented"
        ),
        product_mean=mean("product_mean"),
        endpoint_mean=mean("endpoint_mean"),
        endpoint_second_moment=mean("endpoint_second_moment"),
        source_mean=source_mean,
        target_mean=target_mean,
        source_second_moment=mean("source_second_moment"),
        target_second_moment=mean("target_second_moment"),
        centered_product=mean("centered_product"),
        endpoint_covariance=covariance,
        semivariogram=mean("semivariogram"),
    )


def observe_coherence_resolution(
    G: Any, *, radii: Any, bandwidth: Any
) -> CoherenceResolution:
    """Observe a live graph with explicitly supplied triangular distance widths.

    ``radii`` is a nonempty one-dimensional sequence of finite nonnegative
    represented reals; order and repeated queries are retained. ``bandwidth``
    is one finite strictly positive represented real in the same distance
    unit. Inputs are admitted before coercion. Scale radii and bandwidth with
    geometry when changing that unit.

    The shared canonical profile owner admits graph/state and supplies every
    sampled positive-distance pair, including singleton bins and bins beneath
    its fit floor. Zero-distance and unreachable pairs remain omitted with
    their counts. This function accepts no external cached profile and installs
    no pressure, phase, fitting or control law. A positive kernel mass means a
    finite observation is available; it supplies no robustness or quality flag.
    """
    width = finite_real_scalar(bandwidth, "coherence bandwidth")
    if width <= 0.0:
        raise ValueError("coherence bandwidth must be strictly positive")
    queries = tuple(
        float(value)
        for value in finite_real_series(
            radii, "coherence radii", nonnegative=True, nonempty=True
        )
    )
    from .canonical import observe_coherence_profile

    profile = observe_coherence_profile(G)
    return CoherenceResolution(
        profile=profile,
        bandwidth=width,
        points=tuple(_observe_point(profile, radius, width) for radius in queries),
        weighting="one triangular weight max(0, 1-abs(distance-radius)/bandwidth) per sampled positive-distance pair",
        interpretation="supplied observation resolution; descriptive binary64 moments, not a fitted length or certified perturbation bound",
    )

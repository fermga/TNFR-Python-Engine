r"""Exact local level-set geometry of the constitutive coherence kernel.

For signed local coordinates ``p=DeltaNFR`` and ``v=dEPI``, canonical
coherence is ``C=1/(1+|p|+|v|)``.  Its non-equilibrium level sets are L1
diamonds.  They are piecewise linear and fail to be differentiable at four
vertices, so the kernel does not by itself define a smooth Riemannian coherence
manifold.  It does define an exact L1 distance to the equilibrium point.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ._helpers import finite_real_scalar

__all__ = [
    "CoherenceLevelSetCertificate",
    "coherence_level_set_geometry",
]


@dataclass(frozen=True)
class CoherenceLevelSetCertificate:
    """Geometry forced by one value of the canonical coherence kernel."""

    coherence: float
    l1_radius: float
    euclidean_radius_minimum: float
    euclidean_radius_maximum: float
    intrinsic_dimension: int
    singular_points: tuple[tuple[float, float], ...]
    smooth_away_from_singular_points: bool
    is_globally_smooth_embedded_manifold: bool
    is_equilibrium_point: bool
    regular_gradient_available: bool
    regular_gradient_norm: float
    claim_status: str


def coherence_level_set_geometry(coherence: float) -> CoherenceLevelSetCertificate:
    r"""Return the exact ``C=c`` level geometry in the ``(DeltaNFR,dEPI)`` chart.

    For ``0<c<1``, ``|DeltaNFR|+|dEPI|=1/c-1``.  The Euclidean distance to the
    equilibrium therefore varies between ``r/sqrt(2)`` and ``r`` even though
    the L1 distance is exactly ``r``.  At ``c=1`` the level set collapses to the
    unique equilibrium point. There is no regular locus at that point, so
    ``regular_gradient_available`` is false and the retained numeric field is
    zero. ``c=0`` is attained only at unbounded pressure or velocity and is not
    a finite-state level set. A nonzero regular-gradient magnitude that is too
    small for binary64 is rejected instead of being reported as zero.
    """
    try:
        value = finite_real_scalar(coherence, "coherence")
    except ValueError as exc:
        raise ValueError("coherence must be a finite scalar in (0, 1]") from exc
    if not 0.0 < value <= 1.0:
        raise ValueError("coherence must be a finite scalar in (0, 1]")
    try:
        radius = (1.0 - value) / value
    except OverflowError as exc:
        raise ValueError("coherence level radius exceeds finite range") from exc
    if not math.isfinite(radius):
        raise ValueError("coherence level radius exceeds finite range")
    if radius == 0.0:
        singular = ((0.0, 0.0),)
        return CoherenceLevelSetCertificate(
            coherence=value,
            l1_radius=0.0,
            euclidean_radius_minimum=0.0,
            euclidean_radius_maximum=0.0,
            intrinsic_dimension=0,
            singular_points=singular,
            smooth_away_from_singular_points=False,
            is_globally_smooth_embedded_manifold=True,
            is_equilibrium_point=True,
            regular_gradient_available=False,
            regular_gradient_norm=0.0,
            claim_status="EXACT constitutive equilibrium level",
        )
    regular_gradient_norm = math.sqrt(2.0) * value * value
    if regular_gradient_norm == 0.0 or not math.isfinite(regular_gradient_norm):
        raise ValueError(
            "regular gradient norm is outside finite floating-point range"
        )
    singular = ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius))
    return CoherenceLevelSetCertificate(
        coherence=value,
        l1_radius=radius,
        euclidean_radius_minimum=radius / math.sqrt(2.0),
        euclidean_radius_maximum=radius,
        intrinsic_dimension=1,
        singular_points=singular,
        smooth_away_from_singular_points=True,
        is_globally_smooth_embedded_manifold=False,
        is_equilibrium_point=False,
        regular_gradient_available=True,
        regular_gradient_norm=regular_gradient_norm,
        claim_status=(
            "EXACT local kernel geometry; global network-state geometry and a "
            "preferred smooth metric remain OPEN"
        ),
    )

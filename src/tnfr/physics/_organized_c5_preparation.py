"""Shared preparation bounds for the supplied collective C5 matrix law.

``theory/nodal/ORGANIZED_EFFECTIVE_LAW.md`` owns the signed centered form U,
the centered continuous phase perturbation theta about winding +1 or -1,
and ``W_r=I+r*L**2`` on the fixed unit C5 support. The bounds below concern
that preparation and own-law storage only. Sensor storage, forcing, work,
identity retention and observation error remain with each consuming proof.
"""

from __future__ import annotations

from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real

__all__ = (
    "IDENTITY_RADIUS",
    "STORAGE_BARRIER",
    "prepared_c5_initial_energy_upper",
)

IDENTITY_RADIUS = Q(1, 10)
STORAGE_BARRIER = Q(1, 1000)


def prepared_c5_initial_energy_upper(
    *, response_upper: object, preparation_radius: object
) -> Q:
    """Bound initial excess storage for ``||U||+||theta||<=delta``.

    The actual response must satisfy ``0<=r<=response_upper``. Both supplied
    bounds are independently admitted as nonnegative exact or represented
    real scalars before arithmetic. This evaluates the conditional upper
    bound; it does not infer preparation membership or admit a phase collar.
    """
    response = exact_or_represented_real(response_upper, "response_upper")
    radius = exact_or_represented_real(preparation_radius, "preparation_radius")
    if response < 0 or radius < 0:
        raise ValueError(
            "response upper bound and preparation radius must be nonnegative"
        )
    return (4 + 32 * response) * radius**2

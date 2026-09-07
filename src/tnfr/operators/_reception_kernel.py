"""Compatibility names for the shared local neighbour-EPI kernel.

New code should import the operator-neutral names from
``tnfr.operators._neighbor_epi_kernel``.  These aliases retain the original EN
internal surface for downstream code while EN and RA share one implementation.
"""

from __future__ import annotations

from ._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_represented_affine_row,
    neighbor_epi_unweighted_mean,
)

__all__ = [
    "reception_blend_value",
    "reception_represented_affine_row",
    "reception_unweighted_mean",
]


reception_unweighted_mean = neighbor_epi_unweighted_mean
reception_blend_value = neighbor_epi_blend_value
reception_represented_affine_row = neighbor_epi_represented_affine_row

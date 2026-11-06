"""Detached numerical image of one fixed hard-projection Contraction branch.

Only binary64 input coordinates inside supplied finite endpoint boxes are
enclosed. The caller owns target binding, the common stage snapshot, live
grammar and policy admission, and the successor law. This helper executes no
graph event and proves neither identity preservation nor future stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .._binary64 import uses_ieee_binary64_rounding
from .._exact_time import represented_fraction
from ..operators._scale_operator_kernel import (
    ScaleOperatorProposal,
    propose_scale_operator,
)
from ..types import Glyph

__all__ = (
    "NulCoordinateBounds",
    "NulResetEnclosure",
    "NulCapacityResetEnclosure",
    "bound_nul_reset",
    "bound_nul_capacity_reset",
)


@dataclass(frozen=True, slots=True)
class NulCoordinateBounds:
    """Exact represented endpoints, retaining both nominal error directions."""

    lower: Fraction
    nominal: Fraction
    upper: Fraction

    @property
    def lower_error(self) -> Fraction:
        return self.nominal - self.lower

    @property
    def upper_error(self) -> Fraction:
        return self.upper - self.nominal

    @property
    def radius(self) -> Fraction:
        return max(self.lower_error, self.upper_error)


@dataclass(frozen=True, slots=True)
class NulResetEnclosure:
    """Numerical image only, conditional on a single fixed NUL branch.

    The three proposals pair lower/lower, nominal/nominal and upper/upper form
    and pressure inputs. These independent channels have monotone numerical
    maps, so the result encloses their complete Cartesian product, not merely
    those three pairs. Capacities and the supplied factor are held identical
    across that product. The retained proposals are detached numerical records;
    they do not certify live-state or whole-word admission.
    """

    factor: Fraction
    capacity_before: Fraction
    capacity_after: Fraction
    source_form: NulCoordinateBounds
    image_form: NulCoordinateBounds
    source_stored_pressure: NulCoordinateBounds
    image_stored_pressure: NulCoordinateBounds
    proposals: tuple[ScaleOperatorProposal, ...]

    @property
    def actual_capacity_ratio(self) -> Fraction:
        return self.capacity_after / self.capacity_before

    @property
    def endpoint_stored_product_defects(self) -> tuple[Fraction, ...]:
        """Exact endpoint product changes, not coefficient-only telemetry.

        These three diagnostic values are not an enclosure of product defects
        at all intermediate pressures. Constitutively refreshed rates require
        their own law and are not represented by these stored products.
        """
        return tuple(
            self.capacity_after * represented_fraction(proposal.dnfr_after, "p after")
            - self.capacity_before
            * represented_fraction(proposal.dnfr_before, "p before")
            for proposal in self.proposals
        )


@dataclass(frozen=True, slots=True)
class NulCapacityResetEnclosure:
    """Joint numerical image with actual rounded capacity endpoints.

    The three proposals pair the lower, nominal and upper coordinates. Each
    channel has its own monotone map, so their image box encloses the complete
    Cartesian input box, including correlated subsets. These detached records
    supply neither live eligibility nor a grammar or whole-word verdict.
    """

    factor: Fraction
    source_capacity: NulCoordinateBounds
    image_capacity: NulCoordinateBounds
    source_form: NulCoordinateBounds
    image_form: NulCoordinateBounds
    source_stored_pressure: NulCoordinateBounds
    image_stored_pressure: NulCoordinateBounds
    proposals: tuple[ScaleOperatorProposal, ...]

    @property
    def endpoint_capacity_ratios(self) -> tuple[Fraction, ...]:
        """Actual ratios at the three sampled capacity endpoints only."""
        return tuple(
            represented_fraction(proposal.vf_after, "capacity after")
            / represented_fraction(proposal.vf_before, "capacity before")
            for proposal in self.proposals
        )

    @property
    def endpoint_stored_product_defects(self) -> tuple[Fraction, ...]:
        """Exact diagonal diagnostics, not a Cartesian product-defect bound."""
        return tuple(
            represented_fraction(proposal.vf_after, "capacity after")
            * represented_fraction(proposal.dnfr_after, "pressure after")
            - represented_fraction(proposal.vf_before, "capacity before")
            * represented_fraction(proposal.dnfr_before, "pressure before")
            for proposal in self.proposals
        )


def _coordinate_bounds(values, label):
    if type(values) not in (tuple, list) or len(values) != 3:
        raise TypeError(f"{label} requires three ordered binary64 endpoints")
    exact = tuple(
        represented_fraction(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if not exact[0] <= exact[1] <= exact[2]:
        raise ValueError(f"{label} requires lower <= nominal <= upper")
    return NulCoordinateBounds(*exact)


def bound_nul_reset(
    *,
    factor: float,
    capacity: float,
    form_bounds: tuple[float, float, float],
    pressure_bounds: tuple[float, float, float],
    epi_min: float = -1.0,
    epi_max: float = 1.0,
) -> NulResetEnclosure:
    """Enclose the actual hard-clipped NUL map on a binary64 coordinate box.

    All scalars must be genuine finite floats with canonical positive zero.
    The shared proposal owner admits ``0 < factor < 1`` and the hard rails;
    positive represented capacity is required both before and after scaling.
    Fixed positive rounded multiplication and hard projection are monotone, so
    their endpoint images enclose every represented interior input, including
    saturation crossings and asymmetric rounding. No soft-clip or disabled
    edge-awareness branch is inferred from this result.

    The exact output fractions describe numerical image endpoints. They do
    not replace an ideal-real preparation family by rematerialized floats.
    Stored pressure is consumed for NUL's reciprocal proposal only; a later
    sine-law refresh must be rebuilt with the actual new capacities and forms.
    """
    exact_capacity = represented_fraction(capacity, "capacity")
    image = bound_nul_capacity_reset(
        factor=factor,
        capacity_bounds=(capacity, capacity, capacity),
        form_bounds=form_bounds,
        pressure_bounds=pressure_bounds,
        epi_min=epi_min,
        epi_max=epi_max,
    )
    return NulResetEnclosure(
        image.factor,
        exact_capacity,
        image.image_capacity.nominal,
        image.source_form,
        image.image_form,
        image.source_stored_pressure,
        image.image_stored_pressure,
        image.proposals,
    )


def bound_nul_capacity_reset(
    *,
    factor: float,
    capacity_bounds: tuple[float, float, float],
    form_bounds: tuple[float, float, float],
    pressure_bounds: tuple[float, float, float],
    epi_min: float = -1.0,
    epi_max: float = 1.0,
) -> NulCapacityResetEnclosure:
    """Enclose one actual NUL image with positive capacity uncertainty.

    All endpoints are genuine finite binary64 floats with positive zero.
    The capacity lower endpoint and its rounded image must be positive; an
    admissible nominal endpoint cannot hide a lower-member underflow. The
    factor, enabled edge awareness and hard rails are common to all members.
    Monotone rounded multiplication and hard projection give actual image
    endpoints, without substituting ideal factor-scaled error widths.
    """
    if not uses_ieee_binary64_rounding():
        raise ValueError("NUL image requires IEEE binary64 nearest-even rounding")
    exact_factor = represented_fraction(factor, "factor")
    source_capacity = _coordinate_bounds(capacity_bounds, "capacity bounds")
    represented_fraction(epi_min, "epi_min")
    represented_fraction(epi_max, "epi_max")
    if source_capacity.lower <= 0:
        raise ValueError("NUL image requires positive initial capacity lower endpoint")
    source_form = _coordinate_bounds(form_bounds, "form bounds")
    source_pressure = _coordinate_bounds(pressure_bounds, "pressure bounds")

    proposals = tuple(
        propose_scale_operator(
            glyph=Glyph.NUL,
            factor=factor,
            vf_before=capacity,
            dnfr_before=pressure,
            edge_aware_enabled=True,
            epi_before=form,
            epi_min=epi_min,
            epi_max=epi_max,
            clip_mode="hard",
        )
        for capacity, form, pressure in zip(
            capacity_bounds, form_bounds, pressure_bounds, strict=True
        )
    )
    image_capacity = _coordinate_bounds(
        tuple(proposal.vf_after for proposal in proposals), "capacity image"
    )
    if image_capacity.lower <= 0:
        raise ValueError("NUL image requires positive represented successor capacity")
    image_form = _coordinate_bounds(
        tuple(proposal.epi_after for proposal in proposals), "form image"
    )
    image_pressure = _coordinate_bounds(
        tuple(proposal.dnfr_after for proposal in proposals), "pressure image"
    )
    return NulCapacityResetEnclosure(
        exact_factor,
        source_capacity,
        image_capacity,
        source_form,
        image_form,
        source_pressure,
        image_pressure,
        proposals,
    )

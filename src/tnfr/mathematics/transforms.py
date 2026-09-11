"""Transform contracts and composite EPI regularity diagnostics.

The isometry helpers describe structural metric contracts that remain pending.
The implemented trend helper evaluates the unbounded composite EPI regularity
functional.  That diagnostic is separate from canonical TNFR structural
coherence ``C(t)`` and from its operator postconditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ..errors import TNFRValueError
from .epi import (
    BEPIElement,
    COMPOSITE_EPI_REGULARITY_KIND,
    COMPOSITE_EPI_REGULARITY_PROVENANCE,
)
from .unified_numerical import np

if TYPE_CHECKING:
    from .spaces import BanachSpaceEPI

logger = logging.getLogger(__name__)

__all__ = [
    "CompositeEPIRegularityTrendReport",
    "RegularityTrendViolation",
    "assess_composite_epi_regularity_trend",
    "IsometryFactory",
    "build_isometry_factory",
    "validate_norm_preservation",
    # Historical compatibility aliases; none denotes canonical C(t).
    "CoherenceMonotonicityReport",
    "CoherenceViolation",
    "ensure_coherence_monotonicity",
]


@runtime_checkable
class IsometryFactory(Protocol):
    """Callable creating isometric transforms aligned with TNFR semantics.

    Implementations produced by :func:`build_isometry_factory` must accept a
    structural basis and return a transform whose adjoint composes to identity
    on the relevant space.
    """

    def __call__(
        self,
        *,
        basis: Sequence[Sequence[complex]] | None = None,
        enforce_phase: bool = True,
    ) -> Callable[[Sequence[complex]], Sequence[complex]]:
        """Return an isometric transform for the provided basis."""


def build_isometry_factory(
    *,
    source_dimension: int,
    target_dimension: int,
    allow_expansion: bool = False,
) -> IsometryFactory:
    """Create a factory for constructing TNFR-aligned isometries.

    Parameters
    ----------
    source_dimension:
        Dimensionality of the input structural space.
    target_dimension:
        Dimensionality of the destination structural space.
    allow_expansion:
        Whether the isometry may expand into a higher-dimensional space while
        preserving the declared metric.
    """

    raise NotImplementedError(
        "Phase 2 will provide the canonical TNFR isometry factory; "
        "current stage only documents the expected contract."
    )


def validate_norm_preservation(
    transform: Callable[[Sequence[complex]], Sequence[complex]],
    *,
    probes: Iterable[Sequence[complex]],
    metric: Callable[[Sequence[complex]], float],
    atol: float = 1e-9,
) -> None:
    """Assert that a transform preserves the explicitly supplied metric.

    This pending generic contract does not identify the supplied metric with
    canonical structural coherence ``C(t)``.
    """

    raise NotImplementedError(
        "Norm preservation checks will be introduced in Phase 2; implementers "
        "should ensure transform(metric(state)) == metric(state) within atol."
    )


@dataclass(frozen=True)
class RegularityTrendViolation:
    """A decrease or forbidden plateau in a regularity trace."""

    index: int
    previous_value: float
    current_value: float
    tolerated_drop: float
    drop: float
    kind: str
    metric_kind: str = COMPOSITE_EPI_REGULARITY_KIND
    provenance: str = COMPOSITE_EPI_REGULARITY_PROVENANCE


@dataclass(frozen=True)
class CompositeEPIRegularityTrendReport:
    """Trend report for the unbounded composite EPI regularity functional."""

    regularity_values: tuple[float, ...]
    violations: tuple[RegularityTrendViolation, ...]
    allow_plateaus: bool
    tolerated_drop: float
    atol: float
    metric_kind: str = COMPOSITE_EPI_REGULARITY_KIND
    provenance: str = COMPOSITE_EPI_REGULARITY_PROVENANCE

    @property
    def is_monotonic(self) -> bool:
        """Return ``True`` when no trend violations were recorded."""

        return not self.violations

    @property
    def coherence_values(self) -> tuple[float, ...]:
        """Compatibility alias for :attr:`regularity_values`.

        The historical property name does not denote canonical ``C(t)``.
        """

        return self.regularity_values


def _as_regularity_values(
    regularity_series: Sequence[float | BEPIElement],
    *,
    space: "BanachSpaceEPI | None",
    regularity_kwargs: Mapping[str, float],
) -> tuple[float, ...]:
    if not regularity_series:
        raise TNFRValueError(
            "regularity_series must contain at least one entry.",
            context={"series_length": 0},
            suggestion="Provide a non-empty regularity sequence.",
        )

    first = regularity_series[0]
    if isinstance(first, BEPIElement):
        from .spaces import BanachSpaceEPI  # Local import avoids circular dependency.

        working_space = space if space is not None else BanachSpaceEPI()
        values = []
        for element in regularity_series:
            if not isinstance(element, BEPIElement):
                raise TypeError(
                    "All entries must be BEPIElement instances when the series "
                    "contains BEPI data."
                )
            value = working_space.composite_epi_regularity(
                element.f_continuous,
                element.a_discrete,
                x_grid=element.x_grid,
                **regularity_kwargs,
            )
            values.append(float(value))
        return tuple(values)

    values = []
    for value in regularity_series:
        if isinstance(value, BEPIElement):
            raise TypeError(
                "All entries must be numeric when the series is treated as "
                "regularity values."
            )
        numeric = float(value)
        if not np.isfinite(numeric):
            raise TNFRValueError(
                "Regularity values must be finite numbers.",
                context={"value": numeric},
                suggestion="Check for NaN or Inf values in the regularity series.",
            )
        values.append(numeric)
    return tuple(values)


def assess_composite_epi_regularity_trend(
    regularity_series: Sequence[float | BEPIElement],
    *,
    allow_plateaus: bool = True,
    tolerated_drop: float = 0.0,
    atol: float = 1e-9,
    space: "BanachSpaceEPI | None" = None,
    regularity_kwargs: Mapping[str, float] | None = None,
) -> CompositeEPIRegularityTrendReport:
    """Assess a nondecreasing composite EPI regularity trace.

    Numeric inputs are interpreted as already computed regularity values;
    :class:`BEPIElement` inputs are evaluated with
    :meth:`BanachSpaceEPI.composite_epi_regularity`.  The functional is
    unbounded, and an increase can be caused by additional derivative energy.
    Consequently this trend is descriptive and does not enforce, estimate, or
    certify canonical structural coherence ``C(t)``.
    """

    if not np.isfinite(tolerated_drop) or tolerated_drop < 0:
        raise TNFRValueError(
            "tolerated_drop must be finite and non-negative.",
            context={"tolerated_drop": tolerated_drop},
            suggestion="Provide a finite, non-negative tolerated_drop.",
        )
    if not np.isfinite(atol) or atol < 0:
        raise TNFRValueError(
            "atol must be finite and non-negative.",
            context={"atol": atol},
            suggestion="Provide a finite, non-negative atol.",
        )

    if regularity_kwargs is None:
        regularity_kwargs = {}

    values = _as_regularity_values(
        regularity_series,
        space=space,
        regularity_kwargs=regularity_kwargs,
    )
    violations: list[RegularityTrendViolation] = []

    for index in range(1, len(values)):
        previous_value = values[index - 1]
        current_value = values[index]
        drop = previous_value - current_value

        if current_value + tolerated_drop + atol < previous_value:
            violation = RegularityTrendViolation(
                index=index,
                previous_value=previous_value,
                current_value=current_value,
                tolerated_drop=tolerated_drop,
                drop=drop,
                kind="drop",
            )
            violations.append(violation)
            logger.warning(
                "Composite EPI regularity drop at step %s: "
                "previous=%s current=%s tolerated_drop=%s",
                index,
                previous_value,
                current_value,
                tolerated_drop,
            )
            continue

        if not allow_plateaus and current_value <= previous_value + atol:
            violation = RegularityTrendViolation(
                index=index,
                previous_value=previous_value,
                current_value=current_value,
                tolerated_drop=tolerated_drop,
                drop=max(0.0, drop),
                kind="plateau",
            )
            violations.append(violation)
            logger.warning(
                "Composite EPI regularity plateau at step %s: "
                "previous=%s current=%s",
                index,
                previous_value,
                current_value,
            )

    return CompositeEPIRegularityTrendReport(
        regularity_values=values,
        violations=tuple(violations),
        allow_plateaus=allow_plateaus,
        tolerated_drop=tolerated_drop,
        atol=atol,
    )


# Historical type aliases.  They now refer to explicitly named regularity data.
CoherenceViolation = RegularityTrendViolation
CoherenceMonotonicityReport = CompositeEPIRegularityTrendReport


def ensure_coherence_monotonicity(
    coherence_series: Sequence[float | BEPIElement],
    *,
    allow_plateaus: bool = True,
    tolerated_drop: float = 0.0,
    atol: float = 1e-9,
    space: "BanachSpaceEPI | None" = None,
    norm_kwargs: Mapping[str, float] | None = None,
) -> CompositeEPIRegularityTrendReport:
    """Compatibility alias for composite EPI regularity trend assessment.

    The historical name and parameter names remain callable, but this function
    does not inspect canonical ``C(t)`` or its history and does not interpret a
    rising regularity value as rising structural coherence.
    """

    return assess_composite_epi_regularity_trend(
        coherence_series,
        allow_plateaus=allow_plateaus,
        tolerated_drop=tolerated_drop,
        atol=atol,
        space=space,
        regularity_kwargs=norm_kwargs,
    )

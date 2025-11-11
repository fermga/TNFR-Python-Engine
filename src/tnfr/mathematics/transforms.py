"""Canonical transform contracts for TNFR coherence tooling.

This module intentionally provides *contracts* rather than concrete
implementations.  Phase 2 of the mathematics roadmap will plug the actual
algorithms into these helpers.  Until then, the functions below raise
``NotImplementedError`` with descriptive guidance so downstream modules know
which structural guarantees each helper must provide.

The three exposed contracts cover:

``build_isometry_factory``
    Expected to output callables that embed or project states while preserving
    the TNFR structural metric.  Implementations must return operators whose
    adjoint composes to identity inside the target Hilbert or Banach space so
    no coherence is lost during modal changes.

``validate_norm_preservation``
    Should perform diagnostic checks that a provided transform keeps the
    νf-aligned norm invariant (within tolerance) across representative states.
    Validation must surface informative errors so simulation pipelines can
    gate potentially destructive transforms before they act on an EPI.

``ensure_coherence_monotonicity``
    Designed to assert that a transform (or sequence thereof) does not break
    the monotonic coherence requirements captured in the repo-wide invariants.
    Implementations should report any drop in ``C(t)`` outside authorised
    dissonance windows and annotate the offending timestep to ease triage.
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
    Union,
    runtime_checkable,
)

import numpy as np

from .epi import BEPIElement

if TYPE_CHECKING:
    from .spaces import BanachSpaceEPI

logger = logging.getLogger(__name__)

__all__ = [
    "CoherenceMonotonicityReport",
    "CoherenceViolation",
    "IsometryFactory",
    "build_isometry_factory",
    "validate_norm_preservation",
    "ensure_coherence_monotonicity",
]


@runtime_checkable
class IsometryFactory(Protocol):
    """Callable creating isometric transforms aligned with TNFR semantics.

    Implementations produced by :func:`build_isometry_factory` must accept a
    structural basis (modal decomposition, eigenvectors, or similar spectral
    anchors) and return a transform that preserves both the vector norm and the
    encoded coherence structure.  The returned callable should accept the raw
    state data and emit the mapped state in the target representation while
    guaranteeing ``T* · T == I`` on the relevant space.
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
        Dimensionality of the destination structural space.  When the target
        dimension is larger than the source, implementations must specify how
        coherence is embedded without dilution.
    allow_expansion:
        Flag indicating whether the isometry may expand into a higher
        dimensional space (still norm-preserving via padding and phase guards).

    Returns
    -------
    IsometryFactory
        A callable that can produce concrete isometries on demand once a basis
        or spectral frame is available.
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
    """Assert that a transform preserves the TNFR structural norm.

    The validator should iterate through ``probes`` (representative EPI states)
    and confirm that applying ``transform`` leaves the provided ``metric``
    unchanged within ``atol``.  Any detected drift must be reported via
    exceptions that include the offending probe and the measured deviation so
    callers can attribute potential coherence loss to specific conditions.
    """

    raise NotImplementedError(
        "Norm preservation checks will be introduced in Phase 2; implementers "
        "should ensure transform(metric(state)) == metric(state) within atol."
    )


@dataclass(frozen=True)
class CoherenceViolation:
    """Details about a monotonicity violation detected in a coherence trace."""

    index: int
    previous_value: float
    current_value: float
    tolerated_drop: float
    drop: float
    kind: str


@dataclass(frozen=True)
class CoherenceMonotonicityReport:
    """Structured report generated by :func:`ensure_coherence_monotonicity`."""

    coherence_values: tuple[float, ...]
    violations: tuple[CoherenceViolation, ...]
    allow_plateaus: bool
    tolerated_drop: float
    atol: float

    @property
    def is_monotonic(self) -> bool:
        """Return ``True`` when no violations were recorded."""

        return not self.violations


def _as_coherence_values(
    coherence_series: Sequence[Union[float, BEPIElement]],
    *,
    space: "BanachSpaceEPI" | None,
    norm_kwargs: Mapping[str, float],
) -> tuple[float, ...]:
    if not coherence_series:
        raise ValueError("coherence_series must contain at least one entry.")

    first = coherence_series[0]
    if isinstance(first, BEPIElement):
        from .spaces import BanachSpaceEPI  # Local import to avoid circular dependency

        working_space = space if space is not None else BanachSpaceEPI()
        values = []
        for element in coherence_series:
            if not isinstance(element, BEPIElement):
                raise TypeError(
                    "All entries must be BEPIElement instances when the series contains BEPI data.",
                )
            value = working_space.coherence_norm(
                element.f_continuous,
                element.a_discrete,
                x_grid=element.x_grid,
                **norm_kwargs,
            )
            values.append(float(value))
        return tuple(values)

    values = []
    for value in coherence_series:
        if isinstance(value, BEPIElement):
            raise TypeError(
                "All entries must be numeric when the series is treated as coherence values.",
            )
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Coherence values must be finite numbers.")
        values.append(numeric)
    return tuple(values)


def ensure_coherence_monotonicity(
    coherence_series: Sequence[Union[float, BEPIElement]],
    *,
    allow_plateaus: bool = True,
    tolerated_drop: float = 0.0,
    atol: float = 1e-9,
    space: "BanachSpaceEPI" | None = None,
    norm_kwargs: Mapping[str, float] | None = None,
) -> CoherenceMonotonicityReport:
    """Validate monotonic behaviour of coherence measurements ``C(t)``.

    Parameters
    ----------
    coherence_series:
        Ordered sequence of coherence measurements (as floats) or
        :class:`BEPIElement` instances recorded after each transform
        application.
    allow_plateaus:
        When ``True`` the contract tolerates flat segments, otherwise every
        subsequent value must strictly increase.
    tolerated_drop:
        Maximum allowed temporary decrease in coherence, representing approved
        dissonance windows.  Values greater than zero should only appear when a
        higher-level scenario explicitly references controlled dissonance tests.

    Returns
    -------
    CoherenceMonotonicityReport
        Structured report describing the evaluated coherence trajectory and any
        detected violations.  Callers can inspect ``report.is_monotonic`` to
        determine whether the constraint holds.
    """

    if tolerated_drop < 0:
        raise ValueError("tolerated_drop must be non-negative.")
    if atol < 0:
        raise ValueError("atol must be non-negative.")

    if norm_kwargs is None:
        norm_kwargs = {}

    values = _as_coherence_values(coherence_series, space=space, norm_kwargs=norm_kwargs)

    violations: list[CoherenceViolation] = []

    for index in range(1, len(values)):
        previous_value = values[index - 1]
        current_value = values[index]
        drop = previous_value - current_value

        if current_value + tolerated_drop + atol < previous_value:
            violation = CoherenceViolation(
                index=index,
                previous_value=previous_value,
                current_value=current_value,
                tolerated_drop=tolerated_drop,
                drop=drop,
                kind="drop",
            )
            violations.append(violation)
            logger.warning(
                "Coherence drop detected at step %s: previous=%s current=%s tolerated_drop=%s",
                index,
                previous_value,
                current_value,
                tolerated_drop,
            )
            continue

        if not allow_plateaus and current_value <= previous_value + atol:
            violation = CoherenceViolation(
                index=index,
                previous_value=previous_value,
                current_value=current_value,
                tolerated_drop=tolerated_drop,
                drop=max(0.0, drop),
                kind="plateau",
            )
            violations.append(violation)
            logger.warning(
                "Coherence plateau detected at step %s: previous=%s current=%s",
                index,
                previous_value,
                current_value,
            )

    return CoherenceMonotonicityReport(
        coherence_values=values,
        violations=tuple(violations),
        allow_plateaus=allow_plateaus,
        tolerated_drop=tolerated_drop,
        atol=atol,
    )

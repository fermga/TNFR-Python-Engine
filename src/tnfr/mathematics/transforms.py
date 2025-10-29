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

from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable

__all__ = [
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


def ensure_coherence_monotonicity(
    coherence_series: Sequence[float],
    *,
    allow_plateaus: bool = True,
    tolerated_drop: float = 0.0,
) -> None:
    """Validate monotonic behaviour of coherence measurements ``C(t)``.

    Parameters
    ----------
    coherence_series:
        Ordered sequence of coherence measurements recorded after each
        transform application.
    allow_plateaus:
        When ``True`` the contract tolerates flat segments, otherwise every
        subsequent value must strictly increase.
    tolerated_drop:
        Maximum allowed temporary decrease in coherence, representing approved
        dissonance windows.  Values greater than zero should only appear when a
        higher-level scenario explicitly references controlled dissonance tests.

    Notes
    -----
    Implementations should raise a descriptive error pinpointing the timestep
    and values responsible for the monotonicity violation.  This information is
    required for Phase 2 telemetry pipelines to correlate failures with the
    structural operators in play.
    """

    raise NotImplementedError(
        "Coherence monotonicity enforcement is scheduled for Phase 2; current "
        "stage records the expected validation semantics."
    )


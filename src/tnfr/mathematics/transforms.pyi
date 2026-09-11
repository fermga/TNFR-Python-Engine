from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from .epi import BEPIElement
from .spaces import BanachSpaceEPI

__all__ = [
    "CompositeEPIRegularityTrendReport",
    "RegularityTrendViolation",
    "assess_composite_epi_regularity_trend",
    "CoherenceMonotonicityReport",
    "CoherenceViolation",
    "IsometryFactory",
    "build_isometry_factory",
    "validate_norm_preservation",
    "ensure_coherence_monotonicity",
]

class IsometryFactory(Protocol):
    def __call__(
        self,
        *,
        basis: Sequence[Sequence[complex]] | None = None,
        enforce_phase: bool = True,
    ) -> Callable[[Sequence[complex]], Sequence[complex]]: ...

def build_isometry_factory(
    *, source_dimension: int, target_dimension: int, allow_expansion: bool = False
) -> IsometryFactory: ...
def validate_norm_preservation(
    transform: Callable[[Sequence[complex]], Sequence[complex]],
    *,
    probes: Iterable[Sequence[complex]],
    metric: Callable[[Sequence[complex]], float],
    atol: float = 1e-09,
) -> None: ...

@dataclass(frozen=True)
class RegularityTrendViolation:
    index: int
    previous_value: float
    current_value: float
    tolerated_drop: float
    drop: float
    kind: str
    metric_kind: str = ...
    provenance: str = ...

@dataclass(frozen=True)
class CompositeEPIRegularityTrendReport:
    regularity_values: tuple[float, ...]
    violations: tuple[RegularityTrendViolation, ...]
    allow_plateaus: bool
    tolerated_drop: float
    atol: float
    metric_kind: str = ...
    provenance: str = ...
    @property
    def is_monotonic(self) -> bool: ...
    @property
    def coherence_values(self) -> tuple[float, ...]: ...

CoherenceViolation = RegularityTrendViolation
CoherenceMonotonicityReport = CompositeEPIRegularityTrendReport

def assess_composite_epi_regularity_trend(
    regularity_series: Sequence[float | BEPIElement],
    *,
    allow_plateaus: bool = True,
    tolerated_drop: float = 0.0,
    atol: float = 1e-09,
    space: BanachSpaceEPI | None = None,
    regularity_kwargs: Mapping[str, float] | None = None,
) -> CompositeEPIRegularityTrendReport: ...

def ensure_coherence_monotonicity(
    coherence_series: Sequence[float | BEPIElement],
    *,
    allow_plateaus: bool = True,
    tolerated_drop: float = 0.0,
    atol: float = 1e-09,
    space: BanachSpaceEPI | None = None,
    norm_kwargs: Mapping[str, float] | None = None,
) -> CompositeEPIRegularityTrendReport: ...

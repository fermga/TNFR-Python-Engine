from __future__ import annotations

import numpy as np
from .backend import MathematicsBackend
from .spaces import HilbertSpace
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Sequence

__all__ = ["MathematicalDynamicsEngine", "ContractiveDynamicsEngine"]

class TraceValue(NamedTuple):
    backend: Any
    numpy: complex

@dataclass
class MathematicalDynamicsEngine:
    generator: np.ndarray
    hilbert_space: HilbertSpace
    atol: float = ...
    backend: MathematicsBackend = field(init=False, repr=False)
    def __init__(
        self,
        generator: Sequence[Sequence[complex]] | np.ndarray | Any,
        hilbert_space: HilbertSpace,
        *,
        atol: float = 1e-09,
        use_scipy: bool | None = None,
        backend: MathematicsBackend | None = None,
    ) -> None: ...
    def step(
        self,
        state: Sequence[complex] | np.ndarray | Any,
        *,
        dt: float = 1.0,
        normalize: bool = True,
    ) -> Any: ...
    def evolve(
        self,
        state: Sequence[complex] | np.ndarray | Any,
        *,
        steps: int,
        dt: float = 1.0,
        normalize: bool = True,
    ) -> Any: ...

@dataclass
class ContractiveDynamicsEngine:
    generator: np.ndarray
    hilbert_space: HilbertSpace
    atol: float = ...
    backend: MathematicsBackend = field(init=False, repr=False)
    def __init__(
        self,
        generator: Sequence[Sequence[complex]] | np.ndarray | Any,
        hilbert_space: HilbertSpace,
        *,
        atol: float = 1e-09,
        ensure_contractive: bool = True,
        use_scipy: bool | None = None,
        backend: MathematicsBackend | None = None,
    ) -> None: ...
    def frobenius_norm(
        self,
        density: Sequence[Sequence[complex]] | np.ndarray | Any,
        *,
        center: bool = False,
    ) -> float: ...
    @property
    def last_contractivity_gap(self) -> float: ...
    def step(
        self,
        density: Sequence[Sequence[complex]] | np.ndarray | Any,
        *,
        dt: float = 1.0,
        normalize_trace: bool = True,
        enforce_contractivity: bool = True,
        raise_on_violation: bool = False,
        symmetrize: bool = True,
    ) -> Any: ...
    def evolve(
        self,
        density: Sequence[Sequence[complex]] | np.ndarray | Any,
        *,
        steps: int,
        dt: float = 1.0,
        normalize_trace: bool = True,
        enforce_contractivity: bool = True,
        raise_on_violation: bool = False,
        symmetrize: bool = True,
    ) -> Any: ...

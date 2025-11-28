from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Sequence

__all__ = ["build_delta_nfr", "build_lindblad_delta_nfr"]

def build_delta_nfr(
    dim: int,
    *,
    topology: str = "laplacian",
    nu_f: float = 1.0,
    scale: float = 1.0,
    rng: Generator | None = None,
) -> np.ndarray: ...
def build_lindblad_delta_nfr(
    *,
    hamiltonian: Sequence[Sequence[complex]] | np.ndarray | None = None,
    collapse_operators: Sequence[Sequence[Sequence[complex]] | np.ndarray] | None = None,
    dim: int | None = None,
    nu_f: float = 1.0,
    scale: float = 1.0,
    ensure_trace_preserving: bool = True,
    ensure_contractive: bool = True,
    atol: float = 1e-09,
) -> np.ndarray: ...

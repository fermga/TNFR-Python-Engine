"""Compatibility backend using the canonical fused NumPy computation.

The shared dynamics pipeline owns weight handling, neighborhood orientation,
state aliases and optional JIT dispatch. Keeping one implementation ensures
that ΔNFR remains pressure: νf is applied by nodal integration exactly once.
"""

from __future__ import annotations

from typing import Any, MutableMapping

from ..mathematics.unified_numerical import np
from ..types import TNFRGraph
from .numpy_backend import NumPyBackend


class OptimizedNumPyBackend(NumPyBackend):
    """Retain the optimized backend API over the shared NumPy kernels.

    Optimization and graph-local buffer reuse live in ``dynamics.dnfr``.
    No separate speedup over ``NumPyBackend`` is implied by this legacy name.
    """

    def __init__(self):
        self._np = np
        if self._np is None:
            raise RuntimeError("OptimizedNumPy backend requires numpy to be installed.")
        from ..dynamics.fused_dnfr import _NUMBA_AVAILABLE, _numba

        self._numba = _numba
        self._has_numba = _NUMBA_AVAILABLE
        self._workspace_cache: dict[tuple, Any] = {}

    @property
    def name(self) -> str:
        return "optimized_numpy"

    @property
    def supports_jit(self) -> bool:
        return self._has_numba

    def _get_workspace(self, size: int, dtype: Any) -> Any:
        """Retain the legacy scratch-buffer helper for existing callers."""
        key = (size, dtype)
        if key not in self._workspace_cache:
            self._workspace_cache[key] = self._np.empty(size, dtype=dtype)
        return self._workspace_cache[key][:size]

    def compute_delta_nfr(
        self,
        graph: TNFRGraph,
        *,
        cache_size: int | None = 1,
        n_jobs: int | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> None:
        """Compute pressure through the canonical pipeline for every graph size."""
        super().compute_delta_nfr(
            graph, cache_size=cache_size, n_jobs=n_jobs, profile=profile,
        )
        if profile is not None:
            profile["dnfr_optimization"] = "shared_canonical"

    def compute_si(
        self,
        graph: TNFRGraph,
        *,
        inplace: bool = True,
        n_jobs: int | None = None,
        chunk_size: int | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> dict[Any, float] | Any:
        """Compute Si through the shared normalization and phase pipeline."""
        result = super().compute_si(
            graph, inplace=inplace, n_jobs=n_jobs, chunk_size=chunk_size,
            profile=profile,
        )
        if profile is not None:
            profile["si_optimization"] = "shared_canonical"
        return result

    def _compute_delta_nfr_vectorized(
        self,
        graph: TNFRGraph,
        *,
        cache_size: int | None = 1,
        n_jobs: int | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> None:
        """Compatibility entry point using the shared kernel dispatcher."""
        self.compute_delta_nfr(
            graph, cache_size=cache_size, n_jobs=n_jobs, profile=profile,
        )

    def clear_cache(self) -> None:
        """Release backend-owned scratch buffers; graph caches belong to graphs."""
        self._workspace_cache.clear()

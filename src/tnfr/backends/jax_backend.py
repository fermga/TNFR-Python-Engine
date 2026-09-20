"""Experimental JAX adapter delegating graph computations to shared CPU owners.

JAX is an optional adapter dependency. These pressure and sense-index methods
do not use JAX arrays, JIT compilation, autodiff or GPU execution. Capabilities
refer to these exposed graph computations, not the installed JAX library.

Examples
--------
>>> from tnfr.backends import get_backend
>>> backend = get_backend("jax")  # doctest: +SKIP
>>> backend.supports_jit  # doctest: +SKIP
False
"""

from __future__ import annotations

from typing import Any, MutableMapping

from ..types import TNFRGraph
from . import TNFRBackend


class JAXBackend(TNFRBackend):
    """Experimental JAX adapter with shared CPU graph-kernel semantics.

    Attributes
    ----------
    name : str
        Returns "jax"
    supports_gpu : bool
        False; graph computations execute on the CPU
    supports_jit : bool
        False; these methods do not use JAX JIT

    Notes
    -----
    Requires JAX to be installed: `pip install jax jaxlib`
    """

    def __init__(self) -> None:
        """Initialize JAX backend."""
        try:
            import jax
            import jax.numpy as jnp

            self._jax = jax
            self._jnp = jnp
        except ImportError as exc:
            raise RuntimeError(
                "JAX backend requires jax to be installed. "
                "Install with: pip install jax jaxlib"
            ) from exc

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "jax"

    @property
    def supports_gpu(self) -> bool:
        """The currently exposed graph kernels execute on the CPU."""
        return False

    @property
    def supports_jit(self) -> bool:
        """JAX JIT is not used by the graph kernels."""
        return False

    def compute_delta_nfr(
        self,
        graph: TNFRGraph,
        *,
        cache_size: int | None = 1,
        n_jobs: int | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> None:
        """Compute pressure through the shared CPU dispatcher.

        All execution options are forwarded. Profiling identifies the JAX
        adapter separately from the actual CPU kernel and dispatch path.

        Parameters
        ----------
        graph : TNFRGraph
            NetworkX graph with TNFR node attributes
        cache_size : int or None, optional
            Forwarded cache-size hint
        n_jobs : int or None, optional
            Forwarded worker-count hint
        profile : MutableMapping[str, Any] or None, optional
            Mapping for timings and execution provenance
        """
        from ..dynamics.dnfr import default_compute_delta_nfr

        if profile is not None:
            profile["dnfr_backend"] = "jax"
            profile["dnfr_device"] = "cpu"
            profile["dnfr_implementation"] = "canonical"

        default_compute_delta_nfr(
            graph,
            cache_size=cache_size,
            n_jobs=n_jobs,
            profile=profile,
        )

    def compute_si(
        self,
        graph: TNFRGraph,
        *,
        inplace: bool = True,
        n_jobs: int | None = None,
        chunk_size: int | None = None,
        profile: MutableMapping[str, Any] | None = None,
    ) -> dict[Any, float] | Any:
        """Compute sense index through the shared CPU implementation.

        Parameters
        ----------
        graph : TNFRGraph
            NetworkX graph with TNFR node attributes
        inplace : bool, default=True
            Whether to write Si values back to graph
        n_jobs : int or None, optional
            Forwarded worker-count hint
        chunk_size : int or None, optional
            Forwarded chunk-size hint
        profile : MutableMapping[str, Any] or None, optional
            dict to collect timing metrics

        Returns
        -------
        dict[Any, float] or numpy.ndarray
            Node-to-Si mapping or array of Si values
        """
        from ..metrics.sense_index import compute_Si

        return compute_Si(
            graph,
            inplace=inplace,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            profile=profile,
        )

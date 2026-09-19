"""PyTorch adapter for the canonical TNFR pressure and sense-index kernels.

Graph computations delegate to the shared CPU implementations at every graph
size. PyTorch remains an optional adapter dependency; no independent tensor
pressure law or GPU execution is selected by the number of nodes.
"""

from __future__ import annotations

from typing import Any, MutableMapping

from ..types import TNFRGraph
from . import TNFRBackend


class TorchBackend(TNFRBackend):
    """Experimental PyTorch adapter with canonical graph-kernel semantics.

    The configured tensor device is retained for interface compatibility.
    Pressure and sense-index computations currently execute on the CPU.
    """

    def __init__(self) -> None:
        """Initialize the optional PyTorch adapter."""
        try:
            import torch

            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch backend requires torch to be installed. "
                "Install with: pip install torch"
            ) from exc

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "torch"

    @property
    def supports_gpu(self) -> bool:
        """The currently exposed graph kernels execute on the CPU."""
        return False

    @property
    def supports_jit(self) -> bool:
        """TorchScript is not used by the graph kernels."""
        return False

    @property
    def device(self) -> Any:
        """Return the configured tensor device, not a graph-kernel placement."""
        return self._device

    def compute_delta_nfr(
        self,
        graph: TNFRGraph,
        *,
        cache_size: int | None = 1,
        n_jobs: int | None = None,
        profile: MutableMapping[str, float] | None = None,
    ) -> None:
        """Compute structural pressure through the canonical dispatcher.

        The shared kernel preserves outgoing effective conductance, pressure
        aliases, circular phase means and channel weights. Structural capacity
        belongs to EPI' = vf * pressure and is not multiplied into pressure a
        second time. Graph size does not select a different physical law.

        All execution options are forwarded to default_compute_delta_nfr.
        Profiling reports this adapter separately from the actual CPU kernel
        device and the canonical dispatch path. The configured tensor device
        does not imply GPU execution.
        """
        from ..dynamics.dnfr import default_compute_delta_nfr

        if profile is not None:
            profile["dnfr_backend"] = "torch"
            profile["dnfr_device"] = "cpu"
            profile["dnfr_implementation"] = "canonical"

        default_compute_delta_nfr(
            graph, cache_size=cache_size, n_jobs=n_jobs, profile=profile
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

        The inplace, worker, chunk-size and profiling options are forwarded
        unchanged to compute_Si.
        """
        from ..metrics.sense_index import compute_Si

        return compute_Si(
            graph,
            inplace=inplace,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            profile=profile,
        )

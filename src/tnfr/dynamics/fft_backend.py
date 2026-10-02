"""Common interfaces for TNFR FFT backends.

The spectral factorization roadmap requires multiple FFT implementations
(CPU, GPU, distributed, partition-aware).  This module defines the minimal
interface that any backend must provide so higher-level code can remain
agnostic about execution details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .advanced_fft_arithmetic import FFTArithmeticResult, SpectralState


@dataclass(frozen=True)
class FFTBackendCapabilities:
    """Describe resource and feature limits for a backend."""

    backend_name: str
    max_nodes: int | None = None
    precision: str = "float64"
    supports_distributed: bool = False
    extra: Mapping[str, Any] | None = None


class FFTBackend(Protocol):
    """Protocol implemented by FFT engines used across TNFR."""

    backend_name: str

    def get_capabilities(self) -> FFTBackendCapabilities:
        """Return static capability metadata used for planning."""

    def get_spectral_state(
        self, G: Any, force_recompute: bool = False
    ) -> "SpectralState":
        """Return the spectral decomposition of ``G``."""

    def spectral_convolution(
        self,
        G: Any,
        signal1: Any | None = None,
        signal2: Any | None = None,
        operation: str = "multiply",
    ) -> "FFTArithmeticResult":
        """Perform spectral convolution or related operations."""

    # Additional spectral convenience methods are optional, but concrete engines
    # usually expose harmonic analysis/filtering helpers as well.  The protocol
    # focuses on the fundamental routines required by factorization code.

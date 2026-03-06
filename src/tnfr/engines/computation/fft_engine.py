"""FFT dynamics engine wrapper.

This module re-exports the canonical implementation from
``tnfr.dynamics.fft_engine`` so that legacy imports from the
``tnfr.engines.computation`` namespace remain valid without duplicating code.
"""

from ...dynamics.fft_engine import (
    FFTDynamicsState,
    FFTDynamicsEngine,
    create_fft_engine,
    run_fft_optimized_simulation,
)

__all__ = [
    "FFTDynamicsState",
    "FFTDynamicsEngine",
    "create_fft_engine",
    "run_fft_optimized_simulation",
]

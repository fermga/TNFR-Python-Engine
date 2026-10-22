"""Standalone arithmetic-pressure primality compatibility package.

The pressure zero set characterizes primes for n>=2 when exact divisor and
factor statistics and positive coefficients are supplied. Runtime uses
floating arithmetic and a declared zero tolerance. Unit defaults are a
normalization choice, not coefficients derived from phi, gamma, pi or e.
Static EPI/capacity summaries do not supply phase or an autonomous NFR law.
See the subproject README and theory/TNFR_NUMBER_THEORY.md."""

from .core import (
    tnfr_component_breakdown,
    tnfr_delta_nfr,
    tnfr_is_prime,
    tnfr_structural_triad,
)
from .optimized import OptimizedTNFRPrimality

__version__ = "1.1.0"
__author__ = "F. F. Martinez Gamo"
__license__ = "MIT"
__doi__ = "10.5281/zenodo.17764749"

__all__ = [
    "tnfr_is_prime",
    "tnfr_delta_nfr",
    "tnfr_component_breakdown",
    "tnfr_structural_triad",
    "OptimizedTNFRPrimality",
]

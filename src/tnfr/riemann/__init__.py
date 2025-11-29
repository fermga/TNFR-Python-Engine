"""Experimental TNFR–Riemann utilities.

This module provides a tiny, non-canonical sandbox for exploring
operator prototypes inspired by the TNFR–Riemann research notes.

Everything here is strictly experimental and not part of the
stable public API.
"""

from .operator import build_prime_path_graph, build_h_tnfr

__all__ = [
    "build_prime_path_graph",
    "build_h_tnfr",
]

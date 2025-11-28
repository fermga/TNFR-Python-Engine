"""Network analysis utilities for TNFR structural operators.

This module provides network-level analysis tools for structural operators,
particularly for detecting and analyzing emission sources, phase compatibility,
and coherence flow patterns.

TNFR Context
------------
According to TNFR.pdf §2.2.1 (EN - Recepción estructural), Reception (EN) is
not passive absorption but active reorganization that requires:

- Detection of compatible emission sources in the network
- Phase compatibility validation (θᵢ ≈ θⱼ for coupling)
- Integration efficiency measurement (coherence received vs. integrated)
- Source traceability (which nodes contribute to EPI)

This module implements these capabilities for the Reception operator and other
operators that require network-level coherence analysis.
"""

from __future__ import annotations

__all__ = [
    "detect_emission_sources",
]

from .source_detection import detect_emission_sources

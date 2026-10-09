"""Compatibility layer for legacy sequencing APIs.

The canonical implementations now live under ``tnfr.operators.patterns``.
This package forwards imports so external callers relying on the historical
``tnfr.sequencing`` namespace continue to operate without modification.
"""

from .patterns import AdvancedPatternDetector

__all__ = ["AdvancedPatternDetector"]

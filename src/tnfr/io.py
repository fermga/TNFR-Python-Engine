"""Legacy JSON utilities module.

The backwards-compatible re-export was removed; use :mod:`tnfr.utils.io`
directly. Importing :mod:`tnfr.io` now raises an :class:`ImportError` with a
clear migration hint.
"""

from __future__ import annotations

raise ImportError("`tnfr.io` was removed. Import helpers from `tnfr.utils.io` instead.")

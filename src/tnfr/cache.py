"""Legacy cache helpers module.

This compatibility shim was removed in favour of :mod:`tnfr.utils.cache`.
Importing :mod:`tnfr.cache` now fails with a clear message so that callers
update their imports instead of relying on the removed re-export behaviour.
"""

from __future__ import annotations

raise ImportError(
    "`tnfr.cache` was removed. Import helpers from `tnfr.utils.cache` instead."
)

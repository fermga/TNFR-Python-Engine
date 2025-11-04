"""Legacy cache helpers module.

This compatibility shim was removed in favour of :mod:`tnfr.utils.cache`.
Importing :mod:`tnfr.cache` now fails with a clear message so that callers
update their imports instead of relying on the removed re-export behaviour.

Notes
-----
Any replacement graph wrapper interacting with the modern cache utilities
must expose ``nodes``, ``neighbors``, ``number_of_nodes`` and the ``.graph``
metadata mapping as described by :class:`tnfr.types.GraphLike`.  These
attributes are consumed by cache invalidation routines when reconstructing the
canonical coherence state for TNFR graphs.
"""

from __future__ import annotations

raise ImportError(
    "`tnfr.cache` was removed. Import helpers from `tnfr.utils.cache` instead."
)

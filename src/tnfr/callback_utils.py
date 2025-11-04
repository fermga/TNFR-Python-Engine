"""Legacy callback utilities module.

This compatibility shim redirects imports to :mod:`tnfr.utils.callbacks`.
All callback utilities are now centralized in the utils package for better
organization and discoverability.

**Migration Guide**::

    # Old imports (still work via this shim)
    from tnfr.callback_utils import CallbackEvent, CallbackManager, callback_manager

    # New imports (recommended)
    from tnfr.utils import CallbackEvent, CallbackManager, callback_manager
    # or
    from tnfr.utils.callbacks import CallbackEvent, CallbackManager, callback_manager

This module will be removed in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'tnfr.callback_utils' is deprecated. "
    "Use 'from tnfr.utils import CallbackEvent, CallbackManager, callback_manager' instead. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from .utils.callbacks import (
    CallbackError,
    CallbackEvent,
    CallbackManager,
    callback_manager,
    CallbackSpec,
    _normalize_callbacks,
    _normalize_callback_entry,
)

__all__ = (
    "CallbackEvent",
    "CallbackManager",
    "callback_manager",
    "CallbackError",
    "CallbackSpec",
    "_normalize_callbacks",
    "_normalize_callback_entry",
)

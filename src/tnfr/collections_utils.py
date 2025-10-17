"""Compatibility wrapper for collection helpers moved to :mod:`tnfr.utils`."""

from __future__ import annotations

import warnings

from .utils.data import (
    MAX_MATERIALIZE_DEFAULT,
    STRING_TYPES,
    ensure_collection,
    flatten_structure,
    is_non_string_sequence,
    mix_groups,
    negative_weights_warn_once,
    normalize_counter,
    normalize_materialize_limit,
    normalize_weights,
)

__all__ = (
    "MAX_MATERIALIZE_DEFAULT",
    "normalize_materialize_limit",
    "is_non_string_sequence",
    "flatten_structure",
    "STRING_TYPES",
    "ensure_collection",
    "normalize_weights",
    "negative_weights_warn_once",
    "normalize_counter",
    "mix_groups",
)

warnings.warn(
    "'tnfr.collections_utils' is deprecated; import from 'tnfr.utils.data' instead",
    DeprecationWarning,
    stacklevel=2,
)

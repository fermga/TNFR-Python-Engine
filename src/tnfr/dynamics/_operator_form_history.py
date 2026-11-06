"""Invalidate own physical samples after a native instantaneous form jump.

This boundary supplies neither a clock nor a growth sample. Explicit operator
interventions and continuous integration retain their existing separate owners.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ..constants.aliases import ALIAS_EPI
from ..operators._epi_domain import AFFINE_EPI_GLYPHS, require_real_scalar_epi

_PHYSICAL_HISTORY = "epi_time_history"
_MISSING = object()


def _field(mapping: Any, key: str, default: Any) -> Any:
    if type(mapping) in (dict, OrderedDict):
        return dict.get(mapping, key, default)
    from ..operators.network_stage import _runtime_mapping_known_value

    return _runtime_mapping_known_value(mapping, key, default)


def _form(node_data: Any) -> float:
    for alias in ALIAS_EPI:
        value = _field(node_data, alias, _MISSING)
        if value is not _MISSING:
            return require_real_scalar_epi(
                value, operator="native form boundary", label="EPI"
            )
    raise ValueError("native form boundary requires an explicit EPI coordinate")


def capture_native_form_history(node_data: Any, glyph: Any) -> float | None:
    """Read a scalar left endpoint only when this dispatch can stale a source."""
    if glyph not in AFFINE_EPI_GLYPHS:
        return None
    if _field(node_data, _PHYSICAL_HISTORY, None) is None:
        return None
    return _form(node_data)


def finish_native_form_history(node_data: Any, before: float | None) -> bool:
    """Clear existing own physical evidence after a changed/unreadable endpoint.

    History contents are not consumed. No-op writes preserve their exact
    object, even when that object is not currently admissible as evidence.
    Invalid post-state is not repaired or certified by this cleanup.
    """
    if before is None:
        return False
    try:
        changed = _form(node_data) != before
    except (OverflowError, TypeError, ValueError):
        changed = True
    if not changed or _field(node_data, _PHYSICAL_HISTORY, None) is None:
        return False

    from ..operators.network_stage import _set_runtime_mapping_item

    _set_runtime_mapping_item(node_data, _PHYSICAL_HISTORY, ())
    return True

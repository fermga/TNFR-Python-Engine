"""Validation utilities."""

from __future__ import annotations

import numbers
import sys

from .constants import get_aliases, get_param
from .alias import get_attr
from .sense import sigma_vector_from_graph
from .helpers.numeric import within_range
from .config.constants import GLYPHS_CANONICAL_SET

ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")

__all__ = ("validate_window", "run_validators")


def validate_window(window: int, *, positive: bool = False) -> int:
    """Validate ``window`` as an ``int`` and return it.

    Non-integer values raise :class:`TypeError`. When ``positive`` is ``True``
    the value must be strictly greater than zero; otherwise it may be zero.
    Negative values always raise :class:`ValueError`.
    """

    if isinstance(window, bool) or not isinstance(window, numbers.Integral):
        raise TypeError("'window' must be an integer")
    if window < 0 or (positive and window == 0):
        kind = "positive" if positive else "non-negative"
        raise ValueError(f"'window'={window} must be {kind}")
    return int(window)


def _require_attr(data, alias, node, name):
    """Return attribute value or raise if missing."""
    val = get_attr(data, alias, None)
    if val is None:
        raise ValueError(f"Missing {name} attribute in node {node}")
    return val


def _validate_sigma(G) -> None:
    sv = sigma_vector_from_graph(G)
    if sv.get("mag", 0.0) > 1.0 + sys.float_info.epsilon:
        raise ValueError("σ norm exceeds 1")


def _check_epi_vf(epi, vf, epi_min, epi_max, vf_min, vf_max, n):
    _check_range(epi, epi_min, epi_max, "EPI", n)
    _check_range(vf, vf_min, vf_max, "VF", n)


def _out_of_range_msg(name, node, val):
    return f"{name} out of range in node {node}: {val}"


def _check_range(val, lower, upper, name, node, tol: float = 1e-9):
    if not within_range(val, lower, upper, tol):
        raise ValueError(_out_of_range_msg(name, node, val))


def _check_glyph(g, n):
    if g and g not in GLYPHS_CANONICAL_SET:
        raise KeyError(f"Invalid glyph {g} in node {n}")


def run_validators(G) -> None:
    """Run all invariant validators on ``G`` with a single node pass."""
    from .glyph_history import last_glyph

    epi_min = float(get_param(G, "EPI_MIN"))
    epi_max = float(get_param(G, "EPI_MAX"))
    vf_min = float(get_param(G, "VF_MIN"))
    vf_max = float(get_param(G, "VF_MAX"))

    for n, data in G.nodes(data=True):
        epi = _require_attr(data, ALIAS_EPI, n, "EPI")
        vf = _require_attr(data, ALIAS_VF, n, "VF")
        _check_epi_vf(epi, vf, epi_min, epi_max, vf_min, vf_max, n)
        _check_glyph(last_glyph(data), n)

    _validate_sigma(G)

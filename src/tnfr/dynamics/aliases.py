"""Shared alias tokens used across TNFR dynamics submodules."""

from __future__ import annotations

from ..constants import get_aliases

ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_EPI = get_aliases("EPI")
ALIAS_SI = get_aliases("SI")
ALIAS_D2EPI = get_aliases("D2EPI")
ALIAS_DSI = get_aliases("DSI")

__all__ = (
    "ALIAS_VF",
    "ALIAS_DNFR",
    "ALIAS_EPI",
    "ALIAS_SI",
    "ALIAS_D2EPI",
    "ALIAS_DSI",
)


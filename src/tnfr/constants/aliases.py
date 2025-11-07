"""Shared alias constants for TNFR attributes."""

from __future__ import annotations

from . import get_aliases

ALIAS_VF = get_aliases("VF")
ALIAS_THETA = get_aliases("THETA")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_EPI = get_aliases("EPI")
ALIAS_EPI_KIND = get_aliases("EPI_KIND")
ALIAS_SI = get_aliases("SI")
ALIAS_DEPI = get_aliases("DEPI")
ALIAS_D2EPI = get_aliases("D2EPI")
ALIAS_DVF = get_aliases("DVF")
ALIAS_D2VF = get_aliases("D2VF")
ALIAS_DSI = get_aliases("DSI")
ALIAS_EMISSION_TIMESTAMP = get_aliases("EMISSION_TIMESTAMP")

__all__ = [
    "ALIAS_VF",
    "ALIAS_THETA",
    "ALIAS_DNFR",
    "ALIAS_EPI",
    "ALIAS_EPI_KIND",
    "ALIAS_SI",
    "ALIAS_DEPI",
    "ALIAS_D2EPI",
    "ALIAS_DVF",
    "ALIAS_D2VF",
    "ALIAS_DSI",
    "ALIAS_EMISSION_TIMESTAMP",
]

from __future__ import annotations

from typing import Final

from .types import GraphLike

__all__ = ("get_hz_bridge", "hz_str_to_hz", "hz_to_hz_str")

HZ_STR_BRIDGE_KEY: Final[str]

def get_hz_bridge(G: GraphLike) -> float: ...
def hz_str_to_hz(value: float, G: GraphLike) -> float: ...
def hz_to_hz_str(value: float, G: GraphLike) -> float: ...

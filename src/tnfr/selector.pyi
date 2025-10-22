from __future__ import annotations

from typing import Any, Mapping

__all__: Any

def __getattr__(name: str) -> Any: ...
def _apply_selector_hysteresis(
    nd: dict[str, Any],
    Si: float,
    dnfr: float,
    accel: float,
    thr: Mapping[str, float],
    margin: float | None,
) -> str | None: ...

_calc_selector_score: Any
_selector_norms: Any
_selector_thresholds: Any

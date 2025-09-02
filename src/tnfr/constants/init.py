"""Initialization constants."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any
import math


@dataclass(frozen=True)
class InitDefaults:
    INIT_RANDOM_PHASE: bool = True
    INIT_THETA_MIN: float = -math.pi
    INIT_THETA_MAX: float = math.pi
    INIT_VF_MODE: str = "uniform"
    INIT_VF_MIN: float | None = None
    INIT_VF_MAX: float | None = None
    INIT_VF_MEAN: float = 0.5
    INIT_VF_STD: float = 0.15
    INIT_VF_CLAMP_TO_LIMITS: bool = True


INIT_DEFAULTS = asdict(InitDefaults())
DEFAULTS_PART: Dict[str, Any] = INIT_DEFAULTS

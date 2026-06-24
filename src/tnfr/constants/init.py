"""Initialization constants."""

from __future__ import annotations

import math
from dataclasses import asdict

from ..compat.dataclass import dataclass


@dataclass(frozen=True, slots=True)
class InitDefaults:
    """Default parameters for node initialisation.

    The fields are collected into :data:`INIT_DEFAULTS` and may therefore
    appear unused to tools like Vulture.
    """

    INIT_RANDOM_PHASE: bool = True
    INIT_THETA_MIN: float = -math.pi
    INIT_THETA_MAX: float = math.pi
    INIT_VF_MODE: str = "uniform"
    INIT_VF_MIN: float | None = None
    INIT_VF_MAX: float | None = None
    INIT_VF_MEAN: float = 0.5
    INIT_VF_STD: float = 0.15
    INIT_VF_CLAMP_TO_LIMITS: bool = True
    INIT_SI_MIN: float = 0.4
    INIT_SI_MAX: float = 0.7
    INIT_EPI_VALUE: float = 0.0


INIT_DEFAULTS = asdict(InitDefaults())

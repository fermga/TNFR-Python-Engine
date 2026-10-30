from enum import Enum

__all__ = [
    "TelemetryVerbosity",
    "TELEMETRY_VERBOSITY_LEVELS",
    "TELEMETRY_VERBOSITY_DEFAULT",
]

class TelemetryVerbosity(str, Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"

TELEMETRY_VERBOSITY_LEVELS: tuple[str, ...]
TELEMETRY_VERBOSITY_DEFAULT: str

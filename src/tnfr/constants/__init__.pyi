from __future__ import annotations

from collections.abc import Mapping
from typing import Callable, TypeVar

from ..types import GraphLike, TNFRConfigValue
from .core import CORE_DEFAULTS as CORE_DEFAULTS
from .core import REMESH_DEFAULTS as REMESH_DEFAULTS
from .init import INIT_DEFAULTS as INIT_DEFAULTS
from .metric import COHERENCE as COHERENCE
from .metric import DIAGNOSIS as DIAGNOSIS
from .metric import GRAMMAR_CANON as GRAMMAR_CANON
from .metric import METRIC_DEFAULTS as METRIC_DEFAULTS
from .metric import METRICS as METRICS
from .metric import SIGMA as SIGMA
from .metric import TRACE as TRACE

T = TypeVar("T")

__all__ = (
    "CORE_DEFAULTS",
    "INIT_DEFAULTS",
    "REMESH_DEFAULTS",
    "METRIC_DEFAULTS",
    "SIGMA",
    "TRACE",
    "METRICS",
    "GRAMMAR_CANON",
    "COHERENCE",
    "DIAGNOSIS",
    "DEFAULTS",
    "DEFAULT_SECTIONS",
    "ALIASES",
    "inject_defaults",
    "merge_overrides",
    "get_param",
    "get_graph_param",
    "get_aliases",
    "VF_KEY",
    "THETA_KEY",
    "VF_PRIMARY",
    "THETA_PRIMARY",
    "DNFR_PRIMARY",
    "EPI_PRIMARY",
    "EPI_KIND_PRIMARY",
    "SI_PRIMARY",
    "dEPI_PRIMARY",
    "D2EPI_PRIMARY",
    "dVF_PRIMARY",
    "D2VF_PRIMARY",
    "dSI_PRIMARY",
    "STATE_STABLE",
    "STATE_TRANSITION",
    "STATE_DISSONANT",
    "CANONICAL_STATE_TOKENS",
    "normalise_state_token",
)

ensure_node_offset_map: Callable[[GraphLike], None] | None
DEFAULT_SECTIONS: Mapping[str, Mapping[str, TNFRConfigValue]]
DEFAULTS: Mapping[str, TNFRConfigValue]
ALIASES: dict[str, tuple[str, ...]]
VF_KEY: str
THETA_KEY: str
VF_PRIMARY: str
THETA_PRIMARY: str
DNFR_PRIMARY: str
EPI_PRIMARY: str
EPI_KIND_PRIMARY: str
SI_PRIMARY: str
dEPI_PRIMARY: str
D2EPI_PRIMARY: str
dVF_PRIMARY: str
D2VF_PRIMARY: str
dSI_PRIMARY: str
STATE_STABLE: str
STATE_TRANSITION: str
STATE_DISSONANT: str
CANONICAL_STATE_TOKENS: frozenset[str]

def inject_defaults(
    G: GraphLike,
    defaults: Mapping[str, TNFRConfigValue] = ...,
    override: bool = ...,
) -> None: ...
def merge_overrides(G: GraphLike, **overrides: TNFRConfigValue) -> None: ...
def get_param(G: GraphLike, key: str) -> TNFRConfigValue: ...
def get_graph_param(G: GraphLike, key: str, cast: Callable[[object], T] = ...) -> T | None: ...
def get_aliases(key: str) -> tuple[str, ...]: ...
def normalise_state_token(token: str) -> str: ...

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from ..config import apply_config
from ..config.presets import get_preset
from ..constants import METRIC_DEFAULTS
from ..dynamics import (
    default_glyph_selector,
    parametric_glyph_selector,
    run,
    validate_canon,
)
from ..execution import CANONICAL_PRESET_NAME, play, seq
from ..flatten import parse_program_tokens
from ..glyph_history import ensure_history
from ..metrics import (
    build_metrics_summary,
    export_metrics,
    glyph_top,
    register_metrics_callbacks,
)
from ..metrics.core import _metrics_step
from ..ontosim import prepare_network
from ..sense import register_sigma_callback
from ..trace import register_trace
from ..types import Glyph, ProgramTokens
from ..utils import (
    StructuredFileError,
    get_logger,
    json_dumps,
    read_structured_file,
    safe_write,
)
from .arguments import _args_to_dict

DEFAULT_SUMMARY_SERIES_LIMIT: int
logger: Any

def _save_json(path: str, data: Any) -> None: ...
def _attach_callbacks(G: nx.Graph) -> None: ...
def _persist_history(G: nx.Graph, args: argparse.Namespace) -> None: ...
def build_basic_graph(args: argparse.Namespace) -> nx.Graph: ...
def apply_cli_config(G: nx.Graph, args: argparse.Namespace) -> None: ...
def register_callbacks_and_observer(G: nx.Graph) -> None: ...
def _build_graph_from_args(args: argparse.Namespace) -> nx.Graph: ...
def _load_sequence(path: Path) -> ProgramTokens: ...
def resolve_program(
    args: argparse.Namespace, default: Optional[ProgramTokens] = ...
) -> Optional[ProgramTokens]: ...
def run_program(
    G: Optional[nx.Graph],
    program: Optional[ProgramTokens],
    args: argparse.Namespace,
) -> nx.Graph: ...
def _run_cli_program(
    args: argparse.Namespace,
    *,
    default_program: Optional[ProgramTokens] = ...,
    graph: Optional[nx.Graph] = ...,
) -> tuple[int, Optional[nx.Graph]]: ...
def _log_run_summaries(G: nx.Graph, args: argparse.Namespace) -> None: ...
def cmd_run(args: argparse.Namespace) -> int: ...
def cmd_sequence(args: argparse.Namespace) -> int: ...
def cmd_metrics(args: argparse.Namespace) -> int: ...
def cmd_profile_si(args: argparse.Namespace) -> int: ...

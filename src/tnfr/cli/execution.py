from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx  # type: ignore[import-untyped]

from ..constants import inject_defaults, DEFAULTS, METRIC_DEFAULTS
from ..sense import register_sigma_callback, sigma_rose
from ..metrics import (
    register_metrics_callbacks,
    Tg_global,
    latency_series,
    glyphogram_series,
    glyph_top,
    export_metrics,
    _metrics_step,
)
from ..trace import register_trace
from ..program import play, seq, block
from ..dynamics import (
    step,
    default_glyph_selector,
    parametric_glyph_selector,
    validate_canon,
)
from ..scenarios import build_graph
from ..presets import get_preset
from ..config import apply_config
from ..io import read_structured_file, safe_write
from ..helpers.numeric import list_mean
from ..observers import attach_standard_observer
from ..logging_utils import get_logger
from ..types import Glyph
from ..json_utils import json_dumps_str

from .arguments import _args_to_dict
from .token_parser import _parse_tokens

logger = get_logger(__name__)


def _save_json(path: str, data: Any) -> None:
    payload = json_dumps_str(
        data, ensure_ascii=False, indent=2, default=list
    )
    safe_write(path, lambda f: f.write(payload))


def _attach_callbacks(G: "nx.Graph") -> None:
    inject_defaults(G, DEFAULTS)
    register_sigma_callback(G)
    register_metrics_callbacks(G)
    register_trace(G)
    _metrics_step(G)


def _persist_history(G: "nx.Graph", args: argparse.Namespace) -> None:
    if args.save_history or args.export_history_base:
        history = G.graph.get("history", {})
        if args.save_history:
            _save_json(args.save_history, history)
        if args.export_history_base:
            export_metrics(G, args.export_history_base, fmt=args.export_format)


def build_basic_graph(args: argparse.Namespace) -> "nx.Graph":
    return build_graph(
        n=args.nodes, topology=args.topology, seed=args.seed, p=args.p
    )


def apply_cli_config(G: "nx.Graph", args: argparse.Namespace) -> None:
    if args.config:
        apply_config(G, Path(args.config))
    arg_map = {
        "dt": ("DT", float),
        "integrator": ("INTEGRATOR_METHOD", str),
        "remesh_mode": ("REMESH_MODE", str),
        "glyph_hysteresis_window": ("GLYPH_HYSTERESIS_WINDOW", int),
    }
    for attr, (key, conv) in arg_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            G.graph[key] = conv(val)

    gcanon = {
        **METRIC_DEFAULTS["GRAMMAR_CANON"],
        **_args_to_dict(args, prefix="grammar_"),
    }
    if getattr(args, "grammar_canon", None) is not None:
        gcanon["enabled"] = bool(args.grammar_canon)
    G.graph["GRAMMAR_CANON"] = gcanon

    selector = getattr(args, "selector", None)
    if selector is not None:
        sel_map = {
            "basic": default_glyph_selector,
            "param": parametric_glyph_selector,
        }
        G.graph["glyph_selector"] = sel_map.get(
            selector, default_glyph_selector
        )

    if hasattr(args, "gamma_type"):
        G.graph["GAMMA"] = {
            "type": args.gamma_type,
            "beta": args.gamma_beta,
            "R0": args.gamma_R0,
        }


def register_callbacks_and_observer(
    G: "nx.Graph", args: argparse.Namespace
) -> None:
    _attach_callbacks(G)
    if args.observer:
        attach_standard_observer(G)
    validate_canon(G)


def _build_graph_from_args(args: argparse.Namespace) -> "nx.Graph":
    G = build_basic_graph(args)
    apply_cli_config(G, args)
    register_callbacks_and_observer(G, args)
    return G


def _load_sequence(path: Path) -> list[Any]:
    data = read_structured_file(path)
    return seq(*_parse_tokens(data))


def resolve_program(
    args: argparse.Namespace, default: Optional[Any] = None
) -> Optional[Any]:
    if getattr(args, "preset", None):
        return get_preset(args.preset)
    if getattr(args, "sequence_file", None):
        return _load_sequence(Path(args.sequence_file))
    return default


def run_program(
    G: Optional["nx.Graph"], program: Optional[Any], args: argparse.Namespace
) -> "nx.Graph":
    if G is None:
        G = _build_graph_from_args(args)

    if program is None:
        steps = int(getattr(args, "steps", 100) or 100)
        for _ in range(steps):
            step(G)
    else:
        play(G, program)

    _persist_history(G, args)
    return G


def _log_run_summaries(G: "nx.Graph", args: argparse.Namespace) -> None:
    cfg_coh = G.graph.get("COHERENCE", METRIC_DEFAULTS["COHERENCE"])
    cfg_diag = G.graph.get("DIAGNOSIS", METRIC_DEFAULTS["DIAGNOSIS"])
    hist = G.graph.get("history", {})

    if cfg_coh.get("enabled", True):
        Wstats = hist.get(cfg_coh.get("stats_history_key", "W_stats"), [])
        if Wstats:
            logger.info("[COHERENCE] último paso: %s", Wstats[-1])

    if cfg_diag.get("enabled", True):
        last_diag = hist.get(cfg_diag.get("history_key", "nodal_diag"), [])
        if last_diag:
            sample = list(last_diag[-1].values())[:3]
            logger.info("[DIAGNOSIS] ejemplo: %s", sample)

    if args.summary:
        tg = Tg_global(G, normalize=True)
        lat = latency_series(G)
        logger.info("Tg global: %s", tg)
        logger.info("Top operadores por Tg: %s", glyph_top(G, k=5))
        if lat["value"]:
            logger.info("Latencia media: %s", list_mean(lat["value"], 0.0))


def cmd_run(args: argparse.Namespace) -> int:
    program = resolve_program(args)
    G = run_program(None, program, args)
    _log_run_summaries(G, args)
    return 0


def cmd_sequence(args: argparse.Namespace) -> int:
    if args.preset and args.sequence_file:
        logger.error(
            "No se puede usar --preset y --sequence-file al mismo tiempo"
        )
        return 1
    program = resolve_program(
        args,
        default=seq(
            Glyph.AL,
            Glyph.EN,
            Glyph.IL,
            block(Glyph.OZ, Glyph.ZHIR, Glyph.IL, repeat=1),
            Glyph.RA,
            Glyph.SHA,
        ),
    )

    run_program(None, program, args)
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    G = _build_graph_from_args(args)
    for _ in range(int(args.steps or 200)):
        step(G)

    tg = Tg_global(G, normalize=True)
    lat = latency_series(G)
    rose = sigma_rose(G)
    glyph = glyphogram_series(G)

    out = {
        "Tg_global": tg,
        "latency_mean": list_mean(lat["value"], 0.0),
        "rose": rose,
        "glyphogram": {k: v[:10] for k, v in glyph.items()},
    }
    if args.save:
        _save_json(args.save, out)
    else:
        logger.info("%s", json_dumps_str(out))
    return 0

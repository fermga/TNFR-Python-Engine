from __future__ import annotations

import argparse

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from ..constants import METRIC_DEFAULTS
from ..sense import register_sigma_callback
from ..metrics import (
    register_metrics_callbacks,
    glyph_top,
    export_metrics,
    build_metrics_summary,
)
from ..metrics.core import _metrics_step
from ..trace import register_trace
from ..execution import CANONICAL_PRESET_NAME, play
from ..dynamics import (
    run,
    default_glyph_selector,
    parametric_glyph_selector,
    validate_canon,
)
from ..config.presets import (
    PREFERRED_PRESET_NAMES,
    get_preset,
    legacy_preset_guidance,
)
from ..config import apply_config
from ..io import read_structured_file, safe_write, StructuredFileError
from ..glyph_history import ensure_history
from ..ontosim import prepare_network
from ..types import ProgramTokens
from ..utils import get_logger, json_dumps
from ..flatten import parse_program_tokens

from .arguments import _args_to_dict

logger = get_logger(__name__)


# CLI summaries should remain concise by default while allowing callers to
# inspect the full glyphogram series when needed.
DEFAULT_SUMMARY_SERIES_LIMIT = 10

_PREFERRED_PRESETS_DISPLAY = ", ".join(PREFERRED_PRESET_NAMES)


def _save_json(path: str, data: Any) -> None:
    payload = json_dumps(data, ensure_ascii=False, indent=2, default=list)
    safe_write(path, lambda f: f.write(payload))


def _attach_callbacks(G: "nx.Graph") -> None:
    register_sigma_callback(G)
    register_metrics_callbacks(G)
    register_trace(G)
    _metrics_step(G, ctx=None)


def _persist_history(G: "nx.Graph", args: argparse.Namespace) -> None:
    if getattr(args, "save_history", None) or getattr(args, "export_history_base", None):
        history = ensure_history(G)
        if getattr(args, "save_history", None):
            _save_json(args.save_history, history)
        if getattr(args, "export_history_base", None):
            export_metrics(G, args.export_history_base, fmt=args.export_format)


def build_basic_graph(args: argparse.Namespace) -> "nx.Graph":
    n = args.nodes
    topology = getattr(args, "topology", "ring").lower()
    seed = getattr(args, "seed", None)
    if topology == "ring":
        G = nx.cycle_graph(n)
    elif topology == "complete":
        G = nx.complete_graph(n)
    elif topology == "erdos":
        if getattr(args, "p", None) is not None:
            prob = float(args.p)
        else:
            if n <= 0:
                fallback = 0.0
            else:
                fallback = 3.0 / n
            prob = min(max(fallback, 0.0), 1.0)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"p must be between 0 and 1; received {prob}")
        G = nx.gnp_random_graph(n, prob, seed=seed)
    else:
        raise ValueError(
            f"Invalid topology '{topology}'. Accepted options are: ring, complete, erdos"
        )
    if seed is not None:
        G.graph["RANDOM_SEED"] = int(seed)
    return G


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

    for attr, key in (
        ("trace_verbosity", "TRACE"),
        ("metrics_verbosity", "METRICS"),
    ):
        cfg = G.graph.get(key)
        if not isinstance(cfg, dict):
            cfg = deepcopy(METRIC_DEFAULTS[key])
            G.graph[key] = cfg
        value = getattr(args, attr, None)
        if value is not None:
            cfg["verbosity"] = value


def register_callbacks_and_observer(G: "nx.Graph") -> None:
    _attach_callbacks(G)
    validate_canon(G)


def _build_graph_from_args(args: argparse.Namespace) -> "nx.Graph":
    G = build_basic_graph(args)
    apply_cli_config(G, args)
    if getattr(args, "observer", False):
        G.graph["ATTACH_STD_OBSERVER"] = True
    prepare_network(G)
    register_callbacks_and_observer(G)
    return G


def _load_sequence(path: Path) -> ProgramTokens:
    try:
        data = read_structured_file(path)
    except (StructuredFileError, OSError) as exc:
        if isinstance(exc, StructuredFileError):
            message = str(exc)
        else:
            message = str(StructuredFileError(path, exc))
        logger.error("%s", message)
        raise SystemExit(1) from exc
    return parse_program_tokens(data)


def resolve_program(
    args: argparse.Namespace, default: Optional[ProgramTokens] = None
) -> Optional[ProgramTokens]:
    if getattr(args, "preset", None):
        try:
            return get_preset(args.preset)
        except KeyError as exc:
            guidance = legacy_preset_guidance(args.preset)
            if guidance is not None:
                details = guidance
            else:
                details = (
                    exc.args[0]
                    if exc.args
                    else "Legacy preset identifier rejected."
                )
            logger.error(
                (
                    "Unknown preset '%s'. Available presets: %s. %s "
                    "Use --sequence-file to execute custom sequences."
                ),
                args.preset,
                _PREFERRED_PRESETS_DISPLAY,
                details,
            )
            raise SystemExit(1) from exc
    if getattr(args, "sequence_file", None):
        return _load_sequence(Path(args.sequence_file))
    return default


def run_program(
    G: Optional["nx.Graph"],
    program: Optional[ProgramTokens],
    args: argparse.Namespace,
) -> "nx.Graph":
    if G is None:
        G = _build_graph_from_args(args)

    if program is None:
        steps = getattr(args, "steps", 100)
        steps = 100 if steps is None else int(steps)
        if steps < 0:
            steps = 0

        run_kwargs: dict[str, Any] = {}
        for attr in ("dt", "use_Si", "apply_glyphs"):
            value = getattr(args, attr, None)
            if value is not None:
                run_kwargs[attr] = value

        run(G, steps=steps, **run_kwargs)
    else:
        play(G, program)

    _persist_history(G, args)
    return G


def _run_cli_program(
    args: argparse.Namespace,
    *,
    default_program: Optional[ProgramTokens] = None,
    graph: Optional["nx.Graph"] = None,
) -> tuple[int, Optional["nx.Graph"]]:
    try:
        program = resolve_program(args, default=default_program)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code or 1, None

    result_graph = run_program(graph, program, args)
    return 0, result_graph


def _log_run_summaries(G: "nx.Graph", args: argparse.Namespace) -> None:
    cfg_coh = G.graph.get("COHERENCE", METRIC_DEFAULTS["COHERENCE"])
    cfg_diag = G.graph.get("DIAGNOSIS", METRIC_DEFAULTS["DIAGNOSIS"])
    hist = ensure_history(G)

    if cfg_coh.get("enabled", True):
        Wstats = hist.get(cfg_coh.get("stats_history_key", "W_stats"), [])
        if Wstats:
            logger.info("[COHERENCE] last step: %s", Wstats[-1])

    if cfg_diag.get("enabled", True):
        last_diag = hist.get(cfg_diag.get("history_key", "nodal_diag"), [])
        if last_diag:
            sample = list(last_diag[-1].values())[:3]
            logger.info("[DIAGNOSIS] sample: %s", sample)

    if args.summary:
        summary_limit = getattr(args, "summary_limit", DEFAULT_SUMMARY_SERIES_LIMIT)
        summary, has_latency_values = build_metrics_summary(
            G, series_limit=summary_limit
        )
        logger.info("Global Tg: %s", summary["Tg_global"])
        logger.info("Top operators by Tg: %s", glyph_top(G, k=5))
        if has_latency_values:
            logger.info("Average latency: %s", summary["latency_mean"])


def cmd_run(args: argparse.Namespace) -> int:
    code, graph = _run_cli_program(args)
    if code != 0:
        return code

    if graph is not None:
        _log_run_summaries(graph, args)
    return 0


def cmd_sequence(args: argparse.Namespace) -> int:
    if args.preset and args.sequence_file:
        logger.error(
            "Cannot use --preset and --sequence-file at the same time"
        )
        return 1
    code, _ = _run_cli_program(
        args, default_program=get_preset(CANONICAL_PRESET_NAME)
    )
    return code


def cmd_metrics(args: argparse.Namespace) -> int:
    if getattr(args, "steps", None) is None:
        # Default a longer run for metrics stability
        args.steps = 200

    code, graph = _run_cli_program(args)
    if code != 0 or graph is None:
        return code

    summary_limit = getattr(args, "summary_limit", None)
    out, _ = build_metrics_summary(graph, series_limit=summary_limit)
    if args.save:
        _save_json(args.save, out)
    else:
        logger.info("%s", json_dumps(out))
    return 0

from __future__ import annotations
import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import os
from collections import deque

import networkx as nx

logger = logging.getLogger(__name__)

from .constants import inject_defaults, DEFAULTS, METRIC_DEFAULTS
from .sense import register_sigma_callback, sigma_rose
from .metrics import (
    register_metrics_callbacks,
    Tg_global,
    latency_series,
    glifogram_series,
    glyph_top,
    export_history,
)
from .trace import register_trace
from .program import play, seq, block, wait, target
from .types import Glyph
from .dynamics import step, _update_history, default_glyph_selector, parametric_glyph_selector, validate_canon
from .gamma import GAMMA_REGISTRY
from .scenarios import build_graph
from .presets import get_preset
from .config import apply_config
from .helpers import read_structured_file
from .observers import attach_standard_observer
from . import __version__


def _parse_tokens(obj: Any) -> List[Any]:
    from collections import deque

    out: List[Any] = []
    queue = deque(obj if isinstance(obj, list) else [obj])
    while queue:
        tok = queue.popleft()
        if isinstance(tok, list):
            queue.extendleft(reversed(tok))
            continue
        if isinstance(tok, dict):
            if len(tok) != 1:
                raise ValueError(f"Token inválido: {tok}")
            key, val = next(iter(tok.items()))
            handler = TOKEN_MAP.get(key)
            if handler is None:
                raise ValueError(f"Token no reconocido: {key}")
            out.append(handler(val))
            continue
        if isinstance(tok, str):
            out.append(tok)
            continue
        raise ValueError(f"Token inválido: {tok}")
    return out


TOKEN_MAP: Dict[str, Callable[[Any], Any]] = {
    "WAIT": lambda v: wait(int(v)),
    "TARGET": lambda v: target(v),
    "THOL": lambda spec: block(
        *_parse_tokens(spec.get("body", [])),
        repeat=int(spec.get("repeat", 1)),
        close=Glyph(spec.get("close")) if isinstance(spec.get("close"), str) else spec.get("close"),
    ),
}


def _save_json(path: str, data: Any) -> None:
    def _default(obj):
        if isinstance(obj, deque):
            return list(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)


def ensure_parent(path: str) -> None:
    """Create parent directory of ``path`` if needed."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def _str2bool(s: str) -> bool:
    s = s.lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


# Metadatos para las opciones de gramática y del glifo
GRAMMAR_ARG_SPECS = [
    ("--grammar.enabled", _str2bool),
    ("--grammar.zhir_requires_oz_window", int),
    ("--grammar.zhir_dnfr_min", float),
    ("--grammar.thol_min_len", int),
    ("--grammar.thol_max_len", int),
    ("--grammar.thol_close_dnfr", float),
    ("--grammar.si_high", float),
    ("--glyph.hysteresis_window", int),
]


def _args_to_dict(args: argparse.Namespace, prefix: str) -> Dict[str, Any]:
    """Extract arguments matching a prefix.

    Parameters
    ----------
    args:
        Namespace produced by ``argparse``.
    prefix:
        Prefix to match against the argument names.  It must include the
        trailing underscore, for example ``"grammar_"``.

    Returns
    -------
    dict
        Mapping of argument names with the prefix stripped.

    Examples
    --------
    >>> ns = argparse.Namespace(grammar_enabled=True, grammar_thol_min=2, other=1)
    >>> _args_to_dict(ns, "grammar_")
    {'enabled': True, 'thol_min': 2}
    """

    return {
        k[len(prefix):]: v
        for k, v in vars(args).items()
        if k.startswith(prefix) and v is not None
    }


def _load_sequence(path: Path) -> List[Any]:
    data = read_structured_file(path)

    return seq(*_parse_tokens(data))


def _attach_callbacks(G: nx.Graph) -> None:
    inject_defaults(G, DEFAULTS)
    register_sigma_callback(G)
    register_metrics_callbacks(G)
    register_trace(G)
    _update_history(G)


def _persist_history(G: nx.Graph, args: argparse.Namespace) -> None:
    """Guardar o exportar el histórico si se solicitó."""
    if getattr(args, "save_history", None):
        path = args.save_history
        ensure_parent(path)
        _save_json(path, G.graph.get("history", {}))
    if getattr(args, "export_history_base", None):
        base = args.export_history_base
        ensure_parent(base)
        export_history(G, base, fmt=getattr(args, "export_format", "json"))


def _build_graph_from_args(args: argparse.Namespace) -> nx.Graph:
    """Construye y configura un grafo a partir de los argumentos del CLI."""
    G = build_graph(n=args.nodes, topology=args.topology, seed=args.seed, p=args.p)
    if args.config:
        apply_config(G, Path(args.config))
    _attach_callbacks(G)
    if args.observer:
        attach_standard_observer(G)
    validate_canon(G)
    if args.dt is not None:
        G.graph["DT"] = float(args.dt)
    if args.integrator is not None:
        G.graph["INTEGRATOR_METHOD"] = str(args.integrator)
    if args.remesh_mode:
        G.graph["REMESH_MODE"] = str(args.remesh_mode)

    gcanon = {
        **METRIC_DEFAULTS["GRAMMAR_CANON"],
        **_args_to_dict(args, prefix="grammar_"),
    }
    if hasattr(args, "grammar_canon") and args.grammar_canon is not None:
        gcanon["enabled"] = bool(args.grammar_canon)
    G.graph["GRAMMAR_CANON"] = gcanon

    if args.glyph_hysteresis_window is not None:
        G.graph["GLYPH_HYSTERESIS_WINDOW"] = int(args.glyph_hysteresis_window)

    if hasattr(args, "selector"):
        G.graph["glyph_selector"] = (
            default_glyph_selector if args.selector == "basic" else parametric_glyph_selector
        )

    if hasattr(args, "gamma_type"):
        G.graph["GAMMA"] = {
            "type": args.gamma_type,
            "beta": args.gamma_beta,
            "R0": args.gamma_R0,
        }

    return G


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Agrega los argumentos compartidos entre los subcomandos."""
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--topology", choices=["ring", "complete", "erdos"], default="ring")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--p", type=float, default=None, help="Probabilidad de arista si topology=erdos")
    parser.add_argument("--observer", action="store_true", help="Adjunta observador estándar")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--integrator", choices=["euler", "rk4"], default=None)
    parser.add_argument("--remesh-mode", choices=["knn", "mst", "community"], default=None)
    parser.add_argument("--gamma-type", choices=list(GAMMA_REGISTRY.keys()), default="none")
    parser.add_argument("--gamma-beta", type=float, default=0.0)
    parser.add_argument("--gamma-R0", type=float, default=0.0)


def add_grammar_args(parser: argparse.ArgumentParser) -> None:
    """Agrega las opciones de gramática y de histéresis del glifo."""
    for opt, typ in GRAMMAR_ARG_SPECS:
        dest = opt.lstrip("-").replace(".", "_")
        parser.add_argument(opt, dest=dest, type=typ, default=None)


def cmd_run(args: argparse.Namespace) -> int:
    G = _build_graph_from_args(args)

    if args.preset:
        program = get_preset(args.preset)
        play(G, program)
    else:
        steps = int(args.steps or 100)
        for _ in range(steps):
            step(G)

    _persist_history(G, args)

    # Resúmenes rápidos (si están activados)
    if G.graph.get("COHERENCE", METRIC_DEFAULTS["COHERENCE"]).get("enabled", True):
        Wstats = G.graph.get("history", {}).get(
            G.graph.get("COHERENCE", METRIC_DEFAULTS["COHERENCE"]).get("stats_history_key", "W_stats"), []
        )
        if Wstats:
            logger.info("[COHERENCE] último paso: %s", Wstats[-1])
    if G.graph.get("DIAGNOSIS", METRIC_DEFAULTS["DIAGNOSIS"]).get("enabled", True):
        last_diag = G.graph.get("history", {}).get(
            G.graph.get("DIAGNOSIS", METRIC_DEFAULTS["DIAGNOSIS"]).get("history_key", "nodal_diag"), []
        )
        if last_diag:
            sample = list(last_diag[-1].values())[:3]
            logger.info("[DIAGNOSIS] ejemplo: %s", sample)

    if args.summary:
        tg = Tg_global(G, normalize=True)
        lat = latency_series(G)
        logger.info("Tg global: %s", tg)
        logger.info("Top operadores por Tg: %s", glyph_top(G, k=5))
        if lat["value"]:
            logger.info(
                "Latencia media: %s",
                sum(lat["value"]) / max(1, len(lat["value"])) ,
            )
    return 0


def cmd_sequence(args: argparse.Namespace) -> int:
    G = _build_graph_from_args(args)

    if args.preset:
        program = get_preset(args.preset)
    elif args.sequence_file:
        program = _load_sequence(Path(args.sequence_file))
    else:
        program = seq(
            Glyph.AL,
            Glyph.EN,
            Glyph.IL,
            block(Glyph.OZ, Glyph.ZHIR, Glyph.IL, repeat=1),
            Glyph.RA,
            Glyph.SHA,
        )

    play(G, program)

    _persist_history(G, args)
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    G = _build_graph_from_args(args)
    for _ in range(int(args.steps or 200)):
        step(G)

    tg = Tg_global(G, normalize=True)
    lat = latency_series(G)
    rose = sigma_rose(G)
    glifo = glifogram_series(G)

    out = {
        "Tg_global": tg,
        "latency_mean": (sum(lat["value"]) / max(1, len(lat["value"])) ) if lat["value"] else 0.0,
        "rose": rose,
        "glifogram": {k: v[:10] for k, v in glifo.items()},
    }
    if args.save:
        dir_name = os.path.dirname(args.save)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        _save_json(args.save, out)
    else:
        logger.info("%s", json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)

    p = argparse.ArgumentParser(prog="tnfr")
    p.add_argument("--version", action="store_true", help="muestra versión y sale")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Correr escenario libre o preset y opcionalmente exportar history")
    add_common_args(p_run)
    p_run.add_argument("--steps", type=int, default=200)
    p_run.add_argument("--preset", type=str, default=None)
    p_run.add_argument("--save-history", dest="save_history", type=str, default=None)
    p_run.add_argument("--export-history-base", dest="export_history_base", type=str, default=None)
    p_run.add_argument("--export-format", dest="export_format", choices=["csv", "json"], default="json")
    p_run.add_argument("--summary", action="store_true")
    p_run.add_argument("--no-canon", dest="grammar_canon", action="store_false", default=True, help="Desactiva gramática canónica")
    add_grammar_args(p_run)
    p_run.add_argument("--selector", choices=["basic", "param"], default="basic")
    p_run.set_defaults(func=cmd_run)

    p_seq = sub.add_parser("sequence", help="Ejecutar una secuencia (preset o YAML/JSON)")
    add_common_args(p_seq)
    p_seq.add_argument("--preset", type=str, default=None)
    p_seq.add_argument("--sequence-file", type=str, default=None)
    p_seq.add_argument("--save-history", dest="save_history", type=str, default=None)
    p_seq.add_argument("--export-history-base", dest="export_history_base", type=str, default=None)
    p_seq.add_argument("--export-format", dest="export_format", choices=["csv", "json"], default="json")
    add_grammar_args(p_seq)
    p_seq.set_defaults(func=cmd_sequence)

    p_met = sub.add_parser("metrics", help="Correr breve y volcar métricas clave")
    add_common_args(p_met)
    p_met.add_argument("--steps", type=int, default=300)
    p_met.add_argument("--no-canon", dest="grammar_canon", action="store_false", default=True, help="Desactiva gramática canónica")
    add_grammar_args(p_met)
    p_met.add_argument("--selector", choices=["basic", "param"], default="basic")
    p_met.add_argument("--save", type=str, default=None)
    p_met.set_defaults(func=cmd_metrics)

    args = p.parse_args(argv)
    if args.version:
        logger.info("%s", __version__)
        return 0
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

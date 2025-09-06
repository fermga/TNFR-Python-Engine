"""Command-line interface."""

from __future__ import annotations
import argparse
import json
import logging
import sys
from typing import Any, Optional, Callable, TYPE_CHECKING
from collections.abc import Sequence, Iterable
from pathlib import Path
from collections import deque

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

from .constants import inject_defaults, DEFAULTS, METRIC_DEFAULTS
from .sense import register_sigma_callback, sigma_rose
from .metrics import (
    register_metrics_callbacks,
    Tg_global,
    latency_series,
    glyphogram_series,
    glyph_top,
    export_metrics,
    _metrics_step,
)
from .trace import register_trace
from .program import play, seq, block, wait, target
from .types import Glyph
from .dynamics import (
    step,
    default_glyph_selector,
    parametric_glyph_selector,
    validate_canon,
)
from .gamma import GAMMA_REGISTRY
from .scenarios import build_graph
from .presets import get_preset
from .config import apply_config
from .io import read_structured_file, safe_write
from .helpers import list_mean
from .observers import attach_standard_observer
from . import __version__

logger = logging.getLogger(__name__)

__all__ = [
    "main",
    "add_common_args",
    "add_grammar_args",
    "add_grammar_selector_args",
    "add_history_export_args",
    "add_canon_toggle",
    "build_basic_graph",
    "apply_cli_config",
    "register_callbacks_and_observer",
    "run_program",
    "resolve_program",
    "_build_graph_from_args",
    "_load_sequence",
    "_parse_tokens",
    "TOKEN_MAP",
]


def _flatten_tokens(obj: Any):
    """Recursive generator yielding each token in order."""

    if isinstance(obj, Sequence) and not isinstance(obj, str):
        for item in obj:
            yield from _flatten_tokens(item)
    else:
        yield obj


def validate_token(tok: Any, pos: int) -> Any:
    if isinstance(tok, dict):
        if len(tok) != 1:
            raise ValueError(f"Token inválido: {tok} (posición {pos}, token {tok!r})")
        key, val = next(iter(tok.items()))
        handler = TOKEN_MAP.get(key)
        if handler is None:
            raise ValueError(
                f"Token no reconocido: {key} (posición {pos}, token {tok!r})"
            )
        try:
            return handler(val)
        except (KeyError, ValueError) as e:
            raise type(e)(f"{e} (posición {pos}, token {tok!r})") from e
    if isinstance(tok, str):
        return tok
    raise ValueError(f"Token inválido: {tok} (posición {pos}, token {tok!r})")


def _parse_tokens(obj: Any) -> list[Any]:
    return [
        validate_token(tok, pos)
        for pos, tok in enumerate(_flatten_tokens(obj), start=1)
    ]


def parse_thol(spec: dict[str, Any]) -> Any:
    """Parse the specification of a ``THOL`` block.

    Parameters
    ----------
    spec:
        Dictionary with keys ``body``, ``repeat`` and ``close``.

    Returns
    -------
    Any
        Result of :func:`block` after parsing body tokens.

    Raises
    ------
    ValueError
        If ``close`` is a string that does not correspond to a valid
        :class:`Glyph` name.
    """

    close = spec.get("close")
    if isinstance(close, str):
        if close not in Glyph.__members__:
            raise ValueError(f"Glyph de cierre desconocido: {close!r}")
        close = Glyph[close]

    return block(
        *_parse_tokens(spec.get("body", [])),
        repeat=int(spec.get("repeat", 1)),
        close=close,
    )


TOKEN_MAP: dict[str, Callable[[Any], Any]] = {
    "WAIT": lambda v: wait(int(v)),
    "TARGET": lambda v: target(v),
    "THOL": parse_thol,
}


def _default(obj: Any) -> Any:
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return list(obj)
    raise TypeError(
        f"Object of type {obj.__class__.__name__} is not JSON serializable"
    )


def _save_json(path: str, data: Any) -> None:
    def _write(f):
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)

    safe_write(path, _write)


# Metadatos para las opciones de gramática y del glyph
# Utiliza acciones y tipos estándar de ``argparse`` en lugar de
# conversores personalizados.
GRAMMAR_ARG_SPECS = [
    ("--grammar.enabled", {"action": argparse.BooleanOptionalAction}),
    ("--grammar.zhir_requires_oz_window", {"type": int}),
    ("--grammar.zhir_dnfr_min", {"type": float}),
    ("--grammar.thol_min_len", {"type": int}),
    ("--grammar.thol_max_len", {"type": int}),
    ("--grammar.thol_close_dnfr", {"type": float}),
    ("--grammar.si_high", {"type": float}),
    ("--glyph.hysteresis_window", {"type": int}),
]

# Especificaciones para opciones relacionadas con el histórico
HISTORY_ARG_SPECS = [
    ("--save-history", {"dest": "save_history", "type": str, "default": None}),
    (
        "--export-history-base",
        {"dest": "export_history_base", "type": str, "default": None},
    ),
    (
        "--export-format",
        {
            "dest": "export_format",
            "choices": ["csv", "json"],
            "default": "json",
        },
    ),
]

# Argumentos comunes a los subcomandos
COMMON_ARG_SPECS = [
    ("--nodes", {"type": int, "default": 24}),
    (
        "--topology",
        {"choices": ["ring", "complete", "erdos"], "default": "ring"},
    ),
    ("--seed", {"type": int, "default": 1}),
    (
        "--p",
        {
            "type": float,
            "default": None,
            "help": "Probabilidad de arista si topology=erdos",
        },
    ),
    (
        "--observer",
        {"action": "store_true", "help": "Adjunta observador estándar"},
    ),
    ("--config", {"type": str, "default": None}),
    ("--dt", {"type": float, "default": None}),
    (
        "--integrator",
        {"choices": ["euler", "rk4"], "default": None},
    ),
    (
        "--remesh-mode",
        {"choices": ["knn", "mst", "community"], "default": None},
    ),
    (
        "--gamma-type",
        {"choices": list(GAMMA_REGISTRY.keys()), "default": "none"},
    ),
    ("--gamma-beta", {"type": float, "default": 0.0}),
    ("--gamma-R0", {"type": float, "default": 0.0}),
]


def add_arg_specs(parser: argparse.ArgumentParser, specs) -> None:
    """Register arguments from ``specs`` on ``parser``."""
    for opt, kwargs in specs:
        parser.add_argument(opt, **kwargs)


def _args_to_dict(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    """Extract arguments matching a prefix.

    Parameters
    ----------
    args:
        Namespace produced by ``argparse``.
    prefix:
        Prefix to match against the argument names.  It must include the
        trailing underscore, for example ``"grammar_"``.  Options with this
        prefix are defined in :data:`GRAMMAR_ARG_SPECS`.

    Returns
    -------
    dict
        Mapping of argument names with the prefix stripped. Only entries
        whose values are not ``None`` are included.

    Examples
    --------
    >>> ns = argparse.Namespace(
    ...     grammar_enabled=True, grammar_thol_min_len=2, other=1
    ... )
    >>> _args_to_dict(ns, "grammar_")
    {'enabled': True, 'thol_min_len': 2}
    """

    return {
        k.removeprefix(prefix): v
        for k, v in vars(args).items()
        if k.startswith(prefix) and v is not None
    }


def _load_sequence(path: Path) -> list[Any]:
    data = read_structured_file(path)

    return seq(*_parse_tokens(data))


def _attach_callbacks(G: "nx.Graph") -> None:
    inject_defaults(G, DEFAULTS)
    register_sigma_callback(G)
    register_metrics_callbacks(G)
    register_trace(G)
    _metrics_step(G)


def _persist_history(G: "nx.Graph", args: argparse.Namespace) -> None:
    """Save or export history if requested."""
    if args.save_history or args.export_history_base:
        history = G.graph.get("history", {})
        if args.save_history:
            _save_json(args.save_history, history)
        if args.export_history_base:
            export_metrics(G, args.export_history_base, fmt=args.export_format)


def build_basic_graph(args: argparse.Namespace) -> "nx.Graph":
    """Build base graph from CLI arguments."""
    return build_graph(
        n=args.nodes, topology=args.topology, seed=args.seed, p=args.p
    )


def apply_cli_config(G: "nx.Graph", args: argparse.Namespace) -> None:
    """Apply settings from the CLI or external files."""
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
        G.graph["glyph_selector"] = sel_map.get(selector, default_glyph_selector)

    if hasattr(args, "gamma_type"):
        G.graph["GAMMA"] = {
            "type": args.gamma_type,
            "beta": args.gamma_beta,
            "R0": args.gamma_R0,
        }


def register_callbacks_and_observer(
    G: "nx.Graph", args: argparse.Namespace
) -> None:
    """Register standard callbacks and observers."""
    _attach_callbacks(G)
    if args.observer:
        attach_standard_observer(G)
    validate_canon(G)


def _build_graph_from_args(args: argparse.Namespace) -> "nx.Graph":
    """Build a configured graph from CLI arguments."""
    G = build_basic_graph(args)
    apply_cli_config(G, args)
    register_callbacks_and_observer(G, args)
    return G


def resolve_program(
    args: argparse.Namespace, default: Optional[Any] = None
) -> Optional[Any]:
    """Obtain a program from a preset or sequence file."""
    if getattr(args, "preset", None):
        return get_preset(args.preset)
    if getattr(args, "sequence_file", None):
        return _load_sequence(Path(args.sequence_file))
    return default


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    add_arg_specs(parser, COMMON_ARG_SPECS)


def add_grammar_args(parser: argparse.ArgumentParser) -> None:
    """Add grammar and glyph hysteresis options."""
    group = parser.add_argument_group("Grammar")
    specs = [
        (opt, {**kwargs, "dest": opt.lstrip("-").replace(".", "_"), "default": None})
        for opt, kwargs in GRAMMAR_ARG_SPECS
    ]
    add_arg_specs(group, specs)


def add_grammar_selector_args(parser: argparse.ArgumentParser) -> None:
    """Add grammar options and glyph selector."""
    add_grammar_args(parser)
    parser.add_argument(
        "--selector", choices=["basic", "param"], default="basic"
    )


def add_history_export_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments to save or export history."""
    add_arg_specs(parser, HISTORY_ARG_SPECS)


def add_canon_toggle(parser: argparse.ArgumentParser) -> None:
    """Add option to disable canonical grammar."""
    parser.add_argument(
        "--no-canon",
        dest="grammar_canon",
        action="store_false",
        default=True,
        help="Desactiva gramática canónica",
    )


def _add_run_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``run`` subcommand."""
    p_run = sub.add_parser(
        "run",
        help=(
            "Correr escenario libre o preset y opcionalmente exportar history"
        ),
    )
    add_common_args(p_run)
    p_run.add_argument("--steps", type=int, default=200)
    p_run.add_argument("--preset", type=str, default=None)
    add_history_export_args(p_run)
    p_run.add_argument("--summary", action="store_true")
    add_canon_toggle(p_run)
    add_grammar_selector_args(p_run)
    p_run.set_defaults(func=cmd_run)


def _add_sequence_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``sequence`` subcommand."""
    p_seq = sub.add_parser(
        "sequence",
        help="Ejecutar una secuencia (preset o YAML/JSON)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplo de secuencia JSON:\n"
            "[\n"
            '  "A",\n'
            '  {"WAIT": 1},\n'
            '  {"THOL": {"body": ["A", {"WAIT": 2}], "repeat": 2}}\n'
            "]"
        ),
    )
    add_common_args(p_seq)
    p_seq.add_argument("--preset", type=str, default=None)
    p_seq.add_argument("--sequence-file", type=str, default=None)
    add_history_export_args(p_seq)
    add_grammar_args(p_seq)
    p_seq.set_defaults(func=cmd_sequence)


def _add_metrics_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``metrics`` subcommand."""
    p_met = sub.add_parser(
        "metrics", help="Correr breve y volcar métricas clave"
    )
    add_common_args(p_met)
    p_met.add_argument("--steps", type=int, default=300)
    add_canon_toggle(p_met)
    add_grammar_selector_args(p_met)
    p_met.add_argument("--save", type=str, default=None)
    p_met.set_defaults(func=cmd_metrics)


def run_program(
    G: Optional["nx.Graph"], program: Optional[Any], args: argparse.Namespace
) -> "nx.Graph":
    """Build graph if needed, execute a program and save history."""
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
    """Record quick summaries and optional metrics."""
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
            logger.info(
                "Latencia media: %s",
                list_mean(lat["value"], 0.0),
            )


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
        logger.info("%s", json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True
    )

    p = argparse.ArgumentParser(
        prog="tnfr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplo: tnfr sequence --sequence-file secuencia.json\n"
            "secuencia.json:\n"
            '[\n  {"WAIT": 1},\n  {"TARGET": "A"}\n]'
        ),
    )
    p.add_argument(
        "--version", action="store_true", help="muestra versión y sale"
    )
    sub = p.add_subparsers(dest="cmd")

    _add_run_parser(sub)
    _add_sequence_parser(sub)
    _add_metrics_parser(sub)

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

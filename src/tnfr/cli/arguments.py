from __future__ import annotations

import argparse
from typing import Any

from ..gamma import GAMMA_REGISTRY
from .utils import spec, specs


GRAMMAR_ARG_SPECS = specs(
    spec("--grammar.enabled", action=argparse.BooleanOptionalAction),
    spec("--grammar.zhir_requires_oz_window", type=int),
    spec("--grammar.zhir_dnfr_min", type=float),
    spec("--grammar.thol_min_len", type=int),
    spec("--grammar.thol_max_len", type=int),
    spec("--grammar.thol_close_dnfr", type=float),
    spec("--grammar.si_high", type=float),
    spec("--glyph.hysteresis_window", type=int),
)


# Especificaciones para opciones relacionadas con el histórico
HISTORY_ARG_SPECS = specs(
    spec("--save-history", type=str),
    spec("--export-history-base", type=str),
    spec("--export-format", choices=["csv", "json"], default="json"),
)


# Argumentos comunes a los subcomandos
COMMON_ARG_SPECS = specs(
    spec("--nodes", type=int, default=24),
    spec("--topology", choices=["ring", "complete", "erdos"], default="ring"),
    spec("--seed", type=int, default=1),
    spec(
        "--p",
        type=float,
        help="Probabilidad de arista si topology=erdos",
    ),
    spec("--observer", action="store_true", help="Adjunta observador estándar"),
    spec("--config", type=str),
    spec("--dt", type=float),
    spec("--integrator", choices=["euler", "rk4"]),
    spec("--remesh-mode", choices=["knn", "mst", "community"]),
    spec("--gamma-type", choices=list(GAMMA_REGISTRY.keys()), default="none"),
    spec("--gamma-beta", type=float, default=0.0),
    spec("--gamma-R0", type=float, default=0.0),
)


def add_arg_specs(parser: argparse._ActionsContainer, specs) -> None:
    """Register arguments from ``specs`` on ``parser``."""
    for opt, kwargs in specs:
        parser.add_argument(opt, **kwargs)


def _args_to_dict(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    """Extract arguments matching a prefix."""
    return {
        k.removeprefix(prefix): v
        for k, v in vars(args).items()
        if k.startswith(prefix) and v is not None
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    add_arg_specs(parser, COMMON_ARG_SPECS)


def add_grammar_args(parser: argparse.ArgumentParser) -> None:
    """Add grammar and glyph hysteresis options."""
    group = parser.add_argument_group("Grammar")
    add_arg_specs(group, GRAMMAR_ARG_SPECS)


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
    from .execution import cmd_run

    p_run = sub.add_parser(
        "run",
        help=(
            "Correr escenario libre o preset y opcionalmente exportar history"
        ),
    )
    add_common_args(p_run)
    p_run.add_argument("--steps", type=int, default=100)
    add_canon_toggle(p_run)
    add_grammar_selector_args(p_run)
    add_history_export_args(p_run)
    p_run.add_argument("--preset", type=str, default=None)
    p_run.add_argument("--sequence-file", type=str, default=None)
    p_run.add_argument("--summary", action="store_true")
    p_run.set_defaults(func=cmd_run)


def _add_sequence_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``sequence`` subcommand."""
    from .execution import cmd_sequence

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
    from .execution import cmd_metrics

    p_met = sub.add_parser(
        "metrics", help="Correr breve y volcar métricas clave"
    )
    add_common_args(p_met)
    p_met.add_argument("--steps", type=int, default=300)
    add_canon_toggle(p_met)
    add_grammar_selector_args(p_met)
    p_met.add_argument("--save", type=str, default=None)
    p_met.set_defaults(func=cmd_metrics)

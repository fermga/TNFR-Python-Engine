from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from .arguments import (
    add_common_args,
    add_grammar_args,
    add_grammar_selector_args,
    add_history_export_args,
    add_canon_toggle,
    _add_run_parser,
    _add_sequence_parser,
    _add_metrics_parser,
    _args_to_dict,
)
from .token_parser import _parse_tokens
from ..token_map import TOKEN_MAP
from .execution import (
    build_basic_graph,
    apply_cli_config,
    register_callbacks_and_observer,
    run_program,
    resolve_program,
    _build_graph_from_args,
    _load_sequence,
    _save_json,
)
from ..logging_utils import get_logger
from .. import __version__

logger = get_logger(__name__)

__all__ = (
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
    "_save_json",
    "_args_to_dict",
)


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

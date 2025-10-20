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
)
from .execution import (
    build_basic_graph,
    apply_cli_config,
    register_callbacks_and_observer,
    run_program,
    resolve_program,
)
from .. import __version__
from ..utils import _configure_root, get_logger

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
)


def main(argv: Optional[list[str]] = None) -> int:
    _configure_root()

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    p = argparse.ArgumentParser(
        prog="tnfr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example: tnfr sequence --sequence-file sequence.json\n"
            "sequence.json:\n"
            '[\n  {"WAIT": 1},\n  {"TARGET": "A"}\n]'
        ),
    )
    p.add_argument(
        "--version",
        action="store_true",
        help=(
            "show the actual version and exit (reads pyproject.toml in development)"
        ),
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

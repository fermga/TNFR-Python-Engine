"""Command-line interface entry points for TNFR."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from .. import __version__
from ..errors import TNFRUserError
from ..utils import get_logger
from .arguments import (
    _add_epi_validate_parser,
    _add_math_run_parser,
    _add_metrics_parser,
    _add_run_parser,
    _add_sequence_parser,
    add_canon_toggle,
    add_common_args,
    add_grammar_args,
    add_grammar_selector_args,
    add_history_export_args,
)
from .execution import (
    apply_cli_config,
    build_basic_graph,
    register_callbacks_and_observer,
    resolve_program,
    run_program,
)
from .study import add_study_parsers

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


@contextmanager
def _command_logging() -> Iterator[None]:
    """Route TNFR progress to stderr and restore the embedding application's state."""
    library_logger = logging.getLogger("tnfr")
    previous = (
        list(library_logger.handlers),
        library_logger.level,
        library_logger.propagate,
    )
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    library_logger.handlers = [handler]
    library_logger.setLevel(logging.INFO)
    library_logger.propagate = False
    try:
        yield
    finally:
        library_logger.handlers, level, library_logger.propagate = previous
        library_logger.setLevel(level)
        handler.close()


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; structured results use stdout and progress/errors use stderr."""

    p = argparse.ArgumentParser(
        prog="tnfr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="TNFR - graph-coupled nodal dynamics and scoped diagnostics",
        epilog=(
            "Common examples:\n"
            "  # Create, evolve and diagnose through the SDK\n"
            "  tnfr network --nodes 6 --topology ring --seed 42 --steps 1\n\n"
            "  # Save a declaration and its finite observations\n"
            "  tnfr network --export-spec study.json --output result.json\n"
            "  tnfr network --spec study.json --output repeated-result.json\n\n"
            "  # Discover current registered contracts and SDK words\n"
            "  tnfr operators reception\n"
            "  tnfr sequences basic_activation\n\n"
            "  # Advanced timed runtime and stored-history metrics\n"
            "  tnfr run --nodes 6 --steps 10 --summary\n"
            "  tnfr metrics --nodes 6 --steps 10 --save metrics.json\n\n"
            "For detailed help on any subcommand:\n"
            "  tnfr <subcommand> --help"
        ),
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="show the package version and exit",
    )
    sub = p.add_subparsers(dest="cmd", help="Available subcommands")

    add_study_parsers(sub)
    _add_run_parser(sub)
    _add_math_run_parser(sub)
    _add_epi_validate_parser(sub)
    _add_sequence_parser(sub)
    _add_metrics_parser(sub)

    args = p.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    with _command_logging():
        try:
            return int(args.func(args))
        except (TNFRUserError, ValueError, OSError) as exc:
            logger.error("error: %s", exc)
            return 2

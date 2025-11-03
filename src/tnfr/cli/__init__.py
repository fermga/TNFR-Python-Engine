"""Command-line interface entry points for TNFR."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from .. import __version__
from ..utils import _configure_root, get_logger
from .arguments import (
    _add_epi_validate_parser,
    _add_math_run_parser,
    _add_metrics_parser,
    _add_profile_parser,
    _add_profile_pipeline_parser,
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
    """Entry point for the ``tnfr`` CLI returning the exit status."""

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
        description="TNFR - Resonant Fractal Nature Theory computational engine",
        epilog=(
            "Common examples:\n"
            "  # Run a preset scenario\n"
            "  tnfr run --preset resonant_bootstrap --steps 100\n\n"
            "  # Run with math engine validation\n"
            "  tnfr math.run --nodes 24 --steps 50\n\n"
            "  # Validate EPI integrity\n"
            "  tnfr epi.validate --preset coupling_exploration\n\n"
            "  # Execute custom sequence from YAML\n"
            "  tnfr sequence --sequence-file presets/resonant_bootstrap.yaml\n\n"
            "  # Export metrics to JSON\n"
            "  tnfr metrics --save metrics.json --steps 200\n\n"
            "For detailed help on any subcommand:\n"
            "  tnfr <subcommand> --help"
        ),
    )
    p.add_argument(
        "--version",
        action="store_true",
        help=("show the actual version and exit (reads pyproject.toml in development)"),
    )
    sub = p.add_subparsers(dest="cmd", help="Available subcommands")

    _add_run_parser(sub)
    _add_math_run_parser(sub)
    _add_epi_validate_parser(sub)
    _add_sequence_parser(sub)
    _add_metrics_parser(sub)
    _add_profile_parser(sub)
    _add_profile_pipeline_parser(sub)

    args = p.parse_args(argv)
    if args.version:
        logger.info("%s", __version__)
        return 0
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return int(args.func(args))

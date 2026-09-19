"""CLI adapters for the shared SDK study and catalog interfaces."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

from ..sdk.utils import export_to_json, import_from_json
from ..utils.io import json_dumps


def _emit(data: Any, output: Path | None) -> None:
    """Keep structured results on stdout and use the shared atomic file writer."""
    # ASCII escapes keep redirected JSON usable on Windows legacy code pages;
    # the shared file writer still emits UTF-8. Both decode to the same values.
    payload = json_dumps(data, indent=2, ensure_ascii=True, allow_nan=False)
    if output is None:
        print(payload)
    else:
        export_to_json(data, output)


def _distinct_paths(args: argparse.Namespace) -> None:
    """Reject an output that would overwrite its input or the other artifact."""
    paths = [getattr(args, name, None) for name in ("spec", "export_spec", "output")]
    resolved = [Path(path).resolve() for path in paths if path is not None]
    if len(set(resolved)) != len(resolved):
        raise ValueError("spec input, exported spec and result must use distinct paths")


def cmd_network(args: argparse.Namespace) -> int:
    """Execute exactly the declaration consumed by the SDK study runner."""
    from ..sdk import StudySpec, run_study

    _distinct_paths(args)
    supplied = {
        field.name: getattr(args, field.name)
        for field in fields(StudySpec)
        if getattr(args, field.name, None) is not None
    }
    if args.spec is not None:
        if supplied:
            raise ValueError(
                "--spec cannot be combined with network construction options"
            )
        declaration_data = import_from_json(args.spec)
    else:
        declaration_data = supplied
    try:
        specification = StudySpec.from_dict(declaration_data)
    except TypeError as exc:
        raise ValueError(f"Invalid study recipe: {exc}") from exc

    result = run_study(specification).to_dict()
    # Validate both payloads before any destination can change. Individual file
    # replacement is atomic; this is not a multi-file transaction or checkpoint.
    declaration = specification.to_dict()
    json_dumps(result, allow_nan=False)
    json_dumps(declaration, allow_nan=False)
    _emit(result, args.output)
    if args.export_spec is not None:
        export_to_json(declaration, args.export_spec)
    return 0


def cmd_operators(args: argparse.Namespace) -> int:
    """Show registered contracts without executing any operators."""
    from ..sdk import TNFR

    try:
        contracts = TNFR.operators(args.name)
    except KeyError as exc:
        raise ValueError(
            f"Unknown operator {args.name!r}; use 'tnfr operators'"
        ) from exc
    _emit(contracts, args.output)
    return 0


def cmd_sequences(args: argparse.Namespace) -> int:
    """Show SDK word declarations without claiming live-state admission."""
    from ..sdk import list_sequences

    _emit(list_sequences(args.name), args.output)
    return 0


def add_study_parsers(sub: argparse._SubParsersAction) -> None:
    """Register stable SDK-backed commands; the SDK owns argument semantics."""
    from ..sdk.study import STUDY_TOPOLOGIES

    network = sub.add_parser(
        "network", help="Create, evolve and diagnose a declared SDK network study"
    )
    network.add_argument("--nodes", type=int)
    network.add_argument("--topology", choices=STUDY_TOPOLOGIES)
    network.add_argument("--seed", type=int)
    network.add_argument("--sequence", help="Registered SDK word; see tnfr sequences")
    network.add_argument(
        "--steps",
        "--cycles",
        dest="cycles",
        type=int,
        help="Number of complete operator-word cycles, not physical time",
    )
    network.add_argument(
        "--probability", type=float, help="Random-topology edge probability"
    )
    network.add_argument("--name", help="Name stored in the study declaration")
    network.add_argument(
        "--spec", type=Path, help="JSON StudySpec, exclusive with construction flags"
    )
    network.add_argument(
        "--export-spec", type=Path, help="Save the successful run's declaration as JSON"
    )
    network.add_argument(
        "--output", type=Path, help="Save result JSON; default: stdout"
    )
    network.set_defaults(func=cmd_network)

    for name, command, help_text in (
        ("operators", cmd_operators, "Inspect the operator contract registry"),
        ("sequences", cmd_sequences, "Inspect declared SDK operator words"),
    ):
        parser = sub.add_parser(name, help=help_text)
        parser.add_argument("name", nargs="?", help="Optional single catalog entry")
        parser.add_argument(
            "--output", type=Path, help="Save catalog JSON; default: stdout"
        )
        parser.set_defaults(func=command)

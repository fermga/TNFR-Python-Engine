#!/usr/bin/env python3
"""Check or regenerate the typed physics facade without importing the engine."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def render_stub(source: Path) -> str:
    """Derive explicit typed re-exports from the sole runtime owner registry."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    registries = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_EXPORT_MODULES"
            for target in node.targets
        )
    ]
    if len(registries) != 1 or not isinstance(registries[0], ast.Dict):
        raise ValueError("Expected one literal _EXPORT_MODULES owner registry")
    registry = registries[0]
    exports = ast.literal_eval(registry)
    if len(exports) != len(registry.keys):
        raise ValueError("The physics owner registry contains duplicate names")
    if not all(
        isinstance(name, str)
        and name.isidentifier()
        and isinstance(owner, str)
        and owner.startswith(".")
        and all(part.isidentifier() for part in owner[1:].split("."))
        for name, owner in exports.items()
    ):
        raise ValueError("Expected public names mapped to relative owner modules")

    lines = [
        '"""Generated typed re-exports for the lazy physics facade.',
        "",
        "Regenerate with: python scripts/generate_physics_stub.py --write",
        "The runtime _EXPORT_MODULES registry is the sole maintained source.",
        '"""',
        "",
        "# isort: skip_file",
        "# Preserve the runtime public export order.",
        "",
    ]
    for name, owner in exports.items():
        statement = f"from {owner} import {name} as {name}"
        if len(statement) <= 88:
            lines.append(statement)
        else:
            lines.extend((f"from {owner} import (", f"    {name} as {name},", ")"))
    lines.extend(("", "__all__ = ["))
    lines.extend(f'    "{name}",' for name in exports)
    lines.extend(("]", ""))
    return "\n".join(lines)


def main() -> int:
    """Check by default, or replace the generated stub when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="check without writing (default)"
    )
    mode.add_argument(
        "--write", action="store_true", help="regenerate the typed facade"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "tnfr" / "physics" / "__init__.py"
    target = source.with_suffix(".pyi")
    expected = render_stub(source)
    if args.write:
        target.write_text(expected, encoding="utf-8", newline="\n")
        print("Updated src/tnfr/physics/__init__.pyi")
        return 0
    if not target.is_file() or target.read_text(encoding="utf-8") != expected:
        parser.exit(
            1,
            "Physics type stub is stale; run "
            "python scripts/generate_physics_stub.py --write\n",
        )
    print("Physics type stub matches the runtime export registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

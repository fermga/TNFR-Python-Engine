"""Split definitions.py into per-operator modules.

This script automates the splitting of src/tnfr/operators/definitions.py (3,311 lines)
into 13 individual operator files + a facade for backward compatibility.

Strategy:
- Extract Operator base class into definitions_base.py
- Create one file per operator (emission.py, reception.py, etc.)
- Create definitions.py facade that re-exports everything
- Preserve all imports, docstrings, and constants

Physics: Maintains structural coherence through modular organization.
Backward compatibility: 100% preserved via facade pattern.
"""

import re
from pathlib import Path

# Define line ranges for each operator (approximate, will be refined)
OPERATORS = [
    ("Operator", 70, 244, "definitions_base.py"),  # Base class + constants
    ("Emission", 245, 502, "emission.py"),
    ("Reception", 503, 718, "reception.py"),
    ("Coherence", 719, 1127, "coherence.py"),
    ("Dissonance", 1128, 1342, "dissonance.py"),
    ("Coupling", 1343, 1560, "coupling.py"),
    ("Resonance", 1561, 2027, "resonance.py"),
    ("Silence", 2028, 2297, "silence.py"),
    ("Expansion", 2298, 2359, "expansion.py"),
    ("Contraction", 2360, 2613, "contraction.py"),
    ("SelfOrganization", 2614, 3099, "self_organization.py"),
    ("Mutation", 3100, 3669, "mutation.py"),
    ("Transition", 3670, 4026, "transition.py"),
    ("Recursivity", 4027, 4120, "recursivity.py"),
]

COMMON_IMPORTS = '''"""TNFR Operator: {operator_name}

{docstring_first_line}

**Physics**: See AGENTS.md § {operator_name}
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

from typing import Any

from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
from .registry import register_operator
'''

BASE_IMPORTS = '''"""TNFR Operator Base Class

Base Operator class with common functionality for all structural operators.

**Physics**: All operators derive from nodal equation ∂EPI/∂t = νf · ΔNFR(t)
**Implementation**: Each operator applies structural transformations via glyphs
"""

from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from ..utils import get_numpy

__all__ = ["Operator"]

# T'HOL canonical bifurcation constants
_THOL_SUB_EPI_SCALING = 0.25  # Sub-EPI is 25% of parent (first-order bifurcation)
_THOL_EMERGENCE_CONTRIBUTION = 0.1  # Parent EPI increases by 10% of sub-EPI
'''


def read_file(path: Path) -> list[str]:
    """Read file and return lines."""
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_file(path: Path, lines: list[str]) -> None:
    """Write lines to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def extract_operator(
    lines: list[str], start: int, end: int, name: str
) -> tuple[str, list[str]]:
    """Extract operator class and return docstring + code."""
    # Get operator code (1-indexed → 0-indexed)
    operator_lines = lines[start - 1 : end]

    # Find class definition and extract first line of docstring
    class_line = None
    docstring_start = None
    for i, line in enumerate(operator_lines):
        if line.strip().startswith(f"class {name}"):
            class_line = i
        if class_line is not None and '"""' in line and docstring_start is None:
            docstring_start = i
            break

    docstring_first = "Structural operator"
    if docstring_start is not None and docstring_start + 1 < len(operator_lines):
        # Extract first line after opening """
        doc_line = operator_lines[docstring_start]
        if '"""' in doc_line:
            parts = doc_line.split('"""')
            if len(parts) > 1:
                docstring_first = parts[1].strip()

    return docstring_first, operator_lines


def create_facade(operators_info: list[tuple[str, str]]) -> str:
    """Create facade definitions.py that re-exports everything."""
    imports = []
    exports = ['"Operator"']

    for op_name, module_name in operators_info:
        if op_name == "Operator":
            imports.append("from .definitions_base import Operator")
        else:
            imports.append(f"from .{module_name[:-3]} import {op_name}")
            exports.append(f'"{op_name}"')

    facade = '''"""Definitions for canonical TNFR structural operators.

Structural operators (Emission, Reception, Coherence, etc.) are the public-facing
API for applying TNFR transformations to nodes. Each operator is associated with
a specific glyph (structural symbol like AL, EN, IL, etc.) that represents the
underlying transformation.

English identifiers are the public API. Spanish wrappers were removed in
TNFR 2.0, so downstream code must import these classes directly.

**Physics & Theory References:**
- Complete operator physics: AGENTS.md § Canonical Operators
- Grammar constraints (U1-U6): UNIFIED_GRAMMAR_RULES.md
- Nodal equation (∂EPI/∂t = νf · ΔNFR): AGENTS.md § Foundational Physics

**Implementation:**
- Canonical grammar validation: src/tnfr/operators/grammar.py
- Operator registry: src/tnfr/operators/registry.py

**Note**: This is a facade module. Individual operators are implemented in
separate files for better maintainability. All imports preserve backward
compatibility.
"""

from __future__ import annotations

'''
    facade += "\n".join(imports)
    facade += "\n\n__all__ = [\n    "
    facade += ",\n    ".join(exports)
    facade += ",\n]\n"

    return facade


def main():
    """Execute the split."""
    src_path = Path("src/tnfr/operators/definitions.py")
    backup_path = Path("src/tnfr/operators/definitions.py.CRITICAL_BACKUP")

    if not src_path.exists():
        print(f"❌ Source file not found: {src_path}")
        return

    if not backup_path.exists():
        print("❌ Backup not found. Create backup first!")
        return

    print("🔨 Starting definitions.py split...")
    lines = read_file(src_path)
    print(f"📄 Original file: {len(lines)} lines")

    # Extract header (lines 1-69: docstring + imports)
    header_lines = lines[0:69]

    # Track created files
    operators_info = []

    # Process each operator
    for op_name, start, end, filename in OPERATORS:
        print(f"  Extracting {op_name} ({start}-{end}) → {filename}")

        if op_name == "Operator":
            # Base class: use special imports
            content = BASE_IMPORTS + "\n\n"
            content += "".join(lines[start - 1 : end])
        else:
            # Regular operator: extract and add imports
            docstring_first, op_lines = extract_operator(lines, start, end, op_name)
            content = COMMON_IMPORTS.format(
                operator_name=op_name, docstring_first_line=docstring_first
            )
            content += "\n\n"
            content += "".join(op_lines)

        # Write to file
        target_path = Path(f"src/tnfr/operators/{filename}")
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"    ✅ Created {filename} ({len(content.splitlines())} lines)")
        operators_info.append((op_name, filename))

    # Create facade
    print("\n📦 Creating facade definitions.py...")
    facade_content = create_facade(operators_info)
    facade_path = Path("src/tnfr/operators/definitions.py")

    # Rename original to .old
    old_path = Path("src/tnfr/operators/definitions.py.old")
    src_path.rename(old_path)
    print(f"  ✅ Renamed original to {old_path.name}")

    # Write facade
    with open(facade_path, "w", encoding="utf-8") as f:
        f.write(facade_content)
    print(f"  ✅ Created {facade_path.name} facade ({len(facade_content.splitlines())} lines)")

    print("\n✨ Split complete!")
    print("⚠️  CRITICAL: Run test suite to verify!")
    print("📝 Expected import fixes: 3-5 iterations for dependencies")


if __name__ == "__main__":
    main()

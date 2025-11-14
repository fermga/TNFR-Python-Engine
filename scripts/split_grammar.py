"""Script to split grammar.py into modular files.

This script divides src/tnfr/operators/grammar.py (2,921 lines) into
specialized modules plus a facade for backward compatibility.

CRITICAL: This is the most important file in TNFR. Handle with extreme care.
"""

import os
import shutil

# Read the original file
with open("src/tnfr/operators/grammar.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Common header for all modules
def make_header(title: str, description: str) -> list[str]:
    return [
        f'"""TNFR Grammar: {title}\n',
        "\n",
        f"{description}\n",
        "\n",
        'Terminology (TNFR semantics):\n',
        '- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility\n',
        '- Future semantic aliasing ("locus") must preserve public API stability\n',
        '"""\n',
        "\n",
        "from __future__ import annotations\n",
        "\n",
    ]

# Extract header (lines 1-85)
header_lines = lines[0:85]

# Define sections (1-indexed, inclusive ranges)
sections = {
    "grammar_types.py": {
        "range": (86, 567),
        "title": "Types and Exceptions",
        "desc": "Enums, exception classes, and validation result types for TNFR grammar.",
    },
    "grammar_context.py": {
        "range": (568, 699),
        "title": "Grammar Context",
        "desc": "Runtime context for grammar validation and operator application tracking.",
    },
    "grammar_core.py": {
        "range": (700, 1581),
        "title": "Core Grammar Validator",
        "desc": "GrammarValidator class - central validation engine for all grammar rules U1-U6.",
    },
    "grammar_u6.py": {
        "range": (1582, 1712),
        "title": "U6 Structural Potential Validation",
        "desc": "U6: STRUCTURAL POTENTIAL CONFINEMENT - Validate Δ Φ_s < 2.0 escape threshold.",
    },
    "grammar_telemetry.py": {
        "range": (1765, 2011),
        "title": "U6 Telemetry Functions",
        "desc": "Phase gradient, phase curvature, and coherence length telemetry for U6 validation.",
    },
    "grammar_application.py": {
        "range": (2017, 2190),
        "title": "Grammar Application",
        "desc": "Functions for applying operators with grammar enforcement at runtime.",
    },
    "grammar_patterns.py": {
        "range": (2191, 2921),
        "title": "Sequence Pattern Recognition",
        "desc": "Sequence validation, parsing, pattern recognition, and optimization helpers.",
    },
}

# Special handling: validate_grammar function (lines 1718-1759) goes in its own module
sections["grammar_validate.py"] = {
    "range": (1713, 1759),
    "title": "Main Validation Entry Point",
    "desc": "Primary validate_grammar() function - the main public API for grammar checking.",
}

print("🔨 Starting grammar.py split...")
print(f"📄 Original file: {len(lines)} lines\n")

# Create each module
for module_name, spec in sorted(sections.items()):
    start, end = spec["range"]
    title = spec["title"]
    desc = spec["desc"]
    
    module_lines = []
    
    # Add header
    module_lines.extend(make_header(title, desc))
    
    # Add necessary imports based on module
    if module_name == "grammar_types.py":
        module_lines.extend([
            "from enum import Enum\n",
            "from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple\n",
            "\n",
            "if TYPE_CHECKING:\n",
            "    from ..types import NodeId, TNFRGraph, Glyph\n",
            "    from .definitions import Operator\n",
            "else:\n",
            "    NodeId = Any\n",
            "    TNFRGraph = Any\n",
            "    from ..types import Glyph\n",
            "\n",
            "from ..validation.base import ValidationOutcome\n",
            "\n",
        ])
    elif module_name == "grammar_context.py":
        module_lines.extend([
            "from typing import Any\n",
            "\n",
            "from .grammar_types import GrammarConfigurationError\n",
            "\n",
        ])
    elif module_name == "grammar_core.py":
        module_lines.extend([
            "from typing import TYPE_CHECKING, Any\n",
            "\n",
            "if TYPE_CHECKING:\n",
            "    from ..types import NodeId, TNFRGraph, Glyph\n",
            "else:\n",
            "    NodeId = Any\n",
            "    TNFRGraph = Any\n",
            "    from ..types import Glyph\n",
            "\n",
            "from .grammar_types import (\n",
            "    StructuralGrammarError,\n",
            "    RepeatWindowError,\n",
            "    MutationPreconditionError,\n",
            "    TholClosureError,\n",
            "    TransitionCompatibilityError,\n",
            "    StructuralPotentialConfinementError,\n",
            "    record_grammar_violation,\n",
            ")\n",
            "from .grammar_context import GrammarContext\n",
            "from ..config.operator_names import (\n",
            "    BIFURCATION_WINDOWS,\n",
            "    CANONICAL_OPERATOR_NAMES,\n",
            "    DESTABILIZERS_MODERATE,\n",
            "    DESTABILIZERS_STRONG,\n",
            "    DESTABILIZERS_WEAK,\n",
            "    INTERMEDIATE_OPERATORS,\n",
            "    SELF_ORGANIZATION,\n",
            "    SELF_ORGANIZATION_CLOSURES,\n",
            "    VALID_END_OPERATORS,\n",
            "    VALID_START_OPERATORS,\n",
            ")\n",
            "from ..validation.compatibility import (\n",
            "    CompatibilityLevel,\n",
            "    get_compatibility_level,\n",
            ")\n",
            "\n",
        ])
    else:
        # Other modules get minimal imports
        module_lines.extend([
            "from typing import Any\n",
            "\n",
        ])
    
    # Add content
    module_lines.extend(lines[start-1:end])
    
    # Write module
    with open(f"src/tnfr/operators/{module_name}", "w", encoding="utf-8") as f:
        f.writelines(module_lines)
    
    print(f"✅ Created {module_name:30s} ({end-start+1:4d} lines)")

# Create facade grammar.py
print("\n📦 Creating facade grammar.py...")

facade_lines = []
facade_lines.extend(header_lines)
facade_lines.append("\n")
facade_lines.append("# Re-export all grammar components for backward compatibility\n")
facade_lines.append("\n")
facade_lines.append("# Types and exceptions\n")
facade_lines.append("from .grammar_types import (\n")
facade_lines.append("    StructuralPattern,\n")
facade_lines.append("    StructuralGrammarError,\n")
facade_lines.append("    RepeatWindowError,\n")
facade_lines.append("    MutationPreconditionError,\n")
facade_lines.append("    TholClosureError,\n")
facade_lines.append("    TransitionCompatibilityError,\n")
facade_lines.append("    StructuralPotentialConfinementError,\n")
facade_lines.append("    SequenceSyntaxError,\n")
facade_lines.append("    SequenceValidationResult,\n")
facade_lines.append("    GrammarConfigurationError,\n")
facade_lines.append("    record_grammar_violation,\n")
facade_lines.append("    glyph_function_name,\n")
facade_lines.append("    function_name_to_glyph,\n")
facade_lines.append(")\n")
facade_lines.append("\n")
facade_lines.append("# Context\n")
facade_lines.append("from .grammar_context import GrammarContext\n")
facade_lines.append("\n")
facade_lines.append("# Core validator\n")
facade_lines.append("from .grammar_core import GrammarValidator\n")
facade_lines.append("\n")
facade_lines.append("# U6 validation\n")
facade_lines.append("from .grammar_u6 import validate_structural_potential_confinement\n")
facade_lines.append("\n")
facade_lines.append("# Main validation entry point\n")
facade_lines.append("from .grammar_validate import validate_grammar\n")
facade_lines.append("\n")
facade_lines.append("# Telemetry\n")
facade_lines.append("from .grammar_telemetry import (\n")
facade_lines.append("    warn_phase_gradient_telemetry,\n")
facade_lines.append("    warn_phase_curvature_telemetry,\n")
facade_lines.append("    warn_coherence_length_telemetry,\n")
facade_lines.append(")\n")
facade_lines.append("\n")
facade_lines.append("# Application\n")
facade_lines.append("from .grammar_application import (\n")
facade_lines.append("    apply_glyph_with_grammar,\n")
facade_lines.append("    on_applied_glyph,\n")
facade_lines.append("    enforce_canonical_grammar,\n")
facade_lines.append(")\n")
facade_lines.append("\n")
facade_lines.append("# Pattern recognition\n")
facade_lines.append("from .grammar_patterns import (\n")
facade_lines.append("    validate_sequence,\n")
facade_lines.append("    parse_sequence,\n")
facade_lines.append("    SequenceValidationResultWithHealth,\n")
facade_lines.append("    validate_sequence_with_health,\n")
facade_lines.append("    recognize_il_sequences,\n")
facade_lines.append("    optimize_il_sequence,\n")
facade_lines.append("    suggest_il_sequence,\n")
facade_lines.append(")\n")
facade_lines.append("\n")
facade_lines.append("__all__ = [\n")
facade_lines.append('    # Types\n')
facade_lines.append('    "StructuralPattern",\n')
facade_lines.append('    # Exceptions\n')
facade_lines.append('    "StructuralGrammarError",\n')
facade_lines.append('    "RepeatWindowError",\n')
facade_lines.append('    "MutationPreconditionError",\n')
facade_lines.append('    "TholClosureError",\n')
facade_lines.append('    "TransitionCompatibilityError",\n')
facade_lines.append('    "StructuralPotentialConfinementError",\n')
facade_lines.append('    "SequenceSyntaxError",\n')
facade_lines.append('    "GrammarConfigurationError",\n')
facade_lines.append('    # Validation\n')
facade_lines.append('    "SequenceValidationResult",\n')
facade_lines.append('    "validate_grammar",\n')
facade_lines.append('    "validate_sequence",\n')
facade_lines.append('    "parse_sequence",\n')
facade_lines.append('    "validate_sequence_with_health",\n')
facade_lines.append('    # U6\n')
facade_lines.append('    "validate_structural_potential_confinement",\n')
facade_lines.append('    # Core\n')
facade_lines.append('    "GrammarContext",\n')
facade_lines.append('    "GrammarValidator",\n')
facade_lines.append('    # Application\n')
facade_lines.append('    "apply_glyph_with_grammar",\n')
facade_lines.append('    "on_applied_glyph",\n')
facade_lines.append('    "enforce_canonical_grammar",\n')
facade_lines.append('    # Helpers\n')
facade_lines.append('    "glyph_function_name",\n')
facade_lines.append('    "function_name_to_glyph",\n')
facade_lines.append('    "record_grammar_violation",\n')
facade_lines.append('    "SequenceValidationResultWithHealth",\n')
facade_lines.append('    "recognize_il_sequences",\n')
facade_lines.append('    "optimize_il_sequence",\n')
facade_lines.append('    "suggest_il_sequence",\n')
facade_lines.append('    # Telemetry\n')
facade_lines.append('    "warn_phase_gradient_telemetry",\n')
facade_lines.append('    "warn_phase_curvature_telemetry",\n')
facade_lines.append('    "warn_coherence_length_telemetry",\n')
facade_lines.append("]\n")

# Backup original
shutil.move("src/tnfr/operators/grammar.py", "src/tnfr/operators/grammar.py.old")

# Write facade
with open("src/tnfr/operators/grammar.py", "w", encoding="utf-8") as f:
    f.writelines(facade_lines)

print("✅ Created grammar.py facade")
print("✅ Renamed original to grammar.py.old")
print("\n✨ Split complete!")
print("\n⚠️  CRITICAL: Run full test suite to verify!")
print("   python -m pytest tests/ -q --tb=short")

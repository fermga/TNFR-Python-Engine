"""Tests covering the consolidated validation re-exports."""

from __future__ import annotations

import importlib


def test_validation_all_includes_runtime_exports() -> None:
    validation = importlib.import_module("tnfr.validation")

    expected_exports = {
        "ValidationOutcome",
        "Validator",
        "GraphCanonicalValidator",
        "apply_canonical_clamps",
        "validate_canon",
        "GRAPH_VALIDATORS",
        "run_validators",
        "CANON_COMPAT",
        "CANON_FALLBACK",
        "validate_window",
        "coerce_glyph",
        "get_norm",
        "glyph_fallback",
        "normalized_dnfr",
        "acceleration_norm",
        "check_repeats",
        "maybe_force",
        "soft_grammar_filters",
        "NFRValidator",
    }

    exported_names = set(validation.__all__)

    missing = expected_exports - exported_names
    assert not missing, f"validation exports missing: {sorted(missing)}"

    for name in expected_exports:
        assert hasattr(validation, name), f"{name} should be accessible on tnfr.validation"

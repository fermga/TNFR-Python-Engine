"""Smoke test for the Fase 2 integration documentation doctest."""
from __future__ import annotations

import doctest
from pathlib import Path

def test_fase2_integration_doc_executes() -> None:
    doc_path = Path("docs/fase2_integration.md")
    result = doctest.testfile(
        str(doc_path),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0, f"Doctest failed for {doc_path}"

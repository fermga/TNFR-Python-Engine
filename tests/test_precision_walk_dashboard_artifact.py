"""Regression tests for precision walk dashboard artifacts."""
from __future__ import annotations

import json
from pathlib import Path


def test_precision_walk_dashboard_has_no_nan_literals() -> None:
    """Dashboard JSON must be parseable without NaN literals."""
    dashboard_path = Path("benchmarks/results/precision_walk_dashboard.json")
    assert dashboard_path.exists(), (
        "Dashboard JSON is missing; regenerate via "
        "tools/precision_walk_dashboard.py"
    )

    content = dashboard_path.read_text()
    assert "NaN" not in content, (
        "Dashboard JSON should never contain NaN literals"
    )

    data = json.loads(content)
    assert "runs" in data and isinstance(data["runs"], list)
    assert "by_topology" in data and isinstance(data["by_topology"], dict)

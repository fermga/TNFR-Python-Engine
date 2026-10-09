"""Cold-process import order for shared runtime and form-observation facades."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "imports",
    (
        "import tnfr.physics; import tnfr.operators",
        "import tnfr.operators; import tnfr.physics",
    ),
)
def test_runtime_facades_are_cold_import_order_safe(
    imports: str, source_tree_environment
) -> None:
    root = Path(__file__).resolve().parents[2]
    # Check actual facade resolution after both import orders, not just import.
    code = (
        imports
        + "\n"
        + """
import importlib
import tnfr.physics as physics
for name in (
    "form_geometry",
    "source_relative_form",
    "runtime_remesh_schedule_block_margin",
    "remesh_schedule_policy_stability",
    "remesh_schedule_relative_defect_stability",
    "runtime_remesh_schedule_relative_defect",
    "binary64_p2_reception_stability",
    "runtime_p2_reception_stage",
):
    module = importlib.import_module("tnfr.physics." + name)
    for exported in module.__all__:
        assert exported in physics.__all__
        assert getattr(physics, exported) is getattr(module, exported)
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=source_tree_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

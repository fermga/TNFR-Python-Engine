"""Integration smoke tests for ``tnfr profile-pipeline``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tnfr.cli import main


def test_cli_profile_pipeline_generates_profiles(tmp_path: Path) -> None:
    """Ensure ``tnfr profile-pipeline`` emits profiling artefacts with NumPy."""

    pytest.importorskip("numpy")

    output_dir = tmp_path / "profiles"
    exit_code = main(
        [
            "profile-pipeline",
            "--nodes",
            "16",
            "--edge-probability",
            "0.2",
            "--loops",
            "1",
            "--seed",
            "2",
            "--output-dir",
            str(output_dir),
            "--sort",
            "cumtime",
            "--si-chunk-sizes",
            "auto",
            "--dnfr-chunk-sizes",
            "auto",
            "--si-workers",
            "auto",
            "--dnfr-workers",
            "auto",
        ]
    )

    assert exit_code == 0

    vectorized_pstats = sorted(output_dir.glob("full_pipeline_vectorized_*.pstats"))
    vectorized_json = sorted(output_dir.glob("full_pipeline_vectorized_*.json"))
    fallback_pstats = sorted(output_dir.glob("full_pipeline_fallback_*.pstats"))
    fallback_json = sorted(output_dir.glob("full_pipeline_fallback_*.json"))

    assert vectorized_pstats, "Expected vectorized profiling dumps to be generated"
    assert vectorized_json, "Expected vectorized profiling summaries to be generated"
    assert fallback_pstats, "Expected fallback profiling dumps to be generated"
    assert fallback_json, "Expected fallback profiling summaries to be generated"

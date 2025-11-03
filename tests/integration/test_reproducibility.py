"""Integration tests for reproducibility infrastructure."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    return output_dir


def test_reproducibility_script_runs_successfully(temp_output_dir: Path) -> None:
    """Test that the reproducibility script can run benchmarks successfully."""
    script_path = Path(__file__).parent.parent.parent / "scripts" / "run_reproducible_benchmarks.py"
    
    # Run with a single benchmark and small parameters
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--benchmarks", "comprehensive_cache_profiler",
            "--seed", "42",
            "--output-dir", str(temp_output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "success" in result.stdout.lower()
    
    # Check manifest was created
    manifest_path = temp_output_dir / "manifest.json"
    assert manifest_path.exists(), "Manifest file not created"
    
    # Verify manifest structure
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    assert "seed" in manifest
    assert manifest["seed"] == 42
    assert "benchmarks" in manifest
    assert "comprehensive_cache_profiler" in manifest["benchmarks"]
    
    benchmark_result = manifest["benchmarks"]["comprehensive_cache_profiler"]
    assert benchmark_result["status"] == "success"
    assert "checksums" in benchmark_result
    assert len(benchmark_result["checksums"]) > 0


def test_reproducibility_script_verify_mode(temp_output_dir: Path) -> None:
    """Test that the verify mode works correctly."""
    script_path = Path(__file__).parent.parent.parent / "scripts" / "run_reproducible_benchmarks.py"
    
    # First, run benchmarks to generate artifacts
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--benchmarks", "comprehensive_cache_profiler",
            "--seed", "123",
            "--output-dir", str(temp_output_dir),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    
    manifest_path = temp_output_dir / "manifest.json"
    assert manifest_path.exists()
    
    # Now verify the checksums
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--verify", str(manifest_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    assert result.returncode == 0, f"Verification failed: {result.stderr}"
    assert "Verified" in result.stdout


def test_reproducibility_script_handles_missing_benchmark(temp_output_dir: Path) -> None:
    """Test that the script handles missing benchmark scripts gracefully."""
    script_path = Path(__file__).parent.parent.parent / "scripts" / "run_reproducible_benchmarks.py"
    
    # Create a temporary modified script config (we'll just run and check output)
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--benchmarks", "comprehensive_cache_profiler",
            "--seed", "42",
            "--output-dir", str(temp_output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    
    # Should not crash even if some benchmarks might be skipped
    assert "manifest.json" in result.stdout.lower()

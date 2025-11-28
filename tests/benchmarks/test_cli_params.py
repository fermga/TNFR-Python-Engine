import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
BENCH_DIR = PROJECT_ROOT / "benchmarks"


def run_script(script: str, args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(BENCH_DIR / script), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    return proc


def test_asymptotic_freedom_cli_minimal() -> None:
    # Minimal nodes/seeds for speed; quiet mode suppresses output
    proc = run_script(
        "asymptotic_freedom_test.py",
        ["--nodes", "12", "--seeds", "2", "--topologies", "ring", "--quiet"],
    )
    assert proc.returncode == 0


def test_confinement_zones_cli_minimal() -> None:
    proc = run_script(
        "confinement_zones_test.py",
        ["--nodes", "12", "--seeds", "2", "--topologies", "ring", "--quiet"],
    )
    assert proc.returncode == 0


def test_coherence_length_cli_minimal() -> None:
    # Uses existing CLI; param grid resolution small to keep runtime short
    proc = run_script(
        "coherence_length_critical_exponent.py",
        [
            "--nodes", "12",
            "--seeds", "2",
            "--topologies", "ring",
            "--param-grid-resolution", "coarse",
            "--dry-run",
            "--quiet",
        ],
    )
    assert proc.returncode == 0


def test_bifurcation_landscape_cli_dry_run() -> None:
    """Test bifurcation benchmark dry-run validation."""
    proc = run_script(
        "bifurcation_landscape.py",
        [
            "--nodes",
            "20",
            "--seeds",
            "2",
            "--topologies",
            "ring",
            "--oz-intensity-grid",
            "0.5,0.75",
            "--mutation-thresholds",
            "0.2,0.4",
            "--vf-grid",
            "0.5,0.6",
            "--dry-run",
            "--quiet",
        ],
    )
    assert proc.returncode == 0


def test_bifurcation_landscape_cli_real_run() -> None:
    """Test bifurcation benchmark with minimal real execution.
    
    Validates JSONL output structure and presence of all metric keys.
    """
    proc = run_script(
        "bifurcation_landscape.py",
        [
            "--nodes", "8",
            "--seeds", "1",
            "--topologies", "ring",
            "--oz-intensity-grid", "0.5",
            "--mutation-thresholds", "0.3",
            "--vf-grid", "0.6",
            # No --quiet here; we need JSONL output for validation
        ],
    )
    assert proc.returncode == 0, "Benchmark execution failed"
    
    # Parse JSONL output (filter out the summary line)
    lines = [
        line.strip() for line in proc.stdout.split("\n")
        if line.strip() and line.strip().startswith("{")
    ]
    assert len(lines) > 0, "No JSONL output produced"
    
    # Validate first record has all expected keys
    record = json.loads(lines[0])
    expected_keys = {
        "topology", "intensity_oz",
        "mutation_threshold", "vf_scale", "seed",
        "delta_phi_s", "delta_phase_gradient_max",
        "delta_phase_curvature_max", "coherence_length_ratio",
        "delta_dnfr_variance", "bifurcation_score_max",
        "handlers_present", "classification",
    }
    missing = expected_keys - set(record.keys())
    assert not missing, f"Missing keys in output: {missing}"
    
    # Validate classification is one of the expected values
    assert record["classification"] in (
        "none", "incipient", "bifurcation", "fragmentation"
    ), f"Invalid classification: {record['classification']}"

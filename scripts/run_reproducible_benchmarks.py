#!/usr/bin/env python3
"""Run benchmarks with deterministic seeds and generate checksums for artifacts.

This script ensures reproducibility by:
1. Setting global seeds for all benchmarks
2. Running selected benchmarks with consistent parameters
3. Generating checksums (SHA256) for all output artifacts
4. Creating a manifest file with checksums for verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


# Default global seed for all benchmarks
DEFAULT_SEED = 42

# Output truncation limits
MAX_STDERR_LENGTH = 500
MAX_STDOUT_PREVIEW_LENGTH = 200

# Benchmark configurations with their default parameters
BENCHMARK_CONFIGS = {
    "comprehensive_cache_profiler": {
        "script": "benchmarks/comprehensive_cache_profiler.py",
        "args": ["--nodes", "100", "--steps", "20"],
        "output_arg": "--output",
        "output_format": "json",
        "output_suffix": ".json",
        "supports_seed": True,
    },
    "full_pipeline_profile": {
        "script": "benchmarks/full_pipeline_profile.py",
        "args": ["--nodes", "100", "--edge-probability", "0.25", "--loops", "3"],
        "output_arg": "--output-dir",
        "output_format": "dir",
        "output_suffix": "",
        "supports_seed": True,
    },
    "cache_hot_path_profiler": {
        "script": "benchmarks/cache_hot_path_profiler.py",
        "args": ["--nodes", "100", "--steps", "20"],
        "output_arg": "--output",
        "output_format": "json",
        "output_suffix": ".json",
        "supports_seed": True,
    },
    "compute_si_profile": {
        "script": "benchmarks/compute_si_profile.py",
        "args": ["--nodes", "100", "--loops", "5"],
        "output_arg": "--output-dir",
        "output_format": "dir",
        "output_suffix": "",
        "extra_args": ["--format", "json"],
        "supports_seed": False,  # Uses deterministic chord graph
    },
}


def compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_benchmark(
    name: str,
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a single benchmark and return its results."""
    script_path = Path(config["script"])
    if not script_path.exists():
        print(f"Warning: Benchmark script {script_path} not found, skipping.")
        return {"status": "skipped", "reason": "script not found"}

    # Prepare output file/directory
    output_suffix = config.get("output_suffix", ".json")
    output_path = output_dir / f"{name}_seed{seed}{output_suffix}"
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        *config["args"],
    ]
    
    # Add seed if supported
    if config.get("supports_seed", True):
        cmd.extend(["--seed", str(seed)])
    
    # Add extra args if specified
    if "extra_args" in config:
        cmd.extend(config["extra_args"])
    
    # Add output argument if benchmark supports it
    output_arg = config.get("output_arg")
    if output_arg:
        cmd.extend([output_arg, str(output_path)])
    
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd() / "src")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Error running {name}:")
            print(result.stderr)
            return {
                "status": "failed",
                "returncode": result.returncode,
                "stderr": result.stderr[:MAX_STDERR_LENGTH],  # First N chars
            }
        
        # Compute checksum based on output format
        checksums = {}
        output_files = []
        
        output_format = config.get("output_format", "none")
        if output_format == "dir" and output_path.exists() and output_path.is_dir():
            # Directory output: checksum all JSON files
            for json_file in output_path.glob("*.json"):
                filename = json_file.name
                checksums[filename] = compute_checksum(json_file)
                output_files.append(str(json_file))
        elif output_path.exists() and output_path.is_file():
            # Single file output
            filename = output_path.name
            checksums[filename] = compute_checksum(output_path)
            output_files.append(str(output_path))
        
        # If no output file was created, save stdout
        if not output_files and result.stdout:
            stdout_file = output_dir / f"{name}_seed{seed}_stdout.txt"
            stdout_file.write_text(result.stdout)
            checksums["stdout.txt"] = compute_checksum(stdout_file)
            output_files.append(str(stdout_file))
        
        return {
            "status": "success",
            "output_files": output_files,
            "checksums": checksums,
            "stdout_preview": result.stdout[-MAX_STDOUT_PREVIEW_LENGTH:] if verbose else "",
        }
    
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Global seed for all benchmarks (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store benchmark outputs (default: artifacts)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(BENCHMARK_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to output manifest file with checksums (default: <output-dir>/manifest.json)",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify checksums against an existing manifest file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Verify mode
    if args.verify:
        return verify_manifest(args.verify, args.verbose)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which benchmarks to run
    benchmarks_to_run = (
        list(BENCHMARK_CONFIGS.keys())
        if "all" in args.benchmarks
        else args.benchmarks
    )
    
    # Run benchmarks
    print(f"Running {len(benchmarks_to_run)} benchmark(s) with seed={args.seed}")
    results: dict[str, Any] = {
        "seed": args.seed,
        "benchmarks": {},
    }
    
    for name in benchmarks_to_run:
        print(f"\n[{name}]")
        config = BENCHMARK_CONFIGS[name]
        result = run_benchmark(name, config, args.output_dir, args.seed, args.verbose)
        results["benchmarks"][name] = result
        
        status_symbol = "✓" if result["status"] == "success" else "✗"
        print(f"{status_symbol} {result['status']}")
        
        if result["status"] == "success":
            checksums = result.get("checksums", {})
            if checksums:
                print(f"  Generated {len(checksums)} artifact(s)")
                for name_part, checksum in list(checksums.items())[:3]:
                    print(f"    {name_part}: {checksum[:16]}...")
                if len(checksums) > 3:
                    print(f"    ... and {len(checksums) - 3} more")
    
    # Save manifest
    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nManifest saved to: {manifest_path}")
    
    # Count successes
    successes = sum(
        1 for r in results["benchmarks"].values() if r["status"] == "success"
    )
    print(f"Successfully ran {successes}/{len(benchmarks_to_run)} benchmarks")
    
    return 0 if successes == len(benchmarks_to_run) else 1


def verify_manifest(manifest_path: Path, verbose: bool = False) -> int:
    """Verify checksums against a manifest file."""
    if not manifest_path.exists():
        print(f"Error: Manifest file {manifest_path} not found")
        return 1
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"Verifying checksums from: {manifest_path}")
    print(f"Manifest seed: {manifest.get('seed', 'unknown')}")
    
    mismatches = []
    missing = []
    verified = 0
    
    for benchmark_name, result in manifest.get("benchmarks", {}).items():
        if result.get("status") != "success":
            continue
        
        checksums = result.get("checksums", {})
        output_files = result.get("output_files", [])
        
        for file_name, expected_checksum in checksums.items():
            # Find the actual file by matching filename
            actual_file = None
            for output_path in output_files:
                file_path = Path(output_path)
                if file_path.name == file_name or file_path.name.endswith(file_name):
                    actual_file = file_path
                    break
            
            if not actual_file or not actual_file.exists():
                missing.append(f"{benchmark_name}/{file_name}")
                print(f"✗ {benchmark_name}/{file_name}: file missing")
                continue
            
            actual_checksum = compute_checksum(actual_file)
            if actual_checksum == expected_checksum:
                verified += 1
                if verbose:
                    print(f"✓ {benchmark_name}/{file_name}: verified")
            else:
                mismatches.append(f"{benchmark_name}/{file_name}")
                print(f"✗ {benchmark_name}/{file_name}: checksum mismatch")
                if verbose:
                    print(f"  Expected: {expected_checksum}")
                    print(f"  Actual:   {actual_checksum}")
    
    print(f"\nVerified {verified} checksum(s)")
    
    if mismatches:
        print(f"Mismatches: {', '.join(mismatches)}")
    if missing:
        print(f"Missing files: {', '.join(missing)}")
    
    return 0 if not (mismatches or missing) else 1


if __name__ == "__main__":
    sys.exit(main())

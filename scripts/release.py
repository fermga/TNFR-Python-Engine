#!/usr/bin/env python3
"""
Release automation script for TNFR-Python-Engine.

This script helps automate the release process to PyPI and ensures
all metadata is properly updated.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        sys.exit(1)
    return result


def update_version_in_citation(version):
    """Update version in CITATION.cff."""
    citation_file = Path("CITATION.cff")
    if citation_file.exists():
        content = citation_file.read_text()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('version:'):
                lines[i] = f'version: "{version}"'
                break
        citation_file.write_text('\n'.join(lines))
        print(f"Updated CITATION.cff with version {version}")


def main():
    parser = argparse.ArgumentParser(description="Release TNFR to PyPI")
    parser.add_argument("version", help="Version to release (e.g., 9.0.1)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    
    args = parser.parse_args()
    version = args.version
    
    # Ensure we're in the correct directory
    if not Path("pyproject.toml").exists():
        print("Error: Must be run from the project root directory")
        sys.exit(1)
    
    print(f"Preparing release {version}")
    
    # Update version in citation file
    update_version_in_citation(version)
    
    # Run tests unless skipped
    if not args.skip_tests:
        print("Running tests...")
        run_command([sys.executable, "-m", "pytest", "tests/", "-x"])
    
    # Clean previous builds
    print("Cleaning previous builds...")
    run_command([sys.executable, "-c", "import shutil; shutil.rmtree('dist', ignore_errors=True)"])
    run_command([sys.executable, "-c", "import shutil; shutil.rmtree('build', ignore_errors=True)"])
    
    # Build the package
    print("Building package...")
    run_command([sys.executable, "-m", "build"])
    
    # Check the distribution
    print("Checking distribution...")
    run_command([sys.executable, "-m", "twine", "check", "dist/*"])
    
    if args.dry_run:
        print("Dry run complete. To actually release, run without --dry-run")
        return
    
    # Create and push git tag
    tag = f"v{version}"
    print(f"Creating git tag {tag}")
    run_command(["git", "tag", tag])
    run_command(["git", "push", "origin", tag])
    
    # Upload to PyPI
    print("Uploading to PyPI...")
    run_command([sys.executable, "-m", "twine", "upload", "dist/*"])
    
    print(f"Successfully released version {version}")
    print(f"PyPI: https://pypi.org/project/tnfr/{version}/")
    print("GitHub Actions will handle Zenodo integration automatically.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Modern Windows shim for TNFR Makefile targets."""

import os
import sys
import subprocess
from pathlib import Path

# Set UTF-8 encoding for Windows console
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = Path(__file__).resolve().parents[1]


def run_command(command, description=None):
    """Execute a command and handle errors."""
    if description:
        print(f">> {description}")
    
    try:
        result = subprocess.run(command, shell=True, cwd=ROOT, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print_help()
        return 1
    
    target = sys.argv[1]
    
    # Define modern targets
    targets = {
        'help': print_help,
        'clean': clean,
        'hello': run_hello,
        'validate': validate,
        'examples': run_examples,
        'test': run_tests
    }
    
    if target in targets:
        try:
            targets[target]()
            return 0
        except Exception as e:
            print(f"ERROR: Error executing {target}: {e}")
            return 1
    else:
        print(f"ERROR: Unknown target: {target}")
        print_help()
        return 1


def print_help():
    """Print help message."""
    print("~ TNFR Modern Build System ~")
    print("=" * 30)
    print()
    print("Available targets:")
    print("  help          - Show this help")
    print("  clean         - Remove generated artifacts")
    print("  hello         - Run hello world example")
    print("  validate      - Quick validation check")
    print("  examples      - Run key examples")
    print("  test          - Run core tests")
    print()
    print("Usage: .\\make.cmd <target>")


def clean():
    """Clean generated artifacts."""
    print(">> Cleaning generated artifacts...")
    
    dirs_to_clean = ['examples/output', 'results', 'dist', 'build']
    
    for dir_name in dirs_to_clean:
        dir_path = ROOT / dir_name
        if dir_path.exists():
            run_command(f'rmdir /s /q "{dir_path}"', f"Removing {dir_name}")
    
    print("OK: Clean complete")


def run_hello():
    """Run hello world example."""
    print(">> Running Hello World example...")
    
    # Ensure output directory exists
    output_dir = ROOT / 'examples' / 'output'
    output_dir.mkdir(exist_ok=True)
    
    success = run_command(
        'python examples/01_hello_world.py',
        "Executing hello world demo"
    )
    
    if success:
        print("OK: Hello World complete")
    else:
        print("ERROR: Hello World failed")


def validate():
    """Quick validation check."""
    print(">> Running quick validation...")
    
    # Test TNFR import
    try:
        import tnfr  # noqa: F401
        print("OK: TNFR import: OK")
    except ImportError as e:
        print(f"ERROR: TNFR import: FAILED - {e}")
        return False
    
    # Test hello world (skip output due to encoding)
    try:
        result = subprocess.run(
            ['python', 'examples/01_hello_world.py'],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
        success = result.returncode == 0
    except Exception:
        success = False
    
    if success:
        print("OK: Hello World: OK")
    else:
        print("ERROR: Hello World: FAILED")
    
    # Test repository validation
    success = run_command(
        'python -m pytest tests/test_repository_validation.py -q',
        "Testing repository structure"
    )
    
    if success:
        print("OK: Repository: OK")
    else:
        print("ERROR: Repository: FAILED")
    
    print("OK: Quick validation complete")


def run_examples():
    """Run essential examples."""
    print(">> Running essential examples...")
    
    examples = [
        ('examples/01_hello_world.py', 'Hello World'),
        ('examples/atom_atlas.py', 'Atom Atlas'),
        ('examples/periodic_table_atlas.py', 'Periodic Table')
    ]
    
    # Ensure output directory exists
    output_dir = ROOT / 'examples' / 'output'
    output_dir.mkdir(exist_ok=True)
    
    for script, name in examples:
        success = run_command(f'python {script}', f"Running {name}")
        if success:
            print(f"OK: {name}: Complete")
        else:
            print(f"ERROR: {name}: Failed")
    
    print("OK: All essential examples complete")


def run_tests():
    """Run core test suite."""
    print(">> Running core TNFR test suite...")
    
    test_dirs = [
        'tests/core_physics',
        'tests/grammar',
        'tests/operators',
        'tests/physics'
    ]
    
    for test_dir in test_dirs:
        success = run_command(
            f'python -m pytest {test_dir} -v --tb=short',
            f"Testing {test_dir}"
        )
        if not success:
            print(f"ERROR: Tests in {test_dir} failed")
            return False
    
    print("OK: Core tests complete")


if __name__ == '__main__':
    sys.exit(main())

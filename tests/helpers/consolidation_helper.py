"""Helper module for consolidating redundant test patterns.

This module provides utilities to identify and document test redundancies
across the test suite, helping maintain DRY principles.

Usage:
    python -m tests.helpers.consolidation_helper
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def extract_test_functions(filepath: Path) -> List[Tuple[str, int]]:
    """Extract test function names and line numbers from a file.

    Parameters
    ----------
    filepath : Path
        Path to the Python test file

    Returns
    -------
    List[Tuple[str, int]]
        List of (test_name, line_number) tuples
    """
    tests = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            match = re.match(r'^def (test_\w+)\(', line.strip())
            if match:
                tests.append((match.group(1), line_num))
    return tests

def find_similar_tests(test_dirs: List[Path]) -> Dict[str, List[Tuple[Path, str, int]]]:
    """Find tests with similar names across directories.

    Parameters
    ----------
    test_dirs : List[Path]
        List of test directories to analyze

    Returns
    -------
    Dict[str, List[Tuple[Path, str, int]]]
        Mapping of test pattern to list of (file, test_name, line_number)
    """
    # Common patterns that indicate potential redundancy
    patterns = [
        'conservation',
        'hermitian',
        'operator.*dimension',
        'validator.*bounds',
        'homogeneous.*stability',
        'dnfr.*balanced',
        'phase.*sync',
        'topology',
        'composition',
    ]

    similar_tests = defaultdict(list)

    for test_dir in test_dirs:
        if not test_dir.exists():
            continue

        for test_file in test_dir.rglob("test_*.py"):
            tests = extract_test_functions(test_file)

            for test_name, line_num in tests:
                for pattern in patterns:
                    if re.search(pattern, test_name, re.IGNORECASE):
                        similar_tests[pattern].append((test_file, test_name, line_num))

    return similar_tests

def analyze_redundancy():
    """Analyze test redundancy and print report."""
    test_dirs = [
        Path("tests/integration"),
        Path("tests/mathematics"),
        Path("tests/property"),
        Path("tests/stress"),
    ]

    similar = find_similar_tests(test_dirs)

    print("=" * 80)
    print("TEST REDUNDANCY ANALYSIS REPORT")
    print("=" * 80)
    print()

    for pattern, matches in sorted(similar.items()):
        if len(matches) <= 1:
            continue

        print(f"\n{pattern.upper()} Pattern ({len(matches)} tests):")
        print("-" * 40)

        # Group by directory
        by_dir = defaultdict(list)
        for filepath, test_name, line_num in matches:
            dir_name = filepath.parts[0] if len(filepath.parts) > 1 else "unknown"
            by_dir[dir_name].append((filepath, test_name, line_num))

        for dir_name, tests in sorted(by_dir.items()):
            print(f"  {dir_name}: {len(tests)} tests")
            for filepath, test_name, line_num in tests[:2]:  # Show first 2
                rel_path = str(filepath).replace(str(test_dirs[0].parent) + "/", "")
                print(f"    - {rel_path}:{line_num} {test_name}")
            if len(tests) > 2:
                print(f"    ... and {len(tests) - 2} more")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
Based on analysis:

1. Tests with similar names in different directories may be redundant IF:
   - They test the same structural properties
   - They use the same validation logic
   - They differ only in scale or parameters

2. Tests are NOT redundant IF:
   - Property tests use Hypothesis (fuzzing vs deterministic)
   - Stress tests focus on performance (vs correctness)
   - Mathematics tests verify operator contracts (vs integration)

3. To consolidate redundant tests:
   - Use parametrized fixtures in shared helpers
   - Document what each test consolidates
   - Keep property/stress tests separate (different purpose)
    """)

if __name__ == "__main__":
    analyze_redundancy()

#!/usr/bin/env python3
"""Audit script to analyze health of all sequences in examples.

This script scans all example files and the SDK to find TNFR operator sequences,
analyzes their structural health using the SequenceHealthAnalyzer, and reports
which sequences need optimization (health < 0.7).

Usage:
    python tools/audit_example_health.py [--verbose] [--output FILE]
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.config.operator_names import ALL_OPERATOR_NAMES


def extract_sequences_from_file(filepath: Path) -> List[Tuple[int, List[str], str]]:
    """Extract operator sequences from a Python file.
    
    Parameters
    ----------
    filepath : Path
        Path to the Python file
        
    Returns
    -------
    List[Tuple[int, List[str], str]]
        List of (line_number, sequence, context) tuples
    """
    sequences = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        
        # Look for list literals that contain operator names
        for node in ast.walk(tree):
            if isinstance(node, ast.List):
                # Check if this is a list of strings
                if all(isinstance(elt, ast.Constant) and isinstance(elt.value, str) 
                       for elt in node.elts):
                    sequence = [elt.value for elt in node.elts]
                    
                    # Check if this looks like an operator sequence
                    # (at least 3 operators, all valid operator names)
                    if len(sequence) >= 3:
                        valid_count = sum(1 for op in sequence if op in ALL_OPERATOR_NAMES)
                        if valid_count >= len(sequence) * 0.8:  # 80% valid operators
                            # Get context (variable name or assignment target)
                            context = "unknown"
                            # Try to find the parent assignment
                            for parent in ast.walk(tree):
                                if hasattr(parent, 'value') and parent.value == node:
                                    if isinstance(parent, ast.Assign):
                                        if parent.targets:
                                            target = parent.targets[0]
                                            if isinstance(target, ast.Name):
                                                context = target.id
                                    break
                            
                            sequences.append((node.lineno, sequence, context))
        
        # Also look for dictionary values in NAMED_SEQUENCES
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant) and 
                        isinstance(value, ast.List)):
                        if all(isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                               for elt in value.elts):
                            sequence = [elt.value for elt in value.elts]
                            if len(sequence) >= 3:
                                valid_count = sum(1 for op in sequence if op in ALL_OPERATOR_NAMES)
                                if valid_count >= len(sequence) * 0.8:
                                    context = key.value if isinstance(key.value, str) else "dict_value"
                                    sequences.append((value.lineno, sequence, context))
    
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)
    
    return sequences


def audit_repository(
    examples_dir: Path,
    sdk_dir: Path,
    verbose: bool = False
) -> Dict[str, List[Dict]]:
    """Audit all sequences in the repository.
    
    Parameters
    ----------
    examples_dir : Path
        Path to examples directory
    sdk_dir : Path
        Path to SDK source directory
    verbose : bool, optional
        Print detailed progress
        
    Returns
    -------
    Dict[str, List[Dict]]
        Results grouped by file
    """
    analyzer = SequenceHealthAnalyzer()
    results = {}
    
    # Scan examples
    example_files = sorted(examples_dir.glob("*.py"))
    
    if verbose:
        print(f"Scanning {len(example_files)} example files...")
    
    for filepath in example_files:
        if verbose:
            print(f"  Analyzing {filepath.name}...")
        
        sequences = extract_sequences_from_file(filepath)
        
        if sequences:
            file_results = []
            for line_num, sequence, context in sequences:
                health = analyzer.analyze_health(sequence)
                
                file_results.append({
                    "line": line_num,
                    "context": context,
                    "sequence": sequence,
                    "health": {
                        "overall": health.overall_health,
                        "coherence": health.coherence_index,
                        "balance": health.balance_score,
                        "sustainability": health.sustainability_index,
                        "completeness": health.pattern_completeness,
                        "pattern": health.dominant_pattern,
                    },
                    "recommendations": health.recommendations,
                    "needs_optimization": health.overall_health < 0.7,
                })
            
            if file_results:
                results[str(filepath.relative_to(repo_root))] = file_results
    
    # Scan SDK
    sdk_files = sorted(sdk_dir.glob("*.py"))
    
    if verbose:
        print(f"Scanning {len(sdk_files)} SDK files...")
    
    for filepath in sdk_files:
        if verbose:
            print(f"  Analyzing {filepath.name}...")
        
        sequences = extract_sequences_from_file(filepath)
        
        if sequences:
            file_results = []
            for line_num, sequence, context in sequences:
                health = analyzer.analyze_health(sequence)
                
                file_results.append({
                    "line": line_num,
                    "context": context,
                    "sequence": sequence,
                    "health": {
                        "overall": health.overall_health,
                        "coherence": health.coherence_index,
                        "balance": health.balance_score,
                        "sustainability": health.sustainability_index,
                        "completeness": health.pattern_completeness,
                        "pattern": health.dominant_pattern,
                    },
                    "recommendations": health.recommendations,
                    "needs_optimization": health.overall_health < 0.7,
                })
            
            if file_results:
                results[str(filepath.relative_to(repo_root))] = file_results
    
    return results


def print_summary(results: Dict[str, List[Dict]]) -> None:
    """Print human-readable summary of audit results.
    
    Parameters
    ----------
    results : Dict[str, List[Dict]]
        Audit results from audit_repository
    """
    print("\n" + "="*80)
    print("TNFR SEQUENCE HEALTH AUDIT SUMMARY")
    print("="*80)
    
    total_sequences = sum(len(seqs) for seqs in results.values())
    needs_optimization = sum(
        1 for seqs in results.values() 
        for seq in seqs 
        if seq["needs_optimization"]
    )
    
    print(f"\nTotal sequences found: {total_sequences}")
    print(f"Sequences needing optimization (health < 0.7): {needs_optimization}")
    if total_sequences > 0:
        print(f"Optimization rate: {needs_optimization/total_sequences*100:.1f}%")
    else:
        print(f"Optimization rate: N/A (no sequences found)")
    
    print("\n" + "-"*80)
    print("SEQUENCES NEEDING OPTIMIZATION")
    print("-"*80)
    
    for filepath, file_results in sorted(results.items()):
        sequences_to_optimize = [s for s in file_results if s["needs_optimization"]]
        
        if sequences_to_optimize:
            print(f"\n📄 {filepath}")
            for seq_info in sequences_to_optimize:
                print(f"  Line {seq_info['line']:4d} | {seq_info['context']:20s} | Health: {seq_info['health']['overall']:.3f}")
                print(f"           | Sequence: {' → '.join(seq_info['sequence'])}")
                if seq_info['recommendations']:
                    print(f"           | Issues: {seq_info['recommendations'][0][:60]}...")
    
    print("\n" + "-"*80)
    print("HEALTHY SEQUENCES (health >= 0.7)")
    print("-"*80)
    
    for filepath, file_results in sorted(results.items()):
        healthy_sequences = [s for s in file_results if not s["needs_optimization"]]
        
        if healthy_sequences:
            print(f"\n📄 {filepath}")
            for seq_info in healthy_sequences:
                print(f"  Line {seq_info['line']:4d} | {seq_info['context']:20s} | Health: {seq_info['health']['overall']:.3f} | Pattern: {seq_info['health']['pattern']}")


def main():
    """Run audit script."""
    parser = argparse.ArgumentParser(
        description="Audit structural health of TNFR sequences in examples"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Run audit
    examples_dir = repo_root / "examples"
    sdk_dir = repo_root / "src" / "tnfr" / "sdk"
    
    results = audit_repository(examples_dir, sdk_dir, verbose=args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    
    # Exit with error code if optimizations needed
    needs_optimization = sum(
        1 for seqs in results.values() 
        for seq in seqs 
        if seq["needs_optimization"]
    )
    
    if needs_optimization > 0:
        print(f"\n⚠️  {needs_optimization} sequence(s) need optimization")
        return 1
    else:
        print(f"\n✓ All sequences have good health (>= 0.7)")
        return 0


if __name__ == "__main__":
    sys.exit(main())

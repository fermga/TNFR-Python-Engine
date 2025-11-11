#!/usr/bin/env python3
"""
Centralized Documentation Synchronization Tool

Synchronizes all TNFR grammar documentation with implementation in grammar.py.
Single source of truth for documentation sync tasks.

Usage:
    python tools/sync_documentation.py --audit      # Audit only
    python tools/sync_documentation.py --sync       # Full sync
    python tools/sync_documentation.py --validate   # Validate examples
    python tools/sync_documentation.py --all        # Everything
"""

import argparse
import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tnfr.operators import grammar
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Coupling,
    Resonance, Silence, Expansion, Contraction, SelfOrganization,
    Mutation, Transition, Recursivity
)


class DocumentationSynchronizer:
    """Centralized tool for synchronizing documentation with implementation."""
    
    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self.grammar_py = repo_root / "src" / "tnfr" / "operators" / "grammar.py"
        self.docs_dir = repo_root / "docs" / "grammar"
        self.examples_dir = self.docs_dir / "examples"
        self.schema_file = self.docs_dir / "schemas" / "canonical-operators.json"
        
        self.audit_report = {
            "functions": {},
            "operator_sets": {},
            "examples": {},
            "cross_references": [],
            "issues": []
        }
    
    def run_full_sync(self):
        """Execute complete documentation synchronization."""
        print("=" * 70)
        print("TNFR Documentation Synchronization - Full Sync")
        print("=" * 70)
        
        # Step 1: Audit
        print("\n[1/5] Auditing grammar.py...")
        self.audit_grammar_py()
        
        # Step 2: Update docstrings
        print("\n[2/5] Analyzing docstrings...")
        self.analyze_docstrings()
        
        # Step 3: Check cross-references
        print("\n[3/5] Checking cross-references...")
        self.check_cross_references()
        
        # Step 4: Validate examples
        print("\n[4/5] Validating examples...")
        self.validate_examples()
        
        # Step 5: Validate schema
        print("\n[5/5] Validating schema...")
        self.validate_schema()
        
        # Generate report
        print("\n" + "=" * 70)
        print("SYNCHRONIZATION REPORT")
        print("=" * 70)
        self.generate_report()
        
        return self.audit_report
    
    def audit_grammar_py(self):
        """Audit all functions in grammar.py."""
        print(f"  Reading {self.grammar_py}...")
        
        # Operator sets
        self.audit_report["operator_sets"] = {
            "GENERATORS": list(grammar.GENERATORS),
            "CLOSURES": list(grammar.CLOSURES),
            "STABILIZERS": list(grammar.STABILIZERS),
            "DESTABILIZERS": list(grammar.DESTABILIZERS),
            "COUPLING_RESONANCE": list(grammar.COUPLING_RESONANCE),
            "BIFURCATION_TRIGGERS": list(grammar.BIFURCATION_TRIGGERS),
            "BIFURCATION_HANDLERS": list(grammar.BIFURCATION_HANDLERS),
            "TRANSFORMERS": list(grammar.TRANSFORMERS),
        }
        print(f"  ✓ Found {len(self.audit_report['operator_sets'])} operator sets")
        
        # Functions
        public_functions = [
            name for name in dir(grammar)
            if not name.startswith('_') and callable(getattr(grammar, name))
        ]
        
        for func_name in public_functions:
            func = getattr(grammar, func_name)
            if inspect.isfunction(func) or inspect.ismethod(func):
                self.audit_report["functions"][func_name] = {
                    "signature": str(inspect.signature(func)),
                    "docstring": inspect.getdoc(func) or "NO DOCSTRING",
                    "has_docstring": inspect.getdoc(func) is not None,
                    "module": func.__module__,
                }
        
        print(f"  ✓ Audited {len(self.audit_report['functions'])} functions")
        
        # GrammarValidator methods
        validator_methods = [
            name for name in dir(grammar.GrammarValidator)
            if not name.startswith('_') and callable(getattr(grammar.GrammarValidator, name))
        ]
        
        for method_name in validator_methods:
            method = getattr(grammar.GrammarValidator, method_name)
            full_name = f"GrammarValidator.{method_name}"
            self.audit_report["functions"][full_name] = {
                "signature": str(inspect.signature(method)),
                "docstring": inspect.getdoc(method) or "NO DOCSTRING",
                "has_docstring": inspect.getdoc(method) is not None,
                "class": "GrammarValidator",
            }
        
        print(f"  ✓ Audited {len(validator_methods)} GrammarValidator methods")
    
    def analyze_docstrings(self):
        """Analyze quality of docstrings."""
        print("  Checking docstring quality...")
        
        critical_functions = [
            "validate_grammar",
            "GrammarValidator.validate",
            "GrammarValidator.validate_initiation",
            "GrammarValidator.validate_closure",
            "GrammarValidator.validate_convergence",
            "GrammarValidator.validate_resonant_coupling",
            "GrammarValidator.validate_bifurcation_triggers",
            "GrammarValidator.validate_transformer_context",
            "GrammarValidator.validate_remesh_amplification",
        ]
        
        for func_name in critical_functions:
            if func_name in self.audit_report["functions"]:
                func_info = self.audit_report["functions"][func_name]
                if not func_info["has_docstring"]:
                    self.audit_report["issues"].append({
                        "type": "missing_docstring",
                        "function": func_name,
                        "severity": "high"
                    })
                else:
                    docstring = func_info["docstring"]
                    # Check for key elements
                    if "Parameters" not in docstring and "Args" not in docstring:
                        self.audit_report["issues"].append({
                            "type": "incomplete_docstring",
                            "function": func_name,
                            "missing": "parameters",
                            "severity": "medium"
                        })
                    if "Returns" not in docstring:
                        self.audit_report["issues"].append({
                            "type": "incomplete_docstring",
                            "function": func_name,
                            "missing": "returns",
                            "severity": "medium"
                        })
        
        print(f"  ✓ Analyzed {len(critical_functions)} critical functions")
    
    def check_cross_references(self):
        """Check for cross-references between docs and code."""
        print("  Scanning for cross-references...")
        
        # Check if documentation references grammar.py
        doc_files = list(self.docs_dir.glob("*.md"))
        
        for doc_file in doc_files:
            content = doc_file.read_text()
            
            # Check if it references grammar.py
            if "grammar.py" in content:
                self.audit_report["cross_references"].append({
                    "doc": doc_file.name,
                    "references": "grammar.py",
                    "type": "doc_to_code"
                })
            
            # Check for specific function references
            for func_name in self.audit_report["functions"]:
                if func_name in content:
                    self.audit_report["cross_references"].append({
                        "doc": doc_file.name,
                        "references": func_name,
                        "type": "doc_to_function"
                    })
        
        print(f"  ✓ Found {len(self.audit_report['cross_references'])} cross-references")
    
    def validate_examples(self):
        """Validate all example files execute correctly."""
        print("  Testing example files...")
        
        example_files = list(self.examples_dir.glob("*.py"))
        
        for example_file in example_files:
            print(f"    Testing {example_file.name}...", end=" ")
            try:
                result = subprocess.run(
                    [sys.executable, str(example_file)],
                    cwd=self.repo_root,
                    capture_output=True,
                    timeout=30,
                    text=True
                )
                
                success = result.returncode == 0
                self.audit_report["examples"][example_file.name] = {
                    "executes": success,
                    "exit_code": result.returncode,
                    "error": result.stderr if not success else None
                }
                
                if success:
                    print("✓")
                else:
                    print("✗")
                    self.audit_report["issues"].append({
                        "type": "example_failure",
                        "file": example_file.name,
                        "error": result.stderr[:200],
                        "severity": "high"
                    })
            
            except subprocess.TimeoutExpired:
                print("⏱ TIMEOUT")
                self.audit_report["examples"][example_file.name] = {
                    "executes": False,
                    "error": "Timeout after 30s"
                }
                self.audit_report["issues"].append({
                    "type": "example_timeout",
                    "file": example_file.name,
                    "severity": "high"
                })
            except Exception as e:
                print(f"✗ {e}")
                self.audit_report["examples"][example_file.name] = {
                    "executes": False,
                    "error": str(e)
                }
    
    def validate_schema(self):
        """Validate schema.json against implementation."""
        print(f"  Checking {self.schema_file}...")
        
        if not self.schema_file.exists():
            self.audit_report["issues"].append({
                "type": "missing_schema",
                "severity": "high"
            })
            print("  ✗ Schema file not found")
            return
        
        schema = json.loads(self.schema_file.read_text())
        
        # Check that all operators in schema match operator sets
        schema_operators = {op["name"] for op in schema.get("operators", [])}
        
        # Get all unique operators from sets
        all_operators = set()
        for op_set in self.audit_report["operator_sets"].values():
            all_operators.update(op_set)
        
        # Check for mismatches
        missing_in_schema = all_operators - schema_operators
        extra_in_schema = schema_operators - all_operators
        
        if missing_in_schema:
            self.audit_report["issues"].append({
                "type": "schema_incomplete",
                "missing_operators": list(missing_in_schema),
                "severity": "medium"
            })
        
        if extra_in_schema:
            self.audit_report["issues"].append({
                "type": "schema_extra",
                "extra_operators": list(extra_in_schema),
                "severity": "low"
            })
        
        print(f"  ✓ Schema contains {len(schema_operators)} operators")
    
    def generate_report(self):
        """Generate and display final report."""
        print("\n📊 SUMMARY")
        print("-" * 70)
        
        # Functions
        total_funcs = len(self.audit_report["functions"])
        funcs_with_docs = sum(
            1 for f in self.audit_report["functions"].values()
            if f["has_docstring"]
        )
        print(f"Functions: {total_funcs} total, {funcs_with_docs} documented "
              f"({funcs_with_docs/total_funcs*100:.1f}%)")
        
        # Operator sets
        print(f"Operator sets: {len(self.audit_report['operator_sets'])}")
        
        # Examples
        total_examples = len(self.audit_report["examples"])
        passing_examples = sum(
            1 for e in self.audit_report["examples"].values()
            if e["executes"]
        )
        print(f"Examples: {passing_examples}/{total_examples} passing "
              f"({passing_examples/total_examples*100:.1f}%)")
        
        # Cross-references
        print(f"Cross-references: {len(self.audit_report['cross_references'])}")
        
        # Issues
        issues_by_severity = {}
        for issue in self.audit_report["issues"]:
            severity = issue.get("severity", "unknown")
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        
        print(f"\n⚠️  ISSUES: {len(self.audit_report['issues'])} total")
        for severity in ["high", "medium", "low"]:
            count = issues_by_severity.get(severity, 0)
            if count > 0:
                print(f"  - {severity.upper()}: {count}")
        
        # Detail issues
        if self.audit_report["issues"]:
            print("\n📋 ISSUE DETAILS:")
            for i, issue in enumerate(self.audit_report["issues"], 1):
                print(f"\n  {i}. [{issue.get('severity', '?').upper()}] "
                      f"{issue['type']}")
                for key, value in issue.items():
                    if key not in ['type', 'severity']:
                        print(f"     {key}: {value}")
        
        # Save detailed report
        report_file = self.repo_root / "docs" / "grammar" / "SYNC_REPORT.json"
        report_file.write_text(json.dumps(self.audit_report, indent=2))
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Return status
        high_issues = issues_by_severity.get("high", 0)
        if high_issues > 0:
            print(f"\n❌ SYNC INCOMPLETE: {high_issues} high-priority issues")
            return False
        else:
            print(f"\n✅ SYNC COMPLETE: All critical checks passed")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize TNFR grammar documentation with implementation"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit grammar.py only"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate examples only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full synchronization"
    )
    
    args = parser.parse_args()
    
    syncer = DocumentationSynchronizer()
    
    if args.audit:
        syncer.audit_grammar_py()
        syncer.generate_report()
    elif args.validate:
        syncer.validate_examples()
        syncer.generate_report()
    else:
        # Default: full sync
        success = syncer.run_full_sync()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

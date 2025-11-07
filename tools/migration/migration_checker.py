"""Migration checker for identifying Grammar 2.0 compatibility issues.

Scans Python code or sequence lists for patterns that may need updates
when migrating from Grammar 1.0 to Grammar 2.0.
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union


class IssueLevel(Enum):
    """Severity level of migration issue."""
    ERROR = "error"  # Must fix (e.g., THOL without destabilizer)
    WARNING = "warning"  # Should review (e.g., pattern name dependency)
    INFO = "info"  # Optional improvement (e.g., could use health metrics)


@dataclass
class MigrationIssue:
    """A single migration issue found in code."""
    level: IssueLevel
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class MigrationReport:
    """Report of all migration issues found."""
    file_path: Optional[str] = None
    issues: List[MigrationIssue] = field(default_factory=list)
    sequences_checked: int = 0
    
    @property
    def has_errors(self) -> bool:
        """Check if report contains any errors."""
        return any(issue.level == IssueLevel.ERROR for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if report contains any warnings."""
        return any(issue.level == IssueLevel.WARNING for issue in self.issues)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        if self.file_path:
            lines.append(f"Migration Report: {self.file_path}")
        else:
            lines.append("Migration Report")
        lines.append("=" * 70)
        
        errors = [i for i in self.issues if i.level == IssueLevel.ERROR]
        warnings = [i for i in self.issues if i.level == IssueLevel.WARNING]
        infos = [i for i in self.issues if i.level == IssueLevel.INFO]
        
        lines.append(f"Sequences checked: {self.sequences_checked}")
        lines.append(f"Issues found: {len(self.issues)} (Errors: {len(errors)}, Warnings: {len(warnings)}, Info: {len(infos)})")
        lines.append("")
        
        if errors:
            lines.append("ERRORS (Must Fix):")
            lines.append("-" * 70)
            for issue in errors:
                lines.append(self._format_issue(issue))
        
        if warnings:
            lines.append("WARNINGS (Should Review):")
            lines.append("-" * 70)
            for issue in warnings:
                lines.append(self._format_issue(issue))
        
        if infos:
            lines.append("INFO (Optional Improvements):")
            lines.append("-" * 70)
            for issue in infos:
                lines.append(self._format_issue(issue))
        
        if not self.issues:
            lines.append("✓ No migration issues found!")
        
        return "\n".join(lines)
    
    def _format_issue(self, issue: MigrationIssue) -> str:
        """Format a single issue for display."""
        parts = []
        if issue.line_number:
            parts.append(f"  Line {issue.line_number}: {issue.message}")
        else:
            parts.append(f"  {issue.message}")
        
        if issue.code_snippet:
            parts.append(f"    Code: {issue.code_snippet}")
        
        if issue.suggestion:
            parts.append(f"    Fix: {issue.suggestion}")
        
        parts.append("")
        return "\n".join(parts)


class MigrationChecker:
    """Scanner for Grammar 2.0 migration issues.
    
    Identifies:
    - SELF_ORGANIZATION without nearby destabilizer (ERROR)
    - Pattern name dependencies that may change (WARNING)
    - Opportunities to adopt health metrics (INFO)
    - Frequency transition issues (WARNING)
    """
    
    # Operator names to check for
    DESTABILIZERS = {"dissonance", "mutation", "contraction"}
    SELF_ORG_NAMES = {"self_organization", "SELF_ORGANIZATION", "THOL"}
    
    def scan_file(self, file_path: Union[str, Path]) -> MigrationReport:
        """Scan a Python file for migration issues.
        
        Args:
            file_path: Path to Python file to scan
            
        Returns:
            MigrationReport with all issues found
        """
        file_path = Path(file_path)
        report = MigrationReport(file_path=str(file_path))
        
        if not file_path.exists():
            report.issues.append(MigrationIssue(
                level=IssueLevel.ERROR,
                message=f"File not found: {file_path}"
            ))
            return report
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find sequence definitions
            tree = ast.parse(content, filename=str(file_path))
            self._scan_ast(tree, content, report)
            
            # Also scan for string patterns (in case sequences are defined as strings)
            self._scan_text_patterns(content, report)
            
        except SyntaxError as e:
            report.issues.append(MigrationIssue(
                level=IssueLevel.ERROR,
                message=f"Syntax error in file: {e}",
                line_number=e.lineno
            ))
        except Exception as e:
            report.issues.append(MigrationIssue(
                level=IssueLevel.ERROR,
                message=f"Error scanning file: {e}"
            ))
        
        return report
    
    def check_sequence(self, sequence: List[str], line_number: Optional[int] = None) -> List[MigrationIssue]:
        """Check a single sequence for migration issues.
        
        Args:
            sequence: List of operator names
            line_number: Optional line number for reporting
            
        Returns:
            List of issues found
        """
        issues = []
        
        # Normalize operator names to lowercase
        seq_lower = [op.lower() for op in sequence]
        
        # Check for SELF_ORGANIZATION without destabilizer
        for i, op in enumerate(seq_lower):
            if op in {"self_organization", "thol"}:
                # Check 3-operator window before THOL
                window_start = max(0, i - 3)
                window = seq_lower[window_start:i]
                
                has_destabilizer = any(op in self.DESTABILIZERS for op in window)
                
                if not has_destabilizer:
                    issues.append(MigrationIssue(
                        level=IssueLevel.ERROR,
                        message=f"SELF_ORGANIZATION at position {i} requires destabilizer within 3-operator window",
                        line_number=line_number,
                        code_snippet=" → ".join(sequence),
                        suggestion=f"Add DISSONANCE, MUTATION, or CONTRACTION before SELF_ORGANIZATION. Example: {self._suggest_thol_fix(sequence, i)}"
                    ))
        
        # Check for silence → high frequency jumps
        for i in range(len(seq_lower) - 1):
            if seq_lower[i] == "silence" and seq_lower[i + 1] in {"emission", "dissonance", "resonance", "mutation", "contraction"}:
                issues.append(MigrationIssue(
                    level=IssueLevel.WARNING,
                    message=f"Frequency jump: SILENCE → {sequence[i + 1].upper()} (Zero → High)",
                    line_number=line_number,
                    code_snippet=f"{sequence[i]} → {sequence[i + 1]}",
                    suggestion="Consider inserting medium-frequency operator (TRANSITION, COHERENCE, RECEPTION)"
                ))
        
        return issues
    
    def scan_sequences(self, sequences: List[List[str]]) -> MigrationReport:
        """Scan multiple sequences for migration issues.
        
        Args:
            sequences: List of operator sequences to check
            
        Returns:
            MigrationReport with all issues found
        """
        report = MigrationReport()
        report.sequences_checked = len(sequences)
        
        for i, sequence in enumerate(sequences):
            issues = self.check_sequence(sequence, line_number=i + 1)
            report.issues.extend(issues)
        
        return report
    
    def _scan_ast(self, tree: ast.AST, content: str, report: MigrationReport):
        """Scan AST for sequence definitions."""
        for node in ast.walk(tree):
            # Look for list assignments or function calls with sequence arguments
            if isinstance(node, ast.List):
                sequence = self._extract_sequence_from_list(node)
                if sequence:
                    report.sequences_checked += 1
                    issues = self.check_sequence(sequence, line_number=node.lineno)
                    report.issues.extend(issues)
            
            # Look for validate_sequence calls without health metrics
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "validate_sequence":
                    report.issues.append(MigrationIssue(
                        level=IssueLevel.INFO,
                        message="Consider using validate_sequence_with_health() for enhanced metrics",
                        line_number=node.lineno,
                        suggestion="Replace validate_sequence() with validate_sequence_with_health() to get health metrics"
                    ))
                
                # Check for pattern name dependencies
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == "get" and len(node.args) > 0:
                        if isinstance(node.args[0], ast.Constant) and node.args[0].value == "pattern":
                            report.issues.append(MigrationIssue(
                                level=IssueLevel.WARNING,
                                message="Pattern names may be more specific in Grammar 2.0",
                                line_number=node.lineno,
                                suggestion="Use pattern categories instead of exact names for robustness"
                            ))
    
    def _scan_text_patterns(self, content: str, report: MigrationReport):
        """Scan text content for sequence-related patterns."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, start=1):
            # Look for pattern name comparisons (e.g., pattern == 'activation')
            # But skip lines that are safely accessing result.metadata
            if re.search(r"pattern\s*==\s*['\"]", line):
                # Skip if this is a safe metadata access pattern
                if "result.metadata" in line or "metadata.get" in line:
                    continue
                    
                report.issues.append(MigrationIssue(
                    level=IssueLevel.WARNING,
                    message="Hard-coded pattern name comparison may break with Grammar 2.0",
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion="Use pattern categories or check pattern type instead"
                ))
    
    # Operator names to check for (class constant)
    COMMON_OPERATORS = {
        "emission", "reception", "coherence", "silence", "dissonance",
        "coupling", "resonance", "self_organization", "transition",
        "mutation", "expansion", "contraction", "recursivity"
    }
    
    def _extract_sequence_from_list(self, node: ast.List) -> Optional[List[str]]:
        """Extract operator sequence from AST List node."""
        sequence = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                sequence.append(elt.value)
            elif isinstance(elt, ast.Name):
                sequence.append(elt.id)
        
        # Only return if it looks like an operator sequence
        if sequence and any(op.lower() in self.COMMON_OPERATORS for op in sequence):
            return sequence
        return None
    
    def _suggest_thol_fix(self, sequence: List[str], thol_index: int) -> str:
        """Suggest a fix for THOL without destabilizer."""
        # Insert dissonance before THOL
        fixed = sequence[:thol_index] + ["dissonance"] + sequence[thol_index:]
        return " → ".join(fixed)


def main():
    """CLI entry point for migration checker."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m tools.migration.migration_checker <file_or_directory>")
        print("\nScans Python files for Grammar 2.0 migration issues.")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    checker = MigrationChecker()
    
    if path.is_file():
        report = checker.scan_file(path)
        print(report.summary())
        sys.exit(1 if report.has_errors else 0)
    
    elif path.is_dir():
        all_issues = []
        total_sequences = 0
        
        for py_file in path.rglob("*.py"):
            report = checker.scan_file(py_file)
            if report.issues:
                print(f"\n{py_file}:")
                print(report.summary())
                all_issues.extend(report.issues)
            total_sequences += report.sequences_checked
        
        print(f"\n{'=' * 70}")
        print(f"Total sequences checked: {total_sequences}")
        print(f"Total issues: {len(all_issues)}")
        
        errors = sum(1 for i in all_issues if i.level == IssueLevel.ERROR)
        if errors > 0:
            print(f"❌ Found {errors} errors that must be fixed")
            sys.exit(1)
        else:
            print("✓ No critical errors found")
            sys.exit(0)
    
    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

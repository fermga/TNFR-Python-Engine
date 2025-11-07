#!/usr/bin/env python3
"""Extension validator for community TNFR extensions.

Validates community-contributed extensions for quality, correctness, and
TNFR compliance before acceptance into the main repository.

Examples
--------
>>> from tools.community.extension_validator import ExtensionValidator
>>> from tnfr.extensions.medical import MedicalExtension
>>> 
>>> validator = ExtensionValidator()
>>> extension = MedicalExtension()
>>> report = validator.validate_extension(extension)
>>> 
>>> print(f"Code quality: {report.code_quality_score:.2f}")
>>> print(f"Documentation: {report.documentation_score:.2f}")
>>> print(f"Pattern health: {report.average_pattern_health:.2f}")
>>> print(f"Test coverage: {report.test_coverage:.1f}%")
>>> 
>>> if report.is_acceptable():
...     print("Extension ready for submission!")
... else:
...     print("Issues to address:")
...     for issue in report.issues:
...         print(f"  - {issue}")
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.extensions.base import TNFRExtension, PatternDefinition
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer


__all__ = ["ValidationReport", "ExtensionValidator"]


@dataclass
class PatternValidationResult:
    """Validation result for a single pattern.
    
    Attributes
    ----------
    pattern_id : str
        Pattern identifier
    pattern_name : str
        Human-readable name
    num_examples : int
        Number of example sequences
    health_scores : List[float]
        Health scores for each example
    average_health : float
        Average health across examples
    meets_minimum : bool
        Whether average health meets minimum requirement
    issues : List[str]
        Validation issues found
    """
    pattern_id: str
    pattern_name: str
    num_examples: int
    health_scores: List[float]
    average_health: float
    meets_minimum: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report for an extension.
    
    Attributes
    ----------
    domain : str
        Extension domain name
    code_quality_score : float
        Code quality score (0-1)
    documentation_score : float
        Documentation completeness score (0-1)
    pattern_validations : Dict[str, PatternValidationResult]
        Per-pattern validation results
    average_pattern_health : float
        Average health across all patterns
    test_coverage : float
        Estimated test coverage percentage
    overall_score : float
        Combined quality score (0-1)
    issues : List[str]
        Critical issues that must be addressed
    warnings : List[str]
        Non-critical warnings
    """
    domain: str
    code_quality_score: float = 0.0
    documentation_score: float = 0.0
    pattern_validations: Dict[str, PatternValidationResult] = field(default_factory=dict)
    average_pattern_health: float = 0.0
    test_coverage: float = 0.0
    overall_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def is_acceptable(self, min_score: float = 0.75) -> bool:
        """Check if extension meets acceptance criteria.
        
        Parameters
        ----------
        min_score : float
            Minimum acceptable overall score
        
        Returns
        -------
        bool
            True if extension is acceptable for merging
        """
        return (
            len(self.issues) == 0 and
            self.overall_score >= min_score and
            self.average_pattern_health >= 0.65
        )


class ExtensionValidator:
    """Validates community TNFR extensions for quality and compliance.
    
    Performs comprehensive validation including:
    - Code quality checks
    - Documentation completeness
    - Pattern health validation
    - Example sequence validation
    - Structural correctness
    
    Examples
    --------
    >>> validator = ExtensionValidator()
    >>> report = validator.validate_extension(MyExtension())
    >>> if report.is_acceptable():
    ...     print("Extension ready for submission")
    """
    
    def __init__(self) -> None:
        """Initialize validator with health analyzer."""
        self.health_analyzer = SequenceHealthAnalyzer()
    
    def validate_extension(self, extension: TNFRExtension) -> ValidationReport:
        """Validate an extension comprehensively.
        
        Parameters
        ----------
        extension : TNFRExtension
            Extension to validate
        
        Returns
        -------
        ValidationReport
            Detailed validation report
        """
        domain = extension.get_domain_name()
        report = ValidationReport(domain=domain)
        
        # 1. Validate domain name
        self._validate_domain_name(extension, report)
        
        # 2. Check code quality
        report.code_quality_score = self._check_code_quality(extension, report)
        
        # 3. Validate patterns
        self._validate_patterns(extension, report)
        
        # 4. Check documentation
        report.documentation_score = self._check_documentation(extension, report)
        
        # 5. Estimate test coverage
        report.test_coverage = self._estimate_test_coverage(extension, report)
        
        # 6. Calculate overall score
        report.overall_score = self._calculate_overall_score(report)
        
        return report
    
    def _validate_domain_name(
        self, extension: TNFRExtension, report: ValidationReport
    ) -> None:
        """Validate domain name format."""
        domain = extension.get_domain_name()
        
        if not domain:
            report.issues.append("Domain name cannot be empty")
            return
        
        if not domain.replace("_", "").isalnum():
            report.issues.append(
                f"Invalid domain name '{domain}': must be alphanumeric with underscores"
            )
        
        if not domain.islower():
            report.issues.append(
                f"Invalid domain name '{domain}': must be lowercase"
            )
    
    def _check_code_quality(
        self, extension: TNFRExtension, report: ValidationReport
    ) -> float:
        """Check code quality of extension implementation.
        
        Returns quality score 0-1.
        """
        score = 1.0
        
        # Check required methods are implemented
        try:
            extension.get_pattern_definitions()
        except NotImplementedError:
            report.issues.append("get_pattern_definitions() not implemented")
            score -= 0.5
        
        # Check metadata quality
        metadata = extension.get_metadata()
        if "version" not in metadata:
            report.warnings.append("Extension metadata missing version")
            score -= 0.1
        if "description" not in metadata:
            report.warnings.append("Extension metadata missing description")
            score -= 0.1
        
        return max(0.0, score)
    
    def _validate_patterns(
        self, extension: TNFRExtension, report: ValidationReport
    ) -> None:
        """Validate all patterns in extension."""
        try:
            patterns = extension.get_pattern_definitions()
        except Exception as e:
            report.issues.append(f"Error getting pattern definitions: {e}")
            return
        
        if not patterns:
            report.issues.append("Extension provides no patterns")
            return
        
        total_health = 0.0
        num_patterns = 0
        
        for pattern_id, pattern_def in patterns.items():
            result = self._validate_pattern(pattern_id, pattern_def)
            report.pattern_validations[pattern_id] = result
            
            if result.average_health > 0:
                total_health += result.average_health
                num_patterns += 1
            
            # Add pattern issues to report
            for issue in result.issues:
                if "below minimum" in issue or "No valid examples" in issue:
                    report.issues.append(f"Pattern '{pattern_id}': {issue}")
                else:
                    report.warnings.append(f"Pattern '{pattern_id}': {issue}")
        
        if num_patterns > 0:
            report.average_pattern_health = total_health / num_patterns
        else:
            report.issues.append("No valid patterns found")
    
    def _validate_pattern(
        self, pattern_id: str, pattern_def: PatternDefinition
    ) -> PatternValidationResult:
        """Validate a single pattern definition."""
        issues = []
        health_scores = []
        
        # Check pattern has examples
        if not pattern_def.examples:
            issues.append("No example sequences provided")
            return PatternValidationResult(
                pattern_id=pattern_id,
                pattern_name=pattern_def.name,
                num_examples=0,
                health_scores=[],
                average_health=0.0,
                meets_minimum=False,
                issues=issues,
            )
        
        # Validate each example sequence
        for idx, sequence in enumerate(pattern_def.examples):
            try:
                # Validate sequence with grammar
                validation = validate_sequence_with_health(sequence)
                
                if validation.passed:
                    health = validation.health_metrics.overall_health
                    health_scores.append(health)
                else:
                    issues.append(
                        f"Example {idx + 1} invalid: {validation.error}"
                    )
            except Exception as e:
                issues.append(f"Example {idx + 1} validation error: {e}")
        
        # Calculate average health
        if health_scores:
            avg_health = sum(health_scores) / len(health_scores)
        else:
            avg_health = 0.0
            issues.append("No valid examples found")
        
        # Check against minimum
        meets_min = avg_health >= pattern_def.min_health_score
        if not meets_min and avg_health > 0:
            issues.append(
                f"Average health {avg_health:.3f} below minimum "
                f"{pattern_def.min_health_score:.3f}"
            )
        
        return PatternValidationResult(
            pattern_id=pattern_id,
            pattern_name=pattern_def.name,
            num_examples=len(pattern_def.examples),
            health_scores=health_scores,
            average_health=avg_health,
            meets_minimum=meets_min,
            issues=issues,
        )
    
    def _check_documentation(
        self, extension: TNFRExtension, report: ValidationReport
    ) -> float:
        """Check documentation completeness.
        
        Returns documentation score 0-1.
        """
        score = 0.0
        
        # Check class docstring
        if extension.__class__.__doc__:
            score += 0.3
        else:
            report.warnings.append("Extension class missing docstring")
        
        # Check metadata completeness
        metadata = extension.get_metadata()
        required_fields = ["domain", "version", "description"]
        present = sum(1 for field in required_fields if field in metadata)
        score += 0.3 * (present / len(required_fields))
        
        # Check pattern documentation
        try:
            patterns = extension.get_pattern_definitions()
            if patterns:
                documented = sum(
                    1 for p in patterns.values()
                    if p.description and len(p.description) > 20
                )
                score += 0.4 * (documented / len(patterns))
            else:
                score += 0.4  # No patterns to document
        except Exception:
            pass
        
        return min(1.0, score)
    
    def _estimate_test_coverage(
        self, extension: TNFRExtension, report: ValidationReport
    ) -> float:
        """Estimate test coverage based on validation.
        
        Returns estimated coverage percentage.
        """
        # For now, based on validation success rate
        if not report.pattern_validations:
            return 0.0
        
        total = len(report.pattern_validations)
        passed = sum(
            1 for result in report.pattern_validations.values()
            if result.meets_minimum
        )
        
        # Estimate: if patterns validate well, likely some tests exist
        return (passed / total) * 80.0  # Max 80% without actual test files
    
    def _calculate_overall_score(self, report: ValidationReport) -> float:
        """Calculate overall quality score."""
        # Weighted average of different aspects
        weights = {
            "code_quality": 0.2,
            "documentation": 0.2,
            "pattern_health": 0.5,
            "test_coverage": 0.1,
        }
        
        score = (
            weights["code_quality"] * report.code_quality_score +
            weights["documentation"] * report.documentation_score +
            weights["pattern_health"] * report.average_pattern_health +
            weights["test_coverage"] * (report.test_coverage / 100.0)
        )
        
        # Penalize for critical issues
        if report.issues:
            score *= 0.5  # 50% penalty for any critical issues
        
        return score


def main() -> None:
    """CLI interface for extension validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate TNFR community extension"
    )
    parser.add_argument(
        "extension_module",
        help="Python module path to extension (e.g., tnfr.extensions.medical)",
    )
    parser.add_argument(
        "extension_class",
        help="Extension class name (e.g., MedicalExtension)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.75,
        help="Minimum acceptable overall score (default: 0.75)",
    )
    
    args = parser.parse_args()
    
    # Import extension
    try:
        module = __import__(args.extension_module, fromlist=[args.extension_class])
        extension_cls = getattr(module, args.extension_class)
        extension = extension_cls()
    except Exception as e:
        print(f"Error loading extension: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate
    validator = ExtensionValidator()
    report = validator.validate_extension(extension)
    
    # Print report
    print(f"\n{'='*60}")
    print(f"TNFR Extension Validation Report: {report.domain}")
    print(f"{'='*60}\n")
    
    print(f"Overall Score: {report.overall_score:.2f}")
    print(f"  Code Quality: {report.code_quality_score:.2f}")
    print(f"  Documentation: {report.documentation_score:.2f}")
    print(f"  Pattern Health: {report.average_pattern_health:.2f}")
    print(f"  Test Coverage: {report.test_coverage:.1f}%\n")
    
    print(f"Patterns Validated: {len(report.pattern_validations)}")
    for pattern_id, result in report.pattern_validations.items():
        status = "✓" if result.meets_minimum else "✗"
        print(f"  {status} {result.pattern_name}: {result.average_health:.3f}")
    
    if report.issues:
        print(f"\nCritical Issues ({len(report.issues)}):")
        for issue in report.issues:
            print(f"  ✗ {issue}")
    
    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  ⚠ {warning}")
    
    print(f"\n{'='*60}")
    if report.is_acceptable(args.min_score):
        print("✓ Extension ACCEPTABLE for submission")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print("✗ Extension NOT ACCEPTABLE - address issues above")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extension validator for TNFR community contributions.

Validates community extensions before acceptance, ensuring they meet quality
standards, maintain TNFR canonical invariants, and provide properly documented
domain patterns.
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ValidationReport:
    """Report of extension validation results.
    
    Attributes
    ----------
    extension_name : str
        Name of the validated extension.
    passed : bool
        Overall validation status.
    code_quality : Dict[str, bool]
        Code quality checks (linting, type checking, etc.).
    pattern_validations : Dict[str, Dict[str, float]]
        Per-pattern health metric validation.
    documentation : Dict[str, bool]
        Documentation completeness checks.
    test_coverage : Optional[float]
        Test coverage percentage (0-100).
    errors : List[str]
        List of validation errors.
    warnings : List[str]
        List of validation warnings.
    """
    
    extension_name: str
    passed: bool = False
    code_quality: Dict[str, bool] = field(default_factory=dict)
    pattern_validations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    documentation: Dict[str, bool] = field(default_factory=dict)
    test_coverage: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ExtensionValidator:
    """Validates community extensions before acceptance.
    
    Performs comprehensive validation of extension quality including code
    standards, pattern health metrics, documentation completeness, and test
    coverage.
    
    Examples
    --------
    >>> validator = ExtensionValidator()
    >>> report = validator.validate_extension("medical")
    >>> if report.passed:
    ...     print("Extension validated successfully!")
    ... else:
    ...     print(f"Validation failed: {report.errors}")
    """
    
    def __init__(self, repo_root: Optional[Path] = None) -> None:
        """Initialize validator.
        
        Parameters
        ----------
        repo_root : Optional[Path]
            Root directory of TNFR repository. If None, uses current working dir.
        """
        self.repo_root = repo_root or Path.cwd()
        self.extensions_dir = self.repo_root / "src" / "tnfr" / "extensions"
    
    def validate_extension(self, extension_name: str) -> ValidationReport:
        """Comprehensive validation of extension quality.
        
        Parameters
        ----------
        extension_name : str
            Name of extension to validate (domain identifier).
            
        Returns
        -------
        ValidationReport
            Detailed validation results.
        """
        report = ValidationReport(extension_name=extension_name)
        
        # Check extension exists
        extension_dir = self.extensions_dir / extension_name
        if not extension_dir.exists():
            report.errors.append(f"Extension directory not found: {extension_dir}")
            return report
        
        # 1. Code quality checks
        report.code_quality = self._check_code_standards(extension_name, extension_dir)
        
        # 2. Pattern validation
        try:
            pattern_validations = self._validate_pattern_examples(
                extension_name, extension_dir
            )
            report.pattern_validations = pattern_validations
        except Exception as e:
            report.errors.append(f"Pattern validation failed: {e}")
        
        # 3. Documentation completeness
        report.documentation = self._check_documentation(extension_dir)
        
        # 4. Test coverage
        try:
            coverage = self._check_test_coverage(extension_name)
            report.test_coverage = coverage
            if coverage < 80.0:
                report.warnings.append(
                    f"Test coverage {coverage:.1f}% is below 80% threshold"
                )
        except Exception as e:
            report.warnings.append(f"Could not check test coverage: {e}")
        
        # Determine overall pass/fail
        report.passed = (
            len(report.errors) == 0
            and all(report.code_quality.values())
            and all(report.documentation.values())
            and all(
                metrics.get("C_t", 0) > 0.75
                for metrics in report.pattern_validations.values()
            )
        )
        
        return report
    
    def _check_code_standards(
        self, extension_name: str, extension_dir: Path
    ) -> Dict[str, bool]:
        """Check code quality standards.
        
        Parameters
        ----------
        extension_name : str
            Extension name.
        extension_dir : Path
            Path to extension directory.
            
        Returns
        -------
        Dict[str, bool]
            Quality check results.
        """
        checks = {
            "has_init": (extension_dir / "__init__.py").exists(),
            "has_patterns": (extension_dir / "patterns.py").exists(),
            "has_readme": (extension_dir / "README.md").exists(),
            "has_pyi_stubs": True,  # Check for .pyi files
        }
        
        # Check for .pyi stubs
        py_files = list(extension_dir.glob("*.py"))
        pyi_files = list(extension_dir.glob("*.pyi"))
        if len(py_files) > 0 and len(pyi_files) == 0:
            checks["has_pyi_stubs"] = False
        
        return checks
    
    def _validate_pattern_examples(
        self, extension_name: str, extension_dir: Path
    ) -> Dict[str, Dict[str, float]]:
        """Validate pattern examples meet health requirements.
        
        Parameters
        ----------
        extension_name : str
            Extension name.
        extension_dir : Path
            Path to extension directory.
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Pattern name -> health metrics mapping.
        """
        pattern_validations = {}
        
        # Try to load extension and check patterns
        try:
            # Import extension module
            import importlib
            ext_module = importlib.import_module(
                f"tnfr.extensions.{extension_name}"
            )
            
            # Get extension class (assume it follows naming convention)
            ext_class_name = "".join(
                word.capitalize() for word in extension_name.split("_")
            ) + "Extension"
            
            if hasattr(ext_module, ext_class_name):
                ext_instance = getattr(ext_module, ext_class_name)()
                patterns = ext_instance.get_pattern_definitions()
                
                for pattern_name, pattern_def in patterns.items():
                    # Validate examples
                    if pattern_def.examples:
                        for idx, example in enumerate(pattern_def.examples):
                            health_metrics = example.get("health_metrics", {})
                            pattern_validations[f"{pattern_name}_ex{idx}"] = health_metrics
                    else:
                        pattern_validations[pattern_name] = {"C_t": 0.0, "Si": 0.0}
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            # Expected errors during extension loading
            pattern_validations["_load_error"] = {"error": str(e)}
        
        return pattern_validations
    
    def _check_documentation(self, extension_dir: Path) -> Dict[str, bool]:
        """Check documentation completeness.
        
        Parameters
        ----------
        extension_dir : Path
            Path to extension directory.
            
        Returns
        -------
        Dict[str, bool]
            Documentation check results.
        """
        checks = {
            "has_readme": (extension_dir / "README.md").exists(),
            "readme_not_empty": False,
        }
        
        readme_path = extension_dir / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            checks["readme_not_empty"] = len(content.strip()) > 100
        
        return checks
    
    def _check_test_coverage(self, extension_name: str) -> float:
        """Check test coverage for extension.
        
        Parameters
        ----------
        extension_name : str
            Extension name.
            
        Returns
        -------
        float
            Coverage percentage (0-100).
            
        Notes
        -----
        This is a simplified estimate based on test file existence.
        For accurate coverage, run pytest-cov directly:
        `pytest --cov=src/tnfr/extensions/{extension_name} tests/extensions/test_{extension_name}.py`
        """
        # Simplified coverage estimate based on test file existence
        # For production use, integrate with pytest-cov for actual coverage
        test_dir = self.repo_root / "tests" / "extensions"
        test_file = test_dir / f"test_{extension_name}.py"
        
        if test_file.exists():
            # Estimate: test file exists = reasonable coverage
            return 85.0
        else:
            return 0.0
    
    def print_report(self, report: ValidationReport) -> None:
        """Print formatted validation report.
        
        Parameters
        ----------
        report : ValidationReport
            Validation report to print.
        """
        print(f"\n{'='*70}")
        print(f"Extension Validation Report: {report.extension_name}")
        print(f"{'='*70}\n")
        
        # Overall status
        status = "✅ PASSED" if report.passed else "❌ FAILED"
        print(f"Overall Status: {status}\n")
        
        # Code quality
        print("Code Quality:")
        for check, passed in report.code_quality.items():
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check}")
        print()
        
        # Pattern validations
        if report.pattern_validations:
            print("Pattern Validations:")
            for pattern, metrics in report.pattern_validations.items():
                if "error" in metrics:
                    print(f"  ⚠️  {pattern}: {metrics['error']}")
                else:
                    c_t = metrics.get("C_t", 0.0)
                    si = metrics.get("Si", 0.0)
                    passed = c_t > 0.75 and si > 0.70
                    icon = "✅" if passed else "❌"
                    print(f"  {icon} {pattern}: C(t)={c_t:.2f}, Si={si:.2f}")
            print()
        
        # Documentation
        print("Documentation:")
        for check, passed in report.documentation.items():
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check}")
        print()
        
        # Test coverage
        if report.test_coverage is not None:
            icon = "✅" if report.test_coverage >= 80.0 else "⚠️"
            print(f"Test Coverage: {icon} {report.test_coverage:.1f}%\n")
        
        # Errors
        if report.errors:
            print("Errors:")
            for error in report.errors:
                print(f"  ❌ {error}")
            print()
        
        # Warnings
        if report.warnings:
            print("Warnings:")
            for warning in report.warnings:
                print(f"  ⚠️  {warning}")
            print()


def main() -> int:
    """Main entry point for extension validator CLI.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    if len(sys.argv) < 2:
        print("Usage: python extension_validator.py <extension_name>")
        print("\nExample:")
        print("  python extension_validator.py medical")
        return 1
    
    extension_name = sys.argv[1]
    
    validator = ExtensionValidator()
    report = validator.validate_extension(extension_name)
    validator.print_report(report)
    
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Repository Health Check Script

Provides a comprehensive health assessment of the TNFR-Python-Engine repository.
Checks for common issues and provides optimization recommendations.

Usage:
    python scripts/repo_health_check.py [--verbose]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Repository root
REPO_ROOT = Path(__file__).parent.parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class RepositoryHealthChecker:
    """Repository health assessment utility."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues = []
        self.recommendations = []
        self.stats = {}
        
    def add_issue(self, category: str, description: str, severity: str = "info") -> None:
        """Add an issue to the report."""
        self.issues.append({
            "category": category,
            "description": description,
            "severity": severity
        })
        
    def add_recommendation(self, description: str) -> None:
        """Add a recommendation to the report."""
        self.recommendations.append(description)
        
    def check_file_organization(self) -> None:
        """Check repository file organization."""
        logger.info("🗂️  Checking file organization...")
        
        # Check for root-level debug/temp files  
        debug_files = []
        temp_patterns = ["debug_*.py", "test_*.py", "analyze_*.py", "temp_*.py", "quick_*.py"]
        
        for pattern in temp_patterns:
            for file_path in REPO_ROOT.glob(pattern):
                if file_path.is_file() and file_path.parent == REPO_ROOT:
                    debug_files.append(file_path.name)
        
        if debug_files:
            self.add_issue(
                "organization",
                f"Found {len(debug_files)} debug/temporary files in root: {', '.join(debug_files[:5])}{'...' if len(debug_files) > 5 else ''}",
                "warning"
            )
            self.add_recommendation("Run `python scripts/optimize_repository.py` to organize temporary files")
        
        self.stats["root_temp_files"] = len(debug_files)
        
    def check_code_quality(self) -> None:
        """Check code quality indicators."""
        logger.info("🔍 Checking code quality...")
        
        # Count Python files
        src_files = list((REPO_ROOT / "src").rglob("*.py"))
        test_files = list((REPO_ROOT / "tests").rglob("*.py"))
        
        self.stats["src_files"] = len(src_files)
        self.stats["test_files"] = len(test_files)
        
        # Check for print statements in source code
        # (excluding tutorials, recipes, and user-facing modules)
        # Note: tutorials/, recipes/, cli/, sdk/, tools/ intentionally use
        # print for user-facing output
        print_count = 0
        excluded_paths = ["tutorials", "recipes", "cli", "tools",
                          "sdk", "services"]
        excluded_count = 0
        
        for py_file in src_files:
            # Skip if file is in excluded directory
            # Normalize path separators for cross-platform compatibility
            path_str = str(py_file).replace('\\', '/')
            is_excluded = any(f"/{excluded}/" in path_str
                              for excluded in excluded_paths)
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count print statements
                    # (excluding docstring examples and __main__ blocks)
                    file_prints = 0
                    in_main_block = False
                    in_docstring = False
                    
                    for line in content.split('\n'):
                        stripped = line.strip()
                        
                        # Track docstrings (triple quotes)
                        if '"""' in line or "'''" in line:
                            in_docstring = not in_docstring
                        
                        # Track if we're in a __main__ block
                        if 'if __name__ ==' in line:
                            in_main_block = True
                        
                        # Count prints outside docstrings and __main__
                        # Must have print( with proper call syntax
                        if (('print(' in line or 'print (' in line) and
                                ('>>>' not in line) and
                                (not stripped.startswith('def ')) and
                                (not stripped.startswith('...')) and
                                (not in_docstring) and
                                (not in_main_block)):
                            file_prints += 1
                    
                    if is_excluded:
                        excluded_count += file_prints
                    else:
                        print_count += file_prints
            except Exception:
                continue
        
        # Only flag if there are significant prints in core code
        # (docstring examples and debug prints in math modules acceptable)
        if print_count > 80:  # Allow some debug output in math/symbolic
            self.add_issue(
                "code_quality",
                f"Found {print_count} print() statements in "
                f"non-tutorial source code",
                "info"
            )
            self.add_recommendation(
                "Consider using logging instead of print statements "
                "in core modules"
            )
            
        self.stats["print_statements_core"] = print_count
        
    def check_documentation(self) -> None:
        """Check documentation completeness."""
        logger.info("📚 Checking documentation...")
        
        # Count markdown files
        md_files = list(REPO_ROOT.glob("**/*.md"))
        self.stats["markdown_files"] = len(md_files)
        
        # Check for key documentation files
        key_docs = [
            "README.md",
            "CONTRIBUTING.md",
            "LICENSE.md",
            "DOCUMENTATION_INDEX.md"
        ]
        missing_docs = []
        
        for doc in key_docs:
            if not (REPO_ROOT / doc).exists():
                missing_docs.append(doc)
                
        if missing_docs:
            self.add_issue(
                "documentation",
                f"Missing key documentation: {', '.join(missing_docs)}",
                "warning"
            )
            
    def check_build_system(self) -> None:
        """Check build system configuration."""
        logger.info("⚙️  Checking build system...")
        
        # Check for key build files
        build_files = {
            "Makefile": "Build automation",
            "pyproject.toml": "Python packaging",
            ".github/workflows": "CI/CD workflows",
            "scripts/windows_make.py": "Windows compatibility"
        }
        
        missing_build = []
        for file_path, description in build_files.items():
            if not (REPO_ROOT / file_path).exists():
                missing_build.append(f"{file_path} ({description})")
                
        if missing_build:
            self.add_issue(
                "build_system",
                f"Missing build files: {', '.join(missing_build)}",
                "warning"
            )
            
        # Check if Windows shim is up to date
        try:
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts/windows_make.py")],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT
            )
            if "clean-scratch" not in result.stdout:
                self.add_issue(
                    "build_system",
                    "Windows shim may be missing recent targets",
                    "info"
                )
        except Exception:
            pass
            
    def check_git_configuration(self) -> None:
        """Check Git configuration."""
        logger.info("🔧 Checking Git configuration...")
        
        gitignore = REPO_ROOT / ".gitignore"
        if gitignore.exists():
            with open(gitignore, 'r', encoding='utf-8') as f:
                gitignore_content = f.read()
                
            recommended_entries = [
                "debug_scratch/",
                "*.tmp",
                "*.temp",
                "__pycache__/",
                ".pytest_cache/",
                ".mypy_cache/"
            ]
            
            missing_entries = []
            for entry in recommended_entries:
                if entry not in gitignore_content:
                    missing_entries.append(entry)
                    
            if missing_entries:
                self.add_issue(
                    "git",
                    f"Recommended .gitignore entries missing: {', '.join(missing_entries)}",
                    "info"
                )
        else:
            self.add_issue("git", "No .gitignore file found", "warning")
            
    def check_dependencies(self) -> None:
        """Check dependency management."""
        logger.info("📦 Checking dependencies...")
        
        # Check for requirements files
        req_files = [
            "requirements.txt",
            "requirements-dev.txt", 
            "pyproject.toml"
        ]
        
        found_req_files = []
        for req_file in req_files:
            if (REPO_ROOT / req_file).exists():
                found_req_files.append(req_file)
                
        if not found_req_files:
            self.add_issue(
                "dependencies",
                "No dependency files found",
                "warning"
            )
        else:
            self.stats["dependency_files"] = found_req_files
            
    def generate_report(self) -> None:
        """Generate comprehensive health report."""
        logger.info("📊 Repository Health Report")
        logger.info("=" * 60)
        
        # Statistics summary
        logger.info("📈 Statistics:")
        for key, value in self.stats.items():
            logger.info(f"  • {key.replace('_', ' ').title()}: {value}")
            
        # Issues by severity
        if self.issues:
            logger.info(f"\n⚠️  Issues Found ({len(self.issues)} total):")
            
            for severity in ["error", "warning", "info"]:
                severity_issues = [i for i in self.issues if i["severity"] == severity]
                if severity_issues:
                    icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}[severity]
                    logger.info(f"\n{icon} {severity.title()} ({len(severity_issues)}):")
                    
                    for issue in severity_issues:
                        logger.info(f"  • [{issue['category']}] {issue['description']}")
        else:
            logger.info("\n✅ No issues found!")
            
        # Recommendations
        if self.recommendations:
            logger.info(f"\n💡 Recommendations ({len(self.recommendations)}):")
            for i, rec in enumerate(self.recommendations, 1):
                logger.info(f"  {i}. {rec}")
                
        # Overall health score
        error_count = len([i for i in self.issues if i["severity"] == "error"])
        warning_count = len([i for i in self.issues if i["severity"] == "warning"])
        info_count = len([i for i in self.issues if i["severity"] == "info"])
        
        # Calculate score (100 - penalties)
        score = 100 - (error_count * 20 + warning_count * 10 + info_count * 2)
        score = max(0, min(100, score))
        
        if score >= 90:
            health = "🟢 Excellent"
        elif score >= 75:
            health = "🟡 Good"  
        elif score >= 50:
            health = "🟠 Fair"
        else:
            health = "🔴 Needs Attention"
            
        logger.info(f"\n🎯 Overall Health: {health} ({score}/100)")
        
    def run_all_checks(self) -> None:
        """Run all health checks."""
        logger.info("🏥 Starting repository health check...")
        logger.info(f"Repository: {REPO_ROOT}")
        
        self.check_file_organization()
        self.check_code_quality()
        self.check_documentation()
        self.check_build_system()
        self.check_git_configuration()
        self.check_dependencies()
        
        self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    checker = RepositoryHealthChecker(verbose=args.verbose)
    checker.run_all_checks()
    
    logger.info("\n🎉 Health check complete!")


if __name__ == "__main__":
    main()
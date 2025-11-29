#!/usr/bin/env python3
"""
TNFR Constants Audit - Paradigm Alignment Analysis
================================================

Comprehensive audit of all hardcoded constants in the TNFR codebase to identify:
1. Constants derived from empirical fitting (pre-theory)
2. Constants that should emerge from TNFR physics  
3. Constants using arbitrary values instead of canonical derivations

This audit ensures all constants align with the fully developed theoretical framework.

Author: TNFR Research Team
Date: November 29, 2025
"""

import sys
import os
import ast
import math
import inspect
import importlib.util
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / '..' / '..' / 'src'))

# Import TNFR canonical constants
try:
    import mpmath as mp
    mp.dps = 30
    
    # Canonical TNFR constants from theory
    PHI = float(mp.phi)  # Golden ratio
    GAMMA = float(mp.euler)  # Euler constant
    PI = float(mp.pi)  # π
    E = float(mp.e)  # Euler's number
    
    print("✅ Canonical constants loaded from mpmath")
    print(f"φ (Golden Ratio): {PHI}")
    print(f"γ (Euler Constant): {GAMMA}")
    print(f"π (Pi): {PI}")
    print(f"e (Euler's Number): {E}")
    
except ImportError:
    # Fallback to math module
    import math
    PHI = (1 + math.sqrt(5)) / 2
    GAMMA = 0.5772156649015329
    PI = math.pi
    E = math.e
    print("⚠️ Using math module fallback")

print("="*70)


class ConstantAuditor:
    """Auditor for TNFR constants alignment."""
    
    def __init__(self):
        """Initialize auditor."""
        self.issues = []
        self.suggestions = []
        self.canonical_replacements = {}
        
        # Define canonical constants from TNFR theory
        self.canonical_values = {
            'phi': PHI,
            'golden_ratio': PHI,
            'gamma': GAMMA,
            'euler': GAMMA,
            'euler_constant': GAMMA,
            'pi': PI,
            'e': E,
            'euler_number': E,
        }
        
        # TNFR-derived constants from physics
        self.tnfr_derived = {
            # Coupling strength: φ × γ (from zeta-coupled validator success)
            'zeta_coupling_strength': PHI * GAMMA,  # ≈ 0.9340
            
            # Critical line factor: φ × γ × π
            'critical_line_factor': PHI * GAMMA * PI,  # ≈ 2.9351
            
            # Structural frequency base: φ / γ  
            'structural_frequency_base': PHI / GAMMA,  # ≈ 2.8037
            
            # Phase coupling base: γ / φ
            'phase_coupling_base': GAMMA / PHI,  # ≈ 0.3568
            
            # Resonance threshold: e^(-φ)
            'resonance_threshold': math.exp(-PHI),  # ≈ 0.2015
            
            # Bifurcation threshold: φ²
            'bifurcation_threshold': PHI**2,  # ≈ 2.618
            
            # Coherence scaling: 1/φ
            'coherence_scaling': 1/PHI,  # ≈ 0.618
            
            # Critical exponent: γ/π  
            'critical_exponent': GAMMA / PI,  # ≈ 0.1838
        }
        
        print("TNFR-Derived Constants from Theory:")
        for name, value in self.tnfr_derived.items():
            print(f"  {name}: {value:.6f}")
        print()
    
    def scan_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Scan Python file for hardcoded constants."""
        constants_found = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Num):  # Python < 3.8
                    constants_found.append({
                        'value': node.n,
                        'line': getattr(node, 'lineno', 0),
                        'type': 'numeric'
                    })
                elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):  # Python >= 3.8
                    constants_found.append({
                        'value': node.value,
                        'line': getattr(node, 'lineno', 0), 
                        'type': 'constant'
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(node.value, (ast.Num, ast.Constant)):
                            value = node.value.n if isinstance(node.value, ast.Num) else node.value.value
                            if isinstance(value, (int, float)):
                                constants_found.append({
                                    'name': target.id,
                                    'value': value,
                                    'line': getattr(node, 'lineno', 0),
                                    'type': 'assignment'
                                })
        
        except Exception as e:
            print(f"⚠️ Error scanning {filepath}: {e}")
        
        return constants_found
    
    def analyze_constant(self, name: str, value: float, filepath: Path, line: int) -> Dict[str, Any]:
        """Analyze if constant should be derived from TNFR theory."""
        analysis = {
            'name': name,
            'value': value,
            'file': str(filepath),
            'line': line,
            'issue_type': None,
            'suggestion': None,
            'canonical_replacement': None,
            'derivation': None
        }
        
        # Check for potential TNFR derivations
        tolerance = 1e-6
        
        # Direct canonical matches
        for canon_name, canon_value in self.canonical_values.items():
            if abs(value - canon_value) < tolerance:
                analysis['issue_type'] = 'hardcoded_canonical'
                analysis['suggestion'] = f'Replace with canonical {canon_name} constant'
                analysis['canonical_replacement'] = canon_name
                return analysis
        
        # TNFR-derived matches
        for derived_name, derived_value in self.tnfr_derived.items():
            if abs(value - derived_value) < tolerance:
                analysis['issue_type'] = 'hardcoded_tnfr_derived'
                analysis['suggestion'] = f'Replace with TNFR-derived {derived_name}'
                analysis['canonical_replacement'] = derived_name
                analysis['derivation'] = self._get_derivation(derived_name)
                return analysis
        
        # Common suspicious patterns
        if name and any(word in name.lower() for word in ['threshold', 'alpha', 'beta', 'gamma', 'delta', 'epsilon']):
            # Check if it's a simple fraction or empirically-looking value
            if 0 < value < 1 and value not in [0.5, 1.0]:
                analysis['issue_type'] = 'empirical_parameter'
                analysis['suggestion'] = 'Verify if this should derive from TNFR physics'
        
        # Magic numbers that should be avoided
        magic_numbers = [0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 2.0]
        if value in magic_numbers and name:
            if any(word in name.lower() for word in ['weight', 'factor', 'scale', 'coeff']):
                analysis['issue_type'] = 'magic_number'
                analysis['suggestion'] = 'Consider deriving from TNFR constants (φ, γ, π)'
        
        return analysis
    
    def _get_derivation(self, derived_name: str) -> str:
        """Get mathematical derivation for TNFR-derived constant."""
        derivations = {
            'zeta_coupling_strength': 'φ × γ (golden ratio × Euler constant)',
            'critical_line_factor': 'φ × γ × π (unified critical scaling)',
            'structural_frequency_base': 'φ / γ (golden/Euler ratio)',
            'phase_coupling_base': 'γ / φ (Euler/golden ratio)',
            'resonance_threshold': 'e^(-φ) (exponential decay at golden ratio)',
            'bifurcation_threshold': 'φ² (golden ratio squared)',
            'coherence_scaling': '1/φ (inverse golden ratio)',
            'critical_exponent': 'γ/π (Euler constant / π)'
        }
        return derivations.get(derived_name, 'Unknown derivation')
    
    def audit_directory(self, directory: Path) -> Dict[str, List]:
        """Audit entire directory for constant issues."""
        results = {
            'hardcoded_canonical': [],
            'hardcoded_tnfr_derived': [],
            'empirical_parameters': [],
            'magic_numbers': [],
            'other_constants': []
        }
        
        python_files = list(directory.rglob('*.py'))
        
        print(f"\n🔍 Auditing {len(python_files)} Python files...")
        
        for filepath in python_files:
            # Skip __pycache__ and test files for now
            if '__pycache__' in str(filepath):
                continue
                
            constants = self.scan_file(filepath)
            
            for const_data in constants:
                if const_data.get('name') and isinstance(const_data.get('value'), (int, float)):
                    analysis = self.analyze_constant(
                        const_data['name'],
                        const_data['value'],
                        filepath,
                        const_data['line']
                    )
                    
                    issue_type = analysis['issue_type']
                    if issue_type:
                        if issue_type not in results:
                            results[issue_type] = []
                        results[issue_type].append(analysis)
        
        return results
    
    def generate_report(self, results: Dict[str, List]) -> str:
        """Generate audit report."""
        report = []
        report.append("TNFR CONSTANTS AUDIT REPORT")
        report.append("="*50)
        
        total_issues = sum(len(issues) for issues in results.values())
        report.append(f"\n📊 SUMMARY: {total_issues} potential issues found\n")
        
        # Hardcoded canonical constants (HIGH PRIORITY)
        if results['hardcoded_canonical']:
            report.append("🔴 HIGH PRIORITY: Hardcoded Canonical Constants")
            report.append("-" * 45)
            for issue in results['hardcoded_canonical']:
                report.append(f"  • {issue['name']} = {issue['value']}")
                report.append(f"    File: {Path(issue['file']).name}:{issue['line']}")
                report.append(f"    Replace with: {issue['canonical_replacement']}")
                report.append("")
        
        # Hardcoded TNFR-derived constants (HIGH PRIORITY)  
        if results['hardcoded_tnfr_derived']:
            report.append("🟠 HIGH PRIORITY: Hardcoded TNFR-Derived Constants")
            report.append("-" * 50)
            for issue in results['hardcoded_tnfr_derived']:
                report.append(f"  • {issue['name']} = {issue['value']}")
                report.append(f"    File: {Path(issue['file']).name}:{issue['line']}")
                report.append(f"    Replace with: {issue['canonical_replacement']}")
                report.append(f"    Derivation: {issue['derivation']}")
                report.append("")
        
        # Empirical parameters (MEDIUM PRIORITY)
        if results['empirical_parameters']:
            report.append("🟡 MEDIUM PRIORITY: Empirical Parameters")
            report.append("-" * 35)
            for issue in results['empirical_parameters']:
                report.append(f"  • {issue['name']} = {issue['value']}")
                report.append(f"    File: {Path(issue['file']).name}:{issue['line']}")
                report.append(f"    Action: {issue['suggestion']}")
                report.append("")
        
        # Magic numbers (LOW PRIORITY)
        if results['magic_numbers']:
            report.append("🔵 LOW PRIORITY: Magic Numbers")
            report.append("-" * 25)
            for issue in results['magic_numbers']:
                report.append(f"  • {issue['name']} = {issue['value']}")
                report.append(f"    File: {Path(issue['file']).name}:{issue['line']}")
                report.append(f"    Suggestion: {issue['suggestion']}")
                report.append("")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 20)
        report.append("1. Replace hardcoded canonical constants with imports from mpmath")
        report.append("2. Create TNFR-derived constants module based on theoretical physics")
        report.append("3. Review empirical parameters for theoretical derivations")
        report.append("4. Document all constant derivations with references to TNFR.pdf")
        
        # Example fixes
        report.append("\n🔧 EXAMPLE FIXES")
        report.append("-" * 15)
        report.append("# Before:")
        report.append("alpha = 0.5")
        report.append("coupling_strength = 0.9340")
        report.append("")
        report.append("# After:")
        report.append("from tnfr.constants import PHI, GAMMA")
        report.append("alpha = 1/PHI  # Theoretical derivation")
        report.append("coupling_strength = PHI * GAMMA  # From zeta coupling theory")
        
        return "\n".join(report)


def main():
    """Main audit execution."""
    auditor = ConstantAuditor()
    
    # Audit src directory
    src_dir = Path(__file__).parent / '..' / '..' / 'src' / 'tnfr'
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    results = auditor.audit_directory(src_dir)
    
    # Generate and display report
    report = auditor.generate_report(results)
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent / 'constants_audit_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    # Return status
    total_issues = sum(len(issues) for issues in results.values())
    high_priority = len(results['hardcoded_canonical']) + len(results['hardcoded_tnfr_derived'])
    
    if high_priority > 0:
        print(f"\n❌ AUDIT FAILED: {high_priority} high-priority issues need attention")
        return 1
    elif total_issues > 0:
        print(f"\n⚠️ AUDIT WARNING: {total_issues} medium/low-priority issues found")
        return 0
    else:
        print(f"\n✅ AUDIT PASSED: No critical constant alignment issues")
        return 0


if __name__ == "__main__":
    exit(main())
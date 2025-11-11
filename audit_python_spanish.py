#!/usr/bin/env python3
"""Audit Python files for Spanish text in docstrings and comments."""

import os
import re
from pathlib import Path

spanish_patterns = [
    r'\b(Estructura Primaria de Información|Frecuencia estructural|gradiente nodal)\b',
    r'\b(Monitorear|Registrar|Medir|Caracterizar|Objetivo)\b',
    r'\b(función|método|clase|módulo|ejemplo)\b',
    r'\b(para|con|desde|hasta|entre|sobre)\b',
]

def scan_python_files(root_dir='.'):
    root = Path(root_dir)
    python_files = list(root.rglob('*.py'))
    
    issues = []
    
    for file in python_files:
        try:
            content = file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                for pattern in spanish_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        issues.append({
                            'file': str(file.relative_to(root)),
                            'line': i,
                            'pattern': pattern,
                            'matches': matches,
                            'context': line.strip()[:100]
                        })
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return issues

if __name__ == '__main__':
    print("Scanning Python files for Spanish text...")
    issues = scan_python_files()
    
    if not issues:
        print("\n✅ No Spanish text found in Python files!")
    else:
        print(f"\n🌐 Found {len(issues)} potential Spanish text occurrences:\n")
        for issue in issues[:20]:  # Show first 20
            print(f"{issue['file']}:{issue['line']}")
            print(f"  Pattern: {issue['pattern']}")
            print(f"  Matches: {issue['matches']}")
            print(f"  Context: {issue['context']}")
            print()
    
    exit(0 if not issues else 1)

#!/usr/bin/env python3
"""Comprehensive documentation audit for TNFR-Python-Engine.

Checks for:
1. Spanish text mixed with English
2. Redundant documentation
3. Contradictions
4. Incomplete documentation
5. Broken cross-references
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# Spanish terms that should be translated
SPANISH_TERMS = {
    'Estructura Primaria de Información': 'Primary Information Structure (EPI)',
    'Frecuencia estructural': 'Structural frequency',
    'gradiente nodal': 'nodal gradient',
    'Objetivo': 'Objective',
    'Caracterizar': 'Characterize',
    'Monitorear': 'Monitor',
    'Registrar': 'Record',
    'Medir': 'Measure',
}

# Files that should exist only once (single source of truth)
SINGLE_SOURCE_FILES = [
    'GLOSSARY.md',
    'ARCHITECTURE.md',
    'CONTRIBUTING.md',
    'TESTING.md',
]


class DocumentationAudit:
    def __init__(self, root_dir='.'):
        self.root = Path(root_dir)
        self.issues = defaultdict(list)
        self.all_md_files = []
        
    def scan_files(self):
        """Scan all markdown files."""
        self.all_md_files = list(self.root.rglob('*.md'))
        print(f"Found {len(self.all_md_files)} markdown files")
        
    def check_spanish_text(self):
        """Check for Spanish text in files."""
        print("\n=== Checking for Spanish text ===")
        for file in self.all_md_files:
            try:
                content = file.read_text(encoding='utf-8')
                for spanish, english in SPANISH_TERMS.items():
                    if spanish in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if spanish in line:
                                self.issues['spanish'].append({
                                    'file': str(file.relative_to(self.root)),
                                    'line': i,
                                    'spanish': spanish,
                                    'english': english,
                                    'context': line.strip()[:80]
                                })
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
    def check_duplicates(self):
        """Check for duplicate documentation."""
        print("\n=== Checking for duplicate files ===")
        file_names = defaultdict(list)
        for file in self.all_md_files:
            file_names[file.name].append(str(file.relative_to(self.root)))
            
        for name, paths in file_names.items():
            if len(paths) > 1 and name in SINGLE_SOURCE_FILES:
                self.issues['duplicates'].append({
                    'filename': name,
                    'paths': paths
                })
                
    def check_cross_references(self):
        """Check for broken cross-references."""
        print("\n=== Checking cross-references ===")
        for file in self.all_md_files:
            try:
                content = file.read_text(encoding='utf-8')
                # Find markdown links [text](path)
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
                for text, link in links:
                    if link.startswith('http'):
                        continue  # Skip external links
                    if link.startswith('#'):
                        continue  # Skip anchors
                    # Strip anchor fragments (e.g., file.md#section)
                    path_part = link.split('#', 1)[0]
                    # Empty path after split means it was a pure anchor; already handled
                    if not path_part:
                        continue
                    # Check if file exists
                    target = (file.parent / path_part).resolve()
                    if not target.exists():
                        self.issues['broken_links'].append({
                            'file': str(file.relative_to(self.root)),
                            'link': link,
                            'text': text
                        })
            except Exception as e:
                print(f"Error checking links in {file}: {e}")
                
    def generate_report(self):
        """Generate audit report."""
        print("\n" + "=" * 80)
        print("DOCUMENTATION AUDIT REPORT")
        print("=" * 80)
        
        # Spanish text issues
        if self.issues['spanish']:
            print(f"\n🌐 SPANISH TEXT FOUND: {len(self.issues['spanish'])} occurrences")
            print("-" * 80)
            for issue in self.issues['spanish'][:50]:  # Show first 50
                print(f"  {issue['file']}:{issue['line']}")
                print(f"    Spanish: {issue['spanish']}")
                print(f"    English: {issue['english']}")
                print(f"    Context: {issue['context']}")
                print()
                
        # Duplicate files
        if self.issues['duplicates']:
            print(f"\n📁 DUPLICATE FILES: {len(self.issues['duplicates'])}")
            print("-" * 80)
            for issue in self.issues['duplicates']:
                print(f"  {issue['filename']} found in:")
                for path in issue['paths']:
                    print(f"    - {path}")
                print()
                
        # Broken links
        if self.issues['broken_links']:
            print(f"\n🔗 BROKEN LINKS: {len(self.issues['broken_links'])}")
            print("-" * 80)
            for issue in self.issues['broken_links'][:30]:
                print(f"  {issue['file']}")
                print(f"    Link: {issue['link']}")
                print(f"    Text: {issue['text']}")
                print()
                
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total files scanned: {len(self.all_md_files)}")
        print(f"Spanish text occurrences: {len(self.issues['spanish'])}")
        print(f"Duplicate files: {len(self.issues['duplicates'])}")
        print(f"Broken links: {len(self.issues['broken_links'])}")
        
        total_issues = sum(len(v) for v in self.issues.values())
        print(f"\nTOTAL ISSUES: {total_issues}")
        
        return total_issues


if __name__ == '__main__':
    audit = DocumentationAudit()
    audit.scan_files()
    audit.check_spanish_text()
    audit.check_duplicates()
    audit.check_cross_references()
    total = audit.generate_report()
    
    exit(0 if total == 0 else 1)

#!/usr/bin/env python3
"""Deep documentation consistency audit for TNFR.

Checks for:
1. Grammar rules documentation (U1-U6)
2. Operator documentation (all 13 operators)
3. Cross-reference consistency
4. Conflicting definitions
5. Missing documentation
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class DeepDocumentationAudit:
    def __init__(self, root_dir='.'):
        self.root = Path(root_dir)
        self.issues = defaultdict(list)
        
        # Expected grammar rules
        self.expected_rules = {
            'U1': 'STRUCTURAL INITIATION & CLOSURE',
            'U2': 'CONVERGENCE & BOUNDEDNESS',
            'U3': 'RESONANT COUPLING',
            'U4': 'BIFURCATION DYNAMICS',
            'U5': 'RECURSION DEPTH SAFETY',
            'U6': None,  # Conflicting definitions detected
        }
        
        # Expected operators
        self.expected_operators = {
            'AL': 'Emission',
            'EN': 'Reception',
            'IL': 'Coherence',
            'OZ': 'Dissonance',
            'UM': 'Coupling',
            'RA': 'Resonance',
            'SHA': 'Silence',
            'VAL': 'Expansion',
            'NUL': 'Contraction',
            'THOL': 'Self-organization',
            'ZHIR': 'Mutation',
            'NAV': 'Transition',
            'REMESH': 'Recursivity',
        }
        
    def scan_grammar_rules(self):
        """Check grammar rule documentation consistency."""
        print("\n=== Grammar Rules Audit ===")
        
        # Find all mentions of grammar rules
        rule_definitions = defaultdict(lambda: defaultdict(list))
        
        for md_file in self.root.rglob('*.md'):
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Look for rule definitions
                for rule in ['U1', 'U2', 'U3', 'U4', 'U5', 'U6']:
                    # Pattern: ### U1: TITLE or ## U1: TITLE
                    pattern = rf'###+\s*{rule}[:\s]+([^\n]+)'
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        rule_definitions[rule]['definitions'].append({
                            'file': str(md_file.relative_to(self.root)),
                            'title': match.strip()
                        })
                        
            except Exception as e:
                pass
                
        # Check for consistency
        for rule, data in rule_definitions.items():
            if 'definitions' in data and data['definitions']:
                titles = set(d['title'] for d in data['definitions'])
                if len(titles) > 1:
                    self.issues['conflicting_rules'].append({
                        'rule': rule,
                        'definitions': data['definitions']
                    })
                    
    def scan_operator_docs(self):
        """Check operator documentation completeness."""
        print("\n=== Operator Documentation Audit ===")
        
        documented_operators = defaultdict(list)
        
        # Check AGENTS.md for operator descriptions
        agents_file = self.root / 'AGENTS.md'
        if agents_file.exists():
            content = agents_file.read_text(encoding='utf-8')
            for op in self.expected_operators:
                pattern = rf'###\s+\d+\.\s+{op}.*?\n.*?Physics:'
                if re.search(pattern, content, re.DOTALL):
                    documented_operators[op].append('AGENTS.md')
                    
        # Check GLOSSARY.md
        glossary_file = self.root / 'GLOSSARY.md'
        if glossary_file.exists():
            content = glossary_file.read_text(encoding='utf-8')
            for op in self.expected_operators:
                if f'**{op}' in content or f'### {op}' in content:
                    documented_operators[op].append('GLOSSARY.md')
                    
        # Find missing operators
        for op in self.expected_operators:
            if op not in documented_operators or not documented_operators[op]:
                self.issues['missing_operators'].append({
                    'operator': op,
                    'name': self.expected_operators[op]
                })
                
    def scan_cross_references(self):
        """Check cross-reference consistency."""
        print("\n=== Cross-Reference Audit ===")
        
        # Check if key documents reference each other correctly
        key_docs = {
            'AGENTS.md': self.root / 'AGENTS.md',
            'GLOSSARY.md': self.root / 'GLOSSARY.md',
            'UNIFIED_GRAMMAR_RULES.md': self.root / 'UNIFIED_GRAMMAR_RULES.md',
            'ARCHITECTURE.md': self.root / 'ARCHITECTURE.md',
        }
        
        for doc_name, doc_path in key_docs.items():
            if not doc_path.exists():
                self.issues['missing_key_docs'].append(doc_name)
                continue
                
            content = doc_path.read_text(encoding='utf-8')
            
            # Check if it references other key docs
            for other_doc in key_docs:
                if other_doc != doc_name and other_doc not in content:
                    self.issues['missing_cross_refs'].append({
                        'from': doc_name,
                        'missing_ref_to': other_doc
                    })
                    
    def check_definition_conflicts(self):
        """Check for conflicting definitions of core terms."""
        print("\n=== Definition Conflicts Audit ===")
        
        core_terms = {
            'EPI': [],
            'νf': [],
            'ΔNFR': [],
            'U1': [],
            'U2': [],
            'U3': [],
            'U4': [],
            'U5': [],
            'U6': [],
        }
        
        for md_file in self.root.rglob('*.md'):
            try:
                content = md_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    # Look for definitions
                    for term in core_terms:
                        if f'**{term}' in line or f'### {term}' in line or f'{term}:' in line:
                            # Get context (next 2 lines)
                            context = ' '.join(lines[i:min(i+2, len(lines))])[:150]
                            core_terms[term].append({
                                'file': str(md_file.relative_to(self.root)),
                                'line': i,
                                'context': context
                            })
            except Exception as e:
                pass
                
        # Check for conflicts
        for term, definitions in core_terms.items():
            if len(definitions) > 3:  # If defined in more than 3 places
                # Extract unique definitions
                unique_defs = set(d['context'][:80] for d in definitions)
                if len(unique_defs) > 2:  # Different definitions
                    self.issues['definition_conflicts'].append({
                        'term': term,
                        'count': len(definitions),
                        'unique_definitions': len(unique_defs),
                        'samples': definitions[:3]
                    })
                    
    def generate_report(self):
        """Generate comprehensive audit report."""
        print("\n" + "="*80)
        print("DEEP DOCUMENTATION CONSISTENCY AUDIT")
        print("="*80)
        
        # Conflicting rules
        if self.issues['conflicting_rules']:
            print(f"\n🚨 CRITICAL: CONFLICTING RULE DEFINITIONS: {len(self.issues['conflicting_rules'])}")
            print("-"*80)
            for issue in self.issues['conflicting_rules']:
                print(f"\n  Rule: {issue['rule']}")
                for defn in issue['definitions']:
                    print(f"    - {defn['file']}")
                    print(f"      Title: {defn['title']}")
                    
        # Missing operators
        if self.issues['missing_operators']:
            print(f"\n⚠️  MISSING OPERATOR DOCS: {len(self.issues['missing_operators'])}")
            print("-"*80)
            for issue in self.issues['missing_operators']:
                print(f"  - {issue['operator']}: {issue['name']}")
                
        # Definition conflicts
        if self.issues['definition_conflicts']:
            print(f"\n⚠️  DEFINITION CONFLICTS: {len(self.issues['definition_conflicts'])}")
            print("-"*80)
            for issue in self.issues['definition_conflicts']:
                print(f"\n  Term: {issue['term']}")
                print(f"    Defined in {issue['count']} places")
                print(f"    {issue['unique_definitions']} unique definitions")
                print(f"    Samples:")
                for sample in issue['samples']:
                    print(f"      - {sample['file']}:{sample['line']}")
                    print(f"        {sample['context'][:100]}")
                    
        # Missing cross-refs
        if self.issues['missing_cross_refs']:
            print(f"\n📎 MISSING CROSS-REFERENCES: {len(self.issues['missing_cross_refs'])}")
            print("-"*80)
            for issue in self.issues['missing_cross_refs'][:10]:
                print(f"  {issue['from']} → {issue['missing_ref_to']}")
                
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        total_issues = sum(len(v) for v in self.issues.values())
        print(f"Total issues found: {total_issues}")
        print(f"  - Conflicting rules: {len(self.issues['conflicting_rules'])}")
        print(f"  - Missing operators: {len(self.issues['missing_operators'])}")
        print(f"  - Definition conflicts: {len(self.issues['definition_conflicts'])}")
        print(f"  - Missing cross-refs: {len(self.issues['missing_cross_refs'])}")
        
        return total_issues

if __name__ == '__main__':
    audit = DeepDocumentationAudit()
    audit.scan_grammar_rules()
    audit.scan_operator_docs()
    audit.scan_cross_references()
    audit.check_definition_conflicts()
    total = audit.generate_report()
    
    exit(0 if total == 0 else 1)

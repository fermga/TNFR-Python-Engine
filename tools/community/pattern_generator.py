#!/usr/bin/env python3
"""Pattern generator for TNFR community contributions.

Helps community members generate new domain patterns by providing templates,
validation, and guidance for creating high-quality structural patterns.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class PatternTemplate:
    """Template for a new domain pattern.
    
    Attributes
    ----------
    domain : str
        Domain identifier.
    use_case : str
        Use case description.
    suggested_sequence : List[str]
        Suggested operator sequence based on use case.
    template_code : str
        Python code template for the pattern.
    """
    
    domain: str
    use_case: str
    suggested_sequence: List[str]
    template_code: str


@dataclass
class PatternValidation:
    """Validation result for a community-submitted pattern.
    
    Attributes
    ----------
    valid : bool
        Whether pattern is valid.
    issues : List[str]
        List of validation issues.
    suggestions : List[str]
        Improvement suggestions.
    """
    
    valid: bool
    issues: List[str]
    suggestions: List[str]


class CommunityPatternGenerator:
    """Helps community generate new domain patterns.
    
    Provides templates, validation, and guidance for creating structural
    patterns that meet TNFR quality standards.
    
    Examples
    --------
    >>> generator = CommunityPatternGenerator()
    >>> template = generator.generate_pattern_template(
    ...     "medical", "crisis_intervention"
    ... )
    >>> print(template.template_code)
    """
    
    # Common pattern archetypes based on use case keywords
    ARCHETYPES = {
        "initiation": ["emission", "reception", "coherence"],
        "stabilization": ["coherence", "silence", "resonance"],
        "transformation": ["dissonance", "mutation", "coherence"],
        "expansion": ["expansion", "coupling", "resonance"],
        "integration": ["coupling", "self_organization", "coherence"],
        "crisis": ["dissonance", "silence", "coherence", "stabilization"],
        "exploration": ["expansion", "dissonance", "mutation"],
        "consolidation": ["contraction", "coherence", "silence"],
    }
    
    def generate_pattern_template(
        self, domain: str, use_case: str
    ) -> PatternTemplate:
        """Generate template for new domain pattern.
        
        Parameters
        ----------
        domain : str
            Domain identifier (e.g., "medical", "business").
        use_case : str
            Use case description (e.g., "crisis_intervention").
            
        Returns
        -------
        PatternTemplate
            Template with suggested sequence and code.
        """
        # Suggest sequence based on use case keywords
        suggested_sequence = self._suggest_sequence(use_case)
        
        # Generate template code
        template_code = self._generate_code_template(
            domain, use_case, suggested_sequence
        )
        
        return PatternTemplate(
            domain=domain,
            use_case=use_case,
            suggested_sequence=suggested_sequence,
            template_code=template_code,
        )
    
    def validate_community_pattern(
        self, pattern: List[str], context: str
    ) -> PatternValidation:
        """Validate community-submitted pattern.
        
        Parameters
        ----------
        pattern : List[str]
            Structural operator sequence.
        context : str
            Domain context description.
            
        Returns
        -------
        PatternValidation
            Validation results with issues and suggestions.
        """
        issues = []
        suggestions = []
        
        # Check for valid operators
        valid_operators = {
            "emission", "reception", "coherence", "dissonance",
            "coupling", "resonance", "silence", "expansion",
            "contraction", "self_organization", "mutation",
            "transition", "recursivity",
        }
        
        for op in pattern:
            if op not in valid_operators:
                issues.append(f"Unknown operator: {op}")
        
        # Check pattern length
        if len(pattern) < 2:
            issues.append("Pattern too short (minimum 2 operators)")
        
        if len(pattern) > 10:
            suggestions.append("Consider breaking into multiple patterns")
        
        # Check for coherence
        if "coherence" not in pattern:
            suggestions.append(
                "Consider adding 'coherence' operator for stabilization"
            )
        
        # Check for balance
        disruptive_ops = {"dissonance", "mutation", "expansion"}
        stabilizing_ops = {"coherence", "silence", "contraction"}
        
        disruptive_count = sum(1 for op in pattern if op in disruptive_ops)
        stabilizing_count = sum(1 for op in pattern if op in stabilizing_ops)
        
        if disruptive_count > stabilizing_count * 2:
            suggestions.append(
                "Pattern may be too disruptive; consider adding stabilizing operators"
            )
        
        valid = len(issues) == 0
        
        return PatternValidation(
            valid=valid, issues=issues, suggestions=suggestions
        )
    
    def _suggest_sequence(self, use_case: str) -> List[str]:
        """Suggest operator sequence based on use case.
        
        Parameters
        ----------
        use_case : str
            Use case description.
            
        Returns
        -------
        List[str]
            Suggested operator sequence.
        """
        use_case_lower = use_case.lower()
        
        # Check for archetype keywords
        for keyword, sequence in self.ARCHETYPES.items():
            if keyword in use_case_lower:
                return sequence.copy()
        
        # Default sequence
        return ["emission", "reception", "coherence"]
    
    def _generate_code_template(
        self, domain: str, use_case: str, sequence: List[str]
    ) -> str:
        """Generate Python code template for pattern.
        
        Parameters
        ----------
        domain : str
            Domain identifier.
        use_case : str
            Use case description.
        sequence : List[str]
            Operator sequence.
            
        Returns
        -------
        str
            Python code template.
        """
        pattern_name = use_case.lower().replace(" ", "_")
        sequence_str = str(sequence)
        
        template = f'''"""Pattern definition for {domain} domain: {use_case}"""

from tnfr.extensions.base import PatternDefinition

{pattern_name.upper()}_PATTERN = PatternDefinition(
    name="{pattern_name}",
    sequence={sequence_str},
    description="{use_case} pattern for {domain} domain",
    use_cases=[
        "Use case 1: [Describe specific scenario]",
        "Use case 2: [Describe another scenario]",
        "Use case 3: [Describe third scenario]",
    ],
    health_requirements={{
        "min_coherence": 0.75,
        "min_sense_index": 0.70,
    }},
    domain_context={{
        "real_world_mapping": "[Explain how this maps to {domain} concepts]",
        "expected_outcomes": "[What happens when pattern succeeds]",
        "failure_modes": "[Common failure patterns]",
    }},
    examples=[
        {{
            "name": "Example 1",
            "context": "[Specific situation in {domain}]",
            "sequence": {sequence_str},
            "health_metrics": {{"C_t": 0.82, "Si": 0.76}},
        }},
        {{
            "name": "Example 2",
            "context": "[Another situation]",
            "sequence": {sequence_str},
            "health_metrics": {{"C_t": 0.85, "Si": 0.81}},
        }},
        {{
            "name": "Example 3",
            "context": "[Third situation]",
            "sequence": {sequence_str},
            "health_metrics": {{"C_t": 0.79, "Si": 0.78}},
        }},
    ],
)
'''
        return template


def main() -> int:
    """Main entry point for pattern generator CLI.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    if len(sys.argv) < 3:
        print("Usage: python pattern_generator.py <domain> <use_case>")
        print("\nExample:")
        print("  python pattern_generator.py medical crisis_intervention")
        print("\nAvailable archetypes:")
        generator = CommunityPatternGenerator()
        for keyword, sequence in generator.ARCHETYPES.items():
            print(f"  - {keyword}: {sequence}")
        return 1
    
    domain = sys.argv[1]
    use_case = sys.argv[2]
    
    generator = CommunityPatternGenerator()
    template = generator.generate_pattern_template(domain, use_case)
    
    print(f"\n{'='*70}")
    print(f"Pattern Template: {domain}/{use_case}")
    print(f"{'='*70}\n")
    
    print(f"Suggested Sequence: {template.suggested_sequence}\n")
    print("Generated Code Template:")
    print("-" * 70)
    print(template.template_code)
    print("-" * 70)
    
    # Save to file
    output_file = f"{domain}_{use_case.lower().replace(' ', '_')}_pattern.py"
    with open(output_file, "w") as f:
        f.write(template.template_code)
    
    print(f"\n✅ Template saved to: {output_file}")
    print("\nNext steps:")
    print("1. Review and customize the generated template")
    print("2. Fill in use cases and domain context")
    print("3. Validate with real-world examples")
    print("4. Test health metrics achieve > 0.75 threshold")
    print("5. Add to your extension's patterns.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

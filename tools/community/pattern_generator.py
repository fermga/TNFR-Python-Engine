#!/usr/bin/env python3
"""Pattern generator helper for community contributors.

Helps community members generate new domain patterns with proper structure,
health validation, and documentation templates.

Examples
--------
>>> from tools.community.pattern_generator import CommunityPatternGenerator
>>> 
>>> generator = CommunityPatternGenerator()
>>> template = generator.generate_pattern_template("education", "student_learning")
>>> print(template.code)
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.operators.grammar import validate_sequence_with_health


__all__ = ["PatternTemplate", "CommunityPatternGenerator"]


@dataclass
class PatternTemplate:
    """Template for a new pattern.
    
    Attributes
    ----------
    pattern_id : str
        Snake_case pattern identifier
    pattern_name : str
        Human-readable pattern name
    domain : str
        Target domain
    code : str
        Generated Python code template
    """
    pattern_id: str
    pattern_name: str
    domain: str
    code: str


class CommunityPatternGenerator:
    """Helper for generating new domain patterns.
    
    Provides templates and helpers for community contributors to create
    new patterns following TNFR best practices.
    
    Examples
    --------
    >>> generator = CommunityPatternGenerator()
    >>> template = generator.generate_pattern_template(
    ...     domain="education",
    ...     use_case="student_learning_journey"
    ... )
    >>> print(template.code)
    """
    
    def generate_pattern_template(
        self,
        domain: str,
        use_case: str,
        pattern_name: Optional[str] = None,
    ) -> PatternTemplate:
        """Generate a pattern template for a new use case.
        
        Parameters
        ----------
        domain : str
            Domain name (e.g., "education", "scientific")
        use_case : str
            Use case identifier (snake_case)
        pattern_name : str, optional
            Human-readable name. If None, derived from use_case.
        
        Returns
        -------
        PatternTemplate
            Generated template with code and metadata
        """
        if pattern_name is None:
            # Convert snake_case to Title Case
            pattern_name = use_case.replace("_", " ").title()
        
        pattern_id = use_case
        
        code = f'''"""
"{pattern_name}" pattern for {domain} domain.

Add your pattern definition below. Follow these steps:
1. Design your operator sequence
2. Validate health score > 0.75
3. Add real-world use cases
4. Document structural insights
"""

PatternDefinition(
    name="{pattern_name}",
    description="TODO: Describe what this pattern achieves and when to use it",
    examples=[
        # TODO: Add your operator sequences here
        # Example template (replace with your actual sequence):
        [
            "emission",      # TODO: Explain why emission starts this pattern
            "reception",     # TODO: What is being received?
            "coherence",     # TODO: What coherence is established?
            # Add more operators as needed...
        ],
    ],
    min_health_score=0.75,  # Adjust if needed
    use_cases=[
        # TODO: Add specific real-world applications
        "TODO: Specific use case 1",
        "TODO: Specific use case 2",
        "TODO: Specific use case 3",
    ],
    structural_insights=[
        # TODO: Explain the structural mechanisms
        "TODO: Why does operator X come before Y?",
        "TODO: What coherence is preserved?",
        "TODO: What resonance is amplified?",
    ],
)
'''
        
        return PatternTemplate(
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            domain=domain,
            code=code,
        )
    
    def validate_community_pattern(
        self,
        sequence: List[str],
        context: str = "",
    ) -> Dict[str, any]:
        """Validate a community-submitted pattern sequence.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence to validate
        context : str
            Context/description for feedback
        
        Returns
        -------
        Dict[str, any]
            Validation results with health metrics and feedback
        """
        try:
            validation = validate_sequence_with_health(sequence)
            
            result = {
                "is_valid": validation.passed,
                "health_score": validation.health_metrics.overall_health if validation.passed else 0.0,
                "errors": [validation.error] if validation.error else [],
                "warnings": [],
                "feedback": [],
            }
            
            # Generate helpful feedback
            if validation.passed:
                health = validation.health_metrics.overall_health
                
                if health >= 0.85:
                    result["feedback"].append(
                        "✓ Excellent health score! Pattern is well-structured."
                    )
                elif health >= 0.75:
                    result["feedback"].append(
                        "✓ Good health score. Pattern meets acceptance criteria."
                    )
                else:
                    result["feedback"].append(
                        f"⚠ Health score {health:.3f} is below recommended 0.75. "
                        "Consider refining the sequence."
                    )
                
                # Check for common anti-patterns
                if sequence.count("dissonance") > 2:
                    result["feedback"].append(
                        "⚠ Multiple dissonance operators detected. "
                        "Ensure adequate coherence for stability."
                    )
                
                if "coherence" not in sequence:
                    result["feedback"].append(
                        "⚠ No coherence operator. Consider adding stabilization."
                    )
                
            else:
                result["feedback"].append(
                    "✗ Sequence is invalid. Check errors above."
                )
            
            return result
            
        except Exception as e:
            return {
                "is_valid": False,
                "health_score": 0.0,
                "errors": [str(e)],
                "warnings": [],
                "feedback": ["✗ Validation failed with error."],
            }
    
    def suggest_operators(
        self,
        intent: str,
        current_sequence: Optional[List[str]] = None,
    ) -> List[str]:
        """Suggest appropriate operators based on intent.
        
        Parameters
        ----------
        intent : str
            What you want to achieve (e.g., "stabilize", "explore", "connect")
        current_sequence : List[str], optional
            Current sequence for context-aware suggestions
        
        Returns
        -------
        List[str]
            Suggested operators with explanations
        """
        suggestions = {
            "start": ["emission", "reception"],
            "stabilize": ["coherence", "silence"],
            "explore": ["dissonance", "mutation", "self_organization"],
            "connect": ["coupling", "resonance"],
            "grow": ["expansion", "self_organization"],
            "simplify": ["contraction", "coherence"],
            "transform": ["mutation", "transition"],
            "amplify": ["resonance", "expansion"],
            "pause": ["silence", "coherence"],
        }
        
        intent_lower = intent.lower()
        
        # Find matching suggestions
        for key, operators in suggestions.items():
            if key in intent_lower:
                return operators
        
        # Default suggestions based on sequence state
        if current_sequence is None or len(current_sequence) == 0:
            return ["emission", "reception"]  # Good starting operators
        
        last_op = current_sequence[-1]
        
        # Context-aware suggestions
        if last_op == "emission":
            return ["reception", "coherence"]
        elif last_op == "dissonance":
            return ["coherence", "mutation", "contraction"]
        elif last_op == "coherence":
            return ["coupling", "resonance", "expansion", "silence"]
        else:
            return ["coherence"]  # Always safe to stabilize


def main() -> None:
    """CLI interface for pattern generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate TNFR pattern templates"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Template generation
    template_parser = subparsers.add_parser(
        "template",
        help="Generate pattern template"
    )
    template_parser.add_argument("domain", help="Domain name")
    template_parser.add_argument("use_case", help="Use case identifier")
    template_parser.add_argument(
        "--name",
        help="Human-readable pattern name"
    )
    
    # Pattern validation
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate pattern sequence"
    )
    validate_parser.add_argument(
        "sequence",
        help="Comma-separated operator sequence"
    )
    validate_parser.add_argument(
        "--context",
        default="",
        help="Context description"
    )
    
    # Operator suggestions
    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Suggest operators for intent"
    )
    suggest_parser.add_argument("intent", help="What you want to achieve")
    suggest_parser.add_argument(
        "--sequence",
        help="Current sequence (comma-separated)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    generator = CommunityPatternGenerator()
    
    if args.command == "template":
        template = generator.generate_pattern_template(
            domain=args.domain,
            use_case=args.use_case,
            pattern_name=args.name,
        )
        print(f"\nPattern Template: {template.pattern_name}")
        print(f"Domain: {template.domain}")
        print(f"ID: {template.pattern_id}")
        print(f"\n{template.code}")
    
    elif args.command == "validate":
        sequence = [op.strip() for op in args.sequence.split(",")]
        result = generator.validate_community_pattern(sequence, args.context)
        
        print(f"\nValidation Results:")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Health Score: {result['health_score']:.3f}")
        
        if result['errors']:
            print(f"\nErrors:")
            for error in result['errors']:
                print(f"  ✗ {error}")
        
        if result['warnings']:
            print(f"\nWarnings:")
            for warning in result['warnings']:
                print(f"  ⚠ {warning}")
        
        if result['feedback']:
            print(f"\nFeedback:")
            for feedback in result['feedback']:
                print(f"  {feedback}")
    
    elif args.command == "suggest":
        current = None
        if args.sequence:
            current = [op.strip() for op in args.sequence.split(",")]
        
        suggestions = generator.suggest_operators(args.intent, current)
        print(f"\nSuggested operators for '{args.intent}':")
        for op in suggestions:
            print(f"  - {op}")


if __name__ == "__main__":
    main()

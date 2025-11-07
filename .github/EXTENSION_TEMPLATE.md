# TNFR Domain Extension Template

This template helps you create a new domain extension for TNFR Grammar 2.0.

## Overview

**Domain Name:** `your_domain_name` (lowercase, alphanumeric with underscores)

**Description:** Brief description of your domain

**Target Users:** Who will use this extension?

## Setup

### 1. Create Extension Directory

```bash
mkdir -p src/tnfr/extensions/your_domain_name
```

### 2. Create Extension Class

Create `src/tnfr/extensions/your_domain_name/extension.py`:

```python
"""Your domain TNFR extension.

Provides validated patterns for [domain description].
"""

from __future__ import annotations

from typing import Dict
from ..base import TNFRExtension, PatternDefinition


__all__ = ["YourDomainExtension"]


class YourDomainExtension(TNFRExtension):
    """[Domain] domain extension for TNFR.
    
    Provides patterns validated for [domain] contexts including:
    - Pattern type 1
    - Pattern type 2
    - Pattern type 3
    
    Examples
    --------
    >>> from tnfr.extensions.your_domain_name import YourDomainExtension
    >>> extension = YourDomainExtension()
    >>> patterns = extension.get_pattern_definitions()
    """
    
    def get_domain_name(self) -> str:
        """Return domain identifier."""
        return "your_domain_name"
    
    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Return validated domain patterns.
        
        Returns
        -------
        Dict[str, PatternDefinition]
            Domain patterns with health scores > 0.75
        """
        return {
            "pattern_1": PatternDefinition(
                name="Pattern 1 Name",
                description="What this pattern achieves",
                examples=[
                    [
                        "emission",      # Why emission?
                        "reception",     # What's received?
                        "coherence",     # What coherence?
                        # Add more operators...
                    ],
                ],
                min_health_score=0.75,
                use_cases=[
                    "Specific use case 1",
                    "Specific use case 2",
                    "Specific use case 3",
                ],
                structural_insights=[
                    "Why operator X comes before Y",
                    "What coherence is preserved",
                    "What resonance is amplified",
                ],
            ),
            
            # Add more patterns...
        }
    
    def get_metadata(self) -> Dict[str, any]:
        """Return extension metadata."""
        return {
            "domain": "your_domain_name",
            "version": "1.0.0",
            "author": "Your Name / Organization",
            "description": "Brief description",
            "principles": [
                "Guiding principle 1",
                "Guiding principle 2",
            ],
        }
```

### 3. Create __init__.py

Create `src/tnfr/extensions/your_domain_name/__init__.py`:

```python
"""[Your domain] extension for TNFR.

Brief description of what this extension provides.

Examples
--------
>>> from tnfr.extensions.your_domain_name import YourDomainExtension
>>> from tnfr.extensions import get_global_registry
>>> 
>>> registry = get_global_registry()
>>> registry.register_extension(YourDomainExtension())
"""

from __future__ import annotations

from .extension import YourDomainExtension

__all__ = ["YourDomainExtension"]
```

## Validation

### Step 1: Design Patterns

Use the pattern generator to create templates:

```bash
python tools/community/pattern_generator.py template your_domain_name pattern_name
```

### Step 2: Validate Sequences

Validate each sequence:

```bash
python tools/community/pattern_generator.py validate "emission,reception,coherence,..."
```

Ensure all sequences achieve health score ≥ 0.75.

### Step 3: Validate Extension

Run the extension validator:

```bash
python tools/community/extension_validator.py \
    tnfr.extensions.your_domain_name \
    YourDomainExtension
```

Ensure:
- Overall score ≥ 0.75
- All patterns meet health requirements
- No critical issues

## Testing

### Create Tests

Create `tests/extensions/test_your_domain_name.py`:

```python
"""Tests for [your domain] extension."""

import pytest
from tnfr.extensions.your_domain_name import YourDomainExtension
from tnfr.extensions import ExtensionRegistry


class TestYourDomainExtension:
    """Test suite for your domain extension."""
    
    def test_domain_name(self):
        """Test domain name format."""
        ext = YourDomainExtension()
        assert ext.get_domain_name() == "your_domain_name"
    
    def test_pattern_health_scores(self):
        """Test all patterns meet health requirements."""
        from tnfr.operators.grammar import validate_sequence_with_health
        
        ext = YourDomainExtension()
        patterns = ext.get_pattern_definitions()
        
        assert len(patterns) >= 3, "Minimum 3 patterns required"
        
        for pattern_id, pattern_def in patterns.items():
            for sequence in pattern_def.examples:
                result = validate_sequence_with_health(sequence)
                assert result.is_valid, f"{pattern_id} has invalid sequence"
                assert result.health_metrics.overall_health >= 0.75, \
                    f"{pattern_id} health below 0.75"
    
    def test_registry_integration(self):
        """Test extension registers correctly."""
        registry = ExtensionRegistry()
        ext = YourDomainExtension()
        
        registry.register_extension(ext)
        assert "your_domain_name" in registry.list_extensions()
        
        patterns = registry.get_domain_patterns("your_domain_name")
        assert len(patterns) > 0
```

Run tests:

```bash
pytest tests/extensions/test_your_domain_name.py -v
```

## Documentation

### Add Examples

Create `examples/domain_applications/your_domain_examples.py`:

```python
"""Examples using [your domain] extension."""

from tnfr.extensions.your_domain_name import YourDomainExtension
from tnfr.extensions import get_global_registry
from tnfr.operators.grammar import validate_sequence_with_health


def main():
    """Demonstrate domain patterns."""
    # Register extension
    registry = get_global_registry()
    registry.register_extension(YourDomainExtension())
    
    # Get patterns
    patterns = registry.get_domain_patterns("your_domain_name")
    
    # Demonstrate each pattern
    for pattern_id, pattern_def in patterns.items():
        print(f"\nPattern: {pattern_def.name}")
        print(f"Description: {pattern_def.description}")
        
        for seq in pattern_def.examples:
            result = validate_sequence_with_health(seq)
            print(f"  Sequence: {' -> '.join(seq)}")
            print(f"  Health: {result.health_metrics.overall_health:.3f}")


if __name__ == "__main__":
    main()
```

## Submission

### Pre-Submission Checklist

- [ ] Extension class implemented
- [ ] Minimum 3 patterns with health ≥ 0.75
- [ ] All patterns have ≥ 3 example sequences
- [ ] Real-world use cases documented
- [ ] Structural insights provided
- [ ] Extension validator passes (score ≥ 0.75)
- [ ] Tests written and passing
- [ ] Example usage code provided
- [ ] Documentation complete

### Submit Pull Request

1. Create a fork of TNFR-Python-Engine
2. Create feature branch: `git checkout -b extension-your-domain`
3. Commit your changes following commit conventions
4. Push to your fork
5. Open Pull Request using the template
6. Include validation results in PR description

### Review Process

Maintainers will review:
1. Code quality and style
2. Pattern health scores
3. Documentation completeness
4. Test coverage
5. Real-world applicability
6. TNFR paradigm alignment

## Support

- **Questions:** Open a discussion on GitHub
- **Issues:** Use the "New Domain Extension" issue template
- **Community:** Join TNFR community channels

## Resources

- [TNFR Fundamentals](../../TNFR.pdf)
- [CONTRIBUTING.md](../../CONTRIBUTING.md)
- [Extension Validator](../../tools/community/extension_validator.py)
- [Pattern Generator](../../tools/community/pattern_generator.py)
- [Example Extensions](../../src/tnfr/extensions/)

---

**Happy extending! 🚀**

Remember: TNFR is about coherence, not control. Your patterns should help domains resonate with structural principles, not impose rigid frameworks.

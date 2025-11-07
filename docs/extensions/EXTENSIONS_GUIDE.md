# TNFR Extensions Guide

## Overview

TNFR Grammar 2.0 introduces a **community-driven extension system** that enables domain experts to contribute specialized patterns, health analyzers, and tools for specific application domains. This guide explains how to develop, validate, and contribute extensions to the TNFR ecosystem.

## Quick Start

### 1. Understand the Extension System

Extensions provide domain-specific implementations on top of TNFR's core structural operators:

```python
from tnfr.extensions.base import TNFRExtension

class YourDomainExtension(TNFRExtension):
    """Extension for your application domain."""
    
    def get_domain_name(self) -> str:
        return "your_domain"
    
    def get_pattern_definitions(self):
        # Return domain patterns
        pass
    
    def get_health_analyzers(self):
        # Return specialized analyzers
        pass
```

### 2. Study Example Extensions

- **Medical Extension** (`src/tnfr/extensions/medical/`): Therapeutic and clinical patterns
- **Business Extension** (`src/tnfr/extensions/business/`): Organizational and process patterns

### 3. Use Community Tools

```bash
# Generate pattern template
python tools/community/pattern_generator.py your_domain use_case_name

# Validate your extension
python tools/community/extension_validator.py your_domain
```

## Extension Architecture

### Core Components

1. **Extension Class**: Inherits from `TNFRExtension`, provides domain identification
2. **Patterns**: Domain-specific operator sequences with validation
3. **Health Analyzers**: Specialized metrics for domain assessment
4. **Cookbook**: Validated recipes for common scenarios
5. **Documentation**: READMEs explaining domain mapping

### Directory Structure

```
src/tnfr/extensions/your_domain/
├── __init__.py              # Extension class and registration
├── patterns.py              # PatternDefinition instances
├── health_analyzers.py      # Domain-specific health metrics
├── cookbook.py              # CookbookRecipe instances
└── README.md                # Domain documentation
```

## See Full Guide

For complete documentation, see:
- **Extension Template**: `.github/EXTENSION_TEMPLATE.md`
- **Contributing Guide**: `CONTRIBUTING.md` (Community Contributions section)
- **Example Extensions**: `src/tnfr/extensions/medical/README.md`, `src/tnfr/extensions/business/README.md`

## Community Tools

### Extension Validator

Validates extension quality before submission:

```bash
python tools/community/extension_validator.py your_domain
```

Checks:
- Code quality (structure, stubs, docs)
- Pattern health scores (> 0.75)
- Documentation completeness
- Test coverage (> 80%)

### Pattern Generator

Generates pattern templates with suggested sequences:

```bash
python tools/community/pattern_generator.py your_domain use_case_name
```

Recognizes keywords: initiation, stabilization, transformation, expansion, integration, crisis, exploration, consolidation

## Quick Reference

### Canonical Operators

- **emission**: Initiate resonant pattern
- **reception**: Receive and integrate external patterns
- **coherence**: Stabilize structural form
- **dissonance**: Introduce controlled instability
- **coupling**: Create structural links
- **resonance**: Amplify and propagate patterns
- **silence**: Temporarily freeze evolution
- **expansion**: Increase structural complexity
- **contraction**: Reduce structural complexity
- **self_organization**: Spontaneous pattern formation
- **mutation**: Phase transformation
- **transition**: Movement between states
- **recursivity**: Nested operator application

### Health Requirements

All patterns must achieve:
- **C(t) > 0.75**: Coherence threshold
- **Si > 0.70**: Sense index threshold
- **3+ validated examples**: Real-world validation

### Submission Checklist

- [ ] Extension follows TNFRExtension base class
- [ ] All patterns achieve health score > 0.75
- [ ] Minimum 3 validated use cases per pattern
- [ ] Integration tests included
- [ ] Documentation complete
- [ ] Validator passes
- [ ] Type checking passes (mypy)

## Getting Help

- **GitHub Issues**: Use issue templates for patterns/extensions
- **GitHub Discussions**: Questions and brainstorming
- **PR Reviews**: Implementation feedback

---

**Ready to contribute?** See `.github/EXTENSION_TEMPLATE.md` to get started! 🚀

# TNFR Domain Extension Template

This template guides you through creating a new domain extension for TNFR Grammar 2.0.

## Extension Overview

**Domain Name**: `your_domain_name`  
**Description**: Brief description of the domain and its TNFR applications  
**Maintainer**: Your GitHub username  
**Status**: `draft` | `review` | `stable`

---

## 1. Extension Structure

Your extension should follow this structure:

```
src/tnfr/extensions/your_domain_name/
├── __init__.py              # Extension registration and exports
├── patterns.py              # Domain-specific pattern definitions
├── health_analyzers.py      # Specialized health metrics
├── visualizers.py           # Domain-specific visualizations (optional)
├── cookbook.py              # Validated recipes for common scenarios
└── README.md                # Domain documentation
```

---

## 2. Base Extension Class

All extensions must inherit from `TNFRExtension`:

```python
from tnfr.extensions.base import TNFRExtension, PatternDefinition
from typing import Dict, List, Type

class YourDomainExtension(TNFRExtension):
    """Extension for [your domain] applications.
    
    This extension provides specialized patterns and health analyzers
    for [describe domain purpose and value].
    """
    
    def get_domain_name(self) -> str:
        """Return domain name identifier."""
        return "your_domain_name"
    
    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Return pattern definitions for this domain.
        
        Returns
        -------
        Dict[str, PatternDefinition]
            Mapping of pattern names to their definitions.
        """
        from .patterns import PATTERNS
        return PATTERNS
    
    def get_health_analyzers(self) -> Dict[str, Type]:
        """Return domain-specific health analyzers.
        
        Returns
        -------
        Dict[str, Type]
            Mapping of analyzer names to analyzer classes.
        """
        from .health_analyzers import DomainHealthAnalyzer
        return {
            "domain_health": DomainHealthAnalyzer,
        }
    
    def get_cookbook_recipes(self) -> Dict[str, Dict]:
        """Return validated recipes for common scenarios.
        
        Returns
        -------
        Dict[str, Dict]
            Mapping of recipe names to recipe definitions.
        """
        from .cookbook import RECIPES
        return RECIPES
    
    def get_visualization_tools(self) -> Dict[str, Type]:
        """Return domain-specific visualizers (optional).
        
        Returns
        -------
        Dict[str, Type]
            Mapping of visualizer names to visualizer classes.
        """
        # Optional - return {} if no visualizations
        return {}
```

---

## 3. Pattern Definitions

Define your domain patterns in `patterns.py`:

```python
from tnfr.extensions.base import PatternDefinition

PATTERNS = {
    "pattern_name": PatternDefinition(
        name="pattern_name",
        sequence=["emission", "reception", "coherence", "resonance"],
        description="What this pattern represents in your domain",
        use_cases=[
            "Use case 1: Specific scenario",
            "Use case 2: Another scenario",
            "Use case 3: Third scenario",
        ],
        health_requirements={
            "min_coherence": 0.75,
            "min_sense_index": 0.70,
        },
        domain_context={
            "real_world_mapping": "How this maps to domain concepts",
            "expected_outcomes": "What happens when pattern succeeds",
            "failure_modes": "Common failure patterns",
        },
        examples=[
            {
                "name": "Example 1",
                "context": "Specific situation",
                "sequence": ["emission", "reception", "coherence"],
                "health_metrics": {"C_t": 0.82, "Si": 0.76},
            },
            {
                "name": "Example 2",
                "context": "Another situation",
                "sequence": ["emission", "reception", "coherence", "resonance"],
                "health_metrics": {"C_t": 0.85, "Si": 0.81},
            },
        ],
    ),
}
```

---

## 4. Health Analyzers

Create specialized health metrics in `health_analyzers.py`:

```python
from tnfr.metrics import SequenceHealthAnalyzer
from typing import Dict, Any
import networkx as nx

class DomainHealthAnalyzer(SequenceHealthAnalyzer):
    """Specialized health analyzer for [your domain].
    
    Computes domain-specific health dimensions beyond standard
    coherence and sense index metrics.
    """
    
    def analyze_domain_health(
        self,
        G: nx.Graph,
        sequence: List[str],
        **kwargs: Any
    ) -> Dict[str, float]:
        """Compute domain-specific health metrics.
        
        Parameters
        ----------
        G : nx.Graph
            Network graph
        sequence : List[str]
            Operator sequence to analyze
        **kwargs : Any
            Additional analysis parameters
            
        Returns
        -------
        Dict[str, float]
            Domain-specific health metrics with values in [0, 1]
        """
        # Implement your domain-specific metrics
        metrics = {}
        
        # Example metric
        metrics["domain_quality"] = self._compute_domain_quality(G, sequence)
        metrics["domain_stability"] = self._compute_domain_stability(G, sequence)
        metrics["domain_effectiveness"] = self._compute_effectiveness(G, sequence)
        
        return metrics
    
    def _compute_domain_quality(self, G: nx.Graph, sequence: List[str]) -> float:
        """Compute domain-specific quality metric."""
        # Your implementation
        pass
    
    def _compute_domain_stability(self, G: nx.Graph, sequence: List[str]) -> float:
        """Compute domain-specific stability metric."""
        # Your implementation
        pass
    
    def _compute_effectiveness(self, G: nx.Graph, sequence: List[str]) -> float:
        """Compute domain-specific effectiveness metric."""
        # Your implementation
        pass
```

---

## 5. Cookbook Recipes

Provide validated recipes in `cookbook.py`:

```python
RECIPES = {
    "common_scenario_1": {
        "name": "Common Scenario Name",
        "description": "What this recipe achieves",
        "sequence": ["emission", "reception", "coherence"],
        "parameters": {
            "suggested_nf": 1.5,  # Hz_str
            "suggested_phase": 0.0,
        },
        "expected_health": {
            "min_C_t": 0.75,
            "min_Si": 0.70,
        },
        "validation": {
            "tested_cases": 20,
            "success_rate": 0.95,
            "notes": "Validated on real-world data",
        },
    },
}
```

---

## 6. Validation Requirements

Your extension must pass these validation checks:

### Pattern Validation
- ✅ All patterns use canonical English operator identifiers
- ✅ All examples achieve health score > 0.75
- ✅ Minimum 3 validated use cases per pattern
- ✅ Clear domain mapping documented

### Code Quality
- ✅ Type annotations complete (mypy passes)
- ✅ Docstrings follow NumPy style guide
- ✅ Integration tests included
- ✅ All tests pass

### TNFR Compliance
- ✅ Maintains operator closure
- ✅ Preserves structural invariants
- ✅ Follows domain neutrality in core
- ✅ Uses Hz_str units for frequencies

---

## 7. Testing Your Extension

Create tests in `tests/extensions/test_your_domain.py`:

```python
import pytest
from tnfr.extensions.your_domain import YourDomainExtension

def test_extension_registration():
    """Test extension can be registered."""
    ext = YourDomainExtension()
    assert ext.get_domain_name() == "your_domain_name"

def test_pattern_health_scores():
    """Test all patterns meet health requirements."""
    ext = YourDomainExtension()
    patterns = ext.get_pattern_definitions()
    
    for pattern_name, pattern_def in patterns.items():
        for example in pattern_def.examples:
            assert example["health_metrics"]["C_t"] > 0.75
            assert example["health_metrics"]["Si"] > 0.70

def test_health_analyzer():
    """Test domain health analyzer."""
    from tnfr.extensions.your_domain.health_analyzers import DomainHealthAnalyzer
    import networkx as nx
    
    G = nx.Graph()
    # Set up test network
    
    analyzer = DomainHealthAnalyzer()
    metrics = analyzer.analyze_domain_health(G, ["emission", "coherence"])
    
    assert 0.0 <= metrics["domain_quality"] <= 1.0
    assert 0.0 <= metrics["domain_stability"] <= 1.0
```

---

## 8. Documentation

Create `README.md` in your extension directory:

```markdown
# Your Domain Extension

## Overview
Brief description of the domain and how TNFR applies.

## Patterns

### Pattern 1: pattern_name
- **Sequence**: `["emission", "reception", "coherence"]`
- **Use Cases**: 
  1. Use case 1
  2. Use case 2
- **Domain Mapping**: How this maps to domain concepts

## Health Analyzers

### DomainHealthAnalyzer
Computes:
- `domain_quality`: Description of metric
- `domain_stability`: Description of metric

## Usage Examples

```python
from tnfr.extensions.your_domain import YourDomainExtension
from tnfr.extensions import registry

# Register extension
ext = YourDomainExtension()
registry.register_extension(ext)

# Use domain patterns
patterns = registry.get_domain_patterns("your_domain_name")
```

## Validation
- Tested on X real-world cases
- Average health score: Y
- Success rate: Z%

## References
- Domain-specific papers/resources
- Validation studies
```

---

## 9. Submission Process

1. **Fork the repository**
2. **Create your extension** following this template
3. **Run validation**: `python tools/community/extension_validator.py your_domain_name`
4. **Run tests**: `pytest tests/extensions/test_your_domain.py`
5. **Create PR** using the PR template
6. **Respond to reviews** and iterate

---

## 10. Example Extensions

Study these reference implementations:

- **Medical Extension**: `src/tnfr/extensions/medical/`
  - Therapeutic patterns
  - Clinical health analyzers
  - Treatment journey visualizations

- **Business Extension**: `src/tnfr/extensions/business/`
  - Process patterns
  - KPI analyzers
  - Organizational change tracking

---

## Questions?

- **Documentation**: See main docs at `docs/extensions/`
- **Issues**: Use issue template "Domain Extension"
- **Community**: Join discussions in GitHub Discussions

---

## Validation Checklist

Before submitting, verify:

- [ ] Extension class implements all required methods
- [ ] All patterns have >= 3 validated use cases
- [ ] All examples achieve health score > 0.75
- [ ] Health analyzers return values in [0, 1]
- [ ] Type annotations complete
- [ ] Docstrings follow NumPy style
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Validation script passes
- [ ] PR template filled out completely

**Ready?** Submit your extension and help grow the TNFR ecosystem! 🚀

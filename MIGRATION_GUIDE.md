# TNFR Modular Architecture Migration Guide

This guide helps you migrate to the new modular architecture introduced in TNFR 2.0, which provides clean separation of responsibilities through Protocol-based interfaces and dependency injection.

## What Changed?

TNFR 2.0 introduces:

1. **Core Interfaces** (`tnfr.core.interfaces`): Protocol-based contracts for each architectural layer
2. **Service Layer** (`tnfr.services`): `TNFROrchestrator` for coordinated execution
3. **Dependency Injection** (`tnfr.core.container`): `TNFRContainer` for flexible composition

**Important**: The existing API (`run_sequence`, `create_nfr`, etc.) remains unchanged and fully supported. The new architecture is additive, not breaking.

## Do I Need to Migrate?

**No!** Existing code continues to work without changes:

```python
# This still works exactly as before
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Reception, Coherence

G, node = create_nfr("test", epi=1.0, vf=1.0)
run_sequence(G, node, [Emission(), Reception(), Coherence()])
```

**However**, the new architecture provides benefits for:

- Testing with mocked services
- Custom validation or dynamics implementations
- Structured telemetry capture
- Clear separation of concerns

## When to Use the New Architecture

Use `TNFROrchestrator` when you need:

### 1. Detailed Telemetry Capture

```python
from tnfr.core import TNFRContainer
from tnfr.services import TNFROrchestrator

container = TNFRContainer.create_default()
orchestrator = TNFROrchestrator.from_container(container)

# Enable detailed telemetry
orchestrator.execute_sequence(
    G, node, 
    ["emission", "reception", "coherence", "coupling", "resonance", "silence"],
    enable_telemetry=True
)

# Access captured transitions
transitions = G.graph["_trace_transitions"]
for t in transitions:
    print(f"Operator: {t['operator']}, ΔC: {t['delta_coherence']:.3f}")
```

### 2. Custom Validation Logic

```python
from tnfr.core import TNFRContainer, ValidationService

class StrictValidator:
    """Custom validator with stricter rules."""
    
    def validate_sequence(self, sequence):
        # Add custom validation logic
        if len(sequence) < 3:
            raise ValueError("Sequence must have at least 3 operators")
        # Delegate to standard validation
        from tnfr.validation import validate_sequence
        outcome = validate_sequence(sequence)
        if not outcome.passed:
            raise ValueError(f"Invalid: {outcome.summary['message']}")
    
    def validate_graph_state(self, graph):
        # Add custom graph validation
        if graph.number_of_nodes() == 0:
            raise ValueError("Graph cannot be empty")

# Use custom validator
container = TNFRContainer.create_default()
container.register_singleton(ValidationService, StrictValidator())
orchestrator = TNFROrchestrator.from_container(container)
```

### 3. Testing with Mocks

```python
import pytest
from tnfr.core.interfaces import DynamicsEngine

class MockDynamics:
    """Mock dynamics for testing."""
    def __init__(self):
        self.updates = 0
    
    def update_delta_nfr(self, graph):
        self.updates += 1
    
    def integrate_nodal_equation(self, graph):
        pass
    
    def coordinate_phase_coupling(self, graph):
        pass

def test_orchestrator_calls_dynamics():
    # Inject mock
    container = TNFRContainer.create_default()
    mock_dynamics = MockDynamics()
    container.register_singleton(DynamicsEngine, mock_dynamics)
    
    orchestrator = TNFROrchestrator.from_container(container)
    orchestrator.execute_sequence(G, node, ["emission", "reception", "coherence", "silence"])
    
    # Verify dynamics was called
    assert mock_dynamics.updates == 4  # Once per operator
```

## Gradual Migration Strategy

### Step 1: Understand the Interfaces

Review the Protocol interfaces in `tnfr.core.interfaces`:

- `OperatorRegistry`: Operator token → implementation mapping
- `ValidationService`: Sequence and graph validation
- `DynamicsEngine`: ΔNFR computation and integration
- `TelemetryCollector`: Coherence metrics and traces

### Step 2: Start with Default Implementations

Begin using `TNFROrchestrator` with default implementations:

```python
# Before
from tnfr.structural import run_sequence

run_sequence(G, node, operators)

# After (optional, same behavior)
from tnfr.core import TNFRContainer
from tnfr.services import TNFROrchestrator

orchestrator = TNFROrchestrator.from_container(
    TNFRContainer.create_default()
)
orchestrator.execute_sequence(G, node, operator_tokens)
```

### Step 3: Customize as Needed

Replace specific services when you need custom behavior:

```python
from tnfr.core import TNFRContainer
from tnfr.core.interfaces import TelemetryCollector

class EnhancedTelemetry:
    """Custom telemetry with additional metrics."""
    
    def trace_context(self, graph):
        # Custom trace implementation
        ...
    
    def compute_coherence(self, graph):
        from tnfr.metrics.common import compute_coherence
        return compute_coherence(graph)
    
    def compute_sense_index(self, graph):
        from tnfr.metrics.sense_index import compute_Si
        return {"Si": compute_Si(graph), "custom_metric": ...}

# Use custom telemetry
container = TNFRContainer.create_default()
container.register_singleton(TelemetryCollector, EnhancedTelemetry())
orchestrator = TNFROrchestrator.from_container(container)
```

## API Comparison

### Creating and Running Sequences

#### Traditional API (Still Supported)

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.dynamics import set_delta_nfr_hook, dnfr_epi_vf_mixed
from tnfr.operators.definitions import Emission, Reception, Coherence

G, node = create_nfr("test", epi=1.0, vf=1.0)
set_delta_nfr_hook(G, dnfr_epi_vf_mixed)
run_sequence(G, node, [Emission(), Reception(), Coherence()])
```

#### New Orchestrator API (Optional)

```python
from tnfr.structural import create_nfr
from tnfr.dynamics import set_delta_nfr_hook, dnfr_epi_vf_mixed
from tnfr.core import TNFRContainer
from tnfr.services import TNFROrchestrator

G, node = create_nfr("test", epi=1.0, vf=1.0)
set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

orchestrator = TNFROrchestrator.from_container(
    TNFRContainer.create_default()
)
orchestrator.execute_sequence(
    G, node, 
    ["emission", "reception", "coherence", "coupling", "resonance", "silence"],
    enable_telemetry=False  # Set True for detailed traces
)
```

### Validation

#### Traditional API

```python
from tnfr.validation import validate_sequence

outcome = validate_sequence(["emission", "reception", "coherence"])
if not outcome.passed:
    print(f"Invalid: {outcome.summary['message']}")
```

#### New Orchestrator API

```python
orchestrator = TNFROrchestrator.from_container(...)

try:
    orchestrator.validate_only(["emission", "reception", "coherence"])
except ValueError as e:
    print(f"Invalid: {e}")
```

### Telemetry

#### Traditional API

```python
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si

coherence = compute_coherence(G)
si_metrics = compute_Si(G)
```

#### New Orchestrator API

```python
orchestrator = TNFROrchestrator.from_container(...)

coherence = orchestrator.get_coherence(G)
si_metrics = orchestrator.get_sense_index(G)
```

## Best Practices

### 1. Use Default Container When Starting

```python
# Good: Simple and works for most cases
container = TNFRContainer.create_default()
orchestrator = TNFROrchestrator.from_container(container)
```

### 2. Register Custom Services Selectively

```python
# Good: Only override what you need
container = TNFRContainer.create_default()
container.register_singleton(ValidationService, MyCustomValidator())
# Other services use defaults
```

### 3. Test with Mocks

```python
# Good: Test each layer independently
def test_validation_layer():
    mock_validator = MockValidator()
    container = TNFRContainer()
    container.register_singleton(ValidationService, mock_validator)
    # Test validation behavior
```

### 4. Keep Existing Code for Simple Cases

```python
# Good: Don't over-engineer simple scripts
from tnfr.structural import create_nfr, run_sequence

G, node = create_nfr("simple", epi=1.0)
run_sequence(G, node, [Emission()])
```

## Common Patterns

### Pattern 1: Enhanced Telemetry

```python
class DetailedTelemetry(DefaultTelemetryCollector):
    """Extends default telemetry with custom metrics."""
    
    def compute_sense_index(self, graph):
        si = super().compute_sense_index(graph)
        si["custom_stability"] = self._compute_stability(graph)
        return si
    
    def _compute_stability(self, graph):
        # Custom metric
        return ...

container = TNFRContainer.create_default()
container.register_singleton(TelemetryCollector, DetailedTelemetry())
```

### Pattern 2: Validation with Logging

```python
class LoggingValidator(DefaultValidationService):
    """Logs validation attempts."""
    
    def validate_sequence(self, sequence):
        print(f"Validating sequence: {sequence}")
        super().validate_sequence(sequence)
        print("✓ Validation passed")

container = TNFRContainer.create_default()
container.register_singleton(ValidationService, LoggingValidator())
```

### Pattern 3: Test Fixtures

```python
@pytest.fixture
def orchestrator():
    """Fixture providing orchestrator with default services."""
    container = TNFRContainer.create_default()
    return TNFROrchestrator.from_container(container)

def test_execution(orchestrator):
    G, node = create_nfr("test", epi=1.0)
    orchestrator.execute_sequence(G, node, ["emission", "reception", "coherence", "silence"])
    assert orchestrator.get_coherence(G) > 0
```

## FAQ

### Q: Should I migrate my existing code?

**A**: No, migration is optional. The new architecture is additive and your existing code will continue to work.

### Q: When should I use the orchestrator?

**A**: Use it when you need:
- Structured telemetry capture
- Custom implementations of validation/dynamics/telemetry
- Better testability through dependency injection
- Clear separation of concerns

### Q: Can I mix old and new APIs?

**A**: Yes! You can use `create_nfr`, `set_delta_nfr_hook` from the traditional API and then use `TNFROrchestrator` for execution.

### Q: Is the new architecture slower?

**A**: No. Default implementations wrap existing code with minimal overhead. Telemetry capture adds overhead only when `enable_telemetry=True`.

### Q: What about TNFR invariants?

**A**: All TNFR structural invariants are preserved. The default implementations ensure canonical behavior:
- νf stays in Hz_str
- ΔNFR semantics remain canonical
- Phase coupling is enforced
- Operator closure is maintained

## Support

For questions or issues:

1. Check the [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation
2. Review [tests/unit/core/](tests/unit/core/) for usage examples
3. Open an issue on GitHub

## Summary

The modular architecture provides:

✅ **Backward Compatibility**: Existing code works unchanged  
✅ **Flexibility**: Inject custom implementations  
✅ **Testability**: Mock services for unit testing  
✅ **Clarity**: Clean separation of concerns  
✅ **Gradual Adoption**: Migrate at your own pace  

Start exploring with `TNFRContainer.create_default()` and customize as needed!

"""Typing Migration Guide: Gradual Restoration Strategy

This guide outlines the incremental approach to reintroducing stricter
type annotations in TNFR-Python-Engine after lint-driven simplification.

Background
----------
During lint cleanup, some modules (operators/metrics.py) had type annotations
temporarily relaxed due to "Variable not allowed in type expression" errors
with TNFRGraph and NodeId TypeAliases. This was pragmatic but reduces IDE
support and static analysis benefits.

Root Cause
----------
The error stems from using runtime-assigned TypeAliases in function signatures
when type checkers expect static types. TypeAliases defined with conditional
TYPE_CHECKING blocks require careful handling.

Current Pattern (Simplified)
-----------------------------
```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId
else:
    TNFRGraph = Any  # runtime fallback
    NodeId = Any

def my_function(G, node, value: float) -> dict[str, Any]:
    # No type hints on G, node to avoid lint errors
    pass
```

Target Pattern (Strict)
-----------------------
```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId
else:
    TNFRGraph = Any
    NodeId = Any

def my_function(
    G: TNFRGraph,
    node: NodeId,
    value: float
) -> dict[str, Any]:
    # Full annotations with runtime fallbacks
    pass
```

Migration Strategy
------------------

### Phase 1: Establish Baseline (✓ Complete)
- Create TYPE_CHECKING pattern in affected modules
- Runtime fallbacks (TNFRGraph = Any, NodeId = Any)
- Verify no lint errors with untyped signatures
- **Status**: operators/metrics.py baseline established

### Phase 2: Verify Types Module Stability
**Goal**: Ensure types.py exports are stable and type-checker friendly

Steps:
1. Check types.py for conditional TypeAlias definitions
2. Verify TNFRGraph and NodeId are properly exported
3. Test import in isolated module with strict typing enabled
4. Document any Pyright/Pylance version-specific quirks

**Test file**: `tests/test_typing_baseline.py`
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tnfr.types import TNFRGraph, NodeId

def test_import_succeeds():
    # Should not raise ImportError at runtime
    pass
```

### Phase 3: Incremental Annotation (Module by Module)
**Goal**: Reintroduce annotations to 3-5 functions per iteration

Target modules (priority order):
1. `tnfr.metrics.local_coherence` (small, isolated)
2. `tnfr.operators.metrics_u6` (experimental, lower risk)
3. `tnfr.operators.metrics` (large, critical path)

**Per-module workflow**:
```bash
# 1. Add annotations to 3-5 functions
# 2. Run lint check
python -m pylance --check src/tnfr/module.py  # or IDE lint

# 3. If errors:
#    - Document specific error messages
#    - Check Pyright version: pyright --version
#    - Try alternative patterns (see workarounds below)

# 4. If clean:
#    - Commit
#    - Proceed to next batch
```

### Phase 4: Validate with Strict Mode
Once all annotations restored:
```json
// pyrightconfig.json (optional)
{
  "typeCheckingMode": "strict",
  "reportUnknownMemberType": false,  // networkx stubs incomplete
  "reportUnknownArgumentType": false
}
```

Run full type check:
```bash
pyright src/tnfr/
```

Workarounds for Persistent Issues
----------------------------------

### Workaround 1: Inline Type Comments
If signature annotations fail, use inline comments:
```python
def my_function(G, node, value: float) -> dict[str, Any]:
    G  # type: TNFRGraph
    node  # type: NodeId
    # Rest of function
```
**Issue**: Generates "unused expression" warnings; not ideal.

### Workaround 2: String Literal Annotations
Use forward-reference strings:
```python
def my_function(
    G: "TNFRGraph",
    node: "NodeId",
    value: float
) -> dict[str, Any]:
    pass
```
**Trade-off**: Less IDE autocomplete, but may avoid some lint errors.

### Workaround 3: Protocol Wrapper
For networkx Graph types, define Protocol:
```python
from typing import Protocol

class GraphLike(Protocol):
    nodes: Any
    graph: dict[str, Any]
    def neighbors(self, node: Any) -> Any: ...

def my_function(G: GraphLike, ...) -> ...:
    pass
```
**Benefit**: More precise contract without concrete type.

### Workaround 4: Generic Annotations
If TypeAlias is problematic, use generic bounds:
```python
from typing import TypeVar, Hashable

GraphT = TypeVar("GraphT")
NodeT = TypeVar("NodeT", bound=Hashable)

def my_function(G: GraphT, node: NodeT, ...) -> ...:
    pass
```

Testing Strategy
----------------

For each restored annotation batch:
1. Run lint: `pylance`, `pyright`, or VS Code Problems panel
2. Run tests: Ensure runtime behavior unchanged
3. Check IDE support: Hover over variables, check autocomplete
4. Commit with clear message: "types: restore annotations for X, Y, Z functions"

Success Criteria
----------------
- Zero lint errors in strict mode
- IDE autocomplete functional for typed variables
- No runtime errors (type hints are annotations only)
- Documentation builds without type reference errors

Rollback Plan
-------------
If strict typing proves untenable:
1. Document specific blockers (Pyright version, networkx stubs)
2. Keep TYPE_CHECKING pattern but skip annotations
3. Use `# type: ignore` comments sparingly for specific issues
4. Revisit after Pyright/Pylance updates or networkx stub improvements

Timeline Estimate
-----------------
- Phase 2 (Baseline verification): 30 minutes
- Phase 3 (Incremental restoration): 2-4 hours (depends on issue density)
- Phase 4 (Strict validation): 1 hour

Notes
-----
- Prioritize core public APIs (operators, metrics) over internal utilities
- Accept partial success: 80% annotated > 0% annotated
- Document blockers for future revisit (upstream deps, tooling versions)

References
----------
- PEP 484 – Type Hints: https://peps.python.org/pep-0484/
- PEP 563 – Postponed Evaluation of Annotations: https://peps.python.org/pep-0563/
- Pyright docs: https://github.com/microsoft/pyright
- NetworkX type stubs: https://github.com/networkx/networkx-stubs
"""

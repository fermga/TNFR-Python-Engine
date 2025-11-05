# Optional Dependencies

TNFR follows a modular design where certain features require optional dependencies that are not installed by default. This keeps installs focused while allowing users to add only what they need.

## Overview

The TNFR engine has three categories of dependencies:

1. **Core dependencies** (always installed): `networkx`, `cachetools`, `numpy`
2. **Optional dependencies**: Enable specific features like JAX/PyTorch backends, visualization, serialization
3. **Development dependencies**: For testing, documentation, and type checking

## Installing Optional Dependencies

### Using pip extras

The recommended way to install optional dependencies is using pip extras:

```bash
# Computational backends
pip install tnfr[compute-jax]     # JAX backend (JIT, autodiff)
pip install tnfr[compute-torch]   # PyTorch backend (GPU acceleration)

# Visualization
pip install tnfr[viz-basic]       # Matplotlib for plotting

# Serialization
pip install tnfr[yaml]            # YAML configuration support
pip install tnfr[orjson]          # Fast JSON serialization
pip install tnfr[serialization]   # Both YAML and orjson

# Development environments
pip install tnfr[dev-minimal]     # Basic dev tools (mypy, black, pytest)
pip install tnfr[dev-full]        # Full dev environment

# Testing
pip install tnfr[test-unit]       # Unit testing only
pip install tnfr[test-all]        # All testing tools

# Multiple extras at once
pip install tnfr[compute-jax,viz-basic,yaml]

# Legacy aliases (for backward compatibility)
pip install tnfr[numpy]           # NumPy is now core - this does nothing
pip install tnfr[jax]             # Alias for compute-jax
pip install tnfr[torch]           # Alias for compute-torch
pip install tnfr[viz]             # Alias for viz-basic
```

### Manual installation

You can also install dependencies manually:

```bash
# For JAX backend
pip install jax

# For PyTorch backend
pip install torch

# For visualization
pip install matplotlib

# For YAML support
pip install pyyaml

# For fast JSON
pip install orjson
```

## Feature Matrix

| Feature | Required Dependencies | Install Command |
|---------|----------------------|-----------------|
| Core TNFR operations | networkx, cachetools, numpy | `pip install tnfr` |
| NumPy backend | (included in core) | `pip install tnfr` |
| JAX backend | jax | `pip install tnfr[compute-jax]` |
| PyTorch backend | torch | `pip install tnfr[compute-torch]` |
| Visualization plots | matplotlib | `pip install tnfr[viz-basic]` |
| YAML configuration | pyyaml | `pip install tnfr[yaml]` |
| Fast JSON serialization | orjson | `pip install tnfr[orjson]` |
| JSON schema validation | jsonschema | `pip install jsonschema` |

## Fallback Behavior

When optional dependencies are not installed, TNFR provides graceful fallbacks:

### Computational Backends
- **NumPy backend**: Always available (core dependency) - canonical reference implementation
- **JAX backend**: Optional - provides JIT compilation and autodiff support
- **PyTorch backend**: Optional - provides GPU acceleration
- **Fallback**: Automatically uses NumPy backend if JAX or PyTorch unavailable
- **Detection**: Automatic - no configuration needed

### Matplotlib
- **Without Matplotlib**: Visualization functions raise informative errors
- **With Matplotlib**: Full plotting capabilities via `tnfr.viz`
- **Example**:
  ```python
  from tnfr.viz import plot_coherence_matrix
  # ImportError with installation instructions if matplotlib missing
  ```

### JSON Schema
- **Without jsonschema**: Operator grammar validation still works with basic checks
- **With jsonschema**: Full JSON schema validation for operator sequences
- **Fallback**: Logs a warning and continues with simplified validation

## Type Checking

TNFR includes type stubs for optional dependencies to support static type checking even when packages aren't installed:

### MyPy Configuration

The `pyproject.toml` includes mypy overrides for optional dependencies:

```toml
[[tool.mypy.overrides]]
module = ["numpy", "numpy.*", "matplotlib", "matplotlib.*", "jsonschema", "jsonschema.*"]
ignore_missing_imports = true
```

### Pyright Configuration

A `pyrightconfig.json` is provided with appropriate settings:

```json
{
  "reportMissingImports": "warning",
  "reportMissingTypeStubs": false
}
```

Note: The compat modules are runtime compatibility helpers, not traditional .pyi stub files. Type checkers will use the mypy configuration to ignore missing optional imports.

## Compatibility Stubs

TNFR provides lightweight stubs in `tnfr.compat` for type compatibility:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
else:
    from tnfr.compat import numpy_stub as np
```

These stubs:
- Allow code to type-check correctly
- Raise informative errors at runtime if used without the real package
- Are automatically used by type checkers when real packages aren't available

## Examples and Documentation

All examples that require optional dependencies include installation instructions:

```python
# example_visualization.py
"""
This example requires visualization dependencies.
Install with: pip install tnfr[viz]
"""

from tnfr.viz import plot_coherence_matrix
# ... rest of example
```

## Testing with Optional Dependencies

When running tests, you can control which optional dependencies are required:

```bash
# Run all tests (requires test dependencies)
pip install tnfr[test]
pytest

# Run tests without optional features
pytest -m "not slow" --ignore=tests/viz/
```

## Troubleshooting

### Import Errors

If you see import errors for optional dependencies:

1. Check which feature you're trying to use
2. Install the corresponding extra: `pip install tnfr[extra]`
3. Verify installation: `pip list | grep <package>`

### Type Checking Issues

If your type checker reports missing imports:

1. Ensure `pyrightconfig.json` or mypy configuration is being read
2. Verify stub path is correct
3. For mypy, check that ignore_missing_imports is set for optional packages

### IDE Support

For better IDE support with optional dependencies:

1. Install the optional dependencies in your development environment
2. Or configure your IDE to use the provided stubs in `tnfr.compat`

## Development Setup

For TNFR development, install all dependencies:

```bash
# Minimal development environment (testing and linting)
pip install -e ".[dev-minimal]"

# Complete development environment
pip install -e ".[dev-full]"

# Or install specific groups as needed
pip install -e ".[test-all,typecheck,compute-jax,viz-basic,serialization]"
```

This ensures all tests, documentation, and type checking work correctly.

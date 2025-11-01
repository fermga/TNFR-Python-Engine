# Mathematics backends

TNFR separates structural semantics from numerical implementations via the
`tnfr.mathematics.backend` module.  This lets you couple nodes, evaluate
coherence, and propagate ΔNFR in environments that favour different numerical
libraries.

## Selecting a backend

```python
from tnfr.mathematics import get_backend

backend = get_backend("jax")  # explicit name overrides other signals
xp = backend.as_array
```

Resolution order:

1. Explicit name passed to `get_backend`.
2. `TNFR_MATH_BACKEND` environment variable.
3. `tnfr.config.get_flags().math_backend`.
4. NumPy (default).

The default keeps NumPy active so canonical coherence operators continue to
work when optional dependencies are absent.

## Available adapters

| Backend | Extra dependency | Autodiff | Notes |
| ------- | ---------------- | -------- | ----- |
| NumPy   | `pip install tnfr[numpy]` | No | Canonical reference implementation. Uses SciPy for `expm` when available, otherwise falls back to an eigen decomposition strategy. |
| JAX     | `pip install tnfr[jax]`   | Yes | Requires `jax` and `jax.scipy`. Some imperative NumPy routines (e.g. in observers) remain NumPy-only. |
| PyTorch | `pip install tnfr[torch]` | Yes | Uses `torch.linalg`. Exporting tensors to NumPy moves them to CPU and breaks gradients. |

> Autodifferentiation support is scoped to backend operations.  TNFR structural
> pipelines that execute pure NumPy functions or rely on mutable state will not
> become differentiable automatically.

## Configuration helpers

Use `tnfr.config.context_flags` to adjust the backend temporarily without
mutating global state:

```python
from tnfr.config import context_flags
from tnfr.mathematics import get_backend

with context_flags(math_backend="torch"):
    torch_backend = get_backend()
```

Set `TNFR_MATH_BACKEND` in your environment to persist a preference across
processes:

```bash
export TNFR_MATH_BACKEND=jax
```

Remember to install the matching extra before enabling a backend.

# Torch Backend Support

**Status:** Supported optional numerical backend
**Scope:** Backend compatibility; no dedicated CUDA engine or speedup guarantee

Install the optional Torch dependency with:

```bash
pip install -e ".[compute-torch]"
```

Select the backend through the public mathematics backend interface:

```python
from tnfr.mathematics.backend import get_backend

backend = get_backend("torch")
```

The backend uses the Torch installation and devices available in the caller's
environment. Backend agreement is covered by
[`tests/mathematics/test_backends.py`](../tests/mathematics/test_backends.py).

TNFR does not currently ship the historical
tnfr.engines.computation.gpu_engine.TNFRGPUEngine class. The
[pytorch_cuda_demo.py](../examples/10_applications/pytorch_cuda_demo.py)
filename is retained as a compatibility and provenance check: it verifies
Torch operations against NumPy and reports the selected device. The canonical
graph-pressure adapter currently reports a CPU realization. The example does
not assert a CUDA speedup.

A future GPU acceleration claim must include:

1. an implementation reachable through a supported public API;
2. numerical agreement with the canonical CPU computation;
3. reproducible inputs, seeds, hardware and software versions;
4. warm-up, transfer-time and memory accounting;
5. recorded benchmark results with uncertainty and crossover sizes.

Until those conditions are met, `compute-torch` means optional Torch numerical
compatibility, not guaranteed CUDA acceleration.

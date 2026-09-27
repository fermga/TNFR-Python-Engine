# Torch Backend Support

**Status:** Supported optional numerical backend
**Scope:** Backend compatibility; no dedicated CUDA engine or speedup guarantee

Install the optional Torch dependency with:

```bash
pip install -e ".[compute-torch]"
```

Request the backend through the public mathematics backend interface and inspect
the actual selection:

```python
from tnfr.mathematics.backend import get_backend

backend = get_backend("torch")
print(backend.name, backend.get_device_name())
```

If Torch is unavailable, `get_backend("torch")` can return the NumPy fallback.
Check `backend.name == "torch"` when Torch execution is required. A Torch backend
can itself use CPU or CUDA according to the installation, device availability
and `TNFR_CUDA_ENABLED`; requesting Torch does not guarantee GPU execution.
`backend.get_backend_info()` exposes additional execution metadata.

Backend agreement is covered by
[`tests/mathematics/test_backends.py`](../tests/mathematics/test_backends.py).
Those backend-specific checks skip when the requested dependency is unavailable;
a passing NumPy-only run does not establish Torch agreement.

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

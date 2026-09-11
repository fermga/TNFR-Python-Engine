"""Inspect TNFR Torch compatibility and execution provenance.

The historical filename is retained for discoverability. This example checks
which device the optional Torch numerical backend actually selects, verifies
two linear-algebra results against NumPy, and shows that the current canonical
graph-pressure adapter reports its CPU realization explicitly. It does not
perform or imply a speed benchmark.

Run after installing the optional dependency:

    pip install -e ".[compute-torch]"
    python examples/10_applications/pytorch_cuda_demo.py
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from tnfr.engines.computation.unified_gpu_system import get_unified_gpu_system
from tnfr.mathematics.backend import get_backend

SEED = 42


def _to_numpy(value: object) -> np.ndarray:
    """Detach a backend array through the public conversion operation."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _pressure_graph() -> nx.Graph:
    graph = nx.path_graph(4)
    for node, epi in enumerate((0.0, 1.0, -0.5, 0.25)):
        graph.nodes[node].update(EPI=epi, nu_f=1.0, theta=0.0)
    return graph


def demonstrate_torch_compatibility() -> None:
    """Run deterministic backend-agreement and provenance checks."""

    try:
        import torch
    except ImportError:
        print('Torch is not installed. Install the "compute-torch" extra.')
        return

    backend = get_backend("torch")
    device_info = (
        backend.get_device_info()
        if hasattr(backend, "get_device_info")
        else {"device": "unknown", "use_cuda": False}
    )
    print(f"Torch version: {torch.__version__}")
    print(f"TNFR backend: {backend.name}")
    print(f"Selected device: {device_info.get('device', 'unknown')}")
    print(f"CUDA selected: {bool(device_info.get('use_cuda', False))}")

    rng = np.random.default_rng(SEED)
    matrix = rng.normal(size=(32, 32))
    symmetric = (matrix + matrix.T) / 2.0
    vector = rng.normal(size=32)

    backend_matrix = backend.as_array(symmetric)
    backend_vector = backend.as_array(vector)

    eigenvalues, _ = backend.eigh(backend_matrix)
    product = backend.matmul(backend_matrix, backend_vector)

    np.testing.assert_allclose(
        _to_numpy(eigenvalues),
        np.linalg.eigvalsh(symmetric),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        _to_numpy(product),
        symmetric @ vector,
        rtol=1e-6,
        atol=1e-7,
    )
    print("NumPy agreement checks passed.")

    pressure = get_unified_gpu_system().compute_delta_nfr_from_graph(
        _pressure_graph()
    )
    print(f"Canonical graph-pressure backend: {pressure.backend_used}")
    print(f"Compatibility fallback used: {pressure.fallback_used}")
    print(f"Pressure values: {dict(pressure)}")

    print(
        "No acceleration conclusion is drawn without a matched CPU/GPU "
        "benchmark including warm-up, transfer costs, hardware, and uncertainty."
    )


if __name__ == "__main__":
    demonstrate_torch_compatibility()

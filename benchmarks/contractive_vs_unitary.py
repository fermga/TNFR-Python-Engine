"""Compare unitary vs. contractive ΔNFR evolution on small Hilbert spaces."""
from __future__ import annotations

import argparse
import math
import statistics
import time
from typing import Any, Sequence

import numpy as np

from tnfr.mathematics import (
    ContractiveDynamicsEngine,
    HilbertSpace,
    MathematicalDynamicsEngine,
    build_delta_nfr,
    build_lindblad_delta_nfr,
    ensure_array,
    ensure_numpy,
    get_backend,
)


def _random_state(dim: int, seed: int, backend) -> Any:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(dim)
    imag = rng.standard_normal(dim)
    vector = real + 1j * imag
    norm = np.linalg.norm(vector)
    if math.isclose(norm, 0.0):
        raise ValueError("Random state generation failed: null vector.")
    return ensure_array(vector / norm, backend=backend)


def _lowering_operator(dim: int, amplitude: float) -> Sequence[np.ndarray]:
    operators: list[np.ndarray] = []
    for level in range(1, dim):
        lowering = np.zeros((dim, dim), dtype=np.complex128)
        lowering[level - 1, level] = amplitude
        operators.append(lowering)
    return operators


def _unitary_benchmark(
    engine: MathematicalDynamicsEngine,
    state: Any,
    *,
    steps: int,
    dt: float,
) -> tuple[float, float]:
    start = time.perf_counter()
    current = state
    for _ in range(steps):
        current = engine.step(current, dt=dt)
    duration = time.perf_counter() - start
    norm_backend = engine.backend.norm(current)
    norm = float(np.asarray(ensure_numpy(norm_backend, backend=engine.backend)))
    return duration, norm


def _contractive_benchmark(
    engine: ContractiveDynamicsEngine,
    density: Any,
    *,
    steps: int,
    dt: float,
) -> tuple[float, float, float]:
    start = time.perf_counter()
    current = density
    gap_sum = 0.0
    gap_count = 0
    for _ in range(steps):
        current = engine.step(current, dt=dt)
        if np.isfinite(engine.last_contractivity_gap):
            gap_sum += engine.last_contractivity_gap
            gap_count += 1
    duration = time.perf_counter() - start
    norm = engine.frobenius_norm(current)
    average_gap = gap_sum / gap_count if gap_count else float("nan")
    return duration, norm, average_gap


def run(
    dim: int = 2,
    steps: int = 256,
    repeats: int = 12,
    *,
    dt: float = 0.04,
    gamma: float = 0.35,
    backend_name: str = "numpy",
    use_scipy: bool | None = None,
) -> None:
    """Benchmark unitary vs. contractive ΔNFR evolutions."""

    backend = get_backend(backend_name)
    hilbert = HilbertSpace(dim)
    hermitian_generator = build_delta_nfr(dim, nu_f=1.0)
    unitary_engine = MathematicalDynamicsEngine(
        hermitian_generator,
        hilbert,
        backend=backend,
        use_scipy=use_scipy,
    )

    collapse_ops = _lowering_operator(dim, math.sqrt(gamma))
    lindblad = build_lindblad_delta_nfr(collapse_operators=collapse_ops, dim=dim)
    contractive_engine = ContractiveDynamicsEngine(
        lindblad,
        hilbert,
        backend=backend,
        use_scipy=use_scipy,
    )

    unitary_durations: list[float] = []
    contractive_durations: list[float] = []
    contractive_norms: list[float] = []
    contractive_gaps: list[float] = []

    for rep in range(repeats):
        state = _random_state(dim, seed=rep + 1, backend=backend)
        state_np = ensure_numpy(state, backend=backend)
        density = ensure_array(np.outer(state_np, state_np.conj()), backend=backend)

        duration, norm = _unitary_benchmark(unitary_engine, state, steps=steps, dt=dt)
        unitary_durations.append(duration)
        assert math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9)

        duration, frobenius, avg_gap = _contractive_benchmark(
            contractive_engine,
            density,
            steps=steps,
            dt=dt,
        )
        contractive_durations.append(duration)
        contractive_norms.append(frobenius)
        contractive_gaps.append(avg_gap)

    def _summary(values: Sequence[float]) -> tuple[float, float, float, float]:
        if not values:
            return (0.0, 0.0, 0.0, 0.0)
        best = min(values)
        worst = max(values)
        median = statistics.median(values)
        mean = sum(values) / len(values)
        return best, median, mean, worst

    u_best, u_median, u_mean, u_worst = _summary(unitary_durations)
    c_best, c_median, c_mean, c_worst = _summary(contractive_durations)

    print(
        "ΔNFR evolution benchmark:",
        f"dim={dim} steps={steps} repeats={repeats} dt={dt} gamma={gamma} backend={backend_name}",
    )
    print(
        "unitary  :",
        f"best={u_best:.6f}s median={u_median:.6f}s mean={u_mean:.6f}s worst={u_worst:.6f}s",
    )
    print(
        "contractive:",
        f"best={c_best:.6f}s median={c_median:.6f}s mean={c_mean:.6f}s worst={c_worst:.6f}s",
    )
    if contractive_norms:
        avg_norm = sum(contractive_norms) / len(contractive_norms)
        print(f"contractive Frobenius norm after {steps} steps: avg={avg_norm:.6f}")
    if contractive_gaps:
        avg_gap = sum(contractive_gaps) / len(contractive_gaps)
        print(f"average contractivity gap (centered Frobenius): {avg_gap:.6f}")


if __name__ == "__main__":  # pragma: no cover - manual benchmark entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--backend", type=str, default="numpy")
    parser.add_argument(
        "--use-scipy",
        action="store_true",
        help="Force SciPy propagators even when the backend supports autodiff.",
    )
    args = parser.parse_args()
    run(
        dim=args.dim,
        steps=args.steps,
        repeats=args.repeats,
        dt=args.dt,
        gamma=args.gamma,
        backend_name=args.backend,
        use_scipy=True if args.use_scipy else None,
    )

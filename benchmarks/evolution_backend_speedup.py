"""Benchmark backend trajectories for unitary and contractive dynamics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from tnfr.mathematics import HilbertSpace
from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.dynamics import ContractiveDynamicsEngine, MathematicalDynamicsEngine

DEFAULT_SIZES = (2, 4, 8, 16)
DEFAULT_STEPS = 32
DEFAULT_REPEATS = 3
DEFAULT_DT = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=list(DEFAULT_SIZES),
        help="Hilbert space dimensions to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of evolution steps per run (default: %(default)s)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="How many times to repeat each measurement (default: %(default)s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Time increment between steps (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used to generate synthetic generators (default: %(default)s)",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Subset of backends to benchmark (default: auto-detect all available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to persist JSON results for reproducibility",
    )
    return parser.parse_args()


def _resolve_backends(names: Iterable[str] | None) -> list[tuple[str, object]]:
    requested = list(names) if names else ["numpy", "jax", "torch"]
    resolved: list[tuple[str, object]] = []
    seen: set[str] = set()

    try:
        from tnfr.mathematics import backend as backend_module
    except ImportError:  # pragma: no cover - defensive; module is part of package
        backend_module = None  # type: ignore[assignment]

    if backend_module is not None:
        alias_map = dict(getattr(backend_module, "_BACKEND_ALIASES", {}))
        backend_normalise = getattr(backend_module, "_normalise_name", None)
        if backend_normalise is not None:
            normalise = lambda value: backend_normalise(str(value))
        else:  # pragma: no cover - only when helper is missing
            normalise = lambda value: str(value).strip().lower()
    else:  # pragma: no cover - fallback path when module missing
        alias_map = {}
        normalise = lambda value: str(value).strip().lower()

    for raw_name in requested:
        normalised = normalise(raw_name)
        canonical_requested = normalise(alias_map.get(normalised, normalised))
        backend = get_backend(raw_name)
        canonical_resolved = getattr(backend, "name", canonical_requested)
        if isinstance(canonical_resolved, str):
            canonical_resolved = normalise(canonical_resolved)

        if (
            canonical_resolved == "numpy"
            and canonical_requested != "numpy"
            and canonical_resolved != canonical_requested
        ):
            print(f"[skip] backend '{raw_name}' unavailable (fell back to 'numpy').")
            continue

        if canonical_resolved in seen:
            continue

        resolved.append((canonical_resolved, backend))
        seen.add(canonical_resolved)

    if "numpy" not in seen:
        numpy_backend = get_backend("numpy")
        resolved.insert(0, ("numpy", numpy_backend))
        seen.add("numpy")

    return resolved


def _random_state(rng: np.random.Generator, dim: int) -> np.ndarray:
    vector = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros(dim, dtype=np.complex128)
    return (vector / norm).astype(np.complex128)


def _hermitian_generator(rng: np.random.Generator, dim: int) -> np.ndarray:
    raw = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    hermitian = 0.5 * (raw + raw.conj().T)
    return hermitian.astype(np.complex128)


def _contractive_generator(rng: np.random.Generator, dim: int) -> np.ndarray:
    raw = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    anti_hermitian = raw - raw.conj().T
    generator = -0.3 * np.eye(dim, dtype=np.complex128) + 0.05 * anti_hermitian
    return generator.astype(np.complex128)


def _density_from_state(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conjugate())


def _time_callable(func: Callable[[], None], repeats: int) -> list[float]:
    measurements: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        measurements.append(time.perf_counter() - start)
    return measurements


def _prepare_problems(rng: np.random.Generator, dims: Iterable[int]) -> dict[int, dict[str, np.ndarray]]:
    problems: dict[int, dict[str, np.ndarray]] = {}
    for dim in dims:
        state = _random_state(rng, dim)
        density = _density_from_state(state)
        problems[dim] = {
            "state": state,
            "density": density,
            "unitary_generator": _hermitian_generator(rng, dim),
            "contractive_generator": _contractive_generator(rng, dim * dim),
        }
    return problems


def _benchmark_unitary(
    backend: object,
    generator: np.ndarray,
    state: np.ndarray,
    dim: int,
    *,
    steps: int,
    dt: float,
    repeats: int,
) -> list[float]:
    hilbert = HilbertSpace(dimension=dim)
    engine = MathematicalDynamicsEngine(generator, hilbert, backend=backend, use_scipy=False)
    state_backend = backend.as_array(state)

    def _run() -> None:
        engine.evolve(state_backend, steps=steps, dt=dt)

    return _time_callable(_run, repeats)


def _benchmark_contractive(
    backend: object,
    generator: np.ndarray,
    density: np.ndarray,
    dim: int,
    *,
    steps: int,
    dt: float,
    repeats: int,
) -> list[float]:
    hilbert = HilbertSpace(dimension=dim)
    engine = ContractiveDynamicsEngine(generator, hilbert, backend=backend, use_scipy=False)
    density_backend = backend.as_array(density)

    def _run() -> None:
        engine.evolve(density_backend, steps=steps, dt=dt)

    return _time_callable(_run, repeats)


def _format_seconds(value: float) -> str:
    return f"{value * 1000:.3f} ms"


def _print_table(title: str, dims: Iterable[int], backends: list[str], table: dict[str, dict[int, float]], *, suffix: str = "s") -> None:
    header = "| dim | " + " | ".join(backends) + " |"
    separator = "| --- | " + " | ".join("---" for _ in backends) + " |"
    print(f"\n{title}")
    print(header)
    print(separator)
    for dim in dims:
        cells = [f"{dim}"]
        for backend in backends:
            value = table.get(backend, {}).get(dim)
            if value is None or not np.isfinite(value):
                cells.append("—")
            else:
                cells.append(f"{value:.3f} {suffix}" if suffix == "x" else _format_seconds(value))
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    dims = tuple(sorted(args.sizes))
    backends = _resolve_backends(args.backends)
    if not backends:
        raise SystemExit("No numerical backend available for benchmarking.")

    problems = _prepare_problems(rng, dims)
    results: dict[str, dict[str, dict[int, dict[str, list[float] | float]]]] = {
        "unitary": {},
        "contractive": {},
    }

    mean_unitary: dict[str, dict[int, float]] = {}
    mean_contractive: dict[str, dict[int, float]] = {}

    for name, backend in backends:
        results["unitary"][name] = {}
        results["contractive"][name] = {}
        mean_unitary[name] = {}
        mean_contractive[name] = {}
        for dim, payload in problems.items():
            unitary_timings = _benchmark_unitary(
                backend,
                payload["unitary_generator"],
                payload["state"],
                dim,
                steps=args.steps,
                dt=args.dt,
                repeats=args.repeats,
            )
            contractive_timings = _benchmark_contractive(
                backend,
                payload["contractive_generator"],
                payload["density"],
                dim,
                steps=args.steps,
                dt=args.dt,
                repeats=args.repeats,
            )
            unitary_mean = float(np.mean(unitary_timings))
            contractive_mean = float(np.mean(contractive_timings))
            results["unitary"][name][dim] = {
                "timings": unitary_timings,
                "mean_seconds": unitary_mean,
            }
            results["contractive"][name][dim] = {
                "timings": contractive_timings,
                "mean_seconds": contractive_mean,
            }
            mean_unitary[name][dim] = unitary_mean
            mean_contractive[name][dim] = contractive_mean

    baseline_unitary = mean_unitary.get("numpy", {})
    baseline_contractive = mean_contractive.get("numpy", {})
    speedup_unitary: dict[str, dict[int, float]] = {}
    speedup_contractive: dict[str, dict[int, float]] = {}
    for name, backend_means in mean_unitary.items():
        speedup_unitary[name] = {}
        for dim, timing in backend_means.items():
            base = baseline_unitary.get(dim)
            speedup_unitary[name][dim] = (
                float(base / timing) if base is not None and timing > 0 else float("nan")
            )
    for name, backend_means in mean_contractive.items():
        speedup_contractive[name] = {}
        for dim, timing in backend_means.items():
            base = baseline_contractive.get(dim)
            speedup_contractive[name][dim] = (
                float(base / timing) if base is not None and timing > 0 else float("nan")
            )

    metadata = {
        "seed": args.seed,
        "steps": args.steps,
        "repeats": args.repeats,
        "dt": args.dt,
        "sizes": dims,
    }

    if args.output:
        payload = {
            "metadata": metadata,
            "results": results,
            "speedups": {
                "unitary": speedup_unitary,
                "contractive": speedup_contractive,
            },
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nSaved results to {args.output}")

    backend_labels = [name for name, _ in backends]
    _print_table("Unitary evolution (mean time per run)", dims, backend_labels, mean_unitary)
    _print_table("Contractive evolution (mean time per run)", dims, backend_labels, mean_contractive)
    _print_table("Unitary speed-up vs NumPy", dims, backend_labels, speedup_unitary, suffix="x")
    _print_table("Contractive speed-up vs NumPy", dims, backend_labels, speedup_contractive, suffix="x")


if __name__ == "__main__":
    main()

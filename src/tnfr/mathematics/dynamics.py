"""Spectral dynamics helpers driven by ΔNFR generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Sequence

from .._exact_time import finite_represented_real
from ..config.parsing import parse_bool
from ._complex_arrays import (
    backend_complex_array,
    finite_complex_norm,
    nonnegative_tolerance,
    normalized_complex_vector,
)
from ._integer_admission import _integer_argument
from .backend import MathematicsBackend, ensure_array, ensure_numpy, get_backend
from .spaces import HilbertSpace
from .unified_numerical import TNFRValueError, np

try:  # pragma: no cover - optional SciPy dependency
    from scipy.linalg import expm as _scipy_expm  # type: ignore
except Exception:  # pragma: no cover - SciPy not installed
    _scipy_expm = None

__all__ = ["MathematicalDynamicsEngine", "ContractiveDynamicsEngine"]


def _has_backend_matrix_exp(backend: MathematicsBackend) -> bool:
    """Return ``True`` when ``backend`` exposes a usable ``matrix_exp``."""

    matrix_exp = getattr(backend, "matrix_exp", None)
    if not callable(matrix_exp):
        return False

    try:
        probe = ensure_array([[0.0]], dtype=np.complex128, backend=backend)
        matrix_exp(probe)
    except (AttributeError, NotImplementedError):
        return False
    except Exception:
        # Older backends may surface missing implementations as runtime errors;
        # treat them as signals to fall back to SciPy when available.
        return False
    return True


def _resolve_use_scipy(backend: MathematicsBackend, requested: bool | None) -> bool:
    """Select an exponential route without caching mutable backend capabilities."""
    if requested is not None:
        requested = parse_bool(requested)
        if requested and _scipy_expm is None:
            raise RuntimeError("SciPy expm requested but SciPy is not available.")
        return bool(requested and _scipy_expm is not None)
    if _has_backend_matrix_exp(backend):
        return False
    if _scipy_expm is not None:
        return True
    raise RuntimeError(
        "Backend lacks matrix_exp and SciPy is unavailable for fallback."
    )


def _evolution_time(value: Any, *, backend: MathematicsBackend) -> Any:
    """Admit signed real time while retaining native scalar gradient paths.

    Symbolic JAX times have only a scalar real-dtype check; they do not provide
    a concrete finiteness certificate until evaluated outside the trace.
    """
    torch = getattr(backend, "_torch", None)
    jax = getattr(backend, "_jax", None)
    is_torch = torch is not None and isinstance(value, torch.Tensor)
    is_jax = jax is not None and isinstance(value, (jax.Array, jax.core.Tracer))
    if is_torch or is_jax or isinstance(value, np.ndarray):
        if value.shape != ():
            raise TNFRValueError("dt must be a finite representable real scalar")
        if is_torch:
            if value.is_complex() or value.dtype == torch.bool:
                raise TNFRValueError("dt must be a finite representable real scalar")
            observed = (
                value.to(dtype=torch.float64)
                if value.dtype == torch.bfloat16
                else value
            )
        else:
            if np.dtype(value.dtype).kind not in "iuf":
                raise TNFRValueError("dt must be a finite representable real scalar")
            if is_jax and isinstance(value, jax.core.Tracer):
                return value
            observed = value
        observed = np.asarray(ensure_numpy(observed, backend=backend))[()]
        try:
            admitted = finite_represented_real(observed, "dt")[0]
        except (TypeError, ValueError, OverflowError) as exc:
            raise TNFRValueError(
                "dt must be a finite representable real scalar"
            ) from exc
        return value if is_torch or is_jax else admitted
    try:
        return finite_represented_real(value, "dt")[0]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError("dt must be a finite representable real scalar") from exc


def _evolution_steps(value: int) -> int:
    """Admit a nonnegative integer trajectory length."""
    try:
        steps = _integer_argument(value, "steps")
    except TypeError as exc:
        raise TNFRValueError("steps must be a non-negative integer") from exc
    if steps < 0:
        raise TNFRValueError("steps must be a non-negative integer")
    return steps


def _as_matrix(
    matrix: Sequence[Sequence[complex]] | np.ndarray | Any,
    *,
    backend: MathematicsBackend,
) -> Any:
    arr = backend_complex_array(matrix, backend=backend, label="Generator", owned=True)
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise TNFRValueError(
            "Generator matrix must be square.",
            context={"shape": shape},
            suggestion="Provide a square matrix.",
        )
    return arr


def _is_hermitian(
    matrix: Any, *, atol: float = 1e-9, backend: MathematicsBackend
) -> bool:
    matrix_np = ensure_numpy(matrix, backend=backend)
    return bool(np.allclose(matrix_np, matrix_np.conj().T, atol=atol))


def _vectorize_density(matrix: Any, *, backend: MathematicsBackend) -> Any:
    arr = ensure_array(matrix, dtype=np.complex128, backend=backend)
    return arr.transpose(1, 0).reshape((-1,))


def _devectorize_density(vector: Any, dim: int, *, backend: MathematicsBackend) -> Any:
    arr = ensure_array(vector, dtype=np.complex128, backend=backend)
    return arr.reshape((dim, dim)).transpose(1, 0)


class TraceValue(NamedTuple):
    """Container for trace evaluations in both backend and NumPy space."""

    backend: Any
    numpy: complex | None


def _trace(matrix: Any, *, backend: MathematicsBackend) -> TraceValue:
    traced_backend = backend.einsum("ii->", matrix)
    try:
        traced_numpy = complex(
            np.asarray(ensure_numpy(traced_backend, backend=backend))
        )
    except Exception:
        # Fallback for backends where conversion fails (e.g. JAX tracing)
        traced_numpy = None
    return TraceValue(traced_backend, traced_numpy)


@dataclass(slots=True)
class MathematicalDynamicsEngine:
    """Unitary evolution generated by Hermitian ΔNFR operators.

    The engine accepts inputs expressed as backend-native tensors (NumPy,
    :mod:`jax`, :mod:`torch`).  When the configured backend supports automatic
    differentiation the evolution map ``exp(-i·Δ·dt)`` remains differentiable
    because native propagators are now preferred.  Passing ``use_scipy=True``
    explicitly opts into SciPy's exponential; we only fall back automatically
    when the backend lacks a ``matrix_exp`` implementation.
    """

    hilbert_space: HilbertSpace
    atol: float = 1e-9
    _use_scipy: bool = False
    backend: MathematicsBackend = field(init=False, repr=False)
    _generator_backend: Any = field(init=False, repr=False)
    _numpy_generator: np.ndarray = field(init=False, repr=False)

    def __init__(
        self,
        generator: Sequence[Sequence[complex]] | np.ndarray | Any,
        hilbert_space: HilbertSpace,
        *,
        atol: float = 1e-9,
        use_scipy: bool | None = None,
        backend: MathematicsBackend | None = None,
    ) -> None:
        atol = nonnegative_tolerance(atol)
        resolved_backend = backend or get_backend()
        matrix = _as_matrix(generator, backend=resolved_backend)
        matrix_np = ensure_numpy(matrix, backend=resolved_backend)
        if matrix_np.shape != (hilbert_space.dimension, hilbert_space.dimension):
            raise TNFRValueError(
                "Generator dimension must match the Hilbert space.",
                context={
                    "generator_shape": matrix_np.shape,
                    "hilbert_dimension": hilbert_space.dimension,
                },
                suggestion="Ensure generator matches Hilbert space dimension.",
            )
        if not _is_hermitian(matrix, atol=atol, backend=resolved_backend):
            raise TNFRValueError(
                "Dynamics generator must be Hermitian.",
                context={"atol": atol},
                suggestion="Ensure generator is Hermitian.",
            )
        self.backend = resolved_backend
        self._generator_backend = matrix
        self._numpy_generator = np.array(matrix_np, copy=True)
        self.hilbert_space = hilbert_space
        self.atol = float(atol)
        self._use_scipy = _resolve_use_scipy(self.backend, use_scipy)

    @property
    def generator(self) -> np.ndarray:
        """Return a detached copy of the owned, admitted generator."""
        return self._numpy_generator.copy()

    def _unitary_backend(self, dt: float) -> Any:
        argument = backend_complex_array(
            -1j * (dt * self._generator_backend),
            backend=self.backend,
            label="Unitary exponential argument",
        )
        if self._use_scipy and _scipy_expm is not None:
            return ensure_array(
                _scipy_expm(ensure_numpy(argument, backend=self.backend)),
                backend=self.backend,
            )
        return self.backend.matrix_exp(argument)

    def step(
        self,
        state: Sequence[complex] | np.ndarray | Any,
        *,
        dt: float = 1.0,
        normalize: bool = True,
    ) -> Any:
        """Evolve by signed real ``dt`` using the unitary ``exp(-i·Δ·dt)``."""

        evolved, _ = self._step_with_propagator(state, dt=dt, normalize=normalize)
        return evolved

    def _step_with_propagator(
        self,
        state: Any,
        *,
        dt: Any,
        normalize: bool,
        propagator: Any = None,
    ) -> tuple[Any, Any]:
        """Apply the usual step checks, optionally reusing a local propagator."""
        dt = _evolution_time(dt, backend=self.backend)
        vector = backend_complex_array(state, backend=self.backend, label="State")
        if vector.shape != (self.hilbert_space.dimension,):
            raise TNFRValueError(
                "State vector dimension mismatch.",
                context={
                    "vector_shape": vector.shape,
                    "expected_dimension": self.hilbert_space.dimension,
                },
                suggestion="Ensure state vector matches Hilbert space dimension.",
            )
        if propagator is None:
            propagator = self._unitary_backend(dt)
        evolved = self.backend.matmul(propagator, vector)
        if normalize:
            evolved = normalized_complex_vector(
                evolved, backend=self.backend, atol=self.atol, label="state vector"
            )
        return (
            backend_complex_array(evolved, backend=self.backend, label="Evolved state"),
            propagator,
        )

    def evolve(
        self,
        state: Sequence[complex] | np.ndarray | Any,
        *,
        steps: int,
        dt: float = 1.0,
        normalize: bool = True,
    ) -> Any:
        """Return ``steps + 1`` states, sharing the fixed propagator within this call.

        Subclasses and replaced public steps retain their per-step dispatch.
        No propagator is retained between calls or evaluated for zero steps.
        """

        steps = _evolution_steps(steps)
        dt = _evolution_time(dt, backend=self.backend)
        current = backend_complex_array(state, backend=self.backend, label="State")
        if current.shape != (self.hilbert_space.dimension,):
            raise TNFRValueError(
                "State dimension mismatch.",
                context={
                    "expected_dimension": self.hilbert_space.dimension,
                    "received_shape": current.shape,
                },
                suggestion="Ensure state vector matches Hilbert space dimension.",
            )
        trajectory: list[Any] = [current]
        reuse = (
            type(self) is MathematicalDynamicsEngine
            and getattr(self.step, "__func__", None) is _ORIGINAL_UNITARY_STEP
        )
        propagator = None
        for _ in range(steps):
            if reuse:
                current, propagator = self._step_with_propagator(
                    current, dt=dt, normalize=normalize, propagator=propagator
                )
            else:
                current = self.step(current, dt=dt, normalize=normalize)
            trajectory.append(current)
        return self.backend.stack(trajectory, axis=0)


@dataclass(slots=True)
class ContractiveDynamicsEngine:
    """Density evolution with an optional spectral non-expansion check.

    Non-positive real generator eigenvalues alone do not establish a GKSL law
    or contraction in every norm. Forward dissipative guarantees require the
    corresponding generator hypotheses and nonnegative time.

    Backend-native tensors are accepted for all density operators.  When the
    chosen backend supports automatic differentiation we keep gradients intact
    by default because native semigroup propagators are preferred.  Requesting
    ``use_scipy=True`` still falls back to SciPy's :func:`scipy.linalg.expm`,
    primarily for generators missing backend support.
    """

    hilbert_space: HilbertSpace
    atol: float = 1e-9
    _use_scipy: bool = False
    backend: MathematicsBackend = field(init=False, repr=False)
    _generator_backend: Any = field(init=False, repr=False)
    _numpy_generator: np.ndarray = field(init=False, repr=False)
    _identity_backend: Any = field(init=False, repr=False)
    _last_contractivity_gap: float = field(init=False, repr=False)

    def __init__(
        self,
        generator: Sequence[Sequence[complex]] | np.ndarray | Any,
        hilbert_space: HilbertSpace,
        *,
        atol: float = 1e-9,
        ensure_contractive: bool = True,
        use_scipy: bool | None = None,
        backend: MathematicsBackend | None = None,
    ) -> None:
        atol = nonnegative_tolerance(atol)
        ensure_contractive = parse_bool(ensure_contractive)
        resolved_backend = backend or get_backend()
        matrix = _as_matrix(generator, backend=resolved_backend)
        matrix_np = ensure_numpy(matrix, backend=resolved_backend)
        expected = hilbert_space.dimension * hilbert_space.dimension
        if matrix_np.shape != (expected, expected):
            raise TNFRValueError(
                "Generator must act on vectorised density operators.",
                context={
                    "expected_dimension": (expected, expected),
                    "received_shape": matrix_np.shape,
                },
                suggestion="Ensure generator dimension matches vectorized Hilbert space.",
            )
        self.backend = resolved_backend
        self._generator_backend = matrix
        self._numpy_generator = np.array(matrix_np, dtype=np.complex128, copy=True)
        self.hilbert_space = hilbert_space
        self.atol = float(atol)
        self._use_scipy = _resolve_use_scipy(self.backend, use_scipy)

        self._identity_backend = ensure_array(
            np.eye(hilbert_space.dimension, dtype=np.complex128),
            backend=self.backend,
        )
        self._last_contractivity_gap = float("nan")
        if ensure_contractive:
            eigenvalues_backend, _ = self.backend.eig(self._generator_backend)
            eigenvalues = ensure_numpy(
                backend_complex_array(
                    eigenvalues_backend,
                    backend=self.backend,
                    label="Generator spectrum",
                ),
                backend=self.backend,
            )
            if np.max(eigenvalues.real) > self.atol:
                raise TNFRValueError(
                    "ΔNFR generator is not contractive: positive real eigenvalues detected.",
                    context={"max_real_eigenvalue": np.max(eigenvalues.real)},
                    suggestion="Ensure generator is dissipative.",
                )

    @property
    def generator(self) -> np.ndarray:
        """Return a detached copy of the owned, admitted generator."""
        return self._numpy_generator.copy()

    def _propagator_backend(self, dt: float) -> Any:
        argument = backend_complex_array(
            dt * self._generator_backend,
            backend=self.backend,
            label="Semigroup exponential argument",
        )
        if self._use_scipy and _scipy_expm is not None:
            return ensure_array(
                _scipy_expm(ensure_numpy(argument, backend=self.backend)),
                backend=self.backend,
            )
        return self.backend.matrix_exp(argument)

    def frobenius_norm(
        self,
        density: Sequence[Sequence[complex]] | np.ndarray | Any,
        *,
        center: bool = False,
    ) -> float:
        """Return the Frobenius norm associated with the Hilbert space."""

        matrix = backend_complex_array(density, backend=self.backend, label="Density")
        if matrix.shape != (self.hilbert_space.dimension, self.hilbert_space.dimension):
            raise TNFRValueError(
                "Density operator dimension mismatch.",
                context={
                    "expected_dimension": (
                        self.hilbert_space.dimension,
                        self.hilbert_space.dimension,
                    ),
                    "received_shape": matrix.shape,
                },
                suggestion="Ensure density operator matches Hilbert space dimension.",
            )
        if center:
            trace_value = _trace(matrix, backend=self.backend)
            trace_backend = trace_value.backend / self.hilbert_space.dimension
            matrix = matrix - trace_backend * self._identity_backend
        return finite_complex_norm(matrix, backend=self.backend, label="Density")

    @property
    def last_contractivity_gap(self) -> float:
        """Return the latest monitored contractivity gap (NaN if unavailable)."""

        return float(self._last_contractivity_gap)

    def step(
        self,
        density: Sequence[Sequence[complex]] | np.ndarray | Any,
        *,
        dt: float = 1.0,
        normalize_trace: bool = True,
        enforce_contractivity: bool = True,
        raise_on_violation: bool = False,
        symmetrize: bool = True,
    ) -> Any:
        """Advance by signed real ``dt`` with trace and contractivity monitoring.

        Forward-semigroup guarantees require nonnegative time and the relevant
        generator hypotheses; admitting negative time does not extend them.
        """

        evolved, _ = self._step_with_propagator(
            density,
            dt=dt,
            normalize_trace=normalize_trace,
            enforce_contractivity=enforce_contractivity,
            raise_on_violation=raise_on_violation,
            symmetrize=symmetrize,
        )
        return evolved

    def _step_with_propagator(
        self,
        density: Any,
        *,
        dt: Any,
        normalize_trace: bool,
        enforce_contractivity: bool,
        raise_on_violation: bool,
        symmetrize: bool,
        propagator: Any = None,
    ) -> tuple[Any, Any]:
        """Apply the usual checks and monitors with an optional local propagator."""
        dt = _evolution_time(dt, backend=self.backend)
        matrix = backend_complex_array(density, backend=self.backend, label="Density")
        dim = self.hilbert_space.dimension
        if matrix.shape != (dim, dim):
            raise TNFRValueError(
                "Density operator dimension mismatch.",
                context={
                    "expected_dimension": (dim, dim),
                    "received_shape": matrix.shape,
                },
                suggestion="Ensure density operator matches Hilbert space dimension.",
            )

        initial_norm = None
        if enforce_contractivity:
            trace_value = _trace(matrix, backend=self.backend)
            trace_backend = trace_value.backend / dim
            centered = matrix - trace_backend * self._identity_backend
            if trace_value.numpy is not None:
                initial_norm = finite_complex_norm(
                    centered, backend=self.backend, label="Centered density"
                )

        vector = _vectorize_density(matrix, backend=self.backend)
        if propagator is None:
            propagator = self._propagator_backend(dt)
        evolved_vec = self.backend.matmul(propagator, vector)
        evolved = _devectorize_density(evolved_vec, dim, backend=self.backend)

        if symmetrize:
            evolved = 0.5 * (evolved + self.backend.conjugate_transpose(evolved))

        if normalize_trace:
            trace_value = _trace(evolved, backend=self.backend)
            if trace_value.numpy is not None:
                if not np.isfinite(trace_value.numpy):
                    raise TNFRValueError("Trace must be finite before normalization.")
                if np.isclose(trace_value.numpy, 0.0, atol=self.atol):
                    raise TNFRValueError(
                        "Trace collapsed below tolerance during evolution.",
                        context={"trace": trace_value.numpy, "atol": self.atol},
                        suggestion="Check generator properties or initial state.",
                    )
                if not np.isclose(trace_value.numpy, 1.0, atol=10 * self.atol):
                    evolved = evolved / trace_value.backend
            else:
                # Tracing fallback: always normalize
                evolved = evolved / trace_value.backend

        if enforce_contractivity and initial_norm is not None:
            trace_value = _trace(evolved, backend=self.backend)
            trace_backend = trace_value.backend / dim
            centered = evolved - trace_backend * self._identity_backend
            if trace_value.numpy is None:
                self._last_contractivity_gap = float("nan")
            else:
                evolved_norm = finite_complex_norm(
                    centered, backend=self.backend, label="Centered evolved density"
                )
                self._last_contractivity_gap = initial_norm - evolved_norm
                if raise_on_violation and self._last_contractivity_gap < -5 * self.atol:
                    raise TNFRValueError(
                        "Contractivity violated: Frobenius norm increased beyond tolerance.",
                        context={
                            "initial_norm": initial_norm,
                            "evolved_norm": evolved_norm,
                            "gap": self._last_contractivity_gap,
                            "atol": self.atol,
                        },
                        suggestion="Ensure generator is contractive.",
                    )
        else:
            self._last_contractivity_gap = float("nan")

        return (
            backend_complex_array(
                evolved, backend=self.backend, label="Evolved density"
            ),
            propagator,
        )

    def evolve(
        self,
        density: Sequence[Sequence[complex]] | np.ndarray | Any,
        *,
        steps: int,
        dt: float = 1.0,
        normalize_trace: bool = True,
        enforce_contractivity: bool = True,
        raise_on_violation: bool = False,
        symmetrize: bool = True,
    ) -> Any:
        """Return a density trajectory with one fixed propagator per built-in call.

        Each step retains its trace and contractivity controls. Subclasses and
        replaced public steps keep per-step dispatch; zero steps computes no
        propagator, and subsequent calls always build a fresh one.
        """

        steps = _evolution_steps(steps)
        dt = _evolution_time(dt, backend=self.backend)

        current = backend_complex_array(density, backend=self.backend, label="Density")
        dim = self.hilbert_space.dimension
        if current.shape != (dim, dim):
            raise TNFRValueError(
                "Density operator dimension mismatch.",
                context={"expected_shape": (dim, dim), "actual_shape": current.shape},
                suggestion="Ensure density operator matches Hilbert space dimension.",
            )

        trajectory: list[Any] = [current]
        reuse = (
            type(self) is ContractiveDynamicsEngine
            and getattr(self.step, "__func__", None) is _ORIGINAL_DENSITY_STEP
        )
        propagator = None
        for _ in range(steps):
            if reuse:
                current, propagator = self._step_with_propagator(
                    current,
                    dt=dt,
                    normalize_trace=normalize_trace,
                    enforce_contractivity=enforce_contractivity,
                    raise_on_violation=raise_on_violation,
                    symmetrize=symmetrize,
                    propagator=propagator,
                )
            else:
                current = self.step(
                    current,
                    dt=dt,
                    normalize_trace=normalize_trace,
                    enforce_contractivity=enforce_contractivity,
                    raise_on_violation=raise_on_violation,
                    symmetrize=symmetrize,
                )
            trajectory.append(current)
        return self.backend.stack(trajectory, axis=0)


# Keep class-level replacements of the public step on the dispatch path too.
_ORIGINAL_UNITARY_STEP = MathematicalDynamicsEngine.step
_ORIGINAL_DENSITY_STEP = ContractiveDynamicsEngine.step

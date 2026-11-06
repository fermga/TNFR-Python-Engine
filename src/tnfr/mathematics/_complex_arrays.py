"""Primitive complex-array admission and range-safe native normalization.

Host snapshots validate concrete tensors without replacing their differentiable
native values. JAX tracers retain their numeric dtype/shape domain; finiteness
cannot be observed from a symbolic trace and is not certified by this adapter.
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any

from .._exact_time import finite_represented_real
from ..errors import TNFRValueError
from ._complex_admission import finite_represented_complex
from .unified_numerical import np

if TYPE_CHECKING:
    from .backend import MathematicsBackend


def nonnegative_tolerance(value: Any, label: str = "atol") -> float:
    """Admit a finite represented nonnegative numerical tolerance."""
    try:
        result = finite_represented_real(value, label)[0]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(
            f"{label} must be a finite nonnegative real scalar"
        ) from exc
    if result < 0.0:
        raise TNFRValueError(f"{label} must be a finite nonnegative real scalar")
    return result


def numpy_complex_array(value: Any, *, label: str) -> np.ndarray:
    """Admit original components before materializing a complex128 array."""
    if isinstance(value, np.ndarray) and value.dtype.kind in "iufc":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            array = np.asarray(value, dtype=np.complex128)
        if not np.all(np.isfinite(array)):
            raise TNFRValueError(f"{label} must contain finite complex components")
        if np.any((value.real != 0) & (array.real == 0)) or np.any(
            (value.imag != 0) & (array.imag == 0)
        ):
            raise TNFRValueError(f"{label} contains nonzero components that underflow")
        return array
    try:
        raw = np.asarray(value, dtype=object)
        admitted = [finite_represented_complex(item, label) for item in raw.flat]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(
            f"{label} must contain finite representable real or complex scalars"
        ) from exc
    return np.asarray(admitted, dtype=np.complex128).reshape(raw.shape)


def finite_complex_norm(
    value: Any,
    *,
    backend: MathematicsBackend | None = None,
    label: str = "State",
) -> float:
    """Observe the finite Euclidean norm of an admitted concrete numeric array.

    Real and imaginary channels enter ``hypot`` separately, avoiding squared
    scale overflow or underflow. This detached observation cannot consume a
    symbolic tracer and does not replace primitive admission by its caller.
    """
    observed = np.asarray(value if backend is None else backend.to_numpy(value))
    if not np.all(np.isfinite(observed)):
        raise TNFRValueError(f"{label} norm requires finite vector components.")
    norm = math.hypot(*observed.real.flat, *observed.imag.flat)
    if not math.isfinite(norm):
        raise TNFRValueError(f"{label} norm is not representable as a finite float.")
    return norm


def _jax_tracer(value: Any, backend: MathematicsBackend) -> bool:
    jax = getattr(backend, "_jax", None)
    return jax is not None and isinstance(value, jax.core.Tracer)


def backend_complex_array(
    value: Any, *, backend: MathematicsBackend, label: str, owned: bool = False
) -> Any:
    """Admit source arrays while preserving native automatic differentiation."""
    torch = getattr(backend, "_torch", None)
    jax = getattr(backend, "_jax", None)
    native = (torch is not None and isinstance(value, torch.Tensor)) or (
        jax is not None and isinstance(value, (jax.Array, jax.core.Tracer))
    )
    if _jax_tracer(value, backend):
        if np.dtype(value.dtype).kind not in "iufc":
            raise TNFRValueError(f"{label} must contain real or complex scalars")
        array = backend.as_array(value, dtype=np.complex128)
    else:
        if native:
            # NumPy cannot export bfloat16 directly; float64 preserves its
            # complete represented range and lets the same reader validate it.
            observed = value
            if torch is not None and value.dtype == torch.bfloat16:
                observed = value.to(dtype=torch.float64)
            source = numpy_complex_array(backend.to_numpy(observed), label=label)
        else:
            source = numpy_complex_array(value, label=label)
            if owned:
                # Some CPU backends share NumPy storage during transfer; take
                # ownership before that transfer, rather than relying only on
                # a backend copy that may preserve or asynchronously read it.
                source = np.array(source, copy=True)
        array = backend.as_array(value if native else source, dtype=np.complex128)
        observed = numpy_complex_array(backend.to_numpy(array), label=label)
        if np.any((source.real != 0) & (observed.real == 0)) or np.any(
            (source.imag != 0) & (observed.imag == 0)
        ):
            raise TNFRValueError(f"{label} loses nonzero components in its backend")
    if owned:
        # clone/copy retain a native gradient path and detach storage ownership.
        if jax is not None:
            if "may_alias" in inspect.signature(jax.device_put).parameters:
                array = jax.device_put(array, may_alias=False)
            else:  # Older supported JAX releases lack the explicit alias flag.
                array = array.copy()
            if not _jax_tracer(array, backend):
                # A caller may mutate a NumPy buffer underlying its JAX input
                # immediately after return. Finish the copy before that point.
                array.block_until_ready()
        else:
            array = array.clone() if hasattr(array, "clone") else array.copy()
    return array


def normalized_complex_vector(
    value: Any,
    *,
    backend: MathematicsBackend | None = None,
    atol: float,
    label: str,
) -> Any:
    """Normalize via bounded real/imaginary channels without squaring scale."""
    tolerance = nonnegative_tolerance(atol)
    if backend is not None and _jax_tracer(value, backend):
        jnp = getattr(backend, "_jnp")
        scale = jnp.maximum(jnp.max(jnp.abs(value.real)), jnp.max(jnp.abs(value.imag)))
        scaled = value.real / scale + 1j * (value.imag / scale)
        return scaled / backend.norm(scaled)

    observed = np.asarray(value if backend is None else backend.to_numpy(value))
    if not np.all(np.isfinite(observed)):
        raise TNFRValueError(f"Cannot normalize nonfinite {label}")
    scale = float(
        max(
            np.max(np.abs(observed.real), initial=0.0),
            np.max(np.abs(observed.imag), initial=0.0),
        )
    )
    if scale == 0.0:
        raise TNFRValueError(f"Cannot normalise a null {label}")
    scaled = value.real / scale + 1j * (value.imag / scale)
    norm = np.linalg.norm(scaled) if backend is None else backend.norm(scaled)
    observed_norm = float(norm if backend is None else backend.to_numpy(norm))
    if not math.isfinite(observed_norm) or observed_norm == 0.0:
        raise TNFRValueError(f"Cannot normalize {label} with a nonfinite or zero norm")
    if scale <= tolerance / observed_norm:
        raise TNFRValueError(f"Cannot normalise a null {label}")
    return scaled / norm

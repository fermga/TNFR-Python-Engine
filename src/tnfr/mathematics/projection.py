"""Projection helpers constructing TNFR state vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .._spectral_expectation import finite_spectral_real
from ..errors import TNFRValueError
from ._complex_arrays import nonnegative_tolerance, normalized_complex_vector
from .unified_numerical import np

if TYPE_CHECKING:  # pragma: no cover - typing hook when numpy.typing is available
    import numpy as _np_typing
    import numpy.typing as npt

    ComplexVector = npt.NDArray[
        _np_typing.complexfloating[_np_typing.float64, _np_typing.float64]
    ]
else:  # pragma: no cover - runtime fallback without numpy.typing
    ComplexVector = np.ndarray  # type: ignore[assignment]

__all__ = ["StateProjector", "BasicStateProjector"]


@runtime_checkable
class StateProjector(Protocol):
    """Protocol describing state projection callables.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    state projector implementations conform to the expected callable interface.
    """

    def __call__(
        self,
        epi: float,
        nu_f: float,
        theta: float,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> ComplexVector:
        """Return a normalised TNFR state vector for the provided parameters."""


@dataclass(slots=True)
class BasicStateProjector:
    """Default auxiliary projector building finite complex state vectors.

    The projector maps the structural scalars of a node—its EPI magnitude,
    structural frequency ``νf`` and phase ``θ``—onto the canonical Hilbert
    basis.  The resulting vector encodes a coherent amplitude envelope derived
    from the structural intensity while the complex exponential captures the
    phase progression across the local modes.  Optional stochastic excitation is
    injected via a :class:`numpy.random.Generator` to model controlled
    dissonance while preserving determinism when a seed is provided.
    """

    dtype: np.dtype[np.complexfloating[np.float64, np.float64]] = np.dtype(
        np.complex128
    )
    atol: float = 1e-12

    def __call__(
        self,
        epi: float,
        nu_f: float,
        theta: float,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> ComplexVector:
        # Keep the types owner lazy while the mathematics package initializes.
        from ..types import require_finite_real_scalar_epi
        from .spaces import HilbertSpace

        space = HilbertSpace(dim, dtype=self.dtype)
        if space.dtype.kind != "c":
            raise TNFRValueError("projected state dtype must be complex")
        dim = space.dimension
        tolerance = nonnegative_tolerance(self.atol)
        epi = require_finite_real_scalar_epi(epi)
        nu_f = finite_spectral_real(nu_f, label="capacity nu_f")
        if nu_f < 0:
            raise TNFRValueError("capacity nu_f must be nonnegative")
        theta = finite_spectral_real(theta, label="phase")

        indices = np.arange(1, dim + 1, dtype=float)
        phase_progression = theta + (nu_f + 1.0) * indices / max(dim, 1)
        envelope = np.abs(epi) + 0.5 * indices / dim + 1.0
        base_vector = envelope * np.exp(1j * phase_progression)

        if rng is not None:
            noise_scale = (np.abs(epi) + np.abs(nu_f) + 1.0) * 0.05
            real_noise = rng.standard_normal(dim)
            imag_noise = rng.standard_normal(dim)
            stochastic = noise_scale * (real_noise + 1j * imag_noise)
            base_vector = base_vector + stochastic

        normalised = normalized_complex_vector(
            base_vector, atol=tolerance, label="projected state vector"
        )
        return space._as_vector(normalised)

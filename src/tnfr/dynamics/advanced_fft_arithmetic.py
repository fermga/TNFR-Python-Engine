"""Graph-spectral analysis and filtering for TNFR signals.

This compatibility module retains its historical ``FFT`` names, but its core
transform is a Graph Fourier Transform (GFT) obtained from a dense Laplacian
eigenbasis.  Generic graph diagonalization is normally cubic and each dense
transform is quadratic; it is distinct from a one-dimensional ``numpy.fft``.

Modewise multiplication of two GFT coefficient vectors defines a
basis-dependent graph convolution.  It is not the pointwise physical-space
product ``νf * ΔNFR`` in the nodal equation.  Canonical nodal evolution must
form that product at nodes before any optional change of basis.

Exposed operations include basis-dependent graph convolution, Laplacian-mode
amplitude ranking, diagonal spectral filtering, and index-aligned cross-spectrum
diagnostics.  These are graph-signal read-outs; they do not apply canonical
operators or advance the TNFR nodal state.

Cached bases can avoid repeated diagonalization; they do not change the
complexity of a new dense GFT.  GPU backends may accelerate matrix operations.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_VF
from ..errors import TNFRValueError
from ..errors.contextual import NetworkConfigError, TNFRUserError
from ..mathematics.unified_numerical import np
from ..types import real_scalar_epi

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import mathematical backends
try:
    from ..mathematics.backend import get_backend

    HAS_MATH_BACKENDS = True
except ImportError:
    HAS_MATH_BACKENDS = False

# Operational engine-tuning knobs (not TNFR physics) → tnfr.constants.operational
from ..constants.operational import (
    FFT_ARITHMETIC_IMPORTANCE_CANONICAL,
    FFT_BANDWIDTH_CANONICAL,
    FFT_COHERENT_THRESHOLD_CANONICAL,
    FFT_HIGH_CUTOFF_CANONICAL,
    FFT_LOW_CUTOFF_CANONICAL,
)

# Import spectral analysis
try:
    from ..mathematics.spectral import get_laplacian_spectrum, gft, igft

    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# Import unified cache
try:
    from .multi_modal_cache import CacheEntryType, get_unified_cache

    HAS_UNIFIED_CACHE = True
except ImportError:
    HAS_UNIFIED_CACHE = False

# FFT cache coordinator
try:
    from .fft_cache_coordinator import FFTCacheCoordinator, get_fft_cache_coordinator

    HAS_FFT_CACHE = True
except ImportError:
    HAS_FFT_CACHE = False
    FFTCacheCoordinator = None  # type: ignore

try:
    from .fft_backend import FFTBackendCapabilities
except ImportError:  # pragma: no cover - circular import guard during bootstrap
    FFTBackendCapabilities = None  # type: ignore


class SpectralOperation(Enum):
    """Types of spectral operations."""

    CONVOLUTION = "convolution"  # Basis-dependent graph convolution
    CORRELATION = "correlation"  # Reserved compatibility label
    FILTERING = "filtering"  # Spectral filtering
    WINDOWING = "windowing"  # Reserved compatibility label
    HARMONIC_ANALYSIS = "harmonic_analysis"  # Laplacian-mode ranking
    PHASE_LOCKING = "phase_locking"  # Reserved compatibility label
    COHERENCE_ANALYSIS = "coherence_analysis"  # Snapshot cross-power


@dataclass
class SpectralState:
    """State in spectral domain."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_coeffs: np.ndarray
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    coherence_length: float = 0.0
    spectral_parameter_kind: str = "laplacian_eigenvalue"
    coherence_length_semantics: str = "inverse_root_spectral_centroid_heuristic"
    dominant_modes: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class FFTArithmeticResult:
    """Result container; ``fft_operations`` counts dense GFT/IGFT transforms."""

    operation: SpectralOperation
    input_shape: tuple[int, ...]
    output_data: Any
    spectral_state: SpectralState | None = None
    execution_time: float = 0.0
    fft_operations: int = 0
    cache_hits: int = 0
    backend_used: str = "numpy"
    accuracy_metrics: dict[str, float] = field(default_factory=dict)


class TNFRAdvancedFFTEngine:
    """
    Advanced FFT arithmetic engine for TNFR computations.

    The historical class name is retained for compatibility.  Operations use
    dense graph eigenbases and do not claim FFT complexity or nodal-equation
    equivalence.
    """

    def __init__(
        self,
        default_backend: str = "numpy",
        precision: str = "float64",
        cache_coordinator: FFTCacheCoordinator | None = None,
    ):
        self.default_backend = default_backend
        self.precision = precision
        self.cache_coordinator = (
            cache_coordinator
            if cache_coordinator is not None
            else (get_fft_cache_coordinator() if HAS_FFT_CACHE else None)
        )
        self.backend_name = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # Initialize mathematical backend
        if HAS_MATH_BACKENDS:
            self.backend = get_backend(default_backend)
        else:
            self.backend = None

        # Performance tracking
        self.total_operations = 0
        self.total_fft_ops = 0
        self.cache_hits = 0

        # Spectral cache for eigendecompositions (fallback when coordinator unavailable)
        self._spectral_cache = {} if self.cache_coordinator is None else None

    @staticmethod
    def _graph_epi_signal(G: Any) -> np.ndarray:
        """Return the live signed scalar-EPI chart in graph iteration order."""

        values: list[float] = []
        for node in G.nodes():
            raw = get_attr(
                G.nodes[node],
                ALIAS_EPI,
                0.0,
                strict=True,
                conv=lambda value: value,
            )
            if isinstance(raw, bool):
                scalar = None
            else:
                try:
                    scalar = real_scalar_epi(raw)
                except (OverflowError, TypeError, ValueError):
                    scalar = None
            if scalar is None or not np.isfinite(scalar):
                raise TNFRValueError(
                    "Graph spectral analysis requires finite uniform-real EPI values",
                    context={"node": node, "value": repr(raw)},
                    suggestion=(
                        "Use a raw real scalar or a uniform-real BEPI embedding for "
                        "each graph node."
                    ),
                )
            values.append(float(scalar))
        return np.asarray(values, dtype=float)

    @staticmethod
    def _graph_frequency_signal(G: Any) -> np.ndarray:
        """Return finite nonnegative structural frequencies in node order."""

        values: list[float] = []
        for node in G.nodes():
            raw = get_attr(
                G.nodes[node],
                ALIAS_VF,
                1.0,
                strict=True,
                conv=lambda value: value,
            )
            try:
                scalar = float(raw) if not isinstance(raw, bool) else float("nan")
            except (OverflowError, TypeError, ValueError):
                scalar = float("nan")
            if not np.isfinite(scalar) or scalar < 0.0:
                raise TNFRValueError(
                    "Graph spectral analysis requires finite nonnegative nu_f values",
                    context={"node": node, "value": repr(raw)},
                    suggestion="Store structural frequency as a nonnegative Hz_str scalar.",
                )
            values.append(scalar)
        return np.asarray(values, dtype=float)

    @staticmethod
    def _finite_signal(
        signal: Any, *, label: str, expected_shape: tuple[int, ...]
    ) -> np.ndarray:
        """Validate an explicitly supplied scalar graph signal."""

        array = np.asarray(signal)
        if array.shape != expected_shape:
            raise TNFRValueError(
                "Graph spectral convolution requires equally sized node vectors",
                context={
                    f"{label}_shape": array.shape,
                    "expected_shape": expected_shape,
                },
                suggestion="Pass one scalar value per graph node in graph iteration order.",
            )
        if array.dtype.kind not in "iufc" or not np.all(np.isfinite(array)):
            raise TNFRValueError(
                f"{label} must contain only finite numeric scalars",
                context={"dtype": str(array.dtype), "shape": array.shape},
                suggestion="Remove booleans, strings, NaNs, and infinities.",
            )
        return array

    def get_capabilities(self) -> FFTBackendCapabilities:
        """Return capability metadata for planning purposes."""

        if FFTBackendCapabilities is None:  # pragma: no cover - during import cycle
            raise TNFRUserError(
                message="FFTBackendCapabilities unavailable",
                suggestion="Check import cycles or installation",
                context={"module": "advanced_fft_arithmetic"},
            )

        extra = {"default_backend": self.default_backend}
        if self.cache_coordinator is None:
            extra["cache_strategy"] = "local"
        else:
            extra["cache_strategy"] = "coordinated"

        return FFTBackendCapabilities(
            backend_name=self.backend_name,
            max_nodes=None,
            precision=self.precision,
            supports_distributed=False,
            extra=extra,
        )

    def _calculate_attenuation_db(self, filter_response: np.ndarray) -> float:
        """Return the response dynamic range as nonnegative attenuation dB."""
        if len(filter_response) == 0:
            return 0.0

        magnitudes = np.abs(np.asarray(filter_response, dtype=float))
        peak = float(np.max(magnitudes))
        floor = float(np.min(magnitudes))
        if peak <= 0.0 or floor <= 0.0:
            return float("inf")
        return float(20.0 * np.log10(peak / floor))

    def get_spectral_state(
        self, G: Any, force_recompute: bool = False
    ) -> SpectralState:
        """
        Get an orthonormal symmetric-Laplacian basis for the graph signal.
        """
        if not HAS_NETWORKX or G is None:
            raise NetworkConfigError(
                parameter="graph",
                value=G,
                reason="Graph required for spectral analysis",
            )

        spectral_basis = None

        if self.cache_coordinator is not None:
            spectral_basis = self.cache_coordinator.get_spectral_basis(
                G, force_recompute=force_recompute
            )
            eigenvalues = spectral_basis.eigenvalues
            eigenvectors = spectral_basis.eigenvectors
        else:
            if not HAS_SPECTRAL:
                raise TNFRUserError(
                    message="Spectral analysis not available",
                    suggestion="Install spectral dependencies",
                    context={"has_spectral": HAS_SPECTRAL},
                )

            if HAS_UNIFIED_CACHE:
                cache = get_unified_cache()
                eigenvalues, eigenvectors = cache.get(
                    CacheEntryType.SPECTRAL_DECOMPOSITION,
                    G,
                    computation_func=lambda: get_laplacian_spectrum(G),
                    mathematical_importance=FFT_ARITHMETIC_IMPORTANCE_CANONICAL,
                )
            else:
                eigenvalues, eigenvectors = get_laplacian_spectrum(G)

        # Extract current signal from nodes
        signal = self._graph_epi_signal(G)

        # Compute spectral coefficients
        spectral_coeffs = gft(signal, eigenvectors)

        # Analyze spectral properties
        amplitudes = np.abs(spectral_coeffs)
        phases = np.angle(spectral_coeffs)

        # Find dominant modes (largest amplitude coefficients)
        dominant_indices = np.argsort(amplitudes)[-5:]  # Top 5 modes

        # Estimate coherence length from spectral decay
        coherence_length = self._estimate_coherence_length(eigenvalues, amplitudes)

        # Create spectral state
        spectral_state = SpectralState(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_coeffs=spectral_coeffs,
            # Historical field name: these are Laplacian spectral parameters,
            # not temporal frequencies.  Graph-wave angular frequencies would
            # be sqrt(lambda).
            frequencies=eigenvalues,
            amplitudes=amplitudes,
            phases=phases,
            coherence_length=coherence_length,
            dominant_modes=dominant_indices,
        )

        # Do not cache the complete state locally: it contains live EPI
        # coefficients and would become stale after a nodal update.  The
        # decomposition itself is cached by the spectral/cache layers.

        if self.cache_coordinator is not None:
            self.cache_coordinator.register_spectral_state(G, spectral_state)

        return spectral_state

    def _get_aligned_spectral_states(
        self, G1: Any, G2: Any
    ) -> tuple[SpectralState, SpectralState, np.ndarray]:
        """Return compatible graph states and second coefficients in basis one.

        Matching eigenvalues alone are insufficient because cospectral graphs
        can have different eigenvectors.  This shared validator also removes
        the independent sign/phase choice of each eigenvector.
        """
        spectral1 = self.get_spectral_state(G1)
        spectral2 = self.get_spectral_state(G2)
        if tuple(G1.nodes()) != tuple(G2.nodes()):
            raise TNFRValueError(
                "Cross-spectral comparison requires the same node order",
                context={
                    "first_nodes": tuple(G1.nodes()),
                    "second_nodes": tuple(G2.nodes()),
                },
                suggestion="Align node labels and order before comparing graph modes.",
            )
        compatible_shape = spectral1.frequencies.shape == spectral2.frequencies.shape
        compatible_values = compatible_shape and np.allclose(
            spectral1.frequencies, spectral2.frequencies, rtol=1e-10, atol=1e-12
        )
        if not compatible_values:
            raise TNFRValueError(
                "Cross-spectral comparison requires matching Laplacian spectra",
                context={
                    "first_shape": spectral1.frequencies.shape,
                    "second_shape": spectral2.frequencies.shape,
                },
                suggestion="Compare signals in one shared graph basis.",
            )
        overlap = spectral1.eigenvectors.conj().T @ spectral2.eigenvectors
        diagonal_overlap = np.diag(overlap)
        aligned_basis = np.allclose(
            overlap,
            np.diag(diagonal_overlap),
            rtol=1e-9,
            atol=1e-11,
        ) and np.allclose(np.abs(diagonal_overlap), 1.0, rtol=1e-9, atol=1e-11)
        if not aligned_basis:
            raise TNFRValueError(
                "Cross-spectral comparison requires aligned eigenvectors",
                context={"basis_shape": spectral1.eigenvectors.shape},
                suggestion="Project both signals into one explicitly shared basis.",
            )
        aligned_coefficients2 = diagonal_overlap * spectral2.spectral_coeffs
        return spectral1, spectral2, aligned_coefficients2

    def spectral_convolution(
        self,
        G: Any,
        signal1: np.ndarray | None = None,
        signal2: np.ndarray | None = None,
        operation: str = "multiply",
    ) -> FFTArithmeticResult:
        """Combine graph signals in the cached Laplacian eigenbasis.

        ``multiply`` and the compatibility alias ``convolve`` multiply modal
        coefficients, defining graph spectral convolution. ``add`` adds modal
        coefficients and therefore reconstructs ordinary signal addition.
        None of these operations computes the nodal product ``νf * ΔNFR``.

        When omitted, ``signal1`` is the graph EPI signal and ``signal2`` is the
        structural-frequency signal for backward compatibility.  Their default
        graph convolution must not be interpreted as nodal evolution.
        """
        start_time = time.perf_counter()

        # Get the orthonormal symmetric-Laplacian basis.
        spectral_state = self.get_spectral_state(G)

        # Extract signals from graph if not provided
        if signal1 is None:
            signal1 = self._graph_epi_signal(G)
        if signal2 is None:
            signal2 = self._graph_frequency_signal(G)

        expected_shape = (len(spectral_state.eigenvalues),)
        signal1 = self._finite_signal(
            signal1, label="signal1", expected_shape=expected_shape
        )
        signal2 = self._finite_signal(
            signal2, label="signal2", expected_shape=expected_shape
        )

        actual_backend = "numpy_dense_gft"

        # Use a matrix backend for large signals when available.
        if HAS_MATH_BACKENDS and len(signal1) > 100:
            try:
                backend = self.backend or get_backend()
                if backend.supports_autodiff:
                    # Convert to backend tensors
                    s1_tensor = backend.as_array(signal1)
                    s2_tensor = backend.as_array(signal2)
                    U_tensor = backend.as_array(spectral_state.eigenvectors)

                    # GPU GFT: spectral = U^T @ signal
                    spectral1_tensor = backend.matmul(
                        backend.conjugate_transpose(U_tensor), s1_tensor
                    )
                    spectral2_tensor = backend.matmul(
                        backend.conjugate_transpose(U_tensor), s2_tensor
                    )

                    # Perform operation in spectral domain on GPU
                    if operation == "multiply":
                        result_spectral_tensor = spectral1_tensor * spectral2_tensor
                    elif operation == "add":
                        result_spectral_tensor = spectral1_tensor + spectral2_tensor
                    elif operation == "convolve":
                        result_spectral_tensor = spectral1_tensor * spectral2_tensor
                    else:
                        raise TNFRUserError(
                            message=f"Unknown spectral operation: {operation}",
                            suggestion="Use 'multiply', 'add', or 'convolve'",
                            context={"operation": operation},
                        )

                    # GPU IGFT: result = U @ spectral
                    result_spatial_tensor = backend.matmul(
                        U_tensor, result_spectral_tensor
                    )
                    result_spatial = backend.to_numpy(result_spatial_tensor)
                    actual_backend = f"{backend.name}_dense_gft"

                else:
                    raise TNFRUserError(
                        message="Backend doesn't support autodiff",
                        suggestion=(
                            "Use a backend with autodiff support (e.g., JAX, PyTorch)"
                        ),
                        context={"backend": self.backend_name},
                    )
            except Exception:
                # Fallback to CPU implementation
                spectral1 = gft(signal1, spectral_state.eigenvectors)
                spectral2 = gft(signal2, spectral_state.eigenvectors)

                if operation == "multiply":
                    result_spectral = spectral1 * spectral2
                elif operation == "add":
                    result_spectral = spectral1 + spectral2
                elif operation == "convolve":
                    result_spectral = spectral1 * spectral2
                else:
                    raise TNFRUserError(
                        message=f"Unknown spectral operation: {operation}",
                        suggestion="Use 'multiply', 'add', or 'convolve'",
                        context={"operation": operation},
                    )

                result_spatial = igft(result_spectral, spectral_state.eigenvectors)
        else:
            # CPU implementation for small signals or when GPU unavailable
            spectral1 = gft(signal1, spectral_state.eigenvectors)
            spectral2 = gft(signal2, spectral_state.eigenvectors)

            if operation == "multiply":
                result_spectral = spectral1 * spectral2
            elif operation == "add":
                result_spectral = spectral1 + spectral2
            elif operation == "convolve":
                result_spectral = spectral1 * spectral2
            else:
                msg = f"Unknown operation: {operation}"
                raise TNFRValueError(
                    msg,
                    context={
                        "operation": operation,
                        "allowed": ["multiply", "add", "convolve"],
                    },
                    suggestion="Use 'multiply', 'add', or 'convolve'.",
                )

            result_spatial = igft(result_spectral, spectral_state.eigenvectors)

        execution_time = time.perf_counter() - start_time

        # Update statistics
        self.total_operations += 1
        # One state GFT, two operand GFTs, and one reconstruction.
        self.total_fft_ops += 4

        result = FFTArithmeticResult(
            operation=SpectralOperation.CONVOLUTION,
            input_shape=signal1.shape,
            output_data=result_spatial,
            spectral_state=spectral_state,
            execution_time=execution_time,
            fft_operations=4,
            backend_used=actual_backend,
            accuracy_metrics={"nodal_product_equivalent": 0.0},
        )

        if self.cache_coordinator is not None:
            self.cache_coordinator.register_fft_result(
                G, result, metadata={"mode": operation}
            )

        return result

    def harmonic_analysis(
        self, G: Any, num_harmonics: int = 5, window_size: int | None = None
    ) -> FFTArithmeticResult:
        """Rank graph-Laplacian modes by current EPI amplitude.

        The compatibility output retains historical ``harmonic_*`` keys.  The
        reported values are Laplacian spectral parameters and amplitude ratios,
        not temporal harmonics. ``window_size`` is retained for API compatibility
        and is reported but cannot window a single graph snapshot.
        """
        if (
            isinstance(num_harmonics, bool)
            or not isinstance(num_harmonics, Integral)
            or num_harmonics <= 0
        ):
            raise TNFRValueError(
                "num_harmonics must be a positive integer",
                context={"num_harmonics": num_harmonics},
                suggestion="Request at least one Laplacian mode.",
            )
        num_harmonics = int(num_harmonics)
        start_time = time.perf_counter()

        # Get spectral state
        spectral_state = self.get_spectral_state(G)

        # Extract EPI signal
        epi_signal = self._graph_epi_signal(G)

        # Rank modes by current coefficient amplitude.
        amplitudes = spectral_state.amplitudes
        spectral_parameters = spectral_state.frequencies
        count = min(num_harmonics, len(amplitudes))
        harmonic_indices = np.argsort(amplitudes)[::-1][:count]
        harmonic_freqs = spectral_parameters[harmonic_indices]
        harmonic_amps = amplitudes[harmonic_indices]

        dominant_parameter = harmonic_freqs[0]

        harmonic_ratios = []
        for parameter in harmonic_freqs:
            if dominant_parameter > 0:
                ratio = parameter / dominant_parameter
                harmonic_ratios.append(ratio)
            else:
                harmonic_ratios.append(0.0)

        # Compatibility metric: energy outside the dominant selected mode,
        # normalized by the dominant amplitude.  This is not audio THD.
        total_harmonic_distortion = (
            np.sqrt(np.sum(harmonic_amps[1:] ** 2)) / harmonic_amps[0]
            if harmonic_amps[0] > 0
            else 0.0
        )

        execution_time = time.perf_counter() - start_time

        # Create result
        harmonic_data = {
            "fundamental_frequency": dominant_parameter,
            "harmonic_frequencies": harmonic_freqs,
            "harmonic_amplitudes": harmonic_amps,
            "harmonic_ratios": harmonic_ratios,
            "total_harmonic_distortion": total_harmonic_distortion,
            "dominant_mode_index": harmonic_indices[0],
            "metric_semantics": "laplacian_mode_amplitude_ranking",
            "window_size_applied": False,
            "requested_window_size": window_size,
        }

        self.total_operations += 1
        self.total_fft_ops += 1

        return FFTArithmeticResult(
            operation=SpectralOperation.HARMONIC_ANALYSIS,
            input_shape=epi_signal.shape,
            output_data=harmonic_data,
            spectral_state=spectral_state,
            execution_time=execution_time,
            fft_operations=1,
            backend_used="dense_gft",
        )

    def spectral_filtering(
        self,
        G: Any,
        filter_type: str = "lowpass",
        cutoff_frequency: float | None = None,
        filter_order: int = 4,
    ) -> FFTArithmeticResult:
        """
        Apply spectral filtering to graph signals.

        Returns a filtered EPI signal for downstream analysis.  It does not
        write graph state or advance the nodal equation.
        """
        start_time = time.perf_counter()

        # Get spectral state
        spectral_state = self.get_spectral_state(G)

        # Determine cutoff frequency if not specified
        if cutoff_frequency is None:
            # Use median eigenvalue as default cutoff
            cutoff_frequency = np.median(spectral_state.eigenvalues)

        frequencies = spectral_state.eigenvalues

        def _build_filter() -> np.ndarray:
            return self._build_filter_response(
                frequencies, filter_type, float(cutoff_frequency), filter_order
            )

        if self.cache_coordinator is not None:
            filter_response = self.cache_coordinator.get_kernel(
                G,
                "spectral_filter",
                _build_filter,
                kernel_params={
                    "type": filter_type,
                    "cutoff": float(cutoff_frequency),
                    "order": filter_order,
                },
            )
        else:
            filter_response = _build_filter()

        # Apply filter to current signal
        epi_signal = self._graph_epi_signal(G)
        spectral_coeffs = spectral_state.spectral_coeffs

        # Apply filter
        filtered_coeffs = spectral_coeffs * filter_response

        # Transform back to spatial domain
        filtered_signal = igft(filtered_coeffs, spectral_state.eigenvectors)

        execution_time = time.perf_counter() - start_time

        # Create filtered result
        filtered_data = {
            "original_signal": epi_signal,
            "filtered_signal": filtered_signal,
            "filter_response": filter_response,
            "cutoff_frequency": cutoff_frequency,
            "filter_type": filter_type,
            "attenuation_db": self._calculate_attenuation_db(filter_response),
        }

        self.total_operations += 1
        # The live state GFT plus the filtered reconstruction.
        self.total_fft_ops += 2

        result = FFTArithmeticResult(
            operation=SpectralOperation.FILTERING,
            input_shape=epi_signal.shape,
            output_data=filtered_data,
            spectral_state=spectral_state,
            execution_time=execution_time,
            fft_operations=2,
            backend_used="dense_gft",
        )

        if self.cache_coordinator is not None:
            self.cache_coordinator.register_fft_result(
                G, result, metadata={"filter_type": filter_type}
            )

        return result

    def _build_filter_response(
        self,
        frequencies: np.ndarray,
        filter_type: str,
        cutoff_frequency: float,
        filter_order: int,
    ) -> np.ndarray:
        """Construct smooth spectral filter responses."""

        if (
            isinstance(filter_order, bool)
            or not isinstance(filter_order, Integral)
            or filter_order <= 0
        ):
            raise TNFRValueError(
                "filter_order must be a positive integer",
                context={"filter_order": filter_order},
                suggestion="Use an integer filter order of at least one.",
            )
        filter_order = int(filter_order)
        if not np.isfinite(cutoff_frequency) or cutoff_frequency < 0.0:
            raise TNFRValueError(
                "cutoff_frequency must be finite and nonnegative",
                context={"cutoff_frequency": cutoff_frequency},
                suggestion="Use a nonnegative Laplacian spectral parameter.",
            )

        response = np.ones_like(frequencies)
        safe_cutoff = max(cutoff_frequency, 1e-9)

        if filter_type == "lowpass":
            attenuation = np.maximum(frequencies - safe_cutoff, 0.0)
            response = np.exp(-attenuation * filter_order / safe_cutoff)
        elif filter_type == "highpass":
            attenuation = np.maximum(frequencies - safe_cutoff, 0.0)
            response = 1.0 - np.exp(-attenuation * filter_order / safe_cutoff)
        elif filter_type == "bandpass":
            low_cutoff = (
                safe_cutoff * FFT_LOW_CUTOFF_CANONICAL
            )  # = 0.34 (operational)
            high_cutoff = (
                safe_cutoff * FFT_HIGH_CUTOFF_CANONICAL
            )  # = 0.6 (operational)
            response = np.exp(
                -np.maximum(low_cutoff - frequencies, 0.0) * filter_order / safe_cutoff
            )
            response *= np.exp(
                -np.maximum(frequencies - high_cutoff, 0.0) * filter_order / safe_cutoff
            )
        elif filter_type == "notch":
            bandwidth = (
                safe_cutoff * FFT_BANDWIDTH_CANONICAL
            )  # = 0.1 (operational)
            distance = np.abs(frequencies - safe_cutoff)
            safe_bandwidth = max(bandwidth, 1e-12)
            response = 1.0 - np.exp(
                -((distance / safe_bandwidth) ** filter_order)
            )
        else:
            raise TNFRUserError(
                message=f"Unknown filter type: {filter_type}",
                suggestion="Use 'lowpass', 'highpass', 'bandpass', or 'notch'",
                context={"filter_type": filter_type},
            )

        return np.clip(response, 0.0, 1.0)

    def cross_spectral_coherence(
        self, G1: Any, G2: Any, frequency_bands: int | None = 10
    ) -> FFTArithmeticResult:
        """Compute band-averaged alignment in compatible graph coordinates.

        This snapshot diagnostic compares coefficient vectors only when both
        Laplacian spectra match.  Each band's normalized averaged cross-power is
        assigned to its modes.  It is not a trajectory-level synchronization or
        coupling certificate.
        """
        if frequency_bands is not None and (
            isinstance(frequency_bands, bool)
            or not isinstance(frequency_bands, Integral)
            or frequency_bands <= 0
        ):
            raise TNFRValueError(
                "frequency_bands must be a positive integer or None",
                context={"frequency_bands": frequency_bands},
                suggestion="Use a positive band count, or None for one global band.",
            )
        if frequency_bands is not None:
            frequency_bands = int(frequency_bands)
        start_time = time.perf_counter()

        spectral1, spectral2, coeffs2 = self._get_aligned_spectral_states(G1, G2)
        coeffs1 = spectral1.spectral_coeffs
        freqs1 = spectral1.frequencies
        min_size = len(coeffs1)
        if min_size == 0:
            raise TNFRValueError(
                "Cross-spectral comparison requires a non-empty graph",
                context={"node_count": 0},
                suggestion="Provide graphs with at least one node.",
            )

        # Compute cross-spectral density
        cross_spectrum = coeffs1 * np.conj(coeffs2)

        # Compute power spectra
        power1 = np.abs(coeffs1) ** 2
        power2 = np.abs(coeffs2) ** 2

        band_count = frequency_bands or 1
        lower = float(np.min(freqs1))
        upper = float(np.max(freqs1))
        if upper == lower:
            band_edges = np.array([lower, np.nextafter(upper, np.inf)])
            band_count = 1
        else:
            band_edges = np.linspace(lower, upper, band_count + 1)
            band_edges[-1] = np.nextafter(band_edges[-1], np.inf)

        coherence = np.zeros(min_size, dtype=float)
        band_coherence = []
        for index in range(band_count):
            band_mask = (freqs1 >= band_edges[index]) & (
                freqs1 < band_edges[index + 1]
            )
            if not np.any(band_mask):
                band_coherence.append(0.0)
                continue
            cross_mean = np.mean(cross_spectrum[band_mask])
            auto_power1 = float(np.mean(power1[band_mask]))
            auto_power2 = float(np.mean(power2[band_mask]))
            denominator = auto_power1 * auto_power2
            value = (
                float(abs(cross_mean) ** 2 / denominator)
                if denominator > 0
                else 0.0
            )
            value = float(np.clip(value, 0.0, 1.0))
            coherence[band_mask] = value
            band_coherence.append(value)

        # Compute overall coherence metrics
        mean_coherence = np.mean(coherence)
        max_coherence = np.max(coherence)
        coherent_bandwidth = np.sum(coherence > FFT_COHERENT_THRESHOLD_CANONICAL) / len(
            coherence
        )  # = 0.62 (operational; fraction above threshold)

        execution_time = time.perf_counter() - start_time

        # Create coherence result
        coherence_data = {
            "coherence_spectrum": coherence,
            "cross_spectrum": cross_spectrum,
            "frequencies": freqs1,  # Use first graph's frequencies
            "band_coherence": band_coherence,
            "mean_coherence": mean_coherence,
            "max_coherence": max_coherence,
            "coherent_bandwidth": coherent_bandwidth,
            "frequency_bands": frequency_bands,
            "metric_semantics": "band_averaged_snapshot_cross_power",
        }

        self.total_operations += 1
        self.total_fft_ops += 2

        return FFTArithmeticResult(
            operation=SpectralOperation.COHERENCE_ANALYSIS,
            input_shape=(min_size,),
            output_data=coherence_data,
            execution_time=execution_time,
            fft_operations=2,
            backend_used="dense_gft",
        )

    def _estimate_coherence_length(
        self, eigenvalues: np.ndarray, amplitudes: np.ndarray
    ) -> float:
        """Return an inverse-root spectral-centroid scale heuristic.

        The compatibility field ``coherence_length`` is not the fitted canonical
        ``xi_C`` and is not the topology-only ``1/sqrt(lambda_2)`` fallback.
        """
        # Find spectral centroid (weighted average frequency)
        total_power = np.sum(amplitudes**2)
        if total_power > 0:
            centroid = np.sum(eigenvalues * amplitudes**2) / total_power
        else:
            centroid = 0.0

        # Estimate coherence length as inverse of spectral centroid
        if centroid > 0:
            coherence_length = 1.0 / np.sqrt(centroid)
        else:
            coherence_length = float("inf")  # Infinite coherence

        return coherence_length

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_operations": self.total_operations,
            "total_fft_operations": self.total_fft_ops,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_operations),
            "cached_spectra": (
                len(self._spectral_cache) if self._spectral_cache is not None else 0
            ),
            "backend": self.default_backend,
            "precision": self.precision,
            "spectral_analysis_available": HAS_SPECTRAL,
            "unified_cache_available": HAS_UNIFIED_CACHE,
        }

    def clear_cache(self) -> None:
        """Clear internal caches."""
        if self._spectral_cache is not None:
            self._spectral_cache.clear()


# Factory functions
def create_fft_arithmetic_engine(**kwargs) -> TNFRAdvancedFFTEngine:
    """Create FFT arithmetic engine."""
    return TNFRAdvancedFFTEngine(**kwargs)


def fast_spectral_convolution(
    G: Any, signal1: np.ndarray, signal2: np.ndarray
) -> np.ndarray:
    """Return a graph spectral convolution (historical name retained)."""
    engine = create_fft_arithmetic_engine()
    result = engine.spectral_convolution(G, signal1, signal2, operation="multiply")
    return result.output_data


def analyze_graph_harmonics(G: Any, num_harmonics: int = 5) -> dict[str, Any]:
    """Convenience function for harmonic analysis."""
    engine = create_fft_arithmetic_engine()
    result = engine.harmonic_analysis(G, num_harmonics)
    return result.output_data


def measure_graph_coherence(G1: Any, G2: Any) -> float:
    """Convenience function for measuring cross-graph coherence."""
    engine = create_fft_arithmetic_engine()
    result = engine.cross_spectral_coherence(G1, G2)
    return result.output_data["mean_coherence"]

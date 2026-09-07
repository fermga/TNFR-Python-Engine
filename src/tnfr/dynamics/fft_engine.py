"""Dense graph-spectral coordinates for the isolated EPI diffusion channel.

The engine diagonalizes the fixed random-walk Laplacian representation and
uses GFT/IGFT coordinate changes, while applying the heterogeneous nodal
product nu_f * DeltaNFR in node space. It also advances phase through the
shared simultaneous U3-gated proposal. The implementation is an exact
coordinate realization of its declared model, not a generic FFT speed claim,
a wavelet decomposition, or an implementation of arbitrary DeltaNFR channels.
"""

import hashlib
import math
import time
import weakref
from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

from ..alias import get_attr, set_attr, set_dnfr, set_theta
from ..config.operator_names import BIFURCATION_WINDOW
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..metrics.common import structural_coherence
from ..operators.network_stage import GraphTransactionSnapshot
from ..types import real_scalar_epi
from .phase_evolution import propose_u3_gated_phase_step

if TYPE_CHECKING:
    from .fft_cache_coordinator import FFTCacheCoordinator

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import existing modules
try:
    from ..mathematics.spectral import get_laplacian_spectrum, gft, igft
    from ..physics.structural_diffusion import structural_diffusion_operator
    from .structural_cache import get_structural_cache

    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# Import FFT cache coordinator
try:
    from .fft_cache_coordinator import get_fft_cache_coordinator

    HAS_FFT_CACHE = True
except ImportError:
    HAS_FFT_CACHE = False

# Operational engine-tuning knob (not TNFR physics) → tnfr.constants.operational
from ..constants.operational import (
    FFT_ENGINE_COUPLING_CANONICAL,
)


@dataclass
class FFTDynamicsState:
    """State container for graph-spectral dynamics."""

    spectral_epi: np.ndarray
    spectral_phase: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    time: float = 0.0
    dt: float = 0.01
    node_order: tuple[Any, ...] = ()
    basis_signature: str = ""
    nodal_residual: float = 0.0
    diffusion_operator: np.ndarray | None = None
    eigenbasis_digest: str = ""
    integration_method: str | None = None
    stability_not_certified: bool = False


_PHYSICAL_HISTORY_MAXLEN = max(2, BIFURCATION_WINDOW + 1)


def _finite_real(value: Any, name: str) -> float:
    """Return a finite, non-Boolean scalar used by the spectral flow."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TNFRValueError(f"{name} must be a finite real scalar.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{name} must be a finite real scalar.") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{name} must be a finite real scalar.")
    return result


def _finite_epi(value: Any, name: str) -> float:
    """Read the signed scalar EPI chart without collapsing richer BEPI state."""

    if isinstance(value, (bool, np.bool_)):
        raise TNFRValueError(f"{name} must be a finite scalar EPI embedding.")
    try:
        result = real_scalar_epi(value)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{name} must be a finite scalar EPI embedding."
        ) from exc
    if result is None or not math.isfinite(float(result)):
        raise TNFRValueError(f"{name} must be a finite scalar EPI embedding.")
    return float(result)


def _eigenbasis_digest(eigenvalues: Any, eigenvectors: Any) -> str:
    """Return an exact content digest for one ordered full eigensystem."""

    values = np.ascontiguousarray(eigenvalues)
    vectors = np.ascontiguousarray(eigenvectors)
    payload = (
        values.shape,
        values.dtype.str,
        vectors.shape,
        vectors.dtype.str,
    )
    digest = hashlib.sha256(repr(payload).encode("utf-8"))
    digest.update(values.tobytes())
    digest.update(vectors.tobytes())
    return digest.hexdigest()


def _readonly_basis(eigenvalues: Any, eigenvectors: Any) -> tuple[np.ndarray, np.ndarray]:
    """Detach cached basis arrays and make cache poisoning fail immediately."""

    values = np.array(eigenvalues, copy=True)
    vectors = np.array(eigenvectors, copy=True)
    values.setflags(write=False)
    vectors.setflags(write=False)
    return values, vectors


def _basis_signature(G: Any, nodes: tuple[Any, ...]) -> str:
    """Fingerprint the fixed weighted graph basis, excluding nodal state."""

    indices = {node: index for index, node in enumerate(nodes)}
    records: list[tuple[int, int, str]] = []
    multiple = bool(G.is_multigraph())
    for node, source in indices.items():
        for neighbor, attributes in G.adj[node].items():
            if neighbor not in indices:
                continue
            try:
                weight = (
                    float(
                        sum(
                            edge_data.get("weight", 1.0)
                            for edge_data in attributes.values()
                        )
                    )
                    if multiple
                    else float(attributes.get("weight", 1.0))
                )
            except (OverflowError, TypeError, ValueError) as exc:
                raise TNFRValueError(
                    "FFT dynamics requires finite nonnegative edge weights."
                ) from exc
            if not math.isfinite(weight) or weight < 0.0:
                raise TNFRValueError(
                    "FFT dynamics requires finite nonnegative edge weights."
                )
            if weight:
                records.append((source, indices[neighbor], weight.hex()))

    node_records = tuple(
        (type(node).__module__, type(node).__qualname__, repr(node)) for node in nodes
    )
    payload = (
        bool(G.is_directed()),
        multiple,
        node_records,
        tuple(sorted(records)),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _single_sample_history(raw: Any, sample: float) -> Any:
    """Preserve a legacy history's container contract while removing stale rates."""

    if isinstance(raw, deque):
        return deque((sample,), maxlen=raw.maxlen)
    if isinstance(raw, tuple):
        return (sample,)
    return [sample]


class FFTDynamicsEngine:
    """
    Graph-spectral TNFR dynamics engine.

    The general-graph path stores a full dense eigenbasis and applies dense
    transforms. It is a coordinate-space implementation with reusable
    topology caches, not a generic ``O(N log N)`` FFT algorithm. Structured
    graph backends may provide faster transforms independently.
    """

    def __init__(
        self,
        enable_caching: bool = True,
        cache_coordinator: "FFTCacheCoordinator | None" = None,
    ):
        self.enable_caching = enable_caching
        self.cache_coordinator = (
            cache_coordinator
            if cache_coordinator is not None
            else (get_fft_cache_coordinator() if HAS_FFT_CACHE else None)
        )

        # Every engine keeps a topology-only front cache. Nodal EPI, phase and
        # frequency changes do not alter the fixed graph Fourier basis.
        self._spectral_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._graph_basis_signatures: weakref.WeakKeyDictionary[Any, str] = (
            weakref.WeakKeyDictionary()
        )

        # Performance tracking
        self.total_operations = 0
        self.fft_operations = 0
        self.cache_hits = 0

        # Integration with structural cache
        self.structural_cache = get_structural_cache()

    def preprocess_graph_for_fft(self, G: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess graph to extract spectral basis for FFT operations.

        Returns eigenvalues and eigenvectors of the graph Laplacian.
        """
        if not HAS_NETWORKX or not HAS_SPECTRAL or G is None:
            return np.array([]), np.array([])

        nodes = tuple(G.nodes())
        if not nodes:
            return np.array([]), np.empty((0, 0), dtype=float)
        signature = _basis_signature(G, nodes)

        if self.enable_caching and signature in self._spectral_cache:
            self.cache_hits += 1
            return self._spectral_cache[signature]

        if not self.enable_caching:
            return get_laplacian_spectrum(G)

        # Use cache coordinator if available
        if self.cache_coordinator is not None:
            previous_signature = self._graph_basis_signatures.get(G)
            force_recompute = (
                previous_signature is not None and previous_signature != signature
            )
            before_hits = None
            get_stats = getattr(self.cache_coordinator, "get_stats", None)
            if callable(get_stats):
                before_hits = int(get_stats().get("spectral_hits", 0))
            spectral_basis = self.cache_coordinator.get_spectral_basis(
                G, force_recompute=force_recompute
            )
            eigenvals = spectral_basis.eigenvalues
            eigenvecs = spectral_basis.eigenvectors
            if before_hits is not None:
                after_hits = int(get_stats().get("spectral_hits", before_hits))
                self.cache_hits += max(0, after_hits - before_hits)
        else:
            eigenvals, eigenvecs = get_laplacian_spectrum(G)

        eigenvals, eigenvecs = _readonly_basis(eigenvals, eigenvecs)
        self._spectral_cache[signature] = (eigenvals, eigenvecs)
        self._graph_basis_signatures[G] = signature

        return eigenvals, eigenvecs

    def create_fft_state(self, G: Any) -> FFTDynamicsState:
        """
        Create a graph-spectral state representation of the graph.

        Transforms spatial domain node properties into spectral domain.
        """
        if not HAS_NETWORKX or G is None:
            return FFTDynamicsState(
                spectral_epi=np.array([]),
                spectral_phase=np.array([]),
                eigenvalues=np.array([]),
                eigenvectors=np.array([]),
            )

        nodes = tuple(G.nodes())
        initial_time = _finite_real(G.graph.get("_t", 0.0), "graph runtime time")
        if not nodes:
            return FFTDynamicsState(
                spectral_epi=np.array([]),
                spectral_phase=np.array([]),
                eigenvalues=np.array([]),
                eigenvectors=np.empty((0, 0), dtype=float),
                time=initial_time,
                node_order=nodes,
                basis_signature=_basis_signature(G, nodes),
            )

        # Get spectral basis
        eigenvals, eigenvecs = self.preprocess_graph_for_fft(G)
        diffusion_nodes, diffusion_operator = structural_diffusion_operator(G)
        if tuple(diffusion_nodes) != nodes:
            raise TNFRValueError(
                "Diffusion operator node order differs from the FFT basis order."
            )

        if len(eigenvals) == 0:
            return FFTDynamicsState(
                spectral_epi=np.array([]),
                spectral_phase=np.array([]),
                eigenvalues=eigenvals,
                eigenvectors=eigenvecs,
                time=initial_time,
                node_order=nodes,
                basis_signature=_basis_signature(G, nodes),
                diffusion_operator=np.asarray(diffusion_operator, dtype=float),
            )

        # Extract spatial domain data
        epi_spatial = np.array(
            [
                _finite_epi(
                    get_attr(
                        G.nodes[node],
                        ALIAS_EPI,
                        0.0,
                        strict=True,
                        conv=lambda value: value,
                    ),
                    f"node {node!r} EPI",
                )
                for node in nodes
            ],
            dtype=float,
        )
        phase_spatial = np.array(
            [
                _finite_real(
                    get_attr(G.nodes[node], ALIAS_THETA, 0.0, strict=True),
                    f"node {node!r} phase",
                )
                for node in nodes
            ],
            dtype=float,
        )

        # Transform to spectral domain using Graph Fourier Transform
        spectral_epi = gft(epi_spatial, eigenvecs)
        spectral_phase = gft(phase_spatial, eigenvecs)

        return FFTDynamicsState(
            spectral_epi=spectral_epi,
            spectral_phase=spectral_phase,
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            time=initial_time,
            node_order=nodes,
            basis_signature=_basis_signature(G, nodes),
            diffusion_operator=np.asarray(diffusion_operator, dtype=float),
            eigenbasis_digest=_eigenbasis_digest(eigenvals, eigenvecs),
        )

    def fft_accelerated_step(
        self, G: Any, fft_state: FFTDynamicsState, dt: float
    ) -> FFTDynamicsState:
        """
        Perform one EPI-diffusion step in graph-spectral coordinates.

        Implements: ∂EPI/∂t = νf · ΔNFR in the spectral domain.

        The coordinate realization is exact for the evaluated Euler derivative.
        No arbitrary-dt forward-Euler stability certificate is inferred.
        """
        if not HAS_NETWORKX or len(fft_state.eigenvalues) == 0:
            return fft_state

        dt_value = _finite_real(dt, "FFT integration dt")
        if dt_value <= 0.0:
            raise TNFRValueError("FFT integration dt must be positive.")

        nodes = self._validate_fixed_basis(G, fft_state)

        # Extract νf values in node order
        vf_spatial = np.array(
            [
                _finite_real(
                    get_attr(G.nodes[node], ALIAS_VF, 1.0, strict=True),
                    f"node {node!r} structural frequency",
                )
                for node in nodes
            ],
            dtype=float,
        )
        if np.any(vf_spatial < 0.0):
            raise TNFRValueError("Structural frequency must be nonnegative.")

        # The phase path retains the GFT of νf for its exact inverse transform.
        vf_spectral = gft(vf_spatial, fft_state.eigenvectors)

        # The orthonormal Q basis diagonalizes L_sym, while the canonical EPI
        # pressure is -L_rw*x. Keeping spectral_epi = Q^T*x preserves the
        # public raw-EPI representation, so pressure is evaluated with the
        # fixed L_rw snapshot in node space and projected back with Q^T.
        epi_spatial = igft(fft_state.spectral_epi, fft_state.eigenvectors)
        diffusion_operator = self._fixed_diffusion_operator(G, fft_state, nodes)
        dnfr_spatial = -(diffusion_operator @ epi_spatial)

        # Multiplication by νf is pointwise in node space, not mode space:
        #
        #   GFT(νf · ΔNFR) = Qᵀ diag(νf) Q ΔNFR_hat.
        #
        # GFT(νf) * GFT(ΔNFR) would instead be a modewise product and is not
        # the transform of the canonical nodal product, even for constant νf.
        depi_dt_spatial = vf_spatial * dnfr_spatial
        depi_dt_spectral = gft(depi_dt_spatial, fft_state.eigenvectors)

        # Forward Euler integration in spectral domain
        new_spectral_epi = fft_state.spectral_epi + dt_value * depi_dt_spectral

        # Phase evolution using coupling dynamics
        new_spectral_phase = self._evolve_spectral_phase(
            G, fft_state, dt_value, vf_spectral
        )

        new_epi_spatial = igft(new_spectral_epi, fft_state.eigenvectors)
        realised_rate = (new_epi_spatial - epi_spatial) / dt_value
        nodal_residual = float(np.max(np.abs(realised_rate - depi_dt_spatial)))
        if not math.isfinite(nodal_residual):
            raise TNFRValueError("FFT nodal residual must remain finite.")

        self.total_operations += 1
        self.fft_operations += 1

        return FFTDynamicsState(
            spectral_epi=new_spectral_epi,
            spectral_phase=new_spectral_phase,
            eigenvalues=fft_state.eigenvalues,
            eigenvectors=fft_state.eigenvectors,
            time=fft_state.time + dt_value,
            dt=dt_value,
            node_order=fft_state.node_order,
            basis_signature=fft_state.basis_signature,
            nodal_residual=nodal_residual,
            diffusion_operator=diffusion_operator,
            eigenbasis_digest=fft_state.eigenbasis_digest,
            integration_method="forward_euler",
            stability_not_certified=True,
        )

    def _validate_fixed_basis(
        self, G: Any, fft_state: FFTDynamicsState
    ) -> tuple[Any, ...]:
        """Reject graph or array changes incompatible with the state's basis."""

        if G is None:
            raise TNFRValueError("FFT dynamics requires a graph.")
        nodes = tuple(G.nodes())
        state_nodes = fft_state.node_order or nodes
        if nodes != state_nodes:
            raise TNFRValueError(
                "Graph node order changed after the FFT basis was created."
            )
        if fft_state.basis_signature:
            current_signature = _basis_signature(G, nodes)
            if current_signature != fft_state.basis_signature:
                raise TNFRValueError(
                    "Graph topology or edge weights changed after the FFT basis was created."
                )

        count = len(nodes)
        eigenvalues = np.asarray(fft_state.eigenvalues)
        eigenvectors = np.asarray(fft_state.eigenvectors)
        spectral_epi = np.asarray(fft_state.spectral_epi)
        spectral_phase = np.asarray(fft_state.spectral_phase)
        if (
            eigenvalues.shape != (count,)
            or eigenvectors.shape != (count, count)
            or spectral_epi.shape != (count,)
            or spectral_phase.shape != (count,)
        ):
            raise TNFRValueError(
                "FFT state arrays must match the fixed full graph basis."
            )
        for name, values in (
            ("eigenvalues", eigenvalues),
            ("eigenvectors", eigenvectors),
            ("spectral EPI", spectral_epi),
            ("spectral phase", spectral_phase),
        ):
            if not np.all(np.isfinite(values)):
                raise TNFRValueError(f"FFT state {name} must be finite.")
        if fft_state.eigenbasis_digest:
            current_digest = _eigenbasis_digest(eigenvalues, eigenvectors)
            if current_digest != fft_state.eigenbasis_digest:
                raise TNFRValueError(
                    "FFT state eigensystem does not match its immutable basis snapshot."
                )
        gram = eigenvectors.T.conj() @ eigenvectors
        if not np.allclose(gram, np.eye(count), rtol=1e-10, atol=1e-12):
            raise TNFRValueError("FFT state eigenvectors must be orthonormal.")
        _finite_real(fft_state.time, "FFT state time")
        return state_nodes

    def _fixed_diffusion_operator(
        self,
        G: Any,
        fft_state: FFTDynamicsState,
        nodes: tuple[Any, ...],
    ) -> np.ndarray:
        """Validate the state's L_rw snapshot against the live fixed graph."""
        operator_nodes, current = structural_diffusion_operator(G)
        if tuple(operator_nodes) != nodes:
            raise TNFRValueError(
                "Diffusion operator node order differs from the FFT basis order."
            )
        current_values = np.asarray(current, dtype=float)
        stored = fft_state.diffusion_operator
        if stored is not None:
            stored_values = np.asarray(stored, dtype=float)
            if (
                stored_values.shape != current_values.shape
                or not np.all(np.isfinite(stored_values))
                or not np.array_equal(stored_values, current_values)
            ):
                raise TNFRValueError(
                    "FFT state diffusion operator does not match its graph basis."
                )
        if current_values.shape != (len(nodes), len(nodes)) or not np.all(
            np.isfinite(current_values)
        ):
            raise TNFRValueError(
                "FFT state diffusion operator must be a finite square matrix."
            )
        return current_values

    def _evolve_spectral_phase(
        self, G: Any, fft_state: FFTDynamicsState, dt: float, vf_spectral: np.ndarray
    ) -> np.ndarray:
        """Advance phase with the shared simultaneous U3-gated proposal."""
        phase_spatial = np.real_if_close(
            igft(fft_state.spectral_phase, fft_state.eigenvectors)
        )
        vf_spatial = np.real_if_close(
            igft(vf_spectral, fft_state.eigenvectors)
        )
        new_phase_spatial = propose_u3_gated_phase_step(
            G,
            fft_state.node_order or tuple(G.nodes()),
            phase_spatial,
            vf_spatial,
            dt=dt,
            coupling_strength=FFT_ENGINE_COUPLING_CANONICAL,
        )
        return gft(new_phase_spatial, fft_state.eigenvectors)

    def reconstruct_graph_from_fft(self, G: Any, fft_state: FFTDynamicsState) -> None:
        """
        Reconstruct spatial domain graph from FFT state.

        Updates graph node properties and instantaneous nodal telemetry from
        the spectral representation. Because the state stores no complete
        physical trajectory, reconstruction restarts EPI histories at one
        truthful final endpoint instead of fabricating a finite-difference
        Mutation certificate.
        """
        if not HAS_NETWORKX or G is None or len(fft_state.eigenvalues) == 0:
            return

        nodes = self._validate_fixed_basis(G, fft_state)
        final_time = _finite_real(fft_state.time, "FFT state time")
        current_time = _finite_real(G.graph.get("_t", 0.0), "graph runtime time")
        if final_time < current_time:
            raise TNFRValueError(
                "FFT reconstruction cannot move graph runtime time backwards."
            )

        epi_raw = igft(fft_state.spectral_epi, fft_state.eigenvectors)
        phase_raw = igft(fft_state.spectral_phase, fft_state.eigenvectors)
        diffusion_operator = self._fixed_diffusion_operator(G, fft_state, nodes)
        dnfr_raw = -(diffusion_operator @ epi_raw)
        vf_spatial = np.array(
            [
                _finite_real(
                    get_attr(G.nodes[node], ALIAS_VF, 1.0, strict=True),
                    f"node {node!r} structural frequency",
                )
                for node in nodes
            ],
            dtype=float,
        )
        if np.any(vf_spatial < 0.0):
            raise TNFRValueError("Structural frequency must be nonnegative.")
        depi_raw = vf_spatial * dnfr_raw

        def real_vector(values: Any, name: str) -> np.ndarray:
            values = np.asarray(values)
            if not np.all(np.isfinite(values)):
                raise TNFRValueError(f"Reconstructed {name} must be finite.")
            if np.iscomplexobj(values):
                imaginary = np.max(np.abs(values.imag), initial=0.0)
                scale = max(1.0, float(np.max(np.abs(values.real), initial=0.0)))
                if imaginary > 128.0 * np.finfo(float).eps * scale:
                    raise TNFRValueError(
                        f"Reconstructed {name} has no real nodal representation."
                    )
                values = values.real
            return np.asarray(values, dtype=float)

        epi_spatial = real_vector(epi_raw, "EPI")
        phase_spatial = real_vector(phase_raw, "phase")
        dnfr_spatial = real_vector(dnfr_raw, "DeltaNFR")
        depi_spatial = real_vector(depi_raw, "EPI derivative")

        old_epi = np.array(
            [
                _finite_epi(
                    get_attr(
                        G.nodes[node],
                        ALIAS_EPI,
                        0.0,
                        strict=True,
                        conv=lambda value: value,
                    ),
                    f"node {node!r} EPI",
                )
                for node in nodes
            ],
            dtype=float,
        )
        restart_histories = final_time != current_time or not np.array_equal(
            epi_spatial, old_epi
        )

        physical_histories: dict[Any, deque[tuple[float, float]]] = {}
        legacy_histories: dict[tuple[Any, str], Any] = {}
        if restart_histories:
            for index, node in enumerate(nodes):
                sample = float(epi_spatial[index])
                physical_histories[node] = deque(
                    ((final_time, sample),), maxlen=_PHYSICAL_HISTORY_MAXLEN
                )
                for key in ("epi_history", "_epi_history"):
                    if key in G.nodes[node]:
                        legacy_histories[node, key] = _single_sample_history(
                            G.nodes[node][key], sample
                        )

        transaction = GraphTransactionSnapshot(G)
        try:
            # All proposals above are validated before the first graph write.
            for index, node in enumerate(nodes):
                node_data = G.nodes[node]
                set_attr(node_data, ALIAS_EPI, float(epi_spatial[index]))
                set_theta(G, node, float(phase_spatial[index]))
                set_dnfr(G, node, float(dnfr_spatial[index]))
                set_attr(node_data, ALIAS_DEPI, float(depi_spatial[index]))
                for key in ALIAS_D2EPI:
                    node_data.pop(key, None)
                if restart_histories:
                    node_data["epi_time_history"] = physical_histories[node]
                    for key in ("epi_history", "_epi_history"):
                        if (node, key) in legacy_histories:
                            node_data[key] = legacy_histories[node, key]

            G.graph["_t"] = final_time
            if restart_histories and "_epi_hist" in G.graph:
                snapshot = {
                    node: float(epi_spatial[index]) for index, node in enumerate(nodes)
                }
                raw_history = G.graph["_epi_hist"]
                if isinstance(raw_history, deque):
                    G.graph["_epi_hist"] = deque((snapshot,), maxlen=raw_history.maxlen)
                else:
                    G.graph["_epi_hist"] = [snapshot]

            # Structural-field entries include EPI and phase but use rounded state
            # fingerprints; clearing them prevents small spectral updates from
            # reusing a stale read-out. The topology-only basis cache remains valid.
            self.structural_cache.clear_cache()
        except BaseException:
            transaction.restore(G)
            raise

    def run_fft_simulation(
        self, G: Any, num_steps: int, dt: float = 0.01, return_trajectory: bool = False
    ) -> dict[str, Any]:
        """
        Run a complete graph-spectral simulation.

        Returns performance metrics and final state.
        """
        if not HAS_NETWORKX or G is None:
            return {"status": "error", "message": "Invalid graph"}

        if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
            raise TNFRValueError("num_steps must be a nonnegative integer.")
        steps = int(num_steps)
        if steps < 0:
            raise TNFRValueError("num_steps must be a nonnegative integer.")
        dt_value = _finite_real(dt, "FFT integration dt")
        if steps and dt_value <= 0.0:
            raise TNFRValueError("FFT integration dt must be positive.")
        if not isinstance(return_trajectory, bool):
            raise TNFRValueError("return_trajectory must be boolean.")

        start_time = time.perf_counter()
        operation_start = self.fft_operations
        cache_hit_start = self.cache_hits
        max_nodal_residual = 0.0

        # Initialize FFT state
        fft_state = self.create_fft_state(G)

        # Store trajectory if requested
        trajectory = []
        if return_trajectory:
            # Record initial state
            initial_coherence = self._compute_spectral_coherence(G, fft_state)
            initial_phase_sync = self._compute_phase_sync(fft_state)
            trajectory.append(
                {
                    "time": fft_state.time,
                    "coherence": initial_coherence,
                    "phase_sync": initial_phase_sync,
                    "energy": self._compute_spectral_energy(fft_state),
                }
            )

        # Run simulation steps
        for step in range(steps):
            fft_state = self.fft_accelerated_step(G, fft_state, dt_value)
            max_nodal_residual = max(
                max_nodal_residual, fft_state.nodal_residual
            )

            completed_steps = step + 1
            if return_trajectory and (
                completed_steps % 10 == 0 or completed_steps == steps
            ):
                coherence = self._compute_spectral_coherence(G, fft_state)
                phase_sync = self._compute_phase_sync(fft_state)
                energy = self._compute_spectral_energy(fft_state)
                trajectory.append(
                    {
                        "time": fft_state.time,
                        "coherence": coherence,
                        "phase_sync": phase_sync,
                        "energy": energy,
                    }
                )

        # A zero-step diagnostic must leave graph state and histories untouched.
        if steps:
            self.reconstruct_graph_from_fft(G, fft_state)

        simulation_time = time.perf_counter() - start_time

        # Performance metrics
        results = {
            "status": "success",
            "simulation_time": simulation_time,
            "total_steps": steps,
            "fft_operations": self.fft_operations - operation_start,
            "cache_hits": self.cache_hits - cache_hit_start,
            "steps_per_second": (
                steps / simulation_time if simulation_time > 0 else 0
            ),
            "final_time": fft_state.time,
            "final_coherence": self._compute_spectral_coherence(G, fft_state),
            "final_phase_sync": self._compute_phase_sync(fft_state),
            "max_nodal_residual": max_nodal_residual,
            "integration_method": (
                fft_state.integration_method if steps else None
            ),
            "stability_not_certified": bool(
                steps and fft_state.stability_not_certified
            ),
            "stability_evidence": (
                "arbitrary_dt_without_step_size_certificate"
                if steps
                else "no_integration_step"
            ),
        }

        if return_trajectory:
            results["trajectory"] = trajectory

        return results

    def _compute_spectral_coherence(
        self, G: Any, fft_state: FFTDynamicsState
    ) -> float:
        """Compute canonical C(t) for the state's EPI-diffusion realization."""
        if len(fft_state.spectral_epi) == 0:
            return 0.0
        nodes = self._validate_fixed_basis(G, fft_state)
        epi = np.asarray(
            igft(fft_state.spectral_epi, fft_state.eigenvectors), dtype=float
        )
        operator = self._fixed_diffusion_operator(G, fft_state, nodes)
        pressure = -(operator @ epi)
        frequency = np.asarray(
            [
                _finite_real(
                    get_attr(G.nodes[node], ALIAS_VF, 1.0, strict=True),
                    f"node {node!r} structural frequency",
                )
                for node in nodes
            ],
            dtype=float,
        )
        if np.any(frequency < 0.0):
            raise TNFRValueError("Structural frequency must be nonnegative.")
        rate = frequency * pressure

        def mean_absolute(values: np.ndarray) -> float:
            magnitudes = np.abs(np.asarray(values, dtype=float))
            if magnitudes.size == 0:
                return 0.0
            scale = float(np.max(magnitudes))
            if scale == 0.0:
                return 0.0
            return scale * float(np.mean(magnitudes / scale))

        return float(
            structural_coherence(mean_absolute(pressure), mean_absolute(rate))
        )

    @staticmethod
    def _compute_phase_sync(fft_state: FFTDynamicsState) -> float:
        """Return the Kuramoto phase order parameter, distinct from C(t)."""
        if len(fft_state.spectral_phase) == 0:
            return 0.0
        phase_spatial = igft(fft_state.spectral_phase, fft_state.eigenvectors)
        z = np.mean(np.exp(1j * phase_spatial))
        return float(np.abs(z))

    def _compute_spectral_energy(self, fft_state: FFTDynamicsState) -> float:
        """Compute total energy in spectral domain."""
        if len(fft_state.spectral_epi) == 0:
            return 0.0

        # Energy = ||EPI||²
        energy = np.sum(np.abs(fft_state.spectral_epi) ** 2)
        return float(energy)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_operations": self.total_operations,
            "fft_operations": self.fft_operations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_operations),
            "cached_spectra": len(self._spectral_cache),
            "fft_usage_ratio": self.fft_operations / max(1, self.total_operations),
        }


# Factory functions for easy access
def create_fft_engine(**kwargs) -> FFTDynamicsEngine:
    """Create FFT dynamics engine."""
    return FFTDynamicsEngine(**kwargs)


def run_fft_optimized_simulation(
    G: Any, num_steps: int, dt: float = 0.01, **kwargs
) -> dict[str, Any]:
    """Convenience function for FFT-optimized simulation."""
    engine = create_fft_engine()
    return engine.run_fft_simulation(G, num_steps, dt, **kwargs)

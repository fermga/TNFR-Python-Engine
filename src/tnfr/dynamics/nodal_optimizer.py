"""Vectorized proposals for the isolated TNFR EPI-diffusion realization.

The optimizer computes the simultaneous node-space product

    EPI_next = EPI - dt * diag(nu_f) * L_rw * EPI

and delegates phase proposals to the shared U3-gated kernel. A spectral basis is
available as an explicit comparison API, but the nodal step itself uses the
dense live random-walk operator and makes no unmeasured speedup claim.
Because the step size is unrestricted, each proposal explicitly reports that
Euler stability has not been certified.
"""

import hashlib
import math
import weakref
from dataclasses import dataclass, replace
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..physics.structural_diffusion import structural_diffusion_operator
from ..operators.grammar_types import (
    COUPLING_RESONANCE,
    DESTABILIZERS,
    STABILIZERS,
    glyph_function_name,
)
from ..types import real_scalar_epi
from .phase_evolution import propose_u3_gated_phase_step


def _finite_real(value: Any, name: str) -> float:
    """Return a finite non-Boolean scalar."""
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
    """Read signed scalar EPI without collapsing a richer BEPI state."""
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

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import Spectral Analysis
try:
    from ..mathematics.spectral import get_laplacian_spectrum

    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False


# Operational engine-tuning knobs (not TNFR physics) → tnfr.constants.operational
from ..constants.operational import (
    NODAL_OPT_COUPLING_CANONICAL,
    NODAL_OPT_TARGET_DT_CANONICAL,
)


@dataclass(frozen=True, slots=True)
class NodalOptimizationState:
    """Immutable snapshot of one cached nodal operator state."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    vf_vector: np.ndarray
    node_index: Mapping[Any, int]
    last_topology_hash: str
    node_order: tuple[Any, ...] = ()
    diffusion_operator: np.ndarray | None = None


class NodalEvolutionProposal(dict):
    """Detached nodal proposal carrying an explicit stability-evidence marker."""

    __slots__ = ("metadata",)

    def __init__(self, values: Mapping[Any, tuple[float, float]]) -> None:
        super().__init__(values)
        self.metadata = MappingProxyType(
            {
                "integration_method": "explicit_euler",
                "stability_not_certified": True,
                "stability_evidence": "arbitrary_dt_without_step_size_certificate",
            }
        )

    @property
    def stability_not_certified(self) -> bool:
        """Report that this arbitrary-dt Euler proposal has no stability proof."""

        return True


class NodalEquationOptimizer:
    """
    Optimization engine for the canonical TNFR nodal equation.

    Leverages the mathematical structure of ∂EPI/∂t = νf · ΔNFR(t) to
    implement cache-coherent and vectorized optimizations.
    """

    def __init__(self, enable_cache: bool = True, max_cache_size: int = 1000):
        if not isinstance(enable_cache, bool):
            raise TNFRValueError("enable_cache must be Boolean.")
        if (
            isinstance(max_cache_size, bool)
            or not isinstance(max_cache_size, Integral)
            or int(max_cache_size) <= 0
        ):
            raise TNFRValueError("max_cache_size must be a positive integer.")
        self.enable_cache = enable_cache
        self.max_cache_size = int(max_cache_size)
        self._optimization_states: weakref.WeakKeyDictionary[
            Any, NodalOptimizationState
        ] = weakref.WeakKeyDictionary()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

    def get_graph_topology_hash(self, G: Any) -> str:
        """Fingerprint the labelled canonical diffusion operator and node order."""
        if not HAS_NETWORKX or G is None:
            return "no_graph"

        nodes, diffusion_operator = structural_diffusion_operator(G)
        node_records = tuple(
            (type(node).__module__, type(node).__qualname__, repr(node))
            for node in nodes
        )
        matrix = np.ascontiguousarray(diffusion_operator, dtype=float)
        payload = (
            bool(G.is_directed()),
            bool(G.is_multigraph()),
            node_records,
            matrix.shape,
            matrix.tobytes(),
        )
        return hashlib.sha256(
            repr(payload[:-1]).encode("utf-8") + payload[-1]
        ).hexdigest()

    def precompute_spectral_basis(
        self, G: Any, force_refresh: bool = False
    ) -> NodalOptimizationState:
        """Explicitly cache an L_sym comparison basis for an undirected graph."""

        state = self._prepare_nodal_state(
            G, force_refresh=force_refresh, include_spectral=True
        )
        return self._copy_state(state)

    @staticmethod
    def _readonly_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
        """Return a detached immutable numeric array."""

        result = np.array(values, dtype=dtype, copy=True)
        result.setflags(write=False)
        return result

    @classmethod
    def _copy_state(cls, state: NodalOptimizationState) -> NodalOptimizationState:
        """Detach every mutable member before exposing cached state."""

        return NodalOptimizationState(
            eigenvalues=cls._readonly_array(state.eigenvalues),
            eigenvectors=cls._readonly_array(state.eigenvectors),
            vf_vector=cls._readonly_array(state.vf_vector),
            node_index=MappingProxyType(dict(state.node_index)),
            last_topology_hash=state.last_topology_hash,
            node_order=tuple(state.node_order),
            diffusion_operator=(
                None
                if state.diffusion_operator is None
                else cls._readonly_array(state.diffusion_operator, dtype=float)
            ),
        )

    def _store_state(self, G: Any, state: NodalOptimizationState) -> None:
        """Store one state while enforcing the configured graph-count bound."""

        if G not in self._optimization_states:
            while len(self._optimization_states) >= self.max_cache_size:
                oldest = next(iter(self._optimization_states), None)
                if oldest is None:
                    break
                self._optimization_states.pop(oldest, None)
                self._cache_evictions += 1
        self._optimization_states[G] = state

    def _prepare_nodal_state(
        self,
        G: Any,
        *,
        force_refresh: bool = False,
        include_spectral: bool = False,
    ) -> NodalOptimizationState:
        """Cache the exact L_rw operator, optionally with an L_sym basis."""

        if not HAS_NETWORKX or G is None:
            return NodalOptimizationState(
                eigenvalues=self._readonly_array([]),
                eigenvectors=self._readonly_array(np.empty((0, 0))),
                vf_vector=self._readonly_array([]),
                node_index=MappingProxyType({}),
                last_topology_hash="",
            )

        topology_hash = self.get_graph_topology_hash(G)
        cached = self._optimization_states.get(G) if self.enable_cache else None
        spectral_needed = bool(
            include_spectral and HAS_SPECTRAL and not G.is_directed() and len(G)
        )
        if (
            not force_refresh
            and cached is not None
            and cached.last_topology_hash == topology_hash
            and (not spectral_needed or cached.eigenvalues.size == len(G))
        ):
            self._cache_hits += 1
            live_frequency = self._readonly_array(
                self._read_frequency_vector(G, cached.node_order), dtype=float
            )
            if not np.array_equal(live_frequency, cached.vf_vector):
                cached = replace(cached, vf_vector=live_frequency)
                self._optimization_states[G] = cached
            return cached
        if self.enable_cache:
            self._cache_misses += 1

        nodes, diffusion_operator = structural_diffusion_operator(G)
        node_order = tuple(nodes)
        node_index = {node: index for index, node in enumerate(node_order)}
        if spectral_needed:
            eigenvals, eigenvecs = get_laplacian_spectrum(
                G, operator="symmetric"
            )
        else:
            eigenvals = np.array([])
            eigenvecs = np.empty((len(node_order), 0), dtype=float)

        eigenvalue_array = np.asarray(eigenvals)
        eigenvector_array = np.asarray(eigenvecs)
        expected_vector_shape = (
            (len(node_order), len(node_order))
            if spectral_needed
            else (len(node_order), 0)
        )
        if eigenvalue_array.shape != ((len(node_order),) if spectral_needed else (0,)):
            raise TNFRValueError("Nodal spectral eigenvalue shape is inconsistent.")
        if eigenvector_array.shape != expected_vector_shape:
            raise TNFRValueError("Nodal spectral eigenvector shape is inconsistent.")
        if not np.all(np.isfinite(eigenvalue_array)) or not np.all(
            np.isfinite(eigenvector_array)
        ):
            raise TNFRValueError("Nodal spectral basis must remain finite.")
        diffusion_array = np.asarray(diffusion_operator, dtype=float)
        if diffusion_array.shape != (len(node_order), len(node_order)):
            raise TNFRValueError("Nodal diffusion operator shape is inconsistent.")
        if not np.all(np.isfinite(diffusion_array)):
            raise TNFRValueError("Nodal diffusion operator must remain finite.")

        opt_state = NodalOptimizationState(
            eigenvalues=self._readonly_array(eigenvalue_array),
            eigenvectors=self._readonly_array(eigenvector_array),
            vf_vector=self._readonly_array(
                self._read_frequency_vector(G, node_order), dtype=float
            ),
            node_index=MappingProxyType(node_index),
            last_topology_hash=topology_hash,
            node_order=node_order,
            diffusion_operator=self._readonly_array(diffusion_array, dtype=float),
        )
        if self.enable_cache:
            self._store_state(G, opt_state)
        return opt_state

    def compute_vectorized_nodal_evolution(
        self, G: Any, dt: float, target_time: float | None = None
    ) -> NodalEvolutionProposal:
        """Return one simultaneous nodal proposal from the current graph snapshot.

        The EPI channel is exactly
        x_next = x - dt * diag(nu_f) * L_rw * x. The method returns detached
        proposals and never commits them to the graph. Its metadata records
        that arbitrary-dt explicit Euler stability is not certified.
        """
        if not HAS_NETWORKX or G is None:
            return NodalEvolutionProposal({})

        dt_value = _finite_real(dt, "nodal integration dt")
        if dt_value <= 0.0:
            raise TNFRValueError("nodal integration dt must be positive.")
        if target_time is not None:
            # Compatibility metadata only: the one-step horizon is always dt.
            _finite_real(target_time, "target_time")

        opt_state = self._prepare_nodal_state(G, include_spectral=False)
        nodes = tuple(G.nodes())
        if nodes != opt_state.node_order:
            raise TNFRValueError("Graph node order changed during nodal proposal.")

        epi_vector = np.array(
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
        phase_vector = np.array(
            [
                _finite_real(
                    get_attr(G.nodes[node], ALIAS_THETA, 0.0, strict=True),
                    f"node {node!r} phase",
                )
                for node in nodes
            ],
            dtype=float,
        )
        vf_vector = np.asarray(opt_state.vf_vector, dtype=float)

        dnfr_vector = self._compute_epi_diffusion_pressure(
            G, opt_state, epi_vector, phase_vector
        )
        with np.errstate(over="ignore", invalid="ignore"):
            depi_dt = vf_vector * dnfr_vector
            new_epi_vector = epi_vector + dt_value * depi_dt
        if not np.all(np.isfinite(depi_dt)) or not np.all(
            np.isfinite(new_epi_vector)
        ):
            raise TNFRValueError("Nodal EPI proposal must remain finite.")
        new_phase_vector = self._predict_phase_evolution(
            G, opt_state, phase_vector, dt_value
        )
        if not np.all(np.isfinite(new_phase_vector)):
            raise TNFRValueError("Nodal phase proposal must remain finite.")

        return NodalEvolutionProposal(
            {
                node: (float(new_epi_vector[index]), float(new_phase_vector[index]))
                for index, node in enumerate(nodes)
            }
        )

    @staticmethod
    def _read_frequency_vector(
        G: Any, nodes: tuple[Any, ...] | list[Any]
    ) -> np.ndarray:
        """Read the live nonnegative capacity vector; it is not topology state."""
        values = np.array(
            [
                _finite_real(
                    get_attr(G.nodes[node], ALIAS_VF, 1.0, strict=True),
                    f"node {node!r} structural frequency",
                )
                for node in nodes
            ],
            dtype=float,
        )
        if np.any(values < 0.0):
            raise TNFRValueError("Structural frequency must be nonnegative.")
        return values

    def _compute_epi_diffusion_pressure(
        self,
        G: Any,
        opt_state: NodalOptimizationState,
        epi_vector: np.ndarray,
        phase_vector: np.ndarray,
    ) -> np.ndarray:
        """Return the exact isolated EPI pressure -L_rw @ EPI.

        phase_vector is retained in the private signature for compatibility
        with older instrumented callers; phase is a separate pressure channel.
        """
        del phase_vector
        operator = opt_state.diffusion_operator
        if operator is None:
            raise TNFRValueError("Nodal optimizer has no diffusion operator.")
        values = np.asarray(operator, dtype=float)
        if values.shape != (len(epi_vector), len(epi_vector)):
            raise TNFRValueError("Nodal diffusion operator shape is inconsistent.")
        current_nodes, current_operator = structural_diffusion_operator(G)
        if tuple(current_nodes) != opt_state.node_order:
            raise TNFRValueError("Graph node order changed during nodal proposal.")
        current_values = np.asarray(current_operator, dtype=float)
        if not np.array_equal(values, current_values):
            raise TNFRValueError(
                "Cached nodal diffusion operator does not match the live graph."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            pressure = -(current_values @ epi_vector)
        if not np.all(np.isfinite(pressure)):
            raise TNFRValueError("Nodal EPI pressure must remain finite.")
        return np.asarray(pressure, dtype=float)

    def _predict_phase_evolution(
        self,
        G: Any,
        opt_state: NodalOptimizationState,
        phase_vector: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Delegate to the shared simultaneous U3-gated phase proposal."""
        return propose_u3_gated_phase_step(
            G,
            tuple(G.nodes()),
            phase_vector,
            opt_state.vf_vector,
            dt=dt,
            coupling_strength=NODAL_OPT_COUPLING_CANONICAL,
        )

    def optimize_operator_sequence(
        self,
        G: Any,
        operator_sequence: list[str],
        target_dt: float = NODAL_OPT_TARGET_DT_CANONICAL,
    ) -> dict[str, Any]:
        """Return a static candidate plan without applying transformations.

        The compatibility method name is retained. Candidate labels describe
        possible shared work only; no speedup is reported without a measured
        baseline and no operator ordering is changed.
        """

        dt_value = _finite_real(target_dt, "target_dt")
        if dt_value <= 0.0:
            raise TNFRValueError("target_dt must be positive")
        if not HAS_NETWORKX or G is None:
            return {
                "optimizations": [],
                "candidates": [],
                "predicted_speedup": None,
                "speedup_evidence": "not_benchmarked",
                "sequence_length": len(operator_sequence),
                "spectral_opportunities": 0,
                "caching_opportunities": 0,
                "grammar_dependencies": [],
                "target_dt": dt_value,
            }

        canonical_ops = [
            glyph_function_name(operator, default=None)
            for operator in operator_sequence
        ]
        coherence_ops = [
            operator for operator in canonical_ops if operator == "coherence"
        ]
        phase_ops = [
            operator
            for operator in canonical_ops
            if operator in COUPLING_RESONANCE
        ]
        candidates: list[str] = []
        if len(coherence_ops) > 2:
            candidates.append("batch_coherence_readout_candidate")
        if len(phase_ops) > 1:
            candidates.append("shared_u3_phase_stage_candidate")

        stabilizers = [
            operator for operator in canonical_ops if operator in STABILIZERS
        ]
        destabilizers = [
            operator for operator in canonical_ops if operator in DESTABILIZERS
        ]
        grammar_dependencies = []
        if stabilizers and destabilizers:
            grammar_dependencies.append("preserve_u2_stabilizer_order")

        return {
            "optimizations": list(candidates),
            "candidates": candidates,
            "predicted_speedup": None,
            "speedup_evidence": "not_benchmarked",
            "sequence_length": len(operator_sequence),
            "spectral_opportunities": 0,
            "phase_stage_opportunities": len(phase_ops),
            "caching_opportunities": len(coherence_ops),
            "grammar_dependencies": grammar_dependencies,
            "target_dt": dt_value,
            "unknown_operators": [
                original
                for original, canonical in zip(operator_sequence, canonical_ops)
                if canonical is None
            ],
        }

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get statistics about current optimizations."""
        stats = {
            "cached_graphs": len(self._optimization_states),
            "cache_enabled": self.enable_cache,
            "spectral_available": HAS_SPECTRAL,
        }

        total = self._cache_hits + self._cache_misses
        stats.update(
            {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_evictions": self._cache_evictions,
                "cache_hit_rate": self._cache_hits / max(1, total),
                "max_cache_size": self.max_cache_size,
            }
        )
        return stats

    def clear_optimization_cache(self, graph_id: int | None = None) -> None:
        """Clear optimization caches."""
        if graph_id is not None:
            for graph in tuple(self._optimization_states):
                if id(graph) == graph_id:
                    self._optimization_states.pop(graph, None)
        else:
            self._optimization_states.clear()

        if graph_id is None:
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_evictions = 0


# Factory function for easy access
def create_nodal_optimizer(**kwargs) -> NodalEquationOptimizer:
    """Create a nodal equation optimizer with default settings."""
    return NodalEquationOptimizer(**kwargs)


# Integration with existing dynamics
def optimize_nodal_step(
    G: Any, dt: float, optimizer: NodalEquationOptimizer | None = None
) -> NodalEvolutionProposal:
    """
    Return the vectorized node-space EPI-diffusion proposal.

    The implementation multiplies the live dense L_rw operator in node space;
    an explicit spectral comparison basis is not precomputed on this path.
    """
    if optimizer is None:
        optimizer = create_nodal_optimizer()

    return optimizer.compute_vectorized_nodal_evolution(G, dt)

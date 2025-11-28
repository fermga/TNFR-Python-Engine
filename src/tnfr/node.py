"""Node utilities and structures for TNFR graphs."""

from __future__ import annotations

import copy
import math
from collections.abc import Hashable
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    SupportsFloat,
    TypeVar,
)
from weakref import WeakValueDictionary

import numpy as np

from .alias import (
    get_attr,
    get_attr_str,
    get_theta_attr,
    set_attr,
    set_attr_str,
    set_attr_generic,
    set_dnfr,
    set_theta,
    set_vf,
)
from .config import context_flags, get_flags
from .constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from .mathematics import (
    BasicStateProjector,
    CoherenceOperator,
    FrequencyOperator,
    HilbertSpace,
    StateProjector,
)
from .validation import NFRValidator
from .mathematics.operators_factory import (
    make_coherence_operator,
    make_frequency_operator,
)
from .mathematics.runtime import (
    coherence as runtime_coherence,
    frequency_positive as runtime_frequency_positive,
    normalized as runtime_normalized,
    stable_unitary as runtime_stable_unitary,
)
from .locking import get_lock
from .types import (
    CouplingWeight,
    DeltaNFR,
    EPIValue,
    NodeId,
    Phase,
    SecondDerivativeEPI,
    SenseIndex,
    StructuralFrequency,
    TNFRGraph,
    ZERO_BEPI_STORAGE,
    ensure_bepi,
    serialize_bepi,
)
from .utils import (
    cached_node_list,
    ensure_node_offset_map,
    get_logger,
    increment_edge_version,
    supports_add_edge,
)

T = TypeVar("T")

__all__ = ("NodeNX", "NodeProtocol", "add_edge")

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class AttrSpec:
    """Configuration required to expose a ``networkx`` node attribute.

    ``AttrSpec`` mirrors the defaults previously used by
    :func:`_nx_attr_property` and centralises the descriptor generation
    logic to keep a single source of truth for NodeNX attribute access.
    """

    aliases: tuple[str, ...]
    default: Any = 0.0
    getter: Callable[[MutableMapping[str, Any], tuple[str, ...], Any], Any] = get_attr
    setter: Callable[..., None] = set_attr
    to_python: Callable[[Any], Any] = float
    to_storage: Callable[[Any], Any] = float
    use_graph_setter: bool = False

    def build_property(self) -> property:
        """Create the property descriptor for ``NodeNX`` attributes."""

        def fget(instance: "NodeNX") -> T:
            return self.to_python(
                self.getter(instance.G.nodes[instance.n], self.aliases, self.default)
            )

        def fset(instance: "NodeNX", value: T) -> None:
            value = self.to_storage(value)
            if self.use_graph_setter:
                self.setter(instance.G, instance.n, value)
            else:
                self.setter(instance.G.nodes[instance.n], self.aliases, value)

        return property(fget, fset)


# Canonical adapters for BEPI storage ------------------------------------


def _epi_to_python(value: Any) -> EPIValue:
    if value is None:
        raise ValueError("EPI attribute is required for BEPI nodes")
    return ensure_bepi(value)


def _epi_to_storage(
    value: Any,
) -> Mapping[str, tuple[complex, ...] | tuple[float, ...]]:
    return serialize_bepi(value)


def _get_bepi_attr(mapping: Mapping[str, Any], aliases: tuple[str, ...], default: Any) -> Any:
    return get_attr(mapping, aliases, default, conv=lambda obj: obj)


def _set_bepi_attr(
    mapping: MutableMapping[str, Any], aliases: tuple[str, ...], value: Any
) -> Mapping[str, tuple[complex, ...] | tuple[float, ...]]:
    return set_attr_generic(mapping, aliases, value, conv=lambda obj: obj)


# Mapping of NodeNX attribute specifications used to generate property
# descriptors. Each entry defines the keyword arguments passed to
# ``AttrSpec.build_property`` for a given attribute name.
ATTR_SPECS: dict[str, AttrSpec] = {
    "EPI": AttrSpec(
        aliases=ALIAS_EPI,
        default=ZERO_BEPI_STORAGE,
        getter=_get_bepi_attr,
        to_python=_epi_to_python,
        to_storage=_epi_to_storage,
        setter=_set_bepi_attr,
    ),
    "vf": AttrSpec(aliases=ALIAS_VF, setter=set_vf, use_graph_setter=True),
    "theta": AttrSpec(
        aliases=ALIAS_THETA,
        getter=lambda mapping, _aliases, default: get_theta_attr(mapping, default),
        setter=set_theta,
        use_graph_setter=True,
    ),
    "Si": AttrSpec(aliases=ALIAS_SI),
    "epi_kind": AttrSpec(
        aliases=ALIAS_EPI_KIND,
        default="",
        getter=get_attr_str,
        setter=set_attr_str,
        to_python=str,
        to_storage=str,
    ),
    "dnfr": AttrSpec(aliases=ALIAS_DNFR, setter=set_dnfr, use_graph_setter=True),
    "d2EPI": AttrSpec(aliases=ALIAS_D2EPI),
}


def _add_edge_common(
    n1: NodeId,
    n2: NodeId,
    weight: CouplingWeight | SupportsFloat | str,
) -> Optional[CouplingWeight]:
    """Validate basic edge constraints.

    Returns the parsed weight if the edge can be added. ``None`` is returned
    when the edge should be ignored (e.g. self-connections).
    """

    if n1 == n2:
        return None

    weight = float(weight)
    if not math.isfinite(weight):
        raise ValueError("Edge weight must be a finite number")
    if weight < 0:
        raise ValueError("Edge weight must be non-negative")

    return weight


def add_edge(
    graph: TNFRGraph,
    n1: NodeId,
    n2: NodeId,
    weight: CouplingWeight | SupportsFloat | str,
    overwrite: bool = False,
) -> None:
    """Add an edge between ``n1`` and ``n2`` in a ``networkx`` graph."""

    weight = _add_edge_common(n1, n2, weight)
    if weight is None:
        return

    if not supports_add_edge(graph):
        raise TypeError("add_edge only supports networkx graphs")

    if graph.has_edge(n1, n2) and not overwrite:
        return

    graph.add_edge(n1, n2, weight=weight)
    increment_edge_version(graph)


class NodeProtocol(Protocol):
    """Minimal protocol for TNFR nodes."""

    EPI: EPIValue
    vf: StructuralFrequency
    theta: Phase
    Si: SenseIndex
    epi_kind: str
    dnfr: DeltaNFR
    d2EPI: SecondDerivativeEPI
    graph: MutableMapping[str, Any]

    def neighbors(self) -> Iterable[NodeProtocol | Hashable]:
        """Iterate structural neighbours coupled to this node."""

        ...

    def _glyph_storage(self) -> MutableMapping[str, object]:
        """Return the mutable mapping storing glyph metadata."""

        ...

    def has_edge(self, other: "NodeProtocol") -> bool:
        """Return ``True`` when an edge connects this node to ``other``."""

        ...

    def add_edge(
        self,
        other: NodeProtocol,
        weight: CouplingWeight,
        *,
        overwrite: bool = False,
    ) -> None:
        """Couple ``other`` using ``weight`` optionally replacing existing links."""

        ...

    def offset(self) -> int:
        """Return the node offset index within the canonical ordering."""

        ...

    def all_nodes(self) -> Iterable[NodeProtocol]:
        """Iterate all nodes of the attached graph as :class:`NodeProtocol` objects."""

        ...


class NodeNX(NodeProtocol):
    """Adapter for ``networkx`` nodes."""

    # Statically defined property descriptors for ``NodeNX`` attributes.
    # Declaring them here makes the attributes discoverable by type checkers
    # and IDEs, avoiding the previous runtime ``setattr`` loop.
    EPI: EPIValue = ATTR_SPECS["EPI"].build_property()
    vf: StructuralFrequency = ATTR_SPECS["vf"].build_property()
    theta: Phase = ATTR_SPECS["theta"].build_property()
    Si: SenseIndex = ATTR_SPECS["Si"].build_property()
    epi_kind: str = ATTR_SPECS["epi_kind"].build_property()
    dnfr: DeltaNFR = ATTR_SPECS["dnfr"].build_property()
    d2EPI: SecondDerivativeEPI = ATTR_SPECS["d2EPI"].build_property()

    @staticmethod
    def _prepare_coherence_operator(
        operator: CoherenceOperator | None,
        *,
        dim: int | None = None,
        spectrum: Sequence[float] | np.ndarray | None = None,
        c_min: float | None = None,
    ) -> CoherenceOperator | None:
        if operator is not None:
            return operator

        spectrum_array: np.ndarray | None
        if spectrum is None:
            spectrum_array = None
        else:
            spectrum_array = np.asarray(spectrum, dtype=np.complex128)
            if spectrum_array.ndim != 1:
                raise ValueError("Coherence spectrum must be one-dimensional.")

        effective_dim = dim
        if spectrum_array is not None:
            spectrum_length = spectrum_array.shape[0]
            if effective_dim is None:
                effective_dim = int(spectrum_length)
            elif spectrum_length != int(effective_dim):
                raise ValueError("Coherence spectrum size mismatch with requested dimension.")

        if effective_dim is None:
            return None

        kwargs: dict[str, Any] = {}
        if spectrum_array is not None:
            kwargs["spectrum"] = spectrum_array
        if c_min is not None:
            kwargs["c_min"] = float(c_min)
        return make_coherence_operator(int(effective_dim), **kwargs)

    @staticmethod
    def _prepare_frequency_operator(
        operator: FrequencyOperator | None,
        *,
        matrix: Sequence[Sequence[complex]] | np.ndarray | None = None,
    ) -> FrequencyOperator | None:
        if operator is not None:
            return operator
        if matrix is None:
            return None
        return make_frequency_operator(np.asarray(matrix, dtype=np.complex128))

    def __init__(
        self,
        G: TNFRGraph,
        n: NodeId,
        *,
        state_projector: StateProjector | None = None,
        enable_math_validation: Optional[bool] = None,
        hilbert_space: HilbertSpace | None = None,
        coherence_operator: CoherenceOperator | None = None,
        coherence_dim: int | None = None,
        coherence_spectrum: Sequence[float] | np.ndarray | None = None,
        coherence_c_min: float | None = None,
        frequency_operator: FrequencyOperator | None = None,
        frequency_matrix: Sequence[Sequence[complex]] | np.ndarray | None = None,
        coherence_threshold: float | None = None,
        validator: NFRValidator | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.G: TNFRGraph = G
        self.n: NodeId = n
        self.graph: MutableMapping[str, Any] = G.graph
        self.state_projector: StateProjector = state_projector or BasicStateProjector()
        self._math_validation_override: Optional[bool] = enable_math_validation
        if enable_math_validation is None:
            effective_validation = get_flags().enable_math_validation
        else:
            effective_validation = bool(enable_math_validation)
        self.enable_math_validation: bool = effective_validation
        default_dimension = (
            G.number_of_nodes() if hasattr(G, "number_of_nodes") else len(tuple(G.nodes))
        )
        default_dimension = max(1, int(default_dimension))
        self.hilbert_space: HilbertSpace = hilbert_space or HilbertSpace(default_dimension)
        if coherence_operator is not None and (
            coherence_dim is not None
            or coherence_spectrum is not None
            or coherence_c_min is not None
        ):
            raise ValueError("Provide either a coherence operator or factory parameters, not both.")
        if frequency_operator is not None and frequency_matrix is not None:
            raise ValueError("Provide either a frequency operator or frequency matrix, not both.")

        self.coherence_operator: CoherenceOperator | None = self._prepare_coherence_operator(
            coherence_operator,
            dim=coherence_dim,
            spectrum=coherence_spectrum,
            c_min=coherence_c_min,
        )
        self.frequency_operator: FrequencyOperator | None = self._prepare_frequency_operator(
            frequency_operator,
            matrix=frequency_matrix,
        )
        self.coherence_threshold: float | None = (
            float(coherence_threshold) if coherence_threshold is not None else None
        )
        self.validator: NFRValidator | None = validator
        self.rng: np.random.Generator | None = rng
        # Only add to default cache if not being created by from_graph
        if not G.graph.get("_creating_node", False):
            G.graph.setdefault("_node_cache", {})[n] = self

    def _glyph_storage(self) -> MutableMapping[str, Any]:
        return self.G.nodes[self.n]

    @classmethod
    def from_graph(cls, G: TNFRGraph, n: NodeId, *, use_weak_cache: bool = False) -> "NodeNX":
        """Return cached ``NodeNX`` for ``(G, n)`` with thread safety.

        Parameters
        ----------
        G : TNFRGraph
            The graph containing the node.
        n : NodeId
            The node identifier.
        use_weak_cache : bool, optional
            When True, use WeakValueDictionary for the node cache to allow
            automatic garbage collection of unused NodeNX instances. This is
            useful for ephemeral graphs where nodes are created temporarily
            and should be released when no longer referenced elsewhere.
            Default is False to maintain backward compatibility.

        Returns
        -------
        NodeNX
            The cached or newly created NodeNX instance for the specified node.

        Notes
        -----
        The weak cache mode trades off some cache retention for better memory
        behavior in scenarios with many short-lived graphs or when nodes are
        accessed infrequently. Use weak caching when:

        - Processing many ephemeral graphs sequentially
        - Working with large graphs where only subsets are actively used
        - Memory pressure is a concern and stale node objects should be released

        The default strong cache provides better performance for long-lived
        graphs with repeated node access patterns.
        """
        cache_key = "_node_cache_weak" if use_weak_cache else "_node_cache"

        # Fast path: lock-free read for cache hit (common case)
        cache = G.graph.get(cache_key)
        if cache is not None:
            node = cache.get(n)
            if node is not None:
                return node

        # Slow path: need to create node or initialize cache
        # Use per-node lock for finer granularity and reduced contention
        lock = get_lock(f"node_nx_{id(G)}_{n}_{cache_key}")
        with lock:
            # Double-check pattern: verify node still doesn't exist
            cache = G.graph.get(cache_key)
            if cache is not None:
                node = cache.get(n)
                if node is not None:
                    return node

            # Initialize cache if needed
            if cache is None:
                # Use a separate lock for cache initialization to avoid deadlocks
                graph_lock = get_lock(f"node_nx_cache_init_{id(G)}_{cache_key}")
                with graph_lock:
                    # Triple-check: another thread may have initialized
                    cache = G.graph.get(cache_key)
                    if cache is None:
                        if use_weak_cache:
                            cache = WeakValueDictionary()
                        else:
                            cache = {}
                        G.graph[cache_key] = cache

            # Check again after cache initialization
            node = cache.get(n)
            if node is not None:
                return node

            # Create node - use a sentinel to prevent __init__ from adding to cache
            G.graph["_creating_node"] = True
            try:
                node = cls(G, n)
            finally:
                G.graph.pop("_creating_node", None)

            # Add to requested cache only
            cache[n] = node

            return node

    def neighbors(self) -> Iterable[NodeId]:
        """Iterate neighbour identifiers (IDs).

        Wrap each resulting ID with :meth:`from_graph` to obtain the cached
        ``NodeNX`` instance when actual node objects are required.
        """
        return self.G.neighbors(self.n)

    def has_edge(self, other: NodeProtocol) -> bool:
        """Return ``True`` when an edge connects this node to ``other``."""

        if isinstance(other, NodeNX):
            return self.G.has_edge(self.n, other.n)
        raise NotImplementedError

    def add_edge(
        self,
        other: NodeProtocol,
        weight: CouplingWeight,
        *,
        overwrite: bool = False,
    ) -> None:
        """Couple ``other`` using ``weight`` optionally replacing existing links."""

        if isinstance(other, NodeNX):
            add_edge(
                self.G,
                self.n,
                other.n,
                weight,
                overwrite,
            )
        else:
            raise NotImplementedError

    def offset(self) -> int:
        """Return the cached node offset within the canonical ordering."""

        mapping = ensure_node_offset_map(self.G)
        return mapping.get(self.n, 0)

    def all_nodes(self) -> Iterable[NodeProtocol]:
        """Iterate all nodes of ``self.G`` as ``NodeNX`` adapters."""

        override = self.graph.get("_all_nodes")
        if override is not None:
            return override

        nodes = cached_node_list(self.G)
        return tuple(NodeNX.from_graph(self.G, v) for v in nodes)

    def run_sequence_with_validation(
        self,
        ops: Iterable[Callable[[TNFRGraph, NodeId], None]],
        *,
        projector: StateProjector | None = None,
        hilbert_space: HilbertSpace | None = None,
        coherence_operator: CoherenceOperator | None = None,
        coherence_dim: int | None = None,
        coherence_spectrum: Sequence[float] | np.ndarray | None = None,
        coherence_c_min: float | None = None,
        coherence_threshold: float | None = None,
        frequency_operator: FrequencyOperator | None = None,
        frequency_matrix: Sequence[Sequence[complex]] | np.ndarray | None = None,
        validator: NFRValidator | None = None,
        enforce_frequency_positivity: bool | None = None,
        enable_validation: bool | None = None,
        rng: np.random.Generator | None = None,
        log_metrics: bool = False,
    ) -> dict[str, Any]:
        """Run ``ops`` then return pre/post metrics with optional validation."""

        from .structural import run_sequence as structural_run_sequence

        projector = projector or self.state_projector
        hilbert = hilbert_space or self.hilbert_space

        effective_coherence = (
            self._prepare_coherence_operator(
                coherence_operator,
                dim=coherence_dim,
                spectrum=coherence_spectrum,
                c_min=(
                    coherence_c_min
                    if coherence_c_min is not None
                    else (
                        self.coherence_operator.c_min
                        if self.coherence_operator is not None
                        else None
                    )
                ),
            )
            if any(
                parameter is not None
                for parameter in (
                    coherence_operator,
                    coherence_dim,
                    coherence_spectrum,
                    coherence_c_min,
                )
            )
            else self.coherence_operator
        )
        effective_freq = (
            self._prepare_frequency_operator(
                frequency_operator,
                matrix=frequency_matrix,
            )
            if frequency_operator is not None or frequency_matrix is not None
            else self.frequency_operator
        )
        threshold = (
            float(coherence_threshold)
            if coherence_threshold is not None
            else self.coherence_threshold
        )
        validator = validator or self.validator
        rng = rng or self.rng

        if enable_validation is None:
            if self._math_validation_override is not None:
                should_validate = bool(self._math_validation_override)
            else:
                should_validate = bool(get_flags().enable_math_validation)
        else:
            should_validate = bool(enable_validation)
        self.enable_math_validation = should_validate

        enforce_frequency = (
            bool(enforce_frequency_positivity)
            if enforce_frequency_positivity is not None
            else bool(effective_freq is not None)
        )

        def _project(epi: float, vf: float, theta: float) -> np.ndarray:
            local_rng = None
            if rng is not None:
                bit_generator = rng.bit_generator
                cloned_state = copy.deepcopy(bit_generator.state)
                local_bit_generator = type(bit_generator)()
                local_bit_generator.state = cloned_state
                local_rng = np.random.Generator(local_bit_generator)
            vector = projector(
                epi=epi,
                nu_f=vf,
                theta=theta,
                dim=hilbert.dimension,
                rng=local_rng,
            )
            return np.asarray(vector, dtype=np.complex128)

        active_flags = get_flags()
        should_log_metrics = bool(log_metrics and active_flags.log_performance)

        def _metrics(state: np.ndarray, label: str) -> dict[str, Any]:
            metrics: dict[str, Any] = {}
            with context_flags(log_performance=False):
                norm_passed, norm_value = runtime_normalized(state, hilbert, label=label)
                metrics["normalized"] = bool(norm_passed)
                metrics["norm"] = float(norm_value)
                if effective_coherence is not None and threshold is not None:
                    coh_passed, coh_value = runtime_coherence(
                        state, effective_coherence, threshold, label=label
                    )
                    metrics["coherence"] = bool(coh_passed)
                    metrics["coherence_expectation"] = float(coh_value)
                    metrics["coherence_threshold"] = float(threshold)
                if effective_freq is not None:
                    freq_summary = runtime_frequency_positive(
                        state,
                        effective_freq,
                        enforce=enforce_frequency,
                        label=label,
                    )
                    metrics["frequency_positive"] = bool(freq_summary["passed"])
                    metrics["frequency_expectation"] = float(freq_summary["value"])
                    metrics["frequency_projection_passed"] = bool(freq_summary["projection_passed"])
                    metrics["frequency_spectrum_psd"] = bool(freq_summary["spectrum_psd"])
                    metrics["frequency_spectrum_min"] = float(freq_summary["spectrum_min"])
                    metrics["frequency_enforced"] = bool(freq_summary["enforce"])
                if effective_coherence is not None:
                    unitary_passed, unitary_norm = runtime_stable_unitary(
                        state,
                        effective_coherence,
                        hilbert,
                        label=label,
                    )
                    metrics["stable_unitary"] = bool(unitary_passed)
                    metrics["stable_unitary_norm_after"] = float(unitary_norm)
            if should_log_metrics:
                LOGGER.debug(
                    "node_metrics.%s normalized=%s coherence=%s frequency_positive=%s stable_unitary=%s coherence_expectation=%s frequency_expectation=%s",
                    label,
                    metrics.get("normalized"),
                    metrics.get("coherence"),
                    metrics.get("frequency_positive"),
                    metrics.get("stable_unitary"),
                    metrics.get("coherence_expectation"),
                    metrics.get("frequency_expectation"),
                )
            return metrics

        pre_state = _project(self.EPI, self.vf, self.theta)
        pre_metrics = _metrics(pre_state, "pre")

        structural_run_sequence(self.G, self.n, ops)

        post_state = _project(self.EPI, self.vf, self.theta)
        post_metrics = _metrics(post_state, "post")

        validation_summary: dict[str, Any] | None = None
        if should_validate:
            validator_instance = validator
            if validator_instance is None:
                if effective_coherence is None:
                    raise ValueError("Validation requires a coherence operator.")
                validator_instance = NFRValidator(
                    hilbert,
                    effective_coherence,
                    threshold if threshold is not None else 0.0,
                    frequency_operator=effective_freq,
                )
            outcome = validator_instance.validate(
                post_state,
                enforce_frequency_positivity=enforce_frequency,
            )
            validation_summary = {
                "passed": bool(outcome.passed),
                "summary": outcome.summary,
                "report": validator_instance.report(outcome),
            }

        result = {
            "pre_state": pre_state,
            "post_state": post_state,
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "validation": validation_summary,
        }
        # Preserve legacy structure for downstream compatibility.
        result["pre"] = {"state": pre_state, "metrics": pre_metrics}
        result["post"] = {"state": post_state, "metrics": post_metrics}
        return result

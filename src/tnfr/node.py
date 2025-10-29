"""Node utilities and structures for TNFR graphs."""

from __future__ import annotations

import copy
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    SupportsFloat,
    TypeVar,
)

import numpy as np

from .alias import (
    get_attr,
    get_attr_str,
    get_theta_attr,
    set_attr,
    set_attr_str,
    set_dnfr,
    set_theta,
    set_vf,
)
from .config import get_flags
from .constants import get_aliases
from .mathematics import (
    BasicStateProjector,
    CoherenceOperator,
    FrequencyOperator,
    HilbertSpace,
    NFRValidator,
    StateProjector,
)
from .mathematics.operators_factory import as_coherence_operator, as_frequency_operator
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
)
from .utils import (
    cached_node_list,
    ensure_node_offset_map,
    increment_edge_version,
    supports_add_edge,
)

ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_THETA = get_aliases("THETA")
ALIAS_SI = get_aliases("SI")
ALIAS_EPI_KIND = get_aliases("EPI_KIND")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_D2EPI = get_aliases("D2EPI")

T = TypeVar("T")

__all__ = ("NodeNX", "NodeProtocol", "add_edge")


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


# Mapping of NodeNX attribute specifications used to generate property
# descriptors. Each entry defines the keyword arguments passed to
# ``AttrSpec.build_property`` for a given attribute name.
ATTR_SPECS: dict[str, AttrSpec] = {
    "EPI": AttrSpec(aliases=ALIAS_EPI),
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

    def __init__(
        self,
        G: TNFRGraph,
        n: NodeId,
        *,
        state_projector: StateProjector | None = None,
        enable_math_validation: Optional[bool] = None,
        hilbert_space: HilbertSpace | None = None,
        coherence_operator: CoherenceOperator | Iterable[Sequence[complex]] | Sequence[complex] | np.ndarray | None = None,
        coherence_operator_params: Mapping[str, Any] | None = None,
        frequency_operator: FrequencyOperator | Iterable[Sequence[complex]] | Sequence[complex] | np.ndarray | None = None,
        frequency_operator_params: Mapping[str, Any] | None = None,
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
        self.coherence_operator: CoherenceOperator | None = as_coherence_operator(
            coherence_operator, coherence_operator_params
        )
        self.frequency_operator: FrequencyOperator | None = as_frequency_operator(
            frequency_operator, frequency_operator_params
        )
        self.coherence_threshold: float | None = (
            float(coherence_threshold) if coherence_threshold is not None else None
        )
        self.validator: NFRValidator | None = validator
        self.rng: np.random.Generator | None = rng
        G.graph.setdefault("_node_cache", {})[n] = self

    def _glyph_storage(self) -> MutableMapping[str, Any]:
        return self.G.nodes[self.n]

    @classmethod
    def from_graph(cls, G: TNFRGraph, n: NodeId) -> "NodeNX":
        """Return cached ``NodeNX`` for ``(G, n)`` with thread safety."""
        lock = get_lock(f"node_nx_cache_{id(G)}")
        with lock:
            cache = G.graph.setdefault("_node_cache", {})
            node = cache.get(n)
            if node is None:
                node = cls(G, n)
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
        coherence_operator: CoherenceOperator | Iterable[Sequence[complex]] | Sequence[complex] | np.ndarray | None = None,
        coherence_operator_params: Mapping[str, Any] | None = None,
        coherence_threshold: float | None = None,
        freq_op: FrequencyOperator | Iterable[Sequence[complex]] | Sequence[complex] | np.ndarray | None = None,
        frequency_operator_params: Mapping[str, Any] | None = None,
        validator: NFRValidator | None = None,
        enforce_frequency_positivity: bool | None = None,
        enable_validation: bool | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Run ``ops`` then return pre/post metrics with optional validation."""

        from .structural import run_sequence as structural_run_sequence

        projector = projector or self.state_projector
        hilbert = hilbert_space or self.hilbert_space

        effective_coherence = (
            as_coherence_operator(coherence_operator, coherence_operator_params)
            if coherence_operator is not None
            else self.coherence_operator
        )
        effective_freq = (
            as_frequency_operator(freq_op, frequency_operator_params)
            if freq_op is not None
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
            vector = projector(epi, vf, theta, hilbert.dimension, rng=local_rng)
            return np.asarray(vector, dtype=np.complex128)

        def _metrics(state: np.ndarray, label: str) -> dict[str, Any]:
            metrics: dict[str, Any] = {}
            norm_passed, norm_value = runtime_normalized(state, hilbert, label=label)
            metrics["normalized"] = {"passed": norm_passed, "norm": norm_value}
            if effective_coherence is not None and threshold is not None:
                coh_passed, coh_value = runtime_coherence(
                    state, effective_coherence, threshold, label=label
                )
                metrics["coherence"] = {
                    "passed": coh_passed,
                    "value": coh_value,
                    "threshold": threshold,
                }
            if effective_freq is not None:
                metrics["frequency"] = runtime_frequency_positive(
                    state,
                    effective_freq,
                    enforce=enforce_frequency,
                    label=label,
                )
            if effective_coherence is not None:
                unitary_passed, unitary_norm = runtime_stable_unitary(
                    state,
                    effective_coherence,
                    hilbert,
                    label=label,
                )
                metrics["unitary"] = {
                    "passed": unitary_passed,
                    "norm_after": unitary_norm,
                }
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
            success, summary = validator_instance.validate_state(
                post_state,
                enforce_frequency_positivity=enforce_frequency,
            )
            validation_summary = {
                "passed": bool(success),
                "summary": summary,
                "report": validator_instance.report(summary),
            }

        return {
            "pre": {"state": pre_state, "metrics": pre_metrics},
            "post": {"state": post_state, "metrics": post_metrics},
            "validation": validation_summary,
        }

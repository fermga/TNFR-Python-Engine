"""Detached non-EPI forcing from the bounded default NumPy pressure branch.

The nonlinear phase gradient is materialized by the existing fused kernel.
Its binary64 values and the actual channel coefficients define an exact
rational reference; pressure assembly error and stored operator pressure are
reported separately. This is a read-out, not an execution or solver certificate.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_THETA
from ..dynamics import dnfr, fused_dnfr
from ..metrics.common import merge_and_normalize_weights
from ._cycle_algebra import Vector, ordered_vector
from .support_transport import (
    SupportTransportSnapshot,
    _rebuild,
    observe_support_transport,
)

__all__ = [
    "NonEpiForcingObservation",
    "capture_non_epi_forcing",
    "decompose_non_epi_forcing",
]

_CHANNELS = ("phase", "epi", "vf", "topo")
_MAX_SUPPORT_ENTRIES = 100


@dataclass(frozen=True)
class NonEpiForcingObservation:
    """Exact reference to represented channels on one detached graph snapshot.

    ``forcing`` uses exact products of represented coefficients, the
    materialized unit phase gradient, and exact support capacity/topology
    gradients. It is independent of stored DeltaNFR. ``kernel_pressure_defect``
    compares the complete fresh kernel with that reference; the separate
    ``stored_pressure_residual`` includes prior operator writes or stale input.
    No causal provenance, continuous transcendental accuracy or future state
    is certified by these public data fields.
    """

    snapshot: SupportTransportSnapshot
    phase: Vector
    epi_weight: Fraction
    forcing: Vector
    phase_gradient: Vector
    normalized_weights: tuple[tuple[str, Fraction], ...]
    full_kernel_pressure: Vector
    kernel_pressure_defect: Vector
    stored_pressure_residual: Vector


def _represented_vector(values, label):
    return tuple(
        finite_represented_real(value, f"{label}[{index}]")[1]
        for index, value in enumerate(values)
    )


def _runtime_weights(graph):
    # The pressure engine reuses this cache verbatim. Metadata's weights_norm
    # is normalized again and is not the source of the executed coefficients.
    configured = graph.graph.get("_dnfr_weights")
    if configured is None:
        configured = merge_and_normalize_weights(
            graph,
            "DNFR_WEIGHTS",
            _CHANNELS,
            default=0.0,
        )
    if not isinstance(configured, Mapping):
        raise TypeError("cached DeltaNFR weights must be a mapping")
    weights = []
    for channel in _CHANNELS:
        value, exact = finite_represented_real(
            configured.get(channel, 0.0),
            f"DeltaNFR {channel} weight",
        )
        if value < 0.0:
            raise ValueError("DeltaNFR weights must be nonnegative")
        weights.append((channel, exact))
    return tuple(weights)


def _forcing_components(snapshot, weights, phase_gradient):
    return tuple(
        (name, tuple(weights[name] * value for value in values))
        for name, values in (
            ("phase", phase_gradient),
            ("vf", snapshot.capacity_gradient),
            ("topo", snapshot.topology_gradient),
        )
    )


def decompose_non_epi_forcing(observation) -> tuple:
    """Validate and split a detached capture's F without another kernel call.

    The phase gradient is a supplied represented coefficient. This arithmetic
    check cannot authenticate its graph, phase-resultant branch or causal
    provenance. Support-gradient caches are rebuilt. Kernel rounding and
    stale stored pressure remain separate from these exact forcing channels.
    """
    if type(observation) is not NonEpiForcingObservation:
        raise TypeError("forcing decomposition requires a NonEpiForcingObservation")
    source = _rebuild(observation.snapshot)
    pairs = tuple(observation.normalized_weights)
    if tuple(name for name, _ in pairs) != _CHANNELS:
        raise ValueError(
            "forcing weights require the ordered phase/epi/vf/topo channels"
        )
    weights = {
        name: exact_or_represented_real(value, f"{name} weight")
        for name, value in pairs
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("forcing weights must be nonnegative")
    if weights["epi"] != exact_or_represented_real(
        observation.epi_weight, "epi_weight"
    ):
        raise ValueError("EPI coefficient differs from the captured weights")
    phase = ordered_vector(observation.phase_gradient, "phase gradient")
    if len(phase) != len(source.nodes):
        raise ValueError("phase gradient must match the captured node order")
    components = _forcing_components(source, weights, phase)
    forcing = tuple(
        sum((values[i] for _, values in components), Fraction(0))
        for i in range(len(source.nodes))
    )
    if forcing != ordered_vector(observation.forcing, "forcing"):
        raise ValueError(
            "captured forcing differs from its exact channel decomposition"
        )
    return components


def capture_non_epi_forcing(G) -> NonEpiForcingObservation:
    """Read F = w_phase*g_phase + w_vf*g_vf + w_topo*g_topo without writes.

    Scope is effective symmetric conductance and the default NumPy, non-JIT
    branch with at most 100 directed unique-support entries. A disabled NumPy
    branch or configured custom pressure callback is rejected. The fallback
    branch's small-resultant phase policy and Numba fastmath are not identified
    with this kernel. No live EPI, pressure, configuration or cache is changed.

    Phase uses the actual unweighted neighbor phasors, including zero-weight
    support edges, in the runtime's neighbor insertion order. Neither channel
    removal nor this read-out renormalizes an already materialized weight mix.
    The represented phase gradient is a model coefficient, not an exact-real
    evaluation of trigonometric functions or a linearized phase law.
    """
    if dnfr.np is None or fused_dnfr.np is None:
        raise ValueError("forcing read-out requires the NumPy pressure branch")
    if G.graph.get("vectorized_dnfr") is False:
        raise ValueError("forcing read-out excludes the fallback pressure branch")
    callback = G.graph.get("compute_delta_nfr")
    if callback is not None and callback is not dnfr.default_compute_delta_nfr:
        raise ValueError("forcing read-out requires the default pressure callback")

    snapshot = observe_support_transport(G)
    nodes = snapshot.nodes
    indices = {node: index for index, node in enumerate(nodes)}
    edge_src, edge_dst = dnfr._build_edge_index_arrays(G, nodes, indices)
    if len(edge_src) > _MAX_SUPPORT_ENTRIES:
        raise ValueError(
            "forcing read-out supports at most 100 directed support entries"
        )
    phase = _represented_vector(
        (
            get_attr(G.nodes[node], ALIAS_THETA, 0.0, conv=lambda v: v, strict=True)
            for node in nodes
        ),
        "phase",
    )
    weights = _runtime_weights(G)
    exact_weights = dict(weights)
    array = dnfr.np.asarray
    arguments = {
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "phase": array(tuple(map(float, phase)), dtype=float),
        "epi": array(tuple(map(float, snapshot.epi)), dtype=float),
        "vf": array(tuple(map(float, snapshot.capacity)), dtype=float),
        "accumulate_both_directions": False,
        "use_jit": False,
    }
    phase_gradient = _represented_vector(
        fused_dnfr.compute_fused_gradients_symmetric(
            **arguments,
            weights={"w_phase": 1.0},
        ),
        "unit phase gradient",
    )
    edge_weight = dnfr._build_edge_weight_array(G, nodes, edge_src, edge_dst)
    full_kernel_pressure = _represented_vector(
        fused_dnfr.compute_fused_gradients_symmetric(
            **arguments,
            weights={f"w_{key}": float(value) for key, value in weights},
            edge_weight=edge_weight,
        ),
        "fresh kernel pressure",
    )
    components = _forcing_components(snapshot, exact_weights, phase_gradient)
    forcing = tuple(
        sum((values[i] for _, values in components), Fraction(0))
        for i in range(len(nodes))
    )
    epi_weight = exact_weights["epi"]
    kernel_defect = tuple(
        actual - epi_weight * epi_i - force_i
        for actual, epi_i, force_i in zip(
            full_kernel_pressure,
            snapshot.epi_gradient,
            forcing,
            strict=True,
        )
    )
    stored_residual = tuple(
        stored - fresh
        for stored, fresh in zip(
            snapshot.stored_pressure,
            full_kernel_pressure,
            strict=True,
        )
    )
    return NonEpiForcingObservation(
        snapshot,
        phase,
        epi_weight,
        forcing,
        phase_gradient,
        weights,
        full_kernel_pressure,
        kernel_defect,
        stored_residual,
    )

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
from ._cycle_algebra import Vector, dot, ordered_vector
from .support_transport import (
    SupportTransportSnapshot,
    _rebuild,
    _support_components,
    _support_gradient,
    observe_support_transport,
)

__all__ = [
    "NonEpiForcingObservation",
    "ForcingDirichletBalance",
    "capture_non_epi_forcing",
    "decompose_non_epi_forcing",
    "observe_forcing_dirichlet_balance",
    "ForcingCapacityDifference",
    "observe_forcing_capacity_difference",
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


@dataclass(frozen=True)
class ForcingDirichletBalance:
    """Exact channel rates for E_D=x^T B x/2, not a trajectory certificate.

    Every pressure contribution is paired with diag(capacity)*B*x. Negative
    diffusion and signed source work concern this Dirichlet energy, not the
    tetrad energy or a selected regional identity. Fresh and stored totals
    describe different pressure inputs at the same supplied state.
    """

    source: SupportTransportSnapshot
    diffusion_rate: Fraction
    channel_rates: tuple[tuple[str, Fraction], ...]
    source_rate: Fraction
    modeled_rate: Fraction
    kernel_defect_rate: Fraction
    fresh_rate: Fraction
    stored_residual_rate: Fraction
    stored_rate: Fraction
    identity_residual: Fraction


def _represented_vector(values, label):
    return tuple(
        finite_represented_real(value, f"{label}[{index}]")[1]
        for index, value in enumerate(values)
    )


def _runtime_weights(graph):
    # Read the same effective mix as the next engine refresh, without changing
    # its cache. Metadata's normalized proportions are not executed coefficients.
    configured = dnfr._resolve_dnfr_weights(graph)
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


def _validated_forcing_decomposition(observation):
    """Rebuild one shared structural reference and validate its channels."""
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
    return source, components, weights


def decompose_non_epi_forcing(observation) -> tuple:
    """Validate and split a detached capture's F without another kernel call.

    The phase gradient is a supplied represented coefficient. This arithmetic
    check cannot authenticate its graph, phase-resultant branch or causal
    provenance. Support-gradient caches are rebuilt. Kernel rounding and
    stale stored pressure remain separate from these exact forcing channels.
    """
    return _validated_forcing_decomposition(observation)[1]


def _validated_pressure_realization(observation):
    """Shared exact channel assembly and fresh/stored defect reconstruction."""
    source, channels, weights = _validated_forcing_decomposition(observation)
    full = ordered_vector(observation.full_kernel_pressure, "full pressure")
    if len(full) != len(source.nodes):
        raise ValueError("full pressure must match the captured node order")
    modeled = tuple(
        weights["epi"] * epi + sum((row[i] for _, row in channels), Fraction(0))
        for i, epi in enumerate(source.epi_gradient)
    )
    kernel = tuple(a - b for a, b in zip(full, modeled, strict=True))
    stored = tuple(a - b for a, b in zip(source.stored_pressure, full, strict=True))
    return source, channels, weights, modeled, full, kernel, stored


def observe_forcing_dirichlet_balance(observation) -> ForcingDirichletBalance:
    """Pair existing pressure channels with the heterogeneous nodal mobility.

    Reuse the detached forcing decomposition without another graph or phase
    kernel call. Rebuild structural caches and recompute pressure defects from
    the supplied full pressure vector; cached defect fields are not evidence.
    Materialized phase/full-pressure inputs remain caller data, so these exact
    identities neither authenticate a kernel call nor establish causal history.

    The modeled rate is diffusion plus phase/capacity/topology source work.
    Adding assembly error gives the fresh rate; adding stored-pressure residual
    work gives the held rate. An observed Euler interval keeps its quadratic
    and endpoint defects in ``observe_support_transport_euler`` separately.
    """
    source, channels, weights, _, full_pressure, kernel_defect, stored_residual = (
        _validated_pressure_realization(observation)
    )
    epi_pressure = tuple(weights["epi"] * value for value in source.epi_gradient)
    score = tuple(
        gradient * capacity
        for gradient, capacity in zip(
            source.dirichlet_gradient, source.capacity, strict=True
        )
    )
    diffusion_rate = dot(score, epi_pressure)
    channel_rates = tuple((name, dot(score, values)) for name, values in channels)
    source_rate = sum((rate for _, rate in channel_rates), Fraction(0))
    modeled_rate = diffusion_rate + source_rate
    kernel_defect_rate = dot(score, kernel_defect)
    fresh_rate = dot(score, full_pressure)
    stored_residual_rate = dot(score, stored_residual)
    stored_rate = source.energy_rate
    return ForcingDirichletBalance(
        source=source,
        diffusion_rate=diffusion_rate,
        channel_rates=channel_rates,
        source_rate=source_rate,
        modeled_rate=modeled_rate,
        kernel_defect_rate=kernel_defect_rate,
        fresh_rate=fresh_rate,
        stored_residual_rate=stored_residual_rate,
        stored_rate=stored_rate,
        identity_residual=(
            stored_rate - modeled_rate - kernel_defect_rate - stored_residual_rate
        ),
    )


@dataclass(frozen=True)
class ForcingCapacityDifference:
    """Detached pressure differences at matching form and lifted phase geometry."""

    before: SupportTransportSnapshot
    after: SupportTransportSnapshot
    epi_offset: Fraction
    phase_offset: Fraction
    capacity_change: Vector
    capacity_pressure_change: Vector
    phase_realization_change: Vector
    modeled_pressure_before: Vector
    modeled_pressure_after: Vector
    kernel_defect_change: Vector
    fresh_pressure_change: Vector
    stored_residual_change: Vector
    stored_pressure_change: Vector
    identity_residual: Vector
    stored_identity_residual: Vector
    support_components: tuple
    component_capacity_offsets: tuple[Fraction | None, ...]


def observe_forcing_capacity_difference(before, after) -> ForcingCapacityDifference:
    """Compare capacities with fixed support, weights and relative form/phase.

    Require positive capacities, reciprocal support, an exact common EPI
    offset and an exact common raw phase offset. Differently wrapped lifts are
    not silently identified. The exact capacity contribution uses the shared
    UNWEIGHTED support gradient, even on unequal transport conductance.

    A true common rotation preserves the ideal regular phase source. Captured
    binary64 phase gradients need not agree; their weighted difference is kept
    separately, as are fresh-kernel assembly and stored-pressure defects. Public
    snapshots are rebuilt, but phase/full-pressure payloads remain caller data.
    No graph read, new pressure call, execution or stationary state is certified.

    With an unchanged ideal regular phase source, if both ideal pressures
    vanish and the capacity-channel weight is positive, the capacity change
    must be constant on each connected support component.
    Zero channel weight, zero capacity, Gamma, clipping and EPI drift cannot be
    treated as that stationary inference. This observer does not read Gamma or
    clipping configuration; those remain separate model hypotheses.
    """
    left, channels0, weights, p0, fresh0, kernel0, stored0 = (
        _validated_pressure_realization(before)
    )
    right, channels1, weights_after, p1, fresh1, kernel1, stored1 = (
        _validated_pressure_realization(after)
    )
    if (
        not left.nodes
        or left.nodes != right.nodes
        or left.conductance != right.conductance
        or left.support_neighbors != right.support_neighbors
        or weights != weights_after
    ):
        raise ValueError(
            "capacity comparison requires fixed nodes, support and weights"
        )
    rows = left.support_neighbors
    if any(i not in rows[j] for i, row in enumerate(rows) for j in row):
        raise ValueError("capacity comparison requires reciprocal support")
    if any(value <= 0 for value in left.capacity + right.capacity):
        raise ValueError("stationary capacity comparison requires positive capacities")

    def difference(a, b):
        return tuple(y - x for x, y in zip(a, b, strict=True))

    def common_offset(a, b, label):
        delta = difference(a, b)
        if len(delta) != len(left.nodes) or any(value != delta[0] for value in delta):
            raise ValueError(f"{label} must differ by one exact common offset")
        return delta[0]

    epi_offset = common_offset(left.epi, right.epi, "EPI")
    phase_offset = common_offset(
        ordered_vector(before.phase, "before phase"),
        ordered_vector(after.phase, "after phase"),
        "phase",
    )
    delta = difference(left.capacity, right.capacity)
    capacity_pressure = tuple(
        weights["vf"] * value for value in _support_gradient(rows, delta)
    )
    phase_change = difference(dict(channels0)["phase"], dict(channels1)["phase"])
    kernel_change = difference(kernel0, kernel1)
    fresh_change = difference(fresh0, fresh1)
    stored_change = difference(stored0, stored1)
    stored_pressure = difference(left.stored_pressure, right.stored_pressure)
    residual = tuple(
        f - c - p - k
        for f, c, p, k in zip(
            fresh_change, capacity_pressure, phase_change, kernel_change, strict=True
        )
    )
    stored_residual = tuple(
        s - f - d
        for s, f, d in zip(stored_pressure, fresh_change, stored_change, strict=True)
    )
    components = _support_components(rows)
    offsets = tuple(
        (
            delta[component[0]]
            if all(delta[i] == delta[component[0]] for i in component)
            else None
        )
        for component in components
    )
    return ForcingCapacityDifference(
        left,
        right,
        epi_offset,
        phase_offset,
        delta,
        capacity_pressure,
        phase_change,
        p0,
        p1,
        kernel_change,
        fresh_change,
        stored_change,
        stored_pressure,
        residual,
        stored_residual,
        components,
        offsets,
    )


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

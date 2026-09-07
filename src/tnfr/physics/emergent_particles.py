"""Phase-winding classes on declared closed graph cycles.

The legacy module, ``EmergentParticle``, ``classify_particle``, and
``particle_class`` public names are retained for API compatibility. Prefer
``WindingSector``, ``classify_winding_sector``, and ``winding_class``. The
implemented result is narrower: a non-ambiguous phase map on a declared
oriented cycle has an integer winding class

    W = (1 / 2π) Σ_edges wrap(φ_j - φ_i) ∈ ℤ.

The sign depends on cycle orientation. Winding remains defined only while the
cycle exists and no edge difference crosses the wrap branch boundary. Its
preservation across a trajectory therefore requires those hypotheses and must
be measured; a static integer does not prove conservation under every canonical
operator.

The returned labels are purely topological: zero winding, unit winding, and
multi-winding. They do not encode exchange statistics, spin, charge
conjugation, or a physical particle species. The optional continuous bilinear
``Q`` and energy read-outs from :mod:`tnfr.physics.unified` are whole-graph
same-snapshot telemetry and are distinct from integer winding. They are
reported as unavailable, with ``None`` values and a reason, when any required
whole-graph phase or ``ΔNFR`` input is absent or invalid.

For explicit cycle-existence, branch-margin, U3-margin, and trajectory
certificates, use :mod:`tnfr.physics.winding_certificates`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable

import networkx as nx

from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA
from .unified import compute_energy_density, compute_historical_q_density
from .winding_certificates import certify_phase_winding

_TWO_PI = 2.0 * math.pi


def _finite_real(value: Any, *, name: str) -> float:
    """Return a finite real input while rejecting booleans explicitly."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_map_mean(values: dict[Any, float], *, name: str) -> float:
    """Return a finite mean for a nonempty whole-graph telemetry map."""
    if not values:
        raise ValueError(f"{name} telemetry is unavailable for the graph")
    scalars = [float(value) for value in values.values()]
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError(f"{name} telemetry must be finite")
    return math.fsum(scalars) / len(scalars)


def _finite_alias_value(
    data: dict[str, Any], aliases: Iterable[str], *, name: str
) -> float:
    """Read the first configured alias and require a finite real value."""
    for alias in aliases:
        if alias in data:
            return _finite_real(data[alias], name=name)
    raise ValueError(f"{name} is missing")


def _whole_graph_telemetry_issue(G: nx.Graph) -> str | None:
    """Return why whole-graph field telemetry cannot be evaluated."""
    for node, data in G.nodes(data=True):
        for aliases, field in (
            (ALIAS_THETA, "phase"),
            (ALIAS_DNFR, "delta_nfr"),
        ):
            try:
                _finite_alias_value(data, aliases, name=f"{field} at node {node!r}")
            except (TypeError, ValueError) as exc:
                return str(exc)
    return None


def _wrap_2pi(x: float) -> float:
    """Wrap an angle to [0, 2π)."""
    y = x % _TWO_PI
    if y < 0.0:
        y += _TWO_PI
    return float(y)


# ============================================================================
# CLOSED STRUCTURAL MANIFOLD WITH A WINDING PHASE FIELD
# ============================================================================


def winding_ring(n_nodes: int, winding: float, *, base_dnfr: float = 0.05) -> nx.Graph:
    """Build a closed 1D structural manifold (ring) carrying a phase field with
    the requested winding.

    The phase advances by ``winding`` full turns around the loop:
    φ_i = wrap(2π · winding · i / n). The node ordering 0,1,...,n-1,0 defines the
    closed loop. ΔNFR is set to a mild baseline; coherence = 1/(1+|ΔNFR|).

    ``winding`` may be non-integer on input. Sampling and shortest-arc closure
    produce an integer winding class whenever no edge lies on the wrap branch;
    the result need not equal the requested real value.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, Integral):
        raise TypeError("n_nodes must be an integer")
    n_nodes = int(n_nodes)
    if n_nodes < 3:
        raise ValueError("n_nodes must be >= 3 to form a closed loop")
    winding = _finite_real(winding, name="winding")
    base_dnfr = _finite_real(base_dnfr, name="base_dnfr")
    G = nx.cycle_graph(n_nodes)
    # Only winding modulo n_nodes affects the sampled phases. Reducing first
    # avoids overflow for otherwise valid, very large finite inputs.
    sampled_turns = math.fmod(winding, float(n_nodes))
    for i in range(n_nodes):
        ang = _TWO_PI * ((sampled_turns * i / n_nodes) % 1.0)
        phi = _wrap_2pi(ang)
        G.nodes[i]["theta"] = phi
        G.nodes[i]["phase"] = phi
        G.nodes[i]["delta_nfr"] = float(base_dnfr)
        G.nodes[i]["dnfr"] = float(base_dnfr)
        G.nodes[i]["coherence"] = 1.0 / (1.0 + abs(base_dnfr))
        G.nodes[i]["EPI"] = 1.0 / (1.0 + abs(base_dnfr))
        G.nodes[i]["nu_f"] = 1.0
    return G


# ============================================================================
# INTEGER PHASE-WINDING OBSERVATION
# ============================================================================


def _derive_simple_cycle_order(G: nx.Graph) -> tuple[Any, ...]:
    """Derive one traversal when the entire graph is a simple cycle."""
    nodes = tuple(G.nodes())
    n_nodes = len(nodes)
    error = "order is required unless the entire graph is a simple cycle"
    if n_nodes < 3 or G.number_of_edges() != n_nodes:
        raise ValueError(error)

    if G.is_directed():
        if any(G.in_degree(node) != 1 or G.out_degree(node) != 1 for node in nodes):
            raise ValueError(error)
        start = nodes[0]
        traversal: list[Any] = []
        seen: set[Any] = set()
        current = start
        for _ in range(n_nodes):
            if current in seen:
                raise ValueError(error)
            traversal.append(current)
            seen.add(current)
            successors = tuple(G.successors(current))
            if len(successors) != 1:
                raise ValueError(error)
            current = successors[0]
        if current != start or seen != set(nodes):
            raise ValueError(error)
        return tuple(traversal)

    if any(G.degree(node) != 2 for node in nodes) or not nx.is_connected(G):
        raise ValueError(error)
    start = nodes[0]
    first_neighbors = tuple(G.neighbors(start))
    if len(first_neighbors) != 2:
        raise ValueError(error)

    traversal = [start]
    seen = {start}
    previous, current = start, first_neighbors[0]
    while current != start and len(traversal) <= n_nodes:
        if current in seen:
            raise ValueError(error)
        traversal.append(current)
        seen.add(current)
        next_nodes = tuple(node for node in G.neighbors(current) if node != previous)
        if len(next_nodes) != 1:
            raise ValueError(error)
        previous, current = current, next_nodes[0]
    if current != start or seen != set(nodes) or len(traversal) != n_nodes:
        raise ValueError(error)
    return tuple(traversal)


def _resolve_cycle_order(
    G: nx.Graph, order: Iterable[Any] | None
) -> tuple[Any, ...]:
    """Use the declared traversal, or derive it for a whole simple cycle."""
    return tuple(order) if order is not None else _derive_simple_cycle_order(G)


def winding_number(
    G: nx.Graph, *, order: Iterable[Any] | None = None
) -> tuple[int, float]:
    """Compute winding on a declared, non-ambiguous oriented cycle.

    The historical two-value return is retained. Without ``order``, a traversal
    is derived only when the entire graph is one simple directed or undirected
    cycle; other graphs require an explicit oriented order. The automatically
    selected orientation of an undirected cycle follows graph iteration order,
    so pass ``order`` whenever the sign must be prescribed. Missing cycle edges
    and phase differences on the wrap branch raise ``ValueError`` because this
    API cannot represent an undefined winding. Use
    :func:`certify_phase_winding` when applicability metadata is needed.
    """
    nodes = _resolve_cycle_order(G, order)
    certificate = certify_phase_winding(G, nodes)
    if not certificate.is_defined:
        raise ValueError(f"winding is undefined: {certificate.reason}")
    assert certificate.winding is not None
    assert certificate.raw_winding is not None
    return certificate.winding, certificate.raw_winding


# ============================================================================
# TOPOLOGICAL CLASSIFICATION FROM MEASURED WINDING
# ============================================================================


@dataclass(frozen=True)
class EmergentParticle:
    """Legacy result name for a phase-winding sector and global telemetry."""

    winding: int  # integer winding class W
    raw_winding: float  # circulation / 2π (≈ W)
    chirality: int  # historical name for the orientation-sensitive sign(W)
    energy_density: float | None  # legacy whole-graph mean ℰ snapshot
    charge_density_mean: float | None  # legacy whole-graph historical-Q mean
    particle_class: str  # historical field name for the winding-class label
    is_quantized: bool  # legacy name for |raw - W| being small
    cycle_nodes: tuple[Any, ...] = ()
    telemetry_available: bool = False
    telemetry_scope: str = "unspecified"
    telemetry_unavailable_reason: str | None = None

    @property
    def winding_class(self) -> str:
        """Preferred name for the legacy ``particle_class`` field."""
        return self.particle_class

    @property
    def orientation_sign(self) -> int:
        """Preferred name for the legacy ``chirality`` field."""
        return self.chirality

    @property
    def global_energy_density_mean(self) -> float | None:
        """Whole-graph mean behind the legacy ``energy_density`` field."""
        return self.energy_density

    @property
    def global_q_density_mean(self) -> float | None:
        """Whole-graph mean behind the legacy ``charge_density_mean`` field."""
        return self.charge_density_mean

    @property
    def q_density_mean(self) -> float | None:
        """Non-physical alias for the whole-graph historical Q-density mean."""
        return self.charge_density_mean

    @property
    def is_integral(self) -> bool:
        """Preferred name for the legacy ``is_quantized`` field."""
        return self.is_quantized

    def as_dict(self) -> dict[str, object]:
        return {
            "winding": self.winding,
            "raw_winding": self.raw_winding,
            "orientation_sign": self.orientation_sign,
            "chirality": self.chirality,
            "global_energy_density_mean": self.global_energy_density_mean,
            "energy_density": self.energy_density,
            "global_q_density_mean": self.global_q_density_mean,
            "q_density_mean": self.q_density_mean,
            "charge_density_mean": self.charge_density_mean,
            "winding_class": self.winding_class,
            "particle_class": self.particle_class,
            "is_integral": self.is_integral,
            "is_quantized": self.is_quantized,
            "cycle_nodes": self.cycle_nodes,
            "telemetry_available": self.telemetry_available,
            "telemetry_scope": self.telemetry_scope,
            "telemetry_unavailable_reason": self.telemetry_unavailable_reason,
        }


def _class_from_winding(w: int) -> str:
    aw = abs(w)
    if aw == 0:
        return "zero-winding class"
    if aw == 1:
        return "unit-winding defect class"
    return f"multi-winding defect class (|W|={aw})"


WindingSector = EmergentParticle


def classify_winding_sector(
    G: nx.Graph, *, order: Iterable[Any] | None = None
) -> WindingSector:
    """Classify a declared cycle by its measured integer winding sector.

    Energy and historical-Q values are supporting contextual read-outs over the
    whole graph, even when ``order`` selects a proper subcycle. Their required
    inputs are checked over that whole graph. Missing or invalid inputs produce
    ``None`` values, ``telemetry_available=False``, and an explicit reason;
    unexpected computation failures propagate. No missing read-out is replaced
    by a fabricated zero.
    """
    cycle_nodes = _resolve_cycle_order(G, order)
    w, raw = winding_number(G, order=cycle_nodes)
    orientation_sign = (w > 0) - (w < 0)

    # These supporting values are graph-global, not restricted to cycle_nodes.
    telemetry_issue = _whole_graph_telemetry_issue(G)
    if telemetry_issue is None:
        energy_map = compute_energy_density(G)
        q_map = compute_historical_q_density(G)
        energy_density = _finite_map_mean(energy_map, name="energy_density")
        q_density_mean = _finite_map_mean(q_map, name="historical_q_density")
        telemetry_available = True
    else:
        energy_density = None
        q_density_mean = None
        telemetry_available = False

    return WindingSector(
        winding=w,
        raw_winding=raw,
        chirality=int(orientation_sign),
        energy_density=energy_density,
        charge_density_mean=q_density_mean,
        particle_class=_class_from_winding(w),
        is_quantized=abs(raw - round(raw)) < 1e-6,
        cycle_nodes=cycle_nodes,
        telemetry_available=telemetry_available,
        telemetry_scope="whole_graph_snapshot",
        telemetry_unavailable_reason=telemetry_issue,
    )


def classify_particle(
    G: nx.Graph, *, order: Iterable[Any] | None = None
) -> EmergentParticle:
    """Compatibility wrapper for :func:`classify_winding_sector`."""
    return classify_winding_sector(G, order=order)


__all__ = [
    "EmergentParticle",
    "WindingSector",
    "winding_ring",
    "winding_number",
    "classify_winding_sector",
    "classify_particle",
]

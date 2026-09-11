"""Compartment diagnostics and membrane-pressure dynamics for TNFR graphs.

The read-only detector reports canonical ``C(t)`` on the declared boundary,
an unweighted edge-count selectivity diagnostic, internal-pressure dispersion
and optional flux-based membrane integrity. It classifies only those declared
observables; it does not establish an autopoietic ``A > 1`` certificate or
nested U5 coherence.

The membrane solver specializes the nodal equation through an explicit pressure
channel without defining another glyph::

    DeltaNFR_cell = DeltaNFR_intrinsic + DeltaNFR_membrane
    dEPI_cell/dt = nu_f_cell * DeltaNFR_cell

Canonical glyphs remain the semantic transformations of the operator layer.
This declared domain solver adds membrane pressure only on the boundary, then
advances every node through the shared nodal integrator so the graph clock and
all EPI histories cover the same interval. It records pressure provenance and
reports the boundary equation residual.
"""

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import networkx as nx

from ..alias import get_attr, set_attr
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..constants.canonical import DELTA_PHI_MAX
from ..dynamics.integrators import update_epi_via_nodal_equation
from ..mathematics.unified_numerical import np
from ..metrics.common import compute_coherence
from ..types import real_scalar_epi
from ..utils import angle_diff


@dataclass
class CellTelemetry:
    """Container for cell-emergence telemetry time series.

    Attributes
    ----------
    times : list[float]
        Structural-time coordinates supplied by the observation protocol.
    boundary_coherence : np.ndarray
        C_boundary(t) ∈ [0, 1]. Coherence at cellular boundary regions.
    internal_coherence : np.ndarray
        C_internal(t) ∈ [0, 1]. Coherence within compartmentalized interior.
    selectivity_index : np.ndarray
        ρ_selectivity(t) ∈ [-1, 1]. Membrane coupling preference:
        ρ = (coupling_internal - coupling_external)/(coupling_total).
    homeostatic_index : np.ndarray
        H_index(t) ∈ [0, 1]. Internal-pressure dispersion diagnostic:
        H = 1 - σ(ΔNFR_internal)/(|μ(ΔNFR_internal)| + ε).
    membrane_integrity : np.ndarray
        I_compartment(t) in [0, 1] from supplied flux pairs. NaN means
        that no flux evidence was supplied for that snapshot.
    cell_formation_time : float | None
        First time t where all cellular criteria are satisfied. The default
        ``C_boundary > 0.8`` cut is a selected cell-detector heuristic, not a
        threshold of an operator matrix.
    """

    times: list[float]
    boundary_coherence: np.ndarray
    internal_coherence: np.ndarray
    selectivity_index: np.ndarray
    homeostatic_index: np.ndarray
    membrane_integrity: np.ndarray
    cell_formation_time: float | None = None


@dataclass(frozen=True, slots=True)
class MembraneNodeFlux:
    """One boundary node's membrane-pressure realization and nodal residual."""

    node: Hashable
    epi_before: float
    epi_after: float
    nu_f: float
    intrinsic_delta_nfr: float
    membrane_delta_nfr: float
    effective_delta_nfr: float
    predicted_depi_dt: float
    measured_depi_dt: float
    nodal_residual: float
    compatible_internal_nodes: tuple[Hashable, ...]
    blocked_internal_nodes: tuple[Hashable, ...]
    boundary_projection_applied: bool


@dataclass(frozen=True, slots=True)
class MembraneFluxResult:
    """Detached telemetry for one simultaneous membrane-pressure step."""

    dt: float
    start_time: float
    end_time: float
    phase_threshold: float
    nodes: tuple[MembraneNodeFlux, ...]

    @property
    def max_abs_nodal_residual(self) -> float:
        """Return the largest raw ``dEPI/dt - nu_f*DeltaNFR`` residual."""

        return max((abs(item.nodal_residual) for item in self.nodes), default=0.0)


# Core computations

# Centralised helper — single source of truth in _helpers.py


def compute_boundary_coherence(graph: nx.Graph, boundary_nodes: Sequence[int]) -> float:
    """Compute canonical total coherence ``C(t)`` on the boundary subgraph.

    The centralized constitutive kernel uses boundary-node means of
    ``|DeltaNFR|`` and ``|dEPI|``. This is a restricted graph read-out of
    canonical ``C(t)``, not the auxiliary pairwise ``coherence_matrix``.

    Parameters
    ----------
    graph : nx.Graph
        Network with node attributes 'delta_nfr' (structural pressure).
    boundary_nodes : Sequence[int]
        Node IDs that form the cellular boundary (membrane region).

    Returns
    -------
    float
        Boundary coherence ``C_boundary`` in ``[0, 1]``. The default ``0.8``
        cut in :func:`detect_cell_formation` is a selected cellular heuristic,
        not a threshold derived from the coherence operator notation.
    """
    if not boundary_nodes:
        return 0.0

    # Extract boundary subgraph
    boundary_subgraph = graph.subgraph(boundary_nodes).copy()

    # Compute coherence on boundary only
    if len(boundary_subgraph.nodes()) == 0:
        return 0.0

    return compute_coherence(boundary_subgraph)


def compute_selectivity_index(
    graph: nx.Graph, internal_nodes: Sequence[int], boundary_nodes: Sequence[int]
) -> float:
    """Compute unweighted edge-count selectivity for a declared compartment.

    This diagnostic counts edges wholly inside the declared cell and edges
    crossing from it. It does not inspect phase, edge weights, or transport flux.

    ρ_selectivity = (C_internal - C_external) / C_total

    Where cellular behavior emerges when ρ > 0.6 (preferential internalization).

    Parameters
    ----------
    graph : nx.Graph
        TNFR network with edges representing structural couplings between nodes.
    internal_nodes : Sequence[int]
        Node IDs forming the compartmentalized cellular interior.
    boundary_nodes : Sequence[int]
        Node IDs forming the phase-selective cellular boundary.

    Returns
    -------
    float
        Selectivity index ρ_selectivity ∈ [-1, 1]. Values ρ > 0.6 indicate emergence
        of cellular organization with preferential internal coupling topology.

    Notes
    -----
    This is read-only topological telemetry. A positive value alone does not
    certify phase-selective transport or autopoiesis.
    """
    internal_set = set(internal_nodes)
    boundary_set = set(boundary_nodes)
    cell_nodes = internal_set | boundary_set

    # Count coupling types
    internal_coupling = 0  # Both nodes internal
    external_coupling = 0  # One internal, one external

    for u, v in graph.edges():
        u_in_cell = u in cell_nodes
        v_in_cell = v in cell_nodes

        if u_in_cell and v_in_cell:
            internal_coupling += 1
        elif u_in_cell or v_in_cell:  # Crossing boundary
            external_coupling += 1

    total_coupling = internal_coupling + external_coupling
    if total_coupling == 0:
        return 0.0

    return (internal_coupling - external_coupling) / total_coupling


def compute_homeostatic_index(
    delta_nfr_internal: np.ndarray, epsilon: float = 1e-6
) -> float:
    """Compute the bounded internal-pressure dispersion diagnostic.

    For the cellular pressure decomposition
    ``ΔNFR_cell = ΔNFR_intrinsic + ΔNFR_membrane``, the diagnostic is

    H_homeostatic = 1 - σ(ΔNFR_internal) / (|μ(ΔNFR_internal)| + ε)

    The detector interprets ``H > 0.5`` as a configured formation criterion.

    Parameters
    ----------
    delta_nfr_internal : one-dimensional array-like
        Finite, non-Boolean internal structural-pressure samples.
    epsilon : float, default=1e-6
        Positive finite denominator regularizer.

    Returns
    -------
    float
        Homeostatic dispersion index in ``[0, 1]``. Larger values mean lower
        sample dispersion relative to the regularized absolute mean.

    Notes
    -----
    This is a bounded dispersion diagnostic. It does not by itself certify
    homeostasis, regulation, or cell formation.
    """
    values = np.asarray(delta_nfr_internal, dtype=object)
    if values.ndim != 1:
        raise ValueError("delta_nfr_internal must be a one-dimensional sequence")
    if values.size == 0:
        return 0.0

    epsilon_value = _finite_membrane_scalar(epsilon, "epsilon")
    if epsilon_value <= 0.0:
        raise ValueError("epsilon must be positive")
    finite_values = np.asarray(
        [
            _finite_membrane_scalar(value, f"delta_nfr_internal[{index}]")
            for index, value in enumerate(values.tolist())
        ],
        dtype=float,
    )

    std_internal = float(np.std(finite_values))
    mean_internal = abs(float(np.mean(finite_values)))
    raw_index = 1.0 - std_internal / (mean_internal + epsilon_value)

    return max(0.0, min(1.0, raw_index))


def compute_membrane_integrity(flux_internal: float, flux_external: float) -> float:
    """Compute cellular membrane integrity from compartmentalization effectiveness.

    From phase-selective membrane-pressure physics, this function quantifies
    the membrane integrity I_compartment as the effectiveness of phase-selective
    transport that characterizes cellular boundary compartmentalization.

    I_compartment = 1 - leakage_rate = 1 - |J_external| / (|J_internal| + |J_external|)

    Where cellular compartmentalization emerges when I > 0.7 (effective separation).

    Parameters
    ----------
    flux_internal : float
        Internal membrane flux (controlled, phase-selective transport).
    flux_external : float
        External membrane leakage (uncontrolled, non-selective transport).

    Returns
    -------
    float
        Membrane integrity I_compartment ∈ [0, 1]. Values I > 0.7 indicate emergence
        of effective cellular compartmentalization with phase-selective transport.

    Notes
    -----
    Measures flux-based compartmentalization respecting membrane physics.
    Cellular integrity emerges from autopoietic foundation through selectivity.
    """
    internal = _finite_membrane_scalar(flux_internal, "flux_internal")
    external = _finite_membrane_scalar(flux_external, "flux_external")
    total_flux = abs(internal) + abs(external)
    if total_flux == 0:
        # With neither controlled transport nor leakage, selectivity is
        # unobserved. A zero score prevents absence of flux from becoming
        # positive membrane evidence in the formation classifier.
        return 0.0

    leakage_rate = abs(external) / total_flux
    return 1.0 - leakage_rate


def detect_cell_formation(
    graph_sequence: Sequence[nx.Graph],
    times: Sequence[float],
    internal_nodes: Sequence[int],
    boundary_nodes: Sequence[int],
    c_boundary_threshold: float = 0.8,
    selectivity_threshold: float = 0.6,
    homeostasis_threshold: float = 0.5,
    integrity_threshold: float = 0.7,
    membrane_fluxes: Sequence[tuple[float, float] | None] | None = None,
) -> CellTelemetry:
    """Classify compartment formation from declared observable time series.

    From the canonical cellular equation
    ``∂EPI_cell/∂t = νf_cell·(ΔNFR_intrinsic + ΔNFR_membrane)``, this
    function combines graph diagnostics with optional measured internal/external
    flux pairs. It does not compute or assume an autopoietic coefficient.

    Cellular criteria (all must be satisfied simultaneously):
    - Boundary canonical C(t): C_boundary > c_boundary_threshold (default 0.8)
    - Selectivity index: ρ_selectivity > selectivity_threshold (default 0.6)
    - Homeostatic dispersion score: H_index > homeostasis_threshold (default 0.5)
    - Membrane integrity: I_compartment > integrity_threshold (default 0.7)

    Parameters
    ----------
    graph_sequence : Sequence[nx.Graph]
        Time series of TNFR network states with node attribute ``delta_nfr``
        (structural pressure).
    times : Sequence[float]
        Structural-time coordinates corresponding to each graph state.
    internal_nodes : Sequence[int]
        Node IDs that form the compartmentalized cell interior.
    boundary_nodes : Sequence[int]
        Node IDs that form the phase-selective cellular boundary (membrane).
    c_boundary_threshold : float, default=0.8
        Selected cell-detector heuristic for minimum boundary ``C(t)``. It is
        not a general threshold of canonical coherence or an operator matrix.
    selectivity_threshold : float, default=0.6
        Minimum selectivity index for preferential internal coupling.
    homeostasis_threshold : float, default=0.5
        Minimum internal-pressure dispersion score.
    integrity_threshold : float, default=0.7
        Minimum flux-based membrane integrity.
    membrane_fluxes : sequence of (internal, external) pairs or None
        Optional observed flux pair per graph snapshot. Missing evidence is
        represented by NaN and cannot satisfy the formation predicate.

    Returns
    -------
    CellTelemetry
        Complete cellular telemetry time series including formation time detection.
        cell_formation_time contains first time when all criteria satisfied, or None.

    Notes
    -----
    This read-only classifier does not establish ``A > 1`` or U5. Use
    ``detect_life_emergence`` and the hierarchy validators as separate evidence
    when those claims are required.
    """
    graphs = list(graph_sequence)
    raw_times = np.asarray(list(times), dtype=object)
    if raw_times.ndim != 1:
        raise ValueError("times must be a one-dimensional sequence")
    time_values = np.asarray(
        [
            _finite_membrane_scalar(value, f"times[{index}]")
            for index, value in enumerate(raw_times.tolist())
        ],
        dtype=float,
    )
    normalized_times = time_values.tolist()
    n_timesteps = len(graphs)
    if len(normalized_times) != n_timesteps:
        raise ValueError("times and graph_sequence must have the same length")

    internal = _declared_nodes(internal_nodes, "internal_nodes")
    boundary = _declared_nodes(boundary_nodes, "boundary_nodes")
    if set(internal).intersection(boundary):
        raise ValueError("internal_nodes and boundary_nodes must be disjoint")
    for snapshot_index, graph in enumerate(graphs):
        if not isinstance(
            graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
        ):
            raise TypeError(
                f"graph_sequence[{snapshot_index}] must be a networkx graph"
            )
        _membrane_nodes(
            graph, internal, f"internal_nodes at snapshot {snapshot_index}"
        )
        _membrane_nodes(
            graph, boundary, f"boundary_nodes at snapshot {snapshot_index}"
        )
    if len(time_values) > 1 and np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must increase strictly")

    threshold_specs = (
        ("c_boundary_threshold", c_boundary_threshold, 0.0, 1.0),
        ("selectivity_threshold", selectivity_threshold, -1.0, 1.0),
        ("homeostasis_threshold", homeostasis_threshold, 0.0, 1.0),
        ("integrity_threshold", integrity_threshold, 0.0, 1.0),
    )
    for label, raw, lower, upper in threshold_specs:
        value = _finite_membrane_scalar(raw, label)
        if not lower <= value <= upper:
            raise ValueError(f"{label} must lie in [{lower}, {upper}]")

    fluxes = None if membrane_fluxes is None else list(membrane_fluxes)
    if fluxes is not None and len(fluxes) != n_timesteps:
        raise ValueError(
            "membrane_fluxes and graph_sequence must have the same length"
        )

    # Initialize arrays
    boundary_coherence = np.zeros(n_timesteps)
    internal_coherence = np.zeros(n_timesteps)
    selectivity_index = np.zeros(n_timesteps)
    homeostatic_index = np.zeros(n_timesteps)
    membrane_integrity = np.full(n_timesteps, np.nan, dtype=float)

    # Homeostatic dispersion is a state observable at each snapshot.

    for t_idx, graph in enumerate(graphs):
        # Boundary coherence
        boundary_coherence[t_idx] = compute_boundary_coherence(graph, boundary)

        # Internal coherence
        if internal:
            internal_subgraph = graph.subgraph(internal).copy()
            internal_coherence[t_idx] = (
                compute_coherence(internal_subgraph)
                if len(internal_subgraph.nodes()) > 0
                else 0.0
            )
        else:
            internal_coherence[t_idx] = 0.0

        # Selectivity index
        selectivity_index[t_idx] = compute_selectivity_index(
            graph, internal, boundary
        )

        # Collect internal ΔNFR values
        internal_dnfr = []
        for node in internal:
            raw_dnfr = _raw_alias_value(graph.nodes[node], ALIAS_DNFR, None)
            if raw_dnfr is None:
                raise ValueError(
                    f"graph_sequence[{t_idx}] node {node!r} requires explicit "
                    "delta_nfr evidence for homeostatic dispersion"
                )
            internal_dnfr.append(
                _finite_membrane_scalar(
                    raw_dnfr,
                    f"graph_sequence[{t_idx}] node {node!r} delta_nfr",
                )
            )

        # At least two simultaneous internal samples are needed to observe
        # dispersion. Pooling past snapshots would make H(t) path-dependent.
        if len(internal_dnfr) > 1:
            homeostatic_index[t_idx] = compute_homeostatic_index(
                np.asarray(internal_dnfr, dtype=float)
            )
        else:
            homeostatic_index[t_idx] = 0.0

        if fluxes is not None and fluxes[t_idx] is not None:
            raw_pair = fluxes[t_idx]
            if isinstance(raw_pair, (str, bytes, bytearray)):
                raise ValueError(
                    "each membrane_fluxes entry must be an (internal, external) pair"
                )
            try:
                pair = tuple(raw_pair)
            except TypeError as exc:
                raise ValueError(
                    "each membrane_fluxes entry must be an (internal, external) pair"
                ) from exc
            if len(pair) != 2:
                raise ValueError(
                    "each membrane_fluxes entry must be an (internal, external) pair"
                )
            membrane_integrity[t_idx] = compute_membrane_integrity(
                pair[0], pair[1]
            )

    # Detect cell formation time
    cell_formation_time: float | None = None

    for t_idx in range(n_timesteps):
        criteria_met = (
            boundary_coherence[t_idx] > c_boundary_threshold
            and selectivity_index[t_idx] > selectivity_threshold
            and homeostatic_index[t_idx] > homeostasis_threshold
            and np.isfinite(membrane_integrity[t_idx])
            and membrane_integrity[t_idx] > integrity_threshold
        )

        if criteria_met:
            cell_formation_time = normalized_times[t_idx]
            break

    return CellTelemetry(
        times=normalized_times,
        boundary_coherence=boundary_coherence,
        internal_coherence=internal_coherence,
        selectivity_index=selectivity_index,
        homeostatic_index=homeostatic_index,
        membrane_integrity=membrane_integrity,
        cell_formation_time=cell_formation_time,
    )


_MEMBRANE_PRESSURE_MODEL = "cell_membrane_delta_nfr_v1"
_MEMBRANE_HISTORY_LIMIT = 64
_MEMBRANE_PROVENANCE_KEY = "membrane_pressure_provenance"
_MEMBRANE_SEQUENCE_KEY = "_membrane_pressure_sequence"


class _MembraneOwnedPressure(float):
    """Float-compatible effective pressure carrying its ownership token."""

    membrane_provenance_token: str

    def __new__(cls, value: float, token: str) -> "_MembraneOwnedPressure":
        instance = super().__new__(cls, value)
        instance.membrane_provenance_token = token
        return instance

    def __reduce__(self) -> tuple[Any, tuple[float, str]]:
        """Preserve the ownership token across graph snapshots and rollback copies."""

        return (
            type(self),
            (float(self), str(self.membrane_provenance_token)),
        )


def _finite_membrane_scalar(value: Any, label: str) -> float:
    """Return one finite non-Boolean real used by the membrane model."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real scalar, not boolean")
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite real scalar") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be a finite real scalar")
    return resolved


def _raw_alias_value(
    node_data: dict[str, Any], aliases: tuple[str, ...], default: Any
) -> Any:
    """Read the authoritative alias without applying a scalar projection."""

    return get_attr(
        node_data,
        aliases,
        default,
        strict=True,
        conv=lambda value: value,
    )


def _membrane_epi_scalar(value: Any, label: str) -> float:
    """Require the real scalar EPI chart, including its uniform BEPI embedding."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite real scalar EPI, not boolean")
    try:
        scalar = real_scalar_epi(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a raw scalar or uniform-real BEPI embedding"
        ) from exc
    if scalar is None or not math.isfinite(float(scalar)):
        raise ValueError(
            f"{label} must be a finite raw scalar or uniform-real BEPI embedding"
        )
    return float(scalar)


def _declared_nodes(
    values: Sequence[Hashable], label: str
) -> tuple[Hashable, ...]:
    """Materialize a replayable sequence of unique hashable node identifiers."""

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of node identifiers")
    try:
        nodes = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be a replayable sequence") from exc
    seen: set[Hashable] = set()
    for node in nodes:
        try:
            duplicate = node in seen
        except TypeError as exc:
            raise TypeError(f"{label} contains an unhashable node identifier") from exc
        if duplicate:
            raise ValueError(f"{label} must not contain duplicate nodes")
        seen.add(node)
    return nodes


def _membrane_nodes(
    graph: nx.Graph, values: Sequence[Hashable], label: str
) -> tuple[Hashable, ...]:
    """Require every declared node to exist in one graph snapshot."""

    nodes = _declared_nodes(values, label)
    for node in nodes:
        if node not in graph:
            raise ValueError(f"{label} contains unknown node {node!r}")
    return nodes


def _pressure_alias_key(node_data: Mapping[str, Any]) -> str:
    """Resolve the same authoritative ΔNFR alias used by the shared accessor."""

    return next((key for key in ALIAS_DNFR if key in node_data), ALIAS_DNFR[0])


def _owned_intrinsic_pressure(
    node_data: Mapping[str, Any], raw_pressure: Any, pressure: float, *, node: Hashable
) -> float:
    """Resolve the intrinsic channel only from an attached ownership token."""

    if not isinstance(raw_pressure, _MembraneOwnedPressure):
        return pressure

    provenance = node_data.get(_MEMBRANE_PROVENANCE_KEY)
    if not isinstance(provenance, Mapping):
        raise ValueError(f"node {node!r} membrane pressure provenance is missing")
    token = provenance.get("token")
    if (
        provenance.get("model") != _MEMBRANE_PRESSURE_MODEL
        or provenance.get("source")
        != "tnfr.physics.cell.apply_membrane_flux"
        or provenance.get("node") != node
        or not isinstance(token, str)
        or token != raw_pressure.membrane_provenance_token
    ):
        raise ValueError(f"node {node!r} membrane pressure provenance is invalid")

    intrinsic = _finite_membrane_scalar(
        provenance.get("intrinsic_delta_nfr"),
        f"node {node!r} intrinsic membrane-pressure channel",
    )
    membrane = _finite_membrane_scalar(
        provenance.get("membrane_delta_nfr"),
        f"node {node!r} membrane pressure channel",
    )
    effective = _finite_membrane_scalar(
        provenance.get("effective_delta_nfr"),
        f"node {node!r} effective membrane pressure provenance",
    )
    if effective != pressure or effective != intrinsic + membrane:
        raise ValueError(f"node {node!r} membrane pressure provenance is inconsistent")
    return intrinsic


def _set_owned_pressure(
    node_data: dict[str, Any], value: float, token: str
) -> None:
    """Store an effective ΔNFR value with its in-process ownership token."""

    node_data[_pressure_alias_key(node_data)] = _MembraneOwnedPressure(value, token)


def _event_history(raw: Any, label: str) -> list[Any]:
    """Copy a provenance-event sink without consuming ambiguous iterators."""

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list when present")
    return list(raw)


def _legacy_epi_history(raw: Any, label: str, current_epi: float) -> list[float]:
    """Validate and align a legacy unit-step EPI history with the live state."""

    if raw is None:
        values: list[Any] = []
    elif isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an indexed EPI sequence")
    else:
        try:
            values = list(raw)
        except TypeError as exc:
            raise ValueError(f"{label} must be a replayable EPI sequence") from exc
    history = [
        _membrane_epi_scalar(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    if not history or history[-1] != current_epi:
        history.append(current_epi)
    return history


def _physical_epi_history(
    raw: Any, *, node: Any, start_time: float, current_epi: float
) -> list[tuple[float, float]]:
    """Validate and align timestamped EPI evidence at the step's left endpoint."""

    label = f"node {node!r} epi_time_history"
    if raw is None:
        values: list[Any] = []
    elif isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"{label} must contain (time, EPI) pairs")
    else:
        try:
            values = list(raw)
        except TypeError as exc:
            raise ValueError(f"{label} must be replayable") from exc

    history: list[tuple[float, float]] = []
    for index, entry in enumerate(values):
        if isinstance(entry, (str, bytes, bytearray)):
            pair = None
        else:
            try:
                pair = tuple(entry)
            except TypeError:
                pair = None
        if pair is None or len(pair) != 2:
            raise ValueError(f"{label}[{index}] must be a (time, EPI) pair")
        sample_time = _finite_membrane_scalar(pair[0], f"{label}[{index}].time")
        sample_epi = _membrane_epi_scalar(pair[1], f"{label}[{index}].EPI")
        if history and sample_time <= history[-1][0]:
            raise ValueError(f"{label} timestamps must increase strictly")
        history.append((sample_time, sample_epi))

    if history and history[-1][0] > start_time:
        raise ValueError(f"{label} extends beyond the membrane step start time")
    if history and history[-1][0] == start_time:
        if history[-1][1] != current_epi:
            raise ValueError(f"{label} is stale at the membrane step start time")
    else:
        history.append((start_time, current_epi))
    return history


def _membrane_proposal_graph(graph: nx.Graph, start_time: float) -> nx.Graph:
    """Create an isolated graph that inherits only nodal-integrator policies."""

    if graph.is_directed():
        proposal: nx.Graph = (
            nx.MultiDiGraph() if graph.is_multigraph() else nx.DiGraph()
        )
    else:
        proposal = nx.MultiGraph() if graph.is_multigraph() else nx.Graph()
    for key in ("EPI_MIN", "EPI_MAX", "CLIP_MODE", "CLIP_SOFT_K", "DT_MIN"):
        if key in graph.graph:
            proposal.graph[key] = graph.graph[key]
    proposal.graph["_gamma_spec"] = {"type": "none"}
    proposal.graph["_t"] = start_time
    return proposal


def apply_membrane_flux(
    graph: nx.Graph,
    internal_nodes: Sequence[Hashable],
    boundary_nodes: Sequence[Hashable],
    permeability: float = 0.1,
    phase_threshold: float = np.pi / 3,
    *,
    dt: float = 0.01,
) -> MembraneFluxResult:
    """Advance boundary EPI through a phase-gated membrane-pressure channel.

    This narrow cellular model treats membrane transport as an explicit
    domain-specific contribution to structural pressure rather than as a second
    EPI derivative::

        DeltaNFR_mem(b) = kappa * mean(EPI_i - EPI_b)
        DeltaNFR_eff(b) = DeltaNFR_intrinsic(b) + DeltaNFR_mem(b)
        dEPI_b/dt = nu_f(b) * DeltaNFR_eff(b)

    The mean includes only declared internal nodes reached by
    ``graph.neighbors(boundary_node)`` whose shortest-arc phase separation passes
    U3. Thus directed graphs use outgoing boundary arcs. Internal nodes are
    read-only while the membrane contribution is assembled; during the declared
    time step every graph node advances under its intrinsic pressure, while only
    boundary nodes receive the membrane term.

    Every pressure and EPI proposal is computed from one initial snapshot. An
    isolated graph then advances the complete nodal state together through
    :func:`update_epi_via_nodal_equation`; the live graph changes only after all
    inputs, histories, outputs and provenance sinks have validated.

    Parameters
    ----------
    graph : nx.Graph
        TNFR graph carrying scalar EPI, ``nu_f``, ``delta_nfr`` and phase.
    internal_nodes : Sequence[Hashable]
        Unique existing nodes used as membrane-contrast sources.
    boundary_nodes : Sequence[Hashable]
        Unique existing, disjoint nodes receiving membrane pressure.
    permeability : float, default=0.1
        Finite pressure-response coefficient in the closed interval ``[0, 1]``.
    phase_threshold : float, default=pi/3
        Optional nonnegative U3 tightening.  It cannot exceed the graph's valid
        ``DELTA_PHI_MAX`` hard gate, itself bounded above by ``pi/2``.
    dt : float, keyword-only, default=0.01
        Explicit positive finite structural-time interval.  The default preserves
        the legacy call's historical step while making it inspectable.

    Returns
    -------
    MembraneFluxResult
        Detached boundary pressure, rate and raw nodal-residual telemetry.

    Notes
    -----
    ``delta_nfr_intrinsic`` and ``delta_nfr_membrane`` expose the two channels.
    The effective scalar carries a transaction token also recorded in
    ``membrane_pressure_provenance``. Only that token establishes ownership; a
    plain external ΔNFR write, including a numerically equal write, becomes the
    next intrinsic pressure. Provenance is never presented as a canonical glyph.
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("graph must be a networkx graph instance")

    kappa = _finite_membrane_scalar(permeability, "permeability")
    if not 0.0 <= kappa <= 1.0:
        raise ValueError("permeability must lie in the closed interval [0, 1]")
    requested_phase_limit = _finite_membrane_scalar(
        phase_threshold, "phase_threshold"
    )
    if requested_phase_limit < 0.0:
        raise ValueError("phase_threshold must be nonnegative")
    time_step = _finite_membrane_scalar(dt, "dt")
    if time_step <= 0.0:
        raise ValueError("dt must be positive")

    hard_phase_limit = _finite_membrane_scalar(
        graph.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX), "DELTA_PHI_MAX"
    )
    if not 0.0 <= hard_phase_limit <= float(DELTA_PHI_MAX):
        raise ValueError(
            f"DELTA_PHI_MAX must lie in the canonical interval [0, {DELTA_PHI_MAX}]"
        )
    effective_phase_limit = min(requested_phase_limit, hard_phase_limit)

    internal = _membrane_nodes(graph, internal_nodes, "internal_nodes")
    boundary = _membrane_nodes(graph, boundary_nodes, "boundary_nodes")
    if set(internal).intersection(boundary):
        raise ValueError("internal_nodes and boundary_nodes must be disjoint")

    start_time = _finite_membrane_scalar(
        graph.graph.get("_t", 0.0), "graph structural time"
    )
    nominal_end_time = _finite_membrane_scalar(
        start_time + time_step, "membrane step end time"
    )
    if nominal_end_time <= start_time:
        raise ValueError("dt is below the represented structural-time resolution")
    if not boundary:
        return MembraneFluxResult(
            dt=time_step,
            start_time=start_time,
            end_time=start_time,
            phase_threshold=effective_phase_limit,
            nodes=(),
        )

    participant_state: dict[Any, dict[str, Any]] = {}
    for node in graph.nodes:
        node_data = graph.nodes[node]
        raw_delta_nfr = _raw_alias_value(node_data, ALIAS_DNFR, 0.0)
        pressure = _finite_membrane_scalar(
            raw_delta_nfr, f"node {node!r} delta_nfr"
        )
        state = {
            "epi": _membrane_epi_scalar(
                _raw_alias_value(node_data, ALIAS_EPI, 0.0),
                f"node {node!r} EPI",
            ),
            "nu_f": _finite_membrane_scalar(
                _raw_alias_value(node_data, ALIAS_VF, 0.0),
                f"node {node!r} nu_f",
            ),
            "delta_nfr_raw": raw_delta_nfr,
            "delta_nfr": pressure,
            "intrinsic_pressure": _owned_intrinsic_pressure(
                node_data, raw_delta_nfr, pressure, node=node
            ),
            "phase": _finite_membrane_scalar(
                _raw_alias_value(node_data, ALIAS_THETA, 0.0),
                f"node {node!r} phase",
            ),
            "depi_dt": _finite_membrane_scalar(
                _raw_alias_value(node_data, ALIAS_DEPI, 0.0),
                f"node {node!r} dEPI/dt",
            ),
        }
        if state["nu_f"] < 0.0:
            raise ValueError(f"node {node!r} nu_f must be nonnegative")
        participant_state[node] = state

    graph_events = _event_history(
        graph.graph.get("membrane_flux_events"), "membrane_flux_events"
    )
    sequence_raw = graph.graph.get(_MEMBRANE_SEQUENCE_KEY, 0)
    if isinstance(sequence_raw, (bool, np.bool_)) or not isinstance(
        sequence_raw, Integral
    ):
        raise ValueError(f"{_MEMBRANE_SEQUENCE_KEY} must be a nonnegative integer")
    sequence = int(sequence_raw)
    if sequence < 0:
        raise ValueError(f"{_MEMBRANE_SEQUENCE_KEY} must be a nonnegative integer")
    transaction_token = f"{_MEMBRANE_PRESSURE_MODEL}:{sequence + 1}"
    internal_set = set(internal)
    proposals: dict[Any, dict[str, Any]] = {}
    proposal_graph = _membrane_proposal_graph(graph, start_time)

    for node in boundary:
        node_data = graph.nodes[node]
        state = participant_state[node]
        intrinsic_pressure = state["intrinsic_pressure"]

        candidates = tuple(
            neighbor for neighbor in graph.neighbors(node) if neighbor in internal_set
        )
        compatible: list[Any] = []
        blocked: list[Any] = []
        contrasts: list[float] = []
        for neighbor in candidates:
            neighbor_state = participant_state[neighbor]
            separation = abs(angle_diff(state["phase"], neighbor_state["phase"]))
            if separation <= effective_phase_limit:
                compatible.append(neighbor)
                contrast = neighbor_state["epi"] - state["epi"]
                contrasts.append(
                    _finite_membrane_scalar(
                        contrast, f"membrane EPI contrast {node!r}->{neighbor!r}"
                    )
                )
            else:
                blocked.append(neighbor)

        mean_contrast = (
            math.fsum(sorted(contrasts)) / len(contrasts) if contrasts else 0.0
        )
        membrane_pressure = _finite_membrane_scalar(
            kappa * mean_contrast, f"node {node!r} membrane delta_nfr"
        )
        effective_pressure = _finite_membrane_scalar(
            intrinsic_pressure + membrane_pressure,
            f"node {node!r} effective delta_nfr",
        )
        predicted_rate = _finite_membrane_scalar(
            state["nu_f"] * effective_pressure,
            f"node {node!r} predicted dEPI/dt",
        )
        unconstrained_after = _finite_membrane_scalar(
            state["epi"] + time_step * predicted_rate,
            f"node {node!r} unconstrained EPI proposal",
        )

        pressure_history = _event_history(
            node_data.get("membrane_pressure_history"),
            f"node {node!r} membrane_pressure_history",
        )

        proposals[node] = {
            "state": state,
            "intrinsic_pressure": intrinsic_pressure,
            "membrane_pressure": membrane_pressure,
            "effective_pressure": effective_pressure,
            "predicted_rate": predicted_rate,
            "unconstrained_after": unconstrained_after,
            "compatible": tuple(compatible),
            "blocked": tuple(blocked),
            "pressure_history": pressure_history,
        }

    histories: dict[Any, dict[str, Any]] = {}
    boundary_set = set(boundary)
    for node, state in participant_state.items():
        node_data = graph.nodes[node]
        private_history = None
        if "_epi_history" in node_data:
            private_history = _legacy_epi_history(
                node_data.get("_epi_history"),
                f"node {node!r} _epi_history",
                state["epi"],
            )
        histories[node] = {
            "legacy": _legacy_epi_history(
                node_data.get("epi_history"),
                f"node {node!r} epi_history",
                state["epi"],
            ),
            "private": private_history,
            "physical": _physical_epi_history(
                node_data.get("epi_time_history"),
                node=node,
                start_time=start_time,
                current_epi=state["epi"],
            ),
        }
        pressure = (
            proposals[node]["effective_pressure"]
            if node in boundary_set
            else state["intrinsic_pressure"]
        )
        proposal_graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: state["epi"],
                ALIAS_VF[0]: state["nu_f"],
                ALIAS_DNFR[0]: pressure,
                ALIAS_DEPI[0]: state["depi_dt"],
            },
        )

    update_epi_via_nodal_equation(
        proposal_graph,
        dt=time_step,
        t=start_time,
        method="euler",
        n_jobs=None,
    )
    end_time = _finite_membrane_scalar(
        proposal_graph.graph.get("_t"), "integrated membrane end time"
    )

    node_telemetry: list[MembraneNodeFlux] = []
    replacement_data: dict[Any, dict[str, Any]] = {}
    boundary_set = set(boundary)
    for node in graph.nodes:
        state = participant_state[node]
        integrated = proposal_graph.nodes[node]
        epi_after = _finite_membrane_scalar(
            _raw_alias_value(integrated, ALIAS_EPI, None),
            f"node {node!r} integrated EPI",
        )
        depi_after = _finite_membrane_scalar(
            _raw_alias_value(integrated, ALIAS_DEPI, None),
            f"node {node!r} integrated dEPI/dt",
        )
        d2epi_after = _finite_membrane_scalar(
            _raw_alias_value(integrated, ALIAS_D2EPI, None),
            f"node {node!r} integrated d2EPI/dt2",
        )
        new_data = dict(graph.nodes[node])
        set_attr(new_data, ALIAS_EPI, epi_after)
        set_attr(new_data, ALIAS_DEPI, depi_after)
        set_attr(new_data, ALIAS_D2EPI, d2epi_after)
        history = histories[node]
        new_data["epi_history"] = [
            *history["legacy"], epi_after
        ][-_MEMBRANE_HISTORY_LIMIT:]
        if history["private"] is not None:
            new_data["_epi_history"] = [
                *history["private"], epi_after
            ][-_MEMBRANE_HISTORY_LIMIT:]
        new_data["epi_time_history"] = [
            *history["physical"], (end_time, epi_after)
        ][-_MEMBRANE_HISTORY_LIMIT:]

        if node not in boundary_set:
            set_attr(new_data, ALIAS_DNFR, state["intrinsic_pressure"])
            for key in (
                "delta_nfr_intrinsic",
                "delta_nfr_membrane",
                "membrane_effective_delta_nfr",
                _MEMBRANE_PROVENANCE_KEY,
            ):
                new_data.pop(key, None)
            replacement_data[node] = new_data
            continue

        proposal = proposals[node]
        measured_rate = _finite_membrane_scalar(
            (epi_after - state["epi"]) / time_step,
            f"node {node!r} measured dEPI/dt",
        )
        residual = _finite_membrane_scalar(
            measured_rate - proposal["predicted_rate"],
            f"node {node!r} nodal residual",
        )
        projected = not math.isclose(
            epi_after,
            proposal["unconstrained_after"],
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
        node_telemetry.append(
            MembraneNodeFlux(
                node=node,
                epi_before=state["epi"],
                epi_after=epi_after,
                nu_f=state["nu_f"],
                intrinsic_delta_nfr=proposal["intrinsic_pressure"],
                membrane_delta_nfr=proposal["membrane_pressure"],
                effective_delta_nfr=proposal["effective_pressure"],
                predicted_depi_dt=proposal["predicted_rate"],
                measured_depi_dt=measured_rate,
                nodal_residual=residual,
                compatible_internal_nodes=proposal["compatible"],
                blocked_internal_nodes=proposal["blocked"],
                boundary_projection_applied=projected,
            )
        )
        event = {
            "model": _MEMBRANE_PRESSURE_MODEL,
            "source": "tnfr.physics.cell.apply_membrane_flux",
            "token": transaction_token,
            "start_time": start_time,
            "end_time": end_time,
            "dt": time_step,
            "permeability": kappa,
            "phase_threshold": effective_phase_limit,
            "intrinsic_delta_nfr": proposal["intrinsic_pressure"],
            "membrane_delta_nfr": proposal["membrane_pressure"],
            "effective_delta_nfr": proposal["effective_pressure"],
            "predicted_depi_dt": proposal["predicted_rate"],
            "measured_depi_dt": measured_rate,
            "nodal_residual": residual,
            "compatible_internal_nodes": proposal["compatible"],
            "blocked_internal_nodes": proposal["blocked"],
            "boundary_projection_applied": projected,
        }
        _set_owned_pressure(
            new_data, proposal["effective_pressure"], transaction_token
        )
        new_data["delta_nfr_intrinsic"] = proposal["intrinsic_pressure"]
        new_data["delta_nfr_membrane"] = proposal["membrane_pressure"]
        new_data["membrane_effective_delta_nfr"] = proposal["effective_pressure"]
        new_data[_MEMBRANE_PROVENANCE_KEY] = {
            "model": _MEMBRANE_PRESSURE_MODEL,
            "source": "tnfr.physics.cell.apply_membrane_flux",
            "token": transaction_token,
            "node": node,
            "intrinsic_delta_nfr": proposal["intrinsic_pressure"],
            "membrane_delta_nfr": proposal["membrane_pressure"],
            "effective_delta_nfr": proposal["effective_pressure"],
        }
        new_data["membrane_pressure_history"] = [
            *proposal["pressure_history"], event
        ][-_MEMBRANE_HISTORY_LIMIT:]
        replacement_data[node] = new_data

    result = MembraneFluxResult(
        dt=time_step,
        start_time=start_time,
        end_time=end_time,
        phase_threshold=effective_phase_limit,
        nodes=tuple(node_telemetry),
    )
    graph_event = {
        "model": _MEMBRANE_PRESSURE_MODEL,
        "source": "tnfr.physics.cell.apply_membrane_flux",
        "token": transaction_token,
        "start_time": start_time,
        "end_time": end_time,
        "dt": time_step,
        "permeability": kappa,
        "phase_threshold": effective_phase_limit,
        "internal_nodes": internal,
        "boundary_nodes": boundary,
        "advanced_nodes": tuple(graph.nodes),
        "max_abs_nodal_residual": result.max_abs_nodal_residual,
    }
    replacement_graph_data = dict(graph.graph)
    replacement_graph_data["_t"] = end_time
    replacement_graph_data[_MEMBRANE_SEQUENCE_KEY] = sequence + 1
    replacement_graph_data["membrane_flux_events"] = [*graph_events, graph_event][
        -_MEMBRANE_HISTORY_LIMIT:
    ]

    original_nodes = {node: dict(graph.nodes[node]) for node in graph.nodes}
    original_graph_data = dict(graph.graph)
    try:
        for node in graph.nodes:
            graph.nodes[node].clear()
            graph.nodes[node].update(replacement_data[node])
        graph.graph.clear()
        graph.graph.update(replacement_graph_data)
    except BaseException:
        for node, node_data in original_nodes.items():
            graph.nodes[node].clear()
            graph.nodes[node].update(node_data)
        graph.graph.clear()
        graph.graph.update(original_graph_data)
        raise

    return result


__all__ = [
    "CellTelemetry",
    "MembraneNodeFlux",
    "MembraneFluxResult",
    "compute_boundary_coherence",
    "compute_selectivity_index",
    "compute_homeostatic_index",
    "compute_membrane_integrity",
    "detect_cell_formation",
    "apply_membrane_flux",
]

"""Hierarchical multi-scale TNFR network implementation.

The model manages separate TNFR graphs and composes their pressure channels
through declared directed scale couplings. It is an operational multi-scale
model; it does not by itself enforce U3, U5, or canonical operator grammar.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Sequence

import networkx as nx

from ..alias import collect_attr, get_attr, set_dnfr
from ..constants.aliases import ALIAS_DEPI, ALIAS_DNFR
from ..dynamics import (
    dnfr_epi_vf_mixed,
    set_delta_nfr_hook,
    update_epi_via_nodal_equation,
)
from ..mathematics.unified_numerical import np
from ..types import DeltaNFR, NodeId, TNFRGraph
from ..utils import angle_diff, get_logger

logger = get_logger(__name__)


def _finite_signed_mean(values: Iterable[float], *, name: str) -> float:
    """Return a finite signed mean with an overflow-safe intermediate sum."""

    scalars: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain finite real scalars")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must contain only finite values")
        scalars.append(normalized)
    if not scalars:
        return 0.0
    scale = max(abs(value) for value in scalars)
    if scale == 0.0:
        return 0.0
    result = scale * math.fsum(value / scale for value in scalars) / len(scalars)
    if not math.isfinite(result):
        raise ValueError(f"mean {name} exceeds the finite scalar range")
    return result


@dataclass(frozen=True)
class ScaleDefinition:
    """Definition of a single scale in a hierarchical TNFR network.

    Parameters
    ----------
    name : str
        Identifier for this scale (e.g., "quantum", "molecular", "cellular")
    node_count : int
        Number of nodes at this scale
    coupling_strength : float
        Base coupling strength for nodes within this scale (0.0 to 1.0)
    edge_probability : float, optional
        Probability of edge creation in Erdős-Rényi graph generation
    """

    name: str
    node_count: int
    coupling_strength: float
    edge_probability: float = 0.1

    def __post_init__(self) -> None:
        """Validate and normalize one reproducible scale declaration."""

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scale name must be a non-empty string")
        if isinstance(self.node_count, bool) or not isinstance(
            self.node_count, Integral
        ):
            raise TypeError("scale node_count must be a positive integer")
        if int(self.node_count) <= 0:
            raise ValueError("scale node_count must be positive")

        for label, value in (
            ("coupling_strength", self.coupling_strength),
            ("edge_probability", self.edge_probability),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"scale {label} must be a finite real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"scale {label} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"scale {label} must be in [0, 1]")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "node_count", int(self.node_count))
        object.__setattr__(
            self, "coupling_strength", float(self.coupling_strength)
        )
        object.__setattr__(
            self, "edge_probability", float(self.edge_probability)
        )


@dataclass
class EvolutionResult:
    """Results from multi-scale evolution.

    Attributes
    ----------
    scale_results : dict[str, Any]
        Results indexed by scale name
    total_coherence : float
        Canonical coherence over the union of all scale-node channels
    cross_scale_coupling : float
        Measure of cross-scale synchronization
    """

    scale_results: dict[str, Any]
    total_coherence: float = 0.0
    cross_scale_coupling: float = 0.0


class HierarchicalTNFRNetwork:
    """Multi-scale TNFR network supporting operational fractality (§3.7).

    Manages multiple TNFR graphs with simultaneous pressure staging, directed
    cross-scale pressure coupling and nodal-equation EPI updates. Fixed seeds
    reproduce graph generation and evolution. Canonical operator application,
    U3 admission and a normalized U5 parent/child inequality remain separate
    engine contracts.

    Parameters
    ----------
    scales : Sequence[ScaleDefinition]
        Definitions of each scale in the hierarchy
    seed : int, optional
        Random seed for reproducible network generation
    parallel : bool, optional
        Enable parallel evolution of scales (default: True)
    max_workers : int, optional
        Maximum worker threads/processes for parallel execution

    Examples
    --------
    Create a two-scale network and evolve it:

    >>> from tnfr.multiscale import HierarchicalTNFRNetwork, ScaleDefinition
    >>> scales = [
    ...     ScaleDefinition("micro", 4, 0.8, 0.5),
    ...     ScaleDefinition("macro", 3, 0.5, 0.5),
    ... ]
    >>> network = HierarchicalTNFRNetwork(scales, seed=42, parallel=False)
    >>> result = network.evolve_multiscale(dt=0.1, steps=1)
    >>> 0.0 <= result.total_coherence <= 1.0
    True

    Notes
    -----
    Cross-scale coupling is computed as:
        ΔNFR_total = ΔNFR_scale + Σ(coupling_ij * ΔNFR_other_scale)
    where coupling_ij represents the strength of influence from scale j to i.
    """

    def __init__(
        self,
        scales: Sequence[ScaleDefinition],
        seed: int | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
    ):
        scale_items = tuple(scales)
        if not scale_items:
            raise ValueError("At least one scale definition required")

        scale_names = [scale.name for scale in scale_items]
        if len(set(scale_names)) != len(scale_names):
            raise ValueError("scale names must be unique")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer or None")
            if not 0 <= int(seed) <= 2**32 - 1:
                raise ValueError("seed must be in [0, 2**32 - 1]")
            seed = int(seed)
        if not isinstance(parallel, bool):
            raise TypeError("parallel must be bool")
        if max_workers is not None:
            if isinstance(max_workers, bool) or not isinstance(
                max_workers, Integral
            ):
                raise TypeError("max_workers must be a positive integer or None")
            if int(max_workers) <= 0:
                raise ValueError("max_workers must be positive")
            max_workers = int(max_workers)

        self.scales = list(scale_items)
        self.seed = seed
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize networks for each scale
        self.networks_by_scale: dict[str, TNFRGraph] = {}
        self._initialize_scales()

        # Cross-scale coupling matrix (scale x scale)
        self.cross_scale_couplings: dict[tuple[str, str], float] = {}
        self._initialize_cross_scale_couplings()

        logger.info(
            f"Initialized hierarchical network with {len(self.scales)} scales, "
            f"total {sum(scale.node_count for scale in self.scales)} nodes"
        )

    def _initialize_scales(self) -> None:
        """Initialize TNFR network for each scale."""
        rng = np.random.RandomState(self.seed)

        for scale in self.scales:
            # Create Erdős-Rényi graph for this scale
            G = nx.erdos_renyi_graph(
                scale.node_count, scale.edge_probability, seed=rng.randint(0, 2**31)
            )

            # Initialize each node with TNFR attributes
            for node in G.nodes():
                G.nodes[node]["EPI"] = rng.uniform(0.0, 1.0)
                G.nodes[node]["nu_f"] = rng.uniform(0.5, 1.5)
                G.nodes[node]["phase"] = rng.uniform(0.0, 2 * np.pi)
                G.nodes[node]["delta_nfr"] = 0.0
                G.nodes[node]["Si"] = 0.0

            # set base coupling weights
            for u, v in G.edges():
                G[u][v]["weight"] = scale.coupling_strength * rng.uniform(0.8, 1.2)

            # Install ΔNFR hook
            set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

            self.networks_by_scale[scale.name] = G

    def _initialize_cross_scale_couplings(self) -> None:
        """Initialize coupling strengths between scales.

        Default: Adjacent scales couple more strongly than distant scales.
        Keys use ``(target_scale, source_scale)`` so they match the pressure
        equation evaluated by :meth:`compute_multiscale_dnfr`.
        """
        scale_names = [s.name for s in self.scales]

        for i, scale_i in enumerate(scale_names):
            for j, scale_j in enumerate(scale_names):
                if i == j:
                    continue  # No self-coupling

                # Distance-based coupling: closer scales couple more
                distance = abs(i - j)
                coupling_strength = 0.3 / distance if distance > 0 else 0.0

                self.cross_scale_couplings[(scale_i, scale_j)] = coupling_strength

    def set_cross_scale_coupling(
        self, from_scale: str, to_scale: str, strength: float
    ) -> None:
        """Set an explicit cross-scale coupling strength.

        Parameters
        ----------
        from_scale : str
            Source scale name
        to_scale : str
            Target scale name
        strength : float
            Coupling strength (0.0 to 1.0)
        """
        if from_scale not in self.networks_by_scale:
            raise ValueError(f"Unknown scale: {from_scale}")
        if to_scale not in self.networks_by_scale:
            raise ValueError(f"Unknown scale: {to_scale}")
        if from_scale == to_scale:
            raise ValueError("cross-scale coupling requires distinct scales")
        if isinstance(strength, bool) or not isinstance(strength, Real):
            raise TypeError("Coupling strength must be a finite real scalar")
        strength_value = float(strength)
        if not math.isfinite(strength_value):
            raise ValueError("Coupling strength must be finite")
        if strength_value < 0.0 or strength_value > 1.0:
            raise ValueError("Coupling strength must be in [0.0, 1.0]")

        # Public arguments are source -> target; internal keys are
        # (target, source), matching DeltaNFR_target += g_target,source * source.
        self.cross_scale_couplings[(to_scale, from_scale)] = strength_value

    def compute_multiscale_dnfr(self, node_id: NodeId, target_scale: str) -> DeltaNFR:
        """Compute ΔNFR considering all relevant scales.

        Implements cross-scale ΔNFR computation:
            ΔNFR_total = ΔNFR_base + Σ(coupling * ΔNFR_other)

        Parameters
        ----------
        node_id : NodeId
            Node identifier within the target scale
        target_scale : str
            Scale where the node resides

        Returns
        -------
        DeltaNFR
            Multi-scale ΔNFR value
        """
        if target_scale not in self.networks_by_scale:
            raise ValueError(f"Unknown scale: {target_scale}")

        graph = self.networks_by_scale[target_scale]
        base_dnfr = _finite_signed_mean(
            (get_attr(graph.nodes[node_id], ALIAS_DNFR, 0.0, strict=True),),
            name=f"{target_scale} pressure",
        )

        contributions: list[float] = []
        for source_scale, source_graph in self.networks_by_scale.items():
            if source_scale == target_scale:
                continue
            coupling = self._cross_scale_coupling(target_scale, source_scale)
            if coupling == 0.0:
                continue
            source_mean = _finite_signed_mean(
                (
                    get_attr(data, ALIAS_DNFR, 0.0, strict=True)
                    for _, data in source_graph.nodes(data=True)
                ),
                name=f"{source_scale} pressure",
            )
            contributions.append(coupling * source_mean)

        return self._finite_pressure_sum(
            (base_dnfr, *contributions),
            name=f"{target_scale} multiscale pressure",
        )

    def compute_total_coherence(self) -> float:
        """Compute canonical C(t) over the union of all scale nodes.

        Channel magnitudes are aggregated before applying the nonlinear
        constitutive kernel. Averaging already-reduced per-scale coherence
        values would define a different diagnostic.
        """
        from itertools import chain

        from ..metrics.common import finite_mean_absolute, structural_coherence

        total_nodes = sum(
            graph.number_of_nodes()
            for graph in self.networks_by_scale.values()
        )
        if total_nodes == 0:
            return 0.0

        dnfr_values = chain.from_iterable(
            collect_attr(graph, graph.nodes, ALIAS_DNFR, 0.0)
            for graph in self.networks_by_scale.values()
        )
        depi_values = chain.from_iterable(
            collect_attr(graph, graph.nodes, ALIAS_DEPI, 0.0)
            for graph in self.networks_by_scale.values()
        )
        mean_abs_dnfr = finite_mean_absolute(dnfr_values, name="dnfr")
        mean_abs_depi = finite_mean_absolute(depi_values, name="depi")
        return float(structural_coherence(mean_abs_dnfr, mean_abs_depi))

    def evolve_multiscale(
        self,
        dt: float = 0.1,
        steps: int = 10,
        operators: Sequence[str] | None = None,
    ) -> EvolutionResult:
        """Evolve all scales through pressure staging and the nodal equation.

        Parameters
        ----------
        dt : float
            Positive time step when at least one step is requested.
        steps : int
            Nonnegative number of evolution steps.
        operators : Sequence[str], optional
            Reserved compatibility parameter. Non-empty requests are rejected
            because this class has no canonical operator-execution bridge.

        Returns
        -------
        EvolutionResult
            Final per-scale and aggregate metrics from the same state.
        """
        if isinstance(steps, bool) or not isinstance(steps, Integral):
            raise TypeError("steps must be a nonnegative integer")
        steps_value = int(steps)
        if steps_value < 0:
            raise ValueError("steps must be a nonnegative integer")
        if isinstance(dt, bool) or not isinstance(dt, Real):
            raise TypeError("dt must be a finite real scalar")
        dt_value = float(dt)
        if not math.isfinite(dt_value):
            raise ValueError("dt must be finite")
        if steps_value and dt_value <= 0.0:
            raise ValueError("dt must be positive when steps is nonzero")
        if operators is not None and tuple(operators):
            raise NotImplementedError(
                "multiscale operator execution is unavailable; run canonical "
                "operator words explicitly on each scale graph"
            )

        for _ in range(steps_value):
            if self.parallel and self.max_workers != 1:
                self._evolve_parallel()
            else:
                self._evolve_sequential()

            # Compose local and cross-scale pressure before advancing EPI once.
            # The stored DeltaNFR is therefore the pressure used by the
            # canonical nodal-equation step.
            self._apply_cross_scale_coupling()
            for graph in self.networks_by_scale.values():
                update_epi_via_nodal_equation(graph, dt=dt_value, method="euler")

        # Per-scale and aggregate coherence are evaluated from the same final
        # state, after cross-scale pressure and dEPI/dt have been materialized.
        results = {
            scale_name: {"coherence": self._scale_coherence(graph)}
            for scale_name, graph in self.networks_by_scale.items()
        }
        return EvolutionResult(
            scale_results=results,
            total_coherence=self.compute_total_coherence(),
            cross_scale_coupling=self._compute_cross_scale_synchrony(),
        )

    def _evolve_sequential(self) -> None:
        """Stage each scale's registered local pressure sequentially."""

        for graph in self.networks_by_scale.values():
            self._stage_scale_pressure(graph)

    def _evolve_parallel(self) -> None:
        """Stage independent per-scale pressure fields concurrently."""

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = tuple(
                executor.submit(self._evolve_single_scale, scale_name)
                for scale_name in self.networks_by_scale
            )
            for future in futures:
                future.result()

    def _evolve_single_scale(self, scale_name: str) -> None:
        """Stage one scale's registered local pressure."""

        self._stage_scale_pressure(self.networks_by_scale[scale_name])

    @staticmethod
    def _stage_scale_pressure(graph: TNFRGraph) -> None:
        """Evaluate the graph's single registered ΔNFR realization."""

        compute_pressure = graph.graph.get("compute_delta_nfr")
        if not callable(compute_pressure):
            raise RuntimeError("scale graph has no callable DeltaNFR hook")
        compute_pressure(graph)

    def _apply_cross_scale_coupling(self) -> None:
        """Compose cross-scale contributions into stored nodal pressure.

        Source means are snapshotted before any target update.  The result is
        independent of scale iteration order and exposes the full pressure used
        by the subsequent nodal-equation step.
        """
        mean_pressure = {
            scale_name: _finite_signed_mean(
                (
                    get_attr(data, ALIAS_DNFR, 0.0, strict=True)
                    for _, data in graph.nodes(data=True)
                ),
                name=f"{scale_name} pressure",
            )
            for scale_name, graph in self.networks_by_scale.items()
        }

        proposals: dict[str, dict[NodeId, float]] = {}
        for target_scale, target_graph in self.networks_by_scale.items():
            contributions = [
                self._cross_scale_coupling(target_scale, source_scale)
                * mean_pressure[source_scale]
                for source_scale in self.networks_by_scale
                if source_scale != target_scale
            ]
            cross_contribution = self._finite_pressure_sum(
                contributions,
                name=f"{target_scale} cross-scale pressure",
            )
            proposals[target_scale] = {
                node: self._finite_pressure_sum(
                    (
                        get_attr(data, ALIAS_DNFR, 0.0, strict=True),
                        cross_contribution,
                    ),
                    name=f"{target_scale} node pressure",
                )
                for node, data in target_graph.nodes(data=True)
            }

        for scale_name, node_pressures in proposals.items():
            self._commit_pressure_field(
                self.networks_by_scale[scale_name], node_pressures
            )

    def _cross_scale_coupling(self, target_scale: str, source_scale: str) -> float:
        """Read one finite coupling coefficient from the internal matrix."""

        value = self.cross_scale_couplings.get((target_scale, source_scale), 0.0)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("cross-scale coupling must be a finite real scalar")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("cross-scale coupling must be finite")
        if not 0.0 <= normalized <= 1.0:
            raise ValueError("cross-scale coupling must be in [0, 1]")
        return normalized

    @staticmethod
    def _finite_pressure_sum(values: Iterable[float], *, name: str) -> float:
        """Return a representable finite sum of finite pressure terms."""

        normalized = tuple(
            _finite_signed_mean((value,), name=name) for value in values
        )
        try:
            result = math.fsum(normalized)
        except OverflowError as exc:
            raise ValueError(f"{name} exceeds the finite scalar range") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} exceeds the finite scalar range")
        return result

    @staticmethod
    def _commit_pressure_field(
        graph: TNFRGraph, pressures: dict[NodeId, float]
    ) -> None:
        """Commit one pressure field through alias and maximum-cache hooks."""

        graph.graph.pop("_dnfrmax", None)
        graph.graph.pop("_dnfrmax_node", None)
        for node, pressure in pressures.items():
            set_dnfr(graph, node, pressure)

    def _scale_coherence(self, G: TNFRGraph) -> float:
        """Return canonical per-scale C(t), including pressure and EPI rate."""
        from ..metrics.common import compute_coherence

        return float(compute_coherence(G))

    def _compute_cross_scale_synchrony(self) -> float:
        """Compute cross-scale phase synchronization."""
        if len(self.networks_by_scale) < 2:
            return 0.0

        # Simplified: compare mean phases across scales
        scale_mean_phases = []
        for G in self.networks_by_scale.values():
            phases = [G.nodes[n]["phase"] for n in G.nodes()]
            if phases:
                # Use circular mean for phases
                mean_phase = np.angle(np.mean(np.exp(1j * np.array(phases))))
                scale_mean_phases.append(mean_phase)

        if len(scale_mean_phases) < 2:
            return 0.0

        # Compute phase coherence between scales from shortest-arc separation.
        phase_diffs = []
        for i in range(len(scale_mean_phases)):
            for j in range(i + 1, len(scale_mean_phases)):
                phase_diffs.append(
                    abs(angle_diff(scale_mean_phases[i], scale_mean_phases[j]))
                )

        mean_phase_diff = np.mean(phase_diffs) if phase_diffs else 0.0
        # Convert to synchrony metric (0 = no sync, 1 = perfect sync)
        synchrony = 1.0 - (mean_phase_diff / np.pi)

        return max(0.0, synchrony)

    def get_scale_network(self, scale_name: str) -> TNFRGraph:
        """Get the network graph for a specific scale.

        Parameters
        ----------
        scale_name : str
            Name of the scale

        Returns
        -------
        TNFRGraph
            NetworkX graph for the specified scale
        """
        if scale_name not in self.networks_by_scale:
            raise ValueError(f"Unknown scale: {scale_name}")
        return self.networks_by_scale[scale_name]

    def memory_footprint(self) -> dict[str, float]:
        """Estimate memory usage per scale.

        Returns
        -------
        dict[str, float]
            Memory usage in MB for each scale
        """
        footprint = {}
        for scale_name, G in self.networks_by_scale.items():
            # Rough estimate: graph structure + node attributes
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            # NetworkX overhead + node dict + edge dict + attributes
            # Each node: ~200 bytes (dict overhead) + 4 attributes * 8 bytes
            # Each edge: ~100 bytes (dict overhead) + 1 attribute * 8 bytes
            estimate_bytes = n_nodes * 232 + n_edges * 108
            estimate_mb = estimate_bytes / (1024 * 1024)

            footprint[scale_name] = estimate_mb

        footprint["total"] = sum(v for k, v in footprint.items() if k != "total")
        return footprint

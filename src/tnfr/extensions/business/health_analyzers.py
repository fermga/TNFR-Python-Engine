"""Specialized health analyzers for business domain."""

from typing import Any, Dict, List
import networkx as nx


class ProcessHealthAnalyzer:
    """Specialized health analyzer for business process contexts.

    Computes domain-specific health dimensions beyond standard coherence
    and sense index metrics, focusing on process efficiency, organizational
    alignment, and change readiness.

    Examples
    --------
    >>> analyzer = ProcessHealthAnalyzer()
    >>> G = nx.Graph()
    >>> # ... set up network ...
    >>> metrics = analyzer.analyze_process_health(
    ...     G, ["reception", "contraction", "coupling"]
    ... )
    >>> print(metrics["efficiency_potential"])
    0.82
    """

    def analyze_process_health(
        self, G: nx.Graph, sequence: List[str], **kwargs: Any
    ) -> Dict[str, float]:
        """Compute process-specific health metrics.

        Parameters
        ----------
        G : nx.Graph
            Network graph representing business system.
        sequence : List[str]
            Operator sequence to analyze.
        **kwargs : Any
            Additional analysis parameters.

        Returns
        -------
        Dict[str, float]
            Domain-specific health metrics with values in [0, 1]:
            - efficiency_potential: Capacity for process improvement
            - change_readiness: Readiness for organizational change
            - alignment_strength: Degree of organizational alignment
        """
        metrics = {}

        # Compute business dimensions
        metrics["efficiency_potential"] = self._calculate_efficiency_potential(G, sequence)
        metrics["change_readiness"] = self._calculate_change_readiness(G, sequence)
        metrics["alignment_strength"] = self._calculate_alignment_strength(G, sequence)

        return metrics

    def _calculate_efficiency_potential(self, G: nx.Graph, sequence: List[str]) -> float:
        """Calculate capacity for process efficiency improvement.

        Parameters
        ----------
        G : nx.Graph
            Network graph.
        sequence : List[str]
            Operator sequence.

        Returns
        -------
        float
            Efficiency potential score [0, 1].
        """
        # Check for optimization operators
        optimization_ops = {"contraction", "coupling", "resonance"}
        optimization_count = sum(1 for op in sequence if op in optimization_ops)

        # Check for analysis operators
        analysis_ops = {"reception", "coherence"}
        analysis_count = sum(1 for op in sequence if op in analysis_ops)

        if len(sequence) == 0:
            return 0.0

        # Balance between analysis and optimization
        optimization_ratio = optimization_count / len(sequence)
        analysis_ratio = analysis_count / len(sequence)

        balance_score = min(optimization_ratio + analysis_ratio, 1.0)

        # Factor in network efficiency (shorter paths = more efficient)
        if len(G.nodes()) > 1 and nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            # Lower average path = higher efficiency
            efficiency_score = 1.0 / (1.0 + avg_path_length / len(G.nodes()))
        else:
            efficiency_score = 0.5

        # Combine factors
        efficiency_potential = 0.6 * balance_score + 0.4 * efficiency_score

        return min(efficiency_potential, 1.0)

    def _calculate_change_readiness(self, G: nx.Graph, sequence: List[str]) -> float:
        """Calculate readiness for organizational change.

        Parameters
        ----------
        G : nx.Graph
            Network graph.
        sequence : List[str]
            Operator sequence.

        Returns
        -------
        float
            Change readiness score [0, 1].
        """
        # Check for change-enabling operators
        change_ops = {"dissonance", "mutation", "expansion", "self_organization"}
        change_count = sum(1 for op in sequence if op in change_ops)

        # Check for stabilizing operators (needed for sustainable change)
        stability_ops = {"coherence", "coupling"}
        stability_count = sum(1 for op in sequence if op in stability_ops)

        if len(sequence) == 0:
            return 0.0

        # Change readiness requires both disruption and stabilization
        change_ratio = change_count / len(sequence)
        stability_count / len(sequence)

        # Need balance - too much disruption is risky
        if change_count > 0:
            balance = min(stability_count / change_count, 1.0)
            readiness_base = 0.5 * change_ratio + 0.5 * balance
        else:
            # No change operators = low readiness
            readiness_base = 0.3

        # Network connectivity aids change propagation
        if len(G.nodes()) > 0:
            avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
            connectivity_factor = min(avg_degree / 4.0, 1.0)
        else:
            connectivity_factor = 0.0

        change_readiness = 0.7 * readiness_base + 0.3 * connectivity_factor

        return min(change_readiness, 1.0)

    def _calculate_alignment_strength(self, G: nx.Graph, sequence: List[str]) -> float:
        """Calculate degree of organizational alignment.

        Parameters
        ----------
        G : nx.Graph
            Network graph.
        sequence : List[str]
            Operator sequence.

        Returns
        -------
        float
            Alignment strength score [0, 1].
        """
        # Alignment built through communication and coordination
        alignment_ops = {"emission", "reception", "coupling", "resonance"}
        alignment_count = sum(1 for op in sequence if op in alignment_ops)

        # Coherence strengthens alignment
        coherence_count = sequence.count("coherence")

        if len(sequence) == 0:
            return 0.0

        # More alignment operators = stronger alignment
        alignment_ratio = alignment_count / len(sequence)

        # Coherence multiplier
        coherence_bonus = min(coherence_count * 0.1, 0.3)

        # Network cohesion as proxy for alignment
        if len(G.nodes()) > 2:
            try:
                # Use clustering coefficient as cohesion proxy
                avg_clustering = nx.average_clustering(G)
                cohesion_score = avg_clustering
            except (nx.NetworkXError, ZeroDivisionError):
                cohesion_score = 0.5
        else:
            cohesion_score = 0.5

        # Combine factors
        alignment_strength = (0.5 * alignment_ratio + 0.3 * cohesion_score + 0.2) + coherence_bonus

        return min(alignment_strength, 1.0)

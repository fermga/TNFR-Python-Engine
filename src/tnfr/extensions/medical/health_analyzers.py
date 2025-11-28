"""Specialized health analyzers for medical domain."""

from typing import Any, Dict, List
import networkx as nx


class TherapeuticHealthAnalyzer:
    """Specialized health analyzer for therapeutic contexts.

    Computes domain-specific health dimensions beyond standard coherence
    and sense index metrics, focusing on therapeutic effectiveness and
    patient safety.

    Examples
    --------
    >>> analyzer = TherapeuticHealthAnalyzer()
    >>> G = nx.Graph()
    >>> # ... set up network ...
    >>> metrics = analyzer.analyze_therapeutic_health(
    ...     G, ["emission", "reception", "coherence"]
    ... )
    >>> print(metrics["healing_potential"])
    0.78
    """

    def analyze_therapeutic_health(
        self, G: nx.Graph, sequence: List[str], **kwargs: Any
    ) -> Dict[str, float]:
        """Compute therapeutic-specific health metrics.

        Parameters
        ----------
        G : nx.Graph
            Network graph representing therapeutic system.
        sequence : List[str]
            Operator sequence to analyze.
        **kwargs : Any
            Additional analysis parameters.

        Returns
        -------
        Dict[str, float]
            Domain-specific health metrics with values in [0, 1]:
            - healing_potential: Capacity for positive change
            - trauma_safety: Safety from re-traumatization
            - therapeutic_alliance: Strength of working relationship
        """
        metrics = {}

        # Compute therapeutic dimensions
        metrics["healing_potential"] = self._calculate_healing_potential(G, sequence)
        metrics["trauma_safety"] = self._calculate_trauma_safety(G, sequence)
        metrics["therapeutic_alliance"] = self._calculate_alliance_strength(G, sequence)

        return metrics

    def _calculate_healing_potential(self, G: nx.Graph, sequence: List[str]) -> float:
        """Calculate capacity for positive therapeutic change.

        Parameters
        ----------
        G : nx.Graph
            Network graph.
        sequence : List[str]
            Operator sequence.

        Returns
        -------
        float
            Healing potential score [0, 1].
        """
        # Check for growth-promoting operators
        growth_ops = {"expansion", "self_organization", "coupling"}
        growth_count = sum(1 for op in sequence if op in growth_ops)

        # Check for stabilizing operators
        stability_ops = {"coherence", "resonance"}
        stability_count = sum(1 for op in sequence if op in stability_ops)

        # Balance between growth and stability
        if len(sequence) == 0:
            return 0.0

        growth_ratio = growth_count / len(sequence)
        stability_ratio = stability_count / len(sequence)

        # Optimal balance: some growth, some stability
        balance_score = min(growth_ratio + stability_ratio, 1.0)

        # Factor in network connectivity (more connections = more resources)
        if len(G.nodes()) > 0:
            avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
            connectivity_score = min(avg_degree / 5.0, 1.0)
        else:
            connectivity_score = 0.0

        # Combine factors
        healing_potential = 0.6 * balance_score + 0.4 * connectivity_score

        return min(healing_potential, 1.0)

    def _calculate_trauma_safety(self, G: nx.Graph, sequence: List[str]) -> float:
        """Calculate safety from re-traumatization.

        Parameters
        ----------
        G : nx.Graph
            Network graph.
        sequence : List[str]
            Operator sequence.

        Returns
        -------
        float
            Trauma safety score [0, 1].
        """
        # Check for potentially destabilizing operators
        destabilizing_ops = {"dissonance", "mutation"}
        destabilizing_count = sum(1 for op in sequence if op in destabilizing_ops)

        # Check for safety-promoting operators
        safety_ops = {"silence", "coherence", "reception"}
        safety_count = sum(1 for op in sequence if op in safety_ops)

        if len(sequence) == 0:
            return 1.0  # No sequence = no risk

        # Safety depends on having stabilizing ops when using destabilizing ones
        if destabilizing_count > 0:
            # Need at least as many safety ops as destabilizing ops
            safety_ratio = min(safety_count / destabilizing_count, 1.0)
            base_safety = 0.5 + 0.5 * safety_ratio
        else:
            # No destabilizing ops = inherently safer
            base_safety = 0.8 + 0.2 * (safety_count / len(sequence))

        return min(base_safety, 1.0)

    def _calculate_alliance_strength(self, G: nx.Graph, sequence: List[str]) -> float:
        """Calculate strength of therapeutic alliance.

        Parameters
        ----------
        G : nx.Graph
            Network graph.
        sequence : List[str]
            Operator sequence.

        Returns
        -------
        float
            Alliance strength score [0, 1].
        """
        # Alliance built through connection operators
        connection_ops = {"emission", "reception", "coupling", "resonance"}
        connection_count = sum(1 for op in sequence if op in connection_ops)

        # Coherence strengthens alliance
        coherence_count = sequence.count("coherence")

        if len(sequence) == 0:
            return 0.0

        # More connection = stronger alliance
        connection_ratio = connection_count / len(sequence)

        # Coherence multiplier
        coherence_bonus = min(coherence_count * 0.1, 0.3)

        # Network density as proxy for mutual understanding
        if len(G.nodes()) > 1:
            density = nx.density(G)
        else:
            density = 0.0

        # Combine factors
        alliance_strength = (0.5 * connection_ratio + 0.3 * density + 0.2) + coherence_bonus

        return min(alliance_strength, 1.0)

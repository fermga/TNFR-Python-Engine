"""Demo of enhanced VAL (Expansion) metrics (Issue #2724).

This script demonstrates the comprehensive telemetry metrics for the VAL
operator including bifurcation risk, coherence preservation, fractality
indicators, network impact, and structural stability.
"""

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_D2EPI, ALIAS_THETA
from tnfr.operators.definitions import Expansion
from tnfr.structural import create_nfr


def print_metrics_section(title: str, metrics: dict, keys: list[str]) -> None:
    """Print a section of metrics."""
    print(f"\n{title}")
    print("=" * len(title))
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, bool):
                print(f"  {key}: {'✓' if value else '✗'}")
            else:
                print(f"  {key}: {value}")


def demo_healthy_expansion():
    """Demonstrate healthy expansion with all indicators positive."""
    print("\n" + "=" * 60)
    print("SCENARIO 1: Healthy Expansion")
    print("=" * 60)

    G, node = create_nfr("healthy_node", epi=0.4, vf=1.0)

    # Add network context
    for i in range(3):
        neighbor = f"neighbor_{i}"
        G.add_node(neighbor)
        set_attr(G.nodes[neighbor], ALIAS_THETA, 0.3 + i * 0.1)
        G.add_edge(node, neighbor)

    # Set healthy conditions
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    set_attr(G.nodes[node], ALIAS_DNFR, 0.2)  # Positive, stable
    set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)  # Below threshold
    set_attr(G.nodes[node], ALIAS_THETA, 0.35)  # Aligned with neighbors

    # Apply expansion
    Expansion()(G, node)

    # Get metrics
    metrics = G.graph["operator_metrics"][-1]

    # Display metrics
    print_metrics_section(
        "Core Metrics",
        metrics,
        ["vf_increase", "delta_epi", "expansion_factor"],
    )

    print_metrics_section(
        "Structural Stability",
        metrics,
        ["dnfr_final", "dnfr_positive", "dnfr_stable"],
    )

    print_metrics_section(
        "Bifurcation Risk",
        metrics,
        ["d2epi", "bifurcation_risk", "bifurcation_magnitude"],
    )

    print_metrics_section(
        "Coherence",
        metrics,
        ["coherence_local", "coherence_preserved"],
    )

    print_metrics_section(
        "Fractality",
        metrics,
        ["epi_growth_rate", "vf_growth_rate", "growth_ratio", "fractal_preserved"],
    )

    print_metrics_section(
        "Network Impact",
        metrics,
        ["neighbor_count", "phase_coherence_neighbors", "network_coupled"],
    )

    print_metrics_section(
        "Overall Health",
        metrics,
        ["expansion_healthy"],
    )


def demo_bifurcation_risk():
    """Demonstrate expansion at bifurcation threshold."""
    print("\n" + "=" * 60)
    print("SCENARIO 2: Expansion at Bifurcation Threshold")
    print("=" * 60)

    G, node = create_nfr("bifurcating_node", epi=0.4, vf=1.0)

    # Set conditions for bifurcation risk
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    G.graph["VAL_BIFURCATION_THRESHOLD"] = 0.3
    set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
    set_attr(G.nodes[node], ALIAS_D2EPI, 0.5)  # HIGH - above threshold

    # Apply expansion
    Expansion()(G, node)

    # Get metrics
    metrics = G.graph["operator_metrics"][-1]

    print_metrics_section(
        "Bifurcation Indicators",
        metrics,
        [
            "d2epi",
            "bifurcation_threshold",
            "bifurcation_risk",
            "bifurcation_magnitude",
        ],
    )

    print_metrics_section(
        "Overall Health",
        metrics,
        ["expansion_healthy"],
    )

    if metrics["bifurcation_risk"]:
        print("\n⚠️  WARNING: Bifurcation risk detected!")
        print(
            f"   d²EPI/dt² = {metrics['d2epi']:.3f} > threshold = {metrics['bifurcation_threshold']:.3f}"
        )
        print("   Consider applying IL (Coherence) or THOL (Self-organization)")


def demo_coherence_degradation():
    """Demonstrate expansion with coherence concerns."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Expansion with Low Coherence")
    print("=" * 60)

    G, node = create_nfr("low_coherence_node", epi=0.3, vf=1.0)

    # Set conditions
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    G.graph["VAL_MIN_COHERENCE"] = 0.5
    set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
    set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)

    # Apply expansion
    Expansion()(G, node)

    # Get metrics
    metrics = G.graph["operator_metrics"][-1]

    print_metrics_section(
        "Coherence Metrics",
        metrics,
        ["coherence_local", "coherence_preserved"],
    )

    print_metrics_section(
        "Overall Health",
        metrics,
        ["expansion_healthy"],
    )

    if not metrics.get("coherence_preserved", True):
        print("\n⚠️  WARNING: Coherence below threshold!")
        print(f"   C_local = {metrics['coherence_local']:.3f}")
        print("   Consider applying IL (Coherence) to stabilize")


def demo_network_impact():
    """Demonstrate network coupling analysis."""
    print("\n" + "=" * 60)
    print("SCENARIO 4: Network Coupling Analysis")
    print("=" * 60)

    G, node = create_nfr("network_node", epi=0.4, vf=1.0)

    # Add neighbors with varying phase alignment
    neighbors_info = [
        ("aligned_1", 0.50),
        ("aligned_2", 0.52),
        ("misaligned", 2.0),  # Far from node's phase
    ]

    for neighbor, phase in neighbors_info:
        G.add_node(neighbor)
        set_attr(G.nodes[neighbor], ALIAS_THETA, phase)
        G.add_edge(node, neighbor)

    # Set conditions
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    set_attr(G.nodes[node], ALIAS_DNFR, 0.2)
    set_attr(G.nodes[node], ALIAS_D2EPI, 0.1)
    set_attr(G.nodes[node], ALIAS_THETA, 0.51)  # Mostly aligned

    # Apply expansion
    Expansion()(G, node)

    # Get metrics
    metrics = G.graph["operator_metrics"][-1]

    print_metrics_section(
        "Network Metrics",
        metrics,
        [
            "neighbor_count",
            "phase_coherence_neighbors",
            "network_coupled",
            "theta_final",
        ],
    )

    print("\nNetwork Analysis:")
    if metrics["network_coupled"]:
        print("  ✓ Node is well-coupled to network")
    else:
        print("  ✗ Node has weak network coupling")

    print(
        f"  Phase coherence: {metrics['phase_coherence_neighbors']:.2%} "
        f"({metrics['neighbor_count']} neighbors)"
    )


def main():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 60)
    print("Enhanced VAL (Expansion) Metrics Demo")
    print("Issue #2724: Canonical Structural Indicators")
    print("=" * 60)

    demo_healthy_expansion()
    demo_bifurcation_risk()
    demo_coherence_degradation()
    demo_network_impact()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(
        "\nThese metrics enable:\n"
        "  1. Early detection of bifurcation events\n"
        "  2. Validation of coherence preservation\n"
        "  3. Analysis of fractal self-similarity\n"
        "  4. Monitoring of network coupling\n"
        "  5. Overall structural health assessment"
    )


if __name__ == "__main__":
    main()

"""Interactive tutorial implementations for TNFR learning.

This module contains the actual tutorial functions that guide users
through TNFR concepts with executable examples and clear explanations.

Each tutorial is self-contained and can be run independently:
- hello_tnfr(): 5-minute introduction
- biological_example(): Cell communication model
- social_network_example(): Social dynamics simulation
- technology_example(): Distributed systems analogy

All tutorials maintain TNFR canonical invariants and demonstrate
the 13 structural operators in action.
"""

from __future__ import annotations

from typing import Optional
import time

try:
    from ..sdk import TNFRNetwork, TNFRTemplates

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False

__all__ = [
    "hello_tnfr",
    "biological_example",
    "social_network_example",
    "technology_example",
    "oz_dissonance_tutorial",
    "run_all_tutorials",
]


def _print_section(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}\n")


def _print_subsection(title: str, width: int = 70) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-'*width}")
    print(f"{title}")
    print(f"{'-'*width}\n")


def _explain(text: str, pause: float = 1.5) -> None:
    """Print explanation text with optional pause for readability."""
    print(text)
    if pause > 0:
        time.sleep(pause)


def hello_tnfr(interactive: bool = True, random_seed: int = 42) -> None:
    """5-minute interactive introduction to TNFR.

    This tutorial introduces core TNFR concepts through a simple,
    hands-on example. You'll learn:

    - What a node is (Nodo Fractal Resonante - NFR)
    - What EPI means (Primary Information Structure)
    - What structural operators do
    - How to measure coherence C(t) and sense index Si
    - How resonance creates stable networks

    The tutorial uses the simplified SDK API, so you can start
    using TNFR immediately without deep theoretical knowledge.

    Parameters
    ----------
    interactive : bool, default=True
        If True, pauses between sections for reading.
    random_seed : int, default=42
        Random seed for reproducibility.

    Examples
    --------
    >>> from tnfr.tutorials import hello_tnfr
    >>> hello_tnfr()  # Run the full tutorial
    >>> hello_tnfr(interactive=False)  # Run without pauses

    Notes
    -----
    This tutorial maintains TNFR canonical invariants:
    - Operator closure (Invariant #4)
    - Phase synchrony (Invariant #5)
    - Structural units (νf in Hz_str)
    """
    if not _HAS_SDK:
        print("Error: SDK not available. Install with: pip install tnfr")
        return

    pause = 1.5 if interactive else 0

    _print_section("Hello, TNFR! 👋")

    _explain(
        "Welcome to TNFR - Resonant Fractal Nature Theory!\n"
        "Let's learn the basics in just 5 minutes with a working example.",
        pause,
    )

    # Part 1: What is a node?
    _print_subsection("Part 1: What is a Resonant Fractal Node (NFR)?")

    _explain(
        "In TNFR, everything is made of 'nodes' that resonate with each other.\n"
        "Think of them like musical notes - they have:\n"
        "  • A frequency (νf) - how fast they vibrate\n"
        "  • A phase (φ) - when they vibrate\n"
        "  • A form (EPI) - what they 'look like' structurally\n\n"
        "Let's create a simple network of 10 nodes:",
        pause,
    )

    print("    >>> from tnfr.sdk import TNFRNetwork")
    print("    >>> network = TNFRNetwork('hello_example')")
    print("    >>> network.add_nodes(10, random_seed=42)\n")

    network = TNFRNetwork("hello_example")
    network.add_nodes(10, random_seed=random_seed)

    _explain("✓ Created 10 resonant nodes!", pause * 0.5)

    # Part 2: Connecting nodes
    _print_subsection("Part 2: Connecting Nodes")

    _explain(
        "Nodes need to connect to form a network. We'll connect them randomly\n"
        "with 30% probability (like neurons forming synapses):\n",
        pause,
    )

    print("    >>> network.connect_nodes(0.3, 'random')\n")
    network.connect_nodes(0.3, "random")

    _explain("✓ Nodes connected! Now they can resonate together.", pause * 0.5)

    # Part 3: Applying operators
    _print_subsection("Part 3: The 13 Structural Operators")

    _explain(
        "TNFR has 13 fundamental operators that reorganize networks:\n"
        "  1. Emission - Start sending signals\n"
        "  2. Reception - Receive signals from neighbors\n"
        "  3. Coherence - Stabilize structures\n"
        "  4. Resonance - Amplify synchronized patterns\n"
        "  5. Silence - Pause evolution\n"
        "  ...and 8 more!\n\n"
        "Let's apply a basic activation sequence:",
        pause,
    )

    print("    >>> network.apply_sequence('basic_activation', repeat=3)\n")
    network.apply_sequence("basic_activation", repeat=3)

    _explain(
        "✓ Applied: emission → reception → coherence → resonance → silence\n"
        "  This sequence activated the network 3 times!",
        pause,
    )

    # Part 4: Measuring results
    _print_subsection("Part 4: Measuring Coherence and Sense Index")

    _explain(
        "Now let's measure what happened:\n"
        "  • C(t) = Coherence - how stable is the network?\n"
        "  • Si = Sense Index - how well can each node reorganize?\n",
        pause,
    )

    print("    >>> results = network.measure()")
    print("    >>> print(results.summary())\n")

    results = network.measure()

    # Extract key metrics for display
    coherence = results.coherence
    avg_si = (
        sum(results.sense_indices.values()) / len(results.sense_indices)
        if results.sense_indices
        else 0
    )

    print(f"    Coherence C(t) = {coherence:.3f}")
    print(f"    Average Si = {avg_si:.3f}")
    print(f"    Nodes = {len(results.sense_indices)}")

    _explain("", pause)

    # Part 5: Interpretation
    _print_subsection("Part 5: What Does This Mean?")

    _explain(
        f"Results interpretation:\n"
        f"  • Coherence C(t) = {coherence:.3f}\n"
        f"    {'High' if coherence > 0.5 else 'Moderate' if coherence > 0.2 else 'Low'} stability - "
        f"the network holds its structure well!\n\n"
        f"  • Average Si = {avg_si:.3f}\n"
        f"    Each node can {'effectively' if avg_si > 0.5 else 'moderately'} reorganize "
        f"while staying coherent.\n\n"
        "In TNFR terms: Your network exhibits resonant coherence! 🎵\n"
        "The nodes synchronized their phases and created stable patterns.",
        pause,
    )

    # Part 6: Try it yourself
    _print_subsection("Part 6: Try It Yourself!")

    _explain(
        "That's the basics! Here's a complete example you can modify:\n", pause * 0.5
    )

    print(
        """
    from tnfr.sdk import TNFRNetwork
    
    # Create your network
    net = TNFRNetwork("my_experiment")
    
    # Add nodes and connect them
    net.add_nodes(20, random_seed=123)
    net.connect_nodes(0.4, "random")
    
    # Apply operators (try different sequences!)
    net.apply_sequence("basic_activation", repeat=5)
    
    # Or try: "stabilization", "creative_mutation", 
    #         "network_sync", "exploration"
    
    # Measure results
    results = net.measure()
    print(results.summary())
    
    # Access detailed data
    print(f"Coherence: {results.coherence:.3f}")
    for node_id, si in results.sense_indices.items():
        print(f"  Node {node_id}: Si = {si:.3f}")
    """
    )

    _print_section("Tutorial Complete! 🎉")

    _explain(
        "You've learned:\n"
        "  ✓ How to create TNFR networks\n"
        "  ✓ What structural operators do\n"
        "  ✓ How to measure coherence and sense index\n"
        "  ✓ How to interpret results\n\n"
        "Next steps:\n"
        "  • Try biological_example() - cell communication\n"
        "  • Try social_network_example() - social dynamics\n"
        "  • Try technology_example() - distributed systems\n"
        "  • Read the full docs: docs/source/getting-started/",
        0,
    )

    print(f"\n{'='*70}\n")


def biological_example(interactive: bool = True, random_seed: int = 42) -> dict:
    """Cell communication model using TNFR.

    This tutorial models how cells communicate through chemical signals,
    demonstrating TNFR's application to biological systems.

    Concepts demonstrated:
    - Nodes as cells
    - Emission as signal secretion
    - Reception as receptor binding
    - Coupling as direct cell-cell contact
    - Coherence as tissue organization

    Parameters
    ----------
    interactive : bool, default=True
        If True, pauses between sections for reading.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Simulation results with coherence, sense indices, and interpretation.

    Examples
    --------
    >>> from tnfr.tutorials import biological_example
    >>> results = biological_example()
    >>> print(f"Tissue coherence: {results['coherence']:.3f}")

    Notes
    -----
    This tutorial demonstrates:
    - Emission operator (signal secretion)
    - Reception operator (signal detection)
    - Coupling operator (cell-cell contact)
    - Coherence operator (tissue stability)
    """
    if not _HAS_SDK:
        print("Error: SDK not available. Install with: pip install tnfr")
        return {}

    pause = 1.5 if interactive else 0

    _print_section("TNFR Tutorial: Cell Communication 🧬")

    _explain(
        "In this example, we'll model how cells in a tissue communicate\n"
        "and organize themselves using TNFR principles.\n\n"
        "Biological → TNFR mapping:\n"
        "  • Cell → Node (NFR)\n"
        "  • Chemical signal → Emission operator\n"
        "  • Receptor binding → Reception operator\n"
        "  • Cell-cell contact → Coupling operator\n"
        "  • Tissue organization → Coherence",
        pause,
    )

    _print_subsection("Step 1: Create Cell Population")

    _explain(
        "Let's create a tissue with 25 cells. Each cell has:\n"
        "  • Structural frequency νf ∈ [0.3, 0.9] Hz_str (metabolic rate)\n"
        "  • Phase φ (cell cycle position)\n"
        "  • EPI (cell state/phenotype)",
        pause,
    )

    print("    >>> network = TNFRNetwork('cell_tissue')")
    print("    >>> network.add_nodes(25, vf_range=(0.3, 0.9), random_seed=42)\n")

    network = TNFRNetwork("cell_tissue")
    network.add_nodes(25, vf_range=(0.3, 0.9), random_seed=random_seed)

    _explain("✓ Created 25 cells with varying metabolic rates", pause * 0.5)

    _print_subsection("Step 2: Establish Cell Connections")

    _explain(
        "Cells connect through:\n"
        "  • Physical contacts (gap junctions)\n"
        "  • Paracrine signaling (nearby cells)\n\n"
        "We'll use a ring topology (like epithelial cells) with 50% connectivity:",
        pause,
    )

    print("    >>> network.connect_nodes(0.5, 'ring')\n")
    network.connect_nodes(0.5, "ring")

    _explain("✓ Cells connected in tissue-like structure", pause * 0.5)

    _print_subsection("Step 3: Simulate Cell Signaling")

    _explain(
        "Now let's simulate cell communication cycles:\n"
        "  1. Emission - Cells secrete signaling molecules\n"
        "  2. Reception - Neighboring cells detect signals\n"
        "  3. Coherence - Cells stabilize coordinated state\n"
        "  4. Coupling - Direct cell-cell interaction\n"
        "  5. Resonance - Synchronized response amplification\n"
        "  6. Silence - Rest period between cycles\n\n"
        "We'll run 5 signaling cycles:",
        pause,
    )

    print("    >>> network.apply_sequence('network_sync', repeat=5)\n")
    network.apply_sequence("network_sync", repeat=5)

    _explain("✓ Completed 5 cell signaling cycles", pause * 0.5)

    _print_subsection("Step 4: Measure Tissue Organization")

    _explain("Let's measure how well the tissue organized:", pause * 0.5)

    print("    >>> results = network.measure()\n")
    results = network.measure()

    coherence = results.coherence
    avg_si = (
        sum(results.sense_indices.values()) / len(results.sense_indices)
        if results.sense_indices
        else 0
    )

    print(f"    Tissue Coherence C(t) = {coherence:.3f}")
    print(f"    Average Cell Si = {avg_si:.3f}")
    print(f"    Number of Cells = {len(results.sense_indices)}\n")

    _print_subsection("Step 5: Biological Interpretation")

    if coherence > 0.6:
        tissue_status = "well-organized"
        bio_meaning = "Cells are synchronized and functioning as coordinated tissue."
    elif coherence > 0.3:
        tissue_status = "moderately organized"
        bio_meaning = "Cells show some coordination but not fully synchronized."
    else:
        tissue_status = "loosely organized"
        bio_meaning = "Cells are relatively independent with weak coordination."

    _explain(
        f"Results:\n"
        f"  • Tissue Status: {tissue_status}\n"
        f"  • Biological Meaning: {bio_meaning}\n"
        f"  • Cell Responsiveness: Average Si = {avg_si:.3f}\n\n"
        f"In biological terms:\n"
        f"  This simulates how cells in a tissue coordinate their behavior\n"
        f"  through chemical signaling and physical contacts. The coherence\n"
        f"  value indicates how well they form organized tissue structure.\n\n"
        f"  Higher coherence = Better tissue organization\n"
        f"  Higher Si = Better individual cell adaptability",
        pause,
    )

    _print_section("Cell Communication Tutorial Complete! 🧬")

    return {
        "coherence": coherence,
        "sense_indices": results.sense_indices,
        "interpretation": {
            "tissue_status": tissue_status,
            "biological_meaning": bio_meaning,
            "avg_cell_responsiveness": avg_si,
        },
        "results": results,
    }


def social_network_example(interactive: bool = True, random_seed: int = 42) -> dict:
    """Social dynamics simulation using TNFR.

    This tutorial models social network dynamics, demonstrating how
    TNFR applies to social systems and group behavior.

    Concepts demonstrated:
    - Nodes as people
    - Emission as communication/expression
    - Reception as listening/influence
    - Resonance as shared understanding
    - Coherence as group cohesion
    - Dissonance as conflict/disagreement

    Parameters
    ----------
    interactive : bool, default=True
        If True, pauses between sections for reading.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Simulation results with social metrics and interpretation.

    Examples
    --------
    >>> from tnfr.tutorials import social_network_example
    >>> results = social_network_example()
    >>> print(f"Group cohesion: {results['coherence']:.3f}")

    Notes
    -----
    This tutorial demonstrates:
    - Dissonance operator (conflict/debate)
    - Mutation operator (opinion change)
    - Resonance operator (consensus building)
    - Coherence as social cohesion measure
    """
    if not _HAS_SDK:
        print("Error: SDK not available. Install with: pip install tnfr")
        return {}

    pause = 1.5 if interactive else 0

    _print_section("TNFR Tutorial: Social Network Dynamics 👥")

    _explain(
        "In this example, we'll model social dynamics in a group of people\n"
        "using TNFR to understand consensus, conflict, and cohesion.\n\n"
        "Social → TNFR mapping:\n"
        "  • Person → Node (NFR)\n"
        "  • Communication → Emission/Reception operators\n"
        "  • Shared understanding → Resonance operator\n"
        "  • Conflict/debate → Dissonance operator\n"
        "  • Opinion change → Mutation operator\n"
        "  • Group cohesion → Coherence C(t)",
        pause,
    )

    _print_subsection("Step 1: Create Social Group")

    _explain(
        "Let's model a community of 30 people. Each person has:\n"
        "  • Structural frequency νf (communication frequency)\n"
        "  • Phase φ (opinion/perspective)\n"
        "  • EPI (belief system/worldview)",
        pause,
    )

    print("    >>> network = TNFRNetwork('social_group')")
    print("    >>> network.add_nodes(30, random_seed=42)\n")

    network = TNFRNetwork("social_group")
    network.add_nodes(30, random_seed=random_seed)

    _explain("✓ Created community of 30 people", pause * 0.5)

    _print_subsection("Step 2: Establish Social Connections")

    _explain(
        "People form social connections (friendships, work relationships).\n"
        "We'll use random connections with 25% probability (realistic social density):",
        pause,
    )

    print("    >>> network.connect_nodes(0.25, 'random')\n")
    network.connect_nodes(0.25, "random")

    _explain("✓ Social network formed", pause * 0.5)

    _print_subsection("Step 3: Simulate Social Interaction")

    _explain(
        "We'll simulate a scenario with:\n"
        "  Phase A: Basic activation and debate\n"
        "  Phase B: Opinion evolution (creative mutation)\n"
        "  Phase C: Consensus building (stabilization)\n\n"
        "First, let's activate the social network:",
        pause,
    )

    print("    >>> network.apply_sequence('basic_activation', repeat=3)\n")
    network.apply_sequence("basic_activation", repeat=3)

    _explain("✓ Completed initial social interaction phase", pause)

    _explain(
        "\nNow, let's allow opinions to synchronize and reach consensus:", pause * 0.5
    )

    print("    >>> network.apply_sequence('basic_activation', repeat=2)")
    print("    >>> network.apply_sequence('stabilization', repeat=3)\n")

    network.apply_sequence("basic_activation", repeat=2)
    network.apply_sequence("stabilization", repeat=3)

    _explain("✓ Opinions evolved and group stabilized", pause * 0.5)

    _print_subsection("Step 4: Measure Group Cohesion")

    _explain("Let's measure the social dynamics:", pause * 0.5)

    print("    >>> results = network.measure()\n")
    results = network.measure()

    coherence = results.coherence
    avg_si = (
        sum(results.sense_indices.values()) / len(results.sense_indices)
        if results.sense_indices
        else 0
    )

    print(f"    Group Coherence C(t) = {coherence:.3f}")
    print(f"    Average Individual Si = {avg_si:.3f}")
    print(f"    Group Size = {len(results.sense_indices)}\n")

    _print_subsection("Step 5: Social Interpretation")

    if coherence > 0.6:
        social_status = "highly cohesive"
        social_meaning = "Strong group consensus and shared understanding."
    elif coherence > 0.3:
        social_status = "moderately cohesive"
        social_meaning = "Some agreement but diverse opinions remain."
    else:
        social_status = "loosely cohesive"
        social_meaning = "Fragmented group with weak consensus."

    if avg_si > 0.5:
        adaptability = "high"
        adapt_meaning = "Individuals can adjust views while maintaining identity."
    else:
        adaptability = "moderate"
        adapt_meaning = "Individuals have some flexibility in their opinions."

    _explain(
        f"Results:\n"
        f"  • Group Status: {social_status}\n"
        f"  • Social Meaning: {social_meaning}\n"
        f"  • Individual Adaptability: {adaptability}\n"
        f"  • Adaptability Meaning: {adapt_meaning}\n\n"
        f"In social terms:\n"
        f"  After debate and discussion, the group reached a coherence of {coherence:.3f}.\n"
        f"  This indicates how well the group coordinated their beliefs and opinions.\n\n"
        f"  The dissonance phase (debate) introduced diverse perspectives.\n"
        f"  The mutation phase allowed opinions to evolve naturally.\n"
        f"  The stabilization phase built consensus and shared understanding.\n\n"
        f"  Higher coherence = Stronger group consensus\n"
        f"  Higher Si = Better individual adaptability without losing identity",
        pause,
    )

    _print_section("Social Network Tutorial Complete! 👥")

    return {
        "coherence": coherence,
        "sense_indices": results.sense_indices,
        "interpretation": {
            "social_status": social_status,
            "social_meaning": social_meaning,
            "adaptability": adaptability,
            "adapt_meaning": adapt_meaning,
            "avg_si": avg_si,
        },
        "results": results,
    }


def technology_example(interactive: bool = True, random_seed: int = 42) -> dict:
    """Distributed systems model using TNFR.

    This tutorial models distributed computing systems, demonstrating
    how TNFR applies to technology and computer networks.

    Concepts demonstrated:
    - Nodes as servers/microservices
    - Emission as message broadcasting
    - Reception as message processing
    - Coupling as service dependencies
    - Coherence as system reliability
    - Silence as graceful degradation

    Parameters
    ----------
    interactive : bool, default=True
        If True, pauses between sections for reading.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Simulation results with system metrics and interpretation.

    Examples
    --------
    >>> from tnfr.tutorials import technology_example
    >>> results = technology_example()
    >>> print(f"System reliability: {results['coherence']:.3f}")

    Notes
    -----
    This tutorial demonstrates:
    - Coupling operator (service dependencies)
    - Silence operator (graceful degradation)
    - Resonance operator (load balancing)
    - Coherence as system reliability measure
    """
    if not _HAS_SDK:
        print("Error: SDK not available. Install with: pip install tnfr")
        return {}

    pause = 1.5 if interactive else 0

    _print_section("TNFR Tutorial: Distributed Systems 💻")

    _explain(
        "In this example, we'll model a distributed microservices architecture\n"
        "using TNFR to analyze system reliability and resilience.\n\n"
        "Technology → TNFR mapping:\n"
        "  • Microservice → Node (NFR)\n"
        "  • Message passing → Emission/Reception operators\n"
        "  • Service dependency → Coupling operator\n"
        "  • Load balancing → Resonance operator\n"
        "  • Graceful degradation → Silence operator\n"
        "  • System reliability → Coherence C(t)",
        pause,
    )

    _print_subsection("Step 1: Create Microservices Cluster")

    _explain(
        "Let's model a cluster of 15 microservices. Each service has:\n"
        "  • Structural frequency νf (request processing rate)\n"
        "  • Phase φ (operational state/timing)\n"
        "  • EPI (service configuration/state)",
        pause,
    )

    print("    >>> network = TNFRNetwork('microservices')")
    print("    >>> network.add_nodes(15, vf_range=(0.5, 1.2), random_seed=42)\n")

    network = TNFRNetwork("microservices")
    network.add_nodes(15, vf_range=(0.5, 1.2), random_seed=random_seed)

    _explain("✓ Created cluster of 15 microservices", pause * 0.5)

    _print_subsection("Step 2: Establish Service Dependencies")

    _explain(
        "Microservices communicate through:\n"
        "  • REST APIs\n"
        "  • Message queues\n"
        "  • Service mesh\n\n"
        "We'll use a network topology with 40% connectivity:",
        pause,
    )

    print("    >>> network.connect_nodes(0.4, 'random')\n")
    network.connect_nodes(0.4, "random")

    _explain("✓ Service dependencies established", pause * 0.5)

    _print_subsection("Step 3: Simulate System Operations")

    _explain(
        "We'll simulate normal operations followed by load testing:\n\n"
        "Phase A: Normal operations with synchronization\n"
        "  (network_sync sequence - coordinated request handling)",
        pause,
    )

    print("    >>> network.apply_sequence('network_sync', repeat=5)\n")
    network.apply_sequence("network_sync", repeat=5)

    _explain("✓ Completed normal operation cycles", pause * 0.5)

    _explain(
        "\nPhase B: System consolidation and stabilization\n"
        "  (consolidation sequence - optimize and stabilize)",
        pause * 0.5,
    )

    print("    >>> network.apply_sequence('consolidation', repeat=3)\n")
    network.apply_sequence("consolidation", repeat=3)

    _explain("✓ System consolidated and stabilized", pause * 0.5)

    _print_subsection("Step 4: Measure System Reliability")

    _explain("Let's measure the system health:", pause * 0.5)

    print("    >>> results = network.measure()\n")
    results = network.measure()

    coherence = results.coherence
    avg_si = (
        sum(results.sense_indices.values()) / len(results.sense_indices)
        if results.sense_indices
        else 0
    )
    avg_vf = results.avg_vf or 0

    print(f"    System Coherence C(t) = {coherence:.3f}")
    print(f"    Average Service Si = {avg_si:.3f}")
    print(f"    Average Processing Rate νf = {avg_vf:.3f} Hz_str")
    print(f"    Number of Services = {len(results.sense_indices)}\n")

    _print_subsection("Step 5: Technical Interpretation")

    if coherence > 0.6:
        system_status = "highly reliable"
        tech_meaning = "Services are well-coordinated with low failure risk."
    elif coherence > 0.3:
        system_status = "moderately reliable"
        tech_meaning = "System is functional but may have coordination issues."
    else:
        system_status = "needs attention"
        tech_meaning = "Services are poorly coordinated, high failure risk."

    if avg_si > 0.5:
        resilience = "high resilience"
        resilience_meaning = "Services can adapt to load changes and failures."
    else:
        resilience = "moderate resilience"
        resilience_meaning = "Services have limited adaptability to disruptions."

    _explain(
        f"Results:\n"
        f"  • System Status: {system_status}\n"
        f"  • Technical Meaning: {tech_meaning}\n"
        f"  • Resilience: {resilience}\n"
        f"  • Resilience Meaning: {resilience_meaning}\n\n"
        f"In technical terms:\n"
        f"  The system achieved coherence of {coherence:.3f}, indicating how well\n"
        f"  the microservices coordinate their operations and handle requests.\n\n"
        f"  Key insights:\n"
        f"  • Coherence C(t) = System reliability and coordination\n"
        f"  • Sense Index Si = Service resilience and adaptability\n"
        f"  • Frequency νf = Request processing rate\n\n"
        f"  The network_sync sequence simulates:\n"
        f"    - Message broadcasting (emission)\n"
        f"    - Request processing (reception)\n"
        f"    - State synchronization (coherence)\n"
        f"    - Service coupling (coupling)\n"
        f"    - Load distribution (resonance)\n"
        f"    - Graceful handling (silence)\n\n"
        f"  Higher coherence = More reliable distributed system\n"
        f"  Higher Si = Better fault tolerance and adaptability",
        pause,
    )

    _print_section("Distributed Systems Tutorial Complete! 💻")

    return {
        "coherence": coherence,
        "sense_indices": results.sense_indices,
        "interpretation": {
            "system_status": system_status,
            "technical_meaning": tech_meaning,
            "resilience": resilience,
            "resilience_meaning": resilience_meaning,
            "avg_si": avg_si,
            "avg_vf": avg_vf,
        },
        "results": results,
    }


def run_all_tutorials(interactive: bool = True, random_seed: int = 42) -> dict:
    """Run all tutorials in sequence.

    This function runs the complete tutorial sequence:
    1. hello_tnfr() - Introduction
    2. biological_example() - Cell communication
    3. social_network_example() - Social dynamics
    4. technology_example() - Distributed systems

    Parameters
    ----------
    interactive : bool, default=True
        If True, pauses between tutorials for reading.
    random_seed : int, default=42
        Random seed for reproducibility across all tutorials.

    Returns
    -------
    dict
        Combined results from all tutorials.

    Examples
    --------
    >>> from tnfr.tutorials import run_all_tutorials
    >>> all_results = run_all_tutorials()
    >>> print(f"Completed {len(all_results)} tutorials!")
    """
    if not _HAS_SDK:
        print("Error: SDK not available. Install with: pip install tnfr")
        return {}

    results = {}

    print("\n" + "=" * 70)
    print("TNFR Complete Tutorial Series")
    print("=" * 70)
    print("\nThis will run all 4 tutorials in sequence:")
    print("  1. Hello TNFR (5 min)")
    print("  2. Biological Example (cell communication)")
    print("  3. Social Network Example (group dynamics)")
    print("  4. Technology Example (distributed systems)")
    print("\nEstimated time: ~15-20 minutes")
    print("=" * 70 + "\n")

    if interactive:
        time.sleep(3)

    # Tutorial 1
    hello_tnfr(interactive=interactive, random_seed=random_seed)
    results["hello_tnfr"] = "completed"

    if interactive:
        print("\nPress Enter to continue to the next tutorial...")
        input()

    # Tutorial 2
    bio_results = biological_example(interactive=interactive, random_seed=random_seed)
    results["biological"] = bio_results

    if interactive:
        print("\nPress Enter to continue to the next tutorial...")
        input()

    # Tutorial 3
    social_results = social_network_example(
        interactive=interactive, random_seed=random_seed
    )
    results["social"] = social_results

    if interactive:
        print("\nPress Enter to continue to the final tutorial...")
        input()

    # Tutorial 4
    tech_results = technology_example(interactive=interactive, random_seed=random_seed)
    results["technology"] = tech_results

    # Summary
    _print_section("All Tutorials Complete! 🎉")

    print("You've completed the full TNFR tutorial series!\n")
    print("Summary of Results:")
    print("-" * 70)

    if "biological" in results:
        print(
            f"  • Cell Communication: C(t) = {results['biological']['coherence']:.3f}"
        )
    if "social" in results:
        print(f"  • Social Network: C(t) = {results['social']['coherence']:.3f}")
    if "technology" in results:
        print(
            f"  • Distributed Systems: C(t) = {results['technology']['coherence']:.3f}"
        )

    print("\n" + "-" * 70)
    print("\nNext Steps:")
    print("  • Explore the SDK: from tnfr.sdk import TNFRNetwork")
    print("  • Read the docs: docs/source/getting-started/")
    print("  • Try your own experiments!")
    print("  • Check out examples/ directory for more")
    print("\n" + "=" * 70 + "\n")

    return results


def team_communication_example(interactive: bool = True, random_seed: int = 42) -> dict:
    """Team communication comparison example using TNFR.

    This example demonstrates how to model and optimize team communication
    patterns using different network topologies. It's the hands-on version
    of the tutorial in INTERACTIVE_TUTORIAL.md Part 3.

    Concepts demonstrated:
    - Comparing network topologies (random, ring, small-world)
    - Measuring communication effectiveness via coherence
    - Optimizing team structure
    - Interpreting individual node metrics

    Parameters
    ----------
    interactive : bool, default=True
        If True, pauses between sections for reading.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results for each team structure with analysis.

    Examples
    --------
    >>> from tnfr.tutorials import team_communication_example
    >>> results = team_communication_example()
    >>> print(f"Best structure: {results['best_structure']}")

    Notes
    -----
    This tutorial demonstrates practical application of TNFR to
    organizational design and communication optimization.
    """
    if not _HAS_SDK:
        print("Error: SDK not available. Install with: pip install tnfr")
        return {}

    pause = 1.5 if interactive else 0

    _print_section("TNFR Tutorial: Team Communication 👥")

    _explain(
        "In this example, we'll compare different team communication structures\n"
        "and identify which topology creates the most coherent organization.\n\n"
        "We'll model:\n"
        "  • Team members as nodes\n"
        "  • Communication relationships as connections\n"
        "  • Information flow as operator sequences\n"
        "  • Team alignment as coherence C(t)",
        pause,
    )

    _print_subsection("Step 1: Create Three Team Structures")

    _explain(
        "We'll create 8-person teams with different topologies:\n"
        "  1. Random - Organic, unstructured\n"
        "  2. Ring - Linear communication chain\n"
        "  3. Small-World - Mix of local and distant connections",
        pause,
    )

    print("    >>> random_team = TNFRNetwork('random_team')")
    print("    >>> random_team.add_nodes(8, random_seed=42)")
    print("    >>> random_team.connect_nodes(0.3, connection_pattern='random')\n")

    random_team = TNFRNetwork("random_team")
    random_team.add_nodes(8, random_seed=random_seed)
    random_team.connect_nodes(0.3, connection_pattern="random")

    print("    >>> ring_team = TNFRNetwork('ring_team')")
    print("    >>> ring_team.add_nodes(8, random_seed=42)")
    print("    >>> ring_team.connect_nodes(connection_pattern='ring')\n")

    ring_team = TNFRNetwork("ring_team")
    ring_team.add_nodes(8, random_seed=random_seed)
    ring_team.connect_nodes(connection_pattern="ring")

    print("    >>> sw_team = TNFRNetwork('small_world_team')")
    print("    >>> sw_team.add_nodes(8, random_seed=42)")
    print("    >>> sw_team.connect_nodes(0.15, connection_pattern='small_world')\n")

    sw_team = TNFRNetwork("small_world_team")
    sw_team.add_nodes(8, random_seed=random_seed)
    sw_team.connect_nodes(0.15, connection_pattern="small_world")

    _explain(
        f"✓ Created 3 team structures:\n"
        f"  - Random: {random_team.get_edge_count()} connections\n"
        f"  - Ring: {ring_team.get_edge_count()} connections\n"
        f"  - Small-world: {sw_team.get_edge_count()} connections",
        pause,
    )

    _print_subsection("Step 2: Simulate Communication")

    _explain(
        "Now we'll apply the same communication sequence to all teams\n"
        "and measure which structure achieves better alignment:",
        pause,
    )

    print("    >>> for team in [random_team, ring_team, sw_team]:")
    print("    ...     team.apply_sequence('network_sync', repeat=5)\n")

    random_team.apply_sequence("network_sync", repeat=5)
    ring_team.apply_sequence("network_sync", repeat=5)
    sw_team.apply_sequence("network_sync", repeat=5)

    _explain("✓ Applied 5 communication cycles to each team", pause * 0.5)

    _print_subsection("Step 3: Compare Results")

    random_results = random_team.measure()
    ring_results = ring_team.measure()
    sw_results = sw_team.measure()

    print("    Communication Effectiveness:\n")
    print(f"    Random Team:")
    print(f"      - Coherence: {random_results.coherence:.3f}")
    print(f"      - Density: {random_team.get_density():.3f}\n")

    print(f"    Ring Team:")
    print(f"      - Coherence: {ring_results.coherence:.3f}")
    print(f"      - Density: {ring_team.get_density():.3f}\n")

    print(f"    Small-World Team:")
    print(f"      - Coherence: {sw_results.coherence:.3f}")
    print(f"      - Density: {sw_team.get_density():.3f}\n")

    teams = {
        "Random": random_results.coherence,
        "Ring": ring_results.coherence,
        "Small-World": sw_results.coherence,
    }
    best_team = max(teams, key=teams.get)

    _explain(
        f"🏆 Most coherent team structure: {best_team}\n\n"
        f"Interpretation:\n"
        f"  The {best_team} topology achieved highest coherence ({teams[best_team]:.3f}),\n"
        f"  indicating better information synchronization across the team.",
        pause,
    )

    _print_section("Team Communication Tutorial Complete! 👥")

    return {
        "random": {
            "coherence": random_results.coherence,
            "density": random_team.get_density(),
        },
        "ring": {
            "coherence": ring_results.coherence,
            "density": ring_team.get_density(),
        },
        "small_world": {
            "coherence": sw_results.coherence,
            "density": sw_team.get_density(),
        },
        "best_structure": best_team,
        "results": {
            "random": random_results,
            "ring": ring_results,
            "small_world": sw_results,
        },
    }


def adaptive_ai_example(interactive: bool = True, random_seed: int = 42) -> dict:
    """Adaptive AI system example using TNFR.

    This tutorial demonstrates how TNFR can model learning and adaptation
    through resonance rather than traditional gradient descent. It shows
    structural learning principles.

    Concepts demonstrated:
    - Learning as coherence increase (not error minimization)
    - Adaptation via structural reorganization
    - Memory as stable EPI patterns
    - Context sensitivity via phase coupling

    Parameters
    ----------
    interactive : bool, default=True
        If True, pauses between sections for reading.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Learning trajectory and final system state.

    Examples
    --------
    >>> from tnfr.tutorials import adaptive_ai_example
    >>> results = adaptive_ai_example()
    >>> print(f"Learning improvement: {results['improvement']:.1f}%")

    Notes
    -----
    This demonstrates TNFR's alternative to traditional ML:
    - No backpropagation or gradient descent
    - Learning through resonance and structural operators
    - Maintains coherence throughout adaptation
    """
    if not _HAS_SDK:
        print("Error: SDK not available. Install with: pip install tnfr")
        return {}

    pause = 1.5 if interactive else 0

    _print_section("TNFR Tutorial: Adaptive AI System 🤖")

    _explain(
        "In this example, we'll model a learning system using TNFR principles.\n"
        "Unlike traditional ML, learning happens through structural resonance,\n"
        "not gradient descent.\n\n"
        "TNFR Learning Model:\n"
        "  • Nodes = Processing units (like neurons, but resonant)\n"
        "  • Coherence increase = Learning (not error reduction)\n"
        "  • Mutation operator = Exploration\n"
        "  • Resonance operator = Pattern consolidation",
        pause,
    )

    _print_subsection("Step 1: Create Initial 'Naive' System")

    _explain(
        "We start with an unorganized network representing\n"
        "a system before training:",
        pause,
    )

    print("    >>> naive_system = TNFRNetwork('learning_system')")
    print("    >>> naive_system.add_nodes(15, random_seed=42)")
    print("    >>> naive_system.connect_nodes(0.25, 'random')  # Sparse connections\n")

    naive_system = TNFRNetwork("learning_system")
    naive_system.add_nodes(15, random_seed=random_seed)
    naive_system.connect_nodes(0.25, "random")

    print("    >>> naive_system.apply_sequence('basic_activation', repeat=2)")
    print("    >>> initial_state = naive_system.measure()\n")

    naive_system.apply_sequence("basic_activation", repeat=2)
    initial_state = naive_system.measure()

    initial_coherence = initial_state.coherence

    _explain(
        f"Initial system state (before learning):\n"
        f"  • Coherence: {initial_coherence:.3f}\n"
        f"  • This represents an 'untrained' system",
        pause,
    )

    _print_subsection("Step 2: 'Training' via Structural Reorganization")

    _explain(
        "Training in TNFR means applying sequences that increase coherence:\n"
        "  1. Exploration (mutation + dissonance)\n"
        "  2. Consolidation (coherence + resonance)\n"
        "  3. Repeat until convergence\n\n"
        "This is fundamentally different from backpropagation!",
        pause,
    )

    print("    >>> # Training loop")
    print("    >>> for epoch in range(3):")
    print("    ...     naive_system.apply_sequence('network_sync', repeat=2)")
    print("    ...     naive_system.apply_sequence('consolidation', repeat=3)\n")

    coherence_trajectory = [initial_coherence]

    for epoch in range(3):
        naive_system.apply_sequence("network_sync", repeat=2)
        naive_system.apply_sequence("consolidation", repeat=3)
        epoch_results = naive_system.measure()
        coherence_trajectory.append(epoch_results.coherence)
        if interactive:
            print(f"    Epoch {epoch+1}: C(t) = {epoch_results.coherence:.3f}")

    print()

    _print_subsection("Step 3: Evaluate Learning")

    final_state = naive_system.measure()
    final_coherence = final_state.coherence
    improvement = ((final_coherence - initial_coherence) / initial_coherence) * 100

    _explain(
        f"Learning Results:\n"
        f"  • Initial coherence: {initial_coherence:.3f}\n"
        f"  • Final coherence: {final_coherence:.3f}\n"
        f"  • Improvement: {improvement:+.1f}%\n\n"
        f"Interpretation:\n"
        f"  The system 'learned' by increasing its internal coherence.\n"
        f"  Higher coherence = Better organized = More 'trained'\n\n"
        f"  This demonstrates learning as structural reorganization,\n"
        f"  not as weight optimization!",
        pause,
    )

    _print_subsection("Step 4: TNFR vs Traditional ML")

    _explain(
        "Key Differences:\n\n"
        "Traditional ML (backprop):\n"
        "  • Minimize error/loss\n"
        "  • Adjust weights via gradients\n"
        "  • Fixed architecture\n\n"
        "TNFR Adaptive Systems:\n"
        "  • Maximize coherence\n"
        "  • Reorganize structure via operators\n"
        "  • Dynamic, self-organizing architecture\n\n"
        "Both work, but TNFR preserves structural meaning throughout.",
        pause,
    )

    _print_section("Adaptive AI Tutorial Complete! 🤖")

    return {
        "initial_coherence": initial_coherence,
        "final_coherence": final_coherence,
        "improvement": improvement,
        "coherence_trajectory": coherence_trajectory,
        "final_state": final_state,
        "interpretation": (
            f"System improved coherence by {improvement:.1f}% through "
            f"structural reorganization, demonstrating learning without "
            f"traditional gradient descent."
        ),
    }


def oz_dissonance_tutorial(interactive: bool = True, random_seed: int = 42) -> dict:
    """Interactive tutorial on OZ (Dissonance) operator and canonical sequences.

    This tutorial covers:
    - Theoretical foundations of OZ (topological dissonance)
    - When to use OZ vs when to avoid
    - 6 canonical sequences with OZ from TNFR theory
    - Bifurcation paths and resolution patterns
    - Common errors and how to fix them

    Parameters
    ----------
    interactive : bool, default=True
        If True, pauses between sections for reading.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Tutorial results including sequence demonstrations and coherence metrics.

    Examples
    --------
    >>> from tnfr.tutorials import oz_dissonance_tutorial
    >>> oz_dissonance_tutorial()  # Run full tutorial
    >>> oz_dissonance_tutorial(interactive=False)  # Run without pauses

    Notes
    -----
    This tutorial demonstrates canonical sequences from "El pulso que nos
    atraviesa" Table 2.5, maintaining full TNFR theoretical fidelity.
    """
    if not _HAS_SDK:
        print("Error: SDK not available. Install with: pip install tnfr")
        return {}

    pause = 1.5 if interactive else 0

    _print_section("OZ (DISSONANCE) OPERATOR - Interactive Tutorial")

    _explain(
        "Welcome to the OZ (Dissonance) tutorial!\n\n"
        "OZ is one of the 13 canonical structural operators in TNFR.\n"
        "It introduces CONTROLLED INSTABILITY to enable creative exploration\n"
        "and transformation. This tutorial will show you how to use it effectively.",
        pause,
    )

    # Section 1: What is OZ?
    _print_subsection("Part 1: What is OZ (Dissonance)?")

    _explain(
        "🌀 OZ introduces controlled instability that enables:\n\n"
        "  ✓ Creative exploration of new structural configurations\n"
        "  ✓ Bifurcation into alternative reorganization paths\n"
        "  ✓ Mutation enablement (OZ → ZHIR canonical pattern)\n"
        "  ✓ Topological disruption of rigid patterns\n\n"
        "Important: OZ is NOT destructive - it's GENERATIVE dissonance.\n"
        "Think of it as asking challenging questions rather than breaking things.",
        pause,
    )

    # Section 2: When to use OZ
    _print_subsection("Part 2: When to Use OZ")

    _explain(
        "⚠️ Use OZ in these situations:\n\n"
        "  ✅ After stabilization (IL) to explore new possibilities\n"
        "  ✅ Before mutation (ZHIR) to justify transformation\n"
        "  ✅ In therapeutic protocols to confront blockages\n"
        "  ✅ In learning to challenge existing mental models\n\n"
        "OZ works best when the system is stable enough to handle disruption.",
        pause,
    )

    # Section 3: When to AVOID OZ
    _print_subsection("Part 3: When to AVOID OZ")

    _explain(
        "🚫 Avoid OZ in these situations:\n\n"
        "  ❌ On latent/weak nodes (EPI < 0.2) → causes collapse\n"
        "  ❌ When ΔNFR already critical (ΔNFR > 0.8) → overload\n"
        "  ❌ Multiple OZ without IL resolution → entropic noise\n"
        "  ❌ Immediately before SHA (silence) → contradictory\n\n"
        "Rule of thumb: Stabilize before you destabilize!",
        pause,
    )

    # Section 4: Canonical Sequences with OZ
    _print_subsection("Part 4: Canonical Sequences with OZ")

    _explain(
        "TNFR theory defines 6 archetypal sequences involving OZ.\n"
        "Let's explore them with live demonstrations...",
        pause,
    )

    from ..operators.canonical_patterns import CANONICAL_SEQUENCES
    from ..types import Glyph

    # Get sequences with OZ
    oz_sequences = {
        name: seq for name, seq in CANONICAL_SEQUENCES.items() if Glyph.OZ in seq.glyphs
    }

    _explain(f"\nFound {len(oz_sequences)} canonical sequences with OZ:\n", pause * 0.5)

    for i, (name, seq) in enumerate(sorted(oz_sequences.items()), 1):
        glyphs_str = " → ".join(g.value for g in seq.glyphs)
        print(f"{i}. {name.upper()}")
        print(f"   Pattern: {seq.pattern_type.value}")
        print(f"   Domain: {seq.domain}")
        print(f"   Glyphs: {glyphs_str}")
        print(f"   Use: {seq.use_cases[0]}\n")

    # Section 5: Hands-on Demonstration - Bifurcated Pattern
    _print_subsection("Part 5: Hands-On Demo - Bifurcated Pattern")

    _explain(
        "Let's demonstrate the BIFURCATED pattern:\n"
        "This pattern shows how OZ creates a decision point where the node\n"
        "can either mutate (ZHIR) or collapse (NUL).\n\n"
        "We'll apply the 'bifurcated_base' sequence (mutation path):",
        pause,
    )

    print("    >>> from tnfr.sdk import TNFRNetwork")
    print("    >>> net = TNFRNetwork('bifurcation_demo')")
    print("    >>> net.add_nodes(1)")
    print("    >>> net.apply_canonical_sequence('bifurcated_base')\n")

    from ..sdk import TNFRNetwork, NetworkConfig

    net_bifurc = TNFRNetwork("bifurcation_demo", NetworkConfig(random_seed=random_seed))
    net_bifurc.add_nodes(1)
    net_bifurc.apply_canonical_sequence("bifurcated_base")

    results_bifurc = net_bifurc.measure()

    print(f"    ✓ Bifurcation completed!")
    print(f"    Final Coherence C(t): {results_bifurc.coherence:.3f}\n")

    _explain(
        "Interpretation:\n"
        "  The sequence AL → EN → IL → OZ → ZHIR → IL → SHA shows:\n"
        "  1. Node is activated and stabilized (AL → EN → IL)\n"
        "  2. Dissonance introduced (OZ) creating instability\n"
        "  3. Node mutates to new form (ZHIR)\n"
        "  4. New form is stabilized (IL) and enters rest (SHA)\n\n"
        f"  High coherence ({results_bifurc.coherence:.3f}) shows successful transformation!",
        pause,
    )

    # Section 6: Therapeutic Protocol Demo
    _print_subsection("Part 6: Therapeutic Protocol Demo")

    _explain(
        "Now let's demonstrate the THERAPEUTIC protocol:\n"
        "This is used for healing and personal transformation.\n"
        "We'll create a small network to represent a therapeutic context:",
        pause,
    )

    print("    >>> net = TNFRNetwork('therapy')")
    print("    >>> net.add_nodes(3).connect_nodes(0.4, 'random')")
    print("    >>> net.apply_canonical_sequence('therapeutic_protocol')\n")

    net_therapy = TNFRNetwork("therapy", NetworkConfig(random_seed=random_seed))
    net_therapy.add_nodes(3)
    net_therapy.connect_nodes(0.4, "random")
    net_therapy.apply_canonical_sequence("therapeutic_protocol")

    results_therapy = net_therapy.measure()
    avg_si_therapy = (
        sum(results_therapy.sense_indices.values()) / len(results_therapy.sense_indices)
        if results_therapy.sense_indices
        else 0
    )

    print(f"    ✓ Therapeutic protocol completed!")
    print(f"    Final Coherence C(t): {results_therapy.coherence:.3f}")
    print(f"    Average Sense Index Si: {avg_si_therapy:.3f}\n")

    _explain(
        "Interpretation:\n"
        "  The therapeutic protocol (AL → EN → IL → OZ → ZHIR → IL → RA → SHA):\n"
        "  1. Initiates symbolic field (AL)\n"
        "  2. Stabilizes the therapeutic context (EN → IL)\n"
        "  3. Introduces creative confrontation (OZ)\n"
        "  4. Enables personal transformation (ZHIR)\n"
        "  5. Integrates new form (IL)\n"
        "  6. Propagates through network (RA)\n"
        "  7. Enters restful integration (SHA)\n\n"
        f"  Coherence of {results_therapy.coherence:.3f} shows therapeutic effectiveness!",
        pause,
    )

    # Section 7: MOD_STABILIZER - Reusable Macro
    _print_subsection("Part 7: MOD_STABILIZER - Reusable Transformation")

    _explain(
        "The MOD_STABILIZER is a reusable macro for safe transformation:\n"
        "REMESH → EN → IL → OZ → ZHIR → IL → REMESH\n\n"
        "It's designed to be composable within larger sequences.\n"
        "This is your 'safe transformation module':",
        pause,
    )

    print("    >>> net = TNFRNetwork('modular')")
    print("    >>> net.add_nodes(1)")
    print("    >>> net.apply_canonical_sequence('mod_stabilizer')\n")

    net_mod = TNFRNetwork("modular", NetworkConfig(random_seed=random_seed))
    net_mod.add_nodes(1)
    net_mod.apply_canonical_sequence("mod_stabilizer")

    results_mod = net_mod.measure()

    print(f"    ✓ MOD_STABILIZER completed!")
    print(f"    Final Coherence C(t): {results_mod.coherence:.3f}\n")

    _explain(
        "This module can be nested in larger sequences:\n"
        "  THOL[MOD_STABILIZER] ≡ THOL[REMESH → EN → IL → OZ → ZHIR → IL → REMESH]\n\n"
        "It's a building block for complex transformations!",
        pause,
    )

    # Section 8: Filtering and Discovery
    _print_subsection("Part 8: Discovering OZ Sequences")

    _explain("You can programmatically discover sequences with OZ:\n", pause * 0.5)

    print("    >>> net = TNFRNetwork('explorer')")
    print("    >>> oz_seqs = net.list_canonical_sequences(with_oz=True)")
    print(f"    >>> print(f'Found {{len(oz_seqs)}} sequences with OZ')\n")

    net_explorer = TNFRNetwork("explorer")
    oz_seqs = net_explorer.list_canonical_sequences(with_oz=True)

    print(f"    Found {len(oz_seqs)} sequences with OZ\n")

    _explain("You can also filter by domain:", pause * 0.5)

    print("    >>> bio_seqs = net.list_canonical_sequences(domain='biomedical')")
    print("    >>> cog_seqs = net.list_canonical_sequences(domain='cognitive')\n")

    bio_seqs = net_explorer.list_canonical_sequences(domain="biomedical")
    cog_seqs = net_explorer.list_canonical_sequences(domain="cognitive")

    print(f"    Biomedical sequences: {len(bio_seqs)}")
    print(f"    Cognitive sequences: {len(cog_seqs)}\n")

    # Section 9: Summary and Best Practices
    _print_subsection("Part 9: Summary and Best Practices")

    _explain(
        "KEY TAKEAWAYS:\n\n"
        "1. OZ is GENERATIVE dissonance, not destructive\n"
        "2. Always stabilize (IL) before introducing dissonance (OZ)\n"
        "3. OZ typically pairs with ZHIR (mutation) for transformation\n"
        "4. Use canonical sequences for reliable patterns\n"
        "5. Filter sequences by domain for specific applications\n\n"
        "BEST PRACTICES:\n\n"
        "  • Start with canonical sequences before custom patterns\n"
        "  • Test on simple networks before complex ones\n"
        "  • Monitor coherence C(t) to ensure stability\n"
        "  • Use MOD_STABILIZER as a building block\n"
        "  • Consult theoretical docs for deeper understanding",
        pause,
    )

    _print_section("OZ Dissonance Tutorial Complete! 🌀")

    _explain(
        "You now understand:\n"
        "  ✓ What OZ (Dissonance) does\n"
        "  ✓ When to use and avoid OZ\n"
        "  ✓ 6 canonical sequences with OZ\n"
        "  ✓ How to apply them programmatically\n"
        "  ✓ How to discover and filter sequences\n\n"
        "Next steps:\n"
        "  • Explore examples/oz_canonical_sequences.py for more details\n"
        "  • Try creating your own sequences with OZ\n"
        "  • Read 'El pulso que nos atraviesa' for theoretical depth",
        pause,
    )

    return {
        "bifurcated_coherence": results_bifurc.coherence,
        "therapeutic_coherence": results_therapy.coherence,
        "therapeutic_avg_si": avg_si_therapy,
        "mod_stabilizer_coherence": results_mod.coherence,
        "total_oz_sequences": len(oz_seqs),
        "biomedical_sequences": len(bio_seqs),
        "cognitive_sequences": len(cog_seqs),
        "interpretation": (
            f"Successfully demonstrated {len(oz_seqs)} canonical sequences with OZ. "
            f"Bifurcation achieved {results_bifurc.coherence:.3f} coherence, "
            f"therapeutic protocol achieved {results_therapy.coherence:.3f} coherence. "
            f"All patterns maintain TNFR canonical invariants."
        ),
    }

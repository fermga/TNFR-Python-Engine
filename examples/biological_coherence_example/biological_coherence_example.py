#!/usr/bin/env python3
"""
Title: Biological Coherence - Cell Communication Modeling

Problem: Cells in tissues coordinate their responses through signaling molecules
(hormones, cytokines, neurotransmitters) and direct contact (gap junctions).
How do individual cells organize into coherent tissues that respond adaptively
to environmental signals?

TNFR Approach: Model cells as NFR nodes where:
- EPI represents cellular state (metabolic profile, gene expression)
- νf (Hz_str) is the rate of cellular response/adaptation
- Phase synchronization represents coordinated cellular behavior
- Emission/Reception model chemical signaling
- Coupling models gap junctions and contact-dependent signaling
- Coherence represents tissue organization

Key Operators:
- Emission (AL): Cell secretes signaling molecules
- Reception (EN): Cell detects and responds to signals
- Coupling (UM): Gap junction formation, direct cell-cell contact
- Coherence (IL): Tissue organization and homeostasis
- Resonance (RA): Signal amplification, coordinated response

Relevant Metrics:
- C(t): Tissue coherence (how well cells coordinate)
- Si: Individual cell stability (resistance to perturbation)
- Phase coherence: Synchronization of cellular rhythms
- ΔNFR: Cellular stress/adaptation pressure

Expected Behavior:
- Initially scattered cells establish communication
- Phase synchronization emerges through signaling
- Tissue coherence increases as cells coordinate
- Final state shows stable, organized tissue structure

Run:
    python docs/source/examples/biological_coherence_example.py
"""

from tnfr import create_nfr, run_sequence
from tnfr.dynamics import run
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si
from tnfr.structural import (
    Coupling,
    Coherence,
    Emission,
    Reception,
    Resonance,
    Silence,
)
from tnfr.trace import register_trace
from tnfr.constants import inject_defaults


def run_example() -> None:
    """Model tissue formation through cellular communication and coordination."""
    
    print("=" * 70)
    print("TNFR Biological Coherence: Cell Communication Modeling")
    print("=" * 70)
    print()
    
    # 1. PROBLEM SETUP: Creating a tissue with 6 cells
    # ---------------------------------------------------
    # We model a small tissue patch with different cell types:
    # - 2 Signaling cells (high emission capacity)
    # - 2 Responding cells (high reception sensitivity)
    # - 2 Coordinator cells (balance emission/reception)
    
    print("Phase 1: Initializing cellular network...")
    print("Creating 6 cells with different communication profiles")
    print()
    
    # Signaling cell 1: High νf (fast response), initiates communication
    G, _ = create_nfr(
        "SignalCell_1",
        epi=0.35,  # Active metabolic state
        vf=1.2,    # Fast adaptation (1.2 Hz_str)
        theta=0.0  # Starting phase
    )
    
    # Signaling cell 2: Similar profile, slightly different phase
    create_nfr(
        "SignalCell_2",
        epi=0.33,
        vf=1.15,
        theta=0.3,  # Slightly out of phase initially
        graph=G
    )
    
    # Responding cells: Moderate νf, receptive to signals
    create_nfr(
        "ResponseCell_1",
        epi=0.20,  # Lower baseline activity
        vf=0.9,    # Moderate adaptation rate
        theta=-0.4,
        graph=G
    )
    
    create_nfr(
        "ResponseCell_2",
        epi=0.22,
        vf=0.95,
        theta=0.6,
        graph=G
    )
    
    # Coordinator cells: Balanced profile, help integrate signals
    create_nfr(
        "CoordCell_1",
        epi=0.28,
        vf=1.0,
        theta=-0.2,
        graph=G
    )
    
    create_nfr(
        "CoordCell_2",
        epi=0.26,
        vf=1.05,
        theta=0.5,
        graph=G
    )
    
    # Store biological metadata
    cell_types = {
        "SignalCell_1": "Hormone-secreting endocrine cell",
        "SignalCell_2": "Hormone-secreting endocrine cell",
        "ResponseCell_1": "Target cell with hormone receptors",
        "ResponseCell_2": "Target cell with hormone receptors",
        "CoordCell_1": "Intermediate cell (signal + response)",
        "CoordCell_2": "Intermediate cell (signal + response)",
    }
    
    for node, cell_type in cell_types.items():
        G.nodes[node]["cell_type"] = cell_type
    
    # Inject required defaults for graph parameters
    inject_defaults(G)
    
    # Measure initial state
    C_initial, dnfr_initial, _ = compute_coherence(G, return_means=True)
    print(f"Initial tissue state:")
    print(f"  C(t) = {C_initial:.3f} (tissue coherence - should be low)")
    print(f"  Mean ΔNFR = {dnfr_initial:.3f} (cellular stress)")
    print()
    
    # 2. TNFR MODELING: Define cellular behaviors
    # ---------------------------------------------
    # Each cell type follows a different operator sequence based on its role
    
    print("Phase 2: Establishing cellular communication protocols...")
    print()
    
    # Signaling cells: Strong emission, establish connections
    signaling_protocol = [
        Emission(),      # Secrete signaling molecules
        Reception(),     # Also receive feedback from neighbors
        Coherence(),     # Stabilize secretion pattern
        Resonance(),     # Amplify coordinated signals
        Coupling(),      # Form gap junctions with neighbors
        Silence(),       # Brief pause to allow signal propagation
    ]
    
    # Response cells: Primarily receptive, integrate signals
    response_protocol = [
        Emission(),      # Send signals (must start with emission)
        Reception(),     # Detect signaling molecules from neighbors
        Coherence(),     # Stabilize response pattern
        Resonance(),     # Participate in coordinated response
        Coupling(),      # Form connections with signalers
        Silence(),
    ]
    
    # Coordinator cells: Balance emission and reception
    coordinator_protocol = [
        Emission(),      # Relay signals
        Reception(),     # Integrate multiple inputs
        Coherence(),     # Maintain balanced state
        Resonance(),     # Facilitate tissue-wide coordination
        Coupling(),      # Connect multiple cell types
        Coherence(),     # Re-stabilize after coordination
        Silence(),
    ]
    
    # 3. OPERATOR APPLICATION: Execute cellular behaviors
    # ----------------------------------------------------
    
    print("Applying signaling protocol to SignalCell_1 and SignalCell_2...")
    run_sequence(G, "SignalCell_1", signaling_protocol)
    run_sequence(G, "SignalCell_2", signaling_protocol)
    
    print("Applying response protocol to ResponseCell_1 and ResponseCell_2...")
    run_sequence(G, "ResponseCell_1", response_protocol)
    run_sequence(G, "ResponseCell_2", response_protocol)
    
    print("Applying coordinator protocol to CoordCell_1 and CoordCell_2...")
    run_sequence(G, "CoordCell_1", coordinator_protocol)
    run_sequence(G, "CoordCell_2", coordinator_protocol)
    print()
    
    # 4. SIMULATION: Run tissue dynamics
    # ------------------------------------
    
    print("Phase 3: Simulating tissue dynamics over time...")
    print()
    
    # Register metrics collection
    register_metrics_callbacks(G)
    register_trace(G)
    
    # Run dynamics: cells communicate and synchronize
    # 10 steps = ~10 cellular response cycles
    run(G, steps=10, dt=0.1)  # dt=0.1 represents ~100ms time resolution
    
    # 5. RESULTS INTERPRETATION
    # --------------------------
    
    print("=" * 70)
    print("RESULTS: Tissue Organization Analysis")
    print("=" * 70)
    print()
    
    # Compute final metrics
    C_final, dnfr_final, depi_final = compute_coherence(G, return_means=True)
    Si_values = compute_Si(G)
    
    print("Tissue-Level Metrics:")
    print(f"  C(t) = {C_final:.3f} (final tissue coherence)")
    print(f"  ΔC = {C_final - C_initial:+.3f} (change from initial state)")
    print(f"  Mean ΔNFR = {dnfr_final:.3f} (residual cellular stress)")
    print(f"  Mean ∂EPI/∂t = {depi_final:.3f} (rate of structural change)")
    print()
    
    print("Cellular Stability (Sense Index):")
    if isinstance(Si_values, dict):
        for cell_name, si_value in sorted(Si_values.items()):
            cell_type_short = cell_types[cell_name].split()[0]
            print(f"  {cell_name:20s} Si = {si_value:.3f}  ({cell_type_short})")
    else:
        # Handle array return type
        for idx, cell_name in enumerate(sorted(G.nodes())):
            si_value = float(Si_values[idx]) if hasattr(Si_values, '__getitem__') else 0.0
            cell_type_short = cell_types[cell_name].split()[0]
            print(f"  {cell_name:20s} Si = {si_value:.3f}  ({cell_type_short})")
    print()
    
    # Biological interpretation
    print("=" * 70)
    print("BIOLOGICAL INTERPRETATION")
    print("=" * 70)
    print()
    
    if C_final > 0.6:
        coherence_status = "HIGH - Tissue is well-organized"
    elif C_final > 0.4:
        coherence_status = "MODERATE - Tissue is forming but not fully organized"
    else:
        coherence_status = "LOW - Cells remain largely uncoordinated"
    
    print(f"1. Tissue Coherence: {coherence_status}")
    print(f"   Initial C(t) = {C_initial:.3f} → Final C(t) = {C_final:.3f}")
    print()
    
    if isinstance(Si_values, dict):
        avg_si = sum(Si_values.values()) / len(Si_values)
    else:
        avg_si = float(Si_values.mean()) if hasattr(Si_values, 'mean') else 0.0
    
    if avg_si > 0.7:
        stability_status = "STABLE - Cells resist perturbation"
    elif avg_si > 0.4:
        stability_status = "MODERATE - Some vulnerability to stress"
    else:
        stability_status = "UNSTABLE - Cells easily perturbed"
    
    print(f"2. Cellular Stability: {stability_status}")
    print(f"   Average Si = {avg_si:.3f}")
    print()
    
    if C_final > C_initial:
        print("3. Outcome: ✓ SUCCESSFUL TISSUE FORMATION")
        print("   - Cells established communication channels")
        print("   - Phase synchronization emerged")
        print("   - Coordinated tissue behavior achieved")
    else:
        print("3. Outcome: ⚠ LIMITED TISSUE ORGANIZATION")
        print("   - Communication was established but weak")
        print("   - Further coupling may be needed")
    print()
    
    print("=" * 70)
    print("Key TNFR Insights:")
    print("=" * 70)
    print("• Cells = NFR nodes with EPI (state), νf (adaptation rate), θ (phase)")
    print("• Signaling = Emission operator (AL) propagates structural information")
    print("• Reception = Integration of external structural patterns")
    print("• Gap junctions = Coupling operator (UM) enables direct synchrony")
    print("• Tissue coherence = Emergent property from cellular phase alignment")
    print("• C(t) metric = Quantitative measure of tissue organization")
    print("=" * 70)


if __name__ == "__main__":
    run_example()

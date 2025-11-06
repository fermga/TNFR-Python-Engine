#!/usr/bin/env python3
"""
Title: Adaptive AI System - Self-Organizing Intelligence

Problem: Traditional AI systems learn by minimizing error gradients through
backpropagation. Can we model learning as structural reorganization instead?
How do intelligent agents self-organize to solve tasks?

TNFR Approach: Model AI agents as NFR nodes where:
- EPI represents agent's knowledge/skill structure
- νf (Hz_str) is learning rate (structural reorganization speed)
- Phase represents coordination with environment/task
- Learning = Structural reorganization through ΔNFR, not gradient descent
- Self-organization creates emergent problem-solving strategies
- Coherence measures solution stability

Key Operators:
- SelfOrganization (THOL): Spontaneous strategy formation
- Reception (EN): Gather environmental feedback
- Emission (AL): Execute actions/strategies
- Coherence (IL): Consolidate learned patterns
- Expansion (VAL): Increase solution complexity
- Mutation (ZHIR): Explore alternative strategies

Relevant Metrics:
- C(t): Solution coherence (strategy stability)
- Si: Learning stability (resistance to catastrophic forgetting)
- ΔNFR: Learning pressure (environment mismatch)
- ∂EPI/∂t: Rate of knowledge acquisition

Expected Behavior:
- Initially random/unstructured agents (low EPI)
- Self-organization creates problem-solving structures
- Agents adapt through structural reorganization
- Coherence increases as solutions stabilize
- Final state shows organized, capable agents

Run:
    python docs/source/examples/adaptive_ai_system.py
"""

from tnfr import create_nfr, run_sequence
from tnfr.dynamics import run
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si
from tnfr.structural import (
    Coherence,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    SelfOrganization,
    Silence,
)
from tnfr.trace import register_trace
from tnfr.constants import inject_defaults


def run_example() -> None:
    """Model AI learning as structural reorganization, not gradient descent."""
    
    print("=" * 70)
    print("TNFR Adaptive AI: Self-Organizing Intelligence")
    print("=" * 70)
    print()
    
    # 1. PROBLEM SETUP: Creating a multi-agent AI system
    # ---------------------------------------------------
    # Scenario: 4 AI agents learning to solve a coordination task
    # Task requires: perception, planning, execution, adaptation
    
    print("Phase 1: Initializing naive AI agents...")
    print("Creating 4 agents with minimal initial structure")
    print()
    
    # Perception Agent: Processes environmental input
    G, _ = create_nfr(
        "PerceptionAgent",
        epi=0.12,    # Low initial structure (untrained)
        vf=1.4,      # High learning rate (rapid reorganization)
        theta=0.0    # Starting phase
    )
    
    # Planning Agent: Develops strategies
    create_nfr(
        "PlanningAgent",
        epi=0.10,    # Minimal structure
        vf=1.3,      # High learning rate
        theta=0.5,
        graph=G
    )
    
    # Execution Agent: Implements actions
    create_nfr(
        "ExecutionAgent",
        epi=0.15,    # Slightly more structure (basic actions)
        vf=1.2,
        theta=-0.3,
        graph=G
    )
    
    # Adaptation Agent: Meta-learning, adjusts strategies
    create_nfr(
        "AdaptationAgent",
        epi=0.08,    # Very low initial structure
        vf=1.5,      # Highest learning rate
        theta=0.8,
        graph=G
    )
    
    # Store agent metadata
    agent_roles = {
        "PerceptionAgent": "Sensory processing & pattern recognition",
        "PlanningAgent": "Strategy formation & decision making",
        "ExecutionAgent": "Action execution & motor control",
        "AdaptationAgent": "Meta-learning & strategy adaptation",
    }
    
    for agent, role in agent_roles.items():
        G.nodes[agent]["role"] = role
    
    # Inject required defaults for graph parameters
    inject_defaults(G)
    
    # Measure initial state (before learning)
    C_initial, dnfr_initial, depi_initial = compute_coherence(G, return_means=True)
    Si_initial = compute_Si(G)
    
    print("Initial system state (BEFORE learning):")
    print(f"  C(t) = {C_initial:.3f} (solution coherence - expect low)")
    print(f"  Mean ΔNFR = {dnfr_initial:.3f} (learning pressure)")
    print(f"  Mean ∂EPI/∂t = {depi_initial:.3f} (knowledge acquisition rate)")
    print()
    
    # Handle Si_initial being either dict or array
    if isinstance(Si_initial, dict):
        avg_si_initial = sum(Si_initial.values()) / len(Si_initial)
    else:
        avg_si_initial = float(Si_initial.mean()) if hasattr(Si_initial, 'mean') else 0.0
    print(f"  Average Si = {avg_si_initial:.3f} (learning stability)")
    print()
    
    # 2. TNFR MODELING: Learning as structural reorganization
    # --------------------------------------------------------
    
    print("Phase 2: Defining learning protocols (NOT gradient descent)...")
    print()
    
    # Perception learning: Pattern recognition through self-organization
    perception_learning = [
        Emission(),          # Must start with emission
        Reception(),         # Gather sensory input
        Coherence(),         # Stabilize
        Expansion(),         # Add representational capacity
        Coherence(),         # Consolidate learning
        Silence(),
    ]
    
    # Planning learning: Strategy formation through exploration
    planning_learning = [
        Emission(),          # Must start with emission
        Reception(),         # Receive perception inputs
        Coherence(),         # Stabilize before creating pressure
        Dissonance(),        # Create exploration pressure
        Mutation(),          # Explore strategy variants
        Coherence(),         # Stabilize successful strategies
        Expansion(),         # Increase strategy complexity
        Silence(),
    ]
    
    # Execution learning: Action refinement through practice
    execution_learning = [
        Emission(),          # Execute actions (must start with emission)
        Reception(),         # Receive planning commands
        Coherence(),         # Stabilize
        Expansion(),         # Add action repertoire
        Coherence(),         # Stabilize successful actions
        Silence(),
    ]
    
    # Adaptation learning: Meta-learning, adjust learning itself
    adaptation_learning = [
        Emission(),          # Must start with emission
        Reception(),         # Monitor all agents
        Coherence(),         # Stabilize before creating adaptive pressure
        Dissonance(),        # Create adaptive pressure
        Mutation(),          # Explore learning algorithms
        Coherence(),         # Stabilize
        Expansion(),         # Increase adaptation capacity
        Silence(),
    ]
    
    # 3. OPERATOR APPLICATION: Execute learning
    # ------------------------------------------
    
    print("Phase 3: Training agents through structural reorganization...")
    print()
    
    print("Training PerceptionAgent (pattern recognition)...")
    run_sequence(G, "PerceptionAgent", perception_learning)
    
    print("Training PlanningAgent (strategy formation)...")
    run_sequence(G, "PlanningAgent", planning_learning)
    
    print("Training ExecutionAgent (action refinement)...")
    run_sequence(G, "ExecutionAgent", execution_learning)
    
    print("Training AdaptationAgent (meta-learning)...")
    run_sequence(G, "AdaptationAgent", adaptation_learning)
    print()
    
    # 4. SIMULATION: Run learning dynamics
    # -------------------------------------
    
    print("Phase 4: Simulating learning dynamics...")
    print("(Multiple learning cycles with environmental feedback)")
    print()
    
    register_metrics_callbacks(G)
    register_trace(G)
    
    # Run for 15 time steps = ~15 learning iterations
    run(G, steps=15, dt=0.1)
    
    # 5. RESULTS INTERPRETATION
    # --------------------------
    
    print("=" * 70)
    print("RESULTS: Learning Outcome Analysis")
    print("=" * 70)
    print()
    
    # Compute final metrics (after learning)
    C_final, dnfr_final, depi_final = compute_coherence(G, return_means=True)
    Si_final = compute_Si(G)
    
    print("System-Level Metrics (AFTER learning):")
    print(f"  C(t): {C_initial:.3f} → {C_final:.3f} (ΔC = {C_final - C_initial:+.3f})")
    print(f"  Mean ΔNFR: {dnfr_initial:.3f} → {dnfr_final:.3f}")
    print(f"  Mean ∂EPI/∂t: {depi_initial:.3f} → {depi_final:.3f}")
    print()
    
    print("Per-Agent Learning Outcomes:")
    for agent in sorted(G.nodes()):
        if isinstance(Si_initial, dict):
            si_before = Si_initial.get(agent, 0.0)
            si_after = Si_final.get(agent, 0.0) if isinstance(Si_final, dict) else 0.0
        else:
            si_before = 0.0
            si_after = Si_final.get(agent, 0.0) if isinstance(Si_final, dict) else 0.0
        si_change = si_after - si_before
        role_short = agent_roles[agent].split()[0]
        
        print(f"  {agent:20s}")
        print(f"    Si: {si_before:.3f} → {si_after:.3f} ({si_change:+.3f})")
        print(f"    Role: {role_short}")
    print()
    
    # AI/ML interpretation
    print("=" * 70)
    print("AI/ML INTERPRETATION")
    print("=" * 70)
    print()
    
    # Learning success analysis
    if C_final > 0.6:
        learning_outcome = "SUCCESSFUL LEARNING"
        explanation = "Agents developed coherent, stable solutions"
    elif C_final > 0.4:
        learning_outcome = "MODERATE LEARNING"
        explanation = "Some structure formed, but incomplete"
    elif C_final > C_initial:
        learning_outcome = "EARLY LEARNING"
        explanation = "Initial progress, needs more training"
    else:
        learning_outcome = "LEARNING FAILURE"
        explanation = "No coherent solution structure emerged"
    
    print(f"1. Learning Outcome: {learning_outcome}")
    print(f"   {explanation}")
    print()
    
    # Knowledge consolidation
    if isinstance(Si_final, dict):
        avg_si_final = sum(Si_final.values()) / len(Si_final)
    else:
        avg_si_final = float(Si_final.mean()) if hasattr(Si_final, 'mean') else 0.0
    
    if avg_si_final > 0.7:
        consolidation = "STRONG - Knowledge well-consolidated"
    elif avg_si_final > 0.4:
        consolidation = "MODERATE - Some forgetting risk"
    else:
        consolidation = "WEAK - High forgetting risk"
    
    print(f"2. Knowledge Consolidation: {consolidation}")
    print(f"   Average Si: {avg_si_initial:.3f} → {avg_si_final:.3f}")
    print()
    
    # Learning efficiency
    if abs(depi_final) < abs(depi_initial):
        efficiency = "Learning has stabilized (good)"
    else:
        efficiency = "Still actively learning (needs more time)"
    
    print(f"3. Learning Efficiency: {efficiency}")
    print(f"   ∂EPI/∂t: {depi_initial:.3f} → {depi_final:.3f}")
    print()
    
    # Compare to traditional ML
    print("=" * 70)
    print("TNFR vs. Traditional ML: Key Differences")
    print("=" * 70)
    print()
    print("Traditional ML (Gradient Descent):")
    print("  • Loss function: Minimize prediction error")
    print("  • Learning: Iterative weight updates via backprop")
    print("  • Representation: Fixed architecture, learned weights")
    print("  • Adaptation: Requires retraining on new data")
    print()
    print("TNFR Paradigm (Structural Reorganization):")
    print("  • Coherence function: Maximize C(t), minimize |ΔNFR|")
    print("  • Learning: Structural reorganization via operators")
    print("  • Representation: EPI evolves dynamically, no fixed arch")
    print("  • Adaptation: Continuous via SelfOrganization operator")
    print()
    print("=" * 70)
    print("Key TNFR Insights:")
    print("=" * 70)
    print("• Learning ≠ gradient descent, it's STRUCTURAL REORGANIZATION")
    print("• Knowledge = EPI (information structure), not weights")
    print("• SelfOrganization operator = spontaneous strategy emergence")
    print("• No loss function needed - guided by ΔNFR and coherence")
    print("• Catastrophic forgetting → monitored via Si (stability)")
    print("• Agents coordinate through phase alignment, not explicit comm")
    print()
    print("Advantages of TNFR Learning:")
    print("  ✓ No architecture search needed (structure self-organizes)")
    print("  ✓ Continuous adaptation (no train/test separation)")
    print("  ✓ Interpretable (operators trace learning process)")
    print("  ✓ Forgetting resistance (high Si = stable knowledge)")
    print("=" * 70)


if __name__ == "__main__":
    run_example()

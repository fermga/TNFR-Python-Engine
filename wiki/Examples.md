# Examples & Use Cases 🔬

See TNFR in action across different domains. Each example demonstrates how to apply the resonant fractal paradigm to real-world problems.

## 🧬 Biology & Life Sciences

### Cellular Communication Network

Model how cells coordinate and communicate through chemical signaling:

```python
from tnfr.sdk import TNFRNetwork

# Create cellular network
cells = TNFRNetwork("cell_communication")

# Add cells with biological variation
results = (cells
    .add_nodes(20, epi_range=(0.8, 1.2))  # Natural variation in cell states
    .connect_nodes(0.3, "scale_free")      # Power-law connectivity (biological)
    .apply_operator("emission", target_nodes=[0])  # Initial signal
    .apply_sequence("therapeutic", repeat=5)  # Healing/coordination pattern
    .measure())

print(f"Network health: {results.coherence:.2%}")
print(f"Cell coordination: {results.sense_index:.3f}")

# Visualize cellular network
cells.visualize(
    node_labels=True,
    colormap="viridis",
    title="Cell Communication Network"
)
```

**What's happening:**
- Each node represents a cell with its own state (EPI)
- Connections represent signaling pathways
- "Therapeutic" sequence models healing/regeneration
- C(t) measures overall tissue health

### Neural Synchronization

Model neuronal firing patterns and synchronization:

```python
from tnfr.sdk import TNFRNetwork

# Create neural network
neurons = TNFRNetwork("neural_sync")

# Configure neurons
neurons.add_nodes(
    count=50,
    freq_range=(30, 80),  # Different firing rates (Hz_str)
    phase_sync=False       # Start desynchronized
)

# Connect with small-world topology (brain-like)
neurons.connect_nodes(0.2, "small_world")

# Apply synchronization sequence
for t in range(10):
    neurons.apply_operator("reception")    # Neurons listen to neighbors
    neurons.apply_operator("coherence")    # Synchronize
    neurons.apply_operator("resonance")    # Amplify synchronized patterns
    
    results = neurons.measure()
    print(f"t={t}: Phase coherence = {results.phase_coherence:.3f}")
```

**Applications:**
- Model epileptic seizures (excessive synchronization)
- Study consciousness emergence
- Understand brain oscillations (alpha, beta, gamma waves)

## 🌐 Social Networks & Communities

### Information Spreading

Model how information propagates through a social network:

```python
from tnfr.sdk import TNFRNetwork

# Create social network
community = TNFRNetwork("social_info_spread")

# Add people with varying influence
community.add_nodes(100, epi_range=(0.5, 2.0))

# Connect with small-world topology (Milgram's 6 degrees)
community.connect_nodes(0.15, "small_world")

# Seed information at influencers (high-degree nodes)
influencers = community.get_high_degree_nodes(count=5)
community.apply_operator("emission", target_nodes=influencers)

# Propagate through network
for step in range(20):
    community.apply_operator("resonance")  # Information spreads
    community.apply_operator("reception")  # People receive information
    
    results = community.measure()
    active = results.active_nodes  # Nodes with ΔNFR > 0
    
    print(f"Step {step}: {active}/{results.total_nodes} people reached")
    
    if active == results.total_nodes:
        print(f"Full saturation at step {step}")
        break
```

**Insights:**
- C(t) measures information coherence (clarity of message)
- Si measures network's capacity to sustain the narrative
- Phase coherence shows synchronized belief/understanding

### Opinion Dynamics

Model consensus formation or polarization:

```python
from tnfr.sdk import TNFRNetwork

# Create opinion network
opinions = TNFRNetwork("opinion_dynamics")

# Initialize with diverse opinions
opinions.add_nodes(50)
opinions.connect_nodes(0.25, "random")

# Introduce two opposing viewpoints
community_a = list(range(0, 25))
community_b = list(range(25, 50))

opinions.apply_operator("emission", target_nodes=[0], strength=1.0)
opinions.apply_operator("emission", target_nodes=[25], strength=-1.0)

# Let opinions evolve
for round in range(30):
    opinions.apply_operator("reception")   # People listen
    opinions.apply_operator("coherence")   # Form consensus locally
    
    results = opinions.measure()
    
    if results.sense_index < 0.3:
        print(f"Round {round}: ⚠ Polarization detected (Si={results.sense_index:.2f})")
        opinions.apply_operator("coupling")  # Encourage dialogue
    else:
        print(f"Round {round}: ✓ Healthy discourse (Si={results.sense_index:.2f})")
```

## 🤖 AI & Machine Learning

### Emergent Learning System

Model learning as resonance rather than gradient descent:

```python
from tnfr.sdk import TNFRNetwork
import numpy as np

# Create learning network
learner = TNFRNetwork("resonant_learner")

# Add processing nodes (like neural network layers)
learner.add_nodes(30, freq_range=(20, 50))
learner.connect_nodes(0.4, "scale_free")

# Training loop
for epoch in range(100):
    # Present pattern (emission)
    input_nodes = [0, 1, 2]
    learner.apply_operator("emission", target_nodes=input_nodes)
    
    # Allow resonance to organize representations
    learner.apply_sequence("exploration", repeat=3)  # Explore structure
    learner.apply_operator("coherence")              # Stabilize learning
    
    results = learner.measure()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: C(t)={results.coherence:.3f}, Si={results.sense_index:.3f}")
    
    # Check if learning converged
    if results.coherence > 0.85 and results.sense_index > 0.8:
        print(f"✓ Learning converged at epoch {epoch}")
        break
```

**Key Idea**: Learning is structural reorganization driven by resonance, not error minimization.

### Symbolic Reasoning Network

Model symbolic AI with TNFR:

```python
from tnfr.sdk import TNFRNetwork

# Create symbolic concept network
concepts = TNFRNetwork("symbolic_reasoning")

# Add concept nodes
concept_ids = {
    "animal": 0,
    "mammal": 1,
    "dog": 2,
    "cat": 3,
    "pet": 4,
}

concepts.add_nodes(len(concept_ids))

# Define is-a relationships (coupling)
concepts.couple_nodes(concept_ids["mammal"], concept_ids["animal"])
concepts.couple_nodes(concept_ids["dog"], concept_ids["mammal"])
concepts.couple_nodes(concept_ids["cat"], concept_ids["mammal"])
concepts.couple_nodes(concept_ids["dog"], concept_ids["pet"])
concepts.couple_nodes(concept_ids["cat"], concept_ids["pet"])

# Query: "Activate 'dog', see what resonates"
concepts.apply_operator("emission", target_nodes=[concept_ids["dog"]])
concepts.apply_sequence("propagation", repeat=5)

# Check what concepts are activated
results = concepts.measure_nodes()
for name, node_id in concept_ids.items():
    activation = results.node_activations[node_id]
    if activation > 0.5:
        print(f"✓ Concept '{name}' activated: {activation:.2f}")
```

## 🏗️ Distributed Systems & Technology

### IoT Sensor Coordination

Model distributed sensor network synchronization:

```python
from tnfr.sdk import TNFRNetwork

# Create sensor network
sensors = TNFRNetwork("iot_sensors")

# Deploy sensors in grid topology
sensors.add_nodes(36)  # 6x6 grid
sensors.connect_nodes(0.35, "lattice")

# Sensors need to synchronize their readings
for cycle in range(20):
    sensors.apply_operator("reception")     # Sense environment + neighbors
    sensors.apply_operator("coherence")     # Align measurements
    sensors.apply_operator("silence")       # Pause for reading
    
    results = sensors.measure()
    
    print(f"Cycle {cycle}: Sync quality = {results.sense_index:.3f}")
    
    if results.sense_index > 0.9:
        print(f"✓ Sensors fully synchronized at cycle {cycle}")
        break

# Check network resilience
print(f"\n--- Testing resilience ---")
sensors.remove_nodes([10, 11, 12])  # Simulate sensor failures
results_after = sensors.measure()

print(f"Coherence before: {results.coherence:.3f}")
print(f"Coherence after: {results_after.coherence:.3f}")
print(f"Network maintained: {results_after.coherence > 0.6}")
```

### Blockchain Consensus Model

Model distributed consensus formation:

```python
from tnfr.sdk import TNFRNetwork

# Create validator network
validators = TNFRNetwork("blockchain_consensus")

# Add validator nodes
validators.add_nodes(21, freq_range=(10, 30))  # Different processing speeds
validators.connect_nodes(0.8, "random")         # Highly connected

# Propose a block (emission from proposer)
proposer = 0
validators.apply_operator("emission", target_nodes=[proposer])

# Consensus rounds
for round_num in range(10):
    validators.apply_operator("reception")    # Validators receive proposal
    validators.apply_operator("coherence")    # Check validity
    validators.apply_operator("resonance")    # Vote propagation
    
    results = validators.measure()
    
    # Check consensus threshold
    if results.coherence > 0.67:  # 2/3 majority
        print(f"✓ Consensus reached at round {round_num}")
        print(f"  Coherence: {results.coherence:.2%}")
        break
    else:
        print(f"Round {round_num}: Coherence {results.coherence:.2%} (need >67%)")
```

## 📊 Finance & Markets

### Market Dynamics

Model financial market coordination and crashes:

```python
from tnfr.sdk import TNFRNetwork
import random

# Create market network
market = TNFRNetwork("financial_market")

# Add traders with different strategies
market.add_nodes(100, epi_range=(0.3, 1.5))
market.connect_nodes(0.2, "scale_free")  # Some highly connected traders

# Normal market operation
print("=== Normal Market ===")
for t in range(30):
    # Random trading signals
    active_traders = random.sample(range(100), 10)
    market.apply_operator("emission", target_nodes=active_traders)
    
    market.apply_operator("reception")
    market.apply_operator("coherence")
    
    results = market.measure()
    
    if t % 5 == 0:
        print(f"t={t}: Market coherence = {results.coherence:.3f}")

# Simulate market shock
print("\n=== Market Shock ===")
market.apply_operator("dissonance")  # Introduce instability
market.apply_operator("dissonance")  # Strong shock

results = market.measure()
print(f"After shock: C(t) = {results.coherence:.3f}, Si = {results.sense_index:.3f}")

if results.coherence < 0.3:
    print("⚠ MARKET CRASH - Applying stabilization")
    market.apply_sequence("stabilization", repeat=10)
    results_after = market.measure()
    print(f"After stabilization: C(t) = {results_after.coherence:.3f}")
```

## 🔬 Physics Simulation

### N-Body Gravitational System

Model gravitational interactions as structural resonance:

```python
from tnfr.sdk import TNFRNetwork
import numpy as np

# Create gravitational system
gravity = TNFRNetwork("nbody_gravity", backend="jax")  # Use GPU

# Add celestial bodies
n_bodies = 10
gravity.add_nodes(
    count=n_bodies,
    epi_range=(0.5, 2.0),   # Mass variation
    freq_range=(1, 10)       # Orbital frequency range
)

# Connect all bodies (gravity is long-range)
gravity.connect_nodes(1.0, "complete")

# Initial positions (random)
positions = np.random.randn(n_bodies, 3) * 10

# Simulation loop
dt = 0.01
for step in range(1000):
    # Apply gravitational operators
    gravity.apply_operator("reception")    # Feel gravitational field
    gravity.apply_operator("coupling")     # Mutual attraction
    gravity.apply_operator("transition")   # Update positions
    
    if step % 100 == 0:
        results = gravity.measure()
        print(f"Step {step}: Total energy (C(t)) = {results.coherence:.4f}")

# Check orbital stability
results_final = gravity.measure()
if results_final.sense_index > 0.7:
    print("✓ Stable orbits achieved")
else:
    print("⚠ Chaotic trajectories")
```

See full N-body validation: [`examples/nbody_quantitative_validation.py`](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/nbody_quantitative_validation.py)

## 🎨 Creative Applications

### Generative Art

Use TNFR to create emergent visual patterns:

```python
from tnfr.sdk import TNFRNetwork
import matplotlib.pyplot as plt

# Create artistic network
art = TNFRNetwork("generative_art")

# Create pattern generators
art.add_nodes(25, freq_range=(5, 50))
art.connect_nodes(0.4, "scale_free")

# Generate evolving pattern
pattern_history = []

for frame in range(100):
    # Apply creative operators
    art.apply_operator("emission", target_nodes=[0])
    art.apply_operator("dissonance")      # Introduce variation
    art.apply_operator("self_organization")  # Create structure
    art.apply_operator("coherence")       # Stabilize beauty
    
    # Capture state
    state = art.get_state_vector()
    pattern_history.append(state)
    
    if frame % 10 == 0:
        results = art.measure()
        print(f"Frame {frame}: Aesthetic coherence = {results.coherence:.3f}")

# Visualize evolution
plt.figure(figsize=(12, 4))
plt.imshow(pattern_history, aspect='auto', cmap='twilight')
plt.colorbar(label='Structural State')
plt.xlabel('Node ID')
plt.ylabel('Time Frame')
plt.title('Emergent Pattern Evolution')
plt.savefig('generative_art.png', dpi=150)
```

## 📚 More Examples

### In the Repository

The [`examples/`](https://github.com/fermga/TNFR-Python-Engine/tree/main/examples) directory contains many more examples:

- **`hello_world.py`** - Simplest possible TNFR network
- **`canonical_equation_demo.py`** - Direct demonstration of nodal equation
- **`coupling_network_formation.py`** - How nodes couple and synchronize
- **`regenerative_cycles.py`** - Self-sustaining patterns
- **`multiscale_network_demo.py`** - Hierarchical/fractal structures
- **`oz_propagation_demo.py`** - Dissonance and exploration
- **`health_analysis_demo.py`** - Network health monitoring
- **`backend_performance_comparison.py`** - NumPy vs JAX vs PyTorch
- **`domain_applications/`** - Domain-specific case studies

### Interactive Tutorials

```python
from tnfr.tutorials import (
    hello_tnfr,              # Basic introduction
    biological_example,      # Cell/neural networks
    social_network_example,  # Information dynamics
    technology_example,      # Distributed systems
    adaptive_ai_example,     # Learning through resonance
)

# Run any tutorial
biological_example()
```

## 🎯 Choosing the Right Example

**I want to model...**

- **Biological systems** → Cellular communication, Neural synchronization
- **Social dynamics** → Information spreading, Opinion formation
- **AI/ML** → Emergent learning, Symbolic reasoning
- **Distributed tech** → IoT coordination, Blockchain consensus
- **Finance** → Market dynamics, Risk analysis
- **Physics** → N-body gravity, Wave propagation
- **Art/Creative** → Generative patterns, Music composition

## 💡 Tips for Your Own Applications

### 1. Identify Your Nodes

**What are the coherent units in your system?**
- Biology: cells, proteins, organisms
- Social: people, organizations, ideas
- Tech: servers, sensors, agents

### 2. Define Couplings

**How do your nodes interact?**
- Direct communication (explicit edges)
- Field effects (all-to-all, distance-weighted)
- Hierarchical (parent-child)

### 3. Choose Operators

**What transformations occur?**
- Start: `Emission`, `Coupling`
- Stabilize: `Coherence`, `Silence`
- Adapt: `Dissonance`, `Mutation`
- Propagate: `Resonance`, `Transition`

### 4. Select Metrics

**What defines "health" in your domain?**
- C(t): Overall system stability
- Si: Ability to adapt without breaking
- Phase coherence: Synchronization level
- ΔNFR: Growth vs. contraction

### 5. Validate

**How do you know it's working?**
- Compare to real data
- Check conservation laws (if applicable)
- Test edge cases (failures, shocks, attacks)
- Verify reproducibility

## 🔗 Additional Resources

- **Full Documentation**: https://tnfr.netlify.app
- **API Reference**: https://tnfr.netlify.app/api/overview/
- **Theory**: https://tnfr.netlify.app/theory/mathematical_foundations/
- **GitHub Examples**: https://github.com/fermga/TNFR-Python-Engine/tree/main/examples

## 🤝 Share Your Work

Built something cool with TNFR? We'd love to hear about it!

- Open a [GitHub Discussion](https://github.com/fermga/TNFR-Python-Engine/discussions)
- Submit a PR with your example
- Tag us on social media

---

[← Core Concepts](Core-Concepts.md) | [Back to Home](Home.md) | [Getting Started →](Getting-Started.md)

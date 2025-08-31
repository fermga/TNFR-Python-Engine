# tnfr · Python package

> Engine for **modeling, simulation, and measurement** of multiscale structural coherence through **structural operators** (emission, reception, coherence, dissonance, coupling, resonance, silence, expansion, contraction, self‑organization, mutation, transition, recursivity).

---

## What is `tnfr`?

`tnfr` is a Python library to **operate with form**: build nodes, couple them into networks, and **modulate their coherence** over time using structural operators. It does not describe “things”; it **activates processes**. Its theoretical basis is the Resonant Fractal Nature Theory (TNFR), which understands reality as **networks of coherence** that persist because they **resonate**.

In practical terms, `tnfr` lets you:

* Model **Resonant Fractal Nodes (NFR)** with parameters for **frequency** (νf), **phase** (θ), and **form** (EPI).
* Apply **structural operators** to start, stabilize, propagate, or reconfigure coherence.
* **Simulate** nodal dynamics with discrete/continuous integrators.
* **Measure** global coherence C(t), nodal gradient ΔNFR, and the **Sense Index** (Si).
* **Visualize** states and trajectories (coupling matrices, C(t) curves, graphs).

> **Nodal equation (operational core)**
>
> $\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f\,\cdot\,\Delta\mathrm{NFR}(t)$
>
> A form emerges and persists when **internal reorganization** (ΔNFR) **resonates** with the node’s **frequency** (νf).

---

## Installation

```bash
pip install tnfr
```

Requires **Python ≥ 3.9**.

---

## Why TNFR (in 60 seconds)

* **From objects to coherences:** you model **processes** that hold, not fixed entities.
* **Operators instead of rules:** you compose **structural operators** (e.g., *emission*, *coherence*, *dissonance*) to **build trajectories**.
* **Operational fractality:** the same pattern works for **ideas, teams, tissues, narratives**; the scales change, **the logic doesn’t**.

---

## Getting started (minimal recipe)

> *The high‑level API centers on three things: nodes, operators, simulation.*

```python
# 1) Nodes and network
import tnfr as T

# A minimal set of nodes with initial frequency (νf)
A = T.Node(label="seed", nu_f=0.8)
B = T.Node(label="context", nu_f=0.6)
net = T.Network([A, B], edges=[(A, B, 0.7)])  # coupling 0..1

# 2) Sequence of structural operators
ops = [
    T.ops.Emission(strength=0.4),      # start pattern
    T.ops.Coupling(weight=0.7),        # synchronize nodes
    T.ops.Coherence(),                 # stabilize form
]

# 3) Simulation and metrics
traj = T.sim.run(net, ops, steps=200, dt=0.05)
print("C(t) =", T.metrics.coherence(traj)[-1])
print("Si   =", T.metrics.sense_index(traj))

# 4) Quick visualization
T.viz.plot_coherence(traj)     # C(t) curve
T.viz.plot_network(net)        # graph/couplings
```

> **Note:** Specific class/function names may vary across minor versions. Check `help(T.ops)` and `help(T.sim)` for your installed API.

---

## Key concepts (operational summary)

* **Node (NFR):** a unit that persists because it **resonates**. Parameterized by **νf** (frequency), **θ** (phase), and **EPI** (coherent form).
* **Structural operators:** functions that reorganize the network. We use **functional** names (not phonemes):

  * **Emission** (start), **Reception** (open), **Coherence** (stabilize), **Dissonance** (creative tension), **Coupling** (synchrony), **Resonance** (propagate), **Silence** (latency), **Expansion**, **Contraction**, **Self‑organization**, **Mutation**, **Transition**, **Recursivity**.
* **Magnitudes:**

  * **C(t):** global coherence.
  * **ΔNFR:** nodal gradient (need for reorganization).
  * **νf:** structural frequency (Hz\_str).
  * **Si:** sense index (ability to generate stable shared coherence).

---

## Typical workflow

1. **Model** your system as a network: nodes (agents, ideas, tissues, modules) and couplings.
2. **Select** a **trajectory of operators** aligned with your goal (e.g., *start → couple → stabilize*).
3. **Simulate** the dynamics: number of steps, step size, tolerances.
4. **Measure**: C(t), ΔNFR, Si; identify bifurcations and collapses.
5. **Iterate** with controlled **dissonance** to open mutations without losing form.

---

## High‑level API (orientation map)

> The typical module layout in `tnfr` is:

* `tnfr.core`: `Node`, `Network`, `EPI`, `State`
* `tnfr.ops`: structural operators (Emission, Reception, Coherence, Dissonance, ...)
* `tnfr.sim`: integrators (`run`, `step`, `integrate`), dt control and thresholds
* `tnfr.metrics`: `coherence`, `gradient`, `sense_index`, `phase_sync`
* `tnfr.viz`: plotting utilities (`plot_coherence`, `plot_network`, `plot_phase`)

Usage examples:

```python
from tnfr import core, ops, sim, metrics

net = core.Network.from_edges([
    ("n1", "n2", 0.6),
    ("n2", "n3", 0.8),
])

sequence = [ops.Emission(0.3), ops.Coupling(0.5), ops.Coherence()]
traj = sim.run(net, sequence, steps=500)

print(metrics.coherence(traj))
```

---

## Parametric modeling

```python
import tnfr as T

net = T.Network.uniform(n=25, nu_f=0.4, coupling=0.3)
plan = (
    T.ops.Emission(0.2)
  >> T.ops.Expansion(0.4)
  >> T.ops.Coupling(0.6)
  >> T.ops.Coherence()
)
traj = T.sim.run(net, plan, steps=800)
T.viz.plot_phase(traj)
```

---

## Main metrics

* `coherence(traj) → C(t)`: global stability; higher values indicate sustained form.
* `gradient(state) → ΔNFR`: local demand for reorganization (high = risk of collapse/bifurcation).
* `sense_index(traj) → Si`: proxy for **structural sense** (capacity to generate shared coherence) combining **νf**, phase, and topology.

---

## Best practices

* **Short sequences** and frequent C(t) checks avoid unnecessary collapses.
* Use **dissonance** as a tool: introduce it to open possibilities, but **seal** with coherence.
* **Scale first, detail later:** tune coarse couplings before micro‑parameters.

---

## Project status

* **pre‑1.0 API**: signatures may be refined; concepts and magnitudes are stable.
* **Pure‑Python** core with minimal dependencies (optional: `numpy`, `matplotlib`, `networkx`).

---

## Contributing

Suggestions, issues, and PRs are welcome. Guidelines:

1. Prioritize **operational clarity** (names, docstrings, examples).
2. Add **tests** and **notebooks** that show the structural effect of each PR.
3. Keep **semantic neutrality**: operators act on form, not on contents.

---

## License

MIT 

---

## References & notes

* Theoretical foundations: TNFR operational manual.
* Operational definitions: nodal equation, dimensions (frequency, phase, form), and structural operators.

> If you use `tnfr` in research or projects, please cite the TNFR conceptual framework and link to the PyPI package.

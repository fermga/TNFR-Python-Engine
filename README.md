# TNFR Python Project

Engine for **modeling, simulation and measurement** of multiscale structural coherence through **structural operators** (emission, reception, coherence, dissonance, coupling, resonance, silence, expansion, contraction, self‑organization, mutation, transition, recursivity).

---

## What is `tnfr`?

`tnfr` is a Python library to **operate with form**: build nodes, couple them into networks, and **modulate their coherence** over time using structural operators. It does not describe “things”; it **activates processes**. Its theoretical basis is the *Teoria de la Naturaleza Fractal Resonante (TNFR)*, which understands reality as **networks of coherence** that persist because they **resonate**.

In practical terms, `tnfr` lets you:

* Model **Resonant Fractal Nodes (NFR)** with parameters for **frequency** (νf), **phase** (θ), and **form** (EPI).
* Apply **structural operators** to start, stabilize, propagate, or reconfigure coherence.
* **Simulate** nodal dynamics with discrete/continuous integrators.
* **Measure** global coherence C(t), nodal gradient ΔNFR, and the **Sense Index** (Si).
* **Visualize** states and trajectories (coupling matrices, C(t) curves, graphs).

A form emerges and persists when **internal reorganization** (ΔNFR) **resonates** with the node’s **frequency** (νf).

---

## Installation

```bash
pip install tnfr
```
* https://pypi.org/project/tnfr/
* Requires **Python ≥ 3.9**.
* Install extras:
  * NumPy: `pip install tnfr[numpy]`
  * YAML: `pip install tnfr[yaml]`
  * Both: `pip install tnfr[numpy,yaml]`

---

## Why TNFR (in 60 seconds)

* **From objects to coherences:** you model **processes** that hold, not fixed entities.
* **Operators instead of rules:** you compose **structural operators** (e.g., *emission*, *coherence*, *dissonance*) to **build trajectories**.
* **Operational fractality:** the same pattern works for **ideas, teams, tissues, narratives**; the scales change, **the logic doesn’t**.

---

## Key concepts (operational summary)

* **Node (NFR):** a unit that persists because it **resonates**. Parameterized by **νf** (frequency), **θ** (phase), and **EPI** (coherent form).
* **Structural operators** - functions that reorganize the network:

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

## Main metrics

* `coherence(traj) → C(t)`: global stability; higher values indicate sustained form.
* `gradient(state) → ΔNFR`: local demand for reorganization (high = risk of collapse/bifurcation).
* `sense_index(traj) → Si`: proxy for **structural sense** (capacity to generate shared coherence) combining **νf**, phase, and topology.

---

## History configuration

Recorded series are stored under `G.graph['history']`. Set `HISTORY_MAXLEN` in
the graph (or override the default) to keep only the most recent entries. When
the limit is positive the library uses bounded `deque` objects and removes the
least populated series when the number of history keys grows beyond the limit.

### Random node sampling

To reduce costly comparisons the engine stores a per‑step random subset of
node ids under `G.graph['_node_sample']`. Operators may use this to avoid
scanning the whole network. Sampling is skipped automatically when the graph
has fewer than **50 nodes**, in which case all nodes are included.

---

## Trained GPT

https://chatgpt.com/g/g-67abc78885a88191b2d67f94fd60dc97-tnfr-teoria-de-la-naturaleza-fractal-resonante

---

## MIT License

Copyright (c) 2025 TNFR - Teoría de la naturaleza fractral resonante

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

If you use `tnfr` in research or projects, please cite the TNFR conceptual framework and link to the PyPI package.

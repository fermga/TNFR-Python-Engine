# TNFR Python Project

> Engine for **modeling, simulation, and measurement** of multiscale structural coherence through **structural operators** (emission, reception, coherence, dissonance, coupling, resonance, silence, expansion, contraction, self‑organization, mutation, transition, recursivity).

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

Requires **Python ≥ 3.9**.

---

## Why TNFR (in 60 seconds)

* **From objects to coherences:** you model **processes** that hold, not fixed entities.
* **Operators instead of rules:** you compose **structural operators** (e.g., *emission*, *coherence*, *dissonance*) to **build trajectories**.
* **Operational fractality:** the same pattern works for **ideas, teams, tissues, narratives**; the scales change, **the logic doesn’t**.

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

## Main metrics

* `coherence(traj) → C(t)`: global stability; higher values indicate sustained form.
* `gradient(state) → ΔNFR`: local demand for reorganization (high = risk of collapse/bifurcation).
* `sense_index(traj) → Si`: proxy for **structural sense** (capacity to generate shared coherence) combining **νf**, phase, and topology.

---

---

## GPT

* https://chatgpt.com/g/g-67abc78885a88191b2d67f94fd60dc97-tnfr-teoria-de-la-naturaleza-fractal-resonante

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

## Versioning

The Python package (`pyproject.toml`) and the Node package (`package.json`) share a
unified version number. Releases follow [Semantic Versioning](https://semver.org/)
compatible with [PEP 440](https://peps.python.org/pep-0440/). Ensure both files are
updated together.

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

If you use `tnfr` in research or projects, please cite the TNFR conceptual framework and link to the PyPI package.

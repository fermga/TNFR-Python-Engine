# General Project Structure

* **Package entry point.** `__init__.py` registers modules under short names to avoid circular imports and exposes the public API: `preparar_red`, `step`, `run`, and observation utilities.

* **Configuration & constants.** `constants.py` centralizes default parameters (discretization, EPI and νf ranges, mixing weights, re-mesh limits, etc.) and provides utilities to inject them into the network (`attach_defaults`, `merge_overrides`), along with standardized aliases for node attributes.

* **Cross-cutting utilities.** `helpers.py` offers core numeric helpers, alias-based attribute accessors, neighborhood statistics, glyph history, a callback system, and computation of the sense index `Si` for each node.

* **Dynamics engine.** `dynamics.py` implements the simulation loop: ΔNFR field computation, nodal equation integration, glyph selection/application, clamps, phase coordination, history updates, and conditional re-mesh (`step` and `run`).

* **Glyph operators.** `operators.py` defines the 13 glyphs as local transformations, a dispatcher `aplicar_glifo`, and both direct and stability-conditioned re-mesh utilities.

* **Observers & metrics.** `observers.py` registers standard callbacks and computes global coherence, phase synchrony, Kuramoto order, glyph distribution, and the sense vector `Σ⃗`, among others.

* **Simulation orchestration.** `ontosim.py` prepares a NetworkX graph, attaches configuration, and initializes attributes (EPI, phases, frequencies) before delegating dynamics to `dynamics.step`/`run`.

* **Demo CLI.** `main.py` generates an Erdős–Rényi network, lets you tweak basic parameters, and runs the simulation while displaying final metrics.

---

## Key Concepts to Grasp

* **Aliased dependency tree.** Modules import each other via global aliases to simplify access and prevent cycles—essential for navigating the code unambiguously.

* **Normalized node attributes.** All data (EPI, phase `θ`, frequency `νf`, `ΔNFR`, etc.) live in `G.nodes[n]` under compatible alias names, making extensions and custom hooks straightforward.

* **Sense Index (`Si`).** Combines normalized frequency, phase dispersion, and field magnitude to evaluate each node’s “sense,” influencing glyph selection.

* **Step-wise engine.** `dynamics.step` orchestrates eight phases: field computation, `Si`, glyph selection & application, integration, clamps, phase coordination, history update, and conditioned re-mesh.

* **Glyphs as operators.** Each glyph applies a smooth transformation to node attributes (emission, diffusion, coupling, dissonance, etc.), dispatched by a configurable, typographic name.

* **Network re-mesh.** Mixes the current state with a past one (memory `τ`) to stabilize the network, with clear precedence for `α` and conditions based on recent stability and synchrony history.

* **Callbacks & observers.** The `Γ(R)` system lets you hook functions before/after each step and after re-mesh, enabling monitoring or external intervention.

---

## Recommendations for Going Deeper

* **NetworkX & the Graph API.** Get comfortable with how NetworkX handles attributes and topology; all dynamics operate on `Graph` objects and their properties.

* **Extending the ΔNFR field.** Explore `set_delta_nfr_hook` to implement alternative nodal fields and learn how metadata and mixing weights are recorded.

* **Designing new glyphs.** Review `operators.py` to add operators or adjust factors in `DEFAULTS['GLYPH_FACTORS']`.

* **Custom observers.** Implement your own metrics via `register_callback` or by extending `observers.py` to measure phenomena specific to your study.

* **Theoretical reading.** For conceptual background, see the included PDFs (`TNFR.pdf`, *El Pulso que nos Atraviesa*), which deepen the fractal-resonant framework.

* **Advanced parameters.** Experiment with adaptive phase coordination, stability criteria, and the glyph grammar to observe their impact on network self-organization.

---

**Mastering these pieces will let you extend the simulation, build analysis pipelines and connect the theory with computational applications.**

## Testing

Install the dependencies and project in editable mode before running the test suite with `pytest`:

```
pip install networkx
pip install -e .
pytest
```

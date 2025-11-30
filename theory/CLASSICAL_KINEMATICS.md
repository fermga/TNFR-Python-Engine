# Classical Kinematics Memo

**Status**: Technical memo  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/CLASSICAL_KINEMATICS.md`

---

## 1. Scope and Inertial Regime

When nodes experience zero structural pressure the nodal equation reduces to the classical constant-velocity form:

\[
\Delta \text{NFR} = 0 \quad \Rightarrow \quad \frac{\partial \text{EPI}}{\partial t} = 0 \text{ (co-moving frame)}.
\]

Practical checklist:

- Ensure operator schedules exclude destabilizers (no OZ/VAL) so \(\Delta \text{NFR} = 0\).  
- Confirm telemetry shows \(|\nabla \phi| < 10^{-4}\) and \(K_\phi \approx 0\) over the interval.  
- Record initial phase current \(J_\phi\) as the momentum analog and log \(C(t)\) to verify coherence remains >0.99.

---

## 2. Worked Example: Two-Train Scenario

| Parameter | Train A | Train B |
| --- | --- | --- |
| Initial position | \(x=0\,\text{km}\) | \(x=600\,\text{km}\) |
| Velocity | \(+300\,\text{km/h}\) | \(-250\,\text{km/h}\) |
| Operators | `[AL, IL, SHA]` | `[AL, IL, SHA]` |

Simulation workflow:

1. Initialize two nodes in a 1D manifold with the parameters above; store configuration in `configs/kinematics/train_pair.yaml`.  
2. Integrate using the zero-force symplectic kernel (see `examples/15_train_crossing_demo.py`).  
3. Detect the time \(t_c\) when positions intersect and log \(C(t)\), \(\Phi_s\), and \(J_\phi\) histories to `results/kinematics_demo/run_<seed>.csv`.

Analytical prediction for the intersection:

\[
t_c = \frac{600}{300 + 250} \text{ h} \approx 1.0909 \text{ h}, \quad x_c = v_A t_c \approx 327.27 \text{ km}.
\]

Numerical runs should match these values within integration tolerance (baseline example uses \(\Delta t = 0.1\,\text{min}\)). Deviations larger than \(10^{-3}\) indicate insufficient resolution or unintended structural forces (check for sneaky OZ/VAL events).

---

## 3. Artifacts and Telemetry

Running `python examples/15_train_crossing_demo.py` produces:

- `results/kinematics_demo/01_train_crossing.png` – overlay of simulated trajectories with the analytical intersection point.  
- `results/kinematics_demo/run_<seed>.csv` – time series containing \(x_A\), \(x_B\), \(J_\phi\), and \(C(t)\).

Artifacts must include metadata (seed, \(\Delta t\), integrator) plus telemetry columns for \(|\nabla \phi|\), \(K_\phi\), and \(C(t)\). Store SHA256 manifests under `results/kinematics_demo/manifest.json`.

---

## 4. Outstanding Work

1. Document how non-zero \(\Delta \text{NFR}\) reintroduces acceleration and cross-link to `theory/CLASSICAL_MECHANICS_CORRESPONDENCE.md`.  
2. Add regression tests comparing simulated crossings against analytic solutions for multiple velocity pairs (store expected values in `tests/data/kinematics_cases.json`).  
3. Report floating-point sensitivity (single vs. double precision) for long trajectories and note any impacts on \(C(t)\) or \(J_\phi\).

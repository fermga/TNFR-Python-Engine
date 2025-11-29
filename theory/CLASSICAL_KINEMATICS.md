# Classical Kinematics from Nodal Dynamics

TNFR is a superset of Classical Mechanics. This means it can trivially solve standard kinematic problems (like the "Two Trains" problem) by configuring nodes in the **Inertial Regime**.

## The Inertial Regime

In TNFR, "Inertia" is the persistence of a structural pattern's propagation through the manifold when no external pressure ($\Delta NFR$) acts on it.

- **Condition**: $\Delta NFR = 0$ (Zero Structural Pressure).
- **Nodal Equation**: $\partial EPI / \partial t = \nu_f \cdot 0 = 0$ (in the co-moving frame).
- **Result**: The node maintains its "Phase Current" ($J_\phi$) or Momentum ($p$).

## The Two Trains Problem

**Scenario**:
- Train A leaves Madrid ($x=0$) at $v_A = 300$ km/h.
- Train B leaves Barcelona ($x=600$ km) at $v_B = -250$ km/h.
- **Question**: When and where do they cross?

**TNFR Solution**:
We model the trains as two nodes ($N_A, N_B$) in a 1D structural manifold.
1.  **Initialize**: Set initial positions ($q$) and momenta ($p = m \cdot v$).
2.  **Evolve**: Use the Symplectic Integrator with `force_func = zero_force`.
3.  **Detect**: Monitor the topological intersection $q_A(t) = q_B(t)$.

## Visual Proofs

The script `examples/15_train_crossing_demo.py` demonstrates this exact scenario.

- `results/kinematics_demo/01_train_crossing.png`: Plot of the trajectories $x_A(t)$ and $x_B(t)$ intersecting at the precise analytical solution ($t \approx 65.45$ min, $x \approx 327.27$ km).

**Accuracy**: The simulation matches the analytical solution with **zero error** ($< 10^{-6}$), proving that Nodal Dynamics correctly reduces to Galilean Kinematics in the limit of zero structural pressure.

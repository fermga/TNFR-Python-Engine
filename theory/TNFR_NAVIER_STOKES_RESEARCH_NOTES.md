# TNFR–Navier–Stokes — Research Notes

**Status:** auxiliary finite-resolution Navier–Stokes diagnostics; no nodal
derivation of the full fluid model and no Clay proof or counterexample.
**Scope review:** 2026-09-18; documentation correction, no new numerical run.

The [official Clay statement by Fefferman](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf)
poses global existence/smoothness or breakdown alternatives for three-dimensional
incompressible flow at positive viscosity, with specified smooth data on
`R³` or the periodic domain. Global smoothness for unrestricted admissible data
at fixed positive viscosity remains open. A bound uniform as `ν→0` concerns a
separate inviscid-limit question; it is not the Clay statement.

---

## 1. The reading

The repository contains pure-EPI graph diffusion, an auxiliary graph-wave
model and a separate pseudo-spectral fluid solver. Their diagnostics can be
compared after declaring state, operator, units and time. An oscillatory signal
does not alone identify conservative dynamics or a TNFR operator realization.
`verify_overdamped_projection` checks a declared damped graph-wave model; it
does not infer a physical fluid or neural equation from observations.

## 2. Linear diffusion and nonlinear fluid dynamics

For smooth unforced incompressible flow, the vorticity equation is

$$\frac{\partial \omega}{\partial t} + (u\cdot\nabla)\omega
= (\omega\cdot\nabla)u + \nu\nabla^2\omega.$$

The omitted-advection form applies to the material derivative, not to the
displayed partial time derivative. Linearization around rest leaves diffusion;
linearization around other flows can also retain transport and stretching.

The canonical isolated EPI channel has `ẋ=−diag(νf)L_rw x` on a fixed graph.
This supplies a comparison with diffusion, not an identity between its
normalized graph Laplacian and the solver's continuum Fourier multiplier
`|k|²`. A spatial scaling/consistency bridge and channel identification are
required. In particular, fluid vorticity is not the canonical wrapped scalar
phase curvature by definition, and IL/VAL labels do not implement the fluid
terms.

The helper `face_of_flow` selects `γ=1/ν` in an auxiliary graph-wave equation.
Its overdamping condition depends on `γ²` relative to `4λ_max`; it does not
hold for every positive viscosity. This selected comparison neither derives
Navier–Stokes from the nodal equation nor certifies nonlinear regularity.

## 3. Historical comparison dictionary

These are proposed analogies, not established state maps or operator contracts.

| Navier–Stokes | TNFR |
|---|---|
| velocity `u_a` | A possible scalar chart; no phase-valued identification is derived |
| vorticity `ω = ∇×u` | Derivative diagnostic; not the implemented nodewise `K_φ` |
| incompressibility pressure `p` | No established equality with the tetrad potential `Φ_s` |
| viscosity `ν` | Diffusion coefficient; mapping to `ν_f` needs spatial/time scales |
| enstrophy `½‖ω‖²` | Quadratic derivative budget, not a generally conserved tetrad energy |
| stretching `(ω·∇)u` | Nonlinear source in the supplied PDE, not an executed VAL operator |

## 4. Finite cascade telemetry

`conservative_face.measure_cascade_frontier` evolves the pseudo-spectral
Taylor–Green initial condition at several viscosities to a selected scaled time
`τ_str = ν·t` and records the peak enstrophy debt `Ω_peak/Ω₀`, the peak stretching
production and the high-mode enstrophy fraction versus `Re = 2π/ν`.

- **Reported finite runs:** enstrophy peaks and then decays over the sampled
  trajectories. This does not prove continuum or infinite-time regularity,
  even at those fixed positive viscosities.
- **Reported sweep:** peak enstrophy grows with Re. Neither this finite trend
  nor a divergent inviscid-limit trend establishes finite-time breakdown at a
  fixed positive viscosity.

Matched `τ_str=νt` also changes the physical observation horizon as viscosity
changes. Resolution, timestep and horizon are part of the comparison, not
an independently established nodal time identification.

## 5. Reusable diagnostic pattern

Separating a total budget, its distribution across scales and the measured
production/dissipation balance is useful for TNFR pattern-persistence studies.
Those quantities must be rederived from the actual canonical evolution being
tested. A shared spectral vocabulary does not identify fluid, arithmetic or
neural state spaces or transfer a stability theorem between them.

## 6. Spectral moment hierarchy

For the supplied Fourier fluid model, derivative-weighted energies form the
hierarchy `M_p = Σ |k|^(2p) E_k` (`cascade_moment_hierarchy`). These Fourier
weights are not the bounded eigenvalues of a normalized finite-graph `L_rw`:

- `M_0` = energy — nonincreasing for the unforced smooth continuum model;
- `M_1` = enstrophy;
- `M_2` = palinstrophy — weights the small-scale (high-`λ`) tail more.

**Historically reported** (Taylor–Green at sampled peak enstrophy,
Re 157→628, ×4): `M_0` **decreases** (×0.62),
while `M_1` grows (×1.71) and `M_2` grows much faster (×11.3); the moment
**ratios climb** with Re (`M_1/M_0`: 3.0→8.3; `M_2/M_1`: 3.0→19.8). These
are finite numerical read-outs, not an all-time bound on higher derivatives.
The reported `k_max·η>1` flag is a heuristic: the implementation uses `n/2`,
while nonlinear dealiasing uses a per-axis cutoff `floor(n/3)`. It is not a
validated error bound or a proof that all relevant scales are resolved.
Driver: `benchmarks/ns_moment_hierarchy_cascade.py`.

**Balance diagnostic.** For smooth periodic unforced flow, the enstrophy
balance is `dM_1/dt = P − 2ν M_2`
(`moment_ladder_closure`), with the exact modal Cauchy–Schwarz coupling
`M_1² ≤ M_0·M_2` (interpolation saturation `s = M_1²/(M_0 M_2) ∈ [0,1]`
when the denominator is positive). Historically reported
at sampled peaks (Re 314→628), `P/(2ν M_2) ≈ 1`
(`1.00`, `1.00`, `0.98`) is consistent with near balance of the two measured terms;
it does not certify the numerical trajectory derivative. The
interpolation saturation `s` **decreases** with Re (`0.57 → 0.42 → 0.38`) — the
spectrum is less concentrated by this ratio. No regularity implication follows
from that trend alone. Additional historical ratios (`0.70 → 0.59 → 0.45`)
are below one. Under the displayed balance they indicate decreasing enstrophy,
so the former label "growth phase" cannot substantiate growth or closure.
Those records need timestamp/derivative reconciliation before further use.
A sampled ratio below one is not a uniform differential estimate; even a
uniform inviscid-limit estimate would require its own theorem and scope.

**A scoped cross-program comparison.** The two programs admit a useful
moment-ladder analogy:

| | low moment (bounded) | high moment (the open wall) |
|---|---|---|
| **Riemann** | RMS of `S(T)` — Selberg `√(log log T)` | sup of `S(T)` (the extremes) |
| **NS** | energy `M_0` — Leray | enstrophy `M_1`, palinstrophy `M_2`, … |

The state spaces, operators and quantified bounds differ. The comparison does
not identify the two open problems or make one bound imply the other. The
finite measurements close neither problem.

## 7. Static pressure coherence on the emergent geometry

`flow_coherence` constructs a domain-specific pressure on vorticity magnitude,
`ΔNFR = −L_rw·|ω|`, and evaluates the shared scalar map as

```
C_static = structural_coherence(mean|ΔNFR|, 0)
         = 1 / (1 + mean|ΔNFR|).
```

The second argument is an explicit static assumption: this observer does not
materialize an EPI-rate channel for the Navier--Stokes trajectory. Consequently
`C_static` is not the engine's dynamic total `C(t)`. Reusing the numeric map and
zero-pressure predicate does not identify graph, arithmetic, chemical and fluid
state spaces, and it proves no common attractor. The compatibility key
`at_equilibrium` means only that the snapshot's mean pressure magnitude lies
within the selected `ΔNFR` tolerance while `dEPI` is held at zero.

**Historically reported finite-resolution trend.** Applied to successive simulated
vorticity snapshots, the score approaches one at the end of each sampled run
(final `C_static = 0.995 → 0.9985` over Re 157→1257), while the minimum score
decreases with Re (`0.94 → 0.90 → 0.84 → 0.72`). These numbers record flattening
and excursion of the constructed pressure field. They do not self-certify the
full flow, establish convergence, or transfer the uniform-field attractor of
fixed-graph pure-EPI diffusion to nonlinear Navier--Stokes dynamics.

A falsifiable finite-resolution question is whether the observed minimum
`C_static` remains above the selected telemetry floor `1/(π+1)` as resolution
and Re increase. That threshold application is a diagnostic policy; it is not
equivalent to a regularity criterion. It supplies neither a fixed-viscosity
global theorem nor an inviscid-limit bound. U2 remains a grammar policy, the structural
energy is only a Lyapunov candidate outside proved model-specific cases, U5 is
a hierarchy contract, and the REMESH analysis supplies no runtime infinity
limit.

## 8. Honest scope

This program supplies finite observations of an imported PDE solver and
auxiliary comparisons. It does not derive fluid velocity, vorticity,
incompressibility or vortex stretching from TNFR. The open fixed-viscosity
global problem and the separate inviscid-limit question remain unresolved.
Higher resolution alone would not supply an analytic proof or a verified
continuum error bound. Existing compatibility names and older source docstrings
must be read within these limits; this documentation correction changes no
solver, test or historical telemetry.

## 9. Code map

- `src/tnfr/navier_stokes/operator.py` — auxiliary pseudo-spectral 3D NS
  integrator (`TNFRNavierStokes`) + `build_torus_graph_3d` +
  `taylor_green_initial_condition_3d`.
- `src/tnfr/navier_stokes/conservative_face.py` — `verify_diffusive_face`,
  `face_of_flow`, `vorticity_modal_spectrum`, `cascade_moment_hierarchy`,
  `moment_ladder_closure`, `flow_coherence` (static pressure-only read-out),
  `measure_cascade_frontier`, `CascadeFrontierCertificate`.
- `examples/06_navier_stokes/158_navier_stokes_two_face_refounded.py` — demo.
- `examples/10_applications/159_empirical_confrontation_pipeline.py` — the
  data-confrontation pipeline (map a signal to scoped read-outs + face).
- `benchmarks/ns_moment_hierarchy_cascade.py` — the λ-moment hierarchy vs Re.
- Tests: `tests/mathematics/test_navier_stokes_refounded.py`.

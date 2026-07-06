# TNFR–Navier–Stokes — Research Notes

**Status:** research program on the Clay Millennium Problem (existence and
smoothness of 3D incompressible Navier–Stokes). Closes **nothing**; both Clay
directions remain **open**. This note is the canonical, self-contained record of
the program.

---

## 1. The reading

TNFR reads a system through its emergent geometry: the **pulse** `ω_k = √λ_k`, the
two faces (diffusive over-damped vs conservative wave) with the
`verify_overdamped_projection` certificate, and the empirical confrontation of the
canonical magnitudes with real oscillatory data. That empirical arm establishes
that **oscillatory dynamics live on the conservative (wave) face**, which poses
the honest question this program answers: *which face is 3D Navier–Stokes on?*

## 2. The honest two-face answer (physics-first, no forced analogy)

Incompressible NS vorticity obeys

$$\frac{\partial \omega}{\partial t} = (\omega\cdot\nabla)u \;+\; \nu\,\nabla^2\omega .$$

This is **first order in time**. Therefore:

- Its **linear** part is the nodal equation for the phase-curvature field
  `K_φ` (the vorticity): `∂K_φ/∂t = ν_f·ΔNFR` with `ν_f ↔ ν` and
  `ΔNFR = −L_rw·K_φ`. The viscous term **is** the canonical graph diffusion (the
  IL coherence stabiliser). This sits on the **diffusive (over-damped) face**:
  mapping viscosity to the damped-wave damping `γ = 1/ν` (the canonical
  `ν_f = 1/γ` identity), every physical viscosity gives `γ² ≫ 4λ_max`, so
  `verify_diffusive_face` (the engine's `verify_overdamped_projection`) is **VALID**
  and recovers `ν_f = ν`. Unlike EEG — whose *linear* neural dynamics are
  under-damped/oscillatory — **linear NS carries no oscillatory content**.
- Its **conservative / inertial** character — the energy-conserving Euler cascade
  where any blow-up must live — is entirely in the **nonlinear** vortex-stretching
  source `(ω·∇)u` (the VAL destabiliser).

**Consequence (the sharp statement).** NS blow-up is **not** a linear-wave
resonance; it is a purely **nonlinear `K_φ` cascade**: does the stretching source
pump enstrophy into ever-higher structural modes faster than viscous diffusion
removes it, as `ν → 0` (`Re → ∞`)?

## 3. Field dictionary

| Navier–Stokes | TNFR |
|---|---|
| velocity `u_a` | per-component phase field `φ^(a)` |
| vorticity `ω = ∇×u` | `K_φ` per component |
| pressure `p` | `Φ_s` (Leray/incompressibility multiplier) |
| viscosity `ν` | `ν_f` (diffusive-face structural frequency) |
| enstrophy `‖ω‖²` | `Σ K_φ²` (conserved-pressure energy) |
| stretching `(ω·∇)u` | VAL nonlinear destabiliser (the conservative source) |

## 4. The blow-up frontier (measured)

`conservative_face.measure_cascade_frontier` evolves the faithful pseudo-spectral
Taylor–Green vortex at several viscosities to a matched **structural time**
`τ_str = ν·t` and records the peak enstrophy debt `Ω_peak/Ω₀`, the peak stretching
production and the high-mode enstrophy fraction versus `Re = 2π/ν`.

- **Fixed Re:** every run's enstrophy peaks and decays — the diffusive face
  regularises, the debt is bounded (known finite-Re regularity, re-expressed).
- **Sweep:** the peak debt **grows with Re**. Whether it stays finite as
  `Re → ∞` (regularity) or diverges (blow-up) is exactly Clay, now phrased as
  *"is the nonlinear `K_φ` cascade uniformly bounded in Re?"*.

This is the honest analogue of the Riemann coherence-budget measurement: a
per-instance bound that is finite at every finite parameter, with the **uniform
bound over the limiting parameter** left open.

## 5. Cross-program unification

All three active programs now read on the same two-face machinery:

- **Riemann** — the conservative **pulse** `ω_k = √λ_k`; RH content = the
  coherence budget of `S(T)`, bounded at the RMS level, sup open.
- **EEG (empirical arm)** — real brain rhythms sit on the **conservative
  (under-damped) face**; the local phase tetrad carries clinical state.
- **Navier–Stokes** — linear part on the **diffusive (over-damped) face**; the
  conservative content is the **nonlinear** cascade, bounded at fixed Re, the
  `Re → ∞` bound open.

The face a system sits on is not assumed — it is **measured** by the engine's
`verify_overdamped_projection` certificate.

## 6. The λ-moment hierarchy — the synergy the old paradigm blocked

The old diffusive-face program saw only the **scalar** enstrophy budget. The
emergent modal basis (L_rw modes `λ_k`) unlocks the whole **λ-moment hierarchy**
`M_p = Σ λ_k^p E(λ_k)` (`cascade_moment_hierarchy`):

- `M_0` = energy — the **conservative-face budget**, bounded by Leray
  (`M_0(t) ≤ M_0(0)`);
- `M_1` = enstrophy — the classical blow-up quantity;
- `M_2` = palinstrophy — weights the small-scale (high-`λ`) tail more.

**Measured** (Taylor-Green at peak enstrophy, resolved points `k_max·η > 1`,
Re 157→628, ×4): `M_0` **decreases** (×0.62 — the conservative budget is bounded),
while `M_1` grows (×1.71) and `M_2` grows much faster (×11.3); the moment
**ratios climb** with Re (`M_1/M_0`: 3.0→8.3; `M_2/M_1`: 3.0→19.8). **The wall
climbs the λ-moment hierarchy.** In this basis Clay is exactly: *does the ladder
`M_p` (`p ≥ 1`) stay uniformly bounded as `ν → 0` while `M_0` stays bounded?* — the
canonical modal form of the classical `H^s` / Foias–Temam regularity ladder.
Driver: `benchmarks/ns_moment_hierarchy_cascade.py`.

**Closing the rung (measured).** The enstrophy rung is `dM_1/dt = P − 2ν M_2`
(`moment_ladder_closure`), with the exact modal Cauchy–Schwarz coupling
`M_1² ≤ M_0·M_2` (interpolation saturation `s = M_1²/(M_0 M_2) ∈ (0,1]`). Measured
at peak (resolved Re 314→628): the rung is **self-consistent** — `P/(2ν M_2) ≈ 1`
at the peak (`1.00`, `1.00`, `0.98`), confirming `dM_1/dt = 0` there; the
interpolation saturation `s` **decreases** with Re (`0.57 → 0.42 → 0.38`) — the
spectrum **spreads** across scales rather than concentrating (a
regularity-favourable signal; `s ≤ 1` is exact); and in the growth phase
`P/(2ν M_2) < 1` (`0.70 → 0.59 → 0.45`) — the dissipation dominates, so the ladder
**closes at every accessible resolved Re**. The wall is thus relocated to the
sharp question: *does the growth-phase closure ratio stay `< 1` as `Re → ∞`?* —
undecidable from resolution-limited laminar/transitional data (`n ≥ 48–64` needed
at high Re). Closing the rung uniformly in Re is exactly Clay.

**The cross-program synergy (measured, both walls the same statement).** This is
the NS twin of the Riemann coherence budget:

| | low moment (bounded) | high moment (the open wall) |
|---|---|---|
| **Riemann** | RMS of `S(T)` — Selberg `√(log log T)` | sup of `S(T)` (the extremes) |
| **NS** | energy `M_0` — Leray | enstrophy `M_1`, palinstrophy `M_2`, … |

Both Millennium walls: **a low moment of the conservative spectrum is bounded;
the high-moment tail is the wall.** A resolution *closes the moment ladder*
uniformly in the limiting parameter (`T` for Riemann, `Re` for NS). Closes
nothing; the measurement localises the wall, it does not breach it.

## 7. The emergent geometry IS the attractor (closure with nothing added)

The moment ladder (§6) is a *derived* diagnostic; the closure itself needs
**nothing added**. The emergent geometry is the attractor, read by the ONE
universal coherence kernel — `structural_coherence` `C = 1/(1+|ΔNFR|+|dEPI|)` and
the fixed-point predicate `is_structural_equilibrium` (`ΔNFR = 0`) — the *same*
emergent-geometry attractor that governs graph nodes, structural primes and noble
gases. Only the `ΔNFR` realisation is domain-specific; for NS it is the canonical
random-walk-Laplacian action on the vorticity field, `ΔNFR = −L_rw·|ω|`
(`flow_coherence`).

**Measured (raw field, no normalisation — nothing added):** the flow **relaxes to
its emergent-geometry equilibrium by its own evolution** — `C → 1`,
`is_structural_equilibrium = True` at every Re (final `C = 0.995 → 0.9985` over
Re 157→1257). That relaxation *is* the self-certification; it is intrinsic to the
diffusive-face nodal dynamics (the unconditional eigenmode decay to the uniform
field) and is why the linear part is regular. The **peak-turbulence coherence
erodes with Re** — `min C = 0.94 → 0.90 → 0.84 → 0.72` — staying in the coherent
band `[1/(π+1), π/(π+1)]` over the accessible range but drifting down.

**So the uniform closure is purely geometric, with nothing added:** does the
peak-turbulence coherence `min C` stay in the coherent band (`C > 1/(π+1)`) as
`Re → ∞`, or erode to the fragmentation floor? The emergent geometry self-certifies
its *return* to coherence uniformly (in structural time the diffusive relaxation
rate is the spectral gap `λ₂`, Re-independent); the wall is whether the *peak
excursion* the nonlinear VAL source drives stays coherent — the U2 debt rate
(∝ Re) versus the fixed U2 capacity, now read as the erosion of the single
canonical coherence `C`. Every TNFR mechanism (U2 grammar, the `ΔNFR=0` attractor,
the Lyapunov energy, the U5 multi-scale recursion, REMESH-∞) converges on this one
statement — none weakens it. Closes nothing; Clay OPEN.

## 8. Honest scope

This program does **not** claim a proof or a counterexample. The linear diffusive
face is regular by construction; the open question is the **nonlinear** cascade
bound as `Re → ∞`. No uniform-in-Re bound is produced. **Clay stays open.** The
Reynolds/Kolmogorov resolution caveat of the old program persists: high-Re points
need `n ≥ 48–64`; the measured trend over the accessible resolved range cannot fix
the asymptotic scaling.

## 9. Code map

- `src/tnfr/navier_stokes/operator.py` — faithful pseudo-spectral 3D NS
  integrator (`TNFRNavierStokes`) + `build_torus_graph_3d` +
  `taylor_green_initial_condition_3d`.
- `src/tnfr/navier_stokes/conservative_face.py` — `verify_diffusive_face`,
  `face_of_flow`, `vorticity_modal_spectrum`, `cascade_moment_hierarchy`,
  `moment_ladder_closure`, `flow_coherence` (the universal-kernel attractor read),
  `measure_cascade_frontier`, `CascadeFrontierCertificate`.
- `examples/06_navier_stokes/158_navier_stokes_two_face_refounded.py` — demo.
- `examples/10_applications/159_empirical_confrontation_pipeline.py` — the
  data-confrontation pipeline (map a signal to canonical magnitudes + face).
- `benchmarks/ns_moment_hierarchy_cascade.py` — the λ-moment hierarchy vs Re.
- Tests: `tests/mathematics/test_navier_stokes_refounded.py`.

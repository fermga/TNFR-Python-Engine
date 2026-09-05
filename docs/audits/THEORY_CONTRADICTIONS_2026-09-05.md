# Mathematical and implementation consistency audit

Date: 2026-09-05. Baseline commit: `47603cff1ee922df21cbf783aec74a3a8b8acbc5`.

Counterexample outputs and line references below describe the first-pass
baseline. The [second audit](SECOND_REPOSITORY_AUDIT_2026-09-05.md) corrects
the implementation defects T01, T03, T04 and T09 and scopes unsupported
certificate claims. T02 and the stated proof gaps remain unresolved; those
software corrections do not establish the missing mathematical bridges.

The subsequent [resolution pass](RESOLUTION_REMAINING_CONTRADICTIONS_2026-09-05.md)
implements the exact restricted EPI Dirichlet balance and reconciles T07, T08,
T10 and T11 across the canonical guides. Universal confinement, full nodal
variational correspondence, grammar sufficiency and state reconstruction remain
unproved. The table and counterexamples below preserve this audit's historical
findings; the linked resolution records the current implementation and scope.

This audit compares the nodal equation, canonical field definitions, documented
derivations, and executable diagnostics. It distinguishes reproducible software
defects from unsupported generalizations. The counterexamples below use finite
graphs and explicit states; they do not address the open Riemann or
Navier–Stokes programs. `manual/` was excluded.

The bounded corrections made during this audit preserve operator contracts and
grammar rules. The unresolved findings identify the additional assumptions or
mathematical work needed before stronger claims can be certified. Existing
telemetry thresholds remain operationally usable even where their claimed
universal derivation does not follow from the stated equations.

## Findings

| ID | Priority | Finding | Disposition |
| --- | --- | --- | --- |
| T01 | P1 | Single-state products are used to certify symplectic maps | Fixed in second audit; snapshots inconclusive, optional Jacobian checked |
| T02 | P1 | The stated potential gradient does not reproduce the EPI diffusion channel | Reproduced contradiction; model bridge unresolved |
| T03 | P1 | Zero-energy symplectic reduction receives the regular-level dimension | Fixed in second audit; singular point and regular tangent distinguished |
| T04 | P1 | Mean frequency is substituted for heterogeneous nodal mobility | Fixed in second audit; actual nodal generator used |
| T05 | P2 | Duplicate Laplacians assign different dynamics to isolated nodes | Fixed and tested |
| T06 | P2 | SDK coherence length changes under a pure clock rescaling | Fixed and tested |
| T07 | P2 | Potential confinement is described as a universal phase-derived bound | Counterexample; retain thresholds as policy pending proof |
| T08 | P2 | Mean spectral relaxation is presented as an exact perturbation window | Counterexample; retain grammar policy pending proof |
| T09 | P2 | Diffusion certificates assume one connected equilibrium | Fixed in second audit; component/frozen stationarity and true invariants checked |
| T10 | P2 | Boundedness and improper-integral convergence are conflated | Derivation needs explicit hypotheses |
| T11 | P2 | Completeness of the tetrad and coherence scaling are overstated | Completeness proof gap; scaling sentence corrected |

### T01 — The symplectic diagnostic rejects rotations and accepts reflections

References: `src/tnfr/physics/variational.py:578` (`compute_phase_space_volume`),
`:600` (`compute_poisson_bracket_estimate`), and `:796`
(`check_symplectic_preservation`); `AGENTS.md:223`;
`src/tnfr/physics/symplectic_substrate.py:75`.

The diagnostic uses `sum(abs(q_i*p_i))` as symplectic volume and compares its
value before and after a transformation. This is a scalar function of one
state, not the area of a transported neighbourhood or the pullback of a 2-form.

Two exact one-pair counterexamples were executed, using the same pair in both
sectors of a `LagrangianSnapshot`:

| Transformation | Mathematical result | Current diagnostic |
| --- | --- | --- |
| Rotation by pi/4: `(1, 0) -> (1/sqrt(2), -1/sqrt(2))` | Symplectic: `M.T @ J @ M = J` | `is_canonical=False`, `classification='expansive'`, ratio infinity |
| Reflection: `(1, 1) -> (1, -1)` | Anti-symplectic: `M.T @ J @ M = -J` | `is_canonical=True`, `classification='canonical'`, ratio 1 |

The covariance determinant used by `compute_poisson_bracket_estimate` also
depends on the sampled state distribution. A canonical bracket is defined by a
Poisson tensor and derivatives of functions; it cannot be inferred from the
spread of one set of node values.

**Correction required:** relabel these legacy outputs as heuristic field
statistics, or provide a map/Jacobian and verify `D(F).T @ J @ D(F) = J`.
Dimension-changing operators require a specified map between the relevant
spaces. The exact symplecticity of `evolve_substrate_flow` does not certify the
13 nonlinear engine operators. The module's own operator table at
`variational.py:1160` already classifies Coherence and Contraction as
dissipative, contradicting the blanket claim.

### T02 — The variational potential and the nodal pressure are different gradients

References: `src/tnfr/physics/variational.py:38` and `:365`
(`compute_potential_density`); `src/tnfr/physics/symplectic_substrate.py:113`
and `:1241` (`evolve_substrate_flow`);
`src/tnfr/physics/structural_diffusion.py:942`
(`verify_overdamped_projection`).

The documented identity is `DeltaNFR = -dV/dEPI`. Consider one edge, uniform
phase, uniform `nu_f=1`, and the pure EPI channel. Let

```text
L = [[1, -1], [-1, 1]],  x = [1, 0]
DeltaNFR = -Lx = [-1, 1]
Phi_s = [1, -1],  grad_phi = K_phi = 0
V(x) = 0.5*||Phi_s||^2 = (x_0-x_1)^2
-gradient(V) = [-2, 2] != DeltaNFR
```

Finite differences of the actual `compute_potential_density` implementation
with step `1e-6` returned `[-2.000000000002, 2.000000000002]`; the EPI-channel
pressure remained `[-1, 1]`.

There is a second mismatch in the claimed bridge: the substrate implementation
evolves every pair with `q''=-q`, whereas the projection certificate constructs
the different graph-wave equation `q'' + gamma*q' + L*q=0`. For a constant
nonzero graph field, `L*q=0`; the graph wave can remain at rest, while the
isotropic substrate oscillator accelerates. The graph-wave-to-diffusion limit
is a valid calculation for the graph wave, but does not establish that it is
the projection of the implemented isotropic substrate.

**Correction required:** specify the coordinate map, metric, Hamiltonian, and
dissipation law that produce the claimed projection. For the undirected EPI
channel alone, the degree-metric gradient of the Dirichlet energy gives
`-L_rw*x`; this does not prove the claimed identity for the different tetrad
potential or the full four-channel pressure. Keep the harmonic-substrate
certificates scoped to their implemented Hamiltonian.

### T03 — Singular reduction levels are certified as regular symplectic quotients

References: `src/tnfr/physics/symplectic_substrate.py:678`, `:1874`, and `:1921`.

On a two-node path with every phase, EPI, frequency, and pressure zero,
`verify_symplectic_reduction` reports:

```text
moment_map_value = 0.0
phase_space_dimension = 8
reduced_dimension = 6
reduced_form_nondegenerate = True
reduced_form_determinant = 15.999999999999991
```

For the stated moment map `J(z)=0.5*||z||^2`, however, `J^-1(0)={0}`. Its quotient
by U(1) is a point, with dimension zero. The U(1) action is not free there and
zero is not a regular value. A nondegenerate matrix constructed only from
`n_nodes` cannot certify this level set.

The docstring also describes the positive-energy quotient as a globally flat
linear space. For `m=2N` complex coordinates and `mu>0`, the level set is a
sphere `S^(2m-1)`; identifying its common phase yields complex projective space
`CP^(m-1)`. For `N=1` this is a sphere, not a linear plane. Local action-angle
charts do not establish global flatness and fail at zero actions.

**Correction required:** distinguish zero/singular levels, regular positive
levels, and local coordinate-chart certificates. Do not infer global topology
or regularity from the determinant of a constant coordinate matrix.

### T04 — Heterogeneous frequency requires a different diffusion generator

References: `src/tnfr/physics/structural_diffusion.py:380`, `:391`, `:415`,
and `:575`; `src/tnfr/sdk/simple.py:1485`.

With per-node frequency, the nodal equation gives
`x'=-diag(nu_f)*L_rw*x`, rather than `-mean(nu_f)*L_rw*x`. On the path with
three nodes and `nu_f=[1,3,5]`, executable eigenvalue calculations give:

```text
eigenvalues(diag(nu_f)*L_rw) = [0, 2, 7]
relaxation_spectrum(G)       = [0, 3, 6]
```

Degree-weighted mass also requires the scalar-frequency restriction. On one
edge with `x=[1,0]` and `nu_f=[1,3]`, the actual derivative is `[-1,3]`, so the
degree-weighted total has derivative `2`, not zero. For fixed positive
heterogeneous frequencies on an undirected graph, the conserved weight is
`degree_i/nu_f_i`.

`verify_structural_diffusion` reports a diffusivity but integrates
`e <- e-dt*L_rw*e`, without either the reported scalar or nodal frequencies.
Consequently this certificate establishes a unit-frequency model's behavior.

**Correction required:** explicitly identify the current readout as a
homogeneous-frequency approximation, or calculate and certify the actual
heterogeneous generator and its invariant measure. The latter changes model
semantics and is deferred; the bounded SDK correction in T06 leaves the
existing relaxation-rate API intact.

### T05 — Isolated nodes acquired artificial eigenmodes through duplicate code

References: `src/tnfr/physics/structural_diffusion.py:283`, `:320`, and `:1178`;
`tests/physics/test_structural_diffusion.py`, `TestDiscreteModes`.

Before correction, the public normalized Laplacian gave isolated nodes zero
rows, consistent with no neighbours and no EPI diffusion. Its private duplicate
used `I-D^(-1/2)*W*D^(-1/2)` without clearing the isolated diagonal.

For one edge plus an isolated node, `L_rw` had eigenvalues `[0,0,2]`, but
`structural_eigenmodes` returned `[0,1,2]`. Three isolated nodes received
positive collective vibration energy despite having no coupling edges.

**Implemented:** the private compatibility wrapper delegates to the public
normalized Laplacian. Regressions cover a weighted edge plus an isolated node,
the eigenvector equation, matching diffusion/relaxation spectra, and zero
collective vibration on an edgeless graph.

### T06 — SDK spatial coherence length depended on the frequency clock

References: `src/tnfr/sdk/simple.py:1485`;
`src/tnfr/physics/canonical.py`, `_spectral_gap_coherence_length`;
`AGENTS.md:170`.

`Network.spectrum()` computed `1/sqrt(nu_f*lambda_2)`, while the documented
geometric scale and canonical fallback use `1/sqrt(lambda_2)`. Thus a uniform
frequency rescaling from 1 to 4 halved a purported spatial length with no
change to graph, phase, or pressure. Setting the frequency to zero made the
same connected graph's length infinite.

**Implemented:** coherence length now uses the unscaled cached graph spectrum.
The existing `spectral_gap` and `relaxation_rates` retain their temporal-rate
semantics for compatibility. Four analytic cycle-graph tests cover
`nu_f=0, 0.25, 1, 4`; rates scale with the clock and geometry does not.

### T07 — Phase wrapping alone does not bound structural potential

References: `AGENTS.md:172`; `theory/MINIMAL_STRUCTURAL_DEGREES.md:81–86`;
`src/tnfr/physics/canonical.py`, `compute_structural_potential`;
`src/tnfr/constants/canonical.py:305–342`.

The definition `Phi_s(i)=sum(DeltaNFR_j/d(i,j)^2)` is linear in pressure and
contains no phase wrap. Set every phase to zero and every pressure to 1 on
`K_4`. The actual function returns `{0:3, 1:3, 2:3, 3:3}`, exceeding both
pi-fraction thresholds. On `K_n` the result is `n-1`; scaling the pressure by
`a` scales the potential by `a`. Changing pressure from zero to one also
produces drift `n-1`. These are permissible field inputs, though a U6 monitor
can subsequently flag the resulting state.

The cited chain-series argument does not uniquely select exponent 2:
`sum(d^-alpha)` converges for every `alpha>1`; the independent-pressure
variance sum `sum(d^(-2*alpha))` converges for every `alpha>1/2`. On graph
families with increasing neighbourhood volume, even those chain conclusions
need not hold.

**Correction required:** distinguish selected safety thresholds from universal
upper bounds. A derived bound needs explicit pressure, topology, normalization,
and evolution assumptions connecting the pressure sector to phase. The fact
that both constants are fractions of pi is insufficient. The true geometric
phase-wrap bounds `|grad_phi|<=pi` and `|K_phi|<=pi` are unaffected.

### T08 — An average eigenvalue does not determine each perturbation's decay

References: `src/tnfr/config/physics_derivation.py:55–152`;
`theory/UNIFIED_GRAMMAR_RULES.md:111–120`.

The recency-window derivation replaces every mode's multiplier
`1-nu_f*dt*lambda_k` with `q=1-nu_f*dt*mean(lambda)`. On a loopless graph
without isolates, `trace(L_rw)/N=1` is correct; it does not imply that an
arbitrary pressure perturbation decays at this rate.

On a 21-node path with `nu_f=1`, `dt=0.5`, the Fiedler pressure mode gives:

```text
lambda_2                         = 0.01231165940486257
derive_bifurcation_window(...)    = 3
(1-0.5*lambda_2)^3               = 0.981645960340198
target fraction 1/(pi+1)         = 0.24145300700522387
actual steps to that fraction   = 231
```

Isolates and self-loops additionally invalidate the unconditional trace/N=1
premise. The `q<=0 -> one-step removal` branch does not describe general
Euler stability either: a negative modal multiplier can oscillate or grow.

**Correction required:** document the fixed three-operation window and
two-operation debt as grammar policies calibrated to a mean-rate surrogate,
or derive trajectory bounds from the relevant modes, step stability, and
operator gains. Their existing values are not changed by this audit.

### T09 — Multiple connected components have multiple stationary modes

References: `AGENTS.md:105–109`;
`src/tnfr/physics/structural_diffusion.py:575–654`.

On two disconnected edges with EPI `[0,0,1,1]`, equal frequency, and equal
phase, `L_rw*EPI=0` exactly. Nevertheless `verify_structural_diffusion` reports
`relaxes_to_uniform=False` and `final_field_std=0.5`. The field already is an
equilibrium on every component; its global standard deviation need not vanish.

**Correction required:** restrict global-uniformity claims and certificates to
connected graphs, or assess equilibrium component by component. Distinguish
the second-smallest eigenvalue from the smallest positive eigenvalue on
disconnected graphs. Uniform-frequency and pure-EPI-channel assumptions are
also necessary for the current scalar-rate formulas.

### T10 — The grammar derivations require stronger analytic hypotheses

References: `AGENTS.md:117–123` and `:344`;
`theory/UNIFIED_GRAMMAR_RULES.md:28–41` and `:75–95`.

The nodal identity alone does not imply the cited assertions:

- `EPI=0` does not make its derivative undefined: with `nu_f=1` and finite
  `DeltaNFR=1`, it gives `dEPI/dt=1`, including at EPI zero. A generator
  requirement can be an initialization contract; it needs that extra premise.
- Bounded evolution need not converge: `EPI(t)=sin(t)`, `nu_f=1`,
  `DeltaNFR=cos(t)` satisfies the equation and stays bounded, but its integral
  from zero has no limit as the upper endpoint tends to infinity.
- A bounded pressure tending to zero need not have a convergent integral:
  `DeltaNFR(t)=1/(1+t)`, `nu_f=1` gives logarithmic growth.
- Absence of named stabilizers does not imply growing pressure: the autonomous
  diffusion equation itself supplies negative feedback and relaxes without
  additional operator applications.

**Correction required:** distinguish finite-horizon existence, boundedness,
convergence to a limit, and absolute integrability. State the feedback/gain
assumptions under which grammar compliance controls those properties. These
counterexamples concern the purported derivation from the bare nodal equation;
they do not authorize bypassing established operator contracts.

### T11 — Diagnostic usefulness does not prove completeness of state information

References: `theory/MINIMAL_STRUCTURAL_DEGREES.md:9–17`, `:47–54`, `:99`,
and `:141`; `AGENTS.md:197`.

The statement that higher operators are compositions of the gradient and
Laplacian does not prove that every scalar diagnostic is a function of four
reported quantities. Algebraic generation of operators and sufficiency of
lossy field summaries are different claims. On a graph with at least four
distinct Laplacian eigenvalues, `I, L, L^2, L^3` are linearly independent even
though all are generated by powers of L.

The tetrad also does not determine the full nodal dynamics: under uniform
frequency rescaling, the phase fields and pure-EPI-channel pressure can remain
unchanged while `dEPI/dt` scales with the frequency. A completeness theorem
must specify which state variables, gauge identifications, observables, and
equivalence relation it includes. The four selected telemetry channels can
remain useful without establishing that no fifth independent observable exists.

A separate direct contradiction was found in the same document: line 99 correctly
distinguishes canonical coherence from its scale-invariant dispersion variant,
but the former line 141 said doubling pressure leaves `C(t)` unchanged. For `dEPI=0` and
uniform `|DeltaNFR|=1`, canonical coherence is `1/2`; doubling pressure makes
it `1/3`. The scale-invariance sentence applies to the dispersion variant only.
That sentence now explicitly distinguishes the two kernels and includes this
counterexample. The independent completeness proof gap remains unresolved.

## Validation and scope

Focused baselines passed before editing: structural diffusion 69 tests; advanced
SDK 76 tests. After the two bounded corrections, the combined suites passed
151 tests, including six new regression cases. This does not validate the
unresolved mathematical claims in T01–T04 and T07–T11: those need the specific
counterexamples or stronger certificates described above, not a larger count
of tests of the existing surrogate definitions.

All diagnostic graph fixtures were deterministic. The mathematical examples use
explicit initial EPI, pressure, phase, frequency, topology, and steps where
relevant; no stochastic operator sequence was used. The audit does not claim a
before/after improvement in C(t), because both implemented changes correct
read-only spectral telemetry rather than changing nodal evolution.

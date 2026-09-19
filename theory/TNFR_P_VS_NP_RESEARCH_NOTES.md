# TNFR–P vs NP Structural Synthesis Research Notes

**Status**: Auxiliary finite MAX-CUT diagnostic implemented; TNFR dynamics bridge and complexity claims remain open
**Date**: 2026-06-13
**Scope review**: 2026-09-18; documentation correction, no new numerical run
**Scope**: Synthesis-versus-verification comparison; **not** a nodal derivation of the implemented optimizer or a proof about P versus NP
**Primary anchors**: nodal equation `∂EPI/∂t = νf · ΔNFR(t)`, the restricted pure-EPI Dirichlet/mobility identity, and the separately implemented antiphase example

---

## 0. Terminology Discipline

The comparison concerns pattern synthesis and verification. The nodal equation is

$$
\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t).
$$

It does not imply that every canonical pressure is the negative gradient of
the tetrad functional `V = ½Σ(Φ_s² + |∇φ|² + K_φ²)`.
`src/tnfr/physics/variational.py` explicitly records a counterexample: on one
pure-EPI edge at `[1,0]`, pressure is `[-1,1]`, while `−∇V=[-2,2]`.

A restricted identity is available for a fixed symmetric nonnegative
conductance `W`, fixed nonnegative capacities and the isolated EPI channel.
With `D=diag(W1)`, `E_D=½xᵀ(D−W)x` and
`M=diag(νf_i/d_i)` (zero at isolates),
`ẋ=−M∇E_D` and `dE_D/dt=−∇E_DᵀM∇E_D≤0`.
This Dirichlet functional is distinct from the tetrad potential; see
`src/tnfr/physics/structural_diffusion.py` and
`theory/TNFR_VARIATIONAL_PRINCIPLE.md`. It does not derive the antiphase
optimizer used below or a general descent theorem for the operator catalog.

References to P vs NP are treated as an **external comparison target**. The
TNFR object is a nodal structural question: *is synthesising a globally
coherent configuration fundamentally harder than verifying one?* No claim in
this document should be read as a solution of the Clay Millennium Problem. The
Clay problem concerns worst-case separation of the complexity classes P and NP
for general decision problems; nothing here establishes or refutes that
separation.

---

## 1. Existing Canonical Base in the Repository

These owners provide comparison tools. The example's direct phase updates
have not been identified with a canonical operator sequence.

| Component | Existing source | Role |
| --- | --- | --- |
| Restricted EPI gradient flow | `src/tnfr/physics/structural_diffusion.py` | Dirichlet energy and mobility under the stated fixed-graph assumptions |
| Tetrad energy candidate | `src/tnfr/physics/variational.py` | Snapshot functional; no general pressure-gradient identity |
| Canonical phase pressure | `src/tnfr/dynamics/dnfr.py` | Wrapped displacement from a neighbour-phasor mean; not the example's sine-sum update |
| Dissonance operator OZ | `src/tnfr/operators/dissonance.py` | Existing pressure operator; not executed by PNP-1 |
| Coherence and other diagnostics | `src/tnfr/metrics/` | Possible read-outs; costs and meaning require their own specified input |

---

## 2. Synthesis-versus-verification comparison

The finite example compares two tasks on a supplied graph:

- **Verification.** Given a binary assignment and a requested cut threshold,
  count crossing edges in `O(|E|)`. This checks a witness for that threshold;
  it does not verify global optimality or equal the engine's coherence score.
- **Synthesis.** Run the specified continuous antiphase relaxation and round
  its endpoint to a binary assignment. Its finite success rate measures this
  heuristic, not all algorithms or canonical TNFR histories.

> **PNP-1**: On a family of frustrated instances of growing size, does bare
> antiphase relaxation miss the optimum — hit rate of the
> global optimum dropping, required restarts growing — while verification
> stays `O(|E|)`?

> **PNP-2** (open): Does the **full** canonical operator catalog (OZ-controlled
> dissonance, ZHIR mutation, THOL re-organization, REMESH cross-scale echo)
> collapse the trapping to polynomial-cost synthesis, or do the dissonance
> basins remain exponentially many?

PNP-2 is an unimplemented comparison proposal, not a complexity-theoretic
equivalence. Any canonical version first needs a faithful encoding, operator
semantics, precision/resource accounting and explicit worst-case quantifiers.

---

## 3. Auxiliary encoding (MAX-CUT as antiphase coupling)

Each node carries a phase `θ`. Every edge demands antiphase (a cut). The
relaxation

$$
\frac{d\theta_i}{dt} = \sum_{j \sim i} \sin(\theta_i - \theta_j)
$$

is the negative gradient of the chosen energy
`E = Σ_(i,j) cos(θ_i−θ_j)`. It is an auxiliary sine-coupled flow, not the
implemented canonical phasor-mean pressure or an OZ execution.
For `θ ∈ {0, π}^n`, the frustration energy satisfies
`E = |edges| - 2*cut_size`; minimizing it is equivalent to **MAX-CUT**
on the graph (an NP-hard objective). Frustration arises on odd cycles, which
cannot satisfy all antiphase demands simultaneously — the structural origin of
the frustrated optimization landscape. The binary identity does not identify
minima of the continuous relaxation with optimal rounded cuts. Finite Euler
steps also need a separate descent/convergence check.

---

## 4. PNP-1 Result (DONE)

Historically reported on random 3-regular graphs, 3 instances
per size, `R = 200` random initial conditions each; exact MAX-CUT by
enumeration; relaxation 400 steps, `dt = 0.1`. Reproducible in
`examples/09_millennium/109_p_vs_np_coherence_synthesis.py`.

| n | \|E\| | global reachable | hit rate | restarts ≈ 1/hr |
| ---: | ---: | :---: | ---: | ---: |
| 8  | 12 | yes | 0.737 | 1.36 |
| 10 | 15 | yes | 0.648 | 1.54 |
| 12 | 18 | yes | 0.642 | 1.56 |
| 14 | 21 | yes | 0.558 | 1.79 |
| 16 | 24 | yes | 0.422 | 2.37 |
| 18 | 27 | yes | 0.412 | 2.43 |

- Hit-rate trend slope `d(hit_rate)/dn = −0.0341` per node; **monotone
  decreasing** across all sizes.
- "global reachable = yes" means at least one sampled rounded endpoint reaches
  the independently enumerated optimum at every reported size. It is not a
  canonical TNFR reachability certificate.

**PNP-1 verdict**: finite hit rates decrease for this heuristic and sample.
The example does not check stationarity, basin membership or convergence;
rounding and the fixed iteration budget can also affect failures. It therefore
does not certify local-optimum trapping, exponential restart growth, or a
complexity separation. The exact optimum used for scoring is obtained by
exponential enumeration, not predicted by nodal dynamics.

---

## 5. Honest Obstruction Classification

Using the same A/B trichotomy as the other TNFR Millennium programs:

- **Branch A** (closure inside the existing catalog) — *not established*. PNP-1
  measures one auxiliary heuristic and has no proved catalog realization.
- **Branch B** (open; current classification) — a canonical encoding and the
  effect of the **full** catalog remain unimplemented. No basin-count theorem,
  polynomial synthesis algorithm or worst-case lower bound is supplied.
- **Branch B3** (no TNFR closure) — not decidable from PNP-1.

Other repository programmes also separate finite diagnostics from open
theorems. That methodological comparison does not identify their mathematical
obstructions or make one programme's evidence establish another's claim.

---

## 6. Milestone Roadmap

| PNP | Title | Status |
| --- | --- | --- |
| PNP-1 | Finite antiphase MAX-CUT hit-rate diagnostic | **IMPLEMENTED** (`examples/109`); trapping and canonical bridge unproved |
| PNP-2 | Full-catalog escape (OZ/ZHIR/THOL/REMESH): does trapping collapse to polynomial? | open |
| PNP-3 | Basin-count scaling: are local optima exponentially many under U1–U6? | open |
| PNP-4 | Encoding generality: SAT / graph-colouring beyond MAX-CUT | open |
| PNP-5 | Worst-case separation (Clay-hard boundary) | open, **not assumed** |

PNP-5 is not claimed. The table is a historical comparison inventory, not an
active parallel queue; the [single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md)
owns priorities.

---

## 7. What This Program Does and Does Not Do

**Does**: supply an auxiliary finite synthesis-versus-verification diagnostic
with an independently enumerated scoring oracle, declared graph family and
finite resource budget. Its evaluation pattern can inform later canonical
pattern-formation experiments after deriving their dynamics.

**Does not**: prove or disprove `P = NP`; claim that TNFR relaxation is an
efficient general solver; establish trapping or exponentially many basins;
derive the sine-sum flow from the canonical phase channel; or assume that
canonical operators solve the encoded optimization problem.

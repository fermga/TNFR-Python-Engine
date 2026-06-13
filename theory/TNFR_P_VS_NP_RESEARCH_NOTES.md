# TNFR–P vs NP Structural Synthesis Research Notes

**Status**: Pre-registered research programme; PNP-1 diagnostic implemented; obstruction classified as Branch B (open)
**Date**: 2026-06-13
**Scope**: TNFR-internal structural synthesis-vs-verification dynamics; **not** a proof or disproof of the Clay P vs NP problem
**Primary anchors**: nodal equation `∂EPI/∂t = νf · ΔNFR(t)` as gradient flow `ΔNFR = −∂V/∂EPI`, canonical operators, grammar U1–U6, structural field tetrad `(Φ_s, |∇φ|, K_φ, ξ_C)`

---

## 0. Terminology Discipline

This programme is formulated in TNFR language only. TNFR does not introduce a
separate "computation" ontology. The same nodal equation,

$$
\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t),
\qquad \Delta\mathrm{NFR} = -\frac{\partial V}{\partial \mathrm{EPI}},
$$

is a **gradient flow** on the structural potential `V` (established in
`src/tnfr/physics/variational.py`). Coherence relaxation descends `V`.

References to P vs NP are treated as an **external comparison target**. The
TNFR object is a nodal structural question: *is synthesising a globally
coherent configuration fundamentally harder than verifying one?* No claim in
this document should be read as a solution of the Clay Millennium Problem. The
Clay problem concerns worst-case separation of the complexity classes P and NP
for general decision problems; nothing here establishes or refutes that
separation.

---

## 1. Existing Canonical Base in the Repository

The programme starts from already-shipped TNFR machinery; it requires **no new
canonical operator**.

| Component | Existing source | Role |
| --- | --- | --- |
| Nodal equation as gradient flow | `src/tnfr/physics/variational.py` | `ΔNFR = −∂V/∂EPI`; relaxation descends `V` |
| Structural potential | `src/tnfr/physics/variational.py` | `V = ½[Φ_s² + |∇φ|² + K_φ²]` |
| Phase channel (Kuramoto coupling) | `src/tnfr/dynamics/dnfr.py`, `src/tnfr/physics/structural_diffusion.py` | circular neighbour coupling; sign sets align/antialign |
| Dissonance operator OZ | `src/tnfr/operators/dissonance.py` | controlled instability; basin-escape move (PNP-2) |
| Coherence C(t), Sense Index Si | `src/tnfr/metrics/` | polynomial-cost verification telemetry |

---

## 2. TNFR-Native Reformulation

P vs NP, read through the nodal gradient flow, is the asymmetry between two
structural tasks on a graph-coupled network:

- **Verification.** Given a configuration, evaluate its coherence (e.g. the
  cut value / frustration energy, or `C(t)`). Cost `O(|E|)` — polynomial. This
  is the TNFR analogue of checking an NP witness.
- **Synthesis.** Find a *globally* coherent configuration by nodal relaxation.
  On a frustrated topology the potential `V` has many local optima =
  **dissonance (OZ) basins**; gradient flow descends to the nearest basin.

> **PNP-1**: On a family of frustrated instances of growing size, does bare
> coherence relaxation (gradient flow) trap in local optima — hit rate of the
> global optimum dropping, required restarts growing — while verification
> stays `O(|E|)`?

> **PNP-2** (open): Does the **full** canonical operator catalog (OZ-controlled
> dissonance, ZHIR mutation, THOL re-organization, REMESH cross-scale echo)
> collapse the trapping to polynomial-cost synthesis, or do the dissonance
> basins remain exponentially many?

PNP-2 is the TNFR-native P-vs-NP boundary and is **not** assumed.

---

## 3. Encoding (MAX-CUT as TNFR antiphase coupling)

Each node carries a phase `θ`. Every edge demands antiphase (a cut). The
relaxation

$$
\frac{d\theta_i}{dt} = \sum_{j \sim i} \sin(\theta_i - \theta_j)
$$

is the canonical TNFR phase channel with the **anti-aligning sign** — an
all-edge dissonance (OZ) demand. The global minimum of the frustration energy
`E = Σ_{(i,j)} cos(θ_i − θ_j)` over `θ ∈ {0, π}^n` is exactly the **MAX-CUT**
of the graph (an NP-hard objective). Frustration arises on odd cycles, which
cannot satisfy all antiphase demands simultaneously — the structural origin of
the local-optima (dissonance-basin) landscape.

---

## 4. PNP-1 Result (DONE)

Measured on random 3-regular graphs (a standard frustrated family), 3 instances
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
- "global reachable = yes" (best over all restarts reaches the exact MAX-CUT at
  every size): the low hit rate is **genuine trapping in local optima**, not an
  encoding failure.

**PNP-1 verdict**: coherence synthesis by bare gradient flow is trapped by
dissonance basins, with trapping increasing in size, while verification stays
`O(|E|)`. The synthesis-vs-verification asymmetry is the TNFR-native reflection
of the P-vs-NP asymmetry.

---

## 5. Honest Obstruction Classification

Using the same A/B trichotomy as the other TNFR Millennium programs:

- **Branch A** (closure inside the existing catalog) — *not established*. PNP-1
  only measures bare gradient flow, which is one descent strategy.
- **Branch B** (open; current classification) — the trapping is real, but
  whether the **full** canonical catalog (OZ/ZHIR/THOL/REMESH escape moves)
  synthesizes in polynomial time, or whether the dissonance basins are
  exponentially many, is **open** (PNP-2). The honest expectation — frustrated
  landscapes have exponentially many local optima — reflects `P ≠ NP` but is
  **unproven** here.
- **Branch B3** (no TNFR closure) — not decidable from PNP-1.

This obstruction is structurally analogous to:
- the **Riemann** residual `S(T)` (oscillatory half of the admissible rescaling, RH-equivalent);
- the **Navier–Stokes** production residual (cascade at scale → 0, Clay-open);
- the **Yang–Mills** continuum-limit gap (YMG-5, Clay-open).

In each case TNFR reformulates the problem and **localises the obstruction**
precisely, without closing it.

---

## 6. Milestone Roadmap

| PNP | Title | Status |
| --- | --- | --- |
| PNP-1 | Synthesis-vs-verification trapping on frustrated MAX-CUT | **DONE** (`examples/109`) |
| PNP-2 | Full-catalog escape (OZ/ZHIR/THOL/REMESH): does trapping collapse to polynomial? | open |
| PNP-3 | Basin-count scaling: are local optima exponentially many under U1–U6? | open |
| PNP-4 | Encoding generality: SAT / graph-colouring beyond MAX-CUT | open |
| PNP-5 | Worst-case separation (Clay-hard boundary) | open, **not assumed** |

PNP-5 is the Clay-strength statement and is not claimed.

---

## 7. What This Program Does and Does Not Do

**Does**: provide a TNFR-native reformulation of P vs NP as coherence
synthesis vs verification; supply a reproducible diagnostic (PNP-1) showing the
trapping signature; classify the obstruction honestly (Branch B, open).

**Does not**: prove or disprove `P = NP`; claim that TNFR relaxation is an
efficient general solver; assume the full operator catalog escapes the traps.
The program follows the disciplined pattern of the Riemann, Navier–Stokes, and
Yang–Mills programs: reformulate, measure, localise the obstruction, and remain
honest about the open boundary.

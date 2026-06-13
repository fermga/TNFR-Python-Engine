# Nucleus B — Equivariance Obstructions to Spectral RH Approaches on Prime-Path Graphs

**Status**: Organisation plan for potential external publication. NOT a finished paper.
**Date**: May 27, 2026.
**Authority**: Subordinate to [TNFR_RIEMANN_RESEARCH_NOTES.md §13sexagesima-{tertia..novena}](TNFR_RIEMANN_RESEARCH_NOTES.md).

---

## 1. Thesis (one sentence)

> *On the prime-path graph $G_{P14}$ and a wide class of canonical extensions, every operator built by elementary categorical graph operations and graph-uniform parameter rules commutes with the natural $S_n$ prime-relabelling action, and therefore cannot encode the prime-specific information ($\log p_i$) required to reach the oscillatory residue $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$ that is RH-equivalent.*

This is a **structural no-go theorem** for a specific class of constructions. It does **not** say RH is false, and it does **not** rule out spectral approaches in general (e.g. Connes' adelic construction lives outside this class). It says: *within this well-defined class, the route is closed*.

---

## 2. Why this might be new (subject to bibliographic audit)

The class of constructions covered is well-known to mainstream literature (graph automorphism + parameter-uniform spectral graph theory), and individual instances of the obstruction have certainly been observed informally. What seems to **not** be in the literature is:

1. **A unified statement** spanning edge-channel composition, spectral-channel tensor lifts, augmented graphs with $S_n$-invariant weight laws, line graphs, principal $S^1$-bundles, and Kronecker products — all reduced to two source-auditable facts (parameter uniformity + Prime-Cancellation Lemma).
2. **The explicit connection** to RH via the precise identification of $S(T)$ as the structural residue living in $\mathrm{Fix}(S_n)^\perp$.
3. **The CCET methodology** (Canonical Catalog Equivariance Theorem) as a closure protocol: given a finite operator catalog and a base graph with non-trivial automorphism group, decide which constructions are reachable and which are blocked.

**This must be audited before claiming novelty.** Required reading list in §6.

---

## 3. Asset inventory (what we have)

All from §13sexagesima-{tertia..novena} of the research notes. Listed in dependency order.

### 3.1 Foundational lemmas (proven, source-auditable)

| Lemma | Location | Content |
|-------|----------|---------|
| **Fact A — Parameter Uniformity** | source audit at [src/tnfr/operators/remesh.py:1159, 1212–1252](../src/tnfr/operators/remesh.py), [coherence.py](../src/tnfr/operators/coherence.py), [propagation.py:42–156](../src/tnfr/dynamics/propagation.py), [self_organization.py:21–22, 44, 53](../src/tnfr/operators/self_organization.py) | Every canonical operator's coupling constants are graph-level scalars (no per-node parameters) |
| **Fact B — Prime-Cancellation Lemma** | §13vicies-novies.11 | On $G_{P14}$, every edge-propagating canonical operator decomposes as $I_{n_\text{primes}} \otimes O_{P_4}$ with prime-independent kernel |
| **Euler-Orthogonality Lemma** | §13vicies-novies.11 | Catalog-wide edge-channel compositions on $G_{P14}$ commute with $S_n$ |
| **CCET on $G_{P14}$** (Theorem 2, §13vicies-novies.16) | §13vicies-novies | Every operator built from the 13 canonical operators by composition / real-linear combination / auxiliary tensor lift / spectral functional calculus on $G_{P14}$ commutes with $\Pi_\sigma \otimes I_\text{aux}$ |
| **Tetrad-Fix($S_n$) Lemma** | §13sexagesima-octava.3 | On $G_{P14}$ with graph-uniform canonical parameters, every tetrad component and every emergent field lives entirely in $\mathrm{Fix}(S_n)$ |
| **CCET-ext** | §13sexagesima-octava.4 | Extension to UM/IL/THOL composites lifted canonically to $\mathcal{H}_\text{ext} = \mathcal{H}_{P14} \oplus \bigoplus_v \mathcal{H}_v^\text{sub}$ |
| **Canonical Product Equivariance Lemma** | §13sexagesima-quinta.3 | Tensor-product / Kronecker-sum lifts inherit equivariance |
| **Line-Graph Equivariance Lemma** | §13sexagesima-novena.2 | Equivariance survives transition to $L(G_{P14})$; orbit-invariance collapses to non-prime-distinguishing |
| **Lifted-Bundle Dichotomy Lemma** | §13sexagesima-novena.4 | Principal $S^1$-bundles on $G_{P14}$: either $S_n$-invariant connection (C4 FAIL) or non-invariant (requires external prime-pair rule, C1'-β FAIL) |

### 3.2 Closed candidates (negative results, F7-A diagnostic when numerical)

| Candidate | Location | Verdict |
|-----------|----------|---------|
| R∞-1a-operator (REMESH iterated) | §13vicies-novies.8 | Refuted (structural + empirical) |
| R∞-1a-composed (REMESH ∘ IL) | §13vicies-novies.9 | Refuted (structural + empirical) |
| R∞-1c (augmented graph) | §13vicies-novies.12–13 | `INDETERMINATE_DEGENERATE_CONSTRUCTION`, $\|D_\text{can} - D_\text{shuf}\| = 3.13 \times 10^{-13}$ |
| R∞-1b (canonical tensor-product spectral lift) | §13vicies-novies.14–15 | `INDETERMINATE_DEGENERATE_CONSTRUCTION`, $\|D_\text{can} - D_\text{shuf}\| = 1.08 \times 10^{-13}$ |
| Q1 = $G_{P14} \square G_{P14}$ (Cartesian) | §13sexagesima-quinta | `INDETERMINATE_DEGENERATE_CONSTRUCTION`, exact zero |
| Q2 = $G_{P14} \times G_{P14}$ (tensor) | §13sexagesima-quinta | `INDETERMINATE_DEGENERATE_CONSTRUCTION`, exact zero |
| Q3 (disjoint union) | §13sexagesima-quinta | Implicit closure by Canonical Product Equivariance Lemma |
| Q4 ($S_n$-non-invariant quotient) | §13sexagesima-quinta | C1'-α FAIL |
| Q5 = $L(G_{P14})$ (line graph) | §13sexagesima-novena.3 | C4 FAIL by Line-Graph Equivariance + $S_n$-transitivity |
| Q6 ($S_n$-invariant induced subgraph) | §13sexagesima-quinta | Implicit closure |
| E2 LiftedCircleBundleOnPhi | §13sexagesima-novena.5 | Both branches fail (Lifted-Bundle Dichotomy) |
| UM/IL/THOL emergent sub-EPI route | §13sexagesima-octava | C4 FAIL via Tetrad-Fix($S_n$) Lemma |
| P1 = E0 Pontryagin-dual νf measure | §13sexagesima-sexta | C1 NOT-DERIVED; C4 FAIL |
| P2 = NodeIndexedCouplingWeights | §13sexagesima-sexta | C1 FAIL at slot level |
| Dirección A (carrier-type promotion of ΔNFR) | §13sexagesima-septima | A1 FAIL, A2 CLOSED, A3 SUPERSEDED |

This is a substantial body of evidence: roughly **15 distinct construction families closed**, all reducing to the same two structural facts.

---

## 4. Proposed external paper structure (draft, ~25–35 pages)

Working title (no TNFR vocabulary):

> *Permutation Equivariance Obstructions to Spectral Approaches to the Riemann Hypothesis on Prime-Path Graphs*

### Section outline

1. **Introduction** (~3 pages)
   - Spectral approaches to RH: Hilbert–Pólya, Berry–Keating, Connes, Meyer
   - Prime-indexed Hamiltonians (P14 as motivation; cite Nucleus A internal report)
   - Weil–Guinand explicit formula as the natural target
   - Smooth/oscillatory decomposition; $S(T)$ as the RH-equivalent residue
   - **Thesis**: a wide class of natural constructions cannot reach $S(T)$

2. **The setting** (~3 pages)
   - Prime-path graph $G_n$ on $n$ primes
   - Natural $S_n$ action by prime relabelling
   - $\mathrm{Fix}(S_n)$ vs $\mathrm{Fix}(S_n)^\perp$ on $L^2(V(G_n))$ and tensor lifts
   - Self-adjoint operator with prime-ladder spectrum (analogue of P14, stripped of TNFR vocabulary)

3. **The construction class** (~2 pages)
   - Finite operator catalog $\mathcal{O}$ with graph-uniform parameters
   - Closure operations: composition, real-linear combination, tensor lift, spectral functional calculus
   - Elementary categorical graph operations: disjoint union, Cartesian product, tensor product, strong product, line graph, subdivision, induced subgraph, quotient
   - Principal $S^1$-bundles with graph-uniform connection

4. **Main theorem** (~5 pages)
   - **Equivariance Theorem**: every operator in the closure class commutes with the natural $S_n$ action (on $V(G_n)$, on $E(G_n)$ via the line graph, or on the total space of the bundle)
   - Proof reduces to two facts: parameter uniformity (axiomatic for the class) + Prime-Cancellation Lemma (combinatorial)

5. **Corollary** (~3 pages)
   - **Obstruction Corollary**: no operator in the closure class can distinguish $\log p_i$ from $\log p_{\sigma(i)}$ for any $\sigma \in S_n$
   - Hence: spectral data of any such operator is invariant under prime relabelling, hence cannot encode the prime-specific phase information required to reproduce $S(T)$

6. **Explicit no-go instances** (~5 pages)
   - Line graph $L(G_n)$ (corresponds to Q5)
   - Cartesian / tensor / strong products $G_n \square G_n$, $G_n \times G_n$ (Q1/Q2/Q4)
   - Principal $S^1$-bundles (E2)
   - Augmented graphs with $S_n$-invariant weight laws (R∞-1c)
   - Tensor-product spectral lifts (R∞-1b)

7. **What is NOT covered** (~2 pages, important for honest framing)
   - Non-graph-uniform parameters (genuinely prime-specific rules — requires external input not derivable from the bare catalog)
   - Adelic constructions à la Connes (different category entirely)
   - Constructions on a fundamentally different base object (not $G_n$ or its categorical neighbourhood)
   - Off-`G_P14` canonical constructions: each requires its own derivation

8. **Discussion and open questions** (~2 pages)
   - Where can prime-specific phase information enter without violating parameter uniformity?
   - Relationship to the explicit formula and to the Mertens / Selberg trace formula approaches
   - Possible connection to Burnol's adelic Hilbert space

### Required appendices

- **Appendix A**: source audit for parameter uniformity (in TNFR canonical implementation; can be presented abstractly)
- **Appendix B**: full proof of Prime-Cancellation Lemma
- **Appendix C**: F7-A diagnostic methodology and numerical results for the closed candidates

---

## 5. Translation work (TNFR → standard vocabulary)

A glossary for the translation, in dependency order.

| TNFR term | Standard analogue | Notes |
|-----------|-------------------|-------|
| Canonical 13-operator catalog | Finite operator family with graph-uniform parameters | Drop the count; emphasise *parameter uniformity*, not the specific 13 |
| Tetrad $(\Phi_s, \|\nabla\varphi\|, K_\varphi, \xi_C)$ | Graph-Laplacian-derived feature vector | Optional; the obstruction does not depend on the tetrad |
| ΔNFR | Discrete graph Laplacian acting on a scalar field | Standard |
| νf | Coupling rate / eigenfrequency parameter | Standard |
| EPI | State vector on $L^2(V(G_n))$ or its tensor lifts | Standard |
| U1–U6 grammar | Admissible-sequence constraints | Mostly not needed in the no-go; can be relegated to a footnote |
| CCET | "Catalog-class Equivariance Theorem" | Rename for external audience |
| Fix($S_n$) / Fix($S_n$)$^\perp$ | Invariant / coinvariant subspace under the symmetric group action | Standard rep theory |
| `INDETERMINATE_DEGENERATE_CONSTRUCTION` | Spectral data invariant under shuffle to within machine precision | Reframe as "numerical confirmation of the equivariance theorem" |
| F7-A diagnostic | Numerical equivariance test | Standard |
| T-HP, G4, B0★/B1/B2/B3, §13septies trichotomy | Drop entirely | Internal program structure, not part of the mathematical statement |
| P14, P28, P30, P12–P49 | Drop in the main paper; cite as "the prime-ladder Hamiltonian construction of (internal reference / preprint)" | Keep in an internal preprint or in Nucleus A external version |

---

## 6. Required bibliographic audit (before submission)

The novelty claim **depends on** confirming the unified statement does not already exist. Required reading and explicit comparison:

### 6.1 Spectral approaches to RH

- **Berry, M. V., & Keating, J. P.** — "The Riemann zeros and eigenvalue asymptotics" (SIAM Rev. 1999) and follow-ups. Their $xp$-Hamiltonian conjecture.
- **Connes, A.** — "Trace formula in noncommutative geometry and the zeros of the Riemann zeta function" (Selecta Math. 1999). Adelic construction.
- **Meyer, R.** — "On a representation of the idele class group related to primes and zeros of L-functions" (Duke Math. J. 2005).
- **Burnol, J.-F.** — work on the Hilbert space of the explicit formula.
- **Bombieri, E., & Lagarias, J. C.** — "Complements to Li's criterion for the Riemann hypothesis" (J. Number Theory 1999).
- **Sarnak, P.** — survey articles on the Riemann zeta function and random matrix theory.

### 6.2 Spectral graph theory and equivariance

- **Chung, F. R. K.** — *Spectral Graph Theory* (CBMS 1997).
- **Bannai, E., & Ito, T.** — *Algebraic Combinatorics I* (1984).
- **Godsil, C., & Royle, G.** — *Algebraic Graph Theory* (2001).
- **Babai, L.** — work on graph automorphism and the Weisfeiler–Leman algorithm.

### 6.3 Prime-indexed graphs specifically

- Literature on **prime graphs** (a separate, group-theoretic notion — clarify it is *not* the same object).
- Recent work on **Cayley graphs of $(\mathbb{Z}/p\mathbb{Z})^*$** and related spectral structures.

### 6.4 Equivariant operator theory

- Any text on representations of the symmetric group acting on tensor lifts of graph-theoretic Hilbert spaces.
- Schur–Weyl-type decompositions relevant to the Kronecker constructions (Q1/Q2).

**Audit outcome will determine paper viability**: if the unified statement is already in the literature in any form, the paper either pivots to (a) a survey, (b) an explicit catalog of no-go instances framed as a corollary, or (c) is shelved entirely.

---

## 7. Risk assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Unified statement already published | Medium-high | Audit (§6); if confirmed, reframe as expository / corollary |
| Individual obstructions are folklore | High | Acknowledge openly; novelty is the unification + RH connection |
| Reviewer demands proof of RH | Low (with honest framing) | Title / abstract explicitly limit scope: "obstructions to a class of approaches" |
| Reviewer dismisses as "negative result" | Medium | Cite precedent (Razborov–Rudich, Aaronson–Wigderson natural-proofs barriers in complexity theory) |
| TNFR vocabulary leaks into the paper | High during drafting | Apply translation glossary (§5) at every revision pass |

---

## 8. Effort estimate (honest)

| Phase | Effort | Deliverable |
|-------|--------|------------|
| Bibliographic audit | 2–4 weeks | Annotated reading list with explicit comparisons |
| First draft (sections 1–5) | 3–4 weeks | Self-contained statement + proof |
| First draft (sections 6–8 + appendices) | 2–3 weeks | Worked examples + source audit |
| Internal review + translation pass | 2 weeks | TNFR-free manuscript |
| External preprint (arXiv) | 1 week | Submission |
| Journal submission + revisions | 6–12 months | Published version |

**Total to arXiv**: ~3 months of focused work.
**Total to journal acceptance**: ~12–18 months.

---

## 9. Decision needed from author

Three plausible paths (recommend choosing one before drafting):

1. **Audit-first** (recommended): spend 2–4 weeks on §6 before any drafting. If the unified statement is already published, pivot or shelve.
2. **Draft-first**: write the §4 outline assuming novelty, audit in parallel. Risk: wasted writing if §6 turns up a duplicate.
3. **Internal-only**: keep Nucleus B as internal documentation (this file + the existing §13sexagesima-* notes) and do not pursue external publication. This is the **honest, low-risk default** if the bibliographic audit is not undertaken.

---

## 10. Cross-references

- [NUCLEUS_A_PRIME_LADDER_ATLAS.md](NUCLEUS_A_PRIME_LADDER_ATLAS.md) — companion internal atlas for P12–P30
- [TNFR_RIEMANN_RESEARCH_NOTES.md §13sexagesima-{tertia..novena}](TNFR_RIEMANN_RESEARCH_NOTES.md) — full derivations of every lemma cited above
- [TNFR_RIEMANN_RESEARCH_NOTES.md §13septies](TNFR_RIEMANN_RESEARCH_NOTES.md) — T-HP statement and B0★/B1/B2/B3 trichotomy context
- [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) — N15 closure; the structural identification of $\mathrm{range}(\mathcal{R}_\infty)$ ↔ smooth half and $\ker(\mathcal{R}_\infty)$ ↔ $S(T)$ is essential context for the RH connection
- [AGENTS.md](../AGENTS.md) §"TNFR-Riemann Program Overview" — top-level program status

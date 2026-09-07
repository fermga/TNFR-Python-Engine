# REMESH Fixed-Delay Surrogate and Runtime-Limit Boundary

**Status**: CORRECTED N15 HISTORICAL RECORD — restricted finite surrogate;
runtime limit and catalog completeness open
**Date**: May 26, 2026 — corrected September 2026
**Owner**: `theory/REMESH_INFINITY_DERIVATION.md`
**Source implementation**: `src/tnfr/operators/remesh.py::apply_network_remesh`

---

## Abstract

The historical N15 programme used the name $\mathcal R_\infty$ for several
different objects: a delay parameter limit, iterates of a history-advance map,
and a Fourier projection. Those objects are not interchangeable.

One exact result survives after separating them. On a **finite cyclic history
window**, with fixed delays, fixed $0<\alpha<1$, and no clipping, the filter

$$F=\beta I+\gamma S^{\tau_l}+\delta S^{\tau_g}$$

is a normal contraction because $S$ is a unitary cyclic shift and
$\beta,\gamma,\delta>0$ sum to one. Its Cesàro averages converge to the
orthogonal projection onto $\ker(I-F)$. The fixed modes satisfy both delay
conditions, so their period is $\gcd(\tau_l,\tau_g)$.

This finite model is not the clipped, history-gated runtime operation. It does
not establish a literal $\tau_g\to\infty$ limit, conserve the structural
charge, make the structural candidate energy monotone, or prove that the 13
registered operators exhaust all admissible TNFR transformations.

The sections below retain the N15 programme structure and commit anchors while
recording the corrected statements.

---

## §1. Runtime REMESH Semantics

### §1.1 Exact raw update

For every node with sufficient history, `apply_network_remesh` computes

$$
x_{\rm raw}
=(1-\alpha)^2x_0+\alpha(1-\alpha)x_l+\alpha x_g,
$$

where $x_0$ is the current EPI and $x_l,x_g$ are snapshots at the configured
local and global delays. With

$$
\beta=(1-\alpha)^2,
\qquad
\gamma=\alpha(1-\alpha),
\qquad
\delta=\alpha,
$$

the coefficient identity is exact:

$$\beta+\gamma+\delta=1.$$

The coefficients form a convex partition only when $0\leq\alpha\leq1$. The
default is $\alpha=0.5$, but the graph or glyph-factor configuration can
override it; the runtime helper does not itself clamp $\alpha$ to this interval.

### §1.2 Guards and clipping

The public runtime effect is

$$x_{\rm new}=\operatorname{structural\_clip}(x_{\rm raw}).$$

It returns without changing the graph until `_epi_hist` has at least
$\max(\tau_l,\tau_g)+1$ snapshots. Hard or soft clipping can make the map
nonlinear. The function reads `_epi_hist`; it does not append a new history
snapshot. History advancement belongs to the surrounding runtime.

### §1.3 What averaging preserves

If $0\leq\alpha\leq1$ and clipping is inactive, identical inputs are fixed:

$$x_0=x_l=x_g=c \quad\Longrightarrow\quad x_{\rm raw}=c.$$

This is preservation of constant histories, not preservation of every input.
For the squared scalar norm, convexity gives

$$
|x_{\rm raw}|^2
\leq \beta|x_0|^2+\gamma|x_l|^2+\delta|x_g|^2.
$$

The bound compares the output with the weighted energy of **three** snapshots.
It does not imply $|x_{\rm raw}|^2=|x_0|^2$. Depending on the delayed values,
the current-node EPI magnitude can increase or decrease. REMESH averaging is
therefore not an isometry and is not generally energy-neutral.

---

## §2. Three Operators That Must Be Distinguished

### §2.1 Runtime map $M_{G,h}$

The runtime map acts on a graph and a stored history, applies the guard and
clipping, and uses configuration-dependent parameters. It is the canonical
engine operation.

### §2.2 Companion history advance $T$

A mathematical recurrence can insert the new value at the head of a history
vector and shift all prior entries. This companion-style map is useful for
studying an isolated recurrence, but `apply_network_remesh` does not perform
that shift itself.

### §2.3 Finite cyclic filter $F$

On $\mathbb C^n$, let $S$ be the unitary cyclic shift and define

$$F=\beta I+\gamma S^{\tau_l}+\delta S^{\tau_g}.$$

This is a convolution filter on a periodic sample window. It is the model used
by the corrected Fourier projector in
`src/tnfr/riemann/remesh_infinity_residue_split.py`.

The spectrum of $F$ does not describe the companion map $T$ merely because
both contain the same coefficients. The historical N15 derivation conflated
these two constructions.

---

## §3. The Literal $\tau_g\to\infty$ Question

For a fixed finite `_epi_hist`, increasing $\tau_g$ eventually triggers the
insufficient-history guard, so the runtime call becomes a no-op. If the stored
history grows with $\tau_g$, then the domain, selected snapshot, deque length,
and possibly the graph state change together. A nontrivial limit requires a
declared common state space, an embedding of histories, parameter bounds, and a
mode of convergence.

N15 supplied none of those data. Consequently:

- the fixed-history pointwise behavior is eventually the guard-induced no-op;
- a growing-history runtime limit remains undefined until an embedding and
  trajectory are specified;
- neither behavior is the finite cyclic Cesàro projection of §7.

---

## §4. Corrected Finite Cyclic Surrogate

Fix integers $n,\tau_l,\tau_g>0$ with $n$ divisible by both delays, and fix
$0<\alpha<1$. Let $S$ be the cyclic shift on $\mathbb C^n$. The DFT basis
$v_k(j)=n^{-1/2}e^{2\pi i kj/n}$ diagonalizes $S$ and therefore $F$.

For $\omega_k=2\pi k/n$, the eigenvalue is

$$
\mu_k
=\beta+\gamma e^{-i\omega_k\tau_l}
       +\delta e^{-i\omega_k\tau_g}.
$$

This statement is exact for the finite cyclic filter. It is not a transfer
function for the companion history-advance map.

---

## §5. Contractivity and Power Boundedness

Because all three coefficients are positive and sum to one,

$$|\mu_k|\leq\beta+\gamma+\delta=1.$$

The matrix $F$ is a polynomial in the unitary matrix $S$, hence is normal. It
follows that

$$\|F\|_2=\max_k|\mu_k|\leq1,
\qquad
\|F^m\|_2\leq1.
$$

Power boundedness is therefore proved for this finite cyclic model. It was not
proved for the historical infinite companion operator. In particular, for the
old weight $w(k)=\rho^{-k}$, the stated right-shift norm $\sqrt\rho$ had the
direction reversed; the norm is $\rho^{-1/2}$ with that convention.

---

## §6. Fixed Modes: GCD, Not LCM

For $0<\alpha<1$, equality $\mu_k=1$ in the convex combination requires

$$
e^{-i\omega_k\tau_l}=1,
\qquad
e^{-i\omega_k\tau_g}=1.
$$

Let $d=\gcd(\tau_l,\tau_g)$. On a compatible window, the common solutions are

$$
\omega_m=\frac{2\pi m}{d},
\qquad m=0,\ldots,d-1,
$$

or DFT-bin indices $k=mn/d$. The fixed subspace has dimension $d$.

The historical use of $\operatorname{lcm}(\tau_l,\tau_g)$ as the fixed-mode
period was incorrect. The LCM can still be used as a convenient sample-window
alignment, but it does not determine the common fixed modes. For the defaults
$(\tau_l,\tau_g)=(4,8)$, $d=4$, so the fixed frequencies are
$0,\pi/2,\pi,3\pi/2$ modulo $2\pi$.

---

## §7. Cesàro Projection Theorem for the Surrogate

Define

$$
A_M=\frac1M\sum_{j=0}^{M-1}F^j.
$$

In the DFT basis, the multiplier of $A_M$ is $1$ when $\mu_k=1$ and

$$
\frac{1-\mu_k^M}{M(1-\mu_k)}
$$

otherwise. Since the model is finite-dimensional, these nonfixed multipliers
converge to zero. Therefore

$$
A_M\xrightarrow[M\to\infty]{\|\cdot\|_2}P_d,
$$

where $P_d$ is the orthogonal projector onto $\ker(I-F)$.

For a fixed window and fixed delays, an $O(1/M)$ operator-norm bound follows
with a constant depending on
$\min_{\mu_k\ne1}|1-\mu_k|$. No uniform constant follows as the window,
delays, or coefficients vary. No rate for the nonlinear structural candidate
energy follows from this state-space estimate.

---

## §8. Projection Algebra

The finite surrogate projection satisfies

$$P_d^*=P_d,
\qquad
P_d^2=P_d,
\qquad
\|P_d\|_2=1$$

when its range is nonzero. Its spectrum is contained in $\{0,1\}$; both values
occur when the selected subspace is proper and nontrivial.

These are properties of $P_d$ in the Euclidean norm on the declared periodic
history window. They do not transfer automatically to graph observables or to
the runtime map.

---

## §9. Why the Historical $H^2$ Proof Does Not Apply

The historical note represented the companion head-insertion plus shift as if
it were multiplication by
$\beta+\gamma z^{\tau_l}+\delta z^{\tau_g}$. That symbol belongs to a
translation-invariant convolution or cyclic filter, not to the companion map
whose first row is updated while the remaining rows shift.

Additional problems were:

1. constant infinite histories do not belong to unweighted $\ell^2$ or
   $H^2$ coefficient space;
2. the claimed shift norm used the inverse geometric factor;
3. a spectral-radius bound was asserted without a valid spectrum formula;
4. power boundedness of the companion map was assumed rather than proved;
5. symmetrizing an operator does not preserve the spectra of all its powers.

The mean ergodic theorem is valid when its hypotheses hold. Those hypotheses
were not established for the operator that the historical proof defined. The
finite cyclic theorem in §7 supplies a valid restricted replacement.

---

## §10. Relation to the Nodal Equation

REMESH changes EPI through a canonical operator path. The finite filter can be
applied to a sampled EPI history as an auxiliary diagnostic. An identity such
as

$$\partial_t(P_d\mathrm{EPI})=P_d(\partial_t\mathrm{EPI})$$

requires a common linear function space and sufficient regularity. Even when
that commutation is valid, substituting the nodal equation only gives

$$\partial_t(P_d\mathrm{EPI})=P_d(\nu_f\Delta\mathrm{NFR}).$$

It does not show that the projected trajectory is produced by the runtime
REMESH operator or that pressure and capacity close on the projected state.

---

## §11. Structural Charge

The structural charge diagnostic is

$$Q=\sum_i(\Phi_s(i)+K_\phi(i)).$$

It depends on pressure, phase, topology, and the corresponding field
extractions. An EPI-history projector does not act on all of these arguments.
Exact preservation would require a specified lifted state map and an
invariance relation such as $Q\circ P_d=Q$ along a separately conserved
evolution. Neither follows from $P_d^2=P_d$ or $P_d^*=P_d$.

Thus the historical projected-Noether conservation claim is superseded.
Charge before and after REMESH remains trajectory telemetry unless a
model-specific proof supplies the missing commutation and conservation laws.

---

## §12. Energy and Isometry Boundary

The structural functional

$$
E=\frac12\sum_i
(\Phi_s^2+|\nabla\phi|^2+K_\phi^2+J_\phi^2+J_{\Delta\mathrm{NFR}}^2)_i
$$

is a nonnegative candidate energy. EPI is not an explicit term. Therefore an
EPI-only write leaves a same-snapshot evaluation unchanged when every derived
field is held fixed. This dependency fact is not an isometry and not a
conservation theorem after pressure, currents, phase, or fields are updated.

For the finite surrogate, orthogonality gives

$$\|P_dx\|_2\leq\|x\|_2.$$

This contracts the declared history norm, not the structural functional $E$.
For $E[P_dx]\leq E[x]$ one would need an $E$-compatible state space and
contractivity in that energy metric. Monotonic decay along a trajectory needs
an evolution law with a verified nonpositive derivative.

---

## §13. Registry Reuse

`P_d` is a read-only numerical projection derived from a selected surrogate.
It can be computed by a utility function without adding a class to the engine's
operator registry. This establishes only implementation reuse:

$$\text{auxiliary computation} \not\Rightarrow
  \text{new registered state transformation}.$$

The legacy phrase “no fourteenth operator is required” is valid only in this
narrow engineering sense.

---

## §14. Catalog Completeness Remains Open

The current registry has 13 declared operators and coherent metadata. To prove
that these operators exhaust all admissible TNFR transformations would require:

1. a transformation space defined independently of the existing names;
2. admissibility axioms derived from the nodal equation and invariants;
3. a representation or generation theorem for every admissible map;
4. an irreducibility or equivalence criterion for proposed new maps.

Registry size, reload idempotence, metadata alignment, and reuse of `P_d` do
not supply those ingredients. This is the open S10 boundary in
[CORE_RESEARCH_PROGRAM.md](CORE_RESEARCH_PROGRAM.md).

---

## §15. Spectral Comparisons

For fixed $n$ the projector has finitely many selected DFT modes. Calling their
spacing a continuum spectral density adds an unsupported limit. Comparisons
with Riemann-zero density, Kolmogorov spatial spectra, or random-matrix spacing
laws can be posed only after specifying a scaling family and an intertwining
map between observables.

The corrected fixed-mode calculation is useful as a finite periodicity test.
It proves no spectral universality and no mismatch theorem about the full
runtime operation.

---

## §16. P50 Finite Fourier Diagnostic

`build_resonant_bin_mask` retains the historical requirement that the sample
count be divisible by $\operatorname{lcm}(\tau_l,\tau_g)$. Within that window,
it now selects the bins fixed by both delays, using
$d=\gcd(\tau_l,\tau_g)$.

`split_residue_by_remesh_infinity` is a legacy API name. It returns
$P_dx$ and $(I-P_d)x$ using an orthogonal DFT mask. Parseval gives, up to
roundoff,

$$
\frac{\|P_dx\|_2^2}{\|x\|_2^2}
+\frac{\|(I-P_d)x\|_2^2}{\|x\|_2^2}=1.
$$

This identity validates the numerical split. It does not identify the
analytic support of an off-grid signal: a finite rectangular window creates
spectral leakage.

---

## §17. TNFR-Riemann Boundary

The P31 prime-ladder signal can be projected onto the finite periodic subspace
and its complement. A small selected fraction means only that the declared
finite sample has little DFT energy in those bins. No equality relates this
split to the smooth and oscillatory parts of the Riemann counting formula.

Consequently N15 and P50 do not advance T-HP or RH. Historical B1/B2 language
is retained only as programme history, not as a theorem implied by the Fourier
certificate.

---

## §18. Navier–Stokes Boundary

The finite surrogate acts on a one-dimensional sample index. A temporal
projection does not by itself produce, preserve, or rule out a spatial
$k^{-5/3}$ spectrum. Any statement about a Navier–Stokes field requires a
declared space-time tensor product and proof that the temporal map acts as the
identity on the spatial factor.

No conclusion about vortex stretching, regularity, or a cascade follows from
the REMESH surrogate.

---

## §19. Historical Claim Ledger

| Historical N15 claim | Current status |
|---|---|
| $\tau_g\to\infty$ equals a Cesàro projector | Superseded: these are different limits |
| Companion history map has the polynomial Fourier symbol | Superseded: the symbol belongs to the cyclic/convolution filter |
| Fixed modes use $\operatorname{lcm}(\tau_l,\tau_g)$ | Corrected to $\gcd(\tau_l,\tau_g)$ for $0<\alpha<1$ |
| The history operator is power bounded on the stated $H^2$ model | Unproved for that companion map |
| Projected structural charge is conserved | Conditional and unproved |
| Projected structural energy is monotone with universal $O(1/n)$ decay | Superseded |
| The surrogate closes the 13-operator catalog | Superseded; registry reuse only |
| Direct Riemann/K41/RMT verdicts apply to runtime REMESH | Superseded; finite-surrogate comparisons only |

The original commits remain useful provenance:

- `a1f298fd`: historical W1 operator-existence claim;
- `badac156`: historical W2 conservation and Lyapunov claim;
- `48b0574a`: historical W3 spectral and branch verdict.

---

## §20. Corrected Branch Status

- **Branch A**: established only for the finite cyclic fixed-delay surrogate:
  its Cesàro projector exists.
- **Branch B1**: no universality conclusion follows without a scaling family
  and an intertwining map.
- **Branch B2**: no extra registry entry is needed to compute this projection;
  admissible-transformation completeness remains open.
- **Branch B3**: nonexistence is ruled out for the finite cyclic surrogate, not
  for the literal runtime $\tau_g\to\infty$ problem or the historical infinite
  companion map.

---

## §21. Scope Across TNFR Programs

The corrected result is internal and limited:

- it supplies a finite periodic-history diagnostic;
- it separates an auxiliary linear model from the canonical clipped runtime;
- it corrects the fixed-mode arithmetic from LCM to GCD;
- it leaves all classical open problems unchanged;
- it leaves the S10 catalog-completeness problem open.

No “structural/operational universality” follows across arbitrary graphs,
initial states, clipping policies, or time-varying parameters.

---

## §22. Reproducibility and Direct Checks

The corrected numerical surface is
`src/tnfr/riemann/remesh_infinity_residue_split.py`.
For a compatible sample count $n$:

1. the selected bin count is $d=\gcd(\tau_l,\tau_g)$;
2. the selected indices are multiples of $n/d$;
3. applying the split twice leaves each part unchanged;
4. the two parts reconstruct the input up to FFT roundoff;
5. their inner product vanishes up to FFT roundoff;
6. their squared-norm fractions sum to one for nonzero input.

The operator-registry diagnostic in
`src/tnfr/riemann/operator_catalog_discipline_signature.py` checks declared
schema consistency only.

---

## §23. Open Research Questions

The following problems remain open:

1. Define a common state space and convergence mode for a nontrivial runtime
   $\tau_g\to\infty$ limit.
2. Analyze the companion history-advance map without replacing it by a
   convolution operator.
3. State and verify admissible bounds for configurable $\alpha$ when convex
   averaging is required.
4. Determine when a lifted REMESH map preserves a declared structural charge
   or dissipates a declared energy.
5. Establish any bridge from the finite periodic projection to the Riemann,
   Navier–Stokes, or other programme observables.
6. Define the admissible TNFR transformation space independently and resolve
   catalog generation and irreducibility within it.

The exact current conclusion is therefore narrow: **a fixed finite cyclic
REMESH filter has a computable Cesàro fixed-mode projection; the runtime limit,
structural invariants, and global operator completeness are unresolved.**

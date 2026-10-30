# Zwanzig: projection, memory and sufficient state in TNFR

Author: `/root/joint_review_controls`, 2026-10-29.

**Assessment:** the immediate useful transfer is a more precise admission of
the retained state, hidden initialization and approximation error. TNFR already
implements the relevant deterministic linear elimination and observation
closure mechanisms. A second general projection framework is not justified by
this reading. Statistical closures, equilibrium noise identities and Markovian
approximations need additional premises not supplied by an uncertainty box,
an average, a small coupling coefficient or a long relaxation dwell.

This is a selective literature/source review, not a new theorem, numerical
evaluation, physical mapping or permission to change the current research
question. No maintained source, PDF, producer or scientific test was changed
or executed. PDF rendering, file hashes and provenance are documentary work.

## Source and actual reading coverage

Source: Robert Zwanzig, *Nonequilibrium Statistical Mechanics*, the user-supplied
[local PDF](<../../bibliografia/Nonequilibrium Statistical Mechanics.pdf>),
7,956,649 bytes, SHA256
`bdab812df40b219ad6611396a08dee68ca5d974486fadb292bf7c7d3a208e904`.
The file has 233 PDF pages. In the reviewed body, PDF page = printed page + 11.

| Complete PDF pages read | Printed pages | Actual coverage |
| --- | --- | --- |
| 154-170 | 143-159 | Sections 8.1-8.3 and the beginning of 8.4: projection, block elimination, exact generalized Langevin identity, initial ensembles, averaged noise, fluctuation-dissipation and correlation identities. The combined-response continuation beyond page 159 was not read. |
| 176-179 | 165-168 | Section 8.6, slow variables and its final warning; page 165 also contains the ending of the preceding example, not a full reading of section 8.5. |
| 185-198 | 174-187 | Sections 9.2-9.3 and the selected beginning of 9.4: conditional projection, nonlinear memory, constrained initial ensembles, slow-variable and Markovian approximations, and sources of nonlinearity. Page 174 also contains the ending of 9.1, not its complete mode-coupling derivation. |

These are **35 complete pages**, read in retained extracted text and then all
visually inspected in page renders. Contents PDF pages 8-10 were inspected for
navigation. This is not a cover-to-cover reading or a verification of every
theorem in the book. The extraction omits many displayed formulas, so it was
used for navigation and prose, not as an authoritative formula transcription.
Every formula used below was checked on the page image.

The existing text extraction is
`%TEMP%/tnfr-bibliography-audit-20261029/book-2.txt`, SHA256
`631de6a17612ca3975947ce53cecb3706ef7a1da313e218744c869962ea2b7f4`.
[projection-memory-provenance.json](projection-memory-provenance.json) binds
that extraction, the source PDF, all 35 inspected PNGs and the six TNFR owners
compared below. Reading renders are retained only in
`%TEMP%/tnfr-bibliography-audit-20261029/zwanzig-visual/page-NNN.png`, outside
the repository. They were produced with the installed TeX Live `pdftoppm`,
`-scale-to 1400 -png`, and relocated byte-for-byte before this review was
finalized. No rendered book pages are included in project evidence archives.

## What the selected source actually establishes

### 1. An exact reduction can contain memory and hidden initial forcing

The block elimination on printed page 147 / PDF 158, equations (8.12)-(8.14),
starts with a complete linear initial-value problem. Solving its hidden block
and substituting it into the visible block produces both a history integral
and a term depending on the hidden initial value. In generic block notation,
already used by TNFR,

\[
\dot y=Ay+Bh,\qquad \dot h=Cy+Dh,
\]

\[
\dot y(t)=Ay(t)+B e^{Dt}h(0)
 +\int_0^t B e^{D(t-s)}C y(s)\,ds.
\]

The formula is an exact rearrangement, not a new constitutive force or evidence
that the microscopic law is random. A specified microscopic initial state
determines its hidden-initialization term. Giving that term a stochastic
interpretation requires a declared probability distribution over preparations.
It need not be Gaussian, white, independent of the visible preparation, or
negligible. A set of possible preparations is not itself such a distribution.

The more general operator construction on printed pages 148-150 / PDF 159-161
uses a projection onto selected observable functions. Equation (8.18) depends
on an inner product and an inverse observable Gram matrix. Its selection of
relevant variables is a modeling choice, not a proof that the discarded
variables are harmless. The resulting identity, equations (8.24)-(8.29),
separates projected evolution, memory and orthogonal initial information.
An exact formal identity can still be difficult to compute and need not
reduce the cost of prediction.

**Transfer boundary:** a finite coordinate projection, a conditional
expectation on a probability space, and a quotient by a symmetry are different
constructions. They must not share a name while silently exchanging premises.
The book's Mori kernel uses a minus-convolution convention in (8.29); the
generic block elimination above has a plus sign. TNFR must retain its own
generator and kernel definitions rather than copy a sign from the symbol `K`.

### 2. Noise averaging and equilibrium identities require specific measures

Printed pages 144-145 / PDF 155-156 define the equilibrium-weighted inner
product and its anti-Hermitian Liouville generator in the book's setting.
Printed pages 151-156 / PDF 162-167 distinguish the exact operator identity
from an averaged Langevin description. In the specified near-equilibrium
initial family, the chosen inner product makes the average orthogonal force
second order in the deviation, while the retained response is first order.
This is a preparation-dependent approximation; it does not remove the force
for every microscopic initial state.

The fluctuation-dissipation identity on printed page 157 / PDF 168,
equations (8.58)-(8.59), additionally uses the anti-Hermitian relation and the
chosen inner product. It does not license a thermal noise model for arbitrary
TNFR dissipation. TNFR's reversible *nodal diffusion* has a self-adjoint
dissipative generator in its `H` metric. That node-space metric is not an
equilibrium probability measure over full configurations, and self-adjoint
dissipation is not the book's anti-Hermitian microscopic generator.

For a nonlinear retained observation, printed pages 176-180 / PDF 187-191
replace a finite linear span by functions of the selected variables. The
conditional equilibrium distribution in (9.43) and average in (9.45)-(9.46)
explicitly depend on the selected measure and its normalization on a level
set. The exact expression remains nonlocal in time and in observable values.
Writing a conditional average therefore does not supply a closed autonomous
trajectory law. Denominators such as `g_eq(a)` also require an admitted domain;
they cannot be assigned across unsupported or zero-density values.

Printed page 181 / PDF 192, equation (9.61), states the constrained-equilibrium
initial family that removes the averaged source. The discussion on printed
pages 184-185 / PDF 195-196 reiterates that restriction. Adopting this family
would be a new preparation premise for TNFR, not an implementation optimization
of its already retained arbitrary source errors and correlations.

### 3. A Markovian approximation requires more than a small kernel

Printed pages 165-168 / PDF 176-179 discuss a controlled slow variable whose
rate carries a small parameter. The kernel can then be approximated to a
specified order, subject to the time-scale assumptions. The explicit warning
on printed page 168 is directly useful: a small memory kernel does not make
the retained variable slow when its streaming term is still large.

Printed pages 181-183 / PDF 192-194 make the nonlinear version more explicit:
the hidden correlation decays on a time scale short compared with appreciable
change of the selected variables; replacing past values by current ones and
integrating the kernel then gives an approximate local equation. The formal
small-parameter order is not a certified finite-horizon error allowance for
an arbitrary new model. Integrability, initialization, retained dynamics and
the observation horizon still matter.

For TNFR, neither `gamma` being small nor using a long symbolic dwell proves
this separation. A useful reduction needs a bound on the omitted source and
memory contribution under the actual support, law, preparation and requested
error. Changing a reference time cannot silently erase carried initialization.

## Comparison with the existing TNFR implementation

| Existing owner inspected | What is already available | What Zwanzig helps clarify, without transferring a theorem |
| --- | --- | --- |
| [DERIVED_EPI_MEMORY](../../theory/DERIVED_EPI_MEMORY.md), sections 1-4 | Exact fixed reversible diffusion, weighted projection/lift, hidden source, Volterra memory and a model-specific zero-memory criterion. | Memory is derived by elimination. Its positivity and equivalence between vanishing zero-lag kernel and closure use reversibility; they are not universal projection properties. |
| [linear_observation.py](../../src/tnfr/mathematics/linear_observation.py), `derive_coordinate_memory` (line 274) | Exact fixed-matrix block elimination with original hidden initialization; no exponential evaluation or decay claim. | This already implements the immediately reusable content of the block argument. A zero `B*C` need not make the entire kernel zero for a general generator. No duplicate Mori-Zwanzig wrapper is needed. |
| Same shared owner, `derive_linear_observation` (line 339), and DERIVED_EPI_MEMORY section 11 | Complete-level invariant row-space construction for a declared linear output, with exact all-state minimality in that fixed linear class. | Enlarging the retained state is an alternative to history. The result can retain the full dimension, and does not prove nonlinear or reachable-family minimality. |
| [SINE_CLASS_MEDIATED_MEMORY](../../theory/nodal/SINE_CLASS_MEDIATED_MEMORY.md#sine-class-memory-reached-closure-obstruction) | A same-observation/different-future obstruction on an actual acquired source family, plus conditional tangent memory and nonlinear error boundaries. | A local visible state, class label and conserved means can be insufficient even after formation. A memory description is a legitimate response to this obstruction, not a defect or a refutation of identity. Its three-C9 coefficients do not transfer to two-port contact. |
| [SINE_REDUCED_CLASS_PORTS](../../theory/nodal/SINE_REDUCED_CLASS_PORTS.md#sine-reduced-class-ports) | Twenty retained form/phase coordinates, two-channel port inputs, nonlinear bridge, initial-source term and finite-error accounting. | Port composition retains hidden preparation and the input history. A component average alone cannot replace the carried profile; a controlled surrogate is not an exact nonlinear quotient. |
| [joint_quotient.py](../../src/tnfr/physics/joint_quotient.py), `observe_joint_nodal_quotient` (line 112) | Admission checks for held phase/capacity, equitable counted support and exact all-state EPI closure; primitive and effective capacities remain distinct. | A probability-space conditional average cannot bypass those deterministic quotient hypotheses. An EPI-only observation does not establish full phase/capacity/event closure. |
| [CONCEPTUAL_CORE](../../theory/CONCEPTUAL_CORE.md#1-structure-state-and-identity) and DERIVED_EPI_MEMORY section 10 | Sufficient state is tied to a law, prediction, continuation and events; observer changes across events are explicit. | The retained observation, source family and event handoff belong to the reduction contract. Changing degrees can change observation weights without physical state motion. |

The current repository already states most of these boundaries correctly.
The review found no reason to replace its deterministic source-family account
with an equilibrium ensemble, loosen the quotient admission, or add a generic
stochastic engine. The book supplies a useful organization of existing methods,
not missing evidence for source-to-identity acquisition.

## Proposed action matrix for the ongoing admission work

These are recommendations to the sole active question, not separately activated
studies. They require no new runtime abstraction by themselves.

| Priority | Concrete action or decision | Existing owner to reuse | Admission / useful stopping result |
| --- | --- | --- | --- |
| Immediate | State whether the requested prediction is memberwise, uniform over the retained source family, or an ensemble average. Preserve actual hidden initialization and its correlations. | Current source/identity admission plus the core state contract. | An interval family alone supplies no averaging measure; do not turn worst-case uncertainty into zero-mean noise. |
| Immediate | For a proposed reduced joint state, first compare states on the same observation fiber under the same actual source and surrounding preparation. | Source-compatible obstruction method in SINE_CLASS_MEDIATED_MEMORY; existing closure owners for genuinely linear models. | A matched different-rate/future witness rejects that reduction. It does not reject the organization or justify changing the source. No computation is required merely to state this obligation. |
| Immediate | Keep the complete two-channel port and the event-reweighted conserved quantities in the source handoff. | SINE_REDUCED_CLASS_PORTS and the actual contact owner. | A supplied phase origin, zero residual or newly centered mean must not replace the reached source. Boundary inputs must be inherited jointly from the actual neighbor. |
| Conditional | If an admitted fixed linear model really needs compression, choose between a sufficient augmented linear state and an explicit memory description. | `derive_linear_observation` versus `derive_coordinate_memory`. | Preserve coefficient admission and source terms. If no compression or error advantage is proved, retain the full state. Do not run either just to increase the inventory. |
| Conditional | If finite-history or memoryless approximation is scientifically needed, declare an initialization budget, kernel/tail bound, retained-rate bound, domain, horizon and propagated output error. | DERIVED_EPI_MEMORY's fixed P5 finite-history method as a method reference, not transferred coefficients. | Without a useful matched bound, keep memory/full state. Pointwise stability and a small coefficient do not establish this certificate. |
| Deferred | Consider conditional expectation or a statistical reduced law only when the selected claim consumes an explicitly justified measure and preparation ensemble. | A future actual-law owner; no implementation change proposed now. | Require defined conditional normalization, generator/domain and boundary assumptions, and a separate error argument for Markovianization. Otherwise this route adds unsupported premises. |
| Do not do | Treat a fitted memory kernel, averaged closure, Gaussian/white forcing or fluctuation-dissipation relation as an emergent physical law of TNFR. | Preserve the current foundation and physical-readiness gates. | The selected chapters establish none of those identifications for the current dissipative sine model. |

## Specific conceptual pitfalls to prevent

1. **Mean closure is not memberwise closure.** An equation for an expectation
   can coexist with different trajectories having the same initial retained
   coordinates. It does not satisfy the all-state quotient claim.
2. **Hidden initial forcing is not automatically stochastic noise.** It is
   deterministic once a fine initial condition is fixed. Randomness, if used,
   belongs to an additional distribution over that condition.
3. **Memory is not a failure of the underlying dynamics.** It can be the exact
   cost of observing fewer coordinates. Conversely, the presence of a kernel
   proves neither a new organized identity nor its formation.
4. **Orthogonality does not erase every source.** In the linear projection,
   the noise-averaging claim has a near-equilibrium source restriction; in
   the nonlinear conditional construction, it has a constrained-ensemble
   restriction. Neither is the unrestricted retained TNFR preparation.
5. **A Markov limit is not an algebraic renaming.** Streaming, memory time,
   source error and the requested observation scale must fit together.
6. **A projection does not choose the organization.** Relevant variables are
   chosen for a prediction; membership, initial absence, acquisition and
   continued response remain separate obligations.

**Recommended use now:** strengthen the existing source/identity admission
with explicit sufficient-state and initialization obligations. Reuse the
current exact and finite-error owners if that admitted question actually needs
a reduction. Do not postpone source admission to build another projection
library, and do not claim that this reading supplies the missing acquisition
or quantitative tail certificate.

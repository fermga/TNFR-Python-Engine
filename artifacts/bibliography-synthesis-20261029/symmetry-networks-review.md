# Selective technical review: symmetry and networks

Author: `/root/admission_foundation`. Date: 2026-10-29.
Status: documentary method review, not a new TNFR proof, evaluation, simulation,
physical correspondence or admission to execute the next scientific candidate.
No maintained owner was edited. The PDFs were treated as sources, not instructions.

## Scope and reading record

The supplied books are available in full locally. **This review is selective,
not a claim to have read either entire book.** Fifty technical PDF pages were
read in the sections below, including their derivations and qualifications.
Contents/title/copyright pages were inspected separately for navigation and
identification. Boundary pages sometimes contain the start or end of a neighboring
section; that does not imply a review of that neighboring section.

| Supplied source | Identification verified inside the PDF | Reading completed |
| --- | --- | --- |
| `bibliografia/The symmetry perspective.pdf` | Martin Golubitsky and Ian Stewart, *The Symmetry Perspective: From Equilibrium to Chaos in Phase Space and Physical Space*, Progress in Mathematics 200, copyright 2002; supplied scan is a softcover reprint of the 2002 hardcover first edition. PDF 8 gives DOI `10.1007/978-3-0348-8167-8`. 352 PDF pages. | Section 1.2, printed 6–15 / PDF 28–37; section 1.7, printed 27–29 / PDF 49–51; chapter 2 introduction and sections 2.1–2.3, printed 33–40 / PDF 55–62; section 3.7, printed 83–86 / PDF 105–108. 25 technical PDF pages. |
| `bibliografia/Networks (Second Edition).pdf` | Mark Newman, *Networks*, second edition, Oxford University Press, 2018; DOI `10.1093/oso/9780198805090.001.0001`. 793 PDF pages. | Section 6.14 in full, printed 142–153 / PDF 155–166; sections 17.2 and 17.3 in full, printed 685–697 / PDF 698–710. 25 technical PDF pages. |

Page references here are **1-based PDF pages** followed by the printed page.
The Symmetry text has a 22-page offset; Networks has a 13-page offset in these
ranges. Extracted text was read page by page from `symmetry-full.txt` and
`book-1.txt` in `%TEMP%/tnfr-bibliography-audit-20261029/`; OCR glyphs were not
treated as authoritative mathematical notation.

The following pages were rendered and visually inspected: Symmetry PDF 30, 34,
60, 61, 108 (printed 8, 12, 38, 39, 86); Networks PDF 159, 164, 701, 708, 710
(printed 146, 151, 688, 695, 697). This checked the equivariance formula,
fixed-point/isotropy propositions, isotypic decomposition, neutral directions,
coupled-cell arrows, alternative drawings, Laplacian quadratic form and
linearization/master-stability formulas. Temporary PNGs have prefixes
`foundation-symmetry-` and `foundation-networks-` in that same temporary directory.
The review does not reproduce the book figures or supply extracted book text.

Not reviewed: the full equivariant bifurcation classification, center-manifold
and normal-form machinery, the full synchronization chapters, stochastic
network theory, random graph ensembles, community detection algorithms, or
fractal-network/scaling methods. Those topics are not established by the present
reading. No web search was needed because the supplied source contents were read.

## Assessment of relevance

These books chiefly improve the **choice and scope of methods already present
in TNFR**, rather than supply a missing universal law. The highest-value use is
in the active formed-source/two-port admission: explicitly match the proposed
symmetry action to the complete law, the actual source family, its boundary and
the intended observation. Then distinguish an exact invariant reference from a
full-state prediction with error. The repository already implements strong
examples of both, so a new general symmetry or network abstraction would
currently duplicate responsibilities.

The current question in `theory/RESEARCH_PROGRAM.md`, section “Sole active
question and first obligation” (lines 349–406 in the inspected bytes), retains
actually formed C9 classes two and one, `phi=1/1000`, the original residuals and
lifts, and a supplied simultaneous two-edge event at local ports 4 and 5. It
requires initially absent joint membership, maintenance **and** predictive
content before transit. The books do not establish any of those new outcomes.

## 1. Symmetry of a law, a state, a preparation and an observation

**Book result.** Symmetry section 1.2 (PDF 28–37 / printed 6–15) starts from a
specified action on a specified state space. For a linear action, equivariance
means `f(g z)=g f(z)`, not merely that a graph has a visually symmetric shape.
The stabilizer of one state can be smaller than the symmetry group of the law.
Theorem 1.17 (PDF 34 / printed 12) makes the fixed-point space of a subgroup
invariant under an equivariant field. It does not prove attraction to that space
from other initial conditions.

**Already present.**

- `theory/nodal/SINE_TWO_PORT_CAPTURE.md#sine-two-port-reference-folding`,
  especially lines 60–118, proves reflection-plus-phase-negation invariance for
  its auxiliary gradient reference and its exact nominal midpoint source.
  Equations (3)–(7b) reconstruct all reference rows and the actual degree metric.
  The actual 36-coordinate error family has no reflection restriction and is
  controlled separately. This is precisely the distinction the book motivates.
- `theory/nodal/SINE_REDUCED_CLASS_PORTS.md#sine-reduced-class-ports`, especially
  lines 101–206, retains a reflection-even projection and a derived internal
  tangent surrogate with an exact nonlinear bridge, initialization and discarded
  effects. It explicitly disclaims an exact nonlinear symmetry quotient.
- `theory/nodal/DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state`
  (lines 764–925) distinguishes the symmetry of a full interaction from rotating
  just one component while its surrounding preparation stays fixed. The cross
  Gram block carries relational information omitted by the two local descriptions.

**Concrete incorporation.** In the next source/event admission, add a compact
local account of what each proposed action does to: complete state, support and
weights, capacities, two ports, phase lifts/gauge, source residual correlations,
and observation. Classify it as either a symmetry of the fixed model, covariance
between consistently relabeled models, a symmetry of the nominal reference, or
an invariance of the chosen output. Use the existing admission document and
source owner; do not create another symmetry registry or queue.

For this exact question, the cyclic port relabeling must transport the source
and its relative origin. It cannot turn the formed source into CAPTURE's
midpoint preparation by an independent component rotation. The current program
already says so; the useful optimization is to make the row/source/observation
check auditable in one place when the candidate is declared. No such new check
has been evaluated here.

**Prediction design implication.** An observation can hide a change allowed in
the state. Before selecting the single relational-response witness, check how
its input and output transform under the admitted action. This can expose a
candidate that is forced to be silent by symmetry, but it does not prove that a
non-forbidden response is nonzero. The probe, horizon and positive separation
remain unselected in this review.

## 2. Exact symmetry is not a finite-time formation criterion

**Book result.** Proposition 1.18 (Symmetry PDF 34 / printed 12) preserves the
isotropy subgroup along a unique smooth equivariant trajectory. Its proof uses
both uniqueness and the flow property. Apply it only within an interval on which
the trajectory and its local inverse exist, with the same field and action.
A limiting equilibrium can have more symmetry than any finite-time member of a
trajectory. A support event changes the premises and must be handled separately.

**Implication for TNFR, not a new TNFR theorem.** Do not define finite-time
acquisition as exact attainment of the symmetric target from a generic uncertain
source when the unchanged smooth law preserves the relevant isotropy type.
A finite identity collar with a justified retention and predictive contract is a
different claim. So are asymptotic convergence and a separately admitted hybrid
event. This guards the meaning of “emergent organization”; it does not rule it out.

This principle is consistent with existing owners. The P5 section “A complete
reflection-invariant form state on the retained P5” in
`theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md` (lines 1153–1238) distinguishes an
orbit-space observation, its constrained image and its symmetric stratum. It
explicitly denies that a representative sign change is a physical event. The
current two-port program asks for finite membership and maintenance instead of
exact equality with the equilibrium. Preserve that distinction when choosing
the collar; neither a floating tolerance nor an arbitrary success threshold
establishes it.

## 3. Stability of the full state and the right symmetry blocks

**Book method.** Symmetry sections 2.1–2.3 (PDF 55–62 / printed 33–40) distinguish
existence from stability. At an equilibrium, its Jacobian commutes with the
**equilibrium's stabilizer**, which may be smaller than the symmetry group of
the entire law. Proposition 2.4 and Theorem 2.12 justify invariant isotypic
blocks. Equivalent irreducible copies can mix inside an isotypic block; an
arbitrary decomposition into single copies does not automatically diagonalize
the Jacobian. Theorem 2.13 identifies tangent directions to an equilibrium group
orbit that force zero eigenvalues. It does not say every zero eigenvalue is such
a symmetry direction or that a restricted stable branch is transversely stable.

**Already present and preferable for this task.**
`theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md`, “Full-law local recovery without
operator commutation” (lines 248–284), uses the actual phase Hessian and full
relative state. Its positive quadratic pencil does **not** assume the mobility-
weighted graph operator and phase Hessian commute. It removes only the two
justified common modes and proves local attraction; it expressly supplies no
basin entry for the formed sources.

The graph phase Hessian there is
`B^T diag(cos(q_e)) B`, on the actual acute cell (lines 84–108). That state-
dependent weighting is crucial. Replacing it with the bare graph Laplacian, or
with a spectrum computed at the old uniform twists, would remove a consumed
premise. The book suggests optional block organization of an already admitted
Jacobian if it saves work; it supplies no reason to replace this existing full-
state proof or start a separate eigenvalue campaign.

Section 1.7 of Symmetry (PDF 49–51 / printed 27–29) also distinguishes imperfect
symmetry due to parameter changes from stochastic disturbances. Its persistence
arguments require the relevant hyperbolicity assumptions; they give no TNFR
error radius, time horizon or basin. Retain the actual source error family and
proved comparison bounds rather than declare it effectively symmetric. Its
informal examples are motivation, not ready-made quantitative certificates.

## 4. Network geometry must be consumed by the complete law

**Book evidence.** Symmetry section 3.7 (PDF 105–108 / printed 83–86) defines a
coupled-cell system on specified product state spaces with actual coordinate
projections. Graph edges summarize dependencies but do not supply coupling
functions. Figure 3.19 (PDF 108 / printed 86) shows triangular arrangements with
cyclic versus dihedral symmetry depending on directions and equality of input
functions. Even bidirectionality alone does not give exchange symmetry between
two differently acting inputs.

Networks section 6.14.2 and Figure 6.19 (PDF 158–160 / printed 145–147) show two
drawings of the same graph. Their displayed distances differ while adjacency
is unchanged. The Laplacian quadratic form also appears as a graph drawing
objective; this does not identify a drawing coordinate with dynamical form or
an energy with physical energy.

**Already present.**
`theory/nodal/LOCALIZED_ORGANIZATION_GEOMETRY.md#geometry-and-formation`
(lines 19–49) separates support/coefficient geometry, relative form/phase
geometry and observation/boundary geometry. `theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md`
(lines 23–44) routes conductance versus path length to the shared
`src/tnfr/physics/_edge_semantics.py`. These are the right existing owners.

**Concrete clarification.** In the next declaration, explain what the second
port changes in incidence, live degrees and allowed phase-current balance,
using the existing two-port owner. Do not motivate it only with a more complex
picture, a new triangle, or a cycle count. Static two-port compatibility already
provides internal deformation and opposing contact currents at zero interface
period. The active missing capability is entry of the actual formed source into
a predictive organization, not rediscovery of those static properties.

The negative result at
`theory/nodal/SINE_FORMED_CLASS_CONTACT.md#sine-formed-joint-cycle-eligibility`
concerns one relative cycle candidate on a one-bridge support. It does not rule
out feedback or all joint properties. Conversely, adding a graph cycle cannot
by itself establish identity. Neither these chapters nor the selected book
sections show fractality or an iteratable organization rule. A later scale claim
must retain the domain, boundary and information obligations below.

## 5. Spectral methods: useful premises, not a universal shortcut

**Book derivation.** Networks section 6.14 (PDF 155–166 / printed 142–153) fixes
the undirected combinatorial Laplacian and derives its quadratic form,
nonnegative spectrum and one constant kernel direction per connected component.
The random-walk normalization and the resistor analogy are separate models.
Neither transfers its interpretation merely because a TNFR formula has the same
matrix shape.

Sections 17.2–17.3 (PDF 698–710 / printed 685–697) begin with complete pairwise
node equations and derive their actual fixed-point Jacobian. The simple master-
stability decompositions then require special uniform fixed states and shared
coupling derivatives, with adjacency- or Laplacian-factorized structure. For
multiple variables, the small node-space matrices still depend on the complete
local law. An arbitrary multi-channel or state-dependent network does not
inherit that separability.

**Scope limits at TNFR.**

1. The canonical pressure's neighbor-resultant phase channel is not globally
   a sum of independent pairwise functions. Newman section 17.2 explicitly
   excludes terms involving more than one neighbor in a single coupling term.
   Thus that chapter's pairwise setup cannot silently stand for every canonical
   TNFR channel. The selected separate sine comparison has its own pairwise
   rows and its own owner; it is not the whole runtime.
2. The two-port winding target is not a common identical nodal state; phase
   derivatives depend on actual edge gaps, and degrees change at ports. The
   simple master-stability factorization is not admitted for it. Use the
   existing full-law proof unless a separate exact factorization is established.
3. A bare symmetric Laplacian and a row-normalized generator have different
   metrics and conserved averages. The current owners already use actual
   degree mass, full post-contact degrees and the correct common modes. A
   numerical spectral gap alone is not a source-entry certificate, a nonlinear
   remainder bound, a selected relaxation time or an observation model.

**Reader precautions in the books.** The broad wording about directed
Laplacians in Networks printed 143 is a restriction of that presentation, not
an impossibility theorem for directed generators. Its eigenvector expansion
on printed 688 presumes a usable eigenbasis; a nonsymmetric matrix need not
have one. Its sentence on printed 695 that nonnegative-real-part eigenvalues
mean an equilibrium is not stable must not be imported as a nonlinear verdict
in a zero/neutral case. The source's strict-negative test supports hyperbolic
local attraction; unresolved neutral directions need their own treatment.
Symmetry printed 39 explicitly omits technical hypotheses from Theorem 2.14;
that abbreviated bifurcation statement is not an importable TNFR certificate.
These are application safeguards, not identified errors in maintained TNFR owners.

## 6. Symmetry reduction, closure and repeated scale changes

The books give useful ways to organize state space and linear operators, but
they do not make every orbit-average an autonomous macrostate. The repository
already distinguishes three different operations:

| Operation | Existing exact owner or controlled example | Information obligation |
| --- | --- | --- |
| Restrict a reference to a fixed symmetric family | `SINE_TWO_PORT_CAPTURE.md#sine-two-port-reference-folding` | Prove field and nominal-source invariance; compare actual off-family states separately. |
| Identify states modulo a justified action | `TNFR_SCALE_GEOMETRY_AND_BRIDGE.md`, P5 reflection-invariant section; `DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state` | Preserve the valid invariant image, boundary/observation compatibility and equality of projected rates on each fiber. |
| Discard fine coordinates for a smaller predictive surrogate | `SINE_REDUCED_CLASS_PORTS.md#sine-reduced-class-ports`; `TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#1-pure-epi-coarse-graining` | Exact intertwining where proved; otherwise initialization, memory or explicit domain/horizon-dependent omission bounds. |

The pure-EPI reversible closure proof (scale owner lines 89–148) uses an actual
weighted projection and an intertwining condition. It is not a generic
consequence of graph clustering. The P5 invariant description removes a discrete
duplication while retaining the intrinsic generic dimension; fewer orbit labels
need not mean fewer continuous state variables. The directed-form owner states
that equivariance is sufficient but not necessary for nonlinear projectability,
so demanding it as a universal necessary closure test would strengthen the
repository incorrectly.

For possible fractality or repeated composition, retain
`theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#scale-dependent-effective-laws`
(lines 46–87): output of one reduction need not be another object in its input
family. A second step needs compatible domains, state maps, interfaces, clocks
and event descent. This review supports reusing that responsibility, not opening
an additional hierarchy campaign or asserting self-similarity from one successful
composition.

## Prioritized disposition

| Priority | Disposition | Proposed use | Existing destination / why it is bounded |
| --- | --- | --- | --- |
| 1 | **Incorporate** | Explicit action on law, actual source, residuals, boundary and observation in the upcoming two-port admission. Distinguish fixed-model symmetry from relabeling covariance. | Existing source/event admission and `SINE_TWO_PORT_CAPTURE` comparison; a short local table, not a new general API. |
| 1 | **Clarify** | Exact reference restriction, nonlinear observation quotient and approximate predictive reduction are separate claims. State which one each reused result provides. | Existing CAPTURE, REDUCED_CLASS_PORTS and scale owners already establish the distinction. Add routing where needed, preserve proofs. |
| 1 | **Incorporate** | Screen the sole prospective relational witness for symmetry-forced invisibility before freezing it; preserve a matched disconnected control. | Current F1–F4 admission. This only constrains candidate selection; it selects no probe or positive gap now. |
| 1 | **Clarify** | A finite membership collar is not exact finite-time attainment of a symmetric equilibrium. State interval, uncertainty and support-event scope. | Current identity declaration plus ontology/state-map owners; Proposition 1.18 supplies a methodological warning, not an obstruction to all formation. |
| 2 | **Clarify** | Retain the full phase Hessian, degree metric and transverse modes; use isotypic blocks only after proving the actual stabilizer action. | Existing two-port recovery proof already avoids commutation and bare-spectrum assumptions. No replacement theorem or eigenvalue sweep. |
| 2 | **Incorporate** | Explain geometry through actual support, coefficients, phase configuration and observational boundary, using one shared account. | `LOCALIZED_ORGANIZATION_GEOMETRY#geometry-and-formation` and `_edge_semantics.py`; no duplicate geometry chapter. |
| 3 | **Defer** | Normal forms, bifurcation catalogues, master-stability parameter scans and more extensive group classifications. | Need a selected parameter family, complete law, genericity/domain conditions and a consumer not already served by the current recovery proofs. |
| 3 | **Defer** | Iterated effective laws or fractal organization. | Existing scale owner requires an admissible second reduction and predictive comparison; one-level composition is insufficient. |
| — | **Reject as an inference** | Drawing symmetry implies dynamical symmetry; graph cycle implies identity; restricted stability implies full-state stability; spectral similarity implies physical equivalence. | None follows from the selected texts or current owners. This rejects those shortcuts, not their separately testable scientific subjects. |
| — | **Reject as redundant work** | New generic symmetry/network registry or recomputation of existing two-port static deformation just to illustrate the books. | Existing theory, execution owners and frozen evidence already cover those responsibilities. |

## Corrections and limits

No verified mathematical error in the inspected maintained owners was identified.
The strongest findings are scope clarifications and opportunities to reuse existing
proof structure. This does not certify the entire repository or the unread book
chapters. No proposed source-entry, membership, response or transit result has
been calculated, and this note neither admits that calculation nor changes the
active question. Physical identification remains subject to the existing gate.

The concise synthesis worth carrying forward is: **use symmetry to expose the
information and constraints an observation retains; use complete dynamics and
bounds to decide whether an actual organization acquires and preserves a
predictive capability.** The current owners already supply much of this
machinery. The next benefit is a more precise source/identity declaration,
followed only by its separately reviewed discriminating study.

## Inspected source fingerprints

The following byte hashes identify the source versions reviewed. They are
provenance, not scientific validation. Repository paths are relative to the root.

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `bibliografia/The symmetry perspective.pdf` | 19828011 | `a26bda48d21e6bfa3d61435b3b2fe72d5087c34c5af6775b43326e24026e45b7` |
| `bibliografia/Networks (Second Edition).pdf` | 25862122 | `47ca3a1cefe43888ed5f93e2b0e49cb71ec2b915c8be040fb5c774f2d742413e` |
| `theory/nodal/DERIVED_FORM_PHASE.md` | 91427 | `0284bf18f8e952480c64fbd2522df220d3723b0420dc7fdcc95bfbdb474903c0` |
| `theory/nodal/SINE_REDUCED_CLASS_PORTS.md` | 26150 | `5c9c84236a665b0b3269d66c146f33f3b0d67ec59c0195a9a93f8b313457b995` |
| `theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md` | 30415 | `197466f21fcabb6d145a87bbc32ba265b94a27f214c524a0b1f58dcfcf83565e` |
| `theory/nodal/SINE_TWO_PORT_CAPTURE.md` | 22740 | `0efe97bca625d62af70d4636fdf1567e3fd1b4e2517e12372d64e2c581897393` |
| `theory/nodal/LOCALIZED_ORGANIZATION_GEOMETRY.md` | 38535 | `36edd174c91509264454e06d835cc768ca308b939f4070ded95ebcc23b975902` |
| `theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md` | 132818 | `3633a66d89effc64f130b2058b26a8621dc7e514889b0d4442dad6d33de9791d` |
| `theory/RESEARCH_PROGRAM.md` | 86282 | `3b35270f23021370fae5b40e2c90fbe65643ed845c6dcbd61f570f64a45e0adf` |
| `theory/nodal/SINE_FORMED_CLASS_CONTACT.md` | 32645 | `e8bbcd2cc792abd9a555a7687b9a50566ce2ef075665dc94f506d4dc328da227` |

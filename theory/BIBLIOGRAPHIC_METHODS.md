# Bibliographic methods for TNFR foundations

The supplied books help choose and combine methods for a precise TNFR
question. Their strongest common lesson is to retain the relationship between
the complete dynamics, actual preparation, selected observation and the claim
being made. A compatible shape, an acquired organization, its persistence and
a sufficient description of its response require different evidence.

This chapter owns the literature-to-method synthesis. Definitions and proofs
remain at their linked owners; the [research program](RESEARCH_PROGRAM.md#research-current-question)
alone selects work. The applications below are methodological judgments,
not new TNFR theorems or physical identifications. The
[final synthesis](#bibliographic-final-synthesis) consolidates the three
earlier selective reviews. The [source-transport reading](#bibliographic-source-transport)
adds two books for source-to-regime obligations. The
[general reuse assessment](#bibliographic-general-synergies) connects their
reduction and numerical methods to shared engine responsibilities.
The [nine-source integration](#bibliographic-nine-source-integration)
revisits the complete selected methodological bibliography and derives a
source-aware accuracy budget for structural simplification.
The [held-input extension](#bibliographic-held-input-structure) then preserves
the current model's sustained relative form while bounding transient error.

<a id="bibliographic-next-method-decision"></a>
## Current method coverage and further literature

<a id="formal-operator-languages"></a>
### Operator alphabets, grammar and structural rewriting

The selected primary-source comparison is complete within the reading scope
below. Its strongest contribution is a formal account of how local actions can
change the opportunities for later actions, together with tests of when a
collective description remains sufficient in a new context. It supports the
operator-first direction already selected for TNFR. It supplies neither a
replacement ontology nor evidence that the thirteen operators are universally
complete. No new executor, scheduler or numerical campaign was introduced.

#### Primary sources and reading limits

| Primary source | Selected reading and specific use |
| --- | --- |
| Heckel, [Graph Transformation in a Nutshell](https://drops.dagstuhl.de/entities/document/10.4230/DagSemProc.04101.10), 2005 | Complete twelve-page tutorial; especially sections 2.2–2.3 and 3.1–3.4. Typed attributes, actual matches and application conditions distinguish a structural rule from an unqualified graph edit. |
| Ehrig, Habel and Lambers, [Parallelism and Concurrency Theorems for Rules with Nested Application Conditions](https://people.cs.umu.se/drewes/hjk60/essay_6.pdf), 2009 | Sections 2–5, printed pp.110–130: Assumption 1, Shift Lemmas, Definitions 8–11 and Theorems 1–3. The required category, factorizations, coproducts, residual matches and conditions are part of the transfer, not optional implementation details. |
| Lambers, Ehrig, Prange and Orejas, [Embedding and Confluence of Graph Transformations with Negative Application Conditions](https://www.hpi.uni-potsdam.de/giese/misc/publications/pdf/LEPO08_confluenceWithNACs.pdf), 2008 | Sections 3–4, Theorems 1–3, Definitions 9–10 and Example 5. A joining sequence for a minimal overlap need not remain admissible in a larger permitted context; the sufficient local-confluence result requires strict NAC-confluence. The separately cited general technical-report proofs were not read. |
| Milner, [Pure bigraphs](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-614.pdf), 2005 | Printed pp.15–21, 23–29, 33–36, 40–45; selected definitions and results in sections 3–9 through Corollary 9.14; Figure 3 inspected. Theorem 4.6 assumes a wide reactive system with relative pushouts and its standard transition system. Its externally referenced proof, the appendix, full CCS case study and sorting theory were not audited. |
| Jensen and Milner, [Bigraphs and mobile processes (revised)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-580.pdf), 2004 | Printed pp.68–73, section 11 through Theorem 11.8 and its proof. Scoped binding relates place and link constraints; this is not object ownership or arbitrary TNFR history binding. The complete report was not read. |
| Sevegnani and Calder, [Bigraphs with sharing](https://eprints.gla.ac.uk/102874/7/102874.pdf), 2015 | Printed pp.43–51, 54–56; structural/matching definitions through Definitions 13–16. Shared placement offers an alternative to ordinary tree nesting. The complete matching algorithm, SAT implementation and applications were not audited. |
| Milner, [Bigraphs: a Model for Mobile Agents, lecture notes](https://www.cl.cam.ac.uk/archive/rm135/Bigraphs-Notes.pdf), 2008 | Lecture IV, items 49–65, printed pp.11–13: supplied rates and counting distinct reaction matches. Used to examine clock and multiplicity obligations, not to adopt a stochastic TNFR law. |
| Prusinkiewicz and Lindenmayer, *The Algorithmic Beauty of Plants*, [chapter 1](https://algorithmicbotany.org/papers/abop/abop-ch1.lowquality.pdf) and [chapter 3](https://www.algorithmicbotany.org/papers/abop/abop-ch3.pdf) | Chapter 1 sections 1.1–1.3, 1.6–1.8 and 1.10, pp.1–8, 21–35, 40–46; chapter 3 section 3.1, pp.63–70. Parallel, contextual and parametric rewriting, geometric interpretation and development controllers. Figures and whole-book proofs were not audited. |
| Měch and Prusinkiewicz, [Visual Models of Plants Interacting with Their Environment](https://algorithmicbotany.org/papers/enviro.sig96.html), 1996 | Author PDF, printed pp.397–402, principally sections 2–4 and the start of the examples. Addressed plant/environment exchanges suggest explicit source and response interfaces; later applications and validation claims were not audited. |

The author's [graph-transformation course](https://www.cs.le.ac.uk/people/rh122/MGS22/GraphTransformation.html)
served as a reading route, not theorem evidence. The complete plant book was
too large for the browser reader; the cited chapters were used. The environment
PDF was read transiently after browser extraction failed. The reading records
retain URLs, coverage and access limits, not copies of books or extracted text.
These selected passages do not constitute an exhaustive review of the fields.

#### What transfers, and what must first be represented

The common distinction is between a well-formed request, its admission in the
current state, its complete effect, and its occurrence policy. TNFR already
has owners for all four, with unresolved constitutive choices. The useful
comparison is therefore against actual actions and consumed state, not a
translation of glyphs into another alphabet.

The comparison also needs the right types: an L-system module or bigraph node
can represent a constituent, while a TNFR glyph names an action. The natural
candidate correspondence is NFR state to the rewritten structure and operator
instances to guarded transformations. Equating component labels with operator
names would conflate the object with the mechanism that changes it.

| Method | Concrete TNFR use | Remaining obligation |
| --- | --- | --- |
| Guarded attributed rewriting | Describe the numerical and structural domain of an operator, its actual target binding, resets and retained history. Pull successor guards back through an ordered word. | An embedding must preserve and reflect admission and effects, or explicitly offer only one-way simulation. Numerical thresholds and Python callbacks are not automatically finite structural application conditions. |
| Dependency and conflict analysis | Include absent-support conditions, neighbor reads, lifecycle and refresh effects when deciding which actions interfere. | Reuse existing conservative stage resources. A fixed list of operator-name pairs cannot decide all state-dependent conflicts. Preserve the declared sequential or snapshot schedule. |
| Separation of nesting and connectivity | Keep THOL ancestry, collective membership, potential access and realized UM transport distinct. | Genealogy need not be containment. An ordinary forest cannot directly represent overlapping memberships; sharing variants exist but require their own semantics and theorem hypotheses. Direction, multiedges, weights, attributes and history need a specified encoding. |
| Contextual equivalence | Require a collective NFR description to preserve future permitted boundary actions and responses. | Declare the interface and context class; prove admission, outputs and successor closure. Importing a congruence theorem requires its actual structural hypotheses. |
| Parallel developmental rules | Make the generation boundary and all-prestate reads explicit, including when a newborn becomes an eligible target. | Existing TNFR stages have operator-specific merge laws. A symbolic generation is neither a physical interval nor permission to activate a newborn immediately. |
| Explicit environment exchange | Associate returned observations with the actual target, source state, boundary and clock before consumption. | A reply is an input requiring admission, not authenticated structural evidence merely because a protocol returned it. Existing shared readers remain authoritative. |

There are two complementary uses. Formal composition can reduce redundant
work by deriving a successor's required preparation instead of searching
parameters blindly. Structural interfaces can show when an apparently local
rule actually consumes wider context. Neither benefit requires adding another
rule registry or changing the scientific meaning of an operator.

#### Conditional consequences integrated into the operator owner

The [guarded-action analysis](STRUCTURAL_OPERATORS.md#operator-contextual-rewriting)
contains the original TNFR-facing deductions and their proofs:

1. The domain of first `a` then `b` is `D_a intersect inverse_image(T_a,D_b)`.
   It includes the actual history and support updates. The existing birth/contact
   preparation illustrates why two IL steps admit EN's pressure gate in one
   branch while three are needed in the matched no-contact branch.
2. Deterministic actions with stable complete read/write sets commute under
   the stated disjointness and frame hypotheses. Two half-mixing EN actions
   provide an exact counterexample to replacing sequential updates by a common
   snapshot merely because target writes differ. Shared-partner UM requests
   also conflict through the isolated-partner guard.
3. The existing partial-action descent theorem extends to interleaved admitted
   boundary actions when their admission, successors and compared outputs also
   descend. This is a conditional contextual-replacement criterion, not a proof
   that any currently displayed pattern is a collective NFR.
4. A transport-only observation cannot uniquely select an unseen isolated child
   without additional distinguishing information. Actual parent/child metadata
   can supply that information. Consuming it as candidate access is a policy
   choice distinct from creating transport or requiring an event.
5. Identical untimed rules admit distinct clocks. Under a separately declared
   finite-state Markov jump model, common positive rate rescaling changes holding
   times without changing enabled actions. Match multiplicity affects rates;
   projecting it away needs a timed-state argument, not just event descent.

These are analytical consequences and uses of existing finite controls. They
were independently reviewed; no new engine trajectory was evaluated. The
categorical concurrency and bigraph congruence theorems remain external tools
with unproved representation bridges to the full TNFR runtime.

#### Research decision and limits of the analogy

State-derived candidate access has since been integrated in the
[birth-association consumer](STRUCTURAL_OPERATORS.md#coupling-birth-association-access).
It uses actual NFR-held birth information, abstains on empty or ambiguous
access, and retains live UM admission. Parent selection, invocation and the
access policy remain supplied; ancestry is neither contact nor an obligation
to couple. That completed dependency must not be mistaken for another active
task. The [program](RESEARCH_PROGRAM.md#research-current-question) alone selects
the next operator function.

Next in usefulness are guarded compositions and context-preserving collective
interfaces, followed by proved execution optimizations. For the unresolved
roles of Expansion, Contraction, Mutation and Recursivity, these methods help
specify what is created, retained, transformed or observed. They do not decide
that synthesis means deleting nodes, that copying state creates memory, or
that a scalar form is sufficient at the next scale.

Repeated rules can produce repeated structure without establishing geometric
fractality. A fractal claim needs a geometric interpretation, declared finite-scale
or limiting object, scale range and stated similarity or dimension property. A biological growth
image also supplies no TNFR stability, autonomous occurrence or physical
identification. Conversely, a finite alphabet can have rich generative power;
its size alone establishes neither insufficiency nor completeness. The open
TNFR question is the faithful generative reach of its specified actions under
their grammar, state domains, access and occurrence laws.

#### Existing methodological coverage

The selected integrations cover local stability, signed boundaries, symmetry,
projection, memory, scaling, time-scale separation and viability. The
[two-port exclusion](nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-original-period-exclusion)
now closes one source-dependent selection question. Its unresolved outgoing
destination does not automatically make another trajectory the best priority.
A bounded literature assessment may also question a premise, connect existing
results or formulate a testable hypothesis before a method has failed.

Judge a reading by its concrete contribution and limits: a discriminating
hypothesis, a controlled simplification, a missing proof tool or a counterexample.
Keep the scientific question and its preserved evidence explicit. Neither a
bibliographic addition nor an appealing example automatically selects a new law,
controller, numerical campaign or physical interpretation. The research program
owns the live priority; the joint acquisition/maintenance analysis below retains
its historical scope.

The recent reversible EPI integrations remain useful quantitative results:
a reduced prediction can preserve a held profile and retain transient error.
They do not transfer to the nonlinear phase/form law. Match source, boundary,
clock, initialization and output before combining any of these methods.

<a id="bibliographic-hybrid-guarded-lifecycle"></a>
### Hybrid solutions and guarded lifecycle composition

Two additional local books support the Silence/Transition question. Their
title pages, relevant complete passages and selected equations/diagrams were
checked. This is selective reading, not a whole-book review. The file inventory
now contains fourteen local PDFs, including the historical TNFR manuscript;
that count is not a count of fully audited sources.

| Source | Complete selected reading; printed pages followed by one-based PDF pages |
| --- | --- |
| Rafal Goebel, Ricardo G. Sanfelice and Andrew R. Teel, *Hybrid Dynamical Systems: Modeling, Stability, and Robustness*, Princeton, 2012, ISBN 9780691153896; `bibliografia/Hybrid Dynamical Systems.pdf`, 227 PDF pages | pp.12-14 (27-29), 25-39 (40-54), 83-85 (98-100), 101-102 (116-117), 117-125 (132-140), 132-134 (147-149). Solution definition 2.6; propositions 2.10-2.11; logical-memory examples 1.7/4.18; definitions 5.8-5.9; basic conditions 6.5, theorem 6.8, lemma 6.9, proposition 6.10, definition 6.29 and theorem 6.30. The proof of 6.13 continues outside the reading; only the statement of 6.31 was read. Chapters 3 and 7-9 were not audited. |
| Hartmut Ehrig, Karsten Ehrig, Ulrike Prange and Gabriele Taentzer, *Fundamentals of Algebraic Graph Transformation*, Springer, 2006, ISBN 9783540311874; `bibliografia/fundamentals-of-algebraic-graph-transformation_compress.pdf`, 373 PDF pages | pp.54-55 (60-61), 59-61 (65-67), 151-165 (151-165), 237-249 (231-243): sections 3.4.1/3.4.3, chapter 7, sections 12.1-12.2 through remark 12.14. Includes proofs of theorems 7.13, 7.17 and 7.23; not the whole categorical infrastructure, concurrency/confluence proof chapters or applications. PDF production metadata dated 2008 is not the edition date. |

Source hashes are respectively
`39f262b1c7730b837742bda8abc6fe4771f1b3cef6e2bb2b45235555489a6a31` and
`31c5ec86c5c137c71f171deb5464d21357fa1d240a3d97e3dfa19b666c40b4b7`.
Visual checks included hybrid PDF pages 4,5,29,44,45,48,49,99,100,135,148
and graph-transformation PDF pages 161,164,233,240. Book text and rendered
pages remain private and Git-ignored. These locators identify what was read;
neither an extracted whole-file text nor a source hash establishes that every
page or proof was examined. Printed-page continuity was checked within the
selected graph-transformation passages; its PDF offsets vary, and continuity
of the rest of that copy was not audited.

#### Complementary contributions

The hybrid model distinguishes flow data `(C,F)` from jump data `(D,G)` and
continuous time from the jump index. A solution must satisfy each consumed
domain, while existence, uniqueness and selection of an occurrence remain
separate questions. Memory examples show why a logical mode may belong to
the state: equal continuous coordinates need not determine the next action.
The [earlier viability reading](#bibliographic-second-timescales-viability)
already included hybrid resets; the new book adds a systematic solution and
well-posedness framework rather than introducing hybrid reasoning from zero.

The graph-transformation book adds systematic backward construction of
application conditions and typed attributed examples to the already reviewed
2008/2009 papers. It distinguishes guaranteeing a constraint from preserving
one already true. Changing the permitted attribute morphisms can change the
meaning of a constraint. The later paper's nested-condition concurrency
premises cannot be replaced by an unconditional overview theorem.

For TNFR, the immediate common use is the existing ordinary partial-map
calculation: for action T with full domain D and desired postcondition Q,
require `z in D` and `Q(T(z))`; apply it recursively at every prefix. A
preservation implication `I(z) => I(T(z))` does not admit invalid entry states.
Recovering AL eligibility is a desired postcondition, not an invariant that
SHA should preserve. Own grammar, lifecycle, history, configured factors,
actual targets and ports remain part of the consumed state.

| Obligation | Consequence for the existing engine |
| --- | --- |
| State and guards | Retain member-level state and own context; a mean form or latency flag cannot stand for all successor conditions. |
| Full event effect | Derive NAV pressure and phase changes alongside capacity recovery, then check the actual AL/EN/RA domains. |
| Preservation versus continuation | Verify form/support at the SHA jump separately from later permitted motion and the existence of a recovery prefix. |
| Time and occurrence | Keep runtime time, own operator-step count and UTC lifecycle metadata distinct. A supplied finite schedule is not an endogenous clock. |
| Failure alternatives | Retain capacity, pressure, phase or grammar obstruction and atomic rejection; an admitted event need not enable the requested successor. |

These obligations directly produce the
[retained lifecycle calculation](REMESH_STRUCTURE_AND_MEMORY.md#retained-silence-reactivation)
using shared stages. No new rewrite executor, scheduler, registry or
constitutive operator was needed to state the discriminator.

#### Limits of theorem transfer

Hybrid basic conditions 6.5 require closed flow/jump sets and suitable outer
semicontinuity/local boundedness, with convex flow values; they do not require
convex jump values. Theorems 6.8/6.30 concern well-posedness under their full
hypotheses, not guaranteed recovery or asymptotic stability. TNFR's strict
threshold guards cannot simply be assumed to satisfy those hypotheses.
For example, with no flow, `D=(-infinity,L)` and `G(x)=x+b`, admitted
one-jump paths starting at `L-1/n` approach a jump from L that the same model
forbids. This original elementary example concerns that exact jump model,
not every possible encoding of TNFR. Closing a guard or adding hysteresis
would change the selected law and requires a separate justification.

Likewise, an attributed graph encoding must preserve and reflect all consumed
state and effects before categorical theorems can be imported. Fixed support
and a finite word do not prove an encoding for variable-size births, arbitrary
history or collective replacement. Neither book supplies TNFR's occurrence
law. The useful immediate optimization is deriving a small discriminating
preparation from successor guards, with explicit margins, rather than adding
a search over unexamined parameters.

<a id="bibliographic-strategic-pivot"></a>
### From a selected trajectory to a generative organizational mechanism

The global objective remains the development of TNFR structure and dynamics,
oriented by the open matter-emergence hypothesis. The methodological framework
is **organization as a regime that is acquired, persists and supports a
predictable response under one complete causal realization**. Formation,
maintenance, action and an effective description remain distinct obligations.
The bibliography supplies tools for connecting them; it does not establish
that any particular graph, law or preparation is fundamental.

This reassessment uses the recorded selected-passage readings and the current
proof owners; it is not another full reading of the PDFs. Earlier integrations
already questioned preferred shapes, unnecessary generic interface frameworks
and unqualified reductions. Several candidate directions are therefore not
new: autonomous contact onset, constitutive nonselection, prepared identity
retention, hidden-state counterexamples and finite prediction budgets all have
scoped results. A new study must consume an actual missing implication.

The earlier relation-completion line connects acquisition of a maintained joint
identity to its continued response.
The elementary small-source acquisition and controlled source-to-response
results below support the next nonuniform-family discriminator. The
[joint-state owner](nodal/RELATION_FOUNDATIONS.md#contact-and-internal-reorganization)
already retains both internal and contact evolution; its positive-onset
examples do not prove a maintained coupled regime. Candidate access and the
constitutive function remain supplied. Additional conditions for acquisition
must be declared, since the passive class includes zero growth. Merely keeping
the monotonically increasing edge weight positive would not answer the question.

| Recorded methodological source | Contribution to the selected gate | Required model-specific check |
| --- | --- | --- |
| Wagensberg | Define what a boundary and its internal organization preserve during change. | An independent joint identity and response; a created edge or attractive shape is insufficient. |
| Strogatz, Wiggins, Kuehn, viability theory | Distinguish invariant regimes, attraction, trapping, slow tracking and actual source-to-regime access. | Complete changing-weight rows, regularity, preparation and any spectral or boundary hypotheses; local onset is not a bifurcation or capture theorem. |
| Golubitsky/Stewart, Newman, Cross/Greenside | Retain the representation, interfaces and transverse modes of the actual coupled system. | A symmetric graph or tangent does not establish acquisition, and changing weights can change the consumed normalization. |
| Pikovsky and the nonlinear-dynamics readings | Separate synchronization, internal activity and sustained response from static alignment. | Primitive angle, oscillator phase and a supplied signal have different admissions; no unaccounted clock or forcing is introduced. |
| Zwanzig and the reduction readings | Preserve hidden initialization and boundary memory in any collective prediction. | Start with sufficient joint state; a smaller description needs a declared output and exact closure or quantified error. |
| Barenblatt | Compare laws and preparations under consistent dimensionless controls. | Preserve units, source, clock, resources and observation; rescaling cannot select h or repair a missing formation premise. |

An existing reusable exclusion mechanism illustrates the intended level of
transfer. For a complete trajectory precompact in an admitted state space or
symmetry quotient, suppose its omega-limit points are stationary, H is
continuous with a proved future upper bound b, and K is closed in that same
space. If no stationary state in K has H<=b, the trajectory is eventually
absent from K. Arbitrarily late visits would give a stationary accumulation
point in K with H<=b, a contradiction. A continuous nonincreasing storage is
one way to obtain the future bound; monotonicity alone does not establish the
stationary-limit premise. Neither unique limiting equilibrium nor a uniform
departure time follows.

The [single-ring source selection](nodal/SINE_PREPARATION_PATH_SELECTION.md#sine-preparation-transient-winding-obstruction),
[continued-contact analysis](nodal/ORGANIZED_CONTACT_MAINTENANCE.md#full-contact-convergence)
and the two-port result already realize this pattern under their own laws.
This is a synthesis of existing mathematical mechanisms, not a new universal
TNFR law or a positive acquisition criterion. Closedness matters: the orbit
`z(t)=exp(-t)` remains in `(0,1]` although its limiting equilibrium zero is
outside that nonclosed set. A quotient, later events or persistent inputs
require re-admitting H, K and the complete limiting dynamics.

The [fresh joint-organization reading](#bibliographic-joint-organization-reading)
below now connects this gate to additional selected passages and concrete
resource, regularity and retained-response results. The scientific decision
remains solely in the program; this synthesis neither activates an experiment
nor admits a physical mapping.

<a id="bibliographic-joint-organization-reading"></a>
### Fresh reading for joint organization, resources and retained response

This pass returns to the supplied PDFs after the strategic change. It reads
129 complete selected pages across the eleven methodological books and the
historical TNFR manuscript, including explicitly recorded revisits, with
27 rendered page checks. It is not a whole-book rereading or a claim that
every possible connection has been exhausted. Navigation extractions and
search snippets are not counted as complete-page reading. All extracts and
images remain private under `bibliografia/.reading-joint-20261101/`.

| Source | Complete printed pages read (one-based PDF pages) | Specific contribution and limit |
| --- | --- | --- |
| Strogatz, third edition | 267-273 (282-288), 347-353 (362-368), 378 (393) | Full-field feedback, trapping and bifurcation hypotheses; negative divergence alone supplies no invariant regular domain. |
| Wiggins, second edition | 112-119 (133-140), 242-244 (263-265) | Nonautonomous initialization and attraction quantifiers; a frozen-contact attractor is not a theorem for an evolving contact. |
| Kuehn, *Multiple Time Scale Dynamics* | 384-391 (402-409), 633-637 (651-655), 695-699 (713-717) | Moving-reference terms, changing geometry and explicitly supplied adaptive-network rows; no universal adaptation law is inferred. |
| Aubin/Bayen/Saint-Pierre, *Viability Theory*, second edition | 392-398 (415-421), 416-421 (439-444) | Exit-time and robust-domain obligations; a pointwise passive inequality does not establish all-time regularity or identity retention. |
| Golubitsky/Stewart, *The Symmetry Perspective* | 41-48 (63-70) | Representation-dependent invariant/equivariant generators retain coefficient freedom; symmetry does not uniquely select h. |
| Newman, *Networks*, second edition | 108-109 (121-122), 118-120 (131-133), 225-231 (238-244) | Weighted/temporal representation and sparse adjacency; candidate-zero relations cannot be deleted when their support row can activate them. |
| Barenblatt, *Scaling* | 37-44 (52-59) | Match the whole governing problem before normalizing; a model-family scaling differs from changing one contact within a fixed background. |
| Cross/Greenside | 57-62 (73-78), 463-468 (479-484) | Full boundary/state stability and sparse numerical structure; A-stability does not certify nonlinear phase regularity, passivity or accuracy. |
| Wagensberg, *La rebelion de las formas* | 41-47 (34-40), 54-59 (47-52) | Persistence, individuality and observation scale motivate an identity/response contract; biological criteria are not imported as a TNFR theorem or score. |
| Pikovsky/Rosenblum/Kurths, *Synchronization* | 156-159 (178-181), 168-171 (190-193) | Matched interventions distinguish hidden response; a uniform equilibrium is not a self-sustained oscillator merely because phases agree. |
| Zwanzig, *Nonequilibrium Statistical Mechanics* | 188-192 (199-203), section 9.5 | Nonlinear observables need correlations or retained state for closure; the deterministic relation-memory witness needs no added stochastic assumptions. |
| Historical `TNFR.pdf` | 110-115 (same PDF pages) | Historical NU'L/T'HOL intuition about reorganization and maintained identity; no historical assertion replaces the current operator contracts or a proof. |

The source hashes, actual render checks, limits at page boundaries and
original analyses are retained in
`artifacts/bibliography-joint-organization-20261101/` as
`stability-reading.md`, `geometry-reading.md` and `identity-response-reading.md`.
Referenced proofs outside these passages were not silently treated as newly
read sources. Cross/Greenside page 468 and historical TNFR page 113 are
recorded revisits. The broad previous results on memory, source access and
constitutive nonselection are reused, not relabeled as discoveries.

The useful integrations are conditional deductions from TNFR's existing
fully weighted rows, motivated and checked using these methods. Their proof
owner is [relation foundations](nodal/RELATION_FOUNDATIONS.md#passive-contact-resource-budget):

| New integration | Mathematical or implementation consequence | Remaining obligation |
| --- | --- | --- |
| Shared loss and integrated relation growth | `S+kappa beta sum(a_e)` is nonincreasing under the stated stronger allocation/gate condition. The exact [ledger](../src/tnfr/physics/passive_contact.py) rejects duplicate endpoint funding and supplies conditional resource exclusions. | Costs/losses must belong to the same complete state; passivity alone does not supply the stronger condition or acquisition. |
| Storage as a phase-domain bound | `Re Z_i>=d_i^bg-S_0/beta` yields an explicit positive-resultant margin and global continuation for a low-storage subclass. After positive contact connects the background, a compact quotient and stationary omega-set follow. | No selected equilibrium, phase consensus, uniform arrival time or acquired identity follows. A regular non-consensus counterexample remains. |
| Whole-model common-weight scaling | Scaling background and candidate weights together gives an exact model-family conjugacy and a justified constant reference normalization. | Scaling a alone or changing normalization with time is a different operation; absolute weight readouts need their own map. |
| Relations as retained response state | Identical nodal equilibria with different relative contact weights respond differently to a matched form perturbation. The engine control reuses native EPI pressure at phase consensus. | Prepared counterexamples do not determine which relation profile an actual source acquires; weighted nonlinear phase execution is not installed. |
| Moving geometry and loss of the initial gap | The moving phase mean retains its weight-derivative term; a single bridge has `lambda_2(B)<=a(1/n_L+1/n_R)`. | A fixed-contact reduction cannot erase the relative mode or relation coordinate at zero contact. Use the full joint Jacobian. |

The numerical integration is deliberately tied to these consumers: one
admitted set of nodal losses feeds all contacts, shared exact scalar admission
prevents boundary rounding, and the existing first-contact controls now consume
the reusable ledger and domain bound. No second normalized-neighbor kernel,
new integrator, assumed speedup or generic reduced-state framework is introduced.
The sparse owner already handles normalized differences and cancellation;
a future weighted field reader should reuse it after that consumer is admitted.

These results motivated the subsequent predictive-family proof below: a
single exact contact weight cannot be the attracting identity when a continuum
of relation equilibria exists. Further reading targets a specific remaining
proof gap; the [program](RESEARCH_PROGRAM.md#research-current-question) alone
selects that work.

<a id="bibliographic-contact-identity-reading"></a>
### Selective second reading resolves the local-attraction/source-entry distinction

The second pass followed a precise difficulty: a local attracting family at
positive contact need not be reached from a zero-contact source. It read
15 complete selected pages, including three deliberate revisits, with five
rendered formula checks. It did not reread all books. Private extracts/renders
remain under `bibliografia/.reading-contact-identity-20261102/`; source hashes,
reading boundaries and analysis are in
`artifacts/joint-contact-identity-20261102/selective-reading.md`.

| Source | Complete printed pages (one-based PDF pages) | Applied distinction |
| --- | --- | --- |
| Wiggins, second edition | 246-248 (267-269), 263-265 (284-286) | Theorem 18.1.3(ii) gives local exponential tracking when center dynamics is stable. The first three pages revisit earlier coverage. An explicitly verified equilibrium family with all center directions retained is preferable to an assumed isolated attractor. |
| Kuehn, *Multiple Time Scale Dynamics* | 40-48 (58-66) | Normal domination, invariant-manifold hypotheses and stable fibers must be matched to the full model. A truncated contact interval is not automatically a compact boundary-free invariant manifold; no fast-slow parameter is supplied by a vanishing graph gap. |

Wiggins refers the theorem proofs to Carr, and Kuehn refers the general
persistence proofs to Fenichel and related sources; those external proofs
were not newly audited. Kuehn theorem 2.4.2 continues beyond the selected
pages and was not consumed. The earlier symmetry and Zwanzig analyses were
reused to retain relation state and initialization; no new quotient or
stochastic projection required another generic reading of those PDFs.

The [model-specific proof](nodal/RELATION_FOUNDATIONS.md#passive-contact-family-stability)
checks the full node/relation Jacobian. All transverse modes are stable,
while common shifts and relation coordinates span the equilibrium family's
neutral tangent. A direct local Lyapunov argument retains its moving-metric
term and proves return to a source-dependent family member. This supplies
the theorem's local content without importing an unverified global Fenichel
claim. It still gives no source entry at a=0.

The complementary [fixed-reference trapping proof](nodal/RELATION_FOUNDATIONS.md#passive-contact-zero-source-trapping)
resolves that entry issue for a separately declared small-source class. Its
nonlinear and changing-degree terms are controlled by the same edge differences
as the quadratic dissipation, uniformly through zero contact. No division by
the vanishing inter-block gap is needed. Positive onset and the complete
Lyapunov argument then give consensus-family acquisition and local response
under both existing support laws. This is a new conditional deduction from
the model, not a theorem copied from either book.

The engine's [exact source-bound consumer](../src/tnfr/physics/passive_contact.py)
uses rational bounds on the common metric at the explicitly fixed unit
parameters, retaining phase lifts and strict admission. Its controls separate
small-source admission from positive onset, and the initial-state integration
reuses native disconnected fields. No weighted nonlinear trajectory is run.
The original amplitude-one source is merely outside this sufficient proof;
prior constituent formation and a uniform response floor are not derived.

This bounded reading reached its stopping criterion: a usable local tool,
a separate source-access proof and precise remaining scope. It led to the
preparation-versus-response question addressed in the targeted extension below.
Further reading remains tied to a named missing implication, rather than an
automatic repetition of the whole bibliography.

<a id="bibliographic-response-extension-reading"></a>
### Scaling, component geometry and response state extend the acquired-organization result

The next targeted pass read **61 complete selected pages across seven books**,
with 20 rendered checks, beyond their previously recorded ranges. This is
selected-passage coverage, not seven whole-book audits. The three original
notes and independent cross-reviews are retained under
`artifacts/bibliography-response-extension-20261102/`; they bind the raw PDF
hashes, reading boundaries and mathematical deductions. Extracts/renders
stay private under `bibliografia/.reading-response-extension-20261102/`.

| Source | Complete printed pages (one-based PDF pages) | Tool and boundary consumed |
| --- | --- | --- |
| Kuehn, *Multiple Time Scale Dynamics* | 240-254 (258-272) | Initial layers, overlap and composite expansions distinguish fixed-time approximations from controlled future limits. Section 9.1 is complete; the section 9.2 example continuing beyond 254 is not consumed as a completed solution. |
| Barenblatt, *Scaling* | 69-81 (84-96) | Complete chapter 3 links source information, governing balances and admissible scaling. Its fluid model and empirical illustrations are not imported as TNFR laws or observations. |
| Newman, *Networks*, second edition | 133-141 (146-154), 165-167 (178-180) | Components/cuts identify a funded spanning condition; normalization and the degree relation clarify which weight information response retains. The weighted max-flow continuation and an unqualified directed spectral remark are not used. |
| Golubitsky/Stewart, *The Symmetry Perspective* | 171-175 (193-197) | Equilibrium group orbits, relative equilibria and normal directions differ. Neutral contact coordinates can change response and are not automatically symmetries of a fixed model. Cited normal-bundle proofs were not independently reread. |
| Cross/Greenside | 81-88 (97-104) | Linear degeneracy, nonlinear selection and pattern instability require the actual law and unstable modes. More nodes, cycles or visual symmetry do not alone create a new nonuniform TNFR attractor. |
| Zwanzig, *Nonequilibrium Statistical Mechanics* | 138-142 (149-153) | Response magnitude, relaxation and infinite-time integrability have separate premises. The quantum/thermal identities on these pages were inspected but are not transferred to the deterministic TNFR model. |
| Pikovsky/Rosenblum/Kurths, *Synchronization* | 153-155 (175-177) | Event and analytic-signal phases are observation constructions; a form signal does not determine TNFR's primitive phase without a model-specific bridge. The paragraph continuing on 156 was not newly read. |

The scaling group read 28 pages/six renders, geometry 22/six, and response
11/eight; the Newman selections are disjoint. Notes identify page-boundary
continuations and distinguish navigation extracts from substantive reading.
Consumed formulas affected by scan OCR were checked visually. External
papers cited by the books were not audited, and no quoted historical assertion
replaces the complete current model.

The integrations are deductions for that model, rather than imported physical
claims. Their common proof owner is
[relation foundations](nodal/RELATION_FOUNDATIONS.md#passive-contact-source-response):

| Integrated consequence | Justified extension | Remaining boundary |
| --- | --- | --- |
| All-time loss partition | The two-P2 seed obeys `a_infinity=epsilon^2/4+O(abs(epsilon)^3)` with an explicit remainder under both existing laws. An accumulated edge-energy bound controls any slow tail, avoiding an unjustified interchange of limits. The exact reader supplies contact, gain and tangent-rate bounds. | This is a geometry/allocation-specific leading law; finite-amplitude destinations need not coincide, and tangent rates are not arrival deadlines. |
| General source resources and delayed phase onset | Actual initial storage and the common metric bound the final contact for other admitted source shapes. Internal phase can first drive form and then fund contact at cubic order in time. | Relative offsets between internally relaxed blocks alone cannot activate these loss-funded contacts. Prior preparation and candidate access remain supplied. |
| Finite multicomponent acquisition | Positive initial allocations on contacts spanning the background-component graph suffice for joint acquisition and local response. If the enabled component graph is connected, generic forms in the stated open source ball fund all enabled contacts. | Genericity concerns that supplied source space, not every constrained formation image. Connectivity has no size-uniform recovery guarantee and derives no nonuniform pattern or hierarchy. |
| Response identifies relative geometry | Full instantaneous form probes recover symmetric connected weights up to a common scale and determine the consensus tangent under matched coefficients. A calibrated background fixes the scale. | At disconnection, independent component scales remain hidden but affect later contact growth. One port gain and even the complete instantaneous form response do not supply the whole nonlinear state. |

The method of integration is selective. The existing common-metric estimate
already controls the integrated output, so no matched-asymptotic solver or
new weighted integrator is needed for this proof. Component routines and the
shared loss ledger suffice for the funded-spanning premise; no separate graph
dynamics is introduced. Native initial-field controls exercise three components
and hidden background-scale effects. The mathematical all-time claims remain
proofs, not numerical trajectories disguised as tests.

These results broaden available response geometry and connect preparation,
retained relations and recovery. They also make the present limit sharper:
the newly acquired regimes still approach common nodal form and phase.
This motivated the compatibility and full-state recovery question for
**nonuniform stationary families** under changing contacts. The subsequent
operator audit below makes its canonical use conditional on a missing causal
bridge. Existing native fixed-support stiffness and winding results are inputs
with their own laws, not a new proof for this completion.
Higher-order discrimination of h and finite arrival times remain useful
dependencies; neither is automatically another trajectory or reading campaign.

<a id="bibliographic-nonuniform-operator-audit"></a>
### Nonuniform structure and the missing operator realization

The next selected reading began with nonuniform profiles and was redirected
by the user's requirement that structural operators remain fundamental in
coupling, resonance and organization. It supplies useful conditional methods,
but also makes the distinction between a proposed law and its consequences
unavoidable. The [operator audit](STRUCTURAL_OPERATORS.md#continuous-operator-realization-audit)
owns the causal result and counterexample; the program alone owns priority.

| Source | New complete printed pages (one-based PDF pages) | Rendered checks, printed pages | Applicable method and limit |
| --- | --- | --- | --- |
| Wiggins | 356-365 (377-386) | 356,357,364 | Check transverse recovery around a family, including neutral directions and the actual full-state linearization. A local result does not prove acquisition. |
| Golubitsky and Stewart | 16-26 (38-48) | 16,17,18 | Identify symmetries of the actual law and state. A retained contact parameter is not automatically a symmetry coordinate. |
| Newman | 190-194 (203-207) | 192 | Distinguish signed edges, absent relations and balance. Signed-graph balance does not determine the weighted phase Hessian. |
| Aubin, Bayen and Saint-Pierre | 365-373 (388-396) | 366,371 | Continue admissible solution graphs with their domain constraints. Retain graph membership when using inverse/continuation arguments; an arbitrary off-graph pair is not a solution. |
| Cross and Greenside | 228-234 (244-250) | 229,230,233,234 | Test sideband/full-state stability rather than only existence of a patterned state. An eliminated amplitude requires its own slaving/error justification. |
| Pikovsky, Rosenblum and Kurths | 270-278 (292-300); 269 (291) explicitly revisited | 270,274,275,277 | Distinguish phase organization, defects and nonlinear response. Diffusion rates, relaxation times and oscillator phase models cannot be transferred by name. |

This comprises **51 new complete selected pages, one explicit revisited page
and 17 inspected renders across six books**. Raw sources, extracts and page
images remain private and unchanged; no whole-book reading is claimed.
The separate branch, geometry and response notes retain exact source hashes,
reading boundaries and proposed conditional derivations. Their source formulas
were checked against rendered equations rather than treating extraction or a
printed assertion as sufficient proof. No new scientific trajectory was run.

Three reusable obligations follow from the selected analysis:

- A moving stationary profile `theta_*(a)` must retain its contact-dependent
  deformation. The raw contact-to-form Jacobian can be nonzero even though
  support rates have zero linearization at equilibrium. Removing that block
  without the associated change of coordinates would change the stability
  question. Match the existing native stiffness theorem before extending it.
- Geometry must be evaluated through the actual weighted phase Hessian and
  regular resultants. A compatible fixed profile can lose stability as a
  contact grows, and correlated contact changes preserving a profile need not
  describe independent autonomous contact evolution.
- Form response alone omits phase geometry. Even additional nodal tangent
  probes can leave oriented phase circulation unresolved; a suitable contact
  perturbation or nonlinear probe may distinguish it. Circulation here denotes
  a geometric phase quantity, not physical current or temporal oscillation.

These deductions are auxiliary preparation, not a new canonical equilibrium
catalog or installed branch solver. The stronger immediate conclusion is the
need to justify the mechanisms themselves: the fully weighted support laws
are postulates, not continuous realizations of UM/RA/THOL established by these
books. Neither a Lyapunov proof nor a graph pattern supplies the missing
operator/grammar bridge. Preserve the conditional mathematics and establish
that bridge before using it as a foundation for operator-driven organization.

<a id="bibliographic-external-theory-transfer"></a>
## External theories: explicit correspondences before reuse

Conceptual affinity can identify a useful question or method. Transfer of a
result requires matching its hypotheses to the actual TNFR model. A shared
word such as organization, coherence, memory or emergence supplies neither
that match nor a physical identification. Recovering an established effective
law in a justified regime would itself be a useful result.

The additional sources consulted here are Haken's author exposition
[*Synergetics*](https://www.scholarpedia.org/article/Order_parameters) (2007)
and the abstract of [*Slaving principle revisited*](https://www.sciencedirect.com/science/article/pii/0167278996000802)
(1996), Prigogine's [*Time, structure, and fluctuations* abstract](https://pubmed.ncbi.nlm.nih.gov/17738519/)
(1978), and the opening definition in Maturana's
[*Autopoiesis*](https://xenopraxis.net/readings/maturana_autopoiesis1981.pdf)
(1981). These are limited author-source readings, not full treatise or proof
audits. The earlier selected Zwanzig and Barenblatt readings retain their
[recorded coverage](#bibliographic-reading-scope).

| Connection | Useful question or transfer | Admission boundary |
| --- | --- | --- |
| Haken: collective order parameters | Can a few derived coordinates predict an organization's response and reduce retained work? | Identify the relevant regime, spectral separation, invariant restriction or finite approximation, and reconstruction. A named coherence score is not automatically a sufficient order parameter. |
| Mori-Zwanzig projection and elimination | What initialization and memory does a collective description consume? | Reuse the existing [memory distinction](#bibliographic-memory-initialization). Deterministic hidden information needs its own statistical premises before interpretation as noise; elimination alone gives no cheap closed solver. |
| Scaling and effective descriptions | Which transformations preserve the selected response across sizes or regimes? | Apply the existing [scaling distinctions](#bibliographic-scaling), including preparation and boundary data. A rescaling or repeated shape does not select a universal law or an indefinitely iterable hierarchy. |
| Prigogine: dissipative organization | Which exchange and maintenance mechanisms support an active pattern? | The current held, unforced sine target is an equilibrium. Its decreasing mathematical storage supplies no thermodynamic identification, entropy-production law or maintained nonequilibrium activity. |
| Autopoiesis: organizational identity | Are the constituents and boundary required for identity produced by the organization's own processes? | The present supplied support does not establish that production. Use this as a stronger question about resources, occurrence and maintenance, without redefining every persistent TNFR object as autopoietic. |

For any proposed reuse, declare the state map, complete laws, preparation,
clock, domain and observation. Establish an exact equivalence/quotient or a
controlled approximation, retaining the initialization and error needed by
its consumer. Test the correspondence against the existing counterexamples
before replacing an implementation. Compatible reduced equations can expose
common behavior while leaving their microscopic interpretations undecided.

For the preserved two-port transport problem, the [slow-structure analysis](#bibliographic-timescales-viability)
already separates stable fast relaxation from phase curvature. A negative
phase Hessian does not by itself show loss of fast normal hyperbolicity or
license a critical-amplitude model. A useful new reduction must sharpen the
actual source's rate/lag or arrival bounds without resetting its incoming
form, discarding conserved means, or extending a conditional time window.

The [research objective](RESEARCH_PROGRAM.md#research-objective) and
[matter-emergence hypothesis](EMERGENT_ONTOLOGY.md#matter-emergence-hypothesis)
own the mathematical target and long-term physical ambition. These connections
supply method choices within that program; they create no additional queue.

<a id="bibliographic-timescales-viability"></a>
## Initial reading: time-scale separation and viability

The two newly supplied sources are the recommended editions, including the
three-author second edition of *Viability Theory*, rather than Aubin's earlier
sole-author book. Title and copyright pages were inspected visually. Both
copies include their terminal indexes; interior page continuity has not been
audited. This is a selected initial reading, not a whole-book or full-proof
review. The prior nine-source integration retains its historical scope.

| Source and local file | Complete selected printed pages (one-based PDF pages) | Initial methodological use |
| --- | --- | --- |
| Christian Kuehn, *Multiple Time Scale Dynamics*, Springer, 2015, AMS191; `bibliografia/Multiple time scale.pdf`, 842 PDF pages | 8,11-12,53-60 (26,29-30,71-78) | Distinguish singular fast/slow limits from an invariant slow manifold at positive parameter; retain local domain, initialization and reconstruction. |
| Jean-Pierre Aubin, Alexandre M. Bayen and Patrick Saint-Pierre, *Viability Theory: New Directions*, second edition, Springer, 2011; `bibliografia/Viability-Theory.pdf`, 826 PDF pages | 1,85-94,100-104 (24,108-117,123-127) | Distinguish existence of a viable/capturing path from invariance/absorption required of all paths; separate retention from finite arrival. |

The locators were checked within these ranges. Rendered mathematical checks
used Kuehn printed54-56,60/PDF72-74,78 and Aubin et al.
printed86,89,103-104/PDF109,112,126-127. The local record
`artifacts/bibliography-timescales-viability-20261101` retains source hashes,
reading notes and review scope. Extracts and rendered pages are private.
In that initial reading, the end of Kuehn's proposition3.2.4 after printed60,
his general Fenichel proof and tracking chapter, and the later full viability
proofs were not read. The second study below records its additional coverage.

### Slow structure must preserve the incoming state

Kuehn's theorem3.1.4 concerns a smooth compact normally hyperbolic critical
submanifold and a sufficiently small parameter. It supplies a nearby locally
invariant slow manifold, with asymptotic closeness and retained fast stability.
Local invariance allows boundary crossings; the slow manifold is generally
not unique. An unspecified small-parameter threshold and asymptotic error do
not certify our fixed parameter or the incoming source.

The existing [two-row identity](nodal/SINE_FORM_PHASE_REDUCTION.md#form-reconstruction-scope)
has the form `theta_sigma=v`, `eta*v_sigma=-A*(v-f(theta))`.
The fast derivative is the constant `-A` on the declared weighted-mean-zero
space. Loss of positivity of the reference Hessian is therefore not evidence
of loss of fast normal hyperbolicity. The singular graph `v=f(theta)` is not
automatically invariant for positive `eta`.

A potentially useful simplification would retain slow coordinates together
with the actual transverse initialization and a finite reconstruction/output
budget. It must preserve the original form through initial velocity, the
phase origin/lifts and memberwise means. Putting the initial state on a slow
graph by resetting form would change the question. The existing explicit
velocity-lag bound remains valid; an unquantified `O(eta)` statement would
not improve it or prove source arrival. These are opportunities for an actual
consumer of the existing [reduction and memory methods](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#scale-similarity-and-error),
not a new general framework or an established faster implementation.

### Persistence and arrival have different quantifiers

Aubin et al. definitions2.10.2 and2.11.2 separate viability/capture, which
require at least one permitted evolution, from invariance/absorption, which
require the corresponding property of every permitted evolution. A controller
may select a viable path; bounded uncertainty supplies no such selection.
The current lag envelope describes a larger set of phase paths than the actual
complete-law family. Existence of a favorable path in that set is insufficient
for actual-source arrival. A robust absorption argument for the larger set
would suffice; if it is too conservative, retain more of the actual velocity,
form and source correlations instead of treating the uncertainty as a control.

The reading also separates staying inside an environment until possible entry
from reaching a target in finite time. Lemma2.12.5 identifies the extra
condition that no evolution can remain in the environment outside the target
forever. It supplies no uniform deadline by itself. For TNFR the target should
be the admitted finite capture domain; exact finite-time attainment of an
equilibrium is a different claim. The later tangency characterizations retain
closedness, regularity and additional hypotheses; their full proofs have not
been audited or applied here.

A useful method must connect source inclusion, retention before entry,
finite progress and maintenance after entry, keeping their error budgets
separate. This strengthens the formulation of the existing transport question;
no source-containing region, basin arrival, operator occurrence or physical
mapping has been established by this initial reading.

<a id="bibliographic-second-timescales-viability"></a>
## Second reading: predictive reduction, boundary geometry and hybrid continuation

This second study adds 57 selected substantive pages from the two new books.
Its purpose is to connect methods to existing TNFR structure and dynamics,
including later interfaces and operator events, rather than assume that the
current two-port example exhausts the paradigm. The result includes one
conditional comparison calculation with its proof at the model owner, several
specific reuse opportunities, and two source statements that must not be
imported literally. It selects no new trajectory or physical interpretation.

| Source | New complete printed pages (one-based PDF pages) | Scope |
| --- | --- | --- |
| Kuehn | 113-129 (131-147) | Tracking, entrance/exit geometry, C0 exchange lemma through its proof and the C1 statement. The following C1 proof was not read. |
| Aubin, Bayen and Saint-Pierre | 440-460 (463-483) | Contingent cones, regional tangency, viability and invariance; the continued proposition 11.4.1 after 460 and external proofs were not consumed as complete results. |
| Aubin, Bayen and Saint-Pierre | 273-276,291-297,503-510 (296-299,314-320,526-533) | Source/target connection, forward/backward matching and hybrid flow/reset runs. Opening or trailing fragments of adjacent sections are not treated as complete results. |

Consumed diagrams and formulas were checked in rendered pages: Kuehn
printed 116, 120, 124-125, 127-129; Aubin et al.
printed 442, 444, 447, 448, 451, 455, 457, 458, 293, 296, 508-510.
Exact reading limits, source identities, visual checks and independent internal
review are retained in `artifacts/bibliography-second-timescales-viability-20261101`.
These selections do not certify either whole book or all proofs cited by it.

### A concrete calculation: turn model defect into a source-aware error bound

The new [velocity-graph comparison](nodal/SINE_FORM_PHASE_REDUCTION.md#sine-velocity-graph-defect)
connects the fast/slow and boundary methods for the existing held complete
sine law. Given a declared graph `v=g(theta)`, it computes the failure of
that graph to satisfy the full evolution. A direct chain-rule identity and
scalar comparison then separate two sources of error: the incoming velocity
mismatch, which contains initial form, and the graph's invariance defect.
The proof retains the fixed parameter, metric, domain and all consumed rows.

This is a reusable conditional result, not a computed invariant manifold.
It can assess whether an approximate structural description is accurate
enough for a particular boundary or response. Setting `g=f` recovers the
existing lag mechanism. A more elaborate graph earns its cost only if proved
derivative/defect/source bounds improve a specified prediction; a formal
higher order alone is insufficient. No candidate graph, improved numerical
constant or source-arrival verdict has been evaluated here.

The same owner projects the error ball onto a consumed boundary normal while
retaining its zero weighted mean. This gives the exact support of that outer
ball and can improve some observation directions. It gives no extra tightening
for the phase-difference faces already used in the two-port proof. Reuse shared
metric and scalar-comparison arithmetic if a later consumer warrants code;
this reading introduces no parallel numerical implementation.

### Four useful intersections with existing structural results

| Intersection | Existing TNFR owner and additional opportunity | Required discriminator |
| --- | --- | --- |
| Slow tracking and predictive identity | [Initialized slow comparison](nodal/SINE_FORM_PHASE_REDUCTION.md#sine-controlled-slow-phase) already retains the form transient. Kuehn clarifies why position, tangent and exit-map accuracy are different products. | Use a position margin for membership; establish the stronger derivative control actually needed for sensitivity/transversality. A small position error alone supplies no tangent guarantee. |
| Boundary geometry and retained information | [Source transport](nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-source-transport) already uses signed normal rates and correlated error. Compute the support of the admitted error set in each consumed direction instead of summing unrelated coordinate bounds. | Check the whole relevant boundary and preserve source correlations. If an outer inclusion is too permissive, retain lag or internal coordinates before claiming the actual flow fails. |
| Connection and composition of organizations | The connection proof matches full forward/backward states. [Lifted interface composition](nodal/SINE_CLASS_INTERFACE_COMPOSITION.md#sine-composition-lifted-residual) already retains hidden paths and source memory. A concrete consumer is proving the coupled input-domain premise assumed by [feedback substitution](nodal/SINE_CLASS_INTERFACE_COMPOSITION.md#sine-composition-output-substitution). | An interface must match sufficient state, ports, means, clock and errors. Overlapping observations or outer enclosures do not establish a common reachable state or allow unrelated histories to be spliced. |
| Continuous persistence and structural events | [Pulse causal closure](nodal/OPERATOR_PULSE_CLOSURE.md#operator-pulse-causal-closure) already separates flow, jumps, occurrence and resources; [through-word error](nodal/ORGANIZED_SOURCE_CHILD_QUOTIENT.md#organized-source-child-word-error) retains event timing. | A maintained successor must admit the incoming state, actual reset, resources and next law. Infinite viable resets need not give progress, a maintained full-state pulse or eventual target entry. |

The exchange-lemma example has stable and unstable fast directions. The current
A2 fast operator is attracting on the mean-zero space; importing its saddle
geometry or exponentially sensitive exit mechanism into A2 would change the
premises. Similarly, backward flow is an auxiliary mathematical construction,
not a physical reversal or permission to perform a new shooting campaign.

### Critical reading prevents two invalid transfers

**Continuous tangency does not certify every approximate Euler map.**
The extra graphical-approximation clause (11.6), printed 448, fails literally
for `K={0}`, `F(x)={0}` and `F_epsilon(x)={epsilon/2}`. Their graphs satisfy
the displayed closeness premise and the continuous system is tangent, but
`0+h*epsilon/2` is outside `K` for every positive `h`. This refutes that
additional universal approximation claim, not the regional viability theory.
TNFR keeps numerical endpoint error and continuous invariance separate.

**Repeated finite admissible segments do not guarantee eventual arrival.**
The final finite-target-arrival clause of theorem 12.3.8, printed 509, needs an
additional progress premise. Take `K=[0,1]`, target `C={0}`, continuous `x'=1`,
and a reset available only at `1`, sending it to `1/2`. Every state in `K`
reaches `C` or the reset set under the continuous flow; the open interval
`(0,1)` admits no indefinitely remaining continuous trajectory. Yet from
`1/2` every viable full run cycles to `1` and resets to `1/2`, with cadence
`1/2`, indefinitely and without finite-time event accumulation. It never
reaches `C`. The preceding viability equivalence is not refuted. The missing
termination premise matters to any proposed sequence of TNFR operators.

These are narrow analytical counterexamples to printed clauses, checked
against the rendered pages, not a dismissal of either source. The robust
invariance argument also requires regularity of the velocity set in the
state; smoothness in a freely chosen control is not a substitute.

### Consequence for further development

The useful optimization target is the cheapest sufficient predictive state
and error certificate for a declared structural response. Existing exact
quotients, retained-memory interfaces and source-aware comparisons are the
starting points. The new graph-defect bound supplies another conditional tool;
using it requires an actual graph/domain and a benefit over the current bound.
A later event consumer additionally needs matched crossing and reset errors.
The subsequent [source-conditioned application](nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-relative-progress)
retains the evolving defect forcing and proves relative velocity tracking
and actual phase-potential descent for the original source. Its
[correlated endpoint enclosure](nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-transport-seed)
provides a starting set, with transport boundaries and destination still open.
These are later analytical applications, not results supplied by the books.
Further reading should address a named missing proof or structural hypothesis;
applying the available methods is the present priority. No uniform arrival
time, autonomous operator selection or physical mapping follows from this reading.

<a id="bibliographic-reading-scope"></a>
## Sources and reading depth

The three earlier reviews used seven user-supplied books in the local
bibliografia directory; later additions retain separate reading scopes.
Printed page numbers identify the source edition; PDF pages are
one-based positions in the supplied file. Selected passages were read in
full, with visual checks of consumed formulas and diagrams. This is not a
whole-book review, nor an exhaustive verification of their arguments.

| Source | Selected substantive reading: printed pages (PDF pages) | Main contribution |
| --- | --- | --- |
| Jorge Wagensberg, *La rebelión de las formas*, Tusquets, 2004; supplied second printing 2005 | 19-25 (12-18), 124-127 (117-120), 250-255 (240-245) | Structure, boundary, classification, persistence and finite-scale recurring form. The philosophical use of function guides questions; it is not a TNFR law. |
| Martin Golubitsky and Ian Stewart, *The Symmetry Perspective*, Birkhäuser, 2002 | 6-15 (28-37), 27-29 (49-51), 33-40 (55-62), 83-86 (105-108); sections 1.2, 1.7, 2.1-2.3, 3.7 | Symmetry of a complete law, invariant subspaces, perturbations and coupled-cell structure. |
| Mark Newman, *Networks*, second edition, Oxford, 2018 | 142-153 (155-166), 685-697 (698-710); sections 6.14 and 17.2-17.3 | Laplacian modes and network dynamics under specified node and coupling laws. |
| Steven H. Strogatz, *Nonlinear Dynamics and Chaos*, third edition, CRC Press, 2024 | 219-226 (234-241), 235-238 (250-253), 274-280 (289-295) | Attracting cycles versus conservative recurrence, slow dynamics, and nonlinear bifurcation conditions. |
| Arkady Pikovsky, Michael Rosenblum and Jürgen Kurths, *Synchronization*, Cambridge, 2001 | 8-14 (30-36), 31-40 (53-62), 176-188 (198-210) | Operational synchronization, oscillator phase, isochrones, weak forcing and the limits of phase-only reduction. |
| Robert Zwanzig, *Nonequilibrium Statistical Mechanics*, Oxford, 2001 | 143-159 (154-170), 165-168 (176-179), 174-187 (185-198) | Exact elimination, initialization, projected dynamics, measures and the extra assumptions of statistical/Markovian approximations. |
| G. I. Barenblatt, *Scaling*, Cambridge, 2003; supplied printing 2006 | 22-25 (37-40), 82-93 (97-108) | Governing parameters, dimensional analysis, complete/incomplete/no similarity, and the validity of asymptotic reduction. |

The evidence directory artifacts/bibliography-synthesis-20261029 contains
form-scaling-review.md, symmetry-networks-review.md,
dynamics-synchronization-review.md and projection-memory-review.md, with
coverage, source hashes, visual checks and comparison to repository owners.
Its baseline and final receipts preserve the original PDFs and prior evidence.
The Wagensberg scan reaches references on printed page 316 but omits the name
index advertised at 317; interior continuity was not exhaustively audited.
The other six files appear to include their final indexes. Earlier short
academic reviews and the replaced 1996 scaling title are historical inventory,
not substitutes for the seven sources used here. No full book or page image
is reproduced in this chapter.

<a id="bibliographic-form-identity"></a>
## Form becomes useful through a declared response

Wagensberg's distinction between internal organization, boundary shape and
relations to the surroundings suggests asking what a proposed form *does*
under specified conditions. His boundary-based meaning of form must not
replace TNFR's EPI definition, and a natural example does not select a TNFR
pressure law, operator occurrence or preferred polygon.

The [formation and composition contract](EMERGENT_ONTOLOGY.md#generative-bound-organization)
already distinguishes membership, initial absence, acquisition, maintenance,
recovery and response. Its practical use is to choose the same observation
on the source, interacting system and control, then establish the claimed
change and retained predictive consequence. An observation undefined before
contact is not measured absence. Retention of the constituents is not by
itself retention of their joint relation.

The [two-port target](nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-compatibility)
and [prepared-source capture](nodal/SINE_TWO_PORT_CAPTURE.md#sine-two-port-capture-result)
are existing, different results. Connecting an actual formed source to that
organization remains a source-dependent question. Likewise, the
[single-bridge cycle obstruction](nodal/SINE_FORMED_CLASS_CONTACT.md#sine-formed-joint-cycle-eligibility)
excludes one proposed carrier of identity, not interaction, response or all
collective organization. These examples make a new general ontology chapter
or another generic geometry scan unnecessary.

<a id="bibliographic-symmetry-state"></a>
## Symmetry reduces work when the law, state and question match

Golubitsky and Stewart supply a systematic language for symmetry and invariant
families. Newman shows why graph modes become dynamically informative only
after specifying the node and coupling laws. For a TNFR use, distinguish:

| Object | What has to be checked |
| --- | --- |
| Complete law and support | The proposed action preserves all consumed rows, capacities, weights, degree normalization, boundaries and event rules. A graph drawing alone establishes none of these. |
| Reference state | The actual reference is fixed by the claimed symmetry, or is carried to a separately specified equivalent state. A symmetric law also permits nonsymmetric states. |
| Source/error family | Initial errors and their correlations respect the restriction, or off-family deviations are explicitly retained and bounded. Independent node errors do not generally lie in a fixed subspace. |
| Observation | The claimed equivalence really leaves the readout unchanged; otherwise the transformation changes the question. |

Invariance of a family permits an exact restriction of trajectories *within*
that family. It does not prove attraction from surrounding states or a closed
observation for every state. The latter needs the projectability criterion
at [joint form-phase closure](nodal/DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state).
An approximate restriction additionally needs transverse and nonlinear error
control for the actual preparation and horizon.

Under a smooth equivariant flow with uniqueness and a local inverse, the
state's exact isotropy is preserved at finite times within that domain
(Golubitsky/Stewart, section 1.2, proposition 1.18, printed page 12). A limiting
equilibrium can have more symmetry. Thus exact attainment of that symmetry
is a poor generic finite-time formation criterion; a justified finite
membership domain is a different claim. A support event must be treated
separately because it changes the field. This does not exclude formation.

Before choosing a relational-response test, check whether its observation is
forced to hide the proposed change by symmetry. Passing this filter establishes
only that a witness is not excluded that way, not a nonzero response or gap.

The [two-port capture owner](nodal/SINE_TWO_PORT_CAPTURE.md#sine-two-port-capture-result)
already separates its symmetric reference from the surrounding error family.
The [reduced-port owner](nodal/SINE_REDUCED_CLASS_PORTS.md#sine-reduced-class-ports)
retains a controlled linear internal surrogate and nonlinear bridge; it does
not claim that its reflection restriction is an exact quotient of the full
nonlinear sine law. Reuse these distinctions rather than install another
symmetry wrapper. A cyclic relabeling can align port names without making a
formed source equal to a midpoint-aligned preparation.

Laplacian eigenvalues alone do not certify the stability of a coupled
form/phase/capacity system. Use its complete Jacobian, relevant conserved
leaf and nonlinear domain. A master-stability calculation requires the
appropriate identical-unit synchronous solution and coupling structure;
it cannot be imported for an internally twisted organization by matching a
graph spectrum. Also retain the [conductance/length distinction](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md):
visual geometry, path metric and dynamical coupling need not coincide.

<a id="bibliographic-memory-initialization"></a>
## Reduction retains the cost of discarded information

Zwanzig's block elimination explains why an exact observed equation can have
both a history term and a term carrying hidden initialization. TNFR already
has this reusable deterministic mechanism in
[coordinate memory](../src/tnfr/mathematics/linear_observation.py) and the
[derived-memory chapter](DERIVED_EPI_MEMORY.md). A new general projection
library would duplicate it without resolving the current scientific gap.

Distinguish three operations: a deterministic coordinate projection, a
symmetry quotient, and a conditional expectation under a probability measure.
An interval of allowed initial states is not a distribution. The hidden
initialization term is deterministic once the fine initial state is fixed;
calling it noise does not make it white, Gaussian, independent or negligible.
An equation for an ensemble mean is not a closed equation for every member.

The [reached-state closure obstruction](nodal/SINE_CLASS_MEDIATED_MEMORY.md#sine-class-memory-reached-closure-obstruction)
already shows why equal visible data can hide different later responses.
The [reduced-port model](nodal/SINE_REDUCED_CLASS_PORTS.md#sine-reduced-class-ports)
retains initialization and nonlinear discrepancy. Both are useful methods
with model-specific coefficients, not transferable source verdicts.

For a new reduction, first declare the prediction and source family. Either
retain sufficient state, carry explicit memory, or justify an approximation
with a bound on initialization, kernel/tail, retained evolution, domain and
output error over the intended horizon. Small coupling or a small kernel
alone does not show that the retained variable is slow. Equilibrium-weighted
projections and fluctuation-dissipation relations need their own measure,
generator and preparation hypotheses; a dissipative node-space metric supplies
none of those identifications by itself.

<a id="bibliographic-dynamics-phase"></a>
## Stability, pulse and phase have different conclusions

Strogatz and Pikovsky/Rosenblum/Kurths make it useful to keep the following
result types visible rather than compress them into a statement of stability:

| Result | What it establishes and what remains |
| --- | --- |
| Compatible invariant or stationary organization | It can exist under the selected law. This does not establish source-to-target entry. |
| Local attraction | Nearby admitted preparations approach it. A finite entry bound from the actual source needs more. |
| Finite passage and capture | The particular source reaches an admitted trapping domain by a bounded time; preparation, error and domain remain part of the result. |
| Retained observable response | The declared readout persists or discriminates a control on its specified interval; another readout may require more state. |

The [native source-entry audit](nodal/NATIVE_ACQUIRED_RESPONSE.md#native-source-entry-admission)
already identifies a missing useful domain/metric/rate certificate. A generic
Lyapunov argument does not fill it. For semidefinite loss, zero instantaneous
loss need not mean equilibrium; compactness, forward continuation and the
invariant zero-loss set matter. The existing
[permanent-pulse account](nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
owns the actual law-specific argument.

A decaying complex mode, conservative recurrent motion, externally maintained
response and an attracting self-sustained cycle are different mechanisms.
TNFR has [damped collective response](nodal/RELATIONAL_RETURN_PATH_GEOMETRY.md#shared-collective-pulse),
[conservative organization](EMERGENT_ONTOLOGY.md#organization-and-internal-pulse)
and an [exact phase-advance branch with finite timing retention](nodal/GLOBAL_PULSE_HYPOTHESIS.md#phase-advance-nonlinear-retention).
Keep those positive results and their limits. The bibliography neither
derives a universal sustaining pulse nor licenses adding a favorable damping
term or operator schedule. A complex eigenpair is not by itself a Hopf
bifurcation or a new organized identity.

Primitive nodal phase, a geometric observation, a constant
[asymptotic origin](nodal/RELATIONAL_PHASE_MEMORY.md#asymptotic-origin-coordinate-scope)
and an advancing oscillator phase are different coordinates. The book's
isochrone reduction starts from an attracting periodic orbit, then controls
weak forcing and transverse deviations; averaging adds further assumptions.
An exact invariant or conservative reduction may have a different valid
contract. In every case retain the phase map, domain, actual initial state,
complete field, error and horizon. Synchronization also needs a specified
readout: frequency agreement, bounded relative phase and finite tick agreement
are distinct. A [monotone clock](nodal/FORCED_SOURCE_AND_CLOCK.md#25-relational-time-and-synchronization-are-separate-claims)
requires additional state and timing premises.

<a id="bibliographic-scaling"></a>
## Scaling joins geometry, memory and time without conflating them

Barenblatt's analysis starts with the governing parameters, including source
and boundary data. Dimensionless groups constrain dependence but do not
select the complete law. Their size alone does not justify deleting a term.
The [scale distinction](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#scale-similarity-and-error)
separates change of units, exact reduction, geometric repetition and an
asymptotic similarity claim. Complete, incomplete and absent similarity in
the declared limits and power-type variables are all possible outcomes;
the last does not exclude every other form of reduction, and a fitted power
does not choose among them.

This connects three already useful TNFR tools: symmetry can identify a
candidate invariant family; observation can require additional state or
memory; scale/clock transformation must retain their initialization and error.
The [registered-pattern inheritance](nodal/ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md#organized-constitutive-inheritance)
and [nested reduction](DERIVED_EPI_MEMORY.md#92-the-memory-is-removed-by-a-genuine-nested-nodal-quotient)
give scoped examples. They do not imply an indefinitely iterable hierarchy
or a universal fractal exponent.

The [energy-informed slow comparison](nodal/SINE_TWO_PORT_TRANSIT.md#sine-energy-informed-slow-comparison)
already offers a more relevant method than an unqualified slow-manifold
approximation: it carries the fast transient, reconstruction and finite-time
error. Reuse it only after matching the actual source, degree normalization,
phase origin and comparison domain. A small coefficient can accumulate over
a long interval; changing the clock cannot erase the inherited preparation.

<a id="bibliographic-integration-decisions"></a>
## Integration decisions

These are method choices for the program, not separately activated studies.

| Decision | Concrete consequence |
| --- | --- |
| Integrate now | Use one source/event/identity declaration carrying symmetry, retained state or memory, prediction type, finite domain and error. Link the existing proof owners rather than restating their results. |
| Clarify now | Distinguish complete-law symmetry from source symmetry; deterministic uncertainty from statistical ensembles; finite entry from qualitative attraction; geometric repetition from asymptotic or exact dynamical scale transfer. |
| Reuse conditionally | Existing Jacobian/energy, coordinate-memory, linear-observation, reduced-port and slow-comparison methods when their hypotheses match the actual consumer. Retain full state when reduction offers no proved benefit. |
| Defer | Statistical closure, Markovianization, new phase-only models, renormalization exponents and iterative fractal hierarchies until a selected question needs them and their missing premises can be supplied. |
| Reject as unsupported shortcuts | Imported midpoint preparations, silent removal of source errors, privileged polygons, graph-spectrum-only stability, visual fractality as identity, and physical identification from bibliographic resemblance. |

No verified implementation error or obsolete shared kernel was found in this
review. The warranted optimization is method routing and sharper transfer
conditions, not speculative code replacement. The physical-readiness record
and all source-specific mathematical verdicts retain their existing scope.

<a id="bibliographic-emergence-opportunities"></a>
## Beyond one contact: reusable organization mechanisms

The second reading asks how the bibliography could improve TNFR's general
account of emergent organizations, beyond the currently selected experiment.
The most useful common target is an already admitted organization acting as a
**predictive component**: a retained internal state, an interface, a specified
response, and a valid continuation when another organization or an action
interacts with it. Several scoped instances already exist. The opportunities
concern additional consumers and missing bridges, not a new NFR definition,
primitive substance or universal law. The final audit below refines their
priority against the existing interface and event results.

Several seemingly new ideas are already substantial TNFR results: non-orbital
collective families, finite memory forecasts, exact phase-advance pulses and
their distinct labeled/exchanged periods, acquired native alternatives, and
interface-dependent effective laws. The opportunities below identify bridges
between those results rather than reclassifying their existence as unfinished.
Each proposed gain remains unproved and requires its own selected study.

The additional selective coverage and source-transfer notes are recorded in
artifacts/bibliography-emergence-20261029: symmetry-organization-review.md,
activity-interaction-review.md, memory-composition-review.md and
scale-adaptation-review.md. They retain exact printed/PDF locators, original
source hashes, inspected code and visual checks. The first review's reading
limits continue to apply. No new scientific evaluation accompanied this
second synthesis.

| Additional source selection | Printed pages (one-based PDF pages) |
| --- | --- |
| Wagensberg, chapter 7 passage on persistence strategies | 112-118 (105-111) |
| Golubitsky/Stewart, sections 3.2-3.4, 3.6 and 4.8 | 61-71 (83-93), 78-83 (100-105), 110-116 (132-138) |
| Newman, section 17.5 | 701-706 (714-719) |
| Strogatz, sections 3.4 and 8.4 | 62-68 (77-83), 286-291 (301-306) |
| Pikovsky/Rosenblum/Kurths, sections 8.1-8.2, 11.1 and 12.3.1 | 222-232 (244-254), 266-269 (288-291), 286-289 (308-311) |
| Zwanzig, selected elimination, mode-coupling, reversibility and continued-fraction passages | 18-29 (29-40), 160-164 (171-175), 169-173 (180-184), 193-197 (204-208), 205-207 (216-218) |
| Barenblatt, sections 5.3 and 7.1-7.2 | 102-108 (117-123), 123-132 (138-147) |

These are additional substantive selections, not whole-book readings. One
Symmetry boundary page, PDF 105, contains both new section 3.6 material and
the previously read beginning of 3.7. Other boundary pages can contain adjacent
material without implying a full review of those adjacent sections.

<a id="organization-interfaces-and-memory"></a>
### 1. Internal state that survives a change of interaction

Zwanzig's additional elimination, finite-bath and response-chain examples
(printed pages 18-29, 160-164, 193-197 and 205-207) suggest a constructive use
of memory: derive a few internal coordinates from the response they preserve,
and initialize them from the actual constituents. Memory can then be a compact
part of the effective object's state rather than a long recorded history.
The book's ensemble assumptions and bath interpretations do not transfer
automatically to TNFR.

The [linear observation implementation](../src/tnfr/mathematics/linear_observation.py)
already constructs an exact sufficient linear state for a fixed generator and
output, as well as coordinate memory with hidden initialization. The
[organized reservoir](nodal/ORGANIZED_PARENT_RESERVOIR_MEMORY.md) already shows
locally attracting geometries on fixed-J leaves under its supplied
zero-leakage auxiliary law. These results do not yet derive that reservoir from
the interior of an arbitrary organization or make a state sufficient for
every new connection.

Existing results also cover more than a fixed isolated flow:
[finite diffusion/REMESH words](DERIVED_EPI_MEMORY.md#95-exact-composition-with-a-declared-temporal-echo)
have a sufficient retained history state; the
[nonlinear sine interface](nodal/SINE_CLASS_COLLECTIVE_INTERFACE.md)
carries hidden initialization across a finite impulse family; and
[causal component composition](nodal/SINE_CLASS_INTERFACE_COMPOSITION.md#sine-composition-output-substitution)
bounds substitution through both form and phase input channels after feedback.
The [bounded contact-continuation record](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/theory/nodal/SINE_CONTACT_COMPOSITION.md#sine-interface-continuation)
already carries the original preparation and a contact-history bit into its
four allowed queries. This is a source-specific finite prediction, not an
indefinitely reusable autonomous state or proof of a minimal encoding.
Their domains and work/identity premises remain essential. Reusable interfaces
and finite event words are therefore not missing general achievements.

**Consumer-specific question:** does a proposed smaller state preserve the
additional boundary response or continuation that an existing result does
not cover, with a useful cost/error advantage? A mode invisible in isolation
can become visible through a new port. Choose the permitted inputs, outputs
and continuations before eliminating it. Full state remains a valid reference;
compression is not a prerequisite for source or identity admission. Reuse the
[collective action criterion](STRUCTURAL_OPERATORS.md#collective-operator-descent):
if two fine states have the same retained state but the event exposes different
successor observations, the old state cannot determine that successor.

For such a new consumer, compare composition followed by elimination with
elimination followed by composition, for the same joint law and nonzero hidden
preparation. Preserve any input feedthrough actually present, source
correlations, registration, clock, continuous storage and separate attachment
work. The existing causal sine map has no instantaneous feedthrough of its
continuous inputs; its declared form jumps are handled separately. Existing
[mediator composition](nodal/RELATIONAL_MEDIATOR_DYNAMICS.md#two-mediator-composition)
and fixed history packets are controls, not missing achievements. The new
result would be sufficiency across the selected additional continuation, or
a witness identifying the extra information it requires. If exact compression
returns the full state, that is an informative answer.

A further optimization is available only when nonlinear hidden feedback is
actually resolved by the observation. [Constituent memory](nodal/ORGANIZED_CONSTITUENT_MEMORY.md)
already derives quadratic interface corrections and higher-order obstructions;
[compressed memory](nodal/ORGANIZED_COMPRESSED_MEMORY.md) already has a useful
static-variance forecast. For a justified tangent, deterministic products of
hidden coordinates evolve linearly in product space. One could apply the
existing observation method to retain only combinations needed for the chosen
quadratic output, with the known nonlinear remainder propagated separately.
These products are not statistical covariances unless an ensemble is supplied.
A moving carrier needs its actual time-dependent coefficients or a proved
common invariant space. The test is a strict state/error advantage over the
successful static forecast and a same-law full-state reference; no advantage
means no reason to replace the existing method. The
[exact deterministic Gram closure](nodal/DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state)
already retains its own realizability and continuation premises. The proposed
gain here is a useful dynamic subset for the specified nonlinear response,
not the first occurrence of product-state closure in TNFR.

<a id="organization-local-balance"></a>
### 2. Useful imperfect collective organizations

The [imperfect-interface result](nodal/DERIVED_FORM_PHASE.md#imperfect-interface-ambiguity-and-a-finite-predictive-bound)
already supplies both an information-loss lower bound and observation-only
upper error bounds for its directed pure-EPI law. Useful prediction after
loss of exact closure is established there. The star below is a different
possible consumer; its nonlinear coefficients and event bounds need their
own argument, not a transfer of that result's constants.

Golubitsky/Stewart's coupled-cell and spatiotemporal viewpoint suggests looking
at the inputs received by a group, rather than relying only on global graph
automorphisms. TNFR already goes beyond that suggestion: the star construction
in [collective phase dynamics](nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md)
includes a non-orbital grouping with an exact input criterion, retained
collective state and transverse control. Different degrees and capacities
need not prevent a collective description when the complete rows balance.

**New question:** how much departure from that balance can a useful reduced
prediction tolerate after an actual structural action? Start with the local
capacity-reset example already known to break star tangency, under the same
weighted-form/bare-sine law, fixed support and original clock. This is initially
a mathematical primary-capacity reset; it does not establish execution of
Silence. A canonical action additionally requires actual represented outputs,
history, grammar and context admission. Retain the
resulting transverse source and compare the reduced response against the
full post-event law. The desired result is an output-error bound over a fixed
domain and horizon, and a decision about which additional mode must be kept.
An exact all-state quotient and a finite approximate description are different
successful outcomes; the latter need not require perfect geometry.

The smallest informative control is the exactly balanced family with its
known zero defect, followed by one declared balance-breaking reset. A merely
small instantaneous defect is insufficient if transverse dynamics amplifies
it. Use the actual complete propagator or a justified metric bound and retain
event initialization. If the error budget fails, the outcome localizes the
missing transverse coordinate or usable time window. The benefit would be
predictive organizations robust to imperfections, rather than another example
of exact synchrony. It does not show spontaneous acquisition of the group.

<a id="organization-rhythm-response"></a>
### 3. An active organization as a responsive component

Pikovsky/Rosenblum/Kurths, sections 8.1-8.2, make reciprocal amplitude and timing
effects a useful source of questions. TNFR already derives
[collective-wave timing effects and reciprocal response](nodal/GLOBAL_PULSE_HYPOTHESIS.md#free-collective-pulse-interaction),
in addition to its exact phase-advance branch and finite retention. Neither
phase advances, distinct symmetry-related periods nor the existence of that
feedback is a new proposal here.

**New question:** does one actual, admitted local Emission action on the exact
phase-advance branch produce a predictable finite change in *relative* tick
times while the organization's activity and identity remain observable?
This joins a structural action to a functional property of the organization.
A marked port can distinguish members previously identified by a permutation;
the collective state may have to retain that association and the internal
phase. It cannot silently use a smaller symmetry quotient after the port
has broken its observational equivalence.

The first decision is whether the real event's tangent has a nonzero effect
on a preselected relative crossing-time observation. Then, if justified,
bound one finite event image and its full-state remainder using the existing
[branch retention](nodal/GLOBAL_PULSE_HYPOTHESIS.md#phase-advance-nonlinear-retention).
Retain amplitude, neutral shear and collective perturbations. Conservative
pulses do not automatically admit the attracting-cycle phase-reset model in
the book. The branch theorem also supplies no arbitrary practical amplitude
radius. Compare with no action and a common timing shift; admit any uniform
form control under the actual law. Refresh the selected complete-law pressure
after the event and preserve public operator/context requirements.

A positive result would add a structural-action-to-timing response that can
be used in composition. A zero first-order projection closes only that
linear discriminator; it does not exclude a higher-order finite response.
Such a response would need a separately justified question, not an automatic
extension. An unavailable crossing rejects that selected observation on its
declared domain. Neither result excludes all active organization.

Higher-harmonic examples in Synchronization section 12.3.1 also motivate a
bounded alternative: test one second internal envelope pattern against the
already derived quartic pulse interaction. First establish inequivalence and
stationarity, then its constrained and full transverse behavior. An existing
sideband in a stability calculation is not automatically another carrier
organization. Such a result is lower priority unless it enables a new
response or acquisition; a prepared-pattern catalogue is not the objective.

<a id="organization-scale-composition"></a>
### 4. Composable scale descriptions and response classes

Barenblatt section 5.3 (printed pages 102-108) links a specified generalized
homogeneity to transformations of parameters and the response. Sections
7.1-7.2 (123-132) use agreement of direct and staged geometric subdivisions
to motivate a scaling relation under their regularity and similarity
assumptions. This suggests a concrete TNFR test of composition, not a claim
that lossy graph aggregation is the book's invertible transformation group.

**New question:** is there a family of organized descriptions whose first
output is a valid input for a second grouping, with compatible response,
initialization, boundary, storage and clock? Permit inherited registration and
internal state instead of demanding the same scalar formula at every level.
Compare the final output of direct and staged reduction, exactly or with a
controlled error. First check that both operations are admitted; multiplying
parameter factors while ignoring their domains is not this comparison.

The [registered inheritance map](nodal/ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md#organized-inheritance-chart)
already changes form, clock and constitutive coefficients. Its output is not
automatically another C9 organization. [Network compatibility](nodal/ORGANIZED_NETWORK_COMPATIBILITY.md)
already shows why valid pair charts can fail to coexist around a cycle while
a richer description closes. The nested P5 example already separates exact
first reduction from memory at the second. Preserve those controls. The new
capability would be a composable family with an admissible second step, or a
precise description of the additional information needed for it.

A related question concerns *dynamical response classes*: can different
admitted internal laws or geometries have the same bounded response after
justified state/clock transformations? The
[constitutive counterfamilies](nodal/COMPOSITION_COMPATIBLE_CONSTITUTIVE_LAWS.md)
match some targets and local derivatives while retaining different nonlinear
profiles. Select one existing alternative and a fixed response domain to
distinguish coordinate equivalence, finite approximate similarity and a
nonlinear difference. Only afterward ask whether another grouping reduces
that difference. Use either a proved exact observational equivalence or a
declared family compared with one reference and a uniform error budget.
Pairwise agreement within a tolerance is not generally transitive and does
not define an equivalence class. Propagate inherited errors through the next
grouping. Neither construction alone establishes asymptotic universality;
a geometric exponent cannot replace its response contract.

Reuse structural_morphism.py, the existing organized inheritance and memory
owners. The supplied graphs in multiscale/hierarchical.py and the descriptive
fits in fields.py are different instruments. No new hierarchy engine or
renormalization API is justified before a specific composition test succeeds
and a consumer benefits. Formation of the hierarchy remains separate from
representing one that has been supplied.

<a id="organization-adaptive-persistence"></a>
### 5. Persistence, adaptation and a durable change of organization

Wagensberg's passive/active/new-independence discussion (printed pages 112-118)
suggests comparing retention by boundary protection, internal change and a
change of identity. These are questions for TNFR, not a biological selection
law or a reason to require increasing complexity. A retained object should
also retain the declared interactive response; isolation alone can make a
retention score misleading.

The [native foundation](nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md#native-capacity-selection)
already compares different capacity laws and establishes different attracting
destinations. [Acquired response](nodal/NATIVE_ACQUIRED_RESPONSE.md) already
separates organizations under a single dynamic-capacity law and different
preparations. Capacity diffusion smooths capacity contrasts; it is not already
feedback responding to an identity error or environmental signal. A stronger
adaptive-compensation claim would need an independently specified feedback
law and complete balance, distinct from the engine's gated adaptation policy.

**New question with a stronger generative payoff:** can one finite admitted
action switch an *already acquired* native twist into retained consensus
under the same resumed DA1 law, while the unacted continuation keeps the
twist? This would connect acquisition, structural transformation and durable
response, rather than merely demonstrate two coexisting prepared equilibria.

The first gate is an event-image question: does a fixed work-bounded action
family on the actual acquisition image intersect a certified source-to-consensus
domain, or does an invariant/retention bound exclude that range? Begin there,
before any transit. Existing capacity-selection preparations are not that
intersection proof. Preserve evolving capacity, native resultant-domain
admission and event work. The held-capacity native executor does not implement
DA1 merely by a flag, so a later evaluation needs its actual execution path.

Success would be a retained organizational state change under one resumed
law. It would not establish autonomous occurrence or a hysteresis cycle.
Failure can quantify robustness or locate a missing capture certificate.
Existing selector hysteresis is policy memory and does not establish this
dynamical mechanism. A comparison of boundary protection with redistribution
is useful only when the selected response and surrounding perturbation require
it, with resource and timing effects separated from a new identity claim.

<a id="bibliographic-general-priorities"></a>
<a id="bibliographic-final-synthesis"></a>
## Final synthesis: select a missing capability, then reuse its methods

The final review consolidates the earlier opportunities instead of opening
five method-development projects. **State sufficiency, imperfect-interface
error and composable descriptions are obligations of one selected response
problem.** Their theorems and model premises remain distinct. Pulse response
is a possible functional application; durable acquired reorganization has the
additional source-to-destination obligation. Neither is established by a
compact representation alone.

The closing audit found that causal component substitution, finite impulse
families, source-aware memory and useful imperfect observations already have
scoped results in this repository. The priority is therefore not to invent a
general interface framework. Use those results for an actual missing consumer;
derive a smaller state only if it reduces a declared cost or improves useful
accuracy while preserving the requested prediction. Exact compression can
return the full state, and an existing finite/static approximation can
already be sufficient.

### Additional reading and its limits

The closing notes in artifacts/bibliography-final-review-20261029 record
source fingerprints, complete selected passages, visual checks and actual
implementation/proof owners: form-window-final-review.md,
symmetry-final-review.md, dynamics-final-review.md and memory-final-review.md.
Earlier selections from all seven books remain part of the synthesis; a book
need not supply another new suggestion to remain useful. No whole-book or
exhaustive mathematical validation is claimed.

| Additional source selection | Printed pages (one-based PDF pages) | Use in the closing judgment |
| --- | --- | --- |
| Wagensberg, chapter 6 | 93-106 (86-99) | Ask what a feature contributes without inferring intention, event occurrence or a distribution of organizations. |
| Golubitsky/Stewart, sections 6.3 and 9.4 | 176-184 (198-206), 267-268 (289-290) | Distinguish return modulo a group action, moving frames, instantaneous symmetry and averaging assumptions. |
| Strogatz, section 7.6 | 238-250 (253-265) | Distinguish fixed-time perturbation accuracy from cumulative phase error on a longer window. |
| Pikovsky/Rosenblum/Kurths, section 7.2 | 189-199 (211-221) | Separate exact coordinate changes, averaging and amplitude-modulated frequency locking. |
| Barenblatt, section 2.3 | 60-65 (75-80) | Check that a useful intermediate interval exists; neither early source dependence nor later omitted effects may be silently discarded. |
| Zwanzig, Appendix 1 | 198-200 (209-211), only Appendix 1 on the final page | Retain the initial-state term and the constant-operator scope of variation of constants; no frozen-generator shortcut for a changing law. |

Zwanzig Appendix 2 on the remainder of PDF 211 was not read. Strogatz PDF
253 overlaps the first review's boundary page; its section 7.6 reading is new.
Boundary pages may contain adjacent material without constituting a full
review of those adjacent sections. These sources guide the following selection
judgments; they do not supply new TNFR evolution laws or evaluated responses.

### Three distinctions that make the combined methods useful

**Prediction and generation.** A sufficient interface predicts what an
organization does under its admitted preparation and interaction. Acquisition
establishes how it enters an identity, and an occurrence law establishes why
an admitted event happens. Wagensberg's function/selection discussion helps
ask about retained consequences; it does not make a supplied TNFR contact
autonomous or select an optimal score. The
[ontology](EMERGENT_ONTOLOGY.md#generative-bound-organization) already owns these
distinctions. A source family has no probability distribution unless one is
independently specified and justified.

**Exact equivalence and useful approximation.** A quotient must respect the
chosen source domain, laws, port marks, events and outputs. A frame in which
a pattern repeats need not preserve its externally marked interaction.
An averaged symmetry need not be instantaneous symmetry. For approximate
prediction, use a common reference and a uniform error budget; pairwise
closeness at a tolerance is not generally transitive. Propagate error through
the actual next operation, including feedback, instead of assigning a fresh
tolerance at each grouping. Finite output robustness is not a theorem of
structural stability of the full flow.

**A useful finite window and an asymptotic claim.** Before proposing longer
integration or another scale, compare in the same clock the domain/entry and
response-resolution requirements with the source, reduction, event and
numerical error bounds. If a discarded contribution must decay, establish
that decay; if phase shear or cumulative forcing limits the window, retain
that limit. Barenblatt's intermediate-asymptotic viewpoint motivates checking
their common interval. It does not impose scale separation on exact closures.
A finite response needs no general attracting cycle or all-time stability
theorem. Conversely, frequency locking, a new peak or a timing shift alone
does not prove a new identity or a durable switch.

The [P5 memory bound](DERIVED_EPI_MEMORY.md#85-window-choice-limiting-controls-and-scope)
already retains the hidden source even at zero convolution window. The
[imperfect-interface bounds](nodal/DERIVED_FORM_PHASE.md#imperfect-interface-ambiguity-and-a-finite-predictive-bound)
already bracket what can be predicted from incomplete information in their
own model. They exemplify useful finite descriptions; neither their constants
nor the claim of useful accuracy transfers to another law automatically.

### What is retained, merged and deferred

| Earlier opportunity | Final disposition and first meaningful decision |
| --- | --- |
| Reusable internal state | Reuse existing realization, memory and causal composition. A new study must name the additional continuation/output that they do not yet cover, or a strictly useful smaller state with a retained error contract. |
| Imperfect organization | Part of the same prediction contract. The weighted star is one possible new consumer of established defect/ambiguity methods, not a missing general theory of robustness. |
| Active pulse response | Keep as a bounded functional option: an admitted action, available marked relative ticks and resolvable finite response. A zero first derivative closes only that linear test. |
| Durable reorganization | Keep its stronger generative question separate: an actually acquired source, an admitted action image, a different retained identity and compatible same-law capture. |
| Repeatable scales | Require a second admissible operation on the first output and a propagated response/error contract. Do not infer a hierarchy from pairwise charts or fitted exponents. |

This unification concerns how work is selected. It does not fuse pure-EPI,
weighted-sine, native capacity or conservative pulse models into one law.
Keep all admitted differences in capacity, support, phase, source and clock.
No new universal state container, pulse law, geometry catalogue, stochastic
closure, renormalization API or passive benchmark campaign is justified by
this reading. A shared implementation change needs a demonstrated consumer.

For later task selection, apply the existing
[impact, discrimination and feasibility procedure](RESEARCH_PROGRAM.md#research-selection)
to one concrete claim. An exact equal-retained-state/different-response witness can
close insufficiency; a useful upper bound can justify the smaller state; an
unresolved gap between lower and upper bounds remains an unresolved
certificate. An unavailable observation or an unproved capture-domain bridge
must not be reported as a failed physical mechanism. Preserve useful negative
results without automatically launching a higher-order calculation or scan.

This review originally supported the formed two-port source/event/identity
admission. Its subsequent exclusion result now informs the
[strategic reassessment](#bibliographic-strategic-pivot), which selects a joint
acquisition/maintenance mechanism instead of further trajectory classification.
The retained methods admit neither a replay nor a prepared midpoint in place
of the actual source, and do not close the physical-readiness gate. Reopen a
passage when the selected question exposes a specific missing tool rather
than accumulating possible lines of investigation.

<a id="bibliographic-source-transport"></a>
## Source-dependent transport and selection of organizations

The additional Wiggins and Cross/Greenside readings address a concrete gap:
the original formed two-port source converges to an equilibrium, while its
limiting organization remains unselected. They refine the existing methods;
they introduce no new complete law, numerical policy or parallel research
queue. Their original source-dependent transport priority is superseded by
the [current selection](RESEARCH_PROGRAM.md#research-current-question);
the unresolved destination remains a preserved case-specific question.

### Additional sources and actual reading

| Source | Selected substantive reading: printed pages (one-based PDF pages) | Use |
| --- | --- | --- |
| Stephen Wiggins, *Introduction to Applied Nonlinear Dynamical Systems and Chaos*, second edition, Springer, 2003 | 20-24 (41-45), 37-40 (58-61), 104-111 (125-132), 231-233 (252-254) | Lyapunov domains, local stable/unstable manifolds, attraction and basins, invariant limit sets, and the scope of a gradient field. |
| Michael Cross and Henry Greenside, *Pattern Formation and Dynamics in Nonequilibrium Systems*, Cambridge, 2009 | 139-144 (155-160), 160-163 (176-179), 209-212 (225-228), 235-236 (251-252) | Full perturbation stability, distinct phase/director defects, critical-mode amplitude construction and its validity limits. |

These are complete selected pages, with visual checks of consumed formulas
and figures, not complete chapter or whole-book reviews. The Cross/Greenside
worked example and defect discussion continue beyond the selected pages.
The local inventory now contains nine methodological books and historical
TNFR.pdf. Both new files appear to reach their final indexes; interior
continuity has not been exhaustively audited. Exact source identities,
pagination, reading limits and critique are retained in
artifacts/bibliography-transport-20261031. The books, extracts and rendered
pages remain private and excluded from publication archives.

### What the new reading changes in the method

**Source transport needs a domain and an actual flow.** Wiggins separates
local invariant manifolds from a global attraction basin and uses the
orbital derivative of a Lyapunov function under the specified vector field.
For the current question, the useful contract is a region containing the
admitted full-state source, with justified inward or outward signs on its
relevant boundary and a separate implication for the destination basin.
Membership, boundary signs and that implication are not supplied by a local
stable-manifold theorem. Check the spectral splitting and the hypotheses of
the chosen manifold theorem; zero modes need their own treatment. A lower-dimensional stable
set does not exclude a particular correlated source family, and no source
probability measure has been declared.

The state here retains both form and phase, memberwise conserved means and
continuous lifts on the fixed support. A proposed phase-gap crossing must
use the complete form-dependent phase rate. A conditional sign on one face
alone supplies no global trapping region or final identity. The
[stationary reduction and full-cut obstruction](nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-stationary-reduction)
already explain why current balance, curvature and a common energy floor
cannot select this source's destination. No equilibrium catalogue or generic
stable-manifold construction is an additional prerequisite for one useful
source-conditioned argument.

**Retain the stronger result already proved for the actual law.** Wiggins'
Theorem 15.0.3 identifies existing omega-limit points of a gradient field as
stationary. The [complete-law convergence proof](nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-post-acute-control)
already supplies a separate analytic finite-length argument for A2, including
its relative phase torus and both evolution rows. Its conclusion remains
convergence to an unspecified equilibrium, without a uniform arrival time.
The auxiliary gradient reference cannot replace that complete law.
Critically, the introductory sentence on Wiggins page 231 calling equilibria
relative extrema is too restrictive: `V(a,b)=a^2-b^2` has a stationary saddle
at zero. Use critical points, including saddles, in the TNFR classification;
do not import the wording as an exclusion of unstable alternatives.

**A stable pattern and a dynamically sufficient reduced state have different
requirements.** Cross/Greenside test perturbations beyond the periodicity of
the ansatz used to construct a pattern. For TNFR, the current stationary
curvature/pencil analysis already retains full-state perturbations. Their amplitude multiplies a specified critical
mode within a restricted near-onset expansion. The inherited
TNFR error `10^-32` is not a distance to a bifurcation; the fixed small
`gamma=1/(1023*pi)` alone establishes no critical-mode reduction either.
Any proposed amplitude would need its actual parameter family, spectral
premises, reconstruction, initial projection, remainder and time window.

Wiggins' invariant-graph tangency condition and Cross/Greenside's reduction
limits reinforce the existing [elimination contract](#bibliographic-memory-initialization):
static minimization of the fourteen path variations is an exact Hessian
calculation, not an invariant manifold or a four-node evolution law.
A Lyapunov function for a truncated amplitude model cannot settle arbitrary
long times for the original model. Conversely, that caution does not weaken
the already proved complete A2 convergence or extend its separate finite
reference-error bound beyond `7/4`.

**A defect requires its own observable and domain.** Cross/Greenside distinguish
failure of a local pattern description from a singularity of the underlying
state, and distinguish director winding from phase winding. A graph-cycle
winding or principal-gap crossing in TNFR supplies neither a continuum
embedding nor a spatial amplitude core or director quotient. For a later
collective-defect claim, first identify the observable, its domain, loop and
allowed deformations. No new defect taxonomy or physical identification
follows from the present reading.

The practical addition is therefore a sharper source-to-basin argument and
clearer reduction/observation premises. Reuse the completed stationary,
stability and convergence owners; the missing source-dependent transport
remains open. This selective reading changes no engine implementation and
admits no trajectory, root enumeration, parameter scan or diagnostic replay.

<a id="bibliographic-general-synergies"></a>
## General reuse: predictive information and numerical economy

Wiggins and Cross/Greenside also inform work beyond the two-port example.
The useful common principle is to reduce the work required for a declared
prediction without silently reducing its state, domain or evidence. This
assessment extends the earlier synthesis, not the scientific task queue.
The [program](RESEARCH_PROGRAM.md#research-current-question) now selects the
joint acquisition/maintenance admission; the earlier transport study supplies
methodological evidence without remaining an active trajectory task.

### Additional selective coverage

The same locally supplied editions and raw source identities recorded in
artifacts/bibliography-transport-20261031/baseline.json were verified before
this reading. The additional coverage is:

| Source | Printed pages (one-based PDF pages) | Consumed scope |
| --- | --- | --- |
| Wiggins, second edition | 245-249 (266-270) | Spectral splitting, local center-manifold existence and stability, invariant-graph equation and approximation theorem 18.1.4; the worked example was read through equation 18.1.16, not its continuation. |
| Cross and Greenside | 443-447 (459-463) | Separate numerical evolution, stationary-state and stability problems; finite representation, roundoff, conditioning and the opening Galerkin construction. |
| Cross and Greenside | 468-472 (484-488) | Banded linear solves, operator splitting, update ordering, repeated factorization and question-dependent resolution. |

Complete selected pages were read as locally extracted text. Wiggins printed
248/PDF269 and Cross/Greenside printed469/PDF485 were also visually checked
for the consumed invariance and splitting formulas. OCR defects in Wiggins
were not used to select coefficients. No whole-book review, additional
source completeness audit or physical comparison is claimed; private book
text and page images are not publication material.

### Connections across existing owners

| Connection | Reuse and concrete benefit | Premise that must survive |
| --- | --- | --- |
| Wiggins' invariant manifolds + Cross/Greenside's critical amplitudes + Zwanzig's memory | Use [linear observation and coordinate memory](../src/tnfr/mathematics/linear_observation.py) and the [scale/closure owner](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) to distinguish an exact quotient, a local invariant restriction and a controlled approximation. A smaller predictive model can reduce repeated observation cost. | Admit the full law, spectral splitting, reconstruction, hidden initialization and output error over the requested horizon. A stationary Schur complement is not a closed dynamical law. |
| Wiggins' basin geometry + Cross/Greenside's full perturbation stability | Combine [source transport](nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-post-acute-control) with [stationary condensation](nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-stationary-reduction). Use local stability to assess a candidate destination, and a separate signed boundary or trapping argument to connect the actual source to it. | Convergence to some equilibrium does not select one. Full-state, symmetry-breaking perturbations, recrossings and all admitted source members remain relevant. |
| Cross/Greenside's structured numerical algebra + Newman's graph structure | Reuse [exact linear algebra](../src/tnfr/mathematics/_exact_linear_algebra.py), [static path condensation](https://github.com/fermga/TNFR-Python-Engine/blob/279054b8694a9bf5fd3a7e67da65847a3e36d749/src/tnfr/research/formed_two_port_stationary.py) and [spectral kernels](../src/tnfr/mathematics/spectral.py) instead of parallel study-specific solvers. Exploit exact zeros or a proved block structure before changing precision or hardware. | Elimination can create fill-in. A cached operator must retain the actual support, coefficients, ordering and matrix type; symmetric eigensolvers do not admit a general directed generator. |
| Wiggins' Lyapunov geometry + existing signed metric estimates | Share [modified-energy bounds](../src/tnfr/physics/_sine_lyapunov.py) and [metric Taylor arithmetic](../src/tnfr/mathematics/_validated_metric.py). Keeping signed cross terms and direct observation duals can avoid pessimistic independent-coordinate bounds. | The metric must be positive on the declared retained state, and its growth bound must hold on every consumed segment. Improving a bound does not prove source entry or revive a failed attempt. |
| Cross/Greenside's time stepping + Barenblatt's scale discipline | Use the [shared integrator](../src/tnfr/dynamics/integrators.py) and the selected law's owner to separate stiffness, time units and observation resolution. Repeated fixed linear solves are potential preprocessing opportunities. | Splitting order, live versus held pressure, phase/capacity rows, sources and events determine the method and model. Changing them needs a fresh error and admission contract, not just a smaller runtime. |
| Cross/Greenside's phase/defect descriptions + the existing identity contract | Keep [winding observations](../src/tnfr/physics/winding_certificates.py) distinct from [organizational identity](EMERGENT_ONTOLOGY.md). Use the least expensive observation that can actually distinguish the proposed transition. | Acute exit, principal-cut crossing, winding change and acquisition are different conclusions. A detector of one cannot silently certify another or identify a physical defect. |

These connections consolidate mechanisms already present. They justify
neither a universal NFR wrapper nor a new amplitude, defect, GPU or
renormalization subsystem. A reduced model is optional: keeping full state
can be cheaper and safer than establishing an additional closure theorem.

Wiggins' approximation theorem is local and asymptotic. A small residual in
the invariant-graph equation is useful evidence only with its spectral and
regularity hypotheses; big-O notation does not supply a numerical constant
or a source-to-basin deadline. Cross/Greenside's amplitude restrictions and
the existing memory results impose complementary checks on that reduction.
Likewise, an initial-condition-dependent change of final state is not by
itself a bifurcation of a parameterized vector field.
Here operator splitting concerns terms of a continuous vector field, not
execution of a word of named TNFR events. Neither construction substitutes
for the other's timing, admission or balance contract.

### Engineering decisions and acceptance tests

**Implemented without changing the law:** exact LDL semidefinite admission
now skips zero Schur products and reuses one rational division per nonzero
elimination row. It inspects the evolving matrix, not an initial graph mask,
so new fill-in is retained. Primitive Fraction admission, fixed-width integer
normalization, symmetry checks, zero-pivot rejection and strict/non-strict
verdicts are unchanged. The
[independent controls](../tests/physics/test_exact_linear_algebra.py) compare
all principal minors in two and three dimensions, exercise fill-in and
representation boundaries, and bound the number of rational divisions.
This shared improvement serves metric admission and stationary curvature
readers; it is not a proof of their caller-supplied scientific premises.

Other optimizations require a named consumer and measurement before adoption:

- Reuse a fixed matrix factorization only after admission, with immutable
  normalized inputs and complete invalidation. Do not cache a scientific
  verdict or trust equality before scalar admission. Cross/Greenside's
  printed471 footnote motivates this for repeated linear solves; it is not
  permission to reuse a state-dependent Jacobian.
- Consider partial or matrix-free spectral work only when the response needs
  that subspace. The current spectral owner still materializes a dense
  operator; requesting a few modes is not an end-to-end sparse guarantee.
  Compare residuals, retained modes and peak memory on the same operator.
- Consider a different stepping scheme only after measuring the source of
  stiffness and proving its whole-history error and event handling. An
  implicit or split method need not be faster when the observation or
  nonlinear dynamics already requires a short step.

For each candidate, compare the same admitted problem before and after:
exact verdicts or outward error bounds, preprocessing and repeated costs,
memory, and unavailable/rejected cases. A constructed arithmetic timing
does not measure complete research throughput. The book's general advice
to refine resolutions applies to separately prepared controls; it does not
authorize replaying a consumed prospective producer or choosing a new budget
after seeing a reserved response.

This assessment originally informed source-dependent transport. Its finite
full-law bounds retain their proved horizons, and stationary or local stability
tools do not supply source access by themselves. The subsequent exclusion
result informs the [current strategic synthesis](#bibliographic-strategic-pivot).
The program now selects joint acquisition/maintenance admission. No trajectory,
target solve, phase-origin scan, contact guard or physical mapping is admitted
by this preserved methodological assessment.

<a id="bibliographic-nine-source-integration"></a>
## Nine-source integration into TNFR structure and dynamics

The combined bibliography supports a constructive question: **which
structural simplifications preserve a specified response, and what has to
remain dynamic when they do not?** This is more useful than searching for
a universally preferred shape, a larger coherence score, or one more
generic oscillator. It joins the ontology's identity/boundary question to
causal prediction and a quantitative cost of discarded information.

The present contribution includes a new
[reversible memory-omission bound](DERIVED_EPI_MEMORY.md#reversible-memory-omission-budget)
and its shared implementation. Projection, variation of constants and
contraction are established mathematical methods; no claim of priority for
those methods is made. The new TNFR result combines the existing reversible
geometry and kernel identity into a checked finite-horizon budget for a
previously uncertified approximation. It changes neither the primitive
nodal equation nor the selected nonlinear research law.

### Reading scope and contribution of all nine sources

All nine methodological PDFs matched the previously recorded raw identities.
The historical TNFR manuscript was not treated as a tenth methodological
authority. This review combined the earlier selected readings with the
following newly read or revisited complete pages; it is not a whole-book audit.

| Source | Printed pages (one-based PDF pages) in this review | Integration into the paradigm |
| --- | --- | --- |
| Wagensberg | 113-116 (106-109), revisiting persistence strategies | Ask what an organization's boundary, internal activity and exchange preserve. Passive insulation and active regulation are different candidate mechanisms, not an instruction to maximize connectivity or stop all dynamics. |
| Golubitsky and Stewart | 110-113 (132-135), revisiting coupled-cell Hopf structure | Use the actual action on constituent coordinates. Networks with the same abstract symmetry group can support different representations and rhythms; group names alone do not justify reduction or pattern selection. |
| Newman | 697-702 (710-715), partly revisited | Separate graph spectra from the supplied nodal/coupling dynamics. Master-stability criteria are a conditional design method; more or stronger edges need not improve stability for every law. |
| Strogatz | 230-234 (245-249) | Construct trapping boundaries from signed components of the actual vector field. The planar closed-orbit argument and the Lienard theorem retain their dimension and damping hypotheses. |
| Pikovsky, Rosenblum and Kurths | 218-221 (240-243) | The PLL example keeps the filter state and external signal. It shows why a useful phase description may require another evolving coordinate; the designed controller does not become an autonomous TNFR cause. |
| Zwanzig | 143-147 (154-158), revisiting projection/elimination | Retain both the convolution and hidden initialization. The choice of relevant variables must be justified by the response; omitted variables are not automatically random noise. |
| Barenblatt | 31-34 (46-49) | Hold the governing quantities and observation fixed when comparing designs. Dimensionless organization can reduce redundant comparisons, but omitted preparation or boundary data cannot be repaired by rescaling. |
| Wiggins | 250-253 (271-274) | A tangent-space approximation can predict the wrong stability; an invariant nonlinear graph has different obligations. Parameter-dependent center reductions remain local and require their spectral/regularity hypotheses. |
| Cross and Greenside | 315-319 (331-335) | Away from onset, phase dynamics can retain slow structural information while failing near defects, boundaries or additional slow modes. A local phase description is not a universal replacement for full constituent state. |

Zwanzig printed133-137/PDF144-148 were also inspected as response-theory
context; their quantum/material identifications are not used in the new
derivation. Formula/figure checks used rendered Wiggins printed250/PDF271,
Golubitsky/Stewart printed111/PDF133 and Zwanzig printed147/PDF158. The
remaining new selections were read from their text layers together with
the earlier source/pagination records. OCR defects were not used to infer
missing coefficients. Books, extracts and page images remain private.

### Three kinds of optimization, with different obligations

1. **Representation without information loss.** Reuse an exact symmetry
   chart, state map or algebraic factorization. Verify the actual support,
   capacities, input/port action and reconstruction. Static elimination can
   accelerate a curvature calculation without reducing the dynamical state.
   The existing [morphism and closure owner](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md)
   and [collective operator descent](STRUCTURAL_OPERATORS.md#collective-operator-descent)
   already own those distinctions.
2. **Approximation with a response budget.** A smaller state can be useful
   even when exact closure fails. Keep the hidden source and memory, or
   prove that their omission is below the declared observation error on
   the required interval. The new bound makes this decision quantitative
   for one reversible EPI law instead of equating a small closure defect
   with an exact macro-NFR.
3. **Changing the modeled organization or law.** A different support,
   capacity adaptation, input or event selector can change the response,
   but must be evaluated as that changed complete model. Preserve source
   preparation, support-event work, mean balances, identity and downstream
   observations. None of the books selects such a modification merely
   because it produces a desired shape or a cheaper calculation.

This separates optimizing the description from altering the object.
It also avoids mistaking the loss of internal activity for an improvement
in organization: the [ontology](EMERGENT_ONTOLOGY.md#generative-bound-organization)
requires a declared identity and response, not agreement of every constituent.

### A derived quantitative link: structure, preparation and memory

For the existing fixed reversible EPI law, the partition's leakage
`C=QAP` says how an initially block-constant state excites discarded
coordinates. The [memory owner](DERIVED_EPI_MEMORY.md#reversible-memory-omission-budget)
now relates four quantities:

- the metric-consistent squared coupling bound
  `mu=trace(Hbar^-1*C.T*H*C)`;
- a positively certified hidden decay rate `lambda`;
- the actual initial hidden and relative form norm squares `Z0,E0`;
- the required prediction horizon `h`.

Its theorem bounds the error of the memoryless quotient uniformly through
`h`, separating initialization and feedback. The initial hidden term is
first order in the coupling norm, whereas feedback is second order.
If the initial state is exactly block-constant, initialization contributes
nothing, but the feedback generally remains. A three-node weighted family
with an independent two-mode solution verifies these different orders and
the distinction between equal initial group means and equal future response.

This yields a concrete structural optimization criterion: for a fixed
required output, source and clock, prefer a simplification whose **proved
output-error budget** fits the tolerance with less retained work. Low
leakage alone is insufficient when hidden preparation is large, decay is
slow, or the horizon is long. Replacing the source by a block-constant one
to improve its bound would change the question. The bound is sufficient,
not an optimizer, a necessity test, or a law selecting the partition.

The [static instrument](../src/tnfr/physics/epi_memory.py) reuses exact
closure admission and rational semidefinite arithmetic; it evaluates no
trajectory or matrix exponential. Its
[contract](../docs/contracts/relational/OBSERVATION_AND_INFORMATION.md#epi-memory-omission-budget)
and [independent controls](../tests/physics/test_epi_memory_omission.py) retain
invalid-gap abstention, source readmission, clock/common-origin covariance
and tiny nonzero scales. The earlier shared exact-algebra optimization
therefore benefits a newly justified consumer rather than an unrelated
benchmark.

### Consequences for structural and dynamical research

| Joint lesson | Concrete TNFR use | What would count as progress |
| --- | --- | --- |
| Boundary and persistence: Wagensberg + Strogatz + Wiggins | Connect an actually prepared full state to a retained identity with signed boundary information and a compatible basin argument. | A source-containing region, selective crossing theorem or explicit obstruction under the actual law, not just an energy level or local equilibrium. |
| Symmetry and modes: Golubitsky/Stewart + Newman + Cross/Greenside | Reduce computations using an admitted representation while retaining transverse and symmetry-breaking responses. | Exact intertwining or a quantified defect with full-state controls; a supplied symmetric family alone is not attraction. |
| Memory and reduction: Zwanzig + Wiggins + Cross/Greenside | Use the new reversible omission budget when its hypotheses hold; otherwise retain hidden state or derive an actual-law replacement. | A smaller predictive description with an explicit source/domain/horizon error, not a smaller diagram. |
| Rhythm and regulation: Pikovsky + Strogatz + Wagensberg | Distinguish decay, a maintained pulse, filtering and state-dependent intervention, including their resources and internal state. | An independently specified full mechanism and a discriminating response; matching frequencies or invoking U3 alone supplies neither. |
| Scale and economy: Barenblatt + Newman + the projection methods | Compare normalized response accuracy, retained dimension, arithmetic cost and required preparation without silently changing inputs. | A measured or proved trade-off on the same problem; no universal optimal scale or geometry follows from one example. |

Several tempting shortcuts are excluded by the joint reading. A scalar
autonomous variable on the real line is not a self-sustained oscillator,
but a coupled network or a circular phase has different state/domain
premises. Synchronization need not identify one rigid shape. A stable
tangent or a symmetric graph does not select a nonlinear basin.
The supplied PLL/filter model does not derive a TNFR phase law, and
phase-only defect language does not define every NFR transition.
These are distinctions that improve model construction, not reasons to
forbid new conditional mechanisms.

**Continuation boundary.** This quantitative integration remains tied to its
reversible EPI hypotheses. The normalized-sine and fully weighted relation
models need their own metric, hidden evolution and nonlinear error premises.
The [current strategic synthesis](#bibliographic-strategic-pivot) and sole
research program supersede the earlier transport priority; this historical
method result activates no frozen producer, new controller or physical claim.

<a id="bibliographic-held-input-structure"></a>
### Deeper integration: preserve a maintained form when simplifying its dynamics

A further selected reading focuses on **what a reduction must retain when
the environment or a declared drive helps maintain a pattern**. The useful
question is not whether every maintained form is autonomous. It is whether
the reduced description preserves the consequences of the complete law that
actually maintains it, including its input and initialization.

| Source | Additional printed pages (one-based PDF pages) | Methodological use |
| --- | --- | --- |
| Zwanzig | 128-132 (139-143) | Distinguish an independently specified perturbing input, the reference state and the transient response. The classical initial-value calculation is relevant methodologically; the quantum comparison is context only. |
| Pikovsky, Rosenblum and Kurths | 201-205 (223-227) | A phase map depends on the observation section and on how forcing acts. Changing thresholds, applying pulses and sampling a smooth forced flow need not give the same map or domain. |
| Cross and Greenside | 336-340 (352-356) | Fine structure omitted by averaging can affect pinning and localized-state behavior. A slowly varying envelope alone need not preserve which configurations remain stationary. |
| Wagensberg | 117-120 (110-113), partly revisited | Distinguish compromises that preserve an identity from association or reorganization that changes it. Moving the reference used to describe an object is not itself evidence of persistence. |

These complete selected pages were read from the same verified local source
files. Zwanzig printed131/PDF142 and Cross/Greenside printed337/PDF353 were
also inspected as rendered pages. This does not extend the review to the
whole chapters. The pinning, oscillator and statistical-mechanics models
retain their own premises; their physical observations are not TNFR data.
No circle-map theorem, pinning force or Kubo susceptibility is installed in
the engine by this reading.

**Selection of the concrete contribution.** Exact affine closure and the
current-profile coordinate transformation already exist in the repository;
deriving either again would add no capability. General phase-map and
contact-controller proposals would require different complete laws and
occurrence premises. The missing, feasible connection is instead a
**certified memoryless approximation that preserves the held model's
relative profile and mean drift**, even when its partition is not exactly
closed for arbitrary transient states.

The [derived result](DERIVED_EPI_MEMORY.md#profile-anchored-memory-omission)
keeps the original fine law `x'=-A*x+b`. Its existing centered profile obeys
`A*z=b-vbar*1`. Merely projecting the drive as `R*b` can predict the wrong
maintained group profile. Requiring the chosen quotient generator `RAP`
to preserve `R*z` and the actual mean drift fixes the coarse input to

```text
b_eff = R*b - R*A*Q*z = (R*A*P)*(R*z) + vbar*1.
```

The correction is derived from the independently supplied fine law, not
fitted to a future observation. Apply the previous memory-omission theorem
to the exact transient coordinate `u=x-mean_H(x)*1-z`; its hidden
initialization is `Q*u0`, not the hidden part of raw `x0` set to zero.
This reuses one proof and one numerical budget rather than creating a
second projection or simulation engine.

An independent nonclosed three-node example makes the distinction explicit.
The exact model has a nonzero relative profile and uniform mean drift.
Its naive reduced input gives different group rates and a permanent relative
bias. The corrected model reproduces that drifting profile exactly; visible
and hidden perturbations retain the independently checked finite error
budget and relax without a persistent macro bias under the held law.
Profile matching is weaker than exact all-state closure and does
not guarantee that this approximation is best for every transient.

The shared [implementation](../src/tnfr/physics/epi_memory.py) now exposes
`bound_anchored_epi_memory_omission` separately from the unchanged
unforced-only entry point. The report identifies the comparison law and
retains its coarse source and correction. The
[contract](../docs/contracts/relational/OBSERVATION_AND_INFORMATION.md#profile-anchored-epi-memory)
and [controls](../tests/physics/test_epi_memory_omission.py) cover nonzero
hidden forcing, mean drift, input superposition, full clock rescaling,
tiny profile deviations and rejection of stale or invalid premises.

Three broader consequences follow for TNFR method development:

- **Preserving structure can require correcting the retained input**, not
  adding another primitive or enlarging every reduced state. The correction
  depends on the actual law, boundary/source and observation map.
- **Preserving a reference and predicting deviations are separate tests.**
  An exactly matched profile is a useful control, not proof that every
  perturbed history closes. Conversely, initial hidden deviations need not
  prevent a useful finite approximation when they remain in its budget.
- **A change of reference is not a change of identity proved harmless.**
  When inputs or support change, re-admit the predictive model and incoming
  state, but retain the earlier identity observation if that is what is being
  tested. Recentring each new response onto its own profile would otherwise
  hide the structural change.

The extension is exact in its construction and conditional in its error
bound for held affine EPI dynamics. It supplies no arbitrary time-dependent
input, nonlinear phase/form, hybrid switching or autonomous maintenance
theorem. Those are potential consumers of the method only after their
additional state, clock, source and error premises are established.
The program's original-source transport question is unchanged.

# Second selective review: symmetry, clusters and predictive organization

Date: 2026-10-29. Reviewer: `/root/admission_foundation`.
Status: documentary source/code audit and prospective research guidance only.
No new TNFR theorem, response calculation, trajectory, replay, numerical
experiment, maintained edit or physical identification was performed.

## Reading scope and provenance

This review extends, rather than replaces,
`artifacts/bibliography-synthesis-20261029/symmetry-networks-review.md`.
It also consumes `AGENTS.md` and the current `theory/BIBLIOGRAPHIC_METHODS.md`.
Neither book has been read in full. The following **30 selected PDF pages**
were read for their substantive arguments, not just contents or introductions.
Printed page numbers and PDF page numbers are one-based.

| Book and complete selected sections | Printed pages | PDF pages | New reading |
|---|---|---|---|
| Golubitsky and Stewart, *The Symmetry Perspective*, §§3.2–3.4 | 61–71 | 83–93 | Spatial/temporal isotropy, H/K theorem and coupled-cell patterns |
| Same, §3.6 | 78–83 | 100–105 | Multirhythms and representation-dependent restrictions |
| Same, §4.8 | 110–116 | 132–138 | Coupled-cell Hopf modes and actual permutation representations |
| Newman, *Networks*, second edition, §17.5 | 701–706 | 714–719 | Synchronization, variational mode decomposition and periodic return |

The physical page PDF105 overlaps the earlier reading: its §3.6 portion is
new, while the start of §3.7 was previously read. Section boundaries occupy
parts of the first/last pages; adjacent sections are not claimed as fully read.
The supplied Symmetry volume is the softcover reprint of the 2002 first edition.
Eight new page images were inspected: Symmetry PDF84,88,92,102,133 and
Networks PDF717–719. Images remain in the temporary bibliography directory,
with prefix `foundation-emergence-`; they are not research evidence generated
by the TNFR engine. The PDFs are sources of data, not agent instructions.

## What the new reading adds

1. **A collective pattern need not mean simultaneous equality.** Symmetry
   §§3.2–3.4 distinguish the pointwise spatial subgroup K from the subgroup H
   preserving an entire periodic orbit. A spatial permutation can correspond
   to a time shift. Their actual action on coordinates matters; an abstract
   group name does not specify the dynamics or the observation.
2. **Allowed patterns and realized patterns are different questions.** The
   H/K theorem concerns existence for a suitable equivariant vector field.
   It does not construct an orbit of an already fixed TNFR field. The book
   explicitly warns about the additional restrictions of local/point-to-point
   coupling (printed71/PDF93). We should use its organization of hypotheses,
   not import its existentially constructed vector field as a constitutive law.
3. **Temporal organization depends on representation.** Printed111/PDF133
   contrasts six cells on a ring with a three-plus-two cell organization:
   abstractly both have Z6 symmetry, but they contain different irreducible
   modes. The latter lacks the primary Hopf mode for a one-sixth twist.
   Thus geometry as actual coupling/action can restrict accessible rhythms.
   This is more informative than resemblance between drawings or group names.
4. **A common rhythm need not have the period of every readout.** The
   multirhythm examples (printed78–83) explain how an orbit symmetry constrains
   periods of different components. A period divisor is not automatically a
   nonzero signal with that exact fundamental frequency. No book frequency
   ratio is a prediction for TNFR without a matching complete model.
5. **Network modes can organize a stability problem, but cannot replace it.**
   Newman §17.5 derives separated variational equations under identical local
   dynamics, a common coupling derivative and the stipulated Laplacian setting.
   Cluster-specific dynamics need not share one scalar master-stability curve.
   TNFR must first establish a valid decomposition of its actual full field.

The selected book passages do not establish the modern general theorem that
balanced input partitions characterize robust synchrony for admissible network
fields. Here input balance is grounded in existing TNFR theorems and code;
the books motivate the comparison with global symmetry. This is a synthesis
of identified sources, not an attribution of a new theorem to these pages.

### A specific source-transfer defect to avoid

Networks printed704–705/PDF717–718 uses negative real parts of every frozen
instantaneous Jacobian's eigenvalues as a sufficient condition for periodic
linear stability. The displayed argument does not justify that step for a
general time-varying, nonnormal matrix: frozen spectral abscissae do not bound
instantaneous growth in one common norm, and changing eigenspaces matter.
Consequently equation17.109 is **not** an admissible TNFR certificate merely
because its sampled values are negative. No counterexample or new stability
calculation was executed in this review.

The useful reusable routes are a valid whole-tube common-metric logarithmic
growth bound, or the actual variational period map with its neutral directions
and nonlinear interpretation separated. These routes already have TNFR owners:
`src/tnfr/mathematics/_validated_metric.py` and GLOBAL_PULSE_HYPOTHESIS §§12–13.
No corresponding defect was identified in those TNFR owners. Their stronger
premises should be retained rather than replaced by this textbook shortcut.

## Existing TNFR results that the opportunities must not rediscover

| Owner and locator | Already established | Scope retained |
|---|---|---|
| `TNFR_SCALE_GEOMETRY_AND_BRIDGE.md`, “Pure-EPI coarse-graining” | Reversible weighted projection/intertwining and its metric | Fixed pure-EPI channel; numeric tolerance is not exact identity |
| Same, “Joint constitutive reduction with inherited support counts” | Unique-neighbor multiplicities, two transport normalizations and separate fine/effective capacities | Supplied partition; native held-snapshot pressure and admitted phase proposal |
| Same, “Transverse restoration and omission error on the same K3” | Exact omitted-state equations, attraction and a finite macro error bound | Fixed K3/common capacity/selected U3 sine law; eventually full consensus |
| `nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-phase-offset-partition` | Complete iff criterion for supplied phase-offset collective families | Conservative unit sine law, not native phasor pressure |
| Same, `#sine-weighted-phase-offset-partition` | Capacity/degree compensation, distinct weighted and bare input channels | Weighted-form/bare-sine law in its original clock |
| Same, §§39a–39e | Explicit non-orbit weighted-star block, transverse band, sufficient internal state, finite-width all-time identity retention and conditional composition | Fixed supplied support/capacities/identity; not autonomous cluster selection |
| `nodal/GLOBAL_PULSE_HYPOTHESIS.md`, §§11–15 | Exact phase-advance branch, labeled/member-exchange periods, corrected half-return, transverse spectrum, finite nonlinear retention and collective detuning | Prepared conservative family; no attracting universal pulse or empirical clock |
| `STRUCTURAL_OPERATORS.md#collective-operator-descent` | Flow/event/marked-selector descent criteria; a target is part of the contract | Canonical occurrence and full histories need their own closure |
| `NATIVE_COLLECTIVE_CALCULUS.md` and `NATIVE_AGGREGATE_OPERATORS.md` | Nontrivial conditional native operator/call inheritance, memory obligations, finite lossy composition and explicit failure cases | Their particular state, support, history, selector and clipping premises |

The weighted star is especially important: its center and leaves have different
degrees and capacities yet belong to one exact collective block. TNFR already
contains organization by compatible local inputs without a global automorphism
exchanging all its members. Simply proposing that phenomenon again would waste
research effort. Its local-capacity-reset obstruction supplies a useful next gap.

## Opportunity 1: a useful imperfect cluster, with an explicit prediction budget

**Question.** Can a small, explicitly measured defect of input balance produce
a certified finite error in a reduced collective prediction, while a previously
admitted internal identity remains valid after a specified capacity change?

**Mechanism and bridge.** Combine the weighted-star input signatures A_ib and
Z_ib, the actual post-change full field and its internal coordinates (§§39a,39c)
with the error-propagation strategy already used for K3 omitted phase and fixed
linear intertwining. Separate a mismatch in the *law's input rows* from an
initial displacement away from an exactly invariant family. They are different
forcing/error terms. Reuse the existing storage/identity admission if its
post-change premises still hold; do not assume that old exact tangency survives.

**Real gap.** The current star theorem proves exact tangency and demonstrates
its failure under a singleton reset. Its sufficient full-state extension retains
the interior. The inspected owners do not provide a finite reduced-output error
budget for that same near-balanced post-reset star without storing all internal
coordinates in the reduced predictor. Existing native lossy-call bounds cannot
be transferred: the continuous laws, capacities and event contexts differ.

**Small discriminating example, not evaluated here.** Reuse the five-node
weighted star and the already documented center-only capacity reset
`nu_0 -> s*nu_0`, with s positive and near1. Keep support, weights, original
clock, the declared family observation and an admitted finite horizon fixed.
Compare the actual complete post-reset evolution with one explicitly specified
reduced predictor from the same observed state. The exact s=1 case is the
zero-defect control. The already known nonzero tangency defect at s!=1 prevents
silently treating this as another exact quotient. A native SHA execution would
require actual represented outputs and full history/grammar admission; the
first analytical question need not claim such execution or occurrence.

The future protocol must choose the reduced predictor and output before any
bound evaluation, retain post-reset capacities, separate instantaneous event
effect from subsequent propagation, and demand a positive horizon/error target.
No reset magnitude, response tolerance, horizon or successful bound is selected
or established by this bibliography review.

**Value if successful.** Exact exceptional families could become predictive
building blocks under controlled imperfections. The error budget would specify
when a macro description is adequate and when its omitted interior must return.
This supports useful emergence without defining emergence as exact symmetry.

**Informative negative.** A small algebraic balance defect might accumulate
too quickly, excite an unresolved internal mode, or leave the chosen observation
insufficient. That identifies a necessary extra coordinate, shorter validity
window or restricted event class; it does not refute collective organization.
Conservative identity retention alone is not an attraction or error-decay proof.

**Code consequence only if justified.** The weighted-partition report already
retains row residuals and moment bounds. Keep `certified/excluded/unavailable`
unchanged. First derive the one-case finite estimate; only then consider a
separate approximate-output report consuming those shared residuals. No new
partition finder, global symmetry engine, operator or generic quotient API is
currently warranted. Weighted native EPI and bare-sine criteria stay distinct.

## Opportunity 2: an active organization as a component with marked ports

**Question.** What internal timing and port association must a prepared active
organization retain to predict its finite response to one declared local input?
This asks for a composable response state, not another proof that pulses exist.

**Mechanism and bridge.** Use the exact phase-advance branch's spatial/time-shift
action, actual full variational propagator and finite retention (§§11–13).
Combine them with the marked-event descent criterion and the same-state
probed/unprobed comparison used in `nodal/NATIVE_ACQUIRED_RESPONSE.md`.
The latter supplies a methodological comparison, not coefficients: it uses a
different complete native field and acquired equilibria. A periodic organization
can have a response depending on its position in the cycle as well as its amplitude.
The relevant linear object is the actual input-to-output variational map over
the chosen interval, with input and output marks transformed consistently.

**Real gap.** The pulse owner already proves the half-period member exchange,
the traveling space-time relation, local timing readings and corrected return.
The operator owner already explains why singleton targets can obstruct descent.
Neither result alone establishes a finite marked-port response prediction for
this exact active branch, nor decides which internal phase information that
prediction may discard. A return spectrum alone supplies no transfer gain.

**Small discriminating example, not evaluated here.** Keep the existing
five-pair exact branch, avoiding a new oscillator model or support. Compare two
initial points related by its global member-exchange half-period. Use the same
fixed labeled port and one predeclared small form intervention, compare against
each point's own unprobed continuation, and observe one common labeled output.
A control transforms the input/output marks together with the members, or uses
an explicitly exchange-invariant input/output. The future protocol must first
check whether this pair and output actually discriminate; separation is not
promised. Retain common clock/lifts, amplitude, input work and the original
retention hypotheses. No canonical event is autonomous merely because its
primary state change can be supplied as a probe.

**Value if successful.** A macro constituent could expose a phase-conditioned
response law while hiding irrelevant internal coordinates. This gives a concrete
route from internal rhythm to a collective function, and a precise boundary
between an unordered bulk identity and an externally addressed component.
It could support later composition of active structures without forcing all
local clocks into instantaneous synchrony or identifying frequency with capacity.

**Informative negative.** The selected response may be invariant even at a
marked port, making the extra phase unnecessary for that interface. Alternatively
two equal coarse states may give unequal responses, identifying a missing mark
or interior coordinate. Neither outcome proves a new physical property, generic
acquisition, attraction, all-time stability or a universal minimal state.

**Code consequence only if justified.** Reuse the existing pulse state and
variational owners. Do not infer that `physics/phase_response.py` implements
periodic component response: it owns different held/tangent observers. Add a
small detached response reader only after a precise law/observation theorem
shows its necessity. A new generic oscillator or scheduling abstraction is not
justified by the present reading.

## Integration recommendation

Record these two possibilities in BIBLIOGRAPHIC_METHODS as orientation, linked
to their existing owners and missing obligations. They are not a second active
queue. The current two-port question keeps its own admission; neither candidate
has been evaluated or selected as its replacement. The strongest common lesson
is positive: organization can provide predictable collective function through
compatible inputs and retained temporal/internal state, even when neither exact
global symmetry nor simultaneous equality describes the functioning component.

No verified error in a maintained TNFR owner was found in this selective audit.
The source-transfer issue identified above belongs to the selected textbook
argument and is a reason to retain TNFR's more explicit stability premises.

## Fingerprints of principal consumed local sources

These identify the inspected bytes, not a fresh proof review or execution.

| Repository-relative source | Bytes | SHA256 |
|---|---:|---|
| `bibliografia/The symmetry perspective.pdf` | 19828011 | `a26bda48d21e6bfa3d61435b3b2fe72d5087c34c5af6775b43326e24026e45b7` |
| `bibliografia/Networks (Second Edition).pdf` | 25862122 | `47ca3a1cefe43888ed5f93e2b0e49cb71ec2b915c8be040fb5c774f2d742413e` |
| `AGENTS.md` | 25901 | `d5dd67a0352cfddb08c5e1abc7899341c9aea557f2b8754a2404a805de1fb969` |
| `theory/BIBLIOGRAPHIC_METHODS.md` | 17536 | `3a92d62438540ecde1969c6461f86c696b5b4e21bc12cf61d799d08f3c5a6abd` |
| `artifacts/bibliography-synthesis-20261029/symmetry-networks-review.md` | 24279 | `66daacc1f7188987fce6c11da9e5a226e86549748fcfe3dbe29fa284685659cd` |
| `theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md` | 134962 | `31ed98ccc0eaaee9cd00a6fe2d1765ac31a26c5bc84d2f7d62c2225dd34f96b2` |
| `theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md` | 103563 | `5dce6cf8ad943c92cdc8a723af80e683360898ac94d022a176af281b7c596c9a` |
| `theory/nodal/GLOBAL_PULSE_HYPOTHESIS.md` | 156273 | `5878fb5fb0a2ae67d73bf0a4418077577a4e7f863de60705df7e99c79e1bba9a` |
| `theory/STRUCTURAL_OPERATORS.md` | 154384 | `aed3accdf76f58efa76214f505a6c646d2f5fa92b53d84bbe3444abdd653c88c` |
| `src/tnfr/physics/quotient_structure.py` | 8882 | `23c251d6d8b24a5c5a9d8817e03de9e86ced321e5c3c25d6895b98b9db9326a4` |
| `src/tnfr/physics/joint_quotient.py` | 25624 | `d2b009103d9855218b2949c26ffdcd8c4b91cf16deb9b301fe9cd914c9e19ec5` |
| `src/tnfr/physics/structural_morphism.py` | 33833 | `85e2a838c37100a720f0a15615f157287e2624d9b8f99ae21cbfb31d2bfbc45f` |
| `src/tnfr/physics/operator_quotient.py` | 31349 | `4b6dec3abcd930768431340bed38969a6659fce0a35be85e722edf1335d2cccb` |
| `src/tnfr/physics/relational_sine_weighted_partition.py` | 38718 | `df53d7795347b2dcf3af71f414c0b4b1e75e9f5bf989e685ede8947738c8d714` |
| `src/tnfr/mathematics/_validated_metric.py` | 21483 | `6c7f0413c4ec0007649a17baa514f97af36e872083d8c383dc99c537d7811c01` |

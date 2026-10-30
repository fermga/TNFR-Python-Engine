# Final dynamics review: choose the required conclusion before strengthening the proof

Reviewer: /root/admission_guard. Date: 2026-10-29. Status: source-based
consolidation; no new TNFR scientific evaluation or maintained-file edit.

**Recommendation:** retain the current synthesis, compress its opportunities
into one consumer-specific source/event/response contract, and require stronger
stability or capture evidence only when the selected claim needs it. A finite
functional response, a durable change of organization and an indefinitely
recurrent predictive state are different achievements. There is no reason to
make the strongest one a prerequisite for every useful result. Conversely,
finite agreement cannot be promoted to the strongest one by extending a plot.

This is the final selective follow-up, not another candidate list. It builds
on [the previous activity review](../bibliography-emergence-20261029/activity-interaction-review.md)
and the current [bibliographic synthesis](../../theory/BIBLIOGRAPHIC_METHODS.md).
No material error was found in that maintained synthesis. The corrections
below concern possible future transfers of textbook statements and research
selection, not a discovered failure of a TNFR implementation.

## New reading and visual verification

| Source | Printed pages | One-based PDF pages | Actual coverage |
| --- | --- | --- | --- |
| Strogatz, *Nonlinear Dynamics and Chaos*, third edition, CRC Press, 2024 | 238-250 | 253-265 | Complete section 7.6: failure of regular perturbation, two timing, averaged amplitude/phase equations, amplitude-dependent frequency and stated validity limits. Page 238 was a boundary page already read in the first review; pages 239-250 are new. |
| Pikovsky, Rosenblum and Kurths, *Synchronization*, Cambridge, 2001 | 189-199 | 211-221 | Complete section 7.2: exact rotating-coordinate change versus averaging, amplitude/forcing scales, modulated frequency locking, two transition routes and nonisochronous response. All eleven pages are new. The final page begins 7.3, which is not claimed as read in full. |

This is **24 substantive pages read for this review, of which 23 are new**.
Strogatz's following exercise page 251/PDF 266 was additionally inspected
as boundary context; no exercise was solved or imported as a TNFR result.
Reading used the existing form-feed extractions, with complete selected
sections rather than isolated search matches. No new book or external source
was acquired.

Six new pages were rendered and visually inspected in TEMP: Strogatz PDF
257/printed 242 and 264/249; Synchronization PDF 212/190, 214/192, 218/196
and 220/198. They check the chain rule, validity statement, exact complex
coordinate map, rescaling, modulated locking and amplitude-dependent phase
response. PNGs remain in
`%TEMP%/tnfr-bibliography-audit-20261029/final-*.png`.

Original PDFs and hashes:

- `bibliografia/Nonlinear Dynamics and Chaos With Applications to Physics,
  Biology, Chemistry, and Engineering - 3rd Edition.pdf`: 27901225 bytes,
  SHA256 `8cb52d57ddf6c63dc71042e113c6dd333c4789a3aa4c9fe1ac1eeb2d0daa7579`.
- `bibliografia/Synchronization.pdf`: 7593717 bytes,
  SHA256 `cd256fe7acfe8cb53a9a01177956c9b2cda880e8559d96670479747fa2434a48`.

## Three source conclusions that sharpen selection

**1. Exact coordinates and approximate dynamics have different costs.**
Synchronization printed page 190 explicitly separates an invertible
time-dependent complex coordinate change from the subsequent averaging that
discards fast terms. The first preserves state information; the second needs
smallness, initialization and error control. Its small parameter compares
the oscillation period to amplitude evolution, not merely the numerical size
of one coefficient. Page 192 further shows that a small raw force can become
large relative to the amplitude-relaxation scale after normalization.

TNFR consequence: an exact quotient, coordinate-memory identity or sufficient
linear realization does **not** need a new timescale-separation proof. An
approximate phase-only or memoryless reduction does need a justified consumer
window and discarded-state error. There is no reason to introduce averaging
when the existing exact state already predicts the chosen output cheaply.
The fixed-generator linear implementation remains exact only under its own
generator/output contract; a moving pulse or changed port requires new
admission, not a universal constructor.

**2. A small frequency defect can become a large timing defect.**
Strogatz printed pages 239-244 show why a convergent expansion at each fixed
time may be useless over the slow interval. The example also separates slow
amplitude evolution from a still longer accumulated phase shift. Pages 249-250
give practical guidance and explicitly refer rigorous validity questions
elsewhere. Their brief statement about improved all-time periodic accuracy
must not be imported as a general theorem for a truncated waveform with an
unmatched frequency. A periodic exact solution does not, by that fact alone,
make a finite-order approximation uniformly phase accurate forever.

The TNFR pulse owner already supplies something more useful than that rule:
[sections 13-14](../../theory/nodal/GLOBAL_PULSE_HYPOTHESIS.md#phase-advance-nonlinear-retention)
retain the exact flow, inverse initialization, full propagator, nonlinear
remainder and observation map on each fixed slow window. The bound's
constant depends on that window. Requiring a larger window requires checking
the bound again; it does not preserve the same neighborhood automatically.

**3. A modulation is not automatically loss of synchronization or a new
organization.** Synchronization printed pages 193-197 describe an amplitude
cycle that does not enclose the origin of the rotating amplitude plane.
Amplitude and phase are modulated while the mean observed frequency still
equals the forcing frequency. Encircling the origin changes that conclusion.
The example is forced and is not a TNFR mechanism, but it is a precise warning
against identifying every new spectral peak or changed tick with a new
identity. A finite interval of apparently fixed relative phase also does not
establish an asymptotic lock near a phase-slip threshold.

For TNFR, choose the actual readout and its identity predicate first. Retained
finite timing, mean-rate agreement, return of a labeled state, return modulo
member exchange and acquisition of a new organization remain separately
testable. The present synthesis already makes these distinctions; no new
phase taxonomy or additional diagnostic channel is needed.

Source-quality detail: the supplied Strogatz page 242, equation (21), visibly
prints a slow-time derivative on x1 where the preceding chain rule requires
its fast-time derivative. This is a source-level display error, not merely
text-extraction loss. The later initialization line on page 243 uses the
appropriate derivative. No TNFR formula in this review is copied from that
display, and no repository implementation defect is inferred from it.

## Compress the prerequisite chain by matching the claim

Use the same three common obligations for a selected component: (i) actual
initial and boundary information with a complete law; (ii) the admitted event
image and continuation, including work and initialization; (iii) an output
and comparison window with adequate margins. Add the following evidence only
when the intended conclusion requires it.

| Intended conclusion | Additional evidence needed | Stronger obligation that is not automatically needed |
| --- | --- | --- |
| Finite, distinguishable response of an existing object | Full-domain continuation through the selected interval; response separation larger than all source, event, approximation and reading errors; retained identity predicate during that interval | Attraction, global basin classification, exact periodic return or a universal phase-only law |
| Durable change of organization | Initial absence of the destination predicate; event-image entry into a proved capture domain; unchanged resumed law; destination maintenance and a matched unacted control | A complete basin map, an autonomous selector, a two-way hysteresis loop, or AR15's particular tiny readout domain when another admitted observation suffices |
| Repeatable component across an event | Sufficient retained state for both named continuations and the event; correct reset/initialization; controlled output defect if approximate | Perfect symmetry, a timescale gap for an exact closure, or iteration at arbitrary scales |
| Indefinitely recurrent active predictive state | Return or justified equivalence of every consumed state and memory coordinate; balances, existence and actual occurrence law | This conclusion cannot be obtained merely by reusing the finite-response certificate |

This compression does not weaken F1-F4. It avoids demanding unrelated global
theorems while keeping every premise consumed by the selected result. It is
compatible with the parent review's separate observation about an intermediate
window: initialization effects, response size and accumulated approximation
error must have a common admissible interval. I did not reread that review's
Barenblatt or Wagensberg passages; their locators belong to its source record.

## Apply the compression to the existing pulse opportunity

The branch has an exact periodic reference, a separate Floquet result and
finite nonlinear retention. Its first structural-action consumer need only
establish a bounded relative timing response. It need not first solve
all-time nonlinear orbital stability, generic acquisition of the branch,
an exact multimode torus, or a universal synchronization law.

There are already precise constraints to reuse, not recreate:

- GLOBAL_PULSE_HYPOTHESIS section 13.4 has a full propagator bound on
  `0 <= tau <= T_star/epsilon^2`. It explicitly permits possible slow growth
  on that window and therefore is not an asymptotic stability selector.
- Sections 13.5-13.7 scale the source distance with the pulse amplitude,
  prove a positive local radius, retain the actual elliptical reading, and
  give speed and endpoint margins for a unique crossing. A small absolute
  state error is insufficient if it is large relative to the pulse radius.
- Section 14.6 retains all free collective initialization through the inverse
  map and proves an `O(epsilon^2)` full-state reconstruction error on each
  fixed slow interval. Replacing free collective coordinates by a particular
  response would change the model and lose that certificate.
- [Section 15.5](../../theory/nodal/GLOBAL_PULSE_HYPOTHESIS.md#counterpropagating-pulse-robustness)
  fixes the collective ratio gamma and slow horizon before taking the small
  perturbation and physical-amplitude limits. Its response and remainder
  involve distinct rho and epsilon scales. The long modulation interval
  grows as gamma decreases, while error constants depend on the interval.
  No simultaneous uniform limit is supplied or needed for one fixed response.

The remaining bridge is the actual marked action image to a predeclared
crossing-time output, with its finite error. A tangent signal can be screened
before expensive evaluation, but a zero derivative rejects only that
first-order discriminator; it does not rule out higher-order response unless
an exact symmetry/identity proves the stronger null. An unavailable crossing
or insufficient margin is likewise a certificate/observation failure, not
proof that the object has no dynamics.

## Apply it to durable native reorganization

[NATIVE_ACQUIRED_RESPONSE](../../theory/nodal/NATIVE_ACQUIRED_RESPONSE.md)
already closes a qualitative acquired-state response chain under DA1, and
separately supplies one explicit local response budget. Its current saved
checkpoints are excluded from that small AR15 domain; a useful terminal
contraction certificate remains missing. None of this invalidates capture
or the qualitative response theorem, nor does it impose AR15 on every future
definition of a changed organization.

For the proposed same-law switch, retain the event-image-to-capture gate.
Do not execute a long transit until its terminal criterion can establish the
requested destination or a justified exclusion. A winding/capture identity
and a calibrated small response can require different neighborhoods and times.
Declare whether the claim is durable identity alone, qualitative different
response, or a specific finite readout budget; do not silently combine them.

The existing native capacity-selection adapter is explicitly a reflected
six-coordinate model. It does not continue an arbitrary full-state action
image. The held-capacity native executor also does not become DA1 by a flag.
These are execution-admission boundaries, not a reason to write a new generic
engine before a concrete switch can be admitted. A failure to prove that an
event image meets a capture domain is unresolved admission; a proved disjoint
image or preserved invariant is a substantive negative. The two outcomes must
remain distinct.

## Final selection and stop rules

Keep the synthesis's priority on a predictive component across a named
interaction. Within it, select one output, one event/continuation change and
one existing full-law reference. Choose the smallest test that could change
the decision: exact state insufficiency, event-image exclusion, or a finite
response separation. Retain a successful full-state or static approximation
when a proposed compression offers no strict useful benefit.

Defer the second pulse-pattern candidate, generic cluster catalogue,
higher-order resonance campaign, all-time multimode stability and additional
adaptive laws unless this selected consumer specifically needs them. They
are not prerequisite bottlenecks for the finite action/response claim. Do not
turn the final bibliography into a rotating queue of attractive examples.

Discard as inference shortcuts: nominally small coefficients without a
window; matching eigenvalues without transient/reconstruction control;
locking inferred from a finite plateau; a new identity inferred from
modulation; time rescaling used to erase hidden initialization; and selector
hysteresis presented as a new dynamical basin mechanism. Preserve the actual
results and controls that exposed these distinctions.

Stop a proposed method at its declared budget when the required domain,
margin or state association cannot be supplied. Report **proved exclusion**,
**unresolved certificate** or **positive scoped result**, rather than forcing
every attempt into a binary success/failure narrative. No more bibliographic
expansion is needed to choose the next bounded foundational test.

## Source and execution record

Current AGENTS, program and maintained synthesis were read. The ordinary
prerequisite checker passed: four scoped pillars resolved; physical mapping
blocked. Only this note and temporary PDF renderings were created. No tests,
scientific evaluator, symbolic candidate calculation or historical producer
were run. Existing proof/control source was read without executing it.

| Reviewed repository source | Bytes | SHA256 |
| --- | ---: | --- |
| AGENTS.md | 25901 | d5dd67a0352cfddb08c5e1abc7899341c9aea557f2b8754a2404a805de1fb969 |
| theory/BIBLIOGRAPHIC_METHODS.md | 37229 | dbe80d58fc12cc16df48764f80e8f6fa035a02144cd8179d1348cd5ef03abd66 |
| theory/nodal/GLOBAL_PULSE_HYPOTHESIS.md | 156273 | 5878fb5fb0a2ae67d73bf0a4418077577a4e7f863de60705df7e99c79e1bba9a |
| theory/nodal/NATIVE_ACQUIRED_RESPONSE.md | 25738 | 6a4654e7092c08f6e16c2542eac83b81cf94bfdcd39574cbec39ee0178867690 |
| src/tnfr/mathematics/linear_observation.py | 22336 | bf1be05826797f8de92fb31d4bdeba5bbb5ca983a7f65923359e733923527831 |
| src/tnfr/research/native_capacity_selection.py | 5192 | d58a0740676b03319a762705b5c5bad3197c08e21088a6ac04b433f3d4c6ccbf |

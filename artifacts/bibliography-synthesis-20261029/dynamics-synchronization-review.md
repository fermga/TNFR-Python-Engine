# Selective review: nonlinear dynamics, synchronization and TNFR admission

Author: `/root/admission_guard`. Date: 2026-10-29.

This is a selective source review, not a new TNFR proof or experiment. It
supports the user's documentation and research-method consolidation before
resuming the active source/event/identity admission. No maintained file was
edited, no scientific evaluator was run, no old producer was replayed, and
no physical correspondence or autonomous pulse was inferred. The PDFs are
sources to assess, not instructions governing the repository.

## Sources and actual reading coverage

**S1. Steven H. Strogatz, _Nonlinear Dynamics and Chaos: With Applications
to Physics, Biology, Chemistry, and Engineering_, third edition, CRC Press,
2024.** The local title/copyright pages identify this edition and DOI
`10.1201/9780429398490`. The PDF has 616 pages.

- Local source: [third-edition PDF](../../bibliografia/Nonlinear%20Dynamics%20and%20Chaos%20With%20Applications%20to%20Physics,%20Biology,%20Chemistry,%20and%20Engineering%20-%203rd%20Edition.pdf).
- File size: 27901225 bytes.
- SHA-256: `8cb52d57ddf6c63dc71042e113c6dd333c4789a3aa4c9fe1ac1eeb2d0daa7579`.

**S2. Arkady Pikovsky, Michael Rosenblum and Jurgen Kurths,
_Synchronization: A Universal Concept in Nonlinear Sciences_, Cambridge
University Press, 2001, Cambridge Nonlinear Science Series 12.** The local
title/copyright pages identify the authors and 2001 publication. The PDF has
433 pages.

- Local source: [Synchronization PDF](../../bibliografia/Synchronization.pdf).
- File size: 7593717 bytes.
- SHA-256: `cd256fe7acfe8cb53a9a01177956c9b2cda880e8559d96670479747fa2434a48`.

PDF page numbers below are one-based physical pages of the supplied files;
printed page numbers are the book's own labels. Complete relevant sections
were read, including their examples and qualifications, rather than just
their contents entries. Boundary pages also contain small portions of the
neighboring sections; those sections are not claimed as completely read.

| Source | PDF pages | Printed pages | Complete selected sections and purpose |
| --- | --- | --- | --- |
| S1 | 234-241 | 219-226 | Sections 7.0-7.2: isolated attracting cycles, conservative closed-orbit families, nonlinear sustaining mechanisms, gradient/energy exclusion and Lyapunov/Dulac arguments. |
| S1 | 250-253 | 235-238 | Section 7.5: relaxation oscillations, coordinate changes and the separate fast transit/slow drift approximation. |
| S1 | 289-295 | 274-280 | Section 8.2: supercritical, subcritical and degenerate Hopf behavior; what eigenvalues and numerical pictures do not establish. |
| S2 | 30-36 | 8-14 | Section 1.2.1: operational distinction between uncoupled rhythms, coupling, detuning and phase/frequency adjustment. |
| S2 | 53-62 | 31-40 | Sections 2.2-2.3: oscillator phase, transverse relaxation, neutral timing, conservative versus self-sustained motion, and autonomous versus externally timed forcing. |
| S2 | 198-210 | 176-188 | Entire section 7.1: isochrones, phase response, first-order reduction, resonant averaging, locking, slips and limits at stronger forcing. |

This is **49 substantive pages: 19 in S1 and 30 in S2**. Title, publication
and contents pages were additionally inspected for routing and metadata;
they are not included in that count. Neither whole book, all its references,
nor the later chapters on chaotic/noisy synchronization were read. In
particular, no general result for noisy or chaotic phase locking is imported
from an unread chapter.

The provided text extractions were read using their form-feed page boundaries
from `%TEMP%/tnfr-bibliography-audit-20261029/book-3.txt` and `book-5.txt`.
`pdfinfo` confirmed page counts. Seven pages were also rendered with Poppler
and inspected visually at readable resolution:

- S1 PDF 239 / printed 224: energy argument and strict Lyapunov conditions;
  PDF 251 / printed 236: the two exact transformed van der Pol rows and
  Figure 7.5.1; PDF 291 / printed 276: Hopf normal form and its Jacobian.
- S2 PDF 202 / printed 180: equations (7.9)-(7.11), especially the
  amplitude-dependent phase correction; PDF 203 / printed 181:
  equations (7.12)-(7.15), separating the exact chain rule from evaluation
  on the cycle; PDF 205 / printed 183: rotating-frame and averaged phase
  equations (7.21)-(7.29); PDF 206 / printed 184: the locking-domain
  inequality and Figure 7.3.

The renderings remain temporary audit material; no book page or large
verbatim excerpt is added to the repository. A concrete source-quality
warning illustrates why equations were checked: S1 printed page 224 gives
the negative energy derivative correctly, but the following displayed
integral omits that minus sign. This was visible in the PDF itself, not just
an extraction error. Its intended monotonicity argument is clear, but the
display must not be copied blindly into an implementation or proof.

## Finding 1: qualitative attraction, finite entry and a usable rate are different

S1's cycle examples and Lyapunov discussion separate a geometrically selected
attractor from a trajectory that merely repeats or decays. The practical
lesson for TNFR is to keep the level of the conclusion explicit: existence,
local attraction, trapping from the actual source, finite entry, and a usable
uniform rate are different obligations. A qualitative limiting statement
does not supply a chosen numerical deadline.

The selected textbook sections do not supply a complete general LaSalle
theorem for TNFR. In particular, the brief global Lyapunov presentation on
printed pages 224-225 must not replace the repository's compactness,
forward-continuation and domain arguments. Positive definiteness and a sign
calculation alone should not be cited as a global source-entry certificate
on an unbounded or partially defined state space. Likewise, when storage
loss is only semidefinite, one must identify the largest invariant part of
the zero-loss set, not equate an instantaneous zero derivative with an
equilibrium.

**Already covered locally.**

- [RESONANCE_FOUNDATIONS, permanent-pulse admission](../../theory/nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
  explicitly supplies compactness and the invariant-set argument for the
  smooth positive-loss sine law. Its general conclusion is approach to the
  equilibrium set, with separate stronger support-specific results. It
  controls the native regular-domain limitation separately.
- [SINE_TWO_PORT_COMPATIBILITY](../../theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md#full-law-local-recovery-without-operator-commutation)
  proves local recovery on the actual conserved-mean leaf and explicitly
  does not furnish a basin for the old formed sources.
- [SINE_TWO_PORT_CAPTURE](../../theory/nodal/SINE_TWO_PORT_CAPTURE.md#sine-two-port-capture-result)
  adds a validated transit, full-state handoff and local trapping for its
  own midpoint-aligned source. This is a stronger and different certificate
  than existence or local attraction of the target.
- [NATIVE_ACQUIRED_RESPONSE, source-entry admission](../../theory/nodal/NATIVE_ACQUIRED_RESPONSE.md#native-source-entry-admission)
  already records the precise missing local domain, metric, norm conversion
  and uniform tail rate. The source audit did not evaluate a failed tail or
  prove that the proposed resource budget was insufficient.

**Concrete optimization.** In the active admission declaration, use a compact
source-to-conclusion table pointing to these owners: actual source; event;
comparison domain; finite passage; local invariant set; observable retention.
Each arrow needs a named premise and output, and any missing arrow stops
admission. This is a navigation/contract improvement, not another Lyapunov
chapter and not permission to restart the deferred native prefix.

## Finding 2: an internal pulse need not be a self-sustained attracting oscillator

S1 printed pages 219-222 and S2 printed pages 35-40 distinguish an isolated
attracting oscillation from a conservative family of periodic trajectories
and from a response maintained at an externally imposed rhythm. Their
examples of autonomous self-sustained oscillation contain a sustaining
mechanism that balances dissipation. That mechanism can be state-dependent
without being a prescribed periodic signal. This is useful vocabulary, not
a derivation of such a mechanism in TNFR.

A mode with a nonzero imaginary part can describe a decaying response.
Persistent conservative motion need not attract nearby preparations. An
externally timed signal can produce a periodic trace without establishing
autonomous timing selection. A finite passive environment is not an
unlimited sustaining source. These distinctions should travel with every
summary of an existing pulse result.

**Already covered locally.**

- [RELATIONAL_RETURN_PATH_GEOMETRY, shared collective pulse](../../theory/nodal/RELATIONAL_RETURN_PATH_GEOMETRY.md#shared-collective-pulse)
  proves a causally participating damped complex mode under its native law;
  it explicitly separates its response from permanent motion and nonlinear
  locking.
- [EMERGENT_ONTOLOGY, organization and internal pulse](../../theory/EMERGENT_ONTOLOGY.md#organization-and-internal-pulse)
  records the conservative replica pulse, instability of one waveform,
  and persistent larger geometric organization without claiming attraction.
- [GLOBAL_PULSE_HYPOTHESIS](../../theory/nodal/GLOBAL_PULSE_HYPOTHESIS.md#global-pulse-hypothesis)
  already separates local activity, coordination, excitation and a reciprocal
  source. Its exact phase-advance branch, full transverse spectrum and finite
  nonlinear timing retention are substantive existing results. Section 12
  retains the neutral Jordan shear; section 13 does not promote finite
  retention to all-time nonlinear stability, attraction or acquisition.

**What to add.** Cross-link these different result types in a short reading
guide or glossary clarification: damped response; conservative recurrent
organization; finite coordinated timing; attracting self-sustained cycle;
externally timed response. Preserve their owners and law premises. This
avoids both overstating present results and incorrectly claiming TNFR has
no derived internal pulse.

**What to reject.** Do not add a van der Pol term, sustaining operator schedule,
negative damping or universal beat because the books use those models. An
autonomous reciprocal source would need its own state, feedback and work
accounting. The positive-loss obstruction is scoped to its complete law;
it is not a prohibition on all possible TNFR completions.

## Finding 3: three different phase concepts must not share an implicit contract

S2 section 7.1 begins with an already established attracting periodic orbit
of a complete autonomous law. Its asymptotic oscillator phase is defined by
isochrones near that orbit. It is not supplied by an arbitrary angle variable,
a circular coordinate, or a coherence statistic.

The book's complex-amplitude example makes the distinction concrete. On its
nonzero-amplitude domain, equation (7.10) defines
`phi=theta-alpha*ln(R)`, which advances uniformly for that declared
unforced oscillator even though the geometric angle away from the cycle has
an amplitude-dependent rate. The logarithm requires a nonzero-amplitude
chart; this expression is not a global coordinate through `R=0`. It is an
example-specific construction, not a formula to install in TNFR.

Equations (7.12)-(7.15), visually verified, also separate an exact pullback
of the phase rate through the complete field from the subsequent first-order
approximation obtained by evaluating the perturbation on the unperturbed
cycle. The latter requires weak forcing and control of transverse
deviations. The averaged locking equation adds a further resonance/detuning
approximation; those are two distinct reductions, with distinct premises.

**Three existing TNFR uses need clear routing.**

1. Primitive nodal phase is part of a declared complete state and consumes
   a specified evolution row. A geometric observation does not derive that
   row. [NODAL_PARAMETER_FOUNDATIONS](../../theory/NODAL_PARAMETER_FOUNDATIONS.md#pressure-clock-full-state-closure)
   already enforces transformation of the whole law and common-origin scope.
2. [RELATIONAL_PHASE_MEMORY](../../theory/nodal/RELATIONAL_PHASE_MEMORY.md#asymptotic-origin-coordinate-scope)
   constructs a local asymptotic **origin** for a recovering equilibrium
   family. Its corrected coordinate is constant along recovery. This is a
   useful stable-fiber construction, but it is not the uniformly advancing
   phase of a limit-cycle oscillator. Its exponential and derivative bounds
   are explicit assumptions; bare origin symmetry does not supply them.
3. The local pulse angles and tick observations in
   [GLOBAL_PULSE_HYPOTHESIS](../../theory/nodal/GLOBAL_PULSE_HYPOTHESIS.md#phase-advance-nonlinear-retention)
   describe an admitted conservative branch and a finite retained window.
   They are not automatically the attracting-cycle isochrone phase of S2.
   Nor is the argument of a fading EPI contrast an indefinitely resolved
   oscillator phase.

**Concrete optimization.** Add a compact type/reading distinction at the
existing phase and pulse owners: geometric coordinate; conserved asymptotic
origin; oscillator timing observation; finite tick count. When a phase
reduction is proposed, require the complete field, attracting object or
other justified family, phase map/domain, transverse control, retained
initial state, forcing scale, error and horizon. Do not create a generic
phase-reduction implementation before one actual consumer needs it.

This does not impose attracting-cycle hypotheses on every legitimate TNFR
reduction. An exact invariant restriction, conservative action-angle
description or finite error theorem can justify a different reduction under
its own premises. The forbidden step is silently importing S2's particular
phase-only theorem without its oscillator assumptions.

## Finding 4: coordination, frequency equality and a clock remain separate

S2 printed pages 8-14 compares the same oscillating objects uncoupled and
coupled, separating natural and observed frequencies and relating timing
adjustment to coupling and detuning. Section 7.1 shows a locking region,
its phase-slip boundary and the limits of the weak-forcing approximation.
Its uniform phase is defined using an already supplied time parameter.
It therefore supplies no derivation that synchronization creates that
parameter or an absolute clock.

The repo already has stronger concrete controls than a generic warning:

- [FORCED_SOURCE_AND_CLOCK, section 25](../../theory/nodal/FORCED_SOURCE_AND_CLOCK.md#25-relational-time-and-synchronization-are-separate-claims)
  shows a constant alignment statistic during form relaxation, loss of
  orientation in an alignment readout, and the need for a positive local
  clock rate plus retained chart/history or cycle count.
- [NODAL_PARAMETER_FOUNDATIONS, pulse scales](../../theory/NODAL_PARAMETER_FOUNDATIONS.md#derived-pulse-scales)
  distinguishes labeled, unordered-pair and mean observations of the same
  motion and keeps the whole-law time transformation.
- [DERIVED_FORM_PHASE, unequal-capacity mode selection](../../theory/nodal/DERIVED_FORM_PHASE.md#unequal-capacity-mode-selection-and-a-finite-phase-only-discriminator)
  already has a generic limiting relative phase while the contrast decays.
  It has no positive-capacity locking transition and no uniform finite
  approach time over sources arbitrarily near the exceptional fast mode.
- GLOBAL_PULSE_HYPOTHESIS section 13.7 gives actual finite tick-error and
  boundary conditions in its supplied structural clock. Its positive result
  should be retained alongside, not replaced by, the no-global-clock warning.

**Concrete optimization.** Require every use of “synchronization” to identify
the observation and conclusion: bounded relative reading, limiting rate
agreement, finite tick agreement, amplitude/activity retention, or a specified
interaction response. Record whether the result is exact, asymptotic or
finite-window. Reuse the existing controls; no new synchronization campaign
is warranted for the current two-contact source-admission gate.

## Finding 5: slow dynamics can reduce work only with the fast state and its errors retained

S1 section 7.5 obtains a useful separation of motion by an exact coordinate
change before approximating the slow segments and jumps. S2 sections
7.1.4-7.1.8 similarly distinguish the complete field, weak-force phase
approximation and resonant averaging. Neither procedure licenses replacing
an actual full-state source by a convenient slow manifold or target. A small
parameter alone does not specify uniform accuracy for a long observation
window or near a loss of the approximation's hypotheses.

**Already covered locally, with a particularly useful optimization.**

- [SINE_FORM_PHASE_REDUCTION](../../theory/nodal/SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence)
  keeps the exact mixed coordinates, compatible lifts, initial form/phase
  velocity and hidden source. Its second-order or memory description is a
  same-information representation, not deletion of a state coordinate.
- [SINE_TWO_PORT_TRANSIT](../../theory/nodal/SINE_TWO_PORT_TRANSIT.md#sine-energy-informed-slow-comparison)
  already improves a long-horizon exponential envelope using dissipated
  storage and an acute-cell nonexpansive reference. It explicitly retains
  the fast form transient, actual mean leaf, complete-state reconstruction,
  first-exit bootstrap and finite-horizon growth. This is more immediately
  useful for the selected source than importing a generic averaging slogan.
- [SINE_REDUCED_CLASS_PORTS](../../theory/nodal/SINE_REDUCED_CLASS_PORTS.md#sine-reduced-class-ports)
  and [SINE_CLASS_MEDIATED_MEMORY](../../theory/nodal/SINE_CLASS_MEDIATED_MEMORY.md#sine-class-memory-causal-law)
  already retain initialization, causal feedback and nonlinear discrepancy.
  Replacing them with a class-only or instantaneous port law would lose
  demonstrated state, not provide a bibliographic optimization.

**Action for the active gate.** First check whether the chosen actual source,
new degree normalization and unaltered phase origin can satisfy the existing
comparison's domain and error obligations. If so, a later independently
reviewed protocol may reuse that method with a new matched reference. If
not, record the missing domain or source control. Do not import the old
midpoint reference, its apparent symmetries or its passing transit verdict.
The bibliography does not authorize another integration, larger work budget,
new origin, or narrower uncertainty.

## Finding 6: bifurcation language is not evidence of a new pulse or identity

S1 section 8.2 explicitly distinguishes stable damped modes, eigenvalue
crossing, supercritical/subcritical behavior and the degenerate conservative
case. The linearization alone does not distinguish those nonlinear
outcomes. A nonreal stable pair is not a Hopf bifurcation; an imaginary pair
at a conservative boundary is not automatically an attracting limit cycle.
The book itself separates illustrative numerical behavior from proof.

The current TNFR question fixes the complete law and makes a supplied support
event. That event is not a smooth Hopf parameter continuation. This reading
therefore provides no reason to scan loss, capacity, geometry or frequency
in search of a favorable oscillation. Existing conservative exact branches
and transverse results retain their own derivations and amplitude domains.
If a later question genuinely consumes a bifurcation, it must first declare
the parameter family, regular domain, critical-state conditions, neutral
symmetries and the nonlinear conclusion sought. That is a deferred method
requirement, not an additional active task.

## Prioritized integration before research resumes

1. **Preserve the active source/event/identity gate.** In its admission record,
   explicitly connect the actual isolated formation image, simultaneous
   `4--4` and `5--5` contacts, unchanged origin, event work, internal-arc
   identity, initial absence and the conditional predictive witness. The
   books support this dependency discipline; they do not discharge any
   missing arrow or prove passage to the new target.
2. **Clarify phase and pulse vocabulary at existing owners.** Prefer a short
   glossary/reading bridge linking the conserved origin, derived contrast
   phase, conservative pulse/ticks and attracting oscillator phase. Keep
   existing positive results and exact limitations together. Avoid another
   duplicated theory chapter.
3. **Expose method prerequisites in one reusable admission table.** For a
   proposed approximation or tail, record source/domain, neutral modes,
   metric and reconstruction, initial hidden state, error/horizon and the
   stronger conclusion still missing. Existing proofs already provide many
   entries; cross-link them instead of rederiving their generic background.
4. **Defer new phase reduction, Hopf search and numerical optimization.**
   Activate one only when a concrete missing dependency consumes it and a
   new F1-F4 protocol has been independently reviewed and source-frozen.
   The deferred native AR15 source entry remains deferred.

The strongest synergy is between the existing complete-state capture chain
and disciplined reduction/attraction terminology. The largest avoidable
errors are treating a retained angle as a derived oscillator, a conservative
pulse as an attractor, a stable tangent as a usable source-entry bound, or
phase agreement as a clock. The books sharpen the presentation and future
admission tests; they do not supply a missing TNFR law or a physical map.

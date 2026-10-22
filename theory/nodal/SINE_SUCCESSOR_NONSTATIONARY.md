# Successor motion: sufficient state and finite principal confinement

The [principal-chamber theorem](SINE_SUCCESSOR_PRINCIPAL_REDUCTION.md#successor-principal-minimum-exclusion)
excludes stationary storage minima on the proposed maintained-parent UM
support. Its source-specific comparison path also reaches a principal cut
while preserving storage and both means. Neither result describes the
source's motion. This owner admits the complete state and analytical bounds
for nonstationary retention under the unchanged law. The
[fixed conditional certificate](#successor-nonstationary-result) retains all
53 branches through structural time `1/2` and proves a child-parent relative
phase turn within that window.

The [research program](../RESEARCH_PROGRAM.md#research-current-question)
owns gate status and subsequent scope. The discussion below introduces no
new pressure, event selector, primitive coordinate or physical identification.
This certificate was computed for a hypothetical UM family from the retained
THOL source. A separate [committed event and primitive inclusion review](SINE_MAINTAINED_PARENT_UM.md#maintained-um-committed-result)
now admits the actual UM endpoint into that family without repeating the motion.

## 1. Complete state, law and identity

Keep all 21 nodes and 53 edges of the re-admitted
[successor chamber](../../src/tnfr/research/maintained_successor_chamber.py).
Use its exact node order, positive symmetric conductances W, positive held
capacities nu, bare degrees d, original structural clock and zero forcing.
Signed form x and declared real phase lifts theta supply the 42 continuous
coordinates. Support, W and nu have zero derivative on this segment.

For the selected `beta=1` instance of the
[weighted complete law](SINE_PAIR_COUPLING.md#collective-coupling-weighted-law),
write

\[
 M=\operatorname{diag}\frac{\nu_i}{\pi d_i},\qquad
 S_i(\theta)=\sum_{j\in N(i)}\sin(\theta_j-\theta_i),\qquad
 \dot x=MS(\theta),\qquad\dot\theta=ML_Wx.
\]

Pressure is independently specified as `p_i=S_i/(pi*d_i)`. The stored
post-relief UM pressure is not substituted into this continuous row.
Conductances enter the form gradient in the phase row, whereas phase
pressure and degree normalization retain bare support. This is the declared
conservative completion, not the general operator runtime or native
resultant-pressure flow.

The storage and two conserved weighted means are

\[
 H=Q_W(x)+V(\theta),\qquad
 Q_W=\frac12x^TL_Wx,\qquad
 V=\sum_e(1-\cos\delta_e),\qquad
 \rho_i=\frac{d_i}{\nu_i}.
\]

Both means use rho; neither is replaced by arithmetic centering. Each
conditional source member carries its own fixed W, nu, H and means. These
structural quantities require their own measurement and clock bridge before
any laboratory interpretation.

Orient every edge `(i,j)` as `i<j` and let B have row `e_j^T-e_i^T`.
The retained branch vector k gives

\[
 \delta=B\theta-2\pi k,\qquad |\delta_e|<\pi.
\]

All 53 inequalities define the proposed principal identity. Its 33
fundamental cycle rows C obey `C*B=0`, so
`C*delta=-2*pi*C*k`. Retaining just the two older constituent winding
numbers would discard part of this identity. A principal cut is not a
singularity of the smooth sine vector field.

## 2. Phase velocity retains form information; phase alone does not

The existing [form/phase/memory representation](SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence)
provides the method of differentiating the complete phase row and retaining
its initial velocity. Its unit-support coefficients cannot be copied to
the present weighted attachment. With the actual held matrices here,

\[
 \boxed{\ddot\theta=A S(\theta),\qquad A=ML_WM,\qquad
 \dot\theta(0)=ML_Wx_0.}
\]

The constraint `rho^T*dot(theta)=0` is essential. A freely initialized
second-order equation would otherwise admit an extra mean-phase velocity
absent from the original complete law. On the rho-mean-zero subspace,
`ML_W` is invertible: its kernel consists of constants, and its image has
zero rho mean. Thus the original form mean and phase velocity reconstruct
the full form. This is a representation with the same information, rather
than a first-order reduction to phase alone.

For clarity, A above denotes the symmetric acceleration matrix of this
weighted model; it is not the differently defined A in the linked unit-law
owner. It is positive semidefinite with rank 20 and null vector rho.
This property does not establish attraction or a minimum of V.

Integrating the displayed equation gives the exact history identity

\[
 \theta(t)=\theta_0+tML_Wx_0+
 \int_0^t(t-s)A S(\theta(s))\,ds.
\]

The initial form remains an independent source through `ML_W*x_0`.
Eliminating it from a displayed state does not remove it from the dynamics.
This is the same hidden-initialization obligation identified by the
[general memory owner](../DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state).

There is a direct counterexample at the proposed source's own law and
phase state. Let `mu_x=(rho^T*x_0)/(rho^T*1)` and define

\[
 x_+=x_0,\qquad x_-=2\mu_x1-x_0.
\]

These admitted signed forms have the same form mean and the same Q. Keep
the same theta, W and nu: both full states then have the same phase, phase
pressure, storage, both means and all principal branch labels. Yet their
phase velocities are opposite. The
[source storage admission](SINE_SUCCESSOR_STATIONARY_GEOMETRY.md#5-storage-and-both-means-do-not-trap-this-source-in-the-acute-chamber)
has `Q_W(x_0)>0`, so connected positive conductances ensure that these
velocities are nonzero. At least one edge velocity also differs: a common
phase velocity with zero rho mean would have to vanish.

Consequently the instantaneous phases, their pressure and these conserved
values do not determine even the direction of every branch's motion.
This is a sufficient-state obstruction, not a second preparation to execute
or a change to the reserved source. It does not exclude a separately proved
restricted invariant family.

## 3. Exact edge acceleration and an energy velocity bound

Since `S=-B^T*sin(delta)`, the same complete law gives

\[
 \boxed{\ddot\delta=-J\sin\delta,\qquad
 J=BML_WMB^T,
 \qquad\dot\delta(0)=BML_Wx_0.}
\]

J is symmetric positive semidefinite of rank 20. The edge state retains
`C*delta=-2*pi*C*k` and `C*dot(delta)=0`; 53 independently initialized
angles and velocities would be a different model. Retaining these
constraints and the original two means gives another full-information
representation. It neither closes the stationary four-coordinate map nor
forgets the initial form. For interval calculations, each parameter member
keeps its own constant J throughout the window.

Let `b_e=e_j-e_i` for edge `(i,j)` and define

\[
 a_e=(Mb_e)^TL_W(Mb_e)=J_{ee}
 =M_i^2\sigma_i+M_j^2\sigma_j+2M_iM_jW_{ij},
 \qquad\sigma_i=\sum_jW_{ij}.
\]

The semidefinite Cauchy inequality in the actual form quadratic gives

\[
 |\dot\delta_e|^2
 =|(Mb_e)^TL_Wx|^2\leq2Q_W(x)a_e\leq2Ha_e.
\]

The last step uses `V>=0` and the exact conserved H. It does not assume
prior principal confinement: the complete sine law and the nonnegative
potential continue across cuts. Therefore the condition

\[
 \pi-|\delta_e(0)|-T\sqrt{2Ha_e}>0
\]

is sufficient to retain that original edge branch throughout `[0,T]`.
Whole-source interval upper bounds for H and `a_e`, and a lower bound
for the initial margin, make it a uniform conditional certificate.
They cannot mix one endpoint's means or energy with another endpoint's law.

A failed inequality is only an insufficient bound. It is not an observed
crossing or evidence of inconsistent dynamics. Similarly,
`|ddot(delta_e)|<=sum_f |J_ef|` is a valid global acceleration bound, but
can discard substantial cancellations between correlated edge currents.
The full field, cycle constraints and source correlations can therefore
matter when a scalar bound does not decide the complete identity.

The [static motion owner](../../src/tnfr/research/maintained_successor_motion.py)
rebuilds the hypothetical reset primitives before evaluating these bounds.
`predict_successor_motion` returns separate ideal and conditional reports,
including signed initial edge rates and accelerations, source storage,
energy coefficients, speed bounds and horizon margins. Its alternative
global acceleration enclosure uses sparse rows of A:

\[
 |\ddot\delta_{ij}|\leq\sum_k|A_{jk}-A_{ik}|d_k,
\]

because `|S_k|<=d_k`. This global magnitude bound differs from the signed
instantaneous acceleration. The owner does not evolve a source or certify
a cut from a failed margin. Its
[independent controls](../../tests/research/test_maintained_successor_motion.py)
check the complete rows, energy inequality, domains and source immutability.

### 3.1 Static admission of the unchanged proposed source

Fresh interval evaluation of the retained THOL source and hypothetical UM
reset, at `T=1/2`, certifies the energy inequality for exactly 41 original
edges in both the ideal and conditional reports. Every retained conditional
margin is strictly greater than `593/1000`. The 12 remaining edges are
exactly those incident to parent node 2. Their bounds do not certify this
horizon; the statement is not that these edges cross a cut.

For the oriented child-minus-parent edge `(2,20)`, the entire conditional
source interval satisfies the strict rational enclosures

\[
 -\frac1{6000}<\delta_{2,20}(0)<-\frac1{7000},\qquad
 -\frac{269}{50}<\dot\delta_{2,20}(0)<-\frac{537}{100},
\]

\[
 -\frac{77}{100}<\ddot\delta_{2,20}(0)<-\frac34,
 \qquad1475<J_{ee}<1476.
\]

These bounds use the actual unchanged preparation, held coefficients and
complete field; they are static calculations, not a motion response. The
large diagonal edge-acceleration coefficient also makes a straight-line
extrapolation of the initial negative velocity insufficient. The remaining
currents contribute to the same row, so this coefficient alone supplies
neither a frequency nor a restoring-sign theorem. The half-unit attempt
below tests a resolved direction change on the full coupled law instead
of merely certifying an arbitrarily short continuity window.

## 4. Which reductions can describe motion

The stationary reduction imposes equal true-twin phases, equal return-chain
gaps, aligned middle nodes and equality of the leaf and parent phases.
Their role in classifying minima does not make all these conditions
invariant. For example, equal consecutive chain gaps require both
`2*theta_16-theta_14-theta_18=0` and the corresponding rate equality
`2*dot(theta_16)-dot(theta_14)-dot(theta_18)=0` if that constraint is to
persist. Middle alignment and a leaf's stationary equality have their own
tangency conditions. The complete phase row consumes form contrasts that
the four-coordinate map does not retain.

True twins provide a more limited useful symmetry. For any transposition
preserving the actual bare support, W and nu, both complete rows commute
with the transposition. Uniqueness then makes its simultaneous equal-form,
equal-phase fixed-point subspace invariant. Nonuniform capacities across
different groups do not invalidate this argument; unequal capacities or
consumed weights within a proposed symmetry do.

The nine listed twin pairs in the
[principal owner](SINE_SUCCESSOR_PRINCIPAL_REDUCTION.md#2-fine-twins-leaf-and-zero-resultants)
give a possible exact 12-group synchronized description of the ideal source,
with parent 2, node 3 and child 20 still distinct. Admission needs the exact
primitive pair equalities and coefficient symmetries: the ideal UM map gives
paired phase updates and paired contact scores from equal inputs. Equality
of cached interval outputs alone would not prove that premise. This uses
the method of the [weighted pair quotient](SINE_PAIR_COUPLING.md#collective-coupling-weighted-quotient),
with new group multiplicities and actual conductances, not its unchanged
unit-interpair coefficients.

The existing conditional envelope allows independent errors in changed
phase outputs and new conductances. It therefore does not impose those
exact pair equalities on every member. An ideal quotient cannot silently
replace that full uncertainty family. A validated quotient plus transverse
error analysis would need its own proof; it is not assumed in the fixed
attempt below. The ontology likewise distinguishes a
[sufficient effective description](../EMERGENT_ONTOLOGY.md#11-nfr-formation-definition-birth-and-collective-dynamics)
from the existence of the organization being described.

## 5. Fixed prospective confinement and turning test

The analytical attempt retained the unchanged source and froze one
structural-time horizon and one computational budget before evaluating
its reserved response:

- Horizon `T=1/2`, divided into exactly 256 steps of `1/512`.
- Validated Taylor order 12 on the full 42-component form/lifted-phase state.
- A 900-second wall budget checked between steps; an already started final
  step may finish after the budget. Wall time is a resource limit, not the
  structural clock or a mathematical confinement horizon.
- The complete 21-node, 53-edge conditional source box, with held parameter
  uncertainty and all original branch integers; the ideal endpoint remains
  a separate object rather than a replacement for the box.
- A strict positive margin for every original principal edge on every
  complete step, combining the full tube and the independent conserved-energy
  bounds where available, rather than checking only stored endpoints.
- At least one positive child-minus-parent phase-velocity enclosure at the
  fixed endpoints `t_n=n/512`, `1<=n<=256`, following its strictly negative
  initial enclosure. The successful endpoint, if any, must certify that
  sign for the whole admitted source family.

The two requested outcomes are joint: retained principal identity through
the whole window and a direction change of the child-parent relative phase
inside it. The latter distinguishes the attempt from arbitrarily short
retention guaranteed by continuity. It would establish a finite turning
motion, not a period, attraction, formation, all-time maintenance or an
autonomous event. No laboratory clock is inferred.

The endpoint set is fixed in advance. Checking the stated existential sign
criterion on that set does not authorize searching new horizons, selecting
a different observable or retuning the source after seeing the response.
Whole-tube inclusion and remainder admission are required; small numerical
residuals or endpoint margins alone do not establish them. A rectangular
Taylor tube may extend beyond a cut when the independent energy inequality
still proves that the actual trajectory cannot reach it; raw tube and
combined retention margins retain distinct provenance. An enclosure
that proves neither requested result remains explicitly unresolved under
this attempt. A subsequent calculation would require a separate admission
rather than rewriting the evaluated protocol.

This section records the prospective protocol and its mathematical rationale;
the retained outcome is reported separately below. The execution plan remains
the only queue for any subsequent work.

The [fixed-attempt consumer](../../src/tnfr/research/maintained_successor_motion_certificate.py)
uses the shared held sine field and validated Picard/Taylor owner. It retains
all 42 source coordinates and every original branch, records each attempted
step before continuing, and distinguishes unavailable enclosures, unresolved
branch bounds and resource exhaustion from actual escape. It does not itself
authenticate protocol-freeze chronology or execute a structural operator.

<a id="successor-nonstationary-result"></a>
## 6. Retained finite confinement and turning result

The [protocol](../../docs/assets/successor_nonstationary/protocol.json),
[primitive source](../../docs/assets/successor_nonstationary/source.json) and
[complete Python source archive](../../docs/assets/successor_nonstationary/source.zip)
were [frozen](../../docs/assets/successor_nonstationary/freeze.json) before
the single [attempt](../../docs/assets/successor_nonstationary/attempt.json).
The [retained result](../../docs/assets/successor_nonstationary/result.json)
records all 256 accepted steps, reaching exactly `T=1/2` within the fixed
900-second wall budget. No step-size, source, order, horizon or criterion
was retuned.

Every one of the 53 original principal branches remains strict over every
whole step. The minimum certified combined retention margin obeys the
exact rational lower bound

\[
 \min_{e,\,0\leq t\leq1/2}
       \bigl(\pi-|\delta_e(t)|\bigr)>\frac{158}{100}.
\]

The initial child-parent relative velocity is strictly negative as admitted
in section 3.1. The first predetermined endpoint with a uniformly positive
velocity enclosure is `t=21/512`; its
[step record](../../docs/assets/successor_nonstationary/steps/020.zip)
gives the coarser strict containment

\[
 \frac9{50}<\dot\delta_{2,20}(21/512)<\frac{19}{100}.
\]

Continuity therefore gives a relative-phase direction change in
`(0,21/512)` for every member of the conditional family. This endpoint is
the first certified positive member of the frozen observation set, not an
exact turning time shared by all family members. The full-window branch
certificate and this sign change satisfy the joint prospective target.

The statement covers the complete independent `2^-40` conditional source
envelope, including the ideal mathematical reset, under each member's own
held W and nu and the declared unforced complete law. It is a validated
mathematical continuation. That certificate computation executed no UM event,
represented runtime trajectory or SHA closure. Applying it to an actual
endpoint requires separate arithmetic, support, grammar and primitive-source
inclusion admission.

The later [single runtime attempt](SINE_MAINTAINED_PARENT_UM.md#maintained-um-transaction-result)
retains an included raw reset but rolls back after a diagnosed pressure-guard
false rejection. Its detached refreshed copy is not a committed successor.
That failure and its guard correction alone establish no runtime transfer.
The separately frozen [corrected UM transaction](SINE_MAINTAINED_PARENT_UM.md#maintained-um-committed-result)
now supplies a committed endpoint. Its independent primitive review admits
all continuous coordinates, held parameters and branch labels into this
same family. This establishes theorem applicability to that endpoint under
the declared complete law and clock, without executing a new trajectory or
changing any frozen motion evidence. A separate
[SHA event and changed-law admission](SINE_MAINTAINED_PARENT_CLOSURE.md#maintained-parent-sha-result)
now completes the selected word; this certificate does not extend through
that capacity reset or certify the new continuation.
The separate [actual post-SHA result](SINE_MAINTAINED_PARENT_CLOSURE.md#maintained-sha-retention-result)
retains all branches through `1/2` after rebuilding its changed field and
source. Both adapters now share the fixed
principal-retention loop, while this study preserves its original conditional
family, reversal target, record schema and immutable source archive.

The result separates two previously unresolved possibilities: absence of a
stationary storage minimum does not prevent finite principal confinement,
and an invariant shell's access to a cut does not force the actual flow to
take that path during this window. It establishes neither periodic motion
nor all-time maintenance, a newly formed stationary minimum, autonomous
organization or a physical particle interpretation.

The [retained-evidence controls](../../tests/research/test_maintained_successor_motion_evidence.py)
check frozen byte associations and recompute consumed field, branch and
sign readouts from primitive step data without rerunning the producer.
They check Picard inclusion and endpoint arithmetic; the recorded derivative
jets and remainder bounds retain their frozen-producer provenance and are
not regenerated by this audit. Those checks remain distinct from the shared
kernel's mathematical enclosure obligations. Hashes associate evidence bytes
rather than authenticate their acquisition or chronology.

<a id="successor-single-parent-il-admission"></a>
## 7. Effect of a single-parent IL phase reset

The [complete IL action](SINE_COLLECTIVE_PRESSURE_HANDOFF.md#collective-il-full-action)
and its [single-target work identity](SINE_COLLECTIVE_PRESSURE_HANDOFF.md#collective-il-adjacent-work)
supply a reusable mechanism for one public or staged IL action at parent 2.
That mechanism preserves x, W, support and nu while moving the parent's
phase toward its bare-neighbor resultant. Consequently M, rho and J remain
unchanged, as does the complete instantaneous phase velocity `ML_W*x`,
including the child-parent relative velocity. Its subsequent acceleration
can change because the sine-current vector changes.

Conditional on a nonzero resultant `R*exp(i*mu)`, an admitted shortest
displacement `a=wrap(mu-theta_2)` and `0<=lambda<=1`, the continuous event
lift changes only `theta_2` by `lambda*a`. The existing single-target
identity gives

\[
 \Delta H=R\,[\cos a-\cos((1-\lambda)a)]\leq0.
\]

The inequality follows on `|a|<=pi`; a regular source admits its chosen
shortest-arc branch before using the map. It supplies no numerical result
for this successor until the actual resultant, branches, default coefficient,
live preconditions and grammar context are re-admitted. The form mean is
unchanged, while the lifted-phase mean has the event increment
`rho_2*lambda*a/sum(rho)` in this declared lift. A changed mean is not a
failure of the between-event conservation law.

For this one-target map, the 41 non-parent initial edge gaps and their
energy coefficients are unchanged, and H cannot increase. Their existing
half-unit energy certificates therefore transfer whenever the same original
branch identity is retained. The 12 parent edges need fresh initial margins.
This argument does not transfer the full frozen motion response to a new
phase source outside its admitted box. It also cannot introduce a stationary
storage minimum inside the original principal chamber: the
[global minimum exclusion](SINE_SUCCESSOR_PRINCIPAL_REDUCTION.md#successor-principal-minimum-exclusion)
depends on the unchanged support, not on which source reached that chamber.

The pressure-only IL glyph would leave the consumed continuous state
unchanged; immediate constitutive refresh would remove its stored-pressure
contraction. The public/staged IL phase reset is therefore essential to
this proposed comparison. Multiple simultaneous adjacent targets require
the existing shared-edge work correction and are not covered by the
single-target inequality above. These are analytical admission conditions,
not an evaluated IL event, completed word or new motion certificate.

These identities are not needed for the finite confinement certificate in
section 6, which was obtained without an IL action. They supply no new
reserved comparison or automatic requirement to add that operator.

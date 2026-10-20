# Mutation action: phase work, contextual descent and geometric identity

ZHIR has a compatible conditional event map on the selected complete sine
law. Its observed-growth gate does not determine the sign of its storage
work, preserve every acute phase sector or select an occurrence time.
The actual phase jump changes fresh pressure and form rates while retaining
the phase-rate row. Associated histories, target marks, pressure branch and
grammar remain necessary parts of collective execution state.

The [admission owner](SINE_COLLECTIVE_MUTATION_ADMISSION.md) establishes
general temporal compatibility separately from one restricted NAV successor.
This owner completes the event calculation on general admissible states.
No retained research source is mutated and no continuous response campaign
is evaluated. Runtime histories below are explicitly manufactured reader
inputs, not observed responses of this law.

## 1. Complete law, event domain and actual action

Use a finite simple loop-free connected undirected graph with at least two nodes,
positive conductances, signed real form x, circular phase theta and held
positive capacities nu. On each continuous segment the selected law is

\[
 q=L_Wx,\quad S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i),\quad
 M=\operatorname{diag}\frac{\nu_i}{\pi d_i},\quad
 p_i=\frac{S_i}{\pi d_i},\quad
 \dot x=MS,\qquad\dot\theta=Mq.
 \tag{1}
\]

Here d counts distinct bare neighbors. Conductances enter q, not S or d.
Pressure is independently specified by (1); it is not reconstructed from
an observed response. The structural clock, unforced flow and fixed support
are premises. The [weighted-law owner](SINE_COLLECTIVE_RECEPTION.md#1-complete-model-and-event-boundary)
provides its storage and support scope.

At a supplied event, retain target set A and the pre-event runtime pressure
r, which can differ from the freshly evaluated p. The actual
[Mutation proposal](../../src/tnfr/operators/_mutation_stage_kernel.py)
chooses one of two phase branches:

\[
 x_i^+=x_i,\qquad\nu_i^+=\nu_i,\qquad r_i^+=r_i,\qquad
 \theta_i^+=\theta_i+\delta_i\pmod {2\pi},
 \tag{2}
\]

with delta_i=0 off A. On A the default branch uses
`delta_i=f*copysign(1,r_i)*pi/4`; the canonical ideal f=1/pi gives magnitude
1/4 radian. An explicit `ZHIR_theta_shift` instead supplies the fixed shift.
The [factor owner](../../src/tnfr/operators/factor_contracts.py) admits the
active finite factor and excludes a full-turn no-op. Its runtime arithmetic
and modulo operations remain distinct from the ideal equations.

The default branch retains the sign bit at zero: positive and negative zero
select opposite rotations. The fixed branch does not consume the dynamic
direction or its inactive factor, but still requires the temporal gate.
ZHIR consumes no random draw. It preserves support, conductances and
`EPI_kind`; the latter is a semantic identity field, not a geometric proof.

Both branches require active capacity and an accepted live signed two-sample
EPI secant strictly above xi, plus independent grammar context. Prior IL
and a recent destabilizer are required; an instantaneous nu*r prediction
cannot replace the secant. Histories and thresholds retain their time basis.
Timestamped evidence ends at the current form, but the pure reader alone
does not authenticate its last timestamp against an execution clock.

The atomic stage proposes every target from one immutable snapshot and
returns target-associated decision observations, including admitted evidence,
the actual phase map and recent context. It records grammar/lifecycle changes
after commit. Quarter-circle regime labels and an acceleration-threshold
flag are diagnostics. Two EPI samples suffice for the growth gate but not
for a three-sample acceleration observation: its legacy numeric zero must
not be interpreted as measured zero acceleration. Detection creates no child
or support edge and proves no dynamical bifurcation.

## 2. Complete work and both refreshed evolution rows

For any declared real lifts, use

\[
 H(x,\theta)=\frac12x^TL_Wx+
   \sum_{\{i,j\}\in E}[1-\cos(\theta_j-\theta_i)].
 \tag{3}
\]

Equation (1) conserves H between events because its two contributions are
q^T MS and -S^T Mq. This auxiliary storage is not identified with a physical
energy. Let g_ij=theta_j-theta_i and h_ij=delta_j-delta_i. Substitution in
(2), with each undirected edge counted once, gives the exact ideal work

\[
 \Delta H=\sum_{\{i,j\}\in E}
      [\cos g_{ij}-\cos(g_{ij}+h_{ij})].
 \tag{4}
\]

There is no form-storage jump, conductance factor in the phase term or
capacity-storage term. Nevertheless the unchanged form contribution still
uses the actual weighted Laplacian. Different source or target conductances
cannot be replaced by the old unit graph when evaluating complete H.

For adjacent targets, (4) includes their common edge once. With equal shifts,
that edge has zero work. Adding two isolated-target costs generally counts
a change that never occurs in the joint event. At g_ij=0 with both shifts
sigma, joint work minus the sum of isolated work is
`-2*(1-cos(sigma))`. This is independent of the edge's conductance.

After constitutive refresh the exact changed rows are

\[
 \Delta p_i=\frac1{\pi d_i}\sum_{j\sim i}
    [\sin(g_{ij}+h_{ij})-\sin g_{ij}],\qquad
 \Delta\dot x_i=\nu_i\Delta p_i,\qquad
 \Delta\dot\theta_i=0.
 \tag{5}
\]

The last identity follows from unchanged x, capacity and support, hence
unchanged Mq. Even untargeted neighbors can change pressure and form rate.
Stored pressure is unchanged by the event itself; a refresh is a separate
constitutive handoff. Continuing with held r instead of the new p is a
different complete evolution model.

Set rho_i=d_i/nu_i and R=sum rho_i. The two continuous mean leaves use these
weights. With the event lift explicitly chosen as theta^+=theta+delta,

\[
 \Delta\bar x_\rho=0,\qquad
 \Delta\bar\theta_\rho=\frac1R\sum_i\rho_i\delta_i.
 \tag{6}
\]

A principal circular coordinate alone does not define this lifted mean.
Adding a full turn to a chosen event lift changes (6), although it does not
change (4) or (5). The retained lift therefore belongs to any mean claim.
A common shift of every node is an ideal global phase symmetry: storage,
fresh pressure and both rows remain unchanged while the lifted phase mean
changes by that shift. Runtime modulo rounding requires endpoint assessment.

Under an affine clock change s=c*t+b with c>0, scaling every capacity by
1/c scales both rows of (1) by 1/c while leaving p unchanged. Associated
timestamps and xi must transform with their declared units, as in the
admission owner. The angular shift, H work and normalized event means then
stay the same. Keeping the numeric growth threshold while changing its
time unit is a different admission policy.

## 3. Growth eligibility imposes no passivity theorem

Consider a target with phase zero, every neighbor at epsilon in (0,1/8),
and the default positive shift sigma=1/4. Fresh pressure is positive and

\[
 \Delta H=d_i[\cos\epsilon-\cos(\epsilon-1/4)]>0.
 \tag{7}
\]

The event overshoots the neighbors: 1/4-epsilon>epsilon, and cosine strictly
decreases with the magnitude of these small gaps. Take epsilon=1/10,
nu_i=4 and xi=1/10. Then

\[
 \dot x_i=\frac4\pi\sin(1/10)>1/10.
 \tag{8}
\]

For example, `sin(1/10)>1/10-(1/10)^3/6` and `pi<22/7` prove the strict
inequality. Any finite form is allowed. Smooth local backward flow exists,
and continuity keeps the rate above xi on some short preceding interval.
Integrating proves that genuine same-law positive secants can coexist with
positive event work. No numerical trajectory is needed for this existence
argument; no particular manufactured runtime record is authenticated by it.

Conversely, with every neighbor at 1/4 the same positive shift exactly aligns
the target and gives `Delta H=d_i*(cos(1/4)-1)<0`. Capacity 4 also gives a
strict rate margin above xi. Thus both signs occur inside regular acute
zero-winding preparations with true short-secant compatibility. The observed
growth test neither chooses a storage sign nor supplies an event reservoir.

At an exactly synchronized phase state, a default signed-zero direction
instead yields positive work `d_i*(1-cos(1/4))` at a single target. This
additional implementation witness concerns the branch convention; it does
not claim an accepted supplied history was generated from a stationary law.

## 4. Sufficient geometric preservation and an allowed obstruction

Fix principal edge representatives g_ij in an acute source sector. If

\[
 |g_{ij}|+|\delta_j-\delta_i|<\pi/2\quad\hbox{for every edge},
 \tag{9}
\]

then the straight event interpolation theta+s*delta, 0<=s<=1, keeps every
edge in the same principal chart and acute sector. The reference integers
do not change, so each oriented cycle winding is preserved. This is a
sufficient event condition; it establishes neither later maintenance nor
confinement to an older joint form/phase ball.

One positive-width general domain has all phases in a common lift interval
of width less than 1/4. Under arbitrary targeted default shifts of magnitude
1/4, the new interval has width less than 3/4<pi/2. This preserves the acute
zero-winding class. During the actual phase-aligning IL stages, a defined
neighbor resultant within an interval of width less than pi lies in that
same interval; interpolation by lambda in [0,1] remains there. Consequently
this phase-domain argument extends through admitted surrounding IL/OZ/SHA
stages with the usual IL alignment factor: OZ and SHA do not change phase.
All other state, history, capacity, grammar and U6 hypotheses still require
their own admission; this phase argument does not replace them.

No unconditional acute-preservation claim holds even for default ZHIR and
fresh pre-event pressure. A star with center phase 0, two leaves at 7/5 and
one at -7/5 has all gaps acute. The center has

\[
 p_0=\frac{\sin(7/5)}{3\pi}>0,
 \qquad\delta_0=1/4.
 \tag{10}
\]

Its trailing gap becomes `-7/5-1/4=-33/20`, whose magnitude exceeds pi/2.
At center capacity 1, the rate is also greater than xi=1/10, so the same
local-flow argument supplies short genuine compatible secants. Strict source,
rate and exit margins persist on a neighborhood. Valid runtime grammar and
unchanged semantic kind therefore do not guarantee acute geometric identity.
The star has no cycles: this example proves sector exit, not winding change.
For the rate margin, the alternating lower bound through degree seven gives
`sin(7/5)>33/35`, while `3*pi/10<33/35` follows from `pi<22/7`.

## 5. Collective descent and complete-word context

For a permutation-preserving support and capacity profile, the fine phase map
is equivariant when targets, pressure directions/fixed shifts, configurations
and consumed histories transform with their members. A retained contextual
collective state containing that association therefore determines the event.
On a twin-invariant stratum, an orbit-closed target set with equal shifts on
each pair preserves the stratum. Targeting one member can leave it even while
remaining acute. It then requires the larger fine state or a retained mark;
the old reduced dynamics cannot be asserted for the new state.

The marked contact's eleven-cell reduction has two singleton ports and nine
double cells. Acting on either singleton or on an entire double cell with
equal shifts preserves its physical symmetry, conditional on the same
associated context. Equal physical coordinates alone need not descend:
one history can pass the live secant gate and another fail; a recent
destabilizer can be present or missing. Even numerically equal +0/-0 stored
pressure can choose opposite runtime shifts. The continuous mean or an
unordered inventory of histories loses these distinctions.

The complete control word is

`IL -> OZ -> IL -> ZHIR -> IL -> SHA`.

Starting with initialized form and zero debt, its successive U2 debts are
`0,1,0,1,0,0`. The pre-ZHIR IL discharges OZ's debt while leaving OZ recent
enough for U4b; the next IL handles ZHIR, and SHA closes the word. Every
stage still needs live preconditions and its own actual observation. The
shorter `IL,OZ,ZHIR,IL,SHA` is a valid grammar word but retains one unit of
debt, so it does not satisfy the selected zero-terminal-debt condition.
No initial stabilizer can prepay future destabilizations.

The synthetic complete-word control supplies timestamped EPI reader records
before execution. All its chosen stages preserve form, so they do not create
an EPI endpoint mismatch. They do not manufacture a new continuous response
or turn operator steps into elapsed structural time. Local action controls
use ZHIR's genuine validated-word step with an explicitly supplied matching
live prefix context; they do not claim that such a prefix ran in that probe.

## 6. Execution evidence and remaining scope

The [collective Mutation action controls](../../tests/operators/test_collective_mutation_action.py)
reuse the actual public atomic stage and detached sine field owner. They
check independent complete work, adjacent-target corrections, both work
signs with strict nodal rate margins, raw versus refreshed pressure, both
rows and lifted means, signed-zero/fixed branches, associated decision
records, complete-word debt and finite U6, symmetry loss and the acute-exit
counterexample. One invalid target rejects the whole phase stage before
graph-owned commit. Actual represented endpoints are assessed separately
from ideal arithmetic, with no second executor or new event policy.

These results complete ZHIR's bounded action compatibility under the stated
law and execution premises. They do not derive occurrence, a preferred phase
shift, universal collective closure, permanent identity or a physical particle.
Other registered operators keep their own event and history contracts; the
[existing mechanism map](../STRUCTURAL_OPERATORS.md#operator-mechanism-and-activation-audit)
and [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
own cross-operator scope and subsequent work.

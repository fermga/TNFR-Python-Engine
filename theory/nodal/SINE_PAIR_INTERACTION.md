# Pair actions, regional exchange and cancellation

Local and whole-pair actions, autonomous regional form exchange, zero-resultant restoration and the retained interaction interface.

Part of [Coarse-graining, coherence geometry and bridge results](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="sine-pair-emission-descent"></a>
## 22. Local and whole-pair form actions on an exact unordered NFR

### 22.1. Keep the continuous quotient and the supplied event separate

Use the normalized-sine complete law and the nonantipodal pair chart
of [Section 8](SINE_PAIR_STATE.md#sine-replica-unordered-state) and
[Section 19](SINE_PAIR_STATE.md#sine-mixed-pair-state). The selected pair has an
independently valid
fixed-support swap and equal positive held capacities. Other pairs may
remain ordered where their swaps are not symmetries. Write

\[
x_\pm=X\pm u,\quad \theta_\pm=\Theta\pm\delta,
\quad |\delta|<\pi/2,\qquad
(R,U,Q)=(\cos\delta,u^2,u\sin\delta).
\]

The retained state identifies precisely the two lifts related by
`(u,delta) -> (-u,-delta)`, with a single lift at the synchronized
tip `u=delta=0`. Equal phase alone and equal form alone are not tips.
Unequal held capacities or asymmetric attachments are not silently
removed: their sufficient states remain governed by Sections 13 and 19.

Consider the **primary form action** of registered AL on existing
members. Its shared implementation is
[`emission_epi_proposal`](../../src/tnfr/operators/al_sha_stage_proposals.py):
the supplied boost is added, the configured EPI boundary projection
is applied, and a proposal that decreases form is rejected. Let `T(x)`
denote this actual deterministic admitted point map, including projection,
and put `s_+=T(X+u)-(X+u)`, `s_-=T(X-u)-(X-u)`. Both increments are
nonnegative on the comparison domain. All proposals must be finite and
admitted before using the following identities. A soft-boundary rejection
is not an alternative valid endpoint; a saturated no-op is not an
unclipped boost.

This is a supplied hybrid reset attached to the stated sine observation,
not a term already present in its continuous law. The graph, capacity,
phase and clock stay fixed. The general public AL runtime additionally
consumes admission, lifecycle and history state which this quotient does
not retain. No closure of that larger runtime is asserted.

### 22.2. Exact obstruction and the two exceptional cases

Applying the same point map to the first member or to the second gives

\[
\begin{array}{c|ccc}
 &X'&U'&Q'\\\hline
\text{first}&X+s_+/2&U+s_+u+s_+^2/4&Q+(s_+/2)\sin\delta\\
\text{second}&X+s_-/2&U-s_-u+s_-^2/4&Q-(s_-/2)\sin\delta.
\end{array}
\]

Both leave `Theta,R` unchanged and satisfy
`Q'^2=U'(1-R^2)`. Exchanging the input lift while retaining the
same member label exchanges these two candidate outputs. Consequently
the fixed-member action has a well-defined unordered output at this
state **if and only if**

\[
\boxed{(s_+=s_-=0)\quad\text{or}\quad(u=\delta=0).}
\]

Indeed equal means first require `s_+=s_-=s`. If `s=0`, both
projected actions are identities. If `s>0`, equality of `U'` forces
`u=0`, and equality of `Q'` forces `sin(delta)=0`, hence `delta=0`
on the chart. Conversely either displayed condition suffices. This is
an exact statement on the domain where both candidates are admitted;
event admission itself must also agree on equivalent lifts. It does
not depend on a small boost approximation or on an unproved stability
effect of AL.

For an unprojected common boost `b>0`, the two differences are
`U'_first-U'_second=2*b*u` and
`Q'_first-Q'_second=b*sin(delta)`. Thus the equal-phase stratum
`R=1,U>0` still distinguishes the targets through `U'`. The
equal-form stratum `U=0,R<1` distinguishes them through `Q'` even
though their means and new `U'` agree. Losing either stratum would
incorrectly certify a collective action.

At the tip an unprojected singleton action has the unique projected
output `X'=X+b/2`, `R'=1`, `U'=b^2/4`, `Q'=0`. Nevertheless a
deterministic fine-member selector cannot be swap-equivariant there:
the source is fixed by the swap, whereas neither singleton target is.
This is the existing [selector-symmetry obstruction](../../src/tnfr/physics/selector_symmetry.py).
A unique projected outcome therefore does **not** require a unique
labeled fine realization. Nor does its uniqueness choose when the
action occurs. At a geometric tip, separate member histories can still
distinguish the full public AL outcomes.

### 22.3. A sufficient marked port and a symmetric control

For an action on a supplied constituent, retain its offsets

\[
u_p=x_p-X,\quad v_p=\sin(\theta_p-\Theta),\qquad
u_p^2=U,\quad v_p^2=1-R^2,\quad u_pv_p=Q.
\]

These constrained data reconstruct that member and its partner:
`delta_p=atan2(v_p,R)`, then
`(x_p,theta_p)=(X+u_p,Theta+delta_p)` and
`(x_other,theta_other)=(X-u_p,Theta-delta_p)`. The chart has `R>0`,
so this also covers equal phase and equal form without division by
`u` or `sin(delta)`. For `R<1`, the sign of `v_p` and the unmarked
state suffice; for `R=1,U>0`, the sign of `u_p` suffices. The missing
choice is generically one binary orientation, not an extra independent
continuous coordinate. At the tip both offsets vanish; persistence of
a separate label, lineage or external port remains additional data.

With `s_p=T(X+u_p)-(X+u_p)`, the marked update is

\[
X'=X+s_p/2,\quad u_p'=u_p+s_p/2,\quad v_p'=v_p,
\quad U'=U+s_pu_p+s_p^2/4,\quad Q'=Q+s_pv_p/2.
\]

The port transforms with its member when the fine graph is relabeled.
A fixed untransformed label is not such a mark. A rule such as selecting
the larger form would itself be a supplied policy, require tie handling,
and would not derive occurrence from the continuous law.

In contrast, apply the same `T` to **both** members from one source
snapshot. If `y_+=T(X+u)` and `y_-=T(X-u)`, then

\[
X'=(y_++y_-)/2,\qquad
U'=(y_+-y_-)^2/4,\qquad
Q'=(y_+-y_-)\sin\delta/2.
\]

The swap exchanges `y_+,y_-` and changes the sign of `sin(delta)`,
so these outputs are invariant. This whole-pair form map descends on
its invariant admission domain without retaining a member port.
For an unprojected boost, `X'=X+b` while `R,U,Q` are unchanged.
With clipping, `U,Q` can change: their preservation is not part of
the symmetry result. The whole-pair action is a different supplied
event, not a way to reinterpret a singleton action after discarding
its target. Neither the quotient nor AL fixes a per-pair rather than
per-member boost convention.

### 22.4. Storage and occurrence remain separate obligations

For the same fixed fine support, let `q=Lx` and let `h` contain
the actual form increments of the admitted event. The sine storage
`H=x^T Lx/2+beta*V(theta)` obeys the exact reset identity

\[
\boxed{H^+-H^-=q^Th+\tfrac12h^TLh.}
\]

The phase part is unchanged. At one member `a` this is
`s*q_a+d_a*s^2/2`, with its actual fine degree. It may be positive
or negative; increasing a signed local form is not an energy or
pattern-maintenance theorem. Equal-form interchangeable members have
equal `q_a` and degree. Their singleton actions therefore have the
same storage jump even when their distinct phases make `Q'` differ.
A common event balance cannot recover the discarded target.

These resets can change the internal state or leave a protected family;
the applicable barrier needs fresh admission. On the unprojected
whole-pair action, retaining `R,U,Q` preserves those internal
coordinates, but changing `X` can still alter relations to neighboring
pairs. A pressure refresh recomputes the selected law's pressure from
the new state; a retained old pressure is not the same continuation.
Continuous loss does not pay for a reset without a declared reservoir,
and no target, amplitude, time, grammar rule or autonomous event trigger
has been derived by the descent test.

The read-only `assess_sine_pair_emission` in the
[scale owner](../../src/tnfr/physics/relational_sine_scale.py) reuses the
actual AL proposal and the current mixed-state source admission with
common positive held capacity. It reports both
singleton alternatives and the symmetric whole-pair control without
mutating the source or executing public AL lifecycle effects. Exact
represented form arithmetic and member matching determine the descent
verdict; trigonometric enclosures do not substitute for that equality.
The [independent algebra controls](../../tests/physics/test_relational_pair_emission.py)
check the quotient identities and storage distinction. A supplied
collective action is now specified where the criterion holds; a
collective action selected by the dynamics remains a different question.

<a id="sine-autonomous-regional-transfer"></a>
## 23. Autonomous regional transfer and supplied form injection

### 23.1. The regional balance of the complete sine law

Keep finite connected simple unit support, strictly positive held capacities,
and the original normalized-sine complete law, with no input, event,
clipping or added phase velocity. For this balance the form-loss coefficient
may be any `e>=0`; the periodic control below separately requires `e=0`.
Use `a=w/pi`, `b=w/(beta*pi)`, `q=Lx`, `d_i>0` and
`S_i=sum_{j~i}sin(theta_j-theta_i)`. Both rows are retained:

\[
\dot x_i=\frac{\nu_i}{d_i}[-e q_i+aS_i],\qquad
\dot\theta_i=b\frac{\nu_i}{d_i}q_i.
\]

For a supplied region `A`, define its weighted form sum, its fixed weight,
and its weighted form mean by

\[
M_A=\sum_{i\in A}\rho_i x_i,\qquad
\rho_i=d_i/\nu_i>0,\qquad
H_A=\sum_{i\in A}\rho_i,\qquad \bar x_A=M_A/H_A.
\]

This is a structural coordinate, not an identification with physical mass
or stored energy. Sum the fine form row over `A`. Every internal edge
cancels with its reverse orientation in both the form difference and the
odd sine term. Only the boundary remains:

\[
\boxed{\dot M_A=\sum_{\substack{i\in A,\ j\notin A\\j\sim i}}
\left[e(x_j-x_i)+a\sin(\theta_j-\theta_i)\right],
\qquad \dot{\bar x}_A=\dot M_A/H_A.}
\]

The orientation is into `A` from its complement. The same cut contributes
the negative rate to the complement, and the full-support sum `M` is
constant. This recovers the existing weighted-mean invariant and identifies
its exact regional transfer, including the dissipative form-difference
channel. Dissipation of storage does not destroy this particular invariant.
On supplied continuous phase lifts there is also the exact companion row

\[
\frac{d}{dt}\sum_{i\in A}\rho_i\theta_i
=b\sum_{\substack{i\in A,\ j\notin A\\j\sim i}}(x_i-x_j).
\]

The lifted sum is not a globally defined average of circular phases.
Neither boundary identity by itself closes the regional dynamics: the
actual boundary member states, capacities and support remain consumed.
They do not transfer to native Arg dynamics or a state-dependent mobility
merely because those laws share another storage balance.

#### Relative inventory and a physical-property boundary

Let `V` be the same fixed whole support for every region and put `Q=M_V`.
The relative regional excess

\[
q_A=M_A-\frac{H_A}{H_V}Q=H_A(\bar x_A-\bar x_V)
\]

is unchanged by `x -> x+c*1`, is additive over disjoint regions and sums
to zero on a fixed partition of `V`. Because Q is conserved under the
present law, `q_A_dot=M_A_dot`: the same boundary current transports it.
These statements follow directly by summing the fixed positive weights.
The zero total is imposed by centering, not a derived physical neutrality.
Changing V changes the reference; changing weights or membership needs
additional continuous/event accounting. The uncentered Q instead changes
by `c*H_V`. See the [common-origin owner](SINE_CONSTITUTIVE_INFORMATION.md#form-balance-common-origin).

This yields a continuous, origin-independent relative inventory, not electric
charge or a sufficient pattern state. Equal excess can coexist with different
phase, internal motion and subsequent interaction. Physical identification
would require an independently justified measurement and interaction law;
conservation, additivity and sign alone are insufficient.

<a id="sine-two-contact-orientation-response"></a>
#### A finite orientation-sensitive response with retained backreaction

Use the same sine law with `e=0,w=beta=nu_i=1`, the cycle `0,...,4`,
and two leaves A and B attached respectively to vertices 1 and 4. There
are no other edges, inputs or events. In the declared clock `tau=t/pi`,
the complete rows are

\[
x_i'=d_i^{-1}\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
\theta_i'=x_i-d_i^{-1}\sum_{j\sim i}x_j.
\]

Prepare all forms zero, both leaf phases zero and cycle phases
`theta_k=+2*pi*k/5` or `-2*pi*k/5` modulo full turns. Define the ordered
probe response `Y=x_A-x_B`. Both preparations have the same full storage
because cosine is even. The independently specified form row gives

\[
Y(0)=0,\qquad Y'(0)=\mathord\pm2\sin(2\pi/5).
\]

This is also a finite response of the evolving joint system. For every node,
`abs(x_i')<=1`, hence `abs(theta_i'')<=2`. Since all initial phase rates
vanish, each edge-gap rate has magnitude at most `4*tau`. The sine Lipschitz
bound therefore gives

\[
|Y(\tau)-Y'(0)\tau|\le\frac43\tau^3.
\]

At the specified readout `tau=1/4` the remainder is at most `1/48`, while
the linear term has magnitude `sin(2*pi/5)/2>1/4`. The two response signs
are separated without fitting or freezing the source or probes. Every
cycle gap changes by at most `2*tau^2<=1/8<pi/10`, its initial acute
margin. Both leaf gaps start at the same principal magnitude and satisfy
the same bound. All edges remain acute and the cycle retains its winding
throughout this readout. This is a finite statement, not long-term pattern
stability of the whole graph.

Reflection `k -> -k mod 5` exchanges A and B. It relates the two full
trajectories and reverses the ordered readout, whereas their probe sum is
the same. If instead an arbitrary identical environment contacts the cycle
only at vertex 0, that reflection fixes the environment pointwise. Under
reflection-invariant capacities and consistently reflected source state,
equivariance and uniqueness give identical environment histories. This is
the smooth-sine specialization of the existing
[single-port argument](RELATIONAL_MEDIATOR_DYNAMICS.md#mediator-orientation-scope),
not a transfer of its separate native recovery or contact-event theorems.

The result is an orientation relative to labeled contacts, not intrinsic
electric charge, spin or a selected support-creation event. Winding depends
on the declared cycle orientation. It supplies an explicit example of a
joint pattern property becoming visible through an interaction, together
with an interface that cannot reveal it. Shared
[regional-current controls](../../tests/physics/test_relational_regional_transfer.py)
check the engine readout at admitted represented phases; the ideal-angle
finite bound above is analytical, not an executed trajectory certificate.

### 23.2. A positive control inherited from the existing pair pulse

Choose the supplied `K2,2` support, with regions `A={0,1}` and `B={2,3}`,
all four cross-edges and no internal edges. Hold common capacity `nu>0`
and `e=0`, with `w,beta>0` and the same structural clock. Set

\[
x_{0,1}=X_A,\quad x_{2,3}=X_B,\qquad
\theta_{0,1}=\Theta_A,\quad\theta_{2,3}=\Theta_B.
\]

The members of each pair have equal complete rows. Smooth uniqueness
therefore makes this synchronized submanifold invariant, exactly as in
[Section 7](SINE_PAIR_STATE.md#sine-replica-inheritance). It is a supplied preparation,
not a proof that generic pairs synchronize. Its internal coordinates
remain `R=1,U=Q=0`; means close here because these internal states are
fixed, not because arbitrary internal states are irrelevant.

Let `D=X_A-X_B` and `Delta=Theta_B-Theta_A`. Each fine degree is two,
so the two incoming sine terms cancel that normalization. The induced
complete regional rows are

\[
\begin{aligned}
\dot X_A&=a\nu\sin\Delta,&\dot X_B&=-a\nu\sin\Delta,\\
\dot\Theta_A&=b\nu D,&\dot\Theta_B&=-b\nu D,\\
\dot D&=2a\nu\sin\Delta,&\dot\Delta&=-2b\nu D.
\end{aligned}
\]

In particular `H_A=H_B=4/nu`, and the boundary current is
`M_A_dot=4*a*sin(Delta)=-M_B_dot`, consistent with the full-node
cut calculation. Both common origins `(X_A+X_B)/2` and
`(Theta_A+Theta_B)/2` are fixed on these lifts. Differentiating the
relative phase row yields

\[
\ddot\Delta+4ab\nu^2\sin\Delta=0,\qquad
E_f=4E_c,\qquad E_c=\tfrac12D^2+\beta(1-\cos\Delta).
\]

This is precisely the existing
[nonlinear P2 exchange](RESONANCE_FOUNDATIONS.md#permanent-pulse-admission),
inherited under a complete two-replica blow-up. No primitive oscillator,
new coupling law or activation threshold has been appended.

For the stated initial control choose equal forms `X_A=X_B=C` and
a supplied lift `0<|Delta_0|<pi`. Then

\[
\begin{aligned}
\dot X_A(0)&=a\nu\sin\Delta_0=-\dot X_B(0)\ne0,\\
\dot\Theta_A(0)&=\dot\Theta_B(0)=0,\\
\ddot\Theta_A(0)&=2ab\nu^2\sin\Delta_0
                 =-\ddot\Theta_B(0),\\
\ddot\Delta(0)&=-4ab\nu^2\sin\Delta_0.
\end{aligned}
\]

Thus one region gains form while the other loses it for a nonzero initial
time interval, determined by the sign of the supplied relative phase.
The initially zero phase velocity does not permit holding phase fixed:
the acquired form difference immediately induces its second-order
response. In particular
`Theta_A(t)=Theta_A(0)+a*b*nu^2*sin(Delta_0)*t^2+o(t^2)`.
Reversing `Delta_0` reverses these signed responses. With equal forms
and `Delta_0=0` the source is stationary; an exactly antipodal relative
phase is also stationary and is outside the strict pulse preparation.

The existing P2 theorem gives a periodic complete exchange for this
strictly sub-separatrix energy. Its energy relation also fixes the
maximum regional form displacement without fitting a response:

\[
\max_t|X_A(t)-C|=\max_t|X_B(t)-C|
=\sqrt\beta\,|\sin(\Delta_0/2)|.
\]

Indeed the orbit reaches `Delta=0`, where
`|D|=sqrt(2*beta*(1-cos(Delta_0)))`, and the two form changes are
`D/2` and `-D/2`. The coefficients `w,nu` set the clock scale;
they do not change this displacement at fixed initial phase and `beta`.
This is a prepared reversible transfer and return, not creation of a
new support, attraction to the synchronized submanifold or permanent
unidirectional supply.

### 23.3. Why a positive AL-only endpoint is not that closed flow

Now retain the original full sine state, support, positive held capacities
and law, and evaluate any admitted structural AL-only reset. Let
`h_i=x_i^+-x_i^-` be its **actual** increments after boundary projection
and rounding, including zero at unaffected nodes. The shared AL
postcondition is `h_i>=0`. Consequently

\[
\boxed{M^+-M^-=\sum_i\rho_i h_i\ge0,
\qquad M^+-M^->0\ \Longleftrightarrow\ \exists i:h_i>0.}
\]

Since the unchanged unforced sine flow conserves `M`, any strictly
positive increment excludes this full reset endpoint at **every**
elapsed time of that flow. This obstruction holds for the whole-pair
map that descends in Section 22, as well as for a marked singleton;
it is independent of whether the reset raises or lowers storage. It
also holds with `e>0` for this original normalized-sine law.

This is a full-form endpoint claim in the same chart. If an observation
discards a common form origin, equality of the observed endpoints is a
different question and cannot be excluded solely by this invariant.

Clipped or rounded no-ops have zero increment and evade this particular
obstruction. They are structural identities, not certificates of a
positive-time return or of public AL lifecycle equivalence. A rejected
projection is not an admitted no-op. Nor is zero weighted increment
sufficient for flow realizability: for example a nonzero balanced form
reset at a uniform-form, common-phase equilibrium preserves `M` but
cannot be produced by its stationary unforced flow.

The positive regional control does not contradict the obstruction.
Observing only `A` discards the compensating response in `B`; the
full state never undergoes an AL-only increment. Internal interaction
already explains its gain through an existing boundary phase contrast,
and the phase response evolves with it. Reproducing a pure supplied
AL endpoint instead requires changes beyond that unchanged closed
flow, such as a declared input or compensating external state; their
law and preparation cannot be inferred by naming them. No autonomous
target, hybrid jump, occurrence time or loss reservoir follows from
the continuous transfer.

### 23.4. Shared current and endpoint observations

`SineExchangeComparison.regional_transfer(region=...)` evaluates this
boundary ledger from one captured source. It retains the form-difference
and sine contributions and an independently accumulated full-node rate
residual. `assess_form_increment(increments=...)` computes the exact
change of the conserved weighted form for a supplied full increment
vector. These readers live with the
[complete sine comparison](../../src/tnfr/physics/relational_sine_comparison.py),
so they do not reconstruct the law from a reported response or dispatch
the native runtime. Their finite enclosures remain distinct from the
all-state cancellation proof.

The preceding AL report's `form_increment(outcome=...)` delegates its
actual first-member, second-member or whole-pair increments to this
same ledger. A strictly nonzero weighted increment proves the specified
closed-flow endpoint obstruction; a zero value leaves other realizability
obligations open. No captured boundary rate is a time-integrated transfer,
and no observation executes an event or advances the graph. The
[independent algebra controls](../../tests/physics/test_relational_regional_transfer.py)
derive the cut cancellation, the synchronized fine rows and their phase
acceleration, inherited storage, and a storage-decreasing AL endpoint
which still violates weighted-form conservation.

<a id="sine-zero-resultant-restoration"></a>
## 28. Zero pair resultant and autonomous restoration of form current

### 28.1. A global current observation without a midpoint angle

Keep the complete conservative normalized-sine law and unit doubled-C5
support of [Sections 26-27](SINE_PAIR_GROUPING.md#sine-joint-boundary-acquisition),
with `e=0`, held `nu_i=w=beta=1`, structural
time and no input, clipping or event. The five structural pairs are
`a={2a,2a+1}`, for `a=0,...,4`, with base indices taken modulo five.
Each fine degree is four. Retain all fine real forms and circular phases,
and define the observations

\[
X_a=\frac{x_{2a}+x_{2a+1}}2,\qquad
Z_a=\frac{e^{i\theta_{2a}}+e^{i\theta_{2a+1}}}2.
\]

`Z_a` is a globally defined mean phase phasor, including at zero. It is
neither complex EPI nor a new constitutive state variable. Its argument
is unavailable when `Z_a=0`, which for a pair means exactly antipodal
primitive phases. This does not remove or make either primitive phase
undefined, and the complete sine field has no singularity there.

For adjacent base pairs `a,b`, use the existing boundary balance to define
the contribution to the **mean form rate** in `a` from `b`:

\[
I_{b\to a}:=\frac1{8\pi}
 \sum_{i\in a}\sum_{j\in b}\sin(\theta_j-\theta_i)
 =\boxed{\frac1{2\pi}\operatorname{Im}(\overline Z_a Z_b)}.
\]

The equality follows by factoring the four fine phasors. It is the
globally regular expression of the inherited form row from Section 7,
not another coupling law. In particular

\[
\dot X_a=I_{a-1\to a}+I_{a+1\to a},\qquad
I_{b\to a}=-I_{a\to b},\qquad \sum_a\dot X_a=0.
\]

The weights in Section 23 are `rho_i=d_i/nu_i=4`. Thus the weighted pair
form is `M_a=8X_a`, and its contribution from `b` is `8I_(b->a)`.
This normalization distinguishes a mean-form contribution from the
full weighted cut current; it preserves the existing regional ledger.

If `Z_a=0`, every incident block-form contribution is zero, independently
of neighboring phases. The individual fine edge currents need not vanish:
opposite members can cancel in the block sum. Conversely a zero block
current need not mean a zero resultant; aligned nonzero phasors can also
give a zero imaginary product. Current cancellation is not a nearest-pair
criterion, missing support or full dynamical decoupling.

In particular the nearest-partner inequalities from
[Section 26](SINE_PAIR_GROUPING.md#sine-joint-boundary-acquisition) never enter these analytic
current expressions. Crossing one of their equality boundaries cannot
install a causal switch or a discontinuity in the complete field.

### 28.2. The retained phase channel and internal form information

The complete primitive phase row is still

\[
\dot\theta_i=\frac1{4\pi}\sum_{j\sim i}(x_i-x_j).
\]

Its average rate within a pair is the globally meaningful scalar

\[
\boxed{\Omega_a:=\frac{\dot\theta_{2a}+\dot\theta_{2a+1}}2
 =\frac1\pi\left(X_a-\frac{X_{a-1}+X_{a+1}}2\right).}
\]

Rates of primitive angles agree between local lifts that differ by
constant full turns, so this average rate does not require a global
midpoint angle. In particular it must not be called `d(arg Z_a)/dt`
at `Z_a=0`. The form-to-phase susceptibility to either neighboring
block remains `partial Omega_a/partial X_b=-1/(2*pi)` there.
Replacing that row or the graph degree by a factor proportional to
`|Z_a Z_b|` would change the declared complete law.

One useful derivative identity keeps the missing internal information
visible. Define the derived form-phase moment

\[
Y_a=\frac{x_{2a}e^{i\theta_{2a}}
             +x_{2a+1}e^{i\theta_{2a+1}}}2.
\]

Differentiating the phasors with the full primitive phase row gives

\[
\boxed{\dot Z_a=\frac i\pi\left[
Y_a-\frac{X_{a-1}+X_{a+1}}2 Z_a\right].}
\]

This identity is regular at zero resultant. It is an observation of the
retained fine state, not an assertion that `(X,Z,Y)` closes autonomously
or a replacement for the existing sufficient-state theorem. In particular
`Z_a=0` does not force `Z_dot_a=0`: internal form can make `Y_a` nonzero.
This is the zero-resultant instance of the internal-state obligation
already identified in Section 7.

### 28.3. One frozen antipodal preparation and its complete response

Let `A` be structural pair `(0,1)` and declare

\[
(\theta_0,\theta_1)=(0,\pi),\quad
(x_0,x_1)=(u,-u),\quad u\in\{\tfrac18,-\tfrac18,0\},
\]

with all other forms and phases zero. The half-turn is exact mathematical
phase data. A rounded radian approximation to `pi` cannot substitute for
it when asserting exact antipodality, cancellation or equilibrium.
This supplied preparation is separate from the critical-nearest-boundary
source in Section 26 and from all earlier frozen response protocols.

Initially every pair mean form is zero, `Z_A=0`, and all other pair
phasors equal one. All block currents vanish. In this particular witness
every fine sine current also vanishes, because each fine phase difference
is zero or an exact half-turn; the general cancellation identity does
not require that additional property. The fine form and phase rows give

\[
\dot x(0)=0,\qquad
\dot\theta(0)=\frac u\pi(1,-1,0,0,0,0,0,0,0,0),
\qquad \ddot\theta(0)=0.
\]

Thus zero instantaneous form pressure and zero mean phase rate do not
make the nonzero-`u` state an equilibrium. The phase row moves the two
members in opposite directions. Since `Y_A(0)=u`,

\[
\boxed{\dot Z_A(0)=\frac{iu}\pi,\qquad
\dot Z_b(0)=0\quad(b\ne A).}
\]

For completeness, differentiating every fine form row gives

\[
\ddot x_i(0)=\frac1{4\pi}
\sum_{j\sim i}\cos(\theta_j-\theta_i)
                  [\dot\theta_j-\dot\theta_i]\big|_{t=0},
\]
\[
\boxed{\ddot x(0)=\frac u{\pi^2}
(-1,-1,\tfrac12,\tfrac12,0,0,0,0,\tfrac12,\tfrac12).}
\]

The two neighbors of `A` are the blocks `(2,3)` and `(8,9)`. Their
initial collective accelerations are consequently

\[
\boxed{\ddot X_A(0)=-\frac u{\pi^2},\qquad
\ddot X_1(0)=\ddot X_4(0)=\frac u{2\pi^2},\qquad
\ddot X_2(0)=\ddot X_3(0)=0.}
\]

Equivalently, each incident current satisfies

\[
\dot I_{b\to A}(0)
 =\frac1{2\pi}\operatorname{Im}
       (\overline{\dot Z_A(0)}Z_b(0))
 =-\frac u{2\pi^2},\qquad b\in\{1,4\}.
\]

The opposite current is its negative. The leading collective gains in
the two neighbors exactly compensate the change in `A`; this is also
the all-time cancellation in Section 28.1, not just a Taylor residual.
The full storage is `E=8+4u^2` initially and is conserved. The two nonzero
orientations therefore have equal storage and opposite leading transfers.
There is no supplied AL increment, input, event reserve or new edge.

For `u=0`, both complete fine rows vanish exactly at this preparation.
Uniqueness makes it stationary and its currents remain zero. This control
does not assert that every zero-resultant state with initially zero
internal form is stationary; its full surrounding phase and form state
matters. All three preparations share their initial `(X,Z)` observations
and block currents, while their full internal form states differ.

### 28.4. Immediate restoration, a magnitude cusp and exact silence

For either nonzero frozen orientation, the nonzero derivative proves
that on some sufficiently small punctured interval around zero,

\[
Z_A(t)=\frac{iu}\pi t+O(t^2)\ne0,\qquad
I_{b\to A}(t)=-\frac u{2\pi^2}t+O(t^2)\ne0.
\]

For positive time, both neighboring mean-form currents point away from
`A` when `u>0` and toward it when `u<0`. In particular

\[
X_A(t)=-\frac u{2\pi^2}t^2+O(t^3),\qquad
X_1(t)=X_4(t)=\frac u{4\pi^2}t^2+O(t^3).
\]

The equality between the last two full functions also follows from the
reflection symmetry of the support and this supplied preparation.
Only the local signed response is asserted; there is no numerical
duration, permanent current or selected oscillation period here.

The magnitude behaves differently from the analytic complex phasor:

\[
|Z_A(t)|=\frac{|u|}\pi|t|+O(t^2).
\]

It is not two-sided differentiable at zero. The limiting phasor directions
on the two sides differ by a half-turn, and `arg Z_A(0)` is undefined.
Neither fact is a singularity or jump of the fine state or its field.
One may continue `cos(delta)` as a signed quantity on chosen real lifts,
but may not thereby extend its interpretation as the nonnegative
midpoint-chart magnitude through the antipodal boundary. This observation
does not supply a replacement midpoint angle. Section 29's
[global invariant state](SINE_PAIR_STATE.md#sine-global-pair-state) instead retains the same
fine information without one, on this fixed support and complete law.

There is a stronger limit on the word "restoration." On this unchanged
law the finite-dimensional field is real analytic in real form and local
phase lifts. Its solutions are real analytic in time. The global phasor
products and currents are periodic analytic functions of those coordinates,
so each `I_(b->a)(t)` is real analytic on the connected trajectory interval.
The identity theorem then gives

\[
\boxed{I_{b\to a}(t)=0\text{ on a nonempty open time interval}
\ \Longrightarrow\ I_{b\to a}(t)\equiv0
\text{ on the same connected trajectory interval}.}
\]

The present full sine flow is
[globally continuable](RESONANCE_FOUNDATIONS.md#nonlinear-recurrence),
so the interval can be the entire real time axis. A nonidentically-zero
current instead has isolated zeros, with no accumulation at a finite time.
Thus the witness crosses an isolated instant of exact cancellation; it
does not wait with an exactly silent current for a finite interval and
later switch it on.
Thresholded observations, changed inputs or a declared hybrid event are
different questions. The analytic statement requires this complete law
and its unchanged analytic evolution, not merely smoothness.

The mechanism is therefore an actual, compensated change in collective
form transport caused by retained internal form and evolving primitive
phase. It is more than a relabeling by a nearest-pair rule, but it is
still neither creation of the full coupling channel nor birth of support
or a maintained physical constituent. The phase row and all primitive
edges remain present throughout.

### 28.5. Existing source owners and exact-phase scope

The [complete sine comparison](../../src/tnfr/physics/relational_sine_comparison.py)
already provides the needed detached observations.
`regional_transfer(region=...)` reports the weighted cut current, which is
`8*X_dot_a` for these pairs, rather than the block mean rate itself.
`resultant_kinematics()` differentiates the **node-relative neighbor**
resultant `z_i=sum_(j~i) exp(i*(theta_j-theta_i))`. That is distinct from
the pair phasor `Z_a`. Since `Im(z_i)=S_i`, its imaginary derivative gives
`x_ddot_i=Im(z_dot_i)/(4*pi)` in this fixed conservative unit-capacity
law. Neither observation divides by a resultant or needs a derived angle.

The [regional-transfer controls](../../tests/physics/test_relational_regional_transfer.py)
check the phasor/cut identity, all complete fine jets, retained phase
susceptibility and compensated response. Exact symbolic half-turn controls
establish the stated cancellation and stationary preparation. Separately,
represented phases on either side of mathematical `pi` are evaluated at
their actual captured values: their small nonzero currents remain visible,
and the zero-form case is not mislabeled as an exact equilibrium.
The shared reader's interval evidence and the exact-phase theorem therefore
keep distinct provenance. No new report, runtime law, source search or
global quotient is required for this result.

<a id="sine-moving-pattern-interface"></a>

### 28.6. Retained interaction contract for a moving constituent

The [unordered state](SINE_PAIR_STATE.md#sine-replica-unordered-state),
[internal pulse](SINE_REPLICA_PULSE.md#sine-replica-internal-pulse) and
[current observation](#sine-zero-resultant-restoration) together answer the
retained-interface question on
the **existing doubled C5**, with the same unit capacities, `e=0,w=beta=1`,
fixed complete replica support and structural clock `t`. This is a
consolidation of those conditional results, not a new network, law or
trajectory campaign. A declared pair becomes a predictive constituent only
with its internal state and surrounding interactions still accounted for.

| Obligation | Sufficient information in this model | Existing reason |
| --- | --- | --- |
| Instantaneous mean-form exchange and mean primitive phase rate | Pair means `X_a`, complex mean phasors `Z_a`, actual neighbors and their corresponding values | Section 28.1 factors the four cross edges; Section 28.2 retains the separate form-to-phase row |
| Future interaction on the regular pair chart | Each realized `(X_a,Theta_a,R_a,U_a,Q_a)`, neighboring retained states and relative form/phase origins | Sections 8.1–8.3 reconstruct the full fine state modulo allowed swaps and prove exact inherited evolution |
| Zero resultant with the same support symmetry | The fine state or Section 29's [global invariant state](SINE_PAIR_STATE.md#sine-global-pair-state) | Section 28.3 supplies equal-message/different-future controls; Section 29 retains the discarded information without a midpoint |
| Lost support symmetry | Fine state or the separately admitted mixed state retaining member-to-attachment association | Sections 18–19 identify when a swap ceases to be a symmetry |

In the chart, the internal contribution to the global phasor rate is
already the retained association:

\[
Z_a=R_ae^{i\Theta_a},\qquad
Y_a=e^{i\Theta_a}(X_aR_a+iQ_a).
\]

Substitution into Section 28.2 gives exactly the `R_dot` and `Theta_dot`
rows of Section 8. The new phase/rate observations and the older internal
state therefore describe the same mechanism on their common domain.
For `0<R<1`, the constraint determines `U=Q^2/(1-R^2)`; at `R=1`,
`Q=0` does not determine `U`, so it must be retained. At `R=0`, the
midpoint chart ends, although primitive phases and the current remain
defined. The [regular star chart](SINE_PAIR_STATE.md#sine-star-moment-chart) provides a
separate illustration of such information loss; its rates are not installed
on the doubled cycle. No universal minimal-output realization is asserted.

The surrounding retained network closes jointly. A subsystem still consumes
its neighbors' changing state: declaring that boundary constant changes the
model. Eliminating it instead requires its initial state and a derived causal
memory. The [nonlinear mediator](SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure)
and [tangent C6 bridge](RESONANCE_FOUNDATIONS.md#sine-bridge-causal-memory)
have their own distinct elimination premises. Neither supplies an automatic
kernel for this nonlinear pair network.

**Internal pulse can change exchange while the means stay fixed.** In the
[admitted unit-capacity pulse](SINE_REPLICA_PULSE.md#sine-replica-internal-pulse),
let `alpha=2*pi/5` and retain
the common internal amplitude `R(t)=cos(delta(t))`. Its means obey
`X_a=bar X`, `Theta_a=bar Theta+a*alpha`, while the two incident contributions
are

\[
I_{a+1\to a}(t)=\frac{R(t)^2\sin\alpha}{2\pi},\qquad
I_{a-1\to a}(t)=-I_{a+1\to a}(t),\qquad \dot X_a=0.
\]

For the already derived nonstationary libration `0<m<1`,
`1-m<=R(t)^2<=1`. Thus the positive oriented contribution varies between
`(1-m)*sin(alpha)/(2*pi)` and `sin(alpha)/(2*pi)` despite constant means
and zero net regional current. The other orientation compensates it.
This follows by substitution into the established phasor/cut identity and
amplitude bound. It is form circulation, not a regional energy-transfer law,
external drive, proof of transverse stability or spontaneous preparation.
The full fine-edge acute domain retains Section 9's additional amplitude
restriction; the entire libration range is not thereby declared fully acute.
An observer of only net regional change misses this interaction.

Implementation remains with the existing
[scale/mixed-state owner](../../src/tnfr/physics/relational_sine_scale.py),
[regional ledger and phase kinematics](../../src/tnfr/physics/relational_sine_comparison.py).
The [interface contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-moving-pattern-interface)
connects them without a second state registry or executor. Native Arg
attachment cards, prepared phase-offset partitions and tangent reductions
retain their separate laws and information budgets. Existing full-row,
cut-current and antipodal controls supply verification; their proofs need
neither a new frozen response nor additional fitting parameters.

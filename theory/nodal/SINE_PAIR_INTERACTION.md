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

<a id="sine-pair-finite-exchange"></a>

## 30. Finite form exchange near an invisible pair family

### 30.1. Declared autonomous preparations and finite observation

Use Section 28's complete conservative normalized-sine law on the fixed
unit doubled C5. Fine node order is `0,...,9`, pair `a` is `(2a,2a+1)`, and
each adjacent base pair has all four unit cross edges. There are no internal
pair edges. Hold every capacity at one, `e=0,w=beta=1`, the support and its
weights fixed, with no input, event, clipping or added pressure. Signed real
form and primitive circular phase remain separate coordinates. In the
structural clock `t`, put `tau=t/pi`; primes throughout this section mean
derivatives with respect to `tau`. Both fine evolution rows become

\[
x_i'=\frac14\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
\theta_i'=x_i-\frac14\sum_{j\sim i}x_j.
\]

These are the supplied complete law, not a pressure reconstructed from the
response. Use the existing [global pair state](SINE_PAIR_STATE.md#sine-global-pair-state)
to retain all internal associations and the actual evolving environment.
The phase phasors below are exact unit Cartesian values, not rounded radian
approximations to an antipodal pair.

All ten initial forms are zero. Fix selected pair 0 and compare its two
unordered orientations

\[
A:\ (z_0,z_1)=(1,-1),\qquad
B:\ (z_0,z_1)=(i,-i).
\]

In both cases `X_0=Z_0=U_0=W_0=0`; the retained products are respectively
`P_0=-1` and `P_0=1`. For a declared unit `q=a+ib`, each environmental pair
has coincident member phasors and zero internal form. Define two classes
of complete initial preparations by their ordered environmental resultants:

\[
\begin{aligned}
\text{cancellation control: }&(Z_1,Z_2,Z_3,Z_4)=(q,q,-q,-q),\\
\text{broken cancellation: }&(Z_1,Z_2,Z_3,Z_4)=(q,1,-1,-1).
\end{aligned}
\]

Each class compares A and B with exactly the same surrounding retained
state. Both classes tend to the stationary invisible family at `q=1`.
Here breaking symmetry means breaking the environmental cancellation
`Z_4=-Z_1`; the support's independent member-swap symmetries are preserved.
These are separately prepared autonomous trajectories. No source is held
fixed after initialization, and no trajectory is silently interrupted.

Freeze the signed directed observation and its normalization as

\[
j_{1\to0}(\tau)=\pi I_{1\to0}(\pi\tau)
 =\frac12\operatorname{Im}(\overline Z_0Z_1),\qquad
Q_{1\to0}(\sigma)=\int_0^{\pi\sigma}I_{1\to0}(t)\,dt
 =\int_0^\sigma j_{1\to0}(\tau)\,d\tau,
\]

with `Delta Q=Q_A-Q_B`. This is the contribution to pair 0's mean form
from pair 1, in form units. It is neither the net mean-form change nor an
energy-current observation. The weighted-cut transfer is `8*Q`.

The fixed certificate preparation and horizon are

\[
q=\frac{399+40i}{401},\qquad \sigma=\frac1{20},\qquad T=\pi\sigma.
\]

The declared error budget is the analytic remainder
`E(sigma)=8*sigma^5/15+8*sigma^7/105`, proved below. The acceptance condition
is that the signed difference enclosure excludes zero. Calculation uses
exact rational arithmetic for this Cartesian preparation, horizon and
enclosure; no time step, random seed, trajectory solver or simulated
response is part of this certificate. An enclosure containing zero remains
unresolved under this budget; neither horizon nor preparation is adjusted
to obtain a sign. The formulas also apply to other unit `q` and positive
finite `sigma`, with their own explicitly supplied preparations and horizons.

### 30.2. Exact control, realizability and retained balances

Every declared fine phasor has norm one, so the fine preparations realize
the [global invariant constraints](SINE_PAIR_STATE.md#sine-global-pair-state)
exactly, including `Z_0=0`. In the cancellation control the neighboring
resultants `B_a=(Z_(a-1)+Z_(a+1))/2` are

\[
(B_0,B_1,B_2,B_3,B_4)=(0,q/2,0,0,-q/2).
\]

For pair 0 its fine form row is zero because `B_0=0`. For every other
pair its phasor is aligned with its nonzero `B_a`, if any; hence its fine
form row is also zero. All phase rows vanish because all forms are zero.
Uniqueness makes each preparation stationary for all time. The same proof
holds for any selected antipodal pair `(d,-d)` with `|d|=1`: this is a
symmetry-preserving family of perturbations retaining a continuum of
invisible orientations. Every selected incident block current and its
integral is exactly zero in this control.

For broken cancellation only `B_0(0)=(q-1)/2` is supplied as an initial
value. It subsequently evolves with all five pairs; replacing it by a
constant would invalidate the proof's model. Realizability and finite-time
continuation follow from the complete fine law and its global invariant
representation, not from a tolerance on constraints.

Retain the original fine storage and its pair expression:

\[
\begin{aligned}
F&=\frac12\sum_{\{i,j\}\in E_f}(x_i-x_j)^2
 =2\sum_{\{a,b\}\in C_5}(X_a-X_b)^2+4\sum_aU_a,\\
V&=\sum_{\{i,j\}\in E_f}[1-\cos(\theta_i-\theta_j)]
 =4\sum_{\{a,b\}\in C_5}[1-\operatorname{Re}(\overline Z_aZ_b)],\\
H&=F+V.
\end{aligned}
\]

The fine gradients in the declared clock are
`partial F/partial x_i=4*theta_i'` and
`partial V/partial theta_i=-4*x_i'`. Therefore

\[
F'=4\sum_i\theta_i'x_i',\qquad
V'=-4\sum_i x_i'\theta_i',\qquad H'=0.
\]

Antisymmetry of the sine terms also gives `sum_a X_a'=0`. Initially `F=0`;
the control has `H=16`, and broken cancellation has `H=20-4*a` for both A
and B. Thus the orientation comparison has identical full storage, although
changing the environmental preparation can change that common storage.
There is no external reservoir, support jump or supplied work.

Internal form is retained rather than suppressed when the initial pair
resultant vanishes. With `D=(z_+-z_-)/2`, `u=(x_+-x_-)/2`, `U=u^2` and
`W=uD`, the exact internal rows include

\[
u'=\operatorname{Im}(\overline D B),\qquad
U'=2\operatorname{Im}(\overline W B).
\]

Writing `r=(a-1)/2` and `s=b/2`, the selected pair initially has
`u_A'=s`, `u_B'=-r`, `W_A'=s` and `W_B'=-ir`. In particular

\[
U_A'(0)=U_B'(0)=0,\qquad
U_A''(0)=2s^2,\qquad U_B''(0)=2r^2.
\]

These internal balances contribute to `F` as the full law evolves. Neither
zero initial mean form nor zero initial resultant removes them. They do not
turn the directed form ledger into an energy ledger.

### 30.3. The orientation-dependent current coefficient

At the initial instant all phase rates, all `Z_a'` and the selected `h_0`
vanish, where `h_a=X_a-(X_(a-1)+X_(a+1))/2`. The inherited pair rows at
`Z_0=U_0=W_0=0` give

\[
Z_0''(0)=\frac{B_0(0)+P_0(0)\overline{B_0(0)}}2.
\]

Consequently broken cancellation has

\[
Z_{0,A}''(0)=\frac{ib}{2},\qquad
Z_{0,B}''(0)=\frac{a-1}{2},\qquad
\Delta Z_0''(0)=-\overline{B_0(0)}.
\]

For either orientation, `j_(1->0)(0)=j_(1->0)'(0)=0`. Differentiating
the actual directed current, including the evolving neighboring phasor,
gives at that instant

\[
j_{1\to0}''(0)=\frac12\operatorname{Im}(\overline{Z_0''(0)}q),\qquad
j_A''(0)=-\frac{ab}{4},\qquad
j_B''(0)=\frac{(a-1)b}{4}.
\]

Thus the candidate leading term in its time integral is

\[
L(q,\sigma)=-\frac{\operatorname{Im}(q^2-q)}{24}\sigma^3
 =-\frac{b(2a-1)}{24}\sigma^3.
\]

A difference in these derivatives alone is insufficient for the declared
finite observation. The following bound controls all remaining contributions,
including the selected pair's backreaction on its neighbors.

### 30.4. A global remainder for the finite transfer

For either complete trajectory and every `s>=0`, the fine rows imply

\[
|x_i'(s)|\le1,\qquad |x_i(s)|\le s,\qquad
|\theta_i'(s)|\le2s,\qquad |\theta_i''(s)|\le2.
\]

The first estimate uses the average of four bounded sine terms; the second
uses the zero initial forms. The next two follow by applying respectively
the phase row and its derivative to those bounds. Differentiating the form
row once more gives

\[
|x_i''(s)|
 \le\frac14\sum_{j\sim i}|\theta_j'(s)-\theta_i'(s)|\le4s,
\qquad |\theta_i'''(s)|\le8s.
\]

Any continuous lift of each primitive phase has these derivatives. Constant
full-turn changes of lift leave the bounds and phasors unchanged. For
`z_i=exp(i*theta_i)`, direct differentiation gives

\[
\begin{aligned}
z_i'&=i\theta_i'z_i,\\
z_i''&=[i\theta_i''-(\theta_i')^2]z_i,\\
z_i'''&=[i\theta_i'''-3\theta_i'\theta_i''-i(\theta_i')^3]z_i.
\end{aligned}
\]

Since `|z_i|=1`, it follows that

\[
|z_i'|\le2s,\qquad |z_i''|\le2+4s^2,\qquad
|z_i'''|\le20s+8s^3.
\]

Averaging within a pair preserves these bounds, and `|Z_a|<=1`. For any
directed pair current `j=(1/2)*Im(conj(Z_a)*Z_b)`, the product rule now yields

\[
\begin{aligned}
|j'''(s)|
&\le\frac12\left[
 |Z_a'''||Z_b|+3|Z_a''||Z_b'|
 +3|Z_a'||Z_b''|+|Z_a||Z_b'''|\right]\\
&\le\frac12\left[2(20s+8s^3)+6(2+4s^2)(2s)\right]
 =32s+32s^3.
\end{aligned}
\]

This estimate uses all actual fine rows. It does not freeze any environment
coordinate or truncate its feedback. The selected incident currents have
zero value and first derivative initially. Taylor's integral remainder gives

\[
j(\tau)=\frac{j''(0)}2\tau^2+R_j(\tau),\qquad
|R_j(\tau)|\le\frac12\int_0^\tau(\tau-s)^2(32s+32s^3)\,ds
 =\frac43\tau^4+\frac4{15}\tau^6.
\]

Integrating once more, each orientation's directed transfer satisfies

\[
Q(\sigma)=\frac{j''(0)}6\sigma^3+R_Q(\sigma),\qquad
|R_Q(\sigma)|\le\frac4{15}\sigma^5+\frac4{105}\sigma^7.
\]

Subtracting the two trajectories proves the finite consequence

\[
\boxed{\Delta Q_{1\to0}(\sigma)\in
 [L(q,\sigma)-E(\sigma),\ L(q,\sigma)+E(\sigma)],\qquad
E(\sigma)=\frac8{15}\sigma^5+\frac8{105}\sigma^7.}
\]

The bound holds for every unit `q` and every positive finite `sigma`.
Excluding zero requires the additional strict inequality `|L|>E`; a
zero-containing enclosure proves no sign or equality. For any nonzero
`Im(q^2-q)`, sufficiently small positive horizons satisfy that inequality.
In particular `q=exp(i*epsilon)` with small nonzero `epsilon` permits
arbitrarily nearby symmetry-breaking preparations with a certified finite
difference, using horizons small enough that
`8*sigma^2/15+8*sigma^4/105<|Im(q^2-q)|/24`. This is an existence and
quantitative error statement for declared preparations, not an automatic
horizon search.

### 30.5. The declared finite certificate and its scope

For Section 30.1's fixed preparation and horizon, exact arithmetic gives

\[
\operatorname{Im}(q^2-q)=\frac{15880}{160801},\qquad
L=-\frac{397}{771844800},\qquad
E=\frac{2801}{16800000000}.
\]

Therefore

\[
\boxed{
\Delta Q_{1\to0}\!\left(\frac1{20}\right)
\in\left[
-\frac{1839903601}{2701456800000000},
-\frac{313032133}{900485600000000}
\right]\subset(-\infty,0).
}
\]

The approximate endpoints are `-6.8108e-7` and `-3.4763e-7` in the declared
form units; the rational endpoints supply the certificate. Both complete
preparations conserve `H=6424/401` and total mean form zero. Their initial
internal second derivatives are `U_A''(0)=800/160801` and
`U_B''(0)=2/160801`. The corresponding cancellation controls have exactly
zero transfer, proved for all time in Section 30.2. The weighted-cut
difference lies in eight times the displayed interval.

**Directed exchange and net change have different sensitivity.** The other
incident contribution uses `Z_4(0)=-1`, so the same derivative calculation
and remainder argument give

\[
\Delta Q_{4\to0}(\sigma)=\frac b{24}\sigma^3+R_4,
\qquad |R_4|\le E(\sigma).
\]

Because `X_0(0)=0` in both preparations, the exact regional ledger implies

\[
\Delta X_0(T)
 =\Delta Q_{1\to0}(\sigma)+\Delta Q_{4\to0}(\sigma)
 =\frac{b(1-a)}{12}\sigma^3+R_X,\qquad |R_X|\le2E(\sigma).
\]

Here the mean-form endpoint uses the original clock `T=pi*sigma`.
For `q=exp(i*epsilon)`, the two directed leading coefficients are order
`epsilon`, whereas their sum is order `epsilon^3`. This compensation is why
the declared observation selects an actual directed edge contribution. At
the fixed preparation and horizon the net leading term is `1/192961200`
and its bound has radius `2801/8400000000`; that net enclosure contains
zero. The directed certificate therefore makes no finite sign claim about
the net mean change under this error budget.

The result establishes a finite orientation-dependent interaction under
the supplied complete law: identical initial selected interfaces `(X,Z)`
with the same surrounding preparation can produce different accumulated
pair-to-pair form exchange when environmental cancellation is broken.
The proof retains neighboring motion, internal form and storage throughout.
It compares separate autonomous preparations; exact interval silence has
not been made to restart along an unchanged analytic trajectory.

The shared `assess_sine_pair_finite_exchange(phase_rotation=...,
horizon_tau=...)` calculation in the
[pair owner](../../src/tnfr/physics/relational_sine_pair.py) admits the
primitive Cartesian rotation and horizon, reconstructs the declared fine
preparations through the global-state owner, and evaluates this analytic
enclosure. Its [contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-finite-exchange)
and [usage](../../docs/guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-finite-exchange)
retain the original clock, current normalization and unavailable-sign
branch. It certifies a bound on the continuous trajectories; it supplies
no sampled trajectory or executed numerical propagation.

Preparation, fixed support, held capacities, grouping and normalized sine
remain premises. The result establishes neither formation of these
preparations, autonomous selection of an occurrence, stability under an
unspecified perturbation class, physical constituent identification nor
selection of a unique pressure law. A different observation or complete
law requires its own finite consequence and error bound.

<a id="sine-pair-receiver-readout"></a>

## 31. A finite receiver readout with preparation and observation errors

### 31.1. Declared complete preparations, readout and error budget

Retain Section 30's conservative normalized-sine law, fine node order,
unit doubled-C5 support, held unit capacities, `e=0,w=beta=1`, and absence
of inputs, events and clipping. The exact structural clock is `t`, with
`tau=t/pi`; primes below mean derivatives in `tau`. Keep the same broken-
cancellation nominal preparations A and B: all ten forms zero, selected
pair 0 phasors respectively `(1,-1)` and `(i,-i)`, and ordered
environmental pair phasors `(q,1,-1,-1)`, each repeated on its two members.
The stationary controls instead use `(q,q,-q,-q)` as before.

The new observation is one external pair's complete mean-form endpoint:

\[
Y(T)=X_1(T)=\frac{x_2(T)+x_3(T)}2,\qquad T=\pi h.
\]

Both incident connections contribute to that endpoint. Its measurement
does not supply a selected-edge integral, the hidden pair's internal
coordinates, phase derivatives or a record of the intervening trajectory.
Freeze the nominal rotation, elapsed horizon and independent error budgets
before assessing the readout:

\[
q=\frac{399+40i}{401},\qquad h=\frac1{20},\qquad
\rho_x=\rho_\theta=\eta=\frac1{10^8}.
\]

For each alternative A or B, admit every complete initial state satisfying

\[
|x_i(0)-x_i^*(0)|\le\rho_x,\qquad
|\operatorname{wrap}(\theta_i(0)-\theta_i^*(0))|\le\rho_\theta
\quad(i=0,\ldots,9).
\]

Here a star denotes that alternative's nominal preparation. The form and
phase errors are independently bounded on every fine node, including the
receiver, hidden pair and environment. The budgets use the declared form
units and radians, respectively. No common-error cancellation, unknown
zero-error environment or exact perturbed antipodality is assumed. All
perturbed phases remain primitive circle coordinates, hence realizable.
The support, capacity, coefficients and clock have no error in this
admission. The recorded scalar is `y=Y(T)+epsilon_y` with
`|epsilon_y|<=eta`; this last budget is an observation assumption, not a
sensor model derived from the nodal law.

The form preparation bounds also bound the common form origin. An
unrestricted unknown common offset would translate the absolute readout
by that offset and preclude this one-time discrimination. No unobserved
baseline is subtracted and no second measurement is implicitly consumed.

The prospective nominal error budget for each branch is Section 30's
`E(h)=8*h^5/15+8*h^7/105`, now accounting for the two incoming currents.
Add the preparation allowance
`(rho_x+2*h*rho_theta)/(1-4*h^2)` and then `eta` to each branch's endpoint
radius. The two budgets remain distinct; the coefficients use this law's
fixed form and clock normalization. The complete-law argument below proves
this conservative rational majorant on `0<h<1/2`. That domain belongs to
the chosen bound; it is not a loss of existence or a physical stability
threshold at `h=1/2`.

The stopping rule is strict disjointness of the two predicted recorded-
readout intervals with these positive error budgets. An overlap or a
shared endpoint leaves discrimination unavailable under the declared
bound. No preparation, horizon or budget is adjusted to force separation.
This is an analytic prediction for the specified alternatives; no measured
sample or executed numerical trajectory is supplied or claimed.

### 31.2. Absolute nominal predictions retain both incoming currents

Write `q=a+ib`. Both nominal preparations have the initial mean-form rates

\[
(X_0',X_1',X_2',X_3',X_4')(0)=(0,-b/2,b/2,0,0).
\]

All initial primitive phase rates and phasor first derivatives vanish.
The surrounding pairs have coincident members, so their initial internal
form and phasor-difference rates vanish as well. The complete phase row
therefore gives, in both preparations,

\[
Z_1''(0)=-\frac{3ibq}{4},\qquad Z_2''(0)=\frac{3ib}{4}.
\]

The full receiver row is `X_1'=j_(0->1)+j_(2->1)`. For its second
connection, the actual current and its derivatives satisfy

\[
j_{2\to1}(0)=-\frac b2,\qquad j_{2\to1}'(0)=0,\qquad
j_{2\to1}''(0)
 =\frac12\operatorname{Im}
   (\overline{Z_1''(0)}Z_2(0)+\overline{Z_1(0)}Z_2''(0))
 =\frac{3ab}{4}.
\]

For the selected connection, antisymmetry and Section 30.3 give zero
initial current and first derivative, with
`j_(0->1),A''(0)=ab/4` and `j_(0->1),B''(0)=(1-a)*b/4`. Hence

\[
X_{1,A}'''(0)=ab,\qquad X_{1,B}'''(0)=\frac{(2a+1)b}{4}.
\]

The common initial linear drift remains in each absolute prediction:

\[
C_A(q,h)=-\frac{bh}{2}+\frac{ab}{6}h^3,\qquad
C_B(q,h)=-\frac{bh}{2}+\frac{(2a+1)b}{24}h^3.
\]

Section 30.4's bound `|j'''(s)|<=32*s+32*s^3` applies to both incoming
currents, including the one whose initial value is nonzero. Retain that
constant term in Taylor's formula before integration. Each integrated
current has remainder at most `4*h^5/15+4*h^7/105`; summing the two gives

\[
|X_{1,A}^*(\pi h)-C_A(q,h)|\le E(h),\qquad
|X_{1,B}^*(\pi h)-C_B(q,h)|\le E(h).
\]

Their center difference is `Im(q^2-q)*h^3/24`, but the absolute intervals
also require the common drift and the other incoming connection. The
bound follows from all actual fine rows, so subsequent changes of either
environmental current and every backreaction are included.

### 31.3. Independent preparation errors and the recorded intervals

For each actual initial circle state, choose a real phase lift whose
difference from the corresponding nominal lift realizes the initial
wrapped-distance bound. Such a lift exists independently at every node.
Continue both lifts under the complete equations. Their differences need
not remain in one principal branch: the sine field is periodic, and the
comparison below uses these continuous lifts rather than rewrapping a
trajectory or observing its phase derivatives.

Let `u_x(tau)=max_i|x_i(tau)-x_i^*(tau)|` and let `u_theta` be the analogous
maximum for the chosen phase-lift differences. The global sine Lipschitz
bound and the actual degree-four averages imply the coupled integral
inequalities

\[
u_x(\tau)\le\rho_x+2\int_0^\tau u_\theta(s)\,ds,\qquad
u_\theta(\tau)\le\rho_\theta+2\int_0^\tau u_x(s)\,ds.
\]

The first factor two bounds a neighbor phase error plus the receiving
node's phase error; the second follows from the same difference structure
in the phase row. These coefficients belong to the fixed normalization
in Section 31.1; a change of form units or clock also changes the
corresponding row coefficients. The two error channels are not identified.

Comparison with the nonnegative matrix `[[0,2],[2,0]]`, or iteration of
the integral inequalities, gives

\[
u_x(h)\le\rho_x\cosh(2h)+\rho_\theta\sinh(2h).
\]

For `0<h<1/2`, bound each nonnegative exponential-series coefficient
`1/k!` by one. The even and odd geometric sums give respectively
`cosh(2h)<=1/(1-4*h^2)` and `sinh(2h)<=2*h/(1-4*h^2)`. Thus the actual
receiver mean differs from its nominal endpoint by at most

\[
|X_1(\pi h)-X_1^*(\pi h)|\le u_x(h)\le
P(h,\rho_x,\rho_\theta):=\frac{\rho_x+2h\rho_\theta}{1-4h^2}.
\]

This controls independent errors throughout the complete preparation,
including its common form origin. It does not condition the environment
on a prescribed future or use cancellation between the two alternatives.
Each actual trajectory conserves the original `H` and total mean form
appropriate to its own initial state. Neither the nominal common storage
`6424/401` nor its zero total mean is imposed on uncertain preparations.

Adding the scalar readout error yields the two closed prediction intervals

\[
\boxed{\mathcal J_A=[C_A-R,C_A+R],\qquad
\mathcal J_B=[C_B-R,C_B+R],\qquad
R=E(h)+P(h,\rho_x,\rho_\theta)+\eta.}
\]

Every admitted recorded value from either alternative lies in its own
interval. Strict separation is sufficient for binary discrimination.
Overlapping outer intervals establish only that this bound is unavailable,
not actual indistinguishability or inconsistency of the complete law.

### 31.4. The fixed readout certificate and the stationary control

For the preparation, horizon and positive budgets declared in Section 31.1,
exact rational evaluation gives

\[
C_A=-\frac{160267}{64320400},\qquad
C_B=-\frac{1923601}{771844800},\qquad
P=\frac1{90000000},\qquad R=\frac{9467}{50400000000}.
\]

With `D=8104370400000000`, the recorded intervals are exactly

\[
\mathcal J_A=\frac1D[-20195164303067,-20192119696933],\qquad
\mathcal J_B=\frac1D[-20199332803067,-20196288196933].
\]

They have the strictly positive separation

\[
\boxed{\inf\mathcal J_A-\sup\mathcal J_B
 =\frac{561946933}{4052185200000000}>0.}
\]

The gap is approximately `1.3868e-7` in the declared form units after
including both preparation and readout errors. The exact rational
intervals certify the sign; no rounded displayed endpoint is used for
admission. Conditional on a record being generated by one of the two
admitted complete preparation families and the stated readout-error bound,
its interval identifies which alternative generated it using this receiver
alone. Membership of an arbitrary supplied number in an outer interval
establishes neither source realizability nor authentication. A value outside
both intervals would require checking the full preparation, law, clock and
observation premises; no measured value is supplied here.

For Section 30's cancellation control, each exact nominal trajectory is
stationary with receiver center zero. Arbitrary admitted initial errors
can break that stationarity and cancellation. The same comparison proof
and readout budget give the common control enclosure

\[
\mathcal J_{A,\mathrm{control}}=\mathcal J_{B,\mathrm{control}}
 =[-(P+\eta),P+\eta]
 =\left[-\frac{19}{900000000},\frac{19}{900000000}\right].
\]

There is no nominal Taylor error for this exact stationary center. The
shared outer interval does not prove identical perturbed histories or
zero actual readout. It retains the exact symmetry control while making
its uncertainty explicit.

The shared `assess_sine_pair_receiver_readout` calculation in the
[pair owner](../../src/tnfr/physics/relational_sine_pair.py) consumes the
primitive rotation, horizon and three independent error bounds and returns
`SinePairReceiverReadout`. The
[contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-readout)
and [usage](../../docs/guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-readout)
own its source reconstruction and interval ordering. The report tests strict
separation after outward endpoint materialization, so its gap lower bound
may be slightly smaller than the exact rational theorem value above. An
unresolved reported gap remains unavailable. This assessment predicts
admitted readouts; it consumes no actual measurement or numerical trajectory.

This result gives a finite external record that separates two declared
internal-orientation families under one supplied law. It does not recover
an arbitrary hidden state, select that law, infer an independently measured
physical quantity, or demonstrate formation and long-term maintenance.
The original Section 30 finite-exchange certificate remains unchanged;
the new observation, full-state preparation budgets and scalar readout
budget are separate premises of this receiver certificate.

<a id="sine-pair-receiver-constitutive-confounding"></a>

## 32. A finite collision between constitutive law and internal orientation

### 32.1. Declared law family and finite comparison

Section 31's receiver certificate conditions on the normalized-sine law.
Now allow one already admitted constitutive freedom: the
[cubic-sine storage family](SINE_CONSTITUTIVE_INFORMATION.md#phase-storage-selection-boundary).
Keep the same ordered ten fine nodes, doubled-C5 support, unit held
capacities, fixed form units, and structural clock `tau=t/pi`. Supply the
complete conservative rows

\[
x_i'=\frac14\sum_{j\sim i}j_\epsilon(\theta_j-\theta_i),\qquad
\theta_i'=x_i-\frac14\sum_{j\sim i}x_j,\qquad
j_\epsilon(\delta)=\sin\delta+\epsilon\sin^3\delta.
\]

There are no inputs, events, clipping or extra phase velocities. The
parameter `epsilon` is constant along each trajectory; its value selects
a complete supplied law. This changes the phase-storage premise and
phase-to-form row. It is distinct from changing both rows through the
alternative reciprocal-mobility family.

Retain Section 30's exact broken-cancellation preparations A and B and
the absolute receiver readout `Y(T)=(x_2(T)+x_3(T))/2`. Freeze

\[
q=\frac{399+40i}{401}=a+ib,\qquad h=\frac1{20},\qquad T=\pi h,
\qquad 0\le\epsilon\le\frac1{10}.
\]

All ten initial forms are exactly zero, including their common origin;
the primitive phases and observation are exact in this comparison. In A,
selected pair 0 has phasors `(1,-1)`; in B they are `(i,-i)`; the same
environmental pair phasors `(q,1,-1,-1)` are each repeated on their two
members. Compare the actual endpoint `Y_A^epsilon(T)` with `Y_B^0(T)`.
The unknown law coefficient and the internal orientation are the only
changed premises. No initial form shift, clock adjustment, measurement
error or fitted source is used to create a collision.

The declared stopping rule is an actual equality of these finite
endpoints for at least one `epsilon` in `(0,1/10)`. Prove opposite strict
signs of their difference at the two fixed coefficient endpoints with
complete-law remainder bounds, then use continuity in the coefficient.
Overlapping outer prediction intervals do not satisfy this rule. No root
search or estimate of a coefficient matching an evaluated reading is
part of the certificate. The law family, coefficient interval, preparation,
readout and horizon precede the endpoint sign calculation below.

### 32.2. Own-law storage, control and continuation

For each admitted coefficient the nonnegative phase potential is

\[
U_\epsilon(\delta)=1-\cos\delta+
 \frac\epsilon3(1-\cos\delta)^2(2+\cos\delta),\qquad
U_\epsilon'=j_\epsilon.
\]

With `S_epsilon,i=sum_j j_epsilon(theta_j-theta_i)`, retain the original
form storage and this law's phase storage:

\[
H_\epsilon=\frac12\sum_{\{i,j\}\in E_f}(x_i-x_j)^2
 +\sum_{\{i,j\}\in E_f}U_\epsilon(\theta_j-\theta_i).
\]

Its gradients are `L_f*x` and `-S_epsilon`, so the complete rows give
`H_epsilon'=sum_i[(L_f*x)_i*S_epsilon,i/4
-S_epsilon,i*(L_f*x)_i/4]=0`. Odd edge currents also conserve total form.
Each trajectory uses its own declared law and storage. In particular the
cosine storage of the `epsilon=0` reference is not imposed as a conserved
quantity on the other members. Changing the potential changes its value
on a fixed preparation; no common initial storage across different laws
is required for this comparison.

All laws have `j_epsilon'(0)=U_epsilon''(0)=1`, hence the same complete
consensus tangent in the same clock. This local agreement does not identify
their nonlinear responses. For `epsilon>0`, the third harmonic in
`sin^3(delta)` also invalidates importing the sine-only first-moment current
formula. The fine rows, rather than a reused sine current report, determine
the alternative prediction below.

The cancellation control remains exact for every coefficient and every
selected antipodal orientation `(d,-d)`. The identity
`j_epsilon(delta+pi)=-j_epsilon(delta)` cancels the currents from each
antipodal neighbor pair. The other incident gaps in the control are zero
or a half-turn, where the current vanishes. Thus every fine form rate is
zero; the zero forms also make every phase rate zero. Uniqueness keeps
the entire control stationary, including `Y=0`. This argument uses the
actual fine kernel, not first-resultant sufficiency for the changed law.

For any fixed finite `epsilon>=0`, put `M=1+epsilon`. Since `|x_i'|<=M`, the zero
initial forms satisfy `|x_i(tau)|<=M*|tau|`. The phase row then has at most
linear growth in `|tau|`. The smooth complete field therefore continues
for every finite time. Its analytic dependence on state and coefficient
gives continuous endpoint dependence throughout the closed coefficient
interval, as needed for the finite equality argument.

### 32.3. Receiver coefficients from the complete alternative law

Put `g=b+epsilon*b^3=j_epsilon(arg(q))`. In preparation A, direct summation
of the four incident fine currents gives initial selected-member form
rates `g/2,-g/2` and the pair-mean rates

\[
(X_0',X_1',X_2',X_3',X_4')(0)=(0,-g/2,g/2,0,0).
\]

All primitive phase first derivatives vanish initially. Differentiating
the complete phase row therefore gives the selected-member accelerations
`3*g/4,-g/4`, receiver-member acceleration `-3*g/4`, and acceleration
`3*g/4` for both members of pair 2. For a receiver fine node, the four
incident current derivatives with respect to their phase gaps are

\[
a(1+3\epsilon b^2),\quad -a(1+3\epsilon b^2),\quad
a(1+3\epsilon b^2),\quad a(1+3\epsilon b^2).
\]

Their corresponding gap accelerations are `3*g/2,g/2,3*g/2,3*g/2`.
In `x_i'''(0)`, terms quadratic in gap velocities vanish because every
initial phase velocity is zero. Averaging the four products yields

\[
Y_A^\epsilon(0)=0,\qquad (Y_A^\epsilon)'(0)=-g/2,\qquad
(Y_A^\epsilon)''(0)=0,\qquad
(Y_A^\epsilon)'''(0)=a(1+3\epsilon b^2)g.
\]

Here derivatives of the readout use `tau`; its finite endpoint remains
at `T=pi*h` in the original clock. Consequently its cubic center is

\[
C_A^\epsilon(q,h)=-\frac{gh}{2}
 +\frac{a(1+3\epsilon b^2)g}{6}h^3.
\]

Preparation B uses the reference law, so Section 31 gives

\[
C_B^0(q,h)=-\frac{bh}{2}+\frac{(2a+1)b}{24}h^3.
\]

The coefficient changes the common receiver drift already at first order
in time. Keeping only the orientation-dependent cubic term would omit
the constitutive effect being tested.

### 32.4. A finite remainder uniform over the declared coefficient interval

The global kernel bounds

\[
|j_\epsilon|\le M=1+\epsilon,\quad
|j_\epsilon'|\le L=1+3\epsilon,\quad
|j_\epsilon''|\le J_2=1+9\epsilon,\quad
|j_\epsilon'''|\le J_3=1+27\epsilon
\]

follow by differentiating `sin(delta)+epsilon*sin(delta)^3` and bounding
each sine and cosine by one. For example, the extra second derivative is
`6*sin(delta)*cos(delta)^2-3*sin(delta)^3`, and its third derivative is
`6*cos(delta)^3-21*sin(delta)^2*cos(delta)`. These bounds are conservative
and require no acute-gap restriction.

At time `s>=0`, the complete zero-form preparation gives

\[
|x_i'|\le M,\quad |x_i|\le Ms,\quad
|\theta_i'|\le2Ms,\quad |\theta_i''|\le2M.
\]

For a continuous lifted gap `delta_ij=theta_j-theta_i`, this implies
`|delta_ij'|<=4*M*s` and `|delta_ij''|<=4*M`. Differentiating the form row
gives `|x_i''|<=4*L*M*s`; the phase row then gives
`|delta_ij'''|<=16*L*M*s`. Differentiate the form row three times:

\[
x_i''''=\frac14\sum_{j\sim i}
 \left[j_\epsilon'''(\delta_{ij})(\delta_{ij}')^3
 +3j_\epsilon''(\delta_{ij})\delta_{ij}'\delta_{ij}''
 +j_\epsilon'(\delta_{ij})\delta_{ij}'''\right].
\]

Therefore every fine form, and hence their receiver mean, satisfies

\[
|x_i''''(s)|\le
 (48J_2M^2+16L^2M)s+64J_3M^3s^3.
\]

Taylor's integral remainder through order three proves

\[
\begin{aligned}
|Y_A^\epsilon(\pi h)-C_A^\epsilon(q,h)|
&\le E_\epsilon(h),\\
E_\epsilon(h)
&=\frac{48J_2M^2+16L^2M}{120}h^5
 +\frac{64J_3M^3}{840}h^7.
\end{aligned}
\]

Indeed the integrals of `s*(h-s)^3/6` and `s^3*(h-s)^3/6` are
`h^5/120` and `h^7/840`. Every evolving fine coordinate was retained in
these estimates; no environmental phase or form was held fixed.
At `epsilon=0`, this is exactly Section 31's
`E_0(h)=8*h^5/15+8*h^7/105`. Thus the actual contrast

\[
F(\epsilon,h):=Y_A^\epsilon(\pi h)-Y_B^0(\pi h)
\]

obeys, for the declared preparations and any positive finite horizon,

\[
\boxed{F(\epsilon,h)\in
 [C_A^\epsilon-C_B^0-(E_\epsilon+E_0),
  C_A^\epsilon-C_B^0+(E_\epsilon+E_0)].}
\]

This is a finite bound for each complete law, not merely an instantaneous
derivative comparison or a transferred reference-law remainder.

### 32.5. Opposite endpoint signs imply an actual finite collision

At the frozen `q,h`, the exact contrast centers and radii are

| Coefficient | `C_A^epsilon-C_B^0` | `E_epsilon+E_0` |
| --- | --- | --- |
| `0` | `397/771844800` | `2801/8400000000` |
| `1/10` | `-39091977466163/19957561355531524800` | `29783749/56000000000000` |

In particular, the exact rational enclosures prove the simpler strict
sign margins

\[
F\!\left(0,\frac1{20}\right)>\frac1{10^7}>0,\qquad
F\!\left(\frac1{10},\frac1{20}\right)<-\frac1{10^6}<0.
\]

For reference, the lower bound at zero coefficient is exactly
`244346399/1350728400000000`; the upper bound at `1/10` is approximately
`-1.4269e-6`. The exact centers and radii, not that decimal display,
establish both signs. The bound `E_(1/10)=61341247/168000000000000`
uses the changed complete law; `E_0=2801/16800000000` remains the
reference budget.

By continuity of the actual endpoint with respect to the constant law
coefficient and the intermediate value theorem, there exists

\[
\boxed{\epsilon_*\in(0,1/10)\quad\text{such that}\quad
Y_A^{\epsilon_*}(\pi/20)=Y_B^0(\pi/20).}
\]

This is equality of two actual continuous-law endpoints. No assertion
that a point in overlapping outer intervals is realizable is needed: the
intervals certify strict signs at the ends of a continuous family, which
forces a zero between them. No trajectory was sampled and no value of
`epsilon_*` was estimated. The theorem asserts neither uniqueness of this
coefficient at the fixed horizon nor equality of entire output histories.

Consequently this one exact receiver reading cannot distinguish orientation
and constitutive coefficient jointly over the declared family. Adding
preparation or readout uncertainty cannot restore uniform discrimination
over a class that still includes the two colliding exact preparations.
An independently fixed law, a justified restriction of the coefficient
family, or a separately admitted additional observation is needed for that
stronger inverse claim.

### 32.6. The local collision branch and the conditional sine premise

The collision is not restricted to this particular small horizon. Since
all initial forms vanish, time reversal of either complete law gives
`x(-tau)=-x(tau)` and `theta(-tau)=theta(tau)` on continuous lifts.
Uniqueness proves these identities by substituting the reversed fields
into the original equations. The analytic contrast is therefore odd in
`h`, and `G(epsilon,h)=F(epsilon,h)/h` has an analytic even extension
through `h=0`. The receiver coefficients above yield

\[
G(\epsilon,h)=-\frac{\epsilon b^3}{2}
 +\left[\frac{a(1+3\epsilon b^2)(b+\epsilon b^3)}6
       -\frac{(2a+1)b}{24}\right]h^2+O(h^4).
\]

Here `G(0,0)=0` and `partial_epsilon G(0,0)=-b^3/2!=0`. The same law formula
supplies its analytic coefficient extension locally around zero. The
implicit function theorem gives a unique local analytic branch through zero,
even in `h`, with

\[
\boxed{\epsilon(h)=\frac{2a-1}{12b^2}h^2+O(h^4).}
\]

The fixed preparation has `b>0` and `2a-1>0`; sufficiently small positive
horizons therefore have a positive collision coefficient below `1/10`.
This local asymptotic statement does not locate or establish uniqueness
of the collision at the separately frozen horizon `h=1/20`.

The conditional [first-moment selection theorem](SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency)
remains intact. With odd circular current, requiring primitive pressure to
depend on an arbitrary neighbor list only through its degree and first
circular moment selects a multiple of sine; the consensus normalization
fixes that multiple. Every `epsilon>0` member used here violates that
additional information premise. If it is independently required, those
members are excluded and the fixed-sine receiver result retains its scope.

The present collision shows that this one endpoint, the stationary
cancellation control and the shared consensus tangent do not independently
justify selecting sine from the declared larger family. It neither
contradicts the conditional selection theorem nor disproves sine as a
candidate law. The constructive ambiguity is a mathematical result about
supplied complete laws and a named readout, not a physical calibration,
a fitted pressure, a selected fundamental law or a constituent identity.

The shared `assess_sine_pair_receiver_confounding(phase_rotation=...,
horizon_tau=..., epsilon_upper=...)` in the
[pair owner](../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairReceiverConfounding`. It reconstructs primitive preparations and
evaluates the two complete-law centers and their separate remainders;
it does not transfer sine-derived rates or storage to the cubic member.
The same sufficient bounds apply to any declared finite positive upper
coefficient and horizon. The report requires strict opposite signs after
outward interval materialization and otherwise leaves collision existence
unavailable; it supplies no coefficient estimate. The
[contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-constitutive-confounding),
[usage](../../docs/guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-constitutive-confounding)
and [independent fine-law controls](../../tests/physics/test_sine_pair_receiver_confounding.py)
retain this scope, own-law storage and the distinction between an actual
existence theorem and a numerical trajectory or observed collision.

<a id="sine-pair-receiver-two-time"></a>
## 33. Two readings with one constant constitutive coefficient

### 33.1. Prospective observation and stopping rule

This admission retains Section 32's complete A-epsilon versus B-zero law
family, ten ordered nodes, doubled-C5 support, unit held capacities, zero
loss, unforced continuous rows and clock `tau=t/pi`. The nominal full forms
and phasors are unchanged. Each branch is one realization: its coefficient
and initial state are shared by both readings. B retains epsilon zero;
A has one unknown constant coefficient in the declared interval.

The following protocol is fixed before evaluating this two-time certificate:

| Item | Declared value |
| --- | --- |
| Exact nominal rotation | `q=(399/401,40/401)` |
| Coefficient family for A | `0<=epsilon<=1/10` |
| Ordered observation times | `horizons_tau=(1/40,1/20)`, original times `t=pi*h` |
| Sole receiver | `Y=(x_2+x_3)/2` at both times |
| Independent all-node form preparation radius | `rho_x=1/10^11` |
| Independent all-node phase preparation radius | `rho_theta=1/10^11` radians about nominal lifts |
| Absolute error per recorded scalar | `eta=1/10^11` |
| Coefficient and clock error | No time variation or additional clock uncertainty |
| Arithmetic budget | Exact rational formulas and shared outward dyadic128 endpoint admission |
| Method | Cubic full-law Taylor bounds, convex coefficient chords for remainders/preparation errors, and two necessary coefficient intervals |
| Stopping rule | Strict disjointness of the two outward necessary coefficient intervals, or an explicit unavailable result |

The previously evaluated endpoint at `tau=1/20` is retained prior evidence,
not a fresh reserved response. The additional reading at `tau=1/40` and
the joint uncertain prediction are the new observation. No measured record,
trajectory integration, parameter fit, coefficient search or horizon search
enters this protocol. Do not change these values to improve its outcome.

Each actual preparation may perturb every fine form and phase independently
within its budget, including the common form origin and the environment.
It need not remain antipodal, replicated or on the nominal storage level.
Its one perturbed initial state evolves through both observations under its
own complete law and storage. Readout errors may be arbitrarily correlated
within the declared componentwise bounds. Endpoint enclosures will relax
some correlations conservatively; they must retain the same epsilon at
both times. Separate marginal overlaps are not a joint collision proof.

The question is restricted to this A-family versus B-reference comparison.
It is neither recovery of arbitrary internal state nor identification of
two independently unknown law coefficients, and it does not select a
physical fundamental law.

### 33.2. Complete-law preparation errors across both readings

Write `bar epsilon` for the declared upper coefficient. The sufficient
domain used below is `a>1/2`, `b>0`, `bar epsilon>0`, `0<h_0<h_1`, nonnegative error budgets
and `4*(1+3*bar epsilon)*h_1^2<1`, with the exact unit rotation `q=a+ib`.
Every member retains Section 32's complete fine rows and its own storage.
For a perturbed preparation, `|x_i(tau)|<=rho_x+(1+epsilon)*|tau|` extends
the finite-time continuation argument to the whole admitted initial set.

Compare each perturbed trajectory with its own nominal trajectory under
the same coefficient. Choose initial phase lifts realizing the circular
error bounds and continue these lifts under the full phase row, as in
Section 31.3. Let `u_x` and `u_theta` denote their respective maximum
all-node differences. The global derivative bound
`|j_epsilon'|<=L_epsilon=1+3*epsilon` gives

\[
u_x(\tau)\le\rho_x+2L_\epsilon\int_0^\tau u_\theta(s)\,ds,
\qquad
u_\theta(\tau)\le\rho_\theta+2\int_0^\tau u_x(s)\,ds.
\]

The two factors of two retain both ends of each phase or form difference
in the degree-four averages. Comparison with the nonnegative matrix
`[[0,2*L_epsilon],[2,0]]` and its exponential series yields

\[
u_x(h)\le\rho_x\cosh(2\sqrt{L_\epsilon}h)
 +\sqrt{L_\epsilon}\rho_\theta\sinh(2\sqrt{L_\epsilon}h)
\le
P_\epsilon(h):=
\frac{\rho_x+2L_\epsilon h\rho_\theta}{1-4L_\epsilon h^2}.
\]

The last inequality bounds the even and odd factorial coefficients by
one and sums the resulting geometric series. Its denominator is positive
throughout the admitted coefficient and time ranges. This denominator
limits the rational comparison bound; it does not define a stability or
existence threshold of the complete law.

The receiver mean has error at most `P_epsilon(h)`. This comparison allows
independent perturbations on all ten forms and phases. The common form
origin is bounded by the form budget; its uncertainty is retained. The
nominal zero mean, storage level, antipodality and replicated environmental
phases are not imposed on perturbed states. At both readings the same
initial state and law generate the errors covered by these bounds.

Let `y_A,h^epsilon` and `y_B,h^0` be any two recorded values from the
admitted A and B preparations at time `pi*h`, including their scalar
readout errors. Sections 31--32 therefore give

\[
\left|y_{A,h}^\epsilon-y_{B,h}^0
 -(C_A^\epsilon(q,h)-C_B^0(q,h))\right|
\le R_\epsilon(h),
\]

where

\[
R_\epsilon(h)=E_\epsilon(h)+E_0(h)
 +P_\epsilon(h)+P_0(h)+2\eta.
\]

The two Taylor remainders apply to the nominal complete trajectories.
The separate preparation comparison covers departures from those nominal
trajectories; in particular, no odd-in-time Taylor expansion is assumed
for perturbed nonzero initial forms. Readout errors may be correlated
between times or branches. The displayed sum follows from componentwise
bounds and requires no statistical independence.

### 33.3. Convex coefficient bounds and necessary intervals

For fixed positive `h`, Section 32's `E_epsilon(h)` is a polynomial in
`epsilon` with nonnegative coefficients. Its factors `M=1+epsilon`,
`L=1+3*epsilon`, `J_2=1+9*epsilon`, and `J_3=1+27*epsilon` make this explicit
in the formula in Section 32.4. Thus it is nondecreasing and convex for
nonnegative epsilon. The preparation bound is also nondecreasing and
convex on the admitted domain, since

\[
\frac{\partial P_\epsilon(h)}{\partial\epsilon}
=\frac{6h\rho_\theta+12h^2\rho_x}{(1-4L_\epsilon h^2)^2},
\qquad
\frac{\partial^2P_\epsilon(h)}{\partial\epsilon^2}
=\frac{144h^3\rho_\theta+288h^4\rho_x}
 {(1-4L_\epsilon h^2)^3}\ge0.
\]

Consequently the chord joining coefficient endpoints bounds both errors
from above. Define exact nonnegative quantities

\[
R_h^0=2(E_0(h)+P_0(h)+\eta),\qquad
Q_h=\frac{E_{\bar\epsilon}(h)-E_0(h)
              +P_{\bar\epsilon}(h)-P_0(h)}{\bar\epsilon}.
\]

Then `R_epsilon(h)<=R_h^0+Q_h*epsilon` for every
`epsilon` in `[0,bar epsilon]`. In particular, using only the worst
endpoint radius independently of epsilon would discard information that
this prospective chord retains.

The complete-law cubic contrast is

\[
C_A^\epsilon(q,h)-C_B^0(q,h)
=-c\epsilon h+(D_0+D_1\epsilon+D_2\epsilon^2)h^3,
\]

with

\[
c=\frac{b^3}{2},\qquad
D_0=\frac{b(2a-1)}{24},\qquad
D_1=\frac{2ab^3}{3},\qquad D_2=\frac{ab^5}{2}.
\]

All four are positive on the stated rotation domain. The common nominal
linear term `-b*h/2` cancels in this contrast, while the changed law's
linear contribution `-c*epsilon*h` remains. Dropping the nonnegative
quadratic term gives a lower bound; using
`epsilon^2<=bar epsilon*epsilon` gives an upper bound. For the actual
recorded contrast `F_h=y_A,h^epsilon-y_B,h^0`, put

\[
\begin{aligned}
N_h&=D_0h^3-R_h^0,& A_h&=ch-D_1h^3+Q_h,\\
U_h&=D_0h^3+R_h^0,& B_h&=ch-(D_1+D_2\bar\epsilon)h^3-Q_h.
\end{aligned}
\]

The resulting full prediction is

\[
\boxed{N_h-A_h\epsilon\le F_h\le U_h-B_h\epsilon.}
\]

Whenever both slopes are strictly positive, equality of the two recorded
values requires

\[
\epsilon\in K_h:=\left[\frac{N_h}{A_h},\frac{U_h}{B_h}\right]
\quad\hbox{and}\quad \epsilon\in[0,\bar\epsilon].
\]

Write `I_h` for the reported outward enclosure of `K_h` on the dyadic128
grid. It is not clipped to the declared coefficient domain. A common two-reading
record requires the same epsilon in both necessary intervals. Their
strict disjointness therefore excludes such a record. Overlap or touching
of these outer intervals leaves the test unavailable and does not produce
a fitted coefficient or a realizable joint collision. Nonpositive slopes
also leave this particular interval test unavailable.

Each endpoint inequality holds for every admitted full trajectory, so
intersecting its necessary coefficient conditions is valid even though
the construction relaxes correlations between the two endpoint errors.
It enlarges the possible recorded sets. Excluding an intersection of
these enlarged sets is sufficient to exclude an intersection of the actual
sets; passing both marginal conditions would not establish the converse.

### 33.4. Exact evaluation of the frozen protocol

For Section 33.1's declared values, the preparation denominators at
`epsilon=1/10` are respectively `3987/4000` and `987/1000`. The exact
remainder and preparation quantities entering the chords are

| `h` | `E_0(h)` | `E_(1/10)(h)` | `P_0(h)` | `P_(1/10)(h)` |
| --- | --- | --- | --- | --- |
| `1/40` | `11201/2150400000000` | `245217247/21504000000000000` | `1/95000000000` | `71/6645000000000` |
| `1/20` | `2801/16800000000` | `61341247/168000000000000` | `1/90000000000` | `113/9870000000000` |

Substitution gives strictly positive `A_h` and `B_h` at both times.
The outward necessary intervals have the following integer endpoint
numerators, each divided by exactly `2^128`:

| `h` | Lower numerator | Upper numerator |
| --- | --- | --- |
| `1/40` | `1470434397896162650815225270546171032` | `2062269644807637613095454024944732170` |
| `1/20` | `2303660690491471484582193649471228535` | `12683274044151222542666137470396035794` |

For orientation only, these are approximately
`[0.00432121832,0.00606046579]` and
`[0.00676985032,0.03727279247]`. Their exact outward separation is

\[
\boxed{
\inf I_{1/20}-\sup I_{1/40}
=\frac{241391045683833871486739624526496365}{2^{128}}
>\frac7{10000}>0.}
\]

The integer endpoints, rather than their decimal displays, establish the
strict stopping rule. Thus no single constant coefficient in `[0,1/10]`
and no admitted pair of perturbed preparations can make both recorded
receiver values agree between A and B. All three positive error budgets
from Section 33.1 are already included in this statement.

### 33.5. A separating readout margin and its scope

The same affine inequalities also give a separation in form units.
Write `s=1/40` and `l=1/20`. Combining the later lower bound with the
earlier upper bound cancels their common epsilon:

\[
\frac{B_sF_l-A_lF_s}{A_l+B_s}
\ge\frac{B_sN_l-A_lU_s}{A_l+B_s}=:m>0.
\]

The two exact observation weights, in increasing-time order, are

\[
(w_s,w_l)=
\frac{(-283478023070798177716294315962752,
        130882803955284792029973526868541)}
     {414360827026082969746267842831293}.
\]

They satisfy `|w_s|+|w_l|=1`. For every pair of admitted records,
`w_s*(y_A,s-y_B,s)+w_l*(y_A,l-y_B,l)>=m`. This is a consequence of the
already declared affine bounds, and supplies no additional observation,
trajectory approximation or coefficient fit. The exact margin is

\[
m=\frac{418793386092522070667871120001294269449797}
        {69961117113952225999078096164870482977650000000000}.
\]

Its outward dyadic lower bound is

\[
\boxed{
\underline m=
\frac{2036960107973714942770656040905}{2^{128}}
>\frac5{10^9}.}
\]

The exact margin is approximately `5.98608775e-9` in the declared form
units. The unit sum of absolute weights also gives a lower bound
`underline m` on the maximum componentwise distance between any A-family
and B-reference recorded pair. It includes preparation and measurement
uncertainty; errors at the two times need not be independent.

The shared `assess_sine_pair_receiver_two_time(phase_rotation=...,
horizons_tau=..., epsilon_upper=..., form_error_bound=...,
phase_error_bound=..., readout_error_bound=...)` in the
[pair owner](../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairReceiverTwoTime`. It re-admits primitive inputs and rebuilds the
nominal phasors, complete-law remainders, preparation bounds, chords and
affine coefficients. `necessary_coefficient_bounds` and
`coefficient_gap_lower_bound` retain the outward interval decision;
`joint_readout_weights` and `joint_readout_gap_lower_bound` retain the
derived separating observable. The status is `certified_disjoint` for
strict coefficient-interval separation and `unavailable` otherwise.
For other admitted inputs, an extremely small positive exact readout
margin can round down to zero; the optional displayed readout margin
does not replace the strict coefficient-interval decision.

The [contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-two-time),
[usage](../../docs/guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-two-time)
and [independent controls](../../tests/physics/test_sine_pair_receiver_two_time.py)
retain the admission, executable example and verification boundaries.

The two readings separate the stated A-family and fixed-sine B-reference
while preserving Section 32's one-time collision. The extra reading
restricts one unchanged law coefficient through both observations.
Interval membership still gives only necessary compatibility. No actual
record has been supplied, no arbitrary internal state has been recovered,
and no conclusion is obtained about two independently unknown coefficients,
formation, maintenance or physical selection of sine. The support,
capacity, clock, reference law and bounded common form origin remain
explicit premises of this conditional separation.

<a id="sine-pair-receiver-two-law"></a>
## 34. Two receiver readings with neither constitutive coefficient fixed

### 34.1. Prospective comparison and stopping rule

Retain Section 33's nominal full preparations, doubled-C5 support, ordered
nodes, unit held capacities, receiver, exact clock, zero loss and absence of
inputs or events. Now both hypotheses use Section 32's complete supplied
family `j_epsilon(delta)=sin(delta)+epsilon*sin(delta)^3`: preparation A
has coefficient `alpha` and preparation B has coefficient `beta`. These
coefficients may differ between the hypotheses. Each coefficient and each
perturbed initial state remain constant/shared across that branch's two
readings; no coefficient may be refitted separately at each time.

Freeze the following protocol before evaluating its joint prediction:

| Item | Declared value |
| --- | --- |
| Exact nominal rotation | `q=(399/401,40/401)` |
| Independent coefficient domains | `0<=alpha<=1/10`, `0<=beta<=1/10` |
| Ordered structural times | `horizons_tau=(1/40,1/20)`, with original times `t=pi*h` |
| Receiver at both times | `Y=(x_2+x_3)/2` |
| All-node form and initial phase-lift error radii | `rho_x=rho_theta=1/10^11` |
| Absolute error per scalar recorded value | `eta=1/10^11` |
| Joint observable | `Z=(-h_1*Y(h_0)+h_0*Y(h_1))/(h_0+h_1)`, weights `(-2/3,1/3)` |
| Arithmetic | Exact rational formulas and shared outward dyadic128 intervals |
| Analytic budget | Complete-law cubic coefficients; exact quadratic extrema over each coefficient interval; cubic Taylor remainder bounds using kernel derivatives and nominal initial-rate bounds; coupled preparation-error majorants |
| Stopping rule | Strict positive outward lower bound for every admitted `Z_A-Z_B`, or an explicit unavailable result |

The joint observable cancels a constant linear-in-time receiver contribution
separately within either hypothesis. Its weights depend only on the retained
times and are not fitted to any response. Full-law remainders, all-node
preparation errors and componentwise readout errors remain in its prediction.
Errors may be correlated across time; bounding them separately enlarges the
admitted record sets conservatively. Every actual trajectory retains its
own law-specific storage, without requiring equal storage across hypotheses.

Both times and earlier restricted-family results are already evaluated prior
information. The new prediction concerns the enlarged two-coefficient family;
it is not new measured data. No new horizon, preparation, coefficient range,
error budget, trajectory integration, parameter search or additional reading
is admitted. A failed sufficient bound is not proof of a realizable collision.
General law identification, unknown clocks, time-varying coefficients and
physical measurement bridges remain outside this comparison.

### 34.2. Complete-law coefficients for either orientation

For a generic constant coefficient `e>=0`, put

\[
g_e=b+eb^3,\qquad d_e=1-a+e(1-a^3).
\]

Here `e` is the cubic-current coefficient; the phase-row coupling remains
fixed at one. In A the selected fine-node initial form rates are
`(g_e/2,-g_e/2)`. In B they are `(d_e/2,-d_e/2)`: each selected node
receives two currents from the `q` pair and two from the `-1` pair.
In either preparation, both receiver nodes have rate `-g_e/2`, both
members of pair 2 have rate `g_e/2`, and the remaining nodes have zero
rate. All initial forms and phase velocities vanish.

The receiver's initial phase acceleration is therefore `-3*g_e/4`.
In B the selected nodes have accelerations `d_e/2+g_e/4` and
`-d_e/2+g_e/4`. The receiver's current derivatives toward those nodes
are `b*(1+3*e*a^2)` and its negative; toward the two nodes of pair 2
they are both `a*(1+3*e*b^2)`. Substitution into the complete fine rows
gives, for either orientation `o`,

\[
Y_o^e(h)=-\frac{g_eh}{2}+k_o(e)h^3+r_o^e(h),
\]
\[
k_A(e)=\frac{a(1+3eb^2)g_e}{6},\qquad
k_B(e)=\frac{b(1+3ea^2)d_e+3a(1+3eb^2)g_e}{24}.
\]

These are cubic Taylor coefficients, so each includes the factor `1/6`
from the third derivative. The B formula reduces at `e=0` to Section 31's
coefficient. It has been derived from the B cubic law, without importing
a sine-only response into that law.

For an exact unit `q=a+ib` with `a>1/2,b>0`, one has `0<a<1`. Every
coefficient of both displayed quadratic polynomials in `e` is positive.
Consequently their exact ranges on `[0,u]`, for `u>0`, are
`[k_A(0),k_A(u)]` and `[k_B(0),k_B(u)]`. The uniform maximum initial
fine-node rate for each family is

\[
m_A=\frac{g_u}{2},\qquad m_B=\frac{\max(g_u,d_u)}2.
\]

Both `g_e` and `d_e` increase on this interval. The maximum retains all
fine nodes: for other admitted rotations the selected B pair can dominate
the environmental rates. Section 32.2's continuation, odd-current form
balance and own-law storage apply to each coefficient and either orientation.
Perturbed preparations need not have the nominal storage or antipodality.

### 34.3. Remainders from nominal initial rates

Use uniform kernel bounds
`L=1+3*u`, `J_2=1+9*u`, and `J_3=1+27*u`. Fix a nominal trajectory,
a positive horizon `h`, and its family's initial-rate bound `m`.
The finite-time continuation already established makes

\[
X_*:=\sup_{0<s\le h}\frac{\max_i|x_i(s)|}{s}
\]

finite; the quotient extends continuously at zero through the initial
derivative. The full phase row gives `|theta_i'(s)|<=2*X_*s`.
Thus any continuously lifted edge gap changes by at most `2*X_*s^2`.
The global Lipschitz bound on the current yields
`|x_i'(s)-x_i'(0)|<=2*L*X_*s^2`. Integrating and taking the supremum gives

\[
X_*\le m+\frac23Lh^2X_*.
\]

Whenever `1-(2/3)*L*h^2>0`, define

\[
X_h=\frac{m}{1-\frac23Lh^2},\qquad
V_h=m+2LX_hh^2.
\]

Then `|x_i(s)|<=X_h*s` and `|x_i'(s)|<=V_h` throughout `[0,h]`.
These bounds follow the evolving phase and pressure; they do not hold the
initial pressure fixed. For every edge gap `delta`, the complete rows give

\[
|\delta'(s)|\le4X_hs,\qquad
|\delta''(s)|\le4V_h,\qquad
|\delta'''(s)|\le16LX_hs.
\]

The receiver form derivative averages eight edge currents with total
absolute weight one. Differentiating each current three times therefore gives

\[
|Y^{(4)}(s)|\le
(48J_2X_hV_h+16L^2X_h)s+64J_3X_h^3s^3.
\]

The integral Taylor remainder after degree three is bounded by

\[
|r_o^e(h)|\le\widehat E_o(h):=
\frac{48J_2X_hV_h+16L^2X_h}{120}h^5
+\frac{64J_3X_h^3}{840}h^7,
\]

using `m=m_o` separately for the two orientation families. This is uniform
over every coefficient in `[0,u]`. It uses nominal zero-form trajectories;
it is not applied directly to perturbed nonzero initial forms.

Section 33.2's separate form/phase comparison supplies the uniform
preparation error

\[
P_u(h)=\frac{\rho_x+2Lh\rho_\theta}{1-4Lh^2}.
\]

Choose the initial lifts realizing the circular error bounds, and evolve
them under the full phase row. The comparison covers all nodes and the
absolute form origin. Each perturbed path is compared with its own nominal
path under the same coefficient. The admitted domain
`4*L*h_1^2<1` ensures both rational denominators are positive at both times;
it limits these majorants, not existence or physical stability.

### 34.4. Uniform separation of the joint records

Write `y_o(h)` for an admitted recorded receiver value. The fixed weights
`(w_0,w_1)=(-h_1,h_0)/(h_0+h_1)` satisfy

\[
\sum_jw_jh_j=0,\qquad \sum_j|w_j|=1,\qquad
\sum_jw_jh_j^3=h_0h_1(h_1-h_0)=:F>0.
\]

Thus `Z_o=sum_j w_j*y_o(h_j)` has nominal center `F*k_o(e)` and error at most

\[
B_o=\sum_{j=0}^1|w_j|
       [\widehat E_o(h_j)+P_u(h_j)+\eta].
\]

This bound permits arbitrary correlations between the errors. It retains one
coefficient and one initial state per branch across time. The coefficients
of different hypotheses remain independent. For every `alpha,beta` in
`[0,u]` and every admitted pair of recorded trajectories,

\[
\boxed{
F[k_A(0)-k_B(u)]-B_A-B_B
\le Z_A-Z_B\le
F[k_A(u)-k_B(0)]+B_A+B_B.}
\]

The extrema use opposite corners of the two-coefficient square. Constraining
`alpha=beta`, or allowing a different coefficient at each reading, would
change the comparison. The bound is a forward statement for every admitted
record; it does not fit a coefficient or consume a measured response.

For the frozen inputs of Section 34.1, the exact cubic endpoint ranges are

\[
k_A([0,u])=\left[\frac{2660}{160801},
 \frac{69053469769060}{4157825282402401}\right],
\qquad
k_B([0,u])=\left[\frac{5995}{482403},
 \frac{1558057852689593}{124734758472072030}\right].
\]

Both initial-rate bounds are `3219220/64481201`, the weights are
`(-2/3,1/3)`, and `F=1/32000`. Substitution into the complete error formulas
above and outward dyadic128 materialization gives

\[
\boxed{
Z_A-Z_B\in
\left[
\frac{42072687116468740883070817825397}{2^{128}},
\frac{45464133054438654008708268970258}{2^{128}}
\right].}
\]

The lower endpoint is strictly greater than `1/10^7` in form units
(approximately `1.236405150733066e-7`). All three positive error budgets are
already included. This meets the frozen stopping rule: no pair of admitted
orientations, independent constant coefficients and perturbed preparations
can produce the same two-reading record. Since the weights have absolute
sum one, the lower bound also bounds the maximum componentwise distance
between the two recorded pairs from below.

### 34.5. Shared certificate and retained boundaries

`assess_sine_pair_receiver_two_law` in the
[pair owner](../../src/tnfr/physics/relational_sine_pair.py) returns
`SinePairReceiverTwoLaw`. It re-admits primitive inputs and rebuilds the
phasors, complete-law coefficients, exact endpoint extrema, initial-rate
bounds and full error budget. `joint_difference_bounds` retains the outward
interval above; status is `certified_disjoint` only for a strictly positive
lower endpoint, and `unavailable` otherwise. An exact positive margin smaller
than the outward grid may become unavailable. A failed sufficient bound
does not prove an actual collision.

The [contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-two-law),
[usage](../../docs/guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-two-law)
and [independent controls](../../tests/physics/test_sine_pair_receiver_two_law.py)
own admission, export, executable usage and implementation checks. The
`phase_exchange_beta` report field denotes the held unit phase-row coupling,
separately from either unknown cubic-current coefficient.

The sharper nominal-state remainder belongs to this new comparison. It does
not replace earlier frozen bounds or change the earlier one-time collision
and restricted two-time results. No coefficient value or physical law is
identified. Exact support, capacity, clock and the supplied constitutive
family remain premises. The absolute form origin is bounded rather than
eliminated: the weights do not sum to zero. Arbitrary hidden-state recovery,
time-varying laws, formation and physical measurement admission require their
own arguments.

<a id="sine-pair-receiver-defect"></a>
## 35. Receiver discrimination with bounded continuous model defects

### 35.1. Prospective defect family and stopping rule

Retain Section 34's full nominal preparations, independent constant
coefficients `alpha,beta` in `[0,1/10]`, doubled-C5 support, held unit
capacities, exact clock `tau=t/pi`, two receiver times and fixed readout
weights. Now admit complete-row departures on every fine node:

\[
x_i'=\frac14\sum_{j\sim i}j_e(\theta_j-\theta_i)+r_{x,i},\qquad
\theta_i'=x_i-\frac14\sum_{j\sim i}x_j+r_{\theta,i}.
\]

For each hypothesis, one coefficient, one initial state and one pair of
residual histories govern both readings. The residuals may differ between
hypotheses. They are measurable, essentially bounded histories on the full
window; the rows hold almost everywhere along absolutely continuous paths.
No statistical independence or differentiability of the residuals is assumed.
This is a declared family of complete-row departures, not an inferred forcing
mechanism or an autonomous selector of the residual histories.

Freeze the following inputs before evaluating the expanded prediction:

| Item | Declared value |
| --- | --- |
| Nominal preparation, law domain and observation | Unchanged Section 34.1 protocol, `q=(399/401,40/401)` |
| Ordered times and weights | `(1/40,1/20)` in `tau`, weights `(-2/3,1/3)` |
| Form, phase and readout budgets | `rho_x=rho_theta=eta=1/10^11`, unchanged |
| All-node form-rate defect bound | `delta_x=1/10^6` in form units per unit `tau` |
| All-node phase-rate defect bound | `delta_theta=1/10^6` radians per unit `tau` |
| Residual scope | Bounds hold almost everywhere on the entire `[0,1/20]` window, independently for both hypotheses |
| Method | Same-law coupled error comparison; integrated matrix-exponential series bounded by rational majorants; unchanged nominal cubic certificate |
| Arithmetic | Exact rational formulas and shared outward dyadic128 intervals |
| Stopping rule | Strict positive outward lower bound for every admitted `Z_A-Z_B`, or explicit unavailability |

The new budgets are mathematical premises, not apparatus tolerances or bounds
derived from a measured response. Support, capacities, coefficient constancy
and the structural clock remain exact. The residuals are distinct from
preparation/readout errors and need not conserve total form or own-law
storage. No support events or state resets occur. No numerical integration,
parameter fit, new reading or retuning of the retained protocol is admitted.
The earlier certificates remain evaluated prior evidence. Applying this
result to a physical source requires independently justified mappings and
residual bounds throughout the window; none are supplied by this protocol.

### 35.2. Residual balances and complete-state comparison

For a fixed branch coefficient `e`, write
`S_e,i=sum_j j_e(theta_j-theta_i)` and let `L_f` be the fine combinatorial
Laplacian. The declared rows are `x'=S_e/4+r_x` and
`theta'=L_f*x/4+r_theta`. The own-law storage of Section 32.2 has gradients
`(L_f*x,-S_e)`. Its balance and the total-form balance become

\[
\boxed{
H_e'=(L_fx)\mathbin{\cdot}r_x-S_e\mathbin{\cdot}r_\theta,\qquad
\left(\sum_i x_i\right)'=\sum_i r_{x,i}}
\]

almost everywhere. Integration accounts for the supplied residual work.
Neither balance has a prescribed sign or vanishes for arbitrary admitted
histories. Exact conservation of the reference law and stationarity of its
nominal controls therefore do not transfer to this enlarged family.

For prescribed essentially bounded histories, the smooth reference field
with additive inputs admits the ordinary absolutely continuous evolution.
The bound `|x_i(tau)|<=|x_i(0)|+(1+u+delta_x)*tau`, with `u=epsilon_upper`,
prevents finite-time form escape; the phase row then has finite growth.
The following comparison also applies to any admitted absolutely continuous
path satisfying the declared rows almost everywhere. It does not select a
residual history from the state or assert uniqueness for an unspecified
feedback rule.

Compare each such path with its own zero-defect nominal preparation under
the same coefficient. Choose initial phase lifts within the admitted
phase-error budget and continue them through the full phase rows. For the
maximum all-node form and lifted-phase differences, the global kernel bound
`L=1+3*u` gives

\[
u_x(h)\le\rho_x+\delta_xh+2L\int_0^h u_\theta(s)\,ds,\qquad
u_\theta(h)\le\rho_\theta+\delta_\theta h+2\int_0^h u_x(s)\,ds.
\]

The nonnegative comparison matrix is

\[
A=\begin{pmatrix}0&2L\\2&0\end{pmatrix},\qquad A^2=4LI.
\]

Variation of constants bounds the two differences by the corresponding
components of `exp(Ah)*(rho_x,rho_theta)+int_0^h exp(As)*(delta_x,delta_theta) ds`.
The initial-error term retains Section 33.2's bound `P_u(h)`. The form
component of the integrated residual term is

\[
\delta_x\sum_{k\ge0}\frac{(4L)^kh^{2k+1}}{(2k+1)!}
+2L\delta_\theta\sum_{k\ge0}\frac{(4L)^kh^{2k+2}}{(2k+2)!}.
\]

Using `1/(2k+1)!<=1` and `2/(2k+2)!<=1`, the admitted domain
`4*L*h^2<1` gives the separate defect majorant

\[
\boxed{D_u(h)=\frac{h\delta_x+Lh^2\delta_\theta}{1-4Lh^2},
\qquad u_x(h)\le P_u(h)+D_u(h).}
\]

The direct form defect enters at first order in time; the phase defect
reaches form through the coupled law at second order. These are uniform
whole-window bounds, including the environment and common form origin.
They require neither residual derivatives nor uncorrelated errors. The
nominal Taylor remainder is still applied only to the nominal smooth
reference path, not to a defective path that may be merely absolutely
continuous. Under `t=pi*tau`, every evolution row, including its residual,
acquires the same factor `1/pi`; the frozen rate budgets are in `tau` units.

### 35.3. Joint recorded separation and frozen evaluation

Keep Section 34's weights and exact cubic factor. If `B_o` is its reference
error radius for orientation `o`, the defect comparison adds

\[
W_\delta=\sum_{j=0}^1|w_j|D_u(h_j)
\]

to each branch. The hypotheses may choose different coefficients, initial
errors and residual histories; each choice persists across its two readings.
Consequently every admitted pair of recorded trajectories satisfies

\[
\boxed{
F[k_A(0)-k_B(u)]-B_A-B_B-2W_\delta
\le Z_A-Z_B\le
F[k_A(u)-k_B(0)]+B_A+B_B+2W_\delta.}
\]

The fixed weights cancel each nominal linear term. They need not cancel
residual-driven changes. For example, a common all-node form input
`r_x(s)=delta_x*s/h_1` adds `delta_x*h^2/(2*h_1)` to every form while leaving
phase differences unchanged. Its weighted receiver contribution is nonzero
because `sum_j w_j*h_j^2>0`. This exact control lies within the declared
whole-window bound and illustrates why residual errors must be propagated
rather than removed by a linear-cancellation argument.

For the frozen budgets,

\[
D_u(1/40)=\frac{413}{15948000000},\qquad
D_u(1/20)=\frac{71}{1316000000},\qquad
W_\delta=\frac{554831}{15740676000000}.
\]

Combining these exact values with the rebuilt Section 34 quantities gives
the outward interval

\[
\boxed{
Z_A-Z_B\in
\left[
\frac{18083983464718179646457142934182}{2^{128}},
\frac{69452836706189215245321943861473}{2^{128}}
\right].}
\]

The lower endpoint is greater than `1/20000000` in form units
(approximately `5.3144051007849695e-8`). Thus the same recorded pair cannot
belong to both admitted orientation families, even with the frozen
continuous-row departures and all original preparation/readout errors.
The unit sum of absolute weights also gives this lower bound on the maximum
componentwise distance between any pair of competing records.

### 35.4. Execution and physical-admission boundary

`assess_sine_pair_receiver_defect` in the
[pair owner](../../src/tnfr/physics/relational_sine_pair.py) takes the original
two-law primitives and separate `form_rate_defect_bound` and
`phase_rate_defect_bound`. It returns `SinePairReceiverDefect`, rebuilding
`reference_certificate` internally rather than accepting an edited report.
The exact reference centers and radii are combined with the new defect
radius before one final outward interval materialization. Zero defect
budgets reproduce the previous interval and verdict exactly.

The reference certificate documents the conservative comparison; it does not
certify conservation of the new defective paths. The new report retains
separate response factors, propagated defect errors and joint radii. Its
status is `certified_disjoint` only when the new outward lower endpoint is
strictly positive. Otherwise it is `unavailable`; neither overlapping bounds
nor a sub-grid positive exact margin establish an actual collision.

The [contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-defect),
[usage](../../docs/guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-receiver-defect)
and [independent controls](../../tests/physics/test_sine_pair_receiver_defect.py)
own scalar admission, export, executable usage and checks. This is detached
conditional robustness evidence; it installs no runtime source or controller.
The [source-matching boundary](../research/PHASE_AMPLITUDE_MEASUREMENT_PROTOCOL.md#primitive-phase-information-boundary)
still requires independently established mappings, hidden-state treatment,
residual bounds, preparation, sensor and clock models. The declared numerical
budgets supply none of those physical identifications. Arbitrary coefficient
variation or clock error is covered only if separately justified bounds place
the resulting complete-row departures inside the stated family.

<a id="sine-pair-persistent-response"></a>
## 36. Internal storage allocation and interaction within a persistent pattern

### 36.1. Prospective joint claim and frozen protocol

This gate asks whether two preparations with identical nominal collective
means and total storage, but different internal form/phase allocation, have
separated finite receiver responses while their complete trajectories retain
the same geometric identity. The supplied law and support are unchanged.
The existing [persistent family](SINE_REPLICA_PULSE.md#sine-replica-joint-persistence)
and [moving interface](#sine-moving-pattern-interface) supply the separate
ingredients; neither alone proves this joint response claim.

Freeze the following before evaluating its response bounds:

- Nodes are `0,...,9`, with ordered pairs `(2*a,2*a+1)` for `a=0,...,4`.
  Adjacent pairs on the oriented C5 have all four unit cross edges; there
  are no internal pair edges. Every fine degree is four.
- Held capacities, exchange weight and storage scale are one; form loss is
  zero. In `tau=t/pi`, every fine row is
  `x_i'=sum_j sin(theta_j-theta_i)/4`,
  `theta_i'=sum_j (x_i-x_j)/4`, summing over the four neighbors.
  All twenty coordinates evolve; no inputs, resets or support events occur.
- Put `alpha=2*pi/5`, `c=cos(alpha)`, `s=sin(alpha)`. The exact target has
  zero form and phases `a*alpha` in both members of pair `a`. Mathematical
  pi is symbolic, not a rounded source phase. The phase lift is centered
  on this winding-one target; distances and readouts use that declared chart.
- For each `delta` in `[3/100,1/25]`, preparation A changes only pair 0's
  phases to `(+delta,-delta)`. Preparation B instead changes only pair 0's
  forms to `(+u,-u)`, where `u=sqrt(2*c*(1-cos(delta)))`; its phases retain
  the target. Their nominal pair mean forms and midpoint phases agree.
  Their nominal excess storage is the same `8*c*(1-cos(delta))` for the
  same delta. Independent perturbed preparations need not have equal storage.
- Each branch admits independent componentwise initial errors
  `abs(e_x,i)<=1/1000000` and `abs(e_theta,i)<=1/1000000` on every fine node.
  The interiors are full-dimensional open preparation sets; certification
  will cover the closed boxes as well. No tip or symmetry restriction is
  imposed on those perturbations.
- The fixed readout is `Y=(x_2+x_3)/2` at `h=1/2` in tau, with an
  independent additive readout error of magnitude at most `1/1000000`
  per branch. No initial-value subtraction or fitted pressure is used.
- Geometric identity means retention of the acute winding-one doubled-C5
  chart around the target. Use full-state relative radius `r=1/8` and the
  existing conserved-storage first-exit barrier. Require strictly positive
  initial-radius, acute-radius and storage-barrier margins for both boxes.
- The predicted recorded contrast is `Y_A(h)-Y_B(h)>1/100000` uniformly
  over the stated delta interval, preparation errors and readout errors.
  The branches may even choose different deltas within this interval;
  exact nominal energy matching concerns the same-delta subcomparison.
  A positive response without both persistence admissions does not pass.
- Evaluate analytic bounds with the shared rational outward dyadic128
  interval arithmetic and Machin trigonometry. No trajectory search,
  sampling campaign, numerical integrator, seed or alternate precision is
  part of this gate. Record the exact bound endpoints and margins once.
  Close with the joint certificate or the precise failed obligation, without
  changing the horizon, preparations, observation or budget after evaluation.

The zero-amplitude state is an exact stationary control. Swapping the two
members of pair 0 reverses its signed internal coordinate but leaves the
receiver response unchanged. Reflection of the base cycle, combined with
form/phase sign reversal and the selected-pair swap, gives opposite nominal
receiver responses at pairs 1 and 4. These are symmetry checks on the same
law, not assumptions about arbitrary perturbed paths.

Only geometric identity is promised. Four nominal pairs lie on invariant
synchronized tips; all-five-pair internal activity is not inferred. Support,
partition, initial preparation and the complete sine law remain supplied.
This gate does not claim formation, autonomous law selection, physical binding
or identification of a material constituent.

### 36.2. Matched nominal storage and the complete mean rows

Write \(d=(\theta_{0,+}-\theta_{0,-})/2\) and
\(v=(x_{0,+}-x_{0,-})/2\) for the selected pair's internal coordinates.
Every other nominal pair starts at its synchronized tip. Those tips are
invariant under the complete law, as proved in
[Section 12.2](SINE_REPLICA_PULSE.md#122-every-nontip-pair-remains-internally-active).
Consequently the following reduction is exact for these nominal paths:

\[
\begin{aligned}
X_i'&=\frac{R_i}{2}\sum_{j=i\pm1}R_j\sin(\Theta_j-\Theta_i),&
\Theta_i'&=X_i-\frac{X_{i-1}+X_{i+1}}2,\\
d'&=v,&
v'&=-F_0(\Theta)\sin d,\\
R_0&=\cos d,\qquad R_i=1\ (i\ne0),&
F_0&=\frac{\cos(\Theta_1-\Theta_0)+\cos(\Theta_4-\Theta_0)}2 .
\end{aligned}
\]

In particular \(|F_0|\le1\), while its actual value evolves with both
neighboring pairs. No environmental phase or pressure is held fixed.
This tip reduction is not imposed on the perturbed preparations.

Let \(H_* =20(1-c)\) be the target storage. In A, only the eight edges
incident to pair 0 change their phase contribution. Summing the two signs
on each neighboring block gives

\[
H_A(0)-H_*=8c(1-\cos\delta).
\]

In B, the phase potential stays at its target value. Each of the two
selected forms has four zero-form neighbors, so its total form storage is
\(4u^2=8c(1-\cos\delta)\). Thus the nominal preparations match total
storage for the same delta, as well as all pair mean forms and midpoint
phases. They retain different internal coordinates:

\[
(R_0,U_0,Q_0)_A=(\cos\delta,0,0),\qquad
(R_0,U_0,Q_0)_B=(1,u^2,0).
\]

This is an internal-state comparison at equal nominal storage, rather
than an inactive reference against an active source. Storage equality is
not imposed on independently perturbed states or on different deltas.

At the target midpoint phases the mean form row has only two nonzero
components,

\[
I_1(d)=k(d):=\frac{s}{2}(1-\cos d),\qquad I_4(d)=-k(d).
\]

The selected pair's mean row and the other two mean rows vanish there.
Thus \(Y_A'(0)=k(\delta)>0\), whereas \(Y_B'(0)=0\).
More precisely, the smooth B rows give
\(d(\tau)=u\tau+O(\tau^3)\) and
\(\Theta(\tau)-\Theta_* =O(\tau^4)\); hence

\[
Y_A(h)=k(\delta)h+O(h^3),\qquad
Y_B(h)=\frac{su^2}{12}h^3+O(h^5).
\]

For B, the absence of a quadratic term in \(d\) follows from
\(v'(0)=0\). Substitution into \(I_1(d)\), with the midpoint displacement
of order four, gives the displayed cubic coefficient. The finite proof
below supplies explicit bounds instead of relying on these asymptotic
statements.

For fixed positive delta the matching relation gives
\(su^2/12=(c/3)k(\delta)\), and hence
\[
\frac{Y_B(h)}{Y_A(h)}=\frac{c}{3}h^2+O(h^4)\qquad(h\longrightarrow0).
\]
This is an early-time consequence of placing the same nominal storage in
different internal coordinates. It is not a universal delay law, an
independent clock or a ratio bound for arbitrary perturbed records.

The fine storage and total form are conserved throughout both nominal
and perturbed paths. Pair-member interchange is an exact symmetry, so it
reverses the signed internal coordinate without changing any pair-mean
readout. Base-cycle reflection followed by form/phase sign reversal and
the selected-pair interchange fixes each nominal preparation and exchanges
the receiver means with opposite sign. Uniqueness therefore gives
\(X_4(\tau)=-X_1(\tau)\) on either nominal path. This compensation does
not assert the same symmetry for arbitrary perturbations.

### 36.3. Explicit finite response with the environment retained

Write \(q=\delta_{\rm hi}\),
\(p=\sqrt{2c(1-\cos q)}\), and take \(0<h<1/\sqrt2\).
The reusable sufficient domain is \(0<\delta_{\rm lo}\le q\le1\);
it keeps sine and \(1-\cos\delta\) increasing over the amplitude range.
Let \(z=\Theta-\Theta_*\) use the continuous target-compatible midpoint
lifts. The mean current is globally Lipschitz in these midpoint phases
with constant two in the maximum norm, since each row averages two gaps
and \(|R_iR_j|\le1\). At the target midpoint phases,
\(\|I(d)\|_\infty\le d^2/4\). Therefore

\[
\|X'(\tau)\|_\infty\le2\|z(\tau)\|_\infty+\frac{d(\tau)^2}{4},
\qquad \|z'(\tau)\|_\infty\le2\|X(\tau)\|_\infty.
\]

For A, \(d(0)=\delta\le q\) and \(v(0)=0\). The bound
\(|d''|\le|d|\) yields
\[
|d(\tau)|\le V:=\frac{q}{1-h^2/2},\qquad
|d(\tau)-d(0)|\le\frac{V\tau^2}{2}.
\]
Indeed that maximum \(M_d\) satisfies
\(M_d\le q+(h^2/2)M_d\). The cosine mean-value bound then gives
\[
|\cos d(\tau)-\cos d(0)|\le\frac{V^2\tau^2}{2}.
\]
Define
\[
D_A=\frac{q^2}{4(1-h^2/2)^2},\qquad
K_A=\frac{D_A}{1-2h^2/3}.
\]
To establish the mean bound, put
\(M_1=\sup_{0<\tau\le h}\|X(\tau)\|_\infty/\tau\), which is finite
by smoothness and \(X(0)=0\). Then
\(\|z(\tau)\|_\infty\le M_1\tau^2\) and integration of the preceding
current inequality gives \(M_1\le D_A+(2/3)M_1h^2\).
Thus \(\|X(\tau)\|_\infty\le K_A\tau\).
For the receiver itself,
\[
|Y_A'(\tau)-k(\delta)|
\le2K_A\tau^2+\frac{s}{2}|\cos d(\tau)-\cos d(0)|
\le(2K_A+D_A)\tau^2 .
\]
Consequently the uniform A remainder is
\[
\boxed{|Y_A(h)-k(\delta)h|\le E_A:=
                    \frac{2K_A+D_A}{3}h^3.}
\]

For B, \(d(0)=0\) and \(v(0)=u\le p\). At each time \(\tau\), the
maximum \(M_d(\tau)\) of \(|d|\) up to that time satisfies
\(M_d(\tau)\le p\tau+(\tau^2/2)M_d(\tau)\). Hence
\[
|d(\tau)|\le\frac{p\tau}{1-h^2/2},\qquad
D_B=\frac{p^2}{4(1-h^2/2)^2}.
\]
Initially \(X=X'=X''=0\): all midpoint velocities vanish, and
\(R_0'(0)=-\sin d(0)v(0)=0\). Smoothness makes
\(M_3=\sup_{0<\tau\le h}\|X(\tau)\|_\infty/\tau^3\) finite.
Now \(\|z(\tau)\|_\infty\le M_3\tau^4/2\), so
\[
\|X'(\tau)\|_\infty\le D_B\tau^2+M_3\tau^4,\qquad
M_3\le\frac{D_B}{3}+\frac{M_3h^2}{5}.
\]
The B enclosure is therefore
\[
\boxed{|Y_B(h)|\le E_B:=K_Bh^3,\qquad
             K_B=\frac{D_B}{3(1-h^2/5)}.}
\]
These constants distinguish constant, quadratic and cubic time factors;
they are not interchangeable growth rates. Both estimates retain the
complete nominal mean motion and the evolving internal restoring
coefficient. They use no frozen pressure, external drive or trajectory fit.

### 36.4. Full-dimensional preparation and recorded contrast

Compare each perturbed fine trajectory with its own nominal trajectory
under the same complete sine law. Choose initial phase lifts realizing
the circular error bounds and continue them continuously. The maximum
fine form and lifted-phase differences satisfy the same two-channel
comparison as Section 33.2, now with kernel derivative bound one. Thus
\[
u_x(h)\le\rho_x\cosh(2h)+\rho_\theta\sinh(2h).
\]
The factorial inequalities \((2k)!\ge2^k\) and
\((2k+1)!\ge6^k\) give the sufficient rational bound
\[
\boxed{P(h)=\frac{\rho_x}{1-2h^2}
       +\frac{2h\rho_\theta}{1-2h^2/3},\qquad u_x(h)\le P(h).}
\]
The domain \(h<1/\sqrt2\) keeps both denominators positive. It limits
this comparison formula rather than existence or stability of the law.
The errors include every fine form and phase, the environment and the
absolute common form origin. The nominal tip reduction and its Taylor
information are not imposed on perturbed states.

For each recorded scalar, the additional allowance is \(P(h)+\eta\).
Because \(k\) increases on the admitted positive amplitude interval,
every A record and every B record satisfy
\[
\begin{aligned}
y_A&\in[h k(\delta_{\rm lo})-E_A-P-\eta,\,
         h k(q)+E_A+P+\eta],\\
y_B&\in[-E_B-P-\eta,\,E_B+P+\eta].
\end{aligned}
\]
In particular,
\[
\boxed{y_A-y_B\ge
 h k(\delta_{\rm lo})-E_A-E_B-2P-2\eta.}
\]
Errors need not be statistically independent. The bound permits the
branches to choose different amplitudes in the retained interval and
covers their entire closed preparation boxes. Their interiors are open
sets in all twenty fine state coordinates; no symmetry, mean, tip or
equal-energy equation restricts the perturbations.

### 36.5. Admission of both complete boxes to the same all-time barrier

Let \(P_f=I-\mathbf1\mathbf1^T/10\), and use the unique target phase
chart from [Section 12.1](SINE_REPLICA_PULSE.md#121-an-admitted-open-full-state-family).
Its relative squared radius is
\(Z_f^2=\|P_fx\|^2+\|\vartheta\|^2\), where
\(\theta=\theta_*+c_0\mathbf1+\vartheta\) and
\(\vartheta\perp\mathbf1\). Componentwise preparation errors imply
the following upper bounds:
\[
\begin{aligned}
N_A&=10\rho_x^2+(\sqrt2q+\sqrt{10}\rho_\theta)^2,\\
N_B&=(\sqrt2p+\sqrt{10}\rho_x)^2+10\rho_\theta^2.
\end{aligned}
\]
Centering cannot increase the norm of either error vector. The nominal
relative displacement has norm \(\sqrt2\delta\) in A's phase channel,
and \(\sqrt2u\) in B's form channel, which proves these bounds.

The fine graph has degree four. Both its form Hessian and the absolute
operator norm of the phase Hessian are at most eight. A Taylor remainder
for a componentwise error of width rho is therefore at most \(40\rho^2\).
In A, the phase-gradient absolute sum at the nominal state is
\(8c\sin\delta+8s(1-\cos\delta)\); the form gradient is zero.
This follows directly from the two selected form rates
\(\mp c\sin\delta\) and the four neighboring rates of magnitude
\(k(\delta)\), since the phase gradient is minus four times the form row.
In B, the phase gradient is zero and the form gradient has only the
selected entries \(+4u,-4u\). Uniform excess-storage bounds are thus
\[
\begin{aligned}
G_A={}&8c(1-\cos q)
 +8\rho_\theta[c\sin q+s(1-\cos q)]
 +40(\rho_x^2+\rho_\theta^2),\\
G_B={}&8c(1-\cos q)+8p\rho_x
 +40(\rho_x^2+\rho_\theta^2).
\end{aligned}
\]
These are upper bounds on each actual initial excess, not an assertion
that independently perturbed states retain equal energy.

Set
\[
\mu_{\rm acute}=\frac{\pi}{2}-\alpha-\sqrt2r,\qquad
c_r=\cos(\alpha+\sqrt2r),\qquad
\kappa=\frac{5-\sqrt5}{2}c_r .
\]
The complete fine spectral gap is \(5-\sqrt5\). If
\(\mu_{\rm acute}>0\), the target's criticality and its phase Hessian
bound on the closed radius ball give
\[
H-H_*\ge\frac{5-\sqrt5}{2}
  \bigl(\|P_fx\|^2+c_r\|\vartheta\|^2\bigr)
\ge\kappa Z_f^2 .
\]
The strict sufficient box admissions are
\[
\boxed{r^2-N_A>0,\quad r^2-N_B>0,\quad
       \kappa r^2-G_A>0,\quad\kappa r^2-G_B>0.}
\]
Choose an intermediate energy ceiling strictly between
\(\max(G_A,G_B)\) and \(\kappa r^2\), and a finite open form-mean
interval containing \([-\rho_x,\rho_x]\). Both boxes lie in the
existing invariant family. Conservation precludes a first radius exit,
because such an exit would require excess at least \(\kappa r^2\).
The smooth complete flow exists for all time, so the argument holds
in both time directions.

Every admitted trajectory therefore keeps the acute fine-edge chart
and all target cycle windings for every real time. The unordered pair
chart remains valid. This is the same complete trajectory whose finite
receiver record was bounded above. No all-five-pair activity is inferred:
four nominal pairs remain at tips, and arbitrary box members may also
have tips. The protected identity here is exactly the geometric identity
declared in Section 36.1.

### 36.6. Exact evaluation of the frozen joint certificate

The shared interval evaluation uses the exact identity
\(1-\cos\delta=2\sin^2(\delta/2)\), the symbolic target angle,
and outward Machin trigonometry. In particular it does not materialize
the target as a binary floating-point phase vector. Rational growth
coefficients are evaluated exactly; transcendental and square-root
endpoints are enclosed on the declared dyadic128 grid.

For the frozen inputs, \(E_A=34/459375\), \(P=1/312500\), and
the certified B remainder is approximately \(7.080003879559978\,10^{-6}\).
The resulting recorded contrast enclosure is

\[
\boxed{
y_A-y_B\in
\left[
\frac{5952295542928897572800287330908990}{2^{128}},
\frac{95170020078990130093368345419210539}{2^{128}}
\right].}
\]

Its lower endpoint is approximately \(1.7492224462844\,10^{-5}\)
and is strictly greater than the declared \(1/100000\).
The same evaluation gives the following exact certified lower bounds,
each numerator divided by \(2^{128}\):

| Required margin | Lower-bound numerator |
| --- | --- |
| Acute radius, \(\pi/2-\alpha-\sqrt2r\) | 46748866114495546899709594196780814092 |
| A initial relative radius, \(r^2-N_A\) | 4227886659066376311040093052496675910 |
| B initial relative radius, \(r^2-N_B\) | 4980399457955558130784895944386858634 |
| A excess-storage barrier, \(\kappa r^2-G_A\) | 333360649174566922112906803499587740 |
| B excess-storage barrier, \(\kappa r^2-G_B\) | 333335832765852420127717913460847076 |

The norm and excess-storage upper bounds are rounded upward and the
coercivity/barrier bounds downward before these conservative margins are
formed. Every listed margin is strictly positive. Both full preparation
boxes therefore retain the declared geometric identity for all time,
while every competing pair of recorded responses satisfies the required
strict finite separation. This is a joint certificate on the same law,
support and complete trajectories; a response sign alone was insufficient
for the frozen stopping rule.

### 36.7. Shared assessment and boundaries of the joint result

The shared
[pair owner](../../src/tnfr/physics/relational_sine_pair.py)
implements `assess_sine_pair_persistent_response` and returns
`SinePairPersistentResponse`. It re-admits the amplitude interval,
structural horizon, three error budgets and trapping radius, then rebuilds
the symbolic preparation, matched nominal storage, finite mean bounds,
full-flow preparation comparison and both persistence admissions.
`recorded_difference_bounds` retains the direct outward contrast;
`initial_radius_margin_bounds`, `acute_radius_margin_bounds` and
`storage_barrier_margin_bounds` retain the separate geometric obligations.
The reusable status `certified_persistent_response` requires both
persistence flags and a strictly positive recorded contrast. The frozen
gate additionally requires the stronger lower margin \(1/100000\),
which the displayed exact endpoint exceeds.

The [contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-persistent-response),
[usage](../../docs/guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-persistent-response)
and [independent controls](../../tests/physics/test_sine_pair_persistent_response.py)
own primitive admission, export, executable usage and implementation checks.
No measured response, numerical trajectory, parameter search or incoming
certificate is consumed. An unavailable sufficient bound would prove
neither escape from the identity chart nor equality of actual responses.
No older evaluated protocol or result has been altered.

The collective means and nominal total storage do not determine this
receiver response: the retained internal form/phase allocation matters.
This does not identify a unique law, physical energy, binding mechanism or
material constituent. The common form origin is bounded by preparation,
and the support, pair partition, capacities, clock and complete sine law
remain supplied premises. Pair identity is retained; it was not created.

The broader trapping family used in the first-exit argument is invariant
in both time directions. Uniqueness therefore prevents this same complete
flow from entering that invariant family from outside it. This statement
concerns the invariant family, not the smaller initial boxes. A formation
claim would need a separately specified source set, target criterion and
admitted law or event mechanism. The present prepared persistence and
interaction certificate supplies no such formation claim.

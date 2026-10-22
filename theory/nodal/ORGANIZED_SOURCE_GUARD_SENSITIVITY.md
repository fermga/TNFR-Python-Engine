# State-guard sensitivity of the organized word

<a id="organized-source-guard-sensitivity"></a>

**Conditional result.** A mandatory first downward crossing of a supplied
energy level gives a locally unique occurrence of Reception (EN), Coherence
(IL) and Silence (SHA) near the existing organized-source witness. Its
response derivative contains the change in event time as well as the reset
derivative. The correction is nonzero in a fixed internal observation, even
after both compared words have completed at a common structural time.

Under one fixed guard, equal child means, energy and instantaneous loss can
still give different next occurrence times. The
[predictive-state obstruction](#organized-source-guard-predictive-state)
identifies fine information omitted by that instantaneous summary.

This owner extends the [complete flow/word comparison](ORGANIZED_SOURCE_PORT.md#organized-source-flow-word-timing).
The new statement is a local hybrid sensitivity law, not another generic
argument that policies are supplied. Existing
[support-law countermodels](RELATIONAL_SUPPORT_EVENTS.md#support-law-choice-and-clock)
and [transverse admission crossings](RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact)
retain their separate models. Neither a guard's mathematical regularity nor
its implementation would derive the choice of threshold from TNFR.

## 1. Complete model and mandatory occurrence

Retain the organized source's 45 labeled nodes, ninety unit edges, marked
targets `(0,1)`, raw form, primitive phase lifts and structural clock tau.
The support and targets are supplied. Initially all capacities are one;
after the word they are kappa at the targets and one elsewhere. Let L be
the actual fine Laplacian, `q=Lx`, `gamma=(1023*pi)^-1` and

\[
 S_i(\Theta)=\sum_{j\sim i}\sin(\Theta_j-\Theta_i),\qquad
 F_D(x,\Theta)=\frac14
 \begin{pmatrix}D[-q+\gamma S(\Theta)]\\ \gamma Dq\end{pmatrix}.
\]

Both rows use the same positive held capacity matrix D. Pressure is the
independent formula `p=(-q+gamma*S)/4`, refreshed before admission and
continuation. There is no external forcing. The actual excess is

\[
 E(x,\Theta)=\tfrac12 x^T Lx+
        \sum_{\{i,j\}}[1-\cos(\Theta_j-\Theta_i)]-H_*,
 \qquad H_*=45[1-\cos(2\pi/9)].
\]

Use the [section 23 ideal domain](ORGANIZED_SOURCE_PORT.md#organized-source-word-reuse),
including its acute lift branch, `E<1/100`, initial `abs(m(1))<1/30`,
shared represented factors and all execution-context premises. These include
the common `organized-source` kind, initial `AL, IL` history, ordinal two,
zero debt, prior Coherence, initialized core grammar, `raise`, hard unit rails,
explicit IL `event_domain` policy, actual Reception ceilings and Silence
floor. Retain the actual context updates and weighted means, with carrier
subtraction for phase origins. No custom monitor or additional callback
mutation is present.

For a **fixed constant** ell, supply the following occurrence policy:

1. With both target ordinals two, flow while `E>ell`. At the first downward
   crossing `E=ell`, Reception is mandatory; continued pre-word flow through
   that crossing is not another permitted choice.
2. After successful Reception both target ordinals are three. Immediately
   apply Coherence, then Silence at ordinal four. Each two-target stage reads
   its own snapshot and is separately atomic. There is no intervening flow
   and no claim of a single rollback transaction.
3. At ordinal five disable further word requests and continue with the
   changed held capacities. No second word or renewal mechanism is present.

The monotonic ordinal is existing retained context, owned by
[`read_grammar_state`](../../src/tnfr/operators/grammar_debt.py) and
[`current_operator_step`](../../src/tnfr/glyph_history.py); it is not the
length of a bounded trace. This policy specifies how that context is consumed;
recording an ordinal alone would neither execute nor authenticate an event.
Urgency and the ordered requests are additional model premises. Starting on
or below the guard, mismatched ordinals and failed live admission lie outside
the source domain proved below.

The rule reads the complete network state. No local sensor, unmarked-target
selection or default native-selector realization is established. It is
autonomous on the mathematical and grammar state because it does not read
absolute time. Silence's wall-clock timestamp remains execution telemetry.
The threshold, target marks and request policy are supplied law parameters,
not an unrecorded clock or an additional hidden latch. In particular ell is
not recomputed from each perturbed initial state's energy.

## 2. Local positive waiting and unique crossings

Write `z=(x,Theta)` on the fixed-capacity source slice. The energy differential
and its pre-word rate are

\[
 n=DE=(q^T,-S^T),\qquad nF_1=-\tfrac14 q^Tq=-\rho.
\]

The phase terms cancel because both complete rows are retained. Nonincrease
alone does not imply `rho>0`; transversality is a separate local hypothesis.

Use the exact prior witness `x_*=h(e_0-e_1)`, `h=1/64`, and
`Theta_*=psi`, where `psi_(9c+j)=2*pi*(j-4)/9`. Its values are

\[
 E_*=6h^2=3/2048,\qquad \|Lx_*\|^2=80h^2,
 \qquad\rho_*=20h^2=5/1024>0.
\]

The [previous radius-10^-5 box](ORGANIZED_SOURCE_PORT.md#organized-source-flow-word-timing)
is already inside the ideal domain. In that box the degree-five target rows
give `abs(q_i-q_*i)<10*epsilon` for `i=0,1`, hence

\[
 \rho\ge\tfrac12(6h-10\varepsilon)^2>0,
 \qquad\varepsilon=10^{-5}.
\]

This estimate applies to independent phase perturbations as well. Smoothness
gives a short pre-word flow segment remaining inside the box. Its energy
decreases strictly, so every sufficiently close level below E_* is reached
once on that segment, at a positive time. The first crossing is unique;
the prescribed urgent stages and subsequent smooth flow then give a unique
local hybrid solution for each specified level and admitted initial state.
There are only three zero-duration stages and no recurrent event sequence.

For a crossing state `y=Phi^-_tau(s)`, define
`G(s,ell,tau)=E(Phi^-_tau(s))-ell`. Since `G_tau=-rho(y)<0`, the
implicit-function theorem gives a smooth crossing time in a neighborhood
of each such source and level. For each fixed sufficiently close level the
initial neighborhood can be chosen open in all ninety continuous coordinates,
with `E(initial)>ell`, the same context and a positive waiting time. This
does not claim all section 23 levels or preparations cross transversely.

## 3. The derivative at a common structural time

Let W be the complete primary word and let `F_+=F_D` use its actual Silence
successor capacities. On the fixed discrete-context domain define

\[
 K(y)=DW(y)F_1(y)-F_+(W(y)).
\]

The [earlier complete formula](ORGANIZED_SOURCE_PORT.md#organized-source-flow-word-timing)
retains Coherence's nonidentity phase derivative. At a fixed common time T
strictly after the crossing, set

\[
 Z(T;s,\ell)=\Phi^+_{T-\tau(s,\ell)}
                      W\Phi^-_{\tau(s,\ell)}(s),
 \quad M_-=D\Phi^-_\tau(s),\quad
 M_+=D\Phi^+_{T-\tau}(W(y)).
\]

All quantities below are evaluated at this crossing. Differentiating G gives

\[
 \delta\tau=\frac{nM_-\delta s-\delta\ell}{\rho}.
\]

Differentiating the full endpoint, including the changed duration of the
post-word flow, then gives

\[
 \begin{split}
 \delta Z&=M_+\{DW M_-\delta s+K\delta\tau\}\\
 &=M_+\left\{\left(DW+\frac{K\otimes n}{\rho}\right)
                     M_-\delta s-\frac K\rho\delta\ell\right\}.
 \end{split}
\]

Thus the event-local derivative at matched clocks, often called the
saltation matrix, and the threshold sensitivity are

\[
 \mathcal S=DW+K\otimes n/\rho,
 \qquad\partial_\ell Z=-M_+K/\rho.
\]

Using only DW would omit the rank-one timing correction. Its sign has an
independent consistency check:

\[
 \mathcal S F_1=DW F_1+K(nF_1)/\rho=F_+\circ W.
\]

A perturbation along the incoming flow must follow the outgoing flow after
matching clocks. Both phase and form participate in this identity. Under a
common constant clock rescaling, K and rho scale by the same factor, so
`K/rho` and the timing correction retain their coordinate meaning; the
crossing-time covector itself transforms with the clock unit.

These derivatives concern the full ninety-coordinate primary state on the
declared fixed initial-capacity and fixed-factor domain. Successor capacity
is the same D for every compared level and contributes no threshold derivative.
The observation clock T is held fixed. Derived fresh pressure has variation
`delta p=(-L*delta x+gamma*DS*delta Theta)/4`. Record/lifecycle updates remain
their separate maps. Neither weighted means nor primary coordinates are
projected away to obtain this derivative.

## 4. An exact internal correction and two fixed levels

At the carrier, `n=(q^T,0)` although the incoming phase row is nonzero.
For the fixed observation

\[
 \mathcal O(z)=\bigl(x_8-x_2,
              (\Theta_8-\psi_8)-(\Theta_2-\psi_2)\bigr),
\]

the previous exact result gives `D O K_*=(3*beta*h/5)*(1,-gamma)`.
Consequently the **event-local** threshold coefficient is

\[
 -D\mathcal O K_*/\rho_*
       =\frac{48\beta}{25}(-1,\gamma)\ne0.
\]

This is a local crossing-state coefficient, not an untransported coefficient
at an arbitrary later T. A separate initial-state control takes the direction
`v=(e_0,0)`. Here `n v/rho=96/5` and

\[
 D\mathcal O(\mathcal S-DW)v
          =\frac{9\beta}{50}(1,-\gamma)\ne0.
\]

In that direction DW alone gives zero for this contrast. The timing correction
therefore changes an actual prediction of the local derivative, not just a
bookkeeping label. Common form translations and common phase rotations give
`n v=0`; the correction vanishes on those offset directions. As before, only
offsets that stay inside the admitted absolute rails are compared.

To relate this derivative to two positive waiting times, fix the two law
parameters `ell_1=E_*-eta` and `ell_2=E_*-2*eta` for sufficiently small
`eta>0`. Their common initial state is s_*, and

\[
 \tau_1=\eta/\rho_*+O(\eta^2),\qquad
 \tau_2=2\eta/\rho_*+O(\eta^2).
\]

There is a `T_0>0` such that, for every fixed `0<T<T_0`, a sufficiently small
positive eta makes both crossings occur before T and gives

\[
 \mathcal O(Z(T;s_*,\ell_2))-\mathcal O(Z(T;s_*,\ell_1))
   =\eta\,D\mathcal O\,D\Phi^+_T(Ws_*)K_*/\rho_*+O(\eta^2).
\]

The displayed coefficient tends to `(48*beta/25)*(1,-gamma)` as T tends to
zero, so both components remain nonzero for sufficiently small fixed positive
T. This continuity statement includes the required post-word transport; the
bare event-local coefficient cannot replace it at finite T.

For each chosen such eta and T, continuous dependence gives a smaller open
initial neighborhood where both positive crossings and the observed distinction
persist, with the two levels held fixed. It need not be uniform as eta tends
to zero. Both histories have completed `AL, IL, EN, IL, SHA`, both capacities
equal D and both structural clocks equal T. The source's geometric identity,
form rails and fresh-pressure bounds follow from the existing one-word domain.

This is a distinction between two complete models with different supplied
thresholds, not two solutions of one fully specified law. The internal contrast
removes common coordinate offsets. No numerical value for T_0, eta's admissible
radius or a finite separation margin has been evaluated. There is no claim of
asymptotic separation, optimized activation or physical identification.

## 5. Detached implementation and controls

The private
[`_assess_organized_source_carrier_guard_sensitivity`](../../src/tnfr/physics/_organized_source_guard_sensitivity.py)
reuses the [complete carrier timing reader](../../src/tnfr/physics/_organized_source_timing.py).
It re-admits the primitive form and represented factors, derives rho from the
shared gradient, and rejects a zero-loss carrier state before division. Its
guard level is only the energy at the supplied crossing point; the reader does
not choose a threshold for another preparation or predict a first-hit time.

The report retains K, the exact event-time covectors, the factorization of
the timing correction, and `-K/rho` in both rows. Phase expressions retain
their rational coefficients of `cos(2*pi/9)` and their gamma factor, with
outward interval enclosures. Noncarrier crossings and propagation to T are
covered by the conditional theorem, not evaluated by this carrier reader.

[Independent tests](../../tests/physics/test_organized_source_guard_sensitivity.py)
check the incidence loss, the implicit crossing condition, complete incoming
and outgoing flow consistency, the failure of a reset-only derivative,
threshold signs, common-offset controls and nontransverse/invalid rejection.
Runtime events, integration and frozen producers are prohibited in these
algebraic checks. No live guard, scheduler or new occurrence policy is installed.

The result identifies what a supplied state guard contributes to a complete
prediction: its level and urgency, a transverse crossing, and the event-time
term in the derivative. Justifying that guard from an independent premise,
or using it in a finite numerical prediction, remains a separate question.

<a id="organized-source-guard-predictive-state"></a>
## 6. Collective summaries do not predict the same guard time

**Conditional obstruction.** Even with the threshold and occurrence policy
fixed, the five child form/phase means, capacities and grammar context,
excess energy E and instantaneous loss rho do not form sufficient predictive
state for the next guard occurrence. Two admitted fine preparations have the
same such snapshot but different first-hit times. They also have identical
initial primary state, fresh pressure and continuous rates at the marked
targets. The distinction concerns hidden fine preparation under one law,
rather than different choices of event threshold.

This specializes the
[collective descent criterion](../STRUCTURAL_OPERATORS.md#collective-operator-descent)
to the first occurrence time of this word. Generic failures of scalar storage
and loss as predictive state already appear in
[constituent memory](ORGANIZED_CONSTITUENT_MEMORY.md#organized-constituent-memory-result)
and the [native formation controls](RELATIONAL_FORMATION_CONTROLS.md#relational-reversed-form-control).
Those results keep their own supports, complete laws and predictions; the
present calculation derives the same-guard time distinction on this support.

### Observation and complete incoming law

Let `m_c=(1/9)*sum_j x_(9c+j)` and
`a_c=(1/9)*sum_j(Theta_(9c+j)-psi_(9c+j))`, using the admitted primitive
lifts. These are phase-offset means, not arguments of circular resultants;
the resultant of the exact winding-one carrier is zero. Retain the five
pairs `(m_c,a_c)`, the initial capacity vector and common supplied grammar
context, E and rho. The support, marks, factors, clock origin and fixed level
are the same. The strengthened snapshot may additionally retain the primary
coordinates, fresh pressure and both continuous rates at targets zero and one.
It does not retain the full fine form, fine pressure or their association.

For unit incoming capacities, the complete law gives

\[
 \dot E=-\rho=-\tfrac14q^Tq,\qquad
 \dot q=\tfrac14L[-q+\gamma S(\Theta)],\qquad
 \ddot E=\tfrac18\{q^TLq-\gamma q^TLS(\Theta)\}.
\]

At the exact initial carrier `S(psi)=0`, so
`ddot E(0)=x^T L^3 x/8`. This is an instantaneous identity of the coupled
model: `dot Theta=gamma*q/4` is generally nonzero, and the subsequent phase
is not held at the carrier. Away from that initial point the displayed
phase-dependent term remains part of the law.

### An exact matched pair in the admitted domain

Let `h=1/128` and repeat one profile in all five children:

\[
 x^A_{9c+j}=h(\delta_{j3}-\delta_{j6}),\qquad
 x^B_{9c+j}=h(\delta_{j3}-\delta_{j7}),\qquad
 \Theta^A=\Theta^B=\psi.
\]

All child form and phase-offset means are zero. Matching edges have zero
form and phase difference because corresponding child profiles agree. The
repeated-child subspace is invariant before the event: matching currents
vanish and every child obeys the same two coupled cycle rows, regardless of
its degree in the child graph. Antisymmetry of the cycle currents therefore
preserves each of those means during pre-word continuation.

For the nine-cycle Laplacian `L_9`, write `A_k` for its distance-k adjacency
matrix. Exact multiplication gives

\[
 L_9^2=6I-4A_1+A_2,\qquad
 L_9^3=20I-15A_1+6A_2-A_3.
\]

The two signed pairs have cycle distances three and four. Both have first
quadratic moment four and second moment twelve; their third moments are
forty-two and forty. The matching-edge terms vanish also on their successive
Laplacian images. Thus the full fine quantities are

| Quantity | Preparation A | Preparation B |
| --- | --- | --- |
| Initial excess E_0 | `10*h^2 = 5/8192` | `10*h^2 = 5/8192` |
| Initial loss rho_0 | `15*h^2 = 15/16384` | `15*h^2 = 15/16384` |
| Initial energy acceleration sigma | `105*h^2/4 = 105/65536` | `25*h^2 = 25/16384` |

Both have `E_0<1/100`, unit capacities and zero initial global form mean;
the exact carrier lies in the required acute lift branch. They therefore
belong to the section 23 ideal one-word domain with the shared initialized
context and explicit IL event-domain policy. Zero target forms are permitted
with the retained established history. No claim of acquiring these fine
preparations by that history is made.

The source is different from section 2's particular witness and radius box.
Here both target form values and q values are zero initially, so fresh target
pressure and both target rates are zero in both preparations. The full q is
nonzero and `rho_0>0`. Smoothness gives a neighborhood of each preparation
with strictly positive loss; the general transverse-crossing argument applies
without using the earlier degree-five target lower bound.

No support-preserving relabeling identifies the pair: `x^T L^3 x` is invariant
under every such permutation but differs here. A common form shift cannot
remove that difference, and both states already have the same zero means.
The observation has discarded dynamically relevant fine arrangement.

### One fixed level, different positive occurrence times

Choose one constant level `ell=E_0-eta` with sufficiently small `eta>0`,
and hold it fixed for both preparations. Apply the same mandatory first
crossing and EN–IL–SHA policy from section 1. Strict positive initial loss
gives a unique positive first crossing for each preparation in a short
pre-word interval. Both words remain within the shared ideal admission domain.

For a fixed preparation with initial loss rho_0 and acceleration sigma,
smoothness and the implicit-function theorem give

\[
 E(t)=E_0-\rho_0t+\tfrac12\sigma t^2+O(t^3),\qquad
 \tau(\eta)=\frac{\eta}{\rho_0}
       +\frac{\sigma}{2\rho_0^3}\eta^2+O(\eta^3).
\]

This is an expansion in a family of fixed levels for one initial point, not
the fixed-level derivative with respect to a varying source in section 3.
At every chosen eta the two compared experiments use exactly the same law
parameter ell. Substitution of the exact pair yields

\[
 \tau_A(\eta)-\tau_B(\eta)
    =\frac{\eta^2}{5400h^4}+O(\eta^3)>0
\]

for every sufficiently small positive eta. Preparation A reaches the level
later despite equal initial energy and loss. The nonzero quadratic term
proves unequal exact times by the local remainder statement; it is not a
computed finite waiting time or an evaluated error bar. No numerical eta
radius, crossing forecast or later endpoint separation is claimed.

### Predictive-state consequence and implementation boundary

A deterministic next-time function of this instantaneous summary, with the
same fixed level and parameters, would give the same output for both inputs.
The proved distinction rules it out on any source class containing this pair.
The summary also fails the continuous descent criterion directly:
`dot rho=-sigma` differs at the matched snapshot. It cannot have a single
autonomous first-order evolution law reproducing the full projected flow
on that source class.

This does not obstruct detection from an externally supplied exact live
energy stream: that stream already reveals when `E=ell`. Predicting its
future evolution from a snapshot is the question decided here. Retaining
sigma resolves the displayed quadratic discrepancy, but neither that fact
nor the pair proves closure of a larger moment system, minimal sufficient
state, or sufficiency of a finite noisy observation history.

The existing detached carrier guard reader additionally reports exact
`energy_second_time_derivative` and `drop_time_quadratic_coefficient`.
It rebuilds `Lq` using the shared support and primitive-state admission;
the second value is `sigma/(2*rho^3)`. Independent incidence tests check
both complete-law derivatives, the matched pair, exact series reversion,
scale and offset controls and tiny positive loss. No integration, live
guard, event execution or frozen response producer is used for this result.

The [child-swap hybrid representation](ORGANIZED_SOURCE_CHILD_QUOTIENT.md)
provides a sufficient state on a declared invariant sector containing both
preparations. It retains their distinct internal profiles, the counted word
maps and the same guard. It neither makes the insufficient summary above
closed nor reconstructs arbitrary off-sector fine states from averaged data.

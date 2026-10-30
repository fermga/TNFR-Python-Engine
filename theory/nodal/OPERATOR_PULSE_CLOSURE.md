# Operator origin and closure of a structural pulse

<a id="operator-pulse-closure"></a>
<a id="operator-pulse-causal-closure"></a>
## 1. A causal model must close the action and its occurrence

The question here is whether structural operators can constitute a dynamics
that produces and maintains a pulse. An operator need not be an additional
acting entity: it can name a transformation realized in an NFR's state and
relations. A constitutive law specifying that transformation and its
activation is a legitimate hypothesis. It does not have to be derived from
nothing. Its consequences must, however, follow from the law actually
specified, rather than from the operator's name or intended interpretation.

The [operator mechanism audit](../STRUCTURAL_OPERATORS.md#operator-mechanism-and-activation-audit)
distinguishes the implemented action, its admission, its selection and its
realization. The [native occurrence owner](../NATIVE_OPERATOR_DYNAMICS.md#native-selector-memory-and-occurrence)
already proves that equal instantaneous physical coordinates can give
different realized operators when consumed selector memory differs. Its
[decision-clock comparison](../NATIVE_OPERATOR_DYNAMICS.md#native-selector-clock-admission)
also shows that changing outer runtime calls changes the event schedule,
even when the continuous elapsed time and internal substeps agree.
Those results are retained; they are not new evidence in this study.

The [continuous pulse results](GLOBAL_PULSE_HYPOTHESIS.md#global-pulse-hypothesis)
concern a complete conservative sine law with declared pressure and held
capacities. They remain valid under those premises. They do not by themselves
derive that law, or the occurrence of its pulses, from the general operator
runtime. The following audit changes the causal question without changing
those frozen proofs or installing a new runtime law.

A deterministic autonomous hybrid completion specifies:

- The state z, including every consumed structural coordinate, target and
  support association, retained history, grammar state and selector memory.
- Every continuous row `z_dot=F(z)`, its units and clock, and any boundary
  inputs. State-dependent pressure is specified independently of the
  subsequently measured form rate.
- The actual event maps, admission conditions, state-dependent guards and
  selection or tie-breaking rule, with an explicit ordering for coincident
  events. A supplied periodic list is a different occurrence hypothesis.
- How pressure, histories and policy state are updated after each event,
  and what happens if a requested event is inadmissible. A fallback cannot
  silently replace the action used in a theorem.
- A domain and existence rule between events, including exclusion or an
  explicit treatment of infinitely many events in finite elapsed time.

This specifies what the structure does and why the model requests an action
at a particular state. It needs no intentional agent or additional operator
substance. A configured guard remains a constitutive postulate unless a
further realization argument derives it from retained interactions.
A proposed continuous generator of an operator needs its own continuous
family and limit proof; a registered zero-duration jump is not already such
a generator. The present result instead uses existing primary jump maps.

## 2. Observation recurrence and predictive state return are different

A pulse observation must be stated first. Repeated excursions of signed
form, a recurring local phase, a return of a complete structural state and
a return of the state sufficient to predict future events are different
claims. In particular, equality of one observed EPI value does not imply
equal capacities, pressure, next-event time or future waveform.

No return theorem here requires absolute clock time or an unused cumulative
log counter to return. The clock origin can be shifted. A proposed quotient
of lifecycle or history records is legitimate only when it preserves the
future law and occurrence rule on the selected domain. The existing
[collective action and occurrence criterion](../STRUCTURAL_OPERATORS.md#collective-operator-descent)
owns that sufficiency requirement. Discarding capacity while retaining a
flow that consumes it is not an autonomous closure merely because one form
observation repeats.

Fix a finite interval containing finitely many admitted events. For each
labeled constituent let x be its signed form and let `Delta x_e` be its
actual form jump. Under the unforced nodal row,

\[
x(T)-x(0)=\int_0^T\nu_f(t)p(t)\,dt+\sum_e\Delta x_e.
\]

Thus exact form return requires cancellation of the continuous contribution
and every event increment. If a declared Gamma term is present, its integral
must be added; it cannot be hidden in an operator label. Clipping changes
the actual increment and must be retained rather than replaced by a nominal
gain. This identity imposes no sign on the pressure integral.

For a strictly positive capacity along all continuous segments and event
endpoints, the independent necessary return balance is

\[
\boxed{\quad
\log\frac{\nu_f(T)}{\nu_f(0)}
 =\int_0^T\frac{\dot\nu_f}{\nu_f}\,dt
   +\sum_e\log\frac{\nu_f(e^+)}{\nu_f(e^-)}.
\quad}
\]

For a nonzero pressure of fixed sign the analogous balance uses
`log(|p(T)|/|p(0)|)`, its continuous logarithmic derivative and all pressure
reset ratios. A pressure zero or sign change requires the original pressure
row and jump identity instead; no logarithm is assigned there. Circular
phase return additionally requires its admitted lift increment to be an
integer multiple of `2*pi`. Support, constituent associations and consumed
memory must also return, or descend through an independently justified
identity equivalence.

For any declared differentiable storage S, the corresponding exact identity is

\[
S(z(T))-S(z(0))=
 \int_0^T DS(z)F(z)\,dt+\sum_e[S(z(e^+))-S(z(e^-))].
\]

It counts event work separately from continuous change. None of the names
Emission, Coherence or Silence supplies a universal storage sign. For
example, the conditional relational storage `E_D+beta*V` has no explicit
capacity term, so a capacity-only Silence event changes that storage by
zero even while it changes future mobility. This is not a claim of zero
work for every possible constitutive model or of universal physical energy.

## 3. A strict capacity loss excludes positive-capacity return

Consider a fixed set of existing targets. Between events capacities are
held. The selected events are Emission (AL), Coherence (IL) and Silence
(SHA), on their admitted branches, with no other capacity writer.
Emission preserves capacity even when its lifecycle code clears silence
metadata. Coherence also preserves capacity. Silence has the ideal map

\[
\nu_f^+=s_e\nu_f^-,\qquad 0\le s_e<1.
\]

These statements follow from the existing
[Emission and Silence proposals](../../src/tnfr/operators/al_sha_stage_proposals.py),
[primary glyph kernels](../../src/tnfr/operators/__init__.py) and
[factor contracts](../../src/tnfr/operators/factor_contracts.py).
Public Coherence may additionally change phase; that does not restore
capacity. The conclusion below therefore does not rely on removing its
phase proposal.

For any target starting with positive capacity, if every applied Silence
factor is positive, exact real composition gives

\[
\nu_f(T)=\nu_f(0)\prod_{e\in\mathrm{SHA(target)}}s_e.
\]

The empty product is one. A nonempty product is strictly below one, so no
interval containing such an event can return the target to its initial
positive capacity. A zero factor instead sets its capacity to zero, which
also cannot return to the positive initial value under these maps. The
result is independent of the form trajectory, event order, waiting times
and whether event selection is state-dependent. Choosing a different time
of occurrence cannot reverse a one-way change in another coordinate.

On a fixed finite population, total capacity is nonincreasing and decreases
strictly when any positive target undergoes a strict attenuation. Therefore
allowing a permutation of the retained constituent labels cannot restore
the whole capacity state either. Birth, removal, a capacity-transfer law,
an adaptation policy or an additional capacity-increasing event changes
these premises and must be accounted for separately.

This is a necessary-condition obstruction to exact structural return in
this selected model class. It does not exclude repeated EPI observations,
finite active episodes or pulses under other operator combinations and
continuous laws. Zero capacity freezes the unforced form row only; it does
not prove absence of the NFR or freeze every other coordinate.

### 3.1. Represented endpoints and pressure contraction

The source admits factors below one but promises nonincrease, not strict
decrease for every binary64 operand. Near the subnormal boundary a positive
attenuated product can round back to its source value. The shared Silence
kernel rejects a positive product rounded to zero, but a positive rounded
fixed point is a different case. Consequently the represented corollary
uses actual endpoints: if all accepted changes are nonincreasing and at
least one is strictly decreasing, positive capacity return is impossible.
Nominal `s<1` alone cannot certify that strict represented event. Exact
ideal products and represented endpoint claims remain distinct.

The same argument applies to stored pressure in the narrower model with
held pressure between events, no refresh or other pressure writer, and
Coherence factors `0<r_e<1`. Emission and Silence preserve stored pressure,
so a target with `p(0)!=0` satisfies

\[
p(T)=p(0)\prod_{e\in\mathrm{IL(target)}}r_e.
\]

A nonempty product excludes return to the original nonzero pressure.
A zero retention gives the separate zero-pressure case. Actual represented
strictness again needs its endpoint, not only its nominal factor.
A constitutive pressure refresh or a fresh relational pressure law can
change this product and lies outside the held-pressure premise. The general
native runtime includes such policies and is not identified with this
restricted event/flow composition.

More precisely, let a declared constitutive completion assign `p=P(y)`,
where y contains every consumed argument of that pressure law. A primary
Coherence event that holds y fixed changes the stored value to `r*P(y)`.
A subsequent refresh from the unchanged y restores `P(y)` exactly in the
ideal law. Unless `P(y)=0` or `r=1`, these are different pressure states.
Thus the held-pressure composition and the refreshed composition are
different complete models, even with the same primary glyph and without
any appeal to history or log counters. If an event also changes y, its
actual new arguments must instead be used in the refresh.

A finite source control uses the actual
[`default_compute_delta_nfr`](../../src/tnfr/dynamics/dnfr.py) kernel on a
unit-weight two-node edge, forms `(1/8,3/8)`, equal capacities one and equal
phases `1/4`. The explicit `DNFR_WEIGHTS` select `epi=1` and
`phase=vf=topo=0`. Each node's fresh pressure is its neighbor's form minus
its own, giving `(1/4,-1/4)`. Primary Coherence with retention `1/2` on the
first node changes only its stored pressure to `1/8`; refreshing restores
`1/4` because the constitutive arguments are unchanged. This control
excludes public/staged phase alignment. It establishes a concrete breaker
of the held-pressure premise, not a maintained pulse or a new occurrence
law.

### 3.2. An increasing operator need not undo Silence

Expansion (VAL), with its edge-aware form change disabled, multiplies
capacity by its configured factor and leaves pressure unchanged. Its
standard factor is `1+delta`, while standard Silence uses `1-delta`, with
`delta=1/(4*pi)`. In exact real arithmetic one of each gives

\[
\nu_f^{\rm final}=(1-\delta^2)\nu_f^{\rm initial}
                       <\nu_f^{\rm initial}.
\]

Thus their standard factors are not inverses. The source's separately
represented constants and endpoint products require the same finite
arithmetic distinction above. With edge awareness enabled Expansion also
changes form, whose actual event work and return balance must be retained.
Other admission and grammar conditions are not implied by this capacity
calculation. Balancing capacity alone would still leave any unreplenished
pressure contraction, form change and consumed memory to be closed.

## 4. An autonomous threshold completion with recurrent form excursions

A constructive countermodel shows why an occurrence rule alone is not the
missing maintenance mechanism. This is an explicitly supplied ideal hybrid
law built from the three primary maps, not an engine integration, a claim
about all public operator side effects or a newly derived universal law.
It is autonomous: its event guard uses current state, not an external
periodic schedule.

Take one retained target with state `(x,p,nu_f,theta)`, fixed support and
constant theta. Choose `b>0`, a form threshold L with both L and `H=L+b`
strictly inside the declared hard-clipping interval, and `0<r,s<1`.
Between events declare all rows explicitly:

\[
\dot x=\nu_f p,\qquad \dot p=\dot\nu_f=\dot\theta=0,
\qquad p<0,\quad\nu_f>0.
\]

At the downward crossing `x=L`, apply the ordered primary word
`Emission -> Coherence -> Silence` exactly once:

\[
(x,p,\nu_f,\theta)^-=(L,p,\nu_f,\theta)
 \longmapsto (L+b,rp,s\nu_f,\theta)^+.
\]

The positive jump removes the state from the guard. One may equivalently
use a reset latch rearmed only after leaving the guard; it supplies no
period. Each next event is the first downward crossing. The retained
pressure is a declared constitutive state, not a pressure reconstructed
from an observed form derivative. It is also not the fresh native pressure
formula or the conservative sine pressure. Disabling or changing that
holding law would define a different model.

The units are `[b]=[L]=[x]`, dimensionless r and s, and
`[p]=[x]/([nu_f][t])`. No equality between capacity and observed pulse
frequency is assumed. A positive clock rescaling transforms every continuous
row and the inferred event times together.

### 4.1. Exact event map and elapsed times

Prepare the post-event section at time `t0=0` with
`x0=H`, `p0<0` and `nu0>0`. Let n count completed subsequent events. At
the post-event section after event n,

\[
x_n=H,\qquad p_n=r^n p_0,\qquad\nu_n=s^n\nu_0.
\]

On its following flight,

\[
x(t)=H-\nu_0|p_0|(rs)^n(t-t_n),\qquad
\Delta t_n=t_{n+1}-t_n=
       \frac b{\nu_0|p_0|}(rs)^{-n}.
\]

The formula follows by solving the declared constant-rate row and setting
its endpoint to L. It is not a numerical trajectory. Since `0<rs<1`, the
flight lengths increase geometrically and their sum diverges. Thus this
ideal model has no accumulation of infinitely many events in finite time.
Its successive form excursions have the same endpoints and jump amplitude,
but they do not have a common elapsed period. Capacity and pressure both
drift, and the next duration is different at each repeated high-form state.

For `q=rs` and `B=b/(nu0*|p0|)`, the event times are exactly

\[
t_N=B\frac{q^{-N}-1}{q^{-1}-1}.
\]

Each completed excursion satisfies the form balance
`integral(nu_f*p dt)=-b` followed by the form jump `+b`.
Over N excursions the two totals cancel. In contrast, the capacity and
pressure log changes are `N*log(s)` and `N*log(r)`, both negative.
This gives repeated measured form excursions without positive-capacity or
nonzero-pressure structural return, even though the event times are
endogenous to the supplied guard. The observation x alone is insufficient
to predict the next duration.

### 4.2. A dyadic example and its admission boundary

Use exact parameters in declared structural units,

\[
L=\frac18,\quad b=\frac14,\quad H=\frac38,\quad
p_0=-1,\quad\nu_0=1,\quad r=s=\frac12.
\]

The nonclipped primary event maps give

\[
p_n=-2^{-n},\qquad\nu_n=2^{-n},\qquad
\Delta t_n=\frac{4^n}{4},\qquad
t_N=\frac{4^N-1}{12}.
\]

The first three threshold times are `1/4`, `5/4` and `21/4`.
These are exact consequences of the hypothetical completion, not a recorded
simulation or a claim of represented accuracy for arbitrary n. Ideal real
positivity for every finite n is not an infinite binary64 execution promise.

The optional strict public/staged Emission policy additionally requires
capacity at least `VF_BASAL_THRESHOLD`, whose default is `1/2`.
The comparison explicitly sets `EPI_LATENT_MAX=1/2`, whose form ceiling
also admits L in this example. At the first two threshold
requests the capacities are respectively one and one half; at the third
request capacity is one quarter and fails that default necessary gate.
Assuming the other live grammar and execution conditions are admitted,
this therefore permits only the first two complete words under that
capacity policy. This is a local admission statement, not a claim that
primary-map algebra supplies every lifecycle or grammar premise. Coherence
and Silence retain their own gates. No replacement action is silently
inserted at the third threshold.

More generally, for a positive Emission capacity floor `nu_min`, initial
capacity at least that floor and repeated constant attenuation s, the
number of such guard requests that can meet the capacity floor is at most

\[
1+\left\lfloor
      \frac{\log(\nu_0/\nu_{\min})}{-\log s}
       \right\rfloor.
\]

Equality of the pre-event capacity with the floor is allowed. This ideal
formula counts requests before the next attenuation; represented decisions
use admitted actual values. Removing the optional floor changes the policy
but does not remove the declining capacity and pressure or restore a period.
Clearing Silence's latency metadata in Emission is not replenishment.

## 5. What an operator-derived maintained pulse still requires

The selected operators can participate in a declared autonomous occurrence
model. The explicit guard completion shows this without invoking an
external acting entity or prescribing a period. Its repeated form events
nevertheless fail the stronger return requirement because two consumed
channels decay. A pulse claimed to be exactly periodic in predictive
structural state, or periodic after a justified identity equivalence, must
close those channels as well as its chosen observable. A maintained but
aperiodic signal or an asymptotic regime is a different claim and need not
return exactly after a finite interval.

For positive recurrent capacity, the general log balance requires enough
continuous replenishment, capacity-increasing events or admitted transfer
to compensate all negative jump contributions. For recurrent nonzero
pressure, its actual evolution and refresh must likewise balance its
contractions. Form return has its independent flow/jump requirement, and
phase, support and predictive memory retain theirs. These are necessary
conditions, not a sufficient existence or stability theorem.

In particular, the finite strict-loss argument excludes exact return; it
does not assert that every variable attenuation sequence tends to zero.
Strict factors approaching one can have a positive infinite product.
The explicit constant-factor threshold model has the stronger decay and
growing-wait properties proved above under its own premises.

The compensating mechanism may use existing operators, a justified native
adaptation/pressure law or a newly stated constitutive premise. It need not
be an external periodic forcing, but it must be declared and evaluated with
its actual target associations, ordering, admission and work balance.
A state-dependent restoration is still a specified law until its structural
realization is proved. Exactly canceling numerical gains does not select
when either action occurs, establish recurrence of every other channel,
or prove attraction of a pulse.

The result excludes only the selected nonreplenishing compositions, not
the entire operator catalog or the user's operator-origin hypothesis.
It identifies a concrete closure obligation that an operator-derived pulse
model can meet or fail. No scalar EPI excursion is thereby identified with
a material particle, a universal clock or a measured atomic property.
The research program owns which restorative mechanism is investigated next.

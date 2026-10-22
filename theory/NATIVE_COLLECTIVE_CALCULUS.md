# Native collective calculus

This owner combines conditional moment identities, finite admission margins
and bounded collective observations for the general native operator runtime.
The [source and branch studies](NATIVE_OPERATOR_DYNAMICS.md) own the underlying
preparations and counterexamples. These are detached mathematical descriptions
of the declared execution path, not a new runtime or a derived occurrence law.
The [aggregate transition owner](NATIVE_AGGREGATE_OPERATORS.md#native-uniform-capacity-transition)
separately proves a restricted lossy reduction under the actual uniform-capacity
native policy, retaining its admission margins and discarded-output boundary.
The [research program](RESEARCH_PROGRAM.md#research-current-question)
remains the sole research queue.

<a id="native-collective-calculus"></a>
### Conditional collective calculus for five native operators

Emission (AL), Reception (EN), Coherence (IL), Resonance (RA) and Transition
(NAV) share an exact conditional event-plus-held-flow description on doubled
C5. Unequal constituent forms are allowed. The event mean and internal
contrast have distinct roles; the complete native state also retains
constituent-associated memory. This synthesis reuses the
[block descent](NATIVE_OPERATOR_DYNAMICS.md#native-ra-il-block-descent),
[four-operator extension](NATIVE_OPERATOR_DYNAMICS.md#native-block-composition-exit),
[Transition map](NATIVE_OPERATOR_DYNAMICS.md#native-transition-contrast) and
[correlated Reception](NATIVE_OPERATOR_DYNAMICS.md#native-correlated-reception) results. It does not
assert that the default selector realizes every assignment of these operators.

**Domain and request origin.** Keep the block-descent execution premises:
unit undirected doubled C5, consecutive nonadjacent twins, whole-pair order,
default mixed pressure/Si and direct glyph execution, strict live grammar,
hard clipping, live built-in zero Gamma and one Euler substep `delta=1/16`.
No external nodes, custom callbacks, optional mathematics engine or optional
Resonance tracking are included. Automatic REMESH must independently abstain;
insufficient retained `stable_frac` history is one sufficient reason. It is
not removed from the law. Each node receives exactly one effective request.

Forms are finite admitted signed scalars (including admitted uniform-real
BEPI), with common nonempty semantic kind. Capacities are positive and paired;
primitive phases are paired and admit an inherited common lift of width less
than pi. Every proposed event and flow form lies strictly inside its rails.
Post-Resonance capacities also lie inside their rails: integration uses the
amplified value before the later clamp. Consumed histories, counters, rates,
clocks, normalized diagnostics and produced arithmetic retain their admission.
The equations below are ideal-real identities over represented coefficients,
not universal binary64 enclosures.

Both effective requests in each pair must agree and pass their own live
grammar, identity and operator gates. Retain their actual origin, including
selection preparation and the increment-before-comparison AL/EN lag priority.
A supplied paired transcript can test the algebra conditionally; it is neither
a default-selector witness nor a derived occurrence law. Prepared requests
can differ within a pair because its pressures and Si differ. Rejection is
sequential: a successful first twin or earlier pair is not rolled back by a
later rejection. The theorem concerns successful complete calls.

**Prepared pressure.** Write each incoming pair as

```text
x_(2i)=M_i-d_i,                  x_(2i+1)=M_i+d_i,
N_i(z)=(z_(i-1)+z_(i+1))/2,
B_i=w_phase*(N_i(theta)-theta_i)/pi_rep
    +k*(N_i(M)-M_i)+b*(N_i(V)-V_i),
p_(2i)=B_i+k*d_i,                p_(2i+1)=B_i-k*d_i.
```

Here `k=w_epi>0`, `b=w_vf`, and the common arc justifies the neighbor-resultant
midpoint. Regular support makes degree contrast zero. Preparation fixes this
independent pressure before any event. Its actual global normalization peak is
`P=max_i(abs(B_i)+k*abs(d_i))`, with divisor one only when `P=0`.
Si also consumes individual absolute pressure, paired capacity divided by its
maximum, and circular phase offset. The diagnostics are generally non-affine;
means alone do not determine selection or stability qualification.

**One ordered event table.** Initialize live means, phases and capacities with
`M,theta,V`, then process blocks `0..4`. Each block is still at its incoming
state when its pair begins; earlier pairs have updated its neighbors. Denote
its event mean by `Y_i`, event contrast by `A_i*d_i`, event capacity by `U_i`,
and event phase by `phi_i`. Let `H_i` be its pressure multiplier and `g_i,zeta_i`
the extra pressure mean/contrast. The following are moment identities, not a
parallel operator registry:

| Registered action | `Y_i` | `A_i` | `H_i` | `g_i` | `zeta_i` |
| --- | --- | --- | --- | --- | --- |
| Emission (AL) | `M_i+e` | `1` | `1` | `0` | `0` |
| Reception (EN) | `(1-m)*M_i+m*N_i(live Y)` | `1-m` | `1` | `0` | `0` |
| Coherence (IL) | `M_i` | `1` | `h` | `0` | `0` |
| Resonance (RA) | `(1-r)*M_i+r*S_i` | `1-r` | `1` | `0` | `0` |
| Transition (NAV) | `M_i` | `1` | `1-eta` | `eta*sigma_i*V_i+xi_bar_i` | `xi_delta_i` |

Use the actual default factors `e=AL_boost`, `m=EN_mix`, `h=IL_dnfr_factor`,
`r=RA_epi_diff`, `eta=NAV_eta`. Except for Resonance, `U_i=V_i` and
`phi_i=theta_i`. Live neighbor averages include earlier event changes but
never refresh `B_i` or its incoming contrast. Nonadjacent twins share the
same live neighbor input, so their two fine proposals yield these moments.
Equality holds after each complete pair, not at its singleton prefix.

For Resonance, with the actual U3 limit `L` and current paired phases, define

```text
J_i={j in {i-1,i+1}: abs(wrap(phi_j-phi_i))<=L},
S_i=mean_(j in J_i)(live Y_j),
alpha_i=Arg(sum_(j in J_i) exp(i*phi_j)),
phi_i=theta_i+c*wrap(alpha_i-theta_i),
U_i=(1+a*I(abs(S_i)>T))*V_i.
```

Here `c=RA_phase_coupling`, `a=RA_vf_amplification`, and `T` is the represented
runtime amplification trigger. Require nonempty `J_i`; the common arc then
makes the selected resultant defined. Recompute this subset at the live
prefix. Pairwise phase equality selects whole adjacent blocks, but it does
not justify selecting every neighbor or freezing the preparation-time subset.
Both fine sign identities must pass, not merely the mean sign identity.
Zero is neutral under the actual sign contract; the common kind remains
explicit. Amplification reads the selected signed mean, not mean absolute
constituent magnitude. Outside this domain an empty subset rejects, and an
undefined resultant requires the engine's separate phase policy; neither is
an arbitrary assigned angle.

For Transition use the default nonstrict random path, with common *strict*
prepared pressure sign `sigma_i`:

```text
sigma_i*B_i>k*abs(d_i),
xi_bar_i=(xi_(2i)+xi_(2i+1))/2,
xi_delta_i=(xi_(2i+1)-xi_(2i))/2,
abs(xi_bar_i)+abs(xi_delta_i)<=NAV_jitter.
```

Each innovation is its actual target-associated stream input. Both fine
nonidentity gates require `eta*(sigma_i*V_i-p_n)+xi_n!=0`. Common sign alone
does not prove them. The native zero-pressure branch takes positive sign;
it is outside this strict-sign sector, not an engine defect. Opposite signs
require the two sign contributions separately. Averaging innovations gives
the displayed observation, not a replacement RNG law or permission to
resample one collective draw.

**Held flow and sufficient means.** Every row has

```text
q_bar_i=H_i*B_i+g_i,
q_delta_i=-H_i*k*d_i+zeta_i,
M_i_next=Y_i+delta*U_i*q_bar_i,
d_i_next=(A_i-delta*U_i*H_i*k)*d_i+delta*U_i*zeta_i,
rate_bar_i_next=U_i*q_bar_i,
rate_delta_i_next=U_i*q_delta_i.
```

Subtract the associated prior stored rate and divide by `delta` to obtain
each acceleration moment. The held pressure contains incoming `d_i`, not
event contrast `A_i*d_i`; replacing the contrast factor by
`A_i*(1-delta*U_i*H_i*k)` changes the model. Post-event capacity `U_i` is
used before later adaptation. Integration does not recompute pressure or Si.

Conditionally on the same admitted paired requests, common Transition signs,
innovation means and retained phase/capacity inputs, form/rate means factor
through incoming means. Live Resonance subsets, amplification and phase rows
then agree by ordered induction. This is useful observation sufficiency;
internal contrasts still determine fine reconstruction and joint admission.
It does not eliminate histories, old rates, counters, stream association or
the actual grammar context. A finite composition uses these rows only while
each successor is re-admitted. It is not a complete bare-node stochastic
realization or an invariant five-operator sector.

**Temporal records and phase policy.** Each same-time event form jump is

```text
jump_(2i)=Y_i-M_i-(A_i-1)*d_i,
jump_(2i+1)=Y_i-M_i+(A_i-1)*d_i.
```

History restarts precisely for the constituent with a realized nonzero jump;
its flow endpoint is then appended with the existing four-entry cap. For an
EN pair with live neighbor mean `n` and `M_i=n+d_i`, the even form already
equals `n` and is unchanged, while the odd form changes by `-2*m*d_i`.
The analogous RA witness has `m` replaced by `r`, with its separate admission.
Thus a common glyph need not give common temporal-reset status. Do not replace
the two admitted histories by one block history. Prior rate association also
remains necessary, as the Reception same-form/different-rate witness shows.

The post-event phase coordinator uses the actual full fine glyph histories
to compute disruptor load, and the retained global gains and phase resultant.
Unequal traces may not be replaced by one representative trace. Given this
actual load, paired phases have identical local/global updates. Default gain
rails keep the common lifted arc convex and phases paired. This retains
primitive circular phase and its inherited lift, not a phase derived from form.
Different gain/load branches are allowed; their inputs must remain available.

**Capacity exit and hidden counters.** Let `s_n` be the incoming individual
stability count and use fresh prepared Si with held post-event pressure:

```text
chi_n=I(Si_pre_n>=si_hi and abs(q_n)<=EPS_DNFR_STABLE),
s_n_next=s_n+1 if chi_n else 0,
due_n=I(s_n_next>=tau),
W_i=N_i(U),
v_n_next=U_i+mu*due_n*(W_i-U_i),          n in {2i,2i+1}.
```

`W` comes from the immutable post-event capacity snapshot; fixed-point and
hull guards remain in force. In this unclipped ideal-real domain the successor
capacity half-contrast is exactly

```text
dV_i_next=mu*(due_(2i+1)-due_(2i))*(W_i-U_i)/2.
```

Consequently capacity pairing survives exactly when this product vanishes.
Equal due flags suffice, but are not necessary at a neighbor fixed point.
Zero mixing also annihilates the product algebraically; default mixing is
positive. Equal no-write flags do not imply equal qualification or counts:
from zero counts, one qualifying constituent can reach one while the other
stays zero, with neither write due. A later unequal due decision with nonzero
mixing and `W_i!=U_i` leaves the paired-capacity domain. This is a declared
exit, not a contradiction of the form map already evaluated with `U_i`.
Retain both counters even when the capacity values happen to remain paired.

The [analytical controls](../tests/physics/test_native_collective_calculus.py)
compare independently written fine-node sweeps and collective rows, including
live neighbor inputs and held-pressure timing. Detached boundary controls
retain Resonance subset/identity/amplification, Transition sign/nonidentity,
clipping, split temporal resets and split qualification/write decisions.
Production runtime, pressure, selection, operator and integrator evaluation
are blocked. The result supplies a reusable conditional calculus without a
new trajectory or engine API. It does not prove autonomous request selection,
repeated-call retention, formation or physical identification.

<a id="native-transverse-retention"></a>
### Finite internal-contrast retention with admission margins

The [five-operator calculus](#native-collective-calculus) gives a finite
retention criterion once its actual request and admission boundaries are
included. A contracting form contrast alone is insufficient: normalized
selection, sign identity, temporal resets and adaptation have independent
boundaries. This result concerns transverse form differences at fixed block
means, primitive phases and capacities, under that calculus's complete law.
It supplies neither an invariant domain for all time nor autonomous formation.

**Reference and comparison.** Fix a finite independently admitted reference
segment, its complete state, structural clock, whole-pair order, grammar,
request origin and post-event policies. Its physical form is pair constant,
with means `M_i`; retain every constituent's own history, rate and counter.
The comparison has the same means, phases and capacities but forms
`M_i-d_i,M_i+d_i`, with `abs(d_i)<=D_i`. Its actual temporal histories must
end at these forms and pass their own admission. Corresponding glyph traces,
debt, prior-Coherence flags, operator ordinals, AL/EN lags, stability counts
and global phase gains agree with the reference; they need not agree between
the two constituents. Initial rate differences remain associated and are
used to reconstruct acceleration. All other calculus premises still apply.

For Transition (NAV), keep each declared innovation mean `xi_bar_i` and bound
the associated half-difference by `Z_i`, with
`abs(xi_bar_i)+Z_i<=NAV_jitter`. The reference's duplicated mean input is an
algebraic comparison within that amplitude domain. It is not automatically
a seed-realizable native path. An actual stochastic claim must independently
retain and admit each target's stream and draws; do not replace them by a
new collective RNG. The concrete reference below uses only Resonance (RA)
and Coherence (IL), so it consumes no innovations. Other conditional words
retain their supplied origin unless native occurrence is separately proved.

**Prepared selection margins.** Let `B_i` be the reference's independently
computed pressure, `P0=max_i(abs(B_i))` and `Dmax=max_i D_i`. For `P0>0`,
the exact correlation between pair pressures gives

```text
p_(2i)=B_i+k*d_i,                p_(2i+1)=B_i-k*d_i,
P=max_i(abs(B_i)+k*abs(d_i)),
P0<=P<=P0+k*Dmax,
rho0_i=abs(B_i)/P0,
abs(abs(p_n)/P-rho0_i)<=E_i=k*(D_i+Dmax)/P0.
```

To prove the last bound, separate the numerator change, at most `k*D_i`,
from the denominator change, at most `k*Dmax`; use `P>=P0` and
`abs(B_i)<=P0`. This does not assume a fixed maximizing node. Intersect
`[rho0_i-E_i,rho0_i+E_i]` with `[0,1]` to obtain `rho_lo,rho_hi`.
All nonpressure Si terms are identical, so `abs(Si_n-Si0_i)<=gamma*E_i`.
Equivalently, the sharper endpoints are
`Si0_i-gamma*(rho_hi-rho0_i)` and
`Si0_i-gamma*(rho_lo-rho0_i)`; call them `S_lo,S_hi`.
These are sufficient enclosures, not independent freedom for Si and pressure.

Retained lag counts first determine the actual higher-priority requests:
increment both counts, then force Emission (AL) when its strict lag limit is
exceeded, otherwise Reception (EN) when its strict limit is exceeded. Matching
counts proves the same override; it does not waive prepared-state admission.
Without an override, sufficient conditions for the default selector are:

| Reference request | Sufficient comparison inequalities |
| --- | --- |
| Coherence (IL) | `S_lo>=si_hi` |
| Resonance (RA) | `S_lo>si_lo`, `S_hi<si_hi`, `rho_hi<=dnfr_hi` |
| Transition (NAV) | `S_lo>si_lo`, `S_hi<si_hi`, `rho_lo>dnfr_hi` |

These reproduce the actual precedence and strictness; in the intermediate
Si band, equality at the pressure threshold selects Resonance. The default
path does not apply the parametric selector's score, acceleration decision,
hysteresis or soft filters. It still admits and normalizes old acceleration.
A possible low-Si request, Mutation abstention or unresolved interval exits
this sufficient five-request certificate. It is not silently substituted by
Coherence. Original grammar records remain admitted, and live grammar still
checks each effective request. No threshold is retuned to obtain agreement.

**Live event margins.** At block `i`, previous complete pairs already have
their reference means. Consequently all live EN neighbor means, RA selected
means and paired phase inputs agree exactly. In particular, a zero selected
mean can stay exactly zero under these correlated form perturbations; an
arbitrary independent perturbation estimate would discard that information.
Live U3 subsets, resultant angles and capacity amplification branches agree
with the reference, including an inactive amplification boundary. They must
be recomputed at each reference prefix, rather than frozen at preparation.

Use `Y_i,A_i,U_i,H_i` from the shared event rows. The following sufficient
conditions admit both fine proposals:

- Resonance: `abs(M_i)>D_i`, `M_i*Y_i>0` and
  `abs(Y_i)>abs(A_i)*D_i`, with the common retained kind and admitted U3.
  These nonzero sign margins are sufficient, not necessary. A sign-neutral
  zero reference form or proposal does not itself give a positive transverse
  admission radius.
- Transition: `abs(B_i)>k*D_i` retains the common strict sign `sigma_i`.
  With `G_i=eta*(sigma_i*V_i-B_i)+xi_bar_i`, require
  `abs(G_i)>eta*k*D_i+Z_i`. This keeps both fine pressure changes nonzero;
  common pressure sign alone does not do so.
- Form rails: the incoming interval `M_i+[-D_i,D_i]`, event interval
  `Y_i+[-abs(A_i)*D_i,abs(A_i)*D_i]` and the flow interval below must all
  lie strictly inside the admitted rails. Reference post-event capacities
  and the later capacity proposals must also remain in their declared domain.

No pair transaction is inferred. The margins admit each singleton prefix;
failure of a later check does not imply native rollback of an earlier action.

**Contrast propagation.** Set `Z_i=0` outside Transition. Prepared pressure
is held through the event and form flow, so

```text
lambda_i=A_i-delta*U_i*H_i*k,
d_i_next=lambda_i*d_i+delta*U_i*xi_delta_i,
D_i_next=abs(lambda_i)*D_i+delta*U_i*Z_i.
```

The input half-difference is zero for the other four actions. For a finite
sequence of admitted calls, write `a_(t,i)=abs(lambda_(t,i))` and
`b_(t,i)=delta*U_(t,i)*Z_(t,i)`. Induction gives

```text
D_(N,i) = (product_(t=0..N-1) a_(t,i))*D_(0,i)
          +sum_(s=0..N-1) b_(s,i)*product_(t=s+1..N-1) a_(t,i).
```

An empty product is one. If throughout this finite admitted segment
`a_(t,i)<=a<1` and `b_(t,i)<=b`, then
`D_(N,i)<=a^N*D_(0,i)+b*(1-a^N)/(1-a)`. This is a conditional finite bound,
not a proof that the required branches persist indefinitely. Transition
differences supply forcing; even with contracting multipliers, a nonzero
forcing budget need not yield convergence to exact constituent equality.
Positive capacity alone does not give a horizon-independent rate: some
multipliers approach one as capacity approaches zero.

**Memory and capacity margins.** Let `J_i=Y_i-M_i`. If `J_i!=0`, the inequality
`abs(J_i)>abs(A_i-1)*D_i` makes both event jumps nonzero, preserving the
reference's reset status. If `J_i=0` and `A_i=1`, both jumps vanish identically.
If `J_i=0` and `A_i!=1`, an unknown nonzero contrast can change the reset mask;
the matching-history certificate abstains unless this is separately resolved.
It does not discard or replace the fine histories. Admitted histories then
restart/append by the actual rules, with the existing retention cap.

After the event, write `Q_i=H_i*k*D_i+Z_i`. The held fine pressures lie in
`q_bar_i+[-Q_i,Q_i]`. The adaptation qualification is certified true when
`S_lo>=si_hi` and `abs(q_bar_i)+Q_i<=EPS_DNFR_STABLE`; it is certified false
when `S_hi<si_hi` or `abs(q_bar_i)-Q_i>EPS_DNFR_STABLE`. Otherwise this bound
leaves qualification unresolved. The original pressure/Si correlation may
support a sharper independent proof; an unresolved enclosure is not an
observed execution failure.

Certified matching qualification and the retained individual counts give
the same next counts and due decisions as the reference. The immutable
capacity snapshot and neighbor means therefore give its same capacity
proposals. Reference pairing must itself satisfy the previously derived
`mu*(due_odd-due_even)*(W-U)=0` condition. Equal capacities alone never replace
the counts. Identical glyph traces and global gains give the same phase
controller load, branch and updates, preserving the paired phases. REMESH
abstention and all requested output/history records remain separately checked.

**Finite induction and meaning of failure.** At each call, certify prepared
selection first, then every live event, flow rails, temporal update and
adaptation. Update the reference and contrast bound only after these checks.
The shared means, phases, capacities and retained policy context then supply
the next induction premise. Store the earliest unresolved boundary if any;
do not extrapolate the formula beyond it. Such an outcome rejects this
certificate, not every possible collective description. Actual rejection,
changed selection and a conservative inconclusive bound remain distinct.

**An admitted box with unequal constituents.** Reuse the existing
[live Resonance prefix](NATIVE_OPERATOR_DYNAMICS.md#native-resonance-live-prefix) source with
`q=15/128`, means `(q,q,-q,-1/2,-q)`, capacities `(1/4,1/4,1,1,1)`, zero
phases and its original supplied AL/IL/EN context, lags, rates and zero
stability counts. Vary only five independent signed half-contrasts in
`[-1/4096,1/4096]`. This is a different domain from unrestricted independent
form perturbations: every neighboring complete pair retains its mean.

Let `epsilon=k/4096`. Exact represented-coefficient bounds give
`P0>0.0701`, `epsilon<0.000045` and ratio error below `0.0013` on the whole
box. For this source, direct ratio endpoints

```text
max(0,abs(B_i)-epsilon)/(P0+epsilon)
    <=rho_n<=min(1,(abs(B_i)+epsilon)/P0)
```

give RA ratios below `0.007`, RA Si strictly between `0.4307` and `0.4309`,
and IL Si above `0.94`. The original lags force neither AL nor EN. The actual
default requests are therefore RA/RA/IL/IL/IL uniformly, before any response
is evaluated. Fine corner/axis controls check the algebra independently;
these uniform rational inequalities, not corner sampling of non-affine Si,
admit the complete box.

The first live RA neighbor mean remains exactly zero. The second becomes
`-r*q/2`, so the first amplification stays inactive and the second active
throughout the box. Both fine proposals exceed `0.09`, retain their original
positive sign/kind, and have event jumps below `-0.018`. Coherence has no
form jump. All event/flow forms have absolute value below `0.6`, and all
post-event capacities remain interior. Adaptation qualification is false
at the RA targets because Si is too low, and false at IL targets because
held pressure magnitude exceeds `0.026`; thus all counts remain zero.
The matched traces preserve phase load/gains and zero phases. No innovation
is consumed, and the separately insufficient REMESH history still abstains.

The five contrast factors are positive and strictly below `0.992`
(approximately `0.8379833,0.8378694,0.9913168,0.9913168,0.9913168`). The
comparison consequently retains exactly the reference's terminal block means,
phases and capacities, with a strictly smaller bound on internal form
differences. Fine pressures, histories, rates and accelerations keep their
constituent association. This is a one-call ideal-real box certificate using
an existing source; it makes no claim that its successors retain those
requests indefinitely or that every box member is a newly acquired state.

**The zero-pressure boundary.** The positive-peak hypothesis cannot be
discarded from the general ratio estimate. Set phases to zero,
`V=(1/4,1/4,1,1,1)` and `M_i=-(b/k)*V_i`. All forms are negative and
interior, and the independent law gives `B_i=0` exactly. At zero contrast,
each low-capacity pair has `rho=0` and
`Si=alpha/4+beta+gamma`, approximately `0.43109`; without a lag override,
its default request is Resonance. With any sufficiently small nonzero `d_0`
alone, the peak is `k*abs(d_0)`, both block-zero ratios are one, and
`Si=alpha/4+beta`, approximately `0.37279`. Both requests at block zero become
Transition; other pairs retain their previous requests. The means, phases,
capacities and supplied context have not changed. This proves ideal-real selection discontinuity,
not an arbitrarily small binary64 witness or an executed alternative response.

Zero peak does not make every request unstable. A fixed lag override retains
its request when the other gates still admit. Likewise, if the nonpressure
Si contribution already satisfies `alpha*V_i/max(V)+beta*phase_score>=si_hi`,
the default request remains Coherence for every pressure ratio in `[0,1]`.
Uniform positive capacities and zero phases give this robust case under the
defaults. Request robustness alone still does not certify temporal or
adaptation branches. These two controls separate a source-specific decision
margin from an assumed universal continuity of the normalization policy.

The [detached retention controls](../tests/physics/test_native_transverse_retention.py)
retain production-execution guards and unchanged earlier source producers.
They check uniform enclosures, fine ordered prefixes, associated memory,
finite forced-contrast products and explicit boundary failures. No new native
trajectory, observation model, occurrence law or engine API is introduced.

<a id="native-joint-collective-calculus"></a>
### Joint form and capacity calculus across the adaptation exit

The five-operator description extends through unequal-capacity adaptation
if it retains the signed form and capacity contrasts with their constituent
association. The new term is the covariance of event capacity and held
pressure. It affects the collective mean, including its response to Transition
(NAV) innovations. This extends the earlier
[capacity-feedback identities](NATIVE_OPERATOR_DYNAMICS.md#native-capacity-feedback) and
[joint admission](NATIVE_OPERATOR_DYNAMICS.md#native-joint-admission) to the shared event calculus;
it does not turn an invertible coordinate change into a reduced model.

**Domain and sufficient coordinates.** Keep the
[shared calculus's execution premises](#native-collective-calculus), including
doubled C5, unit conductances, nonadjacent twins, whole-pair order, one direct
request per constituent, default preparation/policies, held event pressure,
live built-in zero Gamma and one Euler substep `delta=1/16`. Keep paired
primitive phases on their common arc, the common semantic kind, actual request
origin, grammar and all associated histories, rates, counters and streams.
Automatic REMESH must independently abstain. Supplied conditional transcripts
are not default-selection or fixed-seed occurrence certificates.

Replace paired capacities by

```text
x_(2i)=M_i-d_i,                  x_(2i+1)=M_i+d_i,
v_(2i)=V_i-e_i,                  v_(2i+1)=V_i+e_i.
```

Both capacities must be strictly positive and interior to their rails;
in particular `V_i>abs(e_i)`. Zero capacity is admitted by the engine under
its own contracts but is outside this theorem. All individual event and flow
forms, amplified capacities and adapted capacities remain unclipped and
admitted. Exact formulas below are ideal real over represented coefficients;
concrete floating execution also needs representable arithmetic. These four
coordinates reconstruct the pair's four scalar values exactly. No physical
degree of freedom has been eliminated by naming them collective moments.

**Fresh pressure and selection.** With `N_i(z)` the adjacent-block mean,
the independently supplied mixed law becomes

```text
B_i=w_phase*(N_i(theta)-theta_i)/pi_rep
    +k*(N_i(M)-M_i)+b*(N_i(V)-V_i),
C_i=k*d_i+b*e_i,
p_(2i)=B_i+C_i,                  p_(2i+1)=B_i-C_i,
P=max_i(abs(B_i)+abs(C_i)),       Vmax=max_i(V_i+abs(e_i)).
```

Neighbor contrasts cancel because each adjacent block contributes both
constituents with equal multiplicity. Own capacity contrast still enters
pressure. For the corresponding even/odd signs, prepared Si is

```text
phase_term_i=beta*(1-abs(N_i(theta)-theta_i)/pi_rep),
Si_(2i)=alpha*(V_i-e_i)/Vmax+phase_term_i
        +gamma*(1-abs(B_i+C_i)/(P or 1)),
Si_(2i+1)=alpha*(V_i+e_i)/Vmax+phase_term_i
          +gamma*(1-abs(B_i-C_i)/(P or 1)).
```

Both normalizers therefore require the joint fine association. Rebuild these
fields from admitted primitive state before selecting, retain the actual lag
priority and validate each live request. Equal means do not imply equal
constituent requests. The earlier form-only robustness bounds cannot simply
be reused with nonzero `e`: their capacity and pressure premises have changed.

**Shared event rows.** Reuse `Y_i,A_i,H_i` from the five-operator table:
event form is `Y_i-/+A_i*d_i`, and `H_i` multiplies prepared pressure.
Emission (AL), Reception (EN) and Coherence (IL) retain their original rows.
Reception and Resonance (RA) still use the ordered live neighbor form means.
Let

```text
f_i=1+a*I(abs(S_i)>T) if the request is RA, otherwise 1,
U_i=f_i*V_i,                    E_i=f_i*e_i,
q_bar_i=H_i*B_i+g_i,
q_delta_i=-H_i*C_i+zeta_i.
```

`U_i,E_i` are the event capacity mean and half-contrast. Outside Transition,
`g_i=zeta_i=0`. Resonance retains the same live whole-block U3 subset,
selected form mean `S_i` and phase proposal. Its two capacities acquire the
same factor, rather than becoming equal. Admit both sign identities
`(M_i-d_i)*(Y_i-A_i*d_i)>=0` and
`(M_i+d_i)*(Y_i+A_i*d_i)>=0`, the common kind, nonempty subset and every
individual post-amplification capacity. Mean identity alone remains
insufficient. The first twin does not change the second's live neighbors.

For default nonstrict random Transition with common strict pressure sign,

```text
sigma_i*B_i>abs(C_i),
g_i=eta*sigma_i*V_i+xi_bar_i,
zeta_i=eta*sigma_i*e_i+xi_delta_i,
abs(xi_bar_i)+abs(xi_delta_i)<=NAV_jitter.
```

The extra `eta*sigma_i*e_i` is required because each target uses its own
capacity. Define `G_i=eta*(sigma_i*V_i-B_i)+xi_bar_i` and
`J_i=eta*(C_i+sigma_i*e_i)+xi_delta_i`; the two pressure changes are
`G_i-J_i,G_i+J_i`. Both must be nonzero. Opposite or zero prepared signs
need their separate native branches, not this common-strict-sign formula.
Innovations remain the actual associated inputs; a mean draw cannot replace
their target-dependent contribution.

**Held flow and the missing mean term.** After every complete event pair,
the independent fine products give

```text
rate_bar_i=U_i*q_bar_i+E_i*q_delta_i,
rate_delta_i=U_i*q_delta_i+E_i*q_bar_i,
M_i_next=Y_i+delta*rate_bar_i,
d_i_next=A_i*d_i+delta*rate_delta_i.
```

Subtract associated old rate moments and divide by `delta` for acceleration.
The pressure contrast contains incoming `C_i`, not the post-event form or
capacity contrast. Current flow uses `U_i-/+E_i`; later adaptation must not
be inserted early. Stored Si remains the preparation value even if Resonance
has changed the live maximum capacity. A cache update is not a Si refresh.

For Emission, Reception, Coherence and Resonance, the mean rate is
`f_i*H_i*(V_i*B_i-k*e_i*d_i-b*e_i^2)`. This recovers the old Coherence
covariance and variance terms with the correct event-specific multiplier.
For Transition, the expanded mean rate is

```text
(1-eta)*(V_i*B_i-k*e_i*d_i-b*e_i^2)
+eta*sigma_i*(V_i^2+e_i^2)+V_i*xi_bar_i+e_i*xi_delta_i.
```

Thus capacity variance has a different coefficient in this action. In
particular, swapping the two innovations at fixed admitted state and fixed
innovation mean reverses `xi_delta_i`; the corresponding mean-form difference
is `2*delta*e_i*xi_delta_i`. Innovation means alone were sufficient only in
the paired-capacity sector. This is a conditional input-association result,
not a new stochastic law or a claim that both assignments occur at one seed.

The exact error of discarding the covariance term in one admitted flow is
`delta*E_i*q_delta_i`. If only `abs(d_i)<=D_x`, `abs(e_i)<=D_v` are retained,
the default nonnegative factors imply

```text
abs(mean_error)<=delta*f_i*D_v*(H_i*(k*D_x+b*D_v)+Zeta),
Zeta=0 outside NAV; Zeta=eta*D_v+Z_input for NAV,
abs(xi_delta_i)<=Z_input.
```

This is a useful conditional one-call bound that need not reconstruct each
constituent. It does not propagate the next mean domain, admit the next
request or assert that independent marginal extremes are jointly realizable.

**Phase, memory and adaptation.** The native coordinator reads paired phases,
actual full glyph histories and global gains, without a capacity-driven free
angular advance. The five admitted events and this coordinator therefore
preserve phase pairing despite unequal capacities. This is a property of the
declared native law; the separate capacity-driven phase model has different
premises. Common-arc/resultant admission still applies.

Temporal event jumps remain `Y_i-M_i-/+(A_i-1)*d_i`; retain each actual reset,
sample value, clock and old rate. Compute qualification from individual fresh
Si and held post-event pressure, update each count, then obtain due flags
`a_minus,a_plus`. Set `a_bar=(a_minus+a_plus)/2`,
`a_delta=(a_plus-a_minus)/2` and `W_i=N_i(U)` from the immutable event-capacity
snapshot. The final writer is exactly

```text
V_i_next=U_i+mu*(a_bar*(W_i-U_i)-a_delta*E_i),
e_i_next=E_i+mu*(a_delta*(W_i-U_i)-a_bar*E_i).
```

Fixed-point and hull guards remain in force. Unequal due flags can create
capacity contrast from zero. When both are due the contrast becomes
`(1-mu)*E_i`; when neither is due it remains `E_i`. Equal flags alone do not
erase it. Counts do not reset after a successful write.
The next preparation uses these new capacities and flow forms, not the old
stored pressure or a capacity substituted into the preceding flow rate.

**Admitted entry and limits.** The existing positive
[joint-admission source](NATIVE_OPERATOR_DYNAMICS.md#native-joint-admission) supplies a conditional entry
through this boundary: its first call's node-two/node-three due split creates
`e_1=mu*epsilon/4`, with `epsilon=1/1024`. Fresh Si/pressure and retained lag
and grammar state admit its second all-Coherence call. Applying the joint
rows to that already-declared source recovers its independent fine forecast,
including active capacity writes, associated rates/history and the nonzero
mean gap between opposite internal form assignments. No response is rerun
to manufacture a new preparation or occurrence claim.

The [joint-calculus controls](../tests/physics/test_native_joint_collective_calculus.py)
compare independent fine sweeps with all five joint moment rows, live phase
and neighbor inputs, held pressure and post-flow adaptation. They retain the
actual-source continuation separately from supplied mixed transcripts, and
test the capacity-innovation association and covariance-error bound. Production
runtime, pressure, selector, operator and integrator evaluation are blocked.
The result supplies sufficient exact coordinates across the former capacity
exit. It neither proves mean-only closure, a smaller complete NFR state,
unlimited retention, autonomous formation nor physical identification.

<a id="native-lossy-collective-enclosure"></a>
### Bounded prediction from a lossy collective observation

The joint moment identities permit a genuinely lossy one-call observation:
retain intervals for the two means and upper bounds for the two internal
contrasts, without retaining their signed values. The prediction is a set of
possible successors. Its width and unresolved decisions matter separately;
an enclosing interval is not a unique response or a smaller complete NFR.

**Preparation and information boundary.** Keep the complete law, support,
whole-pair order and successful-call premises of the
[joint calculus](#native-joint-collective-calculus). For each pair retain

```text
M_i in X_i,             V_i in N_i,
abs(d_i)<=D_xi,         abs(e_i)<=D_vi.
```

Here `N_i` is a capacity interval; write `neighbor_i(z)` for the adjacent-block
mean. Retain exact paired primitive phases and their inherited common arc,
actual supplied grammar/lag context, node-associated sets of possible counts,
global gains and target-associated innovation constraints. Old rates and
temporal sample values may be represented by associated envelopes. Any
nonempty admitted history ends at its fine lift's actual current form;
empty histories remain explicit and supply no measured trigger evidence.
This is a premise on the family, not permission to invent independent samples.
The predictor consumes neither exact signed contrasts nor exact history
endpoints from which to recover them. Fine lifts are evaluated only after
freezing the observation, protocol and output budget.

The proposed requests retain their origin. For actual default occurrence,
certify the fresh selection and lag precedence uniformly. A supplied paired
transcript instead asserts the requests and still has to pass every live
grammar, identity, nonidentity, support and state gate. Passing such a
transcript does not prove that the default selector would choose it. Supplied
selector requests remain subject to the batch's Emission/Reception lag
overrides; the retained effective transcript must agree with those overrides.

**Fresh normalization from retained data.** Form the interval pressure mean
`B_i` from `X,N` and the exact phase row. Let `minabs(J)=min_(z in J)|z|` and
`maxabs(J)=max_(z in J)|z|`. For the nonnegative default pressure weights,

```text
R_i=k*D_xi+b*D_vi,
p_n in B_i+[-R_i,R_i],                   n in {2i,2i+1},
P_lo=max_i minabs(B_i),
P_hi=max_i(maxabs(B_i)+R_i),
Vmax_lo=max_i N_i.lo,
Vmax_hi=max_i(N_i.hi+D_vi).
```

These bounds follow from the exact correlated peaks
`P=max_i(|B_actual_i|+|k*d_i+b*e_i|)` and `Vmax=max_i(V_i+|e_i|)`,
where the actual scalar pressure mean `B_actual_i` belongs to `B_i`.
A contrast radius supplies no positive lower bound for its realized value.
When `P_lo>0`, divide the pressure magnitude enclosure by the peak bounds
and intersect the ratio with `[0,1]`. If zero peak remains possible, use
`[0,1]`, except that `P_hi=0` gives the exact ratio zero. Bound each capacity
ratio from `N_i+[-D_vi,D_vi]` and the positive capacity-maximum interval.
Insert these bounds and the exact phase score into the fresh Si formula.
This admits requests with the actual strict/inclusive thresholds; it does
not freeze a maximizing node or assume affine normalized diagnostics.

**Ordered event admission.** Use interval arithmetic on the shared event
rows, updating live form-mean intervals in order. Reception (EN) consumes
the live adjacent means. Resonance (RA) recomputes its whole-block U3 subset
from the live paired phases and its selected form-mean interval `S_i`.
Amplification is inactive if `maxabs(S_i)<=T`, active if `minabs(S_i)>T`.
Otherwise this fixed-branch certificate is unavailable; it must not choose
one branch from a fine witness. The same requirement applies to unresolved
request, sign and identity gates. An unresolved temporal reset instead needs
associated alternative history records or an unavailable history certificate.
A more elaborate event union is possible but is not supplied by a silently
selected branch.

For Transition (NAV), admit a common strict sign with
`min(sigma_i*B_i)>R_i`. With retained innovation mean interval `Ibar_i` and
half-difference bound `Z_i`, require
`maxabs(Ibar_i)+Z_i<=NAV_jitter`. Its nonidentity changes lie in

```text
G_i+[-Jmax_i,Jmax_i],
G_i=eta*(sigma_i*N_i-B_i)+Ibar_i,
Jmax_i=eta*(k*D_xi+(b+1)*D_vi)+Z_i.
```

Excluding zero admits both changes. This sufficient symmetric bound is not
an assertion that every mean/difference combination is realizable by a
fixed seed. Keep the original target association and declared input domain.
Both Resonance sign identities and every individual incoming, event, flow
and capacity rail require uniform admission. A failed enclosure can be
conservative; it is not itself a native rejection witness.

**Mean and contrast bounds.** Let `Y_i,A_i,H_i,f_i` be the admitted interval
event mean and exact branch factors. Write `n_i=1` for Transition and zero
otherwise, with `Ibar_i=0,Z_i=0` outside Transition. Define

```text
Qbar_i=H_i*B_i+n_i*eta*sigma_i*N_i+Ibar_i,
Cov_i=f_i*[-H_i*k*D_xi*D_vi,H_i*k*D_xi*D_vi]
      +f_i*(n_i*eta*sigma_i-H_i*b)*[0,D_vi^2]
      +f_i*[-D_vi*Z_i,D_vi*Z_i],
X_i_next=Y_i+delta*(f_i*N_i*Qbar_i+Cov_i),
D_xi_next=maxabs(A_i-delta*f_i*H_i*k*N_i)*D_xi
    +delta*f_i*maxabs(H_i*(B_i-b*N_i)
                     +2*n_i*eta*sigma_i*N_i+Ibar_i)*D_vi
    +delta*f_i*maxabs(N_i)*Z_i.
```

Terms multiplied by `n_i=0` vanish without requiring a pressure sign for
the other actions. The covariance includes both the signed capacity-variance
coefficient and capacity-input uncertainty. These equations enclose the
exact fine products algebraically. Replacing them by a product of mean
capacity and mean pressure would discard a consumed term. Independent
interval extrema provide an enclosure, not simultaneous attainable values.

**Associated policies and temporal evidence.** Use the prepared Si enclosure
and held post-event pressure to classify each stability qualification as
true, false or unresolved. Update each node's count set accordingly, then
derive its possible due flags using the original threshold. With
`U_i=f_i*N_i`, `D_Ei=f_i*D_vi`, `W_i=neighbor_i(U)`, each possible due pair
gives `a_bar=(a_minus+a_plus)/2`, `a_delta=(a_plus-a_minus)/2` and

```text
N_i_candidate=U_i+mu*a_bar*(W_i-U_i)
              +[-mu*abs(a_delta)*D_Ei,mu*abs(a_delta)*D_Ei],
D_vi_candidate=abs(1-mu*a_bar)*D_Ei
               +mu*abs(a_delta)*maxabs(W_i-U_i).
```

Take the capacity hull and largest contrast radius over those pairs. This
reuses the [joint-admission owner](NATIVE_OPERATOR_DYNAMICS.md#native-joint-admission)
and preserves unresolved count outcomes instead of replacing them by false.
The Cartesian enumeration is a safe outer bound; shared pressures may forbid
some enumerated combinations. Recheck positive capacity, rails and the
fixed-point/hull writer premises. Adaptation follows the current form flow.

The admitted event jumps determine each temporal reset. A reset sample lies
in the individual event-form envelope; otherwise preserve the associated
prior sample envelopes. Append the individual flow endpoint envelope with
the original timestamps and retention cap. Old-rate envelopes and the new
rate enclosure bound acceleration. They do not prove arbitrary combinations
of samples and rates consistent, or admit a future Mutation trigger.
Paired phase continuation uses its own retained glyph-load/gain evidence,
without inserting capacity as a free phase advance. Automatic REMESH still
requires its independently declared abstention. This output is a scoped
observation certificate, not a complete restart record.

**Frozen one-call classes and budget.** The
[detached enclosure controls](../tests/physics/test_native_collective_enclosure.py)
freeze three observations before evaluating any fine lift. Keep the shared
default represented coefficients, source/support order `0..9`, clock `t=1`,
one call with `delta=1/16`, zero built-in Gamma and the original no-REMESH
premise. The source convention is common kind `seed`, empty incoming temporal
histories, zero old acceleration, the admitted EN trace with prior Coherence,
and associated rates, debt, ordinals, lags and count sets. Old rates remain
declared state; empty history supplies no measured trigger evidence.

1. **Supplied five-action transcript.** Use Emission (AL), Reception (EN),
   Coherence (IL), Transition (NAV), Resonance (RA) in block order. Retain mean
   intervals of radius `2^-16` around the joint-calculus transcript's form and
   capacity means, its absolute form/capacity contrast bounds and paired phases
   `(0,1/8,1/4,1/8,0)`. Innovation constraints are
   `abs(xi_bar)<=NAV_jitter/4`, `abs(xi_delta)<=NAV_jitter/2`, a rectangle inside
   the admitted amplitude diamond. This is a supplied input domain, not a
   fixed-seed realization. All five event maps are covered conditionally;
   default occurrence is not claimed for this transcript.
2. **Actual default selection.** Reuse the live-prefix source means
   `(q,q,-q,-1/2,-q)`, `q=15/128`, capacity means `(1/4,1/4,1,1,1)`, zero phases
   and associated context/rates. Keep these means exact but discard all ten
   signed internal form/capacity contrasts, retaining only
   `D_xi=D_vi=1/4096`. Fresh interval Si/ratios admit RA/RA/IL/IL/IL uniformly.
   The first live RA selected mean stays exactly zero and does not amplify;
   the second is `-r*q/2` and amplifies. Exact means here are substantive:
   an independent mean envelope around the first zero signal could leave its
   amplification branch unresolved. No adaptation qualifies in this class.
3. **Actual ambiguous adaptation.** Reuse the joint-admission observation with
   all form means `1/8`, capacity means `(1+epsilon,1,1,1,1)`,
   `epsilon=1/1024`, `D_x0=59/8192`, other form radii zero, and zero capacity
   radii. Node zero's incoming count is zero; all other counts are four.
   Default requests are all Coherence, while node one's due decision remains
   unresolved. The enclosure must preserve both outcomes.

All three must have maximum successor form-mean interval width below `1/200`,
form half-contrast radius below `1/20`, capacity-mean interval width below
`1/100`, and strict individual source/event/output rails. These budgets are
fixed before fine controls, not selected from their errors. The shared
128-bit outward dyadic interval kernel encloses ideal-real arithmetic over
represented coefficients. Exact fine containment adds no tolerance; this is
not a universal binary64 execution certificate.

The outward calculations give the following displayed approximations. The
exact interval endpoints in the controls, not rounded table entries, supply
the certificates.

| Frozen class | Largest form-mean width | Largest form radius | Largest capacity-mean width | Largest capacity radius |
| --- | --- | --- | --- | --- |
| Supplied five-action transcript | `0.000352828` | `0.0311475` | `0.0000317319` | `0.00812335` |
| Actual RA/RA/IL/IL/IL | `1.64472e-9` | `0.000243106` | `0` | `0.000253855` |
| Actual ambiguous all-IL | `5.87748e-39` | `0.00713955` | `0.000048828125` | `0.000048828125` |

The exact uniform bound on the actual RA/IL class's deviation from the
conditional product-of-means form prediction is below `1e-9`
(approximately `9.35272e-10`). Its mean enclosure is narrow despite ten discarded signed
coordinates. Internal capacity differences can grow under RA amplification;
form-radius contraction alone is not complete-state contraction.

For the ambiguous class, two admissible lifts with opposite `d_0` have the
same retained observation but distinct node-one adaptation decisions. Their
terminal first-block capacity means differ by `mu*epsilon/2`. Any single
common capacity-mean prediction must err by at least `mu*epsilon/4` on one
of them. This is an information obstruction, not merely interval overestimate;
the enclosing union remains useful without selecting a hidden branch.

The controls evaluate a fixed 32/4/2 list of exact fine lifts after the three
coarse certificates. Their finite inclusion checks are separate from the
uniform interval proof. Boundary controls preserve a possible zero pressure
peak, unresolved Transition sign and actual lag-override priority. A single
shared phase-origin interval retains phase correlations after coordination;
individual temporal envelopes retain `right=left+delta*rate` at times
`1,17/16`. Arbitrary Cartesian combinations are not declared valid histories.
These results establish bounded one-call prediction and its limits. Reusing
the successor description for another call still needs its own admission;
complete collective NFR identity, formation and physical identification remain
separate questions.

<a id="native-lossy-collective-composition"></a>
### Finite composition with the propagated collective memory

A useful one-call enclosure need not supply an admissible next input. The
second call consumes the evolved physical observation, histories and policy
state. Restoring the initial context or reconstructing a representative fine
state would answer a different question. This result studies two calls of the
existing actual Resonance (RA)/Coherence (IL) class using only its first
abstract successor.

**Composition criterion.** Let `A` contain the admitted mean intervals,
contrast bounds and associated context. Write `Lift(A)` for the fine states
that satisfy these bounds and their retained relations. In particular, an
individual history endpoint and the live form are the same variable, not
independent values drawn from overlapping intervals. The latest flow
relation's velocity is also that constituent's retained rate coordinate;
equal marginal rate intervals alone do not establish this association.
Configuration, support,
node order, clock and the actual complete law remain fixed. If the detached
observer `F` proves

```text
every s in Lift(A) admits the declared complete call T,
T(Lift(A)) is a subset of Lift(F(A)),
```

and proves the same statements again on `F(A)`, then
`T(T(Lift(A)))` is contained in `Lift(F(F(A)))`. This follows by set inclusion
and induction. The second observer may enclose more states than are reachable;
all consumed decisions must still be admitted on its declared family. Neither
one representative lift nor agreement of marginal means proves this inclusion.
The [exact ordered-action criterion](STRUCTURAL_OPERATORS.md#collective-grammar-descent)
has stronger equality obligations; a bounded successor set does not establish
an exact smaller node or a uniquely determined collective response.

**Source and numerical protocol.** Retain the previous actual RA/RA/IL/IL/IL
observation: means `(q,q,-q,-1/2,-q)`, `q=15/128`, capacity means
`(1/4,1/4,1,1,1)`, zero paired phases and form/capacity contrast radii
`1/4096`. All ten signed contrasts remain discarded. Use its unchanged
associated rates, grammar, counts, kinds and empty initial histories, default
represented coefficients and no additional input. The fixed horizon is two
complete calls from `t=1` with `delta=1/16`, ending at `t=9/8`.

Before evaluating fine controls, require every final form-mean and
capacity-mean interval width below `1e-7`, every form contrast radius below
`1/4096`, every capacity contrast radius below `1/3500`, and strict individual
rails throughout. These are cumulative endpoint budgets, not an extra local
tolerance or a budget reset after the first call. The shared outward dyadic
128-bit kernel encloses ideal-real arithmetic over represented coefficients.
Fine inclusion is exact, without an added tolerance. Keep the already frozen
one-call producer unchanged.

**What the successor must retain.** The second preparation reads the first
output's mean/radius bounds and rebuilds pressure, both normalization peaks
and Si. Stored first-call pressure or Si cannot stand in for this preparation.
The retained old-rate interval gives the next acceleration through
`(new_rate-old_rate)/delta`. The old-acceleration envelope also remains admitted
although the default selector does not use acceleration to choose its mark.
An unknown diagnostic is not replaced by zero.

Carry each constituent's trace, persistent U2 debt, prior Coherence flag,
operator ordinal, kind, lag counts and stability-count set. The actual batch
increments lags before applying its Emission/Reception priority. Traces from
the first call already contain the new event. For this sector, Resonance
leaves debt unchanged and Coherence reduces it by one down to zero. Replacing
Coherence's decrement by an unconditional reset would be incorrect outside
the source's zero/one debt stratum.
Runtime invocation ordinals have their own provenance and are not physical
time or constituent operator counts. With the source's initially absent
runtime marker, the two calls use ordinals zero and one and leave the next
ordinal two; constituent operator ordinals instead advance `3 -> 4 -> 5`.

Retain the first coordinator's gains and common phase-origin observation,
with its relative phases. In this source every phase stays exactly zero, so
the shared origin is exactly zero as well. The gain recurrence nevertheless
continues from the first output rather than restarting at configured initial
gains. The complete updated traces contain no disruptors; the stable branch
has phase resultant one and zero disruptor load. This scoped composition
does not establish a general phase-rotation quotient for dispersed phases.

The second event is applied to a state whose recorded endpoint already
matches its live form. A uniformly nonzero Resonance form jump at `17/16`
restarts that constituent's temporal record at the new event form; the
subsequent flow adds its endpoint at `9/8`. Coherence leaves form unchanged
at the event and preserves its first two samples, then appends the new one.
Thus the final RA records have times `(17/16,9/8)` and the IL records have
times `(1,17/16,9/8)`, within the four-sample limit. Each retained flow segment
keeps its actual sample/rate association. These predicted envelopes are not
measured evidence for a future Mutation trigger.

The abstract successor retains envelopes for the successive whole-support
snapshots written by native REMESH-history sampling. The admitted runtime
has no callback adding stability samples;
its initially empty `stable_frac` remains empty, so the automatic REMESH
policy abstains on both calls before reading delayed form values. Declaring
that abstention does not remove REMESH from the law or promise it indefinitely.

**Re-admitting the second call.** Use the same interval pressure/Si,
live-neighbor, covariance and adaptation identities as the one-call owner.
The second call's input is its propagated observation, never an initial-state
adapter with substituted traces, time, acceleration or histories. Both fresh
requests and live operator gates must pass over the whole abstract family.
The actual second requests remain RA/RA/IL/IL/IL. The first live selected-mean
interval is now strictly negative, separated from the amplification trigger;
the first RA therefore changes from inactive amplification to active
amplification. The second RA also amplifies. Their live inputs are recomputed
in order; freezing the first-call amplification pattern would omit a real
event-capacity factor from the second form flow.

Both RA targets remain below the Si stability threshold. The IL targets'
held pressures stay outside the adaptation band. Hence every qualification
is false and each stability count remains zero. This is a re-admitted
decision, not an assumption that unchanged counts waive capacity adaptation.
The amplified capacities are used by the current held-pressure flow before
the later writer. Covariance continues to affect means despite the small
enclosure widths.

**Concrete bounds and controls.** The second preparation has pressure peak
between `0.0691967` and `0.0692554`. RA pressure ratios are below `0.032`;
their Si intervals remain strictly inside the default middle band, while
the other blocks have Si above `0.941`. The two ordered RA selected-mean
intervals are contained respectively in `(-0.010904808,-0.010904804)` and
`(-0.018872395,-0.018872390)`, strictly beyond the magnitude trigger. These
whole-family bounds precede the fine controls.

At `t=9/8`, the largest form-mean interval width is below `3.094e-9`, the
largest form contrast radius below `0.000242093`, and the largest capacity
contrast radius below `0.000263956`. Capacity means remain point intervals,
approximately `(0.259947184,0.270290154,1,1,1)`. The exact outward endpoints,
not these rounded displays, verify every predeclared budget. Increasing
capacity contrast under amplification remains compatible with decreasing
form contrast; no complete-state contraction is claimed.

The [composition controls](../tests/physics/test_native_collective_composition.py)
freeze the source, complete law, both abstract outputs and budgets before
evaluating four independent exact paths. Their initial contrasts are
`d_i=s_d*(-1)^i/4096`, `e_i=s_e*(-1)^i/4096`, with independent signs
`s_d,s_e in {-1,1}`. Each second fine call continues its own first endpoint,
rates and history. Their inclusion checks validate the implementation of
the interval argument; four paths alone do not prove the continuous domain.
Runtime, production pressure, selector, operator and integrator evaluations
remain disabled in this new owner. Earlier producers remain unchanged.

The observer binds each current-form and rate coordinate to its corresponding
history relation. Missing, stale or reassigned records reject this detached
admission; it never reconstructs hidden values or restarts with initial
defaults. An unavailable certificate is not an executed failed runtime call
and consumes no native invocation ordinal. Reusing initial phase gains could
leave the zero-phase endpoint unchanged while giving the wrong policy state;
the controls therefore check the actual gain recurrence separately. Likewise,
restoring initial rates changes the acceleration calculation even when mean
form remains enclosed. These are consumed-state distinctions, not extra
physical laws.

The result establishes reusable bounded collective information across this
two-call source class, including a changed live event branch. It retains
constituent-associated policy and temporal state; it does not establish
indefinite retention, a minimal autonomous NFR, newly formed organization,
general dispersed-phase closure or physical identification.

<a id="native-relative-phase-descent"></a>
### Relative phases for the complete native call

The common phase origin can be removed from the admitted five-operator map
without removing its relative phases or constituent-associated context. This
is an exact symmetry reduction of the declared ideal circular model. It is
not a proof that an arbitrary rotated binary64 graph executes identically,
or that the remaining state constitutes a smaller autonomous NFR.

**Model and equivalence.** Retain the joint-calculus domain: doubled C5,
consecutive nonadjacent twins, paired primitive phases, positive individual
capacities, admitted signed forms and common kind, fresh default pressure/Si,
direct ordered glyphs, zero live Gamma and held Euler flow. Keep all rails,
grammar, clock, history and independently inactive REMESH premises. Every
effective pair request must agree. Its origin is either the fresh admitted
default selector or an explicitly supplied transcript; these are different
inputs. Exclude anchored callbacks, additional phase writers and optional
tracking that consumes absolute phases.

For this theorem the circular operations use mathematical radians and their
true period `2*pi`. Configured finite coefficients, including the pressure
normalization `pi_rep`, retain their represented values as exact real
parameters. On an inherited common lift of width less than pi, all selected
phasor targets use the representative in that lift. Default Resonance phase
coupling lies in `[0,1]`; coordinator gains are nonnegative with sum at most
one. Their convex proposals stay in the same lifted hull. This separates the
ideal circular law from modulo normalization with a represented period and
from materialized trigonometric arithmetic.

Write `theta_i=beta+r_i`, with `beta=theta_0` and `r_0=0`. Let `H` retain the
consumed non-phase state: individual rates and acceleration, kinds, associated
temporal samples, grammar traces/debt/prior-Coherence flags, operator and
runtime ordinals, lags, stability counts, phase gains and controller history,
random inputs/stream progress and REMESH history/abstention evidence. The
complete law, support, node order and clock are also retained. Define

```text
Q(s)=(M,d,V,e,r,H),
rho_a(s): theta_i -> theta_i+a for every constituent,
          all retained non-phase state and associated inputs unchanged.
```

The equivalence is one shared rotation of every participating primitive
phase, not independently rotating blocks or deleting their relative state.
For the admitted successful call `T`, its consumed semantic-state map obeys
`T(rho_a(s))=rho_a(T(s))`. Consequently `Q(T(s))` depends only on `Q(s)` and
the retained input, defining `T_relative(Q(s))=Q(T(s))`. A declared coherent
lift is a chart for this statement; raw centered extrema need not describe
that chart after rotation. This quotient removes one common angle from the
five paired phases. It retains the joint form/capacity moments and all their
constituent associations. Arbitrary graph dictionaries, absolute-angle debug
logs and untransformed trigonometric caches are not this state equivalence.

**Fresh preparation and live events.** The pressure phase term is
`w_phase*(N_i(r)-r_i)/pi_rep`. The other channels, both normalization peaks,
individual absolute pressures and capacity ratios are unchanged. The Si
phase term also uses relative offsets. Thus fresh default selection, its
incremented lag priority and all grammar decisions have the same admitted
inputs. Stored old acceleration still requires admission even when the base
selector does not use it to choose a mark.

Emission (AL), Reception (EN), Coherence (IL) and Transition (NAV) on this
direct native path have no phase write. Their form/pressure actions depend
only on retained invariant quantities. Reception still reads live ordered
neighbor forms; Coherence still leaves primitive phase unchanged. Transition
retains each constituent's own capacity and innovation, including their
association and stream progress.

For Resonance (RA), initialize live offsets `y=r`. At every ordered prefix,

```text
J_i={j in {i-1,i+1}: abs(y_j-y_i)<=L},
S_i=mean_(j in J_i)(live Y_j),
a_i=mean_(j in J_i)(y_j),
y_i <- (1-c)*y_i+c*a_i.
```

These expressions use the admitted common lift. The selected adjacent
blocks have equal multiplicity, so their circular mean is exactly the
lifted midpoint (or the one selected phase). Nonempty selection and the
open-semicircle domain exclude an undefined resultant. Recompute `J_i` at
each live prefix: it may gain a neighbor during the event batch. The scoped
[compatible-edge retention](NATIVE_OPERATOR_DYNAMICS.md#native-resonance-compatible-edge-retention)
result excludes losing an already compatible edge under these default paired
events; it does not replace the live subset with its initial value. A common
rotation preserves every U3 inequality, both fine sign/kind identity gates,
the selected signed form mean and the capacity-amplification decision.
The phase proposal gains precisely the same common rotation. Induction over
the ordered pairs proves event equivariance without freezing their subsets.
This proof does not admit arbitrary unequal requests inside a pair or supply
rollback for a rejected sequential prefix.

**Coordinator and its retained policy.** After the events, write
`l_i(y)=N_i(y)` and choose the global target representative `b(y)` in the
same common interval. The global resultant magnitude is

```text
R(y)=abs(sum_i exp(1j*y_i))/5.
```

The twin multiplicity cancels in this expression. Rotating all phases
multiplies the phasor sum by one unit complex factor, so R is invariant.
Disruptor load depends on the actual updated traces and is also invariant.
The configured controller therefore selects the same branch, updates the
same incoming gains and appends the same state/R/load/gain records. Its
global angle need not be calculated to advance the relative coordinates:

```text
r_i_next=(1-kG-kL)*(y_i-y_0)+kL*(l_i(y)-l_0(y)),
beta_next=beta+(1-kG-kL)*y_0+kG*b(y)+kL*l_0(y).
```

The first row is closed on retained relative phases and policy state;
`r_0_next=0`. The second row reconstructs an absolute lift only if its
initial origin is supplied. Global coupling still changes relative dynamics
through its coefficient; cancellation of the common target does not remove
global coupling. Nor does it remove R or the gain history needed by the next
invocation. The policy acts per call, not per unit of physical time.

**Conditional phase-resource consequence.** Let `D_j` be the lifted phase
diameter before call j. Every admitted Resonance update is a convex move
inside the current hull, including a move with a newly compatible neighbor.
The other four events, held form flow and capacity adaptation leave phase
unchanged. The coordinator then gives

```text
D_(j+1) <= (1-kG_j)*D_j,
D_N <= D_0*product_(j=0..N-1)(1-kG_j) <= q^N*D_0,
q=1-m < 1,  m=the represented default adaptive global-gain floor.
```

The last bound assumes the default enabled adaptive policy. The hull step
and the [coordinator diameter theorem](nodal/PRIMITIVE_PHASE_CLOSURE.md#native-phase-contrast-budget)
prove the result by induction for any successful segment remaining in the
five-action domain. Unequal evolving capacities do not defeat it because
they supply no free phase advance on this native path. The phase chart is
preserved; admission of later identities, rails, selections and histories
is a separate obligation. Infinite successful invocation count is not proved.

This extends the earlier uniform-capacity reachable IL/AL/EN restriction to
the admitted live five-action class. Along an infinite admitted ideal sequence
with the gain floor, the diameter tends to zero. This excludes a uniform
positive lower bound, not positive diameter at every finite invocation:
`D_N=q^N*D_0` is positive for every finite N when `D_0>0` and `0<q<1`.
It does not exclude persistent form/capacity organization or finite identity.
Binary64 defects retain their separate ledger. An identity requiring a fixed
positive phase margin eventually leaves that identity domain if the admitted
sequence continues. Maintaining such a margin would require a justified
departure from these contraction premises, not necessarily strict diameter
increase after a complete call. The [observable admission](#native-canonical-observable-admission)
separates that obligation from Mutation admission and phase recharge.

**Flow, adaptation and memory.** Every event form, capacity and held pressure
is unchanged by the rotation. The nodal products, covariance contribution,
form endpoints and accelerations are therefore unchanged. Event jumps reset
exactly the same individual temporal records, and each history endpoint and
flow-rate relation retains its association with the live constituent. The
clock is unchanged. This is not a phase-derived clock or a time change.

Adaptation reads the unchanged prepared Si and post-event held pressure,
updates the same stability counts and due flags, and uses the same immutable
capacity snapshot. Thus both capacity moments, including any contrast
created by unequal due flags, are invariant. Grammar records, lags, ordinals,
whole-support EPI snapshots and the independently empty stability series have
the same updates. These component identities complete the whole-call proof;
the earlier pressure or coordinator symmetry alone did not establish it.

Caches containing absolute trigonometric components are not invariant
physical records. They must be transformed consistently or rebuilt from the
admitted primitive phase. The quotient does not authorize reusing a stale
cache, replacing retained memory with initialization defaults, or treating an
absolute phase log as an invariant observation.

**What the quotient cannot predict.** Signed forms, capacities, pressures,
Si, event decisions, relative phases, R and the retained non-phase histories
factor through Q. An absolute phase reading or the signed real/imaginary
components of a phasor sum generally do not. A callback selecting a mark from
an absolute phase threshold introduces an external reference; it needs that
reference as retained state/input and its own covariance argument. The
present exclusion is substantive, not a declaration that every callback is
unphysical.

Relative phase cannot be discarded with the origin. Keeping the same forms,
capacities and context but changing a nonconstant `r` changes fresh phase
pressure; it can also change Si, live U3 subsets and the controller. Likewise,
restarting incoming phase gains can preserve a synchronized endpoint while
giving the wrong next controller state. Origin reduction does not resolve
these distinct state-sufficiency obligations. A future physical sensor may
require an external phase reference even when the internal model does not.

**Numerical boundary.** A binary64 implementation is not an exact action of
the continuous rotation group. Forming `float(theta+a)` can change a gap
before any operator runs. Subsequent trig reduction, branch representatives,
represented-period normalization and summation introduce separate defects.
The runtime's signed circular difference and centered phase storage also use
different endpoint conventions. Even exact arithmetic with a represented
`2*pi_rep` modulo cannot be identified with true-radian periodicity across
all branch crossings. The regular lifted identities above do not waive any
of these implementation boundaries.

For a claimed numerical comparison, retain the chosen lifts, backend and
every realization defect, with independent margins for U3, resultants,
selection, identity, amplification and controller thresholds. At a decision
boundary, a small angular discrepancy can change a discrete branch; an
endpoint tolerance alone does not establish equivalent calls. This section
supplies neither a uniform roundoff bound nor permission to replace the
native executor by an origin-centered implementation.

**Frozen controls.** The
[detached owner](../tests/physics/test_native_relative_phase_descent.py)
prepares two origin-free forecasts before six independent fine lifted
controls: one complete call for each source at origins `-7/4,0,5/2`, with
exact comparisons and no tolerance. Both sources have nonzero relative phase.
The source, law, context, input, horizon and output protocol digest is
`bad6429813ae9fdcd07b8cd904862429a26bbd2f518818e37585ff6b6540e1bb`.

The joint five-action transcript retains its supplied request origin: fresh
default selection on that source would instead give Coherence everywhere.
Its updated trace load is `1/10`; exact cosine-tail bounds certify the stable
controller branch. Its phase diameter falls from `1/4` to approximately
`0.212219`. The separate unchanged
[neighbor-gain source](NATIVE_OPERATOR_DYNAMICS.md#native-resonance-neighbor-gain)
has actual fresh RA/RA/IL/IL/IL requests. Its live RA subsets are first `(4,)`,
then `(0,2)` after the earlier phase change. Exact resultant-square bounds
inside `(0.2547626,0.2547695)` and zero disruptor load certify the dissonant
controller branch. The event diameter `2.998046875` contracts to approximately
`2.165845` or less. Rounded displays are not the exact comparisons.

Both complete-call sources have no capacity qualification; a separate
admitted static adaptation control checks dependence on the retained input
and unequal due flags. Prefix hulls, fresh/default requests, identities,
amplification, held products, reset/debt masks and associated temporal/rate
records have their own checks. Absolute readings, omitted relative phases
and binary64 materialization supply distinct counterexamples to stronger
claims. No new native scientific execution occurs and earlier producers stay
unchanged. The proof supplies the all-origin ideal symmetry; finite controls
check its algebra and retained-state implementation. No physical
identification, formation or indefinitely admitted execution follows.

<a id="native-mutation-renewal-admission"></a>
### Mutation admission and the complete phase-renewal obligation

Mutation (ZHIR) supplies a phase-changing event that is absent from the
five-action hull argument. Its request, temporal eligibility, grammar,
physical proposal and the remaining live call have independent obligations.
The following bounded analytical study separates them. It proves an
admitted Mutation prefix followed by a later Resonance identity obstruction,
and a distinct conditional renewal map whose simple history acquisition is
excluded. These results cannot be combined into a successfully executed
renewing organization.

**Actual native boundary.** Reuse the same ideal circular doubled-C5 law,
joint coordinates, default represented coefficients, strict direct-glyph
grammar, zero Gamma and held Euler flow with `delta=1/16`. The added pressure
event is the default non-noise Dissonance (OZ) primitive: it multiplies stored
pressure by its configured factor when `abs(p)>1e-9`; its separate near-zero
branch assigns `0.1`. Do not replace that piecewise law by a universal
multiplier. The declared source below stays strictly on the multiplying
branch. It changes no form, capacity or phase at the event.

Mutation requires active capacity, a live signed two-sample EPI secant
strictly above the default represented threshold, and target-local grammar.
Both the retained recent handler for U4a and the recent destabilizer/prior
Coherence requirements for U4b matter. A lifetime prior-Coherence flag is
not a recent handler. The five earlier actions cannot create a canonical
destabilizer: Transition (NAV) contributes to the phase controller's
configured disruptor load but is not a U2/U4b destabilizer.

Fresh default selection requests Mutation only if the high-Si branch has
not already applied, `Si<=si_lo`, and `abs(p)/P<=dnfr_hi`. The divisor is one
when the pressure peak is zero. With positive capacities, the necessary
low-pressure-branch bound is

```text
Si >= alpha*(nu/vmax)+beta*(1-abs(phase_offset)/pi_rep)
      +gamma*(1-dnfr_hi).
```

This excludes some capacity/phase domains but not all heterogeneous states.
The default selector admits old acceleration independently even though its
base choice does not use it. AL/EN lag priority can still replace that base
request. Missing, stale or subthreshold temporal evidence instead causes
the native Mutation abstention and a Coherence request before grammar.
Valid evidence with invalid grammar reaches strict rejection; these are not
the same outcome.

The signed direct Mutation proposal retains form, capacity and stored
pressure and changes only primitive phase and its own telemetry. Write its
ideal signed increment as `s*copysign(1,p_stored)`, with
`s=ZHIR_theta_shift_factor*pi_rep/4`. This uses exact real arithmetic over
represented factors. The default binary64 multiplication yields `1/4`;
that rounded scalar and the ideal product need not be identical. Signed
zero has the primitive's explicit `copysign` convention. Neither direction
nor magnitude is inferred from the positive historical secant.

**General renewal test.** Let y be the final admitted event-phase lift and
let the controller recompute its branch and gains from y and the actual
updated glyph load. On the paired common-arc domain, put

```text
A(g)y=(1-kG-kL)*y+kL*N(y),
osc(z)=max_i(z_i)-min_i(z_i),
D_out=osc(A(g)y).
```

The omitted global target is a common additive term. For the declared
diameter observation, strict one-call renewal is exactly `D_out>D_in`.
This criterion needs every event, flow and policy boundary to be admitted
before it describes a completed call. A proposed phase pulse or a favorable
event-only diameter is insufficient. A common phase increment alone is not
renewal. Positive diameter is also not necessary for every possible
form/capacity organization, nor does its renewal establish sustained form
work, formation or a physical identity.

On doubled C5 this bound can be sharpened without another trajectory. Let
`a_g=1-kG-kL`. The default gain rails give `a_g>=kL/2`. Rows of A have mass
`1-kG`, diagonal entry `a_g` and two neighbor entries `kL/2`. Adjacent rows
overlap by `kL`, while nonadjacent rows overlap by `kL/2`. The exact
oscillation contraction coefficient is therefore

```text
tau=1-kG-kL/2,
osc(A(g)y)<=tau*osc(y).
```

The bound is sharp over real phase vectors, including vectors inside the
common-arc domain after a sufficiently small rescaling. For one whole-pair
Mutation `theta -> theta+s*e_k`, with all other event-phase rows inert,
`osc(A(g)e_k)=a_g` and hence

```text
D_out <= tau*D_in+a_g*abs(s).
```

Strict renewal consequently requires
`a_g*abs(s)>(kG+kL/2)*D_in`. This necessary resource condition is not
sufficient: the geometry, sign, controller and full execution must still
agree. Use the actual post-event gains. If another event changes phase,
rebuild its live proposal instead of treating this single-shift formula as
the entire call.

**Frozen source and preparation.** The analytical source family is

```text
M=(a,0,-9/10,9/10,0),   abs(a-19/20)<=1/4096,
V=(27/50,27/50,6,6,27/50),   d=e=0,
r=(0,3/2,0,3,3/2).
```

Keep all ten node kinds `seed`, supplied initial traces `(AL,IL,EN)`, prior
Coherence true, zero debt, constituent ordinal three and zero lags/stability
counts. Initial temporal histories are empty at `t=1`; old rates and
accelerations are zero. These are explicit preparation data, not a claim
that the engine generated this initial organization. The retained IL handler
is essential: importing the earlier EN-only trace would reject the first OZ
despite its prior-Coherence flag. The earlier source is not rewritten.

Fix support/order `0..9`, default controller gains, empty REMESH stability
history and the existing no-callback/no-math-engine premises. Allow the two
first-call Transition inputs independently in the full configured jitter
amplitude interval. Their target association is retained; this input family
does not claim that one seed realizes every combination. The declared horizon
is at most two calls, stopping at the first live rejection. No request,
threshold, initial form or clock is adjusted after evaluating the control.

**First complete call: acquisition of conditional evidence.** Fresh
preparation uniformly selects OZ/OZ/IL/IL/NAV by block. All target-local
grammar checks pass; lags do not override. The first two pressure multipliers
and the two Transition inputs then act on the held prepared pressure. No
event changes form. The flow therefore predicts two associated samples for
each constituent, `(1,x_initial)` and `(17/16,x_endpoint)`, without a jump
being mistaken for continuous growth.

For the first pair, the exact domain certificate places the sampled secant
inside `(0.1339,0.1341)`, strictly above the default `0.1` threshold. The
first call also appends its own OZ, creating the missing recent destabilizer
and debt one. This is a conditional forecast of the recorder's output from
the same admitted flow, not measured native evidence inserted into a graph.
An actual later execution must consume its actual records. The general
[Mutation-history contract](nodal/SINE_COLLECTIVE_MUTATION_ADMISSION.md#4-sufficient-temporal-and-execution-state)
does not permit replacing those records by the predicted nodal product.

All first endpoints stay inside the form rails. The held pressures remain
outside the capacity-adaptation band, so qualification is false, counts
remain zero and capacities do not change. Exact cosine-tail bounds give
`R^2` inside `(0.23555,0.23557)`, certifying the dissonant phase-controller
branch with the actual load `3/20`. Its relative map supplies the next
phases without evaluating an absolute global angle. All consumed memory,
gains, rates, input progress and REMESH sampling/abstention are retained.

**Second preparation and accepted prefix.** Refresh pressure and Si from
the first endpoint. The prospective batch is ZHIR/RA/IL/IL/NAV. For the first
pair, the certificate has fresh pressure inside `(0.0842,0.0846)`, Si inside
`(0.2403,0.2404)` and normalized pressure inside `(0.1156,0.1162)`. Its live
positive history, active capacity, recent IL/OZ context and debt allowance
admit Mutation on both constituents. Lags advance to two without forcing
another mark. Each accepted Mutation changes phase by the positive s,
appends its mark, raises debt to two and advances its operator ordinal.
Form, capacity, pressure, rate and the associated EPI history stay unchanged.

The next block has Si inside `(0.2540,0.2541)` and normalized pressure inside
`(0.0496,0.0498)`, so its fresh default request is Resonance (RA). Its first
constituent, fine node two, still has a strictly negative form below
`-0.008828`. Both neighboring blocks remain U3-compatible after Mutation;
their live selected form mean lies inside `(0.1188,0.1192)`. The actual
Resonance blend proposes a strictly positive form inside `(0.01149,0.01154)`.
Thus the sign-preservation clause rejects the proposal throughout the
declared source/input domain. Kind preservation and phase compatibility
cannot waive the independent sign clause.

This is a live identity obstruction, not a grammar rejection, undefined
phase, missing Mutation evidence or numerical solver failure. The same
compatible subset and sign failure occur if the preceding phase reset is
omitted in a detached counterfactual. Mutation is therefore not established
as the cause of this failure. The fresh selector proposes an action whose
live identity gate does not admit it; a base request is not an admission
certificate.

**Partial-call accounting.** The native sequential path preserves the two
accepted Mutation events. The failing Resonance commits no form, phase,
capacity or glyph history. Later targets are not executed. There is no
second flow, coordinator, adaptation or terminal REMESH snapshot, and no
second-call Transition input is consumed. Physical time remains `17/16`;
all first-call EPI records, old rates and incoming phase gains remain.
Second preparation has already refreshed pressure/Si and incremented every
node's lag counters. The runtime invocation ordinal would still record the
admitted failed invocation; a detached certificate consumes no runtime
ordinal. This chronology is not all-target rollback.

The accepted phase prefix itself narrows the diameter, but no complete
second-call diameter or renewed organization is available. Continuing its
algebra past the rejected event would silently replace the specified law.
The eight fixed fine controls use the endpoints of the initial-form interval
and both independent first-call innovation intervals. They check the
whole-domain forecast; their number does not prove its interval inequalities.
The [detached owner](../tests/physics/test_native_mutation_renewal.py) blocks
production execution and preserves the earlier frozen producers.

**Separate conditional renewal and its history obstruction.** A second
static family illustrates why the preceding rejection is not a universal
geometric impossibility. On the same support let

```text
M=(1/8,0,3/4,-3/4,0),   V=(1/64,1,1,1,1),   d=e=0,
r=(1/10,0,0,0,0).
```

Its fresh default base choices are ZHIR/IL/IL/IL/IL, with positive pressure
at the first pair. Without qualifying history, the actual selector abstains
from Mutation and requests Coherence. Conditional on separately admitted
Mutation evidence and grammar, the increment increases the sole phase
maximum to `1/10+s`.
The default gain rails give `1-kG-kL>0.6816` and `1-kG-kL>kL/2`, so the
resulting diameter is `(1-kG-kL)*(1/10+s)>0.2385>1/10` for every admitted
default controller branch. No extra low-disruptor-load premise is needed.
This is a conditional action/geometry calculation; no qualifying temporal
record is supplied and no Mutation is executed on that source.

Indeed, let `P_bound=w_phase+2*w_epi+w_vf*VF_MAX`. Under the same admitted
form/capacity rails and a common lift narrower than `pi_rep`, fresh pressure
satisfies `abs(p)<=P_bound`. For a target held at capacity `1/64`, with no
form jumps between samples and at most the declared multiplying OZ pressure
event before its held flow,

```text
abs(secant) <= (1/64)*OZ_dnfr_factor*P_bound < 0.031 < xi.
```

The separate near-zero OZ branch also lies below this bound. The simple
history-producing route therefore cannot supply the required trigger on
this conditional renewing source. Different past capacities, inputs, laws
or sampling histories need their own provenance and rate argument. The
prospective phase benefit cannot manufacture missing evidence.

Finally, Mutation's primitive always retains branch-specific telemetry.
The signed mode includes absolute before/after angles and quadrant labels;
these do not descend through the common-origin quotient even though the
physical circular shift does. Retain their reference or report them as
reference-dependent readouts. Public/staged Mutation has additional lifecycle
effects and is not the direct route used here. The two complementary
obstructions leave successful complete renewal open, with its missing joint
conditions identified rather than hidden in a favorable operator name.

<a id="native-joint-mutation-reflection-obstruction"></a>
### Joint history and renewal: a reflection-sector obstruction

The previous study leaves a joint question: can a predecessor supply the
history that admits Mutation (ZHIR), while the next complete call renews
the relative-phase diameter? A declared family now has a negative answer.
This is an analytical exclusion of a continuum of preparations, not a
failed sampled trajectory or a universal obstruction to TNFR organizations.

**Complete question and frozen scope.** Keep the preceding native direct
law, doubled-C5 support, whole-pair order, represented defaults, zero Gamma,
held Euler `delta=1/16`, initial gains and supplied `(AL,IL,EN)` context.
Initial debt, lags, stability counts, rates and accelerations are zero;
temporal records are empty at `t=1`. Prior Coherence is true, constituent
ordinals are three, and callbacks, math engine and initial REMESH stability
history are absent. The preparation's origin remains supplied.

One target pair has capacity u; all four other pairs have the same capacity
W, with `0<u<W<=VF_MAX`. All capacities remain fixed over the two-call
horizon: from zero counts, even qualifying nodes cannot reach the default
adaptation threshold of five. Their counts need not remain zero. Because
each other pair has the network maximum capacity, its Si is at least alpha,
above the default high cut, so its fresh base choice is Coherence (IL).
Lags one and two do not override. This removes the earlier Resonance
obstruction through an explicit source restriction, without changing the
selector or weakening an identity contract.

The target must first receive freshly selected Dissonance (OZ), then freshly
selected and temporally admitted Mutation. No word is imposed to obtain
those actions. All forms and both unclipped flow endpoints must lie within
the default rails. Renewal means `D2>D1`, comparing the second completed
call with its own incoming diameter, not with the initial preparation.
The initial, first coordinated and post-Mutation phase states consumed by
the second coordinator must all admit the common-lift calculation.

General necessary inequalities with arbitrary phase orientation did not
decide this question. Before any source evaluation, the decision domain was
restricted explicitly to reflection-symmetric block coordinates

```text
M=(x,y,z,z,y),             (x,y,z) in [-1,1]^3,
theta=(D,a,0,0,a),        0<=a<=D<=3<pi,
nu=(u,W,W,W,W),           0<u<W<=VF_MAX,
d=e=0.
```

A common absolute phase origin may be retained separately. The target is
initially at a phase maximum. Freeze eighteen algebraic outer regions:
two directions of the second Mutation, three first controller branches,
and three second branches. Each second gain update starts from the first
updated gains. No sampled sources, candidate scans, native responses or
additional horizons belong to this protocol. An outer region may include
unrealizable controller branches; excluding it is still sufficient. This
avoids needing trigonometric branch selection to establish the obstruction.

**Shared gap and pressure identities.** Write pressure weights as `(k,b,w)`,
Si weights as `(alpha,beta,gamma)`, `d_p=w/pi_rep`, OZ multiplier as f and
IL retention as ell. For controller j write `(G_j,L_j)` for its updated
global/local gains and `A_j=1-G_j-L_j`. These are configured quantities,
not additional TNFR laws. The common global target cancels from phase gaps.

Let `H=D-a`. After the first coordinator define h as target-minus-neighbor
phase and t as neighbor-minus-far phase. The exact paired map gives

```text
h=(A1-L1/2)*H+(L1/2)*a,
t=(L1/2)*H+A1*a,
D1=h+t.
```

The gain rails preserve `h,t>=0`. With
`det=A1*(A1-L1/2)-L1^2/4>0`, set

```text
c_a=(A1-L1/2)/det,
c_t=L1/(2*det),
c_h=A1/det-1,
a=c_a*t-c_t*h,
H-h=c_h*h-c_t*t.
```

All three coefficients are positive for the admitted default gain updates.
Reflection and the regular support also give `p0+2*p1+2*p2=0`. On the
nonzero multiplying OZ branch, putting
`lambda=delta*k*W*ell` and `mu=delta*k*u*f` yields the exact two-boundary row

```text
p0_next=(1-mu-lambda/2)*p0-lambda*p2+d_p*(H-h).
```

Here `p0_next` is freshly computed pressure at the next boundary, not a
pressure reconstructed from the response. The two associated flow samples
would have secant `u*f*p0`; its required strict lower bound is xi. This
identity applies only when the first OZ uses its multiplying branch and
no form clipping changes the endpoint. Its tiny-pressure branch is handled
separately below.

Let c be the default normalized-pressure cut. The second Mutation base
choice requires `rho<=c` and low Si. Consequently

```text
u <= U(h)=VF_MAX*(r0+r1*h),
r0=(si_lo-beta-gamma*(1-c))/alpha,
r1=beta/(alpha*pi_rep).
```

Both coefficients are positive here. Keep pressure and Si weights separate
in this derivation, even when their configured values agree. Missing actual
history would still cause selector abstention; these necessary inequalities
do not certify execution by themselves.

**Positive increment.** Suppose the second pressure selects the positive
increment s. After the event the phase gaps are `(h+s,t)`, so the completed
coordinator has diameter

```text
D2=A2*(h+s)+(A2+L2/2)*t.
```

Strict renewal therefore requires

```text
(G2+L2)*h+(G2+L2/2)*t < A2*s,
h < h_max=A2*s/(G2+L2),
t < T-J*h,
T=A2*s/(G2+L2/2),   J=(G2+L2)/(G2+L2/2).
```

The first flow endpoint's form rails give
`-k+d_p*t/2<=p2_next<=k+d_p*t/2`. Combined with pressure zero-sum and the
normalized-pressure cut, nonnegative `p0_next` satisfies
`p0_next<=kappa*(k+d_p*t/2)`, where `kappa=c/(1-c/2)`.
This also covers a zero pressure peak: then every fresh pressure is zero
and the divisor-one selector convention does not invalidate the inequality.
The first rails similarly give `p2<=k+d_p*a/2`.

Use `lambda_max=delta*k*VF_MAX*ell`, and define

```text
n0=k*(kappa+lambda_max),
nt=d_p*(kappa/2+lambda_max*c_a/2+c_t),
nh=d_p*(lambda_max*c_t/2+c_h),
N0=n0+nt*T,              N1=nh+nt*J,
den_min=1-lambda_max/2-delta*k*f*U(h_max).
```

The exact certificates have `den_min>0`. A positive acquired secant then
obeys

```text
u*f*p0 <= f*VF_MAX*(r0+r1*h)*(N0-N1*h)/den_min,
0<=h<=h_max.
```

If the numerator is nonpositive, positive acquisition is already impossible.
Otherwise its concave quadratic has an exact maximum at an endpoint or its
rational vertex. Across all nine gain pairs that maximum gives
`u*f*p0<0.037<xi`. No source choice or unresolved phase-controller branch
inside the declared sector can supply the required trigger on this branch.
An independent certificate using the weaker oscillation condition
`h+t<T` also leaves a strictly positive quadratic gap to xi in every cell.

**Negative increment.** If `D1>=s`, lowering the target by s keeps all event
phases in the incoming hull. The coordinator's strict oscillation contraction
then excludes renewal. Thus renewal with a negative increment requires
`h+t<s`, and in particular `u<=U(s)`.

For `p0_next<=0`, the pressure identity, inverse gap map and first form
rails give the necessary bound

```text
u*f*p0 <= f*U(s)*[lambda_max*k
                  +d_p*(lambda_max*c_a/2+c_t)*s]
             /[1-lambda_max/2-delta*k*f*U(s)]
          < 0.008 < xi.
```

All three possible first gain updates satisfy it; the second gain branch
is irrelevant to the hull argument. Include zero pressure with either
declared signed-zero direction in the corresponding case. Neither zero
pressure nor a zero network peak leaves a missing boundary case.

**The actual small-pressure OZ branch.** When `abs(p0)<=1e-9`, direct OZ
assigns its represented literal `0.1`; it does not multiply p0. The same
selection and renewal constraints, independently of the transport identity,
give `u<=U(h_max)` in the positive-increment cells and `u<=U(s)` in the
negative-increment cells. These imply secants below `0.076` and `0.026`,
respectively, still below xi. The equality boundary at the small-pressure
cut belongs to this branch. Thus the exclusion applies to the complete
piecewise OZ law, not just its multiplying part.

**Result and retained limits.** In this reflection sector, no two-call
preparation can simultaneously acquire the required Mutation history,
select/admit the target event and complete a strict diameter renewal under
the declared law. Even relaxing the first selector and grammar inequalities
does not remove the contradiction; restoring them cannot create a witness.
The family cannot repair its missing trigger by extending a predicted
history, suppressing a gate or continuing an unavailable execution.

The [exact certificate owner](../tests/physics/test_native_mutation_joint_feasibility.py)
checks the eighteen outer regions, gain carry, positive denominators,
coefficient identities, quadratic extrema, signed-zero/peak boundaries and
the separate OZ fallback. Production execution is guarded. Algebraic basis
controls verify identities; they are not sampled physical preparations.
No fine response, new temporal record or completed call is reported.
All inequalities concern the ideal real model over represented constants;
they are not an unproved bit-for-bit statement about floating-point paths.

The obstruction does not cover asymmetric block states, targets away from
the initial phase maximum, different capacity layouts, clipped histories,
other words or longer horizons. Nor does it exclude organizations sustained
without this diameter-renewal criterion. It identifies incompatible joint
requirements of this declared preparation and policy, not a new universal
law or a physical identification.

Reflection matters to the pressure-peak bound. Without it, the target
transport row still uses the mean far pressure `(p2+p3)/2`, by the same
regular-support zero-sum identity. However, opposite pressure contrasts can
increase the actual network maximum without changing that mean. Its bound
cannot be transferred from a reflected mean to the full pressure peak.
Independent phase contrasts also affect extrema and the resultant consumed
by the gain controller. Retaining these quantities is necessary when using
the calculation outside the reflection sector.

<a id="native-joint-mutation-general-obstruction"></a>
### Arbitrary paired geometry with a uniform-capacity environment

Removing reflection does not remove the joint obstruction under the same
two-call protocol. The following conditional theorem allows all five paired
forms and phases to vary independently. It concerns acquisition followed by
one strict phase-diameter renewal, not equilibrium, all NFR maintenance or
physical geometry.

**Declared domain and decision budget.** Keep the preceding support, law,
context, default coefficients, initial gains, clock and horizon. Forms range
over `[-1,1]^5`; the initial phase lift has oscillation at most three radians,
and the target need not be an extremum. Capacities are still
`(u,W,W,W,W)`, with `0<u<W<=VF_MAX`, and within-pair contrasts remain zero.
Require both unclipped flow endpoints in the form rails and the common
phase chart at every consumed boundary, including after Mutation (ZHIR).
The requested sector remains freshly selected OZ/IL/IL/IL/IL followed by
ZHIR/IL/IL/IL/IL, with an acquired positive secant above xi. Preparation is
supplied; predicted records cannot replace an actual later recorder.
The source-width bound and default gains also imply
`3*(1-G1)+s<pi_rep`: the first coordinator and subsequent single-pair shift
stay within a common arc. No unexamined circular branch is introduced.

Before calculation, freeze eighteen direction/gain outer cells. A reserved
partition allowed at most 7,200 regions: those eighteen cells, twenty ordered
renewal witnesses, ten signed pressure-peak owners and two local-gap signs.
No source evaluation, parameter scan or optimizer was authorized by that
partition alone. The coarse inequalities below close all eighteen cells,
so the larger partition and any optimizer are unused. No native research
response or independent fine response is evaluated.

**What asymmetry changes.** Write the paired pressure as

```text
p=(p0, p_near+d_near, p_far+d_far, p_far-d_far, p_near-d_near).
P=max(abs(p0), abs(p_near)+abs(d_near), abs(p_far)+abs(d_far)).
```

The peak depends on the retained contrasts. Replacing it by a peak of
means would change selection. The first and second controller resultants
also consume the full phase state. The outer cells allow all three target
branches at each update, with the actual preceding gains carried forward;
excluding even this relaxation does not require evaluating a resultant.

Nevertheless the target transport row still closes in paired means. Let y
be the first coordinated phase lift, and define

```text
h=y0-(y1+y4)/2,
t=(y1+y4-y2-y3)/2,
D1=osc(y).
```

Use H and a for the same two signed mean gaps before the first coordinator.
Unlike the reflected case, h, t, H and a can be negative, and `h+t` is not
the diameter. The same two-by-two restriction and its inverse remain exact:

```text
H-h=c_h*h-c_t*t,
a=c_a*t-c_t*h,
p0_next=(1-mu-lambda/2)*p0-lambda*p_far+d_p*(H-h).
```

The last identity uses `p0+2*p_near+2*p_far=0`, fixed capacity and the
multiplying OZ branch. The meanings of `lambda`, `mu`, `d_p`, `c_a`, `c_t`
and `c_h` are those derived above. Arbitrary antisymmetric components cancel
from this mean identity, but not from the peak or controller.

The affine phase pressure is valid on the admitted common arc: each fine
node has two equally repeated neighboring block phases whose circular
resultant has their lift midpoint as its angle. This is ideal-circle
algebra; no exact binary64 trigonometric covariance is inferred.

**General geometric restrictions.** The exact C5 oscillation bound gives

```text
D2 <= (1-G2-L2/2)*D1+A2*s,
D2>D1  =>  D1<T=A2*s/(G2+L2/2).
```

For a positive shift, the column carrying the impulse is
`(A2,L2/2,0,0,L2/2)`. Only eight ordered output pairs can have a favorable
impulse difference. If the renewed maximum is on an adjacent row, renewal
forces `D1<(L2*s/2)/(G2+L2/2)<s`. If it is on the target, translate the
incoming lift to minimum zero. For `h>=0`, its coordinated row is at most
`(1-G2)*D1-L2*h`, hence renewal requires
`A2*s>G2*D1+L2*h>=(G2+L2)*h`. For `h<0`, the target obeys
`y0<=D1+h`, giving `abs(h)<s` instead. Reversing the lift proves the
corresponding negative-shift statements.

Thus an outward shift, whose direction agrees with the sign of h, requires
`abs(h)<H_max=A2*s/(G2+L2)`; an inward shift requires `abs(h)<s`.
All admitted gain pairs have `H_max>s`. The Si argument consequently gives
`u<=U(abs(h))` using the same r0 and r1, now without reflection. In particular
`u<0.752` throughout the renewing domain. The literal small-pressure OZ
branch therefore has secant below `0.076<xi`, including its closed threshold
boundary, independently of any multiplied-pressure transport identity.

These restrictions concern the full incoming diameter. Comparing an output
pair only with its own incoming gap would not prove renewal. Also, an event
remaining inside the old phase hull cannot renew the diameter after the
strictly contracting coordinator. A phase pulse alone is not the observation.

**Bounds retaining the complete pressure peak.** At any non-target block,
the endpoint rails and common lift give

```text
abs(p_i_next) <= B(D1)=2*k+b*VF_MAX/2+d_p*D1.
```

Only the two adjacent blocks have a capacity-contrast contribution; the
bound covers them and the far blocks. If the positive target pressure is
not zero, its low normalized-pressure ratio prevents it from owning the
peak. Hence `p0_next<=c*B(D1)` for the positive-shift case. If the whole peak
is zero, the same inequality holds with the actual divisor-one convention.
This bound replaces the reflection-specific peak estimate; it never drops
the antisymmetric pressure contribution.

The first rails still imply `p_far<=k+d_p*a/2`. Independently of reflection,
`t<=D1` and `h+t<=D1`. These inequalities allow positive upper bounds before
replacing the actual lambda by its maximum. Set

```text
lambda_max=delta*k*VF_MAX*ell,
K=lambda_max*c_a/2+c_t,
S=lambda_max*c_t/2+c_h,
C0=c*(2*k+b*VF_MAX/2)+lambda_max*k,
den(v)=1-lambda_max/2-delta*k*f*v.
```

All used denominators are strictly positive. The four orientation cases
can now be excluded by scalar bounds over the declared gain cells:

| Mutation direction and local gap | Necessary acquired secant bound |
| --- | --- |
| Positive, `h>=0` | `<0.091` |
| Positive, `h<0` | `<0.089` |
| Negative, `h>=0` | `<0.030` |
| Negative, `h<0` | `<0.078` |

For the first row, use `t<=D1-h`, `D1<T` and `0<=h<=H_max`. A positive
history-producing pressure satisfies

```text
u*f*p0 <= f*U(h)*[C0+d_p*(c+K)*T-d_p*(K+c_h)*h]/den(U(H_max)).
```

Its numerator is a concave quadratic in h, so endpoint and rational-vertex
evaluation certifies the bound for the whole interval. A nonpositive
pressure numerator already excludes the positive trigger.

For the second row put `z=-h<s`. Using `t<=T` in the mean transport gives
the sufficient upper bound

```text
u*f*p0 <= f*U(s)*[C0+d_p*((c+K)*T+S*s)]/den(U(s)) < 0.089.
```

For the third row `0<=h<s` and `p0_next<=0`; discard the latter nonpositive
term, use the positive far-pressure upper bound and obtain

```text
u*f*p0 <= f*U(s)*[lambda_max*k+d_p*K*T]/den(U(s)) < 0.030.
```

The final row needs the capacity/phase relation instead of maximizing them
independently. With `z=-h>=0`, nonpositive fresh target pressure and form
rails require `b*(W-u)+d_p*z<=2*k`. Put `R(z)=r0+r1*z`; low Si gives
`u/W<=R(z)<1`. Consequently

```text
u <= R(z)*(2*k-d_p*z)/(b*(1-R(z))),
0<=z<=H_max.
```

Bound u by the exact maximum of the concave numerator on this interval,
divided by `b*(1-R(H_max))`. Call this bound u_star; every gain cell has
`u_star<0.292`. The independent transport bound then gives

```text
u*f*p0 <= f*u_star*[lambda_max*k+d_p*(K*T+S*H_max)]/den(u_star)
          < 0.078.
```

This proof retains the correlation between capacity and the signed phase
pressure. The simpler independent bound `u<=U(H_max)` is too loose for this
case. Both possible signed-zero directions are included by the non-strict
pressure inequalities; zero local gap is also covered.

**Conclusion and execution boundary.** Every multiplying-branch bound is
below `0.091<xi`, and the fallback bound is below `0.076<xi`. Therefore no
preparation in this entire arbitrary-paired-geometry domain can both acquire
the required Mutation history and complete strict second-call phase-diameter
renewal. This conclusion already holds after relaxing some first-selection
and grammar constraints, so restoring them cannot create a witness.

The [general certificate owner](../tests/physics/test_native_mutation_asymmetric_feasibility.py)
freezes the constant certificates before independent coefficient and
optimization checks. It verifies general five-coordinate pressure transport,
retained peaks, signed-gap geometry, gain carry, positive denominators and
the separate OZ fallback. Production execution is guarded; no preparation,
temporal record, completed call or optimizer result is reported. The earlier
reflection and admission producers remain unchanged. All arithmetic belongs
to the ideal real model over represented coefficients.

The result removes the freedom of paired forms and phases as an escape
within this protocol. It still assumes a uniform-capacity environment,
the declared pressure and selection laws, one OZ acquisition interval,
unclipped forms, the common chart and this diameter observation. It does
not exclude other operator histories, longer organizations or maintenance
that does not require this particular renewal. No material geometry or
fundamental physical property is identified by the exclusion.

<a id="native-mutation-capacity-resource-budget"></a>
### Heterogeneous capacities: retained association and a necessary spread

Unequal surrounding capacities change the target transport even when the
support, pressure law and operators are unchanged. The following calculation
retains those terms and excludes a finite neighborhood of the equal-capacity
environment. It does not exhibit an acquired organization or identify a
physical geometry.

**Domain and prospective budget.** Retain the preceding arbitrary paired
forms and phases, doubled C5, represented defaults, strict grammar, supplied
context, empty initial temporal history and two-call horizon. Replace the
capacity vector by `(u,v1,v2,v3,v4)`, with `0<u<=VF_MAX` and `0<=vi<=VF_MAX`.
The actual maximum is `V=max(u,v1,v2,v3,v4)>0`. Both unclipped endpoints,
every consumed common phase chart and all live-event obligations remain
required. The requested sectors are freshly selected Dissonance (OZ) and
then acquired Mutation (ZHIR) at the target, with high-Si Coherence (IL) on
every surrounding pair at both boundaries. These are selection conditions,
not an imposed operator word. Capacity adaptation remains inactive over this
horizon from the declared initial counts.

Before coefficient evaluation, freeze the eighteen carried-gain/direction
cells and a single neighborhood width `eta0=1/16`. No other width, tuning,
source search, optimizer, native trajectory or finer partition belongs to
this decision budget. Algebraic coordinate and tensor-basis controls check
identities rather than physical preparations.

**What must be retained.** Let n and F be the near and far capacity means,
and C the covariance of source capacity and freshly computed source pressure
over the two neighboring pairs:

```text
n=(v1+v4)/2,              F=(v2+v3)/2,
C=(v1*p1+v4*p4)/2-n*(p1+p4)/2
 =(v1-v4)*(p1-p4)/4,
lambda_n=delta*k*ell*n,   mu=delta*k*f*u.
```

The pressure in C precedes the first IL multiplication and flow. Neither
the held post-event pressure nor the refreshed next-boundary pressure can
replace it. The regular-support identity `p0+2*p_near+2*p_far=0` still holds.
The exact target row, on the multiplying OZ branch, is now

```text
p0_next=(1-mu-lambda_n/2)*p0-lambda_n*p_far
        +delta*k*ell*C+d_p*(H-h).
```

With source form means M_near and M_far, the far pressure is

```text
p_far=k*(M_near-M_far)/2+b*(n-F)/2+d_p*a/2.
```

Thus both capacity contrast and its pressure association survive in the
target row. In particular, equal neighboring capacities can make C vanish
while `n-F` remains nonzero. Also, the maximum V used by Si is a different
observation from either mean. Retaining only capacity and pressure means
does not determine this transport term.

Define the signed capacity contribution J and the positive denominator A:

```text
J=delta*k*ell*[b*n*(n-F)/2-C],
A=1-mu-lambda_n/2,
A*p0=p0_next+lambda_n*[k*(M_near-M_far)/2+d_p*a/2]
     +J-d_p*(H-h).
```

Here A is positive throughout the default capacity rails. A negative C
assists the removal of initial target pressure, holding the other terms
fixed; positive C has the opposite sign. This is a pressure-transport
identity, not an energy reservoir or a new microscopic law.

The required positive secant `u*f*p0>xi`, together with the source form
rails, implies the necessary joint resource inequality

```text
J > A*xi/(u*f)-p0_next-lambda_n*[k+d_p*a/2]+d_p*(H-h).
```

For the positive Mutation direction use `p0_next<=c*P_next`; for the negative
direction use `p0_next<=0`. The complete pressure peak P_next must be
retained. The right-hand side can be negative: this argument does not assert
that J or its covariance component must independently be positive. Satisfying
the inequality does not supply grammar, temporal evidence, selection or
renewal.

**Actual Coherence selection.** At each of the two fresh boundaries every
surrounding pair must satisfy

```text
alpha*vi/V + beta*(1-abs((L*theta)_i)/pi_rep)
           + gamma*(1-rho_i) >= si_hi.
```

The full peak defines rho, including its zero-peak convention. The other
two terms are nonnegative on the admitted chart and are bounded above by
beta and gamma. Therefore
`vi/V>=(si_hi-beta-gamma)/alpha>0` is necessary; a zero-capacity surrounding
pair cannot belong to this high-Si sector. This coarse floor is not
equivalent to the full mask. Conversely `alpha*vi/V>=si_hi` is sufficient
for the base IL choice, with the declared lags and history still retained.

**A finite excluded neighborhood.** Define

```text
eta=(V-min(v1,v2,v3,v4))/V,
0<=eta<=eta0=1/16,
epsilon0=eta0*VF_MAX,
P_bar=2*k+b*VF_MAX+3*d_p.
```

All source pressures are bounded by P_bar using the source form rails,
capacity rails and phase width three. The entire neighborhood satisfies

```text
abs(C)<=epsilon0*P_bar/2,
n-F<=epsilon0,
abs(p_i_next)<=2*k+b*VF_MAX/2+b*epsilon0/2+d_p*D1,  i!=0.
```

For the last bound, each far capacity row is at most `eta*V` in magnitude;
each adjacent row is at most `(1+eta)*V/2`. The bound thus retains every
non-target pressure contrast. Since `alpha*(15/16)>si_hi`, all four
surrounding high-Si base choices actually follow at both boundaries within
this neighborhood. No missing selection mask is silently replaced by IL.

The coordinator does not consume capacity. Hence the previously derived
geometric restrictions remain valid: `D1<T`, outward `abs(h)<H_max` and
inward `abs(h)<s`, with actual gains carried between the two updates.
Target low Si gives `u<=U(abs(h))`; the OZ literal fallback again has
secant below `0.076<xi`, independently of the multiplying-branch identity.

The additional pressure numerator can be bounded before dividing:

```text
R_minus=delta*k*ell*epsilon0*(P_bar+b*VF_MAX)/2,
R_plus=R_minus+c*b*epsilon0/2.
```

The two terms in R_minus cover adverse covariance and far capacity contrast;
the extra term in R_plus covers the enlarged non-target pressure peak.
Positive-shift bounds require R_plus, while negative-shift bounds use
R_minus. These are rederived joint bounds, not a subtraction of the old
numerical gap from a covariance term alone.

The negative outward case additionally changes its capacity bound. With
`z=-h>=0`, nonpositive fresh target pressure requires
`b*(n-u)+d_p*z<=2*k`. Since `n>=(1-eta0)*V` and `u/V<=R(z)`, it follows that

```text
u <= R(z)*(2*k-d_p*z)/(b*(1-eta0-R(z))).
```

The denominator is positive on `0<=z<=H_max`; a negative numerator would
already contradict `u>0`. Let u_eta be the maximum of the concave numerator
on that interval divided by `b*(1-eta0-R(H_max))`. Exact endpoint/vertex
evaluation gives `u_eta<0.315`. The old uniform capacity bound cannot be
reused without this correction.

Use the previously defined K, S, C0, T and `den(v)`, recomputed from the
same admitted coefficients. Let B_out be the exact positive-outward scalar
quadratic bound in the preceding proof. The following bounds cover all
four orientations. For the positive-outward row, a nonpositive old numerator
is discarded upward before bounding the added resource; it no longer alone
excludes acquisition. A positive old numerator uses B_out, which is positive.

| Direction and local gap | Necessary acquired secant upper bound |
| --- | --- |
| Positive, `h>=0` | `B_out+f*U(H_max)*R_plus/den(U(H_max)) < 0.096` |
| Positive, `h<0` | `f*U(s)*[C0+d_p*((c+K)*T+S*s)+R_plus]/den(U(s)) < 0.087` |
| Negative, `h>=0` | `f*U(s)*[lambda_max*k+d_p*K*T+R_minus]/den(U(s)) < 0.031` |
| Negative, `h<0` | `f*u_eta*[lambda_max*k+d_p*(K*T+S*H_max)+R_minus]/den(u_eta) < 0.085` |

All denominators remain positive. These are ideal-real inequalities over
represented coefficients, not an assertion of bitwise trigonometric
equivariance. Both shift directions, zero local gap, zero pressure peak and
the literal OZ threshold boundary are included.

**Result.** Every bound is strictly below `xi=0.1`. Consequently any
successful acquired renewal in the declared heterogeneous sector must have
`eta>1/16`: at least one surrounding capacity must be strictly below
`15*V/16`. This is a necessary spread, not an optimal threshold, a universal
constant or a sufficient condition for renewal. The general high-Si floor
further requires `eta<=1-(si_hi-beta-gamma)/alpha`. The resulting interval
is only a necessary domain; it does not establish that any source in it is
jointly admissible.

The [capacity-budget certificate](../tests/physics/test_native_mutation_capacity_budget.py)
checks the generalized transport and covariance on coefficient bases,
the uniform limit, pressure envelopes, actual Coherence floor and all
eighteen carried-gain cells. It preserves the earlier producers and guards
native execution. No source, new history or completed trajectory is claimed.
The retained association is a dynamical dependency; it does not by itself
establish a smaller autonomous NFR, formation or sustained organization.

<a id="native-mutation-joint-orientation"></a>
### Joint-source constraints select the remaining renewal orientation

The heterogeneous resource budget does not permit its pressure, capacity
and phase extrema to be realized independently. Enforcing their common
source excludes every positive Mutation (ZHIR) increment, every inward
increment and every renewal whose new extreme is owned by an adjacent pair
within the declared two-call protocol. The remaining negative outward
target-extreme case is unresolved; this result is a necessary orientation,
not a jointly admitted source or a full-domain impossibility theorem.

**Unchanged domain and finite calculation.** Keep the complete heterogeneous
domain above, including arbitrary five paired forms/phases, capacity rails,
both actual high-Si Coherence (IL) masks, the default selector and gains,
strict grammar, empty initial temporal history and the supplied context.
All pressure fields must arise from the same source and its declared flow.
Retain the unclipped source/endpoints and common phase charts. Successful
acquisition on the multiplying Dissonance (OZ) branch requires `p0>0` and
an independently acquired positive secant above xi. The separate literal
fallback remains below xi throughout the renewing domain.
These are ideal common-lift certificates with represented coefficients;
they do not assert exact binary64 wrap equivariance.

The prospective budget contains eighteen carried-gain/direction cells,
exact source identities, signed-row support bounds and piecewise polynomial
certificates. It contains no source search, optimizer, linear-programming
run or native response. A declared extension used degree-at-most-three
Bernstein bounds only at natural geometric/capacity breakpoints for the
negative outward case. Its failure to exclude that case is retained below;
no adaptive subdivisions or posterior threshold changes followed.

**The source fixes the covariance.** Put
`m=(si_hi-beta-gamma)/alpha`; the admitted default coefficients give `m>1/3`.
Both surrounding high-Si masks imply `vi>=m*V`. Write near and far capacity
contrasts as e and e_far, and use r and s for the corresponding contrasts
in another coordinate. The freshly computed source pressure contrast is

```text
e_p=k*(s_x/2-r_x)+b*(e_far/2-e)+d_p*(s_theta/2-r_theta),
C=e*e_p.
```

It cannot be chosen independently after the forms and phases have been
fixed. With `zeta=delta*k*ell`, the recovered target numerator contains
`zeta*(n*p_far-C)`. Its source-form coefficients, in order `(x1,x4,x2,x3)`,
are

```text
(k/4)*(n+2*e, n-2*e, -n-e, -n+e).
```

Since `2*abs(e)<=n`, their exact absolute row sum is k*n. The phase and
capacity association therefore does not supply an additional freely chosen
form contribution. The capacity contribution is b times

```text
Q_capacity=n*(n-F)/2+e^2-e*e_far/2.
```

Both far-capacity coefficients are nonpositive. Its upper bound places the
far capacities at m*V, then keeps the near capacities jointly: the three
endpoint types, up to reflection, are `(m*V,m*V)`, `(V,m*V)` and `(V,V)`.
Convexity gives this endpoint support; it is not a scan of preparations.
The actual IL masks can make those extrema unattainable, which only makes
the necessary relaxation more permissive.

**Joint renewal geometry.** Let y denote the first coordinated lift,
`h=y0-(y1+y4)/2` and `D=osc(y)`. For a positive increment whose renewed
maximum is on the target, set `delta_top=max(y)-y0`. A necessary inequality
retains the distance from the target to the old maximum:

```text
G2*D+(1-G2)*delta_top+L2*h < A2*s.
```

For `h>=0`, write a for the near mean above the incoming minimum and
`b_phase=D-a=delta_top+h`. Then

```text
a>=0, b_phase>=h, a+b_phase<T,
b_phase+G2*a<A2*(s+h).
```

When the relevant linear support is nondecreasing in a at fixed D, taking
`delta_top=0` enlarges its upper bound and relaxes the last constraint. This
is a support maximization, not an assumption that the original target was
an extremum. It leaves

```text
a<=a_max(h)=min(T-h, [A2*s-(G2+L2)*h]/G2),
0<=h<=H_max.
```

The two affine branches meet at `h=T/2`. Also
`min(a_max(h),h)` equals h on the first branch and a_max on the second.
This joint polygon is strictly stronger than maximizing D and h separately.
An adjacent renewed extreme instead requires `D<s`; that separate case
does not use the target-extreme constraint. Negative directions follow by
reflecting the phase lift, with the corresponding pressure sign retained.

**Keep the signed pressure peak in the same phase calculation.** Define

```text
K=zeta*n*c_a/2+c_t,       S=zeta*n*c_t/2+c_h,
E_theta=c_a+c_t/2,       F_theta=c_t+(1+c_h)/2.
```

The inverse first coordinator gives the recovered phase numerator

```text
d_p*[K*t-S*h+zeta*e*(E_theta*r_y-F_theta*s_y)].
```

For the next pressure peak, retain each of the eight signed non-target
rows rather than separately assigning its phase contribution D:

```text
(L*y)1=(h-t)/2-r_y+s_y/2,
(L*y)4=(h-t)/2+r_y-s_y/2,
(L*y)2=t/2+r_y/2-3*s_y/2,
(L*y)3=t/2-r_y/2+3*s_y/2.
```

Adding the selected signed row multiplied by c to the recovered phase
expression gives `A*t+B*h+C_r*r_y+C_s*s_y`, with A nonnegative in all
declared cells. If the far mean is f_phase, then `t=a-f_phase` and
`abs(s_y)<=min(f_phase,D-f_phase)`. Exact far support is

```text
sup[-A*f_phase+C_s*s_y]=D*max(abs(C_s)-A,0)/2.
```

Let `R_peak=max(abs(C_s)-A,0)/2` and raise A to
`A_prime=max(A,abs(C_r))`. Since a is nonnegative, this is an upper bound.
It makes near support monotone in a at fixed D, as required above. The
remaining phase bound is

```text
(A_prime+R_peak)*a_max(h)+(B+R_peak)*h
    +abs(C_r)*min(a_max(h),h).
```

The endpoint form contribution to a non-target peak is at most 2*k; its
capacity contribution is bounded using the heterogeneous rails, not the
old equal-capacity value. In particular `(1-m/2)*V` bounds every non-target
capacity contrast. All source, capacity and phase support terms retain
their coefficients before taking the maximum over endpoint types.

For outward positive h, discarding the favorable term
`-zeta*n*c_t*h/2` permits `S=c_h` in an upper bound. This alone does not
make every absolute-value expression monotone in V. The full numerator
support is convex in V; endpoint support at zero and VF_MAX must be checked
before replacing the actual maximum. Zero here is an algebraic bounding
endpoint, not an active-capacity preparation. The required rate is then
bounded with the admitted `U(h)` and a positive denominator. Each natural
phase branch gives an exact scalar quadratic certificate.

**Excluded orientations.** Recomputed coefficient certificates give:

| Mutation increment and renewed extreme | Necessary acquired secant bound |
| --- | --- |
| Positive, outward, target maximum | `<0.095` |
| Positive, inward, target maximum | `<0.094` |
| Negative, inward, target minimum | `<0.032` |
| Either direction, adjacent extreme | `<0.053` |
| OZ literal fallback, any orientation | `<0.076` |

Every bound is below `xi=0.1`. Inward cases retain `abs(h)<s`; adjacent
cases retain `D<s`; both keep the same source pressure and capacity
constraints. Zero local gap is covered by the inward bounds. A zero next
pressure peak does not erase Si admission: its actual zero-ratio convention
is retained, while both possible signed-zero increment directions are
included in the respective necessary inequalities.

**The remaining question is open.** Any successful route in this protocol
must use a negative Mutation increment, have `h<0`, and place the renewed
minimum on the target. Its first acquired positive secant requires `p0>0`;
the second selected pressure must satisfy `p0_next<=0`. This need not be a
strict sign reversal because negative signed zero is included. Neither the
history requirement nor the current pressure can be substituted for the
other.

The same coefficients give `R(H_max)<m`, so every surrounding capacity
exceeds the target capacity u. With `z=-h>0`, the fresh target pressure
`p0_next<=0` requires the post-flow form contrast

```text
x0_next-(x1_next+x4_next)/2 >= [b*(n-u)+d_p*z]/k > 0.
```

Thus a still-possible target is below its neighbors' mean in phase and
above their mean in form, while retaining lower capacity. This is a necessary
compatibility condition, not a constructed state or a new constitutive law.

For this outward negative case, writing `z=-h` gives the exact restriction

```text
n<=n_bar(z)=min(VF_MAX, U(z)+(2*k-d_p*z)/b).
```

The finite cubic attempt joined this with source capacity support and the
renewal polygon. Its rational Bernstein upper envelope exceeds 0.157 in
the worst cell and therefore does not exclude the required rate 0.1. That
is an inconclusive upper bound, not an attained rate, an admissible source
or evidence of renewal. No finer partition or response was evaluated.

The remaining actual IL masks still couple near capacity to geometry and
pressure. For example, averaging the second near masks gives
`n/V>=m+beta*abs(h-t)/(2*alpha*pi_rep)`, even after discarding the
nonnegative pressure penalty. The owner of the actual maximum V and the
common source/endpoint form rails also remain unresolved joint constraints.
They cannot be replaced by a favorable covariance or independent extrema.

The [joint-orientation certificate](../tests/physics/test_native_mutation_joint_orientation.py)
retains the closed orientations and the inconclusive negative outward
attempt separately. Earlier producers remain unchanged. This is progress
on the declared acquisition/renewal mechanism; formation, maintenance,
equilibrium and physical identification remain separate questions.

<a id="native-mutation-source-constraints"></a>
### Same-source feasibility with retained pressure peaks

The remaining negative-outward renewal question is still open. Two finite
algebraic attempts below retain more of the shared source, but neither
certifies exclusion or supplies a feasible source. An exact pressure-coordinate
reduction now exposes the remaining compatibility problem directly, including
all three form boundaries. It preserves the complete two-call domain above;
no threshold, support, history or horizon changes, source search or native
response belongs to this calculation.

**Actual capacity ownership.** Because `u/V<R(H_max)<m`, the target cannot
own the maximum capacity. A maximum must occur at a near or far pair; ties
can belong to both cases. This is an exhaustive decomposition of the same
domain, not a selection of prepared capacity profiles. If a near pair owns V,
then `n>=(1+m)*V/2`, and the negative fresh target pressure gives

```text
u <= min(VF_MAX*R(z), R(z)*(2*k-d_p*z)/(b*((1+m)/2-R(z)))).
```

All denominators are positive. In either ownership case, `n>u` also requires
`z<2*k/d_p`. If a far pair owns V, the same-source capacity resource obeys
`Q_capacity<=(1-m)*V^2/4`. To see this, reflect the cycle so `v2=V` and
maximize over `v3` at `m*V`. The four near-capacity corner values, divided
by `V^2`, are

```text
-m*(1-m)/4, (1-m)^2/8, 3*(1-m)^2/8, (1-m)/4.
```

Coordinatewise convexity and `m>1/3` give the stated bound. The stronger
fixed-mean expression keeps

```text
Q_capacity <= n^2/2+e^2-V*((1+m)*n-(1-m)*abs(e))/4,
abs(e) <= min(n-m*V,V-n).
```

The frozen first attempt combined the coarse ownership bounds with the
renewal polygon over nine carried gain pairs. Its worst acquired-rate
upper bounds are approximately `0.10994` for a near maximum and `0.17459`
for a far maximum. Both exceed xi; neither is an attained rate. The exact
failed calculations remain with the certificate rather than being replaced
by a favorable interpretation.

**The first pressure peak remains associated with the acquired history.**
On the multiplying branch, successful history requires `p0>0`. The actual
default Dissonance (OZ) selection also requires `p0/P0>c`, where
`c=dnfr_hi` and P0 is the first full pressure peak. Hence `P0<p0/c`.
This comparison is strict; hysteresis and score overrides belong to the
separate parametric selector, and strict grammar cannot substitute OZ.
The declared lag priorities cannot create an OZ request either.

Write `q=p_next`. Before zero-sum elimination, the same-source target row is

```text
q0=(1-delta*k*f*u)*p0
   +zeta*(v1*p1+v4*p4)/2-d_p*(c_h*z+c_t*t).
```

Because `n>0`, its near contribution is strictly greater than
`-zeta*n*p0/c`. This couples the allowed pressure transport to the very
pressure that must generate the positive secant; the two cannot be maximized
independently.

Put `B(z)=(2*k-d_p*z)/b`, `R(z)=r0+r1*z`, and
`J(z,t)=m+r1*abs(z+t)/2-R(z)`. The second near Coherence (IL) masks and
fresh target pressure imply

```text
n<=u+B(z),        u<=R(z)*B(z)/J(z,t).
```

These use the actual t of the source. A phase value chosen solely to maximize
another bound cannot be charged to the IL mask. If `t<-z`, replacing t by
`-z` raises the phase numerator and removes a nonnegative IL penalty, giving
a valid upward relaxation. The renewal geometry also gives
`t<=A2*(s+z)`. With `t>=-z`, the resulting acquired-rate envelope is

```text
f*d_p*R*B*(c_h*z+c_t*t)
---------------------------------------------------------------
[m+r1*(z+t)/2-R]*(1-zeta*B/c)-(delta*k*f+zeta/c)*R*B.
```

The frozen certificate proves the substituted denominator positive. At
fixed z the numerator and denominator are affine in t, so their ratio is
bounded by its two endpoint values. These have cubic numerators and
quadratic denominators in z. Exact rational Bernstein bounds on the
declared natural intervals give a positive denominator margin above
`0.17749`, but the threshold-gap coefficients do not certify exclusion.
The resulting conservative rate envelope remains above xi.
This second attempt is also inconclusive. No subdivision, alternative
envelope, source evaluation or threshold adjustment followed it.

**Exact source coordinates and all fifteen form rails.** Let
`L=N-I` on the five-cycle, with indices modulo five. On zero-sum pressure,
the inverse on the mean-zero subspace is

```text
(Q*p)i=-4*p_i/5+2*(p_(i-2)+p_(i+2))/5,
L*Q=Q*L=I-1*1^T/5.
```

For fixed capacities v and a source phase lift theta, the declared pressure
law `p=L*(k*x+b*v+d_p*theta)` is equivalent to

```text
sum(p)=0,
a0=(Q*p-b*v-d_p*theta)/k,
x0=a0+eta*1.
```

This inverts the supplied constitutive pressure law before evolution; it
does not reconstruct pressure from an evaluated form derivative. The common
form offset eta is a free source coordinate subject to the form rails.
With `F1=diag(f,ell,ell,ell,ell)` and the first relative coordinator y,

```text
a1=a0+delta*diag(v)*F1*p,
q=p+delta*k*L*diag(v)*F1*p+d_p*L*(y-theta),
a2=a1+delta*diag(v)*F2*q,
F2=diag(1,ell,ell,ell,ell).
```

Mutation changes phase without a form jump. Its terminal flow still uses
the selected, held q, not a pressure refreshed after the phase event. Thus
the three form boundaries are `xj=aj+eta*1`. For the fifteen coordinates
of `(a0,a1,a2)`, a single offset exists exactly when

```text
max(a0,a1,a2)-min(a0,a1,a2)<=2,
eta in [-1-min(a0,a1,a2), 1-max(a0,a1,a2)].
```

Separate offset choices at each boundary would admit inconsistent histories.
The associated first-flow secant remains `u*f*p0`; the second endpoint is
retained for complete-call rails, not a second Mutation trigger. Since the
admitted capacities are positive, the exact weighted balances are

```text
sum_i (x1_i-x0_i)/(v_i*F1_ii)=delta*sum_i p_i=0,
sum_i (x2_i-x1_i)/(v_i*F2_ii)=delta*sum_i q_i=0.
```

These identities constrain one source and its evolution. They neither
admit an event nor establish conservation of unweighted form.

**Conditional linear feasibility; the parameter domain remains open.**
Fix capacities and primitive relative phases, admit their common charts,
and determine the actual two controller branches and complete phase
renewal. Then four independent source pressures and eta suffice for the
remaining source/flow constraints. The three forms and q above are affine
in these five unknowns. Retaining eta uses thirty linear rail inequalities.

For a positive pressure peak P with fixed owner j and sign sigma, impose
`P=sigma*p_j>0` and `-P<=p_i<=P`. Define, at each selection boundary,
`S_i=alpha*v_i/V+beta*(1-abs((L*theta)_i)/pi_rep)+gamma`.
The surrounding high-Si masks are the two linear inequalities
`gamma*p_i<=(S_i-si_hi)*P` and
`-gamma*p_i<=(S_i-si_hi)*P`. Source target selection, its positive secant,
and successor target selection similarly retain

```text
(S0_initial-si_lo)*P0-gamma*p0<=0,
p0>c*P0,             u*f*p0>xi,
p0>OZ_TINY,
(S0_next-si_lo)*P1+gamma*q0<=0,
q0<=0,               -q0<=c*P1.
```

There are nine possible signed source-peak owners: positive target or
either sign at a non-target pair. When `P1>0`, `c<1` excludes the target
as its owner, leaving eight signed non-target choices. Thus each fixed
admitted parameter configuration has at most 72 linear nonzero-peak
systems, plus nine separate `q=0` outer systems. Ties may overlap. At
`q=0`, retain the actual zero-ratio Si tests; multiplying by a zero peak
would erase them.

Here `OZ_TINY` is the actual multiplying-branch cutoff. Strict inequalities
require exact positive slack or a verified witness, not a solver tolerance.
A common slack must use fixed declared scales for pressure and rate units;
separate exact margins avoid that extra convention. Rational capacities and
relative phases, with admitted controller branches and represented law
coefficients, give rational systems; arbitrary fixed real parameters need
not. A constructive negative action requires `q0<0` or a separate proof
that the fresh represented producer yields negative zero. An assigned
algebraic zero cannot supply that sign bit. Both directions may still be
retained in an exclusion domain.

This is a reduction of the conditional pressure/form problem. The
capacities, phases and their admissible controller branches remain joint
parameters; the full problem has not become a solved linear program.
No linear solver or source was evaluated. The
[source-constraint certificate](../tests/physics/test_native_mutation_source_constraints.py)
retains the failed envelopes and checks the inverse, transport, shared
offset and weighted balances independently. These are conditional
compatibility restrictions of the declared model. Autonomous occurrence,
formation and selection of a fundamental law remain separate questions.

<a id="native-mutation-feasibility-compiler"></a>
### Exact conditional pressure-system compilation

The private [research compiler](../src/tnfr/research/_native_mutation_feasibility.py)
implements the preceding reduction for the unchanged paired unit five-cycle,
two-call Dissonance (OZ)/Coherence (IL) then Mutation (ZHIR)/Coherence (IL)
protocol. It is a detached algebraic instrument. It selects no source, installs
no execution law and evaluates neither a native trajectory nor a linear program.

`NativeMutationSlice` supplies five capacities, five primitive phase lifts and
two declared controller branches. `compile_native_mutation_slice` re-admits
those primitives and rebuilds the law from fresh owned defaults. Rational
inputs remain exact; other admitted reals use their represented values. Raw
represented default weights are normalized in rational arithmetic, as in the
preceding ideal certificates. This is not bitwise equivalence to floating-point
normalization, trigonometry or wrapped native execution. Removing a common phase
origin is justified only by the declared ideal relative-phase quotient.

The result contains the three affine form boundaries, fresh successor pressure,
associated ideal temporal samples and all 81 canonical cells in the fixed
coordinate order `(p0,p1,p2,p3,eta)`. Each cell retains the thirty common-offset
rails and labelled weak or strict inequalities. Signed peak ties can overlap;
the nine zero-successor-pressure cells retain constant zero-ratio Si masks
instead of multiplying them by zero. A zero target pressure in any cell still
does not prove the represented negative-zero sign required for that action.

**Parameter and execution obligations remain explicit.** Algebraic checks
retain the common charts, negative outward displacement, complete diameter
renewal and target-minimum geometry. A failing check marks the parameter slice
incompatible; passing them yields only conditional compilation. Actual controller
branches remain unevaluated obligations. For the paired source,

```text
R^2 = 1/5 + (2/25)*sum_(i<j) cos(theta_i-theta_j).
```

The first predicate uses source phases; the second uses the first coordinated
phases after the negative Mutation jump. Glyph histories give disruptor loads
`1/20` and `2/25`; both lie below the owned low-load threshold. The ordered native
policy therefore requires `R>=R_hi` for the stable branch, `R<=R_lo` for the
dissonant branch, and `R_lo<R<R_hi` for transition. The obligations store these
equivalent squared bounds and carry the first gain update into the second.
Compilation does not declare a chosen branch realized. Actual associated
measured history, strict grammar context and represented execution retain their
separate admission duties; ideal affine samples are not observed native samples.

**One-way exact contradiction checking.** Write each compiled row as
`a_i dot u <= b_i`, or `< b_i` for a strict row. For nonnegative exact weights
`lambda_i`, the conditions

```text
sum_i lambda_i*a_i = 0,
sum_i lambda_i*b_i < 0
```

contradict the weighted sum of the rows. The same follows with zero summed
right-hand side if at least one strict row has positive weight: it would require
`0<0`. A zero-weight strict row contributes no strictness. This proves the
checker directly; no solver completeness or numerical tolerance is assumed.
Weights apply to the declared row units, with reciprocal-unit factors when
combining pressure, form, rate and dimensionless inequalities. No shared
unscaled slack variable is introduced.

`check_cell_contradiction` receives primitive parameters, a canonical `CellKey`
and labelled multiplier pairs. It rebuilds the consumed cell rather than trusting
a compiled report. Unknown or repeated labels, invalid or negative scalars and
any nonzero coefficient residual reject the certificate. Exact cancellation
without a contradictory bound returns an inconclusive result, never feasibility.
The result retains normalized parameters, the rebuilt law, execution premises
and unevaluated controller obligations alongside its weights and exact residuals.
A successful check excludes only its conditional fixed-parameter cell. It does
not cover the other cells, the unfixed capacity/phase domain or a different law.

The [independent controls](../tests/research/test_native_mutation_feasibility.py)
exercise coefficient identities, carried gains, row equivalence, strict and
zero-peak boundaries and certificate admission. Their algebraic frames are not
admitted sources. No source search or controller-branch response is evaluated.
The earlier frozen envelopes and producers remain unchanged; heterogeneous
acquired renewal existence, autonomous occurrence and formation remain open.

<a id="native-mutation-joint-parameter-restrictions"></a>
### Joint phase and pressure-mask parameter restrictions

The same two-call negative-outward target-minimum problem has two useful
parameter projections. One constrains the actual second phase controller from
renewal geometry; the other eliminates pressure ratios from the selector masks
and zero-sum law. Neither supplies a source or changes the constitutive law.
Both preserve the preceding source, history, support, clock and form rails.

**Renewal constrains the actual second controller.** Write the carried second
gains as `(g,l)`, `A=1-g-l`, the Mutation shift magnitude as s, and the
first coordinated phase diameter as D. For the five-cycle and `A>=l/2`, the
largest half-l1 difference of two coordinator rows is `A+l/2`. A negative
target jump adds at most `A*s` to a row difference. Complete renewal therefore
requires

```text
D < T = A*s/(g+l/2).
```

If the jumped target remained inside the old phase interval, the positive
coordinator, whose row sum is `1-g<1`, could not renew its diameter. Gauge the
old minimum to zero. The jumped target must be `-a`, with `0<a<=s`, while the
four other phases lie in `[0,D]`. These are necessary consequences of renewal,
not a prepared phase profile.

Suppose `T+s<pi`. For fixed other phasors, their resultant is nonzero and its
angle lies in the common arc. The squared resultant as a function of one
environmental phase has no interior minimum on that arc: its possible interior
stationary point is a maximum. Moving each of the four phases to an endpoint
therefore gives a lower bound. Increasing a to s and D to T decreases every
nonconstant cosine term. With h of the four phases at the upper endpoint,

```text
R_event^2 >= min_(h=0,...,4) [1+(4-h)^2+h^2
    +2*(4-h)*cos(s)+2*h*cos(T+s)+2*h*(4-h)*cos(T)]/25.
```

The paired ten-node resultant equals this five-block resultant. The bound
retains the single shifted target; replacing all five phases by an arbitrary
interval loses that information. The endpoint patterns belong to an enlarged
algebraic domain and need not satisfy first-phase descent, pressure or form rails.

The [controller restriction owner](../src/tnfr/research/_native_mutation_controller.py)
rebuilds the default law and both carried gains. It uses the shared rational
cosine enclosure with eight terms (degree fourteen lower polynomial) and
128-bit outward dyadic rounding. Its finite budget is the nine branch pairs
and five endpoint multiplicities per pair, with no refinement or source search.
The common-arc and row-sign prerequisites are checked separately. The actual
second disruptor load is `2/25`, below the low-load threshold; the native
ordered predicates, including their strict boundaries, determine which branch
claims contradict the resulting lower bound.

This phase-box check excludes `(transition,dissonant)` and
`(dissonant,dissonant)`: their universal squared-resultant lower bounds exceed
`0.49153` and `0.52391`, respectively, whereas `R_lo^2<0.40529`. Thus seven
of the nine carried branch pairs are not excluded by this necessary test.
The pair `(stable,dissonant)` is not excluded by this box: its lower bound is about
`0.404720`, below the actual cutoff. The other six pairs also remain conditional.
Its retained relaxed-minimum upper bound is also below the cutoff, by more than
`1/2000`. Greater cosine precision cannot close this relaxation; the missing
correlations with the actual first-phase image and source remain essential.
Here `dissonant` names the phase coordinator's configured branch; it is distinct
from Dissonance (OZ), which remains the required first target operator.
The [actual first-phase image](#native-mutation-source-image) supplies the
missing exclusion separately. Excluding controller pairs does not prove that
a surviving pair is realized.

**Exact single-boundary pressure-mask projection.** At either selection
boundary set

```text
S_i = alpha*v_i/V + beta*(1-abs((L*theta)_i)/pi_rep) + gamma,
B_i = min(1,(S_i-si_hi)/gamma), i=1,...,4,
H = max(0,(S_0-si_lo)/gamma), c=dnfr_hi.
```

All four B_i must be nonnegative. When the pressure peak P is positive,
Coherence (IL) is equivalent to `abs(p_i/P)<=B_i` at the surrounding blocks.
A signed non-target peak at j additionally requires `B_j=1`. Let
`M_j=sum_(i!=0,j) B_i`. The remaining three normalized pressures fill the entire
interval `[-M_j,M_j]`: sums of closed intervals have no gaps. This proves the
following exact existence conditions for the mask/peak/zero-sum subsystem.

| Boundary and signed peak | Additional necessary and sufficient conditions |
| --- | --- |
| Source target positive peak | `H<=1`, `sum_(i=1,...,4) B_i>=1` |
| Source non-target negative peak j | `B_j=1`, `H<=1` |
| Source non-target positive peak j | `B_j=1`, `H<=1`, `M_j>1+c`, `M_j>=1+H` |
| Successor non-target positive peak j | `B_j=1`, `H<=c`, `M_j>=1-c` |
| Successor non-target negative peak j | `B_j=1`, `H<=c`, `M_j>=1+H` |
| Successor zero pressure | `S_0<=si_lo`, with all four surrounding masks passing |

For the source, `p0/P=t` obeys `t>c` and `t>=H`; a positive non-target peak
requires the others to sum to `-1-t`. Thus `M_j>1+c` is strict, whereas equality
in `M_j>=1+H` is allowed when H exceeds c. For a negative non-target peak,
the permitted target tie `t=1` balances it. At the successor, `q0/P=-t` obeys
`H<=t<=c`: a positive peak requires the others to sum to `t-1`, and a negative
peak requires `1+t`. Choosing `t=c` or `t=H`, respectively, proves sufficiency
for these projected conditions. Peak ties remain admitted.

In particular, a lone pressure-capable surrounding block cannot balance a
nonzero successor peak when the other three B_i vanish: the target's allowed
ratio is below one. More generally the other blocks must supply at least
`1-c` of normalized pressure for a positive peak, or `1+H` for a negative one.
If every B_i is below one, every nonzero successor peak owner is excluded.
Zero pressure retains its separate constant masks and no acquired negative-zero
direction is inferred.

`project_native_mutation_masks` in the [private compiler](../src/tnfr/research/_native_mutation_feasibility.py)
re-admits primitive parameters and computes these two boundary projections.
The three-form shared-offset rows and fresh transport still couple the source
and successor pressures; passing both projections does not solve that coupling.
The source's strict OZ cutoff and acquired-rate threshold also constrain its
absolute pressure scale. No normalized-pressure witness here is an admitted
source, measured history or native Mutation event. The original 81 cells remain
available with all those obligations intact.

The [controller controls](../tests/research/test_native_mutation_controller.py)
and [mask projection controls](../tests/research/test_native_mutation_mask_projection.py)
check the finite coefficient bounds, interval-sum derivation, strict endpoints,
zero-peak masks and agreement with the original affine rows. They preserve
all earlier frozen producers and responses. Joint source existence and actual
formation remain open until the full associated conditions are discharged.

<a id="native-mutation-source-image"></a>
### Actual first-phase image and joint capacity boundaries

The first phase coordinator and both Coherence (IL) masks act on one source.
Retaining that association gives two additional conditional obstructions in
the same ideal paired five-cycle model. No prepared source, solver, trajectory,
threshold adjustment or refinement of the preceding cosine bounds is used.

**High-resultant phase averaging.** For five real phase lifts in a common
interval of width less than pi, set `S=sum(exp(i*theta_j))` and `R=abs(S)/5`.
If `R>=3/5`, every doubly stochastic averaging of these lifts preserves or
increases R. This is a conditional phase-resultant theorem, not a universal
monotonicity statement about the diagnostic C or the operator runtime.

First consider a pair with midpoint m and half-gap d, where `abs(d)<pi/2`.
Its phasor sum is `a*exp(i*m)`, with `a=2*cos(d)>0`. Write `psi=arg(S)`.
Projecting the other three unit phasors onto psi gives

```text
a*cos(m-psi) = abs(S) - sum_(other three j) cos(theta_j-psi)
             >= abs(S)-3 >= 0.
```

A pair average reduces the absolute gap and adds `b*exp(i*m)` to S for
some `b>=0`. Therefore

```text
abs(S+b*exp(i*m))^2-abs(S)^2
    = 2*b*abs(S)*cos(m-psi)+b^2 >= 0.
```

The new phases remain in the old interval and retain `R>=3/5`. A pair
average past the midpoint differs only by a permutation, which leaves R
unchanged. These facts allow the pair operation to be repeated.

For completeness, a finite construction extends the pair result to any
doubly stochastic matrix B. Sort x and `y=B*x` decreasingly. The sum of the
largest k coordinates of y is `sum(c_j*x_j)` with `0<=c_j<=1` and
`sum(c_j)=k`; it is at most the sum of the largest k coordinates of x.
The total sums agree. If `y1=x1`, keep that coordinate and continue on the
remainder. Otherwise choose adjacent sorted coordinates with
`x_k>=y1>=x_(k+1)` and average them to
`(y1, x_k+x_(k+1)-y1)`. The residual lies between the original pair.
Move y1 to the front. For a prefix of length r in the remaining sorted
list, if `r<k`, every chosen coordinate is at least y1 and hence at least
each remaining target coordinate. If `r>=k`, the prefix sum is the old
largest `r+1` sum minus y1, which dominates the corresponding target sum.
Induction constructs y with at most four pair averages and permutations.
All intermediate vectors stay in the same phase interval, so the preceding
high-resultant argument applies at every step.

The high-resultant premise matters. As a formal lemma counterexample, take
two unit phasors `(24+7i)/25`, `(24-7i)/25` and three copies of `(-4+3i)/5`.
They occupy a common arc shorter than pi. Replacing the first pair by two
copies of 1 averages their phase lifts but decreases R squared by exactly
`44/15625`. Its initial R is below `3/5`. This is neither a native source
nor a counterexample to the particular five-cycle coordinator; it prevents
an unsupported general averaging claim.

**The stable first branch cannot lead to the dissonant second branch.**
Write its actual first gains as `(g,l)` and let N be the two-neighbor mean.
Up to a common angle, which does not change R, the first phase image is

```text
y=C1*theta, C1=(1-g-l)*I+l*N=(1-g)*B,
B=C1/(1-g).
```

For `0<g<1`, `0<l<=1-g`, B is nonnegative and doubly stochastic. After
averaging by B, multiplication by `1-g` also cannot lower R: every phase
difference has absolute value below pi, and contracting it increases its
cosine in `R^2=(5+2*sum_(i<j) cos(theta_i-theta_j))/25`.
The first controller's stable branch requires `R_source>=R_hi`. Provided
`R_hi>=3/5`, its actual coordinated phase thus has `R_y>=R_hi`.

Mutation (ZHIR) then changes one block phase by the negative shift s. The
paired ten-node resultant equals the five-block resultant, so the triangle
inequality gives

```text
R_before_second >= R_y-abs(exp(-i*s)-1)/5 >= R_hi-s/5.
```

The [source-image instrument](../src/tnfr/research/_native_mutation_source_image.py)
rebuilds the owned law, first stable gains and actual disruptor loads, and
checks these exact coefficient prerequisites. It applies no trigonometric
approximation and consumes no prepared source or cached verdict. Under the
current owned law `R_hi-s/5>R_lo`: the successor lower bound is about
`0.708547`, versus `R_lo` about `0.636620`, with a positive exact gap greater
than `0.071927`. The actual second load `2/25` is below
`disr_lo`, so the dissonant branch would require `R_before_second<=R_lo`.
This contradiction excludes `(stable,dissonant)` through the actual first
image. It leaves the old, inconclusive phase-box calculation unchanged.

Together with the preceding two renewal obstructions, this excludes every
second-dissonant branch in the declared problem. Six carried branch pairs
remain conditional: each of the three first branches followed by stable or
transition. This does not establish that any of those pairs has a compatible
source, acquired Mutation history or complete renewal. The new stable-first
obstruction itself needs no complete-renewal inequality; the other two do.

**Two saturated IL capacity floors are incompatible with outward geometry.**
Let `h=L*theta`, `L=N-I`, and
`m=(si_hi-beta-gamma)/alpha`. At a surrounding block i, the actual IL mask
at either boundary is equivalent to

```text
alpha*(v_i/V-m) >= beta*abs(h_i)/pi_rep + gamma*rho_i,
```

where rho is the normalized absolute pressure, or zero when the full peak
is zero. Both right-hand terms are nonnegative. Since capacities are held,
a block at `v_i/V=m` that passes both masks must satisfy

```text
p_i=q_i=0, h_i=(C1*h)_i=0.
```

Here `L*C1=C1*L` and the common phase angle is annihilated by L. For every
admitted first branch, `C1=(1-g)*I+l*L` with `l>0`; hence also `(L*h)_i=0`.
Moreover `sum(h)=0`. Two distinct such blocks force h to vanish on C5:
up to rotation/reflection, an adjacent pair has
`h0=h1=0`, giving `h4=h2=0` through the two Laplacian rows and then `h3=0`
by the sum. A distance-two pair has `h0=h2=0`; its rows give
`h4=h3=-h1`, and the zero sum forces `h1=0`.
Since the cycle is connected, `L*theta=0` makes theta constant. The first
image is constant too, contradicting the strictly negative outward target
gap. Consequently at most one of the four surrounding capacities can equal
the common IL floor in a valid source. Capacities below the floor already
fail a single mask. This excludes exact boundary configurations, not a
neighborhood of them and not every heterogeneous configuration.

There is also a same-source pressure constraint at any saturated block.
On the admitted multiplying OZ branch `p0>OZ_TINY`, set
`D=diag(f*u,ell*v1,...,ell*v4)`. The exact pressure transport is
`q=p+delta*k*L*D*p+d_p*(C1-I)*h`. At a saturated block all terms except
`delta*k*(L*D*p)_i` vanish, so `(L*D*p)_i=0`. If that block neighbors
the target, its other neighbor j must obey

```text
ell*v_j*p_j=-f*u*p0 < 0.
```

With `r_j=v_j/V`, `r0=u/V`, the source IL mask at j and strict source
ratio `p0/P0>c` further require

```text
r_j*[alpha*(r_j-m)-beta*abs(h_j)/pi_rep]
    > gamma*(f/ell)*r0*c.
```

The [joint-boundary instrument](../src/tnfr/research/_native_mutation_joint_boundary.py)
retains the fresh law and exact two-distance-class reconstruction identities.
Its [controls](../tests/research/test_native_mutation_joint_boundary.py) check
formal coordinate bases and associated pressure cancellation. The
[source-image controls](../tests/research/test_native_mutation_source_image.py)
check the averaging algebra, actual matrix, threshold premises and low-resultant
counterexample. These are conditional proof and coefficient controls, not
source evaluations. Full pressure transport, both masks, common form rails,
history and actual branch admission still have to hold together. None of these
results derives autonomous occurrence, formation or physical identification.

<a id="native-mutation-zero-pressure-reduction"></a>
### Exact elimination when the entire successor pressure vanishes

The zero-successor-peak subcase retains the complete equation `q=0`, with
all five components zero. This is stronger than the target condition `q0=0`.
It determines the source pressure from the same capacities and primitive
phases, rather than allowing separately compatible pressure masks to choose
different fields. The result is an exact conditional reduction of the nine
zero-peak cells. It neither excludes the whole parameter domain nor supplies
an admitted source.

Use the unchanged multiplying Dissonance (OZ) source branch, held capacities,
Euler interval delta, and pressure coefficients `(k,b,w)` from the owned law.
The phase coefficient in the common-lift pressure is `d_p=w/pi_rep`, not w.
Write

```text
L=N-I, h=L*theta, y=C1*theta,
D=diag(f*v0,ell*v1,...,ell*v4), tau=delta*k,
A=I+tau*L*D,
q=A*p+d_p*(C1-I)*h.
```

The common global phase angle is annihilated by L, and the first coordinator
commutes with L. The matrix D belongs to the first call's nodal form rate;
using D to express transport does not make it the second call's rate law.

**The pressure transport is invertible throughout the admitted capacity rails.**
Set `D_max=max_i D_ii`. Every column of A sums to one. Its diagonal is
`1-tau*D_ii` and its two nonzero off-diagonal entries are `tau*D_ii/2`.
Thus A is nonnegative when `tau*D_max<1`. The stronger uniform condition

```text
2*delta*k*VF_MAX*max(f,ell) < 1
```

implies strict column diagonal dominance for every admitted capacity tuple.
Here is an elementary invertibility proof. If `A^T*z=0`, choose i with
maximal nonzero `abs(z_i)`. Its row would require
`(1-tau*D_ii)*abs(z_i)<=tau*D_ii*abs(z_i)`, contradicting
`2*tau*D_ii<1`. Hence the kernel of the square matrix `A^T` is trivial,
so A is invertible. Nonnegativity of A does not imply nonnegativity of its
inverse. This proof establishes invertibility; evaluating one owned coefficient
bound only verifies that the present model lies in its domain.
The frozen owned-coefficient check gives `tau*D_max<=0.094819` and a
uniform strict-dominance margin `1-2*tau*D_max>0.81036`. These are bounds
over the capacity rails, not measurements of a selected source.

Consequently `q=0` has the unique pressure

```text
p_star=-d_p*A^(-1)*(C1-I)*h.
```

Since the forcing has zero sum and `1^T*A=1^T`, `sum(p_star)=0`. There is
no independent pressure amplitude or peak owner left to tune. Once the
primitive parameters are fixed, the actual peak is `max(abs(p_star))`;
ties may admit more than one canonical owner, but they describe the same
pressure. The strict positive source peak, source OZ ratio, multiplying
cutoff and acquired-rate threshold still have to hold for that pressure.
If it vanishes, the strict source requirements reject it.

**One shared form offset remains.** At the first endpoint, `q=0` is equivalent
to `L*(k*x1+b*v+d_p*y)=0`. The connected cycle therefore gives

```text
r1=-(b*v+d_p*y)/k,
r0=r1-delta*D*p_star,
x0=r0+eta1*1, x1=r1+eta1*1, x2=x1.
```

The last equality follows because the second form row uses held q, including
after Mutation (ZHIR). The phase event and subsequent coordination can still
change phase. Zero pressure at this boundary is therefore not an equilibrium
of the complete state, and a later refreshed pressure need not remain zero.

For the ten coordinates of `(r0,r1)`, the exact rail condition is

```text
eta1 in [-1-min(r0,r1), 1-max(r0,r1)],
max(r0,r1)-min(r0,r1)<=2.
```

A single-point interval is allowed because the rails are closed. Different
offsets for the two endpoints are not allowed. The original compiler uses
eta relative to its mean-zero pressure-potential inverse. That eta differs
from eta1 by one determined common constant. The implementation retains the
original gauge and all thirty labelled rail rows for comparison, even though
only the ten source/first-endpoint coordinate slots determine their interval.

**Exact substitution, with source admission still conditional.**
The private [zero-pressure reducer](../src/tnfr/research/_native_mutation_zero_pressure.py)
re-admits `NativeMutationSlice`, rebuilds the complete law and original rows,
checks the uniform invertibility premise, and uses the shared exact rational
five-by-five matrix inverse. It substitutes `p_star` into all nine original
zero-peak cells and eliminates their one common form offset. The source's
strict peak, OZ and history inequalities retain exact strictness; the successor
uses its actual zero-ratio Si masks. No division by a zero peak or zero-scaled
mask is used to certify those masks.

The resulting algebraic compatibility is necessary and sufficient for the
pressure/form rows of each fixed-parameter zero-peak cell. Actual controller
predicates, common-lift and renewal geometry, supplied execution context and
native realization retain their separate obligations. In particular, an exact
algebraic zero does not establish a represented negative-zero Mutation action.
The six surviving controller pairs and global source existence remain open.

The [independent controls](../tests/research/test_native_mutation_zero_pressure.py)
use the previously declared capacity frame, formal phase coordinate bases and
the previous algebraic phase frame. They check exact inverse residuals,
ten-constituent pressure/flow identities, common-offset equivalence, original
cell rows and rejected domains. These are coefficient and contract controls,
not admitted preparations or a parameter search. The preceding controller
exclusions, failed envelopes and frozen source evidence remain unchanged.

<a id="native-mutation-zero-pressure-obstruction"></a>
### Joint zero-pressure restrictions and an inconclusive rate test

The complete zero-successor-pressure equation also couples the endpoint form
rails to the actual zero-ratio selector masks. Keep the preceding law, notation,
both masks, coherent phase lifts and complete negative-outward target-minimum
renewal. The argument concerns multiplying Dissonance (OZ); the separate source
fallback result remains unchanged.

Write the stability weights as `(alpha,beta,gamma)`, target capacity as u and
actual maximum capacity as V. At zero pressure, the difference between each
surrounding high-Si mask and the target low-Si mask gives

```text
alpha*(v_j-u)/V >= si_hi-si_lo-beta*(abs((L*y)_0)-abs((L*y)_j))/pi_rep
                > si_hi-si_lo-beta.
```

The strict inequality uses the common arc, which gives `abs((L*y)_0)<pi_rep`.
The owned coefficient check has `kappa=(si_hi-si_lo-beta)/alpha>0`, so
`v_j-u>kappa*V>0` for every surrounding block. Constant endpoint potential and
the common form rails therefore imply

```text
d_p*(y_j-y_0) = k*(x1_0-x1_j)-b*(v_j-u) < 2*k-b*kappa*V.
```

A renewing negative Mutation (ZHIR) must put the target below the previous
phase hull: otherwise positive phase averaging contracts that hull. Thus the
postevent diameter satisfies `D_event<s+(2*k-b*kappa*V)/d_p<B`, where
`B=s+2*k/d_p`. If `D1=osc(y)` and D2 is the complete second-call diameter,
renewal and the second phase coordinator give
`D1<D2<=(1-G2)*D_event<(1-G2)*B`. This is an associated phase/capacity
restriction; it is not an independently selectable phase preparation.

Let `z=(y1+y4)/2-y0>0`, and let t be the near-block mean minus the far-block
mean. The preceding target-minimum renewal geometry gives
`z<A2*s/(G2+K2)` and `t<A2*(s+z)`, with `A2=1-G2-K2`.
The endpoint potential also gives `d_p*z<2*k`. Define

```text
H=min(A2*s/(G2+K2), 2*k/d_p),
r(z)=r0+r1*z,
r0=(si_lo-beta-gamma)/alpha, r1=beta/(alpha*pi_rep),
m=(si_hi-beta-gamma)/alpha.
```

The zero-pressure masks require `u<=V*r(z)` and `v_j>=m*V` for surrounding
blocks. Their near mean n obeys `b*(n-u)+d_p*z<=2*k`. When `m>r(H)`, these
joint relations license the uniform capacity ceiling

```text
U=min(VF_MAX*r(H),
      max_{0<=z<=H} r(z)*(2*k-d_p*z) / (b*(m-r(H)))).
```

The numerator is a concave quadratic; its maximum is obtained from the two
endpoints and clipped vertex. Zero is a closure endpoint of this calculation,
not an admitted active-capacity source or outward gap.

**Full transport bounds the pressure from phase forcing.** The exact identity
`A*p=d_p*(C1^(-1)-I)*L*y` follows from the preceding zero-pressure reduction.
The matrix `J=(C1^(-1)-I)*L` is circulant with zero row sums. If K is the
sum of its positive coefficients in one row, then
`abs((J*y)_i)<=K*osc(y)` by subtracting the common minimum of y. At a row
where `abs(p_i)=P`, the diagonal term of A is at least `1-tau*D_max`, while
the sum of absolute off-diagonal terms is at most `tau*D_max`. Consequently

```text
P <= d_p*K*D1 / (1-2*tau*D_max).
```

This uses row bounds as well as the complete `q=0` equation, and does not
assert positivity or contractivity of `A^(-1)`. The zero-pressure target mask
also yields `u/V<=(si_lo-gamma)/alpha`. The checked owned inequality
`f*(si_lo-gamma)/alpha<=ell` permits `D_max<=ell*VF_MAX` here.

For first gains `(G1,K1)`, put `A1=1-G1-K1` and
`det=A1*(A1-K1/2)-K1^2/4`. The grouped phase inverse gives
`c_h=A1/det-1`, `c_t=K1/(2*det)` and the exact target row

```text
(1-tau*f*u)*p0 + tau*ell*(v1*p1+v4*p4)/2 = d_p*(c_h*z+c_t*t).
```

Check positive determinant, nonnegative coefficients and all denominators
before taking upper bounds. The frozen calculation uses

```text
Dbar=min(A2*s/(G2+K2/2), (1-G2)*B),
Pbar=d_p*K*Dbar/(1-2*tau*D_max),
Fbar=d_p*(c_h*H+c_t*A2*(s+H)),
u*f*p0 <= f*U*(Fbar+tau*ell*VF_MAX*Pbar)/(1-tau*f*U).
```

**The fixed six-pair pass is inconclusive.** In order `(stable,stable)`,
`(stable,transition)`, `(transition,stable)`, `(transition,transition)`,
`(dissonant,stable)`, `(dissonant,transition)`, the rate ceilings are approximately
`0.143303`, `0.114018`, `0.132666`, `0.105984`, `0.128148`, and `0.101957`.
Every ceiling exceeds the strict acquired-rate threshold `xi=0.1`; none excludes
a remaining pair. These are upper bounds, not attainable rates, source witnesses
or probabilities. The separate maxima discard correlations even though each
bound follows from joint premises. No precision refinement, subdivision,
parameter search or native response was performed after this outcome.

The [same private owner](../src/tnfr/research/_native_mutation_zero_pressure.py)
provides `derive_native_mutation_zero_obstruction()`, a no-argument fresh-law
coefficient report with intermediate bounds and explicit `inconclusive` status.
The original reducer remains available. The
[independent controls](../tests/research/test_native_mutation_zero_obstruction.py)
check the exact phase/forcing identities, support norm, quadratic extremum,
coefficient admission and absence of source or cached-report consumption.
Global feasibility still requires all pressure, form, phase, controller and
history conditions at the same primitive parameters. A source search failure
would not prove exclusion, and algebraic zero would still not realize a native
negative-zero Mutation. Formation and physical identification remain separate.

<a id="native-mutation-joint-source-attempt"></a>
### Bounded joint source proposal: no admitted candidate

The [frozen protocol](../docs/assets/native_mutation_joint_source/protocol.txt)
tests the constructive question with all quantities associated to one source.
It targets strictly negative successor target pressure, `q0<0`, and a positive
successor peak. The all-zero-pressure boundary remains a separate open subcase;
a small numerical residual is never substituted for algebraic zero or its sign.

The complete two-call law, support, clock, supplied grammar/history, both actual
high-Si Coherence (IL) masks, multiplying Dissonance (OZ), acquired Mutation
(ZHIR), form rails and negative-outward target-minimum renewal remain unchanged.
The search uses thirteen coordinates: five capacities, four source phase
differences with `theta0=0`, and four independent source pressures with
`p4=-sum(p0,p1,p2,p3)`. The complete pressure-potential inverse then determines
all fifteen source/endpoint form offsets. Their one common interval eliminates
the remaining form origin; there is no independent endpoint adjustment.

The finite search restricts capacities to `[1/64,VF_MAX]` and source phase
differences to `[-3,3]`, with the original common width at most three. The
pressure coordinates lie within the necessary source bound
`Pbar=2*k+b*VF_MAX+3*w/pi_rep`; the fifth pressure must obey the same bound.
The positive lower capacity cutoff belongs to this search policy, not to the
complete model or a new physical hypothesis.

**Proposal budget and evidence.** The six surviving declared controller pairs
each use the same four Sobol initial points, with seed `20261018`. Each SLSQP
run has a limit of 400 iterations and 5,000 objective evaluations, including
finite-difference probes. The fixed objective penalizes violated normalized
constraints, with a desired margin `1e-6` for strict predicates. That margin
does not replace any exact admission threshold. Maxima, pressure ratios,
both Si fields, carried gains, both controller resultants and all geometry
are recomputed from the same proposed state. Binary64 calculations only rank
proposals; a solver termination flag certifies no source.

The [recorded summary](../docs/assets/native_mutation_joint_source/result-summary.json)
contains 71,204 objective evaluations across 24 runs; two runs exhaust their
evaluation budget. One lowest-penalty evaluated point per run is retained,
with the first point chosen in a tie. The remaining 22 optimizer success
flags describe numerical termination, not scientific admission. No additional
restart, objective change, threshold change or parameter search followed.
The [artifact manifest](../docs/assets/native_mutation_joint_source/manifest.json)
associates the protocol, environment and selected proposals, the losslessly
compressed full evaluation trace, exact results and source archive. Byte hashes
establish consistency, not authenticated chronology or scientific truth.

**Exact primitive readmission.** Each selected physical coordinate is converted
once to the nearest dyadic with denominator `2^40`, with ties to even. The
private [point verifier](../src/tnfr/research/_native_mutation_point.py) rebuilds
the complete law and compiler expressions from those primitives. It determines
the actual pressure peaks and tied owners, intersects all fifteen form-offset
slots, and uses their single midpoint. An empty interval is a rejection,
not a correction. Wrong target signs or unavailable canonical peak owners leave
the cell explicitly unavailable. Otherwise all original strict/weak cell rows
are checked at that point, including all thirty form rails and both Si masks.

The first secant is calculated from the associated ideal form samples at
`t=1` and `t=17/16`. No supplied temporal rate can replace those samples.
This remains a calculated ideal history, not measured native evidence.
Both controller predicates are admitted independently from their consumed
phase differences with the shared rational cosine enclosure, fixed at twelve
terms and 128 outward dyadic bits. A straddling interval is unresolved, not
passing; precision is not increased after seeing a result. The second Si mask
uses the pre-Mutation phases, while the second controller uses the post-Mutation
phases and carried gains.

**Finite result.** All 24 retained points are exactly rejected. Every retained
point fails both strictly negative successor target pressure and complete
phase-diameter renewal; some also fail other conditions. These statements
concern the 24 retained, rationalized points, not an exact assessment of all
71,204 optimizer evaluations. The rejected sources supply neither a joint
witness nor an all-domain contradiction. In particular, they do not prove that
negative pressure and renewal are universally incompatible. Both the remaining
nonzero-pressure domain and the separate zero-pressure domain stay open.

The [point controls](../tests/research/test_native_mutation_point.py) independently
reconstruct ten-constituent fields, flows, temporal samples and controller
enclosures on earlier formal coefficient frames. Explicit mocked wiring checks
cell and interval boundaries without adding searched sources. The
[read-only evidence controls](../tests/research/test_native_mutation_joint_source_evidence.py)
check archive association, the finite budget and selected evaluations, then
rebuild the consumed primitive fields and phase diameters that reject the
recorded candidates. They do not rerun the proposer or the point evaluator.

**Scope of the research observable.** Increasing complete-call phase diameter
is the renewal observable chosen for this study. It is not itself a grammar
requirement, a Mutation admission condition or a general criterion for collective
organization. This finite failure therefore cannot obstruct every canonical
operator investigation. Any change of observable requires its own scientific
justification and prospective declaration; it cannot turn the frozen failed
attempt into a positive result. Native execution, maintained collective identity,
formation and physical identification retain their independent obligations.

<a id="native-canonical-observable-admission"></a>
### Canonical action, identity and the phase-diameter observable

The scientific target is a declared NFR identity and its admitted structural
response under a complete law. A successful registered action, a maintained
identity and a useful diagnostic are different claims. The
[ontology](EMERGENT_ONTOLOGY.md#collective-grammar-readiness) and
[ordered-action criterion](STRUCTURAL_OPERATORS.md#collective-grammar-descent)
already supply their obligations; no new universal scalar criterion is added.

**What the renewal problem assumes.** The frozen two-call problem retains
paired doubled-C5 support, represented mixed pressure, held capacity, a
declared clock and held Euler step, zero Gamma and the native selector/controller. Its
target Dissonance (OZ) followed by Mutation (ZHIR), surrounding high-Si
Coherence (IL), strict complete-call diameter increase and remaining
negative-outward orientation are restrictions of that question. The finite
proposal adds its own capacity cutoff and numerical budget. None is a
general definition of an NFR or a canonical Mutation requirement. The
conditional exclusions and all 24 rejected retained points remain valid
with their original definitions; this audit does not reclassify them.

Mutation admission instead consumes an actual target with active capacity,
live associated signed EPI samples giving a secant above the configured threshold,
and independent grammar context. Current pressure determines its signed
phase action, not the historical growth sign. Invocation still requires a
supplied request or a declared selector. These contracts are implemented
choices whose structural necessity remains a separate foundational question.

The [existing acquired-history calculation](#native-mutation-renewal-admission)
already admits Mutation on both target twins while its accepted phase prefix
narrows the diameter. The later Resonance (RA) sign gate rejects independently;
there is no complete second call or maintenance conclusion. Conversely, that
owner's conditional expanding phase map lacks qualifying history on its
specified acquisition route. Expansion therefore neither supplies admission
nor is required by it. No new trajectory is needed to establish this distinction.

**A phase margin is not finite extinction or strict renewal.** Fix a declared
identity requirement `D>=epsilon>0`. In the five-action contraction domain,
define `N_epsilon=min{n>=0: q^n*D_0<epsilon}`, using exact comparisons. Any
successful prefix reaching that index has left this margin, or has already
left the theorem's premises earlier. Equality of the bound with epsilon
does not exclude the closed condition. The index counts calls, not physical
time, and proves neither successful continuation nor finite exact consensus.

The [phase-writer ledger](nodal/PRIMITIVE_PHASE_CLOSURE.md#native-phase-contrast-budget)
separately records intervening diameter increase r and realization defect e:
`D_next<=q*(D+r)+e`. At the level of this inequality, `e=0`,
`r=(1/q-1)*D_star` and `D_next=D=D_star>0` are compatible. Thus maintenance
with a phase margin need not mean `D_next>D` at every call, or even at one
call of a constant-diameter sequence. This is a counterexample to that
logical implication, not a native source, selected event or realized writer.
Such equality is impossible for positive D under the unchanged five-action
law with `r=e=0`. The ledger supplies no phase recharge mechanism or input.

**Diameter and resultant do not retain relational response.** On the same
unit paired C5, take homogeneous form and positive capacity, fix target block
zero and let `0<a<pi`. The phase lifts

```text
r=(0,a,0,0,0),       r_tilde=(0,0,a,0,0)
```

have the same phase multiset, diameter a and resultant magnitude
`abs(4+exp(1j*a))/5`. Nevertheless the freshly prepared target pressure is
`w_phase*a/(2*pi_rep)` in the first state and zero in the second: block one
is a neighbor of zero and block two is not. The form, capacity and degree
contrast channels vanish in both. No relabeling may discard the fixed target
and its support association. Even the combined observation `(D,R)` therefore
fails to determine this consumed pressure or its unforced form rate. This
does not claim that the two states have different default operator marks.
The [static controls](../tests/physics/test_native_canonical_observable_scope.py)
use `a=1/4`, independently compare the prepared pressure and check the exact
margin/ledger distinctions. They install no history, event or selection law.

Current nodal channels alone also fail to determine Mutation admission when
associated histories differ. The existing
[context controls](../tests/operators/test_collective_grammar_context.py)
hold the physical state fixed and distinguish fresh positive growth from a
flat record, or equal recent traces with different prior Coherence/debt.
These are supplied-context counterexamples, not acquired histories. Together
the controls identify distinct missing information: relational geometry and
target-associated execution context. Resultant magnitude does not solve these
omissions; any proposed coherence score needs its own state-sufficiency argument.

**Canonical continuation contract.** Use the existing sufficient candidate
`Q=(M,d,V,e,r,H)` with its complete law, support, clock and retained inputs.
For a declared action, equal Q must give equal admission decisions and equal
projected successors; a retained occurrence mark needs its own equality as
well. Rejection, abstention and committed prefixes retain their actual state.
If an identity predicate `I(Q)` is claimed, declare it and its admissible
symmetries independently, then check it on the relevant prefixes and flow
intervals. Admission does not establish `I`, and a prospective predicate must
not be chosen merely because the evaluated source satisfies it. Formation
additionally requires initially absent identity and subsequent acquisition.

The common-origin quotient does not retain absolute Mutation telemetry such
as before/after angles and quadrant labels. Preserve their reference or mark
them as reference-dependent outputs; semantic-state descent is not whole-graph
equivariance. Retaining sufficient state likewise proves neither a smaller
autonomous macro-node nor an autonomous occurrence law. This admission result
returns the question to source-bound identity and complete action, while the
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone chooses the next proof obligation.

<a id="native-mutation-pair-closure"></a>
### Mutation preserves paired phases only for matched realized shifts

The joint native state retains unequal constituent forms and capacities but
one primitive phase per pair. Its [five-action theorem](#native-relative-phase-descent)
does not establish preservation of that chart by Mutation (ZHIR). The
[general equal-shift criterion](nodal/SINE_COLLECTIVE_MUTATION_ACTION.md#5-collective-descent-and-complete-word-context)
has an existing event owner; the result here specializes it to the native
mixed-pressure law, actual constituent admission and sequential execution.
No source search, acquired-history claim or diameter-renewal condition is used.

**Domain, admission and origin.** Retain fixed unit undirected doubled C5,
consecutive nonadjacent twins and declared node order. Use signed forms,
positive individual capacities, associated temporal records, grammar and the
direct native glyph map. This is a supplied-target action question. A default
selector's request still requires its own live Si, pressure and lag admission.
Direct `apply_glyph` validates Mutation's temporal/capacity gate and grammar
state representation; it does not itself enforce incremental grammar. The
strict native selector or an explicitly declared grammar-enforcement caller
must admit each request. Setting `GRAMMAR_REJECTION_MODE="raise"` alone does
not add a grammar call to direct dispatch.

For the exact equations use a common real phase lift and mathematical circle
period, with represented coefficients treated as exact parameters. Runtime
endpoints retain their binary64 proposal/wrap arithmetic separately. Supplied
temporal samples end at the live form and declared clock, and their positive
secants must independently exceed the configured threshold. Their presence
does not prove that the current law acquired them. Mutation changes no form,
capacity or stored pressure; it retains its phase telemetry and glyph context.

**Exact chart boundary.** On the incoming paired-phase chart, the existing
fresh mixed-pressure rows give `p_-=B+C`, `p_+=B-C`, with `C=k*d+b*e`.
These equal the consumed stored pressures only if preparation and intervening
events justify that association. For two actually admitted unwrapped shifts
`delta_-` and `delta_+`, write the extended phase coordinates as

```text
theta_-=mu-a,                  theta_+=mu+a,
mu_next=mu+(delta_-+delta_+)/2,
a_next=a+(delta_+-delta_-)/2.
```

Starting at `a=0`, circular pairing survives if and only if
`delta_+-delta_-` is an integer multiple of `2*pi`. This follows by subtracting
the two phase endpoints; the mean/contrast rows are their sum and difference.
An inherited coherent lift with equal small shifts gives `a_next=0`.
For the default nonzero signed shift s, with `2*s` not a full turn and both
pressures strictly nonzero, the two signs agree exactly when

```text
(B+C)*(B-C)>0  <=>  B^2>C^2  <=>  abs(B)>abs(C).
```

If `abs(B)<abs(C)`, the signs are opposite: `mu_next=mu` and
`a_next=-s*sign(p_-)`. Identical operator names and positive prior EPI growth
therefore do not preserve paired phases. This is exit from the old coordinate
family, not proof that an independently specified organizational identity dies.

At `abs(B)=abs(C)>0`, one pressure is zero; its stored sign bit must agree
with the nonzero pressure's sign for the default shifts to match. If both
pressures are zero, both actual sign bits matter. Exact real cancellation
does not determine binary64 signed zero. A shared fixed shift preserves
pairing when both events execute, independently of pressure. Zero shifts,
full-turn differences and other configured factors require the general
modular criterion rather than the default nonzero-sign shortcut.

**Sequential prefixes and abstention.** After the first even-twin shift
`s*sigma_-`, but before the second action, the pair already has
`mu_next=mu+s*sigma_-/2`, `a_next=-s*sigma_-/2`. A second equal shift restores
pairing at the completed-pair boundary. It cannot make the earlier singleton
prefix paired. Each target still needs its own live history and grammar.

On direct dispatch, invalid second-target temporal evidence raises before
that target's mutation. On the native selector path, missing/stale/subthreshold
evidence normally substitutes Coherence (IL) before grammar and records the
abstention after successful dispatch. The resulting ZHIR/IL pair retains the
half-shift split; IL also changes its own held pressure. Strict grammar does
not disable this temporal substitution. If the second target has valid
temporal evidence but fails strict grammar, the earlier Mutation remains
committed and the batch stops. The native batch has already incremented its
lag counters. These outcomes differ from all-target atomic rejection.

The [action controls](../tests/operators/test_native_mutation_pair_closure.py)
use fixed analytic preparations with `M=1/4,d=1/16,V=1,e=1/8`, initially
common phase one, and adjacent-block mean-form shifts `0,+1/2,-1/2`.
They independently reconstruct B and C before actual direct actions and
distinguish opposite-sign splitting from either common sign. Their matching
timestamped samples and AL/IL/OZ context are supplied. Separate controls retain
signed-zero/fixed branches, direct rejection and the actual native batch's
IL abstention versus strict grammar failure. The batch uses a supplied callable;
none of these cases is a default-selector occurrence or a full trajectory.

**The split contains consumed relational information.** For all five blocks,
retain `theta_(2i)=mu_i-a_i`, `theta_(2i+1)=mu_i+a_i` alongside `M,d,V,e`.
Require one common lift of width less than `min(pi,pi_rep)`; width at most
three is sufficient here. Then `cos(a_i)>0`, and each twin pair contributes

```text
exp(1j*(mu_i-a_i))+exp(1j*(mu_i+a_i))=2*cos(a_i)*exp(1j*mu_i).
L_i=Arg_I(cos(a_(i-1))*exp(1j*mu_(i-1))
          +cos(a_(i+1))*exp(1j*mu_(i+1))).
```

The open-semicircle cone makes this resultant nonzero and selects its
representative inside the common lift I. Both constituents of block i see
the same four neighbors, hence the same L_i. With `t=w_phase/pi_rep`, the
**next fresh preparation** is exactly

```text
B_i=t*(L_i-mu_i)+k*(N_i(M)-M_i)+b*(N_i(V)-V_i),
C_i=k*d_i+b*e_i+t*a_i,
p_(2i)=B_i+C_i,                  p_(2i+1)=B_i-C_i,
P=max_i(abs(B_i)+abs(C_i)),       Vmax=max_i(V_i+abs(e_i)),
Si_(2i)=alpha*(V_i-e_i)/Vmax+beta*(1-abs(L_i-mu_i+a_i)/pi_rep)
         +gamma*(1-abs(B_i+C_i)/(P or 1)),
Si_(2i+1)=alpha*(V_i+e_i)/Vmax+beta*(1-abs(L_i-mu_i-a_i)/pi_rep)
           +gamma*(1-abs(B_i-C_i)/(P or 1)).
```

Here `P or 1` denotes the exact zero-peak convention. Regular support removes
degree contrast. Form/capacity neighbor contrasts cancel; the own phase
contrast does not. The width and positive-capacity premises keep these Si
rows in their unclamped domain. Other domains retain actual wrapping,
resultant availability and the outer clamp. Never replace held pressure or
prepared Si inside the current event batch with these refreshed values.

Two effects require a: its signed own contribution `t*a_i` and its neighboring
resultant weight `cos(a_i)`. To isolate the latter, keep phase means
`(0,1/4,1/2,-1/2,-1/4)` plus common origin one, and split only block zero
by `a_0=s`, `0<s<pi/2`. Block one's neighbor angle, relative to that origin,
changes from `1/4` to

```text
Arg(cos(s)+exp(1j/2))=1/4+atan(tan(s/2)^2*tan(1/4)) > 1/4.
```

Multiplying by `exp(-1j/4)` gives real part `(1+cos(s))*cos(1/4)` and
imaginary part `(1-cos(s))*sin(1/4)`, proving the identity and strict sign.
Unchanged mean phases therefore do not determine next pressure. Reversing
a leaves the cosine weights unchanged but changes the own signed pressure
contrast, so magnitudes alone do not retain the required association either.
The [static readout controls](../tests/physics/test_native_split_phase_readout.py)
compare independent block formulas with shared fine pressure and Si, including
nonzero form/capacity contrasts. They do not execute a trajectory or create history.

**A coordinator contracts the split without restoring exact pairing.** Let g,l
be the actual shared coordinator gains, with their retained controller/load
state. Its global target b and resultant magnitude are obtained from
`sum_i cos(a_i)*exp(1j*mu_i)`, with `R=abs(sum)/5`. The fine affine proposals
give

```text
mu_i_next=(1-g-l)*mu_i+g*b+l*L_i,
a_i_next=(1-g-l)*a_i.
```

Default gain rails have `0<1-g-l<1`. A nonzero ideal split remains nonzero
after this coordinator. Thus, conditional on a completed call with no other
phase writer, held form flow, phase-inactive capacity adaptation and the
existing REMESH-abstention premises, the old paired-phase chart is not
restored. This is an ideal conditional consequence, not a guarantee that
the native batch completes, an exact binary64 inequality near zero, or
maintenance of a newly formed NFR.

The extended coordinates `Q_plus=(M,d,V,e,r,a,H)`, `r_i=mu_i-mu_0`, retain
nine relative fine-phase coordinates instead of four. They reconstruct the
fine phases up to a common origin in this chart. Jointly permuting twins
also transports `d,e,a` and their histories/ports; independent inventories
lose that association. Absolute Mutation telemetry still needs its reference.
This adds information already present in the fine engine, not a new force,
operator or constitutive law. The pressure/readout and coordinator rows are
closed on this retained state in their domains; a complete future operator
word and any organizational identity require their own admission and proof.

<a id="native-split-phase-resonance"></a>
### Resonance on the constituent-specific split-phase chart

The [Mutation boundary](#native-mutation-pair-closure) retains an internal
phase contrast because admitted constituents need not keep a common phase.
Resonance (RA) consumes that contrast through its live U3 neighborhoods. The
earlier [partial-subset result](NATIVE_OPERATOR_DYNAMICS.md#native-resonance-partial-subset)
already permits split neighbors of a paired target. The additional boundary
here is that the **two target constituents select different subsets of the
same neighbor snapshot**. Their phase targets, signed form means and capacity
triggers can consequently differ.

**Domain and execution.** Keep the fixed unit undirected doubled C5, ordered
nonadjacent twins, positive capacities and admitted signed forms. Use the
extended chart `Q_plus` above, a common phase lift I of width less than
`min(pi,pi_rep)`, and associated constituent kinds and grammar context.
These are supplied states and requests, not histories acquired by Mutation
or requests produced by the default selector. Each direct action below uses
an explicitly declared strict grammar-enforcement caller before shared glyph
dispatch. The native selector path retains its own preparation, lags and
sequential admission. No continuous trajectory, threshold search, new operator
implementation or collective executor is introduced.

The equations describe the ideal action with represented coefficients and
an admitted subset. Actual binary64 angle comparisons, resultant arithmetic,
two-stage form blending, wrapping and boundaries remain separate numerical
operations. The fixed controls below keep a strict U3 margin; a tolerance
comparison does not certify a boundary decision for arbitrary inputs.

**Live constituent selection.** Write the current associated coordinates as

```text
x_(i,sigma)=M_i+sigma*d_i,    v_(i,sigma)=V_i+sigma*e_i,
theta_(i,sigma)=mu_i+sigma*a_i,                    sigma in {-1,+1}.
J_(i,sigma)={(j,tau): j in {i-1,i+1}, tau in {-1,+1},
             abs(wrap(theta_(j,tau)-theta_(i,sigma)))<=L}.
```

Indices are modulo five; the even constituent has sign -1. The actual
configured U3 limit satisfies `0<=L<=pi_rep/2`. Each admitted subset must be
nonempty. It counts selected fine neighbors once, without conductance
weighting. A selected neighbor does not imply admission of its twin. All
neighbor phases are admitted before selecting; only selected forms enter
the propagation mean. Recompute these sets after any earlier adjacent action.

For each target and adjacent block j, let n be the number of selected members
and m their signed count (selected odd minus selected even). Then

```text
N_sigma=sum_j n_(j,sigma),
S_sigma=sum_j (n_(j,sigma)*M_j+m_(j,sigma)*d_j)/N_sigma,
Z_sigma=sum_j exp(1j*mu_j)*(n_(j,sigma)*cos(a_j)
                          +1j*m_(j,sigma)*sin(a_j)),
eta_sigma=Arg_I(Z_sigma).
```

The selected phasors lie in the same open semicircle, so their projection
onto its midpoint direction is positive and Z is nonzero. The formula for
Z follows by adding the selected positive/negative phase exponentials.
Whole-block selection has m=0; selecting one member retains both its signed
form contrast and its signed phase contribution. The cosine weights used by
the fresh four-neighbor pressure are therefore insufficient for this action.
Neither unordered contrasts nor independent form and phase inventories
recover S and eta. Preserve their common constituent/port association.

**Admission and pair action.** Denote the actual factors by
`rho=RA_epi_diff`, `b=RA_vf_amplification`, `c=RA_phase_coupling`, with
`0<rho<=1`, `b>0`, `0<c<=1`. Let T be the shared represented amplification
trigger, and let clip denote the admitted operator boundary policy. Each
target proposes

```text
F_sigma=clip((1-rho)*(M_i+sigma*d_i)+rho*S_sigma),
u_sigma=I(abs(S_sigma)>T),
v_sigma_star=(1+b*u_sigma)*(V_i+sigma*e_i),
theta_sigma_star=(1-c)*(mu_i+sigma*a_i)+c*eta_sigma.
```

The common lift justifies the last row; outside it retain the actual circular
policy and resultant availability. A strict nonzero form-sign inversion
rejects. Exact zero remains neutral, and an established kind must independently
survive the selected-neighbor kind rule. A finite nondecreasing capacity
proposal, finite arithmetic, U3 and grammar are additional gates. The trigger
uses the selected **signed mean**, not selected magnitude or the block mean.
An admitted zero selected mean suppresses amplification without suppressing
form or phase propagation. Kind/sign rejection precedes target-state, glyph
history and RA-telemetry writes; it does not promise rollback of graph caches
or an earlier accepted target.

For consecutive actions on the two nonadjacent twins, the first changes none
of the second's neighbors. If both requests pass, their completed-pair moments
are consequently

```text
M_i_star=(F_-+F_+)/2,                  d_i_star=(F_+-F_-)/2,
V_i_star=V_i+b*(u_-*(V_i-e_i)+u_+*(V_i+e_i))/2,
e_i_star=e_i+b*(u_+*(V_i+e_i)-u_-*(V_i-e_i))/2,
mu_i_star=(1-c)*mu_i+c*(eta_-+eta_+)/2,
a_i_star=(1-c)*a_i+c*(eta_+-eta_-)/2.
```

With inactive clipping, the first two rows become
`M_i_star=(1-rho)*M_i+rho*(S_-+S_+)/2` and
`d_i_star=(1-rho)*d_i+rho*(S_+-S_-)/2`. Thus unequal subsets introduce a
contrast source and can introduce unequal amplification even from equal
capacities. The older common-subset multiplier is a special case, not a
coefficient transferable to every split target. If only one action executes,
average its accepted endpoint with the untouched twin instead. Rejection
does not complete that pair.

**Order preservation on the same neighbor snapshot.** Suppose `a_i>0` and
both target subsets are nonempty. Their selected mean angles satisfy
`eta_-<=eta_+`. To prove this, move the closed interval `[theta-L,theta+L]`
right from the even to the odd target. If the selected sets overlap, every
removed phase is below every common phase and every newly selected phase
is above every common phase. Within the open-semicircle cone, adding lower
phasors cannot raise the resultant angle, and adding higher phasors cannot
lower it. Thus the mean of the common set lies between the two means. If
the sets are disjoint, all phases of the first set are below those of the
second, giving the same ordering. Boundary ties retain weak inequalities.
Consequently,

```text
a_i_star >= (1-c)*a_i > 0,                         for 0<c<1.
```

For negative a, exchange the constituent labels to obtain the corresponding
absolute-value bound; their associations must move with the labels. At a=0
the two sets and mean angles agree. The default factor is strictly below one,
so this ideal admitted pair action cannot exactly remove an existing split.
The bound is sharp when both targets select the same mean; `c=1` can merge
them in that case. It is not a bound for different neighbor snapshots,
undefined resultants, failed actions or binary64 differences near zero.

**Internal separation need not contract.** A static default-limit control
sets target block zero to `mu_0=2,a_0=1/8`, and both neighboring blocks to
`mu_j=2,a_j=3/2`. Their phases occupy the common interval `[1/2,7/2]` of
width three. For either target, the same-sign neighbor gap is `11/8<pi_rep/2`
and the opposite-sign gap is `13/8>pi_rep/2`. Hence the even target selects
only phases 1/2, and the odd target only phases 7/2. Both selected resultants
are defined. Positive interior forms with a common kind independently admit
the identity gates. For the default positive phase factor,

```text
eta_-=1/2,    eta_+=7/2,
a_0_star=1/8+(11/8)*c > 1/8.
```

Each target moves toward its own compatible neighbors, while the unchanged
neighbor extrema keep the global phase hull fixed. This is compatible with
the earlier global-hull retention argument. It refutes universal contraction
of an internal twin contrast under RA; it does not establish global diameter
growth, a completed native-call increase, sustained separation or formation.
The later coordinator still has its own contraction row and actual gains.

**Checks and composition boundary.** The
[static action controls](../tests/physics/test_native_split_phase_resonance.py)
also use target phases `1,3/2`, neighboring block-one phases `5/2,11/4`, and
block-four phases `1/4,1/4`. The default subsets are respectively `{2,8,9}`
and `{2,3,8,9}`. Reversing block one's form contrast at fixed means changes
only the first target's selected mean. Independent predictions check each
form, circular phase target, capacity and reconstructed pair contrast.
Later-target controls distinguish live adjacent propagation from a frozen
snapshot and preserve an accepted first action when a second fails its
independent sign or kind identity gate. These controls are supplied action
requests with explicit grammar, not a default-selector occurrence witness.

RA changes no stored pressure. A later adjacent RA consumes the updated
neighbor form/phase/kind and its own capacity, without refreshing the batch's
prepared pressure or Si. Neighbor capacity changes enter later preparation
and adaptation, not that local RA mean. Histories and bookkeeping follow the actual
execution path. Atomic stages instead propose from one immutable snapshot;
they cannot replace this sequential map. `Q_plus` retains enough associated
coordinates to reconstruct these local decisions, but that invertible chart
is neither a smaller autonomous NFR nor a new identity predicate. A complete
call must separately join the admitted event prefix to held-pressure flow,
individual temporal/rate records, coordination and capacity adaptation.

<a id="native-split-phase-call"></a>
### Conditional complete-call composition with split phases

The [constituent-specific Resonance action](#native-split-phase-resonance),
[joint held-flow calculus](#native-joint-collective-calculus) and
[split readout/coordinator](#native-mutation-pair-closure) compose on the same
retained state. The essential distinction is between post-event capacity,
held pressure and terminal adapted capacity. Their association determines
the form response and the temporal evidence supplied to a later request.
This is a conditional complete-law description, not a new execution API.

**Scope and order.** Fix unit undirected doubled C5, consecutive nonadjacent
twins, node order `0..9`, signed forms, positive capacities and one common
phase lift of width less than `min(pi,pi_rep)`. Retain `Q_plus`, associated
context, incoming rates, the declared clock and the actual request origin.
Consider one admitted complete batch with either Resonance (RA) on both
members of a pair or Coherence (IL) on both. Each member has one request and
its own strict live grammar and physical gates. Incoming phase splits may
be supplied or separately acquired; Mutation is not inserted as another
request on those targets during this call.

Use default mixed pressure and Si preparation, the live built-in zero Gamma
source, a single Euler substep `delta=1/16`, and default phase/capacity policies.
Require every event, flow and adapted form/capacity to remain strictly inside
its rails; canonical post-flow clamps then leave those channels unchanged.
The exact equations use ideal arithmetic with represented parameters and a
declared lift. Represented phase storage, rounding, finite arithmetic and
decision boundaries remain separate execution obligations. No external
callbacks, optional mathematics engine or support-changing event is included;
validators must pass and automatic REMESH must independently abstain.

The actual runtime order is: left temporal boundary, fresh pressure/Si,
prepared requests with live sequential admission, post-event temporal
boundary, held integration, canonical clamps, phase coordination, capacity
adaptation, final temporal boundary, whole-support EPI archive and REMESH
admission. Failure stops at its actual prefix. Rejection does not supply the
missing flow, coordinator or terminal state. A supplied request schedule
does not establish that the default selector chooses that schedule.

**Event moments and held flow.** Reconstruct incoming pressures `B_i+/-C_i`
and prepared Si from the split-phase readout above. After the sequential
batch, denote its actual admitted endpoints by

```text
x_(i,sigma)^E=Y_i+sigma*D_i,     v_(i,sigma)^E=U_i+sigma*E_i,
p_(i,sigma)^E=qbar_i+sigma*qdelta_i,
theta_(i,sigma)^E=psi_i+sigma*alpha_i.
```

On an RA pair, use the two distinct selected means, phase targets and
amplification indicators from the preceding action theorem; `qbar=B` and
`qdelta=-C`. On an IL pair, form, capacity and phase retain their incoming
values, while `(qbar,qdelta)=IL_dnfr_factor*(B,-C)`. Earlier adjacent events
enter each RA's live subset and propagation inputs. They never replace the
prepared pressure or Si by a refreshed value within this batch.

Expanding the two fine nodal products gives

```text
fbar_i=U_i*qbar_i+E_i*qdelta_i,
fdelta_i=U_i*qdelta_i+E_i*qbar_i,
M_i^F=Y_i+delta*fbar_i,          d_i^F=D_i+delta*fdelta_i,
f_(i,sigma)=fbar_i+sigma*fdelta_i.
```

The stored rate is `f_(i,sigma)`. Its stored acceleration is
`(f_(i,sigma)-old_rate_(i,sigma))/delta`, retaining each old rate's association.
For RA, the mean is `U_i*B_i-E_i*C_i`, where C includes the **incoming**
phase contrast. Substituting the new phase contrast into C would silently
change the held-pressure law. Replacing the mean product by `U_i*qbar_i`
loses the exact term `E_i*qdelta_i`; its one-step form error is
`delta*E_i*qdelta_i`. Unequal RA triggers can create that term even when
incoming capacities are equal. All these are identities of the admitted
held model, not licenses to reconstruct independent pressure from response.

**Individual temporal evidence.** Let `x_n` be the live form at the initial
boundary t and `x_n^E` its event endpoint. If they differ, the recorder resets
that node's same-time record to `[(t,x_n^E)]`. If they agree, it retains its
admitted earlier history. The final recorder appends the flow endpoint at
`t+delta`, with the existing bounded history length. Thus the last two samples
give the ideal secant

```text
(x_n^F-x_n^E)/delta=f_n.
```

The event jump divided by delta is excluded. RA and IL constituents may
therefore keep different history lengths despite sharing the same clock.
Represented samples give their actual represented secant; exact equality to
the separately rounded stored rate is not assumed. Capacity adaptation later
changes neither these form endpoints nor the already stored rate. Their
values remain measured-flow evidence, not a prediction using terminal capacity.
The next Mutation request still needs its active-capacity, threshold and
independent grammar gates; a valid record alone does not select or admit it.

**Coordinator and relative state.** Rebuild the post-event phase targets from
all four support neighbors, not the RA-compatible subset:

```text
L_i^E=Arg_I(sum_(j=i-1,i+1) cos(alpha_j)*exp(1j*psi_j)),
b^E=Arg_I(sum_j cos(alpha_j)*exp(1j*psi_j)),
R^E=abs(sum_j cos(alpha_j)*exp(1j*psi_j))/5.
```

All these resultants are nonzero on the common open semicircle. The controller
consumes this R, the actual updated glyph-history load and retained incoming
gains. Let g,l be its resulting gains, not restarted defaults. The shared
coordinator gives

```text
mu_i^C=(1-g-l)*psi_i+g*b^E+l*L_i^E,
a_i^C=(1-g-l)*alpha_i,
r_i^C=(1-g-l)*(psi_i-psi_0)+l*(L_i^E-L_0^E).
```

Retain the gains and their history for the next invocation. A common rotation
cancels from the relative row, R and local decisions, extending the existing
ideal origin quotient to the split chart. Absolute telemetry still requires
a reference, and bit-exact rotation symmetry of the implementation is not
asserted.

For a complete RA pair with incoming `a_i!=0`, the same-neighbor-snapshot
order theorem and default `0<c<1`, `0<1-g-l<1` imply

```text
abs(a_i^C)>=(1-g-l)*(1-c)*abs(a_i)>0.
```

For IL the corresponding equality is `abs(a_i^C)=(1-g-l)*abs(a_i)`.
Meanwhile every RA event stays in the current common phase hull and the
coordinator gives `diameter_next<=(1-g)*diameter_in`. Nonzero internal
separation at every admitted finite call is therefore compatible with global
contraction. This neither supplies an infinite admitted sequence nor a fixed
positive identity margin. Unequal requests inside a pair, Mutation, other
phase writers or failed admission require their own phase-order argument.

**Capacity adaptation and next preparation.** Qualification uses the stored
prepared Si and post-event held pressure. Apply its actual threshold/count
policy per constituent and let `h_-,h_+` be the resulting due flags. With
`hbar=(h_-+h_+)/2`, `hdelta=(h_+-h_-)/2`, `m=VF_ADAPT_MU` and the immutable
event-capacity neighbor mean `W_i=(U_(i-1)+U_(i+1))/2`,

```text
V_i^A=U_i+m*(hbar*(W_i-U_i)-hdelta*E_i),
e_i^A=E_i+m*(hdelta*(W_i-U_i)-hbar*E_i).
```

The existing fixed-point/hull guards and admitted capacity rails remain in
force. Counters count qualifying calls and do not reset after a due write.
If `v_n^A!=v_n^E`, replacing the consumed capacity in this call's rate would
add the spurious term `(v_n^A-v_n^E)*p_n^E`. The new capacity belongs to
subsequent laws and observations. Refreshing Si before this writer would also
change the selected adaptation policy.

The physical successor is `(M^F,d^F,V^A,e^A,r^C,a^C)`, together with the actual
updated H: rates, samples, kinds, grammar/debt, counters, lags, ordinals,
controller records and REMESH evidence. Its old stored pressure/Si remain
distinguishable from a next preparation. To evaluate that preparation, rebuild
the split pressure/Si equations using these terminal coordinates, including
their new pressure peak and maximum individual capacity. Re-admit every
next request and live prefix. Current-call success alone proves no next-call
request pattern, identity preservation or future successful execution.

**Verification boundary.** The
[composition controls](../tests/physics/test_native_split_phase_composition.py)
freeze the existing asymmetric cancellation geometry, supplied RA requests
on nodes 0/1 and IL on nodes 2..9, distinct incoming rates and associated
temporal samples. The selected signed means at the two RA targets are zero
and `5/32`, so their capacity multipliers differ. Independent predictions
precede shared stage execution in runtime order. They test the covariance,
individual resets, controller/coordinator, actual adaptation eligibility and
terminal refreshed readouts. Separate algebraic adaptation controls retain
their own supplied flags rather than inventing eligibility for this source.
The checks execute neither `runtime.step` nor a trajectory campaign; existing
runtime wiring controls supply the stage-order integration boundary. This
closes the declared conditional composition, not a smaller autonomous NFR,
acquired source, default-selection witness or maintained collective identity.

<a id="native-aggregate-resonance-obstruction"></a>
[The aggregate Resonance capacity obstruction](NATIVE_AGGREGATE_OPERATORS.md#native-aggregate-resonance-obstruction)
and its separate [sufficient inheritance domain](NATIVE_AGGREGATE_OPERATORS.md#native-resonance-inheritance-domain)
now belong to the aggregate operator owner. This anchor preserves published links.

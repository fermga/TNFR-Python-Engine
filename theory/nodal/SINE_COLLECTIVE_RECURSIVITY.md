# Recursivity: associated history and collective sine dynamics

REMESH has a precise conditional role on the organized sine objects. Its
registered node-level action records an advisory; its separately invoked
network map mixes each member's present form with that member's delayed
forms. The latter is a genuine hybrid form reset, with a calculable effect
on storage and subsequent phase motion. It does not derive a delay law from
continuous nodal dynamics or automatically select when to apply it.

This owner specializes existing mechanisms to the
[weighted sine law](SINE_COLLECTIVE_RECEPTION.md#1-complete-model-and-event-boundary)
and [contextual collective chart](SINE_COLLECTIVE_IDENTITY_TRANSITION.md#2-a-contextual-chart-for-the-exact-twin-stratum).
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md) owns task order;
the [runtime REMESH derivation](../REMESH_INFINITY_DERIVATION.md#11-exact-raw-update)
and [event contracts](../../docs/contracts/OPERATOR_EVENTS.md) retain the
general map, transactions, history advancement and recurrence results.

## 1. Three distinct operations and their admission

The public `Recursivity(depth=...)` operator and its atomic
[`execute_recursivity_stage`](../../src/tnfr/operators/network_stage.py)
leave structural form, phase, capacity, pressure and support unchanged
before any separately supplied refresh callback. They record the target's
actual glyph/lifecycle context and one graph-level advisory per runtime
ordinal (the legacy retained-sample index outside that runtime).
Equal physical states with different histories or advisory ordinals
therefore need not have equal execution descriptions.

REMESH is a registered generator/closure. Declared depth must be a positive
integer; depth greater than one invokes the actual word's U5 stabilizer
requirement. A validated word and live admission remain separate from the
availability of delayed forms. Passing the advisory does not execute a
temporal echo, create a child, prove fractal structure or select a delay.
When the optional application precondition is enabled, `REMESH_MIN_NODES`
is a strict nonnegative integer (default 2); zero remains an admitted
configuration. Boolean, textual, floating or negative counts reject before
the advisory writes. This uses the shared operator argument validator.

The explicit [`apply_network_remesh`](../../src/tnfr/operators/remesh.py)
operation instead consumes graph `_epi_hist`, a uniform factor `0<a<=1`,
strict positive integer local/global delays, and the clipping policy. It
does not execute the glyph or supply its grammatical justification. Its
caller must declare any composition with a grammatical word and its event
schedule. No implicit bridge from advisory to mixing is assumed here.

The optional `apply_topological_remesh` helper is yet another supplied
operation. A change of support would change degrees, weights, pressure,
storage and the collective chart. It is excluded from every fixed-support
formula below and cannot be inferred from the delayed-map name.

The delayed map requires at least `max(tau_local,tau_global)+1` indexed
snapshots. It returns an explicit no-op for insufficient history or empty
live support. Selected lag entries must contain exactly the current node
support, with admitted signed scalar forms (including uniform-real BEPI).
Missing or invalid historical members cannot be replaced by current values.
Both selected rows are admitted even when a coefficient vanishes at a=1.
The runtime reads lag tau at index `-(tau+1)`; the newest history row is not
substituted for live x. Local/global label temporal lags, not a neighborhood
average and a network-wide mean.

## 2. Complete state and the actual reset

Between events retain a fixed finite connected simple undirected support
with at least two nodes and no self-loops, actual nonnegative conductances W,
bare degrees d, primitive circular phase theta, held nonnegative capacities
nu, zero forcing/loss and the declared structural clock:

\[
 q=L_Wx,\quad S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i),\quad
 M=\operatorname{diag}\frac{\nu_i}{\pi d_i},\quad
 p_i=\frac{S_i}{\pi d_i},\quad \dot x=MS,\quad\dot\theta=Mq.
 \tag{1}
\]

For the event retain its selected node-associated lag fields y,z, row order,
history support, delays, primitive representations, clipping bounds/mode,
factor source, runtime clock and consumed execution context. For repeated
use retain the full required indexed suffix and its advancement rule;
the two currently selected rows alone need not determine the next selection.
Association matters: a multiset of past values does not say which member
receives which value. `_epi_hist` is distinct from per-node mutation evidence
and glyph history. A supplied temporal sampling model is necessary before
an index delay is interpreted as elapsed structural or laboratory time.

In exact real arithmetic set

\[
 b=(1-a)^2,\quad c=a(1-a),\quad e=a,\quad b+c+e=1,
 \qquad r=bx+cy+ez.
 \tag{2}
\]

The implementation evaluates the nested binary64 expression
`(1-a)*((1-a)*x+a*y)+a*z`, then structural clipping. It exposes raw and
bounded node proposals. Let x+ be the actual admitted committed endpoint,
including those interventions, and h=x+-x. Form changes; theta, nu, support
and stored pressure do not. The executor does not append or shift `_epi_hist`.
Its timestamped EPI boundary recorder prevents a same-time form jump from
being misread as a continuous two-sample Mutation rate. That recorder is not
the delayed-history advancement rule.

The shared executor plans before writing and protects graph-owned state
against commit/observer failure. External callback effects have their own
limits. The formulas use the existing executor and sine field reader; no
new reset implementation or alternate history interpreter is introduced.

## 3. Exact storage work and a history-dependent passivity condition

Use the selected storage

\[
 H(x,\theta)=F(x)+V(\theta),\qquad
 F(x)=\tfrac12x^TL_Wx,\qquad
 V(\theta)=\sum_{\{i,j\}}(1-\cos(\theta_j-\theta_i)).
\]

Phase is fixed during this jump, so the existing form-reset identity gives

\[
 \Delta H=q^Th+\tfrac12h^TL_Wh.
 \tag{3}
\]

This is exact for the actual represented/clipped endpoint interpreted as a
real vector. Nonpositive work holds if and only if
`q^T h <= -h^T L_W h/2`; it is not guaranteed by the operator label.

For the ideal unclipped endpoint r, quadratic convexity gives the stronger
history-resolved identity, writing `(v_0,v_1,v_2)=(x,y,z)` and
`(c_0,c_1,c_2)=(b,c,e)`:

\[
 F(r)=\sum_k c_k F(v_k)
 -\frac12\sum_{k<l}c_kc_l
     (v_k-v_l)^TL_W(v_k-v_l).
 \tag{4}
\]

Thus `F(y)<=F(x)` and `F(z)<=F(x)` are sufficient for passive ideal
mixing. They are not necessary. A common hard clipping interval is
nonexpansive on every pair of real scalar values, so `F(clip(r))<=F(r)`;
the same sufficient condition survives ideal hard clipping. It does not
remove the binary64 evaluation residual. Soft clipping and arbitrary
represented outputs must use (3) or their explicitly justified bounds.

There is no passive guarantee from current form alone. On the weighted path
0--1--2 with conductances 1 and 2 and capacities one, take x=0,
y=z=(1,0,-1) and a=1/2, with clipping inactive. All three histories have
zero mean in the conserved metric rho=(1,2,1), but

\[
 x^+=\tfrac34(1,0,-1),\qquad \Delta H=F(x^+)=\frac{27}{32}>0.
 \tag{5}
\]

The phase may be any admitted acute phase configuration and is unchanged.
These supplied histories are a counterexample to unconditional passivity,
not a claim that this particular triple is sampled from one sine trajectory.
Convex mixing can import stored historical disagreement into a current
consensus. The augmented-history Lyapunov theorem is not contradicted:
it consumes a different energy and an explicitly supplied history shift.

## 4. Both continuous rows and their means after the jump

Because (1) obtains pressure from phase and bare support,

\[
 p^+=p,\qquad \dot x^+=\dot x,\qquad
 q^+=q+L_Wh,\qquad \dot\theta^+-\dot\theta=ML_Wh.
 \tag{6}
\]

These are refreshed-law comparisons. An unrelated stored-pressure alias
or opaque refresh cannot replace (1). An unchanged instantaneous form rate
does not imply unchanged subsequent motion: its derivative changes by
`M DS(theta) M L_W h` when both continuations use the same held law.
For positive capacities and a connected positive-conductance graph, the
phase-rate change vanishes exactly when h is spatially uniform. With zero
conductances or capacities use the actual kernel of `M L_W` instead.

At positive capacities let `rho_i=d_i/nu_i` and define means using the
retained phase lifts. At the jump

\[
 \overline\theta^+=\overline\theta,\qquad
 \overline x^+-\overline x=\frac{\sum_i\rho_i h_i}{\sum_i\rho_i}.
 \tag{7}
\]

For ideal (2), the form mean is `b*mean(x)+c*mean(y)+e*mean(z)`.
If y,z really come from the same event-free fixed-support, held-capacity
conservative flow, its conserved weighted form mean makes this equal to
the current mean. Earlier jumps, changed metrics or arbitrary history do
not supply that premise. Rounding/clipping require their actual residual.
Equal historical means alone do not imply passive work, as (5) shows.

The phase mean is unchanged immediately and remains the appropriate
conserved lifted mean during each subsequently admitted continuous segment.
Neither mean is a complete collective state. In particular, a nonuniform
reset can preserve both means while changing the internal phase row.

## 5. A sufficient collective domain and a symmetry obstruction

Use the marked twenty-node contact and its nine physical twin pairs from
the contextual chart. Keep its support/conductance/capacity involution and
marked acute phase sector. Require current x and both selected history
rows to have equal forms within each twin pair. Uniform factor/delays and
one scalar clipping map then produce equal outputs for each pair, even
under the same deterministic represented arithmetic. Phase and capacity
equalities, marked support and cycle windings are unchanged by the jump.

For a positive-width subdomain, take every selected form in `(1/4,3/8)`,
hard bounds `[-1,1]`, any a in `(0,1]`, and any admitted acute phase in the
existing twin chart. Ideal outputs stay in that interval by convexity.
This proves immediate preservation of that organizational class for a
whole history family, not one replayed source. At an admitted actual
endpoint, smooth uniqueness and phase-sector openness also supply a
possibly endpoint-dependent nonzero continuation interval in the class.
No uniform lifetime, passive work or formation result follows. A separate
barrier or enclosure is needed for a claimed longer interval.

Current twin symmetry alone is insufficient. In the unclipped exact model,
for paired i,j with x_i=x_j the output difference is

\[
 x_i^+-x_j^+=c(y_i-y_j)+e(z_i-z_j).
 \tag{8}
\]

Vanishing of this combination is the exact single-event condition. Equal
lag rows are a reusable sufficient condition; cancellation can admit a
larger family. At a=1 only the global difference contributes to the ideal
output, although both rows remain runtime inputs. Clipping can erase a
difference but does not justify forgetting the original history/context.

The eleven-cell physical chart therefore extends by associated histories.
One may compress pairwise-equal history coordinates and lift them back to
the same fine members; the actual history order, multiplicities and all
nonphysical member records remain. On that stratum the uniform temporal
map commutes with this lift. For arbitrary linear observations, (2) also
commutes with the observation in exact arithmetic, but clipping and
nonlinear sine evolution do not inherit arbitrary linear closure from that
fact. The exact P5 diffusion/echo composition in
[Derived EPI memory](../DERIVED_EPI_MEMORY.md#95-exact-composition-with-a-declared-temporal-echo)
has a different generator and quotient; its coefficients and stability
verdicts cannot be transferred to this complete sine model.

## 6. Verification and resulting boundary

[Synthetic controls](../../tests/operators/test_collective_recursivity.py)
exercise the existing advisory, word admission, delayed planner/executor,
history support, storage, means, refreshed rows, clipping and the contact's
history-dependent symmetry. Their supplied histories are labeled controls,
not new continuous response evidence or retained-producer replays.

The result admits a canonical temporal form operation on a declared family
of collective objects with sufficient associated history. Its precise
limits include missing history, unsupported history association, active
clipping, and event work that can be positive. Runtime validity, grammar,
organizational identity, passivity and selected occurrence remain distinct.
Completing the event contract does not require every object to admit every
operator or every grammatical word to preserve the same identity.

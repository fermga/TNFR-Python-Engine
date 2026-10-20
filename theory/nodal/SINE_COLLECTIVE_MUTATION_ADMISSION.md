# Mutation compatibility: continuous rates, observed history and grammar

Compatibility of ZHIR with the selected complete law is a question about
admissible states, temporal observations and execution context. It is not
a requirement to execute every operator in one prescribed sequence. The
EN/IL/SHA/NAV successor supplies a useful conditional control, while the
law's larger state domain has different trigger possibilities.

This owner establishes a sharp general rate-admission condition and then
separates three restrictions on that control: unavailable temporal evidence,
recent-grammar rejection and a conditional continuous-rate obstruction.
No trajectory, manufactured source history or Mutation event is evaluated.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns subsequent investigation.

## 1. Complete law and the independent Mutation requirements

Use the [selected weighted-form, bare-phase sine law](SINE_COLLECTIVE_RECEPTION.md#1-complete-model-and-event-boundary)
on fixed finite simple connected undirected support, with at least two nodes,
positive edge conductances, signed real form x, primitive circular phase and
held positive capacities:

\[
 q=L_Wx,\qquad S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
 M=\operatorname{diag}\frac{\nu_i}{\pi d_i},\qquad
 p_i=\frac{S_i}{\pi d_i},\qquad
 \dot x=MS,\quad\dot\theta=Mq.
 \tag{1}
\]

Actual conductances enter q; distinct bare neighbors define d and S. No
forcing or continuous loss is present. The clock is the declared structural
time. The general state domain below is not restricted to the old marked
winding-one sector or to one event preparation.

The [registered Mutation contract](../STRUCTURAL_OPERATORS.md#91-mutation-zhir)
requires finite active capacity, an observed signed two-sample EPI secant
strictly above its configured nonnegative threshold xi, and independent
grammar context. The shared
[runtime gate](../../src/tnfr/operators/_mutation_gate.py) and
[pure certificate](../../src/tnfr/physics/mutation_trigger.py) separate the
current nodal prediction `nu_f*DeltaNFR` from that observed secant. Neither
an instantaneous prediction nor an acceleration diagnostic replaces the
two-sample evidence. Prior IL and a recent destabilizer belong to U4b;
capacity and the threshold comparison do not establish them.

The default represented `ZHIR_THRESHOLD_XI=0.1` is an operational calibration,
not a constant derived from (1). The following mathematical comparisons
take its value and time basis as supplied. A different calibrated threshold
can have a legitimate scope, but choosing one solely to force an execution
would not establish a new property of this preparation.

## 2. Sharp general continuous-rate compatibility

Since each sine term lies in [-1,1], every state of (1) satisfies

\[
 |\dot x_i|\leq\frac{\nu_i}{\pi}.
 \tag{2}
\]

For held capacity and an interval `[t_0,t_1]` of the same unforced law,
with no intervening form jumps, its true signed secant satisfies

\[
 r_i=\frac{x_i(t_1)-x_i(t_0)}{t_1-t_0}
 =\frac1{t_1-t_0}\int_{t_0}^{t_1}\dot x_i(t)\,dt,
 \qquad |r_i|\leq\frac{\nu_i}{\pi}.
 \tag{3}
\]

Thus `nu_i<=pi*xi` excludes the strict positive trigger for every such
interval, without any acute-sector or winding premise. Equality is still
excluded because ZHIR requires strict growth. This statement concerns
true continuous-time evidence under the declared law, not every numeric
history accepted by a record reader.

The capacity condition is sharp over the unrestricted phase domain. If
`nu_i>pi*xi`, choose an angle `0<phi<pi/2` with
`sin(phi)>pi*xi/nu_i`. Set the target phase to zero and every other phase
to phi, so in particular each of its d_i neighbors contributes sin(phi).
Then

\[
 \dot x_i(0)=\frac{\nu_i}{\pi}\sin\phi>\xi.
 \tag{4}
\]

All edge gaps are zero or have magnitude phi, so this is a regular acute
state, with zero cycle windings. It belongs to the complete law's state
domain but not the preceding winding-one contact class. Finite signed
form may be chosen independently, for example uniform form. The smooth
finite-dimensional vector field has a local solution, and continuity
preserves the strict rate margin on some nonzero time interval. Integrating
there proves the existence of genuine short positive secants above xi.

Consequently, for a fixed nonisolated node with held positive capacity and xi>=0,
**there exists an admissible preparation and a short true continuous
secant above xi if and only if `nu_i>pi*xi`**. This is an existence theorem
for temporal compatibility, not observed history from the retained source,
grammar admission, a selected sampling interval or autonomous occurrence.
It supplies no instruction to insert the constructed values into an
execution record.

Positive local growth is consistent with the conserved weighted form mean:
with `rho_i=d_i/nu_i` at positive capacities, reciprocal support gives
`sum_i rho_i*xdot_i=sum_i S_i/pi=0`. Other nodes provide the compensating
contribution. This balance does not impose zero growth at each node, and
it does not prove that the constructed phase state is reachable from a
particular preparation at its storage and mean values.

## 3. What the actual NAV successor does and does not supply

Keep the [contextual Transition successor](SINE_COLLECTIVE_TRANSITION.md):
the fixed twenty-node, 45-edge contact, its marked ports 0 and 1, nine
physical twin pairs, actual weights, retained associated histories, and
the selected latent NAV reset. Port capacities satisfy `nu_i<3/8`.
The general bound (2) only gives `|xdot_i|<3/(8*pi)`, approximately 0.1194.
It does not exclude the default threshold for the entire capacity family.
The lower-capacity part satisfying `nu_i<=pi*xi` is globally excluded by
(3); no additional claim is inferred for the rest from that bound alone.

The earlier [sector-specific bound](SINE_COLLECTIVE_TRANSITION.md#7-a-conditional-continuous-rate-budget-at-the-ports)
has stronger hypotheses and conclusion. On the retained marked acute
sector with constituent winding +1, every original forward C5 gap is
positive. Its four original port neighbors contribute two positive and
two negative sine terms, with total magnitude less than two; the other
port and two cross neighbors contribute at most three more. Hence

\[
 |S_i|<5,\qquad
 |\dot x_i|<B:=\frac{15}{56\pi}<\frac1{10}<\xi
 \quad(i=0,1),
 \tag{5}
\]

where the final strict inequality uses the represented default 0.1.
A true secant over an interval wholly in that sector, with the same held
capacities, no form jumps and the same clock, obeys the same bound.
Two acute endpoints do not prove intermediate-sector membership. The
full earlier EN source contains a NAV sector-departure counterexample;
it cannot inherit (5) merely because the word was grammatical or U6 passed.
Conversely, leaving that sector does not prove that a threshold crossing
will occur. The conditional obstruction and the general existence result
in section 2 have different source premises.

At the actual immediate successor, two additional restrictions precede
any proposed flow experiment:

- The selected atomic EN/IL/SHA/NAV path supplies no two-sample continuous
  EPI history. Its form-changing jumps, stored pressure, glyph history
  and latency timestamps do not construct such observations. Missing
  evidence is unavailable, not a measured zero or a failed crossing.
- The last three actual glyphs are `IL,SHA,NAV`, containing no registered
  destabilizer. Prior coherence and zero persistent debt do not replace
  this missing U4b condition. NAV is a generator/closure, not a destabilizer.
  The shared grammar therefore rejects ZHIR on that recent context.

These conclusions are independent. Supplying an otherwise valid crossing
record to a separate synthetic control would not repair the actual recent
grammar. Ordinary continuous flow or sampling does not append a destabilizing
glyph, so waiting alone does not change this U4b result. Inserting an event
solely to make the history pass would be a changed preparation and word,
requiring its own scientific purpose and admission.

## 4. Sufficient temporal and execution state

The [contextual collective chart](SINE_COLLECTIVE_IDENTITY_TRANSITION.md#2-a-contextual-chart-for-the-exact-twin-stratum)
must retain the data actually consumed by this operator. For each marked
target, extend the physical state and associated execution context with:

- the selected history channel, last two signed scalar EPI samples, their
  declared clock/units and timestamps when present, plus the live EPI
  endpoint and its sample association;
- finite active capacity, any explicitly configured minimum, the threshold
  and its units, and the actual stored pressure used for prediction and
  the phase-reset direction, including its represented signed-zero branch;
- the ordered recent glyphs, persistent prior-coherence and debt records,
  complete-word context, target order and all consumed runtime configuration;
- any third sample required by the separately requested acceleration
  diagnostic and the provenance of all intervening events and refreshes.

Per-port histories cannot be replaced by a mean-form history of the whole
organization: an average rate need not equal either target's rate. Physical
twin equality also does not identify independently associated lifecycle or
temporal records. A collective transition is a partial map on this richer
context, not an operation determined by winding and instantaneous fields.

The current timestamped reader uses the last two replayable `(time, epi)`
records. It requires finite signed real samples, a finite positive interval
and agreement of the final represented EPI with the current EPI. Runtime
uses zero endpoint tolerance; the pure certificate can declare a finite
nonnegative absolute tolerance, without weakening the strict rate threshold.
Current uniform-real BEPI is admitted through the shared signed-scalar
boundary; a magnitude projection cannot supply signed growth. Nonfinite,
Boolean and nonzero values lost on materialization are not valid primitive
evidence, and output arithmetic must remain representable.

When `epi_time_history` is supplied, it is authoritative: invalid or stale
endpoint evidence does not fall back to an untimestamped alternative.
Without it, a nonempty `epi_history` takes precedence over `_epi_history`;
the existing empty-canonical compatibility case can use the latter. These
legacy channels declare one operator step between samples and do not check
a current endpoint. They are explicitly not continuous-time evidence.

The field `physical_time_resolved` records that an interval was supplied
by timestamps. It is not laboratory calibration or provenance authentication.
The pure certificate and runtime adapter do not consume an independently
authoritative live clock: equal final EPI values alone cannot establish
that the last timestamp is the present instant. In this research model,
the observation protocol must additionally bind the endpoint to the declared
live structural time, target identity, source, law and intervening events.
The graph's `_t` cannot be compared blindly when another execution path
uses it for operator positions. A record reader cannot establish these
missing associations by inspecting two scalar values.

The current product `nu_f*p` is an endpoint prediction; the secant is an
interval average. Their difference is not automatically a solver residual
or a law violation, even with valid records. A bound on rate variation and
the complete intervening model would be needed for that interpretation.
The optional three-sample acceleration diagnostic is separate from both
the strict two-sample trigger and the independent U4b check.

## 5. Clock changes cannot manufacture a genuine crossing

For an affine change `s=c*t+b` with c>0, transform both rows, not only the
history timestamps:

\[
 \frac{dx}{ds}=\frac1c MS,\qquad
 \frac{d\theta}{ds}=\frac1c Mq.
 \tag{6}
\]

One consistent parameterization retains p and uses `nu_s=nu_t/c`, hence
`M_s=M_t/c`. Histories transform to `s_k=c*t_k+b` and the rate-calibrated
threshold transforms to `xi_s=xi_t/c`. Then

\[
 r_s=\frac{r_t}{c},\qquad
 r_s>\xi_s\ \Longleftrightarrow\ r_t>\xi_t,
 \qquad \nu_s>\pi\xi_s\ \Longleftrightarrow\nu_t>\pi\xi_t.
 \tag{7}
\]

Capacity floors and every other consumed dimensional setting need their
own corresponding transformation. This is not an automatic equivalence
of all configured event policies. Keeping xi numerically fixed while
compressing timestamps changes the trigger policy; changing only the
nodal form row changes the complete model.

For a nonlinear monotone clock, a finite secant transforms by the interval
ratio `Delta t/Delta s`, not just the inverse clock derivative at the final
sample. No endpoint-only rescaling proves finite-window equivalence.
Untimestamped operator-step histories need a separately declared timing
model before either clock comparison applies. None of these structural
clock choices supplies a laboratory-unit bridge.

## 6. Represented measurements and model-domain checks

Suppose two recorded forms differ from their true same-law values by
errors of magnitude at most epsilon_0 and epsilon_1, at exact declared
times separated by h>0. Before accounting for arithmetic, their secant
error obeys

\[
 |\widehat r-r|\leq\frac{\epsilon_0+\epsilon_1}{h}.
 \tag{8}
\]

Solver, scalar-conversion and division errors require their additional
budget. Timestamp uncertainty requires a positive interval enclosure and
its own quotient bound. The analytical ceiling B in (5) implies a
represented non-crossing only when B plus all applicable rate errors
does not exceed xi. A very short interval can amplify fixed endpoint
errors; increasing sampling precision does not establish missing grammar
or source provenance.

Conversely, a record-level crossing alone does not refute (3) or (5).
Its history may come from a different clock, an intervening jump, another
capacity/law, an out-of-sector interval or unresolved numerical error.
These possibilities must be distinguished rather than classifying every
rejection as a dynamical obstruction. Invalid primitive evidence must
reject; unavailable evidence must remain unavailable, not acquire an
invented zero-rate estimate.

## 7. Scope of the resulting admission decision

The complete sine law is compatible with strict continuous growth on
states satisfying the sharp condition in section 2. The actual NAV
successor is not ready for ZHIR: its temporal evidence is unavailable and
its recent grammar independently rejects the transformer. Its retained
acute winding-one continuations also have the conditional obstruction
(5). None of these results excludes Mutation in every TNFR model or
requires this one organization to realize every registered operator.

Reopening an actual collective event comparison requires independently
justified source/capacity/clock and grammar context, real endpoint-bound
temporal evidence with an adequate error budget, and admission of the
actual reset and post-event observation. A supplied event selector would
remain separate from eligibility. The general integration question is
which states and laws admit each operator's contract; it does not impose
EN/IL/SHA/NAV as a compulsory causal chain for all operator studies.

Synthetic reader and runtime controls exercise missing/stale evidence,
signed strict comparisons, representation boundaries, independent grammar
rejection and declared clock behavior. They are not observed research
histories or a new response campaign. The shared owners above retain
execution; this analysis introduces no alternative trigger reader.

The [collective Mutation admission controls](../../tests/operators/test_collective_mutation_admission.py)
exercise these reader and grammar boundaries on independent inputs. The
[verification guide](../../TESTING.md) owns the maintained test selection.

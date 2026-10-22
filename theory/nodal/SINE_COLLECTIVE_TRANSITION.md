# Collective Transition with retained lifecycle context

On the successor of the admitted EN/IL/SHA word, NAV selects its latent
regime because SHA sets a lifecycle flag, even though capacity exceeds
the numerical latent cutoff. Its phase and capacity reset survives the
declared sine-pressure refresh; its raw-pressure jitter does not survive
as an independent continuous coordinate. The same contextual chart admits
this conditional event. Preservation of the old acute phase sector needs
an additional margin: a counterexample in the full Reception source box
fails it, while an explicit smaller source domain passes.

This owner closes a bounded analytical question, without a new retained-source
word, trajectory or physical observation. The
[research program](../RESEARCH_PROGRAM.md#research-current-question)
owns subsequent work. Regime labels, event timing and jitter configuration
remain supplied policies, not autonomous dynamical causes.

## 1. The actual successor and the context NAV consumes

Keep the [Reception model](SINE_COLLECTIVE_RECEPTION.md#1-complete-model-and-event-boundary):
twenty associated nodes on the fixed 45-edge weighted contact, with two
singleton ports 0 and 1 and nine exact physical twin pairs. Its conservative
continuous rows and storage are

\[
 q=L_Wx,\quad S_i(\theta)=\sum_{j\sim i}\sin(\theta_j-\theta_i),\quad
 M=\operatorname{diag}\frac{\nu_i}{\pi d_i},\quad
 p_i=\frac{S_i}{\pi d_i},\qquad
 \dot x=MS,\quad\dot\theta=Mq,
 \tag{1}
\]

\[
 H=\tfrac12x^TL_Wx+V(\theta),\qquad
 V(\theta)=\sum_{\{i,j\}\in U}(1-\cos(\theta_j-\theta_i)).
 \tag{2}
\]

The old conductances are one, the five added conductances lie in `(3/4,1]`,
and bare degrees retain all fine neighbors. There is no forcing or loss.
Capacities are held between events; time is the same structural clock.
Geometric lengths equal conductances only for the finite U6 comparison.

The [prior source domain and word](SINE_COLLECTIVE_RECEPTION.md#5-one-grammatical-word-on-a-positive-width-domain)
give port form in `(1/4,3/8)` and post-SHA capacity in `(3/16,5/16)`.
SHA preserves form and phase and writes `latent=True`, a supplied
`latency_start_time`, the current `preserved_epi`, zero `silence_duration`
and its initial-form classification. The complete word leaves persistent
U2 debt zero and the associated history ending in SHA. Retain the freshly
evaluated pressure after that word. No intervening flow or form event is
inserted before the NAV event considered here.

The [Transition owner](../../src/tnfr/operators/transition.py) selects the
latent regime whenever the flag is true or capacity is below 0.05. Thus
the actual successor uses the latent branch, although its capacity is
strictly greater than 3/16. Deleting the flag from an otherwise identical
physical state selects the active branch: its default phase shift and
capacity multiplier differ. The resonant branch is excluded here because
form remains below 3/8 and capacity below 5/16. Other branches are not
inferred to be reachable from this successor.

Consequently the predictive chart must retain lifecycle fields in addition
to form, phase, capacity and weighted support. It also retains target
association, ordered histories, debt, all consumed configuration, a
timezone-aware stage timestamp and any consumed per-node random stream
identity and progress. Phase/capacity equivalence alone does not select
NAV's map. The lossless eleven-cell physical chart still lifts these
records to the twenty associated members before executing the shared
atomic stage. It does not replace them with cell averages.

The supplied NAV instant must not precede the SHA timestamp. Their elapsed
wall-clock duration is latency telemetry, not the structural flow clock
or a finite interval spent evolving (1). With no intervening form change,
the preserved-form drift is zero. An explicit maximum silence duration,
when configured, still has its own admission and warning behavior. NAV
clears its consumed latency flag/start/preserved-form fields on successful
commit and records the duration and regime origin; it does not erase every
historical lifecycle record.

## 2. Selected primary reset and raw pressure

Let T be the two ports and let `chi_T` be its indicator. The ideal primary
map for the latent preset is

\[
 x^+=x,\qquad \nu^+=A\nu,\qquad
 \theta^+=\theta+\delta,\qquad
 \delta=\sigma\chi_T,
 \tag{3}
\]

where A has entry a at each port and one elsewhere. Use the represented
preset parameters as exact inputs to this ideal-real map:

\[
 a=\operatorname{fl}(1.2),\quad
 \sigma=\operatorname{fl}(0.1),\quad
 \kappa=\operatorname{fl}(0.7),
\]

\[
 \frac{119}{100}<a<\frac65,\qquad
 \frac1{10}<\sigma<\frac{101}{1000},\qquad
 \frac{69}{100}<\kappa<\frac7{10}.
 \tag{4}
\]

The implemented phase write normalizes modulo one turn; equation (3) uses
the corresponding continuous lift. Actual multiplication and modulo
arithmetic have separately retained endpoint defects. A requested phase
increment does not prove that arbitrary large represented phase origins
realize it exactly.

Select the admitted non-strict handler with `NAV_eta=1/2`. If p denotes the
pre-event stored pressure, the target handler and final raw pressure are

\[
 b_i=\tfrac12p_i+\tfrac12 s_i\nu_i,\qquad
 p_{\mathrm{handler},i}^+=b_i+\zeta_i,\qquad
 p_{\mathrm{raw},i}^+=\kappa(b_i+\zeta_i),
 \tag{5}
\]

where `s_i=+1` for `p_i>=0` and `s_i=-1` otherwise. The selected jitter
amplitude satisfies `0<=j<=1/24`, including the configured default
`1/(8*pi)`. In random mode, the supplied per-node draw has `|zeta_i|<=j`.
In deterministic mode, the handler uses `zeta_i=j*sign_+(b_i)`; setting
`NAV_RANDOM=False` does not by itself remove jitter. Setting j to zero
is a separate configuration. Both flags use their actual Boolean admission.

The low-level direct glyph has its own pressure-only contract. Equation
(3), regime retention, lifecycle updates and the following chart concern
the public/atomic staged path, not that direct glyph. The strict handler
uses a different base and is outside the bounded preset in (5).

The full previous source domain already supplies a margin for a nontrivial
handler. The prior IL phase bound gives port error `E=343/2000` and
untargeted error less than 1/8 about the reference. The port has six external
neighbors and the other port; since the reference sine sum vanishes,

\[
 |p_i|<\frac{6(E+1/8)+2E}{7\pi}
 =\frac{1061}{3500\pi}<\frac{1061}{10500}.
 \tag{6}
\]

Together with `nu_i>3/16`, this bounds every admitted jitter draw away from
the handler's no-change rejection:

\[
 |b_i-p_i|-j>
 \tfrac12\left(\frac3{16}-\frac{1061}{10500}\right)-\frac1{24}
 =\frac{131}{84000}>0.
 \tag{7}
\]

The default live checks `nu>=0.01`, `|p|<=1` and the latent form advisory
threshold 0.05 also have strict margins here. All-target atomic admission
still validates finite values, timestamps, flags, sinks, phase, requested
parameters and actual proposals before committing. Equation (7) is an
ideal-domain bound, not permission to bypass those represented checks.

## 3. Event work, changed means and refreshed rows

For an actual admitted phase/capacity endpoint with form and support held,
use a declared lift delta of its actual phase change. Its exact structural
work is

\[
 \Delta H=V(\theta+\delta)-V(\theta)
 =\sum_{\{i,j\}\in U}
   [\cos(\theta_j-\theta_i)
    -\cos(\theta_j-\theta_i+\delta_j-\delta_i)].
 \tag{8}
\]

The phase potential uses bare edges. Conductance weights do not multiply
these terms, and capacities add no separate term to (2). Equal shifts on
both ports leave their internal-edge phase difference fixed; only eight
old edges and four cross edges change. Adding two isolated-port work
calculations would incorrectly include their internal-edge contributions.
The sign of (8) is not determined by the regime name.

At the reference phase, EN and IL leave phase at that reference and SHA
also preserves it. The latent NAV reset there has

\[
 \Delta H=(8\cos(2\pi/5)+4)(1-\cos\sigma)>0.
 \tag{9}
\]

This is a source in the analytical successor class, with a supplied event
and positive work, not a new evaluation of the historical preparation.
The pressure attenuation in (5) does not make this phase event passive.

Put `rho_i=d_i/nu_i`, `rho_i^+=rho_i/a_i`, and `R^+=sum_i rho_i^+`.
The post-event normalized means, using the same lifts, satisfy

\[
 \mu_x^+-\mu_x=
 \frac{(\rho^+-\rho)^T(x-\mu_x\mathbf1)}{R^+},
 \tag{10}
\]

\[
 \mu_\theta^+-\mu_\theta=
 \frac{\rho^{+T}\delta+
       (\rho^+-\rho)^T(\theta-\mu_\theta\mathbf1)}{R^+}.
 \tag{11}
\]

Form itself is unchanged, but its capacity-weighted mean can change. The
continuing conservative flow preserves the new mean leaves. A circular
representative difference is not automatically the lift used in (11).

Constitutive refresh computes `p^+=S(theta+delta)/(pi*d)` independently of
(5). With `Delta S=S(theta+delta)-S(theta)`, the complete fresh rows give

\[
 \dot x^+-\dot x=(A-I)MS+AM\Delta S,\qquad
 \dot\theta^+-\dot\theta=(A-I)Mq.
 \tag{12}
\]

The phase jump changes fresh pressure and can change form rates even at
untargeted neighbors. Their phase rates are unchanged at this immediate
boundary because their capacity and q were unchanged. The two rows do
not generally share one global clock factor. Work and changed responses
must be evaluated on this actual reset rather than inferred from an
activation label.

Under immediate pressure-only refresh with no intervening flow, all
accepted draws in (5) have the same refreshed continuous coordinates
`(x,theta,nu,U,W)`: the raw pressure is overwritten, while the phase and
capacity reset remains. This is a conditional projection equivalence.
Histories, transition telemetry and random progress can still differ, so
the complete execution states are not identified. Consuming raw pressure
over a held interval, allowing another event to inspect it, or declaring
an independent pressure-memory law would be a different model.

## 4. The previous phase box does not guarantee acute preservation

An acute EN/IL/SHA successor is not enough to infer that the fixed positive
NAV phase shift preserves every incident acute gap. The following exact
counterexample lies strictly inside the earlier source box.

Write `alpha=2*pi/5` and `epsilon=3/25<1/8`. Give ports 0 and 1, forward
neighbors 2 and 3, and cross neighbors 10 and 11 phase error +epsilon;
give trailing neighbors 8 and 9 error -epsilon. Set the other errors to
zero and retain all physical twins, admitted forms, weights and capacities.
EN does not change phase. The target neighbor resultant before IL is

\[
 R=3e^{i\epsilon}+4\cos(\alpha+\epsilon),\qquad
 \mu=\arg R
 =\arctan\frac{3\sin\epsilon}
                 {3\cos\epsilon+4\cos(\alpha+\epsilon)}.
 \tag{13}
\]

The denominator is positive. Also `cos(alpha)<31/100`,
`sin(epsilon)>epsilon-epsilon^3/6`, and monotonicity of cosine on this
interval give `tan(mu)>2/25`. Therefore
`mu>arctan(2/25)>1/16`. After IL with any `0<=lambda<=3/10` and then
latent NAV, the trailing port-edge gap is

\[
 g^+=\alpha+(2-\lambda)\epsilon+\lambda\mu+\sigma
 >\alpha+\frac{1291}{4000}>\frac\pi2.
 \tag{14}
\]

The final inequality follows from `pi<22/7`. The counterexample is still
well below the principal cut pi: it proves loss of the acute sector, not
a change of either constituent winding or destruction of organization.
It also does not prove that a subsequent trajectory is unstable.

## 5. Explicit acute successor domain and whole-word grammar

For an actual pre-NAV lifted state, the direct admission criterion is

\[
 |g_{ij}+\delta_j-\delta_i|<\pi/2
 \quad\text{on every fine edge},
 \tag{15}
\]

with the original marked reference integers kept. Endpoints and their
straight phase interpolation then stay in the same convex acute sector.
This criterion retains actual phase geometry instead of replacing it by
a winding label.

A positive-width sufficient family is obtained by explicitly restricting
the earlier EN source to `max_i|eta_i|<1/16`. This is a new admission
premise for NAV, not a revision of the earlier EN domain or evidence.
All other form, capacity, weight, history and configuration premises stay
as previously declared. The reference target resultant exceeds four, and
its perturbation is less than 7/16, so its mean angle satisfies

\[
 |\mu-\tau|<\frac7{57},\qquad
 |\eta_{\mathrm{port,after\ IL}}|
 <\frac7{10}\frac1{16}+\frac3{10}\frac7{57}
 =\frac{49}{608}.
 \tag{16}
\]

After the latent shift, every port-to-old-neighbor edge has error below

\[
 \frac{49}{608}+\frac1{16}+\frac{101}{1000}
 =\frac{18551}{76000}<\frac\pi{10}.
 \tag{17}
\]

Its old reference magnitude is alpha, leaving the full gap strictly
below pi/2. Untargeted old edges have error below 1/8; the internal port
edge cancels the common shift and has error below 49/304. Cross-edge
reference gaps are zero and obey the bound in (17). Thus the ideal reset
preserves the marked acute sector and its two original windings on this
positive-width source stratum. It supplies no future lifetime bound.

The selected complete successor word is a singleton NAV on both ports,
using `ValidatedSequence` with the retained nonzero-form context, strict
rejection and live preconditions. NAV already has both generator and
closure roles and introduces no U2 debt; the successor debt remains zero.
The actual history ends in SHA, matching its retained predecessor context.
No additional IL or SHA is implied by this analytical reset. Applicable
stage validation and the finite U6 observations below remain distinct
from the geometric criterion (15).

## 6. Finite U6 and implementation controls

For the declared geometry, the prior structural-potential estimate is

\[
 \frac1{20}\sum_i |(B\Delta p)_i|
 \leq\frac{76}{45}\|\Delta p\|_1,
 \qquad B_{ij}=\operatorname{dist}(i,j)^{-2}\ (i\ne j).
 \tag{18}
\]

Only two stored pressures change in (5). Using (4), `|p_i|<=1/pi`,
`nu_i<5/16` and `j<=1/24`, their changes are bounded by

\[
 |p_{\mathrm{raw},i}^+-p_i|
 \leq\frac{131}{200}|p_i|+\frac7{20}\nu_i+\frac7{10}j
 <\frac{131}{200\pi}+\frac7{64}+\frac7{240}.
 \tag{19}
\]

For equal port shifts sigma, the internal-edge contribution cancels.
The sine Lipschitz estimate with the actual degrees gives

\[
 \|p_{\mathrm{fresh}}^+-p\|_1
 \leq\frac{\sigma}{\pi}
   \left(\frac{12}{7}+2+\frac23\right)
 <\frac{2323}{5250\pi}.
 \tag{20}
\]

Combining (18)--(20) and `pi>3` bounds the largest selected before/raw,
before/refreshed or raw/refreshed comparison by

\[
 \frac{76}{45}\left[
  2\left(\frac{131}{600}+\frac7{64}+\frac7{240}\right)
  +\frac{2323}{15750}\right]
 =\frac{4123627}{2835000}<\frac32<\frac\pi2.
 \tag{21}
\]

This holds on the full prior EN successor, including the nonacute
counterexample in section 4, for accepted draws under the selected
non-strict handler. Thus passing this finite U6 policy does not prove
acute-sector preservation. Other strict-mode pressure bases, jitter
budgets, geometric lengths or baselines need their own assessment.

## 7. A conditional continuous rate budget at the ports

In the retained marked acute sector, every edge of an oriented original
C5 cycle has a strictly positive principal gap. Indeed, its five gaps sum
to 2*pi by winding +1; if one were nonpositive, the other four strict
acute bounds would make that sum less than 2*pi. Every original fine
edge belongs to such a constituent cycle, including either marked port.

At a port, the four original neighbors consequently supply two positive
and two negative sine terms, whose total has magnitude less than two.
The other port and two cross neighbors add at most three in magnitude.
Together with post-NAV capacity below 3/8 and degree seven, this gives

\[
 |S_i|<5,\qquad |\dot x_i|<\frac{15}{56\pi}<\frac1{10}
 \quad(i=0,1).
 \tag{22}
\]

A true continuous-time EPI secant over a nonzero interval wholly governed
by (1), with these held capacities and the same retained phase sector,
obeys the same bound by integration. Two acute endpoints alone do not
establish the intermediate-sector premise. An interval containing an
operator form jump, a different clock or a capacity reset also has
different assumptions.

This is a conditional budget relevant to a later temporal-trigger
admission: it cannot exceed the configured default mutation threshold
0.1 on that clock and domain. It is not measured trigger evidence, a
statement about operator-step histories, or an exclusion of every ZHIR
configuration or source. The execution plan alone selects any later
investigation.

## 8. Representation and verification boundaries

Ideal input-parameter bounds, represented endpoint checks and finite
software observations retain separate scopes. Synthetic controls reuse
the existing EN/IL/SHA preparation and NAV atomic proposal, including
live grammar, latency, raw/refresh boundaries and random progress. They
do not replay a frozen preparation, install a new pressure law or
establish autonomous occurrence or physical identification.

The [collective Transition controls](../../tests/operators/test_collective_transition.py)
exercise the existing public stage on independent preparations, including
the latent/active context discriminator, actual phase work and fresh rows,
changed mean leaves, accepted random progress, invalid-lifecycle rollback
and the nonacute counterexample despite passing grammar and finite U6.
The [verification guide](../../TESTING.md) owns the maintained selection.

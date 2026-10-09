# Clock drift, sampled exposure and robust mean-rate inference

<a id="sine-clock-drift-inference"></a>

## Question and conditional result

The [four-reading result](SINE_FINITE_CURVATURE_INFERENCE.md#sine-curvature-inference-result)
separates necessary gain and clock intervals under a held clock rate. It
does not test whether an unobserved clock is constant between readings.
Here the complete structural law and held sensor are unchanged, while the
clock rate belongs to a specified positive class with bounded derivative.
The target is the **first-window mean rate**, not an instantaneous rate or
the clock's complete time profile.

Two distinct results apply. Equal integrated clock exposures give exactly
equal sampled full states, including the declared phase events. This
obstructs profile and instantaneous-rate identification from these samples.
Separately, a finite comparison bound transfers uncertain-clock readings
to a constant-mean reference and permits reuse of the existing necessary
inverse. The added intervals represent clock-model discrepancy; the
physical sensor-error allowance is unchanged.

The theoretical results and implementation controls require no reserved
response. The separately specified evaluation below tests their finite
software application; earlier frozen protocols, sources and results remain
unchanged. The [ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization)
continues to separate collective observation from physical identification.

<a id="sine-clock-drift-source-and-law"></a>
## F1-F2: one structural law and a declared clock class

Retain the [complete source and law](SINE_CLOCK_INFERENCE.md#sine-clock-source-and-law):
the eighteen-node two-port C9 support, degree matrix \(M\),
\(A=M^{-1}L\), \(q=e_4-e_5\), held unit capacities and
\[
\frac{dx}{d\tau}=-Ax+\gamma f(\theta),\qquad
\frac{d\theta}{d\tau}=\gamma Ax,
\qquad \gamma=\frac1{1023\pi}.
\tag{1}
\]
The full original source has arbitrary common means and centered residuals
\(u,v\) satisfying \(\|u\|_M\le X\), \(\|v\|_M\le Y\).
The affine phase geometry, its original acute guard, the nominal priors
\(b\in[11/8,3/2]\), \(c\in[2/3,1]\), and the actual original
statistic \(B_{\rm initial}=b-(v_1-v_0)/8\) retain their existing owners.
All thirty-six coordinates evolve; no source recentering or reset is added.

On the observation interval \([0,2H]\), declare
\[
\rho\in C^1([0,2H]),\qquad
0<\rho_-\le\rho(s)\le\rho_+,\qquad
|\rho'(s)|\le\Lambda,
\tag{2}
\]
with \(H>0\), \(\Lambda\ge0\), and \(h_*:=\rho_+H\le1/2\).
The independently supplied bound \(\Lambda\) has units of structural
time per squared observation time. Define
\[
\tau(s)=\int_0^s\rho(v)\,dv,\qquad
\bar\rho_1:=\frac{\tau(H)}H\in[\rho_-,\rho_+].
\tag{3}
\]
Both rows of (1) multiply by \(\rho(s)\) in observation time. This
follows the [whole-law clock contract](../NODAL_PARAMETER_FOUNDATIONS.md#31-capacity-clock-and-positivity-require-compatible-laws).
The model class is a premise, not a profile estimated from the evaluated
response. No clock samples or cached admission report substitute for it.

For \(0<a_1\le a_2\le1\), the phase jumps are \(a_1q\) at
\(s=0\) and \((a_2-a_1)q\) at \(s=H\), with form continuous.
There is no event at \(H/2\). At the same four times
\[
(s_0,s_1,s_2,s_3)=(0,H/2,H,2H),\qquad
r_i=Gq^Tx(s_i)+O+\eta_i,\quad |\eta_i|\le\delta,
\tag{4}
\]
use one held positive gain \(G\in[G_-,G_+]\) and one held offset
\(O\). The readings are instantaneous passive samples with primitive
intervals \([\ell_i,u_i]\). Averaging over finite sensor apertures would
be a different observation law. Input calibration, support, capacity and
the structural coefficients are all held; no other nuisance law is relaxed.

<a id="sine-clock-drift-exposure-equivalence"></a>
## F3-F4: exact dependence on integrated exposures

Let \(\Phi_t\) denote the autonomous full flow of (1), and let
\(\mathcal J_a\) add \(aq\) to phase without changing form. Define
the three structural exposures
\[
e_1=\tau(H/2),\qquad
e_2=\tau(H)-\tau(H/2),\qquad
e_3=\tau(2H)-\tau(H).
\tag{5}
\]
Since \(\rho>0\), the chain rule and uniqueness give
\[
\begin{aligned}
z_{1/2}&=\Phi_{e_1}\mathcal J_{a_1}z^-,\\
z_{H^-}&=\Phi_{e_2}z_{1/2},\qquad
z_{H^+}=\mathcal J_{a_2-a_1}z_{H^-},\\
z_{2H}&=\Phi_{e_3}z_{H^+},\qquad z=(x,\theta).
\end{aligned}\tag{6}
\]
Thus the complete sampled history factors through the three exposures.
Two clocks with the same \((e_1,e_2,e_3)\), source and observed-time
event schedule give exactly the same sampled full states. With the same
held sensor and allowed error realization, they give the same readings.
This is a sufficient equality condition; distinct exposures may also be
indistinguishable through a partial observation or a special trajectory.
It does not prove that the three exposures themselves are identified.

An explicit smooth obstruction works for every \(\Lambda>0\) and
nondegenerate rate prior. Choose an interior \(\rho_0\), and
\[
\rho_\epsilon(s)=\rho_0+\epsilon\cos(4\pi s/H),\qquad
0<\epsilon<\min\{\rho_0-\rho_-,\rho_+-\rho_0,
                         \Lambda H/(4\pi)\}.
\tag{7}
\]
This clock satisfies (2). Each interval in (5) contains an integer number
of its cosine periods, so its exposures equal those of the constant clock
\(\rho_0\). Nevertheless \(\rho_\epsilon(s_i)=\rho_0+\epsilon\)
at every reading time, whereas the constant clock's rate is \(\rho_0\).
Even noiseless full-state samples at those times cannot distinguish their
instantaneous rates or verify clock constancy. Their first-window mean is
the same, so this obstruction does not preclude bounds on \(\bar\rho_1\).

The equivalence relies on an autonomous structural law, events at the fixed
observation times and instantaneous samples. A different event selector,
structural-time forcing or observation aperture requires its own proof.
Between the sampled times the full observed-time histories generally differ.

<a id="sine-clock-drift-finite-transfer"></a>
## A finite comparison with an exactly matching event state

Use a reference history with the same full source, law, inputs, gain,
offset and errors, but constant rate \(\bar\rho_1\). It agrees with
the actual history at \(0\) and \(H\), because both have elapsed
structural time \(\tau(H)=H\bar\rho_1\) before the second event.
Their complete post-event states therefore agree as well. No stability
approximation or inferred intermediate state is needed at that jump.

Set \(\Delta\rho=\rho_+-\rho_-\). The two possibly nonzero
exposure mismatches satisfy
\[
\begin{aligned}
|\tau(H/2)-\tau(H)/2|&\le
 d_{1/2}:=\min\{\Lambda H^2/8,\Delta\rho H/4\},\\
|\tau(2H)-2\tau(H)|&\le
 d_2:=\min\{\Lambda H^2,\Delta\rho H\}.
\end{aligned}\tag{8}
\]
For the first bound, write the difference as
\(\frac12\int_0^{H/2}[\rho(v)-\rho(v+H/2)]\,dv\).
For the second, use
\(\int_0^H[\rho(v+H)-\rho(v)]\,dv\).
The derivative bound and global rate range give the two respective bounds
inside each minimum. Linear clocks \(\rho(s)=\rho_0+\kappa s\)
have exact signed differences \(-\kappa H^2/8\) and \(\kappa H^2\),
so the derivative-based coefficients cannot be uniformly reduced when
such clocks are admitted by the rate range.

The following global bound avoids duplicating or reevaluating the inverse's
specialized source remainder. Let \(g=1/3069>\gamma\). On this
undirected support, \(A\) is positive semidefinite and self-adjoint in
the \(M\) inner product, so \(e^{-tA}\) is a contraction. Pairwise
sine currents have zero degree-weighted mean and
\(\|f(\theta)\|_M\le\sqrt{40}<7\) for every phase state.
Variation of constants, unchanged form at the phase events, and
\(\tau(2H)\le2h_*\) consequently give
\[
\|P_Mx\|_M\le Q_0:=X+14gh_*,\qquad
\left\|\frac{dx}{d\tau}\right\|_M
 \le U:=2Q_0+7g.
\tag{9}
\]
The bound includes both continuous rows: phase evolves throughout and its
sine forcing is bounded globally, not held at its initial value. Arbitrary
common form and phase means do not enter it. The same bound holds along
the reference and along the comparison arcs between their sampled points,
because each such arc lies on the same pre-event or post-event full flow
with total structural elapsed time at most \(2h_*\).

Since \(\|q\|_{M^{-1}}=1\), the actual/reference recorded differences,
coupled with the same four sensor errors, obey
\[
|r_i-r_i^{\rm ref}|\le b_i,\qquad
(b_0,b_1,b_2,b_3)=(0,G_+U d_{1/2},0,G_+U d_2).
\tag{10}
\]
Consequently every actual compatible history induces reference readings in
\[
\widetilde I_i=[\ell_i-b_i,u_i+b_i].\tag{11}
\]
These are necessary comparison bands. They are not new measured readings,
independent clock errors or independent sensor noise. Their midpoint is
unchanged, and the exact common sensor offset still cancels. In particular,
the first whole-window increment is unchanged, the second changes by at
most \(b_3\), and the finite curvature changes by at most \(2b_1\).

The instantaneous chain-rule expression makes the same scope visible:
\[
\frac{d^2}{ds^2}(Gq^Tx)
 =G\rho(s)^2q^Tx_{\tau\tau}+G\rho'(s)q^Tx_\tau.
\tag{12}
\]
The second term must not be silently assigned to structural curvature.
Equations (8)-(11) bound its finite-reading consequence without supplying
observed derivatives. After dividing a second difference by \(H^2\),
the derivative-based discrepancy need not vanish as \(H\) decreases.
A shorter observation window alone does not remove clock uncertainty.

<a id="sine-clock-drift-mean-inference"></a>
## Necessary mean-rate inference through one shared reference envelope

Re-admit the eleven primitives, including the original four reading pairs,
positive global rate/gain priors and nonnegative \(\Lambda\). Compute
(8)-(11) with normalized exact scalars before interval materialization.
Invoke the [curvature inverse](SINE_FINITE_CURVATURE_INFERENCE.md#sine-curvature-necessary-projections)
once with the comparison bands, original \(\delta\), observed duration
\(H\) and otherwise unchanged source/input priors. Its unknown held clock
now denotes \(\bar\rho_1\), and its effective gain is
\[
J_1=G\bar\rho_1.\tag{13}
\]
Every actual history in (1)-(4) produces a compatible constant-mean
reference, so every necessary reference marginal contains the actual
original geometry, held gain and first-window mean. A strict reference
exclusion rules out all actual histories satisfying these premises.
The converse does not follow: marginal interval intersection discards
correlations among source, clock exposures and reading errors and need not
describe a jointly realizable history.

The parent report explicitly names
`first_window_mean_clock_rate_outer_bounds`. Its nested
`reference_envelope` is an auxiliary necessary calculation, not a claim
that the actual clock was constant. The original readings and sensor error
remain separate from the exact comparison bands and transfer allowances.
The source and original-angle meaning remain unchanged. No instantaneous
rate, clock profile or verification of the supplied derivative bound is
returned.

`bounded_candidate` means admitted necessary marginals; `incompatible`
means strict necessary exclusion; `unavailable` retains the child method's
source, rank or arithmetic limitation. A wide available interval is not a
proof of nonidentifiability. Global transfer bounds are candidates until
the report's original-source admission supplies their inference scope.
If \(\Lambda=0\), or the global rate prior is a singleton, both
exposure allowances vanish and the comparison child is exactly the existing
constant-clock calculation on the original readings.

<a id="sine-clock-drift-conditioning"></a>
## An informative positive-drift regime without a response evaluation

Retain the previous [finite-error budget](SINE_FINITE_CURVATURE_INFERENCE.md#sine-curvature-conditioning)
and add a strictly positive derivative allowance:
\[
\begin{gathered}
H=2^{-24},\quad X=Y=2^{-48},\quad (a_1,a_2)=(1/4,3/4),\\
G\in[1,2],\quad\rho(s)\in[1/2,2],\quad
\delta=2^{-90},\quad t_i\le\delta,\quad\Lambda=2^{-22}.
\end{gathered}\tag{14}
\]
Here \(t_i\) are the original primitive reading half-widths. No clock
realization, source realization or reading is selected. Let \(R_*\)
be the previous radius bound for the two leading inverse coordinates at
\(h_*=2H\). Only the final reading's extra allowance enters that
three-reading inverse. The inherited column norm is at most
\(36000/h_*\), so a sufficient radius is
\[
R_{\rm drift}=R_*+36000b_3/h_*<1/12500.
\tag{15}
\]
The original first-quadrant and positive-radius arguments therefore give
\[
w_{J_1}\le4R_{\rm drift}<1/3000,\qquad
w_b\le\frac{2R_{\rm drift}}{1/4-2R_{\rm drift}}<1/1500.
\tag{16}
\]
The reference's complete finite-curvature remainder \(D_C\) is unchanged
and remains below \(10^{-9}\). Its normalized numerator diameter now
satisfies
\[
d_A\le\frac{64\delta+16b_1}{H^2}<2^{-28}.
\tag{17}
\]
Reuse the positive natural coefficient bound \(K_->k_0=1/18000\)
and \(w_K\le11g w_b/8\), with the same proof and priors. Put
\(\beta=w_K/k_0\) and
\(t=2w_{J_1}+\beta+2w_{J_1}\beta\). Compatibility still guarantees
positive quotient numerators. The same endpoint-width argument yields
\[
\begin{aligned}
w_{\bar\rho_1}&\le
 2d_A/k_0+(2+D_C/k_0)t+2D_C/k_0<1/80,\\
w_G&\le2w_{J_1}+4w_{\bar\rho_1}<1/16.
\end{aligned}\tag{18}
\]
The strict bounds follow by rational substitution; the last inequalities
also hold with the conservative allowances in (16)-(17) and
\(D_C<10^{-9}\). Original actual-angle width remains at most
\(w_b+Y/4\). These are conditional exact-arithmetic budgets, not new
observations, a physical sensor specification or an evaluated drift trial.
Executable outward arithmetic and inverse availability remain separate
checks. The positive drift class still contains the exact ambiguity (7).

<a id="sine-clock-drift-boundary"></a>
## Implementation, evidence and research boundary

The [clock-drift adapter](../../src/tnfr/physics/relational_sine_clock_drift_inference.py)
uses shared primitive admission, one fresh curvature inverse and the global
comparison bound (9). It introduces no second inverse solver, profile fit,
trajectory generator or cached-report admission. The
[contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-clock-drift-inference)
owns input/output semantics; the [controls](../../tests/physics/test_sine_clock_drift_inference.py)
cover exact clock limits, exposure bounds, original-error association and
fresh complete-flow comparisons separately from frozen evidence.

The result does not infer a supplied clock law, verify its derivative
allowance from four samples, or identify physical time units. Common
all-row-law-rate/clock equivalence also remains if that additional unknown
is admitted. Source formation, support selection, work supply and future
maintenance retain their separate obligations. The
[sole execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns any subsequent admission; no previous frozen response is replayed or
reinterpreted under this larger clock class.

<a id="sine-clock-drift-reserved-protocol"></a>
## Prospective software evaluation with nonconstant clocks

This protocol evaluates the finite-transfer method on new complete-flow
responses. Its preparation, clocks, sensor, numerical budget and stopping
criteria are fixed before the first response. The existing held-clock
records are not inputs to this evaluation. The exact-exposure theorem
permits reuse of the unchanged structural-time producer; no new variable-
time solver or approximate clock quadrature is introduced.

### F1-F2: shared preparation and three declared histories

Use the public budget (14), including the positive allowance
\(\Lambda=2^{-22}\), and the same public geometry priors. Set the
hidden original geometry to \(b=23/16\), \(c=4/5\), the held sensor
gain to \(G=7/5\), and the target first-window mean to
\(\bar\rho_1=9/8\). These are supplied software preparations, not
parameters selected from an evaluated response. Both variable-clock cases
share their entire original source and sensor, with offset \(O=5/13\)
and the same four errors
\[
(\eta_0,\eta_{1/2},\eta_1,\eta_2)=2^{-91}(-1,1,-1,1).
\tag{19}
\]

For the eighteen nodes, use residual recipe \(r=4\):
\[
\widetilde u_i=\frac{(i+2)(r+2)}{2^{62}},\qquad
\widetilde v_i=\frac{(5i+3r)\bmod23+1}{2^{62}},\qquad
u=P_M\widetilde u,\quad v=P_M\widetilde v.
\tag{20}
\]
The nominal phase geometry and its full weighted centering retain the
source owner. Add common form \(6/11\), common phase \(-6/13\)
and the corresponding centered residual to each coordinate. Retain the
exact arrays and outward mathematical-pi source box. Every residual entry
is nonzero; weighted centering preserves the squared-norm bound, and
\(40\cdot114^2/2^{124}<2^{-96}\) admits both source norms.
The original actual long-arc mean is \(b-5/2^{65}\), since the first
two phase residuals differ by \(5/2^{62}\). Source-box materialization
must separately retain the declared norm bounds and this statistic.

The two variable clocks and one shared constant-mean reference are
\[
\rho_\pm(s)=\bar\rho_1\pm\frac{\Lambda}{2}(s-H/2),\qquad
\rho_{\rm ref}(s)=\bar\rho_1.
\tag{21}
\]
All satisfy the global prior and derivative allowance. Their first-window
means agree exactly. For \(\kappa=\pm\Lambda/2\), their three
structural segment durations are
\[
\left(\bar\rho_1H/2-\kappa H^2/8,\quad
      \bar\rho_1H/2+\kappa H^2/8,\quad
      \bar\rho_1H+\kappa H^2\right).
\tag{22}
\]
These are positive exact rational exposures. The reference uses the same
expression with \(\kappa=0\). Thus both nonconstant cases have
strictly nonzero half-time and final exposure differences from the
reference. Both rows of the complete autonomous law undergo the same
time change; the reduction is exact, rather than freezing phase or using
the mean rate separately in every actual segment.

An additional ambiguity control is the declared profile
\[
\rho_{\rm companion}(s)=\rho_+(s)
       +\epsilon\cos(4\pi s/H),\qquad \epsilon=\Lambda H/32.
\tag{23}
\]
Its derivative has magnitude at most
\(\Lambda/2+\pi\Lambda/8<\Lambda\), and its distance from
\(\bar\rho_1\) is at most \(25\Lambda H/32\), well inside
the fixed global prior. Its three exposures equal those of \(\rho_+\),
while every sampled instantaneous rate differs by \(\epsilon>0\).
By (6), the same complete trajectory record encloses its sampled states.
Retain this explicit association and the analytic exposure certificate;
do not rerun an identical producer or count the association as a second
independent numerical response. No profile is inferred from a mean-rate
output.

### F3: exact exposure execution and separate public inference

Generate exactly three complete structural histories: the two clocks in
(21) and their shared reference. Each uses the existing
[full-state producer](../../src/tnfr/physics/relational_sine_two_port_readout.py)
with three sequential exposures (22), phase increments \((1/4,0,1/2)\),
and all thirty-six endpoint intervals carried unchanged. The observation
times remain \((0,H/2,H,2H)\). There is no event at the half-time
sample and no state reset at either sample or the second phase jump.

Use one direct source-box order-four Taylor step per segment, its
whole-tube fifth-order remainder, the fixed sixteen-attempt Picard budget,
and shared outward dyadic-128 arithmetic. Retain all nine full certificates,
their source associations, strict inclusion margins, achieved horizons
and both global clock coordinates. Do not subdivide, increase precision,
replace the source box by its midpoint or retry a failed budget.
Each sensor reading must have numerical half-width at most
\(\delta=2^{-90}\), separately from its physical-error allowance.
Optional acute flags remain observations, not stopping requirements.

For each variable-clock case, pass only the eleven documented primitives
to the isolated inverse worker. In particular, `probe_duration` is observed
\(H\), and the clock inputs are the global prior and derivative bound.
The source arrays, exact geometry, true gain, clock profiles, exposures,
mean rate, reference trajectory and realized errors remain on the
producer/evidence side. Retain allowlisted packets and hashes. The
reference is used only for the independent transfer assessment, with no
inverse evaluation. This is software information exclusion, not a secrecy
or execution-authentication guarantee.

### Response-free signed separation budget

The transfer controls have a prospective finite-flow sign, not merely
different clock formulas. Set \(T=4H\), \(Q_0=X+7gT\) and
\(V=Y+2gTQ_0\). Equation (9) and the phase row give, throughout
either pulse window and all comparison arcs,
\[
\|P_Mx\|_M\le Q_0,\qquad
\|\theta-\theta_{\rm nominal}-m_\theta\mathbf1-a_jq\|_M
 \le V.\tag{24}
\]
The centered nominal phase uses the same source convention as (20).
At its pulse-shifted value,
\(q^Tf=-F_j\), where
\(F_j=2\sin(3a_j/2)\cos(b-a_j/2)>1/6\).
For \(a_1=1/4\), use \(\sin(3/8)>1/3\) and the alternating
cosine lower polynomial at \(21/16\), which exceeds \(1/4\).
For \(a_2=3/4\), use \(\sin(9/8)>1/2\) and
\(\cos(17/16)\ge223/512>1/6\). The sine map has weighted
Lipschitz bound two, \(\|A\|_M\le2\), and
\(1/3216<\gamma<g\). Consequently
\[
q^Tx_\tau<-m,\qquad
m:=\frac1{6\cdot3216}-2Q_0-2gV>\frac1{20000}.
\tag{25}
\]
All bounds follow from the declared primitives and elementary inequalities;
no reserved response or measured derivative is used.

The common sensor errors cancel between histories. Write \(D_{\pm,i}\)
for the true variable/reference recorded difference. The half-time and
final signs are
\[
D_{+,1}>0>D_{-,1},\qquad D_{+,3}<0<D_{-,3},
\tag{26}
\]
with half-time magnitudes exceeding
\(G\Lambda H^2/(16\cdot20000)=(14336/3125)\delta>4\delta\),
and final magnitudes exceeding eight times that amount. Exact equality at
\(0,H\) gives
\(C_- - C_+>(57344/3125)\delta>16\delta\).
An individual recorded interval of radius at most \(\delta\) has
width at most \(2\delta\). Thus its variable/reference difference
can extend at most \(4\delta\) beyond the true difference, and the
two curvature intervals' difference at most \(16\delta\).
The frozen numerical half-width criterion therefore suffices to retain
all signs and the strict curvature ordering in the recorded intervals.

For containment, the actual gain and half-budget slopes yield
\(|D_{\pm,i}|\le(7/20)b_i\), for \(i=1,3\).
The derivative cap is the active term in (8), and
\(b_1/\delta=2^{18}U>500\), with \(b_3=8b_1\).
Hence the additional \(4\delta\) enclosure allowance is less than
\((13/20)b_i\). The entire retained difference intervals must fit
strictly inside the transfer bands whenever their individual reading-width
criteria pass. Inverse availability and the complete validated horizons
still require the declared executable checks.

### F4: fixed coverage, transfer and ambiguity criteria

The first evaluation passes only if all source/clock/sensor admissions,
complete horizons, endpoint associations and numerical budgets pass, and:

- Both public inverses return `bounded_candidate`, with source, drift
  transfer, finite curvature, rank and positive-division evidence admitted.
  Their necessary marginals cover the original nominal and actual angle,
  the retained initial actual-angle box, \(J_1=63/40\), \(G=7/5\)
  and \(\bar\rho_1=9/8\).
- Strict widths are below `1/1024` for the actual original angle,
  `1/2048` for effective gain, `1/16` for gain and `1/80` for mean rate.
  Nonempty marginals do not imply joint realizability of every retained
  parameter tuple.
- Each variable-clock half-time and final recorded interval has the strict
  signed separation (26) from its reference interval. The entire difference
  interval lies in the corresponding \([-b_i,b_i]\) from (10).
  At \(H\), every pair of actual/reference full-state coordinate boxes
  intersects, before and after the common phase event. This is numerical
  consistency with the exact theorem, not an overlap proof of equality.
- The finite-curvature intervals have the strict ordering \(C_+<C_-\).
  Each uses the same four recorded readings and their original sensor
  errors, without an independently supplied derivative observation.
- False global clock prior `[25/32,13/16]` and false gain prior `[31/16,2]`
  leave the corresponding coarse child available but produce a strict
  curvature exclusion. The false angle prior `[11/8,353/256]` is excluded;
  equal cumulative amplitudes `(1/4,1/4)` give the inherited rank abstention.
  These controls use the same actual readings with altered declarations.
- Both full-window increments exclude the complete phase-blind alternative
  \(x_\tau=-Ax,\ \theta_\tau=\gamma Ax\), whose necessary recorded
  band remains \([-B_{\rm heat},B_{\rm heat}]\), with
  \(B_{\rm heat}=2G_+\rho_+HX+2\delta\). The varying clock still
  gives at most \(\rho_+H\) structural time per window.
- The companion clock's global and derivative bounds, strictly different
  sampled rates, equal exposures and association with the positive-slope
  complete record pass independently reconstructed exact checks.

Freeze this prospective text, the machine-readable protocol, committed
runtime source, evaluator and public worker before any response. The
exclusive attempt ledger precedes the first flow. Retain the first outcome,
including partial responses, failed criteria, unavailability or exception;
no budget adjustment or second attempt is admitted. Corrections require
separate evidence and cannot replace that outcome.

Success would establish finite software robustness of the mean-rate/gain
inference to the declared clock variation, and exercise a nonzero transfer
allowance. It would not identify the instantaneous profile, establish
physical clock units or feasible laboratory noise, select a unique law,
or derive source formation and maintenance. The execution plan alone owns
the next research boundary.

<a id="sine-clock-drift-result"></a>
## Retained first nonconstant-clock evaluation

The first reserved evaluation returned
`certified_reserved_clock_drift_inference`: all **112 fixed conditions**
passed. The count comprises sixteen reference-history conditions,
forty-three for each variable-clock history, one shared-source condition,
one paired-curvature condition and eight companion-clock conditions.
The [protocol](../../docs/assets/sine_formed_classes/clock-drift-inference-v1.protocol.json),
[source archive](../../docs/assets/sine_formed_classes/clock-drift-inference-v1.source.zip),
[attempt](../../docs/assets/sine_formed_classes/clock-drift-inference-v1.attempt.json),
[response](../../docs/assets/sine_formed_classes/clock-drift-inference-v1.json)
and [manifest](../../docs/assets/sine_formed_classes/clock-drift-inference-v1.manifest.json)
preserve that first outcome. The prospective statements above retain their
pre-evaluation wording; no retry, altered budget or older producer replay
was used.

All nine full thirty-six-coordinate segments completed their prescribed
structural exposures with strict Picard inclusion. Every endpoint was
carried unchanged into the next declared phase event or passive
continuation. Both public inverse packets contained only the eleven
documented primitives. Neither the reference history nor the companion
profile entered those packets or received a separate inverse evaluation.

Both necessary marginal reports cover the original nominal angle,
\(B_{\rm initial}=23/16-5/2^{65}\), its retained initial source box,
\(J_1=63/40\), \(G=7/5\) and \(\bar\rho_1=9/8\).
The following widths are decimal **upper bounds**, rounded upwards to
twelve places from the stored exact rational widths. The stopping tests
use the rational values and their frozen strict thresholds.

| Clock history | Original-angle width | Effective-gain width | Gain width | First-window mean-rate width |
| --- | --- | --- | --- | --- |
| Positive slope | `0.000057836407` | `0.000212441681` | `0.000803235837` | `0.000493713258` |
| Negative slope | `0.000057836407` | `0.000212441681` | `0.000803235731` | `0.000493713316` |
| Frozen strict threshold | `1/1024` | `1/2048` | `1/16` | `1/80` |

The independently generated reference comparisons exercise nonzero
transfer, rather than only exposure-equivalent clocks. In units of the
supplied sensor allowance \(\delta\), the retained differences have
the following outward-rounded enclosures:

| Clock history | Half-time recorded difference / delta | Final recorded difference / delta |
| --- | --- | --- |
| Positive slope | `[5.341891,5.341892]` | `[-200.581040,-200.581039]` |
| Negative slope | `[-5.341892,-5.341891]` | `[200.581039,200.581040]` |

Every entire difference interval lies inside its independently rebuilt
transfer band. Both histories have the predicted strict sign at each
sample, and their recorded curvature intervals satisfy \(C_+<C_-\).
At \(H\), all actual/reference coordinate enclosures intersect before
and after the second event. The exact equality follows from the exposure
theorem; the overlap is its numerical consistency check.

The false clock/gain priors leave their coarse children available but are
excluded by the finite-curvature refinement. The false angle is also
excluded; equal cumulative inputs produce rank-based unavailability; both
full-window increments exclude the specified complete phase-blind law.
These controls retain their prospective meanings and do not establish
joint realizability of alternative marginal tuples.

The companion record reconstructs its positive global rate, derivative
bound and integer cosine turns. Its sampled rates differ from those of
the positive-slope history by \(\epsilon=2^{-51}\), yet its sample/event
exposures agree exactly. It therefore uses the same full-state enclosure
by theorem, with an explicit source/sensor/trajectory association. This
is an exact ambiguity control, not a second independently simulated or
measured response.

The result establishes finite software robustness of the mean-rate/gain
inference within the frozen preparation, clock class and held observation
law. The extremely small stipulated sensor error and clock variation are
numerical protocol premises; no experimental sensor performance or clock
calibration follows. Instantaneous-profile ambiguity, joint source/noise
feasibility, physical units, formation and maintenance remain separate
questions. The global smooth-flow domain flags establish no acute-sector
or identity-retention certificate.

The source base is `1a717ca8f4f264f277bfdf661d9aeee4dd103af9`, with no
runtime overlays. Its archive SHA-256 is
`1a08af6a46c091cb49b6a1da8d12186768b824470facf14db5131193500a10b0`.
The archived prospective owner contains **25,898 bytes**, with SHA-256
`f1cebe3ab44153e51d700dbc7aba6bbc270cf87b375d08a70b2df1d916e98ce7`.
Those bytes remain the unchanged prefix of this owner, modulo checkout
newline conversion. Hashes associate retained records; they do not
independently authenticate chronology, execution or physical acquisition.
The [read-only evidence audit](../../tests/physics/test_sine_clock_drift_evidence.py)
rebuilds the exposure, transfer, certificate and inverse arithmetic without
rerunning any producer, inverse or archived worker. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the closed gate and its separate inactive resumption boundary.

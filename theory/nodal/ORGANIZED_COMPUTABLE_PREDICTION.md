# A computable forecast after finite observation

<a id="organized-computable-prediction-declaration"></a>
## Frozen prospective comparison

The [preparation theorem](ORGANIZED_MEMORY_INITIALIZATION.md#organized-memory-initialization-result)
initializes one carried ten-coordinate reference without measuring hidden
phases or resetting its post-formation state. This comparison asks whether
that reference produces a certified finite forecast of the state retained
after the [unchanged observation word](ORGANIZED_FINITE_OBSERVATION.md).
The endpoint distinguishes retained observation backaction from omission of
the intervention; it does not identify the same winding a second time.

| Item | Fixed choice and scope |
| --- | --- |
| Fine preparation | Original independent class-one C9 children, zero-sum primitive errors at most `1e-6`, quarter formation 200, admitted full-state radius `1/12` and energy trap, uninterrupted recovery dwell `8e10`, decay bound `2^-80`; retain every original acquisition, support and work premise |
| Branch and attachment | Only `outer_one`: five recovered children, supplied common forms `4092*(c-2)`, zero common phase origins, unchanged-state same-slot attachment at `tau=0`; no hidden coordinate is set to zero |
| Fine law | Unit C9-by-C5, nodes `9*c+j`, 90 unit edges, held unit capacities, full-circle pairwise sine; `x'=(-L_f*x+gamma*S(theta))/4+R*f`, `theta'=gamma*L_f*x/4`, `gamma=1/(1023*pi)` |
| Clock and complete history | `tau=(1023/1024)*t`; unforced outer formation on `[0,200]`, original fifteen ramps and fifteen preceding plateaus on `[200,215.00015]`, then unforced continuation to `tau=216`; no donor, changed support or further event |
| Input word | Mark children `0,1,2,3,4`; each has plateaus of length `1e-5` at command levels `0,+epsilon,-epsilon`, each followed by a unit-duration cubic ramp with increment `epsilon,-2*epsilon,epsilon`, `epsilon=1/10`; use the existing waveform constants and ordering |
| Reduced initialization | At attachment, `U_c=4092*(c-2)`, `Phi_c=0`, with exact zero numerical radius; retain the analytically bounded fine-source discrepancy separately |
| Comparator | Identical preparation, law, support, capacities and clock, with the observation word omitted: `f=0` throughout `[0,216]`; it shares the actual formation history, not a target reset |
| Reserved observable | The five signed mean forms in child order `0,1,2,3,4` at `tau=216`; no in-word reading is fitted or used for initialization |
| Numerical method | Shared prescribed-time identity-metric validated Taylor steps, order 16, dyadic128 outward interval arithmetic, maximum step `1/8`, proved global same-time growth bound `g=1/3000`; separate steps at every input switch |
| Computation budget | At most 1,800 accepted steps per branch including its common prefix; planned grid: 1,600 formation steps, then 15 plateau steps, 120 ramp steps and 8 unforced-tail steps per branch. Compute the common prefix once and fork its unchanged center and retained radius: 1,743 steps per branch, 1,886 distinct steps in total |
| Numerical acceptance | Retained ten-coordinate Euclidean radius at most `1e-15` at every accepted step, successful whole-step domain and Picard checks, completed prescribed horizon; no adaptive step refinement, precision/order escalation, parameter scan or retry |
| Source and reading allowances | Each branch's intrinsic mean-form error is bounded by `E_source=49/31088000000000000000000000<1.6e-24`; each endpoint scalar has arbitrary correlated additive observation error at most `eta=1e-14` |
| Conditional exact quantities | Input, clock, support and capacities are stipulated exact; their additional error budgets are zero conditional premises, not instrument certifications |
| Decision | Compute the maximum coordinate separation of the two numerical endpoint interval vectors. Admit the distinction only if it is strictly greater than `2*(eta+E_source)` and both numerical certificates and all identity margins pass; retain all five intervals and the margin |
| Failure | Preserve an unavailable result if any declared numerical/domain/budget obligation or the separation fails; do not reinterpret it as a failed source formation or a physical falsification |
| Evidence | Freeze this owner, consuming implementation and driver before one new endpoint assessment. Preserve source/configuration, ordered stages, arithmetic/backend, all compact step witnesses and any unavailable reason. A losslessly compressed archive may retain the exact report bytes. No original source assessor or frozen response producer is rerun |

The comparator is a declared counterfactual input history, not a competing
constitutive law. Separation would establish a conditional structural
forecast of finite backaction even though the final commanded offsets are
zero and both histories retain winding one. It would establish neither a
measured response nor resolved hidden constituent feedback or physical
identification. Independent laboratory preparation and observation remain open.

## 1. Exact removal of the commanded offsets

Let L be the five-cycle Laplacian and let `k_c=e_c-1/5`. On a unit-duration
ramp with signed increment a, local time `r in [0,1]` and preceding command
level b, the existing waveform is

\[
 s(r)=3r^2-2r^3,\qquad H=(b+a s(r))k_c,
 \qquad z_{cmd}=\frac{4a}{\gamma}L^+k_c s'(r),
 \qquad f=z_{cmd}'+Lz_{cmd}/4.                 \tag{1}
\]

On a plateau H is constant and `z_cmd=f=0`. Both command offsets vanish
before the word and after its release. Earlier child blocks finish at zero,
so only the current marked child's direction is present in H. The command
satisfies `H'=gamma*L*z_cmd/4`. Its phase and form offsets are continuous
at the switches; bounded input switches introduce no state jumps.

For the exact carrier reference define `Y=U-z_cmd`, `Psi=Phi-H`. Substitution
in both complete rows gives

\[
 Y'=-LY/4+\gamma S_5(\Psi+H)/4,\qquad
 \Psi'=\gamma LY/4.                           \tag{2}
\]

Thus large input terms cancel analytically rather than through subtraction
of rounded numerical values. This is an exact time-dependent chart change,
not a new constitutive law or an approximation to the input. At every
plateau endpoint the form offset is zero; at `tau=216` both offsets are zero,
so `Y=U` and `Psi=Phi`. For the omitted-word comparator H is identically
zero. Each branch continues its entire reached state across every boundary.

The consumer uses the shared full-sine field. Its five-cycle implementation
has degree two; halving both returned structural-time rows reproduces the
degree-four carrier inherited from the full product graph. This common
factor changes neither gamma nor the declared tau clock. Direct comparison
with (2) and the unchanged waveform checks this adaptation; resemblance of
the two supports alone would not justify transferring their coefficients.

## 2. A global same-time numerical growth bound

For an oriented ring incidence B put
`K=B*diag(cos(B^T*(Psi+H)))*B^T`. The state Jacobian of (2), with the same
prescribed time in both compared solutions, is

\[
 J=\frac14
 \begin{pmatrix}-L&-\gamma K\\\gamma L&0\end{pmatrix}.
                                                        \tag{3}
\]

No acute-angle assumption is required. Since
`0<=L-K<=2L` in the quadratic-form order and `||L||2<=4`,
`||L-K||2<=8`. For a real perturbation `(a,b)`,

\[
 (a,b)^T J(a,b)
 =-a^TLa/4+\gamma a^T(L-K)b/4
 \le2\gamma\|a\|_2\|b\|_2
 \le\gamma(\|a\|_2^2+\|b\|_2^2).             \tag{4}
\]

The identity-metric logarithmic norm is therefore at most
`gamma<g=1/3000` globally. H is a prescribed time function, not an extra
uncertain state coordinate; its time derivative does not enter this
same-time state Jacobian. Timing or input uncertainty would require a new
error argument and is not absorbed in (4).

The [shared metric Taylor kernel](../../src/tnfr/mathematics/_validated_metric.py)
retains a full Euclidean ball. On each fixed smooth stage it establishes a
whole-step Picard tube, bounds the Taylor polynomial and remainder with
outward arithmetic, and propagates the radius using the proved growth
bound. Prescribed-time jets include time derivatives of H. Switches are
separate step endpoints; no Taylor interval straddles a different waveform
branch. The small growth bound is not a replacement for the local remainder
or whole-step existence checks.

The common formation prefix is one exact history shared by both comparisons.
Reusing its computed center and complete radius preserves its uncertainty;
neither tail starts from a newly exact endpoint. Per-branch interval bounds
can discard correlations between the two numerical tails conservatively.
The declared maximum-coordinate separation uses those full bounds, not a
cancellation of unproved common numerical error.

## 3. Source uncertainty through the enlarged endpoint

The [initialization proof](ORGANIZED_MEMORY_INITIALIZATION.md#organized-memory-initialization-result)
retains separate conservative recovered child radii
`q_x=1e-9`, `q_theta=2e-12`. The original zero-sum hypotheses give exact
attachment means, while the five actual hidden states remain present.
Their combined initial radii obey `r0<3e-9`, `z0<6e-12`.

The full-state heat comparison is valid for the entire horizon `Horizon=216`:
identical uniform-child inputs cancel in the fine/carrier difference rows,
including on the unforced suffix. Rebuilding from these primitives gives

\[
 Z=2(6\cdot10^{-12}+g\,3\cdot10^{-9})=1.4\cdot10^{-11},
 \quad \rho=gZ^2/9,\quad k=216g=9/125,
\]
\[
 E_U\le\frac{216\rho}{1-k^2}
 =\frac{49}{31088000000000000000000000}
 <1.6\cdot10^{-24},\qquad E_\Phi\le kE_U.       \tag{5}
\]

The quadratic collective sine remainder supplies rho. Its initial mean
error is zero; no post-formation phase is observed or inserted. The source
theorem's previous displayed interval ended with the word, but its actual
216-unit estimates and input-cancellation proof give (5) through the new
endpoint. This extends the stated comparison interval without changing the
preparation, dwell, recovery allowance, law or numerical precision. The same
argument covers the omitted-word history. It asserts a uniform conditional
source cover, not knowledge of an individual fine preparation.

If a numerical branch ends with center `m_c` and retained radius r, its
actual mean form lies within `m_c +/- (r+E_U)`. An admitted future scalar
reading lies within `m_c +/- (r+E_U+eta)`. Correlated reading errors are
allowed; they are not reduced by averaging or by taking five coordinates.
For the two numerical interval vectors I and J define

\[
 d_\infty=\max_c\max(0,\inf I_c-\sup J_c,
                              \inf J_c-\sup I_c).
                                                        \tag{6}
\]

The declaration's strict `d_infinity>2*(eta+E_U)` condition makes their
admitted record boxes disjoint in at least one coordinate. Numerical radii
are already included in I and J and must not be added again. The choice of
the maximum over all five coordinates is fixed before evaluation; no
coordinate, time, input or tolerance is chosen from the endpoint response.

## 4. Identity and work on the unforced continuation

Extend the finite-word comparison to observation-relative time `T=16`.
Set `y=x-R*z_cmd` and `e=theta-theta(200)-R*H`. The exact carried rows of
the finite-word owner still hold after release, with both command offsets
zero. For `X=1/100`, `g=1/3000`, they imply

\[
 \|y\|_\infty\le M=X+16g=23/1500,\qquad
 \|e\|_\infty\le E=2gX+32g^2=23/2250000.       \tag{7}
\]

The command changes every outer edge by at most `epsilon=1/10` throughout
the entire interval and every internal edge by zero. With `zeta=1e-6`,
`A=7/5`, the sufficient margins remain

\[
 2\pi/9-2\zeta-2E,\quad
 \pi/2-2\pi/9-2\zeta-2E,\quad
 \pi/2-A-\epsilon-2\zeta-2E,\quad
 \pi/2-2\zeta-4E.                              \tag{8}
\]

All are strictly positive; `pi>3.14` alone puts the outer margin above
`314899/4500000`. Hence the five internal and nine outer winding-one
identities and the circular observation collar persist through `tau=216`.
The unforced comparator satisfies the same conservative margins with zero
command. This continuation uses the already proved actual-source inclusion
at `tau=200`; it does not claim that every admitted norm-box member forms.

The unchanged fine storage satisfies

\[
 H_f'=-\|L_f x\|_2^2/4+9(LU)^Tf.              \tag{9}
\]

On the suffix `f=0`, so supplied work is exactly zero and storage cannot
increase. The unit product graph has degree four and therefore
`||L_f x||infinity<=8M`, not `4M`. For suffix duration
`delta=216-215.00015=19997/20000`, its additional loss is bounded by

\[
 Loss_{suffix}\le(45/4)(8M)^2\delta
 =720M^2\delta
 =\frac{10578413}{62500000}=0.169254608.        \tag{10}
\]

All earlier preparation, attachment and finite-word work obligations retain
their existing owners. Zero suffix work does not erase their costs or
make the preceding intervention autonomous. The continuation introduces no
support-event storage jump or reservoir. These are structural units, not
calibrated laboratory energy or time.

## 5. Instrument and evidence boundary

The [fixed consumer](../../src/tnfr/physics/_organized_computable_prediction.py)
reuses the shared full-sine field, interval arithmetic, prescribed-time
Taylor and metric-ball owners. The
[exclusive driver](../../benchmarks/organized_computable_prediction.py)
records one assessment after the declaration revision is fixed. Compact
step witnesses retain each time, duration, tube, Picard margin, polynomial,
remainder, growth, local error, center and radius; initial state and the
ordered preceding endpoints reconstruct each incoming ball. They permit
inspection without discarding the numerical uncertainty history. Any
compressed evidence must decompress to the preserved exact report bytes.

The [independent controls](../../tests/physics/test_organized_computable_prediction.py)
check the waveform, shared-field adaptation, source/identity/resource
arithmetic, budget decisions and error accounting without evaluating this
reserved endpoint. They supplement, rather than replace, the conditional
enclosure proof. A point trajectory, two-solver agreement, or a cached
passing flag is insufficient evidence for this declaration.

The [sole execution queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns any subsequent investigation. This owner changes no original frozen
protocol, response or source producer and activates no physical campaign.

<!-- FROZEN DECLARATION AND PROOF END: append evaluated results only below. -->

<a id="organized-computable-prediction-result"></a>

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

These are conditional results and implementation controls. No new reserved
response is evaluated and no earlier frozen protocol, source or result is
changed. The [ontology](../EMERGENT_ONTOLOGY.md#generative-bound-organization)
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

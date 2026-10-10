# Current-amplitude discrimination with calibrated uncertainty

<a id="robust-current-protocol"></a>

## Prospective decision and scope

The [five-child specification](ORGANIZED_ELECTRICAL_PROTOCOL.md) gives a
phase-free current statistic for an exact model. This gate asks whether
one fixed input and one fixed observation separate that amplitude from a
doubled amplitude after admitting nonzero uncertainty. It does not ask
which law nature follows. No physical record, apparatus or new acquisition
is evaluated.

| Item | Fixed declaration |
| --- | --- |
| Models | Full C9-by-C5 laws (3) with mu equal to 1 or 2; 45 forms and phases, 90 unit edges, held unit capacities, full-circle sine, neutral uniform-child inputs. The C5 carrier is an exact electrical reference; discarded fine state remains in the full model |
| Preparation | Independently supplied observation-start class: zero global form mean, `abs(x_i)<=X=.01`, `theta_cj=psi_j+Phi_c+z_cj`, zero child means of z, `abs(z_cj)<=zeta=1e-6`, principal mean gaps at most `Aphase=7/5`, winding in `{-1,0,1}`. Common phase is retained. This is not a formation certificate for the alternative or for changed calibration coefficients |
| Input | Same nominal cubic word for both laws: fifteen unit-duration ramps, each preceded by a plateau `h=1e-5`; epsilon `.1`, marked increments `.1,-.2,.1`; duration `T=15.00015`. Nominal coefficients and sensor conversion are fixed before any response |
| Calibration and clock | Independent positive ratios C,G,A,D defined below, each in `[1-1e-8,1+1e-8]`. Exact C5 component shapes and a constant command-clock ratio; arbitrary parasitics or time-dependent clock drift are outside this first domain |
| Tracking | Neutral child-uniform discrepancy d in (3), with total `integral(norm_inf(d))<=F=1e-4`; on each nominal plateau its integral is at most `P=1e-15`. No derivative or pointwise bound is presumed |
| Observation | Same 150 nominal-voltage-to-form scalars in `(5,3,2,5)` order for both laws; arbitrary correlated additive error `eta_raw<=1e-14`. Each reading is a sample or positive normalized average within its own inward endpoint aperture of radius `delta<=1e-13` in nominal clock units |
| Statistic | Five unnormalized current norms (13), computed with nominal coefficients; no measured or inferred phase and no amplitude-dependent redefinition of the observed statistic |
| Decision | Require every whole-history identity/collar margin to be positive for both laws and the upper mu=1 current envelope to be strictly below the lower mu=2 envelope. Failed sufficient bounds mean unavailable separation, not impossibility or physical rejection |
| Evidence | Commit declaration, helper and driver before one analytical assessment of the stated joint box using outward dyadic128 intervals. No trajectory, scan, source replay, fitting or reserved experimental response |

The error allowances are prospective mathematical premises, not measured
instrument capabilities. A common initial-state class need not specify
identical hidden states under the two hypotheses: separation must hold
uniformly for either admissible initial state. The existing acquired source
is included at mu=1 and nominal coefficients through its earlier proof.
That proof does not establish acquisition under mu=2 or perturbed laws.

## 1. One nominal laboratory input and observation

Choose independently calibrated nominal `c0,g0,a00>0` and keep

\[
 \gamma=1/(1023\pi),\quad b_0=4g_0/c_0,\quad
 \kappa_0=a_{00}\gamma g_0/9,\quad v_0=a_{00}\gamma g_0/c_0,
 \quad j_0=a_{00}\gamma^2g_0^2/c_0.
 \tag{1}
\]

Here a00 is the nominal voltage-phase constant, not a derived TNFR constant.
Let `C=c/c0`, `G=g/g0`, `A=a0/a00`. A positive constant D relates the command
coordinate to actual laboratory time by `s=b0*D*t_lab`. Thus D is instrument
clock calibration, separate from the matching relation `b=4g/c`.

The physically specified current is
`I_cmd=9*b0*kappa0*f(s)`, with the same f for every calibration member and
both laws. Delivery discrepancy is retained explicitly. Readout always uses
the nominal voltage coordinate `U=L5+ V/v0` and nominal inferred charge
`q_nom=c0*L5+ V=9*kappa0*U`. Actual charge is `q=C*q_nom`. A common voltage
gauge cancels; different unknown offsets require the declared sensor error.
For per-island voltage error eta_V the nominal form error is at most
`4*eta_V/(5*v0)`. Calibration is not silently refitted in this conversion.

For a circuit with `j_actual=mu*a0*gamma^2*g^2/c`, transforming both rows
to these fixed coordinates gives the coefficients

\[
 \lambda=G/(CD),\quad \rho=1/(CD),\quad
 \eta=1/(AD),\quad \beta=AG^2/(C^2D).
 \tag{2}
\]

The full fine comparison explicitly supplies the corresponding laws

\[
 x'=(-\lambda L_{45}x+\mu\gamma\beta S(\theta))/4
                   +R(\rho f+d),\qquad
 \theta'=\gamma\eta L_{45}x/4.
 \tag{3}
\]

Prime denotes s. R repeats a child coordinate across its nine slots. On
the helical carrier these equations give exactly the C5 circuit rows.
Off it, internal cancellation leaves the actual mean external currents,
not an assumed closed sine law. The complete fine extension is a model
premise; a passive physical fine-node circuit has not been constructed.
In particular the full fine source is not replaced by an ideal carrier.

Physical current discrepancy maps to
`d=delta_I/(9*b0*kappa0*C*D)`. The total integral F must be admitted from
independent delivery evidence, including switch timing and leakage. The
separate bound P is essential: a small total input error can otherwise
concentrate in a reading window and mimic its signal. Neutrality preserves
the zero form mean; nonneutral errors require another law and observation.

## 2. Whole-history bound for each law

Reuse only the explicit waveform, not the old response bound. Let z(s)
be its known form offset and H(s) its phase command:

\[
 f=z'+L_5z/4,\qquad H'=\gamma L_5z/4,
 \quad \|H\|_\infty\le4\epsilon/5,
 \quad J_z:=\int_0^T\|L_5z\|_\infty ds=32/(5\gamma).
 \tag{4}
\]

The last equality follows from `L5*z=4*a*k_c*s_cubic'/gamma`,
`norm_inf(k_c)=4/5`, nonnegative cubic derivative and total absolute
increment two. Each offset vanishes at its ramp endpoints and throughout
plateaus. H is continuous, starts and ends at zero, and each ring edge
has nominal excursion at most epsilon.

Write `y=x-rho*R*z`. Its actual residual input is
`w=rho*(1-lambda)*R*L5*z/4+R*d`. Heat contraction on the unchanged support
and `norm_inf(S)<=4` give, for each fixed calibration member,

\[
 J=\rho|1-\lambda|J_z/4+F,\qquad
 M_\mu=X+\mu\gamma\beta T+J,\qquad \|y\|_\infty\le M_\mu.
 \tag{5}
\]

The phase drift after subtracting the actual command eta*rho*H satisfies

\[
 e_b'=\gamma\eta L_{45}y/4
 =-\frac{\gamma\eta}{\lambda}y'
   +\frac{\mu\gamma^2\eta\beta}{4\lambda}S
   +\frac{\gamma\eta}{\lambda}w.
 \tag{6}
\]

Integration across the whole word, including every switch, bounds the
departure from the *nominal* initial shifted state by

\[
 \|\theta(s)-\theta(0)-RH(s)\|_\infty\le
 E_\mu=2\gamma\eta M_\mu/\lambda
                 +|\eta\rho-1|\,4\epsilon/5.
 \tag{7}
\]

No stationary background or hidden-phase input is used. The sufficient
identity margins are, separately for mu=1 and mu=2,

\[
 2\pi/9-2\zeta-2E_\mu,\quad
 \pi/2-2\pi/9-2\zeta-2E_\mu,\quad
 \pi/2-A_{phase}-\epsilon-2\zeta-2E_\mu,\quad
 \pi/2-2\zeta-4E_\mu.
 \tag{8}
\]

Positive margins retain all five internal and nine outer identities and
the circular collar. They neither select a winding nor prove formation.
At final release the command and offset vanish; residual state is retained.

For boxes, compute positive upper bounds using shared outward gamma and
coefficient intervals, lambda's positive lower endpoint where divided,
and the absolute maxima of `1-lambda` and `eta*rho-1`. Discarding box
correlations enlarges the bounds; it cannot silently improve a verdict.

## 3. Readings with aperture and arbitrary integrable tracking

On a nominal plateau, actual child means obey

\[
 U'=-\lambda L_5U/4+\mu\gamma\beta\bar S/4+d,
 \qquad \|U\|_\infty\le M_\mu,\quad \|\bar S\|_\infty\le2.
 \tag{9}
\]

An inward aperture stays entirely within that plateau. Relative to its
endpoint the form error of a sample, or positive normalized average, is
bounded by

\[
 n_\mu=\eta_{raw}+(\lambda M_\mu+\mu\gamma\beta/2)\delta+P.
 \tag{10}
\]

The P term needs no pointwise tracking bound. Signed observation kernels
need their own total-variation bound and are outside this declaration.
Two-sided jitter crossing a ramp cannot use (10). The first sample looks
forward into its plateau and the last backward; windows may overlap but
must remain inside it. We require `0<=delta<=h`.

Keep the same nominal corrected increment at receiver `(c+1) mod 5`:
`D=Q_r-P_r+h*[L5(P+Q)]_r/8`. Its reference current is the initial fine
current shifted by nominal H. Define

\[
 B_{2,\mu}=4\lambda^2 M_\mu+2\lambda\mu\gamma\beta
                         +4\mu\gamma^2\beta\eta M_\mu.
 \tag{11}
\]

The corrected-increment error is bounded by

\[
 b_\mu=\mu\gamma\beta hE_\mu
       +|1-\lambda|hM_\mu+B_{2,\mu}h^3/12
       +P(1+h/2+\lambda h^2/4)+(2+h/2)n_\mu.
 \tag{12}
\]

The first term bounds phase/current transport; the second is the unmatched
diffusion coefficient in the nominal correction. The last is the exact
absolute sum of the six raw endpoint coefficients. Arbitrary correlations
between raw errors are permitted.

The tracking/trapezoid terms do not differentiate d. To see this, set
`v(s)=integral_plateau_start^s R*d` and `w=x-v` on that plateau.
The smooth drift differentiated along the absolutely continuous state
has norm at most B2 apart from `-lambda*L45*R*d/4`. The trapezoid Peano
kernel has maximum `h^2/8`; `norm_inf(L45)<=8` therefore contributes at
most `lambda*h^2*P/4`. The integral of d in v has trapezoid kernel
`s-h/2`, bounded by h/2, contributing `h*P/2`. Direct input in the mean
increment contributes P. This proves (12) for integrable tracking, without
assuming finite derivative, exact switches or an unforced actual plateau.

## 4. A common phase-free amplitude observation

For every marked block form `O=D_+-D_-`, `W=D_++D_--2*D_0` from its
same three corrected readings. Define the physical nominal-current
statistic with coefficients fixed before comparison:

\[
 A_c=\frac{2j_0 O_c}{\gamma h\sin\epsilon},\qquad
 B_c=\frac{2j_0 W_c}{\gamma h(1-\cos\epsilon)},\qquad
 m_c=\sqrt{A_c^2+B_c^2}.
 \tag{13}
\]

Neither candidate changes the raw readings, the input or this conversion.
There is no division by its tested amplitude and no reconstructed phase.
Equivalently compute the quadratures directly from nominal charge using
the nominal plateau duration h/b0. Gamma cancels from the resulting charge
conversion because `2*j0/(9*kappa0*gamma)=b0/2`; it also cancels from the
physical nominal waveform after substituting `kappa0` and the cubic f.
Thus choosing the command and observation does not fit the tested amplitude.
Since `q_nom=q/C` and nominal instrument seconds equal `D*t_lab`, the
reference amplitude in this statistic is `j_actual/(C*D)=mu*beta*j0`.
It is not the uncorrected physical current amplitude when C or D differs
from one.

Let Cc+iSc be the initial fine interface phasor. The initial residual
prior gives `cos(2*zeta)<=abs(Cc+iSc)<=1`. For each model,

\[
 |A_c/j_0-\mu\beta C_c|\le a_\mu=4b_\mu/(\gamma h\sin\epsilon),\qquad
 |B_c/j_0-\mu\beta S_c|\le b_\mu^*=8b_\mu/[\gamma h(1-\cos\epsilon)].
 \tag{14}
\]

The normalized bounds below are only a dimensionless way of displaying
the same measured current, not an amplitude fit. With
`r_mu=sqrt(a_mu^2+(b_mu*)^2)` and coefficient endpoints beta_minus/plus,

\[
 m_c/j_0\in I_\mu=
 [\max(0,\mu\beta_-\cos(2\zeta)-r_\mu),
                         \mu\beta_++r_\mu].
 \tag{15}
\]

Use outward sine, cosine and square-root enclosures. If the lower endpoint
of I2 exceeds the upper endpoint of I1, the complete 150-reading record
sets are disjoint: even one marked norm cannot belong to both intervals.
This is uniform over the admitted unknown phases, fine deviations and
correlated errors. It is a prospective discriminator, not an identification
of either law by an unevaluated experiment. Real records outside both
envelopes can invalidate premises or exclude both hypotheses; records
inside one envelope still need complete physical admission.

## 5. Storage and boundaries of physical interpretation

With fixed positive coefficients each model has its own storage

\[
 H_\mu=\tfrac12x^TL_{45}x+
          \frac{\mu\beta}{\eta}\sum_e(1-\cos\delta_e),\qquad
 H_\mu'=-\lambda\|L_{45}x\|_2^2/4
                +(L_{45}x)^TR(\rho f+d).
 \tag{16}
\]

This follows by cancellation using both rows. It is not the earlier
mu=1 storage when the coefficients change. Explicit finite bounds follow
without a pointwise tracking assumption. Set
`Zpeak=24*epsilon/(5*gamma_lower)`,
`Gpeak=48*epsilon/(5*gamma_lower)`,
`Xpeak=rho*Zpeak+M_mu`, `Gfull=rho*Gpeak+8*M_mu` and
`F1=56/(5*gamma_lower)`, an upper bound on the nominal input integral.
Then

\[
 \sup H_\mu\le180X_{peak}^2+180\mu\beta/\eta,\quad
 Loss\le45\lambda T G_{full}^2/4,\quad
 W_{abs}\le45G_{full}(\rho F_1+F).
 \tag{17}
\]

The last bound also covers the sum of positive powers over all fine
input channels integrated over time; no slot cancellation hides cost.
Use positive interval upper endpoints throughout. The earlier numerical
work ceilings are not inherited, and an L1 tracking bound does not supply
an actuator peak or bandwidth limit.
Preparation costs under the alternative and a physical fine-state energy
map remain separate. All inputs and their timing are supplied; no
autonomous force or event occurrence is inferred.

Constant scalar component ratios preserve the reference's exact shapes.
Parasitics, nonuniform junction variation, nonlinear capacitance, phase
noise and a time-dependent clock require additional terms and their own
admission. This first comparison changes one scalar current amplitude;
it is not a classification of possible laws or a proof that the sine
profile is uniquely selected. The [ontology](../EMERGENT_ONTOLOGY.md)
continues to separate mathematical organization from physical identity.

## 6. Execution and evidence boundary

The [pure bounds and raw statistic](../../src/tnfr/physics/_organized_robust_current.py)
admit primitive intervals, nonnegative errors and ordered readings using
shared owners. They rebuild all consumed fields; no cached margin or
source report certifies the present law. The fixed corner is reserved to
the [exclusive analytical driver](../../benchmarks/organized_robust_current.py).
The [instrument tests](../../tests/physics/test_organized_robust_current.py)
and [independent theory controls](../../tests/physics/test_organized_robust_current_theory.py)
use other synthetic inputs and algebraic counterexamples, without assessing
that corner or executing previous frozen producers.

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the result-dependent next gate. No physical observation or claim of
practical sensor capability follows from evaluating this sufficient bound.

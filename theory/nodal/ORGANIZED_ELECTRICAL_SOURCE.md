# Electrical description of the actual acquired collective source

<a id="organized-collective-electrical-source"></a>

## Prospective source and decision declaration

The [ideal electrical construction](ORGANIZED_ELECTRICAL_REALIZATION.md)
is exact on an invariant carrier. This comparison retains the actual fine
deviations of the already acquired source. It decides whether the ideal
collective circuit approximates that source uniformly on its original
contact horizon, with a justified circular observation and explicit hidden
storage and work. It supplies neither a new physical law nor a sensor.

| Item | Fixed choice and scope |
| --- | --- |
| Source | Both acquired outer-zero and outer-one five-child hosts, after the entire sequential observation word, and an independent recovered C9 donor. Keep each actual reached state |
| Primitive resources | The box declared at `0ca8630d`: twelve component allowances in `[0,1e-6]`, quarter formation 200, recovery `8e10`, recovered Euclidean allowance `q=1e-8`, decay power 80, sequential raw error `1e-14`, contact raw error `eta=1e-9` |
| Support | The original same-slot C9-by-W6 contact, node `9*c+j`, six marked children of nine nodes, 144 unit edges. No shifted registration |
| Complete rows | `x'=(-L*x+gamma*S(theta))/4`, `theta'=gamma*L*x/4`, `gamma=1/(1023*pi)`, held unit capacities, full-circle sine, no continuous input or event after contact |
| Clock | Prime is `tau=(1023/1024)*t`; fixed contact horizon `h=1e-3`. Physical `tau=tau0+b*t_lab` remains a free uncalibrated map |
| Observation | Actual signed child means and the circular strict-collar mean of the electrical owner; template `psi_j=2*pi*(j-4)/9`, retained donor origin `beta=pi/2`, charge reference `r=0`. Observation is unavailable outside its admitted collar |
| Comparator | The ideal six-island passive circuit, initialized at those actual collective means. No replacement by the nominal carrier, reset, inference from previous form readings or extra measurement is asserted |
| Transport | Orthogonal within-child projection; Laplacian upper bounds 8 during the complete observation word and 10 during contact; exact rational coupled integral bounds below |
| Decision | Require every actual history, support, law, work and identity premise, strict collar validity, positive comparison denominators and uniform collective form error strictly below the original `eta=1e-9` raw-reading allowance. Report phase, signed storage, hidden loss and work bounds separately |
| Evidence budget | One analytical assessment at the maximum resource-box corner, exact rational arithmetic and the existing outward dyadic128 trigonometric bounds. No trajectory, precision change, parameter search or frozen-producer replay |
| Unsupported result | A failed premise or sufficient bound leaves this comparison unavailable. It does not disprove the ideal carrier construction or authorize a new candidate |

Commit this declaration, consuming source and instrument before the single
assessment. The form-error threshold is an explicit resolution policy in
structural units. It does not identify a physical noise floor or constitute
an empirical comparison. Earlier frozen responses remain unchanged.

## 1. Actual coordinates, history and origins

Use the definitions and exact departure identities (27)-(33) of the
[electrical owner](ORGANIZED_ELECTRICAL_REALIZATION.md#collective-electrical-departure-identities).
Let T repeat each child coordinate over its nine slots, and put

\[
 P=TT^T/9,\qquad Q=I-P,\qquad
 u=Qx,\qquad z=Q(\theta-\psi),\qquad
 x=TX+u,\quad\theta=\psi+T\Phi+z.                 \tag{1}
\]

Here Q is an orthogonal projection, not electric charge. Continuous phase
lifts in (1) are selected by the prepared history and the collar justified
below; arbitrary principal representatives cannot be averaged. The template
has zero child mean. All host form and lifted-phase totals are zero, the
donor form total is zero and its lifted-phase total is `9*beta`. Consequently
`sum X=0`, `sum Phi=beta` and the common collective phase `beta/6` are retained
throughout contact. The source reference is `r=0`; a changed form origin
must carry its reference. The donor origin is not omitted from its phase.

The [outer-acquisition proof](ORGANIZED_OUTER_WINDING.md) provides global
Euclidean errors R and Z from a child-uniform carrier at the actual end of
formation. Since `||Q||=1`, these also bound `||u||` and `||z||` without the
factor two needed by a separate coordinatewise mean-subtraction estimate.
The same bounds apply to the paired outer-zero control: its nominal carrier
is stationary, while the global sine-Lipschitz proof uses the same support,
initial errors at most `3*q` and complete rows. It does not require the
nonzero nominal form ramp of the outer-one source.

Every event in the complete
[sequential observation word](ORGANIZED_SEQUENTIAL_OBSERVATION.md) is constant
within each child and leaves form unchanged. Therefore it leaves u and z
unchanged exactly. It changes Phi, and may cost work, but it supplies no
hidden-state reset. All continuous motion between the fifteen events is
retained over `T_o=1/5000`. Its raw observations keep the original error model.
The new phase observation is a function of complete state, not an assertion
that those twenty-five form readings reconstruct it.

## 2. Uniform transport of hidden deviations

On either product support, Q commutes with L. Internal helical sine currents
cancel, and same-slot inter-child currents are constant within each child on
the carrier. Thus `Q*S(psi+T*Phi)=0` for every Phi, including nonacute external
phase differences. The signed sine Hessian satisfies

\[
 \|S(\theta)-S(\widetilde\theta)\|_2
 \le\lambda\|\theta-\widetilde\theta\|_2,
 \qquad \|L\|_2\le\lambda.                           \tag{2}
\]

Indeed its quadratic form is an edge sum with cosine coefficients of
magnitude at most one, bounded in absolute value by the Laplacian form.
The complete projected rows and heat contraction give, for initial bounds
`a_0,z_0`, duration d and any `g>=gamma`,

\[
 \sup\|u\|_2\le a_0+s\sup\|z\|_2,\qquad
 \sup\|z\|_2\le z_0+s\sup\|u\|_2,\qquad
 s=g\lambda d/4.
\]

For `s<1` the resulting rational bounds are

\[
 A(a_0,z_0,s)=\frac{a_0+sz_0}{1-s^2},\qquad
 B(a_0,z_0,s)=\frac{z_0+sa_0}{1-s^2}.                 \tag{3}
\]

These are finite nonlinear bounds, not a tangent integration. Continuous
solutions exist globally on the fixed finite graph: the form row has
linear diffusion and bounded sine forcing, and the phase row is linear in
form. Uniform-child phase jumps preserve the projected norms, so the same
integral argument covers the entire piecewise continuous observation word.

For the observation graph, maximum degree four gives `lambda_o=8`. Set

\[
 s_o=2gT_o,\qquad A_o=A(R,Z,s_o),\quad B_o=B(R,Z,s_o).
\]

After adding the independent actual donor, the concatenated hidden norms
are at most `a_0=A_o+q`, `z_0=B_o+q`; this safe sum bounds the Euclidean
concatenation without equating source members. On the contact graph
`L=I_6 tensor L_C9 + L_W tensor I_9`, the C9 factor is at most 4I. Since W6
is a subgraph of K6, `L_W<=6P_6<=6I`, hence `lambda_c=10`. Finally put

\[
 s_c=5gh/2,\qquad A=A(a_0,z_0,s_c),\quad B=B(a_0,z_0,s_c).
                                                               \tag{4}
\]

Both source families and every allowed preparation error satisfy these
same uniform bounds. Their actual trajectories and collective means need
not be equal. Monotonicity of the positive rational expressions justifies
assessing the already declared maximum resource corner.

## 3. Admission of the circular collective observation

For each child, `|z_cj|<=B` and the deviations sum to zero. If `2B<pi/2`,
all its demodulated phasors lie in the arc of radius B about Phi. Their
resultant is nonzero and its argument differs from Phi by at most B.
Every wrapped residual about that argument therefore has magnitude at most
`2B<pi/2`. Averaging those consistently wrapped residuals as in (28) of the
electrical owner returns exactly Phi modulo a full turn and the same z.

The sufficient margin is

\[
 m_{\rm collar}=\pi_-/2-2B>0.                          \tag{5}
\]

This proves availability on every actual source throughout the declared
contact. It does not define a phase at zero resultant or extend the observer
outside its strict collar. Independent full turns of stored representatives
leave the circular observation unchanged. The detached
[observer](../../src/tnfr/physics/_organized_collective_observation.py)
encloses these coordinates from admitted primitive data; finite interval
arithmetic can return unavailable when it cannot establish the strict domain.

## 4. Closure and finite ideal-circuit error

Let `e_cd=A_cd-sin(Phi_d-Phi_c)` denote the exact sine-covariance departure.
The owner's quadratic bound and `L_W<=6I` give

\[
 \sum_{\{c,d\}}|e_{cd}|
 \le\frac1{18}z^T(L_W\mathbin\otimes I_9)z\le B^2/3,
 \qquad \|\mathcal R\|_1\le gB^2/6=:\rho.             \tag{6}
\]

Also `sum R=0`, so any sum over a union of whole children has defect at
most `rho/2`; in particular each node defect has that bound. The exact rows
remain

\[
 X'=(-L_WX+\gamma S_W(\Phi))/4+\mathcal R,\qquad
 \Phi'=\gamma L_WX/4.                                 \tag{7}
\]

The defect consumes hidden state. It is neither an independently supplied
external current source nor a reconstructed pressure chosen to fit a response.

Initialize the ideal closed wheel at the actual `X(0),Phi(0)`, with the
same origins and continuous lifts. Heat contraction, the global wheel
sine-Lipschitz constant six and `||R||_2<=rho` imply

\[
 E_X\le h\rho+c_WE_\Phi,\quad E_\Phi\le c_WE_X,
 \quad c_W=3gh/2,
 \quad E_X=\frac{h\rho}{1-c_W^2},\quad E_\Phi=c_WE_X.    \tag{8}
\]

Here E_X and E_Phi bound the suprema of Euclidean differences over the
entire contact, not merely the endpoint. The corresponding circular phase
distance is bounded by E_Phi; display-cut jumps are immaterial. Each
collective charge differs by at most `9*kappa*E_X`; arbitrary unions of
children retain their sums and the zero total. The resolution decision
is `eta-E_X>0`, for the original structural raw-form allowance eta. It
does not say the error is zero, or that the physical scales are calibrated.

## 5. Storage, loss, regional transfer and supplied work

The signed departure identity and `||L_54||<=10` give

\[
 -5B^2\le R_H\le5(A^2+B^2),\qquad
 0\le\tfrac14\|L_{54}u\|^2\le25A^2.                  \tag{9}
\]

Retain `H_54=H_int+9H_W+R_H` with the original internal constant; R_H is
not a nonnegative reservoir. The integrated hidden loss is at most
`25*h*A^2`. If M is the previously justified actual fine form bound on
contact, then `||L_W X||_infty<=10M`. Equations (6)-(7) give

\[
 |(L_WX)^T\mathcal R|\le10M\rho,\qquad
 \left|\int_0^h(L_WX)^T\mathcal R\,d\tau\right|
 \le10Mh\rho.                                        \tag{10}
\]

These bounds do not prove that the actual collective storage is
nonincreasing: its derivative includes this signed hidden-work term.
The full state still has the exact nonnegative loss from (33) of the
electrical owner, while `R_H'=-||L_54*u||^2/4-9*(L_W*X)^T R`.

For the reference circuit scales `K=a_0*b*kappa*gamma/4`, actual collective
electrical storage is `9K*H_W`. Its endogenous power correction is
`9K*b*(L_W*X)^T R`, and the integrated correction is bounded by `9K`
times (10). Its charge equation has current defect `9*b*kappa*R`. For
each union of children, the exact regional charge balance retains that
defect, bounded by `9*b*kappa*rho/2`. Electrical capacitor/resistor wiring
need not equal fine TNFR incidence or assign the same individual wire work.

A supplied neutral uniform-child form input f has exact work pairing even
off the carrier:

\[
 (L_{54}x)^TTf=9(L_WX)^Tf,\qquad
 I_{\rm ext}=9b\kappa f,\quad
 V_W^TI_{\rm ext}=9Kb(L_WX)^Tf.                        \tag{11}
\]

This follows from orthogonality and `L_54*T=T*L_W`. It adds no input to
the present unforced contact; a driven comparison needs its own transport
bounds. Nonuniform input and phase forcing are outside (11).

The proof begins immediately after the supplied contact support has been
installed. Changing that support also changes collective capacitances and
conductances, potentially far from the added edge. The old TNFR attachment
budget below 100 and joined storage below 200 remain source premises; they
are not an electrical switching payment, component law or occurrence
certificate. Neither hidden loss nor signed hidden storage supplies one.

## 6. Implementation and evidence boundary

The [source assessor](../../src/tnfr/physics/_organized_electrical_source.py)
freshly reconstructs primitive resources, requires six recoveries and every
nonresponse history margin, and checks the product support, complete contact
law and observation word. It does not reuse the former response verdict or
take an incoming cached certificate. The
[instrument](../../benchmarks/organized_electrical_source.py) uses the shared
exclusive analytical lifecycle, preserving source inventory and exact results.

The [transport and admission controls](../../tests/physics/test_organized_electrical_source.py)
use independent coupled-majorant solutions, invalid-domain controls and
synthetic earlier source histories. The
[circular observation controls](../../tests/physics/test_organized_collective_observation.py)
exercise its distinct availability and representation boundary. Neither
test owner reruns the reserved assessment or frozen response producers.

This is a conditional finite bridge to the ideal electrical reference. Its
Josephson relations and physical constants are supplied by that reference;
TNFR has not derived elementary charge, microscopic constituents or quantum
behavior. An equivalence cannot empirically discriminate against the very
same conventional circuit. Independent sensors, laboratory clock,
preparation and physical constitutive evidence remain separate obligations.

<a id="source-electrical-admission-result"></a>
## 7. Frozen source-inclusive result

Declaration and consuming source were committed at
`10bf3e1003d2bb864d8bbbb9f2e2900957923c39` before the exclusive assessment.
The [attempt](../../docs/assets/organized_electrical_source/attempt.json)
records that source inventory and budget; the
[result](../../docs/assets/organized_electrical_source/result.json) retains
the exact report. Its verdict is `finite_collective_electrical_source_admitted`.
All eleven prerequisite flags and twenty-five history margins pass. This
is one analytical assessment, without a trajectory or physical experiment.

The following displayed bounds are rounded conservatively; the evidence
retains exact fractions and outward trigonometric endpoints.

| Quantity, uniform unless integrated | Certified bound in declared structural units |
| --- | --- |
| Hidden form norm A | `<4.05e-8` |
| Hidden phase norm B | `<7.01e-8` |
| Strict circular collar margin | `>1.57079618` radians |
| Collective closure defect, l1 norm | `<2.55e-19` |
| Ideal-comparator form error, Euclidean norm | `<2.55e-22` |
| Ideal-comparator phase error, consistent-lift Euclidean norm | `<1.19e-28` radians |
| Signed hidden storage R_H | `-2.46e-14 < R_H < 3.27e-14` |
| Integrated hidden loss | `<4.10e-17` |
| Absolute integrated collective hidden-work correction | `<2.55e-23` |

The form-error bound is below the unchanged `1e-9` raw-reading allowance.
It bounds every member of both original acquired source families on
`0<=tau<=1e-3`, when the ideal comparator starts at that member's actual
collective means. It is not a measured error or a relative error guarantee;
the phase and storage bounds are not extra laboratory observations. Smaller
actual reading errors do not change the declared resolution threshold.

This closes the finite source-departure obligation: internal deviations do
not obstruct this particular conditional collective electrical description.
The hidden term remains present, and exact aggregate passivity, all-future
maintenance, source switching and empirical identification remain unproved.
The [read-only evidence controls](../../tests/physics/test_organized_electrical_source_evidence.py)
reconstruct the arithmetic and source associations without rerunning the
assessment. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next dependency.

# Native pattern reduction and memory

Sections 1–8 retain the native single-bridge two-C5 state, hidden initialization
and conditional approximation bounds. Each chapter below declares its complete
law, support and preparation. Native neighbor-argument dynamics and the
normalized-sine comparison share methods, not interchangeable trajectories
or certificates.

<a id="reading-map-by-model-and-question"></a>

## Chapters and scope

| Chapter | Responsibility |
| --- | --- |
| [Native mediated pattern dynamics](RELATIONAL_MEDIATOR_DYNAMICS.md) | One-, two- and three-port reduction, fast-limit premises, memory clocks and causal response. |
| [Sine environmental state and causal memory](SINE_ENVIRONMENTAL_MEMORY.md) | Cancellation, conserved hidden inventory, exact elimination and conditional state/capacity inference. |
| [Return-path geometry and collective response](RELATIONAL_RETURN_PATH_GEOMETRY.md) | Native equilibrium and oscillatory response; separately declared sine/cubic storage geometry. |
| [Retained phase records and finite contact readout](RELATIONAL_PHASE_MEMORY.md) | Isolated recovery, phase records, contact/removal budgets and retained receiver means. |
| [Sine relative state, recovery and conservative identity](SINE_PATTERN_RECOVERY.md) | Exact relative state, whole-support recovery and separate zero-loss protection and recurrence. |
| [Eleven-node receiver formation bounds](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md) | Preparation-specific directional loss and formation obstructions. |
| [Eleven-node transfer conditions and endpoint capture](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md) | Port work, passage conditions, equilibrium classification and donor-well capture. |

Exact elimination retains an independent hidden initial-state source; finite
conservative memory does not become irreversible damping. Whole-support
recovery and formation claims must retain the live environment. Related
[sine formation results](SINE_PATTERN_DYNAMICS.md#reading-map) and
[replica scale states](SINE_PAIR_STATE.md#sine-replica-inheritance) keep their
own complete-law hypotheses.

Section numbers remain stable. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone assigns research work; retained case studies do not activate campaigns.

## 1. Exact symmetry quotient: ten even and eight odd coordinates

Fix the existing single-bridge two-C5 graph, held unit capacities,
`e>=0`, `w,beta>0`, uniform reference form zero and winding-one reference
phase `theta_*`. Let `z=(u,v)` consist of form and local lifted phase
deviations. All statements concern a neighborhood in the strictly acute
domain, where the supplied full vector field `F` is smooth. Put
`kappa=2*pi/5`, `c=cos(kappa)>0`, and `k=w/beta`.

For either ten-node species `b`, retain the same five even observations:

\[
C_5b=\left(m_L-m_R,\ b_0-m_L,\ b_5-m_R,\
\frac{b_1+b_4-b_2-b_3}{2},\
\frac{b_6+b_9-b_7-b_8}{2}\right).
\]

The missing odd observations are

\[
P_4b=\left(\frac{b_1-b_4}{2},\frac{b_2-b_3}{2},\
\frac{b_6-b_9}{2},\frac{b_7-b_8}{2}\right).
\]

Set `C=diag(C_5,C_5)`, `P=diag(P_4,P_4)` and

\[
y=Cz\in\mathbb R^{10},\qquad h=Pz\in\mathbb R^8.
\]

Use the natural even lift `T_y` from the
[composition theorem](RELATIONAL_PATTERN_COMPOSITION.md#3-ten-coordinates-are-necessary-and-sufficient-at-first-variation):
the two means are opposite and each ring's values are even under reflection
through its port. The odd lift `T_h` places each ring's pair `(a,b)` at
`(0,a,b,-b,-a)`, separately for form and phase. These maps satisfy

\[
CT_y=I_{10},\quad PT_h=I_8,\quad CT_h=0,\quad PT_y=0,\qquad
T_yC+T_hP=\Pi_0,
\]

where `Pi_0` removes the arithmetic global mean of each species. Thus
`z=common_offsets+T_y*y+T_h*h` exactly. No internal coordinates have been
discarded in this eighteen-dimensional representation.

Common form and phase shifts are exact symmetries of this unforced fixed
model: `F` depends on neither offset. Their means may drift; that fact does
not invalidate the quotient. In a zero-mean gauge its vector field is
`Pi_0*F`, and the two projected rows are exactly

\[
\dot y=f_y(y,h):=CF(T_yy+T_hh),\qquad
\dot h=f_h(y,h):=PF(T_yy+T_hh).
\]

Recovering absolute frames additionally requires integrating their offset
rates. This quotient predicts the stated relative observations, without
claiming that the discarded offsets are conserved.

## 2. Linear hidden modes and the appropriate norm

The proved equilibrium Jacobian commutes with ring reflections, so the
linearized quotient is block diagonal. Write its even block as `G` and
its odd block as `A_h`. For

\[
M=\begin{pmatrix}2&-1\\-1&3\end{pmatrix},\qquad M_4=\operatorname{diag}(M,M),
\]

and hidden order `(h_x,h_theta)`, the exact block is

\[
A_h=\begin{pmatrix}
-(e/2)M_4&-[w/(2\pi)]M_4\\
[w/(2\beta\pi c)]M_4&0
\end{pmatrix}.
\]

The two eigenvalues of `M` are `(5-sqrt(5))/2` and `(5+sqrt(5))/2`.
Each spatial mode has generator
`mu*[[-e/2,-w/(2*pi)],[w/(2*beta*pi*c),0]]`. With `e>0` its two
eigenvalues have negative real parts; this is a conditional linear damping
statement, not an additional restoring law. The finite-horizon result below
needs only contraction and also allows `e=0`.

Let `B` be the unit graph Laplacian and `K_*` the phase Hessian at the
lock. The quadratic storage norm uses

\[
S_*=\tfrac12\operatorname{diag}(B,\beta K_*),\qquad
\|y\|_Y^2=(T_yy)^TS_*(T_yy),\quad
\|h\|_H^2=(T_hh)^TS_*(T_hh).
\]

Both are positive definite on their coordinate spaces. The two subspaces
are orthogonal in this metric, and specifically

\[
\|h\|_H^2=h_x^TM_4h_x+\beta c\,h_\theta^TM_4h_\theta.
\]

The full quadratic storage derivative is
`-e*(B*u)^T*D^-1*(B*u)<=0`; restriction to the two invariant subspaces
therefore proves

\[
\|e^{Gt}\|_Y\le1,\qquad \|e^{A_ht}\|_H\le1\qquad(t\ge0).
\]

These are norms of the declared linearization. They do not assert that
the same quadratic approximation decreases along every nonlinear trajectory
or that the storage is a measured physical energy.

## 3. Exact hidden memory retains its initial condition

Define the nonlinear remainders by the actual quotient field:

\[
\dot y=Gy+R_y(y,h),\qquad \dot h=A_hh+R_h(y,h).
\]

On every admitted existence interval, variation of constants gives

\[
\boxed{\quad
h(t)=e^{A_ht}h_0+
\int_0^t e^{A_h(t-s)}R_h(y(s),h(s))\,ds.
\quad}
\]

This reuses the elimination principle of
[derived EPI memory, section 3](../DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state),
including its indispensable initial hidden-state term. Here the source is
nonlinear and still depends on hidden history. Substituting this identity
into the visible row does not turn it into a known linear convolution of
`y`, nor does it remove the need to determine that history. Without an
additional estimate or reduction it is an exact rewrite of the full quotient.

The joint form/phase generator is not the self-adjoint pure-diffusion
generator used by that earlier owner's positivity theorem. No positive
memory kernel, finite REMESH history or fixed memory cutoff is inherited.
In particular the
[equal-state-and-rate counterexample](RELATIONAL_PATTERN_COMPOSITION.md#state-rate-predictivity)
has different `h_0` and different subsequent visible accelerations. Discarding
that initial information cannot predict both preparations.

The existing memory studies supply distinct reusable arguments, not one
interchangeable runtime mechanism:

| Existing owner | Reused argument and boundary |
| --- | --- |
| [Derived EPI memory, section 3](../DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state) | Retain both hidden initialization and generated hidden evolution; the current source is nonlinear. |
| [Derived EPI memory, sections 8.2--8.3](../DERIVED_EPI_MEMORY.md#82-exact-reference-and-omitted-forcing) | Propagate omitted forcing to obtain trajectory error. The executable [P5 truncation reference](../../src/tnfr/physics/p5_memory_truncation.py) is restricted to its stated path, partition and common-capacity model. |
| [Derived EPI memory, sections 9.4 and 10.1](../DERIVED_EPI_MEMORY.md) | Separate visible/hidden quadratic storage and distinguish projected autonomy from lift invariance; the present joint metric and parity require their own derivation. |
| [Derived form phase, section 21.1](DERIVED_FORM_PHASE.md#211-directed-form-rotation-and-exact-eliminated-state-memory) | Its exact negative kernel already excludes universal memory-kernel positivity; its directed diffusion adapter is not this joint law. |
| [Cycle memory relaxation](../CYCLE_MEMORY_RELAXATION.md) | Configured delayed REMESH execution is distinct from eliminating current hidden coordinates; no finite delay is derived here. |
| [Coupling winding persistence, section 51](../COUPLING_WINDING_PERSISTENCE.md) | Carried C6 return-state correlations prevent loss of discrete constraints; they supply neither this continuous kernel nor a new active C6 campaign. |

## 4. Symmetry supplies a finite-horizon approximation theorem

Simultaneously invert form and phase and reflect both rings. This preserves
the chosen lock, modulo its fixed phase representatives. In quotient
coordinates the transformation is `(y,h)->(-y,h)`, and exact equivariance
implies

\[
f_y(-y,h)=-f_y(y,h),\qquad f_h(-y,h)=f_h(y,h).
\]

Thus `f_y(0,h)=0`, while `f_y(y,0)` has no quadratic Taylor term. Choose
a closed ball `Y+H<=r` contained in the smooth admitted neighborhood, where
`Y=||y||_Y` and `H=||h||_H`. Mixed differentiation and Taylor's theorem give
nonnegative constants `A,B,D`, written below as `mathsf A,B,D` to distinguish
them from graph matrices, such that

\[
\|R_y(y,h)\|_Y\le\mathsf A\,YH+\mathsf B\,Y^3,\qquad
\|R_h(y,h)\|_H\le\mathsf D(Y^2+H^2).
\]

For example, `mathsf A` can bound the mixed derivative `D_h D_y f_y` on
the ball, `mathsf B` can be one sixth of a bound for `D_y^3 f_y(y,0)`, and
`mathsf D` can bound the second derivative of `f_h` in the product norm
`Y+H`. The mixed bound follows by integrating from both `y=0` and `h=0`;
the hidden Taylor remainder uses `(Y+H)^2/2<=Y^2+H^2`. These constants
come from the supplied field's derivatives, not calibration or extra model
parameters. Their finite existence on this compact regular domain is proved;
their numerical values and a numerical radius have not been computed here.

Take `h_0=0`, `eta=||y_0||_Y`, and a fixed finite horizon `T` in the
declared structural clock. Put

\[
\mathcal K=\mathsf D+\mathsf A/4+\mathsf B r.
\]

Assume the explicit smallness conditions

\[
2\eta<r,\qquad \mathcal K\eta T\le\tfrac12.
\]

Applying both contractive semigroups in Duhamel's formula and using
`YH<=(Y+H)^2/4` gives, up to any first exit from the ball,

\[
Y(t)+H(t)\le\eta+\mathcal K\int_0^t[Y(s)+H(s)]^2\,ds
\le\frac{\eta}{1-\mathcal K\eta t}\le2\eta.
\]

The comparison inequality and `2*eta<r` exclude such an exit before `T`.
The hidden integral then yields `H(t)<=4*mathsf D*t*eta^2`. For the
visible linear approximation `y_lin(t)=exp(G*t)*y_0`, integrate its actual
remainder rather than identifying a force residual with trajectory error:

\[
\boxed{\quad
\|y(t)-y_{\rm lin}(t)\|_Y
\le (4\mathsf A\mathsf D T^2+8\mathsf B T)\eta^3,
\qquad 0\le t\le T.
\quad}
\]

This distinction reuses the methodological boundary in
[derived EPI memory, sections 8.2--8.3](../DERIVED_EPI_MEMORY.md#82-exact-reference-and-omitted-forcing):
a bound on an omitted rate must be propagated through the evolution to bound
the response. The diffusion-specific positive resolvent from that example
is not invoked here.

For a general initial hidden state of the same small order as `y_0`, the
`YH` term permits a quadratic, rather than cubic, visible correction.
Pure odd preparations can still have `y=0` exactly, so this is an allowed
generic order, not a lower bound for every preparation. The cubic theorem
requires the specified `h_0=0` condition and a fixed admitted finite horizon;
it does not provide an unrestricted long-time error or a runtime certificate.

## 5. Zero initial hidden state is not an invariant preparation

The distinction between projected prediction and lift invariance also appears
in [derived EPI memory, section 10.1](../DERIVED_EPI_MEMORY.md#101-projected-autonomy-and-affine-lift-invariance-differ).
Here there is a direct exact witness. Let `eta_L` be the even form shape
from the composition owner, set `x=epsilon*eta_L`, and choose the even phase
deviation `v=sigma*e_0`, with `epsilon>0` and `0<sigma<pi/10`. Both
species have `h=0`. The symbol `sigma` is a preparation amplitude, not time.

The neighboring form gradients satisfy `q_1=q_4=3*epsilon`, but

\[
H_1=2\pi\cos(\kappa-\sigma/2)\operatorname{sinc}(\sigma/2),\qquad
H_4=2\pi\cos(\kappa+\sigma/2)\operatorname{sinc}(\sigma/2).
\]

Hence the odd near-phase coordinate has the exact nonzero rate

\[
\frac{\dot\theta_1-\dot\theta_4}{2}
=-\frac{3k\epsilon\sin\kappa\,\sigma}
{4\pi[c^2-\sin^2(\sigma/2)]}.
\]

This is a quadratic hidden source at small amplitudes, even though `h_0=0`.
At the donor port write `h_*=1+2c`. Its resultant is `h_* exp(-i*sigma)`,
so `H_0=pi*h_*sinc(sigma)`. The difference between its exact phase rate
and the linear prediction is

\[
-\frac{2k\epsilon}{\pi h_*}
\left[\frac1{\operatorname{sinc}(\sigma)}-1\right]
=-\frac{k\epsilon\sigma^2}{3\pi h_*}+O(\epsilon\sigma^4).
\]

All right phase rates and their linear predictions vanish. This difference
is therefore also present in the retained port-minus-right-mean phase
observable. It exhibits a nonzero cubic rate correction while the hidden
source is already quadratic. Neither the error order nor the prepared
linear approximation asserts nonlinear invariance of the even lift.

As a related storage consequence, the quadratic form splits orthogonally
into visible and hidden parts. The exact declared storage also obeys
`E(-y,h)=E(y,h)` and has zero gradient at the lock. Its allowed cubic
Taylor terms have types `y^2*h` and `h^3`. Under the preceding prepared
finite-horizon bounds, hidden quadratic storage is `O(eta^4)`, and
`E(y(t),h(t))-E_* - E_2(y_lin(t),0)=O(eta^4)`, where `E_2` is the
quadratic Taylor storage. This is a conditional Taylor consequence for the
supplied storage, not measured energy conservation or another evolution law.

## 6. A prospective cubic memory forecast from the same law

<a id="derived-cubic-memory-forecast"></a>

The preceding result permits a more precise, still conditional forecast:
derive the first hidden response and its return to the visible state before
evaluating a reserved nonlinear response. This section supplies the analytic
coefficients; it contains no evaluated trajectory or numerical verdict.

### Neighbor sums determine the Taylor coefficients

At the ideal lock put `delta_*ij=theta_*j-theta_*i` and
`rho_i=sum_j cos(delta_*ij)>0`. Thus `rho_i=1+2c` at ports and `2c`
elsewhere. For a full deviation `z=(u,v)`, write `d_ij=v_j-v_i` and
`q=Bu`. At each node define the homogeneous neighbor sums

\[
a_1=-\frac{\sum_j\sin\delta_{*ij}\,d_{ij}}{\rho_i},\qquad
a_2=-\frac{\sum_j\cos\delta_{*ij}\,d_{ij}^2}{2\rho_i},
\]
\[
b_1=\frac{\sum_j\cos\delta_{*ij}\,d_{ij}}{\rho_i},\qquad
b_2=-\frac{\sum_j\sin\delta_{*ij}\,d_{ij}^2}{2\rho_i},\qquad
b_3=-\frac{\sum_j\cos\delta_{*ij}\,d_{ij}^3}{6\rho_i}.
\]

For relative resultant `X_i+iY_i`, these are its normalized real and
imaginary expansions. Its argument is
`b_1+alpha_2+alpha_3+O(||v||^4)`, where

\[
\alpha_2=b_2-a_1b_1,\qquad
\alpha_3=b_3-a_1b_2+(a_1^2-a_2)b_1-b_1^3/3.
\]

The exact identity `H_i^-1=atan(Y_i/X_i)/(pi*Y_i)`, continuously extended
at `Y_i=0`, gives the inverse-metric expansion

\[
H_i^{-1}=\frac1{\pi\rho_i}
\left[1-a_1+a_1^2-a_2-b_1^2/3+O(\|v\|^3)\right].
\]

Consequently the full field has homogeneous maps
`F(z)=Jz+Q_2(z)+Q_3(z)+O(||z||^4)`, with

\[
(Q_2)_x=\frac w\pi\alpha_2,\qquad
(Q_2)_\theta=-\frac{kq_i a_1}{\pi\rho_i},
\]
\[
(Q_3)_x=\frac w\pi\alpha_3,\qquad
(Q_3)_\theta=\frac{kq_i}{\pi\rho_i}
\left(a_1^2-a_2-b_1^2/3\right).
\]

These coefficients come from the existing pressure argument and phase metric.
No finite-difference fit or new constitutive coefficient is involved. Define
the quadratic cross term without an implicit factor convention:

\[
\mathcal B_2(z,z')=Q_2(z+z')-Q_2(z)-Q_2(z').
\]

### The first hidden contribution gives a triangular prediction

For the prepared family `z_0=epsilon*(eta_L,e_0)`, let
`a=C*(eta_L,e_0)` and define coefficient functions by

\[
\dot y_1=Gy_1,\qquad y_1(0)=a,
\]
\[
\dot h_2=A_hh_2+P Q_2(T_yy_1),\qquad h_2(0)=0,
\]
\[
\dot y_3=Gy_3+C\left[
\mathcal B_2(T_yy_1,T_hh_2)+Q_3(T_yy_1)\right],\qquad y_3(0)=0.
\]

The even lift of `a` differs from `(eta_L,e_0)` only by a common phase
offset, which these maps do not consume. The system is triangular: the
linear visible evolution generates the quadratic hidden response, which
returns through the derived cross term. Its initial hidden phase source is
the section 5 coefficient `-3*k*sin(kappa)/(4*pi*c^2)` at the left near
pair. These equations approximate an existing state; they do not prescribe
an additional microscopic memory process.

Smooth dependence on initial amplitude on a common admitted finite interval,
together with exact parity, gives

\[
y(t;\epsilon)=\epsilon y_1(t)+\epsilon^3y_3(t)+O(\epsilon^5),\qquad
h(t;\epsilon)=\epsilon^2h_2(t)+O(\epsilon^4).
\]

Indeed `y(t;-epsilon)=-y(t;epsilon)` and
`h(t;-epsilon)=h(t;epsilon)`, so the intervening even or odd powers vanish.
The smooth local field supplies the required parameter derivatives; the
remainder constants and common amplitude domain are not numerically bounded
here. This asymptotic order is not a finite-amplitude error certificate.

A direct cubic comparison omits only the derived memory cross term:
`y_3,direct'=G*y_3,direct+C*Q_3(T_y*y_1)`, with zero initial value.
For `d=y_3-y_3,direct` and the retained phase observable
`ell*y=mu_v+p_L_v`, the exact initial controls are

\[
d(0)=\dot d(0)=0,\qquad
\ell\ddot d(0)=
\frac{3k^2\sin^2\kappa}{\pi^2c^2(1+2c)^2}>0.
\]

This follows directly from
`ell*C*B_2((eta_L,e_0),T_h*P*Q_2((eta_L,e_0)))`.
It proves a nonzero memory contribution before evaluating a trajectory.
It does not, by itself, decide its sign or accuracy at the reserved finite
endpoint. The direct cubic and memory predictions both retain coefficients
derived before that response is examined.

### A matched Euler comparison keeps numerical and model claims separate

The prospective preparation fixes `e=w=1/2`, `beta=1`, unit capacities,
`epsilon=sigma=1/64` and structural horizon `T=1/4`, with 64, 128 and 256
steps. Detailed protocol, provenance and eventual verdict belong to the
retained study record, not a second task list here.

For a declared explicit-Euler grid, let `Y_n=T_y*y_1,n` and
`H_n=T_h*h_2,n`. Its matching amplitude-coefficient recurrence is

\[
\begin{pmatrix}y_{1,n+1}\\h_{2,n+1}\\y_{3,n+1}\end{pmatrix}
=\begin{pmatrix}y_{1,n}\\h_{2,n}\\y_{3,n}\end{pmatrix}
+\Delta t\begin{pmatrix}
Gy_{1,n}\\
A_hh_{2,n}+P Q_2(Y_n)\\
Gy_{3,n}+C[\mathcal B_2(Y_n,H_n)+Q_3(Y_n)]
\end{pmatrix}.
\]

Every right-hand side uses the same old coefficient state. Updating `h_2`
before using it in the `y_3` row would change the Taylor coefficients of the
declared simultaneous Euler map. The direct cubic comparison uses the same
grid and initial data and omits only the cross term.

For a fixed admitted number of steps these are the Taylor coefficients of
the ideal Euler composition, not exact continuous-flow samples. That ideal
map retains amplitude parity, while the native execution separately
materializes trigonometric values, initial phases, pressure and rates.
Its rounded reference need not be an exact lock, and arithmetic residuals
must not be relabeled as high-order model error. Agreement on multiple
grids or with higher-precision evaluation can inform this finite comparison;
it is not an enclosure of the exact ODE trajectory or a certified error bound.

<a id="reserved-memory-response"></a>
## 7. Reserved finite memory response

The [coefficient instrument](../../benchmarks/relational_memory_prediction.py)
evaluates the preceding polynomials analytically and advances their triangular
system through the shared Euler arithmetic. The
[response runner](../../benchmarks/relational_memory_response.py) uses the full
production `step_relational_exchange`, without adding a second nonlinear
solver. The [prediction](../../docs/assets/relational_memory_response/result.prediction.json)
and [package-source archive](../../docs/assets/relational_memory_response/result.sources.zip)
were frozen before the [reserved response](../../docs/assets/relational_memory_response/result.json).
All TNFR Python sources and the three producer/helper files are archived with
their recorded hashes; external dependency versions are recorded separately.

The single preparation is exactly that specified in section 6, on the
single-bridge pair of C5 rings. The endpoint observable is all ten visible
coordinates in the quadratic-storage norm `||.||_Y`. The frozen comparison
requires the memory error to be at most one tenth of **each** control's
error plus `1e-12`, with both control errors larger than `100*1e-12`.
The direct cubic control retains `Q_3` and omits only hidden feedback.
Additional gates require resolved hidden motion with at most 5% prediction
error, the held support/capacity and native pressure path, acute margin at
least `pi/20`, and small observed work-balance/evaluation residuals. These
are experiment decision policies, not universal TNFR constants.

All gates passed, without changing the preparation, predictions or cutoffs:

| Steps to `T=1/4` | Linear visible error | Direct cubic error | Memory cubic error |
| --- | --- | --- | --- |
| 64 | `1.4976770e-6` | `5.9376594e-8` | `1.0112225e-9` |
| 128 | `1.4978459e-6` | `5.9785753e-8` | `1.0124092e-9` |
| 256 | `1.4979305e-6` | `5.9990003e-8` | `1.0130042e-9` |

At 256 steps the memory prediction reduces endpoint error by about 1,479
relative to the tangent prediction and 59.2 relative to the direct cubic
control. The latter comparison isolates the benefit of generated hidden
motion; the entire tangent-to-cubic gain cannot be assigned to memory alone.
Hidden second-order error is about `0.0682%` of the hidden response norm.
The minimum observed acute margin is `0.2985343` and maximum absolute
actual work-balance residual is `4.80e-18`.

The run used Python 3.13.6, NumPy 2.5.3, NetworkX 3.6.1 on Windows AMD64,
binary64 native states, ascending nodes `0,...,9`, sorted unit edges and no
randomness. Decimal50 re-evaluation of the same represented coefficient
constants differs from binary64 by at most `1.10e-17` in the compared entries.
The rounded reference lock has visible rate norm `3.23e-17`; it was recorded
without subtracting that drift from the evaluated trajectory. Projection and
norm-square arithmetic is exact **for the materialized matrix coefficients**:
binary64 `1/5` is not exact rational `1/5`. Independent edge/mean accounting
in the retained-record tests bounds this representation difference separately.

The raw native endpoint difference between 128 and 256 steps is `1.63616e-6`,
much larger than the same-grid memory error. The corresponding predictor
difference is also `1.63616e-6`. This is why the comparison matches Euler grids:
it tests an amplitude approximation of the same finite executor, not
`1e-9` accuracy for the continuous ODE. Step arithmetic defects, storage-step
defects and grid differences are retained separately; none is a certified
continuous trajectory error bar. Neither this one amplitude nor its three
numerical grids establishes an empirical fifth-order scaling law.

The finite prediction therefore supplies a useful consequence of the existing
law: generated internal differences improve prediction of the visible pattern,
without a fitted memory kernel. It does not supply a smaller autonomous state,
a runtime speedup, a new primitive field or physical identification. The
bounded approximation gate is complete; the sole execution plan owns subsequent
research rather than extending this into an accuracy or derivative campaign.

## 8. Result and stopping boundary

The existing [composition instrument](../../benchmarks/relational_local_composition.py)
now exposes `analyze_local_memory`, reusing its fixed support, invariant-row
algebra and exact matrix/semidefinite helpers. It retains both projections and
lifts, the two tangent generators, quadratic energy and dissipation matrices,
and checked reconstruction identities. The matrices use the energy convention
above, including the factor one half. Its rational coefficient family is
explicitly distinct from the actual irrational phase lock; it does not
evaluate a nonlinear memory integral or certify the derivative constants.

[Static controls](../../tests/physics/test_relational_pattern_memory.py) check
independent graph-energy sums, the hidden block, zero-loss admission and
native inversion/reflection parity. The prepared-even witness uses
`epsilon=1/64`, `sigma=1/128`, `e=w=1/2`, `beta=1`, unit capacities, fixed
node order `0,...,9` and binary64 native field evaluation. It compares the
displayed exact hidden and visible-rate formulas with the captured rates;
no trajectory is integrated. These numerical tolerances check implementation,
not the finite-horizon error theorem. The existing state-and-rate witness
also verifies that the hidden initialization survives the complete split.

The benefit beyond an exact memory rewrite is the proved cubic visible
error for an explicitly prepared, sufficiently small state over a fixed
finite horizon. Hidden coordinates remain real parts of the original state;
their initial contribution and nonlinear generation have not disappeared.
No numerical constant, radius, cutoff or empirical observation is supplied
as a substitute for the theorem's admission conditions.

The full nonlinear engine state remains authoritative. No compressed
nonlinear SDK, fitted memory kernel, derivative ladder or new dynamical law
is installed by this result. A prospective finite numerical comparison is
permitted with declared inputs and arithmetic scope; it does not supply the
uncomputed constants of an ODE error certificate. A certified numerical
reduction would additionally need admitted derivative bounds and domain
control, and another memory closure would need its own predictive
justification. This scoped result supplies neither autonomous fractal
composition nor an identification with physical constituents.

## Section link directory

These aliases route existing citations to their substantive owner.

- <a id="pattern-memory-and-mediated-collective-geometry"></a>[Pattern memory and mediated collective geometry](#pattern-memory-and-mediated-collective-geometry)

- <a id="mediated-pattern-interaction"></a>[mediated-pattern-interaction](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction)

- <a id="9-a-nodal-intermediary-mediates-joint-formphase-interaction"></a>[9. A nodal intermediary mediates joint form/phase interaction](RELATIONAL_MEDIATOR_DYNAMICS.md#9-a-nodal-intermediary-mediates-joint-formphase-interaction)

- <a id="full-state-and-conditional-recovery"></a>[Full state and conditional recovery](RELATIONAL_MEDIATOR_DYNAMICS.md#full-state-and-conditional-recovery)

- <a id="derived-memory-with-two-indispensable-hidden-coordinates"></a>[Derived memory, with two indispensable hidden coordinates](RELATIONAL_MEDIATOR_DYNAMICS.md#derived-memory-with-two-indispensable-hidden-coordinates)

- <a id="mediator-pressure-boundary-chart"></a>[mediator-pressure-boundary-chart](RELATIONAL_MEDIATOR_DYNAMICS.md#mediator-pressure-boundary-chart)

- <a id="an-exact-nonlinear-pressure-chart-retains-the-moving-boundary"></a>[An exact nonlinear pressure chart retains the moving boundary](RELATIONAL_MEDIATOR_DYNAMICS.md#an-exact-nonlinear-pressure-chart-retains-the-moving-boundary)

- <a id="fast-mediator-reduction"></a>[fast-mediator-reduction](RELATIONAL_MEDIATOR_DYNAMICS.md#fast-mediator-reduction)

- <a id="a-controlled-nonlinear-reduction-when-the-mediator-is-fast"></a>[A controlled nonlinear reduction when the mediator is fast](RELATIONAL_MEDIATOR_DYNAMICS.md#a-controlled-nonlinear-reduction-when-the-mediator-is-fast)

- <a id="two-mediator-composition"></a>[two-mediator-composition](RELATIONAL_MEDIATOR_DYNAMICS.md#two-mediator-composition)

- <a id="two-fast-intermediaries-compose-through-their-inherited-interface"></a>[Two fast intermediaries compose through their inherited interface](RELATIONAL_MEDIATOR_DYNAMICS.md#two-fast-intermediaries-compose-through-their-inherited-interface)

- <a id="three-port-collective-interaction"></a>[three-port-collective-interaction](RELATIONAL_MEDIATOR_DYNAMICS.md#three-port-collective-interaction)

- <a id="a-branching-intermediary-induces-a-collective-three-port-interaction"></a>[A branching intermediary induces a collective three-port interaction](RELATIONAL_MEDIATOR_DYNAMICS.md#a-branching-intermediary-induces-a-collective-three-port-interaction)

- <a id="sine-mediator-common-geometry"></a>[sine-mediator-common-geometry](SINE_ENVIRONMENTAL_MEMORY.md#sine-mediator-common-geometry)

- <a id="common-mediator-geometry-does-not-select-its-transmitted-pressure"></a>[Common mediator geometry does not select its transmitted pressure](SINE_ENVIRONMENTAL_MEMORY.md#common-mediator-geometry-does-not-select-its-transmitted-pressure)

- <a id="finite-environment-reuse-audit"></a>[finite-environment-reuse-audit](SINE_ENVIRONMENTAL_MEMORY.md#finite-environment-reuse-audit)

- <a id="finite-environmental-state-cancellation-and-causal-response-reuse"></a>[Finite environmental state, cancellation and causal-response reuse](SINE_ENVIRONMENTAL_MEMORY.md#finite-environmental-state-cancellation-and-causal-response-reuse)

- <a id="a-faithful-state-that-remains-regular-at-cancellation"></a>[A faithful state that remains regular at cancellation](SINE_ENVIRONMENTAL_MEMORY.md#a-faithful-state-that-remains-regular-at-cancellation)

- <a id="static-agreement-hides-different-memory-clocks"></a>[Static agreement hides different memory clocks](SINE_ENVIRONMENTAL_MEMORY.md#static-agreement-hides-different-memory-clocks)

- <a id="reuse-the-signed-memory-owner-not-diffusion-only-shortcuts"></a>[Reuse the signed memory owner, not diffusion-only shortcuts](SINE_ENVIRONMENTAL_MEMORY.md#reuse-the-signed-memory-owner-not-diffusion-only-shortcuts)

- <a id="conserved-hidden-inventory"></a>[conserved-hidden-inventory](SINE_ENVIRONMENTAL_MEMORY.md#conserved-hidden-inventory)

- <a id="conserved-inventory-constrains-both-memory-and-its-initial-source"></a>[Conserved inventory constrains both memory and its initial source](SINE_ENVIRONMENTAL_MEMORY.md#conserved-inventory-constrains-both-memory-and-its-initial-source)

- <a id="autonomous-path-cancellation"></a>[autonomous-path-cancellation](SINE_ENVIRONMENTAL_MEMORY.md#autonomous-path-cancellation)

- <a id="an-autonomous-three-node-crossing-closes-the-moving-environment-obstruction"></a>[An autonomous three-node crossing closes the moving-environment obstruction](SINE_ENVIRONMENTAL_MEMORY.md#an-autonomous-three-node-crossing-closes-the-moving-environment-obstruction)

- <a id="one-rational-preparation-and-an-explicit-finite-crossing-bound"></a>[One rational preparation and an explicit finite crossing bound](SINE_ENVIRONMENTAL_MEMORY.md#one-rational-preparation-and-an-explicit-finite-crossing-bound)

- <a id="transient-cancellation-not-persistent-formation"></a>[Transient cancellation, not persistent formation](SINE_ENVIRONMENTAL_MEMORY.md#transient-cancellation-not-persistent-formation)

- <a id="causal-sine-environmental-pressure"></a>[causal-sine-environmental-pressure](SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure)

- <a id="exact-causal-environmental-pressure-with-retained-internal-state"></a>[Exact causal environmental pressure with retained internal state](SINE_ENVIRONMENTAL_MEMORY.md#exact-causal-environmental-pressure-with-retained-internal-state)

- <a id="exact-interface-and-a-sufficient-hidden-state"></a>[Exact interface and a sufficient hidden state](SINE_ENVIRONMENTAL_MEMORY.md#exact-interface-and-a-sufficient-hidden-state)

- <a id="derivative-free-nonlinear-memory"></a>[Derivative-free nonlinear memory](SINE_ENVIRONMENTAL_MEMORY.md#derivative-free-nonlinear-memory)

- <a id="work-must-be-accounted-for-across-the-interface"></a>[Work must be accounted for across the interface](SINE_ENVIRONMENTAL_MEMORY.md#work-must-be-accounted-for-across-the-interface)

- <a id="a-non-reflected-initial-state-discriminator"></a>[A non-reflected initial-state discriminator](SINE_ENVIRONMENTAL_MEMORY.md#a-non-reflected-initial-state-discriminator)

- <a id="shared-implementation-and-evidence-scope"></a>[Shared implementation and evidence scope](SINE_ENVIRONMENTAL_MEMORY.md#shared-implementation-and-evidence-scope)

- <a id="sine-hidden-state-observability"></a>[sine-hidden-state-observability](SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-state-observability)

- <a id="prior-visible-observations-and-hidden-state-observability"></a>[Prior visible observations and hidden-state observability](SINE_ENVIRONMENTAL_MEMORY.md#prior-visible-observations-and-hidden-state-observability)

- <a id="two-observed-rows-remove-the-diffusive-contribution"></a>[Two observed rows remove the diffusive contribution](SINE_ENVIRONMENTAL_MEMORY.md#two-observed-rows-remove-the-diffusive-contribution)

- <a id="rank-consistency-and-an-exceptional-unique-branch"></a>[Rank, consistency and an exceptional unique branch](SINE_ENVIRONMENTAL_MEMORY.md#rank-consistency-and-an-exceptional-unique-branch)

- <a id="observation-geometry-differs-from-the-phase-resultant"></a>[Observation geometry differs from the phase resultant](SINE_ENVIRONMENTAL_MEMORY.md#observation-geometry-differs-from-the-phase-resultant)

- <a id="what-instantaneous-observation-still-cannot-identify"></a>[What instantaneous observation still cannot identify](SINE_ENVIRONMENTAL_MEMORY.md#what-instantaneous-observation-still-cannot-identify)

- <a id="sound-bounded-admission-and-shared-implementation"></a>[Sound bounded admission and shared implementation](SINE_ENVIRONMENTAL_MEMORY.md#sound-bounded-admission-and-shared-implementation)

- <a id="sine-hidden-capacity-observability"></a>[sine-hidden-capacity-observability](SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-capacity-observability)

- <a id="prior-acceleration-identifies-capacity-only-through-an-active-hidden-response"></a>[Prior acceleration identifies capacity only through an active hidden response](SINE_ENVIRONMENTAL_MEMORY.md#prior-acceleration-identifies-capacity-only-through-an-active-hidden-response)

- <a id="a-shared-tangent-chain-rule"></a>[A shared tangent chain rule](SINE_ENVIRONMENTAL_MEMORY.md#a-shared-tangent-chain-rule)

- <a id="a-second-cancellation-improves-the-inference"></a>[A second cancellation improves the inference](SINE_ENVIRONMENTAL_MEMORY.md#a-second-cancellation-improves-the-inference)

- <a id="capacity-information-can-survive-phase-ambiguity"></a>[Capacity information can survive phase ambiguity](SINE_ENVIRONMENTAL_MEMORY.md#capacity-information-can-survive-phase-ambiguity)

- <a id="informative-and-blind-preparations"></a>[Informative and blind preparations](SINE_ENVIRONMENTAL_MEMORY.md#informative-and-blind-preparations)

- <a id="bounded-prior-admission-and-its-limits"></a>[Bounded prior admission and its limits](SINE_ENVIRONMENTAL_MEMORY.md#bounded-prior-admission-and-its-limits)

- <a id="sine-prior-reserved-forecast"></a>[sine-prior-reserved-forecast](SINE_ENVIRONMENTAL_MEMORY.md#sine-prior-reserved-forecast)

- <a id="a-finite-forecast-from-jointly-admitted-prior-environmental-evidence"></a>[A finite forecast from jointly admitted prior environmental evidence](SINE_ENVIRONMENTAL_MEMORY.md#a-finite-forecast-from-jointly-admitted-prior-environmental-evidence)

- <a id="joint-admission-and-a-regular-circular-coordinate"></a>[Joint admission and a regular circular coordinate](SINE_ENVIRONMENTAL_MEMORY.md#joint-admission-and-a-regular-circular-coordinate)

- <a id="a-prepared-source-and-an-explicit-counterfactual"></a>[A prepared source and an explicit counterfactual](SINE_ENVIRONMENTAL_MEMORY.md#a-prepared-source-and-an-explicit-counterfactual)

- <a id="an-analytic-separation-before-any-response-is-evaluated"></a>[An analytic separation before any response is evaluated](SINE_ENVIRONMENTAL_MEMORY.md#an-analytic-separation-before-any-response-is-evaluated)

- <a id="retained-prospective-software-response"></a>[Retained prospective software response](SINE_ENVIRONMENTAL_MEMORY.md#retained-prospective-software-response)

- <a id="sine-finite-sample-admission"></a>[sine-finite-sample-admission](SINE_ENVIRONMENTAL_MEMORY.md#sine-finite-sample-admission)

- <a id="finite-samples-derivative-errors-and-the-remaining-observation-boundary"></a>[Finite samples, derivative errors and the remaining observation boundary](SINE_ENVIRONMENTAL_MEMORY.md#finite-samples-derivative-errors-and-the-remaining-observation-boundary)

- <a id="one-time-one-stencil-and-separate-error-sources"></a>[One time, one stencil and separate error sources](SINE_ENVIRONMENTAL_MEMORY.md#one-time-one-stencil-and-separate-error-sources)

- <a id="smoothness-from-an-independently-declared-nodal-class"></a>[Smoothness from an independently declared nodal class](SINE_ENVIRONMENTAL_MEMORY.md#smoothness-from-an-independently-declared-nodal-class)

- <a id="a-fixed-prospective-derivative-budget"></a>[A fixed prospective derivative budget](SINE_ENVIRONMENTAL_MEMORY.md#a-fixed-prospective-derivative-budget)

- <a id="boundaries-that-prevent-a-false-observation-claim"></a>[Boundaries that prevent a false observation claim](SINE_ENVIRONMENTAL_MEMORY.md#boundaries-that-prevent-a-false-observation-claim)

- <a id="capacity-sets-the-memory-clock-and-the-kernel-need-not-be-positive"></a>[Capacity sets the memory clock, and the kernel need not be positive](RELATIONAL_MEDIATOR_DYNAMICS.md#capacity-sets-the-memory-clock-and-the-kernel-need-not-be-positive)

- <a id="a-prospective-onset-discriminator-in-the-actual-nonlinear-field"></a>[A prospective onset discriminator in the actual nonlinear field](RELATIONAL_MEDIATOR_DYNAMICS.md#a-prospective-onset-discriminator-in-the-actual-nonlinear-field)

- <a id="shared-implementation-and-evidence-scope-1"></a>[Shared implementation and evidence scope](RELATIONAL_MEDIATOR_DYNAMICS.md#shared-implementation-and-evidence-scope-1)

- <a id="mediator-orientation-scope"></a>[mediator-orientation-scope](RELATIONAL_MEDIATOR_DYNAMICS.md#mediator-orientation-scope)

- <a id="orientation-sensitivity-winding-sign-is-invisible-at-a-single-symmetric-port"></a>[Orientation sensitivity: winding sign is invisible at a single symmetric port](RELATIONAL_MEDIATOR_DYNAMICS.md#orientation-sensitivity-winding-sign-is-invisible-at-a-single-symmetric-port)

- <a id="finite-mediated-response"></a>[finite-mediated-response](RELATIONAL_MEDIATOR_DYNAMICS.md#finite-mediated-response)

- <a id="10-a-finite-causal-response-test-of-the-effective-connection"></a>[10. A finite causal-response test of the effective connection](RELATIONAL_MEDIATOR_DYNAMICS.md#10-a-finite-causal-response-test-of-the-effective-connection)

- <a id="influence-has-an-onset-order-not-a-selected-activation-time"></a>[Influence has an onset order, not a selected activation time](RELATIONAL_MEDIATOR_DYNAMICS.md#influence-has-an-onset-order-not-a-selected-activation-time)

- <a id="a-frozen-mediator-can-organize-both-rings-without-coupling-their-changes"></a>[A frozen mediator can organize both rings without coupling their changes](RELATIONAL_MEDIATOR_DYNAMICS.md#a-frozen-mediator-can-organize-both-rings-without-coupling-their-changes)

- <a id="a-symmetric-port-prediction-has-no-quadratic-amplitude-error"></a>[A symmetric port prediction has no quadratic amplitude error](RELATIONAL_MEDIATOR_DYNAMICS.md#a-symmetric-port-prediction-has-no-quadratic-amplitude-error)

- <a id="frozen-finite-executor-protocol"></a>[Frozen finite-executor protocol](RELATIONAL_MEDIATOR_DYNAMICS.md#frozen-finite-executor-protocol)

- <a id="reserved-response-and-decision"></a>[Reserved response and decision](RELATIONAL_MEDIATOR_DYNAMICS.md#reserved-response-and-decision)

- <a id="mediated-geometric-boundary"></a>[mediated-geometric-boundary](RELATIONAL_RETURN_PATH_GEOMETRY.md#mediated-geometric-boundary)

- <a id="11-return-path-geometry-and-the-formation-boundary"></a>[11. Return-path geometry and the formation boundary](RELATIONAL_RETURN_PATH_GEOMETRY.md#11-return-path-geometry-and-the-formation-boundary)

- <a id="return-path-equilibrium"></a>[return-path-equilibrium](RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-equilibrium)

- <a id="111-complete-state-support-and-equilibrium-equations"></a>[11.1 Complete state, support and equilibrium equations](RELATIONAL_RETURN_PATH_GEOMETRY.md#111-complete-state-support-and-equilibrium-equations)

- <a id="112-existence-uniqueness-and-the-forced-zero-mixed-period"></a>[11.2 Existence, uniqueness and the forced zero mixed period](RELATIONAL_RETURN_PATH_GEOMETRY.md#112-existence-uniqueness-and-the-forced-zero-mixed-period)

- <a id="113-recovery-observable-distinction-and-storage-cost"></a>[11.3 Recovery, observable distinction and storage cost](RELATIONAL_RETURN_PATH_GEOMETRY.md#113-recovery-observable-distinction-and-storage-cost)

- <a id="114-static-computational-contract"></a>[11.4 Static computational contract](RELATIONAL_RETURN_PATH_GEOMETRY.md#114-static-computational-contract)

- <a id="return-path-storage-dependence"></a>[return-path-storage-dependence](RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-storage-dependence)

- <a id="115-coupled-cycle-geometry-changes-with-admitted-phase-storage"></a>[11.5 Coupled-cycle geometry changes with admitted phase storage](RELATIONAL_RETURN_PATH_GEOMETRY.md#115-coupled-cycle-geometry-changes-with-admitted-phase-storage)

- <a id="the-scalar-equation-describes-every-acute-critical-state-in-this-sector"></a>[The scalar equation describes every acute critical state in this sector](RELATIONAL_RETURN_PATH_GEOMETRY.md#the-scalar-equation-describes-every-acute-critical-state-in-this-sector)

- <a id="existence-uniqueness-and-constitutive-dependence-for-every-finite-coefficient"></a>[Existence, uniqueness and constitutive dependence for every finite coefficient](RELATIONAL_RETURN_PATH_GEOMETRY.md#existence-uniqueness-and-constitutive-dependence-for-every-finite-coefficient)

- <a id="reconstruction-identity-and-local-protection"></a>[Reconstruction, identity and local protection](RELATIONAL_RETURN_PATH_GEOMETRY.md#reconstruction-identity-and-local-protection)

- <a id="return-path-geometry-response"></a>[return-path-geometry-response](RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-geometry-response)

- <a id="116-a-geometric-interval-constrains-a-law-and-a-separate-nodal-response"></a>[11.6 A geometric interval constrains a law and a separate nodal response](RELATIONAL_RETURN_PATH_GEOMETRY.md#116-a-geometric-interval-constrains-a-law-and-a-separate-nodal-response)

- <a id="the-inverse-exists-on-a-proper-half-open-geometric-branch"></a>[The inverse exists on a proper, half-open geometric branch](RELATIONAL_RETURN_PATH_GEOMETRY.md#the-inverse-exists-on-a-proper-half-open-geometric-branch)

- <a id="finite-geometric-accuracy-does-not-imply-uniform-coefficient-accuracy"></a>[Finite geometric accuracy does not imply uniform coefficient accuracy](RELATIONAL_RETURN_PATH_GEOMETRY.md#finite-geometric-accuracy-does-not-imply-uniform-coefficient-accuracy)

- <a id="the-full-nodal-response-probes-curvature-rather-than-only-current-balance"></a>[The full nodal response probes curvature rather than only current balance](RELATIONAL_RETURN_PATH_GEOMETRY.md#the-full-nodal-response-probes-curvature-rather-than-only-current-balance)

- <a id="outward-bounds-retain-the-same-implicit-geometry-and-law"></a>[Outward bounds retain the same implicit geometry and law](RELATIONAL_RETURN_PATH_GEOMETRY.md#outward-bounds-retain-the-same-implicit-geometry-and-law)

- <a id="geometry-does-not-determine-the-clock-or-the-missing-dynamical-scales"></a>[Geometry does not determine the clock or the missing dynamical scales](RELATIONAL_RETURN_PATH_GEOMETRY.md#geometry-does-not-determine-the-clock-or-the-missing-dynamical-scales)

- <a id="frozen-known-source-control"></a>[Frozen known-source control](RELATIONAL_RETURN_PATH_GEOMETRY.md#frozen-known-source-control)

- <a id="shared-collective-pulse"></a>[shared-collective-pulse](RELATIONAL_RETURN_PATH_GEOMETRY.md#shared-collective-pulse)

- <a id="12-a-causally-shared-oscillatory-response-with-a-dissipation-limit"></a>[12. A causally shared oscillatory response, with a dissipation limit](RELATIONAL_RETURN_PATH_GEOMETRY.md#12-a-causally-shared-oscillatory-response-with-a-dissipation-limit)

- <a id="121-the-oscillatory-mechanism-uses-the-existing-two-rows"></a>[12.1 The oscillatory mechanism uses the existing two rows](RELATIONAL_RETURN_PATH_GEOMETRY.md#121-the-oscillatory-mechanism-uses-the-existing-two-rows)

- <a id="122-a-nonzero-transfer-is-stronger-than-a-shared-eigenvector"></a>[12.2 A nonzero transfer is stronger than a shared eigenvector](RELATIONAL_RETURN_PATH_GEOMETRY.md#122-a-nonzero-transfer-is-stronger-than-a-shared-eigenvector)

- <a id="123-certified-existence-of-an-ideal-damped-complex-mode"></a>[12.3 Certified existence of an ideal damped complex mode](RELATIONAL_RETURN_PATH_GEOMETRY.md#123-certified-existence-of-an-ideal-damped-complex-mode)

- <a id="124-persistence-is-a-separate-claim"></a>[12.4 Persistence is a separate claim](RELATIONAL_RETURN_PATH_GEOMETRY.md#124-persistence-is-a-separate-claim)

- <a id="125-retained-static-evidence-and-numerical-boundaries"></a>[12.5 Retained static evidence and numerical boundaries](RELATIONAL_RETURN_PATH_GEOMETRY.md#125-retained-static-evidence-and-numerical-boundaries)

- <a id="13-a-retained-collective-phase-offset-after-isolated-pattern-recovery"></a>[13. A retained collective phase offset after isolated-pattern recovery](RELATIONAL_PHASE_MEMORY.md#13-a-retained-collective-phase-offset-after-isolated-pattern-recovery)

- <a id="relational-retained-phase-memory"></a>[relational-retained-phase-memory](RELATIONAL_PHASE_MEMORY.md#relational-retained-phase-memory)

- <a id="state-reference-and-exact-mean-identities"></a>[State, reference and exact mean identities](RELATIONAL_PHASE_MEMORY.md#state-reference-and-exact-mean-identities)

- <a id="the-first-nonzero-retained-offset"></a>[The first nonzero retained offset](RELATIONAL_PHASE_MEMORY.md#the-first-nonzero-retained-offset)

- <a id="asymptotic-origin-coordinate-scope"></a>[asymptotic-origin-coordinate-scope](RELATIONAL_PHASE_MEMORY.md#asymptotic-origin-coordinate-scope)

- <a id="two-equal-mean-preparations-with-opposite-retained-offsets"></a>[Two equal-mean preparations with opposite retained offsets](RELATIONAL_PHASE_MEMORY.md#two-equal-mean-preparations-with-opposite-retained-offsets)

- <a id="a-signed-operational-readout-separate-from-attachment-work"></a>[A signed operational readout, separate from attachment work](RELATIONAL_PHASE_MEMORY.md#a-signed-operational-readout-separate-from-attachment-work)

- <a id="14-a-finite-amplitude-retained-phase-certificate-without-a-trajectory"></a>[14. A finite-amplitude retained-phase certificate without a trajectory](RELATIONAL_PHASE_MEMORY.md#14-a-finite-amplitude-retained-phase-certificate-without-a-trajectory)

- <a id="relational-finite-phase-memory"></a>[relational-finite-phase-memory](RELATIONAL_PHASE_MEMORY.md#relational-finite-phase-memory)

- <a id="storage-controls-the-state-and-the-integrated-form-response"></a>[Storage controls the state and the integrated form response](RELATIONAL_PHASE_MEMORY.md#storage-controls-the-state-and-the-integrated-form-response)

- <a id="a-cross-term-controls-the-integrated-phase-deformation"></a>[A cross term controls the integrated phase deformation](RELATIONAL_PHASE_MEMORY.md#a-cross-term-controls-the-integrated-phase-deformation)

- <a id="an-integrated-cubic-remainder-for-the-approximate-phase-invariant"></a>[An integrated cubic remainder for the approximate phase invariant](RELATIONAL_PHASE_MEMORY.md#an-integrated-cubic-remainder-for-the-approximate-phase-invariant)

- <a id="one-declared-nonzero-amplitude-and-its-signed-readout"></a>[One declared nonzero amplitude and its signed readout](RELATIONAL_PHASE_MEMORY.md#one-declared-nonzero-amplitude-and-its-signed-readout)

- <a id="15-a-finite-time-contact-readout-with-residual-state-and-event-budgets"></a>[15. A finite-time contact readout with residual-state and event budgets](RELATIONAL_PHASE_MEMORY.md#15-a-finite-time-contact-readout-with-residual-state-and-event-budgets)

- <a id="relational-finite-time-memory"></a>[relational-finite-time-memory](RELATIONAL_PHASE_MEMORY.md#relational-finite-time-memory)

- <a id="an-explicit-decay-estimate-on-the-already-admitted-ball"></a>[An explicit decay estimate on the already admitted ball](RELATIONAL_PHASE_MEMORY.md#an-explicit-decay-estimate-on-the-already-admitted-ball)

- <a id="a-phase-tail-bound-in-the-retained-reference-frame"></a>[A phase-tail bound in the retained reference frame](RELATIONAL_PHASE_MEMORY.md#a-phase-tail-bound-in-the-retained-reference-frame)

- <a id="fresh-contact-rates-including-the-no-contact-comparison"></a>[Fresh contact rates, including the no-contact comparison](RELATIONAL_PHASE_MEMORY.md#fresh-contact-rates-including-the-no-contact-comparison)

- <a id="one-finite-horizon-and-its-separate-event-work-budget"></a>[One finite horizon and its separate event-work budget](RELATIONAL_PHASE_MEMORY.md#one-finite-horizon-and-its-separate-event-work-budget)

- <a id="16-a-finite-contact-transmits-a-signed-form-response"></a>[16. A finite contact transmits a signed form response](RELATIONAL_PHASE_MEMORY.md#16-a-finite-contact-transmits-a-signed-form-response)

- <a id="relational-finite-contact-memory"></a>[relational-finite-contact-memory](RELATIONAL_PHASE_MEMORY.md#relational-finite-contact-memory)

- <a id="a-state-scaled-continuous-flow-bound"></a>[A state-scaled continuous-flow bound](RELATIONAL_PHASE_MEMORY.md#a-state-scaled-continuous-flow-bound)

- <a id="one-duration-with-accumulated-contact-and-no-contact-responses"></a>[One duration with accumulated contact and no-contact responses](RELATIONAL_PHASE_MEMORY.md#one-duration-with-accumulated-contact-and-no-contact-responses)

- <a id="event-work-and-continuous-loss-retain-different-accounts"></a>[Event work and continuous loss retain different accounts](RELATIONAL_PHASE_MEMORY.md#event-work-and-continuous-loss-retain-different-accounts)

- <a id="17-a-receiver-mean-survives-contact-removal-and-isolated-recovery"></a>[17. A receiver mean survives contact removal and isolated recovery](RELATIONAL_PHASE_MEMORY.md#17-a-receiver-mean-survives-contact-removal-and-isolated-recovery)

- <a id="relational-retained-receiver-record"></a>[relational-retained-receiver-record](RELATIONAL_PHASE_MEMORY.md#relational-retained-receiver-record)

- <a id="bound-the-regional-mean-before-invoking-conservation"></a>[Bound the regional mean before invoking conservation](RELATIONAL_PHASE_MEMORY.md#bound-the-regional-mean-before-invoking-conservation)

- <a id="the-cut-has-its-own-storage-jump"></a>[The cut has its own storage jump](RELATIONAL_PHASE_MEMORY.md#the-cut-has-its-own-storage-jump)

- <a id="both-isolated-rings-remain-in-their-recovery-basins"></a>[Both isolated rings remain in their recovery basins](RELATIONAL_PHASE_MEMORY.md#both-isolated-rings-remain-in-their-recovery-basins)

- <a id="the-retained-mean-is-exactly-conserved-during-that-recovery"></a>[The retained mean is exactly conserved during that recovery](RELATIONAL_PHASE_MEMORY.md#the-retained-mean-is-exactly-conserved-during-that-recovery)

- <a id="18-relative-pattern-state-without-discarding-internal-dynamics"></a>[18. Relative pattern state without discarding internal dynamics](SINE_PATTERN_RECOVERY.md#18-relative-pattern-state-without-discarding-internal-dynamics)

- <a id="sine-relative-pattern-state"></a>[sine-relative-pattern-state](SINE_PATTERN_RECOVERY.md#sine-relative-pattern-state)

- <a id="an-exact-quotient-by-two-common-origins"></a>[An exact quotient by two common origins](SINE_PATTERN_RECOVERY.md#an-exact-quotient-by-two-common-origins)

- <a id="the-same-complete-law-has-a-dissipative-hamiltonian-representation"></a>[The same complete law has a dissipative Hamiltonian representation](SINE_PATTERN_RECOVERY.md#the-same-complete-law-has-a-dissipative-hamiltonian-representation)

- <a id="reconstruction-from-conserved-means-when-every-capacity-is-positive"></a>[Reconstruction from conserved means when every capacity is positive](SINE_PATTERN_RECOVERY.md#reconstruction-from-conserved-means-when-every-capacity-is-positive)

- <a id="what-observation-uncertainty-cancels-and-what-remains"></a>[What observation uncertainty cancels, and what remains](SINE_PATTERN_RECOVERY.md#what-observation-uncertainty-cancels-and-what-remains)

- <a id="winding-is-retained-information-not-an-unconditional-invariant"></a>[Winding is retained information, not an unconditional invariant](SINE_PATTERN_RECOVERY.md#winding-is-retained-information-not-an-unconditional-invariant)

- <a id="a-whole-set-sufficient-recovery-criterion-under-the-sine-law"></a>[A whole-set sufficient recovery criterion under the sine law](SINE_PATTERN_RECOVERY.md#a-whole-set-sufficient-recovery-criterion-under-the-sine-law)

- <a id="sine-cycle-recovery"></a>[sine-cycle-recovery](SINE_PATTERN_RECOVERY.md#sine-cycle-recovery)

- <a id="computable-whole-set-recovery-for-an-exact-cycle-twist"></a>[Computable whole-set recovery for an exact cycle twist](SINE_PATTERN_RECOVERY.md#computable-whole-set-recovery-for-an-exact-cycle-twist)

- <a id="chart-choice-and-cancellation-before-interval-evaluation"></a>[Chart choice and cancellation before interval evaluation](SINE_PATTERN_RECOVERY.md#chart-choice-and-cancellation-before-interval-evaluation)

- <a id="quadratic-excess-storage-without-subtracting-twist-energy"></a>[Quadratic excess storage without subtracting twist energy](SINE_PATTERN_RECOVERY.md#quadratic-excess-storage-without-subtracting-twist-energy)

- <a id="original-observation-sets-and-propagated-boxes-are-different-inputs"></a>[Original observation sets and propagated boxes are different inputs](SINE_PATTERN_RECOVERY.md#original-observation-sets-and-propagated-boxes-are-different-inputs)

- <a id="an-explicitly-deformed-c5-observation-class-that-passes"></a>[An explicitly deformed C5 observation class that passes](SINE_PATTERN_RECOVERY.md#an-explicitly-deformed-c5-observation-class-that-passes)

- <a id="19-recovering-patterns-with-a-live-retained-intermediary"></a>[19. Recovering patterns with a live retained intermediary](SINE_PATTERN_RECOVERY.md#19-recovering-patterns-with-a-live-retained-intermediary)

- <a id="sine-interacting-recovery"></a>[sine-interacting-recovery](SINE_PATTERN_RECOVERY.md#sine-interacting-recovery)

- <a id="an-exact-critical-target-on-the-full-eleven-node-support"></a>[An exact critical target on the full eleven-node support](SINE_PATTERN_RECOVERY.md#an-exact-critical-target-on-the-full-eleven-node-support)

- <a id="an-independent-full-graph-spectral-gap-bound"></a>[An independent full-graph spectral-gap bound](SINE_PATTERN_RECOVERY.md#an-independent-full-graph-spectral-gap-bound)

- <a id="a-finite-uncertain-class-within-the-interacting-basin"></a>[A finite uncertain class within the interacting basin](SINE_PATTERN_RECOVERY.md#a-finite-uncertain-class-within-the-interacting-basin)

- <a id="exact-nonlinear-donor--intermediary--receiver-onset"></a>[Exact nonlinear donor--intermediary--receiver onset](SINE_PATTERN_RECOVERY.md#exact-nonlinear-donor--intermediary--receiver-onset)

- <a id="influence-followed-by-recovery-and-a-retained-common-form-record"></a>[Influence followed by recovery and a retained common-form record](SINE_PATTERN_RECOVERY.md#influence-followed-by-recovery-and-a-retained-common-form-record)

- <a id="a-symmetry-obstruction-to-forming-the-receiver-twist"></a>[A symmetry obstruction to forming the receiver twist](SINE_PATTERN_RECOVERY.md#a-symmetry-obstruction-to-forming-the-receiver-twist)

- <a id="sine-formation-eligibility"></a>[sine-formation-eligibility](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-formation-eligibility)

- <a id="20-formation-eligibility-and-a-finite-time-exclusion"></a>[20. Formation eligibility and a finite-time exclusion](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#20-formation-eligibility-and-a-finite-time-exclusion)

- <a id="exact-odd-response-and-its-controls"></a>[Exact odd response and its controls](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#exact-odd-response-and-its-controls)

- <a id="storage-is-necessary-but-target-storage-is-not-the-entry-barrier"></a>[Storage is necessary, but target storage is not the entry barrier](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#storage-is-necessary-but-target-storage-is-not-the-entry-barrier)

- <a id="an-analytic-time-window-can-exclude-an-apparently-eligible-preparation"></a>[An analytic time window can exclude an apparently eligible preparation](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#an-analytic-time-window-can-exclude-an-apparently-eligible-preparation)

- <a id="a-uniform-exclusion-for-the-bounded-localized-pulse-family"></a>[A uniform exclusion for the bounded localized-pulse family](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#a-uniform-exclusion-for-the-bounded-localized-pulse-family)

- <a id="what-this-excludes-and-what-it-leaves-open"></a>[What this excludes and what it leaves open](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#what-this-excludes-and-what-it-leaves-open)

- <a id="sine-balanced-formation"></a>[sine-balanced-formation](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-balanced-formation)

- <a id="21-a-balanced-preparation-and-a-directional-loss-obstruction"></a>[21. A balanced preparation and a directional loss obstruction](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#21-a-balanced-preparation-and-a-directional-loss-obstruction)

- <a id="a-bound-that-retains-the-initial-direction"></a>[A bound that retains the initial direction](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#a-bound-that-retains-the-initial-direction)

- <a id="a-rational-finite-window-certificate"></a>[A rational finite-window certificate](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#a-rational-finite-window-certificate)

- <a id="exact-moments-for-the-balanced-eleven-node-family"></a>[Exact moments for the balanced eleven-node family](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#exact-moments-for-the-balanced-eleven-node-family)

- <a id="the-same-bounded-amplitude-family-is-still-excluded"></a>[The same bounded amplitude family is still excluded](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#the-same-bounded-amplitude-family-is-still-excluded)

- <a id="corollary-the-whole-constant-donor-preparation-family"></a>[Corollary: the whole constant-donor preparation family](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#corollary-the-whole-constant-donor-preparation-family)

- <a id="sine-internal-form-geometry"></a>[sine-internal-form-geometry](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-internal-form-geometry)

- <a id="22-donor-shape-phase-action-and-a-uniform-formation-obstruction"></a>[22. Donor shape, phase action and a uniform formation obstruction](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#22-donor-shape-phase-action-and-a-uniform-formation-obstruction)

- <a id="the-exact-six-coordinate-quadratic-forms"></a>[The exact six-coordinate quadratic forms](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#the-exact-six-coordinate-quadratic-forms)

- <a id="which-donor-information-reaches-the-receiver-first"></a>[Which donor information reaches the receiver first](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#which-donor-information-reaches-the-receiver-first)

- <a id="sine-source-receiver-excitation"></a>[sine-source-receiver-excitation](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-source-receiver-excitation)

- <a id="source-symmetry-nonlinear-work-and-a-uniform-tangent-obstruction"></a>[Source symmetry, nonlinear work and a uniform tangent obstruction](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#source-symmetry-nonlinear-work-and-a-uniform-tangent-obstruction)

- <a id="equal-energy-and-spectral-moments-do-not-determine-receiver-work"></a>[Equal energy and spectral moments do not determine receiver work](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#equal-energy-and-spectral-moments-do-not-determine-receiver-work)

- <a id="every-communicating-tangent-response-stays-below-the-receiver-barrier"></a>[Every communicating tangent response stays below the receiver barrier](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#every-communicating-tangent-response-stays-below-the-receiver-barrier)

- <a id="a-full-nonlinear-storage-bound-forces-the-donor-barrier-first"></a>[A full nonlinear storage bound forces the donor barrier first](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#a-full-nonlinear-storage-bound-forces-the-donor-barrier-first)

- <a id="a-receiver-passage-needs-a-nonzero-quantified-nonlinear-contribution"></a>[A receiver passage needs a nonzero, quantified nonlinear contribution](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#a-receiver-passage-needs-a-nonzero-quantified-nonlinear-contribution)

- <a id="sine-weighted-receiver-exclusion"></a>[sine-weighted-receiver-exclusion](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-weighted-receiver-exclusion)

- <a id="a-weighted-nonlinear-proof-function-excludes-receiver-identity-below-a-source-threshold"></a>[A weighted nonlinear proof function excludes receiver identity below a source threshold](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#a-weighted-nonlinear-proof-function-excludes-receiver-identity-below-a-source-threshold)

- <a id="one-nonuniform-donor-preparation-at-the-same-storage-budget"></a>[One nonuniform donor preparation at the same storage budget](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#one-nonuniform-donor-preparation-at-the-same-storage-budget)

- <a id="a-class-level-lower-bound-on-any-target-entry-time"></a>[A class-level lower bound on any target-entry time](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#a-class-level-lower-bound-on-any-target-entry-time)

- <a id="sine-maintained-target-obstruction"></a>[sine-maintained-target-obstruction](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-maintained-target-obstruction)

- <a id="a-global-auxiliary-function-excludes-the-maintained-target-for-the-whole-class"></a>[A global auxiliary function excludes the maintained target for the whole class](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#a-global-auxiliary-function-excludes-the-maintained-target-for-the-whole-class)

- <a id="sine-formation-clock-covariance"></a>[sine-formation-clock-covariance](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-formation-clock-covariance)

- <a id="constitutive-ratio-and-constant-clock-changes-are-different-operations"></a>[Constitutive ratio and constant clock changes are different operations](../research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#constitutive-ratio-and-constant-clock-changes-are-different-operations)

- <a id="sine-receiver-transfer-admission"></a>[sine-receiver-transfer-admission](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-receiver-transfer-admission)

- <a id="receiver-identity-transfer-requires-donor-loss-and-its-own-passage-proof"></a>[Receiver identity transfer requires donor loss and its own passage proof](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#receiver-identity-transfer-requires-donor-loss-and-its-own-passage-proof)

- <a id="exact-endpoint-conserved-origins-and-full-state-recovery"></a>[Exact endpoint, conserved origins and full-state recovery](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#exact-endpoint-conserved-origins-and-full-state-recovery)

- <a id="endpoint-budgets-permit-transfer-but-do-not-establish-it"></a>[Endpoint budgets permit transfer but do not establish it](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#endpoint-budgets-permit-transfer-but-do-not-establish-it)

- <a id="a-phase-passage-barrier-and-exact-excluded-controls"></a>[A phase passage barrier and exact excluded controls](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#a-phase-passage-barrier-and-exact-excluded-controls)

- <a id="two-necessary-winding-changes-consume-disjoint-parts-of-one-loss"></a>[Two necessary winding changes consume disjoint parts of one loss](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#two-necessary-winding-changes-consume-disjoint-parts-of-one-loss)

- <a id="what-remains-undecided"></a>[What remains undecided](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#what-remains-undecided)

- <a id="sine-receiver-port-passage"></a>[sine-receiver-port-passage](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-receiver-port-passage)

- <a id="receiver-acquisition-requires-accumulated-port-work-and-a-geometric-passage"></a>[Receiver acquisition requires accumulated port work and a geometric passage](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#receiver-acquisition-requires-accumulated-port-work-and-a-geometric-passage)

- <a id="use-the-actual-full-node-loss-and-the-existing-boundary-work-convention"></a>[Use the actual full-node loss and the existing boundary-work convention](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#use-the-actual-full-node-loss-and-the-existing-boundary-work-convention)

- <a id="the-exact-receiver-barrier-demands-a-running-maximum-not-final-work"></a>[The exact receiver barrier demands a running maximum, not final work](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#the-exact-receiver-barrier-demands-a-running-maximum-not-final-work)

- <a id="a-necessary-order-of-potential-barriers-for-the-original-budget-range"></a>[A necessary order of potential barriers for the original budget range](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#a-necessary-order-of-potential-barriers-for-the-original-budget-range)

- <a id="the-remaining-estimate-is-a-bound-on-actual-accumulated-receiver-supply"></a>[The remaining estimate is a bound on actual accumulated receiver supply](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#the-remaining-estimate-is-a-bound-on-actual-accumulated-receiver-supply)

- <a id="sine-localized-receiver-exclusion"></a>[sine-localized-receiver-exclusion](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-localized-receiver-exclusion)

- <a id="a-complete-nonlinear-prefix-and-dissipative-tail-can-exclude-receiver-acquisition"></a>[A complete nonlinear prefix and dissipative tail can exclude receiver acquisition](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#a-complete-nonlinear-prefix-and-dissipative-tail-can-exclude-receiver-acquisition)

- <a id="the-two-enclosure-obligations-are-different"></a>[The two enclosure obligations are different](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#the-two-enclosure-obligations-are-different)

- <a id="why-these-finite-inequalities-cover-the-whole-future"></a>[Why these finite inequalities cover the whole future](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#why-these-finite-inequalities-cover-the-whole-future)

- <a id="availability-failed-bounds-and-actual-passages-remain-distinct"></a>[Availability, failed bounds and actual passages remain distinct](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#availability-failed-bounds-and-actual-passages-remain-distinct)

- <a id="evaluated-result-for-the-frozen-localized-preparation"></a>[Evaluated result for the frozen localized preparation](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#evaluated-result-for-the-frozen-localized-preparation)

- <a id="exact-relative-coordinates-do-not-replace-failed-numerical-evidence"></a>[Exact relative coordinates do not replace failed numerical evidence](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#exact-relative-coordinates-do-not-replace-failed-numerical-evidence)

- <a id="boundary-of-this-result"></a>[Boundary of this result](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#boundary-of-this-result)

- <a id="sine-eleven-node-asymptotic-equilibria"></a>[sine-eleven-node-asymptotic-equilibria](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-eleven-node-asymptotic-equilibria)

- <a id="every-positive-loss-trajectory-on-this-support-converges-to-one-relative-equilibrium"></a>[Every positive-loss trajectory on this support converges to one relative equilibrium](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#every-positive-loss-trajectory-on-this-support-converges-to-one-relative-equilibrium)

- <a id="reuse-compactness-and-approach-to-the-equilibrium-set"></a>[Reuse compactness and approach to the equilibrium set](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#reuse-compactness-and-approach-to-the-equilibrium-set)

- <a id="critical-bridge-currents-vanish-rather-than-being-discarded"></a>[Critical bridge currents vanish, rather than being discarded](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#critical-bridge-currents-vanish-rather-than-being-discarded)

- <a id="an-odd-cycle-has-finitely-many-complete-sine-critical-branches"></a>[An odd cycle has finitely many complete sine-critical branches](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#an-odd-cycle-has-finitely-many-complete-sine-critical-branches)

- <a id="connectedness-of-the-limit-set-gives-one-relative-endpoint"></a>[Connectedness of the limit set gives one relative endpoint](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#connectedness-of-the-limit-set-gives-one-relative-endpoint)

- <a id="conserved-lifted-phase-reconstructs-the-final-common-origin"></a>[Conserved lifted phase reconstructs the final common origin](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#conserved-lifted-phase-reconstructs-the-final-common-origin)

- <a id="scope-of-the-stronger-long-time-conclusion"></a>[Scope of the stronger long-time conclusion](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#scope-of-the-stronger-long-time-conclusion)

- <a id="sine-eleven-node-equilibrium-stability"></a>[sine-eleven-node-equilibrium-stability](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-eleven-node-equilibrium-stability)

- <a id="exact-local-stability-of-every-classified-equilibrium"></a>[Exact local stability of every classified equilibrium](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#exact-local-stability-of-every-classified-equilibrium)

- <a id="ring-constraints-and-bridge-directions-determine-the-inertia-exactly"></a>[Ring constraints and bridge directions determine the inertia exactly](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#ring-constraints-and-bridge-directions-determine-the-inertia-exactly)

- <a id="the-full-reciprocal-jacobian-rather-than-phase-gradient-descent"></a>[The full reciprocal Jacobian, rather than phase gradient descent](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#the-full-reciprocal-jacobian-rather-than-phase-gradient-descent)

- <a id="nine-local-attractors-with-every-other-relative-branch-unstable"></a>[Nine local attractors, with every other relative branch unstable](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#nine-local-attractors-with-every-other-relative-branch-unstable)

- <a id="stable-composition-retains-causal-interaction-and-its-observations"></a>[Stable composition retains causal interaction and its observations](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#stable-composition-retains-causal-interaction-and-its-observations)

- <a id="the-original-source-budget-excludes-all-four-attracting-coexistence-states"></a>[The original source budget excludes all four attracting coexistence states](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#the-original-source-budget-excludes-all-four-attracting-coexistence-states)

- <a id="shared-geometry-and-law-readers-retain-different-claims"></a>[Shared geometry and law readers retain different claims](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#shared-geometry-and-law-readers-retain-different-claims)

- <a id="sine-donor-well-retention"></a>[sine-donor-well-retention](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-donor-well-retention)

- <a id="a-subcritical-auxiliary-well-selects-the-original-donor-endpoint"></a>[A subcritical auxiliary well selects the original donor endpoint](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#a-subcritical-auxiliary-well-selects-the-original-donor-endpoint)

- <a id="properness-and-critical-points-of-the-existing-auxiliary-function"></a>[Properness and critical points of the existing auxiliary function](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#properness-and-critical-points-of-the-existing-auxiliary-function)

- <a id="different-one-twist-wells-cannot-connect-below-72"></a>[Different one-twist wells cannot connect below `7/2`](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#different-one-twist-wells-cannot-connect-below-72)

- <a id="the-actual-preparation-enters-and-remains-in-the-donor-component"></a>[The actual preparation enters and remains in the donor component](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#the-actual-preparation-enters-and-remains-in-the-donor-component)

- <a id="a-stronger-preparation-control-without-a-claimed-dynamical-threshold"></a>[A stronger preparation control, without a claimed dynamical threshold](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#a-stronger-preparation-control-without-a-claimed-dynamical-threshold)

- <a id="sine-donor-dissipative-capture"></a>[sine-donor-dissipative-capture](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-donor-dissipative-capture)

- <a id="early-dissipation-captures-an-above-threshold-preparation-in-the-donor-well"></a>[Early dissipation captures an above-threshold preparation in the donor well](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#early-dissipation-captures-an-above-threshold-preparation-in-the-donor-well)

- <a id="full-state-norm-bounds-from-the-exact-initial-sine-balance"></a>[Full-state norm bounds from the exact initial sine balance](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#full-state-norm-bounds-from-the-exact-initial-sine-balance)

- <a id="a-directional-initial-loss-of-the-same-auxiliary-function"></a>[A directional initial loss of the same auxiliary function](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#a-directional-initial-loss-of-the-same-auxiliary-function)

- <a id="the-phase-path-stays-below-the-barrier-as-a-consequence-of-the-same-test"></a>[The phase path stays below the barrier as a consequence of the same test](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#the-phase-path-stays-below-the-barrier-as-a-consequence-of-the-same-test)

- <a id="one-fixed-above-threshold-witness-and-its-open-preparation-neighborhood"></a>[One fixed above-threshold witness and its open preparation neighborhood](../research/archive/receiver/SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#one-fixed-above-threshold-witness-and-its-open-preparation-neighborhood)

- <a id="sine-conservative-identity"></a>[sine-conservative-identity](SINE_PATTERN_RECOVERY.md#sine-conservative-identity)

- <a id="23-conservative-phase-identity-with-nonlinear-recurrence"></a>[23. Conservative phase identity with nonlinear recurrence](SINE_PATTERN_RECOVERY.md#23-conservative-phase-identity-with-nonlinear-recurrence)

- <a id="retain-the-law-and-distinguish-trapping-from-recovery"></a>[Retain the law and distinguish trapping from recovery](SINE_PATTERN_RECOVERY.md#retain-the-law-and-distinguish-trapping-from-recovery)

- <a id="a-genuine-phase-chart-with-common-origins-retained"></a>[A genuine phase chart with common origins retained](SINE_PATTERN_RECOVERY.md#a-genuine-phase-chart-with-common-origins-retained)

- <a id="the-coercive-barrier-is-independent-of-dissipation"></a>[The coercive barrier is independent of dissipation](SINE_PATTERN_RECOVERY.md#the-coercive-barrier-is-independent-of-dissipation)

- <a id="open-positive-volume-family-and-almost-everywhere-motion"></a>[Open positive-volume family and almost-everywhere motion](SINE_PATTERN_RECOVERY.md#open-positive-volume-family-and-almost-everywhere-motion)

- <a id="relative-observations-and-the-absolute-mean-slab-are-different-evidence"></a>[Relative observations and the absolute mean slab are different evidence](SINE_PATTERN_RECOVERY.md#relative-observations-and-the-absolute-mean-slab-are-different-evidence)

# Pattern memory and mediated collective geometry

**Status:** exact nonlinear quotient and hidden-state identity, conditional
finite-horizon error bounds for memory approximation and fast-mediator reduction,
stationary series composition with retained phase lifts, collective three-port
interaction, internal/mediator predictions, and conditional return-path
equilibrium and causal-pulse admission.
No numerical recovery
neighborhood or derivative bounds are certified here. The study
uses the same supplied relational law as
[local composition](RELATIONAL_PATTERN_COMPOSITION.md); it adds no state
variable, delay law, controller or physical identification. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
remains the sole task queue.

Sections 1-8 concern the single-bridge ten-node pattern. Section 9 derives
single-intermediary memory, controlled fast limits, stationary path composition
and collective three-port reduction. Section 10 retains the single-intermediary
finite response. Section 11 admits an additional supplied return edge;
Section 12 tests its collective oscillatory response. These supports and
reductions are not interchangeable.

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

<a id="mediated-pattern-interaction"></a>
## 9. A nodal intermediary mediates joint form/phase interaction

### Full state and conditional recovery

Replace the direct `0--5` bridge between the prepared C5 rings by
`0--10--5`. All twelve edges have unit conductance. The eleven-node state
retains signed form and primitive phase at every node, including mediator 10.
Hold capacity `nu>0` on both rings and `mu>0` on the mediator, with fixed
`e,w,beta>0` and the same structural clock. Here `e,w` are the effective
normalized channel coefficients. The law is the existing
`RelationalExchangeModel`, not an additional coupling or memory equation.

At uniform form, winding-one phases `theta_i=2*pi*(i mod 5)/5` on each ring
and `theta_10=0`, all sine sums vanish and every edge cosine is positive.
Set `c=cos(2*pi/5)`, `r=1+2*c`. Port phase metrics are `H_p=pi*r`, and
the mediator metric is `H_m=2*pi`. The
[existing local recovery theorem](RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
therefore applies to this connected graph: two common-offset freedoms and
twenty stable transverse directions, with a sufficiently small acute recovery
neighborhood. The two rings can have a joint restoring geometry through an
intermediary without a direct ring-to-ring edge. This is local recovery of a
prepared geometry on supplied support, not capture from disconnected support,
a global formation theorem or a physical bound-state identification.

### Derived memory, with two indispensable hidden coordinates

Write `u,v` for form and lifted phase deviations. Retain all twenty ring
coordinates as `y`, and hide only `z_m=(u_10,v_10)`. Define

\[
D_0=\begin{pmatrix}-e&-w/\pi\\w/(\beta\pi)&0\end{pmatrix},\qquad
B_0=-\tfrac12D_0,\qquad
C_0=\begin{pmatrix}e/3&w/(\pi r)\\-w/(\beta\pi r)&0\end{pmatrix}.
\]

The Jacobian of the admitted full field has hidden row
`z_m_dot=mu*D_0*z_m+mu*B_0*(z_0+z_5)`. Each visible port receives
`nu*C_0*z_m`; its other instantaneous terms remain in the full visible block
`A`. Here `z_0,z_5` denote the two form/phase port pairs within `y`, not
closed states of the rings. Let `B` inject `nu*C_0` at both port rows, and
let `C` read both port pairs with `mu*B_0`. Exact tangent elimination gives

\[
\dot y(t)=Ay(t)+B e^{\mu D_0t}z_m(0)
+\int_0^t B e^{\mu D_0(t-s)}C y(s)\,ds.
\]

The cross-region and self-memory port blocks are identical:

\[
\boxed{\quad K_\mu(t)=\mu\nu C_0 e^{\mu D_0t}B_0.\quad}
\]

This kernel follows from the declared field; no delay, fitted gain or primitive
pulse is introduced. It is exact for the tangent model. Nonlinear continuation
retains nonlinear hidden forcing, as in section 3, and cannot substitute this
fixed convolution for the full engine. The initial hidden-state term remains
necessary even when a particular experiment prepares it to zero.

Since `det(C_0)=w^2/(beta*pi^2*r^2)>0`, both mediator coordinates influence
the observed ring rates. For the supplied full-state Jacobian `J` and coordinate
observation `O`, `rank(O)=20` and `rank((O;OJ))=22`. Thus a twenty-coordinate
instantaneous all-state linear closure is impossible; retaining both hidden
coordinates or their exact memory is necessary. This reuses the invariant-row
argument, rather than introducing another state-minimality criterion.

<a id="mediator-pressure-boundary-chart"></a>
### An exact nonlinear pressure chart retains the moving boundary

The degree-two mediator also admits an exact change of coordinates beyond
the tangent approximation. Retain the full visible ring state and the same
fixed support, coefficients and capacities. On a continuous local phase lift,
retain full-field admission, including all ring edges, and write

\[
\bar x=\frac{x_0+x_5}{2},\qquad
\bar\theta=\frac{\theta_0+\theta_5}{2},\qquad
\delta=\theta_5-\theta_0,\qquad
u=x_m-\bar x,\qquad v=\theta_m-\bar\theta.
\]

Here `u,v` are mediator contrasts with the moving endpoints, not absolute
deviations from the equilibrium. Assume both incident gaps remain acute:
`|v+delta/2|<pi/2` and `|v-delta/2|<pi/2`. These imply `|delta|<pi`
and `|v|<pi/2`. The mediator's relative neighbor resultant is exactly

\[
z_m=2\cos(\delta/2)e^{-iv},\qquad
H_m=2\pi\cos(\delta/2)\operatorname{sinc}(v)>0,
\qquad g_m=-v/\pi,\qquad q_m=2u.
\]

Consequently its evaluated pressure, not a stale stored observation, obeys

\[
p_m=-eu-\frac w\pi v,\qquad
v=-\frac\pi w(p_m+eu).
\]

At fixed visible state, `(u,p_m)` is therefore an invertible replacement for
the two mediator coordinates. It preserves their initial information; it
does not remove the node, its capacity or either incident edge. Applying the
unchanged full field and differentiating the moving endpoint averages gives

\[
\dot u=\mu p_m-\dot{\bar x},\qquad
\dot v=\frac{\mu w u}
 {\beta\pi\cos(\delta/2)\operatorname{sinc}(v)}
 -\dot{\bar\theta},\qquad
\dot p_m=-e\dot u-\frac w\pi\dot v.
\]

These are exact nonlinear coordinate identities throughout the admitted
interval. The endpoint rates come from the retained full network; they are
not new external forcing or a closed two-port law. In particular, with held
`mu`, pressure history determines the contrast only after retaining its
initial value and the endpoint motion:

\[
u(t)=u(0)+\mu\int_0^t p_m(s)\,ds
       -[\bar x(t)-\bar x(0)].
\]

Substituting this expression and `v=-pi*(p_m+e*u)/w` into the pressure row
gives a nonlinear memory representation with moving-boundary terms. Dropping
`u(0)`, either endpoint-rate term, or the visible phase contrast changes the
model. This is the same closure principle as the
[P2 pressure chart](JOINT_PARAMETER_RESPONSE.md#pressure-state-closure), with
the additional boundary motion required by the mediator's environment.

The nonzero determinant of `C_0` above also rules out an exact smooth scalar
mediator summary for this full visible-rate observation on an open
neighborhood of the reference. At fixed visible state, the two rates at one
port have derivative `nu*C_0` with respect to the two mediator coordinates,
of rank two; any smooth factorization through one scalar has rank at most
one. This is local minimality for the stated observation, not a claim about
restricted one-dimensional preparations, lossy approximations or histories.
The [finite nonlinear compensation witness](RELATIONAL_PATTERN_COMPOSITION.md#environmental-capture-domain)
exhibits the corresponding failure of pressure alone inside a recovery
domain. The coordinate identities themselves also hold at `mu=0`, where
the mediator's absolute state freezes but its endpoint-relative coordinates
can still change. Positive-capacity recovery remains a separate theorem.
The [native controls](../../tests/physics/test_relational_mediation.py) check
both moving-boundary rows and an independent pressure derivative at nonuniform
visible states, including a frozen mediator. They test these coordinate
identities, not a reduced nonlinear trajectory solver.

<a id="fast-mediator-reduction"></a>
### A controlled nonlinear reduction when the mediator is fast

Hold the ring capacities at one and vary only the positive mediator capacity
`mu`, with the same unforced law, support, clock and effective coefficients. This is a
time-scale regime of the existing law, not a new interaction. Let `y` retain
all twenty visible ring coordinates and let `z=(u,v)` be the endpoint-relative
mediator coordinates just defined. Reconstruct the full state by
`x_m=bar(x)+u`, `theta_m=bar(theta)+v` and evaluate the native visible field
`F(y,z)`. Set `b(y,z)=(dot(bar(x)),dot(bar(theta)))` using its port rows. Then

\[
\dot y=F(y,z),\qquad \dot z=\mu f_\delta(z)-b(y,z),\qquad
f_\delta(u,v)=
\begin{pmatrix}
-eu-wv/\pi\\
wu/[\beta\pi\cos(\delta/2)\operatorname{sinc}(v)]
\end{pmatrix}.
\]

Both `F` and `b` are independent of `mu`: visible capacities are held, and
this relational pressure does not consume neighbor capacity. The unique
frozen-boundary fast equilibrium is `z=0`. The candidate reduced equation is
`dot(y_*)=F(y_*,0)`, with the same visible initial state. At finite `mu`, however,
`dot(z)=-b(y,0)` on `z=0`; the midpoint constraint is generally not invariant.
Its validity as an approximation requires the estimate below, not merely
instantaneous storage minimization.

**The reduced field retains the actual interface.** Write `q_ring` for the
form gradient from each ring's internal edges alone. At the reconstructed
midpoint,

\[
\widetilde q_0=q_{{\rm ring},0}+\frac{x_0-x_5}{2},\qquad
\widetilde q_5=q_{{\rm ring},5}+\frac{x_5-x_0}{2},
\]

with all other visible gradients unchanged. The relative phase resultants
at the two ports are

\[
\begin{aligned}
\widetilde Z_0&=e^{i(\theta_1-\theta_0)}+
 e^{i(\theta_4-\theta_0)}+e^{i\delta/2},\\
\widetilde Z_5&=e^{i(\theta_6-\theta_5)}+
 e^{i(\theta_9-\theta_5)}+e^{-i\delta/2}.
\end{aligned}
\]

Retain the original port degree three, the other visible degrees two, and
the native definitions `g_tilde=Arg(Z_tilde)/pi` and
`H_tilde=pi*|Z_tilde|*sinc(Arg(Z_tilde))`. Thus the visible rows are
`dot(x_i)=-e*q_tilde_i/d_i+w*g_tilde_i` and
`dot(theta_i)=w*q_tilde_i/(beta*H_tilde_i)`. In particular, this is not the
law on a ten-node graph with an ordinary unit `0--5` edge: that edge would
give the full form contrast and a full-angle neighbor phasor. The native
eleven-node field evaluated at the midpoint already supplies the required
reduction; no second pressure implementation is needed.

Its storage is also inherited, rather than fitted:

\[
\begin{aligned}
S_{\rm eff}={}&\frac12\sum_{\{i,j\}\in E_{\rm ring}}(x_i-x_j)^2
 +\frac{(x_0-x_5)^2}{4}\\
&+\beta\left[\sum_{\{i,j\}\in E_{\rm ring}}
 (1-\cos(\theta_i-\theta_j))+2(1-\cos(\delta/2))\right],\\
\dot S_{\rm eff}={}&-e\sum_{i\ne m}\frac{\widetilde q_i^2}{d_i}\le0.
\end{aligned}
\]

To obtain the last identity, both hidden storage derivatives vanish at the
midpoint, so differentiating its reconstruction contributes no extra term.
The remaining derivatives and form/phase exchange cancellation are the native
ones. The half-angle potential belongs to the declared `|delta|<pi` lift;
it is not a globally defined replacement cosine edge on the phase circle.
The selected phase lift remains part of its domain. Away from the midpoint,
the exact storage difference is

\[
S_{\rm full}(y,z)-S_{\rm eff}(y)
 =u^2+2\beta\cos(\delta/2)(1-\cos v)\ge0.
\]

A fixed nonzero hidden initial state can therefore carry finite excess
storage even when its visible trajectory effect becomes small. Its initial
layer loss is retained by the full law; the reduction does not claim uniform
storage or instantaneous-rate convergence at time zero.

**Uniform attraction for the default coefficients.** The following explicit
bound uses `e=w=1/2`, `beta=1`; it is not a uniform claim over all coefficients.
For `|delta|<=1/2`, `|v|<=1/4`, write `f_delta(z)=A_0*z+R_delta(z)` with

\[
A_0=\begin{pmatrix}-1/2&-1/(2\pi)\\1/(2\pi)&0\end{pmatrix},\qquad
P=\begin{pmatrix}2&\pi\\\pi&2+\pi^2\end{pmatrix}.
\]

Direct multiplication gives `A_0^T*P+P*A_0=-I`. The leading minor and
determinant of `P-I` are one, while `trace(P)=4+pi^2<14`, hence `I<P<14I`.
Moreover,

\[
\cos(\delta/2)\operatorname{sinc}(v)\ge
 \frac{31}{32}\frac{95}{96}=\frac{2945}{3072},\qquad
\|R_\delta(z)\|\le\frac{127}{6\cdot2945}\|z\|.
\]

Here `cos(t)>=1-t^2/2`, `sinc(t)>=1-t^2/6` on the stated intervals and
`pi>3` suffice. Therefore `2*||P||*127/(6*2945)<1/2`, and the fast field obeys

\[
2z^TPf_\delta(z)\le-\tfrac12\|z\|^2.
\]

This common quadratic bound does not differentiate a moving `P(delta)` and
holds for arbitrary admitted changes of the visible phase contrast. It uses
positive form damping: the `e=0` periodic boundary does not justify the same
attracting reduction.

**Domain and continuation hypotheses.** Fix a finite horizon `T` on which
the reduced solution `y_*` exists. Choose constants `d>0`, `0<rho<1/4` such
that its closed visible tube of radius `d`, together with
`W(z)=sqrt(z^T*P*z)<=rho`, is compactly contained in a smooth full-state
chart. Require all reconstructed ring edges to remain acute and
`|delta|<=1/2` there. The mediator edges are then also acute, because
`|v+-delta/2|<=1/2<pi/2`. In this fixed tube choose bounds, independent of
`mu`,

\[
\|b(y,z)\|\le B,\qquad
\|F(y,z)-F(y_*(t),0)\|
 \le L(\|y-y_*(t)\|+\|z\|).
\]

Smoothness supplies finite such bounds on an admitted tube; no numerical
values are inferred merely from a trajectory. Its margin must be justified
for the reduced solution, rather than assuming the full solution stays in
the domain whose preservation is to be proved. Near the aligned winding
reference, its strict acute margins and smooth reduced field provide such
a neighborhood for sufficiently close preparations on any fixed finite
horizon, by ordinary local existence and continuous dependence.

**Initial layer and visible error.** Up to the first exit from this tube,
the previous quadratic inequality and `|b|<=B` give

\[
D^+W\le-\frac\mu{56}W+14B,\qquad
W(t)\le e^{-\mu t/56}W(0)
 +\frac{784B}{\mu}(1-e^{-\mu t/56}).
\]

The upper right derivative formulation includes `W=0`. Since `||z||<=W`,
comparison of the two visible equations and Gronwall's inequality imply

\[
\boxed{\quad
\sup_{0\le t\le T}\|y(t)-y_*(t)\|
 \le\frac{Le^{LT}}{\mu}\,[56W(0)+784BT].\quad}
\]

For `W(0)<rho`, choose `mu` so that `784B/mu<rho` and the displayed visible
bound is strictly below `d`. The estimate for `W` is a convex combination of
two values below `rho`. Both estimates therefore prevent a first exit;
compact smooth continuation extends the solution through `T` and validates
the bounds on that whole interval. These conditions are sufficient, not
optimal, and do not supply a numerical cutoff without actual tube bounds.

A fixed small hidden initial state is allowed. Its visible contribution is
included in the `56W(0)/mu` term, while the hidden state retains an initial
layer of decay `exp(-mu*t/56)`. It is not uniformly small at time zero:
for `mu>=1` it becomes `O(1/mu)` after a time of order `log(mu)/mu`.
Thus finite-memory influence can be approximated in this regime without
denying the exact hidden-state obstruction at fixed capacity. Neither
arbitrary initial environments nor infinite-horizon errors are covered.

This is an ideal continuous-law result. Its field evaluation reuses detached
[`evaluate_relational_exchange`](../../src/tnfr/dynamics/relational.py)
at the reconstructed midpoint and projects the visible rows. The
[native static controls](../../tests/physics/test_relational_mediation.py)
can check its field substitution and storage identities; they do not certify
the tube constants or a reduced trajectory. Fixed-step Euler is not justified
as `mu` grows: resolving the fast initial layer needs a separate numerical
step-size analysis. No new solver, primitive edge, occurrence law or physical
identification follows from eliminating this already supplied mediator.

<a id="two-mediator-composition"></a>
### Two fast intermediaries compose through their inherited interface

Replace the intermediary path by `0--10--11--5`, retaining the same two C5
rings and all thirteen unit edges. There are twenty visible coordinates and
four hidden coordinates. Hold ring capacities at one and initially give both
intermediaries capacity `mu>0`, with `e=w=1/2`, `beta=1`. The unforced law
and clock are unchanged. Choose a local acute path lift and write
`X=x_5-x_0`, `delta=theta_5-theta_0`. Frozen-endpoint hidden equilibrium is

\[
x_{10}^*=x_0+X/3,\quad x_{11}^*=x_0+2X/3,\qquad
\theta_{10}^*=\theta_0+\delta/3,\quad
\theta_{11}^*=\theta_0+2\delta/3.
\]

Let `u=(x_10-x_10^*,x_11-x_11^*)` and
`v=(theta_10-theta_10^*,theta_11-theta_11^*)`. For `z=(u_1,u_2,v_1,v_2)`, set

\[
M=\begin{pmatrix}2&-1\\-1&2\end{pmatrix},\qquad
\begin{aligned}
C_1&=\cos(\delta/3+v_2/2)\operatorname{sinc}(v_1-v_2/2),\\
C_2&=\cos(\delta/3-v_1/2)\operatorname{sinc}(v_2-v_1/2).
\end{aligned}
\]

The hidden metrics are `H_10=2*pi*C_1`, `H_11=2*pi*C_2`, the form gradient
is `M*u`, and the phase source is `-M*v/(2*pi)`. Thus the exact fast map is

\[
f_\delta(z)=
\begin{pmatrix}
-(e/2)Mu-(w/(2\pi))Mv\\
(w/(2\beta\pi))\operatorname{diag}(C_1^{-1},C_2^{-1})Mu
\end{pmatrix}.
\]

The moving-boundary system again has `dot(y)=F(y,z)` and
`dot(z)=mu*f_delta(z)-b(y,z)`. Here `b` consists of the derivatives of the
four interpolated coordinates above, for example
`dot(x_10^*)=(2*dot(x_0)+dot(x_5))/3`. All these endpoint rates come from
the retained visible field. The hidden equilibrium is unique in this chart:
the phase rows first require `M*u=0`, and the form rows then require `M*v=0`.

**Exact static composition and reduced balance.** Evaluating the full native
field at this lift gives

\[
\widetilde q_0=q_{{\rm ring},0}-X/3,\qquad
\widetilde q_5=q_{{\rm ring},5}+X/3,
\]

and port resultants equal to their two internal ring phasors plus
`exp(i*delta/3)` at port 0 and `exp(-i*delta/3)` at port 5. Retain original
port degree three and all native metric/phase-source definitions. The storage
and its derivative under the induced visible law are

\[
S_{\rm eff}=S_{\rm rings}+X^2/6+3\beta(1-\cos(\delta/3)),\qquad
\dot S_{\rm eff}=-e\sum_{i\ne10,11}\widetilde q_i^2/d_i\le0.
\]

Both hidden storage gradients vanish at the lift, so the same envelope
chain rule used for one intermediary proves the balance. Hidden initial
storage is not removed from the full system. Its form excess is
`u^T*M*u/2`; its phase excess is

\[
\beta[3\cos(\delta/3)-\cos(\delta/3+v_1)
 -\cos(\delta/3+v_2-v_1)-\cos(\delta/3-v_2)].
\]

The latter is nonnegative when the three lifted gaps remain acute, by strict
convexity of `1-cos` on that interval and their fixed sum. A fixed hidden
initial displacement can therefore again carry finite initial-layer storage.

Successive static elimination gives the same result if its inherited
interface is retained. Eliminating node 10 first gives
`x_10=(x_0+x_11)/2`, `theta_10=(theta_0+theta_11)/2`. The remaining gradient is
`q_11=(x_11-x_0)/2+(x_11-x_5)`, still with original degree two, while its
phase resultant is

\[
e^{-i(\theta_{11}-\theta_0)/2}+e^{i(\theta_5-\theta_{11})}.
\]

Setting both remaining rows to zero yields the same thirds; reversing the
order does too. By contrast, replacing the eliminated path by an ordinary
unit `0--11` edge would produce the false interpolation
`x_10=x_0+X/4`, `x_11=x_0+X/2`. Reconstructing the original graph gives
`q_11=-X/4`, and the analogous phase preparation has hidden gradient
`sin(delta/4)-sin(delta/2)`, generally nonzero. Static elimination therefore
composes through the inherited field, not through an arbitrary replacement
graph. This equality does not by itself interchange finite-capacity dynamics
or establish nested initial layers for successively separated time scales.

The same static statement holds for a finite path of `ell` unit-conductance
edges, with only degree-two interior nodes and a specified path lift.
Here `ell` counts fine links; it is not the separate graph `length` attribute
or an inferred physical distance:

\[
S_{\rm path}^{\rm eff}(X,\delta)
 =\frac{X^2}{2\ell}+\beta\ell(1-\cos(\delta/\ell)).
\]

Each gap is `delta/ell` in the selected acute sector. Joining lengths
`ell_1,ell_2` and minimizing their common boundary places that boundary at
`x_A+ell_1*(x_B-x_A)/(ell_1+ell_2)`, with the same lifted phase interpolation.
The phase condition follows from the injectivity of sine on the acute
interval. Energy and endpoint messages therefore agree with length
`ell_1+ell_2`, associatively. The message retains length, lift, endpoint form
contrast, adjacent phase and original node degrees. Different allowed lift
sectors cannot be silently identified using endpoint phases modulo full
turns. This is a static series result, not a dynamical estimate uniform in
path length or a result for branching interiors.

For example, a three-edge path whose endpoint circular gap is `3*pi/4`
admits both lifted gaps `pi/4` on every edge and `-5*pi/12` on every edge.
Its two hidden phase pairs are respectively `(pi/4,pi/2)` and
`(-5*pi/12,-5*pi/6)` relative to the left endpoint. Both have zero hidden
phase source and acute fine edges, yet their adjacent port phasors differ.
With uniform form and the maintained ring twists, donor pressure has opposite
signs. Thus length and circular endpoint phases alone are insufficient;
static composition must retain the selected lift sector. This control is
outside the small-`delta` quantitative neighborhood used next.

**Joint fast attraction and a finite-horizon bound.** At the default
coefficients and `delta=v=0`, the joint fast Jacobian and a common quadratic
matrix are

\[
J_0=\begin{pmatrix}-M/4&-M/(4\pi)\\M/(4\pi)&0\end{pmatrix},\qquad
P_4=\begin{pmatrix}2I&\pi I\\\pi I&(2+\pi^2)I\end{pmatrix}.
\]

They satisfy `I<P_4<14I` and
`J_0^T*P_4+P_4*J_0=-diag(M,M)/2<=-I/2`. For
`|delta|<=1/2`, `|v_1|,|v_2|<=1/8`, the cosine arguments have magnitude at
most `11/48` and the sinc arguments at most `3/16`. Hence

\[
C_i\ge\frac{4487}{4608}\frac{509}{512}
 =\frac{2283883}{2359296},\qquad
0\le C_i^{-1}-1\le\frac{75413}{2283883}<\frac1{28}.
\]

Since `||M||=3` and `pi>3`, the nonlinear remainder obeys
`||f_delta(z)-J_0*z||<=||z||/112`. Consequently
`2*z^T*P_4*f_delta(z)<=-||z||^2/4`. This controls the four hidden coordinates
jointly; independent single-node decay estimates are not substituted for the
coupled proof.

Reuse the preceding finite-horizon tube and first-exit construction, now with
`W=sqrt(z^T*P_4*z)`, `rho<1/8`, and bounds `B,L` for this twelve-node
reconstruction. On its admitted interval,

\[
\begin{aligned}
W(t)&\le e^{-\mu t/112}W(0)
 +\frac{1568B}{\mu}(1-e^{-\mu t/112}),\\
\sup_{0\le t\le T}\|y(t)-y_*(t)\|
 &\le\frac{Le^{LT}}\mu[112W(0)+1568BT].
\end{aligned}
\]

Require `W(0)<rho`, `1568B/mu<rho` and a visible error bound strictly below
the tube radius `d`. These close the same first-exit argument. The three
intermediary edge gaps are acute throughout the box, with magnitude at most
`5/12`; the tube must separately retain all ring-edge and visible-state
margins. Thus the reduction has an `O(1/mu)` visible error with its hidden
initial layer retained, on a fixed finite horizon and admitted neighborhood.
Neither a fixed-step Euler certificate nor an infinite-horizon bound follows.

Fixed positive unequal capacity ratios also admit a local stability
certificate. For capacities `mu*a_1,mu*a_2`, put
`D=diag(a_1,a_2)`, `D_4=diag(D,D)` and `P_a=P_4*D_4^-1`. The fast map is
`D_4*f_delta`, and exactly
`2*z^T*P_a*D_4*f_delta=2*z^T*P_4*f_delta<=-||z||^2/4`.
Moreover `I/a_max<P_a<14I/a_min`. The same proof therefore gives constants
depending on these fixed ratios, with its hidden tube rescaled to preserve
`||z||<=sqrt(a_max)*sqrt(z^T*P_a*z)<1/8`. They are not uniform as a ratio
vanishes or becomes unbounded. The explicit `112,1568` bounds above are for
equal capacities only.

The [native static controls](../../tests/physics/test_relational_mediation.py)
check the reconstructed field, inherited balance and the wrong-edge control.
They do not evaluate a reserved trajectory or supply numerical tube constants.
No additional pressure owner, topology mutation or reduced executor is
introduced: composition is performed on the already supplied nodal system.

<a id="three-port-collective-interaction"></a>
### A branching intermediary induces a collective three-port interaction

Take three C5 rings on nodes `0,...,14`, with ports `A=0`, `B=5`, `C=10`,
and connect one intermediary `m=15` to all three ports. All eighteen edges
have unit conductance. Retain all thirty visible form/phase coordinates,
unit visible capacities, a held positive intermediary capacity, and the same
unforced relational law with `e=w=1/2`, `beta=1`. The support is supplied;
stationary elimination does not assert that the full system is stationary.

For the three port phases define

\[
Z=e^{i\theta_A}+e^{i\theta_B}+e^{i\theta_C}=Re^{i\Psi},\qquad
\bar x=(x_A+x_B+x_C)/3.
\]

Work in a local chart with `R>0` and all three gaps
`alpha_p=theta_p-Psi` strictly acute, and retain admission of every ring
edge. A nonzero resultant alone does not guarantee this acute-star premise.
The unique conditional hidden equilibrium is

\[
x_m^*=\bar x,\qquad \theta_m^*=\Psi.
\]

Indeed the hidden phase row first requires `q_m=0`, fixing the form average;
the form row then requires zero phase source. In the admitted chart this
selects the circular resultant direction. The antipodal stationary point of
the phase storage is outside this domain. Both hidden storage gradients
vanish at the admitted lift.

**Inherited field and storage.** At each port the hidden contribution to
the form gradient is `x_p-bar(x)`, and its relative neighbor phasor is
`exp(i*(Psi-theta_p))`. Add these to the port's two internal ring contributions
and retain degree three, the native phase source and the native phase metric.
Other visible rows are unchanged. Consequently the full sixteen-node native
field at the conditional lift supplies the induced visible law without
another pressure formula. Its storage is

\[
S_{\rm eff}=S_{\rm rings}
 +\frac16\sum_{p<q}(x_p-x_q)^2+\beta(3-R),\qquad
\dot S_{\rm eff}=-e\sum_{i\ne m}\widetilde q_i^2/d_i\le0.
\]

The envelope chain rule again removes the hidden reconstruction derivative
because its storage gradients vanish there. This is a closed conditional
field on the full visible state, not a closed state of three rigid rings
or an exact finite-capacity elimination of hidden history.

**The phase storage is not a sum of independent pair interactions.** Define
`V=3-R`. Even allowing arbitrary three-times differentiable pair functions
of their two absolute phases, plus one-port terms, a representation

\[
V=V_{AB}(\theta_A,\theta_B)+V_{AC}(\theta_A,\theta_C)
 +V_{BC}(\theta_B,\theta_C)+\sum_p V_p(\theta_p)
\]

would require `partial_A partial_B partial_C V=0` throughout its domain.
But at `(theta_A,theta_B,theta_C)=(0,0,t)`, with `0<t<pi/4`, direct
differentiation gives

\[
\begin{aligned}
\partial_A\partial_B V
 &=-\frac{(2+\cos t)^2}{(5+4\cos t)^{3/2}},\\
\partial_A\partial_B\partial_C V
 &=-\frac{2\sin t(1-\cos t)(2+\cos t)}{(5+4\cos t)^{5/2}}\ne0.
\end{aligned}
\]

These preparations lie in the acute-star domain and can be arbitrarily
close to alignment. Thus no independent-pair identity holds on an open
neighborhood of that reference. Allowing a nominal pair to consume the
third port's state would instead declare a collective interaction.

Low-order agreement does not remove this obstruction. For local differences
`Delta_pq=theta_p-theta_q`, set `S_k=sum_pairs Delta_pq^k` and
`P_6=product_pairs Delta_pq^2`. Expansion about alignment gives

\[
V=\frac{S_2}{6}-\frac{S_4}{216}-\frac{S_6}{19440}
 +\frac{P_6}{648}+O(\|\Delta\|^8).
\]

Here `S_4=S_2^2/2` and `S_6=S_2^3/4+3*P_6`. General pair potentials can
match the quadratic and quartic terms; the displayed genuinely mixed
storage term first occurs at sixth order. The particular cosine surrogate
`sum_pairs(1-cos(Delta_pq))/3` already fails at fourth order. This order
statement concerns storage, not every component of the nonlinear field.

**The actual port rate is collective too.** Keep all forms zero, hold ring A
in its winding-one reference, and rotate rings B and C by independent `s,t`.
Let `a=2*cos(2*pi/5)`, so `0<a<1`, and define

\[
\Psi(s,t)=\arg(1+e^{is}+e^{it}),\qquad
g(\psi)=\arg(a+e^{i\psi}),\qquad
\dot x_A=\frac w\pi g(\Psi(s,t)).
\]

The complete state and both internal neighbors of A remain fixed in this
comparison. At `s=t>0` sufficiently small, with `Q=5+4*cos(t)`,

\[
\Psi_s=\Psi_t=\frac{2+\cos t}{Q},\qquad
\Psi_{st}=\frac{2(2+\cos t)\sin t}{Q^2}>0,
\]

while

\[
g'(\psi)=\frac{1+a\cos\psi}{1+a^2+2a\cos\psi}>0,\qquad
g''(\psi)=\frac{a(1-a^2)\sin\psi}{(1+a^2+2a\cos\psi)^2}>0
\]

at the resulting positive `Psi`. Hence
`partial_s partial_t dot(x_A)=(w/pi)*(g''*Psi_s*Psi_t+g'*Psi_st)>0`.
An additive rate depending separately on `(A,B)` and `(A,C)`, plus A alone,
has zero mixed derivative under these same variations. This rules out that
independent-pair field representation, even without requiring a pairwise
storage. Shared state-dependent normalizations that read the third ring
would not satisfy the proposed independence contract.

Near alignment, this mixed rate derivative is
`2*w*t*(1+3*a)/(27*pi*(1+a)^3)+O(t^3)`. Thus the native visible field
already has a collective cubic contribution, although the storage's first
genuinely mixed contribution is sixth order. The inherited phase metric
and readout prevent those two order statements from being interchanged.

**A controlled fast regime is available locally.** To justify a dynamical
approximation, set `u=x_m-bar(x)`, `v=theta_m-Psi(y)`, `z=(u,v)` and give
the mediator capacity `mu`. The exact moving-boundary rows are

\[
\dot z=\mu
\begin{pmatrix}-eu-wv/\pi\\3wu/[\beta\pi R\operatorname{sinc}(v)]\end{pmatrix}
-\begin{pmatrix}\dot{\bar x}\\\dot\Psi\end{pmatrix},\qquad
\dot\Psi=\frac1R\sum_{p=A,B,C}\cos(\alpha_p)\dot\theta_p.
\]

The phase-boundary velocity is a weighted circular-mean derivative, not an
arithmetic average. Its port rates are native visible rates. At the default
coefficients, require `|alpha_p|<=1/4` and `|v|<=1/4` in a compact admitted
tube about a reduced visible trajectory. Then

\[
\frac R3\operatorname{sinc}(v)\ge
 \frac{31}{32}\frac{95}{96}=\frac{2945}{3072}.
\]

The same two-dimensional matrix `P` from the fast-mediator theorem therefore
gives exactly its common quadratic estimate and `56,784` bounds. Apply its
first-exit argument with newly justified bounds `B,L`, hidden radius
`rho<1/4`, and visible margin `d` for this sixteen-node reconstruction.
The boundary velocity and visible field remain independent of `mu`, so the
visible trajectory error is `O(1/mu)` on the fixed finite horizon, including
the hidden initial-layer contribution. Neither the earlier graph's tube
constants nor its preparation are silently reused. At finite capacity the
conditional lift is not generally invariant.

The hidden storage excess is exactly
`3*u^2/2+beta*R*(1-cos(v))`; a fixed nonzero hidden initial state can carry
finite storage into that layer. Its loss and initial rate discrepancy are
not erased by the visible approximation. The
[native mediation controls](../../tests/physics/test_relational_mediation.py)
check the induced rows, storage and mixed-response discriminator without a
new response campaign. The result derives a collective conditional interaction
from the supplied nodal law and support; it installs no new connection rule
or reduced executor.

### Capacity sets the memory clock, and the kernel need not be positive

For the two-port single-intermediary model at the start of this section,
the hidden eigenvalues solve
`s^2+mu*e*s+mu^2*w^2/(beta*pi^2)=0`. They have negative real parts for the
positive premises above. Default `e=w=1/2`, `beta=1` gives two negative real
roots; an oscillating mediator is not required. For dimensionless `s>0`,
`K_(s*mu)(t)=s*K_mu(s*t)` retains the same ring capacity `nu`. Hence

\[
\int_0^\infty K_\mu(t)\,dt=\frac\nu2 C_0\qquad(\mu>0),
\]

because `B_0=-D_0/2`. Changing mediator capacity changes the transient memory
while leaving this integrated tangent kernel unchanged. This is not an
invariance of full trajectories or final nonlinear form offsets. At `mu=0`
the mediator freezes and transmits no donor-induced change from identical
mediator initial states. The positive-capacity integral limit is not uniform
as `mu` tends to zero; it cannot be assigned to the frozen case.

The phase-to-phase cross entry satisfies

\[
[K_\mu(0)]_{vv}=-\frac{\mu\nu w^2}{2\beta\pi^2r}<0,
\qquad \int_0^\infty[K_\mu(t)]_{vv}\,dt=0.
\]

Continuity and exponential decay force a positive contribution at some later
lag. Thus even the overdamped default has a signed memory entry. This describes
feedback from past phase deviations in a chosen tangent chart, not alternating
physical attraction, a receiver trajectory reversal or an autonomous bond.
Pure-diffusion positive-kernel results cannot be transferred to this joint law.

### A prospective onset discriminator in the actual nonlinear field

Prepare only donor form `u_0(0)=epsilon!=0`; all other form deviations and all
phase deviations, including the mediator's, are zero. Direct differentiation
at this preparation gives, for the mediated receiver,

\[
\dot u_5(0)=\dot v_5(0)=0,\qquad
\ddot u_5(0)=\epsilon\mu\nu
 \left[\frac{e^2}{6}-\frac{w^2}{2\beta\pi^2r}\right],\qquad
\ddot v_5(0)=-\frac{\epsilon\mu\nu ew}{2\beta\pi r}.
\]

These initial derivatives are exact for the nonlinear ideal field as well:
the receiver's initial form gradient is zero, so its phase-metric derivative
does not enter the phase acceleration. The full subsequent memory remains
nonlinear. A direct `0--5` bridge instead gives initial receiver rates
`e*nu*epsilon/3` and `-w*nu*epsilon/(beta*pi*r)`. Removing the intermediary
path leaves the independent receiving ring with no donor response.

The distinction is first-order versus second-order onset, not a finite waiting
time before transmission. The phase contribution competes with diffusion in
the form acceleration: at `e=w=1/2`, `beta=1` it reduces but does not reverse
the positive form signal for `epsilon>0`. The displayed coefficient changes
sign below `beta=3*w^2/(e^2*pi^2*r)`; this is a conditional family boundary,
not a reason to fit beta after a response. Varying only mediator capacity
predicts proportional initial accelerations, including a frozen-mediator null.

### Shared implementation and evidence scope

[`derive_coordinate_memory`](../../src/tnfr/mathematics/linear_observation.py)
centralizes the detached exact block split of any supplied fixed generator
`z_dot=J*z`. It retains visible/hidden coordinate order, the four blocks and
`K(0)=B*C`; it does not assume diffusion, positivity, stability or a memory
cutoff. Rationalized trigonometric entries describe the supplied rational
matrix, not exact real pi or the ideal C5 cosine. `derive_linear_observation`
remains the independent owner of invariant-row closure.

The [mediator controls](../../tests/physics/test_relational_mediation.py) compare
the analytic joint Jacobian and prepared response against the production
field. Static finite-difference checks are complemented by a closed-form
two-step receiver control at `dt=1/32`, distinct from the reserved horizon
below. Neither supplies an ODE error enclosure. Model defaults, ordered nodes `0,...,10`,
unit edges, held capacities, explicit donor amplitude and binary64 arithmetic
are retained in the tests. No stochastic preparation, new event selector or
change to the nodal executor is involved. The mechanism is conditional mediated
interaction with local recovery; primitive support origin and a finite capture
prediction remain separate obligations in the sole execution plan.

<a id="mediator-orientation-scope"></a>
### Orientation sensitivity: winding sign is invisible at a single symmetric port

The [magnetic binding comparison](../PHYSICAL_REGIME_CORRESPONDENCES.md#magnetic-binding-comparison)
asks whether a pattern's orientation changes its interaction. The present
one-port-per-ring geometry has an exact limitation. Reflect the donor ring by
`R=(1 4)(2 3)`, fixing port 0, mediator 10 and the entire receiving ring.
This is an automorphism of the supplied graph. With capacities and all other
node data transformed consistently, the ideal full field is equivariant:

\[
F(Rx,R\theta)=R F(x,\theta).
\]

For the uniform-capacity preparation above, this reflection maps the donor's
positive twist to its negative twist modulo full phase turns, while leaving
a donor-port form impulse and the receiving preparation unchanged. Uniqueness
on the admitted smooth domain then gives `z_minus(t)=R*z_plus(t)`.
Mediator and receiving-ring histories are identical throughout their common
admitted interval. This is a nonlinear symmetry statement, not merely the
fact that the tangent coefficients contain the even quantity `cos(kappa)`.

Consequently, the current experiment cannot distinguish equal from opposite
winding signs using its receiver response. Assigning those signs magnetic
polarities would introduce a physical meaning that this interface cannot
resolve. Winding itself depends on a declared cycle orientation; no spatial
magnetic moment follows from its integer value. Nor does a signed memory
entry establish magnetic attraction or repulsion.

This does not rule out orientation-sensitive TNFR interactions under other
admitted preparations or structures. The existing
[hidden form-orientation control](RELATIONAL_PATTERN_COMPOSITION.md#hidden-form-orientation-changes-the-metrics-next-response)
already shows nonlinear dependence on relative form/phase orientation.
An asymmetric internal preparation, two separately distinguished contacts or
a justified geometric observation can remove the independent reflection
symmetry. Their state, support and observation must be declared before such a
claim is tested; new contacts also require fresh phase-domain admission.
They are not an automatic extension of the single-port result. This comparison
does not replace the finite mediator-capacity prediction in the sole queue.

<a id="finite-mediated-response"></a>
## 10. A finite causal-response test of the effective connection

### Influence has an onset order, not a selected activation time

For the section 9 donor preparation, compare the perturbed receiver with the
unperturbed equilibrium at the same supplied support and capacities. Smoothness
and the exact initial acceleration imply

\[
\Delta\theta_5(t)=
-\frac{\epsilon\mu\nu ew}{4\beta\pi r}\,t^2+O(t^3).
\]

With positive coefficients and nonzero epsilon, this difference is nonzero
for every sufficiently small positive time. The first derivative vanishes
at zero; a finite waiting interval does not follow. A threshold-based time
of detection would be an observation policy, not an autonomous support event.
The unperturbed equilibrium itself generates no signal. At zero mediator
capacity, equal mediator initial states stay equal and the receiving dynamics
are independent of the donor. Removing the fine path also removes influence.

This defines the useful causal question without adding a primitive edge:
does an intervention in one region change another through the retained
intermediary? The memory law answers it conditionally. Joint geometric
recovery has its separate positive-capacity theorem; capture from a distant
preparation, support birth and physical binding do not follow from a nonzero
response alone.

### A frozen mediator can organize both rings without coupling their changes

The zero-capacity control is stronger than an absent signal at equilibrium.
With mediator state `(x_m,theta_m)` held fixed, each ring is a separate system
with one anchored boundary. Its acute reference is uniform form `x_m` and
the same ring twist rotated by `theta_m`. Write `B_g,K_g` for the grounded
form Laplacian and cosine-weighted phase Hessian, including the anchoring edge.
Both are positive definite. On each ring, `A=N*D^-1` and `M=N*H_*^-1`
remain positive diagonal, with degrees including that edge. The local Jacobian
is

\[
J_g=\begin{pmatrix}
-eAB_g&-wMK_g\\ (w/\beta)MB_g&0
\end{pmatrix}.
\]

Storage coordinates transform this into
`[[-D_cal,-C_cal],[C_cal^T,0]]`, where
`D_cal=e*B_g^(1/2)*A*B_g^(1/2)>0` and
`C_cal=(w/sqrt(beta))*B_g^(1/2)*M*K_g^(1/2)` is invertible.
The real-part argument in the existing
[local recovery proof](RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
then makes this grounded Jacobian Hurwitz. Each ring has local exponential
recovery to its fixed-boundary reference, without an internal common-offset
freedom. This is a separate anchored theorem, not an extension of the
whole-network strictly-positive-capacity theorem to a zero capacity.

Nevertheless, the two systems consume no changing state from each other.
Perturbing one cannot alter the other when mediator and receiver preparations
are held fixed. Thus common geometry and individual restoration can occur
without mutual influence. A claim of a collective bond must retain this
common-boundary control as well as its geometry and recovery observations.

### A symmetric port prediction has no quadratic amplitude error

Reflect both rings around their ports, fixing nodes 0, 5 and 10, and call the
permutation `R`. Let `Q=diag(R,R)` act on the 22 form/phase deviations from
the ideal reference. Simultaneous form/phase inversion and graph equivariance
give, on the admitted lifted chart,

\[
G(-Qz)=-QG(z),\qquad -R\theta_*=\theta_*\pmod {2\pi}.
\]

The donor direction `a=(e_0,0)` is reflection-even. Each simultaneous Euler
map inherits the symmetry, so its N-step state satisfies
`z_N(-epsilon)=-Q*z_N(epsilon)`. A linear observation `O` of the receiver
port or mediator obeys `OQ=O`. Smooth dependence on the initial amplitude
therefore yields, for a fixed admitted grid,

\[
Oz_N(\epsilon)=\epsilon O(I+hJ_\mu)^Na+O(\epsilon^3).
\]

The full 22-coordinate state need not have cubic error: internal
reflection-odd modes can carry quadratic corrections. This result specifies
the approximation order near zero; it supplies neither a numerical cubic
remainder constant at a chosen amplitude nor an exact-ODE error bound.
Native represented reference phases can have tiny residual drift, which
must be recorded rather than subtracted from the response after evaluation.

### Frozen finite-executor protocol

Use the existing law with `e=w=1/2`, `beta=1`, ring capacities one,
donor form `epsilon=1/64`, and otherwise the section 9 equilibrium. Fix mediator
capacities `mu=0,1,2`, horizon `T=1/4`, and 64 and 128 simultaneous Euler
steps. These two grids are numerical controls of the same preparation, not
independent physical trials or validated continuous-time enclosures.

The memory predictor advances its exact tangent blocks with old-state values:

\[
y_{n+1}=y_n+h(Ay_n+Bh_n),\qquad
h_{n+1}=h_n+h(Cy_n+Dh_n).
\]

The hidden initial deviation is zero in this preparation. The memory-omitting
control retains `y_{n+1}=y_n+hAy_n`; its receiving ring remains at zero.
It is a deliberate ablation, not a proposed closed law for arbitrary states.
For positive mediator capacity, a stronger instantaneous comparison sets
`0=Cy+Dh`, hence `h=-D^-1*C*y=(z_0+z_5)/2`. Its visible generator is
`A-B*D^-1*C`. Both the capacity factor and its inverse cancel, so this
approximation predicts identical responses at `mu=1` and `mu=2` on every
matched grid. It also replaces the prepared zero hidden state with its
instantaneous equilibrium and need not match the initial onset. It is not
an exact transient reduction and is undefined by this inverse at `mu=0`.
The capacity intervention therefore discriminates retained memory from a
fixed instantaneous connection even when their integrated kernel agrees.
Observe the receiver port pair `(x_5,theta_5-theta_{*,5})`, retaining the full
state and mediator endpoint as additional evidence.

Before native execution, freeze each tangent prediction and its source/runtime
provenance. The prospective decision requires positive-capacity endpoint and
`mu=2` minus `mu=1` intervention errors in maximum norm no greater than 1%
of their respective predicted signals plus `1e-12`. Each predicted signal
must exceed `100e-12`; the frozen-mediator receiver must stay within `1e-12`.
Also require held support/capacities, the declared fresh-pressure path,
acute margin at least `pi/20`, and actual work-balance residual at most `1e-12`.
These are finite experiment decision tolerances, not emergent constants.
A pass would test the derived interaction without fitting an edge or kernel;
it would not promote the tolerance to a proved nonlinear or solver error bound.

### Reserved response and decision

The [mediator instrument](../../benchmarks/relational_mediation_response.py)
uses the shared coordinate-memory decomposition and Euler arithmetic for the
prediction, and `step_relational_exchange` for the full nonlinear response.
The [prediction](../../docs/assets/relational_mediation_response/result.prediction.json)
and [source archive](../../docs/assets/relational_mediation_response/result.sources.zip)
were frozen before the [reserved response](../../docs/assets/relational_mediation_response/result.json).
All declared gates passed on both grids without changing the preparation,
law, coefficients or decision tolerances.

At 128 steps, the receiver results were:

| Mediator capacity | Predicted form | Observed form | Predicted phase deviation | Observed phase deviation | Pair maximum error |
| --- | --- | --- | --- | --- | --- |
| 1 | `1.4859487561e-5` | `1.4859487442e-5` | `-1.0748178383e-5` | `-1.0748178470e-5` | `1.18910e-13` |
| 2 | `2.8689674324e-5` | `2.8689674024e-5` | `-2.0729626866e-5` | `-2.0729627404e-5` | `5.37972e-13` |

The capacity intervention had predicted maximum norm `1.3830186762e-5`
and observed norm `1.3830186582e-5`; its pair error was `4.51169e-13`.
Instantaneous elimination predicted exactly zero intervention difference.
The omitted-memory control predicted zero receiver response. The frozen
mediator's observed receiver norm stayed below `2.41e-18` on both grids,
consistent with its null prediction and materialized reference drift.

These values describe the retained finite executor comparison. In particular,
the largest positive-capacity receiver error relative to its predicted pair
was `1.88e-8`, but full-state tangent errors reached `7.22e-7`, consistent
with retaining a different approximation scope for hidden internal shape.
The largest 64-to-128-grid full-state difference was `1.884e-6`; this is
not a continuous-time error certificate. No further refinement or amplitude
campaign is needed for the frozen decision.

The minimum observed acute margin was `0.31220729`, the largest instantaneous
work-balance residual was `2.51e-18`, and the largest Euler storage-step
defect was `5.52e-9`. These are different quantities. The static rounded
reference had maximum rate `1.95e-17`; that drift was not subtracted.
Execution used Python 3.13.6, NumPy 2.5.3, NetworkX 3.6.1, Windows AMD64,
binary64 states, ascending node order, sorted unit edges and no randomness.
The archive retains all 606 TNFR Python source files plus this producer;
dependency versions are recorded separately. The
[record tests](../../tests/physics/test_relational_mediation_response.py)
check frozen evidence without rerunning its trajectories.

This closes the bounded F4 question: a derived intermediary memory predicts
a finite response and its capacity intervention better than the stated
memory-omitting controls. It establishes neither a new primitive edge nor
a unique law of physical binding. A ring's response can be causally linked
to another without a direct edge, while the fine paths carrying that
interaction remain premises of the model.

<a id="mediated-geometric-boundary"></a>
## 11. Return-path geometry and the formation boundary

The present 11-node, 12-edge graph has cycle rank two: its independent cycles
are the two supplied C5 rings. Both edges along the mediator path are graph
bridges. At an equilibrium, sum `grad(V_phi)=0` over one side of either
bridge. Internal sine terms cancel and the only remaining term is the sine
of its phase gap. Strict acute admission then forces that gap to be zero.
Thus the current equilibrium has inherited ring periods and aligned connecting
phases, but no additional independent inter-region circulation period.

This does not exclude a composite NFR or stable binding on this topology.
An additional cycle is only one possible geometric discriminator, not a
necessary condition for every collective relation. Conversely, repeating
the local recovery test cannot manufacture that extra cycle invariant.

A supplied return path changes the cycle space. For example, adding
`1--6` to the current graph supplies a five-edge mixed cycle, whereas the
older two-direct-bridge geometry has a four-edge mixed cycle. Strictly acute
gaps force the latter's integer period to zero; a longer cycle merely removes
that elementary length obstruction. It does not prove a nonzero period is
compatible with the two existing ring periods or with the critical sine
balance. The following admission solves those simultaneous constraints without
a trajectory search. Supplying a return path still does not explain the origin
or timing of a primitive edge.

<a id="return-path-equilibrium"></a>
### 11.1 Complete state, support and equilibrium equations

Keep both oriented C5 rings `0->1->2->3->4->0` and `5->6->7->8->9->5`,
the path `0->10->5`, and add the supplied unit edge `1--6`. This connected
11-node, 13-edge support has cycle rank three. Use the same relational law,
held strictly positive capacities, `e,w,beta>0`, no forcing or events, and
strictly acute gaps. These are constitutive and support premises, not a
derived rule for creating the return edge.

The phase row at equilibrium implies `Bx=0`, hence uniform form. The form
row then requires `g=0`. In the acute domain, every neighbor resultant has
positive real part relative to its node; therefore `g=0` is equivalent to
zero neighbor sine sum. This reuses the algebraic circulation and integral
period criterion of [support balance, Section 30](../FORCED_SUPPORT_BALANCE.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods).
Its separately supplied sine phase evolution is **not** imported into this
relational model. Positive capacities affect recovery rates, not these
equilibrium equations.

Choose the two ring cycles and the mixed cycle `0->10->5->6->1->0`.
They form an integer cycle basis: the edges `1->2`, `6->7` and `0->10`
give an identity submatrix, and removing them leaves a spanning tree.
Write their sine-circulation coordinates as `a,b,q`. The special ring edges
`0->1` and `5->6` carry `a-q` and `b+q`; the other four edges of each ring
carry `a` and `b`. The three connecting edges, oriented `0->10`, `10->5`
and `6->1`, carry `q`. Thus the necessary and sufficient equations are

\[
\begin{aligned}
\arcsin(a-q)+4\arcsin a&=2\pi,\\
\arcsin(b+q)+4\arcsin b&=2\pi\sigma,\\
3\arcsin q+\arcsin(b+q)-\arcsin(a-q)&=2\pi m,
\end{aligned}
\]

where `sigma=+1` or `-1`, `m` is an integer, all currents have magnitude
strictly below one, and arcsines use their acute principal branch.
The possible mixed period must be solved jointly, not assigned by cycle length.

### 11.2 Existence, uniqueness and the forced zero mixed period

For a positive ring period, write its special angle as `t`. The period equation
forces `0<t<pi/2`, bulk angle `A=pi/2-t/4`, and bulk sine `cos(t/4)`.
Define

\[
Q(t)=\cos(t/4)-\sin t,\qquad
Q'(t)=-\tfrac14\sin(t/4)-\cos t<0.
\]

Its range is `(-d,1)`, where `d=1-cos(pi/8)<1/2`.

**Equal ring periods `(1,1)`.** If the other special angle is `s`, then
`q=Q(t)=-Q(s)`, so `abs(q)<d`. The mixed angle sum
`3*arcsin(q)+s-t` has magnitude below `pi/2+3*arcsin(d)<pi`.
Consequently `m=0`. For nonzero `q`, monotonicity makes `s-t` have the
same sign as `q`, so their sum cannot vanish. The unique solution is

\[
q=0,\qquad t=s=2\pi/5,\qquad a=b=\sin(2\pi/5).
\]

Both original twists survive unchanged; all three connecting gaps vanish.

**Opposite ring periods `(1,-1)`.** Write the right special angle as `-s`.
Then `q=Q(t)=Q(s)`, hence `s=t` and `b=-a`. The mixed sum
`3*arcsin(Q(t))-2*t` lies strictly between `-3*pi/2` and `3*pi/2`.
Again only `m=0` is possible. Therefore the connecting angle is `u=2*t/3`
and the remaining scalar equation is

\[
\boxed{F(t)=\cos(t/4)-\sin t-\sin(2t/3)=0.}
\]

Here `F'(t)=-sin(t/4)/4-cos(t)-(2/3)*cos(2t/3)<0` throughout
`(0,pi/2)`. Its signs bracket a unique root `t_*`:

\[
\pi/6<t_*<\pi/4.
\]

For an elementary endpoint proof, `pi<4`, `cos(z)>=1-z*z/2` and
`sin(z)<z` give `F(pi/6)>71/72-1/2-4/9=1/24>0`.
At the other endpoint, `F(pi/4)<1-1/sqrt(2)-1/2<0`.
Continuity and strict monotonicity establish existence and uniqueness
independently of a numerical solver.

Set `u=2*t_*/3` and `A=pi/2-t_*/4`. A nodal reconstruction, modulo `2*pi`, is

\[
\begin{aligned}
\theta_L&=(0,t_*,t_*+A,t_*+2A,t_*+3A),\\
\theta_R&=(2u,2u-t_*,2u-t_*-A,2u-t_*-2A,2u-t_*-3A),\\
\theta_{10}&=u.
\end{aligned}
\]

It has periods `(1,-1,0)` and nonzero connecting sine circulation
`q=sin(u)>0`. The largest absolute gap is `A`, so its minimum acute margin
is exactly `t_*/4`, between `pi/24` and `pi/16`. A zero mixed period
therefore does not imply zero connecting circulation.

### 11.3 Recovery, observable distinction and storage cost

For either exact equilibrium, the existing
[local recovery theorem](RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
applies: a sufficiently small joint form/phase perturbation recovers
exponentially modulo one common form offset and one common phase rotation.
There are twenty stable quotient directions. The statement admits unequal
strictly positive held capacities. It supplies neither a quantified basin
here nor a capture certificate for attachment from the previous support.

The independent reflection of one ring used in Section 9 is no longer a
symmetry: it sends the supplied return edge to an absent edge. Only the
identity and whole-ring exchange preserve this graph. In particular, the
mediator is its unique degree-two node adjacent to two degree-three nodes;
fixing it fixes or swaps both ring ports, leaving no independent reflection.
The connecting gap distinguishes the two equilibria: zero versus `2*t_*/3`.
This is an actual geometric distinction, not merely the loss of a symmetry
argument. Neither equilibrium has nonzero pressure or nodal rates. Its sine
circulation is an algebraic balance quantity, not dissipative transport or
an identified magnetic current.

Connection interpreted as a causally shared pulse remains a dynamical
hypothesis. These equilibria provide reference geometries, not persistent
oscillations. The existing [pulse boundary](RELATIONAL_EXCHANGE_ADMISSION.md#relational-pulse-scope)
distinguishes damped collective motion from a nontrivial recurrent full state
under strict unforced dissipation. Moreover, freezing the mediator on this
new support leaves the direct return edge active: the null of Section 10
cannot be transferred to this graph as a no-influence control.

Their phase storage is

\[
\begin{aligned}
V_+&=10(1-\cos(2\pi/5)),\\
V_-&=8(1-\sin(t_*/4))+2(1-\cos t_*)+3(1-\cos(2t_*/3)),\\
V_-&>V_+.
\end{aligned}
\]

The strict inequality needs no numerical fit: `1-cos(z)` is strictly convex
on the acute interval. Each ring's five signed gaps have fixed sum `+2*pi`
or `-2*pi`, so Jensen's inequality bounds its storage below by that of the
uniform twist. The opposite solution has nonuniform ring gaps and positive
connecting storage. This comparison does not select a winding or imply that
the dynamics can change sectors while remaining acute.

It also gives a **formation obstruction**. The exact opposite-winding
equilibrium on the original mediator-only support has uniform form and storage
`beta*V_+`. A zero-supply passive event, including a simultaneous nodal reset,
followed by unforced fixed-support relational evolution cannot converge to
the new opposite equilibrium with greater storage `beta*V_-`. Even immediate
formation of that target requires excess initial storage or declared supply
of at least `beta*(V_--V_+)`; dissipation adds its own nonnegative cost.
An available budget is only necessary, not a selector or a sufficient
capture condition. Passivity is an additional event premise throughout.
For equal windings, inserting the zero-gap return edge is storage-neutral
and preserves the old equilibrium, but no occurrence time follows.
Simply adding the edge to the old opposite equilibrium also fails the
implemented phase domains: its gap is `-4*pi/5`, and each new port resultant
has relative real part `2*cos(2*pi/5)+cos(4*pi/5)=(sqrt(5)-3)/4<0`.
This violates both acute and positive-resultant admission; it does not rule
out a separately declared state-reorganizing event or every regular chart.

### 11.4 Static computational contract

The [return-geometry instrument](../../benchmarks/relational_return_geometry.py)
reuses the engine's exact cycle reconstruction, one-chord topology extension,
rational interval trigonometry and detached relational field. It encloses
the scalar root in turns, with certified opposite endpoint signs, and
constructs a rational midpoint witness for represented native checks.
Exact periods and strict acute admission hold for that witness; its sine
balance remains explicitly unresolved by the rational reconstruction owner.
The exact equilibrium belongs to the analytic root above. Small binary64
rates at the midpoint are reported residuals, not a zero-rate proof.
With forty bisections, the exact turn bracket is
`[2619154484885/26388279066624, 436525747481/4398046511104]`.
Its midpoint gives approximately `t=0.6236341875541443` radians,
connecting gap `0.4157561250360962` and acute margin `0.1559085468885361`.
The ideal storage-excess interval is contained in `[0.47999183896,0.47999183898]`;
these are structural units with `beta=1`, not laboratory energy measurements.
The instrument also reports native midpoint residuals without subtracting
represented drift. Run it with `python -m benchmarks.relational_return_geometry`;
the default preparation fixes unit capacities, `e=w=1/2`, `beta=1`, ascending
node order and no randomness. The analytic equilibrium and storage comparison
retain the wider parameter hypotheses above.
The [static controls](../../tests/physics/test_relational_return_geometry.py)
also reject simply copying the old opposite twists onto the added edge.
No trajectory, automatic connection, new pressure law or physical bridge
is introduced by this instrument.

<a id="shared-collective-pulse"></a>
## 12. A causally shared oscillatory response, with a dissipation limit

**Question and scope.** Can the admitted form/phase dynamics itself generate
an oscillatory response shared by the two regions? Here a local collective
oscillation means a nonreal mode of the native equilibrium derivative that
contributes to a donor-to-receiver response. Matching frequencies or an
eigenvector drawn across both rings is insufficient. A maintained pulse would
add a separate persistence obligation. Neither meaning is a definition of
every possible NFR or a physical identification.

Fix the opposite-winding equilibrium of Section 11, unit capacities,
`e=w=1/2`, `beta=1`, its existing structural clock and no inputs or events.
The graph, constitutive law and preparation remain explicit premises.
An initial donor perturbation probes the response; it is not a periodic
forcing or an autonomous explanation of how that perturbation arose.

### 12.1 The oscillatory mechanism uses the existing two rows

Let `B` be the unweighted Laplacian, `D` the degree diagonal, and `K` the
Laplacian weighted by the equilibrium edge cosines. At this equilibrium
`H_i=pi*sum_j cos(delta_ij)>0`. With `u=delta x`, `v=delta theta`, the
[existing recovery derivative](RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
is

\[
\frac{d}{dt}\begin{pmatrix}u\\v\end{pmatrix}
=J\begin{pmatrix}u\\v\end{pmatrix},\qquad
J=\begin{pmatrix}
-eD^{-1}B&-wH^{-1}K\\
(w/\beta)H^{-1}B&0
\end{pmatrix}.
\]

Phase deformation changes pressure and hence form; form contrast drives
phase motion back. This is a derived feedback within the supplied nodal law,
not an additional oscillator equation or a prescribed frequency. Its restoring
and damping rates depend on geometry through `B,K,H,D` and on the declared
model coefficients and clock. The same coefficients can be overdamped on
P2; oscillation is not guaranteed merely by naming a phase coordinate.

The shared engine now provides `evaluate_relational_uniform_tangent` and
`Network.relational_uniform_tangent`. It admits exactly uniform represented
form even away from phase equilibrium, retaining the native field and the
general Arg derivative. At an ideal equilibrium that derivative reduces to
`-H^-1 K`. The observer does not declare equilibrium from a small residual,
repair the numerical common-offset defect, or change the time-evolution law.
Its [contract](../../docs/contracts/RELATIONAL_DYNAMICS.md#uniform-form-tangent-observation)
separates materialized coefficients from exact derivatives and spectral proof.

### 12.2 A nonzero transfer is stronger than a shared eigenvector

Let `a` be a declared initial perturbation and `O_R` an observation of the
receiving ring. The Laplace transform of its tangent response is

\[
T_R(s)=O_R(sI-J)^{-1}a.
\]

For a simple eigenvalue, a right vector `v_k` and a left row `l_k` normalized
by `l_k v_k=1` give residue `O_R v_k (l_k a)`. Both preparation and observation
must participate. This expression is invariant to eigenvector rescaling;
using a Hermitian Laplacian projector for this generally nonnormal `J` would
give a different calculation. Degenerate independent oscillators permit
extended eigenvectors while their cross-region transfer is exactly zero.

There is also an exact causal result for the ideal return graph. Order the
two coordinates by node. Every off-diagonal edge block from node `j` to `i` is

\[
J_{ij}=\begin{pmatrix}
e/d_i&w\cos\delta_{ij}/H_i\\
-w/(\beta H_i)&0
\end{pmatrix},\qquad
\det J_{ij}=\frac{w^2\cos\delta_{ij}}{\beta H_i^2}>0.
\]

Suppose a right eigenvector vanishes on the entire receiving ring. Its
equations at nodes 5 and 6 force respectively node 10 and node 1 to vanish;
the equation at 10 then forces node 0. The equations at 0, 1 and 2 force
nodes 4, 2 and 3 in turn. No nonzero eigenvector can be invisible on that
ring. The transpose argument starting with a left eigenvector zero on the
donor forces nodes 10, 6, 5, 9, 7 and 8 to vanish. Thus complete donor
form/phase preparation and complete receiver observation meet the eigenvector
controllability/observability criteria: no mode is absent from the full
donor-to-receiver transfer. This does not assert that every single-node
preparation or scalar observation detects every mode, nor that the mediator
moves in every mode.

For the numerical comparison, fix `a=e_x0` before evaluation. The primary
receiver outputs are `x_5-mean_R(x)` and `v_5-mean_R(v)`. Mediator outputs
are its two **rates**, avoiding a spurious apparent motion from subtracting
a changing common reference. No phase angle derived from a mode is identified
with primitive nodal phase.

The causal null sets cross-region blocks of the full tangent to zero,
retaining its three diagonal blocks for donor, receiver and mediator. This
is an explicit linear ablation with held local coefficients, not a newly
admitted disconnected nonlinear geometry. Its donor-only subspace is
invariant, so receiver response is identically zero for every time and every
resolvent point outside the spectrum, even if eigenvalues are degenerate.
Its offsets need not have the native quotient symmetry; use the complete
22-coordinate null. Merely freezing node 10 leaves edge `1--6` active and
cannot serve as that null.

### 12.3 Certified existence of an ideal damped complex mode

A scalar spectral moment proves presence without certifying individual
numerical eigenvalues. Write `L=D^-1 B` and `M=H^-1`. Block multiplication
at the exact equilibrium gives

\[
\operatorname{tr}J^3=-e^3\operatorname{tr}L^3
 +\frac{3ew^2}{\beta}\operatorname{tr}(LMKMB).
\]

The graph is triangle-free. Its degrees give
`sum_edges 1/(d_i*d_j)=7/3`, so `tr(L^3)=11+6*(7/3)=25`.
Set `r=cos(t_*)`, `z=sin(t_*/4)`, `b=cos(2*t_*/3)` and `s=r+z+b`.
The cosine strengths `s_i=H_i/pi` are `s` on four ports, `2*z` on the
six internal ring nodes and `2*b` on the mediator. Expanding the remaining
trace into diagonal and two-edge walks gives

\[
\begin{aligned}
T:=\pi^2\operatorname{tr}(LMKMB)
 &=\sum_i\frac{d_i}{s_i}
  +\sum_{\{i,j\}\in E}\left(\frac1{d_i s_j}+\frac1{d_j s_i}
                          +\frac{4\cos\delta_{ij}}{s_i s_j}\right)\\
 &=\frac{29}{s}+\frac{38}{3z}+\frac4{3b}
                         +\frac{4(2r+b)}{s^2},\\
\operatorname{tr}J^3&=-\frac{25}{8}+\frac{3T}{8\pi^2}.
\end{aligned}
\]

The exact root bracket of Section 11 and shared outward rational trigonometry
enclose this ideal trace strictly inside `[0.72428869449,0.72428869451]`.
No numerical eigensolver or trajectory enters that sign certificate.
The two common-offset eigenvalues are zero; all twenty quotient eigenvalues
have negative real part by local recovery. If they were all real, the sum
of their cubes would be negative. The certified positive trace contradicts
that possibility. Therefore at least one damped complex-conjugate pair exists
for this exact geometry and model. Section 12.2 further proves that some donor
preparation and receiver observation retain such a pair causally.

This theorem does not count all complex pairs, certify individual frequencies,
or prove that this same pair moves the mediator. Those more specific statements
retain their separate numerical evidence below. The certificate is conditional
on the stated geometry and coefficients, not a universal pulse law for TNFR.

### 12.4 Persistence is a separate claim

All nonneutral modes of the ideal acute equilibrium have negative real part
by the existing recovery theorem. More strongly, the
[storage recurrence argument](RELATIONAL_EXCHANGE_ADMISSION.md#relational-pulse-scope)
excludes nonstationary full-state periodic or relative-periodic motion in the
specified regular chamber under strictly positive dissipation and capacity.
These statements do not exclude every recurrent lossy diagnostic or all
other TNFR completions. They do exclude using this model as evidence of a
permanent unforced full-state pulse.

A mode with `lambda=-alpha+i*omega` contributes a damped sinusoid to the
infinitesimal response. Its period `2*pi/abs(omega)` and amplitude decay time
`1/alpha` have units of the declared structural clock. Neither is a calibrated
physical time. A sum of such modes need not visibly complete a cycle; a
nonzero complex residue alone is not a nonlinear limit cycle or phase-locking
theorem. For a fixed finite horizon, smooth dependence supplies the usual
first-order response `epsilon*exp(J*t)*a` about the exact equilibrium, with
a local higher-order remainder; no numerical remainder bound is certified here.

### 12.5 Retained static evidence and numerical boundaries

The [collective-pulse instrument](../../benchmarks/relational_collective_pulse.py)
uses the shared uniform-form tangent and the same forty-bisection rational
midpoint as Section 11. Its
[static record](../../docs/assets/relational_collective_pulse/result.json)
retains the complete state, matrices, declared observations, initial direction,
all poles and residues, controls, root enclosure and ideal trace certificate.
There is no finite-amplitude evolution or fitted frequency in this calculation.

Common form and phase offsets are removed by the exact difference map
`D_0 z=(x_i-x_10, v_i-v_10)`, `i=0,...,9`, with lift `L_0` setting the
mediator coordinates to zero. Here `D_0` is an observation map, distinct from
the degree diagonal `D` above. `D_0 L_0=I`; the ideal law induces
`J_q=D_0 J L_0`. The materialized generator's nonzero row sums and the defect
`D_0 J-J_q D_0` are retained, not silently corrected. Their maximum magnitude
was `4.17e-17`. This numerical quotient is not asserted to close the represented
generator exactly.

The numerical quotient has six conjugate pairs and eight real poles. The
positive-imaginary representatives are listed below; the report retains their
conjugates and the real modes as well. "Unresolved" means the mediator-rate
residue magnitude is below the stated numerical resolution, not a missing
coordinate or a proof of zero motion.

| Real part | Positive imaginary part | Period | Decay time | Mediator-rate response |
| --- | --- | --- | --- | --- |
| -0.438279 | 0.530782 | 11.8376 | 2.28165 | Unresolved |
| -0.438120 | 0.534008 | 11.7661 | 2.28248 | Resolved |
| -0.265023 | 0.269281 | 23.3332 | 3.77326 | Resolved |
| -0.242613 | 0.272205 | 23.0825 | 4.12179 | Unresolved |
| -0.134737 | 0.050636 | 124.0860 | 7.42187 | Resolved |
| -0.045952 | 0.058180 | 107.9960 | 21.76184 | Unresolved |

All six have resolved primary receiver residues for the declared donor form
direction. For example, the `-0.265023+0.269281i` component has receiver
form/phase residue magnitudes approximately `(0.0438520,0.0273921)` and
mediator-rate magnitudes `(0.0195353,0.00876979)`. Its amplitude falls to
`exp(-1)` in only `0.161712` cycles; after one period its modal envelope is
approximately `0.002063` of its initial value. The existence of an oscillatory
component therefore should not be presented as a long train of shared pulses.

Numerical pole and residue zero policies were `5.71e-14` and `6.93e-13`,
respectively, based on binary64 precision, matrix/observation scale and
eigenbasis conditioning. They are resolution policies, not physical thresholds.
The eigenbasis condition number was `24.35`, the smallest distinct-pole
separation `0.00323028`, and the maximum eigenvector residual below `3.44e-15`.
Reassembling the full transfer at the declared `s=1+2i` differed from direct
solution by `1.06e-16`; full and quotient resolvents differed by `1.56e-17`.
These are consistency checks, not rigorous error bounds on individual ideal
poles or residues. The certified trace sign remains separate evidence.

The all-route block-diagonal control has identically zero receiver transfer.
Freezing only the mediator does not: the receiver's second derivative per
unit donor direction is approximately `(-0.00424688,0.00281912)` through the
remaining return route. This is why apparent shared timing and intervention
on just one route cannot settle connection.

Execution used Python 3.13.6, NumPy 2.5.3, NetworkX 3.6.1, Windows AMD64,
binary64 native coefficients and eigensystems, exact rational observation
maps/defects, the shared interval kernel, and no randomness. Listed owner
fingerprints are a **partial** source manifest, not a complete dependency
archive or provenance authentication. The stored matrices permit independent
checks without regenerating the producer. Its
[tests](../../tests/physics/test_relational_collective_pulse.py) also use an
analytic oscillator transfer and disconnected degenerate oscillators as
independent controls; those test matrices are not added to TNFR dynamics.

This closes local pulse admission: oscillatory causal participation follows
under the specified law and geometry, while the individual residues and
strong damping have the stated numerical scope. A visibly oscillating finite
nonlinear response and a sustained pulse remain distinct further claims.

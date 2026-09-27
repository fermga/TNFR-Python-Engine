# Hidden-state memory and a controlled local pattern approximation

**Status:** exact nonlinear quotient and hidden-state identity, a conditional
finite-horizon error theorem, and a derived prospective cubic forecast. No
numerical neighborhood or derivative bounds are certified here. The result
uses the same supplied relational law as
[local composition](RELATIONAL_PATTERN_COMPOSITION.md); it adds no state
variable, delay law, controller or physical identification. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
remains the sole task queue.

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

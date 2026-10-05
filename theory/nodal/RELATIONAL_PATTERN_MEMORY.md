# Pattern memory and mediated collective geometry

**Status:** conditional state reduction, memory, interaction, recovery and
identity results. Each section declares its complete law, support, preparation
and evidence scope. Native neighbor-argument dynamics and the normalized-sine
comparison share mathematical tools; their trajectories and certificates are
not interchangeable. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
is the sole research queue.

## Reading map by model and question

| Sections | Model and retained structure | Result and boundary |
| --- | --- | --- |
| [1-8](#1-exact-symmetry-quotient-ten-even-and-eight-odd-coordinates) | Native single-bridge two-C5 pattern | Exact symmetry quotient, nonlinear closure obstructions and conditional memory approximations. Earlier finite responses do not certify their recovery neighborhoods or continuous numerical error. |
| [9-12](#mediated-pattern-interaction) | Native interacting patterns with explicit intermediaries, paths or a return edge; separately marked sine comparisons | Hidden-state memory, controlled local fast limits, stationary composition, three-port interaction and causal oscillatory response. Distinct supports and reductions cannot be substituted for each other. |
| [13-17](#relational-retained-phase-memory) | Native isolated C5, followed by supplied contact/removal between prepared rings | Preparation-family capture, limiting phase record, finite-amplitude remainder, contact readout and retained receiver mean. Event work, no-contact controls and recovery premises remain explicit. |
| [18-19](#sine-relative-pattern-state) | Complete normalized-sine law, relative state and positive-loss recovery | Moving-reference correction, whole-set cycle recovery and a full two-C5/live-intermediary basin with causal response. An interacting region cannot be certified by deleting its environment. |
| [20-22](#sine-formation-eligibility) | Sine donor/intermediary/initially flat receiver on eleven supplied nodes | Formation barriers, endpoint classification and prepared capture/exclusion have distinct premises. Use the question routes below; a possible equilibrium is not an accessible endpoint. |
| [23](#sine-conservative-identity) | Conservative sine law on an admitted isolated cycle | Open invariant geometry protects identity for every member and supports recurrence almost everywhere. No attraction, chosen-state return time or shared waveform follows. |

For state and conservation questions, start with the
[exact relative state and sine bracket](#sine-relative-pattern-state),
[hidden inventory and memory source](#conserved-hidden-inventory), or
[local asymptotic-origin coordinate](#asymptotic-origin-coordinate-scope).
These distinguish a symmetry quotient, a linear conserved total and a nonlinear
coordinate of a selected recovering law. For incomplete observations, follow
[hidden-state inference](#sine-hidden-state-observability),
[capacity inference](#sine-hidden-capacity-observability),
[joint prior forecast](#sine-prior-reserved-forecast) and
[finite-sample admission](#sine-finite-sample-admission).

For the eleven-node receiver problem, choose the actual obligation:

- **Possible endpoints:** [asymptotic classification](#sine-eleven-node-asymptotic-equilibria)
  and [local stability](#sine-eleven-node-equilibrium-stability).
- **Necessary transfer conditions:** [target admission](#sine-receiver-transfer-admission),
  [source excitation](#sine-source-receiver-excitation) and
  [running port work](#sine-receiver-port-passage).
- **Prepared donor capture:** [donor well](#sine-donor-well-retention) and
  [early dissipation](#sine-donor-dissipative-capture).
- **Receiver exclusions:** [weighted nonlinear bound](#sine-weighted-receiver-exclusion)
  and [validated prefix with dissipative tail](#sine-localized-receiver-exclusion).

Each owner states its source family, thresholds and unresolved boundary. These
routes locate retained evidence; they do not activate a receiver campaign.

The [replica scale owner](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-inheritance)
separately retains internal constituents at a collective scale, including its
prepared pulse and transverse instability. Reusing memory or a storage barrier
there requires the matching law and full-support hypotheses. None of these
results derives support birth, a controller or physical identification.

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

<a id="sine-mediator-common-geometry"></a>
### Common mediator geometry does not select its transmitted pressure

The [complete-law comparison](RELATIONAL_EXCHANGE_ADMISSION.md#global-closure-pressure-comparison)
permits a direct test of whether native argument pressure is an effective
version of primitive pairwise sine exchange. Retain one supplied intermediary
`m`, attached to `k>=2` visible ports, and any reciprocal unit edges among
visible nodes. Use the sine law with held capacities, `nu_m=mu>0` and
`e>=0,w,beta>0`. Visible degrees retain their original mediator incidence.
No edge is created by the following elimination.

Set `Z=sum_p exp(i*theta_p)=R*exp(i*Psi)` with `R>0`. The conditional
storage minimum is again `x_m=bar(x),theta_m=Psi`, now without requiring
every incident gap to be acute. It is the unique **minimum** modulo a turn,
not the only stationary state: `theta_m=Psi+pi` is a maximum of the
hidden phase cost and a saddle of the hidden dynamics when `e>0`.
At `R=0,x_m=bar(x)` every hidden phase is stationary and no direction is
selected.

At the minimum the visible gradients are

\[
\widetilde q_p=q_{{\rm internal},p}+x_p-\bar x,\qquad
\widetilde S_p=S_{{\rm internal},p}
 +\frac{\sum_r\sin(\theta_r-\theta_p)}R ,
\]

with internal gradients unchanged at nonports. The visible law is

\[
\dot x_i=\nu_i(-e\widetilde q_i/d_i+
 w\widetilde S_i/(\pi d_i)),\qquad
\dot\theta_i=w\nu_i\widetilde q_i/(\beta\pi d_i).
\]

This is the full sine field evaluated at the conditional minimum. In
particular, the transmitted phasor contributes `sin(Psi-theta_p)`, not
`Arg(Z)` as a force. Where the complete reconstructed native state is also
admitted, both primitive candidates have the same conditional geometry and
inherited storage,

\[
E_{\rm eff}=E_{\rm internal}
 +\frac1{2k}\sum_{p<r}(x_p-x_r)^2+\beta(k-R).
\]

The hidden gradients vanish, so the envelope chain rule and visible exchange
cancellation give `Edot_eff=-e*sum_visible nu_i*q_tilde_i^2/d_i`.
This is a balance of the conditional reduced equation. It is not equality
to a general finite-capacity full trajectory's storage.

**Collective response survives primitive additivity.** The previous
nonpairwise storage proof applies to `k=3` unchanged. The actual sine
port response is also collective. In the same three-C5 preparation, hold A
fixed, rotate B and C by `s,t`, keep all forms zero and visible capacities
one. With `Psi=Arg(1+exp(i*s)+exp(i*t))`,

\[
\dot x_A=\frac w{3\pi}\sin\Psi,\qquad
\left.\partial_s\partial_t\dot x_A\right|_{s=t}
 =-\frac{2w\sin t(1-\cos t)(2+\cos t)}
 {3\pi(5+4\cos t)^{5/2}}<0\quad(0<t<\pi/4).
\]

Independent pair responses would have zero mixed derivative. The native
mixed derivative in the preceding subsection is positive. Thus a primitive
additive law can yield effective collective interaction, but this mediator
does not make the two effective laws equal. The sign comparison uses a
frozen family of initial states and instantaneous fields, without fitting
or running a new response. It is not a physical identification.

**Finite capacity retains a tracking defect.** Let
`u=x_m-bar(x),v=theta_m-Psi` on a local lift, `r=R/k` and
`alpha_p=theta_p-Psi`. The exact moving-boundary equations are

\[
\begin{pmatrix}\dot u\\\dot v\end{pmatrix}
=\mu\begin{pmatrix}-eu-(w/\pi)r\sin v\\wu/(\beta\pi)\end{pmatrix}
-\begin{pmatrix}\dot{\bar x}\\\dot\Psi\end{pmatrix},\qquad
\dot\Psi=\frac1R\sum_p\cos\alpha_p\,\dot\theta_p .
\]

On `u=v=0` the defect is the negative velocity of the moving conditional
minimum, generally nonzero. For example, on a two-port path take endpoint
`(x,nu)=(1,1),(0,2)`, all phases zero and `x_m=1/2`. The hidden rows
vanish, but `(udot,vdot)=(-e/4,w/(4*beta*pi))`. A freshly minimized
environment therefore does not remain minimized merely because its
instantaneous hidden rates vanish.

**A damped-pendulum equation follows from these two rows.** Freeze the visible
boundary, for example by zero visible capacity, and use fast time
`tau=mu*t`. Eliminating `u` gives exactly

\[
\frac{d^2v}{d\tau^2}
 +e\frac{dv}{d\tau}
 +\frac{w^2R}{\beta\pi^2 k}\sin v=0,\qquad
\frac{d}{d\tau}\left[\frac k2u^2+\beta R(1-\cos v)\right]
 =-ek u^2 .
\]

This damped-pendulum form is derived from the supplied nodal comparison;
no oscillator is added to it. Its form alone does not establish oscillatory
decay. A moving environment supplies the explicit
boundary terms above. At the minimum the fast exponents solve

\[
\lambda^2+e\lambda+\frac{w^2r}{\beta\pi^2}=0
\quad\hbox{(sine)},\qquad
\lambda^2+e\lambda+\frac{w^2}{\beta\pi^2r}=0
\quad\hbox{(native)}.
\]

The native comparison retains admission of every visible resultant; a
positive hidden resultant alone does not supply that premise.
For `R>0,e>0` the minimum attracts a sufficiently small hidden perturbation;
oscillatory versus real decay depends on the discriminant. Decreasing
resultant strength softens the sine restoring term, whereas the native
tangent term grows. At cancellation the sine full field is still smooth,
but its conditional direction and uniform attraction are lost. A singular
reduced description can therefore coexist with regular fine dynamics;
that fact does not derive native Arg pressure or its singular response.

**A controlled fast limit can be proved with the existing method.** At
`e=w=1/2,beta=1` require `|alpha_p|<=1/4,|v|<=1/4` in a compact
smooth tube about the new reduced visible solution. Then
`r*sinc(v)>=2945/3072`. Relative to the existing `A_0` and `P` in the
[fast-mediator proof](#fast-mediator-reduction), the sine remainder obeys
`||R_s(z)||<=127*||z||/18432`. This follows from
`1-r*sinc(v)<=127/3072` and `1/(2*pi)<1/6`. Since `||P||<14`,
`2*||P||*127/18432<1/2`, which reproves the same sufficient quadratic
decay estimate for this different law.

With `W=sqrt(z^T*P*z)`, newly justified bounds `B,L` for its moving
boundary and visible field, and the same first-exit hypotheses, it follows
that

\[
W(t)\le e^{-\mu t/56}W(0)
 +\frac{784B}{\mu}(1-e^{-\mu t/56}),\qquad
\sup_{[0,T]}\|y-y_*\|
 \le\frac{Le^{LT}}{\mu}[56W(0)+784BT].
\]

Require `W(0)<rho<1/4`, `784B/mu<rho` and the visible bound strictly
below the chosen tube margin. These conditions prevent a first exit as in
the earlier proof. Its native numerical constants `B,L`, trajectory,
clock samples and records do not transfer. The estimate is finite-horizon,
includes the initial layer and is not uniform near `R=0`.

The [detached sine mediation owner](../../src/tnfr/physics/relational_sine_mediation.py)
encloses the conditional visible field, storage, work and tracking defects
using captured represented inputs. It uses `Z/R` directly instead of
rounding `Psi` into a new graph phase. A certified positive lower bound on
`R` is mandatory; a zero-containing enclosure is unresolved and rejects
this reduction even though the fine sine law remains defined.
The [controls](../../tests/physics/test_relational_sine_mediation.py)
verify the independent field, inherited work, mixed response, hidden
oscillator and noninvariance. The report advances no trajectory, certifies
no fast-limit tube and selects no primitive pressure law.

<a id="finite-environment-reuse-audit"></a>
### Finite environmental state, cancellation and causal-response reuse

The stationary-minimum reduction and the older memory/chart owners supply
a more precise next calculation than assuming an instantaneous pressure.
Keep the sine comparison, one supplied mediator of capacity `mu>0`,
fixed unit incidence and the original signed form and circular phase.
The following are conditional results of that complete law.

#### A faithful state that remains regular at cancellation

The [existing cylinder encoding](JOINT_PARAMETER_RESPONSE.md#93-a-faithful-encoding-without-a-new-law)
retains signed form and unit phase separately. Put
`X=mean(x_ports)`, `Z=A+i*B=sum_p exp(i*theta_p)`, `u=x_m-X` and
`(c,s)=(cos(theta_m),sin(theta_m))`. Direct substitution into the sine
rows gives

\[
\dot u=\mu[-eu+\tfrac{w}{\pi k}(Bc-As)]-\dot X,\qquad
\omega=\tfrac{\mu w}{\beta\pi}u,\qquad
\dot c=-\omega s,\quad \dot s=\omega c .
\]

The circle constraint `c^2+s^2=1` is preserved. This is a faithful encoding
of the two existing hidden coordinates, not three independent coordinates
or a new law. It never divides by `|Z|` or chooses `Arg(Z)` and remains
regular at cancellation. In a coupled network `X,A,B` and their derivatives
come from the visible nodal rows; they are not automatically externally
prescribed inputs. With a supplied differentiable visible path, the same
identity instead describes an explicitly driven hidden subsystem.

An absolute local lift of the hidden phase obeys

\[
\ddot\theta_m+\mu e\dot\theta_m
 =\frac{\mu^2w^2}{\beta\pi^2k}(B\cos\theta_m-A\sin\theta_m)
 -\frac{\mu w}{\beta\pi}\dot X .
\]

There is no derivative of an undefined mean angle here. Define
`H=k*u^2/2+beta*[k-(A*c+B*s)]`. Its exact work identity is

\[
\dot H=-\mu e k u^2-ku\dot X-\beta(\dot A c+\dot B s).
\]

The actual incident-edge storage additionally contains
`sum_p(x_p-X)^2/2`. Its derivative is equivalently

\[
\dot E_{\rm star}=-\mu ek u^2
 +\sum_p(x_p-x_m)\dot x_p
 +\beta\sum_p\sin(\theta_p-\theta_m)\dot\theta_p .
\]

Thus boundary work is explicit; a moving environment need not decrease
the hidden storage by itself. Full-network exchange retains its existing
global balance.

For a frozen visible boundary with `Z=0,e>0` there is an exact response:

\[
u(t)=u_0e^{-\mu et},\qquad
\theta_m(t)=\theta_{m0}
 +\frac{wu_0}{\beta\pi e}(1-e^{-\mu et}).
\]

The hidden form relaxes, but the limiting phase retains the initial hidden
orientation and form impulse. Positive capacity changes the relaxation time,
not this limiting phase. At `mu=0` the hidden state is frozen instead;
at `e=0` division by damping is unavailable and that formula is not used.
The infinite-time frozen-boundary conclusion requires the ports really to
remain fixed, for example through zero visible capacities.

A separate instantaneous visible-closure counterexample permits active
ports: take ideal port phases `(0,pi)`, all forms zero, and hidden phase
`+pi/2` or `-pi/2`. Both hidden rates vanish and both preparations have
the same visible state and storage. Yet the visible form-rate pair is
`(+nu_0*w/pi,-nu_1*w/pi)` or its negative. Their subsequent ports need
not stay frozen. This distinguishes stationary hidden state from dispensable
hidden state. The exact mathematical phases in this proof are not assertions
that represented `float(pi)` has an exact zero sine.

#### Static agreement hides different memory clocks

Let `rho=R/k>0`, `a=w/pi`, `b=w/(beta*pi)`. Linearize at one frozen
conditional minimum, retaining its geometry. If `h` denotes absolute hidden
perturbations and `eta=(delta X,delta Psi)` the visible displacement of that
minimum, the two hidden tangents are

\[
\dot h=D(h-\eta),\qquad
D_s=\mu\begin{pmatrix}-e&-a\rho\\b&0\end{pmatrix},\qquad
D_n=\mu\begin{pmatrix}-e&-a\\b/\rho&0\end{pmatrix}.
\]

This is a linearization, not a global nonlinear closure. Full native
comparison still requires every native resultant to be admitted. With zero
initial hidden perturbation the Laplace response is
`H(z)=(zI-D)^-1*(-D)`. For the sine model,

\[
H_s(z)=
\frac{\begin{pmatrix}
\mu ez+\mu^2ab\rho&\mu a\rho z\\
-\mu bz&\mu^2ab\rho
\end{pmatrix}}{z^2+\mu ez+\mu^2ab\rho}.
\]

Both models have `H(0)=I`. Their stationary reconstruction therefore cannot
discriminate them; their poles and finite responses can. Nonzero hidden
initial state adds `exp(Dt)*h(0)` and must be retained. In an autonomous
network the visible equations close the feedback around these rows; treating
`eta` as a freely set input would be a different preparation.

For `e>0` and `0<4ab*rho<=e^2` the sine slow decay rate satisfies

\[
\gamma=\frac{2\mu ab\rho}{e+\sqrt{e^2-4ab\rho}},\qquad
\frac{\mu ab\rho}{e}\le\gamma\le\frac{2\mu ab\rho}{e}.
\]

Hence large `mu` alone does not justify an instantaneous approximation
uniformly near cancellation; the weak-resultant relaxation scale involves
`mu*rho`. At the default `e=w=1/2,beta=1`,
`e^2-4ab*rho=1/4-rho/pi^2>0` for every `0<rho<=1`.
The local hidden sine modes at the frozen conditional minimum are therefore
distinct negative real modes,
not small-amplitude oscillations. This does not classify a whole network's
spectrum or finite nonlinear trajectory. Native hidden modes instead become
nonreal for `rho<4/pi^2`.

The pole ratio `det(D)/trace(D)^2` cancels positive hidden capacity and
a common affine clock scale: it is `ab*rho/e^2` for sine and
`ab/(rho*e^2)` for native. This is a prospective tangent discriminator
at declared geometry, not a physical measurement or a derived clock.

#### Reuse the signed memory owner, not diffusion-only shortcuts

The full sine field has a particularly simple derivative. With `B` now
denoting the support Laplacian and `K(theta)` the cosine-weighted Laplacian,
its form/phase Jacobian is

\[
J_s=\begin{pmatrix}
-eND^{-1}B&-(w/\pi)ND^{-1}K(\theta)\\
(w/(\beta\pi))ND^{-1}B&0
\end{pmatrix}.
\]

At a fixed equilibrium this supplies the joint tangent for the existing
linear-memory algebra. The native phase mobility must not be substituted
into this different law. At a nonequilibrium state the displayed derivative
is still correct, but freezing it does not give the exact evolving tangent
along a nonlinear trajectory.

For a fixed joint tangent partitioned as
`y'=A_v*y+B_v*h, h'=C_v*y+D_v*h`, the existing
[coordinate-memory owner](../../src/tnfr/mathematics/linear_observation.py)
retains `B_v*exp(D_v*t)*h(0)` and kernel
`K(t)=B_v*exp(D_v*t)*C_v` exactly at the symbolic level.
It evaluates rational blocks rather than a certified exponential.
The same owner's output-Krylov construction already supplies a finite
test for identically zero kernel: with `m` hidden coordinates,

\[
K\equiv0\ \Longleftrightarrow\
B_vD_v^jC_v=0\quad(j=0,\ldots,m-1).
\]

Taylor coefficients prove necessity and Cayley-Hamilton proves sufficiency.
Replace `C_v` by a particular `h(0)` to test its initial-source contribution;
the two tests are distinct. A zero `K(0)` alone is insufficient. No new
exponential or stability assumption is needed for this algebraic test.
The [static response controls](../../tests/physics/test_relational_environment_response.py)
reuse these owners on explicitly declared rational tangent families; their
exact coefficients are not promoted to exact transcendental TNFR coefficients.

The reversible positivity/Gram criterion in
[EPI memory](../DERIVED_EPI_MEMORY.md#4-kernel-positivity-and-the-exact-closure-criterion),
the P5 finite-history formula and the REMESH echo law have narrower,
different hypotheses. They cannot select or truncate this signed joint
kernel automatically. Likewise, the
[Jacobi result](../TNFR_VARIATIONAL_PRINCIPLE.md#1319-structural-closure-tests-exchange-jacobi-and-the-remaining-potential)
supplies a conditional mobility restriction only when Poisson structure is
an additional premise. Work cancellation alone does not select that premise.
These reuse boundaries replace speculative new closures with existing
state, algebra and independently checkable responses.

<a id="conserved-hidden-inventory"></a>

#### Conserved inventory constrains both memory and its initial source

For the preceding fixed linear block law, suppose the full quantity
`I=l_v^T*y+l_h^T*h` is conserved for every state. There is no input, and all
blocks and covectors are held. Equivalently,

\[
\ell_v^{\mathsf T}A_v=-\ell_h^{\mathsf T}C_v,
\qquad
\ell_v^{\mathsf T}B_v=-\ell_h^{\mathsf T}D_v.
\]

The exact hidden-state reconstruction therefore retains the inventory as

\[
I(t)=\ell_v^{\mathsf T}y(t)
 +\ell_h^{\mathsf T}e^{D_vt}h(0)
 +\int_0^t\ell_h^{\mathsf T}e^{D_v(t-s)}C_vy(s)\,ds=I(0).
\]

In particular the visible charge alone need not be constant. Its memory
kernel and initial source satisfy

\[
\ell_v^{\mathsf T}K(t)
 =-\frac d{dt}\left[\ell_h^{\mathsf T}e^{D_vt}C_v\right],\qquad
\ell_v^{\mathsf T}B_ve^{D_vt}h(0)
 =-\frac d{dt}\left[\ell_h^{\mathsf T}e^{D_vt}h(0)\right].
\]

These identities need neither stability nor an evaluated exponential.
At each derivative order `j>=0` they reduce to the exact matrix checks
`l_v^T B_v D_v^j C_v=-l_h^T D_v^(j+1) C_v` and the same expression with
`C_v` replaced by `h(0)`. Dropping a nonzero hidden initial source changes
the visible dynamics. Its contribution to the chosen inventory requires a
separate projection: a charge-neutral source can still drive visible motion.
For unrestricted hidden states, the full inventory is a function of the
visible coordinates alone only if `l_h=0`; a constrained preparation can
supply a separate relation, which must remain in its contract.

Even an instantaneous stationary substitution can change the accounting.
If `D_v` is invertible, put `H=-D_v^-1 C_v` and
`A_s=A_v-B_v D_v^-1 C_v`. Substituting `h=Hy` gives `y'=A_s y` and
`l_v^T A_s=0`. But the reconstructed full inventory has covector
`l_eff^T=l_v^T+l_h^T H`, and `l_eff^T A_s` need not vanish.
The stationary hidden row is zero, whereas its reconstruction moves at
`H A_s y`. This graph is invariant for every visible state only if
`C_v A_s=0`.

For an explicit same-law tangent control, take unit P4 with unit capacities
at sine consensus, `e,a=w/pi,b=w/(beta*pi)>0`, and eliminate node 1's form
and phase. Retain nodes `(0,2,3)`, with all forms before all phases. Its
stationary-substitution generator and form covectors are

\[
T=\begin{pmatrix}-1/2&1/2&0\\1/4&-3/4&1/2\\0&1&-1\end{pmatrix},
\qquad A_s=\begin{pmatrix}eT&aT\\-bT&0\end{pmatrix},
\]
\[
\ell_v^{\mathsf T}=(1,2,1,0,0,0),\qquad
\ell_{\rm eff}^{\mathsf T}=(2,3,1,0,0,0),\qquad
\ell_{\rm eff}^{\mathsf T}A_s=
(-e/4,-e/4,e/2,-a/4,-a/4,a/2)\ne0.
\]

The extra weights account for the eliminated form
`x_1=(x_0+x_2)/2`. Thus apparent conservation in the substituted visible
system need not preserve the original inventory, even when the initial
hidden state satisfies that stationary relation. The
[static response controls](../../tests/physics/test_relational_environment_response.py)
use the shared block owner and independently declared exact rational
coefficients to check this counterexample and the hidden initial source.
This is a held-coefficient tangent statement, not a nonlinear elimination
theorem or a contradiction of the controlled fast-mediator approximation,
which retains its finite-capacity error and initialization obligations.

<a id="autonomous-path-cancellation"></a>
### An autonomous three-node crossing closes the moving-environment obstruction

The preceding frozen-boundary result does not require a new simulation to
test whether autonomous neighbors can invalidate instantaneous minimization.
Reuse the [reflection/uniqueness method](RELATIONAL_PATTERN_COMPOSITION.md#regular-seeded-reachability-audit)
on the smaller path `1--h--2`, with unit edges, equal endpoint capacities
`nu>0` and any fixed finite hidden capacity `mu>0`. Use the complete sine
comparison with held `e,w,beta>0` and no input or support event.

On a local phase lift, remove the common form and phase offsets by defining

\[
m=(x_1+x_2)/2-x_h,\quad b=(\theta_1+\theta_2)/2-\theta_h,\quad
p=(x_1-x_2)/2,\quad a=(\theta_1-\theta_2)/2 .
\]

Direct projection of the full nodal rows gives the exact four-coordinate
quotient

\[
\begin{aligned}
\dot m&=-(\nu+\mu)[em+(w/\pi)\cos a\sin b],&
\dot b&=(\nu+\mu)wm/(\beta\pi),\\
\dot p&=\nu[-ep-(w/\pi)\cos b\sin a],&
\dot a&=\nu wp/(\beta\pi).
\end{aligned}
\]

This is an exact symmetry quotient for this support and equal endpoint
capacities, not a closure of arbitrary regional averages. The half-angle
coordinates retain their chosen lift; adding an endpoint turn changes both
`a` and `b` consistently. Its storage and loss are

\[
E=m^2+p^2+2\beta(1-\cos a\cos b),\qquad
\dot E=-2e[\nu p^2+(\nu+\mu)m^2].
\]

The subspace `m=b=0` is invariant. If `x_h=theta_h=0` initially, both
hidden rows remain exactly zero. The endpoint motion then obeys
`pdot=nu*(-e*p-w*sin(a)/pi)`, `adot=nu*w*p/(beta*pi)` independently
of hidden capacity. The hidden relative resultant is `z_h=2*cos(a)`.

#### One rational preparation and an explicit finite crossing bound

Fix `e=w=1/2,beta=nu=1` and the exact rational preparation

\[
(x_1,x_h,x_2)=(1,0,-1),\qquad
(\theta_1,\theta_h,\theta_2)=(3/2,0,-3/2).
\]

All initial edge gaps are acute. Until `a=pi/2`, and while `p>=1/2`,
`adot=p/(2*pi)>0` and

\[
\frac{dp}{da}=-\pi-\frac{\sin a}{p}\ge-\pi-2>-\frac{36}{7}.
\]

Using `3<pi<22/7` gives `pi/2-3/2<1/14` and hence

\[
p>1-\frac{36}{7}\frac1{14}=\frac{31}{49}>\frac12.
\]

Thus `p` cannot reach `1/2` first. The globally smooth full sine law
and `adot>31/308` force a crossing at

\[
0<T<\frac{22}{31},\qquad
z_h(T)=0,\qquad \dot z_h(T)=-p(T)/\pi<-\frac{31}{154}.
\]

This is a rigorous finite-time sign change from a fully specified autonomous
preparation, not an inferred crossing from a numerical step. Mathematical
`pi/2` identifies the event surface; no represented graph phase is rounded
to that value. Full-state rates stay finite through it.

Before the crossing the hidden state is its exact conditional minimum and
both tracking defects vanish identically. Immediately after it, that same
hidden phase is the conditional maximum: the minimum has moved from `0`
to `pi` while the actual hidden phase remains `0`. Its phase storage
exceeds the minimized value by `4*beta*|cos(a)|`. This happens for every
fixed finite `mu>0`, however large. An imposed switch to the new minimum
would add an undeclared phase jump. With visible state held at the event,
the chosen `+pi` switch changes the conserved lifted weighted phase sum by
`2*pi/mu`. Any odd-pi representative changes that lifted sum by a nonzero
odd multiple of `2*pi/mu`; this weighted lift is not a phase sum with a
generally defined torus modulus.

This proves that zero tracking defect and high capacity are insufficient
without a uniform attraction/resultant margin. Both conserved weighted
means remain zero on the actual reflected solution, so conservation does
not remove the obstruction.

The same initial state is a native boundary-access control, within that
law's admitted interval only. Its reflected rows are

\[
\dot p=-p/2-a/(2\pi),\qquad
\dot a=\frac{pa}{2\pi\sin a},\qquad
\frac{dp}{da}=-\pi\operatorname{sinc}a-\frac{\sin a}{p}.
\]

The same lower bound on `p` and `adot>=p/(2*pi)` give a native endpoint
before `22/31` as well. Its full law is undefined at the hidden zero
resultant. This does not identify the two trajectories or their crossing
times, and gives no native continuation convention.

#### Transient cancellation, not persistent formation

For the sine preparation,
`E(0)=3-2*cos(3/2)<3` and `Edot=-p^2`. The reflected trajectory cannot
reach `a=+-2*pi/3`, where phase storage is three. Its sublevel component
is compact in `(p,a)` on that lift. The only invariant subset of zero loss is
`p=0,sin(a)=0`, hence `p=a=0`. LaSalle therefore gives eventual consensus
on this reflected branch. The cancellation crossing is a transient failure
of instantaneous elimination, not evidence of a newly maintained pattern.

Along the reflected orbit the transverse variational block is

\[
(\nu+\mu)\begin{pmatrix}
-e&-(w/\pi)\cos a(t)\\ w/(\beta\pi)&0
\end{pmatrix}.
\]

When `cos(a)<0` its frozen restoring sign is reversed. The moving reference
does not permit an all-future instability conclusion from those instantaneous
eigenvalues alone. For each fixed finite `mu`, continuity of the smooth
flow and transversality preserve a nearby cancellation crossing under small
initial perturbations: the full path has
`z_h=2*exp(i*b)*cos(a)` even outside reflection. No perturbation radius
uniform in unbounded capacity or nonlinear amplification bound is claimed.

The [static path controls](../../tests/physics/test_relational_sine_autonomous_path.py)
compare the exact quotient and work identities against the full sine field,
and check the analytic rational bounds. The comparison report's opt-in
`resultant_kinematics()` reuses the existing chain-rule owner: every ideal
sine phase rate is `a_i/pi` with exact rational
`a_i=(w/beta)*nu_i*q_i/d_i`, so one common pi division follows the rational
directional calculation. It retains ideal-rate provenance and admits
kinematics at cancellation without changing the native runtime. These
instantaneous bounds are not the finite-time crossing proof or a solver.

<a id="causal-sine-environmental-pressure"></a>
### Exact causal environmental pressure with retained internal state

The cancellation obstruction rules out unqualified instantaneous-minimum
replacement. It does not obstruct controlled reductions under their stated
uniform margins or an exact causal representation. Keep the declared
normalized-sine law on fixed simple unit support, held capacities, no source
or support event, and `e>=0,w,beta>0`. Let one supplied hidden node `h`
have `k>=2` visible neighbors `P` and capacity `mu>=0`. All other nodes
remain visible, including their internal edges. No equal-capacity,
reflection, nonzero-resultant or stationary-minimum premise is needed.

#### Exact interface and a sufficient hidden state

Write `y=x_h`, `theta=theta_h`, `X=sum_P x_p/k`,
`a=w/pi`, `b=w/(beta*pi)` and
`T=sum_P sin(theta_p-theta)`. The already derived hidden rows are

\[
\dot y=-\mu e(y-X)+\frac{\mu a}{k}T,\qquad
\dot\theta=\mu b(y-X).
\]

They retain two coordinates, with circular phase or a continuous chosen
lift. In particular, `T` uses the actual hidden phase, not the argument of
the neighbor resultant. For each port `p` the environmental pressure and
phase-rate contribution are

\[
P^h_p=-\frac{e}{d_p}(x_p-y)
       +\frac{a}{d_p}\sin(\theta-\theta_p),\qquad
V^h_p=\frac{\nu_p b}{d_p}(x_p-y).
\]

The original degree `d_p` includes the hidden edge. Let
`q^V_p=sum_(j visible neighbor of p)(x_p-x_j)` and
`S^V_p=sum_(j visible neighbor of p)sin(theta_j-theta_p)`. The internal
contributions are
`P^V_p=(-e*q^V_p+a*S^V_p)/d_p` and
`V^V_p=nu_p*b*q^V_p/d_p`. Therefore

\[
\dot x_p=\nu_p(P^V_p+P^h_p),\qquad
\dot\theta_p=V^V_p+V^h_p.
\]

Nonports keep their original rows. This decomposition exactly reproduces
the complete fine field, including ports that also share visible edges.
Removing the hidden edge from degree normalization would change the law.
The hidden node is retained as state; no live support mutation, independent
pair approximation or new pressure postulate occurs.

For `mu>0` the change to hidden phase and its velocity
`v=dot(theta)` is reversible:
`y=X+v/(mu*b)`. The equivalent second-order row is

\[
\ddot\theta+\mu e\dot\theta
 =\frac{\mu^2ab}{k}T-\mu b\dot X,
\qquad v(0)=\mu b[y(0)-X(0)].
\]

This is a same-information representation, not removal of a degree of
freedom. At `mu=0` that inversion is unavailable: both hidden coordinates
freeze and their supplied values still affect the ports.

#### Derivative-free nonlinear memory

Set `lambda=mu*e` and

\[
F_\lambda(t)=\int_0^t e^{-\lambda s}\,ds
 =\begin{cases}(1-e^{-\lambda t})/\lambda,&\lambda>0,\\
t,&\lambda=0.\end{cases}
\]

Here `theta(t)` denotes the continuous lift starting at supplied
`theta_0`. Variation of constants in the form row, followed by integration
of the phase row, gives

\[
\begin{aligned}
y(t)&=e^{-\lambda t}y_0+
 \int_0^t e^{-\lambda(t-s)}
       \left[\lambda X(s)+\frac{\mu a}{k}T(s)\right]ds,\\
\theta(t)&=\theta_0+\mu bF_\lambda(t)y_0
 -\mu b\int_0^t e^{-\lambda(t-s)}X(s)\,ds
 +\frac{\mu^2ab}{k}\int_0^t F_\lambda(t-s)T(s)\,ds .
\end{aligned}
\]

The second equation is a causal nonlinear Volterra equation: `T(s)`
depends on the unknown hidden phase at that same earlier time and on the
visible phase history. It is not a convolution with a fixed linear memory
kernel or a formula using future response. Together with the visible rows
and the supplied initial hidden state it is equivalent to the original
autonomous network. If the visible path is independently prescribed instead,
it represents an explicitly driven subsystem. These two uses must not be
interchanged.

The identities follow by integrating the form row and swapping finite
continuous integrals; conversely differentiation with the given initial
values recovers both hidden rows. The globally Lipschitz complete sine
field supplies uniqueness. Thus the representation inherits its
well-posedness, including `Z=sum_P exp(i*theta_p)=0`, without division by
`Z` or a resultant-argument branch convention. It also covers `e=0` through `F_0`
and `mu=0` through its zero coefficients. No fitted memory timescale has
been introduced: `lambda` is fixed by the supplied capacity and dissipation.
A common constant form shift cancels between `y_0` and `X` in the phase
formula because `F_lambda(t)=integral_0^t exp(-lambda*(t-s)) ds`.

Linearizing this exact realization at a frozen conditional minimum
recovers the preceding `D_s` block and its hidden initial source.
The signed tangent memory owner remains reusable in that stated limit;
replacing `T` by a fitted linear kernel would require a separate error
argument.

#### Work must be accounted for across the interface

Let the incident-edge storage be

\[
E_h=\sum_{p\in P}\frac{(x_p-y)^2}{2}
        +\beta\sum_{p\in P}[1-\cos(\theta_p-\theta)].
\]

Differentiating the actual edges gives the same boundary-work identity as
the cylinder calculation:

\[
\dot E_h=-\mu e k(y-X)^2+\mathcal W_P,\qquad
\mathcal W_P=\sum_P(x_p-y)\dot x_p
 +\beta\sum_P\sin(\theta_p-\theta)\dot\theta_p .
\]

The boundary rates are the full port rates, including visible internal
neighbors. `E_h` need not decrease independently. In the full-network
identity the visible loss is
`e*nu_p*(q^V_p+x_p-y)^2/d_p`, not the sum of separately squared
internal and environmental gradients. Discarding that cross term or the
boundary work creates a false passivity claim for the interface.

#### A non-reflected initial-state discriminator

Take a path with ports
`(x_1,theta_1,nu_1)=(1,0,1)` and
`(x_2,theta_2,nu_2)=(0,1/2,2)`, hidden form `y=1/4` and fixed
`mu>0`. Compare hidden phase `0` with `1/2` under the same coefficients.
These are rational, non-reflected preparations with identical visible
states. Switching the hidden phase from `0` to `1/2` changes the visible
form rates by

\[
(\Delta\dot x_1,\Delta\dot x_2)
 =a\sin(1/2)(1,2),
\]

both strictly positive. It changes the hidden form rate by
`-mu*a*sin(1/2)`. Every phase rate is unchanged; the hidden phase rate
is `-mu*b/4`. This follows directly from the pressure law before evaluating
any trajectory. Hence an instantaneous visible-only pressure cannot
represent both preparations. Retaining phase is necessary here, while
the exact causal representation predicts the distinction. This is not
physical identification or a theorem that one primitive law is unique.

#### Shared implementation and evidence scope

`comparison.mediated_pressure(mediator=...)` derives a
`SineMediatedPressure` report through the existing
[sine mediation owner](../../src/tnfr/physics/relational_sine_mediation.py).
It retains the full captured comparison, actual hidden state, original
port incidence, separated pressure/phase contributions and memory
coefficients. It computes reconstruction residuals, hidden acceleration
and incident storage/work from the existing field and interval kernels.
The report does not replace the hidden state by a minimum or claim to
evaluate the nonlinear memory integrals.

The [independent controls](../../tests/physics/test_relational_sine_mediated_pressure.py)
check the non-reflected response, internal-edge normalization, full-field
reconstruction, cross-term work, degenerate capacities, phase cancellation
and exact export. This integrates the reusable instantaneous realization;
the memory/equivalence theorem supplies its continuous interpretation.
No numerical trajectory, native execution switch or primitive connection
creation is implied.

<a id="sine-hidden-state-observability"></a>
### Prior visible observations and hidden-state observability

The causal representation requires initial hidden form and phase. Their
presence in a full simulator state does not make them observable. This
section fixes the same sine law, one hidden node and its known incident
ports, visible internal support, coefficients, visible capacities and clock.
The observations below precede any reserved response. The ideal theorem
treats visible state and rates as exact; the implementation separately
propagates supplied rate intervals.

#### Two observed rows remove the diffusive contribution

Let `V_p=dot(x_p)` and `Omega_p=dot(theta_p)` be visible port rates
at the declared observation time. For an active observed port `nu_p>0`,
the known phase row gives

\[
y_p=x_p+q^V_p-\frac{d_p\Omega_p}{\nu_p b},\qquad b=w/(\beta\pi).
\]

All ports used for this reconstruction must give the same hidden form
`y`. Substituting into the form row gives
`sin(theta_h-theta_p)=r_p`, with the cancellation-improved expression

\[
r_p=\frac{d_p}{\nu_p a}
       \left[V_p+\frac e b\Omega_p\right]-S^V_p,\qquad a=w/\pi.
\]

Thus the combination of observed form and phase rates removes the
diffusive term without subtracting large form coordinates or inserting
the reconstructed form into every phase equation. This is an identity of
the supplied law, not a pressure fitted to the evaluated future.
Internal gradients and currents keep the original degree
`d_p=degree_visible(p)+1`. A different hidden incidence changes the inverse
problem.

Set `h=(cos(theta_h),sin(theta_h))`. The remaining exact conditions are

\[
A_ph=r_p,\qquad A_p=(-\sin\theta_p,\cos\theta_p),\qquad h^\mathsf Th=1.
\]

Two ports `p,q` with `sin(theta_q-theta_p)!=0` determine `h` uniquely,
provided all other rows and the unit-circle equality agree. A useful
reference-relative form puts `delta=theta_q-theta_p`:

\[
s_{\rm rel}=r_p,\qquad
c_{\rm rel}=\frac{r_p\cos\delta-r_q}{\sin\delta}.
\]

These are `sin(theta_h-theta_p)` and `cos(theta_h-theta_p)`. No
inverse-trigonometric branch or normalization is needed. One active
phase-rate observation plus two active form-rate observations at
nonparallel phases is generically sufficient; supplying both rates at
every observed port permits the simplified formula and cross-checks.

#### Rank, consistency and an exceptional unique branch

At rank one, the observed phase rows are equal up to sign. After checking
the corresponding signs of `r_p`, choose one unit row `A_0` and a
perpendicular unit vector `B_0`. The circular possibilities are

\[
h=r_0 A_0\ \mathord{\pm}\ \sqrt{1-r_0^2}\,B_0.
\]

For `|r_0|<1` there are two global possibilities; a separately supplied
branch prior may distinguish them, but an arbitrary arcsine convention may
not. At `|r_0|=1` there is one exceptional solution. It is not a regular
inverse: perturbing the datum toward the interior splits the solutions
with square-root sensitivity. For `|r_0|>1` there is no circular solution.
Rank deficiency alone therefore does not prove two distinct states in
every case. The bounded implementation abstains from this branch inversion
rather than silently selecting one.

With no active observed ports there is no information from these rate
equations. A zero-capacity port must have both rates exactly zero;
otherwise the data contradict this unforced law. Its phase can still
affect the hidden dynamics, but its zero output supplies no state
constraint here. Missing measurements and unknown capacities are not
replaced by zero.

#### Observation geometry differs from the phase resultant

Let `m` count active observed phase rows, and define the derived
second-harmonic resultant `Z_2=sum exp(2i*theta_p)` over those rows.
Then

\[
A^\mathsf TA=\frac12
\begin{pmatrix}
m-\operatorname{Re}Z_2&-\operatorname{Im}Z_2\\
-\operatorname{Im}Z_2&m+\operatorname{Re}Z_2
\end{pmatrix},
\qquad
\lambda_\pm=\frac{m\pm|Z_2|}{2}.
\]

Equivalently, by the two-column Gram determinant identity,

\[
\det(A^\mathsf TA)
 =\sum_{p<q}\sin^2(\theta_q-\theta_p)
 =\frac{m^2-|Z_2|^2}{4}.
\]

For `m>=2` this determinant is positive exactly when the observed rows
have rank two. The ideal least-squares phasor error amplification is
`1/sqrt(lambda_-)`; it is not controlled by the ordinary neighbor resultant
`|Z|=|sum exp(i*theta_p)|`. Three active ports at phases
`0,2*pi/3,4*pi/3` have `Z=Z_2=0` and `A^T A=(3/2)I`:
the conditional-minimum direction is undefined but hidden phase remains
observable from their responses. Aligned ports have maximal `|Z|` yet
rank one. For two ports only, exact resultant cancellation implies
antipodal directions and rank one.

This is a derived observation-conditioning diagnostic. It neither adds
a primitive phase coordinate nor drives a nodal row, selects an operator
or supplies a new physical constant. Exact ideal-angle examples are
distinct from represented numerical probes near those configurations.

#### What instantaneous observation still cannot identify

Hidden capacity `mu` does not appear in any instantaneous visible
port-rate equation above. For the same hidden form/phase and visible
state, every admitted hidden capacity gives the same visible first rates.
Therefore even exact form/phase recovery does not identify `mu` or
generally determine a unique future unless its value or law is independently
supplied. A complete equilibrium can remain identical for every capacity.
The hidden rows, and hence later visible rates, can differ. This is
nonidentifiability under this information budget, not absence of a
capacity effect.

These observations also do not identify the primitive pressure law among
competing models. Derivative evidence and a clock/preparation bridge must
be independently justified; inferred pressure from the reserved future
cannot be relabeled prediction.

#### Sound bounded admission and shared implementation

The [sine observation owner](../../src/tnfr/physics/relational_sine_observation.py)
accepts a visible-only graph and declared incident ports. The hypothetical
hidden node and its coordinates are not input. Shared relational admission
retains signed scalar form, circular phase values, nonnegative capacity,
unit support and absent Gamma. Visible components may be disconnected,
but each must touch a supplied port so that the declared hidden incidence
would complete a connected support. No fictitious hidden state is staged.

Paired rate intervals retain all uncertainty supplied by the caller.
The observer intersects per-port form bounds, evaluates the cancellation-
improved phase projections, and uses a pair whose determinant interval
is separated from zero. It intersects proven component ranges with
`[-1,1]`, checks the other phase rows and tests unit-norm compatibility.
It never rescales an estimate onto the unit circle.

An empty form intersection, impossible projection, or residual/norm
interval excluding its required value proves inconsistency with the
declared premises. The reverse does not hold: interval overlap ignores
some shared-data correlations and does not prove that one joint state
exists. A surviving `bounded_candidate` is a conditional enclosure for
every consistent state, not an exact identification or existence
certificate. `inconsistent` and `unavailable` remain distinct.
An unresolved sine interval is not a rank-one proof; exact equal captured
phases or a single active row can establish that rank. The Gram determinant
report uses only active observed rows.

Source identifier, clock identifier, observation time, evidence window
and forecast start are explicit. The evidence window must contain the
observation time and end strictly before the forecast starts. The
[existing three-sample rate observer](../../src/tnfr/physics/relational_observations.py)
estimates the rate at its first sample, not at its window endpoint.
An inferred state cannot be silently moved to the end of that window
or to a later forecast start. Declared provenance is not authentication;
the supplied error budget must include the derivative-estimation error.

The [independent observation controls](../../tests/physics/test_relational_sine_observation.py)
withhold hidden coordinates from inference, supply independently bounded
prior port rates and check recovery, rank loss, impossible data, source
isolation and uncertainty. This establishes conditional software-state
observability. It does not constitute a physical measurement bridge,
hidden-capacity identification or an evaluated temporal prediction.

<a id="sine-hidden-capacity-observability"></a>
### Prior acceleration identifies capacity only through an active hidden response

Keep the same fixed support, held capacities, coefficients and clock as the
causal sine model. The preceding visible-only inverse bounds hidden form and
phase but its instantaneous port equations do not consume hidden capacity.
Differentiating those already supplied equations supplies the missing
capacity dependence without postulating a new acceleration law.

#### A shared tangent chain rule

Use `a=w/pi`, `b=w/(beta*pi)`, `y=x_h`, `X=mean_P x_p`, and define
the hidden response per unit capacity

\[
f=-e(y-X)+\frac ak\sum_{p\in P}\sin(\theta_p-\theta_h),\qquad
g=b(y-X).
\]

Then `dot(y)=mu*f` and `dot(theta_h)=mu*g`. The sum includes all
structural ports, even ports with zero capacity and zero output rates.
For visible state `v` and hidden state `h=(y,theta_h)`, the full rows have
the form `dot(v)=F_v(v,h)`, `dot(h)=mu*F_h(v,h)`, hence

\[
\ddot v=D_vF_v\,F_v+\mu D_hF_v\,F_h.
\]

The sensitivity to capacity is precisely the hidden-to-visible tangent
block applied to the hidden response, reusing the same geometry as the
existing memory and observation calculations. This identity holds at
the current state; it is not a linearized replacement for its trajectory.

Let `V_p=dot(x_p)`, `Omega_p=dot(theta_p)` and
`c_p=cos(theta_h-theta_p)`. Define quantities that exclude the
unknown hidden rates:

\[
D_p=d_pV_p-\sum_{\substack{j\sim p\\j\ {\rm visible}}}V_j,
\qquad
C_p=\sum_{\substack{j\sim p\\j\ {\rm visible}}}
\cos(\theta_j-\theta_p)(\Omega_j-\Omega_p)-c_p\Omega_p .
\]

The original incidence degree includes the hidden neighbor. Differentiating
the form gradient and sine current gives
`dot(q_p)=D_p-mu*f` and `dot(S_p)=C_p+mu*c_p*g`, so

\[
\begin{aligned}
\ddot\theta_p&=\frac{\nu_pb}{d_p}(D_p-\mu f),\\
\ddot x_p&=\frac{\nu_p}{d_p}
  [-eD_p+aC_p+\mu(e f+a c_p g)].
\end{aligned}
\]

Every acceleration is affine in `mu`. Visible nonports have no hidden
neighbor: their first rates can be calculated from visible state and
the already supplied law. Port first rates remain prior observations.
Using a nonport's model-derived rate is a declared use of the law, not an
additional measurement or a hidden-state read.

#### A second cancellation improves the inference

When both acceleration channels are observed, their combination eliminates
the form-gradient rate and the hidden form-response contribution:

\[
\ddot x_p+\frac e b\ddot\theta_p
 =\frac{\nu_pa}{d_p}(C_p+\mu c_p g).
\]

This is the differentiated version of the paired-rate cancellation in the
state inverse. Computing it directly avoids duplicating uncertain terms
that cancel algebraically. It supplies a complementary phase-exchange
channel; it is not statistically independent of its two input measurements.
Intersecting all channel enclosures is sound, but cannot recover correlations
that the input intervals did not provide.

For any measured channel write `A=B+mu*S`. Exact known `S!=0` gives
`mu=(A-B)/S`. In particular, an active port's phase acceleration identifies
capacity whenever `f!=0`. If `f=0,g!=0`, a form acceleration with
`c_p!=0` supplies the information instead. When both channels are observed
on rank-two active port geometry, all their sensitivities vanish if and
only if `f=g=0`: phase rows first force `f=0`, and the remaining form
rows cannot all have `c_p=0` at rank two unless `g=0`.
This is sufficient, not a necessary rank condition for every individual
preparation or measurement subset.

There is no division by `mu`. Informative data can therefore identify
`mu=0`, and `e=0` does not by itself prevent identification. This
determines a capacity under the declared clock and law; it does not
derive its value universally, establish a capacity evolution law or
identify laboratory units.

#### Capacity information can survive phase ambiguity

A unique hidden phase is sufficient but not always necessary for capacity
inference. Each already bounded projection
`r_p=sin(theta_h-theta_p)` contributes

\[
f=-e(y-X)-\frac ak\sum_P r_p .
\]

If form and these projections are known, a phase acceleration can identify
capacity even when the complete phase has two branches. For example, aligned
ports with `r_p=0` and `y-X!=0,e>0` have hidden phase alternatives separated
by a half-turn, but the same nonzero `f=-e(y-X)`. Their phase acceleration
still reveals `mu`. This does not resolve the phase needed for a general
future response.

In bounded admission, an unobserved projection or cosine can retain its
proved range `[-1,1]`. This permits a conservative capacity bound if a
sensitivity remains separated from zero. It does not fabricate a unit
phase, discard inactive structural ports or certify that all independently
enclosed projections share one possible phase.

#### Informative and blind preparations

One rational non-reflected example has hidden state `(y,theta_h)=(1,0)`
and ports `(x,theta,nu)=(0,0,1),(0,1/2,2)`. Here
`f=-e+(a/2)*sin(1/2)` and `g=b`. Changing hidden capacity by
`Delta mu` preserves every visible first rate but changes the phase
accelerations by `-Delta mu*b*f*(1,2)`. At the default coefficients `f`
is nonzero; it is also nonzero when `e=0`. These facts can be checked
before any trajectory is evaluated.

The earlier [autonomous path quotient](#autonomous-path-cancellation)
supplies a stronger blind control. Take hidden form and phase zero, endpoints
`(x,theta)=(1,1/2),(-1,-1/2)` and equal positive endpoint capacities.
The endpoint phase rows have rank two, so form and phase are identifiable,
yet reflection keeps `f=g=0` and the hidden state fixed for every
finite held `mu>=0`. The entire visible temporal history is independent
of that capacity, even though the endpoints move. Higher derivatives or
a longer observation of this same preparation cannot identify it.
Unequal endpoint capacities need not preserve that invariant branch;
instantaneous blindness alone is not an all-future theorem.

#### Bounded prior admission and its limits

`state_inference.infer_capacity(...)` in the existing
[sine observation owner](../../src/tnfr/physics/relational_sine_observation.py)
retains the prior state report and accepts supplied form and/or phase
acceleration intervals on a subset of its ports. Missing channels remain
unobserved. The clock and observation time must match the state evidence;
both windows must contain that time and end strictly before the inherited
forecast start.
The report derives required nonport first rates from the visible snapshot
and shared sine kernel, with explicit provenance.

The observer bounds `f,g,D_p,C_p`, preserves actual port incidence and
constructs the form, phase and available combined exchange channels.
It inverts only sensitivities certified away from zero, intersects their
capacity enclosures with `mu>=0`, and checks all supplied channels against
the surviving interval. A zero-containing sensitivity is unresolved, not
proof of exact stationarity. A separated impossible residual or negative-only
capacity excludes the declared premises. Inconsistent source evidence
cannot become a successful capacity estimate.

A surviving `bounded_candidate` encloses every jointly consistent capacity;
it does not prove existence, exact uniqueness under uncertain data,
independence of channels or authenticated provenance. Form information may
support capacity bounds while phase stays unavailable, so capacity status
alone does not admit a complete predictive state. All bounds belong to
the original observation time, not automatically to the forecast start.

The [independent controls](../../tests/physics/test_relational_sine_capacity_observation.py)
compute full fine-field accelerations by differentiating edge gradients
and currents, withholding hidden capacity from inference. They distinguish
informative, zero-capacity, no-damping, phase-ambiguous and blind cases,
including visible internal edges and inactive structural ports. These are
software-state observations under a supplied law, not a physical bridge or
an evaluated reserved forecast.

<a id="sine-prior-reserved-forecast"></a>
### A finite forecast from jointly admitted prior environmental evidence

The state and capacity observers supply necessary outer bounds. Their next
use is a finite forecast under the **same supplied normalized sine law**,
with the response source's hidden coordinates and capacity withheld from
the predictor. Neither an overlap of inferred intervals nor a successful
forward computation establishes that the original observations have a
joint realization. This admission obligation precedes the reserved response.

#### Joint admission and a regular circular coordinate

Let `Y`, `(C,S)` and `M` be the inferred hidden form, relative unit-phase
rectangle and nonnegative capacity interval at the actual observation time.
For this preparation require `C.lo>0`. The existing rational argument
enclosure then gives

\[
\Theta=\theta_{\rm anchor}+\operatorname{atan}(S/C).
\]

Every admissible circular phase represented by `(C,S)` has a lift in
`Theta`; replacing its lift by an integer turn leaves the sine-law visible
response unchanged. This is a chart enclosure, not normalization of a
possibly nonunit midpoint. The rectangle can contain points off the unit
circle, and the resulting angle interval can enlarge the feasible set.
That enlargement is acceptable for an outer forecast enclosure.

Construct one rational candidate `(y_*,theta_*,mu_*)` from the prior
inference alone, with a declared rounding rule. Require membership in
`Y x Theta x M` and certify that its forward form/phase first rates and
all supplied acceleration channels lie wholly inside the original prior
evidence intervals. The forward checks use the same full sine equations,
including the original degrees and held-capacity chain rule. They do not
read the source's hidden truth or fit any reserved response. If these
inclusions hold, this one circular state witnesses a nonempty joint
realization of the supplied evidence. If they fail, admission fails;
interval overlap alone cannot repair the missing witness. This verifies
mathematical compatibility, not the authenticity of the declared source.

The witness does **not** replace the predictive state by a point. Retain
the product outer enclosure `Y x Theta x M` with the captured visible
coordinates. It contains every jointly compatible hidden state, although
it discards correlations and may be conservative. For the three-node
path below, propagate the seven coordinates

\[
(x_L,x_R,y,\theta_L,\theta_R,\theta_h,\mu),\qquad \dot\mu=0.
\]

An interval/jet field with this held-capacity row lets the shared
[validated Taylor owner](../../src/tnfr/mathematics/_validated_taylor.py)
propagate initial state and capacity uncertainty together. Every phase is
an ordinary continuous lift of a circular state; no stationary hidden
minimum or native Arg pressure enters the field. Propagation must begin
at the observation time and include the complete gap to the reserved
observation, including any declared forecast-start boundary. Prior error,
propagated initial uncertainty and integration remainder remain separate.

#### A prepared source and an explicit counterfactual

Take a unit path `L--h--R`, with no direct visible edge, and set

\[
e=w=\tfrac12,\quad \beta=1,\quad
(x_L,\theta_L,\nu_L)=(0,0,1),\quad
(x_R,\theta_R,\nu_R)=(0,\tfrac12,2),\quad
(y,\theta_h,\mu)=(1,0,1).
\]

These exact source coordinates specify the software preparation, not
predictor inputs. The predictor receives visible state and separately
bounded prior first/second rates, then uses the observers and joint
admission above. The reserved observable is the left-port phase at elapsed
time `H=1/16` in the declared structural clock.

The control retains the same initial visible and hidden coordinates but
sets `mu=0`, freezing the mediator's form and phase. Its initial visible
first rates coincide with those of the source; its accelerations generally
do not. It is therefore an explicit counterfactual capacity intervention,
**not** an equally fitting alternative to the complete prior evidence.
The practical control inherits the same inferred initial-coordinate
enclosure and changes only this stated capacity premise. It cannot claim
to have independently fitted the source's acceleration record.

#### An analytic separation before any response is evaluated

Write `a=b=1/(2*pi)` and consider both exact arms, `mu=1` and `mu=0`.
Use the whole-time candidate tube

\[
|x_L|,|x_R|\le\tfrac18,\qquad
|y-1|\le\tfrac18,\qquad
|\theta_i-\theta_i(0)|\le\tfrac18.
\]

Each incident form difference has magnitude at most `5/4`. Since
`3<pi<22/7`, the full degree-normalized rows obey

\[
|\dot x_i|\le\frac{19\nu_i}{24},\qquad
|\dot\theta_i|\le\frac{5\nu_i}{24},\qquad
(\nu_L,\nu_R,\nu_h)=(1,2,\mu).
\]

For `H=1/16`, these bounds permit at most `19/192` form displacement
and `5/192` phase displacement, both strictly below `1/8`. The usual
first-exit argument therefore certifies the tube for both arms throughout
this window; no evaluated trajectory is needed to establish it.

Differentiating the rows within that tube, and using `|cos|<=1`, gives

\[
\begin{aligned}
|\ddot x_L|
&\le \tfrac12\left(\tfrac{19}{24}+\tfrac{19}{24}\right)
 +\tfrac16\left(\tfrac5{24}+\tfrac5{24}\right)
 =\tfrac{31}{36},\\
|\ddot y|
&\le \tfrac12\left(\tfrac{19}{24}+\tfrac{19+38}{48}\right)
 +\tfrac16\left(\tfrac5{24}+\tfrac{5+10}{48}\right)
 =\tfrac{155}{144}.
\end{aligned}
\]

The hidden row is exactly zero in the control, so these same upper bounds
remain valid there. The left-port phase row is
`dot(theta_L)=b(x_L-y)`, hence each arm satisfies

\[
|\theta_L^{(3)}|
\le \tfrac16\left(\tfrac{31}{36}+\tfrac{155}{144}\right)
=\tfrac{31}{96}.
\]

At the exact initial preparation the hidden response per unit capacity is

\[
f=-\tfrac12+\frac{\sin(1/2)}{4\pi}<-\tfrac{11}{24}.
\]

Both arms have identical initial left phase and phase rate. Their initial
phase-acceleration contrast is `-b*f`. Since `b>7/44`, this contrast is
strictly greater than `7/96`. Taylor's theorem, with a separate third-order
remainder for each arm, therefore yields

\[
\begin{aligned}
\theta_L^{(\mu=1)}(H)-\theta_L^{(\mu=0)}(H)
&>\frac7{192}H^2-\frac{31}{288}H^3\\
&=\frac{137}{1179648}>\frac1{10000},\qquad H=\tfrac1{16}.
\end{aligned}
\]

This is an exact conditional separation theorem for the stated preparation.
It supplies a prospective direction and scale; it does not automatically
transfer the margin to uncertain reconstructed states. Their forecast and
counterfactual enclosures must retain the prior uncertainty and certified
numerical errors, and meet the separately frozen width and separation
budgets before comparison with the reserved response. No response result
is asserted by this derivation.

#### Retained prospective software response

The fixed v1 protocol was prepared, predicted and evaluated in separate
invocations on 2026-10-03. The public prior contains outward first/second
derivative bounds padded by `2^-30`, obtained independently from the exact
source's forward edge equations at `t=0`. These are synthetic instantaneous
derivatives, with zero finite-difference truncation by construction; they
are not observations acquired from time samples. The predictor receives no
source hidden state or capacity.

The prior-derived grid witness passed joint forward containment. The full
inferred form/phase/capacity box was retained, including hidden phase width
approximately `4.85e-8` and capacity width approximately `3.56e-8`. Both
prediction and counterfactual were issued before the source response was
evaluated. Each used eight fixed `1/128` steps, order six and 128-bit
outward rational arithmetic from `t=0` through `H=1/16`, including the
declared preforecast gap to `1/32`.

| Frozen check | Retained outcome |
| --- | --- |
| Joint prior witness and full-horizon inclusion | Admitted; all three chains have eight strict Picard/Taylor steps |
| Maximum endpoint width at most `1e-6` | Prediction `<4.89e-8`; control `<4.85e-8`; source response `<2.00e-18` |
| Issued left-phase prediction | Outward displayed interval `[-0.009654176181, -0.009654176112]` |
| Reserved left-phase response | Approximately `-0.00965417614677594`; its complete certified interval lies inside the issued prediction |
| Issued frozen-control phase | Outward displayed interval `[-0.009793204164, -0.009793204102]` |
| Response minus control greater than `1e-5` | Lower bound `>0.000139027955`; positive and separated before response evaluation |

The exact rational intervals, whole-time tubes, initial-radius propagation
and remainder bounds remain in
`artifacts/research/relational_sine_prior_forecast/response-v1.prediction.json`
and `response-v1.json`. Sibling `.protocol.json`, `.source-state.json` and
`.source.zip` retain the public declaration, separate source preparation and
source archive. The protocol SHA-256 is
`0c5bcf6497f16ec0c4e2438c3af87901b17f7d80d1809fd822563acbc7d3a882`;
the issued prediction file SHA-256 is
`24fef37f090607b6e58d057cdeba2f76cd57a301d0ce313dc186ba35b35facda`.
These bind retained bytes, not authenticated chronology. Missing local
artifacts remain unavailable; tests do not regenerate the producer.

This result closes one finite prior-to-future software gate under the
supplied sine law. It establishes neither a physical measurement bridge
nor unknown-law discrimination: the control is a changed-capacity
intervention, and source and predictor share the declared dynamics and
validated numerical owner. Finite sampled evidence is a separate admission
obligation; the execution plan owns its next bounded step.

<a id="sine-finite-sample-admission"></a>
### Finite samples, derivative errors and the remaining observation boundary

The retained forecast uses synthetic instantaneous derivatives. The following
contract instead bounds first and second derivatives from the same three
earlier samples. It is a conditional observation result, not a new response
evaluation or a claim that suitable samples have already been acquired.

#### One time, one stencil and separate error sources

Let a real coordinate `f` have three recorded values `z_j` at nominal
times `t_0+jh`, `j=0,1,2`, with `h>0`. Its actual acquisition time may
differ by at most `tau_j`, and its value error at that actual time is at
most `epsilon_j`. Independently assume `|f'|<=B_1` on the enlarged
time window containing both actual and nominal times, and
`|f'''|<=M_3` throughout the nominal stencil window. The mean-value
bound transfers the timing error to the nominal sample:

\[
|z_j-f(t_0+jh)|\le E_j:=\epsilon_j+B_1\tau_j.
\]

This transfer does not identify the actual timestamps or synchronize a
physical clock. All derivative units refer to the declared common clock.
Define the two forward estimates

\[
\widehat v=\frac{-3z_0+4z_1-z_2}{2h},\qquad
\widehat a=\frac{z_0-2z_1+z_2}{h^2}.
\]

Both refer to **the initial time `t_0`**, not the middle or final sample.
Their simultaneous conditional enclosures are

\[
\begin{aligned}
|\widehat v-f'(t_0)|
&\le \frac{3E_0+4E_1+E_2}{2h}+\frac{M_3h^2}{3},\\
|\widehat a-f''(t_0)|
&\le \frac{E_0+2E_1+E_2}{h^2}+M_3h.
\end{aligned}
\]

For the first formula, Taylor's integral remainder gives a Peano kernel,
in relative time `s`, equal to
`(3*s^2-4*h*s)/(4*h)` on `[0,h]` and
`-(2*h-s)^2/(4*h)` on `[h,2*h]`. It is nonpositive and its absolute
integral is `h^2/3`. For the second formula, the noiseless second
difference is the average of `f''(t_0+s)` with triangular density
`s/h^2` on `[0,h]` and `(2*h-s)/h^2` on `[h,2*h]`.
That density has mass one and mean `h`; the `M_3` Lipschitz bound on
`f''` therefore supplies the stated remainder. Applying the absolute
stencil weights to each sample error proves the remaining terms.

Measurement error, clock error and differentiation remainder stay
separate in the report even though their sum provides the final interval.
The same sample errors affect both estimates. A rectangular pair of
intervals encloses the joint possibilities conservatively; it does not
declare independent errors or assert that every point of the rectangle is
realizable. The observed state at `t_0` has its own interval
`[z_0-E_0,z_0+E_0]`, which must not be discarded.

The existing uniform-error rate and coefficient observers are special
cases with `tau_j=0` and `epsilon_j=epsilon`. Their constants
`4*epsilon/h+M_3*h^2/3` and `4*epsilon/h^2+M_3*h` are preserved by
the shared joint stencil. For positive `epsilon,M_3` these bounds are
minimized, respectively, at `h^3=6*epsilon/M_3` and
`h^3=8*epsilon/M_3`. Their minima are
`(6*epsilon)^(2/3)*M_3^(1/3)` and
`3*epsilon^(1/3)*M_3^(2/3)`. Reducing sample spacing indefinitely
amplifies fixed observation error; it does not guarantee better inference.
These optima describe the bounds, not a prescription to retune an evaluated
record.

#### Smoothness from an independently declared nodal class

The complete normalized-sine law supplies a trajectory-free smoothness
bound when its **whole network**, including any hidden node, satisfies
independent preparation and capacity limits. Retain fixed simple unit
support without isolates, held capacities `0<=nu_i<=N`, fixed
`e>=0,w>0,beta>0`, no forcing and no events. Put
`a=w/pi` and `b=w/(beta*pi)`. Suppose the initial form diameter
`max_i x_i-min_i x_i` is at most `D_0`.

At a maximum-form node the diffusive term is nonpositive and the
normalized sine current is at most one. At a minimum it gives the opposite
bound. The upper Dini derivative of the diameter is therefore at most
`2*N*a`, even when a maximizing node changes. On `0<=t<=T`,

\[
D(t)\le D:=D_0+2NaT.
\]

This depends on relative form rather than an unnecessary absolute form
origin. The normalized edge differences then yield the uniform bounds

\[
\begin{aligned}
|\dot x_i|&\le V:=N(eD+a),&
|\dot\theta_i|&\le\Omega:=NbD,\\
|\ddot x_i|&\le A:=N(2eV+2a\Omega),&
|\ddot\theta_i|&\le B:=2NbV,\\
|x_i^{(3)}|&\le M_x:=N[2eA+a(4\Omega^2+2B)],&
|\theta_i^{(3)}|&\le M_\theta:=2NbA.
\end{aligned}
\]

To obtain the second row, differentiate each form difference and each
`sin(theta_j-theta_i)` in the supplied nodal law. Their normalized sums
are bounded by `2*V` and `2*Omega`. Differentiating once more gives
`2*A` for the form sum and `4*Omega^2+2*B` for the sine sum:
the latter contains both the squared phase-rate difference and the
phase-acceleration difference. No stationary, acute-phase or nonzero
resultant premise is used. Zero capacity is admitted and the corresponding
row remains zero.

These are supplied-class bounds, not reconstructed hidden-state estimates.
The ceiling `N` includes hidden capacity; it cannot be inferred using
derivative errors whose justification already presumes that ceiling.
Similarly, the form diameter limit must include the hidden node. A
validated whole-window tube can supply sharper bounds under its own
premises, but three recorded values alone do not authenticate smoothness:
`A_0*t*(t-h)*(t-2*h)` vanishes at all three sample times while its
initial rate `2*A_0*h^2` and acceleration `-6*A_0*h` are unbounded
as `A_0` varies. This last example is an observation-level obstruction;
it is not asserted to solve the sine law.

#### A fixed prospective derivative budget

One declared candidate uses the default `e=w=1/2,beta=1` with
`D_0<=2,N<=2`, `t_0=0`, `h=1/4096`,
sample errors `(0,2^-44,2^-44)` and timing errors
`(0,2^-50,2^-50)`. The first visible state is independently prepared
exactly; its zero error is an additional premise, not a conclusion from
measurement precision. Since `2^-50<h`, all acquisitions lie in the
nonnegative window `[0,T]`, `T=2*h+2^-50`.

Using only `pi>3` in the preceding bounds gives rational upper estimates
`a,b<1/6`, `M_x<11.853637` and `M_theta<3.407890`. The separate
source terms above imply the following outward displayed total errors:

| Coordinate | First derivative at zero | Second derivative at zero |
| --- | --- | --- |
| Form | `<2.362e-7` | `<0.002897` |
| Lifted phase | `<6.830e-8` | `<0.0008350` |

The shared exact-arithmetic owner may use a sharper certified pi
enclosure. This table is a prospective error budget computed without
samples or a trajectory. It does not reuse the earlier `2^-30`
derivative padding, inherit the earlier forecast-width gate or certify
that an instrument meets these errors. Conditional composition with the
inverse requires actual prior records, the stated exact visible
preparation and all its other support, law and capacity premises.

#### Boundaries that prevent a false observation claim

The current hidden-state inverse stores visible form and phase as exact
point coordinates. General noisy first samples cannot be inserted as those
points. A simple same-law ambiguity is the common form shift
`x_i(t)->x_i(t)+c` at every node: it preserves all form differences,
phase dynamics and derivatives, while changing every absolute form.
Finite form errors may admit both histories. Choosing the recorded
center as an exact state discards such uncertainty; a downstream
outer-state forecast is then no longer justified for the complete
measurement class. The exact prepared anchor above avoids this issue
only within its declared scope.

The common phase shift `theta_i(t)->theta_i(t)+c` is another exact
symmetry of this law. Uncertain absolute phase therefore persists even
when all phase differences and derivative evidence agree. A forecast of
an absolute lifted port phase must retain its initial reference uncertainty;
a relative-phase observable has a different, explicitly stated contract.

Circular samples also need a declared consistent lift or separately
proved unwrapping rule. At nominal spacing `h`, phases differing by
`2*pi*k*t/h` have identical circle samples but different rates.
This is another observation-level aliasing obstruction, not a claim that
both functions solve a fixed supplied nodal law. Whole-window phase-rate
limits can constrain lifting; treating a wrapping jump as acceleration
cannot.

In particular, if circular observation errors are bounded by
`epsilon_j` and the phase speed by `Omega`, the strict margins

\[
m_j=\pi-\Omega(h+\tau_j+\tau_{j+1})
       -\epsilon_j-\epsilon_{j+1}>0,\qquad j=0,1,
\]

are sufficient for unique nearest-increment lifting. The actual phase
change between the two acquisition times has magnitude at most
`Omega*(h+tau_j+tau_(j+1))`. Adding the two observation errors still
leaves the corresponding measured lifted difference strictly between
`-pi` and `pi`, so its principal circular difference selects that
unique increment. An initial reference selects the common lift; its error
is not removed. A certified lower bound for pi yields conservative
margins, and a nonpositive margin means this sufficient condition is
unresolved, not that every data record is ambiguous. This is a
conditional lifting criterion; the budget observer does not unwrap or
authenticate actual circular samples.

Finally, instantaneous derivative intervals are necessary consequences
of the raw sample constraints. A witness accepted by `admit_sine_prior`
proves compatibility with those relaxed derivative intervals, not that
its entire trajectory passes through every sample/time/error box.
Conservative inverse and forecast boxes can still enclose every genuinely
compatible state, but raw-record existence remains a separate obligation.
No sample source, acquisition, future response or physical bridge has
been established here. The bounded result is the joint derivative
contract and its explicit preparation requirements; general noisy-anchor
inference and any new sample-based forecast retain their own admission.

### Capacity sets the memory clock, and the kernel need not be positive

Return here to the native two-port model at the start of section 9, with
`c=cos(2*pi/5)` and its original port quantity `r=1+2*c`.
This `r` is distinct from the resultant ratio `rho=R/k` above.
For that two-port single-intermediary model,
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
Simply adding the edge to the old opposite equilibrium fails the
acute and positive-resultant options: its gap is `-4*pi/5`, and each new port resultant
has relative real part `2*cos(2*pi/5)+cos(4*pi/5)=(sqrt(5)-3)/4<0`.
The nonzero imaginary parts instead permit the full regular law and its
explicit regular-domain executor. This removes that particular chart
restriction, not the preceding passive-storage obstruction to the final target.

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

## 13. A retained collective phase offset after isolated-pattern recovery

<a id="relational-retained-phase-memory"></a>

An isolated ring can recover the same internal winding geometry while
retaining a different common phase relative to a separately declared
reference. This result concerns the selected relational law on a supplied
C5, not a new state variable or a physical interpretation of phase. It uses
the [isolated-ring capture theorem](RELATIONAL_PATTERN_COMPOSITION.md#relational-pattern-detachment)
and keeps the offset rows that the earlier relative-state memory studies
explicitly removed from their observations.

### State, reference and exact mean identities

Use the oriented unit cycle `i -> i+1` modulo 5, one common held capacity
`nu>0`, `e,w,beta>0`, the reference phase completion
`theta_dot=(nu*w/beta)*H^-1*Bx`, and no input or event during relaxation.
Set

\[
\kappa=2\pi/5,\qquad C=\cos\kappa>0,\qquad
\theta_i^*=i\kappa.
\]

Choose continuous lifts in the acute winding-one component and write

\[
x=\bar x\mathbf1+u,\qquad
\theta=\theta^*+m\mathbf1+v,\qquad
\mathbf1^Tu=\mathbf1^Tv=0.
\]

The scalar `m` is the arithmetic mean of the lifted deviation from this
reference, not an angle reconstructed from form. A second, untouched ring
at the same twist and common phase defines the comparison frame. Independently
resetting the two component phase frames would erase the chosen comparison
and is not part of the protocol. A common rotation of both rings leaves
the final relative offset unchanged.

Let `(Sz)_i=z_(i+1)`, `B=2I-S-S^-1`, and `J=S-S^-1`. Here `J` denotes the
skew cyclic difference, not the earlier full-state Jacobian. The actual
wrapped edge gaps are

\[
\delta_i=\kappa+v_{i+1}-v_i.
\]

On this acute branch the native relative resultant is

\[
z_i=e^{i\delta_i}+e^{-i\delta_{i-1}}
=2\cos\!\left(\frac{\delta_i+\delta_{i-1}}2\right)
 e^{i(\delta_i-\delta_{i-1})/2}.
\]

The cosine factor is positive and the displayed angle belongs to the
principal regular branch. Thus the phase source is **exactly linear** in
the centered phase deviation on this particular cycle component:

\[
g=-\frac1{2\pi}Bv,\qquad
\dot u=-aBu-bBv,\qquad
a=\frac{\nu e}{2},\quad b=\frac{\nu w}{2\pi},\qquad
\dot{\bar x}=0.
\]

This exact form-mean conservation uses both degree two and common capacity;
it is not imported into an irregular or heterogeneous-capacity graph.
The phase metric still contains nonlinear internal geometry:

\[
H_i=2\pi\cos\!\left(\kappa+\frac{(Jv)_i}{2}\right)
          \operatorname{sinc}\!\left(\frac{(Bv)_i}{2}\right),
\qquad
\dot m=\frac{\nu w}{5\beta}\sum_i\frac{(Bu)_i}{H_i}.
\]

Although `sum(H_i*theta_dot_i)=0` exactly, the weights depend on the
evolving phase geometry. That instantaneous weighted identity cannot be
integrated as conservation of `sum(theta_i)` or of `sum(H_i*theta_i)`.
The calculation below shows that the arithmetic phase mean actually changes.

### The first nonzero retained offset

Put `c=nu*w/(2*beta*pi*C)`. Expansion at the twist gives

\[
H_i^{-1}=\frac1{2\pi C}
 \left[1+\frac{\tan\kappa}{2}(Jv)_i+O(\|v\|^2)\right],
\]

and hence

\[
\dot m=\frac{c\tan\kappa}{10}(Bu)^TJv
        +O(\|u\|\,\|v\|^2).
\]

The term linear in form vanishes because `sum(Bu)=0`. For centered
preparations `u(0)=epsilon*u_0`, `v(0)=epsilon*v_0`, their first variations
obey

\[
\dot u_1=-aBu_1-bBv_1,\qquad \dot v_1=cBu_1.
\]

The commuting identities `B^T=B`, `J^T=-J`, `BJ=JB` imply

\[
\frac{d}{dt}(u_1^TJv_1)=-a\,u_1^TBJv_1.
\]

The other two terms are zero quadratic forms of the skew matrix `BJ`.
On every nonconstant Fourier mode the two-row tangent matrix has negative
trace `-a*lambda` and positive determinant `b*c*lambda^2`; all centered
first variations decay. Integrating the preceding identity therefore gives

\[
\int_0^\infty u_1^TBJv_1\,dt=\frac1a u_0^TJv_0.
\]

Consequently the limiting phase offset satisfies the conditional asymptotic
law

\[
\boxed{\quad
m_\infty-m(0)=
\frac{w\tan\kappa}{10\beta\pi e\cos\kappa}
\,\epsilon^2u_0^TJv_0+O(\epsilon^3).
\quad}
\]

This is an integrated nonlinear effect of the supplied nodal phase metric,
not a fitted trajectory or a new phase equation. Common capacity cancels
from its coefficient. In fact changing the common positive capacity
multiplies both evolution rows by the same factor, so it changes the
relaxation clock but not the limiting offset from a fixed preparation.
No claim uniform in `e -> 0`, or independent of the selected phase law,
follows.

For the negative-winding reference, use `kappa=-2*pi/5` throughout.
The sine and skew coefficient change sign; cosine, the positive metric
at the twist and the recovery barrier remain the same. This is the second
sector admitted by the shared coefficient helper, not another law.

The infinite-time remainder is justified by local stability, not by
integrating a fixed-time expansion without control. For fixed model
coefficients, the smooth quotient field and its Hurwitz linearization give
constants independent of sufficiently small `epsilon` such that
`||z(t,epsilon)||<=K*|epsilon|*exp(-lambda*t)` and
`z(t,epsilon)=epsilon*z_1(t)+O(epsilon^2*exp(-lambda_1*t))`, with positive
decay exponents. Variation of constants supplies the second bound after
shrinking the neighborhood if needed. The quadratic mean-rate expression
then differs from its first-variation value by an integrable
`O(epsilon^3)` bound; its cubic remainder is integrable as well. This proves
the displayed asymptotic coefficient and finite phase limit. This general
derivation supplies no numerical remainder constant at a chosen nonzero
amplitude; [Section 14](#relational-finite-phase-memory) closes that separate
obligation for one fixed pair and model.

This result does not exclude invariants involving the internal coordinates.
Indeed, the limiting phase defines a local asymptotic-phase projection that
is constant along each recovering trajectory. Its quadratic expansion is
the expression above, including the internal skew product. What fails to
be conserved is the bare arithmetic phase mean, not every possible
state-dependent phase coordinate.

<a id="asymptotic-origin-coordinate-scope"></a>

The general local mechanism clarifies the distinction between a drifting
mean and a conserved origin coordinate. In a declared smooth relative chart
write `y'=f(y)` and `m'=h(y)`, with `h(0)=0`. Assume a forward-invariant
neighborhood on which the relative flow `Phi_t` and its initial-state
derivative satisfy uniform exponential decay bounds

\[
\|\Phi_t(y)\|\le C e^{-\lambda t}\|y\|,\qquad
\|D_y\Phi_t(y)\|\le C e^{-\lambda t},\qquad \lambda>0,
\]

and let `h` be `C^1` with bounded derivative there. These are local recovery
and regularity hypotheses, not consequences of common-origin symmetry
alone. They make the following integral and its first derivative converge
uniformly on smaller neighborhoods:

\[
\psi(y)=\int_0^\infty h(\Phi_t(y))\,dt,\qquad
D\psi(y)f(y)=-h(y),\qquad I=m+\psi(y),\qquad \dot I=0.
\]

The derivative identity follows by shifting the lower integration limit
along the flow. The map `(y,m)->(y,m+psi(y))` is an invertible local chart,
and `I` is the limiting origin of the recovering trajectory. Conversely,
such a time-independent triangular chart removes mean drift only if its
correction solves that derivative equation. On any larger proposed domain,
a closed relative orbit of period `T` necessarily satisfies
`integral_0^T h(Phi_t(y)) dt=0` for a single-valued correction to exist.
The local recovery result does not supply a global chart through other
basins or recurrent relative motion.

For an exact scalar control, `y'=-lambda*y`, `m'=c*y^2`, `lambda>0`, has
`psi=c*y^2/(2*lambda)`: its bare mean drifts while this nonlinear origin is
constant. The [static controls](../../tests/physics/test_relational_environment_response.py)
verify both transformed rows without evolving a graph. This construction
depends on the selected complete law and the retained relative state;
eliminated coordinates can carry information needed by `psi`. It is neither
the bare linear form charge `Q` nor, by default, a primitive-local chart or
an independently justified constitutive law. Circular phase origins retain
their chosen lift and local domain throughout.

### Two equal-mean preparations with opposite retained offsets

For a concrete Fourier pair, use

\[
u_{0,i}^{\pm}=\pm\cos(i\kappa),\qquad
v_{0,i}=\sin(i\kappa).
\]

Choose one shared origin `bar(x)=m(0)=0` for both rings; any common initial
offsets are recovered by the exact shift symmetries. This choice does not
recenter the components independently after they evolve.

Both have zero means, the same initial primitive phases and winding, and
identical form/phase storage. Since
`J*sin(i*kappa)=2*sin(kappa)*cos(i*kappa)` and
`sum(cos(i*kappa)^2)=5/2`, the preceding result becomes

\[
m_\infty^{\pm}-m(0)
=\pm A\epsilon^2+O(\epsilon^3),\qquad
A=\frac{w\tan^2\kappa}{2\beta\pi e}>0.
\]

The signs are nonzero and opposite for all sufficiently small positive
amplitudes. There is also an exact comparison symmetry: the map
`(x,theta) -> (-R*x,-R*theta)`, where `(Rz)_i=z_(-i)`, preserves the
positive-winding reference modulo fixed nodewise turns. It maps the positive
form preparation to the negative one while leaving the sine phase
preparation unchanged. Relabeling and simultaneous form/phase reversal are
symmetries of this law with common capacity. With `m(0)=0`, uniqueness and
the same continuous lift give `m_infinity^-=-m_infinity^+` exactly within
their admitted recovery family.

Capture can be admitted separately from the offset's asymptotic error. For
general centered `u_0,v_0`, it is sufficient that

\[
|\epsilon|\max_i|v_{0,i+1}-v_{0,i}|<\pi/10,
\qquad
\frac{\epsilon^2}{2}
\left(u_0^TBu_0+\beta v_0^TBv_0\right)
<\beta(V_{\rm face}-V_*).
\]

The first condition retains strict acute gaps. The second bounds storage
above its critical twist value, using the zero first variation and
`abs(cos(delta))<=1` in the second variation; it places the state inside
the isolated-ring basin. For the displayed Fourier pair, `beta=1` and
`|epsilon|<=1/32` suffice: the storage excess is at most `5*epsilon^2`,
while

\[
V_{\rm face}-V_*=5\cos(2\pi/5)-4\cos(3\pi/8)>\frac{13}{1000}.
\]

For example, the exact radical formulas give
`cos(2*pi/5)>309/1000` and `cos(3*pi/8)<383/1000`, proving this lower
bound. The gap perturbation is at most `2*|epsilon|<pi/10`.
These explicit basin conditions prove recovery; they do not certify the
size or sign of the truncated offset formula at `epsilon=1/32`.

### A signed operational readout, separate from attachment work

An isolated component's common phase is not an intrinsic gauge-independent
memory label. Keep the untouched reference ring, the inherited common frame
and a declared matching port on each ring. After ideal recovery, the two
forms have the same conserved mean and their relative port phase is
`delta=m_infinity-m(0)`, on the small continuous branch. A hypothetical
unit attachment at those ports has storage cost

\[
\Delta E_{\rm attach}=\beta[1-\cos\delta]
=\frac{\beta A^2}{2}\epsilon^4+O(\epsilon^5).
\]

That cost is even in `delta`. In particular it is exactly equal for the
opposite-offset preparations and cannot distinguish their signs.

The fresh signed port response does distinguish them. Put the recovered
pattern on the left, with phase offset `delta`, and the untouched reference
on the right. Their relative port resultants after the hypothetical
attachment are `2*C+exp(-i*delta)` and `2*C+exp(i*delta)`. All form
gradients are zero at this prepared instant. Therefore

\[
\begin{aligned}
g_L&=\pi^{-1}\operatorname{Arg}(2C+e^{-i\delta})
    =-\frac{\delta}{\pi(1+2C)}+O(\delta^3),& g_R&=-g_L,\\
\dot x_L&=\nu w g_L
    =\mp\frac{\nu w A}{\pi(1+2C)}\epsilon^2+O(\epsilon^3),&
\dot x_R&=-\dot x_L.
\end{aligned}
\]

The immediate phase rates remain zero because `q=0`; the form response is
the nodal phase-pressure channel. The already established
[attachment observation](RELATIONAL_PATTERN_COMPOSITION.md#represented-admission-and-shared-implementation)
evaluates such candidate state/support data without executing an edge event.
An actual attachment with nonzero cost still requires declared work or
another admitted state/event budget. Its occurrence and timing are not
derived by this readout.

The retained information is thus a relation between the recovered pattern
and the declared reference, accessible through a signed port observation
even when winding, internal geometry, mean form and attachment cost agree.
It is not an absolute phase, a selected physical clock, a complete coarse
state, or a demonstration of physical memory or spin. Preparation, reference,
selected law and readout remain explicit. No finite-amplitude trajectory,
finite-time completion of relaxation or numerical asymptotic-error bound
has been evaluated by this general derivation. The following fixed-case
proof supplies a finite-amplitude bound without evaluating a trajectory.

The shared
[`bound_relational_cycle_memory`](../../src/tnfr/physics/relational_cycle_memory.py)
kernel retains exact centered preparation directions, the sufficient
capture-amplitude family and the bounded leading coefficient as separate
evidence. Its `RelationalCycleMemoryBounds` report does not assign an
uncomputed finite-amplitude remainder or evolve a graph. The
[coefficient controls](../../tests/test_relational_cycle_memory.py) include
the exact tangent skew integral and the direction/basin contracts. The
[mechanism controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
check the native mean-rate cross term and the existing hypothetical
attachment's signed response versus even storage cost.

## 14. A finite-amplitude retained-phase certificate without a trajectory

<a id="relational-finite-phase-memory"></a>

Fix `e=w=1/2`, `beta=nu=1`, the isolated winding-one C5 and untouched
reference of Section 13. The two supplied centered directions are now the
exact rational vectors

\[
U=(1,-1,0,0,0),\qquad V=(0,1,-1,0,0),\qquad
u(0)=\pm\epsilon U,\quad v(0)=\epsilon V,\quad m(0)=0.
\]

They have identical initial means, phases and storage, and
`U^T*J*V=2`. They are **not** the Fourier mirror pair from Section 13;
no exact opposite-offset or equal-attachment-cost symmetry is assumed for
their subsequent responses. The following prior inequalities bound the
entire infinite recovery. A dyadic amplitude is selected only after those
inequalities have been established, without sampling either response.

### Storage controls the state and the integrated form response

Let `z=(u,v)` have the Euclidean norm and use the ball of radius
`r=1/100`. In this ball every edge differs from its twist gap by at most
`2*r`; thus its cosine exceeds `C-2*r>7/25`, with
`3/10<C=cos(kappa)<1/2`. The centered C5 Laplacian satisfies
`lambda_min(B)>1`, `||B||<=4`, and `||J||<=2`. Taylor's integral formula
at the critical twist consequently gives

\[
\mathcal E:=E-V_*\ge\frac18\|z\|^2.
\]

For the specified directions, `U^T*B*U=V^T*B*V=6`, so the initial
excess obeys `mathcal E(0)<=6*epsilon^2`. If
`0<epsilon<=1/1024`, this is strictly below `r^2/8`; the initial state
is inside the ball. Storage nonincrease prevents a first exit and yields

\[
\sup_{t\ge0}\|z(t)\|\le7\epsilon,\qquad
\dot{\mathcal E}=-\frac14\|Bu\|^2,
\qquad
\int_0^\infty\|u\|^2dt\le24\epsilon^2.
\]

The ball stays strictly acute and regular. The isolated-ring capture theorem
gives `u,v -> 0`; no quantitative exponential constant is needed in the
estimates below. Common offsets are retained as in Section 13.

### A cross term controls the integrated phase deformation

Let `B^+` be the inverse on the centered subspace, extended by zero on
constants, and let `Pi` be the orthogonal centering projection. Write

\[
a=\tfrac14,\qquad b=\frac1{4\pi}>\tfrac1{16},\qquad
M=\operatorname{diag}\left(\frac1{2H_i}\right),\qquad
\dot v=\Pi MBu.
\]

The explicit two-neighbor metric gives `0<M_i<1/2` throughout the ball.
For `F=u^T*B^+*v`, differentiation of the actual rows gives

\[
\dot F=-a\,u^Tv-b\|v\|^2+u^TB^+\Pi MBu
\le-\frac1{32}\|v\|^2+\frac52\|u\|^2.
\]

Here `||B^+||<1` bounds the last term by `2*||u||^2`, and
`|u^T*v|/4<=||u||^2/2+||v||^2/32` supplies the remaining constants.
Since `|F(0)|<=2*epsilon^2` and capture gives `F(infinity)=0`, integration
and Cauchy--Schwarz imply

\[
\int_0^\infty\|v\|^2dt\le1984\epsilon^2<2048\epsilon^2,
\qquad
\int_0^\infty\|u\|\,\|v\|dt\le224\epsilon^2.
\]

These are whole-recovery integral estimates. They are neither sampled
losses nor a claim that capacity, storage or a phase mean can be omitted.

### An integrated cubic remainder for the approximate phase invariant

Put `d=1/(4*pi*C)<1/3`, `tau=tan(kappa)<10/3`, and
`K=d*tau/(10*a)<1/2`. For each node define

\[
t_i=(Jv)_i/2,\qquad s_i=(Bv)_i/2,\qquad
h_i=\frac{C}{\cos(\kappa+t_i)\operatorname{sinc}s_i}.
\]

The denominator after dividing by `C` is
`D_i=(cos(t_i)-tau*sin(t_i))*sinc(s_i)`. Since
`|t_i|<=||v||<=1/100` and `|s_i|<=2*||v||`, the elementary sine/cosine
remainders give

\[
D_i>\frac9{10},\qquad
D_i=1-\tau t_i+d_{2,i},\quad |d_{2,i}|\le2\|v\|^2.
\]

For example, the three contributions to `d_2` are `cos(t)-1`,
`tau*(t-sin(t))` and `(cos(t)-tau*sin(t))*(sinc(s)-1)`; the bounds
`t^2/2`, `tau*|t|^3/6` and `(1+tau*|t|)*s^2/6` suffice.
Dividing by the positive `D_i` now proves

\[
|h_i-1|\le4\|v\|,\qquad
|h_i-1-\tau t_i|\le17\|v\|^2.
\]

These are bounds on the actual inverse metric, including its zero-source
limit, rather than a substituted polynomial phase law.

Use the quadratic approximation to the asymptotic phase,

\[
I_2=m+K u^TJv.
\]

The exact linear form row and the quadratic mean-rate term cancel in
`I_2_dot`. The remaining mean-rate term is bounded by
`14*||u||*||v||^2`; the nonlinear correction in
`K*u^T*J*v_dot` is bounded by `6*||u||^2*||v||`. These follow directly
from the two inverse-metric inequalities, `||B||<=4`, `||J||<=2`,
`d<1/3`, `K<1/2` and `||Pi||=1`. Consequently

\[
\begin{aligned}
|\dot I_2|&\le14\|u\|\|v\|^2+6\|u\|^2\|v\|,\\
\left|m_\infty-I_2(0)\right|
&\le\left(14\cdot7\cdot224+6\cdot7\cdot24\right)\epsilon^3\\
&=22960\epsilon^3<2^{15}\epsilon^3.
\end{aligned}
\]

The limiting skew product is zero because both centered coordinates decay.
Thus this is an explicit error bound on the limiting phase itself, obtained
without an infinite-time numerical integration or an assumed decay rate.

### One declared nonzero amplitude and its signed readout

Select

\[
\epsilon=2^{-20}
\]

after the preceding uniform proof. It is inside the stated trapping family.
Let

\[
\mathcal C=2K=\frac{\sin\kappa}{5\pi\cos^2\kappa}.
\]

Certified trigonometric bounds give `3/5<mathcal C<1`. For instance,
`cos(kappa)<31/100` implies `cos(kappa)^2<1/10` and
`sin(kappa)>19/20`; together with `pi<22/7`, these give
`mathcal C>133/220>3/5`. The prior remainder bound now yields

\[
\boxed{\qquad
\left|m_\infty^\pm\mp\mathcal C\,2^{-40}\right|<2^{-45}.
\qquad}
\]

In particular,

\[
m_\infty^+>\frac{91}{160}\,2^{-40}>0,\qquad
m_\infty^-<-\frac{91}{160}\,2^{-40}<0,\qquad
|m_\infty^\pm|<\frac{33}{32}\,2^{-40}<\frac14.
\]

These symmetric **enclosures** do not assert that the two actual offsets
are exact negatives. They prove nonzero offsets of opposite signs for the
two finite, rationally directed preparations.

For either offset `delta`, the same hypothetical matching-port attachment
to the untouched reference has

\[
\dot x_L=\frac1{2\pi}\operatorname{Arg}(2C+e^{-i\delta}),
\qquad \dot x_R=-\dot x_L.
\]

Because `|delta|<1/4` and `2C+cos(delta)>0`, the left response has the
opposite sign to `delta` and cannot be zero. Interval evaluation of this
same expression preserves the two separated readouts. The nonnegative
cost `1-cos(delta)` remains even; unlike the Fourier mirror case, equal
costs for these rational preparations have not been proved. Any actual
attachment still needs its independently declared event budget and timing.

The shared
[`certify_relational_cycle_memory`](../../src/tnfr/physics/relational_cycle_memory.py)
returns this fixed-case `RelationalFiniteMemoryCertificate`, with capture,
integrated-remainder and signed readout bounds kept separate. The
[finite-memory controls](../../tests/test_relational_finite_memory.py) check
admission and the nonlinear interval consequences. The separate
[exact proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
verify the derivative cancellation, integral budgets and centered inverse
without advancing a trajectory. This does not replace the general coefficient report's
unavailable remainder by a bound for arbitrary models or directions.

The result is an analytic finite-amplitude distinction in the limiting
relative state. It supplies no finite recovery duration or finite-time
readout guarantee, and does not identify physical memory, spin or a unique
law of nature. The small amplitude is a conservative mathematical witness,
not a fitted physical scale or an optimal robustness radius.

## 15. A finite-time contact readout with residual-state and event budgets

<a id="relational-finite-time-memory"></a>

Retain exactly the law, two rational preparation directions, untouched
reference and amplitude `epsilon=2^-20` of Section 14. The clock is the
declared structural clock with `nu=1`; the numbers below are not laboratory
seconds. The proposed contact joins position zero of the recovering left
ring to position zero of the untouched right ring, preserving both states.
No edge is executed in this calculation.

Convergence alone already implies that sufficiently late readouts have the
limiting signs. Here the additional result is a prior quantitative time and
a full-state error budget. In particular, the remaining form deformation is
not set to zero, and the finite-time phase is not replaced by its limit.

### An explicit decay estimate on the already admitted ball

Use the same centered state `z=(u,v)`, storage excess `mathcal E` and cross
term `F=u^T*B^+*v` as in Section 14. In the trapped radius-`1/100` ball,
Taylor's formula and `||B||<=4` give

\[
\frac18\|z\|^2\le\mathcal E\le2\|z\|^2,
\qquad |F|\le\frac12\|z\|^2.
\]

The auxiliary proof functional

\[
\mathcal H=\mathcal E+\frac1{32}F
\]

is not a new physical storage or a change to the dynamics. It satisfies

\[
\begin{aligned}
\frac7{64}\|z\|^2&\le\mathcal H\le\frac{129}{64}\|z\|^2,\\
\dot{\mathcal H}
&\le-\frac{11}{64}\|u\|^2-\frac1{1024}\|v\|^2
\le-\frac1{1024}\|z\|^2
\le-\frac1{2064}\mathcal H.
\end{aligned}
\]

The first differential inequality uses exactly
`mathcal E_dot=-||Bu||^2/4`, `lambda_min(B)>1`, and the previously proved
cross inequality `F_dot<=-||v||^2/32+(5/2)*||u||^2`.
Since `mathcal H(0)<=(97/16)*epsilon^2`, integration yields

\[
\|z(t)\|\le8\epsilon e^{-t/4128}=:R(t).
\]

The existing first-exit argument keeps the trajectory in the ball on which
these inequalities hold. This estimate therefore covers the entire
continuous recovery, rather than assuming that a finite numerical endpoint
has entered a local regime.

### A phase-tail bound in the retained reference frame

The exact mean phase rate is
`m_dot=(d/5)*sum_i(h_i-1)*(Bu)_i`, because `sum_i(Bu)_i=0`.
Here `d<1/3` and `|h_i-1|<=4*||v||` on the same ball. Consequently,

\[
|\dot m|
\le\frac{16\sqrt5}{15}\|u\|\|v\|
\le\frac32\|z\|^2.
\]

Integrating the squared decay envelope from the observation time gives

\[
|m_\infty-m(t)|
\le3096 R(t)^2<4096R(t)^2=:Q(t).
\]

Thus each recovering-ring phase differs from its limiting twist-plus-offset
by at most `R(t)+Q(t)`. Its form differs from the common zero form by at most
`R(t)`. The untouched reference remains exactly at its supplied equilibrium.

For a nonnegative integer `n`, choose the declared observation time
`T_n=4128*n`. Since `exp(1)>2`, the exact rational envelopes

\[
R_n=8\epsilon\,2^{-n},\qquad Q_n=4096R_n^2
\]

bound the residual state and phase tail at that time. They also bound all
later times provided recovery continues without an event. No transcendental
time inversion or trajectory evaluation is required.

### Fresh contact rates, including the no-contact comparison

At the declared time write `delta_t=m(t)+v_0(t)` for the relative port phase
and abbreviate `R=R_n`, `Q=Q_n`. The two relative neighbor resultants after
the hypothetical contact are exactly

\[
\begin{aligned}
Z_L={}&e^{i(\kappa+v_1-v_0)}
       +e^{i(-\kappa+v_4-v_0)}+e^{-i\delta_t},\\
Z_R={}&2C+e^{i\delta_t}.
\end{aligned}
\]

The state-dependent form gradients and degree-three rates are

\[
\begin{aligned}
q_L&=3u_0-u_1-u_4,&
\dot x_L^{\rm contact}&=-\frac{q_L}{6}
                         +\frac{\operatorname{Arg}Z_L}{2\pi},\\
q_R&=-u_0,&
\dot x_R^{\rm contact}&=\frac{u_0}{6}
                         +\frac{\operatorname{Arg}Z_R}{2\pi}.
\end{aligned}
\]

These are the actual native rows on the proposed support. They need not be
opposites at finite time. The residual `q` also permits nonzero immediate
phase rates; the zero-phase-rate statement for the ideal recovered contact
in Section 13 does not apply here.

For comparison, on the unchanged isolated support the left rate is

\[
\dot x_L^{\rm free}=-\frac{(Bu)_0}{4}
                    -\frac{(Bv)_0}{4\pi},
\qquad
|\dot x_L^{\rm free}|\le\frac43R,
\qquad \dot x_R^{\rm free}=0.
\]

This baseline distinguishes the effect of the proposed contact from motion
that would already have occurred without it.

There are two equivalent ways to enclose the readout. Direct rational interval
evaluation uses `u_i,v_i in [-R,R]`, the Section 14 enclosure for
`m_infinity`, and `m(t)-m_infinity in [-Q,Q]` in the exact formulas above.
Discarding the correlations among those intervals only enlarges the bounds.
The contact increment is enclosed separately by subtracting the no-contact
baseline. The following simpler estimates also prove sign admission before
that evaluation.

Let `r_L^infinity` and `r_R^infinity` be the ideal limiting contact rates from
Section 14. The resultant differences from that ideal contact satisfy

\[
|Z_L-(2C+e^{-im_\infty})|\le5R+Q,\qquad
|Z_R-(2C+e^{im_\infty})|\le R+Q.
\]

For `|m_infinity|<1/4`, both ideal real parts exceed `3/2`. If
`5R+Q<1/2`, the straight segments to the actual resultants stay in `Re Z>1`;
therefore `|d Arg Z|<=|dZ|` along them. Combining this fact with the form
terms and `pi>3` gives

\[
\begin{aligned}
|\dot x_L^{\rm contact}-r_L^\infty|
&\le\frac53R+\frac16Q<2R+Q,\\
|\dot x_R^{\rm contact}-r_R^\infty|
&\le\frac13R+\frac16Q<2R+Q,\\
|\dot x_L^{\rm contact}-\dot x_L^{\rm free}-r_L^\infty|
&\le3R+\frac16Q<4R+Q.
\end{aligned}
\]

The right contact increment equals its contact rate, since its baseline is
zero. These are bounds on one-sided hypothetical rates at the supplied
state, not on a trajectory following the event.

### One finite horizon and its separate event-work budget

Select one sufficient horizon from those prior inequalities:

\[
n=40,\qquad T=165120,\qquad R=2^{-57},\qquad Q=2^{-102}.
\]

There is no search for the first or fastest readable time. Both the internal
ring edges and the proposed contact are strictly acute at this horizon;
`5R+Q<1/2` also certifies the resultant branches used above.

For completeness, Section 14 gives
`|m_infinity|>(91/160)*2^-40`. For `0<|delta|<1/4`,
`|sin(delta)|>|delta|/2`, `2C+cos(delta)<2`, and
`atan(y)>y/2` for the relevant `0<y<1`. With `pi<4`, these imply

\[
|r_L^\infty|=|r_R^\infty|
>\frac{|m_\infty|}{64}>2^{-47}.
\]

At the chosen horizon `4R+Q<2^-54`. Hence the finite-time left and right
contact rates, and their respective contact-minus-no-contact increments,
retain the certified limiting signs in each preparation. The two
preparations give opposite signs. Each arm keeps its own enclosure; their
actual magnitudes are not asserted to be equal.

The storage jump for the same state-preserving unit contact is separately

\[
\Delta E_{\rm attach}(T)
=\frac12u_0(T)^2+1-\cos\delta_T
=\frac12u_0(T)^2+2\sin^2\!\left(\frac{\delta_T}{2}\right).
\]

Here `|u_0|<=R` and
`delta_T in [m_infinity_lower-R-Q,m_infinity_upper+R+Q]` give a direct
outward enclosure. At this horizon the port-phase intervals exclude zero,
so the jump is strictly positive. The form contribution is bounded by
`R^2/2`; it is not silently discarded. The earlier continuous loss is not
an event reservoir. Executing this contact would require a supplied event
and its positive work or another independently admitted budget.

The shared
[`certify_relational_cycle_memory_readout`](../../src/tnfr/physics/relational_cycle_memory.py)
returns a `RelationalMemoryReadoutCertificate` for nonnegative integer
`decay_blocks`, retaining the fixed amplitude and preparation. Its default
40 blocks give the witness above. Earlier inconclusive bounds remain
explicitly unavailable instead of reporting a zero signal or an executed
measurement. The reader encloses the full contact, no-contact and event
expressions; it neither advances a graph nor inserts an edge.
The [readout controls](../../tests/test_relational_memory_readout.py) check
the exact report and unavailable-result contracts. The
[proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
verify the cross-storage decay, tail and horizon budgets; the
[native mechanism controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
check finite residual contact/no-contact rates, their support degrees and
the state-preserving storage jump.

This closes the finite-time readout obligation for this supplied experiment
within the native model. It does not certify post-contact maintenance,
nondestructive readout, an autonomous connection schedule, optimal timing,
laboratory units, physical memory or a uniquely selected microscopic law.

## 16. A finite contact transmits a signed form response

<a id="relational-finite-contact-memory"></a>

Keep the two preparations, amplitude, reference and observation time
`T=165120` from Section 15. Supply the state-preserving matching-port unit
connection at that time and retain it for one declared duration `h>0`.
The joined graph consists of two C5 rings and one bridge, with unit held
capacities, `e=w=1/2`, `beta=1`, no forcing and the same native relational
law. The comparison without contact continues the two original isolated
flows from exactly the same state at `T`.

The result below concerns continuous model solutions. The certificate does
not execute the connection, sample a trajectory, choose the event or supply
its work. Its new observation is an accumulated form change over a nonzero
duration, including the receiver's response, rather than an instantaneous
rate at the proposed event.

### A state-scaled continuous-flow bound

Use the full twenty-coordinate state deviation `z` from the joined
equilibrium with zero forms and two aligned winding-one twists. In particular,
retain both common phases and all form coordinates; no centering or regional
mean constraint is imposed on the joined flow. Write `rho=||z||_infinity`.
For each preparation a sufficient initial bound is

\[
\rho_0=\max\{|m_{\infty,\rm lower}|,|m_{\infty,\rm upper}|\}
       +Q+R<2^{-39},
\qquad R=2^{-57},\quad Q=2^{-102}.
\]

Consider the fixed sup-norm ball of radius `r=1/100`. Each node has degree
`d_i` equal to two or three. If its relative neighbor resultant is
`C_i+i*S_i`, the equilibrium has `S_i=0` and each incident equilibrium
cosine is at least `C=cos(2*pi/5)>3/10`. The unit Lipschitz bounds for sine
and cosine therefore give, throughout the ball,

\[
C_i\ge\frac7{25}d_i>0,\qquad
|S_i|\le2d_i\rho,\qquad |q_i|\le2d_i\rho.
\]

With `c_0=7/25`, the phase metric and its zero-source extension satisfy

\[
H_i=\frac{\pi S_i}{\arctan(S_i/C_i)}\ge\pi C_i,
\qquad H_i\big|_{S_i=0}=\pi C_i.
\]

There is no singular phase row at `S_i=0`: its mobility can also be written
as the smooth function

\[
\frac1{2H_i}
=\frac1{2\pi}\int_0^1\frac{C_i}{C_i^2+s^2S_i^2}\,ds.
\]

The actual native form and phase rows consequently obey

\[
|\dot\theta_i|\le\frac{\rho}{\pi c_0}\le\frac{25}{21}\rho,
\qquad
|\dot x_i|\le\left(1+\frac1{\pi c_0}\right)\rho\le3\rho.
\]

The form row alone is also uniformly Lipschitz with constant three in this
norm. Its form-coordinate derivative row sum is one. The phase derivative
of the resultant argument has row sum at most `2*d_i/|C_i+i*S_i|`, so its
contribution after multiplying by `1/(2*pi)` is at most `1/(pi*c_0)`.
The ball is convex, and this derivative bound therefore compares any two
states in it. A bound on a phase-row derivative is not needed below.

Let local time `s` start at contact. Gronwall's inequality and a first-exit
argument now give, whenever `rho_0*exp(3*h)<r`,

\[
\begin{aligned}
\|z(s)\|_\infty&\le\rho_0e^{3s},\\
\|z(s)-z(0)\|_\infty&\le\rho_0(e^{3s}-1),\\
|\dot x_i(s)-\dot x_i(0)|&\le3\rho_0(e^{3s}-1),
\qquad 0\le s\le h.
\end{aligned}
\]

Integration yields the state-scaled remainder

\[
\left|x_i(h)-x_i(0)-h\dot x_i(0)\right|
\le\rho_0(e^{3h}-1-3h).
\]

This error vanishes with the actual prepared displacement. It does not
replace the small memory response by an amplitude-independent acceleration
bound.

### One duration with accumulated contact and no-contact responses

For `0<h<=1/3`, the shared exact-clock exponential bounds apply to `3*h`.
If `E_hi` encloses `exp(3*h)` from above, use

\[
\rho_{\rm tube}=\rho_0E_{\rm hi},\qquad
D=\rho_0(E_{\rm hi}-1),\qquad
B=\rho_0(E_{\rm hi}-1-3h).
\]

These respectively enclose the state tube, total state disturbance and
form-response remainder. The contact rate intervals `[a_i,b_i]` from
Section 15 give accumulated form changes in `[h*a_i-B,h*b_i+B]`.
Without contact, the continued isolated decay keeps the left form rate in
`[-4*R/3,4*R/3]` at every later time, while the reference rate stays zero.
Thus the no-contact form changes lie in `h*[-4*R/3,4*R/3]` and `{0}`.
Subtract these enclosures to retain the contact-induced changes as a
separate comparison; no second joined-flow remainder is required.

Choose the single duration

\[
h=2^{-12}.
\]

The inequalities already prove that this duration is sufficient before
evaluating any trajectory. Since `exp(3*h)<2`,
`rho_tube<2*rho_0<r`, `D<6*rho_0*h`, and
`B<9*rho_0*h^2`. Section 15 gives each initial contact-rate magnitude
greater than `(255/256)*2^-47`. Meanwhile,

\[
\frac Bh<\frac9{16}\,2^{-47},\qquad
\frac43R<\frac1{512}\,2^{-47}.
\]

Therefore both accumulated contact form changes, and their respective
contact-minus-no-contact changes, keep their predicted signs. Even after
both error budgets, the per-duration sign margin exceeds
`(221/512)*2^-47`. The two preparations produce opposite signs. In
particular, the initially unperturbed right ring acquires a nonzero signed
form change during the contact; its no-contact change is exactly zero.
The two arms have independent bounds, not an assumed equality of response
magnitudes.

Every original edge remains strictly acute because its phase gap differs
from `+/-kappa` by at most `2*rho_tube<2*r`. The bridge gap is at most
`2*rho_tube`. Continuous motion in this domain preserves the winding-one
identity of each original cycle throughout the declared contact duration.
This is not a claim of unchanged internal shape: the bounded form/phase
disturbance is the interaction being measured.

### Event work and continuous loss retain different accounts

The initial attachment cost is exactly the finite-time cost from Section 15;
its enclosure includes the residual form mismatch. The maximum of the two
cost upper bounds is a sufficient prospective work budget for either
preparation; it is not asserted to be the minimal required work. This is a
conditional budget bound, not an inferred reservoir or an executed supply
of work.

On the joined support the continuous loss is nonnegative and satisfies

\[
L=\frac12\sum_i\frac{q_i^2}{d_i}
\le2\rho_{\rm tube}^2\sum_i d_i
=44\rho_{\rm tube}^2,
\qquad
0\le\int_0^h L\,ds\le44\rho_{\rm tube}^2h,
\]

because the eleven-edge graph has total degree twenty-two. This optional
loss bound is separate from the event jump; it is not work available to pay
for insertion and does not make the addition passive. No removal event or
post-contact continuation is included in this protocol.

The shared
[`certify_relational_memory_contact`](../../src/tnfr/physics/relational_memory_contact.py)
returns a `RelationalMemoryContactCertificate` and per-preparation
`RelationalMemoryContactCase` reports. It reuses the fixed 40-block readout
certificate and the shared exponential enclosure, with default
`duration=1/4096`. Other admitted durations `0<h<=1/3` retain conservative
bounds and explicit sign availability; they do not select an optimal
contact time. The
[contact controls](../../tests/test_relational_memory_contact.py),
[continuous proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
and [native mechanism controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
keep report arithmetic, analytic premises and the engine's nodal rows distinct.

This establishes finite-duration information transfer under one supplied
connection while both cycle windings persist. It does not derive the event's
occurrence, certify nondestructive readout or arbitrarily long contact, or
identify a physical memory carrier, physical time or a unique constitutive law.

## 17. A receiver mean survives contact removal and isolated recovery

<a id="relational-retained-receiver-record"></a>

Keep the complete fixed protocol of Section 16, including `h=1/4096`.
At `T+h`, supply one state-preserving removal of the matching-port bridge.
Then evolve each isolated unit C5 independently under its native unforced
law with held unit capacities and the same structural clock. Recompute
each component's degree, pressure and phase metric on its actual support;
the connected relational executor is not thereby extended to a disconnected
union. No further event, input or capacity change is part of this continuation.

The observation is the receiver's arithmetic mean form relative to its
declared initial mean zero. This differs from the port form change proved
in Section 16. It also differs from winding, internal shape or an absolute
form-origin claim. A common translation of all initial and final forms
leaves this mean **change** invariant.

### Bound the regional mean before invoking conservation

At the start of contact the entire receiver is still at its exact supplied
equilibrium. Only its port acquires a nonzero form rate when the bridge is
added; the other four receiver nodes keep the same neighbor states and
support. If the initial right-port rate is enclosed by `[a_R,b_R]`, the
sum of initial receiver form rates is therefore enclosed by that same
interval.

Let `B=rho_0*(E_hi-1-3*h)` be the per-node continuous remainder from
Section 16. Summing all five form changes and dividing by five gives

\[
\boxed{\qquad
\mu_R(T+h)\in
\left[\frac h5 a_R-B,\ \frac h5 b_R+B\right].
\qquad}
\]

The error is `B`, not `B/5`: all five nodes may contribute a remainder.
The no-contact receiver mean remains exactly zero. No equal-and-opposite
regional transfer is assumed on the irregular joined graph.

The regional intervals have opposite nonzero signs for the two fixed
preparations. This also follows from conservative inequalities independent
of any sampled response. For `0<|delta|<1/4`, use
`|sin(delta)|>(3/4)*|delta|`, `2C+cos(delta)<2`, and
`atan(y)>(2/3)*y` for the relevant `0<y<1`. With `pi<4`, the ideal
contact-rate magnitude exceeds `|delta|/32`. Section 14 gives
`|m_infinity|>(91/160)*2^-40`, and Section 15 changes that rate by less
than `2R+Q<2^-55`.

The already fixed preparation also obeys
`rho_0<(17/16)*2^-40`. At `h=2^-12`, the bound
`B<9*rho_0*h^2` consequently gives

\[
\frac{h}{5}|\dot x_R^{\rm contact}(0)|-B
>
h\left(\frac{91}{25600}-\frac1{163840}-\frac{153}{65536}\right)2^{-40}
=\frac{1989}{1638400}\,2^{-52}>0.
\]

Thus the positive-form preparation writes a positive receiver mean, and
the negative-form preparation writes a negative one. Exact outward
evaluation of the existing intervals provides tighter bounds; it does not
fit or evaluate a new trajectory.

### The cut has its own storage jump

Removal leaves every nodal coordinate, and hence each regional mean,
unchanged. Its exact storage jump is

\[
\Delta E_{\rm cut}
=-\left\{\frac12(x_{L0}-x_{R0})^2
          +1-\cos(\theta_{L0}-\theta_{R0})\right\}\le0.
\]

For example, the whole-contact state radius `rho_tube` immediately gives
`-4*rho_tube^2<=Delta E_cut<=0`. A sharper endpoint enclosure reuses
the already certified all-coordinate disturbance `D`: the final bridge
phase lies in its initial interval enlarged by `2D`, and the final form
mismatch lies in `[-R-2D,R+2D]`. Evaluate the nonnegative edge cost with
these intervals and negate its endpoints to enclose the jump. The phase
intervals still exclude zero in this fixed case, so this sharper jump is
strictly negative.

Deletion therefore adds no positive storage requirement. It does not
retroactively fund the positive insertion work, select when either event
occurs or identify the removed edge storage with an accessible reservoir.

### Both isolated rings remain in their recovery basins

For either ring at the cut, every form and lifted phase deviation from its
original aligned twist is at most `rho_tube`. Each form-edge difference is
at most `2*rho_tube`, giving at most `10*rho_tube^2` of form storage over
the five edges. For phase storage, write the oriented edge gaps as
`kappa+v_(i+1)-v_i`. The linear Taylor terms around the twist sum to zero;
the second derivative of `1-cos` is at most one. Thus another
`10*rho_tube^2` bounds the phase excess, and

\[
0\le E_R-V_*\le20\rho_{\rm tube}^2
<V_{\rm face}-V_*.
\]

The last strict inequality follows already from
`rho_tube<2^-38` and the previously proved
`V_face-V_*>13/1000`. Section 16 preserves the strict acute winding-one
sector, and the state-preserving cut changes none of its internal gaps.
Both post-cut components therefore satisfy the
[isolated-ring capture theorem](RELATIONAL_PATTERN_COMPOSITION.md#relational-pattern-detachment).
Each remains regular, retains its winding and converges to a uniform form
and uniform twist, with its own common phase offset.

### The retained mean is exactly conserved during that recovery

On a strictly acute winding-one C5 let `B_5` be the cycle Laplacian and
`v` any consistent lifted phase deformation from the uniform twist. The
two-neighbor circular resultant gives exactly

\[
g=-\frac1{2\pi}B_5v,
\qquad
\dot x=-\frac14B_5x-\frac1{4\pi}B_5v.
\]

The pair of relative neighbor angles stays within an interval of width
less than `pi`, so its argument is their arithmetic midpoint on this
branch; this is the domain needed for the first identity. Since the
Laplacian has zero column sums,

\[
\frac d{dt}\left(\frac15\sum_{i\in R}x_i\right)=0.
\]

Combining conservation with capture identifies the receiver's limiting
uniform form with `mu_R(T+h)`. Its certified nonzero interval is therefore
retained for every later time as a regional mean, even while the individual
node forms continue to reorganize. The isolated no-contact receiver keeps
uniform form zero. The signs distinguish the preparations after the
connection that transmitted the signal has been removed.

This conservation step uses common held capacity and degree-two native
pressure; it is not a generic arithmetic-mean conservation theorem for
irregular networks or heterogeneous capacities. It does not depend on the
particular post-cut phase row beyond remaining in the admitted acute
domain. Accordingly, the same retained mean follows if the post-cut phase
row is replaced by any law already admitted by the isolated capture theorem,
with the same native form row and common capacity. That conditional
continuation does not transfer the preceding preparation or contact-response
calculation to another law.

The shared
[`certify_relational_memory_retention`](../../src/tnfr/physics/relational_memory_contact.py)
reuses the fixed contact report and separately reports the receiver-mean
interval, component capture margins and removal jump. The
[retention controls](../../tests/test_relational_memory_contact.py),
[exact proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
and [native regional controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
check these distinct obligations without advancing a recovery trajectory.

The result is a conditional persistent receiver record in the declared
form coordinate. It does not equate that coordinate with a measured physical
quantity, demonstrate immunity to arbitrary future forcing or topology
changes, select an autonomous contact/cut mechanism, or claim that the
record is encoded by a different winding or internal geometry.

## 18. Relative pattern state without discarding internal dynamics

<a id="sine-relative-pattern-state"></a>

Return to the explicitly supplied normalized-sine law. Fix a finite
connected simple unit graph with at least two nodes, its node identities,
held capacities `nu_i>=0`, coefficients `e>=0,w>0,beta>0` and one
clock, with no forcing, clipping or support events. These are model
premises, not an inferred origin of support. The
[comparison-law owner](RELATIONAL_EXCHANGE_ADMISSION.md#global-closure-pressure-comparison)
establishes its smooth complete field and storage identity; native
trajectory records remain evidence of the native law.

### An exact quotient by two common origins

Choose any node `r` as reference. Define

\[
u_i=x_i-x_r,\qquad
z_i=\exp\!\bigl(i(\theta_i-\theta_r)\bigr),\qquad
u_r=0,\quad z_r=1.
\]

The remaining coordinates lie in
`R^(n-1) x (S^1)^(n-1)`. This is a global circular quotient, not a
global choice of real angle. It identifies precisely one common form
translation and one common phase rotation. It retains all other nodal
coordinates, including hidden or intermediary nodes, and requires the
same support, capacities, coefficients and clock. No regional averaging,
node deletion, phase-magnitude projection or memory truncation occurs.

Put `k_i=nu_i/d_i`, `a=w/pi`, `b=w/(beta*pi)` and compute from the
relative state

\[
\begin{aligned}
q_i&=\sum_{j\sim i}(u_i-u_j),&
S_i&=\sum_{j\sim i}\operatorname{Im}(z_j\overline z_i),\\
V_i&=k_i(-e q_i+aS_i),&
\Omega_i&=b k_iq_i.
\end{aligned}
\]

These are the original nodal form and phase rates: all consumed
differences are unchanged by the quotient. Differentiation gives the
exact closed induced law

\[
\boxed{\dot u_i=V_i-V_r,\qquad
       \dot z_i=i(\Omega_i-\Omega_r)z_i.}
\]

The phase row preserves `|z_i|=1`. A real relative lift
`rho_i=theta_i-theta_r` instead obeys
`rho'_i=Omega_i-Omega_r`, with `rho_r=0`. Such lifts are legitimate
on a declared chart or along a continuous trajectory, but their
independent full-turn representatives are not extra physical coordinates.

Conversely, a quotient solution and initial common origins reconstruct
the original solution through

\[
\dot x_r=V_r,\qquad \dot\theta_r=\Omega_r,\qquad
x_i=x_r+u_i,\qquad
\exp(i\theta_i)=\exp(i\theta_r)z_i.
\]

Thus holding the reference coordinate at zero inside the original nodal
equations would change the model whenever its rates are nonzero.
Subtracting the reference rows is essential. A zero-capacity node has
`V_i=Omega_i=0` in the original coordinates, but its relative
coordinates can move when the reference moves. If a zero-capacity node
is chosen as reference, both reconstruction rates vanish and the
remaining relative rates equal their original rows. This special case
does not justify freezing a general active reference.

The storage is a function of this complete relative state:

\[
E(u,z)=\frac12\sum_{\{i,j\}}(u_i-u_j)^2+
       \beta\sum_{\{i,j\}}
       \bigl[1-\operatorname{Re}(z_j\overline z_i)\bigr],
\qquad
\dot E=-e\sum_i k_iq_i^2.
\]

Common-reference velocity contributes no work because
`sum_i q_i=sum_i S_i=0`. Removing the two origins therefore preserves
the actual exchange and dissipation, rather than replacing them with a
diagnostic score.

### The same complete law has a dissipative Hamiltonian representation

On the full form/phase coordinates, the preceding storage has gradient
`grad(E)=(q,-beta*S)`. For the held diagonal matrix
`K=diag(nu_i/d_i)` and `b=w/(beta*pi)`, define

\[
J=\begin{pmatrix}0&-bK\\ bK&0\end{pmatrix},\qquad
R_{\!d}=\begin{pmatrix}eK&0\\0&0\end{pmatrix}.
\]

Direct multiplication, using `beta*b=w/pi`, gives exactly

\[
(\dot x,\dot\theta)=(J-R_{\!d})\nabla E.
\]

Here `J` is constant and skew, so its coordinate derivatives vanish
and its bracket satisfies the Jacobi identity. The angular coordinate
fields are well defined on the phase torus; no global real phase lift
is needed for the bracket of smooth periodic functions. Zero capacities
make this Poisson structure degenerate rather than invalid.
`R_d` is positive semidefinite, and
`E'=-grad(E)^T R_d grad(E)=-e*q^T K q` recovers the exact loss.

This is an unforced dissipative Hamiltonian instance of the declared
complete sine model. Network energy storage, interconnection and
dissipation already have an established general framework; see
[van der Schaft and Maschke, *Port-Hamiltonian systems on graphs*](https://arxiv.org/abs/1107.2006).
That reference supplies context, not a proof of this particular TNFR
identity, which follows from the multiplication above. In contrast to
an auxiliary Hamiltonian analogy, the displayed matrices reproduce
both consumed evolution rows. The result remains restricted to held
support and capacities and this pressure/phase law: it neither gives
every TNFR runtime a Poisson structure nor identifies the storage with
physical energy. Such a representation alone establishes no novelty,
physical validation or uniquely selected microscopic law.

For strictly positive capacities this tensor also explains the form
invariant within the selected law. Put `rho=K^-1*1`, `W=sum_i rho_i` and
`Q_x=rho^T*x`. Its Hamiltonian vector field is

\[
J\nabla Q_x=(0,b\mathbf1).
\]

Thus `Q_x/b` generates common phase rotation. Rotation invariance of the
storage gives `{Q_x,E}=0`; the dissipative term also contributes zero,
since `rho^T eKq=e*sum_i q_i=0`. This recovers conservation of `Q_x` from
the same complete field. It does not derive conservation from phase
symmetry alone: choosing the constant reciprocal tensor already fixes an
additional constitutive structure that the native Arg law does not share.

On a continuous real phase lift define `Q_theta=rho^T*theta`. Then

\[
J\nabla Q_\theta=(-b\mathbf1,0),\qquad
\{Q_x,Q_\theta\}=-bW\ne0.
\]

The lifted quantity generates a common form translation with the displayed
sign, but is not a globally single-valued observable on the phase torus.
Neither quantity is a Casimir: their vector fields are nonzero, even though
both are conserved by this particular storage and loss. These identities
are conditional interpretations of the existing bracket, not independent
selection of the bracket, pressure or a physical charge. The
[static matrix controls](../../tests/physics/test_relational_pressure_composition.py)
check the generators, their nonzero mutual bracket and both consumed rows.

### Reconstruction from conserved means when every capacity is positive

For `nu_i>0`, set `omega_i=d_i/nu_i` and `W=sum_i omega_i`.
Reciprocity gives

\[
\sum_i\omega_iV_i=-e\sum_iq_i+a\sum_iS_i=0,\qquad
\sum_i\omega_i\Omega_i=b\sum_iq_i=0.
\]

Hence the weighted form mean `M_x` and the weighted mean `M_theta`
of a chosen continuous phase lift are constant. Reconstruction can then
be written algebraically:

\[
x_r(t)=M_x-\frac1W\sum_i\omega_i u_i(t),\qquad
\theta_r(t)=M_\theta-\frac1W\sum_i\omega_i\rho_i(t).
\]

The lifted mean is not a single-valued scalar on the phase torus.
Changing initial phase representatives changes its representation
consistently, without changing the reconstructed circular state.
These formulas are unavailable at zero capacity; the previous direct
reconstruction remains valid.

There are no nonzero uniformly drifting relative equilibria when every
capacity is positive. A stationary quotient would require
`V_i=v` and `Omega_i=omega` for every node. The two weighted identities
force `v=omega=0`. The phase row then gives `q=0` and the form row
`S=0`. Thus the stationary relative patterns are exactly uniform form
with a critical phase geometry. A common pulse or uniform rotation is
not generated by merely changing the reference.

### What observation uncertainty cancels, and what remains

At one common actual time, suppose reported form coordinates are
`y_i=x_i+C_x+epsilon_i` and consistently lifted phases are
`psi_i=theta_i+C_theta+eta_i`. The shared offsets can be unknown and
can vary between synchronous captures. Taking relative coordinates
first cancels them exactly:

\[
y_i-y_r=u_i+\epsilon_i-\epsilon_r,\qquad
\psi_i-\psi_r=\rho_i+\eta_i-\eta_r.
\]

If only separate bounds `|epsilon_i|<=delta_i` are supplied, the
sharp independent-error radius for an anchored form difference is
`delta_i+delta_r`; the phase statement is identical in its admitted
lift. There is no basis for subtracting this uncertainty merely because
the common origin was removed.

The shared reference error also creates dependence between relative
coordinates. For an edge `i--j`, subtract before interval evaluation:
its error is `epsilon_i-epsilon_j`, with sharp radius
`delta_i+delta_j`. Subtracting two separately expanded anchor boxes
would produce the valid but unnecessarily broad radius
`delta_i+delta_j+2*delta_r`. An explicit common-error model permits
algebraic cancellation; matching marginal error sizes alone do not
prove such dependence. Circular uncertainty must retain its phase-set
or justified lift interpretation, including any branch ambiguity.

Synchronous means the same actual time, not merely matching nominal
timestamps. With different acquisition jitter, both the nodal state
and a moving sensor reference may be evaluated at different times.
Then `C(t_i)-C(t_r)` need not vanish. Node-speed bounds control the
former error; an independently justified reference-drift bound is needed
for the latter. The prior sample budget bounds the declared coordinate
and clock, not an undeclared moving instrument reference.

Likewise, the existing hidden-state inverse consumes original nodal
rates, not rates relative to a moving node. Supplying
`V_i-V_r,Omega_i-Omega_r` unchanged to that inverse can produce a
consistent but wrong reconstruction. Consider a three-node star with
ports `r,p` of unit capacity and one hidden node `h`:

\[
(x_r,x_p,x_h)=(0,1,2),\qquad
(\theta_r,\theta_p,\theta_h)=(0,\delta,0),\qquad
\sin\delta\ne0.
\]

Its original port rows are

\[
(V_r,\Omega_r)=(2e,-2b),\qquad
(V_p,\Omega_p)=(e-a\sin\delta,-b).
\]

The anchored rows are therefore `(0,0)` at `r` and
`(-e-a*sin(delta),b)` at `p`. Those are exactly the **absolute**
port rows of a different hidden state `x_h=0,theta_h=0` with the
same visible coordinates. The observation geometry has rank two, so
the original inverse can identify this phantom form instead of the
actual value `2`. The issue is the missing reference velocity, not
loss of dynamics in the complete quotient. Restore the reference rows
with their evidence or derive a separately admitted relative inverse;
a relabeled rate dictionary does neither.

Only one global origin of each kind is redundant on this connected
model. Removing independent origins from several coupled regions
would discard their relative offsets, which drive interaction across
the connecting edges. Likewise, omitting a hidden state, changing
capacity or losing a supplied boundary input is not this symmetry.
The previous causal-memory and hidden-state requirements remain in
force under a change of reference.

### Winding is retained information, not an unconditional invariant

The quotient determines every edge phase ratio
`z_j*conjugate(z_i)`. Away from antipodal edges, it therefore determines
the principal phase increment on each oriented edge and the winding of
each supplied oriented cycle. Common rotation changes none of them.
At an antipodal ratio `-1` the two limiting increments `+pi` and
`-pi` meet: a principal-increment winding cannot be continuously
assigned through that boundary.

The sine law itself remains smooth there. Consequently an antipodal
crossing can change winding without a singular nodal state, deleted
edge or new event. A snapshot winding and positive initial margin
alone do not establish its indefinite preservation. Uncertain edge
sets touching that boundary require an unresolved winding or separate
branch evidence. Temporal preservation needs an invariant region,
such as the sufficient local recovery domain below.

### A whole-set sufficient recovery criterion under the sine law

One concrete maintained-pattern identity is the common-origin orbit
of an exact critical phase geometry `theta_*` with uniform form.
Assume now **`e,w,beta>0` and every held `nu_i>0`**. Require
`S(theta_*)=0` exactly and all reference principal edge increments
`delta_*,ij` strictly acute. A small computed residual alone does not
establish this premise.

Let `L` be the unit graph Laplacian, `lambda_2(L)>0` its spectral gap,
and `P=I-11^T/n`. In a declared continuous deviation lift from
`theta_*`, put

\[
Z^2=\|Pu\|^2+\|P(\rho-\rho_*)\|^2,\qquad
\mathcal E=E(u,z)-\beta V_\phi(\theta_*).
\]

This norm uses the declared model coordinates; it is not a universal
physical metric or a diagnostic definition of identity. With
`m=min_edges(pi/2-|delta_*,ij|)`, choose

\[
0<r<\frac m{\sqrt2},\qquad
c_r=\min_{\{i,j\}}\cos(|\delta_{*,ij}|+\sqrt2r)>0,\qquad
\kappa_r=\frac{\lambda_2(L)}2\min(1,\beta c_r).
\]

Let `U` be an admitted state uncertainty set in this relative chart,
with all its internal coordinates retained and the declared fixed
support, coefficients and capacities unchanged. The
sufficient conditions are

\[
\boxed{\sup_{U}Z<r,\qquad
       \sup_{U}\mathcal E<\kappa_r r^2.}
\]

They imply that **every actual state in `U`** stays in this phase
chart and converges to the reference geometry modulo common origins.
The set need not be a singleton; correlated uncertainty may be
conservatively enclosed by a larger set if that entire enclosure
passes the inequalities.

The geometric barrier `kappa_r` does not depend on the positive
capacity values. With support and coefficients still fixed, the same
criterion consequently holds for each held capacity vector in an
independently admitted strictly positive family. This covers capacity
uncertainty without choosing its midpoint. It does not supply a uniform
decay rate as capacities approach zero, and it does not admit a zero
capacity by continuity of a positive-capacity theorem.

**Proof of trapping and convergence.** Inside `Z<=r`, each edge's
phase-deviation difference is bounded by `sqrt(2)*r`. The line
segment from the reference remains acute, and its cosine Hessian
is bounded below by `c_r L` on the common-phase quotient.
Criticality removes the linear phase-storage term. Taylor's formula
and the graph spectral gap give

\[
\mathcal E\ge
\frac{\lambda_2(L)}2\|Pu\|^2+
\frac{\beta c_r\lambda_2(L)}2
 \|P(\rho-\rho_*)\|^2
\ge\kappa_r Z^2.
\]

The exact sine loss `mathcal E'=-e*q^T K q<=0`, with
`K=diag(nu_i/d_i)>0`, prevents a first exit through `Z=r`.
The trajectory remains in a compact interior quotient sublevel.
On its largest invariant zero-loss subset, `q=0`, hence
`Omega=0`. Preserving `q=0` further requires `LKS=0`, so
`KS=c*1`. But `sum_i S_i=0` and `K>0` give
`c*sum_i(1/k_i)=0`; therefore `c=0` and `S=0`.
The positive phase Hessian on this convex local chart makes
`theta_*` its unique critical phase geometry modulo rotation.
LaSalle's principle thus gives convergence to that orbit.

Local exponential recovery follows from the same sine rows, rather
than from a transferred native trajectory. Choose orthonormal columns
`R` spanning `1^perp` and define
`B=R^T L R>0`, `A=R^T K R>0` and
`C=R^T Hess(V_phi)(theta_*) R>0`. The quotient Jacobian is

\[
J=\begin{pmatrix}-eAB&-aAC\\ bAB&0\end{pmatrix}.
\]

In storage coordinates it is similar to
`[[-D,-H],[H^T,0]]` with
`D=e*B^(1/2)*A*B^(1/2)>0` and
`H=(a/sqrt(beta))*B^(1/2)*A*C^(1/2)` invertible.
For an eigenvector `(v,w)`,
`Re(lambda)*(|v|^2+|w|^2)=-v^*D*v`. A zero real part would force
`v=0` and then `H*w=0`, contradicting a nonzero eigenvector.
All quotient eigenvalues have negative real part; smoothness gives
local exponential recovery. The weighted-mean reconstruction then
also determines the finite limiting common origins.

The positive-dissipation and positive-capacity premises matter.
At `e=0` a nonzero local excess storage is conserved and cannot
converge to its zero value at the target. Two inactive nodes with
distinct frozen forms can preclude uniform-form recovery however
small their difference. Positive stiffness or the above acute
criterion is sufficient, not a classification of every stable sine
geometry or its full basin.

This theorem closes a bounded mechanism question: irrelevant common
origins can be removed exactly, while all sufficiently small admitted
**relative** perturbations in the stated domain recover under the
supplied nodal law. It does not prove spontaneous entry into that
domain, select the law, create support or identify a pattern with
matter. The inequalities are a mathematical whole-set criterion;
their application to a concrete uncertain pattern requires outward
admission of every premise, as in the cycle family below.

<a id="sine-cycle-recovery"></a>
### Computable whole-set recovery for an exact cycle twist

Supply an ordered cycle `(v_0,...,v_(n-1))` containing every node
exactly once, `n>=3`, whose consecutive and closing edges equal the
complete support. Extra edges, a selected cycle inside a larger network
and an unspecified orientation do not meet this contract. Let the
target winding `ell` be an integer with `4*|ell|<n` and define

\[
\alpha=\frac{2\pi\ell}{n},\qquad
\theta^*_{v_j}=j\alpha\pmod{2\pi}.
\]

This is an exact symbolic target, including consensus `ell=0`.
Each node has neighboring increments `+alpha,-alpha`, so
`S_i(theta*)=sin(alpha)+sin(-alpha)=0` identically. Its Hessian is
`cos(alpha)*L` and `cos(alpha)>0`. No rounded phase vector or small
numerical residual is used to recognize an equilibrium. The cyclic
Fourier modes of the unit Laplacian have eigenvalues
`2-2*cos(2*pi*j/n)`, giving the exact nonzero gap

\[
\lambda_2(L)=2-2\cos(2\pi/n).
\]

#### Chart choice and cancellation before interval evaluation

The caller supplies one integer turn `m_j` per observed phase lift;
zero turns are an explicit permissible choice. Define the deviation
lift `h_j=theta_(v_j)+2*pi*m_j-j*alpha`. Turns choose a chart for
the same circular data; they are not automatically fitted to force
admission and do not reduce an observation error.

For original synchronous observations with independent residual bounds
`epsilon_i` around their nominal coordinates, every pair satisfies

\[
\begin{aligned}
X_{ij}&=
 (x^{\rm nom}_{v_j}-x^{\rm nom}_{v_i})
+[-\delta_i-\delta_j,\delta_i+\delta_j],\\
H_{ij}&=
 (\theta^{\rm nom}_{v_j}-\theta^{\rm nom}_{v_i})+
 \pi\left[2(m_j-m_i)-\frac{2\ell(j-i)}n\right]
+[-\epsilon_i-\epsilon_j,\epsilon_i+\epsilon_j].
\end{aligned}
\]

Here `delta_i` denotes the form residual radius; `epsilon_i` denotes
the phase residual radius. Common form and phase origins cancel
exactly. The entire rational coefficient of mathematical pi is formed
before outward interval evaluation; separately materializing target
angles would introduce avoidable cancellation error.

The exact identity
`||Pf||^2=(1/n)*sum_(i<j)(f_j-f_i)^2` gives the sufficient squared
norm bound

\[
Z^2\le Z_+^2:=
\frac1n\sum_{i<j}
\left(\sup X_{ij}^2+\sup H_{ij}^2\right).
\]

Pairwise interval bounds can be conservative because the same residual
appears in several pairs, but they never add an artificial reference-node
error. They retain every node and certify the whole declared observation
class rather than its midpoint.

#### Quadratic excess storage without subtracting twist energy

Orient every cycle edge from `v_j` to `v_(j+1 mod n)` and let
`d_j=h_(j+1 mod n)-h_j`. The closing edge has the same circular
target increment `alpha`, even though its target real-lift difference
is `-(n-1)*alpha`. Its full-turn difference must be retained when
computing `d_(n-1)`. The deviations telescope exactly:
`sum_j d_j=0`.

Since every target edge has the same sine, the entire linear term in
phase excess storage cancels:

\[
\begin{aligned}
V_\phi(\theta)-V_\phi(\theta_*)
 &=\sum_j[\cos\alpha-\cos(\alpha+d_j)-\sin\alpha\,d_j],\\
\cos\alpha-\cos(\alpha+d)-\sin\alpha\,d
 &=d^2\int_0^1(1-s)\cos(\alpha+sd)\,ds.
\end{aligned}
\]

This is an exact integral remainder for either sign of `d`. It avoids
subtracting two nonzero twist energies and has quadratic rather than
linear sensitivity to small interval widths.

Let `H_j` enclose the edge deviation `d_j` and set

\[
c_j^+=
\sup\cos\!\left(\alpha+\operatorname{hull}(0,H_j)\right).
\]

In computation, the cosine interval's upper endpoint is an outward
upper bound for this supremum. The segment contains `alpha`, so
`c_j^+>=cos(alpha)>0`, and `c_j^+<=1`. Therefore

\[
\mathcal E\le Q_+:=
\frac12\sum_j\sup X_{j,j+1}^2+
\frac\beta2\sum_j c_j^+\sup H_j^2.
\]

Using `c_j^+=1` remains safe but less informative. No individual
Taylor term is presumed independent of another; their upper bounds
enclose the complete sum. This upper-bound inequality is valid before
basin admission. Nonnegative excess storage follows only after the
appropriate acute neighborhood is established, not from clipping a
possibly negative general excess to zero.

For supplied `r>0`, the cycle constants are

\[
m=\frac\pi2-|\alpha|,\qquad
c_r=\cos(|\alpha|+\sqrt2r),\qquad
\kappa_r=\frac{\lambda_2(L)}2\min(1,\beta c_r).
\]

Outward pi, square-root and cosine bounds provide certified lower
endpoints. Admission requires `sqrt(2)*r<m`, a strictly positive
resolved lower bound `kappa_-<=kappa_r`, and

\[
\boxed{Z_+^2<r^2,\qquad Q_+<\kappa_-r^2.}
\]

Positive dissipation and every strictly positive held capacity remain
separate hypotheses. Failure to resolve these sufficient inequalities
is unavailability of this certificate, not proof that the state cannot
recover or that every point in its uncertainty set leaves the basin.

#### Original observation sets and propagated boxes are different inputs

The preceding pair formulas use the original per-node residuals once.
Their conclusion covers that synchronous observation set. A Cartesian
outer box from full-state propagation generally contains additional
states because residual correlations have been lost. It must be
checked in full, even if the tighter original set passed.

For such a propagated box, form each `X_ij` by subtracting the two
full endpoint form intervals, and each `H_ij` by subtracting the two
full phase intervals followed by the same exact target/turn
coefficient. Apply the identical norm and remainder proof to these
wider pair intervals. Use the original full endpoint boxes, before
another subtraction of the moving reference; do not reuse the source's
smaller edge bounds or its earlier storage bound.

The report's actual validated endpoint time identifies the state being
certified. A partial forecast can therefore admit recovery starting at
that earlier validated time, but cannot silently claim certification
at an unreached requested time. Original observations provide exact
held capacities. A full solver box may instead carry its final node's
held capacity as an augmented interval: its whole interval must be
strictly positive, while the other held capacities satisfy the same
condition. The positive-family extension above then applies without
selecting a capacity midpoint. A box including zero is outside this
recovery theorem.

#### An explicitly deformed C5 observation class that passes

Choose the positively oriented unit C5, target `ell=1`,
`e=w=1/2,beta=1` and unit capacities. Supply no extra phase turns,
`r=1/16` and dyadic nominal observations

\[
x^{\rm nom}=(1/1024,-1/1024,0,0,0),\qquad
\theta^{\rm nom}_{v_j}=5j/4.
\]

Every node has form and phase residual radius `1/4096`; both common
origins remain arbitrary. These observations are not the exact target:
the target increment remains `alpha=2*pi/5`, not `5/4`.
No trajectory or generated response is needed to admit this class.

The elementary bounds `25/8<pi<22/7` give
`0<alpha-5/4<1/140`. Put

\[
A=\frac1{140}+\frac1{2048},\qquad
B=\frac1{35}+\frac1{2048},\qquad
F=\frac5{2048}.
\]

All form-pair magnitudes are at most `F` and all phase-deviation
pair magnitudes at most `B`. Hence
`Z_+^2<=2*(B^2+F^2)=2184757/1284505600<1/256=r^2`.
The four nonclosing phase-edge deviations have magnitude at most `A`;
the closing one has magnitude at most `B`.

Every phase interpolation segment lies between `31/25` and `pi/2`.
The cosine upper Taylor bound at `31/25` gives
`c_j^+<=1-(31/25)^2/2+(31/25)^4/24<1/3` as an analytic upper
bound. Consequently

\[
Q_+\le\frac52F^2+\frac{4A^2+B^2}{6}
 =\frac{59951}{308281344}<\frac1{2560}.
\]

Furthermore `lambda_2(L)>1`, and
`alpha+sqrt(2)*r<377/280<27/20`. The alternating lower bound
`cos(27/20)>=1-(27/20)^2/2+(27/20)^4/24-(27/20)^6/720>1/5`
gives `c_r>1/5` and `kappa_r*r^2>1/2560`.
All strict recovery conditions therefore hold for this complete
uncertain class. An implementation still records its own outward
computed margins; the analytic inequalities are an independent
admission proof, not substituted fixture values.

This result proves conditional recovery of an already supplied
geometric identity despite nonzero relative observation uncertainty.
It does not create the target winding, establish spontaneous entry
into this neighborhood, select the microscopic law, add support or
identify the cycle with a physical constituent.

## 19. Recovering patterns with a live retained intermediary

<a id="sine-interacting-recovery"></a>

Two locally recoverable regions can belong to a single recoverable
**interacting** geometry under the supplied sine law. The statement here
retains the complete fine support and the intermediary's form, phase and
capacity. It neither replaces the intermediary by a direct edge nor
derives when a new edge appears.

### An exact critical target on the full eleven-node support

Take cycles `(0,1,2,3,4)` and `(5,6,7,8,9)` and the two additional
edges `0--10` and `5--10`. Node `h=10` is the intermediary, and
nodes `0,5` are the ports. There are eleven nodes and twelve unit
edges; each port has degree three and the intermediary degree two.
For `alpha=2*pi/5`, specify

\[
\theta^*_{j}=\theta^*_{5+j}=j\alpha\pmod{2\pi}
\quad(0\le j<5),\qquad \theta^*_{10}=0.
\]

At an ordinary ring node the incoming sine terms are
`sin(alpha)+sin(-alpha)=0`. The same cancellation holds at a port,
and its additional intermediary edge contributes `sin(0)=0`.
Both intermediary edges have zero phase difference. Therefore
`S(theta*)=0` exactly on the entire graph, not just in either
isolated cycle. Uniform form completes a stationary state.

The phase Hessian assigns ring edges weight `cos(alpha)>0` and
the intermediary edges weight one. It is positive on the full
common-phase quotient, including perturbations that move the two
rings against each other or change the intermediary. Independent
regional stability would not by itself prove this full-network result.
For strictly positive held capacities and `e,w,beta>0`, Section 18
therefore supplies local exponential recovery of this complete geometry.

More generally, the shared exact phase-cycle reconstruction can admit a
supplied rational-turn target on the full graph. It checks integral
cycle periods, strict acute edges and symbolic cancellation of opposite
sine terms. Only these algebraic facts are reused here. An unproved sine
residual, an approximate equilibrium or that owner's separate dynamical
interpretation cannot substitute for the full sine-law hypotheses.

For an exactly critical target with possibly different edge angles
`alpha_e`, the excess-storage cancellation extends directly:
`sum_e sin(alpha_e)*(h_j-h_i)=grad(V_phi)(theta*)^T h=0`.
The quadratic remainder bound in Section 18 consequently applies
edge by edge with its own target angle. No uniform twist or
equal-capacity premise is needed for that cancellation.

### An independent full-graph spectral-gap bound

For every unordered node pair choose a path `P_ij` with length
`l_ij`. Cauchy--Schwarz and the pairwise centering identity give

\[
\begin{aligned}
\|Pv\|^2
 &=\frac1n\sum_{i<j}(v_j-v_i)^2\\
 &\le\frac1n\sum_e C_e(v_{e^+}-v_{e^-})^2,\qquad
C_e=\sum_{\{i,j\}:e\in P_{ij}}l_{ij}.
\end{aligned}
\]

Thus `lambda_2(L)>=n/max_e C_e`. This is a finite path-counting
proof, not a numerical eigensolver assumption. For the stated support,
the shortest paths give:

| Edges, with the same pattern on the other ring | Path load `C_e` |
| --- | --- |
| `0--10` and `5--10` | `121` |
| `0--1` and `0--4` | `57` |
| `1--2` and `3--4` | `34` |
| `2--3` | `5` |

For example, distances from a ring's nodes to its port are
`(0,1,2,2,1)`. The bridge's paths from that ring to the
intermediary contribute `6+5=11`; paths between the rings contribute
`5*6+25*2+5*6=110`, giving `121`. Hence

\[
\lambda_2(L)\ge\frac{11}{121}=\frac1{11}.
\]

With `beta=1` and `r=1/16`, the earlier elementary cosine bound
still gives `c_r>1/5`, because the largest target edge angle is
`2*pi/5`. The whole-network trapping barrier therefore satisfies
`kappa_r*r^2>1/28160`. This analytic lower bound is independent
of the engine's general rational quotient-gap certificate; a more
conservative computed gap remains valid if its own margins pass.

### A finite uncertain class within the interacting basin

Use default `e=w=1/2,beta=1` and unit capacities. For both rings
supply the dyadic nominal phases
`theta_j^nom=theta_(5+j)^nom=1287*j/1024`; supply intermediary
phase zero. The nominal form is `x_0=1/4096` and zero elsewhere.
Every node has form and phase residual radius `1/65536`, and the
two common origins may be arbitrary. Target turns remain exactly
`j/5` on the rings and zero at the intermediary. No phase turns
are added to the nominal observations.

The elementary bounds `157/50<pi<22/7` imply
`|1287/1024-2*pi/5|<1/1024`. Define

\[
R=\frac1{32768},\qquad A=\frac{33}{32768},\qquad
B=\frac{129}{32768},\qquad F=\frac9{32768}.
\]

Here `R` bounds a difference of two residual errors, `A` bounds
each nonclosing ring-edge phase deviation, `B` bounds each closing
one and every pair's phase deviation, and `F` bounds every form
difference. The two port/intermediary phase deviations are at most
`R`. The pairwise identity gives

\[
Z_+^2\le5(B^2+F^2)
 =\frac{41805}{536870912}<\frac1{256}=r^2.
\]

All ring-edge interpolation segments lie between `31/25` and
`pi/2`, so the independent Taylor bound used above supplies a
cosine upper bound `1/3`. Bridge segments can use the bound one.
Three form edges touch the perturbed donor port; the other nine
have only residual error. Therefore

\[
\begin{aligned}
\mathcal E
&\le\frac{3F^2+9R^2}{2}
    +\frac{4A^2+B^2}{3}+R^2\\
&=\frac{3563}{536870912}
 <\frac1{28160}<\kappa_r r^2.
\end{aligned}
\]

The entire synchronous uncertainty class is consequently within the
full interacting recovery basin. The same argument covers any
strictly positive held capacity vector; it gives no uniform recovery
time as a capacity approaches zero. A computed certificate must retain
its own full-graph gap and error bounds. In particular, the general
quotient-gap owner is not claimed to return the stronger path bound
`1/11`; its separately computed uncertainty and barrier margins
can still establish admission.

### Exact nonlinear donor--intermediary--receiver onset

A separate, exactly specified preparation makes causality transparent.
Use the exact target phases, nominal forms
`x_0=epsilon>0` and `x_i=0` otherwise, ring capacities one and
intermediary capacity `mu>=0`. This is not the preceding dyadic
phase-observation class. All statements in this paragraph concern the
actual nonlinear rows at this finite preparation, not a fitted
small-amplitude derivative.

Write `a=w/pi` and `b=w/(beta*pi)`. At time zero,
`q_h=-epsilon` and `S_h=0`, giving

\[
\dot x_h=\frac{\mu e\epsilon}{2},\qquad
\dot\theta_h=-\frac{\mu b\epsilon}{2}.
\]

Every receiver-ring node initially has zero form and phase rate.
At its port `5`, differentiating the three incident contributions
gives

\[
\dot q_5=-\dot x_h=-\frac{\mu e\epsilon}{2},\qquad
\dot S_5=\dot\theta_h=-\frac{\mu b\epsilon}{2}.
\]

The latter uses the intermediary edge cosine one and zero initial
phase rates at both receiver neighbors. Retaining a general positive
receiver-port capacity `nu_5` for this identity yields

\[
\boxed{
\ddot x_5=\frac{\nu_5\mu\epsilon}{6}(e^2-ab),\qquad
\ddot\theta_5=-\frac{\nu_5\mu e b\epsilon}{6}.}
\]

The donor capacity does not enter this leading two-edge response:
the intermediary already consumes the donor's initial form contrast.
For `mu>0,e>0` the receiver phase thus obeys

\[
\theta_5(t)-\theta_5(0)=
-\frac{\nu_5\mu e b\epsilon}{12}t^2+O(t^3).
\]

It is strictly negative for all sufficiently small positive times.
The exact smooth law supplies this local onset; no numerical horizon,
time-series evaluation or fitted remainder is asserted. At the default
coefficients `e^2-ab=(1-1/pi^2)/4>0`, the initial receiver form
curvature is positive as well.

If `mu=0`, the intermediary instead remains exactly at its initial
form and phase. The unperturbed receiver twist with that fixed
boundary is an exact stationary solution, so uniqueness makes the
receiver stay unchanged for all time. This is a declared capacity
intervention on the same existing support. It separates a causally
active intermediary from a frozen one, not one unknown physical law
from another.

### Influence followed by recovery and a retained common-form record

For the exact-phase preparation, the full quotient norm and initial
excess storage are
`Z^2=10*epsilon^2/11` and `mathcal E=3*epsilon^2/2`.
Taking `epsilon=1/4096` places them strictly below the same
`r=1/16` trapping conditions. Thus for every fixed `mu>0` this
causal response coexists with full-network recovery: both ring
geometries and their intermediary recover to the supplied target.
The perturbation is not required to destroy the pattern to transmit
an effect.

The conserved weighted form mean gives an additional exact endpoint
statement. The two rings have total degree `22` and unit capacity;
the intermediary contributes weight `2/mu`. Only the degree-three
donor port initially carries form `epsilon`. Hence the limiting
uniform form is

\[
x_i(\infty)=\frac{3\epsilon}{22+2/\mu}
\quad\hbox{for every node and each fixed }\mu>0.
\]

For `mu=1` this is `epsilon/8`. The initial exact target phase
has its target weighted lift mean, so the limiting common phase
shift is zero in this local lift. The frozen-intermediary receiver
instead keeps form zero forever. The positive-capacity comparison
therefore supplies both a transient causal signal and a lasting
common-form offset under the permanently retained support.

This offset is a conserved coordinate record; it is not a new
winding, a new constituent or an independently identified physical
memory. It lies in the common-origin coordinate removed by the relative
quotient: distinguishing it requires the retained preparation frame or
another admitted reference, not the final internal pair differences alone.
It tends to zero as `mu` tends to zero. Neither the
recovery theorem nor this limit supplies a uniform finite response
time near that boundary.

### A symmetry obstruction to forming the receiver twist

Recovery of the supplied receiver identity does not establish formation
from a flat receiver. Reuse the exact automorphism mechanism from
the [orientation audit](#mediator-orientation-scope), now acting only
on the receiver:
`R=(6 9)(7 8)` fixes its port `5`, the intermediary and all donor
nodes. If paired receiver capacities are equal, this permutation
preserves the full sine law.

A flat receiver phase and uniform receiver form are fixed by `R`,
regardless of the donor or intermediary state. Equivariance and
uniqueness keep that receiver reflection exact under the endogenous
single-port drive. A winding-one or winding-minus-one receiver twist
is not reflection-fixed; a common rotation cannot repair this because
the port is fixed. Such an initial state therefore cannot converge
to either of those receiver identities. Away from antipodal edges,
its principal-increment receiver winding remains zero by paired
edge cancellation.

This is a consequence of an already justified symmetry, not a new
simulation campaign or a claim that asymmetric formation is impossible.
A meaningful formation test must declare information that breaks
this symmetry and satisfy the relevant storage/transition budget
before evaluating a trajectory. Changing an initial state or held
capacity supplies that information; it does not derive its origin.
No new pressure term or operator selector is installed by the present
recovery and causality results.

<a id="sine-formation-eligibility"></a>

## 20. Formation eligibility and a finite-time exclusion

The recovery and causal-response results above start with both identities
already present. This section instead supplies a receiver with flat phase.
It asks whether an existing intermediary can form the receiver twist under
the same smooth, unforced sine law. A nonzero initial response, enough
initial storage and eventual formation are separate assertions.

Keep the eleven-node support of Section 19: donor ring `(0,1,2,3,4)`,
receiver ring `(5,6,7,8,9)`, intermediary `h=10`, and bridges
`0--h` and `5--h`. The ports have degree three and every other
node has degree two. Write

\[
\alpha=\frac{2\pi}{5},\qquad a=\frac w\pi,\qquad
b=\frac{w}{\beta\pi},\qquad
q_i=\sum_{j\sim i}(x_i-x_j),\qquad
S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i).
\]

The complete rows remain

\[
\dot x_i=\frac{\nu_i}{d_i}(-e q_i+aS_i),\qquad
\dot\theta_i=b\frac{\nu_i}{d_i}q_i,
\quad e,w,\beta>0.
\]

The initial preparation and held capacities are

\[
\begin{aligned}
&\theta_j=j\alpha\pmod{2\pi}\quad(0\le j\le4),\qquad
\theta_5=\cdots=\theta_9=\theta_h=0,\\
&x_h=A,\qquad x_i=0\quad(i\ne h),\\
&\nu_6=1+\delta,\qquad \nu_9=1-\delta,\qquad
\nu_i=1\quad(i\notin\{6,9\}),\qquad |\delta|<1.
\end{aligned}
\]

Here `A` is signed and the capacity contrast is supplied independently.
They are preparation parameters, not a new pressure term or a derived
event-selection law. The desired endpoint is the same full critical
geometry as in Section 19: both rings have winding `+1` and equal
increments `alpha`, the bridge phases agree, and form is uniform.
Convergence is understood modulo the admitted common origins.

### Exact odd response and its controls

At this preparation all sine sums vanish. The only nonzero form gradients
are `q_0=q_5=-A` and `q_h=2A`. Consequently

\[
\dot x_0=\dot x_5=\frac{eA}{3},\qquad
\dot x_h=-eA,\qquad
\dot\theta_5=-\frac{bA}{3}.
\]

For the initially flat receiver, let `chi=theta_6-theta_9` on the
continuous lift through zero. Its value and first derivative vanish.
Since `dot q_6=dot q_9=-eA/3`,

\[
\ddot\theta_6(0)=-\frac{beA(1+\delta)}6,\qquad
\ddot\theta_9(0)=-\frac{beA(1-\delta)}6,
\]

and hence

\[
\boxed{\ddot\chi(0)=-\frac{be\delta A}{3},\qquad
\chi(t)=-\frac{be\delta A}{6}t^2+O(t^3).}
\]

Thus `delta*A!=0` produces an exact local reflection-breaking
response. This Taylor identity neither certifies a finite prediction
horizon nor selects the final winding. In particular, the sign of its
initial curvature is not a proof of the eventual identity.

The two controls are exact. When `delta=0`, receiver reflection is
preserved as proved in Section 19, for every `A`, so the receiver
cannot converge to either nonzero twist. When `A=0`, both `q`
and `S` vanish everywhere: the mixed donor-twist/flat-receiver state
is a full equilibrium for every admitted capacity contrast. Reflection
also exchanges the preparations `delta` and `-delta`, rather
than providing a mechanism that chooses one of them.

### Storage is necessary, but target storage is not the entry barrier

Use the same storage and balance as in Section 18:

\[
E=\frac12\sum_{\{i,j\}}(x_i-x_j)^2
 +\beta\sum_{\{i,j\}}\bigl(1-\cos(\theta_j-\theta_i)\bigr),
\qquad
\dot E=-e\sum_i\frac{\nu_i}{d_i}q_i^2.
\]

Define the unit phase storage of one exact twist by

\[
V_5=5\left(1-\cos\frac{2\pi}{5}\right)
    =\frac{25-5\sqrt5}{4}.
\]

For the supplied family,

\[
E(0)=\beta V_5+A^2,\qquad
E_{\mathrm{target}}=2\beta V_5,\qquad
\dot E(0)=-\frac{8e}{3}A^2.
\]

The capacity contrast does not change this initial loss: the two
contrast nodes have zero initial form gradient. For `A!=0`, the
loss is strictly positive over some initial time interval. Therefore
`A^2>beta*V_5` is necessary for convergence to the desired endpoint.
It is not sufficient, even at the level of a continuous phase path.

Two tempting stronger arguments need qualification. The closure of
winding-one C5 configurations contains an antipodal face represented
by increments `(pi,pi/4,pi/4,pi/4,pi/4)`. Its unit phase
storage is `6-2*sqrt(2)`, which is smaller than `V_5`.
The antipodal winding itself is branch-dependent; nearby increments
`(pi-epsilon,(pi+epsilon)/4,...,(pi+epsilon)/4)` have unambiguous
winding one and still cost less than `V_5` for sufficiently small
positive `epsilon`. Thus winding alone does not impose the twist's
storage lower bound. The critical geometry
`(2*pi/3,pi/3,pi/3,pi/3,pi/3)` has storage `7/2`, but a
critical value alone is not a proof that every relevant path crosses it.
Neither observation supplies the required full-network transition theorem.

Instead define `U` as the set in which **both** rings have strictly
acute principal edge increments and winding `(+1,+1)`. No restriction
on form or bridge phase is imposed in this definition. The initial
receiver is outside the closure of `U`. Any trajectory converging
to the desired target enters `U` at a finite time. At the positive
infimum of those entry times, continuity supplies a boundary state
in which both rings are closed acute with these same windings.
At least one ring is on an acute face.
This argument allows the donor to unwind or leave its acute region
earlier; it makes no assumption about the donor's entire past.

On a closed-acute winding-one C5 the five increments sum to `2*pi`.
A boundary increment cannot be `-pi/2`, since the remaining four
increments are at most `pi/2` each. Thus at least one is `pi/2`;
the remaining four sum to `3*pi/2`. Convexity of `1-cos` on
`[-pi/2,pi/2]` gives the sharp acute-face lower bound

\[
B_5=1+4\left(1-\cos\frac{3\pi}{8}\right)
   =5-4\cos\frac{3\pi}{8},
\]

attained when the remaining increments are all `3*pi/8`. The
other ring has storage at least `V_5` by the same convexity and
its fixed sum. Bridge and form storage are nonnegative. At the joint
entry boundary, therefore,

\[
E\ \ge\ \beta(V_5+B_5).
\]

Here `B_5>V_5` by strict convexity: its five boundary increments
are not equal. Combining this bound with the strict initial loss gives
the stronger necessary condition

\[
\boxed{A^2>\beta B_5.}
\]

For example, at `beta=1`, `A=931/500` gives
`A^2=3.467044`, strictly between
`V_5=3.454915...` and `B_5=3.469266...`.
Its initial storage exceeds the final target storage, but cannot
finance entry to the joint acute target region. This is a path
obstruction, not a failure of a numerical solver.

### An analytic time window can exclude an apparently eligible preparation

The energy bound `E(t)<=E(0)=E_0` yields, by Cauchy--Schwarz,

\[
|q_i(t)|\le\sqrt{2d_iE_0},\qquad
|\dot x_i(t)|\le
M_i:=\nu_i\left(e\sqrt{\frac{2E_0}{d_i}}+a\right).
\]

Since `q_h=2x_h-x_0-x_5`,

\[
|\dot q_h|\le C:=2M_h+M_0+M_5,\qquad
|q_h(t)|\ge (q_{\mathrm{init}}-Ct)_+,\qquad
q_{\mathrm{init}}=2|A|.
\]

The intermediary alone then supplies a lower bound on accumulated loss
over any supplied window `[0,tau]`:

\[
\begin{aligned}
s&=\min\left(\tau,\frac{q_{\mathrm{init}}}{C}\right),\\
L(\tau)&=\frac e2\left(
 q_{\mathrm{init}}^2s-q_{\mathrm{init}}Cs^2+\frac{C^2s^3}{3}
\right)
\ \le\ E(0)-E(\tau).
\end{aligned}
\]

The positive part is essential: squaring the negative continuation of
`q_init-C*t` would invent a loss lower bound after its zero.
Certified upper bounds for `E_0` and `C` can replace their
exact values conservatively.

For every receiver edge, the phase-rate bound is

\[
|\dot\theta_j-\dot\theta_i|
\le b\sqrt{2E_0}
\left(\frac{\nu_i}{\sqrt{d_i}}+\frac{\nu_j}{\sqrt{d_j}}\right).
\]

Let `G` be the maximum right-hand side over the receiver edges.
If `G*tau<pi/2`, every receiver increment stays strictly acute
through that window, on its continuous lift from zero. Its winding
therefore stays zero: joint target entry cannot occur before `tau`.
If also

\[
\boxed{\beta B_5+L(\tau)-A^2>0,}
\]

then `E(tau)<beta*(V_5+B_5)`, so joint target entry cannot
occur later either. This combines a finite phase-speed bound with
irreversible loss; it does not require a simulated trajectory or a
claim about the donor's intervening winding.

### A uniform exclusion for the bounded localized-pulse family

Take the default coefficients `e=w=1/2,beta=1`. The preceding
argument excludes **every** preparation

\[
\boxed{|A|\le2,\qquad |\delta|<1}
\]

from convergence to the specified two-twist target. For `|A|<=1`
the initial storage already fails the joint entry barrier. For
`u=|A|` in `[1,2]`, set `tau=1/5`. Since
`V_5<7/2`,

\[
E_0<\frac{15}{2},\qquad
\sqrt{E_0}<\frac{11}{4},\qquad
\sqrt{\frac{2E_0}{3}}<\frac94,\qquad a=b<\frac16.
\]

The three nodes entering `C` all have unit capacity, so

\[
C=2e\left(\sqrt{E_0}+\sqrt{\frac{2E_0}{3}}\right)+4a
 <\frac{17}{3}<6.
\]

Throughout `[0,1/5]`,
`|q_h(t)|>=2u-6t>=4/5`. In particular, the positive part
has not expired. The accumulated loss is bounded below by

\[
L\ \ge\ \frac14\int_0^{1/5}(2u-6t)^2\,dt
 =\frac{u^2}{5}-\frac{3u}{25}+\frac3{125}.
\]

Compare it with the loss the initial state could afford before losing
access to the joint entry boundary:

\[
L-(u^2-B_5)
\ \ge\ B_5-\frac45u^2-\frac3{25}u+\frac3{125}
\ \ge\ B_5-\frac{427}{125}
\ >\frac3{125}.
\]

The middle expression is decreasing on `[1,2]`. The last strict
inequality uses `B_5>86/25`, an elementary exact bound:
`sqrt(2)>7/5` implies
`cos^2(3*pi/8)<3/20<(39/100)^2`, hence
`cos(3*pi/8)<39/100`.

It remains to exclude entry before this loss has occurred. Adjacent
receiver capacities sum to at most `2+|delta|<3`, and their
degrees are at least two. Thus the receiver-edge speed satisfies
`G<3*b*sqrt(E_0)<11/8`, and every receiver phase increment
has magnitude less than

\[
G\tau<\frac{11}{40}<\frac{\pi}{2}
\]

through `tau=1/5`. Its winding remains zero until the total
storage is already strictly below the target-entry requirement.
This proves the claimed exclusion for the whole bounded family.

The case `A=2,delta=1/2` makes the distinction especially clear:
it has a nonzero odd response and passes both initial storage tests,
but the simple analytic loss bound is `73/125=0.584` by
`tau=1/5`, exceeding the available margin `4-B_5`.
Neither breaking the symmetry nor supplying that much localized form
storage suffices to form the desired receiver pattern.

### What this excludes and what it leaves open

The shared owner `physics/relational_sine_formation.py` evaluates
the exact preparation identities, joint entry barrier and optional
time-window exclusion. Passing a necessary condition means only that
this check has not ruled the preparation out. It does not certify
formation, a basin of attraction or physical emergence. The strict
loss arguments require `e>0`; the uniform exclusion additionally
uses the declared default coefficients and amplitude range.

The result does not exclude other preparations, larger amplitudes or
other targets. It also identifies a concrete source of avoidable loss
without changing the law. If donor form is a common `D`, intermediary
form is `H` and receiver form remains zero, put `s=D-H,r=H`.
The initial form storage and loss are exactly

\[
F=\frac{s^2+r^2}{2},\qquad
-\frac{\dot E(0)}e
 =\frac{s^2+r^2}{3}+\frac{(r-s)^2}{2}
 =\frac{2F}{3}+\frac{(r-s)^2}{2}.
\]

Within this declared two-parameter family, fixed `F` is least
dissipative initially at `s=r`, or `D=2H`. Setting `H=A`
retains `F=A^2` and the same receiver odd acceleration, while
reducing initial loss from `8eF/3` to `2eF/3`. No receiver
phase or winding seed is added. This is a restricted preparation
comparison, not a global optimizer or a formation theorem; its later
admission still requires a separate analysis. The present exclusion
concerns the original localized preparation `D=0,H=A` only.

<a id="sine-balanced-formation"></a>

## 21. A balanced preparation and a directional loss obstruction

Retain the full support, initial phases, positive held capacities and
desired two-twist target of Section 20. Change only the supplied form:

\[
x_0=\cdots=x_4=2A,\qquad x_h=A,\qquad
x_5=\cdots=x_9=0.
\]

This is the fixed-storage, least-initial-loss preparation within the
two-parameter family proved there. It does not place a receiver phase,
winding or form pattern into the initial state. It also does not
minimize loss over all possible full-network preparations.

The initial form gradient and sine sum are exactly

\[
q(0)=A(\mathbf e_0-\mathbf e_5),\qquad S(0)=0.
\]

Thus the intermediary initially has zero form gradient and zero rate,
while the donor and receiver port rates are

\[
\dot x_0=-\frac{eA}{3},\qquad
\dot x_5=\frac{eA}{3},\qquad
\dot\theta_0=\frac{bA}{3},\qquad
\dot\theta_5=-\frac{bA}{3}.
\]

All other initial rates vanish. The initial storage is still
`beta*V_5+A^2`, but its loss is now `2eA^2/3`.
The receiver's initial rates and odd acceleration
`(theta_6-theta_9)''=-be*delta*A/3` are unchanged; other
derivatives are not. In particular, the earlier localized
intermediary-loss estimate cannot transfer because `q_h(0)=0`.
A separate full-network argument is needed.

### A bound that retains the initial direction

Let `L` be the full unit-support Laplacian and put

\[
K=\operatorname{diag}\left(\frac{\nu_i}{d_i}\right),\qquad
B=K^{1/2}LK^{1/2},\qquad
H(\theta)=\nabla_\theta^2
 \sum_{\{i,j\}}\bigl(1-\cos(\theta_j-\theta_i)\bigr).
\]

Positive capacity makes `K` positive definite. For every real
vector `v`, the edge representation gives

\[
|v^\mathsf T K^{1/2}H(\theta)K^{1/2}v|
 \le v^\mathsf T Bv,\qquad
0\preceq B\preceq 2\max_i\nu_i\,I.
\]

The first bound uses only `|cos|<=1`, and holds outside acute
phase regions as well. The second follows by bounding each squared
edge difference by twice the sum of its squared endpoints.
Choose any positive spectral upper bound `lambda` satisfying
`||B||<=lambda`; it then also bounds
`||K^(1/2) H(theta) K^(1/2)||` for every phase state.

Define the exact variables

\[
y=K^{1/2}q,\qquad z=K^{1/2}S,\qquad
C(\theta)=K^{1/2}H(\theta)K^{1/2}.
\]

Differentiating the full nonlinear rows yields

\[
\dot y=-eBy+aBz,\qquad
\dot z=-bC(\theta)y,\qquad
-\dot E=e\|y\|^2.
\]

Here `S=-grad V`, so its derivative carries the displayed
minus sign. No phase linearization or diffusion-only trajectory
has replaced the supplied law.

Suppose `S(0)=0` and `q(0)!=0`. Write
`Y_0=||y(0)||` and `u_0=y(0)/Y_0`, and retain the exact
initial spectral moments

\[
R=u_0^\mathsf TBu_0,\qquad
\gamma=\|Bu_0\|,\qquad
\omega=\sqrt{ab}\,\lambda.
\]

For `Y=||y||` and `Z=||z||`, norm upper derivatives satisfy

\[
D^+Y\le a\lambda Z,\qquad D^+Z\le b\lambda Y.
\]

The omitted contribution in the first inequality is nonpositive
because `B` is positive semidefinite. Comparison with this
cooperative scalar system, starting from `(Y_0,0)`, gives

\[
Y(t)\le Y_0\cosh(\omega t),\qquad
Z(t)\le Y_0\sqrt{\frac ba}\sinh(\omega t).
\]

For a lower bound, use variation of constants only as an exact
identity for the first full row:

\[
\langle u_0,y(t)\rangle
 =Y_0\langle u_0,e^{-eBt}u_0\rangle
 +a\int_0^t
 \langle Be^{-eB(t-s)}u_0,z(s)\rangle\,ds.
\]

The spectral weights of the first inner product are nonnegative
and sum to one. Convexity of the scalar exponential therefore gives
`<u_0,exp(-eBt)u_0> >= exp(-eRt)`.
Moreover, `||B exp(-eBs)u_0||<=gamma` for `s>=0`.
Combining these facts with the bound on `Z` proves

\[
\boxed{
\frac{Y(t)}{Y_0}\ge
e^{-eRt}-\frac{\gamma}{\lambda}
 \bigl(\cosh(\omega t)-1\bigr).}
\]

The right-hand side need not stay positive indefinitely; a negative
value cannot be squared to infer a loss lower bound. The exact
initial direction enters through `R` and `gamma` rather
than treating all of `y(0)` as the fastest Laplacian mode.

### A rational finite-window certificate

There is no need to evaluate a matrix exponential or a hyperbolic
function to obtain a useful certificate. As one sufficient case,
if a supplied horizon
`tau` satisfies `omega*tau<=2/5`, then

\[
\cosh(\omega t)-1
 \le \frac{11}{20}\omega^2t^2,\qquad 0\le t\le\tau.
\]

Indeed, the nonnegative Taylor series and `(2k)!>=2^k` give
`cosh(2/5)<=25/23<11/10`; integrating the bound on the
second derivative twice gives the displayed inequality. More
generally, any supplied rational bound
`v>=omega^2*tau^2` with `0<=v<2` gives

\[
\cosh(\omega t)
\le\sum_{j=0}^{\infty}\left(\frac v2\right)^j
=\frac1{1-v/2}=:M_\tau,\qquad 0\le t\le\tau.
\]

This is the same factorial estimate, without the earlier
`2/5` restriction. The earlier `M_tau=11/10` remains a
valid short-window choice when `v<=4/25`. Neither choice
selects a horizon or changes the law. At `v>=2` the
geometric-series estimate is unavailable, not a proof of
dynamical failure.

Using either admitted bound `M_tau` and a certified
`gamma_bar>=gamma`, set

\[
d=eR,\qquad c=\frac{M_\tau}{2}\bar\gamma\,\lambda ab,\qquad
g(t)=1-dt-ct^2.
\]

Since `exp(-eRt)>=1-eRt`, if `g(tau)>0` then
`Y(t)>=Y_0*g(t)>0` on the whole window. Accumulated loss
consequently obeys the computable bound

\[
\begin{aligned}
E(0)-E(\tau)
&\ge eY_0^2\int_0^\tau g(t)^2\,dt\\
&=eY_0^2\left[
\tau-d\tau^2+\frac{d^2-2c}{3}\tau^3
 +\frac{dc}{2}\tau^4+\frac{c^2}{5}\tau^5
\right].
\end{aligned}
\]

It applies to the actual nonlinear solution over the supplied
window. Its hypotheses are admitted independently of a measured
response. If a separate nodal argument supplies another lower
bound on the same accumulated loss, their maximum is valid;
adding them would generally count the same loss twice.
Specifically, the form-speed bounds of Section 20 give
`|q_i'|<=d_i*M_i+sum_(j~i) M_j` at every node.
Integrating each positive affine lower bound on `|q_i|`,
with its actual initial gradient and weight `e*nu_i/d_i`,
and summing those disjoint nodal losses gives one such alternative.
This also explains why a zero initial intermediary gradient does
not make the balanced preparation's full loss zero.

### Exact moments for the balanced eleven-node family

For the present preparation,

\[
Y_0^2=\frac{2A^2}{3},\qquad R=1,\qquad
\gamma^2=\frac43.
\]

To verify these values, `Kq(0)` has only the two port entries
`+A/3,-A/3`. The vector `LKq(0)` has entries `+A,-A`
at the ports, `-A/3` at donor nodes `1,4`,
`+A/3` at receiver nodes `6,9`, and zero elsewhere.
In particular, its intermediary entry cancels exactly. It follows
that `y(0)^T B y(0)=2A^2/3` and
`||B y(0)||^2=8A^2/9`. The two receiver capacities enter
the latter sum only through `nu_6+nu_9=2`, so both moments
hold for every `|delta|<1`. They also imply
`D'(0)=-2eD(0)` for `D=-E'`, since `z(0)=0`.

At the default coefficients `e=w=1/2,beta=1`, use
`lambda=4` and `gamma_bar=7/6`:
`max nu_i<2`, `sqrt(4/3)<7/6`, and
`a=b=1/(2*pi)<1/6`. For `tau=3/5` the preceding
horizon condition holds. A convenient rational majorant for the
quadratic coefficient is

\[
c_0=\frac{77}{1080}
 \ >\ \frac{11}{20}\bar\gamma\,\lambda ab.
\]

Thus on `[0,3/5]`,

\[
\frac{Y(t)}{Y_0}\ge g_0(t)
 :=1-\frac t2-\frac{77}{1080}t^2,\qquad
g_0(3/5)=\frac{2023}{3000}>0.
\]

Its exact integral and resulting loss bound at `A=2` are

\[
\int_0^{3/5}g_0(t)^2\,dt
 =\frac{32259179}{75000000},\qquad
E(0)-E(3/5)\ge
\frac{32259179}{56250000}>0.57349.
\]

These are rational inequalities derived from the law, not values
sampled from a computed trajectory.

### The same bounded amplitude family is still excluded

The result is uniform over `|A|<=2,|delta|<1`. Handle
`A=0` by the exact equilibrium control. Otherwise,
`c_0<1/8` implies

\[
g_0(t)^2
 =1-t+\left(\frac14-2c_0\right)t^2
   +c_0t^3+c_0^2t^4>1-t
\quad(0<t\le3/5).
\]

Therefore

\[
E(0)-E(3/5)>
\frac{A^2}{3}\int_0^{3/5}(1-t)\,dt
=\frac{7A^2}{50}.
\]

The receiver starts flat, and the same global energy estimate as
in Section 20 gives `G<11/8` for every `|A|<=2`.
Through `tau=3/5` its edge increments consequently have
magnitude less than `33/40<pi/2`: it remains acute with
winding zero. At that time the storage deficit relative to the
joint target-entry boundary is strictly larger than

\[
B_5+\frac{7A^2}{50}-A^2
 =B_5-\frac{43A^2}{50}
 \ge B_5-\frac{86}{25}>0.
\]

It cannot have entered the joint target region before this window
and lacks enough storage to enter later. Thus the balanced
preparation is excluded throughout the same bounded family,
despite its fourfold reduction in initial loss. For `A=2,delta=1/2`
the exact odd response still occurs; it does not lead to the
specified two-twist endpoint.

This proves a new preparation-specific obstruction, not a universal
absence of pattern formation. The following corollary extends it
to the full constant-donor form family; neither result excludes
larger storage, arbitrary form profiles, other targets or other
coefficients.
Its reusable contribution is a nonlinear loss certificate that
retains the initial spectral direction, coupled to the independently
proved phase-speed and joint-entry conditions. The shared formation
owner keeps this evidence distinct from the localized-pulse bound.
The remaining question is whether a declared preparation can change
the required phase sector before losing access to its target, rather
than whether it improves an instantaneous loss statistic.

### Corollary: the whole constant-donor preparation family

At the default coefficients `e=w=1/2,beta=1`, retain the same
initial phases, capacities `|delta|<1` and two-twist target.
Allow arbitrary signed constants `D,H` with donor form `D`,
intermediary form `H` and receiver form zero. Define

\[
u=\frac D2,\qquad v=H-\frac D2,\qquad
F=\frac{(D-H)^2+H^2}{2}=u^2+v^2.
\]

Every such preparation with `F<=4` is excluded from convergence
to the target. When `F=0`, both constants vanish and the
prepared state is the exact equilibrium already identified.
For `F>0`, put `rho=v^2/F`, so `0<=rho<=1`.
The initial gradient has only three possibly nonzero entries,

\[
q_0=u-v,\qquad q_5=-u-v,\qquad q_h=2v,
\]

and `S(0)=0` still holds. The same full-support calculation
as above gives the exact moments

\[
\begin{aligned}
N=\|y(0)\|^2&=\frac{2u^2+8v^2}{3},\\
m_1=y(0)^\mathsf TBy(0)&=\frac{2u^2}{3}+4v^2,\\
m_2=\|By(0)\|^2&=\frac{8u^2+58v^2}{9}.
\end{aligned}
\]

For example, `LKq` has port entries `u-2v,-u-2v`,
intermediary entry `8v/3`, donor-neighbor entries
`(-u+v)/3` and receiver-neighbor entries `(u+v)/3`.
The receiver capacity sum `nu_6+nu_9=2` cancels all
contrast dependence in the moments. Thus

\[
\frac NF=\frac{2+6\rho}{3},\qquad
R=\frac{m_1}{N}=\frac{1+5\rho}{1+3\rho}\in[1,3/2],
\qquad
\gamma^2=\frac{m_2}{N}
 =\frac{8+50\rho}{6+18\rho}\le\frac{29}{12}
 <\left(\frac85\right)^2.
\]

Use `lambda=4,gamma_bar=8/5,tau=3/5` in the same
directional certificate. Its quadratic coefficient is bounded
above by `c_1=22/225<1/8`. The polynomial
`g(t)=1-R*t/2-c_1*t^2` satisfies

\[
g(3/5)\ge\frac{1287}{2500}>0,\qquad
g(t)^2>1-Rt\quad(0<t\le3/5).
\]

The second inequality follows by expansion:
its quadratic coefficient is
`R^2/4-2c_1>=49/900>0` and its cubic and quartic
coefficients are positive. The accumulated loss therefore obeys

\[
\begin{aligned}
E(0)-E(3/5)
&>\frac N2\left(\frac35-\frac{9R}{50}\right)\\
&=\frac{F(7+15\rho)}{50}
\ \ge\ \frac{7F}{50}.
\end{aligned}
\]

This holds for both signs of `u,v` and does not require
a nonzero odd initial receiver acceleration. Meanwhile,
`E(0)=V_5+F<15/2` for `F<=4`, so the same receiver
gap bound keeps its winding zero through the entire window:
`G*tau<33/40<pi/2`. The remaining storage is then below
the joint target-entry boundary, because its deficit is strictly
greater than

\[
B_5-\frac{43F}{50}\ge B_5-\frac{86}{25}>0.
\]

Consequently no other choice of `D,H` inside this fixed
storage budget can evade the obstruction. This removes the need
to guess further profiles within the same two-parameter family.
The engine report retains its `localized` and `balanced`
profiles. The explicit donor-form interface described below also
admits this whole constant-donor family as a subcase; neither
interface evaluates its trajectories.

<a id="sine-internal-form-geometry"></a>

## 22. Donor shape, phase action and a uniform formation obstruction

Keep the complete eleven-node support, donor twist, flat receiver
and intermediary phase, and default coefficients of Sections 20-21.
For this gate fix `delta=1/2`. Supply arbitrary signed donor
form `D=(D_0,...,D_4)` and intermediary form `H`, while
receiver form remains zero. These six independent initial values
replace a constant-donor restriction; no new evolution law,
edge or input is introduced.

### The exact six-coordinate quadratic forms

Let `z=(D_0,D_1,D_2,D_3,D_4,H)^T` and let `P`
place these coordinates at full-network nodes `(0,1,2,3,4,10)`,
placing zeros at the receiver. With the same full `L,K` as
in Section 21, the exact storage and spectral moment matrices are

\[
\begin{aligned}
F&=\tfrac12 z^\mathsf TP^\mathsf TLPz,\\
N&=z^\mathsf TP^\mathsf TLKLPz,\\
m_1&=z^\mathsf TP^\mathsf TLKLKLPz,\\
m_2&=z^\mathsf TP^\mathsf TLKLKLKLPz.
\end{aligned}
\]

These expressions retain all receiver and intermediary rows;
they do not substitute a six-node graph. Equivalently,

\[
F=\frac12\left[
\sum_{j=0}^4(D_j-D_{j+1\bmod5})^2+(D_0-H)^2+H^2
\right],
\]

and the only possibly nonzero initial gradient entries are

\[
\begin{aligned}
q_0&=3D_0-D_1-D_4-H,\\
q_j&=2D_j-D_{j-1}-D_{j+1}\quad(1\le j\le4),\\
q_5&=-H,\qquad q_h=2H-D_0,
\end{aligned}
\]

where the donor subscripts in the second line are taken modulo
five. Thus

\[
N=\frac{q_0^2}{3}
 +\frac12\sum_{j=1}^4q_j^2+\frac{H^2}{3}
 +\frac{(2H-D_0)^2}{2}.
\]

The remaining forms can also be evaluated without a matrix square
root: set `k_i=nu_i/d_i` and `v_i=k_iq_i`. Then

\[
m_1=\sum_{\{i,j\}}(v_i-v_j)^2,\qquad
m_2=\sum_i k_i
 \left[\sum_{j\sim i}(v_i-v_j)\right]^2.
\]

All four forms are rational quadratic forms on the declared six
coordinates. In fact their coefficients are independent of
`delta` throughout `|delta|<1`: the two contrast capacities
first enter `m_2` through receiver neighbors with equal
`(Lv)_6=(Lv)_9=H/3`, and their sum is fixed.
The interval loss and receiver-speed bounds still consume the
actual individual capacities.

### Which donor information reaches the receiver first

The exact initial sine currents vanish for every supplied form.
Let `chi=theta_6-theta_9` and `eta=e^2-ab`. Direct
differentiation of the full rows gives

\[
\begin{aligned}
\dot x_5(0)&=\frac{eH}{3},&
\dot\theta_5(0)&=-\frac{bH}{3},\\
\ddot\theta_5(0)&=\frac{be}{6}(4H-D_0),&
\ddot\chi(0)&=-\frac{be\delta H}{3},\\
\theta_5^{(3)}(0)&=
\frac{b\eta}{18}(9D_0-D_1-D_4-22H),&
\chi^{(3)}(0)&=\frac{b\eta\delta}{6}(8H-D_0).
\end{aligned}
\]

Here `chi(0)=chi'(0)=0`. These are derivatives of the
full nonlinear law at the supplied preparation, not derivatives
of an independently fitted or linearized response. The receiver
first reads the intermediary form; the donor port and then its
neighbor sum appear at higher derivative orders. The two remaining
donor coordinates do not appear in these displayed receiver
derivatives. Their absence at these orders does not prove absence
of later influence.

There is, however, an exact silent two-dimensional subspace:

\[
D_0=H=0,\qquad
D_1=-D_4,\qquad D_2=-D_3.
\]

Let `Q=(1\,4)(2\,3)` reflect only the donor, fixing its
port, the intermediary and every receiver node. The full law is
equivariant under the composition

\[
(x,\theta)\longmapsto(-Qx,-Q\theta)
\]

on the circle-valued phase state. The donor capacities are paired
equally, while this permutation does not exchange receiver
capacities. The prepared donor twist and the above form subspace
are fixed by this transformation. Uniqueness therefore preserves
the combined symmetry for all time.

At each fixed receiver or intermediary node it forces form zero
and a phase in `{0,pi}`. Continuity from the prepared phase
zero fixes that phase to zero forever. Thus the receiver and
intermediary remain exactly unchanged, even at nonzero capacity
contrast and arbitrarily large form storage in this subspace.
This is cancellation under an existing interaction law, not absence
of the supplied edges. Receiver formation is impossible for these
preparations without a separately supplied symmetry-breaking change.

The six-coordinate space also splits into donor-reflection-even
and donor-reflection-odd parts. The four quadratic forms above have
no cross terms between those parts. The odd two-dimensional part
is exactly this silent form subspace. For `D_1=s,D_2=t`
its storage and initial dissipative norm are
`F=2s^2-2st+3t^2` and `N=5s^2-10st+10t^2`.
Nonzero internal loss alone therefore does not establish a
receiver response.

<a id="sine-source-receiver-excitation"></a>
### Source symmetry, nonlinear work and a uniform tangent obstruction

Retain the same six-coordinate source, `e=w=1/2`, `beta=1`,
`delta=1/2`, all eleven fine nodes and no input or event. The
following decomposition is exact; it does not declare the receiver's
future input independently of the donor and intermediary.

Write the initial source in coordinates

\[
\begin{aligned}
C&=D_0-H,&
s_1&=(D_1+D_4)/2-D_0,&s_2&=(D_2+D_3)/2-D_0,\\
o_1&=(D_1-D_4)/2,&o_2&=(D_2-D_3)/2.
\end{aligned}
\]

`H,C,s_1,s_2` specify the donor-reflection-even form and its
live port contrasts. The two `o` coordinates specify the odd form.
With `Q=(1 4)(2 3)` extended by the identity outside the donor,
these are `x_e=(x+Qx)/2` and `x_o=(x-Qx)/2`. Direct edge
accounting gives

\[
\boxed{\begin{aligned}
F&=F_e+F_o,\\
F_e&=\frac12(H^2+C^2)+s_1^2+(s_2-s_1)^2,\\
F_o&=o_1^2+(o_2-o_1)^2+2o_2^2.
\end{aligned}}
\]

The same orthogonal parity split holds for `N,m_1,m_2`, and
indeed every initial moment `y_0^T B^j y_0`, because `Q`
commutes with `L,K` and `B=K^(1/2)LK^(1/2)`. This does not
make the nonlinear field a direct sum of two systems. The exact
silence theorem applies to the odd source **alone**. Removing its
coordinates from a mixed preparation requires an additional closure
argument, which the next counterexample disproves.

#### Equal energy and spectral moments do not determine receiver work

Compare the two exact rational preparations

\[
H=1,\qquad D^+=(0,1,0,0,-1),\qquad D^-=-D^+.
\]

They have identical even coordinates, `F_e=1`, `F_o=2`,
`F=3`, `N=23/3`, and every initial quadratic spectral moment.
They differ only in the sign of the silent odd source. Keep the
initial donor phase orientation `+1` fixed in both preparations;
reversing form is not a reversal of that prepared phase geometry.
For a quantity `f`, write `Delta f=f^+-f^-`, and put
`alpha=2*pi/5`, `a=w/pi`, `b=w/(beta*pi)` as above.

The first mixed term is at the donor port. If `v=KLx`, its
even and odd components satisfy `v_0=-1/3`, `v_{1,e}=0`
and `v_{1,o}=1` for the positive preparation. Differentiating
the actual sine current twice gives

\[
\boxed{n_0:=\Delta x_0^{(3)}(0)
 =-\frac83ab^2\sin\alpha\,
             (v_{1,e}-v_0)v_{1,o}
 =-\frac89ab^2\sin\alpha.}
\]

For clarity, the quadratic contribution is
`-sin(alpha)*[(dot theta_1-dot theta_0)^2
-(dot theta_4-dot theta_0)^2]` in `S_0''`; the bridge has
zero initial sine. Linear contributions at the fixed port agree
by reflection. Thus the displayed difference is a derivative of
the full nonlinear law, not a fitted higher-order response.

Transmission through the retained intermediary yields

\[
\Delta x_{10}^{(4)}(0)=\frac e2n_0,\qquad
\Delta x_5^{(5)}(0)=\frac{e^2-ab}{6}n_0,\qquad
\Delta\theta_5^{(5)}(0)=-\frac{eb}{6}n_0.
\]

The receiver form and phase derivatives of orders zero through
four agree. Reuse the receiver's exact
[signed work and full-nodal loss ledger](#sine-receiver-port-passage),
`J_R=E_R+D_R`, rather than substituting a port-motion score.
At the preparation `q_5=-1`, and
`Delta q_5^{(4)}=-e*n_0/2`; hence

\[
\boxed{\begin{aligned}
\Delta J_R^{(5)}(0)&=\Delta D_R^{(5)}(0)=\frac{e^2}{3}n_0,\\
\Delta E_R^{(6)}(0)&=\frac{2e^3}{3}n_0.
\end{aligned}}
\]

In the stated half-weight model both displayed nonzero work and
storage coefficients equal `-sin(alpha)/(108*pi^3)`. Their
orders differ: the leading change in accumulated incoming work
is matched by receiver loss; retained storage first differs one
derivative later. This is neither a positive formation verdict
nor an assertion that more incoming work must be useful.

Analyticity makes these strict local differences an exact
counterexample to a receiver predictor retaining only the four
even coordinates plus `F,N` and initial spectral moments. Pure
odd silence cannot justify such a nonlinear source reduction.
The example supplies a closure obstruction, not a newly evaluated
trajectory or a preparation chosen to force barrier passage.

#### Every communicating tangent response stays below the receiver barrier

Let `theta_*` be the original donor-twist/flat-receiver phase,
and let `H_*` be its full phase Hessian: donor cycle edges have
weight `cos(alpha)>0`; receiver and bridge edges have weight one.
The complete constant tangent system for a real phase deviation
`eta`, with initial `eta=0`, is

\[
\dot x=-eKLx-aKH_*\eta,\qquad
\dot\eta=aKLx.
\]

There is no simultaneous-mode assumption. The matrices
`B=K^(1/2)LK^(1/2)` and `C_*=K^(1/2)H_*K^(1/2)`
need not commute. Their already proved support bounds suffice:

\[
\lambda_+(B)\ge1/44,\qquad 0\le C_*\le B,\qquad
\lambda_{\max}(C_*)\le3.
\]

First consider any initial form, and put
`F_t=x^T Lx/2`, `P_t=eta^T H_*eta/2`,
`D_t=e*(Lx)^T K(Lx)`. Direct differentiation gives

\[
\dot F_t=-D_t-\dot P_t,\qquad
\dot P_t=a(Lx)^\mathsf TK H_*\eta,\qquad
D_t\ge\gamma F_t,\quad \gamma=2e\lambda_+(B),
\]
\[
|\dot P_t|\le A\sqrt{D_tP_t},\qquad
A^2=\frac{2a^2\lambda_{\max}(C_*)}{e},\qquad
\frac A{\sqrt\gamma}
\le\frac{\sqrt{132}}\pi<4.
\]

The storage-angle argument now supplies an all-time bound. Where
`F_t,P_t>0`, set `phi=atan(sqrt(P_t/F_t))`, `k=D_t/F_t`
and `h=dot P_t/(2*sqrt(F_t*P_t))`. Then

\[
\dot\phi=h+\frac k2\sin\phi\cos\phi,\qquad
\frac{2h}{k}<4.
\]

For `G(phi)=(2/9)*(phi+sin(phi)*cos(phi))`,
`G'=4*cos(phi)^2/9`. It follows that
`(F_t+P_t)*exp(G(phi))` is nonincreasing, because

\[
\frac{d}{dt}\log[(F_t+P_t)e^{G(\phi)}]
\le k\cos^2\phi\left[-1+\frac49
                  \left(2+\frac14\right)\right]=0.
\]

The function `sin(phi)^2*exp(-G(phi))` increases to
`exp(-pi/9)` on `[0,pi/2]`. At zero-norm strata the same
continuous, locally absolutely continuous extension as in the
[full-state storage-angle proof](RELATIONAL_EXCHANGE_ADMISSION.md#relational-full-consensus-formation-obstruction)
applies; no zero form or zero phase state is excluded by division.
Consequently `P_t<=F_t(0)*exp(-pi/9)` for every future time.

For the present receiver observation this improves to a bound using
only `F_e`. The donor reflection commutes with `L,H_*` and `K`,
so the odd tangent solution has zero receiver coordinates for all
time. The full tangent receiver response equals the even-only one.
If `eta_R^tan` is its receiver phase deviation and `L_R` the
internal five-edge receiver Laplacian, then

\[
\boxed{\frac12\|L_R^{1/2}\eta_R^{\rm tan}(t)\|^2
\le F_e e^{-\pi/9}\le\frac34F_e\le3,
\qquad t\ge0.}
\]

The middle inequality is strict when `F_e>0`, since
`exp(pi/9)>1+1/3=4/3`. At `F_e=0` the receiver tangent
response is exactly zero. This is a uniform bound over the
original source budget, not a search for a favorable spectral
direction or a certificate for the nonlinear receiver.

#### A full nonlinear storage bound forces the donor barrier first

The same angle method also gives a distinct result for the actual
nonlinear field, provided its nonzero initial donor phase storage is
retained. This step does not replace that storage by a tangent cost.
For the full unit phase potential `V`, nodal Cauchy--Schwarz and
`sin(u)^2<=2*(1-cos(u))` give globally

\[
\begin{aligned}
S^\mathsf TKS
&\le\sum_i\nu_i\sum_{j\sim i}\sin^2(\theta_j-\theta_i)\\
&\le4\max_i\nu_i\,V=6V.
\end{aligned}
\]

Therefore `dot E=-D`, `D>=gamma*F(t)` and
`|dot V|<=sqrt(6*a^2/e)*sqrt(D*V)` obey precisely the
preceding angle inequalities, now with `P_t` replaced by `V(t)`.
No acute phase condition, simultaneous modal basis or linearization
is used. Initially `V(0)=V_5`; for initial form storage `F>0`
the resulting all-time bound is

\[
\boxed{V(t)\le\mathcal B(F)
:=(F+V_5)\exp\!\left[
G\!\left(\arctan\sqrt{V_5/F}\right)-\frac\pi9\right].}
\]

At `F=0` the source is the exact stationary donor-only state,
consistent with the continuous value `mathcal B(0)=V_5`.
This bound increases with the supplied form budget, since

\[
\frac{d}{dF}\log\mathcal B(F)
=\frac1{F+V_5}-\frac{2\sqrt{FV_5}}{9(F+V_5)^2}>0.
\]

A rational majorant at `F=4` suffices for the entire class. The
exact inequalities `559/250<sqrt(5)<56/25` imply
`69/20<V_5<691/200`. Thus

\[
\phi_4=\arctan\sqrt{V_5/4}
<\arctan\frac{93}{100}
=\frac\pi4-\arctan\frac7{193}
<\frac{11}{14}-\frac7{193}+\frac13\left(\frac7{193}\right)^3
<\frac34.
\]

Here `pi<22/7` and the alternating lower bound
`atan(t)>t-t^3/3` are sufficient. Hence `G(phi_4)<5/18`.
Using `pi>157/50` and the positive exponential series gives

\[
\frac\pi9-G(\phi_4)>\frac{16}{225},\qquad
e^{16/225}>1+\frac{16}{225}+\frac12\left(\frac{16}{225}\right)^2
=\frac{54353}{50625}.
\]

Consequently the following bound is rigorous without a trajectory
or a numerical optimization:

\[
\boxed{V(t)\le\mathcal B(F)\le\mathcal B(4)
<\frac{3019275}{434824}<\frac{139}{20}
<V_5+\frac72,\qquad F\le4,\ t\ge0.}
\]

Suppose the receiver reaches its first potential barrier at `tau_R`.
All other edge potentials are nonnegative, so at that instant

\[
\boxed{V_D(\tau_R)<\frac{139}{20}-\frac72=\frac{69}{20}<V_5.}
\]

Before crossing its own `7/2` barrier, the donor remains in its
initial twist component, where `V_D>=V_5`. It must therefore
have crossed that barrier **strictly before** `tau_R`, for the
entire original budget `F<=4`. This extends the earlier
energy-only order restriction beyond `F<=7/2`. It also excludes
simultaneous donor and receiver barrier states, which would require
`V>=7`. The required donor phase-storage release at receiver passage
is strictly greater than

\[
V_5-\frac{69}{20}=\frac{56-25\sqrt5}{20}>0.
\]

This is a structural restriction on any successful receiver-directed
mechanism: it must use a path that first escapes the donor well and
releases donor phase storage. It neither predicts a passage time nor
excludes sequential receiver acquisition. Potential-well departure
is not an assertion about the donor's instantaneous wrapped winding
or its eventual equilibrium.

#### A receiver passage needs a nonzero, quantified nonlinear contribution

Let `eta_R^full` be the receiver's continuous phase lift from its
initial flat phase under the actual law. At any actual first
potential barrier `V_R=7/2`, the global inequality `1-cos u<=u^2/2`
and the triangle inequality imply

\[
\boxed{\|L_R^{1/2}(\eta_R^{\rm full}-\eta_R^{\rm tan})\|
\ge\sqrt7-\sqrt{2F_e e^{-\pi/9}}
>\sqrt7-\sqrt6.}
\]

This uses the same initial state and clock for the full and tangent
solutions. Removing a common phase origin does not change the norm.
It does not equate primitive phase with an inferred regional angle.
For `F_e=0`, exact nonlinear silence already excludes the passage;
the conditional inequality remains consistent with that control.

The required correction has an exact source. Define, on the full
graph and the retained real lift,

\[
R(\eta)=S(\theta_*+\eta)+H_*\eta,
\qquad
\mathcal A_*=
\begin{pmatrix}-eKL&-aKH_*\\aKL&0\end{pmatrix}.
\]

With `z=(x,eta)` and matching initial conditions, variation of
constants gives

\[
\boxed{z^{\rm full}(t)-z^{\rm tan}(t)
=a\int_0^t e^{\mathcal A_*(t-s)}
       \binom{K R(\eta^{\rm full}(s))}{0}\,ds.}
\]

Every edge term of `R` is the actual sine remainder
`sin(alpha_ij+u)-sin(alpha_ij)-cos(alpha_ij)*u`, whose magnitude
is at most `u^2/2`. It contains the mixed source mechanism above
and possible release of donor phase storage. It is not an added
input or a freely selected receiver waveform.

Thus no improved tangent source direction within `F<=4` can by
itself explain receiver barrier passage. A positive route must
justify this finite nonlinear conversion; a negative route must
bound the actual convolution or receiver work below its required
gap. The current bound does neither for all mixed preparations.
No new reserved source, basin verdict, loss law or numerical
response follows from these source and closure results alone.

The existing [formation owner](../../src/tnfr/physics/relational_sine_formation.py)
exposes `source.receiver_excitation()` as `SineReceiverExcitation`.
It reconstructs both full source components and their actual form
storage and dissipative norms before admitting the separate fixed-law
tangent result. Its rational ceiling is `3*F_e/4`, with strictness
only for positive `F_e`. The
`necessary_nonlinear_correction_norm_bounds` field encloses the
analytic threshold `sqrt(7)-sqrt(3*F_e/2)`, not the realized
nonlinear correction. A positive lower endpoint supplies a conservative
necessary condition; an unsupported law or unresolved threshold supplies
no new response verdict. Odd coordinates remain present in the report.
The separately admitted `full_phase_budget_status` requires the
original complete law and total `F<=4`. It exposes
`actual_full_phase_storage_upper_bound=139/20`, the conditional
`donor_potential_barrier_first_required` and an outward lower bound
on the necessary donor phase-potential decrease. The latter is not
identified incoming receiver work or a reusable reserve: released
potential can dissipate or remain elsewhere in the system. Outside
that law or budget these separate fields stay unavailable, without
discarding the geometric decomposition or changing the tangent scope.
The [independent excitation controls](../../tests/physics/test_relational_sine_receiver_excitation.py)
check the complete-support parity, energy-angle and nonlinear-mixture
identities; the [formation controls](../../tests/physics/test_relational_sine_formation.py)
retain source admission and report boundaries. None evaluates a new
trajectory or turns the tangent ceiling into a nonlinear upper bound.

<a id="sine-weighted-receiver-exclusion"></a>
#### A weighted nonlinear proof function excludes receiver identity below a source threshold

Keep the same complete half-weight sine law, unit storage scale, held
capacities, initial donor twist and initially flat receiver. All six source
coordinates remain free. The two bridges are cut edges, which supplies
more regional information than a bound on total phase storage alone.

Let `chi_D,chi_R` be the indicator columns of the two rings and let
`e_0,e_5` be the corresponding port coordinate columns. Define the
constant maps

\[
T_D=\operatorname{diag}(\chi_D)-e_0\chi_D^{\mathsf T},\qquad
T_R=\operatorname{diag}(\chi_R)-e_5\chi_R^{\mathsf T},\qquad
T_B=I-T_D-T_R.
\]

The full sine current satisfies `sum_i S_i=0`. Internal edge currents
cancel in each regional sum, so `chi_R^T S=sin(theta_10-theta_5)`;
the corresponding donor identity uses port zero. Consequently `T_R S`
is precisely the receiver's internal sine-current vector, embedded in
all eleven coordinates. Likewise `T_D S` is the donor internal current
and `T_B S` is the two-bridge current. Thus, for the internal ring
potentials and total bridge potential `V_B`,

\[
\nabla V_D=-T_DS,\qquad \nabla V_R=-T_RS,\qquad
\nabla V_B=-T_BS.
\]

These identities retain the full nodal current; they introduce neither
an independent receiver input nor a removed intermediary. They depend
on this supplied cut-edge geometry, not on arbitrary graph partitions.

Write `a=1/(2*pi)`, `q=Lx`, and define

\[
\boxed{\mathcal U
=\mathcal F+V_D+\frac75V_R+\frac76V_B
 -\frac13q^{\mathsf T}KS,\qquad
\mathcal F=\frac12x^{\mathsf T}Lx.}
\]

The unequal weights and cross term belong to an auxiliary proof
function. They are not storage allocations or additional coefficients
in the nodal dynamics. Put
`M=T_D+(7/5)*T_R+(7/6)*T_B` and `A=KLK`. The actual rows give

\[
\dot q=-\frac12LKq+aLKS,\qquad
\dot S=-aH(\theta)Kq,\qquad H(\theta)\preceq L.
\]

Direct differentiation, with no phase linearization, therefore yields

\[
\dot{\mathcal U}\le
-\binom{q}{S}^{\!\mathsf T}Q(a)\binom{q}{S},
\qquad
Q(a)=\begin{pmatrix}
\frac12K-\frac a3A &
-\frac12\left[aK(I-M)+\frac16A\right]\\
-\frac12\left[a(I-M)^{\mathsf T}K+\frac16A\right] &
\frac a3A
\end{pmatrix}.
\]

Here `q,S` both have zero ordinary sum. Testing unrelated constant
vectors in this quadratic form would impose a condition on states that
the full nodal law never supplies. Instead use the exact rational basis

\[
P=\begin{pmatrix}I_{10}\\-\mathbf1_{10}^{\mathsf T}\end{pmatrix},
\qquad Z=\operatorname{diag}(P,P),\qquad
Q_*(a)=Z^{\mathsf T}Q(a)Z.
\]

The original capacities fix
`K=diag(1/3,1/2,1/2,1/2,1/2,1/3,3/4,1/2,1/2,1/4,1/2)`.
Thus both matrices `Q_*(7/44)` and `Q_*(25/157)` are rational
twenty-dimensional matrices determined by the displayed formulas and
the twelve unit edges. Exact LDL elimination without pivoting gives
twenty positive diagonal pivots at each endpoint, each strictly larger
than `1/10000`. This claim is reproducible using the recurrence

\[
A^{(0)}=Q_*(a),\qquad d_j=A^{(j)}_{jj},\qquad
A^{(j+1)}_{rs}=A^{(j)}_{rs}
 -\frac{A^{(j)}_{rj}A^{(j)}_{js}}{d_j},\quad r,s>j.
\]

Only exact rational arithmetic is needed for these forty pivot signs.
The shared
[exact matrix owner](../../src/tnfr/mathematics/_exact_linear_algebra.py)
implements the same test. Since `157/50<pi<22/7`, the actual `a`
lies strictly between the two rational endpoints. The matrix `Q_*(a)`
is affine in `a`; convexity of the positive-definite cone proves
`Q_*(a)>0` for that entire interval. The finite rational certificate
therefore proves the all-state inequality for the actual transcendental
coefficient, rather than approximating a trajectory or inferring a
spectral sign from a floating-point eigensolver. In particular,

\[
\boxed{\dot{\mathcal U}<0\quad\text{whenever }(q,S)\ne(0,0).}
\]

At the declared preparation, both bridges are aligned, the receiver is
flat and `S=0`. Hence `U(0)=F+V_5`. At any relative equilibrium
the common form makes `F(t)=0`, and `q=S=0`; thus
`U_infinity=V_D+(7/5)*V_R+(7/6)*V_B`. The complete
[equilibrium catalog](#sine-eleven-node-asymptotic-equilibria)
has zero sine current on each bridge and each isolated ring. Every
nonflat receiver critical geometry has `V_R>=V_5`, and every other
term in this limiting weighted potential is nonnegative. It follows
that a nonflat receiver limit requires `U_infinity>=7*V_5/5`.

The already proved full-state convergence theorem applies unchanged.
For a nonzero source, `q(0)!=0`, so strict decrease over an initial
time interval gives `U_infinity<U(0)`. Therefore

\[
\boxed{F\le F_*:=\frac25V_5=\frac{5-\sqrt5}{2}
\quad\Longrightarrow\quad
\text{the receiver converges to relative phase consensus}.}
\]

At `F=0` the initial donor-only equilibrium is stationary, so the same
conclusion holds. Strict initial decrease includes the closed threshold
for real preparations; a finite rational source cannot equal this
irrational threshold exactly. Its admission can nevertheless use only
rational comparisons: with `z=5-2*F`, the condition is equivalent to
`z>=0` and `z*z>=5`.

This theorem treats all six source coordinates and their nonlinear
mixtures, without a time cutoff or assumed waveform. It determines the
receiver's limiting geometry, not the donor's endpoint, any intervening
wrapped winding, or the absence of a transient potential-barrier visit.
The cross term prevents interpreting `U` as a pointwise upper bound on
receiver phase storage. Above `F_*`, failure of this sufficient
certificate is not evidence of receiver acquisition. The original
`F<=4` accessibility problem remains open on that unresolved range.

The mathematical argument uses the fixed initial phase geometry and total
form storage, not zero receiver form: it also holds for arbitrary initial
forms on the eleven nodes with those same phases and capacities. The public
reader retains the declared six-coordinate preparation contract; this wider
theorem scope does not install a new preparation or response campaign.

The shared [formation owner](../../src/tnfr/physics/relational_sine_formation.py)
exposes this separate result through `source.receiver_localization()`
as `SineReceiverLocalization`. It validates the same complete law and
preparation before applying the exact algebraic threshold. The existing
receiver-transfer reader retains that report and uses a certified
localization as an exclusion of maintained receiver acquisition. The
[localization controls](../../tests/physics/test_relational_sine_receiver_localization.py)
independently reconstruct the cut-current maps and rational derivative
matrices, verify their LDL signs and check public scope boundaries.
No trajectory or source search is part of this certificate.

### One nonuniform donor preparation at the same storage budget

Consider the exact rational preparation

\[
D=\frac13(10,13,14,14,13),\qquad H=\frac43,
\qquad \delta=\frac12.
\]

It is donor-reflection-even, lies outside the silent subspace,
and has

\[
\begin{aligned}
F&=4,\\
q&=(0,2/3,1/3,1/3,2/3,-4/3,0,0,0,0,-2/3),\\
N&=\frac{37}{27},\qquad
m_1=\frac{43}{54},\qquad m_2=\frac{47}{54},\\
R&=\frac{43}{74},\qquad
\gamma^2=\frac{47}{74}.
\end{aligned}
\]

Its ratio `N/F=37/108<2/3` disproves extension of the
constant-donor lower bound `N>=2F/3` to arbitrary donor
shape. It is not asserted to minimize any quadratic form.
It also does not retain the same initial receiver drive as the
balanced `F=4,H=2` preparation: its first receiver rates
and odd acceleration have two-thirds of that magnitude.
Less initial loss is not an equal-response improvement here.

The short-window directional certificate can be inconclusive
for this preparation. That is not evidence of formation or of an
interesting surviving basin. The extended rational majorant in
Section 21 resolves the same preparation without evaluating a
trajectory.

For `delta=1/2`, use `lambda=3`. Since
`gamma<4/5` and `a=b<1/6`, choose the supplied horizon
`tau=6/5`. It satisfies `omega*tau<3/5`. The
factorial estimate gives
`cosh(3/5)<=50/41<5/4`, so the polynomial coefficient
can be bounded by

\[
c\le\frac58\bar\gamma\,\lambda ab<\frac1{24}.
\]

With `d=eR=43/148`, a conservative lower polynomial is
`g(t)=1-(43/148)t-t^2/24`. Its endpoint and exact
integrated loss satisfy

\[
g(6/5)=\frac{547}{925}>0,\qquad
E(0)-E(6/5)\ge
\frac{7564289}{13875000}>\frac{27}{50}.
\]

The joint target-entry boundary permits loss of only
`4-B_5<27/50`. One elementary verification of this strict
bound is `sqrt(2)>141/100`, which gives
`cos^2(3*pi/8)<59/400<(77/200)^2` and hence
`B_5>173/50`.

It remains necessary to rule out earlier entry. At this fixed
contrast the largest receiver-edge sum
`nu_i/sqrt(d_i)+nu_j/sqrt(d_j)` is
`5/(2*sqrt(2))`. With `E(0)=V_5+4<15/2`,
the global receiver-gap speed satisfies

\[
G<\frac{55}{48},\qquad
G\,\frac65<\frac{11}{8}<\frac{\pi}{2}.
\]

The receiver remains acute with winding zero until the storage
is already below the joint entry requirement. The nonuniform
preparation is therefore excluded from the two-twist target.
Its nonzero receiver response still occurs, but does not
establish formation. The decisive change was a sharper proof
window for the same law and same initial data.

### A class-level lower bound on any target-entry time

A separate action estimate applies to the whole six-coordinate
preparation class. Write `D_loss(t)=E(0)-E(t)` for accumulated
loss, distinguishing it from donor form. If a first joint acute
target entry occurs at time `T`, its boundary storage implies

\[
D_{\mathrm{loss}}(T)\le F-\beta B_5.
\]

If the right-hand side is nonpositive, the earlier strict-storage
obstruction already applies. Otherwise, the initially flat
receiver can acquire winding one only if some continuous lifted
receiver edge difference first reaches `+pi` or `-pi`
at a time `s<=T`. Before any such crossing, its wrapped
edge increments equal those continuous differences and their
sum around the cycle remains zero.

For `k_i=nu_i/d_i`, weighted Cauchy--Schwarz gives

\[
|\dot\theta_j-\dot\theta_i|^2
\le b^2(k_i+k_j)\,q^\mathsf TKq
=\frac{b^2(k_i+k_j)}e\,\dot D_{\mathrm{loss}}.
\]

Let `k_max` be the largest `k_i+k_j` over receiver
edges. Integrating to that necessary crossing and applying
Cauchy--Schwarz in time yields

\[
\pi^2
\le\frac{b^2k_{\max}}e\,sD_{\mathrm{loss}}(s)
\le\frac{b^2k_{\max}}e\,T(F-\beta B_5).
\]

Consequently every such entry must satisfy

\[
\boxed{
T\ge\frac{e\pi^2}{b^2k_{\max}(F-\beta B_5)}.}
\]

At the present default coefficients and `delta=1/2`,
`k_max=5/4`. For `F<=4` and positive allowable loss,
`B_5>173/50` and `pi>3` imply

\[
T>\frac{80\pi^4}{27}>240.
\]

This is a necessary time conditional on target entry, expressed
in the declared sine-model clock. It is neither a prediction
that entry occurs nor a global phase-speed bound for trajectories
that spend more than the permitted loss. The shared report's
`phase_action_bound` retains the allowable-loss interval, actual maximum
receiver-edge mobility and a lower bound on the necessary entry time.
It supplies no time when the positive allowance is unavailable.
An admitted horizon below that bound can exclude earlier joint entry;
excluding entry afterwards still requires its separately justified loss
deficit. Any successful path would have to preserve the small remaining
allowance for at least that long.

<a id="sine-maintained-target-obstruction"></a>
### A global auxiliary function excludes the maintained target for the whole class

The early-window bounds retain useful preparation-specific information,
but a joint full-field argument closes the stated six-coordinate class
without optimizing independent moment ranges or extending a trajectory
window. Keep exactly `delta=1/2`, `beta=1`, the supplied eleven-node
support, initial phase configuration and held capacities of this section.
For this auxiliary-function argument, allow positive effective coefficients
in the explicit sufficient domain

\[
e>0,\qquad w>0,\qquad 0<r:=\frac we<\frac32.
\]

This is an open neighborhood of the reference ratio `r=1`, not a sharp
formation threshold or an optimized parameter range. It changes only the
relative coefficients of the same complete normalized-sine law. The
earlier finite preparations, responses and default-clock time estimates
retain their original coefficients; they are not reevaluated here.
Let the instantaneous form storage be
`mathcal F(t)=x(t)^T L x(t)/2`, so the preparation's `F` is `mathcal F(0)`.
Write

\[
y=K^{1/2}q,\qquad \zeta=K^{1/2}S,\qquad
B=K^{1/2}LK^{1/2},\qquad C(\theta)=K^{1/2}H(\theta)K^{1/2},
\qquad a=b=\frac w\pi.
\]

These are the full eleven-node gradient variables from Section 21, not
a six-node evolution. Their exact complete rows are

\[
\dot y=-eBy+aB\zeta,\qquad \dot\zeta=-bC(\theta)y.
\]

For every phase state, the edge Hessian representation gives
`C(theta)<=B` in quadratic-form order, since each cosine is at most one.
No acute-phase, winding-preservation or positive-Hessian premise is needed.
Both `sum_i q_i=0` and `sum_i S_i=0`, so `y` and `zeta` are perpendicular
to `ker(B)=span(K^(-1/2)*1)` at all times. This excludes the zero mode
from the matrix estimate below without discarding a consumed coordinate.

The positive spectrum of `B` lies in a known rational interval. The
[full-support path bound](#an-independent-full-graph-spectral-gap-bound)
gives `lambda_2(L)>=1/11`. Here `K>=I/4`, and the nonzero spectrum of
`B` equals that of `L^(1/2) K L^(1/2)>=L/4`. The min-max principle thus
gives `lambda_+(B)>=1/44`. The existing upper estimate
`B<=2*max_i(nu_i)*I` gives `lambda_max(B)<=3`. Hence

\[
\frac1{44}\le\lambda\le3
\quad\text{for every positive eigenvalue of }B.
\]

Now define an **auxiliary proof function**, not a new physical storage
or term in the dynamics:

\[
\boxed{\mathcal W= c\,\mathcal F+V(\theta)-h\,y^\mathsf T\zeta,
\qquad c=\frac45,\qquad h=\frac a{2e}=\frac r{2\pi}.}
\]

The coefficient `c` and the mixed coefficient `h` belong only to this
certificate. At the reference `e=w=1/2`, one has `h=a=1/(2*pi)`, so this
is the same function as the fixed-coefficient obstruction. The full form
and phase storage derivatives are
`mathcal F_dot=-e*||y||^2+a*y^T*zeta` and
`V_dot=-b*y^T*zeta`, with `a=b`. Differentiating `W` and then using
`C(theta)<=B` gives the all-state inequality

\[
\begin{aligned}
\dot{\mathcal W}
={}&-ce\|y\|^2+a(c-1)y^\mathsf T\zeta
 +he\,y^\mathsf TB\zeta-ha\,\zeta^\mathsf TB\zeta
 +hb\,y^\mathsf TC(\theta)y\\
\le{}&-ce\|y\|^2+a(c-1)y^\mathsf T\zeta
 +he\,y^\mathsf TB\zeta-ha\,\zeta^\mathsf TB\zeta
 +hb\,y^\mathsf TBy.
\end{aligned}
\]

Diagonalize the constant symmetric matrix `B` on its positive spectral
subspace and put `rho=a/e=r/pi`. The contribution of one mode to the last
line is

\[
-e\left(c-\frac{\rho^2\lambda}{2}\right)y_\lambda^2
+a\left(c-1+\frac\lambda2\right)y_\lambda\zeta_\lambda
-\frac{e\rho^2\lambda}{2}\zeta_\lambda^2.
\]

The admitted ratio and `pi>3` give `0<rho<1/2`. The positive magnitude
of its first negative term is therefore strictly greater than
`e*(4/5-3/8)=17*e/40`; the last negative term also has positive magnitude.
The determinant condition for strict negative definiteness is

\[
p(\lambda):=(1-c)^2-(c+1)\lambda
 +\left(\frac14+\rho^2\right)\lambda^2<0.
\]

Indeed, four times the product of those two magnitudes minus the square
of the mixed coefficient equals `-a^2*p(lambda)`. This
condition holds uniformly, using only rational bounds:

\[
p(\lambda)<\overline p(\lambda)
 :=\frac1{25}-\frac95\lambda+\frac12\lambda^2,
\]
\[
\overline p(1/44)=-\frac{63}{96800}<0,
\qquad\overline p(3)=-\frac{43}{50}<0.
\]

The polynomial `pbar` is convex, so it is negative throughout the
interval between those two endpoints. Every mode quadratic is therefore
strictly negative unless its two coordinates vanish. Consequently

\[
\boxed{\dot{\mathcal W}\le0,\qquad
\dot{\mathcal W}<0\quad\text{whenever }(q,S)\ne(0,0).}
\]

This is a global consequence of the same complete positive-loss law
throughout the admitted ratio domain. It does not freeze the sine term,
linearize the phase field or assume
the donor remains in its original winding sector. The exact initial
condition `S(0)=0` gives, for every admitted donor/intermediary form,

\[
\mathcal W(0)=\frac45F+V_5.
\]

At the specified maintained two-twist target, uniform form and exact
sine balance give `q=S=0`, while `V=2V_5`. Thus

\[
\mathcal W_{\rm target}=2V_5,\qquad
\mathcal W_{\rm target}-\mathcal W(0)=V_5-\frac45F.
\]

Whenever this last margin is positive, convergence is impossible:
`W` is continuous and invariant under common form and phase origins, so
convergence to the declared target modulo those origins would require
convergence to a larger `W` value, contradicting its nonincrease.
This sufficient criterion keeps `F` from the actual preparation; it does
not require an artificial cutoff in a reader applying the theorem.

For the whole declared research class `F<=4`, the margin satisfies

\[
V_5-\frac45F\ge V_5-\frac{16}{5}>\frac14>0.
\]

The last inequality follows from `sqrt(5)<56/25`, whose square is
strictly greater than five. Therefore **every preparation in the
six-coordinate class with `F<=4` is excluded from convergence to the
specified maintained two-twist target for every admitted ratio
`0<w/e<3/2`**. Neither the margin nor its strict positive lower bound
depends on that ratio: the mixed term vanishes at both endpoints. The
argument does not select a preferred ratio or imply formation outside
this sufficient domain.

In particular no such preparation can enter a recovery basin whose
valid full-state theorem guarantees that convergence under this unchanged
law. This is stronger than the earlier profile-by-profile loss checks
for the stated maintained target. It does not assert that the trajectory
can never visit the larger joint acute winding region: the cross term
and nonuniform form need not vanish at a transient visit. Nor does it
exclude receiver-only winding if the donor identity is lost. The action
bound above retains its separate necessary condition for acute-region
entry and must not be relabeled as a predicted entry time.

<a id="sine-formation-clock-covariance"></a>
### Constitutive ratio and constant clock changes are different operations

Changing `w/e` changes the relative contributions in the complete field.
By contrast, for a constant `kappa>0`, set `t'=kappa*t` and keep the
state coordinates, graph, held capacities, `K` and `beta` fixed. The same
state curve in the new clock satisfies

\[
\frac{dx}{dt'}=-e'Kq+a'KS,\qquad
\frac{d\theta}{dt'}=b'Kq,\qquad
(e',w',a',b')=\frac1\kappa(e,w,a,b).
\]

Both evolution rows change. Consequently `r'=r`, `h'=h`, and the
state function `W` is unchanged, with

\[
\frac{d\mathcal W}{dt'}=\frac1\kappa
\frac{d\mathcal W}{dt}\le0.
\]

The maintained-target margin and exclusion are thus independent of
this constant clock convention. Elapsed times and the phase-action
lower bound above transform by `T'=kappa*T`, since `e'/b'^2=kappa*e/b^2`.
The default numerical statement `T>240` belongs to its stated clock,
not to every rescaling of it. This proves neither a preferred physical
clock nor covariance under state-dependent or time-dependent clock
changes.

The implementation's
[`RelationalExchangeModel`](../../src/tnfr/dynamics/relational.py)
normalizes the supplied EPI and phase weights once and stores finite
effective coefficients. Its formation reader uses those stored values,
including their exact represented ratio. Commonly scaling raw constructor
weights therefore leaves the ideal normalized coefficient pair unchanged;
finite materialization is still governed by the shared admission contract.
It is **not** the clock change just derived. The arbitrary positive
coefficients in that derivation express the whole-field time scale outside
the normalized coefficient convention, without introducing a new runtime
parameter. Using another time unit also requires transforming the
integration step, horizons and all reported rates consistently.

<a id="sine-receiver-transfer-admission"></a>
### Receiver identity transfer requires donor loss and its own passage proof

Return to the original effective coefficients `e=w=1/2`, `beta=1`,
`delta=1/2` and structural clock. Keep all eleven nodes, their held
capacities, the initial phases, arbitrary donor forms `D_0,...,D_4`,
intermediary form `H`, zero receiver forms and `F<=4` unchanged.
The proposed endpoint is now **transfer**, not the excluded coexistence:
the donor is flat, the receiver has winding `+1`, both bridge phase gaps
vanish, and form is uniform. No support event, input or replacement law
is supplied.

#### Exact endpoint, conserved origins and full-state recovery

In turns, choose the reference

\[
v^{\rm tr}=(0,0,0,0,0,\ 0,1/5,2/5,3/5,4/5,\ 0),
\qquad \theta^{\rm tr}=2\pi v^{\rm tr}.
\]

The donor and bridge sine terms vanish individually. The receiver's two
ring terms cancel at every node, including its port. Thus `S=0` on the
full graph; uniform form gives `q=0`. This is an exact equilibrium of
both complete rows. Its phase Hessian assigns weight one to donor and
bridge edges, and weight `cos(2*pi/5)>0` to receiver edges. It is positive
on the full common-phase quotient, including the intermediary and
relative movement of the two rings. The held capacity asymmetry does
not remove that property.

For this capacity vector, the conserved-coordinate weights are

\[
\omega=(3,2,2,2,2,\ 3,4/3,2,2,4,\ 2),\qquad
M_\omega:=\sum_i\omega_i=\frac{76}{3}.
\]

Section 18 therefore fixes the final uniform form to

\[
x_\infty=M_x=\frac3{76}
\left[3D_0+2(D_1+D_2+D_3+D_4)+2H\right].
\]

The initial weighted phase-turn sum is `4`, whereas the displayed
target representative has weighted turn sum `82/15`. If the final
continuous lifts have the form
`theta_i/(2*pi)=v_i^tr+gamma+m_i`, with integers `m_i`, conservation gives

\[
\gamma=-\frac{11}{190}
       -\frac3{76}\sum_i\omega_i m_i.
\]

In particular, the representative with no additional integer turns
has common shift `-11/190` turns. The actual integer vector is a property
of the lifted history, not prescribed by a snapshot or this endpoint
test. The circular target admits the resulting common origin; no
inconsistent fixed phase mean is imposed. This compatibility does not
produce a trajectory realizing that history.

The existing whole-support recovery theorem in Section 18 applies to
this target. For example, retain a consistent local deviation lift,
remove only the two common origins, and let

\[
Z^2=\|P(x-M_x\mathbf1)\|^2+
\|P(\theta-\theta^{\rm tr})\|^2,\qquad
P=I-\mathbf1\mathbf1^\mathsf T/11.
\]

The same maximum target edge angle `2*pi/5` and full-support gap
`lambda_2(L)>=1/11` permit `r=1/16` and `c_r>1/5`. Consequently the
following strict conditions are sufficient for trapping and recovery:

\[
\boxed{Z<\frac1{16},\qquad E-V_5<\frac1{28160}.}
\]

For an uncertainty set both upper bounds must hold uniformly, with all
eleven forms and phases retained. This is a nonempty basin around the
exact target, not a certificate that the original preparation enters
it. Its small radius must not be replaced by a receiver-only norm.

#### Endpoint budgets permit transfer but do not establish it

At the preparation and transfer target, respectively,

\[
\begin{array}{c|cc}
&\text{initial}&\text{transfer target}\\ \hline
E&F+V_5&V_5\\
\mathcal W&\frac45F+V_5&V_5
\end{array}
\]

Both endpoints have exact `S=0`; the target also has `q=0`. Hence the
coexistence obstruction's positive target gap is absent: the transfer
gap in `W` is `-(4/5)*F`. If convergence occurred, the full accumulated
continuous loss would equal `F`. Storage released while the donor
unwinds remains inside this same balance; it is not an external input
or a reusable loss reservoir. For `F>0`, these two endpoint inequalities
are compatible with transfer, but do not prove it.

#### A phase passage barrier and exact excluded controls

Define the target phase region by **donor strictly acute with winding
zero and receiver strictly acute with winding one**. Bridge phases and
forms are not constrained in this definition. Any convergence to the
transfer target enters this open product region at a finite time.
At the positive infimum `T` of entry times, both rings are closed acute
with those windings, and at least one is on an acute face.

If the receiver is on a face, the previous convexity proof gives phase
storage at least `B_5`. If the donor is on a face, its edge at `pi/2`
or `-pi/2` costs one, while the closed-acute winding-one receiver costs
at least `V_5`. Since `1+V_5>B_5` (`V_5>3` and `B_5<4` suffice), every
such first entry satisfies

\[
E(T)\ge B_5,\qquad
D_{\rm loss}(T)=E(0)-E(T)\le
A_{\rm tr}:=F+V_5-B_5.
\]

There is no added `V_5` from a maintained donor at this boundary.
Reusing the coexistence allowance `F-B_5` would reject this different
endpoint for the wrong reason.

For `F>0`, the connected support and zero receiver forms imply
`q(0)!=0`. Loss is strictly positive on an initial interval, so transfer
requires the strict condition

\[
\boxed{F>B_5-V_5.}
\]

This also follows by considering the first exit from the initial product
region (acute donor winding one, acute receiver winding zero): a donor
face costs at least `B_5`, whereas a receiver face together with the
remaining closed-acute donor costs at least `1+V_5>B_5`. Consequently
`F<=B_5-V_5` cannot leave that initial product region. At equality the
strict initial loss prevents the boundary from being reached. When
`F=0`, both full gradients vanish and the preparation is stationary.

The exact silent subspace proved earlier,
`D_0=H=0`, `D_1=-D_4`, `D_2=-D_3`, remains an independent exclusion,
including its nonzero-storage members: its receiver stays flat forever.
Nonzero loss or satisfaction of the endpoint budgets does not remove
that symmetry obstruction.

#### Two necessary winding changes consume disjoint parts of one loss

Before the target-product entry at `T`, the receiver must cross an
antipodal edge, and the donor must also cross one in leaving winding
one. Let corresponding crossing times be `s_R,s_D` with
`0<s_R,s_D<T`. They need not have any prescribed order and may coincide.
An initially flat receiver edge requires at least `pi` displacement of
its continuous lifted phase difference. A donor edge initially has
increment `alpha=2*pi/5` modulo `2*pi`, so any antipodal value requires
at least `pi-alpha=3*pi/5` displacement. This statement also covers
the closing edge's initial raw lift.

For a ring `C`, allocate only its actual nodal loss

\[
D_C(t)=e\int_0^t\sum_{i\in C}k_iq_i(s)^2\,ds,
\qquad k_i=\nu_i/d_i.
\]

For its edge `i--j`, weighted Cauchy--Schwarz gives

\[
|\theta_j(s)-\theta_i(s)-\theta_j(0)+\theta_i(0)|^2
\le\frac{b^2(k_i+k_j)}e\,sD_C(s).
\]

The actual maximum edge mobilities are `k_D,max=1` on the donor and
`k_R,max=5/4` on the receiver. Their node sets are disjoint, so
`D_D(s_D)+D_R(s_R)<=D_loss(T)`: intermediary loss is omitted, not
double-counted. Using the strict inequalities `s_D,s_R<T` therefore yields

\[
D_{\rm loss}(T)
>\frac{e\pi^2}{b^2T}
  \left(\frac{9}{25}+\frac45\right)
 =\frac{29e\pi^2}{25b^2T},
\]
\[
\boxed{T
>\frac{29e\pi^2}{25b^2 A_{\rm tr}}
 =\frac{58\pi^4}{25(F+V_5-B_5)}.}
\]

The bound requires `A_tr>0` and is expressed in the original clock.
It is a necessary earliest-entry condition, not a predicted transfer
time. It neither assumes that the donor unwinds first nor adds two
estimates of the same dissipated quantity. It uses the full support
through each `q_i`, even when the loss is partitioned by ring.

A certified source-specific loss lower bound `L(tau)` can also be
reused without copying the coexistence verdict. If the admitted horizon
`tau` is strictly below the transfer entry-time lower bound, and
`L(tau)>A_tr`, entry is excluded both before `tau` by the action estimate
and after `tau` by nondecreasing accumulated loss. The deficit must be
recomputed against `A_tr`. Independent lower bounds on the same total
loss combine by their maximum, not by addition; this differs from adding
the disjoint nodal contributions in the proof above.

#### What remains undecided

The target is an exact, locally recoverable collective state compatible
with the conserved coordinates. The low-budget, zero-form and silent
controls exclude specified preparations, and the joint action estimate
constrains every remaining successful path. These results do **not**
classify accessibility for the rest of the six-coordinate class `F<=4`.
The [static donor well](#sine-donor-well-retention) and
[dissipative capture](#sine-donor-dissipative-capture) select the original
donor endpoint on their stated source classes. Outside those certificates,
the [receiver port-work and order conditions](#sine-receiver-port-passage)
remain necessary restrictions, not a classification of accessibility.
In particular, a first visit to the larger acute product region need not
satisfy the full-state norm and excess-storage bounds of the recovery
basin, and local receiver derivatives do not bound that later state.

The missing positive argument is an actual evolution from an admitted
preparation into that whole-support basin. A negative argument instead
needs a valid separating invariant or a loss/action bound ruling out
every necessary passage for the remaining preparations. The endpoint
inequalities and existing finite loss estimates supply neither claim
uniformly. No trajectory, new preparation, parameter search or changed
law is inferred from this gap. The maintained coexistence exclusion
remains intact; transfer cannot be counted as replication of a second
maintained organization.

A static control makes this limitation more precise. First shrink the
initial form deviations continuously toward the conserved uniform form
`M_x*1`, keeping the initial phases fixed. Then flatten the donor along
`theta_j=j*t`, from `t=2*pi/5` to zero, and twist the receiver along the
same family from zero to `2*pi/5`, keeping both ports and the
intermediary aligned. At each phase step a single common phase shift
can preserve the weighted lifted mean. Thus this comparison path respects
both conserved means and the supplied support. Its first leg decreases
`W` from `(4/5)*F+V_5` to `V_5`. On either phase leg, uniform form gives

\[
\mathcal W=V=5-4\cos t-\cos4t,\qquad
\frac{dV}{dt}=8\sin(5t/2)\cos(3t/2),\qquad
0\le t\le2\pi/5.
\]

The maximum along these declared legs is `7/2`, at `t=pi/3`. Therefore
for `F>=(5/4)*(7/2-V_5)` the source and transfer target lie on a continuous
path inside `W<=W(0)`, even with both conserved coordinates retained.
The [critical-set argument below](#sine-donor-well-retention) proves that
`7/2` is the exact static minimax between these two equilibrium wells.
The displayed path is still not a dynamics construction: `W` increases
on parts of it, which cannot occur on the actual flow. In particular,
sufficiently energetic members of the silent
subspace also satisfy this connectivity test while remaining exactly
unable to transfer. Connectivity of this auxiliary sublevel is therefore
insufficient; the missing dynamical direction cannot be supplied by
static endpoint and budget checks.

The existing
[`SineMediatedFormation.receiver_transfer()`](../../src/tnfr/physics/relational_sine_formation.py)
reader reuses the admitted preparation and horizon while preserving the
original coexistence report. Its separate `SineReceiverTransferAdmission`
retains exact target geometry, a compatible lifted origin, the
source-minus-target margins `F` and `(4/5)*F`, the entry allowance and
both actual edge mobilities. The necessary time uses the shared outward
interval calculation; a missing positive allowance leaves that time
unavailable. These necessary inequalities do not consume the research
ceiling `F<=4`, so their reader does not impose an artificial budget cap.
A passing necessary-condition status is not a successful
transfer or an initial/evolved recovery-basin certificate. The
[formation controls](../../tests/physics/test_relational_sine_formation.py)
check those distinct target and numerical admission boundaries, and the
[independent algebra controls](../../tests/physics/test_relational_sine_formation_barrier.py)
verify the target, conserved-origin calculation, disjoint action constants
and static comparison path. None advances a transfer trajectory.

<a id="sine-receiver-port-passage"></a>
### Receiver acquisition requires accumulated port work and a geometric passage

Keep the same supplied two-C5/intermediary support and the six-coordinate
preparation: donor winding `+1`, receiver flat, intermediary aligned,
and receiver form initially zero. The following identities and necessary
conditions hold for the complete sine law with `e,w,beta>0` and
strictly positive held capacities, without a restriction on `w/e`.
The numerical specialization retains the original half weights,
`beta=1`, `delta=1/2` and `F<=4`. No external drive, input law or
event is added by treating the receiver boundary as a port.

#### Use the actual full-node loss and the existing boundary-work convention

Let `R={5,...,9}`, with port `p=5` and intermediary `h=10`.
Let `L_R` be the isolated receiver-cycle Laplacian only for defining
its **internal storage**:

\[
E_R=F_R+\beta V_R,\qquad
F_R=\frac12x_R^\mathsf TL_Rx_R,\qquad
V_R=\sum_{\{i,j\}\in C_5^R}(1-\cos(\theta_j-\theta_i)).
\]

Its consumed rates still use the full eleven-node `q=Lx`, `S` and
`k_i=nu_i/d_i`; in particular the port degree remains three. Define

\[
f=x_h-x_p,\qquad s=\sin(\theta_h-\theta_p),\qquad
P_R=f\dot x_p+\beta s\dot\theta_p,
\]
\[
D_R(t)=e\int_0^t\sum_{i\in R}k_iq_i(u)^2\,du.
\]

The receiver's internal form gradient is `(q_i)_(i in R)+f*e_p`,
and `grad V_R=-(S_i)_(i in R)+s*e_p`, with both restricted vectors
retaining their **full-support** values. Therefore

\[
\begin{aligned}
\dot E_R
&=\sum_{i\in R}(q_i\dot x_i-\beta S_i\dot\theta_i)
  +f\dot x_p+\beta s\dot\theta_p\\
&=-e\sum_{i\in R}k_iq_i^2+P_R,
\end{aligned}
\]

using `a=w/pi=beta*b`. Since `E_R(0)=0`, the exact integrated
balance is

\[
\boxed{J_R(t):=\int_0^tP_R(u)\,du=E_R(t)+D_R(t).}
\]

This uses the existing intermediary-star boundary-work sign convention:
`P_R` is the **negative** of the port-5 work rate into the hidden star
in the [mediated-pressure balance](#causal-sine-environmental-pressure).
The form and phase contributions are both necessary. It is not the
weighted-form flux reported by a regional form-transfer observation.
No second regional pressure or dissipation convention is introduced.

Although `J_R(t)>=0` for this zero-storage preparation, neither its
derivative `P_R` nor its increments need be nonnegative. Work can return
through the same port. Likewise a positive instantaneous boundary power
does not prove an increase of internal receiver storage: the simultaneous
full-node loss must be subtracted.
For the actual preparation, intermediary form `H` gives the exact control
`P_R(0)=D_R'(0)=e*H^2/3` and hence `E_R'(0)=0`. Positive incoming
power at the initial port is then spent entirely on simultaneous loss.

#### The exact receiver barrier demands a running maximum, not final work

For the relative phase torus of an isolated C5, the exact minimum
possible maximum of `V_R` along a continuous path from consensus to
either uniform winding-one twist is `7/2`. This reuses the
[critical-set and well-separation proof](#sine-donor-well-retention):
the twist is an isolated minimum of value `V_5`, and there is no
critical value between `V_5` and the first saddle value `7/2`.
Regular sublevel components cannot merge in that interval. The path
`theta_j=j*t`, `0<=t<=2*pi/5`, attains maximum `7/2`; reflection
gives the opposite handedness. This is an exact phase-geometry minimax,
not a sharp dynamical work threshold or a newly selected law.

Any actual trajectory converging to either receiver twist must therefore
have a finite first time `tau_R>0` with `V_R(tau_R)=7/2`.
To make the finite-time implication explicit, take a sufficiently late
phase state near the limiting twist. A short path from it to the exact
twist stays in that minimum's sublevel component below `7/2`.
Appending it to the actual receiver phase path proves that the barrier
was already crossed earlier. An arbitrary transient nonzero winding,
without convergence to this maintained geometry, is a different claim.

At this first barrier,

\[
\boxed{J_R(\tau_R)
=\frac72\beta+F_R(\tau_R)+D_R(\tau_R)
>\frac72\beta.}
\]

Strictness follows from the actual phase row: if `D_R(tau_R)=0`,
positivity and continuity force `q_i(t)=0` at every receiver node
throughout `[0,tau_R]`. Then `dot theta_i=b*k_i*q_i=0` there, so
the receiver phase remains flat and cannot reach the barrier. This
argument also covers preparations with zero intermediary form but a
later donor response; it does not assume positive initial receiver loss.

The condition is on a **running** signed work integral. At a maintained
twisted limit, `F_R` tends to zero and
`J_R(infinity)=beta*V_5+D_R(infinity)` by the same balance. Those
endpoint facts do not imply that this final value exceeds `7*beta/2`:
the receiver can return work after its barrier passage. A final-work
budget cannot replace the required transient maximum.

There is also a quantitative receiver-action cost. Let `lambda_R>0`
be any justified upper bound for the spectrum of
`K_R^(1/2)L_RK_R^(1/2)`. From the initial flat phase, use its
continuous lift `eta_R=theta_R(tau)-theta_R(0)` and `H_R<=L_R`:

\[
V_R(\tau)\le\frac12\eta_R^\mathsf TL_R\eta_R
\le\frac{b^2\lambda_R\tau}{2e}D_R(\tau).
\]

The second inequality follows by inserting
`eta_R=b*K_R^(1/2)*integral_0^tau(K_R^(1/2)*q_R)dt` and applying
Cauchy--Schwarz in time; here `q_R` is the restriction of the full
`q=Lx`, never the internal gradient `L_R*x_R`. Thus at the first barrier

\[
D_R(\tau_R)\ge\frac{7e}{b^2\lambda_R\tau_R},\qquad
J_R(\tau_R)\ge\frac72\beta+
                 \frac{7e}{b^2\lambda_R\tau_R}.
\]

For the original capacities, `K_R^(1/2)L_RK_R^(1/2)` is bounded
above by the receiver principal compression of the full `B`; that
compression additionally retains the bridge contribution on the port
diagonal. Thus `lambda_R<=3`, and at half weights this gives
`D_R(tau_R)>=14*pi^2/(3*tau_R)`. The storage scale enters the
phase rate through `b=w/(beta*pi)` and the barrier through `beta`;
there is no additional factor of `beta` in the unit-phase inequality.
This first potential-barrier action cost is distinct from the earlier
antipodal winding-change estimate. The unknown `tau_R` is not a
forecast, and this additional loss bound tends to zero as `tau_R`
tends to infinity.

#### A necessary order of potential barriers for the original budget range

Let `tau_D` be the donor's first time at `V_D=7/2`, or infinity
if that never occurs. Until that time its continuous phase stays in
the original one-twist component of `{V_D<7/2}`, where `V_D>=V_5`.
For every nonzero preparation, initial `q!=0` and positive held
capacities give strict global loss over an initial time interval.
Consequently `E(t)<E(0)=F+beta*V_5` for all positive `t`.
The zero-form preparation is stationary and has no receiver passage.

If `F<=7*beta/2` and `tau_R` is finite, it is impossible to have
`tau_D>=tau_R`: at the receiver barrier the donor would still have
`V_D>=V_5`, requiring `E>=beta*(7/2+V_5)`, contrary to that
strict energy inequality. Therefore

\[
\boxed{F\le\frac72\beta,\quad\tau_R<\infty
\quad\Longrightarrow\quad \tau_D<\tau_R.}
\]

Indeed energy at the receiver barrier gives the stronger pointwise
restriction `V_D(tau_R)<F/beta+V_5-7/2<=V_5`. The conclusion
concerns potential-well passage. It does **not** say that the donor
has already changed its wrapped winding, entered an acute consensus
chart, or selected its final equilibrium.

At any instant when both ring potentials are at least `7/2`, total
storage is at least `7*beta`. Thus a necessary condition for such a
simultaneous barrier state is

\[
\boxed{F>\beta(7-V_5)=\frac\beta4(3+5\sqrt5).}
\]

For `beta=1`, exclusion of simultaneous barrier states can use exact
rational source arithmetic: put `z=4*F-3`; exclusion holds when
`z<=0` or when `z>0` and `z^2<=125`. This is not exclusion of
sequential passages. In particular it cannot be substituted for a
receiver-transfer verdict on the upper part of the source class.

#### The remaining estimate is a bound on actual accumulated receiver supply

These results constrain a receiver-directed mechanism without proving
that it succeeds. A uniform bound `sup_t J_R(t)<=7*beta/2` would
exclude maintained receiver acquisition; more sharply, a bound on
`sup_t[J_R(t)-D_R(t)]` below the same barrier would do so. A
positive route still needs an actual full state entering the receiver
recovery basin, not only enough boundary work or a barrier crossing.

The current source controls provide neither such a running-supply upper
bound nor a successful basin-entry state for the remaining preparation
class. The instantaneous port owner fixes the correct sign and channels,
but does not integrate or bound their future history. Treating the donor
and intermediary signals as freely adjustable inputs would change this
autonomous problem. The ordering theorem, necessary action and exact
work balance do not fill that gap with a waveform, time cutoff, new
preparation or a replacement law.

The existing
[`comparison.mediated_pressure(mediator=10)`](../../src/tnfr/physics/relational_sine_mediation.py)
retains `port_boundary_form_work`, `port_boundary_phase_work` and
`port_boundary_work` in its actual `ports` order. Their orientation is
into the hidden star; the receiver's incoming rate is the negative of
the port-5 entry. These per-port arrays own the corresponding aggregate
sums and use full port rates. They provide a snapshot, not any of the
future accumulated-work quantities in this theorem.

The formation owner's existing
[`SineReceiverTransferAdmission`](../../src/tnfr/physics/relational_sine_formation.py)
retains the necessary barrier, running-work lower bound and the
conditional `donor_barrier_first_required` and
`simultaneous_barrier_passage_excluded` fields. Their public admission
keeps that reader's original half-weight, unit-beta, positive-half-contrast
scope, although the mathematical balances and ordering have the broader
positive-coefficient scope stated above. None of these necessary passage
fields changes the transfer verdict or records an observed future order.
Unsupported or missing fields remain unavailable, not zero supplied work.
The [mediated-pressure controls](../../tests/physics/test_relational_sine_mediated_pressure.py)
check channel signs and full-rate reuse; the
[formation controls](../../tests/physics/test_relational_sine_formation.py)
and [independent algebra controls](../../tests/physics/test_relational_sine_formation_barrier.py)
check the receiver balance, action factor and exact conditional thresholds.
No history integral, new solver or forced receiver experiment is evaluated.

<a id="sine-localized-receiver-exclusion"></a>
### A complete nonlinear prefix and dissipative tail can exclude receiver acquisition

Fix the original localized preparation, rather than selecting a new
profile after observing a response:

\[
x_{10}(0)=2,\qquad x_i(0)=0\ (i\ne10),\qquad
\theta_j(0)=2\pi j/5\ (0\le j\le4),
\]
\[
\theta_i(0)=0\ (5\le i\le10),\qquad
e=w=\frac12,\quad\beta=1,\quad\delta=\frac12.
\]

All eleven nodes, twelve edges and the original positive held capacities
remain present. In particular the receiver capacities at nodes 6 and 9
are `3/2` and `1/2`; the full port degrees include the bridges.
Here `F=4`, `N=32/3` and `E(0)=4+V_5`. No input, support event,
phase projection, prescribed mediator signal or replacement pressure
law is supplied. A tangent solution or a reduced reflection-symmetric
model does not certify this full nonlinear preparation.

The following is a **conditional proof rule** for a separately frozen
validated response. Its mathematical conclusion is not established by
the source or by naming a future horizon. Let `T>0` be that declared
horizon, and require a validated enclosure of the entire full-state
trajectory on `[0,T]` under the unchanged law.

#### The two enclosure obligations are different

For every accepted integration cell, its full nonlinear Picard tube must
enclose every intermediate state on that cell, not only its two endpoint
boxes. Evaluate the receiver's five-edge unit phase potential through
the shared circular interval kernels on each such tube, retaining all
edge-difference uncertainty. Let `M_R` be a certified upper bound over
the union of those tubes:

\[
V_R(t)\le M_R\quad\text{for every }0\le t\le T.
\]

Independently, let `U_E` be an outward upper bound on the **total**
storage at the validated endpoint, using every one of the twelve
edges and both fields in the full endpoint box:

\[
E(T)=\frac12\sum_{\{i,j\}}(x_i(T)-x_j(T))^2
 +\beta\sum_{\{i,j\}}[1-\cos(\theta_j(T)-\theta_i(T))]
 \le U_E.
\]

The endpoint expression cannot omit either bridge, replace the hidden
state by a midpoint or equilibrium, change the admitted held capacities,
or treat an independently propagated diagnostic as the actual full-state
storage. Correlated bounds can improve sharpness when justified; an
outward sum of actual edge enclosures already gives a valid sufficient
bound. The certificate requires the full frozen horizon to be reached.
A partial valid prefix is not a completed response.

The sufficient strict comparisons are

\[
\boxed{M_R<\frac72,\qquad U_E<\frac72\beta.}
\]

The fixed preparation has `beta=1`; the factor is displayed to make
clear the distinction between unit phase potential and scaled storage.
Neither equality is silently accepted as a strict safety margin.

#### Why these finite inequalities cover the whole future

The prefix bound supplies `V_R(t)<7/2` for all `0<=t<=T`.
For every later time, the exact full sine balance gives

\[
E(t)\le E(T)\le U_E<\frac72\beta,\qquad
0\le\beta V_R(t)\le E(t),\qquad t\ge T.
\]

The sine flow is globally defined under the retained finite-state,
fixed-support, positive-capacity premises. This tail argument uses its
storage dissipation with positive `e`, not a continuation of a numerical
stepper or an assumed relaxation rate. Therefore `V_R(t)<7/2`
for **every** future time, including the part not numerically integrated.
The continuous receiver phase path cannot leave its initial consensus
component of `{V_R<7/2}`. By the exact phase minimax proved above,
convergence to either maintained receiver twist is impossible.

There is a stronger conditional endpoint conclusion on this particular
support. The [complete asymptotic theorem](#sine-eleven-node-asymptotic-equilibria)
gives convergence to one full relative equilibrium. At that equilibrium
the bridge sine currents vanish and the receiver is a C5 critical
configuration. Its only critical configurations below `7/2` are
consensus and the two uniform twists. The uniform bound
`V_R(t)<=max(M_R,U_E/beta)<7/2` keeps the entire receiver path, and
its limit, in the initial consensus component; the two twist minima
belong to different components. Thus both strict checks also certify
**asymptotic receiver phase consensus**. The full form converges to
its conserved weighted mean by the same asymptotic theorem. This
does not select the donor geometry, bridge branches or common phase
origin, and does not assert zero wrapped receiver winding at every
intermediate time.

An endpoint with `E(T)<7*beta/2` alone would not suffice. The receiver
could already have crossed its barrier and entered a twist component
whose limiting value `beta*V_5` is below that same energy ceiling.
Conversely, a prefix safely below the receiver barrier does not control
its infinite future if the endpoint energy remains too high. The two
obligations cannot be substituted for one another.

#### Availability, failed bounds and actual passages remain distinct

If the complete prefix or its intermediate tubes are unavailable, there
is no all-time receiver exclusion from this rule. If the full prefix
is validated but an upper bound reaches or exceeds a threshold, that
comparison is inconclusive: it does not prove the actual trajectory
crossed the barrier. In particular, `M_R<7/2` together with
`U_E>=7*beta/2` leaves an unresolved late-time estimate. A solver
failure, an inconclusive enclosure and a demonstrated physical or
mathematical passage are different outcomes.

Only a completed response satisfying both strict tests certifies the
global exclusion. The protocol must retain the exact source, support,
law, clock, numerical budget and failed checks as well as successful
ones. Increasing a horizon or changing a preparation after seeing a
response is a separately declared evaluation, not completion of the
same frozen prediction. No finite receiver work record is invented by
this energy-tail route; it gives a sufficient upper bound on receiver
storage without replacing the exact boundary-work identity.

The [research reader](../../src/tnfr/research/relational_receiver_barrier.py)
keeps the preparation in `prepare_receiver_barrier`, consumes shared
full-state evidence through `assess_receiver_barrier_forecast`, and
evaluates the separately frozen response through `evaluate_receiver_barrier`.
Its `asymptotic_receiver_consensus_certified` flag requires both complete
prefix and strict endpoint checks, using the preceding conditional
corollary. Retained step coverage includes the exact held intermediary
capacity at every endpoint. A public forecast record still does not
authenticate its production; the frozen source and response retain that
provenance separately.

#### Evaluated result for the frozen localized preparation

The locally retained `artifacts/research/relational_receiver_barrier/`
contains `response-v1.protocol.json`, `response-v1.json` and
`response-v1.source.zip`; these evidence files are not distributed with the
documentation site. The protocol declared `T=32`, step `1/8`, Taylor order 16
and the shared outward dyadic-128 rational interval kernels before evaluation.
The retained response completed all 256 cells under that budget. Its exact rational bounds
give the following rounded displays:

\[
M_R\approx0.01848801719151926<0.018489<\frac72,
\]
\[
U_E\approx3.4768357092831135<3.476836<\frac72,\qquad
\frac72-U_E>0.023164.
\]

`M_R` is an upper bound on the entire validated prefix, not a measured
maximum and not the infinite-time upper bound. The later tail uses
`E(t)<=U_E`. Together these two certificates establish that this exact
localized source never reaches the receiver potential barrier and that
its receiver converges to phase consensus by the preceding corollary.
Neither receiver handedness target is reached asymptotically. The
proof does not select the donor's final geometry, a bridge branch or a
time at which wrapped winding must be zero.

The protocol SHA-256 is
`16f90bbfb534f684e7f80c36ab45dbe37f879ffc703c3160b660232346ceb90c`;
the archived producer source has SHA-256
`e334410abe3ccce14e96d5cd8d4a4a079a05bf3796ae8931d7eff5747c251a94`.
The response retains the actual full-state tubes, endpoint bounds,
declaration, runtime and source manifest. No shortened horizon, changed
preparation, reduced law or repeat response was substituted for the
frozen test. This is one source-specific nonlinear exclusion, not an
exclusion of every six-coordinate preparation with `F<=4` or of TNFR
pattern formation under other complete laws.

A later reader-admission correction rejects Boolean capacity values in
manually supplied forecast records. The actual frozen preparation uses exact
rational capacities and its checked enclosures are unchanged; the archived
producer and evaluated response were not regenerated for that correction.

#### Exact relative coordinates do not replace failed numerical evidence

If independent interval boxes lose useful correlations, a different
coordinate representation can be considered without changing the law.
Only the common form translation and common phase shift are removed
here. With reference node `h=10`, put

\[
u_i=x_i-x_h,\qquad v_i=\theta_i-\theta_h\quad(i\ne h),
\qquad u_h=v_h=0.
\]

For continuous phase lifts, the following twenty-coordinate system is
exact and globally closed:

\[
\begin{aligned}
q_i(u)&=\sum_{j\sim i}(u_i-u_j),&
S_i(v)&=\sum_{j\sim i}\sin(v_j-v_i),\\
\dot u_i&=-e(k_iq_i-k_hq_h)+a(k_iS_i-k_hS_h),&
\dot v_i&=b(k_iq_i-k_hq_h),\qquad i\ne h,
\end{aligned}
\]

where `k_i=nu_i/d_i`, `a=w/pi` and `b=w/(beta*pi)` retain
the actual support and held capacities. These are differences of the
complete fine rows, including loss; they are not a reflection-symmetric
approximation or a closure obtained by removing the intermediary.
They reuse the [reference-node quotient construction](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-mobility-relative-geometry),
with the present constant-mobility law's conserved means.

To reconstruct discarded origins, let `rho_i=1/k_i`,
`M=sum_i rho_i`, and retain the two initial invariants
`I_x=sum_i rho_i*x_i`, `I_theta=sum_i rho_i*theta_i` in their
declared continuous lift. Then

\[
x_h=\frac{I_x-\sum_{i\ne h}\rho_i u_i}{M},\qquad
\theta_h=\frac{I_\theta-\sum_{i\ne h}\rho_i v_i}{M},
\]

followed by `x_i=x_h+u_i`, `theta_i=theta_h+v_i`. Arbitrarily
wrapping each `v_i` discards the integer-lift information needed for
this real-phase reconstruction. Total storage and receiver phase
potential themselves depend only on the relative edge differences.
The held capacity is a parameter in these equations, not an additional
evolving coordinate.

This exact quotient can remove uncertainty in common origins, but it
does not guarantee a narrower interval enclosure: its coupled rate
differences can still produce box overestimation. The original
preparation supplies no donor reflection symmetry that removes further
constituent coordinates. A producer using this representation would
need its own declared layout, transformed initial enclosure and frozen
numerical budget. It cannot reinterpret an unavailable twenty-three-
coordinate prefix as a successful evaluation or replace the unchanged
whole-time and endpoint obligations above. The successful frozen result
uses the original full-state representation and does not require this
alternative numerical layout.

### Boundary of this result

The shared formation owner admits explicit donor form, computes
these full-support quadratic forms and derivative evidence,
and retains the supplied preparation and window. Its general
majorant is a sufficient analytic bound, with explicit numerical
availability and a required positive norm polynomial. An
unavailable bound is not a failed trajectory.

Its separate `maintained_target_obstruction` applies the auxiliary
coexistence certificate only under `beta=1`, `delta=1/2` and the exact
stored-coefficient condition `e>0`, `0<w/e<3/2`.
Its `exchange_to_loss_ratio` and
`sufficient_ratio_upper_bound` expose this admission without treating the
upper endpoint as a critical law constant. The mixed coefficient is
`h=w/(2*pi*e)`, not a separately configured physical parameter.
It evaluates the correlated margin `V_5-(4/5)*F`, rather than subtracting
independently widened initial and target intervals. A positive lower margin
certifies maintained-target exclusion; an unavailable or unresolved margin
does not certify formation. The phase-action and early-loss readers retain
their different transient-entry conclusions. Tests of these owners verify
the matrix premises and report boundaries; they do not replace the
global derivative argument. The
[independent whole-support algebra controls](../../tests/physics/test_relational_sine_formation_barrier.py)
differentiate the actual fine field, verify the path and spectral premises,
and check the uniform endpoint separation, ratio dependence and complete
constant-clock transformation. The
[formation-report controls](../../tests/physics/test_relational_sine_formation.py)
cover action admission, effective-coefficient model guards, unavailable
bounds and correlated maintained-target margins.

The entire six-coordinate class with `F<=4` is now excluded from
the specified maintained two-twist coexistence target throughout the stated
positive coefficient ratio domain and fixed capacities, by the joint
auxiliary-function proof above. The earlier constant-donor, silent-subspace and individual-profile
results remain
valid with their own stronger or different scopes, including their
entry-time statements. Independent ranges of `N/F`, `R` and `gamma`
were not treated as jointly attained or substituted into this proof.

That coexistence certificate does not select the loss law, alter the
preparation budget or transfer to another capacity contrast or constitutive model.
Transient joint acute entry and receiver-only transfer retain the separate
obligations stated above. No successful pattern formation, optimal profile,
autonomous substrate or physical identification follows from this
negative maintained-formation result.

<a id="sine-eleven-node-asymptotic-equilibria"></a>
### Every positive-loss trajectory on this support converges to one relative equilibrium

This conclusion concerns the same connected unit graph consisting of
cycles `(0,1,2,3,4)` and `(5,6,7,8,9)` and bridges `0--10`, `5--10`.
It needs no special preparation, form budget, initial winding or acute
phase premise. Let **every held capacity be strictly positive** and
`e,w,beta>0`, with no input, clipping, event or support change. Use the
same complete normalized-sine rows

\[
\dot x=-eKq+aKS,\qquad \dot\theta=bKq,\qquad
q=Lx,\quad K=\operatorname{diag}(\nu_i/d_i),\quad
a=w/\pi,\quad b=w/(\beta\pi).
\]

The result therefore includes the six-coordinate class above, but is
not restricted to its capacities or half-weight coefficients. Only one
common phase origin is removed when comparing relative geometries.
Node labels, constituent phases and the two bridge gaps are retained.

#### Reuse compactness and approach to the equilibrium set

The
[global sine-law argument](RESONANCE_FOUNDATIONS.md#the-sine-obstruction-also-controls-nonperiodic-long-time-motion)
already gives global continuation and approach to the equilibrium set.
Its compactness hypotheses hold here: `E` is nonincreasing and
nonnegative, so it bounds all form differences on the connected graph;
the conserved weighted form mean bounds the common form origin, and
circular phase lies on a compact torus. Thus every finite initial state
has a forward orbit with compact closure in form times circular phase.

With strictly positive `K`, the largest invariant zero-loss subset is
exactly `q=S=0`. Indeed, zero loss forces `q=0`. Remaining there requires
`LK S=0`, hence `KS=c*1`; reciprocity `sum_i S_i=0` and positivity of
`K` imply `c=0`. Form is therefore uniform at every equilibrium, fixed
by the source's conserved weighted form mean. What remains to strengthen
the earlier limit-set theorem is the full **relative** phase critical set,
including every nonacute branch.

#### Critical bridge currents vanish, rather than being discarded

For either bridge, remove that edge and sum `S_i=0` over one resulting
connected component. Internal edge sine terms cancel pairwise. The only
remaining term is the bridge sine current, which must consequently be
zero. Its phase difference is therefore either `0` or `pi` modulo
`2*pi`; the antipodal branch must not be removed by an acute admission
rule.

Fix intermediary phase zero. The donor and receiver port phases may
each independently be `0` or `pi`, giving four relative bridge choices.
This does not allow an arbitrary independent rotation of either ring:
its port is fixed by its bridge choice relative to the common intermediary.

#### An odd cycle has finitely many complete sine-critical branches

Orient either C5 and let `eta_j` be the phase increment from its
`j`th node to its next node, modulo `2*pi`. At a nonport ring node,
`S_i=0` equates the two oriented ring sine currents. At the port the
bridge current has already been proved zero, so the same equality holds.
Thus all five `sin(eta_j)` equal one value `s`.

Choose its principal inverse-sine representative
`alpha=arcsin(s)` in `[-pi/2,pi/2]`. Every edge increment is then
either `alpha` or `pi-alpha` modulo `2*pi`. If `k` of the five edges
take the second branch, circular closure gives

\[
(5-2k)\alpha+k\pi=2\pi m,\qquad m\in\mathbb Z.
\]

The two branch endpoints do not supply exceptional continuous families.
If `alpha=pi/2`, both edge choices coincide at `pi/2` modulo `2*pi`,
whose fivefold sum does not close. At `alpha=-pi/2` the same argument
uses `-pi/2`. Therefore every actual critical solution has
`|alpha|<pi/2`. Since `5-2k` never vanishes,

\[
\boxed{\alpha=\pi\frac{2m-k}{5-2k},\qquad
\left|\frac{2m-k}{5-2k}\right|<\frac12.}
\]

For each of the finitely many edge masks, this strict inequality admits
only finitely many integers `m`. In exact turn coordinates, a constructive
classification uses base increment

\[
c=\frac{\alpha}{2\pi}=\frac{2m-k}{2(5-2k)},\qquad |c|<\frac14,
\]

with edge turns `c` off the mask and `1/2-c` on it, reduced modulo one
when reconstructing nodal phases. Starting from port phase zero, their
successive sums determine all ring phases and the integer closure
condition determines the closing edge. Every listed state has equal
oriented sine currents, so the construction is sufficient as well as
necessary.

There is no branch duplication hidden by the mask. Because
`cos(alpha)>0`, the mask is precisely the set of negative-cosine edges,
and the common sine current fixes its principal `alpha` uniquely.
The case `alpha=0` is included: its edges are zero or antipodal, and
closure requires an even number of antipodal edges. Cyclic permutations
of distinct masks remain distinct **labeled** geometries; this classification
does not quotient graph automorphisms.

The exact integer possibilities and resulting counts are:

| Negative-cosine edges `k` | Integers `m` | Principal base turns `c` | Labeled C5 geometries |
| --- | --- | --- | --- |
| 0 | `-1,0,1` | `-1/5,0,1/5` | `3` |
| 1 | `0,1` | `-1/6,1/6` | `2*binomial(5,1)=10` |
| 2 | `1` | `0` | `binomial(5,2)=10` |
| 3 | none | none | `0` |
| 4 | `2` | `0` | `binomial(5,4)=5` |
| 5 | `2,3` | `1/10,-1/10` | `2` |

There are exactly `30` relative phase geometries on each labeled ring.
Combining both independent ring choices with the four bridge choices
gives **`4*30^2=3600` relative phase-critical geometries** on the entire
eleven-node support. One canonical representative fixes intermediary
turn zero and records every other turn in `[0,1)`. Uniform form with
any common value completes an equilibrium. On the conserved form-mean
leaf of a particular trajectory, that value is already fixed.

This finite classification keeps critical states with negative cosines,
antipodal bridges and nonzero winding beyond the acute class. Enumeration
alone does not classify their stability or basins. The separate
[inertia argument below](#sine-eleven-node-equilibrium-stability)
settles local stability using the complete law.

#### Connectedness of the limit set gives one relative endpoint

Pass to real form together with phase relative to the intermediary,
`exp(i*(theta_i-theta_10))`. This is a continuous quotient by the one
common circular origin. The orbit is still precompact. For each `T`,
the closure of its connected forward tail `t>=T` is compact and
connected. These nonempty tail closures are nested, so their intersection,
the omega-limit set, is nonempty, compact and connected.

The earlier dissipation theorem puts that limit set inside `q=S=0`.
On the fixed weighted form-mean leaf, the classification just proved
leaves only `3600` possible relative equilibria. A connected subset of
a finite set consists of one point. Precompactness then implies
convergence to that point: otherwise a sequence remaining a positive
distance away would have a different omega-limit point. Thus

\[
\boxed{x(t)\longrightarrow M_x\mathbf1,\qquad
e^{i(\theta_i(t)-\theta_{10}(t))}
\longrightarrow e^{i\theta_i^*}
\quad\text{for one classified geometry }\theta^*.}
\]

The identity of `theta*` is not selected by this argument. In particular,
convergence to some equilibrium is not convergence to the receiver-transfer
target, nor evidence that the limiting equilibrium attracts an open set.
A source initially at any unstable equilibrium is also covered.

#### Conserved lifted phase reconstructs the final common origin

Relative convergence alone would not rule out continuing common rotation.
Here the independently proved weighted phase-lift invariant supplies the
missing reconstruction. Choose any continuous real lift of the actual
phase trajectory, and let

\[
I_\theta=\sum_i\omega_i\theta_i(t),\qquad
\omega_i=1/k_i,\qquad M_\omega=\sum_i\omega_i>0.
\]

Fix the limiting representative with `theta_10^*=0`. For sufficiently
large time every relative phase error is inside one strict circular
chart about zero. Its unique small real representative `epsilon_i(t)`
is continuous and tends to zero, with `epsilon_10=0`. In that chart

\[
\theta_i(t)=\theta_{10}(t)+\theta_i^*+2\pi n_i+
\epsilon_i(t).
\]

The integer vector `n` is fixed for all these later times: an integer
difference of continuous lifts cannot change while the chart remains
admitted. Choose `n_10=0`. Conservation now gives

\[
\theta_{10}(t)=\frac{I_\theta-
\sum_i\omega_i(\theta_i^*+2\pi n_i+\epsilon_i(t))}{M_\omega}
\longrightarrow
\gamma_\infty:=\frac{I_\theta-
\sum_i\omega_i(\theta_i^*+2\pi n_i)}{M_\omega}.
\]

Consequently the chosen continuous lifts themselves converge to finite
limits `gamma_infinity+theta_i^*+2*pi*n_i`, and the circular phases
converge to the corresponding full equilibrium. The integer vector
retains the actual trajectory's prior turns; the exact classifier does
not predict it. No weighted phase mean has been promoted to a
single-valued function on the entire torus.

#### Scope of the stronger long-time conclusion

This result applies to every finite initial state on this fixed support,
not only the earlier six-coordinate preparation. It proves existence of
one limiting geometry, not its selection, a convergence rate, a finite
arrival time or successful identity transfer. The receiver-transfer
accessibility question remains separate even though unclassified
persistent motion is no longer an alternative long-time outcome in
this positive-loss model.

Strictly positive capacities, positive `e,w,beta` and the unchanged sine
law are essential stated premises. Zero capacities retain additional
frozen data; the conservative `e=0` pulse and recurrence theorems are
different results. The odd-cycle closure argument does not extend by
renaming the graph: on an even cycle the coefficient `n-2k` can vanish,
and continuous critical families can remain. Native argument-pressure
dynamics also retains its separate regularity obligations. For example,
flat donor phases zero, flat receiver phases `pi` and intermediary phase
zero define an exact sine equilibrium with uniform form, but the
intermediary's native neighbor resultant is `1+(-1)=0`. That catalog
member is outside the native argument law's regular domain. No theorem
about arbitrary TNFR runtimes, support birth or physical constituents
follows from this classification.

The shared
[`CircularPhaseState` and `reconstruct_circular_phase_state`](../../src/tnfr/physics/phase_cycle_geometry.py)
retain the full circular geometry, including nonacute and antipodal edges,
without weakening the existing acute recovery contract.
`classify_c5_sine_critical_set(geometry, cycles=...)` uses that owner to
construct the exact factored `C5SineCriticalSet`; no numerical root
search or independent phase catalog is required. The separate
[`assess_sine_asymptotic_equilibria`](../../src/tnfr/physics/relational_sine_equilibria.py)
reader, also available as a sine comparison's
`asymptotic_equilibria(cycles=...)`, returns `SineAsymptoticEquilibria`
with the complete-law premises checked against its captured source.
An exact phase classification and the availability of this asymptotic
theorem are distinct from identifying the source's future endpoint.
The reader does not evolve the state or choose a basin.

The
[independent exact controls](../../tests/physics/test_relational_sine_equilibria.py)
check full-support criticality, branch closure and completeness, including
the nonacute controls. The
[observation controls](../../tests/physics/test_relational_sine_equilibria_observation.py)
check circular reconstruction, labeling and the law-admission boundary.
These finite tests exercise the classification and its engine integration;
the infinite-time conclusion follows from the compactness and connected
limit-set argument above, not a sampled convergence trace.

<a id="sine-eleven-node-equilibrium-stability"></a>
### Exact local stability of every classified equilibrium

The [general bridge-tree composition theorem](SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition)
now owns the extension to arbitrary connected components and the full-law
inertia argument on connected support. The calculation below retains the
exact C5 component classification and this support's specialized counts.

Keep the complete eleven-node sine law, fixed unit support, all strictly
positive held capacities and `e,w,beta>0` of the preceding theorem.
Choose any of its exact critical phase geometries and uniform form.
No restriction to acute edges or to either ring's reflection subspace
is imposed. Write `H` for the full cosine-weighted phase Hessian, with
quadratic form

\[
v^\mathsf THv=\sum_{\{i,j\}}\cos(\theta_j^*-\theta_i^*)
                         (v_j-v_i)^2.
\]

Inertia means the numbers of positive, negative and zero directions,
in that order. The geometric calculation first removes the one common
phase direction; the subsequent dynamics also removes the independently
conserved common form coordinate.

#### Ring constraints and bridge directions determine the inertia exactly

For one ring, let `alpha` and its supplementary-edge mask be those of
the critical classification, and let `k` be the mask size. Every edge
cosine has magnitude `c_alpha=cos(alpha)>0`; its sign is positive off
the mask and negative on it. If `z_j` are the five oriented edge
variations of an actual nodal phase perturbation, they satisfy
`sum_j z_j=0`. Conversely every such vector is produced by a ring
perturbation, uniquely modulo its common origin. The ring quadratic form
is therefore the restriction of

\[
D=c_\alpha\operatorname{diag}(\sigma_0,\ldots,\sigma_4),
\qquad \sigma_j\in\{1,-1\},
\]

to `1^perp`. The full edge form has inertia `(5-k,k,0)`. Its
`D`-orthogonal complement to `1^perp` is spanned by `D^(-1)*1`, since
`z^T D(D^(-1)*1)=z^T*1=0`, and that complementary direction has value

\[
\mathbf1^\mathsf TD^{-1}\mathbf1=\frac{5-2k}{c_\alpha}\ne0.
\]

Thus the restriction is nondegenerate: remove one positive direction
when `k<5/2`, or one negative direction when `k>5/2`. This treats the
cycle constraint explicitly; the sign of a single edge alone would not
justify the conclusion on an arbitrary graph.

| Allowed mask size `k` | C5 relative inertia `(positive, negative, zero)` | Number of labeled ring branches |
| --- | --- | --- |
| 0 | `(4,0,0)` | `3` |
| 1 | `(3,1,0)` | `10` |
| 2 | `(2,2,0)` | `10` |
| 4 | `(1,3,0)` | `5` |
| 5 | `(0,4,0)` | `2` |

On the full graph, edge variations satisfy exactly the two ring-sum
constraints. The two bridge variations are independent: the incidence
map from nodal phases modulo one common origin is an isomorphism onto
this ten-dimensional constrained edge space. Consequently the Hessian
form is a direct sum of both constrained ring forms and the two scalar
bridge terms. Each zero-phase bridge contributes one positive direction;
each antipodal bridge contributes one negative direction.

Let `j_D,j_R` be the two negative ring indices in the table, and let
`n_pi` be the number of antipodal bridges. Then

\[
\boxed{j=j_D+j_R+n_\pi,\qquad
\operatorname{inertia}(H_{\rm relative})=(10-j,j,0).}
\]

The unrestricted eleven-dimensional phase Hessian has inertia
`(10-j,j,1)`; its sole kernel is the common phase direction. No other
zero direction occurs in any of the 3,600 branches. In particular, the
two bridge choices cannot be discarded before testing stability.

#### The full reciprocal Jacobian, rather than phase gradient descent

The
[existing complete sine derivative](RESONANCE_FOUNDATIONS.md#resonance-tangent)
and [stiffness argument](RELATIONAL_EXCHANGE_ADMISSION.md#regular-equilibrium-stiffness)
apply to both consumed rows. To make their use explicit, introduce

\[
\xi=K^{-1/2}\delta x,\qquad \eta=K^{-1/2}\delta\theta,
\qquad h=K^{-1/2}\mathbf1,
\]
\[
B=K^{1/2}LK^{1/2},\qquad C=K^{1/2}HK^{1/2}.
\]

Fixing the two conserved weighted means in a local phase lift is exactly
`xi,eta in h^perp`. Both `B` and `C` preserve this subspace. On it,
`B>0`, and `C` has the relative Hessian inertia just computed: the
invertible capacity rescaling identifies this mean-fixed complement
with the nodal phase quotient. In particular `C` is nonsingular.
The exact full tangent rows on this twenty-dimensional space are

\[
\dot\xi=-eB\xi-aC\eta,\qquad
\dot\eta=bB\xi.
\]

Eliminating `xi` and writing `eta=B^(1/2)*z` gives

\[
\ddot z+eB\dot z+A z=0,\qquad
A=abB^{1/2}CB^{1/2},
\]

whose symmetric stiffness `A` has the same inertia as `C`. This is
an exact rewriting of the linearized two-row law, not a new inertial
constitutive assumption. Its quadratic eigenvalue pencil is

\[
P(s)=s^2I+esB+A.
\]

For a possibly complex eigenvector `z!=0`, multiply `P(s)z=0` by
`z^*`. Writing `s=sigma+i*omega`, the imaginary part is

\[
\omega\left(2\sigma\|z\|^2+e\,z^*Bz\right)=0.
\]

Every nonreal eigenvalue therefore has strictly negative real part.
There are no nonzero imaginary eigenvalues; `s=0` is also excluded
because `A` is nonsingular. For real `s>=0`,
`P'(s)=2*s*I+e*B>0`, so its ordered real eigenvalues increase strictly.
At zero there are exactly `j` negative eigenvalues, and for sufficiently
large `s` the entire matrix is positive. Hence exactly `j` eigenvalues
cross zero at positive real values, counted with multiplicity. At each
crossing the derivative restricted to its kernel is positive definite;
eliminating the invertible complementary block gives a first-order
positive term on that kernel. Thus the zero's determinant multiplicity
equals the kernel dimension, with no uncounted tangential crossings.

The full tangent consequently has exactly `j` positive real eigenvalues,
`20-j` eigenvalues with negative real part, and no relative center modes:

\[
\boxed{\dim E^{\rm unstable}=j,\qquad
\dim E^{\rm stable}=20-j,\qquad
\dim E^{\rm center}_{\rm relative}=0.}
\]

This also rules out an instability hidden in a complex right-half-plane
pair. In unrestricted form/phase coordinates the two common-origin
directions remain neutral. They are the conserved-coordinate freedoms,
not unresolved relative degeneracies. Fixing their weighted means gives
the tangent space just analyzed; perturbing those means changes the
equilibrium's eventual common origins.

#### Nine local attractors, with every other relative branch unstable

The condition `j=0` requires both ring masks to have `k=0` and both
bridge phase gaps to be zero. Each ring can then have exactly one of
the three uniform principal increments `0`, `2*pi/5` or `-2*pi/5`.
There are therefore **nine locally exponentially attracting relative
equilibria**, indexed by the pair of ring windings

\[
(w_D,w_R)\in\{-1,0,1\}^2,
\]

with aligned ports and intermediary. Smoothness and the strictly stable
full relative Jacobian give nonlinear local exponential attraction on
each conserved-mean leaf, or attraction to the corresponding common-origin
orbit when those two means are allowed to vary. The
[whole-support recovery theorem](#sine-interacting-recovery) provides
explicit sufficient neighborhoods for these acute critical geometries.
It does not certify a distant preparation's entry into them.

Every other branch has `j>=1`, hence a positive real eigenvalue and
nonlinear instability. All **3,591 remaining relative equilibria are
hyperbolic and unstable**; none has an unclassified relative zero mode.
The dimensions follow from the structural inertia formula, without
numerically diagonalizing 3,600 Jacobians.

Strictly positive changes in held capacities or `e,w,beta` alter
eigenvalues, response rates and potentially basins, but not this local
stability classification on the fixed support. They cannot produce a
local Hopf or zero-mode crossing in this equilibrium family while the
stated positivity premises hold. This is not a claim that global basins
cannot change. In particular, the auxiliary-function proof's sufficient
ratio cutoff `w/e<3/2` is not a detected local bifurcation threshold.

#### Stable composition retains causal interaction and its observations

The result admits an independently chosen consensus or either handed
twist on each ring as one stable **full-network** geometry. Its proof
retains perturbations of every node and both bridges; it does not turn
the regions into dynamically independent copies. The actual matrices
`B` and `C` need not commute, and the existing
[mediated response](RESONANCE_FOUNDATIONS.md#mediated-resonance)
uses this same retained coupling and intermediary. Geometry can preserve
the local identities while permitting a response between them.

That response owner already covers any acute critical target with zero
bridge gaps. The present classification shows this is exactly the nine-member
attracting family on the stated support. At fixed capacities and coefficients,
reversing a ring's twist leaves all its edge cosines, the complete tangent
and every linear transfer unchanged. The nine phase geometries therefore
give four distinct tangent geometries, determined by whether each ring
is flat or twisted. Linear response alone does not recover handedness.
This reuses a proved observation limitation; it does not select a new
controller, onset law or physical identification.

#### The original source budget excludes all four attracting coexistence states

Return only for this corollary to the original six-coordinate source,
`F<=4`, `beta=1`, `delta=1/2` and the proved ratio domain `0<w/e<3/2`.
Each of the four attracting states with both ring windings nonzero has
uniform form, `q=S=0`, and `V=2V_5`, regardless of either handedness.
The [same monotone functional](#sine-maintained-target-obstruction)
therefore gives the unchanged gap

\[
\mathcal W_{\rm coexistence}-\mathcal W(0)
=V_5-\frac45F>\frac14.
\]

All four attracting two-twist endpoints are excluded for this source
class, not only the original `(+1,+1)` target. Among locally attracting
geometries, exactly five remain compatible with that obstruction:

\[
(0,0),\qquad (+1,0),\quad(-1,0),\qquad
(0,+1),\quad(0,-1).
\]

These are remaining candidates, not five demonstrated outcomes. The
classification does not exclude convergence along a stable manifold to
an unstable equilibrium, and the six-dimensional source slice need not
inherit an ambient almost-everywhere statement. Nor do an initial odd
response or an available receiver-only basin choose a terminal winding.
Receiver-transfer accessibility remains the separate missing dynamical
argument identified above.

#### Shared geometry and law readers retain different claims

The exact catalog's `phase_hessian_inertia(...)` method retains the
selected branch's constrained geometric inertia;
`phase_hessian_index_counts` aggregates the factorized catalog without
an equilibrium eigenvalue scan. These geometry readers belong to
[`phase_cycle_geometry.py`](../../src/tnfr/physics/phase_cycle_geometry.py).
The law report `SineAsymptoticEquilibria.classify_equilibrium(...)`
revalidates the consumed source and catalog before returning
`SineEquilibriumStability` from
[`relational_sine_equilibria.py`](../../src/tnfr/physics/relational_sine_equilibria.py).
Its relative mode dimensions and nonlinear stability flags require the
strictly positive law and capacity premises proved here. An available
geometric inertia under unsupported dynamics supplies no such flags.
The reader classifies a declared equilibrium; it does not predict which
branch a particular source reaches or install a different evolution law.

[Independent algebra and complete-Jacobian controls](../../tests/physics/test_relational_sine_equilibria.py)
check the constrained inertia and its full-law connection, including
noncommuting matrices. The
[observation controls](../../tests/physics/test_relational_sine_equilibria_observation.py)
exercise the catalog, source admission and unavailable domains. These
finite controls support the implementation; the proof above supplies
the all-branch and positive-parameter quantifiers.

<a id="sine-donor-well-retention"></a>
### A subcritical auxiliary well selects the original donor endpoint

Retain the same eleven-node support, held capacities with `delta=1/2`,
`beta=1`, donor winding `+1`, flat receiver and aligned bridges. The
initial form remains the six-coordinate donor/intermediary preparation
of Section 22. Use the proved coefficient domain `e>0`,
`0<w/e<3/2`; it includes the original `e=w=1/2` without a parameter
survey or another constitutive law. Let `F` be its actual initial form
storage, and define

\[
F_c=\frac54\left(\frac72-V_5\right)
   =\frac{25\sqrt5-55}{16}>0.
\]

Then

\[
\boxed{0\le F\le F_c
\quad\Longrightarrow\quad
\text{relative equilibrium limit }(w_D,w_R)=(+1,0).}
\]

This determines a terminal relative identity, not just failure to reach
one proposed target. It supplies no finite convergence time or claim
that either ring remains in an acute or fixed-winding chart at every
intermediate time. Its proof uses the full coupled law, including the
intermediary and receiver capacity contrast.

#### Properness and critical points of the existing auxiliary function

Work on the fixed weighted form-mean leaf and quotient only the common
circular phase origin. The form part has dimension ten; the phase part
is a compact ten-dimensional torus. Common lifted-phase means determine
the final representative as in the asymptotic theorem, but are not a
globally defined function on this torus quotient.

Reuse exactly the function already proved to decrease strictly:

\[
\mathcal W=\frac45\mathcal F+V-hq^\mathsf TKS,
\qquad h=\frac{w}{2\pi e},\qquad
\dot{\mathcal W}<0\quad\text{if }(q,S)\ne(0,0).
\]

On the fixed mean leaf, the positive graph gap makes `mathcal F`
coercive in relative form. The sine vector is uniformly bounded, while
`q=Lx` is linear in that form. Thus for some positive constants
`A,C`, independent of phase, `W>=A*||u||^2-C*||u||`, where `u` is
any fixed linear coordinate on this form leaf. Consequently `W` is
bounded below and every sublevel is compact. Keeping the common form
translation free would destroy this properness premise.

A critical point of `W` on this quotient has zero derivative along
the actual flow. Strict decrease therefore forces `q=S=0`.
Conversely, at `q=S=0` every derivative of `W` vanishes: form is
uniform, `V` is critical and both factors of the mixed term vanish.
The critical set is exactly the previously classified equilibrium set,
and its critical values are exactly `V` at those geometries.

The single-C5 critical values supplied by the exact catalog are

\[
0,\quad V_5,\quad \frac72,\quad 4,\quad 8,\quad
\frac{25+5\sqrt5}{4}.
\]

Each bridge adds zero or two. Hence the only full-support critical
values strictly below `7/2` are `0`, `2` and `V_5`. In particular,
there is **no critical value in `(V_5,7/2)`**. The four critical
geometries at `V_5` are the two donor-only twists and the two
receiver-only twists; each is a locally attracting relative equilibrium.

Each is also an isolated strict local minimum of `W`. To see this
without assuming that `W` is the physical energy, take any nearby
nonequilibrium state in its local attracting neighborhood. Its forward
orbit converges to that equilibrium, while `W` decreases strictly
before taking the limiting value `V_5`. Its initial `W` is therefore
strictly greater than `V_5`. The local attraction and strict-decrease
results have compatible coefficients and the same retained state.

#### Different one-twist wells cannot connect below `7/2`

Around the donor-only minimum choose a small closed coordinate ball
that excludes every other critical point. Its boundary has minimum
`W` strictly above `V_5`. For a level `r_0` between `V_5` and that
boundary minimum, also chosen below `7/2`, the donor's component of
`{W<r_0}` stays inside the ball and contains only the donor critical
point.

This component cannot merge with another component as the level rises
to any `r<7/2`. Indeed a closed strip `r_0<=W<=r` is compact and
has no critical point. The auxiliary vector field
`-grad(W)/||grad(W)||^2` lowers `W` at unit rate there; its finite
flow, with a cutoff outside the strip, deforms the higher sublevel to
the lower one without changing its components. This is a topological
proof device, not a replacement for the TNFR dynamics. It shows that
for every `V_5<r<7/2`, the donor component contains no other critical
geometry, including the lower-valued bridge saddles or consensus.

In particular every continuous path from that donor minimum to either
receiver-only minimum must have `max W>=7/2`. The existing path that
first flattens the donor and then twists the receiver, at uniform form,
attains `max W=7/2`. Thus the static minimax is exactly

\[
\inf_{\gamma:D_+\to R_\pm}\ \max_s\mathcal W(\gamma(s))
=\frac72.
\]

The negative receiver orientation uses the reflected second phase leg.
The same reasoning restricted to an isolated donor phase torus gives
its twist-to-flat potential minimax `7/2`; a winding-seam cost alone
does not prove this barrier. The critical-set and component argument
is what supplies the previously missing path statement.

#### The actual preparation enters and remains in the donor component

Initially `S=0`, so `W(0)=V_5+(4/5)*F`. At the fixed initial phases,
contracting form toward its conserved uniform mean gives
`W=V_5+(4/5)*s^2*F` for `0<=s<=1`. If `F<F_c`, this entire path
lies below `7/2`. Choose a regular level `r` strictly between
`W(0)` and `7/2` (and above `V_5`); the actual initial state is
in the donor component of `{W<r}`. Nonincrease of `W` and continuity
keep its actual orbit in that component.

The equality `F=F_c` needs a separate argument; replacing a closed
inequality by a strict numerical tolerance would not suffice. Here
`F>0`, hence `q(0)!=0` on the connected support. Strict decrease
gives `W(t)<7/2` for all sufficiently small positive times.
At those same times `V(theta(t))<7/2`, by continuity from `V_5`.
For each fixed phase, `W` is a convex quadratic in form: its mixed
term is linear in form. The straight form segment from `x(t)` to
the conserved uniform mean consequently lies below `7/2`, since
both endpoint values, `W(t)` and `V(theta(t))`, do. At uniform form,
a short path in a sufficiently small phase-chart neighborhood joins
`theta(t)` to the initial donor phase; throughout it `W=V<7/2`.
These two compact paths place the actual state at time `t` in the
same donor component at some level `r<7/2`. This resolves the
boundary without assigning a new preparation or integrating a response.

Finally the already proved global asymptotic theorem gives convergence
to one relative critical geometry. The forward tail stays in a compact
sublevel of the retained donor component, whose only critical point is
the original donor-only minimum. That point must be its limit. At
`F=0` the original state itself is an equilibrium, consistently with
the conclusion. The conserved means set the terminal uniform form and
lifted phase representative; they do not change its relative geometry.

#### A stronger preparation control, without a claimed dynamical threshold

The theorem excludes convergence to either receiver-only attractor and
entry into any valid full-state recovery basin for either one. It is
stronger than the earlier acute-face budget or the bare energy barrier:
the latter gives only the necessary transfer condition
`F>7/2-V_5`, using strict initial energy loss before donor unwinding.
The auxiliary-well condition gives instead `F>F_c` as a necessary
condition for transfer.

For a rational control, set every donor form to zero and only the
intermediary form to `H=7/30`. Then `F=49/900` and

\[
\frac72-V_5<\frac{49}{900}<F_c.
\]

This nonsilent preparation has `E(0)>7/2`, yet `W(0)<7/2` and
therefore returns to the original donor identity. These are exact
algebraic inequalities, not an evaluated trajectory. For example the
left inequality follows from `sqrt(5)<56/25` and `1/20<49/900`;
the right follows
from `(16*(49/900)+55)^2<3125`.

For any admitted `F>=0`, the exact retention condition is equivalently

\[
\boxed{(16F+55)^2\le3125.}
\]

Both sides of the unsquared inequality `16F+55<=25*sqrt(5)` are
positive, so squaring introduces no extraneous branch. This permits
represented rational inputs to use the shared exact arithmetic rather
than a rounded decimal threshold or a widened difference of intervals.

The threshold is exact for this **static sublevel separation**: above
it the existing comparison path connects the source and target inside
`W<=W(0)`. It is not a demonstrated dynamical transition at `F_c`.
At equality the actual trajectory has already lost auxiliary value
before it could cross; above it, silence, further losses and the actual
flow direction can still prevent transfer. The rest of the original
`F<=4` class retains its unresolved accessibility obligation.

The shared formation report's
[`donor_well_retention()`](../../src/tnfr/physics/relational_sine_formation.py)
returns `SineDonorWellRetention` from the same revalidated exact
preparation. Its `exact_retention_polynomial_margin` controls admission;
the displayed critical-storage and escape-margin intervals do not.
`relative_donor_pattern_convergence_certified` and
`receiver_only_targets_excluded` state the two proved consequences.
Unsupported law premises remain `unavailable`; a valid source outside
this sufficient interval is `not_certified`, with no positive transfer
verdict. The half-weight receiver-transfer reader retains this result
separately from its acute-entry/action checks. The
[formation controls](../../tests/physics/test_relational_sine_formation.py)
and [independent algebra controls](../../tests/physics/test_relational_sine_formation_barrier.py)
exercise the exact threshold and the distinct source-to-basin conclusion;
neither evaluates a trajectory or infers a finite recovery time.

<a id="sine-donor-dissipative-capture"></a>
### Early dissipation captures an above-threshold preparation in the donor well

Return to the original `e=w=1/2`, `beta=1`, `delta=1/2` and structural
clock. Keep the same six initial form coordinates, exact donor twist,
flat receiver, aligned intermediary, complete support and held capacities.
No input, event, clipping or changed law is supplied. Write

\[
F=\mathcal F(0),\qquad N=q(0)^\mathsf TKq(0),\qquad
D=\frac72-V_5=\frac{5\sqrt5-11}{4}>0,
\]
\[
A=\frac45F-\frac{N}{25},\qquad T=\frac14.
\]

The exact source-specific sufficient condition is

\[
\boxed{A<D
\quad\Longrightarrow\quad
\text{relative equilibrium limit }(w_D,w_R)=(+1,0).}
\]

The loss term makes this different from the static well test: a source
with `W(0)>7/2` can satisfy this condition. The conclusion follows from
guaranteed entry into the donor sublevel component by the fixed time
`T`, followed by the existing asymptotic theorem. Neither the state at
`T` nor its numerical trajectory is reconstructed, and `T` is not a
convergence deadline. The constants below are sufficient proof bounds,
not new parameters in the nodal law.

#### Full-state norm bounds from the exact initial sine balance

Reuse `y=K^(1/2)q`, `zeta=K^(1/2)S`, and
`B=K^(1/2)LK^(1/2)`. At the declared preparation `zeta(0)=0` and
`||y(0)||^2=N` for every choice of the six form coordinates. The
full eleven-node equations are

\[
\dot y=-\tfrac12By+aB\zeta,\qquad
\dot\zeta=-aC(\theta)y,\qquad a=\frac1{2\pi}<\frac16.
\]

The actual support and capacities give `||B||<=3`. The edge
representation also gives `-B<=C(theta)<=B`, so
`||C(theta)||<=3` globally, without an acute-phase restriction.
Let `Y(t)=||y(t)||`, `Z(t)=||zeta(t)||`, and `Y_0=sqrt(N)`.
Variation of constants, the contraction of `exp(-Bt/2)` and these
operator bounds yield

\[
Y(t)\le Y_0+3a\int_0^tZ(s)\,ds,\qquad
Z(t)\le3a\int_0^tY(s)\,ds.
\]

Comparison with the corresponding two scalar integral equations gives
`Y(t)<=Y_0*cosh(3*a*t)` and `Z(t)<=Y_0*sinh(3*a*t)`.
On `0<=t<=1/4`, use `3*a*t<1/8` and
`cosh(1/8)<=128/127<11/10`; the first bound follows directly by
majorizing its power series by `sum_k(1/128)^k`. Consequently

\[
Y(t)\le\frac{11}{10}Y_0,\qquad
Z(t)\le\frac{11}{20}tY_0.
\]

The same variation-of-constants formula gives a lower bound, retaining
the initial vector rather than inferring its norm from a derivative:

\[
\begin{aligned}
Y(t)&\ge e^{-3t/2}Y_0-3a\int_0^tZ(s)\,ds\\
&\ge Y_0\left(1-\frac32t-\frac{11}{80}t^2\right)
 \ge(1-2t)Y_0.
\end{aligned}
\]

The final factor is positive throughout the declared window. These
integral inequalities also include the zero-norm source without division
by a norm; when `N=0` the source is the stationary donor equilibrium.

#### A directional initial loss of the same auxiliary function

At half weights `h=a`. The already derived exact derivative and
`C(theta)<=B` give

\[
\dot{\mathcal W}
\le -\frac25Y^2
 +a\,y^\mathsf T(-\tfrac15I+\tfrac12B)\zeta
 -a^2\zeta^\mathsf TB\zeta+a^2y^\mathsf TBy.
\]

Here `||-I/5+B/2||<=13/10`. Dropping only the nonpositive
`zeta` quadratic and using `a<1/6` therefore gives

\[
\dot{\mathcal W}\le-\frac{19}{60}Y^2+\frac{13}{60}YZ.
\]

The norm estimates now supply a lower bound on actual accumulated
auxiliary decrease, not on an independently prescribed pressure:

\[
\begin{aligned}
\mathcal W(0)-\mathcal W(T)
&\ge N\left[
 \frac{19}{60}\int_0^{1/4}(1-2t)^2\,dt
 -\frac{13}{60}\frac{121}{200}\int_0^{1/4}t\,dt
 \right]\\
&=N\left(\frac{133}{2880}-\frac{1573}{384000}\right)
 =\frac{48481}{1152000}N
 \ge\frac{N}{25}.
\end{aligned}
\]

The last comparison is strict when `N>0`; retaining `N/25` gives
a simpler reusable certificate without optimizing its gain. In
particular,

\[
\mathcal W(T)\le V_5+A<\frac72.
\]

This inequality alone would not identify which well contains the state.
The next argument supplies that missing component information.

#### The phase path stays below the barrier as a consequence of the same test

Let `eta=theta(T)-theta(0)` in the actual continuous primitive-phase
lift. The full phase row gives

\[
\eta=aK^{1/2}\int_0^T y(t)\,dt,
\qquad
\eta^\mathsf TL\eta
 \le3a^2NT^2(11/10)^2.
\]

At the exact initial phase, `grad V=0`; globally `H(theta)<=L`.
Taylor's integral formula along the entire straight phase segment
therefore gives, for every `s in [0,1]`,

\[
V(\theta(0)+s\eta)
 \le V_5+\frac{s^2}{2}\eta^\mathsf TL\eta
 \le V_5+P,\qquad P:=\frac{121N}{38400}.
\]

This is a path in the circular state represented through a declared
lift, not an identification of primitive phase with a regional angle.
It does not require the segment to remain acute.

Importantly, `P<D` is **not another independent admission condition**.
The same full-support bound `||B||<=3` implies

\[
N=x(0)^\mathsf TLKLx(0)\le6F,
\qquad A\ge\frac{14}{25}F.
\]

Thus `A<D` already gives

\[
P\le\frac{121}{6400}F
 <\frac{121}{3584}D<D.
\]

At fixed phase `theta(T)`, convexity of `W` in form places the
straight segment from `x(T)` to the conserved uniform form below
`7/2`: its endpoint values are bounded by `V_5+A` and `V_5+P`.
At uniform form, the preceding phase segment joins it to the donor
equilibrium with `W=V<7/2`. This compact concatenation places the
actual state at `T` in the donor component of `{W<r}` for some
`r<7/2`. Subsequent nonincrease and the already proved global
single-equilibrium convergence force the original donor-only limit.
No transient receiver winding or interim current is discarded by this
argument.

#### One fixed above-threshold witness and its open preparation neighborhood

Fix, before evaluating any response, zero donor forms, intermediary
form `H=1/4`, and the analytic window `[0,1/4]`. The source has

\[
F=\frac1{16},\qquad N=\frac16,\qquad
A=\frac{13}{300},\qquad P=\frac{121}{230400}.
\]

The static retention criterion fails strictly, since
`(16*F+55)^2=3136>3125`, and `W(0)=V_5+1/20>7/2`.
Nevertheless `A<D`: equivalently `sqrt(5)>838/375`, whose
squared comparison is `703125>702244`. The source is nonsilent,
and the new theorem certifies its donor-only limit without integrating
its response. Its total initial energy also exceeds `7/2`, so neither
the bare energy barrier nor the earlier static auxiliary test supplies
this conclusion.

Both `F` and `N` are continuous quadratic forms on the six declared
initial form coordinates. At this witness the inequalities `F>F_c`,
`F<4`, and `A<D` are all strict. They consequently persist in a
nonempty open neighborhood in that **six-dimensional preparation
space**. Every source there has the same proved limit. This is not
an assertion about arbitrary phase or capacity perturbations, nor a
profile search or a new source-selection policy.

For represented rational source coordinates, the sole admission test
is exactly

\[
\boxed{125-(4A+11)^2>0.}
\]

There is no sign ambiguity: the actual support already proves
`A>=14F/25>=0`. At the fixed witness this polynomial margin is
`881/5625>0`. Outward display intervals for `V_5` and the analytic
upper-bound expressions do not decide this exact comparison. The certificate
reports a guaranteed state-set property at `T`, not an observed or
reconstructed endpoint, an error-enclosed trajectory, or entry into
a particular finite-radius recovery certificate. Outside this sufficient
criterion, receiver-transfer accessibility remains unresolved.

The existing formation owner exposes
[`donor_dissipative_capture()`](../../src/tnfr/physics/relational_sine_formation.py)
as `SineDonorDissipativeCapture`. It revalidates the primitive preparation,
recomputes `N` through every actual graph row, and checks the fixed-law
premises before using `endpoint_exact_polynomial_margin` for admission.
The retained phase polynomial is derived evidence, not a second
independent gate. `endpoint_functional_bounds` and
`phase_path_storage_bounds` enclose the **analytic upper-bound values**
`V_5+A` and `V_5+P`; they are not two-sided enclosures of an actual
future state or observed storage. The fixed `horizon=1/4` belongs to
this proof and does not reuse another report's finite-time verdict.

`donor_component_entry_certified` records the guaranteed sublevel
component membership by that horizon. Its relative-convergence and
receiver-target-exclusion flags retain the separate all-time consequences.
Unsupported premises remain `unavailable`, and a valid source failing
this sufficient criterion is `not_certified`. The
[formation controls](../../tests/physics/test_relational_sine_formation.py)
cover these consumed-state and interface boundaries; the
[independent algebra controls](../../tests/physics/test_relational_sine_formation_barrier.py)
check the full-field bounds, exact admission and fixed above-threshold
witness. Neither test owner evaluates a transfer trajectory.

<a id="sine-conservative-identity"></a>

## 23. Conservative phase identity with nonlinear recurrence

### Retain the law and distinguish trapping from recovery

Take the complete isolated unit cycle `C_n`, `n>=3`, with
declared orientation and **strictly positive held capacities**.
Use the complete normalized-sine law with `e=0,w,beta>0`
and no input, event, clipping or support change:

\[
\dot x=aKS(\theta),\qquad \dot\theta=bKLx,\qquad
K=\operatorname{diag}(\nu_i/2),\quad
a=w/\pi,\quad b=w/(\beta\pi).
\]

The total storage `E=x^TLx/2+beta*V(theta)` is conserved.
The [existing acute recovery proof](#sine-cycle-recovery) separates
a geometric first-exit barrier from an attracting dynamics argument.
Only the barrier is reused here. The latter requires `e>0`;
it is not a recovery theorem for the present conservative law.

Choose an integer `k` with `4*abs(k)<n` and the exact
critical target

\[
\theta_{*,j}=\frac{2\pi k j}{n},\qquad
\alpha=\frac{2\pi k}{n},\qquad
E_*=\beta n(1-\cos\alpha).
\]

Opposite sine currents cancel at each node, and all target edge
increments are strictly acute. The nontrivial case `n=5,k=1`
is the winding-one identity under study; the same derivation also
covers the other declared acute cycle twists.

### A genuine phase chart with common origins retained

Let `P=I-11^T/n` and write the centered form as `u=Px`.
Represent nearby circular phases by

\[
\theta=\theta_*+c\mathbf1+v\pmod{2\pi},\qquad
v\in\mathbf1^\perp,\quad c\in\mathbb R/2\pi\mathbb Z.
\]

Here `c` is one common circular phase origin, not a separately
adjustable origin at each node. Choose `r>0` such that

\[
|\alpha|+\sqrt2\,r<\frac\pi2.
\]

The representation is injective for `||v||<r`. To see this,
suppose two centered deviations and common origins give the same
phase state. Subtracting any two node equations gives

\[
(v_i-v_j)-(v'_i-v'_j)=2\pi(m_i-m_j)
\]

for integers `m_i`. Its left-hand side has magnitude less than
`2*sqrt(2)*r<2*pi`, so all the integers agree. Centering then
forces `v=v'` and equality of the common origins on the circle.
The linear differential of the chart is invertible, so its image
is open in the phase torus. The strict radius bound also supplies
an injective chart on a slightly larger neighborhood of its closure.
There is no phase-seam exit hidden inside the admitted ball.

Define the full relative norm in the same model coordinates as the
recovery owner,

\[
Z^2=\|u\|^2+\|v\|^2.
\]

For `Z<=r` the oriented wrapped edge increments are
`alpha+v_(j+1)-v_j`, including the closing edge. They remain
strictly acute, and their sum is `2*pi*k`. Every state in this
chart therefore has the same declared oriented winding `k`.

### The coercive barrier is independent of dissipation

Let

\[
\lambda_2=2-2\cos(2\pi/n),\qquad
c_r=\cos(|\alpha|+\sqrt2r)>0,\qquad
\kappa_r=\frac{\lambda_2}{2}\min(1,\beta c_r)>0.
\]

The phase segment from `theta_*` to `theta_*+v` remains
acute throughout the closed radius ball. Criticality cancels its
linear storage term, and its Hessian is bounded below by `c_r L`.
The graph spectral gap thus gives

\[
\mathcal E:=E-E_*
\ge\frac{\lambda_2}{2}\|u\|^2
 +\frac{\beta c_r\lambda_2}{2}\|v\|^2
\ge\kappa_r Z^2
\qquad(Z\le r).
\]

Choose an independently declared excess ceiling and weighted-mean
interval,

\[
0<h<\kappa_r r^2,\qquad m_{\rm lo}<m_{\rm hi},
\]

where

\[
m_\rho(x)=\frac{\sum_i\rho_i x_i}{\sum_i\rho_i},\qquad
\rho_i=\frac{2}{\nu_i}>0.
\]

The proposed full-state family is

\[
\boxed{
\mathcal U=\left\{
Z^2<r^2,\quad \mathcal E<h,\quad
m_{\rm lo}<m_\rho(x)<m_{\rm hi}
\right\},}
\]

with the common phase origin `c` free on its circle.
The initial radius condition cannot be omitted: a small energy
relative to `E_*` outside this chart does not identify this
phase pattern.

For every member of `U`, conservation of `mathcal E`
prevents a first radius exit. At such an exit the coercive bound
would require `mathcal E>=kappa_r*r^2>h`. The argument
works in both time directions because energy and the weighted form
mean are exactly conserved. The chart is retained, and within it

\[
Z(t)^2\le\frac{\mathcal E(0)}{\kappa_r}
 <\frac h{\kappa_r}<r^2
\qquad\text{for all real }t.
\]

Every state in the family therefore retains acute edges and winding
`k` indefinitely under the continuous declared law. This is
**trapping**, not convergence to `theta_*`. For a nonzero
excess energy, convergence to that zero-excess target is impossible
under exact conservation.

### Open positive-volume family and almost-everywhere motion

The family is open in the full `2n`-dimensional state manifold.
Indeed the phase chart is locally invertible, and form can be
parameterized by `(m_rho,u)` with `u perpendicular 1`:

\[
x=\left(m_\rho-\frac{\rho^\mathsf Tu}
                         {\rho^\mathsf T\mathbf1}\right)\mathbf1+u.
\]

This is an invertible linear coordinate change. Uniform form with
mean strictly inside the interval and the exact target phase lies
inside every displayed strict inequality; a full open neighborhood
does too. Thus the family has positive ambient product volume,
not merely volume within a synchronized or fixed-energy surface.

Its volume is finite. Positive mean weights and the radius give

\[
|x_i-m_\rho(x)|
 \le\max_j|u_i-u_j|
 \le\sqrt2\,\|u\|<\sqrt2r.
\]

All forms therefore lie in a bounded box, and all phases lie on
their compact torus. The **open** family need not itself be compact;
its closure in the retained chart and form box is compact. The
strict energy barrier keeps its trajectories away from the chart's
radius boundary, while the conserved mean keeps each trajectory at
its own interior mean value. The smooth sine flow is complete and
maps `U` onto itself in both time directions.

The [full nonlinear recurrence theorem](RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
now applies to this invariant finite-volume family. Its divergence
is zero, so almost every state of `U`, in ambient Lebesgue
form times circular Haar phase measure, returns arbitrarily close
to itself along arbitrarily late times. The equilibrium subset has
measure zero, since positive capacity forces equilibrium form to
be uniform. Consequently almost every member has **nonstationary
recurrent motion while retaining this phase identity**.

The quantifiers differ: every admitted member is trapped with fixed
winding; only almost every member is certified recurrent by this
measure theorem. Neither result selects a common period, return
time, oscillation amplitude, physical frequency or attracting orbit.
The prescribed support and target identity are not formed by this
admission test.

### Relative observations and the absolute mean slab are different evidence

An uncertain source can certify the shape conditions if its entire
declared relative-state set satisfies

\[
\sup Z^2<r^2,\qquad \sup\mathcal E<h.
\]

Reuse the existing owner for centered pairwise norms, exact target
turns and the phase-energy Taylor remainder. Original observations
retain their correlated common-origin-plus-residual set; a forecast
uses its full validated endpoint box at the actual validated time.
A requested future time is not certified merely by a partial forecast.
No midpoint, redefined phase origin or changed law supplies a missing
strict bound.

These shape inequalities prove trapping for every compatible state
even if the common form origin is unspecified. However,
`SineRelativePattern` permits an arbitrary common form shift.
Its entire observation set therefore cannot be declared inside a
finite absolute `m_rho` interval by substituting the nominal mean.
The finite-volume family and relative shape membership must be
reported separately. Absolute-family membership requires independent
mean information, or an explicitly conditional intersection of the
observation set with the declared mean slab.

The same separation applies to recurrence: neither relative trapping,
absolute slab membership nor an error box makes its selected true
state an almost-everywhere random draw. Singular preparations,
fixed-energy subsets and finite numerical grids require separate
evidence. A stationary target member has only trivial recurrence;
the family theorem does not turn it into a pulse.

This combines two existing mechanisms under compatible conservative
hypotheses: an all-state phase barrier and nonlinear volume-preserving
recurrence. It establishes a supplied coherent identity compatible
with persistent recurrent motion, not spontaneous formation,
microscopic selection of zero loss or fractal inheritance.

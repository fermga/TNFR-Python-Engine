# Collective actuation with unresolved fine state

<a id="organized-parent-collective-actuation"></a>

## Fixed observation, input and response question

This analytic question reduces the information and input authority of the
[full-state construction](ORGANIZED_PARENT_CONSTRUCTIVE_ACTUATION.md).
The actual acquired state is retained; projecting its observations constructs
a reference and never resets a fine coordinate. The declaration below fixes
one input recipe and its response budgets before their assessment.

| Item | Fixed choice |
| --- | --- |
| Source | The same four independently uncertain recovered C9 children, each with original form and phase allowances `1e-10`, carried through the unchanged parent acquisition to actual `z4=(x4,theta4)` at absolute slow time `s=4` |
| Complete law | Matching 36-node support with all 72 unit edges, held unit capacities; `x_tau=(-Lx+gamma*S(theta))/4+Gamma_tau`, `theta_tau=gamma*Lx/4`, `gamma=1/(1023*pi)` |
| Clock and duration | `tau=(1023/1024)*t`, `s=gamma^2*tau/2`; forcing runs from `s=4` to `s=8`, with fixed elapsed duration `h=4` |
| Available observations | Exactly the four actual child form means `X[g,a]` and four actual mean lifted phase offsets `Phi[g,a]=mean_j(theta[g,a,j]-psi[j])` at `s=4`, with `psi[j]=(2*pi/9)*(j-4)`. No individual fine coordinate or later observation is supplied to the input or predictor |
| Reference initialization | Repeat each observed form mean within its marked child; initialize reference phases as `psi[j]+Phi[g,a]`. Retain the common form and phase means |
| Input | Apply the previously fixed cubic recipe to this reference, using the same displacement `k[0,a,j]=(2*a-1)/4`, zero on parent one. Supply its resulting form input identically to all nine nodes of each child: four input channels, no fine feedback, phase input, support event or capacity change |
| Actual endpoint | At `s=8`, require `||x8-x4||_2<=1e-7` and `||theta8-theta4-k||_2<=1e-7`, using the declared continuous lifts. These are separate form and phase budgets; no exact fine-state endpoint is assumed |
| Work | Actual signed net input work `integral (Lx)^T*Gamma_tau d(tau)<=3` over the forced interval; include actual storage change and continuous loss |
| Release and predictor | Set `Gamma=0` at `s=8` without a state jump. Continue the exact eight-coordinate child-constant law from `(X4,Phi4+k_child)`, already determined by the initial observations; no measured release state initializes this prediction |
| Readout | At `s=9`, each of the four actual child form means and four lifted phase-offset means differs from that predictor by at most `1e-6`, separately in each channel |
| Identity | Preserve every marked child's winding one throughout `4<=s<=12`. Parent zero must lose registration at release; both parents must recover it at `s=12`, with every intraparent same-slot circular gap strictly below `1/4` in magnitude |
| Decision | Admit the original source, four-channel realization within endpoint and work budgets, child retention, disruption, recovery and predictive accuracy, or a scoped obstruction. A failed sufficient estimate remains unavailable |

Let `P` be the orthogonal projection that replaces each nine-node block by
its mean, applied to forms and carrier-subtracted phase lifts. With the fixed
`L`, `k` and `h`, the input is precisely equation (1) of the full-state
construction initialized at `(P*x4, psi+P*(theta4-psi))`. It is supplied as a
function of the eight initial observations and time, independently of the
subsequently evaluated response. Its uniformity within each child is a proof
obligation, not an additional approximation of that recipe.
The inherited four-child law retains denominator four and both original
clock transformations; its two external neighbors do not renormalize the law.

Compare actual and reference flows under this same input. The earlier
comparison of two different state-dependent inputs cannot establish the new
error bounds. Endpoint mismatch must enter release, work and prediction;
the earlier exact-endpoint handoff cannot be assumed. Initial observation
accuracy is exact in this question; noise, finite precision, actuator tracking
and distribution errors require their own admission.

Marked children, continuous phase lifts, fixed support and capacity, supplied
timing and uniform actuation remain premises. Signed net work does not bound
positive-only supply, amplitude or bandwidth. No registered operator runtime,
autonomous occurrence, universal quotient, physical measurement bridge or
optimal control is claimed. No source refinement, trajectory, duration scan
or new numerical campaign is activated.

The declaration above was committed as `a5fe8f3a` before the following
assessment. The original source, four-unit input duration, endpoint budgets,
readout budgets and work budget remain unchanged. The result is a conditional
input and prediction theorem, without a numerical response trajectory.

## 1. Observation, invariant reference and four input channels

Let \(\Pi\) denote the blockwise orthogonal projection called \(P\) in the
declaration. Write the actual initial observation and its lifted reference as

\[
 X_{g,a}=\frac19\sum_j x_{4,g,a,j},\qquad
 \Phi_{g,a}=\frac19\sum_j(\theta_{4,g,a,j}-\psi_j),\qquad
 \bar x_4=\Pi x_4,\quad
 \bar\theta_4=\psi+\Pi(\theta_4-\psi).
 \tag{1}
\]

The phase observations use the supplied continuous lifts and marked winding
carrier, not a circular average inferred from an arbitrary representative.
The observation is assumed exact here. Its physical measurement, noise and
the selection of the child marks are not derived.

On the matching graph, \(\Pi L=L\Pi\). The child-constant subspace is invariant
under \(L\), and its restriction is the square Laplacian \(\mathcal L\) on
the four child indices. For a phase field \(\psi_j+\Phi_{g,a}\), the two
internal cycle currents cancel exactly. Each matching external contact has
the phase difference of its two child offsets. Hence its complete sine
current is also constant within each child.

The eight-coordinate reference equations, including the supplied input, are

\[
 X_\tau=\frac{-\mathcal LX+\gamma\mathcal S(\Phi)}4+G_\tau,
 \qquad
 \Phi_\tau=\frac{\gamma\mathcal LX}4,\qquad
 \mathcal S_c(\Phi)=\sum_{d\sim c}\sin(\Phi_d-\Phi_c).
 \tag{2}
\]

There are two square neighbors per child, but the denominator remains four.
The original node degree, capacities and complete law have not been replaced
by a newly normalized degree-two model.

Apply the frozen cubic recipe to \((\bar x_4,\bar\theta_4)\). Its form,
form derivative, Laplacian and sine current are all constant within each
child. Therefore its input has exactly four values \(G_{\tau,g,a}\), each
supplied to all nine nodes of that child. The input depends only on (1), the
fixed model and time. The actual flow receives these same functions; no
unknown fine coordinate is used to recompute or correct them.

At slow time eight the reference is exactly
\(T(\bar z_4)=(\bar x_4,\bar\theta_4+k)\). Set the common input to zero
thereafter. The autonomous version of (2), initialized from
\((X_4,\Phi_4+k_{\rm child})\), is the declared predictor. It remains an
exact invariant-family solution of the fine law. No measurement at release
or at readout enters its initial state.

The actual trajectory generally leaves that family. In particular, later
fine errors need not keep zero mean within each child. Predicting those
means requires a comparison theorem; exact reference closure is not exact
closure of the actual observed state.

## 2. Same-input comparison for all original fine directions

The fresh acquisition supplies its existing reference \(z_4^*\), whose form
and carrier-subtracted phase are child-constant, and its full error bounds

\[
 \|x_4-x_4^*\|_2\le R,\qquad
 \|\theta_4-\theta_4^*\|_2\le Z,\qquad R,Z<10^{-60}.
 \tag{3}
\]

These are the original independently uncertain recovered children, with all
64 fine source directions, the same allowances and the same recovery times.
Orthogonality of \(\Pi\) gives

\[
 r_0=x_4-\bar x_4,\quad \zeta_0=\theta_4-\bar\theta_4,\qquad
 \|r_0\|_2\le R,\quad \|\zeta_0\|_2\le Z.
 \tag{4}
\]

Both initial errors have zero mean within every child and hence globally.
The full source has not been reset to its projection. Nor is the projected
state asserted to be another autonomously acquired source.

Compare actual and reference states under the identical supplied input,
including their identical zero input after release. Their differences obey

\[
 r_\tau=-\frac14Lr+\frac\gamma4\Delta S,\qquad
 \zeta_\tau=\frac\gamma4Lr,\qquad
 \Delta S=S(\theta)-S(\bar\theta).
 \tag{5}
\]

The forcing cancels exactly. The global form-error mean remains zero, since
both \(Lr\) and the pairwise sine-current sum have zero global mean. The
Laplacian and sine map satisfy the already admitted bounds
\(\lambda_2(L)>1/3\), \(\|L\|_2\le8\) and
\(\|\Delta S\|_2\le8\|\zeta\|_2\), without an acute-domain assumption.
The heat-integral argument therefore gives

\[
 \begin{split}
 \zeta(\tau)={}&\zeta_0+\gamma(I-e^{-L\tau/4})r_0\\
 &+\frac{\gamma^2}{4}\int_0^\tau
 (I-e^{-L(\tau-u)/4})\Delta S(u)\,du,\\
 \|\zeta(\tau)\|_2&\le(Z+\gamma R)e^{2\gamma^2\tau}.
 \end{split}\tag{6}
\]

For elapsed slow horizon \(H\), \(2\gamma^2\tau\le4H\). Define exact
rational upper bounds, using outward \(\underline\gamma,\bar\gamma\), by

\[
 Z_H=(Z+\bar\gamma R)3^{4H},\qquad
 R_H=R+24\bar\gamma Z_H,\qquad H\in\{4,5,8\}.
 \tag{7}
\]

The phase bound uses \(e<3\); the form convolution uses the global spectral
gap and factor 24. Thus \(\|\zeta\|_2\le Z_H\) and
\(\|r\|_2\le R_H\) throughout each respective interval. The switch to zero
input causes no state jump and does not change (5). This is a same-input
comparison, not the polynomial difference formula for two distinct
state-dependent inputs in the previous construction.

For orientation, the existing coarse source bounds,
\(\bar\gamma<1/3200\), \(3^{16}<10^8\), \(3^{20}<4\cdot10^9\) and
\(3^{32}<2\cdot10^{15}\) already give

\[
 \begin{array}{c|cc}
 H& Z_H\text{ upper bound}&R_H\text{ upper bound}\\ \hline
 4&2\cdot10^{-52}&2\cdot10^{-54}\\
 5&5\cdot10^{-51}&4\cdot10^{-53}\\
 8&3\cdot10^{-45}&3\cdot10^{-47}.
 \end{array}\tag{8}
\]

These are deliberately loose consequences of the unchanged source, not new
preparation tolerances. The implementation retains the sharper exact bounds.

## 3. Actual endpoint, whole-path identity and eight-coordinate readout

Projection commutes with the fixed child-uniform phase displacement. At
release the reference equals \(T(\bar z_4)\), whereas the target is
\(T(z_4)\). Consequently the actual endpoint errors are

\[
 x_8-x_4=r_8-r_0,\qquad
 \theta_8-\theta_4-k=\zeta_8-\zeta_0.
 \tag{9}
\]

Both discarded initial residuals must be retained. The sufficient budgets are

\[
 E_x:=R_4+R,\qquad E_\theta:=Z_4+Z,\qquad
 \|x_8-x_4\|_2\le E_x<10^{-7},\quad
 \|\theta_8-\theta_4-k\|_2\le E_\theta<10^{-7}.
 \tag{10}
\]

This is approximate full-state realization. No exact return of unknown fine
forms or phases has been inferred from their means.

For the reference, every internal C9 edge stays on its original winding
branch with gap \(\alpha=2\pi/9\), throughout forcing and the autonomous
continuation of (2). Equation (7) at \(H=8\) confines every actual internal
edge gap to

\[
 [\alpha-2Z_8,\alpha+2Z_8]\subset(0,\pi/2).
 \tag{11}
\]

The closing edge retains its integer lift correction. The continued gaps
sum to \(2\pi\), so every actual child retains winding one for the complete
absolute interval \(4\le s\le12\). This conclusion needs no later
parent-recovery verdict or acute premise in the comparison equation.

At absolute time nine the common-input comparison has elapsed horizon five.
The Euclidean norm of a nine-node mean functional is \(1/3\). Thus for each
child, with the reference evolved by (2) from the fixed projected target,

\[
 |X_c(9)-\bar X_c(9)|\le R_5/3<10^{-6},\qquad
 |\Phi_c(9)-\bar\Phi_c(9)|\le Z_5/3<10^{-6}.
 \tag{12}
\]

These are separate form and lifted-phase budgets for all eight coordinates.
The predictor consumes only the eight initial observations. The estimate
does not reuse the earlier scalar parent-gap forecast as if it covered every
coordinate, and does not consume an unprovided release observation.

## 4. Actual signed work under the shared input

For both actual and reference trajectories, the complete input balance is

\[
 W_{\rm in}=\Delta H+\frac14\int_0^{T_\tau}\|Lx\|_2^2d\tau,\qquad
 H=\tfrac12x^TLx+\sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)].
 \tag{13}
\]

The actual endpoint storage in this identity is evaluated at its actual
approximate endpoint. It is not replaced by the hybrid-map storage jump.
To bound its work, instead compare the power integrals directly:

\[
 W_{\rm in}-\bar W_{\rm in}
 =\int_0^{T_\tau}(Lr)^T\Gamma_\tau\,d\tau.
 \tag{14}
\]

This identity is valid because both trajectories receive exactly the same
input. It accounts for actual endpoint and loss differences together.

The projected initial state remains within the same \(R,Z\) bounds about
the acquisition reference, since projection contracts those errors. The
static cubic path and endpoint-work estimates from the
[full-state construction](ORGANIZED_PARENT_CONSTRUCTIVE_ACTUATION.md#2-exact-continuous-loss-and-the-original-source-work-budget)
can therefore be rederived for that projected state. They require those
bounds, not a claim that the projection was acquired. In particular, with
\(h=4\), \(A=1+2\bar\gamma^2\) and \(K=9/8>\|k\|_2\), put

\[
 M=3hA(\overline B_4+\overline\Delta_4)
      +\frac{4hR}{\underline\gamma}<\frac14.
 \tag{15}
\]

This bounds \(\|hL\bar x_4/(2\gamma)\|_2\). The reference endpoint work
has the same nominal expression and signed error at most \(18Z\), and its
polynomial loss is at most
\((2/h)[(6/5)(9/8)+(2/5)KM+M^2/5]\). In particular
\(\bar W_{\rm in}<5/2\).

To control (14), the square's nonconstant eigenvalues are two and four, so
\(\|L^+k\|_2\le K/2\) and
\(\|\bar x_4-\operatorname{mean}(\bar x_4)\mathbf1\|_2\le\gamma M/h\).
The fixed cubic derivatives satisfy
\(|P'|\le3/2\), \(|Q'|\le1\), \(|P''|,|Q''|\le6\).
Its reference sine current has norm at most 12: the internal currents cancel
and each of the 36 nodes has at most two unit external sine contributions.
The explicit input formula therefore gives

\[
 \|\Gamma_\tau\|_2\le G:=
 \frac{3\bar\gamma^3}{h^2}(K+M)
 +\frac{\bar\gamma}{4h}(3K+2M)+3\bar\gamma
 <4\bar\gamma.
 \tag{16}
\]

Using \(\|Lr\|_2\le8R_4\) and
\(T_\tau=2h/\gamma^2\), define

\[
 E_W:=\frac{16hR_4G}{\underline\gamma^2},\qquad
 |W_{\rm in}-\bar W_{\rm in}|\le E_W.
 \tag{17}
\]

For example \(\underline\gamma>1/5000\), the existing
\(\bar\gamma<1/3200\) and (8) give \(E_W<4\cdot10^{-48}\).
Thus the actual work satisfies
\(W_{\rm in}\le\bar W_{\rm in}+E_W<3\).
The detached calculation retains the sharper reference work upper bound
and correction (17). This is the actual signed net budget. It does not
certify a bound on total positive supply, amplitude or bandwidth.

## 5. Disruption and recovery with the actual endpoint mismatch

Let \(z^\dagger(u)\) be the already admitted autonomous restoring response
started from the actual target \(T(z_4)\), with elapsed slow time \(u=0\)
at the new release instant \(s=8\). The actual release state differs from
that target by at most \(E_x,E_\theta\) in (10). Both subsequent flows have
zero input, the same support, capacities, complete law and clocks.
Their global form means agree because the prescribed input and the target
preserve the actual original mean.

Reapplying the same-input argument for the next four slow units gives

\[
 Z_{\rm rel}=(E_\theta+\bar\gamma E_x)3^{16},\qquad
 R_{\rm rel}=E_x+24\bar\gamma Z_{\rm rel}.
 \tag{18}
\]

These bounds apply to actual-minus-\(z^\dagger\), not to a reset source.
The coarse bounds above imply \(Z_{\rm rel}<4\cdot10^{-44}\) and
\(R_{\rm rel}<4\cdot10^{-46}\). Parent identity concerns the phase gaps;
the form comparison is retained as part of the complete-state handoff.

At release the nominal target's parent-zero same-slot gap is \(B_4+1/2\).
The admitted actual target differs by at most \(2Z\) per gap, and the new
endpoint adds at most \(2E_\theta\). All these gaps remain on the same
principal branch. Hence the strict disruption margin is

\[
 \underline B_4+\frac12-2Z-2E_\theta-\frac14>0.
 \tag{19}
\]

At elapsed time four the prior target-response theorem supplies a uniform
bound \(U_{\rm old}<1/4\) on every intraparent same-slot gap. Transporting
the actual endpoint mismatch changes each gap by at most \(2Z_{\rm rel}\).
For the unchanged source its admitted return margin gives

\[
 U_{\rm old}+2Z_{\rm rel}<\frac14.
 \tag{20}
\]

The old bound is well inside the threshold (its derived nominal return
sum is below \(1/50\), with its separately retained tiny fine correction).
Thus both actual parents recover the declared registration at \(s=12\).
This is a perturbed release theorem. The exact-endpoint handoff of the
full-state construction has not been assumed.

## 6. Evidence and limits

The [detached assessor](../../src/tnfr/physics/_organized_parent_collective_actuation.py)
rebuilds the original source from all eight primitive
allowances. Static reference-work coefficients, actual same-input errors,
the four input channels and the new readout budgets have separate owners
within that calculation. No cached exact-realization or future-response
verdict substitutes for a new forced-path premise. Parent recovery consumes
the old admitted return bound only after propagating the new mismatch.

The input evaluator accepts only the four form and four lifted phase means
plus time. Evaluating the invariant reference is neither measurement of the
actual response nor execution of a live controller. The
[independent controls](../../tests/physics/test_organized_parent_collective_actuation.py)
check the block projection, matching fine and quotient rows, uniform input,
clock transformations, full-norm endpoint error, every readout coordinate,
work correction and release margins.

The theorem reduces information and input channels for this unchanged source
and fixed protocol. It establishes neither an exact quotient for arbitrary
fine states nor recovery from an arbitrary error ball of the same displayed
endpoint radius. The full carried error bounds, not only that radius's budget
label, enter (18). Marked children, exact initial observations, phase lifts,
fixed capacities/support and supplied actuation timing remain premises.
There is no autonomous occurrence law, physical observation bridge, optimal
control claim or additional chronological no-actuation comparison.

The separate [autonomous reservoir question](ORGANIZED_PARENT_RESERVOIR_EXCHANGE.md)
adds an explicit source state and reciprocal exchange law. Its two network
inputs can differ under joint-state perturbation, so its error proof and
response decision do not follow from the identical-input estimate above.

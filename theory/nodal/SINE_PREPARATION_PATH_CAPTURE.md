# A whole-family unit-winding selector through the actual phase passage

<a id="sine-preparation-path-capture"></a>

Every source in the previously declared interior corridor
\(s\in[3/4,4/5]\) acquires and permanently retains winding \(+1\)
under the same complete C9 sine law. The proof follows the original
trajectories, including their form state and correlated errors. It uses an
exact mixed coordinate, a finite phase-passage cone and an acute contraction
estimate. No independently initialized phase-gradient solution replaces
the actual dynamics.

The [57-state limit theorem](SINE_PREPARATION_PATH_LIMITS.md#sine-preparation-path-limits)
gave necessary terminal alternatives, including exceptional saddles.
The source-specific argument here excludes those saddles, consensus and
the negative unit winding for this entire corridor. It does not assert a
universal attractor for other preparations or for other constitutive laws.

The [constitutive extension](SINE_PREPARATION_PATH_ROBUSTNESS.md) proves the
same source-specific capture for the already declared potential family by
rebuilding its current, continuation, contraction and storage bounds. The
present sine proof and its tighter contraction rate remain unchanged.

<a id="sine-path-capture-carried-state"></a>
## The original source and an exact continuation coordinate

Retain all source, law and clock premises from the
[Cayley path owner](SINE_PREPARATION_PATH_SELECTION.md#sine-preparation-path-law):
the isolated unit C9, degree matrix \(D=2I\), capacities one,
\(A=L/2\), \(\gamma=1/(1023\pi)\), \(\eta=\gamma^2\),
\(\tau=(1023/1024)t\), and
\[
 x'=-Ax+\gamma f(\theta),\qquad
 \theta'=\gamma Ax,\qquad
 f_i(\theta)=\tfrac12\sum_{j\sim i}\sin(\theta_j-\theta_i).
 \tag{1}
\]
The source is exactly
\[
 x_s(0)=Q_s(ma+\xi),\quad \theta_s(0)=\zeta,\quad
 s\in[3/4,4/5],\quad
 |\xi_j|,|\zeta_j|\le10^{-10},\quad
 \mathbf1^T\xi=\mathbf1^T\zeta=0 .
 \tag{2}
\]
The original \(M=512\), rational \(m\), latent correlations, equal initial
storage and source budget \(10^{15}\) are unchanged.
No event, forcing or state reset is introduced.

At the certified transit time \(\tau_0=200\), continue the actual state
in the invertible coordinates
\[
 z=\gamma x,\qquad y=\theta+z,\qquad
 \sigma=\eta(\tau-\tau_0).
 \tag{3}
\]
The exact rows, in both clocks, are
\[
 \begin{array}{ll}
 z'=-Az+\eta f(\theta),&y'=\eta f(\theta),\\
 z_\sigma=-Az/\eta+f(y-z),&
 y_\sigma=f(y-z)=f(y)+e,\quad e=f(y-z)-f(y).
 \end{array}\tag{4}
\]
The consumed primitive phase remains \(\theta=y-z\); \(y\) is an
auxiliary mixed coordinate, not an assumed primitive phase or a new law.
Both original conserved means remain zero.

On the zero-mean space the exact bounds
\[
 \|e^{-A\tau}\|_D\le e^{-\tau/5},\qquad
 \|f\|_D\le F=\sqrt{18},\qquad
 \|f(v)-f(w)\|_D\le2\|v-w\|_D
 \tag{5}
\]
give a uniform bound on the complete later form:
\[
 \|z(\tau)\|_D\le
 Z:=\max\{R_z(\tau_0),\,5\eta F\},\qquad
 \|e\|_D\le\varepsilon:=2Z,\qquad \tau\ge\tau_0 .
 \tag{6}
\]
Here \(R_z(\tau_0)\) is rebuilt from the actual source semigroup estimate.
An edge-difference functional has dual \(D\)-norm one, so every edge
perturbation \(e_j-e_i\) has magnitude at most \(\varepsilon\);
every actual phase edge differs from its \(y\)-edge by at most \(Z\).

The exact integrated identity is particularly useful at the handoff:
\[
 y(\tau_0)-\gamma mQ_sa
 =\zeta+\gamma Q_s\xi+\eta\int_0^{\tau_0}f(\theta(v))\,dv .
 \tag{7}
\]
It retains the same source and complete past history. The initial mixed
error is bounded in \(D\)-norm by
\(E_\theta+E_x+\eta F\tau_0\), with the transported source errors of
the path owner. It does not include a second fast-form remainder;
the \(z(\tau_0)\) terms have canceled by identity.

Fresh scalar bounds for the fixed whole source family give
\[
 Z<0.000002054,\qquad
 \varepsilon<0.000004108<5\,10^{-6},\qquad
 \|y(\tau_0)-\gamma mQ_sa\|_D<0.000082152 .
 \tag{8}
\]
These are exact outward rational bounds on actual states, not observations
of a trajectory.

<a id="sine-path-capture-passage-cone"></a>
## A finite cone for the actual mixed phase passage

Use the eight consecutive nonclosing \(y\)-gaps \(d_0,\ldots,d_7\)
and the continued closing gap \(\beta\), with the same fixed integer
offsets used at the original winding-two handoff. Thus
\[
 \sum_{i=0}^7d_i+\beta=4\pi,\qquad q=\pi-\beta,\qquad
 \sum_{i=0}^7d_i-q=3\pi .
 \tag{9}
\]
Explicitly, put \(\widetilde y=y-2\pi Mh\),
\(d_i=\widetilde y_{i+1}-\widetilde y_i\) and
\(\beta=\widetilde y_0-\widetilde y_8+4\pi\).
These fixed offsets define a chart of the continued state; they perform
no state jump and remove nothing from the earlier form or transit bounds.
The proxy and (7) certify the initial conditions
\[
 0<q(0)<63/100,\qquad
 \min_i d_i(0)-q(0)/2>9/10,\qquad
 \max_i d_i(0)<13/10 .
 \tag{10}
\]
For example the rebuilt bounds are
\(0.3489<q(0)<0.6285\),
\(\min_i d_i(0)>1.2215\),
\(\max_i d_i(0)<1.2570\), and the independently bounded
minimum-minus-half-maximum expression exceeds \(0.9072\).
Each is uniform in \(s,\xi,\zeta\).

All derivatives in the rest of the proof use slow elapsed time \(\sigma\).
The edge equations from (4) are
\[
 \begin{aligned}
 d_i'&=\tfrac12[\sin d_{i-1}-2\sin d_i+\sin d_{i+1}]+r_i,\\
 q'&=\sin q-\tfrac12(\sin d_0+\sin d_7)+r_q,
 \qquad |r_i|,|r_q|\le\varepsilon .
 \end{aligned}\tag{11}
\]
At the two chain endpoints the missing neighbor's sine is \(\sin q\).
These are exact actual mixed-state equations with a bounded residual,
not a substituted gradient trajectory.

Declare the passage target \(q_*=-7/4\) and slow-time budget eight.
Set \(m_d=\min_i d_i\), \(M_d=\max_i d_i\) and
\(C=m_d-q/2\). While \(q\in[-7/4,63/100]\),
the nonclosing gaps remain acute, and
\(\Delta=\sin m_d-\sin q\ge0\), their one-sided extrema satisfy
\[
 C'\ge-\tfrac32\varepsilon,\qquad
 M_d'\le\varepsilon,\qquad
 q'\le-\Delta+\varepsilon .
 \tag{12}
\]
To verify the first inequality, an interior minimum has unperturbed
derivative at least zero. At an endpoint minimum its derivative is at
least \(-\Delta/2\), whereas \(q'/2\le-\Delta/2+\varepsilon/2\).
Both cases give the claimed lower derivative of \(C\).
At a maximum all neighboring sine currents are at most \(\sin M_d\).
The same inequalities hold at tied extrema using one-sided derivatives.

For \(0\le\sigma\le8\), (10)--(12) imply the strict bounds
\[
 C>0.9-\tfrac32(5\,10^{-6})8>0.899,\qquad
 M_d<1.3+(5\,10^{-6})8<1.301,\qquad
 m_d>0.899+q/2\ge0.024 .
 \tag{13}
\]
The required sine ordering and a uniform speed follow without evaluating
a flow. For \(q\ge0\), the function
\(\sin(0.899+q/2)-\sin q\) decreases on \([0,0.63]\), so
\[
 \Delta\ge\sin(1.214)-\sin(0.63)>1/3 .
 \tag{14}
\]
For \(-1/2\le q\le0\), \(\Delta\ge\sin(0.649)>1/3\).
For \(-7/4\le q\le-1/2\), positivity of \(m_d\) and
\(-\sin q\ge\sin(1/2)>1/3\) give the same conclusion.
The three strict rational sine inequalities can be certified directly
by alternating Taylor bounds or the shared outward sine kernel.
Thus \(q'<-3/10\).

Equations (13)--(14) also close the bootstrap: every nonclosing gap is
strictly between zero and \(\pi/2\), and the required current ordering
cannot fail before the declared target or deadline. Consequently the
first hitting time \(\sigma_*\) of \(q_*\) exists and satisfies
\[
 0<\sigma_*<
 \frac{63/100+7/4}{3/10}=\frac{119}{15}<8 .
 \tag{15}
\]
This first hitting time is a proof coordinate. It is not an admitted
hybrid event, a change of law or an observed trigger.

At that time the closing principal \(y\)-gap is
\[
 \beta-2\pi=7/4-\pi,
 \tag{16}
\]
and the other eight gaps lie between \(0.024\) and \(1.30004\).
Their principal sum is now \(2\pi\). Subtracting the actual \(z\)-edges
changes each gap by at most \(Z\), so the actual phase also lies
strictly in the winding-one acute cell.
The actual trajectory had winding two at \(\tau_0\); a phase passage has
therefore occurred. This does not count its primitive-phase branch-cut
crossings or assert monotonicity of the primitive closing phase.

<a id="sine-path-capture-acute-relaxation"></a>
## Finite acute relaxation and a full storage handoff

Acute entry alone is not yet the full trapping certificate.
Beginning at the same actual \(\sigma_*\), declare a further slow elapsed
time \(\Sigma=128\). Use the nine principal \(y\)-gaps in the new winding-one
cell. As long as they remain acute, the edge maximum principle for (11)
gives extrema drift at most \(\varepsilon\Sigma\).
Together with (16) and \(\varepsilon<5\,10^{-6}\), this gives
\[
 \begin{aligned}
 \min(\text{\(y\)-gaps})&\ge7/4-\pi-128\varepsilon>-7/5,\\
 \max(\text{\(y\)-gaps})&\le1.3+136\varepsilon<7/5 .
 \end{aligned}\tag{17}
\]
The inequalities have enough strict margin to include the extra
actual-phase error \(Z<2.5\,10^{-6}\).
Hence both \(y\) and \(\theta=y-z\), and their connecting segments,
remain within the common chart with absolute phase gaps less than
\(7/5<\pi/2\) throughout this finite stage.
This is again a first-exit bootstrap, not an infinite-time inference
from a small perturbation.

Let \(\Theta_1\) be the regular winding-one target in that same cell,
with the same zero mean. Its nine principal gaps are \(2\pi/9\).
In the retained lift it is \(\Theta_1=2\pi Mh+(2\pi/9)r\), with
\(r_j=j-4\), as in the original target construction.
The cycle bound \(\lambda_2(L)\ge2/5\) and
\(\cos(7/5)>1/6\) imply
\[
 \langle v-\Theta_1,f(v)-f(\Theta_1)\rangle_D
 \le-\frac1{30}\|v-\Theta_1\|_D^2
 \tag{18}
\]
on this chart. This uses the complete sine Hessian and the \(D=2I\)
metric; it does not identify the actual full dynamics with phase-only
gradient descent.

At the start of this stage every target edge error has magnitude less
than \(7/5+2\pi/9<21/10\). Poincare's inequality therefore gives
\[
 R(0):=\|y(\sigma_*)-\Theta_1\|_D<15,\qquad
 R(v)\le15e^{-v/30}+30\varepsilon,\quad 0\le v\le128 .
 \tag{19}
\]
The second inequality follows by applying (18) to the actual row
\(y'=f(y)+e\), with \(\|e\|_D\le\varepsilon\).
For example, the squared initial bound is at most
\(5\cdot9(21/10)^2<15^2\).

The positive Taylor sum through degree twelve proves
\(e^{64/15}>70\). Thus
\[
 R(128)<15/70+30(5\,10^{-6})<0.215 .
 \tag{20}
\]
The full actual state, including its form, now satisfies
\[
 \begin{aligned}
 H-H_1^*
 &\le\|y-\Theta_1-z\|_D^2+\|z\|_D^2/\gamma^2\\
 &\le [R(128)+Z]^2+Z^2/\gamma^2<1/20,
 \qquad H_1^*=9[1-\cos(2\pi/9)] .
 \end{aligned}\tag{21}
\]
Target stationarity and the global phase-Hessian upper bound \(L\)
give the phase estimate; the form term follows from
\(\lambda_{\max}(L)\le4\) and \(z=\gamma x\).
For the final coarse inequality one may use
\(R+Z<0.216\), \(Z<2.5\,10^{-6}\) and \(\gamma>1/3300\).

The [acute winding-one boundary barrier](SINE_PREPARATION_PATH_SELECTION.md#sine-preparation-path-capture)
is \(B_1=9-8\cos(3\pi/16)\). The same outward scalar arithmetic gives
\[
 H_1^*<2.106,\qquad B_1>2.348,\qquad
 H<2.156<B_1,\qquad B_1-H>0.192 .
 \tag{22}
\]
Thus every actual source is inside the acute winding-one cell with
complete storage below its full boundary cost. Storage nonincrease
traps it there. The only critical relative phase geometry in that
cell is the regular winding-one target, so the full-state LaSalle
argument gives convergence and permanent winding \(+1\).

<a id="sine-path-capture-deadline-and-scope"></a>
## A common deadline and the resulting source-specific selection

Every member reaches the storage handoff by slow elapsed time
\(\sigma_*+128<136\). Once trapped, it remains trapped.
Consequently the entire original uncertain corridor is captured by
the common structural and original-time bounds
\[
 \tau_{\rm cap}=200+\frac{136}{\eta},\qquad
 t_{\rm cap}=\frac{1024}{1023}
                  \left(200+\frac{136}{\eta}\right).
 \tag{23}
\]
These are finite and conservative structural deadlines. They are not
laboratory times or a speed-of-formation claim. No sampled trajectory
was used to choose them.

The conclusion includes every admitted latent residual, including any
source that could not previously be dismissed by a stable-manifold
volume argument. The earlier 57-state list remains a valid necessary
energy classification; the additional source-local cone and full-state
handoff now exclude its other 56 relative outcomes for this corridor.
Other path intervals, constitutive laws and source classes retain their
separate obligations.

Capture here is entry into a forward-trapped acute identity cell below
its boundary storage. The deadline does not assert the older
\(r_c=1/12\) Euclidean radius, a \(10^{-32}\) recovery radius, a contact
readiness certificate or any subsequent response.
Convergence gives eventual entry into smaller target neighborhoods,
without assigning those stronger postconditions the same deadline.

The proof retains both full state channels, the actual hidden initialization,
both means and all loss from the original preparation. The parameter \(s\)
is a source coordinate, not a time-varying controller.
Failure of the sufficient initial cone, error ceiling, tube or storage
inequalities for another admitted input is an unresolved selector result,
not evidence of a different basin or failed formation.

<a id="sine-path-capture-implementation"></a>
## Implementation and evidence boundary

The private
[capture calculator](../../src/tnfr/physics/_sine_preparation_path_capture.py)
rebuilds the [path source](../../src/tnfr/physics/_sine_preparation_path_selection.py),
the mixed initialization identity and the finite scalar budgets above.
Its [independent controls](../../tests/physics/test_sine_preparation_path_capture.py)
check the actual-source admission, perturbation norms, cone speed and erosion,
the acute contraction and the complete storage handoff.
No incoming verdict or independently prepared slow state replaces a premise.

Only exact algebra and outward scalar inequalities are assessed.
There is no new nonlinear trajectory, time-coefficient producer, frozen
response replay or source sweep. The fixed supplied law's demonstrated
selection does not select a universal fundamental law, autonomous
preparation mechanism or physical identification.

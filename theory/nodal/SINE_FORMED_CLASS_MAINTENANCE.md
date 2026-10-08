# Quantitative maintenance of formed classes under repeated probes

<a id="sine-formed-class-maintenance"></a>

## Prospective return-map claim and frozen protocol

The [formed-class experiment](SINE_PATTERN_DYNAMICS.md#sine-formed-class-response)
admits two actual C9 preparation families, one common supplied phase probe,
a distinct finite form response and recovery afterward. This gate asks for
a quantitative uniform return bound that permits repetition. Qualitative
existence of sufficiently long relaxation is already implied by the earlier
compact capture and asymptotic stability; the new obligation is a fixed
finite dwell, a strict invariant return inclusion and per-cycle response
and work bounds.

Freeze the following before evaluating this certificate:

- Retain exactly the preceding simple unit C9 support, node order `0,...,8`,
  unit capacities, beta one, `e=1023/1024`, `w=1/1024`, and complete law.
  In the scaled clock `tau=e*t`, put `A=L/2`,
  `gamma=1/(1023*pi)` and `f(theta)=S(theta)/2`. Then
  `x'=-A*x+gamma*f(theta)` and `theta'=gamma*A*x`.
- Retain the two original nominally phase-flat sources
  `x_j=k*m*(j-4)`, with `k=(1,2)` and
  `m=(2046/9)*(355/113)**2`. At every node both initial residual
  bounds are `1/10000000000`; their form and continuous phase-lift
  sums vanish separately in each class. The actual conserved means
  remain zero. Their common initial storage budget is `1000000000`;
  their unequal nominal costs are unchanged.
- Preserve the acquisition checkpoint `tau=100`, actual first probe
  time `P=200`, phase increment `delta=1/100`, probe vector
  `q=e_0-(1/9)*1`, readout offset `h=1`, and independent per-readout
  additive error `1/10000000000`. Each event is exactly
  `theta_plus=theta_minus+delta*q`, with form unchanged. No support,
  capacity, coefficient or clock change, state reset, adaptive controller,
  additional disturbance or inverse jump is admitted.
- Define each compact pre-probe set `K_k` on its actual zero-mean leaf by
  `||x||_2<=b_x,k` and `||theta-theta_k^*||_2<=b_theta,k` in the
  retained local lift chart. Its radii are exactly the upper bounds
  `warmup_form_norm_upper_bounds` and
  `warmup_target_phase_radius_upper_bounds` from the preceding frozen
  [complete response report](../../docs/assets/sine_formed_classes/response-v1.json).
  The target lifts remain `theta_k,j^*=k*(2*pi/9)*(j-4)`.
  Rebuild these radii from the original primitive protocol when evaluating;
  an incoming report or cached verdict is not an admission premise.
  Each set has nonempty sixteen-dimensional relative interior and contains
  its entire actual first pre-probe family. Put `K_k^+=J(K_k)` for the
  exact supplied phase jump `J`.
- Fix the common dwell `D=1000000000000` in scaled clock, strictly
  longer than `h`. Apply the same jump at `tau=P+n*D` for every integer
  `n>=0`, and read actual node-zero form at `tau=P+n*D+h`.
  The dwell in the original clock is `1024*D/1023`.
  This deliberately conservative mathematical dwell carries no
  practical-speed or laboratory-time claim.
- Require `Phi_D(J(K_k))` to lie strictly inside the half-radius
  pre-probe set for each class: returned form and phase-error norms
  are respectively less than `b_x,k/2` and `b_theta,k/2`.
  Require the complete intervening state to retain winding `k`, and
  recorded class contrast `Y_2-Y_1>1/10000000` at every declared readout.
  Account for cumulative signed event work and continuous loss separately.
  Indefinite operation is an externally supplied repeated-intervention
  protocol, not finite-budget autonomous maintenance.
- Use the shared exact rational and outward dyadic128/Machin arithmetic.
  The strict Lyapunov method below uses `lambda=1/5`, `Lambda=2`,
  `epsilon=lambda/4`, the common worst-class acute cosine, and the shared
  negative-exponential enclosure with exponent at most `4096`.
  The analytic selection of `D` uses the sufficient rational envelopes
  stated below; no dwell, rate, source, amplitude or precision sweep is
  admitted. Outward rounding must retain strict return margins.
- Pass only if the rebuilt formation and one-probe admissions, both
  uniform half-radius returns, whole-window identity, per-cycle contrast
  and work accounting pass. Otherwise retain the insufficient bound or
  separately proved obstruction without changing this protocol.
  Record evaluated bounds separately from this declaration.

Finite-time recovery here means return to the admitted neighborhood.
Asymptotic convergence to an exact target applies after interventions stop.
An invariant return set does not establish a unique periodic orbit,
autonomous event selection, a fundamental law or physical constituent identity.

## A strict Lyapunov bound on each retained acute chart

All coordinates below are on the mean-zero space. For class \(k\),
write \(y=\theta-\theta_k^*\). Let \(\underline\eta>0\) and
\(\overline\eta\) be the freshly reconstructed rational lower and
upper bounds for \(\eta=\gamma^2\). Choose a rational
\(\underline c>0\) no greater than either
\(\cos(k\alpha+\sqrt2r)\), where \(\alpha=2\pi/9\).
The preceding post-jump certificates place all of \(J(K_k)\) inside
the corresponding compact, forward-trapped acute neighborhood.
They apply uniformly to these sets because their proof consumes only
the two pre-probe norm bounds defining \(K_k\).

The mean-free operator \(A=L/2\) is symmetric positive definite with
\(\lambda I\preceq A\preceq\Lambda I\). Define auxiliary proof
coordinates and a potential by
\[
\xi=A^{-1/2}y,\qquad v=\xi'=\gamma A^{1/2}x,\qquad
W_k(\xi)=\frac{\gamma^2}{2}
 [U(\theta_k^*+A^{1/2}\xi)-U(\theta_k^*)].
\]
They introduce no new physical coordinates or evolution law. Since
\(f=-\nabla U/2\), differentiating the complete original scaled rows
gives exactly
\[
\xi'=v,\qquad v'=-Av-\nabla W_k(\xi).
\]
On the trapped chart, \(2\underline c A\preceq\nabla^2U\preceq2A\).
Consequently the following rational constants bound the transformed Hessian:
\[
\mu=\underline\eta\,\underline c\,\lambda^2>0,\qquad
M=\overline\eta\,\Lambda^2,\qquad
\mu I\preceq\nabla^2W_k\preceq MI.
\]
The interpolation from the target stays in the same convex chart, so
\[
\frac\mu2\|\xi\|^2\le W_k(\xi)\le\frac M2\|\xi\|^2,
\qquad \xi^{\mathsf T}\nabla W_k(\xi)\ge\mu\|\xi\|^2.
\]
These inequalities concern the actual nonlinear phase potential.
Positive tangent eigenvalues alone are not substituted for them.

Set \(\varepsilon=\lambda/4\) and use the modified energy
\[
\mathcal V_k(\xi,v)
 =\frac12\|v\|^2+W_k(\xi)
  +\varepsilon\xi^{\mathsf T}v
  +\frac\varepsilon2\xi^{\mathsf T}A\xi.
\]
Young's inequality
\(\varepsilon|\xi^{\mathsf T}v|
\le\|v\|^2/4+\varepsilon^2\|\xi\|^2\) gives
\[
\begin{aligned}
a_-&=\frac\mu2+\frac{\varepsilon\lambda}{2}-\varepsilon^2
     =\frac\mu2+\frac{\lambda^2}{16}>0,\\
a_+&=\frac M2+\frac{\varepsilon\Lambda}{2}+\varepsilon^2,\\
\frac14\|v\|^2+a_-\|\xi\|^2
 &\le\mathcal V_k
 \le\frac34\|v\|^2+a_+\|\xi\|^2.
\end{aligned}
\]
The two mixed damping terms cancel on differentiation:
\[
\begin{aligned}
\mathcal V_k'
 &=-v^{\mathsf T}(A-\varepsilon I)v
    -\varepsilon\xi^{\mathsf T}\nabla W_k(\xi)\\
 &\le-(\lambda-\varepsilon)\|v\|^2
       -\varepsilon\mu\|\xi\|^2
 \le-\kappa\mathcal V_k,\\
\kappa&=\min\left(
 \frac{4(\lambda-\varepsilon)}3,
 \frac{\varepsilon\mu}{a_+}\right)>0.
\end{aligned}
\]
Both classes therefore have the same proved rate bound
\(\mathcal V_k(s)\le e^{-\kappa s}\mathcal V_k(0)\)
through any uninterrupted post-probe relaxation interval. The chart
assumption is supplied by the independently retained trapping barriers;
it is not inferred from this rate inequality after leaving the chart.

## A computable half-radius return and the chosen dwell

Write \(b_{x,k},b_{\theta,k}\) for the exact rational radii defining
\(K_k\), and let \(b_{\theta,k}^+\) be their retained upper bound
after the jump. Thus
\(b_{\theta,k}^+\ge b_{\theta,k}+\delta\sqrt{8/9}\).
For every state in \(J(K_k)\),
\[
\|v(0)\|^2\le\overline\eta\Lambda b_{x,k}^2,
\qquad
\|\xi(0)\|^2\le\frac{(b_{\theta,k}^+)^2}{\lambda}.
\]
It follows that
\[
V_{0,k}=\frac34\overline\eta\Lambda b_{x,k}^2
       +\frac{a_+}{\lambda}(b_{\theta,k}^+)^2
\]
is a uniform upper bound for \(\mathcal V_k(0)\).
Let \(\rho_D\ge e^{-\kappa D}\) be the outward exponential upper
bound actually used in the calculation. Since
\(\|v\|^2\ge\underline\eta\lambda\|x\|^2\) and
\(\|y\|^2\le\Lambda\|\xi\|^2\), valid returned squared-norm bounds are
\[
X_k^2=\frac{4V_{0,k}\rho_D}{\underline\eta\lambda},\qquad
Y_k^2=\frac{\Lambda V_{0,k}\rho_D}{a_-}.
\]
The required strict half-radius admission is exactly
\[
\frac{b_{x,k}^2}{4}-X_k^2>0,\qquad
\frac{b_{\theta,k}^2}{4}-Y_k^2>0.
\]
Equivalently, the following energy target is sufficient before the
outward endpoint checks:
\[
Q_k=\min\left(
 \frac{\underline\eta\lambda b_{x,k}^2}{16},
 \frac{a_-b_{\theta,k}^2}{4\Lambda}\right),
\qquad V_{0,k}\rho_D<Q_k.
\]
The strict margins use the full coordinates. A small phase displacement
alone cannot supply the form return or repeat the response certificate.

The following coarse rational envelopes of the preceding frozen
one-probe bounds motivate the fixed dwell without a horizon search:
\[
\begin{gathered}
9\times10^{-8}<\underline\eta\le\overline\eta<10^{-7},\qquad
\underline c>1/20,\\
10^{-7}<b_{x,k}<10^{-6},\qquad
10^{-5}<b_{\theta,k}<10^{-4},\qquad
\|q\|<1.
\end{gathered}
With \(\lambda=1/5\), \(\Lambda=2\) and \(\varepsilon=1/20\),
these give
\[
\mu>1.8\times10^{-10},\quad
a_->1/400,\quad a_+<53/1000,\quad
\kappa>10^{-10},\quad V_{0,k}<10^{-3},\quad Q_k>10^{-23}.
\]
For example, the form part of \(Q_k\) exceeds
\(9\times10^{-22}/80=1.125\times10^{-23}\); its phase part
exceeds \(10^{-10}/3200\). These are sufficient design bounds,
not fitted response values. They are independently checked from the
freshly reconstructed premises when certifying this frozen instance.

At \(D=10^{12}\), the exact exponential is less than \(e^{-100}\),
which is less than \(10^{-40}\). One elementary verification is
\(e^5>\sum_{j=0}^6 5^j/j!>100\), followed by the twentieth power.
Thus the exact returned modified energy is below \(10^{-43}\),
well inside both sufficient targets. Also
\(\kappa D<80000/21<4096\), using \(\underline c\le1\),
\(\overline\eta<10^{-7}\) and \(a_+>21/400\), so the shared
exponential work limit is respected. The interval implementation still
checks its own actual outward return margins: rounding an extremely
small decay to a dyadic128 upper endpoint does not preserve every
decimal bound on the exact exponential. No such rounded upper endpoint
is silently replaced by the exact exponential.

## Induction over the actual repeated experiment

The original full-law warmup places every first pre-probe state in
\(K_k\). The existing probe certificate applies uniformly to all of
\(K_k\), because its response-error, work and post-jump bounds use
only \(b_{x,k}\) and \(b_{\theta,k}\). It does not require a
particular warmup history once this state-set admission has been proved.

The post-jump barrier first places every state of \(J(K_k)\) in
its class's forward-trapped acute region. The strict return margins
then give
\[
\Phi_D(J(K_k))\subset\tfrac12K_k\subset\operatorname{int}_{\rm rel}K_k,
\]
where \(\tfrac12K_k\) means halving both radii about the same target
and zero form. This yields the equivalent post-probe inclusion
\(J\circ\Phi_D(K_k^+)\subset K_k^+\), with \(K_k^+=J(K_k)\).
Induction applies the identical jump, uninterrupted flow interval and
readout bound at every cycle. Winding is retained throughout each
interval and across every admitted jump. No exact target reset,
state-dependent waiting rule or new source error is inserted.

Each cycle uses the same two-class receiver comparison and deterministic
heat factor at offset \(h=1\). The original preparation uncertainty is
propagated by the invariant sets; readout errors remain independent
bounded observations and do not drive the state. The common contrast
bound therefore applies at every cycle without assuming error
cancellation or accumulating a new independent state box at each step.
After a last intervention, the unchanged positive-loss law gives
asymptotic recovery to the corresponding target. During indefinite
repetition only neighborhood return and geometric identity are asserted;
no exact equilibrium or unique periodic orbit is claimed.

## Cumulative work and the boundary of conditional maintenance

For each actual event the supplied work remains exactly
\[
W_{k,n}=U(\theta_{k,n}^-+\delta q)-U(\theta_{k,n}^-).
\]
The one-probe work enclosure is uniform on \(K_k\), so it bounds every
\(W_{k,n}\). With original event times \(t_n=(P+nD)/e\), the exact
balance across \(N\) complete cycles is
\[
H_k(t_N^-)-H_k(t_0^-)
 =\sum_{n=0}^{N-1}W_{k,n}
  -\frac e2\sum_{n=0}^{N-1}
       \int_{t_n}^{t_{n+1}}\|Lx_k(t)\|_2^2\,dt.
\]
The values at the finitely many jumps do not affect the continuous
integrals. If \([w_k^-,w_k^+]\) is the retained per-probe work interval,
then supplied work for \(N\) probes lies in
\([Nw_k^-,Nw_k^+]\). Its positive lower endpoint in the frozen
protocol implies unbounded cumulative supplied work for indefinitely
many repetitions. State storage remains bounded in the trapped charts;
continuous loss accounts for the difference, not an invented reservoir.

This is quantitative maintenance under an explicitly powered schedule.
The original \(10^9\) budget concerns preparation, not an unlimited
probe resource. A finite external work allowance would require its own
finite-cycle admission. No practical operating speed, autonomous event
selection, physical identification or independently justified clock
bridge follows from this return-map certificate.

## Retained quantitative return certificate

The frozen protocol passes without changing its dwell or premises.
The [complete evaluated report](../../docs/assets/sine_formed_classes/maintenance-v1.json)
retains the exact rate, pre/post-probe bounds, prerequisite evidence and
return margins. Its [protocol record](../../docs/assets/sine_formed_classes/maintenance-v1.protocol.json)
and [source archive](../../docs/assets/sine_formed_classes/maintenance-v1.source.zip)
were captured before evaluation, including the proof derivation above.
The [evidence manifest](../../docs/assets/sine_formed_classes/maintenance-v1.manifest.json)
records the committed base and producing source overlay separately from
subsequent result documentation.

The common rate lower bound is approximately
\(2.0894361086775972\times10^{-10}\) per scaled time unit.
At the fixed \(D=10^{12}\), the retained outward decay interval is
exactly \([0,2^{-128}]\). Its zero lower endpoint is an enclosure
artifact, not exact finite-time equilibration. With \(B=2^{128}\),
the following retained margin intervals are \([N,N+1]/B\):

| Strict return margin | Lower numerator \(N\) |
| --- | ---: |
| Modified-energy target, winding one | `14574548981967538` |
| Modified-energy target, winding two | `15020686919729160` |
| Quarter form-radius squared, winding one | `3010759966127372195420062` |
| Quarter form-radius squared, winding two | `3102921599673989472552128` |
| Quarter phase-radius squared, winding one | `310975950929306447248182251052` |
| Quarter phase-radius squared, winding two | `320494915300103012563382764566` |

Both form and phase half-radius returns are strict. Therefore every
admitted source member can undergo every declared repetition while
retaining its class winding. The rebuilt per-cycle recorded contrast
keeps the preceding lower bound above \(3.4\times10^{-7}\), exceeding
the frozen \(10^{-7}\) requirement at every cycle. The separately
retained positive work intervals apply to each intervention and scale
linearly as bounds on cumulative supplied work.

The primitive-only
[`assess_sine_formed_class_maintenance`](../../src/tnfr/physics/relational_sine_formed_class_maintenance.py)
returns `SineFormedClassMaintenance`. It rebuilds formation, the first
probe, the pre-probe radii and the post-jump trapping premises from the
original scalar inputs. Return bounds require those trapping premises;
an acute cosine by itself cannot certify an out-of-chart trajectory.
The exponential work limit consumes the proved slow rate times the
dwell, rather than the initial fast semigroup gap. The
[contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-formed-class-maintenance),
[guide](../../docs/guides/relational/SINE_PATTERNS.md#sine-formed-class-maintenance)
and [independent controls](../../tests/physics/test_sine_formed_class_maintenance.py)
retain the execution boundary and unavailable cases.

This certificate proves a uniform return inclusion for the full
nonlinear state sets. It was not obtained by sampling a long pulse
train. It supplies no practical-speed estimate, finite total work
budget, autonomous intervention mechanism or physical identification.

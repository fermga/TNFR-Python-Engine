# Cancellation-preserving form tracking for composed sine ports

<a id="sine-port-form-tracking"></a>

## Prospective claim and frozen protocol

The [all-time relaxation certificate](SINE_PORT_RELAXATION.md#sine-port-relaxation)
admits uniform error envelopes for the formed C9 network, but its frozen
natural-scale test returned phase-only resolution. The conservative form
upper bound did not meet its allowance. That result is retained unchanged;
it did not establish an actual tracking failure.

This gate changes only the form-error estimation method. It derives a
time-domain convolution gain that preserves a cancellation lost by the
previous comparison rectangle. The phase envelope and its premises remain
valid prerequisites, rather than being fitted again.

Freeze the following before the new reserved assessment:

- Retain the three simple unit C9 components, classes \((1,2,1)\),
  central contacts \((0,1),(1,2)\), and phase origins
  \((0,1/1000,0)\). Retain the original unit capacities, beta one,
  \(e=1023/1024\), \(w=1/1024\), \(\tau=et\), and
  \(\gamma=1/(1023\pi)\). Both complete rows, all contact degrees,
  and the nonlinear bridge sine remain unchanged.
- Retain each original source
  \(k_i(2046/9)(355/113)^2(j-4)\), phase-flat initialization at
  its supplied origin, independent per-coordinate form and continuous
  phase-lift error bounds \(10^{-10}\), and the separate exact
  zero-sum constraints. All source coordinates and each component's
  original storage ceiling \(10^9\) remain admitted.
- Retain formation at \(\tau=100\), radius \(1/12\), the unprobed
  dwell \(10^{13}\), and actual endpoint form/phase norm allowances
  \(\epsilon=10^{-32}\). Rebuild the handoff from primitive inputs,
  retaining exact \(2^{-512}\) only under
  \(512\le\kappa D\le4096\). Supply both contacts simultaneously,
  without state reset, under total event-work allowance \(2\,10^{-6}\).
- Retain the same thirty-coordinate degree-aware surrogate and all
  fifty-four actual fine coordinates. Compare the full and lifted
  surrogate flows throughout the entire subsequent uninterrupted
  interval \([0,\infty)\). The nominal surrogate initialization
  does not replace actual source uncertainty or unknown conserved means.
- Freshly rebuild every admitted all-time relaxation bound: both acute
  charts, original actual handoff, event work, global mean allowances,
  the normalized even gap \(3/100\) checked by exact PSD, the odd
  gap \(7/30\), curvature floor \(1/6\), parity defects and initial
  comparison rectangles. Do not accept a cached incoming report or its
  prior phase-only verdict as a premise.
- Replace only the form output estimate by the fixed-stiffness heat
  convolution and finite-prefix feedback estimate proved below. Bound
  the time-dependent bridge-Hessian departure explicitly using the
  existing even phase-error envelope. For either parity sector, use
  the least nonnegative integer \(n\) with
  \(2^n(\lambda/6)\ge2\), giving rational filter factor
  \(G=2+21n/80\). The frozen gaps give \(n_o=6\) and \(n_e=9\).
  No operator commutation, fitted impulse response or forcing-derivative
  bound is assumed.
- Keep both resolution fractions equal to \(1/2\). With unchanged
  origin span \(\varphi=1/1000\), require a uniform phase-coordinate
  error strictly below \(\varphi/2\) and a uniform form-coordinate
  error strictly below \(\gamma\varphi/2\). Use the outward lower
  form allowance and positive outward lower margin. The phase test is
  retained independently; a form bound cannot hide a failed phase test.
- Preserve this protocol, producing source and derivation before the
  first reserved assessment. Use exact rational primitives and the
  shared fixed outward interval arithmetic. Do not integrate a
  trajectory, alter a horizon/source/threshold, search parameters or
  regenerate the previous frozen result. Report separate prerequisite,
  phase and form outcomes, including a partial or unavailable outcome
  if either unchanged criterion remains uncertified.

The long dwell and tiny source/endpoint allowances remain supplied
mathematical premises. A sharper bound does not supply a laboratory clock,
measurement model, physical identification or new occurrence law.

## Comparison on the same centered mean leaves

Reuse the definitions and proofs of the
[relaxation owner](SINE_PORT_RELAXATION.md#same-complete-trajectories-weighted-means-and-two-invariant-charts).
Write \(\mathsf D\) for actual fine degree mass, \(A=KL\), and
use the weighted norm \(\|v\|_{\mathsf D}\). Center only the
global conserved actual-minus-nominal form and phase differences. Their
possible constant offsets remain bounded by
\[
\mu_* = \frac{2|E|\epsilon}{18m+2|E|}, \tag{1}
\]
and will be restored at the final coordinate level. This is an allowance
for uncertainty, not a strictly positive lower bound on actual error.

For each parity sector, the already derived exact difference equations are
\[
\delta x'=-A\delta x-\gamma B_t\delta u+\gamma d,
\qquad \delta u'=\gamma A\delta x. \tag{2}
\]
The operators are restricted to the relevant centered parity subspace.
The full and surrogate trajectories are each trapped in their admitted
acute charts. Their edge-energy bound is \(B_{\mathrm{edge}}^2=12E_H\),
where \(E_H\) is the freshly rebuilt actual initial storage-excess
upper bound. The existing parity forcing and phase bounds are
\[
\|d_o\|_{\mathsf D}\le D_o,\quad
\|d_e\|_{\mathsf D}\le D_e,\quad
\|\delta u_e\|_{\mathsf D}\le P_e=Y_e+Z_e. \tag{3}
\]
These are the old theorem's uniform bounds, not newly measured or fitted
values. Its initial bounds \(N_o,N_e\) bound both centered initial
form and phase norms in their respective sectors. The old phase envelope
continues to include all initial and generated odd effects.

Define the constant comparison operator
\[
B_0=K\left(\sum_i c_iL_{{\rm cycle},i}+L_{\rm contacts}\right),
\qquad c_i=\cos(2\pi k_i/9).
\]
It is the class-linear internal stiffness with unit bridge stiffness at
the aligned critical geometry. It is self-adjoint in the degree metric
and preserves both parity sectors and their mean-free leaves. If the
sector's admitted diffusion gap is \(\lambda\), then
\[
bI\preceq B_0\preceq 2I,\qquad b=\lambda/6>0. \tag{4}
\]
This is only a comparison operator. The evolving surrogate still has
its original exact nonlinear bridge; that bridge has not been linearized
in the model or discarded from the error.

### The bridge departure remains a bounded forcing

The internal entries of \(B_t\) and \(B_0\) agree. Each bridge
entry of \(B_t\) is the averaged cosine between the full and surrogate
gaps. Both endpoint gaps and the intervening segment have magnitude at
most \(B_{\mathrm{edge}}\). Since \(1-\cos s\le s^2/2\),
\[
0\preceq B_0-B_t
 \preceq \tfrac12 B_{\mathrm{edge}}^2 K L_{\rm contacts},
\qquad
\|B_0-B_t\|_{\mathsf D}\le B_{\mathrm{edge}}^2. \tag{5}
\]
The last inequality uses \(K L_{\rm contacts}\preceq A\) and
\(\|A\|_{\mathsf D}\le2\). This bridge operator annihilates
odd vectors because each odd port value is zero. It also annihilates
constant vectors, so no mean allowance is inserted in (5).

Rewrite (2), exactly, with \(B_0\) and the effective forcing
\(d_* = d+(B_0-B_t)\delta u\). The uniform forcing allowances are
therefore
\[
D_{*,o}=D_o,\qquad
D_{*,e}=D_e+B_{\mathrm{edge}}^2P_e. \tag{6}
\]
The second term retains the time dependence of the actual bridge
Hessian. Equation (6) uses the already proved phase bound and makes no
assumption that the forcing is constant, slowly varying or differentiable.

## A rational time-domain gain for a positive heat operator

Let a positive self-adjoint operator \(B\) have spectrum in
\([b,\ell]\), with \(0<b\le\ell\). The spectral theorem gives
\[
\|B\exp(-Bt)\|
 \le\sup_{b\le s\le\ell}s\exp(-st).
\]
On the three time ranges separated by \(1/\ell\) and \(1/b\),
the latter is respectively
\(\ell\exp(-\ell t)\), \(1/[\exp(1)t]\), and
\(b\exp(-bt)\). Their integrals yield
\[
H(B):=\int_0^\infty\|B\exp(-Bt)\|\,dt
 \le 1+\frac{\log(\ell/b)}{\exp(1)}. \tag{7}
\]
This estimate is for the operator norm over the whole spectrum; it does
not assume that one eigenvector supplies that norm at every time.

Take \(\ell=2\) and let \(n\) be the least nonnegative integer
with \(2^n b\ge2\). The elementary strict bounds
\(\log2<7/10\) and \(1/\exp(1)<3/8\) imply
\[
H(B_0)\le1+21n/80,
\qquad G:=2+21n/80\ge1+H(B_0). \tag{8}
\]
For completeness, the exponential series through degree three at
\(7/10\) already exceeds two, while its series at one exceeds
\(1+1+1/2+1/6=8/3\). Thus (8) needs neither a numerical logarithm
nor a fitted spectral partition. The exponential constant here is
\(\exp(1)\), not the model's loss parameter \(e=1023/1024\).

For any positive \(\eta\), rescaling time preserves the integral
\(\int_0^\infty\|\eta B_0\exp(-\eta B_0t)\|dt\).
Consequently the causal filter
\[
\mathcal T_{B_0}=\delta I-\eta B_0\exp(-\eta B_0t)
\]
has induced bounded-input norm at most \(G\). Here \(\delta I\)
denotes the instantaneous identity term in the convolution filter, not
an additional source or a state jump. The subtraction is the cancellation
that the earlier pointwise norm rectangle did not preserve.

## Ordered heat convolutions and the initial-state terms

Put \(\eta=\gamma^2\), \(z=\gamma\delta x\), and
\(y=\delta u+z\), as before. With the exact forcing (6), equation
(2) becomes
\[
y'=-\eta B_0y+\eta p,\qquad z'=-Az+y',
\qquad p=B_0z+d_* . \tag{9}
\]
Let \(H_A(t)=\exp(-At)\) and
\(H_{\eta B_0}(t)=\exp(-\eta B_0t)\). Solving the first row
and then the second gives the exact identity
\[
\begin{aligned}
z(t)={}&H_A(t)z(0)
 -\eta\bigl(H_A*(B_0H_{\eta B_0})\bigr)(t)y(0)\\
 &+\eta\bigl(H_A*\mathcal T_{B_0}*p\bigr)(t).
\end{aligned} \tag{10}
\]
All products retain the displayed order. In particular, (10) does not
interchange \(A\) with \(B_0\), their semigroups, or their spectral
projections. No simultaneous diagonalization is assumed. The two initial
terms are retained; a zero nominal surrogate does not make actual initial
form and phase errors vanish.

On the centered sector, \(\|H_A(t)\|\le\exp(-\lambda t)\),
so its integral norm is at most \(1/\lambda\). Also
\(\|B_0H_{\eta B_0}(t)\|\le2\). For any finite \(T\), let
\(Z_T=\sup_{0\le t\le T}\|z(t)\|_{\mathsf D}\). If both
initial form and phase norms are at most \(N\), then
\(\|z(0)\|\le\gamma N\),
\(\|y(0)\|\le(1+\gamma)N\), and (8)--(10) imply
\[
Z_T\le\gamma N+
 \frac{\eta}{\lambda}\left[2(1+\gamma)N+
                          G(2Z_T+D_*)\right]. \tag{11}
\]
The estimate uses only bounded forcing on that finite prefix. If
\[
\rho=\frac{2\gamma_+^2G}{\lambda}<1, \tag{12}
\]
it closes uniformly in \(T\). Taking arbitrary finite prefixes then
gives the whole-future result; prior boundedness of the desired tracking
error was not assumed in order to solve the feedback inequality.

Divide the exact-gamma inequality by \(\gamma>0\) before applying
interval endpoints. Every numerator term and the reciprocal denominator
is monotone in positive \(\gamma\) on the admitted range. Therefore
the new weighted form-error bound is
\[
\boxed{\mathcal F(\lambda,D_*,N)=
 \frac{N+\dfrac{\gamma_+}{\lambda}
       [2(1+\gamma_+)N+GD_*]}
      {1-2\gamma_+^2G/\lambda}.} \tag{13}
\]
No division by a rounded tiny scaled-form output is needed. This also
avoids introducing \(\gamma_-\) solely to undo a cancellation that
is exact in the proof. A failed condition (12) leaves this particular
gain unavailable; it is not an instability verdict for the original
flow or surrogate.

## Restored raw-coordinate bounds and separate resolution

For each parity use its freshly rebuilt gap, initial norm and (6):
\[
F_o=\mathcal F(7/30,D_o,N_o),\qquad
F_e=\mathcal F(q,D_e+B_{\mathrm{edge}}^2P_e,N_e). \tag{14}
\]
The same general proof covers both sectors, including a noncommuting
even \(A,B_0\). It needs no separate scalar-mode assumption for the
odd blocks. Orthogonality in the degree metric and minimum fine degree
two give the uniform raw form-coordinate error
\[
\boxed{\mathcal E_x^{\rm heat}
 =\mu_*+\frac{\sqrt{F_o^2+F_e^2}}{\sqrt2}.} \tag{15}
\]
The old uniform phase envelope is retained exactly. Both (15) and that
phase bound hold for every admitted actual source member throughout the
same future interval. Use conservative upper square roots and a lower
\(\sqrt2\) for numerical division; preserve (1) once, separately
from the centered norms.

The public certificate rebuilds the baseline from the original primitive
law, source, support and policy inputs. It does not accept the previous
report or its cached derived fields as evidence. The new method needs its
baseline envelopes and both gains (12), then tests the unchanged separate
phase and form allowances. It preserves any partial outcome without
claiming that failure of an upper-bound test proves an actual excess.

This is a sharper conditional trajectory estimate, not a modified model,
new source, changed precision policy or selected physical constitutive
law. The earlier phase-only frozen protocol and report remain valid
evidence of their own specified method and stopping rule.

## Status at archival

At source archival, the prospective protocol and derivation above were
complete and the reserved assessment had not been performed. The following
result was added after that archived declaration and the first evaluation.

## Retained evaluation of the unchanged protocol

The first reserved assessment returned `full`: the all-time envelopes were
admitted, and both separate resolution tests passed. The source, preparation,
contact graph, origin offsets, long relaxation dwell and two allowances were
unchanged. The recorded upper bounds are:

| Channel | Uniform coordinate-error upper bound | Unchanged allowance |
| --- | ---: | ---: |
| Phase | \(2.652351894767271\times10^{-4}\) | \(5\times10^{-4}\) |
| Form | \(5.656486865722837\times10^{-8}\) | \(\gamma/2000\approx1.555766794642183\times10^{-7}\) |

The decimals are summaries of rational bounds. The exact outward margin
intervals, with the common denominator \(2^{128}\), are:

| Allowance minus error bound | Lower numerator | Upper numerator |
| --- | ---: | ---: |
| Phase | `79886325394604946773165096851134274` | `79886325394604946773165096851134275` |
| Form | `33691973334530691648494388925818` | `33691973334530691648494388925820` |

Both lower endpoints are strictly positive. The form margin is approximately
\(9.901181080698994\times10^{-8}\). The odd and even heat bounds use the
prospective band counts six and nine, respectively, giving derivative-filter
gains \(143/40\) and \(349/80\). Their feedback-loop bounds are below one.

The nested baseline is exactly the previously retained `phase_only` report,
including its unsuccessful form-resolution test. Its phase bound and phase
margin are unchanged. The new conclusion follows from the ordered heat-filter
estimate (9)--(15); it does not alter that earlier method's recorded outcome
or supply a new observed trajectory. Initial even and odd errors and the
possible constant global-mean offsets remain included. The reported mean-error
floors are upper allowances for these offsets; they are not positive lower
bounds on the actual error of every source member.

These are conditional all-time coordinate-error bounds for the admitted full
and lifted-surrogate flows, with the acute-chart and identity premises supplied
by the fresh baseline. They do not identify a physical law, validate a sensor
bridge or make the unchanged relaxation dwell \(10^{13}\) operationally
practical. No new trajectory, laboratory observation or source calibration is
part of this evaluation.

The retained [protocol](../../docs/assets/sine_formed_classes/port-form-tracking-v1.protocol.json),
[source archive](../../docs/assets/sine_formed_classes/port-form-tracking-v1.source.zip),
[complete response](../../docs/assets/sine_formed_classes/port-form-tracking-v1.json)
and [evidence manifest](../../docs/assets/sine_formed_classes/port-form-tracking-v1.manifest.json)
preserve the first evaluation. The archive contains the complete pre-evaluation
owner, including its then-prospective status. After archival, only the
publication notation in (10) was normalized to avoid false Markdown links;
the operator order, mathematical derivation and protocol are unchanged.
Its SHA-256 is
`82ba552d0d8008bd22a3a7cdeed02db2f7053eb3b678b0bb316a463ccf68f092`.

The shared implementation is
[`SinePortFormTracking` / `assess_sine_port_form_tracking`](../../src/tnfr/physics/relational_sine_port_form_tracking.py).
The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-port-form-tracking),
[usage guide](../../docs/guides/relational/SINE_PATTERNS.md#sine-port-form-tracking)
and [independent controls](../../tests/physics/test_sine_port_form_tracking.py)
cover original-input admission, noncommuting ordered convolution, initial-state
terms, bridge variation, separate channel policies and preservation of the
saved baseline. The controls support implementation fidelity; the all-time
claim depends on the hypotheses and derivation above.

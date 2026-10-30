
<a id="sine-formed-two-port-entry-method"></a>
## A source-aware reference method for the actually formed pair

The [formed-source admission](SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-admission)
fixes the actual formation images, simultaneous contacts, supplied work,
internal identity and conditional response. The missing implication is entry
of that source into the complete-state basin. This section selects a method
and prospective numerical policy for that implication. It does not report
a new reference trajectory, actual-source capture or response assessment.
The earlier prepared-source proof and its frozen result remain unchanged.

### Retain the actual source while removing numerical stiffness

Use the original labels of (A1)--(A3) in the admission owner: two C9 cycles,
ports `(D4,R4)` and `(D5,R5)`, joined degree matrix `D` of mass forty,
`alpha=2*pi/9`, `phi=1/1000`, `epsilon=10^-32` and
`gamma=1/(1023*pi)`. The contact occurs at `tau_c=100+10^13`.
Thereafter the complete rows are still
`x_tau=-D^-1 L x+gamma D^-1 S(theta)` and
`theta_tau=gamma D^-1 L x`, with held unit capacities and no further events.
Define elapsed slow time `sigma=gamma^2*(tau-tau_c)`. This translation of
the origin changes no row or preparation; it is not an absolute clock reset.

Let `theta^0` be the nominal part of (A1), with the actual relative origin
`phi`, and remove its single joined mean
`m0=phi/2+3*alpha/40`. Define an auxiliary **full eighteen-phase** reference

\[
\psi_\sigma=D^{-1}S(\psi),\qquad
\psi(0)=\theta^0-m_0\mathbf1. \tag{M1}
\]

The reference has a different phase law from the actual complete system.
Its use is justified by the [energy-informed comparison](SINE_TWO_PORT_TRANSIT.md#sine-energy-informed-slow-comparison),
which keeps `z=gamma P_D x`, `y=theta+z` and reconstructs the original form.
It does not approximate the actual source by zero form or discard its thirty-six
coordinates, correlations or conserved means. Bounds below are uniform over
the actual isolated-flow image family; they introduce no probability measure
or independently preparable endpoint box.

Use affine radian deviations `xi=psi-psi(0)`, initially exactly zero.
Orient each cycle `j->j+1` including its closing edge, and contacts donor to
receiver; let `B` have one row `e_right-e_left` for each of these twenty edges.
The fixed nominal edge vector `g0` has donor entries `2*alpha`, receiver
entries `alpha`, and contact entries `phi,phi-alpha`. The closing edges
retain their inherited `+4*pi` and `+2*pi` offsets. Then all reference rows are

\[
\xi_\sigma=-D^{-1}B^T\sin(g^0+B\xi). \tag{M2}
\]

Mathematical pi and rational `phi` stay in these affine coefficients; outward
interval arithmetic encloses their evaluation, rather than supplying uncertain
physical constants. No repeated wrapping, component recentering or old
reflection folding is used. The weighted common mode of the exact reference
is conserved. Retain the full metric `D`, including any numerical common-mode
error, instead of projecting an enclosure onto a smaller state without proof.

On every convex tube where all twenty fixed-branch gaps are acute,
the derivative is `-D^-1 B^T diag(cos(g)) B`. Its logarithmic growth bound
in the degree norm is zero. Thus the existing retained-metric Taylor kernel
applies directly in dimension eighteen, within its current dimension limit.
The kernel's projected coordinate boxes are for inclusion and derivatives;
they must not replace the retained metric ball.

### New source bounds close the complete-law comparison

The four source residual vectors in (A1) each have Euclidean norm at most
`epsilon`. Since joined degrees are at most three, and weighted mean removal
is an orthogonal projection,

\[
\|P_Dx(0)\|_D,\ \|P_Du(0)\|_D
 \le\sqrt6\,\epsilon<3\epsilon. \tag{M3}
\]

For each actual member, translate (M1) by its conserved phase mean. This
changes no reference gap or equation. Its initial mixed-coordinate errors
therefore obey `Z0=3*epsilon/3069` and `v0=3*epsilon+Z0`, using the existing
`1/3216<gamma<1/3069`. No component mean is separately held fixed.

In the acute `(2,1,0)` cell, Jensen's inequality gives the phase floor

\[
U_{\rm floor}=9[1-\cos(4\pi/9)]+9[1-\cos(2\pi/9)].
\]

At the source, the two internal form-storage contributions sum to at most
`4*epsilon^2`, since each cycle Laplacian has norm at most four. Criticality
of each isolated twist cancels the linear phase term. The Hessian upper bound
by that cycle Laplacian similarly bounds their combined phase excess by
`4*epsilon^2`. Adding the actual contact work bound (A5) gives

\[
H(0)-U_{\rm floor}
 <49/200+1/500000+10\epsilon^2<1/4=:E_0. \tag{M4}
\]

This includes the supplied event work; it is not the old midpoint budget
`1/8`, nor is previous dissipation treated as a contact reserve.
Only the four ports have nonzero nominal reference rates. Hence

\[
\|f(\psi(0))\|_D^2
 =\frac23[\sin^2\varphi+\sin^2(\varphi-\alpha)]
 <\frac23[(1/500)^2+(7/10)^2]<(3/5)^2. \tag{M5}
\]

Take `F0=3/5`, `lambda=1/90`, `Lambda=2` and
`eta_bar=1/9000000`. For the prospective horizon `Sigma=1025`,
`sqrt(Sigma*E0)<17`. The existing comparison therefore uses the new bounds

\[
\begin{aligned}
Z_0&=3\epsilon/3069,&v_0&=3\epsilon+Z_0,\\
V&=v_0+17/50000,\\
Z&=\frac{Z_0+10^{-5}(3/5+2V)}{1-1/50000},&Q&=V+Z.
\end{aligned} \tag{M6}
\]

Exact rational inequalities give

\[
V<7/20000,\qquad Z<1/100000,\qquad
Q<9/25000<\rho:=1/2048. \tag{M7}
\]

These bounds initially hold only before a possible chart exit. A validated
reference with whole-tube edge margins strictly above `rho` closes the same
strict first-exit argument for both actual `theta` and mixed phase `y`.
The old stronger global form bound `Z<1/200000` is not imported.

There is also a useful analytic start. The nominal reference's minimum initial
acute margin exceeds `1/6`. Before an exit, its degree-norm speed cannot
increase, so each edge changes by at most `F0*sigma`. On `0<=sigma<=1/4`,
the remaining margin exceeds `1/6-3/20=1/60>rho`. This proves that the exact
reference starts in a viable chart. It does not guarantee Picard inclusion
of a particular computed tube or acuteness for the full long interval.

### The same endpoint obligations imply full-state entry

Select `T=1024` as a prospective reference endpoint. Require every whole-time
reference tube through `T` to have margin above `rho`, and require
`||psi(T)-theta_*||_D<=rho`, including the numerical metric radius and
the correlated target enclosure. The target is the same exact compatibility
root, transported into the original labels and continuous lifts, then centered
by one joined weighted mean. The old nominal midpoint phase is never substituted
for (M1), even when its target geometry is reused.

The [analytic final-unit and local-basin proof](#sine-two-port-capture-handoff)
then transfers with the new `Q,Z`. The reference remains within `rho` of
the target on `[T,T+1]`; its speed there is at most `2*rho`. Consequently

\[
\|z(T+1)\|_D
 <2^{-512}/100000+1/51200000=:Z_{\rm end},
\quad 3216 Z_{\rm end}<1/8192, \tag{M8}
\]

while `||theta(T+1)-theta_*^m||_D<Q+rho<1/1024`.
The exact target's criticality and normalized Hessian upper bound give

\[
\mathcal R^2<65/67108864<1/648000=B,
\qquad \mathcal R<1/12. \tag{M9}
\]

Thus a passing reference certificate would put every actual source member in
the already admitted full-state basin (A8), at the fixed absolute time
`tau_b=tau_c+1025*1023^2*pi^2`. The original `t` clock is `tau_b/(1023/1024)`.
The disconnected comparison and later conditional dipole test keep their
existing separate histories and budgets. Neither that probe nor a new root
search is part of the proposed reference transit.

### Prospective numerical policy and executable admission boundary

The finite mathematical implication above is established; its reference
premises remain unevaluated. Select the following policy before implementation
or any reserved run, without treating the old prepared outcome as its result:

| Item | Fixed policy and required evidence |
| --- | --- |
| Reference and arithmetic | Equations (M1)--(M2), eighteen affine radian coordinates, zero initial center and radius, metric `D`, shared outward dyadic128 intervals and Taylor jets. Compute each undirected edge current once and scatter with opposite signs and the live degrees. |
| Target | Read the retained 32/64 endpoint brackets from the primary `two-port-compatibility-v1.json` with its manifest/source associations. Re-admit exact primitive endpoints, both inner signs, both outer signs, monotone-root premises, affine geometry and margin `>1/8` using shared kernels. Reuse the target-admission mechanism in `research/sine_two_port_handoff.py`; do not call the old capture assessor, consume its reference verdict or rerun its producer. Reject missing evidence or unresolved signs; no new bisection budget. |
| Propagation | Exactly 4096 steps of `1/4` through `T=1024`, Taylor order eight, growth bound zero only on certified acute tubes. Shared Picard admission uses at most sixteen inclusion iterations per step. No early successful endpoint, adaptive subdivision, order/precision change, pilot run or source adjustment. |
| Required observations | All twenty whole-tube margins `>rho`; strict Picard inclusion; complete center/radius chain; endpoint distance including target uncertainty `<=rho`; then (M6)--(M9) for the complete state. A passing nominal endpoint alone is insufficient. |
| Total work policy | One worker, one attempt, 3600 wall seconds including source/target admission and writes; process memory limited to 1 GiB. At most 256 MiB of streamed step evidence plus 64 MiB of source, logs and terminal records, 320 MiB total. Verify enforceable limits before launching; unavailable enforcement means no launch. |
| Retention and stopping | Stream each completed step through the shared exact-record and immutable-writer owners; preserve its full Taylor/Picard certificate, source associations and chain. Do not accumulate all certificates in RAM or claim a compact projection reproduces omitted remainders. Preserve the last complete prefix and any partial write on domain, endpoint, time, memory or output failure. No implicit restart or continuation after the cap. |
| Before execution | Implement the source-specific field/geometry/target adapter and bounded supervisor, independently review them against this section, complete non-transit controls, and freeze the actual producing source, protocol and reviews. The present mathematical selection alone does not authorize executing an unreviewed adapter. |

The static schedule is bounded: a step uses at most sixteen Picard field
calls plus eight center-jet and nine remainder-jet calls, at most 33 field
calls. Across 4096 steps that is at most 135168 field calls and 2703360
edge sine calls, some with Taylor jets; domain checks have their own maximum
of 65536 calls. Matrix admission, arithmetic, serialization and target
checks remain inside the same wall/memory/output limits. These counts are
not timing measurements. The earlier capture record supplies no retained
wall-time or peak-memory measurement from which completion can be promised.

`T=1024` is a fixed test horizon for the same graph and slow scale, not a
derived uniform deadline for the new reference. Its mathematical credibility
comes from a viable initial chart, the compatible target, and the complete-law
error budget (M7) and endpoint implication (M8)--(M9). The long acute prefix
and endpoint are the reserved discriminator. Failure would identify a missing
certificate under this policy, not prove that the actual pair cannot organize.

### Consolidation and the next bounded obligation

This method reuses the existing comparison, local barrier, target mathematics
and retained-metric kernel. It removes the fast form row only from the
numerical reference while retaining its complete-law reconstruction and
endpoint bound. It also avoids the old source-incompatible reflection and
unnecessary repeated target solving. No new constitutive law, state closure,
measured speedup or physical identification is claimed.

The source-aware method and prospective policy are now selected. Their
exact rational design controls and independent reviews are retained in
artifacts/formed-two-port-method-20261029. The next bounded obligation is
the executable adapter/supervisor admission and source freeze under this
unchanged policy, before one reserved transit attempt. Actual entry,
acquisition of the joint identity and its consequent response remain open.

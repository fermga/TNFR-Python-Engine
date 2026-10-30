# Capture after a certified two-port C9 transit

<a id="sine-two-port-capture"></a>

## Prospective question and complete-state scope

The [finite directional transit](SINE_TWO_PORT_TRANSIT.md#sine-two-port-directional-transit)
proves actual short-arc deformation through slow time \(1/4\). Its endpoint
is not an admission to the compatible equilibrium's basin. This separate
gate asks whether the same prepared full-state family can reach a certified
basin after a longer, explicitly declared transit.

Retain exactly that owner's two unit C9 cycles, contacts `(0,9)` and
`(1,10)`, held unit capacities, beta one, \(e=1023/1024\),
\(w=1/1024\), no forcing and no events. Let \(L\) be the actual unit
Laplacian, \(M\) the actual degree matrix, \(K=M^{-1}\), and
\[
\gamma=\frac1{1023\pi},\quad\eta=\gamma^2,\quad
\tau=et,\quad\sigma=\eta\tau.
\]
The complete actual rows are unchanged:
\[
x_\tau=-KLx+\gamma KS(\theta),\qquad
\theta_\tau=\gamma KLx,
\quad S_i(\theta)=\sum_{j\sim i}\sin(\theta_j-\theta_i). \tag{1}
\]

Keep nominal initial form zero and the same centered, midpoint-aligned
undeformed winding pair. In local indices \(j=0,\ldots,8\), its raw
phase turns are
\[
\widetilde\Theta_{{\rm D},j}=(j-\tfrac12)\frac29,\qquad
\widetilde\Theta_{{\rm R},j}=(j-\tfrac12)\frac19.
\]
Subtract the single full-network weighted mean \(21/40\) turns to obtain
\(\Theta^0\). Equivalently, use raw turns \(2j/9\) and
\(1/18+j/9\) and subtract their mean \(229/360\). These constructions
give exactly the same eighteen centered phases. Supply the unchanged family
\[
|x_i(0)|\le r_x=\frac1{65536},\qquad
|\theta_i(0)-2\pi\Theta_i^0|\le r_\theta=\frac1{65536}. \tag{2}
\]
Form and phase errors are independent nodewise budgets in form units and
radians respectively. No zero error sum or reflection constraint is imposed
on the thirty-six actual coordinates. Each member retains its own conserved
weighted form and continuous phase means.

The longer horizon below belongs to this new declaration. It neither
lengthens the earlier evaluated quarter-unit result nor replaces its source
by an implicit target or a state chosen near that target. The target used in
the proof is the unique correlated root of the
[two-port compatibility equations](SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-compatibility).
Its criticality and uniqueness are analytic premises already proved for the
same support, law and \((2,1,0)\) period cell.

## An exact eight-coordinate reference for the nominal source

<a id="sine-two-port-reference-folding"></a>

Use the auxiliary slow gradient flow
\(\psi_\sigma=KS(\psi)\), initialized at the nominal phase in (2).
It is a proof reference, not the actual phase law. The full initial form and
all actual errors enter the separate complete-law comparison below.

The nominal reference is invariant under simultaneous ring reflection
\(j\mapsto1-j\pmod9\) and circular phase negation about the common
contact midpoint. In the raw midpoint frame the fixed nodes have phases
\(\pi k\) for class \(k\). Their two sine terms cancel, so their reference rates
vanish. The support and both contact edges respect the reflection. Uniqueness
of the smooth gradient flow therefore preserves this symmetry for the exact
nominal reference.

For a ring of class \(k\), retain four turn coordinates
\(u_{k,0},\ldots,u_{k,3}\) at local nodes `1,...,4`. Its nine raw turns
are exactly
\[
(-u_{k,0},\ u_{k,0},\ u_{k,1},\ u_{k,2},\ u_{k,3},\ k/2,
\ k-u_{k,3},\ k-u_{k,2},\ k-u_{k,1}). \tag{3}
\]
The full-network weighted mean of this reconstructed vector is always
\(7(k_{\rm D}+k_{\rm R})/40=21/40\) turns, independently of \(u\).
Subtract that mean to obtain the same centered full reference as above.
The exact rational initial coordinates are
\[
u_{k,i}(0)=(i+\tfrac12)k/9,\qquad i=0,\ldots,3. \tag{4}
\]

For either component let \(v_0\) denote the other component's port
coordinate and set \(u_4=k/2\). In slow time the restricted equations are
\[
\begin{aligned}
(u_0)_\sigma
 &=\frac{-\sin(4\pi u_0)+\sin[2\pi(u_1-u_0)]
                  +\sin[2\pi(v_0-u_0)]}{6\pi},\\
(u_i)_\sigma
 &=\frac{\sin[2\pi(u_{i-1}-u_i)]
                  +\sin[2\pi(u_{i+1}-u_i)]}{4\pi},
                 &&i=1,2,3.
\end{aligned} \tag{5}
\]
Reconstruction (3) verifies every one of the eighteen reference rows,
including both degree-three contacts and the two fixed nodes. This is an
exact invariant restriction of that reference, rather than a phase closure
assumed for the actual complete system.

If two folded states differ by \(h\), their reconstructed physical phase
difference has squared full weighted norm
\[
\|\Delta\psi\|_M^2=(2\pi)^2h^TWh,\qquad
W=\operatorname{diag}(6,4,4,4,6,4,4,4). \tag{6}
\]
The factors six and four count both mirrored nodes with their actual
degrees. The fixed nodes contribute no deviation. On any convex tube whose
reconstructed edges remain in the same acute branches, the full gradient
field is dissipative in the \(M\)-norm. Equation (6) proves that the folded
field has logarithmic growth rate at most zero in the \(W\)-norm. No
numerical eigenvalue or absolute-Jacobian growth estimate is substituted for
this exact metric statement.

The correlated equilibrium also has a folded representative. For its root
turns \(A=a/(2\pi)\), \(C=c/(2\pi)\), these coordinates are
\[
u_{{\rm D},i}^*=A/2+i(2-A)/8,\qquad
u_{{\rm R},i}^*=C/2+i(1-C)/8. \tag{7}
\]
Reconstruction and the same mean subtraction agree with the compatibility
owner's exact implicit target. Independent rounded phase midpoints are not
declared to be critical.

For numerical evaluation use an equivalent affine coordinate chart: eight
**radian deviations** \(\xi\) from the fixed nominal phase, at full-node
representatives `(0,2,3,4,9,11,12,13)`. For each component the conversion from
the preceding absolute turns is
\[
\xi_{k,0}=-2\pi[u_{k,0}-u_{k,0}(0)],\qquad
\xi_{k,i}=2\pi[u_{k,i}-u_{k,i}(0)],\quad i=1,2,3. \tag{7a}
\]
The minus sign selects local node zero rather than its mirror, node one.
Write \(p\) for the reflection permutation and let the reconstruction
matrix have entries \(E_{ir}=1\) when \(i\) is representative \(r\),
\(E_{ir}=-1\) when \(i=p(r)\), and zero otherwise. Then
\[
\psi=2\pi\Theta^0+E\xi,\qquad
\xi(0)=0,\qquad E^TME=W,\qquad \mathbf1^TME=0. \tag{7b}
\]
Thus the retained metric radius is directly in the full physical phase
norm: \(\|E\Delta\xi\|_M=\|\Delta\xi\|_W\), with no additional
factor of \(2\pi\). The reference rate of coordinate \(\xi_r\) is
the full gradient rate at its representative node. Each edge angle is its
nominal angle plus the corresponding row difference of \(E\xi\).
This gives exactly (5) under (7a); it does not supply a second reference law.
The constant nominal edge angles retain mathematical pi, while the initial
numerical center and metric radius are exactly zero. Dissipativity and growth
rate zero are unchanged by this scaled sign transformation.

## Fixed reference-enclosure obligations

The shared
[`validated_metric_taylor_step`](../../src/tnfr/mathematics/_validated_metric.py)
retains a full positive-metric uncertainty ball while using projected boxes
only for Picard inclusion, derivatives and domain checks. Apply it to the
affine radian field (7a)--(7b), with metric \(W\), declared growth rate zero
and shared outward rational trigonometry. Pi remains the same mathematical
constant in
every row. Its outward enclosure covers arithmetic evaluation; it does not
introduce independent physical constants in different coefficients.

For each step, reconstruct the twenty principal full-reference edge gaps
with their fixed cycle branches. Require strict Picard inclusion and a
whole-tube acute margin greater than
\[
\rho:=1/2048\quad\hbox{radian} \tag{8}
\]
on every edge. This is stronger than merely checking midpoint or endpoint
acuteness. The integer ring periods and the zero four-edge interface period
remain those of the exact reconstruction on all admitted tubes.

At the fixed reference endpoint \(T=1024\), require
\[
\|\psi(T)-\theta_*\|_M\le\rho. \tag{9}
\]
Here \(\theta_*\) has the same zero nominal phase mean. The endpoint
metric radius and the uncertainty in the correlated implicit root must both
be included in this upper bound. For example, if \(\widehat\xi\) is the
retained radian center, \(r\) its radius in metric \(W\), and \(D_*\)
encloses \(\|\widehat\xi-\xi_*\|_W\), then \(r+D_*\) is an
admissible physical-norm upper bound. At each representative,
\(\xi_{*,r}=\theta_{*,r}-2\pi\Theta_r^0\), using the same full-network
zero gauge and the admitted implicit-root enclosure. A small
center residual alone proves neither (9) nor an exact critical target.

Rebuild the compatibility root using its declared thirty-two outer and
sixty-four inner strict-sign refinements. Require its admitted target to have
minimum acute edge margin strictly greater than \(1/8\) radian. The old
static proof supplies existence and criticality; the fresh enclosure checks
this quantitative margin without replacing either proof by a verdict field.

If a step, target margin or endpoint obligation is unavailable, preserve the
completed prefix and the failed condition. Neither increasing precision,
extending the horizon, changing the source nor retrying a smaller step is
part of this protocol. The reference's acuteness throughout its transit is an
obligation of the enclosure, not a consequence of having acute endpoints.

## One analytic continuation unit controls the original form

<a id="sine-two-port-capture-handoff"></a>

Assume the reference obligations above have passed. The ball in (9) lies
strictly inside the target's acute chart: every edge functional has weighted
dual norm at most one, while \(\rho<1/8\). The gradient-flow identity
\[
\frac12\frac d{d\sigma}\|\psi-\theta_*\|_M^2
 =\langle\psi-\theta_*,f(\psi)-f(\theta_*)\rangle_M\le0
\]
prevents a first exit from this ball. Therefore (9) holds for all later
reference times. During the additional declared unit \([T,T+1]\),
\[
\|f(\psi)\|_M\le2\rho=1/1024. \tag{10}
\]
This continuation does not require additional numerical reference steps.

Apply the
[energy-informed complete-law comparison](SINE_TWO_PORT_TRANSIT.md#sine-energy-informed-slow-comparison)
on the new total horizon \(\Sigma=T+1=1025\). Its weighted normalized
gap \(\lambda=1/90\), Lipschitz bound two and phase floor remain valid
on the same support and period cell. The fixed source satisfies
\[
H(0)-U_{\rm floor}
 <10/81+40r_\theta+40r_x^2<1/8.
\]
Keep the same bounds
\(1/3216<\gamma<1/3069\) and
\(\eta<\overline\eta=1/9000000\). Since
\(\sqrt{1025/8}<12\), define the rational quantities
\[
\begin{aligned}
Z_0&=7r_x/3069,&v_0&=7r_\theta+Z_0,\\
V&=v_0+2\overline\eta\,12\,90,\\
Z&=\frac{Z_0+90\overline\eta(5/12+2V)}
          {1-180\overline\eta},&Q&=V+Z.
\end{aligned} \tag{11}
\]
For (2), exact rational arithmetic gives
\[
Q<1/2048=\rho,\qquad Z<1/200000. \tag{12}
\]
Match the reference's common phase to each actual member's conserved phase
mean. Its geometry and all enclosure margins are unchanged. Before any
possible actual or mixed-coordinate exit, the comparison gives
\(\|\theta-\psi\|_M\le Q\) and
\(\|z\|_M\le Z\), with \(z=\gamma P_Mx\). The reference tube
margins on \([0,T]\) exceed \(\rho>Q\); the reference target ball
provides even larger margins on \([T,T+1]\). The strict first-exit
argument therefore validates these bounds throughout the whole new
interval, for the full thirty-six-coordinate family. Symmetry of its actual
errors was never required.

The global form envelope \(Z/\gamma\) alone is insufficient for a sharp
endpoint handoff. Retain the moving reference on the last unit and use (10)
instead of its larger initial forcing bound. Variation of constants for
\(z_\tau=-Az+\eta f(\theta)\) gives
\[
\|z(T+1)\|_M
 \le e^{-\lambda/\eta}Z
      +\frac{\overline\eta}{\lambda}(2\rho+2Q)
 <2^{-512}Z+\frac1{51200000}=:Z_{\rm end}. \tag{13}
\]
The exponential estimate is analytic:
\(\lambda/\eta>100000>512\) and \(\exp(1)>2\), so
\(e^{-\lambda/\eta}<2^{-512}\). No unknown forcing is differentiated,
and the original form is not replaced by a quasistatic algebraic row.
Using (12) and the lower bound on \(\gamma\) yields
\[
\|P_Mx(T+1)\|_M<3216Z_{\rm end}<1/8192. \tag{14}
\]
The actual phase bound at the same time is
\[
\|\theta(T+1)-\theta_*^{\,m}\|_M
 \le Q+\rho<1/1024, \tag{15}
\]
where \(\theta_*^{\,m}\) is the exact compatible target shifted to
that member's conserved phase mean. Its uniform limiting form will likewise
equal the member's conserved form mean, not necessarily zero.

## A strict full-state basin proves subsequent convergence

On that fixed-mean leaf use
\[
\mathcal R^2=\|P_Mx\|_M^2+\|\theta-\theta_*^{\,m}\|_M^2,
\qquad R=1/12.
\]
Within \(\mathcal R\le R\), each edge differs from its target by at
most \(R\). The target margin exceeds \(1/8\), so every segment from
the target has cosine at least
\[
\sin(1/8-1/12)=\sin(1/24)>1/25.
\]
Criticality cancels the linear phase-potential term. The graph gap and
Taylor's integral formula then give the lower bound
\[
H-H_*
 \ge\frac{\lambda}{2}
       [\|P_Mx\|_M^2+(1/25)\|\theta-\theta_*^{\,m}\|_M^2]
 \ge\frac{\mathcal R^2}{4500}. \tag{16}
\]
Here \(H_*\) is the exact target potential; common form shifts have zero
form storage. On the radius boundary the storage excess is therefore at
least
\[
B_R=R^2/4500=1/648000. \tag{17}
\]
Conversely, the normalized Laplacian upper bound two and
\(\mathcal H(\theta)\preceq L\) give
\(H-H_*\le\mathcal R^2\) for the endpoint, using the same exact
criticality. Equations (14)--(15) prove that endpoint is inside the radius
ball and has
\[
H(T+1)-H_*<\frac1{8192^2}+\frac1{1024^2}
 =\frac{65}{67108864}<\frac1{648000}. \tag{18}
\]
Thus the complete state, including its original form, enters a strict
forward-trapped sublevel. The continuous identity
\(H_\tau=-(Lx)^TK(Lx)\le0\) precludes exit through the radius
boundary. This is an excess-storage barrier around the new target after a
proved transit; it does not contradict the earlier obstruction to a direct
initial full-sector storage certificate.

On the fixed-mean leaf the sublevel is compact. Its largest invariant
zero-loss subset has \(Lx=0\), hence uniform form and zero phase
velocity. Preserving it further requires \(S=0\); summing the sine
rows removes any putative common nonzero form rate. The strict acute
period-cell uniqueness proved by the compatibility owner then leaves only
\(\theta_*^{\,m}\). LaSalle's invariance argument proves convergence
of every member to this compatible geometry and its own uniform form mean.
All later uninterrupted flow retains the periods. The existing local
exponential stability result applies near the limiting equilibrium, without
supplying a new uniform acquisition or decay rate.

## Frozen protocol and stopping rule

<a id="sine-two-port-capture-protocol"></a>

Freeze the following before the first new reference-transit assessment:

- The support, complete rows, clocks, source and independent radii are
  exactly (1)--(2). There is no source, origin, coefficient or error-budget
  search, support event, forcing or reset.
- Rebuild the primary `(2,1)` implicit target with thirty-two outer and
  sixty-four inner strict-sign refinements. Require its analytic equilibrium
  admission and outward minimum acute margin strictly above \(1/8\) radian.
- Propagate only the exact nominal eight-coordinate reference, using the
  affine radian deviations (7a)--(7b), zero initial center and radius, and
  representatives `(0,2,3,4,9,11,12,13)`. Use metric \(W\), growth bound zero,
  shared dyadic128 interval
  trigonometry, Taylor order eight and fixed step \(1/4\) in slow time.
  The reference horizon is \(T=1024\): exactly 4096 steps are required.
  This is a declared finite work policy for this experiment, not a change to
  a shared kernel's mathematical hypotheses or dimension limits.
- Require strict Picard inclusion and all twenty reconstructed whole-tube
  acute margins above \(1/2048\) radian at every step. Retain the metric
  radius; projected endpoint boxes do not replace it.
- Require the final reference metric distance, including target-enclosure
  uncertainty, to be at most \(1/2048\) radian. Do not stop successfully
  at an earlier favorable state or extend the final time after inspecting
  the endpoint.
- Use the analytic last slow-time unit, not additional trajectory steps,
  to obtain (13)--(15). The complete-state capture time is
  \(\sigma_*=1025\),
  \(\tau_*=1025\cdot1023^2\pi^2\), and
  \(t_*=1025\cdot1023\cdot1024\pi^2\).
- Certify capture only if target, complete reference prefix, endpoint,
  comparison bootstrap, original-form tail and strict full-state basin
  obligations all pass. Preserve an unavailable result and its completed
  prefix on any failure. Do not retry, subdivide adaptively or silently
  enlarge the budget.
- Archive the complete producing source, this prospective proof and the
  protocol before evaluation. Retain the response and verdict separately.
  Do not regenerate earlier frozen producers or revise their predictions.

At archival this proof establishes a conditional capture mechanism; the
fixed new reference assessment has not yet been performed. The outcome must
be recorded separately from the declaration above. In particular, the proved
local equilibrium and the earlier directional result do not count as passing
the new transit or endpoint obligations.

Even a successful result concerns relaxation of two supplied winding patterns
on supplied support. It does not derive their preparation, choose contact
occurrence, establish passive attachment, prove autonomous formation or
identify a physical composite. No laboratory clock, sensor model or physical
binding energy follows from this conditional mathematical capture result.

## Retained first capture result

<a id="sine-two-port-capture-result"></a>

The first assessment of the unchanged archived protocol returned
`certified_capture`, with no unavailable reasons. All ten frozen stopping
conditions passed. Every one of the 4096 reference steps was admitted at the
declared quarter-unit step and Taylor order eight, reaching reference slow
time \(T=1024\). The additional unit to complete-state time \(\sigma=1025\)
uses the analytic continuation and original-form estimate above; it is not
an additional numerical trajectory or an equilibrium reset.

The saved record retains exact rational bounds. The following decimal values
are approximate displays of those bounds, not measured errors or replacement
certificate thresholds:

| Retained quantity | Approximate bound | Declared requirement |
| --- | ---: | --- |
| Minimum whole-reference-tube acute margin | `0.04572542278762468` rad | Strictly above `1/2048` rad |
| Reference endpoint metric radius | `5.418710124146611e-9` rad | Retained in the endpoint distance |
| Reference distance to the implicit target at \(T\) | `5.684776349788719e-9` rad | At most `1/2048` rad |
| Full-family phase distance at \(\sigma=1025\) | `3.510605027044329e-4` rad | Below `1/1024` rad |
| Original relative-form norm at \(\sigma=1025\) | `5.39860958891343e-5` | Below `1/8192` |
| Full endpoint excess-storage upper bound | `1.261579751084399e-7` | Below `1/648000` |
| Positive capture-storage margin | `1.41705190143477e-6` | Strictly positive |

The reference endpoint distance includes its retained numerical radius and
the implicit-target enclosure. The actual phase and form bounds instead
include every member of the unchanged error family (2), through the
complete-law comparison. A small nominal-reference distance alone would
not have established either full-state bound.

The strict endpoint storage margin now admits the complete family into the
forward-trapped neighborhood (16)--(18). Consequently, under uninterrupted
evolution by (1), every member converges to the same compatible phase geometry
with its own conserved common phase origin, and to uniform form equal to its
own conserved form mean. Its winding periods remain \((2,1,0)\). This is
a source-to-basin result for the stated finite uncertainty family, beyond
the earlier local stability of an equilibrium supplied without a source.

The earlier initial-storage obstruction remains valid. It excluded a direct
full-sector energy certificate for these undeformed sources. The present
result supplies the previously missing transit information and applies a
local full-state barrier only after that transit. It does not reinterpret
the obstruction as a dynamical failure or replace the old result.

The retained
[protocol](../../docs/assets/sine_formed_classes/two-port-capture-v1.protocol.json),
[producing source archive](../../docs/assets/sine_formed_classes/two-port-capture-v1.source.zip),
[exact response](../../docs/assets/sine_formed_classes/two-port-capture-v1.json)
and [evidence manifest](../../docs/assets/sine_formed_classes/two-port-capture-v1.manifest.json)
preserve this first evaluation. The archive SHA-256 is
`b8df26578b59a32dae82b9cad58450b0d208d6c45dc234769575702830f0fb3f`, based on
revision `29e4c02339f6246d003a7fbfbdf7f37c8a2c1277` with the two declared
runtime overlays. The prospective proof above remains byte-identical to the
archived prefix. Hashes establish content consistency and recoverability,
not independent chronology authentication or physical acquisition.

The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-two-port-capture)
and [inspection guide](../../docs/guides/relational/SINE_PATTERNS.md#sine-two-port-capture)
retain the distinction between the validated nominal reference and the
full-law convergence conclusion. The
[independent field controls](../../tests/physics/test_sine_two_port_capture.py)
and [read-only evidence audit](../../tests/physics/test_sine_formed_evidence.py)
cover affine reconstruction, metric and gauge, all-node field agreement,
step-chain consistency, admitted edge margins and the exact handoff bounds.
The compact step record does not independently reproduce every Taylor
remainder or authenticate the producing execution.

The supplied support, complete law, preparation and structural clocks remain
premises. Convergence to joint geometry does not derive contact occurrence,
passive attachment, autonomous formation or physical binding.

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

while `||theta(T+1)-theta_*^m||_D<=Q+rho<1/1024`.
The exact target's criticality and normalized Hessian upper bound give

\[
H-H_*\le\mathcal R^2<65/67108864<1/648000=B,
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


<a id="sine-formed-two-port-executable-admission"></a>
## Executable admission of the source-aware reference

The source-specific implementation of M1--M9 separates mathematical admission,
complete evidence and process supervision. The implementation and its finite
controls add no evaluated source-to-target transit. In particular, a synthetic
endpoint near the retained equilibrium is not a flow image of the acquired
source family.

### Shared mathematics and exact observation

`tnfr.research.formed_two_port_reference` owns the original eighteen labels,
degree metric and twenty fixed edge lifts. It computes each sine current once
per edge and scatters opposite contributions through the live endpoint degrees.
Interval values and Taylor Jets use the same geometry. The whole-tube predicate
returns twenty strict slacks **above** `rho=1/2048`, retaining the `4*pi` and
`2*pi` closing lifts without wrapping an escaped chart back into admission.

Target admission checks the four retained primary evidence files, reconstructs
their primitive signs through the existing target owner and transports the one
critical target to the original labels. The common `+6*pi/5` gauge correction
compensates the removed turns; it is not a reset of constituent means. M3--M9
retain both actual rows and their inherited errors. The final form observation
is `||P_D x||_D`, relative to the conserved uniform form. Neither that norm nor
the auxiliary phase reference discards the memberwise conserved mean.

### Complete streamed certificates

`tnfr.research.formed_two_port_run` uses the shared metric Taylor kernel under
the fixed policy above. It records all twenty-six certificate fields exactly:
the metric and projection factors once in `steps/certificate-fields.json`,
and the other twenty-four fields in each immutable numbered step. Every step
checks the shared values, time, input center/radius, method and strict margins;
the previous file receipt links the sequence. Taylor remainder intervals,
coordinate errors and full endpoint enclosures are retained. A reconstruction
helper recovers all fields but does not independently reprove a Taylor step.

The writer charges the shared constants, complete step records and partial
step writes to the 256-MiB step allowance. It reserves exact serialized bytes
before opening each exclusive path. Auxiliary evidence has the separate
64-MiB allowance, including a 2-MiB supervisor reserve and a 1-MiB child terminal
reserve. Failure preserves the prefix and partial bytes; no resume, retry,
overwrite or silent error-radius reset is available. Only the current
certificate, next state and preceding receipt are retained in memory.

The worker checks its effective Job limits before source admission. A producing
protocol must bind all installed TNFR Python source, the wrapper actually
executed, the four primary target files, proof owners, exact runtime, reviews
and completed controls. It verifies the bytes actually copied or consumed.
Those receipts establish content associations, not reviewer authorship,
chronology, execution authentication or mathematical truth.

### Resource boundary and terminal verdict

`tnfr.research.bounded_worker` creates a suspended Windows process already in
its Job, checks membership and limits, then resumes it. The one-process cap
also excludes descendants. The 1-GiB cap means committed virtual memory for
both the process and Job, not RSS or a sampled Python allocation count. The
admitted CPython virtual environment is retained without an extra redirector
process. Unsupported enforcement fails before numerical work.

An independent watchdog terminates the Job at the worker deadline. Ten seconds
are reserved **inside** the 3600-second total for termination and supervisor
records, so the worker receives at most 3590 seconds including admission and
writes. The retained log is at most 1 MiB; each of the two parent JSON records
is at most 512 KiB. Filesystem/OS scheduling has no hard-real-time promise;
an overrun is unavailable, not permission to extend calculation.

`execution.json` is explicitly a pre-write snapshot. The caller must also retain
and consume the supervisor's returned post-write result, which can revoke a
success snapshot if terminal writing exhausts the total budget. A worker's
conditional capture record alone is insufficient. Final admission requires
the complete prefix, endpoint/tail conditions, exact source associations and
a successful post-write supervisor return. A timeout or failed sufficient
certificate does not prove non-entry.

### Reviewed controls and remaining scientific question

The declaration, source archive, independent numerical/storage and Windows
reviews, logs and test results are retained in
`artifacts/formed-two-port-executable-20261029`. Controls distinguish detached
field/Jet identities and one retained-target re-admission from mocked runner
records and inert Windows workers. They cover source substitution, exact
serialization, quota boundaries, partial writes, memory denial, timeout,
descendant prevention and late-close failure. No reserved reference step,
actual-source transit, new root search or historical producer replay belongs
to those controls.

The sole remaining scientific question is whether the fixed actual-source
reference reaches its admitted target while preserving every whole-tube margin
and error bound within the one declared attempt. Before that attempt, freeze
the exact producing wrapper/protocol and current sources with the reviewed
control results. Recheck the foundation prerequisite and source associations;
do not change step, order, precision, horizon, source or budget in response to
a pilot. Physical mapping and autonomous contact selection remain outside scope.


<a id="sine-formed-two-port-attempt-result"></a>
## Result of the single formed-source reference attempt

The reserved policy above has now been evaluated once. It produced four
complete reference certificates, through slow time `sigma=1`, and stopped on
the fifth invocation because its candidate tube could not be admitted by the
whole-tube domain predicate. **Actual-source entry remains unproved.** This
is an unavailable sufficient certificate, not a non-entry theorem or evidence
that the proposed joint organization cannot form.

The prospective instructions in the preceding method and executable sections
are retained as the producing contract. Their single attempt is consumed;
they do not authorize another launch. The
[research program](../RESEARCH_PROGRAM.md#research-current-question) owns the
current question and the next methodological obligation.

### Producing source, finite controls and execution

The final caller consumes and retains the supervisor's post-write return and
checks its own closing budget. Its 64-KiB reserve, including the external
return envelope, is deducted from the existing 64-MiB auxiliary allowance.
The numerical field, shared Taylor kernel and M1--M9 remain unchanged. The
101 new synthetic integration controls passed without failures, errors or
skips before the producing freeze; they establish tested orchestration and
record handling, not a source-entry result. Independent launch, audit and
root integration reviews are retained with the producing sources.

The evidence owner is `artifacts/formed-two-port-attempt-20261029`:

| Record or observation | Retained result |
| --- | --- |
| Producing protocol | `producing-protocol.json`, SHA-256 `94b7e48015a3c7767996a1c41ad355f96fd69887a26b01819c13191e829f125a` |
| Producing archive | `producing-source.zip`, 978 source files, SHA-256 `43e2a0760cb5e0e8db5822ebde72ecc7e9bda0992490d26e8b8c3abe47fec374` |
| Complete prefix | `attempt/steps/0000.json` through `0003.json`, four steps of `1/4`, with the shared certificate fields and exact center/radius chain |
| Stopping record | `attempt/failed-step.json`, zero-based index four, reason `formed_two_port_whole_tube_margin_unavailable`; no `0004.json` or endpoint verdict |
| Closing result | `caller-result.json` retains the actual post-write return: unavailable, worker exit two, cleanup complete, within the total budget |
| Measured resources | Caller elapsed about 7.554 seconds; peak process commitment 859734016 bytes, below 1 GiB; no timeout, watchdog, output-quota or memory failure reported |
| Retained output | 145568 step bytes and 5109151 auxiliary bytes including the external return, within their respective limits |
| Documentary audit | `audit.json`: four verified prefix steps, source associations verified, no unverified later step files; complete certificate not admitted |

The audit consumed retained evidence without replaying a reference step,
target sign or scientific producer. Its exit two reports the incomplete
certificate; it does not negate the verified prefix. The last certified
endpoint and metric radius are exactly the failed invocation's input. No
state reset, source replacement, new root solve, restart or parameter change
was used to continue after the failure.

### What the stopping condition establishes

The fifth invocation sought a tube for `[1,5/4]`. The shared Picard owner
checks the domain predicate **before** its field image and strict inclusion
test. The retained failing box was therefore still an unvalidated candidate.
It is not a certified enclosure of the reference throughout that interval.
The failure records that the required twenty strict slacks above
`rho=1/2048` were unavailable for that candidate. It does not establish that
an actual reference point left the acute domain, or even that a raw acute
margin became nonpositive. The positive safety allowance matters.

Neither the Picard iteration count nor earlier inflation candidates were
retained. This evidence cannot uniquely assign the rejection to interval
overestimation, candidate inflation or actual geometric evolution. The four
completed certificates concern the auxiliary eighteen-phase reference;
they do not supply the missing complete transit and M8--M9 endpoint handoff
for every member of the actual thirty-six-variable source family.

A separately declared, independently reviewed post-hoc diagnostic attempted
only rational comparisons of three retained boxes and their twenty edges.
That single invocation failed with `NameError: edge is not defined` in its
record assembly, before writing any diagnostic result. Its plan, source,
review, freeze, traceback and execution record are preserved separately.
It supplies **no edge-specific numerical finding**, corner witness or
explanation of the domain failure. It was not retried and did not advance
the reference, alter the producing evidence or invalidate the four complete
certificates. This auxiliary implementation defect must not be reported as
a dynamical obstruction.

### Next methodological obligation

The first missing premise is a usable, source-compatible transient domain
and comparison certificate. Assess that premise before admitting another
trajectory calculation. Map which M1--M9 arguments consume the acute collar,
the potential floor and the error bootstrap. Then determine whether a
correlation-preserving enclosure in the same domain is justified, or whether
a genuinely wider domain requires a new comparison proof. A candidate box
failure alone does not select between those alternatives.

Any proposed method must preserve both actual rows, original labels and
lifts, memberwise conserved means, inherited source correlations, contact
work and the same target identity. Replacing the twenty acute checks with a
local matrix growth bound alone would not re-establish the potential-floor
and first-exit premises used in M4--M7. A smaller step or larger budget is
likewise not a resolution without a justified new method and prospective
admission. Preserve this failed attempt as evidence when comparing methods.

The useful unification remains the chain from isolated formation, through a
supplied contact, to independently defined collective identity and retained
response. Compatibility and prepared-source capture remain valid in their
own scopes; the actual-source handoff is still missing. No new scale law,
autonomous contact mechanism or physical correspondence follows from this
bounded result.


<a id="sine-formed-two-port-domain-decision"></a>
## Transient-domain decision: energy, direction and collective curvature

Two analytical results refine the missing premise after the retained attempt.
First, the entire buffered acute cell with the source energy sublevel is
**not forward invariant** under M2, even on its source-connected component.
Second, the potential floor in M4 remains valid on a larger explicit branch
domain. These statements separate the energy estimate from the directional
and curvature conditions needed to control a transient. They do not establish
the nominal trajectory's departure, target arrival or actual-source capture.

No new trajectory, target solve, numerical sign admission or retained-box
diagnostic was evaluated. The four certified steps and both previous failures
remain unchanged. Analytical notes and reviews are retained in
`artifacts/formed-two-port-domain-20261029`; the current task belongs only to
the [research program](../RESEARCH_PROGRAM.md#research-current-question).

### A compatible energy sublevel need not be invariant

Retain M1--M2, the original labels and twenty oriented edges, all fixed lifts,
the joined degree metric and the nominal reference origin `phi=1/1000`.
Write

\[
q=4\pi/9,\quad r=2\pi/9,\quad \delta=\pi/9,\quad
\rho=1/2048,\quad a_\rho=\pi/2-\rho,\quad h(s)=1-\cos s.
\]

The proposed domain is

\[
\mathcal C_E=\{\xi:\mathbf1^TD\xi=0,\quad
g=g^0+B\xi,\quad |g_e|\le a_\rho\ (e\in E),\quad
U(g)\le U_{\rm floor}+E\},\qquad E=1/4. \tag{D1}
\]

The nominal reference `xi=0` lies strictly inside its edge and energy bounds:
`q<a_rho`, and
`U(g0)-U_floor=h(phi)+h(phi-r)<1/2000000+20/81<1/4`.
This directly bounds the reference phase energy; it does not identify it
with the actual family's complete storage. With `K=B D^-1 B^T`, M2 gives

\[
g'=-K\sin g,\qquad
U'=-(B^T\sin g)^TD^{-1}(B^T\sin g)\le0. \tag{D2}
\]

The joined mean and all affine cycle sums are also conserved. At a signed
edge facet `g_e=s*a_rho`, with `s` equal to `+1` or `-1`, forward invariance
would require `s*(K sin(g))_e>=0`. Energy decrease and the cycle equalities
do not imply that directional condition.

Reuse the [low-energy geometric path](SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-handoff-obstruction),
equations (15)--(19), transported to the original ports `4,5`. For
`0<=u<=pi/144`, give donor edge `5->6` gap `q+8u`, every other donor
edge gap `q-u`, every receiver edge gap `r`, and contacts `4->13,5->14`
gaps `c_u,-c_u`, where `c_u=delta-u/2`. The ring sums are `4*pi,2*pi`;
the mixed cycle satisfies `2*c_u+r-(q-u)=0`.

These are compatible nodal gaps in the original source chart. Explicitly,
let `h_5=8u` and `h_j=-u` for the other donor edge indices, and set

\[
\eta_{D,0}=0,\quad \eta_{D,j}=\sum_{k=0}^{j-1}h_k,
\quad \eta_{R,j}=\delta-\varphi-9u/2,\qquad
\xi=\eta-\frac{\mathbf1^TD\eta}{40}\mathbf1. \tag{D3}
\]

Then `eta_D,4=-4u` and `eta_D,5=-5u` give the stated contact gaps.
Zero summed donor increments preserve the closing lift. Only the joined
mean is removed; the original relative origin remains in `g0` and (D3).
The path is a geometric construction, not a replacement source or a flow.

Choose the symbolic value `u_rho=(pi/18-rho)/8`, strictly between zero and
`pi/144`. Its gaps are

\[
a=q+8u_\rho=\pi/2-\rho,\quad
b=q-u_\rho=7\pi/16+\rho/8,\quad
c=\delta-u_\rho/2=31\pi/288+\rho/16. \tag{D4}
\]

Only edge `5->6` is on a buffered facet; all other gaps are strictly inside.
The existing path derivative is
`d[U(u)-U_min]/du<-17/288`, where
`U_min=U_floor+2*h(delta)`. Consequently

\[
U(u_\rho)<U_{\min}<U_{\rm floor}+10/81<U_{\rm floor}+1/4.
\tag{D5}
\]

At this state, the actual reference degrees are `d_5=3,d_6=2`:

\[
\psi'_5=\frac{\sin a-\sin b-\sin c}{3},\qquad
\psi'_6=\frac{\sin b-\sin a}{2},\qquad
g'_{5\to6}=\frac56(\sin b-\sin a)+\frac13\sin c.
\]

Use `sin(b)>=cos(pi/16)`, `sin(a)<=1`,
`1-cos(pi/16)<=pi^2/512<5/256`, and
`sin(c)>=sin(31*pi/288)>=31/144`. They prove the exact strict bound

\[
g'_{5\to6}>-25/1536+31/432=767/13824>0. \tag{D6}
\]

Thus M2 points out of (D1) at a feasible boundary state. Its smooth flow
from that state immediately leaves the buffered domain, although energy
decreases. This refutes forward invariance of the proposed whole domain.

The witness also lies in the same component of the sharper sublevel
`U<=U(g0)` as the nominal source. First vary the geometric relative origin
`v` from `phi` to `delta` with uniform rings. All gaps remain buffered and
`U_nom(v)=U_floor+2-2*cos(delta)*cos(v-delta)` decreases, since its
derivative is `2*cos(delta)*sin(v-delta)<0` before the endpoint. Then
follow (D3) from zero to `u_rho`, decreasing energy again. Retain the joined
mean along both segments. This proves connectedness within that sublevel,
not reachability by the specific source trajectory.

In particular, this proof does **not** identify edge `5->6` as the cause of
the failed Picard candidate. It neither proves that the nominal reference
leaves the buffered cell nor excludes a narrower source-specific enclosure.
It closes one proposed invariant-domain argument without altering M1--M9.

### The same potential floor extends beyond the acute cell

For any `0<A<pi/2`, define
`R_A(s)=h(s)-h(A)-sin(A)*(s-A)`. Its derivative is nonpositive on
`[-pi,A]` and nonnegative on `[A,pi-A]`. Since `R_A(A)=0`,

\[
h(s)\ge h(A)+\sin(A)(s-A),\qquad -\pi\le s\le\pi-A. \tag{D7}
\]

This is a supporting-affine bound, not convexity of each edge potential
throughout that larger interval. For nine consistently lifted cycle gaps
with sum `9*A`, summing (D7) cancels the affine terms. Apply it with
`A=q` for the donor and `A=r` for the receiver. The unchanged floor obeys

\[
U\ge9h(q)+9h(r)=U_{\rm floor}\quad\hbox{if}\quad
-\pi\le g_{D,j}\le5\pi/9,\quad
-\pi\le g_{R,j}\le7\pi/9. \tag{D8}
\]

The two contact contributions are nonnegative for every real gap, so this
floor statement requires no further contact bound. The old source upper
bound, contact accounting and `E0=1/4` remain valid on any actual-phase
domain contained in (D8). No numerical extremum or new constant is needed.

The bounds concern fixed affine lifts. They establish neither principal
winding preservation, forward invariance nor positive curvature. Affine
cycle sums alone cannot give a global floor: one ring gap equal to `4*pi`
and eight zero gaps have the donor sum but zero donor potential. Domain
information remains indispensable.

### A domain contract for the unchanged comparison

The dependencies of the existing
[energy-informed comparison](SINE_TWO_PORT_TRANSIT.md#sine-energy-informed-slow-comparison)
can now be stated separately:

| Comparison ingredient | Required premise |
| --- | --- |
| M1--M3 and source bounds | Both full rows, original source/lifts, common-mean matching, positive degree metric and exact mixed coordinates `z=gamma P_D x`, `y=theta+z` |
| M4 and dissipated-storage integral | The complete dissipation identity, graph quotient gap and a valid phase floor everywhere the actual phase is compared; (D8) extends only this floor premise |
| M5 beyond its initial value | The full cosine Hessian is positive semidefinite along the reference, so its speed remains at most `F0` |
| M6 finite-error comparison | Nonexpansion on every segment between `y` and `psi`, plus the global Lipschitz bound; a Hessian check only at the nominal state is insufficient |
| M7 first-exit closure | Whole-time reference clearance sufficient to keep actual phase, mixed phase and comparison segments in their common domain |
| M8--M9 entry | Complete preceding comparison, arrival in the retained target ball, analytical last-unit handoff and original-form reconstruction |

More precisely, let `Omega` be an open convex phase domain invariant under
common shifts. Suppose `U(theta)>=U_floor` for every `theta` in `Omega`, and

\[
\mathcal H(g)=B^T\operatorname{diag}(\cos g)B\succeq0
\quad\hbox{throughout }\Omega. \tag{D9}
\]

If the reference has `dist_D(psi(sigma),Omega^c)>Q` throughout the complete
horizon `Sigma=1025`, then the same M3--M7 bounds follow. Indeed, (D9) on
segments gives nonexpansion, (D9) along the reference gives nonincreasing
speed, and the phase floor supplies the same dissipated-storage integral.
The constants `lambda=1/90`, `Lambda=2`, the source, clock and mean-matched
form semigroup are unchanged. Thus the same `V,Z,Q` estimates hold up to
first exit; the strict metric clearance prevents that exit for both `y`
and `theta`. No positive relative Hessian gap is needed for this particular
nonexpansion argument.

This is a conditional transfer of the existing proof, not a constructed
`Omega` or a new numerical admission. Clearance must cover all reference
states and their uncertainty, not selected centers or times. If it is proved
only through `T=1024`, the final unit needs an explicit compatible local
collar or a justified transition to the existing acute target proof. Its
target-arrival and complete-state obligations remain separate.

The repository already proves that other graph paths can compensate some
negative edge cosines in different admitted models: see
[relative storage](SINE_PREPARATION_ROBUSTNESS.md#sine-relative-preparation-storage)
and [full-network curvature](SINE_COLLECTIVE_IDENTITY_TRANSITION.md#nul-relative-curvature).
Those methods require the actual paths, all relative variations and the
combined loads on shared edges. Their graphs, laws, numerical bounds and
trajectory conclusions do not transfer to this two-port source automatically.
Here they motivate checking the full graph Hessian instead of identifying
its positivity with twenty separate acute inequalities.

A positive growth bound in place of (D9) changes both the error comparison
and the speed estimate. M6--M9's old numerical constants then require a new
derivation; changing only an integrator option cannot supply it.

### Closed decision and remaining foundation

The scalar energy-sublevel invariance route is closed by (D6). The floor
extension (D8) removes one unnecessary restriction from a possible wider
comparison. The remaining question is whether a source-compatible corridor
can supply full-domain curvature and whole-time clearance, or whether a
specific proposed corridor has an obstruction. A correlated enclosure within
the old acute domain remains an alternative if that reference actually stays
there; neither result decides its trajectory.

The first methodological task is to specify such a corridor using the actual
cycle constraints and graph paths, and check its curvature and boundary
obligations before selecting a numerical policy. No new transit, implicit
restart or repair/replay of the failed auxiliary diagnostic is authorized by
this analytical result. Formation of the joint identity from the retained
actual source, autonomous contact occurrence and physical mapping remain open.

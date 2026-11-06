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

### Method consolidation and historical admission boundary

This method reuses the existing comparison, local barrier, target mathematics
and retained-metric kernel. It removes the fast form row only from the
numerical reference while retaining its complete-law reconstruction and
endpoint bound. It also avoids the old source-incompatible reflection and
unnecessary repeated target solving. No new constitutive law, state closure,
measured speedup or physical identification is claimed.

The source-aware method and prospective policy were selected at this stage.
Their exact rational design controls and independent reviews remain in
artifacts/formed-two-port-method-20261029. Executable admission and source
freeze were then required before the reserved transit attempt. That policy
is now consumed: the later [source-origin decision](#sine-formed-two-port-origin-decision)
and [original-period exclusion](SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-original-period-exclusion)
own the current mathematical conclusions. This historical boundary activates
no new transit; the [program](../RESEARCH_PROGRAM.md#research-current-question)
alone selects the next investigation.


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

### Closed domain decision and later source conclusions

The scalar energy-sublevel invariance route is closed by (D6). The floor
extension (D8) removes one unnecessary restriction from a possible wider
comparison. This motivated testing a source-compatible corridor with its
full-domain curvature and whole-time clearance, using actual cycle constraints
and graph paths. The [following corridor analysis](#sine-formed-two-port-corridor-decision)
records that decision; the later
[actual-source exclusion](#sine-formed-two-port-actual-exclusion) separately
closes entry to the named target. The domain results alone would not have
decided that trajectory. No new transit, implicit restart or repair/replay of
the failed auxiliary diagnostic is authorized here. Autonomous contact
occurrence and physical mapping remain separate obligations; the
[program](../RESEARCH_PROGRAM.md#research-current-question) owns current selection.


<a id="sine-formed-two-port-corridor-decision"></a>
## Source-compatible corridor decision: path compensation and curvature exit

This section separates useful comparison geometry from trajectory conclusions
for the unchanged reference law (M1)--(M2) and complete comparison (M3)--(M9).
First,
the eight-edge complementary donor path defines an explicit open convex
corridor beyond the acute cell where one donor edge is slightly obtuse and
both (D8) and (D9) hold. Second, the seven degree-two interior donor bulk
vertices form an unbranched series chain, which excludes multi-edge bulk
compensation and fixes an exact necessary curvature functional. Third,
continuing the source-compatible path (D3) across the acute face to
`u_crit=5*pi/576` stays inside (D8) and strictly below the nominal source
energy while driving that curvature functional negative. Its M2 derivatives
are outward across the intermediate scalar gap and curvature levels, but the
necessary scalar curvature boundary is not a boundary of the full PSD set.
The separate source-dependent argument below proves acute exit; the following
section proves existence of a reference Hessian-positivity exit. Neither the
geometric path nor these finite exits decide eventual capture of the original
formed-source family.

No new trajectory, target sign admission, numerical step or diagnostic replay
is executed. The four certified steps and prior failure records remain untouched.

### Path-compensated convex corridor and the degree-two series ceiling

Retain the original eighteen nodes, twenty oriented edges, joined degree
matrix `D` of mass forty, fixed closing offsets `+4*pi` and `+2*pi`, and
nominal gaps `g0` with `phi=1/1000`. On the affine cycle leaf `g=g0+B*xi`,
let `C_D` be the nine donor ring edges, `P_D^*=C_D\{5->6}` the complementary
eight-edge donor path connecting nodes `5` and `6`, and `E_D,bulk=C_D\{4->5}`
the eight donor bulk edges. For `0<=kappa<=pi/18`, `0<m_D<pi/18` and
`0<m_R<pi/2`,
define the open convex phase corridor

\[
\Omega(\kappa,m_D,m_R)=\left\{\theta\in\mathbb R^{18}:
-\frac\pi2+m_D<g_{5\to6}<\frac\pi2+\kappa,\quad
|g_e|<\frac\pi2-m_D\ (e\in P_D^*),\quad
|g_e|<\frac\pi2-m_R\ (e\notin C_D)\right\}. \tag{C1}
\]

The domain (C1) is open, convex and invariant under common phase shifts, so it
contains every comparison segment `(1-s)*psi+s*y` between two admitted states.
Since `kappa<=pi/18`, all donor gaps lie in `(-pi/2,5*pi/9)` and all receiver
gaps lie in `(-pi/2,pi/2)`, so (D8) guarantees `U(theta)>=U_floor` throughout
`\Omega(\kappa,m_D,m_R)`.

For every `v` in `\mathbb R^18`, the cycle identity along `C_D` gives
`v_6-v_5=-sum_(e in P_D^*) Delta_e v`, where `Delta_e v=(Bv)_e`. By
Cauchy--Schwarz (equivalently the sum-of-squares identity
`8*sum_{e in P_D^*}(Delta_e v)^2-(Delta_{5->6}v)^2=sum_{e<f in P_D^*}(Delta_e v-Delta_f v)^2>=0`),
compensating a negative stiffness `cos(g_{5->6})>-sin(kappa)` along `P_D^*`
places a uniform shared load `8*sin(kappa)/sin(m_D)` on each of its eight
edges:

\[
v^T\mathcal H(g)v\ge
\left(\sin m_D-8\sin\kappa\right)\sum_{e\in P_D^*}(\Delta_e v)^2
+\sin m_R\sum_{e\notin C_D}(\Delta_e v)^2. \tag{C2}
\]

Whenever `8*sin(kappa)<=sin(m_D)`, (D9) holds everywhere in
`\Omega(\kappa,m_D,m_R)`, and strict inequality makes `\mathcal H(g)` coercive
on `\mathbf1^\perp` because `(V,E\{5->6})` is connected.

The graph topology also fixes a sharp converse ceiling on the donor bulk
chain. Every non-port donor vertex `6,7,8,0,1,2,3` has bare degree two in the
joined twenty-edge network. Removing any two distinct donor bulk edges
`e_1!=e_2` in `E_D,bulk` disconnects the intermediate degree-two arc from all
ports and from the receiver ring. A variation `w` equal to `1` on that
detached bulk arc and `0` elsewhere has `Delta_{e_1}w=1`, `Delta_{e_2}w=-1`
and `Delta_e w=0` on the other eighteen edges, so

\[
w^T\mathcal H(g)w=\cos g_{e_1}+\cos g_{e_2}. \tag{C3}
\]

Hence no state with two nonacute donor bulk edges (`cos g_{e_1}<0` and
`cos g_{e_2}<=0`) can satisfy `\mathcal H(g)>=0`. For a single obtuse bulk
edge `5->6`, define the fixed donor-bulk variation `v` by clamping all four
ports and the receiver ring (`v_4=v_5=0`, `v_{9+j}=0` for `0<=j<=8`) and
setting `v_{(5+k) mod 9}=(8-k)/7` for `1<=k<=7`. Then `Delta_{5->6}v=1`,
`Delta_e v=-1/7` on the seven remaining donor bulk edges
`E_D,bulk\{5->6}`, and `Delta_e v=0` on the short donor edge `4->5`, both
contacts and all receiver edges. Evaluating the full twenty-edge quadratic
form on `v` yields the exact necessary condition for (D9):

\[
q_{\rm bulk}(g):=v^T\mathcal H(g)v
=\cos g_{5\to6}+\frac1{49}\sum_{e\in E_{D,{\rm bulk}}\setminus\{5\to6\}}\cos g_e
\ge0. \tag{C4}
\]

Because `v` vanishes on all ports and the receiver ring, neither the short
edge `4->5`, nor the two contacts, nor the receiver ring can contribute
compensating stiffness to `q_bulk(g)`.

### Curvature exit and outward flow inside the nominal-source energy sublevel

Extend the cycle-compatible, joined-mean-zero geometric path (D3) across the
acute face `u_b=pi/144` (`a(u_b)=pi/2`) to

\[
u_{\rm crit}:=\frac{5\pi}{576}=u_b+\frac\pi{576},\qquad
a(u_{\rm crit})=\frac\pi2+\frac\pi{72}=\frac{37\pi}{72}<\frac{5\pi}{9}. \tag{C5}
\]

Throughout `0<=u<=u_crit<pi/81`, the gaps `a(u)=q+8u`, `b(u)=q-u` and
`c(u)=delta-u/2` remain strictly inside the extended floor domain (D8), with
`b(u)>=251*pi/576=pi/2-37*pi/576` and `c(u)>=123*pi/1152=41*pi/384`. Since
`0<=|pi/2-a(u)|=|pi/18-8u|<=pi/18+u=pi/2-b(u)<=37*pi/576<pi/2`, monotonicity
of cosine on `[0,pi/2]` gives `\cos(37\pi/576)\le\sin b(u)\le\sin a(u)\le1`.
Using `1-cos z<=z^2/2` with `pi^2<10` and `sin z>=2z/pi` on `[0,pi/2]`, every
`u` in `[0,u_crit]` satisfies

\[
0\le\sin a(u)-\sin b(u)\le1-\cos\frac{37\pi}{576}<\frac{6845}{331776}<\frac1{48},
\qquad \sin c(u)\ge\sin\frac{41\pi}{384}\ge\frac{41}{192}.
\]

Consequently the energy derivative along (D3) remains strictly negative on the
entire extended interval `[0,u_crit]`:

\[
\frac{d}{du}U(g(u))=8[\sin a(u)-\sin b(u)]-\sin c(u)
<\frac8{48}-\frac{41}{192}=-\frac3{64}<0. \tag{C6}
\]

Every state `xi(u)` on `0<=u<=u_crit` therefore satisfies
`U(g(u))<=U_min<U(g0)<U_floor+1/4`, with equality to `U_min` only at `u=0`,
and lies in the nominal source's connected component of `{U<=U(g0)}`.

At `u=u_crit`, concavity of sine on `[0,pi/6]` gives
`sin(pi/72)>=(3/pi)*(pi/72)=1/24`, while `sin(37*pi/576)<=37*pi/576<407/2016`
from `pi<22/7`. Substituting into (C4) proves

\[
q_{\rm bulk}(g(u_{\rm crit}))
=-\sin\frac\pi{72}+\frac17\sin\frac{37\pi}{576}
<-\frac1{24}+\frac{407}{14112}=-\frac{181}{14112}<-\frac1{96}<0. \tag{C7}
\]

Thus `\mathcal H(g(u_crit))` is strictly indefinite: the low-energy path
exits not only every path-compensated corridor (C1), but **every** domain
`\Omega` satisfying (D9), before `u` reaches `u_crit`.

Furthermore, along the entire segment `0<=u<=u_crit` (which includes the
buffered facet `u_rho`, the acute facet `u_b`, and the unique point
`u_H in (u_b,u_crit)` where `q_bulk(g(u_H))=0`), the M2 reference vector field
`g'=-K*sin(g)` has `g'_{7->8}=g'_{8->0}=g'_{0->1}=g'_{1->2}=g'_{2->3}=0` and

\[
\begin{aligned}
g'_{5\to6}(u)&=\frac56[\sin b(u)-\sin a(u)]+\frac13\sin c(u)
 >-\frac5{288}+\frac{41}{576}=\frac{31}{576}>0,\\
g'_{6\to7}(u)+g'_{3\to4}(u)&=\frac12[\sin a(u)-\sin b(u)]+\frac13\sin c(u)
 \ge\frac13\sin c(u)\ge\frac{41}{576}>0,\\
g'_{4\to5}(u)&=\frac13[\sin a(u)-\sin b(u)]-\frac23\sin c(u)
 <\frac1{144}-\frac{41}{288}=-\frac{13}{96}<0.
\end{aligned} \tag{C8}
\]

Since `\sin a(u)\ge\sin b(u)>47/48` on `[0,u_crit]`, the M2 time derivative
of the necessary curvature functional `q_bulk(g)` at `g(u)` is strictly
negative throughout `[0,u_crit]`:

\[
\frac{d}{d\sigma}\Big|_{M2}q_{\rm bulk}(g)\Big|_{g(u)}
=-\sin a(u)\,g'_{5\to6}(u)-\frac{\sin b(u)}{49}[g'_{6\to7}(u)+g'_{3\to4}(u)]
<-\frac{47}{48}\left(\frac{31}{576}+\frac{41}{28224}\right)
=-\frac{72145}{1354752}<-\frac1{19}<0. \tag{C9}
\]

Equations (C6)--(C9) show that the source energy sublevel contains states
outside the PSD region, and that the necessary scalar test has an outward
derivative on this geometric path. They do not prove non-invariance of the
intersection with full-network Hessian positivity. Indeed, at `u_H`,
`cos a=-cos b/7` and the vector `v` of (C4) obeys

\[
\mathcal H(g(u_H))v=\frac{\cos b(u_H)}7(e_5-e_4)\ne0,
\qquad v^T\mathcal H(g(u_H))v=0.
\]

A PSD matrix would force `H v=0` when `v^T H v=0`. Thus this scalar boundary
point is already outside the PSD set; it cannot serve as an admissible PSD
boundary witness. The later trajectory contradiction (P1)--(P4), with its own
source-dependent bounds, excludes a PSD comparison domain containing the
original reference throughout the interval to `sigma=7/4`.

### Certified reference and full-law acute exit on `(1,5/4)`

Combining the path-compensated corridor (C1)--(C2) with centralized `D`-metric
edge dual norms resolves whether the fifth-step failure in
`formed-two-port-attempt-20261029` was box overestimation or an actual exit from
the acute cell. On the retained `D`-metric ball `||xi-xi_0||_D<=R`,
Cauchy--Schwarz bounds every oriented edge difference `|(B(xi-xi_0))_e|` by
`||b_e||_{D^-1}*R`, where `||b_e||_{D^-1}=sqrt(d_left^-1+d_right^-1)<=1` (`1`
on degree-two bulk edges, `sqrt(5/6)` on port-to-bulk edges and `sqrt(2/3)` on
port-to-port edges). This shared enclosure is implemented by
`metric_difference_dual_norms` and `enclose_metric_ball_differences` in
`tnfr.mathematics._validated_metric`, and exposed on the reference owner by
`FormedTwoPortReference.edge_dual_norms`, `metric_ball_angles` and
`corridor_domain`.

First retain the complete four-step prefix on `[0,1]`, not just its endpoint.
The immutable certificates `attempt/steps/0000.json` through `0003.json`
belong to the previously audited source and consecutive quarter-unit chain.
Their whole-tube gaps, with the original affine lifts, have clearance at
least `rho` from every boundary of
`Omega(1/1024,1/64,1/2)`: the recorded acute slacks already exceed `1/64`
on every non-active donor edge and `1/2` on every non-donor edge; the two
active-edge corridor faces are checked separately. Thus the complete prefix
lies in this same buffered corridor. The read-only admission
`admit_formed_two_port_acute_exit_prefix` consumes the fixed receipts,
time/endpoint chain and retained tubes before the exit certificate is used.
It does not rerun or independently reprove the historical Picard/Taylor
certificates. An unaffiliated endpoint ball is not evidence of this prefix.

Apply the edge enclosure to the already certified endpoint `(xi^(4),R^(4))` at
`sigma=1` from `attempt/steps/0003.json`, where `R^(4)<2^-26`. Every
`psi(1)` in `\mathcal B_D(\xi^{(4)},R^{(4)})` satisfies

\[
\begin{aligned}
&\frac\pi2-\frac{44}{2048}<g_{5\to6}(\psi(1))<\frac\pi2-\rho,\qquad
g'_{5\to6}(\psi(1))>\frac{53}{500},\qquad
\|\psi'(1)\|_D<\frac14,\\
&|g_e(\psi(1))|<\frac\pi2-\frac16\ (e\in P_D^*),\qquad
|g_e(\psi(1))|<\frac\pi2-\frac23\ (e\notin C_D),\\
&g_{6\to7}(\psi(1))>\frac43+\frac1{16},\qquad
g_{4\to5}(\psi(1))>\frac76+\frac1{16}.
\end{aligned} \tag{C10}
\]

In (C1)--(C2), select `kappa=1/1024=2*rho`, `m_D=1/64` and `m_R=1/2`. Since
`8*sin(1/1024)<1/128<sin(1/64)`, both the extended floor (D8) and full-network
Hessian positivity (D9) hold throughout the open convex corridor
`\Omega(1/1024,1/64,1/2)`. Suppose for contradiction that on `sigma in [1,5/4]`
the reference trajectory satisfies `g_{5->6}(psi(sigma))<=pi/2+rho=pi/2+kappa-rho`.
Up to any first exit from the `rho`-buffered corridor on `[1,5/4]`, (D9) keeps
`||psi'(sigma)||_D<=||psi'(1)||_D<1/4`, so each edge changes from its `sigma=1`
value by at most `||b_e||_{D^-1}*||psi'(1)||_D*(sigma-1)<1/16`. Consequently
every `e in P_D^*` retains `|g_e|<pi/2-5/48<pi/2-m_D-rho`, every `e notin C_D`
retains `|g_e|<pi/2-29/48<pi/2-m_R-rho`, `g_{5->6}>4*pi/9` (`|cos g_{5->6}|<1/5`),
`g_{6->7}>4/3` (`|cos g_{6->7}|<1/4`) and `g_{4->5}>7/6` (`|cos g_{4->5}|<2/5`).

Differentiating `g'_{5->6}=-(5/6)sin g_{5->6}+(1/2)sin g_{6->7}+(1/3)sin g_{4->5}-(1/3)sin g_{5->14}`
and bounding `|g'_e|<=||b_e||_{D^-1}*||psi'(1)||_D` gives, on `[1,5/4]`,

\[
|g''_{5\to6}(\psi(\sigma))|\le
\left(\frac56|\cos g_{5\to6}|\sqrt{\frac56}
 +\frac12|\cos g_{6\to7}|
 +\frac13|\cos g_{4\to5}|\sqrt{\frac23}
 +\frac13\sqrt{\frac23}\right)\|\psi'(1)\|_D<\frac{11}{64}. \tag{C11}
\]

Since `g'_{5->6}(psi(1))>0`, (C11) implies
`g_{5->6}(psi(sigma))>=g_{5->6}(psi(1))-(1/32)*(11/64)>pi/2-55/2048` on
`[1,5/4]`, tightening `|cos g_{5->6}|<55/2048` and hence
`|g''_{5->6}(psi(sigma))|<9/64`. Taylor's theorem at `sigma=5/4` (`Delta sigma=1/4`)
then yields

\[
g_{5\to6}\!\left(\psi\left(\frac54\right)\right)-\frac\pi2
\ge g_{5\to6}(\psi(1))-\frac\pi2+\frac14g'_{5\to6}(\psi(1))
 -\frac1{32}\cdot\frac9{64}>\frac{159}{256000}>\rho, \tag{C12}
\]

contradicting `g_{5->6}(psi(sigma))<=pi/2+rho`. By continuity there is a first
time `sigma_exit in (1,5/4)` at which `g_{5->6}(psi(sigma_exit))=pi/2+rho`.
Equation (C12) is a contradiction bound under assumed corridor retention,
not an unconditional bound on the actual reference endpoint at `5/4`; the
certificate labels it `corridor_contradiction_excess_lower`.
Combining the retained prefix with this first-exit argument, the reference
has clearance at least `rho` from the boundary of the open corridor
throughout `[0,sigma_exit]`, including equality at the active upper face of
the buffered corridor at the final time.

Because `\Omega(1/1024,1/64,1/2)` is open, convex and satisfies (D8)--(D9)
with reference boundary clearance at least `rho=1/2048>Q` throughout
`[0,sigma_exit]`, the complete-law comparison (M3)--(M7) applies through
`sigma_exit`. Every member `(x(tau),theta(tau))` of the actual thirty-six-variable
formed-source family (A1) under the complete nonlinear law (A2) therefore
satisfies `||theta(sigma_exit)-psi(sigma_exit)||_D<=Q<9/25000<rho`, and

\[
g_{5\to6}\big(\theta(\sigma_{\rm exit})\big)
\ge g_{5\to6}\big(\psi(\sigma_{\rm exit})\big)-\sqrt{\frac56}\,Q
>\left(\frac\pi2+\rho\right)-\rho=\frac\pi2. \tag{C13}
\]

This is certified without executing a new integration step by
`certify_formed_two_port_fifth_step_acute_exit` in
`tnfr.research.formed_two_port_reference`. Equations (C10)--(C13) prove that
the fifth-step domain rejection was not an enclosure artifact: **both the
reference `psi(sigma)` and every actual thirty-six-variable trajectory of the
`phi=1/1000` formed-source contact at `tau_c=100+10^13` exit the acute phase
cell across `g_{5->6}=pi/2` during `sigma in (1,5/4)`.**

### Closed acute-corridor route and next question

Equations (C10)--(C13) close the all-acute reference route for
the `phi=1/1000` contact at `tau_c=100+10^13`. Acute exit on edge `5->6` is a
proved property of the actual complete law (A2) from that inherited relative
origin, not a proof that the two-port organization is unreachable from all
formed sources or that identity is lost at `pi/2`. These exit bounds alone
do not decide later capture; the [actual-source exclusion](#sine-formed-two-port-actual-exclusion)
below settles that separate question. A possible later principal
winding change would require its own branch-crossing evidence; acute exit
alone proves none. The following section distinguishes the reference's
Hessian exit from capture for a different supplied relative origin. The
research program owns selection of further work. Autonomous contact
occurrence and physical mapping remain open.


<a id="sine-formed-two-port-origin-decision"></a>
## Post-acute Hessian exit at `phi=1/1000` and the admissible contact-origin window

This section gives two distinct conditional results without executing a new
trajectory solver or producer. It does not resolve the original family's
eventual source-to-basin entry:

1. **Trajectory exit from full-network Hessian positivity at `phi=1/1000`:**
   from the same certified `sigma=1` ball `(xi^(4),R^(4))` (`steps/0003.json`),
   an analytical bound using the combined `D^-1` dual norm of the three-edge
   rate functional, exact node-5 cancellation along the seven remaining donor
   bulk edges and Jensen's inequality proves that the reference trajectory
   `psi(sigma)` has a first exit from the positive-semidefinite Hessian set
   `{\mathcal H(g)>=0}` before `sigma=7/4`. The negative terminal bound is a
   contradiction under assumed PSD throughout `[1,7/4]`, not a bound on the
   actual state at `7/4` after that assumption fails. Thus no domain `\Omega`
   satisfying (D9) can contain this reference throughout the interval to
   `sigma=7/4`.
2. **Admissible contact-origin window and formed-source capture:**
   because a constant common phase shift `phi_0` of the isolated receiver ring
   is an exact symmetry of the disjoint evolution up to contact
   (`tau_c=100+10^13`), preparing the isolated receiver in the midpoint window
   `|phi_0-pi/9|<=1/32768-2*10^-32` (for example the exact rational
   `phi_0=355/1017=(1/9)*(355/113)`) preserves the identical isolated formation
   and `10^-32` dwell proofs while placing every thirty-six-variable post-contact
   state (A1) inside the already certified capture box (2) (`r_x=r_theta=1/65536`).
   The retained 4096-step capture certificate (`two-port-capture-v1.json`) and
   dipole theorem (A9)--(A11) therefore certify complete entry into (A8),
   permanent maintenance of the deformation collar (A6)--(A7), and the
   conditional interior dipole response contrast for that shifted preparation.
   The symmetry does not identify the two joined initial-value problems.

### Dynamic trajectory exit from `{\mathcal H(g)>=0}` before `sigma=7/4`

Suppose for contradiction that the exact reference solution `psi(sigma)` of
(M1)--(M2) starting from the `phi=1/1000` nominal source satisfies
`\mathcal H(g(psi(sigma)))>=0` throughout `sigma in [1,7/4]` (`h=3/4`). Then
`||psi'(sigma)||_D<=||psi'(1)||_D<1/4` for all `sigma in [1,7/4]`, and (C4)
requires `q_bulk(g(psi(sigma)))>=0` throughout `[1,7/4]`.

Decompose the active donor bulk derivative as
`g'_{5->6}(sigma)=-(5/6)sin g_{5->6}(sigma)+R_3(sigma)`, where

\[
R_3(\sigma):=\frac12\sin g_{6\to7}(\sigma)
 +\frac13\sin g_{4\to5}(\sigma)-\frac13\sin g_{5\to14}(\sigma).
\]

Since `sin g_{5->6}(sigma)<=1` for every real angle,
`-(5/6)[sin g_{5->6}(sigma)-sin g_{5->6}(1)]>=-(5/6)[1-sin g_{5->6}(1)]=:-delta_56>-1/4096`.
Meanwhile `R'_3(sigma)=w_3(g(sigma))^T psi'(sigma)`, where

\[
w_3(g)=\frac12(\cos g_{6\to7})(e_7-e_6)
 +\frac13(\cos g_{4\to5})(e_5-e_4)
 -\frac13(\cos g_{5\to14})(e_{14}-e_5).
\]

Evaluating the exact `D^-1` quadratic norm of `w_3(g)` (`d_4=d_5=d_{14}=3`,
`d_6=d_7=2`) combines the two node-5 terms before Cauchy--Schwarz:

\[
\|w_3(g)\|_{D^{-1}}^2
=\frac14\cos^2 g_{6\to7}+\frac1{27}\cos^2 g_{4\to5}
 +\frac1{27}\cos^2 g_{5\to14}
 +\frac1{27}(\cos g_{4\to5}+\cos g_{5\to14})^2.
\]

Under the assumed all-time PSD premise, each edge changes from its admitted
`sigma=1` interval by at most `||b_e||_{D^-1}*||psi'(1)||_D*(3/4)`.
For each of edges `6->7` and `4->5`, let `[L_e,U_e]` be that enlarged interval.
The primitive interval checks give `0<L_e<pi/2` and `U_e<=pi-L_e`, hence
`|cos(g_e)|<=cos(L_e)` throughout the interval. In particular, this does not
assert that edge `6->7` remains acute: its admitted interval can cross
`pi/2`. Substitution of these absolute-cosine bounds in the combined dual norm
gives `||w_3(g(sigma))||_{D^-1}*||psi'(1)||_D<=S_3<1/10`.
Therefore, still under the same hypothesis, for every `sigma in [1,7/4]`,

\[
g'_{5\to6}(\sigma)\ge g'_{5\to6}(1)-\delta_{56}-S_3(\sigma-1). \tag{P1}
\]

Integrating (P1) over `sigma in [1,7/4]` (`h=3/4`) and retaining the primitive
`sigma=1` interval bounds, rather than only their rounded display in (C10),
gives under that hypothesis

\[
g_{5\to6}\!\left(\psi\left(\frac74\right)\right)
\ge g_{5\to6}(1)+\frac34\big(g'_{5\to6}(1)-\delta_{56}\big)-\frac9{32}S_3
>\frac\pi2+\frac1{32}. \tag{P2}
\]

For the seven remaining donor bulk edges `E_D,bulk\{5->6}`, the donor cycle sum
`sum_{j=0}^8 g_{D,j}=4*pi` and the telescoping identity
`g_{4->5}+g_{5->6}=8*pi/9+xi_6-xi_4` eliminate the fast-moving port `xi_5`
identically:

\[
\sum_{e\in E_{D,{\rm bulk}}\setminus\{5\to6\}}g_e(\psi(\sigma))
=\frac{28\pi}9-\big(\xi_6(\sigma)-\xi_4(\sigma)\big).
\]

Under the same all-time PSD hypothesis,
`||e_6-e_4||_{D^-1}=sqrt(1/2+1/3)=sqrt(5/6)` gives at `sigma=7/4`
`xi_6(7/4)-xi_4(7/4)<=(xi_6^(4)-xi_4^(4))+sqrt(5/6)*(R^(4)+(3/4)||psi'(1)||_D)=:xi_64_bar`,
and the enlarged primitive intervals keep all eight donor bulk lifts in
`(0,pi)`. Equation (P2) makes the active bulk edge strictly obtuse at this
hypothetical endpoint. If any of the seven remaining bulk cosines were
nonpositive, (C3) would contradict PSD there. Thus those seven gaps must lie
in `(0,pi/2)`. This uses the assumed PSD premise, not an independent acuteness
certificate. Concavity of cosine on `[0,pi/2]` (Jensen's inequality) now yields

\[
\frac1{49}\sum_{e\in E_{D,{\rm bulk}}\setminus\{5\to6\}}
 \cos g_e\!\left(\psi\left(\frac74\right)\right)
\le\frac17\cos\!\left(\frac{4\pi}9-\frac{\overline\xi_{64}}7\right)
<\frac3{100}. \tag{P3}
\]

Combining the unrounded primitive bounds used in (P2) and (P3) gives the
following **contradiction bound under PSD throughout `[1,7/4]`**:

\[
q_{\rm bulk}\!\left(g\left(\psi\left(\frac74\right)\right)\right)
=v^T\mathcal H\!\left(g\left(\psi\left(\frac74\right)\right)\right)v
<-\frac1{256}<0, \tag{P4}
\]

contradicting (C4). The weaker displayed bounds `1/32` in (P2) and `3/100`
in (P3) already give a strictly negative upper bound; the sharper `-1/256`
uses the retained primitive intervals. Thus the exact reference trajectory
from `phi=1/1000`, which is initially in the PSD region, must leave it at
some time before `sigma=7/4`. This is an existence-of-exit conclusion.
It neither gives the actual `q_bulk` at `7/4`, nor excludes later reentry;
the nonincreasing-speed premise cannot be carried past its first failure.
The conditional bounds are verified by
`certify_formed_two_port_fifth_step_acute_exit` in
`tnfr.research.formed_two_port_reference`, whose `psd_contradiction_*` fields
record their hypothesis. The `hessian_exit_horizon` records an upper horizon
for existence of an exit, not a certified terminal state. This reference
result does not by itself prove Hessian exit for the actual complete law.

### Admissible contact-origin window and complete formed-source capture

In the [isolated formation and dwell proof](SINE_FORMED_CLASS_CONTACT.md#the-actual-formation-images-reach-the-contact-tolerance),
the donor and receiver C9 rings evolve on disjoint supports throughout
`0<=tau<=tau_c` (`tau_c=100+10^13`). On the isolated receiver ring, the sine
law depends only on internal phase differences `theta_{R,j}-theta_{R,i}`.
Replacing the single-bridge receiver offset `phi=1/1000` in the initial
preparation by any supplied common receiver phase origin `phi_0` is an **exact
symmetry** of the isolated receiver dynamics: it leaves the initial storage
ceiling `10^9`, acquisition at `tau=100` (`r=1/12`), and the strict nonlinear
Lyapunov dwell bound at `tau_c=100+10^13` (`||v_D||_2,||u_D||_2,||v_R||_2,||u_R||_2<=epsilon=10^-32`,
with zero-sum residuals) identically unchanged.

Admit any supplied receiver common phase origin `phi_0` in the midpoint window

\[
\left|\varphi_0-\frac\pi9\right|\le r_\varphi
:=\frac1{32768}-2\epsilon,\qquad \epsilon=10^{-32}. \tag{O1}
\]

In particular, the exact rational `phi_0=355/1017=(1/9)*(355/113)` built from
the rational `355/113` already used in the formation slope
`m=(2046/9)*(355/113)^2` satisfies `|355/1017-pi/9|<3*10^-8<r_phi`. At
simultaneous two-port contact `tau_c` on `(D4,R4)` and `(D5,R5)`, the contact
gaps `g_4=phi_0+u_{R,4}-u_{D,4}` and `g_5=phi_0-2*pi/9+u_{R,5}-u_{D,5}` obey
`|g_4-pi/9|,|g_5+pi/9|<=r_phi+2*epsilon=1/32768`, so the exact contact work
(A4) satisfies

\[
W_2=\sum_{j=4,5}\left[\tfrac12(v_{R,j}-v_{D,j})^2+1-\cos g_j\right]
\le2\epsilon^2+2\left[1-\cos\left(\frac\pi9+\frac1{32768}\right)\right]
<\frac18<\frac14. \tag{O2}
\]

Transport the post-contact state by the exact relabeling
`i |-> j=(i+4) mod 9` and the `+4*pi/+2*pi` lift restoration of (A3) for
`i=5,...,8`. In these raw ramp coordinates the nominal donor and receiver
phases are `4*pi*i/9` and `2*pi*i/9+phi_0`. Put
`delta_phi=phi_0-pi/9` and `mu_*=229*pi/180`, the midpoint nominal mean
specified above (2). Apply the **complete common nominal gauge shift**

\[
-\left(\mu_*+\frac{\delta_\varphi}{2}\right)\mathbf1
=-\left(\frac{\varphi_0}{2}+\frac{73\pi}{60}\right)\mathbf1.
\]

Call the transformed coordinates `hat x,hat theta`. This common phase shift
changes no edge gap and commutes with (A2). The resulting phase deviations
from (2) are exactly
`hat theta_{D,i}-2*pi*Theta^0_{D,i}=-delta_phi/2+u_{D,j}` and
`hat theta_{R,i}-2*pi*Theta^0_{R,i}=+delta_phi/2+u_{R,j}`.
The nominal states coincide only when `delta_phi=0`. The residuals are
permuted unchanged, with no additional memberwise centering; forms are also
only permuted. Therefore every member of this **shifted** thirty-six-variable
formed-source family satisfies, for all eighteen coordinates,

\[
|\widehat x_i(\tau_c)|\le\epsilon=10^{-32}<r_x=\frac1{65536},\qquad
|\widehat\theta_i(\tau_c)-2\pi\Theta_i^0|
\le\frac12\left|\varphi_0-\frac\pi9\right|+\epsilon
\le r_\theta=\frac1{65536}. \tag{O3}
\]

Equation (2) allows these nodewise errors without a zero-error-mean condition.
Each transformed member retains its own joined form mean and its residual
phase mean, rather than being reset to the nominal member's means. The
original circular trajectory is recovered by the inverse chart map.

The scalar inclusion bounds are checked by `admit_formed_two_port_origin_window`
in `tnfr.research.formed_two_port_reference`; `transport_formed_two_port_origin_source`
implements the full-state chart while retaining the supplied residuals and
memberwise means. Its residual checks do not authenticate isolated acquisition.
Because every formed-source image in
(O1) satisfies (2) and (O2), the already retained and read-only audited
4096-step capture certificate (`two-port-capture-v1.json`, verified by
`audit_sine_two_port_capture_handoff`) and interior dipole theorem (A9)--(A11)
apply directly without a new trajectory evaluation:

- before contact and in the disconnected continuation, `|D_s|<=2*epsilon` keeps
  the strict initial-absence margin `>1/5-2*epsilon` from the internal
  deformation collar `\mathcal I` (A6)--(A7);
- at `tau_b=tau_c+1025*1023^2*pi^2`, every shifted formed-source trajectory with
  `phi_0` in (O1) enters the full-state capture basin (A8) (`R<1/12`,
  `H-H*<1/648000`) on its actual post-contact conserved-mean leaf (A3), with
  `phi` replaced by the supplied `phi_0`,
  acquires and maintains `\mathcal I` (`D_s>1/5`) for all `tau>=tau_b`, and
  converges to the unique `(2,1)` deformed two-port geometry;
- after the unchanged warmup (A9), the interior dipole probe (A10)--(A11)
  separates the joined two-port organization from the disconnected control.

The independent phase shift is a symmetry of isolated evolution, not of the
joined support: changing `phi_0` changes both initial contact gaps. This
inclusion supplies capture for a different preparation; it neither changes
the retained `phi=1/1000` source nor decides that source's eventual fate.

### Pre-contact relative-origin equivariance and scope of the entry result

On the disjoint pre-contact support `C_D union C_R` (`0<=tau<tau_c`), the
one-parameter family of opposite component phase shifts

\[
T_\psi(x_D,\theta_D,x_R,\theta_R)
=\left(x_D,\ \theta_D-\tfrac12\psi\mathbf1_D,\ x_R,\ \theta_R+\tfrac12\psi\mathbf1_R\right)
\tag{O4}
\]

leaves all within-component phase differences `\theta_{C,j}-\theta_{C,i}`, all
form differences `x_{C,j}-x_{C,i}`, both component storages, and the total
joined weighted phase mean unchanged, while shifting the relative contact
origin by `\varphi_0 \mapsto \varphi_0+\psi`. Because the disjoint vector field
depends only on within-component differences, `T_\psi` commutes with the
complete pre-contact flow for all `0<=tau<tau_c`:

\[
\Phi_\tau^{\rm disjoint}\circ T_\psi
=T_\psi\circ\Phi_\tau^{\rm disjoint}. \tag{O5}
\]

Equation (O5) makes the absence of a preferred relative origin a property of
the disjoint law. In the retained complete rows, each component's weighted
phase mean is constant (`dot{bar theta}_C=0`), so `phi_0` does not drift over
`[0,tau_c)`. Observations **invariant under independent component phase
shifts**, such as internal phase differences and component storage, cannot
distinguish the shifted preparations. This does not cover every internal
observation: a retained phase mean relative to a supplied external or donor
reference can distinguish origins. Such observation does not create
autonomous selection or attraction by the disjoint law.

The distinction leaves separate possible preparation mechanisms, not
established alternatives for the original trajectory:

1. **Supplied relative origin:** initializing `\varphi_0` in (O1) (such as
   `\varphi_0=355/1017`), which is admitted by (O1)--(O3);
2. **Cross-component contact guard with relative drift:** gating contact
   admission on both port differences `|\theta_{R,4}-\theta_{D,4}-\pi/9|` and
   `|\theta_{R,5}-\theta_{D,5}+\pi/9|` while supplying a nonzero relative phase
   drift `\dot{\bar\theta}_R-\dot{\bar\theta}_D\ne0`. This would require a
   changed pre-contact law or input and its work/mean accounting; a guard
   alone supplies no drift under the retained disjoint rows; or
3. **Pre-contact coupling:** introducing weak cross-component support prior to
   `\tau_c`, which is a different support model from disjoint formation.

Equations (C1)--(C13), (P1)--(P4) and (O1)--(O5) establish the following
bounded conclusions:

- At the original offset `phi=1/1000`, both the reference and every actual
  thirty-six-variable trajectory leave the acute cell on `(1,5/4)`, and the
  reference has a Hessian-positivity exit before `sigma=7/4`. These results
  exclude the respective all-time comparison premises. They do not certify
  a later winding slip, loss of identity, eventual non-capture, or the
  reference's actual curvature at `7/4`.
- The admissible contact-origin window (O1) (`|phi_0-pi/9|<=1/32768-2*10^-32`,
  including `phi_0=355/1017`) uses isolated-source equivariance to place the
  shifted formed-source family inside the certified midpoint capture box
  (2). It establishes entry, internal-deformation acquisition and conditional
  dipole response for that supplied preparation from retained evidence.
- By (O4)--(O5), disjoint pre-contact evolution is exactly equivariant under
  relative phase shifts, so `phi_0` remains a supplied preparation coordinate
  in that model. A contact guard, drift law or pre-contact coupling would need
  separate admission. The original `phi=1/1000` family's post-acute evolution
  is treated below; its [later target exclusion](#sine-formed-two-port-actual-exclusion)
  and [original-period exclusion](SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-original-period-exclusion)
  supersede the entry question left open at this origin-decision stage.
  Autonomous contact occurrence and physical mapping remain separate obligations.


<a id="sine-formed-two-port-post-acute-control"></a>
## Post-acute control and convergence for the original complete law

The original `phi=1/1000` preparation admits two further conclusions without
a new trajectory. First, the actual phase remains within `73/200000` in the
degree norm of its mean-matched reference through `sigma=7/4`, even after
acute and reference-Hessian exits. Second, every member of the complete
formed-source family converges to one equilibrium under the unchanged
post-contact law. Neither result identifies that equilibrium or proves entry
into the deformed `(2,1)` basin. The supplied origin window (O1) remains a
different preparation.

Retain both rows (A2), the original contacts `(4,13),(5,14)`, degree matrix
`D`, continuous lifts, `phi`, `tau_c`, capacities and coefficients. Put
`A=D^-1 L`, `eta=gamma^2` and `sigma=eta*(tau-tau_c)`. Each member keeps its
own conserved weighted means `mu_x,mu_theta`. The following changes of
coordinates and time origins do not reset its state or repeat the consumed
reference attempt.

### A finite continuation bound without an acute potential floor

Use the exact mixed coordinates of the
[form-phase owner](SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence):
`z=gamma P_D x`, `y=theta+z`, and the same nominal reference (M1) translated
only by the member's common phase mean. With `f(theta)=D^-1 S(theta)`,

\[
z_\sigma=-Az/\eta+f(\theta),\qquad
y_\sigma=f(\theta),\qquad \psi_\sigma=f(\psi). \tag{PA1}
\]

The [global slow comparison](SINE_FORM_PHASE_REDUCTION.md#sine-controlled-slow-phase)
already supplies the relevant domain-independent ingredients. Here
`|f_i|<=1` and `sum_i d_i=40`, so `||f||_D<=sqrt(40)<7` on all real lifts.
The normalized signed-cosine Hessian lies between the negative and positive
normalized graph Laplacian, whose norm is at most two. Thus `f` is globally
two-Lipschitz and has logarithmic growth at most two in the same metric.
Also retain the graph quotient gap `lambda=1/90` and
`eta/lambda<1/100000`. No phase floor, acute chart or positive Hessian is
needed for these global facts.

At `sigma=1`, consume the same four whole-time certificates admitted above
(C10). Their earlier complete-law comparison remains valid. Applying the
[dissipated-storage estimate](SINE_TWO_PORT_TRANSIT.md#sine-energy-informed-slow-comparison)
on this already admitted prefix with `Sigma=1` and `E0=1/4`, rather than the
prospective long horizon 1025, gives

\[
\begin{aligned}
V_1:=\|y(1)-\psi(1)\|_D
&\le v_0+\frac{2\eta\sqrt{1/4}}\lambda
 <\frac1{90000},\\
Z_1:=\|z(1)\|_D&<\frac1{100000}.
\end{aligned} \tag{PA2}
\]

Here `v0=3*epsilon+3*epsilon/3069`, `epsilon=10^-32`; the second bound is
already provided by (M6)--(M7) on the prefix. The shorter analytical estimate
changes no step, tolerance, source, endpoint or producing policy. In
particular, `V1` is an inherited mismatch, not a newly zeroed reference error.

Let `h=sigma-1>=0` and `a=lambda/eta`. The forcing in (PA1) has zero weighted
mean, so variation of constants on the graph quotient gives

\[
\|z(1+h)\|_D
 \le Z_1 e^{-ah}+\frac7a(1-e^{-ah})
 \le Z_*:=\frac7{100000}. \tag{PA3}
\]

For `r(h)=||y(1+h)-psi(1+h)||_D`, global Lipschitz continuity implies the
upper Dini inequality `D^+r<=2r+2Z_*`, including zero error. Hence

\[
r(h)\le e^{2h}V_1+Z_*(e^{2h}-1),\qquad
\|\theta(1+h)-\psi(1+h)\|_D
 \le e^{2h}(V_1+Z_*). \tag{PA4}
\]

This does not assume that either trajectory remains in the previous
comparison corridor. For `0<=h<=3/4`, a rational series bound suffices:
the exponential series after its quadratic term at `1/2` has first term
`1/48` and successive ratios at most `1/8`. Therefore
`exp(1/2)<=13/8+(1/48)/(1-1/8)=277/168<33/20`, and
`exp(3/2)<(33/20)^3=35937/8000<9/2`. Consequently

\[
\|\theta(\sigma)-\psi(\sigma)\|_D
 <\frac92\left(\frac1{90000}+\frac7{100000}\right)
 =\frac{73}{200000}<\rho=\frac1{2048},
\qquad 1\le\sigma\le\frac74. \tag{PA5}
\]

This bound concerns the exact reference, not an uncomputed numerical center
after time one. Its retained numerical radius at one remains a separate
quantity. The source-bound consumer
`bound_formed_two_port_post_acute_window` re-admits the same four-step prefix
and checks these scalar bounds; it computes no future phase state, target
or continuation step. The original form is still reconstructed exactly as
`x=mu_x*1+z/gamma`. Since `1/gamma<3216`, (PA3) also gives
`||P_D x||_D<1407/6250` on this interval.

The form bound is not the small final-unit form estimate (M8), and (PA5)
supplies no target distance. Thus neither bound admits (M9) or a new transit.
Nor does a small phase error turn the conditional terminal expression (P4)
into a real reference value at `7/4`, or prove actual-law Hessian exit.
Positive global growth permits this finite estimate but gives no useful
long-horizon capture bound by itself.

### The complete law converges without selecting an equilibrium

The [complete-contact convergence method](ORGANIZED_CONTACT_MAINTENANCE.md#full-contact-convergence)
extends to this irregular degree metric with the following explicit changes.
This is an own-law statement for the complete post-contact state, rather
than an inference from the gradient reference. Define

\[
\omega=D^{1/2}\mathbf1,\quad \mathcal V=\omega^\perp,\quad
q=D^{1/2}(\theta-\mu_\theta\mathbf1),\quad
\xi=D^{1/2}(x-\mu_x\mathbf1),\quad
L_n=D^{-1/2}LD^{-1/2}\big|_{\mathcal V}.
\]

On `mathcal V`, `L_n` is positive definite with spectrum in `[1/90,2]`.
For `Phi(q)=U(D^-1/2 q+mu_theta*1)`, put
`v=q_tau=gamma L_n xi` and `g=eta grad Phi`. The exact complete rows give

\[
L_n^{-1}v_\tau+v+g=0,\qquad
E=\tfrac12v^TL_n^{-1}v+\eta\Phi=\eta H,\qquad
E_\tau=-\|v\|^2. \tag{PA6}
\]

The inverse acts only on the relative space; there is no extra common-phase
velocity. The common form need not vanish. This mathematical mechanical
representation introduces no physical mass or new evolution law.

The relative circular phase belongs to the compact flat torus

\[
\mathcal V/\Gamma,\qquad
\Gamma=2\pi D^{1/2}\{n\in\mathbb Z^{18}:d^Tn=0\},\quad
d=(d_0,\ldots,d_{17})^T. \tag{PA7}
\]

The integer kernel lattice has rank seventeen and spans `d^perp`;
multiplication by `D^(1/2)` maps that hyperplane isomorphically onto
`mathcal V`. This quotient retains continuous lifts locally. Their
boundedness is not presumed. The potential is real analytic and periodic
on this torus. Its normalized signed Hessian obeys
`-L_n<=Hess Phi<=L_n`, so `||Hess(eta Phi)||<=2eta` globally.

The field is globally Lipschitz in the complete lifted coordinates; energy
nonnegativity and (PA6) bound `xi` and `v` on their relative space. Thus
solutions exist for all forward time and the form/circular-phase orbit is
precompact. LaSalle's invariance argument applied to (PA6) puts its compact
omega-limit set in `v=0,g=0`: invariance of `v=0` requires `g=0` because
`L_n` is invertible. In particular `v->0`, `g->0` and `E->E_infinity`.
The only potential floor needed here is the global identity `U>=0`.

To strengthen convergence to the stationary set into convergence to a
single point, use the auxiliary analytic proof function
`mathcal F=E+g^T v/2`. With `G=Hess(eta Phi)`, differentiation and Young's
inequality give

\[
\begin{aligned}
\mathcal F_\tau
 &=-\|v\|^2+\tfrac12v^TGv-\tfrac12g^TL_nv-\tfrac12g^TL_ng,\\
|g^TL_nv|&\le\tfrac12g^TL_ng+\tfrac12v^TL_nv,\\
\mathcal F_\tau
 &\le-(\tfrac12-\eta)\|v\|^2-\tfrac14g^TL_ng
 \le-\frac{\|v\|^2+\|g\|^2}{360}. \tag{PA8}
\end{aligned}
\]

The last inequality uses `eta<1/4` and the quotient gap, not positive
phase curvature. `mathcal F` is a proof function, not a new storage or
installed diagnostic. It is bounded below on the precompact orbit,
decreases, and tends to `E_infinity`, since its added term tends to zero.
Every omega-limit point is a critical point of this same analytic function
at that same value.

Use the local analytic gradient inequality, as in the existing owner and
the proof of Corollary 5.1 in
[Begout, Bolte and Jendoubi, version 5](https://arxiv.org/html/1411.8005v5).
Its local application here does not import a bounded Euclidean phase-lift
assumption. A finite coordinate-chart cover of the compact omega-limit set
and the minimum of its positive exponents yield, for sufficiently late time,

\[
s^{1-\rho_L}\le C\|\nabla\mathcal F\|
 \le91C(\|v\|+\|g\|),\qquad
s=\mathcal F-E_\infty\in[0,1),\quad 0<\rho_L\le\tfrac12. \tag{PA9}
\]

Here `grad_q mathcal F=g+Gv/2`, `grad_v mathcal F=L_n^-1 v+g/2`, and
`||L_n^-1||<=90` justify the last constant. The orbit eventually lies in
that finite cover. If `s` reaches zero at finite time, (PA8) makes the
state stationary thereafter. Otherwise (PA8)--(PA9) imply

\[
-\frac{d}{d\tau}s^{\rho_L}
 \ge\frac{\rho_L}{65520C}(\|v\|+\|g\|),\qquad
\int_{\tau_c}^{\infty}(\|v\|+\|g\|)\,d\tau<\infty. \tag{PA10}
\]

The finite initial interval contributes finite length. Since `q_tau=v`,
the original continuous relative lift is Cauchy and converges; its boundedness
is a consequence. Also `xi_tau=-(v+g)/gamma`, so the form has finite total
variation. Restoring each member's means gives the complete limit

\[
x(\tau)\longrightarrow\mu_x\mathbf1,\qquad
\theta(\tau)\longrightarrow\theta_\infty,
\qquad S(\theta_\infty)=0. \tag{PA11}
\]

The limit can depend on the member. This proves neither uniqueness of the
network's equilibrium nor stability of the selected limit. The constants
and exponent in the local analytic step supply no explicit settling time
or uniform source-to-target rate. A limit may lie on a principal cut;
finite total phase variation does not exclude infinitely many cut crossings
when that happens. If the limit is cut-free, the principal windings eventually
stabilize, without this proof selecting their values.

### Energy and conserved means do not identify the attained geometry

Two explicit equilibria illustrate this remaining distinction. The consensus
state `x=mu_x*1`, `theta=mu_theta*1` has `H=0`. Alternatively choose any
degree-two node, give it phase `pi` and every other node phase zero, and then
add the common shift `mu_theta-pi/20`. Its weighted phase mean is exactly
`mu_theta`. With the same uniform form, all sine currents vanish, and its two
incident antipodal edges contribute `H=4`. Both states therefore have the
member's conserved means and lie below the source ceiling
`U_floor+1/4`, since `U_floor>9/2`. The latter inequality follows already
from the donor contribution `9[1-cos(4*pi/9)]>9/2`.

These are witnesses against selection by only that ceiling and the means.
They are not claimed reachable from the original formed source, and neither
replaces its initial state or fixed continuous lifts. In particular the
antipodal witness shows why a cut-free limit cannot follow from those data
alone. The retained compatible target and its local basin remain valid;
connecting the original family to that basin, or excluding its entry,
requires the source-dependent information supplied by the later
[actual-source exclusion](#sine-formed-two-port-actual-exclusion). No new numerical
policy, autonomous contact mechanism or physical mapping is admitted here.

<a id="sine-formed-two-port-source-transport"></a>
## Source-dependent velocity control and a conditional outgoing cut

For the original `phi=1/1000` family, two kinds of transport information
can now be kept separate. A global bound controls the lag of the **actual**
phase velocity behind its own instantaneous sine field after a short
initial layer. The admitted prefix and (PA5) also prove that every
principal edge remains away from its cut through `sigma=7/4`, retaining
periods `(2,1,0)`. Neither result selects the eventual equilibrium.
An explicit outgoing neighborhood of the already constructed principal-cut
witness gives a further conditional criterion, not evidence of source arrival.

Keep the complete A2 law, original support, unit capacities, held coefficients,
continuous lifts and each member's conserved means. There is no subsequent
input or event. The clock remains
`sigma=gamma^2*(tau-tau_c)`, `tau=e*t`, `tau_c=100+10^13`.
No reference trajectory, source reset, target sign or numerical continuation
is evaluated in this argument.

### A global velocity-lag bound from both actual evolution rows

Write `A=D^-1 L`, `eta=gamma^2`, `f(theta)=D^-1 S(theta)` and define

\[
v=\theta_\sigma=A P_Dx/\gamma,\qquad e_v=v-f(\theta).
\]

The [exact second-order identity](SINE_FORM_PHASE_REDUCTION.md#form-reconstruction-scope)
retains the initial velocity, hence the original relative form. Differentiating
with both A2 rows in the same slow clock gives

\[
v_\sigma=-\frac A\eta\,[v-f(\theta)],\qquad
(e_v)_\sigma=-\frac A\eta e_v-Df(\theta)[e_v+f(\theta)].
\tag{ST1}
\]

Both `v` and `f` have zero `D`-weighted mean. The graph quotient gap
`lambda=1/90`, `||A||_D<=2`, `||Df||_D<=2` and `||f||_D<7` hold globally
on the fixed real lifts, without acute or Hessian-positivity assumptions.
Since `eta/lambda<1/100000`, the upper Dini derivative satisfies

\[
D^+\|e_v\|_D
\le-\left(\frac\lambda\eta-2\right)\|e_v\|_D+14
\le-99998\,\|e_v\|_D+14. \tag{ST2}
\]

At zero norm this follows by continuity of the vector field; elsewhere it
follows by taking the weighted inner product with `e_v/||e_v||_D`.
The source bound (M3) and `1/gamma<3216` give
`||v(0)||_D<19296*epsilon`. The nominal source field has norm below `3/5`
by (M5); its actual mean-matched phase differs by at most `3*epsilon`.
Thus, for `epsilon=10^-32`,

\[
\|e_v(0)\|_D
< L_0:=\frac35+19302\epsilon<1.
\]

Integrating (ST2) and using
`99998/128>512` and `exp(-512)<2^-512` yields the uniform bound

\[
\boxed{
\|\theta_\sigma-f(\theta)\|_D
< L_0\,2^{-512}+\frac{14}{99998}
<\frac1{7000},\qquad \sigma\ge\frac1{128}.
} \tag{ST3}
\]

This is source-dependent control of the full law, not an exact autonomous
phase equation or an all-time bound on `||theta-psi||`. It compares the
velocity with `f` at the **actual phase**, not at the independently evolved
reference. An arbitrary phase path allowed by the resulting differential
inclusion need not be a trajectory from the original family. The initial
form and memberwise means have not been discarded.

### No principal-cut crossing through the admitted post-acute interval

The re-admitted four-step reference lies in the buffered corridor with
`kappa=1/1024=2*rho`. In particular, at time one every fixed-branch reference
edge satisfies `|g_e(psi(1))|<pi/2+rho`; this uses the whole-tube corridor,
not a rounded endpoint or a fresh field evaluation. Each component of the
reference field obeys `|f_i|<=1`, so
`|g_e(psi(sigma))-g_e(psi(1))|<=2*(sigma-1)` globally.

For `1<=sigma<=7/4`, (PA5), edge dual norms at most one and `pi>25/8` give

\[
\begin{aligned}
\pi-|g_e(\theta(\sigma))|
&>\frac\pi2-\rho-\frac32-Q_{\rm PA}\\
&>\frac1{16}-\rho-Q_{\rm PA}
>\frac{63}{1024}>\frac3{50},
\qquad Q_{\rm PA}<\rho.
\end{aligned} \tag{ST4}
\]

On `[0,1]`, the original full-law first-exit comparison keeps the actual
phase inside the unbuffered corridor; its principal margin exceeds
`pi/2-kappa`, which is larger still. Therefore **all twenty edges of every
actual source member stay strictly between their original principal cuts
through `sigma=7/4`, with margin greater than `3/50` radians**.
The donor, receiver and interface periods remain `(2,1,0)` on that entire
interval. Acute exit and reference-Hessian exit can occur within this
cut-free evolution. No numerical reference state after time one was used.

This excludes a completed donor winding slip by `7/4`; it does not exclude
a later slip or membership in a global attraction basin whose trajectories
eventually change winding. Current winding labels also do not prove entry
into the retained deformed `(2,1)` target's small full-state basin.

### A quantitative outgoing patch at the existing principal-cut witness

For a fixed relative-phase face `h(theta)=ell^T theta-c`, (ST3) gives

\[
\left|h_\sigma-\ell^\top f(\theta)\right|
<\frac1{7000}\sqrt{\ell^\top D^{-1}\ell},
\qquad \sigma\ge1/128. \tag{ST5}
\]

Thus a strict whole-face bound on `ell^T f`, separated from zero by this
allowance, supplies the actual direction. Equality or an overlapping bound
does not. Admission of the phase region and its source reachability remains
a separate obligation.

Use the [existing SR11--SR14 witness](SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-stationary-reduction),
transported to the original labels and fixed continuous lifts. Its donor
gap `5->6` is `pi`; its other eight donor gaps are `3*pi/8`, its nine
receiver gaps are `2*pi/9`, and its two contacts are
`+11*pi/144,-11*pi/144`. These correlated gaps have affine sums
`(4*pi,2*pi,0)` and define one nodal phase modulo a common shift.
Match that common shift to each member's conserved phase mean.

At this phase, with the actual degrees `d_5=3,d_6=2`, the gradient-field
normal to the active cut is

\[
f_6-f_5
=\frac56\sin(3\pi/8)+\frac13\sin(11\pi/144)
\ge\frac56\frac34+\frac13\frac{11}{72}
=\frac{73}{108}.
\]

The chord bound `sin(a)>=2a/pi` on `[0,pi/2]` supplies both estimates.
No trigonometric sign search is needed. In the mean-matched `D`-ball of
radius `1/256` about this fixed lifted phase, global Lipschitz continuity
changes this normal by at most `2/256`. Combining with (ST3),

\[
\boxed{
(g_{5\to6})_\sigma
>\frac{73}{108}-\frac1{128}-\frac1{7000}
>\frac23.
} \tag{ST6}
\]

All other edges remain cut-free in this small neighborhood. Consequently,
**if** an actual source trajectory reaches its `g_{5->6}=pi` face after the
initial layer, the crossing is locally outward and changes the principal
periods from `(2,1,0)` to `(1,1,0)`. No independent vertex wrapping,
component recentering or replacement initial state is admitted. A later
recrossing elsewhere is not excluded, and a local slip selects no final
equilibrium.

This gives an explicit directed cut patch for a source-to-boundary argument,
not a preferred destination selected for the original trajectory.
It also shows why phase-period constraints alone do not provide a
uniformly inward boundary test: this patch has the opposite sign.
The old energy witness establishes neither arrival at the patch nor that
the original source must avoid it.

### A conditional full-state handoff from actual target proximity

The same global estimate supplies the complementary destination test.
Let `theta_*^mu` be the exact retained deformed `(2,1)` critical phase,
shifted to the actual member's conserved phase mean. If, at any
`sigma>=1/128`,
`||theta-theta_*^mu||_D<=delta=1/1024`, then
`||f(theta)||_D<=2*delta`, since `f(theta_*^mu)=0`.
On the mean-zero subspace, `||A^-1||_D<=90` and
`P_Dx=gamma A^-1 v`. Hence, writing `L_v` for the bound in (ST3),

\[
\|P_Dx\|_D
<\frac{90}{3069}(2\delta+L_v)<\frac1{8192},\qquad
H-H_*\le\|P_Dx\|_D^2+\delta^2<\frac1{648000}. \tag{ST7}
\]

The energy estimate uses the exact target's zero phase gradient and the
global normalized Hessian upper bound two. The combined phase/form radius
is also below `1/12`, so the existing full-state basin (A8) applies.
This is a new sufficient **actual-state** handoff, not reinstatement of
the consumed reference schedule or a claim that its endpoint was reached.
It requires distance to the exact target, including any target uncertainty
in a later numerical use; a rounded target or winding label is insufficient.
The original common form is retained separately. No source member has
been proved to enter this phase ball.

<a id="sine-formed-two-port-relative-progress"></a>
### Source-conditioned relative lag and actual phase-potential descent

The fixed lag allowance in (ST3) can be strengthened by retaining how the
forcing of that lag depends on the actual phase field. Keep exactly the
original A1 family and complete A2 law. Put
`E=||e_v||_D`, `F=||f(theta)||_D` and `s_0=1/128`; this `E` is an error
norm, not the mechanical energy in (PA6). From (ST1), almost everywhere,

\[
 E'\le-99998E+2F,\qquad |F'|\le2(E+F).
\]

Both norms are locally absolutely continuous. These inequalities use the
norm derivative bound also at zeros, without dividing by either norm.
The global (ST2) comparison and `E(0)<1` give `E<1`; with `F<7`, the
actual speed is below eight on the whole initial layer. Hence
`|F(sigma)-F(0)|<=16*sigma<=1/8` there.

For the nominal original phase, only the four contact rows contribute:
`F_nom(0)^2=(2/3)[sin(phi)^2+sin(phi-alpha)^2]`, where
`alpha=2*pi/9` and `phi=1/1000`. The source bounds give
`2/3<alpha-phi<7/10`. Since `sin(2/3)>50/81` and
`sqrt(2/3)>4/5`, while mean matching changes phase by at most
`3*epsilon`, global Lipschitz continuity and (M5) imply

\[
 \frac{40}{81}-6\epsilon<F(0)<\frac35+6\epsilon,
 \qquad \frac13<F(\sigma)<\frac34
 \quad(0\le\sigma\le s_0),\qquad \epsilon=10^{-32}.
\]

On this layer the sharper forcing is `2F<3/2`, so scalar comparison gives

\[
 E(s_0)<2^{-512}+\frac3{199996}<\frac1{60000}.
 \tag{ST8}
\]

For the last strict inequality one may replace `2^-512` by `2^-20`;
all remaining arithmetic is rational. No future field value is evaluated.

Set `k=1/20000` and `Q_k=E-kF`. The coupled inequalities imply

\[
 Q_k'\le-(99998-2k)Q_k+P(k)F,
 \qquad P(k)=2-99996k+2k^2<-2.
\]

By (ST8), `Q_k(s_0)<0`. Dropping the nonpositive term `P(k)F` and
integrating shows `Q_k(sigma)<0` at every finite later time. In particular
`F` cannot vanish at such a time, since that would require `E<0`.
This proves, for every actual source member and every finite `sigma>=s_0`,

\[
 \boxed{\begin{aligned}
 &0<F,\qquad E<kF,\qquad
 (1-k)F<\|\theta_\sigma\|_D<(1+k)F,\\
 &-(1+k)F^2<U_\sigma<-(1-k)F^2<0,
 \qquad k=\frac1{20000}.
 \end{aligned}} \tag{ST9}
\]

Indeed, `U_sigma=-<f,theta_sigma>_D=-F^2-<f,e_v>_D`.
This is strict descent of the **actual phase potential** after a
source-dependent initial layer, without acute or positive-Hessian confinement.
It strengthens the earlier total-storage and auxiliary-function descent;
those did not assert monotonicity of phase potential by itself. It does not
assert descent before `s_0`, identify the limiting critical point or derive
a new constitutive law. The unchanged form remains
`P_Dx=gamma A^-1 theta_sigma`, with its conserved mean retained.

For a phase face with `ell^T 1=0`, combine (ST3) and (ST9):

\[
 |\ell^T(\theta_\sigma-f(\theta))|
 <\sqrt{\ell^TD^{-1}\ell}\,
       \min\left\{\frac1{7000},\frac{F}{20000}\right\}.
 \tag{ST10}
\]

The relative term improves the old allowance when `F<20/7`; no uniform
improvement from that term alone is claimed at every phase. Whole-face signs
must use the correlated values of the field and its allowance, or a proved
uniform bound for both. A constant-error phase inclusion permits a spurious
stationary path at any noncritical phase with `F<1/7000`, by choosing
`e_v=-f`; the relative cone excludes that path. This observation supplies
no such point reachable from the original family. True critical phases
remain possible destinations, and the admitted phase inclusion still contains
more paths than the complete source dynamics.

<a id="sine-formed-two-port-transport-seed"></a>
### An explicit correlated phase enclosure at the retained endpoint

The new speed comparison gives a bounded source-containing set at
`sigma_b=7/4` without evaluating a later reference. Let `c_1` be the full
phase center reconstructed from the retained time-one prefix,
`c_1=psi(0)+xi^(4)`, not its raw affine deviation coordinates. Keep its certified radius `R_1<2^-26`.
For each member, project that center to its conserved mean:
`c_1^mu=P_D c_1+mu_theta*1`. Projection is nonexpansive in the `D` metric;
(C10) and (PA2) therefore give

\[
 \|\theta(1)-c_1^\mu\|_D<R_1+b,\qquad
 F(1)<\frac14+2b,\qquad
 b:=\frac1{45000}>
       \frac1{90000}+\frac1{100000}.
\]

Using `F'<=2(1+k)F` and (ST9), for `0<=h<=3/4`,

\[
 \|\theta(1+h)-c_1^\mu\|_D
 <R_1+b+\left(\frac18+b\right)
              \left(e^{2(1+k)h}-1\right).
 \tag{ST11}
\]

The existing series bound `exp(3/2)<35937/8000` and
`exp(3/40000)<=40000/39997` imply
`exp((3/2)(1+k))<9/2`. Consequently the right side at `h=3/4` is below
`7/16+1/10000+2^-26<9/20`.

Define the closed enclosure `K_b(mu_theta,H_0)` by the simultaneous conditions

\[
 \mathbf1^TD\theta=40\mu_\theta,\qquad
 \|\theta-c_1^\mu\|_D\le\frac9{20},\qquad
 |g_e(\theta)|\le\pi-\frac3{50}\quad\hbox{for all twenty edges},\qquad
 U(\theta)\le H_0. \tag{ST12}
\]

Here `H_0` is that member's admitted initial total-storage upper bound, or a
common bound covering all original members. The `g_e` retain the original
fixed affine branches: their three cycle sums are exactly `(4*pi,2*pi,0)`.
Nodal realizability, all cycle relations and the mean are simultaneous
constraints, not independent edge boxes. (ST4), (ST11) and nonincreasing total
storage show that **every original member's actual phase at `7/4` belongs to
this compact enclosure**. A common shift neither folds the two rings nor
changes their relative origin.

The same estimate also gives `F(7/4)<9/8+1/5000`. Reconstructing form with
`gamma<1/3069`, `||A^-1||_D<=90` and (ST9) yields
`||P_Dx(7/4)||_D<(90/3069)(20001/20000)(9/8+1/5000)<1/30`.
Thus the phase enclosure has an actual complete-state form allowance; this
is still much larger than the final capture threshold `1/8192` in (ST7).

This encloses the actual correlated image; it does not declare every point
in `K_b` an obtainable source. It supplies no inward sign on the spherical
or cut faces and does not make `K_b` forward invariant. A later transport
region must retain the source subset and justify all relevant exit faces.
These bounds alone give cut retention through `7/4`; the following refinement
extends that interval.

<a id="sine-formed-two-port-semiconvex-transport"></a>
### A sharper source enclosure and longer principal-cut retention

The acute time-one geometry controls subsequent curvature loss without
assuming that acuteness persists. Write `theta_1=theta(1)` for each actual
member. By (C10), (PA2) and `b=1/45000<rho`, every edge of `theta_1` is
strictly acute. Thus its normalized phase Hessian `H_D(theta_1)` is PSD.
For any later lifted phase, put `R=||theta-theta_1||_D`. Each edge normal
has squared dual norm `1/d_i+1/d_j<=1`; cosine is one-Lipschitz. Hence,
in the ordering of selfadjoint operators in the `D` metric,

\[
 H_D(\theta)\ge H_D(\theta_1)-RA\ge-2R I,
 \qquad \|H_D(\theta)\|_D\le2. \tag{ST14}
\]

This is global in the continuous lifts. It permits negative curvature and
uses PSD only at time one. With `F=||f(theta)||_D>0`, `E=||v-f||_D<kF`
and `k=1/20000` from (ST9), differentiation gives
`F'=-<f,H_D v>_D/F<=(2R+2k)F`.
For elapsed slow time `h=sigma-1`, define the upper length
`L(h)=(1+k)*integral_0^h F(1+s) ds`. Then `R<=L`, `L'=(1+k)F`, and

\[
 F(1+h)\le F(1)+\frac{L(h)^2+2kL(h)}{1+k},\qquad
 L'\le a_0+L^2+2kL,\quad
 a_0=(1+k)\left(\frac14+2b\right)<\frac{2501}{10000}.
 \tag{ST15}
\]

The scalar comparison has the explicit rational supersolution

\[
 L(h)<q(h):=\frac{(251/1000)h}{1-h^2/10}
 \quad(0<h\le2),\qquad
 q(3/4)=\frac{753}{3775},\quad q(2)=\frac{251}{300}. \tag{ST16}
\]

To check it, set `A_0=251/1000`, `B_0=1/10`. Multiplying
`q'-(a_0+q^2+2kq)` by the positive denominator `(1-B_0*h^2)^2` gives

\[
 (A_0-a_0)-2kA_0h
 +(A_0B_0+2a_0B_0-A_0^2)h^2
 +2kA_0B_0h^3-a_0B_0^2h^4.
\]

For `0<=h<=2`, its constant/linear part exceeds
`9/10000-251/5000000>0`; its quadratic/quartic part is at least
`h^2*(419/200000)>0` when `h>0`, using `1/4<a_0<2501/10000`.
The cubic term is nonnegative. This establishes the supersolution without
a future field value, new numerical step or renewed reference policy.

At `sigma=7/4`, (ST15)--(ST16) sharpen (ST12) to

\[
 \boxed{\|\theta-c_1^\mu\|_D<\frac15,\qquad
 F<\frac3{10},\qquad \|P_Dx\|_D<\frac9{1000}.} \tag{ST17}
\]

Indeed, `R_1+b+753/3775<1/5`; the field bound follows by replacing
`L` with `1/5` in (ST15), and the form bound uses
`(90/3069)(20001/20000)(3/10)<9/1000`.
These also hold on `[1,7/4]` with the same center. The original means,
affine cycle relations and energy ceiling remain simultaneous constraints.
The resulting smaller seed is an enclosure, not a preparation replacement
or an invariant ball. The final form threshold `1/8192` is still not met
by this bound.

More substantially, for every actual member and `1<=sigma<=3`, (C10) gives

\[
 |g_e(\theta(\sigma))|
 <\frac\pi2-\rho+b+\frac{251}{300}
 <\pi-\frac7{10}\quad\text{on all twenty original edges}. \tag{ST18}
\]

The last inequality follows already from `pi/2>25/16`, even if `rho` is
dropped. Together with earlier prefix retention this proves that the actual
original periods remain `(2,1,0)` **through slow time three**. The new uniform
`7/10` margin is asserted on `[1,3]`; the earlier prefix keeps its own margins.
No later reference, paired-region retention, target proximity or full-law
Hessian positivity is inferred. In particular, the earlier reference
curvature-exit theorem is compatible with (ST14).

### Progress criterion and the remaining geometric obligation

Suppose a subsequent region contains the actual time-`7/4` source image,
and all actual trajectories remain there until a declared target or outgoing
face is reached. Suppose throughout the remaining region `F>=m>0`, with
`U>=U_-` and incoming `U<=U_+`, uniformly for the consumed source members.
Then (ST9) forces first arrival within

\[
 \Delta\sigma\le
 \frac{U_+-U_-}{(19999/20000)m^2}. \tag{ST13}
\]

Otherwise integrating the actual potential derivative would contradict its
lower bound. Source inclusion, exclusion of non-target exits and the positive
field floor are essential premises. The seed set supplies the first kind of
information at one time; it has not supplied the other two. Compactness alone
gives no positive field floor when critical points have not been excluded.
The existing source storage ceiling can bound `U_+`, but cannot select the
region or its destination. This conditional criterion proves no new deadline
for the actual source yet and does not reactivate the consumed transit policy.

<a id="sine-formed-two-port-reflection-geometry"></a>
### Reflection geometry beyond individual acute edges

Reflect each ring by `P:j -> -j mod 9`, exchanging ports 4 and 5.
Let `theta_s^mu` be the nominal midpoint phase, used only to define a symmetry
and shifted to the actual member's conserved mean. On its phase/form leaves set

\[
 T\theta=(I+P)\theta_s^\mu-P\theta,\qquad
 T_x(x)=2\mu_x\mathbf1-Px.
\]

`P` preserves the support and degree metric. The phase offset differs between
nodes only by full turns and a common shift, so `f(T theta)=-P f(theta)`.
These affine involutions preserve the member's means and commute with both
complete A2 rows. They do not replace the actual preparation by a midpoint
source. All inherited closing offsets are retained.

Write the long gaps as `b_1,...,b_8` along
`5->6->7->8->0->1->2->3->4`, and `d_1,...,d_8` along the receiver path.
Let `p,q` be the two donor-to-receiver contacts. The symmetry exchanges
`b_i,b_(9-i)` and `d_i,d_(9-i)`, takes `(p,q)` to `(-q,-p)` and fixes
both short gaps. Define `G` by the original affine cycle leaf, strict
principal cuts and the paired inequalities

\[
 |b_i+b_{9-i}|<\pi,\quad |d_i+d_{9-i}|<\pi\quad(1\le i\le4),
 \qquad |p-q|<\pi. \tag{RG1}
\]

This convex phase region allows one member of a bulk pair to be obtuse.
It is not assumed invariant. For `h=theta-T theta`, the short-edge
differences of `h` vanish. With incidence matrix `B_inc`, define the secant
operator `B_sec=D^-1 B_inc^T diag(c_e) B_inc`. Give short edges weight zero,
and use

\[
 \begin{aligned}
 c_e&=\cos\frac{g_e+g_{\rm mirror(e)}}2\,
       \operatorname{sinc}\frac{g_e-g_{\rm mirror(e)}}2
       &&\text{on a bulk pair},\\
 c_e&=\cos\frac{p-q}2\,\operatorname{sinc}\frac{p+q}2
       &&\text{on either contact},
 \end{aligned} \tag{RG2}
\]

where `sinc(0)=1`. The exact difference identity is
`f(theta)-f(T theta)=-B_sec h`. In `G`, every retained weight belongs to
`(0,1]`: the cosine factors are positive by (RG1), and principal cuts make
the sinc arguments smaller than `pi` in magnitude. Consequently
`0<=B_sec<=A` in the `D` metric. Removing the two short edges leaves a
connected eighteen-edge cycle.

**Critical-set consequence.** If `f(theta)=0` in `G`, its reflected field
also vanishes. Thus `<h,B_sec h>_D=0`; connectedness and zero weighted mean
force `h=0`. Each mirrored pair is equal and acute. Stationary degree-two
balance then makes every donor bulk gap one acute value and every receiver
bulk gap another; the contacts are opposite and acute too. This excludes
all nonacute bulk/contact critical branches without assuming a PSD Hessian.

The complete balance inequalities in
[the original-chamber classification](SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-original-chamber-psd)
exclude donor short gaps `pi/2<=a<=7*pi/10` independently of PSD. If
`a<pi/2`, its receiver-short argument forces that edge acute too.
Therefore **the retained target is the only critical phase in
`G` with `a<=7*pi/10`**. Any other critical point of `G` must have
`a>7*pi/10`; this does not assert that such a point exists.

The ST12 seed already satisfies the short-gap inequality: (C10), the
center radius and the short-edge dual norm below one give
`a<pi/2-1/6+2^-26+9/20<7*pi/10` throughout that enclosure.
But ST12 does not establish all the paired inequalities (RG1) at `7/4`.
Their source admission and retention cannot be inferred from the short gap.

### A full-state reflection budget under the paired geometry

Let `delta_x=x-T_x(x)`, `z=gamma*delta_x`, `y=h+z` and `eta=gamma^2`.
The exact slow-time difference rows give

\[
 y'=-B_{\rm sec}(y-z),\qquad
 z'=-\frac A\eta z-B_{\rm sec}(y-z).
\]

For the fixed-metric quadratic `W=(||y||_D^2+||z||_D^2)/2`, selfadjointness
cancels both mixed terms:

\[
 \boxed{W'=-\langle y,B_{\rm sec}y\rangle_D
             -\langle z,(A/\eta-B_{\rm sec})z\rangle_D\le0.}
 \tag{RG3}
\]

No derivative of the moving secant matrix is used. Since `eta<1`,
`A/eta-B_sec>0` on the relative space; the connected secant support also
makes the first quadratic positive there. Descent is strict unless both
phase and form are reflection-fixed. If all retained secants have a proved
uniform lower bound `beta>0`, a quotient gap `lambda_C>0` for the cycle
without short edges gives
`W'<=-2*min(beta*lambda_C,(1/eta-1)/90)*W`. This contraction is conditional
on continued membership in the declared paired region.

For the original source put `m=pi/9-phi>0`. Its nominal `h` has values
`+m` on the donor and `-m` on the receiver. The A1 residuals imply

\[
 W(0)<W_+:=\frac12\left[
   \bigl(\sqrt{40}m+6(1+\gamma)\epsilon\bigr)^2
      +(6\gamma\epsilon)^2\right]<\frac52. \tag{RG4}
\]

Here `20*m^2<9680/3969<5/2`, with residual correction below `100*epsilon`.
The actual source is initially in `G`. Its admitted corridor
`Omega(1/1024,1/64,1/2)` is also contained in `G`: only edge `5->6`
can be nonacute, and its mirror retains margin `1/64>1/1024`.
Thus the budget applies through that justified corridor continuation,
including the earlier individual acute exit. It has not been extended
through `7/4` by a separate proof of (RG1). Neither strict descent of `W`
nor of `U` makes their joint geometric domain invariant by itself.

<a id="sine-formed-two-port-paired-boundary"></a>
### A source-budget-compatible outgoing paired boundary

A concrete counterexample prevents treating `G` as the required corridor.
Let `r=pi/18`, `h_0=phi-2r`, and use a geometric parameter `t` unrelated
to either physical or slow time. Starting at the nominal original phase,
set

\[
 \begin{aligned}
 a&=(8-2t)r,& b_1=b_8&=(8+t)r,&b_2=\cdots=b_7&=8r,\\
 c=d_i&=4r,& w&=(2-t)r,&p=h_0+w,\quad q&=h_0-w.
 \end{aligned} \tag{RG5}
\]

For `0<=t<1`, this lies in `G`, with the same cycles, safe principal cuts
and `a<7*pi/10`. It is the nodal path
`theta^0+t*r*(e_D4-e_D5)`, whose direction has zero weighted mean and
is odd under `P`. Its reflection difference stays exactly constant.
At `t=1`, only the paired face `M=(b_1+b_8)/2=pi/2` is met.

Direct differentiation of potential along this geometric path gives

\[
 \frac1{2r}\frac{dU}{dt}
 =\sin((8+t)r)-\sin((8-2t)r)-\cos h_0\sin((2-t)r)
 <-\frac{669}{25600},\qquad \frac{dU}{dt}<-\frac1{120}. \tag{RG6}
\]

The bounds used are `sin(pi/3)>173/200`, `cos h_0>15/16` and
`sin(pi/18)>11/64`. They follow from `25/8<pi<22/7`, the elementary
quadratic cosine and cubic sine bounds, and `sqrt(3)/2>173/200`.
At the endpoint, the gradient-field normal to this face is

\[
 \begin{aligned}
 M_f&=\tfrac12\sin(4\pi/9)-\tfrac56
       +\tfrac13\sin(\pi/3)+\tfrac13\cos h_0\sin(\pi/18)\\
 &>\frac{63}{128}-\frac56+\frac{173}{600}+\frac{55}{1024}
   =\frac{23}{25600}>0.
 \end{aligned} \tag{RG7}
\]

The extra bound is `sin(4*pi/9)=cos(pi/18)>63/64`. The face normal has
squared dual norm `5/12`, so even allowing the full ST3 lag leaves
`M_sigma>23/25600-1/7000>0` at this phase.

This is source-connected geometry, not an orbit. For each actual A1 member,
retain its phase residual `u` and its original form and add the same nodal
path direction. The exact face parameter is
`t_*=1-(ell_M^T u)/r`, with `|t_*-1|<12*epsilon`. Its endpoint differs
from the nominal endpoint by less than `8*epsilon` in `D` norm.
The normal remains greater than `23/25600-16*epsilon-1/7000>0` for any
velocity in the admitted lag ball. The potential derivative remains below
`-1/120+9*epsilon<0`: the residual changes it by less than `3*epsilon`,
and the small parameter extension contributes less than `6*epsilon`.
Holding form fixed preserves `W` and makes total storage decrease from
each member's own value. Those held-form geometric states are not claimed
to satisfy the future velocity cone.

A separate complete-state representative satisfies the cone exactly.
At the nominal endpoint `theta_dagger`, matched to the member's mean, choose
`x_dagger=mu_x*1+gamma A^-1 f(theta_dagger)`. Then its actual phase velocity
is `f(theta_dagger)`, so its lag is zero and (RG7) is an outgoing normal of
the full vector field. Global Lipschitz continuity gives
`F_dagger<3/5+2*sqrt(6)*pi/18<3/2`; its form storage obeys

\[
 \tfrac12x_\dagger^TLx_\dagger
 =\tfrac\eta2\langle f,A^{-1}f\rangle_D
 <\frac{405}{4\cdot3069^2}<\frac1{90000}. \tag{RG8}
\]

The potential drop exceeds `1/120`, so its total storage is below every
actual source's initial storage, using the source lower bound
`H_source>=U_nom-40*epsilon`.

For this endpoint, `h` is constant on each ring. Its contact secant is
`c_s=cos(pi/18)*sinc(h_0)>9/10`, so `B_sec h=c_s A h`.
The chosen form therefore has `z=-eta*c_s*h`, and

\[
 W_\dagger=(1-2\eta c_s+2\eta^2c_s^2)\,20h_0^2<20h_0^2. \tag{RG9}
\]

The decrease exceeds `2/3216^2>1/6000000`, using `|h_0|>1/3`,
`eta>1/3216^2`, `c_s>9/10` and `1-eta*c_s>9/10`.
Each actual source has initial `W>20h_0^2-20*epsilon`; thus this complete
outgoing state also satisfies every member's initial reflection budget.

Consequently the broad paired domain, even with the original means, cycles,
safe cuts, the short-gap restriction, source energy/reflection budgets and
zero lag, admits an outward full-state boundary point. The strict budget
margins and normal persist in nearby interior zero-lag states; smooth local
flow from sufficiently close such states crosses this face. Thus this broad
intersection is not forward invariant for all its admitted states. These
nearby states are supplied counterexamples, not actual original-source images.
The geometric paths and the zero-lag representative have different form histories; they are not
combined into one dynamically admissible path. Neither source arrival nor
membership of that representative in the smaller ST12 seed is asserted.
A narrower source-conditioned corridor may exclude this configuration, or
its paired face may be treated as a transition requiring further control.
The missing all-face transport argument is not supplied by symmetry,
critical uniqueness or either decreasing functional alone.

<a id="sine-formed-two-port-mean-shape"></a>
### The remaining paired face and its exact shape balance

The sharper enclosure (ST17) localizes the possible loss of paired geometry.
Put `B_i=(b_i+b_(9-i))/2`, `D_i=(d_i+d_(9-i))/2` for `i=1,...,4`,
and `w=(p-q)/2=(a-c)/2`. Their squared dual norms in the nodal `D` metric are
`5/12,1/2,1/2,1/4` along either bulk path, `1/3` for `w`, and `2/3`
for either short gap. Combining these with (C10), `R_1<2^-26` and (ST17)
gives throughout `[1,7/4]`

\[
 \begin{aligned}
 |a|&<\pi/2-1/320,& |c|&<\pi/2-1/2,\\
 |B_i|&<\pi/2-1/40\quad(i=2,3,4),&
 |D_i|,|w|&<\pi/2-1/2,\\
 \pi/4&<B_1<\pi/2+1/20. &&
 \end{aligned} \tag{SG1}
\]

For example, use `sqrt(2/3)<49/60`, `sqrt(1/2)<99/140` and
`sqrt(5/12)<13/20`. The lower bound for `B_1` follows from the exact cycle relation
`a+2*sum B_i=4*pi` and the displayed upper bounds on the other terms.
All bounds retain the center radius; none evaluates the future state.
The earlier prefix lies in `G`. Consequently, if an actual member first
leaves `G` by `7/4`, it does so through **`B_1=pi/2`**. No such departure
or arrival at the displayed outgoing witness is asserted.

The exact phase-field normal on that face shows what still needs control.
Let `delta_i=(b_i-b_(9-i))/2`, `h_c=(p+q)/2`, and define the paired currents
`J_i=sin(B_i)*cos(delta_i)` and `J_c=cos(h_c)*sin(w)`. Averaging the actual
degree-two and degree-three sine rows gives

\[
 \begin{aligned}
 (B_1)_f&=J_2/2-5J_1/6+(\sin a+J_c)/3,\\
 (B_i)_f&=(J_{i-1}-2J_i+J_{i+1})/2\quad(i=2,3),\\
 (B_4)_f&=(J_3-J_4)/2,\qquad
 a_f=2(J_1-\sin a-J_c)/3. \tag{SG2}
 \end{aligned}
\]

The receiver rows replace `B,J,a` by `D,K,c` and `J_c` by `-J_c`,
where `K_i=sin(D_i)*cos((d_i-d_(9-i))/2)`.
They preserve `a+2*sum B_i=4*pi` and `c+2*sum D_i=2*pi`.
For the actual phase velocity, each row has the additive observation of
`e_v=v-f`, bounded by its stated dual norm times
`min(1/7000,F/20000)`. Thus on the remaining face the inward comparison is

\[
 \frac{J_2}{2}-\frac56\cos\delta_1
       +\frac{\sin a+\cos h_c\sin w}{3}
 +\sqrt{5/12}\min\{1/7000,F/20000\}<0. \tag{SG3}
\]

It is a sufficient sign test, not an admitted sign for the actual source.
The full-state reflection budget controls a different combination of variables;
by (RG5)--(RG9) it cannot supply this signed current comparison alone.
These exact observations discard no complete-law row and define no autonomous
mean-shape closure: the paired differences and their retained dynamics still
enter the currents.

<a id="sine-formed-two-port-short-energy-barrier"></a>
### A source-energy barrier removes the separate short-gap exit

There is a complementary restriction valid for as long as paired geometry is
retained, with no finite-time deadline. On the closure of `G`, reflection
averaging and cosine concavity give

\[
 U\ge D(a)+R(c)+2[1-\cos((a-c)/2)], \tag{SG4}
\]
\[
 D(a)=1-\cos a+8[1-\cos((4\pi-a)/8)],\qquad
 R(c)=1-\cos c+8[1-\cos((2\pi-c)/8)].
\]

Indeed, each paired contribution is `2-2*cos(B_i)*cos(delta_i)` and is
at least `2-2*cos(B_i)` because `|B_i|<=pi/2`. Jensen then averages the
four paired means subject to the exact cycle sum. The contact term uses
`cos p+cos q=2*cos(w)*cos(h_c)` and `|w|<=pi/2`.
This is a bound for arbitrary admissible phases, not a stationary ansatz.

At the donor-short face `a=2*pi/3`, the other constraints give
`-pi/3<=c<=pi`. The donor part satisfies
`D=19/2-2*(sqrt(6)-sqrt(2))>297/40`, using
`sqrt(6)<49/20` and `sqrt(2)>113/80`. Write
`F_c(c)=R(c)+2[1-cos((2*pi/3-c)/2)]` (distinct from the field norm `F`).
The following bounds cover its entire allowed interval:

- For `-pi/3<=c<=0`, `F_c>=9-4*sqrt(2)>3`.
- For `0<=c<=pi/2`, put `d=(2*pi-c)/8`, `chi=(2*pi/3-c)/2`.
  Then `F_c''=cos c+cos d/8+cos chi/2>27/80`. At `c_0=5*pi/12`,
  `d_0=19*pi/96`, `chi_0=pi/8`, elementary bounds give
  `F_c(c_0)>119/50` and `|F_c'(c_0)|<1/50`. Completing the quadratic
  lower tangent gives `F_c>119/50-2/3375>19/8` on this interval.
- For `pi/2<=c<=2*pi/3`,
  `F_c'=sin c-sin d-sin chi>43/50-3/5-13/50=0`, while
  `F_c(pi/2)>1+8(1-5/6)+2(1-31/32)=115/48>19/8`.
- For `2*pi/3<=c<=pi`, the receiver term has its sole stationary point
  at `6*pi/7`, a maximum. Its endpoint values exceed `5/2`, using
  `sqrt(3)<7/4` and `cos(pi/8)<37/40`; the contact term is nonnegative.

For the second interval, the needed explicit bounds are
`cos c_0<13/50`, `cos chi_0<37/40`, `621/1000<d_0<5/8`,
`4*(621/1000)^2-(5/8)^4/3>149/100`,
`24/25<sin c_0<97/100`, `29/50<sin d_0<59/100`, and
`19/50<sin chi_0<39/100`. They follow from the exact radicals at
`pi/12,pi/8`, `157/50<pi<22/7` and the alternating cubic/quintic sine
and quadratic/quartic cosine bounds on `[0,1]`.
For the third interval, `47/80<3*pi/16<3/5` gives
`cos(3*pi/16)<1-(47/80)^2/2+(3/5)^4/24<5/6`;
`cos(pi/12)=(sqrt(6)+sqrt(2))/4<31/32`.
Thus the entire face satisfies

\[
 \boxed{a=2\pi/3,\ \theta\in\overline G
        \quad\Longrightarrow\quad U>297/40+19/8=49/5.} \tag{SG5}
\]

Every actual source has smaller total storage. From (M4),
`H(0)<U_floor+1/4`. The original constant obeys `U_floor<191/20`:
with `x=44/63`,

\[
 \cos(4\pi/9)+\cos(2\pi/9)
 >\frac{157}{900}-\frac{(11/63)^3}{6}
     +1-\frac{x^2}{2}+\frac{x^4}{24}-\frac{x^6}{720}
 >\frac{169}{180}. \tag{SG6}
\]

The final rational margin is `14933903819/28135575994050>0`.
This uses `U_floor` only to bound the source storage, not as an assumed
later global potential floor. The complete held law therefore gives
`U<=H<=H(0)<49/5` for every member at all later times.

By continuity, an actual segment starting below `a=2*pi/3` and remaining
in the closure of `G` cannot reach that short-gap face. This applies from
the original source, and from any later admitted paired segment below the
barrier; (SG1) supplies the short-gap condition through `7/4`, but does not
supply paired membership there. A departure and later re-entry into `G`
requires its own admission. In the strict paired region on this side of
the barrier, the earlier critical-set result allows only the retained
target as a critical phase. It does not establish that this reached portion
contains a critical phase, exclude a limiting paired boundary, or prove capture.

This removes a separate short-gap growth obligation under paired retention.
The existing outgoing paired-face witness has `a=pi/3` and survives the
new barrier. Its source accessibility, and the sign or controlled passage
of the actual paired face, remain unresolved.

<a id="sine-formed-two-port-paired-face-progress"></a>
### Actual paired retention and a directed redistribution at its remaining face

The curvature-dependent length bound supplies a further actual admission.
Let `psi_1^mu` be the same exact reference at time one, matched only to the
member's conserved common phase mean. From (PA2) and (ST16),
`||theta(1+h)-psi_1^mu||_D<b+q(h)`. At `h=1/2`, `q=251/1950`.
The reference first-pair clearance is greater than `(rho+1/6)/2`, by (C10),
and its dual norm is below `31/48`. Even the conservative radius
`r_half=2^-25+b+251/1950`, which includes both center-radius allowances,
gives

\[
 B_1(\theta)<\frac\pi2-
 \left[\frac{\rho+1/6}{2}-\frac{31}{48}r_{\rm half}\right]
 <\frac\pi2-\frac1{2400}\quad(1\le\sigma\le3/2). \tag{SG7}
\]

All other paired faces are already excluded by (SG1), and the earlier prefix
lies in `G`. Thus **every original member stays in `G` through `3/2`**.
The full-state reflection budget (RG3)--(RG4) consequently applies on
`[0,3/2]`. Any first paired departure by `7/4` must occur in `(3/2,7/4]`
on the upper first donor paired face. This does not assert a departure.

#### A source correlation relevant to the normal sign

At a possible first encounter with `B_1=pi/2` in this window, the direct
reference distance is below `1/5`, and
`Delta B_1=B_1(theta)-B_1(psi_1^mu)>1/12`.
For the contact half-difference `w=(a-c)/2`, (C10) gives
`w(psi_1^mu)>91/96-pi/4>109/672`. The covectors obey

\[
 \|\ell_w-\ell_{B_1}/4\|_{D^{-1}}^2=85/192<4/9.
\]

Consequently `Delta w>Delta B_1/4-2/15>-9/80`, and

\[
 \boxed{w>109/672-9/80=167/3360>1/21.} \tag{SG8}
\]

This uses a joint phase bound; independent marginal intervals lose the
covariance. On this face `B_2>pi/4`, by the cycle relation and (SG1).
Hence the exact phase-field normal from (SG2) can be written as

\[
 \begin{aligned}
 N_f={}&N_{\rm sym}+\frac56(1-\cos\delta_1)
       -\frac{\sin B_2}{2}(1-\cos\delta_2)
       -\frac{\sin w}{3}(1-\cos h_c),\\
 N_{\rm sym}={}&\frac{\sin B_2}{2}-\frac56
                         +\frac{\sin a+\sin w}{3}. \tag{SG9}
 \end{aligned}
\]

The active paired difference increases this normal; the other two displayed
difference terms are nonpositive. Their sizes and the mean contribution are
not fixed by these signs. The actual normal retains the (ST10) lag allowance.
The active pair's potential equals exactly two on this face, independently
of `delta_1`; that contribution cannot measure the outward correction.
This does not exclude another energy/geometry argument using further source
correlations. No normal sign or crossing has been admitted.

#### Directed internal progress at the localized boundary

More can be proved than the signs of individual normal contributions.
Consider the closed radius-`1/5` seed about `c_1^mu`, with the original
principal gaps, means and cycles, intersected with `B_1=pi/2` and the
remaining (SG1) margins. Call this face set `S_mu`. It is an outer set;
neither its nonemptiness nor its actual accessibility is asserted.
Every possible actual first encounter by `7/4` belongs to it.

For this whole set, the exact-reference distance is at most
`r_0=1/5+2^-26`. The two phase functionals

\[
 R_+=a+6b_1-3\pi,\qquad R_-=3\pi-6b_1+a
\]

have squared dual norms `80/3` and `104/3`. At the reference, (C10) gives
`R_+>845/768`, `R_->59/48+3/1024`. Using
`sqrt(80/3)<31/6` and `sqrt(104/3)<59/10`, throughout the face set,

\[
 R_+>845/768-(31/6)r_0>1/20,\qquad
 R_->59/48+3/1024-(59/10)r_0>1/20. \tag{SG10}
\]

Both are signed separations; the center radius is retained. The margins over
`1/20` are respectively `6815713/402653184` and `21101707/10066329600`.
Principal cuts and `b_1+b_8=pi` allow
`zeta=arcsin(sin b_1) in [0,pi/2]`. On the acute branch,
`a+6*zeta-3*pi=R_+`; on the obtuse branch it equals `R_-`.
The exact donor cycle on the face is `a+2(B_2+B_3+B_4)=3*pi`.
Thus, with `B_bar=(B_2+B_3+B_4)/3`,

\[
 \zeta-\overline B>1/120,\qquad
 \pi/4<B_i<\pi/2-1/40\quad(i=2,3,4). \tag{SG11}
\]

The lower bounds follow because their sum exceeds `5*pi/4` while each of
the other two means is below `pi/2`. The sine secant from `B_bar` to `zeta`
is at least `2*sin(1/80)/pi>1/128`: use the half-angle identity, the sine
chord bound, `sin(1/80)>1/81` and `pi<22/7`. Since
`J_i=sin(B_i)*cos(delta_i)<=sin(B_i)`, concavity on `[0,pi/2]` now gives

\[
 I_1-\frac{J_2+J_3+J_4}{3}>\frac1{15360},
 \qquad I_1:=\sin b_1=\sin b_8. \tag{SG12}
\]

This is an interior-current imbalance, not a normal-velocity identity.
It has a fixed linear observation. Give `ell_raw` the following coefficients
on the seven donor interior nodes in path order, and zero elsewhere:

\[
 \begin{array}{c|rrrrrrr}
 i&6&7&8&0&1&2&3\\ \hline
 (\ell_{\rm raw})_i&-2&-5/3&-4/3&-1&-2/3&-1/3&0
 \end{array}
\]

Degree-two current balance telescopes to
`ell_raw^T f=I_1-(J_2+J_3+J_4)/3`. Its coefficient sum is `-7` and
its squared dual norm is `91/18`. Set
`ell=ell_raw+(7/40)D*1` and define the selected observation
`Q(theta)=ell^T theta`. It is unchanged by a common phase shift because
`ell^T 1=0`; it uses the retained lifts, not independently wrapped phases.
Weighted mean conservation makes the added term vanish on both `f` and
the actual velocity `v`. Moreover,

\[
 \|\ell\|_{D^{-1}}^2=91/18-49/40=1379/360<4.
\]

Therefore (SG12) gives `F>1/30720` on `S_mu`, excluding every critical
phase there without root enumeration. For an **actual** source encounter
in the controlled window, use also `F<3/10` from (ST17) and `E<F/20000`:

\[
 \boxed{Q_\sigma=\ell^T v>
     \frac1{15360}-\frac3{100000}>\frac1{30000}.} \tag{SG13}
\]

Both complete rows are retained through the actual lag estimate. `Q` is a
chosen linear phase observation for this proof, not an additional state
coordinate, constitutive law or autonomous reduced dynamics.

The directed progress is robust near the face. The scalar field
`ell^T f` is globally Lipschitz with constant below four. Consequently,
for an actual member during `[1,7/4]`, if its `D`-distance to `S_mu` is
at most `1/240000`, the same field cap and lag bound imply

\[
 Q_\sigma>\frac1{15360}-\frac1{60000}-\frac3{100000}
           >\frac1{60000}. \tag{SG14}
\]

This neighborhood statement is conditional on proximity; it does not assert
that an actual member enters it. It remains valid on either side of the
paired face because it uses global field continuity, not secant positivity.
It supplies a direction of internal redistribution at a possible encounter,
not the sign of `B_1`'s normal velocity. A crossing, tangency or later
recrossing still needs its own source argument. Nor is the seed invariant:
the result does not exclude a different later critical destination or prove
capture. The missing discriminator is now the source-conditioned balance
in (SG9), including its symmetric mean term and actual lag, while preserving
the other boundaries after the proved intervals.

<a id="sine-formed-two-port-local-compression"></a>
### Adjacent current ordering and short-gap contraction at a first encounter

The exact radius in (ST16), rather than its rounded upper bound `1/5`,
retains a useful source correlation. Set
`R_*=2^-26+b+753/3775`, with `b=1/45000`, and intersect the closed
`D`-ball of radius `R_*` about `c_1^mu` on the original mean/lift/cycle
leaf, retaining principal gaps, the first paired face and remaining (SG1)
margins. Call this outer face set `T_mu`. Every possible
actual first encounter by `7/4` belongs to it; its nonemptiness and
accessibility are not asserted. Its exact-reference distance is below
`r=2^-25+b+753/3775`, retaining both center-radius allowances.

Write `M=B_1`, `g=rho+1/6`, so `Delta M>g/2` on the face by (C10).
For `S=b_1+B_2` and `D_*=b_1-B_2`, the inverse-degree covectors satisfy

\[
 \|\ell_S+\tfrac34\ell_M\|_*^2=253/192<(287/250)^2,\qquad
 \|\ell_{D_*}-2\ell_M\|_*^2=5/6<1,
 \quad \|\ell_{B_2}+\ell_M\|_*^2=5/12<(13/20)^2.
\]

Here `*` denotes the `D^-1` norm. At the reference,
`S<pi-g`, `D_*>1/6-11/512`, and `B_2<pi/2-1/6`. Therefore throughout
`T_mu`,

\[
 \begin{aligned}
 \pi-b_1-B_2&>11g/8-(287/250)r>1/1250,\\
 b_1-B_2&>1/6-11/512+g-r>1/10,\\
 B_2&<\pi/2-1/6+(13/20)r-g/2<\pi/2-3/25. \tag{SG15}
 \end{aligned}
\]

The positive margins in these three final rational comparisons are
`147631401727/7125073920000000`, `366080281121/28500295680000` and
`327210566573/570005913600000`, respectively. They use nodal correlations,
not independently chosen gap intervals.

On the face, put `zeta=arcsin(sin b_1)` as before. The acute branch uses
`b_1-B_2`; the obtuse branch uses `pi-b_1-B_2`. Both give
`zeta-B_2>1/1250`. With `pi/4<B_2<pi/2-3/25`, the sine secant is at
least `2*sin(3/50)/pi>3/80`, using the cubic sine bound and `pi<22/7`.
Since `J_2<=sin B_2`, the individual adjacent-current deficit obeys

\[
 d_2:=I_1-J_2>3/100000,\qquad I_1=\sin b_1=\sin b_8. \tag{SG16}
\]

This is stronger local information than the earlier averaged deficit.
The linear observation `Y=2M+a=b_1+b_8+a=theta_6-theta_3` has dual
norm one and preserves the fixed lift and common-phase symmetry.
From (SG2), everywhere `Y_f=J_2-J_1`; on the face `J_1=I_1`.
The actual lag is below `3/200000` during `[1,7/4]`. Hence

\[
 Y_\sigma=2M_\sigma+a_\sigma<-3/200000. \tag{SG17}
\]

At a first encounter from inside `G`, differentiability gives
`M_sigma>=0`, including a tangency. Thus every such actual encounter in
`(3/2,7/4]` necessarily satisfies

\[
 \boxed{a_\sigma<-3/200000-2M_\sigma\le-3/200000,\qquad
 2M_\sigma=-d_2-a_\sigma+e_Y,\quad |e_Y|\le E.} \tag{SG18}
\]

The short **phase gap** contracts. This is not a spatial compression or a
registered operator invocation. An outward crossing requires its decrease
to exceed the adjacent-current deficit after including the actual lag.
The result establishes the deficit and necessary contraction, not that
final competition. A source lower bound `a_sigma>=-3/200000` on the face
would exclude a first encounter, but is not available.

The scalar field `Y_f` is globally two-Lipschitz. Thus an actual member
in `[1,7/4]` within `D`-distance `1/400000` of `T_mu` satisfies
`Y_sigma<-3/100000+2/400000+3/200000=-1/100000`.
This conditional neighborhood bound holds on either side of the face;
it neither admits proximity nor proves a normal sign or a later destination.

<a id="sine-formed-two-port-reflection-forcing"></a>
### Source-directed growth of a local reflection difference

The same possible encounter has a second, complementary mechanism. Keep an
actual source segment `[0,T]`, `T<=7/4`, which stays in the closure of `G`.
In particular this includes the segment to a first paired encounter.
A departure and later re-entry require a new admission.

First bound cumulative lag. On the initial layer `s_0=1/128`, (ST8) and
`E(0)<1` give
`E(s)<exp(-99998s)+(3/2)/99998`. On the admitted prefix the reference
field norm is below `3/5`, by its PSD speed comparison; (M7) and global
field Lipschitz continuity then give actual `F<3/5+18/25000<601/1000`.
Use (ST9) after `s_0` and (ST17) after one to obtain

\[
 \int_0^{7/4}E(s)\,ds
 <\frac{259}{256\cdot99998}
   +\frac1{20000}\frac{601}{1000}\frac{127}{128}
   +\frac9{800000}<\frac1{19000}. \tag{SG19}
\]

The last rational margin is `3523787413/2431951360000000`.
This consumes only the retained prefix and actual error bounds, not a
later evaluated reference or an independently chosen phase perturbation.

For `h=theta-T theta`, the exact complete law gives
`h_sigma=-B_sec(theta)h+(I+P)e_v`. Along the declared paired segment,
`B_sec` is selfadjoint PSD, its retained edge weights lie in `[0,1]`, and
it commutes with reflection. Zero weights at the paired face are allowed.
Define an auxiliary **linear comparison** by

\[
 \bar h_\sigma=-B_{\rm sec}(\theta(\sigma))\bar h,\qquad
 \bar h(0)=\begin{cases}+m&\text{donor},\\-m&\text{receiver},\end{cases}
 \quad m=\pi/9-1/1000>1/3.
\]

Its coefficient path is the actual, generally unknown path. No auxiliary
trajectory is evaluated and no actual source is replaced. The nominal
comparison has the same zero weighted mean as `h`; the A1 residuals give
`||h(0)-hbar(0)||_D<=6*epsilon`, `epsilon=10^-32`.
Contractivity, without a derivative of `B_sec`, yields

\[
 \|h-\bar h\|_D\le6\epsilon+2\int_0^\sigma E(s)\,ds. \tag{SG20}
\]

Each observed half-edge difference has error at most
`3*epsilon+integral E<1/18000`. This controls deviations from a comparison
sign; it does not assert an exact sign cone for the actual state.

For completeness, reflection fixes `hbar` under `P`. On its folded path
use donor half-edge differences `z_Di`, negatives of receiver half-edge
differences `z_Ri`, and negative contact half-difference `z_c`, `i=1,...,4`.
Write the actual secant weights as
`c_Di=cos(B_i)*sinc(delta_i)`,
`c_Ri=cos(D_i)*sinc(epsilon_i)`, and `c_c=cos(w)*sinc(h_c)`.
Direct degree-weighted divergence gives, for `alpha=D,R`,

\[
 \begin{aligned}
 z_{\alpha1}'&=-\tfrac56c_{\alpha1}z_{\alpha1}
                +\tfrac12c_{\alpha2}z_{\alpha2}+\tfrac13c_cz_c,\\
 z_{\alpha i}'&=\tfrac12c_{\alpha,i-1}z_{\alpha,i-1}
                -c_{\alpha i}z_{\alpha i}
                +\tfrac12c_{\alpha,i+1}z_{\alpha,i+1}\quad(i=2,3),\\
 z_{\alpha4}'&=\tfrac12c_{\alpha3}z_{\alpha3}
                -\tfrac32c_{\alpha4}z_{\alpha4},\\
 z_c'&=(c_{D1}z_{D1}+c_{R1}z_{R1}-2c_cz_c)/3.
 \end{aligned}
\]

All off-diagonal coefficients are nonnegative. The initial eight bulk
differences vanish and `z_c(0)=m`, so positivity of this nonautonomous
linear system gives `z>=0`. In particular
`z_c>=m*exp(-2*sigma/3)>4/39` through `7/4`, using `exp(7/6)<13/4`.
For an elementary rational bound, the exponential series gives
`exp(1)<87/32` and `exp(1/6)<7/6+1/68`; their product is below `13/4`.
Returning to the actual state via (SG20) proves

\[
 h_c<-4/39+1/18000<-1/10,\qquad
 \delta_i>-1/18000,\qquad \epsilon_i<1/18000. \tag{SG21}
\]

These approximate signs apply only along the admitted paired segment.

At actual time one, (C10) and (PA2) give
`delta_1(1)>1/12-11/1024-(2/3)b>1/14`.
On `[1,T]`, (SG1) bounds `|w|<pi/2-1/2`. Also
`|h_c|<pi/2-1/2`: both contact gaps at the reference have margin `2/3`,
and their average has squared dual norm `1/3`, with actual reference
distance below `1/5`. Thus
`cos w>sin(1/2)>23/48` and `-sin h_c>sin(1/10)>599/6000`.

Define the reflection-difference currents
`j_i^-=cos(B_i)*sin(delta_i)`, `j_c^-=cos(w)*sin(h_c)`;
these differ from `I_1=sin b_1` and the paired mean currents `J_i`.
The actual first-difference row is

\[
 (\delta_1)_\sigma=\tfrac12j_2^- -\tfrac56j_1^-
                         -\tfrac13j_c^-+e_\delta.
\]

Here `|e_delta|<1/100000`, from dual norm below `2/3` and
`E<3/200000`. Since `B_2` is acute and the gaps retain their principal
branches, (SG21) gives `j_2^->=-1/18000`. The terms other than the
`j_1^-` damping therefore exceed

\[
 \beta=-1/36000+(23/48)(599/6000)/3-1/100000>1/64.
\]

A first zero of `delta_1` would have positive derivative, so the positive
value at one cannot cross zero. While positive in `G`,
`j_1^-<=delta_1`, hence `delta_1'>-(5/6)delta_1+1/64`.
Since `T-1<=3/4` and `exp(5/8)<2`, it remains above `1/28`.
The last exponential bound follows by summing the quadratic-and-higher
tail: `exp(5/8)<1+5/8+[(5/8)^2/2]/[1-5/24]<2`.
At a first paired encounter `B_1=pi/2`, its damping coefficient vanishes,
and the complete actual state satisfies

\[
 \boxed{a_\sigma<-3/200000,\qquad
         \delta_1>1/28,\qquad (\delta_1)_\sigma>1/64.} \tag{SG22}
\]

The first inequality is (SG18); the other two retain the original relative
origin through the comparison step. Local reflection difference can therefore
grow even while the global full-state budget `W` has been nonincreasing up
to the boundary. This is a statement about the held model, not a universal
monotonicity rule for TNFR observations.

The first-pair geometry is necessarily unequal:
`b_1>pi/2+1/28`, `b_8<pi/2-1/28`. Since the `delta_1` covector has
dual norm below `2/3`, this actual encounter is at `D`-distance greater
than `3/56` from any phase with `delta_1=0` in the retained chart, even
after a common phase shift. In particular, the exact mirror-equal outgoing
witness (RG5) cannot be this first encounter. Neither can the reflection-fixed
target: this conditional encounter itself cannot satisfy the `1/1024`
phase-handoff radius. These exclusions concern the encounter, not a later
return or capture.

It also has negative phase curvature. Let `mathcal H` denote the signed-cosine
phase Hessian, not the combinatorial Laplacian. Define a relative test vector
`v=1_{ {6,7,8,0,1,2,3} }-(14/40)*1` and `u=e_4-e_5`.
Both have zero weighted mean, with
`||v||_D^2=91/10`, `||u||_D^2=6` and `<v,u>_D=0`.
Only the first paired edges cross the support boundary of the indicator.
At `B_1=pi/2`, direct edge quadratics give

\[
 v^T\mathcal H v=0,\qquad
 v^T\mathcal H u=-2\sin\delta_1,\qquad
 u^T\mathcal H u=4\cos a+\cos p+\cos q\le6.
\]

Take `t=sin(delta_1)/3`. The numerator of the `D`-Rayleigh quotient of
`v+t*u` is at most `-(2/3)sin(delta_1)^2`, and its denominator is at most
`293/30`. From (SG15), `delta_1<pi/4`; (SG22) and
`sin(1/28)>1/29` therefore imply, for the normalized relative Hessian,

\[
 \boxed{\lambda_{\min}(H_D)<-60/739239<-1/12500.} \tag{SG23}
\]

This is instantaneous negative curvature of the **actual** phase geometry
conditional on its first encounter. It neither asserts that the encounter
occurs nor converts the nonstationary full flow into an unstable equilibrium.
It is distinct from the earlier unconditional reference-curvature exit.

These effects contribute in opposite directions to the normal decomposition
(SG9). On the face, (SG15) and `B_2>pi/4` imply `0<delta_1<pi/4`.
The active correction `A_delta=(5/6)(1-cos delta_1)` consequently has
`(A_delta)_sigma>5/11136`, using `sin(1/28)>1/29` and (SG22).
In contrast, the derivative of the short-gap term `sin a/3` is negative,
because `a` is acute and `a_sigma<0`. These are derivatives of the displayed
scalar contributions at the encounter, not the derivative of the full normal.
These contribution signs alone do not determine normal acceleration. The
following complete-law calculation controls the lag derivative while retaining
the remaining signed current-rate obligation. Neither the local effects nor
the derivative estimate makes an outer face set or seed invariant.

<a id="sine-formed-two-port-rate-transport"></a>
### Source-conditioned control of acceleration and normal-rate error

Keep the original source, both complete rows, held degrees and slow clock.
The [acceleration comparison](SINE_FORM_PHASE_REDUCTION.md#sine-acceleration-tracking)
controls a derivative that a velocity-error bound alone leaves open. Put
`J=Df(theta)` and `r_v=(e_v)_sigma=v_sigma-Jv`. Its exact equation is
`(r_v)_sigma=-(A/eta+J)r_v-D^2f[v,v]-J^2v`; no commutation of `A` and
`J` is used.

For the actual unit-weight graph, let `B` be the oriented incidence matrix.
The field is `f=-D^-1 B^T sin(g)`, with the retained affine closing offsets.
Its second derivative is `D^-1 B^T diag(sin(g))*(Bu odot Bw)` on vectors
`u,w`. The incidence and adjoint norms are at most `sqrt(2)`. Every edge
has squared degree-dual norm at most one because all degrees are at least
two. Hence `||D^2f[u,w]||_D<=2||u||_D||w||_D` globally. Together with
`||J||_D<=2` and `lambda/eta>100000`, this gives

\[
 D^+\|r_v\|_D\le-99998\|r_v\|_D+2\|v\|_D^2+4\|v\|_D. \tag{SG24}
\]

The source and (ST2) give `E<1`, `F<7`, `||v||_D<8` globally, and
`||r_v(0)||_D<2*3216^2+16<2^25`. Thus
`||r_v(sigma)||_D<2^25*exp(-99998*sigma)+160/99998`.
For `sigma>=1/128`, the initial term is below `2^-487<2^-20`, and
`2^-20+160/99998<1/600`. This retains the initially large acceleration
mismatch instead of resetting the source to a slow velocity graph.

On `[1,7/4]`, (ST17) and (ST9) give
`||v||_D<(20001/20000)*(3/10)<301/1000`, so the forcing in (SG24) is
below `7/5`. Apply the scalar comparison again, from actual time one.
For `h>=1/128`,
`exp(-99998*h)<2^-512<2^-20`, and
`1/(600*2^20)+7/(5*99998)<1/70000`. Consequently

\[
 \boxed{\|(e_v)_\sigma\|_D<1/70000,\qquad
 \|\theta_{\sigma\sigma}-Df(\theta)f(\theta)\|_D<1/22000,
 \quad129/128\le\sigma\le7/4.} \tag{SG25}
\]

The second bound uses `v_sigma-Jf=r_v+Je_v` and
`1/70000+3/100000<1/22000`. It evaluates no future phase, field or sign.
The comparison field acceleration is evaluated symbolically at the actual
phase; it is not the acceleration of the separately evolved reference.

For the first-pair normal, write `M_sigma=N_f+e_M`, where
`N_f=ell_M^T f` and `e_M=ell_M^T e_v`. The fixed mean-zero covector has
squared dual norm `5/12<(2/3)^2`. Therefore throughout the same interval,

\[
 \boxed{|(e_M)_\sigma|<1/105000,\qquad
 |M_{\sigma\sigma}-\ell_M^T Df(\theta)f(\theta)|<1/33000.} \tag{SG26}
\]

In particular this covers every possible first paired encounter by `7/4`.
A signed bound on the remaining field expression is still required; small
lag and small lag derivative do not determine its sign.

A coarser version remains valid through the already admitted time three.
From (ST15)--(ST16), with `q_2=251/300` and `k=1/20000`,
`(1+k)F<2501/10000+q_2^2+2*k*q_2<1` on `[1,3]`. This proves both
`F<1` and `||v||_D<1` independently of paired membership. The forcing in
(SG24) is below six. The same comparison from time one gives

\[
 \boxed{\|(e_v)_\sigma\|_D<1/16000,\qquad
 \|\theta_{\sigma\sigma}-Df(\theta)f(\theta)\|_D<1/6000,
 \quad129/128\le\sigma\le3.} \tag{SG27}
\]

Here `1/(600*2^20)+6/99998<1/16000` and
`1/16000+1/10000<1/6000`; the first-pair acceleration error is below
`1/9000`. No post-exit use of a paired-region inequality is involved.
These bounds control derivative errors without admitting a crossing,
recrossing, target handoff or new numerical continuation.

<a id="sine-formed-two-port-first-tangency"></a>
### A necessary current-rate deficit for first tangency

At a possible first paired encounter `T<=7/4`, differentiate the full
normal identity (SG2) before imposing `M=B_1=pi/2`. With
`J_1=sin(M)*cos(delta_1)`, its actual derivative on that face is
`(J_1)_sigma=-sin(delta_1)*(delta_1)_sigma`. Define the retained rate

\[
 \mathcal R=\tfrac12(J_2)_\sigma
       +\tfrac13\cos(a)a_\sigma+\tfrac13(J_c)_\sigma.
\]

Both contact contributions remain in
`(J_c)_sigma=cos(h_c)*cos(w)*w_sigma-sin(h_c)*sin(w)*(h_c)_sigma`.
The short-gap term is negative by (SG22); the other current-rate signs
are not established. The exact complete-law acceleration and its bound are

\[
 \begin{aligned}
 M_{\sigma\sigma}
 &=\mathcal R+\tfrac56\sin(\delta_1)(\delta_1)_\sigma+(e_M)_\sigma\\
 &>\mathcal R+\frac5{11136}-\frac1{105000}
  >\mathcal R+\frac1{2300}. \tag{SG28}
 \end{aligned}
\]

This uses (SG22), `sin(delta_1)>1/29` and the separate derivative bound
(SG26). A first tangency has `M_sigma(T)=0` and necessarily
`M_sigma_sigma(T)<=0`: a positive second derivative would give
`M(T-h)>pi/2` for sufficiently small positive `h`, contradicting first
arrival from inside. Therefore every such tangency requires
`mathcal R(T)<-1/2300`. A source proof of the opposite non-strict bound
would force positive normal velocity at any first encounter covered by it;
it would not establish that an encounter occurs.

The remaining combination can be expressed using the actual phase field
with a quantified full-state allowance. Put
`Phi(theta)=J_2/2+(sin(a)+J_c)/3` and
`mathcal R_f=DPhi(theta)[f(theta)]`. These are selected scalar expressions,
not additional state coordinates or a new dynamics. In the degree-dual
metric, `B_2` and `delta_2` have squared norm `1/2` and are orthogonal;
`w,h_c` have squared norm `1/3` and are orthogonal; `a` has squared norm
`2/3`. Since `cos(u)^2*cos(v)^2+sin(u)^2*sin(v)^2<=1`,

\[
 \|D\Phi\|_*
 \le\frac1{2\sqrt2}+\frac{\sqrt{2/3}}3+\frac1{3\sqrt3}
 <\frac{99}{280}+\frac{49}{180}+\frac7{36}<\frac56.
\]

Thus `|mathcal R-mathcal R_f|=|DPhi[e_v]|<1/80000`, and

\[
 \boxed{M_{\sigma\sigma}>\mathcal R_f+\frac1{2400},\qquad
 M_\sigma(T)=0\ \Longrightarrow\ \mathcal R_f(T)<-\frac1{2400}.}
 \tag{SG29}
\]

The strict rational comparison is
`5/11136-1/105000-1/80000>1/2400`. The two necessary tangency thresholds
are not equivalent sign tests. Neither value of `mathcal R_f` nor its sign
at a future source phase has been evaluated. In particular a proof that
`mathcal R_f>=-1/2400` would exclude first tangency, not prove first arrival,
retention or final capture; failure of that sufficient test proves none of
the opposite conclusions.

A missing contact correlation cannot be supplied just by reflection signs.
Let `k_1^-=cos(D_1)*sin(epsilon_1)` be the first receiver difference
current. Direct degree-three port balance gives
`(h_c)_f=(k_1^- -j_1^- -2*j_c^-)/3`. At the donor face, `j_1^-=0`.
Although `j_c^-<0`, (SG21) bounds `k_1^-` above, whereas a positive
`(h_c)_sigma` would need a lower bound on `k_1^- -2*j_c^-` with its lag.
Positivity alone does not supply that current ordering. The initialized
transport argument below supplies the missing comparison while retaining
the distinct normal-velocity and first-tangency obligations.

<a id="sine-formed-two-port-initialized-contact"></a>
### Initialized receiver transport determines a contact-rate sign

Consider the original actual segment `[0,T]` up to a possible first paired
encounter by `7/4`. Its existence is not assumed; the earlier results imply
`3/2<T<=7/4` and `M(T)=pi/2` if it occurs. The comparison requires this
segment to remain in the closure of `G`; no later re-entry is covered.

First retain the source geometry. The exact reference starts with receiver
short gap `2*pi/9` and has speed below `3/5` on its admitted `[0,1]` prefix.
The actual time-one error is below `b=1/45000`, and (ST16) bounds the later
actual length by `753/3775` through `7/4`. With the short-gap dual norm
below `49/60`,

\[
 c>\frac23-\frac{49}{60}
       \left(\frac35+\frac1{45000}+\frac{753}{3775}\right)
   =\frac{5605001}{407700000}>\frac1{80}.
\]

At the exact time-one reference, `c_ref>2*pi/9-49/100` and
`a_ref<pi/2-1/6`, hence `w_ref<5*pi/36+97/600`. The lower bound
`w_ref>109/672` follows from (C10). Actual distance to this same
mean-matched reference is below `1/5` on `[1,7/4]`; its `w` covector
has norm below `7/12`. Consequently, throughout that actual window,

\[
 \boxed{c>1/80,\qquad
 51/1120=109/672-7/60<w
 <55/126+97/600+7/60<18/25.} \tag{SG30}
\]

These estimates consume the retained prefix length and actual comparison;
they require no paired membership or new reference evaluation. At the first
face the stronger lower bound `w>1/21` from (SG8) remains available.

Now retain the initialized folded comparison (SG20)--(SG21). Its nine
differences `z_Di,z_Ri,z_c` are nonnegative and initially only `z_c=m`,
where `m=pi/9-1/1000`. Summing all nine rows gives
`Z_sigma=-c_D4*z_D4-c_R4*z_R4<=0`, where
`Z=z_c+sum_i(z_Di+z_Ri)`. Thus every comparison difference is at most `m`.
Equivalently, the nodal maximum principle keeps `hbar` between `-m` and `m`.
The actual-to-comparison half-edge error below `e_*=1/18000` then yields

\[
 \boxed{|\delta_i|,|\epsilon_i|,|h_c|
 <m+e_*<22/63-1/1000+1/18000<7/20.} \tag{SG31}
\]

This bound holds on the original paired segment. It does not assert an
exact sign cone for the actual state or reinterpret a comparison variable
as its independently selected initialization.

Let `S_R=sum_i z_Ri`. The exact receiver and contact rows give

\[
 (S_R)_\sigma=c_cz_c/3-c_{R1}z_{R1}/3-c_{R4}z_{R4}
                  \le c_cz_c/3,\qquad
 (z_c)_\sigma\ge-2c_cz_c/3.
\]

Since `z_c>0`, the ratio `r=S_R/z_c` obeys
`r_sigma<=(c_c/3)(1+2r)`. Put
`I_c(s)=integral_0^s c_c(u) du`. Its integrating factor and the original
`S_R(0)=0` give

\[
 \boxed{\frac{z_{R1}}{z_c}\le\frac{S_R}{z_c}
 \le\frac{e^{2I_c(s)/3}-1}{2}
 \le\frac{e^{2s/3}-1}{2}<\frac98,\quad 0\le s\le T.} \tag{SG32}
\]

The last inequality uses `T<=7/4` and the proved `exp(7/6)<13/4`.
No coefficient history is numerically evaluated. A nonzero initial ratio
would contribute `r(0)*exp(2I_c/3)`; later positivity alone cannot remove it.
The actual implication is
`-epsilon_1<(9/8)*(-h_c)+17/144000`, with both half-edge errors retained.
This is a source-initialized temporal correlation, not an autonomous law for
one contact variable or an arbitrary positive state.

On `[1,T]`, the angle bounds give
`c_c=cos(w)*sinc(h_c)>(463/625)*(2351/2400)>29/40`.
With `c_R1<=1` and the existing `z_c>4/39`,
`2*c_c*z_c-c_R1*z_R1>(13/40)*z_c>1/30`.
Use the actual currents `k_1^-=c_R1*epsilon_1` and `j_c^-=c_c*h_c`.
The half-edge error changes this current combination by less than `3e_*`.
Therefore

\[
 k_1^- -2j_c^->1/30-1/6000=199/6000\quad(1\le s\le T).
\]

At the first face, `j_1^-=cos(M)*sin(delta_1)=0`. The complete-law port
row is `(h_c)_sigma=(k_1^- -j_1^- -2j_c^-)/3+e_h`, where
`|e_h|<(3/5)(3/200000)=9/1000000`. Hence

\[
 \boxed{(h_c)_\sigma(T)>199/18000-9/1000000>11/1000.} \tag{SG33}
\]

The negative mean contact gap is relaxing toward zero at such an encounter.
This signs one contact component; the contrast `w` and total contact current
`J_c=cos(h_c)*sin(w)` remain distinct observations.

There is also a finite pre-encounter band result. Whenever the actual
original segment has `1<=s<=T` and `pi/2-1/20<=M(s)<=pi/2`,
`j_1^-<7/400` by (SG31). The same port estimate and the strict forcing
bound `beta>1/64` in the proof of (SG22) give

\[
 \boxed{(h_c)_\sigma>1/200,\qquad
 (\delta_1)_\sigma>1/64-(5/6)(7/400)=1/960.} \tag{SG34}
\]

This proves simultaneous relaxation of the mean contact gap and growth of
one donor reflection difference while the source occupies this band. It
does not assert occupancy, monotone `M`, an invariant band or a post-exit
extension. The earlier global reflection budget can still be nonincreasing.

Finally the contact-origin part of the actual normal acceleration is signed
at the first face. From `h_c<-1/10`, `w>1/21`, (SG33), and the retained
small-angle ranges,

\[
 K_h:=-\tfrac13\sin(h_c)\sin(w)(h_c)_\sigma
 >\frac{599}{6000}\frac1{22}\frac{11}{1000}\frac13
 >\frac1{61000}. \tag{SG35}
\]

Here `sin(1/21)>1/22` follows from the cubic sine bound. Define the different
remaining rate combination
`mathcal R_rem=(J_2)_sigma/2+cos(a)*a_sigma/3+cos(h_c)*cos(w)*w_sigma/3`.
The exact normal acceleration, (SG22) and (SG26) imply

\[
 \boxed{M_{\sigma\sigma}
 >\mathcal R_{\rm rem}+\frac5{11136}+\frac1{61000}-\frac1{105000}
 >\mathcal R_{\rm rem}+\frac1{2200}.} \tag{SG36}
\]

A first tangency therefore requires `mathcal R_rem<-1/2200`. The older
(SG28)--(SG29) thresholds remain valid for their different remainders; none
is an equivalent normal-sign test. The short-gap contribution is negative
at an encounter, while neighboring-current and contact-contrast rates remain
to be ordered against it. This removes the missing contact-origin sign, not
the full rate competition, encounter existence, later recrossings or capture.

<a id="sine-formed-two-port-actual-inclusion"></a>
### Actual-law inclusion with capture and exclusion decisions

The global tracking results permit a new transport method after the consumed
reference attempt. It encloses the **actual** phases and reconstructs the
relative form through `P_D x=gamma A^-1 v`. It does not replace the original
source, evaluate the old reference beyond its admitted prefix, or require
acute, paired or Hessian-positive confinement.

Let `R` reconstruct a mean-zero eighteen-vector from its first seventeen
coordinates: its first seventeen rows are the identity and
`R[17,j]=-d_j/d_17`. Then `G=R^T D R` is positive definite and
`||R q||_D=||q||_G`. The coordinates are deviations from the same nominal
source used by the prefix. They remove one **common conserved phase mean**;
the two components are never recentered independently. The original affine
edge offsets and continuous lifts remain in every field evaluation.

Project the recorded time-one reference center onto the common mean-zero
space before taking its first seventeen coordinates. Orthogonal projection
cannot increase the error to an actual mean-zero state. The admitted initial
ball is therefore

\[
 q(1)\in B_G(c_1,r_1),\qquad r_1=R_1+1/45000. \tag{AI1}
\]

Here `R_1` is the retained fourth-step radius, including its outward
arithmetic, and the additional term is the actual/reference comparison
already proved at time one. No later analytic displacement bound is used as
if it were a newly evaluated trajectory center.

Writing `F(q)` for the first seventeen rows of `f(theta_0+R q)`, (ST3) and
(ST9) imply the actual differential inclusion

\[
 q'=F(q)+u,\qquad
 \|u\|_G<\min(1/7000,\|F(q)\|_G/20000). \tag{AI2}
\]

For an initial ball, a strict Picard tube must contain the **forced**
inclusion throughout the step. Coordinate forcing intervals use
`(G^-1)_{ii}^{1/2}/7000`. On that tube `X`, the smaller valid norm bound
`epsilon_X=min(1/7000,sup_X ||F||_G/20000)` may enter the endpoint estimate.
Taylor jets and Jacobians remain those of the smooth nominal field `F`;
the unknown forcing is not differentiated or inserted as a constant jet.

The graph supplies a signed local growth certificate without discarding
edge correlations. Let `B` be the oriented incidence matrix and let `l_e`
be outward lower bounds for `cos(g_e)` on the whole forced tube. Define

\[
 K_X=R^T B^T\operatorname{diag}(l_e)BR.
 \quad \mu G+K_X\succeq0
 \ \Longrightarrow\ \mu_G(DF(q))\le\mu\quad(q\in X). \tag{AI3}
\]

Indeed `G DF=-R^T B^T diag(cos(g_e))BR` is symmetric and bounded above
by `-K_X`. Negative edge cosines are admitted. This is a bound on the
relative phase field, not a stability verdict for the complete flow. The
common phase zero mode would preclude a negative full eighteen-dimensional
logarithmic bound. Lower edge cosines at least `-1` and `L<=2D` provide the
global upper bracket two; every computed certificate still checks its exact
semidefinite slack.

If the nominal center Taylor solution has endpoint enclosure error `a`,
the actual endpoint error satisfies

\[
 \|q(t+h)-c_+\|_G\le e^{\mu h}r+a+\epsilon_X\Phi(\mu,h),\qquad
 \Phi(\mu,h)=\int_0^h e^{\mu s}\,ds. \tag{AI4}
\]

The stored radius is an outward upper bound for this expression.
Use outward bounds for both factors, including the reversed division sign
when `mu<0`. The formula follows by the one-sided Lipschitz comparison of
two paths in the convex forced tube and variation of constants. It applies
to arbitrary bounded measurable forcing admitted by (AI2). Radius is never
recovered by shrinking a coordinate box. A failed Picard, growth or Taylor
condition gives an unavailable step and retains the preceding certificate.

There are two distinct terminal implications. If the entire actual phase
ball, including the retained target uncertainty, lies within `1/1024` of
the exact target, (ST7) proves entry of all thirty-six variables into the
existing forward capture domain. This establishes subsequent convergence
for every member of the original source family.

Alternatively, let `U(c)` and `F_c=||F(c)||_G` have outward upper bounds at
an endpoint center with radius `r`, and let `k=1/20000`. The global Hessian
bound two and the relative inverse bound ninety give

\[
 H\le U(c)+F_c r+r^2+
       45\eta(1+k)^2(F_c+2r)^2=:H_{\rm up}. \tag{AI5}
\]

The implementation uses `eta<1/9000000` and may sharpen `(1+k)F` to
`F+min(1/7000,F/20000)`, with `F=F_c+2r`.
The form term follows from
`x^T L x/2=eta <v,A^-1 v>_D/2`; it retains the actual velocity cone.
The uniform form contributes zero Laplacian storage and is preserved
separately. If `H_up` is strictly below an outward **lower** bound for the
exact target potential, full-law storage dissipation excludes later
convergence to that target. It also excludes any earlier genuine entry into
its forward capture domain. It does not identify a different destination,
rule out every organization, or rule out an earlier transient phase visit.
Failure of both tests establishes neither capture nor exclusion.

The producing owners are the shared
[inclusion kernel](../../src/tnfr/mathematics/_validated_metric_inclusion.py),
the [actual-source adapter](../../src/tnfr/research/formed_two_port_arrival.py)
and the [bounded runner](../../src/tnfr/research/formed_two_port_arrival_run.py).
The [response auditor](../../src/tnfr/research/formed_two_port_arrival_audit.py)
checks byte associations, carried balls, exact matrix/radius/terminal arithmetic
and the supervisor's post-write result. It retains the reviewed producer's
Picard, Taylor and primitive enclosure premises; it does not reexecute them.
Their source-independent controls and substantive review must precede
evaluation. The new finite policy selects times `1+j/16`, `0<=j<=496`,
Taylor order eight, at most sixteen Picard iterations per call and twelve
fixed growth bisections in `[-1/64,2]`, without bracket expansion. It tests
both terminal implications initially and after every complete step, stopping
at the first certified verdict, first unavailable step, or time thirty-two.
This horizon is a finite discriminator, not a derived convergence deadline.
One worker and one nonresumable attempt have at most 3,600 seconds, one GiB
of process/job committed memory, 256 MiB of step evidence and 64 MiB of
auxiliary evidence including supervisor/caller reserves. No scientific pilot,
parameter tuning, new root solve or post-failure retry belongs to that policy.
The previous attempt and diagnostic remain consumed and immutable.

The first actual-inclusion execution also remains consumed. It stopped with
`MemoryError` after input admission, with no complete step or `admission.json`.
That evidence establishes neither arrival nor exclusion and cannot locate the
failed operation or rule out an initial terminal evaluation. A separate
source-independent control reproduced an oversized allocation in the shared
artifact reader: requesting a 256-MiB allowance for a 34,816-byte synthetic
file exceeded the unchanged one-GiB process limit. Reading in blocks of at
most 64 KiB preserved exact bytes, the size sentinel and growth rejection;
the same bounded synthetic control then passed.

Execution revision two retains (AI1)--(AI5), the original source/target,
all numerical parameters and both terminal outcomes. It corrects that reader
and adds immutable entered/completed-stage records plus bounded stack-location
metadata, without locals or scientific-value logging. Its separately frozen
policy admits one fresh attempt after controls and independent review. It
does not resume or overwrite the failed execution, enlarge its resources,
or infer the unrecorded historical failure location from the synthetic test.

<a id="sine-formed-two-port-actual-exclusion"></a>
### Certified exclusion of the named organization for the original source

The separately reviewed execution revision two completed 313 actual-inclusion
steps of `1/16` from slow time one. At `sigma=329/16=20.5625`, the first
declared terminal test to certify a verdict was strict **total-storage
exclusion**, not capture. The exact endpoint inequalities imply

\[
 H(329/16)\le H_{\rm up}<9.58021<9.58023<H_*,
 \qquad H_* - H_{\rm up}>1/40000. \tag{AI6}
\]

The recorded upper bound is approximately `9.580202736204335`; the retained
target lower bound is approximately `9.580231243493376`. The exact fractional
margin exceeds `2.8507e-5`. These decimals summarize rational enclosures;
rounded values are not inputs to the decision. The positive form-storage
allowance, approximately `4.1869182e-9`, is included in `H_up` by (AI5).
The final retained phase radius is below `1/3500` in the degree metric.

Thus **every member of the original formed-source family, with
`phi=1/1000`, is excluded from convergence to this retained deformed
two-port target under the unchanged complete law**. Since full-law storage
cannot increase, no later convergence can have limit `H_*`. An earlier
entry into the forward capture domain (A8) would imply that same limit and
is therefore excluded as well. The statement includes the memberwise
conserved phase and form means; it uses neither a source reset nor a
replacement relative origin. It does not identify a different destination
or infer a particular winding slip.

The existing per-member convergence theorem (PA6)--(PA11) now implies a
stationary limit distinct from the excluded target for each member. That
theorem uses the analytic damped-gradient argument and finite total path
length, not storage monotonicity alone. A common limit across the family,
its stable basin and its internal organization are still unproved. The
shifted-origin capture window remains a different valid preparation.

Evidence is retained in
`artifacts/formed-two-port-actual-arrival-io-20261101`: the prospective
protocol/source archive, all 313 complete certificates, immutable phase
journal, post-write supervisor result and `audit.json`. The bounded worker
closed successfully in about 270.09 seconds with peak committed process
memory 869,560,320 bytes, below the unchanged one-GiB ceiling. Step evidence
occupies 42,035,649 bytes. All terminal conditions were declared before the
response; the execution stopped at its first certified verdict.

The first documentary audit incorrectly equated the virtual environment's
logical executable with the operating-system base image. A separately
reviewed audit-only correction checked the frozen runtime and reviewed
direct-image bootstrap, then verified the same 313 records. The producing
archive and all response files remain unchanged; no scientific response was
rerun for that correction. The audit verifies recorded associations and exact
arithmetic while retaining the reviewed producer's primitive-enclosure
premises, as stated above. This is a conditional mathematical exclusion,
not an empirical identification.

### Implementation and the remaining destination question

`bound_formed_two_port_transport` in
[formed_two_port_reference.py](../../src/tnfr/research/formed_two_port_reference.py)
reuses the existing prefix/post-acute owner and rebuilds the source, lag,
principal-margin, conditional cut and target-ball constants with rational arithmetic.
The shared default corridor width has one code owner. The result keeps
the global tracking start, finite retention interval and conditional
outgoing periods distinct; it contains no newly evaluated future state.
The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-formed-source-transport)
and [controls](../../tests/research/test_formed_two_port_transport.py) cover
both-row clock factors, weighted means, graph/edge coefficients, strict
margins, source substitution and altered prefix evidence.

The implemented report retains (ST3)--(ST7); (ST8)--(ST18) and (SG1)--(SG36)
are further analytical consequences, without new report fields or numerical
evaluation. In particular, its retained interval fields have not been extended
to time three by an implementation change.
The separate inclusion producer and audited (AI6) close the original named
target's accessibility question negatively. Which stationary organization the
same actual family selects, and whether its destination is common, remain
case-specific open questions. The subsequent
[critical-energy and integer-representative proof](SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-original-period-exclusion)
establishes eventual absence of the original periods under the unchanged law,
without selecting another class or common destination. Neither the displayed
cut patch nor another visually attractive equilibrium is a selected destination.
The [research program](../RESEARCH_PROGRAM.md#research-current-question) now
prioritizes a joint acquisition/maintenance mechanism; this unresolved case
does not automatically activate further transport. Earlier attempts remain
consumed, and physical mapping remains blocked.

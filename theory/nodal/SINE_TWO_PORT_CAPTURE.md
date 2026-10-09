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

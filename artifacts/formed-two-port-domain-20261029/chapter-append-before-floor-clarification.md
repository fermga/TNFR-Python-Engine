

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
common shifts. Suppose its actual phases satisfy `U>=U_floor`, and

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

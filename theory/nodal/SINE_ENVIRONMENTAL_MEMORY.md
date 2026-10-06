# Sine environmental state and causal memory

Complete sine mediator geometry, cancellation, conserved hidden inventory, exact causal elimination and conditional state/capacity inference with its sampling boundaries.

Part of [Native pattern reduction and memory](RELATIONAL_PATTERN_MEMORY.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="sine-mediator-common-geometry"></a>
### Common mediator geometry does not select its transmitted pressure

The [complete-law comparison](SINE_CONSTITUTIVE_INFORMATION.md#global-closure-pressure-comparison)
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
[fast-mediator proof](RELATIONAL_MEDIATOR_DYNAMICS.md#fast-mediator-reduction), the sine remainder obeys
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

The stationary-minimum reduction and the memory/chart owners support a
causal description with retained environmental state.
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
Reuse the [reflection/uniqueness method](RELATIONAL_NATIVE_FORMATION.md#regular-seeded-reachability-audit)
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

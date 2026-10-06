# Eleven-node receiver formation bounds

**Archived preparation-specific study.** The eleven-node receiver campaign is deferred. Its conditional proofs, negative results and unresolved transfer range remain reusable under their stated assumptions; this document assigns no current task.

The supplied donor/intermediary/receiver case study: preparation geometry, directional loss and auxiliary-function obstructions. No general absence of formation follows.

Part of [Native pattern reduction and memory](../../../nodal/RELATIONAL_PATTERN_MEMORY.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="sine-formation-eligibility"></a>

## 20. Formation eligibility and a finite-time exclusion

The recovery and causal-response results above start with both identities
already present. This section instead supplies a receiver with flat phase.
It asks whether an existing intermediary can form the receiver twist under
the same smooth, unforced sine law. A nonzero initial response, enough
initial storage and eventual formation are separate assertions.

Keep the eleven-node support of Section 19: donor ring `(0,1,2,3,4)`,
receiver ring `(5,6,7,8,9)`, intermediary `h=10`, and bridges
`0--h` and `5--h`. The ports have degree three and every other
node has degree two. Write

\[
\alpha=\frac{2\pi}{5},\qquad a=\frac w\pi,\qquad
b=\frac{w}{\beta\pi},\qquad
q_i=\sum_{j\sim i}(x_i-x_j),\qquad
S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i).
\]

The complete rows remain

\[
\dot x_i=\frac{\nu_i}{d_i}(-e q_i+aS_i),\qquad
\dot\theta_i=b\frac{\nu_i}{d_i}q_i,
\quad e,w,\beta>0.
\]

The initial preparation and held capacities are

\[
\begin{aligned}
&\theta_j=j\alpha\pmod{2\pi}\quad(0\le j\le4),\qquad
\theta_5=\cdots=\theta_9=\theta_h=0,\\
&x_h=A,\qquad x_i=0\quad(i\ne h),\\
&\nu_6=1+\delta,\qquad \nu_9=1-\delta,\qquad
\nu_i=1\quad(i\notin\{6,9\}),\qquad |\delta|<1.
\end{aligned}
\]

Here `A` is signed and the capacity contrast is supplied independently.
They are preparation parameters, not a new pressure term or a derived
event-selection law. The desired endpoint is the same full critical
geometry as in Section 19: both rings have winding `+1` and equal
increments `alpha`, the bridge phases agree, and form is uniform.
Convergence is understood modulo the admitted common origins.

### Exact odd response and its controls

At this preparation all sine sums vanish. The only nonzero form gradients
are `q_0=q_5=-A` and `q_h=2A`. Consequently

\[
\dot x_0=\dot x_5=\frac{eA}{3},\qquad
\dot x_h=-eA,\qquad
\dot\theta_5=-\frac{bA}{3}.
\]

For the initially flat receiver, let `chi=theta_6-theta_9` on the
continuous lift through zero. Its value and first derivative vanish.
Since `dot q_6=dot q_9=-eA/3`,

\[
\ddot\theta_6(0)=-\frac{beA(1+\delta)}6,\qquad
\ddot\theta_9(0)=-\frac{beA(1-\delta)}6,
\]

and hence

\[
\boxed{\ddot\chi(0)=-\frac{be\delta A}{3},\qquad
\chi(t)=-\frac{be\delta A}{6}t^2+O(t^3).}
\]

Thus `delta*A!=0` produces an exact local reflection-breaking
response. This Taylor identity neither certifies a finite prediction
horizon nor selects the final winding. In particular, the sign of its
initial curvature is not a proof of the eventual identity.

The two controls are exact. When `delta=0`, receiver reflection is
preserved as proved in Section 19, for every `A`, so the receiver
cannot converge to either nonzero twist. When `A=0`, both `q`
and `S` vanish everywhere: the mixed donor-twist/flat-receiver state
is a full equilibrium for every admitted capacity contrast. Reflection
also exchanges the preparations `delta` and `-delta`, rather
than providing a mechanism that chooses one of them.

### Storage is necessary, but target storage is not the entry barrier

Use the same storage and balance as in Section 18:

\[
E=\frac12\sum_{\{i,j\}}(x_i-x_j)^2
 +\beta\sum_{\{i,j\}}\bigl(1-\cos(\theta_j-\theta_i)\bigr),
\qquad
\dot E=-e\sum_i\frac{\nu_i}{d_i}q_i^2.
\]

Define the unit phase storage of one exact twist by

\[
V_5=5\left(1-\cos\frac{2\pi}{5}\right)
    =\frac{25-5\sqrt5}{4}.
\]

For the supplied family,

\[
E(0)=\beta V_5+A^2,\qquad
E_{\mathrm{target}}=2\beta V_5,\qquad
\dot E(0)=-\frac{8e}{3}A^2.
\]

The capacity contrast does not change this initial loss: the two
contrast nodes have zero initial form gradient. For `A!=0`, the
loss is strictly positive over some initial time interval. Therefore
`A^2>beta*V_5` is necessary for convergence to the desired endpoint.
It is not sufficient, even at the level of a continuous phase path.

Two tempting stronger arguments need qualification. The closure of
winding-one C5 configurations contains an antipodal face represented
by increments `(pi,pi/4,pi/4,pi/4,pi/4)`. Its unit phase
storage is `6-2*sqrt(2)`, which is smaller than `V_5`.
The antipodal winding itself is branch-dependent; nearby increments
`(pi-epsilon,(pi+epsilon)/4,...,(pi+epsilon)/4)` have unambiguous
winding one and still cost less than `V_5` for sufficiently small
positive `epsilon`. Thus winding alone does not impose the twist's
storage lower bound. The critical geometry
`(2*pi/3,pi/3,pi/3,pi/3,pi/3)` has storage `7/2`, but a
critical value alone is not a proof that every relevant path crosses it.
Neither observation supplies the required full-network transition theorem.

Instead define `U` as the set in which **both** rings have strictly
acute principal edge increments and winding `(+1,+1)`. No restriction
on form or bridge phase is imposed in this definition. The initial
receiver is outside the closure of `U`. Any trajectory converging
to the desired target enters `U` at a finite time. At the positive
infimum of those entry times, continuity supplies a boundary state
in which both rings are closed acute with these same windings.
At least one ring is on an acute face.
This argument allows the donor to unwind or leave its acute region
earlier; it makes no assumption about the donor's entire past.

On a closed-acute winding-one C5 the five increments sum to `2*pi`.
A boundary increment cannot be `-pi/2`, since the remaining four
increments are at most `pi/2` each. Thus at least one is `pi/2`;
the remaining four sum to `3*pi/2`. Convexity of `1-cos` on
`[-pi/2,pi/2]` gives the sharp acute-face lower bound

\[
B_5=1+4\left(1-\cos\frac{3\pi}{8}\right)
   =5-4\cos\frac{3\pi}{8},
\]

attained when the remaining increments are all `3*pi/8`. The
other ring has storage at least `V_5` by the same convexity and
its fixed sum. Bridge and form storage are nonnegative. At the joint
entry boundary, therefore,

\[
E\ \ge\ \beta(V_5+B_5).
\]

Here `B_5>V_5` by strict convexity: its five boundary increments
are not equal. Combining this bound with the strict initial loss gives
the stronger necessary condition

\[
\boxed{A^2>\beta B_5.}
\]

For example, at `beta=1`, `A=931/500` gives
`A^2=3.467044`, strictly between
`V_5=3.454915...` and `B_5=3.469266...`.
Its initial storage exceeds the final target storage, but cannot
finance entry to the joint acute target region. This is a path
obstruction, not a failure of a numerical solver.

### An analytic time window can exclude an apparently eligible preparation

The energy bound `E(t)<=E(0)=E_0` yields, by Cauchy--Schwarz,

\[
|q_i(t)|\le\sqrt{2d_iE_0},\qquad
|\dot x_i(t)|\le
M_i:=\nu_i\left(e\sqrt{\frac{2E_0}{d_i}}+a\right).
\]

Since `q_h=2x_h-x_0-x_5`,

\[
|\dot q_h|\le C:=2M_h+M_0+M_5,\qquad
|q_h(t)|\ge (q_{\mathrm{init}}-Ct)_+,\qquad
q_{\mathrm{init}}=2|A|.
\]

The intermediary alone then supplies a lower bound on accumulated loss
over any supplied window `[0,tau]`:

\[
\begin{aligned}
s&=\min\left(\tau,\frac{q_{\mathrm{init}}}{C}\right),\\
L(\tau)&=\frac e2\left(
 q_{\mathrm{init}}^2s-q_{\mathrm{init}}Cs^2+\frac{C^2s^3}{3}
\right)
\ \le\ E(0)-E(\tau).
\end{aligned}
\]

The positive part is essential: squaring the negative continuation of
`q_init-C*t` would invent a loss lower bound after its zero.
Certified upper bounds for `E_0` and `C` can replace their
exact values conservatively.

For every receiver edge, the phase-rate bound is

\[
|\dot\theta_j-\dot\theta_i|
\le b\sqrt{2E_0}
\left(\frac{\nu_i}{\sqrt{d_i}}+\frac{\nu_j}{\sqrt{d_j}}\right).
\]

Let `G` be the maximum right-hand side over the receiver edges.
If `G*tau<pi/2`, every receiver increment stays strictly acute
through that window, on its continuous lift from zero. Its winding
therefore stays zero: joint target entry cannot occur before `tau`.
If also

\[
\boxed{\beta B_5+L(\tau)-A^2>0,}
\]

then `E(tau)<beta*(V_5+B_5)`, so joint target entry cannot
occur later either. This combines a finite phase-speed bound with
irreversible loss; it does not require a simulated trajectory or a
claim about the donor's intervening winding.

### A uniform exclusion for the bounded localized-pulse family

Take the default coefficients `e=w=1/2,beta=1`. The preceding
argument excludes **every** preparation

\[
\boxed{|A|\le2,\qquad |\delta|<1}
\]

from convergence to the specified two-twist target. For `|A|<=1`
the initial storage already fails the joint entry barrier. For
`u=|A|` in `[1,2]`, set `tau=1/5`. Since
`V_5<7/2`,

\[
E_0<\frac{15}{2},\qquad
\sqrt{E_0}<\frac{11}{4},\qquad
\sqrt{\frac{2E_0}{3}}<\frac94,\qquad a=b<\frac16.
\]

The three nodes entering `C` all have unit capacity, so

\[
C=2e\left(\sqrt{E_0}+\sqrt{\frac{2E_0}{3}}\right)+4a
 <\frac{17}{3}<6.
\]

Throughout `[0,1/5]`,
`|q_h(t)|>=2u-6t>=4/5`. In particular, the positive part
has not expired. The accumulated loss is bounded below by

\[
L\ \ge\ \frac14\int_0^{1/5}(2u-6t)^2\,dt
 =\frac{u^2}{5}-\frac{3u}{25}+\frac3{125}.
\]

Compare it with the loss the initial state could afford before losing
access to the joint entry boundary:

\[
L-(u^2-B_5)
\ \ge\ B_5-\frac45u^2-\frac3{25}u+\frac3{125}
\ \ge\ B_5-\frac{427}{125}
\ >\frac3{125}.
\]

The middle expression is decreasing on `[1,2]`. The last strict
inequality uses `B_5>86/25`, an elementary exact bound:
`sqrt(2)>7/5` implies
`cos^2(3*pi/8)<3/20<(39/100)^2`, hence
`cos(3*pi/8)<39/100`.

It remains to exclude entry before this loss has occurred. Adjacent
receiver capacities sum to at most `2+|delta|<3`, and their
degrees are at least two. Thus the receiver-edge speed satisfies
`G<3*b*sqrt(E_0)<11/8`, and every receiver phase increment
has magnitude less than

\[
G\tau<\frac{11}{40}<\frac{\pi}{2}
\]

through `tau=1/5`. Its winding remains zero until the total
storage is already strictly below the target-entry requirement.
This proves the claimed exclusion for the whole bounded family.

The case `A=2,delta=1/2` makes the distinction especially clear:
it has a nonzero odd response and passes both initial storage tests,
but the simple analytic loss bound is `73/125=0.584` by
`tau=1/5`, exceeding the available margin `4-B_5`.
Neither breaking the symmetry nor supplying that much localized form
storage suffices to form the desired receiver pattern.

### What this excludes and what it leaves open

The shared owner `physics/relational_sine_formation.py` evaluates
the exact preparation identities, joint entry barrier and optional
time-window exclusion. Passing a necessary condition means only that
this check has not ruled the preparation out. It does not certify
formation, a basin of attraction or physical emergence. The strict
loss arguments require `e>0`; the uniform exclusion additionally
uses the declared default coefficients and amplitude range.

The result does not exclude other preparations, larger amplitudes or
other targets. It also identifies a concrete source of avoidable loss
without changing the law. If donor form is a common `D`, intermediary
form is `H` and receiver form remains zero, put `s=D-H,r=H`.
The initial form storage and loss are exactly

\[
F=\frac{s^2+r^2}{2},\qquad
-\frac{\dot E(0)}e
 =\frac{s^2+r^2}{3}+\frac{(r-s)^2}{2}
 =\frac{2F}{3}+\frac{(r-s)^2}{2}.
\]

Within this declared two-parameter family, fixed `F` is least
dissipative initially at `s=r`, or `D=2H`. Setting `H=A`
retains `F=A^2` and the same receiver odd acceleration, while
reducing initial loss from `8eF/3` to `2eF/3`. No receiver
phase or winding seed is added. This is a restricted preparation
comparison, not a global optimizer or a formation theorem; its later
admission still requires a separate analysis. The present exclusion
concerns the original localized preparation `D=0,H=A` only.

<a id="sine-balanced-formation"></a>

## 21. A balanced preparation and a directional loss obstruction

Retain the full support, initial phases, positive held capacities and
desired two-twist target of Section 20. Change only the supplied form:

\[
x_0=\cdots=x_4=2A,\qquad x_h=A,\qquad
x_5=\cdots=x_9=0.
\]

This is the fixed-storage, least-initial-loss preparation within the
two-parameter family proved there. It does not place a receiver phase,
winding or form pattern into the initial state. It also does not
minimize loss over all possible full-network preparations.

The initial form gradient and sine sum are exactly

\[
q(0)=A(\mathbf e_0-\mathbf e_5),\qquad S(0)=0.
\]

Thus the intermediary initially has zero form gradient and zero rate,
while the donor and receiver port rates are

\[
\dot x_0=-\frac{eA}{3},\qquad
\dot x_5=\frac{eA}{3},\qquad
\dot\theta_0=\frac{bA}{3},\qquad
\dot\theta_5=-\frac{bA}{3}.
\]

All other initial rates vanish. The initial storage is still
`beta*V_5+A^2`, but its loss is now `2eA^2/3`.
The receiver's initial rates and odd acceleration
`(theta_6-theta_9)''=-be*delta*A/3` are unchanged; other
derivatives are not. In particular, the earlier localized
intermediary-loss estimate cannot transfer because `q_h(0)=0`.
A separate full-network argument is needed.

### A bound that retains the initial direction

Let `L` be the full unit-support Laplacian and put

\[
K=\operatorname{diag}\left(\frac{\nu_i}{d_i}\right),\qquad
B=K^{1/2}LK^{1/2},\qquad
H(\theta)=\nabla_\theta^2
 \sum_{\{i,j\}}\bigl(1-\cos(\theta_j-\theta_i)\bigr).
\]

Positive capacity makes `K` positive definite. For every real
vector `v`, the edge representation gives

\[
|v^\mathsf T K^{1/2}H(\theta)K^{1/2}v|
 \le v^\mathsf T Bv,\qquad
0\preceq B\preceq 2\max_i\nu_i\,I.
\]

The first bound uses only `|cos|<=1`, and holds outside acute
phase regions as well. The second follows by bounding each squared
edge difference by twice the sum of its squared endpoints.
Choose any positive spectral upper bound `lambda` satisfying
`||B||<=lambda`; it then also bounds
`||K^(1/2) H(theta) K^(1/2)||` for every phase state.

Define the exact variables

\[
y=K^{1/2}q,\qquad z=K^{1/2}S,\qquad
C(\theta)=K^{1/2}H(\theta)K^{1/2}.
\]

Differentiating the full nonlinear rows yields

\[
\dot y=-eBy+aBz,\qquad
\dot z=-bC(\theta)y,\qquad
-\dot E=e\|y\|^2.
\]

Here `S=-grad V`, so its derivative carries the displayed
minus sign. No phase linearization or diffusion-only trajectory
has replaced the supplied law.

Suppose `S(0)=0` and `q(0)!=0`. Write
`Y_0=||y(0)||` and `u_0=y(0)/Y_0`, and retain the exact
initial spectral moments

\[
R=u_0^\mathsf TBu_0,\qquad
\gamma=\|Bu_0\|,\qquad
\omega=\sqrt{ab}\,\lambda.
\]

For `Y=||y||` and `Z=||z||`, norm upper derivatives satisfy

\[
D^+Y\le a\lambda Z,\qquad D^+Z\le b\lambda Y.
\]

The omitted contribution in the first inequality is nonpositive
because `B` is positive semidefinite. Comparison with this
cooperative scalar system, starting from `(Y_0,0)`, gives

\[
Y(t)\le Y_0\cosh(\omega t),\qquad
Z(t)\le Y_0\sqrt{\frac ba}\sinh(\omega t).
\]

For a lower bound, use variation of constants only as an exact
identity for the first full row:

\[
\langle u_0,y(t)\rangle
 =Y_0\langle u_0,e^{-eBt}u_0\rangle
 +a\int_0^t
 \langle Be^{-eB(t-s)}u_0,z(s)\rangle\,ds.
\]

The spectral weights of the first inner product are nonnegative
and sum to one. Convexity of the scalar exponential therefore gives
`<u_0,exp(-eBt)u_0> >= exp(-eRt)`.
Moreover, `||B exp(-eBs)u_0||<=gamma` for `s>=0`.
Combining these facts with the bound on `Z` proves

\[
\boxed{
\frac{Y(t)}{Y_0}\ge
e^{-eRt}-\frac{\gamma}{\lambda}
 \bigl(\cosh(\omega t)-1\bigr).}
\]

The right-hand side need not stay positive indefinitely; a negative
value cannot be squared to infer a loss lower bound. The exact
initial direction enters through `R` and `gamma` rather
than treating all of `y(0)` as the fastest Laplacian mode.

### A rational finite-window certificate

There is no need to evaluate a matrix exponential or a hyperbolic
function to obtain a useful certificate. As one sufficient case,
if a supplied horizon
`tau` satisfies `omega*tau<=2/5`, then

\[
\cosh(\omega t)-1
 \le \frac{11}{20}\omega^2t^2,\qquad 0\le t\le\tau.
\]

Indeed, the nonnegative Taylor series and `(2k)!>=2^k` give
`cosh(2/5)<=25/23<11/10`; integrating the bound on the
second derivative twice gives the displayed inequality. More
generally, any supplied rational bound
`v>=omega^2*tau^2` with `0<=v<2` gives

\[
\cosh(\omega t)
\le\sum_{j=0}^{\infty}\left(\frac v2\right)^j
=\frac1{1-v/2}=:M_\tau,\qquad 0\le t\le\tau.
\]

This is the same factorial estimate, without the earlier
`2/5` restriction. The earlier `M_tau=11/10` remains a
valid short-window choice when `v<=4/25`. Neither choice
selects a horizon or changes the law. At `v>=2` the
geometric-series estimate is unavailable, not a proof of
dynamical failure.

Using either admitted bound `M_tau` and a certified
`gamma_bar>=gamma`, set

\[
d=eR,\qquad c=\frac{M_\tau}{2}\bar\gamma\,\lambda ab,\qquad
g(t)=1-dt-ct^2.
\]

Since `exp(-eRt)>=1-eRt`, if `g(tau)>0` then
`Y(t)>=Y_0*g(t)>0` on the whole window. Accumulated loss
consequently obeys the computable bound

\[
\begin{aligned}
E(0)-E(\tau)
&\ge eY_0^2\int_0^\tau g(t)^2\,dt\\
&=eY_0^2\left[
\tau-d\tau^2+\frac{d^2-2c}{3}\tau^3
 +\frac{dc}{2}\tau^4+\frac{c^2}{5}\tau^5
\right].
\end{aligned}
\]

It applies to the actual nonlinear solution over the supplied
window. Its hypotheses are admitted independently of a measured
response. If a separate nodal argument supplies another lower
bound on the same accumulated loss, their maximum is valid;
adding them would generally count the same loss twice.
Specifically, the form-speed bounds of Section 20 give
`|q_i'|<=d_i*M_i+sum_(j~i) M_j` at every node.
Integrating each positive affine lower bound on `|q_i|`,
with its actual initial gradient and weight `e*nu_i/d_i`,
and summing those disjoint nodal losses gives one such alternative.
This also explains why a zero initial intermediary gradient does
not make the balanced preparation's full loss zero.

### Exact moments for the balanced eleven-node family

For the present preparation,

\[
Y_0^2=\frac{2A^2}{3},\qquad R=1,\qquad
\gamma^2=\frac43.
\]

To verify these values, `Kq(0)` has only the two port entries
`+A/3,-A/3`. The vector `LKq(0)` has entries `+A,-A`
at the ports, `-A/3` at donor nodes `1,4`,
`+A/3` at receiver nodes `6,9`, and zero elsewhere.
In particular, its intermediary entry cancels exactly. It follows
that `y(0)^T B y(0)=2A^2/3` and
`||B y(0)||^2=8A^2/9`. The two receiver capacities enter
the latter sum only through `nu_6+nu_9=2`, so both moments
hold for every `|delta|<1`. They also imply
`D'(0)=-2eD(0)` for `D=-E'`, since `z(0)=0`.

At the default coefficients `e=w=1/2,beta=1`, use
`lambda=4` and `gamma_bar=7/6`:
`max nu_i<2`, `sqrt(4/3)<7/6`, and
`a=b=1/(2*pi)<1/6`. For `tau=3/5` the preceding
horizon condition holds. A convenient rational majorant for the
quadratic coefficient is

\[
c_0=\frac{77}{1080}
 \ >\ \frac{11}{20}\bar\gamma\,\lambda ab.
\]

Thus on `[0,3/5]`,

\[
\frac{Y(t)}{Y_0}\ge g_0(t)
 :=1-\frac t2-\frac{77}{1080}t^2,\qquad
g_0(3/5)=\frac{2023}{3000}>0.
\]

Its exact integral and resulting loss bound at `A=2` are

\[
\int_0^{3/5}g_0(t)^2\,dt
 =\frac{32259179}{75000000},\qquad
E(0)-E(3/5)\ge
\frac{32259179}{56250000}>0.57349.
\]

These are rational inequalities derived from the law, not values
sampled from a computed trajectory.

### The same bounded amplitude family is still excluded

The result is uniform over `|A|<=2,|delta|<1`. Handle
`A=0` by the exact equilibrium control. Otherwise,
`c_0<1/8` implies

\[
g_0(t)^2
 =1-t+\left(\frac14-2c_0\right)t^2
   +c_0t^3+c_0^2t^4>1-t
\quad(0<t\le3/5).
\]

Therefore

\[
E(0)-E(3/5)>
\frac{A^2}{3}\int_0^{3/5}(1-t)\,dt
=\frac{7A^2}{50}.
\]

The receiver starts flat, and the same global energy estimate as
in Section 20 gives `G<11/8` for every `|A|<=2`.
Through `tau=3/5` its edge increments consequently have
magnitude less than `33/40<pi/2`: it remains acute with
winding zero. At that time the storage deficit relative to the
joint target-entry boundary is strictly larger than

\[
B_5+\frac{7A^2}{50}-A^2
 =B_5-\frac{43A^2}{50}
 \ge B_5-\frac{86}{25}>0.
\]

It cannot have entered the joint target region before this window
and lacks enough storage to enter later. Thus the balanced
preparation is excluded throughout the same bounded family,
despite its fourfold reduction in initial loss. For `A=2,delta=1/2`
the exact odd response still occurs; it does not lead to the
specified two-twist endpoint.

This proves a new preparation-specific obstruction, not a universal
absence of pattern formation. The following corollary extends it
to the full constant-donor form family; neither result excludes
larger storage, arbitrary form profiles, other targets or other
coefficients.
Its reusable contribution is a nonlinear loss certificate that
retains the initial spectral direction, coupled to the independently
proved phase-speed and joint-entry conditions. The shared formation
owner keeps this evidence distinct from the localized-pulse bound.
The remaining question is whether a declared preparation can change
the required phase sector before losing access to its target, rather
than whether it improves an instantaneous loss statistic.

### Corollary: the whole constant-donor preparation family

At the default coefficients `e=w=1/2,beta=1`, retain the same
initial phases, capacities `|delta|<1` and two-twist target.
Allow arbitrary signed constants `D,H` with donor form `D`,
intermediary form `H` and receiver form zero. Define

\[
u=\frac D2,\qquad v=H-\frac D2,\qquad
F=\frac{(D-H)^2+H^2}{2}=u^2+v^2.
\]

Every such preparation with `F<=4` is excluded from convergence
to the target. When `F=0`, both constants vanish and the
prepared state is the exact equilibrium already identified.
For `F>0`, put `rho=v^2/F`, so `0<=rho<=1`.
The initial gradient has only three possibly nonzero entries,

\[
q_0=u-v,\qquad q_5=-u-v,\qquad q_h=2v,
\]

and `S(0)=0` still holds. The same full-support calculation
as above gives the exact moments

\[
\begin{aligned}
N=\|y(0)\|^2&=\frac{2u^2+8v^2}{3},\\
m_1=y(0)^\mathsf TBy(0)&=\frac{2u^2}{3}+4v^2,\\
m_2=\|By(0)\|^2&=\frac{8u^2+58v^2}{9}.
\end{aligned}
\]

For example, `LKq` has port entries `u-2v,-u-2v`,
intermediary entry `8v/3`, donor-neighbor entries
`(-u+v)/3` and receiver-neighbor entries `(u+v)/3`.
The receiver capacity sum `nu_6+nu_9=2` cancels all
contrast dependence in the moments. Thus

\[
\frac NF=\frac{2+6\rho}{3},\qquad
R=\frac{m_1}{N}=\frac{1+5\rho}{1+3\rho}\in[1,3/2],
\qquad
\gamma^2=\frac{m_2}{N}
 =\frac{8+50\rho}{6+18\rho}\le\frac{29}{12}
 <\left(\frac85\right)^2.
\]

Use `lambda=4,gamma_bar=8/5,tau=3/5` in the same
directional certificate. Its quadratic coefficient is bounded
above by `c_1=22/225<1/8`. The polynomial
`g(t)=1-R*t/2-c_1*t^2` satisfies

\[
g(3/5)\ge\frac{1287}{2500}>0,\qquad
g(t)^2>1-Rt\quad(0<t\le3/5).
\]

The second inequality follows by expansion:
its quadratic coefficient is
`R^2/4-2c_1>=49/900>0` and its cubic and quartic
coefficients are positive. The accumulated loss therefore obeys

\[
\begin{aligned}
E(0)-E(3/5)
&>\frac N2\left(\frac35-\frac{9R}{50}\right)\\
&=\frac{F(7+15\rho)}{50}
\ \ge\ \frac{7F}{50}.
\end{aligned}
\]

This holds for both signs of `u,v` and does not require
a nonzero odd initial receiver acceleration. Meanwhile,
`E(0)=V_5+F<15/2` for `F<=4`, so the same receiver
gap bound keeps its winding zero through the entire window:
`G*tau<33/40<pi/2`. The remaining storage is then below
the joint target-entry boundary, because its deficit is strictly
greater than

\[
B_5-\frac{43F}{50}\ge B_5-\frac{86}{25}>0.
\]

Consequently no other choice of `D,H` inside this fixed
storage budget can evade the obstruction. This removes the need
to guess further profiles within the same two-parameter family.
The engine report retains its `localized` and `balanced`
profiles. The explicit donor-form interface described below also
admits this whole constant-donor family as a subcase; neither
interface evaluates its trajectories.

<a id="sine-internal-form-geometry"></a>

## 22. Donor shape, phase action and a uniform formation obstruction

Keep the complete eleven-node support, donor twist, flat receiver
and intermediary phase, and default coefficients of Sections 20-21.
For this preparation family fix `delta=1/2`. Supply arbitrary signed donor
form `D=(D_0,...,D_4)` and intermediary form `H`, while
receiver form remains zero. These six independent initial values
replace a constant-donor restriction; no new evolution law,
edge or input is introduced.

### The exact six-coordinate quadratic forms

Let `z=(D_0,D_1,D_2,D_3,D_4,H)^T` and let `P`
place these coordinates at full-network nodes `(0,1,2,3,4,10)`,
placing zeros at the receiver. With the same full `L,K` as
in Section 21, the exact storage and spectral moment matrices are

\[
\begin{aligned}
F&=\tfrac12 z^\mathsf TP^\mathsf TLPz,\\
N&=z^\mathsf TP^\mathsf TLKLPz,\\
m_1&=z^\mathsf TP^\mathsf TLKLKLPz,\\
m_2&=z^\mathsf TP^\mathsf TLKLKLKLPz.
\end{aligned}
\]

These expressions retain all receiver and intermediary rows;
they do not substitute a six-node graph. Equivalently,

\[
F=\frac12\left[
\sum_{j=0}^4(D_j-D_{j+1\bmod5})^2+(D_0-H)^2+H^2
\right],
\]

and the only possibly nonzero initial gradient entries are

\[
\begin{aligned}
q_0&=3D_0-D_1-D_4-H,\\
q_j&=2D_j-D_{j-1}-D_{j+1}\quad(1\le j\le4),\\
q_5&=-H,\qquad q_h=2H-D_0,
\end{aligned}
\]

where the donor subscripts in the second line are taken modulo
five. Thus

\[
N=\frac{q_0^2}{3}
 +\frac12\sum_{j=1}^4q_j^2+\frac{H^2}{3}
 +\frac{(2H-D_0)^2}{2}.
\]

The remaining forms can also be evaluated without a matrix square
root: set `k_i=nu_i/d_i` and `v_i=k_iq_i`. Then

\[
m_1=\sum_{\{i,j\}}(v_i-v_j)^2,\qquad
m_2=\sum_i k_i
 \left[\sum_{j\sim i}(v_i-v_j)\right]^2.
\]

All four forms are rational quadratic forms on the declared six
coordinates. In fact their coefficients are independent of
`delta` throughout `|delta|<1`: the two contrast capacities
first enter `m_2` through receiver neighbors with equal
`(Lv)_6=(Lv)_9=H/3`, and their sum is fixed.
The interval loss and receiver-speed bounds still consume the
actual individual capacities.

### Which donor information reaches the receiver first

The exact initial sine currents vanish for every supplied form.
Let `chi=theta_6-theta_9` and `eta=e^2-ab`. Direct
differentiation of the full rows gives

\[
\begin{aligned}
\dot x_5(0)&=\frac{eH}{3},&
\dot\theta_5(0)&=-\frac{bH}{3},\\
\ddot\theta_5(0)&=\frac{be}{6}(4H-D_0),&
\ddot\chi(0)&=-\frac{be\delta H}{3},\\
\theta_5^{(3)}(0)&=
\frac{b\eta}{18}(9D_0-D_1-D_4-22H),&
\chi^{(3)}(0)&=\frac{b\eta\delta}{6}(8H-D_0).
\end{aligned}
\]

Here `chi(0)=chi'(0)=0`. These are derivatives of the
full nonlinear law at the supplied preparation, not derivatives
of an independently fitted or linearized response. The receiver
first reads the intermediary form; the donor port and then its
neighbor sum appear at higher derivative orders. The two remaining
donor coordinates do not appear in these displayed receiver
derivatives. Their absence at these orders does not prove absence
of later influence.

There is, however, an exact silent two-dimensional subspace:

\[
D_0=H=0,\qquad
D_1=-D_4,\qquad D_2=-D_3.
\]

Let `Q=(1\,4)(2\,3)` reflect only the donor, fixing its
port, the intermediary and every receiver node. The full law is
equivariant under the composition

\[
(x,\theta)\longmapsto(-Qx,-Q\theta)
\]

on the circle-valued phase state. The donor capacities are paired
equally, while this permutation does not exchange receiver
capacities. The prepared donor twist and the above form subspace
are fixed by this transformation. Uniqueness therefore preserves
the combined symmetry for all time.

At each fixed receiver or intermediary node it forces form zero
and a phase in `{0,pi}`. Continuity from the prepared phase
zero fixes that phase to zero forever. Thus the receiver and
intermediary remain exactly unchanged, even at nonzero capacity
contrast and arbitrarily large form storage in this subspace.
This is cancellation under an existing interaction law, not absence
of the supplied edges. Receiver formation is impossible for these
preparations without a separately supplied symmetry-breaking change.

The six-coordinate space also splits into donor-reflection-even
and donor-reflection-odd parts. The four quadratic forms above have
no cross terms between those parts. The odd two-dimensional part
is exactly this silent form subspace. For `D_1=s,D_2=t`
its storage and initial dissipative norm are
`F=2s^2-2st+3t^2` and `N=5s^2-10st+10t^2`.
Nonzero internal loss alone therefore does not establish a
receiver response.

<a id="sine-source-receiver-excitation"></a>
### Source symmetry, nonlinear work and a uniform tangent obstruction

Retain the same six-coordinate source, `e=w=1/2`, `beta=1`,
`delta=1/2`, all eleven fine nodes and no input or event. The
following decomposition is exact; it does not declare the receiver's
future input independently of the donor and intermediary.

Write the initial source in coordinates

\[
\begin{aligned}
C&=D_0-H,&
s_1&=(D_1+D_4)/2-D_0,&s_2&=(D_2+D_3)/2-D_0,\\
o_1&=(D_1-D_4)/2,&o_2&=(D_2-D_3)/2.
\end{aligned}
\]

`H,C,s_1,s_2` specify the donor-reflection-even form and its
live port contrasts. The two `o` coordinates specify the odd form.
With `Q=(1 4)(2 3)` extended by the identity outside the donor,
these are `x_e=(x+Qx)/2` and `x_o=(x-Qx)/2`. Direct edge
accounting gives

\[
\boxed{\begin{aligned}
F&=F_e+F_o,\\
F_e&=\frac12(H^2+C^2)+s_1^2+(s_2-s_1)^2,\\
F_o&=o_1^2+(o_2-o_1)^2+2o_2^2.
\end{aligned}}
\]

The same orthogonal parity split holds for `N,m_1,m_2`, and
indeed every initial moment `y_0^T B^j y_0`, because `Q`
commutes with `L,K` and `B=K^(1/2)LK^(1/2)`. This does not
make the nonlinear field a direct sum of two systems. The exact
silence theorem applies to the odd source **alone**. Removing its
coordinates from a mixed preparation requires an additional closure
argument, which the next counterexample disproves.

#### Equal energy and spectral moments do not determine receiver work

Compare the two exact rational preparations

\[
H=1,\qquad D^+=(0,1,0,0,-1),\qquad D^-=-D^+.
\]

They have identical even coordinates, `F_e=1`, `F_o=2`,
`F=3`, `N=23/3`, and every initial quadratic spectral moment.
They differ only in the sign of the silent odd source. Keep the
initial donor phase orientation `+1` fixed in both preparations;
reversing form is not a reversal of that prepared phase geometry.
For a quantity `f`, write `Delta f=f^+-f^-`, and put
`alpha=2*pi/5`, `a=w/pi`, `b=w/(beta*pi)` as above.

The first mixed term is at the donor port. If `v=KLx`, its
even and odd components satisfy `v_0=-1/3`, `v_{1,e}=0`
and `v_{1,o}=1` for the positive preparation. Differentiating
the actual sine current twice gives

\[
\boxed{n_0:=\Delta x_0^{(3)}(0)
 =-\frac83ab^2\sin\alpha\,
             (v_{1,e}-v_0)v_{1,o}
 =-\frac89ab^2\sin\alpha.}
\]

For clarity, the quadratic contribution is
`-sin(alpha)*[(dot theta_1-dot theta_0)^2
-(dot theta_4-dot theta_0)^2]` in `S_0''`; the bridge has
zero initial sine. Linear contributions at the fixed port agree
by reflection. Thus the displayed difference is a derivative of
the full nonlinear law, not a fitted higher-order response.

Transmission through the retained intermediary yields

\[
\Delta x_{10}^{(4)}(0)=\frac e2n_0,\qquad
\Delta x_5^{(5)}(0)=\frac{e^2-ab}{6}n_0,\qquad
\Delta\theta_5^{(5)}(0)=-\frac{eb}{6}n_0.
\]

The receiver form and phase derivatives of orders zero through
four agree. Reuse the receiver's exact
[signed work and full-nodal loss ledger](SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-receiver-port-passage),
`J_R=E_R+D_R`, rather than substituting a port-motion score.
At the preparation `q_5=-1`, and
`Delta q_5^{(4)}=-e*n_0/2`; hence

\[
\boxed{\begin{aligned}
\Delta J_R^{(5)}(0)&=\Delta D_R^{(5)}(0)=\frac{e^2}{3}n_0,\\
\Delta E_R^{(6)}(0)&=\frac{2e^3}{3}n_0.
\end{aligned}}
\]

In the stated half-weight model both displayed nonzero work and
storage coefficients equal `-sin(alpha)/(108*pi^3)`. Their
orders differ: the leading change in accumulated incoming work
is matched by receiver loss; retained storage first differs one
derivative later. This is neither a positive formation verdict
nor an assertion that more incoming work must be useful.

Analyticity makes these strict local differences an exact
counterexample to a receiver predictor retaining only the four
even coordinates plus `F,N` and initial spectral moments. Pure
odd silence cannot justify such a nonlinear source reduction.
The example supplies a closure obstruction, not a newly evaluated
trajectory or a preparation chosen to force barrier passage.

#### Every communicating tangent response stays below the receiver barrier

Let `theta_*` be the original donor-twist/flat-receiver phase,
and let `H_*` be its full phase Hessian: donor cycle edges have
weight `cos(alpha)>0`; receiver and bridge edges have weight one.
The complete constant tangent system for a real phase deviation
`eta`, with initial `eta=0`, is

\[
\dot x=-eKLx-aKH_*\eta,\qquad
\dot\eta=aKLx.
\]

There is no simultaneous-mode assumption. The matrices
`B=K^(1/2)LK^(1/2)` and `C_*=K^(1/2)H_*K^(1/2)`
need not commute. Their already proved support bounds suffice:

\[
\lambda_+(B)\ge1/44,\qquad 0\le C_*\le B,\qquad
\lambda_{\max}(C_*)\le3.
\]

First consider any initial form, and put
`F_t=x^T Lx/2`, `P_t=eta^T H_*eta/2`,
`D_t=e*(Lx)^T K(Lx)`. Direct differentiation gives

\[
\dot F_t=-D_t-\dot P_t,\qquad
\dot P_t=a(Lx)^\mathsf TK H_*\eta,\qquad
D_t\ge\gamma F_t,\quad \gamma=2e\lambda_+(B),
\]
\[
|\dot P_t|\le A\sqrt{D_tP_t},\qquad
A^2=\frac{2a^2\lambda_{\max}(C_*)}{e},\qquad
\frac A{\sqrt\gamma}
\le\frac{\sqrt{132}}\pi<4.
\]

The storage-angle argument now supplies an all-time bound. Where
`F_t,P_t>0`, set `phi=atan(sqrt(P_t/F_t))`, `k=D_t/F_t`
and `h=dot P_t/(2*sqrt(F_t*P_t))`. Then

\[
\dot\phi=h+\frac k2\sin\phi\cos\phi,\qquad
\frac{2h}{k}<4.
\]

For `G(phi)=(2/9)*(phi+sin(phi)*cos(phi))`,
`G'=4*cos(phi)^2/9`. It follows that
`(F_t+P_t)*exp(G(phi))` is nonincreasing, because

\[
\frac{d}{dt}\log[(F_t+P_t)e^{G(\phi)}]
\le k\cos^2\phi\left[-1+\frac49
                  \left(2+\frac14\right)\right]=0.
\]

The function `sin(phi)^2*exp(-G(phi))` increases to
`exp(-pi/9)` on `[0,pi/2]`. At zero-norm strata the same
continuous, locally absolutely continuous extension as in the
[full-state storage-angle proof](../../../nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-full-consensus-formation-obstruction)
applies; no zero form or zero phase state is excluded by division.
Consequently `P_t<=F_t(0)*exp(-pi/9)` for every future time.

For the present receiver observation this improves to a bound using
only `F_e`. The donor reflection commutes with `L,H_*` and `K`,
so the odd tangent solution has zero receiver coordinates for all
time. The full tangent receiver response equals the even-only one.
If `eta_R^tan` is its receiver phase deviation and `L_R` the
internal five-edge receiver Laplacian, then

\[
\boxed{\frac12\|L_R^{1/2}\eta_R^{\rm tan}(t)\|^2
\le F_e e^{-\pi/9}\le\frac34F_e\le3,
\qquad t\ge0.}
\]

The middle inequality is strict when `F_e>0`, since
`exp(pi/9)>1+1/3=4/3`. At `F_e=0` the receiver tangent
response is exactly zero. This is a uniform bound over the
original source budget, not a search for a favorable spectral
direction or a certificate for the nonlinear receiver.

#### A full nonlinear storage bound forces the donor barrier first

The same angle method also gives a distinct result for the actual
nonlinear field, provided its nonzero initial donor phase storage is
retained. This step does not replace that storage by a tangent cost.
For the full unit phase potential `V`, nodal Cauchy--Schwarz and
`sin(u)^2<=2*(1-cos(u))` give globally

\[
\begin{aligned}
S^\mathsf TKS
&\le\sum_i\nu_i\sum_{j\sim i}\sin^2(\theta_j-\theta_i)\\
&\le4\max_i\nu_i\,V=6V.
\end{aligned}
\]

Therefore `dot E=-D`, `D>=gamma*F(t)` and
`|dot V|<=sqrt(6*a^2/e)*sqrt(D*V)` obey precisely the
preceding angle inequalities, now with `P_t` replaced by `V(t)`.
No acute phase condition, simultaneous modal basis or linearization
is used. Initially `V(0)=V_5`; for initial form storage `F>0`
the resulting all-time bound is

\[
\boxed{V(t)\le\mathcal B(F)
:=(F+V_5)\exp\!\left[
G\!\left(\arctan\sqrt{V_5/F}\right)-\frac\pi9\right].}
\]

At `F=0` the source is the exact stationary donor-only state,
consistent with the continuous value `mathcal B(0)=V_5`.
This bound increases with the supplied form budget, since

\[
\frac{d}{dF}\log\mathcal B(F)
=\frac1{F+V_5}-\frac{2\sqrt{FV_5}}{9(F+V_5)^2}>0.
\]

A rational majorant at `F=4` suffices for the entire class. The
exact inequalities `559/250<sqrt(5)<56/25` imply
`69/20<V_5<691/200`. Thus

\[
\phi_4=\arctan\sqrt{V_5/4}
<\arctan\frac{93}{100}
=\frac\pi4-\arctan\frac7{193}
<\frac{11}{14}-\frac7{193}+\frac13\left(\frac7{193}\right)^3
<\frac34.
\]

Here `pi<22/7` and the alternating lower bound
`atan(t)>t-t^3/3` are sufficient. Hence `G(phi_4)<5/18`.
Using `pi>157/50` and the positive exponential series gives

\[
\frac\pi9-G(\phi_4)>\frac{16}{225},\qquad
e^{16/225}>1+\frac{16}{225}+\frac12\left(\frac{16}{225}\right)^2
=\frac{54353}{50625}.
\]

Consequently the following bound is rigorous without a trajectory
or a numerical optimization:

\[
\boxed{V(t)\le\mathcal B(F)\le\mathcal B(4)
<\frac{3019275}{434824}<\frac{139}{20}
<V_5+\frac72,\qquad F\le4,\ t\ge0.}
\]

Suppose the receiver reaches its first potential barrier at `tau_R`.
All other edge potentials are nonnegative, so at that instant

\[
\boxed{V_D(\tau_R)<\frac{139}{20}-\frac72=\frac{69}{20}<V_5.}
\]

Before crossing its own `7/2` barrier, the donor remains in its
initial twist component, where `V_D>=V_5`. It must therefore
have crossed that barrier **strictly before** `tau_R`, for the
entire original budget `F<=4`. This extends the earlier
energy-only order restriction beyond `F<=7/2`. It also excludes
simultaneous donor and receiver barrier states, which would require
`V>=7`. The required donor phase-storage release at receiver passage
is strictly greater than

\[
V_5-\frac{69}{20}=\frac{56-25\sqrt5}{20}>0.
\]

This is a structural restriction on any successful receiver-directed
mechanism: it must use a path that first escapes the donor well and
releases donor phase storage. It neither predicts a passage time nor
excludes sequential receiver acquisition. Potential-well departure
is not an assertion about the donor's instantaneous wrapped winding
or its eventual equilibrium.

#### A receiver passage needs a nonzero, quantified nonlinear contribution

Let `eta_R^full` be the receiver's continuous phase lift from its
initial flat phase under the actual law. At any actual first
potential barrier `V_R=7/2`, the global inequality `1-cos u<=u^2/2`
and the triangle inequality imply

\[
\boxed{\|L_R^{1/2}(\eta_R^{\rm full}-\eta_R^{\rm tan})\|
\ge\sqrt7-\sqrt{2F_e e^{-\pi/9}}
>\sqrt7-\sqrt6.}
\]

This uses the same initial state and clock for the full and tangent
solutions. Removing a common phase origin does not change the norm.
It does not equate primitive phase with an inferred regional angle.
For `F_e=0`, exact nonlinear silence already excludes the passage;
the conditional inequality remains consistent with that control.

The required correction has an exact source. Define, on the full
graph and the retained real lift,

\[
R(\eta)=S(\theta_*+\eta)+H_*\eta,
\qquad
\mathcal A_*=
\begin{pmatrix}-eKL&-aKH_*\\aKL&0\end{pmatrix}.
\]

With `z=(x,eta)` and matching initial conditions, variation of
constants gives

\[
\boxed{z^{\rm full}(t)-z^{\rm tan}(t)
=a\int_0^t e^{\mathcal A_*(t-s)}
       \binom{K R(\eta^{\rm full}(s))}{0}\,ds.}
\]

Every edge term of `R` is the actual sine remainder
`sin(alpha_ij+u)-sin(alpha_ij)-cos(alpha_ij)*u`, whose magnitude
is at most `u^2/2`. It contains the mixed source mechanism above
and possible release of donor phase storage. It is not an added
input or a freely selected receiver waveform.

Thus no improved tangent source direction within `F<=4` can by
itself explain receiver barrier passage. A positive route must
justify this finite nonlinear conversion; a negative route must
bound the actual convolution or receiver work below its required
gap. The current bound does neither for all mixed preparations.
No new reserved source, basin verdict, loss law or numerical
response follows from these source and closure results alone.

The existing [formation owner](../../../../src/tnfr/physics/relational_sine_formation.py)
exposes `source.receiver_excitation()` as `SineReceiverExcitation`.
It reconstructs both full source components and their actual form
storage and dissipative norms before admitting the separate fixed-law
tangent result. Its rational ceiling is `3*F_e/4`, with strictness
only for positive `F_e`. The
`necessary_nonlinear_correction_norm_bounds` field encloses the
analytic threshold `sqrt(7)-sqrt(3*F_e/2)`, not the realized
nonlinear correction. A positive lower endpoint supplies a conservative
necessary condition; an unsupported law or unresolved threshold supplies
no new response verdict. Odd coordinates remain present in the report.
The separately admitted `full_phase_budget_status` requires the
original complete law and total `F<=4`. It exposes
`actual_full_phase_storage_upper_bound=139/20`, the conditional
`donor_potential_barrier_first_required` and an outward lower bound
on the necessary donor phase-potential decrease. The latter is not
identified incoming receiver work or a reusable reserve: released
potential can dissipate or remain elsewhere in the system. Outside
that law or budget these separate fields stay unavailable, without
discarding the geometric decomposition or changing the tangent scope.
The [independent excitation controls](../../../../tests/physics/test_relational_sine_receiver_excitation.py)
check the complete-support parity, energy-angle and nonlinear-mixture
identities; the [formation controls](../../../../tests/physics/test_relational_sine_formation.py)
retain source admission and report boundaries. None evaluates a new
trajectory or turns the tangent ceiling into a nonlinear upper bound.

<a id="sine-weighted-receiver-exclusion"></a>
#### A weighted nonlinear proof function excludes receiver identity below a source threshold

Keep the same complete half-weight sine law, unit storage scale, held
capacities, initial donor twist and initially flat receiver. All six source
coordinates remain free. The two bridges are cut edges, which supplies
more regional information than a bound on total phase storage alone.

Let `chi_D,chi_R` be the indicator columns of the two rings and let
`e_0,e_5` be the corresponding port coordinate columns. Define the
constant maps

\[
T_D=\operatorname{diag}(\chi_D)-e_0\chi_D^{\mathsf T},\qquad
T_R=\operatorname{diag}(\chi_R)-e_5\chi_R^{\mathsf T},\qquad
T_B=I-T_D-T_R.
\]

The full sine current satisfies `sum_i S_i=0`. Internal edge currents
cancel in each regional sum, so `chi_R^T S=sin(theta_10-theta_5)`;
the corresponding donor identity uses port zero. Consequently `T_R S`
is precisely the receiver's internal sine-current vector, embedded in
all eleven coordinates. Likewise `T_D S` is the donor internal current
and `T_B S` is the two-bridge current. Thus, for the internal ring
potentials and total bridge potential `V_B`,

\[
\nabla V_D=-T_DS,\qquad \nabla V_R=-T_RS,\qquad
\nabla V_B=-T_BS.
\]

These identities retain the full nodal current; they introduce neither
an independent receiver input nor a removed intermediary. They depend
on this supplied cut-edge geometry, not on arbitrary graph partitions.

Write `a=1/(2*pi)`, `q=Lx`, and define

\[
\boxed{\mathcal U
=\mathcal F+V_D+\frac75V_R+\frac76V_B
 -\frac13q^{\mathsf T}KS,\qquad
\mathcal F=\frac12x^{\mathsf T}Lx.}
\]

The unequal weights and cross term belong to an auxiliary proof
function. They are not storage allocations or additional coefficients
in the nodal dynamics. Put
`M=T_D+(7/5)*T_R+(7/6)*T_B` and `A=KLK`. The actual rows give

\[
\dot q=-\frac12LKq+aLKS,\qquad
\dot S=-aH(\theta)Kq,\qquad H(\theta)\preceq L.
\]

Direct differentiation, with no phase linearization, therefore yields

\[
\dot{\mathcal U}\le
-\binom{q}{S}^{\!\mathsf T}Q(a)\binom{q}{S},
\qquad
Q(a)=\begin{pmatrix}
\frac12K-\frac a3A &
-\frac12\left[aK(I-M)+\frac16A\right]\\
-\frac12\left[a(I-M)^{\mathsf T}K+\frac16A\right] &
\frac a3A
\end{pmatrix}.
\]

Here `q,S` both have zero ordinary sum. Testing unrelated constant
vectors in this quadratic form would impose a condition on states that
the full nodal law never supplies. Instead use the exact rational basis

\[
P=\begin{pmatrix}I_{10}\\-\mathbf1_{10}^{\mathsf T}\end{pmatrix},
\qquad Z=\operatorname{diag}(P,P),\qquad
Q_*(a)=Z^{\mathsf T}Q(a)Z.
\]

The original capacities fix
`K=diag(1/3,1/2,1/2,1/2,1/2,1/3,3/4,1/2,1/2,1/4,1/2)`.
Thus both matrices `Q_*(7/44)` and `Q_*(25/157)` are rational
twenty-dimensional matrices determined by the displayed formulas and
the twelve unit edges. Exact LDL elimination without pivoting gives
twenty positive diagonal pivots at each endpoint, each strictly larger
than `1/10000`. This claim is reproducible using the recurrence

\[
A^{(0)}=Q_*(a),\qquad d_j=A^{(j)}_{jj},\qquad
A^{(j+1)}_{rs}=A^{(j)}_{rs}
 -\frac{A^{(j)}_{rj}A^{(j)}_{js}}{d_j},\quad r,s>j.
\]

Only exact rational arithmetic is needed for these forty pivot signs.
The shared
[exact matrix owner](../../../../src/tnfr/mathematics/_exact_linear_algebra.py)
implements the same test. Since `157/50<pi<22/7`, the actual `a`
lies strictly between the two rational endpoints. The matrix `Q_*(a)`
is affine in `a`; convexity of the positive-definite cone proves
`Q_*(a)>0` for that entire interval. The finite rational certificate
therefore proves the all-state inequality for the actual transcendental
coefficient, rather than approximating a trajectory or inferring a
spectral sign from a floating-point eigensolver. In particular,

\[
\boxed{\dot{\mathcal U}<0\quad\text{whenever }(q,S)\ne(0,0).}
\]

At the declared preparation, both bridges are aligned, the receiver is
flat and `S=0`. Hence `U(0)=F+V_5`. At any relative equilibrium
the common form makes `F(t)=0`, and `q=S=0`; thus
`U_infinity=V_D+(7/5)*V_R+(7/6)*V_B`. The complete
[equilibrium catalog](SINE_RECEIVER_TRANSFER_AND_CAPTURE.md#sine-eleven-node-asymptotic-equilibria)
has zero sine current on each bridge and each isolated ring. Every
nonflat receiver critical geometry has `V_R>=V_5`, and every other
term in this limiting weighted potential is nonnegative. It follows
that a nonflat receiver limit requires `U_infinity>=7*V_5/5`.

The already proved full-state convergence theorem applies unchanged.
For a nonzero source, `q(0)!=0`, so strict decrease over an initial
time interval gives `U_infinity<U(0)`. Therefore

\[
\boxed{F\le F_*:=\frac25V_5=\frac{5-\sqrt5}{2}
\quad\Longrightarrow\quad
\text{the receiver converges to relative phase consensus}.}
\]

At `F=0` the initial donor-only equilibrium is stationary, so the same
conclusion holds. Strict initial decrease includes the closed threshold
for real preparations; a finite rational source cannot equal this
irrational threshold exactly. Its admission can nevertheless use only
rational comparisons: with `z=5-2*F`, the condition is equivalent to
`z>=0` and `z*z>=5`.

This theorem treats all six source coordinates and their nonlinear
mixtures, without a time cutoff or assumed waveform. It determines the
receiver's limiting geometry, not the donor's endpoint, any intervening
wrapped winding, or the absence of a transient potential-barrier visit.
The cross term prevents interpreting `U` as a pointwise upper bound on
receiver phase storage. Above `F_*`, failure of this sufficient
certificate is not evidence of receiver acquisition. The original
`F<=4` accessibility problem remains open on that unresolved range.

The mathematical argument uses the fixed initial phase geometry and total
form storage, not zero receiver form: it also holds for arbitrary initial
forms on the eleven nodes with those same phases and capacities. The public
reader retains the declared six-coordinate preparation contract; this wider
theorem scope does not install a new preparation or response campaign.

The shared [formation owner](../../../../src/tnfr/physics/relational_sine_formation.py)
exposes this separate result through `source.receiver_localization()`
as `SineReceiverLocalization`. It validates the same complete law and
preparation before applying the exact algebraic threshold. The existing
receiver-transfer reader retains that report and uses a certified
localization as an exclusion of maintained receiver acquisition. The
[localization controls](../../../../tests/physics/test_relational_sine_receiver_localization.py)
independently reconstruct the cut-current maps and rational derivative
matrices, verify their LDL signs and check public scope boundaries.
No trajectory or source search is part of this certificate.

### One nonuniform donor preparation at the same storage budget

Consider the exact rational preparation

\[
D=\frac13(10,13,14,14,13),\qquad H=\frac43,
\qquad \delta=\frac12.
\]

It is donor-reflection-even, lies outside the silent subspace,
and has

\[
\begin{aligned}
F&=4,\\
q&=(0,2/3,1/3,1/3,2/3,-4/3,0,0,0,0,-2/3),\\
N&=\frac{37}{27},\qquad
m_1=\frac{43}{54},\qquad m_2=\frac{47}{54},\\
R&=\frac{43}{74},\qquad
\gamma^2=\frac{47}{74}.
\end{aligned}
\]

Its ratio `N/F=37/108<2/3` disproves extension of the
constant-donor lower bound `N>=2F/3` to arbitrary donor
shape. It is not asserted to minimize any quadratic form.
It also does not retain the same initial receiver drive as the
balanced `F=4,H=2` preparation: its first receiver rates
and odd acceleration have two-thirds of that magnitude.
Less initial loss is not an equal-response improvement here.

The short-window directional certificate can be inconclusive
for this preparation. That is not evidence of formation or of an
interesting surviving basin. The extended rational majorant in
Section 21 resolves the same preparation without evaluating a
trajectory.

For `delta=1/2`, use `lambda=3`. Since
`gamma<4/5` and `a=b<1/6`, choose the supplied horizon
`tau=6/5`. It satisfies `omega*tau<3/5`. The
factorial estimate gives
`cosh(3/5)<=50/41<5/4`, so the polynomial coefficient
can be bounded by

\[
c\le\frac58\bar\gamma\,\lambda ab<\frac1{24}.
\]

With `d=eR=43/148`, a conservative lower polynomial is
`g(t)=1-(43/148)t-t^2/24`. Its endpoint and exact
integrated loss satisfy

\[
g(6/5)=\frac{547}{925}>0,\qquad
E(0)-E(6/5)\ge
\frac{7564289}{13875000}>\frac{27}{50}.
\]

The joint target-entry boundary permits loss of only
`4-B_5<27/50`. One elementary verification of this strict
bound is `sqrt(2)>141/100`, which gives
`cos^2(3*pi/8)<59/400<(77/200)^2` and hence
`B_5>173/50`.

It remains necessary to rule out earlier entry. At this fixed
contrast the largest receiver-edge sum
`nu_i/sqrt(d_i)+nu_j/sqrt(d_j)` is
`5/(2*sqrt(2))`. With `E(0)=V_5+4<15/2`,
the global receiver-gap speed satisfies

\[
G<\frac{55}{48},\qquad
G\,\frac65<\frac{11}{8}<\frac{\pi}{2}.
\]

The receiver remains acute with winding zero until the storage
is already below the joint entry requirement. The nonuniform
preparation is therefore excluded from the two-twist target.
Its nonzero receiver response still occurs, but does not
establish formation. The decisive change was a sharper proof
window for the same law and same initial data.

### A class-level lower bound on any target-entry time

A separate action estimate applies to the whole six-coordinate
preparation class. Write `D_loss(t)=E(0)-E(t)` for accumulated
loss, distinguishing it from donor form. If a first joint acute
target entry occurs at time `T`, its boundary storage implies

\[
D_{\mathrm{loss}}(T)\le F-\beta B_5.
\]

If the right-hand side is nonpositive, the earlier strict-storage
obstruction already applies. Otherwise, the initially flat
receiver can acquire winding one only if some continuous lifted
receiver edge difference first reaches `+pi` or `-pi`
at a time `s<=T`. Before any such crossing, its wrapped
edge increments equal those continuous differences and their
sum around the cycle remains zero.

For `k_i=nu_i/d_i`, weighted Cauchy--Schwarz gives

\[
|\dot\theta_j-\dot\theta_i|^2
\le b^2(k_i+k_j)\,q^\mathsf TKq
=\frac{b^2(k_i+k_j)}e\,\dot D_{\mathrm{loss}}.
\]

Let `k_max` be the largest `k_i+k_j` over receiver
edges. Integrating to that necessary crossing and applying
Cauchy--Schwarz in time yields

\[
\pi^2
\le\frac{b^2k_{\max}}e\,sD_{\mathrm{loss}}(s)
\le\frac{b^2k_{\max}}e\,T(F-\beta B_5).
\]

Consequently every such entry must satisfy

\[
\boxed{
T\ge\frac{e\pi^2}{b^2k_{\max}(F-\beta B_5)}.}
\]

At the present default coefficients and `delta=1/2`,
`k_max=5/4`. For `F<=4` and positive allowable loss,
`B_5>173/50` and `pi>3` imply

\[
T>\frac{80\pi^4}{27}>240.
\]

This is a necessary time conditional on target entry, expressed
in the declared sine-model clock. It is neither a prediction
that entry occurs nor a global phase-speed bound for trajectories
that spend more than the permitted loss. The shared report's
`phase_action_bound` retains the allowable-loss interval, actual maximum
receiver-edge mobility and a lower bound on the necessary entry time.
It supplies no time when the positive allowance is unavailable.
An admitted horizon below that bound can exclude earlier joint entry;
excluding entry afterwards still requires its separately justified loss
deficit. Any successful path would have to preserve the small remaining
allowance for at least that long.

<a id="sine-maintained-target-obstruction"></a>
### A global auxiliary function excludes the maintained target for the whole class

The early-window bounds retain useful preparation-specific information,
but a joint full-field argument closes the stated six-coordinate class
without optimizing independent moment ranges or extending a trajectory
window. Keep exactly `delta=1/2`, `beta=1`, the supplied eleven-node
support, initial phase configuration and held capacities of this section.
For this auxiliary-function argument, allow positive effective coefficients
in the explicit sufficient domain

\[
e>0,\qquad w>0,\qquad 0<r:=\frac we<\frac32.
\]

This is an open neighborhood of the reference ratio `r=1`, not a sharp
formation threshold or an optimized parameter range. It changes only the
relative coefficients of the same complete normalized-sine law. The
earlier finite preparations, responses and default-clock time estimates
retain their original coefficients; they are not reevaluated here.
Let the instantaneous form storage be
`mathcal F(t)=x(t)^T L x(t)/2`, so the preparation's `F` is `mathcal F(0)`.
Write

\[
y=K^{1/2}q,\qquad \zeta=K^{1/2}S,\qquad
B=K^{1/2}LK^{1/2},\qquad C(\theta)=K^{1/2}H(\theta)K^{1/2},
\qquad a=b=\frac w\pi.
\]

These are the full eleven-node gradient variables from Section 21, not
a six-node evolution. Their exact complete rows are

\[
\dot y=-eBy+aB\zeta,\qquad \dot\zeta=-bC(\theta)y.
\]

For every phase state, the edge Hessian representation gives
`C(theta)<=B` in quadratic-form order, since each cosine is at most one.
No acute-phase, winding-preservation or positive-Hessian premise is needed.
Both `sum_i q_i=0` and `sum_i S_i=0`, so `y` and `zeta` are perpendicular
to `ker(B)=span(K^(-1/2)*1)` at all times. This excludes the zero mode
from the matrix estimate below without discarding a consumed coordinate.

The positive spectrum of `B` lies in a known rational interval. The
[full-support path bound](../../../nodal/SINE_PATTERN_RECOVERY.md#an-independent-full-graph-spectral-gap-bound)
gives `lambda_2(L)>=1/11`. Here `K>=I/4`, and the nonzero spectrum of
`B` equals that of `L^(1/2) K L^(1/2)>=L/4`. The min-max principle thus
gives `lambda_+(B)>=1/44`. The existing upper estimate
`B<=2*max_i(nu_i)*I` gives `lambda_max(B)<=3`. Hence

\[
\frac1{44}\le\lambda\le3
\quad\text{for every positive eigenvalue of }B.
\]

Now define an **auxiliary proof function**, not a new physical storage
or term in the dynamics:

\[
\boxed{\mathcal W= c\,\mathcal F+V(\theta)-h\,y^\mathsf T\zeta,
\qquad c=\frac45,\qquad h=\frac a{2e}=\frac r{2\pi}.}
\]

The coefficient `c` and the mixed coefficient `h` belong only to this
certificate. At the reference `e=w=1/2`, one has `h=a=1/(2*pi)`, so this
is the same function as the fixed-coefficient obstruction. The full form
and phase storage derivatives are
`mathcal F_dot=-e*||y||^2+a*y^T*zeta` and
`V_dot=-b*y^T*zeta`, with `a=b`. Differentiating `W` and then using
`C(theta)<=B` gives the all-state inequality

\[
\begin{aligned}
\dot{\mathcal W}
={}&-ce\|y\|^2+a(c-1)y^\mathsf T\zeta
 +he\,y^\mathsf TB\zeta-ha\,\zeta^\mathsf TB\zeta
 +hb\,y^\mathsf TC(\theta)y\\
\le{}&-ce\|y\|^2+a(c-1)y^\mathsf T\zeta
 +he\,y^\mathsf TB\zeta-ha\,\zeta^\mathsf TB\zeta
 +hb\,y^\mathsf TBy.
\end{aligned}
\]

Diagonalize the constant symmetric matrix `B` on its positive spectral
subspace and put `rho=a/e=r/pi`. The contribution of one mode to the last
line is

\[
-e\left(c-\frac{\rho^2\lambda}{2}\right)y_\lambda^2
+a\left(c-1+\frac\lambda2\right)y_\lambda\zeta_\lambda
-\frac{e\rho^2\lambda}{2}\zeta_\lambda^2.
\]

The admitted ratio and `pi>3` give `0<rho<1/2`. The positive magnitude
of its first negative term is therefore strictly greater than
`e*(4/5-3/8)=17*e/40`; the last negative term also has positive magnitude.
The determinant condition for strict negative definiteness is

\[
p(\lambda):=(1-c)^2-(c+1)\lambda
 +\left(\frac14+\rho^2\right)\lambda^2<0.
\]

Indeed, four times the product of those two magnitudes minus the square
of the mixed coefficient equals `-a^2*p(lambda)`. This
condition holds uniformly, using only rational bounds:

\[
p(\lambda)<\overline p(\lambda)
 :=\frac1{25}-\frac95\lambda+\frac12\lambda^2,
\]
\[
\overline p(1/44)=-\frac{63}{96800}<0,
\qquad\overline p(3)=-\frac{43}{50}<0.
\]

The polynomial `pbar` is convex, so it is negative throughout the
interval between those two endpoints. Every mode quadratic is therefore
strictly negative unless its two coordinates vanish. Consequently

\[
\boxed{\dot{\mathcal W}\le0,\qquad
\dot{\mathcal W}<0\quad\text{whenever }(q,S)\ne(0,0).}
\]

This is a global consequence of the same complete positive-loss law
throughout the admitted ratio domain. It does not freeze the sine term,
linearize the phase field or assume
the donor remains in its original winding sector. The exact initial
condition `S(0)=0` gives, for every admitted donor/intermediary form,

\[
\mathcal W(0)=\frac45F+V_5.
\]

At the specified maintained two-twist target, uniform form and exact
sine balance give `q=S=0`, while `V=2V_5`. Thus

\[
\mathcal W_{\rm target}=2V_5,\qquad
\mathcal W_{\rm target}-\mathcal W(0)=V_5-\frac45F.
\]

Whenever this last margin is positive, convergence is impossible:
`W` is continuous and invariant under common form and phase origins, so
convergence to the declared target modulo those origins would require
convergence to a larger `W` value, contradicting its nonincrease.
This sufficient criterion keeps `F` from the actual preparation; it does
not require an artificial cutoff in a reader applying the theorem.

For the whole declared research class `F<=4`, the margin satisfies

\[
V_5-\frac45F\ge V_5-\frac{16}{5}>\frac14>0.
\]

The last inequality follows from `sqrt(5)<56/25`, whose square is
strictly greater than five. Therefore **every preparation in the
six-coordinate class with `F<=4` is excluded from convergence to the
specified maintained two-twist target for every admitted ratio
`0<w/e<3/2`**. Neither the margin nor its strict positive lower bound
depends on that ratio: the mixed term vanishes at both endpoints. The
argument does not select a preferred ratio or imply formation outside
this sufficient domain.

In particular no such preparation can enter a recovery basin whose
valid full-state theorem guarantees that convergence under this unchanged
law. This is stronger than the earlier profile-by-profile loss checks
for the stated maintained target. It does not assert that the trajectory
can never visit the larger joint acute winding region: the cross term
and nonuniform form need not vanish at a transient visit. Nor does it
exclude receiver-only winding if the donor identity is lost. The action
bound above retains its separate necessary condition for acute-region
entry and must not be relabeled as a predicted entry time.

<a id="sine-formation-clock-covariance"></a>
### Constitutive ratio and constant clock changes are different operations

Changing `w/e` changes the relative contributions in the complete field.
By contrast, for a constant `kappa>0`, set `t'=kappa*t` and keep the
state coordinates, graph, held capacities, `K` and `beta` fixed. The same
state curve in the new clock satisfies

\[
\frac{dx}{dt'}=-e'Kq+a'KS,\qquad
\frac{d\theta}{dt'}=b'Kq,\qquad
(e',w',a',b')=\frac1\kappa(e,w,a,b).
\]

Both evolution rows change. Consequently `r'=r`, `h'=h`, and the
state function `W` is unchanged, with

\[
\frac{d\mathcal W}{dt'}=\frac1\kappa
\frac{d\mathcal W}{dt}\le0.
\]

The maintained-target margin and exclusion are thus independent of
this constant clock convention. Elapsed times and the phase-action
lower bound above transform by `T'=kappa*T`, since `e'/b'^2=kappa*e/b^2`.
The default numerical statement `T>240` belongs to its stated clock,
not to every rescaling of it. This proves neither a preferred physical
clock nor covariance under state-dependent or time-dependent clock
changes.

The implementation's
[`RelationalExchangeModel`](../../../../src/tnfr/dynamics/relational.py)
normalizes the supplied EPI and phase weights once and stores finite
effective coefficients. Its formation reader uses those stored values,
including their exact represented ratio. Commonly scaling raw constructor
weights therefore leaves the ideal normalized coefficient pair unchanged;
finite materialization is still governed by the shared admission contract.
It is **not** the clock change just derived. The arbitrary positive
coefficients in that derivation express the whole-field time scale outside
the normalized coefficient convention, without introducing a new runtime
parameter. Using another time unit also requires transforming the
integration step, horizons and all reported rates consistently.

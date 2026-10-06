# Retained phase records and finite contact readout

Native isolated-pattern recovery, retained phase offset, finite contact and removal budgets and the receiver mean record.

Part of [Native pattern reduction and memory](RELATIONAL_PATTERN_MEMORY.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 13. A retained collective phase offset after isolated-pattern recovery

<a id="relational-retained-phase-memory"></a>

An isolated ring can recover the same internal winding geometry while
retaining a different common phase relative to a separately declared
reference. This result concerns the selected relational law on a supplied
C5, not a new state variable or a physical interpretation of phase. It uses
the [isolated-ring capture theorem](RELATIONAL_EFFECTIVE_CONNECTIONS.md#relational-pattern-detachment)
and keeps the offset rows that the earlier relative-state memory studies
explicitly removed from their observations.

### State, reference and exact mean identities

Use the oriented unit cycle `i -> i+1` modulo 5, one common held capacity
`nu>0`, `e,w,beta>0`, the reference phase completion
`theta_dot=(nu*w/beta)*H^-1*Bx`, and no input or event during relaxation.
Set

\[
\kappa=2\pi/5,\qquad C=\cos\kappa>0,\qquad
\theta_i^*=i\kappa.
\]

Choose continuous lifts in the acute winding-one component and write

\[
x=\bar x\mathbf1+u,\qquad
\theta=\theta^*+m\mathbf1+v,\qquad
\mathbf1^Tu=\mathbf1^Tv=0.
\]

The scalar `m` is the arithmetic mean of the lifted deviation from this
reference, not an angle reconstructed from form. A second, untouched ring
at the same twist and common phase defines the comparison frame. Independently
resetting the two component phase frames would erase the chosen comparison
and is not part of the protocol. A common rotation of both rings leaves
the final relative offset unchanged.

Let `(Sz)_i=z_(i+1)`, `B=2I-S-S^-1`, and `J=S-S^-1`. Here `J` denotes the
skew cyclic difference, not the earlier full-state Jacobian. The actual
wrapped edge gaps are

\[
\delta_i=\kappa+v_{i+1}-v_i.
\]

On this acute branch the native relative resultant is

\[
z_i=e^{i\delta_i}+e^{-i\delta_{i-1}}
=2\cos\!\left(\frac{\delta_i+\delta_{i-1}}2\right)
 e^{i(\delta_i-\delta_{i-1})/2}.
\]

The cosine factor is positive and the displayed angle belongs to the
principal regular branch. Thus the phase source is **exactly linear** in
the centered phase deviation on this particular cycle component:

\[
g=-\frac1{2\pi}Bv,\qquad
\dot u=-aBu-bBv,\qquad
a=\frac{\nu e}{2},\quad b=\frac{\nu w}{2\pi},\qquad
\dot{\bar x}=0.
\]

This exact form-mean conservation uses both degree two and common capacity;
it is not imported into an irregular or heterogeneous-capacity graph.
The phase metric still contains nonlinear internal geometry:

\[
H_i=2\pi\cos\!\left(\kappa+\frac{(Jv)_i}{2}\right)
          \operatorname{sinc}\!\left(\frac{(Bv)_i}{2}\right),
\qquad
\dot m=\frac{\nu w}{5\beta}\sum_i\frac{(Bu)_i}{H_i}.
\]

Although `sum(H_i*theta_dot_i)=0` exactly, the weights depend on the
evolving phase geometry. That instantaneous weighted identity cannot be
integrated as conservation of `sum(theta_i)` or of `sum(H_i*theta_i)`.
The calculation below shows that the arithmetic phase mean actually changes.

### The first nonzero retained offset

Put `c=nu*w/(2*beta*pi*C)`. Expansion at the twist gives

\[
H_i^{-1}=\frac1{2\pi C}
 \left[1+\frac{\tan\kappa}{2}(Jv)_i+O(\|v\|^2)\right],
\]

and hence

\[
\dot m=\frac{c\tan\kappa}{10}(Bu)^TJv
        +O(\|u\|\,\|v\|^2).
\]

The term linear in form vanishes because `sum(Bu)=0`. For centered
preparations `u(0)=epsilon*u_0`, `v(0)=epsilon*v_0`, their first variations
obey

\[
\dot u_1=-aBu_1-bBv_1,\qquad \dot v_1=cBu_1.
\]

The commuting identities `B^T=B`, `J^T=-J`, `BJ=JB` imply

\[
\frac{d}{dt}(u_1^TJv_1)=-a\,u_1^TBJv_1.
\]

The other two terms are zero quadratic forms of the skew matrix `BJ`.
On every nonconstant Fourier mode the two-row tangent matrix has negative
trace `-a*lambda` and positive determinant `b*c*lambda^2`; all centered
first variations decay. Integrating the preceding identity therefore gives

\[
\int_0^\infty u_1^TBJv_1\,dt=\frac1a u_0^TJv_0.
\]

Consequently the limiting phase offset satisfies the conditional asymptotic
law

\[
\boxed{\quad
m_\infty-m(0)=
\frac{w\tan\kappa}{10\beta\pi e\cos\kappa}
\,\epsilon^2u_0^TJv_0+O(\epsilon^3).
\quad}
\]

This is an integrated nonlinear effect of the supplied nodal phase metric,
not a fitted trajectory or a new phase equation. Common capacity cancels
from its coefficient. In fact changing the common positive capacity
multiplies both evolution rows by the same factor, so it changes the
relaxation clock but not the limiting offset from a fixed preparation.
No claim uniform in `e -> 0`, or independent of the selected phase law,
follows.

For the negative-winding reference, use `kappa=-2*pi/5` throughout.
The sine and skew coefficient change sign; cosine, the positive metric
at the twist and the recovery barrier remain the same. This is the second
sector admitted by the shared coefficient helper, not another law.

The infinite-time remainder is justified by local stability, not by
integrating a fixed-time expansion without control. For fixed model
coefficients, the smooth quotient field and its Hurwitz linearization give
constants independent of sufficiently small `epsilon` such that
`||z(t,epsilon)||<=K*|epsilon|*exp(-lambda*t)` and
`z(t,epsilon)=epsilon*z_1(t)+O(epsilon^2*exp(-lambda_1*t))`, with positive
decay exponents. Variation of constants supplies the second bound after
shrinking the neighborhood if needed. The quadratic mean-rate expression
then differs from its first-variation value by an integrable
`O(epsilon^3)` bound; its cubic remainder is integrable as well. This proves
the displayed asymptotic coefficient and finite phase limit. This general
derivation supplies no numerical remainder constant at a chosen nonzero
amplitude; [Section 14](#relational-finite-phase-memory) closes that separate
obligation for one fixed pair and model.

This result does not exclude invariants involving the internal coordinates.
Indeed, the limiting phase defines a local asymptotic-phase projection that
is constant along each recovering trajectory. Its quadratic expansion is
the expression above, including the internal skew product. What fails to
be conserved is the bare arithmetic phase mean, not every possible
state-dependent phase coordinate.

<a id="asymptotic-origin-coordinate-scope"></a>

The general local mechanism clarifies the distinction between a drifting
mean and a conserved origin coordinate. In a declared smooth relative chart
write `y'=f(y)` and `m'=h(y)`, with `h(0)=0`. Assume a forward-invariant
neighborhood on which the relative flow `Phi_t` and its initial-state
derivative satisfy uniform exponential decay bounds

\[
\|\Phi_t(y)\|\le C e^{-\lambda t}\|y\|,\qquad
\|D_y\Phi_t(y)\|\le C e^{-\lambda t},\qquad \lambda>0,
\]

and let `h` be `C^1` with bounded derivative there. These are local recovery
and regularity hypotheses, not consequences of common-origin symmetry
alone. They make the following integral and its first derivative converge
uniformly on smaller neighborhoods:

\[
\psi(y)=\int_0^\infty h(\Phi_t(y))\,dt,\qquad
D\psi(y)f(y)=-h(y),\qquad I=m+\psi(y),\qquad \dot I=0.
\]

The derivative identity follows by shifting the lower integration limit
along the flow. The map `(y,m)->(y,m+psi(y))` is an invertible local chart,
and `I` is the limiting origin of the recovering trajectory. Conversely,
such a time-independent triangular chart removes mean drift only if its
correction solves that derivative equation. On any larger proposed domain,
a closed relative orbit of period `T` necessarily satisfies
`integral_0^T h(Phi_t(y)) dt=0` for a single-valued correction to exist.
The local recovery result does not supply a global chart through other
basins or recurrent relative motion.

For an exact scalar control, `y'=-lambda*y`, `m'=c*y^2`, `lambda>0`, has
`psi=c*y^2/(2*lambda)`: its bare mean drifts while this nonlinear origin is
constant. The [static controls](../../tests/physics/test_relational_environment_response.py)
verify both transformed rows without evolving a graph. This construction
depends on the selected complete law and the retained relative state;
eliminated coordinates can carry information needed by `psi`. It is neither
the bare linear form charge `Q` nor, by default, a primitive-local chart or
an independently justified constitutive law. Circular phase origins retain
their chosen lift and local domain throughout.

### Two equal-mean preparations with opposite retained offsets

For a concrete Fourier pair, use

\[
u_{0,i}^{\pm}=\pm\cos(i\kappa),\qquad
v_{0,i}=\sin(i\kappa).
\]

Choose one shared origin `bar(x)=m(0)=0` for both rings; any common initial
offsets are recovered by the exact shift symmetries. This choice does not
recenter the components independently after they evolve.

Both have zero means, the same initial primitive phases and winding, and
identical form/phase storage. Since
`J*sin(i*kappa)=2*sin(kappa)*cos(i*kappa)` and
`sum(cos(i*kappa)^2)=5/2`, the preceding result becomes

\[
m_\infty^{\pm}-m(0)
=\pm A\epsilon^2+O(\epsilon^3),\qquad
A=\frac{w\tan^2\kappa}{2\beta\pi e}>0.
\]

The signs are nonzero and opposite for all sufficiently small positive
amplitudes. There is also an exact comparison symmetry: the map
`(x,theta) -> (-R*x,-R*theta)`, where `(Rz)_i=z_(-i)`, preserves the
positive-winding reference modulo fixed nodewise turns. It maps the positive
form preparation to the negative one while leaving the sine phase
preparation unchanged. Relabeling and simultaneous form/phase reversal are
symmetries of this law with common capacity. With `m(0)=0`, uniqueness and
the same continuous lift give `m_infinity^-=-m_infinity^+` exactly within
their admitted recovery family.

Capture can be admitted separately from the offset's asymptotic error. For
general centered `u_0,v_0`, it is sufficient that

\[
|\epsilon|\max_i|v_{0,i+1}-v_{0,i}|<\pi/10,
\qquad
\frac{\epsilon^2}{2}
\left(u_0^TBu_0+\beta v_0^TBv_0\right)
<\beta(V_{\rm face}-V_*).
\]

The first condition retains strict acute gaps. The second bounds storage
above its critical twist value, using the zero first variation and
`abs(cos(delta))<=1` in the second variation; it places the state inside
the isolated-ring basin. For the displayed Fourier pair, `beta=1` and
`|epsilon|<=1/32` suffice: the storage excess is at most `5*epsilon^2`,
while

\[
V_{\rm face}-V_*=5\cos(2\pi/5)-4\cos(3\pi/8)>\frac{13}{1000}.
\]

For example, the exact radical formulas give
`cos(2*pi/5)>309/1000` and `cos(3*pi/8)<383/1000`, proving this lower
bound. The gap perturbation is at most `2*|epsilon|<pi/10`.
These explicit basin conditions prove recovery; they do not certify the
size or sign of the truncated offset formula at `epsilon=1/32`.

### A signed operational readout, separate from attachment work

An isolated component's common phase is not an intrinsic gauge-independent
memory label. Keep the untouched reference ring, the inherited common frame
and a declared matching port on each ring. After ideal recovery, the two
forms have the same conserved mean and their relative port phase is
`delta=m_infinity-m(0)`, on the small continuous branch. A hypothetical
unit attachment at those ports has storage cost

\[
\Delta E_{\rm attach}=\beta[1-\cos\delta]
=\frac{\beta A^2}{2}\epsilon^4+O(\epsilon^5).
\]

That cost is even in `delta`. In particular it is exactly equal for the
opposite-offset preparations and cannot distinguish their signs.

The fresh signed port response does distinguish them. Put the recovered
pattern on the left, with phase offset `delta`, and the untouched reference
on the right. Their relative port resultants after the hypothetical
attachment are `2*C+exp(-i*delta)` and `2*C+exp(i*delta)`. All form
gradients are zero at this prepared instant. Therefore

\[
\begin{aligned}
g_L&=\pi^{-1}\operatorname{Arg}(2C+e^{-i\delta})
    =-\frac{\delta}{\pi(1+2C)}+O(\delta^3),& g_R&=-g_L,\\
\dot x_L&=\nu w g_L
    =\mp\frac{\nu w A}{\pi(1+2C)}\epsilon^2+O(\epsilon^3),&
\dot x_R&=-\dot x_L.
\end{aligned}
\]

The immediate phase rates remain zero because `q=0`; the form response is
the nodal phase-pressure channel. The already established
[attachment observation](RELATIONAL_SUPPORT_EVENTS.md#represented-admission-and-shared-implementation)
evaluates such candidate state/support data without executing an edge event.
An actual attachment with nonzero cost still requires declared work or
another admitted state/event budget. Its occurrence and timing are not
derived by this readout.

The retained information is thus a relation between the recovered pattern
and the declared reference, accessible through a signed port observation
even when winding, internal geometry, mean form and attachment cost agree.
It is not an absolute phase, a selected physical clock, a complete coarse
state, or a demonstration of physical memory or spin. Preparation, reference,
selected law and readout remain explicit. No finite-amplitude trajectory,
finite-time completion of relaxation or numerical asymptotic-error bound
has been evaluated by this general derivation. The following fixed-case
proof supplies a finite-amplitude bound without evaluating a trajectory.

The shared
[`bound_relational_cycle_memory`](../../src/tnfr/physics/relational_cycle_memory.py)
kernel retains exact centered preparation directions, the sufficient
capture-amplitude family and the bounded leading coefficient as separate
evidence. Its `RelationalCycleMemoryBounds` report does not assign an
uncomputed finite-amplitude remainder or evolve a graph. The
[coefficient controls](../../tests/test_relational_cycle_memory.py) include
the exact tangent skew integral and the direction/basin contracts. The
[mechanism controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
check the native mean-rate cross term and the existing hypothetical
attachment's signed response versus even storage cost.

## 14. A finite-amplitude retained-phase certificate without a trajectory

<a id="relational-finite-phase-memory"></a>

Fix `e=w=1/2`, `beta=nu=1`, the isolated winding-one C5 and untouched
reference of Section 13. The two supplied centered directions are now the
exact rational vectors

\[
U=(1,-1,0,0,0),\qquad V=(0,1,-1,0,0),\qquad
u(0)=\pm\epsilon U,\quad v(0)=\epsilon V,\quad m(0)=0.
\]

They have identical initial means, phases and storage, and
`U^T*J*V=2`. They are **not** the Fourier mirror pair from Section 13;
no exact opposite-offset or equal-attachment-cost symmetry is assumed for
their subsequent responses. The following prior inequalities bound the
entire infinite recovery. A dyadic amplitude is selected only after those
inequalities have been established, without sampling either response.

### Storage controls the state and the integrated form response

Let `z=(u,v)` have the Euclidean norm and use the ball of radius
`r=1/100`. In this ball every edge differs from its twist gap by at most
`2*r`; thus its cosine exceeds `C-2*r>7/25`, with
`3/10<C=cos(kappa)<1/2`. The centered C5 Laplacian satisfies
`lambda_min(B)>1`, `||B||<=4`, and `||J||<=2`. Taylor's integral formula
at the critical twist consequently gives

\[
\mathcal E:=E-V_*\ge\frac18\|z\|^2.
\]

For the specified directions, `U^T*B*U=V^T*B*V=6`, so the initial
excess obeys `mathcal E(0)<=6*epsilon^2`. If
`0<epsilon<=1/1024`, this is strictly below `r^2/8`; the initial state
is inside the ball. Storage nonincrease prevents a first exit and yields

\[
\sup_{t\ge0}\|z(t)\|\le7\epsilon,\qquad
\dot{\mathcal E}=-\frac14\|Bu\|^2,
\qquad
\int_0^\infty\|u\|^2dt\le24\epsilon^2.
\]

The ball stays strictly acute and regular. The isolated-ring capture theorem
gives `u,v -> 0`; no quantitative exponential constant is needed in the
estimates below. Common offsets are retained as in Section 13.

### A cross term controls the integrated phase deformation

Let `B^+` be the inverse on the centered subspace, extended by zero on
constants, and let `Pi` be the orthogonal centering projection. Write

\[
a=\tfrac14,\qquad b=\frac1{4\pi}>\tfrac1{16},\qquad
M=\operatorname{diag}\left(\frac1{2H_i}\right),\qquad
\dot v=\Pi MBu.
\]

The explicit two-neighbor metric gives `0<M_i<1/2` throughout the ball.
For `F=u^T*B^+*v`, differentiation of the actual rows gives

\[
\dot F=-a\,u^Tv-b\|v\|^2+u^TB^+\Pi MBu
\le-\frac1{32}\|v\|^2+\frac52\|u\|^2.
\]

Here `||B^+||<1` bounds the last term by `2*||u||^2`, and
`|u^T*v|/4<=||u||^2/2+||v||^2/32` supplies the remaining constants.
Since `|F(0)|<=2*epsilon^2` and capture gives `F(infinity)=0`, integration
and Cauchy--Schwarz imply

\[
\int_0^\infty\|v\|^2dt\le1984\epsilon^2<2048\epsilon^2,
\qquad
\int_0^\infty\|u\|\,\|v\|dt\le224\epsilon^2.
\]

These are whole-recovery integral estimates. They are neither sampled
losses nor a claim that capacity, storage or a phase mean can be omitted.

### An integrated cubic remainder for the approximate phase invariant

Put `d=1/(4*pi*C)<1/3`, `tau=tan(kappa)<10/3`, and
`K=d*tau/(10*a)<1/2`. For each node define

\[
t_i=(Jv)_i/2,\qquad s_i=(Bv)_i/2,\qquad
h_i=\frac{C}{\cos(\kappa+t_i)\operatorname{sinc}s_i}.
\]

The denominator after dividing by `C` is
`D_i=(cos(t_i)-tau*sin(t_i))*sinc(s_i)`. Since
`|t_i|<=||v||<=1/100` and `|s_i|<=2*||v||`, the elementary sine/cosine
remainders give

\[
D_i>\frac9{10},\qquad
D_i=1-\tau t_i+d_{2,i},\quad |d_{2,i}|\le2\|v\|^2.
\]

For example, the three contributions to `d_2` are `cos(t)-1`,
`tau*(t-sin(t))` and `(cos(t)-tau*sin(t))*(sinc(s)-1)`; the bounds
`t^2/2`, `tau*|t|^3/6` and `(1+tau*|t|)*s^2/6` suffice.
Dividing by the positive `D_i` now proves

\[
|h_i-1|\le4\|v\|,\qquad
|h_i-1-\tau t_i|\le17\|v\|^2.
\]

These are bounds on the actual inverse metric, including its zero-source
limit, rather than a substituted polynomial phase law.

Use the quadratic approximation to the asymptotic phase,

\[
I_2=m+K u^TJv.
\]

The exact linear form row and the quadratic mean-rate term cancel in
`I_2_dot`. The remaining mean-rate term is bounded by
`14*||u||*||v||^2`; the nonlinear correction in
`K*u^T*J*v_dot` is bounded by `6*||u||^2*||v||`. These follow directly
from the two inverse-metric inequalities, `||B||<=4`, `||J||<=2`,
`d<1/3`, `K<1/2` and `||Pi||=1`. Consequently

\[
\begin{aligned}
|\dot I_2|&\le14\|u\|\|v\|^2+6\|u\|^2\|v\|,\\
\left|m_\infty-I_2(0)\right|
&\le\left(14\cdot7\cdot224+6\cdot7\cdot24\right)\epsilon^3\\
&=22960\epsilon^3<2^{15}\epsilon^3.
\end{aligned}
\]

The limiting skew product is zero because both centered coordinates decay.
Thus this is an explicit error bound on the limiting phase itself, obtained
without an infinite-time numerical integration or an assumed decay rate.

### One declared nonzero amplitude and its signed readout

Select

\[
\epsilon=2^{-20}
\]

after the preceding uniform proof. It is inside the stated trapping family.
Let

\[
\mathcal C=2K=\frac{\sin\kappa}{5\pi\cos^2\kappa}.
\]

Certified trigonometric bounds give `3/5<mathcal C<1`. For instance,
`cos(kappa)<31/100` implies `cos(kappa)^2<1/10` and
`sin(kappa)>19/20`; together with `pi<22/7`, these give
`mathcal C>133/220>3/5`. The prior remainder bound now yields

\[
\boxed{\qquad
\left|m_\infty^\pm\mp\mathcal C\,2^{-40}\right|<2^{-45}.
\qquad}
\]

In particular,

\[
m_\infty^+>\frac{91}{160}\,2^{-40}>0,\qquad
m_\infty^-<-\frac{91}{160}\,2^{-40}<0,\qquad
|m_\infty^\pm|<\frac{33}{32}\,2^{-40}<\frac14.
\]

These symmetric **enclosures** do not assert that the two actual offsets
are exact negatives. They prove nonzero offsets of opposite signs for the
two finite, rationally directed preparations.

For either offset `delta`, the same hypothetical matching-port attachment
to the untouched reference has

\[
\dot x_L=\frac1{2\pi}\operatorname{Arg}(2C+e^{-i\delta}),
\qquad \dot x_R=-\dot x_L.
\]

Because `|delta|<1/4` and `2C+cos(delta)>0`, the left response has the
opposite sign to `delta` and cannot be zero. Interval evaluation of this
same expression preserves the two separated readouts. The nonnegative
cost `1-cos(delta)` remains even; unlike the Fourier mirror case, equal
costs for these rational preparations have not been proved. Any actual
attachment still needs its independently declared event budget and timing.

The shared
[`certify_relational_cycle_memory`](../../src/tnfr/physics/relational_cycle_memory.py)
returns this fixed-case `RelationalFiniteMemoryCertificate`, with capture,
integrated-remainder and signed readout bounds kept separate. The
[finite-memory controls](../../tests/test_relational_finite_memory.py) check
admission and the nonlinear interval consequences. The separate
[exact proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
verify the derivative cancellation, integral budgets and centered inverse
without advancing a trajectory. This does not replace the general coefficient report's
unavailable remainder by a bound for arbitrary models or directions.

The result is an analytic finite-amplitude distinction in the limiting
relative state. It supplies no finite recovery duration or finite-time
readout guarantee, and does not identify physical memory, spin or a unique
law of nature. The small amplitude is a conservative mathematical witness,
not a fitted physical scale or an optimal robustness radius.

## 15. A finite-time contact readout with residual-state and event budgets

<a id="relational-finite-time-memory"></a>

Retain exactly the law, two rational preparation directions, untouched
reference and amplitude `epsilon=2^-20` of Section 14. The clock is the
declared structural clock with `nu=1`; the numbers below are not laboratory
seconds. The proposed contact joins position zero of the recovering left
ring to position zero of the untouched right ring, preserving both states.
No edge is executed in this calculation.

Convergence alone already implies that sufficiently late readouts have the
limiting signs. Here the additional result is a prior quantitative time and
a full-state error budget. In particular, the remaining form deformation is
not set to zero, and the finite-time phase is not replaced by its limit.

### An explicit decay estimate on the already admitted ball

Use the same centered state `z=(u,v)`, storage excess `mathcal E` and cross
term `F=u^T*B^+*v` as in Section 14. In the trapped radius-`1/100` ball,
Taylor's formula and `||B||<=4` give

\[
\frac18\|z\|^2\le\mathcal E\le2\|z\|^2,
\qquad |F|\le\frac12\|z\|^2.
\]

The auxiliary proof functional

\[
\mathcal H=\mathcal E+\frac1{32}F
\]

is not a new physical storage or a change to the dynamics. It satisfies

\[
\begin{aligned}
\frac7{64}\|z\|^2&\le\mathcal H\le\frac{129}{64}\|z\|^2,\\
\dot{\mathcal H}
&\le-\frac{11}{64}\|u\|^2-\frac1{1024}\|v\|^2
\le-\frac1{1024}\|z\|^2
\le-\frac1{2064}\mathcal H.
\end{aligned}
\]

The first differential inequality uses exactly
`mathcal E_dot=-||Bu||^2/4`, `lambda_min(B)>1`, and the previously proved
cross inequality `F_dot<=-||v||^2/32+(5/2)*||u||^2`.
Since `mathcal H(0)<=(97/16)*epsilon^2`, integration yields

\[
\|z(t)\|\le8\epsilon e^{-t/4128}=:R(t).
\]

The existing first-exit argument keeps the trajectory in the ball on which
these inequalities hold. This estimate therefore covers the entire
continuous recovery, rather than assuming that a finite numerical endpoint
has entered a local regime.

### A phase-tail bound in the retained reference frame

The exact mean phase rate is
`m_dot=(d/5)*sum_i(h_i-1)*(Bu)_i`, because `sum_i(Bu)_i=0`.
Here `d<1/3` and `|h_i-1|<=4*||v||` on the same ball. Consequently,

\[
|\dot m|
\le\frac{16\sqrt5}{15}\|u\|\|v\|
\le\frac32\|z\|^2.
\]

Integrating the squared decay envelope from the observation time gives

\[
|m_\infty-m(t)|
\le3096 R(t)^2<4096R(t)^2=:Q(t).
\]

Thus each recovering-ring phase differs from its limiting twist-plus-offset
by at most `R(t)+Q(t)`. Its form differs from the common zero form by at most
`R(t)`. The untouched reference remains exactly at its supplied equilibrium.

For a nonnegative integer `n`, choose the declared observation time
`T_n=4128*n`. Since `exp(1)>2`, the exact rational envelopes

\[
R_n=8\epsilon\,2^{-n},\qquad Q_n=4096R_n^2
\]

bound the residual state and phase tail at that time. They also bound all
later times provided recovery continues without an event. No transcendental
time inversion or trajectory evaluation is required.

### Fresh contact rates, including the no-contact comparison

At the declared time write `delta_t=m(t)+v_0(t)` for the relative port phase
and abbreviate `R=R_n`, `Q=Q_n`. The two relative neighbor resultants after
the hypothetical contact are exactly

\[
\begin{aligned}
Z_L={}&e^{i(\kappa+v_1-v_0)}
       +e^{i(-\kappa+v_4-v_0)}+e^{-i\delta_t},\\
Z_R={}&2C+e^{i\delta_t}.
\end{aligned}
\]

The state-dependent form gradients and degree-three rates are

\[
\begin{aligned}
q_L&=3u_0-u_1-u_4,&
\dot x_L^{\rm contact}&=-\frac{q_L}{6}
                         +\frac{\operatorname{Arg}Z_L}{2\pi},\\
q_R&=-u_0,&
\dot x_R^{\rm contact}&=\frac{u_0}{6}
                         +\frac{\operatorname{Arg}Z_R}{2\pi}.
\end{aligned}
\]

These are the actual native rows on the proposed support. They need not be
opposites at finite time. The residual `q` also permits nonzero immediate
phase rates; the zero-phase-rate statement for the ideal recovered contact
in Section 13 does not apply here.

For comparison, on the unchanged isolated support the left rate is

\[
\dot x_L^{\rm free}=-\frac{(Bu)_0}{4}
                    -\frac{(Bv)_0}{4\pi},
\qquad
|\dot x_L^{\rm free}|\le\frac43R,
\qquad \dot x_R^{\rm free}=0.
\]

This baseline distinguishes the effect of the proposed contact from motion
that would already have occurred without it.

There are two equivalent ways to enclose the readout. Direct rational interval
evaluation uses `u_i,v_i in [-R,R]`, the Section 14 enclosure for
`m_infinity`, and `m(t)-m_infinity in [-Q,Q]` in the exact formulas above.
Discarding the correlations among those intervals only enlarges the bounds.
The contact increment is enclosed separately by subtracting the no-contact
baseline. The following simpler estimates also prove sign admission before
that evaluation.

Let `r_L^infinity` and `r_R^infinity` be the ideal limiting contact rates from
Section 14. The resultant differences from that ideal contact satisfy

\[
|Z_L-(2C+e^{-im_\infty})|\le5R+Q,\qquad
|Z_R-(2C+e^{im_\infty})|\le R+Q.
\]

For `|m_infinity|<1/4`, both ideal real parts exceed `3/2`. If
`5R+Q<1/2`, the straight segments to the actual resultants stay in `Re Z>1`;
therefore `|d Arg Z|<=|dZ|` along them. Combining this fact with the form
terms and `pi>3` gives

\[
\begin{aligned}
|\dot x_L^{\rm contact}-r_L^\infty|
&\le\frac53R+\frac16Q<2R+Q,\\
|\dot x_R^{\rm contact}-r_R^\infty|
&\le\frac13R+\frac16Q<2R+Q,\\
|\dot x_L^{\rm contact}-\dot x_L^{\rm free}-r_L^\infty|
&\le3R+\frac16Q<4R+Q.
\end{aligned}
\]

The right contact increment equals its contact rate, since its baseline is
zero. These are bounds on one-sided hypothetical rates at the supplied
state, not on a trajectory following the event.

### One finite horizon and its separate event-work budget

Select one sufficient horizon from those prior inequalities:

\[
n=40,\qquad T=165120,\qquad R=2^{-57},\qquad Q=2^{-102}.
\]

There is no search for the first or fastest readable time. Both the internal
ring edges and the proposed contact are strictly acute at this horizon;
`5R+Q<1/2` also certifies the resultant branches used above.

For completeness, Section 14 gives
`|m_infinity|>(91/160)*2^-40`. For `0<|delta|<1/4`,
`|sin(delta)|>|delta|/2`, `2C+cos(delta)<2`, and
`atan(y)>y/2` for the relevant `0<y<1`. With `pi<4`, these imply

\[
|r_L^\infty|=|r_R^\infty|
>\frac{|m_\infty|}{64}>2^{-47}.
\]

At the chosen horizon `4R+Q<2^-54`. Hence the finite-time left and right
contact rates, and their respective contact-minus-no-contact increments,
retain the certified limiting signs in each preparation. The two
preparations give opposite signs. Each arm keeps its own enclosure; their
actual magnitudes are not asserted to be equal.

The storage jump for the same state-preserving unit contact is separately

\[
\Delta E_{\rm attach}(T)
=\frac12u_0(T)^2+1-\cos\delta_T
=\frac12u_0(T)^2+2\sin^2\!\left(\frac{\delta_T}{2}\right).
\]

Here `|u_0|<=R` and
`delta_T in [m_infinity_lower-R-Q,m_infinity_upper+R+Q]` give a direct
outward enclosure. At this horizon the port-phase intervals exclude zero,
so the jump is strictly positive. The form contribution is bounded by
`R^2/2`; it is not silently discarded. The earlier continuous loss is not
an event reservoir. Executing this contact would require a supplied event
and its positive work or another independently admitted budget.

The shared
[`certify_relational_cycle_memory_readout`](../../src/tnfr/physics/relational_cycle_memory.py)
returns a `RelationalMemoryReadoutCertificate` for nonnegative integer
`decay_blocks`, retaining the fixed amplitude and preparation. Its default
40 blocks give the witness above. Earlier inconclusive bounds remain
explicitly unavailable instead of reporting a zero signal or an executed
measurement. The reader encloses the full contact, no-contact and event
expressions; it neither advances a graph nor inserts an edge.
The [readout controls](../../tests/test_relational_memory_readout.py) check
the exact report and unavailable-result contracts. The
[proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
verify the cross-storage decay, tail and horizon budgets; the
[native mechanism controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
check finite residual contact/no-contact rates, their support degrees and
the state-preserving storage jump.

This closes the finite-time readout obligation for this supplied experiment
within the native model. It does not certify post-contact maintenance,
nondestructive readout, an autonomous connection schedule, optimal timing,
laboratory units, physical memory or a uniquely selected microscopic law.

## 16. A finite contact transmits a signed form response

<a id="relational-finite-contact-memory"></a>

Keep the two preparations, amplitude, reference and observation time
`T=165120` from Section 15. Supply the state-preserving matching-port unit
connection at that time and retain it for one declared duration `h>0`.
The joined graph consists of two C5 rings and one bridge, with unit held
capacities, `e=w=1/2`, `beta=1`, no forcing and the same native relational
law. The comparison without contact continues the two original isolated
flows from exactly the same state at `T`.

The result below concerns continuous model solutions. The certificate does
not execute the connection, sample a trajectory, choose the event or supply
its work. Its new observation is an accumulated form change over a nonzero
duration, including the receiver's response, rather than an instantaneous
rate at the proposed event.

### A state-scaled continuous-flow bound

Use the full twenty-coordinate state deviation `z` from the joined
equilibrium with zero forms and two aligned winding-one twists. In particular,
retain both common phases and all form coordinates; no centering or regional
mean constraint is imposed on the joined flow. Write `rho=||z||_infinity`.
For each preparation a sufficient initial bound is

\[
\rho_0=\max\{|m_{\infty,\rm lower}|,|m_{\infty,\rm upper}|\}
       +Q+R<2^{-39},
\qquad R=2^{-57},\quad Q=2^{-102}.
\]

Consider the fixed sup-norm ball of radius `r=1/100`. Each node has degree
`d_i` equal to two or three. If its relative neighbor resultant is
`C_i+i*S_i`, the equilibrium has `S_i=0` and each incident equilibrium
cosine is at least `C=cos(2*pi/5)>3/10`. The unit Lipschitz bounds for sine
and cosine therefore give, throughout the ball,

\[
C_i\ge\frac7{25}d_i>0,\qquad
|S_i|\le2d_i\rho,\qquad |q_i|\le2d_i\rho.
\]

With `c_0=7/25`, the phase metric and its zero-source extension satisfy

\[
H_i=\frac{\pi S_i}{\arctan(S_i/C_i)}\ge\pi C_i,
\qquad H_i\big|_{S_i=0}=\pi C_i.
\]

There is no singular phase row at `S_i=0`: its mobility can also be written
as the smooth function

\[
\frac1{2H_i}
=\frac1{2\pi}\int_0^1\frac{C_i}{C_i^2+s^2S_i^2}\,ds.
\]

The actual native form and phase rows consequently obey

\[
|\dot\theta_i|\le\frac{\rho}{\pi c_0}\le\frac{25}{21}\rho,
\qquad
|\dot x_i|\le\left(1+\frac1{\pi c_0}\right)\rho\le3\rho.
\]

The form row alone is also uniformly Lipschitz with constant three in this
norm. Its form-coordinate derivative row sum is one. The phase derivative
of the resultant argument has row sum at most `2*d_i/|C_i+i*S_i|`, so its
contribution after multiplying by `1/(2*pi)` is at most `1/(pi*c_0)`.
The ball is convex, and this derivative bound therefore compares any two
states in it. A bound on a phase-row derivative is not needed below.

Let local time `s` start at contact. Gronwall's inequality and a first-exit
argument now give, whenever `rho_0*exp(3*h)<r`,

\[
\begin{aligned}
\|z(s)\|_\infty&\le\rho_0e^{3s},\\
\|z(s)-z(0)\|_\infty&\le\rho_0(e^{3s}-1),\\
|\dot x_i(s)-\dot x_i(0)|&\le3\rho_0(e^{3s}-1),
\qquad 0\le s\le h.
\end{aligned}
\]

Integration yields the state-scaled remainder

\[
\left|x_i(h)-x_i(0)-h\dot x_i(0)\right|
\le\rho_0(e^{3h}-1-3h).
\]

This error vanishes with the actual prepared displacement. It does not
replace the small memory response by an amplitude-independent acceleration
bound.

### One duration with accumulated contact and no-contact responses

For `0<h<=1/3`, the shared exact-clock exponential bounds apply to `3*h`.
If `E_hi` encloses `exp(3*h)` from above, use

\[
\rho_{\rm tube}=\rho_0E_{\rm hi},\qquad
D=\rho_0(E_{\rm hi}-1),\qquad
B=\rho_0(E_{\rm hi}-1-3h).
\]

These respectively enclose the state tube, total state disturbance and
form-response remainder. The contact rate intervals `[a_i,b_i]` from
Section 15 give accumulated form changes in `[h*a_i-B,h*b_i+B]`.
Without contact, the continued isolated decay keeps the left form rate in
`[-4*R/3,4*R/3]` at every later time, while the reference rate stays zero.
Thus the no-contact form changes lie in `h*[-4*R/3,4*R/3]` and `{0}`.
Subtract these enclosures to retain the contact-induced changes as a
separate comparison; no second joined-flow remainder is required.

Choose the single duration

\[
h=2^{-12}.
\]

The inequalities already prove that this duration is sufficient before
evaluating any trajectory. Since `exp(3*h)<2`,
`rho_tube<2*rho_0<r`, `D<6*rho_0*h`, and
`B<9*rho_0*h^2`. Section 15 gives each initial contact-rate magnitude
greater than `(255/256)*2^-47`. Meanwhile,

\[
\frac Bh<\frac9{16}\,2^{-47},\qquad
\frac43R<\frac1{512}\,2^{-47}.
\]

Therefore both accumulated contact form changes, and their respective
contact-minus-no-contact changes, keep their predicted signs. Even after
both error budgets, the per-duration sign margin exceeds
`(221/512)*2^-47`. The two preparations produce opposite signs. In
particular, the initially unperturbed right ring acquires a nonzero signed
form change during the contact; its no-contact change is exactly zero.
The two arms have independent bounds, not an assumed equality of response
magnitudes.

Every original edge remains strictly acute because its phase gap differs
from `+/-kappa` by at most `2*rho_tube<2*r`. The bridge gap is at most
`2*rho_tube`. Continuous motion in this domain preserves the winding-one
identity of each original cycle throughout the declared contact duration.
This is not a claim of unchanged internal shape: the bounded form/phase
disturbance is the interaction being measured.

### Event work and continuous loss retain different accounts

The initial attachment cost is exactly the finite-time cost from Section 15;
its enclosure includes the residual form mismatch. The maximum of the two
cost upper bounds is a sufficient prospective work budget for either
preparation; it is not asserted to be the minimal required work. This is a
conditional budget bound, not an inferred reservoir or an executed supply
of work.

On the joined support the continuous loss is nonnegative and satisfies

\[
L=\frac12\sum_i\frac{q_i^2}{d_i}
\le2\rho_{\rm tube}^2\sum_i d_i
=44\rho_{\rm tube}^2,
\qquad
0\le\int_0^h L\,ds\le44\rho_{\rm tube}^2h,
\]

because the eleven-edge graph has total degree twenty-two. This optional
loss bound is separate from the event jump; it is not work available to pay
for insertion and does not make the addition passive. No removal event or
post-contact continuation is included in this protocol.

The shared
[`certify_relational_memory_contact`](../../src/tnfr/physics/relational_memory_contact.py)
returns a `RelationalMemoryContactCertificate` and per-preparation
`RelationalMemoryContactCase` reports. It reuses the fixed 40-block readout
certificate and the shared exponential enclosure, with default
`duration=1/4096`. Other admitted durations `0<h<=1/3` retain conservative
bounds and explicit sign availability; they do not select an optimal
contact time. The
[contact controls](../../tests/test_relational_memory_contact.py),
[continuous proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
and [native mechanism controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
keep report arithmetic, analytic premises and the engine's nodal rows distinct.

This establishes finite-duration information transfer under one supplied
connection while both cycle windings persist. It does not derive the event's
occurrence, certify nondestructive readout or arbitrarily long contact, or
identify a physical memory carrier, physical time or a unique constitutive law.

## 17. A receiver mean survives contact removal and isolated recovery

<a id="relational-retained-receiver-record"></a>

Keep the complete fixed protocol of Section 16, including `h=1/4096`.
At `T+h`, supply one state-preserving removal of the matching-port bridge.
Then evolve each isolated unit C5 independently under its native unforced
law with held unit capacities and the same structural clock. Recompute
each component's degree, pressure and phase metric on its actual support;
the connected relational executor is not thereby extended to a disconnected
union. No further event, input or capacity change is part of this continuation.

The observation is the receiver's arithmetic mean form relative to its
declared initial mean zero. This differs from the port form change proved
in Section 16. It also differs from winding, internal shape or an absolute
form-origin claim. A common translation of all initial and final forms
leaves this mean **change** invariant.

### Bound the regional mean before invoking conservation

At the start of contact the entire receiver is still at its exact supplied
equilibrium. Only its port acquires a nonzero form rate when the bridge is
added; the other four receiver nodes keep the same neighbor states and
support. If the initial right-port rate is enclosed by `[a_R,b_R]`, the
sum of initial receiver form rates is therefore enclosed by that same
interval.

Let `B=rho_0*(E_hi-1-3*h)` be the per-node continuous remainder from
Section 16. Summing all five form changes and dividing by five gives

\[
\boxed{\qquad
\mu_R(T+h)\in
\left[\frac h5 a_R-B,\ \frac h5 b_R+B\right].
\qquad}
\]

The error is `B`, not `B/5`: all five nodes may contribute a remainder.
The no-contact receiver mean remains exactly zero. No equal-and-opposite
regional transfer is assumed on the irregular joined graph.

The regional intervals have opposite nonzero signs for the two fixed
preparations. This also follows from conservative inequalities independent
of any sampled response. For `0<|delta|<1/4`, use
`|sin(delta)|>(3/4)*|delta|`, `2C+cos(delta)<2`, and
`atan(y)>(2/3)*y` for the relevant `0<y<1`. With `pi<4`, the ideal
contact-rate magnitude exceeds `|delta|/32`. Section 14 gives
`|m_infinity|>(91/160)*2^-40`, and Section 15 changes that rate by less
than `2R+Q<2^-55`.

The already fixed preparation also obeys
`rho_0<(17/16)*2^-40`. At `h=2^-12`, the bound
`B<9*rho_0*h^2` consequently gives

\[
\frac{h}{5}|\dot x_R^{\rm contact}(0)|-B
>
h\left(\frac{91}{25600}-\frac1{163840}-\frac{153}{65536}\right)2^{-40}
=\frac{1989}{1638400}\,2^{-52}>0.
\]

Thus the positive-form preparation writes a positive receiver mean, and
the negative-form preparation writes a negative one. Exact outward
evaluation of the existing intervals provides tighter bounds; it does not
fit or evaluate a new trajectory.

### The cut has its own storage jump

Removal leaves every nodal coordinate, and hence each regional mean,
unchanged. Its exact storage jump is

\[
\Delta E_{\rm cut}
=-\left\{\frac12(x_{L0}-x_{R0})^2
          +1-\cos(\theta_{L0}-\theta_{R0})\right\}\le0.
\]

For example, the whole-contact state radius `rho_tube` immediately gives
`-4*rho_tube^2<=Delta E_cut<=0`. A sharper endpoint enclosure reuses
the already certified all-coordinate disturbance `D`: the final bridge
phase lies in its initial interval enlarged by `2D`, and the final form
mismatch lies in `[-R-2D,R+2D]`. Evaluate the nonnegative edge cost with
these intervals and negate its endpoints to enclose the jump. The phase
intervals still exclude zero in this fixed case, so this sharper jump is
strictly negative.

Deletion therefore adds no positive storage requirement. It does not
retroactively fund the positive insertion work, select when either event
occurs or identify the removed edge storage with an accessible reservoir.

### Both isolated rings remain in their recovery basins

For either ring at the cut, every form and lifted phase deviation from its
original aligned twist is at most `rho_tube`. Each form-edge difference is
at most `2*rho_tube`, giving at most `10*rho_tube^2` of form storage over
the five edges. For phase storage, write the oriented edge gaps as
`kappa+v_(i+1)-v_i`. The linear Taylor terms around the twist sum to zero;
the second derivative of `1-cos` is at most one. Thus another
`10*rho_tube^2` bounds the phase excess, and

\[
0\le E_R-V_*\le20\rho_{\rm tube}^2
<V_{\rm face}-V_*.
\]

The last strict inequality follows already from
`rho_tube<2^-38` and the previously proved
`V_face-V_*>13/1000`. Section 16 preserves the strict acute winding-one
sector, and the state-preserving cut changes none of its internal gaps.
Both post-cut components therefore satisfy the
[isolated-ring capture theorem](RELATIONAL_EFFECTIVE_CONNECTIONS.md#relational-pattern-detachment).
Each remains regular, retains its winding and converges to a uniform form
and uniform twist, with its own common phase offset.

### The retained mean is exactly conserved during that recovery

On a strictly acute winding-one C5 let `B_5` be the cycle Laplacian and
`v` any consistent lifted phase deformation from the uniform twist. The
two-neighbor circular resultant gives exactly

\[
g=-\frac1{2\pi}B_5v,
\qquad
\dot x=-\frac14B_5x-\frac1{4\pi}B_5v.
\]

The pair of relative neighbor angles stays within an interval of width
less than `pi`, so its argument is their arithmetic midpoint on this
branch; this is the domain needed for the first identity. Since the
Laplacian has zero column sums,

\[
\frac d{dt}\left(\frac15\sum_{i\in R}x_i\right)=0.
\]

Combining conservation with capture identifies the receiver's limiting
uniform form with `mu_R(T+h)`. Its certified nonzero interval is therefore
retained for every later time as a regional mean, even while the individual
node forms continue to reorganize. The isolated no-contact receiver keeps
uniform form zero. The signs distinguish the preparations after the
connection that transmitted the signal has been removed.

This conservation step uses common held capacity and degree-two native
pressure; it is not a generic arithmetic-mean conservation theorem for
irregular networks or heterogeneous capacities. It does not depend on the
particular post-cut phase row beyond remaining in the admitted acute
domain. Accordingly, the same retained mean follows if the post-cut phase
row is replaced by any law already admitted by the isolated capture theorem,
with the same native form row and common capacity. That conditional
continuation does not transfer the preceding preparation or contact-response
calculation to another law.

The shared
[`certify_relational_memory_retention`](../../src/tnfr/physics/relational_memory_contact.py)
reuses the fixed contact report and separately reports the receiver-mean
interval, component capture margins and removal jump. The
[retention controls](../../tests/test_relational_memory_contact.py),
[exact proof controls](../../tests/physics/test_relational_finite_memory_proof.py)
and [native regional controls](../../tests/physics/test_relational_cycle_memory_mechanism.py)
check these distinct obligations without advancing a recovery trajectory.

The result is a conditional persistent receiver record in the declared
form coordinate. It does not equate that coordinate with a measured physical
quantity, demonstrate immunity to arbitrary future forcing or topology
changes, select an autonomous contact/cut mechanism, or claim that the
record is encoded by a different winding or internal geometry.

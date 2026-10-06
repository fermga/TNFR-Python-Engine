# Conservative source preparation and finite retention

Energy-speed retention, reverse-source certification, sign/reflection saddle reduction, nonlinear corridors, source-to-target construction and its finite operational and constitutive limitations.

Part of [Sine pattern geometry and dissipative capture](SINE_PATTERN_DYNAMICS.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 46. A conserved budget bounds finite relative-phase travel

<a id="sine-energy-speed-retention"></a>

The full storage need not lie below the acute-face barrier to give a useful
**finite** retention window. A trajectory has limited phase speed while it
remains in an acute winding sector. This supplies a target certificate that
does not require proximity to an exactly rigid family or a small contact pulse.

Keep the complete conservative unit-capacity law with `e=0`, `w=beta=1`,
fixed connected simple undirected unit support, no forcing or events, and
`tau=t/pi`. Select a cycle on five distinct nodes and retain every other
node and edge in the full law. Extra cycle chords are permitted: they add
nonnegative storage and contribute to the actual degrees. Let `L_full`
be the full Laplacian and `K=diag(1/d_i)`. The full form storage and the
minimum acute unit-winding phase storage on the five selected edges are

\[
F=\frac12x^{\mathsf T}L_{\rm full}x,\qquad
V_5=5\left(1-\cos\frac{2\pi}{5}\right)
 =\frac{25-5\sqrt5}{4}.
\]

### An edge-speed bound from the actual full form storage

For a selected edge oriented `i->j`, put `b_e=e_j-e_i` and
`a_e=K b_e`. Its lifted phase gap satisfies

\[
\delta_e'=b_e^{\mathsf T}KL_{\rm full}x
 =a_e^{\mathsf T}L_{\rm full}x.
\]

Cauchy–Schwarz for the positive semidefinite Laplacian gives

\[
|\delta_e'|^2\le
 (a_e^{\mathsf T}L_{\rm full}a_e)
 (x^{\mathsf T}L_{\rm full}x)
 =2\kappa_e F,\qquad
\kappa_e=\frac1{d_i}+\frac1{d_j}+\frac2{d_i d_j}.
\]

The coefficient is `a_e^T L_full a_e`, computed from the full degrees and
the selected connecting edge. On the C5/private-leaf support of Sections
35–45, receiver degrees are three, so every `kappa_e=8/9` and the bound is
`|delta_e'|^2<=16F/9`. Environmental form is retained in `F`; replacing it
by receiver form storage would invalidate this estimate.

While every receiver gap lies in `(-pi/2,pi/2)` and their winding is
`sigma=+1` or `-1`, convexity of `1-cos(delta)` on that interval and
`sum(delta)=2pi*sigma` give `V_R>=V_5`. Every other edge's phase storage
is nonnegative.
For a conserved full-storage upper bound `H_bar`, therefore,

\[
F\le H_{\rm bar}-V_5,\qquad
\boxed{|\delta_e'|\le c_e
 :=\sqrt{2\kappa_e(H_{\rm bar}-V_5)}.}
\]

A nonempty admitted acute target necessarily has `H_bar>=V_5`. A smaller
bound cannot be repaired by clipping the square-root argument to zero.
Write `c_H=max_e c_e`; for the private-leaf support this is
`4sqrt(H_bar-V_5)/3`.

### First-exit certificate, including full-state uncertainty

Suppose every state in a declared full-state target has the same acute
unit winding, initial acute margin at least `m>0`, and full storage at most
`H_bar`. For any supplied finite `T>0`, the strict inequality

\[
\boxed{m-c_HT>0}
\]

certifies that every such trajectory remains acute with its original winding
throughout `[-T,T]`, with margin at least `m-c_H T`. Indeed, up to a proposed
first boundary exit, the speed estimate bounds each gap's displacement by
`c_H` times elapsed time. Reaching a face by time `T` would require a
displacement at least `m`, contradicting the strict bound. The same argument
applies backwards. This is a first-exit bootstrap, not an assumption that
the trajectory remains acute after the initial observation.

At a uniform unit twist on the private-leaf support, `m=pi/10`. For `T=1`,
the admissible storage ceiling is consequently

\[
H_{\rm bar}<V_5+\frac{9\pi^2}{1600}
 =3.51043155\ldots\;>\;\frac72.
\]

Thus the finite retention condition has a nonempty overlap with the storage
budgets necessary for acquisition from zero winding. For example, at
`H_bar=7/2` the strict duration ceiling is
`3pi/(40sqrt(7/2-V_5))=1.10967355...`. This does not establish acquisition
at the barrier or select any trajectory reaching the target.

For a general independent box around a supplied full state `(x*,theta*)`,
with form radii `r_x` and lifted-phase radii `r_theta`, one valid upper bound
is

\[
\begin{aligned}
H_{\rm bar}={}&H(x^*,\theta^*)
 +\sum_i|(L_{\rm full}x^*)_i|r_{x,i}
 +\sum_i|S_i(\theta^*)|r_{\theta,i}\\
&+\frac12\sum_{\{i,j\}\in E}
 \left[(r_{x,i}+r_{x,j})^2
       +(r_{\theta,i}+r_{\theta,j})^2\right].
\end{aligned}
\]

This follows from the exact form quadratic and the global phase remainder
bound `cos(delta)<=1`. It retains nominal torques at represented phase
centers; approximating a critical twist numerically does not permit declaring
those torques zero. The initial acute margin must likewise account for the
sum of both endpoint phase radii on every selected edge.

### An explicit target with uncertainty in all twenty coordinates

On the C5/private-leaf support, let `L_R` be the receiver cycle Laplacian
and take the nonzero centered form perturbation

\[
q=\kappa(1,-1,0,0,0),\qquad \kappa=\frac1{128}.
\]

For receiver indices `i=0,...,4` and their matched leaves, choose centers

\[
x_R^*=\frac u4\mathbf1+q,\qquad
x_Q^*=-\frac{3u}{4}\mathbf1+(I_5+L_R)q,\qquad
\theta_{R,i}^*=\theta_{Q,i}^*=\frac{2\pi(i-2)}5,
\qquad u=\frac7{50}.
\]

Allow independent errors of absolute size at most `epsilon=1/4096` in
every fine form and lifted phase coordinate. The capacities and support
remain fixed premises. The initial winding is one throughout this box and
the initial acute margin satisfies

\[
m\ge\frac\pi{10}-2\epsilon.
\]

The center has contact form contrasts `r=u*1-L_R*q`, which are all positive,
and form storage `5u^2/2+13kappa^2`. Its cycle form differences have total
absolute magnitude `4kappa`, while its contact contrasts sum to `5u`.
Every edge form difference acquires an error of at most `2epsilon`.
The receiver phase errors telescope around the cycle: their edge increments
sum to zero.
Taylor's bound `1-cos(alpha+eta)<=1-cos(alpha)+sin(alpha)eta+eta^2/2`
therefore cancels the cycle's linear term. Contact phase gaps are at most
`2epsilon`. Together these give the full, correlated bounds

\[
V_5+\frac52(u-2\epsilon)^2\le H\le
H_{\rm bar}:=V_5+\frac52u^2+13\kappa^2
 +10u\epsilon+8\kappa\epsilon+40\epsilon^2.
\]

The lower bound also uses the already verified acute unit-winding geometry,
which gives `V_R>=V_5`, and Jensen's inequality for the five contact form
contrasts, whose mean lies within `2epsilon` of `u`. It does not discard a
possibly negative phase term. For these declared rational values,

\[
\frac52u^2+13\kappa^2+10u\epsilon+8\kappa\epsilon+40\epsilon^2
 =\frac{13147281}{262144000},\qquad
\frac52(u-2\epsilon)^2=\frac{51022449}{1048576000}.
\]

All target states have `H>7/2`; the enclosing full-storage interval is
approximately `[3.50357382,3.50506793]`. At `T=1`, the retained acute margin
is greater than `1/100` radians. This strict rational conclusion follows
already from `pi>157/50` and `c_H<3/10`; the sharper evaluated lower
margin is about `0.0150730` radians. The box has positive width in every
form and phase coordinate. Its center is already nonrigid: Section 45 gives
zero first and second relative phase derivatives but a nonzero third
derivative for this compensated form preparation. The retained identity
therefore includes actual changing geometry, not only errors around an
exactly moving collective family.

### What this closes, and what it leaves open

This provides a full-state retention target for scaled duration one
(original-clock duration `pi`) whose entire storage range is above the
zero-winding acquisition barrier. It avoids requiring entry into a particular
moving-family window. It does not supply a
zero-winding preparation whose actual trajectory enters this target, and
overlapping source and target budgets do not prove that connection.

Because the certificate also protects the preceding duration one, an
identity-absent preparation must precede that backward window. An entry
checkpoint is not necessarily the first instant at which acute identity
appears. Longer or indefinite retention, an attractive basin, autonomous
preparation and physical identification remain outside this certificate.

The shared
[`assess_sine_cycle_retention`](../../src/tnfr/physics/relational_sine_regional.py)
reader applies this argument to actual admitted centers, full-coordinate
error budgets and graph degrees. Its
[contract](../../docs/contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-cycle-retention)
distinguishes the exact source set from displayed interval bounds. The
[rational example](../../docs/guides/relational/SINE_REGIONAL_DYNAMICS.md#energy-speed-retention)
retains the actual small phase torques of a near-twist center; it does not
replace that center with the symbolic geometry used in the proof above.

### A directed preparation for a possible reverse-time check

One directed checkpoint center selects phase travel toward the sector
saddle in reversed time. On the same private-leaf support, set

\[
\theta_{R,i}=\theta_{Q,i}=\frac{2\pi(i-2)}5,\qquad
x_R=\frac1{32}(-4,-4,0,4,4),\qquad
x_Q=\frac1{32}(-6,-5,0,5,6).
\]

Its receiver and contact form storages are `48/1024` and `5/1024`.
Consequently `H=V_5+53/1024=3.50667284...`, with strict duration-one
acute reserve `pi/10-sqrt(53)/24>0`. The actual full rows give

\[
\theta_R'=\theta_Q'=\frac1{32}(-2,-1,0,1,2),\qquad \phi'=0.
\]

Under the exact reversing involution `R(x,theta)=(-x,theta)`, the vector
field obeys

\[
\mathcal F(Rz)=-R\mathcal F(z).
\]

Reversing this checkpoint therefore gives
initial oriented cycle-gap rates `(-1,-1,-1,-1,4)/32`: the four consecutive
gaps decrease from `2pi/5` toward `pi/3`, while the closing principal gap
increases toward `2pi/3`. These rates select the direction of the sector
saddle geometrically. They do not establish its crossing, later zero winding,
or constancy of that initial direction. The factor `1/32` is a supplied
experimental amplitude, not a coefficient derived for the constitutive law.

The strict retention margin permits a nonzero-width full-state checkpoint
box, admitted with the general box-storage estimate above. Any finite reverse
check must declare that box, its horizon and numerical budget before
evaluation. If a later validated check proves that the whole backward image
of an open checkpoint box `B` has zero winding at time `-T_back`, the
corresponding source is the exact flow image `Phi_{-T_back}(B)`. Its openness
follows from the smooth invertible flow, and its forward trajectories return
to `B`. An outer rectangular enclosure of that image is not interchangeable
with the source: an independently supplied rectangular source requires its
own forward-containment proof. Neither construction derives spontaneous
preparation. Section 47 specifies the represented-point evaluation and its
evidence obligations; this symbolic recipe alone leaves formation open.
The execution plan owns any prospective check.

## 47. A reversible enclosure can certify an independent formation source

<a id="sine-reversible-preparation"></a>

An open backward image in Section 46 need not be a rectangular preparation
that can be specified independently. The complete law also supplies a
global error bound that can bridge this distinction. The resulting theorem
requires actual validated endpoint evidence and tests the whole proposed
source; it does not infer formation from reversibility or storage overlap.

### Complete state, clock and global error bound

Keep the same conservative unit-capacity law and fixed full support. In
this section `Phi_A` denotes the flow for **original-clock** duration `A`.
Only the twenty form and lifted-phase coordinates vary on the private-leaf
support. Capacity is held at one and is not a perturbation coordinate.
For two full states `z=(x,theta)` and `z_tilde`, use the common sup norm of
all form and phase differences.

In the scaled clock, the form row is a normalized sum of neighbor sines,
and the phase row is `K L_full x`. Since sine is globally Lipschitz with
constant one and `||K L_full||_infinity<=2`,

\[
\|\mathcal F_\tau(z)-\mathcal F_\tau(\widetilde z)\|_\infty
 \le2\|z-\widetilde z\|_\infty.
\]

Therefore the complete original-clock flow obeys

\[
\boxed{\|\Phi_A(z)-\Phi_A(\widetilde z)\|_\infty
 \le e^{2|A|/\pi}\|z-\widetilde z\|_\infty.}
\]

This comparison uses continuous phase lifts. It does not replace a lifted
distance by independently wrapped endpoint differences. The global bound
retains all environmental coordinates and is independent of the trajectory's
intermediate winding sector.

### From a reverse endpoint to a forward source ball

Declare an exact target center `z_*` and an independent box
`B_epsilon(z_*)` already certified by Section 46 to retain acute unit
winding for scaled duration `T`. The reversal `R(x,theta)=(-x,theta)` is
an isometry in the same norm and satisfies

\[
\Phi_A(Rz_*)=R\Phi_{-A}(z_*).
\]

Suppose a validated positive-time forecast from `Rz_*` supplies an outward
endpoint enclosure `I_A` containing `Phi_A(Rz_*)`. Let `m_A` be its exact
componentwise midpoint and `eta_A` the maximum of its form/phase half-widths.
Put

\[
c_A=R m_A,\qquad s_A=\Phi_{-A}(z_*).
\]

Then `||c_A-s_A||_infinity<=eta_A`. For every state in an independently
declared source ball `B_delta(c_A)`, the forward error bound gives

\[
\|\Phi_A(z)-z_*\|_\infty
 \le e^{2A/\pi}(\eta_A+\delta).
\]

Consequently the following conditions suffice for formation followed by
the declared retention interval:

1. Every point of the full source ball has receiver winding zero, certified
   with the same node order and unique circular branches on the complete
   source phase intervals.
2. The target box has the stated full-state retention certificate, including
   its actual law, support, capacity, phase margins and conserved budget.
3. An outward exponential upper bound `E_A>=exp(2A/pi)` satisfies the strict
   inclusion `E_A*(eta_A+delta)<epsilon`.

Every source member then enters `B_epsilon(z_*)` at the common original-clock
time `A` and retains acute unit winding for at least the following
original-clock duration `pi*T`. Unlike a mere outer enclosure of a backward
flow image, the ball in this theorem has an independent forward-containment
proof. It has nonzero width in every form and phase coordinate and does
not constrain their conserved origins to one lower-dimensional surface.

[`assess_sine_reversible_preparation`](../../src/tnfr/physics/relational_sine_regional.py)
owns this conditional composition and returns `SineReversiblePreparation`.
It rebuilds the target through `assess_sine_cycle_retention` and retains
every grid endpoint's source, error bound, conditions and availability.

The target certificate also holds backwards. A zero-winding preparation
must therefore precede that protected interval: require `A>pi*T`, checked
against an outward upper bound for `pi*T`. Reversal constructs a comparison
initial condition; it is not a support event, a sign-flipping operation
inserted into the physical evolution, or a source of forcing.

### Numerical admission is part of the composition

The forecast must enclose the declared exact initial point `Rz_*` and
retain a valid chain from time zero to the selected endpoint. Read and
re-admit primitive state, full support, law, clock, node order and interval
evidence before consuming that endpoint. A source identifier, cached
success flag or midpoint-only phase observation cannot replace those
premises. Structural admission alone does not authenticate a claimed
enclosure or replace the numerical method's proof obligations and retained
provenance.

Compute midpoint, half-width and source endpoints without float conversion.
The source may be declared by exact rational endpoints `c_A +/- delta`;
outward interval arithmetic can enclose these endpoints for admission. If
instead the exported source is the enlarged materialized box, its actual
maximum radius about `c_A` must replace `delta` in the forward error bound.
The same rule applies to any enlargement of the endpoint enclosure. The
exponential uses an upper exponent, including the lower bound of `pi`
in the denominator; strict inequalities cannot rely on rounded midpoint
values or equality accepted as tolerance.

Conservation supplies an additional consistency check: each source member's
full storage equals that of its reached target member. Independently computed
source and target storage intervals must not contradict this identity.
Their overlap alone is not a proof of entry. The forward inclusion, rather
than that overlap, establishes the required connection.

A valid prefix is enough for a certificate at its endpoint. A later failure
to extend a forecast does not undo an already validated prefix, but the
declared full-horizon status must remain separately visible. No failed or
unavailable step may be skipped to construct a later endpoint certificate.

### Frozen bounded evaluation and its scope

The declared preparation uses the form vectors at the end of Section 46
and the exact represented phase centers

\[
\theta_{R,i}^*=\theta_{Q,i}^*
 =(i-2)\frac{1256637}{10^6}.
\]

These are the actual rational phases consumed by the forecast; they are not
declared equal to `(i-2)*2pi/5`. The target box has `epsilon=1/4096` in
every form and phase coordinate, with scaled retention duration `T=1`.
Its retention preflight does not establish any backward crossing.

One bounded evaluation is specified in original time by horizon `128/5`,
step `1/10`, Taylor order 16 and at most 256 steps. The independent source
radius is fixed at `delta=2^-100`. Evaluate the predeclared grid endpoints
in increasing time, after the backward-window guard, and select the first
endpoint satisfying every source and target condition above. This is a
deterministic constructive rule fixed before the response, not permission
to retune the checkpoint, radius, horizon or solver after observing failure.
The selected source is a derived mathematical preparation, not an
independently observed natural occurrence or a reserved physical prediction.
The nonzero radius `2^-100` gives the source box a nonempty interior;
it is not a practically demonstrated laboratory tolerance or a broad basin.

These numerical choices are an evaluation budget, not constants selected
by the nodal law. Incomplete coverage without a passing prefix is
`unavailable`; complete coverage without a passing endpoint is
`no_certificate_on_declared_grid`. Neither rules out formation through
other preparations. The execution plan remains the sole task queue.

### Retained response: the directed checkpoint does not supply a zero-winding source

The single evaluation retains its
[declaration](../../docs/assets/conservative_reverse_preparation/declaration.json),
[frozen protocol](../../docs/assets/conservative_reverse_preparation/response-v1.protocol.json),
[source archive](../../docs/assets/conservative_reverse_preparation/response-v1.source.zip)
and [response](../../docs/assets/conservative_reverse_preparation/response-v1.json).
The response has no evaluation error and covers all 256 steps through
original time `128/5`. Its verdict is `no_certificate_on_declared_grid`.

| Declared obligation | Retained result |
| --- | --- |
| Target retention | Certified for `T=1`; full-box storage upper bound about `3.5068583304`, strict retention margin greater than `0.00978` |
| Whole source winding zero | Fails at all 256 endpoints: each exact source ball has winding `+1` |
| Strict source-to-target radius inclusion | Passes at all 256 endpoints |
| Source/target storage compatibility | Passes at all 256 endpoints |
| Endpoint after the protected backward interval | Passes at the final 225 endpoints |
| Complete numerical coverage | All 256 steps; no unavailable suffix |

At the last endpoint, the enclosure halfwidth is about `4.06837e-30`,
the amplification upper bound is about `1.19647e7`, and the propagated
source-to-target error is less than `5.82e-23`, well below `1/4096`.
These decimal values summarize exact retained bounds. Numerical precision
does not explain this candidate's failure: the missing property is initial
zero winding. The successful inclusion tests connect already wound source
balls to the retention target and therefore do not establish formation.

A separate **retrospective reading**, without solver replay, checks the
unique branches of every whole-time tube. All 256 tubes retain winding `+1`,
so the enclosed reverse trajectory never reaches zero winding anywhere on
this declared horizon, including between endpoints. For example, the
closing principal-gap hull is contained in `(1.2561, 1.8298)`, below the
sector saddle's closing gap `2*pi/3`. Only 98 tubes certify acuteness
throughout their own interval; sector preservation is a different assertion
from acute identity retention. This reading does not alter the original
grid verdict or establish behavior beyond `128/5`.

The same retained tubes and the full linear phase row also bound the closing
gap's scaled-time derivative strictly inside `(0.0287, 0.1254)` throughout
the horizon. Its directed motion therefore persists in this finite response;
the record does not demonstrate a turn-back, permanent trapping or impossibility
of later crossing. A prospective continuation would need its own justified
bound and evidence, rather than rewriting the completed horizon.

The local initial gap direction in Section 46 is consequently insufficient
to establish travel to the saddle under full contact feedback. This result
does not prove a new trapping invariant, nor refute the reversible-source
theorem. Sections 48-50 establish nonlinear connections under a separate
exact preparation, using coupled phase/contact dynamics. They do not
extend this completed response or declare its checkpoint successful.

## 48. An exact sign/reflection reduction retains the sector saddle

<a id="sine-involution-saddle-reduction"></a>

This section uses the same complete zero-loss, unit-capacity sine law on
the five-node receiver cycle and its five matched private leaves. Receiver
indices are `0,...,4`, leaf indices are `5,...,9`, and the cycle orientation
is `0 -> 1 -> 2 -> 3 -> 4 -> 0`. Every contact remains live. With
`tau=t/pi`, the complete rows are

\[
x'=K S(\theta),\qquad \theta'=K Lx,
\qquad K=\operatorname{diag}(1/d_i).
\]

The reduction below is an exact representation of a declared invariant
family of those rows. It does not replace the nonlinear law with its
tangent, choose a preparation, or discard a coupled environmental coordinate.

### Exact invariant family and reconstruction

Let `P` reflect receiver indices by `i -> 4-i` and reflect the matched
leaves in the same way. It is an automorphism of the full support and
preserves the capacities and degrees. The vector field `F` obeys both

\[
F(Pz)=P F(z),\qquad F(-z)=-F(z),\qquad z=(x,\theta).
\]

The second identity follows from sine oddness and the linear phase row.
Thus `Tz=-Pz` is an involutive **equivariance**. Uniqueness of the globally
smooth full flow preserves `Fix(T)` at every time. With real phase lifts,
its exact coordinates are

\[
\begin{aligned}
x_R&=(a,b,0,-b,-a), &x_Q&=(c,d,0,-d,-c),\\
\theta_R&=(u,v,0,-v,-u), &\theta_Q&=(p,q,0,-q,-p).
\end{aligned}
\]

The fixed receiver and leaf have zero form and phase in this declared
origin convention. Their full derivatives vanish by paired cancellation;
they remain nodes of the system. No coordinate is an inferred sensor or
an automatically wrapped phase. Common form and phase origins can be
restored by their exact shift symmetries, but such shifts are a separate
declared reconstruction.

Writing `X=(a,b,c,d)` and `Theta=(u,v,p,q)`, the eight nonlinear rows are

\[
\begin{aligned}
3a'&=\sin(v-u)-\sin(2u)+\sin(p-u),\\
3b'&=\sin(u-v)-\sin v+\sin(q-v),\\
c'&=\sin(u-p), &d'&=\sin(v-q),\\
3u'&=4a-b-c, &3v'&=3b-a-d,\\
p'&=c-a, &q'&=d-b.
\end{aligned}
\]

Substitution reconstructs every full nodal row, including both fixed
nodes. Conversely, every full state fixed by `T` has this representation.
The result holds globally in real phase lifts, including nonacute gaps;
it is not a small-oscillation or near-equilibrium approximation.

The general shared owner
[`assess_sine_involution_reduction`](../../src/tnfr/physics/relational_sine_symmetry.py)
checks an admitted signed involution and supplies `SineInvolutionReduction`.
Its `.evaluate(form_coordinates, phase_coordinates)` reconstructs the
family's full state and rows. For this reflection the representatives are
`(0,1,5,6)`, giving the coordinate order above. That API reports rates in
original structural `t`; divide the displayed scaled-time rows by `pi`
when comparing them. The source anchors the law and support and need not
already belong to the declared family.

This `T` differs from the time reverser of Section 47:
`R(x,theta)=(-x,theta)` satisfies `F(Rz)=-R F(z)`. The map `T` commutes
with forward evolution; `R` exchanges forward and backward evolution.
Neither is an event applied during a trajectory.

### Inherited storage and oriented winding

The complete storage restricted to this family is exactly

\[
\begin{aligned}
H={}&(a-b)^2+b^2+2a^2+(c-a)^2+(d-b)^2\\
 &+2[1-\cos(v-u)]+2[1-\cos v]+[1-\cos(2u)]\\
 &+2[1-\cos(p-u)]+2[1-\cos(q-v)].
\end{aligned}
\]

No environmental storage is omitted. With
`D=diag(1/6,1/6,1/2,1/2)`, the reduced equations satisfy

\[
X'=-D\nabla_\Theta H,\qquad
\Theta'=D\nabla_X H,\qquad H'=0.
\]

The factors are inherited from the two equal-multiplicity receiver and
leaf coordinates and their full degrees. This identity is a restriction
of the actual conserved storage and nodal field, not a new auxiliary
Hamiltonian or an added law.

Where all principal branches are unambiguous, the receiver period is

\[
W=\frac{2\operatorname{wrap}(v-u)
       +2\operatorname{wrap}(-v)
       +\operatorname{wrap}(2u)}{2\pi}.
\]

The corresponding raw gaps telescope to zero. Integer branch offsets,
rather than rounded division by `2pi`, establish the winding. Reflection
alone reverses the oriented period, and phase sign reversal also reverses
it. Their combination preserves it. Consequently this invariant family
can contain unit-winding states; it does not have the zero-winding
obstruction of an ordinary reflection-fixed preparation.

### The full saddle's unique hyperbolic pair lies in this family

At

\[
X_*=0,\qquad
\Theta_*=(-2\pi/3,-\pi/3,-2\pi/3,-\pi/3),
\]

all contacts have zero phase difference. The five principal receiver
gaps are `(pi/3,pi/3,pi/3,pi/3,2pi/3)`, and each oriented sine current
is the same. The full state is therefore an equilibrium with `H=7/2`
and `W=1`. It is not acute: its closing gap is `2pi/3`.

The full form Hessian restricted to `X` and the phase Hessian restricted
to `Theta` are, respectively,

\[
G=\begin{pmatrix}
8&-2&-2&0\\-2&6&0&-2\\-2&0&2&0\\0&-2&0&2
\end{pmatrix},\qquad
B=\begin{pmatrix}
1&-1&-2&0\\-1&4&0&-2\\-2&0&2&0\\0&-2&0&2
\end{pmatrix}.
\]

The leading principal minors of `G` are `8,44,64,80`, so `G` is positive
definite. Eliminating the two contact differences makes `B` congruent to
the receiver block `[[−1,−1],[−1,2]]` and two positive entries `2`.
The receiver block has negative determinant, so `B` has three positive
directions and one negative direction, with no zero direction. The
linearized reduced system is

\[
\dot X=-D B\Theta,\qquad \dot\Theta=D G X,\qquad
\ddot\Theta=-D G D B\Theta.
\]

Because `D G D` is symmetric positive definite, its product with `B`
is similar to a real symmetric matrix with the same inertia as `B`.
Thus the eight-dimensional tangent has one real stable/unstable pair
and three purely imaginary pairs. These are exact eigenvalue-type
counts, not a nonlinear attraction or stability claim.

The full graph has the same single negative phase direction. Its contact
differences contribute five positive squares. For the four successive
receiver path differences `d_k`, its remaining Hessian quadratic form is

\[
\frac12\sum_{k=0}^3d_k^2-rac12\left(\sum_{k=0}^3d_k\right)^2.
\]

Writing `d=m*1+r`, with `sum(r)=0`, gives
`||r||^2/2-6m^2`: three positive directions and one negative direction,
besides the global phase origin. Hence the complete phase Hessian has
inertia `(8 positive, 1 negative, 1 zero)`. Removing the conserved global
form and phase origins makes the form Hessian positive definite. The
full twenty-dimensional linearization has one real pair, eight imaginary
pairs and two semisimple zero modes for those origins. The odd reduction
already contains the only real pair; the complementary even subspace
contains five imaginary pairs and the two origin modes.

For a reproducible algebraic identification, let `lambda=sigma^2>0` be
the positive eigenvalue of `-D G D B`. Its characteristic polynomial is

\[
P(\lambda)=324\lambda^4+1404\lambda^3
             +1527\lambda^2+61\lambda-15.
\]

Descartes' sign rule gives exactly one positive root. Exact rational
signs at `19709/250000` and `78837/1000000` isolate it between those
endpoints. Set

\[
b_\lambda=\frac{4\lambda+1}{18\lambda^2+43\lambda+5},\qquad
v_\lambda=\left(1,b_\lambda,
       \frac{7-b_\lambda}{6\lambda+8},
       \frac{-1+10b_\lambda}{6\lambda+8}\right).
\]

Substitution into `(-D G D B-lambda I)v_lambda` gives zero, using
`P(lambda)=0`. The corresponding form vector is
`X_lambda=-D B v_lambda/sigma`. The tangent eigenvectors are
`(X_lambda,v_lambda)` and `(-X_lambda,v_lambda)`, with rates `+sigma`
and `-sigma` in scaled time, or `+sigma/pi` and `-sigma/pi` in original
time. Their closing-gap derivative is nonzero because the first phase
component is one. This gives a direction transverse to the wider-sector
boundary; crossing that boundary still does **not** mean crossing the
principal phase seam or reaching `W=0`.

The shared
[`assess_sine_c5_leaf_saddle`](../../src/tnfr/physics/relational_sine_resonance.py)
owns this exact conditional saddle analysis. Its mathematical phase turns
must remain distinct from rational approximations to radians in a numerical
preparation. Rounded phases cannot be declared an exact critical point.
In its report, `odd_phase_hessian` is `B` above,
`odd_phase_from_form` is `D G`, and `odd_negative_form_from_phase` is `D B`;
the rate blocks must not be confused with the unweighted storage Hessians.

### A finite nonlinear envelope for the tangent comparison

Let `z_*` denote the exact saddle in full coordinates, and write

\[
F(z_*+h)=Jh+\mathcal R(h).
\]

The phase row is linear, so `R_phase=0`. For each sine edge, the global
Taylor bound
`|sin(delta+eta)-sin(delta)-cos(delta)*eta|<=eta^2/2`, together with
`|eta|<=2||h_phase||_infinity`, yields

\[
\|\mathcal R_{\rm form}(h)\|_\infty
 \le 2\|h_{\rm phase}\|_\infty^2.
\]

The complete field is globally Lipschitz with constant two in the joint
form/phase maximum norm in `tau`, and `||J||_infinity<=2`. The same bounds
hold for the exact reduced field because its signed reconstruction is an
isometry in that norm. For `||h(0)||_infinity<=r`, Gronwall and Duhamel
therefore give the fully nonlinear comparison

\[
\boxed{
\|h(\tau)-e^{J\tau}h(0)\|_\infty
 \le r^2 e^{2|\tau|}\bigl(e^{2|\tau|}-1\bigr).
}
\]

Indeed, `||h(s)||<=r exp(2s)` for positive `s`; integrating
`exp(2(t-s))*2r^2 exp(4s)` gives the displayed bound. For negative time,
apply the same argument to `-F` and `-J`. This requires no acute chart
and no phase normalization. It is useful only where its retained error
is small enough for the intended directional comparison; it does not
turn a tangent eigenvector into a remote acquisition certificate.

### A nonlinear local sector passage with an explicit error budget

The envelope proves a small but nontrivial passage under the complete law,
without evaluating a trajectory. Normalize the eigenvector as above, with
receiver phase component `v_lambda[0]=1`, and choose

\[
0<\varepsilon\le2^{-24},\qquad
h(0)=\varepsilon(3X_\lambda,-v_\lambda).
\]

This is an exact, correlated preparation around the saddle. Its algebraic
eigenvector and exact phase turns specify a mathematical family, not a
measured preparation or an independent twenty-coordinate uncertainty box.
The rational eigenvalue bracket implies
`1/10<b_lambda<1/5`, `79/100<v_lambda[2]<83/100`,
`0<v_lambda[3]<b_lambda` and `sigma>7/25`. Substitution in
`X_lambda=-D B v_lambda/sigma` then gives
`||v_lambda||_infinity=1` and `||X_lambda||_infinity<1`.
Thus `||h(0)||_infinity<=3*epsilon`.

The phase component of the tangent solution is exactly

\[
h^{\rm lin}_\Theta(\tau)
 =\varepsilon\bigl(e^{\sigma\tau}-2e^{-\sigma\tau}\bigr)v_\lambda.
\]

Initially its first component is `-epsilon`, so the closing principal
gap is `2pi/3-2epsilon`. The state is strictly inside the wider sector
`Omega_1` of Section 42. At `tau=3`,

\[
e^{3\sigma}>e^{21/25}
 >1+\frac{21}{25}+\frac12\left(\frac{21}{25}\right)^2>2,
\qquad e^{3\sigma}-2e^{-3\sigma}>1.
\]

The full nonlinear comparison error at that time is less than

\[
9\varepsilon^2 e^6(e^6-1)
 <9\cdot729\cdot728\,\varepsilon^2<\varepsilon,
\]

using `e<3` and `9*729*728<2^24`. The actual first phase perturbation
is therefore positive at `tau=3`: its closing gap has crossed above
`2pi/3`. The orbit has exited `Omega_1` within that fixed finite time.

This is not a hidden winding or acute-entry claim. Throughout `[0,3]`,
`||h_phase||_infinity<2187*epsilon<pi/12`. The four path gaps remain
in `(pi/6,pi/2)` and the closing principal gap remains in
`(pi/2,5pi/6)`. No principal branch changes; the winding remains `1`
and the closing gap remains nonacute throughout the passage. Negating
this perturbation gives the opposite local sector passage. Neither
construction proves travel from the earlier numerical checkpoint,
zero-winding preparation, or a distant acute pattern. It establishes
that the unique full-system hyperbolic direction permits a quantitatively
controlled nonlinear crossing of the local sector boundary.

### What the reduction does not remove

A nonzero-width independent box in all twenty form and phase coordinates
is not confined to `Fix(T)`. Exact symmetry does not justify dropping its
even perturbations. If `S q_nom(tau)` is an exact reconstructed reduced
solution and an actual full initial state differs from it by at most
`delta` in the joint maximum norm, the same complete-field Lipschitz bound
gives

\[
\|z(\tau)-S q_{\rm nom}(\tau)\|_\infty
 \le\delta e^{2|\tau|}.
\]

A validated error bound for a numerically computed reduced nominal
solution must also be retained. This supplies a route for using a reduced
nominal calculation with full transverse uncertainty; it does not assert
that the uncertain members stay symmetric. The frozen rational checkpoint
and its time-reversed center belong to the exact family, whereas their
outward endpoint boxes contain independent full-state perturbations.
The absence of an additional real tangent eigenpair is not a nonlinear
stability theorem.

The exact saddle remains stationary. No distinct orbit reaches that exact
equilibrium at a finite time under the same smooth law. Nearby states and
transverse tangent directions can be studied without making that false
finite-entry claim. Finally, a trajectory leaving the `2pi/3` protected
sector can still have `W=1` and be nonacute. The
[same-orbit construction](#sine-conservative-formation-retention) supplies
the additional whole-state passage and finite-retention proof under its
explicit preparation and uncertainty hypotheses; the local result alone
does not imply those conclusions.

## 49. A directed nonlinear corridor reaches the winding seam

<a id="sine-directed-saddle-corridor"></a>

Retain Section 48's complete conservative unit law, C5/private-leaf
support, signed invariant family and clock `tau=t/pi`. The following
argument uses its actual nonlinear rows throughout a finite phase corridor.
It does not extrapolate a saddle eigenvector or replace the environmental
nodes by a force. The result connects a nonacute unit-winding state outside
the protected sector to zero winding. Reversing it does not by itself
connect zero winding to an acute finite-retention target.

### The principal seam has a lower minimum than the sector saddle

At a principal seam, one cycle edge has potential `2`; the remaining
four oriented gaps sum to `pi` modulo `2pi`. Divide those four gaps into
two pairs. If the first pair sums to `A` modulo `2pi`, their cosine sum
is at most `2|cos(A/2)|`; the other pair contributes at most
`2|sin(A/2)|`. Cauchy's inequality therefore bounds the four cosines
by `2sqrt(2)`, with equality when each remaining principal gap is `pi/4`
up to orientation. Consequently

\[
\min_{\text{principal seam}}V_R=6-2\sqrt2<\frac72.
\]

This minimum is attained within the signed family: take
`u=-pi/2`, `v=-pi/4`, matched leaf phases `p=u,q=v`, and zero form.
The four path gaps are `pi/4` and the closing gap is at its seam. Thus
the signed reduction introduces no stronger static seam barrier. The
`7/2` saddle separates the wider sector from its exterior; it is not
the minimum cost of changing the integer winding.

The [closed-budget result](SINE_REGIONAL_FORMATION.md#sine-cycle-sector-barrier) nevertheless proves
that a finite outside-to-acute orbit needs **strictly** `H>7/2`.
At equality its sector boundary can only contain full equilibria, which
a distinct orbit cannot reach in finite time. An energy-compatible
seam configuration and a local unstable direction do not override this
obstruction.

### Exact transverse storage and a directed momentum

Use the real lifted coordinates

\[
s=v-u/2,\qquad r=p-u,\qquad h=q-v,\qquad C=\cos(u/2).
\]

The inherited storage is `H=F+V`, where `F=X^T G X/2` and

\[
\begin{aligned}
V(u,s,r,h)
 &=4[1-C\cos s]+1-\cos(2u)
      +2[1-\cos r]+2[1-\cos h],\\
V_0(u)&=5-4\cos(u/2)-\cos(2u).
\end{aligned}
\]

Thus `V>=V_0` whenever `C>0`. The transverse receiver and contact
coordinates are still dynamic; setting them to their minimizing values
in a bound does not set their actual rows to zero.

Define the full-state linear momentum

\[
P=6a+3b+2c+d.
\]

Adding the eight inherited rows with these coefficients cancels the
explicit contact currents and gives the exact identity

\[
\boxed{P'=-2[\sin(u/2)\cos s+\sin(2u)].}
\]

This cancellation retains the contact state in `P` and in the subsequent
evolution of `u,s`; it does not remove environmental feedback. Equivalently,
the constant mechanical mass in phase coordinates is `(D G D)^(-1)`.
After the displayed coordinate change, its first row is
`(53/2,17,6,5)`, so

\[
P=\frac{53}{2}u'+17s'+6r'+5h'.
\]

In particular, positive or increasing `P` alone does not mean that `u`
increases at every instant. The transverse velocities must be retained.

### A uniform force bound throughout the corridor

Choose declared real endpoints and a full-storage upper bound satisfying

\[
-\frac{2\pi}{3}<u_L<u_0<-\frac\pi2<u_R<0,
\qquad H\le\overline H<\min\{4,B(u_L)\},
\]

where

\[
B(u)=4+24\cos^4(u/2)-8\cos^2(u/2).
\]

On this interval `C>0`. The nonnegative form and contact stores give

\[
\cos s\ge\frac{5-\cos(2u)-\overline H}{4C}.
\]

Substituting this lower bound in the exact momentum row, with
`sin(u/2)<0`, yields

\[
P'\ge-\frac12\tan(u/2)[B(u)-\overline H]
 \ge m,
\qquad
m=-\frac12\tan(u_R/2)[B(u_L)-\overline H]>0.
\]

Here `B` is strictly increasing on `(-2pi/3,0)` because
`cos^2(u/2)>1/4`, while `-tan(u/2)` is positive and decreasing.
This is a nonlinear, full-state bound valid while `u` remains in the
closed corridor; it uses no local truncation. `V_0` is strictly decreasing
on the same interval. Indeed
`V_0'=2sin(u/2)[1+4cos(u/2)cos u]`, and the bracket increases from
zero when expressed as `1+8C^3-4C`, with `C>1/2`.

### A directional exit condition keeps the lower face closed

Let `l=(6,3,2,1)` and `k=(4,-1,-1,0)/3`, so `P=l^T X` and
`u'=k^T X`. The exact inherited form matrix gives

\[
l^{\mathsf T}G^{-1}l=\frac{53}{2},\qquad
k^{\mathsf T}G^{-1}k=\frac29,\qquad
k^{\mathsf T}G^{-1}l=1.
\]

Consequently `P^2<=53F`. For a prospective lower-face exit, a sharper
bound follows by setting `l_perp=l-(9/2)k`:

\[
l_\perp^{\mathsf T}G^{-1}l_\perp=22,\qquad
P=\frac92u'+l_\perp^{\mathsf T}X.
\]

At a first lower-face exit `u'<=0`. If `P>0`, Cauchy's inequality
therefore gives `P^2<=44F`. Suppose the actual initial momentum obeys

\[
\boxed{P_0>0,\qquad
P_0^2>44\max\{0,\overline H-V_0(u_L)\}.}
\]

Since `P` increases while the orbit remains in the corridor, this
inequality excludes a lower-face exit: there `F<=overline H-V_0(u_L)`
would contradict the necessary exit bound. If the latter upper bound
is negative, that face is already excluded by storage.

Meanwhile the entire closed corridor gives

\[
P\le P_{\max}
 =\sqrt{53[\overline H-V_0(u_R)]}.
\]

Combining this finite upper bound with `P'>=m` proves that the orbit
cannot stay in the corridor indefinitely. It must reach the upper face
within

\[
\boxed{T_{\rm exit}\le\frac{P_{\max}-P_0}{m}.}
\]

This bound concerns scaled time and the actual complete flow, not a
prescribed path in phase space. Its endpoints, energy and strict momentum
margin are prospective premises.

### Why this exit changes winding, and what it leaves open

Require initially that the four raw path gaps `v-u,-v,-v,v-u` lie
in `(-pi,pi)`. Since `2u_0` lies in `(-4pi/3,-pi)`, the closing edge
has one branch offset and the initial winding is `+1`. Each of the path
gaps occurs twice in the inherited potential, so reaching a principal
path seam would cost at least `4`, which `overline H<4` excludes.
The same bound excludes a contact seam, since each nonzero contact
coordinate also occurs twice. All these branch offsets therefore remain
fixed. At the upper face `2u_R` lies in `(-pi,0)`: the closing seam
has been crossed, the raw gaps still telescope, and the winding is `0`.
No assumption of monotone `u` was used.

The sufficient class is nonempty without solving an orbit. For example,
choose any permitted `u_0`, take `s=r=h=0`, and set

\[
X=U(2,2,3,5/2),\qquad U>0.
\]

These are supplied initial data, not maintained constraints. They give
`F=53U^2/4`, `P_0=53U/2` and `P_0^2=53F`; since
`V_0(u_0)<V_0(u_L)`, the strict lower-exit test follows automatically.
Any such preparation satisfying the declared upper-energy bound supplies
the directed passage. Because `V_0(u_0)<7/2<B(u_L)`, this family includes
energies strictly above `7/2`; a saddle-level or sub-saddle budget is not
required.

The retained rational illustration uses `u_L=-2`, `u_0=-9/5`,
`u_R=-3/2` and `U=1/12`: its reduced form is
`(1/6,1/6,1/4,5/24)`, its phase is `(-9/5,-9/10,-9/5,-9/10)`,
and its full storage is `V_0(-9/5)+53/576`, about `3.50233`.
The shared
[`assess_sine_saddle_corridor`](../../src/tnfr/physics/relational_sine_corridor.py)
re-admits the full state and signed family before applying these bounds;
`lower_phase` and `upper_phase` declare the two corridor faces.
It adds no solver, law, damping or live support event. A failed sufficient
inequality does not prove trapping. The proof concerns an exact correlated
signed-family source. A full independent source box additionally needs
the transverse error control of Section 48 and sufficient endpoint branch
margins; it is not contained in the eight-dimensional family.

This establishes a genuine nonlinear connection from the outer saddle
corridor to the winding seam. Its initial closing gap exceeds `2pi/3`,
so the source is outside `Omega_1` and is nonacute. Reversing the result
connects zero winding back to that outer corridor, not across the saddle
into an acute target. Joining this passage to the local sector crossing
and a finite-retention region requires one compatible full-state orbit;
separate existential passages cannot simply be concatenated. The stronger
formation-to-retention obligation is not settled by this corridor alone.
Section 50 supplies a compatible same-orbit construction under additional
quantitative premises.

## 50. Conservative formation and retention on the same full-state orbit

<a id="sine-conservative-formation-retention"></a>

There is a conditional existence construction for the strengthened
formation-to-retention question. It uses the same complete conservative
unit law and supplied C5/private-leaf support throughout. One exactly
specified near-saddle preparation connects both nonlinear corridors, rather
than treating their separate examples as one trajectory. Time reversal then
provides an initially zero-winding source, an actually reached target, and
acute retention for `T=1` in `tau=t/pi`. Positive uncertainty widths cover
every form and phase coordinate of the ten-node system.

This is an analytic existence theorem. The source and target centers below
are defined through the unique mathematical flow and finite hitting events;
their numerical coordinates have not been computed. It is neither an
evaluated frozen prediction nor a claim that a source is autonomously
selected, easy to prepare or physically identified.

### One exact preparation supplies both corridor admissions

Use the exact saddle and algebraic eigenvectors of Section 48, with
`v_lambda[0]=1`, and set

\[
0<\varepsilon\le2^{-32},\qquad
z(0)=z_*+\varepsilon(3X_\lambda,-v_\lambda).
\]

Let `z(tau)` be this one complete solution. Its real phases and forms are
exactly in the signed invariant family; no rounded value of `pi` or an
independent box around an eigenvector is substituted for that preparation.
Write `Q=X_lambda^T G X_lambda` and `P_lambda=l^T X_lambda`, with the
matrix and momentum row already defined. The exact rational root enclosure
of Section 48 implies

\[
1<Q<\frac65,\qquad P_\lambda>\frac{53}{10},\qquad
\|v_\lambda\|_\infty=1,\quad\|X_\lambda\|_\infty<1.
\]

The saddle is critical and `v_lambda^T B v_lambda=-Q`. The initial
form and quadratic phase terms therefore give `4Q epsilon^2` above its
storage. The third-order remainder of each of the ten edge potentials is
at most `(2epsilon)^3/6`. Consequently the **actual conserved storage** obeys

\[
H=\frac72+4Q\varepsilon^2+R_H,
\quad |R_H|\le\frac{40}{3}\varepsilon^3,
\quad
\boxed{\frac72<H<\overline H:=\frac72+5\varepsilon^2
 <\frac{35001}{10000}.}
\]

For either time sign at `|tau|=3`, the full nonlinear comparison error
of Section 48 is at most

\[
\rho=9\cdot729\cdot728\,\varepsilon^2<\frac{\varepsilon}{512}.
\]

Put `E=exp(3sigma)`. Since `sigma>7/25`, the cubic lower Taylor sum
for `exp(21/25)` gives the rational inequalities

\[
E-2/E>7/5,\qquad E+2/E>79/25,
\qquad E^{-1}-2E<-4,\qquad E^{-1}+2E>9/2.
\]

The tangent expressions, with the retained error `rho`, now certify the
following properties of **actual states of the same solution**:

| State | Reaction coordinate | Directed momentum |
| --- | --- | --- |
| `z(3)` | `u>u_*+epsilon` | `P>(33/2)epsilon` |
| `R z(-3)`, where `R(x,theta)=(-x,theta)` | `u<u_*-epsilon` | `P<-23epsilon` |

For example, the outer momentum is bounded below by
`epsilon*(53/10)*(79/25)-12rho>(33/2)epsilon`; the coefficient `12`
is the sum of the absolute entries of `l`. The analogous bound with
`9/2` gives the inner negative momentum after time reversal. The global
bound `||z(tau)-z_*||_infinity<2187epsilon` for `|tau|<=3` also places
these states strictly inside their declared coordinate strips and preserves
all their initial path branches. Both remain nonacute at these local times.

At `u_*= -2pi/3`, `V_0'=0`, `V_0''=-3/2`, and
`|V_0'''|<=17/2`. Taylor's inequality gives

\[
V_0(u_*\mathbin{\pm}\varepsilon)\ge\frac72-\varepsilon^2,
\qquad
44[\overline H-V_0(u_*\mathbin{\pm}\varepsilon)]
 \le264\varepsilon^2.
\]

This last upper bound is strictly smaller than both retained momentum
squares. Thus the actual local states pass the directional face tests;
the argument does not reset their form, contact phase or conserved storage
to a new example.

### Outer and inner nonlinear connections

For `z(3)`, use Section 49 with
`u_L=u_*+epsilon`, `u_R=-3/2`. On `[u_*,-pi/2]`,
`B'(u)>=sqrt(3)`, so `B(u_L)-overline H>epsilon`. The positive
force is greater than `epsilon/3`. The inherited Cauchy bound gives
`|P|^2<=53H<196`, hence `|P|<14`. The same orbit reaches the upper
face `u=-3/2`, and therefore `W=0`, within a further `42/epsilon`
scaled time units. Denote one such finite upper-face time by `t_o`.

For the inner direction, evolve the complete reversed solution
`w(tau)=R z(-tau)`, starting with its actual state at `tau=3`. Use
the strip

\[
u_A=-5/2\le u\le u_* -\varepsilon.
\]

Here `cos s<=1` in the exact momentum law gives
`P'<=-V_0'(u)<0`. The function `V_0'` is concave on this strip:
`V_0'''=-sin(u/2)/2-8sin(2u)<0`. Its left endpoint satisfies
`V_0'(-5/2)>1/100>epsilon`; at the right endpoint,
`V_0''<=-1` near `u_*` gives `V_0'(u_*-epsilon)>=epsilon`.
Thus `P'<=-epsilon` throughout the strip.

At a prospective upper-face exit `u'>=0`, the same orthogonal decomposition
`P=(9/2)u'+l_perp^T X` gives `P^2<=44F` when `P<0`.
The local momentum bound excludes that exit. Since `|P|<14`, the solution
must instead reach `u_A=-5/2` within a further `14/epsilon` time units.
Let `t_d` be its first such deep hit. The paired path phases remain on
their inherited branches because `H<4`; contacts remain fully dynamic.

The resulting forward formation orbit is

\[
\chi(s)=R z(t_o-s),\qquad s\ge0.
\]

It begins with `W=0`, passes the same near-saddle preparation, and reaches
the deep state at `s=t_o+t_d`. Its elapsed time to any earlier checkpoint
is less than `6+56/epsilon<64/epsilon`. This is a conservative analytic
upper bound, not a computed hitting time or a proposed simulation horizon.

### An explicit acute band supplies a unit of retained evolution

Consider the closed phase band

\[
-13/5\le u\le-12/5,
\qquad H\le35001/10000.
\]

It contains the deep hit `u=-5/2`. A paired receiver gap reaching its
acute face has minimum cycle potential at fixed `u`

\[
A(u)=\frac72+2[\sin u+1/2]^2.
\]

This follows by setting one of the two distinct path gaps to `pi/2`
in the exact potential; a negative path face is incompatible with the
other path gap remaining acute on this lifted branch. On the displayed
band the minimum is at `u=-13/5`. A rational Taylor lower bound
`sin(13/5)>103/200` proves

\[
A(u)-H>\frac1{3000}.
\]

Since `|partial V_R/partial s|<=4`, each paired gap has acute margin
greater than `1/12000` throughout the admitted band component. The closing
gap is `2pi+2u` and its acute margin is at least
`24/5-3pi/2>1/12000`. These estimates both prove acuteness at the deep
state and prevent an acute face from being crossed before a band exit.
The path to the deep state inherits this component: between the saddle
neighborhood and `u=-5/2`, its paired acute-face cost also exceeds `H`.

While acute, `V_R>=V_5=5(1-cos(2pi/5))`. The exact form metric gives

\[
|u'|^2\le\frac49F\le\frac49(H-V_5)<\frac1{49}.
\]

For example, `sqrt(5)<161/72` and `H<=35001/10000` reduce the last
strict inequality to the rational comparison
`13/288+1/10000<9/196`. Starting at the deep state, either band face
is `1/10` away. A first exit in either time direction must therefore take
more than `7/10` scaled time units. In particular, the complete nominal
orbit on the interval from one half-unit before the deep hit to one
half-unit after it is acute with every edge margin greater than `1/12000`.

Define the target checkpoint at
`S=t_o+t_d-1/2` along `chi`. It is later than the initial zero-winding
source, and its next full scaled unit stays inside that acute band.

### Positive widths include all twenty form and phase coordinates

Let the target be the **exact real** maximum-norm ball of radius

\[
\delta_T=2^{-18}
\]

around the actual state `chi(S)`, with support and unit capacities fixed.
The complete field is globally Lipschitz with constant two, regardless
of whether an uncertain state belongs to the signed family. Over `T=1`,
each full-state error is at most `delta_T exp(2)`, and an edge phase error
is at most twice that. Since

\[
2\delta_T e^2<18\delta_T<\frac1{12000},
\]

every independent target member retains strict acute winding `+1` for
the full unit interval. This includes form and phase perturbations at
all receiver and environmental nodes; it does not impose symmetry on them.

At the initial source `chi(0)`, the closing gap has principal-seam margin
`pi-3>1/10`. A paired path gap within `1/10` of a seam would cost
`2[1+cos(1/10)]>H`, so each of those margins also exceeds `1/10`.
Take the full independent source ball of exact radius

\[
\boxed{\delta_S=2^{-19}\exp(-128/\varepsilon)>0.}
\]

Every source member still has `W=0`. Since `S<64/epsilon`, its error
relative to the same nominal orbit at the target checkpoint is strictly
less than `2^-19`, which lies inside the radius-`delta_T` target.
Thus **one full-width zero-winding source reaches one full-width
acute-retention target under the unchanged complete law**. The target
ball need not equal the reachable image; inclusion is the proved claim.

The source-width bound is extremely small even at the largest admitted
`epsilon`. It is a formal positive robustness radius, not evidence of
practical preparation tolerance. Its logarithm or exact analytic expression
must be retained; replacing an underflowed floating value by zero would
discard the theorem's premise. The unknown numerical hitting states must
likewise not be replaced by nearby hand-selected snapshots.

The shared
[`assess_sine_saddle_formation`](../../src/tnfr/physics/relational_sine_corridor.py)
reports this scoped analytic family, and `assess_sine_saddle_retention_band`
owns the admitted band estimate. The complete source law and support are
re-admitted; a supplied snapshot is not automatically either constructed
hitting state. No trajectory is evaluated by this proof reader.

This closes the mathematical **conditional existence** obligation for
formation followed by `T=1` finite retention on this supplied support.
It does not select an autonomous preparation or microscopic law, establish
indefinite lifetime, compute an operational source/target pair, or identify
the resulting pattern with a physical constituent. Those are distinct
questions from the existence established here.

## 51. Rational preparation and retained-metric propagation

<a id="sine-operational-saddle-preparation"></a>

Section 50 establishes a formal source-to-target connection. Making that
construction operational requires actual represented preparation coordinates
and a validated propagation that preserves enough information to reach its
hitting states. These are numerical obligations under the existing complete
law, not missing pressure terms or new constitutive parameters. Increasing
arithmetic precision and proving tolerance to physical preparation error
are separate questions.

### A rational preparation must retain the same finite gates

The shared
[`prepare_sine_saddle_state`](../../src/tnfr/physics/relational_sine_corridor.py)
constructs a rational midpoint of an enclosure of Section 50's correlated
algebraic preparation. It re-admits that actual rational center, including
its complete form, phase, support and conserved storage. The input source
anchors the law and support; it is not relabeled as the constructed state.

If the rational center differs from the exact correlated preparation by
at most `r` in the full maximum norm, the complete-field Lipschitz bound
adds at most `exp(6)r<729r` to the local comparison at either `tau=3`
or `tau=-3`. Thus the total normalized local error becomes

\[
\eta_{\rm loc}
 =9\cdot729\cdot728\,\varepsilon
       +\frac{729r}{\varepsilon}.
\]

The two phase-displacement and two momentum inequalities of Section 50
must be recomputed with this retained error. Exact signed-family membership,
initial winding and branch admission are checked again; a component interval
containing the desired value is not itself an exact symmetry proof. The
actual center's energy must independently satisfy
`7/2<H<7/2+5epsilon^2`. Using the shared outward rational trigonometric
arithmetic preserves the small difference above the saddle budget; rounding
that difference away with a binary floating cosine would not establish
either inequality.

If the retained precision cannot separate the actual center's storage from
`7/2`, preparation admission remains unavailable even when the ideal
analytic family exists. The reader does not replace that unresolved
represented state by the exact theorem's preparation.

This supplies usable rational coordinates for the **near-saddle**
preparation and retains its conditional forward and backward corridor gates.
When those checks pass, the same-orbit corridor and finite-retention argument
applies to the actual rational preparation, with its retained errors and
budget; the proof does not continue from an ideal center instead.
The preparation reader alone does not supply the zero-winding source
coordinates, a computed hitting time or a validated future response. The
rounding allowance `r`
concerns computational representation; it must not be silently substituted
for a laboratory preparation tolerance.

### Why independent coordinate boxes lose essential cancellations

At the exact saddle, in scaled time, the full twenty-coordinate Jacobian is

\[
J=\begin{pmatrix}0&-K H_\theta\\K L&0\end{pmatrix},
\]

where `H_theta` is the actual sine-potential Hessian. Its only expanding
eigenvalue is `sigma<1/3`; it also has the oscillatory pairs and two
conserved-origin modes identified in Section 48. The current componentwise
Taylor comparison replaces off-diagonal derivatives by absolute bounds.
Since the diagonal entries of `J` vanish, its exact-saddle twenty-coordinate
comparison block is `|J|`.

Take the positive comparison vector with all form entries `1` and all
phase entries `sqrt(2)`. Every row of `|K L|` sums to `2`. The row sums
of `|K H_theta|` are `1` at the two receivers touching the negative-cosine
edge, `4/3` at the other receivers, and `2` at the leaves. Consequently

\[
|J|\begin{pmatrix}\mathbf1\\\sqrt2\,\mathbf1\end{pmatrix}
 \ge\sqrt2
 \begin{pmatrix}\mathbf1\\\sqrt2\,\mathbf1\end{pmatrix},
\qquad \rho(|J|)\ge\sqrt2.
\]

Thus a rigorous independent-box comparison can grow much faster than the
actual tangent. The common-origin vectors are an especially clear example:
the signed field cancels their contributions, while its absolute comparison
does not. This is a limitation of the enclosure representation, not evidence
of an additional unstable nodal mode or an error in the completed finite
forecasts. Their retained enclosures and negative results remain unchanged.

### An exact rational metric retains all relative coordinates

No arbitrary transverse error may be dropped by replacing the full state
with the eight-dimensional signed family. Instead retain both conserved
weighted means and all nine relative coordinates for each field.
Let `d=K^(-1)1` be the full degree vector, and choose a rational full-rank
matrix `U` whose columns span `d^perp`. For example, with a private leaf
last, take `U_i=e_i-(d_i/d_last)e_last` for the other nine nodes. Write

\[
x=\bar x\mathbf1+Ua,\qquad
\theta=\bar\theta\mathbf1+Ub,
\qquad
\bar x=\frac{d^{\mathsf T}x}{d^{\mathsf T}\mathbf1},\quad
\bar\theta=\frac{d^{\mathsf T}\theta}{d^{\mathsf T}\mathbf1}.
\]

Both origins are retained constants, not discarded observations. Define

\[
G=U^{\mathsf T}LU,\qquad
B=U^{\mathsf T}H_\theta U,\qquad
D=(U^{\mathsf T}K^{-1}U)^{-1},\qquad
M=(DGD)^{-1}.
\]

These are the full relative matrices, distinct in dimension from the
four-coordinate matrices of Section 48. The relative tangent is exactly
`a'=-DBb`, `b'=DGa`. All matrices are rational at the exact saddle.
The following symmetric positive-definite metric requires no fitted
eigenvector basis or numerical Lyapunov-equation solution:

\[
W_{\rm rel}=\operatorname{diag}\left(G,\,B+\frac16M\right).
\]

For `r=1/3`, the Schur complement of
`2rW_rel-(J_rel^T W_rel+W_rel J_rel)` is

\[
\frac23\left(B+\frac5{48}M\right).
\]

Exact rational positive-definiteness checks therefore prove

\[
J_{\rm rel}^{\mathsf T}W_{\rm rel}
 +W_{\rm rel}J_{\rm rel}\preceq\frac23W_{\rm rel}.
\]

Changing `J_rel` to `-J_rel` changes the off-diagonal Schur factors' signs
but not their product, so the same inequality holds for reverse time.
Add a positive unit weight for each conserved origin and pull the metric
back through the exact coordinate transformation. This gives a rational
positive-definite matrix `W` on **all twenty** original coordinates.
It contains the even perturbations, both origins and every live contact.
The induced error norm is `||h||_W=sqrt(h^T W h)`.
This metric controls numerical sensitivity; it is not an added conserved
storage, physical observable or constitutive law.

### A finite-neighborhood nonlinear bound

Suppose each primitive phase stays within `rho` radians of the declared
saddle phase lift. Every edge phase difference then departs by at most
`2rho`. Since cosine is globally one-Lipschitz, the change `Delta B` in
the full relative phase Hessian obeys

\[
-2\rho G\preceq\Delta B\preceq2\rho G.
\]

The numerical owner verifies the further exact matrix inequalities

\[
2G-GDG\succeq0,\qquad B+\frac1{12}M\succeq0.
\]

The first is the normalized-Laplacian bound in these coordinates. The
second, together with the definition of `W_rel`, gives
`W_phase>=M/12`. Factor
`Delta B=G^(1/2) C G^(1/2)`, where `||C||_2<=2rho`. Then

\[
\|G^{1/2}DG^{1/2}\|_2\le2,\qquad
\|G^{1/2}W_{\rm phase}^{-1/2}\|_2\le\sqrt{48}.
\]

Only the form-from-phase block of the Jacobian changes. Its induced metric
norm is consequently at most

\[
2(2\rho)\sqrt{48}=16\sqrt3\,\rho<28\rho.
\]

The same full-state metric therefore yields the local nonlinear logarithmic
growth bound

\[
\boxed{\gamma_W\le\frac13+28\rho.}
\]

For instance `rho=1/1000` gives a bound below `0.362` in scaled time,
while retaining all coordinates. The radius is a declared proof domain,
not a new physical constant. This is not a global hyperbolic-rate theorem:
the relevant whole-time phase enclosures must stay in that neighborhood.
For two complete flows whose phases remain there, the segment between them
also stays in the phase box, giving
`||h(tau)||_W<=exp(gamma_W*|tau|)||h(0)||_W`.
A numerical center with a residual requires its separate inhomogeneous
error integral; the homogeneous estimate does not cover truncation error
by itself.

### Conversion to primitive coordinates and execution boundary

The exact full metric supplies conservative conversion factors

\[
\|h\|_W\le
 \sqrt{\sum_{i,j}|W_{ij}|}\,\|h\|_\infty,
\qquad
|h_i|\le\sqrt{(W^{-1})_{ii}}\,\|h\|_W.
\]

The shared
[`assess_sine_saddle_sensitivity`](../../src/tnfr/physics/relational_sine_sensitivity.py)
re-admits the law and support and checks the rational matrix premises,
both time signs and these conversions. The captured source is a support
anchor, not evidence that a trajectory stays near the saddle. This reader
neither advances state nor produces an ellipsoid-based validated forecast.

### A validated step retains the metric radius

The shared
[`validated_metric_taylor_step`](../../src/tnfr/mathematics/_validated_metric.py)
now propagates the exact real set

\[
\mathcal E(c,r)=\{c+e:e^{\mathsf T}We\le r^2\},
\]

with a rational center `c`, rational positive-definite `W` and nonnegative
radius `r`. Its enclosing coordinate box uses outward bounds on
`sqrt((W^(-1))_ii)r`. That box is used for strict Picard inclusion and
whole-time derivative and domain bounds; it does not replace the retained
metric ball. In particular, the next step does not reconvert that enclosing
box into a new metric radius.

Let `h>0` be the step duration in the declared execution clock. Strict Picard
inclusion must enclose the exact center flow and every flow starting in
`E(c,r)`. The model-specific growth inequality must hold throughout the
resulting convex tube, including the segments between those flows. If its
logarithmic bound is `gamma`, their separation at the endpoint obeys

\[
\|\Phi_h(c+e)-\Phi_h(c)\|_W\le e^{\gamma h}r.
\]

Compute an interval Taylor polynomial from the exact center and its full
order-`p+1` remainder over the Picard tube. Let `C_i` be the resulting
enclosure of the exact center endpoint, let `c_i^+` be its exact rational
midpoint and let `a_i` bound the distance from that midpoint to both
materialized endpoints. Thus `a_i` includes Taylor truncation, interval
arithmetic and any outward materialization of a nondyadic initial center.
The local endpoint error satisfies

\[
\|\Phi_h(c)-c^+\|_W^2
 \le\sum_{i,j}|W_{ij}|a_i a_j\le\ell^2
\]

when `ell` is chosen as an outward upper bound on the displayed square root.
The triangle inequality therefore proves the retained-ball update

\[
\boxed{\Phi_h\bigl(\mathcal E(c,r)\bigr)
 \subseteq\mathcal E\left(c^+,\,e^{\gamma h}r+\ell\right).}
\]

The local remainder is an endpoint error, so it is added after propagation
of the initial radius. Every scalar exponential, square root and final
radius is rounded outward. The current exact exponential evaluation admits
`abs(gamma*h)<=1`; this is a numerical work limit, not a restriction of the
mathematical inclusion. The generic kernel checks positive definiteness,
Picard inclusion and the arithmetic. The caller remains responsible for
the supplied logarithmic-norm theorem on the whole tube.

The retained endpoint ball can contain unreachable states or extend beyond
the previous Picard tube. The next step therefore checks its whole initial
projection again. Intersecting an observation box with an earlier tube
would not justify shrinking this independent ball. Failed domain or
remainder admission leaves only the already validated prefix available.

### Sine execution with retained full-state uncertainty

[`forecast_sine_saddle_metric`](../../src/tnfr/physics/relational_sine_metric_forecast.py)
re-admits the complete unit-capacity C5/private-leaf law, reconstructs the
metric and advances the actual captured source with the shared sine field.
It retains all twenty coordinates in source-node order: first every form,
then every primitive phase. The common origins remain present, and no exact
signed-family assumption is imposed on the uncertainty.

The API declares elapsed original structural time `t`. In its fixed
saddle-domain mode, `tau=t/pi` gives the outward original-clock growth bound

\[
\gamma_t=\frac{1/3+28\rho}{\pi_{\rm lower}}.
\]

Both forward and reverse field directions use the corresponding proved
metric inequality. In that mode, every whole-time phase interval is compared with an
outward enclosure of its exact saddle lift `2*pi*turn_i`; neither a rounded
value of pi nor an implicit phase wrap substitutes for this real-lift
domain check. The positive domain margins apply to the nominal and
perturbed flows together.

An independent initial coordinate cube is enclosed in the metric ball once.
Successive steps carry only the exact recentered state and retained radius
in the same metric. Coordinate endpoint intervals are observations of that
ball, not the uncertainty representation consumed by the next step.
Declared step, order, domain policy and work budget remain fixed: a failure
does not authorize a larger domain or an undeclared retry.

### Certifying growth on a full Picard tube beyond the saddle neighborhood

The metric `W` itself is constant and positive definite on the entire
twenty-coordinate state space. Only the previously supplied bound
`1/3+28rho` depends on remaining near the saddle. A direct Jacobian
inequality on each complete Picard tube can replace that local bound
without changing the metric, law or uncertainty representation.

Let `J(z)` be the Jacobian of the declared execution field on such a tube.
Enclose every entry of the symmetric matrix

\[
S(z)=J(z)^{\mathsf T}W+WJ(z)
\]

by symmetric rational intervals. Write their exact midpoint matrix as `M`
and their symmetric nonnegative radius matrix as `R`. For every vector `v`,

\[
\begin{aligned}
v^{\mathsf T}[S(z)-M]v
 &\le\sum_{i,j}R_{ij}|v_i v_j|\\
 &\le\sum_i\left(\sum_jR_{ij}\right)v_i^2.
\end{aligned}
\]

Consequently the single exact matrix inequality

\[
\boxed{2\gamma W-M-
 \operatorname{diag}(R\mathbf1)\succeq0}
\]

proves `S(z)<=2gamma W` at every point of the whole tube. It is a nonlinear
flow bound because all possible Jacobians on that tube are enclosed, rather
than just the Jacobian at its midpoint. Interval correlations discarded in
forming the majorant can weaken the estimate, but cannot strengthen it
without proof.

The shared step can select this route with `growth_rate=None`, a declared
`growth_rate_bounds` bracket and a fixed `growth_bisections` count. The sine
adapter's default bracket is `(0,1)` with eight exact positive-semidefiniteness
bisections. The upper endpoint must first pass the matrix test; failure
leaves the step unavailable, without expanding the bracket or changing the
time step. Each selected rate and its matrix premises belong to the
retained step evidence. This deterministic numerical policy is fixed before
evaluating the reserved response; it does not fit or alter the nodal law.

For this sine model, `W` is block diagonal in form and phase, while `J`
and hence `S` have only off-diagonal form/phase blocks. Conjugation by
`diag(I,-I)` leaves `W` and the diagonal radius inflation unchanged and
changes `M` to `-M`. Thus a successful matrix test also supplies the
corresponding reverse-field inequality. This shortcut uses the admitted
block structure; it is not a property of an arbitrary supplied Jacobian.

The sine adapter selects the tube-derived mode with `phase_radius=None`.
It rebuilds the same full-state metric, then admits arbitrary primitive
phase lifts under the globally smooth, fixed-support conservative sine law.
Every actual tube still needs strict Picard inclusion and the matrix
certificate. The saddle's primitive-phase radius is no longer a consumed
growth premise. Since the shared automatic differentiation now acts on the
**original-time** field, its `J`, `S` and certified `gamma` already contain
the clock factor `1/pi`; dividing the selected rate by pi again would be
incorrect. The earlier fixed-domain mode keeps its separate clock conversion.

Across successful steps, the propagated part of the uncertainty is bounded
by `exp(sum_k gamma_k h_k)` in the same metric, with each local endpoint
error added by the retained-radius recurrence. There is no need to switch
metrics or turn a coordinate observation box into an independent source
when the trajectory leaves the small saddle neighborhood.

### Retained finite connection and its preparation boundary

A forward endpoint is wholly at zero winding only if every receiver edge
has a certified unique principal branch and the branch sum gives `W=0`.
On a backward leg from the same admitted preparation, acute retention
requires a contiguous collection of **whole-time** tubes with all receiver
gaps strictly in `(-pi/2,pi/2)`, fixed winding `+1` or `-1`, and total
original-time length at least an outward upper bound on pi. A sampled
event midpoint cannot establish the required scaled duration `T=1`.
Time reversal relates those two legs along the same complete flow,
including each admitted perturbed preparation.

The frozen [declaration](../../docs/assets/sine_metric_connection/declaration.json)
uses the rational preparation with `epsilon=2^-32`, independent errors
`2^-100` in all twenty coordinates, both time directions, original-time
horizon `256`, unit steps, Taylor order `16`, and the declared `(0,1)`
growth bracket with eight bisections. Its first-endpoint/first-window rule
was fixed before evaluation. The
[retained evidence bundle](../../docs/assets/sine_metric_connection/response-v1.evidence.zip)
and [manifest](../../docs/assets/sine_metric_connection/response-v1.manifest.json)
preserve the source, protocol, full response and verdict.

| Evaluated obligation | Retained result in original structural time |
| --- | --- |
| Forward zero-winding endpoint | The first wholly certified `W=0` endpoint occurs at `t=237` |
| Forward declared horizon | The prefix through `t=240` is admitted; the next step fails the declared growth-bracket upper endpoint `1` |
| Backward declared horizon | All `256` units are admitted |
| Backward whole-time acute identity | The first retained `W=+1` window is `[230,234]`, with acute margin greater than `0.01001717` radians |
| Combined verdict | Connection is certified on the validated prefixes; completion of both declared horizons is false, so the protocol verdict is `passed=false` |

Source checks before and after evaluation agree, and no evaluation error
is recorded. The forward stopping reason is
`declared_growth_upper_endpoint_not_certified`. It does not invalidate
earlier tubes or prove a dynamical singularity; the declared sufficient
matrix bound was unavailable on the next tube. The bracket, horizon and
preparation were not retuned after that result.

Let `Phi_t` denote the original-time flow and `R(x,theta)=(-x,theta)` its
reversing symmetry. For every initial state `z` in the admitted preparation,
the source `R Phi_237(z)` is at zero winding. Its forward evolution obeys

\[
\Phi_s R\Phi_{237}(z)=R\Phi_{237-s}(z).
\]

Consequently it retains acute winding `+1` throughout `s in [467,471]`.
The certified duration is `4/pi>1` in scaled time. This is a finite
**mapped source-family** connection under the supplied complete law, with
all environmental coordinates retained. The initial independent cube was
prepared near the winding-one saddle; its image is not an independently
prepared zero-winding coordinate ball.

### An inverse-inclusion test separates an image from a preparation

The same retained step evidence supplies a sufficient inverse-inclusion
test without a new trajectory. Write `g_k` for each outward metric growth
factor and `ell_k` for its local endpoint error. Starting with `P_0=1` and
`eta_0=0`, propagate

\[
P_{k+1}=g_kP_k,\qquad
\eta_{k+1}=g_k\eta_k+\ell_k.
\]

Rounding these nonnegative products and sums upward retains valid upper bounds.
Thus `eta_F` bounds the exact nominal endpoint's distance from the retained
rational endpoint center `c_F`. The two-sided metric inequality also gives
`||Phi_A(z)-Phi_A(c)||_W >= ||z-c||_W/P_F` for admitted trajectories.
On the boundary of the initial metric ball of radius `r`, the image is
therefore at distance at least `r/P_F` from `Phi_A(c)`.

The complete sine field is globally Lipschitz on the full real lift, so its
finite-time flow is a global diffeomorphism. Its image boundary is the image
of the initial boundary. A segment leaving that image from `Phi_A(c)` must
meet this boundary; the distance bound forbids such a crossing inside the
radius-`r/P_F` ball. Closure includes its boundary. The triangle inequality
then proves

\[
\boxed{\mathcal E(c_F,q)\subseteq\Phi_A(\mathcal E(c,r)),
\qquad q=\frac r{P_F}-\eta_F>0.}
\]

This argument uses the already admitted forward trajectories and global
invertibility, rather than assuming a candidate inverse trajectory stays
inside an unverified domain. The reversing symmetry is an isometry of the
block-diagonal metric. A positive `q` would therefore supply an explicit
reflected source ball, and a coordinate cube of radius at most
`q/sqrt(sum_ij|W_ij|)` inside it.

The solver propagates an enclosing initial metric ball, which is larger
than the originally declared coordinate cube. To require preimages inside
that cube of radius `delta`, use instead its admitted metric ball of radius
`r_in=delta/max_i sqrt((W^(-1))_ii)`, with an outward denominator. These are
different preparation claims even though the same tubes can bound both.

The [exact postprocessing record](../../docs/assets/sine_metric_connection/derived-inclusion-v1.json)
binds the retained response digest and recurrence, without a new forecast.
Postprocessing the retained data through the first zero-winding endpoint
gives outward bounds approximately `P_F=7.60523305384e10` and
`eta_F=7.36366081855e-13`. Both the enclosing-ball and inscribed-ball
expressions for `q` are negative. Thus this sufficient test supplies **no
independent zero-winding preparation ball**. It neither disproves the mapped
connection nor establishes that an independent source ball is dynamically
impossible. The finite connection, the incomplete forward horizon and the
unavailable operational source width remain separate conclusions.

The [local-agreement/global-storage discriminator](SINE_CONSTITUTIVE_INFORMATION.md#local-phase-storage-nonselection)
keeps this frozen response intact while testing which constitutive premises
its winding passage consumes. Agreement with the preparation, local saddle
motion or acute geometry does not alone select a global pressure law.

<a id="sine-constitutive-robustness"></a>
## 52. A constitutive change preserves equilibria but blocks the same source's formation

The declared comparison `eta=1/100` gives an energetic obstruction, not
merely an unavailable trajectory-error estimate. The same source that
forms and retains acute winding under sine lies below the changed law's
acquisition barrier. This result reuses the retained sine response; it
requires no new trajectory, altered preparation or coefficient search.

### The complete changed law and fixed source

The [cubic-storage completion](SINE_CONSTITUTIVE_INFORMATION.md#phase-storage-selection-boundary)
keeps the C5/private-leaf support, held unit capacities and phase row:

\[
\begin{aligned}
j_\eta(\delta)&=\sin\delta+\eta\sin^3\delta,\\
U_\eta(\delta)&=1-\cos\delta+
 \eta\left(\frac23-\cos\delta+\frac{\cos^3\delta}{3}\right),\\
x'&=KS_\eta(\theta),\qquad \theta'=KLx,
\qquad S_{\eta,i}=\sum_{j\sim i}j_\eta(\theta_j-\theta_i).
\end{aligned}
\]

Primes denote `tau=t/pi`; there are no inputs, events or discarded
coordinates. For `eta>=0`, the changed nonnegative storage
`H_eta=x^T Lx/2+sum_e U_eta(delta_e)` is conserved by its own reciprocal
rows. The sine storage is not conserved by this changed field in general.
The fixed scale `eta=1/100` bounds the current change by one percent of
normalized unit current. It is a declared robustness question, not a
physical constant or a replacement engine default.

Use exactly the [retained source family](#sine-operational-saddle-preparation)

\[
p_z=R\Phi^{\sin}_{237}(z),\qquad R(x,\theta)=(-x,\theta),
\]

where `z` belongs to the frozen intermediate preparation. Flow times in
this source definition are original structural time. Both laws start at
this same exact `p_z`, whose zero winding is certified by the retained
endpoint enclosure. Its sine reference path is

\[
y_z(s)=R\Phi^{\sin}_{237-s}(z),\qquad 0\le s\le471.
\]

The whole sine interval `[467,471]` is acute with winding one, of scaled
length `4/pi>1`. Its retained tubes are forward steps `236,...,0`
traversed in reverse and reflected, then backward steps `0,...,233`
reflected. Preparing `R Phi_eta,237(z)` instead would change the source.
An independent rectangular source ball or positive inverse-inclusion
radius is unnecessary for this comparison of the same exact mapped
states; the comparison does not supply either admission.

### The exact changed acquisition barrier

Keep the entire sector from [Section 42](SINE_REGIONAL_FORMATION.md#sine-cycle-sector-barrier):

\[
\Omega_\sigma=\{W=\sigma,\ |\delta_i|<2\pi/3\text{ for all five edges}\},
\qquad \sigma\in\{-1,1\}.
\]

Every acute state of winding `sigma` belongs to this sector. For every
`eta>=0`, the map `s -> s+eta*s^3` is strictly increasing. Consequently
`j_eta(t)-j_eta(pi/3)` is nonpositive on `[-2pi/3,pi/3]` and nonnegative
on `[pi/3,2pi/3]`. Integrating gives the global tangent bound on this
whole interval,

\[
U_\eta(t)\ge U_\eta(\pi/3)+j_\eta(\pi/3)(t-\pi/3).
\]

On the positive-winding boundary with one gap `2pi/3`, the other four sum
to `4pi/3`. Their linear terms therefore cancel, proving

\[
\boxed{B_\eta:=\min_{\partial\Omega_\sigma}\sum_iU_\eta(\delta_i)
 =U_\eta(2\pi/3)+4U_\eta(\pi/3)
 =\frac72+\frac{47}{24}\eta.}
\]

Equality is attained by the five permutations of
`(2pi/3,pi/3,pi/3,pi/3,pi/3)`. A gap `-2pi/3` forces the other four to
be `2pi/3`, and has the strictly larger cost `5U_eta(2pi/3)`.
Evenness gives the negative-winding result. No nonnegative-gap
restriction or small-eta approximation enters this minimum.

For each finite `eta`, the periodic current is bounded, so form grows at
most linearly on bounded time intervals; the linear phase row then also
remains finite. The smooth full flow therefore exists in both time directions.
All other full-system storage is nonnegative. Thus `H_eta<B_eta`
forbids sector-boundary crossing in either time direction. Equality
also forbids a finite crossing: it forces constant form and zero contact
storage, while all cycle currents at the minimizing geometry are equal.
The entire state is then stationary, and uniqueness excludes reaching
or leaving it at finite time from a different state. Therefore a source
outside both sectors with `H_eta<=B_eta` cannot acquire acute winding
`+1` or `-1` at any time. This does not prohibit nonacute winding outside
these sectors or identify every possible future state.

### Applying the barrier to the retained source

Define the nonnegative added phase storage on **all ten edges**,

\[
A(\theta)=\sum_e\frac{(1-\cos\delta_e)^2(\cos\delta_e+2)}3.
\]

Sine conservation and form reversal give the exact source identity

\[
H_\eta(p_z)=H_0(z)+\eta A(\theta(p_z)).
\]

Here `H_0(z)` is bounded from the original preparation, and `A` is bounded
from the complete retained endpoint at original time `237`. Neither term
uses the changed law's unknown response. The
[retained evidence](../../docs/assets/sine_metric_connection/response-v1.evidence.zip)
contains the initial full-state enclosure and forward endpoint `236`.
The [constitutive comparison record](../../docs/assets/sine_metric_connection/derived-constitutive-v1.json)
binds those retained premises without modifying the frozen response.
Outward evaluation gives, conservatively rounded,

\[
1.71785519306<A(\theta(p_z))<1.71785519309,
\]
\[
H_{1/100}(p_z)<3.51717855194
 <B_{1/100}=\frac72+\frac{47}{2400},
\qquad B_{1/100}-H_{1/100}(p_z)>0.00240478139.
\]

The full source enclosure has winding zero. Thus the declared cubic law
cannot acquire either acute unit-winding identity from any member of this
same source family, although sine certifies the retained finite window.
This is an all-time acquisition obstruction under the changed complete
law, not evidence that an already formed identity disappears, that no
other preparation can form it, or that the sine result was incorrect.

The sharp budget comparison can also be written

\[
B_\eta-H_\eta(p_z)=
 \eta\left(\frac{47}{24}-A(\theta(p_z))\right)
 -\left(H_0(z)-\frac72\right).
\]

It explains the sensitivity: the sine construction uses a very small
excess above its own saddle barrier, while the changed law raises that
barrier more than it raises this fixed source's storage. A positive
budget margin above a law's barrier is necessary for crossing, not a
sufficient formation theorem. This identity supplies no instruction to
retune the source until a preferred outcome passes.

### The complete critical geometry and its inertia remain unchanged

The effect is not removal of the critical patterns. On this connected
support, equilibrium requires constant form. Each private leaf has one
neighbor and must have contact phase `0` or `pi` modulo `2pi`. Cycle
stationarity then requires one common oriented value of `j_eta(delta)`.
Its strict monotonicity as a function of `sin(delta)` makes this exactly
the sine condition: all oriented cycle sines have the same value `s`.
Thus the entire critical phase set is unchanged for every `eta>=0`.

At any such critical state the cycle Hessian weights are

\[
U_\eta''(\delta_i)=(1+3\eta s^2)\cos\delta_i,
\]

while contact Hessian weights remain `+1` or `-1`. Change phase
coordinates to the receiver phases and the five leaf-minus-receiver
contrasts. The phase quadratic splits into a positive scalar
`1+3eta*s^2` times the original cycle quadratic plus the five unchanged
signed contact squares. This is a congruence argument, so the full phase
Hessian has the same inertia, including nullity. The full Hessian is
**not** generally a common scalar multiple: the cycle and contact blocks
scale differently. The form quadratic and mobility are unchanged, but
individual dynamical rates and frequencies need not be.

This extends the [prepared-geometry result](RESONANCE_FOUNDATIONS.md#storage-family-pattern-robustness)
to the entire critical set on the present unicyclic support. It explains
why matching equilibrium geometries or their stability types cannot
establish matching accessibility from one fixed preparation.

### The sufficient error-transport method has a separate role

A finite comparison can also write the original-time fields as
`F_eta=F_0+eta D`, where only the form rows of `D` are nonzero and each
has absolute value at most `1/pi`. For the retained block metric,

\[
\|D\|_W\le\frac{\sqrt{\sum_{i,j}|(W_x)_{ij}|}}{\pi_{\rm lower}}\le M.
\]

On a common convex tube with admitted original-time sine logarithmic rate
`gamma`, the paired-state error starting at `e_0` obeys

\[
e(s)\le e^{\gamma s}e_0+|\eta|M\varphi(\gamma,s),\qquad
\varphi(\gamma,s)=\begin{cases}(e^{\gamma s}-1)/\gamma,&\gamma\ne0,\\
s,&\gamma=0.\end{cases}
\]

Initially the paired error is zero because the source is identical.
The retained Picard image `initial_box+[0,h]*direction*F_0(tube)` gives
strict clearance of the reference path inside each tube, also after
reflection and reverse traversal. Keeping every coordinate projection
`sqrt((W^-1)_ii)*max e(s)` below that clearance closes the common-domain
premise by first exit. On target tubes, subtract the sum of both endpoint
phase-error bounds from every strict acute margin. The block symmetry
from Section 51 supplies the needed direction change, and its retained
rates already contain the original clock factor `1/pi`.

Numerical enclosure error and intermediate preparation uncertainty enter
those retained domains and margins; they are not extra constitutive
forcing. A failed sufficient bound would only leave the comparison
unavailable. Here the stronger conserved-budget obstruction already
settles the fixed-scale question, so no 471-step defect-transport campaign,
coefficient reduction or new trajectory is needed. Neither this negative
comparison nor critical-geometry agreement selects a fundamental law or
establishes physical identification.

The detached research reader
[`assess_sine_constitutive_robustness(evidence_directory)`](../../src/tnfr/research/sine_constitutive_robustness.py)
returns `SineConstitutiveRobustness`. It re-admits the frozen source,
preparation, support and required endpoint evidence before rebuilding
the changed storage and sector barrier. It neither installs the changed
law nor executes a new forecast.

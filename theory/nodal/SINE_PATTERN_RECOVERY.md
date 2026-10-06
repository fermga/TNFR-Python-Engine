# Sine relative state, recovery and conservative identity

Exact relative coordinates and bracket, whole-set recovery with a live environment, and the separate zero-loss protection and recurrence result.

Part of [Native pattern reduction and memory](RELATIONAL_PATTERN_MEMORY.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 18. Relative pattern state without discarding internal dynamics

<a id="sine-relative-pattern-state"></a>

Return to the explicitly supplied normalized-sine law. Fix a finite
connected simple unit graph with at least two nodes, its node identities,
held capacities `nu_i>=0`, coefficients `e>=0,w>0,beta>0` and one
clock, with no forcing, clipping or support events. These are model
premises, not an inferred origin of support. The
[comparison-law owner](SINE_CONSTITUTIVE_INFORMATION.md#global-closure-pressure-comparison)
establishes its smooth complete field and storage identity; native
trajectory records remain evidence of the native law.

### An exact quotient by two common origins

Choose any node `r` as reference. Define

\[
u_i=x_i-x_r,\qquad
z_i=\exp\!\bigl(i(\theta_i-\theta_r)\bigr),\qquad
u_r=0,\quad z_r=1.
\]

The remaining coordinates lie in
`R^(n-1) x (S^1)^(n-1)`. This is a global circular quotient, not a
global choice of real angle. It identifies precisely one common form
translation and one common phase rotation. It retains all other nodal
coordinates, including hidden or intermediary nodes, and requires the
same support, capacities, coefficients and clock. No regional averaging,
node deletion, phase-magnitude projection or memory truncation occurs.

Put `k_i=nu_i/d_i`, `a=w/pi`, `b=w/(beta*pi)` and compute from the
relative state

\[
\begin{aligned}
q_i&=\sum_{j\sim i}(u_i-u_j),&
S_i&=\sum_{j\sim i}\operatorname{Im}(z_j\overline z_i),\\
V_i&=k_i(-e q_i+aS_i),&
\Omega_i&=b k_iq_i.
\end{aligned}
\]

These are the original nodal form and phase rates: all consumed
differences are unchanged by the quotient. Differentiation gives the
exact closed induced law

\[
\boxed{\dot u_i=V_i-V_r,\qquad
       \dot z_i=i(\Omega_i-\Omega_r)z_i.}
\]

The phase row preserves `|z_i|=1`. A real relative lift
`rho_i=theta_i-theta_r` instead obeys
`rho'_i=Omega_i-Omega_r`, with `rho_r=0`. Such lifts are legitimate
on a declared chart or along a continuous trajectory, but their
independent full-turn representatives are not extra physical coordinates.

Conversely, a quotient solution and initial common origins reconstruct
the original solution through

\[
\dot x_r=V_r,\qquad \dot\theta_r=\Omega_r,\qquad
x_i=x_r+u_i,\qquad
\exp(i\theta_i)=\exp(i\theta_r)z_i.
\]

Thus holding the reference coordinate at zero inside the original nodal
equations would change the model whenever its rates are nonzero.
Subtracting the reference rows is essential. A zero-capacity node has
`V_i=Omega_i=0` in the original coordinates, but its relative
coordinates can move when the reference moves. If a zero-capacity node
is chosen as reference, both reconstruction rates vanish and the
remaining relative rates equal their original rows. This special case
does not justify freezing a general active reference.

The storage is a function of this complete relative state:

\[
E(u,z)=\frac12\sum_{\{i,j\}}(u_i-u_j)^2+
       \beta\sum_{\{i,j\}}
       \bigl[1-\operatorname{Re}(z_j\overline z_i)\bigr],
\qquad
\dot E=-e\sum_i k_iq_i^2.
\]

Common-reference velocity contributes no work because
`sum_i q_i=sum_i S_i=0`. Removing the two origins therefore preserves
the actual exchange and dissipation, rather than replacing them with a
diagnostic score.

### The same complete law has a dissipative Hamiltonian representation

On the full form/phase coordinates, the preceding storage has gradient
`grad(E)=(q,-beta*S)`. For the held diagonal matrix
`K=diag(nu_i/d_i)` and `b=w/(beta*pi)`, define

\[
J=\begin{pmatrix}0&-bK\\ bK&0\end{pmatrix},\qquad
R_{\!d}=\begin{pmatrix}eK&0\\0&0\end{pmatrix}.
\]

Direct multiplication, using `beta*b=w/pi`, gives exactly

\[
(\dot x,\dot\theta)=(J-R_{\!d})\nabla E.
\]

Here `J` is constant and skew, so its coordinate derivatives vanish
and its bracket satisfies the Jacobi identity. The angular coordinate
fields are well defined on the phase torus; no global real phase lift
is needed for the bracket of smooth periodic functions. Zero capacities
make this Poisson structure degenerate rather than invalid.
`R_d` is positive semidefinite, and
`E'=-grad(E)^T R_d grad(E)=-e*q^T K q` recovers the exact loss.

This is an unforced dissipative Hamiltonian instance of the declared
complete sine model. Network energy storage, interconnection and
dissipation already have an established general framework; see
[van der Schaft and Maschke, *Port-Hamiltonian systems on graphs*](https://arxiv.org/abs/1107.2006).
That reference supplies context, not a proof of this particular TNFR
identity, which follows from the multiplication above. In contrast to
an auxiliary Hamiltonian analogy, the displayed matrices reproduce
both consumed evolution rows. The result remains restricted to held
support and capacities and this pressure/phase law: it neither gives
every TNFR runtime a Poisson structure nor identifies the storage with
physical energy. Such a representation alone establishes no novelty,
physical validation or uniquely selected microscopic law.

For strictly positive capacities this tensor also explains the form
invariant within the selected law. Put `rho=K^-1*1`, `W=sum_i rho_i` and
`Q_x=rho^T*x`. Its Hamiltonian vector field is

\[
J\nabla Q_x=(0,b\mathbf1).
\]

Thus `Q_x/b` generates common phase rotation. Rotation invariance of the
storage gives `{Q_x,E}=0`; the dissipative term also contributes zero,
since `rho^T eKq=e*sum_i q_i=0`. This recovers conservation of `Q_x` from
the same complete field. It does not derive conservation from phase
symmetry alone: choosing the constant reciprocal tensor already fixes an
additional constitutive structure that the native Arg law does not share.

On a continuous real phase lift define `Q_theta=rho^T*theta`. Then

\[
J\nabla Q_\theta=(-b\mathbf1,0),\qquad
\{Q_x,Q_\theta\}=-bW\ne0.
\]

The lifted quantity generates a common form translation with the displayed
sign, but is not a globally single-valued observable on the phase torus.
Neither quantity is a Casimir: their vector fields are nonzero, even though
both are conserved by this particular storage and loss. These identities
are conditional interpretations of the existing bracket, not independent
selection of the bracket, pressure or a physical charge. The
[static matrix controls](../../tests/physics/test_relational_pressure_composition.py)
check the generators, their nonzero mutual bracket and both consumed rows.

### Reconstruction from conserved means when every capacity is positive

For `nu_i>0`, set `omega_i=d_i/nu_i` and `W=sum_i omega_i`.
Reciprocity gives

\[
\sum_i\omega_iV_i=-e\sum_iq_i+a\sum_iS_i=0,\qquad
\sum_i\omega_i\Omega_i=b\sum_iq_i=0.
\]

Hence the weighted form mean `M_x` and the weighted mean `M_theta`
of a chosen continuous phase lift are constant. Reconstruction can then
be written algebraically:

\[
x_r(t)=M_x-\frac1W\sum_i\omega_i u_i(t),\qquad
\theta_r(t)=M_\theta-\frac1W\sum_i\omega_i\rho_i(t).
\]

The lifted mean is not a single-valued scalar on the phase torus.
Changing initial phase representatives changes its representation
consistently, without changing the reconstructed circular state.
These formulas are unavailable at zero capacity; the previous direct
reconstruction remains valid.

There are no nonzero uniformly drifting relative equilibria when every
capacity is positive. A stationary quotient would require
`V_i=v` and `Omega_i=omega` for every node. The two weighted identities
force `v=omega=0`. The phase row then gives `q=0` and the form row
`S=0`. Thus the stationary relative patterns are exactly uniform form
with a critical phase geometry. A common pulse or uniform rotation is
not generated by merely changing the reference.

### What observation uncertainty cancels, and what remains

At one common actual time, suppose reported form coordinates are
`y_i=x_i+C_x+epsilon_i` and consistently lifted phases are
`psi_i=theta_i+C_theta+eta_i`. The shared offsets can be unknown and
can vary between synchronous captures. Taking relative coordinates
first cancels them exactly:

\[
y_i-y_r=u_i+\epsilon_i-\epsilon_r,\qquad
\psi_i-\psi_r=\rho_i+\eta_i-\eta_r.
\]

If only separate bounds `|epsilon_i|<=delta_i` are supplied, the
sharp independent-error radius for an anchored form difference is
`delta_i+delta_r`; the phase statement is identical in its admitted
lift. There is no basis for subtracting this uncertainty merely because
the common origin was removed.

The shared reference error also creates dependence between relative
coordinates. For an edge `i--j`, subtract before interval evaluation:
its error is `epsilon_i-epsilon_j`, with sharp radius
`delta_i+delta_j`. Subtracting two separately expanded anchor boxes
would produce the valid but unnecessarily broad radius
`delta_i+delta_j+2*delta_r`. An explicit common-error model permits
algebraic cancellation; matching marginal error sizes alone do not
prove such dependence. Circular uncertainty must retain its phase-set
or justified lift interpretation, including any branch ambiguity.

Synchronous means the same actual time, not merely matching nominal
timestamps. With different acquisition jitter, both the nodal state
and a moving sensor reference may be evaluated at different times.
Then `C(t_i)-C(t_r)` need not vanish. Node-speed bounds control the
former error; an independently justified reference-drift bound is needed
for the latter. The prior sample budget bounds the declared coordinate
and clock, not an undeclared moving instrument reference.

Likewise, the existing hidden-state inverse consumes original nodal
rates, not rates relative to a moving node. Supplying
`V_i-V_r,Omega_i-Omega_r` unchanged to that inverse can produce a
consistent but wrong reconstruction. Consider a three-node star with
ports `r,p` of unit capacity and one hidden node `h`:

\[
(x_r,x_p,x_h)=(0,1,2),\qquad
(\theta_r,\theta_p,\theta_h)=(0,\delta,0),\qquad
\sin\delta\ne0.
\]

Its original port rows are

\[
(V_r,\Omega_r)=(2e,-2b),\qquad
(V_p,\Omega_p)=(e-a\sin\delta,-b).
\]

The anchored rows are therefore `(0,0)` at `r` and
`(-e-a*sin(delta),b)` at `p`. Those are exactly the **absolute**
port rows of a different hidden state `x_h=0,theta_h=0` with the
same visible coordinates. The observation geometry has rank two, so
the original inverse can identify this phantom form instead of the
actual value `2`. The issue is the missing reference velocity, not
loss of dynamics in the complete quotient. Restore the reference rows
with their evidence or derive a separately admitted relative inverse;
a relabeled rate dictionary does neither.

Only one global origin of each kind is redundant on this connected
model. Removing independent origins from several coupled regions
would discard their relative offsets, which drive interaction across
the connecting edges. Likewise, omitting a hidden state, changing
capacity or losing a supplied boundary input is not this symmetry.
The previous causal-memory and hidden-state requirements remain in
force under a change of reference.

### Winding is retained information, not an unconditional invariant

The quotient determines every edge phase ratio
`z_j*conjugate(z_i)`. Away from antipodal edges, it therefore determines
the principal phase increment on each oriented edge and the winding of
each supplied oriented cycle. Common rotation changes none of them.
At an antipodal ratio `-1` the two limiting increments `+pi` and
`-pi` meet: a principal-increment winding cannot be continuously
assigned through that boundary.

The sine law itself remains smooth there. Consequently an antipodal
crossing can change winding without a singular nodal state, deleted
edge or new event. A snapshot winding and positive initial margin
alone do not establish its indefinite preservation. Uncertain edge
sets touching that boundary require an unresolved winding or separate
branch evidence. Temporal preservation needs an invariant region,
such as the sufficient local recovery domain below.

### A whole-set sufficient recovery criterion under the sine law

One concrete maintained-pattern identity is the common-origin orbit
of an exact critical phase geometry `theta_*` with uniform form.
Assume now **`e,w,beta>0` and every held `nu_i>0`**. Require
`S(theta_*)=0` exactly and all reference principal edge increments
`delta_*,ij` strictly acute. A small computed residual alone does not
establish this premise.

Let `L` be the unit graph Laplacian, `lambda_2(L)>0` its spectral gap,
and `P=I-11^T/n`. In a declared continuous deviation lift from
`theta_*`, put

\[
Z^2=\|Pu\|^2+\|P(\rho-\rho_*)\|^2,\qquad
\mathcal E=E(u,z)-\beta V_\phi(\theta_*).
\]

This norm uses the declared model coordinates; it is not a universal
physical metric or a diagnostic definition of identity. With
`m=min_edges(pi/2-|delta_*,ij|)`, choose

\[
0<r<\frac m{\sqrt2},\qquad
c_r=\min_{\{i,j\}}\cos(|\delta_{*,ij}|+\sqrt2r)>0,\qquad
\kappa_r=\frac{\lambda_2(L)}2\min(1,\beta c_r).
\]

Let `U` be an admitted state uncertainty set in this relative chart,
with all its internal coordinates retained and the declared fixed
support, coefficients and capacities unchanged. The
sufficient conditions are

\[
\boxed{\sup_{U}Z<r,\qquad
       \sup_{U}\mathcal E<\kappa_r r^2.}
\]

They imply that **every actual state in `U`** stays in this phase
chart and converges to the reference geometry modulo common origins.
The set need not be a singleton; correlated uncertainty may be
conservatively enclosed by a larger set if that entire enclosure
passes the inequalities.

The geometric barrier `kappa_r` does not depend on the positive
capacity values. With support and coefficients still fixed, the same
criterion consequently holds for each held capacity vector in an
independently admitted strictly positive family. This covers capacity
uncertainty without choosing its midpoint. It does not supply a uniform
decay rate as capacities approach zero, and it does not admit a zero
capacity by continuity of a positive-capacity theorem.

**Proof of trapping and convergence.** Inside `Z<=r`, each edge's
phase-deviation difference is bounded by `sqrt(2)*r`. The line
segment from the reference remains acute, and its cosine Hessian
is bounded below by `c_r L` on the common-phase quotient.
Criticality removes the linear phase-storage term. Taylor's formula
and the graph spectral gap give

\[
\mathcal E\ge
\frac{\lambda_2(L)}2\|Pu\|^2+
\frac{\beta c_r\lambda_2(L)}2
 \|P(\rho-\rho_*)\|^2
\ge\kappa_r Z^2.
\]

The exact sine loss `mathcal E'=-e*q^T K q<=0`, with
`K=diag(nu_i/d_i)>0`, prevents a first exit through `Z=r`.
The trajectory remains in a compact interior quotient sublevel.
On its largest invariant zero-loss subset, `q=0`, hence
`Omega=0`. Preserving `q=0` further requires `LKS=0`, so
`KS=c*1`. But `sum_i S_i=0` and `K>0` give
`c*sum_i(1/k_i)=0`; therefore `c=0` and `S=0`.
The positive phase Hessian on this convex local chart makes
`theta_*` its unique critical phase geometry modulo rotation.
LaSalle's principle thus gives convergence to that orbit.

Local exponential recovery follows from the same sine rows, rather
than from a transferred native trajectory. Choose orthonormal columns
`R` spanning `1^perp` and define
`B=R^T L R>0`, `A=R^T K R>0` and
`C=R^T Hess(V_phi)(theta_*) R>0`. The quotient Jacobian is

\[
J=\begin{pmatrix}-eAB&-aAC\\ bAB&0\end{pmatrix}.
\]

In storage coordinates it is similar to
`[[-D,-H],[H^T,0]]` with
`D=e*B^(1/2)*A*B^(1/2)>0` and
`H=(a/sqrt(beta))*B^(1/2)*A*C^(1/2)` invertible.
For an eigenvector `(v,w)`,
`Re(lambda)*(|v|^2+|w|^2)=-v^*D*v`. A zero real part would force
`v=0` and then `H*w=0`, contradicting a nonzero eigenvector.
All quotient eigenvalues have negative real part; smoothness gives
local exponential recovery. The weighted-mean reconstruction then
also determines the finite limiting common origins.

The positive-dissipation and positive-capacity premises matter.
At `e=0` a nonzero local excess storage is conserved and cannot
converge to its zero value at the target. Two inactive nodes with
distinct frozen forms can preclude uniform-form recovery however
small their difference. Positive stiffness or the above acute
criterion is sufficient, not a classification of every stable sine
geometry or its full basin.

This theorem closes a bounded mechanism question: irrelevant common
origins can be removed exactly, while all sufficiently small admitted
**relative** perturbations in the stated domain recover under the
supplied nodal law. It does not prove spontaneous entry into that
domain, select the law, create support or identify a pattern with
matter. The inequalities are a mathematical whole-set criterion;
their application to a concrete uncertain pattern requires outward
admission of every premise, as in the cycle family below.

<a id="sine-cycle-recovery"></a>
### Computable whole-set recovery for an exact cycle twist

Supply an ordered cycle `(v_0,...,v_(n-1))` containing every node
exactly once, `n>=3`, whose consecutive and closing edges equal the
complete support. Extra edges, a selected cycle inside a larger network
and an unspecified orientation do not meet this contract. Let the
target winding `ell` be an integer with `4*|ell|<n` and define

\[
\alpha=\frac{2\pi\ell}{n},\qquad
\theta^*_{v_j}=j\alpha\pmod{2\pi}.
\]

This is an exact symbolic target, including consensus `ell=0`.
Each node has neighboring increments `+alpha,-alpha`, so
`S_i(theta*)=sin(alpha)+sin(-alpha)=0` identically. Its Hessian is
`cos(alpha)*L` and `cos(alpha)>0`. No rounded phase vector or small
numerical residual is used to recognize an equilibrium. The cyclic
Fourier modes of the unit Laplacian have eigenvalues
`2-2*cos(2*pi*j/n)`, giving the exact nonzero gap

\[
\lambda_2(L)=2-2\cos(2\pi/n).
\]

#### Chart choice and cancellation before interval evaluation

The caller supplies one integer turn `m_j` per observed phase lift;
zero turns are an explicit permissible choice. Define the deviation
lift `h_j=theta_(v_j)+2*pi*m_j-j*alpha`. Turns choose a chart for
the same circular data; they are not automatically fitted to force
admission and do not reduce an observation error.

For original synchronous observations with independent residual bounds
`epsilon_i` around their nominal coordinates, every pair satisfies

\[
\begin{aligned}
X_{ij}&=
 (x^{\rm nom}_{v_j}-x^{\rm nom}_{v_i})
+[-\delta_i-\delta_j,\delta_i+\delta_j],\\
H_{ij}&=
 (\theta^{\rm nom}_{v_j}-\theta^{\rm nom}_{v_i})+
 \pi\left[2(m_j-m_i)-\frac{2\ell(j-i)}n\right]
+[-\epsilon_i-\epsilon_j,\epsilon_i+\epsilon_j].
\end{aligned}
\]

Here `delta_i` denotes the form residual radius; `epsilon_i` denotes
the phase residual radius. Common form and phase origins cancel
exactly. The entire rational coefficient of mathematical pi is formed
before outward interval evaluation; separately materializing target
angles would introduce avoidable cancellation error.

The exact identity
`||Pf||^2=(1/n)*sum_(i<j)(f_j-f_i)^2` gives the sufficient squared
norm bound

\[
Z^2\le Z_+^2:=
\frac1n\sum_{i<j}
\left(\sup X_{ij}^2+\sup H_{ij}^2\right).
\]

Pairwise interval bounds can be conservative because the same residual
appears in several pairs, but they never add an artificial reference-node
error. They retain every node and certify the whole declared observation
class rather than its midpoint.

#### Quadratic excess storage without subtracting twist energy

Orient every cycle edge from `v_j` to `v_(j+1 mod n)` and let
`d_j=h_(j+1 mod n)-h_j`. The closing edge has the same circular
target increment `alpha`, even though its target real-lift difference
is `-(n-1)*alpha`. Its full-turn difference must be retained when
computing `d_(n-1)`. The deviations telescope exactly:
`sum_j d_j=0`.

Since every target edge has the same sine, the entire linear term in
phase excess storage cancels:

\[
\begin{aligned}
V_\phi(\theta)-V_\phi(\theta_*)
 &=\sum_j[\cos\alpha-\cos(\alpha+d_j)-\sin\alpha\,d_j],\\
\cos\alpha-\cos(\alpha+d)-\sin\alpha\,d
 &=d^2\int_0^1(1-s)\cos(\alpha+sd)\,ds.
\end{aligned}
\]

This is an exact integral remainder for either sign of `d`. It avoids
subtracting two nonzero twist energies and has quadratic rather than
linear sensitivity to small interval widths.

Let `H_j` enclose the edge deviation `d_j` and set

\[
c_j^+=
\sup\cos\!\left(\alpha+\operatorname{hull}(0,H_j)\right).
\]

In computation, the cosine interval's upper endpoint is an outward
upper bound for this supremum. The segment contains `alpha`, so
`c_j^+>=cos(alpha)>0`, and `c_j^+<=1`. Therefore

\[
\mathcal E\le Q_+:=
\frac12\sum_j\sup X_{j,j+1}^2+
\frac\beta2\sum_j c_j^+\sup H_j^2.
\]

Using `c_j^+=1` remains safe but less informative. No individual
Taylor term is presumed independent of another; their upper bounds
enclose the complete sum. This upper-bound inequality is valid before
basin admission. Nonnegative excess storage follows only after the
appropriate acute neighborhood is established, not from clipping a
possibly negative general excess to zero.

For supplied `r>0`, the cycle constants are

\[
m=\frac\pi2-|\alpha|,\qquad
c_r=\cos(|\alpha|+\sqrt2r),\qquad
\kappa_r=\frac{\lambda_2(L)}2\min(1,\beta c_r).
\]

Outward pi, square-root and cosine bounds provide certified lower
endpoints. Admission requires `sqrt(2)*r<m`, a strictly positive
resolved lower bound `kappa_-<=kappa_r`, and

\[
\boxed{Z_+^2<r^2,\qquad Q_+<\kappa_-r^2.}
\]

Positive dissipation and every strictly positive held capacity remain
separate hypotheses. Failure to resolve these sufficient inequalities
is unavailability of this certificate, not proof that the state cannot
recover or that every point in its uncertainty set leaves the basin.

#### Original observation sets and propagated boxes are different inputs

The preceding pair formulas use the original per-node residuals once.
Their conclusion covers that synchronous observation set. A Cartesian
outer box from full-state propagation generally contains additional
states because residual correlations have been lost. It must be
checked in full, even if the tighter original set passed.

For such a propagated box, form each `X_ij` by subtracting the two
full endpoint form intervals, and each `H_ij` by subtracting the two
full phase intervals followed by the same exact target/turn
coefficient. Apply the identical norm and remainder proof to these
wider pair intervals. Use the original full endpoint boxes, before
another subtraction of the moving reference; do not reuse the source's
smaller edge bounds or its earlier storage bound.

The report's actual validated endpoint time identifies the state being
certified. A partial forecast can therefore admit recovery starting at
that earlier validated time, but cannot silently claim certification
at an unreached requested time. Original observations provide exact
held capacities. A full solver box may instead carry its final node's
held capacity as an augmented interval: its whole interval must be
strictly positive, while the other held capacities satisfy the same
condition. The positive-family extension above then applies without
selecting a capacity midpoint. A box including zero is outside this
recovery theorem.

#### An explicitly deformed C5 observation class that passes

Choose the positively oriented unit C5, target `ell=1`,
`e=w=1/2,beta=1` and unit capacities. Supply no extra phase turns,
`r=1/16` and dyadic nominal observations

\[
x^{\rm nom}=(1/1024,-1/1024,0,0,0),\qquad
\theta^{\rm nom}_{v_j}=5j/4.
\]

Every node has form and phase residual radius `1/4096`; both common
origins remain arbitrary. These observations are not the exact target:
the target increment remains `alpha=2*pi/5`, not `5/4`.
No trajectory or generated response is needed to admit this class.

The elementary bounds `25/8<pi<22/7` give
`0<alpha-5/4<1/140`. Put

\[
A=\frac1{140}+\frac1{2048},\qquad
B=\frac1{35}+\frac1{2048},\qquad
F=\frac5{2048}.
\]

All form-pair magnitudes are at most `F` and all phase-deviation
pair magnitudes at most `B`. Hence
`Z_+^2<=2*(B^2+F^2)=2184757/1284505600<1/256=r^2`.
The four nonclosing phase-edge deviations have magnitude at most `A`;
the closing one has magnitude at most `B`.

Every phase interpolation segment lies between `31/25` and `pi/2`.
The cosine upper Taylor bound at `31/25` gives
`c_j^+<=1-(31/25)^2/2+(31/25)^4/24<1/3` as an analytic upper
bound. Consequently

\[
Q_+\le\frac52F^2+\frac{4A^2+B^2}{6}
 =\frac{59951}{308281344}<\frac1{2560}.
\]

Furthermore `lambda_2(L)>1`, and
`alpha+sqrt(2)*r<377/280<27/20`. The alternating lower bound
`cos(27/20)>=1-(27/20)^2/2+(27/20)^4/24-(27/20)^6/720>1/5`
gives `c_r>1/5` and `kappa_r*r^2>1/2560`.
All strict recovery conditions therefore hold for this complete
uncertain class. An implementation still records its own outward
computed margins; the analytic inequalities are an independent
admission proof, not substituted fixture values.

This result proves conditional recovery of an already supplied
geometric identity despite nonzero relative observation uncertainty.
It does not create the target winding, establish spontaneous entry
into this neighborhood, select the microscopic law, add support or
identify the cycle with a physical constituent.

## 19. Recovering patterns with a live retained intermediary

<a id="sine-interacting-recovery"></a>

Two locally recoverable regions can belong to a single recoverable
**interacting** geometry under the supplied sine law. The statement here
retains the complete fine support and the intermediary's form, phase and
capacity. It neither replaces the intermediary by a direct edge nor
derives when a new edge appears.

### An exact critical target on the full eleven-node support

Take cycles `(0,1,2,3,4)` and `(5,6,7,8,9)` and the two additional
edges `0--10` and `5--10`. Node `h=10` is the intermediary, and
nodes `0,5` are the ports. There are eleven nodes and twelve unit
edges; each port has degree three and the intermediary degree two.
For `alpha=2*pi/5`, specify

\[
\theta^*_{j}=\theta^*_{5+j}=j\alpha\pmod{2\pi}
\quad(0\le j<5),\qquad \theta^*_{10}=0.
\]

At an ordinary ring node the incoming sine terms are
`sin(alpha)+sin(-alpha)=0`. The same cancellation holds at a port,
and its additional intermediary edge contributes `sin(0)=0`.
Both intermediary edges have zero phase difference. Therefore
`S(theta*)=0` exactly on the entire graph, not just in either
isolated cycle. Uniform form completes a stationary state.

The phase Hessian assigns ring edges weight `cos(alpha)>0` and
the intermediary edges weight one. It is positive on the full
common-phase quotient, including perturbations that move the two
rings against each other or change the intermediary. Independent
regional stability would not by itself prove this full-network result.
For strictly positive held capacities and `e,w,beta>0`, Section 18
therefore supplies local exponential recovery of this complete geometry.

More generally, the shared exact phase-cycle reconstruction can admit a
supplied rational-turn target on the full graph. It checks integral
cycle periods, strict acute edges and symbolic cancellation of opposite
sine terms. Only these algebraic facts are reused here. An unproved sine
residual, an approximate equilibrium or that owner's separate dynamical
interpretation cannot substitute for the full sine-law hypotheses.

For an exactly critical target with possibly different edge angles
`alpha_e`, the excess-storage cancellation extends directly:
`sum_e sin(alpha_e)*(h_j-h_i)=grad(V_phi)(theta*)^T h=0`.
The quadratic remainder bound in Section 18 consequently applies
edge by edge with its own target angle. No uniform twist or
equal-capacity premise is needed for that cancellation.

### An independent full-graph spectral-gap bound

For every unordered node pair choose a path `P_ij` with length
`l_ij`. Cauchy--Schwarz and the pairwise centering identity give

\[
\begin{aligned}
\|Pv\|^2
 &=\frac1n\sum_{i<j}(v_j-v_i)^2\\
 &\le\frac1n\sum_e C_e(v_{e^+}-v_{e^-})^2,\qquad
C_e=\sum_{\{i,j\}:e\in P_{ij}}l_{ij}.
\end{aligned}
\]

Thus `lambda_2(L)>=n/max_e C_e`. This is a finite path-counting
proof, not a numerical eigensolver assumption. For the stated support,
the shortest paths give:

| Edges, with the same pattern on the other ring | Path load `C_e` |
| --- | --- |
| `0--10` and `5--10` | `121` |
| `0--1` and `0--4` | `57` |
| `1--2` and `3--4` | `34` |
| `2--3` | `5` |

For example, distances from a ring's nodes to its port are
`(0,1,2,2,1)`. The bridge's paths from that ring to the
intermediary contribute `6+5=11`; paths between the rings contribute
`5*6+25*2+5*6=110`, giving `121`. Hence

\[
\lambda_2(L)\ge\frac{11}{121}=\frac1{11}.
\]

With `beta=1` and `r=1/16`, the earlier elementary cosine bound
still gives `c_r>1/5`, because the largest target edge angle is
`2*pi/5`. The whole-network trapping barrier therefore satisfies
`kappa_r*r^2>1/28160`. This analytic lower bound is independent
of the engine's general rational quotient-gap certificate; a more
conservative computed gap remains valid if its own margins pass.

### A finite uncertain class within the interacting basin

Use default `e=w=1/2,beta=1` and unit capacities. For both rings
supply the dyadic nominal phases
`theta_j^nom=theta_(5+j)^nom=1287*j/1024`; supply intermediary
phase zero. The nominal form is `x_0=1/4096` and zero elsewhere.
Every node has form and phase residual radius `1/65536`, and the
two common origins may be arbitrary. Target turns remain exactly
`j/5` on the rings and zero at the intermediary. No phase turns
are added to the nominal observations.

The elementary bounds `157/50<pi<22/7` imply
`|1287/1024-2*pi/5|<1/1024`. Define

\[
R=\frac1{32768},\qquad A=\frac{33}{32768},\qquad
B=\frac{129}{32768},\qquad F=\frac9{32768}.
\]

Here `R` bounds a difference of two residual errors, `A` bounds
each nonclosing ring-edge phase deviation, `B` bounds each closing
one and every pair's phase deviation, and `F` bounds every form
difference. The two port/intermediary phase deviations are at most
`R`. The pairwise identity gives

\[
Z_+^2\le5(B^2+F^2)
 =\frac{41805}{536870912}<\frac1{256}=r^2.
\]

All ring-edge interpolation segments lie between `31/25` and
`pi/2`, so the independent Taylor bound used above supplies a
cosine upper bound `1/3`. Bridge segments can use the bound one.
Three form edges touch the perturbed donor port; the other nine
have only residual error. Therefore

\[
\begin{aligned}
\mathcal E
&\le\frac{3F^2+9R^2}{2}
    +\frac{4A^2+B^2}{3}+R^2\\
&=\frac{3563}{536870912}
 <\frac1{28160}<\kappa_r r^2.
\end{aligned}
\]

The entire synchronous uncertainty class is consequently within the
full interacting recovery basin. The same argument covers any
strictly positive held capacity vector; it gives no uniform recovery
time as a capacity approaches zero. A computed certificate must retain
its own full-graph gap and error bounds. In particular, the general
quotient-gap owner is not claimed to return the stronger path bound
`1/11`; its separately computed uncertainty and barrier margins
can still establish admission.

<a id="exact-nonlinear-donor--intermediary--receiver-onset"></a>
### Exact nonlinear donor--intermediary--receiver onset

A separate, exactly specified preparation makes causality transparent.
Use the exact target phases, nominal forms
`x_0=epsilon>0` and `x_i=0` otherwise, ring capacities one and
intermediary capacity `mu>=0`. This is not the preceding dyadic
phase-observation class. All statements in this paragraph concern the
actual nonlinear rows at this finite preparation, not a fitted
small-amplitude derivative.

Write `a=w/pi` and `b=w/(beta*pi)`. At time zero,
`q_h=-epsilon` and `S_h=0`, giving

\[
\dot x_h=\frac{\mu e\epsilon}{2},\qquad
\dot\theta_h=-\frac{\mu b\epsilon}{2}.
\]

Every receiver-ring node initially has zero form and phase rate.
At its port `5`, differentiating the three incident contributions
gives

\[
\dot q_5=-\dot x_h=-\frac{\mu e\epsilon}{2},\qquad
\dot S_5=\dot\theta_h=-\frac{\mu b\epsilon}{2}.
\]

The latter uses the intermediary edge cosine one and zero initial
phase rates at both receiver neighbors. Retaining a general positive
receiver-port capacity `nu_5` for this identity yields

\[
\boxed{
\ddot x_5=\frac{\nu_5\mu\epsilon}{6}(e^2-ab),\qquad
\ddot\theta_5=-\frac{\nu_5\mu e b\epsilon}{6}.}
\]

The donor capacity does not enter this leading two-edge response:
the intermediary already consumes the donor's initial form contrast.
For `mu>0,e>0` the receiver phase thus obeys

\[
\theta_5(t)-\theta_5(0)=
-\frac{\nu_5\mu e b\epsilon}{12}t^2+O(t^3).
\]

It is strictly negative for all sufficiently small positive times.
The exact smooth law supplies this local onset; no numerical horizon,
time-series evaluation or fitted remainder is asserted. At the default
coefficients `e^2-ab=(1-1/pi^2)/4>0`, the initial receiver form
curvature is positive as well.

If `mu=0`, the intermediary instead remains exactly at its initial
form and phase. The unperturbed receiver twist with that fixed
boundary is an exact stationary solution, so uniqueness makes the
receiver stay unchanged for all time. This is a declared capacity
intervention on the same existing support. It separates a causally
active intermediary from a frozen one, not one unknown physical law
from another.

### Influence followed by recovery and a retained common-form record

For the exact-phase preparation, the full quotient norm and initial
excess storage are
`Z^2=10*epsilon^2/11` and `mathcal E=3*epsilon^2/2`.
Taking `epsilon=1/4096` places them strictly below the same
`r=1/16` trapping conditions. Thus for every fixed `mu>0` this
causal response coexists with full-network recovery: both ring
geometries and their intermediary recover to the supplied target.
The perturbation is not required to destroy the pattern to transmit
an effect.

The conserved weighted form mean gives an additional exact endpoint
statement. The two rings have total degree `22` and unit capacity;
the intermediary contributes weight `2/mu`. Only the degree-three
donor port initially carries form `epsilon`. Hence the limiting
uniform form is

\[
x_i(\infty)=\frac{3\epsilon}{22+2/\mu}
\quad\hbox{for every node and each fixed }\mu>0.
\]

For `mu=1` this is `epsilon/8`. The initial exact target phase
has its target weighted lift mean, so the limiting common phase
shift is zero in this local lift. The frozen-intermediary receiver
instead keeps form zero forever. The positive-capacity comparison
therefore supplies both a transient causal signal and a lasting
common-form offset under the permanently retained support.

This offset is a conserved coordinate record; it is not a new
winding, a new constituent or an independently identified physical
memory. It lies in the common-origin coordinate removed by the relative
quotient: distinguishing it requires the retained preparation frame or
another admitted reference, not the final internal pair differences alone.
It tends to zero as `mu` tends to zero. Neither the
recovery theorem nor this limit supplies a uniform finite response
time near that boundary.

### A symmetry obstruction to forming the receiver twist

Recovery of the supplied receiver identity does not establish formation
from a flat receiver. Reuse the exact automorphism mechanism from
the [orientation audit](RELATIONAL_MEDIATOR_DYNAMICS.md#mediator-orientation-scope), now acting only
on the receiver:
`R=(6 9)(7 8)` fixes its port `5`, the intermediary and all donor
nodes. If paired receiver capacities are equal, this permutation
preserves the full sine law.

A flat receiver phase and uniform receiver form are fixed by `R`,
regardless of the donor or intermediary state. Equivariance and
uniqueness keep that receiver reflection exact under the endogenous
single-port drive. A winding-one or winding-minus-one receiver twist
is not reflection-fixed; a common rotation cannot repair this because
the port is fixed. Such an initial state therefore cannot converge
to either of those receiver identities. Away from antipodal edges,
its principal-increment receiver winding remains zero by paired
edge cancellation.

This is a consequence of an already justified symmetry, not a new
simulation campaign or a claim that asymmetric formation is impossible.
A meaningful formation test must declare information that breaks
this symmetry and satisfy the relevant storage/transition budget
before evaluating a trajectory. Changing an initial state or held
capacity supplies that information; it does not derive its origin.
No new pressure term or operator selector is installed by the present
recovery and causality results.

<a id="sine-conservative-identity"></a>

## 23. Conservative phase identity with nonlinear recurrence

### Retain the law and distinguish trapping from recovery

Take the complete isolated unit cycle `C_n`, `n>=3`, with
declared orientation and **strictly positive held capacities**.
Use the complete normalized-sine law with `e=0,w,beta>0`
and no input, event, clipping or support change:

\[
\dot x=aKS(\theta),\qquad \dot\theta=bKLx,\qquad
K=\operatorname{diag}(\nu_i/2),\quad
a=w/\pi,\quad b=w/(\beta\pi).
\]

The total storage `E=x^TLx/2+beta*V(theta)` is conserved.
The [existing acute recovery proof](#sine-cycle-recovery) separates
a geometric first-exit barrier from an attracting dynamics argument.
Only the barrier is reused here. The latter requires `e>0`;
it is not a recovery theorem for the present conservative law.

Choose an integer `k` with `4*abs(k)<n` and the exact
critical target

\[
\theta_{*,j}=\frac{2\pi k j}{n},\qquad
\alpha=\frac{2\pi k}{n},\qquad
E_*=\beta n(1-\cos\alpha).
\]

Opposite sine currents cancel at each node, and all target edge
increments are strictly acute. The nontrivial case `n=5,k=1`
is the winding-one identity under study; the same derivation also
covers the other declared acute cycle twists.

### A genuine phase chart with common origins retained

Let `P=I-11^T/n` and write the centered form as `u=Px`.
Represent nearby circular phases by

\[
\theta=\theta_*+c\mathbf1+v\pmod{2\pi},\qquad
v\in\mathbf1^\perp,\quad c\in\mathbb R/2\pi\mathbb Z.
\]

Here `c` is one common circular phase origin, not a separately
adjustable origin at each node. Choose `r>0` such that

\[
|\alpha|+\sqrt2\,r<\frac\pi2.
\]

The representation is injective for `||v||<r`. To see this,
suppose two centered deviations and common origins give the same
phase state. Subtracting any two node equations gives

\[
(v_i-v_j)-(v'_i-v'_j)=2\pi(m_i-m_j)
\]

for integers `m_i`. Its left-hand side has magnitude less than
`2*sqrt(2)*r<2*pi`, so all the integers agree. Centering then
forces `v=v'` and equality of the common origins on the circle.
The linear differential of the chart is invertible, so its image
is open in the phase torus. The strict radius bound also supplies
an injective chart on a slightly larger neighborhood of its closure.
There is no phase-seam exit hidden inside the admitted ball.

Define the full relative norm in the same model coordinates as the
recovery owner,

\[
Z^2=\|u\|^2+\|v\|^2.
\]

For `Z<=r` the oriented wrapped edge increments are
`alpha+v_(j+1)-v_j`, including the closing edge. They remain
strictly acute, and their sum is `2*pi*k`. Every state in this
chart therefore has the same declared oriented winding `k`.

### The coercive barrier is independent of dissipation

Let

\[
\lambda_2=2-2\cos(2\pi/n),\qquad
c_r=\cos(|\alpha|+\sqrt2r)>0,\qquad
\kappa_r=\frac{\lambda_2}{2}\min(1,\beta c_r)>0.
\]

The phase segment from `theta_*` to `theta_*+v` remains
acute throughout the closed radius ball. Criticality cancels its
linear storage term, and its Hessian is bounded below by `c_r L`.
The graph spectral gap thus gives

\[
\mathcal E:=E-E_*
\ge\frac{\lambda_2}{2}\|u\|^2
 +\frac{\beta c_r\lambda_2}{2}\|v\|^2
\ge\kappa_r Z^2
\qquad(Z\le r).
\]

Choose an independently declared excess ceiling and weighted-mean
interval,

\[
0<h<\kappa_r r^2,\qquad m_{\rm lo}<m_{\rm hi},
\]

where

\[
m_\rho(x)=\frac{\sum_i\rho_i x_i}{\sum_i\rho_i},\qquad
\rho_i=\frac{2}{\nu_i}>0.
\]

The proposed full-state family is

\[
\boxed{
\mathcal U=\left\{
Z^2<r^2,\quad \mathcal E<h,\quad
m_{\rm lo}<m_\rho(x)<m_{\rm hi}
\right\},}
\]

with the common phase origin `c` free on its circle.
The initial radius condition cannot be omitted: a small energy
relative to `E_*` outside this chart does not identify this
phase pattern.

For every member of `U`, conservation of `mathcal E`
prevents a first radius exit. At such an exit the coercive bound
would require `mathcal E>=kappa_r*r^2>h`. The argument
works in both time directions because energy and the weighted form
mean are exactly conserved. The chart is retained, and within it

\[
Z(t)^2\le\frac{\mathcal E(0)}{\kappa_r}
 <\frac h{\kappa_r}<r^2
\qquad\text{for all real }t.
\]

Every state in the family therefore retains acute edges and winding
`k` indefinitely under the continuous declared law. This is
**trapping**, not convergence to `theta_*`. For a nonzero
excess energy, convergence to that zero-excess target is impossible
under exact conservation.

### Open positive-volume family and almost-everywhere motion

The family is open in the full `2n`-dimensional state manifold.
Indeed the phase chart is locally invertible, and form can be
parameterized by `(m_rho,u)` with `u perpendicular 1`:

\[
x=\left(m_\rho-\frac{\rho^\mathsf Tu}
                         {\rho^\mathsf T\mathbf1}\right)\mathbf1+u.
\]

This is an invertible linear coordinate change. Uniform form with
mean strictly inside the interval and the exact target phase lies
inside every displayed strict inequality; a full open neighborhood
does too. Thus the family has positive ambient product volume,
not merely volume within a synchronized or fixed-energy surface.

Its volume is finite. Positive mean weights and the radius give

\[
|x_i-m_\rho(x)|
 \le\max_j|u_i-u_j|
 \le\sqrt2\,\|u\|<\sqrt2r.
\]

All forms therefore lie in a bounded box, and all phases lie on
their compact torus. The **open** family need not itself be compact;
its closure in the retained chart and form box is compact. The
strict energy barrier keeps its trajectories away from the chart's
radius boundary, while the conserved mean keeps each trajectory at
its own interior mean value. The smooth sine flow is complete and
maps `U` onto itself in both time directions.

The [full nonlinear recurrence theorem](RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
now applies to this invariant finite-volume family. Its divergence
is zero, so almost every state of `U`, in ambient Lebesgue
form times circular Haar phase measure, returns arbitrarily close
to itself along arbitrarily late times. The equilibrium subset has
measure zero, since positive capacity forces equilibrium form to
be uniform. Consequently almost every member has **nonstationary
recurrent motion while retaining this phase identity**.

The quantifiers differ: every admitted member is trapped with fixed
winding; only almost every member is certified recurrent by this
measure theorem. Neither result selects a common period, return
time, oscillation amplitude, physical frequency or attracting orbit.
The prescribed support and target identity are not formed by this
admission test.

### Relative observations and the absolute mean slab are different evidence

An uncertain source can certify the shape conditions if its entire
declared relative-state set satisfies

\[
\sup Z^2<r^2,\qquad \sup\mathcal E<h.
\]

Reuse the existing owner for centered pairwise norms, exact target
turns and the phase-energy Taylor remainder. Original observations
retain their correlated common-origin-plus-residual set; a forecast
uses its full validated endpoint box at the actual validated time.
A requested future time is not certified merely by a partial forecast.
No midpoint, redefined phase origin or changed law supplies a missing
strict bound.

These shape inequalities prove trapping for every compatible state
even if the common form origin is unspecified. However,
`SineRelativePattern` permits an arbitrary common form shift.
Its entire observation set therefore cannot be declared inside a
finite absolute `m_rho` interval by substituting the nominal mean.
The finite-volume family and relative shape membership must be
reported separately. Absolute-family membership requires independent
mean information, or an explicitly conditional intersection of the
observation set with the declared mean slab.

The same separation applies to recurrence: neither relative trapping,
absolute slab membership nor an error box makes its selected true
state an almost-everywhere random draw. Singular preparations,
fixed-energy subsets and finite numerical grids require separate
evidence. A stationary target member has only trivial recurrence;
the family theorem does not turn it into a pulse.

This combines two existing mechanisms under compatible conservative
hypotheses: an all-state phase barrier and nonlinear volume-preserving
recurrence. It establishes a supplied coherent identity compatible
with persistent recurrent motion, not spontaneous formation,
microscopic selection of zero loss or fractal inheritance.

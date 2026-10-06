# Sufficient unordered pair state

Realizable pair invariants, internal-state reconstruction, unequal-capacity correlations and support-symmetry boundaries; global cancellation uses the declared fixed conservative doubled C5.

Part of [Coarse-graining, coherence geometry and bridge results](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="sine-replica-inheritance"></a>

## 7. Nonlinear sine inheritance with retained internal coordinates

This result uses the complete
[normalized-sine relational law](RESONANCE_FOUNDATIONS.md#reciprocal-exchange),
with `e=0`, `w,beta>0`, fixed unit support, strictly positive held
capacities, one common structural clock, and no input, clipping or operator
event. It extends the existing
[synchronized replica construction](RELATIONAL_EXCHANGE_ADMISSION.md#4-origin-units-and-exact-replication)
by retaining the coordinates transverse to that construction. It does not
transfer the native resultant-pressure law or the independently supplied
phase law used in earlier sections to this model.

### 7.1. Supplied support and an invertible description

Let a finite connected simple unit base graph have degrees `d_i>0`.
Replace each node `i` by the ordered pair `(i,+),(i,-)`. For every base
edge `{i,j}` retain all four fine edges `{(i,s),(j,t)}`, with
`s,t in {+,-}`, and no other edge. In particular, there is no edge within
a pair. The fine degree is `2*d_i`. Both members of pair `i` have the
same held capacity `nu_i>0`; different pairs may have different capacities.
The support and partition are supplied, not produced by the dynamics.

Use the exact invertible form coordinates and declared continuous phase
lifts

\[
x_{i,\pm}=X_i\pm u_i,\qquad
\theta_{i,\pm}=\Theta_i\pm\delta_i .
\]

Here `X,Theta` describe the paired means and `u,delta` their internal
differences. No fine node or coordinate has disappeared. Near synchronized
phases, choose the pair chart `|delta_i|<pi/2`, so the two phases are not
antipodal and `Theta_i` is their unambiguous circular midpoint. Beyond a
chosen chart, real-lift averages are not a globally defined circular
observation; another lift may change the midpoint by `pi`. The equations
below hold on supplied continuous lifts, with the circular state obtained
by projection, rather than by repeatedly guessing a phase unwrapping.

Put `a=w/pi`, `b=w/(beta*pi)` and
`Delta_ji=Theta_j-Theta_i`. The fine nodal rows are

\[
\dot x_{i,s}
 =\frac{a\nu_i}{2d_i}
   \sum_{j\sim i}\sum_{t=\pm1}
   \sin(\Theta_j+t\delta_j-\Theta_i-s\delta_i),
\]

\[
\dot\theta_{i,s}
 =\frac{b\nu_i}{2d_i}
   \sum_{j\sim i}\sum_{t=\pm1}
   (X_i+s u_i-X_j-t u_j).
\]

Taking their half-sum and half-difference gives the **complete nonlinear
retained-coordinate law**

\[
\boxed{\begin{aligned}
\dot X_i
 &=\frac{a\nu_i}{d_i}\cos\delta_i
   \sum_{j\sim i}\cos\delta_j\sin\Delta_{ji},\\
\dot\Theta_i
 &=\frac{b\nu_i}{d_i}\sum_{j\sim i}(X_i-X_j),\\
\dot u_i
 &=-\frac{a\nu_i}{d_i}\sin\delta_i
   \sum_{j\sim i}\cos\delta_j\cos\Delta_{ji},\\
\dot\delta_i&=b\nu_i u_i .
\end{aligned}}
\]

The identities
`sin(A+B)+sin(A-B)=2*sin(A)*cos(B)` and the cancellation of the two
neighbor form differences prove these rows directly. They are not a
tangent approximation or a fitted effective pressure. In particular,
the coarse phase row is autonomous in `X`, but the coarse form row still
consumes the internal phases.

### 7.2. Internal phase geometry modulates inherited coupling

The mean fine phasor in a pair is exactly

\[
Z_i=\frac{e^{i\theta_{i,+}}+e^{i\theta_{i,-}}}{2}
   =e^{i\Theta_i}R_i,\qquad R_i=\cos\delta_i .
\]

In the nonantipodal midpoint chart, `R_i>0` is the magnitude of this
resultant. The inherited form row and its amplitude derivative are

\[
\dot X_i=\frac{a\nu_i}{d_i}
  \sum_{j\sim i}R_iR_j\sin(\Theta_j-\Theta_i),
\qquad
\dot R_i=-b\nu_i u_i\sin\delta_i.
\]

Thus the factor `R_i*R_j` is calculated from retained internal phase
geometry. It is a dynamical contribution to the paired response, not a
threshold that activates a control action. Its evolution still requires
internal information. Using this resultant does not justify dropping
`u_i` or the phase difference sign, and no general closure in
`(X,Theta,R)` is claimed.

The original fine support remains fixed. The varying factor neither
creates a new edge nor derives an occurrence time for one. Moreover,
`Theta_dot` still contains the unmodified base Laplacian and the original
degree `d_i`. Replacing every adjacency or degree by `R_i*R_j` would
change that row and define another model. This is an inherited
phase-to-form interaction with internal state, not the unchanged bare
coarse law on an arbitrarily weighted graph.

### 7.3. Exact synchronized inheritance and storage scaling

If `u=delta=0` initially, their two rows vanish. Smooth uniqueness makes
this synchronized submanifold invariant. The remaining rows are exactly
the original conservative sine law on the base graph, with the same
`nu_i,beta,w` and clock. This is inheritance on a supplied invariant
submanifold, not synchronization from a general preparation.

Write

\[
E_c(X,\Theta)=\frac12\sum_{\{i,j\}}(X_i-X_j)^2
              +\beta\sum_{\{i,j\}}[1-\cos(\Theta_j-\Theta_i)] .
\]

Summing over the four fine edges associated with each base edge yields
the exact identity

\[
\boxed{
E_f=4E_c+
       2\sum_i d_i u_i^2+
       4\beta\sum_{\{i,j\}}\cos(\Theta_j-\Theta_i)
                   [1-\cos\delta_i\cos\delta_j] .}
\]

For the form term, the four squared differences sum to
`4*((X_i-X_j)^2+u_i^2+u_j^2)`. For phase, their four cosines sum to
`4*cos(Theta_j-Theta_i)*cos(delta_i)*cos(delta_j)`. These two
identities give the displayed formula and, on synchronization, `E_f=4*E_c`.
Neither capacity nor the clock has been rescaled to obtain the factor four.

Both full fine storage and synchronized coarse storage are conserved in
their respective conservative laws. Off the synchronized submanifold,
`E_c` is not independently conserved in general: the other retained terms
exchange storage with it. Their displayed phase contribution is
nonnegative in a local acute coarse chart, but not for arbitrary coarse
phase differences. Thus the algebraic decomposition alone is not a
global positive split into separate reservoirs.

### 7.4. A nearby exact obstruction to autonomous block means

Take the base `C5`, ordered `0,1,2,3,4`, with
`alpha=2*pi/5`, constant `X` and `Theta_j=j*alpha`. First prepare
`u=delta=0`. This is an exact equilibrium of the fine and coarse laws.
Now keep the same `X,Theta,u=0` and change only
`delta_0=epsilon`, with every other `delta_i=0` and

\[
0<|\epsilon|<\pi/2-\alpha.
\]

The fine edges remain strictly acute, and the entire preparation can be
arbitrarily close to the synchronized target. Nevertheless its exact
coarse form rates are

\[
\dot X_1=\frac{a\nu_1}{2}\sin\alpha(1-\cos\epsilon)>0,\qquad
\dot X_4=-\frac{a\nu_4}{2}\sin\alpha(1-\cos\epsilon)<0,
\]

with `X_dot=0` at the other three nodes. All coarse phase rates remain
zero at this instant. The first preparation had all these rates zero.
The two states therefore have identical observed `(X,Theta)` and
different derivatives. No autonomous first-order vector field on those
block means can reproduce every fine state in any neighborhood of this
target. This is a failure of that proposed coarse closure, not a failure
of the complete retained-coordinate nodal law.

The all-state closure obstruction also holds for any connected base graph
in the stated domain. Choose a node `j`, set its coarse phase to a small
`gamma!=0` and all other coarse phases to zero, and compare `delta=0`
against a preparation with only `delta_j=epsilon!=0`. At every neighbor
`i` of `j` the coarse form defect is
`(a*nu_i/d_i)*(cos(epsilon)-1)*sin(gamma)!=0`. The two preparations have
the same coarse state; choosing `|gamma|+|epsilon|<pi/2` keeps the
comparison within the pair chart with all fine phase gaps acute.
This general obstruction does not require that
either preparation be a critical target.

Internal form can also be hidden from an instantaneous rate comparison.
At the same target, prepare `delta=0` and only `u_k=U!=0`. Define the
coarse form defect relative to the base sine row by

\[
D_i=\frac{a\nu_i}{d_i}\sum_{j\sim i}
       (\cos\delta_i\cos\delta_j-1)\sin\Delta_{ji}.
\]

Initially `D=0` and `D_dot=0`. For a neighboring node `i`, however,

\[
\ddot D_i(0)
 =-\frac{a\nu_i}{2}
     \sin(\Theta_k-\Theta_i)(b\nu_kU)^2 .
\]

This follows by differentiating the exact cosine factors twice and using
`delta_k_dot=b*nu_k*U`; their value and first derivative vanish in the
defect at the preparation. It is an exact field-jet statement, not a
trajectory or a tangent replacement of the nonlinear law. Internal form
changes internal phase and hence the later coarse response even when the
initial coarse rate defect vanishes.

### 7.5. Trapping permits persistent internal organization

For this full ten-node fine graph, the lifted twist is itself an exact
acute critical target. On pair-constant vectors its combinatorial
Laplacian is twice the `C5` Laplacian; on pair-antisymmetric vectors it is
`4*I`. Thus its smallest positive eigenvalue is

\[
\lambda_{2,f}=\min(2\lambda_{2,C5},4)=5-\sqrt5>0 .
\]

Every target edge has cosine `cos(alpha)>0`. The existing
[local first-exit proof](SINE_PATTERN_RECOVERY.md#sine-relative-pattern-state)
therefore applies to the **full** fine graph. With centered fine form
and target phase deviations of combined squared norm `Z_f^2`, choose
`r>0` such that `alpha+sqrt(2)*r<pi/2` and set

\[
\kappa_f=\frac{5-\sqrt5}{2}
  \min\{1,\beta\cos(\alpha+\sqrt2r)\}.
\]

If initially

\[
Z_f^2<r^2,\qquad E_f-4E_{*,C5}<\kappa_f r^2,
\]

conservation and the strict boundary coercivity exclude a first exit.
The argument is valid in both time directions. It keeps both internal
and collective coordinates in the local tube: in the declared chart,

\[
Z_f^2=
2\|PX\|^2+2\|P(\Theta-\Theta_*)\|^2
+2\|u\|^2+2\|\delta\|^2,\qquad
P=I-\frac15\mathbf1\mathbf1^T .
\]

This is a sufficient trapping condition on the full state, not a sum of
five independent pair certificates. All target cycles retain their
winding inside the acute tube. The previous small-`epsilon` obstruction
can be chosen within such a tube, so persistent fine organization and
failure of block-mean closure are compatible.

The transverse linearization at the synchronized twist is

\[
\dot u_i=-a\nu_i\cos\alpha\,\delta_i,\qquad
\dot\delta_i=b\nu_i u_i .
\]

Its eigenvalues are
`+/-i*nu_i*sqrt(a*b*cos(alpha))`. These conditional tangent modes describe
internal exchange, not decay or a nonlinear period certificate. The full
conservative flow preserves volume. The existing
[nonlinear recurrence theorem](RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
implies that almost every point outside the synchronized submanifold in
an admitted finite-volume trapped family cannot converge to it: near
returns to its initial state retain a positive distance from that closed
submanifold. Conservative trapping therefore does not establish attracting
synchronization. This does not assert recurrence of a selected moving
state or supply an observed return time.

The same support also distinguishes an unstable geometry without changing
the law. The uniform `C5` twist with winding two is another exact critical
target, with edge cosine `cos(4*pi/5)<0`. Its transverse equation is

\[
\ddot\delta_i
 =-ab\nu_i^2\cos(4\pi/5)\,\delta_i ,
\]

so the tangent has a positive real eigenvalue
`nu_i*sqrt(-a*b*cos(4*pi/5))`. The smooth full sine equilibrium is
therefore locally unstable. In contrast, winding one's positive edge
cosine supports the preceding full-state trapping domain. This is a
conditional geometric distinction under identical support and coefficients:
losing the winding-two organization does **not** mean that its storage is
dissipated. With `e=0` the total storage remains constant. It also does
not prove that every perturbation of the unstable target follows the same
transition or ends in the winding-one family.

For the different law `e>0`, decay and target recovery must use the
existing [positive-loss theorem](RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
and [local recovery hypotheses](SINE_PATTERN_RECOVERY.md#sine-relative-pattern-state).
The conservative recurrence argument no longer applies, and the
conservative transverse modes cannot be relabeled as its attractors.

### 7.6. Meaning and limits of the scale result

The paired description preserves a larger organization together with its
internal fine patterns. It does not require that the fine patterns vanish
when a collective description is introduced. Conversely, grouping two
nodes does not by itself establish either constituent or the group as an
autonomously formed NFR; their preparation, identity and support remain
explicit premises here.

Exact same-law inheritance is proved on synchronization. Near that
submanifold, the exact larger description retains internal coordinates,
and their emergent resultant amplitudes modify collective interactions.
This supplies a specific relation between organization and internal
exchange. It does not establish generic autonomous coarse closure,
spontaneous hierarchical formation, a scale-independent fractal dimension,
a universal pulse, or a microscopic physical identification.

The detached
[replica observer](../../src/tnfr/physics/relational_sine_scale.py)
retains the complete captured graph and independently compares fine
half-sum/half-difference rates with the factored expressions. It uses exact
represented form values and supplied phase-lift integers, and retains
interval bounds for trigonometric quantities. Zero-containing arithmetic
residuals are checks of that capture, distinct from the all-state
identities proved above. The
[replica tests](../../tests/physics/test_relational_sine_replica.py)
separate synchronized inheritance, hidden internal motion, nonlinear
coarse nonclosure and the two target geometries. The observer does not
certify an arbitrary capture as lying in the local trapping domain or
execute a trajectory.

<a id="sine-replica-unordered-state"></a>

## 8. An exact collective state for unordered pairs

Keep the complete conservative sine law, complete two-replica support,
strictly positive held paired capacities and nonantipodal pair chart of
[section 7](#sine-replica-inheritance). There are no external inputs or
node-selected events. Swapping the two members of any pair is then an
exact symmetry: it preserves the support, capacities and complete nodal
rows. This section removes only those independent swap labels. Internal
form and phase information remain in the collective description.

### 8.1. Realizable invariants and their fibers

For each pair define

\[
R_i=\cos\delta_i,\qquad U_i=u_i^2,\qquad
Q_i=u_i\sin\delta_i .
\]

The symbol `Q_i` here denotes an internal form-phase correlation, not
nodal pressure `p=DeltaNFR`. Their units are respectively `1`, `[x]^2`
and `[x]`. Each is unchanged by the pair swap
`(u_i,delta_i)->(-u_i,-delta_i)`. The retained coarse coordinates
`X_i,Theta_i` are unchanged as well.

Their exact realized domain is

\[
\boxed{\quad
0<R_i\le1,\qquad U_i\ge0,\qquad
Q_i^2=U_i(1-R_i^2).
\quad}
\]

Necessity follows from `|delta_i|<pi/2` and the sine-cosine identity.
For sufficiency and uniqueness of the unordered pair:

- If `0<R_i<1`, choose
  `delta_i=arccos(R_i)>0` and
  `u_i=Q_i/sqrt(1-R_i^2)`. The constraint gives `u_i^2=U_i`.
  The other possible lift is its simultaneous negative, exactly the pair
  swap. This also covers `U_i=Q_i=0` with nonzero phase separation.
- If `R_i=1`, the chart forces `delta_i=0` and the constraint forces
  `Q_i=0`. The choices `u_i=+sqrt(U_i)` and `u_i=-sqrt(U_i)`
  are the same unordered pair.
- At `R_i=1,U_i=Q_i=0` there is just one lift,
  `u_i=delta_i=0`.

Thus `(X,Theta,R,U,Q)` separates all fine states in this chart up to
the independent pair swaps. Together with the supplied support and held
capacities, it specifies the unordered fine configuration. In particular,
the fully coherent phase value `R_i=1` does not imply synchronized
form: `U_i>0` remains distinguishable.

This is a quotient by a finite symmetry group, not a reduction in the
generic number of continuous degrees of freedom. Each internal pair has
two degrees of freedom, represented by three invariants with one
constraint. At the synchronized tip `(R,U,Q)=(1,0,0)`, the gradient of
`Q^2-U*(1-R^2)` vanishes and the swap fixes the original state. These
invariants are not an ordinary smooth coordinate chart at that tip.
Their algebraic singularity must not be interpreted as an extra physical
degree of freedom, a dynamical instability or failure of the original
smooth nodal law.

### 8.2. Closed dynamics with internal state retained

Let

\[
A_i=\frac{a\nu_i}{d_i}
       \sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),
\qquad c_i=b\nu_i .
\]

These are calculated from the retained state and declared coefficients,
with units `[A]=[x]/[t]` and `[c]=1/([x][t])`. They introduce no new
constitutive parameter. The original internal rows are
`u_i_dot=-A_i*sin(delta_i)` and `delta_i_dot=c_i*u_i`.
Differentiating the invariants gives

\[
\boxed{\begin{aligned}
\dot X_i&=\frac{a\nu_i}{d_i}
             \sum_{j\sim i}R_iR_j\sin(\Theta_j-\Theta_i),\\
\dot\Theta_i&=\frac{b\nu_i}{d_i}
             \sum_{j\sim i}(X_i-X_j),\\
\dot R_i&=-c_iQ_i,\\
\dot U_i&=-2A_iQ_i,\\
\dot Q_i&=-A_i(1-R_i^2)+c_iU_iR_i .
\end{aligned}}
\]

Every consumed quantity is determined by the unordered collective state.
These are exact nonlinear quotient rows. Unlike block means alone, the
augmented state closes because it retains the internal information that
caused the earlier obstruction. No hidden initial condition was set to
zero, fitted to a response or erased.

The constraint is an exact first integral of these rows:

\[
\begin{aligned}
\frac{d}{dt}[Q_i^2-U_i(1-R_i^2)]
={}&2Q_i[-A_i(1-R_i^2)+c_iU_iR_i]\\
 &+2A_iQ_i(1-R_i^2)-2c_iU_iR_iQ_i=0 .
\end{aligned}
\]

This algebra alone does not prove realizability: the same equality also
has nonphysical solutions such as `R_i>1,U_i<0`. The inequalities,
phase chart and lifting argument are essential.

### 8.3. Evolution, boundaries and lift independence

Start with a realizable collective state and choose any fine lift supplied
by section 8.1. Evolve the original smooth fine nodal law. On every time
interval for which all pairs remain in `|delta_i|<pi/2`, its invariants
solve the displayed collective rows and remain realizable.

Conversely, those rows define a smooth vector field in the ambient
`(X,Theta,R,U,Q)` variables; there is no division by `U`, `Q` or
`1-R^2` in them. Local uniqueness therefore makes every collective
solution from that initial state equal to the invariant image of the
fine solution. Choosing the other fine lift merely performs fixed pair
swaps, which are symmetries of the full law. It yields the same collective
solution. This proves dynamical sufficiency even at a tip where the
inverse invariant coordinates are not differentiable.

Two boundary controls make the role of the retained variables explicit:

\[
\begin{array}{ll}
R_i=1,\ U_i>0:
 &\dot Q_i=c_iU_i,\quad
   \ddot R_i=-c_i^2U_i<0,\\[2mm]
U_i=0,\ 0<R_i<1:
 &\dot Q_i=-A_i(1-R_i^2),\quad
   \ddot U_i=2A_i^2(1-R_i^2)\ge0 .
\end{array}
\]

The first state has aligned primitive phases but a form difference that
immediately changes its internal correlation and subsequently its
resultant amplitude. The second can acquire a form difference from
phase separation when `A_i!=0`. At the tip `R_i=1,U_i=Q_i=0` all
three internal derivatives vanish. That internal tip remains invariant
even when the means or neighboring internal states evolve.

No further chart guarantee follows just from the collective equality
constraint. At `R_i=0` the primitive phases are antipodal, their mean
phasor vanishes, and its circular midpoint is not uniquely defined.
The original fine law remains smooth there. The chosen collective chart
must end or be replaced by a separately justified chart; continuing a
signed `R` or clipping it to zero is not an automatic continuation of
this unordered observation. The present result is exact up to that chart
boundary unless an independent trapping condition excludes it.
For the fixed conservative unit doubled C5, Section 29 provides a
[global invariant representation](#sine-global-pair-state) through the boundary;
it does not extend this midpoint chart or its input contract.

### 8.4. Storage and identity pass to the exact quotient

The full fine storage becomes the invariant function

\[
\boxed{
\mathcal H(X,\Theta,R,U)=
2\sum_{\{i,j\}}(X_i-X_j)^2+
2\sum_i d_iU_i+
4\beta\sum_{\{i,j\}}
 [1-R_iR_j\cos(\Theta_j-\Theta_i)] .
}
\]

This is exactly `E_f` from section 7, not an additional Hamiltonian or
an independently calibrated energy. Its conservation follows by lifting
the collective rows to the same conservative law, or by direct
differentiation using those rows. It retains internal form storage even
at `R_i=1`. On the realized domain every displayed contribution is
nonnegative. Setting `R=1,U=Q=0` recovers `4*E_c` and the supplied
synchronized same-law sector.

The previous full ten-node acute-twist barrier can also be expressed
without the swap labels. In its declared target chart,

\[
Z_f^2=
2\|PX\|^2+2\|P(\Theta-\Theta_*)\|^2
+2\sum_i[U_i+(\arccos R_i)^2],
\qquad P=I-\frac15\mathbf1\mathbf1^T .
\]

This equals the fine centered norm because
`delta_i^2=(arccos(R_i))^2` in the admitted pair chart. Thus the
existing sufficient conditions
`Z_f^2<r^2` and
`H-4*E_*,C5<kappa_f*r^2` admit exactly the same trapping conclusion
when read through these invariants. In particular,

\[
U_i<r^2/2,\qquad
R_i>\cos(r/\sqrt2)>0
\]

throughout such a trapped trajectory. The acute radius condition in
section 7 implies `r/sqrt(2)<pi/2`. Consequently that independent
full-state barrier keeps the unordered pair chart valid for all time;
the constraint by itself did not do so. This rewrites a proved barrier
rather than creating another approximation, recovery criterion or
trajectory calculation.

### 8.5. What has and has not been reduced

The result removes a redundant choice of member labels while retaining
the complete unordered pair, including its internal form variance,
phase separation and their signed correlation. The collective object
therefore still contains its smaller constituents and their dynamics.
It is not a claim that a pair has become a single fundamental scalar
node with the original bare nodal state space.

The complete bipartite support, partition, equal paired capacities and
phase chart remain assumptions. Unequal member capacities, selective
forcing, member-specific events or another fine support can break the
swap symmetry and require another observation model. No autonomous
partition formation, generic fractality, unlimited hierarchical closure,
material particle or microscopic law selection follows from this finite
symmetry quotient.

The [existing replica observer](../../src/tnfr/physics/relational_sine_scale.py)
exposes the invariant state of an admitted fine capture and compares
its directly calculated derivatives with these closed rows. Its exact
represented `U` and interval `R,Q` retain the fine capture and
arithmetic provenance; an interval constraint residual containing zero
does not admit every tuple in that surrounding rectangular box.
The [same test owner](../../tests/physics/test_relational_sine_replica.py)
checks pair-swap invariance, boundary controls, direct fine-to-collective
rates and storage. This is not an API for accepting arbitrary collective
tuples, reconstructing a chosen numerical lift, or running a trajectory.

<a id="sine-replica-inherited-poisson"></a>

### 8.6. The collective state inherits the same Poisson structure

The exact quotient inherits more than a closed vector field. Use the
[same-law Poisson tensor](RESONANCE_FOUNDATIONS.md#reciprocal-exchange),
with bracket convention `dot f={f,E_f}`. Each fine member has degree
`2*d_i`, hence

\[
\{x_{i,\pm},\theta_{i,\pm}\}
   =-\frac{b\nu_i}{2d_i}.
\]

All other fine coordinate brackets vanish. The mean/half-difference
change therefore gives

\[
\{X_i,\Theta_i\}=\{u_i,\delta_i\}=-\kappa_i,\qquad
\kappa_i=\frac{b\nu_i}{4d_i}>0,
\]

with zero brackets between distinct pairs and between a pair's mean
and internal coordinates. The factor four follows from both the fine
degree and the two factors of one half in the coordinate change.
Holding support and paired capacities fixed makes each `kappa_i`
constant.

The chain rule for `R=cos(delta), U=u^2, Q=u*sin(delta)` gives

\[
\boxed{
\{R_i,U_i\}=-2\kappa_iQ_i,\qquad
\{R_i,Q_i\}=-\kappa_i(1-R_i^2),\qquad
\{U_i,Q_i\}=-2\kappa_iU_iR_i .
}
\]

Together with `{X_i,Theta_i}=-kappa_i`, these are all the
nonzero independent brackets. They descend under every pair swap:
simultaneously negating `u_i,delta_i` preserves their constant
bracket and leaves the invariant functions unchanged.

Jacobi follows by pushforward of the fine bracket on the invariant
algebra. It can also be checked without choosing a lift. Set

\[
\mathcal C_i=Q_i^2-U_i(1-R_i^2).
\]

In the ordered coordinates `(R_i,U_i,Q_i)`, the internal tensor is
`J_ab=-kappa_i*epsilon_abc*partial_c C_i`. A three-dimensional
bracket of this form satisfies Jacobi because its defining vector
`-kappa_i*grad C_i` has zero curl. Moreover
`J*grad C_i=0`, so `C_i` is a **Casimir**: its bracket with
every smooth retained function vanishes, not only with this storage.
The direct sum with the constant mean blocks is again Poisson.

For a realized nontip state the internal tensor has rank two.
If `R_i<1`, its `(R_i,Q_i)` entry is nonzero; if `R_i=1`
and `U_i>0`, its `(U_i,Q_i)` entry is nonzero. At the tip
`(R_i,U_i,Q_i)=(1,0,0)` it has rank zero. The independent mean
block retains rank two there. This is the finite-symmetry quotient's
singular internal stratum, not an additional continuous degree of
freedom or a singularity of the fine nodal law.

The Hamiltonian is exactly the already derived `H=E_f` in section
8.4. Its nonzero coordinate derivatives are

\[
\begin{aligned}
\partial_{X_i}\mathcal H&=4\sum_{j\sim i}(X_i-X_j),\\
\partial_{\Theta_i}\mathcal H
 &=-4\beta R_i\sum_{j\sim i}R_j\sin(\Theta_j-\Theta_i),\\
\partial_{R_i}\mathcal H
 &=-4\beta\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),\\
\partial_{U_i}\mathcal H&=2d_i,\qquad
\partial_{Q_i}\mathcal H=0 .
\end{aligned}
\]

Using `a=beta*b`, the mean brackets reproduce the exact `X,Theta`
rows, while the internal brackets give

\[
\begin{aligned}
\{R_i,\mathcal H\}&=-4\kappa_i d_iQ_i=-c_iQ_i,\\
\{U_i,\mathcal H\}
 &=-8\kappa_i\beta Q_i\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i)
   =-2A_iQ_i,\\
\{Q_i,\mathcal H\}
 &=-4\kappa_i\beta(1-R_i^2)
       \sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i)
      +4\kappa_i d_iU_iR_i\\
 &=-A_i(1-R_i^2)+c_iU_iR_i .
\end{aligned}
\]

Thus all five collective rows, the internal constraint and storage
conservation share one inherited structure. No Hamiltonian, pressure
channel or feedback controller is added.

The polynomial Poisson identity also exists outside the realized set;
it does not admit those states. The inequalities, lifting and chart
continuation conditions of sections 8.1–8.3 remain essential. In
particular, a Casimir value or a numerical residual cannot establish
realizability, and `R=0` still requires a different collective chart.
Likewise, a recurrence argument must retain the fine-state measure or
a justified quotient measure: the constrained invariants are not
independent variables carrying full-dimensional Lebesgue volume.
The `e=0` law, supplied support and equal held member capacities remain
premises; this identity selects neither microscopic loss nor a physical
interpretation of storage.

<a id="sine-replica-internal-observability"></a>

### 8.7. Temporal resultant information recovers the unordered internal state

The same closed rows yield an exact observation consequence, without
introducing another variable or law. Retain the section 8 premises and
the structural clock. Suppose the means `X,Theta`, all neighboring
resultant amplitudes `R` and the held coefficients are independently
known. Thus `c_i=b*nu_i>0` and
`A_i=(a*nu_i/d_i)*sum_j R_j*cos(Theta_j-Theta_i)` are known;
only relative neighboring mean phases are consumed by `A_i`.
For observations of an admitted smooth fine trajectory,

\[
\dot R_i=-c_iQ_i,\qquad
\ddot R_i=c_iA_i(1-R_i^2)-c_i^2U_iR_i.
\]

Consequently, everywhere in the local chart `R_i>0`,

\[
\boxed{
Q_i=-\frac{\dot R_i}{c_i},\qquad
U_i=\frac{c_iA_i(1-R_i^2)-\ddot R_i}{c_i^2R_i}.
}
\]

These observations recover the exact unordered internal state, not
the constituent labels or a preferred signed lift. Held capacity is
essential: if `c_i` varied, the second derivative would also contain
`-dot(c_i)*Q_i`. The formula does not estimate an unknown capacity,
select a clock, or identify a laboratory sensor with this resultant.

For `0<R_i<1`, the Casimir additionally gives the first-derivative
formula

\[
U_i=\frac{\dot R_i^2}{c_i^2(1-R_i^2)}.
\]

That expression becomes singular at aligned phases. The second-derivative
formula remains regular there: at `R_i=1`, any admitted trajectory
has `dot R_i=Q_i=0` but

\[
\boxed{\qquad U_i=-\ddot R_i/c_i^2.\qquad}
\]

Thus a perfectly aligned instantaneous pair can conceal a form imbalance
that its resultant curvature distinguishes. The neighbor term vanishes
at this exact boundary, although it is needed for the general
second-derivative reconstruction. This reuses the boundary dynamics
of section 8.3; zero first derivative alone did not imply a stationary
internal state.

The inverse retains admission and observation limits. Its output must
satisfy `0<R_i<=1`, `U_i>=0` and the exact Casimir constraint.
Small `c_i` or proximity to `R_i=0` amplifies uncertainty in the
second-derivative inverse; the first-derivative inverse instead loses
conditioning near `R_i=1`. Independent intervals for state and
derivatives need not come from one joint trajectory. Numerical
differentiation requires its own sampling, timing and remainder bounds;
neither interval overlap nor this exact identity supplies them.
There is no arbitrary noisy-tuple admission, raw-data reconstruction
API or replacement of retained internal state by the single value R.

<a id="sine-replica-capacity-asymmetry"></a>

## 13. Unequal constituent capacities: persistence and its symmetry boundary

Keep the same complete conservative sine law and doubled-cycle support.
Change only the supplied held capacities: each constituent now has its
own `nu_(i,+)>0` and `nu_(i,-)>0`. Define

\[
v_i=\frac{\nu_{i,+}+\nu_{i,-}}2,\qquad
\eta_i=\frac{\nu_{i,+}-\nu_{i,-}}2,\qquad
v_i>|\eta_i|.
\]

The letter `v_i` in this section is mean capacity, not the centered
fine-form vector used in Section 12. Both `v_i` and `eta_i` are
held parameters. The same form/phase coordinates
`x_(i,+/-)=X_i+/-u_i`, `theta_(i,+/-)=Theta_i+/-delta_i`
remain invertible on supplied lifts. The pair chart is still
`|delta_i|<pi/2`. No new force, state law or capacity evolution is
postulated.

### 13.1. Exact asymmetric mean and internal rows

For a base degree `d_i` define the normalized neighboring sums

\[
S_i=\frac1{d_i}\sum_{j\sim i}R_j\sin(\Theta_j-\Theta_i),
\quad
C_i=\frac1{d_i}\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),
\quad
G_i=\frac1{d_i}\sum_{j\sim i}(X_i-X_j),
\quad R_i=\cos\delta_i .
\]

For the doubled `C5`, `d_i=2`. Before multiplying by the
individual capacity, the two fine form rows are
`a*(R_i*S_i-/+sin(delta_i)*C_i)`, and the two fine phase rows
are `b*(G_i+/-u_i)`. Their exact half-sums and half-differences
therefore give

\[
\boxed{\begin{aligned}
\dot X_i&=a(v_iR_iS_i-\eta_i\sin\delta_i\,C_i),\\
\dot\Theta_i&=b(v_iG_i+\eta_i u_i),\\
\dot u_i&=a(\eta_iR_iS_i-v_i\sin\delta_i\,C_i),\\
\dot\delta_i&=b(\eta_iG_i+v_i u_i).
\end{aligned}}
\]

The additional terms are consequences of multiplying the existing fine
rows by unequal capacities. Setting `eta_i=0` recovers Section 7,
even if `v_i` differs between pairs. Variation **between** equal
member pairs and inequality **within** a pair are thus distinct
changes. No rounded near-equality can replace the exact equality used
in the earlier invariant-tip theorem.

The inherited signed Poisson bracket makes the same distinction.
With `k_i=b/(4*d_i)` its nonzero independent entries are

\[
\{X_i,\Theta_i\}=\{u_i,\delta_i\}=-k_i v_i,\qquad
\{X_i,\delta_i\}=\{u_i,\Theta_i\}=-k_i\eta_i .
\]

Different pairs have zero brackets. These entries follow by taking
half-sums/differences of
`{x_(i,+/-),theta_(i,+/-)}=-b*nu_(i,+/-)/(2*d_i)`.
The bracket is constant for held capacities and hence satisfies Jacobi.
Its determinant in each four-coordinate block is
`k_i^4*(v_i^2-eta_i^2)^2>0`. The same full fine storage from
Section 7 generates all four displayed rows, including the cross terms.
Equal capacities eliminate those cross terms; they do not supply the
underlying storage balance.

### 13.2. Collective trapping and ambient recurrence survive

The exact storage still satisfies `E_f_dot=0`. The full field
still has zero divergence: each fine form row consumes phase, and
each fine phase row consumes form, with all capacities held. Its
conserved form mean is now

\[
m_\rho(x)=
\frac{\sum_{i,s}\rho_{i,s}x_{i,s}}{\sum_{i,s}\rho_{i,s}},
\qquad \rho_{i,s}=\frac{2d_i}{v_i+s\eta_i}>0 .
\]

In retained coordinates this is

\[
\boxed{
m_\rho=
\frac{\displaystyle\sum_i
      \frac{d_i(v_iX_i-\eta_i u_i)}{v_i^2-\eta_i^2}}
     {\displaystyle\sum_i
      \frac{d_i v_i}{v_i^2-\eta_i^2}}.}
\]

The ordinary mean is not a substitute. For the same winding-one
target the fine storage, spectral gap and acute geometry do not depend
on capacity. Retain `r,kappa_f,E_*` from Section 12, choose
`0<epsilon_*<kappa_f*r^2` and a finite open interval
`m_lo<m_hi`, and define

\[
\mathcal U_\rho=
\{Z_f^2<r^2,\ E_f-E_*<\varepsilon_*,
                 \ m_{\rm lo}<m_\rho<m_{\rm hi}\}.
\]

For each supplied positive held capacity list this is an open invariant
family with finite positive fine volume. The same energy argument
prevents radius exit in both time directions. To verify the volume
claim explicitly, use centered form `z=P_f x` and write

\[
x=\left(m_\rho-\frac{\rho^\mathsf Tz}
                          {\rho^\mathsf T\mathbf1}\right)\mathbf1+z .
\]

This is an invertible form coordinate change. Positive `rho` makes
`m_rho` a convex average, so

\[
|x_j-m_\rho|
\le\max_k|x_j-x_k|
\le\sqrt2\,\|z\|<\sqrt2r .
\]

Thus the closure of `U_rho` lies in a bounded form box times the
phase torus; target neighborhoods prove positive volume. The smooth
two-sided flow preserves the family and its fine ambient measure.
The existing recurrence proof gives almost-everywhere nonstationary
recurrence there, with its original selected-state and singular-measure
limitations. This family includes synchronized tips: deleting them
would generally destroy invariance, as the next subsection shows.

These conclusions apply separately to each fixed positive capacity
list. They do not install a time-varying capacity law, give a common
return deadline, or transfer the equal-capacity internal-circulation
claim. The singular boundary at a zero constituent capacity is outside
this theorem.

### 13.3. Exact counterexamples to tip invariance and pointwise activity

At a synchronized tip `u_i=delta_i=0` the asymmetric internal
rows reduce to

\[
\dot u_i=a\eta_i S_i,\qquad
\dot\delta_i=b\eta_i G_i .
\]

They need not vanish. The tip and its complement are therefore not
generally invariant. This is a structural obstruction, not a numerical
threshold effect.

An explicit preparation also shows that a **nontip** pair can have
zero instantaneous internal velocity. Take `v_i=1` for all pairs,
`eta_0=eta` with `0<eta<1`, and all other `eta_i=0`.
Use exact target phases `Theta_i=i*alpha` and `delta_i=0`.
Prepare

\[
X_0=\epsilon,\quad X_{i\ne0}=0,\qquad
u_0=-\eta\epsilon,\quad u_{i\ne0}=0,\qquad \epsilon>0.
\]

All `S_i=0` and `G_0=epsilon`. Hence

\[
\dot u_0=\dot\delta_0=0,\qquad
u_0\ne0,\qquad
\dot\Theta_0=b\epsilon(1-\eta^2)>0 .
\]

The pair's old invariant rates `R_dot,U_dot,Q_dot` all vanish
at this instant, although its mean phase moves. This also makes the
normalized signed internal angular velocity zero, excluding the earlier
strict positive angular lower bound. The storage excess and full norm
are exactly

\[
E_f-E_*=4(1+\eta^2)\epsilon^2,\qquad
Z_f^2=\left(\frac85+2\eta^2\right)\epsilon^2 .
\]

For example `eta=1/2,epsilon=1/1024` gives capacities
`(nu_(0,+),nu_(0,-))=(3/2,1/2)` and forms
`(x_(0,+),x_(0,-))=(epsilon/2,3*epsilon/2)`, with
excess `5*epsilon^2` and norm squared `21*epsilon^2/10`.
Its conserved weighted mean is `5*epsilon/16`, rather than
the ordinary mean `epsilon/5`. The target phases in this proof
are symbolic angles; substituting rounded radians would not prove the
exact zero derivative.

The zero internal rate at pair zero also has an exactly represented
phase control. Choose rational `h` near `alpha` and raw pair
means `(0,h,2h,-2h,-h)`, with respective full-turn lifts
`(0,0,0,1,1)`. The two neighboring phases of pair zero are
`+h,-h`, so their sine currents cancel exactly. The same form
preparation still has `u_0_dot=delta_0_dot=0` without claiming
that the entire phase state is critical. Taking rational `h`
arbitrarily close to `alpha` preserves any strict neighborhood
admission of the symbolic preparation by continuity.

Both the asymmetry `eta` and preparation size `epsilon` can
be arbitrarily small. For any fixed admitted radius and excess ceiling,
sufficiently small `epsilon` puts the state in the collective
trapping family; a common form translation places its weighted mean
inside the chosen slab without changing any displayed rate or bound.
Thus collective geometric persistence can coexist with an internal
stall arbitrarily close to the symmetric regime.

For a separate tip-crossing control keep the same preparation but set
`u_0=0`. Now `delta_0_dot=b*eta*epsilon!=0` at the tip.
Smooth local uniqueness gives nontip states just before and after this
instant on the same full solution. Choosing the preparation inside
`U_rho` retains those states in its two-sided invariant family.
A nontip preparation can consequently reach the tip at a finite time.

Neither control proves permanent internal cessation, loss of the
collective winding, or absence of later turns. They disprove the
specific all-state no-tip-entry and nonzero-internal-velocity statements
outside their equal-member-capacity premises. They do not contradict
the surviving full-state recurrence theorem.

### 13.4. The unordered state must retain capacity correlations

Swapping only `(u_i,delta_i)` while fixing unequal labeled
capacities is no longer a symmetry. For example at `delta_i=0`,
the two forms `u_i` and `-u_i` have the same old `R,U,Q`
but different mean phase rates through `eta_i*u_i`. Even retaining
the signed parameter `eta_i` does not restore closure of those
old observed coordinates.

For an unordered pair, interchange the **whole** constituent,
including its held capacity. This exact relabeling sends
`(eta_i,u_i,delta_i)` to `(-eta_i,-u_i,-delta_i)` while
preserving `v_i,X_i,Theta_i`. The missing correlations can be
retained as

\[
P_i=\eta_i u_i,\qquad T_i=\eta_i\sin\delta_i,\qquad
E_i=\eta_i^2 .
\]

Here `E_i` is squared capacity contrast, not the total fine
storage `E_f`. Together with `R_i,U_i,Q_i` these quantities
identify the exact joint swap orbits on the pair chart. Their realizable
domain is

\[
0<R_i\le1,\quad U_i,E_i\ge0,\quad v_i>0,\quad E_i<v_i^2,
\]

\[
Q_i^2=U_i(1-R_i^2),\quad
P_i^2=E_iU_i,\quad T_i^2=E_i(1-R_i^2),\quad
P_iT_i=E_iQ_i .
\]

Equivalently the symmetric matrix

\[
\begin{pmatrix}
E_i&P_i&T_i\\
P_i&U_i&Q_i\\
T_i&Q_i&1-R_i^2
\end{pmatrix}
\]

is positive semidefinite of rank at most one: it is the outer product
of `(eta_i,u_i,sin(delta_i))` with itself. If `E_i>0`,
choose `eta_i=sqrt(E_i)`, `u_i=P_i/eta_i` and
`delta_i=asin(T_i/eta_i)`; its only other lift is the simultaneous
sign reversal. If `E_i=0`, then `P_i=T_i=0` and the
earlier `R,U,Q` lifting applies. This is a finite symmetry
quotient with retained internal information, not fewer continuous
dynamical degrees of freedom.

The exact forward equations require no division by `eta_i`:

\[
\boxed{\begin{aligned}
\dot X_i&=a(v_iR_iS_i-T_iC_i),\\
\dot\Theta_i&=b(v_iG_i+P_i),\\
\dot R_i&=-b(v_iQ_i+T_iG_i),\\
\dot U_i&=2a(P_iR_iS_i-v_iQ_iC_i),\\
\dot Q_i&=a(R_iT_iS_i-v_i(1-R_i^2)C_i)
              +bR_i(P_iG_i+v_iU_i),\\
\dot P_i&=a(E_iR_iS_i-v_iT_iC_i),\\
\dot T_i&=bR_i(E_iG_i+v_iP_i),\\
\dot E_i&=\dot v_i=0 .
\end{aligned}}
\]

These are exact derivatives of the realized invariants, so their
constraints, the same fine storage and their lift are preserved for
as long as the pair chart persists. The displayed algebra is not
permission to evolve arbitrary incompatible interval tuples; domain
preservation comes from the fine lift and smooth uniqueness. On
`U_rho` the full-state barrier keeps the chart for all time.
At `E_i=0` the correlations vanish and the equal-capacity
closure is recovered without a singular forward limit.

Their joint consistency can also be checked without expanding separate
squared constraints. The vector `z_i=(eta_i,u_i,sin(delta_i))`
obeys `z_i_dot=M_i*z_i` along the full solution, where

\[
M_i=\begin{pmatrix}
0&0&0\\
aR_iS_i&0&-av_iC_i\\
bR_iG_i&bR_iv_i&0
\end{pmatrix}.
\]

Consequently its Gram matrix obeys
`(z_i*z_i^T)_dot=M_i*(z_i*z_i^T)+(z_i*z_i^T)*M_i^T`.
This is an exact consistency identity with coefficients supplied by the
same retained state, not a separately prescribed linear model.

### 13.5. Weighted centers remove direct cross terms, not internal information

One useful coordinate check uses `lambda_i=eta_i/v_i` and
the inverse-capacity weighted pair centers

\[
\widehat X_i=X_i-\lambda_i u_i,\qquad
\widehat\Theta_i=\Theta_i-\lambda_i\delta_i,\qquad
\nu_{{\rm eff},i}=\frac{v_i^2-\eta_i^2}{v_i}
=\frac{2\nu_{i,+}\nu_{i,-}}{\nu_{i,+}+\nu_{i,-}} .
\]

Because capacities are held, the exact rows give

\[
\dot{\widehat X}_i=a\nu_{{\rm eff},i}R_iS_i,\qquad
\dot{\widehat\Theta}_i=b\nu_{{\rm eff},i}G_i .
\]

This explains the harmonic-mean capacity and cancellation of direct
cross terms in weighted coordinates. It does **not** give an autonomous
law for those means: `G_i` still consumes
`X_i=widehat(X)_i+lambda_i*u_i` and `S_i` consumes
`Theta_i=widehat(Theta)_i+lambda_i*delta_i` and neighboring
`R_j`. The weighted phase center is only a local lifted coordinate;
unequal noninteger phase weights do not define a global circular average.
Nor does a change of mean coordinates undo the actual signed
internal stall in Section 13.3.

The symmetry boundary therefore refines the persistent-pattern claim:
positive held capacities preserve the conservative storage, local
collective identity and ambient recurrence result, while the stronger
all-pair circulation theorem requires its stated symmetry. Capacity
correlations restore an exact unordered description of the asymmetric
law; they do not restore that stronger theorem, select capacities,
or prove autonomous formation of the graph or material constituents.

### 13.6. Shared engine scope

The [scale owner](../../src/tnfr/physics/relational_sine_scale.py)
exposes `assess_sine_replica_capacity` and
`SineReplicaCapacityAssessment`. It reuses one full capture and
the shared support/lift coordinates, then compares the four signed rows
and eight capacity-aware invariant rows against the fine pushforward.
Its storage gradients and two signed Poisson coefficients retain the
cross terms. Constraint residuals check that capture's realizable image;
the reader does not admit arbitrary independent invariant tuples.
Capacity contrasts refer to the captured represented values, rather
than precision discarded before that capture.

Optional `SineReplicaCapacityFamily` evidence uses the shared acute
barrier, exact degree/capacity mean weights and a declared radius,
excess ceiling and open mean slab on the complete ordered doubled
`C5`. It includes tips and distinguishes family admission, source
trapping, source membership and ambient almost-everywhere recurrence.
It supplies neither an all-time internal-activity certificate nor an
angular-circulation certificate for unequal capacities.

The earlier equal-capacity readers retain their exact capacity
requirements. The [replica tests](../../tests/physics/test_relational_sine_replica.py)
separately check fine-row factorization, capacity-state swaps,
constraints, weighted conservation and the symmetry-breaking controls.
Neither reader changes the model, adjusts a capacity, evolves a
trajectory or promotes a finite residual check to recurrence of the
captured state.

<a id="sine-pair-support-symmetry"></a>

## 18. Which attachments permit an unordered pair-state law?

### 18.1. Equivalence concerns the entire member state on a fixed graph

Keep the complete conservative normalized-sine law on a fixed finite
simple connected unit graph, with a common held capacity `nu>0`,
`w,beta>0` and no input or event. Let an explicit partition put
every fine node in a two-member pair. Define `G` as the finite
group generated by independently exchanging the two members of each
pair. Each permutation acts on **both** form and circular phase.
The support, capacities and clock stay fixed.

On the pair chart of Section 8, the retained coordinates
`(X,Theta,R,U,Q)` identify exactly these swap orbits. They retain
the internal member information up to interchange; they are not merely
pair means or a reduction in continuous dimension. The all-state
condition for removing member labels from the evolution is

\[
\boxed{F(Pz)=P F(z)\quad\hbox{for every }z
\hbox{ and every independent pair swap }P.}
\]

Equivariance suffices: smooth uniqueness gives
`Phi_t(Pz)=P Phi_t(z)`, so the orbit of a state has a
well-defined future orbit. It is also necessary for the exact
all-state orbit quotient. On the dense set where no pair consists
of identical member states, the finite action is free. Locally the
quotient is invertible up to its finitely many separate lifts.
If two such lifts have the same quotient flow, continuity forces
their initial relative permutation to remain the same for sufficiently
small time. Differentiating at zero gives the displayed identity.
Continuity of the field then extends it to the fixed strata.
The singular invariant chart at a tip does not remove this obligation.

This is different from simultaneous graph/state relabeling.
Relabeling `A` to `P A P^T` while also relabeling `z`
always describes the same unlabeled fine system. Here `A` is
held unchanged; the question is whether exchanging member states
alone can lose predictive information. Nor does failure of the
all-state criterion exclude a separately proved special invariant
subfamily or a smaller joint-swap symmetry.

### 18.2. Necessary and sufficient fixed-support condition

Write `L_rw=I-D^{-1}A` for the normalized Laplacian of this
unit graph. The complete field is

\[
\dot x=a\nu D^{-1}S(\theta),\qquad
\dot\theta=b\nu L_{\rm rw}x,\qquad
a=w/\pi,\quad b=w/(\beta\pi).
\]

Since `b*nu>0`, equivariance of the phase row for every form
already requires

\[
L_{\rm rw}P=P L_{\rm rw}.
\]

For distinct nodes `p,q`, its matrix entry is `-1/d_p`
exactly when an edge is present, and zero otherwise. Permutation
commutation therefore preserves this nonzero pattern: `P` is
an automorphism of the fixed adjacency. Conversely, a graph
automorphism preserves degrees and changes neither the linear
neighbor sum nor the nonlinear sum
`sum_j sin(theta_j-theta_i)` after the same state permutation.
It consequently commutes with **both** complete rows.

Thus, under the stated capacity and support hypotheses,

\[
\boxed{\begin{split}
\text{independent pair-state quotient for all states}
&\ \Longleftrightarrow\
\text{every independent pair swap is a graph automorphism}\\
&\ \Longleftrightarrow\
\text{each interpair block is complete or empty.}
\end{split}}
\]

The last line allows an edge within any pair. More explicitly,
members `p,q` are interchangeable precisely when
`N(p) minus {q} = N(q) minus {p}`; adjacency between `p`
and `q` is itself unchanged by their transposition. For two
different pairs, independent exchange of their row and column
members forces all four entries of their `2-by-2` adjacency
block to agree. Each block is therefore either all four cross-edges
or no edge. The presence or absence of each within-pair edge is
unrestricted by the swap symmetry.

Positivity and the fixed unit graph are substantive premises here.
A frozen zero-capacity row need not reveal its attachments through
the phase field. Unequal capacities require the joint
capacity-state treatment in Section 13, and a weighted graph
requires the corresponding weighted equivariance test. Neither
case is covered by simply dropping a hypothesis.

Equal neighbor counts alone are insufficient. For example, pairs
`(0,1)` and `(2,3)` with internal edges `0--1,2--3`
and cross-edges `0--2,1--3` form an equitable partition:
each member has one neighbor in each relevant block. Exchanging
only `0,1` changes the cross-edge pattern. The full unordered
member-state quotient fails even though a linear mean reduction
can have matching neighbor counts. A purely diffusive or linear
quotient certificate cannot replace the complete-law condition.

### 18.3. Internal edges preserve symmetry but change the inherited rows

The preceding criterion is broader than the existing replica
runtime's no-internal-edge admission. To see the distinction
directly, let `epsilon_i in {0,1}` mark an internal pair edge,
let `d_i` count adjacent base pairs, and put
`D_i=2*d_i+epsilon_i`. Connected support gives `D_i>0`.
Use the same signed coordinates as Section 7 and define

\[
R_i=\cos\delta_i,\qquad
J_i=\sum_{j\sim i}R_j\sin(\Theta_j-\Theta_i),\qquad
C_i=\sum_{j\sim i}R_j\cos(\Theta_j-\Theta_i),\qquad
q_i=\sum_{j\sim i}(X_i-X_j).
\]

Exact averaging and differencing of the fine rows give

\[
\boxed{\begin{aligned}
\dot X_i&=\frac{2a\nu}{D_i}R_iJ_i,&
\dot\Theta_i&=\frac{2b\nu}{D_i}q_i,\\
\dot u_i&=-\frac{2a\nu}{D_i}
 \sin\delta_i\,(C_i+\epsilon_iR_i),&
\dot\delta_i&=\frac{2b\nu}{D_i}(d_i+\epsilon_i)u_i.
\end{aligned}}
\]

Indeed the optional internal edge contributes
`-epsilon_i*sin(2*delta_i)` to the plus-member sine sum
and `2*epsilon_i*u_i` to its form-gradient sum, with the
opposite contributions for the minus member. The degree used to
normalize every fine row changes at the same time.

For `epsilon_i=0` these expressions recover Section 7.
For `epsilon_i=1` the same finite swap quotient still exists,
but its internal restoring term and normalization differ. An
internal edge also adds
`2*u_i^2+2*beta*sin(delta_i)^2` to the fine storage.
It is therefore incorrect either to reject its mathematical
swap symmetry or to apply the old no-internal-edge formulas
unchanged. The current strict replica consumer keeps its original
admission; the larger symmetry test does not install these broader
rows as a runtime law.

### 18.4. Frozen one-edge control: identical retained state, different future

Return to the ten-node preparation of Section 16, with
`d=u=1/8`, `beta=w=nu=1` and structural pairs
`(0,1),(2,3),(4,5),(6,7),(8,9)`. Remove exactly edge
`(0,8)` in the **detached preparation**. Compare two states
on this same nineteen-edge graph: the original state and the
one obtained by exchanging both `x` and `theta` of
nodes `0` and `1`. No support-removal event or change in
its energy balance is being executed.

The full degree of node `8` is now three, with neighbors
`{1,6,7}`, whereas node `9` has degree four and neighbors
`{0,1,6,7}`. The original forms imply

\[
(L_fx)_8=u,\qquad (L_fx)_9=0,\qquad
\dot\theta_8=\frac{u}{3\pi},\qquad
\dot\theta_9=0.
\]

The retained phase mean of pair `(8,9)` consequently has rate

\[
\dot\Theta_{(8,9)}=\frac{u}{6\pi}
=\frac1{48\pi}.
\]

After the state-only swap, `x_1=u` instead of `-u`;
the two neighbor sums at node `9` still cancel. Therefore

\[
(L_f\widetilde x)_8=-u,\qquad
(L_f\widetilde x)_9=0,\qquad
\dot{\widetilde\Theta}_{(8,9)}
=-\frac{u}{6\pi}=-\frac1{48\pi}.
\]

The derivative difference is exactly `-1/(24*pi)`.
These are the complete-law phase rows with the actual changed
degrees, not a perturbative estimate or an inherited base-row formula.

Nevertheless, every retained unordered pair coordinate is initially
identical in the two states. Only pair `(0,1)` has changed
its signed lift: `u_0` and `delta_0` both negate.
Its means, `R=cos(delta)`, `U=u^2` and
`Q=u*sin(delta)` are unchanged; all other pairs are untouched.
The phase-mean-rate difference in an untouched pair is therefore
an exact obstruction to any autonomous law on these retained
unordered coordinates alone for this support.

The lost information is concrete: which member state of pair
`(0,1)` is attached to node `8`. Keeping the graph fixed
does not recover that state-to-attachment correlation after the
member labels have been discarded. Equal capacities do not
remove it.

### 18.5. What a larger collective description must retain

An observed pair can remain a useful organization even when its
members are not independently interchangeable under the law.
The phase-only observer, the support symmetry and the future
closure are separate claims. A failed symmetry test neither
erases an observed grouping nor proves that it immediately disappears.

The same nineteen-edge source also passes the existing analytic pairing-window
reader with the unchanged independent error radii `2^-20` and window
`[1/128,1/64]`. Its center phase-rate numerators are
`(1/8,-1/8,1/8,-1/8,0,0,0,0,1/24,0)`; the general full-field remainder from
Section 17 retains this changed rate and does not assume pair `(8,9)` stays
synchronized. Every original pair remains the strict mutual nearest choice
throughout the admitted box/window, while strict replica support is rejected.
This supplies a finite-time instance of observed organization without the
proposed fully unordered dynamical closure. It is a reuse of the same bound,
not a support event, retuned budget or trajectory experiment.

On asymmetric support, retain the constituent states together
with their incidence information, or quotient only by actual
symmetries of that joint data. Correlations between member state
and external attachments cannot in general be reconstructed
from pair means or `R,U,Q`. A more compressed sufficient
description needs its own proof; the counterexample does not
select a universal new macro variable or force law.

This result establishes an information boundary under a supplied
network. It neither selects that network, generates a microscopic
connection nor establishes autonomous birth of a collective
constituent. Exact finite-symmetry reduction retains internal
dynamics and removes only labels whose interchange truly
preserves the complete field.

### 18.6. Shared support reader and state-only evidence

`assess_sine_pair_support_symmetry` in the existing
[scale owner](../../src/tnfr/physics/relational_sine_scale.py) consumes
one complete sine comparison, an explicit pair partition and,
optionally, a caller-selected `witness_pair`. Its domain keeps
zero form loss and common positive held capacity. It checks
external neighbor equality directly, retains internal pair edges
and every interpair edge count, and records a nonzero exact
phase-row matrix witness for each failed swap. No sampled state
or tolerance decides the all-state criterion.

The optional `SinePairSwapWitness` exchanges form and phase
on the selected pair while keeping the captured graph and
capacities unchanged. It reuses the shared form-gradient,
phase-rate-numerator and sine-rate owners, retains both full
fields, and compares pair phase-mean rates. Exact equality of
the raw member-state multisets establishes the same swap orbit
without testing equality of trigonometric enclosures. Phase
means here use the supplied continuous real lifts locally;
the report does not infer a global circular mean chart.

`SinePairSupportSymmetryAssessment` keeps this concrete
countercontrol separate from its support-wide verdict and
from the existing strict replica admission. A symmetric graph
with an internal pair edge can pass the first test while
remaining outside the no-internal-edge consumer. Conversely,
a chosen state with vanishing witness rates does not restore
all-state closure on asymmetric support.

The [support-symmetry contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-support-symmetry)
and [independent replica tests](../../tests/physics/test_relational_sine_replica.py)
cover the complete/empty criterion, internal edges, equitable
but noninterchangeable members, exact degree-sensitive
countercontrol and graph/state relabeling distinction. The
reader neither mutates support nor installs a broader
autonomous collective runtime.

<a id="sine-mixed-pair-state"></a>

## 19. Sufficient collective state on asymmetric support

### 19.1. Remove only the independently redundant member labels

Keep the fixed simple connected unit support, common held capacity
`nu>0`, conservative normalized-sine law and explicit pair partition
of Section 18. Write `S` for the pair indices whose individual
member transposition is a graph automorphism, and `O` for the
remaining pair indices. The graph determines this distinction through
the exact external-neighbor criterion. It does not depend on an
instantaneous synchronization score or an observed small rate.

For every pair retain the means `X_i,Theta_i` and the same declared
nonantipodal chart

\[
x_{i,\pm}=X_i\pm u_i,\qquad
\theta_{i,\pm}=\Theta_i\pm\delta_i,
\qquad |\delta_i|<\pi/2.
\]

`Theta_i` denotes the circular midpoint with a supplied local continuous
lift; adding the same full turn to its members changes that lift, not
the circular state. The retained internal coordinates are

\[
\boxed{
\begin{cases}
(u_i,\delta_i), & i\in O,\\
(R_i,U_i,Q_i)=(\cos\delta_i,u_i^2,u_i\sin\delta_i), & i\in S.
\end{cases}}
\]

For an ordered pair, the plus/minus entries remain associated with
their named incident edges. The sign is this supplied member ordering,
not a physical orientation selected by the model. For an unordered
pair, `0<R_i<=1`, `U_i>=0` and
`Q_i^2=U_i*(1-R_i^2)` are essential realizability conditions.
The full support, pair membership, capacity, coefficients and clock
remain part of the specification.

Let `H` be the finite group generated by the individual swaps in `S`.
Two fine circular states in this chart have the same mixed description
if and only if they lie in the same `H` orbit. Indeed each ordered
pair reconstructs its assigned member states directly. For an
unordered pair with `0<R<1`, choose

\[
\delta=\arccos R>0,\qquad
u=Q/\sqrt{1-R^2}.
\]

The other lift negates both coordinates and is exactly its permitted
swap. At `R=1`, `delta=0`, `Q=0` and `u=+sqrt(U)` or
`-sqrt(U)` give the same permitted orbit. At the synchronized tip
`(R,U,Q)=(1,0,0)` there is one fine lift. These pairwise choices
prove both reconstruction and exact separation of the product orbits.
The sign of `Q` is necessary: erasing it would in general identify
states outside a permitted swap orbit.

Every generator of `H` preserves the full field by Section 18, so
these orbits have a well-defined future while the chart is valid.
This is sufficient state for the declared law, not a reduction in the
generic number of continuous degrees of freedom. Each pair still
has its two internal degrees of freedom. Nor is `H` asserted to be
the entire graph automorphism group: a simultaneous exchange of
several otherwise asymmetric pairs may be another symmetry, but
finding or quotienting such joint permutations is outside this result.

### 19.2. Inherited rates use the actual fine attachments

For any representative reconstructed above, evaluate the original rows

\[
f_v=\frac{a\nu}{D_v}\sum_{k\sim v}\sin(\theta_k-\theta_v),
\qquad
h_v=\frac{b\nu}{D_v}\sum_{k\sim v}(x_v-x_k),
\qquad a=w/\pi,\quad b=w/(\beta\pi),
\]

where `D_v` is its actual fine degree. For every pair the means obey
`X_dot=(f_plus+f_minus)/2` and
`Theta_dot=(h_plus+h_minus)/2`. For each ordered pair retain also

\[
\dot u_i=(f_{i,+}-f_{i,-})/2,\qquad
\dot\delta_i=(h_{i,+}-h_{i,-})/2.
\]

For an unordered pair differentiation gives

\[
\dot R_i=-\sin\delta_i\,\dot\delta_i,\quad
\dot U_i=2u_i\dot u_i,\quad
\dot Q_i=\sin\delta_i\,\dot u_i+
          u_i\cos\delta_i\,\dot\delta_i.
\]

These expressions are independent of the choice of permitted fine
lift. Equivariance exchanges the two rates when the member states
are exchanged; their means and the three differentiated invariants
are unchanged. Rates of untouched ordered members also stay the
same. The complete-bipartite replica formulas are not being applied
to an incomplete cross-block.

The closure can also be made explicit without an inverse square root.
For `i in S`, let `T_i` be the common external neighbor set of
its members, `m_i=|T_i|`, and let `epsilon_i` indicate the optional
internal edge. Their common fine degree is `D_i=m_i+epsilon_i>0`.
Define from retained state

\[
J_i=\sum_{k\in T_i}\sin(\theta_k-\Theta_i),\qquad
C_i=\sum_{k\in T_i}\cos(\theta_k-\Theta_i),\qquad
q_i=\sum_{k\in T_i}(X_i-x_k),
\]

\[
A_i=\frac{a\nu}{D_i}(C_i+2\epsilon_iR_i),\qquad
c_i=\frac{b\nu}{D_i}(m_i+2\epsilon_i).
\]

There is no hidden sign choice in these sums. If a member of a
different unordered pair belongs to `T_i`, its other member belongs
as well, since that pair's own swap is a graph automorphism. Its
combined contributions to `J_i,C_i,q_i` are therefore

\[
2R_j\sin(\Theta_j-\Theta_i),\qquad
2R_j\cos(\Theta_j-\Theta_i),\qquad 2(X_i-X_j).
\]

Any ordered neighbor contributes its retained assigned form and
phase individually. The same both-or-neither property lets an
ordered receiver evaluate its neighboring unordered pairs using
`2R_j*sin(Theta_j-theta_v)` and `2*(x_v-X_j)`.
Thus all required sums are functions of the mixed state and support.

Exact averaging of the external sine terms gives `R_i*J_i`;
their half-difference is `-sin(delta_i)*C_i`. An internal edge
adds `-sin(2*delta_i)` to the plus-member sine row and
`2*u_i` to its form-gradient row. Consequently the unordered
rows are exactly

\[
\boxed{\begin{aligned}
\dot X_i&=\frac{a\nu}{D_i}R_iJ_i,&
\dot\Theta_i&=\frac{b\nu}{D_i}q_i,\\
\dot R_i&=-c_iQ_i,&
\dot U_i&=-2A_iQ_i,\\
\dot Q_i&=-A_i(1-R_i^2)+c_iU_iR_i.
\end{aligned}}
\]

This includes symmetric pairs attached to only one member of an
ordered neighboring pair, as well as internal edges. In the special
complete/empty interpair support of Section 18.3, substituting
`m_i=2*d_i` recovers its formulas. The more general rows follow
from the same microscopic field, rather than an added interaction.

### 19.3. Realizability, synchronized tips and the chart boundary

The explicit mixed rows are smooth functions of the ambient
retained variables in a local midpoint chart. They contain no
division by `U`, `Q` or `1-R^2`. Their derivative of the
unordered constraint `Q^2-U*(1-R^2)` vanishes identically,
with the same cancellation as Section 8.2. More strongly, lift
any realizable initial state, evolve the original smooth fine law,
and project it. This produces a realizable solution of the mixed
rows. Local uniqueness of their smooth ambient field implies that
every solution from that mixed initial state is this projection,
including at a singular invariant tip. Thus the inequalities and
the full realizable image are preserved for as long as the phase
chart remains valid; the algebraic constraint alone would not have
proved this.

For an unordered pair at `R=1,U=Q=0`, all internal derivatives
vanish and the tip remains invariant. This follows from its actual
swap symmetry, even when neighboring ordered members differ.
At `R=1,U>0`, the nonzero `Q_dot=c_i*U` retains the form-driven
departure from phase alignment. By contrast, an ordered pair at
`u=delta=0` need not remain synchronized: unequal attachments
can give unequal fine rows immediately. Equality of a pair's
instantaneous member states is not a substitute for graph symmetry.

At `|delta_i|=pi/2` the circular midpoint becomes ambiguous;
for an unordered pair its resultant magnitude is zero. The
chosen mixed chart then ends even though the original sine field
remains smooth. No new global chart or trapping theorem on arbitrary
asymmetric support is supplied here. In numerical admission, a
chart margin not certified strictly positive requires refusal,
not clipping, sign selection or a tolerance-based assertion of
nonantipodality.

### 19.4. The retained receiver control now discriminates the two states

On the fixed nineteen-edge preparation in Section 18.4, the
unordered indices are `S={1,2,3}`, while `O={0,4}` retains
the attachment-bound member states of `(0,1)` and `(8,9)`.
For the original source, pair `(0,1)` has

\[
(X_0,\Theta_0,u_0,\delta_0)
 =(0,1/16,1/8,-1/16).
\]

Its state-only swap on the same graph instead has `u_0=-1/8`
and `delta_0=1/16`. The mixed initial states are now distinct.
The original receiver `(8,9)` has `u_4=delta_4=0` but

\[
\dot\Theta_4=\dot\delta_4=\frac{1}{48\pi};
\]

both signs reverse for the swapped source. Its equal initial
member states therefore do not make its ordered internal
coordinate redundant. These derivatives reproduce the already
known fine-row control; they are not new reserved observations.
The new result is the sufficient state and rate construction
that retains the cause of their distinction without changing
the underlying law.

Independently swapping any subset of pairs `1,2,3` changes
neither the mixed circular state nor its rates. A simultaneous
relabeling of the graph, fine state and ordered pair identities
only relabels the description. Reversing an asymmetric pair's
declared ordering while tracking its incidence reverses `u,delta`
and their rates; exchanging its states on a fixed incidence
assignment is instead the different fine model state
examined by the control. No absolute orientation or extra force
is inferred from this bookkeeping.

### 19.5. Shared observer and scope

`assess_sine_mixed_pair_state` in the
[scale owner](../../src/tnfr/physics/relational_sine_scale.py)
combines the existing exact support-symmetry assessment with
the shared pair-lift chart and complete fine sine rates. It
records each pair's ordered or unordered mode, retained coordinates,
reconstruction stratum and pushforward rates. The captured fine
state remains available as provenance; an unordered pair's signed
representative is not an additional retained mixed coordinate.

The theorem concerns exact realized states. Trigonometric interval
enclosures in a numerical report are neither exact invariant values
nor a joint uncertainty inverse for arbitrary proposed collective
data. The reader does not reconstruct a new fine state from
independently supplied `R,U,Q` boxes, silently promote overlap
to equality, or integrate an autonomous macro-node runtime.
The existing strict replica admission remains separate, and
none of its persistence, pulse or recurrence certificates are
broadened merely by accepting this mixed description.

The [mixed-state contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-mixed-pair-state)
and [independent replica tests](../../tests/physics/test_relational_sine_replica.py)
cover the full-node rates, the fixed receiver distinction, allowed
swaps, simultaneous relabeling, chart refusal and synchronized tips.
The unchanged finite observation window from Section 18.5 does
not become an all-time grouping theorem. The construction removes
a specific information loss under supplied support; it neither
creates connections nor proves spontaneous permanent constituents
or identifies physical matter.

<a id="sine-star-moment-chart"></a>

### 19.6. A regular phase/rate chart of the existing unordered state

The [phase/motion observation](SINE_CONSTITUTIVE_INFORMATION.md#phase-motion-information)
has an exact sufficient-state specialization. Use the complete three-node
star `0--1,0--2`, unit held capacities, fixed unit support, `e=0,w=beta=1`,
no inputs or events, and the conservative sine law. All primes below use
`tau=t/pi`. This support is not the complete replica graph; its coefficients
must come from its own nodal rows.

Remove the common form and phase origins and write the two leaf states as

\[
x_\pm-x_0=X\pm u,\qquad
\theta_\pm-\theta_0=\Theta\pm\delta,\qquad |\delta|<\pi/2.
\]

With `R=cos(delta), U=u^2, Q=u*sin(delta)`, the simultaneous leaf swap
removes only a redundant label. As in Sections 8 and 19.1, realizability
requires `0<R<=1`, `U>=0` and `Q^2=U*(1-R^2)`. The six fine coordinates
lose two common origins and a discrete label, leaving four continuous
degrees of freedom. Neither `U` nor `Q` is an additional primitive.

The root rows are `x_0'=R*sin(Theta)` and `theta_0'=-X`; the leaf rows give
`u'=-cos(Theta)*sin(delta)` and relative phase rates `2X+/-u`.
Their exact pushforward is therefore

\[
\boxed{
X'=-2R\sin\Theta,\quad \Theta'=2X,\quad R'=-Q,\quad
U'=-2Q\cos\Theta,\quad Q'=UR-(1-R^2)\cos\Theta.
}
\]

These are Section 19.2's symmetric pair with one common external neighbor,
together with that root's actual motion. The constraint is conserved by
direct differentiation. The full fine storage is

\[
H=X^2+U+2-2R\cos\Theta,\qquad H'=0.
\]

The common degree-weighted form mean is conserved. Given it and an initial
common phase origin, the full representative is recoverable: if the mean
is `m`, then `x_0=m-X/2`, `x_+/-=m+X/2+/-u`, while the root phase obeys
`theta_0'=-X`. The relative description has not erased a needed external
reference for comparison with another system.

**Observable inverse.** Let `z_+/-=exp(i*(Theta+/-delta))` and retain the
raw incident sums, without the pair normalization used in Section 8,

\[
Z=z_++z_-=2Re^{i\Theta},\qquad
M=(2X+u)z_++(2X-u)z_-=e^{i\Theta}(4XR+2iQ).
\]

On the strict regular domain `0<|Z|<2`, these four real observations
reconstruct the existing quotient:

\[
R=|Z|/2,\quad \Theta=\arg Z,\quad
X=\tfrac12\operatorname{Re}(M/Z),\quad
Q=R\operatorname{Im}(M/Z),\quad U=\frac{Q^2}{1-R^2}.
\]

Choosing either sign of `delta` reconstructs its matching `u`; the two lifts
differ by precisely the permitted leaf swap. Conversely, every complex pair
`(Z,M)` in this domain has a real fine lift. Rational observable coordinates
can require irrational unit phasors; exact observation admission does not
restrict the inverse to rational fine coordinates.

This is a faithful local representation of four degrees of freedom, not
their dimensional reduction. The inverse of the internal motion also reuses
[Section 8.7](#sine-replica-internal-observability): in the moving midpoint
frame the normalized pair moments are `Z_pair=R`, `M_pair=i*c*Q`, so
`R_dot=-c*Q`. Its first/second derivative reconstructions retain the known
held `c`, neighboring state and complete law. Raw incident sums, normalized
pair phasors and node-relative neighbor resultants share the chain rule but
are different observations.

**An exact rational field, without inverse angles.** Put `n=|Z|^2`,
`a+i*b=M/Z`, `U=b^2*n/(4-n)`, `P=Z/conj(Z)` and `Z2=Z^2-2P`.
Here `P=z_+*z_-` and `Z2=z_+^2+z_-^2`. Differentiating the full phase row
gives each relative acceleration `r_j'=-sin(delta_j)-(3/2)*Im(Z)`.
The identities
`sum(r_j^2*z_j)=(U-a^2)*Z+2*a*M` and
`sum(sin(delta_j)*z_j)=(Z2-2)/(2i)` yield

\[
\boxed{
Z'=iM,\qquad
M'=\frac{i}{2}(Z2-2)-\frac32\operatorname{Im}(Z)Z
       +i[(U-a^2)Z+2aM].
}
\]

Storage in this chart is `H=a^2/4+U+2-Re(Z)`. The field is smooth on its
domain. Fine-law uniqueness and the inverse establish exact local closure
until the trajectory leaves the chart, not an all-time domain guarantee.

**The two singular boundaries lose different information.**

- At coincident leaf phases `R=1`, `Q=0`, the observation loses `U`.
  With all phases zero, forms `(0,0,0)` and `(0,1,-1)` both give `Z=2,M=0`,
  but `M'=0` and `M'=2i`. The retained pair state remains sufficient;
  Section 8.7's `U=-R''` at this boundary explains the missing information.
- At antipodal phases `z,-z`, one has `Z=0,M=2u*z`. If `M!=0`, it fixes
  the unordered phase pair and its associated form difference. The missing
  scalar is the relative mean form `X`, not simply the pair orientation:
  `M'=-2*Im(z)*z+4i*X*M`. For phases `(0,0,pi)`, forms `(-1/2,1,0)` and
  `(1/2,0,-1)` have the same weighted form mean zero, `H=13/4,Z=0,M=1`,
  but `M'=4i` and `M'=-4i`. Adding scalar storage does not resolve the sign.
  If also `M=0`, orientation is lost as well: zero form with leaf phasors
  `(1,-1)` and `(i,-i)` gives the same observation but `M'=0` and `-2i`.

These are observation failures, not singularities of the fine law or
evidence for a new forcing channel. Near either boundary the inverse is
ill-conditioned; tolerances, clipping or a noisy derivative estimate cannot
replace joint state/observation admission. Section 28 retains the analogous
distinction between a zero collective resultant and continuing fine motion.

The shared [phase-response owner](../../src/tnfr/physics/phase_response.py)
implements `derive_sine_star_moment_closure` using exact represented-real
moment pairs and rejects the singular domain. Its
[independent controls](../../tests/physics/test_sine_star_moment_closure.py)
evaluate the full nodal rows, both singular witnesses, conserved storage,
leaf exchange, origin symmetry and representation limits. The
[contract](../../docs/contracts/relational/OBSERVATION_AND_INFORMATION.md#sine-star-moment-chart)
specifies the clock, SDK projection and detached evaluation scope.

Finally, the inverse uses geometry and the phase row, not the sine form
current. Other supplied common smooth odd periodic edge currents, applied
uniformly with the same phase row, preserve the removed symmetries and also
have a pushforward on this chart, with their own vector field and storage.
Closure here therefore does not select sine, a universal pressure
law, persistent identity, autonomous grouping or a physical constituent.

<a id="sine-global-pair-state"></a>

## 29. Global unordered pair state through phase cancellation

Keep [Section 28's current and support](SINE_PAIR_INTERACTION.md#sine-zero-resultant-restoration):
fixed unit doubled C5, consecutive pairs `{2a,2a+1}`
for `a=0,...,4`, held unit capacities, conservative normalized sine,
`e=0,w=beta=1`, structural time `t` and no inputs or events. All four fine
cross edges join each adjacent pair, and each fine degree is four. The
following is a global representation of this same law, including zero
resultants. It does not select a law, create a grouping or change support.

### 29.1. Derived coordinates and exact realizability

For each pair let `z_+/-=exp(i*theta_+/-)` and define

\[
X=\frac{x_++x_-}{2},\quad u=\frac{x_+-x_-}{2},\quad
Z=\frac{z_++z_-}{2},\quad D=\frac{z_+-z_-}{2},
\]
\[
\boxed{P=z_+z_-=Z^2-D^2,\qquad U=u^2,\qquad W=uD.}
\]

The retained state is `(X,Z,P,U,W)`. Here `X` is real form, `U` has squared
form units, `W` has form units, and `Z,P` are dimensionless circular-phase
observations. None is a complex EPI or an independent constitutive parameter.
Exchanging the two members sends `(u,D)` to `(-u,-D)` and preserves all five
quantities. The complex product `P` does not require a branch or a midpoint.

Writing `n=|Z|^2`, the necessary and sufficient realized domain is

\[
\boxed{
X\in\mathbb R,\quad |P|^2=1,\quad Z=P\overline Z,\quad
n\le1,\quad U\ge0,\quad W^2=U(Z^2-P).
}
\]

Necessity follows from unit fine phasors. For sufficiency note first that
`Z^2=n*P`. If `U>0`, choose `u=sqrt(U)` and `D=W/u`; if `U=0`, the
constraint forces `W=0`, and choose either square root of `D^2=Z^2-P`.
In both cases

\[
D^2=-(1-n)P,\qquad |D|^2=1-n,\qquad
\operatorname{Re}(Z\overline D)=0.
\]

The last equality follows because `(Z*conj(D))^2=-n*(1-n)`. Hence
`z_+/-=Z+/-D` are unit phasors and `x_+/-=X+/-u` are a fine lift. The
two choices differ only by the simultaneous member swap. This also proves
that equal invariant states correspond exactly to that swap orbit, including
`U=0`, coincident phases and antipodal phases. In particular
`|W|^2=U*(1-n)` is a derived consistency identity, not an extra premise.

At `n=1`, `D=W=0` but `U` still distinguishes unequal forms. At `Z=0`,
`P` retains the antipodal orientation even when `U=W=0`. The invariant
description has eight real components with constraints and the same four
continuous degrees of freedom per pair as the fine state. At swap-fixed
states it is not an ordinary smooth coordinate chart. Common phase and
form origins remain retained; only redundant member labels are removed.

### 29.2. The inherited polynomial field

Let neighboring pair indices be read modulo five and define derived quantities

\[
B_a=\frac{Z_{a-1}+Z_{a+1}}2,\qquad
h_a=X_a-\frac{X_{a-1}+X_{a+1}}2.
\]

The actual fine rows give
`pi*x_dot_+/-=Im(conj(z_+/-)*B_a)` and
`pi*theta_dot_+/-=h_a+/-u_a`.
Differentiating the retained coordinates yields

\[
\boxed{\begin{aligned}
\pi\dot X_a&=\operatorname{Im}(\overline Z_a B_a),\\
\pi\dot Z_a&=i(h_a Z_a+W_a),\\
\pi\dot P_a&=2i h_a P_a,\\
\pi\dot U_a&=2\operatorname{Im}(\overline W_a B_a),\\
\pi\dot W_a&=i\left[h_aW_a+U_aZ_a+
 \frac{(Z_a^2-P_a)\overline B_a-(1-|Z_a|^2)B_a}{2}\right].
\end{aligned}}
\]

For example, `W_dot=u_dot*D+u*D_dot`, and the form contribution satisfies
`D*Im(conj(D)*B)=[(1-n)*B-D^2*conj(B)]/(2i)`. This is the last row's
source. No division by `Z`, `U`, resultant magnitude or a phase angle occurs.
The rows are real polynomial functions of the retained components. Their
coefficients come from the same fine degree, capacity and law, not from the
three-node star or an imposed oscillator.

The previously observed `Y=<x*z>` is exactly `Y=X*Z+W`. Thus its contribution
to `Z_dot` is retained, but its own sufficient continuation also needs `P,U`.
On the regular chart, `Z=R*exp(i*Theta)`, `P=exp(2i*Theta)` and
`W=i*Q*exp(i*Theta)`. Substitution recovers all Section 8 rows with unit
coefficients, including `pi*R_dot=-Q`. The old chart is a useful local
description of this state, not a different dynamics.

### 29.3. Storage, current and global continuation

The fine graph storage, expressed without a midpoint, is

\[
F=2\sum_{\{a,b\}\in C_5}(X_a-X_b)^2+4\sum_a U_a,\qquad
V=4\sum_{\{a,b\}\in C_5}
       [1-\operatorname{Re}(\overline Z_a Z_b)],\qquad H=F+V.
\]

This is the expansion of `sum_fine_edges (x_i-x_j)^2/2` and
`sum_fine_edges [1-cos(theta_i-theta_j)]`. No storage term has been added.
Both components are nonnegative on the realized set, `H_dot=0`, and
`sum_a X_a` is conserved. The individual component rates can be nonzero.

For the directed neighbor contribution `b->a`, retain the existing mean-form
normalization `I_ba=Im(conj(Z_a)*Z_b)/(2*pi)`. If `G_a=pi*Z_dot_a`, then

\[
\pi^2\dot I_{b\to a}
 =\tfrac12\operatorname{Im}(\overline G_a Z_b+\overline Z_a G_b).
\]

This formula is valid at cancellation. The degree/capacity weighted cut
contribution is still `8*I_ba`; the phasor representation does not change the
regional ledger or interpret form current as energy current.

To establish continuation, lift any admitted invariant state to the fine
state just constructed. The smooth sine law is equivariant under each
permitted pair swap, so either lift projects to the same invariant solution.
Local uniqueness of the ambient polynomial field makes that projection the
unique solution from the realized datum, even at singular strata of the
invariant set. Realizability is preserved by this argument, not inferred
from a small constraint residual. On connected C5 the conserved `H` and
`sum X` bound all forms and `U`; `|Z|<=1`, `|P|=1` and
`|W|^2=U*(1-|Z|^2)` bound the remaining coordinates. Consequently the
realized solution continues for every finite positive or negative time.
This is neither global stability of a chosen pattern nor a claim that
arbitrary unconstrained ambient initial data describe fine nodes.

### 29.4. Why cancellation needs the retained phase product

On the same doubled C5, put all forms zero and all surrounding pair phasors
equal to one. Compare one selected pair `(z_+,z_-)=(1,-1)` with `(i,-i)`.
Both have `X=Z=U=W=Y=0` there, the same surrounding retained state, full
storage `H=8`, all block currents zero and `Z_dot=0`. Only `P` differs:
it is respectively `-1` and `+1`. The last row gives

\[
\pi\dot W=0\quad\hbox{versus}\quad -i,\qquad
\pi^2\ddot Z=0\quad\hbox{versus}\quad 1.
\]

Thus `(X,Z,Y)` or `(X,Z,U,W)` alone is not globally sufficient. This
counterexample uses the complete original law, not an added force or hidden
event. Likewise, coincident phases with different `U` have the same `X,Z,P,W`
and the same initial `Z_dot`, but different subsequent motion, already in
`Z_ddot`; changing the sign of `W` at fixed
nonzero `Z,U,P` changes the form/phase association and `Z_dot`.

The raw second circular moment is `z_+^2+z_-^2=4Z^2-2P`. Its information
can therefore be required by a global collective **state** even though the
sine **current** consumes first phasor moments. This does not introduce
a second-harmonic pressure channel or select a fundamental interaction law.

### 29.5. Implementation and numerical scope

The shared [scale owner](../../src/tnfr/physics/relational_sine_scale.py) provides
`derive_sine_global_pair_state` from ten exact unit phasors and signed forms,
and `evaluate_sine_global_pair_state` from the five invariant tuples after
full realizability admission. Both use the same inherited field. Outputs
retain exact numerators of the original-clock rates, storage components and
directed current observations; SDK export preserves their rational values.
The [contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-global-pair-state)
specifies ordering, admission and the distinction from graph capture.

Eliminating a retained pair instead reuses the existing
[nonlinear mediator identity](SINE_ENVIRONMENTAL_MEMORY.md#causal-sine-environmental-pressure)
twice. Its members have no mutual edge and each has the same four visible
neighbors. Conditional on those neighbors' histories, use `k=4,mu=w=beta=1,e=0`
for each hidden member, retaining both initial states, original visible degrees
and both form/phase contributions. The zero-loss factor is `F_0(t)=t`;
the identity introduces neither decaying memory nor a time-scale separation.
This reuse does not justify replacing the hidden preparation by a stationary
state or transferring a different support's approximation bounds.

These evaluators provide neither a time integrator nor a numerical trajectory
certificate. A finite Euler update of constrained invariants need not remain
realizable; no projection or tolerance repair is supplied. Existing regular
graph/scale adapters retain their own input and phase domains. General support,
unequal capacities, native Arg pressure, occurrence of grouping and physical
identification require separate results. The
[independent controls](../../tests/physics/test_sine_global_pair_state.py)
differentiate all fine rows, test the boundary strata, full storage, symmetry,
realizability and exact export without replaying a frozen experiment.

<a id="sine-pair-cancellation-observability"></a>
### 29.6. Observable internal state at cancellation

Retain exactly Section 29's complete law, support and unit coefficients.
Observe all pair interfaces `(X,Z)` with their ordering and clock; hold the
surrounding retained preparation fixed when comparing two selected-pair states.
Primes in this subsection use **`tau=t/pi`**, so `Z'=pi*Z_dot`.
The derivatives below are exact causal left derivatives from an already
observed interval, or separately supplied exact evidence. Their existence as
mathematical quantities does not make them available from noisy finite samples.

At an instant with selected `Z=0`, put `v=Z'` and `a=Z''`. Realizability and
the inherited rows give

\[
\boxed{W=-iv,\qquad U=|v|^2,\qquad
 a=2ihv+\frac{B+P\overline B}{2}.}
\]

Indeed `Z'=iW`, `|W|^2=U` and `W^2=-UP` at cancellation; differentiating
`Z'=i(hZ+W)` gives the last identity. Consequently:

- If `v!=0`, the first derivative recovers `P=v^2/|v|^2`, as well as `U,W`.
  The supplied second derivative must satisfy the last identity above.
- If `v=0`, then `U=W=0`. When `B!=0`, the second derivative recovers
  `P=(2a-B)/conj(B)`. This remains informative when **`a=0`**: the unique
  compatible orientation is `P=-B/conj(B)`.
- If `v=B=0`, the second derivative must be zero. Every unit `P` is compatible
  with these local observations. This is a two-derivative ambiguity, not yet
  a statement about a whole history.

Every recovered `P` must have unit norm. These are conditional inverse results
for the supplied law, not an independent validation of that law. In particular,
the reconstruction does not infer pressure by fitting the evaluated response.

**Delayed neighborhood information.** Suppose `v=0` and the first nonzero
neighborhood derivative is `B^(m)`, with all lower derivatives zero. Repeated
differentiation of the inherited rows gives

\[
Z^{(k)}=0\ (0\le k\le m+1),\qquad
Z^{(m+2)}=\frac{B^{(m)}+P\overline{B^{(m)}}}{2},\qquad
\boxed{P=\frac{2Z^{(m+2)}-B^{(m)}}{\overline{B^{(m)}}}.}
\]

To see the derivative order directly, choose a unit `d` with `d^2=-P_0` and
put `H(tau)=integral h`, with integral starting at the observation instant.
The rotating-frame fine lift can be written

\[
 e^{-iH}Z=id\sin\eta,\quad u=\eta',\quad
 b=e^{-iH}B,\quad
 \eta''=\operatorname{Im}(\overline d b)\cos\eta,
 \qquad \eta_0=\eta'_0=0.
\]

Multiplication by `exp(-iH)` preserves the first nonzero derivative order
of `B`. Thus `eta` and `Z` vanish through order `m+1`; differentiating at
order `m+2` gives the displayed identity. Also `X-X_0=O((tau-tau_0)^(2m+3))`
from `X'=Im(conj(Z)B)`, so mean form reveals no earlier orientation information.
In the original clock the inverse numerator is
`2*pi^2*d_t^(m+2)Z-d_t^m B`, divided by `conj(d_t^m B)`.

The finite autonomous field is real analytic on every realized trajectory.
Either this finite order exists or `B` vanishes identically on the connected
trajectory. A finite zero jet does not establish the latter without an
independently proved sufficient-order or invariance argument. No uniform
sufficient derivative order is established here.
Analyticity also prevents exact silence on an open time interval followed by
spontaneous reactivation on the same unchanged trajectory. Delayed revelation
above concerns a finite zero jet at one instant. Comparing differently
prepared autonomous neighborhoods is a separate experiment, not a later
intervention silently added to the law.

**Persistent silence and indistinguishability are different.** The same lift
and uniqueness of the fine equations imply, on an interval containing the
initial instant,

\[
\boxed{Z\equiv0\quad\Longleftrightarrow\quad
b+P_0\overline b\equiv0\quad\Longleftrightarrow\quad b(\tau)\in\mathbb R d.}
\]

This tests the actual evolving neighborhood, not an externally held source.
A nonzero `b` confined to a fixed real line permits exactly one silent
orientation. A `b` not confined to such a line permits none. When `b=0`
identically, all antipodal orientations remain silent. The primitive phases
may still rotate: `P=exp(2iH)*P_0`. Silence is not full equilibrium.

If identical interface histories become nonzero in `Z` anywhere, realizability
gives `P=Z/conj(Z)` there; their common `h` transports this equality back to
the initial instant. Thus distinct indistinguishable initial orientations
must have `Z=0` throughout. Two such orientations require
`B=0` identically: subtracting `B+P_1*conj(B)=0` and
`B+P_2*conj(B)=0` proves it, since their phase products retain the same
nonzero difference up to `exp(2iH)`. Conversely, if one such trajectory has
`B=0`, changing only its initial `P` preserves the selected `(X_0,0)` and
the entire surrounding evolution. The surrounding rows consume that pair
only through `X,Z`; uniqueness supplies this converse. Hence permanently
unobservable orientation is an exact environmental restriction, not a generic
consequence of a zero snapshot resultant.

**Whole-network witnesses.** The following preparations use the actual
doubled C5, with pair 0 selected and omitted within-pair differences zero:

- For an invisible continuum, set every form to zero, pair 0 to `(d,-d)`,
  pairs 1 and 2 to `(1,1)`, and pairs 3 and 4 to `(-1,-1)`.
  Every unit `d` modulo sign gives a stationary solution with identical
  interfaces and identical surrounding retained state. Nonzero surrounding
  resultants do not preclude their cancellation at the selected pair.
- For an invisible moving family, make every pair antipodal and assign equal
  within-pair forms with arbitrary means `X_a`. Then `Z=U=W=B=0` throughout,
  the means stay constant, and `P_a=exp(2ih_a*tau)*P_a(0)` can rotate.
- For delayed revelation, take pair 0 as `(d,-d)` with both forms zero
  and `Z_1=Z_2=Z_3=1,Z_4=-1`. With `X_1=2`
  and all other means zero, `B_0=0,B'_0=i`, and
  `Z'''_0=i(1-P_0)/2`. Alternatively, all means zero and `U_1=1` give
  `B_0=B'_0=0,B''_0=-1/2` and `Z''''_0=-(1+P_0)/4`.
- Initial alignment alone is insufficient: choose pair 0 as `(1,-1)`, all
  surrounding phasors one, `X_1=2`, and other means zero. Here
  `B_0=1`, `Z'_0=Z''_0=0`, but `Z'''_0=2i`.

These are exact preparations and derivative consequences, not formation,
attraction or robustness theorems. They establish neither material screening
nor a quantum state. The shared `observe_sine_pair_cancellation` evaluator
implements the two-derivative inverse and rejects inconsistent evidence.
Its unavailable branch preserves the higher-order/persistent distinction;
it does not certify either from a finite zero record. See its
[contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pair-cancellation-observability),
[usage](../../docs/guides/relational/SINE_PAIR_DYNAMICS.md#sine-pair-cancellation-observability)
and [independent fine-row controls](../../tests/physics/test_sine_pair_cancellation_observation.py).

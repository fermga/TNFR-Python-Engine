# Retained reservoir information and local organized equilibria

<a id="organized-parent-reservoir-memory"></a>

## Declared comparison and mathematical question

The [fixed reservoir experiment](ORGANIZED_PARENT_RESERVOIR_EXCHANGE.md)
failed its prescribed disruption readout. This separate analytic question
keeps that result and its protocol unchanged. It asks whether retained
auxiliary state can change stable geometry, and what reservoir leakage changes.

| Item | Declared scope |
| --- | --- |
| Fine support | The same 36-node matching product `C9 x K2 x K2`, with 72 unit edges, node `18*g+9*a+j`, and held unit network capacities |
| Complete law | `x_tau=(-Lx+gamma*S(theta))/4+b*r`, `theta_tau=gamma*Lx/4`, `r_tau=-b^T*Lx-lambda*r`, with `gamma=1/(1023*pi)` and unchanged `b[g,a,j]=gamma*(2*a-1)/12` |
| Compared laws | Only `lambda=0` and the original `lambda=gamma^2/2`; no coupling scan or change to the sine law |
| Auxiliary state and units | Signed real form `r`, held unit capacity in `tau`, independently supplied pressure `-b^T*Lx-lambda*r`, storage `r^2/2` in the existing normalized units; no auxiliary phase is consumed |
| Clock | `tau=(1023/1024)*t`, `s=gamma^2*tau/2`; a local stability problem has no new finite readout or source dwell |
| Common modes | Fix `sum(x)=0` and a continuous phase lift with `sum(theta)=0`; arbitrary common modes are outside this declared representative |
| Phase chart | Write `theta[g,a,j]=psi[j]+delta[g,a,j]`, `psi[j]=2*pi*(j-4)/9`, with real mean-zero `delta`. On each oriented child edge `j -> (j+1) mod 9`, the adjusted gap is `2*pi/9+delta[next]-delta[j]`, including its fixed closing-edge turn. Require these gaps in `(0,pi/2)` and all matching contact gaps in `(-pi/2,pi/2)` |
| Retained-coordinate candidate | Define `c=4*b/gamma` and the lifted quantity `J=r+c^T*theta`. Determine its exact evolution before eliminating any state. For the zero-leakage classification restrict to `abs(J)<=3` |
| Identity observations | Each marked child has winding one in this chart. A parent is registered when all its intraparent same-slot circular gaps have magnitude strictly below `1/4`; equality is a boundary, not registration |
| Perturbations | Admit all fine form and phase directions compatible with the fixed means and chart, and independent auxiliary perturbations. State explicitly when stability is relative to a fixed `J` leaf rather than to the complete family |
| Required evidence | Full-state equilibrium classification, a valid local trapping neighborhood and attraction argument, or a scoped obstruction. A scalar reference root, positive Hessian or dissipative balance alone does not establish entry from an arbitrary preparation |
| Discriminating question | Can the same passive port support different locally stable registration geometries through retained initial auxiliary information, and which equilibrium/attraction conclusions survive the original positive leakage? |

Use the original joint storage `H_joint=x^T*L*x/2+sum_edges(1-cos(gap))+r^2/2`.
Derive every elimination, balance, domain and stability claim for the actual
complete law. Carry the phase lift used by `J`; a circular representative
change cannot silently change the supplied auxiliary state or its preparation.

This is a local conditional classification, not a new evaluation of the
earlier acquired source. Any trapping condition is an explicit restriction
on initial state; it does not prove that the original preparation enters it.
No finite convergence time, trajectory campaign, response retuning, source
refinement or replay of frozen producers is authorized by this question.

The auxiliary coordinate, its origin, reciprocal coupling and preparation
remain supplied constitutive premises. The continuous path installs no
operator, support event or occurrence selector. This comparison neither
identifies a physical bath nor selects a unique TNFR law or material property.

The declaration above was committed as `53cc7b68` before this proof. The
zero-leakage law has a family of locally attracting equilibria indexed by
the retained lifted quantity. Attraction to a particular member is relative
to its invariant leaf. With the original positive leakage, only the
zero-reservoir equilibrium survives in the declared chart, and it has a
full-state local trapping neighborhood. Neither result admits the earlier
acquired preparation into that neighborhood.

## 1. Complete law, lift and retained quantity

Let \(\sigma_a[g,a,j]=2a-1\), \(\alpha=2\pi/9\), and
\(\psi[g,a,j]=\alpha(j-4)\). Write

\[
 U(\theta)=\sum_{\{i,k\}}[1-\cos(\theta_k-\theta_i)],\qquad
 S_i(\theta)=\sum_{k\sim i}\sin(\theta_k-\theta_i),\qquad
 c=\frac{\sigma_a}{3}=\frac{4b}{\gamma}.
 \tag{1}
\]

Every sum has the fixed 72 unit edges. Thus \(\nabla U=-S\),
\(\mathbf1^Tc=0\), \(\|c\|_2=2\), and \(c^T\psi=0\).
The declared chart is an open convex subset of the mean-zero lifted phase
space: all its gap constraints are strict affine inequalities in
\(\theta-\psi\). On a closing child edge, the fixed added turn is \(2\pi\).
The cosine and its derivatives are unaffected by that turn, while the
chart and winding statement depend on retaining it.

The two fixed means are conserved by the complete law. Its vector field is
globally Lipschitz on the finite-dimensional real lifted state, so solutions
exist uniquely for every finite time. The joint balance and retained
quantity follow directly, without restricting to child-constant states:

\[
 \begin{split}
 (H_{\rm joint})_\tau
   &=-\frac14\|Lx\|_2^2-\lambda r^2,\\
 J&=r+c^T\theta,\\
 J_\tau
   &=-b^TLx-\lambda r+c^T\frac{\gamma Lx}{4}
     =-\lambda r .
 \end{split}\tag{2}
\]

For the slow clock, put \(\ell=2\lambda/\gamma^2\), which is either zero
or one. Every row becomes

\[
 x_s=\frac{-Lx+\gamma S(\theta)}{2\gamma^2}
                   +\frac{2b}{\gamma^2}r,\quad
 \theta_s=\frac{Lx}{2\gamma},\quad
 r_s=-\frac{2b^TLx}{\gamma^2}-\ell r,\quad
 J_s=-\ell r .
 \tag{3}
\]

The positive clock changes do not change equilibrium or local attraction.
They do not supply a laboratory time scale.

The conserved quantity for zero leakage is lift dependent. Replacing
\(\theta\) by \(\theta+2\pi k\) while retaining the same physical auxiliary
state changes its numerical label by \(2\pi c^Tk\). Such a change cannot
be used to compare preparations in different charts without transporting
the chart and its retained data. The following classification uses exactly
the lift and means in the declaration.

## 2. Full equilibrium classification in the declared chart

First take \(\lambda=0\) and a fixed \(J\). Eliminating the auxiliary
coordinate on its invariant leaf gives

\[
 r=J-c^T\theta,\qquad
 U_J(\theta)=U(\theta)+\frac12(J-c^T\theta)^2,\qquad
 \nabla U_J=-S-c(J-c^T\theta).
 \tag{4}
\]

The complete restricted rows and their storage are

\[
 x_\tau=-\frac{Lx}{4}-\frac{\gamma}{4}\nabla U_J(\theta),
 \qquad
 \theta_\tau=\frac{\gamma Lx}{4},\qquad
 H_J=\frac12x^TLx+U_J(\theta),\qquad
 (H_J)_\tau=-\frac14\|Lx\|_2^2.
 \tag{5}
\]

In the acute chart, the Hessian of \(U\) is the graph Laplacian whose edge
weights are the positive cosines of the adjusted gaps. The graph is
connected. Consequently \(U\), and also
\(U_J\) with Hessian \(\nabla^2U+cc^T\), are strictly convex on the
mean-zero chart. This is a statement about every fine phase direction,
including directions outside the four-child invariant family.

Any equilibrium has \(Lx=0\), hence \(x=0\) under the fixed form mean.
Its remaining equation is \(\nabla U_J=0\). A candidate can be constructed
without assuming that the equilibrium must inherit a particular symmetry:

\[
 \begin{split}
 6B_*+3\sin B_*&=J,\\
 x_*&=0,\qquad
 \theta_*=\psi+\frac{\sigma_a B_*}{2},\qquad
 r_*=3\sin B_* .
 \end{split}\tag{6}
\]

The scalar left-hand side is strictly increasing; its derivative is
\(6+3\cos B\ge3\). For \(|J|\le3\), its unique root satisfies
\(|B_*|<1/2\): its value at \(1/2\) is strictly greater than three.
For nonnegative \(J\), useful enclosing inequalities are
\(J/9\le B_*\le J/6\); negative roots follow by odd symmetry.
At this candidate all child gaps equal \(\alpha\), all interparent gaps
vanish, and all intraparent gaps equal \(B_*\). It lies inside the declared
chart. The internal sine contributions cancel, and
\(S(\theta_*)=-\sigma_a\sin B_*=-c r_*\), so it is an exact equilibrium.
Strict convexity then proves that it is the only equilibrium in that chart
on this \(J\) leaf. This uniqueness argument does not discard transverse
coordinates or infer equilibrium from a scalar root alone.

Both parents are registered at equilibrium precisely when

\[
 |J|<J_{\rm reg},\qquad
 J_{\rm reg}=\frac32+3\sin(1/4).
 \tag{7}
\]

At equality their gaps equal the strict identity boundary. For
\(J_{\rm reg}<|J|\le3\), both equilibrated parents are unregistered while
all four children retain winding one. In particular \(J=0\) gives registered
parents, whereas \(J=3\) gives unregistered parents, since
\(J_{\rm reg}<9/4<3\). These are different preparations of the same
zero-leakage law, not different fitted couplings.

For \(\lambda=\gamma^2/2>0\), equilibrium again requires \(Lx=0\).
The reservoir row then requires \(r=0\), and the form row requires
\(S(\theta)=0\). Strict convexity of \(U\) in the declared chart, together
with the explicit critical point \(\psi\), gives the unique equilibrium

\[
 (x_*,\theta_*,r_*)=(0,\psi,0).
 \tag{8}
\]

The variable \(J\) is not a constraint on this positive-leakage equilibrium:
equation (2) changes it during evolution. Removing \(r\) as though \(J\)
were conserved would change this complete model.

## 3. Uniform local trapping bounds

The fixed product graph has
\(\lambda_2(L)>1/3\) and \(\|L\|_2\le8\).
Indeed its Laplacian spectrum is the sum of the C9 spectrum and two
copies of \(\{0,2\}\); the C9 gap is \(2-2\cos\alpha>1/3\).
The [acquisition proof](ORGANIZED_PARENT_ACQUISITION.md#2-source-admission-and-full-fine-state-error)
establishes these same fixed-support bounds. No graph coefficient from a
different support is used here.

Set \(\rho=1/8\) and write \(\eta=\theta-\theta_*\), with zero mean.
In the closed phase ball \(\|\eta\|_2\le\rho\), every edge-gap perturbation
has magnitude at most \(2\rho=1/4\). The adjusted internal gaps lie in
\((\alpha-1/4,\alpha+1/4)\subset(0,\pi/3)\); the absolute matching gaps
are at most \(|B_*|+1/4<3/4<\pi/3\).
The positive-leakage target is the case \(B_*=0\).
Thus the entire closed ball stays strictly inside the declared chart,
retains all four child windings, and has every edge cosine greater than
\(1/2\). In particular,

\[
 \nabla^2U\succeq\frac12L,\qquad
 \nabla^2U_J\succeq\frac12L,\qquad
 \eta^TL\eta\ge\frac13\|\eta\|_2^2.
 \tag{9}
\]

For zero leakage, use the storage excess on the actual fixed \(J\) leaf,

\[
 \begin{split}
 E_J(x,\theta)
 &=\frac12x^TLx+U_J(\theta)-U_J(\theta_*)\\
 &=\frac12x^TLx+
   U(\theta)-U(\theta_*)-(c r_*)^T\eta
                    +\frac12(c^T\eta)^2 .
 \end{split}\tag{10}
\]

The second equality uses both
\(\nabla U(\theta_*)=c r_*\) and \(r-r_*=-c^T\eta\).
It is the affine cancellation that makes this excess positive near the
target. The separate differences
\(U(\theta)-U(\theta_*)\) and \((r^2-r_*^2)/2\) need not be positive.
Nor is this a positive excess relative to an arbitrarily selected target
on a different \(J\) leaf.

For positive leakage use the full-state excess

\[
 E_+(x,\theta,r)=\frac12x^TLx+U(\theta)-U(\psi)+\frac12r^2 .
 \tag{11}
\]

Taylor's formula along the line segment in the ball, the critical-point
identities, and (9) give the lower bounds

\[
 \begin{split}
 E_J&\ge\frac16\|x\|_2^2+\frac1{12}\|\eta\|_2^2,\\
 E_+&\ge\frac16\|x\|_2^2+\frac1{12}\|\eta\|_2^2+\frac12r^2 .
 \end{split}\tag{12}
\]

Every point on the phase-ball boundary therefore has excess at least
\[
 E_{\rm boundary}=\frac{\rho^2}{12}=\frac1{768}.
 \tag{13}
\]

A sufficient initial-state condition is

\[
 \|\eta(0)\|_2<\frac18,\qquad
 0\le E(0)\le e<\frac1{768},
 \tag{14}
\]

where \(E=E_J\) on the specified zero-leakage leaf, or \(E=E_+\) for the
positive-leakage model. The phase-ball admission is required in addition
to the energy bound; a low periodic storage elsewhere is not an entry
certificate. The storage derivative is nonpositive. If a solution first
reached the phase-ball boundary, (13) would contradict \(E\le e\).
It remains in the ball for all future time.

For a simple rational sufficient admission from norm allowances, let
\(\|x(0)\|_2\le X\), \(\|\eta(0)\|_2\le\Theta<1/8\).
The global Hessian upper bounds
\(\nabla^2U\preceq L\preceq8I\) and
\(\nabla^2U_J\preceq L+cc^T\preceq12I\) imply

\[
 \begin{array}{ll}
 \lambda=0,\ \hbox{same }J:
   &4X^2+6\Theta^2<1/768,\\[2pt]
 \lambda=\gamma^2/2,\ |r(0)|\le R:
   &4X^2+4\Theta^2+R^2/2<1/768 .
 \end{array}\tag{15}
\]

These are sufficient bounds, not the exact basin. On a fixed zero-leakage
leaf the auxiliary deviation is constrained by
\(r-r_*=-c^T\eta\). An independent auxiliary perturbation must instead be
included when determining the state's actual \(J\) and target. It cannot
be dropped while invoking the first row of (15).

## 4. Full-state attraction and the remaining neutral information

For each \(e<1/768\), the portion of the closed phase ball with
\(E\le e\) is compact and positively invariant under the corresponding
complete law. Compactness follows from the form bound in (12), the bounded
phase ball, and either \(r=J-c^T\theta\) or the reservoir bound in (12).
Invariance was proved by the boundary barrier, not assumed from passivity.

In the zero-leakage case, zero storage loss requires \(Lx=0\), hence
\(x=0\). To remain in this zero-loss set, the form row in (5) also requires
\(\nabla U_J=0\). The unique equilibrium in (6) is therefore the largest
invariant subset of the zero-loss set. The finite-dimensional LaSalle
invariance argument gives

\[
 (x(\tau),\theta(\tau),r(\tau))
       \longrightarrow(0,\theta_*(J),r_*(J))
       \quad\hbox{as }\tau\longrightarrow\infty .
 \tag{16}
\]

This controls every fine transverse form and phase direction on the
invariant leaf: with the two means fixed, the full joint state has
dimension 71 and the fixed-\(J\) leaf has dimension 70.
It is not only stability in the scalar \(B\) direction or in the
four-child invariant family. The positive energy bounds also establish
Lyapunov stability relative to that leaf.

In the positive-leakage case, zero joint loss requires both \(x=0\) and
\(r=0\). Invariance additionally requires \(S(\theta)=0\), so the largest
invariant zero-loss set is the single equilibrium (8). The same argument
proves local asymptotic stability in all 71 mean-fixed joint directions,
including independent auxiliary perturbations. This conclusion does not
give global attraction from the full acute chart or entry from the
previous finite-reservoir source.

For zero leakage, a particular equilibrium cannot attract arbitrary
nearby states on different \(J\) leaves: nearby members of (6) are exact
equilibria themselves. The locally attracting object without fixing
\(J\) is the family, with each sufficiently close state subject to its
own conditions (14). Its target varies continuously, since

\[
 \frac{dB_*}{dJ}=\frac1{6+3\cos B_*}>0 .
 \tag{17}
\]

This is retained initial information, not an unforced periodic motion.
For a trapped state whose \(J\) lies strictly below the threshold in (7),
convergence implies eventual registration. Strictly above that threshold,
it implies eventual nonregistration of both parents. At the threshold,
convergence to the identity boundary alone settles neither strict
finite-time alternative. The uniform phase ball by itself preserves
children, not registration of every parent throughout the transient.

The extra Hessian term \(cc^T\) is positive semidefinite at a fixed phase
state. It does not show that every transverse stiffness increases when
comparing different equilibria: the equilibrium shift also changes the
edge cosines. Neither this rank-one term nor dissipation selects the
port law, the auxiliary coordinate or its initial value.

## 5. Evidence boundary

This proof classifies the supplied joint law on one declared lifted chart.
It derives an exact information constraint for zero leakage, distinct
registered and unregistered equilibrium geometries, and full-state local
attraction under explicit trapping premises. The positive-leakage model
has a different local equilibrium selection because it changes that
constraint. No finite deadline or convergence rate is asserted.

The [static implementation](../../src/tnfr/physics/_organized_parent_reservoir_memory.py)
reads admitted primitive states and rebuilds their retained quantity,
equilibrium, chart and local-energy evidence. The norm-allowance conditions
in (15) are additional mathematical sufficient conditions, not a required
implementation API. The
[owner tests](../../tests/physics/test_organized_parent_reservoir_memory.py)
check the static identities and admission boundaries without trajectories
or source replay.

Root and threshold enclosures must retain unresolved strict boundaries,
and local sufficient admission must remain distinct from an actual
source-capture result. Cached verdicts are not premises. This owner
supplies no second research queue.

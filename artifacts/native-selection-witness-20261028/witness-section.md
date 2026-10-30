

## A fixed candidate for a quantitative selection certificate

<a id="native-selection-witness"></a>

The selection theorem proves existence but does not numerically locate a
shared preparation. This construction fixes one candidate at `kappa=1` and
a bounded paired interval-transit protocol. Its approximate spectral
calculation proposes the state; only whole-time enclosures and strict
capture of both complete laws can admit it. A truncated boundary series is
not treated as a finite-state certificate.

The code owner is
[`research/native_capacity_selection.py`](../../src/tnfr/research/native_capacity_selection.py).
It is a detached proof adapter over the existing native field and validated
Taylor kernel, not an installed dynamic-capacity graph executor. Declaration,
recipe, reviewed frozen source, controls, accepted steps and any unavailable
prefix belong to `artifacts/native-selection-witness-20261028/`. The program
owns the evaluated status; the rules below do not presume a successful run.

### W1. An exact capacity direction with a simple held comparison

Use CS1--CS2 with the actual two-bridge graph, `e=w=1/2`, `beta=kappa=1`.
Choose a scalar `epsilon>0` and the mean-one profile

\[
s=t=1+\epsilon,\qquad u=1-4\epsilon.
\tag{W1}
\]

It is positive for `epsilon<1/4`. Under the held model every reflected
native row is multiplied by `1+epsilon`. Thus its native paths and local
selection boundary agree exactly with the uniform model after a positive
constant clock rescaling. The central capacity does not enter the four
native rows in this invariant subspace. It is nevertheless retained, and
its evolution changes the other capacities in the dynamic model.

For dynamic capacity, (CS4) gives

\[
s(t)-t(t)=\epsilon\sqrt5
 \left(e^{-c_-t}-e^{-c_+t}\right),\qquad
c_\pm=(5\pm\sqrt5)/4.
\tag{W2}
\]

This is a prescribed consequence of the admitted linear capacity row, not
an imposed phase or form forcing. The contrast begins at zero and has
initial derivative `5*epsilon/2`.

### W2. The shared native curvature and the different mixed term

At the regular saddle write native deviations as
`y=(alpha,beta_dev,A,B_0)`. Let `lambda>0` be the unique positive eigenvalue
of `J_0` in (CS7). Use its right vector `r_u` oriented toward increasing `a`
and left row `ell`, normalized by `ell*r_u=1`. Let `P_s=I-r_u*ell` and let
`R` have orthonormal columns spanning `ker(ell)`. The stable restriction is
`S=R^T*J_0*R`. Then `y=r_u*xi+R*eta`, with `xi=ell*y`; the stable
projection is oblique even though the basis within its range is orthonormal.

The unique positive eigenvalue can be specified without a fitted vector
by the existing two-dimensional symmetric pencil:

\[
P(\lambda)=\begin{pmatrix}
2\lambda^2/5+\lambda/6-1/(8\pi^2)&\lambda^2/5-1/(8\pi^2)\\
\lambda^2/5-1/(8\pi^2)&3\lambda^2/5+\lambda/4+1/(4\pi^2)
\end{pmatrix}.
\tag{W3}
\]

Its positive determinant root lies in `(0,1)`. Write
`m=(1/(8*pi^2)-lambda^2/5)/(3*lambda^2/5+lambda/4+1/(4*pi^2))`.
Then

\[
r_u=(1,m,\tfrac{2\pi\lambda}{5}(2+m),
                 \tfrac{2\pi\lambda}{5}(1+3m))^T,\qquad
\ell_0=((1+m)/(4\pi\lambda),(1-2m)/(4\pi\lambda),1,m),
\quad \ell=\ell_0/(\ell_0r_u).
\tag{W4}
\]

These definitions fix the unstable-coordinate orientation independently
of a floating eigensolver's arbitrary signs. Approximate residual checks
in the recipe are consistency controls, not rigorous spectral enclosures.

At uniform capacity the native field has
`f(y)=J_0*y+Q_2(y)+O(||y||^3)`, with `q=3A-B_0`, `r=2B_0-A` and

\[
Q_2(y)=\left(
-\frac{\sqrt3}{4\pi}(\alpha+\beta_{\rm dev})q,
\frac{\sqrt3}{4\pi}\alpha r,
-\frac{\sqrt3}{2\pi}\alpha(\alpha+\beta_{\rm dev}),0
\right)^T.
\tag{W5}
\]

Define symmetric `K_2` by `eta^T*K_2*eta=ell*Q_2(R*eta)`.
The common native quadratic graph `h_20(eta)=eta^T*H*eta` satisfies

\[
S^TH+HS-\lambda H=K_2.
\tag{W6}
\]

All stable eigenvalues have negative real part, so this equation is
uniquely solvable. It retains the common quadratic contribution that would
be lost by using only the capacity-induced difference.

Let `E_a=diag(1,0,1,0)` and `L=ell*E_a*J_0*R`. On the stable directions
the common row scaling contributes zero to the unstable forcing at mixed
order. Integrating the remaining forcing (W2) gives the dynamic mixed row

\[
\Delta=-\frac52 L[(\lambda+c_-)I-S]^{-1}
                        [(\lambda+c_+)I-S]^{-1}.
\tag{W7}
\]

The held mixed row is zero by W1. The two resolvents are invertible.
Moreover `L!=0`: otherwise invariance of `ker(ell)` and invertibility of its
native restriction would force `ell*E_a` proportional to `ell`, contrary
to the cross-group argument in CS4. Hence `Delta!=0`.

Lift the separation to the ambient row `delta=Delta*R^T*P_s` and choose
`v=P_s*delta^T/||P_s*delta^T||`, `eta_0=R^T*v`. Since `delta*P_s=delta`,
this guarantees `ell*v=0`, `||v||=1` and the strictly positive separation
`d=Delta*eta_0=||delta||^2/||P_s*delta^T||`. The denominator cannot vanish,
since that would contradict `delta*P_s*delta^T=||delta||^2>0`. A raw
transpose without the stable projection would not automatically be a
stable native direction. Along this normalized direction the boundary
graphs have expansions

\[
\begin{aligned}
h_H(\epsilon\eta_0,\epsilon(1,1))
 &=\epsilon^2\eta_0^TH\eta_0+O(\epsilon^3),\\
h_D(\epsilon\eta_0,\epsilon(1,1))
 &=\epsilon^2(\eta_0^TH\eta_0+d)+O(\epsilon^3).
\end{aligned}
\tag{W8}
\]

Thus their quadratic midpoint proposes

\[
y_0=\epsilon v+
 \epsilon^2(\eta_0^TH\eta_0+d/2)r_u.
\tag{W9}
\]

For sufficiently small epsilon the asymptotic signs predict held evolution
toward the twist and dynamic evolution toward consensus. Equation (W8)
alone supplies no numerical epsilon threshold. Neither the recipe's
floating accuracy checks nor these asymptotic signs certify the finite
choice below.

### W3. One represented preparation and a whole-time acceptance rule

The declared candidate uses **`epsilon=1/32`**, so `s=t=33/32` and `u=7/8`.
The recipe computes (W3)--(W9), materializes `A,B_0,a,b` as binary64 once,
and retains those exact represented rationals. It then forms
`q=3*A-B_0`, `r=2*B_0-A` exactly. It does not round those derived
coordinates a second time or use different seeds for the two models.
Both exact six-coordinate initial boxes are identical.

If `F=(F_q,F_r,F_a,F_b)` denotes the existing unit-capacity reflected field,
the adapter uses its unchanged native pressure and phase metric through

\[
\dot q=\frac{(6s-t)F_q+3(s-t)F_r}{5},\quad
\dot r=\frac{2(t-s)F_q+(6t-s)F_r}{5},\quad
\dot a=sF_a,\quad\dot b=tF_b.
\tag{W10}
\]

The capacity rows are (CS3), or zero for the held model. They participate
in all Taylor jets and comparison derivatives; appending two capacity rows
to an otherwise unit-capacity native field would be a different model.
The shared validated kernel supplies strict Picard tubes, normalized Taylor
remainders and initial-uncertainty propagation in outward 128-bit rational
intervals. Every tube checks all three relative resultants, including the
central one, positive `s,t,u`, the reflected branch and the admitted sinc
domain. No graph state is advanced by this proof calculation.

The prospective attempt has one candidate, two models, step `1/8`, order
14 and at most 512 steps per model, up to structural time 64. Each model
stops at its first whole-box protected capture or first unavailable step.
The whole endpoint must satisfy the selected rectangle's strict margins
and `E.hi<7`. The paired success condition is sectors **`{0,+1}`**; the
predicted orientation held `+1`, dynamic `0` is recorded separately.
Other sectors, equal sectors, a resource stop or an inconclusive enclosure
do not pass that condition.

The total supervised allowance is 1200 seconds for interpreter setup,
controls, candidate generation and both interleaved transits, with 2 GiB
sampled aggregate RSS and 64 MiB of retained response allowance as specified
in the frozen protocol. RSS sampling is not a hard allocation cap. The
frozen source archive and final source review precede every scientific
execution. There is no retry, parameter scan or change of epsilon after
inspecting a response. An unavailable attempt retains its valid prefix
and reason; it does not refute CS's existence theorem or admit a modified
candidate. The resulting conditional certificate, if obtained, concerns
the ideal complete ODE and its capture theorem, not arbitrary future
floating integration steps or physical identification.

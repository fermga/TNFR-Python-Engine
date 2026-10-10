# Constructive continuous realization of the acquired-parent phase event

<a id="organized-parent-constructive-actuation"></a>

## Fixed path and realization question

The [one-unit obstruction](ORGANIZED_PARENT_CONTINUOUS_ACTUATION.md) remains
unchanged. This separate analytic question fixes one longer path before
assessing its work and identity bounds. Its purpose is to prove existence
under the same input channel and budget, not to optimize the obstruction or
select a controller from a scan.

| Item | Fixed choice |
| --- | --- |
| Source | The same four independently uncertain acquired C9 children, each with original form and phase allowances `1e-10`, carried to actual `z4=(x4,theta4)` at absolute slow time `s=4`; retain every fine coordinate and actual reached velocity |
| Target | Exactly `T(z4)=(x4,theta4+k)`, with `k[0,a,j]=(2*a-1)/4` and all other entries zero; restore every form and every unaffected phase lift |
| Complete law | Fixed matching 36-node support, all 72 unit edges, held unit capacities; `x_tau=(-Lx+gamma*S(theta))/4+Gamma_tau`, `theta_tau=gamma*Lx/4`, with `gamma=1/(1023*pi)` |
| Duration and clock | Elapsed slow duration `h=4`; `tau=(1023/1024)*t`, `s=gamma^2*tau/2`, so elapsed `T_tau=2*h/gamma^2`. Path parameter `v` runs from zero to one during this interval |
| Prescribed path | The cubic path (1) below, including both actual endpoint phase velocities; no nominal-state or zero-velocity replacement |
| Input | The explicit finite form input induced by that path and the initial complete state, declared before its response. No phase input, support/capacity change or intermediate state jump |
| Work budget | Signed net work `W_in=integral (Lx)^T*Gamma_tau d(tau)<=3`, retaining the actual endpoint storage rise plus all continuous loss. Positive-only supply, input amplitude and bandwidth are different obligations |
| Identity | Every marked C9 retains winding one throughout the forced path; parent registration may be lost as required by the target |
| Release | Set `Gamma=0` at absolute `s=8` without a state jump; match the exact endpoint and autonomous law before transferring the earlier restoring response by elapsed-time translation |
| Decision | A full-source realization with exact endpoints, retained children, admitted input and work, or an explicit obstruction. A failed sufficient bound for this chosen path is unavailable |

Write `c=mean(x4)`, `r=x4-c*1`, and let `L^+` be the inverse of `L`
on its mean-zero subspace, zero on the constant mode. For the supplied
displacement, `L^+k=(2*a-1)/16-(2*g-1)*(2*a-1)/32` at node `(g,a,j)`.
With `b=h*L*x4/(2*gamma)`, fix

\[
\begin{split}
 P(v)&=3v^2-2v^3,\qquad Q(v)=v-3v^2+2v^3,\\
 \theta(v)&=\theta_4+kP(v)+bQ(v),\\
 x(v)&=c\mathbf1+\frac{2\gamma}{h}L^+kP'(v)+rQ'(v),\\
 \Gamma_\tau(v;z_4)&=\frac1{T_\tau}\frac{dx(v)}{dv}
                       +\frac{Lx(v)-\gamma S(\theta(v))}{4}.
\end{split}\tag{1}
\]

This is a supplied input recipe depending on initial complete state and marked
targets. Its all-node actuation authority and initial-state knowledge are
premises. It does not infer pressure from an evaluated response, realize a
named operator automatically, select its occurrence, or establish optimal work.
No numerical trajectory, duration scan or source refinement is activated.

The declaration above was committed as `2f0eab07` before the bounds below
were assessed. The duration remains four slow units and the signed input-work
budget remains three. The construction supplies one state-dependent recipe,
not one input independent of the initial state: for every admitted actual
`z4`, that state's prescribed input realizes its own exact target.

## 1. Complete rows, regularity and exact endpoints

Index the nodes by \((g,a,j)\in\{0,1\}^2\times\{0,\ldots,8\}\),
and write \(\sigma_g=2g-1\), \(\sigma_a=2a-1\). On the fixed matching
support, the child and mixed vectors obey

\[
 L\sigma_a=2\sigma_a,\qquad
 L(\sigma_g\sigma_a)=4\sigma_g\sigma_a,\qquad
 k=\frac{\sigma_a-\sigma_g\sigma_a}{8}.
 \tag{2}
\]

Thus the expression for \(L^+k\) in the declaration is exact. It does not
require a numerical inverse, an inferred support or removal of a fine-state
coordinate. The connected graph's sole Laplacian kernel is the constant
mode; \(\mathbf1^Tk=0\) admits the inverse on its complement.

The endpoint identities for the fixed polynomials are

\[
 \begin{gathered}
 P(0)=0,\quad P(1)=1,\quad Q(0)=Q(1)=0,\\
 P'(0)=P'(1)=0,\qquad Q'(0)=Q'(1)=1.
 \end{gathered}\tag{3}
\]

Consequently (1) gives \((x(0),\theta(0))=(x_4,\theta_4)\) and
\((x(1),\theta(1))=(x_4,\theta_4+k)\), for every complete actual
initial state. Every form, including fine errors and the common mode, returns
to its starting value. Every unaffected phase lift also returns exactly.
The endpoint phase velocities are the inherited ones, not zero velocities:
\(\theta_\tau(0)=\theta_\tau(T_\tau)=\gamma Lx_4/4\).

Indeed \(d v/d\tau=1/T_\tau=\gamma^2/(2h)\), so

\[
 \theta_\tau
 =\frac{\gamma^2}{2h}(kP'+bQ')
 =\frac\gamma4\left(\frac{2\gamma}{h}kP'+Lx_4Q'\right)
 =\frac\gamma4Lx.
 \tag{4}
\]

The last row of (1) then supplies the form row exactly. Its derivative is
explicit:

\[
 x_v=\frac{2\gamma}{h}L^+kP''+(x_4-c\mathbf1)Q'',\qquad
 \Gamma_\tau=\frac{\gamma^2}{2h}x_v+\frac{Lx-\gamma S(\theta)}4.
 \tag{5}
\]

The path is polynomial and the input is smooth on the finite forcing
interval, since the fixed sine currents are smooth. In particular the input
is finite and integrable. The resulting initial-value problem has a unique
solution, and direct substitution proves that it is this prescribed path.
No solver residual or measured-response reconstruction is used in (5).
All-node actuation and exact initial-state knowledge are supplied premises;
the recipe does not give a controller using only parent observations.

Both means are retained: \(\mathbf1^T\theta(v)=\mathbf1^T\theta_4\),
\(\mathbf1^Tx(v)=\mathbf1^Tx_4\), and
\(\mathbf1^T\Gamma_\tau=0\). For the original acquired source the
form mean is zero, but it has not been discarded from the realization.
Under the original clock \(e=1023/1024\), the input is
\(\Gamma_t=e\Gamma_\tau\); under elapsed slow time
\(\sigma=\gamma^2\tau/2\), it is
\(\Gamma_\sigma=2\Gamma_\tau/\gamma^2\). Transforming both rows
therefore gives the same input-work integral in all three clocks.

## 2. Exact continuous loss and the original-source work budget

Retain the complete storage
\(H=\tfrac12x^TLx+\sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)]\).
The [input balance](ORGANIZED_PARENT_CONTINUOUS_ACTUATION.md#2-actual-endpoint-storage-and-input-work-balance)
applies to this smooth realization:

\[
 W_{\rm in}=\Delta H+\mathcal D,\qquad
 \mathcal D=2\int_0^h\|\theta_\sigma\|_2^2d\sigma
           =\frac2h\int_0^1\|kP'+bQ'\|_2^2dv.
 \tag{6}
\]

Here \(\Delta H=H(x_4,\theta_4+k)-H(x_4,\theta_4)\) is the same
actual endpoint storage change as in the hybrid theorem. No preceding loss
is credited as a work reservoir. Since \(Q'=1-P'\), integration gives

\[
 \int_0^1(P')^2dv=\frac65,\qquad
 \int_0^1P'Q'\,dv=-\frac15,\qquad
 \int_0^1(Q')^2dv=\frac15.
\]

Thus the loss is known exactly for every supplied actual state:

\[
 \mathcal D=\frac2h\left[
 \frac65\|k\|_2^2-\frac25 k^Tb+\frac15\|b\|_2^2\right]
 =\frac2h\left[\|k\|_2^2+\frac15\|k-b\|_2^2\right],
 \qquad \|k\|_2^2=\frac98.
 \tag{7}
\]

This includes the cost of matching the actual endpoint velocities. It is
not the unattainable constant-phase-velocity substitution from the earlier
necessary bound, and no minimum-work claim is made.

Fresh [acquisition admission](ORGANIZED_PARENT_ACQUISITION.md) supplies the
same reference at slow time four, with

\[
 \begin{gathered}
 x_4^*=\tfrac12(\sigma_g U_4+\sigma_a V_4),\qquad
 U_4=-\gamma Y_\Delta,\quad V_4=-\gamma Y_B,\\
 0\le Y_\Delta\le A\Delta_4,\quad 0\le Y_B\le AB_4,\quad
 A=1+2\bar\gamma^2,\\
 \|x_4-x_4^*\|_2\le R,\qquad
 \|\theta_4-\theta_4^*\|_2\le Z,
 \qquad R,Z<10^{-60}.
 \end{gathered}\tag{8}
\]

The bars denote outward bounds, not a changed law. These are the original
64 independent fine source directions, after the same isolated recovery and
acquisition; neither the allowances nor a dwell time is reduced. In
particular the actual four child means and forms need not equal the reference.

Using \(\|L\|_2\le8\), orthogonality of the two reference modes and
\(\|\sigma_g\|_2=\|\sigma_a\|_2=6\), a convenient uniform bound is

\[
 \|b\|_2\le M
 :=3hA(\overline B_4+\overline\Delta_4)
       +\frac{4hR}{\underline\gamma}.
 \tag{9}
\]

The admitted bounds \(B_4<1/64\), \(\Delta_4<1/500\),
\(A<1001/1000\) and \(\gamma>1/5000\) already imply
\(M<1/4\) at the frozen \(h=4\). Since
\(\|k\|_2=\sqrt{9/8}<9/8\), (7) gives the sufficient upper bound

\[
 \mathcal D\le\frac2h\left[
 \frac65\frac98+\frac25\frac98 M+\frac15M^2\right]
 <\frac{59}{80}.
 \tag{10}
\]

The source-dependent endpoint work has the already proved signed correction

\[
 \begin{split}
 W_*&=9[\cos B_4-\cos(B_4+1/2)]
       +18\cos\Delta_4[1-\cos(1/4)],\\
 |\Delta H-W_*|&\le18Z,\qquad 1<W_*<225/128.
 \end{split}\tag{11}
\]

The correction is linear and signed; acquired child means have not been
reset to recover an inapplicable quadratic remainder. Equations (10)-(11)
give, uniformly over the complete admitted original source,

\[
 0<W_{\rm in}
 <\frac{225}{128}+\frac{59}{80}+18Z
 =\frac{1597}{640}+18Z
 <\frac52<3.
 \tag{12}
\]

The detached assessor can retain the sharper outward version of (9)-(11).
The coarse rational margin already proves the frozen signed net budget.
It does not establish a bound of three on total positive supplied work,
input amplitude or bandwidth. A negative power segment, if present, is
accounted for as extraction in the signed balance; no actuator or reservoir
realizing such exchange has been derived.

## 3. Whole-path child identities with every fine direction retained

Write \(r_4=x_4-x_4^*\), \(z_4=\theta_4-\theta_4^*\), and apply
the same recipe to the reference in (8). Both \(k\) and
\(b^*=hLx_4^*/(2\gamma)\) are constant within each C9 child. Therefore
the reference path has the original winding carrier
\(\psi_j=(2\pi/9)(j-4)\) plus a time-dependent uniform child offset.
Its internal circular gaps remain exactly \(\alpha=2\pi/9\).

The actual path difference is exactly

\[
 x(v)-x^*(v)=r_4Q'(v),\qquad
 \theta(v)-\theta^*(v)=z_4+\frac{h}{2\gamma}Lr_4Q(v).
 \tag{13}
\]

The source form error is globally mean zero, so no extra common term enters
(13). The exact polynomial bounds are
\(\max|Q|=1/(6\sqrt3)<1/10\) and
\(-1/2\le Q'\le1\). Consequently, throughout the full forcing interval,

\[
 \|x-x^*\|_2\le R,\qquad
 \|\theta-\theta^*\|_2\le Z_{\rm path}
 :=Z+\frac{2hR}{5\underline\gamma}<10^{-55}.
 \tag{14}
\]

This is an exact source-dependent path comparison, not an unforced
stability estimate applied to a forced trajectory. In particular the long
structural duration does not license resetting errors to a symmetric family.

Every oriented child gap on its original winding branch lies in
\([\alpha-2Z_{\rm path},\alpha+2Z_{\rm path}]\subset(0,\pi/2)\).
This includes the closing C9 edge with its original integer lift correction.
No gap crosses a circular branch boundary, and the sum of the continued
principal gaps remains \(2\pi\). All four winding-one child identities
are therefore retained for every actual source state along the entire path.
The parent registration is allowed to fail, as required by the target;
no preservation of its identity throughout the intervention is claimed.

## 4. Release into the already admitted autonomous response

Forcing starts at absolute slow time four and ends at absolute time eight.
Set \(\Gamma=0\) at the latter instant, with no state jump. The input
may have a finite discontinuity when switched on or off; the state remains
continuous and the phase velocity remains continuous because form does.
Such a switch remains within the integrable input contract.

By (3), the release state is exactly the previous theorem's actual
post-event state \(T(z_4)\), not an interval-reset substitute or a state
whose parent means merely agree. The fixed support, capacities, complete
sine law and clocks are identical. Uniqueness of the autonomous flow then
identifies every subsequent state with the
[restoring response](ORGANIZED_PARENT_RESTORING_RESPONSE.md) translated by
four slow units. Its finite child retention and parent recovery apply through
absolute time twelve; its actual-state eight-coordinate prediction applies
at absolute time nine. No additional acquisition interval or tighter source
is required for this handoff.

The previous paired no-kick comparison, if used, must likewise be initialized
from its original actual state \(z_4\) at the translated release origin.
It is not automatically a comparison against a second system evolving freely
from absolute time four throughout this forcing interval. No such new
chronological control is proved here. Nor does translating a finite theorem
establish indefinite maintenance or autonomous event occurrence.

## 5. Evidence, implementation and scope

The [detached implementation](../../src/tnfr/physics/_organized_parent_constructive_actuation.py)
re-admits the eight original primitive source
allowances and rebuilds the acquired state and release premises. Its path
arithmetic evaluates (1) on admitted complete state; it does not install an
input in a live network or certify that arbitrary supplied vectors belong to
the acquired family. Conditional path/work bounds and source-dependent
admission remain separate. A failed source or sufficient estimate is
unavailable, not a proof of nonreachability.

The [independent controls](../../tests/physics/test_organized_parent_constructive_actuation.py)
check the exact polynomial integrals, endpoint and
phase-row identities, signed storage balance, circular child branches and
whole-source bounds. The result is an explicit mathematical realization
under supplied full-node actuation and complete initial-state knowledge.
It preserves the one-unit impossibility result, without claiming optimal
duration, optimal work, a registered operator's runtime admission, a selected
natural forcing law or a laboratory actuator and clock.

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

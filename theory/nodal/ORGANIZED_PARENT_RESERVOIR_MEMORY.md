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

# Autonomous exchange with a finite auxiliary reservoir

<a id="organized-parent-reservoir-exchange"></a>

## Fixed joint model and prospective question

The [collective construction](ORGANIZED_PARENT_COLLECTIVE_ACTUATION.md)
still supplies its input as a function of time. This separate question fixes
one autonomous joint law and a finite reservoir preparation before assessing
its response. A valid balance alone will not establish registration loss or
return. Failure of the declared response remains an admissible result.

| Item | Fixed choice |
| --- | --- |
| Network source | The same four independently uncertain recovered C9 children, each with original form and phase allowances `1e-10`, carried through unchanged acquisition to actual `z4=(x4,theta4)` at absolute slow time `s=4`; retain every reached form, phase and velocity |
| Network support and capacity | Matching 36-node support, all 72 unit edges, held unit capacities; no support or phase event during the new flow |
| Auxiliary state | One signed real form coordinate `r`, with held unit capacity in the normalized `tau` clock and the pressure law specified below. No phase coordinate is consumed for this auxiliary element |
| Reservoir preparation | Independently supplied `r0 in [12/5-1e-10,12/5+1e-10]`; added reservoir storage `H_r=r^2/2` has budget three. This budget excludes the network's already accounted acquired storage |
| Complete connected law | `x_tau=(-Lx+gamma*S(theta))/4+b*r`, `theta_tau=gamma*Lx/4`, `r_tau=-b^T*Lx-lambda*r`, with `gamma=1/(1023*pi)`, `b[g,a,j]=gamma*(2*a-1)/12`, `lambda=gamma^2/2` |
| Clock | `tau=(1023/1024)*t`, absolute slow time `s=gamma^2*tau/2`, elapsed slow time `h=s-4`; transform all three rows under a clock change |
| Storage and work | Complete joint storage `H_joint=H(x,theta)+r^2/2`, with the existing `H=x^T*L*x/2+sum_edges(1-cos(gap))`; network input work `integral (Lx)^T*b*r d(tau)` must be supplied by the finite reservoir, with its remaining storage and loss retained |
| Disconnected control | The same actual network and reservoir preparation, but `b=0` and unchanged `lambda`; the network then follows its original unforced law. Zero initial reservoir alone is not this control |
| Registration-loss readout | At `h=1/2` (absolute `s=9/2`), every intraparent same-slot circular gap must exceed `1/4` in magnitude in BOTH connected parents; both disconnected parents must retain registration |
| Return deadline | At `h=8` (absolute `s=12`), both connected parents must have every intraparent same-slot circular gap strictly below `1/4` in magnitude |
| Retained identity | Every marked C9 must retain winding one throughout `0<=h<=8` in both branches |
| Decision | Source-inclusive loss/return discrimination with finite supplied reservoir work and retained children, or a scoped obstruction to the fixed criterion. A failed sufficient estimate remains unavailable rather than proving failure |

The network and reservoir are initialized jointly at the start of this
comparison; their subsequent equations contain no time profile, timed release
or encoded copy of the earlier cubic input. Initializing this coupling and
preparing the reservoir are supplied premises, not autonomously selected events.
The existing 72-edge network Laplacian is unchanged: `b*r` is an added
constitutive exchange channel, not an unaccounted new graph edge.

Use the existing normalized structural form, phase, storage and time units.
The new coordinate has the form scale that makes `r^2/2` a storage in the
same units as `H`; the coefficients in its supplied pressure and network
input have the rate units required by the three displayed rows. No laboratory
units, mass, particle or thermal-bath identification is supplied by this choice.

This port excites the shared child-index mode of both parents, not the prior
one-parent phase map. A reference may be parameterized by each actual `r0`
in its frozen interval; no midpoint substitution discards reservoir uncertainty.
Actual and reference reservoir states subsequently respond to their own
networks, so a new joint-state error proof is required. The preceding
identical-input comparison does not apply unchanged.

The auxiliary coordinate is not claimed to be an acquired or formed NFR mode.
Its origin, pressure, storage, marked coupling and preparation remain supplied
constitutive assumptions; the joint balance can restrict consistency without
selecting a unique law. No source refinement, coefficient scan or numerical
trajectory is activated by this analytic question. The specified readouts and
budgets will not be moved after assessing the response.

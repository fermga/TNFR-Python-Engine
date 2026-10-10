# Continuous form actuation of the acquired-parent phase event

<a id="organized-parent-continuous-actuation"></a>

## Fixed realization question

The [restoring-response theorem](ORGANIZED_PARENT_RESTORING_RESPONSE.md)
admits a supplied hybrid phase map and its subsequent unforced response.
This question tests a different premise: finite-duration realization of that
same endpoint map through an added form input. The preceding work-balance
review motivates an obstruction test; this is an analytic admission, not a
blind experiment or a new response trajectory.

| Item | Fixed choice |
| --- | --- |
| Actual source | The same four independently uncertain acquired C9 children, with original form and phase allowances `1e-10`, carried to the actual state `z4=(x4,theta4)` at slow time `s=4`; no nominal reset or smaller error family |
| Target | The actual mapped endpoint `T(z4)`: restore every form to `x4`, shift every phase of child `(0,0)` by `-1/4` and child `(0,1)` by `+1/4`, and return all other phases to their original declared lifts |
| Fixed substrate | Original matching 36-node support, all 72 unit edges, held unit capacities, signed real forms and primitive circular phases with their continuous lifts |
| Complete forced law | `dx/dtau=(-Lx+gamma*S(theta))/4+Gamma_tau`, `dtheta/dtau=gamma*Lx/4`; no phase forcing, support event, capacity change or intermediate hybrid jump |
| Input class | Any integrable `Gamma_tau` on the finite interval, with absolutely continuous form and the stated rows almost everywhere; this includes continuous inputs and excludes impulsive resets. No amplitude or bandwidth restriction is used to manufacture an obstruction |
| Clock and duration | `tau=(1023/1024)*t`, `gamma=1/(1023*pi)`, `s=gamma^2*tau/2`; elapsed slow duration `h=1`, so elapsed `tau=2/gamma^2`. Transform the input along with both rows |
| Storage and input work | Retain `H=0.5*x^T*L*x+sum_edges(1-cos(gap))`; supplied signed work is `W_in=integral (Lx)^T*Gamma_tau d(tau)`, including the actual endpoint storage change and continuous loss |
| Budget | `W_in<=3` structural storage units. A bound on total positive supplied work would be at least as restrictive; no preceding loss is a reservoir |
| Path scope | Allow all paths in this input class when testing impossibility. If realization is not excluded, a constructive certificate must additionally establish the required child identities and endpoint conditions |
| Decision | A source-inclusive obstruction for every admitted form input, or an admitted realization with its full path and work. An unavailable sufficient estimate is unresolved |

The target is the exact hybrid-map endpoint of the state at `s=4`, reached
after one elapsed slow unit. It is not the unforced evolution of that endpoint
over the same interval. Prescribed phase lifts make the displacement explicit;
any extension to a circular-only endpoint must account for possible winding
increments separately. The earlier hybrid theorem and its frozen evidence
remain unchanged. Failure under this duration, budget and input channel does
not rule out another duration, work supply or independently admitted phase law.

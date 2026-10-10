# Collective actuation with unresolved fine state

<a id="organized-parent-collective-actuation"></a>

## Fixed observation, input and response question

This analytic question reduces the information and input authority of the
[full-state construction](ORGANIZED_PARENT_CONSTRUCTIVE_ACTUATION.md).
The actual acquired state is retained; projecting its observations constructs
a reference and never resets a fine coordinate. The declaration below fixes
one input recipe and its response budgets before their assessment.

| Item | Fixed choice |
| --- | --- |
| Source | The same four independently uncertain recovered C9 children, each with original form and phase allowances `1e-10`, carried through the unchanged parent acquisition to actual `z4=(x4,theta4)` at absolute slow time `s=4` |
| Complete law | Matching 36-node support with all 72 unit edges, held unit capacities; `x_tau=(-Lx+gamma*S(theta))/4+Gamma_tau`, `theta_tau=gamma*Lx/4`, `gamma=1/(1023*pi)` |
| Clock and duration | `tau=(1023/1024)*t`, `s=gamma^2*tau/2`; forcing runs from `s=4` to `s=8`, with fixed elapsed duration `h=4` |
| Available observations | Exactly the four actual child form means `X[g,a]` and four actual mean lifted phase offsets `Phi[g,a]=mean_j(theta[g,a,j]-psi[j])` at `s=4`, with `psi[j]=(2*pi/9)*(j-4)`. No individual fine coordinate or later observation is supplied to the input or predictor |
| Reference initialization | Repeat each observed form mean within its marked child; initialize reference phases as `psi[j]+Phi[g,a]`. Retain the common form and phase means |
| Input | Apply the previously fixed cubic recipe to this reference, using the same displacement `k[0,a,j]=(2*a-1)/4`, zero on parent one. Supply its resulting form input identically to all nine nodes of each child: four input channels, no fine feedback, phase input, support event or capacity change |
| Actual endpoint | At `s=8`, require `||x8-x4||_2<=1e-7` and `||theta8-theta4-k||_2<=1e-7`, using the declared continuous lifts. These are separate form and phase budgets; no exact fine-state endpoint is assumed |
| Work | Actual signed net input work `integral (Lx)^T*Gamma_tau d(tau)<=3` over the forced interval; include actual storage change and continuous loss |
| Release and predictor | Set `Gamma=0` at `s=8` without a state jump. Continue the exact eight-coordinate child-constant law from `(X4,Phi4+k_child)`, already determined by the initial observations; no measured release state initializes this prediction |
| Readout | At `s=9`, each of the four actual child form means and four lifted phase-offset means differs from that predictor by at most `1e-6`, separately in each channel |
| Identity | Preserve every marked child's winding one throughout `4<=s<=12`. Parent zero must lose registration at release; both parents must recover it at `s=12`, with every intraparent same-slot circular gap strictly below `1/4` in magnitude |
| Decision | Admit the original source, four-channel realization within endpoint and work budgets, child retention, disruption, recovery and predictive accuracy, or a scoped obstruction. A failed sufficient estimate remains unavailable |

Let `P` be the orthogonal projection that replaces each nine-node block by
its mean, applied to forms and carrier-subtracted phase lifts. With the fixed
`L`, `k` and `h`, the input is precisely equation (1) of the full-state
construction initialized at `(P*x4, psi+P*(theta4-psi))`. It is supplied as a
function of the eight initial observations and time, independently of the
subsequently evaluated response. Its uniformity within each child is a proof
obligation, not an additional approximation of that recipe.
The inherited four-child law retains denominator four and both original
clock transformations; its two external neighbors do not renormalize the law.

Compare actual and reference flows under this same input. The earlier
comparison of two different state-dependent inputs cannot establish the new
error bounds. Endpoint mismatch must enter release, work and prediction;
the earlier exact-endpoint handoff cannot be assumed. Initial observation
accuracy is exact in this question; noise, finite precision, actuator tracking
and distribution errors require their own admission.

Marked children, continuous phase lifts, fixed support and capacity, supplied
timing and uniform actuation remain premises. Signed net work does not bound
positive-only supply, amplitude or bandwidth. No registered operator runtime,
autonomous occurrence, universal quotient, physical measurement bridge or
optimal control is claimed. No source refinement, trajectory, duration scan
or new numerical campaign is activated.

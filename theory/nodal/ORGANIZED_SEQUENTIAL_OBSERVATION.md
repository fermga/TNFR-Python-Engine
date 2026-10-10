# Observing outer winding on one continuing organization

<a id="organized-sequential-observation"></a>

## Declared single-history observation contract

The [matched-source inverse](ORGANIZED_OUTER_OBSERVATION.md) identifies a
collective winding but assumes eleven continuations of one initial state.
This gate asks whether one prescribed observation history can remove that
premise while retaining unknown background motion and probe backaction.

| Item | Fixed choice and scope |
| --- | --- |
| Complete model | Same 45-node C9-by-C5 product, nodes `9*c+j`, 90 unit edges, held unit capacities and smooth-sine phase domain; `x_tau=(-Lx+gamma*S(theta))/4`, `theta_tau=gamma*Lx/4`, `gamma=1/(1023*pi)`, with no continuous input |
| Clock | `tau=(1023/1024)*t`; elapsed tau starts at the actual unknown initial source; all intervals and both continuous rows use this clock |
| Initial class | Exactly the earlier admitted class: every `abs(x_i)<=X=1/100`; `theta[c,j]=2*pi*(j-4)/9+Phi_c+z[c,j]`, unknown lifted origins Phi, zero child-mean z and every `abs(z[c,j])<=zeta=1e-6`; principal adjacent origin gaps have magnitude at most `A=7/5` and common winding `w in {-1,0,1}` |
| Source scope | Both earlier actually acquired families belong to this initial class by the existing inclusion theorem. All fine coordinates are retained; generic members are not thereby proved acquired |
| One carried history | Five consecutive marked blocks, c=0,1,2,3,4. Each block starts at the actual endpoint of its predecessor; no source copy, reset, new preparation or separate baseline branch is an observation input |
| Window schedule | Four equal windows per block, each `h=1/100000` in tau; block start `s_c=4*c*h`, total duration `T=20*h=1/5000` |
| Supplied phase events | In block c let `k_c[b,j]=delta[b,c]-1/5`. Apply jumps `+epsilon*k_c`, `-2*epsilon*k_c`, `+epsilon*k_c` at local times h, 2h, 3h, with `epsilon=1/10`; forms, support and capacities are unchanged. Cumulative commanded offsets in the four windows are respectively `0,+epsilon*k_c,-epsilon*k_c,0` |
| Event meaning | The last event restores the commanded offset, not the evolved state. All drift and backaction survive into later windows and blocks. These are supplied hybrid jumps; no registered operator descent, controller or occurrence law is inferred |
| Readings | In each block record only the receiver child `(c+1) mod 5` mean signed form at the five local endpoints `0,h,2h,3h,4h`. The inverse receives five ordered rows of length five, hence 25 primitive scalar readings. Forms are continuous at the phase events |
| Reading errors | Each raw scalar has arbitrary additive error of magnitude at most `eta=1e-15`; no independence is assumed. Reuse the same endpoint reading in adjacent increments; do not count it as independent measurements |
| Fixed combinations | With `I_k=Y_(k+1)-Y_k`, use `D_C=-I_0/3+I_1-I_2+I_3/3` and `D_S=-I_0+I_1+I_2-I_3`. Their exact raw-error bounds are `(16/3)*eta` and `6*eta`; constant and affine background slopes cancel algebraically |
| Inverse boundary | Only the 25 raw readings and the fixed known protocol enter the inverse. Hidden phases, source reports, assigned winding and extra reference readings are excluded. A hypothetical unprobed continuation may serve a proof, never a second measured branch |
| Required inference | Retain complete nonlinear motion and the cumulative history when bounding both initial boundary quadratures and the initial full-cycle winding. Prove uniform conditional identification on the admitted class, or return an honest scoped obstruction or unavailable certificate |
| Candidate meaning | Sector intervals give necessary compatibility, not joint realizability witnesses. A singleton identifies an actual admitted source only conditional on this one-history state/law/observation contract; arbitrary records are not authenticated |
| Work budgets | Evaluate signed jumps from the actual pre-event states. Require every absolute event work below 4 and the sum of the 15 absolute event works below 40 structural storage units. Do not cancel paid events using a formal inverse command, credit continuous loss as a reserve, or infer a reservoir |
| Identity horizon | Retain all five internal winding-one cycles and all nine respective outer cycles at every event and throughout `0<=tau<=T`; only then does observation of the initial winding identify the continuing object's winding. No generic all-future maintenance is required or inferred |
| Numerical boundary | Exact rational admission and the existing 128-bit outward interval arithmetic. Freeze this word, readings, errors, duration and budgets before assessment. No trajectory or frozen-producer replay, amplitude scan, precision refinement or post-assessment protocol tuning |

The analytical task is to justify the sequential cancellation under the
complete rows, not to substitute sequential samples into the earlier
matched-source formula. A successful result would remove a mathematical
preparation dependency while retaining externally supplied interventions,
known law and clock, and an independently uncalibrated observation model.

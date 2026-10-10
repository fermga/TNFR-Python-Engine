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

The declaration above was committed as `8c0d4cca` before assessment. The
following proof evaluates its fixed rational bounds without integrating a
trajectory or changing the protocol after observing the result.

## 1. One actual history and the initial identity

Write \(\psi_j=2\pi(j-4)/9\), and let
\(a_c=\operatorname{wrap}(\Phi_{c+1}-\Phi_c)\) denote the
initial child-origin gaps. The initial fine boundary gaps and their moments
are

\[
 d_{c,j}^0=a_c+z_{c+1,j}-z_{c,j},\qquad
 C_c^0=\frac19\sum_j\cos d_{c,j}^0,\qquad
 S_c^0=\frac19\sum_j\sin d_{c,j}^0.
 \tag{1}
\]

The actual initial fine errors telescope around every outer slot cycle;
each has sum \(2\pi w\), with the same unknown integer
\(w\in\{-1,0,1\}\). The internal C9 cycles have winding one.
These are identities of all 45 actual phase coordinates, not five assigned
labels. The [matched-source state argument](ORGANIZED_OUTER_OBSERVATION.md#1-state-marked-cycles-and-the-scope-of-the-unknown-source)
establishes this initial class and includes both previously acquired sources
without resetting their residuals. The present argument does not require
the larger class to have a demonstrated acquisition mechanism.

Let \(g\) be the upper endpoint of the shared outward interval for
\(\gamma\), and set

\[
 T=20h,\qquad B=X+gT,\qquad D=4gBT.
 \tag{2}
\]

Heat contraction for \(-L/4\), together with
\(\|S(\theta)\|_\infty\le4\), gives
\(\|x(t)\|_\infty\le X+gt\le B\) along the entire
history. Phase events leave x unchanged. The full phase row gives
\(\|\dot\theta\|_\infty\le2gB\), hence every edge
accumulates continuous phase drift of magnitude at most D. Remove only the
explicit cumulative commanded offset from the phase lift; its remaining
drift is continuous through all fifteen events. At the start of every block
the cumulative command is zero, but the evolved form and phase drift remain.

For a block starting at \(s_c=4ch\), let \(C_c^s,S_c^s\)
be its actual boundary moments before any of that block's events. The
global Lipschitz bounds for sine and cosine imply

\[
 |C_c^s-C_c^0|\le D,\qquad |S_c^s-S_c^0|\le D.
 \tag{3}
\]

Thus the proof can transport local information back to the original
moments. It does not identify the starts of different blocks or reuse a
state discarded by an earlier observation.

## 2. Exact cancellation and the unobserved comparison

For one block let \(Y_k\) be its five actual receiver mean forms,
temporarily without readout error. In endpoint coordinates the two fixed
combinations are

\[
 \begin{split}
 D_C&=(Y_0-4Y_1+6Y_2-4Y_3+Y_4)/3,\\
 D_S&=Y_0-2Y_1+2Y_3-Y_4.
 \end{split}\tag{4}
\]

Both annihilate every quadratic polynomial in time. Their raw coefficient
absolute sums are \(16/3\) and 6, respectively, giving exactly the
declared worst-case reading-error bounds. A common endpoint is one reading,
even when it occurs in two adjacent increments.

For the proof alone, continue the block's actual initial state without its
three local events for a time \(H=4h\). Denote this hypothetical
reference by \((\bar x,\bar\theta)\), and its receiver mean
form by b. It is not measured and is never passed to the inverse. Its form
bound is also B: it begins with
\(\|x(s_c)\|_\infty\le X+gs_c\), and
\(s_c+H\le T\). In particular, each block's reference retains all
history accumulated before that block.

The degree-four full rows imply

\[
 \begin{split}
 \|\dot{\bar x}\|_\infty&\le2B+g,\\
 \|\ddot{\bar x}\|_\infty&\le4B+2g+4g^2B,\\
 |b'''|\le\|\bar x'''\|_\infty
 &\le M_3:=8B+4g+16g^2B+4g^3+16g^3B^2.
 \end{split}\tag{5}
\]

Indeed, each edge phase rate is at most \(4gB\), each edge phase
acceleration at most \(4g(2B+g)\), and differentiating the four
sine terms gives
\(\|S''\|_\infty\le64g^2B^2+16g(2B+g)\).
Together with \(\|L/4\|_\infty=2\), these yield (5).
No equilibrium or constant background is assumed.

Writing \(E_h\) for time translation by h, the first combination
on b is \((E_h-1)^4b/3\), and the second is
\(-(E_h+1)(E_h-1)^3b\). A third finite difference is an
integral of \(b'''\) over a cube of side h. Bounding the two
third differences in each expression therefore gives

\[
 |D_C[b]|\le\frac23M_3h^3,\qquad
 |D_S[b]|\le2M_3h^3.
 \tag{6}
\]

This is the residual unknown background motion after the exact algebraic
cancellation. It differs from the matched-source cancellation, where the
same independently measured endpoint baseline cancels directly.

## 3. Complete nonlinear backaction within each block

At local time t let \(q(t)\in\{0,+\epsilon,-\epsilon,0\}\)
be the commanded amplitude in its four windows. Write

\[
 y=x-\bar x,\qquad
 v=\theta-\bar\theta-q(t)k_c.
 \tag{7}
\]

The explicit event is removed from v, so both y and v are continuous and
start at zero. Their differential equations, away from the event times,
retain the full form diffusion and phase feedback. The phase row yields
\(\|v(t)\|_\infty\le2g\int_0^t\|y(r)\|_\infty\,dr\).
Every edge of the commanded offset has magnitude at most \(\epsilon\),
and the global sine Lipschitz bound yields

\[
 \|y(t)\|_\infty
 \le g\epsilon t+2g\int_0^t\|v(r)\|_\infty\,dr.
 \tag{8}
\]

If \(2g^2H^2\le1/2\), a supremum argument at every time up
to H gives

\[
 \|y(t)\|_\infty\le2g\epsilon t,\qquad
 \|v(t)\|_\infty\le2g^2\epsilon t^2.
 \tag{9}
\]

These estimates do not reset y or v at an event. In particular, setting
q back to zero in the fourth window leaves the accumulated backaction.

Freeze only the reference phase at the block's initial state when defining
the leading vector slope

\[
 F_q=\frac\gamma4
   \{S(\bar\theta(0)+qk_c)-S(\bar\theta(0))\}.
 \tag{10}
\]

The function \(\sin(a+u)-\sin a\) has derivative with
respect to a of magnitude at most \(|u|\). The reference edge
drift at time t is at most \(4gBt\); the additional phase v
has edge magnitude at most \(2\|v\|_\infty\).
Using all four incident edges and (9),

\[
 \|\dot y-F_q\|_\infty
 \le J(t):=4g\epsilon(1+gB)t+4g^3\epsilon t^2.
 \tag{11}
\]

The first term includes \(\|Ly/4\|_\infty\), not just
the pressure perturbation. This keeps the complete finite evolution rather
than replacing a mean-form increment by an onset derivative.

For the receiver child \(c+1\), the mean of (10) at the two
nonzero amplitudes is exactly

\[
 A_\pm=\frac\gamma4
  \{\mathord\pm\sin\epsilon\,C_c^s
           +(1-\cos\epsilon)S_c^s\}.
 \tag{12}
\]

The four leading mean slopes are consequently \(0,A_+,A_-,0\).
Integrate (11) using the absolute increment weights
\((1/3,1,1,1/3)\) and \((1,1,1,1)\). Their first
time moments are \((16/3)h^2\) and \(8h^2\), and
their second time moments are \((116/9)h^3\) and
\((64/3)h^3\). The resulting backaction remainders are

\[
 \begin{split}
 E_C&=\frac{64}{3}g\epsilon(1+gB)h^2
                     +\frac{464}{9}g^3\epsilon h^3,\\
 E_S&=32g\epsilon(1+gB)h^2
                     +\frac{256}{3}g^3\epsilon h^3.
 \end{split}\tag{13}
\]

This argument applies independently to every block's actual carried start;
the uniform constants retain all earlier effects through B and D.

## 4. Initial quadratures and uniform winding identification

Define positive outward response factors
\(f_C=g h[\sin\epsilon]^+/2\) and
\(f_S=g h[1-\cos\epsilon]^+/2\), where a plus denotes
the interval upper endpoint. Combining the background terms (6), the
backaction terms (13), the exact reading coefficients, and the transport
bound (3) gives

\[
 \begin{split}
 D_C&=\frac{\gamma h\sin\epsilon}{2}C_c^0+\delta_C,
 &|\delta_C|&\le b_C:=E_C+\frac23M_3h^3
                         +\frac{16}{3}\eta+f_CD,\\
 D_S&=\frac{\gamma h(1-\cos\epsilon)}2S_c^0+\delta_S,
 &|\delta_S|&\le b_S:=E_S+2M_3h^3+6\eta+f_SD.
 \end{split}\tag{14}
\]

The two inferred moments are now explicitly those of the original unknown
state. Later blocks need no pretended common preparation. Both errors are
uniform over all allowed initial forms, fine phases and reading errors.

The [initial-moment inverse](ORGANIZED_OUTER_OBSERVATION.md#3-angle-inclusion-cycle-consistency-and-uniform-identification)
therefore applies with these new error bounds: scale the raw combinations
by \(K_C=2/([\gamma]h[\sin\epsilon])\) and
\(K_S=2/([\gamma]h[1-\cos\epsilon])\), and intersect
with the original cosine prior \([c_0,1]\),
\(c_0=[\cos(A+2\zeta)]^->0\), and sine prior
\([-1,1]\). The original annulus
\(\cos(2\zeta)\le|C_c^0+iS_c^0|\le1\) remains
a necessary check because (14) transports to the initial moments.
Every raw form reading must also have magnitude at most \(B+\eta\).

Take the outward argument of each surviving moment rectangle, enlarge by
\(2\zeta\) on each side, and clip to \([-A,A]\).
This encloses each original origin gap \(a_c\). Sum the five
intervals and retain the candidate sectors whose outward \(2\pi w\)
intervals intersect that sum. Under the actual-source premises the true
sector is retained. A singleton then identifies it; the filters and
sector intersections alone do not prove that arbitrary raw records arise
from one admitted trajectory.

For clarity, the uniform interval-width bound uses the same shared outward
rounding allowances as the matched-source proof. With \(u=2^{-128}\),

\[
 \begin{split}
 \rho_K&=(2K^++\operatorname{wid}(K)+2)u,\\
 w_C&=2K_C^+b_C+\operatorname{wid}(K_C)(f_C+2b_C)+\rho_{K_C},\\
 w_S&=2K_S^+b_S+\operatorname{wid}(K_S)(f_S+2b_S)+\rho_{K_S},\\
 d&=w_S/c_0+w_C/c_0^2+4\zeta+10^{-20}.
 \end{split}\tag{15}
\]

Here d bounds each initial-angle interval width, and \(5d\) bounds
the cycle-sum width. The existing argument-evaluation proof applies on the
same domain \(c_0>1/6\), \(|S|\le1\); no precision or
transcendental kernel is changed. The fixed exact-rational outward
assessment gives conservative strict bounds

\[
 \begin{split}
 B&<0.010000063,&D&<2.489243\cdot10^{-9},&M_3&<0.081245127,\\
 b_C&<7.176748\cdot10^{-14},&b_S&<1.057319\cdot10^{-13},&c_0&>0.169965,\\
 w_C&<0.000924139,&w_S&<0.027207175,&d&<0.192069229,\\
 5d&<0.960346143,&2\pi-5d&>5.322839.
 \end{split}\tag{16}
\]

The separated winding values are more than 6.28 apart, whereas the cycle
interval is less than 0.961 wide. Outward widths of the three fixed
\(2\pi w\) enclosures are smaller than the positive separation
margin. Thus every actual admitted initial state and every allowed reading
error yields exactly one necessary sector. This is a uniform conditional
identification theorem, not a classification of sampled sources.

## 5. Signed event work and retention of the continuing identity

Each event changes eighteen outer edges and no internal gaps or forms.
For its actual pre-event gaps, the signed storage jump is

\[
 W=\sum_{e\ \mathrm{changed}}
    \{\cos d_e-\cos(d_e+\Delta q\,\Delta k_{c,e})\}.
 \tag{17}
\]

The cosine Lipschitz bound gives \(|W|\le18|\Delta q|\).
The three event amplitudes in each block have magnitudes
\(\epsilon,2\epsilon,\epsilon\); hence every absolute
work is at most \(36\epsilon=3.6<4\), and

\[
 \sum_{\text{15 events}}|W|
 \le5\cdot18(\epsilon+2\epsilon+\epsilon)=36<40.
 \tag{18}
\]

These bounds evaluate the storage law on each event's actual state without
prescribing the sign of its work. The final inverse command does not undo
the middle event's payment or return the evolved state. No continuous
loss is credited as a reserve, and no reservoir or event-occurrence law is
deduced.

At any point in the word, each commanded outer edge offset has magnitude
at most \(\epsilon\), including the endpoints of the larger
middle jump. The full continuous drift is bounded by D. Thus

\[
 \begin{split}
 \alpha-2\zeta-D&>0.698129698,\\
 \pi/2-\alpha-2\zeta-D&>0.872662623,\\
 \pi/2-A-2\zeta-\epsilon-D&>0.070794324,
 \qquad \alpha=2\pi/9.
 \end{split}\tag{19}
\]

Internal gaps remain positive and acute; outer gaps remain strictly acute
at all times and on both sides of every event. Each supplied event is a
single-valued vertex phase offset, so its increments telescope around
every marked cycle. Continuous drift cannot cross a phase-chart boundary.
All five internal winding-one identities and all nine original outer
windings therefore persist throughout \([0,T]\). Identification
of the original winding consequently identifies the continuing object's
winding over this finite history. This does not prove future maintenance
after T or recovery under indefinitely repeated observations.

## 6. Result and implementation boundary

The fixed word identifies the unknown outer winding from 25 collective
form readings of one carried organization. It removes the earlier need
for eleven matched initial-state branches. The theorem retains unknown
background motion, nonlinear backaction, the complete fine state, all
fifteen signed events and arbitrary bounded reading errors. Work and
identity have separately proved finite bounds.

The [sequential observation owner](../../src/tnfr/physics/_organized_sequential_observation.py)
admits primitive readings and rebuilds the fixed protocol before applying
the shared necessary initial-moment inverse. Its
[contract tests](../../tests/physics/test_organized_sequential_observation.py)
cover exact observation coefficients, actual signed event transformations,
invalid domains, uniform margins and conditional inference. Algebraic
synthetic records exercise the proved information envelope; they are not
presented as numerically evaluated trajectories or new formation evidence.
The previous matched-source protocol and its result remain unchanged.

Known support, law and structural clock, access to each prescribed event,
and the bounded mean-form observation model remain assumptions. The word
is externally supplied, not an autonomously selected sensor or an admitted
registered operator sequence. The collective NFR keeps its declared
identity together with constituent and boundary state; a winding label
does not replace that state or establish a physical identification.

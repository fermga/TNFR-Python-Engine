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

The declaration above was committed as `be66de2d` before its certificate was
assessed. The obstruction below concerns that unchanged input class, endpoint,
duration and budget. No input is fitted or simulated.

## 1. Complete forced rows, regularity and clocks

Let the local actuation interval start at the actual acquired state
\(z_4=(x_4,\theta_4)\). Use elapsed structural time \(\tau\) and
write \(T_\tau=2h/\gamma^2\). On the unchanged matching support, the
forced initial-value problem is

\[
 x_\tau=\frac{-Lx+\gamma S(\theta)}4+\Gamma_\tau,
 \qquad \theta_\tau=\frac{\gamma Lx}4,
 \qquad S_i(\theta)=\sum_{k\sim i}\sin(\theta_k-\theta_i).
 \tag{1}
\]

Admit \(\Gamma_\tau\in L^1([0,T_\tau];\mathbb R^{36})\).
The sine currents are smooth and bounded on the fixed finite graph; the
form diffusion is linear. Solutions have absolutely continuous form, and
the phase row gives a continuously differentiable real lift because form is
continuous. The equations and storage balance hold almost everywhere.
On this compact interval the state is bounded, the loss is finite and the
input power is integrable. These regularity premises include continuous
inputs and measurable inputs with finite integral. They do not admit a
Dirac impulse or an intermediate state reset.

The obstruction does not restrict which nodes may be forced, the input's
amplitude or bandwidth, or whether it was obtained through open-loop design
or feedback. Every resulting input and trajectory in the stated class is
covered. Retaining the live form source is essential: it is not absorbed into
an unforced certificate or reconstructed as a pressure after observing motion.

For \(e=1023/1024\), the original clock obeys
\(\Gamma_t=e\Gamma_\tau\). In elapsed slow time \(\sigma\),
with \(d\sigma/d\tau=\gamma^2/2\), put
\(\Gamma_\sigma=2\Gamma_\tau/\gamma^2\). Both rows become

\[
 x_\sigma=\frac{-Lx+\gamma S(\theta)}{2\gamma^2}
                  +\Gamma_\sigma,
 \qquad \theta_\sigma=\frac{Lx}{2\gamma}.
 \tag{2}
\]

The local interval is \(0\le\sigma\le h\); its absolute slow-time
origin is the already reached \(s=4\). Input and loss integrals transform
with their clocks. These are structural clocks, without a laboratory-unit
identification.

## 2. Actual endpoint storage and input-work balance

Let \(k\in\mathbb R^{36}\) be the fixed lifted displacement:

\[
 k_{g,a,j}=\begin{cases}(2a-1)/4,&g=0,\\0,&g=1.\end{cases}
 \qquad \mathbf1^Tk=0,\qquad \|k\|_2^2=18/16=9/8.
 \tag{3}
\]

The target is exactly \((x_4,\theta_4+k)\). In particular, every
form must return to its actual starting value, and every unaffected lifted
phase must return to its original value. Evolving the hybrid endpoint freely
for one slow unit would give a different target and is not substituted here.

The complete storage and its gradients are

\[
 H=\tfrac12x^TLx+\sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)],
 \qquad \nabla_xH=Lx,\qquad \nabla_\theta H=-S(\theta).
\]

Substituting (1), the reciprocal terms cancel, leaving

\[
 H_\tau=-\tfrac14\|Lx\|_2^2+(Lx)^T\Gamma_\tau.
 \tag{4}
\]

Hence every admitted realization of the specified endpoint satisfies the
exact accounting identity

\[
 W_{\rm in}=\Delta H+\mathcal D,
 \qquad
 \Delta H=H(x_4,\theta_4+k)-H(x_4,\theta_4),
 \qquad
 \mathcal D=\tfrac14\int_0^{T_\tau}\|Lx\|_2^2d\tau.
 \tag{5}
\]

The endpoint term is the actual hybrid-map storage jump. It is independent
of the path chosen to realize that endpoint. The
[previous event-work proof](ORGANIZED_PARENT_RESTORING_RESPONSE.md#7-event-work-and-its-actual-state-correction)
rebuilds it from the original uncertain acquired source:

\[
 \begin{split}
 W_*={}&9[\cos B_4-\cos(B_4+1/2)]
        +18\cos\Delta_4[1-\cos(1/4)],\\
 |\Delta H-W_*|&\le18Z_{\rm pre},\qquad 1<W_*<225/128.
 \end{split}\tag{6}
\]

Here \(B_4,\Delta_4\) are the original reference's reached phase gaps,
and \(Z_{\rm pre}\) bounds the actual phase error at that time. The
signed linear correction is retained: acquired child-mean errors are not
reset to zero. Fresh source admission and these static work bounds suffice;
no verdict about the later restoring trajectory is required.

The input work in (5) is signed. If some part of an input extracts storage,
that contribution remains negative in the balance. Total positive supplied
work is at least signed work, so a bound on the former cannot bypass the
obstruction established for the latter.

## 3. A necessary loss for the prescribed phase transport

The unchanged phase row fixes \(Lx=4\theta_\tau/\gamma\).
Substitution into the loss, followed by the integral Cauchy-Schwarz inequality,
gives

\[
 \begin{split}
 \mathcal D
 &=\frac4{\gamma^2}\int_0^{T_\tau}\|\theta_\tau\|_2^2d\tau\\
 &\ge\frac4{\gamma^2T_\tau}
       \left\|\int_0^{T_\tau}\theta_\tau d\tau\right\|_2^2
 =\frac4{\gamma^2T_\tau}\|k\|_2^2
 =\frac9{4h}.
 \end{split}\tag{7}
\]

Equivalently, in the slow clock, (2) gives
\(\mathcal D=2\int_0^h\|\theta_\sigma\|_2^2d\sigma\).
The same inequality is then \(\mathcal D\ge2\|k\|_2^2/h\).
The cancellation of \(\gamma\) reflects the prescribed slow duration;
it does not identify a physical constant or eliminate clock dependence.

This bound is global. It assumes no small-angle approximation, invariant
family, monotone path, controlled trajectory, or intermediate child identity.
Even paths that temporarily lose all the organizational identities must pay
this loss if they satisfy the phase law and the prescribed endpoints. Thus
proving the impossibility does not first require a retention certificate for
every candidate path.

Both common modes are harmless to the bound. The lifted phase mean is
conserved because \(\mathbf1^TL=0\), and the displacement (3) respects
that requirement. Arbitrary common form motion or a common component of the
input does not change \(Lx\), phase velocity or the loss. Returning every
form also requires the net common form input to vanish, but imposing that
additional necessary condition cannot lower (7).

## 4. Decision for the fixed one-unit realization

Combining the exact endpoint work and the necessary loss yields

\[
 W_{\rm in}\ge\Delta H+\frac9{4h}.
 \tag{8}
\]

For the frozen \(h=1\), (6) alone already gives

\[
 W_{\rm in}>\frac{13}{4}-18Z_{\rm pre}>3.
 \tag{9}
\]

The last margin is \(1/4-18Z_{\rm pre}>0\) for the complete original
source. The detached certificate may use the sharper outward lower endpoint
of (6) plus \(9/4\), while preserving the same decision and budget.
Consequently no admitted form input can realize the fixed actual endpoint
within one slow unit and three structural work units. This is an obstruction
for the entire declared input class and original uncertain family, not a failed
controller, a solver failure or the absence of one attempted path.

There is therefore no need to construct a forcing or evolve a driven path
for this gate. An unavailable original-source or endpoint-work admission
would instead leave the decision unavailable; cached status labels alone are
not evidence for (9).

## 5. Equality, alternative lifts and limits of the conclusion

Equality in (7) requires a constant lifted phase velocity almost everywhere,
\(\theta_\tau=k/T_\tau\). Since \(x\) is continuous and the phase
row is unchanged, this also requires
\(Lx(0)=4k/(\gamma T_\tau)\), with the same condition at the final
endpoint. It is not enough to draw a straight phase path while choosing
unrelated endpoint forms.

For the acquired source, equality is impossible. Define the mixed form
projection
\(m(x)=\frac1{36}\sum_{g,a,j}(2g-1)(2a-1)x_{g,a,j}\).
Its graph mode has Laplacian eigenvalue four. The acquired reference has
\(m(x^*)=0\), and its actual fine form deviation gives
\(|m(x_4)|\le R_{\rm pre}/6\). In contrast, the constant-velocity
condition at \(h=1\) would require

\[
 m(k)=-1/8,\qquad m(x_4)=-\gamma/16,
 \tag{10}
\]

whose magnitude exceeds that source-error allowance. This nonattainment does
not establish an additional uniform positive gap above (7). No amplitude or
bandwidth bound has been imposed, and no sharp infimum or optimal input has
been constructed. The necessary bound, without such an optimization claim,
already resolves the fixed budget.

The protocol specifies continuous lifts. If one separately weakens only the
endpoint phase condition to equality modulo \(2\pi\) at every marked
node, each continuous path instead has displacement \(k+2\pi n\) for
an integer vector \(n\). Since every \(|k_i|\le1/4<\pi\),

\[
 \|k+2\pi n\|_2^2\ge\|k\|_2^2.
 \tag{11}
\]

The conserved lifted mean additionally requires \(\sum_i n_i=0\).
The periodic storage endpoint is unchanged. Thus extra full turns cannot
reduce this particular work-duration lower bound. This argument concerns
the same complete nodewise circular endpoint; targeting only a winding,
parent mean, approximate endpoint or another observation is a different
problem and inherits no such conclusion without its own admission.

The obstruction leaves the hybrid map and its restoring-response theorem
valid under their original premises. It also leaves open a longer duration,
a different work budget, an admitted phase input or another complete law.
Passing a necessary bound in one of those alternatives would not certify a
realizing path, retained child identity or autonomous event occurrence.

## 6. Evidence and ownership

The [detached assessor](../../src/tnfr/physics/_organized_parent_continuous_actuation.py)
rebuilds the eight primitive source allowances and
the actual acquired-state evidence before computing the fixed displacement,
endpoint-work enclosure and loss lower bound. The shared signed event-work
calculation retains the source's linear uncertainty correction. No incoming
restoring-response verdict, force callback, numerical path or frozen response
is needed to establish the obstruction.

The [independent controls](../../tests/physics/test_organized_parent_continuous_actuation.py)
check the forced storage identity, clock transformations,
the exact displacement norm, source association, signed work correction and
strict rejection margin. They validate the implementation of the conditional
theorem; a finite collection of force samples would not prove its universal
input quantifier. Structural work and duration retain their declared units,
without a sensor bridge, material interpretation or uniquely selected law.

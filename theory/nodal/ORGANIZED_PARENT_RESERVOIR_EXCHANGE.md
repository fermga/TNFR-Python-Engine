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

The declaration above was committed as `9a0071de` before this analytic
assessment. The fixed connected branch does not meet its registration-loss
criterion: both parents still have every same-slot gap below the threshold
at the reserved half-unit readout. The proof below also admits the finite
reservoir balance, retained children and registered deadline. It does not
infer absence of a disruption at every other time.

## 1. Complete joint balance, clocks and reference restriction

The original structural clock is absolute: the new comparison starts at
\(\tau_0=8/\gamma^2\), corresponding to \(s=4\). A subscript zero below
denotes this initial state, and elapsed slow time is
\(h=\gamma^2(\tau-\tau_0)/2=s-4\).
Write \(q=Lx\). The autonomous equations are globally well posed on finite
intervals: the sine map is smooth and globally Lipschitz, and every other
term is linear in the declared state. The global form mean and lifted phase
mean are conserved because \(\mathbf1^Tb=0\). No auxiliary phase is needed.
With unit auxiliary capacity in the normalized clock, its independently
supplied pressure is \(p_r=-b^TLx-\lambda r\).

The complete storage derivative is

\[
 \begin{split}
 H_\tau&=-\tfrac14\|Lx\|_2^2+(Lx)^Tbr,\\
 (r^2/2)_\tau&=-r b^TLx-\lambda r^2,\\
 (H+r^2/2)_\tau&=-\tfrac14\|Lx\|_2^2-\lambda r^2.
 \end{split}\tag{1}
\]

Thus the network's signed net input work through any finite time is

\[
 W_{\rm net}
 =\frac{r_0^2-r(\tau)^2}{2}-\lambda\int_{\tau_0}^\tau r(u)^2du
 \le\frac{r_0^2}{2}
 \le\frac{(12/5+10^{-10})^2}{2}<3.
 \tag{2}
\]

The network's initial acquired storage remains part of the joint preparation.
It is not included in the three-unit added-reservoir budget. Equation (2)
allows the network to charge the reservoir; it is not a bound on cumulative
positive-only supply. The finite reservoir is an actual coordinate of this
supplied model, not a reinterpretation of previously dissipated storage.

Every row transforms under \(\tau=e t\), \(e=1023/1024\), by
multiplication by \(e\). Under elapsed slow time
\(h=\gamma^2(\tau-\tau_0)/2\),
the complete equations become

\[
 x_h=\frac{-Lx+\gamma S(\theta)}{2\gamma^2}
            +\frac{2b}{\gamma^2}r,\qquad
 \theta_h=\frac{Lx}{2\gamma},\qquad
 r_h=-\frac{2b^TLx}{\gamma^2}-r.
 \tag{3}
\]

These are structural clocks. The rate units of \(b,\lambda\), pressure and
storage are those fixed in the declaration; no laboratory clock is inferred.

The exact child-constant restriction retains all four forms \(X_c\), four
lifted offsets \(\Phi_c\), and \(r\). Let \(\mathcal L\) be the square
Laplacian, \(\beta_c=\gamma\sigma_a/12\), and
\(\mathcal S_c=\sum_{d\sim c}\sin(\Phi_d-\Phi_c)\). Then

\[
 X_\tau=\frac{-\mathcal LX+\gamma\mathcal S(\Phi)}4+\beta r,\qquad
 \Phi_\tau=\frac{\gamma\mathcal LX}4,\qquad
 r_\tau=-9\beta^T\mathcal LX-\lambda r.
 \tag{4}
\]

The factor nine in the reservoir row counts the fine nodes of each child;
the network denominator remains four. Internal C9 sine currents cancel,
and matching contacts preserve the winding carrier. Both facts hold for
arbitrary child offsets in this invariant family.

For the acquired reference write

\[
 x^*_{g,a,j}=\frac{\sigma_g U+\sigma_a V}{2},\qquad
 \theta^*_{g,a,j}=\psi_j+\frac{\sigma_g\Delta+\sigma_a B}{2}.
 \tag{5}
\]

The auxiliary input preserves this separated subfamily. With
\(\varepsilon=\gamma^2\), \(u=U/\gamma\) and \(v=V/\gamma\), its exact
slow equations are

\[
 \begin{gathered}
 \Delta_h=u,\qquad \varepsilon u_h=-u-\sin\Delta,\\
 B_h=v,\qquad \varepsilon v_h=-v-\sin B+r/3,\qquad
 r_h=-6v-r.
 \end{gathered}\tag{6}
\]

In particular \(b^TLx^*=3\gamma V\). All acquired reference velocities
are retained. At the new initial time, the acquisition proof gives
\(0<B_0<1/64\) and
\(-(1+2\varepsilon)B_0\le v_0\le0\); neither coordinate is reset.
The reference reservoir starts at the particular actual \(r_0\) in the
frozen interval. Uniform bounds below retain that entire interval.

The reference storage, apart from its constant internal-cycle contribution,
is \(9\varepsilon(u^2+v^2)+18(1-\cos\Delta)+18(1-\cos B)+r^2/2\).
Its slow derivative is \(-18(u^2+v^2)-r^2\). These balances check the
coefficients in (6); they alone do not establish the reserved response.
Replacing the middle equation by \(v=-\sin B+r/3\) would instead be a
singular reduced model, and is not used in the proof.

## 2. A new joint-state error estimate

Let \(R,Z<10^{-60}\) be the freshly rebuilt acquisition's full form and
phase error bounds at absolute slow time four. They retain the original
64 fine source directions. Compare each actual connected state with the
reference (5) initialized with its same actual \(r_0\). Write the form,
phase-lift and reservoir differences as \(\xi,\zeta,\rho\). Initially
\(\|\xi_0\|_2\le R\), \(\|\zeta_0\|_2\le Z\), \(\rho_0=0\).
The independent reservoir uncertainty has not been replaced by its midpoint
or folded into a smaller fine-state allowance.

The deviations satisfy

\[
 \xi_\tau=-L\xi/4+\gamma\Delta S/4+b\rho,\qquad
 \zeta_\tau=\gamma L\xi/4,\qquad
 \rho_\tau=-b^TL\xi-\lambda\rho,
 \quad \Delta S=S(\theta)-S(\theta^*).
 \tag{7}
\]

The two network inputs differ whenever the reservoir errors differ. Thus the
previous identical-input theorem cannot be invoked. Instead use the positive
comparison quantity

\[
 \mathcal V=\tfrac12\xi^TL\xi+\tfrac12\rho^2+4\|\zeta\|_2^2.
 \tag{8}
\]

It is an error estimate, not an additional physical reservoir. The global
form-error mean stays zero because the coupling vector is mean zero. The
fixed graph has \(\lambda_2(L)>1/3\), \(\|L\|_2\le8\), and
\(\|\Delta S\|_2\le8\|\zeta\|_2\), globally in the lifted phases.
The reciprocal reservoir terms cancel exactly:

\[
 \begin{split}
 \mathcal V_\tau
 &=-\tfrac14\|L\xi\|_2^2-\lambda\rho^2
   +\tfrac\gamma4(L\xi)^T(\Delta S+8\zeta)\\
 &\le\frac{\gamma^2}{16}\|\Delta S+8\zeta\|_2^2
 \le16\gamma^2\|\zeta\|_2^2
 \le4\gamma^2\mathcal V.
 \end{split}\tag{9}
\]

Since \(\mathcal V_0\le4(R^2+Z^2)\), slow time gives
\(\mathcal V(h)\le\mathcal V_0 e^{8h}\). For either horizon
\(H\in\{1/2,8\}\), the convenient rational bounds are

\[
 Z_H=(R+Z)3^{4H},\qquad R_H=6Z_H,\qquad Q_H=4Z_H,
 \tag{10}
\]

and throughout that interval
\(\|\zeta\|_2\le Z_H\), \(\|\xi\|_2\le R_H\),
\(|\rho|\le Q_H\). Here \(e<3\), the graph gap controls the mean-zero
form, and the integer constants conservatively dominate the square-root
coefficients from (8). In particular \(Z_{1/2}<2\cdot10^{-59}\) and
\(Z_8<4\cdot10^{-45}\).

The same estimate applies to the disconnected branch with \(b=0\). Its
reference and actual reservoir both decay as \(r_0e^{-h}\), while the
network follows the original unforced law. No assertion that a connected
reservoir initialized at zero stays zero is required.

## 3. Cooperative bounds for the fixed connected response

The following comparison treats the finite \(\varepsilon\) equations
directly. Define

\[
 p=r+6B,\qquad y=v+4B.
 \tag{11}
\]

Then (6) is equivalent to

\[
 B_h=y-4B,\qquad
 \varepsilon y_h=-(1-4\varepsilon)y
                  +(2-16\varepsilon)B-\sin B+p/3,\qquad
 p_h=6B-p.
 \tag{12}
\]

Its off-diagonal derivatives are nonnegative: in particular the derivative
of the second numerator in \(B\) is at least \(1-16\varepsilon>0\).
The nonnegative orthant is invariant, using \(\sin B\le B\) for \(B\ge0\).
The initial state lies there because
\(y_0\ge[4-(1+2\varepsilon)]B_0>0\) and
\(p_0=r_0+6B_0>0\). Standard componentwise differential comparison
therefore applies; no prior assumption that \(r\) stays positive is needed.

Put

\[
 d=1/64,\qquad c=9/10,\qquad r_+=12/5+10^{-10},
 \qquad
 \begin{split}
 \bar B(h)&=d+\tfrac14(1-e^{-4h}),\\
 \bar y(h)&=1+4d,\\
 \bar p(h)&=6d+\tfrac32+(r_+-2)e^{-h}+\tfrac12e^{-4h}.
 \end{split}\tag{13}
\]

These coordinates dominate \((B_0,y_0,p_0)\). The first and third
equations in (12) hold exactly. For the second, the required residual is

\[
 \begin{split}
 &\varepsilon\bar B_{hh}+\bar B_h+\sin\bar B+2\bar B-\bar p/3\\
 &\quad=\sin\bar B+(1/3-4\varepsilon)e^{-4h}
                         -(r_+-2)e^{-h}/3\\
 &\quad\ge cd+c/4+(1/3-c/4-4\varepsilon)e^{-4h}
                         -(r_+-2)e^{-h}/3>0.
 \end{split}\tag{14}
\]

Indeed \(0<\bar B\le17/64\), and
\(1-(17/64)^2/6>c\) gives \(\sin\bar B\ge c\bar B\).
Both \(1/3-c/4-4\varepsilon\) and
\(cd+c/4-(r_+-2)/3\) are strictly positive for the unchanged
\(\varepsilon<10^{-7}\). Thus \(0\le B(h)\le\bar B(h)\le17/64\).

At the frozen half-unit readout, \(e^2<8\) gives the useful strict margin

\[
 B(1/2)\le\frac1{64}+\frac14(1-e^{-2})<\frac{15}{64}.
 \tag{15}
\]

This evaluates the declared candidate, without a parameter scan or a
replacement of its deadline. It is a genuine upper bound below the desired
disruption threshold, not the failure of a lower-bound method.

For the later registered deadline, the preceding global bound admits
\(\sin B\ge cB\) along the actual reference. Replace the second row's
sine term by that lower linear bound. The resulting linear system is again
cooperative. A supersolution is

\[
 \widehat B(h)=\tfrac13e^{-h/5},\qquad
 \widehat y(h)=(4-1/5)\widehat B(h),\qquad
 \widehat p(h)=\tfrac52e^{-h/5}.
 \tag{16}
\]

Its initial values dominate the reference, in particular
\(p_0<r_++6/64<5/2\). Its first and third rows are exact, and its
second-row residual is
\((\varepsilon/25+1/5)\widehat B>0\). Therefore

\[
 0\le B(8)\le\tfrac13e^{-8/5}<\frac1{12},
 \tag{17}
\]

where \(e^{8/5}>4\) follows already from its first four Taylor terms.
No small-\(\varepsilon\) replacement of the acquired velocity was used in
either comparison.

## 4. Actual identities, control and decision

In the reference, both parents have the same internal child-registration
gap \(B\). The actual deviation of any same-slot gap is at most
\(2Z_H\). At the reserved readout, (15) and the new joint estimate give

\[
 |\theta_{g,1,j}-\theta_{g,0,j}|_{\rm circ}
 \le\frac{15}{64}+2Z_{1/2}<\frac14
 \quad\text{for every }g,j.
 \tag{18}
\]

The displayed bound is within the principal circular branch. Both
connected parents therefore retain registration at the time when the frozen
criterion requires every one of their gaps to exceed \(1/4\). The complete
original source and the full reservoir interval are covered. This excludes
the specified registration-loss discrimination for this fixed candidate.

For the disconnected control, the original acquired reference's unforced
cone gives \(0<B(h)\le B_0<1/64\). Applying (10) shows its actual parents
retain registration at the same readout, and also at the deadline. This is
the declared \(b=0\) control with the same preparation; it is not a claim
about a connected zero-initial-reservoir preparation.

Equation (17) gives, for the connected branch at the deadline,

\[
 |\theta_{g,1,j}-\theta_{g,0,j}|_{\rm circ}
 \le\frac1{12}+2Z_8<\frac14.
 \tag{19}
\]

This establishes registration at that deadline. Since the prescribed loss
criterion failed, it is not reported as a demonstrated loss-and-return cycle.
The proof has not excluded a transient threshold crossing at every other
time between the two readouts.

In either reference branch, each child remains a winding carrier plus a
uniform offset under its exact inherited dynamics. The full error bound
confines every actual oriented internal edge gap, including the closing
edge with its retained integer lift, to

\[
 [\alpha-2Z_8,\alpha+2Z_8]\subset(0,\pi/2),\qquad \alpha=2\pi/9.
 \tag{20}
\]

Their sum stays \(2\pi\), proving winding one for all four original children
through the full eight-unit elapsed interval in both branches.

The result is therefore a source-inclusive negative decision for the frozen
registration-loss readout, with an admitted autonomous balance, finite net
reservoir work, child retention and registered deadline. No source allowance,
coupling coefficient, preparation or readout has been moved. A different
port, reservoir, horizon or complete law is a distinct question and is not
excluded by this result.

## 5. Evidence, reuse and scientific scope

The [detached implementation](../../src/tnfr/physics/_organized_parent_reservoir_exchange.py)
admits all eight network-source allowances and
the reservoir allowance before rebuilding the original source and assessing
the strict response margins. The
joint comparison treats the two reservoir responses as part of the state;
an old identical-input verdict is not reused. The scalar response bounds
are prospective consequences of the fixed law, not sampled trajectories.
An unavailable source or sufficient margin is reported separately from the
proved negative decision.

The [independent controls](../../tests/physics/test_organized_parent_reservoir_exchange.py)
check the full and inherited rows, factor-nine
reservoir reaction, all clock conversions, storage power, finite reservoir
budget, joint-error cancellation, comparison supersolutions and circular
identity margins. The original full fine uncertainty and independent
reservoir interval remain explicit throughout.

The auxiliary scalar, its pressure and storage, the reciprocal coupling and
their joint preparation remain supplied premises. A successful balance does
not derive their occurrence or identify the scalar as a formed NFR. Conversely
failure of this fixed response does not invalidate the existing organized
objects, supplied-input constructions or every autonomous source mechanism.
No unique fundamental law, material property or physical measurement bridge
has been selected.

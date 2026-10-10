# Restoring response of an acquired organized parent

<a id="organized-parent-restoring-response"></a>

## Fixed intervention and analytic question

This declaration continues the actual source family of
[parent acquisition](ORGANIZED_PARENT_ACQUISITION.md) at its slow time `s=4`.
It fixes one supplied intervention and its paired comparison before evaluating
the analytic response bounds. It does not evaluate a numerical trajectory.

| Item | Fixed choice |
| --- | --- |
| Source | The four original independently uncertain recovered C9 children, each with form and phase allowances `1e-10`, carried through the unchanged acquisition law to `s=4`; retain all original fine errors and reached forms and phases |
| Complete continuous law | The matching 36-node law, held unit capacities, all 72 unit edges and the original structural clocks `tau=(1023/1024)*t`, `s=gamma^2*tau/2`, `gamma=1/(1023*pi)` |
| Supplied event | At `s=4`, simultaneously set `theta+[0,a,j]=theta-[0,a,j]+(2*a-1)/4` for every slot of parent zero; leave all other phases, all forms, support and capacities unchanged |
| Event interpretation | A specified zero-duration hybrid phase intervention with marked targets and no autonomous occurrence law; no registered operator name is claimed |
| Preserved observations | Each child's winding and both parents' form and lifted-phase means at the event; retain the same continuous phase lifts |
| Work | Account for the exact complete storage jump `H+ - H-`, supplied externally with a fixed budget of three structural storage units; previous dissipation is not a reservoir |
| Paired control | The same actual acquired state continued without the event, under the same complete continuous law and clocks |
| Finite response readout | At `s=5`, the intervened outer parent lifted-phase gap minus the paired control gap must exceed `1e-6` for the entire original source family |
| Predictive state | Initialize the exact eight-child-mean law from the actual post-event child means; its outer parent phase-gap error at `s=5` must be below `1e-7`. Compare with four parent means, which are identical across the paired preparations at the event |
| Recovery deadline | By `s=8`, restore both marked parent identities: every intraparent same-slot gap has magnitude strictly below `1/4` |
| Child retention | Every marked C9 keeps winding one throughout `0<=s<=8`, including the event |
| Decision | Admit source-inclusive disruption, finite recovery, accounted work and a resolved predictive distinction, or an explicit scoped obstruction; a failed sufficient estimate remains unavailable |

The half-radian relative displacement deliberately exceeds the registration
threshold while preserving each child internally. One and four subsequent
slow units are fixed readout and recovery horizons, without a scan or source
refinement. The eight-coordinate prediction retains its supplied law and
hidden initialization. The four-coordinate comparison concerns sufficiency
of that state alone; adding an intervention history supplies extra information.
Neither the fixed grouping, event occurrence, work supply nor continuous law
is selected autonomously. This question does not identify physical binding,
particles, a laboratory clock or a unique hierarchy.

The declaration above was committed as `37add507` before its analytic
certificate was assessed. The proof below retains those choices and the
complete original uncertainty. No fine or reduced trajectory is evaluated.

## 1. The actual event and the retained continuous law

Write \(\sigma_g=2g-1\), \(\sigma_a=2a-1\),
\(\alpha=2\pi/9\) and \(\psi_j=\alpha(j-4)\). The matching graph is
\(C_9\square K_2\square K_2\), with its original marked nodes and
unnormalized Laplacian \(L\). Between events the complete rows remain

\[
 x_\tau=(-Lx+\gamma S(\theta))/4,\qquad
 \theta_\tau=\gamma Lx/4,\qquad
 S_i(\theta)=\sum_{k\sim i}\sin(\theta_k-\theta_i).
 \tag{1}
\]

The supplied jump is the simultaneous map \(T\) declared in the table:
add \(-1/4\) to every phase of child \((0,0)\), add \(1/4\) to
every phase of child \((0,1)\), and preserve the other state. It acts on
the actual pre-event state, not on its nominal target. Uniform translation
within a child preserves every internal phase difference and its winding.
The opposite translations preserve each parent's lifted phase mean; all
form means are unchanged. Retain the existing lifts through the jump.

This defines a detached hybrid intervention with explicit fine realization,
target marks and externally supplied work. It does not claim registered
Mutation, Self-organization or another runtime operator: their live evidence,
grammar, descent and occurrence obligations are separate. No finite-duration
pressure input is invented for the jump. The subsequent sine current is
evaluated from the post-event phases under (1).

## 2. The exact four-child reference after intervention

The original moving reference reaches its own values
\(B_4,\Delta_4,v_4,u_4\) at \(s=4\). These are the acquired
values from the previous theorem; none is reset. After applying the same
jump to that reference, the eight-coordinate child-constant family remains
exactly invariant under (1). Its zero-global-mean relative chart is

\[
 \begin{split}
 x^*_{g,a,j}&=\tfrac12(\sigma_g U+\sigma_a V+
                                      \sigma_g\sigma_a W),\\
 \theta^*_{g,a,j}&=\psi_j+\tfrac12(\sigma_g D+\sigma_a B+
                                      \sigma_g\sigma_a C).
 \end{split}\tag{2}
\]

The conserved common form and phase origins can be added independently.
Here \(D\) is the outer parent phase gap, the two intraparent child gaps
are \(B-C\) and \(B+C\), and \(C\) records their difference and
the associated outer-port mismatch. Direct substitution into both fine rows
gives

\[
 \begin{array}{ll}
 U_\tau=-U/2-\gamma\sin D\cos C/2,&D_\tau=\gamma U/2,\\
 V_\tau=-V/2-\gamma\sin B\cos C/2,&B_\tau=\gamma V/2,\\
 W_\tau=-W-\gamma(\cos B+\cos D)\sin C/2,&C_\tau=\gamma W.
 \end{array}\tag{3}
\]

The internal cycle currents cancel. The three graph modes have Laplacian
eigenvalues two, two and four, respectively; omitting the last mode would
change both the form and phase dynamics after this intervention.

Use elapsed slow time \(h=s-4\), \(\varepsilon=\gamma^2\), and

\[
 K=-C,\qquad y_D=-U/\gamma,\qquad y_B=-V/\gamma,
 \qquad z=2W/\gamma.
\]

Both rows transform, yielding

\[
 \begin{array}{ll}
 D_h=-y_D,&\varepsilon(y_D)_h=\sin D\cos K-y_D,\\
 B_h=-y_B,&\varepsilon(y_B)_h=\sin B\cos K-y_B,\\
 K_h=-z,&\varepsilon z_h=2[(\cos B+\cos D)\sin K-z].
 \end{array}\tag{4}
\]

The post-event reference initialization is exactly

\[
 D(0)=\Delta_4,\quad B(0)=B_4+1/4,\quad K(0)=1/4,
 \quad y_D(0)=-u_4/\gamma,\quad y_B(0)=-v_4/\gamma,
 \quad z(0)=0.
 \tag{5}
\]

The first two velocities are the nonzero acquired form coordinates. Setting
them to zero or to an assumed equilibrium current would be a different
initial-value problem. The no-event reference instead continues the original
separated family, with \(C=W=0\).

## 3. Full source transport and retained children

Rebuild the four original isolated sources and their acquired-state bounds
as in [parent acquisition](ORGANIZED_PARENT_ACQUISITION.md). Let
\(q=10^{-70}\) be the unchanged recovered norm allowance per child and
per channel. The global initial deviations from the moving reference have
norms at most \(2q\). All 64 independent fine source directions remain
admitted; no smaller preparation, later dwell or symmetry is imposed.

Apply the same jump to the actual and reference phases. Their differences
\(r=x-x^*\), \(\zeta=\theta-\theta^*\) have no jump. The support
and diffusion operator remain fixed. Consequently the exact integrated heat
identity from the acquisition proof applies piecewise across the event,
with the same global constants

\[
 \lambda_2(L)>1/3,\qquad \|L\|_2\le8,\qquad
 \|S(\theta)-S(\theta^*)\|_2\le8\|\zeta\|_2.
\]

It gives, on either paired branch throughout \(0\le s\le8\),

\[
 Z_8=2q(1+\bar\gamma)3^{32},\qquad
 R_8=2q+24\bar\gamma Z_8,\qquad
 \|\zeta\|_2\le Z_8,\quad\|r\|_2\le R_8.
 \tag{6}
\]

Here \(\bar\gamma\) is an outward upper bound, and
\(2\gamma^2\tau=4s\le32\). The form deviation stays globally
mean zero. This is the original-source comparison with a moving exact
reference, not a new source at the event and not a local energy trap.

For every marked C9 the reference's internal edge branches remain
\(\alpha\). Actual branches therefore lie in
\([\alpha-2Z_8,\alpha+2Z_8]\subset(0,\pi/2)\); their residual
differences telescope around the cycle. The branch integer on the closing
edge is retained, giving winding one throughout the declared interval.
The uniform event translations preserve that winding exactly at the jump.

## 4. Disruption and a finite restoring response

Put \(a=1+2\varepsilon\). The earlier source theorem supplies

\[
 1/1200<\Delta_4<1/500,\qquad 0<B_4<1/64,
 \quad 0\le-u_4/\gamma\le a\Delta_4,
 \quad 0\le-v_4/\gamma\le aB_4.
\]

In the notation of (5), this says directly
\(0\le y_D(0)\le a\Delta_4\) and
\(0\le y_B(0)\le aB_4<aB(0)\). Also
\(B(0)<4/15\), \(K(0)=1/4\), and \(z(0)=0\).

For \(0<\varepsilon<10^{-7}\), the domain

\[
 \begin{split}
 &0<D\le D(0),\quad 0<B\le B(0),\quad 0<K\le1/4,\\
 &0\le y_D\le aD,\qquad 0\le y_B\le aB,
                  \qquad 0\le z\le2aK
 \end{split}\tag{7}
\]

is invariant. At either \(y=aD\) boundary, the scaled outward derivative
is at most \((1-a+\varepsilon a^2)D<0\); replacing the forcing
by \(\sin D\cos K\le D\) only strengthens that bound. At
\(z=2aK\), it is at most
\(4(1-a+\varepsilon a^2)K<0\). Lower velocity boundaries have
nonnegative derivatives. The cones imply positive exponential lower bounds
for all three gaps, precluding a finite-time exit through zero.

Thus the three positive gaps decrease. Elementary sine/cosine bounds on
the above rectangular domain give

\[
 \sin B\cos K\ge(19/20)B,\qquad
 (\cos B+\cos D)\sin K\ge(19/10)K.
 \tag{8}
\]

For example, use \(\sin B/B\ge1-B^2/6\),
\(\cos K\ge1-K^2/2\), and the corresponding bounds for
\(B<4/15\), \(D<1/500\), \(K\le1/4\).
The variation-of-constants formulas for the velocities, their nonnegative
initial values and monotonicity of \(B,K\) then imply

\[
 B(h)\le(B_4+1/4)e^{-(19/20)(h-\varepsilon)},\qquad
 K(h)\le\tfrac14e^{-(19/10)(h-\varepsilon/2)}.
 \tag{9}
\]

Define \(B_+,K_+\) by these right sides at \(h=4\), with outward
source/coefficient bounds. The fixed inequalities give
\(B_++K_+<1/50\). At the event, every parent-zero same-slot gap is
at least \(B_4+1/2-2Z_{\rm pre}>1/4\), so the entire actual family
leaves that parent's registration domain. Here \(Z_{\rm pre}\) is the
previous acquisition phase-error bound, valid at \(s=4\).

At the fixed recovery deadline, let \(X_g\) be the actual parent form
mean and \(\Phi_g\) the mean of \(\theta_{g,a,j}-\psi_j\).
The latter equals the raw lifted phase mean because the winding carrier has
zero mean. Define the phase residual using those offsets, not by subtracting
only the parent mean from the primitive winding phase:

\[
 r_{\theta,g}=\max_{a,j}|\theta_{g,a,j}-\psi_j-\Phi_g|,
 \qquad r_{x,g}=\max_{a,j}|x_{g,a,j}-X_g|.
\]

Both actual parents then satisfy

\[
 \begin{split}
 \max_j|\theta_{g,1,j}-\theta_{g,0,j}|
     &\le B_++K_++2Z_8<1/4,\\
 r_{\theta,g}&\le(B_++K_+)/2+2Z_8<1/100,\\
 r_{x,g}&\le\bar\gamma(1+2\bar\varepsilon)(B_++K_+)/2
                         +2R_8<1/100.
 \end{split}\tag{10}
\]

The last two residuals are taken about each actual parent mean. They use
\(|B\pm K|\le B+K\) and
\(|V\pm W|\le\gamma a(B+K)\), together with the full-source
deviations. Thus the parents return even to the stronger centiradian domain,
with all children retained. This proves a finite return by the supplied
deadline, not indefinite post-return stability or physical binding.

## 5. A finite interaction that four parent coordinates cannot determine

Let \(d=\Delta_4\). In the intervened reference, the hidden mode \(K\)
reduces the external restoring current to \(\sin D\cos K\). The
invariant cone gives \(K(h)\ge(1/4)e^{-2ah}\). Therefore the missing
unit-time restoring factor obeys

\[
 I_K:=\int_0^1[1-\cos K(h)]dh
 \ge\frac{1-e^{-4a}}{192a}>1/200.
 \tag{11}
\]

This uses \(1-\cos K\ge K^2/3\) on \([0,1/4]\).
One simple rational lower bound for implementation replaces the numerator
by \(1-e^{-4}\) and the denominator by \(192\bar a\).

Write \(G(t,d)=2\arctan(e^{-t}\tan(d/2))\) for the solution of
\(G_t=-\sin G\). The gradient approximation with the actual reference
history \(K(h)\) ends at \(G(1-I_K,d)\); the no-event gradient
approximation ends at \(G(1,d)\). Since \(G\) decreases and
\(G(1,d)\ge d/e\),

\[
 \begin{split}
 G(1-I_K,d)-G(1,d)
 &\ge I_K\sin G(1,d)\\
 &>\frac1{200}\frac3{10000}
   =\frac3{2\cdot10^6}.
 \end{split}\tag{12}
\]

Indeed \(\sin G(1,d)>(99/100)d/e>3/10000\), using
\(d>1/1200\), \(d<1/500\) and \(e<11/4\).
This is a finite response bound, not the already known instantaneous
port-current difference.

The gradient approximations are controlled without replacing either form
row. For the intervened reference, let
\(f=\sin D\cos K\) and \(\ell=y_D-f\). Then

\[
 \ell_h=-\ell/\varepsilon-f_h,\qquad
 |\ell(0)|<1/400,\qquad
 |f_h|\le a d(1+2K(0)^2)<1/400.
 \tag{13}
\]

The bounds follow directly from (7), with the actual acquired initial form
coordinate in (5). Exponential convolution gives
\(\int_0^1|\ell|dh<\varepsilon/200\). The time-dependent gradient
row \(-\sin D\cos K(h)\) is nonexpansive on the positive small-angle
domain, so this integral bounds its endpoint error. The no-event reference
retains the previous lag error at most \(\varepsilon/10\). A common
safe bound for their sum is \(\varepsilon/5\).

Each actual branch differs from its corresponding reference parent phase
gap by at most \(2Z_8\). Combining this with (12)-(13) proves the
prospectively selected contrast for every paired original source:

\[
 D_{\rm actual,kick}(5)-D_{\rm actual,control}(5)
 >\frac3{2\cdot10^6}-\frac{\bar\varepsilon}{5}-4Z_8
 >10^{-6}.
 \tag{14}
\]

Both actual preparations have exactly the same four parent means immediately
after the event. The induced event map on those four observations is the
identity. Their later responses nevertheless differ by (14), so those four
coordinates alone do not determine this finite response. An intervention
label, retained child association or memory would supply additional state;
the result does not exclude such an augmented predictor.
In particular, one forecast based only on the common four-coordinate state
must miss at least one paired endpoint by more than \(5\cdot10^{-7}\),
which exceeds the fixed \(10^{-7}\) prediction budget.

## 6. Eight-coordinate prediction from actual post-event observations

Initialize the exact four-child law from all eight actual child means just
after the event, including their reached forms. In full child coordinates
that law is

\[
 X_h=\frac{-\bar L X+\gamma\bar S(\phi)}{2\gamma^2},
 \qquad \phi_h=\frac{\bar L X}{2\gamma},
 \tag{15}
\]

where \(\bar L\) is the unit square Laplacian and
\(\bar S\) sums sine currents over its two neighbors. This transforms
both rows of the existing eight-coordinate restriction; it installs no
additional law and makes no exact-quotient claim for arbitrary fine states.

The averaging projection onto child means is orthogonal after constant
lifting. For a 36-vector of fine errors, its four mean errors therefore have
Euclidean norm at most one third of the fine norm. Also
\(\|\bar L\|_2=4\), so the same integrated-heat argument amplifies
the predictor's initial phase/form discrepancy by at most \(e^{2h}\).
The parent gap is the child-mean linear observation with coefficients
\((-1/2,-1/2,1/2,1/2)\), of norm one. Consequently, at the readout,

\[
 |D_{\rm actual}-D_{\rm eight}|
 \le\frac{Z_8+e^2(Z_{\rm pre}+\bar\gamma R_{\rm pre})}{3}
 <2Z_8+18(Z_8+\bar\gamma R_8)<10^{-7}.
 \tag{16}
\]

The first term is the actual fine-source error about the moving reference;
the second carries the eight-coordinate predictor's true initialization.
Using the looser shared bound in the last expression keeps every original
fine direction and the fixed resolution. These correlations come from the
carried source and event history. A generic parent residual ball or a tangent
restriction alone would not justify this slow-time prediction.

## 7. Event work and its actual-state correction

With the complete storage from the acquisition theorem, the exact jump is

\[
 W_T=H(T(x,\theta))-H(x,\theta)
   =\sum_{\text{affected }\{i,k\}}
       [\cos(\theta_k-\theta_i)-\cos(\theta_k-\theta_i+\delta_{ik})].
 \tag{17}
\]

All forms, internal C9 phase differences and support remain unchanged.
There is no additional form-storage term. The nine intraparent contacts of
parent zero have relative increment \(1/2\); the eighteen outer contacts
have increments of magnitude \(1/4\). On the moving pre-event reference,

\[
 W_T^*(B_4,\Delta_4)
 =9[\cos B_4-\cos(B_4+1/2)]
      +18\cos\Delta_4[1-\cos(1/4)].
 \tag{18}
\]

The acquired residual child means need not vanish. There is no first-order
cancellation analogous to the initial attachment, and no quadratic error
claim is made. For \(g_k(v)=\cos v-\cos(v+k)\),
\(|g_k'(v)|\le|k|\); each actual edge-gap error is at most
\(2Z_{\rm pre}\). Thus

\[
 |W_T-W_T^*|\le18Z_{\rm pre}.
 \tag{19}
\]

Elementary bounds using \(0<B_4<1/64\) and the small positive outer
gap give

\[
 1<W_T^*<225/128,
 \qquad 0<W_T<3.
 \tag{20}
\]

For the upper bound, the intraparent term is at most
\(9(B_4/2+1/8)\), and the outer term at most \(18/32\).
For the lower bound, the intraparent term alone exceeds
\(9[1-\cos(1/2)]>1\). The signed correction (19) leaves strict
margin to the fixed external-work budget. This event is not passive, its
occurrence is supplied, and earlier continuous loss does not fund it. The
original preparation and initial contact work remain separate obligations.

## 8. Evidence and scope

The [detached assessor](../../src/tnfr/physics/_organized_parent_restoring_response.py)
retains fresh primitive-source reconstruction,
the acquired-state history, both branch errors, event work and every strict
margin. Its trigonometric and exponential bounds assess only the fixed
analytic inequalities; no response trajectory, controller or horizon search
is consumed. The eight-coordinate forecast is a conditional continuous-law
comparison initialized from actual observations, not a substituted target.
The [independent controls](../../tests/physics/test_organized_parent_restoring_response.py)
check the complete fine rows and inherited modes, jump invariants and work,
clock transformations, source rejection, strict margins and the finite
prediction bounds against independent arithmetic. Both reuse the original
[source acquisition owner](ORGANIZED_PARENT_ACQUISITION.md); neither
regenerates frozen trajectories or evaluated evidence.

The accepted result is finite loss and recovery of the declared parental
registration, with children retained, a paid intervention and a resolved
need for retained child state in its ensuing interaction. The two binary
support factors still have the exchange symmetry of the original matching
graph. The result supplies neither a unique grouping nor autonomous scale
selection, an indefinite lifetime, a binding-energy interpretation, an
unselected fundamental law or a laboratory measurement bridge.

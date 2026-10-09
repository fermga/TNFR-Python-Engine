# Constitutive inheritance through oriented weighted C9 contacts

<a id="oriented-interface-profile-inheritance"></a>

Sine is the unique current in the previously admitted analytic class that
keeps the same profile, up to the complete chart below, throughout a
nonempty open interval of the prescribed oriented contacts. This is a
conditional selection theorem. Requiring that unchanged profile is an
additional composition premise; the NFR ontology does not derive it.
The previous nonsine family still has an exact collective law, with
retained boundary-dependent sine and cosine coefficients.

The [preceding classification](COMPOSITION_COMPATIBLE_CONSTITUTIVE_LAWS.md)
owns the analytic class, its nonnegative potentials, target conditions and
two symmetric controls. Here an explicitly supplied weighted fine law
adds a distinct orientation-sensitive control. Neither source acquisition,
contact identity nor event occurrence transfers to this extension.

<a id="oriented-interface-state"></a>
## Fixed support, complete weighted law and registered identity

Keep two marked C9 components, node order \(9r+\ell\), \(r=0,1\),
\(\ell=0,\ldots,8\), with internal unit cycle edges and
\(\alpha=2\pi/9\). For a fixed real \(s\in[0,1]\), add the undirected
contacts

\[
 (0,\ell)--(1,\ell),\quad w=1-s,\qquad
 (0,\ell)--(1,\ell+1),\quad w=s .
 \tag{1}
\]

Indices are modulo nine. Zero-weight endpoint contacts are absent. Each
node has total cross conductance one and final weighted degree three.
For \(0<s<1\), there are eighteen cross edges; at either endpoint, nine.
The same weight acts at both ends of each undirected edge. The orientation
in (1) records the relative marked attachment, not a directed force.

The state is signed real form \(x\), primitive circular phase with declared
continuous lifts \(\theta\), held unit capacities and the fixed support
including \(s\). The structural clock is
\(\tau=(1023/1024)t\). Supply a positive constant reciprocal coefficient
\(\gamma\). With the actual weighted Laplacian \(L_s\), set

\[
 \begin{split}
 S_{j,s,i}(\theta)&=\sum_{m\sim i}w_{im}j(\theta_m-\theta_i),\\
 x'&=-\tfrac13L_sx+\tfrac{\gamma}{3}S_{j,s}(\theta),\\
 \theta'&=\tfrac{\gamma}{3}L_sx,\\
 H_s&=\tfrac12x^TL_sx+
                     \sum_{\{i,m\}}w_{im}V(\theta_m-\theta_i).
 \end{split}\tag{2}
\]

Primes denote \(\tau\)-derivatives. Both rows have the same factor
\(1023/1024\) in original time. The form row specifies the driver of the
unit-capacity nodal equation; no pressure is inferred from an evaluated
response. Structural units obey \([p]=[x]/([\nu_f][t])\); no laboratory
unit or inverse-mass interpretation is inferred.

Equation (2) is a supplied weighted extension of the detached reciprocal
comparison. It is neither the native pressure rule, whose phase channel
uses unweighted unique neighbors, nor the general operator runtime.
Its fixed weights are used consistently in form diffusion, phase current,
the reciprocal phase row and storage. Oddness of \(j\), evenness of \(V\)
and \(V'=j\) give

\[
 H_s'=-\tfrac13\|L_sx\|_2^2,\qquad
 (\mathbf1^Tx)'=(\mathbf1^T\theta)'=0 .
 \tag{3}
\]

The candidate registered identity is the invariant embedded family

\[
 \mathcal M:\quad
 x_{r,\ell}=X_r,\qquad
 \theta_{r,\ell}=\Phi_r+\alpha(\ell-4).
 \tag{4}
\]

Constituent membership, cyclic orientation, marked ports, relative phase
registration and the weight \(s\) remain part of the description. The
uniform phase-origin symmetry adds the same angle to every \(\Phi_r\);
it does not erase their difference. A simultaneous cyclic relabeling
preserves the contact rule. Reversing a constituent order or exchanging
the two components also transforms the marked contact offsets. Holding
the offsets unchanged while performing such a transformation need not
be a symmetry.

The first phase resultant of a winding-one C9 vanishes, and its ninth
moment loses registration modulo \(\alpha\). Neither is a substitute for
the lifted \(\Phi_r\) with its known marked embedding. This note makes no
claim that (4), or any particular attached equilibrium, is acquired or
transversely stable for every admitted law.

<a id="oriented-interface-restriction"></a>
## Exact collective rows and retained boundary coefficients

The admissible fine current from the preceding classification is

\[
 \begin{gathered}
 j(\delta)=\sum_{n\in\mathcal N}b_n\sin(n\delta),\qquad
 V(\delta)=\sum_{n\in\mathcal N}\frac{b_n}{n}(1-\cos n\delta)\ge0,\\
 \mathcal N=\{n\ge1:n\equiv1\text{ or }-1\pmod9\},\qquad
 \eta_n=\begin{cases}1&n\equiv1\pmod9,\\-1&n\equiv-1\pmod9,\end{cases}\\
 \sum_n b_n/n=1,\qquad \sum_n nb_n=1,\qquad
 \sum_n\eta_n n^2b_n=1 .
 \end{gathered}\tag{5}
\]

The real coefficients decay exponentially. This is the previous real
analyticity assumption and justifies the termwise operations below.
It matches potential values, complete tangent and quadratic field at
the earlier targets; it does not require equality of every target edge
current. Nonnegative potential is a separate global premise.

Write \(u=X_1-X_0\), \(\Delta=\Phi_1-\Phi_0\), and define

\[
 \begin{split}
 J_s(\Delta)&=(1-s)j(\Delta)+s j(\Delta+\alpha),\\
 U_s(\Delta)&=(1-s)V(\Delta)+s V(\Delta+\alpha),\\
 a_s&=1-s+s\cos\alpha,\qquad d_s=s\sin\alpha .
 \end{split}\tag{6}
\]

Internal cycle currents cancel at each node. A component-zero node sees
gaps \(\Delta,\Delta+\alpha\). At component one the same undirected
contacts give the negative of that weighted current. Thus the exact rows
on (4) are

\[
 \begin{array}{ll}
 X_0'=\tfrac13[u+\gamma J_s(\Delta)],&
 X_1'=-\tfrac13[u+\gamma J_s(\Delta)],\\
 \Phi_0'=-\tfrac{\gamma}{3}u,&
 \Phi_1'=\tfrac{\gamma}{3}u .
 \end{array}\tag{7}
\]

Periodic analytic currents have bounded derivatives on the real circle,
so the lifted fine vector field is globally Lipschitz. The tangent
calculation and uniqueness therefore prove invariance of (4).

For every law (5), not only sine, the current and potential close exactly:

\[
 \begin{split}
 J_s(\Delta)&=\sum_n b_n
       [a_s\sin(n\Delta)+\eta_n d_s\cos(n\Delta)],\\
 U_s(\Delta)&=1-\sum_n\frac{b_n}{n}
       [a_s\cos(n\Delta)-\eta_n d_s\sin(n\Delta)]\\
 &=1-a_s+a_sV(\Delta)+d_sK(\Delta),\\
 K(\Delta)&=\sum_n\frac{\eta_nb_n}{n}\sin(n\Delta).
 \end{split}\tag{8}
\]

The retained law coefficients are therefore the pairs
\((a_sb_n,\eta_nd_sb_n)\), with the potential constant fixed as in (8).
They depend on supplied boundary geometry, not on an inferred force.
For a finite Fourier family there are finitely many such coefficients;
for the analytic class the convergent sequence specifies the law.
On the invariant family no extra evolving hidden coordinate is needed
for fixed \(s\). Arbitrary off-family states retain their constituent
information and need a separate closure or approximation argument.

Although the fine current is odd, \(J_s\) in the original relative
coordinate need not be odd. The reverse-oriented contact has potential
\(U_s(-\Delta)\) and the corresponding sign-reversed derivative. This
is consistent with the same undirected reciprocal exchange; the marked
interface carries the orientation.

Let \(B=18(1-\cos\alpha)\), the internal potential storage fixed by (5).
The full storage and loss restrict to

\[
 H_s|_{\mathcal M}=B+9[\tfrac12u^2+U_s(\Delta)],\qquad
 H_s'|_{\mathcal M}=-6u^2 .
 \tag{9}
\]

The inherited potential \(U_s\) is nonnegative because it is a convex
combination of nonnegative fine potentials. Its value at the original
alignment is \(U_s(0)=s(1-\cos\alpha)\). Subtracting that alignment value
does not generally leave a nonnegative function: alignment is not in
general an equilibrium of (7), because \(J_s(0)=s\,j(\alpha)\).
For the explicit family (20), \(j(\alpha)=\sin\alpha\), so original
alignment is not an equilibrium at any strict interior contact.

<a id="oriented-interface-selection"></a>
## Same-profile inheritance on an interface interval selects sine

Fix any nonempty open interval \(I\subset(0,1)\) before evaluating a law.
The additional profile premise requires that for each \(s\in I\) there
exist a positive common form chart \(X_r=A_sZ_r\), a positive constant
common clock rescaling, a constant registration shift
\[
 \Psi_r=\Phi_r+(2r-1)\beta_s/2 ,
 \tag{10}
\]
and one positive reciprocal coefficient making both complete rows the
same normalized two-node reciprocal law with current \(j\).
The shift and all charts may depend on the fixed interface, but not on
state, trajectory or observation. No regularity in \(s\) is assumed for
the chosen shift; its relation to the original marked ports is retained.

For \(z=Z_1-Z_0\), comparing diffusion terms fixes
\(\sigma=\tau/3\). Comparing both reciprocal rows then requires

\[
 J_s(\Delta)=\rho_sj(\Delta+\beta_s),\qquad
 A_s=\sqrt{\rho_s},\qquad
 \gamma_{\rm eff}=\gamma\sqrt{\rho_s},\qquad \rho_s>0 .
 \tag{11}
\]

This is the allowed map class, not a claim about every conceivable
nonlinear state conjugacy. A fixed reciprocal coefficient would be a
different and stronger condition.

Define

\[
 \mu(s)=1-s+s e^{i\alpha}=\rho(s)e^{i\phi(s)},\qquad
 \rho(s)=\sqrt{1-2s(1-s)(1-\cos\alpha)} .
 \tag{12}
\]

Its real part is positive, \(\rho>0\), and the continuous argument obeys
\[
 0<\phi(s)<\alpha,\qquad
 \phi'(s)=\frac{\sin\alpha}{\rho(s)^2}>0\quad(0<s<1).
 \tag{13}
\]
For an active Fourier mode \(b_n\ne0\), its multiplier is
\[
 1-s+s e^{in\alpha}=\rho(s)e^{i\eta_n\phi(s)}.
 \tag{14}
\]
Uniqueness of complex Fourier coefficients in (11) consequently gives
\[
 \rho_s=\rho(s),\qquad
 n\beta_s-\eta_n\phi(s)\in2\pi\mathbb Z
 \quad\text{for every active }n .
 \tag{15}
\]

There is at least one active coefficient by the target moments.
Suppose two distinct positive modes \(m,n\) are active. Eliminate
\(\beta_s\) from (15):
\[
 (m\eta_n-n\eta_m)\phi(s)\in2\pi\mathbb Z
 \quad\text{for every }s\in I .
 \tag{16}
\]
The integer factor is nonzero. With equal residue signs it is a signed
difference of distinct positive modes; with opposite signs it is a
signed sum. Equation (13) maps every nonempty open interval to an
interval of positive length, while (16) would map it to a discrete set.
This is impossible. The argument permits arbitrary discontinuous choices
of \(\beta_s\) and never assumes that the first harmonic is present.

Exactly one mode remains, say \(j=b_n\sin n\delta\).
The first two moments of (5) require \(b_n/n=1\) and \(nb_n=1\).
Hence \(n=1\), \(b_1=1\); the third moment also holds. Conversely sine
satisfies every premise with \(\beta_s=\phi(s)\) modulo \(2\pi\),
\(\rho_s=\rho(s)\). Thus

\[
 \boxed{\text{same-profile inheritance on }I
        \quad\Longleftrightarrow\quad j(\delta)=\sin\delta}
 \tag{17}
\]

within (5) and the map class (10)--(11). The proof uses full profile
equality, not a finite sample of responses or the separately stronger
first-resultant information premise.

The interval quantifier matters. At \(s=0\), every current has
\(J_0=j\); at \(s=1\), every current has \(J_1(\Delta)=j(\Delta+\alpha)\).
They use \(\rho=1\) and shifts zero or \(\alpha\).
Even a strict interior singleton does not in general select sine:
at \(s=1/2\), put
\[
 \widetilde V_\epsilon(\delta)=1-\cos\delta+
               \epsilon\cos\delta(1-\cos18\delta)^2,\qquad
 0<|\epsilon|\le1/1296 .
 \tag{18}
\]
Its current has only frequencies \(1,17,19,35,37\), all congruent to
\(\pm1\) modulo eighteen. It satisfies (5), since the perturbation
vanishes to order four at every \(k\alpha\). The bound
\[
 |\cos\delta(1-\cos18\delta)^2|
               \le648(1-\cos\delta)
 \tag{19}
\]
proves global nonnegativity in (18). At the midpoint
\(\rho=\cos(\alpha/2)\), \(\phi=\alpha/2\), every one of its modes
satisfies (15) with \(\beta=\alpha/2\). It therefore preserves its
nonsine profile at that single interface, though not on an open interval.

<a id="oriented-interface-counterfamily"></a>
## The existing nonsine family remains closed but changes profile

The previously admitted potential
\[
 V_\epsilon(\delta)=1-\cos\delta+
             \epsilon\cos\delta(1-\cos9\delta)^2,\qquad
 |\epsilon|\le1/324
 \tag{20}
\]
has current coefficients
\[
 b_1=1-\tfrac32\epsilon,\quad b_8=8\epsilon,\quad
 b_{10}=10\epsilon,\quad b_{17}=-\tfrac{17}{4}\epsilon,\quad
 b_{19}=-\tfrac{19}{4}\epsilon.
 \tag{21}
\]
The exact boundary coefficients in (8) give its inherited complete law
for every \(s\in[0,1]\), with no fitted parameters.

When \(\epsilon\ne0\), both modes one and eight are active. Their
conditions (15) force \(9\phi(s)\in2\pi\mathbb Z\), whereas
\(0<9\phi(s)<2\pi\) throughout the strict interior.
This particular family fails same-profile inheritance at every
\(0<s<1\), not merely somewhere in an interval. That failure changes
neither the exact closure (7) nor the nonnegative storage (9).

Applying the sine phase shift \(\phi(s)\) to a nonsine law still gives
a valid invertible chart, but generally not the same current profile or
an equilibrium at zero shifted phase. Nor is \(1-\rho(s)\) necessarily
the minimum of its actual \(U_s\). Equations (8)--(9) retain the actual
potential and all its constants without that inference.

More explicitly, set \(\psi=\Delta+\phi(s)\) and retain the same
positive form/clock chart \(X_r=\sqrt\rho Z_r\),
\(\sigma=\tau/3\), \(\gamma_{\rm eff}=\gamma\sqrt\rho\).
Every current (5) then has the exact normalized boundary law
\[
 \begin{split}
 R_s(\psi)&=\frac{J_s(\psi-\phi)}{\rho}
     =\sum_n b_n\sin\{n\psi+(\eta_n-n)\phi\},\\
 Q_s(\psi)&=\frac{U_s(\psi-\phi)}{\rho}\ge0,\qquad Q_s'=R_s .
 \end{split}\tag{21a}
\]
Thus its transformed sine/cosine coefficient pair is
\((b_n\cos((\eta_n-n)\phi),b_n\sin((\eta_n-n)\phi))\).
Both transformed reciprocal rows have the form (23), with sine replaced
by \(R_s\), while
\(H_s=B+9\rho[\tfrac12z^2+Q_s(\psi)]\).
The full potential constant is retained: \(Q_s\) need not vanish at
zero. This exact chart changes the profile, rather than asserting that
the nonsine law has become sine.

<a id="oriented-interface-complete-map"></a>
## Complete sine map, work, sources and measured coordinates

For the uniquely selected profile under the additional premise,
\[
 U_s(\Delta)=1-\rho(s)\cos(\Delta+\phi(s))
            =1-\rho(s)+\rho(s)V(\Delta+\phi(s)).
 \tag{22}
\]
Using (10) with \(\beta_s=\phi(s)\), \(X_r=\sqrt\rho Z_r\),
\(\sigma=\tau/3\), \(\widehat\Delta=\Psi_1-\Psi_0\) and
\(\gamma_{\rm eff}=\gamma\sqrt\rho\), both rows are
\[
 \begin{array}{ll}
 dZ_0/d\sigma=z+\gamma_{\rm eff}\sin\widehat\Delta,&
 dZ_1/d\sigma=-z-\gamma_{\rm eff}\sin\widehat\Delta,\\
 d\Psi_0/d\sigma=-\gamma_{\rm eff}z,&
 d\Psi_1/d\sigma=\gamma_{\rm eff}z .
 \end{array}\tag{23}
\]
The full fine storage and loss become
\[
 \begin{split}
 H_s|_{\mathcal M}&=B+9(1-\rho)
                       +9\rho[\tfrac12z^2+1-\cos\widehat\Delta],\\
 dH_s/d\sigma&=-18\rho z^2 .
 \end{split}\tag{24}
\]
At strict interior contacts \(\rho<1\); thus the additive boundary
baseline is positive and the reciprocal coefficient changes. The common
factor \(1/3\) is a derived rate in a declared clock, not a newly derived
capacity or physical mass.

Uniform fine form inputs \(f_r(\tau)\) and independent phase inputs
\(g_r(\tau)\) preserve (4). For any current (5), their exact balance is
\[
 H_s'=-6u^2+
             9[u(f_1-f_0)+J_s(\Delta)(g_1-g_0)].
 \tag{25}
\]
In the sine chart they become \(3f_r/\sqrt\rho\) and \(3g_r\),
respectively, with original time arguments \(\tau=3\sigma\).
Dividing (25) by \(1/3\) and applying the chart agrees with (24)'s
storage derivative and source work. A uniform form jump \(h_r\)
becomes \(h_r/\sqrt\rho\); a phase jump keeps its angle while the same
fixed registration offset remains in the chart. Its work is the
difference of the full storage, not just its dissipation term.

Measured original coordinates remain
\[
 X_r=\sqrt\rho Z_r,\qquad
 \Phi_r=\Psi_r-(2r-1)\phi(s)/2,\qquad
 \theta_{r,\ell}=\Phi_r+\alpha(\ell-4).
 \tag{26}
\]
Thus the contact-dependent phase shift is not an unobserved change of
physical registration. Any other observation is composed with this
same embedding and chart. Sensor calibration is an independent model.

Attaching (1) at fixed state changes storage by the exact nonnegative
support work
\[
 \mathcal W_s=\sum_{\ell=0}^{8}\left[
 (1-s)\{\tfrac12(x_{1,\ell}-x_{0,\ell})^2+
                 V(\theta_{1,\ell}-\theta_{0,\ell})\}
 +s\{\tfrac12(x_{1,\ell+1}-x_{0,\ell})^2+
                 V(\theta_{1,\ell+1}-\theta_{0,\ell})\}\right].
 \tag{27}
\]
On (4), \(\mathcal W_s=9[\tfrac12u^2+U_s(\Delta)]\). For all
laws (5), attachment at \(u=\Delta=0\) requires
\(9s(1-\cos\alpha)\). For sine at its shifted minimum
\(u=0,\Delta=-\phi(s)\), it requires \(9(1-\rho(s))\).
Both are positive for strict interior \(s\). The additive constant in
(24) therefore has an observable role in the support-event budget.
A positive cost needs a declared supply; continuous loss does not
provide a reservoir or select the event.

All these charts hold \(s\) fixed. A time-dependent contact weight changes
the fine storage explicitly, introduces registration and form-chart
derivatives, and requires its own weight evolution and work law.
Neither (23) nor (25) supplies those missing terms.

<a id="oriented-interface-ontology"></a>
## The composition premise and the ontological boundary

The [NFR ontology](../EMERGENT_ONTOLOGY.md#12-can-nfr-properties-justify-a-joint-law)
requires an inherited description of the actual organization and its
relations, with sufficient internal/boundary state, derived memory or
controlled error. Equation (8) meets that continuous requirement on
the invariant family even when (11) fails. An object can retain its
constitutive coefficients and marked attachment data while its effective
interaction profile depends on its boundary.

Consequently the requirement that every weighted contact in a declared
interval reduce to the same scalar profile under (10)--(11) is an
independently supplied additional hypothesis. It is mathematically
consistent and sharply restrictive, but it is not deduced from nesting,
phase circularity, positive storage, the nodal identity or continuous
closure. Sine selection under that hypothesis does not establish that
nature or all TNFR organizations satisfy it.

The supplied contact is likewise not an automatically realized Coupling
operator. Its live admission, marked targets, retained history,
[collective descent](../STRUCTURAL_OPERATORS.md#collective-operator-descent)
and occurrence need their own contracts. This result derives neither
the objects' acquisition under the weighted law nor an autonomous
attachment, universal hierarchy or physical identification. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns the next research decision.

The [three-object network comparison](ORGANIZED_NETWORK_COMPATIBILITY.md)
separately tests simultaneous use of these pairwise charts. Its cycle-phase
and amplitude conditions can fail even though every pair has a valid sine
description; the complete network then retains its boundary coefficients.
Neither pairwise equivalence nor this retained network closure supplies an
acquired-source identity certificate.

## Implementation and independent controls

The detached static owner
[_oriented_interface_constitutive.py](../../src/tnfr/physics/_oriented_interface_constitutive.py)
represents the supplied weighted contacts, actual boundary coefficients,
complete collective rows and storage of the explicit family (20). It
admits primitive state and parameters before constructing these reports.
It fixes the fine reciprocal coefficient to \(1/(1023\pi)\); the proof
allows any positive fine coefficient and its explicitly mapped value.
It does not install a native law, execute a support event or transfer
an acquired-source certificate.

The [independent controls](../../tests/physics/test_oriented_interface_constitutive.py)
rebuild the eighteen-node weighted field and storage, the registration
chart, source work and harmonic obstructions.
They check static implemented behavior under the stated proof; no
trajectory, source producer, physical dataset or frozen evidence is
evaluated here.

# Dynamical protection and regeneration of organized contact

<a id="organized-contact-dynamic-cancellation"></a>

This owner answers the bounded mechanism question declared at `173265a3` in
the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#organized-contact-dynamic-cancellation).
The ideal fivefold host cancels the initial aggregate contact current for
both retained constitutive laws. That cancellation persists at particular
symmetry origins; at every other origin reciprocal dynamics regenerates a
nonzero donor mean. The first nonzero derivative depends on the supplied law.

These are exact conditional results about the declared limiting profiles.
The [sine contact](ORGANIZED_EXTERNAL_REGISTRATION.md) and
[alternative contact](ORGANIZED_ALTERNATIVE_CONTACT.md) retain their actual
acquired sources, finite error bounds and observation histories. No finite
source is reset to the profiles studied here, and no response derivative is
identified with an experimental measurement.

## 1. Complete model, initial state and balances

Retain the [constitutive family](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-constitutive-freedom)
at its two declared endpoints, \(\lambda=0,1\):

\[
 \begin{split}
 V_\lambda(a)&=1-\cos a+
       \frac{\lambda(1-\cos9a)^2}{6\,9^4},\\
 j_\lambda(a)&=V_\lambda'(a)
   =\sin a+\frac\lambda{2187}
                (\sin9a-\tfrac12\sin18a).
 \end{split}\tag{1}
\]

The support is \(C_9\mathbin\square W_6\): six internal nine-cycles,
five host rim edges and five donor spokes at each slot. Its 54 nodes are
\((c,j)\), where \(c=0,\ldots,4\) labels the host, \(c=D\) the donor,
and \(j=0,\ldots,8\). All 144 edges have weight one. On that full support,

\[
 x'=\tfrac14[-Lx+\gamma S_\lambda(\theta)],\qquad
 \theta'=\tfrac\gamma4Lx,\qquad
 S_{\lambda,a}=\sum_{b\sim a}j_\lambda(\theta_b-\theta_a),\qquad
 \gamma=\frac1{1023\pi}.
 \tag{2}
\]

Prime denotes the declared clock \(\tau=(1023/1024)t\). Forms are signed,
capacities are held at one, and phases occupy the full circle. Both rows
have the same supplied quarter factor; neither row uses inverse degree.
There is no continuous input or later event. The pressure in the form row
is supplied independently of the response. Smooth periodic currents and
the linear terms give global existence and uniqueness.

The conditional initial family is exactly

\[
 x_{c,j}(0)=0,\quad
 \theta_{c,j}(0)=\psi_j+\phi_c,\quad
 \theta_{D,j}(0)=\psi_j+\beta,\qquad
 \psi_j=\frac{2\pi(j-4)}9,\quad
 \phi_c=\frac{2\pi(c-2)}5.
 \tag{3}
\]

The symbolic registration \(\beta\) ranges modulo a turn. The observation
is \(R_D(\tau;\beta)=\frac19\sum_jx_{D,j}(\tau)\). The initial zero
total form remains zero; the declared phase lifts have constant total
\(9\beta\). This is the exact limiting organization supplied to a new
conditional question, with no observation word or residual replacement.

Reciprocity retains both contact reactions and the full storage identity:

\[
 H=\tfrac12x^TLx+\sum_{\{a,b\}}V_\lambda(\theta_b-\theta_a),\qquad
 H'=-\tfrac14\|Lx\|_2^2.
 \tag{4}
\]

The attachment in (3) supplies the separate work
\(W=9\sum_cV_\lambda(\beta-\phi_c)=45+5\lambda/2916\), independent
of registration. The root-of-unity sums of harmonics 1, 9 and 18 vanish.
Thus initial storage is \(6V_9+9V_{5,\lambda}+W\), where
\(V_9=9V_\lambda(2\pi/9)\) and
\(V_{5,\lambda}=5V_\lambda(2\pi/5)\). Independence of \(\beta\)
does not make the positive attachment cost passive or select its occurrence.

## 2. Exact carrier and complete-field symmetries

The twelve-coordinate carrier
\(x_{c,j}=X_c\), \(\theta_{c,j}=\psi_j+\Phi_c\) is invariant for
each law in (1): the internal clockwise and counterclockwise currents
cancel, as do the internal form differences. With the wheel Laplacian

\[
 K=\begin{pmatrix}B&-\mathbf1\\-\mathbf1^T&5\end{pmatrix},\qquad
 B=3I-A_{C_5},
 \tag{5}
\]

the complete restriction is
\(X'=(-KX+\gamma S_\lambda(\Phi))/4\),
\(\Phi'=\gamma KX/4\), and \(R_D=X_D\). All six forms and six
phase coordinates evolve. The carrier storage is
\(6V_9+(9/2)X^TKX+9\sum_{\text{wheel edges}}V_\lambda\), with
loss \(-9\|KX\|^2/4\). This restriction preserves the full fine-state
embedding; it is not a closed equation for an instantaneous contact moment.

The eigenvalues of \(B\) on the nonconstant rim Fourier modes are

\[
 \ell_1=\ell_4=\frac{7-\sqrt5}2,\qquad
 \ell_2=\ell_3=\frac{7+\sqrt5}2.
 \tag{6}
\]

Support permutations and common phase translations are exact symmetries
of (2). Since each current is odd, every graph automorphism \(P\) also
gives the complete-field conjugation

\[
 (x,\theta)\longmapsto(-Px,\ 2b\mathbf1-P\theta).
 \tag{7}
\]

Indeed both right-hand sides acquire the same required minus sign and
permutation. Equality of phases here is circular; constant integer turns
can be supplied to the transformed lifts to match initial representatives.

For \(\beta=m\pi/5\), take \(b=\beta\) and the full-support
reflection

\[
 P(c,j)=(r-c\bmod5,8-j),\qquad
 P(D,j)=(D,8-j),\qquad r=4+m\bmod5.
 \tag{8}
\]

Here \(\psi_{8-j}=-\psi_j\) and
\(\phi_c+\phi_{r-c}\equiv2\beta\pmod{2\pi}\). The state (3)
is fixed by (7). Uniqueness therefore preserves this fixed sector for all
future time, and \(x_{D,j}=-x_{D,8-j}\) gives

\[
 R_D(\tau;m\pi/5)=0\qquad\text{for every }\tau\ge0,
 \quad \lambda=0,1.
 \tag{9}
\]

This proof uses the actual complete field, including slot reflection and
form sign reversal. A stationary reflection constraint alone would not
establish invariant dynamics.

For general registration, a common phase shift by \(2\pi/5\) followed
by a rim relabeling restores the host family. Conjugation with \(b=0\),
\(r=4\) reverses its donor origin. Consequently

\[
 R_D(\tau;\beta+2\pi/5)=R_D(\tau;\beta),\qquad
 R_D(\tau;-\beta)=-R_D(\tau;\beta).
 \tag{10}
\]

These restrictions will remove impossible response harmonics. They also
reduce the odd multiples of \(\pi/10\) to the two representatives
\(\pm\pi/10\), related by sign.

## 3. Alternative-law generic onset

Put \(\delta_c=\beta-\phi_c\),
\(f_c=j_\lambda(\delta_c)\), and
\(p_c=j_\lambda'(\delta_c)\). Root-of-unity filtering gives
\(\sum_cf_c=0\). The exact hub equation, using zero total form, is

\[
 R_D'=-\tfrac32R_D-\tfrac\gamma4 A,\qquad
 A=\sum_cj_\lambda(\Phi_D-\Phi_c).
 \tag{11}
\]

Initially the phase velocity is zero. Thus
\(R_D(0)=R_D'(0)=R_D''(0)=0\),
\(\delta''=-\gamma^2Bf/16\), and

\[
 R_D'''(0;\beta)=\frac{\gamma^3}{64}p^TBf.
 \tag{12}
\]

Write the three current coefficients as
\(b_1=1\), \(b_9=\lambda/2187\),
\(b_{18}=-\lambda/4374\). In \(f^TBf\), only the pair
\(1+9=10\) contributes registration dependence: every other nonconstant
sum or difference fails the fivefold filter. Modes 1 and 9 have the same
eigenvalue \(\ell_1\), so

\[
 f^TBf=\text{constant}-5\ell_1b_1b_9\cos10\beta,\qquad
 R_D'''(0;\beta)=
 \frac{25\ell_1\lambda\gamma^3}{64\cdot2187}\sin10\beta.
 \tag{13}
\]

For \(\lambda=1\) and \(\beta\notin(\pi/10)\mathbb Z\), this
is the first nonzero derivative, with a strictly positive prefactor.
For the sine law it vanishes identically, requiring the separate argument
below. Matching initial cancellation does not transfer the later response.

## 4. Alternative-law exceptional origins

Even multiples of \(\pi/10\) are already protected by (9). At
\(\beta=\pi/10\), put \(u=1/2187\). On the five initial gaps,
\(10\delta_c\equiv\pi\pmod{2\pi}\), so the actual current and
its derivatives, differentiated before this substitution, are

\[
 \begin{split}
 f&=(1+u)\sin\delta+(u/2)\sin2\delta,\\
 p&=(1-9u)\cos\delta-9u\cos2\delta,\\
 q:=j_1''(\delta)&=-(1+81u)\sin\delta-162u\sin2\delta.
 \end{split}\tag{14}
\]

The third derivative is zero by (13). The fourth contains only bilinear
terms of these discrete modes with powers of \(B\); their sums are zero
at this origin. More explicitly, \(\delta'''=\gamma^2B^2f/64\) and
\(p^TB^2f=0\), so (11) gives \(R_D^{(4)}(0)=0\).

For the fifth derivative set \(v=Bf\). Differentiating both complete
rows, with the first four donor derivatives zero, gives

\[
 A^{(4)}(0)=\frac{\gamma^4}{256}
       \left[p^TB(pv)+3\sum_cq_cv_c^2\right],\qquad
 R_D^{(5)}(0)=-\frac{\gamma^5}{1024}\mathcal K,
 \quad \mathcal K=p^TB(pv)+3\sum_cq_cv_c^2.
 \tag{15}
\]

Here \(pv\) is componentwise multiplication. To check that the rim
reaction has not been omitted, let \(J=DS_1(\Phi(0))\). On
\((v,0)\) its rim row is
\(-wL_{C_5}v-pv\), where \(w=j_1'(2\pi/5)\), and its hub row is
\(p^Tv=0\). The fourth gap derivative therefore contains
\(-\gamma^2B^3f/256-\gamma^4(KJ(v,0))_{\rm rim}/256\).
The terms \(p^TB^3f\) and \(p^TBL_{C_5}Bf\) vanish by the same
bilinear filter; the remaining terms give (15). The factor 3 comes from
the second phase derivative in the fourth derivative of the current.

For an explicit sign calculation, write
\(U=\ell_1(1+u)\), \(V=\ell_2u/2\),
\(D=1-9u\), \(E=-9u\), \(F=-(1+81u)\), \(G=-162u\).
The only surviving cubic products have total discrete harmonic five.
The sums of \(\cos\delta\cos2\delta\sin2\delta\) and
\(\cos^22\delta\sin\delta\) are \(5/4\); the sum of
\(\sin\delta\sin^22\delta\) is \(-5/4\). Thus

\[
 \begin{split}
 \mathcal K
  &=\tfrac54[(\ell_1+\ell_2)DEV+\ell_2E^2U-3FV^2-6GUV]\\
  &=\frac5{16}\ell_2u^2
       [(2250+2430u)\ell_1+(-15+405u)\ell_2]>0.
 \end{split}\tag{16}
\]

Positivity follows already from \(\ell_2<2\ell_1\): the bracket
exceeds \(2250\ell_1-15\ell_2>0\). Substitution simplifies (15) to

\[
 R_D^{(5)}(0;\beta)=
 \frac{1750\sqrt5-828975}{528958107648}\,
       \gamma^5\sin5\beta,
 \qquad \beta\in\frac\pi{10}(2\mathbb Z+1).
 \tag{17}
\]

The coefficient is strictly negative. Equation (10) extends the calculation
at \(\pi/10\) to every odd exceptional origin, where
\(\sin5\beta=\pm1\). The fifth derivative is therefore the first
nonzero one throughout this exceptional set.

## 5. Sine-law onset and the fivefold filter

For \(\lambda=0\), use the temporary algebraic clock \(s=\tau/4\)
in both carrier rows. This removes their quarter factors; derivatives in
the declared clock will be restored explicitly. Every contact derivative
is a finite Fourier polynomial in \(z=e^{i\beta}\). Induction in the
complete rows gives

\[
 \deg_\beta\Phi^{(n)}(0)\le\lfloor n/2\rfloor,\qquad
 \deg_\beta X^{(n)}(0)\le\lfloor(n+1)/2\rfloor.
 \tag{18}
\]

For phase, the bound concerns its displacement from the initial phase.
The displacement has no constant or first-order term. A contact current
has one initial factor \(z^{\pm1}\); a rim current has none. Expanding
its phase exponential, each displacement coefficient of time order
\(n\) contributes at most \(\lfloor n/2\rfloor\) to Fourier
degree. Multiplication, one integration in the form row, and the next
integration in the phase row preserve (18). Linear damping adds no
degree. By (10), the donor coefficients are odd and contain only multiples
of harmonic five. Their degree is at most four through order eight, so
all those donor derivatives vanish exactly.

Only the maximal positive Fourier degree can contribute harmonic five
at order nine. At these minimal time orders, damping and the rim current
cannot attain that degree. Retain the maximal-degree part of the host
phase displacement in the form

\[
 [\Phi_c(s)-\phi_c]_{\mathrm{max},+}
   =i\sum_{n=1}^4a_n(\gamma^2s^2z e^{-i\phi_c})^n.
 \tag{19}
\]

Here the projection retains only \(s^{2n}z^n\), \(1\le n\le4\);
the conjugate negative modes remain in the real phase but cannot contribute
to this maximal positive-degree coefficient. The rim Laplacian remains
present through its eigenvalues \(\ell_n\)
in the phase row. The donor phase cannot contribute at these orders:
\(\Phi_{D,s}=6\gamma R_D\), and its displacement starts at order
ten or later. Equating maximal-degree terms gives the finite recurrence

\[
 a_n=-\frac{\ell_n}{2(2n)(2n-1)}
       [y^{n-1}]\exp\!\left(\sum_{r<n}a_ry^r\right).
 \tag{20}
\]

In particular,

\[
 \begin{split}
 a_1&=-\ell_1/4,&
 a_2&=-\ell_2a_1/24,\\
 a_3&=-\ell_2(a_2+a_1^2/2)/60,&
 a_4&=-\ell_1(a_3+a_1a_2+a_1^3/6)/112,\\
 b_4&=a_4+a_1a_3+a_2^2/2+a_1^2a_2/2+a_1^4/24
       =\frac{2603}{20160}-\frac{139\sqrt5}{3840}>0.
 \end{split}\tag{21}
\]

For example \(\sqrt5<9/4\) already proves the last strict inequality.
The resulting donor current term is
\(-5b_4\gamma^8s^8\sin5\beta\). Its form damping only affects
later orders, so integration and restoration of the tau clock give

\[
 R_D^{(9)}(0;\beta)
 =-\frac{5\,8!\,b_4}{4^9}\gamma^9\sin5\beta
 =\frac{14595\sqrt5-52060}{524288}\gamma^9\sin5\beta.
 \tag{22}
\]

The prefactor is strictly negative. For every
\(\beta\notin(\pi/5)\mathbb Z\), the ninth derivative is the
first nonzero one. Every remaining origin is protected for all time by
(9), rather than inferred invariant from a zero finite jet.

## 6. Exact implementation and outcome

The [exact coefficient kernel](../../src/tnfr/physics/_organized_contact_dynamic_cancellation.py)
retains all twelve carrier coordinates and all ten reciprocal wheel
edges. It expands each edge exponential by its differential recurrence
in the exact coefficient algebra
\(\mathbb Q[\zeta_{20}][\gamma,z,z^{-1}]\), where
\(\Phi_{20}(a)=a^8-a^6+a^4-a^2+1\), \(i=\zeta_{20}^5\), and
\(z=e^{i\beta}\). This preserves exact cancellation without floating
phase substitutions. The generic rows and the one symmetry-distinct
exception at \(\beta=\pi/10\) stop at their first nonzero donor
coefficient, within the frozen ceiling of twelve. Derivatives are
factorial times the separately retained Taylor coefficients.

The independent root sums and maximal-degree calculation above determine
the signs and the origin domains; the exact recurrence checks all preceding
coefficients and the complete carrier wiring. No zero jet alone certifies
an invariant sector, and no trajectory or acquired-source producer is used.
The kernel shares the exact Fourier coefficients with the
[constitutive owner](../../src/tnfr/physics/_sine_class_constitutive_comparison.py)
and existing contact observer. Its
[contract tests](../../tests/physics/test_organized_contact_dynamic_cancellation.py)
check the cyclotomic algebra, analytical coefficients, complete 54-node
field residuals, conserved sums, reflection and bounded admission/stopping.

| Supplied law | Registration domain | First nonzero donor derivative |
| --- | --- | --- |
| Either endpoint | \(\beta\in(\pi/5)\mathbb Z\) | None: complete-field symmetry proves zero response for all future time |
| Sine, \(\lambda=0\) | Every other origin | Order 9, equation (22) |
| Alternative, \(\lambda=1\) | \(\beta\notin(\pi/10)\mathbb Z\) | Order 3, equation (13) |
| Alternative, \(\lambda=1\) | Odd multiples of \(\pi/10\) | Order 5, equation (17) |

Thus the initial fivefold cancellation is not an origin-independent
interaction constraint. Reciprocal motion generates a collective response
at a law-dependent order, while an exact symmetry protects a discrete
set of origins. All registrations are classified within the original
budget. A nonzero leading derivative guarantees a nonzero response for
sufficiently small positive time at that fixed origin; it supplies no
noise-resolved horizon or actual-source detection threshold.

At common generic origins, where \(\sin10\beta\ne0\), the onset orders
nine and three cannot be made equal by a regular clock change or regular
scalar re-expression of this observation. Indeed, if
\(R(\tau)=a\tau^n+o(\tau^n)\), \(a\ne0\), a clock
\(\tau=f(s)\) with \(f(0)=0\), \(f'(0)>0\) preserves order \(n\).
So does \(h(R)-h(0)\) with \(h'(0)\ne0\). This is a local obstruction
for the same matched preparation and donor observable, not an exclusion
of arbitrary state transformations or observation maps consuming hidden
coordinates. A common formed identity does not force a common effective
interaction law.

All-future zero donor mean does not imply zero local currents, stationary
fine state or permanent retention of the marked windings. Nor does the
difference of onset orders select a fundamental law, establish a particle
property, provide autonomous attachment or complete a physical measurement
bridge. The prior finite-source contact bounds and the separate unresolved
maintenance problem retain their original scopes.

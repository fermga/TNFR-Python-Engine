# Predictive repeated composition with retained port association

<a id="repeated-organized-composition"></a>

Four winding-one C9 organizations, grouped into two marked two-child parents,
admit a useful four-coordinate parent description for one finite interaction.
The exact registered restriction reuses the
[organized inheritance method](ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md).
The additional result is a uniform predictive bound for independent fine
deviations, with child and parent identity retained, and an explicit obstruction
to exact parent closure when child-to-port association is discarded.

This is a conditional theorem for the supplied positive-loss sine law.
The support, grouping, relative preparation and structural clock are supplied.
The two parents interact through their actual boundary; no prescribed forcing
trajectory, new solver or evaluated numerical response is used.

## 1. Fixed question, source and observations

The test fixes the following quantities before assessing the response:

| Item | Declared value or scope |
| --- | --- |
| Children and parents | Four marked C9 children; two children in each parent |
| Complete law | Sine potential, held unit capacities, weighted degree four |
| Structural clock | \(\tau=(1023/1024)t\), \(\gamma=1/(1023\pi)\) |
| Initial parent forms | \(X_0=X_1=0\) |
| Initial parent origins | \(\Phi_0=-1/40,\ \Phi_1=1/40\), in declared continuous lifts |
| Fine residual family | Parentwise mean-zero form and phase residuals, each coordinate at most \(\rho=1/100\) in magnitude |
| Interaction controls | Matching outer ports, or equal weights on offsets \(0,-1,1\) |
| Horizon | \(0\le\tau\le1\) |
| Selected readout | Parent-zero mean form increment \(X_0(1)-X_0(0)\) |
| Required prediction error | Less than \(10^{-7}\) in the declared form chart |
| Child identity | Winding one on every marked internal C9 |
| Parent identity | Its two children retain those identities and every same-slot intraparent phase gap has magnitude less than \(1/4\) |

The parent identity is a declared structural observation, not a new definition
of every NFR or a parent winding number. Its selected interaction must remain
predictable while its children differ. A stationary nested target alone would
not answer this question. These are structural units and mathematical
observations, without a laboratory clock, physical energy or sensor bridge.

## 2. Complete 36-node model and the two grouping maps

Label nodes \((g,a,j)\), with parent \(g=0,1\), child \(a=0,1\) and
\(j=0,\ldots,8\), in order \(18g+9a+j\). Put
\(\alpha=2\pi/9\) and \(\psi_j=\alpha(j-4)\).

Each child has its nine unit cycle edges. Within each parent, add the nine
unit edges \((g,0,j)--(g,1,j)\). Between parents, join
\((0,a,j)--(1,a,j+s)\), with indices modulo nine:

\[
 \begin{array}{c|c|c}
 \text{control}&s&\text{conductance per edge}\\ \hline
 \text{matching}&0&1\\
 \text{symmetric}&0,-1,1&1/3
 \end{array}
 \qquad
 \kappa=1\quad\text{or}\quad\frac{1+2\cos\alpha}{3}.
 \tag{1}
\]

Both graphs have weighted degree four and total undirected conductance 72.
The symmetric control has more distinct neighbors, but its weighted degree
does not change. All diffusion and sine sums below consume conductance.
This is not the native resultant-direction pressure or its neighbor-count
normalization.

For the actual weighted Laplacian \(L\), define
\[
 S_i(\theta)=\sum_{k\sim i}w_{ik}\sin(\theta_k-\theta_i),\qquad
 x'=\frac{-Lx+\gamma S(\theta)}4,\qquad
 \theta'=\frac{\gamma Lx}4.
 \tag{2}
\]
Both rows acquire the factor \(1023/1024\) in the original clock \(t\).
The signed form, primitive phase and held capacities retain the
[foundation contracts](../FUNDAMENTAL_THEORY.md). In this chart the nodal
pressure is independently supplied by the first row, not reconstructed from
a measured derivative.

The complete storage and loss are
\[
 H=\frac12x^TLx+\sum_{\{i,k\}}w_{ik}[1-\cos(\theta_k-\theta_i)],
 \qquad H'=-\frac14\|Lx\|_2^2.
 \tag{3}
\]
This follows from \(\nabla_xH=Lx,\ \nabla_\theta H=-S\) and cancellation
of the reciprocal terms. The two global lifted means are conserved.

The first map takes each child's form mean and the mean of
\(\theta_{g,a,j}-\psi_j\). Its exact family is
\[
 x_{g,a,j}=X_{g,a},\qquad
 \theta_{g,a,j}=\phi_{g,a}+\psi_j.
 \tag{4}
\]
The second map averages the two child forms and the two registered child
origins in each parent. Its exact family additionally imposes
\(X_{g,0}=X_{g,1}=X_g\) and
\(\phi_{g,0}=\phi_{g,1}=\Phi_g\).
The composite observation therefore retains four coordinates from 72.

These phase means use declared lifts and marked slots. They are not angles
of the fine first resultant, which vanishes for an ideal winding-one C9.
Independent child relabeling with contacts held fixed is not a symmetry.
The whole first and second families are invariant, as shown next; neither
map is asserted to be an exact quotient outside its family.

## 3. Exact inherited rows, clock and storage

For \(\bar g=1-g,\ \bar a=1-a\), substitution in (2) gives
\[
 \begin{split}
 q_{g,a}&=2X_{g,a}-X_{g,\bar a}-X_{\bar g,a},\\
 s_{g,a}&=\sin(\phi_{g,\bar a}-\phi_{g,a})
          +\kappa\sin(\phi_{\bar g,a}-\phi_{g,a}),\\
 X_{g,a}'&=(-q_{g,a}+\gamma s_{g,a})/4,\qquad
 \phi_{g,a}'=\gamma q_{g,a}/4.
 \end{split}\tag{5}
\]
Internal cycle currents cancel. Every offset is a bijection on the marked
cycle, and the symmetric offsets cancel the cosine harmonic. Thus both
fine rows are constant within each child. Smooth uniqueness preserves (4).
The first-stage storage is
\[
 \begin{split}
 H_8={}&36(1-\cos\alpha)\\
 &+9\sum_g\left[\frac12(X_{g,1}-X_{g,0})^2+
                         1-\cos(\phi_{g,1}-\phi_{g,0})\right]\\
 &+9\sum_a\left[\frac12(X_{1,a}-X_{0,a})^2+
                         1-\kappa\cos(\phi_{1,a}-\phi_{0,a})\right].
 \end{split}\tag{6}
\]
The inner and outer phase coefficients differ. Replacing them by a single
unweighted graph coefficient would change the inherited model.

On the second family put \(u=X_1-X_0,\ \Delta=\Phi_1-\Phi_0\). Then
\[
 \begin{array}{ll}
 X_0'=(u+\gamma\kappa\sin\Delta)/4,&X_1'=-X_0',\\
 \Phi_0'=-\gamma u/4,&\Phi_1'=\gamma u/4,
 \end{array}
 \qquad
 u'=-u/2-\gamma\kappa\sin\Delta/2,\quad \Delta'=\gamma u/2.
 \tag{7}
\]
These rows also preserve the second family. The pullback of (3) is
\[
 H_4=B_\kappa+9u^2+18\kappa(1-\cos\Delta),\qquad
 B_\kappa=36(1-\cos\alpha)+18(1-\kappa),\qquad
 H_4'=-9u^2.
 \tag{8}
\]

The complete normalized pair chart uses
\[
 \sigma=\tau/4,\qquad Z_g=X_g/\sqrt\kappa,\qquad
 \gamma_{\rm eff}=\gamma\sqrt\kappa.
 \tag{9}
\]
In that clock both rows have the normalized reciprocal pair form:
\(dZ_0/d\sigma=Z_1-Z_0+\gamma_{\rm eff}\sin\Delta\) and
\(d\Phi_0/d\sigma=\gamma_{\rm eff}(Z_0-Z_1)\), with opposite rows for
parent one. The normalized relative storage is
\((H_4-B_\kappa)/(18\kappa)\). Changing only the form coefficient or only
the clock would not give this equivalence. Fine capacity remains held at
one; (9) does not select a new autonomous capacity law.

## 4. What the second grouping loses

Choose \(h=1/200\), retain zero forms, and set
\[
 \theta_{g,a,j}=\Phi_g+(-1)^a\eta_g+\psi_j.
 \tag{10}
\]
Compare preparation A with \((\eta_0,\eta_1)=(h,h)\) and preparation B
with \((\eta_0,\eta_1)=(h,-h)\). Both belong to the declared fine source.
Their four parent coordinates agree. Each parent's unordered set of child
origins agrees, as does its child-origin resultant magnitude \(\cos h\).
Each has internal storage
\(18(1-\cos\alpha)+9[1-\cos(2h)]\) and retains both child windings.
This resultant concerns the child origins, not the vanishing fine C9
resultants.

The actual average outer current into parent zero is
\[
 J=\kappa\sin\Delta\cos(\eta_1-\eta_0).
 \tag{11}
\]
It equals \(\kappa\sin\Delta\) in A and
\(\kappa\sin\Delta\cos(2h)\) in B. Their parent-zero form rates differ by
\[
 \frac{\gamma\kappa}{4}\sin(1/20)[1-\cos(2h)]>0.
 \tag{12}
\]
Thus parent means, separate unordered child states and internal parent
storage cannot define an exact four-coordinate law on this source.
The missing information is the association between the states meeting at
the marked ports. Complete outer storage is not claimed equal in A and B.
This is a specific second-grouping obstruction, not an inference from
overlapping error bounds. It remains compatible with an accurate approximate
parent law because the missing current is quadratic in small internal offsets.

<a id="repeated-composition-fine-domain"></a>
## 5. A full transverse domain and identity bound

At every time let \(X_g\) and \(\Phi_g\) be the actual parent means, and write
\[
 x_{g,a,j}=X_g+\xi_{g,a,j},\qquad
 \theta_{g,a,j}=\Phi_g+\psi_j+\eta_{g,a,j}.
 \tag{13}
\]
The residual sums vanish separately within each parent by definition.
At preparation the fixed parent means are exact; all 68 remaining form/phase
directions are allowed within the coordinate bounds. No child symmetry or
first-stage-family membership is assumed.

Let \(\Pi\) average each parent and \(Q=I-\Pi\). The equitable weighted
support gives \(L\Pi=\Pi L\). The exact transverse equations are
\[
 \xi'=-L\xi/4+\gamma Q S/4,\qquad \eta'=\gamma L\xi/4.
 \tag{14}
\]
The ideal current is constant within each parent. Let
\(r_x=\|\xi\|_\infty,\ r_\theta=\|\eta\|_\infty\).
For the mean outer current, the Taylor-linear terms cancel separately for
every offset: its bijection pairs two parentwise zero sums. Consequently
\[
 J=\kappa\sin\Delta+E,\qquad |E|\le2r_\theta^2.
 \tag{15}
\]
Indeed, each sine Taylor remainder is at most half the squared edge residual,
and each residual difference has magnitude at most \(2r_\theta\).
Internal currents cancel exactly in the parent mean.

The nodewise sine difference from the ideal family is at most \(8r_\theta\),
since weighted degree is four. Subtracting the parent mean therefore gives
\(\|QS\|_\infty\le8r_\theta+2r_\theta^2\).
The heat semigroup generated by \(-L/4\) contracts the infinity norm.
Duhamel's formula and \(\|L\|_\infty=8\) now give, while both residuals
are at most \(R\),
\[
 r_x(\tau)\le\rho+\gamma\tau(2R+R^2/2),\qquad
 r_\theta(\tau)\le\rho+2\gamma R\tau.
 \tag{16}
\]
Fix
\[
 R=\frac{1001}{100000}=0.01001.
 \tag{17}
\]
Using \(\gamma<1/3200\), both right-hand sides of (16) are strictly below
\(R\) for \(\tau\le1\). A first-exit argument proves the bound throughout
the declared interval. The field is globally Lipschitz; no hidden continuation
or phase-domain premise is needed for (2).

Every internal cycle edge has a continuous representative in
\([\alpha-2R,\alpha+2R]\subset(0,\pi/2)\).
The closing edge has raw carrier gap \(-8\alpha\); its retained branch
integer adds \(2\pi\), giving the same representative \(\alpha\).
Their oriented sum is \(2\pi\), because the residual differences telescope,
so every child retains winding one. Every same-slot intraparent gap has
magnitude at most \(2R<1/4\). This proves the declared parent identity for
the entire fine source and horizon, not only the nominal stationary geometry.

## 6. Quantitative parent dynamics and resolved interaction

The actual parent means satisfy (7) with \(\kappa\sin\Delta\) replaced by
\(J\) in the form rows. The diffusion and phase rows remain exact for all
fine states. The four-coordinate unforced predictor is (7), initialized from
the same parent observations.

To compare this predictor with the actual means, use conservation
\(X_1=-X_0\). In the fixed form/phase chart \((X,\Phi)\), the error
\(e=|X_0-\bar X_0|+|\Delta-\bar\Delta|\) obeys
\[
 D^+e\le\gamma e+\gamma R^2/2,\qquad e(0)=0.
 \tag{18}
\]
The form inequality retains the nonpositive diffusion contribution; the
phase inequality contributes \(\gamma|X_0-\bar X_0|\).
Thus for \(\tau\le1\),
\[
 e(\tau)\le\frac{R^2}{2}(e^{\gamma\tau}-1)
          <\frac{R^2}{6398}<\frac1{60000000}.
 \tag{19}
\]
The last step uses \(e^z-1\le z/(1-z)\) for \(0\le z<1\) and
\(\gamma<1/3200\). This is a controlled approximation despite the exact
closure obstruction (12); it is not a four-coordinate exact quotient.

An explicit finite readout avoids solving even the reduced trajectory.
Write \(\delta=1/20\) and define
\[
 P_\kappa(\tau)=\frac{\gamma\kappa\sin\delta}{2}
                              (1-e^{-\tau/2}).
 \tag{20}
\]
Since the average outer current has \(|J|\le1\), variation of constants in
the exact mean rows gives
\[
 |u(\tau)|\le\gamma\tau/2,\qquad
 |\Delta(\tau)-\delta|\le\gamma^2\tau^2/8.
 \tag{21}
\]
Together with (15), the exact convolution for the form yields
\[
 |X_0(\tau)-P_\kappa(\tau)|
 \le\frac{\gamma\tau}{4}
             \left(\frac{\gamma^2\tau^2}{8}+2R^2\right).
 \tag{22}
\]
At \(\tau=1\) this is less than \(1/60000000<10^{-7}\), uniformly over
the fine source. Equation (22) includes both the internal-state defect and
the change in parent phase; it is distinct from (19)'s error against the
complete nonlinear reduced law.

The prescribed matching/symmetric control is resolved. Elementary strict
bounds \(\gamma>1/3300\), \(1-\kappa_{\rm symmetric}>3/20\),
\(\sin(1/20)>49/1000\) and \(1-e^{-1/2}>3/8\) imply
\[
 P_1(1)-P_{\kappa_{\rm symmetric}}(1)>4\cdot10^{-7}.
 \tag{23}
\]
Subtracting both full-source errors proves
\[
 X_{0,\rm matching}(1)-X_{0,\rm symmetric}(1)>3.6\cdot10^{-7}.
 \tag{24}
\]
The result even allows independent members of the same admitted source
family in the two controls; pairing identical sources is included.
The prospectively allowed intervals \(P_\kappa(1)\pm10^{-7}\) are disjoint.
This is a finite analytic discrimination with structural units, not an
evaluated laboratory reading or a new numerical trajectory.

## 7. Carried acquisition and work at the second attachment

The broad residual domain above is independently supplied. It also contains
two actual acquired parent families under additional stated preparation.
For each parent, separately rebuild the existing matching-interface
[acquired-pair admission](ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md#organized-inheritance-acquired-state)
at deformation zero from its original primitive form/phase error allowances.
The original \(10^{-10}\) allowances pass its isolated acquisition, fixed
recovery dwell and first attachment admission. The all-future parent relative
norm is bounded by \(15\cdot10^{-70}\), which is below \(\rho\).

The original parent's global means are conserved. Translating its entire
phase by \(-1/40\) or \(1/40\) preserves its disconnected internal law and
storage. These relative origins are additional supplied preparation
information, not an autonomous choice or the unchanged all-aligned source.
After this preparation, adding the outer contacts changes the degree from
three to four. Equations (14)-(24), not the old degree-three dynamics,
admit the ensuing interaction. No residual is reset to its target.
The inherited long dwell remains a mathematical premise, not demonstrated
laboratory preparation.

For any actual fine state the outer support-event work is exactly
\[
 W_{\rm outer}=\sum_{\text{outer }\{i,k\}}w_{ik}
       \left[\frac12(x_k-x_i)^2+1-\cos(\theta_k-\theta_i)\right].
 \tag{25}
\]
Its nominal value at the fixed preparation is
\[
 W_\kappa^*=18(1-\kappa\cos\delta).
 \tag{26}
\]
The same zero-sum cancellation used in (15), Taylor's bound
\(|\cos''|\le1\), and total outer conductance 18 give
\[
 -36\rho_\theta^2
 \le W_{\rm outer}-W_\kappa^*
 \le36(\rho_x^2+\rho_\theta^2).
 \tag{27}
\]
Here the radii bound actual initial residuals. The exact observer retains
their work, while acquired-source bounds retain the tiny correction
separately from irrational nominal intervals. The controls have different
attachment costs; equal degree is not equal preparation energy.

If support is supplied from preparation, this cost is already part of the
initial storage. If it is added later, the actual jump needs an external
work supply. No continuous loss is reused as an event reservoir. Neither
this work identity nor successful maintenance selects the attachment event,
forms the parent autonomously or establishes physical binding.

## 8. Executable evidence and limits

The [detached owner](../../src/tnfr/physics/_repeated_organized_composition.py)
rebuilds the fixed weighted geometry, full rows and work from primitive
form/lift inputs. It encloses coefficients and finite analytic bounds with
shared outward rational arithmetic. Source admission reconstructs the two
parent reports from primitive allowances; cached verdicts are not premises.
Within one static observation, identical admitted phase-gap intervals share
their sine/cosine evaluations. This bounded local reuse preserves every
outward-rounded operation and is discarded before the next observation.
The [independent controls](../../tests/physics/test_repeated_organized_composition.py)
compare direct weighted edge sums, both grouping stages, the port-association
witness, bootstrap/resolution boundaries and original-source admission.

The accepted advance is finite predictive composition for perturbed organized
parents, together with a concrete limit on discarded port information.
It neither selects the sine law nor transfers this result to the native
runtime, arbitrary events, other interfaces or indefinite horizons.
Constituent formation, supplied parent preparation, conditional maintenance,
autonomous assembly and physical identification remain distinct.
The separate [parent-acquisition theorem](ORGANIZED_PARENT_ACQUISITION.md)
admits entry from four independently uncertain acquired children under the
matching control, then predicts from the actual reached state. Its longer
source-specific horizon retains correlations absent from the generic residual
ball here; it does not extend this theorem by changing only its clock or radius.

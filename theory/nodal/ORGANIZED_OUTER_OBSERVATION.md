# Observing outer winding with unknown child phase origins

<a id="organized-outer-observation"></a>

## Declared state class, probes and observation contract

The [finite response comparison](ORGANIZED_OUTER_RESPONSE.md) distinguishes
two known preparations. This gate asks whether collective form readings can
identify an unknown outer winding without supplying its phases to the inverse.

| Item | Fixed choice and scope |
| --- | --- |
| Complete law | Same C9-by-C5 support with nodes `9*c+j`, 90 unit edges and held unit capacities; `x_tau=(-Lx+gamma*S(theta))/4`, `theta_tau=gamma*Lx/4`, `gamma=1/(1023*pi)`, regular smooth-sine phase domain and no continuous input |
| Clock | `tau=(1023/1024)*t`; elapsed observation clock h begins at the unknown actual source, with both continuous rows retained |
| Admitted form | Every actual signed form satisfies `abs(x_i)<=X=1/100`; neither the fine forms nor their common or child means are supplied to the inverse |
| Admitted phase | `theta[c,j]=2*pi*(j-4)/9+Phi_c+z[c,j]`, with unknown lifted child origins Phi, `sum_j z[c,j]=0`, and every `abs(z[c,j])<=zeta=1e-6`; global phase is arbitrary |
| Outer chart | Each oriented principal child-origin difference `a_c=wrap(Phi_(c+1)-Phi_c)` satisfies `abs(a_c)<=A=7/5`; the five angles are jointly compatible with actual origins, so their sum is `2*pi*w` for unknown `w in {-1,0,1}` |
| Source scope | This is an admitted state class, not a newly derived preparation. Check inclusion of both earlier actual acquired families at tau=200 with all 80 original relative directions; a broader member is not thereby proved acquired |
| Probe collection | Ten separate branches from the same unknown actual state: for each marked child c and sign s in {-1,+1}, add `s*epsilon*(delta[b,c]-1/5)` to every phase of child b, with `epsilon=1/10`; forms, support and capacities are unchanged |
| Baseline | One unprobed continuation of that same source supplies five receiver readings; it is shared between both signed probes of each interface. Eleven matched mathematical branches are assumed, not sequential interventions or an authenticated laboratory cloning protocol |
| Readout | Fixed delay `h=1/100000` in tau; for interface c read the arithmetic mean signed form of receiver child `(c+1) mod 5`. Three ordered vectors of length five contain baseline B, positive P and negative M, giving 15 raw scalar readings |
| Observation error | Each of the 15 raw readings has arbitrary additive error bounded by `eta=1e-15`. Preserve exact common-baseline cancellation: `P-M` has error at most `2*eta`; `P+M-2*B` has error at most `4*eta`; no noise independence is assumed |
| Inverse boundary | Consume only those 15 primitive readings and the frozen known model/protocol. Hidden phases, source reports, sector labels and evaluated response flags are not inputs. Unknown preparation is not reconstructed from an assigned answer |
| Required inference | Bound the actual mean boundary cosine/sine from finite nonlinear responses, infer child-origin angle intervals with the fine-error allowance, and intersect the complete cycle sum with `2*pi*w`. Prove unique identification for every admitted source and allowed readout error, or report a scoped obstruction or unavailable sufficient certificate |
| Candidate meaning | Sector intersections are necessary compatibility conditions, not joint realizability witnesses. A singleton identifies the actual sector only under the declared same-source/state/law premises; empty candidates mean inconsistent records or premises, and unresolved alternatives remain explicit |
| Work | Each probe has its own actual signed jump `W=H_plus-H_minus`; require `abs(W)<2` structural storage units per branch. No cumulative work budget, reservoir or funding law is inferred |
| Identity horizon | Preserve all five internal winding-one cycles and all nine respective outer slot-cycle windings at each jump and throughout `0<=t<=h`. Generic all-future maintenance is outside this broader state-class contract |
| Numerical boundary | Exact rational admission and the existing shared 128-bit outward trigonometric arithmetic; finite nonlinear remainder retains the unknown background and both rows. No new trajectory, frozen-producer replay, amplitude scan or resolution refinement |

Freeze this declaration before evaluating the uniform inverse bounds. Reuse
the complete finite-increment mechanism while preserving the earlier probe's
fixed constants and evaluated result. A phase-origin chart or exact carrier
alone cannot replace the actual fine-error or finite-time obligations.

The positive single-probe onset retains only one combination of sine and
cosine; assess its ambiguity at the level actually proved, without inventing
identical finite histories. The complete observation contract must retain
all five interfaces and the compatibility of all nine fine slot cycles.

This tests observation of a conditional collective identity. It neither
reduces the whole organization to one scalar EPI nor proves autonomous
preparation, arbitrary-scale fractality or experimental physical identity.

The declaration above was committed as `52126ac0` before assessment. The
following argument uses the fixed protocol and exact rational outward
bounds; it evaluates neither a trajectory nor a frozen response producer.

## 1. State, marked cycles and the scope of the unknown source

Write \(\alpha=2\pi/9\), \(\psi_j=\alpha(j-4)\), and
\(L=I_5\otimes L_9+L_5\otimes I_9\). All 45 signed forms and
45 primitive phase coordinates evolve under the declared complete rows.
The origin \(\Phi_c\) is the arithmetic mean of the continuous
lift \(\theta_{c,j}-\psi_j\) over child c. It is not the
argument of that child's circular resultant, which vanishes for the exact
C9 profile. Its residual \(z_{c,j}\) has zero child mean by
definition; these means need not remain constant during the joined flow.

Let \(a_c\) be the admitted principal origin gap on the oriented
interface from c to \(c+1\), cyclically modulo five. At slot j the
corresponding actual lifted edge gap is

\[
 d_{c,j}=a_c+z_{c+1,j}-z_{c,j},\qquad
 |d_{c,j}-a_c|\le2\zeta.
 \tag{1}
\]

The fixed outer chart is unambiguous because \(A+2\zeta<\pi/2\).
The residual differences telescope around each slot cycle, so

\[
 \sum_{c=0}^4d_{c,j}=\sum_{c=0}^4a_c=2\pi w
 \quad\text{for every }j,
 \qquad w\in\{-1,0,1\}.
 \tag{2}
\]

The allowed integer list follows from \(|\sum_c a_c|\le5A=7<4\pi\).
Each internal C9 edge, including its declared closing turn, has gap
\(\alpha+z_{c,j+1}-z_{c,j}\); its cycle sum is exactly
\(2\pi\). Thus the admitted state has five internal winding-one
identities and nine common outer winding identities before any observation.
These are identities of the actual fine cycles, not a replacement by five
independently assigned angle labels.

This wider class includes both actual previously acquired families at
\(\tau=200\). The [source theorem](ORGANIZED_OUTER_WINDING.md)
gives \(\|r\|_2<R<4\cdot10^{-70}\),
\(\|z^{\mathrm{old}}\|_2<Z<7\cdot10^{-70}\), and the
outer-one carrier bounds \(|X_c|<0.001394\) and
\(|\Phi_c-v_c|<0.000022083\), with
\(v_c=4(c-2)/\pi\). Subtracting each actual child's residual
mean is an orthogonal projection, hence the resulting zero-mean fine
residual still has norm at most Z. Incorporate the subtracted mean into the
unknown origin; this does not reset or modify the reached state. The form
bound is then below \(1/100\), the fine phase bound is below
\(\zeta\), and the principal origin gaps remain below \(7/5\).
For the outer-zero control the carrier is zero and the same residual
argument applies, including its nonstationary actual residuals. All eighty
original relative preparation directions are retained. This inclusion does
not derive preparation of every other member of the larger class.

## 2. Finite response and the two boundary quadratures

For each marked interface c define its actual initial boundary moments

\[
 C_c=\frac19\sum_{j=0}^8\cos d_{c,j},\qquad
 S_c=\frac19\sum_{j=0}^8\sin d_{c,j}.
 \tag{3}
\]

Neither moment nor any phase is supplied to the inverse. They are the
quantities bounded from the finite mean-form readings. The positive or
negative child-c probe changes the receiver's incident phase difference
by \(s\epsilon\), where \(s=+1\) or \(-1\); the
other initial incident differences at receiver child \(c+1\) are
unchanged. Forms are unchanged across the event. Direct substitution into
the full form row gives the exact initial probe-minus-baseline mean slope

\[
 A_c^s=\frac\gamma4
       \{s\sin\epsilon\,C_c+(1-\cos\epsilon)S_c\}.
 \tag{4}
\]

The [finite increment proof](ORGANIZED_OUTER_RESPONSE.md#2-finite-nonlinear-increment-under-the-complete-rows)
uses heat contraction, \(\|L\|_\infty=8\), degree four,
\(\|S(\theta)\|_\infty\le4\), and the global sine
derivative bound. These are the same full law and support here. Reapply its
argument with the present \(\epsilon,h,X\), including the unknown
background, rather than transferring the earlier evaluated response. With
\(g\ge\gamma\) an outward rational upper bound and
\(2g^2h^2<1/2\), the actual finite increment obeys

\[
 R_c^s=hA_c^s+e_c^s,\qquad |e_c^s|\le E,
 \quad
 E=2g\epsilon h^2+2g^2\epsilon(X+gh)h^2
       +\frac43g^3\epsilon h^3.
 \tag{5}
\]

This bound retains both continuous rows, form diffusion, background phase
motion and nonlinear probe feedback. In particular, the observed endpoint
is not identified with the initial derivative. It applies to every actual
fine state in the admitted class, with either probe sign and every receiver.

For one interface write the three noisy raw readings as \(B,P,M\).
The same unprobed reading B occurs in both differences. Define

\[
 D_C=P-M,\qquad D_S=P+M-2B,
 \qquad b_C=2E+2\eta,\quad b_S=2E+4\eta.
 \tag{6}
\]

Equations (4)-(5) and the arbitrary raw error bounds imply

\[
 \begin{split}
 D_C&=\frac{\gamma h\sin\epsilon}{2}C_c+\delta_C,
       &|\delta_C|&\le b_C,\\
 D_S&=\frac{\gamma h(1-\cos\epsilon)}2S_c+\delta_S,
       &|\delta_S|&\le b_S.
 \end{split}\tag{7}
\]

The odd combination cancels baseline motion and its actual reading error
exactly. The even combination carries twice that error. No independence,
zero-mean noise, stationary baseline or equality of different branches'
later fine states is assumed. All eleven branches share the same initial
actual state; measurements from different unknown preparations do not
satisfy this contract merely because they have equal winding labels.

Let outward intervals enclose \(\gamma\), \(\sin\epsilon\)
and \(1-\cos\epsilon\), and form the positive scale intervals

\[
 K_C=\frac2{[\gamma]h[\sin\epsilon]},\qquad
 K_S=\frac2{[\gamma]h[1-\cos\epsilon]}.
 \tag{8}
\]

Then the moment rectangles are

\[
 \mathcal C_c=K_C(D_C+[-b_C,b_C])\cap[c_0,1],\qquad
 \mathcal S_c=K_S(D_S+[-b_S,b_S])\cap[-1,1],
 \quad c_0\le\cos(A+2\zeta),\quad c_0>0.
 \tag{9}
\]

The lower cosine prior follows from (1), not from an assigned answer.
Empty intersections are inconsistent with the declared premises; they do
not prove that no mathematical state under any model could give the data.

Two further necessary checks reject direct contradictions of the state class.
Heat contraction gives \(\|x(h)\|_\infty\le X+gh\) on every
branch, so each raw reading must have magnitude at most \(X+gh+\eta\).
Rotating the mean boundary phasor by \(-a_c\) puts each constituent
phasor in \([-2\zeta,2\zeta]\). Its real part is at least
\(\cos(2\zeta)>0\), hence
\(\cos(2\zeta)\le |C_c+iS_c|\le1\).
Each computed moment rectangle must therefore intersect this annulus.
These filters retain all actual admissible data. Passing them still does
not prove that one full trajectory realizes an arbitrary supplied record.

## 3. Angle inclusion, cycle consistency and uniform identification

Every phasor in (3) has its argument in
\([a_c-2\zeta,a_c+2\zeta]\), an acute interval. Positive
averaging therefore keeps its nonzero resultant in the same angular cone.
If \(\beta_c=\arg(C_c+iS_c)\), then

\[
 |\beta_c-a_c|\le2\zeta.
 \tag{10}
\]

This argument uses the actual fine boundary deviations. It does not set
them to zero or equate the argument of an average phasor with an arithmetic
mean phase. Since the entire rectangle (9) has positive real part, the
shared outward principal-argument calculation is regular. Define

\[
 \mathcal A_c=
 \left(\arg(\mathcal C_c+i\mathcal S_c)+[-2\zeta,2\zeta]\right)
 \cap[-A,A],\qquad
 \mathcal T=\sum_{c=0}^4\mathcal A_c.
 \tag{11}
\]

Each true \(a_c\) belongs to its interval. Keep precisely those
\(w\in\{-1,0,1\}\) for which the outward enclosure of
\(2\pi w\) intersects \(\mathcal T\). These are necessary
sector candidates. Independent interval corners need not correspond to
one jointly realizable source, so the retained set is not a collection of
constructed witnesses. Nevertheless, under the declared actual-source
premises it contains the true sector; a singleton then identifies that
sector, simultaneously on all nine fine cycles by (2).

A uniform bound shows that every admitted source and every allowed reading
error yields a singleton. For a scale interval K, write \(K^+\) for
its upper endpoint and \(\operatorname{wid}(K)\) for its width.
Set

\[
 \begin{split}
 F_C&=g h[\sin\epsilon]^+/2,&
 F_S&=g h[1-\cos\epsilon]^+/2,\\
 w_C&=2K_C^+b_C+\operatorname{wid}(K_C)(F_C+2b_C)+\rho_{K_C},&
 w_S&=2K_S^+b_S+\operatorname{wid}(K_S)(F_S+2b_S)+\rho_{K_S},\\
 \rho_K&=(2K^++\operatorname{wid}(K)+2)u,&u&=2^{-128}.
 \end{split}\tag{12}
\]

Since \(|C_c|,|S_c|\le1\), (7) gives
\(|D_C|\le F_C+b_C\) and \(|D_S|\le F_S+b_S\).
The product-width inequality therefore bounds the two pre-intersection
moment widths by \(w_C,w_S\), including the outward uncertainty
in the scale intervals. Materializing \(D+[-b,b]\) adds at most u
at each endpoint. Its width is therefore at most \(2b+2u\), and
its maximum magnitude at most \(F+2b+u\). Multiplication by K
followed by endpoint rounding adds the remaining \(2u\), giving
exactly the additional allowance \(\rho_K\). Clipping cannot
increase either width. On
\(C\ge c_0\), \(|S|\le1\),
\(\arg(C+iS)=\arctan(S/C)\), and the elementary derivative
bounds give

\[
 \operatorname{wid}(\mathcal A_c)
 \le d:=\frac{w_S}{c_0}+\frac{w_C}{c_0^2}+4\zeta+10^{-20}.
 \tag{13}
\]

The last term covers the shared argument evaluation and subsequent interval
arithmetic uniformly, not a sampled rounding error. Here \(c_0>1/6\)
and \(|S|\le1\), so \(|S/C|<6\); division adds fewer than
ten grid units to its endpoint enclosure. The shared arctangent reduces
point arguments to magnitude at most \(1/2\). Its 64 alternating
terms have omitted tail at most \((1/2)^{128}/129\); interval
term multiplication contracts by less than \(0.251\), and bounding
each term and accumulated sum rounding gives fewer than \(10^4u\)
of point-enclosure width. At most two pi-based argument transformations,
both endpoints and the remaining additions/clips/cycle sum stay below the
looser total \(10^6u<3\cdot10^{-33}\). The \(10^{-20}\)
allowance in each angle is consequently sufficient. The transcendental
owners and their precision are unchanged.

The fixed outward calculation gives the strict conservative bounds

\[
 \begin{split}
 E&<6.224\cdot10^{-15},&c_0&>0.169965,\\
 w_C&<0.000186022,&w_S&<0.004231968,\\
 d&<0.031343,&\operatorname{wid}(\mathcal T)&<0.156715.
 \end{split}\tag{14}
\]

Adjacent exact sector sums are separated by \(2\pi>6.28\).
The shared 128-bit outward \(2\pi\) intervals retain a separation
greater than 6.28 as well; this checks the actual numerical intersection
rule, not only ideal real arithmetic. Thus (14) precludes two candidates.
True-source inclusion precludes zero candidates. The protocol identifies
the unknown outer winding uniformly on the complete admitted class,
including both signs of winding one and winding zero. It recovers neither
all hidden fine coordinates nor absolute phase origins. A common phase
translation leaves the complete law and every reading unchanged.

For arbitrary supplied records outside the assumed family, the inverse
must retain empty, multiple or otherwise unavailable outcomes explicitly.
An empty set signals inconsistency of records or premises; multiple
necessary candidates are unresolved by this sufficient calculation, not
proof of identical complete histories. The frozen protocol has no provision
for smaller noise, a longer horizon or an amplitude retry after assessment.

## 4. What one positive onset loses

The positive slope alone is proportional to
\(\cos(a-\epsilon/2)\) on a uniform boundary. It is unchanged
by \(a\mapsto\epsilon-a\). This ambiguity occurs within the
admitted state class and can span different cycle sectors. Take zero forms
and zero fine residuals. One state has

\[
 a_0=4/5,\qquad a_1=a_2=a_3=a_4=(2\pi-4/5)/4,
 \qquad w=1.
 \tag{15}
\]

A second has

\[
 a_0=-7/10,\qquad a_1=a_2=a_3=a_4=7/40,
 \qquad w=0.
 \tag{16}
\]

All gaps satisfy \(|a_c|\le7/5\); choose successive lifted
origins to realize them, using the respective closing turn. The two marked
opening gaps obey \(-7/10=\epsilon-4/5\), so the positive
initial receiver slopes are exactly equal. This disproves identification
of winding from that local onset alone. It does not assert equal finite
histories or equal noisy records under every finite observation protocol.
The two signed probes recover both boundary quadratures, while the five
marked interfaces supply the cycle information absent from one local
current. Winding itself is not a sufficient state for predicting all those
responses.

## 5. Actual work and the finite identity horizon

For a marked child c, the supplied jump is constant within each child and
has zero global phase sum. It preserves forms, capacities, support, every
internal gap and the phase mean. Exactly eighteen outer edges change, each
by \(+\epsilon\) or \(-\epsilon\). The actual signed work is

\[
 W=\sum_{e\ \mathrm{changed}}\{\cos d_e-\cos(d_e+\Delta k_e)\}.
 \tag{17}
\]

By the global cosine Lipschitz bound,
\(|W|\le18\epsilon=9/5<2\), independently of its sign. This
accounts for each branch's own work. It supplies no reservoir, cumulative
budget or occurrence law, and credits no prior continuous loss. The same
eleven-branch preparation assumption used in the inference is retained;
this is not a theorem about ten sequential probes on one evolving object.

Across a jump the outer gaps have magnitude at most
\(A+2\zeta+\epsilon\), and each slot-cycle sum is unchanged.
For every probed or unprobed branch, heat contraction and the complete
phase row give, throughout \(0\le t\le h\),

\[
 \|x(t)\|_\infty\le X+gt,\qquad
 |d_e(t)-d_e(0)|\le D:=4g(X+gh)h<1.245\cdot10^{-10}.
 \tag{18}
\]

The frozen bounds imply

\[
 \pi/2-(A+2\zeta+\epsilon+D)>0.070794,
 \qquad \alpha-2\zeta-D>0,
 \qquad \pi/2-(\alpha+2\zeta+D)>0.
 \tag{19}
\]

Every internal edge stays positive and acute, and every outer edge stays
strictly acute. No branch crossing can occur. Consequently all five
internal winding-one cycles and all nine respective outer cycles retain
their identities through every declared observation. This is a finite
identity theorem for the larger state class. The earlier acquired sources'
unprobed capture is a separate result; their earlier smaller-probe work and
capture certificates do not transfer to the present stronger probes.
Generic all-future maintenance or recovery after repeated observations is
not established here.

## 6. Result and computational boundary

Fifteen fixed finite mean-form readings identify the unknown outer winding
under the admitted state, complete law, marked support, known clock and
matched-source assumptions. The bound includes hidden child origins, every
allowed fine phase deviation, arbitrary bounded fine forms, full nonlinear
finite motion and all permitted reading errors. Both previously acquired
organizations lie in this class without replacing their actual states by
limiting profiles. Work and identity have separate finite guarantees.

The [observation implementation](../../src/tnfr/physics/_organized_outer_observation.py)
keeps the public-data inverse separate from hidden-state checks and source
inclusion. The [independent tests](../../tests/physics/test_organized_outer_observation.py)
exercise complete-row coefficients, the one-onset counterexample, signed
work, all marked cycles, raw-error cancellation, invalid domains and the
uniform interval argument. Exact rational/outward arithmetic is shared
with the existing proof owners; no numerical trajectory or frozen producer
is replayed. The earlier finite-response protocol remains unchanged.

This is observation of a conditional collective identity, not derivation
of a laboratory measurement bridge. It neither makes winding a complete
state nor selects the supplied law, support, preparation or probe event.
The organization can be treated as a composite NFR only with its declared
regional identity and retained constituent/boundary state; the result does
not justify collapsing those coordinates into one scalar nodal variable.

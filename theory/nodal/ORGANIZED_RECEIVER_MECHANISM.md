# Interaction at a locally unprepared C9 receiver

<a id="organized-receiver-mechanism"></a>

The declared two-ramp source acquires winding one in both prepared C9s
by structural time 200. The initially unprepared receiver has winding
zero throughout a much longer certified interval: its first possible
branch crossing is later than structural time 16,000,000. Nevertheless,
the actual nonlinear current at time 200 drives its mixed form-phase
coordinate in the direction of the donors' winding profile. Thus the
fast form layer does not copy the missing preparation into the receiver;
any eventual acquisition needs a later nonlinear phase mechanism.

This is a complete-law mechanism and delay result. Eventual receiver
acquisition, donor retention during that later evolution, and joined
full-state capture remain unresolved. The delay is a proved consequence
of the frozen source and law, not a response horizon chosen after a
trajectory. No trajectory or reserved response is evaluated here.

<a id="organized-receiver-source"></a>
## Complete source and its preparation cost

Retain the 27 labelled nodes, three unit C9s and oriented contacts
\((0,1),(1,2),(2,0)\) from the
[joint-acquisition model](ORGANIZED_JOINT_ACQUISITION.md#organized-joint-acquisition-model).
Only its equal control is admitted: each contact joins offsets zero and
one with weight one half. All fine edges are present from preparation;
each node has weighted degree four. Capacities are held at one and
\(\tau=(1023/1024)t\), \(\gamma=1/(1023\pi)\). With the actual
weighted Laplacian \(L\), the complete detached law is
\[
 S_i(\theta)=\sum_jw_{ij}\sin(\theta_j-\theta_i),\qquad
 x'=\tfrac14[-Lx+\gamma S(\theta)],\qquad
 \theta'=\tfrac\gamma4Lx .                                      \tag{1}
\]
There is no input, state reset, support event or evolving conductance.
The law is the supplied weighted sine comparison, not native
unique-neighbor phase pressure or registered operator execution.

Put \(q_j=j-4\), \(\alpha=2\pi/9\), and
\[
 m=\frac{2046}{9}\left(\frac{355}{113}\right)^2,\qquad
 Q_{r,j}=\begin{cases}q_j,&r=0,1,\\0,&r=2.\end{cases}          \tag{2}
\]
The full preparation is
\[
 x(0)=mQ+\eta_x,\quad \theta(0)=\eta_\theta,\quad
 |\eta_{x,r,j}|\le r_x=10^{-10},\quad
 |\eta_{\theta,r,j}|\le r_\theta=10^{-10},\quad
 \sum_j\eta_{x,r,j}=\sum_j\eta_{\theta,r,j}=0 .                 \tag{3}
\]
Every allowed combination of errors is retained. The zero sums in (3)
are separate initial constituent premises; only the global form and
lifted-phase means remain conserved during the connected evolution.
All 54 fine coordinates and the boundary state remain part of the
model. Winding is the sum of principal oriented phase differences around
a marked C9, divided by \(2\pi\), away from a branch boundary. All
three constituents initially have winding zero. The donors start with
form ramps, not previously acquired objects.

The unchanged complete storage and loss are
\[
 H=\tfrac12x^TLx+\sum_{\{i,j\}}w_{ij}[1-\cos(\theta_j-\theta_i)],
 \qquad H'=-\tfrac14\|Lx\|_2^2 .                              \tag{4}
\]
The two internal form ramps contribute \(72m^2\). Their mutual
contact contributes \(18m^2\), and each contact to the zero-form
receiver contributes \(30m^2\). Consequently
\[
 H(mQ,0)=150m^2=72m^2+78m^2,\qquad
 \|Q\|_2^2=120,\qquad \|LQ\|_1=117 .                        \tag{5}
\]
The last identity follows from the three actual Laplacian blocks
\[
\begin{split}
 (LQ)_0&=(-27/2,-7/2,-5/2,-3/2,-1/2,1/2,3/2,5/2,17),\\
 (LQ)_1&=(-17,-5/2,-3/2,-1/2,1/2,3/2,5/2,7/2,27/2),\\
 (LQ)_2&=(7/2,6,4,2,0,-2,-4,-6,-7/2).
\end{split}                                                   \tag{6}
\]
Since \(0\preceq L\preceq8I\), all sources satisfy
\[
 150m^2-117mr_x\le H(x(0),\theta(0))
 \le150m^2+117mr_x+108(r_x^2+r_\theta^2).                     \tag{7}
\]
The nominal cost is approximately \(755122024.11569\) in the
model's structural storage units. Removing one local ramp therefore
does not remove its contact cost. Neither initial storage nor later
dissipation supplies an autonomous preparation or event reservoir.

## The full fast layer has a different endpoint reference

Set
\[
 A=L/4,\quad a=\gamma x,\quad \epsilon=\gamma^2,\quad
 F(\theta)=S(\theta)/4,\quad z=\theta+a .                     \tag{8}
\]
The exact complete rows are
\[
 a'=-Aa+\epsilon F(\theta),\qquad
 \theta'=Aa,\qquad z'=\epsilon F(\theta).                   \tag{9}
\]
The connected-support proof gives a mean-zero gap \(g=1/10\)
for \(A\), while \(|F_i|\le1\) and
\(\|F\|_2\le M=\sqrt{27}\) globally. Hence
\[
 \|a(\tau)\|_2\le
 \gamma e^{-\tau/10}(m\sqrt{120}+r_x\sqrt{27})
                  +10\epsilon\sqrt{27}.                    \tag{10}
\]
Define its time-200 upper bound
\[
 B=\gamma e^{-20}(m\sqrt{120}+r_x\sqrt{27})
                  +10\epsilon\sqrt{27}.                    \tag{11}
\]
The norm bound (10) is at most \(B\) for every \(\tau\ge200\).
The appropriate reference phase is now
\[
 P=\alpha Q=(\alpha q,\alpha q,0),                           \tag{12}
\]
which is flat on the receiver. Integrating the exact mixed row gives
\[
 \|\theta(200)-P\|_2\le b,
 \quad b=\sqrt{120}|\gamma m-\alpha|
       +\sqrt{27}(r_\theta+\gamma r_x)+B
       +200\epsilon\sqrt{27}<1/5000 .                      \tag{13}
\]
In particular, \(B\) is approximately \(0.00000504649136123\),
\(B/\gamma<1/50\), and \(b\) is approximately
\(0.00010696036\). The rational ramp mismatch is retained rather
than setting \(355/113=\pi\).

Each donor's principal internal edge difference at time 200 is
\(\alpha\) plus an error of magnitude at most \(\sqrt2b\),
with its one seam integer fixed as in the winding-one target.
Their sums equal \(2\pi\), so both donor windings are one.
The receiver's real edge differences have magnitude at most
\(\sqrt2b<\pi\), telescope to zero and give winding zero.
These are actual full-state conclusions for every source in (3).
In particular this state has not entered the all-winding-one joined
capture cell. The earlier three-ramp source certificate cannot replace
the missing receiver source in (12).

<a id="organized-receiver-delay"></a>
## No receiver branch crossing throughout the certified delay

The receiver's winding is zero throughout the initial layer as well as
at its endpoint. To prove this without a trajectory, split
\(A=A_I+A_C\) into internal-cycle and contact Laplacians divided
by four. On the equal support they are commuting convolution operators
on \(\mathbb Z_3\times\mathbb Z_9\). Their heat kernels are
nonnegative and preserve constants.

Let \(h_\tau\) be the internal C9 heat kernel. Its nonconstant
Fourier modes decay at least as fast as \(e^{-\tau/10}\), whence
\[
 |h_\tau(j)-1/9|\le(8/9)e^{-\tau/10}.                       \tag{14}
\]
The contact heat kernel started in region two has total mass
\[
 p(\tau)=\tfrac23(1-e^{-3\tau/4})                           \tag{15}
\]
in the two donor regions. This is the exact marginal three-region
heat law; it does not assume independence between region and offset.
Aggregate those donor offsets into a nonnegative C9 kernel of mass
\(p\). Since the forward difference of the ramp is
\(\Delta q=1-9\delta_{\mathrm{seam}}\), (14) bounds every
receiver edge of the unforced transported source by
\[
 |\Delta(e^{-A\tau}Q)_{2,j}|
 \le8p(\tau)e^{-\tau/10}<4 .                               \tag{16}
\]
For completeness, the maximum of
\(e^{-\tau/10}(1-e^{-3\tau/4})\) is
\((15/17)(2/17)^{2/15}<3/4\). The strict inequality is the
integer comparison \(4\,20^{15}<17^{17}\).

Variation of constants in (9) gives the exact full-law identity
\[
 \theta(\tau)=\theta(0)+(I-e^{-A\tau})a(0)
       +\epsilon\int_0^\tau
          (I-e^{-A(\tau-v)})F(\theta(v))\,dv .              \tag{17}
\]
Use (16) for the nominal receiver source, Markov contraction for the
initial errors, and \(|F_i|\le1\) for both integral terms. For
every receiver edge and \(0\le\tau\le200\),
\[
 |\Delta\theta_{2,j}(\tau)|
 <4\gamma m+2r_\theta+4\gamma r_x+4\epsilon\tau
 \le4\gamma m+2r_\theta+4\gamma r_x+800\epsilon<\pi .      \tag{18}
\]
The upper bound is approximately \(2.79260473077\). Thus no
receiver edge can have crossed a principal-phase boundary in that layer.

For later times the integrated mixed row in (9), the zero nominal
receiver value of \(z(0)\), and (10) instead give
\[
 |\Delta\theta_{2,j}(\tau)|
 \le2(r_\theta+\gamma r_x)+\sqrt2B+2\epsilon\tau,
 \qquad \tau\ge200 .                                      \tag{19}
\]
Define the derived threshold
\[
 T_*:=\frac{\pi-2(r_\theta+\gamma r_x)-\sqrt2B}
                  {2\epsilon}.                            \tag{20}
\]
It is approximately \(16224447.0054\), and outward bounds
certify \(T_*>16000000\). Combining (18) and (19), every member
of (3) retains receiver winding zero on \([0,T_*)\). At
\(T_*\) the strict edge estimate ends; the result does not claim
that a crossing occurs then, or at any later time. No corresponding
long-window donor-retention conclusion follows from this receiver bound.

<a id="organized-receiver-memory"></a>
## Nonlinear interaction and a controlled slow comparison

The delay is not a symmetry cancellation of all receiver interaction.
At the reference (12), direct evaluation on the oriented fine support
gives, with \(q=q_j\),
\[
\begin{split}
 S_{0,j}(P)&=\tfrac12\sin\alpha-\sin(\alpha q),\\
 S_{1,j}(P)&=-\tfrac12\sin\alpha-\sin(\alpha q),\\
 S_{2,j}(P)&=(1+\cos\alpha)\sin(\alpha q).
\end{split}                                                  \tag{21}
\]
For example, the receiver sees donor phases at offsets \(j-1,j,j+1\),
with weights one half, one and one half. This gives the last row of
(21), not a prescribed replacement of its own phase by a donor ramp.
The receiver current is positive for \(j=5,\ldots,8\), negative
for \(j=0,\ldots,3\), and zero at the central reference node.

These noncentral signs also hold for the actual state at time 200.
The sine Lipschitz bound and (13) imply
\[
 |S_i(\theta(200))-S_i(P)|\le4\sqrt2 b .                   \tag{22}
\]
The smallest noncentral reference magnitude is
\((1+\cos\alpha)\sin(\pi/9)>3/5\), while
\(4\sqrt2b<1/500\). Thus the exact row \(z'=\epsilon S/4\)
at the end of the layer drives each noncentral receiver coordinate
with the sign of its desired donor ramp. This is a statement about
the mixed-coordinate rate. The actual phase row remains
\(\theta'=Aa\); (21) is not its instantaneous replacement.

The nominal preparation respects the graph involution
\((r,j)\mapsto(1-r\pmod3,8-j)\) combined with sign reversal of
both form and lifted phase. The full law preserves that symmetry.
The all-winding-one target respects it as well, so this symmetry
does not exclude the desired outcome. General errors in (3) need
not obey that nominal symmetry.

For an exact description of the slow mechanism put
\(s=\epsilon\tau\), retaining the same phase lifts and form.
Equation (17) becomes
\[
 \theta(s)=\theta_0+(I-e^{-As/\epsilon})a_0
       +\int_0^s(I-e^{-A(s-v)/\epsilon})F(\theta(v))\,dv .   \tag{23}
\]
This is the actual nonlinear memory equation with its original form
source. Equivalently,
\[
 z_s=F(z-a),\qquad
 a_s=-\epsilon^{-1}Aa+F(z-a),\qquad \theta=z-a .             \tag{24}
\]
The reference phase-descent law \(y_s=F(y)\) is an approximation
to (23), not an independently exact closure of (1).

A finite-window comparison can be admitted without discarding the
initial layer. Let \(y(0)=P\), \(K=2\), \(M=\sqrt{27}\),
\(k=g/\epsilon\), and
\[
 A_0=\gamma(m\sqrt{120}+r_x\sqrt{27}),\qquad
 d_0=\sqrt{120}|\gamma m-\alpha|
          +\sqrt{27}(r_\theta+\gamma r_x).
\]
The global Lipschitz constant \(K\) follows from
\(\|DF\|_2\le\|L\|_2/4\le2\). Subtract the integral
equation for \(y\) from (23), use the mean-zero heat gap on the
source and on \(F\), and apply the scalar Gronwall resolvent. This
gives the full-source bound
\[
 \|\theta(s)-y(s)\|_2
 \le d_0e^{Ks}+A_0e^{-ks}
        +\frac{KA_0+M}{k+K}(e^{Ks}-e^{-ks}).                \tag{25}
\]
Indeed the pre-resolvent error is bounded by
\(d_0+A_0e^{-ks}+(M/k)(1-e^{-ks})\); evaluating its
convolution with \(Ke^{Ks}\) gives (25). Form retains its own
bound (10). A future phase comparison must keep both errors, the
fixed source and clock, and actual joined-capture margins. A large
right-hand side is an unavailable sufficient estimate, not a failure
of acquisition. No reduced trajectory or infinite-time interchange
is used here.

<a id="organized-receiver-work"></a>
## Regional boundary work and the remaining capture obligation

For a marked region let \(L_Ix\) and \(S_I\) contain its internal
cycle terms, while \(L_Cx\) and \(S_C\) contain its incident
contact terms. Restrict these vectors to the region and write
\(\ell=L_Ix\), \(b_C=L_Cx\), \(u=S_I\), \(v=S_C\).
Its internal storage
\[
 H_r=\tfrac12x_r^TL_{I,r}x_r+
           \sum_{\{i,j\}\subset r}[1-\cos(\theta_j-\theta_i)]
\]
satisfies the exact balance
\[
 H_r'=-\tfrac14\|\ell\|^2
       -\tfrac14\ell\cdot b_C
       +\tfrac\gamma4(\ell\cdot v-u\cdot b_C).              \tag{26}
\]
To verify it, use \(\nabla_{x_r}H_r=\ell\),
\(\nabla_{\theta_r}H_r=-u\), and both actual rows in (1).
The internal reciprocal terms cancel. The remaining contact terms
are the regional boundary work; they have no fixed sign and are not
an externally supplied forcing. Adding regional storage alone omits
contact storage. Including that contact storage recovers the complete
loss identity (4).

Equations (21)--(26) specify a causal route for later phase-mediated
change while retaining its form memory and work. They do not yet show
that the receiver reaches winding one before a donor loses its own
identity, or that the complete state reaches the
[joined trapping sublevel](ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md#organized-network-capture-barrier).
That remaining nonlinear basin and transport estimate is the precise
missing obligation. Its resolution must use this two-ramp source;
neither the three-ramp certificate nor an equilibrium tangent supplies it.

The [implementation](../../src/tnfr/physics/_organized_receiver_mechanism.py)
re-admits primitive error allowances, rebuilds the support and complete
source cost, and evaluates the preparation, fast-layer, delay and
current-sign inequalities with shared exact rational and outward interval
owners. It reuses the prepared form-cost and transit kernels also consumed
by the three-ramp assessment. The slow comparison (25) is a proved method
for a future admitted consumer; this assessment does not evaluate it or a
reduced trajectory. Its frozen conclusion uses the original errors in
(3); different admitted errors require fresh inequalities. Failed
sufficient bounds remain unavailable. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md) is the sole
owner of the next research task.

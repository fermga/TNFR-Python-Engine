# TNFR–Riemann: Current Evidence and Scope

**Status**: secondary mathematical comparison program; no proof or disproof of RH,
no derived Hilbert–Pólya bridge, and no autonomous arithmetic NFR mechanism.
**Review**: 2026-09-19. Documentation/source audit; historical measurements were
not rerun for this consolidation.
**Authority**: this note owns the current scope of the arithmetic comparison.
The active research queue remains in
[FIVE_STAGE_EXECUTION_PLAN.md](research/FIVE_STAGE_EXECUTION_PLAN.md).

## 1. Current conclusion

The repository contains finite arithmetic constructions, classical analytic
identities, numerical comparisons and conditional symmetry arguments. They are
useful as reproducibility baselines and examples of information retained or
discarded by an observation. They do not derive prime labels, logarithmic
frequencies, arithmetic carriers or a physical particle ontology from the nodal
equation.

The identity $\dot x_i=\nu_i p_i$ determines a form rate after the capacity and
pressure are specified. It does not select the arithmetic assignment
$\nu_i=\log i$, the phase law $\dot\theta_i=-\log i$, a Hamiltonian, or a
schedule that produces a prime ladder. Positive capacity is not proof of an
oscillator. The current foundations and joint scale obligations are maintained in
[NODAL_PARAMETER_FOUNDATIONS.md](NODAL_PARAMETER_FOUNDATIONS.md) and
[TNFR_SCALE_GEOMETRY_AND_BRIDGE.md](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md).

RH concerns the real parts of **all** nontrivial zeta zeros. Finite critical-line
scans, agreement at known zeros, a target spectrum placed on a diagonal, or a
small residual against those targets do not settle that assertion. The historic
label G4 denotes this unresolved external target; it is not a quantified
fraction of the remaining TNFR development.

## 2. Reusable construction and evidence inventory

| Owner | What is supplied | Supported result | Boundary |
|---|---|---|---|
| [von_mangoldt.py](../src/tnfr/riemann/von_mangoldt.py) | Prime labels, integer ladder depth, weights $\log p$ | Finite weighted trace; classical infinite Dirichlet identity on $\Re s>1$ | Finite cutoff is not the full analytic function |
| [prime_ladder_hamiltonian.py](../src/tnfr/riemann/prime_ladder_hamiltonian.py) | Frequencies $k\log p$, selected Hamiltonian coefficients | Decoupled real diagonal spectrum and finite trace readback | Frequencies are assigned; ladder edges are not an executed REMESH history |
| [analytic_continuation.py](../src/tnfr/riemann/analytic_continuation.py) | Classical zeta evaluator and known-zero comparisons | Numerical evaluation of the established meromorphic continuation | Does not derive continuation or establish that every zero is on the critical line |
| [weil_explicit_formula.py](../src/tnfr/riemann/weil_explicit_formula.py) | Test functions, cutoffs and prime/zero data | Finite residual comparison for a classical explicit formula | A reported small residual is not a uniform remainder estimate or a new proof |
| [li_keiper.py](../src/tnfr/riemann/li_keiper.py) | Finite supplied critical-line zeros or line-restricted peak coordinates | Truncated conjugate-pair sums | No certified zero-sum tail; signs are not an independent zero-location test |
| [structural_zero_density.py](../src/tnfr/riemann/structural_zero_density.py) | Classical Riemann–Siegel theta function | Smooth counting targets from a declared scalar inversion | Target construction does not emerge from nodal evolution |
| [admissible_rescaling.py](../src/tnfr/riemann/admissible_rescaling.py) | Positive source and target spectra, eigenvectors | Finite spectral congruence identity | Does not independently predict targets or establish an analytic range/kernel decomposition |
| [nodal_pulse.py](../src/tnfr/riemann/nodal_pulse.py) | Integer labels, amplitudes, logarithmic phase rates, cutoff | Finite interference sum and known-ordinate comparisons | No certified critical-line Dirichlet-series limit or autonomous graph evolution |
| [pulse_coherence.py](../src/tnfr/riemann/pulse_coherence.py) | Finite pulse or classical analytic zeta evaluation | Phase/counting and rectified-phase diagnostics | Pulse, analytic evaluator and supplied comparison oracle must be distinguished |
| [dirichlet_l.py](../src/tnfr/riemann/dirichlet_l.py) and twisted modules | Explicit Dirichlet characters and arithmetic data | Analogous finite/classical GL(1) comparisons | No GRH result or elliptic-curve GL(2) construction follows |

### Reproduction entry points

The [example catalog](../examples/README.md) owns runnable usage. These finite
comparisons exercise the constructions above under their declared inputs:

| Example | Comparison |
| --- | --- |
| [Von Mangoldt trace](../examples/03_riemann_zeta/41_von_mangoldt_zeta_demo.py) | Truncated prime-ladder sum against the classical evaluator on its declared domain |
| [Analytic continuation](../examples/03_riemann_zeta/42_riemann_zeros_as_resonances.py) | Selected known critical-line ordinates and poles of the supplied classical function |
| [Prime-ladder Hamiltonian](../examples/03_riemann_zeta/43_prime_ladder_hamiltonian_demo.py) | Eigenvalues against the spectrum assigned to the diagonal |
| [Weil explicit formula](../examples/03_riemann_zeta/44_weil_explicit_formula_demo.py) | Finite residuals under chosen test functions and cutoffs |
| [Li–Keiper sums](../examples/03_riemann_zeta/45_li_keiper_demo.py) | Truncated sums from supplied zeros, with the input-dependence boundary below |
| [Spectral rescaling](../examples/03_riemann_zeta/57_admissible_rescaling_demo.py) | Supplied smooth targets and the finite congruence residual |

Record the environment, source, inputs, cutoffs and residuals for each run.
Historical tolerances are not guarantees for a new configuration; no examples
were rerun for this consolidation. These checks provide comparison baselines,
not an exhaustion of possible nodal constructions or a novelty claim for
classical identities. The old atlas remains recoverable through the
[retirement record](research/archive/README.md#recovery-records).

## 3. Exact arithmetic identities and their domains

The classical identity is

$$
-\frac{\zeta'(s)}{\zeta(s)}
=\sum_{n\ge1}\Lambda(n)n^{-s}
=\sum_{p}\sum_{k\ge1}\log(p)\,e^{-sk\log p},
\qquad \Re s>1.
$$

The implemented finite trace uses a finite prime set and ladder depth.
Convergence of the infinite series in this half-plane is distinct from the
meromorphic continuation outside it. At a zero $\rho$ of multiplicity $m_\rho$,
the continued function $-\zeta'/\zeta$ has residue $-m_\rho$; at the pole $s=1$
of zeta its residue is $+1$. Writing an arbitrary nontrivial zero as
$\rho=1/2+it$ would assume RH. Critical-line examples only use selected zeros
already known or supplied on that line.

The decoupled P14 finite matrix with diagonal $k\log p$ has that spectrum by
construction. With diagonal trace weights $\log p$, its finite exponential
trace is precisely the finite sum above. This is a useful consistency identity;
reading assigned entries back as eigenvalues is not independent evidence that
the nodal law generated prime arithmetic.

### An inserted spectral threshold is not an independent prediction

The retired prime-path family was `H(sigma)=L+(sigma-1/2)D`, with symmetric
positive-semidefinite graph Laplacian `L`, `L*1=0`, and positive diagonal
`D=diag(log(p_i))`. Its lowest-eigenvalue sign changes exactly at the inserted
`1/2` for every finite size: at that value the potential vanishes; above it
`H` is positive definite; below it the Rayleigh quotient of `1` is negative.
This works for any positive diagonal, with no prime-distribution theorem.
The former asymptotic claim supplies no critical-line or generative-transition
evidence. Eigenvector-variation readouts of that prototype are not primitive
circular phase or the canonical tetrad.

### Finite zero sums and the Li criterion

The implemented P16 evaluator forms only

$$
\lambda_n^{(K)}=2\operatorname{Re}\sum_{k=1}^{K}
\left[1-\left(1-\frac{1}{\rho_k}\right)^n\right].
$$

The complete classical Li criterion concerns all zeros with the required
summation convention and every positive integer index. The finite evaluator
supplies neither a certified omitted-zero tail nor a bound on its numerical
rounding. A sign of $\lambda_n^{(K)}$ is therefore not by itself a certified
sign of the complete coefficient, even for one index.

There is also an input-dependence boundary. If $\rho=1/2+it$ is supplied, then
$|(\rho-1)/\rho|=1$, and its paired contribution is
$2[1-\cos(n\arg(1-1/\rho))]\geq0$ in exact arithmetic, whether or not $t$
is actually a zero ordinate. The default known-zero list and the optional
critical-line scan both use this real part; the latter explicitly constructs
`complex(0.5, detected_ordinate)`. Nonnegative truncated sums consequently do
not validate zero location, pole detection or a TNFR-derived resonance law.
Negative numerical output would require investigating the supplied coordinates
and arithmetic before drawing any conclusion about the analytic criterion.

## 4. Finite pulse and analytic phase are different objects

The evaluator forms

$$
P_N(T)=\sum_{n=1}^{N} n^{-1/2}e^{-iT\log n}.
$$

The ordinary infinite Dirichlet series for zeta is not valid at
$\Re s=1/2$. No statement equates this unregularized finite sum with
$\zeta(1/2+iT)$, and the default cutoff has no certified remainder bound
making it a convergent critical-line approximation. Frequency and amplitude
are declared arithmetic inputs. This evaluator does not execute its optional
graph helper or the shared nodal integrator.

The known-zero tuple in the source supplies comparison ordinates, scan
endpoints and nearest-dip matches. These checks are not blind predictions.
A dip of a finite trigonometric sum need not be an analytic zero; a phase
branch or a zero resultant also needs its own availability treatment.

The analytic argument term $S(T)=\pi^{-1}\arg\zeta(1/2+iT)$ requires the
standard continuation/branch convention. Away from zero ordinates, the
classical counting formula relates it to the Riemann–Siegel theta function.
At a zero ordinate, the counting and argument conventions must be stated.
Using an analytic evaluator for $S(T)$ and then checking that classical
formula is not a TNFR derivation.

Finite phase, RMS, peak and off-axis comparisons remain descriptive. They
neither validate grammar U2 nor imply a uniform estimate at unobserved
heights, criticality, a pressure-equilibrium axis or RH.

### A finite residue-spectrum comparison

The [residue-phase instrument](../benchmarks/residue_phase_vs_riemann.py)
compares fifteen supplied primes `p=3 mod 4`, starting at 7. The directed
quadratic-residue circulant is normal but not self-adjoint; its maximum
imaginary spectral magnitude follows the classical `sqrt(p)/(p-1)` formula.
The selected pairing decreases with prime size while the supplied zeta-zero
ordinates increase. The historical Pearson value was approximately `-0.9068`;
it is not a bound on other inputs and was not rerun for this consolidation.

The maintained verdict `GAUSS_SCALE_MATCHED_DIRECT_ORDERING_DIFFERS` concerns
that positional comparison. Complex eigenvalues do not identify primitive
phase motion or zeta ordinates; the result neither excludes other
representations nor derives an analytic Riemann obstruction. A symmetry
restriction for the separate prime-ladder model cannot be transferred without
matching its group action and complete law.

## 5. What the smooth rescaling actually proves

Suppose $H U=U\,\mathrm{diag}(\lambda_i)$, $U^*U=I$ and both the retained
source values $\lambda_i$ and supplied targets $\mu_i$ are positive. Define

$$
F=U\,\mathrm{diag}\!\left(\sqrt{\mu_i/\lambda_i}\right)U^*.
$$

Then

$$
FHF^*=U\,\mathrm{diag}(\mu_i)U^*.
$$

This finite congruence identity holds for **any** such target list. It is not a
similarity transformation preserving the source spectrum. If $U$ spans the
whole space, $F$ is positive and invertible there; if only selected columns
are retained, it is invertible on their range and zero on the complement.

P28 obtains its smooth targets from the classical theta counting function.
P30 transfers those targets to a selected operator by this identity. A smaller
distance to known zero ordinates reports the chosen target approximation; it
does not measure a percentage of RH proved or a fraction of an ontological
mechanism discovered. Optional irrational probe frequencies are configured
test inputs, not newly derived TNFR constants.

The optional P30 amplitude sweep selects its best value using the W1
discrepancy against the same known zero ordinates later reported as its score.
That is in-sample calibration, not held-out prediction. Compatibility flags
named structurally_derived do not certify derivation from nodal dynamics.

No analytic decomposition $F=F_{\rm smooth}\oplus F_{\rm osc}$, identification
of $S(T)$ with a finite Fourier-mask kernel, or REMESH-infinity operator is
established by this construction. The [fixed-delay surrogate result](REMESH_INFINITY_DERIVATION.md)
retains its own hypotheses and does not supply that missing identification.

## 6. Conditional symmetry results that remain useful

[Structural observability](TNFR_STRUCTURAL_OBSERVABILITY.md) owns the exact
fixed-diffusion, Reynolds-projection and pointed-selector results, composition
conditions and nonlinear counterexamples. A proposed arithmetic application
must supply the same group action, full state, support, weights, capacity,
selector and history assumed by that result. Finite pointed probes do not
establish arbitrary-state or arbitrary-word equivariance.

Relabel-invariant spectra can retain arithmetic information; no representation
here places analytic `S(T)` in a finite symmetry complement. The shared method
is useful for testing observation loss, but is not a universal arithmetic no-go
or a claim that all admissible constructions have been exhausted.

## 7. Superseded claims and preserved historical record

The full earlier notebook is recoverable from the Git revision identified by
the [retirement record](research/archive/README.md).
It is a chronological record, not an active task queue or current theorem
authority. Specifically superseded readings include:

- P1–P11's eliminated combinatorial prototype as a live canonical model.
- The June/July remapping as a derivation of autonomous arithmetic pulses,
  a physical substrate, a critical-line attractor or an analytic zeta identity.
- P28/P30 finite rescaling as a solved smooth part of a derived Hilbert–Pólya
  mechanism, or a universal characterization of the remaining RH content.
- CCET, prime-cancellation, tetrad-fixed-sector and lifted-bundle arguments
  promoted from restricted hypotheses or samples to every engine operator,
  state, support or candidate construction.
- Registry closure as completeness of all admissible TNFR transformations;
  Python scalar storage as proof of the physical dimension of EPI.
- Tetrad completeness, a Hilbert state space inferred from a nonnegative
  snapshot functional, and global conservation/convergence inferred from
  grammar or auxiliary harmonic evolution.

Finite artifacts, source names and historical theorem labels are retained for
traceability. Their validity must be assessed from the actual hypotheses and
implementation; a label such as CLOSED, CANONICAL or CERTIFIED is not a proof.
The archive contains superseded instructions and publication plans that are
not current authorization or research priorities.

## 8. Remaining boundary and reuse policy

A new arithmetic bridge would need independently specified carriers,
operators, domains, observables and a theorem connecting them to the analytic
target, with construction inputs separated from scoring oracles. A nodal
emergence claim would additionally need the coupled evolution and a mechanism
producing and maintaining the effective pattern. Neither requirement is
satisfied by assigning arithmetic coordinates.

The existing exact linear algebra, symmetry controls, spectrum provenance
and observation-loss examples can be reused by the main TNFR research.
They do not warrant restarting RH parameter sweeps or the old envelope
catalog. The single active plan decides priorities; the arithmetic and other
Millennium notes are comparison/reference material.

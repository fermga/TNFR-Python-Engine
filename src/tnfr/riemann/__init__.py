"""TNFR-Riemann spectral analysis module.

Provides the discrete TNFR-Riemann operator H^(k)(sigma) = L_k + V_sigma,
spectral proof framework, alternative graph topologies, per-eigenmode
structural field tetrad, complex-s non-Hermitian extension, and discrete
spectral zeta / trace formula analysis.
See theory/TNFR_RIEMANN_RESEARCH_NOTES.md for the full theoretical background.

Sub-modules
-----------
operator
    Graph builders and discrete operator construction.
spectral_proof
    Four-line spectral analysis and integrated assessment.
topology
    Alternative graph topologies and convergence studies.
eigenmode_fields
    Per-eigenmode structural field tetrad (Phi_s, |grad_phi|, K_phi, xi_C).
complex_extension
    P4 complex-s extension: non-Hermitian operator H(s) for s in C.
spectral_zeta
    P5 discrete spectral zeta, heat kernel trace, and Conjecture 10.1.
random_ensemble
    P6 random prime-graph ensembles and RMT universality analysis.
spectral_conservation
    P7 conservation laws and grammar compliance at criticality.
analytical_convergence
    P8 analytical proof of σ* → 1/2 rate via PNT.
functional_equation
    P9 functional equation analog and spectral reflection symmetry.
convergence_proof
    P10 formal machine-verified proof that σ*(k) → 1/2.
zeta_bridge
    P11 explicit bridge between spectral ζ_H(k) and Riemann ζ_R.
von_mangoldt
    P12 TNFR prime-ladder construction reproducing the von Mangoldt
    series  -ζ'(s)/ζ(s) = Σ_n Λ(n) n^{-s}.
analytic_continuation
    P13 analytic continuation of the prime-ladder vM zeta to the
    whole complex plane; Riemann zeros realised as resonance poles.
prime_ladder_hamiltonian
    P14 self-adjoint prime-ladder Hamiltonian whose spectrum and
    weighted spectral trace reproduce the P12 prime-ladder data
    (operational closure of gap G1 in the TNFR-Riemann programme).
weil_explicit_formula
    P15 numerical verification of the Weil-Guinand explicit formula
    using the P14 Hamiltonian for the prime side; operational
    closure of gap G3 (zeros↔spectrum bridge).
li_keiper
    P16 Li-Keiper positivity criterion computed from the TNFR
    resonance spectrum (RH-equivalent diagnostic; not a new gap
    closure, but a TNFR-native witness for RH).
weil_positivity
    P17 Weil-TNFR positivity bridge: tabulates the RH-equivalent
    Weil functional W[σ] = Σ_γ h_σ(γ) against the canonical TNFR
    Lyapunov energy E_TNFR[σ] across a Gaussian-width grid.
    Experimental research diagnostic (does NOT close gap G4 = RH).
alpha_sweep
    P18 admissibility / gauge sweep of α(σ) = W[σ] / E_TNFR[σ]:
    dense σ-grid combined with a gauge family parameterising how
    h_σ is encoded into (ΔNFR, φ, EPI), reusing the gauge-invariant
    Weil functional.  Stress-tests the P17 bridge against
    canonical-mapping ambiguity.
admissible_family_sweep
    P19 admissible-family extension of P18: sweeps α over multiple
    Schwartz-even test families (not only Gaussian), together with
    gauge and σ grids.
nodeaware_gauge_sweep
    P20 node-aware gauge extension: sweeps α with gauges depending on
    local structural frequency ν_f and node-weight channels.
coercivity_uniform
    P22 empirical uniform-coercivity certificate over sigma intervals,
    combining P19 and P20 alpha surfaces with mesh-corrected lower
    bound diagnostics. P23 adds stratified and segment-local interval
    bounds; P24 adds adaptive sigma refinement that bisects worst
    segments under the local Lipschitz envelope to tighten
    interval_lb_local near the coercivity bottleneck.
paley_gap_coercivity
    P25 Paley-gap coercivity diagnostic in the style of Martínez
    Gamo, Zenodo 10.5281/zenodo.17665853 v2 (2025). Defines three
    Paley-gaps between the P12 closed form, the P14 spectral trace,
    and the classical von Mangoldt truncation. The cross gap
    g_cross = |Z_P14 - Z_P12| collapses to machine precision at
    coupling = 0 (Paley-style identity between the closed-form
    construction and the self-adjoint operator realisation); at
    coupling > 0 it measures structural deformation. Consistency
    diagnostic; does NOT close gap G4.
hilbert_polya
    P27 Hilbert-Polya scaffold. Constructs the reference operator
    T_HP = diag(gamma_1, ..., gamma_N) on ell^2_N(N) populated by
    the imaginary parts of the non-trivial Riemann zeros from
    mpmath.zetazero. Certifies self-adjointness, trace-class
    shifted resolvent, Weil-Guinand closure against P14, and
    quantifies the operator-level gap G4 via Wasserstein-1
    distance between spec(P14) = {k log p} and spec(T_HP) =
    {gamma_n}. Does NOT prove RH: T_HP is populated by inputting
    the zeros, not derived from TNFR first principles.
structural_zero_density
    P28 Structural derivation of the smooth Riemann zero density.
    Derives the n-th smooth zero position ~gamma_n by Newton-solving
    Backlund's smooth counting function bar N(T) = theta(T)/pi + 1,
    where theta(T) = Im log Gamma(1/4 + iT/2) - (T/2) log pi is the
    Riemann-Siegel theta function -- exactly the archimedean kernel
    of the Weil-Guinand formula already computed by P15. Constructs
    the structural Hilbert-Polya operator tilde T_HP =
    diag(~gamma_1, ..., ~gamma_N) using ONLY TNFR archimedean
    ingredients (no mpmath.zetazero on the derivation side) and
    shows that W_1(spec(tilde T_HP), spec(T_HP)) is orders of
    magnitude smaller than the P27 P14<->T_HP gap. The residuals
    r_n = gamma_n - ~gamma_n encode the oscillating part
    S(T) = (1/pi) arg zeta(1/2 + iT) -- the RH content. Closes
    the structural origin of the smooth zero density; does NOT
    close G4 (bounding S(T) is the open arithmetic problem).
spectral_emergence
    P29 Spectral universality emergence under canonical UM+RA
    coupling. Sweeps three canonical inter-prime coupling laws
    (Kuramoto-U3, phi-multiscale THOL+REMESH, PNT-logarithmic RA)
    on the P14 prime-ladder Hamiltonian and measures the
    Kolmogorov-Smirnov distance of the resulting unfolded
    nearest-neighbour spacing distribution to the GUE Wigner
    surmise (the conjectural universality class of the Riemann
    zeros after Montgomery 1973 / Odlyzko 1987). Exploratory
    diagnostic; convergence to GUE under some canonical law would
    constitute structural-compatibility evidence; does NOT close
    gap G4 (RH localisation on Re(s)=1/2).
lyapunov_spectral_positivity
    P26 Lyapunov-spectral positivity certificate for the P14
    Hamiltonian. Combines (i) exact diagonal positivity at coupling=0
    with explicit gap log(2), (ii) a quantitative Kato-Rellich
    perturbation envelope guaranteeing strict positivity for
    |J_0| * ||H_coupling|| < log(2), (iii) trace-class resolvent
    Schatten norms, and (iv) numerical certification of the unitary
    flow exp(-i t H). Closes the operator-level positivity question
    on the finite-dimensional prime-ladder Hilbert space; does NOT
    close gap G4 (RH localisation on Re(s)=1/2).
admissible_rescaling
    P30 Candidate admissible spectral-rescaling operator
    F_cand built ONLY from canonical TNFR ingredients
    (P14 eigendata + P28 smooth zero positions +
    canonical constants phi, gamma, pi, e). Constructs the
    smooth half of F_cand as the explicit operator
    F_smooth = U * diag(sqrt(tilde_gamma_i / lambda_i)) * U^*
    in the P14 eigenbasis; verifies self-adjointness of
    F_smooth H_P14 F_smooth^* and that its spectrum equals
    {tilde_gamma_i} exactly. Lifts the P28 density-level
    closure of the smooth zero distribution to an
    operator-level explicit unitary-rescaling object. Tests
    one canonical oscillatory enrichment (phi-modulated
    multiplicative perturbation) and reports the W_1 gap to
    the true zeros honestly (typically NOT an improvement;
    structural evidence for branch B2 of section 13octies).
    Closes sub-problem (1) of Conjecture T-HP for the
    smooth half only; does NOT close gap G4 = RH.
dirichlet_l
    P32 First L-function extension: chi-twisted TNFR prime-ladder
    spectrum reproducing the twisted von Mangoldt series
    -L'(s, chi) / L(s, chi) = Sum_n chi(n) Lambda(n) n^{-s} for any
    Dirichlet character chi mod q. Structural analogue of P12 with
    weights w_{p,k} = chi(p)^k log p; primes p | q drop out of the
    spectrum (chi(p) = 0). Provides canonical real-character builders
    chi mod 3, mod 4, mod 5 (recovering Dirichlet beta function as a
    special case). Structural extension of the canonical TNFR-Riemann
    representation catalog; does NOT advance the generalised Riemann
    Hypothesis (GRH is RH-equivalent in every L-function and inherits
    the same arithmetic obstruction as G4 for zeta).
analytic_continuation_dirichlet
    P33 Analytic continuation of the chi-twisted prime-ladder L-series
    (structural analogue of P13 for general Dirichlet L-functions).
    Provides high-precision continuation of L(s, chi) and -L'(s,
    chi)/L(s, chi) to all of C via mpmath.dirichlet, plus a
    numerical certificate that the P32 chi-twisted prime ladder
    agrees with the continuation on Re(s) > 1 and a critical-line
    scan locating non-trivial zeros of L(s, chi) as resonance poles
    on Re(s) = 1/2. Tested against LMFDB-tabulated first zeros of
    L(s, chi_3) and L(s, chi_4) = Dirichlet beta(s) with exact
    agreement at three-decimal resolution. Structural extension of
    the P13 continuation construct; does NOT advance GRH (same
    arithmetic obstruction as G4 for zeta).
twisted_prime_ladder_hamiltonian
    P34 Canonical TNFR Hamiltonian on the chi-twisted prime-ladder
    graph (structural analogue of P14 for general Dirichlet
    L-functions; operational closure of gap G1_chi). Instantiates the
    canonical InternalHamiltonian on the disjoint union of per-prime
    REMESH echo ladders for primes coprime to the conductor q, then
    exposes the diagonal chi-twisted weight operator W^(chi) with
    entries W[(p,k),(p,k)] = chi(p)^k log p (complex for non-real
    chi). In the decoupled limit the Hamiltonian spectrum matches the
    P32 chi-twisted prime-ladder eigenvalues exactly, and the
    chi-twisted weighted spectral trace Tr(W^(chi) exp(-s H_freq))
    reproduces the P32 twisted Dirichlet trace -L'(s,chi)/L(s,chi)
    to machine precision. Does NOT advance G4 = RH or GRH (same
    arithmetic obstruction); does NOT establish the chi-twisted
    Weil-Guinand explicit formula (that is the future P35, G3_chi).
twisted_weil_explicit_formula
    P35 Weil-Guinand explicit formula for general primitive Dirichlet
    L-functions L(s, chi) (structural analogue of P15; operational
    closure of gap G3_chi). Verifies the identity
        sum_gamma h(gamma)
           = -g(0) log(q/pi)
             + (1/2 pi) integral h(t) Re psi(1/4 + a/2 + i t/2) dt
             + (-2 Re sum_n chi(n) Lambda(n) / sqrt(n) g(log n))
    using the P34 Hamiltonian for the prime side (-2 Re Tr(W^(chi)
    exp(-H/2) g(H))) and Hardy-Z bisection on the critical line for
    the zero side. Implemented for primitive real characters
    (chi_3, chi_4, chi_5). Closes G3_chi operationally for every
    Dirichlet L-function; does NOT advance G4 = RH or GRH (the
    identity holds for arbitrary zero locations on Re(s) = 1/2).
twisted_li_keiper
    P36 chi-twisted Li-Keiper positivity criterion (structural
    analogue of P16 for primitive real Dirichlet L-functions;
    GRH_chi-equivalent diagnostic). Computes the twisted Li-Keiper
    coefficients
        lambda_n(chi) = sum_rho [1 - (1 - 1/rho)^n]
    where rho runs over non-trivial zeros of L(s, chi) on the
    critical line, sourced via the P35 Hardy-Z bisection enumerator
    (find_dirichlet_l_zeros). Checks positivity of every
    lambda_n(chi) for n = 1, ..., n_max -- Lagarias' generalisation
    of Li 1997 makes positivity GRH_chi-equivalent. Implemented for
    primitive real characters (chi_3, chi_4, chi_5). Provides a
    TNFR-native finite diagnostic witness for GRH on each L(s, chi);
    does NOT prove GRH (a finite positivity check is necessary but
    not sufficient) and does NOT advance G4 for zeta or the
    arithmetic obstruction of GRH.
twisted_weil_positivity
    P37 chi-twisted Weil-TNFR positivity bridge (structural analogue
    of P17 for primitive real Dirichlet L-functions). Two operations:
    (1) chi-twisted Weil positivity certificate, computing
        W_chi[sigma] = 2 sum_{gamma > 0} h_sigma(gamma)
    over zeros of L(s, chi) two ways -- via the P35 Hardy-Z bisection
    enumerator (zero side) and via the chi-twisted Weil-Guinand
    explicit formula with the P34 chi-twisted prime-ladder
    Hamiltonian for the prime side -- and checking W_chi[sigma] >= 0;
    (2) chi-twisted TNFR Lyapunov bridge alpha_chi(sigma) :=
        W_chi[sigma] / E_TNFR_chi[sigma]
    tabulated across a Gaussian width grid using a canonical
    structural test state on the P34 graph. Implemented for primitive
    real characters (chi_3, chi_4, chi_5). GRH_chi-equivalent
    diagnostic on a finite Gaussian grid; does NOT prove GRH for any
    L(s, chi) and does NOT advance G4 = RH.
twisted_alpha_sweep
    P38 chi-twisted admissibility / gauge sweep of
        alpha_chi(sigma; g) = W_chi[sigma] / E_TNFR_chi[sigma; g]
    across a Gaussian width grid and the canonical six-gauge family
    DEFAULT_GAUGES inherited from the zeta-track P18 stress test.
    Structural analogue of P18 for primitive real Dirichlet
    L-functions: probes robustness of the P37 chi-twisted positivity
    bridge under canonical-mapping ambiguity by computing
    alpha_chi(sigma; g) over (sigma, gauge) and reporting aggregate
    flags (W_chi positivity, alpha_chi positivity, alpha_chi extrema).
    Implemented for primitive real characters (chi_3, chi_4, chi_5).
    Diagnostic only; does NOT prove GRH for any L(s, chi) and does
    NOT advance G4 = RH.
twisted_admissible_family_sweep
    P39 chi-twisted admissible-family + gauge sweep of
        alpha_chi(sigma; f, g) = W_chi[sigma; f] / E_TNFR_chi[sigma; f, g]
    across the three admissible Schwartz-even families inherited
    from the zeta-track P19 (gaussian, gaussian_mixture,
    hermite2_gaussian) crossed with the six canonical structural
    gauges DEFAULT_GAUGES from P18. Structural analogue of P19 for
    primitive real Dirichlet L-functions: probes robustness of the
    P37 chi-twisted positivity bridge jointly under (i)
    canonical-mapping ambiguity (gauge sweep, P38) and (ii)
    test-profile ambiguity (admissible-family sweep, P19),
    producing a dense (family, gauge, sigma) certificate.
    Implemented for primitive real characters (chi_3, chi_4, chi_5).
    Diagnostic only; does NOT prove GRH for any L(s, chi) and does
    NOT advance G4 = RH.
twisted_nodeaware_gauge_sweep
    P40 chi-twisted node-aware gauge sweep of
        alpha_chi(sigma; f, g) = W_chi[sigma; f] / E_TNFR_chi[sigma; f, g]
    where g varies over the node-aware gauges DEFAULT_NODEAWARE_GAUGES
    inherited unchanged from the zeta-track P20 (nuf_pressure,
    nuf_phase, weight_pressure, mixed_affine). Each gauge depends on
    the per-node normalised structural frequency hat nu_f(n) and
    normalised node-weight hat w(n) = log p / max log p of the P34
    chi-twisted prime-ladder graph. Structural analogue of P20 for
    primitive real Dirichlet L-functions: probes robustness of the
    P37 chi-twisted positivity bridge under node-aware canonical-
    mapping ambiguity crossed with the admissible-family sweep (P19).
    Implemented for primitive real characters (chi_3, chi_4, chi_5).
    Diagnostic only; does NOT prove GRH for any L(s, chi) and does
    NOT advance G4 = RH.
twisted_hermite_family
    P41 chi-twisted Hermite2-Gaussian eta-parameter sweep of
        alpha_chi(sigma; eta, g) = W_chi[sigma; h_{sigma,eta}]
            / E_TNFR_chi[sigma; eta, g]
    where h_{sigma,eta}(t) = (1 + eta (t/sigma)^2) exp(-t^2/(2 sigma^2))
    is the second-order Hermite-Gaussian admissible profile inherited
    from the zeta-track P21 (admissible_family_sweep). P39 fixed
    eta = 0.25; P41 sweeps eta over DEFAULT_HERMITE2_ETAS =
    (0.0, 0.1, 0.25, 0.5, 1.0, 2.0) where eta = 0.0 recovers the
    pure Gaussian baseline. Structural analogue of P21 on the
    chi-twisted bundle: probes robustness of the P37 positivity
    bridge under polynomial-envelope deformation of the test profile.
    Implemented for primitive real characters (chi_3, chi_4, chi_5).
    Diagnostic only; does NOT prove GRH for any L(s, chi) and does
    NOT advance G4 = RH.
twisted_coercivity_uniform
    P42 empirical interval-level chi-twisted uniform-coercivity
    certificate.  Structural analogue of P22 / P23 / P24 on the
    chi-twisted bundle: samples alpha_chi(sigma; f, g) on a dense
    log-spaced sigma grid via P39 (scalar gauges) and P40 (node-aware
    gauges), computes a finite-difference Lipschitz proxy, and lifts
    pointwise positivity to three interval lower bounds (global,
    stratified, segment-local) with optional adaptive bisection of
    the worst segments.  Implemented for primitive real characters
    (chi_3, chi_4, chi_5).  Diagnostic only; does NOT prove GRH for
    any L(s, chi) and does NOT advance G4 = RH.
twisted_paley_gap_coercivity
    P43 chi-twisted Paley-gap consistency diagnostic.  Structural
    analogue of P25 on the chi-twisted bundle: compares three
    representations of -L'(s,chi)/L(s,chi) -- the P32 closed-form
    weighted spectrum, the P34 chi-twisted spectral trace, and the
    classical truncated Dirichlet series sum chi(n) Lambda(n)/n^s --
    via three absolute Paley-gap quantities g_P32, g_P34, g_cross.
    The cross gap vanishes identically at coupling = 0 (Paley-style
    identity between closed-form and self-adjoint realisations);
    non-zero coupling exposes the deformation magnitude.  Implemented
    for primitive real characters (chi_3, chi_4, chi_5).  Diagnostic
    only; does NOT prove GRH for any L(s, chi) and does NOT advance
    G4 = RH.
twisted_lyapunov_spectral_positivity
    P44 chi-twisted Lyapunov-spectral positivity certificate.
    L-track analogue of P26 on the chi-twisted prime-ladder
    Hamiltonian (P34): four-ingredient certificate (self-adjointness,
    strict positivity with explicit Kato-Rellich envelope, trace-class
    resolvent, unitary flow) on the finite-dimensional chi-twisted
    prime-ladder Hilbert space.  The unperturbed gap is
    log(min prime not dividing q): log 2 for chi_3/chi_5, log 3 for
    chi_4.  Implemented for primitive real characters (chi_3, chi_4,
    chi_5).  Diagnostic only; does NOT prove GRH for any L(s, chi)
    and does NOT advance G4 = RH.
twisted_hilbert_polya
    P45 chi-twisted Hilbert-Polya scaffold.  L-track analogue of P27:
    builds the explicit reference operator T_HP^(chi) = diag(gamma_n)
    where gamma_n are positive zeros of L(s, chi) located by Hardy-Z
    bisection, and certifies self-adjointness, trace-class resolvent,
    chi-twisted Weil-Guinand consistency with P34, and Wasserstein-1
    spectral gap against spec(P34 | primes_active).  Implemented for
    primitive real characters (chi_3, chi_4, chi_5).  Diagnostic only;
    does NOT prove GRH for any L(s, chi) and does NOT advance G4 = RH.
twisted_structural_zero_density
    P46 chi-twisted structural zero density.  L-track analogue of P28:
    derives the smooth zero positions tilde gamma_n^(chi) of L(s, chi)
    purely from the archimedean phase theta_chi(T) = Im log Gamma(
    (1/2+a)/2 + iT/2) + (T/2) log(q/pi) of the completed L-function,
    builds tilde T_HP^(chi) = diag(tilde gamma_n^(chi)), and quantifies
    the structural reduction
    W_1(spec(tilde T_HP^(chi)), spec(T_HP^(chi))) vs the P45 baseline
    W_1(spec(P34|p!|q), spec(T_HP^(chi))).  Residuals r_n^(chi) = gamma_n
    - tilde gamma_n encode S_chi(T) = (1/pi) arg L(1/2 + iT, chi).
    Implemented for primitive real characters (chi_3, chi_4, chi_5).
    Diagnostic only; does NOT prove GRH for any L(s, chi) and does NOT
    advance G4 = RH.
twisted_spectral_emergence
    P47 chi-twisted spectral emergence under canonical coupling.
    L-track analogue of P29: sweeps three canonical TNFR inter-prime
    coupling laws (kuramoto_u3, phi_multiscale, pnt_logarithmic) on
    the P34 chi-twisted prime-ladder Hamiltonian with explicit
    chi(p)chi(q) twist factors, then measures Kolmogorov-Smirnov
    distance of the unfolded nearest-neighbour spacing distribution
    to the GUE Wigner surmise (conjectural universality class of the
    non-trivial zeros of L(s, chi)).  Diagnostic only; KS_GUE -> 0
    would constitute structural-compatibility evidence; does NOT
    prove GRH for any L(s, chi) and does NOT close gap G4 = RH.
twisted_admissible_rescaling
    P48 chi-twisted admissible spectral-rescaling operator.  L-track
    analogue of P30: lifts the P46 chi-twisted smooth zero density
    from density level to operator level by constructing the
    canonical diagonal rescaling F^(chi)_smooth = U_P34 *
    diag(sqrt(tilde gamma_i^(chi) / lambda_i)) * U_P34^*.  Verifies
    self-adjointness preservation and exact spectrum match against
    the P46 smooth targets, then measures W_1 gap to true chi-twisted
    L(s, chi) zeros and sweeps the three canonical oscillatory
    enrichments (phi_log, gamma_e, pi_density) honestly per
    character.  Closes sub-problem (1) of Conjecture T-HP for the
    smooth half only, per primitive real character.  G4 = RH and
    GRH_chi BOTH remain OPEN.
oscillatory_correction
    P31 Branch B1 retry: prime-ladder oscillatory correction of the
    P28 / P30 smooth targets. Reconstructs S(T) = pi^{-1} arg zeta(1/2
    + iT) from the canonical TNFR prime-ladder spectrum {(k log p,
    log p)} via the Riemann-von Mangoldt template, then applies the
    Newton step gamma_i = tilde gamma_i - S(tilde gamma_i) / bar N'
    (tilde gamma_i) on the canonical P28 smooth targets. Uses ONLY
    canonical TNFR ingredients (no mpmath.zetazero on the
    construction side). Reports the residual W_1 vs the true Riemann
    zeros honestly. Positive improvement is branch B1 evidence;
    negative improvement corroborates branch B2. Does NOT close gap
    G4 = RH, does NOT close sub-problems (2) and (3) of Conjecture
    T-HP.
twisted_oscillatory_correction
    P49 chi-twisted prime-ladder oscillatory correction.  L-track
    analogue of P31: reconstructs S_chi(T) = pi^{-1} arg L(1/2 + iT,
    chi) from the canonical P34 chi-twisted prime-ladder spectrum
    {(k log p, chi(p)^k log p)} via the chi-twisted Riemann-von
    Mangoldt template, then applies the Newton step gamma_n^(chi) =
    tilde gamma_n^(chi) - S_chi(tilde gamma_n^(chi)) / bar N'_chi(
    tilde gamma_n^(chi)) on the canonical P46 chi-twisted smooth
    targets per primitive real character.  Uses ONLY canonical
    chi-twisted TNFR ingredients (no mpmath chi-zeros on the
    construction side).  Reports the residual W_1 vs the true L(s,
    chi) zeros honestly per character.  Positive improvement is
    branch B1 evidence at the L-track level; negative improvement
    corroborates branch B2.  Closes the final ZETA/L attack-surface
    parity item (P31 -> P49): every canonical zeta-track operator
    from P12 through P31 now has a matching chi-twisted L-track
    counterpart.  Does NOT close G4 = RH, does NOT prove GRH for
    any L(s, chi).
remesh_infinity_residue_split
    P50 function-space diagnostic that lifts the N15 REMESH-infinity
    closure (theory/REMESH_INFINITY_DERIVATION.md) into the
    TNFR-Riemann program.  Splits the canonical P31 oscillatory
    correction signal S_TNFR(T) into its projections on
    range(R_infinity) and ker(R_infinity) using the N15-resonant
    Fourier-mode mask at the canonical pair (tau_l, tau_g) = (4, 8).
    Pre-registered structural prediction: the prime-ladder spectrum
    has Fourier support exclusively at {k log p}, which by Baker's
    theorem on linear independence of logarithms of algebraic
    numbers is disjoint from the N15-resonant rational-multiple-of-
    pi lattice; therefore the canonical reconstruction must lie
    asymptotically in ker(R_infinity).  Verdicts:
    RESIDUE_IN_KER_ONLY (branch B2 evidence at the function-space
    level; corroborates the section 13septies / 13nonies structural
    identification of the T-HP residual obstruction with the
    oscillatory half), RESIDUE_IN_RANGE_ONLY (would refute P31),
    RESIDUE_MIXED (gauge leak in P30 or boundary artefact).  Does
    NOT advance G4 = RH; does NOT close T-HP; does NOT promote any
    new canonical operator beyond the 13-operator catalog.
    Complementary to the section 13vicies-novies graph-iteration-
    matrix tests (which act on EPI-history state vectors): P50 acts
    on a function in H^2(T-axis), a different mathematical object.
"""

from .operator import (
    build_prime_path_graph,
    build_prime_cycle_graph,
    build_prime_star_graph,
    build_prime_complete_graph,
    build_prime_tree_graph,
    build_prime_random_graph,
    build_h_tnfr,
    build_tridiagonal_h_tnfr,
    default_prime_potential,
    build_h_tnfr_complex,
    build_tridiagonal_h_tnfr_complex,
    default_prime_potential_complex,
)
from .topology import (
    # Data structures
    TopologyResult,
    TopologyConvergenceResult,
    # Registry
    TOPOLOGY_BUILDERS,
    # Analysis
    analyze_graph_topology,
    compare_topologies,
    topology_convergence_study,
)
from .spectral_proof import (
    # Data structures
    EquilibriumResult,
    ThermodynamicResult,
    EigenvalueFlowResult,
    SpectralMomentResult,
    TNFRRiemannAssessment,
    # Core
    compute_eigenspectrum,
    compute_eigensystem,
    # Line 1 - Structural Equilibrium
    verify_equilibrium,
    verify_equilibrium_sequence,
    # Line 2 - Thermodynamic Attractor
    compute_analytic_sigma_star,
    compute_frobenius_energy,
    compute_thermodynamic_landscape,
    verify_thermodynamic_convergence,
    # Line 3 - Eigenvalue Flow
    compute_eigenvalue_velocities,
    analyze_eigenvalue_flow,
    # Line 4 - Spectral Moments
    compute_eigenvalue_spacings,
    compute_spectral_moments,
    # Integration
    run_tnfr_riemann_analysis,
)
from .eigenmode_fields import (
    # Data structures
    EigenmodeTetrad,
    EigenmodeFieldAnalysis,
    # Constants
    PHI_S_VON_KOCH_THRESHOLD,
    PHI_S_GOLDEN_THRESHOLD,
    # Core
    compute_eigenmode_tetrad,
    compute_eigenmode_fields_general,
    # Diagnostics
    check_u6_confinement,
    compare_confinement_at_sigma,
)
from .complex_extension import (
    # Data structures
    ComplexEigenResult,
    CriticalLineScan,
    PseudoSpectrumResult,
    ResolventAnalysis,
    ComplexPlaneAnalysis,
    # Constants
    KNOWN_RIEMANN_ZEROS,
    # Core
    compute_complex_eigenspectrum,
    compute_complex_eigensystem,
    analyze_non_hermiticity,
    # Critical line
    scan_critical_line,
    find_eigenvalue_zero_crossings,
    # Pseudo-spectrum & resolvent
    compute_pseudospectrum,
    compute_resolvent_norm,
    analyze_resolvent_along_critical_line,
    # Riemann zero comparison
    compare_with_riemann_zeros,
    # Integration
    run_complex_plane_analysis,
)
from .spectral_zeta import (
    # Data structures
    SpectralZetaResult,
    HeatKernelResult,
    MellinBridgeResult,
    ConjectureTestResult,
    SpectralZetaAnalysis,
    # Constants
    RIEMANN_ZETA_KNOWN_VALUES,
    # Core computation
    compute_positive_eigenvalues,
    compute_spectral_zeta,
    compute_spectral_zeta_derivative,
    # Heat kernel
    compute_heat_kernel_trace,
    compute_partition_function,
    compute_free_energy,
    # Mellin bridge
    verify_mellin_bridge,
    # Conjecture 10.1
    riemann_zeta_approx,
    test_conjecture_10_1,
    test_conjecture_10_1_sequence,
    # Integration
    run_spectral_zeta_analysis,
)
from .random_ensemble import (
    # Data structures
    EnsembleConfig,
    EnsembleSample,
    SpacingStats,
    RMTComparison,
    EnsembleAnalysis,
    # Reference distributions
    goe_wigner_surmise,
    gue_wigner_surmise,
    poisson_spacing_pdf,
    # Ensemble generation
    generate_er_ensemble,
    generate_wigner_ensemble,
    # Spacing statistics
    compute_ensemble_spacings,
    compute_spacing_ratio,
    compute_mean_spacing_ratio,
    compute_level_repulsion_exponent,
    # Long-range statistics
    compute_number_variance,
    compute_spectral_rigidity,
    # RMT comparison
    ks_test_vs_reference,
    classify_ensemble,
    # Integration
    run_rmt_ensemble_analysis,
    rmt_convergence_study,
)
from .spectral_conservation import (
    # Data structures
    EigenmodeConservation,
    ConservationAtSigma,
    ConservationSigmaScan,
    GrammarComplianceResult,
    CriticalConservationAnalysis,
    # Core eigenmode conservation
    compute_spectral_j_phi,
    compute_spectral_j_dnfr,
    compute_eigenmode_conservation,
    # Sigma scan
    scan_conservation_vs_sigma,
    # Grammar compliance
    test_grammar_conservation,
    # Integration
    run_critical_conservation_analysis,
)
from .analytical_convergence import (
    # Data structures
    TelescopingIdentity,
    PNTAsymptoticBound,
    ConvergenceRateBound,
    EffectiveConstantResult,
    AnalyticalConvergenceProof,
    # Telescoping identity (Theorem 1)
    compute_telescoping_trace,
    verify_telescoping_identity,
    # PNT asymptotics (Theorem 2)
    pnt_prime_estimate,
    euler_maclaurin_log_squared_sum,
    pnt_sum_log_squared,
    # Convergence rate (Theorem 3)
    compute_convergence_rate_bound,
    compute_effective_constant,
    analyze_convergence_sequence,
    # Integration
    run_analytical_convergence_proof,
)
from .functional_equation import (
    # Data structures
    SpectralReflection,
    TraceFormulaResult,
    CompletedXiFunction,
    Conjecture12_1Result,
    Conjecture12_2Result,
    LargeKConvergence,
    FunctionalEquationAnalysis,
    # Core — reflection symmetry
    verify_spectral_reflection,
    verify_reflection_sequence,
    # Core — trace formulas
    compute_trace_formulas,
    verify_trace_formula_pnt,
    # Core — completed xi
    compute_completed_xi,
    verify_xi_functional_equation,
    # Conjectures
    test_conjecture_12_1,
    test_conjecture_12_2,
    # Large-k verification
    verify_large_k_convergence,
    # Integration
    run_functional_equation_analysis,
)
from .convergence_proof import (
    # Data structures
    ProofStep,
    DusartVerification,
    ExplicitBoundResult,
    CurvatureGrowthResult,
    CKAsymptoticFit,
    FormalConvergenceProof,
    # Dusart bounds
    dusart_lower_bound,
    dusart_upper_bound,
    verify_dusart_bounds,
    # Proof steps
    prove_bilinear_decomposition,
    prove_telescoping,
    prove_sum_lower_bound,
    prove_convergence_rate,
    prove_explicit_bound,
    prove_curvature_divergence,
    # Explicit bound
    scan_effective_constant,
    compute_explicit_bound_constant,
    # C(k) asymptotics
    fit_ck_asymptotics,
    # Integration
    run_formal_convergence_proof,
)
from .zeta_bridge import (
    # Data structures
    WeylAsymptotic,
    HeatKernelReflection,
    SpectralZetaReflection,
    ScalingLaw,
    PrimeEncoding,
    ZetaBridgeAnalysis,
    # Functions
    compute_weyl_asymptotic,
    compute_heat_kernel_reflection,
    compute_spectral_zeta_reflection,
    extract_scaling_law,
    compute_prime_encoding,
    run_zeta_bridge_analysis,
)
from .von_mangoldt import (
    # Classical helpers
    mangoldt_lambda,
    classical_log_zeta_derivative,
    classical_log_zeta_derivative_matched,
    # Prime-ladder spectrum (P12)
    PrimeLadderSpectrum,
    build_prime_ladder_spectrum,
    tnfr_log_zeta_derivative,
    # Verification
    VonMangoldtReproductionResult,
    verify_von_mangoldt_reproduction,
)
from .analytic_continuation import (
    # Continuation evaluator (P13)
    von_mangoldt_zeta_continued,
    # Agreement on Re(s) > 1
    ContinuationAgreement,
    verify_continuation_agreement,
    # Pole detection on the critical line
    CriticalLinePoleScan,
    scan_critical_line_for_poles,
    # Explicit-formula reconstruction of psi(x)
    ExplicitFormulaResult,
    reconstruct_psi_via_explicit_formula,
    fetch_riemann_zeros,
)
from .prime_ladder_hamiltonian import (
    # Graph + weight operator (P14)
    build_prime_ladder_graph,
    build_prime_ladder_weight_operator,
    # Hamiltonian bundle
    PrimeLadderHamiltonian,
    build_prime_ladder_hamiltonian,
    # Spectral observable
    weighted_spectral_trace,
    # Certificate
    PrimeLadderHamiltonianCertificate,
    verify_hamiltonian_reproduces_prime_ladder,
)
from .weil_explicit_formula import (
    # Test function (P15)
    GaussianTestFunction,
    gaussian_test_function,
    # Individual terms
    weil_pole_side,
    weil_archimedean_integral,
    weil_prime_side_from_hamiltonian,
    weil_zero_side,
    # Certificate + driver
    WeilExplicitFormulaCertificate,
    verify_weil_explicit_formula,
)
from .li_keiper import (
    # P16: Li-Keiper positivity criterion via TNFR resonance spectrum
    li_coefficients_from_zeros,
    LiKeiperCertificate,
    verify_li_keiper_criterion,
)
from .weil_positivity import (
    # P17: Weil-TNFR positivity bridge
    WeilPositivityCertificate,
    WeilTNFRBridgeCertificate,
    build_structural_test_state,
    tnfr_lyapunov_of_test_state,
    verify_weil_positivity,
    verify_weil_tnfr_bridge,
)
from .alpha_sweep import (
    # P18: admissibility / gauge sweep for alpha(sigma)
    GaugeFn,
    DEFAULT_GAUGES,
    AlphaSweepCertificate,
    build_test_state_with_gauge,
    sweep_alpha,
)
from .admissible_family_sweep import (
    # P19: admissible-family sweep (family × gauge × sigma)
    AdmissibleTestFunction,
    GaussianMixtureTestFunction,
    gaussian_mixture_test_function,
    Hermite2GaussianTestFunction,
    hermite2_gaussian_test_function,
    FamilyFactory,
    DEFAULT_TEST_FAMILIES,
    build_test_state_from_test_function,
    AdmissibleFamilySweepCertificate,
    sweep_alpha_admissible_family,
)
from .nodeaware_gauge_sweep import (
    # P20: node-aware gauge sweep (nu_f + node weight)
    NodeAwareGaugeFn,
    DEFAULT_NODEAWARE_GAUGES,
    build_test_state_nodeaware,
    NodeAwareGaugeSweepCertificate,
    sweep_alpha_nodeaware,
)
from .coercivity_uniform import (
    # P22: empirical interval-level coercivity certificate
    UniformCoercivityCertificate,
    verify_uniform_coercivity_empirical,
)
from .paley_gap_coercivity import (
    # P25: Paley-gap coercivity diagnostic (Zenodo 17665853 v2 style)
    PaleyGapSweep,
    paley_gap_p12,
    paley_gap_p14,
    paley_gap_cross,
    sweep_paley_gap,
)
from .lyapunov_spectral_positivity import (
    # P26: Lyapunov-spectral positivity certificate for P14
    LyapunovSpectralCertificate,
    compute_spectrum,
    operator_norm,
    kato_rellich_lower_bound,
    resolvent_schatten_norms,
    verify_unitary_flow,
    compute_lyapunov_spectral_certificate,
)
from .hilbert_polya import (
    # P27: Hilbert-Polya scaffold
    HilbertPolyaCertificate,
    fetch_zero_imaginary_parts,
    build_hp_operator,
    verify_hp_self_adjoint,
    hp_resolvent_schatten_norms,
    hp_zero_side_from_operator,
    wasserstein_1_distance,
    structural_gap_p14_vs_hp,
    compute_hilbert_polya_certificate,
)
from .structural_zero_density import (
    # P28: Structural smooth zero density
    StructuralZeroDensityCertificate,
    riemann_siegel_theta,
    smooth_zero_count,
    smooth_zero_density,
    derive_smooth_zero_position,
    build_structural_t_hp,
    compute_structural_zero_density_certificate,
)
from .admissible_rescaling import (
    # P30: Candidate admissible spectral-rescaling operator
    AdmissibleRescalingCertificate,
    extract_positive_spectrum,
    build_smooth_rescaling_operator,
    apply_rescaling,
    verify_self_adjointness_preserved,
    verify_spectrum_match,
    oscillatory_correction_canonical,
    compute_admissible_rescaling_certificate,
)
from .oscillatory_correction import (
    # P31: Prime-ladder oscillatory correction (branch B1 retry)
    OscillatoryCorrectionCertificate,
    prime_ladder_oscillatory_sum,
    apply_oscillatory_correction,
    compute_oscillatory_correction_certificate,
)
from .remesh_infinity_residue_split import (
    # P50: R_infinity residue split of the P31 oscillatory correction
    ResidueSplitCertificate,
    build_resonant_bin_mask,
    split_residue_by_remesh_infinity,
    compute_residue_split_certificate,
)
from .nuf_type_signature import (
    # §13triginta-prima: νf-Type Signature diagnostic (foundational sub-question)
    NufTypeSignatureCertificate,
    compute_nuf_type_signature,
)
from .epi_type_signature import (
    # §13triginta-quarta: EPI-Type Signature diagnostic (foundational sub-question)
    EpiTypeSignatureCertificate,
    compute_epi_type_signature,
)
from .phi_type_signature import (
    # §13triginta-octava: phi-Type Signature diagnostic (foundational sub-question)
    PhiTypeSignatureCertificate,
    compute_phi_type_signature,
)
from .dnfr_type_signature import (
    # §13quadraginta: DeltaNFR-Type Signature diagnostic (foundational sub-question)
    DnfrTypeSignatureCertificate,
    compute_dnfr_type_signature,
)
from .remesh_window_type_signature import (
    # §13quadraginta-tertia: REMESH-window-Type Signature diagnostic (foundational sub-question)
    RemeshWindowTypeSignatureCertificate,
    compute_remesh_window_type_signature,
)
from .delta_phi_max_type_signature import (
    # §13quadraginta-sexta: Delta-Phi-Max-Type Signature diagnostic (foundational sub-question)
    DeltaPhiMaxTypeSignatureCertificate,
    compute_delta_phi_max_type_signature,
)
from .coupling_weights_type_signature import (
    # §13quadraginta-nona: Coupling-Weights-Type Signature diagnostic (B6a)
    CouplingWeightsTypeSignatureCertificate,
    compute_coupling_weights_type_signature,
)
from .tetrad_closure_signature import (
    # §13quinquaginta-secunda: Tetrad-Closure Signature diagnostic (B7a)
    TetradClosureSignatureCertificate,
    compute_tetrad_closure_signature,
)
from .currents_closure_signature import (
    # §13quinquaginta-quarta: Currents-Closure Signature diagnostic (B8a)
    CurrentsClosureSignatureCertificate,
    compute_currents_closure_signature,
)
from .dirichlet_l import (
    # P32: Dirichlet L-function extension (chi-twisted prime ladder)
    DirichletCharacter,
    principal_character,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    TwistedPrimeLadderSpectrum,
    build_twisted_prime_ladder_spectrum,
    tnfr_log_l_derivative,
    classical_log_l_derivative,
    classical_log_l_derivative_matched,
    DirichletLReproductionResult,
    verify_dirichlet_l_reproduction,
)
from .analytic_continuation_dirichlet import (
    # P33: Analytic continuation of chi-twisted prime-ladder L-series
    dirichlet_l_continued,
    dirichlet_log_l_derivative_continued,
    TwistedContinuationAgreement,
    verify_twisted_continuation_agreement,
    DirichletCriticalLinePoleScan,
    scan_critical_line_for_l_poles,
)
from .twisted_prime_ladder_hamiltonian import (
    # P34: Canonical Hamiltonian for chi-twisted prime ladder (G1_chi)
    build_twisted_prime_ladder_graph,
    build_twisted_prime_ladder_weight_operator,
    TwistedPrimeLadderHamiltonian,
    build_twisted_prime_ladder_hamiltonian,
    twisted_weighted_spectral_trace,
    TwistedPrimeLadderHamiltonianCertificate,
    verify_twisted_hamiltonian_reproduces_prime_ladder,
)
from .twisted_weil_explicit_formula import (
    # P35: chi-twisted Weil-Guinand explicit formula (G3_chi)
    character_parity,
    twisted_weil_constant_term,
    twisted_weil_archimedean_integral,
    twisted_weil_prime_side_from_hamiltonian,
    find_dirichlet_l_zeros,
    twisted_weil_zero_side,
    TwistedWeilExplicitFormulaCertificate,
    verify_twisted_weil_explicit_formula,
)
from .twisted_li_keiper import (
    # P36: chi-twisted Li-Keiper positivity criterion (GRH_chi diagnostic)
    twisted_li_coefficients,
    TwistedLiKeiperCertificate,
    verify_twisted_li_keiper_criterion,
)
from .twisted_weil_positivity import (
    # P37: chi-twisted Weil-TNFR positivity bridge (GRH_chi diagnostic)
    TwistedWeilPositivityCertificate,
    TwistedWeilTNFRBridgeCertificate,
    build_twisted_structural_test_state,
    twisted_tnfr_lyapunov_of_test_state,
    verify_twisted_weil_positivity,
    verify_twisted_weil_tnfr_bridge,
)
from .twisted_alpha_sweep import (
    # P38: chi-twisted admissibility / gauge sweep (GRH_chi diagnostic)
    TwistedAlphaSweepCertificate,
    build_twisted_test_state_with_gauge,
    sweep_twisted_alpha,
)
from .twisted_admissible_family_sweep import (
    # P39: chi-twisted admissible-family + gauge sweep (diagnostic)
    TwistedAdmissibleFamilySweepCertificate,
    build_twisted_test_state_from_test_function,
    sweep_twisted_admissible_family,
)
from .twisted_nodeaware_gauge_sweep import (
    # P40: chi-twisted node-aware gauge sweep (diagnostic)
    TwistedNodeAwareGaugeSweepCertificate,
    build_twisted_test_state_nodeaware,
    sweep_twisted_nodeaware_gauge,
)
from .twisted_hermite_family import (
    # P41: chi-twisted Hermite2 eta-parameter sweep (diagnostic)
    DEFAULT_HERMITE2_ETAS,
    TwistedHermite2EtaSweepCertificate,
    sweep_twisted_hermite2_eta,
)
from .twisted_coercivity_uniform import (
    # P42: chi-twisted uniform-coercivity certificate (diagnostic)
    TwistedUniformCoercivityCertificate,
    verify_twisted_uniform_coercivity_empirical,
)
from .twisted_paley_gap_coercivity import (
    # P43: chi-twisted Paley-gap consistency diagnostic
    TwistedPaleyGapSweep,
    sweep_twisted_paley_gap,
    twisted_paley_gap_cross,
    twisted_paley_gap_p32,
    twisted_paley_gap_p34,
)
from .twisted_lyapunov_spectral_positivity import (
    # P44: chi-twisted Lyapunov-spectral positivity certificate
    TwistedLyapunovSpectralCertificate,
    compute_twisted_lyapunov_spectral_certificate,
    twisted_compute_spectrum,
    twisted_kato_rellich_lower_bound,
    twisted_verify_unitary_flow,
)
from .twisted_hilbert_polya import (
    # P45: chi-twisted Hilbert-Polya scaffold
    TwistedHilbertPolyaCertificate,
    compute_twisted_hilbert_polya_certificate,
    fetch_chi_zero_imaginary_parts,
    twisted_hp_zero_side_from_operator,
    twisted_structural_gap_p34_vs_hp,
)
from .twisted_structural_zero_density import (
    # P46: chi-twisted structural zero density (L-track analogue of P28)
    TwistedStructuralZeroDensityCertificate,
    build_twisted_structural_t_hp,
    compute_twisted_structural_zero_density_certificate,
    derive_twisted_smooth_zero_position,
    twisted_smooth_zero_count,
    twisted_smooth_zero_density,
    twisted_theta,
)
from .twisted_spectral_emergence import (
    # P47: chi-twisted spectral emergence under canonical coupling
    TWISTED_CANONICAL_COUPLING_LAWS,
    TwistedInterPrimeCoupling,
    TwistedSpectralEmergenceReport,
    build_twisted_inter_prime_coupling,
    couple_twisted_prime_ladder_hamiltonian,
    twisted_sweep_coupling_strength,
    compute_twisted_spectral_emergence_report,
)
from .twisted_admissible_rescaling import (
    # P48: chi-twisted admissible spectral-rescaling operator
    TwistedAdmissibleRescalingCertificate,
    compute_twisted_admissible_rescaling_certificate,
)
from .twisted_oscillatory_correction import (
    # P49: chi-twisted prime-ladder oscillatory correction
    TwistedOscillatoryCorrectionCertificate,
    apply_twisted_oscillatory_correction,
    compute_twisted_oscillatory_correction_certificate,
    twisted_prime_ladder_oscillatory_sum,
)
from .spectral_emergence import (
    # P29: Spectral universality emergence under canonical UM+RA coupling
    CANONICAL_COUPLING_LAWS,
    InterPrimeCoupling,
    SpectralEmergenceReport,
    build_inter_prime_coupling,
    couple_prime_ladder_hamiltonian,
    unfold_spectrum,
    nearest_neighbour_spacings,
    wigner_surmise_gue_cdf,
    ks_distance_to_gue,
    sweep_coupling_strength,
    compute_spectral_emergence_report,
)

__all__ = [
    # Graph builders
    "build_prime_path_graph",
    "build_prime_cycle_graph",
    "build_prime_star_graph",
    "build_prime_complete_graph",
    "build_prime_tree_graph",
    "build_prime_random_graph",
    "build_h_tnfr",
    "build_tridiagonal_h_tnfr",
    "default_prime_potential",
    # Topology comparison (P2)
    "TopologyResult",
    "TopologyConvergenceResult",
    "TOPOLOGY_BUILDERS",
    "analyze_graph_topology",
    "compare_topologies",
    "topology_convergence_study",
    # Data structures
    "EquilibriumResult",
    "ThermodynamicResult",
    "EigenvalueFlowResult",
    "SpectralMomentResult",
    "TNFRRiemannAssessment",
    # Core
    "compute_eigenspectrum",
    "compute_eigensystem",
    # Line 1 - Structural Equilibrium
    "verify_equilibrium",
    "verify_equilibrium_sequence",
    # Line 2 - Thermodynamic Attractor
    "compute_analytic_sigma_star",
    "compute_frobenius_energy",
    "compute_thermodynamic_landscape",
    "verify_thermodynamic_convergence",
    # Line 3 - Eigenvalue Flow
    "compute_eigenvalue_velocities",
    "analyze_eigenvalue_flow",
    # Line 4 - Spectral Moments
    "compute_eigenvalue_spacings",
    "compute_spectral_moments",
    # Integration
    "run_tnfr_riemann_analysis",
    # Per-eigenmode tetrad (P3)
    "EigenmodeTetrad",
    "EigenmodeFieldAnalysis",
    "PHI_S_VON_KOCH_THRESHOLD",
    "PHI_S_GOLDEN_THRESHOLD",
    "compute_eigenmode_tetrad",
    "compute_eigenmode_fields_general",
    "check_u6_confinement",
    "compare_confinement_at_sigma",
    # Complex-s extension (P4)
    "ComplexEigenResult",
    "CriticalLineScan",
    "PseudoSpectrumResult",
    "ResolventAnalysis",
    "ComplexPlaneAnalysis",
    "KNOWN_RIEMANN_ZEROS",
    "build_h_tnfr_complex",
    "build_tridiagonal_h_tnfr_complex",
    "default_prime_potential_complex",
    "compute_complex_eigenspectrum",
    "compute_complex_eigensystem",
    "analyze_non_hermiticity",
    "scan_critical_line",
    "find_eigenvalue_zero_crossings",
    "compute_pseudospectrum",
    "compute_resolvent_norm",
    "analyze_resolvent_along_critical_line",
    "compare_with_riemann_zeros",
    "run_complex_plane_analysis",
    # Spectral zeta & trace formula (P5)
    "SpectralZetaResult",
    "HeatKernelResult",
    "MellinBridgeResult",
    "ConjectureTestResult",
    "SpectralZetaAnalysis",
    "RIEMANN_ZETA_KNOWN_VALUES",
    "compute_positive_eigenvalues",
    "compute_spectral_zeta",
    "compute_spectral_zeta_derivative",
    "compute_heat_kernel_trace",
    "compute_partition_function",
    "compute_free_energy",
    "verify_mellin_bridge",
    "riemann_zeta_approx",
    "test_conjecture_10_1",
    "test_conjecture_10_1_sequence",
    "run_spectral_zeta_analysis",
    # Random ensemble RMT (P6)
    "EnsembleConfig",
    "EnsembleSample",
    "SpacingStats",
    "RMTComparison",
    "EnsembleAnalysis",
    "goe_wigner_surmise",
    "gue_wigner_surmise",
    "poisson_spacing_pdf",
    "generate_er_ensemble",
    "generate_wigner_ensemble",
    "compute_ensemble_spacings",
    "compute_spacing_ratio",
    "compute_mean_spacing_ratio",
    "compute_level_repulsion_exponent",
    "compute_number_variance",
    "compute_spectral_rigidity",
    "ks_test_vs_reference",
    "classify_ensemble",
    "run_rmt_ensemble_analysis",
    "rmt_convergence_study",
    # Spectral conservation at criticality (P7)
    "EigenmodeConservation",
    "ConservationAtSigma",
    "ConservationSigmaScan",
    "GrammarComplianceResult",
    "CriticalConservationAnalysis",
    "compute_spectral_j_phi",
    "compute_spectral_j_dnfr",
    "compute_eigenmode_conservation",
    "scan_conservation_vs_sigma",
    "test_grammar_conservation",
    "run_critical_conservation_analysis",
    # Analytical convergence proof (P8)
    "TelescopingIdentity",
    "PNTAsymptoticBound",
    "ConvergenceRateBound",
    "EffectiveConstantResult",
    "AnalyticalConvergenceProof",
    "compute_telescoping_trace",
    "verify_telescoping_identity",
    "pnt_prime_estimate",
    "euler_maclaurin_log_squared_sum",
    "pnt_sum_log_squared",
    "compute_convergence_rate_bound",
    "compute_effective_constant",
    "analyze_convergence_sequence",
    "run_analytical_convergence_proof",
    # Functional equation & reflection symmetry (P9)
    "SpectralReflection",
    "TraceFormulaResult",
    "CompletedXiFunction",
    "Conjecture12_1Result",
    "Conjecture12_2Result",
    "LargeKConvergence",
    "FunctionalEquationAnalysis",
    "verify_spectral_reflection",
    "verify_reflection_sequence",
    "compute_trace_formulas",
    "verify_trace_formula_pnt",
    "compute_completed_xi",
    "verify_xi_functional_equation",
    "test_conjecture_12_1",
    "test_conjecture_12_2",
    "verify_large_k_convergence",
    "run_functional_equation_analysis",
    # Formal convergence proof (P10)
    "ProofStep",
    "DusartVerification",
    "ExplicitBoundResult",
    "CurvatureGrowthResult",
    "CKAsymptoticFit",
    "FormalConvergenceProof",
    "dusart_lower_bound",
    "dusart_upper_bound",
    "verify_dusart_bounds",
    "prove_bilinear_decomposition",
    "prove_telescoping",
    "prove_sum_lower_bound",
    "prove_convergence_rate",
    "prove_explicit_bound",
    "prove_curvature_divergence",
    "scan_effective_constant",
    "compute_explicit_bound_constant",
    "fit_ck_asymptotics",
    "run_formal_convergence_proof",
    # Zeta bridge (P11)
    "WeylAsymptotic",
    "HeatKernelReflection",
    "SpectralZetaReflection",
    "ScalingLaw",
    "PrimeEncoding",
    "ZetaBridgeAnalysis",
    "compute_weyl_asymptotic",
    "compute_heat_kernel_reflection",
    "compute_spectral_zeta_reflection",
    "extract_scaling_law",
    "compute_prime_encoding",
    "run_zeta_bridge_analysis",
    # Von Mangoldt construction (P12)
    "mangoldt_lambda",
    "classical_log_zeta_derivative",
    "classical_log_zeta_derivative_matched",
    "PrimeLadderSpectrum",
    "build_prime_ladder_spectrum",
    "tnfr_log_zeta_derivative",
    "VonMangoldtReproductionResult",
    "verify_von_mangoldt_reproduction",
    # Analytic continuation (P13)
    "von_mangoldt_zeta_continued",
    "ContinuationAgreement",
    "verify_continuation_agreement",
    "CriticalLinePoleScan",
    "scan_critical_line_for_poles",
    "ExplicitFormulaResult",
    "reconstruct_psi_via_explicit_formula",
    "fetch_riemann_zeros",
    # Prime-ladder Hamiltonian (P14)
    "build_prime_ladder_graph",
    "build_prime_ladder_weight_operator",
    "PrimeLadderHamiltonian",
    "build_prime_ladder_hamiltonian",
    "weighted_spectral_trace",
    "PrimeLadderHamiltonianCertificate",
    "verify_hamiltonian_reproduces_prime_ladder",
    # Weil-Guinand explicit formula (P15)
    "GaussianTestFunction",
    "gaussian_test_function",
    "weil_pole_side",
    "weil_archimedean_integral",
    "weil_prime_side_from_hamiltonian",
    "weil_zero_side",
    "WeilExplicitFormulaCertificate",
    "verify_weil_explicit_formula",
    # P16: Li-Keiper positivity criterion
    "li_coefficients_from_zeros",
    "LiKeiperCertificate",
    "verify_li_keiper_criterion",
    # P17: Weil-TNFR positivity bridge
    "WeilPositivityCertificate",
    "WeilTNFRBridgeCertificate",
    "build_structural_test_state",
    "tnfr_lyapunov_of_test_state",
    "verify_weil_positivity",
    "verify_weil_tnfr_bridge",
    # P18: admissibility / gauge sweep
    "GaugeFn",
    "DEFAULT_GAUGES",
    "AlphaSweepCertificate",
    "build_test_state_with_gauge",
    "sweep_alpha",
    # P19: admissible-family sweep
    "AdmissibleTestFunction",
    "GaussianMixtureTestFunction",
    "gaussian_mixture_test_function",
    "Hermite2GaussianTestFunction",
    "hermite2_gaussian_test_function",
    "FamilyFactory",
    "DEFAULT_TEST_FAMILIES",
    "build_test_state_from_test_function",
    "AdmissibleFamilySweepCertificate",
    "sweep_alpha_admissible_family",
    # P20: node-aware gauge sweep
    "NodeAwareGaugeFn",
    "DEFAULT_NODEAWARE_GAUGES",
    "build_test_state_nodeaware",
    "NodeAwareGaugeSweepCertificate",
    "sweep_alpha_nodeaware",
    # P22: empirical uniform-coercivity certificate
    "UniformCoercivityCertificate",
    "verify_uniform_coercivity_empirical",
    # P25: Paley-gap coercivity diagnostic
    "PaleyGapSweep",
    "paley_gap_p12",
    "paley_gap_p14",
    "paley_gap_cross",
    "sweep_paley_gap",
    # P26: Lyapunov-spectral positivity certificate (P14)
    "LyapunovSpectralCertificate",
    "compute_spectrum",
    "operator_norm",
    "kato_rellich_lower_bound",
    "resolvent_schatten_norms",
    "verify_unitary_flow",
    "compute_lyapunov_spectral_certificate",
    # P27: Hilbert-Polya scaffold
    "HilbertPolyaCertificate",
    "fetch_zero_imaginary_parts",
    "build_hp_operator",
    "verify_hp_self_adjoint",
    "hp_resolvent_schatten_norms",
    "hp_zero_side_from_operator",
    "wasserstein_1_distance",
    "structural_gap_p14_vs_hp",
    "compute_hilbert_polya_certificate",
    # P28: Structural smooth zero density
    "StructuralZeroDensityCertificate",
    "riemann_siegel_theta",
    "smooth_zero_count",
    "smooth_zero_density",
    "derive_smooth_zero_position",
    "build_structural_t_hp",
    "compute_structural_zero_density_certificate",
    # P30: Admissible spectral-rescaling operator (smooth half of F_cand)
    "AdmissibleRescalingCertificate",
    "extract_positive_spectrum",
    "build_smooth_rescaling_operator",
    "apply_rescaling",
    "verify_self_adjointness_preserved",
    "verify_spectrum_match",
    "oscillatory_correction_canonical",
    "compute_admissible_rescaling_certificate",
    # P31: Prime-ladder oscillatory correction (branch B1 retry)
    "OscillatoryCorrectionCertificate",
    "prime_ladder_oscillatory_sum",
    "apply_oscillatory_correction",
    "compute_oscillatory_correction_certificate",
    # P50: R_infinity residue split of P31 oscillatory correction
    "ResidueSplitCertificate",
    "build_resonant_bin_mask",
    "split_residue_by_remesh_infinity",
    "compute_residue_split_certificate",
    # §13triginta-prima: νf-Type Signature (foundational diagnostic)
    "NufTypeSignatureCertificate",
    "compute_nuf_type_signature",
    # §13triginta-quarta: EPI-Type Signature (foundational diagnostic)
    "EpiTypeSignatureCertificate",
    "compute_epi_type_signature",
    # §13triginta-octava: phi-Type Signature (foundational diagnostic)
    "PhiTypeSignatureCertificate",
    "compute_phi_type_signature",
    # §13quadraginta: DeltaNFR-Type Signature (foundational diagnostic)
    "DnfrTypeSignatureCertificate",
    "compute_dnfr_type_signature",
    # §13quadraginta-tertia: REMESH-window-Type Signature (foundational diagnostic)
    "RemeshWindowTypeSignatureCertificate",
    "compute_remesh_window_type_signature",
    # §13quadraginta-sexta: Delta-Phi-Max-Type Signature (foundational diagnostic)
    "DeltaPhiMaxTypeSignatureCertificate",
    "compute_delta_phi_max_type_signature",
    # §13quadraginta-nona: Coupling-Weights-Type Signature (B6a)
    "CouplingWeightsTypeSignatureCertificate",
    "compute_coupling_weights_type_signature",
    # §13quinquaginta-secunda: Tetrad-Closure Signature (B7a)
    "TetradClosureSignatureCertificate",
    "compute_tetrad_closure_signature",
    # §13quinquaginta-quarta: Currents-Closure Signature (B8a)
    "CurrentsClosureSignatureCertificate",
    "compute_currents_closure_signature",
    # P32: Dirichlet L-function extension (chi-twisted prime ladder)
    "DirichletCharacter",
    "principal_character",
    "real_character_mod_3",
    "real_character_mod_4",
    "real_character_mod_5",
    "TwistedPrimeLadderSpectrum",
    "build_twisted_prime_ladder_spectrum",
    "tnfr_log_l_derivative",
    "classical_log_l_derivative",
    "classical_log_l_derivative_matched",
    "DirichletLReproductionResult",
    "verify_dirichlet_l_reproduction",
    # P33: Analytic continuation of chi-twisted prime-ladder L-series
    "dirichlet_l_continued",
    "dirichlet_log_l_derivative_continued",
    "TwistedContinuationAgreement",
    "verify_twisted_continuation_agreement",
    "DirichletCriticalLinePoleScan",
    "scan_critical_line_for_l_poles",
    # P34: Canonical Hamiltonian for chi-twisted prime ladder (G1_chi)
    "build_twisted_prime_ladder_graph",
    "build_twisted_prime_ladder_weight_operator",
    "TwistedPrimeLadderHamiltonian",
    "build_twisted_prime_ladder_hamiltonian",
    "twisted_weighted_spectral_trace",
    "TwistedPrimeLadderHamiltonianCertificate",
    "verify_twisted_hamiltonian_reproduces_prime_ladder",
    # P35: chi-twisted Weil-Guinand explicit formula (G3_chi)
    "character_parity",
    "twisted_weil_constant_term",
    "twisted_weil_archimedean_integral",
    "twisted_weil_prime_side_from_hamiltonian",
    "find_dirichlet_l_zeros",
    "twisted_weil_zero_side",
    "TwistedWeilExplicitFormulaCertificate",
    "verify_twisted_weil_explicit_formula",
    # P36: chi-twisted Li-Keiper positivity criterion (GRH_chi diagnostic)
    "twisted_li_coefficients",
    "TwistedLiKeiperCertificate",
    "verify_twisted_li_keiper_criterion",
    # P37: chi-twisted Weil-TNFR positivity bridge (GRH_chi diagnostic)
    "TwistedWeilPositivityCertificate",
    "TwistedWeilTNFRBridgeCertificate",
    "build_twisted_structural_test_state",
    "twisted_tnfr_lyapunov_of_test_state",
    "verify_twisted_weil_positivity",
    "verify_twisted_weil_tnfr_bridge",
    # P38: chi-twisted admissibility / gauge sweep (GRH_chi diagnostic)
    "TwistedAlphaSweepCertificate",
    "build_twisted_test_state_with_gauge",
    "sweep_twisted_alpha",
    # P39: chi-twisted admissible-family + gauge sweep (diagnostic)
    "TwistedAdmissibleFamilySweepCertificate",
    "build_twisted_test_state_from_test_function",
    "sweep_twisted_admissible_family",
    # P40: chi-twisted node-aware gauge sweep (diagnostic)
    "TwistedNodeAwareGaugeSweepCertificate",
    "build_twisted_test_state_nodeaware",
    "sweep_twisted_nodeaware_gauge",
    # P41: chi-twisted Hermite2 eta-parameter sweep (diagnostic)
    "DEFAULT_HERMITE2_ETAS",
    "TwistedHermite2EtaSweepCertificate",
    "sweep_twisted_hermite2_eta",
    # P42: chi-twisted uniform-coercivity certificate (diagnostic)
    "TwistedUniformCoercivityCertificate",
    "verify_twisted_uniform_coercivity_empirical",
    # P43: chi-twisted Paley-gap consistency diagnostic
    "TwistedPaleyGapSweep",
    "sweep_twisted_paley_gap",
    "twisted_paley_gap_cross",
    "twisted_paley_gap_p32",
    "twisted_paley_gap_p34",
    # P44: chi-twisted Lyapunov-spectral positivity certificate
    "TwistedLyapunovSpectralCertificate",
    "compute_twisted_lyapunov_spectral_certificate",
    "twisted_compute_spectrum",
    "twisted_kato_rellich_lower_bound",
    "twisted_verify_unitary_flow",
    # P45: chi-twisted Hilbert-Polya scaffold
    "TwistedHilbertPolyaCertificate",
    "compute_twisted_hilbert_polya_certificate",
    "fetch_chi_zero_imaginary_parts",
    "twisted_hp_zero_side_from_operator",
    "twisted_structural_gap_p34_vs_hp",
    # P46: chi-twisted structural zero density (L-track analogue of P28)
    "TwistedStructuralZeroDensityCertificate",
    "build_twisted_structural_t_hp",
    "compute_twisted_structural_zero_density_certificate",
    "derive_twisted_smooth_zero_position",
    "twisted_smooth_zero_count",
    "twisted_smooth_zero_density",
    "twisted_theta",
    # P47: chi-twisted spectral emergence under canonical coupling
    "TWISTED_CANONICAL_COUPLING_LAWS",
    "TwistedInterPrimeCoupling",
    "TwistedSpectralEmergenceReport",
    "build_twisted_inter_prime_coupling",
    "couple_twisted_prime_ladder_hamiltonian",
    "twisted_sweep_coupling_strength",
    "compute_twisted_spectral_emergence_report",
    # P48: chi-twisted admissible spectral-rescaling operator
    "TwistedAdmissibleRescalingCertificate",
    "compute_twisted_admissible_rescaling_certificate",
    # P49: chi-twisted prime-ladder oscillatory correction
    "TwistedOscillatoryCorrectionCertificate",
    "apply_twisted_oscillatory_correction",
    "compute_twisted_oscillatory_correction_certificate",
    "twisted_prime_ladder_oscillatory_sum",
    # P29: Spectral universality emergence under canonical UM+RA coupling
    "CANONICAL_COUPLING_LAWS",
    "InterPrimeCoupling",
    "SpectralEmergenceReport",
    "build_inter_prime_coupling",
    "couple_prime_ladder_hamiltonian",
    "unfold_spectrum",
    "nearest_neighbour_spacings",
    "wigner_surmise_gue_cdf",
    "ks_distance_to_gue",
    "sweep_coupling_strength",
    "compute_spectral_emergence_report",
]

"""Finite arithmetic constructions and scoped spectral comparisons.

This package supplies prime-ladder matrices, finite trace and pulse read-outs,
classical analytic evaluators, supplied-zero comparisons and conditional
linear-algebra diagnostics. Prime labels, logarithmic frequencies, characters,
weights and reference zeros are declared inputs where used. They are not
selected autonomously by the nodal equation or its operator catalog.

The finite pulse in ``nodal_pulse`` is not the ordinary infinite Dirichlet
series for zeta on the critical line. That series requires real part above
one; classical analytic continuation is a separate construction. A spectrum
assembled from known zeros, or agreement with such a reference, supplies no
independent proof of RH or GRH. Finite diagnostic labels do not establish
uniform bounds, physical emergence or a complete graph dynamics.

The maintained evidence inventory and interpretation belong to
``theory/TNFR_RIEMANN_RESEARCH_NOTES.md``. Constitutive and state dependencies
belong to ``theory/NODAL_PARAMETER_FOUNDATIONS.md``; actual graph transport
belongs to ``tnfr.physics.structural_diffusion``. These public exports reuse
their specialized owners and do not create a separate research queue.
"""

from .admissible_family_sweep import (  # P19: admissible-family sweep (family × gauge × sigma)
    DEFAULT_TEST_FAMILIES,
    AdmissibleFamilySweepCertificate,
    AdmissibleTestFunction,
    FamilyFactory,
    GaussianMixtureTestFunction,
    Hermite2GaussianTestFunction,
    build_test_state_from_test_function,
    gaussian_mixture_test_function,
    hermite2_gaussian_test_function,
    sweep_alpha_admissible_family,
)
from .admissible_rescaling import (  # P30: Candidate admissible spectral-rescaling operator
    AdmissibleRescalingCertificate,
    apply_rescaling,
    build_smooth_rescaling_operator,
    compute_admissible_rescaling_certificate,
    extract_positive_spectrum,
    oscillatory_correction_canonical,
    verify_self_adjointness_preserved,
    verify_spectrum_match,
)
from .alpha_sweep import (  # P18: admissibility / gauge sweep for alpha(sigma)
    DEFAULT_GAUGES,
    AlphaSweepCertificate,
    GaugeFn,
    build_test_state_with_gauge,
    sweep_alpha,
)
from .analytic_continuation import (  # Continuation evaluator (P13); Agreement on Re(s) > 1; Pole detection on the critical line; Explicit-formula reconstruction of psi(x)
    ContinuationAgreement,
    CriticalLinePoleScan,
    ExplicitFormulaResult,
    fetch_riemann_zeros,
    reconstruct_psi_via_explicit_formula,
    scan_critical_line_for_poles,
    verify_continuation_agreement,
    von_mangoldt_zeta_continued,
)
from .analytic_continuation_dirichlet import (  # P33: Analytic continuation of chi-twisted prime-ladder L-series
    DirichletCriticalLinePoleScan,
    TwistedContinuationAgreement,
    dirichlet_l_continued,
    dirichlet_log_l_derivative_continued,
    scan_critical_line_for_l_poles,
    verify_twisted_continuation_agreement,
)
from .coercivity_uniform import (  # P22: empirical interval-level coercivity certificate
    UniformCoercivityCertificate,
    verify_uniform_coercivity_empirical,
)
from .dirichlet_l import (  # P32: Dirichlet L-function extension (chi-twisted prime ladder)
    DirichletCharacter,
    DirichletLReproductionResult,
    TwistedPrimeLadderSpectrum,
    build_twisted_prime_ladder_spectrum,
    classical_log_l_derivative,
    classical_log_l_derivative_matched,
    principal_character,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    tnfr_log_l_derivative,
    verify_dirichlet_l_reproduction,
)
from .epi_type_signature import (  # §13triginta-quarta: EPI-Type Signature diagnostic (foundational sub-question)
    EpiTypeSignatureCertificate,
    compute_epi_type_signature,
)
from .hilbert_polya import (  # P27: Hilbert-Polya scaffold
    HilbertPolyaCertificate,
    build_hp_operator,
    compute_hilbert_polya_certificate,
    fetch_zero_imaginary_parts,
    hp_resolvent_schatten_norms,
    hp_zero_side_from_operator,
    structural_gap_p14_vs_hp,
    verify_hp_self_adjoint,
    wasserstein_1_distance,
)
from .li_keiper import (  # P16: Li-Keiper positivity criterion via TNFR resonance spectrum
    LiKeiperCertificate,
    li_coefficients_from_zeros,
    verify_li_keiper_criterion,
)
from .lyapunov_spectral_positivity import (  # P26: Lyapunov-spectral positivity certificate for P14
    LyapunovSpectralCertificate,
    compute_lyapunov_spectral_certificate,
    compute_spectrum,
    kato_rellich_lower_bound,
    operator_norm,
    resolvent_schatten_norms,
    verify_unitary_flow,
)
from .nodal_pulse import (  # Canonical foundation: prime-NFR nodal pulse (re-founded)
    KNOWN_RIEMANN_ZEROS,
    NodalPulseCertificate,
    build_prime_nfr_graph,
    detect_zeros_by_interference,
    first_primes,
    nodal_pulse,
    nodal_pulse_magnitude,
    prime_structural_frequencies,
    verify_nodal_pulse,
)
from .nodeaware_gauge_sweep import (  # P20: node-aware gauge sweep (nu_f + node weight)
    DEFAULT_NODEAWARE_GAUGES,
    NodeAwareGaugeFn,
    NodeAwareGaugeSweepCertificate,
    build_test_state_nodeaware,
    sweep_alpha_nodeaware,
)
from .oscillatory_correction import (  # P31: Prime-ladder oscillatory correction (branch B1 retry)
    OscillatoryCorrectionCertificate,
    apply_oscillatory_correction,
    compute_oscillatory_correction_certificate,
    prime_ladder_oscillatory_sum,
)
from .paley_gap_coercivity import (  # P25: Paley-gap coercivity diagnostic (Zenodo 17665853 v2 style)
    PaleyGapSweep,
    paley_gap_cross,
    paley_gap_p12,
    paley_gap_p14,
    sweep_paley_gap,
)
from .prime_ladder_hamiltonian import (  # Graph + weight operator (P14); Hamiltonian bundle; Spectral observable; Certificate
    PrimeLadderHamiltonian,
    PrimeLadderHamiltonianCertificate,
    build_prime_ladder_graph,
    build_prime_ladder_hamiltonian,
    build_prime_ladder_weight_operator,
    verify_hamiltonian_reproduces_prime_ladder,
    weighted_spectral_trace,
)
from .pulse_coherence import (  # Pulse-phase / coherence attack-surface tooling
    PulseCoherenceCertificate,
    argument_fluctuation,
    coherence_defect,
    generalized_pulse,
    prime_side_fluctuation,
    rectified_pulse,
    verify_pulse_coherence,
    zero_count,
)
from .remesh_infinity_residue_split import (  # P50: legacy names; finite fixed-delay DFT split
    ResidueSplitCertificate,
    build_resonant_bin_mask,
    compute_residue_split_certificate,
    split_residue_by_remesh_infinity,
)
from .remesh_window_type_signature import (  # §13quadraginta-tertia: REMESH-window-Type Signature diagnostic (foundational sub-question)
    RemeshWindowTypeSignatureCertificate,
    compute_remesh_window_type_signature,
)
from .spectral_emergence import (  # P29: Spectral universality emergence under canonical UM+RA coupling
    CANONICAL_COUPLING_LAWS,
    InterPrimeCoupling,
    SpectralEmergenceReport,
    build_inter_prime_coupling,
    compute_spectral_emergence_report,
    couple_prime_ladder_hamiltonian,
    ks_distance_to_gue,
    nearest_neighbour_spacings,
    sweep_coupling_strength,
    unfold_spectrum,
    wigner_surmise_gue_cdf,
)
from .structural_zero_density import (  # P28: Structural smooth zero density
    StructuralZeroDensityCertificate,
    build_structural_t_hp,
    compute_structural_zero_density_certificate,
    derive_smooth_zero_position,
    riemann_siegel_theta,
    smooth_zero_count,
    smooth_zero_density,
)
from .twisted_admissible_family_sweep import (  # P39: chi-twisted admissible-family + gauge sweep (diagnostic)
    TwistedAdmissibleFamilySweepCertificate,
    build_twisted_test_state_from_test_function,
    sweep_twisted_admissible_family,
)
from .twisted_admissible_rescaling import (  # P48: chi-twisted admissible spectral-rescaling operator
    TwistedAdmissibleRescalingCertificate,
    compute_twisted_admissible_rescaling_certificate,
)
from .twisted_alpha_sweep import (  # P38: chi-twisted admissibility / gauge sweep (GRH_chi diagnostic)
    TwistedAlphaSweepCertificate,
    build_twisted_test_state_with_gauge,
    sweep_twisted_alpha,
)
from .twisted_coercivity_uniform import (  # P42: chi-twisted uniform-coercivity certificate (diagnostic)
    TwistedUniformCoercivityCertificate,
    verify_twisted_uniform_coercivity_empirical,
)
from .twisted_hermite_family import (  # P41: chi-twisted Hermite2 eta-parameter sweep (diagnostic)
    DEFAULT_HERMITE2_ETAS,
    TwistedHermite2EtaSweepCertificate,
    sweep_twisted_hermite2_eta,
)
from .twisted_hilbert_polya import (  # P45: chi-twisted Hilbert-Polya scaffold
    TwistedHilbertPolyaCertificate,
    compute_twisted_hilbert_polya_certificate,
    fetch_chi_zero_imaginary_parts,
    twisted_hp_zero_side_from_operator,
    twisted_structural_gap_p34_vs_hp,
)
from .twisted_li_keiper import (  # P36: chi-twisted Li-Keiper positivity criterion (GRH_chi diagnostic)
    TwistedLiKeiperCertificate,
    twisted_li_coefficients,
    verify_twisted_li_keiper_criterion,
)
from .twisted_lyapunov_spectral_positivity import (  # P44: chi-twisted Lyapunov-spectral positivity certificate
    TwistedLyapunovSpectralCertificate,
    compute_twisted_lyapunov_spectral_certificate,
    twisted_compute_spectrum,
    twisted_kato_rellich_lower_bound,
    twisted_verify_unitary_flow,
)
from .twisted_nodeaware_gauge_sweep import (  # P40: chi-twisted node-aware gauge sweep (diagnostic)
    TwistedNodeAwareGaugeSweepCertificate,
    build_twisted_test_state_nodeaware,
    sweep_twisted_nodeaware_gauge,
)
from .twisted_oscillatory_correction import (  # P49: chi-twisted prime-ladder oscillatory correction
    TwistedOscillatoryCorrectionCertificate,
    apply_twisted_oscillatory_correction,
    compute_twisted_oscillatory_correction_certificate,
    twisted_prime_ladder_oscillatory_sum,
)
from .twisted_paley_gap_coercivity import (  # P43: chi-twisted Paley-gap consistency diagnostic
    TwistedPaleyGapSweep,
    sweep_twisted_paley_gap,
    twisted_paley_gap_cross,
    twisted_paley_gap_p32,
    twisted_paley_gap_p34,
)
from .twisted_prime_ladder_hamiltonian import (  # P34: Canonical Hamiltonian for chi-twisted prime ladder (G1_chi)
    TwistedPrimeLadderHamiltonian,
    TwistedPrimeLadderHamiltonianCertificate,
    build_twisted_prime_ladder_graph,
    build_twisted_prime_ladder_hamiltonian,
    build_twisted_prime_ladder_weight_operator,
    twisted_weighted_spectral_trace,
    verify_twisted_hamiltonian_reproduces_prime_ladder,
)
from .twisted_spectral_emergence import (  # P47: chi-twisted spectral emergence under canonical coupling
    TWISTED_CANONICAL_COUPLING_LAWS,
    TwistedInterPrimeCoupling,
    TwistedSpectralEmergenceReport,
    build_twisted_inter_prime_coupling,
    compute_twisted_spectral_emergence_report,
    couple_twisted_prime_ladder_hamiltonian,
    twisted_sweep_coupling_strength,
)
from .twisted_structural_zero_density import (  # P46: chi-twisted structural zero density (L-track analogue of P28)
    TwistedStructuralZeroDensityCertificate,
    build_twisted_structural_t_hp,
    compute_twisted_structural_zero_density_certificate,
    derive_twisted_smooth_zero_position,
    twisted_smooth_zero_count,
    twisted_smooth_zero_density,
    twisted_theta,
)
from .twisted_weil_explicit_formula import (  # P35: chi-twisted Weil-Guinand explicit formula (G3_chi)
    TwistedWeilExplicitFormulaCertificate,
    character_parity,
    find_dirichlet_l_zeros,
    twisted_weil_archimedean_integral,
    twisted_weil_constant_term,
    twisted_weil_prime_side_from_hamiltonian,
    twisted_weil_zero_side,
    verify_twisted_weil_explicit_formula,
)
from .twisted_weil_positivity import (  # P37: chi-twisted Weil-TNFR positivity bridge (GRH_chi diagnostic)
    TwistedWeilPositivityCertificate,
    TwistedWeilTNFRBridgeCertificate,
    build_twisted_structural_test_state,
    twisted_tnfr_lyapunov_of_test_state,
    twisted_tnfr_structural_energy_of_test_state,
    verify_twisted_weil_positivity,
    verify_twisted_weil_tnfr_bridge,
)
from .von_mangoldt import (  # Classical helpers; Prime-ladder spectrum (P12); Verification
    PrimeLadderSpectrum,
    VonMangoldtReproductionResult,
    build_prime_ladder_spectrum,
    classical_log_zeta_derivative,
    classical_log_zeta_derivative_matched,
    mangoldt_lambda,
    tnfr_log_zeta_derivative,
    verify_von_mangoldt_reproduction,
)
from .weil_explicit_formula import (  # Test function (P15); Individual terms; Certificate + driver
    GaussianTestFunction,
    WeilExplicitFormulaCertificate,
    gaussian_test_function,
    verify_weil_explicit_formula,
    weil_archimedean_integral,
    weil_pole_side,
    weil_prime_side_from_hamiltonian,
    weil_zero_side,
)
from .weil_positivity import (  # P17: Weil-TNFR positivity bridge
    WeilPositivityCertificate,
    WeilTNFRBridgeCertificate,
    build_structural_test_state,
    tnfr_lyapunov_of_test_state,
    tnfr_structural_energy_of_test_state,
    verify_weil_positivity,
    verify_weil_tnfr_bridge,
)

__all__ = [
    # === Canonical nodal-pulse foundation (re-founded 2026-07) ===
    "KNOWN_RIEMANN_ZEROS",
    "first_primes",
    "prime_structural_frequencies",
    "nodal_pulse",
    "nodal_pulse_magnitude",
    "detect_zeros_by_interference",
    "build_prime_nfr_graph",
    "NodalPulseCertificate",
    "verify_nodal_pulse",
    # Pulse-phase / coherence attack surface (re-founded)
    "argument_fluctuation",
    "zero_count",
    "generalized_pulse",
    "rectified_pulse",
    "coherence_defect",
    "prime_side_fluctuation",
    "PulseCoherenceCertificate",
    "verify_pulse_coherence",
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
    "tnfr_structural_energy_of_test_state",
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
    # P50: legacy names for the finite fixed-delay DFT split
    "ResidueSplitCertificate",
    "build_resonant_bin_mask",
    "split_residue_by_remesh_infinity",
    "compute_residue_split_certificate",
    # §13triginta-quarta: EPI-Type Signature (foundational diagnostic)
    "EpiTypeSignatureCertificate",
    "compute_epi_type_signature",
    # §13quadraginta-tertia: REMESH-window-Type Signature (foundational diagnostic)
    "RemeshWindowTypeSignatureCertificate",
    "compute_remesh_window_type_signature",
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
    "twisted_tnfr_structural_energy_of_test_state",
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

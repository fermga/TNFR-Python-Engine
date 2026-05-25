r"""TNFR-Riemann P45: chi-twisted Hilbert-Polya scaffold.

L-track analogue of P27 (:mod:`hilbert_polya`).  This module is the
L-function counterpart of the zeta-track Hilbert-Polya scaffold.  It
constructs an explicit chi-twisted reference Hilbert-Polya operator on
a truncated TNFR Hilbert space and certifies its internal consistency
with the rest of the L-track stack (P34 chi-twisted prime-ladder
Hamiltonian, P36 chi-twisted Weil-Guinand explicit formula).

The reference operator is

    T_HP^{(chi)} = diag(gamma_1^{(chi)}, gamma_2^{(chi)}, ..., gamma_N^{(chi)})
                                                              on ell^2_N(N)

where ``gamma_n^{(chi)}`` are the positive imaginary parts of the
non-trivial zeros ``rho_n^{(chi)} = 1/2 + i gamma_n^{(chi)}`` of
``L(s, chi)`` located by **Hardy-Z bisection** (the same enumerator
used by P36 in :mod:`twisted_weil_explicit_formula`).  By construction:

* ``T_HP^{(chi)}`` is self-adjoint (real diagonal).
* For ``s > 0`` the shifted resolvent
  ``(T_HP^{(chi)})^2 + s^2 I)^{-1/2}`` belongs to Schatten class
  ``S_p`` for every ``p > 1``; its trace and Hilbert-Schmidt norms
  are computed exactly from the gamma list.
* The zero-side ``2 sum h(gamma_n^{(chi)})`` of the chi-twisted Weil
  explicit formula evaluated through ``T_HP^{(chi)}`` reproduces P36
  to within the Hardy-Z truncation tolerance, because both routes
  consume the same gamma data.
* The spectral gap between ``spec(T_HP^{(chi)}) = {gamma_n^{(chi)}}``
  and ``spec(P34 | primes_active) = {k log p : p not dividing q}`` is
  quantified by Wasserstein-1 distance on the truncated empirical
  measures.  This number is the L-track operator-level expression of
  the structural derivation gap: the open construction of
  ``T_HP^{(chi)}`` from TNFR first principles.

Honest scope (mandatory, see AGENTS.md):

The P45 module does **not** prove GRH for ``L(s, chi)``.  ``T_HP^{(chi)}``
is populated by *inputting* the zeros from Hardy-Z bisection of the
classical ``L(s, chi)``; we do not derive them from the nodal
equation, conservation, or grammar.  What P45 delivers is the explicit
operator-level slot into which a chi-twisted Hilbert-Polya-style attack
must fit, plus numerical evidence that the L-track stack is internally
compatible with such a slot, for each of the three primitive real
characters ``chi_3``, ``chi_4``, ``chi_5``.  The genuinely open piece
(structural derivation of ``T_HP^{(chi)}`` from TNFR first principles
without reference to the zeros) is **not** addressed here.

This module does **not** contribute to closing gap G4 = RH; it is the
L-track structural mirror of the ζ-track diagnostic scaffold P27.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..mathematics.unified_numerical import np
from .dirichlet_l import DirichletCharacter
from .hilbert_polya import (
    build_hp_operator,
    hp_resolvent_schatten_norms,
    verify_hp_self_adjoint,
    wasserstein_1_distance,
)
from .twisted_prime_ladder_hamiltonian import (
    TwistedPrimeLadderHamiltonian,
    build_twisted_prime_ladder_hamiltonian,
)
from .twisted_weil_explicit_formula import (
    character_parity,
    find_dirichlet_l_zeros,
    twisted_weil_archimedean_integral,
    twisted_weil_constant_term,
    twisted_weil_prime_side_from_hamiltonian,
)
from .weil_explicit_formula import (
    GaussianTestFunction,
    gaussian_test_function,
)


__all__ = [
    "TwistedHilbertPolyaCertificate",
    "fetch_chi_zero_imaginary_parts",
    "twisted_hp_zero_side_from_operator",
    "twisted_structural_gap_p34_vs_hp",
    "compute_twisted_hilbert_polya_certificate",
]


# ----------------------------------------------------------------------
# Atomic primitives (L-track specialisations)
# ----------------------------------------------------------------------


def fetch_chi_zero_imaginary_parts(
    chi: DirichletCharacter,
    n_zeros: int,
    *,
    initial_t_max: float = 30.0,
    initial_step: float = 0.25,
    dps: int = 30,
    max_doublings: int = 6,
) -> np.ndarray:
    r"""Return the first ``n_zeros`` positive imaginary parts of
    ``L(s, chi)`` zeros on the critical line.

    Implementation: adaptive Hardy-Z bisection.  The routine starts
    with ``t_max = initial_t_max`` and doubles the search window up to
    ``max_doublings`` times until at least ``n_zeros`` zeros are
    enumerated, then truncates to the first ``n_zeros``.

    Parameters
    ----------
    chi : DirichletCharacter
        Primitive real character.
    n_zeros : int
        Number of positive-axis zeros to return.
    initial_t_max : float, default 30.0
        Initial upper bound of zero search.
    initial_step : float, default 0.25
        Hardy-Z scan step (in ``t``); must be smaller than the typical
        zero spacing.
    dps : int, default 30
        Mpmath working decimal precision.
    max_doublings : int, default 6
        Maximum number of times the search window is doubled before
        giving up.

    Returns
    -------
    np.ndarray
        Strictly positive ``gamma`` values, length ``n_zeros``,
        ascending.

    Raises
    ------
    ValueError
        If ``n_zeros <= 0``.
    RuntimeError
        If the search window cannot be expanded enough to enumerate
        ``n_zeros`` zeros.
    """
    if n_zeros <= 0:
        raise ValueError("n_zeros must be positive")

    t_max = float(initial_t_max)
    zeros: list[float] = []
    for _ in range(max_doublings + 1):
        zeros = find_dirichlet_l_zeros(
            chi,
            t_min=0.5,
            t_max=t_max,
            initial_step=initial_step,
            dps=dps,
        )
        if len(zeros) >= n_zeros:
            break
        t_max *= 2.0
    if len(zeros) < n_zeros:
        raise RuntimeError(
            f"Could not enumerate {n_zeros} zeros of L(s, {chi.name}) "
            f"within t_max={t_max:.1f} after {max_doublings} doublings; "
            f"only found {len(zeros)} zeros."
        )
    gammas = np.array(zeros[:n_zeros], dtype=float)
    if not np.all(gammas > 0):
        raise RuntimeError(
            "Hardy-Z bisection returned a non-positive imaginary part"
        )
    return gammas


def twisted_hp_zero_side_from_operator(
    gammas: np.ndarray,
    test: GaussianTestFunction,
) -> float:
    r"""Evaluate ``2 sum_n h(gamma_n^{(chi)})`` directly from
    ``diag(T_HP^{(chi)})``.

    For primitive real ``chi`` the complex zeros come in conjugate
    pairs ``rho = 1/2 +/- i gamma``, so the full zero side of the
    chi-twisted Weil explicit formula is twice the sum over positive
    ``gamma``.  This is identical to
    :func:`twisted_weil_zero_side` evaluated on the same gamma list,
    but exposes the dependence as an inner product
    ``Tr h((T_HP^{(chi)})^2)^{1/2}`` against the spectral measure of
    ``T_HP^{(chi)}``.
    """
    gammas = np.asarray(gammas, dtype=float)
    h_values = np.array([test.h(float(g)) for g in gammas], dtype=float)
    return float(2.0 * np.sum(h_values))


def twisted_structural_gap_p34_vs_hp(
    bundle: TwistedPrimeLadderHamiltonian,
    gammas: np.ndarray,
) -> dict:
    r"""Quantify the L-track operator-level structural gap on truncated
    spectra.

    Compares ``spec(P34 | primes_active) = {k log p : p not dividing q}``
    (positive eigenvalues only) with ``spec(T_HP^{(chi)}) = {gamma_n}``
    on the same truncation length.  The Wasserstein-1 distance is the
    relevant scalar because both spectra are real and unbounded with
    different growth: P34 grows like ``log n`` (over the active primes)
    while ``T_HP^{(chi)}`` grows like ``2 pi n / log n``.

    The growth-rate mismatch is the L-track operator-level
    manifestation of the open structural derivation problem.  No
    transformation that sends one spectrum to the other can be a
    smooth structural map; any chi-twisted Hilbert-Polya-style
    derivation must therefore introduce a non-linear spectral
    rescaling derived from TNFR first principles.
    """
    p34_eigs, _ = bundle.hamiltonian.get_spectrum()
    p34_eigs = np.sort(np.real(p34_eigs))
    p34_eigs = p34_eigs[p34_eigs > 0.0]
    n_compare = min(len(p34_eigs), len(gammas))
    p34_trunc = p34_eigs[:n_compare]
    hp_trunc = np.sort(np.asarray(gammas, dtype=float))[:n_compare]
    w1 = wasserstein_1_distance(p34_trunc, hp_trunc)
    if n_compare > 0:
        growth_ratio = float(hp_trunc[-1] / p34_trunc[-1])
    else:
        growth_ratio = float("nan")
    return {
        "n_compared": int(n_compare),
        "p34_min": float(p34_trunc[0]) if n_compare > 0 else float("nan"),
        "p34_max": float(p34_trunc[-1]) if n_compare > 0 else float("nan"),
        "hp_min": float(hp_trunc[0]) if n_compare > 0 else float("nan"),
        "hp_max": float(hp_trunc[-1]) if n_compare > 0 else float("nan"),
        "wasserstein_1": w1,
        "asymptotic_growth_ratio": growth_ratio,
    }


# ----------------------------------------------------------------------
# Certificate dataclass and orchestrator
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedHilbertPolyaCertificate:
    """Internal-consistency certificate for the L-track chi-twisted
    Hilbert-Polya scaffold."""

    # Character
    character_name: str
    character_modulus: int
    character_parity: int

    # Truncation parameters
    n_zeros: int
    n_primes: int
    max_power: int

    # Self-adjointness
    asymmetry_frobenius: float
    self_adjoint: bool

    # Resolvent
    resolvent_shift: float
    schatten_1_norm: float
    schatten_2_norm: float
    operator_norm_inverse: float
    trace_class: bool

    # Weil-Guinand consistency (chi-twisted)
    gaussian_sigma: float
    zero_side_via_hp: float
    constant_term: float
    archimedean_side: float
    prime_side_via_p34: float
    rhs_total: float
    residual: float
    relative_residual: float
    weil_tolerance: float
    weil_verified: bool

    # Operator-level structural gap
    spectral_gap_n_compared: int
    spectral_gap_wasserstein_1: float
    spectral_gap_growth_ratio: float

    # Overall verdict
    scaffold_consistent: bool
    notes: Tuple[str, ...]

    def summary(self) -> str:
        return (
            f"TwistedHilbertPolyaCertificate("
            f"chi={self.character_name}, q={self.character_modulus}, "
            f"a={self.character_parity}, "
            f"n_zeros={self.n_zeros}, primes={self.n_primes}, "
            f"self_adjoint={self.self_adjoint}, "
            f"trace_class={self.trace_class}, "
            f"||R||_1={self.schatten_1_norm:.4e}, "
            f"weil_residual={self.residual:.3e}, "
            f"W_1(P34,HP)={self.spectral_gap_wasserstein_1:.4e}, "
            f"scaffold_consistent={self.scaffold_consistent})"
        )


def compute_twisted_hilbert_polya_certificate(
    chi: DirichletCharacter,
    *,
    n_primes: int = 30,
    max_power: int = 6,
    n_zeros: int = 40,
    gaussian_sigma: float = 2.0,
    resolvent_shift: float = 1.0,
    weil_tolerance: float = 1e-2,
    initial_t_max: float = 30.0,
    initial_step: float = 0.25,
    dps: int = 30,
) -> TwistedHilbertPolyaCertificate:
    r"""Build ``T_HP^{(chi)}`` and certify L-track stack consistency.

    Parameters
    ----------
    chi : DirichletCharacter
        Primitive real character.
    n_primes, max_power : int
        Chi-twisted prime-ladder bundle dimensions; passed to the P34
        builder.  The bundle is built with ``coupling = 0`` so the
        Hamiltonian is exactly diagonal.
    n_zeros : int, default 40
        Length of the gamma list used to populate ``T_HP^{(chi)}``.
    gaussian_sigma : float, default 2.0
        Width of the Gaussian test function used for chi-twisted
        Weil-Guinand.  Smaller than the ζ-track default because the
        Hardy-Z enumeration grows quickly with ``sigma``.
    resolvent_shift : float, default 1.0
        Positive shift ``s`` for ``(T_HP^2 + s^2 I)^{-1/2}``.
    weil_tolerance : float, default 1e-2
        Acceptance tolerance for the chi-twisted Weil-Guinand
        residual.  Looser than the ζ-track default because the zero
        enumeration is truncated at ``t_max = 12 * sigma`` and
        residuals scale with the test-function tail.
    initial_t_max, initial_step, dps : passed to
        :func:`fetch_chi_zero_imaginary_parts`.

    Returns
    -------
    TwistedHilbertPolyaCertificate
    """
    if n_zeros <= 0:
        raise ValueError("n_zeros must be positive")

    bundle = build_twisted_prime_ladder_hamiltonian(
        chi,
        n_primes=n_primes,
        max_power=max_power,
        coupling=0.0,
    )
    gammas = fetch_chi_zero_imaginary_parts(
        chi,
        n_zeros,
        initial_t_max=initial_t_max,
        initial_step=initial_step,
        dps=dps,
    )
    T_hp = build_hp_operator(gammas)

    self_adj = verify_hp_self_adjoint(T_hp)
    resolvent = hp_resolvent_schatten_norms(gammas, shift=resolvent_shift)

    test = gaussian_test_function(gaussian_sigma)
    zero_side = twisted_hp_zero_side_from_operator(gammas, test)
    const = twisted_weil_constant_term(chi, test)
    arch = twisted_weil_archimedean_integral(chi, test)
    prime = twisted_weil_prime_side_from_hamiltonian(bundle, test)
    rhs = const + arch + prime
    residual = abs(zero_side - rhs)
    denom_norm = max(abs(zero_side), abs(rhs), 1.0)
    rel_residual = residual / denom_norm
    weil_ok = residual <= weil_tolerance

    gap = twisted_structural_gap_p34_vs_hp(bundle, gammas)

    scaffold_ok = bool(
        self_adj["self_adjoint"]
        and resolvent["trace_class"]
        and weil_ok
    )

    notes: Tuple[str, ...] = (
        "T_HP^(chi) is populated by inputting Hardy-Z bisection",
        "outputs from the classical L(s, chi); the scaffold does not",
        "derive the zeros from TNFR first principles.",
        "spec(P34|primes_active) grows like log n while spec(T_HP^(chi))",
        "grows like 2*pi*n/log n; the Wasserstein-1 distance reported",
        "below quantifies the L-track operator-level structural gap.",
        "P45 does NOT close GRH for L(s, chi) and does NOT advance G4.",
    )

    return TwistedHilbertPolyaCertificate(
        character_name=str(chi.name),
        character_modulus=int(chi.modulus),
        character_parity=int(character_parity(chi)),
        n_zeros=int(n_zeros),
        n_primes=int(n_primes),
        max_power=int(max_power),
        asymmetry_frobenius=self_adj["asymmetry_frobenius"],
        self_adjoint=bool(self_adj["self_adjoint"]),
        resolvent_shift=resolvent["shift"],
        schatten_1_norm=resolvent["schatten_1_norm"],
        schatten_2_norm=resolvent["schatten_2_norm"],
        operator_norm_inverse=resolvent["operator_norm_inverse"],
        trace_class=bool(resolvent["trace_class"]),
        gaussian_sigma=float(gaussian_sigma),
        zero_side_via_hp=float(zero_side),
        constant_term=float(const),
        archimedean_side=float(arch),
        prime_side_via_p34=float(prime),
        rhs_total=float(rhs),
        residual=float(residual),
        relative_residual=float(rel_residual),
        weil_tolerance=float(weil_tolerance),
        weil_verified=bool(weil_ok),
        spectral_gap_n_compared=int(gap["n_compared"]),
        spectral_gap_wasserstein_1=float(gap["wasserstein_1"]),
        spectral_gap_growth_ratio=float(gap["asymptotic_growth_ratio"]),
        scaffold_consistent=scaffold_ok,
        notes=notes,
    )

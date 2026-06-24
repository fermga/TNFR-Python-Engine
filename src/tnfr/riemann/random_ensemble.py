r"""P6: Random prime-graph ensembles and RMT statistics.

Implements **randomised TNFR operator ensembles** and their comparison
with canonical Random Matrix Theory (RMT) universality classes.

Motivation
----------
The deterministic prime-path operator H^(k)(sigma) = L_k + V_sigma has a
fixed eigenvalue spectrum for each (k, sigma).  Attempting to compare a
*single* spectrum to GUE/GOE statistics is a category error: RMT
universality is a property of **ensemble averages**.

The correct RMT context arises from randomised graph topologies:

1. Fix prime node labels {p_1, ..., p_k}.
2. Randomise the edge structure (Erdos-Renyi, Wigner perturbation).
3. Solve H_TNFR = L_random + V_sigma for each realisation.
4. Average eigenvalue statistics over the ensemble.
5. Test whether ensemble-averaged spacings approach GOE/GUE/Poisson.

This is where Tao-Vu type universality would legitimately apply: random
graph Laplacians with fixed diagonal potentials.

TNFR physics basis
------------------
Dissonance (OZ operator) introduces controlled stochastic perturbation.
Ensemble averaging tests the statistical mechanics of TNFR networks under
random perturbation, connecting to U2 (convergence/boundedness).  The
ensemble spread quantifies structural resilience: networks where grammar
rules are satisfied should exhibit convergent statistics as sample size
grows, while grammar-violating topologies diverge.

Ensemble types
--------------
- **Erdos-Renyi (ER)**: Edge (i,j) included with probability p_edge.
  Laplacian has random structure; V_sigma remains deterministic.
- **Wigner perturbation**: Start from deterministic path-graph Laplacian
  and add a scaled GOE/GUE random matrix.  Models small structural noise
  on top of the canonical TNFR operator.

RMT reference distributions
----------------------------
- **GOE** (beta=1): Real symmetric matrices. Wigner surmise
  P(s) = (pi/2) s exp(-pi s^2/4).
- **GUE** (beta=2): Complex Hermitian matrices. Wigner surmise
  P(s) = (32/pi^2) s^2 exp(-4 s^2/pi).
- **Poisson** (beta=0): Uncorrelated levels. P(s) = exp(-s).

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P6 program.

References
----------
- AGENTS.md: TNFR-Riemann Program Overview, Dissonance (OZ) operator
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md sec. 5-8
- Mehta, *Random Matrices* (3rd ed.)
- Tao & Vu, *Random matrices: universality of local eigenvalue statistics*
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ..mathematics.unified_numerical import np
from .operator import _first_primes, build_h_tnfr
from .spectral_proof import _unfold_eigenvalues, compute_eigenvalue_spacings

if TYPE_CHECKING:
    import networkx as nx

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "EnsembleConfig",
    "EnsembleSample",
    "SpacingStats",
    "RMTComparison",
    "EnsembleAnalysis",
    # Reference distributions
    "goe_wigner_surmise",
    "gue_wigner_surmise",
    "poisson_spacing_pdf",
    # Ensemble generation
    "generate_er_ensemble",
    "generate_wigner_ensemble",
    # Spacing statistics
    "compute_ensemble_spacings",
    "compute_spacing_ratio",
    "compute_mean_spacing_ratio",
    "compute_level_repulsion_exponent",
    # Long-range statistics
    "compute_number_variance",
    "compute_spectral_rigidity",
    # RMT comparison
    "ks_test_vs_reference",
    "classify_ensemble",
    # Integration
    "run_rmt_ensemble_analysis",
    "rmt_convergence_study",
]

# ---------------------------------------------------------------------------
# Constants: RMT reference values
# ---------------------------------------------------------------------------

# Mean spacing ratio <r> = <min(s_i, s_{i+1}) / max(s_i, s_{i+1})>
# Atas et al., PRL 110, 084101 (2013)
GOE_MEAN_RATIO = 0.5307
GUE_MEAN_RATIO = 0.5996
POISSON_MEAN_RATIO = 2.0 * math.log(2.0) - 1.0  # ~0.3863

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for a random prime-graph ensemble.

    Parameters
    ----------
    k : int
        Number of prime nodes.
    n_samples : int
        Number of random realisations.
    sigma : float
        Structural parameter for V_sigma.
    ensemble_type : str
        "erdos_renyi" or "wigner".
    edge_prob : float
        Edge probability for ER ensembles.
    wigner_scale : float
        Noise scale for Wigner perturbation (std of off-diagonal entries).
    seed : int
        Master random seed (Invariant #6).
    weight_by_log_gap : bool
        Use log-prime-gap weights for the base graph edges.
    """

    k: int
    n_samples: int = 100
    sigma: float = 0.5
    ensemble_type: str = "erdos_renyi"
    edge_prob: float = 0.3
    wigner_scale: float = 0.1
    seed: int = 42
    weight_by_log_gap: bool = True


@dataclass
class EnsembleSample:
    """Eigenvalue data for a single ensemble realisation.

    Attributes
    ----------
    eigenvalues : ndarray
        Sorted eigenvalues of H_TNFR for this realisation.
    spacings : ndarray
        Normalised nearest-neighbour spacings (unfolded).
    """

    eigenvalues: np.ndarray
    spacings: np.ndarray


@dataclass
class SpacingStats:
    """Aggregated spacing statistics from an ensemble.

    Attributes
    ----------
    all_spacings : ndarray
        Pooled spacings from every realisation in the ensemble.
    mean_spacing_ratio : float
        Ensemble-averaged spacing ratio <r>.
    level_repulsion_beta : float
        Fitted exponent beta from P(s) ~ s^beta for small s.
    histogram_edges : ndarray
        Bin edges for the spacing histogram.
    histogram_counts : ndarray
        Normalised counts (probability density) for each bin.
    n_spacings : int
        Total number of spacings collected.
    """

    all_spacings: np.ndarray
    mean_spacing_ratio: float
    level_repulsion_beta: float
    histogram_edges: np.ndarray
    histogram_counts: np.ndarray
    n_spacings: int


@dataclass
class RMTComparison:
    """Result of comparing ensemble spacings with RMT references.

    Attributes
    ----------
    ks_goe : float
        Kolmogorov-Smirnov statistic vs GOE Wigner surmise.
    ks_gue : float
        KS statistic vs GUE Wigner surmise.
    ks_poisson : float
        KS statistic vs Poisson.
    best_match : str
        "GOE", "GUE", or "Poisson" (smallest KS).
    ratio_distance_goe : float
        |<r>_ensemble - <r>_GOE|.
    ratio_distance_gue : float
        |<r>_ensemble - <r>_GUE|.
    ratio_distance_poisson : float
        |<r>_ensemble - <r>_Poisson|.
    ratio_best_match : str
        Best match by spacing ratio proximity.
    """

    ks_goe: float
    ks_gue: float
    ks_poisson: float
    best_match: str
    ratio_distance_goe: float
    ratio_distance_gue: float
    ratio_distance_poisson: float
    ratio_best_match: str


@dataclass
class EnsembleAnalysis:
    """Complete P6 ensemble analysis result.

    Attributes
    ----------
    config : EnsembleConfig
        Parameters used.
    samples : list of EnsembleSample
        Individual realisations (eigenvalues + spacings).
    spacing_stats : SpacingStats
        Aggregated spacing statistics.
    rmt_comparison : RMTComparison
        Comparison with GOE/GUE/Poisson.
    number_variance : ndarray or None
        Sigma^2(L) values at sampled L points.
    number_variance_L : ndarray or None
        L values for number variance.
    spectral_rigidity : ndarray or None
        Delta_3(L) values at sampled L points.
    spectral_rigidity_L : ndarray or None
        L values for spectral rigidity.
    """

    config: EnsembleConfig
    samples: list[EnsembleSample]
    spacing_stats: SpacingStats
    rmt_comparison: RMTComparison
    number_variance: np.ndarray | None = None
    number_variance_L: np.ndarray | None = None
    spectral_rigidity: np.ndarray | None = None
    spectral_rigidity_L: np.ndarray | None = None


# ---------------------------------------------------------------------------
# RMT reference distributions
# ---------------------------------------------------------------------------


def goe_wigner_surmise(s: np.ndarray | float) -> np.ndarray:
    r"""GOE Wigner surmise for nearest-neighbour spacing.

    .. math::

        P_{\mathrm{GOE}}(s) = \frac{\pi}{2}\, s\, \exp\!\left(-\frac{\pi s^2}{4}\right)

    Characteristic: linear level repulsion (beta=1).

    Parameters
    ----------
    s : array_like
        Normalised spacings (mean 1).

    Returns
    -------
    ndarray
        Probability density at each s.
    """
    s = np.asarray(s, dtype=float)
    return (math.pi / 2.0) * s * np.exp(-math.pi * s**2 / 4.0)


def gue_wigner_surmise(s: np.ndarray | float) -> np.ndarray:
    r"""GUE Wigner surmise for nearest-neighbour spacing.

    .. math::

        P_{\mathrm{GUE}}(s) = \frac{32}{\pi^2}\, s^2\, \exp\!\left(-\frac{4 s^2}{\pi}\right)

    Characteristic: quadratic level repulsion (beta=2).

    Parameters
    ----------
    s : array_like
        Normalised spacings (mean 1).

    Returns
    -------
    ndarray
        Probability density at each s.
    """
    s = np.asarray(s, dtype=float)
    return (32.0 / math.pi**2) * s**2 * np.exp(-4.0 * s**2 / math.pi)


def poisson_spacing_pdf(s: np.ndarray | float) -> np.ndarray:
    r"""Poisson spacing distribution for uncorrelated levels.

    .. math::

        P_{\mathrm{Poisson}}(s) = \exp(-s)

    No level repulsion (beta=0).

    Parameters
    ----------
    s : array_like
        Normalised spacings (mean 1).

    Returns
    -------
    ndarray
        Probability density at each s.
    """
    s = np.asarray(s, dtype=float)
    return np.exp(-s)


# ---------------------------------------------------------------------------
# Ensemble generation
# ---------------------------------------------------------------------------


def _build_er_graph_with_seed(
    k: int,
    edge_prob: float,
    rng: np.random.RandomState,
    weight_by_log_gap: bool,
) -> "nx.Graph":
    """Build a single Erdos-Renyi prime graph using the given RNG.

    We inline the generation rather than calling build_prime_random_graph
    to allow per-sample seed control from a master RNG stream.
    """
    import networkx as nx

    primes = _first_primes(k)
    G = nx.Graph()
    for idx, p in enumerate(primes):
        G.add_node(idx, label=p)

    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            if rng.random() < edge_prob:
                if weight_by_log_gap:
                    w = abs(np.log(float(primes[j])) - np.log(float(primes[i])))
                else:
                    w = 1.0
                G.add_edge(i, j, weight=float(w))

    # Ensure connectivity (TNFR U2: bounded dynamics require connected graph)
    if k >= 2 and not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for ci in range(1, len(components)):
            u = min(components[0])
            v = min(components[ci])
            if weight_by_log_gap:
                w = abs(np.log(float(primes[v])) - np.log(float(primes[u])))
            else:
                w = 1.0
            G.add_edge(u, v, weight=float(w))
            components[0] = components[0] | components[ci]

    return G


def generate_er_ensemble(
    k: int,
    n_samples: int = 100,
    *,
    edge_prob: float = 0.3,
    sigma: float = 0.5,
    seed: int = 42,
    weight_by_log_gap: bool = True,
) -> list[EnsembleSample]:
    r"""Generate an Erdos-Renyi random ensemble on k prime nodes.

    For each realisation, edges are included independently with
    probability ``edge_prob``.  The operator is H = L_random + V_sigma
    where L_random is the (random) graph Laplacian and V_sigma is the
    deterministic prime potential.

    TNFR physics basis: This models Dissonance (OZ) as random edge
    perturbation on the structural network.  Ensemble averaging reveals
    the statistical mechanics under controlled stochastic perturbation.

    Parameters
    ----------
    k : int
        Number of prime nodes (>= 4 for meaningful spacing statistics).
    n_samples : int
        Number of random realisations.
    edge_prob : float
        Edge inclusion probability (0 < p <= 1).
    sigma : float
        Structural parameter for V_sigma potential.
    seed : int
        Master random seed (Invariant #6: Reproducible Dynamics).
    weight_by_log_gap : bool
        Use log-prime-gap edge weights.

    Returns
    -------
    list of EnsembleSample
        Sorted eigenvalues and normalised spacings per realisation.
    """
    rng = np.random.RandomState(seed)
    samples: list[EnsembleSample] = []

    for _ in range(n_samples):
        G = _build_er_graph_with_seed(k, edge_prob, rng, weight_by_log_gap)
        H, _ = build_h_tnfr(G, sigma=sigma)
        evals = np.sort(np.linalg.eigh(H)[0])
        spacings = compute_eigenvalue_spacings(evals) if k >= 4 else np.array([])
        samples.append(EnsembleSample(eigenvalues=evals, spacings=spacings))

    return samples


def generate_wigner_ensemble(
    k: int,
    n_samples: int = 100,
    *,
    wigner_scale: float = 0.1,
    sigma: float = 0.5,
    seed: int = 42,
    weight_by_log_gap: bool = True,
) -> list[EnsembleSample]:
    r"""Generate a Wigner-perturbation ensemble on the prime path graph.

    Starts from the deterministic H_TNFR for the path graph and adds
    a scaled GOE matrix: H_sample = H_det + (scale/sqrt(k)) * W,
    where W is a k×k real symmetric matrix with i.i.d. N(0,1) entries
    (upper triangle) and diagonal N(0, 2).

    The 1/sqrt(k) scaling ensures that the perturbation has O(1) effect
    on eigenvalue spacing regardless of matrix size (standard RMT
    normalisation).

    TNFR physics basis: Wigner perturbation models infinitesimal
    Dissonance (OZ) applied uniformly across all couplings.  The
    wigner_scale parameter encodes the ΔNFR amplitude of the stochastic
    perturbation.

    Parameters
    ----------
    k : int
        Number of prime nodes.
    n_samples : int
        Number of random realisations.
    wigner_scale : float
        Amplitude of the GOE perturbation (before 1/sqrt(k) scaling).
    sigma : float
        Structural parameter for V_sigma.
    seed : int
        Master random seed (Invariant #6).
    weight_by_log_gap : bool
        Use log-prime-gap edge weights.

    Returns
    -------
    list of EnsembleSample
    """
    from .operator import build_prime_path_graph

    rng = np.random.RandomState(seed)

    # Build deterministic base operator
    G_det = build_prime_path_graph(k, weight_by_log_gap=weight_by_log_gap)
    H_det, _ = build_h_tnfr(G_det, sigma=sigma)

    scale = wigner_scale / math.sqrt(max(k, 1))
    samples: list[EnsembleSample] = []

    for _ in range(n_samples):
        # GOE matrix: symmetric with N(0,1) off-diagonal, N(0,2) diagonal
        W_upper = rng.randn(k, k)
        W = (W_upper + W_upper.T) / math.sqrt(2.0)
        # Diagonal has variance 2 in GOE convention
        np.fill_diagonal(W, rng.randn(k) * math.sqrt(2.0))

        H_sample = H_det + scale * W
        evals = np.sort(np.linalg.eigh(H_sample)[0])
        spacings = compute_eigenvalue_spacings(evals) if k >= 4 else np.array([])
        samples.append(EnsembleSample(eigenvalues=evals, spacings=spacings))

    return samples


# ---------------------------------------------------------------------------
# Spacing statistics
# ---------------------------------------------------------------------------


def compute_ensemble_spacings(samples: Sequence[EnsembleSample]) -> np.ndarray:
    """Pool normalised spacings from all samples in an ensemble.

    Parameters
    ----------
    samples : sequence of EnsembleSample
        Ensemble realisations.

    Returns
    -------
    ndarray
        Concatenated array of all normalised spacings.
    """
    all_sp = [s.spacings for s in samples if len(s.spacings) > 0]
    if not all_sp:
        return np.array([])
    return np.concatenate(all_sp)


def compute_spacing_ratio(spacings: np.ndarray) -> np.ndarray:
    r"""Compute consecutive spacing ratios r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1}).

    The spacing ratio statistic avoids the unfolding step entirely and
    is a robust RMT discriminator.

    Reference: Atas et al., PRL 110, 084101 (2013).

    Parameters
    ----------
    spacings : ndarray
        Normalised spacings (but ratio is unfolding-invariant).

    Returns
    -------
    ndarray
        Array of spacing ratios, length len(spacings) - 1.
    """
    if len(spacings) < 2:
        return np.array([])
    s = np.asarray(spacings, dtype=float)
    s_min = np.minimum(s[:-1], s[1:])
    s_max = np.maximum(s[:-1], s[1:])
    # Guard division by zero
    mask = s_max > 1e-15
    ratios = np.where(mask, s_min / s_max, 0.0)
    return ratios


def compute_mean_spacing_ratio(spacings: np.ndarray) -> float:
    r"""Compute ensemble-averaged spacing ratio <r>.

    Reference values (exact):
    - GOE:     <r> ~ 0.5307
    - GUE:     <r> ~ 0.5996
    - Poisson: <r> = 2 ln 2 - 1 ~ 0.3863

    Parameters
    ----------
    spacings : ndarray
        Normalised spacings from ensemble.

    Returns
    -------
    float
        Mean spacing ratio.  Returns 0.0 if insufficient data.
    """
    ratios = compute_spacing_ratio(spacings)
    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))


def compute_level_repulsion_exponent(
    spacings: np.ndarray,
    s_max: float = 0.5,
) -> float:
    r"""Estimate the level repulsion exponent beta from P(s) ~ s^beta.

    For small s, the spacing distribution behaves as P(s) ~ s^beta:
    - beta = 0: Poisson (no repulsion)
    - beta = 1: GOE (linear repulsion)
    - beta = 2: GUE (quadratic repulsion)

    We fit log P(s) = beta * log(s) + const using a histogram of
    spacings in [0, s_max].

    Parameters
    ----------
    spacings : ndarray
        Normalised spacings.
    s_max : float
        Upper cutoff for the small-s fit region.

    Returns
    -------
    float
        Estimated beta.  Returns -1.0 if insufficient data.
    """
    spacings = np.asarray(spacings, dtype=float)
    small = spacings[(spacings > 1e-6) & (spacings < s_max)]
    if len(small) < 10:
        return -1.0

    # Build histogram in the small-s region
    n_bins = max(10, min(50, len(small) // 5))
    counts, edges = np.histogram(small, bins=n_bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Keep bins with positive density
    mask = counts > 0
    if np.sum(mask) < 3:
        return -1.0

    log_s = np.log(centres[mask])
    log_p = np.log(counts[mask])

    # Linear fit: log_p = beta * log_s + const
    coeffs = np.polyfit(log_s, log_p, 1)
    beta = float(coeffs[0])
    return beta


# ---------------------------------------------------------------------------
# Long-range spectral statistics
# ---------------------------------------------------------------------------


def compute_number_variance(
    unfolded_evals: np.ndarray,
    L_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the number variance Sigma^2(L).

    Sigma^2(L) = <(N(E, E+L) - L)^2> averaged over the spectral window,
    where N(E, E+L) counts unfolded eigenvalues in [E, E+L].

    RMT predictions:
    - Poisson: Sigma^2(L) = L
    - GOE:     Sigma^2(L) ~ (2/pi^2)(ln(2 pi L) + gamma + 1 - pi^2/8) for large L
    - GUE:     Sigma^2(L) ~ (1/pi^2)(ln(2 pi L) + gamma + 1) for large L

    Parameters
    ----------
    unfolded_evals : ndarray
        Unfolded (unit mean spacing) sorted eigenvalues.
    L_values : ndarray, optional
        Window sizes to evaluate.  Defaults to linspace(0.1, 5.0, 50).

    Returns
    -------
    (L_values, sigma2)
        L values and corresponding Sigma^2(L).
    """
    if L_values is None:
        L_values = np.linspace(0.1, 5.0, 50)
    L_values = np.asarray(L_values, dtype=float)

    evals = np.sort(np.asarray(unfolded_evals, dtype=float))
    n = len(evals)
    if n < 4:
        return L_values, np.zeros_like(L_values)

    sigma2 = np.zeros(len(L_values))
    e_min = evals[0]
    e_max = evals[-1]

    for idx, L in enumerate(L_values):
        if L <= 0:
            continue
        # Slide window across the spectrum
        variances = []
        # Use starting points: each eigenvalue defines a window start
        for i in range(n):
            start = evals[i]
            end = start + L
            if end > e_max + L * 0.1:
                break
            count = np.searchsorted(evals, end, side="left") - i
            variances.append((count - L) ** 2)
        if variances:
            sigma2[idx] = float(np.mean(variances))

    return L_values, sigma2


def compute_spectral_rigidity(
    unfolded_evals: np.ndarray,
    L_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the Dyson-Mehta spectral rigidity Delta_3(L).

    Delta_3(L) measures the least-square deviation of the staircase
    function N(E) from the best-fit straight line over intervals of
    length L.

    RMT predictions:
    - Poisson: Delta_3(L) = L/15
    - GOE:     Delta_3(L) ~ (1/pi^2)(ln(2 pi L) + gamma - 5/4 - pi^2/8) for large L
    - GUE:     Delta_3(L) ~ (1/2 pi^2)(ln(2 pi L) + gamma - 5/4) for large L

    Parameters
    ----------
    unfolded_evals : ndarray
        Unfolded sorted eigenvalues.
    L_values : ndarray, optional
        Window sizes.  Defaults to linspace(0.5, 5.0, 40).

    Returns
    -------
    (L_values, delta3)
        L values and corresponding Delta_3(L).
    """
    if L_values is None:
        L_values = np.linspace(0.5, 5.0, 40)
    L_values = np.asarray(L_values, dtype=float)

    evals = np.sort(np.asarray(unfolded_evals, dtype=float))
    n = len(evals)
    if n < 4:
        return L_values, np.zeros_like(L_values)

    delta3 = np.zeros(len(L_values))
    e_max = evals[-1]

    for idx, L in enumerate(L_values):
        if L <= 0:
            continue

        d3_samples = []
        for i in range(n):
            start = evals[i]
            end = start + L
            if end > e_max + L * 0.1:
                break

            # Eigenvalues in [start, start+L]
            j_end = np.searchsorted(evals, end, side="right")
            window_evals = evals[i:j_end]
            if len(window_evals) < 2:
                continue

            # Staircase values: N(E) = rank within window
            x = window_evals - start  # shift to [0, L]
            y = np.arange(1, len(window_evals) + 1, dtype=float)

            # Best linear fit y = a*x + b, then compute residual
            n_w = len(x)
            sx = np.sum(x)
            sy = np.sum(y)
            sxx = np.sum(x * x)
            sxy = np.sum(x * y)
            denom = n_w * sxx - sx * sx
            if abs(denom) < 1e-30:
                continue
            a = (n_w * sxy - sx * sy) / denom
            b = (sy - a * sx) / n_w
            residual = np.sum((y - a * x - b) ** 2) / n_w
            d3_samples.append(residual)

        if d3_samples:
            delta3[idx] = float(np.mean(d3_samples))

    return L_values, delta3


# ---------------------------------------------------------------------------
# RMT comparison
# ---------------------------------------------------------------------------


def ks_test_vs_reference(
    spacings: np.ndarray,
    reference: str = "GOE",
) -> float:
    r"""Kolmogorov-Smirnov statistic for spacings vs an RMT reference CDF.

    Computes D = max |F_empirical(s) - F_reference(s)| where
    F_reference is the CDF of the chosen reference distribution.

    Parameters
    ----------
    spacings : ndarray
        Normalised spacings.
    reference : str
        "GOE", "GUE", or "Poisson".

    Returns
    -------
    float
        KS statistic D.  Smaller = better fit.
    """
    s_sorted = np.sort(np.asarray(spacings, dtype=float))
    n = len(s_sorted)
    if n == 0:
        return 1.0

    # Empirical CDF
    ecdf = np.arange(1, n + 1, dtype=float) / n

    # Reference CDF via numerical integration on the same grid
    ref_lower = reference.upper()
    if ref_lower == "GOE":
        pdf_fn = goe_wigner_surmise
    elif ref_lower == "GUE":
        pdf_fn = gue_wigner_surmise
    elif ref_lower == "POISSON":
        pdf_fn = poisson_spacing_pdf
    else:
        raise ValueError(f"Unknown reference: {reference!r}")

    # Compute reference CDF at observed spacings via trapezoidal integration
    # on a fine grid from 0 to max(s) + margin
    s_max = s_sorted[-1] + 0.5
    n_grid = max(1000, n * 5)
    grid = np.linspace(0, s_max, n_grid)
    pdf_grid = pdf_fn(grid)
    cdf_grid = np.zeros(n_grid)
    # Cumulative trapezoid
    ds = grid[1] - grid[0]
    for i in range(1, n_grid):
        cdf_grid[i] = cdf_grid[i - 1] + 0.5 * (pdf_grid[i - 1] + pdf_grid[i]) * ds

    # Interpolate reference CDF at observed spacings
    ref_cdf = np.interp(s_sorted, grid, cdf_grid)

    # KS statistic
    ks = float(np.max(np.abs(ecdf - ref_cdf)))
    return ks


def classify_ensemble(
    spacing_stats: SpacingStats,
) -> RMTComparison:
    """Compare ensemble spacing statistics with GOE, GUE, and Poisson.

    Uses both the Kolmogorov-Smirnov statistic on P(s) and the
    mean spacing ratio <r> for classification.

    Parameters
    ----------
    spacing_stats : SpacingStats
        Aggregated ensemble statistics.

    Returns
    -------
    RMTComparison
    """
    sp = spacing_stats.all_spacings

    ks_goe = ks_test_vs_reference(sp, "GOE")
    ks_gue = ks_test_vs_reference(sp, "GUE")
    ks_poisson = ks_test_vs_reference(sp, "Poisson")

    ks_vals = {"GOE": ks_goe, "GUE": ks_gue, "Poisson": ks_poisson}
    best_ks = min(ks_vals, key=ks_vals.get)  # type: ignore[arg-type]

    r_mean = spacing_stats.mean_spacing_ratio
    rd_goe = abs(r_mean - GOE_MEAN_RATIO)
    rd_gue = abs(r_mean - GUE_MEAN_RATIO)
    rd_poisson = abs(r_mean - POISSON_MEAN_RATIO)

    rd_vals = {"GOE": rd_goe, "GUE": rd_gue, "Poisson": rd_poisson}
    best_rd = min(rd_vals, key=rd_vals.get)  # type: ignore[arg-type]

    return RMTComparison(
        ks_goe=ks_goe,
        ks_gue=ks_gue,
        ks_poisson=ks_poisson,
        best_match=best_ks,
        ratio_distance_goe=rd_goe,
        ratio_distance_gue=rd_gue,
        ratio_distance_poisson=rd_poisson,
        ratio_best_match=best_rd,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _compute_spacing_stats(
    samples: Sequence[EnsembleSample],
    n_bins: int = 60,
) -> SpacingStats:
    """Compute aggregated spacing statistics from ensemble samples."""
    all_sp = compute_ensemble_spacings(samples)
    mean_r = compute_mean_spacing_ratio(all_sp)
    beta = compute_level_repulsion_exponent(all_sp)

    if len(all_sp) > 0:
        counts, edges = np.histogram(
            all_sp, bins=n_bins, range=(0.0, 4.0), density=True
        )
    else:
        edges = np.linspace(0.0, 4.0, n_bins + 1)
        counts = np.zeros(n_bins)

    return SpacingStats(
        all_spacings=all_sp,
        mean_spacing_ratio=mean_r,
        level_repulsion_beta=beta,
        histogram_edges=edges,
        histogram_counts=counts,
        n_spacings=len(all_sp),
    )


# ---------------------------------------------------------------------------
# Integration: run full RMT ensemble analysis
# ---------------------------------------------------------------------------


def run_rmt_ensemble_analysis(
    config: EnsembleConfig | None = None,
    *,
    k: int = 30,
    n_samples: int = 100,
    sigma: float = 0.5,
    ensemble_type: str = "erdos_renyi",
    edge_prob: float = 0.3,
    wigner_scale: float = 0.1,
    seed: int = 42,
    weight_by_log_gap: bool = True,
    compute_long_range: bool = True,
) -> EnsembleAnalysis:
    """Run a complete P6 RMT ensemble analysis.

    Parameters
    ----------
    config : EnsembleConfig, optional
        If provided, overrides the individual keyword arguments.
    k, n_samples, sigma, ensemble_type, edge_prob, wigner_scale, seed,
    weight_by_log_gap : various
        Same as EnsembleConfig fields.
    compute_long_range : bool
        If True, compute number variance and spectral rigidity
        (more expensive but gives long-range spectral correlations).

    Returns
    -------
    EnsembleAnalysis
        Full analysis with samples, statistics, and RMT comparison.
    """
    if config is not None:
        k = config.k
        n_samples = config.n_samples
        sigma = config.sigma
        ensemble_type = config.ensemble_type
        edge_prob = config.edge_prob
        wigner_scale = config.wigner_scale
        seed = config.seed
        weight_by_log_gap = config.weight_by_log_gap
    else:
        config = EnsembleConfig(
            k=k,
            n_samples=n_samples,
            sigma=sigma,
            ensemble_type=ensemble_type,
            edge_prob=edge_prob,
            wigner_scale=wigner_scale,
            seed=seed,
            weight_by_log_gap=weight_by_log_gap,
        )

    etype = ensemble_type.lower().replace("-", "_")
    if etype == "erdos_renyi":
        samples = generate_er_ensemble(
            k,
            n_samples,
            edge_prob=edge_prob,
            sigma=sigma,
            seed=seed,
            weight_by_log_gap=weight_by_log_gap,
        )
    elif etype == "wigner":
        samples = generate_wigner_ensemble(
            k,
            n_samples,
            wigner_scale=wigner_scale,
            sigma=sigma,
            seed=seed,
            weight_by_log_gap=weight_by_log_gap,
        )
    else:
        raise ValueError(
            f"Unknown ensemble_type: {ensemble_type!r}. "
            "Use 'erdos_renyi' or 'wigner'."
        )

    # Compute spacing statistics
    stats = _compute_spacing_stats(samples)

    # RMT comparison
    rmt = classify_ensemble(stats)

    # Long-range statistics (ensemble-averaged)
    nv_L: np.ndarray | None = None
    nv_vals: np.ndarray | None = None
    sr_L: np.ndarray | None = None
    sr_vals: np.ndarray | None = None

    if compute_long_range and len(samples) > 0:
        # Pool all unfolded eigenvalues
        all_evals = [s.eigenvalues for s in samples if len(s.eigenvalues) >= 4]
        if all_evals:
            # Unfold each spectrum, then pool
            all_unfolded = []
            for ev in all_evals:
                uf = _unfold_eigenvalues(ev)
                all_unfolded.append(uf)

            # Average number variance across samples
            L_vals = np.linspace(0.1, 5.0, 40)
            nv_accum = np.zeros(len(L_vals))
            sr_accum = np.zeros(len(L_vals))
            n_valid = 0

            for uf in all_unfolded:
                if len(uf) < 4:
                    continue
                _, nv = compute_number_variance(uf, L_vals)
                _, sr = compute_spectral_rigidity(uf, L_vals)
                nv_accum += nv
                sr_accum += sr
                n_valid += 1

            if n_valid > 0:
                nv_L = L_vals
                nv_vals = nv_accum / n_valid
                sr_L = L_vals
                sr_vals = sr_accum / n_valid

    return EnsembleAnalysis(
        config=config,
        samples=samples,
        spacing_stats=stats,
        rmt_comparison=rmt,
        number_variance=nv_vals,
        number_variance_L=nv_L,
        spectral_rigidity=sr_vals,
        spectral_rigidity_L=sr_L,
    )


def rmt_convergence_study(
    k_values: Sequence[int],
    n_samples: int = 100,
    *,
    sigma: float = 0.5,
    ensemble_type: str = "erdos_renyi",
    edge_prob: float = 0.3,
    wigner_scale: float = 0.1,
    seed: int = 42,
    weight_by_log_gap: bool = True,
) -> list[EnsembleAnalysis]:
    r"""Study how RMT statistics converge as k grows.

    For each k in ``k_values``, runs a full ensemble analysis and
    returns the list of results.  The seed is advanced per k to ensure
    independent but reproducible ensembles.

    TNFR physics basis: U2 (convergence/boundedness) predicts that
    ensemble statistics converge as system size grows.  This study
    verifies that prediction for the random-topology TNFR operator.

    Parameters
    ----------
    k_values : sequence of int
        Number of prime nodes at each scale.
    n_samples : int
        Realisations per k value.
    sigma, ensemble_type, edge_prob, wigner_scale, seed, weight_by_log_gap
        Passed through to :func:`run_rmt_ensemble_analysis`.

    Returns
    -------
    list of EnsembleAnalysis
        One analysis per k value.
    """
    results: list[EnsembleAnalysis] = []
    for i, k in enumerate(k_values):
        cfg = EnsembleConfig(
            k=k,
            n_samples=n_samples,
            sigma=sigma,
            ensemble_type=ensemble_type,
            edge_prob=edge_prob,
            wigner_scale=wigner_scale,
            seed=seed + i,  # advance seed per k for independence
            weight_by_log_gap=weight_by_log_gap,
        )
        analysis = run_rmt_ensemble_analysis(
            cfg,
            compute_long_range=False,
        )
        results.append(analysis)
    return results

r"""Predictive audit for non-normal ``DeltaNFR`` transients.

For the fixed directed pure-EPI channel

``x_dot = -L_rw x`` and ``p = DeltaNFR_epi = -L_rw x``, so

``p_dot = -L_rw p``.

On a strongly connected graph, ``range(L_rw) = {p : pi.T p = 0}``.  Every
non-consensus pressure is therefore reachable from an EPI state, and the
restricted semigroup norm

``max_t ||exp(-t L_sub)||_2``

is exactly the worst-case ``DeltaNFR`` amplification in the declared Euclidean
metric.  The logarithmic norm supplies an exact qualitative test:

``mu_2(-L_sub) = lambda_max((-L_sub - L_sub.T) / 2)``.

If ``mu_2 <= 0`` the semigroup is a contraction; if ``mu_2 > 0`` some pressure
direction grows immediately.  A negative spectral abscissa only establishes
asymptotic decay and cannot exclude that transient.

This module combines the exact finite-dimensional statement with an explicitly
finite, deterministic benchmark.  Rank correlations and sampled peak values are
measured evidence for that family only.  They are not a universal predictor of
nonlinear Dissonance, coherence loss, operator sequences, or U2/U6 compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..rng import validate_seed
from .directed_diffusion import directed_rw_laplacian
from ._helpers import finite_real_scalar
from .spectral_projectors import (
    commutator_norm,
    derived_tolerance,
    matrix_exponential,
    pseudospectral_bound,
    spectral_abscissa,
)
from .transient_u2 import restricted_generator

__all__ = [
    "NonnormalPredictorRecord",
    "NonnormalPredictionCertificate",
    "deterministic_directed_family",
    "measure_nonnormal_pressure_prediction",
    "benchmark_nonnormal_prediction",
]


@dataclass(frozen=True)
class NonnormalPredictorRecord:
    r"""Predictors and measured worst-case pressure gain for one graph.

    ``spectral_rule_predicts_burst`` is the ordinary asymptotic rule
    ``alpha(-L_sub) > 0`` evaluated with the numerical tolerance.
    ``lognorm_rule_predicts_burst`` represents the exact continuous-time
    predicate ``mu_2(-L_sub) > 0`` only when
    ``lognorm_numerical_sign_status`` is resolved.  The raw sign of the
    floating-point estimate is retained separately in ``lognorm_sign_estimate``;
    a value inside the backward-error-scaled tolerance is reported as
    unresolved and the Boolean prediction is withheld.
    """

    index: int
    node_count: int
    spectral_abscissa: float
    spectral_gap: float
    numerical_abscissa: float
    normality_residual: float
    kreiss_lower_bound: float
    peak_pressure_gain: float
    peak_time_structural: float
    measured_pressure_burst: bool
    spectral_rule_predicts_burst: bool
    lognorm_sign_estimate: str
    lognorm_numerical_sign_status: str
    lognorm_rule_predicts_burst: bool | None
    tolerance: float


@dataclass(frozen=True)
class NonnormalPredictionCertificate:
    """Finite-family comparison of asymptotic and transient predictors.

    Even-indexed records form a predeclared calibration partition and
    odd-indexed records form the holdout partition.  No coefficient or threshold
    is fitted: zero is fixed analytically for both abscissae, and the numerical
    tolerance follows matrix scale and machine precision.  Logarithmic-norm
    accuracies use only records whose sign is numerically resolved; they are
    ``NaN`` if the selected partition has no resolved sign.
    """

    family_size: int
    calibration_size: int
    holdout_size: int
    measured_burst_count: int
    resolved_lognorm_sign_count: int
    unresolved_lognorm_sign_count: int
    all_lognorm_signs_numerically_resolved: bool
    lognorm_verification_status: str
    all_spectrally_stable: bool
    pressure_semigroup_identity: str
    split_rule: str
    spectral_rule_accuracy: float
    lognorm_rule_accuracy: float
    spectral_rule_balanced_accuracy: float
    lognorm_rule_balanced_accuracy: float
    calibration_spectral_accuracy: float
    calibration_lognorm_accuracy: float
    holdout_spectral_accuracy: float
    holdout_lognorm_accuracy: float
    spearman_spectral_abscissa_vs_gain: float
    spearman_numerical_abscissa_vs_gain: float
    spearman_normality_residual_vs_gain: float
    spearman_kreiss_bound_vs_gain: float
    scan_consistent_with_lognorm_theorem: bool
    claim_status: str
    records: tuple[NonnormalPredictorRecord, ...]


def deterministic_directed_family(
    *,
    seed: int = 1_618_033,
    size: int = 16,
    nodes: int = 10,
    extra_edge_probability: float = 0.2,
    log10_weight_span: float = 6.0,
) -> tuple[np.ndarray, ...]:
    r"""Return a deterministic finite family of strongly connected digraphs.

    A positive directed cycle is installed first, so every graph is strongly
    connected.  Additional arcs and conductances are drawn from the declared
    seeded fixture distribution.  The defaults are benchmark design settings,
    not TNFR constants.  They deliberately include both contracting and
    transiently amplifying cases; no population-level sampling claim is made.
    """
    seed = validate_seed(seed, allow_none=False)
    if isinstance(size, (bool, np.bool_)) or not isinstance(
        size, (int, np.integer)
    ) or size < 1:
        raise ValueError("size must be a positive integer")
    if isinstance(nodes, (bool, np.bool_)) or not isinstance(
        nodes, (int, np.integer)
    ) or nodes < 2:
        raise ValueError("nodes must be an integer of at least two")
    try:
        edge_probability = finite_real_scalar(
            extra_edge_probability, "extra_edge_probability"
        )
    except ValueError as exc:
        raise ValueError("extra_edge_probability must lie in [0, 1]") from exc
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("extra_edge_probability must lie in [0, 1]")
    try:
        weight_span = finite_real_scalar(log10_weight_span, "log10_weight_span")
    except ValueError as exc:
        raise ValueError(
            "log10_weight_span must be finite and nonnegative"
        ) from exc
    if weight_span < 0.0:
        raise ValueError("log10_weight_span must be finite and nonnegative")

    maximum_half_span = float(
        np.log10(np.finfo(float).max / float(nodes))
    )
    if weight_span / 2.0 > maximum_half_span:
        raise ValueError(
            "log10_weight_span is too large to keep every row conductance finite"
        )

    rng = np.random.default_rng(seed)
    half_span = weight_span / 2.0
    family: list[np.ndarray] = []
    for _ in range(int(size)):
        weights = np.zeros((int(nodes), int(nodes)), dtype=float)
        for source in range(int(nodes)):
            weights[source, (source + 1) % int(nodes)] = 10.0 ** rng.uniform(
                -half_span, half_span
            )
        for source in range(int(nodes)):
            for target in range(int(nodes)):
                if source == target:
                    continue
                if rng.random() < edge_probability:
                    weights[source, target] = 10.0 ** rng.uniform(
                        -half_span, half_span
                    )
        family.append(weights)
    return tuple(family)


def _peak_pressure_gain(
    generator: np.ndarray, *, t_max: float, samples: int
) -> tuple[float, float]:
    """Sample ``max ||exp(t generator)||_2`` on a declared finite window."""
    best_gain = 1.0
    best_time = 0.0
    for time in np.linspace(0.0, t_max, samples):
        gain = float(np.linalg.norm(matrix_exponential(generator * time), 2))
        if gain > best_gain:
            best_gain = gain
            best_time = float(time)
    return best_gain, best_time


def _is_strongly_connected_support(weights: np.ndarray) -> bool:
    """Check strong connectivity from positive-conductance support only."""
    support = np.asarray(weights, dtype=float) > 0.0
    node_count = len(support)
    if node_count == 0:
        return False

    def reaches_every_node(arcs: np.ndarray) -> bool:
        reached = {0}
        frontier = [0]
        while frontier:
            source = frontier.pop()
            for target in np.flatnonzero(arcs[source]):
                candidate = int(target)
                if candidate not in reached:
                    reached.add(candidate)
                    frontier.append(candidate)
        return len(reached) == node_count

    return reaches_every_node(support) and reaches_every_node(support.T)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with exact ties, sufficient for deterministic telemetry."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _spearman(first: Iterable[float], second: Iterable[float]) -> float:
    """Spearman rank correlation, or NaN when either ranking is constant."""
    x = _average_ranks(np.asarray(tuple(first), dtype=float))
    y = _average_ranks(np.asarray(tuple(second), dtype=float))
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _accuracy(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(observed == predicted))


def _balanced_accuracy(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean class recall; NaN when the finite sample contains one class only."""
    positive = observed
    negative = ~observed
    if not np.any(positive) or not np.any(negative):
        return float("nan")
    sensitivity = np.mean(predicted[positive])
    specificity = np.mean(~predicted[negative])
    return float((sensitivity + specificity) / 2.0)


def _accuracy_on_mask(
    observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> float:
    """Accuracy on numerically resolved predictions, or NaN if none exist."""
    if not np.any(mask):
        return float("nan")
    return _accuracy(observed[mask], predicted[mask])


def _balanced_accuracy_on_mask(
    observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> float:
    """Balanced accuracy on resolved predictions only."""
    if not np.any(mask):
        return float("nan")
    return _balanced_accuracy(observed[mask], predicted[mask])


def _resolve_lognorm_sign(
    estimate: float, tolerance: float
) -> tuple[str, str, bool | None]:
    """Separate the raw ``mu_2`` sign from its numerical certification."""
    sign_estimate = (
        "positive" if estimate > 0.0 else "negative" if estimate < 0.0 else "zero"
    )
    if estimate > tolerance:
        return sign_estimate, "resolved_positive", True
    if estimate < -tolerance:
        return sign_estimate, "resolved_nonpositive", False
    return sign_estimate, "unresolved_near_zero", None


def measure_nonnormal_pressure_prediction(
    adjacency,
    *,
    index: int = 0,
    t_max: float = 30.0,
    samples: int = 151,
    resolvent_grid: int = 12,
) -> NonnormalPredictorRecord:
    r"""Measure one graph and return its exact-sign and finite-scan readings.

    The graph must have one consensus mode: equivalently, ``L_sub`` must be
    invertible at the matrix-derived numerical tolerance.  This condition makes
    every non-consensus pressure reachable and is satisfied by a finite strongly
    connected directed graph with positive conductance on its arcs.
    """
    if isinstance(index, (bool, np.bool_)) or not isinstance(
        index, (int, np.integer)
    ):
        raise ValueError("index must be an integer")
    try:
        horizon = finite_real_scalar(t_max, "t_max")
    except ValueError as exc:
        raise ValueError("t_max must be finite and positive") from exc
    if horizon <= 0.0:
        raise ValueError("t_max must be finite and positive")
    if isinstance(samples, (bool, np.bool_)) or not isinstance(
        samples, (int, np.integer)
    ):
        raise ValueError("samples must be an integer")
    if samples < 2:
        raise ValueError("samples must be at least two")
    if (
        isinstance(resolvent_grid, (bool, np.bool_))
        or not isinstance(resolvent_grid, (int, np.integer))
        or resolvent_grid < 2
    ):
        raise ValueError("resolvent_grid must be an integer of at least two")

    raw_weights = np.asarray(adjacency, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in raw_weights.flat):
        raise ValueError("adjacency must contain real conductance, not booleans")
    weights = np.asarray(adjacency, dtype=float)
    directed_rw_laplacian(weights)
    if not _is_strongly_connected_support(weights):
        raise ValueError("the graph must have exactly one consensus mode")
    restricted = restricted_generator(weights)
    if restricted.shape[0] == 0:
        raise ValueError("the graph must contain at least two supported nodes")
    tolerance = derived_tolerance(restricted)
    if np.linalg.matrix_rank(restricted, tol=tolerance) != restricted.shape[0]:
        raise ValueError(
            "the non-consensus gap is numerically unresolved at the derived "
            "tolerance; strong connectivity still gives one structural "
            "consensus mode"
        )

    generator = -restricted
    generator_norm = float(np.linalg.norm(generator, 2))
    if horizon > np.finfo(float).max / max(1.0, generator_norm):
        raise ValueError("t_max is too large for a finite scaled generator")
    alpha = spectral_abscissa(generator)
    numerical_alpha = float(
        np.max(np.linalg.eigvalsh((generator + generator.T) / 2.0))
    )
    (
        lognorm_sign_estimate,
        lognorm_numerical_sign_status,
        lognorm_prediction,
    ) = _resolve_lognorm_sign(numerical_alpha, tolerance)
    peak_gain, peak_time = _peak_pressure_gain(
        generator, t_max=horizon, samples=int(samples)
    )
    return NonnormalPredictorRecord(
        index=int(index),
        node_count=restricted.shape[0] + 1,
        spectral_abscissa=alpha,
        spectral_gap=-alpha,
        numerical_abscissa=numerical_alpha,
        normality_residual=commutator_norm(restricted),
        kreiss_lower_bound=pseudospectral_bound(
            generator, grid=int(resolvent_grid)
        ),
        peak_pressure_gain=peak_gain,
        peak_time_structural=peak_time,
        measured_pressure_burst=peak_gain > 1.0 + tolerance,
        spectral_rule_predicts_burst=alpha > tolerance,
        lognorm_sign_estimate=lognorm_sign_estimate,
        lognorm_numerical_sign_status=lognorm_numerical_sign_status,
        lognorm_rule_predicts_burst=lognorm_prediction,
        tolerance=tolerance,
    )


def benchmark_nonnormal_prediction(
    adjacency_family,
    *,
    t_max: float = 30.0,
    samples: int = 151,
    resolvent_grid: int = 12,
) -> NonnormalPredictionCertificate:
    r"""Compare spectral and non-normal predictors on a finite graph family.

    The target is the sampled worst-case pressure amplification
    ``max_t ||exp(-t L_sub)||_2``.  This is an actual ``DeltaNFR`` gain because
    pressure obeys the same restricted equation and every restricted pressure
    is reachable.  ``t_max`` and ``samples`` limit only the measured peak; the
    logarithmic-norm sign criterion itself is an exact matrix theorem.  Its
    floating-point evaluation can still be unresolved near zero; those records
    receive no Boolean theorem prediction and are excluded from accuracies.

    Reported rank correlations and accuracies describe only the supplied
    family.  In particular, they do not establish prediction for nonlinear
    canonical operators or a universal directed-graph U2 rule.
    """
    family = tuple(adjacency_family)
    if len(family) < 4:
        raise ValueError(
            "adjacency_family must contain at least four graphs for the "
            "predeclared calibration/holdout split"
        )

    records: list[NonnormalPredictorRecord] = []
    for index, adjacency in enumerate(family):
        records.append(
            measure_nonnormal_pressure_prediction(
                adjacency,
                index=index,
                t_max=t_max,
                samples=samples,
                resolvent_grid=resolvent_grid,
            )
        )

    observed = np.asarray(
        [record.measured_pressure_burst for record in records], dtype=bool
    )
    spectral_prediction = np.asarray(
        [record.spectral_rule_predicts_burst for record in records], dtype=bool
    )
    resolved_lognorm = np.asarray(
        [record.lognorm_rule_predicts_burst is not None for record in records],
        dtype=bool,
    )
    lognorm_prediction = np.asarray(
        [
            False
            if record.lognorm_rule_predicts_burst is None
            else record.lognorm_rule_predicts_burst
            for record in records
        ],
        dtype=bool,
    )
    calibration = np.arange(len(records)) % 2 == 0
    holdout = ~calibration
    gains = [record.peak_pressure_gain for record in records]

    resolved_count = int(np.sum(resolved_lognorm))
    unresolved_count = len(records) - resolved_count
    all_lognorm_resolved = unresolved_count == 0
    scan_consistent = bool(
        all_lognorm_resolved and np.array_equal(observed, lognorm_prediction)
    )
    if not all_lognorm_resolved:
        verification_status = (
            "unresolved_numerical_signs_no_finite_family_verification"
        )
        numerical_scope = (
            f"NUMERICALLY UNRESOLVED for {unresolved_count} supplied record(s); "
            "no exact finite-family verification is claimed. "
        )
    elif scan_consistent:
        verification_status = "all_signs_resolved_and_scan_consistent"
        numerical_scope = "NUMERICALLY RESOLVED on the supplied family. "
    else:
        verification_status = "all_signs_resolved_but_finite_scan_disagrees"
        numerical_scope = (
            "NUMERICALLY RESOLVED, but the finite scan disagrees and does not "
            "verify the theorem on the supplied family. "
        )
    claim_status = (
        "EXACT theorem for fixed linear pure-EPI pressure in the declared "
        "Euclidean non-consensus metric: the mathematical sign of mu_2 "
        "characterizes possible transient gain. "
        + numerical_scope
        + "MEASURED for finite-window peaks, rank correlations, and the "
        "supplied finite graph family. OPEN for nonlinear Dissonance, C(t), "
        "heterogeneous time-dependent capacity, changing topology, and a "
        "canonical directed U2 metric."
    )
    return NonnormalPredictionCertificate(
        family_size=len(records),
        calibration_size=int(np.sum(calibration)),
        holdout_size=int(np.sum(holdout)),
        measured_burst_count=int(np.sum(observed)),
        resolved_lognorm_sign_count=resolved_count,
        unresolved_lognorm_sign_count=unresolved_count,
        all_lognorm_signs_numerically_resolved=all_lognorm_resolved,
        lognorm_verification_status=verification_status,
        all_spectrally_stable=bool(
            all(record.spectral_abscissa <= record.tolerance for record in records)
        ),
        pressure_semigroup_identity=(
            "p=-L_rw x implies p_dot=-L_rw p; on range(L_rw), sampled "
            "semigroup gain is worst-case DeltaNFR amplification"
        ),
        split_rule=(
            "even indices calibration; odd indices holdout; analytic zero "
            "thresholds; no fitted coefficients"
        ),
        spectral_rule_accuracy=_accuracy(observed, spectral_prediction),
        lognorm_rule_accuracy=_accuracy_on_mask(
            observed, lognorm_prediction, resolved_lognorm
        ),
        spectral_rule_balanced_accuracy=_balanced_accuracy(
            observed, spectral_prediction
        ),
        lognorm_rule_balanced_accuracy=_balanced_accuracy_on_mask(
            observed, lognorm_prediction, resolved_lognorm
        ),
        calibration_spectral_accuracy=_accuracy(
            observed[calibration], spectral_prediction[calibration]
        ),
        calibration_lognorm_accuracy=_accuracy_on_mask(
            observed,
            lognorm_prediction,
            calibration & resolved_lognorm,
        ),
        holdout_spectral_accuracy=_accuracy(
            observed[holdout], spectral_prediction[holdout]
        ),
        holdout_lognorm_accuracy=_accuracy_on_mask(
            observed,
            lognorm_prediction,
            holdout & resolved_lognorm,
        ),
        spearman_spectral_abscissa_vs_gain=_spearman(
            (record.spectral_abscissa for record in records), gains
        ),
        spearman_numerical_abscissa_vs_gain=_spearman(
            (record.numerical_abscissa for record in records), gains
        ),
        spearman_normality_residual_vs_gain=_spearman(
            (record.normality_residual for record in records), gains
        ),
        spearman_kreiss_bound_vs_gain=_spearman(
            (record.kreiss_lower_bound for record in records), gains
        ),
        scan_consistent_with_lognorm_theorem=scan_consistent,
        claim_status=claim_status,
        records=tuple(records),
    )
